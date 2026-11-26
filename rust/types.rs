//! Shared vocabulary of every UForm verb: statuses, scalars, tapes, rows, executors and arenas.
//!
//! An [`On`] is the receiver every verb runs through: it refers to a model or a trainer and an
//! executor, which both outlive it, and its completion decides whether a verb answers what the
//! call produced or a [`Deferred`] once the call is submitted.
//!
//! File: rust/types.rs
//! Author: Ash Vardanian
//! Date: October 18, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types, non_upper_case_globals)]

use core::borrow::BorrowMut;
use core::cell::UnsafeCell;
use core::ffi::{CStr, c_char, c_int, c_void};
use core::fmt;
use core::future::Future;
use core::marker::{PhantomData, PhantomPinned};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::ControlFlow;
use core::pin::{Pin, pin};
use core::ptr;
use core::str::FromStr;
use core::task::{Context, Poll};
use std::path::{Path, PathBuf};

// region: Raw Declarations

/// A run of bytes the caller owns, never null-terminated by contract.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_bytes_view_t {
    pub data: *const u8,
    pub length: usize,
}

impl From<&[u8]> for uform_bytes_view_t {
    /// Points the view the C interface accepts at the bytes; it must not outlive them.
    fn from(bytes: &[u8]) -> Self {
        uform_bytes_view_t { data: bytes.as_ptr(), length: bytes.len() }
    }
}

impl Default for uform_bytes_view_t {
    fn default() -> Self {
        uform_bytes_view_t { data: ptr::null(), length: 0 }
    }
}

/// The C enums lower to the platform `int`.
pub type uform_status_t = c_int;

pub const uform_status_success_k: uform_status_t = 0;
pub const uform_status_deferred_k: uform_status_t = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_outcome_t {
    pub status: uform_status_t,
    pub message: *const c_char,
}

impl Default for uform_outcome_t {
    fn default() -> Self {
        uform_outcome_t { status: uform_status_success_k, message: ptr::null() }
    }
}

pub type uform_scalar_t = c_int;

/// A run of rows along a call's batch axis, written packed into an arena.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_rows_view_t {
    pub data: *const c_void,
    pub stride: usize,
    pub columns: usize,
    pub scalar: uform_scalar_t,
}

impl Default for uform_rows_view_t {
    fn default() -> Self {
        uform_rows_view_t { data: ptr::null(), stride: 0, columns: 0, scalar: 0 }
    }
}

/// A ragged batch the caller owns: elements end to end, and one more offset than entries.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_tape_view_t {
    pub data: *const c_void,
    pub offsets: *const usize,
    pub scalar: uform_scalar_t,
}

impl Default for uform_tape_view_t {
    fn default() -> Self {
        uform_tape_view_t { data: ptr::null(), offsets: ptr::null(), scalar: 0 }
    }
}

/// A ragged run of rows a call wrote into its arena, and one more offset than items.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_rows_tape_view_t {
    pub rows: uform_rows_view_t,
    pub offsets: *const usize,
}

impl Default for uform_rows_tape_view_t {
    fn default() -> Self {
        uform_rows_tape_view_t { rows: uform_rows_view_t::default(), offsets: ptr::null() }
    }
}

/// Periodic advance report and the caller's chance to stop the call.
pub type uform_progress_callback_t = Option<unsafe extern "C" fn(usize, usize, *mut c_void) -> c_int>;

/// Mirrors `uform_progress_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_progress_t {
    pub callback: uform_progress_callback_t,
    pub state: *mut c_void,
    pub steps: usize,
}

pub type uform_completion_t = c_int;

pub const uform_completion_blocking_k: uform_completion_t = 0;
pub const uform_completion_deferred_k: uform_completion_t = 1;

pub type uform_cpu_class_t = c_int;
pub type uform_gpu_class_t = c_int;
pub type uform_gpu_vendor_t = c_int;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_gpu_device_t {
    pub vendor: uform_gpu_vendor_t,
    pub vendor_ordinal: u32,
    pub gpu_class: uform_gpu_class_t,
    pub compute_units: u32,
    pub memory_bytes: u64,
    pub name: [c_char; 64],
}

#[repr(C)]
pub struct uform_gpu_devices_t {
    pub outcome: *mut uform_outcome_t,
    pub devices: *mut uform_gpu_device_t,
    pub devices_capacity: usize,
    pub devices_count: *mut usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_ram_options_t {
    pub memory_domains: *const i32,
    pub memory_domains_count: usize,
    pub cpu_class_ceiling: uform_cpu_class_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_vram_options_t {
    pub devices: *const i32,
    pub devices_count: usize,
    pub gpu_class_ceiling: uform_gpu_class_t,
}

pub type uform_executor_t = *mut c_void;
pub type uform_arena_t = *mut c_void;

#[repr(C)]
pub struct uform_executor_for_ram_t {
    pub executor: *mut uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub ram: uform_ram_options_t,
    pub threads: usize,
}

#[repr(C)]
pub struct uform_executor_for_vram_t {
    pub executor: *mut uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub vram: uform_vram_options_t,
    pub ram: uform_ram_options_t,
    pub streams: *const *mut c_void,
    pub scratch_limit: usize,
}

pub type uform_mood_t = c_int;

#[repr(C)]
pub struct uform_executor_drain_t {
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub mood: uform_mood_t,
}

#[repr(C)]
pub struct uform_arena_init_t {
    pub arena: *mut uform_arena_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
}

pub type uform_residency_t = c_int;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct uform_arena_description_t {
    pub residency: uform_residency_t,
    pub device: i32,
}

#[repr(C)]
pub struct uform_arena_describe_t {
    pub arena: uform_arena_t,
    pub outcome: *mut uform_outcome_t,
    pub description: *mut uform_arena_description_t,
}

unsafe extern "C" {
    pub fn uform_outcome_wait(outcome: *mut uform_outcome_t) -> uform_status_t;
    pub fn uform_outcome_status(outcome: *mut uform_outcome_t) -> uform_status_t;

    pub fn uform_executor_for_ram(arguments: *mut uform_executor_for_ram_t);
    pub fn uform_executor_for_vram(arguments: *mut uform_executor_for_vram_t);
    pub fn uform_executor_drain(arguments: *mut uform_executor_drain_t);
    pub fn uform_executor_free(executor: uform_executor_t);

    pub fn uform_arena_init(arguments: *mut uform_arena_init_t);
    pub fn uform_arena_free(arena: uform_arena_t);
    pub fn uform_arena_describe(arguments: *mut uform_arena_describe_t);

    pub fn uform_status_name(status: uform_status_t) -> *const c_char;
    pub fn uform_status_from_name(name: uform_bytes_view_t, status: *mut uform_status_t) -> uform_status_t;
    pub fn uform_scalar_name(scalar: uform_scalar_t) -> *const c_char;
    pub fn uform_scalar_from_name(name: uform_bytes_view_t, scalar: *mut uform_scalar_t) -> uform_status_t;
    pub fn uform_mood_name(mood: uform_mood_t) -> *const c_char;
    pub fn uform_mood_from_name(name: uform_bytes_view_t, mood: *mut uform_mood_t) -> uform_status_t;
    pub fn uform_cpu_class_name(cpu_class: uform_cpu_class_t) -> *const c_char;
    pub fn uform_cpu_class_from_name(name: uform_bytes_view_t, cpu_class: *mut uform_cpu_class_t) -> uform_status_t;
    pub fn uform_gpu_class_name(gpu_class: uform_gpu_class_t) -> *const c_char;
    pub fn uform_gpu_class_from_name(name: uform_bytes_view_t, gpu_class: *mut uform_gpu_class_t) -> uform_status_t;
    pub fn uform_gpu_vendor_name(vendor: uform_gpu_vendor_t) -> *const c_char;
    pub fn uform_gpu_vendor_from_name(name: uform_bytes_view_t, vendor: *mut uform_gpu_vendor_t) -> uform_status_t;

    pub fn uform_gpu_devices(arguments: *mut uform_gpu_devices_t);

    pub fn uform_version_name() -> *const c_char;
    pub fn uform_capabilities_compiled() -> *const c_char;
    pub fn uform_capabilities_available() -> *const c_char;
}

// endregion: Raw Declarations

// region: Sealed Traits

pub(crate) use crate::sealed::{Accepts, Batch, Calling, Completion, Gathered, Handles, TapeView};

/// A scalar type a tape holds or a run of rows reads as, tied to the [`Scalar`] the C interface
/// tags its buffers with.
///
/// Rust-only.
///
/// # Safety
///
/// The associated scalar must describe the type's in-memory encoding exactly; the library
/// reads raw buffers under that tag.
pub unsafe trait Element: Copy + 'static {
    /// The tag the C interface knows this type by.
    const SCALAR: Scalar;
}

unsafe impl Element for f64 {
    const SCALAR: Scalar = Scalar::F64;
}

unsafe impl Element for f32 {
    const SCALAR: Scalar = Scalar::F32;
}

unsafe impl Element for i8 {
    const SCALAR: Scalar = Scalar::I8;
}

unsafe impl Element for i32 {
    const SCALAR: Scalar = Scalar::I32;
}

unsafe impl Element for u8 {
    const SCALAR: Scalar = Scalar::U8;
}

impl<Value: Element, Item: AsRef<[Value]>> TapeView<Value> for [Item] {
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> {
        // A deferred call asks twice, at creation and at submission, and gathers once.
        if gathered.offsets.len() != self.len() + 1 {
            gathered.data.clear();
            gathered.offsets.clear();
            gathered.offsets.reserve(self.len() + 1);
            gathered.offsets.push(0);
            for item in self {
                gathered.data.extend_from_slice(item.as_ref());
                gathered.offsets.push(gathered.data.len());
            }
        }
        Ok(Batch {
            tape: uform_tape_view_t {
                data: gathered.data.as_ptr().cast(),
                offsets: gathered.offsets.as_ptr(),
                scalar: Value::SCALAR as uform_scalar_t,
            },
            count: self.len(),
        })
    }
}

impl<Value: Element, Item: AsRef<[Value]>> TapeView<Value> for Vec<Item> {
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> {
        self.as_slice().tape(gathered)
    }
}

impl<Value: Element, Item: AsRef<[Value]>, const COUNT: usize> TapeView<Value> for [Item; COUNT] {
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> {
        self.as_slice().tape(gathered)
    }
}

impl<Value: Element, Data: AsRef<[Value]>, Offsets: AsRef<[usize]>> TapeView<Value> for (Data, Offsets) {
    fn tape(&self, _: &mut Gathered<Value>) -> Result<Batch> {
        let (data, offsets) = (self.0.as_ref(), self.1.as_ref());
        let Some((&last, _)) = offsets.split_last() else {
            return Err(fail(Status::InvalidArgument, "a tape holds one more offset than items"));
        };
        if offsets.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(fail(Status::Malformed, "the tape's offsets must not decrease"));
        }
        if last > data.len() {
            return Err(fail(Status::ShapeMismatch, "the tape's last offset runs past its data"));
        }
        Ok(Batch {
            tape: uform_tape_view_t {
                data: data.as_ptr().cast(),
                offsets: offsets.as_ptr(),
                scalar: Value::SCALAR as uform_scalar_t,
            },
            count: offsets.len() - 1,
        })
    }
}

impl<Value: Element, Tape: TapeView<Value> + ?Sized> TapeView<Value> for &Tape {
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> {
        (**self).tape(gathered)
    }
}

// endregion: Sealed Traits

// region: Errors and Introspection

/// Every value the C status word takes besides success, numbered alike.
///
/// An [`Error`] never carries [`Status::Deferred`], which names a call not yet final.
///
/// Mirrors `uform_status_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The call was submitted deferred and is not final yet.
    Deferred = 1,
    /// Device, pinned, or host memory could not be allocated, or the executor's scratch ceiling
    /// is too low.
    OutOfMemory = 2,
    /// Item counts, a pixel shape, or a batch envelope disagree with the call or the model.
    ShapeMismatch = 3,
    /// A stride or packing the model cannot read.
    UnsupportedLayout = 4,
    /// A scalar, modality, or name this build does not know.
    UnknownKind = 5,
    /// A checkpoint file could not be opened, read, or written.
    Io = 7,
    /// The driver refused or the device faulted; after a fault the executor refuses
    /// every later call.
    Device = 8,
    /// Supplied bytes could not be decoded: a checkpoint, an image, a clip, or a recording.
    Malformed = 9,
    /// The library reported neither success nor a failure.
    Internal = 10,
    /// The handle the call operates on is missing, refused before any other check.
    Uninitialized = 12,
    /// The request contradicts a frozen configuration, like resuming a checkpoint
    /// of another geometry.
    ConfigMismatch = 13,
    /// The call was refused rather than waited for: another call holds its executor or its arena.
    Contended = 14,
    /// A progress handler stopped the call; it unwound and published nothing.
    Cancelled = 15,
    /// The verb is declared for every residency or model, but this one does not implement it.
    Unsupported = 16,
    /// An argument breaks the verb's contract.
    InvalidArgument = 17,
}

impl Status {
    /// Lifts the C status word, reading a value this binding has no name for as `Internal`.
    pub(crate) fn from_raw(value: uform_status_t) -> core::result::Result<(), Self> {
        let status = match value {
            uform_status_success_k => return Ok(()),
            uform_status_deferred_k => Status::Deferred,
            2 => Status::OutOfMemory,
            3 => Status::ShapeMismatch,
            4 => Status::UnsupportedLayout,
            5 => Status::UnknownKind,
            7 => Status::Io,
            8 => Status::Device,
            9 => Status::Malformed,
            12 => Status::Uninitialized,
            13 => Status::ConfigMismatch,
            14 => Status::Contended,
            15 => Status::Cancelled,
            16 => Status::Unsupported,
            17 => Status::InvalidArgument,
            _ => Status::Internal,
        };
        Err(status)
    }
}

impl fmt::Display for Status {
    /// Calls `uform_status_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_status_name, *self as c_int))
    }
}

impl FromStr for Status {
    type Err = Error;

    /// Calls `uform_status_from_name`, refusing "success", which names no failure.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, uform_status_from_name, |value| Status::from_raw(value).err())
    }
}

/// A failure, whether the library reported it or the binding caught it before the call.
///
/// Messages are static storage, so nothing allocates on the failure path.
///
/// Mirrors a failed `uform_outcome_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct Error {
    /// The identity of the failure, shared with the C interface.
    pub status: Status,
    /// The detail the status alone cannot express.
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for Error {}

/// The result of every fallible call in this crate.
pub type Result<Value> = core::result::Result<Value, Error>;

/// Turns a final outcome into a `Result`.
pub(crate) fn check(outcome: uform_outcome_t) -> Result<()> {
    match Status::from_raw(outcome.status) {
        Ok(()) => Ok(()),
        Err(Status::Deferred) => Err(fail(Status::Internal, "the outcome was read before it was final")),
        Err(status) => Err(Error { status, message: unsafe { static_string(outcome.message) } }),
    }
}

/// Reports a failure the binding caught before the call crossed the boundary, under the same
/// status the library would have used had the ABI let it see the same thing.
pub(crate) fn fail(status: Status, message: &'static str) -> Error {
    Error { status, message }
}

/// Reads a static null-terminated string the library owns.
///
/// Every such string is ASCII by construction, so its bytes are read as UTF-8 unchecked.
pub(crate) unsafe fn static_string(raw: *const c_char) -> &'static str {
    unsafe {
        if raw.is_null() {
            return "";
        }
        core::str::from_utf8_unchecked(CStr::from_ptr(raw).to_bytes())
    }
}

/// Spells an enumerator through its C `uform_*_name`.
pub(crate) fn spelling(name: unsafe extern "C" fn(c_int) -> *const c_char, value: c_int) -> &'static str {
    unsafe { static_string(name(value)) }
}

/// Reads a spelling back through its C `uform_*_from_name`, passing on the status it fails with.
pub(crate) fn read_spelling<Value>(
    text: &str,
    from_name: unsafe extern "C" fn(uform_bytes_view_t, *mut c_int) -> uform_status_t,
    lift: fn(c_int) -> Option<Value>,
) -> Result<Value> {
    let mut value: c_int = 0;
    let status = unsafe { from_name(text.as_bytes().into(), &mut value) };
    Status::from_raw(status).map_err(|status| fail(status, "no enumerator is spelled that way"))?;
    lift(value).ok_or(fail(Status::InvalidArgument, "no enumerator is spelled that way"))
}

/// The version of the linked library.
///
/// Calls `uform_version_name`.
pub fn version() -> &'static str {
    unsafe { static_string(uform_version_name()) }
}

/// Comma-separated hardware capabilities the library was compiled with.
///
/// Calls `uform_capabilities_compiled`.
pub fn capabilities_compiled() -> &'static str {
    unsafe { static_string(uform_capabilities_compiled()) }
}

/// Comma-separated hardware capabilities this machine offers at runtime.
///
/// Calls `uform_capabilities_available`.
pub fn capabilities_available() -> &'static str {
    unsafe { static_string(uform_capabilities_available()) }
}

// endregion: Errors and Introspection

// region: Scalars

/// Scalar types a model computes in and its outputs land as, numbered as USearch numbers them, so
/// a run of embeddings is a run USearch indexes without conversion.
///
/// Sub-byte kinds pack several dimensions per byte, as their `xN` suffixes spell. The C enum
/// reserves the numbering gaps for kinds that have not landed yet, so this one is open.
///
/// Mirrors `uform_scalar_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Scalar {
    /// IEEE-754 binary64.
    F64 = 1,
    /// IEEE-754 binary32.
    F32 = 2,
    /// Truncated f32 with an 8-bit exponent and a 7-bit mantissa.
    BF16 = 3,
    /// IEEE-754 binary16.
    F16 = 4,
    /// FP8 with a 5-bit exponent: ±57344, with infinities and NaNs.
    E5M2 = 5,
    /// FP8 with a 4-bit exponent: ±448, no infinities, and two NaN codes.
    E4M3 = 6,
    /// FP6 with a 3-bit exponent, byte-aligned with two padding bits.
    E3M2 = 7,
    /// FP6 with a 2-bit exponent, byte-aligned with two padding bits.
    E2M3 = 8,

    /// 8-bit signed integer.
    I8 = 10,
    /// Two signed nibbles per byte, even dimensions in the low nibble.
    I4x2 = 11,
    /// 32-bit signed integer.
    I32 = 12,

    /// 8-bit unsigned integer - the element of UTF-8 text and of encoded files.
    U8 = 20,
    /// Two unsigned nibbles per byte, even dimensions in the low nibble.
    U4x2 = 21,
    /// Eight single-bit dimensions per byte, the sign of each embedding dimension - the form
    /// USearch's Hamming and Jaccard metrics score.
    U1x8 = 22,
}

impl Scalar {
    pub(crate) fn from_raw(value: uform_scalar_t) -> Option<Self> {
        let scalar = match value {
            1 => Scalar::F64,
            2 => Scalar::F32,
            3 => Scalar::BF16,
            4 => Scalar::F16,
            5 => Scalar::E5M2,
            6 => Scalar::E4M3,
            7 => Scalar::E3M2,
            8 => Scalar::E2M3,
            10 => Scalar::I8,
            11 => Scalar::I4x2,
            12 => Scalar::I32,
            20 => Scalar::U8,
            21 => Scalar::U4x2,
            22 => Scalar::U1x8,
            _ => return None,
        };
        Some(scalar)
    }
}

impl fmt::Display for Scalar {
    /// Calls `uform_scalar_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_scalar_name, *self as c_int))
    }
}

impl FromStr for Scalar {
    type Err = Error;

    /// Calls `uform_scalar_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, uform_scalar_from_name, Scalar::from_raw)
    }
}

// endregion: Scalars

// region: Classes

/// The CPU families UForm ships tuned variants for: dispatch reports its pick in this vocabulary,
/// and a RAM placement may pin one.
///
/// Mirrors `uform_cpu_class_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum CPUClass {
    /// The portable floor: no SIMD assumptions.
    Serial = 1,
    /// The one class of a monoversioned build: every capability its floor implies or its preset
    /// vouches for, as the kernel libraries compiled them.
    Monoversioned = 2,
    /// Haswell-class x86: AVX2 & FMA, the 256-bit floor.
    IntelHaswell = 3,
    /// Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI and its successors without AVX-512.
    IntelSierra = 4,
    /// Skylake-class x86: the AVX-512 foundation tier.
    IntelSkylake = 5,
    /// Icelake-class x86: AVX-512 VNNI & VPOPCNT.
    IntelIcelake = 6,
    /// Genoa-class x86 - Zen 4 and newer: native bf16 dot products over AVX-512.
    AMDGenoa = 7,
    /// Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16
    /// & AMX-INT8 tiles.
    IntelSapphire = 8,
    /// Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles.
    IntelGranite = 9,
    /// Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions.
    IntelDiamond = 10,
    /// Graviton2-class Arm - Neoverse N1, the bare NEON floor; phones predating Armv8.6
    /// land here too.
    AWSG2 = 12,
    /// Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2, Cortex-A710 &
    /// X2 class silicon.
    AppleM2 = 13,
    /// Apple M4-class Arm: the Armv8.6 NEON bundle plus the SME outer-product tiles.
    AppleM4 = 14,
    /// Graviton5-class Arm - Neoverse V3, Armv9.2.
    AWSG5 = 15,
    /// Vera-class Arm - NVIDIA's Armv9.2-generation cores with FP8 dot products.
    NVIDIAVera = 16,
    /// Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23 vectors.
    AlibabaC930 = 17,
    /// IBM POWER10-class Power, ISA 3.1: VSX vectors, the MMA outer-product accumulators and
    /// `pause_short` waits.
    IBMPower10 = 18,
    /// WebAssembly every engine loads: 32-bit, single-threaded, the fixed-width simd128 tier.
    Wasm32V128 = 19,
    /// WebAssembly with relaxed SIMD, shared memory for real thread pools, and 64-bit addressing.
    Wasm64Extended = 20,
}

impl CPUClass {
    pub(crate) fn from_raw(value: uform_cpu_class_t) -> Option<Self> {
        let cpu_class = match value {
            1 => CPUClass::Serial,
            2 => CPUClass::Monoversioned,
            3 => CPUClass::IntelHaswell,
            4 => CPUClass::IntelSierra,
            5 => CPUClass::IntelSkylake,
            6 => CPUClass::IntelIcelake,
            7 => CPUClass::AMDGenoa,
            8 => CPUClass::IntelSapphire,
            9 => CPUClass::IntelGranite,
            10 => CPUClass::IntelDiamond,
            12 => CPUClass::AWSG2,
            13 => CPUClass::AppleM2,
            14 => CPUClass::AppleM4,
            15 => CPUClass::AWSG5,
            16 => CPUClass::NVIDIAVera,
            17 => CPUClass::AlibabaC930,
            18 => CPUClass::IBMPower10,
            19 => CPUClass::Wasm32V128,
            20 => CPUClass::Wasm64Extended,
            _ => return None,
        };
        Some(cpu_class)
    }
}

impl fmt::Display for CPUClass {
    /// Calls `uform_cpu_class_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_cpu_class_name, *self as c_int))
    }
}

impl FromStr for CPUClass {
    type Err = Error;

    /// Calls `uform_cpu_class_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, uform_cpu_class_from_name, CPUClass::from_raw)
    }
}

/// The GPU families UForm ships tuned variants for, the device twin of [`CPUClass`]: dispatch
/// reports its pick in this vocabulary, and a VRAM placement may pin one.
///
/// Mirrors `uform_gpu_class_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GPUClass {
    /// Ampere-class NVIDIA - A100, A10 & RTX 30, and newer GPUs whose driver compiles its PTX.
    NVIDIAAmpere = 1,
    /// Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions.
    NVIDIAAda = 2,
    /// Hopper-class NVIDIA - H100 & H200.
    NVIDIAHopper = 3,
    /// Blackwell-class NVIDIA - B200, B300 & GB300.
    NVIDIABlackwell = 4,
    /// Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark.
    NVIDIABlackwellRTX = 5,
    /// Rubin-class NVIDIA - the Vera-Rubin platform's device half.
    NVIDIARubin = 6,
    /// CDNA3 AMD - MI300X & MI325X: gfx942, FNUZ fp8.
    AMDCDNA3 = 7,
    /// CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8.
    AMDCDNA4 = 8,
    /// CDNA5 AMD - MI400: gfx1250.
    AMDCDNA5 = 9,
    /// Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2.
    AppleMetal7 = 10,
    /// Apple9-family GPUs - M3 & M4.
    AppleMetal9 = 11,
    /// Apple10-family GPUs - M5.
    AppleMetal10 = 12,
}

impl GPUClass {
    pub(crate) fn from_raw(value: uform_gpu_class_t) -> Option<Self> {
        let gpu_class = match value {
            1 => GPUClass::NVIDIAAmpere,
            2 => GPUClass::NVIDIAAda,
            3 => GPUClass::NVIDIAHopper,
            4 => GPUClass::NVIDIABlackwell,
            5 => GPUClass::NVIDIABlackwellRTX,
            6 => GPUClass::NVIDIARubin,
            7 => GPUClass::AMDCDNA3,
            8 => GPUClass::AMDCDNA4,
            9 => GPUClass::AMDCDNA5,
            10 => GPUClass::AppleMetal7,
            11 => GPUClass::AppleMetal9,
            12 => GPUClass::AppleMetal10,
            _ => return None,
        };
        Some(gpu_class)
    }
}

impl fmt::Display for GPUClass {
    /// Calls `uform_gpu_class_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_gpu_class_name, *self as c_int))
    }
}

impl FromStr for GPUClass {
    type Err = Error;

    /// Calls `uform_gpu_class_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, uform_gpu_class_from_name, GPUClass::from_raw)
    }
}

/// The GPU vendors one binary drives at once.
///
/// Mirrors `uform_gpu_vendor_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GPUVendor {
    /// NVIDIA, through the CUDA driver.
    Nvidia = 1,
    /// AMD, through the HIP runtime.
    Amd = 2,
    /// Apple, through Metal.
    Apple = 3,
}

impl GPUVendor {
    fn from_raw(value: uform_gpu_vendor_t) -> Option<Self> {
        match value {
            1 => Some(GPUVendor::Nvidia),
            2 => Some(GPUVendor::Amd),
            3 => Some(GPUVendor::Apple),
            _ => None,
        }
    }
}

impl fmt::Display for GPUVendor {
    /// Calls `uform_gpu_vendor_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_gpu_vendor_name, *self as c_int))
    }
}

impl FromStr for GPUVendor {
    type Err = Error;

    /// Calls `uform_gpu_vendor_from_name`, refusing "unknown", which names no vendor.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, uform_gpu_vendor_from_name, GPUVendor::from_raw)
    }
}

/// One GPU as UForm numbers it: its position in what [`gpu_devices`] lists is the ordinal every
/// VRAM placement names it by.
///
/// Mirrors `uform_gpu_device_t`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct GPUDevice(uform_gpu_device_t);

impl Default for GPUDevice {
    fn default() -> Self {
        GPUDevice(uform_gpu_device_t {
            vendor: 0,
            vendor_ordinal: 0,
            gpu_class: 0,
            compute_units: 0,
            memory_bytes: 0,
            name: [0; 64],
        })
    }
}

impl GPUDevice {
    /// The vendor whose runtime drives the device.
    pub fn vendor(&self) -> Option<GPUVendor> {
        GPUVendor::from_raw(self.0.vendor)
    }

    /// The device's ordinal in its vendor's runtime, as `nvidia-smi` or `rocm-smi` list it.
    pub fn vendor_ordinal(&self) -> u32 {
        self.0.vendor_ordinal
    }

    /// The class whose kernels run on it.
    pub fn gpu_class(&self) -> Option<GPUClass> {
        GPUClass::from_raw(self.0.gpu_class)
    }

    /// Streaming multiprocessors or compute units.
    pub fn compute_units(&self) -> u32 {
        self.0.compute_units
    }

    /// Device memory in bytes.
    pub fn memory_bytes(&self) -> u64 {
        self.0.memory_bytes
    }

    /// The vendor's own name for the device, cut before a character the truncation split.
    pub fn name(&self) -> &str {
        // Safety: `c_char` and `u8` share size and alignment, so the name reads as bytes in place.
        let bytes = unsafe { core::slice::from_raw_parts(self.0.name.as_ptr().cast::<u8>(), self.0.name.len()) };
        let name = &bytes[..bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len())];
        match core::str::from_utf8(name) {
            Ok(name) => name,
            Err(error) => unsafe { core::str::from_utf8_unchecked(&name[..error.valid_up_to()]) },
        }
    }
}

impl fmt::Debug for GPUDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GPUDevice")
            .field("vendor", &self.vendor())
            .field("vendor_ordinal", &self.vendor_ordinal())
            .field("gpu_class", &self.gpu_class())
            .field("compute_units", &self.compute_units())
            .field("memory_bytes", &self.memory_bytes())
            .field("name", &self.name())
            .finish()
    }
}

/// Lists every GPU this process can drive into `devices`, NVIDIA devices first, then AMD, then
/// Apple, and answers how many there are, which may exceed what `devices` holds.
///
/// A vendor this build does not compile, or whose runtime is absent, lists no devices, so a machine
/// without GPUs answers zero.
///
/// Calls `uform_gpu_devices`.
pub fn gpu_devices(devices: &mut [GPUDevice]) -> Result<usize> {
    let mut count = 0;
    let mut outcome = uform_outcome_t::default();
    unsafe {
        uform_gpu_devices(&mut uform_gpu_devices_t {
            outcome: &mut outcome,
            devices: devices.as_mut_ptr().cast(),
            devices_capacity: devices.len(),
            devices_count: &mut count,
        })
    };
    check(outcome).map(|()| count)
}

// endregion: Classes

// region: Rows

/// Views `length` values a finished call wrote into a host arena; a call that wrote none may
/// leave a null base, which an empty run never reads.
fn exported_slice<'arena, Value>(data: *const Value, length: usize) -> Result<&'arena [Value]> {
    match (data.is_null(), length) {
        (_, 0) => Ok(&[]),
        (true, _) => Err(fail(Status::Internal, "the library exported rows without a base")),
        // Safety: the arena the view refers to holds a finished call's output in host memory.
        (false, length) => Ok(unsafe { core::slice::from_raw_parts(data, length) }),
    }
}

/// Rows a call wrote into its arena, one per item, packed.
///
/// Mirrors `uform_rows_view_t`.
pub struct Rows<'arena> {
    pub(crate) arena: &'arena Arena<'arena>,
    pub(crate) view: Exported<uform_rows_view_t>,
    pub(crate) count: usize,
}

impl<'arena> Rows<'arena> {
    /// Rows the call wrote.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Values per row, the model's embedding dimensions.
    pub fn columns(&self) -> usize {
        self.view.0.columns
    }

    /// Bytes between the starts of consecutive rows.
    pub fn stride(&self) -> usize {
        self.view.0.stride
    }

    /// The scalar type the rows hold.
    pub fn scalar(&self) -> Option<Scalar> {
        Scalar::from_raw(self.view.0.scalar)
    }

    /// The first row, in host or device memory, for handing to another library as it is.
    pub fn as_ptr(&self) -> *const c_void {
        self.view.0.data
    }

    /// Every row's bytes end to end, refused with [`Status::Device`] over device memory.
    pub fn as_bytes(&self) -> Result<&'arena [u8]> {
        self.arena.host()?;
        exported_slice(self.view.0.data.cast::<u8>(), self.count * self.view.0.stride)
    }

    /// Every row's values end to end, refused with [`Status::InvalidArgument`] when the rows hold
    /// another scalar type, and with [`Status::Device`] over device memory.
    pub fn values<Value: Element>(&self) -> Result<&'arena [Value]> {
        self.arena.host()?;
        let view = self.view.0;
        if self.scalar() != Some(Value::SCALAR) {
            return Err(fail(Status::InvalidArgument, "the rows hold another scalar type"));
        }
        if view.stride != view.columns * core::mem::size_of::<Value>() || !view.data.cast::<Value>().is_aligned() {
            return Err(fail(Status::UnsupportedLayout, "the rows are padded or misaligned"));
        }
        exported_slice(view.data.cast::<Value>(), self.count * view.columns)
    }

    /// Each row's bytes in order, refused with [`Status::Device`] over device memory.
    pub fn iter(&self) -> Result<impl ExactSizeIterator<Item = &'arena [u8]> + use<'arena>> {
        let bytes = self.as_bytes()?;
        let stride = self.view.0.stride;
        Ok((0..self.count).map(move |row| &bytes[row * stride..(row + 1) * stride]))
    }
}

/// Rows a call wrote into its arena, as many per item as the item had tokens left
/// after truncation, end to end, with one more offset than items.
///
/// Mirrors `uform_rows_tape_view_t`.
pub struct RowsTape<'arena> {
    pub(crate) arena: &'arena Arena<'arena>,
    pub(crate) view: Exported<uform_rows_tape_view_t>,
    pub(crate) count: usize,
}

impl<'arena> RowsTape<'arena> {
    /// Items the call encoded.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The call's count plus one non-decreasing offsets, in rows, refused with [`Status::Device`]
    /// over device memory.
    pub fn offsets(&self) -> Result<&'arena [usize]> {
        self.arena.host()?;
        exported_slice(self.view.0.offsets, self.count + 1)
    }

    /// Every item's rows end to end, refused with [`Status::Device`] over device memory.
    pub fn rows(&self) -> Result<Rows<'arena>> {
        let offsets = self.offsets()?;
        Ok(Rows { arena: self.arena, view: Exported(self.view.0.rows), count: offsets.last().copied().unwrap_or(0) })
    }

    /// The rows of one item, refused with [`Status::InvalidArgument`] when `item` is not below
    /// [`RowsTape::count`], and with [`Status::Device`] over device memory.
    pub fn item(&self, item: usize) -> Result<Rows<'arena>> {
        let offsets = self.offsets()?;
        let (Some(&start), Some(&end)) = (offsets.get(item), offsets.get(item + 1)) else {
            return Err(fail(Status::InvalidArgument, "the item is not below the call's count"));
        };
        let rows = self.view.0.rows;
        Ok(Rows {
            arena: self.arena,
            view: Exported(uform_rows_view_t {
                data: rows.data.cast::<u8>().wrapping_add(start * rows.stride).cast(),
                ..rows
            }),
            count: end - start,
        })
    }
}

/// Values a call wrote into its arena, a ragged run per item, end to end, with one more
/// offset than items.
///
/// Mirrors `uform_tape_view_t`.
pub struct Tape<'arena> {
    pub(crate) arena: &'arena Arena<'arena>,
    pub(crate) view: Exported<uform_tape_view_t>,
    pub(crate) count: usize,
}

impl<'arena> Tape<'arena> {
    /// Items the call wrote.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The scalar type the values are.
    pub fn scalar(&self) -> Option<Scalar> {
        Scalar::from_raw(self.view.0.scalar)
    }

    /// The call's count plus one non-decreasing offsets, in values, refused with [`Status::Device`]
    /// over device memory.
    pub fn offsets(&self) -> Result<&'arena [usize]> {
        self.arena.host()?;
        exported_slice(self.view.0.offsets, self.count + 1)
    }

    /// Every item's values end to end, refused with [`Status::InvalidArgument`] when the tape holds
    /// another scalar type, and with [`Status::Device`] over device memory.
    pub fn values<Value: Element>(&self) -> Result<&'arena [Value]> {
        let length = self.offsets()?.last().copied().unwrap_or(0);
        if self.scalar() != Some(Value::SCALAR) {
            return Err(fail(Status::InvalidArgument, "the tape holds another scalar type"));
        }
        exported_slice(self.view.0.data.cast::<Value>(), length)
    }

    /// The values of one item, refused with [`Status::InvalidArgument`] when `item` is not
    /// below [`Tape::count`] or the tape holds another scalar type, and with [`Status::Device`]
    /// over device memory.
    pub fn item<Value: Element>(&self, item: usize) -> Result<&'arena [Value]> {
        let offsets = self.offsets()?;
        let (Some(&start), Some(&end)) = (offsets.get(item), offsets.get(item + 1)) else {
            return Err(fail(Status::InvalidArgument, "the item is not below the call's count"));
        };
        self.values()?.get(start..end).ok_or(fail(Status::Internal, "the library exported offsets past its values"))
    }
}

// endregion: Rows

// region: Placements

/// The memory domains a RAM placement or executor names, and the class it caps dispatch at;
/// `None` sets no ceiling.
pub(crate) fn ram_options(memory_domains: &[i32], cpu_class_ceiling: Option<CPUClass>) -> uform_ram_options_t {
    uform_ram_options_t {
        memory_domains: memory_domains.as_ptr(),
        memory_domains_count: memory_domains.len(),
        cpu_class_ceiling: cpu_class_ceiling.map_or(0, |ceiling| ceiling as uform_cpu_class_t),
    }
}

/// The devices a VRAM placement or executor names, and the class it caps dispatch at; `None`
/// sets no ceiling.
pub(crate) fn vram_options(devices: &[i32], gpu_class_ceiling: Option<GPUClass>) -> uform_vram_options_t {
    uform_vram_options_t {
        devices: devices.as_ptr(),
        devices_count: devices.len(),
        gpu_class_ceiling: gpu_class_ceiling.map_or(0, |ceiling| ceiling as uform_gpu_class_t),
    }
}

/// Views a path as the bytes the C interface exchanges: the OS's own bytes on Unix, not
/// necessarily UTF-8, and UTF-8 where the OS keeps paths in another encoding.
pub(crate) fn path_view(path: &Path) -> Result<uform_bytes_view_t> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;

        Ok(path.as_os_str().as_bytes().into())
    }
    #[cfg(not(unix))]
    {
        let path = path.to_str().ok_or(fail(Status::InvalidArgument, "the path is not valid UTF-8"))?;
        Ok(path.as_bytes().into())
    }
}

// endregion: Placements

// region: Callbacks

/// A periodic report of finished and total units, lowered to the C struct every reporting verb
/// takes; `Break` stops the call at the next slice.
///
/// The handler runs on whichever thread advanced the work, never on two at once, so it owes `Send`
/// and may be `FnMut`. A panic inside it aborts the process, as it cannot unwind through C.
///
/// Mirrors `uform_progress_t`.
pub struct Progress<'handler> {
    raw: uform_progress_t,
    handler: PhantomData<&'handler mut ()>,
}

// Safety: it points at a `Send` handler it holds by `&mut`.
unsafe impl Send for Progress<'_> {}

impl<'handler> Progress<'handler> {
    /// Wraps `handler`, called with the units finished so far and the total the call finishes.
    pub fn new<Handler: FnMut(usize, usize) -> ControlFlow<()> + Send>(handler: &'handler mut Handler) -> Self {
        let raw = uform_progress_t {
            callback: Some(progress_trampoline::<Handler>),
            state: ptr::from_mut(handler).cast(),
            steps: 0,
        };
        Progress { raw, handler: PhantomData }
    }
}

/// # Safety
///
/// `state` must address the `Handler` of a [`Progress`] passed to the call.
unsafe extern "C" fn progress_trampoline<Handler: FnMut(usize, usize) -> ControlFlow<()>>(
    finished: usize,
    total: usize,
    state: *mut c_void,
) -> c_int {
    unsafe {
        let handler = &mut *state.cast::<Handler>();
        c_int::from(handler(finished, total).is_break())
    }
}

/// Lowers an optional progress and its step count as reporting verbs take them; `None` lowers to
/// nulls.
pub(crate) fn lower_progress(progress: &Option<Progress>, steps: usize) -> uform_progress_t {
    match progress {
        Some(progress) => uform_progress_t { steps, ..progress.raw },
        None => uform_progress_t { callback: None, state: ptr::null_mut(), steps },
    }
}

/// Lowers an optional output slot, a reference or owned, to the pointer the library overwrites;
/// `None` lowers to null.
pub(crate) fn output_slot<Value, Slot: BorrowMut<Value>>(slot: &mut Option<Slot>) -> *mut Value {
    slot.as_mut().map_or(ptr::null_mut(), |slot| ptr::from_mut(slot.borrow_mut()))
}

// endregion: Callbacks

// region: Executors

/// What an executor does with its threads or streams once it goes idle.
///
/// Mirrors `uform_mood_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Mood {
    /// Sleep in the kernel until the next call, freeing the cores.
    #[default]
    Parked = 0,
    /// Keep polling, trading the cores for the lowest wake-up latency.
    Spinning = 1,
}

impl Mood {
    fn from_raw(value: uform_mood_t) -> Option<Self> {
        match value {
            0 => Some(Mood::Parked),
            1 => Some(Mood::Spinning),
            _ => None,
        }
    }
}

impl fmt::Display for Mood {
    /// Calls `uform_mood_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_mood_name, *self as c_int))
    }
}

impl FromStr for Mood {
    type Err = Error;

    /// Calls `uform_mood_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, uform_mood_from_name, Mood::from_raw)
    }
}

/// Where an executor computes, and with it where the arenas it makes keep their memory.
///
/// Mirrors `uform_residency_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Residency {
    /// Host pages, which the host reads directly.
    Ram = 1,
    /// One GPU's memory, which only device code reads.
    Vram = 2,
}

impl Residency {
    fn from_raw(value: uform_residency_t) -> Option<Self> {
        match value {
            1 => Some(Residency::Ram),
            2 => Some(Residency::Vram),
            _ => None,
        }
    }
}

/// The host threads or device streams an [`On`] runs its calls on, with the activation scratch and
/// the staging buffers each of them uses.
///
/// One executor serves any number of models of its residency, so several models share one
/// machine's cores or devices instead of each claiming a private set.
///
/// [`Executor::default`] is the C interface's null executor: it owns nothing, and every call
/// handed it runs on the calling thread.
///
/// An executor runs one call at a time, so threads wanting to overlap take one apiece; it is `Send`
/// but not `Sync`. A call finding another one holding it fails with [`Status::Contended`].
/// [`Arena::new_in`] places an arena where it computes, and the arena refers to it, so an arena
/// never outlives the executor that made it:
///
/// ```compile_fail,E0505
/// use uform::*;
///
/// let executor = Executor::for_ram(&[], None, 0)?;
/// let arena = Arena::new_in(&executor)?;
/// drop(executor);
/// let _ = arena.describe();
/// # Ok::<(), uform::Error>(())
/// ```
///
/// Mirrors `uform_executor_t`; dropping it calls `uform_executor_free`.
#[derive(Debug)]
pub struct Executor {
    pub(crate) handle: uform_executor_t,
}

// Safety: nothing in an executor is tied to the thread that created it, so it may move to another;
// staying out of `Sync` keeps submissions to it on one thread at a time.
unsafe impl Send for Executor {}

impl Default for Executor {
    fn default() -> Self {
        Executor::adopt(ptr::null_mut())
    }
}

impl Executor {
    /// Owns the handle a constructor filled, which is null for the executor that owns nothing.
    fn adopt(handle: uform_executor_t) -> Self {
        Executor { handle }
    }

    /// Creates an executor of host threads for RAM-resident models, interleaved across the compute
    /// domains local to `memory_domains`, which is the same list the models were loaded with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains span, and its calls dispatch as the
    /// newest class the CPU runs at or under `cpu_class_ceiling`, where `None` sets no ceiling.
    ///
    /// Calls `uform_executor_for_ram`.
    pub fn for_ram(memory_domains: &[i32], cpu_class_ceiling: Option<CPUClass>, threads: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_executor_for_ram(&mut uform_executor_for_ram_t {
                executor: &mut handle,
                outcome: &mut outcome,
                ram: ram_options(memory_domains, cpu_class_ceiling),
                threads,
            })
        };
        let executor = Executor::adopt(handle);
        check(outcome).map(|()| executor)
    }

    /// Opens one submission stream on every named device, all of them driven by the host thread
    /// running the call - which is why this takes no thread count.
    ///
    /// `devices` are positions in what [`gpu_devices`] lists, and all of them dispatch as the best
    /// class they all load at or under `gpu_class_ceiling`, where `None` sets no ceiling. The host
    /// decoders take 7/8 of the cores local to `memory_domains` and dispatch under
    /// `cpu_class_ceiling` likewise. `streams` is empty, or holds one stream per device in
    /// `devices` order: a CUDA or HIP stream becomes the executor's, which synchronizes and
    /// destroys it; a Metal queue is retained, the executor releasing its own reference while the
    /// caller keeps and releases theirs. Zero `scratch_limit` lets scratch grow as batches ask.
    ///
    /// Calls `uform_executor_for_vram`.
    ///
    /// # Safety
    ///
    /// Every stream must be a live CUDA stream on its device that outlives the executor.
    pub unsafe fn for_vram(
        devices: &[i32],
        gpu_class_ceiling: Option<GPUClass>,
        memory_domains: &[i32],
        cpu_class_ceiling: Option<CPUClass>,
        streams: &[*mut c_void],
        scratch_limit: usize,
    ) -> Result<Self> {
        unsafe {
            if !streams.is_empty() && streams.len() != devices.len() {
                return Err(fail(Status::ShapeMismatch, "pass no streams or one per device"));
            }
            let mut handle = ptr::null_mut();
            let mut outcome = uform_outcome_t::default();
            uform_executor_for_vram(&mut uform_executor_for_vram_t {
                executor: &mut handle,
                outcome: &mut outcome,
                vram: vram_options(devices, gpu_class_ceiling),
                ram: ram_options(memory_domains, cpu_class_ceiling),
                // The C side reads one stream per device from any non-null pointer.
                streams: match streams.is_empty() {
                    true => ptr::null(),
                    false => streams.as_ptr(),
                },
                scratch_limit,
            });
            let executor = Executor::adopt(handle);
            check(outcome).map(|()| executor)
        }
    }

    /// Waits until no call holds this executor and its devices finished their work, then applies
    /// `mood` until its next call; a failure is the executor's own, such as a device fault, never
    /// one of a drained call.
    ///
    /// Calls `uform_executor_drain`.
    pub fn drain(&self, mood: Mood) -> Result<()> {
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_executor_drain(&mut uform_executor_drain_t {
                executor: self.handle,
                outcome: &mut outcome,
                mood: mood as uform_mood_t,
            })
        };
        check(outcome)
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        unsafe { uform_executor_free(self.handle) }
    }
}

// endregion: Executors

// region: Arenas

/// Memory the library writes every output whose length scales with a call into,
/// reused across calls.
///
/// An executor places it where it computes, host pages or its GPU's memory, decided once, and
/// nothing is allocated until a call writes into it. Every call rewinds and refills it,
/// invalidating the previous call's outputs, so a steady loop allocates once. Every verb, blocking
/// or deferred, takes it as `&mut Arena` and answers outputs that refer to it in turn. An arena
/// from a VRAM executor holds device memory, so the host accessors of its outputs refuse with
/// [`Status::Device`]. It grows, fills and frees on the first stream of the executor that made it,
/// so it serves only calls on that executor, and `'executor` keeps it from outliving that executor.
///
/// Mirrors `uform_arena_t`; dropping it calls `uform_arena_free`, which waits for the calls
/// still writing into it.
#[derive(Debug)]
pub struct Arena<'executor> {
    pub(crate) handle: uform_arena_t,
    executor: PhantomData<&'executor ()>,
}

// Safety: an arena serves one call at a time, which the `&mut` every verb takes already enforces,
// and a shared reference only asks the library where it lives.
unsafe impl Send for Arena<'_> {}
unsafe impl Sync for Arena<'_> {}

impl Arena<'static> {
    /// Makes an arena of host pages for calls on the calling thread.
    ///
    /// Calls `uform_arena_init`.
    pub fn new() -> Result<Self> {
        Arena::made_by(ptr::null_mut())
    }
}

impl<'executor> Arena<'executor> {
    /// Makes an arena where `executor` computes, serving the calls of its residency, or only those
    /// on `executor` itself for a GPU arena.
    ///
    /// Calls `uform_arena_init`.
    pub fn new_in(executor: &'executor Executor) -> Result<Self> {
        Arena::made_by(executor.handle)
    }

    /// The arena `executor` places, where null places it for the calling thread.
    fn made_by(executor: uform_executor_t) -> Result<Self> {
        let mut arena = Arena { handle: ptr::null_mut(), executor: PhantomData };
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_arena_init(&mut uform_arena_init_t { arena: &mut arena.handle, executor, outcome: &mut outcome })
        };
        check(outcome).map(|()| arena)
    }

    /// Where the outputs live, as the executor that made the arena decided.
    ///
    /// Calls `uform_arena_describe`.
    pub fn describe(&self) -> Result<ArenaDescription> {
        let mut description = uform_arena_description_t::default();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_arena_describe(&mut uform_arena_describe_t {
                arena: self.handle,
                outcome: &mut outcome,
                description: &mut description,
            })
        };
        check(outcome)?;
        Ok(ArenaDescription {
            residency: Residency::from_raw(description.residency)
                .ok_or_else(|| fail(Status::UnknownKind, "the arena names no known residency"))?,
            device: u32::try_from(description.device).ok(),
        })
    }

    /// Refuses device memory, which no host accessor may read.
    pub(crate) fn host(&self) -> Result<()> {
        match self.describe()?.residency {
            Residency::Vram => Err(fail(Status::Device, "the outputs are device memory")),
            _ => Ok(()),
        }
    }
}

/// Where an [`Arena`]'s outputs live.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct ArenaDescription {
    /// [`Residency::Ram`] for host pages, [`Residency::Vram`] for one device's memory.
    pub residency: Residency,
    /// The device ordinal, as [`Executor::for_vram`] takes it; `None` for host pages.
    pub device: Option<u32>,
}

impl Drop for Arena<'_> {
    fn drop(&mut self) {
        unsafe { uform_arena_free(self.handle) }
    }
}

/// A view the library wrote into the arena of the verb or result.
#[derive(Default)]
pub(crate) struct Exported<View>(pub(crate) View);

// Safety: the pointees live in the arena held beside the view, and references to it are `Send`.
unsafe impl<View> Send for Exported<View> {}

// endregion: Arenas

// region: Receivers

/// A reference to a model or a trainer, with the executor its calls run on and how they complete.
///
/// Neither the receiver nor a [`Deferred`] it makes outlives the handle or the executor.
///
/// Rust-only.
#[derive(Debug)]
pub struct On<'handle, Handle, Completion = Blocking> {
    pub(crate) handle: &'handle Handle,
    pub(crate) executor: &'handle Executor,
    pub(crate) completion: PhantomData<Completion>,
}

impl<Handle, Completion> Clone for On<'_, Handle, Completion> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Handle, Completion> Copy for On<'_, Handle, Completion> {}

/// The completion of a blocking receiver, whose verbs return once their call is final and answer
/// what it produced.
///
/// Rust-only.
#[derive(Debug)]
pub struct Blocking;

impl Completion for Blocking {
    type Deferrable<'handle, Verb: Calling> = Result<Verb::Output>;

    fn answer<'handle, Verb: Calling>(mut verb: Verb) -> Self::Deferrable<'handle, Verb> {
        let mut outcome = uform_outcome_t::default();
        let mut arguments = verb.arguments(&mut outcome, uform_completion_blocking_k)?;
        // Safety: the arguments point into `verb` and `outcome`, which stay in place until the
        // call returns, and a blocking call is final on return.
        unsafe { Verb::call(&mut arguments) };
        verb.finish(check(outcome)).0
    }
}

impl<Input> Accepts<Input> for Blocking {}

/// The completion of a deferring receiver, whose verbs check their arguments and answer a
/// [`Deferred`], submitted once first awaited or waited on.
///
/// Rust-only.
#[derive(Debug)]
pub struct Deferring;

impl Completion for Deferring {
    type Deferrable<'handle, Verb: Calling> = Result<Deferred<'handle, Verb>>;

    fn answer<'handle, Verb: Calling>(mut verb: Verb) -> Self::Deferrable<'handle, Verb> {
        // The arguments are rebuilt where the call is pinned; this pass only surfaces refusals.
        verb.arguments(&mut uform_outcome_t::default(), uform_completion_deferred_k)?;
        Ok(Deferred {
            call: UnsafeCell::new(Call {
                outcome: uform_outcome_t::default(),
                arguments: MaybeUninit::uninit(),
                verb: ManuallyDrop::new(verb),
            }),
            stage: Stage::Unsubmitted,
            receiver: PhantomData,
            pinned: PhantomPinned,
        })
    }
}

impl<Item: 'static> Accepts<Vec<Item>> for Deferring {}
impl<Item: 'static, const COUNT: usize> Accepts<[Item; COUNT]> for Deferring {}
impl<Data: 'static, Offsets: 'static> Accepts<(Data, Offsets)> for Deferring {}
impl Accepts<PathBuf> for Deferring {}
impl Accepts<String> for Deferring {}
impl Accepts<&'static str> for Deferring {}
impl Accepts<&'static Path> for Deferring {}

/// How far a [`Deferred`] got: built, handed to the library, or answered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Unsubmitted,
    Submitted,
    Answered,
}

/// What the library reads and writes while a call is in flight, reached only through raw pointers.
struct Call<Verb: Calling> {
    outcome: uform_outcome_t,
    arguments: MaybeUninit<Verb::Arguments>,
    verb: ManuallyDrop<Verb>,
}

/// A call submitted on its first poll or on [`Deferred::wait`], at the address it is pinned at, so
/// its outcome, arguments, inputs and output slots live inline and need no allocation.
///
/// It is the [`Future`] itself: `.await` submits it and re-polls `uform_outcome_status` until the
/// call is final, and [`Deferred::wait`] blocks in `uform_outcome_wait` instead. Dropping a
/// submitted one waits for it. `'handle` is the lifetime of the [`On`] that made it, and the call
/// holds its arena by `&mut` as a blocking call does, so no deferred call outlives either, while a
/// handler it may still call must live for `'static`:
///
/// ```compile_fail,E0597
/// use uform::*;
///
/// let model = Model::load_ram("model.safetensors")?;
/// let executor = Executor::for_ram(&[], None, 0)?;
/// let mut arena = Arena::new_in(&executor)?;
/// let mut handler = |_: usize, _: usize| core::ops::ControlFlow::Continue(());
/// let config = TextEncodeConfig {
///     progress: Some(Progress::new(&mut handler)),
///     ..TextEncodeConfig::from(Scalar::F32)
/// };
/// let call = model.deferred(&executor).text_encode(vec!["a"], Outputs::EMBEDDINGS, config, &mut arena)?;
/// let encoded = call.wait()?;
/// # Ok::<(), uform::Error>(())
/// ```
///
/// Nor does a second call reuse an arena while the first can still write into it:
///
/// ```compile_fail,E0499
/// use uform::*;
///
/// let model = Model::load_ram("model.safetensors")?;
/// let executor = Executor::for_ram(&[], None, 0)?;
/// let mut arena = Arena::new_in(&executor)?;
/// let first = model.deferred(&executor).text_encode(vec!["a"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
/// let second = model.deferred(&executor).text_encode(vec!["b"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
/// first.wait()?;
/// second.wait()?;
/// # Ok::<(), uform::Error>(())
/// ```
///
/// Rust-only; polling calls `uform_outcome_status`, and waiting and dropping a submitted call call
/// `uform_outcome_wait`.
#[must_use = "a deferred call runs only once awaited or waited on"]
pub struct Deferred<'handle, Verb: Calling> {
    // The library writes here while `stage` is `Submitted`.
    call: UnsafeCell<Call<Verb>>,
    stage: Stage,
    receiver: PhantomData<&'handle ()>,
    #[allow(dead_code, reason = "only its type matters: it keeps a pinned call from moving")]
    pinned: PhantomPinned,
}

// Safety: until submission it is plain data; once submitted it is pinned and only the library
// reaches it.
unsafe impl<Verb: Calling + Send> Send for Deferred<'_, Verb> {}

impl<Verb: Calling> Deferred<'_, Verb> {
    fn outcome(&self) -> *mut uform_outcome_t {
        unsafe { ptr::addr_of_mut!((*self.call.get()).outcome) }
    }

    fn submit(self: Pin<&mut Self>) {
        // Safety: nothing is moved out, and the pointers taken here stay valid while pinned.
        let this = unsafe { self.get_unchecked_mut() };
        let call = this.call.get();
        this.stage = Stage::Submitted;
        unsafe {
            let outcome = ptr::addr_of_mut!((*call).outcome);
            match (*call).verb.arguments(outcome, uform_completion_deferred_k) {
                Ok(arguments) => Verb::call((*call).arguments.write(arguments)),
                Err(_) => (*outcome).status = Status::Internal as uform_status_t,
            }
        }
    }

    /// Hands back the output of a final call, beside its inputs; the library touches it no more.
    fn finish(self: Pin<&mut Self>) -> (Result<Verb::Output>, Verb::Inputs) {
        let this = unsafe { self.get_unchecked_mut() };
        this.stage = Stage::Answered;
        let status = check(unsafe { ptr::read(this.outcome()) });
        unsafe { ManuallyDrop::take(&mut (*this.call.get()).verb) }.finish(status)
    }

    /// Submits the call, then blocks until it is final and hands back what it produced.
    ///
    /// Calls `uform_outcome_wait`.
    pub fn wait(self) -> Result<Verb::Output> {
        self.wait_with_inputs().0
    }

    /// Submits the call, then blocks until it is final and hands back what it produced beside the
    /// inputs it owned, such as a config's handler and stats.
    ///
    /// Calls `uform_outcome_wait`.
    pub fn wait_with_inputs(self) -> (Result<Verb::Output>, Verb::Inputs) {
        let mut this = pin!(self);
        this.as_mut().submit();
        unsafe { uform_outcome_wait(this.outcome()) };
        this.finish()
    }
}

impl<Verb: Calling> Future for Deferred<'_, Verb> {
    type Output = Result<Verb::Output>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match self.stage {
            Stage::Answered => return Poll::Ready(Err(fail(Status::InvalidArgument, "the call was already awaited"))),
            Stage::Unsubmitted => self.as_mut().submit(),
            Stage::Submitted => {}
        }
        if unsafe { uform_outcome_status(self.outcome()) } == uform_status_deferred_k {
            context.waker().wake_by_ref();
            return Poll::Pending;
        }
        Poll::Ready(self.finish().0)
    }
}

impl<Verb: Calling> Drop for Deferred<'_, Verb> {
    fn drop(&mut self) {
        match self.stage {
            Stage::Answered => return,
            Stage::Submitted => unsafe {
                uform_outcome_wait(self.outcome());
            },
            Stage::Unsubmitted => {}
        }
        unsafe { ManuallyDrop::drop(&mut (*self.call.get()).verb) };
    }
}

impl<Verb: Calling> fmt::Debug for Deferred<'_, Verb> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Deferred({:?})", self.stage)
    }
}

// endregion: Receivers

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    #[test]
    fn scalar_kinds_round_trip_through_their_tags() {
        let scalars = [
            Scalar::F64,
            Scalar::F32,
            Scalar::BF16,
            Scalar::F16,
            Scalar::E5M2,
            Scalar::E4M3,
            Scalar::E3M2,
            Scalar::E2M3,
            Scalar::I8,
            Scalar::I4x2,
            Scalar::I32,
            Scalar::U8,
            Scalar::U4x2,
            Scalar::U1x8,
        ];
        for scalar in scalars {
            assert_eq!(Scalar::from_raw(scalar as i32), Some(scalar));
            assert_eq!(scalar.to_string().parse(), Ok(scalar), "{scalar:?}");
        }
        // The C enum's numbering gaps are reserved, not assigned.
        assert_eq!(Scalar::from_raw(0), None);
        assert_eq!(Scalar::from_raw(9), None);
    }

    /// Every C inverse answers a default for a name it does not know, which must not read back
    /// as that default.
    #[test]
    fn names_round_trip_through_the_library_and_refuse_strangers() {
        assert_eq!(Status::ShapeMismatch.to_string(), "shape-mismatch");
        assert_eq!("cancelled".parse(), Ok(Status::Cancelled));
        assert_eq!("invalid-argument".parse(), Ok(Status::InvalidArgument));
        assert_eq!("uninitialized".parse(), Ok(Status::Uninitialized));
        assert_eq!(Mood::Spinning.to_string(), "spinning");
        assert_eq!("parked".parse(), Ok(Mood::Parked));

        assert!("success".parse::<Status>().is_err(), "success names no failure");
        assert!("shape mismatch".parse::<Status>().is_err());
        assert!("unknown".parse::<Scalar>().is_err());
        assert!("parking".parse::<Mood>().is_err());
    }

    #[test]
    fn statuses_survive_the_round_trip_and_tolerate_newer_libraries() {
        let failures = [
            Status::OutOfMemory,
            Status::ShapeMismatch,
            Status::UnsupportedLayout,
            Status::UnknownKind,
            Status::Io,
            Status::Device,
            Status::Malformed,
            Status::Internal,
            Status::Uninitialized,
            Status::ConfigMismatch,
            Status::Contended,
            Status::Cancelled,
            Status::Unsupported,
            Status::InvalidArgument,
        ];
        for status in failures {
            let raw = uform_outcome_t { status: status as i32, message: ptr::null() };
            assert_eq!(check(raw).unwrap_err().status, status);
            assert_eq!(status.to_string().parse(), Ok(status));
        }
        // A final outcome is never deferred, and a status this build has no name for is not
        // success either, so both read as the contract violation they are; USearch's 6 and 11
        // stay unused.
        for status in [Status::Deferred as i32, 6, 11, 4242] {
            let raw = uform_outcome_t { status, message: ptr::null() };
            assert_eq!(check(raw).unwrap_err().status, Status::Internal);
        }
    }

    #[test]
    fn separate_items_gather_into_one_tape() {
        let mut gathered = Gathered::default();
        let batch = TapeView::<u8>::tape(&["", "uform", "ab"], &mut gathered).unwrap();
        assert_eq!(batch.count, 3);
        assert_eq!(gathered.offsets, [0, 0, 5, 7]);
        assert_eq!(gathered.data, b"uformab");
        assert_eq!(batch.tape.scalar, Scalar::U8 as i32);

        let batch = TapeView::<f32>::tape(&vec![vec![0.5f32; 3], vec![]], &mut Gathered::default()).unwrap();
        assert_eq!((batch.count, batch.tape.scalar), (2, Scalar::F32 as i32));
    }

    /// A `(data, offsets)` tape reaches the library as it is, so offsets that would read past its
    /// data are refused before the call.
    #[test]
    fn uncopied_tapes_are_checked_before_the_call() {
        let mut gathered = Gathered::default();
        let data = b"firstsecond".as_slice();
        let batch = TapeView::<u8>::tape(&(data, [0usize, 5, 11]), &mut gathered).unwrap();
        assert_eq!(batch.count, 2);
        assert_eq!(batch.tape.data, data.as_ptr().cast());
        assert!(gathered.data.is_empty(), "an uncopied tape gathers nothing");

        let refusals = [
            ([0usize, 5, 12].as_slice(), Status::ShapeMismatch),
            ([0usize, 6, 5].as_slice(), Status::Malformed),
            ([].as_slice(), Status::InvalidArgument),
        ];
        for (offsets, status) in refusals {
            let refusal = TapeView::<u8>::tape(&(data, offsets), &mut gathered).err().map(|error| error.status);
            assert_eq!(refusal, Some(status), "{offsets:?}");
        }
    }

    /// An arena lives where the executor that made it computes, and the host reads only host pages.
    #[test]
    fn arenas_live_where_their_executor_computes() {
        let local = Arena::new().unwrap();
        let host = ArenaDescription { residency: Residency::Ram, device: None };
        assert_eq!(local.describe().unwrap(), host);
        assert!(local.host().is_ok());

        let executor = testing::executor();
        let placed = Arena::new_in(&executor).unwrap();
        assert_eq!(placed.describe().unwrap(), host);

        let Ok(gpu) = (unsafe { Executor::for_vram(&[0], None, &[], None, &[], 0) }) else {
            return testing::skipped("no GPU device");
        };
        let device = Arena::new_in(&gpu).unwrap();
        assert_eq!(device.describe().unwrap(), ArenaDescription { residency: Residency::Vram, device: Some(0) });
        assert_eq!(device.host().unwrap_err().status, Status::Device);
    }

    /// Streams that do not pair with the devices are refused.
    #[test]
    fn executors_refuse_malformed_placements() {
        let mismatched = unsafe { Executor::for_vram(&[0, 1], None, &[], None, &[ptr::null_mut()], 0) };
        assert_eq!(mismatched.unwrap_err().status, Status::ShapeMismatch);
    }

    #[test]
    fn classes_round_trip_through_the_library() {
        assert_eq!(CPUClass::IntelHaswell.to_string(), "intel-haswell");
        assert_eq!("wasm64-extended".parse(), Ok(CPUClass::Wasm64Extended));
        assert_eq!(GPUClass::NVIDIAHopper.to_string(), "nvidia-hopper");
        assert_eq!("apple-metal10".parse(), Ok(GPUClass::AppleMetal10));
        assert!("unknown".parse::<CPUClass>().is_err());
        assert!("intel".parse::<GPUClass>().is_err());
        assert_eq!(CPUClass::from_raw(0), None);
        assert_eq!(GPUClass::from_raw(13), None);
        for vendor in [GPUVendor::Nvidia, GPUVendor::Amd, GPUVendor::Apple] {
            assert_eq!(vendor.to_string().parse(), Ok(vendor));
        }
        assert_eq!(GPUVendor::Nvidia.to_string(), "nvidia");
        assert!("unknown".parse::<GPUVendor>().is_err());
        assert!("intel".parse::<GPUVendor>().is_err());
    }

    /// A machine without GPUs, or a build without their vendors, lists none and still succeeds.
    #[test]
    fn listed_devices_never_outnumber_their_count() {
        let mut devices = [GPUDevice::default(); 4];
        let count = gpu_devices(&mut devices).unwrap();
        assert_eq!(gpu_devices(&mut []).unwrap(), count);
        for device in &devices[..count.min(devices.len())] {
            assert!(device.vendor().is_some() && !device.name().is_empty(), "{device:?}");
        }
    }

    #[test]
    fn introspection_reads_static_strings() {
        assert!(!capabilities_compiled().is_empty());
        assert!(!capabilities_available().is_empty());
    }
}

// endregion: Tests
