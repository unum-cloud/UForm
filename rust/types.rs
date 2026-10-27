//! Shared vocabulary of every UForm verb: statuses, scalars, tapes, rows, executors and arenas.
//!
//! An [`On`] is the receiver every verb runs through: it borrows a model or a trainer and an
//! executor, and its completion decides whether a verb answers what the call produced or a
//! [`Deferred`] once the call is submitted.
//!
//! File: rust/types.rs
//! Author: Ash Vardanian
//! Date: October 18, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types, non_upper_case_globals)]

use core::any::Any;
use core::cell::Cell;
use core::ffi::{c_char, c_int, c_void, CStr};
use core::fmt;
use core::future::{Future, IntoFuture};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::ControlFlow;
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::str::FromStr;
use core::task::{Context, Poll};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

// region: Raw Declarations

/// A borrowed run of bytes, never null-terminated by contract.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_bytes_view_t {
    pub data: *const u8,
    pub length: usize,
}

impl From<&[u8]> for uform_bytes_view_t {
    /// Borrows the bytes as the view the C interface accepts; it must not outlive them.
    fn from(bytes: &[u8]) -> Self {
        uform_bytes_view_t {
            data: bytes.as_ptr(),
            length: bytes.len(),
        }
    }
}

impl Default for uform_bytes_view_t {
    fn default() -> Self {
        uform_bytes_view_t {
            data: ptr::null(),
            length: 0,
        }
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
        uform_outcome_t {
            status: uform_status_success_k,
            message: ptr::null(),
        }
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
        uform_rows_view_t {
            data: ptr::null(),
            stride: 0,
            columns: 0,
            scalar: 0,
        }
    }
}

/// A borrowed ragged batch: elements end to end, and one more offset than entries.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_tape_view_t {
    pub data: *const c_void,
    pub offsets: *const usize,
    pub scalar: uform_scalar_t,
}

impl Default for uform_tape_view_t {
    fn default() -> Self {
        uform_tape_view_t {
            data: ptr::null(),
            offsets: ptr::null(),
            scalar: 0,
        }
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
        uform_rows_tape_view_t {
            rows: uform_rows_view_t::default(),
            offsets: ptr::null(),
        }
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
    pub cpu_class: uform_cpu_class_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_vram_options_t {
    pub devices: *const i32,
    pub devices_count: usize,
    pub gpu_class: uform_gpu_class_t,
}

pub type uform_executor_t = *mut c_void;
pub type uform_arena_t = *mut c_void;

#[repr(C)]
pub struct uform_executor_for_ram_t {
    pub executor: *mut uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub ram: uform_ram_options_t,
    pub threads: usize,
    pub deferred_limit: usize,
}

#[repr(C)]
pub struct uform_executor_on_pool_t {
    pub executor: *mut uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub ram: uform_ram_options_t,
    pub pool: *mut c_void,
    pub deferred_limit: usize,
}

#[repr(C)]
pub struct uform_executor_for_vram_t {
    pub executor: *mut uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub vram: uform_vram_options_t,
    pub streams: *const *mut c_void,
    pub scratch_limit: usize,
    pub deferred_limit: usize,
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

extern "C" {
    pub fn uform_outcome_wait(outcome: *mut uform_outcome_t) -> uform_status_t;
    pub fn uform_outcome_status(outcome: *mut uform_outcome_t) -> uform_status_t;

    pub fn uform_executor_for_ram(arguments: *mut uform_executor_for_ram_t);
    pub fn uform_executor_on_pool(arguments: *mut uform_executor_on_pool_t);
    pub fn uform_executor_for_vram(arguments: *mut uform_executor_for_vram_t);
    pub fn uform_executor_drain(arguments: *mut uform_executor_drain_t);
    pub fn uform_executor_free(executor: uform_executor_t);

    pub fn uform_arena_init(arguments: *mut uform_arena_init_t);
    pub fn uform_arena_free(arena: uform_arena_t);

    pub fn uform_status_name(status: uform_status_t) -> *const c_char;
    pub fn uform_status_from_name(name: uform_bytes_view_t) -> uform_status_t;
    pub fn uform_scalar_name(scalar: uform_scalar_t) -> *const c_char;
    pub fn uform_scalar_from_name(name: uform_bytes_view_t) -> uform_scalar_t;
    pub fn uform_mood_name(mood: uform_mood_t) -> *const c_char;
    pub fn uform_mood_from_name(name: uform_bytes_view_t) -> uform_mood_t;
    pub fn uform_cpu_class_name(cpu_class: uform_cpu_class_t) -> *const c_char;
    pub fn uform_cpu_class_from_name(name: uform_bytes_view_t) -> uform_cpu_class_t;
    pub fn uform_gpu_class_name(gpu_class: uform_gpu_class_t) -> *const c_char;
    pub fn uform_gpu_class_from_name(name: uform_bytes_view_t) -> uform_gpu_class_t;

    pub fn uform_gpu_devices(arguments: *mut uform_gpu_devices_t);

    pub fn uform_version_name() -> *const c_char;
    pub fn uform_capabilities_compiled() -> *const c_char;
    pub fn uform_capabilities_available() -> *const c_char;
}

// endregion: Raw Declarations

// region: Sealed Traits

/// The handles every call names: the model or trainer it reaches and the executor it runs on.
#[doc(hidden)]
#[derive(Debug)]
pub struct Handles {
    pub(crate) handle: *mut c_void,
    pub(crate) executor: uform_executor_t,
}

// Safety: a model is shared across threads, and an executor goes with the calls submitted to it.
unsafe impl Send for Handles {}

/// One verb over one `#[repr(C)]` argument struct: where the arguments point, which C entry
/// point takes them, and what a successful call hands back.
#[doc(hidden)]
pub trait Calling: Sized {
    /// The argument struct of the verb's C entry point.
    type Arguments;
    /// What a successful call hands back.
    type Output;

    /// The arguments over the inputs and outputs this verb holds, which stays in place until the
    /// call they feed is final, lending its handlers through `callbacks`.
    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments>;

    /// Passes `arguments` to the verb's C entry point.
    ///
    /// # Safety
    /// Everything `arguments` points to stays alive and in place until the outcome is final.
    unsafe fn call(arguments: *mut Self::Arguments);

    /// What the verb hands back once its outcome is a success.
    fn finish(self) -> Result<Self::Output>;
}

/// How the verbs of a receiver complete, which decides what each of them answers.
#[doc(hidden)]
pub trait Completion: Sized {
    /// `Result<Output>` when blocking, `Result<Deferred<'handle, Verb>>` when deferring.
    type Answer<'handle, Verb: Calling>;

    /// Runs `verb`, or submits it.
    fn answer<'handle, Verb: Calling>(verb: Verb) -> Self::Answer<'handle, Verb>;
}

/// The inputs a completion takes: a blocking receiver takes borrows and owned values alike, while
/// a deferring one takes only what it owns, since the library reads its inputs after the verb returns.
/// Both borrow the arena, which no call reads as an input.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "a deferred receiver needs owned inputs",
    label = "the library would still read this borrow after the verb returns",
    note = "pass texts, files and samples in a `Vec` or an array of owned or `'static` items, a tape as \
            `(Vec<_>, Vec<usize>)`, a `PathBuf`, `String` or `'static` path, and `'static` configs and handlers"
)]
pub trait Accepts<Input> {}

/// A scalar type a tape holds or a run of rows reads as, tied to the [`Scalar`] the C interface
/// tags its buffers with.
///
/// Rust-only.
///
/// # Safety
/// The associated scalar must describe the type's in-memory encoding exactly; the library reads
/// raw buffers under that tag.
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

/// A borrowed tape and the batch axis it spans.
#[doc(hidden)]
pub struct Batch {
    pub(crate) tape: uform_tape_view_t,
    pub(crate) count: usize,
}

/// The tape a verb lays separate items into end to end, kept beside the inputs until the call is final.
#[doc(hidden)]
pub struct Gathered<Value> {
    data: Vec<Value>,
    offsets: Vec<usize>,
}

impl<Value> Default for Gathered<Value> {
    fn default() -> Self {
        Gathered {
            data: Vec::new(),
            offsets: Vec::new(),
        }
    }
}

/// A ragged batch a verb reads its items from: separate items, gathered into `gathered`, or a tape
/// of data and offsets, borrowed as it is.
#[doc(hidden)]
pub trait TapeView<Value: Element> {
    /// The tape and its count, refused when the offsets do not fit the data.
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch>;
}

impl<Value: Element, Item: AsRef<[Value]>> TapeView<Value> for [Item] {
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> {
        gathered.data.clear();
        gathered.offsets.clear();
        gathered.offsets.reserve(self.len() + 1);
        gathered.offsets.push(0);
        for item in self {
            gathered.data.extend_from_slice(item.as_ref());
            gathered.offsets.push(gathered.data.len());
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
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> { self.as_slice().tape(gathered) }
}

impl<Value: Element, Item: AsRef<[Value]>, const COUNT: usize> TapeView<Value> for [Item; COUNT] {
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> { self.as_slice().tape(gathered) }
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
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch> { (**self).tape(gathered) }
}

// endregion: Sealed Traits

// region: Errors and Introspection

/// Every value the C status word takes besides success, numbered alike.
///
/// An [`Error`] never carries [`Status::Deferred`], which only [`Deferred::status`] reports.
///
/// Mirrors `uform_status_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The call was submitted deferred and is not final yet.
    Deferred = 1,
    /// Device, pinned, or host memory could not be allocated, or the executor's scratch ceiling is too low.
    OutOfMemory = 2,
    /// Item counts, a pixel shape, or a batch envelope disagree with the call or the model.
    ShapeMismatch = 3,
    /// A stride or packing the model cannot read.
    UnsupportedLayout = 4,
    /// A scalar, modality, or name this build does not know.
    UnknownKind = 5,
    /// A checkpoint file could not be opened, read, or written.
    Io = 7,
    /// The driver refused or the device faulted; after a fault the executor refuses every later call.
    Device = 8,
    /// Supplied bytes could not be decoded: a checkpoint, an image, a clip, or a recording.
    Malformed = 9,
    /// The library reported neither success nor a failure.
    Internal = 10,
    /// The request contradicts a frozen configuration, like resuming a checkpoint of another geometry.
    ConfigMismatch = 13,
    /// The call was refused rather than waited for, like a submission to a full deferred table.
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
        read_spelling(text, uform_status_from_name, uform_status_name, |value| {
            Status::from_raw(value).err()
        })
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
    /// The detail the status alone cannot carry.
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(self.message) }
}

impl std::error::Error for Error {}

/// The result of every fallible call in this crate.
pub type Result<Value> = core::result::Result<Value, Error>;

/// Turns a final outcome into a `Result`.
pub(crate) fn check(outcome: uform_outcome_t) -> Result<()> {
    match Status::from_raw(outcome.status) {
        Ok(()) => Ok(()),
        Err(Status::Deferred) => Err(fail(Status::Internal, "the outcome was read before it was final")),
        Err(status) => Err(Error {
            status,
            message: unsafe { static_string(outcome.message) },
        }),
    }
}

/// Reports a failure the binding caught before the call crossed the boundary, under the same
/// status the library would have used had the ABI let it see the same thing.
pub(crate) fn fail(status: Status, message: &'static str) -> Error { Error { status, message } }

/// Borrows a static null-terminated string the library owns.
///
/// Every such string is ASCII by construction, so its bytes are read as UTF-8 unchecked.
pub(crate) unsafe fn static_string(raw: *const c_char) -> &'static str {
    if raw.is_null() {
        return "";
    }
    core::str::from_utf8_unchecked(CStr::from_ptr(raw).to_bytes())
}

/// Spells an enumerator through its C `uform_*_name`.
pub(crate) fn spelling(name: unsafe extern "C" fn(c_int) -> *const c_char, value: c_int) -> &'static str {
    unsafe { static_string(name(value)) }
}

/// Reads a spelling back through its C `uform_*_from_name`, refusing one the C table spells
/// otherwise, since every such inverse answers a default for a name it does not know.
pub(crate) fn read_spelling<Value>(
    text: &str,
    from_name: unsafe extern "C" fn(uform_bytes_view_t) -> c_int,
    name: unsafe extern "C" fn(c_int) -> *const c_char,
    lift: fn(c_int) -> Option<Value>,
) -> Result<Value> {
    let value = unsafe { from_name(text.as_bytes().into()) };
    match spelling(name, value) == text {
        true => lift(value),
        false => None,
    }
    .ok_or(fail(Status::InvalidArgument, "no enumerator is spelled that way"))
}

/// The version of the linked library.
///
/// Calls `uform_version_name`.
pub fn version() -> &'static str { unsafe { static_string(uform_version_name()) } }

/// Comma-separated hardware capabilities the library was compiled with.
///
/// Calls `uform_capabilities_compiled`.
pub fn capabilities_compiled() -> &'static str { unsafe { static_string(uform_capabilities_compiled()) } }

/// Comma-separated hardware capabilities this machine offers at runtime.
///
/// Calls `uform_capabilities_available`.
pub fn capabilities_available() -> &'static str { unsafe { static_string(uform_capabilities_available()) } }

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
    /// Eight single-bit dimensions per byte, the sign of each embedding dimension - the form USearch's
    /// Hamming and Jaccard metrics score.
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
        read_spelling(text, uform_scalar_from_name, uform_scalar_name, Scalar::from_raw)
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
    /// Haswell-class x86: AVX2 & FMA, the 256-bit floor.
    IntelHaswell = 2,
    /// Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI and its successors without AVX-512.
    IntelSierra = 3,
    /// Skylake-class x86: the AVX-512 foundation tier.
    IntelSkylake = 4,
    /// Icelake-class x86: AVX-512 VNNI & VPOPCNT.
    IntelIcelake = 5,
    /// Genoa-class x86 - Zen 4 and newer: native bf16 dot products over AVX-512.
    AmdGenoa = 6,
    /// Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 & AMX-INT8
    /// tiles.
    IntelSapphire = 7,
    /// Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles.
    IntelGranite = 8,
    /// Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions.
    IntelDiamond = 9,
    /// Graviton2-class Arm - Neoverse N1, the bare NEON floor; phones predating Armv8.6 land here
    /// too.
    AwsG2 = 10,
    /// Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2, Cortex-A710 & X2
    /// class silicon.
    AppleM2 = 11,
    /// Apple M4-class Arm: the Armv8.6 NEON bundle plus the SME outer-product tiles.
    AppleM4 = 12,
    /// Graviton5-class Arm - Neoverse V3, Armv9.2.
    AwsG5 = 13,
    /// Vera-class Arm - NVIDIA's Armv9.2-generation cores with FP8 dot products.
    NvidiaVera = 14,
    /// Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23 vectors.
    AlibabaC930 = 15,
    /// WebAssembly every engine loads: 32-bit, single-threaded, the fixed-width simd128 tier.
    Wasm32V128 = 16,
    /// WebAssembly with relaxed SIMD, shared memory for real thread pools, and 64-bit
    /// addressing.
    Wasm64Extended = 17,
}

impl CPUClass {
    pub(crate) fn from_raw(value: uform_cpu_class_t) -> Option<Self> {
        let cpu_class = match value {
            1 => CPUClass::Serial,
            2 => CPUClass::IntelHaswell,
            3 => CPUClass::IntelSierra,
            4 => CPUClass::IntelSkylake,
            5 => CPUClass::IntelIcelake,
            6 => CPUClass::AmdGenoa,
            7 => CPUClass::IntelSapphire,
            8 => CPUClass::IntelGranite,
            9 => CPUClass::IntelDiamond,
            10 => CPUClass::AwsG2,
            11 => CPUClass::AppleM2,
            12 => CPUClass::AppleM4,
            13 => CPUClass::AwsG5,
            14 => CPUClass::NvidiaVera,
            15 => CPUClass::AlibabaC930,
            16 => CPUClass::Wasm32V128,
            17 => CPUClass::Wasm64Extended,
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
        read_spelling(text, uform_cpu_class_from_name, uform_cpu_class_name, CPUClass::from_raw)
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
    /// Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30.
    NvidiaAmpere = 1,
    /// Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions.
    NvidiaAda = 2,
    /// Hopper-class NVIDIA - H100 & H200.
    NvidiaHopper = 3,
    /// Blackwell-class NVIDIA - B200, B300 & GB300.
    NvidiaBlackwell = 4,
    /// Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark.
    NvidiaBlackwellRtx = 5,
    /// Rubin-class NVIDIA - the Vera-Rubin platform's device half.
    NvidiaRubin = 6,
    /// CDNA3 AMD - MI300X & MI325X: gfx942, FNUZ fp8.
    AmdCdna3 = 7,
    /// CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8.
    AmdCdna4 = 8,
    /// CDNA5 AMD - MI400: gfx1250.
    AmdCdna5 = 9,
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
            1 => GPUClass::NvidiaAmpere,
            2 => GPUClass::NvidiaAda,
            3 => GPUClass::NvidiaHopper,
            4 => GPUClass::NvidiaBlackwell,
            5 => GPUClass::NvidiaBlackwellRtx,
            6 => GPUClass::NvidiaRubin,
            7 => GPUClass::AmdCdna3,
            8 => GPUClass::AmdCdna4,
            9 => GPUClass::AmdCdna5,
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
        read_spelling(text, uform_gpu_class_from_name, uform_gpu_class_name, GPUClass::from_raw)
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
    pub fn vendor(&self) -> Option<GPUVendor> { GPUVendor::from_raw(self.0.vendor) }

    /// The device's ordinal in its vendor's runtime, as `nvidia-smi` or `rocm-smi` list it.
    pub fn vendor_ordinal(&self) -> u32 { self.0.vendor_ordinal }

    /// The class whose kernels run on it.
    pub fn gpu_class(&self) -> Option<GPUClass> { GPUClass::from_raw(self.0.gpu_class) }

    /// Streaming multiprocessors or compute units.
    pub fn compute_units(&self) -> u32 { self.0.compute_units }

    /// Device memory in bytes.
    pub fn memory_bytes(&self) -> u64 { self.0.memory_bytes }

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

/// Lists every GPU this process can drive into `devices`, NVIDIA devices first, then AMD, and
/// answers how many there are, which may exceed what `devices` holds.
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

/// Borrows `length` values a finished call wrote into a host arena; a call that wrote none may
/// leave a null base, which an empty run never reads.
fn exported_slice<'arena, Value>(data: *const Value, length: usize) -> Result<&'arena [Value]> {
    match (data.is_null(), length) {
        (_, 0) => Ok(&[]),
        (true, _) => Err(fail(Status::Internal, "the library exported rows without a base")),
        // Safety: the arena borrowed beside the view holds a finished call's output in host memory.
        (false, length) => Ok(unsafe { core::slice::from_raw_parts(data, length) }),
    }
}

/// Rows a call wrote into the arena it borrows, one per item, packed.
///
/// Mirrors `uform_rows_view_t`.
pub struct Rows<'arena> {
    pub(crate) arena: &'arena Arena,
    pub(crate) view: Exported<uform_rows_view_t>,
    pub(crate) count: usize,
}

impl<'arena> Rows<'arena> {
    /// Rows the call wrote.
    pub fn count(&self) -> usize { self.count }

    /// Values per row, the model's embedding dimensions.
    pub fn columns(&self) -> usize { self.view.0.columns }

    /// Bytes between the starts of consecutive rows.
    pub fn stride(&self) -> usize { self.view.0.stride }

    /// The scalar type the rows hold.
    pub fn scalar(&self) -> Option<Scalar> { Scalar::from_raw(self.view.0.scalar) }

    /// The first row, in host or device memory, for handing to another library as it is.
    pub fn as_ptr(&self) -> *const c_void { self.view.0.data }

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
}

/// Rows a call wrote into the arena it borrows, as many per item as the item had tokens left
/// after truncation, end to end, with one more offset than items.
///
/// Mirrors `uform_rows_tape_view_t`.
pub struct RowsTape<'arena> {
    pub(crate) arena: &'arena Arena,
    pub(crate) view: Exported<uform_rows_tape_view_t>,
    pub(crate) count: usize,
}

impl<'arena> RowsTape<'arena> {
    /// Items the call encoded.
    pub fn count(&self) -> usize { self.count }

    /// The call's count plus one non-decreasing offsets, in rows, refused with [`Status::Device`]
    /// over device memory.
    pub fn offsets(&self) -> Result<&'arena [usize]> {
        self.arena.host()?;
        exported_slice(self.view.0.offsets, self.count + 1)
    }

    /// Every item's rows end to end, refused with [`Status::Device`] over device memory.
    pub fn rows(&self) -> Result<Rows<'arena>> {
        let offsets = self.offsets()?;
        Ok(Rows {
            arena: self.arena,
            view: Exported(self.view.0.rows),
            count: offsets.last().copied().unwrap_or(0),
        })
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

/// Values a call wrote into the arena it borrows, a ragged run per item, end to end, with one more
/// offset than items.
///
/// Mirrors `uform_tape_view_t`.
pub struct Tape<'arena> {
    pub(crate) arena: &'arena Arena,
    pub(crate) view: Exported<uform_tape_view_t>,
    pub(crate) count: usize,
}

impl<'arena> Tape<'arena> {
    /// Items the call wrote.
    pub fn count(&self) -> usize { self.count }

    /// The scalar type the values are.
    pub fn scalar(&self) -> Option<Scalar> { Scalar::from_raw(self.view.0.scalar) }

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

    /// The values of one item, refused with [`Status::InvalidArgument`] when `item` is not below
    /// [`Tape::count`] or the tape holds another scalar type, and with [`Status::Device`] over
    /// device memory.
    pub fn item<Value: Element>(&self, item: usize) -> Result<&'arena [Value]> {
        let offsets = self.offsets()?;
        let (Some(&start), Some(&end)) = (offsets.get(item), offsets.get(item + 1)) else {
            return Err(fail(Status::InvalidArgument, "the item is not below the call's count"));
        };
        self.values()?
            .get(start..end)
            .ok_or(fail(Status::Internal, "the library exported offsets past its values"))
    }
}

// endregion: Rows

// region: Placements

/// The memory domains a RAM placement or executor names, and the class it pins; `None` derives the
/// CPU's own.
pub(crate) fn ram_options(memory_domains: &[i32], cpu_class: Option<CPUClass>) -> uform_ram_options_t {
    uform_ram_options_t {
        memory_domains: memory_domains.as_ptr(),
        memory_domains_count: memory_domains.len(),
        cpu_class: cpu_class.map_or(0, |cpu_class| cpu_class as uform_cpu_class_t),
    }
}

/// The devices a VRAM placement or executor names, and the class it pins; `None` derives the
/// devices' own.
pub(crate) fn vram_options(devices: &[i32], gpu_class: Option<GPUClass>) -> uform_vram_options_t {
    uform_vram_options_t {
        devices: devices.as_ptr(),
        devices_count: devices.len(),
        gpu_class: gpu_class.map_or(0, |gpu_class| gpu_class as uform_gpu_class_t),
    }
}

/// Borrows a path as the bytes the C interface exchanges: the OS's own bytes on Unix, not necessarily
/// UTF-8, and UTF-8 where the OS keeps paths in another encoding.
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

/// The handlers one call lends the library, reached by every trampoline through one state pointer,
/// and the first panic any of them raised, re-raised once the call is final since unwinding out of an
/// `extern "C"` frame aborts.
#[doc(hidden)]
pub struct Callbacks {
    pub(crate) progress: Cell<*mut c_void>,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Default for Callbacks {
    fn default() -> Self {
        Callbacks {
            progress: Cell::new(ptr::null_mut()),
            panic: Mutex::new(None),
        }
    }
}

impl Callbacks {
    /// The state every trampoline of this call receives.
    pub(crate) fn state(&self) -> *mut c_void { ptr::from_ref(self).cast_mut().cast() }

    /// Runs a handler, answering `fallback` in place of a panic and latching the first one.
    pub(crate) fn guard<Answer>(&self, fallback: Answer, handler: impl FnOnce() -> Answer) -> Answer {
        std::panic::catch_unwind(AssertUnwindSafe(handler)).unwrap_or_else(|panic| {
            self.panic
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(panic);
            fallback
        })
    }

    /// Re-raises the first panic a handler raised; called once the call is final.
    pub(crate) fn raise(self) {
        if let Some(panic) = self.panic.into_inner().unwrap_or_else(PoisonError::into_inner) {
            std::panic::resume_unwind(panic)
        }
    }
}

/// A periodic report of finished and total units; `Break` stops the call at the next slice.
///
/// It runs on whichever thread advanced the work, never on two at once, so it owes `Send` and may
/// be `FnMut`. A panic inside it stops the call like a `Break` and is re-raised once the call is final.
///
/// Mirrors `uform_progress_callback_t`.
pub type ProgressCallback<'a> = dyn FnMut(usize, usize) -> ControlFlow<()> + Send + 'a;

/// # Safety
/// `state` must address the live [`Callbacks`] of the call the callback was passed to, lending a
/// `&mut ProgressCallback`.
unsafe extern "C" fn progress_trampoline(completed: usize, total: usize, state: *mut c_void) -> c_int {
    let callbacks = &*state.cast::<Callbacks>();
    let handler = &mut **callbacks.progress.get().cast::<&mut ProgressCallback>();
    callbacks.guard(1, || c_int::from(handler(completed, total).is_break()))
}

/// Lends an optional handler and its step count as the progress every reporting verb takes, so the
/// trampoline is never paired with the wrong state; `None` lends nulls.
pub(crate) fn lower_progress(
    handler: &mut Option<&mut ProgressCallback>,
    steps: usize,
    callbacks: &Callbacks,
) -> uform_progress_t {
    match handler {
        Some(handler) => {
            callbacks.progress.set(ptr::from_mut(handler).cast());
            uform_progress_t {
                callback: Some(progress_trampoline),
                state: callbacks.state(),
                steps,
            }
        }
        None => uform_progress_t {
            callback: None,
            state: ptr::null_mut(),
            steps,
        },
    }
}

// endregion: Callbacks

// region: Executors

/// What an executor does with its threads or streams once it goes idle.
///
/// Mirrors `uform_mood_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Mood {
    /// Sleep on futexes or events, freeing the core; the next call pays a wake-up.
    #[default]
    Parked = 0,
    /// Poll them, trading a core for the lowest wake-up latency.
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
        read_spelling(text, uform_mood_from_name, uform_mood_name, Mood::from_raw)
    }
}

/// Where an executor computes, and with it where the arenas it makes keep their memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Residency {
    Ram,
    Vram,
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
/// Calls on one executor run in submission order, so threads wanting to overlap take one apiece;
/// it is `Send` but not `Sync`.
/// Every constructor takes a `deferred_limit` sizing its table of deferred calls, where zero takes
/// the default; a deferred submission finding the table full fails with [`Status::Contended`].
/// [`Arena::new_in`] places an arena where it computes: a host arena may outlive it, while a GPU
/// arena holds it until the arena drops.
///
/// Mirrors `uform_executor_t`; dropping its last holder calls `uform_executor_free`.
#[derive(Debug)]
pub struct Executor {
    pub(crate) handle: uform_executor_t,
    /// The residency of the constructor that produced it, which places the arenas it makes.
    residency: Residency,
    /// Frees `handle` once neither this nor a GPU arena it made holds it.
    share: Arc<ExecutorShare>,
}

// Safety: nothing in an executor is tied to the thread that created it, so it may move to another;
// staying out of `Sync` keeps submissions to it on one thread at a time.
unsafe impl Send for Executor {}

impl Default for Executor {
    fn default() -> Self { Executor::adopt(ptr::null_mut(), Residency::Ram) }
}

impl Executor {
    /// Owns the handle a constructor filled, which is null for the executor that owns nothing.
    fn adopt(handle: uform_executor_t, residency: Residency) -> Self {
        Executor {
            handle,
            residency,
            share: Arc::new(ExecutorShare(handle)),
        }
    }

    /// Spawns host threads for RAM-resident models, interleaved across the compute domains local to
    /// `memory_domains`, which is the same list the models were loaded with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, and `cpu_class` pins the class
    /// its calls dispatch as, where `None` derives the CPU's own.
    ///
    /// Calls `uform_executor_for_ram`.
    pub fn for_ram(
        memory_domains: &[i32],
        cpu_class: Option<CPUClass>,
        threads: usize,
        deferred_limit: usize,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_executor_for_ram(&mut uform_executor_for_ram_t {
                executor: &mut handle,
                outcome: &mut outcome,
                ram: ram_options(memory_domains, cpu_class),
                threads,
                deferred_limit,
            })
        };
        let executor = Executor::adopt(handle, Residency::Ram);
        check(outcome).map(|()| executor)
    }

    /// Creates a RAM executor over a borrowed ForkUnion pool, so UForm, USearch and UStore in one
    /// process share one set of threads while each keeps its own scratch.
    ///
    /// Calls `uform_executor_on_pool`.
    ///
    /// # Safety
    /// `pool` must address a live ForkUnion `fu_pool_t` that outlives the executor.
    pub unsafe fn on_pool(
        memory_domains: &[i32],
        cpu_class: Option<CPUClass>,
        pool: *mut c_void,
        deferred_limit: usize,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let mut outcome = uform_outcome_t::default();
        uform_executor_on_pool(&mut uform_executor_on_pool_t {
            executor: &mut handle,
            outcome: &mut outcome,
            ram: ram_options(memory_domains, cpu_class),
            pool,
            deferred_limit,
        });
        let executor = Executor::adopt(handle, Residency::Ram);
        check(outcome).map(|()| executor)
    }

    /// Opens one submission stream on every named device, or borrows the caller's, all of them
    /// driven by whichever host thread submits - which is why this takes no thread count.
    ///
    /// `devices` are positions in what [`gpu_devices`] lists, and `gpu_class` pins the class every
    /// one of them dispatches as, where `None` derives theirs. `streams` is empty, or holds one
    /// CUDA stream per device in `devices` order. Zero `scratch_limit` lets scratch grow as batches
    /// ask.
    ///
    /// Calls `uform_executor_for_vram`.
    ///
    /// # Safety
    /// Every stream must be a live CUDA stream on its device that outlives the executor.
    pub unsafe fn for_vram(
        devices: &[i32],
        gpu_class: Option<GPUClass>,
        streams: &[*mut c_void],
        scratch_limit: usize,
        deferred_limit: usize,
    ) -> Result<Self> {
        if !streams.is_empty() && streams.len() != devices.len() {
            return Err(fail(Status::ShapeMismatch, "pass no streams or one per device"));
        }
        let mut handle = ptr::null_mut();
        let mut outcome = uform_outcome_t::default();
        uform_executor_for_vram(&mut uform_executor_for_vram_t {
            executor: &mut handle,
            outcome: &mut outcome,
            vram: vram_options(devices, gpu_class),
            // The C side reads one stream per device from any non-null pointer.
            streams: match streams.is_empty() {
                true => ptr::null(),
                false => streams.as_ptr(),
            },
            scratch_limit,
            deferred_limit,
        });
        let executor = Executor::adopt(handle, Residency::Vram);
        check(outcome).map(|()| executor)
    }

    /// Waits for every call submitted to this executor before the drain began, then adopts `mood`
    /// whenever it next goes idle; a failure is the executor's own, such as a device fault, never
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

/// The executor every holder shares, freed by the last of them to drop.
#[derive(Debug)]
struct ExecutorShare(uform_executor_t);

// Safety: an executor may be freed from any thread, and a share is only ever dropped, never read.
unsafe impl Send for ExecutorShare {}
unsafe impl Sync for ExecutorShare {}

impl Drop for ExecutorShare {
    fn drop(&mut self) { unsafe { uform_executor_free(self.0) } }
}

// endregion: Executors

// region: Arenas

/// Memory the library writes every output whose length scales with a call into, reused across calls.
///
/// An executor places it where it computes, host pages or its GPU's memory, decided once, and
/// nothing is allocated until a call writes into it. Every call rewinds and refills it,
/// invalidating the previous call's outputs, so a steady loop allocates once.
/// Every verb, blocking or deferred, borrows it as `&mut Arena` and answers outputs that borrow it in turn.
/// An arena from a VRAM executor holds device memory, so the host accessors of its outputs refuse
/// with [`Status::Device`]. It grows, fills and frees on the first stream of the executor that made
/// it, so it serves only calls on that executor and holds the executor until it drops.
///
/// Mirrors `uform_arena_t`; dropping it calls `uform_arena_free`, which waits for the calls
/// still writing into it.
#[derive(Debug)]
pub struct Arena {
    pub(crate) handle: uform_arena_t,
    /// The residency of the executor that made it.
    residency: Residency,
    /// The executor a GPU arena frees on, dropped after the arena; `None` for host pages.
    _executor: Option<Arc<ExecutorShare>>,
}

// Safety: an arena serves one call at a time, which the `&mut` every verb takes already enforces,
// and a shared borrow reads only its residency.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Makes an arena of host pages for calls on the calling thread.
    ///
    /// Calls `uform_arena_init`.
    pub fn new() -> Result<Self> { Arena::new_in(&Executor::default()) }

    /// Makes an arena where `executor` computes, serving the calls of its residency, or only those
    /// on `executor` itself for a GPU arena.
    ///
    /// Calls `uform_arena_init`.
    pub fn new_in(executor: &Executor) -> Result<Self> {
        let mut arena = Arena {
            handle: ptr::null_mut(),
            residency: executor.residency,
            _executor: match executor.residency {
                Residency::Vram => Some(Arc::clone(&executor.share)),
                Residency::Ram => None,
            },
        };
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_arena_init(&mut uform_arena_init_t {
                arena: &mut arena.handle,
                executor: executor.handle,
                outcome: &mut outcome,
            })
        };
        check(outcome).map(|()| arena)
    }

    /// Refuses device memory, which no host accessor may read.
    pub(crate) fn host(&self) -> Result<()> {
        match self.residency {
            Residency::Vram => Err(fail(Status::Device, "the outputs are device memory")),
            Residency::Ram => Ok(()),
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) { unsafe { uform_arena_free(self.handle) } }
}

/// A view the library wrote into an arena the verb or result borrows.
#[derive(Default)]
pub(crate) struct Exported<View>(pub(crate) View);

// Safety: the pointees live in the arena borrowed beside the view, and its borrows are `Send`.
unsafe impl<View> Send for Exported<View> {}

// endregion: Arenas

// region: Receivers

/// A model or a trainer borrowed with the executor its calls run on and how they complete.
///
/// Neither the receiver nor a [`Deferred`] it submits outlives the handle or the executor.
///
/// Rust-only.
#[derive(Debug)]
pub struct On<'handle, Handle, Completion = Blocking> {
    pub(crate) handle: &'handle Handle,
    pub(crate) executor: &'handle Executor,
    pub(crate) completion: PhantomData<Completion>,
}

impl<Handle, Completion> Clone for On<'_, Handle, Completion> {
    fn clone(&self) -> Self { *self }
}

impl<Handle, Completion> Copy for On<'_, Handle, Completion> {}

/// The completion of a blocking receiver, whose verbs return once their call is final and answer
/// what it produced.
///
/// Rust-only.
#[derive(Debug)]
pub struct Blocking;

impl Completion for Blocking {
    type Answer<'handle, Verb: Calling> = Result<Verb::Output>;

    fn answer<'handle, Verb: Calling>(mut verb: Verb) -> Self::Answer<'handle, Verb> {
        let mut outcome = uform_outcome_t::default();
        let callbacks = Callbacks::default();
        let mut arguments = verb.arguments(&mut outcome, uform_completion_blocking_k, &callbacks)?;
        // Safety: the arguments point into `verb`, `outcome` and `callbacks`, which stay in place until
        // the call returns, and a blocking call is final on return.
        unsafe { Verb::call(&mut arguments) };
        callbacks.raise();
        check(outcome)?;
        verb.finish()
    }
}

impl<Input> Accepts<Input> for Blocking {}

/// The completion of a deferring receiver, whose verbs return once their call is submitted and
/// answer a [`Deferred`].
///
/// Rust-only.
#[derive(Debug)]
pub struct Deferring;

impl Completion for Deferring {
    type Answer<'handle, Verb: Calling> = Result<Deferred<'handle, Verb>>;

    fn answer<'handle, Verb: Calling>(verb: Verb) -> Self::Answer<'handle, Verb> {
        let lent = Box::into_raw(Box::new(Lent {
            outcome: uform_outcome_t::default(),
            callbacks: Callbacks::default(),
            arguments: None,
            verb,
        }));
        // Safety: the box is fresh and reached through `lent` alone, and the arguments point into its
        // fields, where they stay until the box is freed.
        unsafe {
            let outcome = ptr::addr_of_mut!((*lent).outcome);
            let (callbacks, verb) = (&(*lent).callbacks, &mut (*lent).verb);
            match verb.arguments(outcome, uform_completion_deferred_k, callbacks) {
                Ok(arguments) => Verb::call((*lent).arguments.insert(arguments)),
                Err(refusal) => {
                    drop(Box::from_raw(lent));
                    return Err(refusal);
                }
            }
            Ok(Deferred {
                lent: NonNull::new_unchecked(lent),
                receiver: PhantomData,
            })
        }
    }
}

impl<Item: 'static> Accepts<Vec<Item>> for Deferring {}
impl<Item: 'static, const COUNT: usize> Accepts<[Item; COUNT]> for Deferring {}
impl<Data: 'static, Offsets: 'static> Accepts<(Data, Offsets)> for Deferring {}
impl Accepts<PathBuf> for Deferring {}
impl Accepts<String> for Deferring {}
impl Accepts<&'static str> for Deferring {}
impl Accepts<&'static Path> for Deferring {}

/// One deferred call and everything it lends the library, at one heap address until it is final.
struct Lent<Verb: Calling> {
    /// The outcome the library finishes.
    outcome: uform_outcome_t,
    /// The handlers the call lends, and the first panic they raised.
    callbacks: Callbacks,
    /// The argument struct, pointing into `outcome` and `verb`, filled once the box is in place.
    arguments: Option<Verb::Arguments>,
    /// The inputs, the outputs and the arena borrow.
    verb: Verb,
}

/// A submitted call, which owns its outcome, argument struct and inputs in one box the library
/// reads until the call is final, and borrows its arena as a blocking call does.
///
/// [`Deferred::status`] polls without blocking, and [`Deferred::wait`] blocks until the call is
/// final and hands back what it produced, while `.await` polls it from async code. Dropping an
/// unfinished one waits for it, then frees the box.
/// A panic a handler raised resurfaces from `wait`, `.await`, or a drop outside unwinding.
/// `'handle` is the borrow of the [`On`] that submitted it.
///
/// Rust-only; waiting and dropping call `uform_outcome_wait`.
#[must_use = "a deferred call's outcome is seen only through `status` or `wait`"]
pub struct Deferred<'handle, Verb: Calling> {
    lent: NonNull<Lent<Verb>>,
    receiver: PhantomData<&'handle ()>,
}

impl<Verb: Calling> Deferred<'_, Verb> {
    /// `Err(Status::Deferred)` until the call is final, then its final status, read without blocking.
    ///
    /// Calls `uform_outcome_status`.
    pub fn status(&self) -> core::result::Result<(), Status> {
        // Safety: the outcome lives in the box this `Deferred` owns until it is final.
        Status::from_raw(unsafe { uform_outcome_status(ptr::addr_of_mut!((*self.lent.as_ptr()).outcome)) })
    }

    /// Blocks until the call is final, then hands back what it produced.
    ///
    /// Calls `uform_outcome_wait`.
    pub fn wait(self) -> Result<Verb::Output> {
        let lent = ManuallyDrop::new(self).lent.as_ptr();
        // Safety: once the outcome is final the library touches the box no more, so it is freed here.
        let Lent {
            outcome,
            callbacks,
            verb,
            ..
        } = unsafe {
            uform_outcome_wait(ptr::addr_of_mut!((*lent).outcome));
            *Box::from_raw(lent)
        };
        callbacks.raise();
        check(outcome)?;
        verb.finish()
    }
}

impl<Verb: Calling> Drop for Deferred<'_, Verb> {
    fn drop(&mut self) {
        let lent = self.lent.as_ptr();
        // Safety: once the outcome is final the library touches the box no more, and this owned it.
        let Lent { callbacks, .. } = unsafe {
            uform_outcome_wait(ptr::addr_of_mut!((*lent).outcome));
            *Box::from_raw(lent)
        };
        // A second panic while unwinding would abort, so only a drop outside one re-raises.
        if !std::thread::panicking() {
            callbacks.raise();
        }
    }
}

impl<Verb: Calling> fmt::Debug for Deferred<'_, Verb> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Deferred({:?})", self.status())
    }
}

// Safety: the box moves with the `Deferred`, and until its call is final only the library reaches it.
unsafe impl<Verb: Calling + Send> Send for Deferred<'_, Verb> {}

impl<'handle, Verb: Calling> IntoFuture for Deferred<'handle, Verb> {
    type Output = Result<Verb::Output>;
    type IntoFuture = Wait<'handle, Verb>;

    fn into_future(self) -> Wait<'handle, Verb> { Wait(Some(self)) }
}

/// A [`Deferred`] being awaited: every poll reads its status, and one not yet final asks to be polled again.
///
/// The C library has no wake-up, so an executor re-polls it while other tasks run, spinning only
/// when nothing else is ready; [`Deferred::wait`] blocks without spinning.
///
/// Rust-only; the final poll calls `uform_outcome_wait`, which returns at once.
#[must_use = "an awaited call does nothing unless polled"]
pub struct Wait<'handle, Verb: Calling>(Option<Deferred<'handle, Verb>>);

impl<Verb: Calling> Future for Wait<'_, Verb> {
    type Output = Result<Verb::Output>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let awaited = &mut self.get_mut().0;
        match awaited.take() {
            None => Poll::Ready(Err(fail(Status::InvalidArgument, "the call was already awaited"))),
            Some(call) if call.status() == Err(Status::Deferred) => {
                *awaited = Some(call);
                context.waker().wake_by_ref();
                Poll::Pending
            }
            Some(call) => Poll::Ready(call.wait()),
        }
    }
}

// endregion: Receivers

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Every C inverse answers a default for a name it does not know, which must not read back as
    /// that default.
    #[test]
    fn names_round_trip_through_the_library_and_refuse_strangers() {
        assert_eq!(Status::ShapeMismatch.to_string(), "shape-mismatch");
        assert_eq!("cancelled".parse(), Ok(Status::Cancelled));
        assert_eq!("invalid-argument".parse(), Ok(Status::InvalidArgument));
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
            Status::ConfigMismatch,
            Status::Contended,
            Status::Cancelled,
            Status::Unsupported,
            Status::InvalidArgument,
        ];
        for status in failures {
            let raw = uform_outcome_t {
                status: status as i32,
                message: ptr::null(),
            };
            assert_eq!(check(raw).unwrap_err().status, status);
            assert_eq!(status.to_string().parse(), Ok(status));
        }
        // A final outcome is never deferred, and a status this build has no name for is not success
        // either, so both read as the contract violation they are; USearch's 6, 11 and 12 stay unused.
        for status in [Status::Deferred as i32, 6, 11, 12, 4242] {
            let raw = uform_outcome_t {
                status,
                message: ptr::null(),
            };
            assert_eq!(check(raw).unwrap_err().status, Status::Internal);
        }
    }

    #[test]
    fn separate_items_gather_into_one_tape() -> Result<()> {
        let mut gathered = Gathered::default();
        let batch = TapeView::<u8>::tape(&["", "uform", "ab"], &mut gathered)?;
        assert_eq!(batch.count, 3);
        assert_eq!(gathered.offsets, [0, 0, 5, 7]);
        assert_eq!(gathered.data, b"uformab");
        assert_eq!(batch.tape.scalar, Scalar::U8 as i32);

        let batch = TapeView::<f32>::tape(&vec![vec![0.5f32; 3], vec![]], &mut Gathered::default())?;
        assert_eq!((batch.count, batch.tape.scalar), (2, Scalar::F32 as i32));
        Ok(())
    }

    /// A borrowed tape reaches the library as it is, so offsets that would read past its data are
    /// refused before the call.
    #[test]
    fn borrowed_tapes_are_checked_before_the_call() {
        let mut gathered = Gathered::default();
        let data = b"firstsecond".as_slice();
        let batch = TapeView::<u8>::tape(&(data, [0usize, 5, 11]), &mut gathered).unwrap();
        assert_eq!(batch.count, 2);
        assert_eq!(batch.tape.data, data.as_ptr().cast());
        assert!(gathered.data.is_empty(), "a borrowed tape gathers nothing");

        let refusals = [
            ([0usize, 5, 12].as_slice(), Status::ShapeMismatch),
            ([0usize, 6, 5].as_slice(), Status::Malformed),
            ([].as_slice(), Status::InvalidArgument),
        ];
        for (offsets, status) in refusals {
            let refusal = TapeView::<u8>::tape(&(data, offsets), &mut gathered)
                .err()
                .map(|error| error.status);
            assert_eq!(refusal, Some(status), "{offsets:?}");
        }
    }

    /// An arena keeps the residency of the executor that made it, and the host reads only host pages.
    #[test]
    fn arenas_live_where_their_executor_computes() -> Result<()> {
        assert!(Arena::new()?.host().is_ok());

        let executor = Executor::for_ram(&[], None, 1, 0)?;
        let placed = Arena::new_in(&executor)?;
        drop(executor);
        assert!(placed.host().is_ok(), "an arena outlives its executor");

        // A GPU arena frees on its executor's first stream, so it keeps the executor past its drop.
        let gpu = Executor::adopt(ptr::null_mut(), Residency::Vram);
        let device = Arena::new_in(&gpu)?;
        let share = Arc::downgrade(&gpu.share);
        drop(gpu);
        assert!(share.upgrade().is_some(), "a GPU arena holds its executor");
        assert_eq!(device.host().unwrap_err().status, Status::Device);
        drop(device);
        assert!(share.upgrade().is_none(), "the last GPU arena frees its executor");
        Ok(())
    }

    #[test]
    fn executors_report_what_this_build_lacks() {
        let pooled = unsafe { Executor::on_pool(&[], None, ptr::null_mut(), 0) };
        assert_eq!(pooled.unwrap_err().status, Status::Unsupported);
        let mismatched = unsafe { Executor::for_vram(&[0, 1], None, &[ptr::null_mut()], 0, 0) };
        assert_eq!(mismatched.unwrap_err().status, Status::ShapeMismatch);
    }

    #[test]
    fn classes_round_trip_through_the_library() {
        assert_eq!(CPUClass::IntelHaswell.to_string(), "intel-haswell");
        assert_eq!("wasm64-extended".parse(), Ok(CPUClass::Wasm64Extended));
        assert_eq!(GPUClass::NvidiaHopper.to_string(), "nvidia-hopper");
        assert_eq!("apple-metal10".parse(), Ok(GPUClass::AppleMetal10));
        assert!("unknown".parse::<CPUClass>().is_err());
        assert!("intel".parse::<GPUClass>().is_err());
        assert_eq!(CPUClass::from_raw(0), None);
        assert_eq!(GPUClass::from_raw(13), None);
    }

    /// A machine without GPUs, or a build without their vendors, lists none and still succeeds.
    #[test]
    fn listed_devices_never_outnumber_their_count() -> Result<()> {
        assert_eq!(core::mem::offset_of!(uform_gpu_device_t, name), 24);
        assert_eq!(core::mem::size_of::<GPUDevice>(), 88);
        let mut devices = [GPUDevice::default(); 4];
        let count = gpu_devices(&mut devices)?;
        assert_eq!(gpu_devices(&mut [])?, count);
        for device in &devices[..count.min(devices.len())] {
            assert!(device.vendor().is_some() && !device.name().is_empty(), "{device:?}");
        }
        Ok(())
    }

    #[test]
    fn introspection_reads_static_strings() {
        assert!(!capabilities_compiled().is_empty());
        assert!(!capabilities_available().is_empty());
    }
}

// endregion: Tests
