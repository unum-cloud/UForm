"""The vocabulary every verb shares: statuses, scalars, views, executors, arenas, and the one error type.

`Error` is the only error type the binding raises: Mojo allows at most one per function, so a second
would force every caller raising `Error` to catch and convert. A refusal is a failure, not an
absence, so a call the engine declines raises.

`MutAnyOrigin` rather than `MutUntrackedOrigin` is load-bearing on anything C writes through: with
a narrower origin the compiler folds a read of the slot back to whatever the caller last stored
there, and every out-parameter reads back as zero.

A struct whose address crosses to C is materialized into a `stack_allocation` slot rather than held
in a local whose address is taken. An `alloca` is not a value Mojo owns, so nothing destroys it and
nothing folds its store away, where a local's last tracked use is the address-taking itself. A
deferred call's slots outlive the verb that submitted them, so they live on the heap instead.

File: mojo/uform/types.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from std.ffi import c_char, c_int, c_size_t
from std.memory import ArcPointer, OwnedPointer, stack_allocation
from std.sys.info import size_of
from std.utils import StaticTuple

from uform.library import Library

# region Pointers

comptime Address = Int
"""An opaque C handle or untyped pointer crossing as its numeric value, which is how a null one is
expressed. The integer is pointer-width and shares the pointer's argument class on both x86-64 and
AArch64, so it lands in the same register a pointer would."""

comptime NULL_ADDRESS = 0
"""The null pointer, as `Address` spells it."""

comptime CString = Pointer[c_char, ImmUntrackedOrigin]
"""A static null-terminated C string the engine owns, as the name tables and messages hand them out."""


def _anchor[origin: Origin]() -> ref[origin] NoneType:
    """A reference of the given origin, read only inside `origin_of` to union parameter origins."""
    return Pointer[NoneType, origin](unsafe_from_address=NULL_ADDRESS)[]


# endregion Pointers

# region Errors


@fieldwise_init
struct Status(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Why a call into the engine failed, the `uform_status_t` vocabulary verbatim.

    An `Error` never carries `SUCCESS` or `DEFERRED`: only a `Deferred` reports the latter, and only
    until its call is final. `Library.status_name` spells each one as the engine does.
    """

    var code: Int32
    """The raw `uform_status_t` value, kept even when this build does not name it."""
    comptime SUCCESS = Self(0)
    """The call completed."""
    comptime DEFERRED = Self(1)
    """The call was submitted deferred and is not final yet; the one non-final status."""
    comptime OUT_OF_MEMORY = Self(2)
    """Device, pinned or host memory could not be allocated, or the scratch ceiling is too low."""
    comptime SHAPE_MISMATCH = Self(3)
    """Item counts, a pixel shape, or a batch envelope disagree with the call or the model."""
    comptime UNSUPPORTED_LAYOUT = Self(4)
    """A stride or packing the model cannot read."""
    comptime UNKNOWN_KIND = Self(5)
    """A scalar, modality or name this build does not know."""
    comptime IO = Self(7)
    """A checkpoint file could not be opened, read or written."""
    comptime DEVICE = Self(8)
    """The driver refused or the device faulted; after a fault the executor refuses every later call."""
    comptime MALFORMED = Self(9)
    """Supplied bytes could not be decoded: a checkpoint, an image, a clip or a recording."""
    comptime INTERNAL = Self(10)
    """The library reported neither success nor a failure, which names a defect rather than a state."""
    comptime CONFIG_MISMATCH = Self(13)
    """The request contradicts a frozen configuration, like resuming a checkpoint of another geometry."""
    comptime CONTENDED = Self(14)
    """The call was refused rather than waited for: another call holds its executor or its arena."""
    comptime CANCELLED = Self(15)
    """A progress handler asked the call to stop; it unwound and published nothing."""
    comptime UNSUPPORTED = Self(16)
    """The verb is declared for every residency or model, but this one does not implement it."""
    comptime INVALID_ARGUMENT = Self(17)
    """An argument breaks the verb's contract, like a read of a result whose arena a later call rewound."""
    comptime LIBRARY_MISSING = Self(-1)
    """Nothing to `dlopen`, the binding's own."""
    comptime SYMBOL_MISSING = Self(-2)
    """The library loaded but does not export a symbol the binding needs, the binding's own."""
    comptime VERSION_MISMATCH = Self(-3)
    """The library loaded but reports a different major version than the binding's, the binding's own."""

    @staticmethod
    def of(code: c_int) -> Self:
        """Lifts a raw `uform_status_t`, keeping an unnamed one rather than guessing."""
        return Self(Int32(code))


@fieldwise_init
struct Outcome(ImplicitlyCopyable):
    """The `uform_outcome_t` every verb reports through, laid out as the ABI declares it."""

    var status: c_int
    """The `uform_status_t` the call reported."""
    var message: Optional[CString]
    """The engine's own static message, absent when the call succeeded."""

    def __init__(out self):
        """Zeroes the slot, so a call that writes nothing still reads as a success."""
        self.status = c_int(0)
        self.message = None


@fieldwise_init
struct Error(Copyable, ImplicitlyCopyable, Writable):
    """What went wrong inside the engine, and which entry point it was about."""

    var status: Status
    """Which class of failure the engine reported."""
    var detail: StaticString
    """The entry point it was about, for a reader rather than for a branch."""
    var message: Optional[CString]
    """The engine's own static message, which outlives the call and needs no freeing."""
    var name: Optional[CString]
    """The engine's spelling of `status`, absent where no engine named it."""
    var library: Optional[Library]
    """The engine `message` and `name` live in, kept mapped for as long as the error is; Mojo would
    otherwise unload it at the last handle's last use, while the error is still being read."""

    def __init__(out self, status: Status, detail: StaticString):
        """A failure the binding caught before any engine could name it.

        Args:
            status: Which class of failure it is.
            detail: The entry point or binding type it was about.
        """
        self.status = status
        self.detail = detail
        self.message = None
        self.name = None
        self.library = None

    def write_to(self, mut writer: Some[Writer]):
        """Writes the status name, the entry point, and the engine's own message together.

        Args:
            writer: The sink the message is appended to.
        """
        writer.write("UForm: ")
        if case .Some(name) = self.name:
            writer.write(text_at(name))
        else:
            match self.status:
                case == Status.LIBRARY_MISSING:
                    writer.write("library-missing")
                case == Status.SYMBOL_MISSING:
                    writer.write("symbol-missing")
                case == Status.VERSION_MISMATCH:
                    writer.write("version-mismatch")
                case _:
                    writer.write("status ", self.status.code)
        writer.write(" [", self.detail, "]")
        if case .Some(message) = self.message:
            writer.write(": ", text_at(message))


def text_at(text: CString) -> String:
    """Copies a null-terminated C string the engine owns, stopping at its terminator."""
    var bytes = text.unsafe_bitcast[UInt8]()
    var length = 0
    while bytes[unsafe_offset=length] != 0:
        length += 1
    return String(StringSlice(unsafe_from_utf8=Span[UInt8, ImmUntrackedOrigin](unsafe_ptr=bytes, length=length)))


# endregion Errors

# region Scalars


@fieldwise_init
struct ScalarKind(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The scalar types a model computes in and its outputs land as, `uform_scalar_t` named by width.

    Named `ScalarKind` because Mojo's own `Scalar` is the element type it describes. The `xN`
    suffixes spell how many logical dimensions share one byte. `Library.scalar_name` spells each one
    as the engine does.
    """

    var code: c_int
    """The raw `uform_scalar_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No scalar named, an unset field refused rather than written at the wrong width."""
    comptime F64 = Self(1)
    """IEEE-754 binary64."""
    comptime F32 = Self(2)
    """IEEE-754 binary32."""
    comptime BF16 = Self(3)
    """Truncated f32 with an 8-bit exponent and a 7-bit mantissa."""
    comptime F16 = Self(4)
    """IEEE-754 binary16."""
    comptime E5M2 = Self(5)
    """FP8 with a 5-bit exponent, with infinities and NaNs."""
    comptime E4M3 = Self(6)
    """FP8 with a 4-bit exponent, no infinities."""
    comptime E3M2 = Self(7)
    """FP6 with a 3-bit exponent, byte-aligned with two padding bits."""
    comptime E2M3 = Self(8)
    """FP6 with a 2-bit exponent, byte-aligned with two padding bits."""
    comptime I8 = Self(10)
    """8-bit signed integer."""
    comptime I4X2 = Self(11)
    """Two signed nibbles per byte, even dimensions in the low nibble."""
    comptime I32 = Self(12)
    """32-bit signed integer."""
    comptime U8 = Self(20)
    """8-bit unsigned integer, the element of UTF-8 text and of encoded files."""
    comptime U4X2 = Self(21)
    """Two unsigned nibbles per byte, even dimensions in the low nibble."""
    comptime U1X8 = Self(22)
    """Eight single-bit dimensions per byte, the sign of each embedding dimension."""

    @staticmethod
    def of[dtype: DType]() -> Self:
        """The kind a Mojo scalar type stores one-to-one; the packed and FP6 kinds have none. Binding-only."""
        comptime match dtype:
            case == DType.float64:
                return Self.F64
            case == DType.float32:
                return Self.F32
            case == DType.bfloat16:
                return Self.BF16
            case == DType.float16:
                return Self.F16
            case == DType.float8_e5m2:
                return Self.E5M2
            case == DType.float8_e4m3fn:
                return Self.E4M3
            case == DType.int32:
                return Self.I32
            case == DType.int8:
                return Self.I8
            case == DType.uint8:
                return Self.U8
            case _:
                return Self.UNKNOWN

    def bits_per_scalar(self) -> Int:
        """How many bits one logical dimension occupies, which sub-byte kinds share; zero for `UNKNOWN`."""
        if self == Self.F64:
            return 64
        if self == Self.F32 or self == Self.I32:
            return 32
        if self == Self.BF16 or self == Self.F16:
            return 16
        if self == Self.I4X2 or self == Self.U4X2:
            return 4
        if self == Self.U1X8:
            return 1
        if self == Self.UNKNOWN:
            return 0
        return 8

    def stores[dtype: DType](self) -> Bool:
        """Whether values of this kind read as whole `dtype` elements: equal widths, or bytes for packed
        kinds. Binding-only."""
        var bits = self.bits_per_scalar()
        if bits == 0:
            return False
        if bits < 8:
            return size_of[dtype]() == 1
        return size_of[dtype]() * 8 == bits


# endregion Scalars

# region Execution


@fieldwise_init
struct Completion(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a verb returns with its final status or once submitted, `uform_completion_t`.

    The receiver picks it: the model and `model.on(executor)` pass `BLOCKING`, and
    `model.deferred(executor)` passes `DEFERRED`, so no verb takes it as an argument.
    """

    var code: c_int
    """The raw `uform_completion_t` value the ABI exchanges."""
    comptime BLOCKING = Self(0)
    """Return with the final status stored and every output readable."""
    comptime DEFERRED = Self(1)
    """Return once submitted; the status reads `DEFERRED` until final."""


@fieldwise_init
struct Mood(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """What an executor does with its threads or streams once it goes idle, `uform_mood_t`."""

    var code: c_int
    """The raw `uform_mood_t` value the ABI exchanges."""
    comptime PARKED = Self(0)
    """Sleep in the kernel until the next call, freeing the cores."""
    comptime SPINNING = Self(1)
    """Keep polling, trading the cores for the lowest wake-up latency."""


@fieldwise_init
struct Residency(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Where an arena's outputs live, as `Arena.describe` reports it, `uform_residency_t`."""

    var code: c_int
    """The raw `uform_residency_t` value the ABI exchanges."""
    comptime RAM = Self(1)
    """Host pages, which the host reads directly."""
    comptime VRAM = Self(2)
    """One GPU's memory, which only device code reads."""


@fieldwise_init
struct CPUClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The CPU family dispatch picks, or a placement pins, `uform_cpu_class_t`."""

    var code: c_int
    """The raw `uform_cpu_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the CPU's own class."""
    comptime SERIAL = Self(1)
    """The portable floor: no SIMD assumptions."""
    comptime INTEL_HASWELL = Self(2)
    """Haswell-class x86: AVX2 and FMA."""
    comptime INTEL_SIERRA = Self(3)
    """Sierra Forest-class x86: AVX-VNNI without AVX-512."""
    comptime INTEL_SKYLAKE = Self(4)
    """Skylake-class x86: the AVX-512 foundation tier."""
    comptime INTEL_ICELAKE = Self(5)
    """Icelake-class x86: AVX-512 VNNI and VPOPCNT."""
    comptime AMD_GENOA = Self(6)
    """Genoa-class x86: native bf16 dot products over AVX-512."""
    comptime INTEL_SAPPHIRE = Self(7)
    """Sapphire Rapids-class x86: AVX-512 FP16 and the AMX-BF16 and AMX-INT8 tiles."""
    comptime INTEL_GRANITE = Self(8)
    """Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles."""
    comptime INTEL_DIAMOND = Self(9)
    """Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions."""
    comptime AWS_G2 = Self(10)
    """Graviton2-class Arm, the bare NEON floor."""
    comptime APPLE_M2 = Self(11)
    """Armv8.6+ NEON: dot products, bf16 and f16 fused multiplies."""
    comptime APPLE_M4 = Self(12)
    """Apple M4-class Arm: the Armv8.6 NEON bundle plus the SME outer-product tiles."""
    comptime AWS_G5 = Self(13)
    """Graviton5-class Arm, Neoverse V3."""
    comptime NVIDIA_VERA = Self(14)
    """Vera-class Arm with FP8 dot products."""
    comptime ALIBABA_C930 = Self(15)
    """Alibaba C930-class RISC-V with RVA23 vectors."""
    comptime WASM32_V128 = Self(16)
    """WebAssembly every engine loads: 32-bit, single-threaded, simd128."""
    comptime WASM64_EXTENDED = Self(17)
    """WebAssembly with relaxed SIMD, shared memory, and 64-bit addressing."""
    comptime MONOVERSIONED = Self(18)
    """The one class of a monoversioned build."""


@fieldwise_init
struct GPUClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The GPU family dispatch picks, or a placement pins, `uform_gpu_class_t`."""

    var code: c_int
    """The raw `uform_gpu_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the devices' own class."""
    comptime NVIDIA_AMPERE = Self(1)
    """Ampere-class NVIDIA and every GPU without a class of its own."""
    comptime NVIDIA_ADA = Self(2)
    """Ada-class NVIDIA: the fp8 conversions."""
    comptime NVIDIA_HOPPER = Self(3)
    """Hopper-class NVIDIA."""
    comptime NVIDIA_BLACKWELL = Self(4)
    """Blackwell-class NVIDIA datacenter parts."""
    comptime NVIDIA_BLACKWELLRTX = Self(5)
    """Consumer Blackwell NVIDIA."""
    comptime NVIDIA_RUBIN = Self(6)
    """Rubin-class NVIDIA."""
    comptime AMD_CDNA3 = Self(7)
    """CDNA3 AMD: gfx942, FNUZ fp8."""
    comptime AMD_CDNA4 = Self(8)
    """CDNA4 AMD: gfx950, OCP fp8."""
    comptime AMD_CDNA5 = Self(9)
    """CDNA5 AMD: gfx1250."""
    comptime APPLE_METAL7 = Self(10)
    """Apple7-family GPUs and every Apple GPU without a class of its own."""
    comptime APPLE_METAL9 = Self(11)
    """Apple9-family GPUs."""
    comptime APPLE_METAL10 = Self(12)
    """Apple10-family GPUs."""


@fieldwise_init
struct GPUVendor(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The GPU vendors one binary drives at once, `uform_gpu_vendor_t`."""

    var code: c_int
    """The raw `uform_gpu_vendor_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No vendor named."""
    comptime NVIDIA = Self(1)
    """NVIDIA, through the CUDA driver."""
    comptime AMD = Self(2)
    """AMD, through the HIP runtime."""
    comptime APPLE = Self(3)
    """Apple, through Metal."""


struct GPUDevice(Copyable, Defaultable, Movable):
    """One GPU as UForm numbers it, `uform_gpu_device_t`: its position in `Library.gpu_devices` is the
    ordinal every VRAM placement names it by."""

    var vendor: GPUVendor
    """The vendor whose runtime drives the device."""
    var vendor_ordinal: UInt32
    """The device's ordinal in its vendor's runtime."""
    var gpu_class: GPUClass
    """The class whose kernels run on it."""
    var compute_units: UInt32
    """Streaming multiprocessors or compute units."""
    var memory_bytes: UInt64
    """Device memory in bytes."""
    var _name: StaticTuple[UInt8, 64]
    """The vendor's own name for the device, NUL-terminated and truncated to fit."""

    def __init__(out self):
        """An empty entry for the listing to fill."""
        self.vendor = GPUVendor.UNKNOWN
        self.vendor_ordinal = 0
        self.gpu_class = GPUClass.UNKNOWN
        self.compute_units = 0
        self.memory_bytes = 0
        self._name = StaticTuple[UInt8, 64](fill=0)

    def name(self) -> String:
        """The vendor's own name for the device."""
        return text_at(CString(unsafe_from_address=Int(Pointer(to=self._name))))


@fieldwise_init
struct Continuation(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether the call a progress handler is watching carries on or unwinds, `uform_continuation_t`."""

    var code: c_int
    """The raw `uform_continuation_t` value the ABI exchanges."""
    comptime CONTINUE = Self(0)
    """Carry on to the next unit of work."""
    comptime CANCEL = Self(1)
    """Stop, and finish the call with `CANCELLED`."""


def _progress[
    T: AnyType, handler: def(Int, Int, Pointer[T, MutAnyOrigin]) thin -> Continuation
](completed: c_size_t, total: c_size_t, state: Pointer[T, MutAnyOrigin]) abi("C") -> c_int:
    """The C callback of one handler, instantiated per handler since a capturing closure cannot cross."""
    return handler(Int(completed), Int(total), state).code


@fieldwise_init
struct Progress(Defaultable, ImplicitlyCopyable):
    """How a long call reports its advance, `uform_progress_t`: a handler, its state, and how many reports
    it hears. The handler is never invoked concurrently, nor after its call is final."""

    var callback: Address
    """The handler's C entry point; `NULL_ADDRESS` skips every report."""
    var state: Address
    """Forwarded into every report; it must outlive the call."""
    var steps: c_size_t
    """Reports a call makes at most, the last at completion; zero derives to ten."""

    def __init__(out self):
        """No handler: the call runs as one slice and reports nothing."""
        self = Self(NULL_ADDRESS, NULL_ADDRESS, 0)

    @staticmethod
    def of[
        T: AnyType, origin: MutOrigin, //, handler: def(Int, Int, Pointer[T, MutAnyOrigin]) thin -> Continuation
    ](state: Pointer[T, origin], steps: Int = 0) -> Self:
        """Reports to `handler` with `state`, which returns `Continuation.CANCEL` to stop the call.

        Parameters:
            T: What `state` points at, inferred from it.
            origin: The state's origin, inferred with it.
            handler: Called with the units completed, the units in total, and `state`.

        Args:
            state: Forwarded into every report; it must outlive the call.
            steps: Reports the call makes at most; zero derives to ten.
        """
        var callback = _progress[T, handler]
        return Self(Pointer(to=callback).unsafe_bitcast[Int]()[], Int(state), c_size_t(steps))


# endregion Execution

# region Signatures

comptime Verb[Arguments: AnyType] = def(Pointer[Arguments, MutAnyOrigin]) thin abi("C") -> None
"""Every struct verb: one argument structure named after it, its outputs written through its pointers."""

comptime Release = def(Address) thin abi("C") -> None
"""The positional `_free` family of executors, arenas, models, meshes and trainers; `NULL` is safe."""

comptime OutcomeStatus = def(Pointer[Outcome, MutAnyOrigin]) thin abi("C") -> c_int
"""`uform_outcome_wait` and `uform_outcome_status`: an outcome's status, the first once it is final."""

comptime NameOf = def(c_int) thin abi("C") -> CString
"""The `_name` family: an enumerator's static spelling."""

comptime FromName = def(BytesViewArguments, Pointer[c_int, MutAnyOrigin]) thin abi("C") -> c_int
"""The `_from_name` family: reads a spelling back into its enumerator, or fails with `INVALID_ARGUMENT`."""

comptime GPUDevices = Verb[GPUDevicesArguments]
"""`uform_gpu_devices`."""
comptime ExecutorForRam = Verb[ExecutorForRamArguments]
"""`uform_executor_for_ram`."""
comptime ExecutorOnPool = Verb[ExecutorOnPoolArguments]
"""`uform_executor_on_pool`."""
comptime ExecutorForVram = Verb[ExecutorForVramArguments]
"""`uform_executor_for_vram`."""
comptime ExecutorDrain = Verb[ExecutorDrainArguments]
"""`uform_executor_drain`."""
comptime ArenaInit = Verb[ArenaInitArguments]
"""`uform_arena_init`."""
comptime ArenaDescribe = Verb[ArenaDescribeArguments]
"""`uform_arena_describe`."""


def _cross[Arguments: ImplicitlyCopyable & Deinitable, //](arguments: Arguments, verb: Verb[Arguments]):
    """Materializes one argument structure in a stack slot and hands its address to `verb`."""
    var slot = stack_allocation[1, Arguments]()
    slot[unsafe_offset=0] = arguments
    verb(slot.unsafe_origin_cast[MutAnyOrigin]())


# endregion Signatures

# region Inputs


@fieldwise_init
struct BytesViewArguments(ImplicitlyCopyable, TrivialRegisterPassable):
    """`uform_bytes_view_t`, laid out as the ABI declares it; register-passable, since the name tables
    take it by value, which an `Optional` field would forbid."""

    var data: Address
    var length: c_size_t


@fieldwise_init
struct BytesView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """A borrowed run of bytes and its length, never null-terminated: every path, buffer and name.

    Parameters:
        origin: The borrowed storage, which the view keeps alive for as long as it is used.
    """

    var data: Pointer[UInt8, Self.origin]
    """The first byte."""
    var length: Int
    """How many bytes the run covers, since nothing terminates it."""

    @staticmethod
    def of[text_origin: ImmOrigin, //](text: StringSlice[text_origin]) -> BytesView[text_origin]:
        """Borrows a string's bytes, with no terminator manufactured."""
        return BytesView[text_origin](
            Pointer[UInt8, text_origin](unsafe_from_address=Int(text.unsafe_ptr())), text.byte_length()
        )

    @staticmethod
    def of[bytes_origin: ImmOrigin, //](bytes: Span[UInt8, bytes_origin]) -> BytesView[bytes_origin]:
        """Borrows a run of raw bytes, such as a checkpoint held in memory."""
        return BytesView[bytes_origin](
            Pointer[UInt8, bytes_origin](unsafe_from_address=Int(bytes.unsafe_ptr())), len(bytes)
        )

    def arguments(self) -> BytesViewArguments:
        """The C layout, erased only inside a verb so the borrow lives through the call."""
        return BytesViewArguments(Int(self.data), c_size_t(self.length))


@fieldwise_init
struct TapeViewArguments(Defaultable, ImplicitlyCopyable):
    """`uform_tape_view_t`, laid out as the ABI declares it."""

    var data: Address
    var offsets: Optional[Pointer[c_size_t, ImmUntrackedOrigin]]
    var scalar: ScalarKind

    def __init__(out self):
        """Nothing written, which is what a failed call leaves and an absent tape crosses as."""
        self = Self(NULL_ADDRESS, None, ScalarKind.UNKNOWN)


@fieldwise_init
struct TapeView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, Sized):
    """A borrowed ragged batch laid out as Apache Arrow lays one out: elements end to end, and one more
    offset than entries around them.

    Entry `i` spans elements `offsets[i]` up to `offsets[i + 1]`. The elements may live in host or
    device memory; the offsets always live in host memory, since they drive host-side dispatch.

    Parameters:
        origin: The borrowed elements and offsets, which the view keeps alive for as long as it is used.
    """

    var data: Pointer[NoneType, Self.origin]
    """The first element, which every offset counts from."""
    var offsets: Pointer[c_size_t, Self.origin]
    """The entries' boundaries, one more of them than entries."""
    var count: Int
    """Entries in the tape, which the verb takes as its `count`."""
    var scalar: ScalarKind
    """The scalar type the elements hold: `U8` for text and encoded files, `F32` for decoded samples."""

    def __len__(self) -> Int:
        """The entries in the tape."""
        return self.count

    @staticmethod
    def of[
        dtype: DType, data_origin: ImmOrigin, offsets_origin: ImmOrigin, //
    ](
        data: Span[Scalar[dtype], data_origin],
        offsets: Span[c_size_t, offsets_origin],
        scalar: ScalarKind = ScalarKind.of[dtype](),
    ) raises Error -> TapeView[ImmOrigin(origin_of(_anchor[data_origin](), _anchor[offsets_origin]()))]:
        """Reads host elements and their offsets as a tape, refusing offsets that end past the elements.

        Raises:
            `Error` with `SHAPE_MISMATCH` when no offset is given or the last one exceeds `len(data)`.
        """
        if len(offsets) == 0 or Int(offsets[len(offsets) - 1]) > len(data):
            raise Error(Status.SHAPE_MISMATCH, "TapeView.of")
        # An empty run is never read, so the offsets' address stands in for its possibly null base.
        var base = Int(data.unsafe_ptr()) if len(data) else Int(offsets.unsafe_ptr())
        return Self.of(Pointer[Scalar[dtype], data_origin](unsafe_from_address=base), offsets, scalar)

    @staticmethod
    def of[
        dtype: DType, data_origin: ImmOrigin, offsets_origin: ImmOrigin, //
    ](
        data: Pointer[Scalar[dtype], data_origin],
        offsets: Span[c_size_t, offsets_origin],
        scalar: ScalarKind = ScalarKind.of[dtype](),
    ) raises Error -> TapeView[ImmOrigin(origin_of(_anchor[data_origin](), _anchor[offsets_origin]()))]:
        """Reads elements behind a host or device pointer, such as `buffer.unsafe_ptr()` of a MAX
        `DeviceBuffer`, with their offsets in host memory.

        Raises:
            `Error` with `SHAPE_MISMATCH` when no offset is given.
        """
        comptime union = ImmOrigin(origin_of(_anchor[data_origin](), _anchor[offsets_origin]()))
        if len(offsets) == 0:
            raise Error(Status.SHAPE_MISMATCH, "TapeView.of")
        return TapeView[union](
            Pointer[NoneType, union](unsafe_from_address=Int(data)),
            Pointer[c_size_t, union](unsafe_from_address=Int(offsets.unsafe_ptr())),
            len(offsets) - 1,
            scalar,
        )

    def arguments(self) -> TapeViewArguments:
        """The C layout, erased only inside a verb so the borrow lives through the call."""
        return TapeViewArguments(Int(self.data), self.offsets.unsafe_origin_cast[ImmUntrackedOrigin](), self.scalar)


# endregion Inputs

# region Outputs


@fieldwise_init
struct RowsViewArguments(Defaultable, ImplicitlyCopyable, TrivialRegisterPassable):
    """`uform_rows_view_t`, laid out as the ABI declares it."""

    var data: Address
    var stride: c_size_t
    var columns: c_size_t
    var scalar: ScalarKind

    def __init__(out self):
        """Nothing written, which is what a failed call leaves."""
        self = Self(NULL_ADDRESS, 0, 0, ScalarKind.UNKNOWN)


@fieldwise_init
struct RowsTapeViewArguments(Defaultable, ImplicitlyCopyable):
    """`uform_rows_tape_view_t`, laid out as the ABI declares it."""

    var rows: RowsViewArguments
    var offsets: Optional[Pointer[c_size_t, ImmUntrackedOrigin]]

    def __init__(out self):
        """Nothing written, which is what a failed call leaves."""
        self = Self(RowsViewArguments(), None)


@fieldwise_init
struct Rows[origin: MutOrigin](ImplicitlyCopyable, Sized):
    """Rows a call wrote into the arena it borrows, one per item, packed, `uform_rows_view_t` read in place.

    Parameters:
        origin: The arena the rows live in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the rows."""
    var generation: Int
    """The arena's call these rows came from, so a read after a later call is refused."""
    var view: RowsViewArguments
    """The first row, the stride, the width and the kind, as the engine wrote them."""
    var count: Int
    """Rows written."""

    def __len__(self) -> Int:
        """The rows written."""
        return self.count

    def columns(self) -> Int:
        """Values per row, the model's embedding dimensions or token width."""
        return Int(self.view.columns)

    def stride(self) -> Int:
        """Bytes between the starts of consecutive rows."""
        return Int(self.view.stride)

    def scalar(self) -> ScalarKind:
        """The scalar type the rows hold."""
        return self.view.scalar

    def row[
        dtype: DType
    ](ref self, row: Int) raises Error -> Span[Scalar[dtype], origin_of(self)]:
        """One row's values, or its packed bytes for a sub-byte kind, borrowed from the arena.

        Raises:
            `Error` with `DEVICE` over device memory, `INVALID_ARGUMENT` once a later call rewound the arena, or when the rows hold
            a kind `dtype` does not store.
        """
        _readable(self.arena, self.generation, "Rows.row")
        if not self.view.scalar.stores[dtype]():
            raise Error(Status.INVALID_ARGUMENT, "Rows.row")
        var row_bytes = (Int(self.view.columns) * self.view.scalar.bits_per_scalar() + 7) // 8
        return _in_arena[Scalar[dtype], origin_of(self)](
            self.view.data + row * Int(self.view.stride), row_bytes // size_of[dtype]()
        )

    def values[
        dtype: DType
    ](ref self) raises Error -> Span[Scalar[dtype], origin_of(self)]:
        """Every row's values end to end, refused for padded rows.

        Raises:
            `Error` with `DEVICE` over device memory, `INVALID_ARGUMENT` once a later call rewound the arena or for another kind,
            and with `UNSUPPORTED_LAYOUT` when the rows are padded.
        """
        _readable(self.arena, self.generation, "Rows.values")
        if not self.view.scalar.stores[dtype]():
            raise Error(Status.INVALID_ARGUMENT, "Rows.values")
        var row_bytes = (Int(self.view.columns) * self.view.scalar.bits_per_scalar() + 7) // 8
        if row_bytes != Int(self.view.stride):
            raise Error(Status.UNSUPPORTED_LAYOUT, "Rows.values")
        return _in_arena[Scalar[dtype], origin_of(self)](self.view.data, self.count * row_bytes // size_of[dtype]())



@fieldwise_init
struct RowsTape[origin: MutOrigin](ImplicitlyCopyable, Sized):
    """A ragged run of rows a call wrote into the arena it borrows, as many per item as the item had
    tokens left after truncation, `uform_rows_tape_view_t` read in place.

    Parameters:
        origin: The arena the rows live in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the rows."""
    var generation: Int
    """The arena's call these rows came from, so a read after a later call is refused."""
    var view: RowsTapeViewArguments
    """The rows and their offsets, as the engine wrote them."""
    var count: Int
    """Items the call wrote rows for."""

    def __len__(self) -> Int:
        """The items the call wrote rows for."""
        return self.count

    def offsets(
        ref self,
    ) raises Error -> Span[c_size_t, origin_of(self)]:
        """The items' boundaries in rows, one more of them than items.

        Raises:
            `Error` with `INVALID_ARGUMENT` once a later call rewound the arena, or `DEVICE` over device memory.
        """
        _readable(self.arena, self.generation, "RowsTape.offsets")
        return _in_arena[c_size_t, origin_of(self)](_address_of(self.view.offsets), self.count + 1)

    def rows(self) raises Error -> Rows[Self.origin]:
        """Every item's rows end to end, as one run.

        Raises:
            `Error` with `INVALID_ARGUMENT` once a later call rewound the arena, or `DEVICE` over device memory.
        """
        _readable(self.arena, self.generation, "RowsTape.rows")
        var total = _in_arena[c_size_t, ImmUntrackedOrigin](_address_of(self.view.offsets), self.count + 1)
        return Rows[Self.origin](self.arena, self.generation, self.view.rows, Int(total[self.count]))

    def item(self, item: Int) raises Error -> Rows[Self.origin]:
        """One item's rows.

        Raises:
            `Error` with `INVALID_ARGUMENT` once a later call rewound the arena, or `DEVICE` over device memory.
        """
        _readable(self.arena, self.generation, "RowsTape.item")
        var offsets = _in_arena[c_size_t, ImmUntrackedOrigin](_address_of(self.view.offsets), self.count + 1)
        var first = Int(offsets[item])
        var view = self.view.rows
        view.data += first * Int(view.stride)
        return Rows[Self.origin](self.arena, self.generation, view, Int(offsets[item + 1]) - first)



@fieldwise_init
struct Tape[origin: MutOrigin](ImplicitlyCopyable, Sized):
    """A ragged run of values a call wrote into the arena it borrows, `uform_tape_view_t` read in place.

    Parameters:
        origin: The arena the values live in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the values."""
    var generation: Int
    """The arena's call these values came from, so a read after a later call is refused."""
    var view: TapeViewArguments
    """The values and their offsets, as the engine wrote them."""
    var count: Int
    """Items the call wrote values for."""

    def __len__(self) -> Int:
        """The items the call wrote values for."""
        return self.count

    def scalar(self) -> ScalarKind:
        """The scalar type the values hold."""
        return self.view.scalar

    def offsets(
        ref self,
    ) raises Error -> Span[c_size_t, origin_of(self)]:
        """The items' boundaries in values, one more of them than items.

        Raises:
            `Error` with `INVALID_ARGUMENT` once a later call rewound the arena, or `DEVICE` over device memory.
        """
        _readable(self.arena, self.generation, "Tape.offsets")
        return _in_arena[c_size_t, origin_of(self)](_address_of(self.view.offsets), self.count + 1)

    def item[
        dtype: DType
    ](ref self, item: Int) raises Error -> Span[Scalar[dtype], origin_of(self)]:
        """One item's values.

        Raises:
            `Error` with `DEVICE` over device memory, `INVALID_ARGUMENT` once a later call rewound the arena, or when the values
            hold a kind `dtype` does not store.
        """
        _readable(self.arena, self.generation, "Tape.item")
        if not self.view.scalar.stores[dtype]():
            raise Error(Status.INVALID_ARGUMENT, "Tape.item")
        var offsets = _in_arena[c_size_t, ImmUntrackedOrigin](_address_of(self.view.offsets), self.count + 1)
        var first = Int(offsets[item])
        return _in_arena[Scalar[dtype], origin_of(self)](
            self.view.data + first * size_of[dtype](), Int(offsets[item + 1]) - first
        )



def _address_of[T: AnyType, origin: Origin, //](pointer: Optional[Pointer[T, origin]]) -> Address:
    """The numeric value of a pointer the engine may have left null."""
    if case .Some(present) = pointer:
        return Int(present)
    return NULL_ADDRESS


def _in_arena[T: Copyable, origin: Origin](address: Address, length: Int) -> Span[T, origin]:
    """`length` values a call wrote into an arena; a failed or empty call left `address` null."""
    if length == 0 or address == NULL_ADDRESS:
        return Span[T, origin]()
    return Span[T, origin](unsafe_ptr=Pointer[T, origin](unsafe_from_address=address), length=length)


# endregion Outputs

# region Placement


@fieldwise_init
struct RamOptionsArguments(ImplicitlyCopyable):
    """`uform_ram_options_t`, laid out as the ABI declares it."""

    var memory_domains: Optional[Pointer[Int32, ImmUntrackedOrigin]]
    var memory_domains_count: c_size_t
    var cpu_class: CPUClass


@fieldwise_init
struct RamOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The host memory domains a RAM-resident model and executor place their bytes in, one NUMA node each.

    Parameters:
        origin: The borrowed node ID array, kept alive for as long as the options are used.
    """

    var memory_domains: Optional[Pointer[Int32, Self.origin]]
    """NUMA node IDs; absent derives every node the process may allocate on."""
    var memory_domains_count: Int
    """Entries in `memory_domains`; zero derives."""
    var cpu_class: CPUClass
    """The class to dispatch as; `UNKNOWN` derives the CPU's own."""

    def __init__(out self, *, cpu_class: CPUClass = CPUClass.UNKNOWN):
        """Derives every memory domain the process may allocate on."""
        self.memory_domains = None
        self.memory_domains_count = 0
        self.cpu_class = cpu_class

    @staticmethod
    def on[
        domains_origin: ImmOrigin, //
    ](memory_domains: Span[Int32, domains_origin], cpu_class: CPUClass = CPUClass.UNKNOWN) -> RamOptions[domains_origin]:
        """Names the NUMA nodes explicitly, borrowing the caller's array for the call."""
        return RamOptions[domains_origin](
            Pointer[Int32, domains_origin](unsafe_from_address=Int(memory_domains.unsafe_ptr())),
            len(memory_domains),
            cpu_class,
        )

    def arguments(self) -> RamOptionsArguments:
        """The C layout, erased only inside a verb so the borrow lives through the call."""
        var domains: Optional[Pointer[Int32, ImmUntrackedOrigin]] = None
        if case .Some(named) = self.memory_domains:
            domains = named.unsafe_origin_cast[ImmUntrackedOrigin]()
        return RamOptionsArguments(domains, c_size_t(self.memory_domains_count), self.cpu_class)


@fieldwise_init
struct VramOptionsArguments(ImplicitlyCopyable):
    """`uform_vram_options_t`, laid out as the ABI declares it."""

    var devices: Optional[Pointer[Int32, ImmUntrackedOrigin]]
    var devices_count: c_size_t
    var gpu_class: GPUClass


@fieldwise_init
struct VramOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The devices a VRAM-resident model, trainer and executor place their bytes in.

    Parameters:
        origin: The borrowed ordinal array, kept alive for as long as the options are used.
    """

    var devices: Optional[Pointer[Int32, Self.origin]]
    """UForm device ordinals, positions in `Library.gpu_devices`; absent derives one free device."""
    var devices_count: Int
    """Entries in `devices`; zero derives."""
    var gpu_class: GPUClass
    """The class every named device dispatches as; `UNKNOWN` derives the devices' own."""

    def __init__(out self, *, gpu_class: GPUClass = GPUClass.UNKNOWN):
        """Derives one free device."""
        self.devices = None
        self.devices_count = 0
        self.gpu_class = gpu_class

    @staticmethod
    def on[
        devices_origin: ImmOrigin, //
    ](devices: Span[Int32, devices_origin], gpu_class: GPUClass = GPUClass.UNKNOWN) -> VramOptions[devices_origin]:
        """Names the device ordinals explicitly, borrowing the caller's array for the call."""
        return VramOptions[devices_origin](
            Pointer[Int32, devices_origin](unsafe_from_address=Int(devices.unsafe_ptr())), len(devices), gpu_class
        )

    def arguments(self) -> VramOptionsArguments:
        """The C layout, erased only inside a verb so the borrow lives through the call."""
        var devices: Optional[Pointer[Int32, ImmUntrackedOrigin]] = None
        if case .Some(named) = self.devices:
            devices = named.unsafe_origin_cast[ImmUntrackedOrigin]()
        return VramOptionsArguments(devices, c_size_t(self.devices_count), self.gpu_class)


# endregion Placement

# region Executors


@fieldwise_init
struct GPUDevicesArguments(ImplicitlyCopyable):
    """`uform_gpu_devices_t`, laid out as the ABI declares it."""

    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var devices: Optional[Pointer[GPUDevice, MutUntrackedOrigin]]
    var devices_capacity: c_size_t
    var devices_count: Optional[Pointer[c_size_t, MutUntrackedOrigin]]


@fieldwise_init
struct ExecutorForRamArguments(ImplicitlyCopyable):
    """`uform_executor_for_ram_t`, laid out as the ABI declares it."""

    var executor: Optional[Pointer[Address, MutUntrackedOrigin]]
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var ram: RamOptionsArguments
    var threads: c_size_t


@fieldwise_init
struct ExecutorOnPoolArguments(ImplicitlyCopyable):
    """`uform_executor_on_pool_t`, laid out as the ABI declares it."""

    var executor: Optional[Pointer[Address, MutUntrackedOrigin]]
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var ram: RamOptionsArguments
    var pool: Address


@fieldwise_init
struct ExecutorForVramArguments(ImplicitlyCopyable):
    """`uform_executor_for_vram_t`, laid out as the ABI declares it."""

    var executor: Optional[Pointer[Address, MutUntrackedOrigin]]
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var vram: VramOptionsArguments
    var ram: RamOptionsArguments
    var streams: Optional[Pointer[Address, ImmUntrackedOrigin]]
    var scratch_limit: c_size_t


@fieldwise_init
struct ExecutorDrainArguments(ImplicitlyCopyable):
    """`uform_executor_drain_t`, laid out as the ABI declares it."""

    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var mood: Mood


struct _ExecutorHandle:
    """The `uform_executor_t` an `Executor` owns, shared with every device arena it made and freed with
    the last of them, since such an arena grows, fills and frees on the executor's first stream."""

    var library: Library
    var address: Address

    def __init__(out self, library: Library, address: Address):
        """Takes over a spawned handle, or the null one the caller-thread executor names."""
        self.library = library
        self.address = address

    def __deinit__(deinit self):
        """Drains every thread or stream, releases the scratch, and frees the executor.
        Calls `uform_executor_free`.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo would `dlclose` the engine between
        the load and the branch to it.
        """
        self.library.symbols().executor_free(self.address)
        _ = self.library^


struct Executor:
    """The threads or streams a verb runs on, freed once it and every device arena it made go out of scope.

    `Executor(library)` owns nothing: every verb handed it runs on the calling thread. The spawning
    constructors fix a kind that must match the residency of every model they drive. An executor
    runs one call at a time, so threads that genuinely overlap take one apiece.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `uform_executor_t` every verb hands back to the engine; `NULL_ADDRESS` runs inline."""
    var _handle: ArcPointer[_ExecutorHandle]
    """The handle's owner, which a device arena shares so the executor outlives it."""

    def __init__(out self, library: Library):
        """The caller-thread executor, which every verb accepts whatever the model's residency.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
        """
        self = Self(library=library, address=NULL_ADDRESS)

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle that is already spawned, which is how the factories hand one back."""
        self.library = library
        self.address = address
        self._handle = ArcPointer(_ExecutorHandle(library, address))

    @staticmethod
    def for_ram[
        ram_origin: ImmOrigin, //
    ](
        library: Library, ram: RamOptions[ram_origin] = RamOptions(), threads: Int = 0
    ) raises Error -> Self:
        """Creates an executor of host threads for RAM-resident models. Calls `uform_executor_for_ram`.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            ram: The memory domains of the models it will drive; copied, not borrowed.
            threads: Threads to spawn; zero takes 7/8 of the cores those domains carry.

        Raises:
            `Error` if the domains cannot be resolved or the OS refuses the threads.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ExecutorForRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                ram.arguments(),
                c_size_t(threads),
            ),
            library.symbols().executor_for_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_executor_for_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def on_pool[
        ram_origin: ImmOrigin, //
    ](library: Library, pool: Address, ram: RamOptions[ram_origin] = RamOptions()) raises Error -> Self:
        """Schedules onto a borrowed ForkUnion pool, so UForm, USearch and UStore in one process share one
        set of threads. Calls `uform_executor_on_pool`.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            pool: A live ForkUnion `fu_pool_t`, which must outlive the executor.
            ram: The memory domains of the models it will drive; copied, not borrowed.

        Raises:
            `Error` if the pool or the domains are refused.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ExecutorOnPoolArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                ram.arguments(),
                pool,
            ),
            library.symbols().executor_on_pool,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_executor_on_pool")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def for_vram[
        vram_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        library: Library,
        vram: VramOptions[vram_origin] = VramOptions(),
        ram: RamOptions[ram_origin] = RamOptions(),
        streams: Address = NULL_ADDRESS,
        scratch_limit: Int = 0,
    ) raises Error -> Self:
        """Opens one submission stream on every named device, all driven by the thread running the call,
        and host threads for the decoding the devices cannot do. Calls `uform_executor_for_vram`.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            vram: The devices of the models it will drive; copied, not borrowed.
            ram: The memory domains and CPU class of its host decoders; copied, not borrowed.
            streams: Borrowed CUDA streams, one per device; any but `NULL_ADDRESS` reports `UNSUPPORTED`
                until borrowing is wired.
            scratch_limit: Ceiling on scratch bytes per device; zero lets scratch grow as batches ask.

        Raises:
            `Error` if a device cannot be selected, or with `UNSUPPORTED` in a build without one.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var borrowed: Optional[Pointer[Address, ImmUntrackedOrigin]] = None
        if streams != NULL_ADDRESS:
            borrowed = Pointer[Address, ImmUntrackedOrigin](unsafe_from_address=streams)
        _cross(
            ExecutorForVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                vram.arguments(),
                ram.arguments(),
                borrowed,
                c_size_t(scratch_limit),
            ),
            library.symbols().executor_for_vram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_executor_for_vram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def drain(self, mood: Mood = Mood.PARKED) raises Error:
        """Waits until no call holds the executor, then applies `mood` until its next call.
        Calls `uform_executor_drain`.

        Raises:
            `Error` for a failure of the executor itself, such as a device fault, never one of a
            drained call, which reports through its own outcome.
        """
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ExecutorDrainArguments(self.address, outcome.unsafe_origin_cast[MutUntrackedOrigin](), mood),
            self.library.symbols().executor_drain,
        )
        self.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_executor_drain")


# endregion Executors

# region Arenas


@fieldwise_init
struct ArenaInitArguments(ImplicitlyCopyable):
    """`uform_arena_init_t`, laid out as the ABI declares it."""

    var arena: Optional[Pointer[Address, MutUntrackedOrigin]]
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]


@fieldwise_init
struct ArenaDescription(ImplicitlyCopyable):
    """`uform_arena_description_t`: where an arena's outputs live."""

    var residency: Residency
    """`RAM` for host pages, `VRAM` for one device's memory."""
    var device: Int32
    """The device ordinal, as `Executor.for_vram` takes it, or -1 for host pages."""


@fieldwise_init
struct ArenaDescribeArguments(ImplicitlyCopyable):
    """`uform_arena_describe_t`, laid out as the ABI declares it."""

    var arena: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var description: Optional[Pointer[ArenaDescription, MutUntrackedOrigin]]


struct Arena:
    """The memory every output whose length scales with a call lands in, freed when it goes out of scope.

    Made where an executor computes, host pages or the memory of its GPU, decided once; nothing is
    allocated until a call writes into it. Every call handed it rewinds and refills it, which
    invalidates the previous call's outputs, so a steady loop allocates once. One arena serves one
    call at a time; a second concurrent use is refused by the engine as `CONTENDED`, which leaves
    the holder's results readable.

    A result refuses every read with `INVALID_ARGUMENT` once a later call rewinds its arena, and a
    result over a device arena refuses every host read with `DEVICE`. A device arena serves only
    calls on the executor that made it, and keeps that executor alive until it is freed.
    """

    var _library: Library
    """The loaded engine, held so the release can still reach `uform_arena_free`."""
    var _address: Address
    """The `uform_arena_t`, never null."""
    var _executor: Optional[ArcPointer[_ExecutorHandle]]
    """The device executor that made it, held so the arena is freed first, on that executor's stream."""
    var _generation: Int
    """Calls the engine took on it so far, which every result records to know when a later one rewound it."""

    def __init__(out self, library: Library) raises Error:
        """Host pages for calls on the calling thread. Calls `uform_arena_init`."""
        self = Self(Executor(library))

    def __init__(out self, executor: Executor) raises Error:
        """Memory where `executor` computes, serving the calls of its residency. Calls `uform_arena_init`.

        Args:
            executor: The executor whose calls it serves; host pages may outlive it, and a device arena
                keeps it alive.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ArenaInitArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            ),
            executor.library.symbols().arena_init,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_arena_init")
        self._library = executor.library
        self._address = created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]
        self._executor = None
        self._generation = 0
        if self.describe().residency == Residency.VRAM:
            self._executor = Optional(executor._handle)

    def __deinit__(deinit self):
        """Frees the arena and every output in it, then lets a device arena's executor go.
        Calls `uform_arena_free`.

        The executor and the loader reference are surrendered only after the call, for the reason
        `_ExecutorHandle` records.
        """
        self._library.symbols().arena_free(self._address)
        _ = self._executor^
        _ = self._library^

    def describe(self) raises Error -> ArenaDescription:
        """Where its outputs live, as the executor that made it decided. Calls `uform_arena_describe`."""
        var description = stack_allocation[1, ArenaDescription]()
        description[unsafe_offset=0] = ArenaDescription(Residency.RAM, Int32(-1))
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ArenaDescribeArguments(
                self._address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                description.unsafe_origin_cast[MutUntrackedOrigin](),
            ),
            self._library.symbols().arena_describe,
        )
        self._library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_arena_describe")
        return description.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]


def _served[origin: MutOrigin](arena: Pointer[Arena, origin], status: Status) -> Int:
    """Retires every earlier result once the engine took a call naming the arena; a `CONTENDED` refusal
    never touched it. Answers the generation the call's result is stamped with."""
    var served = Pointer[Arena, MutAnyOrigin](unsafe_from_address=Int(arena))
    if status != Status.CONTENDED:
        served[]._generation += 1
    return served[]._generation


def _readable[origin: MutOrigin](arena: Pointer[Arena, origin], generation: Int, detail: StaticString) raises Error:
    """Refuses a host read of a result over device memory, or over an arena a later call rewound."""
    if arena[].describe().residency == Residency.VRAM:
        raise Error(Status.DEVICE, detail)
    if arena[]._generation != generation:
        raise Error(Status.INVALID_ARGUMENT, detail)


# endregion Arenas

# region Deferred Calls


struct Unwritten(Defaultable, ImplicitlyCopyable):
    """The outputs of a verb that writes nothing but its outcome."""

    def __init__(out self):
        """Holds nothing."""
        pass


@fieldwise_init
struct CallSlots[Outputs: ImplicitlyCopyable & Defaultable & Deinitable](ImplicitlyCopyable):
    """The outcome and the fixed-size outputs one call writes through, at one stable address."""

    var outcome: Outcome
    var outputs: Self.Outputs

    def __init__(out self):
        """Zeroes the outcome and defaults every output."""
        self.outcome = Outcome()
        self.outputs = Self.Outputs()


trait Call(Deinitable, Movable):
    """One verb crossing into the engine: what it lends until final, and how its outputs become a result.

    Each receiver routes the same `Call`: the model and `model.on(executor)` block on it, and
    `model.deferred(executor)` wraps it in a `Deferred`, which is what keeps the verbs symmetric.
    """

    comptime Outputs: ImplicitlyCopyable & Defaultable & Deinitable
    """The fixed-size outputs the engine writes through, such as a description or views into the arena."""
    comptime Result: Movable & Deinitable
    """What a successful call answers."""
    comptime symbol: StaticString
    """The C entry point, carried into every error."""

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[Self.Outputs, MutUntrackedOrigin],
    ):
        """Crosses into the engine, or writes the binding's own refusal into `outcome` without crossing."""
        ...

    def served(self, status: Status) -> Int:
        """Stamps the call's arena once the engine took the call, and answers the stamp; arena-free calls
        keep none."""
        return 0

    def conclude(deinit self, outputs: Self.Outputs, generation: Int) -> Self.Result:
        """Adopts what the final call wrote, success or not, and builds the result around its arena."""
        ...


def _refuse(outcome: Pointer[Outcome, MutUntrackedOrigin], status: Status):
    """Finishes a call the binding refused before it crossed, so every receiver reports it alike."""
    outcome.unsafe_origin_cast[MutAnyOrigin]()[] = Outcome(c_int(status.code), None)


def _block[C: Call](library: Library, var call: C, handle: Address, executor: Address) raises Error -> C.Result:
    """Runs one call to its final status, its slots on the stack, and answers its result or raises."""
    var slots = stack_allocation[1, CallSlots[C.Outputs]]()
    slots[unsafe_offset=0] = CallSlots[C.Outputs]()
    var mutable_slots = slots.unsafe_origin_cast[MutAnyOrigin]()
    call.submit(
        library,
        handle,
        executor,
        Completion.BLOCKING,
        Pointer(to=mutable_slots[unsafe_offset=0].outcome).unsafe_origin_cast[MutUntrackedOrigin](),
        Pointer(to=mutable_slots[unsafe_offset=0].outputs).unsafe_origin_cast[MutUntrackedOrigin](),
    )
    var written = mutable_slots[unsafe_offset=0]
    var generation = call.served(Status.of(written.outcome.status))
    var result = call^.conclude(written.outputs, generation)
    library._check(written.outcome, C.symbol)
    return result^


struct Deferred[C: Call, origin: ImmOrigin](Movable):
    """A call submitted to an executor, which owns its outcome and outputs until it is final. Binding-only.

    `status` reads `DEFERRED` until then, without blocking, and `wait` blocks until the call is
    final and answers its result or raises its failure, as many times as it is asked. The result is
    the view a blocking call answers, borrowing the same arena. Dropping an unfinished one waits the
    same way, so nothing the engine still writes is ever freed under it. The origins keep the model,
    the executor, the arena and every borrowed input alive for as long as this is.

    Parameters:
        C: The verb's crossing, which names the result.
        origin: The union of everything the call borrows.
    """

    var library: Library
    """The loaded engine, held so the wait can still reach `uform_outcome_wait`."""
    var slots: OwnedPointer[CallSlots[Self.C.Outputs]]
    """The outcome and outputs the engine writes, on the heap since they outlive the verb."""
    var call: Optional[Self.C]
    """What the call lends the engine, until a wait concludes it."""
    var answer: Optional[Self.C.Result]
    """The result, once a wait found the call succeeded."""
    var failure: Optional[Error]
    """The failure, once a wait found the call failed."""
    var generation: Int
    """The stamp `served` answered at submission, so a result waited on late still reads as stale."""

    def __init__(out self, library: Library, var call: Self.C, handle: Address, executor: Address):
        """Submits `call` deferred, on `executor`, against `handle`."""
        self.library = library
        self.slots = OwnedPointer(CallSlots[Self.C.Outputs]())
        self.call = call^
        self.answer = None
        self.failure = None
        self.generation = 0
        var slots = self._slots()
        self.call.value().submit(
            self.library,
            handle,
            executor,
            Completion.DEFERRED,
            Pointer(to=slots[].outcome).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=slots[].outputs).unsafe_origin_cast[MutUntrackedOrigin](),
        )
        self.generation = self.call.value().served(self.status())

    def __deinit__(deinit self):
        """Waits until the call is final, then frees what it lent. Calls `uform_outcome_wait`."""
        if self.call:
            var slots = Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin](
                unsafe_from_address=Int(Pointer(to=self.slots[]))
            )
            _ = self.library.symbols().outcome_wait(Pointer(to=slots[].outcome))
            _ = self.call.take().conclude(slots[].outputs, self.generation)
        _ = self.library^

    def _slots(self) -> Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin]:
        """The heap slots, through the origin the engine's writes need."""
        return Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin](unsafe_from_address=Int(Pointer(to=self.slots[])))

    def status(self) -> Status:
        """`DEFERRED` until the call is final, then its final status; never blocks. Calls `uform_outcome_status`."""
        return Status(self.library.symbols().outcome_status(Pointer(to=self._slots()[].outcome)))

    def wait(mut self) raises Error -> ref[self] Self.C.Result:
        """Blocks until the call is final, then answers its result or raises its failure, every time it
        is asked. Calls `uform_outcome_wait`.

        Raises:
            `Error` carrying the call's failure, whether the engine or the binding refused it.
        """
        if self.call:
            var slots = self._slots()
            _ = self.library.symbols().outcome_wait(Pointer(to=slots[].outcome))
            var written = slots[]
            var result = self.call.take().conclude(written.outputs, self.generation)
            if written.outcome.status == c_int(0):
                self.answer = result^
            else:
                self.failure = self.library._error(written.outcome, Self.C.symbol)
        if self.failure:
            raise self.failure.value()
        return Pointer(to=self.answer.value()).unsafe_origin_cast[origin_of(self)]()[]


# endregion Deferred Calls
