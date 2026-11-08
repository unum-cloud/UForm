"""The loaded shared object and the one place a raw symbol address is ever touched.

The engine is reached by `dlopen` rather than by linking, which is what lets a consumer build
without a single linker flag. Mojo's `@extern` cannot serve here: it resolves at link time, so a
symbol from a library that is only opened at run time is an undefined reference under both the JIT
and `lld`, and using it would put a link flag in every consumer's build.

Every entry point is resolved once, when the library loads, into a typed `thin abi("C")` function
pointer. `OwnedDLHandle.get_function` would run `dlsym` on every call and parameterises only on the
return type, leaving arguments unchecked; an encoding on the hot path can afford neither.

File: mojo/uform/library.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from std.ffi import OwnedDLHandle, __fn_type_is_cabi, c_int, c_size_t
from std.memory import ArcPointer, stack_allocation
from std.sys.info import CompilationTarget

from uform.encode import (
    AudioEncode,
    AudioEncodeSamples,
    ImageEncode,
    ImageEncodePixels,
    TextChoose,
    TextEncode,
    VideoEncode,
)
from uform.model import (
    ModelDescribe,
    ModelLoadRam,
    ModelLoadRamBuffer,
    ModelLoadVram,
    ModelLoadVramBuffer,
)
from uform.train import (
    MeshBootstrap,
    MeshDescribe,
    Modality,
    MultimemTier,
    Negatives,
    Objective,
    TrainerInitVram,
    TrainerSave,
    TrainerStep,
)
from uform.types import (
    ArenaDescribe,
    ArenaInit,
    BytesView,
    CPUClass,
    CString,
    Error,
    ExecutorDrain,
    ExecutorForRam,
    ExecutorForVram,
    ExecutorOnPool,
    FromName,
    GPUClass,
    GPUDevice,
    GPUDevices,
    GPUDevicesArguments,
    Mood,
    NameOf,
    Outcome,
    OutcomeStatus,
    Release,
    ScalarKind,
    Status,
    text_at,
)

comptime LIBRARY_FILE = StaticString(
    "libuform_shared.dylib" if CompilationTarget.is_macos() else "libuform_shared.so"
)
"""The loader already searches the environment's library directory, because `mojo build` bakes it
into the binary's runpath, so an unqualified name is enough for a library installed there."""

comptime BINDING_MAJOR = 4
"""The major version this binding was written against; a mismatch is refused at load."""

# region Signatures

comptime StaticText = def() thin abi("C") -> CString
"""`uform_version_name` and both capability getters: a static C string returned by value."""

comptime VersionOf = def(
    Pointer[c_int, MutAnyOrigin], Pointer[c_int, MutAnyOrigin], Pointer[c_int, MutAnyOrigin]
) thin abi("C") -> None
"""`uform_version`: the linked library's major, minor and patch, written through three `int` pointers.

Typed pointers rather than `Address`: an `alloca` whose address crosses only as an integer argument
was measured reading back the value stored before the call."""

# endregion Signatures


@inline(.always)
def _resolve[
    Signature: TrivialRegisterPassable
](ref handle: OwnedDLHandle, name: StaticString) raises Error -> Signature:
    """One `dlsym`, checked once, typed from here on.

    `get_symbol` answers `Optional` where `get_function` aborts the process, which is what lets a
    stale library name the symbol it is missing instead of dying without a message.
    """
    comptime assert __fn_type_is_cabi[Signature](), 'the signature must carry abi("C")'
    if case .Some(address) = handle.get_symbol[NoneType](name):
        return Pointer(to=address).unsafe_bitcast[Signature]()[]
    raise Error(Status.SYMBOL_MISSING, name)


def _version(version: VersionOf) -> Tuple[Int, Int, Int]:
    """The linked library's major, minor and patch, read through `uform_version`."""
    var parts = stack_allocation[3, c_int]().unsafe_origin_cast[MutAnyOrigin]()
    for part in range(3):
        parts[unsafe_offset=part] = c_int(0)
    version(parts, parts.unsafe_offset(1), parts.unsafe_offset(2))
    return (Int(parts[unsafe_offset=0]), Int(parts[unsafe_offset=1]), Int(parts[unsafe_offset=2]))


struct Symbols:
    """Every entry point the C ABI exports, resolved once when the library loads.

    Answers are never cached here, only addresses. A model re-asks the engine for its description
    on every call.
    """

    var version: VersionOf
    """`uform_version`."""
    var version_name: StaticText
    """`uform_version_name`."""
    var capabilities_compiled: StaticText
    """`uform_capabilities_compiled`."""
    var capabilities_available: StaticText
    """`uform_capabilities_available`."""

    var outcome_wait: OutcomeStatus
    """`uform_outcome_wait`."""
    var outcome_status: OutcomeStatus
    """`uform_outcome_status`."""
    var status_name: NameOf
    """`uform_status_name`."""
    var status_from_name: FromName
    """`uform_status_from_name`."""
    var scalar_name: NameOf
    """`uform_scalar_name`."""
    var scalar_from_name: FromName
    """`uform_scalar_from_name`."""
    var mood_name: NameOf
    """`uform_mood_name`."""
    var mood_from_name: FromName
    """`uform_mood_from_name`."""
    var cpu_class_name: NameOf
    """`uform_cpu_class_name`."""
    var cpu_class_from_name: FromName
    """`uform_cpu_class_from_name`."""
    var gpu_class_name: NameOf
    """`uform_gpu_class_name`."""
    var gpu_class_from_name: FromName
    """`uform_gpu_class_from_name`."""
    var gpu_devices: GPUDevices
    """`uform_gpu_devices`."""

    var executor_for_ram: ExecutorForRam
    """`uform_executor_for_ram`."""
    var executor_on_pool: ExecutorOnPool
    """`uform_executor_on_pool`."""
    var executor_for_vram: ExecutorForVram
    """`uform_executor_for_vram`."""
    var executor_drain: ExecutorDrain
    """`uform_executor_drain`."""
    var executor_free: Release
    """`uform_executor_free`."""
    var arena_init: ArenaInit
    """`uform_arena_init`."""
    var arena_describe: ArenaDescribe
    """`uform_arena_describe`."""
    var arena_free: Release
    """`uform_arena_free`."""

    var model_load_ram: ModelLoadRam
    """`uform_model_load_ram`."""
    var model_load_ram_buffer: ModelLoadRamBuffer
    """`uform_model_load_ram_buffer`."""
    var model_load_vram: ModelLoadVram
    """`uform_model_load_vram`."""
    var model_load_vram_buffer: ModelLoadVramBuffer
    """`uform_model_load_vram_buffer`."""
    var model_describe: ModelDescribe
    """`uform_model_describe`."""
    var model_free: Release
    """`uform_model_free`."""

    var text_encode: TextEncode
    """`uform_text_encode`."""
    var text_choose: TextChoose
    """`uform_text_choose`."""
    var image_encode: ImageEncode
    """`uform_image_encode`."""
    var image_encode_pixels: ImageEncodePixels
    """`uform_image_encode_pixels`."""
    var video_encode: VideoEncode
    """`uform_video_encode`."""
    var audio_encode: AudioEncode
    """`uform_audio_encode`."""
    var audio_encode_samples: AudioEncodeSamples
    """`uform_audio_encode_samples`."""

    var modality_name: NameOf
    """`uform_modality_name`."""
    var modality_from_name: FromName
    """`uform_modality_from_name`."""
    var objective_name: NameOf
    """`uform_objective_name`."""
    var objective_from_name: FromName
    """`uform_objective_from_name`."""
    var negatives_name: NameOf
    """`uform_negatives_name`."""
    var negatives_from_name: FromName
    """`uform_negatives_from_name`."""
    var multimem_tier_name: NameOf
    """`uform_multimem_tier_name`."""
    var multimem_tier_from_name: FromName
    """`uform_multimem_tier_from_name`."""
    var mesh_bootstrap: MeshBootstrap
    """`uform_mesh_bootstrap`."""
    var mesh_describe: MeshDescribe
    """`uform_mesh_describe`."""
    var mesh_free: Release
    """`uform_mesh_free`."""
    var trainer_init_vram: TrainerInitVram
    """`uform_trainer_init_vram`."""
    var trainer_step: TrainerStep
    """`uform_trainer_step`."""
    var trainer_save: TrainerSave
    """`uform_trainer_save`."""
    var trainer_free: Release
    """`uform_trainer_free`."""

    def __init__(out self, ref handle: OwnedDLHandle) raises Error:
        """Resolves every entry point at once, so no call ever pays for a `dlsym`.

        Args:
            handle: The opened library, borrowed only while the addresses are read out.

        Raises:
            With `SYMBOL_MISSING`, on the first symbol the loaded library does not export.
        """
        self.version = _resolve[VersionOf](handle, "uform_version")
        self.version_name = _resolve[StaticText](handle, "uform_version_name")
        self.capabilities_compiled = _resolve[StaticText](handle, "uform_capabilities_compiled")
        self.capabilities_available = _resolve[StaticText](handle, "uform_capabilities_available")

        self.outcome_wait = _resolve[OutcomeStatus](handle, "uform_outcome_wait")
        self.outcome_status = _resolve[OutcomeStatus](handle, "uform_outcome_status")
        self.status_name = _resolve[NameOf](handle, "uform_status_name")
        self.status_from_name = _resolve[FromName](handle, "uform_status_from_name")
        self.scalar_name = _resolve[NameOf](handle, "uform_scalar_name")
        self.scalar_from_name = _resolve[FromName](handle, "uform_scalar_from_name")
        self.mood_name = _resolve[NameOf](handle, "uform_mood_name")
        self.mood_from_name = _resolve[FromName](handle, "uform_mood_from_name")
        self.cpu_class_name = _resolve[NameOf](handle, "uform_cpu_class_name")
        self.cpu_class_from_name = _resolve[FromName](handle, "uform_cpu_class_from_name")
        self.gpu_class_name = _resolve[NameOf](handle, "uform_gpu_class_name")
        self.gpu_class_from_name = _resolve[FromName](handle, "uform_gpu_class_from_name")
        self.gpu_devices = _resolve[GPUDevices](handle, "uform_gpu_devices")

        self.executor_for_ram = _resolve[ExecutorForRam](handle, "uform_executor_for_ram")
        self.executor_on_pool = _resolve[ExecutorOnPool](handle, "uform_executor_on_pool")
        self.executor_for_vram = _resolve[ExecutorForVram](handle, "uform_executor_for_vram")
        self.executor_drain = _resolve[ExecutorDrain](handle, "uform_executor_drain")
        self.executor_free = _resolve[Release](handle, "uform_executor_free")
        self.arena_init = _resolve[ArenaInit](handle, "uform_arena_init")
        self.arena_describe = _resolve[ArenaDescribe](handle, "uform_arena_describe")
        self.arena_free = _resolve[Release](handle, "uform_arena_free")

        self.model_load_ram = _resolve[ModelLoadRam](handle, "uform_model_load_ram")
        self.model_load_ram_buffer = _resolve[ModelLoadRamBuffer](handle, "uform_model_load_ram_buffer")
        self.model_load_vram = _resolve[ModelLoadVram](handle, "uform_model_load_vram")
        self.model_load_vram_buffer = _resolve[ModelLoadVramBuffer](handle, "uform_model_load_vram_buffer")
        self.model_describe = _resolve[ModelDescribe](handle, "uform_model_describe")
        self.model_free = _resolve[Release](handle, "uform_model_free")

        self.text_encode = _resolve[TextEncode](handle, "uform_text_encode")
        self.text_choose = _resolve[TextChoose](handle, "uform_text_choose")
        self.image_encode = _resolve[ImageEncode](handle, "uform_image_encode")
        self.image_encode_pixels = _resolve[ImageEncodePixels](handle, "uform_image_encode_pixels")
        self.video_encode = _resolve[VideoEncode](handle, "uform_video_encode")
        self.audio_encode = _resolve[AudioEncode](handle, "uform_audio_encode")
        self.audio_encode_samples = _resolve[AudioEncodeSamples](handle, "uform_audio_encode_samples")

        self.modality_name = _resolve[NameOf](handle, "uform_modality_name")
        self.modality_from_name = _resolve[FromName](handle, "uform_modality_from_name")
        self.objective_name = _resolve[NameOf](handle, "uform_objective_name")
        self.objective_from_name = _resolve[FromName](handle, "uform_objective_from_name")
        self.negatives_name = _resolve[NameOf](handle, "uform_negatives_name")
        self.negatives_from_name = _resolve[FromName](handle, "uform_negatives_from_name")
        self.multimem_tier_name = _resolve[NameOf](handle, "uform_multimem_tier_name")
        self.multimem_tier_from_name = _resolve[FromName](handle, "uform_multimem_tier_from_name")
        self.mesh_bootstrap = _resolve[MeshBootstrap](handle, "uform_mesh_bootstrap")
        self.mesh_describe = _resolve[MeshDescribe](handle, "uform_mesh_describe")
        self.mesh_free = _resolve[Release](handle, "uform_mesh_free")
        self.trainer_init_vram = _resolve[TrainerInitVram](handle, "uform_trainer_init_vram")
        self.trainer_step = _resolve[TrainerStep](handle, "uform_trainer_step")
        self.trainer_save = _resolve[TrainerSave](handle, "uform_trainer_save")
        self.trainer_free = _resolve[Release](handle, "uform_trainer_free")


struct _Loaded:
    """The handle and its symbols, kept together so neither outlives the other."""

    var handle: OwnedDLHandle
    var symbols: Symbols

    def __init__(out self) raises Error:
        try:
            self.handle = OwnedDLHandle(String(LIBRARY_FILE))
        except:
            raise Error(Status.LIBRARY_MISSING, LIBRARY_FILE)
        # Checked first, so a library of another major reports that rather than a missing verb.
        if _version(_resolve[VersionOf](self.handle, "uform_version"))[0] != BINDING_MAJOR:
            raise Error(Status.VERSION_MISMATCH, "uform_version")
        self.symbols = Symbols(self.handle)


struct Library(ImplicitlyCopyable):
    """A reference to the loaded engine, cheap to copy and shared by everything it creates.

    Every handle carries one, which is what keeps the shared object mapped for as long as anything
    can still call into it. It is a loader reference and nothing more: an executor never retains a
    model, and a model never retains an executor.
    """

    var shared: ArcPointer[_Loaded]
    """The handle and its symbols, refcounted so the last holder unmaps the object."""

    def __init__(out self) raises Error:
        """Opens the shared object and resolves it, which every handle then shares.

        Raises:
            With `LIBRARY_MISSING`, when nothing is on the loader path, with `VERSION_MISMATCH`
            when what loaded is another major version, or with `SYMBOL_MISSING` when it does not
            export an entry point the binding needs.
        """
        self.shared = ArcPointer(_Loaded())

    @inline(.always)
    def symbols(self) -> ref[origin_of(self.shared[].symbols)] Symbols:
        """The resolved entry points, borrowed rather than copied on every call.

        Returns:
            A reference that stays valid for as long as this `Library` is held.
        """
        return self.shared[].symbols

    def version(self) -> Tuple[Int, Int, Int]:
        """The loaded engine's major, minor and patch, which may differ from what this binding was
        written against within one major. Calls `uform_version`."""
        return _version(self.symbols().version)

    def version_name(self) -> String:
        """The loaded engine's own version, which is also the cheapest check that loading worked.
        Calls `uform_version_name`."""
        return text_at(self.symbols().version_name())

    def capabilities_compiled(self) -> String:
        """The hardware capabilities this build of the engine was compiled with, comma-separated.
        Calls `uform_capabilities_compiled`."""
        return text_at(self.symbols().capabilities_compiled())

    def capabilities_available(self) -> String:
        """The hardware capabilities this machine offers now, which the build does not imply.
        Calls `uform_capabilities_available`."""
        return text_at(self.symbols().capabilities_available())

    def gpu_devices[devices_origin: MutOrigin, //](self, devices: Span[GPUDevice, devices_origin]) raises Error -> Int:
        """Fills `devices` with every GPU this process can drive, in the ordinal order every VRAM call
        names them by. Calls `uform_gpu_devices`.

        Args:
            devices: Where the listing lands, as many entries as fit; may be empty to only count.

        Returns:
            How many devices this process can drive, which may exceed `len(devices)`.

        Raises:
            `Error` if a vendor runtime is present but refuses the listing.
        """
        var count = stack_allocation[1, c_size_t]()
        count[unsafe_offset=0] = 0
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var listed: Optional[Pointer[GPUDevice, MutUntrackedOrigin]] = None
        if len(devices):
            listed = Pointer[GPUDevice, MutUntrackedOrigin](unsafe_from_address=Int(devices.unsafe_ptr()))
        var arguments = stack_allocation[1, GPUDevicesArguments]()
        arguments[unsafe_offset=0] = GPUDevicesArguments(
            outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            listed,
            c_size_t(len(devices)),
            count.unsafe_origin_cast[MutUntrackedOrigin](),
        )
        self.symbols().gpu_devices(arguments.unsafe_origin_cast[MutAnyOrigin]())
        self._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_gpu_devices")
        return Int(count.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def status_name(self, status: Status) -> String:
        """The engine's spelling of a status, "unrecognized" for a code it does not name. Calls `uform_status_name`."""
        return text_at(self.symbols().status_name(c_int(status.code)))

    def status_from_name(self, name: StringSlice) -> Status:
        """Reads a spelling back; `INVALID_ARGUMENT` when unrecognized. Calls `uform_status_from_name`."""
        return Status.of(self.symbols().status_from_name(BytesView.of(name).arguments()))

    def scalar_name(self, scalar: ScalarKind) -> String:
        """The engine's spelling of a scalar kind, as `F32` spells "f32". Calls `uform_scalar_name`."""
        return text_at(self.symbols().scalar_name(scalar.code))

    def scalar_from_name(self, name: StringSlice) -> ScalarKind:
        """Reads a spelling back; `UNKNOWN` when unrecognized. Calls `uform_scalar_from_name`."""
        return ScalarKind(self.symbols().scalar_from_name(BytesView.of(name).arguments()))

    def mood_name(self, mood: Mood) -> String:
        """The engine's spelling of a mood, as `PARKED` spells "parked". Calls `uform_mood_name`."""
        return text_at(self.symbols().mood_name(mood.code))

    def mood_from_name(self, name: StringSlice) -> Mood:
        """Reads a spelling back; `PARKED` when unrecognized. Calls `uform_mood_from_name`."""
        return Mood(self.symbols().mood_from_name(BytesView.of(name).arguments()))

    def cpu_class_name(self, cpu_class: CPUClass) -> String:
        """The engine's spelling of a CPU class, as `INTEL_HASWELL` spells "intel-haswell". Calls `uform_cpu_class_name`.
        """
        return text_at(self.symbols().cpu_class_name(cpu_class.code))

    def cpu_class_from_name(self, name: StringSlice) -> CPUClass:
        """Reads a spelling back; `UNKNOWN` when unrecognized. Calls `uform_cpu_class_from_name`."""
        return CPUClass(self.symbols().cpu_class_from_name(BytesView.of(name).arguments()))

    def gpu_class_name(self, gpu_class: GPUClass) -> String:
        """The engine's spelling of a GPU class, as `NVIDIA_HOPPER` spells "nvidia-hopper". Calls `uform_gpu_class_name`.
        """
        return text_at(self.symbols().gpu_class_name(gpu_class.code))

    def gpu_class_from_name(self, name: StringSlice) -> GPUClass:
        """Reads a spelling back; `UNKNOWN` when unrecognized. Calls `uform_gpu_class_from_name`."""
        return GPUClass(self.symbols().gpu_class_from_name(BytesView.of(name).arguments()))

    def modality_name(self, modality: Modality) -> String:
        """The engine's spelling of a modality, as `TEXT` spells "text". Calls `uform_modality_name`."""
        return text_at(self.symbols().modality_name(modality.code))

    def modality_from_name(self, name: StringSlice) -> Modality:
        """Reads a spelling back; `TEXT` when unrecognized. Calls `uform_modality_from_name`."""
        return Modality(self.symbols().modality_from_name(BytesView.of(name).arguments()))

    def objective_name(self, objective: Objective) -> String:
        """The engine's spelling of an objective, as `JOINT` spells "joint". Calls `uform_objective_name`."""
        return text_at(self.symbols().objective_name(objective.code))

    def objective_from_name(self, name: StringSlice) -> Objective:
        """Reads a spelling back; `MASKED` when unrecognized. Calls `uform_objective_from_name`."""
        return Objective(self.symbols().objective_from_name(BytesView.of(name).arguments()))

    def negatives_name(self, negatives: Negatives) -> String:
        """The engine's spelling of a negatives policy, as `NAMED` spells "named". Calls `uform_negatives_name`."""
        return text_at(self.symbols().negatives_name(negatives.code))

    def negatives_from_name(self, name: StringSlice) -> Negatives:
        """Reads a spelling back; `AMBIENT` when unrecognized. Calls `uform_negatives_from_name`."""
        return Negatives(self.symbols().negatives_from_name(BytesView.of(name).arguments()))

    def multimem_tier_name(self, tier: MultimemTier) -> String:
        """The engine's spelling of a multimem tier, as `PEER_READ` spells "peer-read". Calls `uform_multimem_tier_name`.
        """
        return text_at(self.symbols().multimem_tier_name(tier.code))

    def multimem_tier_from_name(self, name: StringSlice) -> MultimemTier:
        """Reads a spelling back; `SWITCH_REDUCED` when unrecognized. Calls `uform_multimem_tier_from_name`."""
        return MultimemTier(self.symbols().multimem_tier_from_name(BytesView.of(name).arguments()))

    def _error(self, outcome: Outcome, symbol: StaticString) -> Error:
        """The failure a final outcome reports, its status spelled by the engine's own table."""
        var status = Status.of(outcome.status)
        return Error(status, symbol, outcome.message, self.symbols().status_name(c_int(status.code)), self)

    def _check(self, outcome: Outcome, symbol: StaticString) raises Error:
        """Raises the failure a final outcome reports, and returns on success."""
        if outcome.status != c_int(0):
            raise self._error(outcome, symbol)
