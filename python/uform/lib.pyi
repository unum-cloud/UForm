"""Type stubs for the `uform.lib` extension.

Stub files are never executed, so PEP 695 `type` statements and PEP 688's `Buffer` are read by any checker
regardless of the interpreter running the tests. The package floor is 3.12.

File: python/uform/lib.pyi
Author: Ash Vardanian
Date: October 18, 2026
"""

import os
from collections.abc import Buffer, Callable, Generator, Sequence
from enum import IntEnum, IntFlag
from types import GenericAlias, TracebackType
from typing import Any, Final, NoReturn, Self, final, type_check_only

from _typeshed import structseq

# region: Library Probes

__version__: str

def version() -> str:
    """The library's semantic version. Calls `uform_version_name`."""
    ...

def capabilities_compiled() -> str:
    """The hardware capabilities this binary was compiled with. Calls `uform_capabilities_compiled`."""
    ...

def capabilities_available() -> str:
    """The hardware capabilities this machine offers at runtime. Calls `uform_capabilities_available`."""
    ...

def gpu_devices() -> tuple[GPUDevice, ...]:
    """Every GPU this process can drive, a device's position being its ordinal. Calls `uform_gpu_devices`."""
    ...

# endregion: Library Probes

# region: Shared Vocabulary

# Every input arrives through PEP 3118, so NumPy is never required - at runtime or here.
# Frameworks whose stubs predate PEP 688, PyTorch among them, support the protocol at runtime but
# need a `cast` here.
type _PathLike = str | bytes | os.PathLike[str] | os.PathLike[bytes]
type _Progress = Callable[[int, int], bool | None]
type _Tape = tuple[Buffer, Buffer]
type _Texts = Sequence[str | Buffer] | _Tape
type _Files = Sequence[Buffer] | _Tape
type _Words = Buffer | Sequence[int]

class Status(IntEnum):
    """What a call finished with, valued as `uform_status_t`: `Error.status` and `Deferred.status`."""

    SUCCESS = 0
    DEFERRED = 1
    OUT_OF_MEMORY = 2
    SHAPE_MISMATCH = 3
    UNSUPPORTED_LAYOUT = 4
    UNKNOWN_KIND = 5
    IO = 7
    DEVICE = 8
    MALFORMED = 9
    INTERNAL = 10
    CONFIG_MISMATCH = 13
    CONTENDED = 14
    CANCELLED = 15
    UNSUPPORTED = 16
    INVALID_ARGUMENT = 17

class Scalar(IntEnum):
    """A scalar type a model computes in or its outputs land as, valued as `uform_scalar_t`."""

    F64 = 1
    F32 = 2
    BF16 = 3
    F16 = 4
    E5M2 = 5
    E4M3 = 6
    E3M2 = 7
    E2M3 = 8
    I8 = 10
    I4X2 = 11
    I32 = 12
    U8 = 20
    U4X2 = 21
    U1X8 = 22

class Mood(IntEnum):
    """What an executor does once it goes idle, valued as `uform_mood_t`: sleep, or keep polling."""

    PARKED = 0
    SPINNING = 1

class CPUClass(IntEnum):
    """The CPU families dispatch picks from, valued as `uform_cpu_class_t`; a RAM placement may pin one."""

    SERIAL = 1
    INTEL_HASWELL = 2
    INTEL_SIERRA = 3
    INTEL_SKYLAKE = 4
    INTEL_ICELAKE = 5
    AMD_GENOA = 6
    INTEL_SAPPHIRE = 7
    INTEL_GRANITE = 8
    INTEL_DIAMOND = 9
    AWS_G2 = 10
    APPLE_M2 = 11
    APPLE_M4 = 12
    AWS_G5 = 13
    NVIDIA_VERA = 14
    ALIBABA_C930 = 15
    WASM32_V128 = 16
    WASM64_EXTENDED = 17
    MONOVERSIONED = 18

class GPUClass(IntEnum):
    """The GPU families dispatch picks from, valued as `uform_gpu_class_t`; a VRAM placement may pin one."""

    NVIDIA_AMPERE = 1
    NVIDIA_ADA = 2
    NVIDIA_HOPPER = 3
    NVIDIA_BLACKWELL = 4
    NVIDIA_BLACKWELLRTX = 5
    NVIDIA_RUBIN = 6
    AMD_CDNA3 = 7
    AMD_CDNA4 = 8
    AMD_CDNA5 = 9
    APPLE_METAL7 = 10
    APPLE_METAL9 = 11
    APPLE_METAL10 = 12

class GPUVendor(IntEnum):
    """The vendor whose runtime drives a GPU, valued as `uform_gpu_vendor_t`."""

    NVIDIA = 1
    AMD = 2
    APPLE = 3

@final
class GPUDevice(structseq[Any], tuple[GPUVendor, int, GPUClass | None, int, int, str]):
    """One GPU this process can drive, its position in `gpu_devices()` being its ordinal."""

    __match_args__: Final = ("vendor", "vendor_ordinal", "gpu_class", "compute_units", "memory_bytes", "name")

    @property
    def vendor(self) -> GPUVendor:
        """The vendor whose runtime drives the device."""
        ...

    @property
    def vendor_ordinal(self) -> int:
        """The device's ordinal in its vendor's runtime, as nvidia-smi or rocm-smi list it."""
        ...

    @property
    def gpu_class(self) -> GPUClass | None:
        """The class whose kernels run on it, or None."""
        ...

    @property
    def compute_units(self) -> int:
        """Streaming multiprocessors or compute units."""
        ...

    @property
    def memory_bytes(self) -> int:
        """Device memory in bytes."""
        ...

    @property
    def name(self) -> str:
        """The vendor's own name for the device."""
        ...

class Outputs(IntFlag):
    """Which surfaces an encoding keeps, deciding which of the C output pointers the binding passes."""

    EMBEDDINGS = 1
    STATES = 2
    TOKEN_VECTORS = 4

class Modality(IntEnum):
    """What the bytes of one training item hold, valued as `uform_modality_t`."""

    TEXT = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3

class Objective(IntEnum):
    """Which losses a training step relaxes, valued as `uform_objective_t`."""

    MASKED = 0
    CONTRASTIVE = 1
    JOINT = 2

class Negatives(IntEnum):
    """Which pairs the contrastive loss pushes apart, valued as `uform_negatives_t`."""

    AMBIENT = 0
    NAMED = 1

class PairLossForm(IntEnum):
    """How a contrastive term turns a row of scores into forces, valued as `uform_pair_loss_form_t`."""

    DERIVED = 0
    BALANCED = 1
    RANKED = 2
    SIGMOID = 3

class LateInteractionSide(IntEnum):
    """Which side of a late-interaction score an item takes, valued as `uform_late_interaction_side_t`."""

    NONE = 0
    QUERY = 1
    DOCUMENT = 2
    QUERY_AND_DOCUMENT = 3

class MultimemTier(IntEnum):
    """How a mesh's NVLink domains reduce, valued as `uform_multimem_tier_t`."""

    SWITCH_REDUCED = 0
    PEER_READ = 1

class Error(RuntimeError):
    """The base of every failure the library reports, one subclass per status shared with the C API."""

    status: Status

class OutOfMemoryError(Error, MemoryError):
    """Device, pinned, or host memory could not be allocated, or the scratch ceiling is too low."""

class ShapeMismatchError(Error, ValueError):
    """Item counts, a pixel shape, or a batch envelope disagree with the call or the model."""

class UnsupportedLayoutError(Error, ValueError):
    """A stride or packing the model cannot read."""

class UnknownKindError(Error, ValueError):
    """A scalar, modality, or name this build does not know."""

class InputOutputError(Error, OSError):
    """A checkpoint file could not be opened, read, or written."""

class DeviceError(Error):
    """The driver refused or the device faulted; after a fault the executor refuses every later call."""

class MalformedError(Error, ValueError):
    """Supplied bytes could not be decoded: a checkpoint, an image, a clip, or a recording."""

class InternalError(Error):
    """The library reported neither success nor a failure - a defect."""

class ConfigMismatchError(Error, ValueError):
    """The request contradicts a frozen configuration, like resuming a checkpoint of another geometry."""

class ContendedError(Error):
    """Another call already holds the executor or the arena."""

class CancelledError(Error):
    """A progress handler stopped the call, which published nothing."""

class UnsupportedError(Error, NotImplementedError):
    """The verb is declared, but this residency or model does not implement it."""

class InvalidArgumentError(Error, ValueError):
    """An argument breaks the verb's contract, a closed executor and a rewound arena's output included."""

@final
class Executor:
    """The threads or streams a verb runs on, shared across models and named by `on` and `deferred`."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def for_ram(
        cls,
        threads: int = 0,
        *,
        memory_domains: Sequence[int] | None = None,
        cpu_class: CPUClass | None = None,
    ) -> Executor:
        """Creates an executor of host threads for RAM-resident models. Calls `uform_executor_for_ram`."""
        ...

    @classmethod
    def on_pool(
        cls,
        pool: int,
        /,
        *,
        memory_domains: Sequence[int] | None = None,
        cpu_class: CPUClass | None = None,
    ) -> Executor:
        """Borrows a ForkUnion pool; raises `UnsupportedError` until borrowing is designed. Calls `uform_executor_on_pool`."""
        ...

    @classmethod
    def for_vram(
        cls,
        devices: Sequence[int] | None = None,
        *,
        scratch_limit: int = 0,
        gpu_class: GPUClass | None = None,
        memory_domains: Sequence[int] | None = None,
        cpu_class: CPUClass | None = None,
    ) -> Executor:
        """Opens one submission stream on every named device. Calls `uform_executor_for_vram`."""
        ...

    def drain(self, mood: Mood = ...) -> None:
        """Waits until no call holds the executor, then sets its mood until the next call. Calls `uform_executor_drain`."""
        ...

    def __enter__(self) -> Self: ...
    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None, /
    ) -> None:
        """Closes it once its calls finish, or leaves that to its last live GPU arena. Calls `uform_executor_free`."""
        ...

    def __repr__(self) -> str: ...

@final
class Arena:
    """The memory every output whose length scales with a call lands in, rewound by each call handed it."""

    def __new__(cls, executor: Executor | None = None, /) -> Self:
        """Makes one where `executor` computes, or host pages for the calling thread. Calls `uform_arena_init`."""
        ...

class ArenaBuffer:
    """One output a call left in an `Arena`, readable until a later call rewinds it."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __buffer__(self, flags: int, /) -> memoryview:
        """Host memory only; device memory raises `DeviceError`, a rewound arena `InvalidArgumentError`."""
        ...

    def __len__(self) -> int:
        """Rows. Python-only."""
        ...

    def __dlpack__(
        self,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        """A DLPack capsule over the output in place; the arena refuses to rewind until the consumer lets go."""
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        """The DLPack device type and ID the output lives on."""
        ...

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """Version 3 of the CUDA Array Interface, present only over device memory. Python-only."""
        ...

@final
class RowsTape(ArenaBuffer):
    """A ragged run of rows in an `Arena`, one per token each item kept or one value per choice, split by offsets."""

    @property
    def offsets(self) -> ArenaBuffer:
        """(count + 1,) Apache Arrow offsets into the rows."""
        ...

@final
class Deferred[AnswerT]:
    """A call a `DeferredOn` submitted: `await` it, or `wait()` for it; dropping it waits."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """Names what `wait()` answers, as `Deferred[Encoded]`."""
        ...

    @property
    def status(self) -> Status:
        """`Status.DEFERRED` until the call is final, then the final one. Calls `uform_outcome_status`."""
        ...

    def wait(self) -> AnswerT:
        """Blocks until the call is final, then answers or raises. Calls `uform_outcome_wait`."""
        ...

    def __await__(self) -> Generator[Any, None, AnswerT]:
        """Runs `wait()` on the running event loop's default executor, suspending the task meanwhile."""
        ...

    def __repr__(self) -> str: ...

# endregion: Shared Vocabulary

# region: Models

@final
class ModelDescription(structseq[Any], tuple[int, int, Scalar, int, CPUClass | None, GPUClass | None, int, int]):
    """What a loaded model reports about itself, as `uform_model_description_t` holds it."""

    __match_args__: Final = (
        "dimensions",
        "context",
        "quantization",
        "memory_usage",
        "cpu_class",
        "gpu_class",
        "token_width",
        "choice_temperature_groups",
    )

    @property
    def dimensions(self) -> int:
        """Scalars per embedding row; Matryoshka prefixes of it are embeddings too."""
        ...

    @property
    def context(self) -> int:
        """Tokens one item may span before it is truncated."""
        ...

    @property
    def quantization(self) -> Scalar:
        """The scalar the transformer blocks run in."""
        ...

    @property
    def memory_usage(self) -> int:
        """Bytes the model holds; activations are the executor's."""
        ...

    @property
    def cpu_class(self) -> CPUClass | None:
        """The class a RAM model was packed for, or None for a VRAM model."""
        ...

    @property
    def gpu_class(self) -> GPUClass | None:
        """The class every replica of a VRAM model runs as, or None for a RAM model."""
        ...

    @property
    def token_width(self) -> int:
        """Scalars per token vector; 0 when the checkpoint carries no token head."""
        ...

    @property
    def choice_temperature_groups(self) -> int:
        """Question types `text_choose` was calibrated over: 0 without the byte decoder, 1 when uncalibrated."""
        ...

@final
class Encoded(structseq[Any], tuple[ArenaBuffer | None, RowsTape | None, RowsTape | None]):
    """What an encoding wrote into its arena, unpacking as `embeddings, states, token_vectors = encoded`."""

    __match_args__: Final = ("embeddings", "states", "token_vectors")

    @property
    def embeddings(self) -> ArenaBuffer | None:
        """The (count, dimensions) pooled embeddings, or None when not asked for."""
        ...

    @property
    def states(self) -> RowsTape | None:
        """The (tokens, dimensions) final-norm states and their offsets, or None when not asked for."""
        ...

    @property
    def token_vectors(self) -> RowsTape | None:
        """The (tokens, token_width) unit vectors of the token head, offset as the states, or None."""
        ...

@type_check_only
class _EncodeStats:
    def __new__(cls) -> Self: ...
    @property
    def truncated(self) -> int:
        """Items cut at the model's context."""
        ...

    @property
    def tokens(self) -> int:
        """Tokens the call encoded, after truncation."""
        ...

@final
class TextEncodeStats(_EncodeStats):
    """What `text_encode` and `text_choose` report, overwritten by every call it is passed to."""

@final
class ImageEncodeStats(_EncodeStats):
    """What `image_encode` and `image_encode_pixels` report, overwritten by every call it is passed to."""

@final
class VideoEncodeStats(_EncodeStats):
    """What `video_encode` reports, overwritten by every call it is passed to."""

@final
class AudioEncodeStats(_EncodeStats):
    """What `audio_encode` and `audio_encode_samples` report, overwritten by every call it is passed to."""

@final
class Model:
    """A loaded encoder serving every modality it was trained on; the receiver chooses where its verbs run."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def load_ram(
        cls,
        path: _PathLike,
        /,
        *,
        memory_domains: Sequence[int] | None = None,
        cpu_class: CPUClass | None = None,
        executor: Executor | None = None,
        quantization: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
    ) -> Model:
        """Copies a checkpoint into host memory. Calls `uform_model_load_ram`."""
        ...

    @classmethod
    def load_ram_buffer(
        cls,
        buffer: Buffer,
        /,
        *,
        memory_domains: Sequence[int] | None = None,
        cpu_class: CPUClass | None = None,
        executor: Executor | None = None,
        quantization: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
    ) -> Model:
        """Copies a checkpoint held in memory into host memory. Calls `uform_model_load_ram_buffer`."""
        ...

    @classmethod
    def load_vram(
        cls,
        path: _PathLike,
        /,
        *,
        devices: Sequence[int] | None = None,
        gpu_class: GPUClass | None = None,
        executor: Executor | None = None,
        quantization: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
    ) -> Model:
        """Copies a checkpoint onto the named GPUs. Calls `uform_model_load_vram`."""
        ...

    @classmethod
    def load_vram_buffer(
        cls,
        buffer: Buffer,
        /,
        *,
        devices: Sequence[int] | None = None,
        gpu_class: GPUClass | None = None,
        executor: Executor | None = None,
        quantization: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
    ) -> Model:
        """Copies a checkpoint held in host memory onto the named GPUs. Calls `uform_model_load_vram_buffer`."""
        ...

    def on(self, executor: Executor, /) -> On[Model]:
        """A receiver whose verbs block on `executor`. Python-only."""
        ...

    def deferred(self, executor: Executor, /) -> DeferredOn[Model]:
        """A receiver whose verbs submit to `executor` and answer a `Deferred`. Python-only."""
        ...

    def describe(self) -> ModelDescription:
        """The model's geometry and placement. Calls `uform_model_describe`."""
        ...

    def text_encode(
        self,
        texts: _Texts,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: TextEncodeStats | None = None,
    ) -> Encoded:
        """Encodes UTF-8 texts, a byte per token. Calls `uform_text_encode`."""
        ...

    def image_encode(
        self,
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: ImageEncodeStats | None = None,
    ) -> Encoded:
        """Decodes and encodes JPEG, PNG or WebP files. Calls `uform_image_encode`."""
        ...

    def image_encode_pixels(
        self,
        pixels: Sequence[Buffer],
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: ImageEncodeStats | None = None,
    ) -> Encoded:
        """Encodes [height, width, 3] uint8 images. Calls `uform_image_encode_pixels`."""
        ...

    def video_encode(
        self,
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: VideoEncodeStats | None = None,
    ) -> Encoded:
        """Decodes and encodes MP4 or WebM clips. Calls `uform_video_encode`."""
        ...

    def audio_encode(
        self,
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: AudioEncodeStats | None = None,
    ) -> Encoded:
        """Decodes and encodes WAV, FLAC, MP3 or Opus recordings. Calls `uform_audio_encode`."""
        ...

    def audio_encode_samples(
        self,
        samples: Sequence[Buffer] | _Tape,
        sample_rate: int,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: AudioEncodeStats | None = None,
    ) -> Encoded:
        """Encodes 1-D float32 mono recordings. Calls `uform_audio_encode_samples`."""
        ...

    def text_choose(
        self,
        texts: _Texts,
        choices: _Texts,
        arena: Arena,
        /,
        *,
        temperature_groups: _Words | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: TextEncodeStats | None = None,
    ) -> RowsTape:
        """Scores each text's single-byte choices, one probability each. Calls `uform_text_choose`."""
        ...

    def __repr__(self) -> str: ...

# endregion: Models

# region: Training

@type_check_only
class _Fields:
    def __repr__(self) -> str: ...

@final
class ImageShape(_Fields):
    """The image slice of a `BatchShape`; a zero `patch_side` leaves images out of the run."""

    patch_side: Final[int]
    channels: Final[int]
    def __new__(cls, *, patch_side: int = 0, channels: int = 0) -> Self: ...

@final
class VideoShape(_Fields):
    """The video slice of a `BatchShape`; a zero `patch_side` leaves video out of the run."""

    patch_side: Final[int]
    tube_frames: Final[int]
    frame_stride: Final[int]
    channels: Final[int]
    def __new__(
        cls, *, patch_side: int = 0, tube_frames: int = 0, frame_stride: int = 0, channels: int = 0
    ) -> Self: ...

@final
class AudioShape(_Fields):
    """The audio slice of a `BatchShape`; a zero `chunk_samples` leaves audio out of the run."""

    chunk_samples: Final[int]
    sample_rate: Final[int]
    def __new__(cls, *, chunk_samples: int = 0, sample_rate: int = 0) -> Self: ...

@final
class BatchShape(_Fields):
    """What one device's step carries at most, whatever the modality mix, fixed at creation."""

    max_rows_per_device: Final[int]
    max_segments_per_device: Final[int]
    max_columns_per_row: Final[int]
    max_bags_per_device: Final[int]
    max_choices_per_item: Final[int]
    image: Final[ImageShape | None]
    video: Final[VideoShape | None]
    audio: Final[AudioShape | None]
    def __new__(
        cls,
        *,
        max_rows_per_device: int = 0,
        max_segments_per_device: int = 0,
        max_columns_per_row: int = 0,
        max_bags_per_device: int = 0,
        max_choices_per_item: int = 0,
        image: ImageShape | tuple[int, ...] | None = None,
        video: VideoShape | tuple[int, ...] | None = None,
        audio: AudioShape | tuple[int, ...] | None = None,
    ) -> Self: ...

@final
class Masking(_Fields):
    """How the masked objective corrupts a batch and weighs its heads; every zero derives."""

    token_mask_probability: Final[float]
    token_mean_span: Final[float]
    replace_fraction: Final[float]
    random_fraction: Final[float]
    image_mask_ratio: Final[float]
    video_mask_ratio: Final[float]
    audio_mask_ratio: Final[float]
    text_weight: Final[float]
    image_weight: Final[float]
    video_weight: Final[float]
    audio_weight: Final[float]
    def __new__(
        cls,
        *,
        token_mask_probability: float = 0.0,
        token_mean_span: float = 0.0,
        replace_fraction: float = 0.0,
        random_fraction: float = 0.0,
        image_mask_ratio: float = 0.0,
        video_mask_ratio: float = 0.0,
        audio_mask_ratio: float = 0.0,
        text_weight: float = 0.0,
        image_weight: float = 0.0,
        video_weight: float = 0.0,
        audio_weight: float = 0.0,
    ) -> Self: ...

@final
class Contrastive(_Fields):
    """How the contrastive loss scores the pooled embeddings; every zero derives."""

    matryoshka_dimensions: Final[Sequence[int] | None]
    temperature_groups: Final[int]
    inverse_temperature: Final[float]
    bias: Final[float]
    negatives: Final[Negatives]
    ambient_repulsion_weight: Final[float]
    max_edges_per_device: Final[int]
    pair_loss: Final[PairLossForm]
    def __new__(
        cls,
        *,
        matryoshka_dimensions: Sequence[int] | None = None,
        temperature_groups: int = 0,
        inverse_temperature: float = 0.0,
        bias: float = 0.0,
        negatives: Negatives = ...,
        ambient_repulsion_weight: float = 0.0,
        max_edges_per_device: int = 0,
        pair_loss: PairLossForm = ...,
    ) -> Self: ...

@final
class LateInteraction(_Fields):
    """How the late-interaction loss scores the token vectors of queries and documents; every zero derives."""

    weight: Final[float]
    width: Final[int]
    inverse_temperature: Final[float]
    bias: Final[float]
    max_query_span: Final[int]
    max_query_tokens_per_device: Final[int]
    max_documents_per_device: Final[int]
    max_document_tokens_per_device: Final[int]
    pair_loss: Final[PairLossForm]
    def __new__(
        cls,
        *,
        weight: float = 0.0,
        width: int = 0,
        inverse_temperature: float = 0.0,
        bias: float = 0.0,
        max_query_span: int = 0,
        max_query_tokens_per_device: int = 0,
        max_documents_per_device: int = 0,
        max_document_tokens_per_device: int = 0,
        pair_loss: PairLossForm = ...,
    ) -> Self: ...

@final
class Choices(_Fields):
    """Calibrated choices over the byte decoder, read when the weight is positive; every zero derives."""

    weight: Final[float]
    brier_weight: Final[float]
    def __new__(cls, *, weight: float = 0.0, brier_weight: float = 0.0) -> Self: ...

@final
class Schedule(_Fields):
    """The learning-rate schedule: a linear warmup to the peak, then a cosine decay to the floor."""

    learning_rate: Final[float]
    min_learning_rate: Final[float]
    warmup_steps: Final[int]
    steps: Final[int]
    def __new__(
        cls, *, learning_rate: float = 0.0, min_learning_rate: float = 0.0, warmup_steps: int = 0, steps: int = 0
    ) -> Self: ...

@final
class MeshOptions(_Fields):
    """Where this process sits among the processes training one model, and how they meet."""

    host_domain_index: Final[int]
    host_domain_count: Final[int]
    gpus_per_multimem_domain: Final[int]
    directory: Final[_PathLike | None]
    job: Final[str | None]
    timeout_seconds: Final[float]
    def __new__(
        cls,
        *,
        host_domain_index: int = 0,
        host_domain_count: int = 0,
        gpus_per_multimem_domain: int = 0,
        directory: _PathLike | None = None,
        job: str | None = None,
        timeout_seconds: float = 0.0,
    ) -> Self: ...

@final
class Edges(_Fields):
    """One edge set over a step's items, as compressed sparse rows of buffers borrowed until the step is final."""

    offsets: Final[Buffer | None]
    ids: Final[Buffer | None]
    weights: Final[Buffer | None]
    def __new__(
        cls, *, offsets: Buffer | None = None, ids: Buffer | None = None, weights: Buffer | None = None
    ) -> Self: ...

type _EdgesLike = Edges | tuple[Buffer, Buffer] | tuple[Buffer, Buffer, Buffer] | None

@final
class Relations(_Fields):
    """The contrastive targets of one step: ignored, attracted, repelled, then the run's `negatives`."""

    attract: Final[Edges | None]
    repel: Final[Edges | None]
    ignore: Final[Edges | None]
    def __new__(cls, *, attract: _EdgesLike = None, repel: _EdgesLike = None, ignore: _EdgesLike = None) -> Self: ...

@final
class MeshDescription(structseq[Any], tuple[int, int, int, int, int, MultimemTier, GPUClass | None]):
    """What a mesh reports about itself, as `uform_mesh_description_t` holds it."""

    __match_args__: Final = (
        "gpus_per_host_domain",
        "gpus_per_multimem_domain",
        "gpus_per_colocated_domain",
        "host_domain_index",
        "first_gpu_index",
        "multimem_tier",
        "gpu_class",
    )

    @property
    def gpus_per_host_domain(self) -> int:
        """GPUs one process drives."""
        ...

    @property
    def gpus_per_multimem_domain(self) -> int:
        """GPUs one NVLink domain spans."""
        ...

    @property
    def gpus_per_colocated_domain(self) -> int:
        """GPUs in the whole run."""
        ...

    @property
    def host_domain_index(self) -> int:
        """This process's index among the run's processes."""
        ...

    @property
    def first_gpu_index(self) -> int:
        """The run-wide index of this process's first GPU."""
        ...

    @property
    def multimem_tier(self) -> MultimemTier:
        """How its NVLink domains reduce."""
        ...

    @property
    def gpu_class(self) -> GPUClass | None:
        """The class every GPU of the run dispatches as, or None."""
        ...

@final
class Mesh:
    """One process's place among the processes of a run, borrowed by every trainer."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def bootstrap(cls, executor: Executor, options: MeshOptions | None, /) -> Mesh:
        """Joins this process to the run's mesh. Calls `uform_mesh_bootstrap`."""
        ...

    def describe(self) -> MeshDescription:
        """The mesh's shape and this process's place in it. Calls `uform_mesh_describe`."""
        ...

@final
class TrainerStepStats:
    """What `step` reports, overwritten by every call it is passed to, every loss its world mean."""

    def __new__(cls) -> Self: ...
    @property
    def step(self) -> int:
        """The optimizer step the call applied, counted from one."""
        ...

    @property
    def learning_rate(self) -> float:
        """The learning rate it applied."""
        ...

    @property
    def loss(self) -> float:
        """The total loss the step minimized, its world mean."""
        ...

    @property
    def contrastive_loss(self) -> float:
        """The contrastive term; 0 when absent."""
        ...

    @property
    def masked_text_loss(self) -> float:
        """The text reconstruction term, unweighted."""
        ...

    @property
    def masked_image_loss(self) -> float:
        """The image reconstruction term, unweighted."""
        ...

    @property
    def masked_video_loss(self) -> float:
        """The video reconstruction term, unweighted."""
        ...

    @property
    def masked_audio_loss(self) -> float:
        """The audio reconstruction term, unweighted."""
        ...

    @property
    def late_interaction_loss(self) -> float:
        """The late-interaction term, weighted."""
        ...

    @property
    def choices_loss(self) -> float:
        """The choices term over the items that made a choice, unweighted."""
        ...

    @property
    def truncated(self) -> int:
        """Items cut at a row's width."""
        ...

    @property
    def dropped(self) -> int:
        """Segments dropped whole because the batch had no room for them, since the last stats read."""
        ...

@final
class Trainer:
    """A training run on the devices of the executor it was created on, which its own verbs block on."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def init_vram(
        cls,
        executor: Executor,
        /,
        *,
        mesh: Mesh | None = None,
        resume: _PathLike | None = None,
        depth: int = 0,
        hidden: int = 0,
        batch: BatchShape | None = None,
        objective: Objective = ...,
        masking: Masking | None = None,
        contrastive: Contrastive | None = None,
        late_interaction: LateInteraction | None = None,
        choices: Choices | None = None,
        schedule: Schedule | None = None,
        seed: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
    ) -> Trainer:
        """Builds one replica per device and starts or resumes the run. Calls `uform_trainer_init_vram`."""
        ...

    def on(self, executor: Executor, /) -> On[Trainer]:
        """A receiver whose verbs block on `executor`. Python-only."""
        ...

    def deferred(self, executor: Executor, /) -> DeferredOn[Trainer]:
        """A receiver whose verbs submit to `executor` and answer a `Deferred`. Python-only."""
        ...

    def step(
        self,
        modalities: Sequence[Modality],
        contents: _Texts,
        /,
        *,
        bags: _Words | None = None,
        temperature_groups: _Words | None = None,
        sides: Sequence[LateInteractionSide] | None = None,
        choices: _Texts | None = None,
        choice_targets: Sequence[Buffer] | _Tape | None = None,
        relations: Relations | None = None,
        arena: Arena | None = None,
        outputs_scalar: Scalar = ...,
        stats: TrainerStepStats | None = None,
    ) -> ArenaBuffer | None:
        """Applies one optimizer step over this process's items and every peer's. Calls `uform_trainer_step`."""
        ...

    def save(self, path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0) -> None:
        """Writes the weights and the step as a checkpoint. Calls `uform_trainer_save`."""
        ...

# endregion: Training

# region: Receivers

@final
class On[TargetT]:
    """A model or trainer bound to the executor its verbs block on, as `model.on(executor)` answers it."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """Names the target the receiver addresses, as `On[Model]`."""
        ...

    def describe(self: On[Model]) -> ModelDescription: ...
    def text_encode(
        self: On[Model],
        texts: _Texts,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: TextEncodeStats | None = None,
    ) -> Encoded: ...
    def image_encode(
        self: On[Model],
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: ImageEncodeStats | None = None,
    ) -> Encoded: ...
    def image_encode_pixels(
        self: On[Model],
        pixels: Sequence[Buffer],
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: ImageEncodeStats | None = None,
    ) -> Encoded: ...
    def video_encode(
        self: On[Model],
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: VideoEncodeStats | None = None,
    ) -> Encoded: ...
    def audio_encode(
        self: On[Model],
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: AudioEncodeStats | None = None,
    ) -> Encoded: ...
    def audio_encode_samples(
        self: On[Model],
        samples: Sequence[Buffer] | _Tape,
        sample_rate: int,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: AudioEncodeStats | None = None,
    ) -> Encoded: ...
    def text_choose(
        self: On[Model],
        texts: _Texts,
        choices: _Texts,
        arena: Arena,
        /,
        *,
        temperature_groups: _Words | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: TextEncodeStats | None = None,
    ) -> RowsTape: ...
    def step(
        self: On[Trainer],
        modalities: Sequence[Modality],
        contents: _Texts,
        /,
        *,
        bags: _Words | None = None,
        temperature_groups: _Words | None = None,
        sides: Sequence[LateInteractionSide] | None = None,
        choices: _Texts | None = None,
        choice_targets: Sequence[Buffer] | _Tape | None = None,
        relations: Relations | None = None,
        arena: Arena | None = None,
        outputs_scalar: Scalar = ...,
        stats: TrainerStepStats | None = None,
    ) -> ArenaBuffer | None: ...
    def save(
        self: On[Trainer], path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> None: ...

@final
class DeferredOn[TargetT]:
    """A model or trainer bound to the executor its verbs submit to, each answering a `Deferred` at once."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """Names the target the receiver addresses, as `DeferredOn[Model]`."""
        ...

    def describe(self: DeferredOn[Model]) -> Deferred[ModelDescription]: ...
    def text_encode(
        self: DeferredOn[Model],
        texts: _Texts,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: TextEncodeStats | None = None,
    ) -> Deferred[Encoded]: ...
    def image_encode(
        self: DeferredOn[Model],
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: ImageEncodeStats | None = None,
    ) -> Deferred[Encoded]: ...
    def image_encode_pixels(
        self: DeferredOn[Model],
        pixels: Sequence[Buffer],
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: ImageEncodeStats | None = None,
    ) -> Deferred[Encoded]: ...
    def video_encode(
        self: DeferredOn[Model],
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: VideoEncodeStats | None = None,
    ) -> Deferred[Encoded]: ...
    def audio_encode(
        self: DeferredOn[Model],
        files: _Files,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: AudioEncodeStats | None = None,
    ) -> Deferred[Encoded]: ...
    def audio_encode_samples(
        self: DeferredOn[Model],
        samples: Sequence[Buffer] | _Tape,
        sample_rate: int,
        arena: Arena,
        /,
        *,
        outputs: Outputs = ...,
        outputs_scalar: Scalar = ...,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: AudioEncodeStats | None = None,
    ) -> Deferred[Encoded]: ...
    def text_choose(
        self: DeferredOn[Model],
        texts: _Texts,
        choices: _Texts,
        arena: Arena,
        /,
        *,
        temperature_groups: _Words | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: TextEncodeStats | None = None,
    ) -> Deferred[RowsTape]: ...
    def step(
        self: DeferredOn[Trainer],
        modalities: Sequence[Modality],
        contents: _Texts,
        /,
        *,
        bags: _Words | None = None,
        temperature_groups: _Words | None = None,
        sides: Sequence[LateInteractionSide] | None = None,
        choices: _Texts | None = None,
        choice_targets: Sequence[Buffer] | _Tape | None = None,
        relations: Relations | None = None,
        arena: Arena | None = None,
        outputs_scalar: Scalar = ...,
        stats: TrainerStepStats | None = None,
    ) -> Deferred[ArenaBuffer | None]: ...
    def save(
        self: DeferredOn[Trainer], path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Deferred[None]: ...

# endregion: Receivers
