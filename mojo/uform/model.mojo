"""The encoder: its loads, its description, its receivers, and the verbs every receiver offers.

A `Model` owns its weights and their packed copies, placed once by its load's residency, and is
shared by every executor of that residency. The receiver picks where a verb runs and how it
completes: verbs on the model run on the calling thread, `model.on(executor)` blocks on an
executor, and `model.deferred(executor)` submits there and answers a `Deferred` at once. Each verb
is one `Call` all three route, written once in `ModelVerbs`.

File: mojo/uform/model.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from std.ffi import c_int, c_size_t
from std.memory import stack_allocation

from uform.encode import (
    EncodeConfig,
    Encoded,
    Outputs,
    TextChooseConfig,
    _AudioEncodeSamplesCall,
    _EncodeCall,
    _ImageEncodePixelsCall,
    _TextChooseCall,
)
from uform.library import Library
from uform.types import (
    Address,
    Arena,
    BytesView,
    BytesViewArguments,
    Call,
    Completion,
    CPUClass,
    Deferred,
    Error,
    Executor,
    GPUClass,
    NULL_ADDRESS,
    Outcome,
    Progress,
    RamOptions,
    RamOptionsArguments,
    ScalarKind,
    Tape,
    TapeView,
    Verb,
    VramOptions,
    VramOptionsArguments,
    _block,
    _cross,
)

# region Description


@fieldwise_init
struct ModelDescription(Defaultable, ImplicitlyCopyable):
    """What a loaded model reports about itself, `uform_model_description_t`."""

    var dimensions: c_size_t
    """Scalars per embedding row; Matryoshka prefixes of it are embeddings too."""
    var context: c_size_t
    """Tokens one item may span before it is truncated: bytes of text, patches, tubes or frames."""
    var quantization: ScalarKind
    """The scalar the transformer blocks run in."""
    var memory_usage: c_size_t
    """Bytes the model holds; activations are the executor's."""
    var cpu_class: CPUClass
    """The class a RAM model's tower was packed for; `UNKNOWN` for a VRAM model."""
    var gpu_class: GPUClass
    """The class every replica of a VRAM model runs as; `UNKNOWN` for a RAM model."""
    var token_width: c_size_t
    """Scalars per token vector; zero when the checkpoint carries no token head."""
    var choice_temperature_groups: c_size_t
    """Question types the choice temperatures were calibrated over; zero without the byte decoder."""

    def __init__(out self):
        """Zeroes every field, which is what a failed report leaves."""
        self = Self(0, 0, ScalarKind.UNKNOWN, 0, CPUClass.UNKNOWN, GPUClass.UNKNOWN, 0, 0)


# endregion Description

# region Arguments


@fieldwise_init
struct ModelLoadRamArguments(ImplicitlyCopyable):
    """`uform_model_load_ram_t`, and `uform_model_load_ram_buffer_t` with `source` the buffer."""

    var model: Optional[Pointer[Address, MutUntrackedOrigin]]
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var source: BytesViewArguments
    var ram: RamOptionsArguments
    var quantization: ScalarKind
    var progress: Progress


@fieldwise_init
struct ModelLoadVramArguments(ImplicitlyCopyable):
    """`uform_model_load_vram_t`, and `uform_model_load_vram_buffer_t` with `source` the buffer."""

    var model: Optional[Pointer[Address, MutUntrackedOrigin]]
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var source: BytesViewArguments
    var vram: VramOptionsArguments
    var quantization: ScalarKind
    var progress: Progress


@fieldwise_init
struct ModelDescribeArguments(ImplicitlyCopyable):
    """`uform_model_describe_t`."""

    var model: Address
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var description: Optional[Pointer[ModelDescription, MutUntrackedOrigin]]


comptime ModelLoadRam = Verb[ModelLoadRamArguments]
"""`uform_model_load_ram`."""
comptime ModelLoadRamBuffer = Verb[ModelLoadRamArguments]
"""`uform_model_load_ram_buffer`."""
comptime ModelLoadVram = Verb[ModelLoadVramArguments]
"""`uform_model_load_vram`."""
comptime ModelLoadVramBuffer = Verb[ModelLoadVramArguments]
"""`uform_model_load_vram_buffer`."""
comptime ModelDescribe = Verb[ModelDescribeArguments]
"""`uform_model_describe`."""

# endregion Arguments

# region Calls


@fieldwise_init
struct _ModelDescribeCall(Call):
    """The crossing of `uform_model_describe`."""

    comptime Outputs = ModelDescription
    comptime Result = ModelDescription
    comptime symbol: StaticString = "uform_model_describe"

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[ModelDescription, MutUntrackedOrigin],
    ):
        _cross(
            ModelDescribeArguments(handle, executor, outcome, completion, outputs), library.symbols().model_describe
        )

    def conclude(deinit self, outputs: ModelDescription, generation: Int) -> ModelDescription:
        return outputs


# endregion Calls

# region Model


struct Model(ModelVerbs, Movable):
    """A loaded encoder, released when it goes out of scope.

    Verbs called on the model run on the calling thread; `model.on(executor)` runs the same verbs
    blocking on an executor, and `model.deferred(executor)` submits them there. One model serves any
    number of executors of its residency, whose calls take turns on its session.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `uform_model_t`."""

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle the engine just created."""
        self.library = library
        self.address = address

    @staticmethod
    def load_ram[
        path_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        quantization: ScalarKind = ScalarKind.BF16,
        progress: Progress = Progress(),
    ) raises Error -> Self:
        """Copies a checkpoint file into host memory and packs its products for the CPU.
        Calls `uform_model_load_ram`.

        Args:
            executor: The executor the load runs on; it also lends its engine.
            path: A safetensors checkpoint, read in full and closed before the load completes.
            ram: The memory domains to place the weights in.
            quantization: `BF16` as the checkpoint stores it, or `E4M3` quantized at load.
            progress: Reports the tensors placed, and stops the load where it says to.

        Raises:
            `Error` if the checkpoint cannot be read or decoded.
        """
        return Self._load_ram(
            executor.library,
            executor.address,
            executor.library.symbols().model_load_ram,
            BytesView.of(path),
            ram,
            quantization,
            progress,
            "uform_model_load_ram",
        )

    @staticmethod
    def load_ram_buffer[
        buffer_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        quantization: ScalarKind = ScalarKind.BF16,
        progress: Progress = Progress(),
    ) raises Error -> Self:
        """Copies a checkpoint held in memory into host memory and packs its products for the CPU.
        Calls `uform_model_load_ram_buffer`.

        Raises:
            `Error` if the bytes cannot be decoded.
        """
        return Self._load_ram(
            executor.library,
            executor.address,
            executor.library.symbols().model_load_ram_buffer,
            BytesView.of(buffer),
            ram,
            quantization,
            progress,
            "uform_model_load_ram_buffer",
        )

    @staticmethod
    def _load_ram[
        source_origin: ImmOrigin, ram_origin: ImmOrigin
    ](
        library: Library,
        executor: Address,
        verb: ModelLoadRam,
        source: BytesView[source_origin],
        ram: RamOptions[ram_origin],
        quantization: ScalarKind,
        progress: Progress,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ModelLoadRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source.arguments(),
                ram.arguments(),
                quantization,
                progress,
            ),
            verb,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_vram[
        path_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        quantization: ScalarKind = ScalarKind.BF16,
        progress: Progress = Progress(),
    ) raises Error -> Self:
        """Copies a checkpoint file onto the named GPUs, replicated on each, and packs its products there.
        Calls `uform_model_load_vram`.

        Raises:
            `Error` if the checkpoint cannot be read, or with `UNSUPPORTED` in a build without a GPU.
        """
        return Self._load_vram(
            executor.library,
            executor.address,
            executor.library.symbols().model_load_vram,
            BytesView.of(path),
            vram,
            quantization,
            progress,
            "uform_model_load_vram",
        )

    @staticmethod
    def load_vram_buffer[
        buffer_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        quantization: ScalarKind = ScalarKind.BF16,
        progress: Progress = Progress(),
    ) raises Error -> Self:
        """Copies a checkpoint held in host or device memory onto the named GPUs and packs its products
        there. Calls `uform_model_load_vram_buffer`.

        Raises:
            `Error` if the bytes cannot be decoded, or with `UNSUPPORTED` in a build without a GPU.
        """
        return Self._load_vram(
            executor.library,
            executor.address,
            executor.library.symbols().model_load_vram_buffer,
            BytesView.of(buffer),
            vram,
            quantization,
            progress,
            "uform_model_load_vram_buffer",
        )

    @staticmethod
    def _load_vram[
        source_origin: ImmOrigin, vram_origin: ImmOrigin
    ](
        library: Library,
        executor: Address,
        verb: ModelLoadVram,
        source: BytesView[source_origin],
        vram: VramOptions[vram_origin],
        quantization: ScalarKind,
        progress: Progress,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ModelLoadVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source.arguments(),
                vram.arguments(),
                quantization,
                progress,
            ),
            verb,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def __deinit__(deinit self):
        """Frees the model; every call using it must be final. Calls `uform_model_free`.

        The loader reference is surrendered only after the call, so Mojo cannot `dlclose` the engine
        between the load of the entry point and the branch to it.
        """
        self.library.symbols().model_free(self.address)
        _ = self.library^

    def on(ref self, ref executor: Executor) -> ModelOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, blocking on `executor`, `CONTENDED` while another call holds it. Binding-only.

        Returns:
            A receiver borrowing both, which neither may outlive.
        """
        return ModelOn[ImmOrigin(origin_of(self, executor))](
            Pointer[Model, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def deferred(ref self, ref executor: Executor) -> ModelDeferredOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, each submitted to `executor` and answering a `Deferred` at once. Binding-only.

        Returns:
            A receiver borrowing both, which neither it nor any `Deferred` it answers may outlive.
        """
        return ModelDeferredOn[ImmOrigin(origin_of(self, executor))](
            Pointer[Model, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the calling thread."""
        return _block(self.library, call^, self.address, NULL_ADDRESS)


# endregion Model

# region Verbs


trait ModelVerbs:
    """The verbs every blocking receiver offers, the model on the calling thread and `model.on(executor)`
    on an executor, each written once over the receiver's `_route`."""

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` to its final status on this receiver's executor and answers its result."""
        ...

    def describe(self) raises Error -> ModelDescription:
        """Reports the model's geometry and placement in one call. Calls `uform_model_describe`."""
        return self._route(_ModelDescribeCall())

    def text_encode[
        texts_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        texts: TapeView[texts_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) raises Error -> Encoded[origin_of(arena)]:
        """Encodes UTF-8 texts through one forward, a token per byte, keeping the surfaces `outputs` names.
        Calls `uform_text_encode`.

        Args:
            texts: UTF-8 texts, one item per entry, in bytes.
            outputs: The surfaces to keep from the forward.
            arena: The arena to rewind and refill, which the result borrows.
            config: The scalar every output lands as, and the report slot.

        Raises:
            `Error` if the shapes disagree or the model lacks a kept surface.
        """
        return self._route(
            _EncodeCall[origin_of(arena), texts_origin, stats_origin, "uform_text_encode"](
                texts, outputs, Pointer(to=arena), config
            )
        )

    def text_choose[
        texts_origin: ImmOrigin, choices_origin: ImmOrigin, groups_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        texts: TapeView[texts_origin],
        choices: TapeView[choices_origin],
        mut arena: Arena,
        config: TextChooseConfig[groups_origin, stats_origin] = TextChooseConfig(),
    ) raises Error -> Tape[origin_of(arena)]:
        """Scores each item's legal choices through one forward, a softmax over them alone at the item's
        group's calibrated inverse temperature, answering one `F32` probability per choice.
        Calls `uform_text_choose`.

        Args:
            texts: UTF-8 states, one per item, each ending where its choice goes.
            choices: Each item's legal choices, one byte each.
            arena: The arena to rewind and refill, which the result borrows.
            config: The question type of each item, and the report slot.

        Raises:
            `Error` with `UNSUPPORTED` for a model without the byte decoder.
        """
        return self._route(
            _TextChooseCall[origin_of(arena), texts_origin, choices_origin, groups_origin, stats_origin](
                texts, choices, Pointer(to=arena), config
            )
        )

    def image_encode[
        images_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        images: TapeView[images_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) raises Error -> Encoded[origin_of(arena)]:
        """Decodes and encodes JPEG, PNG or WebP files, one item per entry. Calls `uform_image_encode`.

        Raises:
            `Error` with `MALFORMED` for an undecodable file.
        """
        return self._route(
            _EncodeCall[origin_of(arena), images_origin, stats_origin, "uform_image_encode"](
                images, outputs, Pointer(to=arena), config
            )
        )

    def image_encode_pixels[
        pixels_origin: ImmOrigin, heights_origin: ImmOrigin, widths_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        pixels: TapeView[pixels_origin],
        heights: Span[UInt32, heights_origin],
        widths: Span[UInt32, widths_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) raises Error -> Encoded[origin_of(arena)]:
        """Encodes decoded interleaved 8-bit RGB images, resizing on the executor's side.
        Calls `uform_image_encode_pixels`.

        Raises:
            `Error` with `SHAPE_MISMATCH` when an entry is not `heights[i] × widths[i] × 3` bytes.
        """
        return self._route(
            _ImageEncodePixelsCall[
                origin_of(arena), pixels_origin, heights_origin, widths_origin, stats_origin
            ](pixels, heights, widths, outputs, Pointer(to=arena), config)
        )

    def video_encode[
        videos_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        videos: TapeView[videos_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) raises Error -> Encoded[origin_of(arena)]:
        """Decodes and encodes MP4 or WebM clips, one item per entry. Calls `uform_video_encode`.

        Raises:
            `Error` with `MALFORMED` for an undecodable clip.
        """
        return self._route(
            _EncodeCall[origin_of(arena), videos_origin, stats_origin, "uform_video_encode"](
                videos, outputs, Pointer(to=arena), config
            )
        )

    def audio_encode[
        recordings_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        recordings: TapeView[recordings_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) raises Error -> Encoded[origin_of(arena)]:
        """Decodes and encodes WAV, FLAC, MP3 or Opus recordings, one item per entry. Calls `uform_audio_encode`.

        Raises:
            `Error` with `MALFORMED` for an undecodable recording.
        """
        return self._route(
            _EncodeCall[origin_of(arena), recordings_origin, stats_origin, "uform_audio_encode"](
                recordings, outputs, Pointer(to=arena), config
            )
        )

    def audio_encode_samples[
        samples_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        samples: TapeView[samples_origin],
        sample_rate: Int,
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) raises Error -> Encoded[origin_of(arena)]:
        """Encodes decoded mono `F32` recordings sharing one `sample_rate`, resampled to the model's.
        Calls `uform_audio_encode_samples`.

        Raises:
            `Error` if the shapes disagree.
        """
        return self._route(
            _AudioEncodeSamplesCall[origin_of(arena), samples_origin, stats_origin](
                samples, sample_rate, outputs, Pointer(to=arena), config
            )
        )


struct ModelOn[origin: ImmOrigin](ImplicitlyCopyable, ModelVerbs):
    """A model bound to the executor its verbs block on, as `model.on(executor)` answers it. Binding-only.

    Parameters:
        origin: The model and the executor, which this receiver borrows.
    """

    var model: Pointer[Model, Self.origin]
    """The model every verb addresses."""
    var executor: Address
    """The `uform_executor_t` every verb runs on."""

    def __init__(out self, model: Pointer[Model, Self.origin], executor: Address):
        """Binds a model to an executor."""
        self.model = model
        self.executor = executor

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the bound executor, raising `CONTENDED` while another call holds it."""
        return _block(self.model[].library, call^, self.model[].address, self.executor)


struct ModelDeferredOn[origin: ImmOrigin](ImplicitlyCopyable):
    """A model bound to the executor its verbs are submitted to, as `model.deferred(executor)` answers it:
    every verb answers a `Deferred` at once. Binding-only.

    The verbs and their arguments are those of `ModelVerbs`; the arena stays borrowed until final.

    Parameters:
        origin: The model and the executor, which this receiver borrows.
    """

    var model: Pointer[Model, Self.origin]
    """The model every verb addresses."""
    var executor: Address
    """The `uform_executor_t` every verb is submitted to."""

    def __init__(out self, model: Pointer[Model, Self.origin], executor: Address):
        """Binds a model to an executor."""
        self.model = model
        self.executor = executor

    def describe(self) -> Deferred[_ModelDescribeCall, Self.origin]:
        """Submits `ModelVerbs.describe`. Calls `uform_model_describe`."""
        return {self.model[].library, _ModelDescribeCall(), self.model[].address, self.executor}

    def text_encode[
        texts_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        texts: TapeView[texts_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) -> Deferred[
        _EncodeCall[origin_of(arena), texts_origin, stats_origin, "uform_text_encode"], Self.origin
    ]:
        """Submits `ModelVerbs.text_encode`. Calls `uform_text_encode`."""
        return {
            self.model[].library,
            _EncodeCall[origin_of(arena), texts_origin, stats_origin, "uform_text_encode"](
                texts, outputs, Pointer(to=arena), config
            ),
            self.model[].address,
            self.executor,
        }

    def text_choose[
        texts_origin: ImmOrigin, choices_origin: ImmOrigin, groups_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        texts: TapeView[texts_origin],
        choices: TapeView[choices_origin],
        mut arena: Arena,
        config: TextChooseConfig[groups_origin, stats_origin] = TextChooseConfig(),
    ) -> Deferred[
        _TextChooseCall[origin_of(arena), texts_origin, choices_origin, groups_origin, stats_origin],
        Self.origin,
    ]:
        """Submits `ModelVerbs.text_choose`. Calls `uform_text_choose`."""
        return {
            self.model[].library,
            _TextChooseCall[origin_of(arena), texts_origin, choices_origin, groups_origin, stats_origin](
                texts, choices, Pointer(to=arena), config
            ),
            self.model[].address,
            self.executor,
        }

    def image_encode[
        images_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        images: TapeView[images_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) -> Deferred[
        _EncodeCall[origin_of(arena), images_origin, stats_origin, "uform_image_encode"], Self.origin
    ]:
        """Submits `ModelVerbs.image_encode`. Calls `uform_image_encode`."""
        return {
            self.model[].library,
            _EncodeCall[origin_of(arena), images_origin, stats_origin, "uform_image_encode"](
                images, outputs, Pointer(to=arena), config
            ),
            self.model[].address,
            self.executor,
        }

    def image_encode_pixels[
        pixels_origin: ImmOrigin, heights_origin: ImmOrigin, widths_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        pixels: TapeView[pixels_origin],
        heights: Span[UInt32, heights_origin],
        widths: Span[UInt32, widths_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) -> Deferred[
        _ImageEncodePixelsCall[
            origin_of(arena), pixels_origin, heights_origin, widths_origin, stats_origin
        ],
        Self.origin,
    ]:
        """Submits `ModelVerbs.image_encode_pixels`. Calls `uform_image_encode_pixels`."""
        return {
            self.model[].library,
            _ImageEncodePixelsCall[
                origin_of(arena), pixels_origin, heights_origin, widths_origin, stats_origin
            ](pixels, heights, widths, outputs, Pointer(to=arena), config),
            self.model[].address,
            self.executor,
        }

    def video_encode[
        videos_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        videos: TapeView[videos_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) -> Deferred[
        _EncodeCall[origin_of(arena), videos_origin, stats_origin, "uform_video_encode"], Self.origin
    ]:
        """Submits `ModelVerbs.video_encode`. Calls `uform_video_encode`."""
        return {
            self.model[].library,
            _EncodeCall[origin_of(arena), videos_origin, stats_origin, "uform_video_encode"](
                videos, outputs, Pointer(to=arena), config
            ),
            self.model[].address,
            self.executor,
        }

    def audio_encode[
        recordings_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        recordings: TapeView[recordings_origin],
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) -> Deferred[
        _EncodeCall[origin_of(arena), recordings_origin, stats_origin, "uform_audio_encode"],
        Self.origin,
    ]:
        """Submits `ModelVerbs.audio_encode`. Calls `uform_audio_encode`."""
        return {
            self.model[].library,
            _EncodeCall[origin_of(arena), recordings_origin, stats_origin, "uform_audio_encode"](
                recordings, outputs, Pointer(to=arena), config
            ),
            self.model[].address,
            self.executor,
        }

    def audio_encode_samples[
        samples_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        samples: TapeView[samples_origin],
        sample_rate: Int,
        outputs: Outputs,
        mut arena: Arena,
        config: EncodeConfig[stats_origin] = EncodeConfig(),
    ) -> Deferred[_AudioEncodeSamplesCall[origin_of(arena), samples_origin, stats_origin], Self.origin]:
        """Submits `ModelVerbs.audio_encode_samples`. Calls `uform_audio_encode_samples`."""
        return {
            self.model[].library,
            _AudioEncodeSamplesCall[origin_of(arena), samples_origin, stats_origin](
                samples, sample_rate, outputs, Pointer(to=arena), config
            ),
            self.model[].address,
            self.executor,
        }


# endregion Verbs
