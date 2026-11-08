"""The forward passes: texts, choices, images, clips and recordings, each into the outputs a caller keeps.

Every verb takes one argument structure named after it, mirrored below as an `...Arguments` struct
laid out exactly as the library reads it and crossed by address. Every output whose length scales
with the call lands in the `Arena` the verb takes as its last required argument, and the `Encoded`
result borrows it in turn. `Outputs` names which surfaces of the one forward a call keeps.

The verbs themselves are written once, in `ModelVerbs`; this module holds their crossings.

File: mojo/uform/encode.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from std.ffi import c_int, c_size_t

from uform.library import Library
from uform.types import (
    Address,
    Arena,
    Call,
    Completion,
    Outcome,
    Progress,
    Rows,
    RowsTape,
    RowsTapeViewArguments,
    RowsViewArguments,
    ScalarKind,
    Status,
    Tape,
    TapeView,
    TapeViewArguments,
    Verb,
    _cross,
    _refuse,
    _served,
)

# region Outputs


@fieldwise_init
struct Outputs(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The surfaces an encode verb keeps from its one forward, named after the fields they fill.
    Binding-only; it decides which output pointers a verb passes."""

    var code: UInt32
    """The surfaces kept, one flag each."""
    comptime EMBEDDINGS = Self(1)
    """One pooled embedding per item, past the attention pool and the projection head."""
    comptime STATES = Self(2)
    """One final-norm state per token left after truncation."""
    comptime TOKEN_VECTORS = Self(4)
    """One unit vector of the token head per token left after truncation; a model without a token head
    fails the call with `UNSUPPORTED`."""

    def __or__(self, other: Self) -> Self:
        """Both sets of surfaces."""
        return Self(self.code | other.code)

    def contains(self, other: Self) -> Bool:
        """Whether every surface `other` names is kept."""
        return self.code & other.code == other.code


@fieldwise_init
struct EncodeStats(Defaultable, ImplicitlyCopyable):
    """What an encode call reports beside its outputs: `uform_text_encode_stats_t`, and the image, video
    and audio reports that share its two fields."""

    var truncated: c_size_t
    """Items cut at the model's context."""
    var tokens: c_size_t
    """Tokens the call encoded after truncation: bytes, patches, tubes or frames."""

    def __init__(out self):
        """Zeroes both fields, which is what a failed call leaves."""
        self = Self(0, 0)


@fieldwise_init
struct EncodedArguments(Defaultable, ImplicitlyCopyable):
    """The three output views an encode verb writes through, at one stable address."""

    var embeddings: RowsViewArguments
    var states: RowsTapeViewArguments
    var token_vectors: RowsTapeViewArguments

    def __init__(out self):
        """Nothing written, which is what a failed call leaves."""
        self = Self(RowsViewArguments(), RowsTapeViewArguments(), RowsTapeViewArguments())


@fieldwise_init
struct Encoded[origin: MutOrigin](Movable):
    """What an encode verb wrote into the arena it borrows: the surfaces its `Outputs` named. Binding-only.

    Parameters:
        origin: The arena the surfaces live in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the surfaces."""
    var generation: Int
    """The arena's call these surfaces came from, so a read after a later call is refused."""
    var kept: Outputs
    """The surfaces the call kept."""
    var written: EncodedArguments
    """The views the engine wrote, inside the arena."""
    var count: Int
    """Items the call encoded."""

    def embeddings(self) -> Optional[Rows[Self.origin]]:
        """One pooled embedding of the model's dimensions per item, without normalization; absent unless kept."""
        if not self.kept.contains(Outputs.EMBEDDINGS):
            return None
        return Rows[Self.origin](self.arena, self.generation, self.written.embeddings, self.count)

    def states(self) -> Optional[RowsTape[Self.origin]]:
        """One final-norm state per token left after truncation, item by item; absent unless kept."""
        if not self.kept.contains(Outputs.STATES):
            return None
        return RowsTape[Self.origin](self.arena, self.generation, self.written.states, self.count)

    def token_vectors(self) -> Optional[RowsTape[Self.origin]]:
        """One unit vector of the model's token width per token left after truncation, item by item; absent
        unless kept."""
        if not self.kept.contains(Outputs.TOKEN_VECTORS):
            return None
        return RowsTape[Self.origin](self.arena, self.generation, self.written.token_vectors, self.count)


# endregion Outputs

# region Configs


struct EncodeConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs every encode verb shares, the verbs over texts, images, clips and recordings alike.

    Parameters:
        stats_origin: The lent report slot's origin, inferred where one is lent.
    """

    var outputs_scalar: ScalarKind
    """The scalar type every output lands as."""
    var progress: Progress
    """Reports the items encoded, and stops the call where it says to."""
    var stats: Optional[Pointer[EncodeStats, Self.stats_origin]]
    """Receives the truncation report, overwritten by the call; absent skips it."""

    def __init__(
        out self,
        *,
        outputs_scalar: ScalarKind = ScalarKind.F32,
        progress: Progress = Progress(),
        stats: Optional[Pointer[EncodeStats, Self.stats_origin]] = Optional[Pointer[EncodeStats, Self.stats_origin]](),
    ):
        """Names only the knobs that differ from the defaults."""
        self.outputs_scalar = outputs_scalar
        self.progress = progress
        self.stats = stats


struct TextChooseConfig[
    groups_origin: ImmOrigin = ImmUntrackedOrigin, stats_origin: MutOrigin = MutUntrackedOrigin
](ImplicitlyCopyable):
    """The question types, the progress handler and the report slot of `text_choose`.

    Parameters:
        groups_origin: The borrowed question types' origin, inferred where some are lent.
        stats_origin: The lent report slot's origin, inferred where one is lent.
    """

    var temperature_groups: Span[UInt32, Self.groups_origin]
    """The question type of each item, below the model's `choice_temperature_groups`; empty puts every
    item in group zero."""
    var progress: Progress
    """Reports the items scored, and stops the call where it says to."""
    var stats: Optional[Pointer[EncodeStats, Self.stats_origin]]
    """Receives the truncation report, overwritten by the call; absent skips it."""

    def __init__(
        out self,
        *,
        temperature_groups: Span[UInt32, Self.groups_origin] = Span[UInt32, Self.groups_origin](),
        progress: Progress = Progress(),
        stats: Optional[Pointer[EncodeStats, Self.stats_origin]] = Optional[Pointer[EncodeStats, Self.stats_origin]](),
    ):
        """Names only the knobs that differ from the defaults."""
        self.temperature_groups = temperature_groups
        self.progress = progress
        self.stats = stats


def _lent[T: AnyType, origin: MutOrigin, //](slot: Optional[Pointer[T, origin]]) -> Optional[Pointer[T, MutUntrackedOrigin]]:
    """A lent output slot with its borrow erased, as the C layout names it."""
    if case .Some(pointer) = slot:
        return pointer.unsafe_origin_cast[MutUntrackedOrigin]()
    return None


def _array[T: AnyType, origin: ImmOrigin, //](items: Span[T, origin]) -> Optional[Pointer[T, ImmUntrackedOrigin]]:
    """A borrowed host array with its borrow erased, absent when empty, as the C layout names it."""
    if len(items) == 0:
        return None
    return Pointer[T, ImmUntrackedOrigin](unsafe_from_address=Int(items.unsafe_ptr()))


# endregion Configs

# region Arguments


@fieldwise_init
struct EncodeArguments(ImplicitlyCopyable):
    """`uform_text_encode_t`, and `uform_image_encode_t`, `uform_video_encode_t` and
    `uform_audio_encode_t` with the same layout."""

    var model: Address
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var count: c_size_t
    var inputs: TapeViewArguments
    var embeddings: Optional[Pointer[RowsViewArguments, MutUntrackedOrigin]]
    var states: Optional[Pointer[RowsTapeViewArguments, MutUntrackedOrigin]]
    var token_vectors: Optional[Pointer[RowsTapeViewArguments, MutUntrackedOrigin]]
    var arena: Address
    var outputs_scalar: ScalarKind
    var progress: Progress
    var stats: Optional[Pointer[EncodeStats, MutUntrackedOrigin]]


@fieldwise_init
struct ImageEncodePixelsArguments(ImplicitlyCopyable):
    """`uform_image_encode_pixels_t`."""

    var model: Address
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var count: c_size_t
    var pixels: TapeViewArguments
    var heights: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var widths: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var embeddings: Optional[Pointer[RowsViewArguments, MutUntrackedOrigin]]
    var states: Optional[Pointer[RowsTapeViewArguments, MutUntrackedOrigin]]
    var token_vectors: Optional[Pointer[RowsTapeViewArguments, MutUntrackedOrigin]]
    var arena: Address
    var outputs_scalar: ScalarKind
    var progress: Progress
    var stats: Optional[Pointer[EncodeStats, MutUntrackedOrigin]]


@fieldwise_init
struct AudioEncodeSamplesArguments(ImplicitlyCopyable):
    """`uform_audio_encode_samples_t`."""

    var model: Address
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var count: c_size_t
    var samples: TapeViewArguments
    var sample_rate: UInt32
    var embeddings: Optional[Pointer[RowsViewArguments, MutUntrackedOrigin]]
    var states: Optional[Pointer[RowsTapeViewArguments, MutUntrackedOrigin]]
    var token_vectors: Optional[Pointer[RowsTapeViewArguments, MutUntrackedOrigin]]
    var arena: Address
    var outputs_scalar: ScalarKind
    var progress: Progress
    var stats: Optional[Pointer[EncodeStats, MutUntrackedOrigin]]


@fieldwise_init
struct TextChooseArguments(ImplicitlyCopyable):
    """`uform_text_choose_t`."""

    var model: Address
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var count: c_size_t
    var texts: TapeViewArguments
    var choices: TapeViewArguments
    var temperature_groups: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var probabilities: Optional[Pointer[TapeViewArguments, MutUntrackedOrigin]]
    var arena: Address
    var progress: Progress
    var stats: Optional[Pointer[EncodeStats, MutUntrackedOrigin]]


comptime TextEncode = Verb[EncodeArguments]
"""`uform_text_encode`."""
comptime ImageEncode = Verb[EncodeArguments]
"""`uform_image_encode`."""
comptime VideoEncode = Verb[EncodeArguments]
"""`uform_video_encode`."""
comptime AudioEncode = Verb[EncodeArguments]
"""`uform_audio_encode`."""
comptime ImageEncodePixels = Verb[ImageEncodePixelsArguments]
"""`uform_image_encode_pixels`."""
comptime AudioEncodeSamples = Verb[AudioEncodeSamplesArguments]
"""`uform_audio_encode_samples`."""
comptime TextChoose = Verb[TextChooseArguments]
"""`uform_text_choose`."""

# endregion Arguments

# region Calls


def _kept_embeddings(
    kept: Outputs, written: Pointer[EncodedArguments, MutUntrackedOrigin]
) -> Optional[Pointer[RowsViewArguments, MutUntrackedOrigin]]:
    """The embeddings slot when kept, so the library skips the pool and the head otherwise."""
    if kept.contains(Outputs.EMBEDDINGS):
        return Pointer(to=written[].embeddings)
    return None


def _kept_tape(
    kept: Bool, slot: Pointer[RowsTapeViewArguments, MutUntrackedOrigin]
) -> Optional[Pointer[RowsTapeViewArguments, MutUntrackedOrigin]]:
    """A ragged output slot when kept, so the library skips it otherwise."""
    if kept:
        return slot
    return None


@fieldwise_init
struct _EncodeCall[
    origin: MutOrigin, inputs_origin: ImmOrigin, stats_origin: MutOrigin, name: StaticString
](Call):
    """The crossing of `uform_text_encode`, `uform_image_encode`, `uform_video_encode` or
    `uform_audio_encode`, which share one layout and borrow the arena their surfaces land in."""

    comptime Outputs = EncodedArguments
    comptime Result = Encoded[Self.origin]
    comptime symbol: StaticString = Self.name
    var inputs: TapeView[Self.inputs_origin]
    var kept: Outputs
    var arena: Pointer[Arena, Self.origin]
    var config: EncodeConfig[Self.stats_origin]

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[EncodedArguments, MutUntrackedOrigin],
    ):
        var verb: Verb[EncodeArguments]
        comptime if Self.name == "uform_text_encode":
            verb = library.symbols().text_encode
        elif Self.name == "uform_image_encode":
            verb = library.symbols().image_encode
        elif Self.name == "uform_video_encode":
            verb = library.symbols().video_encode
        else:
            verb = library.symbols().audio_encode
        _cross(
            EncodeArguments(
                handle,
                executor,
                outcome,
                completion,
                c_size_t(self.inputs.count),
                self.inputs.arguments(),
                _kept_embeddings(self.kept, outputs),
                _kept_tape(self.kept.contains(Outputs.STATES), Pointer(to=outputs[].states)),
                _kept_tape(self.kept.contains(Outputs.TOKEN_VECTORS), Pointer(to=outputs[].token_vectors)),
                self.arena[]._address,
                self.config.outputs_scalar,
                self.config.progress,
                _lent(self.config.stats),
            ),
            verb,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: EncodedArguments, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, self.kept, outputs, self.inputs.count)


@fieldwise_init
struct _ImageEncodePixelsCall[
    origin: MutOrigin,
    pixels_origin: ImmOrigin,
    heights_origin: ImmOrigin,
    widths_origin: ImmOrigin,
    stats_origin: MutOrigin,
](Call):
    """The crossing of `uform_image_encode_pixels`, which borrows the arena its surfaces land in."""

    comptime Outputs = EncodedArguments
    comptime Result = Encoded[Self.origin]
    comptime symbol: StaticString = "uform_image_encode_pixels"
    var pixels: TapeView[Self.pixels_origin]
    var heights: Span[UInt32, Self.heights_origin]
    var widths: Span[UInt32, Self.widths_origin]
    var kept: Outputs
    var arena: Pointer[Arena, Self.origin]
    var config: EncodeConfig[Self.stats_origin]

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[EncodedArguments, MutUntrackedOrigin],
    ):
        if len(self.heights) != self.pixels.count or len(self.widths) != self.pixels.count:
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        _cross(
            ImageEncodePixelsArguments(
                handle,
                executor,
                outcome,
                completion,
                c_size_t(self.pixels.count),
                self.pixels.arguments(),
                _array(self.heights),
                _array(self.widths),
                _kept_embeddings(self.kept, outputs),
                _kept_tape(self.kept.contains(Outputs.STATES), Pointer(to=outputs[].states)),
                _kept_tape(self.kept.contains(Outputs.TOKEN_VECTORS), Pointer(to=outputs[].token_vectors)),
                self.arena[]._address,
                self.config.outputs_scalar,
                self.config.progress,
                _lent(self.config.stats),
            ),
            library.symbols().image_encode_pixels,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: EncodedArguments, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, self.kept, outputs, self.pixels.count)


@fieldwise_init
struct _AudioEncodeSamplesCall[
    origin: MutOrigin, samples_origin: ImmOrigin, stats_origin: MutOrigin
](Call):
    """The crossing of `uform_audio_encode_samples`, which borrows the arena its surfaces land in."""

    comptime Outputs = EncodedArguments
    comptime Result = Encoded[Self.origin]
    comptime symbol: StaticString = "uform_audio_encode_samples"
    var samples: TapeView[Self.samples_origin]
    var sample_rate: Int
    var kept: Outputs
    var arena: Pointer[Arena, Self.origin]
    var config: EncodeConfig[Self.stats_origin]

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[EncodedArguments, MutUntrackedOrigin],
    ):
        _cross(
            AudioEncodeSamplesArguments(
                handle,
                executor,
                outcome,
                completion,
                c_size_t(self.samples.count),
                self.samples.arguments(),
                UInt32(self.sample_rate),
                _kept_embeddings(self.kept, outputs),
                _kept_tape(self.kept.contains(Outputs.STATES), Pointer(to=outputs[].states)),
                _kept_tape(self.kept.contains(Outputs.TOKEN_VECTORS), Pointer(to=outputs[].token_vectors)),
                self.arena[]._address,
                self.config.outputs_scalar,
                self.config.progress,
                _lent(self.config.stats),
            ),
            library.symbols().audio_encode_samples,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: EncodedArguments, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, self.kept, outputs, self.samples.count)


@fieldwise_init
struct _TextChooseCall[
    origin: MutOrigin,
    texts_origin: ImmOrigin,
    choices_origin: ImmOrigin,
    groups_origin: ImmOrigin,
    stats_origin: MutOrigin,
](Call):
    """The crossing of `uform_text_choose`, which borrows the arena its probabilities land in."""

    comptime Outputs = TapeViewArguments
    comptime Result = Tape[Self.origin]
    comptime symbol: StaticString = "uform_text_choose"
    var texts: TapeView[Self.texts_origin]
    var choices: TapeView[Self.choices_origin]
    var arena: Pointer[Arena, Self.origin]
    var config: TextChooseConfig[Self.groups_origin, Self.stats_origin]

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[TapeViewArguments, MutUntrackedOrigin],
    ):
        var groups = len(self.config.temperature_groups)
        if self.choices.count != self.texts.count or (groups != 0 and groups != self.texts.count):
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        _cross(
            TextChooseArguments(
                handle,
                executor,
                outcome,
                completion,
                c_size_t(self.texts.count),
                self.texts.arguments(),
                self.choices.arguments(),
                _array(self.config.temperature_groups),
                outputs,
                self.arena[]._address,
                self.config.progress,
                _lent(self.config.stats),
            ),
            library.symbols().text_choose,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: TapeViewArguments, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, outputs, self.texts.count)


# endregion Calls
