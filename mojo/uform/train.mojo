"""Training on the GPU: the mesh joining a run's processes, and the trainer stepping one model across them.

A `Mesh` is one process's place among the processes of a run, made once and borrowed by every
trainer over the same executor. A `Trainer` owns the replicas, gradients, optimizer moments, heads
and temperatures on its executor's devices. Both are made over a VRAM executor, and a build without
the NVIDIA trainer answers every training verb with `UNSUPPORTED`.

The trainer's verbs follow the model's: `trainer.on(executor)` blocks on an executor and
`trainer.deferred(executor)` answers a `Deferred`. The trainer has no calling-thread receiver,
since every one of its verbs runs on the devices of the executor it was made over.

File: mojo/uform/train.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from std.ffi import c_int, c_size_t
from std.memory import stack_allocation

from uform.library import Library
from uform.types import (
    Address,
    Arena,
    BytesView,
    BytesViewArguments,
    Call,
    Completion,
    Deferred,
    Error,
    Executor,
    GPUClass,
    NULL_ADDRESS,
    Outcome,
    Progress,
    Rows,
    RowsViewArguments,
    ScalarKind,
    Status,
    TapeView,
    TapeViewArguments,
    Unwritten,
    Verb,
    _anchor,
    _block,
    _cross,
    _refuse,
    _served,
)

# region Vocabulary


@fieldwise_init
struct Modality(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """What the bytes of one training item hold, `uform_modality_t`."""

    var code: c_int
    """The raw `uform_modality_t` value the ABI exchanges."""
    comptime TEXT = Self(0)
    """UTF-8 text, a byte per token."""
    comptime IMAGE = Self(1)
    """A JPEG, PNG or WebP file."""
    comptime VIDEO = Self(2)
    """An MP4 or WebM file."""
    comptime AUDIO = Self(3)
    """A WAV, FLAC, MP3 or Opus file."""


@fieldwise_init
struct Objective(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which losses a step relaxes, `uform_objective_t`."""

    var code: c_int
    """The raw `uform_objective_t` value the ABI exchanges."""
    comptime MASKED = Self(0)
    """Masked reconstruction alone."""
    comptime CONTRASTIVE = Self(1)
    """The pairwise contrastive loss alone."""
    comptime JOINT = Self(2)
    """Both, joined at the final-norm states."""


@fieldwise_init
struct Negatives(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which pairs the contrastive loss pushes apart, `uform_negatives_t`."""

    var code: c_int
    """The raw `uform_negatives_t` value the ABI exchanges."""
    comptime AMBIENT = Self(0)
    """Repel edges, and every pair no edge names at the ambient weight."""
    comptime NAMED = Self(1)
    """Repel edges alone."""


@fieldwise_init
struct PairLossForm(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How a contrastive term turns a row of scores into forces, `uform_pair_loss_form_t`."""

    var code: c_int
    """The raw `uform_pair_loss_form_t` value the ABI exchanges."""
    comptime DERIVED = Self(0)
    """The term's own form: sigmoid for pooled vertices, ranked for token vectors."""
    comptime BALANCED = Self(1)
    """The sigmoid cross-entropy of every cell, its attract and repel halves normalized apart."""
    comptime RANKED = Self(2)
    """The sigmoid cross-entropy of every positive's score over every negative's."""
    comptime SIGMOID = Self(3)
    """SigLIP's sigmoid cross-entropy of every cell, each repel weighed like an attract."""


@fieldwise_init
struct LateInteractionSide(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which side of a late-interaction score an item takes, `uform_late_interaction_side_t`."""

    var code: c_int
    """The raw `uform_late_interaction_side_t` value the ABI exchanges."""
    comptime NONE = Self(0)
    """The item takes no part in late interaction."""
    comptime QUERY = Self(1)
    """The item's tokens score the documents it attracts and repels."""
    comptime DOCUMENT = Self(2)
    """The item's tokens are scored by the queries naming it."""
    comptime QUERY_AND_DOCUMENT = Self(3)
    """The item plays both sides."""


@fieldwise_init
struct MultimemTier(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How a mesh's NVLink domains reduce, `uform_multimem_tier_t`."""

    var code: c_int
    """The raw `uform_multimem_tier_t` value the ABI exchanges."""
    comptime SWITCH_REDUCED = Self(0)
    """The NVLink switch reduces through multicast memory."""
    comptime PEER_READ = Self(1)
    """Every member reads its peers in a fixed order, where multicast is absent."""


# endregion Vocabulary

# region Geometry


@fieldwise_init
struct ImageShape(ImplicitlyCopyable):
    """The image slice of a batch envelope, `uform_image_shape_t`; a zero `patch_side` leaves images out."""

    var patch_side: c_size_t
    """The square patch side images tile into, in pixels."""
    var channels: c_size_t
    """Channels every pixel carries; zero derives four."""

    def __init__(out self, *, patch_side: Int = 0, channels: Int = 0):
        """Names only the fields that differ from the derived ones."""
        self.patch_side = c_size_t(patch_side)
        self.channels = c_size_t(channels)


@fieldwise_init
struct VideoShape(ImplicitlyCopyable):
    """The video slice of a batch envelope, `uform_video_shape_t`; a zero `patch_side` leaves video out."""

    var patch_side: c_size_t
    """The square patch side one tube tiles into, in pixels."""
    var tube_frames: c_size_t
    """Frames one tube folds ahead of its patch; zero derives two."""
    var frame_stride: c_size_t
    """Source frames between sampled ones; zero derives one."""
    var channels: c_size_t
    """Channels every pixel carries; zero derives three."""

    def __init__(out self, *, patch_side: Int = 0, tube_frames: Int = 0, frame_stride: Int = 0, channels: Int = 0):
        """Names only the fields that differ from the derived ones."""
        self.patch_side = c_size_t(patch_side)
        self.tube_frames = c_size_t(tube_frames)
        self.frame_stride = c_size_t(frame_stride)
        self.channels = c_size_t(channels)


@fieldwise_init
struct AudioShape(ImplicitlyCopyable):
    """The audio slice of a batch envelope, `uform_audio_shape_t`; a zero `chunk_samples` leaves audio out."""

    var chunk_samples: c_size_t
    """Raw samples per frame, the fan-in of the learned frontend."""
    var sample_rate: c_size_t
    """The mono rate every recording is resampled to; zero derives 16000."""

    def __init__(out self, *, chunk_samples: Int = 0, sample_rate: Int = 0):
        """Names only the fields that differ from the derived ones."""
        self.chunk_samples = c_size_t(chunk_samples)
        self.sample_rate = c_size_t(sample_rate)


@fieldwise_init
struct BatchShape(ImplicitlyCopyable):
    """What one device's step carries at most, whatever the modality mix, `uform_batch_shape_t`."""

    var max_rows_per_device: c_size_t
    """Rows one device's step packs."""
    var max_segments_per_device: c_size_t
    """Items one device's step carries across all its rows, which is also its pooling axis."""
    var max_columns_per_row: c_size_t
    """Positions one row spans, the sequence length."""
    var max_bags_per_device: c_size_t
    """Bags one device's step carries; zero keeps every item its own routing unit."""
    var max_choices_per_item: c_size_t
    """Legal bytes one item offers to choose from at most; zero makes no choices."""
    var image: ImageShape
    """The image geometry."""
    var video: VideoShape
    """The video geometry."""
    var audio: AudioShape
    """The audio geometry."""

    def __init__(
        out self,
        *,
        max_rows_per_device: Int = 0,
        max_segments_per_device: Int = 0,
        max_columns_per_row: Int = 0,
        max_bags_per_device: Int = 0,
        max_choices_per_item: Int = 0,
        image: ImageShape = ImageShape(),
        video: VideoShape = VideoShape(),
        audio: AudioShape = AudioShape(),
    ):
        """Names only the fields that differ from the derived ones."""
        self.max_rows_per_device = c_size_t(max_rows_per_device)
        self.max_segments_per_device = c_size_t(max_segments_per_device)
        self.max_columns_per_row = c_size_t(max_columns_per_row)
        self.max_bags_per_device = c_size_t(max_bags_per_device)
        self.max_choices_per_item = c_size_t(max_choices_per_item)
        self.image = image
        self.video = video
        self.audio = audio


# endregion Geometry

# region Losses


@fieldwise_init
struct Masking(Defaultable, ImplicitlyCopyable):
    """How the masked objective corrupts a batch and weighs its heads, `uform_masking_t`; every zero
    derives the default its field names."""

    var token_mask_probability: Float32
    """Fraction of text positions covered by spans; 0.15."""
    var token_mean_span: Float32
    """Mean geometric span length in bytes; 16."""
    var replace_fraction: Float32
    """Of covered positions, the share shown as MASK; 0.8."""
    var random_fraction: Float32
    """Of covered positions, the share replaced by a random byte; 0.1."""
    var image_mask_ratio: Float32
    """Fraction of image patches masked; 0.75."""
    var video_mask_ratio: Float32
    """Fraction of video tubes masked; 0.75."""
    var audio_mask_ratio: Float32
    """Fraction of audio frames masked; 0.75."""
    var text_weight: Float32
    """The weight of the text head; 1."""
    var image_weight: Float32
    """The weight of the image head; 1."""
    var video_weight: Float32
    """The weight of the video head; 1."""
    var audio_weight: Float32
    """The weight of the audio head; 1."""

    def __init__(out self):
        """Derives every default."""
        self = Self(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


@fieldwise_init
struct LateInteraction(Defaultable, ImplicitlyCopyable):
    """How the late-interaction loss scores per-token vectors, `uform_late_interaction_t`; every zero
    derives the default its field names."""

    var weight: Float32
    """The loss weight; zero trains no token head."""
    var width: c_size_t
    """The token vectors' width, a multiple of 8; 128."""
    var inverse_temperature: Float32
    """The starting inverse temperature of every contrastive group; 1."""
    var bias: Float32
    """The starting logit bias of every contrastive group; zero."""
    var max_query_span: c_size_t
    """Tokens one query segment contributes at most; 64."""
    var max_query_tokens_per_device: c_size_t
    """Query token rows one device's step holds at most."""
    var max_documents_per_device: c_size_t
    """Document segments one device's step holds at most."""
    var max_document_tokens_per_device: c_size_t
    """Document token rows one device's step holds at most."""
    var pair_loss: PairLossForm
    """The pair loss over each query's row of hard MaxSim scores; ranked."""

    def __init__(out self):
        """Trains no token head."""
        self = Self(0, 0, 0, 0, 0, 0, 0, 0, PairLossForm.DERIVED)


@fieldwise_init
struct Choices(Defaultable, ImplicitlyCopyable):
    """The calibrated choices over the byte decoder, `uform_choices_t`."""

    var weight: Float32
    """The loss weight; zero trains no choices."""
    var brier_weight: Float32
    """The Brier term's weight beside the soft cross-entropy; zero derives 0.5."""

    def __init__(out self):
        """Trains no choices."""
        self = Self(0, 0)


@fieldwise_init
struct Schedule(Defaultable, ImplicitlyCopyable):
    """The learning-rate schedule, a linear warmup then a cosine decay, `uform_schedule_t`."""

    var learning_rate: Float32
    """The peak learning rate; zero derives 5e-4."""
    var min_learning_rate: Float32
    """The floor the cosine settles at; zero derives 1e-5."""
    var warmup_steps: c_size_t
    """Steps from zero to the peak; zero derives a tenth of `steps`, at most 400."""
    var steps: c_size_t
    """Steps the run takes across resumes, which the cosine spans; zero is refused."""

    def __init__(out self):
        """Derives every default but `steps`, which a run must name."""
        self = Self(0, 0, 0, 0)


# endregion Losses

# region Reports


@fieldwise_init
struct TrainerStepStats(Defaultable, ImplicitlyCopyable):
    """What a step reports, every loss its world mean, `uform_trainer_step_stats_t`."""

    var step: c_size_t
    """The optimizer step the call applied, counted from one across resumes."""
    var learning_rate: Float32
    """The learning rate it applied."""
    var loss: Float32
    """The total loss the step minimized."""
    var contrastive_loss: Float32
    """The contrastive term; zero when the objective is masked alone."""
    var masked_text_loss: Float32
    """The text reconstruction term, before its weight."""
    var masked_image_loss: Float32
    """The image reconstruction term, before its weight."""
    var masked_video_loss: Float32
    """The video reconstruction term, before its weight."""
    var masked_audio_loss: Float32
    """The audio reconstruction term, before its weight."""
    var late_interaction_loss: Float32
    """The late-interaction term, weighted."""
    var choices_loss: Float32
    """The choices term over the items that made a choice, before its weight."""
    var truncated: c_size_t
    """Items cut at a row's width."""
    var dropped: c_size_t
    """Segments dropped whole for lack of room, since the last step that read its stats."""

    def __init__(out self):
        """Zeroes every field, which is what a failed step leaves."""
        self = Self(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


@fieldwise_init
struct MeshDescription(Defaultable, ImplicitlyCopyable):
    """What a mesh reports about itself, `uform_mesh_description_t`."""

    var gpus_per_host_domain: c_size_t
    """GPUs one process drives."""
    var gpus_per_multimem_domain: c_size_t
    """GPUs one NVLink domain spans."""
    var gpus_per_colocated_domain: c_size_t
    """GPUs in the whole run."""
    var host_domain_index: c_size_t
    """This process's index among the run's processes."""
    var first_gpu_index: c_size_t
    """The run-wide index of this process's first GPU, where its share of a sharded dataset begins."""
    var multimem_tier: MultimemTier
    """How its NVLink domains reduce."""
    var gpu_class: GPUClass
    """The class every GPU of the run dispatches as."""

    def __init__(out self):
        """Zeroes every field, which is what a failed report leaves."""
        self = Self(0, 0, 0, 0, 0, MultimemTier.SWITCH_REDUCED, GPUClass.UNKNOWN)


# endregion Reports

# region Configs


@fieldwise_init
struct ContrastiveArguments(ImplicitlyCopyable):
    """`uform_contrastive_t`, laid out as the ABI declares it."""

    var matryoshka_dimensions: Optional[Pointer[c_size_t, ImmUntrackedOrigin]]
    var matryoshka_dimensions_count: c_size_t
    var temperature_groups: c_size_t
    var inverse_temperature: Float32
    var bias: Float32
    var negatives: Negatives
    var ambient_repulsion_weight: Float32
    var max_edges_per_device: c_size_t
    var pair_loss: PairLossForm


@fieldwise_init
struct Contrastive(Copyable, Defaultable, Movable):
    """How the contrastive loss scores the pooled embeddings, `uform_contrastive_t`; every zero derives
    the default its field names."""

    var matryoshka_dimensions: List[c_size_t]
    """Matryoshka prefix widths, strictly ascending, each a multiple of 8, at most 8; empty scores the
    full width alone."""
    var temperature_groups: Int
    """Learnable inverse temperature and bias pairs, one per group a step's items select; one."""
    var inverse_temperature: Float32
    """The starting inverse temperature of every group; 10 under the sigmoid form, else 1."""
    var bias: Float32
    """The starting logit bias of every group; -10 under the sigmoid form, else zero."""
    var negatives: Negatives
    """Which pairs repel."""
    var ambient_repulsion_weight: Float32
    """The weight of every pair no edge names, under `Negatives.AMBIENT`; 1."""
    var max_edges_per_device: Int
    """Edges one device's step names per edge set at most; zero derives it."""
    var pair_loss: PairLossForm
    """The pair loss over each anchor's row; sigmoid."""

    def __init__(out self):
        """Derives every default."""
        self = Self(List[c_size_t](), 0, 0, 0, Negatives.AMBIENT, 0, 0, PairLossForm.DERIVED)

    def _arguments(self) -> ContrastiveArguments:
        """The C layout, borrowing `matryoshka_dimensions` for the call."""
        var dimensions: Optional[Pointer[c_size_t, ImmUntrackedOrigin]] = None
        if len(self.matryoshka_dimensions):
            dimensions = Pointer[c_size_t, ImmUntrackedOrigin](
                unsafe_from_address=Int(self.matryoshka_dimensions.unsafe_ptr())
            )
        return ContrastiveArguments(
            dimensions,
            c_size_t(len(self.matryoshka_dimensions)),
            c_size_t(self.temperature_groups),
            self.inverse_temperature,
            self.bias,
            self.negatives,
            self.ambient_repulsion_weight,
            c_size_t(self.max_edges_per_device),
            self.pair_loss,
        )


@fieldwise_init
struct MeshOptionsArguments(ImplicitlyCopyable):
    """`uform_mesh_options_t`, laid out as the ABI declares it."""

    var host_domain_index: c_size_t
    var host_domain_count: c_size_t
    var gpus_per_multimem_domain: c_size_t
    var directory: BytesViewArguments
    var job: BytesViewArguments
    var timeout_seconds: Float64


@fieldwise_init
struct MeshOptions(Copyable, Defaultable, Movable):
    """Where this process sits among the processes training one model, and how they meet,
    `uform_mesh_options_t`."""

    var host_domain_index: Int
    """This process's index among `host_domain_count`."""
    var host_domain_count: Int
    """Processes in the run, one per host; zero derives one, a run with no peers."""
    var gpus_per_multimem_domain: Int
    """GPUs one NVLink multicast domain spans; zero derives the host."""
    var directory: String
    """The directory every host sees, where processes exchange their facts; required with peers."""
    var job: String
    """The run's unique name inside `directory`; required with peers."""
    var timeout_seconds: Float64
    """Seconds any wait for a peer lasts before the call fails; zero derives 120."""

    def __init__(out self):
        """A run with no peers."""
        self = Self(0, 0, 0, String(), String(), 0)


@fieldwise_init
struct TrainerInitVramConfig[mesh_origin: ImmOrigin = ImmUntrackedOrigin](Copyable, Defaultable, Movable):
    """The geometry, objective and starting point of a run, `uform_trainer_init_vram_t`.

    Parameters:
        mesh_origin: The borrowed mesh's origin, which the trainer then keeps alive.
    """

    var resume: String
    """The checkpoint path to continue from, at its saved step; empty starts from `seed`."""
    var depth: Int
    """Transformer blocks stacked."""
    var hidden: Int
    """Embedding width, a multiple of 128."""
    var batch: BatchShape
    """What one device's step carries at most."""
    var objective: Objective
    """Which losses a step relaxes."""
    var masking: Masking
    """The masked objective's corruption and weights, read unless the objective is contrastive alone."""
    var contrastive: Contrastive
    """The contrastive loss's scoring, read unless the objective is masked alone."""
    var late_interaction: LateInteraction
    """The late-interaction loss beside the pooled one, read when its weight is positive."""
    var choices: Choices
    """The calibrated choices, read when its weight is positive."""
    var schedule: Schedule
    """The learning-rate schedule."""
    var mesh: Optional[Pointer[Mesh, Self.mesh_origin]]
    """The mesh made over the same executor; absent runs this process alone."""
    var seed: UInt64
    """The base of every keyed draw: initialization, corruption and the shuffle."""
    var progress: Progress
    """Reports the tensors placed on a resume, and stops the call where it says to."""

    def __init__(out self):
        """Every field zeroed or derived, which a run then overrides by name."""
        self = Self(
            String(),
            0,
            0,
            BatchShape(),
            Objective.MASKED,
            Masking(),
            Contrastive(),
            LateInteraction(),
            Choices(),
            Schedule(),
            None,
            0,
            Progress(),
        )


@fieldwise_init
struct EdgesViewArguments(ImplicitlyCopyable):
    """`uform_edges_view_t`, laid out as the ABI declares it."""

    var offsets: Optional[Pointer[c_size_t, ImmUntrackedOrigin]]
    var ids: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var weights: Optional[Pointer[Float32, ImmUntrackedOrigin]]


@fieldwise_init
struct RelationsViewArguments(ImplicitlyCopyable):
    """`uform_relations_view_t`, laid out as the ABI declares it."""

    var attract: EdgesViewArguments
    var repel: EdgesViewArguments
    var ignore: EdgesViewArguments


def _array[T: AnyType, origin: ImmOrigin, //](items: Span[T, origin]) -> Optional[Pointer[T, ImmUntrackedOrigin]]:
    """A borrowed host array with its borrow erased, absent when empty, as the C layout names it."""
    if len(items) == 0:
        return None
    return Pointer[T, ImmUntrackedOrigin](unsafe_from_address=Int(items.unsafe_ptr()))


def _reborrow[T: Copyable, origin: ImmOrigin, //, wider: ImmOrigin](items: Span[T, origin]) -> Span[T, wider]:
    """The same items under an origin that unions theirs with others."""
    if len(items) == 0:
        return Span[T, wider]()
    return Span[T, wider](unsafe_ptr=Pointer[T, wider](unsafe_from_address=Int(items.unsafe_ptr())), length=len(items))


@fieldwise_init
struct Edges[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, Defaultable):
    """One edge set over a step's items as compressed sparse rows; empty offsets name no edges.

    Parameters:
        origin: The borrowed offsets, IDs and weights, kept alive for as long as the edges are used.
    """

    var offsets: Span[c_size_t, Self.origin]
    """The step's count plus one non-decreasing offsets into `ids`."""
    var ids: Span[UInt32, Self.origin]
    """Item `i` links to items `ids[offsets[i]]` up to `ids[offsets[i + 1]]`, ascending within each run."""
    var weights: Span[Float32, Self.origin]
    """The weight of each edge, paired with `ids`; empty weighs every edge one."""

    def __init__(out self):
        """Names no edges."""
        self = Self(Span[c_size_t, Self.origin](), Span[UInt32, Self.origin](), Span[Float32, Self.origin]())

    @staticmethod
    def of[
        offsets_origin: ImmOrigin, ids_origin: ImmOrigin, weights_origin: ImmOrigin, //
    ](
        offsets: Span[c_size_t, offsets_origin], ids: Span[UInt32, ids_origin], weights: Span[Float32, weights_origin]
    ) -> Edges[ImmOrigin(origin_of(_anchor[offsets_origin](), _anchor[ids_origin](), _anchor[weights_origin]()))]:
        """Borrows the three arrays of one edge set; an empty `weights` weighs every edge one."""
        comptime union = ImmOrigin(origin_of(_anchor[offsets_origin](), _anchor[ids_origin](), _anchor[weights_origin]()))
        return Edges[union](_reborrow[union](offsets), _reborrow[union](ids), _reborrow[union](weights))

    def _arguments(self) -> EdgesViewArguments:
        """The C layout, erased inside the verb that borrows it."""
        return EdgesViewArguments(_array(self.offsets), _array(self.ids), _array(self.weights))


@fieldwise_init
struct Relations[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, Defaultable):
    """The contrastive targets of one step, resolved per pair in this order: a pair `ignore` names is
    skipped, one `attract` names pulls together, one `repel` names pushes apart, and every other pair
    follows the run's negatives.

    Parameters:
        origin: The borrowed edge sets, kept alive for as long as the relations are used.
    """

    var attract: Edges[Self.origin]
    """Pairs pulled together."""
    var repel: Edges[Self.origin]
    """Pairs pushed apart."""
    var ignore: Edges[Self.origin]
    """Pairs skipped, like known false negatives; their weights are not read."""

    def __init__(out self):
        """Names no pairs."""
        self = Self(Edges[Self.origin](), Edges[Self.origin](), Edges[Self.origin]())

    @staticmethod
    def of[
        attract_origin: ImmOrigin, repel_origin: ImmOrigin, ignore_origin: ImmOrigin, //
    ](
        attract: Edges[attract_origin],
        repel: Edges[repel_origin] = Edges(),
        ignore: Edges[ignore_origin] = Edges(),
    ) -> Relations[ImmOrigin(origin_of(_anchor[attract_origin](), _anchor[repel_origin](), _anchor[ignore_origin]()))]:
        """Gathers the three edge sets of one step."""
        comptime union = ImmOrigin(origin_of(_anchor[attract_origin](), _anchor[repel_origin](), _anchor[ignore_origin]()))
        return Relations[union](
            Edges[union](_reborrow[union](attract.offsets), _reborrow[union](attract.ids), _reborrow[union](attract.weights)),
            Edges[union](_reborrow[union](repel.offsets), _reborrow[union](repel.ids), _reborrow[union](repel.weights)),
            Edges[union](_reborrow[union](ignore.offsets), _reborrow[union](ignore.ids), _reborrow[union](ignore.weights)),
        )

    def _arguments(self) -> RelationsViewArguments:
        """The C layout, erased inside the verb that borrows it."""
        return RelationsViewArguments(self.attract._arguments(), self.repel._arguments(), self.ignore._arguments())


struct TrainerStepConfig[
    bags_origin: ImmOrigin = ImmUntrackedOrigin,
    groups_origin: ImmOrigin = ImmUntrackedOrigin,
    sides_origin: ImmOrigin = ImmUntrackedOrigin,
    choices_origin: ImmOrigin = ImmUntrackedOrigin,
    targets_origin: ImmOrigin = ImmUntrackedOrigin,
    relations_origin: ImmOrigin = ImmUntrackedOrigin,
    stats_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The routing, targets and report slot of a training step, `uform_trainer_step_t`; every array is
    optional and, when given, holds one entry per item."""

    var bags: Span[UInt32, Self.bags_origin]
    """The bag of each item, items sharing a bag landing on one device; empty keeps every item its own."""
    var temperature_groups: Span[UInt32, Self.groups_origin]
    """The temperature group of each item; empty puts every item in group zero."""
    var sides: Span[LateInteractionSide, Self.sides_origin]
    """The late-interaction side of each item; empty makes every item `LateInteractionSide.NONE`."""
    var choices: Optional[TapeView[Self.choices_origin]]
    """Each item's legal choices, one byte each, as `text_choose` reads them; absent makes no choices."""
    var choice_targets: Optional[TapeView[Self.targets_origin]]
    """The target probability of each choice, `F32`, as `text_choose` answers them; read with `choices`."""
    var relations: Relations[Self.relations_origin]
    """The contrastive targets, read unless the objective is masked alone."""
    var outputs_scalar: ScalarKind
    """The scalar type the pooled embeddings land as, read only with an arena."""
    var stats: Optional[Pointer[TrainerStepStats, Self.stats_origin]]
    """Receives the losses, overwritten by the call; absent skips their readback too."""

    def __init__(
        out self,
        *,
        bags: Span[UInt32, Self.bags_origin] = Span[UInt32, Self.bags_origin](),
        temperature_groups: Span[UInt32, Self.groups_origin] = Span[UInt32, Self.groups_origin](),
        sides: Span[LateInteractionSide, Self.sides_origin] = Span[LateInteractionSide, Self.sides_origin](),
        choices: Optional[TapeView[Self.choices_origin]] = Optional[TapeView[Self.choices_origin]](),
        choice_targets: Optional[TapeView[Self.targets_origin]] = Optional[TapeView[Self.targets_origin]](),
        relations: Relations[Self.relations_origin] = Relations[Self.relations_origin](),
        outputs_scalar: ScalarKind = ScalarKind.F32,
        stats: Optional[Pointer[TrainerStepStats, Self.stats_origin]] = Optional[Pointer[TrainerStepStats, Self.stats_origin]](),
    ):
        """Names only the arrays and knobs a step uses."""
        self.bags = bags
        self.temperature_groups = temperature_groups
        self.sides = sides
        self.choices = choices
        self.choice_targets = choice_targets
        self.relations = relations
        self.outputs_scalar = outputs_scalar
        self.stats = stats


# endregion Configs

# region Arguments


@fieldwise_init
struct MeshBootstrapArguments(ImplicitlyCopyable):
    """`uform_mesh_bootstrap_t`."""

    var mesh: Optional[Pointer[Address, MutUntrackedOrigin]]
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var options: MeshOptionsArguments


@fieldwise_init
struct MeshDescribeArguments(ImplicitlyCopyable):
    """`uform_mesh_describe_t`."""

    var mesh: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var description: Optional[Pointer[MeshDescription, MutUntrackedOrigin]]


@fieldwise_init
struct TrainerInitVramArguments(ImplicitlyCopyable):
    """`uform_trainer_init_vram_t`."""

    var trainer: Optional[Pointer[Address, MutUntrackedOrigin]]
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var resume: BytesViewArguments
    var depth: c_size_t
    var hidden: c_size_t
    var batch: BatchShape
    var objective: Objective
    var masking: Masking
    var contrastive: ContrastiveArguments
    var late_interaction: LateInteraction
    var choices: Choices
    var schedule: Schedule
    var mesh: Address
    var seed: UInt64
    var progress: Progress


@fieldwise_init
struct TrainerStepArguments(ImplicitlyCopyable):
    """`uform_trainer_step_t`."""

    var trainer: Address
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var count: c_size_t
    var modalities: Optional[Pointer[Modality, ImmUntrackedOrigin]]
    var contents: TapeViewArguments
    var bags: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var temperature_groups: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var sides: Optional[Pointer[LateInteractionSide, ImmUntrackedOrigin]]
    var choices: TapeViewArguments
    var choice_targets: TapeViewArguments
    var relations: RelationsViewArguments
    var embeddings: Optional[Pointer[RowsViewArguments, MutUntrackedOrigin]]
    var arena: Address
    var outputs_scalar: ScalarKind
    var stats: Optional[Pointer[TrainerStepStats, MutUntrackedOrigin]]


@fieldwise_init
struct TrainerSaveArguments(ImplicitlyCopyable):
    """`uform_trainer_save_t`."""

    var trainer: Address
    var executor: Address
    var outcome: Optional[Pointer[Outcome, MutUntrackedOrigin]]
    var completion: Completion
    var path: BytesViewArguments
    var progress: Progress


comptime MeshBootstrap = Verb[MeshBootstrapArguments]
"""`uform_mesh_bootstrap`."""
comptime MeshDescribe = Verb[MeshDescribeArguments]
"""`uform_mesh_describe`."""
comptime TrainerInitVram = Verb[TrainerInitVramArguments]
"""`uform_trainer_init_vram`."""
comptime TrainerStep = Verb[TrainerStepArguments]
"""`uform_trainer_step`."""
comptime TrainerSave = Verb[TrainerSaveArguments]
"""`uform_trainer_save`."""

# endregion Arguments

# region Calls


@fieldwise_init
struct _TrainerStepCall[origin: MutOrigin](Call):
    """The crossing of `uform_trainer_step`, its borrowed inputs erased by the verb that built it and
    kept alive by that verb, or by its `Deferred`."""

    comptime Outputs = RowsViewArguments
    comptime Result = Optional[Rows[Self.origin]]
    comptime symbol: StaticString = "uform_trainer_step"
    var count: Int
    var refused: Bool
    var modalities: Optional[Pointer[Modality, ImmUntrackedOrigin]]
    var contents: TapeViewArguments
    var bags: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var temperature_groups: Optional[Pointer[UInt32, ImmUntrackedOrigin]]
    var sides: Optional[Pointer[LateInteractionSide, ImmUntrackedOrigin]]
    var choices: TapeViewArguments
    var choice_targets: TapeViewArguments
    var relations: RelationsViewArguments
    var arena: Optional[Pointer[Arena, Self.origin]]
    var outputs_scalar: ScalarKind
    var stats: Optional[Pointer[TrainerStepStats, MutUntrackedOrigin]]

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[RowsViewArguments, MutUntrackedOrigin],
    ):
        if self.refused:
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        var embeddings: Optional[Pointer[RowsViewArguments, MutUntrackedOrigin]] = None
        var arena = NULL_ADDRESS
        if case .Some(lent) = self.arena:
            embeddings = outputs
            arena = lent[]._address
        _cross(
            TrainerStepArguments(
                handle,
                executor,
                outcome,
                completion,
                c_size_t(self.count),
                self.modalities,
                self.contents,
                self.bags,
                self.temperature_groups,
                self.sides,
                self.choices,
                self.choice_targets,
                self.relations,
                embeddings,
                arena,
                self.outputs_scalar,
                self.stats,
            ),
            library.symbols().trainer_step,
        )

    def served(self, status: Status) -> Int:
        if case .Some(lent) = self.arena:
            return _served(lent, status)
        return 0

    def conclude(deinit self, outputs: RowsViewArguments, generation: Int) -> Self.Result:
        if case .Some(lent) = self.arena:
            return Rows[Self.origin](lent, generation, outputs, self.count)
        return None


def _step_call[
    origin: MutOrigin,
    modalities_origin: ImmOrigin,
    contents_origin: ImmOrigin,
    bags_origin: ImmOrigin,
    groups_origin: ImmOrigin,
    sides_origin: ImmOrigin,
    choices_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    relations_origin: ImmOrigin,
    stats_origin: MutOrigin,
](
    modalities: Span[Modality, modalities_origin],
    contents: TapeView[contents_origin],
    arena: Optional[Pointer[Arena, origin]],
    config: TrainerStepConfig[
        bags_origin, groups_origin, sides_origin, choices_origin, targets_origin, relations_origin, stats_origin
    ],
) -> _TrainerStepCall[origin]:
    """Lowers a step's borrowed inputs into its crossing, inside the verb that keeps them alive."""
    var count = contents.count

    def optional_length(length: Int) capturing -> Bool:
        return length == 0 or length == count

    var refused = len(modalities) != count or not (
        optional_length(len(config.bags))
        and optional_length(len(config.temperature_groups))
        and optional_length(len(config.sides))
    )
    var choices = TapeViewArguments()
    if case .Some(tape) = config.choices:
        choices = tape.arguments()
    var choice_targets = TapeViewArguments()
    if case .Some(tape) = config.choice_targets:
        choice_targets = tape.arguments()
    var stats: Optional[Pointer[TrainerStepStats, MutUntrackedOrigin]] = None
    if case .Some(slot) = config.stats:
        stats = slot.unsafe_origin_cast[MutUntrackedOrigin]()
    return _TrainerStepCall[origin](
        count,
        refused,
        _array(modalities),
        contents.arguments(),
        _array(config.bags),
        _array(config.temperature_groups),
        _array(config.sides),
        choices,
        choice_targets,
        config.relations._arguments(),
        arena,
        config.outputs_scalar,
        stats,
    )


@fieldwise_init
struct _TrainerSaveCall(Call):
    """The crossing of `uform_trainer_save`, its path erased by the verb that built it."""

    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "uform_trainer_save"
    var path: BytesViewArguments
    var progress: Progress

    def submit(
        mut self,
        library: Library,
        handle: Address,
        executor: Address,
        completion: Completion,
        outcome: Pointer[Outcome, MutUntrackedOrigin],
        outputs: Pointer[Unwritten, MutUntrackedOrigin],
    ):
        _cross(
            TrainerSaveArguments(handle, executor, outcome, completion, self.path, self.progress),
            library.symbols().trainer_save,
        )

    def conclude(deinit self, outputs: Unwritten, generation: Int) -> NoneType:
        return None


# endregion Calls

# region Mesh


struct Mesh(Movable):
    """One process's place among the processes of a run, borrowed by every trainer over the same executor;
    released when it goes out of scope, after every trainer borrowing it."""

    var library: Library
    """The loaded engine, held so the release can still reach `uform_mesh_free`."""
    var address: Address
    """The opaque `uform_mesh_t`."""

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle the engine just created."""
        self.library = library
        self.address = address

    @staticmethod
    def bootstrap(executor: Executor, options: MeshOptions = MeshOptions()) raises Error -> Self:
        """Joins this process to the run's mesh, waiting for every peer. Calls `uform_mesh_bootstrap`.

        Args:
            executor: A device executor whose devices are this process's members of the mesh.
            options: Where this process sits and how the processes meet.

        Raises:
            `Error` with `CONFIG_MISMATCH` when peers disagree, or `UNSUPPORTED` in a build without
            the trainer.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            MeshBootstrapArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                MeshOptionsArguments(
                    c_size_t(options.host_domain_index),
                    c_size_t(options.host_domain_count),
                    c_size_t(options.gpus_per_multimem_domain),
                    BytesView.of(options.directory).arguments(),
                    BytesView.of(options.job).arguments(),
                    options.timeout_seconds,
                ),
            ),
            executor.library.symbols().mesh_bootstrap,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_mesh_bootstrap")
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def describe(self) raises Error -> MeshDescription:
        """Reports the mesh's shape and this process's place in it. Calls `uform_mesh_describe`."""
        var description = stack_allocation[1, MeshDescription]()
        description[unsafe_offset=0] = MeshDescription()
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            MeshDescribeArguments(
                self.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                description.unsafe_origin_cast[MutUntrackedOrigin](),
            ),
            self.library.symbols().mesh_describe,
        )
        self.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_mesh_describe")
        return description.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]

    def __deinit__(deinit self):
        """Leaves the mesh. Calls `uform_mesh_free`."""
        self.library.symbols().mesh_free(self.address)
        _ = self.library^


# endregion Mesh

# region Trainer


struct Trainer[mesh_origin: ImmOrigin = ImmUntrackedOrigin](Movable):
    """A training run on a device executor's devices, released when it goes out of scope.

    Its verbs run only on an executor: `trainer.on(executor)` blocks there, and
    `trainer.deferred(executor)` submits there and answers a `Deferred`.

    Parameters:
        mesh_origin: The mesh it borrows, kept alive for as long as the trainer is.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `uform_trainer_t`."""
    var mesh: Optional[Pointer[Mesh, Self.mesh_origin]]
    """The mesh the run spans, absent for a process running alone."""

    def __init__(out self, *, library: Library, address: Address, mesh: Optional[Pointer[Mesh, Self.mesh_origin]]):
        """Adopts a handle the engine just created."""
        self.library = library
        self.address = address
        self.mesh = mesh

    @staticmethod
    def init_vram[
        config_mesh_origin: ImmOrigin, //
    ](executor: Executor, config: TrainerInitVramConfig[config_mesh_origin]) raises Error -> Trainer[
        config_mesh_origin
    ]:
        """Builds one replica per device and starts or resumes the run; every process of the run calls it
        with the same geometry, objective and seed. Calls `uform_trainer_init_vram`.

        Raises:
            `Error` with `CONFIG_MISMATCH` for a checkpoint of another geometry, or `UNSUPPORTED` in a
            build without the trainer.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var mesh = NULL_ADDRESS
        if case .Some(borrowed) = config.mesh:
            mesh = borrowed[].address
        _cross(
            TrainerInitVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                BytesView.of(config.resume).arguments(),
                c_size_t(config.depth),
                c_size_t(config.hidden),
                config.batch,
                config.objective,
                config.masking,
                config.contrastive._arguments(),
                config.late_interaction,
                config.choices,
                config.schedule,
                mesh,
                config.seed,
                config.progress,
            ),
            executor.library.symbols().trainer_init_vram,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "uform_trainer_init_vram")
        return Trainer[config_mesh_origin](
            library=executor.library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
            mesh=config.mesh,
        )

    def __deinit__(deinit self):
        """Frees the trainer and leaves the mesh; every call using it must be final. Calls `uform_trainer_free`."""
        self.library.symbols().trainer_free(self.address)
        _ = self.mesh^
        _ = self.library^

    def on(ref self, ref executor: Executor) -> TrainerOn[ImmOrigin(origin_of(self, executor))]:
        """The trainer's verbs, blocking on `executor`, the one it was made over. Binding-only."""
        return TrainerOn[ImmOrigin(origin_of(self, executor))](
            self.library, self.address, executor.address
        )

    def deferred(ref self, ref executor: Executor) -> TrainerDeferredOn[ImmOrigin(origin_of(self, executor))]:
        """The trainer's verbs, each submitted to `executor` and answering a `Deferred` at once. Binding-only."""
        return TrainerDeferredOn[ImmOrigin(origin_of(self, executor))](
            self.library, self.address, executor.address
        )


trait TrainVerbs:
    """The verbs every blocking trainer receiver offers, each written once over the receiver's `_route`."""

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` to its final status on this receiver's executor and answers its result."""
        ...

    def step[
        modalities_origin: ImmOrigin,
        contents_origin: ImmOrigin,
        bags_origin: ImmOrigin,
        groups_origin: ImmOrigin,
        sides_origin: ImmOrigin,
        choices_origin: ImmOrigin,
        targets_origin: ImmOrigin,
        relations_origin: ImmOrigin,
        stats_origin: MutOrigin,
        //,
    ](
        self,
        modalities: Span[Modality, modalities_origin],
        contents: TapeView[contents_origin],
        config: TrainerStepConfig[
            bags_origin, groups_origin, sides_origin, choices_origin, targets_origin, relations_origin, stats_origin
        ] = TrainerStepConfig(),
    ) raises Error:
        """Applies one optimizer step over this process's items and every peer's; every process of the run
        calls it once per step. Calls `uform_trainer_step`.

        Args:
            modalities: The modality of each item.
            contents: Each item's bytes, UTF-8 text or an encoded file as `modalities` tags it.
            config: The routing, the targets and the report slot.

        Raises:
            `Error` with `SHAPE_MISMATCH` when the per-item arrays disagree with `contents`.
        """
        _ = self._route(
            _step_call[MutUntrackedOrigin](
                modalities, contents, Optional[Pointer[Arena, MutUntrackedOrigin]](), config
            )
        )

    def step[
        modalities_origin: ImmOrigin,
        contents_origin: ImmOrigin,
        bags_origin: ImmOrigin,
        groups_origin: ImmOrigin,
        sides_origin: ImmOrigin,
        choices_origin: ImmOrigin,
        targets_origin: ImmOrigin,
        relations_origin: ImmOrigin,
        stats_origin: MutOrigin,
        //,
    ](
        self,
        modalities: Span[Modality, modalities_origin],
        contents: TapeView[contents_origin],
        mut arena: Arena,
        config: TrainerStepConfig[
            bags_origin, groups_origin, sides_origin, choices_origin, targets_origin, relations_origin, stats_origin
        ] = TrainerStepConfig(),
    ) raises Error -> Rows[origin_of(arena)]:
        """The same step, also landing the pooled embedding of every item in `arena`, as the contrastive
        loss scored it before the update. Calls `uform_trainer_step`."""
        return self._route(
            _step_call[origin_of(arena)](modalities, contents, Pointer(to=arena), config)
        ).value()

    def save[path_origin: ImmOrigin, //](self, path: StringSlice[path_origin], progress: Progress = Progress()) raises Error:
        """Writes the weights and the step, which the model loads serve and `init_vram` resumes; only the
        process of index zero writes. Calls `uform_trainer_save`."""
        _ = self._route(_TrainerSaveCall(BytesView.of(path).arguments(), progress))


struct TrainerOn[origin: ImmOrigin](ImplicitlyCopyable, TrainVerbs):
    """A trainer bound to the executor its verbs block on, as `trainer.on(executor)` answers it. Binding-only.

    Parameters:
        origin: The trainer and the executor, which this receiver borrows.
    """

    var library: Library
    """The loaded engine every verb crosses into."""
    var trainer: Address
    """The `uform_trainer_t` every verb addresses."""
    var executor: Address
    """The `uform_executor_t` every verb runs on."""

    def __init__(out self, library: Library, trainer: Address, executor: Address):
        """Binds a trainer to an executor."""
        self.library = library
        self.trainer = trainer
        self.executor = executor

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the bound executor, after every call submitted there before it."""
        return _block(self.library, call^, self.trainer, self.executor)


struct TrainerDeferredOn[origin: ImmOrigin](ImplicitlyCopyable):
    """A trainer bound to the executor its verbs are submitted to: every verb answers a `Deferred` at once.
    Binding-only.

    Parameters:
        origin: The trainer and the executor, which this receiver borrows.
    """

    var library: Library
    """The loaded engine every verb crosses into."""
    var trainer: Address
    """The `uform_trainer_t` every verb addresses."""
    var executor: Address
    """The `uform_executor_t` every verb is submitted to."""

    def __init__(out self, library: Library, trainer: Address, executor: Address):
        """Binds a trainer to an executor."""
        self.library = library
        self.trainer = trainer
        self.executor = executor

    def step[
        modalities_origin: ImmOrigin,
        contents_origin: ImmOrigin,
        bags_origin: ImmOrigin,
        groups_origin: ImmOrigin,
        sides_origin: ImmOrigin,
        choices_origin: ImmOrigin,
        targets_origin: ImmOrigin,
        relations_origin: ImmOrigin,
        stats_origin: MutOrigin,
        //,
    ](
        self,
        modalities: Span[Modality, modalities_origin],
        contents: TapeView[contents_origin],
        mut arena: Arena,
        config: TrainerStepConfig[
            bags_origin, groups_origin, sides_origin, choices_origin, targets_origin, relations_origin, stats_origin
        ] = TrainerStepConfig(),
    ) -> Deferred[
        _TrainerStepCall[origin_of(arena)],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](),
                _anchor[modalities_origin](),
                _anchor[contents_origin](),
                _anchor[bags_origin](),
                _anchor[groups_origin](),
                _anchor[sides_origin](),
                _anchor[choices_origin](),
                _anchor[targets_origin](),
                _anchor[relations_origin](),
                _anchor[stats_origin](),
            )
        ),
    ]:
        """Submits `TrainVerbs.step` with its embeddings; every input stays borrowed until final.
        Calls `uform_trainer_step`."""
        return {
            self.library,
            _step_call[origin_of(arena)](modalities, contents, Pointer(to=arena), config),
            self.trainer,
            self.executor,
        }

    def step[
        modalities_origin: ImmOrigin,
        contents_origin: ImmOrigin,
        bags_origin: ImmOrigin,
        groups_origin: ImmOrigin,
        sides_origin: ImmOrigin,
        choices_origin: ImmOrigin,
        targets_origin: ImmOrigin,
        relations_origin: ImmOrigin,
        stats_origin: MutOrigin,
        //,
    ](
        self,
        modalities: Span[Modality, modalities_origin],
        contents: TapeView[contents_origin],
        config: TrainerStepConfig[
            bags_origin, groups_origin, sides_origin, choices_origin, targets_origin, relations_origin, stats_origin
        ] = TrainerStepConfig(),
    ) -> Deferred[
        _TrainerStepCall[MutUntrackedOrigin],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](),
                _anchor[modalities_origin](),
                _anchor[contents_origin](),
                _anchor[bags_origin](),
                _anchor[groups_origin](),
                _anchor[sides_origin](),
                _anchor[choices_origin](),
                _anchor[targets_origin](),
                _anchor[relations_origin](),
                _anchor[stats_origin](),
            )
        ),
    ]:
        """Submits `TrainVerbs.step`; every input stays borrowed until final. Calls `uform_trainer_step`."""
        return {
            self.library,
            _step_call[MutUntrackedOrigin](
                modalities, contents, Optional[Pointer[Arena, MutUntrackedOrigin]](), config
            ),
            self.trainer,
            self.executor,
        }

    def save[
        path_origin: ImmOrigin, //
    ](self, path: StringSlice[path_origin], progress: Progress = Progress()) -> Deferred[
        _TrainerSaveCall, ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[path_origin]()))
    ]:
        """Submits `TrainVerbs.save`; the path stays borrowed until final. Calls `uform_trainer_save`."""
        return {self.library, _TrainerSaveCall(BytesView.of(path).arguments(), progress), self.trainer, self.executor}


# endregion Trainer
