/**
 *  @file python/train.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief CPython training: the mesh a run's processes meet in, the trainer, its keyword-built configuration
 *      structures, and the step and save verbs.
 *
 *  Training is GPU-only, so a trainer runs its verbs blocking on the executor it was created on, and its
 *  receivers name another completion or another executor of the same devices. The configuration arrives as
 *  keyword-built structures mirroring the C ones field by field, so a field left out is zero, which the library
 *  derives as its header documents.
 */
#include "lib.h"

#pragma region Enums

static char const *spell_modality(int value) { return uform_modality_name((uform_modality_t)value); }
static char const *spell_multimem_tier(int value) { return uform_multimem_tier_name((uform_multimem_tier_t)value); }

static char const *spell_objective(int value) { return uform_objective_name((uform_objective_t)value); }
static char const *spell_negatives(int value) { return uform_negatives_name((uform_negatives_t)value); }

/** The C interface spells no pair-loss form, so the binding names the four it declares. */
static char const *spell_pair_loss_form(int value) {
    switch (value) {
    case uform_pair_loss_form_derived_k: return "derived";
    case uform_pair_loss_form_balanced_k: return "balanced";
    case uform_pair_loss_form_ranked_k: return "ranked";
    case uform_pair_loss_form_sigmoid_k: return "sigmoid";
    default: return NULL;
    }
}

/** The C interface spells no late-interaction side, so the binding names the four it declares. */
static char const *spell_late_interaction_side(int value) {
    switch (value) {
    case uform_late_interaction_side_none_k: return "none";
    case uform_late_interaction_side_query_k: return "query";
    case uform_late_interaction_side_document_k: return "document";
    case uform_late_interaction_side_query_and_document_k: return "query-and-document";
    default: return NULL;
    }
}

uform_py_enum_t uform_py_modality = {.name = "Modality",
                                     .base = "IntEnum",
                                     .first = uform_modality_text_k,
                                     .last = uform_modality_audio_k,
                                     .spell = spell_modality};
uform_py_enum_t uform_py_objective = {.name = "Objective",
                                      .base = "IntEnum",
                                      .first = uform_objective_masked_k,
                                      .last = uform_objective_joint_k,
                                      .spell = spell_objective};
uform_py_enum_t uform_py_negatives = {.name = "Negatives",
                                      .base = "IntEnum",
                                      .first = uform_negatives_ambient_k,
                                      .last = uform_negatives_named_k,
                                      .spell = spell_negatives};
uform_py_enum_t uform_py_pair_loss_form = {.name = "PairLossForm",
                                           .base = "IntEnum",
                                           .first = uform_pair_loss_form_derived_k,
                                           .last = uform_pair_loss_form_sigmoid_k,
                                           .spell = spell_pair_loss_form};
uform_py_enum_t uform_py_late_interaction_side = {.name = "LateInteractionSide",
                                                  .base = "IntEnum",
                                                  .first = uform_late_interaction_side_none_k,
                                                  .last = uform_late_interaction_side_query_and_document_k,
                                                  .spell = spell_late_interaction_side};
uform_py_enum_t uform_py_multimem_tier = {.name = "MultimemTier",
                                          .base = "IntEnum",
                                          .first = uform_multimem_tier_switch_reduced_k,
                                          .last = uform_multimem_tier_peer_read_k,
                                          .spell = spell_multimem_tier};

#pragma endregion Enums

#pragma region Keyword Structures

static uform_py_field_t const ImageShape_fields[] = {
    {"patch_side", uform_py_field_size_k, offsetof(uform_image_shape_t, patch_side), NULL,
     "The square patch side images tile into, in pixels; 0 leaves images out of the run."},
    {"channels", uform_py_field_size_k, offsetof(uform_image_shape_t, channels), NULL,
     "Channels every pixel carries; 0 derives four, RGBA."},
};

uform_py_fields_t ImageShapeType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.ImageShape",
     .tp_doc = "ImageShape(*, patch_side=0, channels=0)\n\nThe image slice of a `BatchShape`."},
    ImageShape_fields,
    sizeof(ImageShape_fields) / sizeof(*ImageShape_fields),
};

static uform_py_field_t const VideoShape_fields[] = {
    {"patch_side", uform_py_field_size_k, offsetof(uform_video_shape_t, patch_side), NULL,
     "The square patch side one tube tiles into, in pixels; 0 leaves video out of the run."},
    {"tube_frames", uform_py_field_size_k, offsetof(uform_video_shape_t, tube_frames), NULL,
     "Frames one tube folds ahead of its patch; 0 derives two."},
    {"frame_stride", uform_py_field_size_k, offsetof(uform_video_shape_t, frame_stride), NULL,
     "Source frames between sampled ones; 0 derives one, every frame."},
    {"channels", uform_py_field_size_k, offsetof(uform_video_shape_t, channels), NULL,
     "Channels every pixel carries; 0 derives three."},
};

uform_py_fields_t VideoShapeType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.VideoShape",
     .tp_doc = "VideoShape(*, patch_side=0, tube_frames=0, frame_stride=0, channels=0)\n\n" //
               "The video slice of a `BatchShape`."},
    VideoShape_fields,
    sizeof(VideoShape_fields) / sizeof(*VideoShape_fields),
};

static uform_py_field_t const AudioShape_fields[] = {
    {"chunk_samples", uform_py_field_size_k, offsetof(uform_audio_shape_t, chunk_samples), NULL,
     "Raw samples per frame; 0 leaves audio out of the run."},
    {"sample_rate", uform_py_field_size_k, offsetof(uform_audio_shape_t, sample_rate), NULL,
     "The mono rate every recording is resampled to; 0 derives 16000."},
};

uform_py_fields_t AudioShapeType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.AudioShape",
     .tp_doc = "AudioShape(*, chunk_samples=0, sample_rate=0)\n\nThe audio slice of a `BatchShape`."},
    AudioShape_fields,
    sizeof(AudioShape_fields) / sizeof(*AudioShape_fields),
};

static uform_py_field_t const BatchShape_fields[] = {
    {"max_rows_per_device", uform_py_field_size_k, offsetof(uform_batch_shape_t, max_rows_per_device), NULL,
     "Rows one device's step packs."},
    {"max_segments_per_device", uform_py_field_size_k, offsetof(uform_batch_shape_t, max_segments_per_device), NULL,
     "Items one device's step carries across all its rows."},
    {"max_columns_per_row", uform_py_field_size_k, offsetof(uform_batch_shape_t, max_columns_per_row), NULL,
     "Positions one row spans, the sequence length."},
    {"max_bags_per_device", uform_py_field_size_k, offsetof(uform_batch_shape_t, max_bags_per_device), NULL,
     "Bags one device's step carries; 0 keeps every item its own routing unit."},
    {"max_choices_per_item", uform_py_field_size_k, offsetof(uform_batch_shape_t, max_choices_per_item), NULL,
     "Legal bytes one item offers to choose from at most; 0 makes no choices."},
    {"image", uform_py_field_struct_k, offsetof(uform_batch_shape_t, image), &ImageShapeType,
     "The `ImageShape`, or None leaving images out."},
    {"video", uform_py_field_struct_k, offsetof(uform_batch_shape_t, video), &VideoShapeType,
     "The `VideoShape`, or None leaving video out."},
    {"audio", uform_py_field_struct_k, offsetof(uform_batch_shape_t, audio), &AudioShapeType,
     "The `AudioShape`, or None leaving audio out."},
};

uform_py_fields_t BatchShapeType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.BatchShape",
     .tp_doc = "BatchShape(*, max_rows_per_device=0, max_segments_per_device=0, max_columns_per_row=0, " //
               "max_bags_per_device=0, max_choices_per_item=0, image=None, video=None, audio=None)\n\n"  //
               "What one device's step carries at most, whatever the modality mix, fixed at creation.\n" //
               "A nested shape takes its own structure or a tuple of its fields in order."},
    BatchShape_fields,
    sizeof(BatchShape_fields) / sizeof(*BatchShape_fields),
};

static uform_py_field_t const Masking_fields[] = {
    {"token_mask_probability", uform_py_field_float_k, offsetof(uform_masking_t, token_mask_probability), NULL,
     "Fraction of text positions covered by spans; 0 derives 0.15."},
    {"token_mean_span", uform_py_field_float_k, offsetof(uform_masking_t, token_mean_span), NULL,
     "Mean geometric span length in bytes; 0 derives 16."},
    {"replace_fraction", uform_py_field_float_k, offsetof(uform_masking_t, replace_fraction), NULL,
     "Of covered positions, the share shown as MASK; 0 derives 0.8."},
    {"random_fraction", uform_py_field_float_k, offsetof(uform_masking_t, random_fraction), NULL,
     "Of covered positions, the share replaced by a random byte; 0 derives 0.1."},
    {"image_mask_ratio", uform_py_field_float_k, offsetof(uform_masking_t, image_mask_ratio), NULL,
     "Fraction of image patches masked; 0 derives 0.75."},
    {"video_mask_ratio", uform_py_field_float_k, offsetof(uform_masking_t, video_mask_ratio), NULL,
     "Fraction of video tubes masked; 0 derives 0.75."},
    {"audio_mask_ratio", uform_py_field_float_k, offsetof(uform_masking_t, audio_mask_ratio), NULL,
     "Fraction of audio frames masked; 0 derives 0.75."},
    {"text_weight", uform_py_field_float_k, offsetof(uform_masking_t, text_weight), NULL,
     "The weight of the text head; 0 derives 1."},
    {"image_weight", uform_py_field_float_k, offsetof(uform_masking_t, image_weight), NULL,
     "The weight of the image head; 0 derives 1."},
    {"video_weight", uform_py_field_float_k, offsetof(uform_masking_t, video_weight), NULL,
     "The weight of the video head; 0 derives 1."},
    {"audio_weight", uform_py_field_float_k, offsetof(uform_masking_t, audio_weight), NULL,
     "The weight of the audio head; 0 derives 1."},
};

uform_py_fields_t MaskingType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.Masking",
     .tp_doc = "Masking(*, token_mask_probability=0.0, ..., audio_weight=0.0)\n\n" //
               "How the masked objective corrupts a batch and weighs its heads; every zero derives."},
    Masking_fields,
    sizeof(Masking_fields) / sizeof(*Masking_fields),
};

/** The index of `matryoshka_dimensions` among the fields of `Contrastive`, which the trainer converts itself. */
enum { contrastive_matryoshka_k = 0 };

static uform_py_field_t const Contrastive_fields[] = {
    {"matryoshka_dimensions", uform_py_field_object_k, 0, NULL,
     "The Matryoshka prefix widths, strictly ascending multiples of 8, at most 8; None scores the full width."},
    {"temperature_groups", uform_py_field_size_k, offsetof(uform_contrastive_t, temperature_groups), NULL,
     "Learnable (inverse temperature, bias) pairs; 0 derives one."},
    {"inverse_temperature", uform_py_field_float_k, offsetof(uform_contrastive_t, inverse_temperature), NULL,
     "The starting inverse temperature of every group; 0 derives 10 under the sigmoid form, SigLIP's, else 1."},
    {"bias", uform_py_field_float_k, offsetof(uform_contrastive_t, bias), NULL,
     "The starting logit bias of every group; 0 derives -10 under the sigmoid form, SigLIP's, else 0."},
    {"negatives", uform_py_field_enum_k, offsetof(uform_contrastive_t, negatives), &uform_py_negatives,
     "Which pairs repel, a `Negatives`."},
    {"ambient_repulsion_weight", uform_py_field_float_k, offsetof(uform_contrastive_t, ambient_repulsion_weight), NULL,
     "The weight of every pair no edge names, under `Negatives.AMBIENT`; 0 derives 1."},
    {"max_edges_per_device", uform_py_field_size_k, offsetof(uform_contrastive_t, max_edges_per_device), NULL,
     "Edges one device's step names per edge set at most; 0 derives `max_segments_per_device`."},
    {"pair_loss", uform_py_field_enum_k, offsetof(uform_contrastive_t, pair_loss), &uform_py_pair_loss_form,
     "The `PairLossForm` over each anchor's row; `PairLossForm.DERIVED` takes sigmoid."},
};

uform_py_fields_t ContrastiveType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.Contrastive",
     .tp_doc = "Contrastive(*, matryoshka_dimensions=None, temperature_groups=0, inverse_temperature=0.0, "    //
               "bias=0.0, negatives=Negatives.AMBIENT, ambient_repulsion_weight=0.0, max_edges_per_device=0, " //
               "pair_loss=PairLossForm.DERIVED)\n\n"                                                           //
               "How the contrastive loss scores the pooled embeddings; every zero derives."},
    Contrastive_fields,
    sizeof(Contrastive_fields) / sizeof(*Contrastive_fields),
};

static uform_py_field_t const LateInteraction_fields[] = {
    {"weight", uform_py_field_float_k, offsetof(uform_late_interaction_t, weight), NULL,
     "The loss weight; 0 trains no token head."},
    {"width", uform_py_field_size_k, offsetof(uform_late_interaction_t, width), NULL,
     "The token vectors' width, a multiple of 8; 0 derives 128."},
    {"inverse_temperature", uform_py_field_float_k, offsetof(uform_late_interaction_t, inverse_temperature), NULL,
     "The starting inverse temperature of every contrastive group; 0 derives 1."},
    {"bias", uform_py_field_float_k, offsetof(uform_late_interaction_t, bias), NULL,
     "The starting logit bias of every contrastive group."},
    {"max_query_span", uform_py_field_size_k, offsetof(uform_late_interaction_t, max_query_span), NULL,
     "Tokens one query segment contributes at most, its first past the prefix; 0 derives 64."},
    {"max_query_tokens_per_device", uform_py_field_size_k,
     offsetof(uform_late_interaction_t, max_query_tokens_per_device), NULL,
     "Query token rows one device's step holds at most; 0 derives `max_query_span` of every segment."},
    {"max_documents_per_device", uform_py_field_size_k, offsetof(uform_late_interaction_t, max_documents_per_device),
     NULL, "Document segments one device's step holds at most; 0 derives `max_segments_per_device`."},
    {"max_document_tokens_per_device", uform_py_field_size_k,
     offsetof(uform_late_interaction_t, max_document_tokens_per_device), NULL,
     "Document token rows one device's step holds at most; 0 derives the batch's tokens per device."},
    {"pair_loss", uform_py_field_enum_k, offsetof(uform_late_interaction_t, pair_loss), &uform_py_pair_loss_form,
     "The `PairLossForm` over each query's row of hard MaxSim scores; `PairLossForm.DERIVED` takes ranked."},
};

uform_py_fields_t LateInteractionType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.LateInteraction",
     .tp_doc = "LateInteraction(*, weight=0.0, width=0, inverse_temperature=0.0, bias=0.0, max_query_span=0, " //
               "max_query_tokens_per_device=0, max_documents_per_device=0, "                                   //
               "max_document_tokens_per_device=0, pair_loss=PairLossForm.DERIVED)\n\n"                         //
               "How the late-interaction loss scores the token vectors of query and document items, read\n"    //
               "when its weight is positive; every zero derives."},
    LateInteraction_fields,
    sizeof(LateInteraction_fields) / sizeof(*LateInteraction_fields),
};

static uform_py_field_t const Choices_fields[] = {
    {"weight", uform_py_field_float_k, offsetof(uform_choices_t, weight), NULL,
     "The loss weight; 0 trains no choices."},
    {"brier_weight", uform_py_field_float_k, offsetof(uform_choices_t, brier_weight), NULL,
     "The Brier term's weight beside the soft cross-entropy; 0 derives 0.5."},
};

uform_py_fields_t ChoicesType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.Choices",
     .tp_doc = "Choices(*, weight=0.0, brier_weight=0.0)\n\n"                                             //
               "Calibrated choices over the byte decoder, its logits read at a mask byte appended to a\n" //
               "state and restricted to the item's legal bytes; needs the batch's `max_choices_per_item`."},
    Choices_fields,
    sizeof(Choices_fields) / sizeof(*Choices_fields),
};

static uform_py_field_t const Schedule_fields[] = {
    {"learning_rate", uform_py_field_float_k, offsetof(uform_schedule_t, learning_rate), NULL,
     "The peak learning rate; 0 derives 5e-4."},
    {"min_learning_rate", uform_py_field_float_k, offsetof(uform_schedule_t, min_learning_rate), NULL,
     "The floor the cosine settles at; 0 derives 1e-5."},
    {"warmup_steps", uform_py_field_size_k, offsetof(uform_schedule_t, warmup_steps), NULL,
     "Steps from zero to the peak; 0 derives 400."},
    {"steps", uform_py_field_size_k, offsetof(uform_schedule_t, steps), NULL,
     "Steps the run takes, which the cosine spans; 0 is refused."},
};

uform_py_fields_t ScheduleType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.Schedule",
     .tp_doc = "Schedule(*, learning_rate=0.0, min_learning_rate=0.0, warmup_steps=0, steps=0)\n\n" //
               "The learning-rate schedule: a linear warmup to the peak, then a cosine decay to the floor."},
    Schedule_fields,
    sizeof(Schedule_fields) / sizeof(*Schedule_fields),
};

/** The indices of the two paths among the fields of `MeshOptions`, which the bootstrap converts itself. */
enum { mesh_options_directory_k = 3, mesh_options_job_k = 4 };

static uform_py_field_t const MeshOptions_fields[] = {
    {"host_domain_index", uform_py_field_size_k, offsetof(uform_mesh_options_t, host_domain_index), NULL,
     "This process's index among `host_domain_count`."},
    {"host_domain_count", uform_py_field_size_k, offsetof(uform_mesh_options_t, host_domain_count), NULL,
     "Processes in the run, one per host; 0 derives one, a run with no peers."},
    {"gpus_per_multimem_domain", uform_py_field_size_k, offsetof(uform_mesh_options_t, gpus_per_multimem_domain), NULL,
     "GPUs one NVLink multicast domain spans; 0 derives the host."},
    {"directory", uform_py_field_object_k, 0, NULL,
     "The directory every host sees, a str, bytes or os.PathLike passed as the OS's bytes; required with peers."},
    {"job", uform_py_field_object_k, 0, NULL,
     "The run's unique name, a str naming one directory inside `directory`; required with peers."},
    {"timeout_seconds", uform_py_field_double_k, offsetof(uform_mesh_options_t, timeout_seconds), NULL,
     "Seconds any wait for a peer lasts before the call fails; 0 derives 120."},
};

uform_py_fields_t MeshOptionsType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.MeshOptions",
     .tp_doc = "MeshOptions(*, host_domain_index=0, host_domain_count=0, gpus_per_multimem_domain=0, " //
               "directory=None, job=None, timeout_seconds=0.0)\n\n"                                    //
               "Where this process sits among the processes training one model, and how they meet."},
    MeshOptions_fields,
    sizeof(MeshOptions_fields) / sizeof(*MeshOptions_fields),
};

/** The indices of the three buffers among the fields of `Edges`. */
enum { edges_offsets_k = 0, edges_ids_k = 1, edges_weights_k = 2 };

static uform_py_field_t const Edges_fields[] = {
    {"offsets", uform_py_field_object_k, 0, NULL,
     "The step's count + 1 non-decreasing size_t offsets into `ids`; None names no edges."},
    {"ids", uform_py_field_object_k, 0, NULL,
     "A uint32 buffer: item `i` links to items ``ids[offsets[i]:offsets[i + 1]]``, ascending within each run."},
    {"weights", uform_py_field_object_k, 0, NULL, "The float32 weight of each edge; None weighs every edge one."},
};

uform_py_fields_t EdgesType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.Edges",
     .tp_doc = "Edges(*, offsets=None, ids=None, weights=None)\n\n"                                           //
               "One edge set over a step's items, as compressed sparse rows of buffers, borrowed until the\n" //
               "step is final."},
    Edges_fields,
    sizeof(Edges_fields) / sizeof(*Edges_fields),
};

/** The indices of the three edge sets among the fields of `Relations`. */
enum { relations_attract_k = 0, relations_repel_k = 1, relations_ignore_k = 2 };

static uform_py_field_t const Relations_fields[] = {
    {"attract", uform_py_field_struct_k, offsetof(uform_relations_view_t, attract), &EdgesType,
     "Pairs pulled together, an `Edges` or an (offsets, ids[, weights]) tuple."},
    {"repel", uform_py_field_struct_k, offsetof(uform_relations_view_t, repel), &EdgesType, "Pairs pushed apart."},
    {"ignore", uform_py_field_struct_k, offsetof(uform_relations_view_t, ignore), &EdgesType,
     "Pairs skipped, like known false negatives; their weights are not read."},
};

uform_py_fields_t RelationsType = {
    {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "uform.Relations",
     .tp_doc = "Relations(*, attract=None, repel=None, ignore=None)\n\n"                                      //
               "The contrastive targets of one step: a pair `ignore` names is skipped, one `attract` names\n" //
               "pulls together, one `repel` names pushes apart, and every other pair follows the run's\n"     //
               "`negatives`."},
    Relations_fields,
    sizeof(Relations_fields) / sizeof(*Relations_fields),
};

#pragma endregion Keyword Structures

#pragma region Mesh

/** One process's place among the processes of a run. */
typedef struct Mesh {
    PyObject ob_base;
    uform_mesh_t handle;
} Mesh;

static char const doc_Mesh[] =                                                                 //
    "Mesh\n"                                                                                   //
    "\n"                                                                                       //
    "One process's place among the processes of a run: the topology every process agreed on\n" //
    "at bootstrap, and the InfiniBand domain joining its NVLink domains. Made once with\n"     //
    "`bootstrap` and borrowed by every trainer, which holds it until the trainer is freed.";

static PyObject *Mesh_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    PyErr_SetString(PyExc_TypeError, "Mesh is built through Mesh.bootstrap()");
    return NULL;
}

static void Mesh_dealloc(Mesh *self) {
    Py_BEGIN_ALLOW_THREADS;
    uform_mesh_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/** Reads an optional path field into @p view, keeping its encoding in @p decoded; returns 0 or -1. */
static int optional_path(PyObject *object, char const *field, uform_bytes_view_t *view, PyObject **decoded) {
    *decoded = NULL;
    if (!object || object == Py_None) return 0;
    return uform_py_path_from_object(object, field, view, decoded);
}

/** Lends an optional name field, a @c str, as its UTF-8 bytes into @p view, which live as long
 *  as the string; returns 0 or -1. */
static int optional_name(PyObject *object, char const *field, uform_bytes_view_t *view) {
    if (!object || object == Py_None) return 0;
    if (!PyUnicode_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' option takes a str, got %S", field, Py_TYPE(object));
        return -1;
    }
    Py_ssize_t length = 0;
    char const *const text = PyUnicode_AsUTF8AndSize(object, &length);
    if (!text) return -1;
    *view = (uform_bytes_view_t) {(uint8_t const *)text, (size_t)length};
    return 0;
}

static char const doc_Mesh_bootstrap[] =                                                         //
    "bootstrap(executor, options, /) -> Mesh\n"                                                  //
    "\n"                                                                                         //
    "Joins this process to the run's mesh through files in `options.directory`; every process\n" //
    "of the run calls it with the same count, multicast extent, directory and job, and peers\n"  //
    "wait for each other.\n"                                                                     //
    "Calls `uform_mesh_bootstrap`.\n"                                                            //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  executor (Executor): A VRAM executor whose devices are this process's members.\n"         //
    "  options (MeshOptions): Where this process sits and how the processes meet.\n"             //
    "\n"                                                                                         //
    "Example:\n"                                                                                 //
    "  >>> mesh = uform.Mesh.bootstrap(executor, uform.MeshOptions(host_domain_count=1))  # doctest: +SKIP";

static PyObject *Mesh_bootstrap(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    PyObject *executor_object, *options_object, *directory = NULL, *result = NULL;
    uform_py_argument_t const arguments[] = {{"executor", &executor_object}, {"options", &options_object}};
    uform_executor_t executor = NULL;
    uform_mesh_bootstrap_t bootstrap = {0};
    if (uform_py_arguments_parse("bootstrap", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_executor_from_object(executor_object, &executor) != 0 ||
        uform_py_fields_write(options_object, "options", &MeshOptionsType, &bootstrap.options) != 0)
        return NULL;
    if (options_object != Py_None &&
        (optional_path(uform_py_fields_value(options_object, mesh_options_directory_k), "directory",
                       &bootstrap.options.directory, &directory) != 0 ||
         optional_name(uform_py_fields_value(options_object, mesh_options_job_k), "job", &bootstrap.options.job) != 0))
        goto release;

    uform_mesh_t handle = NULL;
    uform_outcome_t outcome = {0};
    bootstrap.mesh = &handle;
    bootstrap.executor = executor;
    bootstrap.outcome = &outcome;
    Py_BEGIN_ALLOW_THREADS;
    uform_mesh_bootstrap(&bootstrap);
    Py_END_ALLOW_THREADS;
    if (uform_py_raise_on_outcome(&outcome) != 0) {
        uform_mesh_free(handle);
        goto release;
    }
    Mesh *const mesh = (Mesh *)type->tp_alloc(type, 0);
    if (!mesh) {
        uform_mesh_free(handle);
        goto release;
    }
    mesh->handle = handle;
    result = (PyObject *)mesh;

release:
    Py_XDECREF(directory);
    return result;
}

static PyStructSequence_Field MeshDescription_fields[] = {
    {"gpus_per_host_domain", "GPUs one process drives."},
    {"gpus_per_multimem_domain", "GPUs one NVLink domain spans."},
    {"gpus_per_colocated_domain", "GPUs in the whole run."},
    {"host_domain_index", "This process's index among the run's processes."},
    {"first_gpu_index", "The run-wide index of this process's first GPU."},
    {"multimem_tier", "How its NVLink domains reduce, a `MultimemTier`."},
    {"gpu_class", "The `GPUClass` every GPU of the run dispatches as, or None."},
    {NULL, NULL},
};

PyStructSequence_Desc uform_py_mesh_description_desc = {
    "uform.MeshDescription", "What a mesh reports about itself, as `uform_mesh_description_t` holds it.",
    MeshDescription_fields, 7};

PyTypeObject MeshDescriptionType;

static char const doc_Mesh_describe[] =                          //
    "describe() -> MeshDescription\n"                            //
    "\n"                                                         //
    "Reports the mesh's shape and this process's place in it.\n" //
    "Calls `uform_mesh_describe`.";

static PyObject *Mesh_describe(Mesh *self, PyObject *unused) {
    uform_mesh_description_t description = {0};
    uform_outcome_t outcome = {0};
    uform_mesh_describe_t arguments = {.mesh = self->handle, .outcome = &outcome, .description = &description};
    Py_BEGIN_ALLOW_THREADS;
    uform_mesh_describe(&arguments);
    Py_END_ALLOW_THREADS;
    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    PyObject *fields[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    (void)((fields[0] = PyLong_FromSize_t(description.gpus_per_host_domain)) &&
           (fields[1] = PyLong_FromSize_t(description.gpus_per_multimem_domain)) &&
           (fields[2] = PyLong_FromSize_t(description.gpus_per_colocated_domain)) &&
           (fields[3] = PyLong_FromSize_t(description.host_domain_index)) &&
           (fields[4] = PyLong_FromSize_t(description.first_gpu_index)) &&
           (fields[5] = uform_py_enum_to_object(&uform_py_multimem_tier, (int)description.multimem_tier)) &&
           (fields[6] = uform_py_class_to_object(&uform_py_gpu_class, (int)description.gpu_class)));
    return uform_py_struct_sequence_steal(&MeshDescriptionType, fields, 7);
}

static PyMethodDef Mesh_methods[] = {
    {"bootstrap", (PyCFunction)Mesh_bootstrap, uform_py_method_flags_k | METH_CLASS, doc_Mesh_bootstrap},
    {"describe", (PyCFunction)Mesh_describe, METH_NOARGS, doc_Mesh_describe},
    {NULL, NULL, 0, NULL},
};

PyTypeObject MeshType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.Mesh",
    .tp_doc = doc_Mesh,
    .tp_basicsize = sizeof(Mesh),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Mesh_tp_new,
    .tp_dealloc = (destructor)Mesh_dealloc,
    .tp_methods = Mesh_methods,
};

#pragma endregion Mesh

#pragma region Trainer

/** A training run on the devices of the executor it was created on. */
typedef struct Trainer {
    PyObject ob_base;
    uform_trainer_t handle;

    /** The @c Executor the trainer was created on, held, which its own verbs block on. */
    PyObject *executor;

    /** The @c Mesh the trainer borrows until it is freed, held, or NULL for a run of one process. */
    PyObject *mesh;
} Trainer;

static char const doc_Trainer[] =                                                                  //
    "Trainer\n"                                                                                    //
    "\n"                                                                                           //
    "A training run: one replica of the tower per device, its gradients, the optimizer moments,\n" //
    "the reconstruction heads, the learnable temperatures and the collectives joining the\n"       //
    "replicas across the mesh. Training is GPU-only, so `init_vram` is its one constructor.\n"     //
    "\n"                                                                                           //
    "The trainer runs `step` and `save` blocking on the executor it was created on;\n"             //
    "`on(executor)` blocks on another executor of the same devices, and `deferred(executor)`\n"    //
    "submits there and answers a `Deferred`.";

static PyObject *Trainer_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    PyErr_SetString(PyExc_TypeError, "Trainer is built through Trainer.init_vram()");
    return NULL;
}

static void Trainer_dealloc(Trainer *self) {
    Py_BEGIN_ALLOW_THREADS;
    uform_trainer_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_XDECREF(self->mesh);
    Py_XDECREF(self->executor);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/** Reads a sequence of positive widths into a @c PyMem array; NULL and zero for None. Returns 0 or -1. */
static int widths_from_object(PyObject *object, char const *field, size_t **widths, size_t *count) {
    *widths = NULL;
    *count = 0;
    if (!object || object == Py_None) return 0;
    PyObject *const sequence = PySequence_Fast(object, "expected a sequence of widths");
    if (!sequence) return -1;
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    *widths = (size_t *)PyMem_Malloc((length ? (size_t)length : 1) * sizeof(size_t));
    int failed = !*widths;
    if (failed) PyErr_NoMemory();
    for (Py_ssize_t position = 0; !failed && position != length; ++position)
        failed = uform_py_size_from_object(PySequence_Fast_GET_ITEM(sequence, position), field, &(*widths)[position]);
    Py_DECREF(sequence);
    if (failed) {
        PyMem_Free(*widths);
        *widths = NULL;
        return -1;
    }
    *count = (size_t)length;
    return 0;
}

static char const doc_Trainer_init_vram[] =                                                          //
    "init_vram(executor, /, *, mesh=None, resume=None, depth=0, hidden=0, batch=None, "              //
    "objective=Objective.MASKED, masking=None, contrastive=None, late_interaction=None, "            //
    "choices=None, schedule=None, seed=0, progress=None, progress_steps=0) -> Trainer\n"             //
    "\n"                                                                                             //
    "Builds one replica per device of `executor` and starts or resumes the run; every process\n"     //
    "of the run calls it with the same geometry, objective and seed.\n"                              //
    "Calls `uform_trainer_init_vram`.\n"                                                             //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  executor (Executor): A VRAM executor, every device of which holds one replica.\n"             //
    "  mesh (Mesh, optional): The mesh made over this same executor; None runs alone.\n"             //
    "  resume (str | bytes | os.PathLike, optional): A checkpoint to continue from at its\n"         //
    "    saved step; one of another geometry raises `ConfigMismatchError`.\n"                        //
    "  depth (int): Transformer blocks stacked.\n"                                                   //
    "  hidden (int): Embedding width, a multiple of 128.\n"                                          //
    "  batch (BatchShape): What one device's step carries at most.\n"                                //
    "  objective (Objective, optional): Which losses a step relaxes.\n"                              //
    "  masking (Masking, optional): The masked objective's corruption and weights.\n"                //
    "  contrastive (Contrastive, optional): The contrastive loss's scoring.\n"                       //
    "  late_interaction (LateInteraction, optional): The token-vector loss beside the pooled one.\n" //
    "  choices (Choices, optional): The calibrated choices beside every other term.\n"               //
    "  schedule (Schedule): The learning-rate schedule.\n"                                           //
    "  seed (int, optional): The base of every keyed draw.\n"                                        //
    "  progress (callable, optional): Called with (completed, total) tensors placed on a resume.\n"  //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject *Trainer_init_vram(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {
    PyObject *executor_object, *mesh_object, *resume_object, *depth_object, *hidden_object, *batch_object,
        *objective_object, *masking_object, *contrastive_object, *late_interaction_object, *choices_object,
        *schedule_object, *seed_object, *progress_object, *progress_steps_object, *resume_decoded = NULL,
                                                                                  *result = NULL;
    uform_py_argument_t const arguments[] = {
        {"executor", &executor_object},
        {"mesh", &mesh_object},
        {"resume", &resume_object},
        {"depth", &depth_object},
        {"hidden", &hidden_object},
        {"batch", &batch_object},
        {"objective", &objective_object},
        {"masking", &masking_object},
        {"contrastive", &contrastive_object},
        {"late_interaction", &late_interaction_object},
        {"choices", &choices_object},
        {"schedule", &schedule_object},
        {"seed", &seed_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
    };
    uform_trainer_init_vram_t init = {0};
    uform_py_call_t call = {0};
    uform_py_callback_t progress = {0};
    size_t *matryoshka = NULL, progress_steps = 0;
    int objective = uform_objective_masked_k;
    if (uform_py_arguments_parse("init_vram", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_executor_from_object(executor_object, &init.executor) != 0 ||
        uform_py_size_from_object(depth_object, "depth", &init.depth) != 0 ||
        uform_py_size_from_object(hidden_object, "hidden", &init.hidden) != 0 ||
        uform_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (objective_object &&
         uform_py_enum_from_object(objective_object, "objective", &uform_py_objective, &objective) != 0) ||
        uform_py_fields_write(batch_object, "batch", &BatchShapeType, &init.batch) != 0 ||
        uform_py_fields_write(masking_object, "masking", &MaskingType, &init.masking) != 0 ||
        uform_py_fields_write(contrastive_object, "contrastive", &ContrastiveType, &init.contrastive) != 0 ||
        uform_py_fields_write(late_interaction_object, "late_interaction", &LateInteractionType,
                              &init.late_interaction) != 0 ||
        uform_py_fields_write(choices_object, "choices", &ChoicesType, &init.choices) != 0 ||
        uform_py_fields_write(schedule_object, "schedule", &ScheduleType, &init.schedule) != 0 ||
        optional_path(resume_object, "resume", &init.resume, &resume_decoded) != 0 ||
        uform_py_callback_from_object(progress_object, "progress", &call, &progress) != 0)
        goto release;
    if (mesh_object && mesh_object != Py_None && !Py_IS_TYPE(mesh_object, &MeshType)) {
        PyErr_Format(PyExc_TypeError, "the 'mesh' option must be a Mesh or None, got %S", Py_TYPE(mesh_object));
        goto release;
    }
    if (contrastive_object && contrastive_object != Py_None &&
        widths_from_object(uform_py_fields_value(contrastive_object, contrastive_matryoshka_k), "matryoshka_dimensions",
                           &matryoshka, &init.contrastive.matryoshka_dimensions_count) != 0)
        goto release;
    if (seed_object) {
        init.seed = (uint64_t)PyLong_AsUnsignedLongLong(seed_object);
        if (PyErr_Occurred()) goto release;
    }

    uform_trainer_t handle = NULL;
    init.trainer = &handle;
    init.outcome = &call.outcome;
    init.objective = (uform_objective_t)objective;
    init.contrastive.matryoshka_dimensions = matryoshka;
    init.mesh = mesh_object && mesh_object != Py_None ? ((Mesh *)mesh_object)->handle : NULL;
    init.progress = uform_py_progress_registered(&progress, progress_steps);
    Py_BEGIN_ALLOW_THREADS;
    uform_trainer_init_vram(&init);
    Py_END_ALLOW_THREADS;
    if (call.raised || uform_py_raise_on_outcome(&call.outcome) != 0) {
        uform_trainer_free(handle);
        if (call.raised) PyErr_SetRaisedException(call.raised);
        goto release;
    }
    Trainer *const trainer = (Trainer *)type->tp_alloc(type, 0);
    if (!trainer) {
        uform_trainer_free(handle);
        goto release;
    }
    trainer->handle = handle;
    trainer->executor = Py_NewRef(executor_object);
    trainer->mesh = init.mesh ? Py_NewRef(mesh_object) : NULL;
    result = (PyObject *)trainer;

release:
    PyMem_Free(matryoshka);
    Py_XDECREF(resume_decoded);
    Py_XDECREF(progress.callable);
    return result;
}

/** Reads the receiver of a trainer verb: the trainer itself blocks on the executor it was created on. */
static int trainer_receiver(PyObject *self, uform_py_receiver_t *receiver) {
    if (uform_py_receiver_from_object(self, &TrainerType, receiver) != 0) return -1;
    if (receiver->target != self) return 0;
    return uform_py_executor_from_object(((Trainer *)self)->executor, &receiver->executor);
}

static PyObject *Trainer_on(PyObject *self, PyObject *executor) { return uform_py_on_new(&OnType, self, executor); }

static PyObject *Trainer_deferred(PyObject *self, PyObject *executor) {
    return uform_py_on_new(&DeferredOnType, self, executor);
}

#pragma endregion Trainer

#pragma region Steps

/** Telemetry a step fills, overwritten by every call it is passed to. */
typedef struct TrainerStepStats {
    PyObject ob_base;
    uform_trainer_step_stats_t stats;
} TrainerStepStats;

static PyObject *TrainerStepStats_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "TrainerStepStats() takes no arguments");
        return NULL;
    }
    return type->tp_alloc(type, 0);
}

static void TrainerStepStats_dealloc(TrainerStepStats *self) { Py_TYPE(self)->tp_free((PyObject *)self); }

/** Reads the @c size_t counter at the byte offset @p closure names inside the report. */
static PyObject *TrainerStepStats_get_count(TrainerStepStats *self, void *closure) {
    size_t value = 0;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(self);
#endif
    value = *(size_t const *)(void const *)((char const *)&self->stats + (size_t)closure);
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    return PyLong_FromSize_t(value);
}

/** Reads the @c float at the byte offset @p closure names inside the report. */
static PyObject *TrainerStepStats_get_float(TrainerStepStats *self, void *closure) {
    float value = 0;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(self);
#endif
    value = *(float const *)(void const *)((char const *)&self->stats + (size_t)closure);
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    return PyFloat_FromDouble(value);
}

static PyGetSetDef TrainerStepStats_getset[] = {
    {"step", (getter)TrainerStepStats_get_count, NULL, "The optimizer step the call applied, counted from one.",
     (void *)offsetof(uform_trainer_step_stats_t, step)},
    {"learning_rate", (getter)TrainerStepStats_get_float, NULL, "The learning rate it applied.",
     (void *)offsetof(uform_trainer_step_stats_t, learning_rate)},
    {"loss", (getter)TrainerStepStats_get_float, NULL, "The total loss the step minimized, its world mean.",
     (void *)offsetof(uform_trainer_step_stats_t, loss)},
    {"contrastive_loss", (getter)TrainerStepStats_get_float, NULL, "The contrastive term; 0 when absent.",
     (void *)offsetof(uform_trainer_step_stats_t, contrastive_loss)},
    {"masked_text_loss", (getter)TrainerStepStats_get_float, NULL, "The text reconstruction term, unweighted.",
     (void *)offsetof(uform_trainer_step_stats_t, masked_text_loss)},
    {"masked_image_loss", (getter)TrainerStepStats_get_float, NULL, "The image reconstruction term, unweighted.",
     (void *)offsetof(uform_trainer_step_stats_t, masked_image_loss)},
    {"masked_video_loss", (getter)TrainerStepStats_get_float, NULL, "The video reconstruction term, unweighted.",
     (void *)offsetof(uform_trainer_step_stats_t, masked_video_loss)},
    {"masked_audio_loss", (getter)TrainerStepStats_get_float, NULL, "The audio reconstruction term, unweighted.",
     (void *)offsetof(uform_trainer_step_stats_t, masked_audio_loss)},
    {"late_interaction_loss", (getter)TrainerStepStats_get_float, NULL, "The late-interaction term, weighted.",
     (void *)offsetof(uform_trainer_step_stats_t, late_interaction_loss)},
    {"choices_loss", (getter)TrainerStepStats_get_float, NULL,
     "The choices term over the items that made a choice, unweighted.",
     (void *)offsetof(uform_trainer_step_stats_t, choices_loss)},
    {"truncated", (getter)TrainerStepStats_get_count, NULL, "Items cut at a row's width.",
     (void *)offsetof(uform_trainer_step_stats_t, truncated)},
    {"dropped", (getter)TrainerStepStats_get_count, NULL,
     "Segments dropped whole because the batch had no room for them, since the last stats read.",
     (void *)offsetof(uform_trainer_step_stats_t, dropped)},
    {NULL, NULL, NULL, NULL, NULL},
};

static char const doc_TrainerStepStats[] =                                                   //
    "TrainerStepStats()\n"                                                                   //
    "\n"                                                                                     //
    "What `step` reports, overwritten by every call it is passed to, every loss its world\n" //
    "mean. Passing none skips the readback of the losses.";

PyTypeObject TrainerStepStatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.TrainerStepStats",
    .tp_doc = doc_TrainerStepStats,
    .tp_basicsize = sizeof(TrainerStepStats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = TrainerStepStats_tp_new,
    .tp_dealloc = (destructor)TrainerStepStats_dealloc,
    .tp_getset = TrainerStepStats_getset,
};

/** One edge set's buffers, lent until the step is final. */
typedef struct edges_t {
    Py_buffer offsets, ids, weights;
} edges_t;

static void edges_release(edges_t *edges) {
    if (edges->offsets.obj) PyBuffer_Release(&edges->offsets);
    if (edges->ids.obj) PyBuffer_Release(&edges->ids);
    if (edges->weights.obj) PyBuffer_Release(&edges->weights);
}

/** Acquires @p object as a C-contiguous buffer of @p item_bytes items of format @p format; returns 0 or -1. */
static int edges_buffer(PyObject *object, char const *field, Py_ssize_t item_bytes, char const *formats,
                        Py_buffer *buffer) {
    if (PyObject_GetBuffer(object, buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) return -1;
    char const *format = buffer->format ? buffer->format : "B";
    while (*format == '@' || *format == '=' || *format == '<') ++format;
    if (buffer->itemsize == item_bytes && strchr(formats, *format)) return 0;
    return uform_py_raise(uform_status_unsupported_layout_k, "the '%s' buffer holds items of another type", field);
}

/**
 *  @brief Lends the edge set @p object - an @c Edges or None - over @p count items into @p view, checking every
 *      offset stays inside its ids and every weight pairs with one.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int edges_from_object(PyObject *object, char const *field, size_t count, edges_t *edges,
                             uform_edges_view_t *view) {
    if (!object || object == Py_None || uform_py_fields_value(object, edges_offsets_k) == Py_None) return 0;
    PyObject *const ids = uform_py_fields_value(object, edges_ids_k);
    PyObject *const weights = uform_py_fields_value(object, edges_weights_k);
    if (edges_buffer(uform_py_fields_value(object, edges_offsets_k), field, sizeof(size_t), "NQL", &edges->offsets) !=
            0 ||
        edges_buffer(ids, field, sizeof(uint32_t), "I", &edges->ids) != 0 ||
        (weights != Py_None && edges_buffer(weights, field, sizeof(float), "f", &edges->weights) != 0))
        return -1;
    size_t const *const offsets = (size_t const *)edges->offsets.buf;
    size_t const ids_count = (size_t)edges->ids.len / sizeof(uint32_t);
    if ((size_t)edges->offsets.len != (count + 1) * sizeof(size_t) || offsets[count] > ids_count ||
        (edges->weights.obj && (size_t)edges->weights.len != ids_count * sizeof(float)))
        return uform_py_raise(uform_status_shape_mismatch_k,
                              "the '%s' edges need %zu offsets ending inside their ids, and one weight per id", field,
                              count + 1);
    *view = (uform_edges_view_t) {offsets, (uint32_t const *)edges->ids.buf,
                                  edges->weights.obj ? (float const *)edges->weights.buf : NULL};
    return 0;
}

/**
 *  @brief Lends the @b optional choices and their targets over @p count items: both tapes or
 *      neither, one entry per item each, the library checking each target entry is as long as
 *      its choice entry.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int choices_from_objects(PyObject *choices_object, PyObject *targets_object, size_t count,
                                uform_py_tape_t *choices, uform_py_tape_t *targets) {
    int const with_choices = choices_object && choices_object != Py_None;
    if (with_choices != (targets_object && targets_object != Py_None))
        return uform_py_raise(uform_status_invalid_argument_k, "pass both 'choices' and 'choice_targets', or neither");
    if (!with_choices) return 0;
    if (uform_py_tape_from_object(choices_object, "choices", uform_py_element_text_k, choices) != 0 ||
        uform_py_tape_from_object(targets_object, "choice_targets", uform_py_element_samples_k, targets) != 0)
        return -1;
    if (choices->count != count || targets->count != count)
        return uform_py_raise(uform_status_shape_mismatch_k,
                              "the 'choices' and 'choice_targets' need one entry per item, %zu", count);
    return 0;
}

/** A @c uform_trainer_step call, the inputs it owns or borrows until it is final, and what it lands. */
typedef struct step_call_t {
    uform_py_call_t call;
    uform_trainer_step_t arguments;
    uform_py_tape_t contents, choices, choice_targets;
    uform_modality_t *modalities;
    uform_late_interaction_side_t *sides;
    uform_py_words_t bags, temperature_groups;
    edges_t attract, repel, ignore;
    uform_rows_view_t embeddings;
    uform_trainer_step_stats_t stats;

    /** The caller's report the stats are copied into once the call succeeds, held, or NULL. */
    PyObject *stats_object;
} step_call_t;

static void step_submit(void *call) { uform_trainer_step(&((step_call_t *)call)->arguments); }

static PyObject *step_answer(void *erased) {
    step_call_t *const call = (step_call_t *)erased;
    if (call->stats_object) {
#if PY_VERSION_HEX >= 0x030D0000
        Py_BEGIN_CRITICAL_SECTION(call->stats_object);
#endif
        ((TrainerStepStats *)call->stats_object)->stats = call->stats;
#if PY_VERSION_HEX >= 0x030D0000
        Py_END_CRITICAL_SECTION();
#endif
    }
    if (!call->call.arena) Py_RETURN_NONE;
    return uform_py_rows_answer(&ArenaBufferType, call->call.arena, &call->embeddings, call->contents.count);
}

static void step_release(void *erased) {
    step_call_t *const call = (step_call_t *)erased;
    uform_py_tape_release(&call->contents);
    PyMem_Free(call->modalities);
    call->modalities = NULL;
    PyMem_Free(call->sides);
    call->sides = NULL;
    uform_py_words_release(&call->bags);
    uform_py_words_release(&call->temperature_groups);
    edges_release(&call->attract);
    edges_release(&call->repel);
    edges_release(&call->ignore);
    uform_py_tape_release(&call->choices);
    uform_py_tape_release(&call->choice_targets);
    Py_CLEAR(call->stats_object);
    uform_py_call_release(&call->call);
}

static int step_traverse(void *erased, visitproc visit, void *arg) {
    Py_VISIT(((step_call_t *)erased)->stats_object);
    return 0;
}

static uform_py_verb_t const step_verb = {sizeof(step_call_t), step_submit, step_answer, step_release, step_traverse};

/** Gathers one modality per item into a @c PyMem array of @p count; returns 0 or -1 with an exception set. */
static int modalities_from_object(PyObject *object, size_t count, uform_modality_t **modalities) {
    PyObject *const sequence = PySequence_Fast(object, "expected a sequence of uform.Modality");
    if (!sequence) return -1;
    if ((size_t)PySequence_Fast_GET_SIZE(sequence) != count) {
        Py_DECREF(sequence);
        return uform_py_raise(uform_status_shape_mismatch_k, "the 'modalities' need one entry per item, %zu", count);
    }
    *modalities = (uform_modality_t *)PyMem_Malloc((count ? count : 1) * sizeof(uform_modality_t));
    int failed = !*modalities;
    if (failed) PyErr_NoMemory();
    for (size_t position = 0; !failed && position != count; ++position) {
        int modality = 0;
        failed = uform_py_enum_from_object(PySequence_Fast_GET_ITEM(sequence, position), "modalities",
                                           &uform_py_modality, &modality) != 0;
        if (!failed) (*modalities)[position] = (uform_modality_t)modality;
    }
    Py_DECREF(sequence);
    return failed ? -1 : 0;
}

/** Gathers one side per item into a @c PyMem array of @p count; None leaves it NULL.
 *  Returns 0 or -1 with an exception set. */
static int sides_from_object(PyObject *object, size_t count, uform_late_interaction_side_t **sides) {
    if (!object || object == Py_None) return 0;
    PyObject *const sequence = PySequence_Fast(object, "expected a sequence of uform.LateInteractionSide");
    if (!sequence) return -1;
    if ((size_t)PySequence_Fast_GET_SIZE(sequence) != count) {
        Py_DECREF(sequence);
        return uform_py_raise(uform_status_shape_mismatch_k, "the 'sides' need one entry per item, %zu", count);
    }
    *sides = (uform_late_interaction_side_t *)PyMem_Malloc((count ? count : 1) * sizeof(uform_late_interaction_side_t));
    int failed = !*sides;
    if (failed) PyErr_NoMemory();
    for (size_t position = 0; !failed && position != count; ++position) {
        int side = 0;
        failed = uform_py_enum_from_object(PySequence_Fast_GET_ITEM(sequence, position), "sides",
                                           &uform_py_late_interaction_side, &side) != 0;
        if (!failed) (*sides)[position] = (uform_late_interaction_side_t)side;
    }
    Py_DECREF(sequence);
    return failed ? -1 : 0;
}

static char const doc_Trainer_step[] =                                                            //
    "step(modalities, contents, /, *, bags=None, temperature_groups=None, sides=None, "           //
    "choices=None, choice_targets=None, relations=None, arena=None, "                             //
    "outputs_scalar=Scalar.F32, stats=None) "                                                     //
    "-> ArenaBuffer | None\n"                                                                     //
    "\n"                                                                                          //
    "Applies one optimizer step over this process's items and every peer's: forward, losses,\n"   //
    "backward, the world gradient reduction and the update. Every process calls it once per\n"    //
    "step.\n"                                                                                     //
    "Calls `uform_trainer_step`.\n"                                                               //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  modalities (Sequence[Modality]): The modality of each item.\n"                             //
    "  contents (Sequence[str | bytes] | tuple[buffer, buffer]): UTF-8 text or an encoded file\n" //
    "    per item, gathered into one tape, or a (data, offsets) pair lent as is.\n"               //
    "  bags (buffer | Sequence[int], optional): The bag of each item; items sharing one land\n"   //
    "    on one device. None keeps every item its own bag.\n"                                     //
    "  temperature_groups (buffer | Sequence[int], optional): The temperature group of each\n"    //
    "    item; None puts every item in group zero.\n"                                             //
    "  sides (Sequence[LateInteractionSide], optional): The late-interaction side of each\n"      //
    "    item; None leaves every item out of late interaction.\n"                                 //
    "  choices (Sequence[str | bytes] | tuple[buffer, buffer], optional): Each item's legal\n"    //
    "    choices, one byte each, as `Model.text_choose` takes them, read when the run trains\n"   //
    "    choices; the call appends the mask byte to each choosing text.\n"                        //
    "  choice_targets (Sequence[buffer] | tuple[buffer, buffer], optional): The float32 target\n" //
    "    of each choice, an entry as long as its item's choices and summing to one, as\n"         //
    "    `Model.text_choose` answers its probabilities; required with `choices`.\n"               //
    "  relations (Relations, optional): The contrastive targets over the step's items.\n"         //
    "  arena (Arena, optional): Where the pooled embeddings the loss scored land; None skips\n"   //
    "    the copy.\n"                                                                             //
    "  outputs_scalar (Scalar, optional): The scalar the embeddings land as.\n"                   //
    "  stats (TrainerStepStats, optional): Overwritten with this step's report.\n"                //
    "\n"                                                                                          //
    "Returns:\n"                                                                                  //
    "  ArenaBuffer | None: The (count, hidden) embeddings when an arena was given.";

static PyObject *Trainer_step(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    PyObject *modalities_object, *contents_object, *bags_object, *temperature_groups_object, *sides_object,
        *choices_object, *choice_targets_object, *relations_object, *arena_object, *outputs_scalar_object,
        *stats_object;
    uform_py_argument_t const arguments[] = {
        {"modalities", &modalities_object},
        {"contents", &contents_object},
        {"bags", &bags_object},
        {"temperature_groups", &temperature_groups_object},
        {"sides", &sides_object},
        {"choices", &choices_object},
        {"choice_targets", &choice_targets_object},
        {"relations", &relations_object},
        {"arena", &arena_object},
        {"outputs_scalar", &outputs_scalar_object},
        {"stats", &stats_object},
    };
    uform_py_receiver_t receiver;
    uform_scalar_t outputs_scalar = uform_scalar_f32_k;
    if (uform_py_arguments_parse("step", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        trainer_receiver(self, &receiver) != 0 ||
        uform_py_scalar_from_object(outputs_scalar_object, "outputs_scalar", &outputs_scalar) != 0)
        return NULL;
    if (stats_object && stats_object != Py_None && !Py_IS_TYPE(stats_object, &TrainerStepStatsType)) {
        PyErr_Format(PyExc_TypeError, "the 'stats' option must be a TrainerStepStats or None, got %S",
                     Py_TYPE(stats_object));
        return NULL;
    }
    if (relations_object && relations_object != Py_None && !Py_IS_TYPE(relations_object, &RelationsType.type)) {
        PyErr_Format(PyExc_TypeError, "the 'relations' option must be a Relations or None, got %S",
                     Py_TYPE(relations_object));
        return NULL;
    }

    step_call_t stack;
    step_call_t *const call = uform_py_call_open(&receiver, &step_verb, &stack);
    if (!call) return NULL;
    uform_relations_view_t relations = {{0}};
    int const has_relations = relations_object && relations_object != Py_None;
    if (uform_py_tape_from_object(contents_object, "contents", uform_py_element_text_k, &call->contents) != 0)
        return uform_py_call_discard(&receiver, &step_verb, call);
    size_t const count = call->contents.count;
    if (modalities_from_object(modalities_object, count, &call->modalities) != 0 ||
        sides_from_object(sides_object, count, &call->sides) != 0 ||
        choices_from_objects(choices_object, choice_targets_object, count, &call->choices, &call->choice_targets) !=
            0 ||
        uform_py_words_from_object(bags_object, "bags", count, &call->bags) != 0 ||
        uform_py_words_from_object(temperature_groups_object, "temperature_groups", count, &call->temperature_groups) !=
            0 ||
        (has_relations && (edges_from_object(uform_py_fields_value(relations_object, relations_attract_k), "attract",
                                             count, &call->attract, &relations.attract) != 0 ||
                           edges_from_object(uform_py_fields_value(relations_object, relations_repel_k), "repel", count,
                                             &call->repel, &relations.repel) != 0 ||
                           edges_from_object(uform_py_fields_value(relations_object, relations_ignore_k), "ignore",
                                             count, &call->ignore, &relations.ignore) != 0)) ||
        (arena_object && arena_object != Py_None && uform_py_arena_claim(arena_object, &call->call.arena) != 0))
        return uform_py_call_discard(&receiver, &step_verb, call);
    if (stats_object && stats_object != Py_None) call->stats_object = Py_NewRef(stats_object);
    call->arguments = (uform_trainer_step_t) {
        .trainer = ((Trainer *)receiver.target)->handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .count = count,
        .modalities = call->modalities,
        .contents = call->contents.view,
        .bags = call->bags.data,
        .temperature_groups = call->temperature_groups.data,
        .sides = call->sides,
        .choices = call->choices.view,
        .choice_targets = call->choice_targets.view,
        .relations = relations,
        .embeddings = call->call.arena ? &call->embeddings : NULL,
        .arena = call->call.arena ? ((Arena *)call->call.arena)->handle : NULL,
        .outputs_scalar = outputs_scalar,
        .stats = call->stats_object ? &call->stats : NULL,
    };
    return uform_py_call_run(&receiver, &step_verb, call);
}

/** A @c uform_trainer_save call, and the encoded path and handler it borrows until it is final. */
typedef struct save_call_t {
    uform_py_call_t call;
    uform_trainer_save_t arguments;
    PyObject *decoded;
    uform_py_callback_t progress;
} save_call_t;

static void save_submit(void *call) { uform_trainer_save(&((save_call_t *)call)->arguments); }

static void save_release(void *erased) {
    save_call_t *const call = (save_call_t *)erased;
    Py_CLEAR(call->decoded);
    Py_CLEAR(call->progress.callable);
    uform_py_call_release(&call->call);
}

static int save_traverse(void *erased, visitproc visit, void *arg) {
    Py_VISIT(((save_call_t *)erased)->progress.callable);
    return 0;
}

static uform_py_verb_t const save_verb = {sizeof(save_call_t), save_submit, uform_py_call_answer_none, save_release,
                                          save_traverse};

static char const doc_Trainer_save[] =                                                          //
    "save(path, /, *, progress=None, progress_steps=0) -> None\n"                               //
    "\n"                                                                                        //
    "Writes the weights and the step as a safetensors checkpoint, whole and renamed into\n"     //
    "place, which `Model.load_vram` serves and `Trainer.init_vram` resumes; only the process\n" //
    "of index zero writes.\n"                                                                   //
    "Calls `uform_trainer_save`.\n"                                                             //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  path (str | bytes | os.PathLike): Where the checkpoint lands.\n"                         //
    "  progress (callable, optional): Called with (completed, total) tensors written.\n"        //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject *Trainer_save(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    PyObject *path_object, *progress_object, *progress_steps_object;
    uform_py_argument_t const arguments[] = {
        {"path", &path_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
    };
    uform_py_receiver_t receiver;
    size_t progress_steps = 0;
    if (uform_py_arguments_parse("save", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        trainer_receiver(self, &receiver) != 0 ||
        uform_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0)
        return NULL;
    save_call_t stack;
    save_call_t *const call = uform_py_call_open(&receiver, &save_verb, &stack);
    if (!call) return NULL;
    uform_bytes_view_t path;
    if (uform_py_path_from_object(path_object, "path", &path, &call->decoded) != 0 ||
        uform_py_callback_from_object(progress_object, "progress", &call->call, &call->progress) != 0)
        return uform_py_call_discard(&receiver, &save_verb, call);
    call->arguments = (uform_trainer_save_t) {
        .trainer = ((Trainer *)receiver.target)->handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .path = path,
        .progress = uform_py_progress_registered(&call->progress, progress_steps),
    };
    return uform_py_call_run(&receiver, &save_verb, call);
}

#pragma endregion Steps

PyMethodDef uform_py_trainer_verbs[] = {
    {"step", (PyCFunction)Trainer_step, uform_py_method_flags_k, doc_Trainer_step},
    {"save", (PyCFunction)Trainer_save, uform_py_method_flags_k, doc_Trainer_save},
    {NULL, NULL, 0, NULL},
};

PyMethodDef uform_py_trainer_methods[] = {
    {"init_vram", (PyCFunction)Trainer_init_vram, uform_py_method_flags_k | METH_CLASS, doc_Trainer_init_vram},
    {"on", (PyCFunction)Trainer_on, METH_O,
     "on(executor) -> On[Trainer]\n\nA receiver whose verbs block on `executor`. Python-only."},
    {"deferred", (PyCFunction)Trainer_deferred, METH_O,
     "deferred(executor) -> DeferredOn[Trainer]\n\nA receiver whose verbs answer a `Deferred`. Python-only."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject TrainerType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.Trainer",
    .tp_doc = doc_Trainer,
    .tp_basicsize = sizeof(Trainer),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Trainer_tp_new,
    .tp_dealloc = (destructor)Trainer_dealloc,
};
