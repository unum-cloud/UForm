/**
 *  @file c/uform/train.h
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Training steps of the UForm C interface: multimodal batches in, masked reconstruction and pairwise
 *      contrastive losses joined into one optimizer step, checkpoints out.
 *
 *  A @c uform_trainer_t owns everything a run keeps between steps on its executor's devices: one replica of
 *  the tower per device, its gradients, the Adafactor moments, the reconstruction heads, the learnable
 *  temperatures, and the collectives joining the replicas across the mesh. Training is GPU-only, so the one
 *  constructor is @c uform_trainer_init_vram, and one @c uform_trainer_step is one optimizer step: forward,
 *  losses, backward, the world gradient reduction, and the update.
 *
 *  @section train_objectives Objectives
 *
 *  The two losses the engine implements join additively at the final-norm states, so a joint step carries
 *  @c L = L_contrastive + Σ λ·L_masked exactly:
 *
 *      Objective     Scores                                    Against
 *      masked        byte cross-entropy for text, normalized   the clean copy of the batch the step
 *                    pixel MSE for image and video patches,    corrupted: geometric byte spans, masked
 *                    normalized sample MSE for audio frames    patches, tubes, and frames
 *      contrastive   the sigmoid pairwise loss on the cosine   the relation graph: attract, repel and
 *                    of every Matryoshka prefix of the         ignore edges between items, and ambient
 *                    pooled embeddings                         repulsion for every unnamed pair
 *
 *  The contrastive loss is a sum over independent pairs with no softmax normalization, so it spans the whole
 *  mesh: every item of the step repels every unnamed item of every process, while each device moves only its
 *  own embeddings and their forces.
 *
 *  @section train_batches Batches
 *
 *  A step takes this process's share of the global batch as items: one encoded file or UTF-8 text each,
 *  tagged with its modality, packed into fixed-width rows on the executor's devices and shuffled across the
 *  mesh before the forward. Items sharing a bag travel together. Relations index the step's own items; the
 *  library rebases them onto the global axis after the shuffle.
 *
 *  @section train_paths Paths
 *
 *  A path arrives as a @c uform_bytes_view_t of the operating system's own bytes, not necessarily
 *  UTF-8 and never terminated; an embedded NUL, which no path can hold, is refused with
 *  @c uform_status_invalid_argument_k. A name, like the mesh's @c job, is UTF-8.
 *
 *  @code{.c}
 *  uform_mesh_t mesh = NULL;
 *  uform_mesh_bootstrap(&(uform_mesh_bootstrap_t){.mesh = &mesh, .executor = executor, .outcome = &outcome,
 *      .options = {.host_domain_index = node, .host_domain_count = nodes, .directory = {exchange, exchange_length},
 *                  .job = {job, job_length}}});
 *
 *  uform_trainer_t trainer = NULL;
 *  uform_trainer_init_vram(&(uform_trainer_init_vram_t){
 *      .trainer = &trainer, .executor = executor, .outcome = &outcome, .mesh = mesh, .depth = 12, .hidden = 512,
 *      .batch = {.max_rows_per_device = 64, .max_segments_per_device = 256, .max_columns_per_row = 1024,
 *                .max_bags_per_device = 128, .image = {.patch_side = 16}},
 *      .objective = uform_objective_joint_k, .schedule = {.steps = 100000}, .seed = 42});
 *
 *  // A caption and its image per bag, each attracting the other.
 *  size_t const attract_offsets[] = {0, 1, 2, 3, 4};
 *  uint32_t const attract_ids[] = {1, 0, 3, 2};
 *  uform_trainer_step_stats_t stats = {0};
 *  uform_trainer_step(&(uform_trainer_step_t){
 *      .trainer = trainer, .executor = executor, .outcome = &outcome, .count = 4,
 *      .modalities = modalities, .contents = {bytes, offsets, uform_scalar_u8_k}, .bags = (uint32_t[]){0, 0, 1, 1},
 *      .relations = {.attract = {attract_offsets, attract_ids, NULL}}, .stats = &stats});
 *
 *  uform_trainer_save(&(uform_trainer_save_t){
 *      .trainer = trainer, .executor = executor, .outcome = &outcome, .path = {path, path_length}});
 *  uform_trainer_free(trainer);
 *  uform_mesh_free(mesh);
 *  @endcode
 */
#ifndef UNUM_UFORM_TRAIN_H
#define UNUM_UFORM_TRAIN_H

#include "uform/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** What the bytes of one training item hold, in the order the engine packs a batch. */
typedef enum uform_modality_t {

    /** UTF-8 text, a byte per token. */
    uform_modality_text_k = 0,

    /** A JPEG, PNG or WebP file. */
    uform_modality_image_k = 1,

    /** An MP4 or WebM file. */
    uform_modality_video_k = 2,

    /** A WAV, FLAC, MP3 or Opus file. */
    uform_modality_audio_k = 3,
} uform_modality_t;

/** The canonical spelling of a modality, as @c uform_modality_text_k spells "text". */
UFORM_EXPORT char const* uform_modality_name(uform_modality_t modality);

/** Reads a modality back; @c uform_modality_text_k when unrecognized. */
UFORM_EXPORT uform_modality_t uform_modality_from_name(uform_bytes_view_t name);

/** The image slice of a batch envelope; a zero @c patch_side leaves images out of the run. */
typedef struct uform_image_shape_t {

    /** The square patch side images tile into, in pixels. */
    size_t patch_side;

    /** Channels every pixel carries; zero derives four, RGBA end to end. */
    size_t channels;
} uform_image_shape_t;

/** The video slice of a batch envelope; a zero @c patch_side leaves video out of the run. */
typedef struct uform_video_shape_t {

    /** The square patch side one tube tiles into, in pixels. */
    size_t patch_side;

    /** Frames one tube folds ahead of its patch; zero derives two. */
    size_t tube_frames;

    /** Source frames between sampled ones; zero derives one, every frame. */
    size_t frame_stride;

    /** Channels every pixel carries; zero derives three. */
    size_t channels;
} uform_video_shape_t;

/** The audio slice of a batch envelope; a zero @c chunk_samples leaves audio out of the run. */
typedef struct uform_audio_shape_t {

    /** Raw samples per frame, the fan-in of the learned frontend. */
    size_t chunk_samples;

    /** The mono rate every recording is resampled to; zero derives 16000. */
    size_t sample_rate;
} uform_audio_shape_t;

/** What one device's step carries at most, whatever the modality mix: capacity fixed at creation, with
 *  padding through the data. Text always takes part; a media modality takes part when its slice is live. */
typedef struct uform_batch_shape_t {

    /** Rows one device's step packs. */
    size_t max_rows_per_device;

    /** Items one device's step carries across all its rows, which is also its pooling axis. */
    size_t max_segments_per_device;

    /** Positions one row spans, the sequence length. */
    size_t max_columns_per_row;

    /** Bags one device's step carries; zero keeps every item its own routing unit. */
    size_t max_bags_per_device;

    /** Legal bytes one item offers to choose from at most; zero makes no choices. */
    size_t max_choices_per_item;

    /** The image geometry. */
    uform_image_shape_t image;

    /** The video geometry. */
    uform_video_shape_t video;

    /** The audio geometry. */
    uform_audio_shape_t audio;
} uform_batch_shape_t;

/** Which losses a step relaxes, one enumerator per combination the engine trains. */
typedef enum uform_objective_t {

    /** Masked reconstruction alone, as the Base pretrains. */
    uform_objective_masked_k = 0,

    /** The pairwise contrastive loss alone, as the Embedder trains. */
    uform_objective_contrastive_k = 1,

    /** Both, joined at the final-norm states. */
    uform_objective_joint_k = 2,
} uform_objective_t;

/** The canonical spelling of an objective, as @c uform_objective_joint_k spells "joint". */
UFORM_EXPORT char const* uform_objective_name(uform_objective_t objective);

/** Reads an objective back; @c uform_objective_masked_k when unrecognized. */
UFORM_EXPORT uform_objective_t uform_objective_from_name(uform_bytes_view_t name);

/** How the masked objective corrupts a batch and weighs its heads; every zero derives the default named. */
typedef struct uform_masking_t {

    /** Fraction of text positions covered by spans; 0.15. */
    float token_mask_probability;

    /** Mean geometric span length in bytes; 16. */
    float token_mean_span;

    /** Of covered positions, the share shown as MASK; 0.8. */
    float replace_fraction;

    /** Of covered positions, the share replaced by a random byte, the rest kept clean; 0.1. */
    float random_fraction;

    /** Fraction of image patches masked; 0.75. */
    float image_mask_ratio;

    /** Fraction of video tubes masked, tubed across time; 0.75. */
    float video_mask_ratio;

    /** Fraction of audio frames masked; 0.75. */
    float audio_mask_ratio;

    /** The weight λ of the text head; 1. */
    float text_weight;

    /** The weight λ of the image head; 1. */
    float image_weight;

    /** The weight λ of the video head; 1. */
    float video_weight;

    /** The weight λ of the audio head; 1. */
    float audio_weight;
} uform_masking_t;

/** Which pairs the contrastive loss pushes apart. */
typedef enum uform_negatives_t {

    /** Repel edges, and every pair no edge names, at @c ambient_repulsion_weight. */
    uform_negatives_ambient_k = 0,

    /** Repel edges alone. */
    uform_negatives_named_k = 1,
} uform_negatives_t;

/** The canonical spelling of a negatives policy, as @c uform_negatives_named_k spells "named". */
UFORM_EXPORT char const* uform_negatives_name(uform_negatives_t negatives);

/** Reads a negatives policy back; @c uform_negatives_ambient_k when unrecognized. */
UFORM_EXPORT uform_negatives_t uform_negatives_from_name(uform_bytes_view_t name);

/** How a contrastive term turns a row of scores into forces, mirroring the trainer's pair-loss
 *  forms; zero keeps the term's own form: sigmoid for pooled vertices, ranked for token vectors. */
typedef enum uform_pair_loss_form_t {

    /** The term's own form. */
    uform_pair_loss_form_derived_k = 0,

    /** The sigmoid cross-entropy of every cell, its attract and repel halves normalized apart. */
    uform_pair_loss_form_balanced_k = 1,

    /** The sigmoid cross-entropy of every positive's score over every negative's, where the bias
     *  cancels. */
    uform_pair_loss_form_ranked_k = 2,

    /** SigLIP's sigmoid cross-entropy of every cell, each repel weighed like an attract. */
    uform_pair_loss_form_sigmoid_k = 3,
} uform_pair_loss_form_t;

/** How the contrastive loss scores the pooled embeddings; every zero derives the default named. */
typedef struct uform_contrastive_t {

    /** The @b optional Matryoshka prefix widths, strictly ascending, each a multiple of 8, at most 8 of them,
     *  every one scored; @c NULL scores the full width alone. Copied, not borrowed. */
    size_t const* matryoshka_dimensions;

    /** Entries in @c matryoshka_dimensions. */
    size_t matryoshka_dimensions_count;

    /** Learnable (inverse temperature, bias) pairs, one per group a step's items select; one. */
    size_t temperature_groups;

    /** The starting inverse temperature of every group; 10 under the sigmoid form, SigLIP's,
     *  else 1. */
    float inverse_temperature;

    /** The starting logit bias of every group; −10 under the sigmoid form, SigLIP's, else zero. */
    float bias;

    /** Which pairs repel. */
    uform_negatives_t negatives;

    /** The weight of every pair no edge names, under @c uform_negatives_ambient_k; 1. */
    float ambient_repulsion_weight;

    /** Edges one device's step names per edge set at most; zero derives it from the segments and
     *  the largest bag. */
    size_t max_edges_per_device;

    /** The pair loss over each anchor's row; sigmoid. */
    uform_pair_loss_form_t pair_loss;
} uform_contrastive_t;

/** How the late-interaction loss scores the per-token vectors of query and document segments; every
 *  zero derives the default named. */
typedef struct uform_late_interaction_t {

    /** The loss weight; zero trains no token head. */
    float weight;

    /** The token vectors' width, a multiple of 8; 128. */
    size_t width;

    /** The starting inverse temperature of every contrastive group; 1. */
    float inverse_temperature;

    /** The starting logit bias of every contrastive group; zero is zero. */
    float bias;

    /** Tokens one query segment contributes at most, its first past the prefix; 64. */
    size_t max_query_span;

    /** Query token rows one device's step holds at most; @c max_query_span of every segment. */
    size_t max_query_tokens_per_device;

    /** Document segments one device's step holds at most; @c max_segments_per_device. */
    size_t max_documents_per_device;

    /** Document token rows one device's step holds at most; the batch's tokens per device. */
    size_t max_document_tokens_per_device;

    /** The pair loss over each query's row of hard MaxSim scores; ranked. */
    uform_pair_loss_form_t pair_loss;
} uform_late_interaction_t;

/** Which side of a late-interaction score an item takes, mirroring the trainer's sides: a query's
 *  tokens score a document's, and a symmetric pair plays both. */
typedef enum uform_late_interaction_side_t {

    /** The item takes no part in late interaction. */
    uform_late_interaction_side_none_k = 0,

    /** The item's tokens score the documents it attracts and repels. */
    uform_late_interaction_side_query_k = 1,

    /** The item's tokens are scored by the queries naming it. */
    uform_late_interaction_side_document_k = 2,

    /** The item plays both sides. */
    uform_late_interaction_side_query_and_document_k = 3,
} uform_late_interaction_side_t;

/** SearchJev's calibrated choices over the byte decoder, the masked head's untied byte logits read
 *  at a mask byte appended to a state and restricted to the item's legal bytes. Each choice item
 *  scores @c −Σt·log(p) + brier_weight·Σ(p−t)², with @c p that restricted softmax and @c t its
 *  targets. */
typedef struct uform_choices_t {

    /** The loss weight; zero trains no choices. */
    float weight;

    /** The Brier term's weight beside the soft cross-entropy; 0.5. */
    float brier_weight;
} uform_choices_t;

/** The learning-rate schedule: a linear warmup to the peak, then a cosine decay to the floor. */
typedef struct uform_schedule_t {

    /** The peak learning rate; zero derives 5e-4. */
    float learning_rate;

    /** The floor the cosine settles at; zero derives 1e-5. */
    float min_learning_rate;

    /** Steps from zero to the peak; zero derives a tenth of @c steps, at most 400. */
    size_t warmup_steps;

    /** Steps the run takes across resumes, which the cosine spans; zero is refused. */
    size_t steps;
} uform_schedule_t;

/** Where this process sits among the processes training one model, and how they meet. */
typedef struct uform_mesh_options_t {

    /** This process's index among @c host_domain_count. */
    size_t host_domain_index;

    /** Processes in the run, one per host; zero derives one, a run with no peers. */
    size_t host_domain_count;

    /** GPUs one NVLink multicast domain spans, a fabric property no local probe sees; zero derives the host. */
    size_t gpus_per_multimem_domain;

    /** The path of the directory every host sees, where processes exchange their facts; required
     *  with peers. */
    uform_bytes_view_t directory;

    /** The run's unique name, naming one directory inside @c directory; required with peers. */
    uform_bytes_view_t job;

    /** Seconds any wait for a peer lasts before the call fails; zero derives 120. */
    double timeout_seconds;
} uform_mesh_options_t;

/**
 *  @brief One process's place among the processes of a run: the topology every process agreed on at
 *      bootstrap, and the InfiniBand domain joining its NVLink domains.
 *
 *  The process-wide half of a run's transport, made once and borrowed by every trainer, so no consumer wires
 *  its own adapters. The NVLink multicast domain is not here: its size is the sum of what its consumers
 *  reserve, so each trainer makes its own over the mesh's devices. Bootstrap is a one-time rendezvous through
 *  files in @c directory; after it, processes talk only over NVLink and InfiniBand.
 */
typedef struct uform_mesh_handle_t* uform_mesh_t;

/** The arguments of @c uform_mesh_bootstrap. */
typedef struct uform_mesh_bootstrap_t {

    /** Output for the new mesh, written once every process has joined. */
    uform_mesh_t* mesh;

    /** A VRAM executor whose devices are this process's members of the mesh. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. Peers wait for each other either way. */
    uform_completion_t completion;

    /** Where this process sits and how the processes meet. */
    uform_mesh_options_t options;
} uform_mesh_bootstrap_t;

/** Joins this process to the run's mesh; every process of the run calls it with the same count, multicast
 *  extent, directory, job and GPU class, or the call reports @c uform_status_config_mismatch_k. */
UFORM_EXPORT void uform_mesh_bootstrap(uform_mesh_bootstrap_t* arguments);

/** How a mesh's NVLink domains reduce. */
typedef enum uform_multimem_tier_t {

    /** The NVLink switch reduces through multicast memory. */
    uform_multimem_tier_switch_reduced_k = 0,

    /** Every member reads its peers in a fixed order, where multicast is absent; bit-identical results. */
    uform_multimem_tier_peer_read_k = 1,
} uform_multimem_tier_t;

/** What a mesh reports about itself. */
typedef struct uform_mesh_description_t {

    /** GPUs one process drives. */
    size_t gpus_per_host_domain;

    /** GPUs one NVLink domain spans. */
    size_t gpus_per_multimem_domain;

    /** GPUs in the whole run. */
    size_t gpus_per_colocated_domain;

    /** This process's index among the run's processes. */
    size_t host_domain_index;

    /** The run-wide index of this process's first GPU, where its share of a sharded dataset begins. */
    size_t first_gpu_index;

    /** How its NVLink domains reduce. */
    uform_multimem_tier_t multimem_tier;

    /** The class every GPU of the run dispatches as: the executor's pin, or the devices' own. */
    uform_gpu_class_t gpu_class;
} uform_mesh_description_t;

/** The arguments of @c uform_mesh_describe. */
typedef struct uform_mesh_describe_t {

    /** The mesh to be described. */
    uform_mesh_t mesh;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Output, filled whole. */
    uform_mesh_description_t* description;
} uform_mesh_describe_t;

/** Reports a mesh's shape and this process's place in it. */
UFORM_EXPORT void uform_mesh_describe(uform_mesh_describe_t* arguments);

/** Leaves the mesh; @c NULL is safe, and it cannot fail, so takes no outcome. Every trainer borrowing it must
 *  be freed first. */
UFORM_EXPORT void uform_mesh_free(uform_mesh_t mesh);

/** The canonical spelling of a multimem tier, as @c uform_multimem_tier_peer_read_k spells "peer-read". */
UFORM_EXPORT char const* uform_multimem_tier_name(uform_multimem_tier_t tier);

/** Reads a multimem tier back; @c uform_multimem_tier_switch_reduced_k when unrecognized. */
UFORM_EXPORT uform_multimem_tier_t uform_multimem_tier_from_name(uform_bytes_view_t name);

/** A training run: the tower replicas, gradients, optimizer moments, heads, temperatures and the collectives
 *  sized by them, on the executor's devices. */
typedef struct uform_trainer_handle_t* uform_trainer_t;

/** The arguments of @c uform_trainer_init_vram. */
typedef struct uform_trainer_init_vram_t {

    /** Output for the new trainer, written once the call completes. */
    uform_trainer_t* trainer;

    /** A VRAM executor, every device of which holds one replica. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. Peers wait for each other either way. */
    uform_completion_t completion;

    /** The @b optional path of the checkpoint to continue from, at its saved step; empty starts
     *  from @c seed. A checkpoint of another geometry reports @c uform_status_config_mismatch_k. */
    uform_bytes_view_t resume;

    /** Transformer blocks stacked. */
    size_t depth;

    /** Embedding width, a multiple of 128. */
    size_t hidden;

    /** What one device's step carries at most. */
    uform_batch_shape_t batch;

    /** Which losses a step relaxes. */
    uform_objective_t objective;

    /** The masked objective's corruption and weights, read unless the objective is contrastive alone. */
    uform_masking_t masking;

    /** The contrastive loss's scoring, read unless the objective is masked alone. */
    uform_contrastive_t contrastive;

    /** The late-interaction loss beside the pooled one, read when its weight is positive. */
    uform_late_interaction_t late_interaction;

    /** The calibrated choices beside every other term, read when its weight is positive, which
     *  needs the batch's @c max_choices_per_item. */
    uform_choices_t choices;

    /** The learning-rate schedule. */
    uform_schedule_t schedule;

    /** The @b optional mesh, borrowed until the trainer is freed and made over this same executor; @c NULL
     *  runs this process alone. */
    uform_mesh_t mesh;

    /** The base of every keyed draw: initialization, corruption and the shuffle. */
    uint64_t seed;

    /** The @b optional handler, called with tensors placed on a resume. */
    uform_progress_t progress;
} uform_trainer_init_vram_t;

/** Builds one replica per device, the NVLink domain its consumers reserve over the mesh, and starts or
 *  resumes the run; every process of the run calls it with the same geometry, objective and seed. */
UFORM_EXPORT void uform_trainer_init_vram(uform_trainer_init_vram_t* arguments);

/** One edge set over a step's items, as compressed sparse rows. */
typedef struct uform_edges_view_t {

    /** The step's @c count + 1 non-decreasing offsets into @c ids, in host memory; @c NULL names no edges. */
    size_t const* offsets;

    /** Item @c i links to items @c ids[offsets[i]] up to @c ids[offsets[i+1]], indices into the step's items,
     *  ascending within each run. */
    uint32_t const* ids;

    /** The @b optional weight of each edge, paired with @c ids; @c NULL weighs every edge one. */
    float const* weights;
} uform_edges_view_t;

/** The contrastive targets of one step, resolved per pair in this order: a pair @c ignore names is skipped,
 *  one @c attract names pulls together, one @c repel names pushes apart, and every other pair follows the
 *  run's @c negatives. */
typedef struct uform_relations_view_t {

    /** Pairs pulled together. */
    uform_edges_view_t attract;

    /** Pairs pushed apart. */
    uform_edges_view_t repel;

    /** Pairs skipped, like known false negatives; @c weights is not read. */
    uform_edges_view_t ignore;
} uform_relations_view_t;

/** What a step reports, every loss its world mean. */
typedef struct uform_trainer_step_stats_t {

    /** The optimizer step the call applied, counted from one across resumes. */
    size_t step;

    /** The learning rate it applied. */
    float learning_rate;

    /** The total loss the step minimized. */
    float loss;

    /** The contrastive term; zero when the objective is masked alone. */
    float contrastive_loss;

    /** The text reconstruction term, before its weight; zero when absent. */
    float masked_text_loss;

    /** The image reconstruction term, before its weight; zero when absent. */
    float masked_image_loss;

    /** The video reconstruction term, before its weight; zero when absent. */
    float masked_video_loss;

    /** The audio reconstruction term, before its weight; zero when absent. */
    float masked_audio_loss;

    /** The late-interaction term, weighted; zero when absent. */
    float late_interaction_loss;

    /** The choices term over the items that made a choice, before its weight; zero when absent. */
    float choices_loss;

    /** Items cut at a row's width. */
    size_t truncated;

    /** Segments dropped whole because the batch had no room for them, since the last step that
     *  read its stats. */
    size_t dropped;
} uform_trainer_step_stats_t;

/** The arguments of @c uform_trainer_step. */
typedef struct uform_trainer_step_t {

    /** The run to advance. */
    uform_trainer_t trainer;

    /** The executor the trainer was created on. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Items in this process's share of the step; more than its devices' envelopes hold reports
     *  @c uform_status_shape_mismatch_k. */
    size_t count;

    /** The modality of each item, @c count of them in host memory; a modality the run left out reports
     *  @c uform_status_unsupported_k. */
    uform_modality_t const* modalities;

    /** The items' bytes, one entry each, of @c uform_scalar_u8_k: UTF-8 text or an encoded file,
     *  as @c modalities tags it. One tape mixes the modalities, so it is not the encode verbs'
     *  @c texts. */
    uform_tape_view_t contents;

    /** The @b optional bag of each item, @c count of them in host memory; items sharing a bag land on one
     *  device. @c NULL keeps every item its own bag. */
    uint32_t const* bags;

    /** The @b optional temperature group of each item, @c count of them in host memory, each below the run's
     *  @c temperature_groups; @c NULL puts every item in group zero. */
    uint32_t const* temperature_groups;

    /** The @b optional late-interaction side of each item, @c count of them in host memory;
     *  @c NULL makes every item @c uform_late_interaction_side_none_k. */
    uform_late_interaction_side_t const* sides;

    /** The @b optional legal choices of each item, one byte each, @c uform_scalar_u8_k, as
     *  @c uform_text_choose_t reads them; @c NULL offsets make no choices, and an item offering
     *  none makes none. Read when the run trains choices: a text item offering any is the state
     *  its answer follows, and the call appends the mask byte, cutting a state longer than a row
     *  at its head. */
    uform_tape_view_t choices;

    /** The target probability of each choice, @c uform_scalar_f32_k, in @c choices order, as
     *  @c uform_text_choose_t lands its probabilities: each item's entry as long as its entry of
     *  @c choices, summing to one. Read with @c choices. */
    uform_tape_view_t choice_targets;

    /** The contrastive targets, read unless the objective is masked alone; a contrastive step naming no
     *  attract edge reports @c uform_status_invalid_argument_k. */
    uform_relations_view_t relations;

    /** The @b optional output for the pooled embedding of every item, as the contrastive loss scored it
     *  before the update, inside @c arena; @c NULL skips the copy. */
    uform_rows_view_t* embeddings;

    /** Memory for @c embeddings, rewound by the call. */
    uform_arena_t arena;

    /** The scalar type @c embeddings lands as. */
    uform_scalar_t outputs_scalar;

    /** The @b optional report; @c NULL skips it, and with it the readback of the losses. */
    uform_trainer_step_stats_t* stats;
} uform_trainer_step_t;

/** Applies one optimizer step over this process's items and every peer's; every process of the run calls
 *  it once per step. */
UFORM_EXPORT void uform_trainer_step(uform_trainer_step_t* arguments);

/** The arguments of @c uform_trainer_save. */
typedef struct uform_trainer_save_t {

    /** The run to save. */
    uform_trainer_t trainer;

    /** The executor the trainer was created on. */
    uform_executor_t executor;

    /** Output for the outcome, final once its status leaves @c uform_status_deferred_k. */
    uform_outcome_t* outcome;

    /** Whether the call blocks or returns once queued; zero blocks. */
    uform_completion_t completion;

    /** Destination path of the safetensors checkpoint, written whole and renamed into place. */
    uform_bytes_view_t path;

    /** The @b optional handler, called with tensors written. */
    uform_progress_t progress;
} uform_trainer_save_t;

/** Writes the weights and the step, which @c uform_model_load_vram and its siblings serve and
 *  @c uform_trainer_init_vram resumes; only the process of index zero writes, the rest return at once. */
UFORM_EXPORT void uform_trainer_save(uform_trainer_save_t* arguments);

/** Frees a trainer and leaves the mesh; @c NULL is safe, and it cannot fail, so takes no outcome. Every call
 *  using it must be final. */
UFORM_EXPORT void uform_trainer_free(uform_trainer_t trainer);

#ifdef __cplusplus
}
#endif

#endif // UNUM_UFORM_TRAIN_H
