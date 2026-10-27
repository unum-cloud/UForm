/**
 *  @file include/uform/train/base.cuh
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief The Base pre-training library: one @ref trainer_config_t run description and one
 *      @ref base_replica_t model copy per GPU, composed by @c trainer_t in
 *      @c include/uform/train/trainer.hpp .
 *
 *  Declared here, defined in @c src/train/base.cu , and consumed by the composition in @c src/train/trainer.cu ,
 *  which the training binary ( @c tools/base.cu ), the conformance test ( @c test/base.cu ) and the C verbs
 *  ( @c src/train.cpp ) all drive.
 *
 *  @section base_mixture Modality mixtures, not per-modality caps
 *
 *  Every batch is a modality mixture: any number of token / image / video / audio rows, constrained only by the
 *  aggregate row and segment budgets @ref multimodal_batch_shape_t carries — the same envelope the loaders fill
 *  and @c multimodal_batch_mixer_t shuffles in real training. A modality participates iff its nested shape slice
 *  is live (a nonzero @c patch_side / @c chunk_samples ), so presence is read off the shape everywhere. The
 *  synthetic corpus draws a keyed per-step partition of the row budget across the live modalities and orders the
 *  rows tokens → image → video → audio (the @ref modality_t order), each row modality-uniform with one segment,
 *  so projections tile contiguous row runs.
 *
 *  @section base_regions The step's regions
 *
 *  Each step runs on one per-replica stream: the @b eager preamble fills the batch, snapshots every live head's
 *  clean reconstruction target, and corrupts the batch in place — writing the capture-stable addresses the model
 *  graphs re-read at replay. The @b captured forward region runs the tower and scores every live head into the
 *  one shared @c d(normalized) buffer; the eager contrastive collective, when the objective has one, writes the
 *  pooled vertices' forces; the @b captured backward region folds both through the tower; the collective
 *  @c optimizer.step stays eager.
 */
#pragma once

#include <array> // `std::array` — the summand pairs

#include "uform/utilities/array.hpp"              // `per_gpu_array`
#include "uform/utilities/safetensors_format.hpp" // `checkpoint_to_load_t`
#include "uform/train/contrastive_segments.cuh"   // `max_matryoshka_prefixes_k`, `contrastive_relations_t`
#include "uform/train/encoder_tower.hpp"          // `encoder_tower_bf16_t`
#include "uform/utilities/vram_nvidia_device.hpp" // `gpu_arena_t`
#include "uform/train/introspect.cuh"             // `telemetry_losses_k` and the other ledger channels
#include "uform/encoder/multimodal_batch.cuh" // `multimodal_batch_shape_t`, `multimodal_batch_on_gpu_t`, `modality_t`
#include "uform/train/multimodal_masking.hpp" // `masking_recipe_t`, the masked heads
#include "uform/utilities/vram_nvidia_schedule.cuh" // `gpu_stream_t`, replay graphs

namespace unum::uform::train {
using namespace unum::uform::utilities;
using encoder::multimodal_batch_on_gpu_t;
using encoder::multimodal_batch_shape_t;

/** Which losses a step relaxes; the objective decides at @c make which heads and scratch exist. */
enum class objective_t : u32_t {

    /** Masked reconstruction alone. */
    masked_k = 0,

    /** The pairwise contrastive loss alone. */
    contrastive_k = 1,

    /** Both, joined at the final-norm states. */
    joint_k = 2,
};

/** How the contrastive loss scores the pooled vertices. */
struct contrastive_config_t {

    /** The Matryoshka prefix widths, strictly ascending multiples of 8; the first @ref matryoshka_count are read. */
    u32_t matryoshka_dimensions[max_matryoshka_prefixes_k] = {};

    /** Prefixes in @ref matryoshka_dimensions ; zero scores the full width alone. */
    u32_t matryoshka_count = 0;

    /** Learnable (inverse temperature, bias) entries a step's items select by group. */
    u32_t temperature_groups = 1;

    /** The starting inverse temperature of every group, SigLIP's for the default sigmoid form. */
    f32_t inverse_temperature = 10.0f;

    /** The starting logit bias of every group, SigLIP's for the default sigmoid form. */
    f32_t bias = -10.0f;

    /** The weight of every pair no edge names; zero repels named pairs alone. */
    f32_t ambient_repulsion_weight = 1.0f;

    /** Edges one GPU's step names per edge set at most; zero derives
     *  @ref relation_edges_capacity . */
    u32_t max_edges_per_gpu = 0;

    /** The pair loss over each anchor row. */
    contrastive_pair_loss_form_t pair_loss = contrastive_pair_loss_form_t::sigmoid_k;
};

/** How the late-interaction loss scores the per-token vectors of query and document segments. */
struct late_interaction_config_t {

    /** The loss weight λ; zero trains no token head and allocates nothing for it. A positive weight
     *  needs the batch's rows set explicitly, since the in-batch MaxSim costs rows² times the query
     *  span times the columns, a bound free memory cannot derive. */
    f32_t weight = 0.0f;

    /** The token vectors' width, a multiple of 8. */
    u32_t width = 128;

    /** The starting inverse temperature of every group the contrastive loss's items select. */
    f32_t inverse_temperature = 1.0f;

    /** The starting logit bias of every group. */
    f32_t bias = 0.0f;

    /** The pair loss over each query row. */
    contrastive_pair_loss_form_t pair_loss = contrastive_pair_loss_form_t::ranked_k;

    /** The per-query token cap: the tokens one query segment contributes at most, its first past
     *  the prefix, as the reranker shape bounds a query. */
    u32_t max_query_span = 64;

    /** Query token rows one GPU's step holds at most; zero derives @ref max_query_span tokens of
     *  every segment, within the batch's tokens per GPU. */
    u32_t max_query_tokens_per_gpu = 0;

    /** Document segments one GPU's step holds at most; zero derives @c max_segments_per_gpu . */
    u32_t max_documents_per_gpu = 0;

    /** Document token rows one GPU's step holds at most; zero derives the batch's tokens per
     *  GPU. */
    u32_t max_document_tokens_per_gpu = 0;
};

/** One tier: width and depth, each step of the ladder doubling exactly one, the row widths of
 *  text-only and media runs, and the peak learning rate. */
struct trainer_tier_t {
    char const *name = "";
    u32_t hidden = 0, depth = 0, text_columns = 0, media_columns = 0;
    f32_t learning_rate = 0.0f;
};

/** Learning rates follow the 2026-10-21 width sweep at depth 12, deeper tiers' probed at launch. */
inline constexpr trainer_tier_t trainer_tiers_k[] = {
    {"tiny", 512u, 12u, 4096u, 1024u, 5.0e-4f},    //
    {"small", 1024u, 12u, 4096u, 1024u, 2.0e-4f},  //
    {"medium", 1024u, 24u, 4096u, 1024u, 2.0e-4f}, //
    {"large", 2048u, 24u, 4096u, 1024u, 1.0e-4f},  //
    {"xl", 2048u, 48u, 4096u, 1024u, 1.0e-4f},
};

/** One run's whole description, parsed once and threaded by const reference. The modality mixture is the
 *  @ref batch envelope itself: a media modality trains iff its nested slice is live, and the byte
 *  decoder is present whenever the objective reconstructs or the run chooses. */
struct trainer_config_t {

    /** The batch envelope — the aggregate row / segment / column budgets plus each modality's own
     *  shape slice; a zero slice keeps that modality's tensors, tower entry, and head out of the
     *  composition. Zero rows derive the rows per GPU from a measuring pass at the configured
     *  columns, and given rows are honoured as they are; either way, zero segments follow the rows
     *  through @ref batch_capacity . */
    multimodal_batch_shape_t batch {0, 0, 0};

    /** The tier the shape, row widths and learning rate come from. */
    trainer_tier_t tier {};

    /** Transformer blocks stacked; set by the tier. */
    u32_t depth = 0;

    /** Embedding width, a multiple of the block's head dimension; set by the tier. */
    u32_t hidden = 0;

    /** The run's length in optimizer steps, counted across resumes; the cosine spans all of
     *  them. */
    u32_t steps = 100;

    /** The peak Adafactor learning rate the warmup ramps to and the cosine decays from. */
    f32_t learning_rate = 5.0e-4f;

    /** The cosine-decay floor the learning rate settles at over the run's tail. */
    f32_t min_learning_rate = 1.0e-5f;

    /** Linear warmup steps from zero to @ref learning_rate before the cosine decay begins; zero
     *  derives a tenth of @ref steps , at most 400. */
    u32_t warmup_steps = 0;

    /** Adafactor's first-moment decay of the clipped update; zero keeps none. */
    f32_t adafactor_beta1 = 0.9f;

    /** Which losses a step relaxes. */
    objective_t objective = objective_t::masked_k;

    /** The masked objective's corruption; its seed is derived from @ref seed at @c make . */
    masking_recipe_t masking {};

    /** Fractions of image patches, video tubes and audio frames masked. */
    f32_t image_mask_ratio = 0.75f, video_mask_ratio = 0.90f, audio_mask_ratio = 0.75f;

    /** The weights λ of the text, image, video and audio reconstruction heads. */
    f32_t text_weight = 1.0f, image_weight = 1.0f, video_weight = 1.0f, audio_weight = 1.0f;

    /** The contrastive loss's scoring, read unless the objective is masked alone. */
    contrastive_config_t contrastive {};

    /** The late-interaction loss over the token head, beside the pooled one when its weight is
     *  positive. */
    late_interaction_config_t late_interaction {};

    /** The calibrated choices over the byte decoder, beside every other term when its weight is
     *  positive. */
    choices_config_t choices {};

    /** The tensor-condition-regularization distortion target @c ε² . Not consumed anywhere yet —
     *  carried from the validated draft envelope until the regularizer lands. */
    f32_t tcr_epsilon_squared = 0.5f;

    /** Abort ceiling on a single step's gradient-norm spike; zero disables. Not consumed anywhere yet —
     *  carried from the validated draft envelope until the health gates land. */
    f32_t grad_spike_threshold = 0.0f;

    /** The tensor-condition-regularization loss weight; zero disables. Not consumed anywhere yet. */
    f32_t tcr_lambda = 0.0f;

    /** The Lipschitz-penalty loss weight; zero disables. Not consumed anywhere yet. */
    f32_t lipschitz_lambda = 0.0f;

    /** The run's base seed; the masking recipe and every synthetic draw key derive from it. */
    u64_t seed = 42;

    /** The run's model-region mode — @ref replayed_k in production, @ref eager_k under @c --eager ; the
     *  parity test drives both over identical seeds. */
    capture_t capture = replayed_k;

    /** Checkpoint path; empty writes none. Borrows the argv storage it was parsed from. */
    sz::string_view_t checkpoint {};

    /** Steps between checkpoints, beside the one at the run's end; zero writes only that one. */
    u32_t checkpoint_every = 0;

    /** Steps between telemetry lines. */
    u32_t report_every = 10;

    /** Steps between held-out retrieval evaluations; zero runs none. */
    u32_t eval_every = 0;

    /** The mixture JSON of the held-out batches; empty evaluates on the training source's next
     *  batch. Borrows the storage it was parsed from. */
    sz::string_view_t eval_mixture {};

    /** Warm-start path; empty starts from the keyed seed. Every replica loads the same file and the run
     *  continues from the file's saved step. Borrows the storage it was parsed from. */
    sz::string_view_t resume {};

    /** The ledger channels this run arms — losses and rounds by default; @c --telemetry adds
     *  more (e.g. @c signals for the optimizer's gradient-boundary probes). The entry point may
     *  also arm channels from its launcher contract; the library itself never reads environment. */
    u32_t telemetry_channels = telemetry_losses_k | telemetry_rounds_k;

    /** The mixture JSON the run loads its data from; empty trains on the synthetic draw. Borrows
     *  the storage it was parsed from. */
    sz::string_view_t mixture {};

    /** The NVDEC engines a GPU carries, which no query reports; a mixture with video needs it. */
    u32_t nvdec_engines = 0;

    /** How many steps' batches one data fill packs per GPU from one shard read, so the read is paid
     *  once per that many steps. */
    std::size_t batches_per_fill = 16;

    /** The device bytes a data source's reservoir takes outside the arena at @c rows rows,
     *  @c reservoir_fixed_bytes+rows·reservoir_row_bytes , charged by the source from its
     *  shards. */
    std::size_t reservoir_fixed_bytes = 0, reservoir_row_bytes = 0;

    /** How many segments a row of the mixture's densest entry holds, charged by the source from
     *  its footers like the reservoir; @ref batch_capacity multiplies the rows by it. */
    f64_t segments_per_row = 1.0;

    /**
     *  @brief Fold the @c --flag value pairs into a config, each flag adjusting one knob.
     *
     *  @c --mixture takes a @c .json path, or the synthetic @c tokens|omni preset, where omni fills
     *  the media slices of @ref batch with 8-pixel RGB image patches, 2-frame video tubes and
     *  256-sample audio frames.
     *  @retval status_t::invalid_input_k An unknown flag, a flag without its value, or a value it
     *      does not take.
     */
    [[nodiscard]] static expected<trainer_config_t> parse(int argc, char **argv) noexcept;

    /** Step @p optimizer_step 's scheduled learning rate: the linear warmup to @ref learning_rate over
     *  @ref warmup_steps , then the cosine decay to @ref min_learning_rate over the rest of
     *  @ref steps . The ordinal is absolute, so a resumed run continues the same curve. Host math,
     *  bitwise-stable across runs — replicas step in lockstep on it. */
    [[nodiscard]] f32_t learning_rate_for(optimizer_step_t optimizer_step) const noexcept;

    /** Whether the objective reconstructs masked inputs. */
    [[nodiscard]] bool reconstructs() const noexcept { return objective != objective_t::contrastive_k; }

    /** Whether the objective contrasts pooled vertices. */
    [[nodiscard]] bool contrasts() const noexcept { return objective != objective_t::masked_k; }

    /** Whether the run trains the token head through the late-interaction loss. */
    [[nodiscard]] bool late_interacts() const noexcept { return late_interaction.weight > 0.0f; }

    /** Whether the run trains calibrated choices through the byte decoder. */
    [[nodiscard]] bool chooses() const noexcept { return choices.weight > 0.0f; }
};

/**
 *  @brief One GPU's model replica — the tower, its input batch, every live masked head with its clean
 *      reconstruction target and running summands, and the pooled vertices the contrastive loss scores.
 *
 *  "Replica" is literal: every GPU holds bitwise-identical weights, indexed by the same @c member_index the
 *  @ref multimem_domain_t and @ref distributed_adafactor_t take. A head or the contrastive scratch exists iff
 *  the objective and the batch slice call for it, so presence is read off the allocation. Movable: every borrow
 *  points outward at the composition's address-stable session tables, never into this struct.
 */
struct base_replica_t {

    /** The encoder — per-modality entry, block stack, final RMSNorm, pool and head; borrows the placed session. */
    encoder_tower_bf16_t tower {};

    /** The input batch; after @ref prepare its tokens / media carry the distorted stream. */
    multimodal_batch_on_gpu_t batch {};

    /** The span-masking corruptor + untied byte decoder with fused cross-entropy, present when the
     *  run reconstructs or chooses. */
    masked_tokens_t tokens {};

    /** The image / video masked-patch reconstruction heads. */
    masked_patches_t image {}, video {};

    /** The audio masked-frame reconstruction head. */
    masked_frames_t audio {};

    /** The clean bytes kept as the token target while the batch holds the distorted stream. */
    tokens_tensor_arena_t clean_tokens {};

    /** The clean image / video patch targets @b [max_tokens, patch_scalars] staged before corruption. */
    bf16r2_tensor_arena_t clean_image_patches {}, clean_video_patches {};

    /** The clean audio snapshot @b [frames, chunk_samples] — the frame reconstruction target. */
    f32r2_tensor_arena_t clean_audio {};

    /** Per-position weights the token corruptor emits and the token cross-entropy reads. */
    u32r1_tensor_arena_t token_loss_weights {};

    /** @c {λ·Σ, count} Float32 pairs per head — the running loss readouts telemetry divides, summed
     *  over the steps since the last report: @ref prepare zeroes them on every step after a
     *  multiple of @c report_every . */
    f32r1_tensor_arena_t tokens_summands {}, image_summands {}, video_summands {}, audio_summands {};

    /** The choices term's @c {λ·Σ, count} pair, present when the run chooses, reset with the
     *  heads'. */
    f32r1_tensor_arena_t choices_summands {};

    /** The one shared @c d(normalized) @b [max_tokens, hidden] every live head accumulates into (the
     *  token head overwrites first, media heads and the choices add) and the tower's backward
     *  consumes. */
    bf16r2_tensor_arena_t normalized_grad {};

    /** The pooled vertices @b [max_segments, hidden] the contrastive loss scores, and their forces. */
    bf16r2_tensor_arena_t vertices {}, vertices_grad {};

    /** The relation graph over this replica's vertex rows, restaged every step. */
    contrastive_relations_t relations {};

    /** The @b [temperature_groups] inverse temperatures and biases the vertices' groups select. */
    f32r1_tensor_arena_t inverse_temperatures {}, biases {};

    /** The @b [max_segments] temperature group of every vertex row. */
    nk::tensor<temperature_group_t, gpu_arena_alloc<temperature_group_t>, 1> temperature_groups {};

    /** The @b [vertices_per_mesh] live length of every vertex of the mesh by global id, zero for PAD. */
    u32r1_tensor_arena_t to_vertex_lengths {};

    /** The @b [max_tokens,width] gradient of the tower's token vectors the late-interaction loss
     *  writes. */
    bf16r2_tensor_arena_t token_vectors_grad {};

    /** The @b [temperature_groups] late-interaction inverse temperatures and biases, selected by
     *  the same vertex groups as the pooled ones. */
    f32r1_tensor_arena_t token_inverse_temperatures {}, token_biases {};

    /**
     *  @brief This replica's parameter registrations: one named @ref differentiable_weights_t
     *      @c {name, weights, weights_grad} entry per matrix — the views non-owning, the checkpoint name
     *      owned — in one canonical order (tower first, then each live head).
     *
     *  The scalars live once inside this GPU's tensors; the optimizer's @c make copies these views into its
     *  own per-replica state, so the concatenated table is a construction-time transient.
     */
    st::basic_vector<differentiable_weights_t> weights {};

    /** Build the replica over the world-owned @p arena , publishing its parameter matrices into
     *  @ref weights in the same order on every replica. @p vertices_per_mesh sizes the contrastive
     *  to-vertex column, and every component calls the kernels of @p numerics . Constructs no
     *  stream or arena, and seeds nothing. */
    [[nodiscard]] static expected<base_replica_t> make(gpu_arena_t &arena, trainer_config_t const &config,
                                                       std::size_t vertices_per_mesh,
                                                       gpu_numerics_t const &numerics) noexcept;

    /** Draw the keyed synthetic mixture into @ref batch — row partition and contents, pure functions of
     *  @c config.seed , @p data_epoch , and @p gpu_index_in_colocated_domain , the world coordinate, so every
     *  world GPU draws distinct data. Pure enqueue onto @p stream ; never captured. */
    [[nodiscard]] expected_t draw(trainer_config_t const &config, u32_t gpu_index_in_colocated_domain,
                                  data_epoch_t data_epoch, gpu_stream_t &stream) noexcept;

    /** The eager preamble over a filled @ref batch : snapshot every live head's clean target, and corrupt each
     *  modality in place into the tower's distorted input. A no-op without masked heads. */
    [[nodiscard]] expected_t prepare(trainer_config_t const &config, data_epoch_t data_epoch,
                                     gpu_stream_t &stream) noexcept;

    /** The captured forward region: tower forward, pooling into @ref vertices when they exist, then
     *  every live head scores its clean target into @ref normalized_grad , and the choices term of
     *  @p config adds its force there. Pure enqueue, shape-stable. */
    [[nodiscard]] expected_t forward_region(trainer_config_t const &config, gpu_stream_t &stream) noexcept;

    /** The captured backward region: the tower's backward from the vertex forces, the masked forces
     *  and the token-vector forces, joined at the final-norm states. Pure enqueue, shape-stable. */
    [[nodiscard]] expected_t backward_region(gpu_stream_t &stream) noexcept;

    /** The contrastive recipe over this replica's device temperatures and vertex groups. */
    [[nodiscard]] contrastive_recipe_t recipe(trainer_config_t const &config) const noexcept;

    /** The late-interaction recipe over this replica's token temperatures and the same vertex
     *  groups. */
    [[nodiscard]] contrastive_recipe_t token_recipe(trainer_config_t const &config) const noexcept;

  private:

    /** The four summand pairs in @ref modality_t order, then the choices', absent ones empty. */
    [[nodiscard]] std::array<f32r1_tensor_arena_t *, 5> summand_pairs_() noexcept {
        return {&tokens_summands, &image_summands, &video_summands, &audio_summands, &choices_summands};
    }
};

/** @p shape at @p rows rows, with as many segments as @p segments_per_row fill them, at least one
 *  per row and at most one per cell, and its bags keeping their share of the segments. The only
 *  writer of the row and segment budgets. */
[[nodiscard]] multimodal_batch_shape_t batch_capacity(multimodal_batch_shape_t shape, std::size_t rows,
                                                      f64_t segments_per_row) noexcept;

/** Re-seed every matrix of @p weights to the identical deterministic init: full matrices from a
 *  keyed hash fill of standard deviation @c 1/√fan_in , the blocks' output projections further by
 *  @c 1/√(2·depth) , and vector scales exactly one. Identical on every replica. */
[[nodiscard]] expected_t seed_weights(st::basic_vector<differentiable_weights_t> const &weights,
                                      gpu_stream_t &stream) noexcept;

/**
 *  @brief Upload every matrix of @p weights from @p checkpoint by name, all or nothing, on @p stream ; the caller
 *      drains before the checkpoint is released.
 *
 *  A @c masked_tokens decoder or a @c token_head.* matrix the checkpoint misses is re-seeded as
 *  @ref seed_weights seeds it, so a run adds the head its base never trained.
 *  @retval status_t::invalid_input_k A matrix the checkpoint misses.
 *  @retval status_t::shape_unsupported_k A matrix of another shape.
 */
[[nodiscard]] expected_t load_weights(checkpoint_to_load_t const &checkpoint,
                                      st::basic_vector<differentiable_weights_t> const &weights,
                                      gpu_stream_t &stream) noexcept;

/** Snapshot @p weights into @p checkpoint as @c model.* , with @p steps and the geometry a reader
 *  rebuilds the tower from. */
[[nodiscard]] expected_t save_weights(checkpoint_to_save_t &checkpoint, trainer_config_t const &config,
                                      st::basic_vector<differentiable_weights_t> const &weights, u32_t steps,
                                      gpu_stream_t &stream) noexcept;

/** Print one line to standard output, newline-terminated. */
void report_line(sz::string_view_t line) noexcept;

} // namespace unum::uform::train
