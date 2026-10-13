/**
 *  @brief The Base pre-training library: one @ref base_config_t run description, one @ref base_replica_t model
 *      copy per GPU, and one @ref base_trainer_t composition owning the per-GPU tables — declared here, defined
 *      in @c cuda_train/base.cu, and consumed by both the training binary ( @c cuda_train/base_main.cu ) and the
 *      conformance test ( @c cuda_test/base.cu ), so neither copies the bring-up.
 *  @file cuda_train/base.cuh
 *  @author Ash Vardanian
 *  @date July 6, 2026
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
 *  @section base_regions The two-region step
 *
 *  Each step is two regions on one per-replica stream: the @b eager preamble draws the keyed synthetic batch,
 *  snapshots every live head's clean reconstruction target, and corrupts the batch in place — writing the
 *  capture-stable addresses the model graph re-reads at replay — then the @b captured model region runs the
 *  tower forward, scores every live head into the one shared @c d(normalized) buffer, and folds it back through
 *  the tower's backward; the collective @c optimizer.step stays eager. The synthetic pass key coincides with the
 *  optimizer step ( @c data_epoch_t {optimizer_step.ordinal} ) inside @ref base_trainer_t::step.
 *
 *  @section base_ownership Sessions placed, trainer built in place, domains threaded by reference
 *
 *  The tower borrows its cuDNN session by pointer, so the sessions live in address-stable
 *  @ref per_gpu_array tables and the trainer is built where it lives ( @ref base_trainer_t::make fills a
 *  caller-placed reference) and never moved. The one @ref multimem_domain_t, the one @c colocated_domain_t, and
 *  the @ref distributed_adafactor_t are made in @ref base_trainer_t::run and threaded by reference — the trainer
 *  owns none of them.
 */
#pragma once

#include <span> // `std::span` — the replay-graph table decays to a view for `step`

#include "cuda_core/array.hpp"          // `array`, `per_gpu_array`
#include "cuda_core/gpu_allocators.hpp" // `gpu_arena_t`
#include "cuda_core/schedule.cuh"       // `gpu_stream_t`, `cudnn_session_t`, replay graphs
#include "cuda_core/mesh.hpp"           // `mesh_topology_t`
#include "cuda_core/introspect.cuh"     // `telemetry_ledger_t`, `telemetry_probe_t` — the run's one ledger
#include "cuda_core/multimem_collectives.cuh"
#include "cuda_core/multimodal_batch.hpp" // `multimodal_batch_shape_t`, `multimodal_batch_on_gpu_t`, `modality_t`
#include "cuda_core/multimodal_masking.hpp"
#include "cuda_core/encoder_tower.hpp"
#include "cuda_core/optimizer.hpp"

namespace unum::uform::train {

using namespace unum::uform::core; // the train tier builds directly on the core cell, like the loaders

/** @brief One run's whole description, parsed once and threaded by const reference. The modality mixture is the
 *      @ref batch envelope itself: a media modality trains iff its nested slice is live, and the token head is
 *      always present, so a media-less shape is the span-masked byte Base and a full shape is Base Omni. */
struct base_config_t {
    /** @brief The batch envelope — the aggregate row / segment / column budgets plus each modality's own shape
     *      slice; a zero slice keeps that modality's tensors, tower entry, and head out of the composition. */
    multimodal_batch_shape_t batch {4, 4, 256};
    /** @brief Transformer blocks stacked. */
    u32_t depth = 12;
    /** @brief Embedding width — a multiple of the block's head dimension. */
    u32_t hidden = 512;
    /** @brief Optimizer steps to run. */
    u32_t steps = 100;
    /** @brief The peak Adafactor learning rate the warmup ramps to and the cosine decays from. */
    f32_t learning_rate = 5.0e-3f;
    /** @brief The cosine-decay floor the learning rate settles at over the run's tail. */
    f32_t min_learning_rate = 1.0e-5f;
    /** @brief Linear warmup steps from zero to @ref learning_rate before the cosine decay begins. */
    u32_t warmup_steps = 400;
    /** @brief The tensor-condition-regularization distortion target @c ε² . Not consumed anywhere yet —
     *      carried from the validated draft envelope until the regularizer lands. */
    f32_t tcr_epsilon_squared = 0.5f;
    /** @brief Abort ceiling on a single step's gradient-norm spike; zero disables. Not consumed anywhere yet —
     *      carried from the validated draft envelope until the health gates land. */
    f32_t grad_spike_threshold = 0.0f;
    /** @brief The tensor-condition-regularization loss weight; zero disables. Not consumed anywhere yet. */
    f32_t tcr_lambda = 0.0f;
    /** @brief The Lipschitz-penalty loss weight; zero disables. Not consumed anywhere yet. */
    f32_t lipschitz_lambda = 0.0f;
    /** @brief The run's base seed; the masking recipe and every synthetic draw key derive from it. */
    u64_t seed = 42;
    /** @brief The run's model-region mode — @ref replayed_k in production, @ref eager_k under @c --eager ; the
     *      parity test drives both over identical seeds. */
    capture_t capture = replayed_k;
    /** @brief The cuDNN sessions' engine policy — the library's own heuristic pick by default,
     *      @ref algo_pick_t::benchmarked_k under @c --tune . */
    algo_pick_t algo_pick = algo_pick_t::heuristic_k;
    /** @brief Checkpoint path; empty writes none. Borrows the argv storage it was parsed from. */
    sz::string_view_t checkpoint {};
    /** @brief Warm-start path; empty starts from the keyed seed. Every replica loads the same file
     *      (they are bitwise-identical by construction) and the run continues from the file's saved
     *      step — the learning-rate schedule and the keyed data draws pick up where the save left off.
     *      Borrows the argv storage it was parsed from. */
    sz::string_view_t resume {};
    /** @brief The ledger channels this run arms — losses and rounds by default; @c --telemetry adds
     *      more (e.g. @c signals for the optimizer's gradient-boundary probes). The entry point may
     *      also arm channels from its launcher contract; the library itself never reads environment. */
    u32_t telemetry_channels = telemetry_losses_k | telemetry_rounds_k;

    /** @brief Fold the @c --flag value pairs into a config; unknown flags keep their default. The
     *      @c --mixture @c tokens|omni preset fills the media slices of @ref batch (omni: 8-pixel RGB image
     *      patches, 2-frame video tubes, 256-sample audio frames); every other flag adjusts one knob. */
    [[nodiscard]] static base_config_t parse(int argc, char **argv) noexcept;

    /** @brief Step @p optimizer_step 's scheduled learning rate: the linear warmup to @ref learning_rate over
     *      @ref warmup_steps , then the cosine decay to @ref min_learning_rate over the run's remainder. Host
     *      math, bitwise-stable across runs — replicas step in lockstep on it. */
    [[nodiscard]] f32_t learning_rate_for(optimizer_step_t optimizer_step) const noexcept;
};

#if defined(__CUDACC__)

/** @brief One GPU's model replica — the tower, its input batch, and every live masked head (present iff the
 *      config's batch slice for that modality is live) with its clean reconstruction target and running
 *      summands. "Replica" is literal: every GPU holds bitwise-identical weights, indexed by the same
 *      @c member_index the @ref multimem_domain_t and @ref distributed_adafactor_t take. Movable: every borrow
 *      points outward at the trainer's address-stable session tables, never into this struct. */
struct base_replica_t {
    /** @brief The encoder — per-modality entry, block stack, final RMSNorm; borrows the placed sessions. */
    encoder_tower_bf16_t tower {};
    /** @brief The input batch; after @ref prepare its tokens / media carry the distorted stream. */
    multimodal_batch_on_gpu_t batch {};

    /** @brief The span-masking corruptor + untied byte decoder with fused cross-entropy (always live). */
    masked_tokens_t tokens {};
    /** @brief The image / video masked-patch reconstruction heads. */
    masked_patches_t image {}, video {};
    /** @brief The audio masked-frame reconstruction head. */
    masked_frames_t audio {};

    /** @brief The clean bytes kept as the token target while the batch holds the distorted stream. */
    tokens_tensor_arena_t clean_tokens {};
    /** @brief The clean image / video patch targets @b [max_tokens, patch_scalars] staged before corruption. */
    bf16r2_tensor_arena_t clean_image_patches {}, clean_video_patches {};
    /** @brief The clean audio snapshot @b [frames, chunk_samples] — the frame reconstruction target. */
    f32r2_tensor_arena_t clean_audio {};

    /** @brief Per-position weights the token corruptor emits and the token cross-entropy reads. */
    u32r1_tensor_arena_t token_loss_weights {};
    /** @brief @c {λ·Σ, count} Float32 pairs per head — the running loss readouts telemetry divides. */
    f32r1_tensor_arena_t tokens_summands {}, image_summands {}, video_summands {}, audio_summands {};

    /** @brief The one shared @c d(normalized) @b [max_tokens, hidden] every live head accumulates into (the
     *      token head overwrites first, media heads add) and the tower's backward consumes. */
    bf16r2_tensor_arena_t normalized_grad {};

    /** @brief This replica's parameter registrations: one named @ref differentiable_weights_t
     *      @c {name, weights, weights_grad} entry per matrix — the views non-owning, the checkpoint name
     *      owned — in one canonical order (tower first, then each live head). The scalars live once inside
     *      this GPU's tensors; the optimizer's @c make copies these views into its own per-replica state,
     *      so the concatenated table is a construction-time transient. */
    array<differentiable_weights_t> weights {};

    /** @brief Build the replica over the world-owned @p arena and the placed @p dnn session (borrowed by pointer
     *      for the replica's lifetime), publishing its parameter matrices into @ref weights — the identical order
     *      on every replica. Constructs no stream, arena, or session, and seeds nothing: @ref base_trainer_t::make
     *      seeds all replicas together. */
    [[nodiscard]] static expected<base_replica_t> make(gpu_arena_t &arena, cudnn_session_t &dnn,
                                                       base_config_t const &config) noexcept;

    /** @brief The eager per-step preamble: draw the keyed synthetic mixture (row partition, contents, and
     *      corruption all pure functions of @c config.seed , @p data_epoch , and
     *      @p gpu_index_in_colocated_domain — the world coordinate, so every world GPU draws distinct
     *      data), snapshot every live head's clean target, and corrupt each modality in place into the
     *      tower's distorted input. Pure enqueue onto @p stream ; never captured. */
    [[nodiscard]] expected_t prepare(base_config_t const &config, u32_t gpu_index_in_colocated_domain,
                                     data_epoch_t data_epoch, gpu_stream_t &stream) noexcept;

    /** @brief The captured model region: tower forward → every live head scores its clean target,
     *      accumulating into @ref normalized_grad → tower backward. Pure enqueue, shape-stable, no per-step
     *      inputs — it re-reads the capture-stable addresses @ref prepare wrote. */
    [[nodiscard]] expected_t model_region(gpu_stream_t &stream) noexcept;
};

/** @brief The per-host composition the binary and the test share: the per-GPU streams / arenas / session tables
 *      and one @ref base_replica_t per GPU, each carrying its own parameter registrations. Built in place by
 *      @ref make (the towers borrow the placed sessions, so the trainer never moves) and driven one @ref step at
 *      a time. It owns no domain, colocated surface, or optimizer — those are made in @c main and threaded by
 *      reference, sized off the lead replica's @c weights list. */
struct base_trainer_t {
    /** @brief The run description every method reads — the single copy. */
    base_config_t config {};
    /** @brief This host_domain's first GPU's world coordinate — the synthetic draws key on
     *      @c first_gpu_index_in_colocated_domain+member_index , so every world GPU trains on distinct
     *      data; a single-host world keeps it zero and the draws are the local indexes unchanged. */
    u32_t first_gpu_index_in_colocated_domain = 0;
    /** @brief The device ordinals this host_domain drives; decays to the borrowing collectives. */
    array<i32_t> devices {};
    /** @brief One owning stream per GPU — the ordered chain a replica's whole step enqueues onto. */
    per_gpu_array<gpu_stream_t> streams {};
    /** @brief One owning arena per GPU — the allocation budget its replica draws from. */
    per_gpu_array<gpu_arena_t> arenas {};
    /** @brief One fused-attention session per GPU, placed address-stable so the towers may point into it. */
    per_gpu_array<cudnn_session_t> dnn {};
    /** @brief One model replica per GPU, each holding its own @c weights registration list. */
    per_gpu_array<base_replica_t> replicas {};
    /** @brief The run's one telemetry ledger — borrowed, like the domains; @ref make stores it and
     *      registers the probes below, so every consumer feeds the same root-owned instance. */
    telemetry_ledger_t *ledger = nullptr;
    /** @brief The trainer's registered probes: one loss row per head, one round row per member — inert
     *      zero handles whenever their channels are dark, so feeding them never branches. */
    struct probes_t {
        telemetry_probe_t token {}, image {}, video {}, audio {};
        per_gpu_array<telemetry_probe_t> rounds {};
    };
    probes_t probes {};

    /** @brief One step's readouts off the lead replica — best-effort: a readback slip degrades a field, never
     *      the run. The test's descent check and the binary's progress line read the same record. */
    struct telemetry_t {
        /** @brief The optimizer step this record belongs to. */
        optimizer_step_t optimizer_step {};
        /** @brief The token head's mean masked cross-entropy. */
        f32_t token_ce = 0.0f;
        /** @brief Each live media head's mean normalized reconstruction error (zero when absent). */
        f32_t image_mse = 0.0f, video_mse = 0.0f, audio_mse = 0.0f;
        /** @brief The domain's barrier round after this optimizer step — advanced off zero once bring-up succeeds. */
        concurrency_round_t concurrency_round {};

        /** @brief The composite masked loss — the token cross-entropy plus every live head's error. */
        [[nodiscard]] f32_t composite() const noexcept { return token_ce + image_mse + video_mse + audio_mse; }
        /** @brief Print one progress line for this step. */
        void report() const noexcept;
    };

    /** @brief The whole bring-up, filling the caller-placed @p trainer in place (a move would dangle the towers'
     *      session borrows): per GPU a stream, an arena, the cuDNN session, and the replica — each publishing its
     *      own registration list — then every replica seeded identically. Registers the trainer's loss and
     *      round probes on the borrowed @p ledger (inert rows when the channels are dark). Builds no domain,
     *      colocated surface, or optimizer. */
    [[nodiscard]] static expected_t make(base_trainer_t &trainer, base_config_t const &config,
                                         mesh_topology_t const &topology, telemetry_ledger_t &ledger) noexcept;

    /** @brief Capture each replica's @ref base_replica_t::model_region into one replay graph per GPU, appended
     *      to the caller-owned @p graphs — pass the filled table to @ref step for replayed runs, or an empty
     *      span for eager ones. */
    [[nodiscard]] expected_t capture(per_gpu_array<gpu_replay_graph_t> &graphs) noexcept;

    /** @brief Enqueue step @p optimizer_step on every replica: the eager @ref base_replica_t::prepare (whose
     *      synthetic pass key is @c data_epoch_t {optimizer_step.ordinal} ), the model region ( @p graphs replayed
     *      when non-empty, else inline), then the eager collective @c optimizer.step . */
    [[nodiscard]] expected_t step(optimizer_step_t optimizer_step, distributed_adafactor_t &optimizer,
                                  std::span<gpu_replay_graph_t> graphs) noexcept;

    /** @brief Re-seed every replica's weights to the identical deterministic init — @ref make runs it once, and
     *      the capture-vs-eager parity test reruns it between runs so both start from equals. */
    [[nodiscard]] expected_t reseed() noexcept;

    /** @brief Sync @b every local replica and read one @ref telemetry_t — each live head's @c {λ·Σ, count}
     *      summands accumulated across all members before the fold to means (8–16× the sample mass of a
     *      lead-only reading, the difference between a statistic and a coin flip at wide worlds), plus
     *      @p domain 's barrier round. Feeds the ledger's loss rows and enqueues one round probe per member. */
    [[nodiscard]] telemetry_t telemetry(multimem_domain_t &domain, optimizer_step_t optimizer_step) noexcept;

    /** @brief The whole training binary: bootstrap the mesh, @ref make the trainer, make the one domain /
     *      colocated surface / optimizer as locals, @ref capture unless @c --eager , run @c config.steps of
     *      @ref step with @ref telemetry_t::report on cadence, and optionally checkpoint the lead replica.
     *      Returns @c 0 on success, @c 1 on the first failure. */
    [[nodiscard]] static int run(base_config_t const &config) noexcept;
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::train
