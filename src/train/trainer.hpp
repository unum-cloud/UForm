/**
 *  @file src/train/trainer.hpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief A training run: @ref mesh_t , the transport every trainer of a process borrows, and
 *      @ref trainer_t , what one run owns, stepped from a @ref batch_source_t .
 *
 *  The mesh owns the bootstrapped topology and the InfiniBand domain joining the NVLink domains.
 *  The trainer owns the multimem domain, sized by the sum of its consumers'
 *  @c payload_scalars_needed() . Both are placed and never moved, since the colocated domain
 *  borrows the topology.
 *
 *  Every process builds its collectives and issues its barriers in one order, since rendezvous
 *  IDs and multimem flags count them. The trainer declares its members in that order, so
 *  @ref trainer_t::make reads top to bottom and teardown runs in reverse.
 */
#pragma once

#include <span> // `std::span` — devices and item columns as views

#include "core/infiniband_collectives.cuh" // `colocated_domain_t`, `infiniband_all_reduce`
#include "core/mesh.hpp"                   // `mesh_topology_t`
#include "core/introspect.cuh"             // `telemetry_ledger_t`, `telemetry_probe_t`
#include "core/optimizer.hpp"              // `distributed_adafactor_t`
#include "train/base.cuh"                  // `trainer_config_t`, `base_replica_t`

namespace unum::uform::train {

/** Where this process sits among the processes of a run, and how they meet; every field explicit. */
struct mesh_options_t {

    /** This process's index among @ref host_domain_count . */
    u32_t host_domain_index = 0;

    /** Processes in the run. */
    u32_t host_domain_count = 1;

    /** GPUs one NVLink multicast domain spans; zero derives this process's devices. */
    u32_t gpus_per_multimem_domain = 0;

    /** The directory every host sees, required with peers. */
    sz::string_view_t directory {};

    /** The run's unique name inside @ref directory , required with peers. */
    sz::string_view_t job {};

    /** Seconds any wait for a peer lasts before failing. */
    f64_t timeout_seconds = 120.0;
};

/** One process's place among the processes of a run: the topology every process agreed on at bootstrap, and the
 *  InfiniBand domain joining its NVLink domains. Placed and never moved, since @ref colocated borrows
 *  @ref topology . */
struct mesh_t {

    /** The bootstrapped world. */
    mesh_topology_t topology {};

    /** The InfiniBand adapters and collective ids every trainer's handles wire through. */
    colocated_domain_t colocated {};

    /** Seconds any later wait for a peer lasts, from @ref mesh_options_t::timeout_seconds . */
    f64_t timeout_seconds = 120.0;

    /**
     *  @brief Bootstrap @p placed from @p options over @p devices , this process's members.
     *
     *  A process may drive any contiguous run of its NVLink domain's members — 4 of 72 on an NVL72 tray. @p devices
     *  are those members in member order: every visible ordinal, ordinal @c d being member
     *  @c index_in_multimem_domain(first_gpu_index()+d) .
     */
    [[nodiscard]] static expected_t make(mesh_t &placed, mesh_options_t const &options,
                                         std::span<i32_t const> devices) noexcept;
};

/** The rows a run's batch envelope holds per GPU, and the device measurements they were derived
 *  from or, for configured rows, checked against. */
struct envelope_t {

    /** Rows per GPU, with as many segments. */
    u32_t rows = 0;

    /** The fewest bytes any of this process's devices had free before the run allocated. */
    std::size_t free_bytes = 0;

    /** The device footprint @c fixed_bytes+rows·row_bytes+rows²·pair_bytes fitted through the
     *  probe envelopes, its curvature read off the arena, whose high-water mark is exact. */
    std::size_t fixed_bytes = 0, row_bytes = 0, pair_bytes = 0;

    /** The headroom kept free: the footprint outside the arena the larger probe measured, again. */
    std::size_t margin_bytes = 0;

    /** Every arena's reservation: its measured high-water line one row past @ref rows , the row of
     *  headroom the batch sources allocate their loaded batches from. */
    std::size_t reserve_bytes = 0;
};

/** What one step reports without a readback. */
struct step_report_t {

    /** The optimizer step applied. */
    optimizer_step_t optimizer_step {};

    /** The learning rate applied. */
    f32_t learning_rate = 0.0f;

    /** Items cut at a row's width. */
    std::size_t truncated = 0;
};

/** One step's losses, every member of this process summed before the fold to means. */
struct telemetry_t {

    /** The optimizer step this record belongs to. */
    optimizer_step_t optimizer_step {};

    /** The token head's mean masked cross-entropy, weighted by its λ. */
    f32_t token_ce = 0.0f;

    /** Each live media head's mean normalized reconstruction error, weighted by its λ (zero when absent). */
    f32_t image_mse = 0.0f, video_mse = 0.0f, audio_mse = 0.0f;

    /** The world's contrastive per-vertex-mean potential, every process's summands added, or zero
     *  when the objective has none. */
    f32_t contrastive = 0.0f;

    /** The domain's barrier round after this optimizer step. */
    concurrency_round_t concurrency_round {};

    /** The composite loss — every live term. */
    [[nodiscard]] f32_t composite() const noexcept {
        return contrastive + token_ce + image_mse + video_mse + audio_mse;
    }

    /** Print one progress line for this step. */
    void report() const noexcept;
};

struct trainer_t;

/** The one place data enters a step: fills every replica's batch, and the relations when the objective
 *  contrasts, for one data epoch. */
struct batch_source_t {

    /** Fill every replica's batch of @p trainer for @p data_epoch , counting cut items into @p report . */
    [[nodiscard]] virtual expected_t fill(trainer_t &trainer, data_epoch_t data_epoch,
                                          step_report_t &report) noexcept = 0;

  protected:
    ~batch_source_t() noexcept = default;
};

/** The keyed synthetic draw: every world GPU draws distinct data from the run seed, with self-edge relations. */
struct synthetic_source_t final : batch_source_t {
    [[nodiscard]] expected_t fill(trainer_t &trainer, data_epoch_t data_epoch, step_report_t &report) noexcept override;
};

/** One edge set over a step's items as compressed sparse rows; null offsets name no edges. */
struct item_edges_t {
    std::size_t const *offsets = nullptr;
    u32_t const *ids = nullptr;
    f32_t const *weights = nullptr;
};

/** This process's share of one step: every item's modality and bytes, and its optional bag, temperature group
 *  and relations, all borrowed until the step returns. */
struct training_items_t {
    std::span<modality_t const> modalities {};

    /** The items' bytes end to end, and @c count+1 offsets into them. */
    b8_t const *contents = nullptr;
    std::size_t const *offsets = nullptr;

    /** Null keeps every item its own bag. */
    u32_t const *bags = nullptr;

    /** Null puts every item in group zero. */
    u32_t const *temperature_groups = nullptr;

    item_edges_t attract {}, repel {}, ignore {};
};

/**
 *  @brief Caller items packed per device through the host decoders, routed across the mesh by the mixer, with
 *      the relations rebased onto the vertices the items landed on.
 *
 *  Owns everything the routing needs beside the trainer: the loaded batches the mixer reads, their pinned host
 *  twins, the mixer itself, and the host scratch the relations are rebuilt in — all allocated at @ref make .
 */
struct items_source_t final : batch_source_t {

    /** The next step's items. */
    training_items_t items {};

    /** Allocate the routing state from @p trainer 's arenas and wire the mixer through its mesh. */
    [[nodiscard]] static expected_t make(items_source_t &placed, trainer_t &trainer) noexcept;

    [[nodiscard]] expected_t fill(trainer_t &trainer, data_epoch_t data_epoch, step_report_t &report) noexcept override;

    /** The mesh-wide vertex item @p item landed on in the last step, or @c ~0u when it carried no tokens. */
    [[nodiscard]] u32_t destination(std::size_t item) const noexcept { return destinations_[item]; }

    /** Download every replica's pooled vertices, @b [gpus·max_segments,hidden] by mesh-wide vertex. */
    [[nodiscard]] expected<bf16r2_view_t> pooled(trainer_t &trainer) noexcept;

  private:
    [[nodiscard]] expected_t pack_(trainer_t &trainer, step_report_t &report) noexcept;
    [[nodiscard]] expected_t relate_(trainer_t &trainer) noexcept;

    per_gpu_array<multimodal_batch_on_gpu_t> loaded_ {};
    per_gpu_array<multimodal_batch_on_cpu_t> staged_ {};
    multimodal_batch_mixer_t mixer_ {};

    /** Per item: its tokens and the device segment it was packed into. */
    array<u32_t> tokens_ {}, placements_ {}, destinations_ {}, order_ {};

    /** @b [gpus,segments] per routed segment: its world source, then the item it carries. */
    u32r2_tensor_pinned_t sources_ {};

    /** @b [gpus,segments] the item packed into each device segment of the mesh, or @c ~0u . */
    u32r2_tensor_pinned_t packed_items_ {};

    /** One device's relations rebuilt on host: @b [3,segments+1] offsets and @b [3,edges] IDs of
     *  the attract, repel and ignore sets, @b [2,edges] weights of the first two, and the live
     *  length of every mesh vertex. */
    u32r2_tensor_pinned_t offsets_ {}, ids_ {};
    nk::tensor<f32_t, pinned_alloc<f32_t>, 2> weights_ {};
    array<u32_t> lengths_ {};
    array<temperature_group_t> groups_ {};

    /** The decode window of one recording, and the @b [gpus,segments,hidden] pooled vertices
     *  downloaded for the embeddings. */
    array<f32_t> window_ {};
    nk::tensor<bf16_t, pinned_alloc<bf16_t>, 3> pooled_ {};
};

/** Everything one run owns on this process's devices, in wire order, placed and never moved
 *  because its consumers borrow the domain. */
struct trainer_t {

    /** The run description every method reads — the single copy. */
    trainer_config_t config {};

    /** The borrowed mesh this run's collectives wire through. */
    mesh_t *mesh = nullptr;

    /** The device ordinals this process drives. */
    array<i32_t> devices {};

    /** The measured envelope @ref config 's batch was sized from. */
    envelope_t envelope {};

    /** One owning stream per GPU — the ordered chain a replica's whole step enqueues onto. */
    per_gpu_array<gpu_stream_t> streams {};

    /** One owning arena per GPU, visible to every local peer — the allocation budget its replica draws from. */
    per_gpu_array<gpu_arena_t> arenas {};

    /** The run's one telemetry ledger, its rows in the lead arena. */
    telemetry_ledger_t ledger {};

    /** The ledger rows: one loss row per head, one round row per member — inert zero handles whenever their
     *  channels are dark or their head is absent, so feeding them never branches. */
    struct probes_t {
        telemetry_probe_t token {}, image {}, video {}, audio {}, contrastive {};
        per_gpu_array<telemetry_probe_t> rounds {};
    };
    probes_t probes {};

    /** One model replica per GPU, each holding its own @c weights registration list. */
    per_gpu_array<base_replica_t> replicas {};

    /** The NVLink domain, sized by the optimizer's, the contrastive loss's and the mixer's windows. */
    multimem_domain_t domain {};

    /** The cross-domain gradient all-reduce, wired on member 0 — the gateway — and unwired in every process not
     *  holding it, which still mints its collective id. */
    infiniband_all_reduce<f32_t> world_reduce {};

    /** The gradient reduce and the Adafactor step of every replica. */
    distributed_adafactor_t optimizer {};

    /** The mesh-wide contrastive loss — wired only when the objective contrasts. */
    distributed_contrastive_segments_t contrastive {};

    /** One temperature step per replica over its learnable β and bias groups, present when the
     *  objective contrasts. */
    per_gpu_array<adafactor_temperatures_t> temperatures {};

    /** One captured forward and backward region per GPU under @c replayed_k ; empty runs the regions eagerly. */
    per_gpu_array<gpu_replay_graph_t> forward_graphs {}, backward_graphs {};

    /** Optimizer steps taken, counted across resumes — the step a checkpoint records. */
    u32_t completed_steps = 0;

    /**
     *  @brief Build @p placed over @p mesh in wire order.
     *
     *  Per GPU a stream, an arena and a session, then the ledger, the replicas seeded identically,
     *  the domain, the world all-reduce, the optimizer and the contrastive loss, then the replay
     *  graphs unless the config runs eagerly.
     *
     *  The envelope is measured first: a trainer at one and at two rows per GPU takes one step
     *  each, and every device's footprint and arena high-water mark are fitted linearly in rows.
     *  Zero rows in @p requested take the most rows that leave the margin free on every device of
     *  every process; configured rows are honoured or refused, never shrunk. The arenas reserve the
     *  fitted high-water mark, see @ref envelope_t .
     *  @retval status_t::bad_alloc_k The envelope does not fit, with the measured numbers.
     */
    [[nodiscard]] static expected_t make(trainer_t &placed, mesh_t &mesh, trainer_config_t const &requested,
                                         std::span<i32_t const> devices) noexcept;

    /**
     *  @brief Load every replica from the checkpoint at @p path and continue from its saved step,
     *      with the optimizer's moments and masters and the contrastive temperatures it holds; the
     *      state it misses restarts, the masters re-seeded from the loaded weights.
     *  @retval status_t::invalid_input_k A matrix the checkpoint misses.
     *  @retval status_t::shape_unsupported_k A matrix or a state of another shape.
     */
    [[nodiscard]] expected_t resume(sz::string_view_t path) noexcept;

    /** Capture each replica's forward and backward regions into @ref forward_graphs and @ref backward_graphs . */
    [[nodiscard]] expected_t capture() noexcept;

    /** One optimizer step: @p source fills the batches, then per replica the preamble and the forward region, the
     *  contrastive collective, and per replica the backward region and the optimizer's slice. */
    [[nodiscard]] expected_t step(batch_source_t &source, optimizer_step_t optimizer_step,
                                  step_report_t &report) noexcept;

    /** Re-seed every replica's weights and temperatures to the identical deterministic init. */
    [[nodiscard]] expected_t reseed() noexcept;

    /** Drain every replica and read one @ref telemetry_t , feeding the ledger's loss rows and one round probe per
     *  member. */
    [[nodiscard]] telemetry_t telemetry(optimizer_step_t optimizer_step) noexcept;

    /** Write the lead replica to @p path with @ref completed_steps : the @c model. weights, the
     *  @c optimizer. state and the @c contrastive. temperatures; only host domain 0 writes. */
    [[nodiscard]] expected_t save(sz::string_view_t path) noexcept;

    /** Wait until every replica's stream drains. */
    [[nodiscard]] expected_t synchronize() noexcept;
};

} // namespace unum::uform::train
