/**
 *  @brief Adafactor, the memory-frugal optimizer of the uniform towers: factored Float32 second moments
 *      (one row and one column vector per matrix instead of a full moment), RMS-clipped updates, no
 *      momentum.
 *  @file src/core/optimizer.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  The sibling @ref distributed_adafactor_t reduces the gradients over the same two-level mesh transports
 *  the loss uses and keeps every rank's weight replica bitwise identical.
 *
 *  Per weight matrix @c W[rows,cols] with gradient @c G :
 *
 *      r_i ← β₂·r_i + (1−β₂)·(mean_j G²_ij + ε₁)        — the row moment,
 *      c_j ← β₂·c_j + (1−β₂)·(mean_i G²_ij + ε₁)        — the column moment,
 *      v_ij = r_i·c_j / mean(r)                          — the factored second-moment estimate,
 *      u = G/√v,   u ← u / max(1, rms(u)/d)              — the RMS-clipped update,
 *      W ← W − lr·u .
 *
 *  Vector weights (the norm scales, @c rows==1 ) fall out of the same formulas as a full second
 *  moment, and additionally keep a Float32 @b master copy — a magnitude-1 BFloat16 scale would swallow
 *  small updates in its ULP. Every reduction is a fixed-order tree; the update is bit-reproducible.
 *
 *  @section optimizer_distributed Distributed replicas stay bitwise identical
 *
 *  Weights are data-parallel replicas. Each rank casts its BFloat16 gradient buffers into its multicast
 *  replica, one generation barrier proves every replica complete, and every rank materialises the @b same
 *  Float32 gradient @b mean (the switch sum scaled by @c 1/gpus_per_colocated_domain , so learning-rate
 *  semantics are world-size-invariant) through @c multimem.ld_reduce — deterministic by construction — so
 *  identical Adafactor states receive identical inputs and every replica takes the identical step. Across
 *  multimem_domains the gateway's pre-scaled domain term rides the GDAKI all-reduce first, and every
 *  member reads the gateway's world mean over NVLink.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <array> // `std::array`
#include <span>  // `std::span` — per-rank object lists and the weights tables

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "core/types.hpp"                  // `expected`, `expected_t`, `status_t`, the scalar aliases
#include "core/schedule.cuh"               // `gpu_stream_t`, `launch_kernel`, `grid_size_for`
#include "core/gpu_allocators.hpp"         // `gpu_arena_t`
#include "core/introspect.cuh"             // `telemetry_ledger_t` — the borrowed signal-probe ledger
#include "core/multimem_collectives.cuh"   // `multimem_domain_t` — the main-owned level-1 gradient surface
#include "core/infiniband_collectives.cuh" // `infiniband_all_reduce` — the borrowed level-2 buffer

namespace unum::uform::core {

/** The update's baked knobs; the defaults follow the Adafactor paper's stable regime. The learning rate
 *  is deliberately absent: it varies per step (the trainer owns the warmup / decay policy), so it travels as
 *  a @ref adafactor_t::step argument — legal because the optimizer region is eager, never captured. */
struct adafactor_config_t {

    /** The second-moment decay — how much of the running row / column moments each step retains. */
    f32_t beta2 = 0.999f;

    /** The moment floor added before the running average, keeping @c 1/√v finite for a zero gradient. */
    f32_t epsilon1 = 1.0e-30f;

    /** RMS clip threshold @c d — updates larger than this root-mean-square are scaled down. */
    f32_t clip_threshold = 1.0f;
};

/** One weight matrix's window into the optimizer's combined Float32 state — plain offsets,
 *  resolved once at construction. */
struct adafactor_state_t {

    /** The weights themselves, borrowed from the tower. */
    differentiable_weights_t weights {};

    /** Offsets into the combined moments tensor: @b [rows] then @b [cols] . */
    std::size_t row_moment_offset = 0;
    std::size_t column_moment_offset = 0;

    /** Offset into the combined Float32 master tensor; only vector weights have one. */
    std::size_t master_offset = 0;

    /** Offset into the flattened world gradient (the distributed reduce's currency). */
    std::size_t gradient_offset = 0;
};

/** One rank's Adafactor: the combined moments, masters, and per-state scale scalars as owned
 *  arena tensors, the state table resolved once at @ref make — a @ref step is a pure enqueue over
 *  the tower's gradient buffers (or a reduced world gradient the distributed sibling supplies). */
struct adafactor_t {

    /** Resolve the weights table into combined-state offsets, allocate the state, zero the
     *  moments, and seed the vector masters — once, on @p stream , outside any capture. */
    [[nodiscard]] static expected<adafactor_t> make(gpu_arena_t &arena,
                                                    std::span<differentiable_weights_t const> weights,
                                                    adafactor_config_t config, gpu_stream_t &stream) noexcept;

    /** The resolved states, for the distributed sibling's staging pass. */
    [[nodiscard]] nk::tensor_view<adafactor_state_t, 1> states() const noexcept {
        return {states_.view().data(), states_.size()};
    }

    /** One optimizer step: moments, clip scales, apply — per state, pure enqueue. A non-empty
     *  @p reduced_gradients redirects every gradient read to the Float32 world sum. The scheduled
     *  @p learning_rate is this step's rate — the caller's warmup / decay policy, host-computed. */
    [[nodiscard]] expected_t step(f32_t learning_rate, gpu_stream_t &stream,
                                  f32r1_view_t reduced_gradients = {}) noexcept;

  private:

    /** The distributed sibling reads @ref gradient_scalars over the shared states. */
    friend struct distributed_adafactor_t;

    /** Total gradient scalars across the states — the distributed reduce's surface size, summed
     *  from the live state extents rather than stored. */
    [[nodiscard]] std::size_t gradient_scalars() const noexcept {
        std::size_t scalars = 0;
        for (std::size_t index = 0; index != states_.size(); ++index) {
            differentiable_weights_t const &matrix = states_[index].weights;
            scalars += matrix.weights.extent(0) * matrix.weights.extent(1);
        }
        return scalars;
    }

    adafactor_config_t config_ {};

    /** One optimizer state per weight matrix, resolved once at @ref make . */
    array<adafactor_state_t> states_ {};

    /** Every state's row + column moments, back to back in state order. */
    f32r1_tensor_arena_t moments_ {};

    /** The vector weights' Float32 masters, back to back in state order. */
    f32r1_tensor_arena_t masters_ {};

    /** @b [states,2] per-state (row-moment mean, clip scale) scalars the apply pass reads. */
    f32r2_tensor_arena_t state_scales_ {};
};

/** The world-wide optimizer: one @ref adafactor_t per rank plus the two-level gradient reduce
 *  over the main-owned domains. Every rank's replica takes a bitwise-identical step — the reduce is
 *  switch-deterministic and the states start identical. */
struct distributed_adafactor_t {

    /** One rank's state and its materialised world-gradient buffer (single-domain worlds only —
     *  multi-domain worlds read the gateway member's heap instead), plus the rank's registered
     *  gradient-boundary probes — inert zero handles when the signals channel is dark. */
    struct optimizer_rank_t {
        adafactor_t state {};
        f32r1_tensor_arena_t reduced {};
        telemetry_probe_t staged_window {}, world_payload {}, world_readback {};
    };

    /** How many Float32 payload scalars this consumer will @c reserve from the shared
     *  @ref multimem_domain_t — the flattened gradient surface of one rank's @p weights . */
    [[nodiscard]] static std::size_t payload_scalars_needed(
        std::span<differentiable_weights_t const> weights) noexcept {
        std::size_t scalars = 0;
        for (differentiable_weights_t const &matrix : weights)
            scalars += matrix.weights.extent(0) * matrix.weights.extent(1);
        return scalars;
    }

    /**
     *  @brief One construction absorbs the whole world: one Adafactor per rank over its own tower's
     *      weights (all ranks must expose identically-shaped tables), one reserved gradient window of the
     *      shared @p domain , and the borrowed @p world_reduce buffer on the gateway.
     *
     *  Only member 0 touches InfiniBand, because the domain sums are already identical on every member and
     *  the world sum is peer-read back over NVLink. The optimizer constructs no transport of its own:
     *  the buffer is wired once at the composition root and borrowed here, exactly like the domain —
     *  so many optimizers (the conformance suite's case matrix) share one set of queue pairs.
     *
     *  @param[inout] domain The main-owned level-1 surface; one gradient window is reserved from it.
     *  @param[inout] world_reduce The main-owned cross-domain all-reduce buffer — a zero-round identity
     *      in single-domain worlds. Its payload must span at least this optimizer's gradient scalars;
     *      the unread tail beyond them is harmless (the reduce is elementwise).
     *  @param[in] weights Rank-major @b [ranks·weights_per_rank] weights table.
     */
    [[nodiscard]] static expected<distributed_adafactor_t> make(
        multimem_domain_t &domain, infiniband_all_reduce<f32_t> &world_reduce, std::span<gpu_arena_t> arenas,
        std::span<differentiable_weights_t const> weights, std::size_t weights_per_rank, adafactor_config_t config,
        std::span<gpu_stream_t> streams, telemetry_ledger_t &ledger) noexcept;

    /**
     *  @brief Rank @p rank 's slice of one world step — pure enqueue onto @p stream , the shape a
     *      fork/join capture body composes directly. Rank 0's slice carries the domain's generation
     *      advance.
     *
     *  Stage the rank's BFloat16 grads into its window, barrier, materialise the
     *  switch-reduced world gradient, and take the identical Adafactor step; across
     *  multimem_domains the gateway member's all-reduce runs between the materialise and the
     *  update, and every rank reads the gateway's world sum. Ordering contract: this consumer's
     *  advance/barrier phase must follow the same cross-consumer order on every rank (see
     *  @ref multimem_domain_t ); ranks are enqueued in order @c 0..gpus-1 within a step. The scheduled
     *  @p learning_rate must be identical on every rank of the step — replicas step in lockstep.
     */
    [[nodiscard]] expected_t step(std::size_t rank, f32_t learning_rate, gpu_stream_t &stream) noexcept;

    /** The eager convenience: every rank's slice in order over the per-rank streams. */
    [[nodiscard]] expected_t step(f32_t learning_rate, std::span<gpu_stream_t> streams) noexcept;

  private:

    /** The D>1 tail of one rank's step: the gateway materialises the domain sum into its world
     *  member's heap, all-reduces it across domains, and records the event every rank's update
     *  waits on; every rank then steps against the gateway's world sum. Caller holds the rank's
     *  context. */
    [[nodiscard]] expected_t world_gradient_step_(std::size_t rank, f32_t learning_rate, gpu_stream_t &stream) noexcept;

    /** One Adafactor state (and, in single-domain worlds, one reduced buffer) per local GPU. */
    per_gpu_array<optimizer_rank_t> ranks_ {};

    /** The main-owned level-1 surface — borrowed; its replicas carry the staged gradients. */
    multimem_domain_t *domain_ = nullptr;

    /** This consumer's disjoint gradient window of the domain payload, reserved at @ref make . */
    multimem_window_t window_ {};

    /** The main-owned cross-domain all-reduce buffer — borrowed, like @ref domain_ ; its typed
     *  payload is the world gradient, and its @c identity() marks single-domain worlds. */
    infiniband_all_reduce<f32_t> *world_reduce_ = nullptr;

    /** The run's telemetry ledger — borrowed, like @ref domain_ ; the per-rank probes above feed
     *  it at the gradient boundaries, inert whenever the signals channel is dark. */
    telemetry_ledger_t *ledger_ = nullptr;

    /** The gateway's world-sum-ready ordering token every rank's update waits on. */
    gpu_event_t world_ready_ {};
};

} // namespace unum::uform::core
