/**
 *  @file include/uform/core/optimizer.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief Adafactor, the memory-frugal optimizer of the uniform towers: factored Float32 second
 *      moments (one row and one column vector per matrix instead of a full moment), RMS-clipped
 *      updates, and an optional bias-corrected first moment of the clipped update.
 *
 *  The sibling @ref distributed_adafactor_t reduces the gradients over the same two-level mesh transports
 *  the loss uses and keeps every rank's weight replica bitwise identical.
 *
 *  Per weight matrix @c W[rows,cols] with gradient @c G at step @c t counted from one:
 *
 *      β₂ₜ = min(β₂, 1 − t^−0.8)                         — the paper's decay, unbiased from the
 *                                                          first step,
 *      r_i ← β₂ₜ·r_i + (1−β₂ₜ)·(mean_j G²_ij + ε₁)      — the row moment,
 *      c_j ← β₂ₜ·c_j + (1−β₂ₜ)·(mean_i G²_ij + ε₁)      — the column moment,
 *      v_ij = r_i·c_j / mean(r)                          — the factored second-moment estimate,
 *      u = G/√v,   u ← u / max(1, rms(u)/d)              — the RMS-clipped update,
 *      W ← W − lr·u .
 *
 *  Matrices round the new weight into BFloat16 stochastically, keyed on the step, the state and the
 *  scalar, so a step far below the BFloat16 spacing still moves the weight by itself on average.
 *  Vector weights (the norm scales, @c rows==1 ) fall out of the same formulas as a full second
 *  moment, and keep a Float32 @b master copy instead. Every reduction is a fixed-order tree; the
 *  update is bit-reproducible.
 *
 *  @section optimizer_distributed Distributed replicas stay bitwise identical
 *
 *  Weights are data-parallel replicas. Each rank casts its BFloat16 gradient buffers into its
 *  multicast replica, one generation barrier proves every replica complete, and every rank
 *  materialises the @b same Float32 gradient @b mean (the switch sum scaled by
 *  @c 1/gpus_per_colocated_domain , so learning-rate semantics are world-size-invariant) through
 *  @c multimem.ld_reduce — deterministic by construction — so identical Adafactor states receive
 *  identical inputs and every replica takes the identical step. In a single-domain world a second
 *  barrier after the reduce keeps every window intact until all members read it, since the next
 *  step stages into the same windows. Across multimem_domains the gateway, domain member 0 wherever
 *  its process sits, rides its pre-scaled domain term through the GDAKI all-reduce and writes the
 *  world mean back into its own symmetric window, and every member reads it there over NVLink after
 *  one more domain barrier.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <array> // `std::array`
#include <span>  // `std::span` — per-rank object lists and the weights tables

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "uform/core/checkpoint.hpp"     // `checkpoint_to_save_t`, `checkpoint_to_load_t` — the state a resume reads
#include "uform/core/gpu_allocators.hpp" // `gpu_arena_t`
#include "uform/core/infiniband_collectives.cuh" // `infiniband_all_reduce` — the borrowed level-2 buffer
#include "uform/core/introspect.cuh"             // `telemetry_ledger_t` — the borrowed signal-probe ledger
#include "uform/core/multimem_collectives.cuh"   // `multimem_domain_t` — the main-owned level-1 gradient surface
#include "uform/core/schedule.cuh"               // `gpu_stream_t`, `launch_kernel`, `grid_size_for`
#include "uform/core/types.hpp"                  // `expected`, `expected_t`, `status_t`, the scalar aliases

namespace unum::uform::core {

/** The update's baked knobs; the defaults follow the Adafactor paper's stable regime. The learning rate
 *  is deliberately absent: it varies per step (the trainer owns the warmup / decay policy), so it travels as
 *  a @ref adafactor_t::step argument — legal because the optimizer region is eager, never captured. */
struct adafactor_config_t {

    /** The ceiling of the second-moment decay — how much of the running row / column moments a late
     *  step retains. */
    f32_t beta2 = 0.999f;

    /** The exponent of the decay schedule @c 1−t^−e , the paper's 0.8, which reaches @ref beta2
     *  after 5623 steps. */
    f32_t decay_exponent = 0.8f;

    /** The moment floor added before the running average, keeping @c 1/√v finite for a zero gradient. */
    f32_t epsilon1 = 1.0e-30f;

    /** RMS clip threshold @c d — updates larger than this root-mean-square are scaled down. */
    f32_t clip_threshold = 1.0f;

    /** The first-moment decay of the clipped update, so a step follows the last few batches
     *  instead of one; zero keeps no first moment. */
    f32_t beta1 = 0.0f;
};

/** One weight matrix's window into the optimizer's combined Float32 state — plain spans and
 *  offsets, resolved once at construction, which the kernels take by value. */
struct adafactor_state_t {

    /** The weights and their gradients, borrowed from the tower. */
    nk::tensor_span<bf16_t, 2> weights {};
    nk::tensor_view<bf16_t, 2> weights_grad {};

    /** Offsets into the combined moments tensor: @b [rows] then @b [cols] . */
    std::size_t row_moment_offset = 0;
    std::size_t column_moment_offset = 0;

    /** Offset into the combined Float32 master tensor; only vector weights have one. */
    std::size_t master_offset = 0;

    /** Offset into the flattened world gradient (the distributed reduce's currency). */
    std::size_t gradient_offset = 0;
};

/** One rank's Adafactor: the combined moments, masters, and per-row update energies as owned arena
 *  tensors, the state table resolved once at @ref make — a @ref step is a pure enqueue over the
 *  tower's gradient buffers (or a reduced world gradient the distributed sibling supplies). */
struct adafactor_t {

    /** Resolve the weights table into combined-state offsets, allocate the state, zero the
     *  moments, and seed the vector masters — once, on @p stream , outside any capture. */
    [[nodiscard]] static expected<adafactor_t> make(gpu_arena_t &arena,
                                                    std::span<differentiable_weights_t const> weights,
                                                    adafactor_config_t config, gpu_stream_t &stream) noexcept;

    /** The resolved states, for the distributed sibling's staging pass. */
    [[nodiscard]] nk::tensor_view<adafactor_state_t, 1> states() const noexcept {
        return std::span<adafactor_state_t const> {states_};
    }

    /** One optimizer step: moments, clip scales, apply — per state, pure enqueue. A non-empty
     *  @p reduced_gradients redirects every gradient read to the Float32 world sum. The scheduled
     *  @p learning_rate is this step's rate — the caller's warmup / decay policy, host-computed. */
    [[nodiscard]] expected_t step(f32_t learning_rate, gpu_stream_t &stream,
                                  f32r1_view_t reduced_gradients = {}) noexcept;

    /** Insert every state's moments into @p checkpoint as @c optimizer.{name}.row_moments and
     *  @c optimizer.{name}.column_moments , and each vector weight's master as
     *  @c optimizer.{name}.master ; drains @p stream . */
    [[nodiscard]] expected_t save(checkpoint_to_save_t &checkpoint, gpu_stream_t &stream) const noexcept;

    /**
     *  @brief Continue from @p checkpoint : the moments and masters it holds are uploaded on
     *      @p stream , the rest restart — moments at zero, masters re-seeded from the live weights,
     *      which the caller loads first. The step count continues from the checkpoint's step.
     *
     *  The caller drains @p stream before releasing @p checkpoint .
     *  @retval status_t::shape_unsupported_k A saved state of another shape.
     */
    [[nodiscard]] expected_t load(checkpoint_to_load_t const &checkpoint, gpu_stream_t &stream) noexcept;

  private:

    /** The distributed sibling reads @ref gradient_scalars over the shared states. */
    friend struct distributed_adafactor_t;

    /** Total gradient scalars across the states — the distributed reduce's surface size, summed
     *  from the live state extents rather than stored. */
    [[nodiscard]] std::size_t gradient_scalars() const noexcept {
        std::size_t scalars = 0;
        for (adafactor_state_t const &state : states_) scalars += state.weights.extent(0) * state.weights.extent(1);
        return scalars;
    }

    adafactor_config_t config_ {};

    /** Steps taken, which the decay schedule and the rounding keys read. */
    u64_t steps_ = 0;

    /** One optimizer state per weight matrix, resolved once at @ref make . */
    st::basic_vector<adafactor_state_t> states_ {};

    /** Each state's weight name, in state order, which its checkpoint entries carry. */
    st::basic_vector<sz::string_t> names_ {};

    /** Every state's row + column moments, back to back in state order. */
    f32r1_tensor_arena_t moments_ {};

    /** The vector weights' Float32 masters, back to back in state order. */
    f32r1_tensor_arena_t masters_ {};

    /** Each state row's squared-update energy at its row-moment offset, which the apply pass folds
     *  into the clip scale; the column entries stay unused. */
    f32r1_tensor_arena_t row_energies_ {};

    /** Every state's Float32 first moment of the clipped update, back to back in gradient order;
     *  empty when the config keeps none. */
    f32r1_tensor_arena_t first_moments_ {};
};

/** The world-wide optimizer: one @ref adafactor_t per rank plus the two-level gradient reduce
 *  over the main-owned domains. Every rank's replica takes a bitwise-identical step — the reduce is
 *  switch-deterministic and the states start identical. */
struct distributed_adafactor_t {

    /** One rank's state and its materialised world-gradient buffer (single-domain worlds only —
     *  multi-domain worlds read the gateway member's window instead), plus the rank's registered
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
     *  the world sum is peer-read back over NVLink. The ranks are this process's members of the domain,
     *  @c [domain.member_offset(),+arenas.size()) . The optimizer constructs no transport of its own:
     *  the buffer is wired once at the composition root and borrowed here, exactly like the domain —
     *  so many optimizers (the conformance suite's case matrix) share one set of queue pairs.
     *
     *  @param[inout] domain The main-owned level-1 surface; one gradient window is reserved from it.
     *  @param[inout] world_reduce The main-owned cross-domain all-reduce buffer wired on member 0 — a zero-round
     *      identity in single-domain worlds, unwired in processes not holding member 0. On the gateway its payload
     *      must span at least this optimizer's gradient scalars; the unread tail beyond them is harmless.
     *  @param[in] weights Rank-major @b [ranks·weights_per_rank] weights table.
     */
    [[nodiscard]] static expected<distributed_adafactor_t> make(
        multimem_domain_t &domain, infiniband_all_reduce<f32_t> &world_reduce, std::span<gpu_arena_t> arenas,
        std::span<differentiable_weights_t const> weights, std::size_t weights_per_rank, adafactor_config_t config,
        std::span<gpu_stream_t> streams, telemetry_ledger_t &ledger) noexcept;

    /**
     *  @brief Local rank @p rank 's slice of one world step — pure enqueue onto @p stream , the shape a
     *      fork/join capture body composes directly.
     *
     *  Stage the rank's BFloat16 grads into its window, barrier, materialise the switch-reduced
     *  world gradient, barrier again so no member restages its window while a peer still reads it,
     *  and take the identical Adafactor step; across multimem_domains the gateway stages after the
     *  barrier, all-reduces and publishes the world mean in its window, and a second barrier lets
     *  every rank read it. Ordering contract: this consumer's advance/barrier phase must follow the
     *  same cross-consumer order on every rank (see @ref multimem_domain_t ); ranks are enqueued in
     *  order @c 0..gpus-1 within a step. The scheduled @p learning_rate must be identical on every
     *  rank of the step — replicas step in lockstep.
     */
    [[nodiscard]] expected_t step(std::size_t rank, f32_t learning_rate, gpu_stream_t &stream) noexcept;

    /** The eager convenience: every rank's slice in order over the per-rank streams. */
    [[nodiscard]] expected_t step(f32_t learning_rate, std::span<gpu_stream_t> streams) noexcept;

    /** Insert the first rank's state into @p checkpoint — the ranks are identical. */
    [[nodiscard]] expected_t save(checkpoint_to_save_t &checkpoint, gpu_stream_t &stream) const noexcept {
        return ranks_[0].state.save(checkpoint, stream);
    }

    /** Continue every rank from @p checkpoint on its stream; see @ref adafactor_t::load . */
    [[nodiscard]] expected_t load(checkpoint_to_load_t const &checkpoint, std::span<gpu_stream_t> streams) noexcept;

  private:

    /** The D>1 tail of one rank's step: the gateway materialises the domain sum into its world
     *  buffer, all-reduces it across domains and copies the world mean into its window; after a domain
     *  barrier every rank steps against that window. Caller holds the rank's context. */
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
};

/**
 *  @brief Adafactor over the learnable scalars of the contrastive loss: every temperature group's
 *      inverse temperature β and logit bias, Float32 already, so a full second moment and no
 *      master copy.
 *
 *  The β step runs in log space, @c β ← β·exp(−lr·u) , so β stays positive and the update reads
 *  the log-β gradient the loss reports; the bias steps additively. The RMS clip spans both halves
 *  at once. Every rank steps its own replica's scalars against the same world-summed gradients, so
 *  the replicas stay bitwise identical.
 */
struct adafactor_temperatures_t {

    /** Allocate and zero the @b [2·groups] second moment off @p arena on @p stream ,
     *  checkpointed as @c temperature_moments under the loss's @p prefix , such as
     *  @c contrastive . */
    [[nodiscard]] static expected<adafactor_temperatures_t> make(gpu_arena_t &arena, u32_t groups,
                                                                 adafactor_config_t config, sz::string_view_t prefix,
                                                                 gpu_stream_t &stream) noexcept;

    /** One step of the @b [groups] @p inverse_temperatures and @p biases in place, from the
     *  @b [2·groups] @p gradients : every group's log-β gradient, then its bias gradient. */
    [[nodiscard]] expected_t step(f32r1_span_t inverse_temperatures, f32r1_span_t biases, f32r1_view_t gradients,
                                  f32_t learning_rate, gpu_stream_t &stream) noexcept;

    /** Insert the second moment into @p checkpoint under its key, draining @p stream . */
    [[nodiscard]] expected_t save(checkpoint_to_save_t &checkpoint, gpu_stream_t &stream) const noexcept;

    /**
     *  @brief Upload the second moment @p checkpoint holds on @p stream ; a file without one
     *      restarts it at zero.
     *  @retval status_t::shape_unsupported_k A saved moment of another group count.
     */
    [[nodiscard]] expected_t load(checkpoint_to_load_t const &checkpoint, gpu_stream_t &stream) noexcept;

  private:
    adafactor_config_t config_ {};

    /** @b [2·groups] second moments, the log-β ones first. */
    f32r1_tensor_arena_t moments_ {};

    /** The checkpoint key of @ref moments_ , the loss's prefix then @c .temperature_moments . */
    sz::string_t key_ {};
};

} // namespace unum::uform::core
