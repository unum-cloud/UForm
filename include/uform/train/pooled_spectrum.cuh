/**
 *  @file include/uform/train/pooled_spectrum.cuh
 *  @author Ash Vardanian
 *  @date October 27, 2026
 *  @brief The pooled-code regularizers: Total Coding Rate over the world's unit pooled rows, the
 *      rank-local Lipschitz penalty on their top singular value, and the gradient spike gate riding
 *      the same domain window.
 *
 *  With @c Z the rank's @b [n,D] L2-normalized pooled rows, @c N the world's rows and
 *  @c α=D/(N·ε²) , one call adds the gradient of @c −λ_tcr·R+λ_lip·σ² into the pooled-row gradient
 *  the segments loss already wrote:
 *
 *      Term         Forward                                Gradient on the rank's Z     Scope
 *      Coding rate  R = ½·logdet(I+α·G), G = Σ_ranks ZᵀZ   −λ_tcr·α·Z·(I+α·G)⁻¹         world
 *      Lipschitz    σ² = ‖Z·v‖², v by power iteration      2·λ_lip·(Z·v)·vᵀ             rank
 *
 *  Both gradients then pass through the normalization's Jacobian onto the raw rows.
 *
 *  @section spectrum_order Stream order
 *
 *  Every stage of a call runs on the rank's stream in a fixed order, allocates nothing and reads
 *  no host value, so it is capture-safe:
 *
 *      1. the inverse row norms, then the BFloat16 @c Zᵀ NumKong's symmetric kernel reads;
 *      2. the rank's Gram @c ZᵀZ into its window of the multimem domain, summed over the
 *         domain and, on the gateway, over InfiniBand when the world spans several domains;
 *      3. @c A=I+α·G factored in place by a fixed-order blocked Cholesky, @c R=Σlog(L[i,i]) ;
 *      4. the power iteration from the uniform direction, keeping no state between calls;
 *      5. a warp per row: two triangular solves give @c A⁻¹·z , and the force joins the gradient.
 *
 *  Every member sums the same Gram bits and factors them with the same kernels, so every rank of
 *  the world derives the identical @c R and @c A⁻¹ .
 */
#pragma once

#include "uform/utilities/array.hpp"                  // `per_gpu_array`
#include "uform/utilities/infiniband_collectives.cuh" // `colocated_domain_t`, `infiniband_all_reduce`
#include "uform/utilities/multimem_collectives.cuh"   // `multimem_domain_t`, `multimem_window_t`
#include "uform/utilities/vram_nvidia_schedule.cuh"   // `gpu_numerics_t`, `differentiable_weights_t`

namespace unum::uform::train {
using namespace unum::uform::utilities;

/** The baked geometry of the pooled-code regularizers and the spike gate; a different config is
 *  a different object. */
struct pooled_spectrum_config_t {

    /** Pooled rows this rank holds, the rows of @c Z . */
    u32_t rows_per_gpu = 0;

    /** Embedding width, the Gram's side @c D , a multiple of the warp. */
    u32_t hidden = 0;

    /** Pooled rows across the world, the @c N of the coding rate. */
    u32_t rows_per_world = 0;

    /** Coding-rate weight @c λ_tcr and distortion @c ε² ; a zero weight skips the world Gram. */
    f32_t coding_rate_weight = 0.0f, epsilon_squared = 0.5f;

    /** Lipschitz weight @c λ_lip and its power-iteration steps; a zero weight skips it. */
    f32_t lipschitz_weight = 0.0f;
    u32_t power_iterations = 4;

    /** How many times the last admitted step's gradient L2 norm a step may reach before it
     *  fails; zero gates nothing. */
    f32_t spike_threshold = 0.0f;
};

#if defined(__CUDACC__)

/** One gradient matrix the spike gate's energy kernel reads, flattened. */
struct spectrum_gradient_range_t {
    bf16_t const *data = nullptr; // device-boundary
    std::size_t count = 0;
};

/**
 *  @brief Total Coding Rate over the world's pooled rows, the rank-local Lipschitz penalty and the
 *      gradient spike gate, as one function object baked at @ref make , each call a pure enqueue.
 *
 *  Holds one state per local rank, a reserved window of the main-owned @ref multimem_domain_t
 *  whose replicas carry the Gram and the gradient energy, and the gateway's InfiniBand all-reduces
 *  wired through the main-owned @ref colocated_domain_t . Ranks are enqueued in order @c 0..gpus-1
 *  within a step, and every rank issues its domain barriers in the same order as every other
 *  consumer's.
 */
struct distributed_pooled_spectrum_t {

    /** Float32 scalars this consumer reserves from the multimem domain window: the @c D² Gram
     *  while the coding rate is weighted, then the gradient energy while the gate is armed. */
    [[nodiscard]] static constexpr std::size_t payload_scalars_needed(pooled_spectrum_config_t config) noexcept {
        return gram_scalars_(config) + energy_scalars_(config);
    }

    /**
     *  @brief One construction absorbs the whole world: every rank's scratch, this consumer's
     *      window of the shared @p domain , and the gateway's all-reduces through @p colocated .
     *
     *  @param[inout] domain The main-owned level-1 surface; one window is reserved from it.
     *  @param[inout] colocated The main-owned level-2 surface; the Gram and energy all-reduces
     *      are wired on member 0 in every process, unwired where another process drives it.
     *  @param[in] arenas @b [gpus] per-rank arenas, one per member this process drives.
     *  @param[in] streams @b [gpus] per-rank streams; their contexts scope every device's setup.
     *  @param[in] weights Rank-major @b [gpus·weights_per_rank] weights table the gate
     *      measures; empty while the gate is off.
     *  @param[in] weights_per_rank Matrices one rank registers.
     *  @param[in] config The baked geometry and weights.
     *  @param[in] numerics The NumKong kernels every rank's Gram calls.
     *  @param[in] timeout_seconds Deadline on the all-reduces' rendezvous and flag waits.
     */
    [[nodiscard]] static expected<distributed_pooled_spectrum_t> make(
        multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
        std::span<gpu_stream_t> streams, std::span<differentiable_weights_t const> weights,
        std::size_t weights_per_rank, pooled_spectrum_config_t config, gpu_numerics_t const &numerics,
        f64_t timeout_seconds = 120.0) noexcept;

    /**
     *  @brief Local rank @p rank 's regularizers, pure enqueue onto @p stream : the coding rate
     *      and the Lipschitz penalty of @p pooled , their force added into @p pooled_grad .
     *  @param[in] rank The local rank, enqueued in order within a step.
     *  @param[in] pooled @b [rows_per_gpu,hidden] the raw pooled rows, normalized here.
     *  @param[inout] pooled_grad @b [rows_per_gpu,hidden] their gradient, added onto.
     *  @param[inout] stream The rank's stream.
     */
    [[nodiscard]] expected_t operator()(std::size_t rank, bf16r2_view_t pooled, bf16r2_span_t pooled_grad,
                                        gpu_stream_t &stream) noexcept;

    /** Local rank @p rank 's @b [2] terms of the last call on its device: the world coding rate
     *  @c R , identical on every rank, then the rank's top Gram eigenvalue @c σ² . */
    [[nodiscard]] f32r1_view_t terms(std::size_t rank) const noexcept { return ranks_[rank].terms.view(); }

    /** Enqueue local rank @p rank 's squared gradient L2 norm and its sum over the world, after
     *  the rank's backward; ranks are enqueued in order within a step. */
    [[nodiscard]] expected_t measure_gradients(std::size_t rank, gpu_stream_t &stream) noexcept;

    /**
     *  @brief Read the world's gradient L2 norm off rank 0 through @p stream , its stream, and
     *      fail when it passes the threshold times the last admitted step's or is not finite.
     *
     *  Every process reads the same summed bits, so every process takes the same decision with
     *  no further exchange. Drains @p stream , so never call it inside a capture.
     *  @retval status_t::internal_k The step's gradient spiked; the optimizer must not apply it.
     */
    [[nodiscard]] expected_t admit_gradients(gpu_stream_t &stream) noexcept;

  private:

    /** One local rank's scratch, every buffer on its own arena and empty while its term is off. */
    struct spectrum_rank_t {

        /** @b [rows] the inverse L2 norm of every pooled row, zero for an all-zero row. */
        f32r1_tensor_arena_t inverse_norms {};

        /** @b [hidden,rows_padded] BFloat16 @c Zᵀ , its rows padded to 8 with zeros for
         *  NumKong's 16-byte stride rule. */
        bf16r2_tensor_arena_t transposed {};

        /** @b [hidden,hidden] the world Gram, then @c L in its lower triangle and @c Lᵀ in
         *  its upper. */
        f32r2_tensor_arena_t factor {};

        /** @b [rows,hidden] each row's @c A⁻¹·z , solved in place. */
        f32r2_tensor_arena_t solves {};

        /** @b [hidden] the power iteration's direction @c v , and @b [rows] the projections
         *  @c Z·v . */
        f32r1_tensor_arena_t direction {}, projections {};

        /** @b [2] the coding rate and the top eigenvalue of the last call. */
        f32r1_tensor_arena_t terms {};

        /** @b [matrices] the rank's gradient matrices, and @b [matrices,blocks] each CTA's
         *  squared sum. */
        nk::tensor<spectrum_gradient_range_t, gpu_arena_alloc<spectrum_gradient_range_t>, 1> gradients {};
        f32r2_tensor_arena_t energy_partials {};

        /** @b [1] the world's squared gradient L2 norm. */
        f32r1_tensor_arena_t energy {};
    };

    /** The Gram's part of the window, the energy's offset into it. */
    [[nodiscard]] static constexpr std::size_t gram_scalars_(pooled_spectrum_config_t config) noexcept {
        return config.coding_rate_weight > 0.0f ? std::size_t {config.hidden} * config.hidden : 0u;
    }

    /** The energy's part of the window. */
    [[nodiscard]] static constexpr std::size_t energy_scalars_(pooled_spectrum_config_t config) noexcept {
        return config.spike_threshold > 0.0f ? 1u : 0u;
    }

    /** Sum @p destination 's extent of scalars from window offset @p begin over the world into
     *  @p destination : a barrier, the domain reduce, the gateway's @p world all-reduce and its
     *  publication across domains, and a closing barrier before any member rewrites its window. */
    [[nodiscard]] expected_t world_sum_(std::size_t rank, std::size_t begin, infiniband_all_reduce<f32_t> &world,
                                        f32r1_span_t destination, gpu_stream_t &stream) noexcept;

    /** Rank @p rank 's world Gram of @p pooled , its factorization and the coding rate. */
    [[nodiscard]] expected_t coding_rate_(std::size_t rank, bf16r2_view_t pooled, gpu_stream_t &stream) noexcept;

    /** The baked geometry and weights. */
    pooled_spectrum_config_t config_ {};

    /** One scratch per local rank. */
    per_gpu_array<spectrum_rank_t> ranks_ {};

    /** The main-owned level-1 surface, borrowed; its replicas carry the Gram and the energy. */
    multimem_domain_t *domain_ = nullptr;

    /** This consumer's window of the domain payload: the Gram, then the energy. */
    multimem_window_t window_ {};

    /** The gateway's cross-domain sums of the Gram and of the energy, each sized to its part. */
    infiniband_all_reduce<f32_t> gram_reduce_ {}, energy_reduce_ {};

    /** The NumKong kernels of the arenas' device class. */
    gpu_numerics_t numerics_ {};

    /** @b [1] pinned staging the gate reads the world energy through. */
    f32r1_tensor_pinned_t energy_staging_ {};

    /** The last admitted step's gradient L2 norm, zero before the first. */
    f32_t admitted_gradient_l2_ = 0.0f;
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::train
