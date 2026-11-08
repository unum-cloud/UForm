/**
 *  @file src/train/pooled_spectrum.cu
 *  @author Ash Vardanian
 *  @date October 27, 2026
 *  @brief Device kernels and method bodies of the pooled-code regularizers and the gradient spike
 *      gate declared in @ref pooled_spectrum.cuh .
 */
#include <cmath> // `std::sqrt`, `std::log`, `std::isfinite`

#include "uform/train/pooled_spectrum.cuh"

using namespace unum::uform::train;
using namespace unum::uform::utilities;

/** The power iteration's one CTA, the widest a launch takes, since it walks every row
 *  and channel alone. */
static constexpr unsigned spectrum_power_block_size_k = 1024u;

/** Rows one CTA of a warp-per-row kernel covers. */
static constexpr unsigned spectrum_rows_per_block_k = kernel_block_size_default_k / warp_size_k;

/** The @c [32,8] thread shape of the tile kernels. */
static constexpr dim3 spectrum_tile_block_k {warp_size_k, kernel_block_size_default_k / warp_size_k, 1u};

/**
 *  @brief The inverse L2 norm of every pooled row, zero for an all-zero row, which
 *      then contributes nothing.
 *
 *  __grid_size__ one CTA per @ref spectrum_rows_per_block_k rows, a warp per row; no
 *  dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_inverse_norms_kernel_(bf16r2_view_t pooled, f32r1_span_t inverse_norms) {
    std::size_t const row = std::size_t {blockIdx.x} * spectrum_rows_per_block_k + threadIdx.x / warp_size_k;
    unsigned const lane = threadIdx.x % warp_size_k;
    if (row >= pooled.extent(0)) return;
    f32_t energy = 0.0f;
    for (std::size_t channel = lane; channel < pooled.extent(1); channel += warp_size_k) {
        f32_t const value = __bfloat162float(pooled[row, channel]);
        energy = __fmaf_rn(value, value, energy);
    }
    energy = warp_reduce_broadcast(energy);
    if (lane == 0u) inverse_norms[row] = energy > 0.0f ? 1.0f / std::sqrt(energy) : 0.0f;
}

/**
 *  @brief Write the unit rows transposed and narrowed, @c transposed[channel,row]=z[row,channel] ,
 *      through a shared tile so both sides stay coalesced; the padding columns past the rows keep
 *      the zeros @c make wrote.
 *
 *  __grid_size__ @c [rows/32,hidden/32] tiles of @ref spectrum_tile_block_k ; no
 *  dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_transpose_kernel_(bf16r2_view_t pooled, f32r1_view_t inverse_norms, bf16r2_span_t transposed) {
    __shared__ f32_t tile[warp_size_k][warp_size_k + 1];
    std::size_t const rows = pooled.extent(0);
    std::size_t const row_begin = std::size_t {blockIdx.x} * warp_size_k;
    std::size_t const channel_begin = std::size_t {blockIdx.y} * warp_size_k;
    for (unsigned offset = threadIdx.y; offset < warp_size_k; offset += blockDim.y)
        if (std::size_t const row = row_begin + offset; row < rows)
            tile[offset][threadIdx.x] = __bfloat162float(pooled[row, channel_begin + threadIdx.x]) * inverse_norms[row];
    __syncthreads();
    for (unsigned offset = threadIdx.y; offset < warp_size_k; offset += blockDim.y)
        if (std::size_t const row = row_begin + threadIdx.x; row < rows)
            transposed[channel_begin + offset, row] = __float2bfloat16_rn(tile[threadIdx.x][offset]);
}

/**
 *  @brief Overwrite the lower triangle of @p factor with @c A=I+α · G , reading @c G off the upper
 *      triangle NumKong's symmetric kernel fills.
 *
 *  __grid_size__ grid-stride over the @c [hidden,hidden] cells; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_assemble_kernel_(f32r2_span_t factor, f32_t alpha) {
    std::size_t const hidden = factor.extent(0), cells = hidden * hidden;
    std::size_t const stride = std::size_t {gridDim.x} * blockDim.x;
    for (std::size_t cell = std::size_t {blockIdx.x} * blockDim.x + threadIdx.x; cell < cells; cell += stride) {
        std::size_t const row = cell / hidden, column = cell % hidden;
        if (column <= row) factor[row, column] = __fmaf_rn(alpha, factor[column, row], row == column ? 1.0f : 0.0f);
    }
}

/**
 *  @brief One block column of the Cholesky factorization: every CTA factors the diagonal tile in
 *      shared memory, CTA 0 writes it back, and CTA @c t solves the @c t -th tile below it.
 *
 *  Each lane owns one row of a tile and walks it serially, so the bits never depend on scheduling.
 *  __grid_size__ @c [tiles-block] CTAs of one warp; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(warp_size_k, 1)
    spectrum_cholesky_panel_kernel_(f32r2_span_t factor, u32_t block) {
    __shared__ f32_t diagonal[warp_size_k][warp_size_k + 1], panel[warp_size_k][warp_size_k + 1];
    unsigned const lane = threadIdx.x;
    std::size_t const begin = std::size_t {block} * warp_size_k;
    for (unsigned column = 0; column <= lane; ++column) diagonal[lane][column] = factor[begin + lane, begin + column];
    __syncwarp();
    for (unsigned pivot = 0; pivot != warp_size_k; ++pivot) {
        if (lane == pivot) diagonal[pivot][pivot] = std::sqrt(diagonal[pivot][pivot]);
        __syncwarp();
        if (lane > pivot) diagonal[lane][pivot] /= diagonal[pivot][pivot];
        __syncwarp();
        if (lane > pivot)
            for (unsigned column = pivot + 1u; column <= lane; ++column)
                diagonal[lane][column] = __fmaf_rn(-diagonal[lane][pivot], diagonal[column][pivot],
                                                   diagonal[lane][column]);
        __syncwarp();
    }
    if (blockIdx.x == 0u) {
        for (unsigned column = 0; column <= lane; ++column)
            factor[begin + lane, begin + column] = diagonal[lane][column];
        return;
    }
    std::size_t const row = begin + std::size_t {blockIdx.x} * warp_size_k + lane;
    for (unsigned column = 0; column != warp_size_k; ++column) {
        f32_t value = factor[row, begin + column];
        for (unsigned previous = 0; previous != column; ++previous)
            value = __fmaf_rn(-panel[lane][previous], diagonal[column][previous], value);
        panel[lane][column] = value / diagonal[column][column];
    }
    for (unsigned column = 0; column != warp_size_k; ++column) factor[row, begin + column] = panel[lane][column];
}

/**
 *  @brief The trailing update after block column @p block : every lower tile right of it takes
 *      @c A[I,J]-=L[I,block] · L[J,block]ᵀ , each cell summing its 32 products in order.
 *
 *  __grid_size__ @c [m,m] with @c m the tiles past the block, the upper tiles exiting at once;
 *  @ref spectrum_tile_block_k threads; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_cholesky_update_kernel_(f32r2_span_t factor, u32_t block) {
    if (blockIdx.x > blockIdx.y) return;
    __shared__ f32_t row_panel[warp_size_k][warp_size_k + 1], column_panel[warp_size_k][warp_size_k + 1];
    std::size_t const begin = std::size_t {block} * warp_size_k;
    std::size_t const row_begin = begin + (std::size_t {blockIdx.y} + 1u) * warp_size_k;
    std::size_t const column_begin = begin + (std::size_t {blockIdx.x} + 1u) * warp_size_k;
    for (unsigned offset = threadIdx.y; offset < warp_size_k; offset += blockDim.y) {
        row_panel[offset][threadIdx.x] = factor[row_begin + offset, begin + threadIdx.x];
        column_panel[offset][threadIdx.x] = factor[column_begin + offset, begin + threadIdx.x];
    }
    __syncthreads();
    for (unsigned offset = threadIdx.y; offset < warp_size_k; offset += blockDim.y) {
        std::size_t const row = row_begin + offset, column = column_begin + threadIdx.x;
        if (column > row) continue;
        f32_t product = 0.0f;
        for (unsigned inner = 0; inner != warp_size_k; ++inner)
            product = __fmaf_rn(row_panel[offset][inner], column_panel[threadIdx.x][inner], product);
        factor[row, column] -= product;
    }
}

/**
 *  @brief Copy @c L into the strict upper triangle as @c Lᵀ , so the back substitution reads rows:
 *      tile @c (I,J) of the lower triangle lands transposed on tile @c (J,I) .
 *
 *  __grid_size__ @c [tiles,tiles] , the upper tiles exiting at once; @ref spectrum_tile_block_k
 *  threads; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1) spectrum_mirror_kernel_(f32r2_span_t factor) {
    if (blockIdx.x > blockIdx.y) return;
    __shared__ f32_t tile[warp_size_k][warp_size_k + 1];
    std::size_t const row_begin = std::size_t {blockIdx.y} * warp_size_k;
    std::size_t const column_begin = std::size_t {blockIdx.x} * warp_size_k;
    for (unsigned offset = threadIdx.y; offset < warp_size_k; offset += blockDim.y)
        tile[offset][threadIdx.x] = factor[row_begin + offset, column_begin + threadIdx.x];
    __syncthreads();
    for (unsigned offset = threadIdx.y; offset < warp_size_k; offset += blockDim.y) {
        std::size_t const row = column_begin + offset, column = row_begin + threadIdx.x;
        if (column > row) factor[row, column] = tile[threadIdx.x][offset];
    }
}

/**
 *  @brief The coding rate @c R=½ · logdet(A)=Σlog(L[i,i]) into @c terms[0] .
 *
 *  __grid_size__ @c [1,1,1] ; one CTA stripes the diagonal; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_log_determinant_kernel_(f32r2_view_t factor, f32r1_span_t terms) {
    f32_t partial = 0.0f;
    for (std::size_t index = threadIdx.x; index < factor.extent(0); index += blockDim.x)
        partial += std::log(factor[index, index]);
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(partial);
    if (threadIdx.x == 0u) terms[0] = total;
}

/**
 *  @brief The power iteration on @c ZᵀZ through @c Z itself: from the uniform direction,
 *      @p iterations rounds of @c v←Zᵀ(Z · v)/‖Zᵀ(Z · v)‖ , then the projections @c Z · v and
 *      @c σ²=‖Z · v‖² into @c terms[1] .
 *
 *  __grid_size__ @c [1,1,1] ; a warp per row for @c Z · v and a thread per channel for @c Zᵀ · u ;
 *      no dynamic shared memory.
 */
static __global__ void __launch_bounds__(spectrum_power_block_size_k, 1)
    spectrum_power_iteration_kernel_(bf16r2_view_t pooled, f32r1_view_t inverse_norms, u32_t iterations,
                                     f32r1_span_t direction, f32r1_span_t projections, f32r1_span_t terms) {
    constexpr unsigned warps_k = spectrum_power_block_size_k / warp_size_k;
    std::size_t const rows = pooled.extent(0), hidden = pooled.extent(1);
    unsigned const lane = threadIdx.x % warp_size_k, warp = threadIdx.x / warp_size_k;
    f32_t const uniform = 1.0f / std::sqrt(static_cast<f32_t>(hidden));
    for (std::size_t channel = threadIdx.x; channel < hidden; channel += blockDim.x) direction[channel] = uniform;
    __syncthreads();
    for (u32_t iteration = 0;; ++iteration) {
        for (std::size_t row = warp; row < rows; row += warps_k) {
            f32_t partial = 0.0f;
            for (std::size_t channel = lane; channel < hidden; channel += warp_size_k)
                partial = __fmaf_rn(__bfloat162float(pooled[row, channel]), direction[channel], partial);
            f32_t const projection = warp_reduce_broadcast(partial) * inverse_norms[row];
            if (lane == 0u) projections[row] = projection;
        }
        __syncthreads();
        if (iteration == iterations) break;
        f32_t energy = 0.0f;
        for (std::size_t channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
            f32_t image = 0.0f;
            for (std::size_t row = 0; row != rows; ++row)
                image = __fmaf_rn(__bfloat162float(pooled[row, channel]) * inverse_norms[row], projections[row], image);
            direction[channel] = image;
            energy = __fmaf_rn(image, image, energy);
        }
        f32_t const norm = std::sqrt(block_reduce_broadcast<spectrum_power_block_size_k>(energy));
        f32_t const scale = norm > 0.0f ? 1.0f / norm : 0.0f;
        for (std::size_t channel = threadIdx.x; channel < hidden; channel += blockDim.x) direction[channel] *= scale;
        __syncthreads();
    }
    f32_t energy = 0.0f;
    for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x)
        energy = __fmaf_rn(projections[row], projections[row], energy);
    f32_t const top = block_reduce_broadcast<spectrum_power_block_size_k>(energy);
    if (threadIdx.x == 0u) terms[1] = top;
}

/**
 *  @brief Every row's force on its raw pooled row, added into @p pooled_grad : the unit-row
 *      gradient @c g=s_tcr · A⁻¹ · z+s_lip · (z · v) · v through the normalization's Jacobian,
 *      @c (g − z · (z · g))/‖x‖ .
 *
 *  @c A⁻¹ · z comes from two solves, @c L · y=z over the lower triangle of @p factor and @c Lᵀ ·
 *  x=y over its upper, each step one dot product with a row of the factor. Lane @c l owns the
 *  entries @c k≡l (mod 32) of its row's solve, so it reads only what it wrote, and the shuffle tree
 *  fixes the order of every sum. An empty @p factor or @p direction drops its term.
 *
 *  __grid_size__ one CTA per @ref spectrum_rows_per_block_k rows, a warp per row; no
 *  dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_gradient_kernel_(bf16r2_view_t pooled, f32r1_view_t inverse_norms, f32r2_view_t factor,
                              f32_t coding_rate_scale, f32r1_view_t direction, f32r1_view_t projections,
                              f32_t lipschitz_scale, f32r2_span_t solves, bf16r2_span_t pooled_grad) {
    std::size_t const row = std::size_t {blockIdx.x} * spectrum_rows_per_block_k + threadIdx.x / warp_size_k;
    unsigned const lane = threadIdx.x % warp_size_k;
    if (row >= pooled.extent(0)) return;
    std::size_t const hidden = pooled.extent(1);
    f32_t const inverse_norm = inverse_norms[row];
    bool const coding_rate = !factor.empty(), lipschitz = !direction.empty();
    if (coding_rate) {
        f32r1_span_t const solve = solves[row, nk::slice];
        for (std::size_t index = lane; index < hidden; index += warp_size_k)
            solve[index] = __bfloat162float(pooled[row, index]) * inverse_norm;
        for (std::size_t index = 0; index != hidden; ++index) {
            f32_t partial = 0.0f;
            for (std::size_t inner = lane; inner < index; inner += warp_size_k)
                partial = __fmaf_rn(factor[index, inner], solve[inner], partial);
            f32_t const sum = warp_reduce_broadcast(partial);
            if (index % warp_size_k == lane) solve[index] = (solve[index] - sum) / factor[index, index];
        }
        for (std::size_t index = hidden; index-- != 0;) {
            f32_t partial = 0.0f;
            std::size_t const first = index + 1u + (lane + warp_size_k - (index + 1u) % warp_size_k) % warp_size_k;
            for (std::size_t inner = first; inner < hidden; inner += warp_size_k)
                partial = __fmaf_rn(factor[index, inner], solve[inner], partial);
            f32_t const sum = warp_reduce_broadcast(partial);
            if (index % warp_size_k == lane) solve[index] = (solve[index] - sum) / factor[index, index];
        }
    }
    f32_t const lipschitz_coefficient = lipschitz ? lipschitz_scale * projections[row] : 0.0f;
    auto unit_gradient = [&](std::size_t index) noexcept {
        f32_t const shaped = coding_rate ? coding_rate_scale * solves[row, index] : 0.0f;
        return lipschitz ? __fmaf_rn(lipschitz_coefficient, direction[index], shaped) : shaped;
    };
    f32_t partial = 0.0f;
    for (std::size_t index = lane; index < hidden; index += warp_size_k)
        partial = __fmaf_rn(__bfloat162float(pooled[row, index]) * inverse_norm, unit_gradient(index), partial);
    f32_t const radial = warp_reduce_broadcast(partial);
    for (std::size_t index = lane; index < hidden; index += warp_size_k) {
        f32_t const unit = __bfloat162float(pooled[row, index]) * inverse_norm;
        f32_t const raw = (unit_gradient(index) - unit * radial) * inverse_norm;
        pooled_grad[row, index] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(pooled_grad[row, index]), raw));
    }
}

/**
 *  @brief Each CTA's squared sum over its stripe of one gradient matrix, into
 *      @c partials[matrix,block] .
 *
 *  __grid_size__ @c [blocks,matrices] ; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_energy_kernel_(nk::tensor_view<spectrum_gradient_range_t, 1> gradients, f32r2_span_t partials) {
    spectrum_gradient_range_t const matrix = gradients[blockIdx.y];
    std::size_t const stride = std::size_t {gridDim.x} * blockDim.x;
    f32_t energy = 0.0f;
    for (std::size_t index = std::size_t {blockIdx.x} * blockDim.x + threadIdx.x; index < matrix.count;
         index += stride) {
        f32_t const value = __bfloat162float(matrix.data[index]);
        energy = __fmaf_rn(value, value, energy);
    }
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(energy);
    if (threadIdx.x == 0u) partials[blockIdx.y, blockIdx.x] = total;
}

/**
 *  @brief Sum every partial of @ref spectrum_energy_kernel_ in a fixed order into @p energy .
 *
 *  __grid_size__ @c [1,1,1] ; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    spectrum_energy_fold_kernel_(f32r2_view_t partials, f32r1_span_t energy) {
    std::size_t const matrices = partials.extent(0), blocks = partials.extent(1);
    f32_t partial = 0.0f;
    for (std::size_t index = threadIdx.x; index < matrices * blocks; index += blockDim.x)
        partial += partials[index / blocks, index % blocks];
    f32_t const total = block_reduce_broadcast<kernel_block_size_default_k>(partial);
    if (threadIdx.x == 0u) energy[0] = total;
}

expected<distributed_pooled_spectrum_t> distributed_pooled_spectrum_t::make(
    multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
    std::span<gpu_stream_t> streams, std::span<differentiable_weights_t const> weights, std::size_t weights_per_rank,
    pooled_spectrum_config_t config, gpu_numerics_t const &numerics, f64_t timeout_seconds) noexcept {
    using result_t = expected<distributed_pooled_spectrum_t>;
    std::size_t const gpus = arenas.size();
    std::size_t const rows = config.rows_per_gpu, hidden = config.hidden;
    bool const coding_rate = config.coding_rate_weight > 0.0f, lipschitz = config.lipschitz_weight > 0.0f;
    bool const gated = config.spike_threshold > 0.0f;
    if (gpus == 0 || streams.size() != gpus || domain.local_member_count() != gpus)
        return result_t::fail(status_t::invalid_argument_k,
                              "every rank needs an arena, a stream and a local domain member ({} ranks, {} streams, " //
                              "{} local members)",
                              gpus, streams.size(), domain.local_member_count());
    if (!(config.coding_rate_weight >= 0.0f) || !(config.lipschitz_weight >= 0.0f) ||
        !(config.spike_threshold >= 0.0f) || !std::isfinite(config.coding_rate_weight) ||
        !std::isfinite(config.lipschitz_weight) || !std::isfinite(config.spike_threshold) ||
        (coding_rate && !(config.epsilon_squared > 0.0f)))
        return result_t::fail(status_t::invalid_argument_k,
                              "the weights and the threshold must be finite and non-negative, and ε² positive " //
                              "(λ_tcr={} ε²={} λ_lip={} threshold={})",
                              config.coding_rate_weight, config.epsilon_squared, config.lipschitz_weight,
                              config.spike_threshold);
    if ((coding_rate || lipschitz) &&
        (hidden == 0 || hidden % warp_size_k != 0 || rows == 0 || config.rows_per_world < rows))
        return result_t::fail(status_t::unsupported_k,
                              "the regularizers need whole warps of channels and the world's rows to hold the " //
                              "rank's (hidden={} rows={} world rows={})",
                              hidden, rows, config.rows_per_world);
    if (coding_rate && numerics.symmetric_bf16 == nullptr)
        return result_t::fail(status_t::unsupported_k, "the device class has no symmetric BFloat16 Gram kernel");
    if (gated && (weights_per_rank == 0 || weights.size() != gpus * weights_per_rank))
        return result_t::fail(status_t::invalid_argument_k,
                              "the gate measures {} matrices on each of {} ranks, the table holds {}", weights_per_rank,
                              gpus, weights.size());

    distributed_pooled_spectrum_t spectrum;
    spectrum.config_ = config;
    spectrum.domain_ = &domain;
    spectrum.numerics_ = numerics;
    if (spectrum.ranks_.resize(gpus) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "rank table allocation failed ({} ranks)", gpus);
    if (std::size_t const scalars = payload_scalars_needed(config); scalars != 0) {
        expected<multimem_window_t> window = domain.reserve(scalars);
        if (!window) return result_t::fail(std::move(window));
        spectrum.window_ = window.value_;
    }

    // NumKong's 16-byte row strides: the transposed rows pad to 8 BFloat16 columns.
    std::size_t const rows_padded = round_up_to_multiple(rows, std::size_t {8});
    std::size_t const energy_blocks = gated ? divide_round_up(std::size_t {kernel_grid_cap_k}, weights_per_rank) : 0u;
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        switch_to_gpu_t gpu_scope {streams[rank]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed (rank={})", rank);
        spectrum_rank_t &state = spectrum.ranks_[rank];
        gpu_arena_alloc<f32_t> f32_alloc {arenas[rank]};
        gpu_arena_alloc<bf16_t> bf16_alloc {arenas[rank]};
        state.terms = f32r1_tensor_arena_t::uninitialized({2}, f32_alloc).value;
        bool allocated = static_cast<bool>(state.terms);
        if (coding_rate || lipschitz) {
            state.inverse_norms = f32r1_tensor_arena_t::uninitialized({rows}, f32_alloc).value;
            allocated = allocated && state.inverse_norms;
        }
        if (coding_rate) {
            state.transposed = bf16r2_tensor_arena_t::uninitialized({hidden, rows_padded}, bf16_alloc).value;
            state.factor = f32r2_tensor_arena_t::uninitialized({hidden, hidden}, f32_alloc).value;
            state.solves = f32r2_tensor_arena_t::uninitialized({rows, hidden}, f32_alloc).value;
            allocated = allocated && state.transposed && state.factor && state.solves;
        }
        if (lipschitz) {
            state.direction = f32r1_tensor_arena_t::uninitialized({hidden}, f32_alloc).value;
            state.projections = f32r1_tensor_arena_t::uninitialized({rows}, f32_alloc).value;
            allocated = allocated && state.direction && state.projections;
        }
        if (gated) {
            state.gradients =
                nk::tensor<spectrum_gradient_range_t, gpu_arena_alloc<spectrum_gradient_range_t>, 1>::uninitialized(
                    {weights_per_rank}, gpu_arena_alloc<spectrum_gradient_range_t> {arenas[rank]})
                    .value;
            state.energy_partials =
                f32r2_tensor_arena_t::uninitialized({weights_per_rank, energy_blocks}, f32_alloc).value;
            state.energy = f32r1_tensor_arena_t::uninitialized({1}, f32_alloc).value;
            allocated = allocated && state.gradients && state.energy_partials && state.energy;
        }
        if (!allocated)
            return result_t::fail(status_t::bad_alloc_k, "spectrum scratch allocation failed (rank={})", rank);
        expected_t staged = zero_async(state.terms.span(), streams[rank]);
        if (staged) staged = zero_async(state.transposed.span(), streams[rank]);
        if (staged && gated) {
            st::basic_vector<spectrum_gradient_range_t> ranges;
            if (st::failed(ranges.resize(weights_per_rank)))
                return result_t::fail(status_t::bad_alloc_k, "gradient table staging failed (rank={})", rank);
            std::span<differentiable_weights_t const> const rank_weights = weights.subspan(rank * weights_per_rank,
                                                                                           weights_per_rank);
            for (std::size_t matrix = 0; matrix != weights_per_rank; ++matrix) {
                bf16r2_view_t const gradient = rank_weights[matrix].weights_grad;
                if (!gradient.is_contiguous())
                    return result_t::fail(status_t::unsupported_k, "gradient matrix {} of rank {} is strided", matrix,
                                          rank);
                ranges[matrix].data = gradient.data();
                ranges[matrix].count = gradient.numel();
            }
            staged = copy_host_to_device<spectrum_gradient_range_t>(std::span<spectrum_gradient_range_t const> {ranges},
                                                                    state.gradients.span(), streams[rank]);
            if (staged) // the table is pageable and dies with this scope
                if (CUresult result = cuStreamSynchronize(streams[rank].get()); result != CUDA_SUCCESS)
                    staged = cuda_failure(result, "gradient table upload failed (rank={})", rank);
        }
        if (!staged) return result_t::fail(std::move(staged));
    }

    if (coding_rate || gated) {
        switch_to_gpu_t gpu_scope {streams[0]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed (rank=0)");
        if (coding_rate) {
            expected<infiniband_all_reduce<f32_t>> reduce = infiniband_all_reduce<f32_t>::make(
                colocated, 0, arenas[0], gram_scalars_(config), streams[0], timeout_seconds);
            if (!reduce) return result_t::fail(std::move(reduce));
            spectrum.gram_reduce_ = std::move(reduce.value_);
        }
        if (gated) {
            expected<infiniband_all_reduce<f32_t>> reduce = infiniband_all_reduce<f32_t>::make(
                colocated, 0, arenas[0], energy_scalars_(config), streams[0], timeout_seconds);
            if (!reduce) return result_t::fail(std::move(reduce));
            spectrum.energy_reduce_ = std::move(reduce.value_);
            spectrum.energy_staging_ = f32r1_tensor_pinned_t::uninitialized({1}, pinned_alloc<f32_t> {}).value;
            if (!spectrum.energy_staging_)
                return result_t::fail(status_t::bad_alloc_k, "gradient energy staging allocation failed");
        }
    }
    return result_t {std::move(spectrum)};
}

expected_t distributed_pooled_spectrum_t::world_sum_(std::size_t rank, std::size_t begin,
                                                     infiniband_all_reduce<f32_t> &world, f32r1_span_t destination,
                                                     gpu_stream_t &stream) noexcept {
    std::size_t const member = domain_->member_offset() + rank, scalars = destination.extent(0);
    expected_t summed = domain_->barrier(member, stream);
    if (world.identity()) {
        if (summed) summed = domain_->reduce_scatter(member, destination, begin, stream);
    }
    else {
        // Only the gateway crosses domains; the rest read its sum back out of its window.
        f32r1_span_t const gateway_payload = domain_->payload(0);
        f32r1_span_t const gateway_part = gateway_payload[nk::range(begin, begin + scalars), nk::slice];
        if (summed && member == 0) {
            f32r1_span_t const world_payload = world.payload();
            f32r1_span_t const shard = world_payload[nk::range(0, scalars), nk::slice];
            summed = domain_->reduce_scatter(member, shard, begin, stream);
            if (summed) summed = world(stream);
            if (summed) summed = copy_device_to_device(shard, gateway_part, stream);
        }
        if (summed) summed = domain_->barrier(member, stream);
        if (summed) summed = copy_device_to_device(gateway_part, destination, stream);
    }
    // Every member has read every peer's part before any member's next call writes its own again.
    if (summed) summed = domain_->barrier(member, stream);
    return summed;
}

expected_t distributed_pooled_spectrum_t::coding_rate_(std::size_t rank, bf16r2_view_t pooled,
                                                       gpu_stream_t &stream) noexcept {
    spectrum_rank_t &state = ranks_[rank];
    std::size_t const rows = config_.rows_per_gpu, hidden = config_.hidden, tiles = hidden / warp_size_k;
    std::size_t const rows_padded = state.transposed.extent(1);
    f32r1_span_t const payload = domain_->payload(domain_->member_offset() + rank);
    std::size_t const gram_end = window_.offset + gram_scalars_(config_);
    f32r1_span_t const gram = payload[nk::range(window_.offset, gram_end), nk::slice];
    f32r2_span_t const factor = state.factor.span();
    f32_t const alpha = static_cast<f32_t>(hidden) /
                        (static_cast<f32_t>(config_.rows_per_world) * config_.epsilon_squared);
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};

    expected_t staged = launch_kernel(
        &spectrum_transpose_kernel_,
        dim3 {static_cast<u32_t>(divide_round_up(rows, std::size_t {warp_size_k})), static_cast<u32_t>(tiles), 1u},
        spectrum_tile_block_k, stream, pooled, state.inverse_norms.view(), state.transposed.span());
    if (staged)
        staged = numkong_result(
            numerics_.symmetric_bf16(numkong_scalars(state.transposed.data()), hidden, rows_padded,
                                     rows_padded * sizeof(bf16_t), gram.data(), hidden * sizeof(f32_t), 0, hidden,
                                     nullptr, 0, stream.numkong()),
            "pooled Gram");
    if (staged) staged = world_sum_(rank, window_.offset, gram_reduce_, factor.flatten<1>(), stream);
    if (staged)
        staged = launch_kernel(&spectrum_assemble_kernel_, dim3 {grid_size_for(hidden * hidden), 1u, 1u}, block_shape,
                               stream, factor, alpha);
    for (u32_t block = 0; staged && block != tiles; ++block) {
        u32_t const trailing = static_cast<u32_t>(tiles - block - 1u);
        staged = launch_kernel(&spectrum_cholesky_panel_kernel_, dim3 {trailing + 1u, 1u, 1u},
                               dim3 {warp_size_k, 1u, 1u}, stream, factor, block);
        if (staged && trailing != 0u)
            staged = launch_kernel(&spectrum_cholesky_update_kernel_, dim3 {trailing, trailing, 1u},
                                   spectrum_tile_block_k, stream, factor, block);
    }
    if (staged)
        staged = launch_kernel(&spectrum_mirror_kernel_,
                               dim3 {static_cast<u32_t>(tiles), static_cast<u32_t>(tiles), 1u}, spectrum_tile_block_k,
                               stream, factor);
    if (staged)
        staged = launch_kernel(&spectrum_log_determinant_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream,
                               f32r2_view_t {factor}, state.terms.span());
    return staged;
}

expected_t distributed_pooled_spectrum_t::operator()(std::size_t rank, bf16r2_view_t pooled, bf16r2_span_t pooled_grad,
                                                     gpu_stream_t &stream) noexcept {
    std::size_t const rows = config_.rows_per_gpu, hidden = config_.hidden;
    bool const coding_rate = config_.coding_rate_weight > 0.0f, lipschitz = config_.lipschitz_weight > 0.0f;
    if (rank >= ranks_.size())
        return expected_t::fail(status_t::invalid_argument_k, "rank {} of {}", rank, ranks_.size());
    if (!coding_rate && !lipschitz) return {};
    if (pooled.extent(0) != rows || pooled.extent(1) != hidden || pooled_grad.extent(0) != rows ||
        pooled_grad.extent(1) != hidden)
        return expected_t::fail(status_t::unsupported_k, "pooled rows [{},{}] and gradient [{},{}] must be [{},{}]",
                                pooled.extent(0), pooled.extent(1), pooled_grad.extent(0), pooled_grad.extent(1), rows,
                                hidden);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::device_error_k, "context entry failed ({})", rank);

    spectrum_rank_t &state = ranks_[rank];
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const row_grid {static_cast<u32_t>(divide_round_up(rows, std::size_t {spectrum_rows_per_block_k})), 1u, 1u};
    expected_t staged = launch_kernel(&spectrum_inverse_norms_kernel_, row_grid, block_shape, stream, pooled,
                                      state.inverse_norms.span());
    if (staged && coding_rate) staged = coding_rate_(rank, pooled, stream);
    if (staged && lipschitz)
        staged = launch_kernel(&spectrum_power_iteration_kernel_, dim3 {1u, 1u, 1u},
                               dim3 {spectrum_power_block_size_k, 1u, 1u}, stream, pooled, state.inverse_norms.view(),
                               config_.power_iterations, state.direction.span(), state.projections.span(),
                               state.terms.span());
    f32_t const alpha = coding_rate ? static_cast<f32_t>(hidden) /
                                          (static_cast<f32_t>(config_.rows_per_world) * config_.epsilon_squared)
                                    : 0.0f;
    if (staged)
        staged = launch_kernel(&spectrum_gradient_kernel_, row_grid, block_shape, stream, pooled,
                               state.inverse_norms.view(), state.factor.view(), -config_.coding_rate_weight * alpha,
                               state.direction.view(), state.projections.view(), 2.0f * config_.lipschitz_weight,
                               state.solves.span(), pooled_grad);
    return staged;
}

expected_t distributed_pooled_spectrum_t::measure_gradients(std::size_t rank, gpu_stream_t &stream) noexcept {
    if (rank >= ranks_.size())
        return expected_t::fail(status_t::invalid_argument_k, "rank {} of {}", rank, ranks_.size());
    if (!(config_.spike_threshold > 0.0f))
        return expected_t::fail(status_t::invalid_argument_k, "the spike gate is off, so nothing measures gradients");
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::device_error_k, "context entry failed ({})", rank);
    spectrum_rank_t &state = ranks_[rank];
    std::size_t const begin = window_.offset + gram_scalars_(config_);
    f32r1_span_t const payload = domain_->payload(domain_->member_offset() + rank);
    f32r2_span_t const partials = state.energy_partials.span();
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    expected_t measured = launch_kernel(
        &spectrum_energy_kernel_,
        dim3 {static_cast<u32_t>(partials.extent(1)), static_cast<u32_t>(partials.extent(0)), 1u}, block_shape, stream,
        state.gradients.view(), partials);
    if (measured)
        measured = launch_kernel(&spectrum_energy_fold_kernel_, dim3 {1u, 1u, 1u}, block_shape, stream,
                                 f32r2_view_t {partials}, payload[nk::range(begin, begin + 1u), nk::slice]);
    if (measured) measured = world_sum_(rank, begin, energy_reduce_, state.energy.span(), stream);
    return measured;
}

expected_t distributed_pooled_spectrum_t::admit_gradients(gpu_stream_t &stream) noexcept {
    if (ranks_.size() == 0 || energy_staging_.empty())
        return expected_t::fail(status_t::invalid_argument_k, "the spike gate is off, so nothing admits gradients");
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::device_error_k, "context entry failed (rank=0)");
    if (expected_t copied = copy_device_to_host(ranks_[0].energy.view(), energy_staging_.span(), stream); !copied)
        return copied;
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "gradient energy download failed");
    f32_t const gradient_l2 = std::sqrt(energy_staging_[0]);
    if (!std::isfinite(gradient_l2) ||
        (admitted_gradient_l2_ > 0.0f && gradient_l2 > config_.spike_threshold * admitted_gradient_l2_))
        return expected_t::fail(status_t::internal_k,
                                "the gradient L2 norm {} spiked past {} times the last admitted {}", gradient_l2,
                                config_.spike_threshold, admitted_gradient_l2_);
    admitted_gradient_l2_ = gradient_l2;
    return {};
}
