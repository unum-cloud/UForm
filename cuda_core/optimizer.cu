/**
 *  @brief Device kernels and heavy method bodies of the Adafactor optimizer declared in @ref optimizer.cuh
 *      — the translation unit compiled into the `uform` static library beside @ref multimodal_batch.cu and
 *      @ref contrastive_segments.cu .
 *  @file cuda_core/optimizer.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include <cmath> // `sqrtf`, `rsqrtf`

#include <cuda.h> // `cuMemsetD8`, `cuEventRecord`, `cuStreamWaitEvent`

#include "cuda_core/optimizer.hpp"

using namespace unum::uform::core;

/** @brief Read one gradient scalar at @p row , @p column : the reduced Float32 world sum when present —
 *      through a rank-2 view over this state's block of the packed world tape — the local BFloat16 buffer
 *      otherwise, read straight through its rank-2 handle. */
[[nodiscard]] __device__ inline f32_t adafactor_gradient_(adafactor_state_t const &state, f32r1_view_t reduced,
                                                          u32_t row, u32_t column) noexcept {
    if (!reduced.empty()) {
        std::size_t const rows = state.weights.weights_grad.extent(0), cols = state.weights.weights_grad.extent(1);
        f32r2_view_t const block {reduced.data() + state.gradient_offset, rows, cols};
        return block[row, column];
    }
    return __bfloat162float(state.weights.weights_grad[row, column]);
}

/**
 *  @brief Update one state's row and column moments — a CTA per row for the row means, then the columns
 *      strided serially over the rows; both fixed-order.
 *
 *  __grid_size__ @c [rows + ceil(cols/block),1,1] ; the first @c rows CTAs own rows, the rest stripe
 *  columns; no dynamic shared memory.
 */
static __global__ void adafactor_moments_kernel_(adafactor_state_t state, f32r1_span_t moments,
                                                 f32r1_view_t reduced_gradients, f32_t beta2, f32_t epsilon1) {
    u32_t const rows = static_cast<u32_t>(state.weights.weights.extent(0));
    u32_t const cols = static_cast<u32_t>(state.weights.weights.extent(1));
    if (blockIdx.x < rows) {
        u32_t const row = blockIdx.x;
        f32_t thread_partial = 0.0f;
        for (u32_t column = threadIdx.x; column < cols; column += blockDim.x) {
            f32_t const gradient = adafactor_gradient_(state, reduced_gradients, row, column);
            thread_partial += gradient * gradient;
        }
        f32_t const mean = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial) / f32_t(cols);
        if (threadIdx.x == 0u)
            moments[state.row_moment_offset + row] = beta2 * moments[state.row_moment_offset + row] +
                                                     (1.0f - beta2) * (mean + epsilon1);
    }
    else {
        u32_t const column = (blockIdx.x - rows) * blockDim.x + threadIdx.x;
        if (column >= cols) return;
        f32_t total = 0.0f;
        for (u32_t row = 0; row != rows; ++row) {
            f32_t const gradient = adafactor_gradient_(state, reduced_gradients, row, column);
            total += gradient * gradient;
        }
        f32_t const mean = total / f32_t(rows);
        moments[state.column_moment_offset + column] = beta2 * moments[state.column_moment_offset + column] +
                                                       (1.0f - beta2) * (mean + epsilon1);
    }
}

/**
 *  @brief One CTA: fold the state's row-moment mean and the update's squared RMS into the two per-state
 *      scalars the apply pass reads — the whole state walked serially per thread stripe, fixed order.
 *
 *  __grid_size__ @c [1,1,1] ; no dynamic shared memory.
 */
static __global__ void adafactor_scale_kernel_(adafactor_state_t state, f32r1_span_t moments,
                                               f32r1_view_t reduced_gradients, f32r2_span_t state_scales,
                                               u32_t state_index, f32_t clip_threshold) {
    u32_t const rows = static_cast<u32_t>(state.weights.weights.extent(0));
    u32_t const cols = static_cast<u32_t>(state.weights.weights.extent(1));
    f32_t thread_row_sum = 0.0f;
    for (u32_t row = threadIdx.x; row < rows; row += blockDim.x)
        thread_row_sum += moments[state.row_moment_offset + row];
    f32_t const row_mean = block_reduce_broadcast<kernel_block_size_default_k>(thread_row_sum) / f32_t(rows);
    f32_t thread_update_energy = 0.0f;
    std::size_t const scalars = static_cast<std::size_t>(rows) * cols;
    for (std::size_t index = threadIdx.x; index < scalars; index += blockDim.x) {
        u32_t const row = static_cast<u32_t>(index / cols), column = static_cast<u32_t>(index % cols);
        f32_t const variance = moments[state.row_moment_offset + row] * moments[state.column_moment_offset + column] /
                               row_mean;
        f32_t const gradient = adafactor_gradient_(state, reduced_gradients, row, column);
        f32_t const update = gradient * rsqrtf(variance);
        thread_update_energy += update * update;
    }
    f32_t const update_rms = sqrtf(block_reduce_broadcast<kernel_block_size_default_k>(thread_update_energy) /
                                   f32_t(scalars));
    if (threadIdx.x == 0u) {
        state_scales[state_index, 0] = row_mean;
        state_scales[state_index, 1] = 1.0f / fmaxf(1.0f, update_rms / clip_threshold);
    }
}

/**
 *  @brief Apply the clipped update @c W -= @c lr·scale·G/√v , through the Float32 master for vector
 *      weights (the norm scales) and straight into BFloat16 otherwise.
 *
 *  __grid_size__ grid-stride over the state's scalars; no dynamic shared memory.
 */
static __global__ void adafactor_apply_kernel_(adafactor_state_t state, f32r1_span_t moments,
                                               f32r1_view_t reduced_gradients, f32r2_view_t state_scales,
                                               u32_t state_index, f32r1_span_t masters, f32_t learning_rate) {
    u32_t const rows = static_cast<u32_t>(state.weights.weights.extent(0));
    u32_t const cols = static_cast<u32_t>(state.weights.weights.extent(1));
    f32_t const row_mean = state_scales[state_index, 0];
    f32_t const clip_scale = state_scales[state_index, 1];
    std::size_t const scalars = static_cast<std::size_t>(rows) * cols;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < scalars;
         index += stride) {
        u32_t const row = static_cast<u32_t>(index / cols), column = static_cast<u32_t>(index % cols);
        f32_t const variance = moments[state.row_moment_offset + row] * moments[state.column_moment_offset + column] /
                               row_mean;
        f32_t const gradient = adafactor_gradient_(state, reduced_gradients, row, column);
        f32_t const step = learning_rate * clip_scale * gradient * rsqrtf(variance);
        if (rows == 1u) {
            f32_t const updated = masters[state.master_offset + index] - step;
            masters[state.master_offset + index] = updated;
            state.weights.weights[row, column] = __float2bfloat16_rn(updated);
        }
        else
            state.weights.weights[row, column] = __float2bfloat16_rn(
                __bfloat162float(state.weights.weights[row, column]) - step);
    }
}

/** @brief Cast one BFloat16 gradient buffer into its window of the Float32 world-gradient surface.
 *  __grid_size__ grid-stride; no dynamic shared memory. */
static __global__ void adafactor_stage_gradient_kernel_(bf16r1_span_t gradient, f32r1_span_t destination) {
    std::size_t const total = gradient.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride)
        destination[index] = __bfloat162float(gradient[index]);
}

/** @brief Materialise the deterministic domain-reduced world gradient into this rank's own buffer —
 *      one switch-fused load per element from this consumer's window at @p shard_begin , or the peer
 *      tier's fixed-order serial fold.
 *  __grid_size__ grid-stride; no dynamic shared memory. */
static __global__ void adafactor_reduce_gradient_kernel_(multimem_member_t member, f32r1_span_t destination,
                                                         std::size_t shard_begin) {
    std::size_t const total = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        if (member.multimem_payload != nullptr)
            destination[index] = multimem_load_reduce_add_(member.multimem_payload + shard_begin + index);
        else {
            f32_t total_value = 0.0f;
            for (u32_t peer = 0; peer != member.member_count; ++peer) {
                f32_t const *peer_payload = member.peer_payload(peer);
                total_value = __fadd_rn(total_value, peer_payload[shard_begin + index]);
            }
            destination[index] = total_value;
        }
    }
}

/** @brief Seed the Float32 masters of the vector weights from their live BFloat16 values.
 *  __grid_size__ grid-stride; no dynamic shared memory. */
static __global__ void adafactor_seed_master_kernel_(bf16r1_span_t weights, f32r1_span_t master) {
    std::size_t const total = weights.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride)
        master[index] = __bfloat162float(weights[index]);
}

expected<adafactor_t> adafactor_t::make(gpu_arena_t &arena, std::span<differentiable_weights_t const> weights,
                                        adafactor_config_t config, gpu_stream_t &stream) noexcept {
    using result_t = expected<adafactor_t>;
    std::size_t const count = weights.size();
    if (count == 0) return result_t::fail(status_t::invalid_input_k, "an optimizer needs weights");
    adafactor_t optimizer;
    optimizer.config_ = config;
    if (optimizer.states_.try_resize(count) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "state table allocation failed ({})", count);
    std::size_t moment_scalars = 0, master_scalars = 0, gradient_scalars = 0;
    for (std::size_t index = 0; index != count; ++index) {
        differentiable_weights_t const &matrix = weights[index];
        std::size_t const rows = matrix.weights.extent(0), cols = matrix.weights.extent(1);
        if (rows == 0 || cols == 0 || matrix.weights_grad.extent(0) != rows || matrix.weights_grad.extent(1) != cols)
            return result_t::fail(status_t::shape_unsupported_k, "weights {} [{}x{}] disagree with grads [{}x{}]",
                                  index, rows, cols, matrix.weights_grad.extent(0), matrix.weights_grad.extent(1));
        adafactor_state_t &state = optimizer.states_[index];
        state.weights = matrix;
        state.row_moment_offset = moment_scalars;
        moment_scalars += rows;
        state.column_moment_offset = moment_scalars;
        moment_scalars += cols;
        state.gradient_offset = gradient_scalars;
        gradient_scalars += rows * cols;
        if (rows == 1u) {
            state.master_offset = master_scalars;
            master_scalars += cols;
        }
    }
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    optimizer.moments_ = f32r1_tensor_arena_t::try_empty({moment_scalars}, f32_alloc);
    optimizer.masters_ = f32r1_tensor_arena_t::try_empty({master_scalars != 0 ? master_scalars : 1}, f32_alloc);
    optimizer.state_scales_ = f32r2_tensor_arena_t::try_empty({count, 2}, f32_alloc);
    if (!optimizer.moments_ || !optimizer.masters_ || !optimizer.state_scales_)
        return result_t::fail(status_t::bad_alloc_k, "optimizer state allocation failed ({} moments)", moment_scalars);
    if (CUresult result = cuMemsetD8(reinterpret_cast<CUdeviceptr>(optimizer.moments_.data()), 0,
                                     moment_scalars * sizeof(f32_t));
        result != CUDA_SUCCESS)
        return result_t::fail(result, "moment reset failed");
    for (std::size_t index = 0; index != count; ++index) {
        adafactor_state_t &state = optimizer.states_[index];
        if (state.weights.weights.extent(0) != 1u) continue;
        std::size_t const cols = state.weights.weights.extent(1);
        f32r1_span_t const master {optimizer.masters_.data() + state.master_offset, cols};
        if (expected_t seeded = launch_kernel(
                reinterpret_cast<void const *>(&adafactor_seed_master_kernel_), dim3 {grid_size_for(cols), 1u, 1u},
                dim3 {kernel_block_size_default_k, 1u, 1u}, stream, state.weights.weights, master);
            !seeded)
            return result_t::fail(std::move(seeded));
    }
    return result_t {std::move(optimizer)};
}

expected_t adafactor_t::step(f32_t learning_rate, gpu_stream_t &stream, f32r1_view_t reduced_gradients) noexcept {
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    for (std::size_t index = 0; index != states_.size(); ++index) {
        adafactor_state_t const &state = states_[index];
        u32_t const rows = static_cast<u32_t>(state.weights.weights.extent(0));
        std::size_t const cols = state.weights.weights.extent(1);
        u32_t const column_blocks = static_cast<u32_t>(divide_round_up(cols, kernel_block_size_default_k));
        expected_t staged = launch_kernel(reinterpret_cast<void const *>(&adafactor_moments_kernel_),
                                          dim3 {rows + column_blocks, 1u, 1u}, block_shape, stream, state,
                                          moments_.span(), reduced_gradients, config_.beta2, config_.epsilon1);
        if (staged)
            staged = launch_kernel(reinterpret_cast<void const *>(&adafactor_scale_kernel_), dim3 {1u, 1u, 1u},
                                   block_shape, stream, state, moments_.span(), reduced_gradients, state_scales_.span(),
                                   static_cast<u32_t>(index), config_.clip_threshold);
        if (staged)
            staged = launch_kernel(reinterpret_cast<void const *>(&adafactor_apply_kernel_),
                                   dim3 {grid_size_for(rows * cols), 1u, 1u}, block_shape, stream, state,
                                   moments_.span(), reduced_gradients, state_scales_.view(), static_cast<u32_t>(index),
                                   masters_.span(), learning_rate);
        if (!staged) return staged;
    }
    return {};
}

expected<distributed_adafactor_t> distributed_adafactor_t::make(
    multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
    std::span<differentiable_weights_t const> weights, std::size_t weights_per_rank, adafactor_config_t config,
    std::span<gpu_stream_t> streams, f64_t timeout_seconds) noexcept {
    using result_t = expected<distributed_adafactor_t>;
    std::size_t const gpus = arenas.size();
    u32_t const domains = colocated.multimem_domain_count();
    if (gpus == 0 || streams.size() != gpus || domain.member_count() != gpus ||
        weights.size() != gpus * weights_per_rank)
        return result_t::fail(status_t::invalid_input_k,
                              "every rank needs an arena, stream, domain member, and weights table ({} ranks)", gpus);
    distributed_adafactor_t optimizer;
    optimizer.domain_ = &domain;
    optimizer.colocated_ = &colocated;
    if (optimizer.ranks_.try_resize(gpus) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "rank table allocation failed");
    expected<multimem_window_t> window = domain.reserve(payload_scalars_needed(weights.subspan(0, weights_per_rank)));
    if (!window) return result_t::fail(std::move(window));
    optimizer.window_ = window.value_;

    for (std::size_t rank = 0; rank != gpus; ++rank) {
        switch_to_gpu_t gpu_scope {streams[rank]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed (rank={})", rank);
        expected<adafactor_t> state = adafactor_t::make(
            arenas[rank], weights.subspan(rank * weights_per_rank, weights_per_rank), config, streams[rank]);
        if (!state) return result_t::fail(std::move(state));
        optimizer.ranks_[rank].state = std::move(state.value_);
        if (domains == 1u) {
            gpu_arena_alloc<f32_t> f32_alloc {arenas[rank]};
            std::size_t const scalars = optimizer.ranks_[rank].state.gradient_scalars();
            optimizer.ranks_[rank].reduced = f32r1_tensor_arena_t::try_empty({scalars}, f32_alloc);
            if (!optimizer.ranks_[rank].reduced)
                return result_t::fail(status_t::bad_alloc_k, "reduced-gradient allocation failed (rank={})", rank);
        }
    }

    // The gateway's cross-domain member and the world-ready ordering token; every member must be
    // able to peer-read the gateway's world sum.
    if (domains > 1u)
        for (std::size_t rank = 0; rank != gpus; ++rank)
            if (!arenas[0].is_visible_to(static_cast<int>(rank)))
                return result_t::fail(status_t::invalid_input_k,
                                      "rank 0's arena is not visible to device {} — " //
                                      "construct it with `visible_to`",
                                      rank);
    {
        switch_to_gpu_t gpu_scope {streams[0]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::cuda_error_k, "context entry failed (rank=0)");
        std::size_t const scalars = optimizer.ranks_[0].state.gradient_scalars();
        expected<infiniband_member_t> world_member = colocated.wire_allreduce(
            0, arenas[0], streams[0], scalars * sizeof(f32_t), "adafactor.reduce", timeout_seconds);
        if (!world_member) return result_t::fail(std::move(world_member));
        optimizer.world_member_ = std::move(world_member.value_);
        expected<gpu_event_t> event = gpu_event_t::make();
        if (!event) return result_t::fail(std::move(event));
        optimizer.world_ready_ = std::move(event.value_);
    }
    return result_t {std::move(optimizer)};
}

expected_t distributed_adafactor_t::step(std::size_t rank, f32_t learning_rate, gpu_stream_t &stream) noexcept {
    std::size_t const gpus = ranks_.size();
    if (rank >= gpus) return expected_t::fail(status_t::invalid_input_k, "rank {} of {}", rank, gpus);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed ({})", rank);
    // The shared phase rides the lead rank's slice — see the ordering contract.
    if (rank == 0)
        if (expected_t advanced = domain_->advance(stream); !advanced) return advanced;
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    expected_t staged {};
    f32_t *const window_base = domain_->payload(rank).data() + window_.offset;
    nk::tensor_view<adafactor_state_t, 1> const states = ranks_[rank].state.states();
    for (std::size_t index = 0; index != states.extent(0) && staged; ++index) {
        adafactor_state_t const &state = states[index];
        std::size_t const scalars = state.weights.weights.extent(0) * state.weights.weights.extent(1);
        bf16r1_span_t const gradient {const_cast<bf16_t *>(state.weights.weights_grad.data()), scalars};
        f32r1_span_t const window_block {window_base + state.gradient_offset, scalars};
        staged = launch_kernel(reinterpret_cast<void const *>(&adafactor_stage_gradient_kernel_),
                               dim3 {grid_size_for(scalars), 1u, 1u}, block_shape, stream, gradient, window_block);
    }
    if (staged) staged = domain_->barrier(rank, stream);
    if (!staged) return staged;
    std::size_t const scalars = ranks_[rank].state.gradient_scalars();
    if (world_domains() == 1u) {
        // Materialise the domain sum locally; the world IS the domain.
        staged = launch_kernel(reinterpret_cast<void const *>(&adafactor_reduce_gradient_kernel_),
                               dim3 {grid_size_for(scalars), 1u, 1u}, block_shape, stream, domain_->member(rank),
                               ranks_[rank].reduced.span(), window_.offset);
        if (staged) staged = ranks_[rank].state.step(learning_rate, stream, ranks_[rank].reduced.view());
        return staged;
    }
    return world_gradient_step_(rank, learning_rate, stream);
}

expected_t distributed_adafactor_t::step(f32_t learning_rate, std::span<gpu_stream_t> streams) noexcept {
    std::size_t const gpus = ranks_.size();
    if (streams.size() != gpus)
        return expected_t::fail(status_t::invalid_input_k, "every rank needs its stream ({} ranks)", gpus);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        if (expected_t staged = step(rank, learning_rate, streams[rank]); !staged) return staged;
    return {};
}

expected_t distributed_adafactor_t::world_gradient_step_(std::size_t rank, f32_t learning_rate,
                                                         gpu_stream_t &stream) noexcept {
    std::size_t const scalars = ranks_[rank].state.gradient_scalars();
    f32_t *const world_gradient = reinterpret_cast<f32_t *>(world_member_.payload().data());
    expected_t staged {};
    if (rank == 0) {
        staged = launch_kernel(reinterpret_cast<void const *>(&adafactor_reduce_gradient_kernel_),
                               dim3 {grid_size_for(scalars), 1u, 1u}, dim3 {kernel_block_size_default_k, 1u, 1u},
                               stream, domain_->member(rank), f32r1_span_t {world_gradient, scalars}, window_.offset);
        if (staged) staged = world_member_.advance(stream);
        if (staged) staged = world_member_.allreduce_sum(stream);
        if (staged)
            if (CUresult result = cuEventRecord(world_ready_.get(), stream.get()); result != CUDA_SUCCESS)
                staged = expected_t::fail(result, "world-gradient event record failed");
    }
    else if (CUresult result = cuStreamWaitEvent(stream.get(), world_ready_.get(), 0); result != CUDA_SUCCESS)
        staged = expected_t::fail(result, "world-gradient wait failed (rank={})", rank);
    if (staged) staged = ranks_[rank].state.step(learning_rate, stream, f32r1_view_t {world_gradient, scalars});
    return staged;
}
