/**
 *  @file src/train/optimizer.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *  @brief Device kernels and heavy method bodies of the Adafactor optimizer declared in
 *      @ref optimizer.hpp — the translation unit compiled into the @c uform_train static library
 *      beside @ref multimodal_batch.cu and @ref contrastive_segments.cu .
 */
#include <cmath> // `sqrtf`, `rsqrtf`, `expf`, `std::pow` — the decay schedule

#include <algorithm> // `std::min` — the decay ceiling

#include <cuda.h> // `cuMemsetD8`, `cuEventRecord`, `cuStreamWaitEvent`

#include "uform/train/optimizer.hpp"

using namespace unum::uform::train;
using namespace unum::uform::utilities;

/** Read one gradient scalar at @p row , @p column : the reduced Float32 world sum when present —
 *  through a rank-2 view over this state's block of the packed world tape — the local BFloat16 buffer
 *  otherwise, read straight through its rank-2 handle. */
[[nodiscard]] __device__ inline f32_t adafactor_gradient_(adafactor_state_t const &state, f32r1_view_t reduced,
                                                          u32_t row, u32_t column) noexcept {
    if (!reduced.empty()) {
        std::size_t const rows = state.weights_grad.extent(0), cols = state.weights_grad.extent(1);
        f32r2_view_t const block {reduced.data() + state.gradient_offset, rows, cols};
        return block[row, column];
    }
    return __bfloat162float(state.weights_grad[row, column]);
}

/**
 *  @brief Update one state's row and column moments — a CTA per warp-wide column tile, whose
 *      warps each fold an interleaved band of rows before warp 0 folds the bands in warp order,
 *      then a CTA per row for the row means; both fixed-order.
 *
 *  __grid_size__ @c [column_tiles+rows,1,1] , a tile per warp of columns; the first CTAs own column
 *  tiles, so their longer walks start first, and the rest own rows; no dynamic shared memory.
 */
static __global__ void __launch_bounds__(kernel_block_size_default_k, 1)
    adafactor_moments_kernel_(adafactor_state_t state, f32r1_span_t moments, f32r1_view_t reduced_gradients,
                              f32_t beta2, f32_t epsilon1) {
    constexpr unsigned warps_k = kernel_block_size_default_k / warp_size_k;
    __shared__ f32_t band_totals[warps_k][warp_size_k];
    u32_t const rows = static_cast<u32_t>(state.weights.extent(0));
    u32_t const cols = static_cast<u32_t>(state.weights.extent(1));
    u32_t const column_tiles = static_cast<u32_t>(divide_round_up(cols, warp_size_k));
    if (blockIdx.x < column_tiles) {
        unsigned const lane_index = threadIdx.x % warp_size_k, warp_index = threadIdx.x / warp_size_k;
        u32_t const column = blockIdx.x * warp_size_k + lane_index;
        f32_t band_total = 0.0f;
        if (column < cols) {
#pragma unroll 16 // keeps the band's loads in flight ahead of the ordered sum
            for (u32_t row = warp_index; row < rows; row += warps_k) {
                f32_t const gradient = adafactor_gradient_(state, reduced_gradients, row, column);
                band_total += gradient * gradient;
            }
        }
        band_totals[warp_index][lane_index] = band_total;
        __syncthreads();
        if (warp_index != 0u || column >= cols) return;
        for (unsigned band = 1; band != warps_k; ++band) band_total += band_totals[band][lane_index];
        f32_t const mean = band_total / f32_t(rows);
        moments[state.column_moment_offset + column] = beta2 * moments[state.column_moment_offset + column] +
                                                       (1.0f - beta2) * (mean + epsilon1);
        return;
    }
    u32_t const row = blockIdx.x - column_tiles;
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

/** The mean of @p state 's row moments, folded by the whole CTA in a fixed order, so every CTA of
 *  every pass that calls it gets the same bits. */
[[nodiscard]] __device__ inline f32_t adafactor_row_mean_(adafactor_state_t const &state,
                                                          f32r1_view_t moments) noexcept {
    u32_t const rows = static_cast<u32_t>(state.weights.extent(0));
    f32_t thread_row_sum = 0.0f;
    for (u32_t row = threadIdx.x; row < rows; row += blockDim.x)
        thread_row_sum += moments[state.row_moment_offset + row];
    return block_reduce_broadcast<kernel_block_size_default_k>(thread_row_sum) / f32_t(rows);
}

/**
 *  @brief The squared-update energy of one state row into @p row_energies at its row-moment offset,
 *      the first level of the clip's RMS; the apply pass folds the rows in order.
 *
 *  __grid_size__ @c [rows,1,1] ; threads stripe the row's columns; no dynamic shared memory.
 */
static __global__ void adafactor_scale_kernel_(adafactor_state_t state, f32r1_view_t moments,
                                               f32r1_view_t reduced_gradients, f32r1_span_t row_energies) {
    u32_t const row = blockIdx.x;
    u32_t const cols = static_cast<u32_t>(state.weights.extent(1));
    f32_t const row_mean = adafactor_row_mean_(state, moments);
    f32_t thread_update_energy = 0.0f;
    for (u32_t column = threadIdx.x; column < cols; column += blockDim.x) {
        // Divide before multiplying: a zero gradient leaves both moments at (1-beta2)*epsilon1 ~ 1e-33, and
        // the raw r*c product underflows Float32 to zero - rsqrtf(0) is Inf and 0*Inf poisons the whole
        // matrix with NaN. The paper's own normalization (r/mean(r))*c keeps every factor representable, so
        // a zero-gradient step is a clean no-op.
        f32_t const variance = moments[state.row_moment_offset + row] / row_mean *
                               moments[state.column_moment_offset + column];
        f32_t const gradient = adafactor_gradient_(state, reduced_gradients, row, column);
        f32_t const update = gradient * rsqrtf(variance);
        thread_update_energy += update * update;
    }
    f32_t const update_energy = block_reduce_broadcast<kernel_block_size_default_k>(thread_update_energy);
    if (threadIdx.x == 0u) row_energies[state.row_moment_offset + row] = update_energy;
}

/** Round the finite @p value into BFloat16 up in magnitude with the probability the dropped 16 bits
 *  weigh, against the keyed uniform draw of @p index under @p key , so the rounding is unbiased and
 *  identical on every replica. */
[[nodiscard]] __device__ inline bf16_t round_stochastically_(f32_t value, u64_t key, std::size_t index) noexcept {
    u32_t const noise = static_cast<u32_t>(splitmix64_hash_t {}(key, index)) & 0xFFFFu;
    return __ushort_as_bfloat16(static_cast<unsigned short>((__float_as_uint(value) + noise) >> 16));
}

/**
 *  @brief Apply the clipped update @c W -= @c lr·scale·G/√v , or its running mean under @p beta1
 *      divided by @p first_moment_correction when @p first_moments are kept.
 *
 *  The update lands through the Float32 master for vector weights (the norm scales) and straight
 *  into BFloat16 otherwise, rounded stochastically under @p rounding_key .
 *
 *  Every CTA folds the row-moment mean and the @p row_energies into the clip scale in the same
 *  order, so all of them agree bitwise.
 *
 *  __grid_size__ grid-stride over the state's scalars; no dynamic shared memory.
 */
static __global__ void adafactor_apply_kernel_(adafactor_state_t state, f32r1_view_t moments,
                                               f32r1_view_t reduced_gradients, f32r1_view_t row_energies,
                                               f32r1_span_t masters, f32_t learning_rate, f32_t clip_threshold,
                                               u64_t rounding_key, f32r1_span_t first_moments, f32_t beta1,
                                               f32_t first_moment_correction) {
    u32_t const rows = static_cast<u32_t>(state.weights.extent(0));
    u32_t const cols = static_cast<u32_t>(state.weights.extent(1));
    std::size_t const scalars = static_cast<std::size_t>(rows) * cols;
    f32_t const row_mean = adafactor_row_mean_(state, moments);
    f32_t thread_update_energy = 0.0f;
    for (u32_t row = threadIdx.x; row < rows; row += blockDim.x)
        thread_update_energy += row_energies[state.row_moment_offset + row];
    f32_t const update_rms = std::sqrt(block_reduce_broadcast<kernel_block_size_default_k>(thread_update_energy) /
                                       f32_t(scalars));
    f32_t const clip_scale = 1.0f / std::fmax(1.0f, update_rms / clip_threshold);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < scalars;
         index += stride) {
        u32_t const row = static_cast<u32_t>(index / cols), column = static_cast<u32_t>(index % cols);
        // The same divide-before-multiply order as the scale pass, as the raw r*c product
        // underflows to 0*Inf=NaN on zero-gradient matrices; see the scale kernel's note.
        f32_t const variance = moments[state.row_moment_offset + row] / row_mean *
                               moments[state.column_moment_offset + column];
        f32_t const gradient = adafactor_gradient_(state, reduced_gradients, row, column);
        f32_t step = learning_rate * clip_scale * gradient * rsqrtf(variance);
        if (!first_moments.empty()) {
            f32_t &moment = first_moments[state.gradient_offset + index];
            moment = beta1 * moment + (1.0f - beta1) * clip_scale * gradient * rsqrtf(variance);
            step = learning_rate * moment / first_moment_correction;
        }
        if (rows == 1u) {
            f32_t const updated = masters[state.master_offset + index] - step;
            masters[state.master_offset + index] = updated;
            state.weights[row, column] = __float2bfloat16_rn(updated);
        }
        else
            state.weights[row, column] = round_stochastically_(__bfloat162float(state.weights[row, column]) - step,
                                                               rounding_key, index);
    }
}

/**
 *  @brief Materialise the deterministic domain-reduced gradient into this rank's own buffer — the
 *      peer tier's fixed-order serial fold over this consumer's window at @p shard_begin ; the
 *      switch tier's fused loads live in @c multimem_switch.cu .
 *
 *  The result is scaled by @p world_scale , the @c 1/gpus_per_colocated_domain that turns the world @b sum
 *  into the world @b mean : the learning-rate semantics stay world-size-invariant, and the cross-domain
 *  all-reduce of domain terms each pre-scaled by @c 1/world sums to exactly the world mean (the scale is
 *  linear, so applying it before or after the sum is the same value).
 *  __grid_size__ grid-stride; no dynamic shared memory.
 */
static __global__ void adafactor_reduce_gradient_peer_kernel_(multimem_member_t member, f32r1_span_t destination,
                                                              std::size_t shard_begin, f32_t world_scale) {
    std::size_t const total = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        f32_t total_value = 0.0f;
        for (u32_t peer = 0; peer != member.member_count; ++peer) {
            f32_t const *peer_payload = member.peer_payload(peer);
            total_value = __fadd_rn(total_value, peer_payload[shard_begin + index]);
        }
        destination[index] = total_value * world_scale;
    }
}

/** Enqueue the domain-reduced gradient of member @p member into @p destination on the tier
 *  @p domain runs. */
[[nodiscard]] static expected_t reduce_gradient_(multimem_domain_t &domain, std::size_t member,
                                                 f32r1_span_t destination, std::size_t shard_begin, f32_t world_scale,
                                                 gpu_stream_t &stream) noexcept {
    dim3 const grid_shape {grid_size_for(destination.extent(0)), 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    switch (domain.tier()) {
    case multimem_tier_t::switch_reduced_k:
        return launch_function(domain.switch_kernels(member).reduce_gradient, grid_shape, block_shape, stream,
                               domain.member(member), destination, shard_begin, world_scale);
    case multimem_tier_t::peer_read_k:
        return launch_kernel(&adafactor_reduce_gradient_peer_kernel_, grid_shape, block_shape, stream,
                             domain.member(member), destination, shard_begin, world_scale);
    }
    return expected_t::fail(status_t::invalid_argument_k, "unknown multimem tier {}",
                            static_cast<u32_t>(domain.tier()));
}

/** Enqueue the seeding of @p state 's Float32 master inside @p masters from its live BFloat16
 *  weights. */
[[nodiscard]] static expected_t seed_master_(adafactor_state_t const &state, f32r1_tensor_arena_t &masters,
                                             gpu_numerics_t const &numerics, gpu_stream_t &stream) noexcept {
    std::size_t const cols = state.weights.extent(1);
    return numkong_result(numerics.cast(state.weights.data(), nk_bf16_k, masters.data() + state.master_offset, nk_f32_k,
                                        cols, stream.numkong()),
                          "master seeding");
}

expected<adafactor_t> adafactor_t::make(gpu_arena_t &arena, std::span<differentiable_weights_t const> weights,
                                        adafactor_config_t config, gpu_numerics_t const &numerics,
                                        gpu_stream_t &stream) noexcept {
    using result_t = expected<adafactor_t>;
    std::size_t const count = weights.size();
    if (count == 0) return result_t::fail(status_t::invalid_argument_k, "an optimizer needs weights");
    adafactor_t optimizer;
    optimizer.config_ = config;
    optimizer.numerics_ = numerics;
    if (st::failed(optimizer.states_.resize(count)) || st::failed(optimizer.names_.resize(count)))
        return result_t::fail(status_t::bad_alloc_k, "state table allocation failed ({})", count);
    std::size_t moment_scalars = 0, master_scalars = 0, gradient_scalars = 0;
    for (std::size_t index = 0; index != count; ++index) {
        differentiable_weights_t const &matrix = weights[index];
        std::size_t const rows = matrix.weights.extent(0), cols = matrix.weights.extent(1);
        if (rows == 0 || cols == 0 || matrix.weights_grad.extent(0) != rows || matrix.weights_grad.extent(1) != cols)
            return result_t::fail(status_t::unsupported_k, "weights {} [{}x{}] disagree with grads [{}x{}]", index,
                                  rows, cols, matrix.weights_grad.extent(0), matrix.weights_grad.extent(1));
        if (sz::failed(optimizer.names_[index].assign(matrix.name.view())))
            return result_t::fail(status_t::bad_alloc_k, "weight name copy failed: {}", matrix.name.view());
        adafactor_state_t &state = optimizer.states_[index];
        state.weights = matrix.weights;
        state.weights_grad = matrix.weights_grad;
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
    optimizer.moments_ = f32r1_tensor_arena_t::uninitialized({moment_scalars}, f32_alloc).value;
    optimizer.masters_ =
        f32r1_tensor_arena_t::uninitialized({master_scalars != 0 ? master_scalars : 1}, f32_alloc).value;
    optimizer.row_energies_ = f32r1_tensor_arena_t::uninitialized({moment_scalars}, f32_alloc).value;
    if (config.beta1 != 0.0f) {
        optimizer.first_moments_ = f32r1_tensor_arena_t::uninitialized({gradient_scalars}, f32_alloc).value;
        if (!optimizer.first_moments_)
            return result_t::fail(status_t::bad_alloc_k, "first moment allocation failed ({} scalars)",
                                  gradient_scalars);
        if (expected_t zeroed = zero_async(optimizer.first_moments_.span(), stream); !zeroed)
            return result_t::fail(std::move(zeroed));
    }
    if (!optimizer.moments_ || !optimizer.masters_ || !optimizer.row_energies_)
        return result_t::fail(status_t::bad_alloc_k, "optimizer state allocation failed ({} moments)", moment_scalars);
    if (expected_t zeroed = zero_async(optimizer.moments_.span(), stream); !zeroed)
        return result_t::fail(std::move(zeroed));
    for (adafactor_state_t const &state : optimizer.states_)
        if (state.weights.extent(0) == 1u)
            if (expected_t seeded = seed_master_(state, optimizer.masters_, numerics, stream); !seeded)
                return result_t::fail(std::move(seeded));
    return result_t {std::move(optimizer)};
}

expected_t adafactor_t::save(checkpoint_to_save_t &checkpoint, gpu_stream_t &stream) const noexcept {
    st::basic_vector<f32_t> moments, masters, first_moments;
    if (st::failed(moments.resize(st::assume_uninitialized, moments_.numel())) ||
        st::failed(masters.resize(st::assume_uninitialized, masters_.numel())) ||
        st::failed(first_moments.resize(st::assume_uninitialized, first_moments_.numel())))
        return expected_t::fail(status_t::bad_alloc_k, "optimizer state staging failed");
    expected_t staged = copy_device_to_host(moments_.view(), std::span<f32_t> {moments}, stream);
    if (staged) staged = copy_device_to_host(masters_.view(), std::span<f32_t> {masters}, stream);
    if (staged && !first_moments_.empty())
        staged = copy_device_to_host(first_moments_.view(), std::span<f32_t> {first_moments}, stream);
    if (!staged) return staged;
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "optimizer state download failed");
    for (std::size_t index = 0; index != states_.size(); ++index) {
        adafactor_state_t const &state = states_[index];
        auto insert = [&](char const *part, std::span<f32_t const> values) noexcept {
            sz::string_t name;
            if (!format_into(name, "optimizer.{}.{}", names_[index].view(), part))
                return expected_t::fail(status_t::bad_alloc_k, "optimizer state name format failed");
            return checkpoint.insert(name.view(), f32r2_view_t {values.data(), 1, values.size()});
        };
        std::size_t const rows = state.weights.extent(0), cols = state.weights.extent(1);
        expected_t inserted = insert("row_moments", {moments.data() + state.row_moment_offset, rows});
        if (inserted) inserted = insert("column_moments", {moments.data() + state.column_moment_offset, cols});
        if (inserted && rows == 1u) inserted = insert("master", {masters.data() + state.master_offset, cols});
        if (inserted && !first_moments_.empty())
            inserted = insert("first_moment", {first_moments.data() + state.gradient_offset, rows * cols});
        if (!inserted) return inserted;
    }
    return {};
}

expected_t adafactor_t::load(checkpoint_to_load_t const &checkpoint, gpu_stream_t &stream) noexcept {
    for (std::size_t index = 0; index != states_.size(); ++index) {
        adafactor_state_t const &state = states_[index];
        // Upload the saved part into `destination`, or `restart` it when the file misses that part.
        auto restore = [&](char const *part, f32r1_span_t destination, auto const &restart) noexcept {
            sz::string_t name;
            if (!format_into(name, "optimizer.{}.{}", names_[index].view(), part))
                return expected_t::fail(status_t::bad_alloc_k, "optimizer state name format failed");
            f32r2_view_t const saved = checkpoint.float32(name.view());
            if (saved.empty()) return restart();
            if (saved.extent(0) != 1u || saved.extent(1) != destination.extent(0))
                return expected_t::fail(status_t::unsupported_k, "{} holds [{},{}], the live state [1,{}]", name.view(),
                                        saved.extent(0), saved.extent(1), destination.extent(0));
            return copy_host_to_device(saved.flatten<1>(), destination, stream);
        };
        std::size_t const rows = state.weights.extent(0), cols = state.weights.extent(1);
        f32r1_span_t const row_moments {moments_.data() + state.row_moment_offset, rows};
        f32r1_span_t const column_moments {moments_.data() + state.column_moment_offset, cols};
        f32r1_span_t const master {masters_.data() + state.master_offset, cols};
        expected_t restored = restore("row_moments", row_moments,
                                      [&]() noexcept { return zero_async(row_moments, stream); });
        if (restored)
            restored = restore("column_moments", column_moments,
                               [&]() noexcept { return zero_async(column_moments, stream); });
        if (restored && rows == 1u)
            restored = restore("master", master,
                               [&]() noexcept { return seed_master_(state, masters_, numerics_, stream); });
        if (restored && !first_moments_.empty()) {
            f32r1_span_t const first_moment {first_moments_.data() + state.gradient_offset, rows * cols};
            restored = restore("first_moment", first_moment,
                               [&]() noexcept { return zero_async(first_moment, stream); });
        }
        if (!restored) return restored;
    }
    steps_ = checkpoint.step();
    return {};
}

expected_t adafactor_t::step(f32_t learning_rate, gpu_stream_t &stream, f32r1_view_t reduced_gradients) noexcept {
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    ++steps_;
    f32_t const decay = std::min(config_.beta2, 1.0f - std::pow(static_cast<f32_t>(steps_), -config_.decay_exponent));
    // Adam's bias correction, so the first steps move as far as steps without momentum.
    f32_t const first_moment_correction = 1.0f - std::pow(config_.beta1, static_cast<f32_t>(steps_));
    for (std::size_t index = 0; index != states_.size(); ++index) {
        adafactor_state_t const &state = states_[index];
        u32_t const rows = static_cast<u32_t>(state.weights.extent(0));
        std::size_t const cols = state.weights.extent(1);
        u32_t const column_tiles = static_cast<u32_t>(divide_round_up(cols, warp_size_k));
        expected_t staged = launch_kernel(&adafactor_moments_kernel_, dim3 {column_tiles + rows, 1u, 1u}, block_shape,
                                          stream, state, moments_.span(), reduced_gradients, decay, config_.epsilon1);
        if (staged)
            staged = launch_kernel(&adafactor_scale_kernel_, dim3 {rows, 1u, 1u}, block_shape, stream, state,
                                   moments_.view(), reduced_gradients, row_energies_.span());
        if (staged)
            staged = launch_kernel(&adafactor_apply_kernel_, dim3 {grid_size_for(rows * cols), 1u, 1u}, block_shape,
                                   stream, state, moments_.view(), reduced_gradients, row_energies_.view(),
                                   masters_.span(), learning_rate, config_.clip_threshold,
                                   splitmix64_hash_t {}(steps_, index), first_moments_.span(), config_.beta1,
                                   first_moment_correction);
        if (!staged) return staged;
    }
    return {};
}

expected<distributed_adafactor_t> distributed_adafactor_t::make(
    multimem_domain_t &domain, infiniband_all_reduce<f32_t> &world_reduce, std::span<gpu_arena_t> arenas,
    std::span<differentiable_weights_t const> weights, std::size_t weights_per_rank, adafactor_config_t config,
    gpu_numerics_t const &numerics, std::span<gpu_stream_t> streams, telemetry_ledger_t &ledger) noexcept {
    using result_t = expected<distributed_adafactor_t>;
    std::size_t const gpus = arenas.size();
    bool const single_domain = world_reduce.identity();
    if (gpus == 0 || streams.size() != gpus || domain.local_member_count() != gpus ||
        weights.size() != gpus * weights_per_rank)
        return result_t::fail(status_t::invalid_argument_k,
                              "every rank needs an arena, stream, local domain member, and weights table ({} ranks)",
                              gpus);
    distributed_adafactor_t optimizer;
    optimizer.domain_ = &domain;
    optimizer.ledger_ = &ledger;
    optimizer.world_reduce_ = &world_reduce;
    if (optimizer.ranks_.resize(gpus) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "rank table allocation failed");
    expected<multimem_window_t> window = domain.reserve(payload_scalars_needed(weights.subspan(0, weights_per_rank)));
    if (!window) return result_t::fail(std::move(window));
    optimizer.window_ = window.value_;

    for (std::size_t rank = 0; rank != gpus; ++rank) {
        switch_to_gpu_t gpu_scope {streams[rank]};
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed (rank={})", rank);
        expected<adafactor_t> state = adafactor_t::make(
            arenas[rank], weights.subspan(rank * weights_per_rank, weights_per_rank), config, numerics, streams[rank]);
        if (!state) return result_t::fail(std::move(state));
        optimizer.ranks_[rank].state = std::move(state.value_);
        if (single_domain) {
            gpu_arena_alloc<f32_t> f32_alloc {arenas[rank]};
            std::size_t const scalars = optimizer.ranks_[rank].state.gradient_scalars();
            optimizer.ranks_[rank].reduced = f32r1_tensor_arena_t::uninitialized({scalars}, f32_alloc).value;
            if (!optimizer.ranks_[rank].reduced)
                return result_t::fail(status_t::bad_alloc_k, "reduced-gradient allocation failed (rank={})", rank);
        }
        // The rank's gradient-boundary probes — inert zero handles when the signals channel is dark. A
        // zero gradient is a legal clean no-op, so every boundary registers as finite-only.
        auto signal_probe = [&](sz::string_view_t stage, telemetry_probe_t &into) -> expected_t {
            sz::string_t name;
            if (!format_into(name, "adafactor/{}/{}", stage, domain.member_offset() + rank))
                return expected_t::fail(status_t::bad_alloc_k, "probe name formatting failed");
            expected<telemetry_probe_t> registered = ledger.register_probe(name.view(), telemetry_signals_k,
                                                                           finite_signal_k);
            if (!registered) return expected_t::fail(std::move(registered));
            into = registered.value_;
            return {};
        };
        if (expected_t probed = signal_probe("staged_window", optimizer.ranks_[rank].staged_window); !probed)
            return result_t::fail(std::move(probed));
        if (expected_t probed = signal_probe("world_payload", optimizer.ranks_[rank].world_payload); !probed)
            return result_t::fail(std::move(probed));
        if (expected_t probed = signal_probe("world_readback", optimizer.ranks_[rank].world_readback); !probed)
            return result_t::fail(std::move(probed));
    }

    // The gateway's borrowed world buffer must fit this optimizer's gradient; the unread tail beyond it is
    // harmless, since the reduce is elementwise.
    if (!single_domain && domain.member_offset() == 0) {
        std::size_t const gradient_scalars = optimizer.ranks_[0].state.gradient_scalars();
        if (world_reduce.payload().extent(0) < gradient_scalars)
            return result_t::fail(status_t::invalid_argument_k,
                                  "the borrowed world all-reduce is too small: payload {} < gradient {} scalars",
                                  world_reduce.payload().extent(0), gradient_scalars);
    }
    return result_t {std::move(optimizer)};
}

expected_t distributed_adafactor_t::step(std::size_t rank, f32_t learning_rate, gpu_stream_t &stream) noexcept {
    std::size_t const gpus = ranks_.size();
    if (rank >= gpus) return expected_t::fail(status_t::invalid_argument_k, "rank {} of {}", rank, gpus);
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::device_error_k, "context entry failed ({})", rank);
    std::size_t const member = domain_->member_offset() + rank;
    gpu_fault_t *const fault = domain_->member(member).fault;
    f32_t *const window_base = domain_->payload(member).data() + window_.offset;
    gpu_numerics_t const &numerics = ranks_[rank].state.numerics_;
    auto stage = [&]() noexcept -> expected_t {
        expected_t staged {};
        nk::tensor_view<adafactor_state_t, 1> const states = ranks_[rank].state.states();
        for (std::size_t index = 0; index != states.extent(0) && staged; ++index) {
            adafactor_state_t const &state = states[index];
            std::size_t const scalars = state.weights.extent(0) * state.weights.extent(1);
            staged = numkong_result(
                numerics.cast(state.weights_grad.data(), nk_bf16_k, window_base + state.gradient_offset, nk_f32_k,
                              scalars, stream.numkong()),
                "gradient staging");
        }
        return staged;
    };
    bool const gateway_across_domains = member == 0 && !world_reduce_->identity();
    expected_t staged = gateway_across_domains ? expected_t {} : stage();
    if (staged) staged = domain_->barrier(member, stream);
    if (staged && gateway_across_domains) staged = stage(); // once every member read the last mean out of its window
    if (!staged) return staged;
    std::size_t const scalars = ranks_[rank].state.gradient_scalars();
    if (staged)
        staged = ledger_->probe_signal(ranks_[rank].staged_window, f32r1_view_t {window_base, scalars}, fault, stream);
    if (!staged) return staged;
    // The mean over gpus_per_colocated_domain, so the update's learning-rate semantics are
    // world-size-invariant — the same scalar on every rank, keeping the replicas bitwise identical.
    f32_t const world_scale = 1.0f /
                              static_cast<f32_t>(domain_->member_count() * world_reduce_->multimem_domain_count());
    if (world_reduce_->identity()) {
        // Materialise the domain mean locally; the world IS the domain.
        staged = reduce_gradient_(*domain_, member, ranks_[rank].reduced.span(), window_.offset, world_scale, stream);
        // Every peer reads this member's window in its reduce, so no member stages the next step
        // until all have.
        if (staged) staged = domain_->barrier(member, stream);
        if (staged)
            staged = ledger_->probe_signal(ranks_[rank].world_payload, ranks_[rank].reduced.view(), fault, stream);
        if (staged) staged = ranks_[rank].state.step(learning_rate, stream, ranks_[rank].reduced.view());
        return staged;
    }
    return world_gradient_step_(rank, learning_rate, stream);
}

expected_t distributed_adafactor_t::step(f32_t learning_rate, std::span<gpu_stream_t> streams) noexcept {
    std::size_t const gpus = ranks_.size();
    if (streams.size() != gpus)
        return expected_t::fail(status_t::invalid_argument_k, "every rank needs its stream ({} ranks)", gpus);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        if (expected_t staged = step(rank, learning_rate, streams[rank]); !staged) return staged;
    return {};
}

expected_t distributed_adafactor_t::load(checkpoint_to_load_t const &checkpoint,
                                         std::span<gpu_stream_t> streams) noexcept {
    std::size_t const gpus = ranks_.size();
    if (streams.size() != gpus)
        return expected_t::fail(status_t::invalid_argument_k, "every rank needs its stream ({} ranks)", gpus);
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        switch_to_gpu_t gpu_scope {streams[rank]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::device_error_k, "context entry failed ({})", rank);
        if (expected_t loaded = ranks_[rank].state.load(checkpoint, streams[rank]); !loaded) return loaded;
    }
    return {};
}

expected_t distributed_adafactor_t::world_gradient_step_(std::size_t rank, f32_t learning_rate,
                                                         gpu_stream_t &stream) noexcept {
    std::size_t const scalars = ranks_[rank].state.gradient_scalars();
    std::size_t const member = domain_->member_offset() + rank;
    gpu_fault_t *const fault = domain_->member(member).fault;
    f32r1_span_t const world_mean {domain_->payload(0).data() + window_.offset, scalars};
    expected_t staged {};
    if (member == 0) {
        // Each domain stages its sum pre-scaled by 1/world; the elementwise cross-domain sum of those
        // terms lands exactly on the world mean (the scale is linear either side of the sum).
        f32_t *const world_gradient = world_reduce_->payload().data();
        f32_t const world_scale = 1.0f /
                                  static_cast<f32_t>(domain_->member_count() * world_reduce_->multimem_domain_count());
        staged = reduce_gradient_(*domain_, member, f32r1_span_t {world_gradient, scalars}, window_.offset, world_scale,
                                  stream);
        if (staged) staged = (*world_reduce_)(stream);
        if (staged)
            staged = ledger_->probe_signal(ranks_[rank].world_payload, f32r1_view_t {world_gradient, scalars}, fault,
                                           stream);
        if (staged) staged = copy_device_to_device(f32r1_view_t {world_gradient, scalars}, world_mean, stream);
    }
    if (staged) staged = domain_->barrier(member, stream);
    if (staged && member != 0) // reads the mean out of the gateway's window, peer-mapped wherever its process sits
        staged = ledger_->probe_signal(ranks_[rank].world_readback, f32r1_view_t {world_mean}, fault, stream);
    if (staged) staged = ranks_[rank].state.step(learning_rate, stream, f32r1_view_t {world_mean});
    return staged;
}

/**
 *  @brief One Adafactor step of the temperature scalars: the full second moment, the RMS clip
 *      over all @b [2·groups] updates, then β in log space and the bias additively — one CTA,
 *      fixed order.
 *
 *  __grid_size__ @c [1,1,1] ; threads stripe the scalars; no dynamic shared memory.
 */
static __global__ void adafactor_temperatures_kernel_(f32r1_span_t inverse_temperatures, f32r1_span_t biases,
                                                      f32r1_view_t gradients, f32r1_span_t moments,
                                                      adafactor_config_t config, f32_t learning_rate) {
    std::size_t const groups = inverse_temperatures.extent(0), scalars = 2u * groups;
    f32_t thread_update_energy = 0.0f;
    for (std::size_t index = threadIdx.x; index < scalars; index += blockDim.x) {
        f32_t const gradient = gradients[index];
        f32_t const moment = config.beta2 * moments[index] +
                             (1.0f - config.beta2) * (gradient * gradient + config.epsilon1);
        moments[index] = moment;
        f32_t const update = gradient * rsqrtf(moment);
        thread_update_energy += update * update;
    }
    f32_t const update_rms = std::sqrt(block_reduce_broadcast<kernel_block_size_default_k>(thread_update_energy) /
                                       f32_t(scalars));
    f32_t const clip_scale = 1.0f / std::fmax(1.0f, update_rms / config.clip_threshold);
    for (std::size_t index = threadIdx.x; index < scalars; index += blockDim.x) {
        f32_t const step = learning_rate * clip_scale * gradients[index] * rsqrtf(moments[index]);
        if (index < groups) inverse_temperatures[index] *= std::exp(-step);
        else biases[index - groups] -= step;
    }
}

expected<adafactor_temperatures_t> adafactor_temperatures_t::make(gpu_arena_t &arena, u32_t groups,
                                                                  adafactor_config_t config, sz::string_view_t prefix,
                                                                  gpu_stream_t &stream) noexcept {
    using result_t = expected<adafactor_temperatures_t>;
    adafactor_temperatures_t optimizer;
    optimizer.config_ = config;
    if (!format_into(optimizer.key_, "{}.temperature_moments", prefix))
        return result_t::fail(status_t::bad_alloc_k, "temperature moments key formatting failed");
    optimizer.moments_ =
        f32r1_tensor_arena_t::uninitialized({2u * std::size_t {groups}}, gpu_arena_alloc<f32_t> {arena}).value;
    if (groups == 0 || !optimizer.moments_)
        return result_t::fail(status_t::bad_alloc_k, "temperature moments allocation failed ({} groups)", groups);
    if (expected_t zeroed = zero_async(optimizer.moments_.span(), stream); !zeroed)
        return result_t::fail(std::move(zeroed));
    return result_t {std::move(optimizer)};
}

expected_t adafactor_temperatures_t::step(f32r1_span_t inverse_temperatures, f32r1_span_t biases,
                                          f32r1_view_t gradients, f32_t learning_rate, gpu_stream_t &stream) noexcept {
    std::size_t const scalars = moments_.numel();
    if (inverse_temperatures.extent(0) * 2u != scalars || biases.extent(0) * 2u != scalars ||
        gradients.extent(0) != scalars)
        return expected_t::fail(status_t::unsupported_k,
                                "temperatures [{}], biases [{}] and gradients [{}] disagree with {} moments",
                                inverse_temperatures.extent(0), biases.extent(0), gradients.extent(0), scalars);
    return launch_kernel(&adafactor_temperatures_kernel_, dim3 {1u, 1u, 1u}, dim3 {kernel_block_size_default_k, 1u, 1u},
                         stream, inverse_temperatures, biases, gradients, moments_.span(), config_, learning_rate);
}

expected_t adafactor_temperatures_t::save(checkpoint_to_save_t &checkpoint, gpu_stream_t &stream) const noexcept {
    st::basic_vector<f32_t> moments;
    if (st::failed(moments.resize(st::assume_uninitialized, moments_.numel())))
        return expected_t::fail(status_t::bad_alloc_k, "temperature moments staging failed");
    if (expected_t copied = copy_device_to_host(moments_.view(), std::span<f32_t> {moments}, stream); !copied)
        return copied;
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "temperature moments download failed");
    return checkpoint.insert(key_.view(), f32r2_view_t {moments.data(), 1, moments.size()});
}

expected_t adafactor_temperatures_t::load(checkpoint_to_load_t const &checkpoint, gpu_stream_t &stream) noexcept {
    f32r2_view_t const saved = checkpoint.float32(key_.view());
    if (saved.empty()) return zero_async(moments_.span(), stream);
    if (saved.extent(0) != 1u || saved.extent(1) != moments_.numel())
        return expected_t::fail(status_t::unsupported_k, "{} holds [{},{}], the run [1,{}]", key_.view(),
                                saved.extent(0), saved.extent(1), moments_.numel());
    return copy_host_to_device(saved.flatten<1>(), moments_.span(), stream);
}
