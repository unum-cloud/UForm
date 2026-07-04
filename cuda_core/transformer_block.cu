/**
 *  @brief Device kernels and heavy method bodies of the two transformer blocks declared in
 *      @ref transformer_block.cuh — the one translation unit behind the `uform_attention` target.
 *  @file cuda_core/transformer_block.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include "transformer_block.cuh"

namespace unum::uform::core {

/** @brief Learnable RMSNorm forward — declared in @ref transformer_block.cuh with external linkage so the
 *      encoder tower launches the same kernel. */
__global__ void rmsnorm_forward_kernel(bf16r2_view_t input, bf16r1_view_t scale, bf16r2_span_t output,
                                       f32r1_span_t inverse_rms, f32_t epsilon) {
    std::size_t const rows = input.extent(0);
    u32_t const width = static_cast<u32_t>(input.extent(1));
    std::size_t const row = blockIdx.x;
    if (row >= rows) return;
    f32_t thread_partial = 0.0f;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t const value = __bfloat162float(input[row, channel]);
        thread_partial += value * value;
    }
    f32_t const sum_squares = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial);
    f32_t const inverse = rsqrtf(sum_squares / static_cast<f32_t>(width) + epsilon);
    if (threadIdx.x == 0u) inverse_rms[row] = inverse;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x)
        output[row, channel] = __float2bfloat16_rn(__bfloat162float(input[row, channel]) * inverse *
                                                   __bfloat162float(scale[channel]));
}

/** @brief Learnable RMSNorm backward for the row gradient — external linkage; see the header. */
__global__ void rmsnorm_backward_kernel(bf16r2_view_t input, bf16r1_view_t scale, f32r1_view_t inverse_rms,
                                        bf16r2_view_t output_grad, bf16r2_span_t input_grad, bool accumulate) {
    std::size_t const rows = input.extent(0);
    u32_t const width = static_cast<u32_t>(input.extent(1));
    std::size_t const row = blockIdx.x;
    if (row >= rows) return;
    f32_t const inverse = inverse_rms[row];
    f32_t thread_partial = 0.0f;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t const normalized = __bfloat162float(input[row, channel]) * inverse;
        f32_t const scaled_grad = __bfloat162float(output_grad[row, channel]) * __bfloat162float(scale[channel]);
        thread_partial += scaled_grad * normalized;
    }
    f32_t const projection = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial) /
                             static_cast<f32_t>(width);
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t const normalized = __bfloat162float(input[row, channel]) * inverse;
        f32_t const scaled_grad = __bfloat162float(output_grad[row, channel]) * __bfloat162float(scale[channel]);
        f32_t const row_grad = (scaled_grad - normalized * projection) * inverse;
        f32_t const previous = accumulate ? __bfloat162float(input_grad[row, channel]) : 0.0f;
        input_grad[row, channel] = __float2bfloat16_rn(previous + row_grad);
    }
}

/** @brief Learnable RMSNorm backward for the scale gradient — external linkage; see the header. */
__global__ void rmsnorm_scale_grad_kernel(bf16r2_view_t input, f32r1_view_t inverse_rms, bf16r2_view_t output_grad,
                                          bf16r1_span_t scale_grad) {
    std::size_t const rows = input.extent(0);
    u32_t const width = static_cast<u32_t>(input.extent(1));
    u32_t const channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) return;
    f32_t accumulated = 0.0f;
    for (std::size_t row = 0; row != rows; ++row)
        accumulated += __bfloat162float(output_grad[row, channel]) * __bfloat162float(input[row, channel]) *
                       inverse_rms[row];
    scale_grad[channel] = __float2bfloat16_rn(accumulated);
}

/** @brief Inverted keyed dropout, in place — external linkage; see the header. The @p salt selects the
 *      site (per layer, per residual join) and @p epoch , read from device memory, advances the mask per
 *      step; @ref keyed_uniform re-derives the identical keep decision wherever this runs. */
__global__ void dropout_kernel(bf16r1_span_t values, u32r1_view_t epoch, u32_t salt, f32_t drop_probability) {
    u32_t const key = salt ^ (epoch[0] * 2654435761u);
    f32_t const keep_scale = 1.0f / (1.0f - drop_probability);
    std::size_t const total = values.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        f32_t const draw = keyed_uniform(static_cast<u32_t>(index), key);
        f32_t const value = __bfloat162float(values[index]);
        values[index] = __float2bfloat16_rn(draw < drop_probability ? 0.0f : value * keep_scale);
    }
}

expected<rope_angles_t> rope_angles_t::make(gpu_arena_t &arena, u32_t max_positions, rope_shape_t shape,
                                            f32_t base) noexcept {
    using result_t = expected<rope_angles_t>;
    rope_angles_t angles;
    angles.shape_ = shape;
    angles.max_positions_ = max_positions;
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    u32_t const pair_counts[3] = {shape.temporal / 2u, shape.height / 2u, shape.width / 2u};
    u32_t const denominators[3] = {shape.temporal + shape.height + shape.width, shape.height, shape.width};
    std::size_t total = 0;
    for (u32_t axis = 0; axis != 3; ++axis) total += static_cast<std::size_t>(max_positions) * pair_counts[axis];
    angles.angles_ = f32r1_tensor_arena_t::try_empty({total}, f32_alloc);
    array<f32_t> host_angles {};
    if (!angles.angles_ || host_angles.try_resize(total) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "angle grid allocation failed ({} scalars)", total);
    std::size_t offset = 0;
    for (u32_t axis = 0; axis != 3; ++axis) {
        angles.axis_offsets_[axis] = offset;
        for (u32_t position = 0; position != max_positions; ++position)
            for (u32_t pair = 0; pair != pair_counts[axis]; ++pair)
                host_angles[offset++] = static_cast<f32_t>(position) *
                                        static_cast<f32_t>(
                                            std::pow(static_cast<f64_t>(base),
                                                     -2.0 * pair / static_cast<f64_t>(denominators[axis])));
    }
    std::span<f32_t const> host_view = host_angles.view();
    if (CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(angles.angles_.data()), host_view.data(),
                                       total * sizeof(f32_t));
        result != CUDA_SUCCESS)
        return result_t::fail(result, "angle grid upload failed");
    return result_t {std::move(angles)};
}

/**
 *  @brief Non-learnable unit RMSNorm forward over per-head vectors of @ref transformer_head_dim_k
 *      channels — the QK-norm. Saves the pre-norm values and the Float32 @c 1/rms per head vector.
 *
 *  __grid_size__ @c [tokens·heads,1,1] with one @b warp per head vector packed 8-per-CTA; threads stripe
 *  the 128 channels; no dynamic shared memory.
 *
 *  @note @p values is read-modified in place (the QK-norm rescales each head vector here).
 */
static __global__ void transformer_qk_norm_forward_kernel_(bf16r2_span_t values, bf16r2_span_t prenorm_stash,
                                                           f32r1_span_t inverse_rms, u32_t heads, u32_t role_offset,
                                                           f32_t epsilon) {
    std::size_t const tokens = values.extent(0);
    std::size_t const vectors = tokens * heads;
    std::size_t const vector = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size_k;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    if (vector >= vectors) return;
    std::size_t const token = vector / heads;
    u32_t const head = static_cast<u32_t>(vector % heads);
    u32_t const base_channel = role_offset + head * transformer_head_dim_k;
    u32_t const stash_channel = head * transformer_head_dim_k;
    f32_t lane_partial = 0.0f;
    for (u32_t channel = lane_index; channel < transformer_head_dim_k; channel += warp_size_k) {
        f32_t const value = __bfloat162float(values[token, base_channel + channel]);
        lane_partial += value * value;
    }
    f32_t const inverse = rsqrtf(warp_reduce_broadcast(lane_partial) / static_cast<f32_t>(transformer_head_dim_k) +
                                 epsilon);
    if (lane_index == 0u) inverse_rms[vector] = inverse;
    for (u32_t channel = lane_index; channel < transformer_head_dim_k; channel += warp_size_k) {
        __nv_bfloat16 const raw = values[token, base_channel + channel];
        prenorm_stash[token, stash_channel + channel] = raw;
        values[token, base_channel + channel] = __float2bfloat16_rn(__bfloat162float(raw) * inverse);
    }
}

/**
 *  @brief Unit RMSNorm backward for the QK-norm, in place on the strided gradient:
 *      @c dx = (dy − x̂·mean(dy·x̂))·(1/rms) over each 128-channel head vector.
 *
 *  __grid_size__ mirrors @ref transformer_qk_norm_forward_kernel_ ; no dynamic shared memory.
 *
 *  @note @p grads is read-modified in place (the QK-norm backward rescales each head vector here).
 */
static __global__ void transformer_qk_norm_backward_kernel_(bf16r2_span_t grads, bf16r2_view_t prenorm_stash,
                                                            f32r1_view_t inverse_rms, u32_t heads, u32_t role_offset) {
    std::size_t const tokens = grads.extent(0);
    std::size_t const vectors = tokens * heads;
    std::size_t const vector = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size_k;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    if (vector >= vectors) return;
    std::size_t const token = vector / heads;
    u32_t const head = static_cast<u32_t>(vector % heads);
    u32_t const base_channel = role_offset + head * transformer_head_dim_k;
    u32_t const stash_channel = head * transformer_head_dim_k;
    f32_t const inverse = inverse_rms[vector];
    f32_t lane_partial = 0.0f;
    for (u32_t channel = lane_index; channel < transformer_head_dim_k; channel += warp_size_k) {
        f32_t const normalized = __bfloat162float(prenorm_stash[token, stash_channel + channel]) * inverse;
        lane_partial += __bfloat162float(grads[token, base_channel + channel]) * normalized;
    }
    f32_t const projection = warp_reduce_broadcast(lane_partial) / static_cast<f32_t>(transformer_head_dim_k);
    for (u32_t channel = lane_index; channel < transformer_head_dim_k; channel += warp_size_k) {
        f32_t const normalized = __bfloat162float(prenorm_stash[token, stash_channel + channel]) * inverse;
        f32_t const grad = __bfloat162float(grads[token, base_channel + channel]);
        grads[token, base_channel + channel] = __float2bfloat16_rn((grad - normalized * projection) * inverse);
    }
}

/** @brief Pick the axis table and angle for one channel pair — the rotary storage invariant.
 *      Multi-argument indexing spells @c operator() here: device code follows nvcc's C++20 ceiling
 *      until a C++23-capable toolkit lands. */
[[nodiscard]] __device__ inline f32_t transformer_rope_angle_(u32_t pair_index, i16r2_view_t coords, std::size_t token,
                                                              rope_angles_view_t const &angles) noexcept {
    u32_t const temporal_pairs = static_cast<u32_t>(angles.temporal.extent(1));
    u32_t const height_pairs = static_cast<u32_t>(angles.height.extent(1));
    if (pair_index < temporal_pairs) {
        std::size_t const coord = static_cast<std::size_t>(coords(token, 0));
        return angles.temporal(coord, pair_index);
    }
    if (pair_index < temporal_pairs + height_pairs) {
        std::size_t const coord = static_cast<std::size_t>(coords(token, 1));
        return angles.height(coord, pair_index - temporal_pairs);
    }
    std::size_t const coord = static_cast<std::size_t>(coords(token, 2));
    return angles.width(coord, pair_index - temporal_pairs - height_pairs);
}

/**
 *  @brief Rotary rotation in place over one strided role slice (Q or K inside the packed QKV):
 *      channel pair @b (2k,2k+1) rotates by the axis angle its index selects; @p angle_sign @c -1 gives
 *      the inverse rotation the backward applies to the post-RoPE gradients.
 *
 *  __grid_size__ @c [ceil(tokens·heads·64/block),1,1] grid-stride over channel-pair slots; no dynamic
 *  shared memory.
 *
 *  @note @p values is read-modified in place (the rotation overwrites each channel pair here).
 */
static __global__ void transformer_rope_kernel_(bf16r2_span_t values, rope_angles_view_t angles, i16r2_view_t coords,
                                                u32_t heads, u32_t role_offset, f32_t angle_sign) {
    std::size_t const tokens = values.extent(0);
    u32_t const pairs_per_head = transformer_head_dim_k >> 1u;
    std::size_t const slots_per_token = static_cast<std::size_t>(pairs_per_head) * heads;
    std::size_t const total_slots = slots_per_token * tokens;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; slot < total_slots;
         slot += stride) {
        std::size_t const token = slot / slots_per_token;
        u32_t const head = static_cast<u32_t>((slot % slots_per_token) / pairs_per_head);
        u32_t const pair = static_cast<u32_t>(slot % pairs_per_head);
        u32_t const channel = role_offset + head * transformer_head_dim_k + 2u * pair;
        f32_t const real = __bfloat162float(values[token, channel]);
        f32_t const imaginary = __bfloat162float(values[token, channel + 1u]);
        f32_t sin_value, cos_value;
        __sincosf(angle_sign * transformer_rope_angle_(pair, coords, token, angles), &sin_value, &cos_value);
        values[token, channel] = __float2bfloat16_rn(real * cos_value - imaginary * sin_value);
        values[token, channel + 1u] = __float2bfloat16_rn(real * sin_value + imaginary * cos_value);
    }
}

/**
 *  @brief SwiGLU activation forward over the packed @b [tokens,2·ffn] gate|up block:
 *      @c silu_up[t,f] = SiLU(gate[t,f])·up[t,f] .
 *
 *  __grid_size__ grid-stride over @c tokens·ffn cells; no dynamic shared memory.
 */
static __global__ void transformer_swiglu_forward_kernel_(bf16r2_view_t gate_up, bf16r2_span_t silu_up) {
    std::size_t const tokens = silu_up.extent(0);
    u32_t const ffn = static_cast<u32_t>(silu_up.extent(1));
    std::size_t const total = tokens * ffn;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const token = cell / ffn;
        u32_t const feature = static_cast<u32_t>(cell % ffn);
        f32_t const gate = __bfloat162float(gate_up[token, feature]);
        f32_t const up = __bfloat162float(gate_up[token, ffn + feature]);
        f32_t const sigmoid = 1.0f / (1.0f + expf(-gate));
        silu_up[token, feature] = __float2bfloat16_rn(gate * sigmoid * up);
    }
}

/**
 *  @brief SwiGLU activation backward, writing the packed @b [tokens,2·ffn] gate|up gradient:
 *      @c d_up = dy·SiLU(gate) and @c d_gate = dy·up·σ(gate)·(1+gate·(1−σ(gate))) .
 *
 *  __grid_size__ grid-stride over @c tokens·ffn cells; no dynamic shared memory.
 */
static __global__ void transformer_swiglu_backward_kernel_(bf16r2_view_t gate_up, bf16r2_view_t silu_up_grad,
                                                           bf16r2_span_t gate_up_grad) {
    std::size_t const tokens = silu_up_grad.extent(0);
    u32_t const ffn = static_cast<u32_t>(silu_up_grad.extent(1));
    std::size_t const total = tokens * ffn;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const token = cell / ffn;
        u32_t const feature = static_cast<u32_t>(cell % ffn);
        f32_t const gate = __bfloat162float(gate_up[token, feature]);
        f32_t const up = __bfloat162float(gate_up[token, ffn + feature]);
        f32_t const grad = __bfloat162float(silu_up_grad[token, feature]);
        f32_t const sigmoid = 1.0f / (1.0f + expf(-gate));
        f32_t const silu = gate * sigmoid;
        gate_up_grad[token, feature] = __float2bfloat16_rn(grad * up * sigmoid * (1.0f + gate * (1.0f - sigmoid)));
        gate_up_grad[token, ffn + feature] = __float2bfloat16_rn(grad * silu);
    }
}

/**
 *  @brief Elementwise residual accumulate: @c destination += source .
 *
 *  __grid_size__ grid-stride; no dynamic shared memory.
 *
 *  @note @p destination is read-modified in place (the residual merges here).
 */
static __global__ void transformer_residual_add_kernel_(bf16r1_span_t destination, bf16r1_view_t source) {
    std::size_t const total = destination.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride)
        destination[cell] = __float2bfloat16_rn(__bfloat162float(destination[cell]) + __bfloat162float(source[cell]));
}

/**
 *  @brief Order-free max-abs of one BFloat16 tensor via bitwise @c atomicMax on the non-negative Float32
 *      bit pattern — max is commutative and associative, so the result is deterministic regardless of
 *      thread order.
 *
 *  __grid_size__ grid-stride; @p max_bits must be pre-zeroed; no dynamic shared memory.
 */
static __global__ void transformer_absolute_max_kernel_(bf16r1_view_t values, u32r1_span_t max_bits) {
    std::size_t const total = values.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    f32_t thread_max = 0.0f;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride)
        thread_max = fmaxf(thread_max, fabsf(__bfloat162float(values[cell])));
    // Non-negative floats compare identically as unsigned integers, so one atomic per thread suffices.
    atomicMax(&max_bits[0], __float_as_uint(thread_max));
}

/** @brief Resolve the max-abs bits into the Float32 descale scalar @c max_abs/240 (floored to keep a
 *      zero weight tensor representable). One thread. */
static __global__ void transformer_descale_from_max_kernel_(u32r1_span_t max_bits, f32r1_span_t descale) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    f32_t const max_abs = __uint_as_float(max_bits[0]);
    descale[0] = fmaxf(max_abs / e4m3_range_k, e4m3_descale_floor_k);
}

/**
 *  @brief Quantise BFloat16 into the E4M3 shadow, dividing by the descale scalar the matmuls will
 *      multiply back through their native scale pointers. With a null @p descale (empty view) the cast
 *      is scale-free — the activation path.
 *
 *  __grid_size__ grid-stride; no dynamic shared memory.
 */
static __global__ void transformer_quantize_e4m3_kernel_(bf16r1_view_t values, e4m3r1_span_t shadow,
                                                         f32r1_view_t descale) {
    std::size_t const total = values.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    f32_t const inverse_scale = descale.empty() ? 1.0f : 1.0f / descale[0];
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride)
        shadow[cell] = __nv_fp8_e4m3 {__bfloat162float(values[cell]) * inverse_scale};
}

/**
 *  @brief Per-token E4M3 quantisation of a contiguous @b [tokens,width] buffer: each row's scale is its
 *      own @c max_abs/240 , so a quiet token never wastes the format's range on a loud one — the finer
 *      scaling tier the matmul path uses (the row scale folds back after the product).
 *
 *  __grid_size__ @c [tokens,1,1] , one CTA per token; no dynamic shared memory.
 */
static __global__ void transformer_token_quantize_kernel_(bf16r2_view_t values, e4m3r2_span_t shadow,
                                                          f32r1_span_t token_scales) {
    std::size_t const token = blockIdx.x;
    u32_t const width = static_cast<u32_t>(values.extent(1));
    f32_t thread_max = 0.0f;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x)
        thread_max = fmaxf(thread_max, fabsf(__bfloat162float(values[token, channel])));
    // Max through the additive reduce tree: fold |x| via the shuffle butterfly's max analogue.
    for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffffu, thread_max, delta));
    __shared__ f32_t warp_partials[kernel_block_size_default_k / warp_size_k];
    __shared__ f32_t token_max;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    unsigned const warp_index = threadIdx.x / warp_size_k;
    if (lane_index == 0u) warp_partials[warp_index] = thread_max;
    __syncthreads();
    if (warp_index == 0u) {
        f32_t staged = lane_index < kernel_block_size_default_k / warp_size_k ? warp_partials[lane_index] : 0.0f;
        for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
            staged = fmaxf(staged, __shfl_xor_sync(0xffffffffu, staged, delta));
        if (lane_index == 0u) token_max = staged;
    }
    __syncthreads();
    f32_t const scale = fmaxf(token_max / e4m3_range_k, e4m3_descale_floor_k);
    if (threadIdx.x == 0u) token_scales[token] = scale;
    f32_t const inverse = 1.0f / scale;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x)
        shadow[token, channel] = __nv_fp8_e4m3 {__bfloat162float(values[token, channel]) * inverse};
}

/**
 *  @brief Fold the per-token scales back into a matmul's BFloat16 output rows:
 *      @c values[t,o] *= token_scales[t] .
 *
 *  __grid_size__ grid-stride over @c tokens·width cells; no dynamic shared memory.
 *
 *  @note @p values is read-modified in place (the token rescale folds back here).
 */
static __global__ void transformer_token_rescale_kernel_(bf16r2_span_t values, f32r1_view_t token_scales) {
    std::size_t const tokens = values.extent(0);
    u32_t const width = static_cast<u32_t>(values.extent(1));
    std::size_t const total = tokens * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const token = cell / width;
        u32_t const channel = static_cast<u32_t>(cell % width);
        values[token, channel] = __float2bfloat16_rn(__bfloat162float(values[token, channel]) * token_scales[token]);
    }
}

/**
 *  @brief Pack one strided role slice of the packed BFloat16 QKV into a @b dense zero-padded E4M3
 *      operand — @b [segments,seq_max,width] — quantised by one dynamic per-tensor scale: the FP8 SDPA
 *      engines take per-tensor descales and reject ragged offsets, so the dense layout is theirs.
 *
 *  __grid_size__ grid-stride over @c segments·seq_max·width cells; no dynamic shared memory.
 */
static __global__ void transformer_pack_dense_kernel_(bf16r2_view_t packed, f32r1_view_t tensor_scale,
                                                      u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                      e4m3r3_span_t dense, u32_t role_offset) {
    u32_t const seq_max = static_cast<u32_t>(dense.extent(1));
    u32_t const width = static_cast<u32_t>(dense.extent(2));
    std::size_t const total = dense.extent(0) * seq_max * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    f32_t const inverse = 1.0f / tensor_scale[0];
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const segment = cell / (static_cast<std::size_t>(seq_max) * width);
        u32_t const row = static_cast<u32_t>((cell / width) % seq_max);
        u32_t const channel = static_cast<u32_t>(cell % width);
        f32_t value = 0.0f;
        if (row < segment_lengths[segment]) {
            std::size_t const token = static_cast<std::size_t>(segment_offsets[segment]) + row;
            value = __bfloat162float(packed[token, role_offset + channel]) * inverse;
        }
        dense[segment, row, channel] = __nv_fp8_e4m3 {value};
    }
}

/**
 *  @brief Unpack the dense E4M3 SDPA output back into the packed @b [tokens,width] layout the
 *      out-projection consumes; padded rows are skipped, never read.
 *
 *  __grid_size__ grid-stride over the dense cells; no dynamic shared memory.
 */
static __global__ void transformer_unpack_dense_kernel_(e4m3r3_view_t dense, e4m3r2_span_t packed,
                                                        u32r1_view_t segment_offsets, u32r1_view_t segment_lengths) {
    u32_t const seq_max = static_cast<u32_t>(dense.extent(1));
    u32_t const width = static_cast<u32_t>(dense.extent(2));
    std::size_t const total = dense.extent(0) * seq_max * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const segment = cell / (static_cast<std::size_t>(seq_max) * width);
        u32_t const row = static_cast<u32_t>((cell / width) % seq_max);
        if (row >= segment_lengths[segment]) continue;
        u32_t const channel = static_cast<u32_t>(cell % width);
        std::size_t const token = static_cast<std::size_t>(segment_offsets[segment]) + row;
        packed[token, channel] = dense[segment, row, channel];
    }
}

/**
 *  @brief The one cuDNN Int32 boundary: scale the batch's flat @c u32 segment starts (gapped —
 *      cuDNN's ragged prefix supports gaps; batch stride 0, the prefix supplies each segment's
 *      base) into the per-operand @b element offsets, and cast the stored lengths for the padding
 *      mask. One launch fills the block's whole Int32 staging; nothing outside it sees Int32.
 *      vendor-boundary.
 *
 *  __grid_size__ @c [1,1,1] , threads stripe @c segments+1 entries; no dynamic shared memory.
 */
static __global__ void transformer_ragged_offsets_kernel_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                          i32r1_span_t qkv_offsets, i32r1_span_t hidden_offsets,
                                                          i32r1_span_t seq_lens, u32_t hidden) {
    std::size_t const entries = segment_offsets.extent(0);
    for (std::size_t entry = threadIdx.x; entry < entries; entry += blockDim.x) {
        i32_t const start = static_cast<i32_t>(segment_offsets[entry]);
        qkv_offsets[entry] = start * static_cast<i32_t>(3u * hidden);
        hidden_offsets[entry] = start * static_cast<i32_t>(hidden);
        if (entry + 1u < entries) seq_lens[entry] = static_cast<i32_t>(segment_lengths[entry]);
    }
}

/**
 *  @brief The quantized twin's slice of the same vendor boundary: its FP8 SDPA is dense per segment
 *      (no ragged offsets), so only the Int32 padding-mask lengths need staging. vendor-boundary.
 *
 *  __grid_size__ @c [1,1,1] , threads stripe the segments; no dynamic shared memory.
 */
static __global__ void transformer_seq_lens_cast_kernel_(u32r1_view_t segment_lengths, i32r1_span_t seq_lens) {
    std::size_t const entries = segment_lengths.extent(0);
    for (std::size_t entry = threadIdx.x; entry < entries; entry += blockDim.x)
        seq_lens[entry] = static_cast<i32_t>(segment_lengths[entry]);
}

/** @brief The strided THD tensor description shared by every SDPA operand: logical @b {B,H,S,D} over the
 *      packed token buffer, batch stride 0 (ragged offsets supply segment bases), token stride
 *      @p token_stride so Q / K / V can live inside the fused QKV block. */
[[nodiscard]] inline std::vector<i64_t> sdpa_thd_strides(i64_t heads, i64_t token_stride) { // vendor-boundary
    return {0, transformer_head_dim_k, token_stride, 1};                                    // vendor-boundary
}

/** @brief Attach a @b [segments+1] Int32 ragged prefix-sum input to @p graph under @p uid . */
[[nodiscard]] inline auto sdpa_ragged_offset(fe::graph::Graph &graph, i64_t uid, i64_t segments) {
    return graph.tensor(fe::graph::Tensor_attributes()
                            .set_name("ragged")
                            .set_uid(uid)
                            .set_dim({segments + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
}

/** @brief Attach a @b [segments] Int32 per-segment length input to @p graph under @p uid . */
[[nodiscard]] inline auto sdpa_seq_lengths(fe::graph::Graph &graph, i64_t uid, char const *name, i64_t segments) {
    return graph.tensor(fe::graph::Tensor_attributes()
                            .set_name(name)
                            .set_uid(uid)
                            .set_dim({segments, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
}

expected_t sdpa_graph_t::verify_raw_bindings(i64_t const *uids, u32_t count) noexcept {
    if (variant_pack_uids.size() != count)
        return expected_t::fail(status_t::other_k,
                                "SDPA graph binds {} tensors, the raw table names {} — a frontend-internal tensor " //
                                "(pass-by-value scalar or workspace) slipped into the variant pack",
                                variant_pack_uids.size(), count);
    for (i64_t const uid : variant_pack_uids) {
        bool named = false;
        for (u32_t slot = 0; slot != count && !named; ++slot) named = uids[slot] == uid;
        if (!named)
            return expected_t::fail(status_t::other_k, "SDPA graph demands uid {} the raw table never names", uid);
    }
    i64_t total_workspace = 0;
    if (get_workspace_size(total_workspace).is_bad())
        return expected_t::fail(status_t::cuda_error_k, "SDPA workspace query failed");
    if (total_workspace != plan_workspace_bytes())
        return expected_t::fail(status_t::other_k,
                                "the frontend reserves {} workspace bytes beyond the plan's {} — raw execution " //
                                "would skip its auxiliary kernels",
                                total_workspace - plan_workspace_bytes(), plan_workspace_bytes());
    return {};
}

expected_t sdpa_plan_t::execute(cudnnHandle_t handle, void *workspace) noexcept {
    if (!raw) {
        for (u32_t slot = 0; slot != binding_count; ++slot) fallback_bindings[uids[slot]] = pointers[slot];
        if (auto status = graph.execute(handle, fallback_bindings, workspace); status.is_bad())
            return expected_t::fail(status_t::cuda_error_k, "SDPA fallback execute failed: {}",
                                    status.get_message().c_str()); // vendor-boundary
        return {};
    }
    cudnnBackendDescriptor_t pack = nullptr;
    if (cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &pack);
        status != CUDNN_STATUS_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "variant pack creation failed ({})",
                                cudnnGetErrorString(status));
    cudnnStatus_t status = cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64,
                                                    binding_count, uids.data());
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,
                                          binding_count, const_cast<void **>(pointers.data()));
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendFinalize(pack);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendExecute(handle, graph.raw_plan(), pack);
    cudnnBackendDestroyDescriptor(pack);
    if (status != CUDNN_STATUS_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "SDPA raw execute failed ({})", cudnnGetErrorString(status));
    return {};
}

expected<transformer_block_bf16_t> transformer_block_bf16_t::make(gpu_arena_t &arena, cublaslt_session_t const &blas,
                                                                  cudnn_session_t const &cudnn,
                                                                  transformer_block_config_t config) noexcept {
    if (config.hidden == 0 || config.hidden % transformer_head_dim_k != 0)
        return expected<transformer_block_bf16_t>::fail(status_t::shape_unsupported_k,
                                                        "hidden must be a positive multiple of {} (got {})",
                                                        transformer_head_dim_k, config.hidden);
    if (config.max_tokens == 0 || config.max_segments == 0 || config.max_segment_tokens == 0)
        return expected<transformer_block_bf16_t>::fail(
            status_t::invalid_input_k, "token capacities must be positive (tokens={} segments={} longest={})",
            config.max_tokens, config.max_segments, config.max_segment_tokens);
    transformer_block_bf16_t block;
    block.config_ = config;
    block.blas_ = &blas;
    block.cudnn_ = &cudnn;

    std::size_t const hidden = config.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config.ffn_expansion) * hidden;
    std::size_t const tokens = config.max_tokens;
    std::size_t const heads = hidden / transformer_head_dim_k;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};
    // Plain `try_empty` allocations, checked in named groups so a failure names the arena pressure.

    // Weights and their per-block gradient buffers, `[out,in]` row-major.
    block.qkv_weight_ = bf16r2_tensor_arena_t::try_empty({3 * hidden, hidden}, bf16_alloc);
    block.out_weight_ = bf16r2_tensor_arena_t::try_empty({hidden, hidden}, bf16_alloc);
    block.gate_up_weight_ = bf16r2_tensor_arena_t::try_empty({2 * ffn, hidden}, bf16_alloc);
    block.down_weight_ = bf16r2_tensor_arena_t::try_empty({hidden, ffn}, bf16_alloc);
    block.pre_norm_scale_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    block.post_norm_scale_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    block.qkv_weight_grad_ = bf16r2_tensor_arena_t::try_empty({3 * hidden, hidden}, bf16_alloc);
    block.out_weight_grad_ = bf16r2_tensor_arena_t::try_empty({hidden, hidden}, bf16_alloc);
    block.gate_up_weight_grad_ = bf16r2_tensor_arena_t::try_empty({2 * ffn, hidden}, bf16_alloc);
    block.down_weight_grad_ = bf16r2_tensor_arena_t::try_empty({hidden, ffn}, bf16_alloc);
    block.pre_norm_scale_grad_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    block.post_norm_scale_grad_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    if (!block.qkv_weight_ || !block.out_weight_ || !block.gate_up_weight_ || !block.down_weight_ ||
        !block.pre_norm_scale_ || !block.post_norm_scale_ || !block.qkv_weight_grad_ || !block.out_weight_grad_ ||
        !block.gate_up_weight_grad_ || !block.down_weight_grad_ || !block.pre_norm_scale_grad_ ||
        !block.post_norm_scale_grad_)
        return expected<transformer_block_bf16_t>::fail(status_t::bad_alloc_k,
                                                        "weights allocation failed (arena_used={})", arena.used());
    // Forward stashes the backward consumes.
    block.input_stash_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.normalized_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.qkv_ = bf16r2_tensor_arena_t::try_empty({tokens, 3 * hidden}, bf16_alloc);
    block.query_prenorm_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.key_prenorm_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.attn_output_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.attn_residual_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.mlp_normalized_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.gate_up_ = bf16r2_tensor_arena_t::try_empty({tokens, 2 * ffn}, bf16_alloc);
    block.silu_up_ = bf16r2_tensor_arena_t::try_empty({tokens, ffn}, bf16_alloc);
    block.pre_inverse_rms_ = f32r1_tensor_arena_t::try_empty({tokens}, f32_alloc);
    block.post_inverse_rms_ = f32r1_tensor_arena_t::try_empty({tokens}, f32_alloc);
    block.query_inverse_rms_ = f32r1_tensor_arena_t::try_empty({tokens * heads}, f32_alloc);
    block.key_inverse_rms_ = f32r1_tensor_arena_t::try_empty({tokens * heads}, f32_alloc);
    block.softmax_log_sum_exp_ = f32r1_tensor_arena_t::try_empty(
        {static_cast<std::size_t>(config.max_segments) * heads * config.max_segment_tokens}, f32_alloc);
    if (!block.input_stash_ || !block.normalized_ || !block.qkv_ || !block.query_prenorm_ || !block.key_prenorm_ ||
        !block.attn_output_ || !block.attn_residual_ || !block.mlp_normalized_ || !block.gate_up_ || !block.silu_up_ ||
        !block.pre_inverse_rms_ || !block.post_inverse_rms_ || !block.query_inverse_rms_ || !block.key_inverse_rms_ ||
        !block.softmax_log_sum_exp_)
        return expected<transformer_block_bf16_t>::fail(status_t::bad_alloc_k,
                                                        "stashes allocation failed (arena_used={})", arena.used());
    // Backward transients.
    block.qkv_grad_ = bf16r2_tensor_arena_t::try_empty({tokens, 3 * hidden}, bf16_alloc);
    block.hidden_grad_a_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.hidden_grad_b_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.gate_up_grad_ = bf16r2_tensor_arena_t::try_empty({tokens, 2 * ffn}, bf16_alloc);
    block.silu_up_grad_ = bf16r2_tensor_arena_t::try_empty({tokens, ffn}, bf16_alloc);
    block.attn_scale_ = f32r1_tensor_arena_t::try_empty({1}, f32_alloc);
    // cuDNN's ragged offsets are per-operand element offsets; two stride families cover all nine
    // SDPA operands (the packed QKV block's and the dense hidden buffers').
    gpu_arena_alloc<i32_t> i32_alloc {arena};
    block.ragged_qkv_offsets_ = i32r1_tensor_arena_t::try_empty({config.max_segments + 1u}, i32_alloc);
    block.ragged_hidden_offsets_ = i32r1_tensor_arena_t::try_empty({config.max_segments + 1u}, i32_alloc);
    block.ragged_seq_lens_ = i32r1_tensor_arena_t::try_empty({config.max_segments}, i32_alloc);
    if (!block.qkv_grad_ || !block.hidden_grad_a_ || !block.hidden_grad_b_ || !block.gate_up_grad_ ||
        !block.silu_up_grad_ || !block.attn_scale_ || !block.ragged_qkv_offsets_ || !block.ragged_hidden_offsets_ ||
        !block.ragged_seq_lens_)
        return expected<transformer_block_bf16_t>::fail(
            status_t::bad_alloc_k, "backward transients allocation failed (arena_used={})", arena.used());

    // Seed the device SDPA scale — a synchronous setup-path upload, once, outside any capture.
    f32_t const attention_scale = 1.0f / sqrtf(static_cast<f32_t>(transformer_head_dim_k));
    if (CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(block.attn_scale_.data()), &attention_scale,
                                       sizeof(attention_scale));
        result != CUDA_SUCCESS)
        return expected<transformer_block_bf16_t>::fail(result, "SDPA scale upload failed");

    if (expected_t planned = block.plan_matmuls_(arena); !planned)
        return expected<transformer_block_bf16_t>::fail(std::move(planned));
    if (expected_t built = block.build_sdpa_(arena, b8_alloc); !built)
        return expected<transformer_block_bf16_t>::fail(std::move(built));
    return expected<transformer_block_bf16_t> {std::move(block)};
}

expected_t transformer_block_bf16_t::forward(bf16r2_view_t input, bf16r2_span_t output, u32r1_view_t segment_offsets,
                                             u32r1_view_t segment_lengths, rope_angles_view_t angles,
                                             i16r2_view_t coords, gpu_stream_t const &stream) noexcept {
    std::size_t const tokens = input.extent(0);
    u32_t const hidden = config_.hidden;
    u32_t const ffn = config_.ffn_expansion * hidden;
    u32_t const heads = hidden / transformer_head_dim_k;
    if (tokens == 0 || tokens > config_.max_tokens || input.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "input [{}x{}] disagrees with the baked [{}x{}]", tokens,
                                input.extent(1), config_.max_tokens, hidden);
    if (output.extent(0) != tokens || output.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "output shape disagrees with the input");
    if (segment_lengths.extent(0) != config_.max_segments || segment_offsets.extent(0) != config_.max_segments + 1u)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "segment descriptors disagree with the baked count ({} segments)",
                                config_.max_segments);
    if (coords.extent(0) < tokens)
        return expected_t::fail(status_t::invalid_input_k, "rope coordinates cover {} of {} tokens", coords.extent(0),
                                tokens);
    // Every stash's live extent records the step's token count — the shape the backward reads back,
    // never a shadow scalar. Resizing within the reserved capacity moves no address, and the `tokens
    // <= max_tokens` check above guarantees every one of these succeeds.

    bool const resized =
        input_stash_.try_resize({tokens, hidden}) && normalized_.try_resize({tokens, hidden}) &&
        qkv_.try_resize({tokens, 3 * hidden}) && query_prenorm_.try_resize({tokens, hidden}) &&
        key_prenorm_.try_resize({tokens, hidden}) && attn_output_.try_resize({tokens, hidden}) &&
        attn_residual_.try_resize({tokens, hidden}) && mlp_normalized_.try_resize({tokens, hidden}) &&
        gate_up_.try_resize({tokens, 2 * ffn}) && silu_up_.try_resize({tokens, ffn});
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "activation stashes cannot hold {} tokens (baked {})",
                                tokens, config_.max_tokens);

    // Residual stash: the backward's pre-norm derivative needs the raw input.
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(input_stash_.data()),
                                            reinterpret_cast<CUdeviceptr>(input.data()),
                                            tokens * hidden * sizeof(__nv_bfloat16), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "input stash copy failed ({} tokens)", tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const token_grid {static_cast<u32_t>(tokens), 1u, 1u};
    f32_t const epsilon = config_.rms_eps;

    // Pre-attention RMSNorm.
    expected_t launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), token_grid,
                                        block_shape, stream, input, pre_norm_scale_.view(), normalized_.span(),
                                        pre_inverse_rms_.span(), epsilon);
    if (!launched) return launched;
    // Fused QKV projection into the packed [tokens, 3·hidden] block.
    if (expected_t product = plans_[plan_qkv_forward_k].matmul(*blas_, normalized_.data(), qkv_weight_.data(),
                                                               qkv_.data(), 0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    // Unit QK-norm (stashes pre-norm values), then rotation, in place on the packed slices.
    std::size_t const head_vectors = tokens * heads;
    dim3 const vector_grid {
        static_cast<u32_t>(divide_round_up(head_vectors * warp_size_k, kernel_block_size_default_k)), 1u, 1u};
    u32_t const query_offset = 0u, key_offset = hidden;
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_qk_norm_forward_kernel_), vector_grid,
                             block_shape, stream, qkv_.span(), query_prenorm_.span(), query_inverse_rms_.span(),
                             heads, query_offset, epsilon);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_qk_norm_forward_kernel_), vector_grid,
                                 block_shape, stream, qkv_.span(), key_prenorm_.span(), key_inverse_rms_.span(),
                                 heads, key_offset, epsilon);
    if (!launched) return launched;
    f32_t const forward_sign = 1.0f;
    std::size_t const rope_slots = head_vectors * (transformer_head_dim_k >> 1u);
    dim3 const rope_grid {static_cast<u32_t>(divide_round_up(rope_slots, kernel_block_size_default_k)), 1u, 1u};
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape, stream,
                             qkv_.span(), angles, coords, heads, query_offset, forward_sign);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape,
                                 stream, qkv_.span(), angles, coords, heads, key_offset, forward_sign);
    if (!launched) return launched;

    // Fused SDPA over the packed segments, stashing the log-sum-exp for the backward.
    if (expected_t attended = run_sdpa_forward_(segment_offsets, segment_lengths, stream); !attended) return attended;

    // Out-projection, first residual, post-attention RMSNorm.
    if (expected_t product = plans_[plan_out_forward_k].matmul(*blas_, attn_output_.data(), out_weight_.data(),
                                                               attn_residual_.data(), 0.0f, blas_workspace_.span(),
                                                               stream);
        !product)
        return product;
    dim3 const cell_grid {static_cast<u32_t>(divide_round_up(tokens * hidden, kernel_block_size_default_k)), 1u, 1u};
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_residual_add_kernel_), cell_grid, block_shape,
                             stream, attn_residual_.span().flatten<1>(), input.flatten<1>());
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), token_grid, block_shape,
                                 stream, attn_residual_.view(), post_norm_scale_.view(), mlp_normalized_.span(),
                                 post_inverse_rms_.span(), epsilon);
    if (!launched) return launched;

    // SwiGLU MLP: fused gate/up product, SiLU·up, down product into the caller's output, residual.
    if (expected_t product = plans_[plan_gate_up_forward_k].matmul(*blas_, mlp_normalized_.data(),
                                                                   gate_up_weight_.data(), gate_up_.data(), 0.0f,
                                                                   blas_workspace_.span(), stream);
        !product)
        return product;
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_swiglu_forward_kernel_),
                             dim3 {static_cast<u32_t>(divide_round_up(tokens * ffn, kernel_block_size_default_k)), 1u,
                                   1u},
                             block_shape, stream, gate_up_.view(), silu_up_.span());
    if (!launched) return launched;
    if (expected_t product = plans_[plan_down_forward_k].matmul(*blas_, silu_up_.data(), down_weight_.data(),
                                                                output.data(), 0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    return launch_kernel(reinterpret_cast<void const *>(&transformer_residual_add_kernel_), cell_grid, block_shape,
                         stream, output.flatten<1>(), attn_residual_.view().flatten<1>());
}

expected_t transformer_block_bf16_t::backward(bf16r2_view_t output_grad, bf16r2_span_t input_grad,
                                              u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                              rope_angles_view_t angles, i16r2_view_t coords,
                                              gpu_stream_t const &stream) noexcept {
    std::size_t const tokens = input_stash_.extent(0);
    u32_t const hidden = config_.hidden;
    u32_t const ffn = config_.ffn_expansion * hidden;
    u32_t const heads = hidden / transformer_head_dim_k;
    if (tokens == 0) return expected_t::fail(status_t::invalid_input_k, "backward before a stashed forward");
    if (output_grad.extent(0) != tokens || output_grad.extent(1) != hidden || input_grad.extent(0) != tokens ||
        input_grad.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "gradient shapes disagree with the stashed [{}x{}]",
                                tokens, hidden);
    // Backward-only transients the forward never touches — their live extent still needs setting to
    // this step's token count (the stashes above already carry it from the paired forward call).

    bool const resized = qkv_grad_.try_resize({tokens, 3 * hidden}) && hidden_grad_a_.try_resize({tokens, hidden}) &&
                         hidden_grad_b_.try_resize({tokens, hidden}) && gate_up_grad_.try_resize({tokens, 2 * ffn}) &&
                         silu_up_grad_.try_resize({tokens, ffn});
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "gradient transients cannot hold {} tokens (baked {})",
                                tokens, config_.max_tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const token_grid {static_cast<u32_t>(tokens), 1u, 1u};
    dim3 const cell_grid {static_cast<u32_t>(divide_round_up(tokens * hidden, kernel_block_size_default_k)), 1u, 1u};
    dim3 const channel_grid {static_cast<u32_t>(divide_round_up(hidden, kernel_block_size_default_k)), 1u, 1u};

    // MLP backward: down → SiLU·up → gate/up, weight gradients along the way.
    if (expected_t product = plans_[plan_down_data_grad_k].matmul(*blas_, output_grad.data(), down_weight_.data(),
                                                                  silu_up_grad_.data(), 0.0f, blas_workspace_.span(),
                                                                  stream);
        !product)
        return product;
    if (expected_t product = plans_[plan_down_weight_grad_k].matmul(*blas_, output_grad.data(), silu_up_.data(),
                                                                    down_weight_grad_.data(), 0.0f,
                                                                    blas_workspace_.span(), stream);
        !product)
        return product;
    expected_t launched = launch_kernel(
        reinterpret_cast<void const *>(&transformer_swiglu_backward_kernel_),
        dim3 {static_cast<u32_t>(divide_round_up(tokens * ffn, kernel_block_size_default_k)), 1u, 1u}, block_shape,
        stream, gate_up_.view(), silu_up_grad_.view(), gate_up_grad_.span());
    if (!launched) return launched;
    if (expected_t product = plans_[plan_gate_up_data_grad_k].matmul(*blas_, gate_up_grad_.data(),
                                                                     gate_up_weight_.data(), hidden_grad_a_.data(),
                                                                     0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    if (expected_t product = plans_[plan_gate_up_weight_grad_k].matmul(
            *blas_, gate_up_grad_.data(), mlp_normalized_.data(), gate_up_weight_grad_.data(), 0.0f,
            blas_workspace_.span(), stream);
        !product)
        return product;

    // Post-attention RMSNorm backward: token grad accumulates the residual branch (output_grad).
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(hidden_grad_b_.data()),
                                            reinterpret_cast<CUdeviceptr>(output_grad.data()),
                                            tokens * hidden * sizeof(__nv_bfloat16), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "residual gradient copy failed");
    bool const accumulate = true;
    launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_backward_kernel), token_grid, block_shape, stream,
                             attn_residual_.view(), post_norm_scale_.view(), post_inverse_rms_.view(),
                             hidden_grad_a_.view(), hidden_grad_b_.span(), accumulate);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_scale_grad_kernel), channel_grid,
                                 block_shape, stream, attn_residual_.view(), post_inverse_rms_.view(),
                                 hidden_grad_a_.view(), post_norm_scale_grad_.span());
    if (!launched) return launched;
    // The attn-residual gradient (hidden_grad_b_) now carries d(attn_residual).

    // Out-projection backward.
    if (expected_t product = plans_[plan_out_data_grad_k].matmul(*blas_, hidden_grad_b_.data(), out_weight_.data(),
                                                                 hidden_grad_a_.data(), 0.0f, blas_workspace_.span(),
                                                                 stream);
        !product)
        return product;
    if (expected_t product = plans_[plan_out_weight_grad_k].matmul(*blas_, hidden_grad_b_.data(), attn_output_.data(),
                                                                   out_weight_grad_.data(), 0.0f,
                                                                   blas_workspace_.span(), stream);
        !product)
        return product;

    // SDPA backward into the packed d_qkv block, then inverse rotation and QK-norm backward.
    if (expected_t attended = run_sdpa_backward_(segment_offsets, segment_lengths, stream); !attended) return attended;
    f32_t const inverse_sign = -1.0f;
    std::size_t const head_vectors = tokens * heads;
    std::size_t const rope_slots = head_vectors * (transformer_head_dim_k >> 1u);
    dim3 const rope_grid {static_cast<u32_t>(divide_round_up(rope_slots, kernel_block_size_default_k)), 1u, 1u};
    dim3 const vector_grid {
        static_cast<u32_t>(divide_round_up(head_vectors * warp_size_k, kernel_block_size_default_k)), 1u, 1u};
    u32_t const query_offset = 0u, key_offset = hidden;
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape, stream,
                             qkv_grad_.span(), angles, coords, heads, query_offset, inverse_sign);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape,
                                 stream, qkv_grad_.span(), angles, coords, heads, key_offset, inverse_sign);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_qk_norm_backward_kernel_), vector_grid,
                                 block_shape, stream, qkv_grad_.span(), query_prenorm_.view(),
                                 query_inverse_rms_.view(), heads, query_offset);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_qk_norm_backward_kernel_), vector_grid,
                                 block_shape, stream, qkv_grad_.span(), key_prenorm_.view(), key_inverse_rms_.view(),
                                 heads, key_offset);
    if (!launched) return launched;

    // QKV projection backward.
    if (expected_t product = plans_[plan_qkv_data_grad_k].matmul(
            *blas_, qkv_grad_.data(), qkv_weight_.data(), hidden_grad_a_.data(), 0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    if (expected_t product = plans_[plan_qkv_weight_grad_k].matmul(*blas_, qkv_grad_.data(), normalized_.data(),
                                                                   qkv_weight_grad_.data(), 0.0f,
                                                                   blas_workspace_.span(), stream);
        !product)
        return product;

    // Pre-attention RMSNorm backward: token grad accumulates the attn-residual branch.
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(input_grad.data()),
                                            reinterpret_cast<CUdeviceptr>(hidden_grad_b_.data()),
                                            tokens * hidden * sizeof(__nv_bfloat16), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "attention residual gradient copy failed");
    launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_backward_kernel), token_grid, block_shape, stream,
                             input_stash_.view(), pre_norm_scale_.view(), pre_inverse_rms_.view(),
                             hidden_grad_a_.view(), input_grad, accumulate);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_scale_grad_kernel), channel_grid,
                                 block_shape, stream, input_stash_.view(), pre_inverse_rms_.view(),
                                 hidden_grad_a_.view(), pre_norm_scale_grad_.span());
    return launched;
}

expected_t transformer_block_bf16_t::plan_matmuls_(gpu_arena_t &arena) noexcept {
    u32_t const hidden = config_.hidden;
    u32_t const ffn = config_.ffn_expansion * hidden;
    u32_t const tokens = config_.max_tokens;
    std::size_t workspace_bytes = 0;
    std::size_t plan_slot = 0;
    auto pin_plan = [&](cublaslt_plan_config_t plan_config) noexcept -> expected_t {
        plan_config.c_type = CUDA_R_16BF;
        expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(*blas_, plan_config);
        if (!plan) return expected_t::fail(std::move(plan));
        if (plan.value_.workspace_bytes() > workspace_bytes) workspace_bytes = plan.value_.workspace_bytes();
        plans_[plan_slot++] = std::move(plan.value_);
        return {};
    };
    // Forward: Y[T,O] = X[T,I]·W[O,I]ᵀ.
    expected_t pinned = pin_plan(
        {tokens, hidden, hidden, 3u * hidden, hidden, hidden, tokens, 3u * hidden, 3u * hidden, false, true});
    if (pinned)
        pinned = pin_plan({tokens, hidden, hidden, hidden, hidden, hidden, tokens, hidden, hidden, false, true});
    if (pinned)
        pinned = pin_plan({tokens, hidden, hidden, 2u * ffn, hidden, hidden, tokens, 2u * ffn, 2u * ffn, false, true});
    if (pinned) pinned = pin_plan({tokens, ffn, ffn, hidden, ffn, ffn, tokens, hidden, hidden, false, true});
    // Data gradients: dX[T,I] = dY[T,O]·W[O,I].
    if (pinned)
        pinned = pin_plan(
            {tokens, 3u * hidden, 3u * hidden, 3u * hidden, hidden, hidden, tokens, hidden, hidden, false, false});
    if (pinned)
        pinned = pin_plan({tokens, hidden, hidden, hidden, hidden, hidden, tokens, hidden, hidden, false, false});
    if (pinned)
        pinned = pin_plan({tokens, 2u * ffn, 2u * ffn, 2u * ffn, hidden, hidden, tokens, hidden, hidden, false, false});
    if (pinned) pinned = pin_plan({tokens, hidden, hidden, hidden, ffn, ffn, tokens, ffn, ffn, false, false});
    // Weight gradients: dW[O,I] = dY[T,O]ᵀ·X[T,I].
    if (pinned)
        pinned = pin_plan(
            {tokens, 3u * hidden, 3u * hidden, tokens, hidden, hidden, 3u * hidden, hidden, hidden, true, false});
    if (pinned)
        pinned = pin_plan({tokens, hidden, hidden, tokens, hidden, hidden, hidden, hidden, hidden, true, false});
    if (pinned)
        pinned = pin_plan({tokens, 2u * ffn, 2u * ffn, tokens, hidden, hidden, 2u * ffn, hidden, hidden, true, false});
    if (pinned) pinned = pin_plan({tokens, hidden, hidden, tokens, ffn, ffn, hidden, ffn, ffn, true, false});
    if (!pinned) return pinned;
    if (workspace_bytes != 0) {
        gpu_arena_alloc<b8_t> b8_alloc {arena};
        blas_workspace_ = b8r1_tensor_arena_t::try_empty({workspace_bytes}, b8_alloc);
        if (!blas_workspace_)
            return expected_t::fail(status_t::bad_alloc_k, "matmul workspace allocation failed ({} bytes)",
                                    workspace_bytes);
    }
    return {};
}

expected_t transformer_block_bf16_t::build_sdpa_(gpu_arena_t &arena, gpu_arena_alloc<b8_t> b8_alloc) noexcept {
    i64_t const segments = config_.max_segments;
    i64_t const seq = config_.max_segment_tokens;
    i64_t const heads = config_.hidden / transformer_head_dim_k;
    i64_t const dim = transformer_head_dim_k;
    i64_t const hidden_stride = static_cast<i64_t>(config_.hidden);
    i64_t const token_stride = 3 * hidden_stride;
    u32_t const hidden = config_.hidden;
    auto const strides = sdpa_thd_strides(heads, token_stride);
    auto const log_sum_exp_strides = std::vector<i64_t> {heads * seq, seq, 1, 1}; // vendor-boundary
    // The scale rides a device scalar tensor we own — never the frontend's pass-by-value cache,
    // which lives behind private members the raw execution path cannot reach.
    auto make_scale = [&](fe::graph::Graph &graph) {
        return graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("attn_scale")
                                .set_uid(sdpa_uid_attn_scale_k)
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::FLOAT));
    };

    // Forward graph: owned bindings pinned now, per-call slots recorded for the dispatch stores.
    {
        fe::graph::Graph &graph = forward_sdpa_.graph;
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto make_operand = [&](i64_t uid, char const *name) {
            return graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name(name)
                                    .set_uid(uid)
                                    .set_dim({segments, heads, seq, dim})
                                    .set_stride(strides));
        };
        auto queries = make_operand(sdpa_uid_query_k, "Q");
        auto keys = make_operand(sdpa_uid_key_k, "K");
        auto values = make_operand(sdpa_uid_value_k, "V");
        queries->set_ragged_offset(sdpa_ragged_offset(graph, sdpa_uid_ragged_base_k + 0, segments));
        keys->set_ragged_offset(sdpa_ragged_offset(graph, sdpa_uid_ragged_base_k + 1, segments));
        values->set_ragged_offset(sdpa_ragged_offset(graph, sdpa_uid_ragged_base_k + 2, segments));
        auto seq_q = sdpa_seq_lengths(graph, sdpa_uid_seq_len_q_k, "seq_len_q", segments);
        auto seq_kv = sdpa_seq_lengths(graph, sdpa_uid_seq_len_kv_k, "seq_len_kv", segments);
        auto options = fe::graph::SDPA_attributes()
                           .set_name("uform_block_sdpa_fwd")
                           .set_generate_stats(true)
                           .set_attn_scale(make_scale(graph))
                           .set_padding_mask(true)
                           .set_seq_len_q(seq_q)
                           .set_seq_len_kv(seq_kv);
        auto [output, stats] = graph.sdpa(queries, keys, values, options);
        // The attention output is a dense [tokens, hidden] buffer of its own (token stride = hidden).
        output->set_output(true)
            .set_uid(sdpa_uid_output_k)
            .set_dim({segments, heads, seq, dim})
            .set_stride(sdpa_thd_strides(heads, hidden_stride));
        output->set_ragged_offset(sdpa_ragged_offset(graph, sdpa_uid_ragged_base_k + 3, segments));
        stats->set_output(true)
            .set_uid(sdpa_uid_stats_k)
            .set_dim({segments, heads, seq, 1})
            .set_stride(log_sum_exp_strides)
            .set_data_type(fe::DataType_t::FLOAT);
        if (auto status = graph.build(cudnn_->get(), {fe::HeurMode_t::A}); status.is_bad())
            return expected_t::fail(status_t::cuda_error_k, "SDPA forward graph build failed: {}",
                                    status.get_message().c_str()); // vendor-boundary
        forward_sdpa_.bind(sdpa_uid_query_k, qkv_.data());
        forward_sdpa_.bind(sdpa_uid_key_k, qkv_.data() + hidden);
        forward_sdpa_.bind(sdpa_uid_value_k, qkv_.data() + 2u * hidden);
        forward_sdpa_.bind(sdpa_uid_output_k, attn_output_.data());
        forward_sdpa_.bind(sdpa_uid_stats_k, softmax_log_sum_exp_.data());
        forward_sdpa_.bind(sdpa_uid_attn_scale_k, attn_scale_.data());
        forward_seq_slot_ = forward_sdpa_.bind(sdpa_uid_seq_len_q_k);
        forward_sdpa_.bind(sdpa_uid_seq_len_kv_k);
        forward_ragged_slot_ = forward_sdpa_.bind(sdpa_uid_ragged_base_k + 0);
        forward_sdpa_.bind(sdpa_uid_ragged_base_k + 1);
        forward_sdpa_.bind(sdpa_uid_ragged_base_k + 2);
        forward_sdpa_.bind(sdpa_uid_ragged_base_k + 3);
        if (expected_t finalized = forward_sdpa_.finalize_dispatch(); !finalized) return finalized;
    }
    // Backward graph.
    {
        fe::graph::Graph &graph = backward_sdpa_.graph;
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto make_operand = [&](i64_t uid, char const *name, i64_t operand_token_stride, i64_t ragged_uid) {
            auto node = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name(name)
                                         .set_uid(uid)
                                         .set_dim({segments, heads, seq, dim})
                                         .set_stride(sdpa_thd_strides(heads, operand_token_stride)));
            node->set_ragged_offset(sdpa_ragged_offset(graph, ragged_uid, segments));
            return node;
        };
        auto queries = make_operand(sdpa_uid_query_k, "Q", token_stride, sdpa_uid_ragged_base_k + 0);
        auto keys = make_operand(sdpa_uid_key_k, "K", token_stride, sdpa_uid_ragged_base_k + 1);
        auto values = make_operand(sdpa_uid_value_k, "V", token_stride, sdpa_uid_ragged_base_k + 2);
        auto output = make_operand(sdpa_uid_output_k, "O", hidden_stride, sdpa_uid_ragged_base_k + 3);
        auto output_grad = make_operand(sdpa_uid_output_grad_k, "dO", hidden_stride, sdpa_uid_ragged_base_k + 4);
        auto stats = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_uid(sdpa_uid_stats_k)
                                      .set_dim({segments, heads, seq, 1})
                                      .set_stride(log_sum_exp_strides)
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto seq_q = sdpa_seq_lengths(graph, sdpa_uid_seq_len_q_k, "seq_len_q", segments);
        auto seq_kv = sdpa_seq_lengths(graph, sdpa_uid_seq_len_kv_k, "seq_len_kv", segments);
        auto options = fe::graph::SDPA_backward_attributes()
                           .set_name("uform_block_sdpa_bwd")
                           .set_attn_scale(make_scale(graph))
                           .set_padding_mask(true)
                           .set_seq_len_q(seq_q)
                           .set_seq_len_kv(seq_kv);
        auto [query_grad, key_grad, value_grad] = graph.sdpa_backward(queries, keys, values, output, output_grad, stats,
                                                                      options);
        auto configure_grad_output = [&](auto &node, i64_t uid, i64_t ragged_uid) {
            node->set_output(true).set_uid(uid).set_dim({segments, heads, seq, dim}).set_stride(strides);
            node->set_ragged_offset(sdpa_ragged_offset(graph, ragged_uid, segments));
        };
        configure_grad_output(query_grad, sdpa_uid_query_grad_k, sdpa_uid_ragged_base_k + 5);
        configure_grad_output(key_grad, sdpa_uid_key_grad_k, sdpa_uid_ragged_base_k + 6);
        configure_grad_output(value_grad, sdpa_uid_value_grad_k, sdpa_uid_ragged_base_k + 7);
        if (auto status = graph.build(cudnn_->get(), {fe::HeurMode_t::A}); status.is_bad())
            return expected_t::fail(status_t::cuda_error_k, "SDPA backward graph build failed: {}",
                                    status.get_message().c_str()); // vendor-boundary
        backward_sdpa_.bind(sdpa_uid_query_k, qkv_.data());
        backward_sdpa_.bind(sdpa_uid_key_k, qkv_.data() + hidden);
        backward_sdpa_.bind(sdpa_uid_value_k, qkv_.data() + 2u * hidden);
        backward_sdpa_.bind(sdpa_uid_output_k, attn_output_.data());
        backward_sdpa_.bind(sdpa_uid_output_grad_k, hidden_grad_a_.data());
        backward_sdpa_.bind(sdpa_uid_stats_k, softmax_log_sum_exp_.data());
        backward_sdpa_.bind(sdpa_uid_query_grad_k, qkv_grad_.data());
        backward_sdpa_.bind(sdpa_uid_key_grad_k, qkv_grad_.data() + hidden);
        backward_sdpa_.bind(sdpa_uid_value_grad_k, qkv_grad_.data() + 2u * hidden);
        backward_sdpa_.bind(sdpa_uid_attn_scale_k, attn_scale_.data());
        backward_seq_slot_ = backward_sdpa_.bind(sdpa_uid_seq_len_q_k);
        backward_sdpa_.bind(sdpa_uid_seq_len_kv_k);
        backward_ragged_slot_ = backward_sdpa_.bind(sdpa_uid_ragged_base_k + 0);
        for (i64_t ragged = 1; ragged != 8; ++ragged) backward_sdpa_.bind(sdpa_uid_ragged_base_k + ragged);
        if (expected_t finalized = backward_sdpa_.finalize_dispatch(); !finalized) return finalized;
    }
    i64_t const sdpa_workspace = forward_sdpa_.workspace_bytes > backward_sdpa_.workspace_bytes
                                     ? forward_sdpa_.workspace_bytes
                                     : backward_sdpa_.workspace_bytes;
    if (sdpa_workspace > 0) {
        sdpa_workspace_ = b8r1_tensor_arena_t::try_empty({static_cast<std::size_t>(sdpa_workspace)}, b8_alloc);
        if (!sdpa_workspace_)
            return expected_t::fail(status_t::bad_alloc_k, "SDPA workspace allocation failed ({} bytes)",
                                    sdpa_workspace);
    }
    return {};
}

expected_t transformer_block_bf16_t::stage_ragged_offsets_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                           gpu_stream_t const &stream) noexcept {
    return launch_kernel(reinterpret_cast<void const *>(&transformer_ragged_offsets_kernel_), dim3 {1u, 1u, 1u},
                         dim3 {warp_size_k, 1u, 1u}, stream, segment_offsets, segment_lengths,
                         ragged_qkv_offsets_.span(), ragged_hidden_offsets_.span(), ragged_seq_lens_.span(),
                         config_.hidden);
}

expected_t transformer_block_bf16_t::run_sdpa_forward_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                       gpu_stream_t const &stream) noexcept {
    if (expected_t bound = cudnn_->bind(stream); !bound) return bound;
    if (expected_t staged = stage_ragged_offsets_(segment_offsets, segment_lengths, stream); !staged) return staged;
    forward_sdpa_.pointers[forward_seq_slot_] = ragged_seq_lens_.data();
    forward_sdpa_.pointers[forward_seq_slot_ + 1u] = ragged_seq_lens_.data();
    for (u32_t ragged = 0; ragged != 3u; ++ragged)
        forward_sdpa_.pointers[forward_ragged_slot_ + ragged] = ragged_qkv_offsets_.data();
    forward_sdpa_.pointers[forward_ragged_slot_ + 3u] = ragged_hidden_offsets_.data();
    return forward_sdpa_.execute(cudnn_->get(), sdpa_workspace_.data());
}

expected_t transformer_block_bf16_t::run_sdpa_backward_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                        gpu_stream_t const &stream) noexcept {
    if (expected_t bound = cudnn_->bind(stream); !bound) return bound;
    if (expected_t staged = stage_ragged_offsets_(segment_offsets, segment_lengths, stream); !staged) return staged;
    backward_sdpa_.pointers[backward_seq_slot_] = ragged_seq_lens_.data();
    backward_sdpa_.pointers[backward_seq_slot_ + 1u] = ragged_seq_lens_.data();
    i32_t *const qkv_offsets = ragged_qkv_offsets_.data();
    i32_t *const hidden_offsets = ragged_hidden_offsets_.data();
    i32_t *const per_operand[8] = {qkv_offsets,    qkv_offsets, qkv_offsets, hidden_offsets,
                                   hidden_offsets, qkv_offsets, qkv_offsets, qkv_offsets};
    for (u32_t ragged = 0; ragged != 8u; ++ragged)
        backward_sdpa_.pointers[backward_ragged_slot_ + ragged] = per_operand[ragged];
    return backward_sdpa_.execute(cudnn_->get(), sdpa_workspace_.data());
}

expected_t transformer_block_e4m3_t::forward(bf16r2_view_t input, bf16r2_span_t output, u32r1_view_t segment_offsets,
                                             u32r1_view_t segment_lengths, rope_angles_view_t angles,
                                             i16r2_view_t coords, gpu_stream_t const &stream) noexcept {
    std::size_t const tokens = input.extent(0);
    u32_t const hidden = config_.hidden;
    u32_t const ffn = config_.ffn_expansion * hidden;
    u32_t const heads = hidden / transformer_head_dim_k;
    if (tokens == 0 || tokens > config_.max_tokens || input.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "input [{}x{}] disagrees with the baked [{}x{}]", tokens,
                                input.extent(1), config_.max_tokens, hidden);
    if (output.extent(0) != tokens || output.extent(1) != hidden)
        return expected_t::fail(status_t::shape_unsupported_k, "output shape disagrees with the input");
    // No backward reads any of these, but each still needs its live extent set before `.span()` /
    // `.view()` reflects this step's token count rather than the baked `max_tokens` capacity.

    bool const resized =
        normalized_.try_resize({tokens, hidden}) && qkv_.try_resize({tokens, 3 * hidden}) &&
        attn_residual_.try_resize({tokens, hidden}) && gate_up_.try_resize({tokens, 2 * ffn}) &&
        silu_up_.try_resize({tokens, ffn}) && prenorm_scratch_.try_resize({tokens, hidden}) &&
        normalized_e4m3_.try_resize({tokens, hidden}) && attn_output_e4m3_.try_resize({tokens, hidden}) &&
        silu_up_e4m3_.try_resize({tokens, ffn});
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "activation stashes cannot hold {} tokens (baked {})",
                                tokens, config_.max_tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const token_grid {static_cast<u32_t>(tokens), 1u, 1u};
    dim3 const cell_grid {static_cast<u32_t>(divide_round_up(tokens * hidden, kernel_block_size_default_k)), 1u, 1u};
    f32_t const epsilon = config_.rms_eps;
    f32r1_view_t const token_scales_view = token_scales_.view();

    // Pre-attention RMSNorm in BFloat16, per-token E4M3 quantisation, QKV product, scale fold.
    expected_t launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), token_grid,
                                        block_shape, stream, input, pre_norm_scale_.view(), normalized_.span(),
                                        inverse_rms_.span(), epsilon);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_quantize_kernel_), token_grid,
                                 block_shape, stream, normalized_.view(), normalized_e4m3_.span(),
                                 token_scales_.span());
    if (!launched) return launched;
    if (expected_t product = plans_[plan_qkv_k].matmul(*blas_, normalized_e4m3_.data(), qkv_weight_e4m3_.data(),
                                                       qkv_.data(), 0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    launched = launch_kernel(
        reinterpret_cast<void const *>(&transformer_token_rescale_kernel_),
        dim3 {static_cast<u32_t>(divide_round_up(tokens * 3u * hidden, kernel_block_size_default_k)), 1u, 1u},
        block_shape, stream, qkv_.span(), token_scales_view);
    if (!launched) return launched;
    // Unit QK-norm (no stash consumers — reuse the norm kernel with throwaway stash buffers) + M-RoPE.
    std::size_t const head_vectors = tokens * heads;
    dim3 const vector_grid {
        static_cast<u32_t>(divide_round_up(head_vectors * warp_size_k, kernel_block_size_default_k)), 1u, 1u};
    u32_t const query_offset = 0u, key_offset = hidden;
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_qk_norm_forward_kernel_), vector_grid,
                             block_shape, stream, qkv_.span(), prenorm_scratch_.span(), inverse_rms_heads_.span(),
                             heads, query_offset, epsilon);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_qk_norm_forward_kernel_), vector_grid,
                                 block_shape, stream, qkv_.span(), prenorm_scratch_.span(), inverse_rms_heads_.span(),
                                 heads, key_offset, epsilon);
    if (!launched) return launched;
    f32_t const forward_sign = 1.0f;
    std::size_t const rope_slots = head_vectors * (transformer_head_dim_k >> 1u);
    dim3 const rope_grid {static_cast<u32_t>(divide_round_up(rope_slots, kernel_block_size_default_k)), 1u, 1u};
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape, stream,
                             qkv_.span(), angles, coords, heads, query_offset, forward_sign);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape,
                                 stream, qkv_.span(), angles, coords, heads, key_offset, forward_sign);
    if (!launched) return launched;

    // Pack the rotated Q / K and raw V slices into dense zero-padded E4M3 operands under one dynamic
    // per-tensor scale (the FP8 SDPA's own granularity), then run the dense per-segment attention.
    // `qkv_e4m3_` / `attn_dense_e4m3_` stay flat rank-1 arenas — no tensor owns the Q|K|V role split —
    // so each role's `[segments,seq_max,hidden]` operand is a manual offset + shape, mirroring
    // `rope_angles_t::view()`'s per-axis slices.

    std::size_t const segments = config_.max_segments;
    std::size_t const seq_max = config_.max_segment_tokens;
    std::size_t const dense_role_scalars = segments * seq_max * hidden;
    if (CUresult result = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(max_bits_.data()), 0, sizeof(u32_t),
                                          stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "dynamic-scale reset failed");
    launched = launch_kernel(
        reinterpret_cast<void const *>(&transformer_absolute_max_kernel_),
        dim3 {static_cast<u32_t>(divide_round_up(tokens * 3u * hidden, kernel_block_size_default_k)), 1u, 1u},
        block_shape, stream, qkv_.view().flatten<1>(), max_bits_.span());
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_descale_from_max_kernel_),
                                 dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u}, stream, max_bits_.span(), qkv_descale_.span());
    dim3 const dense_grid {static_cast<u32_t>(divide_round_up(dense_role_scalars, kernel_block_size_default_k)), 1u,
                           1u};
    for (u32_t role = 0; launched && role != 3u; ++role) {
        e4m3r3_span_t const dense_slice {qkv_e4m3_.data() + role * dense_role_scalars,
                                         {segments, seq_max, static_cast<std::size_t>(hidden)}};
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_pack_dense_kernel_), dense_grid,
                                 block_shape, stream, qkv_.view(), qkv_descale_.view(), segment_offsets,
                                 segment_lengths, dense_slice, role * hidden);
    }
    if (!launched) return launched;
    if (expected_t attended = run_sdpa_(segment_lengths, stream); !attended) return attended;
    // Unpack the dense E4M3 attention output back to the packed layout the out-projection reads.
    e4m3r3_view_t const attn_dense_view {attn_dense_e4m3_.data(),
                                         {segments, seq_max, static_cast<std::size_t>(hidden)}};
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_unpack_dense_kernel_), dense_grid, block_shape,
                             stream, attn_dense_view, attn_output_e4m3_.span(), segment_offsets, segment_lengths);
    if (!launched) return launched;

    // Out-projection consumes the unpacked E4M3 attention output; the SDPA's `scale_o` is the unit,
    // so no activation fold is owed here.
    if (expected_t product = plans_[plan_out_k].matmul(*blas_, attn_output_e4m3_.data(), out_weight_e4m3_.data(),
                                                       attn_residual_.data(), 0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_residual_add_kernel_), cell_grid, block_shape,
                             stream, attn_residual_.span().flatten<1>(), input.flatten<1>());
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&rmsnorm_forward_kernel), token_grid, block_shape,
                                 stream, attn_residual_.view(), post_norm_scale_.view(), normalized_.span(),
                                 inverse_rms_.span(), epsilon);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_quantize_kernel_), token_grid,
                                 block_shape, stream, normalized_.view(), normalized_e4m3_.span(),
                                 token_scales_.span());
    if (!launched) return launched;
    if (expected_t product = plans_[plan_gate_up_k].matmul(*blas_, normalized_e4m3_.data(), gate_up_weight_e4m3_.data(),
                                                           gate_up_.data(), 0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    dim3 const ffn_grid {static_cast<u32_t>(divide_round_up(tokens * ffn, kernel_block_size_default_k)), 1u, 1u};
    launched = launch_kernel(
        reinterpret_cast<void const *>(&transformer_token_rescale_kernel_),
        dim3 {static_cast<u32_t>(divide_round_up(tokens * 2u * ffn, kernel_block_size_default_k)), 1u, 1u}, block_shape,
        stream, gate_up_.span(), token_scales_view);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_swiglu_forward_kernel_), ffn_grid,
                                 block_shape, stream, gate_up_.view(), silu_up_.span());
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_quantize_kernel_), token_grid,
                                 block_shape, stream, silu_up_.view(), silu_up_e4m3_.span(), token_scales_.span());
    if (!launched) return launched;
    if (expected_t product = plans_[plan_down_k].matmul(*blas_, silu_up_e4m3_.data(), down_weight_e4m3_.data(),
                                                        output.data(), 0.0f, blas_workspace_.span(), stream);
        !product)
        return product;
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_rescale_kernel_), cell_grid, block_shape,
                             stream, output, token_scales_view);
    if (!launched) return launched;
    return launch_kernel(reinterpret_cast<void const *>(&transformer_residual_add_kernel_), cell_grid, block_shape,
                         stream, output.flatten<1>(), attn_residual_.view().flatten<1>());
}

expected_t transformer_block_e4m3_t::run_sdpa_(u32r1_view_t segment_lengths, gpu_stream_t const &stream) noexcept {
    if (expected_t bound = cudnn_->bind(stream); !bound) return bound;
    if (expected_t staged = launch_kernel(reinterpret_cast<void const *>(&transformer_seq_lens_cast_kernel_),
                                          dim3 {1u, 1u, 1u}, dim3 {warp_size_k, 1u, 1u}, stream, segment_lengths,
                                          ragged_seq_lens_.span());
        !staged)
        return staged;
    sdpa_.pointers[seq_slot_] = ragged_seq_lens_.data();
    sdpa_.pointers[seq_slot_ + 1u] = ragged_seq_lens_.data();
    return sdpa_.execute(cudnn_->get(), sdpa_workspace_.data());
}

expected<transformer_block_e4m3_t> transformer_block_bf16_t::quantize_weights(gpu_arena_t &arena,
                                                                              gpu_stream_t const &stream) noexcept {
    using result_t = expected<transformer_block_e4m3_t>;
    transformer_block_e4m3_t block;
    block.config_ = config_;
    block.blas_ = blas_;
    block.cudnn_ = cudnn_;

    std::size_t const hidden = config_.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config_.ffn_expansion) * hidden;
    std::size_t const tokens = config_.max_tokens;
    std::size_t const heads = hidden / transformer_head_dim_k;
    gpu_arena_alloc<e4m3_t> e4m3_alloc {arena};
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};
    // Plain `try_empty` allocations, checked in named groups so a failure names the arena pressure.

    // Static FP8 weights, their descales, and the BFloat16 norm scales.
    u32r1_tensor_arena_t max_bits {};
    block.qkv_weight_e4m3_ = e4m3r2_tensor_arena_t::try_empty({3 * hidden, hidden}, e4m3_alloc);
    block.out_weight_e4m3_ = e4m3r2_tensor_arena_t::try_empty({hidden, hidden}, e4m3_alloc);
    block.gate_up_weight_e4m3_ = e4m3r2_tensor_arena_t::try_empty({2 * ffn, hidden}, e4m3_alloc);
    block.down_weight_e4m3_ = e4m3r2_tensor_arena_t::try_empty({hidden, ffn}, e4m3_alloc);
    block.weight_descales_ = f32r1_tensor_arena_t::try_empty({4}, f32_alloc);
    block.pre_norm_scale_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    block.post_norm_scale_ = bf16r1_tensor_arena_t::try_empty({hidden}, bf16_alloc);
    if (!block.qkv_weight_e4m3_ || !block.out_weight_e4m3_ || !block.gate_up_weight_e4m3_ || !block.down_weight_e4m3_ ||
        !block.weight_descales_ || !block.pre_norm_scale_ || !block.post_norm_scale_)
        return result_t::fail(status_t::bad_alloc_k, "weights allocation failed (arena_used={})", arena.used());
    // BFloat16 activation stashes shared with the BFloat16 arm's layout.
    block.normalized_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.qkv_ = bf16r2_tensor_arena_t::try_empty({tokens, 3 * hidden}, bf16_alloc);
    block.attn_output_e4m3_ = e4m3r2_tensor_arena_t::try_empty({tokens, hidden}, e4m3_alloc);
    block.attn_residual_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    block.gate_up_ = bf16r2_tensor_arena_t::try_empty({tokens, 2 * ffn}, bf16_alloc);
    block.silu_up_ = bf16r2_tensor_arena_t::try_empty({tokens, ffn}, bf16_alloc);
    block.prenorm_scratch_ = bf16r2_tensor_arena_t::try_empty({tokens, hidden}, bf16_alloc);
    if (!block.normalized_ || !block.qkv_ || !block.attn_output_e4m3_ || !block.attn_residual_ || !block.gate_up_ ||
        !block.silu_up_ || !block.prenorm_scratch_)
        return result_t::fail(status_t::bad_alloc_k, "stashes allocation failed (arena_used={})", arena.used());
    // FP8 shadows, scales, and quantisation scratch. `qkv_e4m3_` / `attn_dense_e4m3_` stay flat rank-1
    // arenas — no tensor owns the Q|K|V role split — sliced per role at the call site (see forward()).
    block.normalized_e4m3_ = e4m3r2_tensor_arena_t::try_empty({tokens, hidden}, e4m3_alloc);
    std::size_t const dense_role_scalars = static_cast<std::size_t>(config_.max_segments) * config_.max_segment_tokens *
                                           hidden;
    block.qkv_e4m3_ = e4m3r1_tensor_arena_t::try_empty({3 * dense_role_scalars}, e4m3_alloc);
    block.attn_dense_e4m3_ = e4m3r1_tensor_arena_t::try_empty({dense_role_scalars}, e4m3_alloc);
    block.silu_up_e4m3_ = e4m3r2_tensor_arena_t::try_empty({tokens, ffn}, e4m3_alloc);
    block.token_scales_ = f32r1_tensor_arena_t::try_empty({tokens}, f32_alloc);
    block.qkv_descale_ = f32r1_tensor_arena_t::try_empty({1}, f32_alloc);
    block.max_bits_ = u32r1_tensor_arena_t::try_empty({1}, u32_alloc);
    block.inverse_rms_ = f32r1_tensor_arena_t::try_empty({tokens}, f32_alloc);
    block.inverse_rms_heads_ = f32r1_tensor_arena_t::try_empty({tokens * heads}, f32_alloc);
    block.ones_ = f32r1_tensor_arena_t::try_empty({1}, f32_alloc);
    block.absolute_max_scratch_ = f32r1_tensor_arena_t::try_empty({2}, f32_alloc);
    gpu_arena_alloc<i32_t> i32_alloc {arena};
    block.ragged_seq_lens_ = i32r1_tensor_arena_t::try_empty({config_.max_segments}, i32_alloc);
    max_bits = u32r1_tensor_arena_t::try_empty({1}, u32_alloc);
    if (!block.normalized_e4m3_ || !block.qkv_e4m3_ || !block.attn_dense_e4m3_ || !block.silu_up_e4m3_ ||
        !block.token_scales_ || !block.qkv_descale_ || !block.max_bits_ || !block.inverse_rms_ ||
        !block.inverse_rms_heads_ || !block.ones_ || !block.absolute_max_scratch_ || !block.ragged_seq_lens_ ||
        !max_bits)
        return result_t::fail(status_t::bad_alloc_k, "quantized operands allocation failed (arena_used={})",
                              arena.used());

    // Norm scales copy verbatim (γ stays BFloat16 in every arm); the device one seeds the FP8 SDPA scales.
    if (cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(block.pre_norm_scale_.data()),
                          reinterpret_cast<CUdeviceptr>(pre_norm_scale_.data()), hidden * sizeof(__nv_bfloat16),
                          stream.get()) != CUDA_SUCCESS ||
        cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(block.post_norm_scale_.data()),
                          reinterpret_cast<CUdeviceptr>(post_norm_scale_.data()), hidden * sizeof(__nv_bfloat16),
                          stream.get()) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "norm scale copy failed");
    f32_t const host_one = 1.0f;
    if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(block.ones_.data()), &host_one, sizeof(host_one),
                          stream.get()) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "unit scale upload failed");

    // Per-weight static quantisation: order-free max-abs, descale = max_abs/240, divide-and-cast. Every
    // weight is a rank-2 [out,in] matrix; the quantisation itself is purely elementwise, so each job
    // flattens to the rank-1 handle the house kernels take.
    struct quantize_job_t {
        bf16r2_tensor_arena_t const *source;
        e4m3r2_tensor_arena_t *shadow;
        u32_t descale_slot;
    };
    quantize_job_t const jobs[4] = {{&qkv_weight_, &block.qkv_weight_e4m3_, 0u},
                                    {&out_weight_, &block.out_weight_e4m3_, 1u},
                                    {&gate_up_weight_, &block.gate_up_weight_e4m3_, 2u},
                                    {&down_weight_, &block.down_weight_e4m3_, 3u}};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    for (quantize_job_t const &job : jobs) {
        std::size_t const scalars = job.source->numel();
        dim3 const grid {static_cast<u32_t>(divide_round_up(scalars, kernel_block_size_default_k)), 1u, 1u};
        if (cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(max_bits.data()), 0, sizeof(u32_t), stream.get()) !=
            CUDA_SUCCESS)
            return result_t::fail(status_t::cuda_error_k, "max-abs reset failed");
        bf16r1_view_t const source_flat = job.source->view().flatten<1>();
        expected_t launched = launch_kernel(reinterpret_cast<void const *>(&transformer_absolute_max_kernel_), grid,
                                            block_shape, stream, source_flat, max_bits.span());
        f32r1_span_t const descale_slot {block.weight_descales_.data() + job.descale_slot, 1};
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&transformer_descale_from_max_kernel_),
                                     dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u}, stream, max_bits.span(), descale_slot);
        if (launched) {
            f32r1_view_t const descale_view = descale_slot;
            e4m3r1_span_t const shadow_flat = job.shadow->span().flatten<1>();
            launched = launch_kernel(reinterpret_cast<void const *>(&transformer_quantize_e4m3_kernel_), grid,
                                     block_shape, stream, source_flat, shadow_flat, descale_view);
        }
        if (!launched) return result_t::fail(std::move(launched));
    }

    // Pin the four FP8 matmul plans with their native weight-descale pointers.
    u32_t const hidden32 = config_.hidden;
    u32_t const ffn32 = config_.ffn_expansion * hidden32;
    u32_t const tokens32 = config_.max_tokens;
    std::size_t workspace_bytes = 0;
    std::size_t plan_slot = 0;
    auto pin_plan = [&](u32_t out_width, u32_t in_width, u32_t descale_slot) noexcept -> expected_t {
        cublaslt_plan_config_t plan_config {tokens32, in_width,  in_width,  out_width, in_width, in_width,
                                            tokens32, out_width, out_width, false,     true};
        plan_config.a_type = CUDA_R_8F_E4M3;
        plan_config.b_type = CUDA_R_8F_E4M3;
        plan_config.c_type = CUDA_R_16BF;
        plan_config.b_descale = block.weight_descales_.data() + descale_slot;
        expected<cublaslt_plan_t> plan = cublaslt_plan_t::make(*blas_, plan_config);
        if (!plan) return expected_t::fail(std::move(plan));
        if (plan.value_.workspace_bytes() > workspace_bytes) workspace_bytes = plan.value_.workspace_bytes();
        block.plans_[plan_slot++] = std::move(plan.value_);
        return {};
    };
    expected_t pinned = pin_plan(3u * hidden32, hidden32, 0u);
    if (pinned) pinned = pin_plan(hidden32, hidden32, 1u);
    if (pinned) pinned = pin_plan(2u * ffn32, hidden32, 2u);
    if (pinned) pinned = pin_plan(hidden32, ffn32, 3u);
    if (!pinned) return result_t::fail(std::move(pinned));
    if (workspace_bytes != 0) {
        block.blas_workspace_ = b8r1_tensor_arena_t::try_empty({workspace_bytes}, b8_alloc);
        if (!block.blas_workspace_)
            return result_t::fail(status_t::bad_alloc_k, "quantized matmul workspace allocation failed ({} bytes)",
                                  workspace_bytes);
    }

    // The FP8 SDPA graph: dense BSHD per segment row (the FP8 engines reject ragged offsets), padding
    // masked by the per-segment lengths, output emitted straight in E4M3. The attention scale rides
    // descale_q (descales compose multiplicatively), so no separate scale scalar exists anywhere.
    {
        i64_t const segments = config_.max_segments;
        i64_t const seq = config_.max_segment_tokens;
        i64_t const heads64 = static_cast<i64_t>(heads);
        i64_t const dim = transformer_head_dim_k;
        fe::graph::Graph &graph = block.sdpa_.graph;
        graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto dense = std::vector<i64_t> {seq * heads64 * dim, dim, heads64 * dim, 1}; // vendor-boundary
        auto make_operand = [&](i64_t uid, char const *name) {
            return graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name(name)
                                    .set_uid(uid)
                                    .set_dim({segments, heads64, seq, dim})
                                    .set_stride(dense));
        };
        auto make_scalar = [&](i64_t uid, char const *name) {
            return graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name(name)
                                    .set_uid(uid)
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));
        };
        auto queries = make_operand(sdpa_uid_query_k, "Q");
        auto keys = make_operand(sdpa_uid_key_k, "K");
        auto values = make_operand(sdpa_uid_value_k, "V");
        auto descale_q = make_scalar(sdpa_uid_descale_q_k, "descale_q");
        auto descale_k = make_scalar(sdpa_uid_descale_k_k, "descale_k");
        auto descale_v = make_scalar(sdpa_uid_descale_v_k, "descale_v");
        auto descale_s = make_scalar(sdpa_uid_descale_s_k, "descale_s");
        auto scale_s = make_scalar(sdpa_uid_scale_s_k, "scale_s");
        auto scale_o = make_scalar(sdpa_uid_scale_o_k, "scale_o");
        auto seq_q = sdpa_seq_lengths(graph, sdpa_uid_seq_len_q_k, "seq_len_q", segments);
        auto seq_kv = sdpa_seq_lengths(graph, sdpa_uid_seq_len_kv_k, "seq_len_kv", segments);
        // The FP8 attribute set accepts only a float scale (its tensor overload is disabled in the
        // frontend), which makes it a pass-by-value tensor — the finalize below arms the frontend
        // fallback for this graph rather than the raw path.
        f32_t const attention_scale = 1.0f / sqrtf(static_cast<f32_t>(transformer_head_dim_k));
        auto options = fe::graph::SDPA_fp8_attributes()
                           .set_name("uform_block_sdpa_fp8")
                           .set_generate_stats(false)
                           .set_attn_scale(attention_scale)
                           .set_padding_mask(true)
                           .set_seq_len_q(seq_q)
                           .set_seq_len_kv(seq_kv);
        auto [output, stats, amax_s, amax_o] = graph.sdpa_fp8(queries, keys, values, descale_q, descale_k, descale_v,
                                                              descale_s, scale_s, scale_o, options);
        (void)stats; // generate_stats is false
        output->set_output(true)
            .set_uid(sdpa_uid_output_k)
            .set_dim({segments, heads64, seq, dim})
            .set_stride(dense)
            .set_data_type(fe::DataType_t::FP8_E4M3);
        amax_s->set_output(true)
            .set_uid(sdpa_uid_amax_s_k)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
        amax_o->set_output(true)
            .set_uid(sdpa_uid_amax_o_k)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
        if (auto status = graph.build(cudnn_->get(), {fe::HeurMode_t::A}); status.is_bad())
            return result_t::fail(status_t::cuda_error_k, "FP8 SDPA graph build failed: {}",
                                  status.get_message().c_str()); // vendor-boundary
        std::size_t const role_scalars = static_cast<std::size_t>(config_.max_segments) * config_.max_segment_tokens *
                                         hidden;
        block.sdpa_.bind(sdpa_uid_query_k, block.qkv_e4m3_.data());
        block.sdpa_.bind(sdpa_uid_key_k, block.qkv_e4m3_.data() + role_scalars);
        block.sdpa_.bind(sdpa_uid_value_k, block.qkv_e4m3_.data() + 2u * role_scalars);
        block.sdpa_.bind(sdpa_uid_output_k, block.attn_dense_e4m3_.data());
        block.sdpa_.bind(sdpa_uid_descale_q_k, block.qkv_descale_.data());
        block.sdpa_.bind(sdpa_uid_descale_k_k, block.qkv_descale_.data());
        block.sdpa_.bind(sdpa_uid_descale_v_k, block.qkv_descale_.data());
        block.sdpa_.bind(sdpa_uid_descale_s_k, block.ones_.data());
        block.sdpa_.bind(sdpa_uid_scale_s_k, block.ones_.data());
        block.sdpa_.bind(sdpa_uid_scale_o_k, block.ones_.data());
        block.sdpa_.bind(sdpa_uid_amax_s_k, block.absolute_max_scratch_.data());
        block.sdpa_.bind(sdpa_uid_amax_o_k, block.absolute_max_scratch_.data() + 1u);
        block.seq_slot_ = block.sdpa_.bind(sdpa_uid_seq_len_q_k);
        block.sdpa_.bind(sdpa_uid_seq_len_kv_k);
        if (expected_t finalized = block.sdpa_.finalize_dispatch(); !finalized)
            return result_t::fail(std::move(finalized));
        if (block.sdpa_.workspace_bytes > 0) {
            block.sdpa_workspace_ = b8r1_tensor_arena_t::try_empty(
                {static_cast<std::size_t>(block.sdpa_.workspace_bytes)}, b8_alloc);
            if (!block.sdpa_workspace_)
                return result_t::fail(status_t::bad_alloc_k, "FP8 SDPA workspace allocation failed ({} bytes)",
                                      block.sdpa_.workspace_bytes);
        }
    }
    return result_t {std::move(block)};
}

} // namespace unum::uform::core
