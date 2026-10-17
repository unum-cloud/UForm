/**
 *  @brief Device kernels and heavy method bodies of the two transformer blocks declared in @ref transformer_block.cuh .
 *  @file src/core/transformer_block.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 */
#include "core/transformer_block.cuh"

namespace uc = unum::uform::core;
using namespace uc;

/** Learnable RMSNorm forward — declared in @ref transformer_block.cuh with external linkage so the
 *  encoder tower launches the same kernel. */
__global__ void uc::rmsnorm_forward_kernel(bf16r2_view_t input, bf16r1_view_t scale, bf16r2_span_t output,
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

/** Learnable RMSNorm backward for the row gradient — external linkage; see the header. */
__global__ void uc::rmsnorm_backward_kernel(bf16r2_view_t input, bf16r1_view_t scale, f32r1_view_t inverse_rms,
                                            bf16r2_view_t output_grad, bf16r2_span_t input_grad) {
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
        input_grad[row, channel] = __float2bfloat16_rn(row_grad);
    }
}

/** Learnable RMSNorm backward for the scale gradient — external linkage; see the header. */
__global__ void uc::rmsnorm_scale_grad_kernel(bf16r2_view_t input, f32r1_view_t inverse_rms, bf16r2_view_t output_grad,
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

/** The row-gradient backward of rmscast — external linkage; see the header. */
__global__ void uc::rmscast_backward_kernel(f32r2_view_t input, bf16r1_view_t scale, f32r2_view_t output_grad,
                                            f32r2_view_t residual_grad, f32r2_span_t input_grad,
                                            f32r1_span_t inverse_rms, f32_t epsilon) {
    std::size_t const rows = input.extent(0);
    u32_t const width = static_cast<u32_t>(input.extent(1));
    std::size_t const row = blockIdx.x;
    if (row >= rows) return;
    f32_t squares_partial = 0.0f;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t const value = input[row, channel];
        squares_partial += value * value;
    }
    f32_t const sum_squares = block_reduce_broadcast<kernel_block_size_default_k>(squares_partial);
    f32_t const inverse = rsqrtf(sum_squares / static_cast<f32_t>(width) + epsilon);
    if (threadIdx.x == 0u) inverse_rms[row] = inverse;
    f32_t thread_partial = 0.0f;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t const normalized = input[row, channel] * inverse;
        thread_partial += output_grad[row, channel] * __bfloat162float(scale[channel]) * normalized;
    }
    f32_t const projection = block_reduce_broadcast<kernel_block_size_default_k>(thread_partial) /
                             static_cast<f32_t>(width);
    bool const residual_present = !residual_grad.empty();
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t const normalized = input[row, channel] * inverse;
        f32_t const scaled_grad = output_grad[row, channel] * __bfloat162float(scale[channel]);
        f32_t const residual = residual_present ? residual_grad[row, channel] : 0.0f;
        input_grad[row, channel] = residual + (scaled_grad - normalized * projection) * inverse;
    }
}

/** The scale-gradient backward of rmscast — external linkage; see the header. */
__global__ void uc::rmscast_scale_grad_kernel(f32r2_view_t input, f32r1_view_t inverse_rms, f32r2_view_t output_grad,
                                              bf16r1_span_t scale_grad) {
    std::size_t const rows = input.extent(0);
    u32_t const width = static_cast<u32_t>(input.extent(1));
    u32_t const channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) return;
    f32_t accumulated = 0.0f;
    for (std::size_t row = 0; row != rows; ++row)
        accumulated += output_grad[row, channel] * input[row, channel] * inverse_rms[row];
    scale_grad[channel] = __float2bfloat16_rn(accumulated);
}

/** Inverted keyed dropout, in place — external linkage; see the header. The @p salt selects the
 *  site (per layer, per residual join) and @p data_epoch , read from device memory, advances the mask per
 *  step; @ref splitmix64_hash_t re-derives the identical keep decision wherever this runs. */
__global__ void uc::dropout_kernel(bf16r1_span_t values, u32r1_view_t data_epoch, u32_t salt, f32_t drop_probability) {
    u32_t const key = salt ^ (data_epoch[0] * 2654435761u);
    f32_t const keep_scale = 1.0f / (1.0f - drop_probability);
    std::size_t const total = values.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        f32_t const draw = static_cast<f32_t>(splitmix64_hash_t {}(key, index) >> 40) * 0x1.0p-24f;
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
    angles.angles_ = f32r1_tensor_arena_t::uninitialized({total}, f32_alloc).value;
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
 *  @brief Unit RMSNorm backward for the QK-norm, in place on the packed Q|K gradient:
 *      @c dx = (dy − x̂·mean(dy·x̂))·(1/rms) over each 128-channel head vector, recomputing @c 1/rms from the
 *      Float32 pre-norm product the forward's rmscast read.
 *
 *  __grid_size__ @c [ceil(tokens·vectors·32/block),1,1] with one @b warp per head vector, the Q heads then the
 *  K heads of each token; lanes stripe the 128 channels; no dynamic shared memory.
 *
 *  @note @p grads is read-modified in place (the QK-norm backward rescales each head vector here).
 */
static __global__ void transformer_qk_norm_backward_kernel_(bf16r2_span_t grads, f32r2_view_t prenorm,
                                                            u32_t vectors_per_token, f32_t epsilon) {
    std::size_t const tokens = grads.extent(0);
    std::size_t const vectors = tokens * vectors_per_token;
    std::size_t const vector = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size_k;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    if (vector >= vectors) return;
    std::size_t const token = vector / vectors_per_token;
    u32_t const base_channel = static_cast<u32_t>(vector % vectors_per_token) * transformer_head_dim_k;
    f32_t squares_partial = 0.0f;
    for (u32_t channel = lane_index; channel < transformer_head_dim_k; channel += warp_size_k) {
        f32_t const value = prenorm[token, base_channel + channel];
        squares_partial += value * value;
    }
    f32_t const inverse = rsqrtf(warp_reduce_broadcast(squares_partial) / static_cast<f32_t>(transformer_head_dim_k) +
                                 epsilon);
    f32_t lane_partial = 0.0f;
    for (u32_t channel = lane_index; channel < transformer_head_dim_k; channel += warp_size_k) {
        f32_t const normalized = prenorm[token, base_channel + channel] * inverse;
        lane_partial += __bfloat162float(grads[token, base_channel + channel]) * normalized;
    }
    f32_t const projection = warp_reduce_broadcast(lane_partial) / static_cast<f32_t>(transformer_head_dim_k);
    for (u32_t channel = lane_index; channel < transformer_head_dim_k; channel += warp_size_k) {
        f32_t const normalized = prenorm[token, base_channel + channel] * inverse;
        f32_t const grad = __bfloat162float(grads[token, base_channel + channel]);
        grads[token, base_channel + channel] = __float2bfloat16_rn((grad - normalized * projection) * inverse);
    }
}

/** Pick the axis table and angle for one within-head channel pair — the rotary storage invariant. */
[[nodiscard]] static __device__ f32_t transformer_rope_angle_(u32_t head_pair, i16r2_view_t coords, std::size_t token,
                                                              rope_angles_view_t const &angles) noexcept {
    u32_t const temporal_pairs = static_cast<u32_t>(angles.temporal.extent(1));
    u32_t const height_pairs = static_cast<u32_t>(angles.height.extent(1));
    if (head_pair < temporal_pairs) {
        std::size_t const coord = static_cast<std::size_t>(coords[token, 0]);
        return angles.temporal[coord, head_pair];
    }
    if (head_pair < temporal_pairs + height_pairs) {
        std::size_t const coord = static_cast<std::size_t>(coords[token, 1]);
        return angles.height[coord, head_pair - temporal_pairs];
    }
    std::size_t const coord = static_cast<std::size_t>(coords[token, 2]);
    return angles.width[coord, head_pair - temporal_pairs - height_pairs];
}

/**
 *  @brief Rotary rotation in place over one strided role slice (Q or K inside the packed QKV):
 *      the split-half channel pair @b (k,k+64) rotates by the axis angle its index selects — the
 *      NeoX convention the CPU inference kernels and the wider ecosystem share.
 *
 *  Setting @p angle_sign to @c -1 gives the inverse rotation the backward applies to the post-RoPE gradients.
 *
 *  __grid_size__ @c [ceil(tokens·heads·64/block),1,1] grid-stride over channel pairs; no dynamic
 *  shared memory.
 *
 *  @note @p values is read-modified in place (the rotation overwrites each channel pair here).
 */
static __global__ void transformer_rope_kernel_(bf16r2_span_t values, rope_angles_view_t angles, i16r2_view_t coords,
                                                u32_t heads, u32_t role_offset, f32_t angle_sign) {
    std::size_t const tokens = values.extent(0);
    u32_t const pairs_per_head = transformer_head_dim_k >> 1u;
    std::size_t const pairs_per_token = static_cast<std::size_t>(pairs_per_head) * heads;
    std::size_t const total_pairs = pairs_per_token * tokens;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t pair_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair_index < total_pairs; pair_index += stride) {
        std::size_t const token = pair_index / pairs_per_token;
        u32_t const head = static_cast<u32_t>((pair_index % pairs_per_token) / pairs_per_head);
        u32_t const pair = static_cast<u32_t>(pair_index % pairs_per_head);
        u32_t const channel = role_offset + head * transformer_head_dim_k + pair;
        f32_t const real = __bfloat162float(values[token, channel]);
        f32_t const imaginary = __bfloat162float(values[token, channel + pairs_per_head]);
        f32_t sin_value, cos_value;
        __sincosf(angle_sign * transformer_rope_angle_(pair, coords, token, angles), &sin_value, &cos_value);
        values[token, channel] = __float2bfloat16_rn(real * cos_value - imaginary * sin_value);
        values[token, channel + pairs_per_head] = __float2bfloat16_rn(real * sin_value + imaginary * cos_value);
    }
}

/**
 *  @brief SwiGLU activation backward over the Float32 gate|up product, writing the packed @b [tokens,2·ffn]
 *      BFloat16 gate|up gradient: @c d_up = dy·SiLU(gate) and @c d_gate = dy·up·σ(gate)·(1+gate·(1−σ(gate))) .
 *
 *  __grid_size__ grid-stride over @c tokens·ffn cells; no dynamic shared memory.
 */
static __global__ void transformer_swiglu_backward_kernel_(f32r2_view_t gate_up, f32r2_view_t silu_up_grad,
                                                           bf16r2_span_t gate_up_grad) {
    std::size_t const tokens = silu_up_grad.extent(0);
    u32_t const ffn = static_cast<u32_t>(silu_up_grad.extent(1));
    std::size_t const total = tokens * ffn;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const token = cell / ffn;
        u32_t const feature = static_cast<u32_t>(cell % ffn);
        f32_t const gate = gate_up[token, feature];
        f32_t const up = gate_up[token, ffn + feature];
        f32_t const grad = silu_up_grad[token, feature];
        f32_t const sigmoid = 1.0f / (1.0f + expf(-gate));
        f32_t const silu = gate * sigmoid;
        gate_up_grad[token, feature] = __float2bfloat16_rn(grad * up * sigmoid * (1.0f + gate * (1.0f - sigmoid)));
        gate_up_grad[token, ffn + feature] = __float2bfloat16_rn(grad * silu);
    }
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

/** Resolve the max-abs bits into the Float32 descale scalar @c max_abs/240 (floored to keep a
 *  zero weight tensor representable). One thread. */
static __global__ void transformer_descale_from_max_kernel_(u32r1_span_t max_bits, f32r1_span_t descale) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    f32_t const max_abs = __uint_as_float(max_bits[0]);
    descale[0] = fmaxf(max_abs / e4m3_range_k, e4m3_descale_floor_k);
}

/**
 *  @brief Quantise BFloat16 into the E4M3 shadow, dividing by the descale scalar the matmuls will
 *      multiply back through their native scale pointers.
 *
 *  __grid_size__ grid-stride; no dynamic shared memory.
 */
static __global__ void transformer_quantize_e4m3_kernel_(bf16r1_view_t values, e4m3r1_span_t shadow,
                                                         f32r1_view_t descale) {
    std::size_t const total = values.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    f32_t const inverse_scale = 1.0f / descale[0];
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride)
        shadow[cell] = __nv_fp8_e4m3 {__bfloat162float(values[cell]) * inverse_scale};
}

/**
 *  @brief Per-token E4M3 quantisation of a contiguous BFloat16 or Float32 @b [tokens,width] buffer: each
 *      row's scale is its own @c max_abs/240 , so a quiet token never wastes the format's range on a loud
 *      one — the finer scaling tier the matmul path uses (the row scale folds back after the product).
 *
 *  __grid_size__ @c [tokens,1,1] , one CTA per token; no dynamic shared memory.
 */
template <typename value_type_>
static __global__ void transformer_token_quantize_kernel_(nk::tensor_view<value_type_, 2> values, e4m3r2_span_t shadow,
                                                          f32r1_span_t token_scales) {
    std::size_t const token = blockIdx.x;
    u32_t const width = static_cast<u32_t>(values.extent(1));
    f32_t thread_max = 0.0f;
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x)
        thread_max = fmaxf(thread_max, fabsf(static_cast<f32_t>(values[token, channel])));
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
        shadow[token, channel] = __nv_fp8_e4m3 {static_cast<f32_t>(values[token, channel]) * inverse};
}

/**
 *  @brief Fold the scales back into a Float32 product and write it as BFloat16:
 *      @c values[t,o] = products[t,o]·token_scales[t]·weight_descale[0] + residual[t,o] , where an empty
 *      @p token_scales or @p residual drops its term.
 *
 *  __grid_size__ grid-stride over @c tokens·width cells; no dynamic shared memory.
 */
static __global__ void transformer_token_rescale_kernel_(f32r2_view_t products, f32r1_view_t token_scales,
                                                         f32r1_view_t weight_descale, bf16r2_view_t residual,
                                                         bf16r2_span_t values) {
    std::size_t const tokens = values.extent(0);
    u32_t const width = static_cast<u32_t>(values.extent(1));
    std::size_t const total = tokens * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    f32_t const descale = weight_descale[0];
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride) {
        std::size_t const token = cell / width;
        u32_t const channel = static_cast<u32_t>(cell % width);
        f32_t value = products[token, channel] * descale;
        if (!token_scales.empty()) value *= token_scales[token];
        if (!residual.empty()) value += __bfloat162float(residual[token, channel]);
        values[token, channel] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief The one cuDNN Int32 boundary: scale the batch's flat @c u32 segment starts (gapped —
 *      cuDNN's ragged prefix supports gaps; batch stride 0, the prefix supplies each segment's
 *      base) into the per-operand @b element offsets, and cast the stored lengths for the padding mask.
 *
 *  One launch fills the block's whole Int32 staging; nothing outside it sees Int32. vendor-boundary.
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

/** Mark one SDPA-backward gradient output ( @c dQ / @c dK / @c dV ): the packed-QKV THD @p dims
 *  and @p strides of its operand, plus the ragged offset under @p ragged_uid . */
static void configure_sdpa_grad_output(std::shared_ptr<fe::graph::Tensor_attributes> const &node, // vendor-boundary
                                       fe::graph::Graph &graph, i64_t uid, i64_t ragged_uid,
                                       std::vector<i64_t> const &dims,    // vendor-boundary
                                       std::vector<i64_t> const &strides, // vendor-boundary
                                       i64_t segments) {
    node->set_output(true).set_uid(uid).set_dim(dims).set_stride(strides);
    node->set_ragged_offset(sdpa_ragged_offset(graph, ragged_uid, segments));
}

expected_t sdpa_graph_t::verify_raw_bindings(i64_t const *uids, u32_t count) noexcept {
    if (variant_pack_uids.size() != count)
        return expected_t::fail(status_t::other_k,
                                "SDPA graph binds {} tensors, the raw table names {} — a frontend-internal tensor " //
                                "(pass-by-value scalar or workspace) slipped into the variant pack",
                                variant_pack_uids.size(), count);
    for (i64_t const uid : variant_pack_uids) {
        bool named = false;
        for (u32_t entry = 0; entry != count && !named; ++entry) named = uids[entry] == uid;
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
        for (u32_t entry = 0; entry != binding_count; ++entry) fallback_bindings[uids[entry]] = pointers[entry];
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

expected<transformer_scratch_t> transformer_scratch_t::make(gpu_arena_t &arena,
                                                            transformer_block_config_t const &config,
                                                            std::size_t entry_inputs) noexcept {
    using result_t = expected<transformer_scratch_t>;
    if (config.hidden == 0 || config.max_tokens == 0)
        return result_t::fail(status_t::invalid_input_k, "scratch capacities must be positive (hidden={} tokens={})",
                              config.hidden, config.max_tokens);
    nk_capability_t capabilities = 0;
    if (nk_status_t const status = nk_cuda_capabilities_enabled(static_cast<nk_size_t>(arena.device()), &capabilities);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "NumKong cannot query device {}: {}", arena.device(),
                              nk_status_name(status));
    std::size_t const hidden = config.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config.ffn_expansion) * hidden;
    std::size_t const tokens = config.max_tokens;
    std::size_t const padded_tokens = round_up_to_multiple(tokens, std::size_t {8});
    std::size_t const widest = std::max(3 * hidden, 2 * ffn);
    std::size_t const inputs = std::max({hidden, ffn, entry_inputs});
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};

    std::size_t const pack_shapes[][2] = {{hidden, tokens},
                                          {ffn, tokens},
                                          {inputs, tokens},
                                          {hidden, config.max_segments}};
    std::size_t packed_bytes = 0;
    for (auto const &[columns, depth] : pack_shapes) {
        expected<std::size_t> shape_bytes = pack_bytes_bf16(columns, depth, capabilities);
        if (!shape_bytes) return result_t::fail(std::move(shape_bytes));
        packed_bytes = std::max(packed_bytes, shape_bytes.value_);
    }
    transformer_scratch_t scratch;
    scratch.products = f32r1_tensor_arena_t::uninitialized({tokens * ffn}, f32_alloc).value;
    scratch.residual_grad = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    scratch.inverse_rms = f32r1_tensor_arena_t::uninitialized({tokens}, f32_alloc).value;
    scratch.narrowed_grad = bf16r1_tensor_arena_t::uninitialized({tokens * widest}, bf16_alloc).value;
    scratch.transposed_grad = bf16r1_tensor_arena_t::uninitialized({widest * padded_tokens}, bf16_alloc).value;
    scratch.transposed_input = bf16r1_tensor_arena_t::uninitialized({inputs * padded_tokens}, bf16_alloc).value;
    scratch.transposed_weight = bf16r1_tensor_arena_t::uninitialized({widest * hidden}, bf16_alloc).value;
    scratch.qkv_grad = bf16r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, bf16_alloc).value;
    scratch.weight_grad_products =
        f32r1_tensor_arena_t::uninitialized({std::max(widest, inputs) * hidden}, f32_alloc).value;
    scratch.packed_input = b8r1_tensor_arena_t::uninitialized({packed_bytes}, b8_alloc).value;
    if (!scratch.products || !scratch.residual_grad || !scratch.inverse_rms || !scratch.narrowed_grad ||
        !scratch.transposed_grad || !scratch.transposed_input || !scratch.transposed_weight || !scratch.qkv_grad ||
        !scratch.weight_grad_products || !scratch.packed_input)
        return result_t::fail(status_t::bad_alloc_k, "scratch allocation failed (arena_used={})", arena.used());
    return result_t {std::move(scratch)};
}

expected<transformer_block_bf16_t> transformer_block_bf16_t::make(gpu_arena_t &arena, cudnn_session_t &cudnn,
                                                                  transformer_scratch_t &scratch,
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
    block.cudnn_ = &cudnn;
    block.scratch_ = &scratch;
    if (nk_status_t const status = nk_cuda_capabilities_enabled(static_cast<nk_size_t>(arena.device()),
                                                                &block.capabilities_);
        status != nk_success_k)
        return expected<transformer_block_bf16_t>::fail(status_t::cuda_error_k, "NumKong cannot query device {}: {}",
                                                        arena.device(), nk_status_name(status));

    std::size_t const hidden = config.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config.ffn_expansion) * hidden;
    std::size_t const tokens = config.max_tokens;
    std::size_t const heads = hidden / transformer_head_dim_k;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};
    // Plain `uninitialized` allocations, checked in named groups so a failure names the arena pressure.

    // Weights and their per-block gradient buffers, `[out,in]` row-major.
    block.qkv_weight_ = bf16r2_tensor_arena_t::uninitialized({3 * hidden, hidden}, bf16_alloc).value;
    block.out_weight_ = bf16r2_tensor_arena_t::uninitialized({hidden, hidden}, bf16_alloc).value;
    block.gate_up_weight_ = bf16r2_tensor_arena_t::uninitialized({2 * ffn, hidden}, bf16_alloc).value;
    block.down_weight_ = bf16r2_tensor_arena_t::uninitialized({hidden, ffn}, bf16_alloc).value;
    block.pre_norm_scale_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    block.post_norm_scale_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    block.widened_norm_scales_ = f32r1_tensor_arena_t::uninitialized({2 * hidden}, f32_alloc).value;
    block.qkv_weight_grad_ = bf16r2_tensor_arena_t::uninitialized({3 * hidden, hidden}, bf16_alloc).value;
    block.out_weight_grad_ = bf16r2_tensor_arena_t::uninitialized({hidden, hidden}, bf16_alloc).value;
    block.gate_up_weight_grad_ = bf16r2_tensor_arena_t::uninitialized({2 * ffn, hidden}, bf16_alloc).value;
    block.down_weight_grad_ = bf16r2_tensor_arena_t::uninitialized({hidden, ffn}, bf16_alloc).value;
    block.pre_norm_scale_grad_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    block.post_norm_scale_grad_ = bf16r1_tensor_arena_t::uninitialized({hidden}, bf16_alloc).value;
    if (!block.qkv_weight_ || !block.out_weight_ || !block.gate_up_weight_ || !block.down_weight_ ||
        !block.pre_norm_scale_ || !block.post_norm_scale_ || !block.widened_norm_scales_ || !block.qkv_weight_grad_ ||
        !block.out_weight_grad_ || !block.gate_up_weight_grad_ || !block.down_weight_grad_ ||
        !block.pre_norm_scale_grad_ || !block.post_norm_scale_grad_)
        return expected<transformer_block_bf16_t>::fail(status_t::bad_alloc_k,
                                                        "weights allocation failed (arena_used={})", arena.used());
    // Forward stashes the backward consumes.
    block.input_stash_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    block.qkv_products_ = f32r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, f32_alloc).value;
    block.attn_residual_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    block.gate_up_products_ = f32r2_tensor_arena_t::uninitialized({tokens, 2 * ffn}, f32_alloc).value;
    block.normalized_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    block.qkv_ = bf16r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, bf16_alloc).value;
    block.attn_output_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    block.mlp_normalized_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    block.silu_up_ = bf16r2_tensor_arena_t::uninitialized({tokens, ffn}, bf16_alloc).value;
    block.softmax_log_sum_exp_ =
        f32r1_tensor_arena_t::uninitialized(
            {static_cast<std::size_t>(config.max_segments) * heads * config.max_segment_tokens}, f32_alloc)
            .value;
    if (!block.input_stash_ || !block.qkv_products_ || !block.attn_residual_ || !block.gate_up_products_ ||
        !block.normalized_ || !block.qkv_ || !block.attn_output_ || !block.mlp_normalized_ || !block.silu_up_ ||
        !block.softmax_log_sum_exp_)
        return expected<transformer_block_bf16_t>::fail(status_t::bad_alloc_k,
                                                        "stashes allocation failed (arena_used={})", arena.used());
    block.attn_scale_ = f32r1_tensor_arena_t::uninitialized({1}, f32_alloc).value;
    // cuDNN's ragged offsets are per-operand element offsets; two stride families cover all nine
    // SDPA operands (the packed QKV block's and the dense hidden buffers').
    gpu_arena_alloc<i32_t> i32_alloc {arena};
    block.ragged_qkv_offsets_ = i32r1_tensor_arena_t::uninitialized({config.max_segments + 1u}, i32_alloc).value;
    block.ragged_hidden_offsets_ = i32r1_tensor_arena_t::uninitialized({config.max_segments + 1u}, i32_alloc).value;
    block.ragged_seq_lens_ = i32r1_tensor_arena_t::uninitialized({config.max_segments}, i32_alloc).value;
    if (!block.attn_scale_ || !block.ragged_qkv_offsets_ || !block.ragged_hidden_offsets_ || !block.ragged_seq_lens_)
        return expected<transformer_block_bf16_t>::fail(status_t::bad_alloc_k,
                                                        "transients allocation failed (arena_used={})", arena.used());

    // Each weight packs as the forward's B operand [out,in] and, transposed, as the data gradient's [in,out].
    bf16r2_tensor_arena_t const *const weights[] = {&block.qkv_weight_, &block.out_weight_, &block.gate_up_weight_,
                                                    &block.down_weight_};
    for (std::size_t product = 0; product != 4; ++product) {
        std::size_t const outputs = weights[product]->extent(0), inputs = weights[product]->extent(1);
        expected<b8r1_tensor_arena_t> packed = allocate_pack_bf16(outputs, inputs, block.capabilities_, arena);
        if (!packed) return expected<transformer_block_bf16_t>::fail(std::move(packed));
        expected<b8r1_tensor_arena_t> transposed = allocate_pack_bf16(inputs, outputs, block.capabilities_, arena);
        if (!transposed) return expected<transformer_block_bf16_t>::fail(std::move(transposed));
        block.packed_weights_[product] = std::move(packed.value_);
        block.packed_weights_transposed_[product] = std::move(transposed.value_);
    }

    // Seed the device SDPA scale — a synchronous setup-path upload, once, outside any capture.
    f32_t const attention_scale = 1.0f / sqrtf(static_cast<f32_t>(transformer_head_dim_k));
    if (CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(block.attn_scale_.data()), &attention_scale,
                                       sizeof(attention_scale));
        result != CUDA_SUCCESS)
        return expected<transformer_block_bf16_t>::fail(result, "SDPA scale upload failed");

    if (expected_t built = block.build_sdpa_(arena, b8_alloc); !built)
        return expected<transformer_block_bf16_t>::fail(std::move(built));
    return expected<transformer_block_bf16_t> {std::move(block)};
}

expected_t transformer_block_bf16_t::forward(f32r2_view_t input, f32r2_span_t output, u32r1_view_t segment_offsets,
                                             u32r1_view_t segment_lengths, rope_angles_view_t angles,
                                             i16r2_view_t coords, gpu_stream_t &stream) noexcept {
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

    bool const resized = nk::succeeded(input_stash_.resize({tokens, hidden})) &&
                         nk::succeeded(qkv_products_.resize({tokens, 3 * hidden})) &&
                         nk::succeeded(attn_residual_.resize({tokens, hidden})) &&
                         nk::succeeded(gate_up_products_.resize({tokens, 2 * ffn})) &&
                         nk::succeeded(normalized_.resize({tokens, hidden})) &&
                         nk::succeeded(qkv_.resize({tokens, 3 * hidden})) &&
                         nk::succeeded(attn_output_.resize({tokens, hidden})) &&
                         nk::succeeded(mlp_normalized_.resize({tokens, hidden})) &&
                         nk::succeeded(silu_up_.resize({tokens, ffn}));
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "activation stashes cannot hold {} tokens (baked {})",
                                tokens, config_.max_tokens);

    // Residual stash: the backward's pre-norm derivative needs the raw input.
    if (CUresult result = cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(input_stash_.data()),
                                            reinterpret_cast<CUdeviceptr>(input.data()),
                                            tokens * hidden * sizeof(f32_t), stream.get());
        result != CUDA_SUCCESS)
        return expected_t::fail(result, "input stash copy failed ({} tokens)", tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    f32_t const epsilon = config_.rms_eps;
    f32_t *const pre_scale = widened_norm_scales_.data();
    f32_t *const post_scale = pre_scale + hidden;
    std::size_t const qkv_stride = 3 * hidden * sizeof(f32_t), qkv_narrow_stride = 3 * hidden * sizeof(bf16_t);

    // The norms read Float32 γ, widened from the trained BFloat16 scales every step.
    expected_t staged = numkong_result(
        nk_cast_best(pre_norm_scale_.data(), nk_bf16_k, pre_scale, nk_f32_k, hidden, capabilities_, stream.numkong()),
        "pre-attention scale widening");
    if (staged)
        staged = numkong_result(nk_cast_best(post_norm_scale_.data(), nk_bf16_k, post_scale, nk_f32_k, hidden,
                                             capabilities_, stream.numkong()),
                                "post-attention scale widening");
    // Pre-attention norm narrows the residual stream into the fused QKV product's operand.
    if (staged)
        staged = numkong_result(
            nk_each_rmscast_bf16_best(input.data(), pre_scale, reinterpret_cast<nk_bf16_t *>(normalized_.data()),
                                      tokens, 1, hidden, hidden * sizeof(f32_t), hidden * sizeof(bf16_t), epsilon,
                                      capabilities_, stream.numkong()),
            "pre-attention norm");
    if (staged)
        staged = multiply_bf16(normalized_.view(), packed_weights_[product_qkv_k].view(), qkv_products_.span(),
                               capabilities_, stream);
    // `nk_cast` has no row stride, so narrowing V narrows whole rows; the QK-norm then overwrites the Q|K slices.
    if (staged)
        staged = numkong_result(nk_cast_best(qkv_products_.data(), nk_f32_k, qkv_.data(), nk_bf16_k,
                                             tokens * 3 * hidden, capabilities_, stream.numkong()),
                                "QKV narrowing");
    if (staged)
        staged = numkong_result(
            nk_each_rmscast_bf16_best(qkv_products_.data(), NUMKONG_NULL, reinterpret_cast<nk_bf16_t *>(qkv_.data()),
                                      tokens, 2 * heads, transformer_head_dim_k, qkv_stride, qkv_narrow_stride, epsilon,
                                      capabilities_, stream.numkong()),
            "QK-norm");
    if (!staged) return staged;
    std::size_t const head_vectors = tokens * heads;
    u32_t const query_offset = 0u, key_offset = hidden;
    f32_t const forward_sign = 1.0f;
    std::size_t const rope_pairs = head_vectors * (transformer_head_dim_k >> 1u);
    dim3 const rope_grid {static_cast<u32_t>(divide_round_up(rope_pairs, kernel_block_size_default_k)), 1u, 1u};
    staged = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape, stream,
                           qkv_.span(), angles, coords, heads, query_offset, forward_sign);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape,
                               stream, qkv_.span(), angles, coords, heads, key_offset, forward_sign);
    if (!staged) return staged;

    // Fused SDPA over the packed segments, stashing the log-sum-exp for the backward.
    if (expected_t attended = run_sdpa_forward_(segment_offsets, segment_lengths, stream); !attended) return attended;

    // Out-projection, the residual into the Float32 stream, and the post-attention norm narrowing it.
    staged = multiply_bf16(attn_output_.view(), packed_weights_[product_out_k].view(), attn_residual_.span(),
                           capabilities_, stream);
    if (staged)
        staged = numkong_result(nk_each_sum_f32_best(attn_residual_.data(), input.data(), tokens * hidden,
                                                     attn_residual_.data(), capabilities_, stream.numkong()),
                                "attention residual");
    if (staged)
        staged = numkong_result(nk_each_rmscast_bf16_best(attn_residual_.data(), post_scale,
                                                          reinterpret_cast<nk_bf16_t *>(mlp_normalized_.data()), tokens,
                                                          1, hidden, hidden * sizeof(f32_t), hidden * sizeof(bf16_t),
                                                          epsilon, capabilities_, stream.numkong()),
                                "post-attention norm");

    // SwiGLU MLP: fused gate/up product, SiLU·up in Float32 narrowed by a cast, down product, residual.
    if (staged)
        staged = multiply_bf16(mlp_normalized_.view(), packed_weights_[product_gate_up_k].view(),
                               gate_up_products_.span(), capabilities_, stream);
    f32_t const *const gate = gate_up_products_.data();
    std::size_t const gate_up_stride = 2 * ffn * sizeof(f32_t);
    if (staged)
        staged = numkong_result(
            nk_each_swiglu_f32_best(gate, gate + ffn, scratch_->products.data(), tokens, ffn, gate_up_stride,
                                    gate_up_stride, ffn * sizeof(f32_t), 1.0f, 1.0f, capabilities_, stream.numkong()),
            "SwiGLU");
    if (staged)
        staged = numkong_result(nk_cast_best(scratch_->products.data(), nk_f32_k, silu_up_.data(), nk_bf16_k,
                                             tokens * ffn, capabilities_, stream.numkong()),
                                "SwiGLU narrowing");
    if (staged)
        staged = multiply_bf16(silu_up_.view(), packed_weights_[product_down_k].view(), output, capabilities_, stream);
    if (staged)
        staged = numkong_result(nk_each_sum_f32_best(output.data(), attn_residual_.data(), tokens * hidden,
                                                     output.data(), capabilities_, stream.numkong()),
                                "MLP residual");
    return staged;
}

expected_t transformer_block_bf16_t::backward(f32r2_view_t output_grad, f32r2_span_t input_grad,
                                              u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                              rope_angles_view_t angles, i16r2_view_t coords,
                                              gpu_stream_t &stream) noexcept {
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

    transformer_scratch_t &scratch = *scratch_;
    bool const resized = nk::succeeded(scratch.qkv_grad.resize({tokens, 3 * hidden})) &&
                         nk::succeeded(scratch.residual_grad.resize({tokens, hidden}));
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "gradient transients cannot hold {} tokens (baked {})",
                                tokens, config_.max_tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const token_grid {static_cast<u32_t>(tokens), 1u, 1u};
    dim3 const channel_grid {static_cast<u32_t>(divide_round_up(hidden, kernel_block_size_default_k)), 1u, 1u};
    f32_t const epsilon = config_.rms_eps;
    bf16r2_span_t const hidden_narrowed {scratch.narrowed_grad.data(), tokens, hidden};
    bf16r2_span_t const gate_up_grad {scratch.narrowed_grad.data(), tokens, 2 * ffn};
    f32r2_span_t const silu_up_grad {scratch.products.data(), tokens, ffn};
    f32r2_span_t const hidden_products {scratch.products.data(), tokens, hidden};
    auto weight_gradient = [&](bf16r2_view_t product_grad, bf16r2_view_t product_input,
                               bf16r2_span_t weight_grad) noexcept {
        return weight_gradient_bf16(product_grad, product_input, weight_grad, scratch.transposed_grad.span(),
                                    scratch.transposed_input.span(), scratch.packed_input.span(),
                                    scratch.weight_grad_products.span(), capabilities_, stream);
    };

    // MLP backward: the output gradient narrows into the A operand of both down-projection gradients.
    expected_t staged = numkong_result(nk_cast_best(output_grad.data(), nk_f32_k, hidden_narrowed.data(), nk_bf16_k,
                                                    tokens * hidden, capabilities_, stream.numkong()),
                                       "output gradient narrowing");
    if (staged)
        staged = multiply_bf16(hidden_narrowed, packed_weights_transposed_[product_down_k].view(), silu_up_grad,
                               capabilities_, stream);
    if (staged) staged = weight_gradient(hidden_narrowed, silu_up_.view(), down_weight_grad_.span());
    // The gate/up gradient overwrites the narrowed output gradient both down-projection products consumed.
    if (staged)
        staged = launch_kernel(
            reinterpret_cast<void const *>(&transformer_swiglu_backward_kernel_),
            dim3 {static_cast<u32_t>(divide_round_up(tokens * ffn, kernel_block_size_default_k)), 1u, 1u}, block_shape,
            stream, gate_up_products_.view(), f32r2_view_t {silu_up_grad}, gate_up_grad);
    if (staged)
        staged = multiply_bf16(gate_up_grad, packed_weights_transposed_[product_gate_up_k].view(), hidden_products,
                               capabilities_, stream);
    if (staged) staged = weight_gradient(gate_up_grad, mlp_normalized_.view(), gate_up_weight_grad_.span());

    // Post-attention norm backward: the residual-stream gradient adds the output gradient.
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&rmscast_backward_kernel), token_grid, block_shape,
                               stream, attn_residual_.view(), post_norm_scale_.view(), f32r2_view_t {hidden_products},
                               output_grad, scratch.residual_grad.span(), scratch.inverse_rms.span(), epsilon);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&rmscast_scale_grad_kernel), channel_grid, block_shape,
                               stream, attn_residual_.view(), scratch.inverse_rms.view(),
                               f32r2_view_t {hidden_products}, post_norm_scale_grad_.span());

    // Out-projection backward, then the attention-output gradient narrows into SDPA's dO.
    if (staged)
        staged = numkong_result(nk_cast_best(scratch.residual_grad.data(), nk_f32_k, hidden_narrowed.data(), nk_bf16_k,
                                             tokens * hidden, capabilities_, stream.numkong()),
                                "residual gradient narrowing");
    if (staged)
        staged = multiply_bf16(hidden_narrowed, packed_weights_transposed_[product_out_k].view(), hidden_products,
                               capabilities_, stream);
    if (staged) staged = weight_gradient(hidden_narrowed, attn_output_.view(), out_weight_grad_.span());
    if (staged)
        staged = numkong_result(nk_cast_best(hidden_products.data(), nk_f32_k, hidden_narrowed.data(), nk_bf16_k,
                                             tokens * hidden, capabilities_, stream.numkong()),
                                "attention output gradient narrowing");
    if (!staged) return staged;

    // SDPA backward into the packed d_qkv block, then inverse rotation and QK-norm backward.
    if (expected_t attended = run_sdpa_backward_(segment_offsets, segment_lengths, stream); !attended) return attended;
    f32_t const inverse_sign = -1.0f;
    std::size_t const head_vectors = tokens * heads;
    std::size_t const rope_pairs = head_vectors * (transformer_head_dim_k >> 1u);
    dim3 const rope_grid {static_cast<u32_t>(divide_round_up(rope_pairs, kernel_block_size_default_k)), 1u, 1u};
    dim3 const vector_grid {
        static_cast<u32_t>(divide_round_up(2 * head_vectors * warp_size_k, kernel_block_size_default_k)), 1u, 1u};
    u32_t const query_offset = 0u, key_offset = hidden;
    staged = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape, stream,
                           scratch.qkv_grad.span(), angles, coords, heads, query_offset, inverse_sign);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape,
                               stream, scratch.qkv_grad.span(), angles, coords, heads, key_offset, inverse_sign);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&transformer_qk_norm_backward_kernel_), vector_grid,
                               block_shape, stream, scratch.qkv_grad.span(), qkv_products_.view(), 2u * heads, epsilon);

    // QKV projection backward, then the pre-attention norm backward adding the residual-stream gradient.
    if (staged)
        staged = multiply_bf16(scratch.qkv_grad.view(), packed_weights_transposed_[product_qkv_k].view(),
                               hidden_products, capabilities_, stream);
    if (staged) staged = weight_gradient(scratch.qkv_grad.view(), normalized_.view(), qkv_weight_grad_.span());
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&rmscast_backward_kernel), token_grid, block_shape,
                               stream, input_stash_.view(), pre_norm_scale_.view(), f32r2_view_t {hidden_products},
                               scratch.residual_grad.view(), input_grad, scratch.inverse_rms.span(), epsilon);
    if (staged)
        staged = launch_kernel(reinterpret_cast<void const *>(&rmscast_scale_grad_kernel), channel_grid, block_shape,
                               stream, input_stash_.view(), scratch.inverse_rms.view(), f32r2_view_t {hidden_products},
                               pre_norm_scale_grad_.span());
    return staged;
}

expected_t transformer_block_bf16_t::pack_weights(gpu_stream_t &stream) noexcept {
    bf16r2_tensor_arena_t const *const weights[] = {&qkv_weight_, &out_weight_, &gate_up_weight_, &down_weight_};
    for (std::size_t product = 0; product != 4; ++product)
        if (expected_t packed = pack_weight_bf16(weights[product]->view(), scratch_->transposed_weight.span(),
                                                 packed_weights_[product].span(),
                                                 packed_weights_transposed_[product].span(), capabilities_, stream);
            !packed)
            return packed;
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
    // The scale rides a device scalar tensor we own (bound via `sdpa_scalar`) — never the frontend's
    // pass-by-value cache, which lives behind private members the raw execution path cannot reach.

    // Forward graph: owned bindings pinned now, per-call entries recorded for the dispatch stores.
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
                           .set_attn_scale(sdpa_scalar(graph, sdpa_uid_attn_scale_k, "attn_scale"))
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
        forward_seq_entry_ = forward_sdpa_.bind(sdpa_uid_seq_len_q_k);
        forward_sdpa_.bind(sdpa_uid_seq_len_kv_k);
        forward_ragged_entry_ = forward_sdpa_.bind(sdpa_uid_ragged_base_k + 0);
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
                           .set_attn_scale(sdpa_scalar(graph, sdpa_uid_attn_scale_k, "attn_scale"))
                           .set_padding_mask(true)
                           .set_seq_len_q(seq_q)
                           .set_seq_len_kv(seq_kv);
        auto [query_grad, key_grad, value_grad] = graph.sdpa_backward(queries, keys, values, output, output_grad, stats,
                                                                      options);
        std::vector<i64_t> const grad_dims {segments, heads, seq, dim}; // vendor-boundary
        configure_sdpa_grad_output(query_grad, graph, sdpa_uid_query_grad_k, sdpa_uid_ragged_base_k + 5, grad_dims,
                                   strides, segments);
        configure_sdpa_grad_output(key_grad, graph, sdpa_uid_key_grad_k, sdpa_uid_ragged_base_k + 6, grad_dims, strides,
                                   segments);
        configure_sdpa_grad_output(value_grad, graph, sdpa_uid_value_grad_k, sdpa_uid_ragged_base_k + 7, grad_dims,
                                   strides, segments);
        if (auto status = graph.build(cudnn_->get(), {fe::HeurMode_t::A}); status.is_bad())
            return expected_t::fail(status_t::cuda_error_k, "SDPA backward graph build failed: {}",
                                    status.get_message().c_str()); // vendor-boundary
        backward_sdpa_.bind(sdpa_uid_query_k, qkv_.data());
        backward_sdpa_.bind(sdpa_uid_key_k, qkv_.data() + hidden);
        backward_sdpa_.bind(sdpa_uid_value_k, qkv_.data() + 2u * hidden);
        backward_sdpa_.bind(sdpa_uid_output_k, attn_output_.data());
        backward_sdpa_.bind(sdpa_uid_output_grad_k, scratch_->narrowed_grad.data());
        backward_sdpa_.bind(sdpa_uid_stats_k, softmax_log_sum_exp_.data());
        backward_sdpa_.bind(sdpa_uid_query_grad_k, scratch_->qkv_grad.data());
        backward_sdpa_.bind(sdpa_uid_key_grad_k, scratch_->qkv_grad.data() + hidden);
        backward_sdpa_.bind(sdpa_uid_value_grad_k, scratch_->qkv_grad.data() + 2u * hidden);
        backward_sdpa_.bind(sdpa_uid_attn_scale_k, attn_scale_.data());
        backward_seq_entry_ = backward_sdpa_.bind(sdpa_uid_seq_len_q_k);
        backward_sdpa_.bind(sdpa_uid_seq_len_kv_k);
        backward_ragged_entry_ = backward_sdpa_.bind(sdpa_uid_ragged_base_k + 0);
        for (i64_t ragged = 1; ragged != 8; ++ragged) backward_sdpa_.bind(sdpa_uid_ragged_base_k + ragged);
        if (expected_t finalized = backward_sdpa_.finalize_dispatch(); !finalized) return finalized;
    }
    i64_t const sdpa_workspace = forward_sdpa_.workspace_bytes > backward_sdpa_.workspace_bytes
                                     ? forward_sdpa_.workspace_bytes
                                     : backward_sdpa_.workspace_bytes;
    if (sdpa_workspace > 0) {
        sdpa_workspace_ =
            b8r1_tensor_arena_t::uninitialized({static_cast<std::size_t>(sdpa_workspace)}, b8_alloc).value;
        if (!sdpa_workspace_)
            return expected_t::fail(status_t::bad_alloc_k, "SDPA workspace allocation failed ({} bytes)",
                                    sdpa_workspace);
    }
    return {};
}

expected_t transformer_block_bf16_t::stage_ragged_offsets_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                           gpu_stream_t &stream) noexcept {
    return launch_kernel(reinterpret_cast<void const *>(&transformer_ragged_offsets_kernel_), dim3 {1u, 1u, 1u},
                         dim3 {warp_size_k, 1u, 1u}, stream, segment_offsets, segment_lengths,
                         ragged_qkv_offsets_.span(), ragged_hidden_offsets_.span(), ragged_seq_lens_.span(),
                         config_.hidden);
}

expected_t transformer_block_bf16_t::run_sdpa_forward_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                       gpu_stream_t &stream) noexcept {
    if (expected_t bound = cudnn_->bind(stream); !bound) return bound;
    if (expected_t staged = stage_ragged_offsets_(segment_offsets, segment_lengths, stream); !staged) return staged;
    forward_sdpa_.pointers[forward_seq_entry_] = ragged_seq_lens_.data();
    forward_sdpa_.pointers[forward_seq_entry_ + 1u] = ragged_seq_lens_.data();
    for (u32_t ragged = 0; ragged != 3u; ++ragged)
        forward_sdpa_.pointers[forward_ragged_entry_ + ragged] = ragged_qkv_offsets_.data();
    forward_sdpa_.pointers[forward_ragged_entry_ + 3u] = ragged_hidden_offsets_.data();
    return forward_sdpa_.execute(cudnn_->get(), sdpa_workspace_.data());
}

expected_t transformer_block_bf16_t::run_sdpa_backward_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                        gpu_stream_t &stream) noexcept {
    if (expected_t bound = cudnn_->bind(stream); !bound) return bound;
    if (expected_t staged = stage_ragged_offsets_(segment_offsets, segment_lengths, stream); !staged) return staged;
    backward_sdpa_.pointers[backward_seq_entry_] = ragged_seq_lens_.data();
    backward_sdpa_.pointers[backward_seq_entry_ + 1u] = ragged_seq_lens_.data();
    i32_t *const qkv_offsets = ragged_qkv_offsets_.data();
    i32_t *const hidden_offsets = ragged_hidden_offsets_.data();
    i32_t *const per_operand[8] = {qkv_offsets,    qkv_offsets, qkv_offsets, hidden_offsets,
                                   hidden_offsets, qkv_offsets, qkv_offsets, qkv_offsets};
    for (u32_t ragged = 0; ragged != 8u; ++ragged)
        backward_sdpa_.pointers[backward_ragged_entry_ + ragged] = per_operand[ragged];
    return backward_sdpa_.execute(cudnn_->get(), sdpa_workspace_.data());
}

expected_t transformer_block_e4m3_t::forward(bf16r2_view_t input, bf16r2_span_t output, u32r1_view_t segment_offsets,
                                             u32r1_view_t segment_lengths, rope_angles_view_t angles,
                                             i16r2_view_t coords, gpu_stream_t &stream) noexcept {
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
        nk::succeeded(normalized_.resize({tokens, hidden})) && nk::succeeded(qkv_.resize({tokens, 3 * hidden})) &&
        nk::succeeded(attn_residual_.resize({tokens, hidden})) && nk::succeeded(gate_up_.resize({tokens, 2 * ffn})) &&
        nk::succeeded(silu_up_.resize({tokens, ffn})) && nk::succeeded(normalized_e4m3_.resize({tokens, hidden})) &&
        nk::succeeded(qkv_e4m3_.resize({tokens, 3 * hidden})) &&
        nk::succeeded(attn_output_e4m3_.resize({tokens, hidden})) && nk::succeeded(silu_up_e4m3_.resize({tokens, ffn}));
    if (!resized)
        return expected_t::fail(status_t::shape_unsupported_k, "activation stashes cannot hold {} tokens (baked {})",
                                tokens, config_.max_tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const token_grid {static_cast<u32_t>(tokens), 1u, 1u};
    f32r1_view_t const token_scales_view = token_scales_.view();
    // RMSNorm of `groups` vectors of `columns` per token, in place when `target` is `source`.
    auto const normalize = [&](bf16_t const *source, f32_t const *gamma, bf16_t *target, std::size_t groups,
                               std::size_t columns, std::size_t stride) -> expected_t {
        nk_status_t const status = nk_each_rmsnorm_bf16_best(
            reinterpret_cast<nk_bf16_t const *>(source), gamma, reinterpret_cast<nk_bf16_t *>(target), tokens, groups,
            columns, stride, stride, config_.rms_eps, capabilities_, stream.numkong());
        if (status != nk_success_k)
            return expected_t::fail(status_t::cuda_error_k, "RMSNorm failed: {}", nk_status_name(status));
        return {};
    };

    // Pre-attention RMSNorm in BFloat16, per-token E4M3 quantisation, QKV product, scale fold.
    expected_t launched = normalize(input.data(), pre_norm_scale_.data(), normalized_.data(), 1, hidden,
                                    hidden * sizeof(bf16_t));
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_quantize_kernel_<bf16_t>),
                                 token_grid, block_shape, stream, normalized_.view(), normalized_e4m3_.span(),
                                 token_scales_.span());
    if (!launched) return launched;
    launched = project_(product_qkv_k, normalized_e4m3_.view(), token_scales_view, {}, qkv_.span(), stream);
    if (!launched) return launched;
    // Unit QK-norm of every Q and K head vector, adjacent in each token's row, then M-RoPE.
    launched = normalize(qkv_.data(), nullptr, qkv_.data(), 2 * heads, transformer_head_dim_k,
                         3 * hidden * sizeof(bf16_t));
    if (!launched) return launched;
    std::size_t const head_vectors = tokens * heads;
    u32_t const query_offset = 0u, key_offset = hidden;
    f32_t const forward_sign = 1.0f;
    std::size_t const rope_pairs = head_vectors * (transformer_head_dim_k >> 1u);
    dim3 const rope_grid {static_cast<u32_t>(divide_round_up(rope_pairs, kernel_block_size_default_k)), 1u, 1u};
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape, stream,
                             qkv_.span(), angles, coords, heads, query_offset, forward_sign);
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_rope_kernel_), rope_grid, block_shape,
                                 stream, qkv_.span(), angles, coords, heads, key_offset, forward_sign);
    if (!launched) return launched;

    // QK-norm leaves Q and K at unit RMS per head and V stays well inside E4M3's range, so all three cast
    // unscaled. The gaps between segments ride along as extra query rows of the segment before; nothing reads them.
    std::size_t const segments = segment_lengths.extent(0);
    if (nk_status_t const status = nk_cast_best(qkv_.data(), nk_bf16_k, qkv_e4m3_.data(), nk_e4m3_k,
                                                tokens * 3 * hidden, capabilities_, stream.numkong());
        status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "Q/K/V cast failed: {}", nk_status_name(status));
    if (nk::failed(products_.resize({tokens, hidden})))
        return expected_t::fail(status_t::shape_unsupported_k, "attention scratch cannot hold [{}x{}]", tokens, hidden);
    auto const *const qkv_e4m3 = reinterpret_cast<nk_e4m3_t const *>(qkv_e4m3_.data());
    std::size_t const qkv_bytes = 3 * hidden * sizeof(e4m3_t);
    nk_status_t status = nk_attention_pack_e4m3_best(heads, transformer_head_dim_k, segment_offsets.data(),
                                                     segment_lengths.data(), segments, qkv_e4m3 + hidden, qkv_bytes,
                                                     qkv_e4m3 + 2 * hidden, qkv_bytes, packed_keys_values_.data(), 0,
                                                     segments * heads, capabilities_, stream.numkong());
    if (status == nk_success_k)
        status = nk_attention_packed_e4m3_best(
            heads, heads, transformer_head_dim_k, segment_offsets.data(), tokens,
            1.0f / sqrtf(static_cast<f32_t>(transformer_head_dim_k)), NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX, qkv_e4m3,
            qkv_bytes, packed_keys_values_.data(), products_.data(), hidden * sizeof(f32_t), NUMKONG_NULL, 0,
            NUMKONG_SIZE_MAX, NUMKONG_NULL, 0, capabilities_, stream.numkong());
    if (status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "E4M3 attention failed: {}", nk_status_name(status));

    // Out-projection consumes the attention output under per-token scales, adding the residual input.
    launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_quantize_kernel_<f32_t>), token_grid,
                             block_shape, stream, products_.view(), attn_output_e4m3_.span(), token_scales_.span());
    if (launched)
        launched = project_(product_out_k, attn_output_e4m3_.view(), token_scales_view, input, attn_residual_.span(),
                            stream);
    if (launched)
        launched = normalize(attn_residual_.data(), post_norm_scale_.data(), normalized_.data(), 1, hidden,
                             hidden * sizeof(bf16_t));
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_quantize_kernel_<bf16_t>),
                                 token_grid, block_shape, stream, normalized_.view(), normalized_e4m3_.span(),
                                 token_scales_.span());
    if (!launched) return launched;
    launched = project_(product_gate_up_k, normalized_e4m3_.view(), token_scales_view, {}, gate_up_.span(), stream);
    if (!launched) return launched;
    auto *const gate_up = reinterpret_cast<nk_bf16_t const *>(gate_up_.data());
    if (nk_status_t const status = nk_each_swiglu_bf16_best(
            gate_up, gate_up + ffn, reinterpret_cast<nk_bf16_t *>(silu_up_.data()), tokens, ffn,
            2 * ffn * sizeof(bf16_t), 2 * ffn * sizeof(bf16_t), ffn * sizeof(bf16_t), 1.0f, 1.0f, capabilities_,
            stream.numkong());
        status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "SwiGLU failed: {}", nk_status_name(status));
    if (launched)
        launched = launch_kernel(reinterpret_cast<void const *>(&transformer_token_quantize_kernel_<bf16_t>),
                                 token_grid, block_shape, stream, silu_up_.view(), silu_up_e4m3_.span(),
                                 token_scales_.span());
    if (!launched) return launched;
    return project_(product_down_k, silu_up_e4m3_.view(), token_scales_view, attn_residual_.view(), output, stream);
}

expected_t transformer_block_e4m3_t::project_(std::size_t product, e4m3r2_view_t activations, f32r1_view_t token_scales,
                                              bf16r2_view_t residual, bf16r2_span_t values,
                                              gpu_stream_t &stream) noexcept {
    std::size_t const tokens = activations.extent(0), depth = activations.extent(1), width = values.extent(1);
    if (nk::failed(products_.resize({tokens, width})))
        return expected_t::fail(status_t::shape_unsupported_k, "product scratch cannot hold [{}x{}]", tokens, width);
    nk_status_t const status = nk_dots_packed_e4m3_best(reinterpret_cast<nk_e4m3_t const *>(activations.data()),
                                                        packed_weights_[product].data(), products_.data(), tokens,
                                                        width, depth, depth * sizeof(e4m3_t), width * sizeof(f32_t),
                                                        NUMKONG_NULL, 0, capabilities_, stream.numkong());
    if (status != nk_success_k)
        return expected_t::fail(status_t::cuda_error_k, "E4M3 product {} failed: {}", product, nk_status_name(status));
    f32r1_view_t const weight_descale {weight_descales_.data() + product, 1};
    dim3 const grid {static_cast<u32_t>(divide_round_up(tokens * width, kernel_block_size_default_k)), 1u, 1u};
    return launch_kernel(reinterpret_cast<void const *>(&transformer_token_rescale_kernel_), grid,
                         dim3 {kernel_block_size_default_k, 1u, 1u}, stream, products_.view(), token_scales,
                         weight_descale, residual, values);
}

expected<transformer_block_e4m3_t> transformer_block_bf16_t::quantize_weights(gpu_arena_t &arena,
                                                                              gpu_stream_t &stream) noexcept {
    using result_t = expected<transformer_block_e4m3_t>;
    transformer_block_e4m3_t block;
    block.config_ = config_;
    if (nk_status_t const status = nk_cuda_capabilities_enabled(static_cast<nk_size_t>(stream.device()),
                                                                &block.capabilities_);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "NumKong cannot query device {}: {}", stream.device(),
                              nk_status_name(status));

    std::size_t const hidden = config_.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config_.ffn_expansion) * hidden;
    std::size_t const tokens = config_.max_tokens;
    std::size_t const heads = hidden / transformer_head_dim_k;
    gpu_arena_alloc<e4m3_t> e4m3_alloc {arena};
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};
    // Plain `uninitialized` allocations, checked in named groups so a failure names the arena pressure.

    // Static FP8 weights, their descales, and the Float32 norm scales.
    u32r1_tensor_arena_t max_bits {};
    block.qkv_weight_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({3 * hidden, hidden}, e4m3_alloc).value;
    block.out_weight_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({hidden, hidden}, e4m3_alloc).value;
    block.gate_up_weight_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({2 * ffn, hidden}, e4m3_alloc).value;
    block.down_weight_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({hidden, ffn}, e4m3_alloc).value;
    block.weight_descales_ = f32r1_tensor_arena_t::uninitialized({4}, f32_alloc).value;
    block.pre_norm_scale_ = f32r1_tensor_arena_t::uninitialized({hidden}, f32_alloc).value;
    block.post_norm_scale_ = f32r1_tensor_arena_t::uninitialized({hidden}, f32_alloc).value;
    if (!block.qkv_weight_e4m3_ || !block.out_weight_e4m3_ || !block.gate_up_weight_e4m3_ || !block.down_weight_e4m3_ ||
        !block.weight_descales_ || !block.pre_norm_scale_ || !block.post_norm_scale_)
        return result_t::fail(status_t::bad_alloc_k, "weights allocation failed (arena_used={})", arena.used());
    // BFloat16 activation stashes shared with the BFloat16 arm's layout.
    block.normalized_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    block.qkv_ = bf16r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, bf16_alloc).value;
    block.attn_output_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({tokens, hidden}, e4m3_alloc).value;
    block.attn_residual_ = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    block.gate_up_ = bf16r2_tensor_arena_t::uninitialized({tokens, 2 * ffn}, bf16_alloc).value;
    block.silu_up_ = bf16r2_tensor_arena_t::uninitialized({tokens, ffn}, bf16_alloc).value;
    if (!block.normalized_ || !block.qkv_ || !block.attn_output_e4m3_ || !block.attn_residual_ || !block.gate_up_ ||
        !block.silu_up_)
        return result_t::fail(status_t::bad_alloc_k, "stashes allocation failed (arena_used={})", arena.used());
    // FP8 shadows, scales, the packed keys and values, and quantisation scratch.
    nk_size_t packed_keys_values_bytes = 0;
    if (nk_status_t const status = nk_attention_pack_size_e4m3_best(heads, transformer_head_dim_k, tokens,
                                                                    config_.max_segments, block.capabilities_,
                                                                    &packed_keys_values_bytes);
        status != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "E4M3 attention pack sizing failed: {}", nk_status_name(status));
    block.normalized_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({tokens, hidden}, e4m3_alloc).value;
    block.qkv_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, e4m3_alloc).value;
    block.packed_keys_values_ =
        b8r1_tensor_arena_t::uninitialized({static_cast<std::size_t>(packed_keys_values_bytes)}, b8_alloc).value;
    block.silu_up_e4m3_ = e4m3r2_tensor_arena_t::uninitialized({tokens, ffn}, e4m3_alloc).value;
    block.products_ = f32r2_tensor_arena_t::uninitialized({tokens, std::max(3 * hidden, 2 * ffn)}, f32_alloc).value;
    block.token_scales_ = f32r1_tensor_arena_t::uninitialized({tokens}, f32_alloc).value;
    max_bits = u32r1_tensor_arena_t::uninitialized({1}, u32_alloc).value;
    if (!block.normalized_e4m3_ || !block.qkv_e4m3_ || !block.packed_keys_values_ || !block.silu_up_e4m3_ ||
        !block.products_ || !block.token_scales_ || !max_bits)
        return result_t::fail(status_t::bad_alloc_k, "quantized operands allocation failed (arena_used={})",
                              arena.used());

    // Norm scales widen exactly from BFloat16, the Float32 γ NumKong's RMSNorm reads.
    nk_status_t widened = nk_cast_best(pre_norm_scale_.data(), nk_bf16_k, block.pre_norm_scale_.data(), nk_f32_k,
                                       hidden, block.capabilities_, stream.numkong());
    if (widened == nk_success_k)
        widened = nk_cast_best(post_norm_scale_.data(), nk_bf16_k, block.post_norm_scale_.data(), nk_f32_k, hidden,
                               block.capabilities_, stream.numkong());
    if (widened != nk_success_k)
        return result_t::fail(status_t::cuda_error_k, "norm scale widening failed: {}", nk_status_name(widened));

    // Per-weight static quantisation: order-free max-abs, descale = max_abs/240, divide-and-cast, then the
    // one-time pack into NumKong's layout. Every weight is a rank-2 [out,in] matrix; the quantisation itself
    // is purely elementwise, so each job flattens to the rank-1 handle the house kernels take.
    struct quantize_job_t {
        bf16r2_tensor_arena_t const *source;
        e4m3r2_tensor_arena_t *shadow;
        std::size_t product;
    };
    quantize_job_t const jobs[4] = {
        {&qkv_weight_, &block.qkv_weight_e4m3_, transformer_block_e4m3_t::product_qkv_k},
        {&out_weight_, &block.out_weight_e4m3_, transformer_block_e4m3_t::product_out_k},
        {&gate_up_weight_, &block.gate_up_weight_e4m3_, transformer_block_e4m3_t::product_gate_up_k},
        {&down_weight_, &block.down_weight_e4m3_, transformer_block_e4m3_t::product_down_k}};
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
        f32r1_span_t const descale {block.weight_descales_.data() + job.product, 1};
        if (launched)
            launched = launch_kernel(reinterpret_cast<void const *>(&transformer_descale_from_max_kernel_),
                                     dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u}, stream, max_bits.span(), descale);
        if (launched) {
            f32r1_view_t const descale_view = descale;
            e4m3r1_span_t const shadow_flat = job.shadow->span().flatten<1>();
            launched = launch_kernel(reinterpret_cast<void const *>(&transformer_quantize_e4m3_kernel_), grid,
                                     block_shape, stream, source_flat, shadow_flat, descale_view);
        }
        if (!launched) return result_t::fail(std::move(launched));

        std::size_t const columns = job.shadow->extent(0), depth = job.shadow->extent(1);
        nk_size_t packed_bytes = 0;
        if (nk_status_t const status = nk_dots_pack_size_e4m3_best(columns, depth, block.capabilities_, &packed_bytes);
            status != nk_success_k)
            return result_t::fail(status_t::cuda_error_k, "E4M3 pack sizing failed: {}", nk_status_name(status));
        b8r1_tensor_arena_t &packed = block.packed_weights_[job.product];
        packed = b8r1_tensor_arena_t::uninitialized({static_cast<std::size_t>(packed_bytes)}, b8_alloc).value;
        if (!packed)
            return result_t::fail(status_t::bad_alloc_k, "packed weight allocation failed ({} bytes)", packed_bytes);
        if (nk_status_t const status = nk_dots_pack_e4m3_best(reinterpret_cast<nk_e4m3_t const *>(job.shadow->data()),
                                                              columns, depth, depth * sizeof(e4m3_t), packed.data(), 0,
                                                              columns, block.capabilities_, stream.numkong());
            status != nk_success_k)
            return result_t::fail(status_t::cuda_error_k, "E4M3 weight packing failed: {}", nk_status_name(status));
    }
    return result_t {std::move(block)};
}
