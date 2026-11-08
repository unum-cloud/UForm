/**
 *  @file src/train/transformer_block.cu
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief Device kernels and heavy method bodies of the two transformer blocks declared in
 *      @ref transformer_block.cuh .
 */
#include <cmath> // `std::exp`, `std::fabs`, `std::fmax`

#include <cuda/atomic> // `cuda::atomic_ref` — the max-abs bits every thread folds into

#include "uform/train/transformer_block.cuh"

namespace ut = unum::uform::train;
using namespace ut;
using namespace unum::uform::utilities;
namespace ue = unum::uform::encoder;
using ue::e4m3_descale_floor_k;
using ue::e4m3_range_k;
using ue::quantize_token;
using ue::rescale_tokens_cells;
using ue::rope_angle;
using ue::token_largest;

/** Learnable RMSNorm forward — declared in @ref transformer_block.cuh with external linkage so the
 *  encoder tower launches the same kernel. */
__global__ void ut::rmsnorm_forward_kernel(bf16r2_view_t input, bf16r1_view_t scale, bf16r2_span_t output,
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

/** Learnable RMSNorm backward for the row gradient and the band's scale-gradient partial, with
 *  external linkage; see the header. */
__global__ void ut::rmsnorm_backward_kernel(bf16r2_view_t input, bf16r1_view_t scale, f32r1_view_t inverse_rms,
                                            bf16r2_view_t output_grad, bf16r2_span_t input_grad,
                                            f32r2_span_t scale_grad_partials) {
    constexpr unsigned warps_k = kernel_block_size_default_k / warp_size_k;
    std::size_t const rows = input.extent(0);
    u32_t const width = static_cast<u32_t>(input.extent(1));
    unsigned const lane_index = threadIdx.x % warp_size_k, warp_index = threadIdx.x / warp_size_k;
    std::size_t const band_first = static_cast<std::size_t>(blockIdx.x) * kernel_block_size_default_k;
    std::size_t const band_end = std::min<std::size_t>(rows, band_first + kernel_block_size_default_k);
    for (std::size_t row = band_first + warp_index; row < band_end; row += warps_k) {
        f32_t const inverse = inverse_rms[row];
        f32_t lane_partial = 0.0f;
        for (u32_t channel = lane_index; channel < width; channel += warp_size_k)
            lane_partial += __bfloat162float(output_grad[row, channel]) * __bfloat162float(scale[channel]) *
                            __bfloat162float(input[row, channel]);
        f32_t const projection = warp_reduce_broadcast(lane_partial) * inverse / static_cast<f32_t>(width);
        for (u32_t channel = lane_index; channel < width; channel += warp_size_k) {
            f32_t const normalized = __bfloat162float(input[row, channel]) * inverse;
            f32_t const scaled_grad = __bfloat162float(output_grad[row, channel]) * __bfloat162float(scale[channel]);
            input_grad[row, channel] = __float2bfloat16_rn((scaled_grad - normalized * projection) * inverse);
        }
    }
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t band_total = 0.0f;
#pragma unroll 16 // keeps the band's loads in flight ahead of the ordered sum
        for (std::size_t row = band_first; row < band_end; ++row)
            band_total += __bfloat162float(output_grad[row, channel]) * __bfloat162float(input[row, channel]) *
                          inverse_rms[row];
        scale_grad_partials[blockIdx.x, channel] = band_total;
    }
}

/** Fold the per-band scale-gradient partials in band order — external linkage; see the header. */
__global__ void ut::rmsnorm_scale_grad_kernel(f32r2_view_t scale_grad_partials, bf16r1_span_t scale_grad) {
    constexpr unsigned warps_k = kernel_block_size_default_k / warp_size_k;
    __shared__ f32_t warp_totals[warps_k][warp_size_k];
    std::size_t const bands = scale_grad_partials.extent(0);
    u32_t const width = static_cast<u32_t>(scale_grad_partials.extent(1));
    unsigned const lane_index = threadIdx.x % warp_size_k, warp_index = threadIdx.x / warp_size_k;
    u32_t const channel = blockIdx.x * warp_size_k + lane_index;
    f32_t total = 0.0f;
    if (channel < width)
        for (std::size_t band = warp_index; band < bands; band += warps_k) total += scale_grad_partials[band, channel];
    warp_totals[warp_index][lane_index] = total;
    __syncthreads();
    if (warp_index != 0u || channel >= width) return;
    for (unsigned warp = 1; warp != warps_k; ++warp) total += warp_totals[warp][lane_index];
    scale_grad[channel] = __float2bfloat16_rn(total);
}

/** The row-gradient backward of rmscast and the band's scale-gradient partial, with external
 *  linkage; see the header. */
__global__ void ut::rmscast_backward_kernel(f32r2_view_t input, bf16r1_view_t scale, f32r2_view_t output_grad,
                                            f32r2_view_t residual_grad, f32r2_span_t input_grad,
                                            f32r2_span_t scale_grad_partials, f32_t epsilon) {
    constexpr unsigned warps_k = kernel_block_size_default_k / warp_size_k;
    __shared__ f32_t band_inverse_rms[kernel_block_size_default_k];
    std::size_t const rows = input.extent(0);
    u32_t const width = static_cast<u32_t>(input.extent(1));
    unsigned const lane_index = threadIdx.x % warp_size_k, warp_index = threadIdx.x / warp_size_k;
    std::size_t const band_first = static_cast<std::size_t>(blockIdx.x) * kernel_block_size_default_k;
    std::size_t const band_end = std::min<std::size_t>(rows, band_first + kernel_block_size_default_k);
    bool const residual_present = !residual_grad.empty();
    for (std::size_t row = band_first + warp_index; row < band_end; row += warps_k) {
        f32_t squares_partial = 0.0f, projection_partial = 0.0f;
        for (u32_t channel = lane_index; channel < width; channel += warp_size_k) {
            f32_t const value = input[row, channel];
            squares_partial += value * value;
            projection_partial += output_grad[row, channel] * __bfloat162float(scale[channel]) * value;
        }
        f32_t const inverse = rsqrtf(warp_reduce_broadcast(squares_partial) / static_cast<f32_t>(width) + epsilon);
        f32_t const projection = warp_reduce_broadcast(projection_partial) * inverse / static_cast<f32_t>(width);
        if (lane_index == 0u) band_inverse_rms[row - band_first] = inverse;
        for (u32_t channel = lane_index; channel < width; channel += warp_size_k) {
            f32_t const normalized = input[row, channel] * inverse;
            f32_t const scaled_grad = output_grad[row, channel] * __bfloat162float(scale[channel]);
            f32_t const residual = residual_present ? residual_grad[row, channel] : 0.0f;
            input_grad[row, channel] = residual + (scaled_grad - normalized * projection) * inverse;
        }
    }
    __syncthreads();
    for (u32_t channel = threadIdx.x; channel < width; channel += blockDim.x) {
        f32_t band_total = 0.0f;
#pragma unroll 16 // keeps the band's loads in flight ahead of the ordered sum
        for (std::size_t row = band_first; row < band_end; ++row)
            band_total += output_grad[row, channel] * input[row, channel] * band_inverse_rms[row - band_first];
        scale_grad_partials[blockIdx.x, channel] = band_total;
    }
}

/** Inverted keyed dropout, in place — external linkage; see the header. The @p salt selects
 *  the site (per layer, per residual join) and @p data_epoch , read from device memory,
 *  advances the mask per step; @ref splitmix64_hash_t re-derives the identical keep decision
 *  wherever this runs. */
__global__ void ut::dropout_kernel(bf16r1_span_t values, u32r1_view_t data_epoch, u32_t salt, f32_t drop_probability) {
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
    st::basic_vector<f32_t> host_angles {};
    if (!angles.angles_ || st::failed(host_angles.resize(total)))
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
    if (CUresult result = cuMemcpyHtoD(device_address(angles.angles_.data()), host_angles.data(),
                                       total * sizeof(f32_t));
        result != CUDA_SUCCESS)
        return cuda_failure<rope_angles_t>(result, "angle grid upload failed");
    return result_t {std::move(angles)};
}

/**
 *  @brief Unit RMSNorm backward for the QK-norm, in place on the packed Q|K gradient: @c dx = (dy −
 *      x̂ · mean(dy · x̂)) · (1/rms) over each 128-channel head vector, recomputing @c 1/rms from
 *      the Float32 pre-norm product the forward's rmscast read.
 *
 *  __grid_size__ @c [⌈tokens · vectors · 32/block⌉,1,1] with one @b warp per head vector, the Q
 *      heads then the K heads of each token; each lane holds its 4 channels of both operands in
 *      registers; no dynamic shared memory.
 *
 *  @note @p grads is read-modified in place (the QK-norm backward rescales each head vector here).
 */
static __global__ void transformer_qk_norm_backward_kernel_(bf16r2_span_t grads, f32r2_view_t prenorm,
                                                            u32_t vectors_per_token, f32_t epsilon) {
    constexpr u32_t channels_per_lane_k = transformer_head_dim_k / warp_size_k;
    std::size_t const tokens = grads.extent(0);
    std::size_t const vectors = tokens * vectors_per_token;
    std::size_t const vector = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size_k;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    if (vector >= vectors) return;
    std::size_t const token = vector / vectors_per_token;
    u32_t const lane_channel = static_cast<u32_t>(vector % vectors_per_token) * transformer_head_dim_k + lane_index;
    f32_t values[channels_per_lane_k], value_grads[channels_per_lane_k];
    f32_t squares_partial = 0.0f, projection_partial = 0.0f;
    for (u32_t step = 0; step != channels_per_lane_k; ++step) {
        u32_t const channel = lane_channel + step * warp_size_k;
        values[step] = prenorm[token, channel];
        value_grads[step] = __bfloat162float(grads[token, channel]);
        squares_partial += values[step] * values[step];
        projection_partial += value_grads[step] * values[step];
    }
    f32_t const inverse = rsqrtf(warp_reduce_broadcast(squares_partial) / static_cast<f32_t>(transformer_head_dim_k) +
                                 epsilon);
    f32_t const projection = warp_reduce_broadcast(projection_partial) * inverse /
                             static_cast<f32_t>(transformer_head_dim_k);
    for (u32_t step = 0; step != channels_per_lane_k; ++step) {
        u32_t const channel = lane_channel + step * warp_size_k;
        grads[token, channel] = __float2bfloat16_rn((value_grads[step] - values[step] * inverse * projection) *
                                                    inverse);
    }
}

/**
 *  @brief The per-token cosine and sine rows NumKong's split-half RoPE reads: pair @b k of token
 *      @b t takes the axis angle its index selects, scaled by @p angle_sign , which @c -1 turns
 *      into the inverse rotation the backward applies to the post-RoPE gradients.
 *
 *  __grid_size__ @c [⌈tokens · 64/block⌉,1,1] , one thread per token and pair; no
 *  dynamic shared memory.
 */
static __global__ void transformer_rope_rows_kernel_(f32r2_span_t cosines, f32r2_span_t sines,
                                                     rope_angles_view_t angles, i16r2_view_t coords, f32_t angle_sign) {
    std::size_t const pairs_per_head = cosines.extent(1);
    std::size_t const thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t const token = thread / pairs_per_head;
    u32_t const pair = static_cast<u32_t>(thread % pairs_per_head);
    if (token >= cosines.extent(0)) return;
    f32_t sine = 0, cosine = 0;
    __sincosf(angle_sign * rope_angle(pair, coords, token, angles), &sine, &cosine);
    cosines[token, pair] = cosine;
    sines[token, pair] = sine;
}

/** Rotate every Q and K head vector of the packed QKV @p values in place — the
 *  split-half pair @b (k,k+64) by its axis angle, the NeoX convention — through
 *  NumKong's RoPE over the rows @ref transformer_rope_rows_kernel_ builds into @p rows ,
 *  at least @c tokens · head_dim scalars. */
[[nodiscard]] static expected_t transformer_rotate_(bf16r2_span_t values, f32r1_span_t rows, rope_angles_view_t angles,
                                                    i16r2_view_t coords, u32_t heads, f32_t angle_sign,
                                                    gpu_numerics_t const &numerics, gpu_stream_t &stream) noexcept {
    std::size_t const tokens = values.extent(0), pairs_per_head = transformer_head_dim_k / 2u;
    f32r2_span_t const cosines {rows.data(), tokens, pairs_per_head};
    f32r2_span_t const sines {rows.data() + tokens * pairs_per_head, tokens, pairs_per_head};
    dim3 const grid {static_cast<u32_t>(divide_round_up(tokens * pairs_per_head, kernel_block_size_default_k)), 1u, 1u};
    expected_t staged = launch_kernel(&transformer_rope_rows_kernel_, grid, dim3 {kernel_block_size_default_k, 1u, 1u},
                                      stream, cosines, sines, angles, coords, angle_sign);
    if (staged)
        staged = numkong_result(
            numerics.rope_bf16(numkong_scalars(values.data()), cosines.data(), sines.data(),
                               numkong_scalars(values.data()), tokens, 2u * heads, transformer_head_dim_k,
                               values.stride_bytes(0), values.stride_bytes(0), stream.numkong()),
            "rotary embedding");
    return staged;
}

/**
 *  @brief Zero the rows of @p rows past each segment's length and before the next segment's start:
 *      NumKong attends from those gap rows as from any query, so the forward output and the output
 *      gradient keep them at zero, and no gradient leaks from a gap into the keys it attended.
 *
 *  __grid_size__ @c [segments,1,1] , threads stripe the cells of their segment's gap.
 */
static __global__ void transformer_zero_gaps_kernel_(f32r2_span_t rows, u32r1_view_t segment_offsets,
                                                     u32r1_view_t segment_lengths) {
    u32_t const segment = blockIdx.x;
    std::size_t const columns = rows.extent(1);
    std::size_t const first = static_cast<std::size_t>(segment_offsets[segment]) + segment_lengths[segment];
    std::size_t const cells = (static_cast<std::size_t>(segment_offsets[segment + 1u]) - first) * columns;
    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x)
        rows[first + cell / columns, cell % columns] = 0.0f;
}

/**
 *  @brief SwiGLU activation backward over the Float32 gate|up product, writing the packed
 *      @b [tokens,2·ffn] BFloat16 gate|up gradient: @c d_up = dy · SiLU(gate) and @c d_gate = dy ·
 *      up · σ(gate) · (1+gate · (1 − σ(gate))) .
 *
 *  __grid_size__ @c [tokens,1,1] , one CTA per token whose threads stripe its features; no
 *  dynamic shared memory.
 */
static __global__ void transformer_swiglu_backward_kernel_(f32r2_view_t gate_up, f32r2_view_t silu_up_grad,
                                                           bf16r2_span_t gate_up_grad) {
    std::size_t const token = blockIdx.x;
    u32_t const ffn = static_cast<u32_t>(silu_up_grad.extent(1));
    for (u32_t feature = threadIdx.x; feature < ffn; feature += blockDim.x) {
        f32_t const gate = gate_up[token, feature];
        f32_t const up = gate_up[token, ffn + feature];
        f32_t const grad = silu_up_grad[token, feature];
        f32_t const sigmoid = 1.0f / (1.0f + std::exp(-gate));
        f32_t const silu = gate * sigmoid;
        gate_up_grad[token, feature] = __float2bfloat16_rn(grad * up * sigmoid * (1.0f + gate * (1.0f - sigmoid)));
        gate_up_grad[token, ffn + feature] = __float2bfloat16_rn(grad * silu);
    }
}

/**
 *  @brief Order-free max-abs of one BFloat16 tensor via a bitwise atomic @c fetch_max on the
 *      non-negative Float32 bit pattern — max is commutative and associative, so the result is
 *      deterministic regardless of thread order.
 *
 *  __grid_size__ grid-stride; @p max_bits must be pre-zeroed; no dynamic shared memory.
 */
static __global__ void transformer_absolute_max_kernel_(bf16r1_view_t values, u32r1_span_t max_bits) {
    std::size_t const total = values.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    f32_t thread_max = 0.0f;
    for (std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; cell < total;
         cell += stride)
        thread_max = std::fmax(thread_max, std::fabs(__bfloat162float(values[cell])));
    // Non-negative floats compare identically as unsigned integers, so one atomic
    // per thread suffices.
    cuda::atomic_ref<u32_t, cuda::thread_scope_device> {max_bits[0]}.fetch_max(__float_as_uint(thread_max),
                                                                               cuda::std::memory_order_relaxed);
}

/** Resolve the max-abs bits into the Float32 descale scalar @c max_abs/240 (floored to keep a
 *  zero weight tensor representable). One thread. */
static __global__ void transformer_descale_from_max_kernel_(u32r1_span_t max_bits, f32r1_span_t descale) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    f32_t const max_abs = __uint_as_float(max_bits[0]);
    descale[0] = std::fmax(max_abs / e4m3_range_k, e4m3_descale_floor_k);
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
    tower_scalars_t const scalars;
    f32_t thread_max = token_largest(values, token, threadIdx.x, blockDim.x, scalars);
    // Max through the additive reduce tree: fold |x| via the shuffle butterfly's max analogue.
    for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
        thread_max = std::fmax(thread_max, __shfl_xor_sync(0xffffffffu, thread_max, delta));
    __shared__ f32_t warp_partials[kernel_block_size_default_k / warp_size_k];
    __shared__ f32_t token_max;
    unsigned const lane_index = threadIdx.x % warp_size_k;
    unsigned const warp_index = threadIdx.x / warp_size_k;
    if (lane_index == 0u) warp_partials[warp_index] = thread_max;
    __syncthreads();
    if (warp_index == 0u) {
        f32_t staged = lane_index < kernel_block_size_default_k / warp_size_k ? warp_partials[lane_index] : 0.0f;
        for (unsigned delta = warp_size_k / 2u; delta != 0u; delta /= 2u)
            staged = std::fmax(staged, __shfl_xor_sync(0xffffffffu, staged, delta));
        if (lane_index == 0u) token_max = staged;
    }
    __syncthreads();
    f32_t const scale = quantize_token(values, shadow, token, token_max, threadIdx.x, blockDim.x, scalars);
    if (threadIdx.x == 0u) token_scales[token] = scale;
}

/**
 *  @brief Fold the scales back into a Float32 product and write it as BFloat16:
 *      @c values[t,o] = products[t,o]·token_scales[t]·weight_descale[0] + residual[t,o] , where an empty
 *      @p token_scales or @p residual drops its term.
 *
 *  __grid_size__ grid-stride over @c tokens · width cells; no dynamic shared memory.
 */
static __global__ void transformer_token_rescale_kernel_(f32r2_view_t products, f32r1_view_t token_scales,
                                                         f32r1_view_t weight_descale, bf16r2_view_t residual,
                                                         bf16r2_span_t values) {
    rescale_tokens_cells(products, token_scales, weight_descale[0], residual, values,
                         static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                         static_cast<std::size_t>(gridDim.x) * blockDim.x, tower_scalars_t {});
}

expected<transformer_scratch_t> transformer_scratch_t::make(gpu_arena_t &arena,
                                                            transformer_block_config_t const &config,
                                                            std::span<u32_t const> entry_inputs, u32_t token_width,
                                                            gpu_numerics_t const &numerics) noexcept {
    using result_t = expected<transformer_scratch_t>;
    if (config.hidden == 0 || config.max_tokens == 0)
        return result_t::fail(status_t::invalid_argument_k, "scratch capacities must be positive (hidden={} tokens={})",
                              config.hidden, config.max_tokens);
    std::size_t const hidden = config.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config.ffn_expansion) * hidden;
    std::size_t const tokens = config.max_tokens;
    std::size_t const padded_tokens = round_up_to_multiple(tokens, std::size_t {8});
    std::size_t const widest = std::max(3 * hidden, 2 * ffn);
    std::size_t widest_entry = 0;
    for (u32_t const entry : entry_inputs) widest_entry = std::max<std::size_t>(widest_entry, entry);
    std::size_t const inputs = std::max({hidden, ffn, widest_entry});
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};

    std::size_t const pack_shapes[][2] = {{hidden, tokens},
                                          {ffn, tokens},
                                          {inputs, tokens},
                                          {hidden, config.max_segments}};
    std::size_t packed_bytes = 0;
    for (auto const &[columns, depth] : pack_shapes) {
        expected<std::size_t> shape_bytes = pack_bytes_bf16(columns, depth, numerics);
        if (!shape_bytes) return result_t::fail(std::move(shape_bytes));
        packed_bytes = std::max(packed_bytes, shape_bytes.value_);
    }
    // The split depth's partials are not monotone in a weight's width, so every shape is asked: the
    // block's four, the tower head's, the token head's, then each entry's.
    std::size_t const weight_shapes[][3] = {{3 * hidden, hidden, tokens},
                                            {hidden, hidden, tokens},
                                            {2 * ffn, hidden, tokens},
                                            {hidden, ffn, tokens},
                                            {hidden, hidden, config.max_segments},
                                            {token_width, hidden, tokens}};
    std::size_t const fixed_shapes = std::size(weight_shapes);
    std::size_t workspace_bytes = 0;
    for (std::size_t shape = 0; shape != fixed_shapes + entry_inputs.size(); ++shape) {
        bool const fixed = shape < fixed_shapes;
        std::size_t const outputs = fixed ? weight_shapes[shape][0] : hidden;
        std::size_t const weight_inputs = fixed ? weight_shapes[shape][1] : entry_inputs[shape - fixed_shapes];
        std::size_t const depth = fixed ? weight_shapes[shape][2] : tokens;
        if (outputs == 0 || weight_inputs == 0) continue;
        expected<std::size_t> shape_bytes = weight_gradient_workspace_bytes_bf16(outputs, weight_inputs, depth,
                                                                                 numerics);
        if (!shape_bytes) return result_t::fail(std::move(shape_bytes));
        workspace_bytes = std::max(workspace_bytes, shape_bytes.value_);
    }
    nk_size_t packed_keys_values_bytes = 0;
    if (nk_status_t const status = numerics.attention_pack_size_bf16(hidden / transformer_head_dim_k,
                                                                     transformer_head_dim_k, tokens,
                                                                     config.max_segments, &packed_keys_values_bytes);
        status != nk_success_k)
        return result_t::fail(status_t::device_error_k, "attention pack sizing failed: {}", nk_status_name(status));
    transformer_scratch_t scratch;
    scratch.products = f32r1_tensor_arena_t::uninitialized({tokens * ffn}, f32_alloc).value;
    scratch.residual_grad = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    scratch.narrowed_grad = bf16r1_tensor_arena_t::uninitialized({tokens * widest}, bf16_alloc).value;
    scratch.transposed_grad = bf16r1_tensor_arena_t::uninitialized({widest * padded_tokens}, bf16_alloc).value;
    scratch.transposed_input = bf16r1_tensor_arena_t::uninitialized({inputs * padded_tokens}, bf16_alloc).value;
    scratch.transposed_weight = bf16r1_tensor_arena_t::uninitialized({widest * hidden}, bf16_alloc).value;
    scratch.packed_qkv = bf16r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, bf16_alloc).value;
    scratch.qkv_grad = f32r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, f32_alloc).value;
    scratch.packed_keys_values =
        b8r1_tensor_arena_t::uninitialized({static_cast<std::size_t>(packed_keys_values_bytes)}, b8_alloc).value;
    scratch.weight_grad_products =
        f32r1_tensor_arena_t::uninitialized({std::max(widest, inputs) * hidden}, f32_alloc).value;
    scratch.packed_input = b8r1_tensor_arena_t::uninitialized({packed_bytes}, b8_alloc).value;
    scratch.weight_gradient_workspace = b8r1_tensor_arena_t::uninitialized({workspace_bytes}, b8_alloc).value;
    std::size_t const heads = hidden / transformer_head_dim_k;
    scratch.qkv_products = f32r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, f32_alloc).value;
    scratch.attn_residual = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    scratch.gate_up_products = f32r2_tensor_arena_t::uninitialized({tokens, 2 * ffn}, f32_alloc).value;
    scratch.attention_products = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    scratch.normalized = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    scratch.attn_output = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    scratch.mlp_normalized = bf16r2_tensor_arena_t::uninitialized({tokens, hidden}, bf16_alloc).value;
    scratch.silu_up = bf16r2_tensor_arena_t::uninitialized({tokens, ffn}, bf16_alloc).value;
    scratch.qkv = bf16r2_tensor_arena_t::uninitialized({tokens, 3 * hidden}, bf16_alloc).value;
    scratch.softmax_log_sum_exp = f32r1_tensor_arena_t::uninitialized({tokens * heads}, f32_alloc).value;
    if (!scratch.products || !scratch.residual_grad || !scratch.narrowed_grad || !scratch.transposed_grad ||
        !scratch.transposed_input || !scratch.transposed_weight || !scratch.packed_qkv || !scratch.qkv_grad ||
        !scratch.packed_keys_values || !scratch.weight_grad_products || !scratch.packed_input ||
        (workspace_bytes != 0 && !scratch.weight_gradient_workspace) || !scratch.qkv_products ||
        !scratch.attn_residual || !scratch.gate_up_products || !scratch.attention_products || !scratch.normalized ||
        !scratch.attn_output || !scratch.mlp_normalized || !scratch.silu_up || !scratch.qkv ||
        !scratch.softmax_log_sum_exp)
        return result_t::fail(status_t::bad_alloc_k, "scratch allocation failed (arena_used={})", arena.used());
    return result_t {std::move(scratch)};
}

expected<transformer_block_bf16_t> transformer_block_bf16_t::make(gpu_arena_t &arena, transformer_scratch_t &scratch,
                                                                  transformer_block_config_t config,
                                                                  gpu_numerics_t const &numerics) noexcept {
    if (config.hidden == 0 || config.hidden % transformer_head_dim_k != 0)
        return expected<transformer_block_bf16_t>::fail(status_t::unsupported_k,
                                                        "hidden must be a positive multiple of {} (got {})",
                                                        transformer_head_dim_k, config.hidden);
    if (config.max_tokens == 0 || config.max_segments == 0 || config.max_segment_tokens == 0)
        return expected<transformer_block_bf16_t>::fail(
            status_t::invalid_argument_k, "token capacities must be positive (tokens={} segments={} longest={})",
            config.max_tokens, config.max_segments, config.max_segment_tokens);
    transformer_block_bf16_t block;
    block.config_ = config;
    block.scratch_ = &scratch;
    block.numerics_ = numerics;

    std::size_t const hidden = config.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config.ffn_expansion) * hidden;
    std::size_t const tokens = config.max_tokens;
    std::size_t const heads = hidden / transformer_head_dim_k;
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    // Plain `uninitialized` allocations, checked in named groups so a failure names
    // the arena pressure.

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
    // The input the backward and its forward replay start from; other stashes are the scratch's.
    block.input_stash_ = f32r2_tensor_arena_t::uninitialized({tokens, hidden}, f32_alloc).value;
    if (!block.input_stash_)
        return expected<transformer_block_bf16_t>::fail(status_t::bad_alloc_k,
                                                        "input stash allocation failed (arena_used={})", arena.used());

    // Each weight packs as the forward's B operand [out,in] and, transposed, as the
    // data gradient's [in,out].
    bf16r2_tensor_arena_t const *const weights[] = {&block.qkv_weight_, &block.out_weight_, &block.gate_up_weight_,
                                                    &block.down_weight_};
    for (std::size_t product = 0; product != 4; ++product) {
        std::size_t const outputs = weights[product]->extent(0), inputs = weights[product]->extent(1);
        expected<b8r1_tensor_arena_t> packed = allocate_pack_bf16(outputs, inputs, block.numerics_, arena);
        if (!packed) return expected<transformer_block_bf16_t>::fail(std::move(packed));
        expected<b8r1_tensor_arena_t> transposed = allocate_pack_bf16(inputs, outputs, block.numerics_, arena);
        if (!transposed) return expected<transformer_block_bf16_t>::fail(std::move(transposed));
        block.packed_weights_[product] = std::move(packed.value_);
        block.packed_weights_transposed_[product] = std::move(transposed.value_);
    }
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
        return expected_t::fail(status_t::unsupported_k, "input [{}x{}] disagrees with the baked [{}x{}]", tokens,
                                input.extent(1), config_.max_tokens, hidden);
    if (output.extent(0) != tokens || output.extent(1) != hidden)
        return expected_t::fail(status_t::unsupported_k, "output shape disagrees with the input");
    if (segment_lengths.extent(0) != config_.max_segments || segment_offsets.extent(0) != config_.max_segments + 1u)
        return expected_t::fail(status_t::unsupported_k,
                                "segment descriptors disagree with the baked count ({} segments)",
                                config_.max_segments);
    if (coords.extent(0) < tokens)
        return expected_t::fail(status_t::invalid_argument_k, "rope coordinates cover {} of {} tokens",
                                coords.extent(0), tokens);
    // Every stash's live extent records the step's token count — the shape the backward reads back,
    // never a shadow scalar. Resizing within the reserved capacity moves no address, and the
    // `tokens <= max_tokens` check above guarantees every one of these succeeds.

    bool const resized = nk::succeeded(input_stash_.resize({tokens, hidden})) &&
                         nk::succeeded(scratch_->qkv_products.resize({tokens, 3 * hidden})) &&
                         nk::succeeded(scratch_->attn_residual.resize({tokens, hidden})) &&
                         nk::succeeded(scratch_->gate_up_products.resize({tokens, 2 * ffn})) &&
                         nk::succeeded(scratch_->normalized.resize({tokens, hidden})) &&
                         nk::succeeded(scratch_->qkv.resize({tokens, 3 * hidden})) &&
                         nk::succeeded(scratch_->attention_products.resize({tokens, hidden})) &&
                         nk::succeeded(scratch_->attn_output.resize({tokens, hidden})) &&
                         nk::succeeded(scratch_->mlp_normalized.resize({tokens, hidden})) &&
                         nk::succeeded(scratch_->silu_up.resize({tokens, ffn}));
    if (!resized)
        return expected_t::fail(status_t::unsupported_k, "activation stashes cannot hold {} tokens (baked {})", tokens,
                                config_.max_tokens);

    // The backward's pre-norm derivative needs the raw input; a replay already reads it from here.
    if (input.data() != input_stash_.data())
        if (expected_t stashed = copy_device_to_device(f32r1_view_t {input.data(), tokens * hidden},
                                                       f32r1_span_t {input_stash_.data(), tokens * hidden}, stream);
            !stashed)
            return stashed;

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    f32_t const epsilon = config_.rms_eps;
    f32_t *const pre_scale = widened_norm_scales_.data();
    f32_t *const post_scale = pre_scale + hidden;
    std::size_t const qkv_stride = 3 * hidden * sizeof(f32_t), qkv_narrow_stride = 3 * hidden * sizeof(bf16_t);
    bf16r2_span_t const packed_qkv = scratch_->qkv.span();

    // The norms read Float32 γ, widened from the trained BFloat16 scales every step.
    expected_t staged = numkong_result(
        numerics_.cast(pre_norm_scale_.data(), nk_bf16_k, pre_scale, nk_f32_k, hidden, stream.numkong()),
        "pre-attention scale widening");
    if (staged)
        staged = numkong_result(
            numerics_.cast(post_norm_scale_.data(), nk_bf16_k, post_scale, nk_f32_k, hidden, stream.numkong()),
            "post-attention scale widening");
    // Pre-attention norm narrows the residual stream into the fused QKV product's operand.
    if (staged)
        staged = numkong_result(
            numerics_.rmscast_bf16(input.data(), pre_scale, numkong_scalars(scratch_->normalized.data()), tokens, 1,
                                   hidden, hidden * sizeof(f32_t), hidden * sizeof(bf16_t), epsilon, stream.numkong()),
            "pre-attention norm");
    if (staged)
        staged = multiply_bf16(scratch_->normalized.view(), packed_weights_[product_qkv_k].view(),
                               scratch_->qkv_products.span(), numerics_, stream);
    // `nk_cast` has no row stride, so narrowing V narrows whole rows; the QK-norm then overwrites the Q|K slices.
    if (staged)
        staged = numkong_result(numerics_.cast(scratch_->qkv_products.data(), nk_f32_k, packed_qkv.data(), nk_bf16_k,
                                               tokens * 3 * hidden, stream.numkong()),
                                "QKV narrowing");
    if (staged)
        staged = numkong_result(
            numerics_.rmscast_bf16(scratch_->qkv_products.data(), NUMKONG_NULL, numkong_scalars(packed_qkv.data()),
                                   tokens, 2 * heads, transformer_head_dim_k, qkv_stride, qkv_narrow_stride, epsilon,
                                   stream.numkong()),
            "QK-norm");
    // The rotary rows borrow the SwiGLU products, which the MLP below writes only after the
    // rotation read them.
    if (staged)
        staged = transformer_rotate_(packed_qkv, scratch_->products.span(), angles, coords, heads, 1.0f, numerics_,
                                     stream);

    // Attention over the segments, stashing its Float32 output and log-sum-exp for the backward.
    if (staged) staged = pack_keys_values_(segment_offsets, segment_lengths, stream);
    if (staged)
        staged = numkong_result(
            numerics_.attend_bf16(
                heads, heads, transformer_head_dim_k, segment_offsets.data(), tokens, transformer_attention_scale(),
                NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX, numkong_scalars(scratch_->qkv.data()), qkv_narrow_stride,
                scratch_->packed_keys_values.data(), scratch_->attention_products.data(), hidden * sizeof(f32_t),
                scratch_->softmax_log_sum_exp.data(), 0, NUMKONG_SIZE_MAX, NUMKONG_NULL, 0, stream.numkong()),
            "attention");
    dim3 const segment_grid {static_cast<u32_t>(segment_lengths.extent(0)), 1u, 1u};
    if (staged)
        staged = launch_kernel(&transformer_zero_gaps_kernel_, segment_grid, block_shape, stream,
                               scratch_->attention_products.span(), segment_offsets, segment_lengths);
    if (staged)
        staged = numkong_result(
            numerics_.cast(scratch_->attention_products.data(), nk_f32_k, scratch_->attn_output.data(), nk_bf16_k,
                           tokens * hidden, stream.numkong()),
            "attention narrowing");
    if (!staged) return staged;

    // Out-projection, the residual into the Float32 stream, and the post-attention
    // norm narrowing it.
    staged = multiply_bf16(scratch_->attn_output.view(), packed_weights_[product_out_k].view(),
                           scratch_->attn_residual.span(), numerics_, stream);
    if (staged)
        staged = numkong_result(numerics_.sum_f32(scratch_->attn_residual.data(), input.data(), tokens * hidden,
                                                  scratch_->attn_residual.data(), stream.numkong()),
                                "attention residual");
    if (staged)
        staged = numkong_result(
            numerics_.rmscast_bf16(scratch_->attn_residual.data(), post_scale,
                                   numkong_scalars(scratch_->mlp_normalized.data()), tokens, 1, hidden,
                                   hidden * sizeof(f32_t), hidden * sizeof(bf16_t), epsilon, stream.numkong()),
            "post-attention norm");

    // SwiGLU MLP: fused gate/up product, SiLU·up in Float32 narrowed by a cast,
    // down product, residual.
    if (staged)
        staged = multiply_bf16(scratch_->mlp_normalized.view(), packed_weights_[product_gate_up_k].view(),
                               scratch_->gate_up_products.span(), numerics_, stream);
    f32_t const *const gate = scratch_->gate_up_products.data();
    std::size_t const gate_up_stride = 2 * ffn * sizeof(f32_t);
    if (staged)
        staged = numkong_result(
            numerics_.swiglu_f32(gate, gate + ffn, scratch_->products.data(), tokens, ffn, gate_up_stride,
                                 gate_up_stride, ffn * sizeof(f32_t), 1.0f, 1.0f, stream.numkong()),
            "SwiGLU");
    if (staged)
        staged = numkong_result(numerics_.cast(scratch_->products.data(), nk_f32_k, scratch_->silu_up.data(), nk_bf16_k,
                                               tokens * ffn, stream.numkong()),
                                "SwiGLU narrowing");
    if (staged)
        staged = multiply_bf16(scratch_->silu_up.view(), packed_weights_[product_down_k].view(), output, numerics_,
                               stream);
    if (staged)
        staged = numkong_result(numerics_.sum_f32(output.data(), scratch_->attn_residual.data(), tokens * hidden,
                                                  output.data(), stream.numkong()),
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
    if (tokens == 0) return expected_t::fail(status_t::invalid_argument_k, "backward before a stashed forward");
    if (output_grad.extent(0) != tokens || output_grad.extent(1) != hidden || input_grad.extent(0) != tokens ||
        input_grad.extent(1) != hidden)
        return expected_t::fail(status_t::unsupported_k, "gradient shapes disagree with the stashed [{}x{}]", tokens,
                                hidden);
    // Backward-only transients the forward never touches — their live extent still needs setting to
    // this step's token count (the stashes above already carry it from the paired forward call).

    transformer_scratch_t &scratch = *scratch_;
    bool const resized = nk::succeeded(scratch.packed_qkv.resize({tokens, 3 * hidden})) &&
                         nk::succeeded(scratch.residual_grad.resize({tokens, hidden}));
    if (!resized)
        return expected_t::fail(status_t::unsupported_k, "gradient transients cannot hold {} tokens (baked {})", tokens,
                                config_.max_tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    std::size_t const bands = divide_round_up(tokens, std::size_t {kernel_block_size_default_k});
    dim3 const band_grid {static_cast<u32_t>(bands), 1u, 1u};
    dim3 const channel_grid {static_cast<u32_t>(divide_round_up(hidden, warp_size_k)), 1u, 1u};
    f32_t const epsilon = config_.rms_eps;
    f32r2_span_t const scale_grad_partials {scratch.qkv_grad.data(), bands, hidden}; // dead outside the attention
    bf16r2_span_t const hidden_narrowed {scratch.narrowed_grad.data(), tokens, hidden};
    bf16r2_span_t const gate_up_grad {scratch.narrowed_grad.data(), tokens, 2 * ffn};
    f32r2_span_t const silu_up_grad {scratch.products.data(), tokens, ffn};
    f32r2_span_t const hidden_products {scratch.products.data(), tokens, hidden};
    auto weight_gradient = [&](bf16r2_view_t product_grad, bf16r2_view_t product_input,
                               bf16r2_span_t weight_grad) noexcept {
        return weight_gradient_bf16(product_grad, product_input, weight_grad, scratch.transposed_grad.span(),
                                    scratch.transposed_input.span(), scratch.packed_input.span(),
                                    scratch.weight_grad_products.span(), scratch.weight_gradient_workspace.span(),
                                    numerics_, stream);
    };

    // MLP backward: the output gradient narrows into the A operand of
    // both down-projection gradients.
    expected_t staged = numkong_result(numerics_.cast(output_grad.data(), nk_f32_k, hidden_narrowed.data(), nk_bf16_k,
                                                      tokens * hidden, stream.numkong()),
                                       "output gradient narrowing");
    if (staged)
        staged = multiply_bf16(hidden_narrowed, packed_weights_transposed_[product_down_k].view(), silu_up_grad,
                               numerics_, stream);
    if (staged) staged = weight_gradient(hidden_narrowed, scratch_->silu_up.view(), down_weight_grad_.span());
    // The gate/up gradient overwrites the narrowed output gradient both
    // down-projection products consumed.
    if (staged)
        staged = launch_kernel(&transformer_swiglu_backward_kernel_, dim3 {static_cast<u32_t>(tokens), 1u, 1u},
                               block_shape, stream, scratch_->gate_up_products.view(), f32r2_view_t {silu_up_grad},
                               gate_up_grad);
    if (staged)
        staged = multiply_bf16(gate_up_grad, packed_weights_transposed_[product_gate_up_k].view(), hidden_products,
                               numerics_, stream);
    if (staged) staged = weight_gradient(gate_up_grad, scratch_->mlp_normalized.view(), gate_up_weight_grad_.span());

    // Post-attention norm backward: the residual-stream gradient adds the output gradient.
    if (staged)
        staged = launch_kernel(&rmscast_backward_kernel, band_grid, block_shape, stream, scratch_->attn_residual.view(),
                               post_norm_scale_.view(), f32r2_view_t {hidden_products}, output_grad,
                               scratch.residual_grad.span(), scale_grad_partials, epsilon);
    if (staged)
        staged = launch_kernel(&rmsnorm_scale_grad_kernel, channel_grid, block_shape, stream,
                               f32r2_view_t {scale_grad_partials}, post_norm_scale_grad_.span());

    // Out-projection backward, whose Float32 data gradient is the attention's output gradient.
    if (staged)
        staged = numkong_result(numerics_.cast(scratch.residual_grad.data(), nk_f32_k, hidden_narrowed.data(),
                                               nk_bf16_k, tokens * hidden, stream.numkong()),
                                "residual gradient narrowing");
    if (staged)
        staged = multiply_bf16(hidden_narrowed, packed_weights_transposed_[product_out_k].view(), hidden_products,
                               numerics_, stream);
    if (staged) staged = weight_gradient(hidden_narrowed, scratch_->attn_output.view(), out_weight_grad_.span());

    // Attention backward into the zeroed Float32 dQ|dK|dV rows, narrowed into the packed gradient,
    // then the inverse rotation and the QK-norm backward.
    f32r2_span_t const qkv_grad {scratch.qkv_grad.data(), tokens, 3 * hidden};
    std::size_t const qkv_grad_stride = 3 * hidden * sizeof(f32_t);
    if (staged) staged = zero_async(qkv_grad, stream);
    if (staged)
        staged = launch_kernel(&transformer_zero_gaps_kernel_,
                               dim3 {static_cast<u32_t>(segment_lengths.extent(0)), 1u, 1u}, block_shape, stream,
                               hidden_products, segment_offsets, segment_lengths);
    if (staged) staged = pack_keys_values_(segment_offsets, segment_lengths, stream);
    if (staged)
        staged = numkong_result(
            numerics_.attend_gradients_bf16(
                heads, heads, transformer_head_dim_k, segment_offsets.data(), tokens, transformer_attention_scale(),
                NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX, numkong_scalars(scratch_->qkv.data()), 3 * hidden * sizeof(bf16_t),
                scratch.packed_keys_values.data(), scratch_->attention_products.data(), hidden_products.data(),
                hidden * sizeof(f32_t), scratch_->softmax_log_sum_exp.data(), qkv_grad.data(), qkv_grad_stride,
                qkv_grad.data() + hidden, qkv_grad.data() + 2 * hidden, qkv_grad_stride, 0, NUMKONG_SIZE_MAX,
                NUMKONG_NULL, 0, stream.numkong()),
            "attention backward");
    if (staged)
        staged = numkong_result(numerics_.cast(qkv_grad.data(), nk_f32_k, scratch.packed_qkv.data(), nk_bf16_k,
                                               tokens * 3 * hidden, stream.numkong()),
                                "attention gradient narrowing");
    if (!staged) return staged;
    std::size_t const head_vectors = tokens * heads;
    dim3 const vector_grid {
        static_cast<u32_t>(divide_round_up(2 * head_vectors * warp_size_k, kernel_block_size_default_k)), 1u, 1u};
    // The attention backward has read its output gradient out of the products, so they hold
    // the rotary rows.
    staged = transformer_rotate_(scratch.packed_qkv.span(), scratch.products.span(), angles, coords, heads, -1.0f,
                                 numerics_, stream);
    if (staged)
        staged = launch_kernel(&transformer_qk_norm_backward_kernel_, vector_grid, block_shape, stream,
                               scratch.packed_qkv.span(), scratch_->qkv_products.view(), 2u * heads, epsilon);

    // QKV projection backward, then the pre-attention norm backward adding
    // the residual-stream gradient.
    if (staged)
        staged = multiply_bf16(scratch.packed_qkv.view(), packed_weights_transposed_[product_qkv_k].view(),
                               hidden_products, numerics_, stream);
    if (staged)
        staged = weight_gradient(scratch.packed_qkv.view(), scratch_->normalized.view(), qkv_weight_grad_.span());
    if (staged)
        staged = launch_kernel(&rmscast_backward_kernel, band_grid, block_shape, stream, input_stash_.view(),
                               pre_norm_scale_.view(), f32r2_view_t {hidden_products}, scratch.residual_grad.view(),
                               input_grad, scale_grad_partials, epsilon);
    if (staged)
        staged = launch_kernel(&rmsnorm_scale_grad_kernel, channel_grid, block_shape, stream,
                               f32r2_view_t {scale_grad_partials}, pre_norm_scale_grad_.span());
    return staged;
}

expected_t transformer_block_bf16_t::pack_weights(gpu_stream_t &stream) noexcept {
    bf16r2_tensor_arena_t const *const weights[] = {&qkv_weight_, &out_weight_, &gate_up_weight_, &down_weight_};
    for (std::size_t product = 0; product != 4; ++product)
        if (expected_t packed = pack_weight_bf16(weights[product]->view(), scratch_->transposed_weight.span(),
                                                 packed_weights_[product].span(),
                                                 packed_weights_transposed_[product].span(), numerics_, stream);
            !packed)
            return packed;
    return {};
}

expected_t transformer_block_bf16_t::pack_keys_values_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                       gpu_stream_t &stream) noexcept {
    u32_t const hidden = config_.hidden, heads = hidden / transformer_head_dim_k;
    std::size_t const segments = segment_lengths.extent(0), stride = 3 * hidden * sizeof(bf16_t);
    nk_bf16_t const *const qkv = numkong_scalars(scratch_->qkv.data());
    return numkong_result(
        numerics_.attention_pack_bf16(heads, transformer_head_dim_k, segment_offsets.data(), segment_lengths.data(),
                                      segments, qkv + hidden, stride, qkv + 2 * hidden, stride,
                                      scratch_->packed_keys_values.data(), 0, segments * heads, stream.numkong()),
        "attention key-value pack");
}

expected_t transformer_block_e4m3_t::forward(bf16r2_view_t input, bf16r2_span_t output, u32r1_view_t segment_offsets,
                                             u32r1_view_t segment_lengths, rope_angles_view_t angles,
                                             i16r2_view_t coords, gpu_stream_t &stream) noexcept {
    std::size_t const tokens = input.extent(0);
    u32_t const hidden = config_.hidden;
    u32_t const ffn = config_.ffn_expansion * hidden;
    u32_t const heads = hidden / transformer_head_dim_k;
    if (tokens == 0 || tokens > config_.max_tokens || input.extent(1) != hidden)
        return expected_t::fail(status_t::unsupported_k, "input [{}x{}] disagrees with the baked [{}x{}]", tokens,
                                input.extent(1), config_.max_tokens, hidden);
    if (output.extent(0) != tokens || output.extent(1) != hidden)
        return expected_t::fail(status_t::unsupported_k, "output shape disagrees with the input");
    // No backward reads any of these, but each still needs its live extent set before `.span()` /
    // `.view()` reflects this step's token count rather than the baked `max_tokens` capacity.

    bool const resized =
        nk::succeeded(normalized_.resize({tokens, hidden})) && nk::succeeded(qkv_.resize({tokens, 3 * hidden})) &&
        nk::succeeded(attn_residual_.resize({tokens, hidden})) && nk::succeeded(gate_up_.resize({tokens, 2 * ffn})) &&
        nk::succeeded(silu_up_.resize({tokens, ffn})) && nk::succeeded(normalized_e4m3_.resize({tokens, hidden})) &&
        nk::succeeded(qkv_e4m3_.resize({tokens, 3 * hidden})) &&
        nk::succeeded(attn_output_e4m3_.resize({tokens, hidden})) && nk::succeeded(silu_up_e4m3_.resize({tokens, ffn}));
    if (!resized)
        return expected_t::fail(status_t::unsupported_k, "activation stashes cannot hold {} tokens (baked {})", tokens,
                                config_.max_tokens);

    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const token_grid {static_cast<u32_t>(tokens), 1u, 1u};
    f32r1_view_t const token_scales_view = token_scales_.view();
    // RMSNorm of `groups` vectors of `columns` per token, in place when `target` is `source`.
    auto const normalize = [&](bf16r2_view_t source, std::span<f32_t const> gamma, bf16r2_span_t target,
                               std::size_t groups, std::size_t columns) -> expected_t {
        nk_status_t const status = numerics_.rmsnorm_bf16(
            numkong_scalars(source.data()), gamma.data(), numkong_scalars(target.data()), tokens, groups, columns,
            source.stride_bytes(0), target.stride_bytes(0), config_.rms_eps, stream.numkong());
        if (status != nk_success_k)
            return expected_t::fail(status_t::device_error_k, "RMSNorm failed: {}", nk_status_name(status));
        return {};
    };

    // Pre-attention RMSNorm in BFloat16, per-token E4M3 quantisation, QKV product, scale fold.
    expected_t launched = normalize(input, std::span<f32_t const> {pre_norm_scale_.data(), hidden}, normalized_.span(),
                                    1, hidden);
    if (launched)
        launched = launch_kernel(&transformer_token_quantize_kernel_<bf16_t>, token_grid, block_shape, stream,
                                 normalized_.view(), normalized_e4m3_.span(), token_scales_.span());
    if (!launched) return launched;
    launched = project_(product_qkv_k, normalized_e4m3_.view(), token_scales_view, {}, qkv_.span(), stream);
    if (!launched) return launched;
    // Unit QK-norm of every Q and K head vector, adjacent in each token's row, then M-RoPE.
    launched = normalize(qkv_.view(), {}, qkv_.span(), 2 * heads, transformer_head_dim_k);
    if (!launched) return launched;
    // The QKV product's scale fold has read the Float32 products, so they hold the rotary rows
    // until the attention.
    launched = transformer_rotate_(qkv_.span(), products_.span().flatten<1>(), angles, coords, heads, 1.0f, numerics_,
                                   stream);
    if (!launched) return launched;

    // QK-norm leaves Q and K at unit RMS per head and V stays well inside E4M3's range, so all
    // three cast unscaled. The gaps between segments ride along as extra query rows of the segment
    // before; nothing reads them.
    std::size_t const segments = segment_lengths.extent(0);
    if (nk_status_t const status = numerics_.cast(qkv_.data(), nk_bf16_k, qkv_e4m3_.data(), nk_e4m3_k,
                                                  tokens * 3 * hidden, stream.numkong());
        status != nk_success_k)
        return expected_t::fail(status_t::device_error_k, "Q/K/V cast failed: {}", nk_status_name(status));
    if (nk::failed(products_.resize({tokens, hidden})))
        return expected_t::fail(status_t::unsupported_k, "attention scratch cannot hold [{}x{}]", tokens, hidden);
    nk_e4m3_t const *const qkv_e4m3 = numkong_scalars(qkv_e4m3_.data());
    std::size_t const qkv_bytes = 3 * hidden * sizeof(e4m3_t);
    nk_status_t status = numerics_.attention_pack_e4m3(
        heads, transformer_head_dim_k, segment_offsets.data(), segment_lengths.data(), segments, qkv_e4m3 + hidden,
        qkv_bytes, qkv_e4m3 + 2 * hidden, qkv_bytes, packed_keys_values_.data(), 0, segments * heads, stream.numkong());
    if (status == nk_success_k)
        status = numerics_.attend_e4m3(heads, heads, transformer_head_dim_k, segment_offsets.data(), tokens,
                                       transformer_attention_scale(), NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX, qkv_e4m3,
                                       qkv_bytes, packed_keys_values_.data(), products_.data(), hidden * sizeof(f32_t),
                                       NUMKONG_NULL, 0, NUMKONG_SIZE_MAX, NUMKONG_NULL, 0, stream.numkong());
    if (status != nk_success_k)
        return expected_t::fail(status_t::device_error_k, "E4M3 attention failed: {}", nk_status_name(status));

    // Out-projection consumes the attention output under per-token scales, adding
    // the residual input.
    launched = launch_kernel(&transformer_token_quantize_kernel_<f32_t>, token_grid, block_shape, stream,
                             products_.view(), attn_output_e4m3_.span(), token_scales_.span());
    if (launched)
        launched = project_(product_out_k, attn_output_e4m3_.view(), token_scales_view, input, attn_residual_.span(),
                            stream);
    if (launched)
        launched = normalize(attn_residual_.view(), std::span<f32_t const> {post_norm_scale_.data(), hidden},
                             normalized_.span(), 1, hidden);
    if (launched)
        launched = launch_kernel(&transformer_token_quantize_kernel_<bf16_t>, token_grid, block_shape, stream,
                                 normalized_.view(), normalized_e4m3_.span(), token_scales_.span());
    if (!launched) return launched;
    launched = project_(product_gate_up_k, normalized_e4m3_.view(), token_scales_view, {}, gate_up_.span(), stream);
    if (!launched) return launched;
    nk_bf16_t const *const gate_up = numkong_scalars(gate_up_.data());
    if (nk_status_t const status = numerics_.swiglu_bf16(
            gate_up, gate_up + ffn, numkong_scalars(silu_up_.data()), tokens, ffn, 2 * ffn * sizeof(bf16_t),
            2 * ffn * sizeof(bf16_t), ffn * sizeof(bf16_t), 1.0f, 1.0f, stream.numkong());
        status != nk_success_k)
        return expected_t::fail(status_t::device_error_k, "SwiGLU failed: {}", nk_status_name(status));
    if (launched)
        launched = launch_kernel(&transformer_token_quantize_kernel_<bf16_t>, token_grid, block_shape, stream,
                                 silu_up_.view(), silu_up_e4m3_.span(), token_scales_.span());
    if (!launched) return launched;
    return project_(product_down_k, silu_up_e4m3_.view(), token_scales_view, attn_residual_.view(), output, stream);
}

expected_t transformer_block_e4m3_t::project_(std::size_t product, e4m3r2_view_t activations, f32r1_view_t token_scales,
                                              bf16r2_view_t residual, bf16r2_span_t values,
                                              gpu_stream_t &stream) noexcept {
    std::size_t const tokens = activations.extent(0), depth = activations.extent(1), width = values.extent(1);
    if (nk::failed(products_.resize({tokens, width})))
        return expected_t::fail(status_t::unsupported_k, "product scratch cannot hold [{}x{}]", tokens, width);
    nk_status_t const status = numerics_.multiply_e4m3(
        numkong_scalars(activations.data()), packed_weights_[product].data(), products_.data(), tokens, width, depth,
        depth * sizeof(e4m3_t), width * sizeof(f32_t), NUMKONG_NULL, 0, stream.numkong());
    if (status != nk_success_k)
        return expected_t::fail(status_t::device_error_k, "E4M3 product {} failed: {}", product,
                                nk_status_name(status));
    f32r1_view_t const weight_descale {weight_descales_.data() + product, 1};
    dim3 const grid {static_cast<u32_t>(divide_round_up(tokens * width, kernel_block_size_default_k)), 1u, 1u};
    return launch_kernel(&transformer_token_rescale_kernel_, grid, dim3 {kernel_block_size_default_k, 1u, 1u}, stream,
                         products_.view(), token_scales, weight_descale, residual, values);
}

expected<transformer_block_e4m3_t> transformer_block_bf16_t::quantize_weights(gpu_arena_t &arena,
                                                                              gpu_stream_t &stream) noexcept {
    using result_t = expected<transformer_block_e4m3_t>;
    transformer_block_e4m3_t block;
    block.config_ = config_;
    block.numerics_ = numerics_;

    std::size_t const hidden = config_.hidden;
    std::size_t const ffn = static_cast<std::size_t>(config_.ffn_expansion) * hidden;
    std::size_t const tokens = config_.max_tokens;
    std::size_t const heads = hidden / transformer_head_dim_k;
    gpu_arena_alloc<e4m3_t> e4m3_alloc {arena};
    gpu_arena_alloc<bf16_t> bf16_alloc {arena};
    gpu_arena_alloc<f32_t> f32_alloc {arena};
    gpu_arena_alloc<u32_t> u32_alloc {arena};
    gpu_arena_alloc<b8_t> b8_alloc {arena};
    // Plain `uninitialized` allocations, checked in named groups so a failure names
    // the arena pressure.

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
    if (nk_status_t const status = numerics_.attention_pack_size_e4m3(heads, transformer_head_dim_k, tokens,
                                                                      config_.max_segments, &packed_keys_values_bytes);
        status != nk_success_k)
        return result_t::fail(status_t::device_error_k, "E4M3 attention pack sizing failed: {}",
                              nk_status_name(status));
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
    nk_status_t widened = numerics_.cast(pre_norm_scale_.data(), nk_bf16_k, block.pre_norm_scale_.data(), nk_f32_k,
                                         hidden, stream.numkong());
    if (widened == nk_success_k)
        widened = numerics_.cast(post_norm_scale_.data(), nk_bf16_k, block.post_norm_scale_.data(), nk_f32_k, hidden,
                                 stream.numkong());
    if (widened != nk_success_k)
        return result_t::fail(status_t::device_error_k, "norm scale widening failed: {}", nk_status_name(widened));

    // Per-weight static quantisation: order-free max-abs, descale = max_abs/240, divide-and-cast,
    // then the one-time pack into NumKong's layout. Every weight is a rank-2 [out,in] matrix; the
    // quantisation itself is purely elementwise, so each job flattens to the rank-1 handle the
    // house kernels take.
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
        if (expected_t zeroed = zero_async(max_bits.span(), stream); !zeroed) return result_t::fail(std::move(zeroed));
        bf16r1_view_t const source_flat = job.source->view().flatten<1>();
        expected_t launched = launch_kernel(&transformer_absolute_max_kernel_, grid, block_shape, stream, source_flat,
                                            max_bits.span());
        f32r1_span_t const descale {block.weight_descales_.data() + job.product, 1};
        if (launched)
            launched = launch_kernel(&transformer_descale_from_max_kernel_, dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u},
                                     stream, max_bits.span(), descale);
        if (launched) {
            f32r1_view_t const descale_view = descale;
            e4m3r1_span_t const shadow_flat = job.shadow->span().flatten<1>();
            launched = launch_kernel(&transformer_quantize_e4m3_kernel_, grid, block_shape, stream, source_flat,
                                     shadow_flat, descale_view);
        }
        if (!launched) return result_t::fail(std::move(launched));

        std::size_t const columns = job.shadow->extent(0), depth = job.shadow->extent(1);
        nk_size_t packed_bytes = 0;
        if (nk_status_t const status = numerics_.pack_size_e4m3(columns, depth, &packed_bytes); status != nk_success_k)
            return result_t::fail(status_t::device_error_k, "E4M3 pack sizing failed: {}", nk_status_name(status));
        b8r1_tensor_arena_t &packed = block.packed_weights_[job.product];
        packed = b8r1_tensor_arena_t::uninitialized({static_cast<std::size_t>(packed_bytes)}, b8_alloc).value;
        if (!packed)
            return result_t::fail(status_t::bad_alloc_k, "packed weight allocation failed ({} bytes)", packed_bytes);
        if (nk_status_t const status = numerics_.pack_e4m3(numkong_scalars(job.shadow->data()), columns, depth,
                                                           depth * sizeof(e4m3_t), packed.data(), 0, columns,
                                                           stream.numkong());
            status != nk_success_k)
            return result_t::fail(status_t::device_error_k, "E4M3 weight packing failed: {}", nk_status_name(status));
    }
    return result_t {std::move(block)};
}
