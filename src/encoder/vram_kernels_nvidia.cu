/**
 *  @file src/encoder/vram_kernels_nvidia.cu
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The NVIDIA kernels of the tower steps NumKong has no kernel for, compiled device-only
 *      into the @c uform_encoder_nvidia bundle, which @c vram_entries_nvidia.cpp launches by name.
 *
 *  Every kernel runs a body of @ref tower_kernels.cuh , the one the training kernels run, with
 *  NumKong's SIMT conversions, which match its host casts bit for bit. Each is @c extern "C" , so
 *  its name in the bundle is the one the host unit looks up.
 */
#include <cmath> // `std::fmax`

#include <type_traits> // `std::is_same_v`

#include <numkong/types.h>       // `NUMKONG_ARCH_CUDA_`, which the SIMT casts test before including it
#include <numkong/cast/simt.cuh> // `nk_bf16_to_f32_simt_`, `nk_f32_to_bf16_simt_`, `nk_f32_to_e4m3_simt_`

#include "uform/core/tower_kernels.cuh" // the bodies every vendor's kernels share with training
#include "uform/encoder/entries.hpp"    // `segment_layout_t`, `entries_block_threads_k`

namespace ue = unum::uform::encoder;
using namespace ue;

/** The conversions the bodies take; templates, so a host pass never instantiates the device-only
 *  casts. */
struct vram_scalars_t {
    template <typename value_type_>
    __host__ __device__ f32_t widen(value_type_ value) const noexcept {
        if constexpr (std::is_same_v<value_type_, f32_t>) return value;
        else {
            f32_t wide = 0;
            nk_bf16_to_f32_simt_(&value, &wide);
            return wide;
        }
    }
    template <typename value_type_>
    __host__ __device__ nk_bf16_t narrow(value_type_ value) const noexcept {
        nk_bf16_t narrow {};
        nk_f32_to_bf16_simt_(&value, &narrow);
        return narrow;
    }
    template <typename value_type_>
    __host__ __device__ nk_e4m3_t narrow_e4m3(value_type_ value) const noexcept {
        nk_e4m3_t narrow {};
        nk_f32_to_e4m3_simt_(&value, &narrow);
        return narrow;
    }
    template <typename value_type_>
    __host__ __device__ void sincos(value_type_ angle, f32_t *sine, f32_t *cosine) const noexcept {
        __sincosf(angle, sine, cosine);
    }
};

[[nodiscard]] static __device__ std::size_t stride_first_() noexcept {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

[[nodiscard]] static __device__ std::size_t stride_step_() noexcept {
    return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

/** __grid_size__ @c [1,1,1] , one thread. */
extern "C" __global__ void uform_fill_layout_kernel(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                                    u32r1_span_t pool_offsets) {
    if (threadIdx.x == 0u) uc::fill_packed_layout(lengths, packed_offsets, pool_offsets);
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's cells. */
extern "C" __global__ void uform_fill_coords_kernel(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                                    nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                                    u32_t frame_stride, u32_t positions, i16r2_span_t coords) {
    std::size_t const segment = blockIdx.x;
    uc::fill_coords_segment(layout.modality[segment], segment, layout.packed_offsets[segment], layout.lengths[segment],
                            image_shapes, video_shapes, tube_frames, frame_stride, positions, coords, threadIdx.x,
                            blockDim.x);
}

/** __grid_size__ @c [segments,1,1] , threads stripe the hidden channels. */
extern "C" __global__ void uform_gather_bytes_kernel(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens,
                                                     bf16r2_view_t table, bf16r2_span_t rows) {
    std::size_t const segment = blockIdx.x;
    if (layout.modality[segment] != modality_t::text_k) return;
    uc::gather_bytes_segment(tokens, table, layout.grid_offsets[segment], layout.packed_offsets[segment],
                             layout.lengths[segment], rows, threadIdx.x, blockDim.x);
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's positions. */
extern "C" __global__ void uform_stage_patches_kernel(segment_layout_t layout, modality_t target,
                                                      nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                                      f32r1_view_t inverse_std, nk::tensor_span<nk_bf16_t, 3> patches) {
    std::size_t const segment = blockIdx.x;
    if (layout.modality[segment] != target) return;
    uc::stage_patches_segment(pixels, layout.grid_offsets[segment], layout.packed_offsets[segment],
                              layout.lengths[segment], mean, inverse_std, patches, threadIdx.x, blockDim.x,
                              vram_scalars_t {});
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's samples. */
extern "C" __global__ void uform_stage_frames_kernel(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples,
                                                     bf16r2_span_t frames) {
    std::size_t const segment = blockIdx.x;
    if (layout.modality[segment] != modality_t::audio_k) return;
    uc::stage_frames_segment(samples, layout.grid_offsets[segment], layout.packed_offsets[segment],
                             layout.lengths[segment], frames, threadIdx.x, blockDim.x, vram_scalars_t {});
}

/** __grid_size__ grid-stride over channel pairs. */
extern "C" __global__ void uform_mrope_kernel(bf16r2_span_t values, rope_grids_t angles, i16r2_view_t coords,
                                              u32_t heads, u32_t head_width, u32_t role_offset) {
    uc::rope_rotate(values, angles, coords, heads, head_width, role_offset, 1.0f, stride_first_(), stride_step_(),
                    vram_scalars_t {});
}

/** The body of both quantize kernels: one block of @ref entries_block_threads_k per token. */
template <typename value_type_>
static __device__ void quantize_tokens_(nk::tensor_view<value_type_, 2> values, e4m3r2_span_t shadow,
                                        f32r1_span_t token_scales) noexcept {
    __shared__ f32_t partials[entries_block_threads_k];
    std::size_t const token = blockIdx.x;
    vram_scalars_t const scalars;
    partials[threadIdx.x] = uc::token_largest(values, token, threadIdx.x, blockDim.x, scalars);
    __syncthreads();
    for (unsigned half = entries_block_threads_k / 2; half != 0; half /= 2) {
        if (threadIdx.x < half) partials[threadIdx.x] = std::fmax(partials[threadIdx.x], partials[threadIdx.x + half]);
        __syncthreads();
    }
    f32_t const scale = uc::quantize_token(values, shadow, token, partials[0], threadIdx.x, blockDim.x, scalars);
    if (threadIdx.x == 0u) token_scales[token] = scale;
}

/** __grid_size__ @c [tokens,1,1] with @ref entries_block_threads_k threads, one block per token. */
extern "C" __global__ void uform_quantize_bf16_tokens_kernel(bf16r2_view_t values, e4m3r2_span_t shadow,
                                                             f32r1_span_t token_scales) {
    quantize_tokens_(values, shadow, token_scales);
}

/** __grid_size__ @c [tokens,1,1] with @ref entries_block_threads_k threads, one block per token. */
extern "C" __global__ void uform_quantize_f32_tokens_kernel(f32r2_view_t values, e4m3r2_span_t shadow,
                                                            f32r1_span_t token_scales) {
    quantize_tokens_(values, shadow, token_scales);
}

/** __grid_size__ grid-stride over cells. */
extern "C" __global__ void uform_rescale_tokens_kernel(f32r2_view_t products, f32r1_view_t token_scales,
                                                       f32_t weight_descale, bf16r2_view_t residual,
                                                       bf16r2_span_t values) {
    uc::rescale_tokens_cells(products, token_scales, weight_descale, residual, values, stride_first_(), stride_step_(),
                             vram_scalars_t {});
}
