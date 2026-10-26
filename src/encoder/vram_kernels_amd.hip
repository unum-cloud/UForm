/**
 *  @brief The AMD kernels of the tower steps NumKong has no kernel for, published as @ref vram_amd_kernels_k .
 *  @file src/encoder/vram_kernels_amd.hip
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *
 *  The HIP twin of @c vram_kernels_nvidia.cu , kernel for kernel: every kernel runs a body of @ref tower_kernels.cuh
 *  with NumKong's SIMT conversions, which match its host casts bit for bit. Each launcher enters the device of its
 *  stream and leaves the caller's current device as it was.
 */
#include <bit>         // `std::bit_cast` — kernel symbols
#include <type_traits> // `std::is_same_v`, `std::type_identity_t`

#include <hip/hip_runtime.h> // `hipMemcpyAsync`, `hipMemsetAsync`, `hipStreamGetDevice`

#include <numkong/types.h>       // `NUMKONG_ARCH_ROCM_`, which the SIMT casts test before including it
#include <numkong/cast/simt.cuh> // `nk_bf16_to_f32_simt_`, `nk_f32_to_bf16_simt_`, `nk_f32_to_e4m3_simt_`

#include "uform/core/tower_kernels.cuh" // the bodies every vendor's kernels share with training
#include "uform/encoder/entries.hpp"    // `device_kernels_t`, `segment_layout_t`

namespace ue = unum::uform::encoder;
using namespace ue;

/** Threads per block of every kernel here, a power of two the quantize reduction halves. */
static constexpr unsigned block_threads_k = 256;

/** Most blocks a grid-stride kernel launches. */
static constexpr std::size_t grid_blocks_max_k = 1024;

/** The conversions the bodies take; templates, so the host pass never instantiates the device-only casts. */
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

[[nodiscard]] static hipStream_t native_(nk_stream_t stream) noexcept {
    return static_cast<hipStream_t>(static_cast<void *>(stream));
}

/** Run @p enqueue on the device of @p stream , the current one for a null stream, and report its launch. */
template <typename enqueue_type_>
[[nodiscard]] static nk_status_t on_device_(nk_stream_t stream, enqueue_type_ const &enqueue) noexcept {
    int caller = 0;
    hipDevice_t device = 0;
    if (hipGetDevice(&caller) != hipSuccess) return nk_device_code_mismatch_k;
    device = caller;
    if (stream && hipStreamGetDevice(native_(stream), &device) != hipSuccess) return nk_device_code_mismatch_k;
    if (device != caller && hipSetDevice(device) != hipSuccess) return nk_device_code_mismatch_k;
    hipError_t error = enqueue(native_(stream));
    if (error == hipSuccess) error = hipGetLastError();
    if (device != caller) (void)hipSetDevice(caller);
    return error == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Launch @p kernel over @p blocks blocks of @p threads on @p stream ; the arguments convert to the parameters'
 *  exact types first, as the runtime copies their bytes. */
template <typename... parameter_types_>
[[nodiscard]] static nk_status_t launch_(void (*kernel)(parameter_types_...), unsigned blocks, unsigned threads,
                                         nk_stream_t stream, std::type_identity_t<parameter_types_>... arguments) noexcept {
    void *addresses[] = {static_cast<void *>(&arguments)...};
    return on_device_(stream, [&](hipStream_t native) {
        return hipLaunchKernel(std::bit_cast<void const *>(kernel), dim3(blocks), dim3(threads), addresses, 0, native);
    });
}

/** Blocks for a grid-stride walk over @p total items. */
[[nodiscard]] static unsigned stride_blocks_(std::size_t total) noexcept {
    std::size_t const blocks = (total + block_threads_k - 1) / block_threads_k;
    return static_cast<unsigned>(blocks == 0 ? 1 : blocks < grid_blocks_max_k ? blocks : grid_blocks_max_k);
}

[[nodiscard]] static __device__ std::size_t stride_first_() noexcept {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

[[nodiscard]] static __device__ std::size_t stride_step_() noexcept {
    return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

/** __grid_size__ @c [1,1,1] , one thread. */
static __global__ void fill_layout_kernel_(u32r1_view_t lengths, u32_t *packed_offsets, u32_t *pool_offsets) {
    if (threadIdx.x == 0u) uc::fill_packed_layout(lengths, packed_offsets, pool_offsets);
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's cells. */
static __global__ void fill_coords_kernel_(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                           nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                           u32_t frame_stride, u32_t positions, i16r2_span_t coords) {
    std::size_t const segment = blockIdx.x;
    uc::fill_coords_segment(layout.modality[segment], segment, layout.packed_offsets[segment], layout.lengths[segment],
                            image_shapes, video_shapes, tube_frames, frame_stride, positions, coords, threadIdx.x,
                            blockDim.x);
}

/** __grid_size__ @c [segments,1,1] , threads stripe the hidden channels. */
static __global__ void gather_bytes_kernel_(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens,
                                            bf16r2_view_t table, bf16r2_span_t rows) {
    std::size_t const segment = blockIdx.x;
    if (layout.modality[segment] != modality_t::text_k) return;
    uc::gather_bytes_segment(tokens, table, layout.grid_offsets[segment], layout.packed_offsets[segment],
                             layout.lengths[segment], rows, threadIdx.x, blockDim.x);
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's positions. */
static __global__ void stage_patches_kernel_(segment_layout_t layout, modality_t target,
                                             nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                             f32r1_view_t inverse_std, nk::tensor_span<nk_bf16_t, 3> patches) {
    std::size_t const segment = blockIdx.x;
    if (layout.modality[segment] != target) return;
    uc::stage_patches_segment(pixels, layout.grid_offsets[segment], layout.packed_offsets[segment],
                              layout.lengths[segment], mean, inverse_std, patches, threadIdx.x, blockDim.x,
                              vram_scalars_t {});
}

/** __grid_size__ @c [segments,1,1] , threads stripe each segment's samples. */
static __global__ void stage_frames_kernel_(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples,
                                            bf16r2_span_t frames) {
    std::size_t const segment = blockIdx.x;
    if (layout.modality[segment] != modality_t::audio_k) return;
    uc::stage_frames_segment(samples, layout.grid_offsets[segment], layout.packed_offsets[segment],
                             layout.lengths[segment], frames, threadIdx.x, blockDim.x, vram_scalars_t {});
}

/** __grid_size__ grid-stride over channel pairs. */
static __global__ void mrope_kernel_(bf16r2_span_t values, rope_grids_t angles, i16r2_view_t coords, u32_t heads,
                                     u32_t head_width, u32_t role_offset) {
    uc::rope_rotate(values, angles, coords, heads, head_width, role_offset, 1.0f, stride_first_(), stride_step_(),
                    vram_scalars_t {});
}

/** __grid_size__ @c [tokens,1,1] with @ref block_threads_k threads, one block per token. */
template <typename value_type_>
static __global__ void quantize_tokens_kernel_(nk::tensor_view<value_type_, 2> values, e4m3r2_span_t shadow,
                                               f32r1_span_t token_scales) {
    __shared__ f32_t partials[block_threads_k];
    std::size_t const token = blockIdx.x;
    vram_scalars_t const scalars;
    partials[threadIdx.x] = uc::token_largest(values, token, threadIdx.x, blockDim.x, scalars);
    __syncthreads();
    for (unsigned half = block_threads_k / 2; half != 0; half /= 2) {
        if (threadIdx.x < half) partials[threadIdx.x] = fmaxf(partials[threadIdx.x], partials[threadIdx.x + half]);
        __syncthreads();
    }
    f32_t const scale = uc::quantize_token(values, shadow, token, partials[0], threadIdx.x, blockDim.x, scalars);
    if (threadIdx.x == 0u) token_scales[token] = scale;
}

/** __grid_size__ grid-stride over cells. */
static __global__ void rescale_tokens_kernel_(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                              bf16r2_view_t residual, bf16r2_span_t values) {
    uc::rescale_tokens_cells(products, token_scales, weight_descale, residual, values, stride_first_(),
                             stride_step_(), vram_scalars_t {});
}

static nk_status_t describe_(std::size_t ordinal, device_description_t &description) noexcept {
    int const device = static_cast<int>(ordinal);
    int compute_units = 0;
    std::size_t memory_bytes = 0;
    if (hipDeviceGetName(description.name, sizeof(description.name), device) != hipSuccess ||
        hipDeviceGetAttribute(&compute_units, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
        hipDeviceTotalMem(&memory_bytes, device) != hipSuccess)
        return nk_device_code_mismatch_k;
    description.compute_units = static_cast<u32_t>(compute_units);
    description.memory_bytes = memory_bytes;
    return nk_success_k;
}

static nk_status_t copy_(void *target, void const *source, std::size_t bytes, nk_stream_t stream) noexcept {
    return on_device_(stream, [&](hipStream_t native) {
        return hipMemcpyAsync(target, source, bytes, hipMemcpyDefault, native);
    });
}

static nk_status_t zero_(void *target, std::size_t bytes, nk_stream_t stream) noexcept {
    return on_device_(stream, [&](hipStream_t native) { return hipMemsetAsync(target, 0, bytes, native); });
}

static nk_status_t fill_layout_(u32r1_view_t lengths, u32_t *packed_offsets, u32_t *pool_offsets,
                                nk_stream_t stream) noexcept {
    return launch_(&fill_layout_kernel_, 1, 1, stream, lengths, packed_offsets, pool_offsets);
}

static nk_status_t fill_coords_(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                u32_t positions, i16r2_span_t coords, nk_stream_t stream) noexcept {
    return launch_(&fill_coords_kernel_, static_cast<unsigned>(layout.lengths.extent(0)), block_threads_k, stream,
                   layout, image_shapes, video_shapes, tube_frames, frame_stride, positions, coords);
}

static nk_status_t gather_bytes_(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens, bf16r2_view_t table,
                                 bf16r2_span_t rows, nk_stream_t stream) noexcept {
    return launch_(&gather_bytes_kernel_, static_cast<unsigned>(layout.lengths.extent(0)), block_threads_k, stream,
                   layout, tokens, table, rows);
}

static nk_status_t stage_patches_(segment_layout_t layout, modality_t target, nk::tensor_view<pixel_t, 3> pixels,
                                  f32r1_view_t mean, f32r1_view_t inverse_std, bf16r2_span_t patches,
                                  nk_stream_t stream) noexcept {
    nk::tensor_span<nk_bf16_t, 3> const cells {patches.data(),
                                               {patches.extent(0), pixels.extent(1), pixels.extent(2)}};
    return launch_(&stage_patches_kernel_, static_cast<unsigned>(layout.lengths.extent(0)), block_threads_k, stream,
                   layout, target, pixels, mean, inverse_std, cells);
}

static nk_status_t stage_frames_(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples, bf16r2_span_t frames,
                                 nk_stream_t stream) noexcept {
    return launch_(&stage_frames_kernel_, static_cast<unsigned>(layout.lengths.extent(0)), block_threads_k, stream,
                   layout, samples, frames);
}

static nk_status_t mrope_(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width, i16r2_view_t coords,
                          rope_grids_t angles, nk_stream_t stream) noexcept {
    return launch_(&mrope_kernel_, stride_blocks_(values.extent(0) * heads * (head_width / 2u)), block_threads_k,
                   stream, values, angles, coords, heads, head_width, role_offset);
}

template <typename value_type_>
static nk_status_t quantize_tokens_(nk::tensor_view<value_type_, 2> values, e4m3r2_span_t shadow,
                                    f32r1_span_t token_scales, nk_stream_t stream) noexcept {
    return launch_(&quantize_tokens_kernel_<value_type_>, static_cast<unsigned>(values.extent(0)), block_threads_k,
                   stream, values, shadow, token_scales);
}

static nk_status_t rescale_tokens_(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                   bf16r2_view_t residual, bf16r2_span_t values, nk_stream_t stream) noexcept {
    return launch_(&rescale_tokens_kernel_, stride_blocks_(values.extent(0) * values.extent(1)), block_threads_k,
                   stream, products, token_scales, weight_descale, residual, values);
}

/** Returned from a call, so a Clang device pass never emits this table of host addresses as a device constant. */
[[nodiscard]] static device_kernels_t tabulate_() noexcept {
    return {
        .vendor = nk_cap_rocm_k,
        .describe = &describe_,
        .copy = &copy_,
        .zero = &zero_,
        .fill_layout = &fill_layout_,
        .fill_coords = &fill_coords_,
        .gather_bytes = &gather_bytes_,
        .stage_patches = &stage_patches_,
        .stage_frames = &stage_frames_,
        .mrope = &mrope_,
        .quantize_bf16_tokens_e4m3 = &quantize_tokens_<nk_bf16_t>,
        .quantize_f32_tokens_e4m3 = &quantize_tokens_<f32_t>,
        .rescale_tokens = &rescale_tokens_,
    };
}

device_kernels_t const ue::vram_amd_kernels_k = tabulate_();
