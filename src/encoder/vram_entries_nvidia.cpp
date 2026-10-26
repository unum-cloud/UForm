/**
 *  @file src/encoder/vram_entries_nvidia.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The NVIDIA entries kit @ref vram_nvidia_entries_kit_t : copies and fills through the
 *      Driver API, and launches by name of the kernels @c vram_kernels_nvidia.cu compiles
 *      device-only.
 *
 *  Compiled host-only, so no NVCC host pass registers a fatbin with the CUDA runtime. Each device
 *  loads the bundle once, into the context of the first stream that launches on it, and keeps it
 *  for the life of the process, as the runtime keeps a registered fatbin; a later stream on that
 *  device must share the context. Every entry runs inside its stream's context and leaves the
 *  caller's current context as it was.
 */
#include <bit> // `std::bit_cast` — device addresses

#include <cuda.h> // `cuModuleLoadData`, `cuLaunchKernel`, `cuMemcpyAsync`, `cuStreamGetCtx`

#include "uform/encoder/vram_kits.hpp" // `vram_nvidia_entries_kit_t`, `segment_layout_t`, `entries_block_threads_k`

namespace ue = unum::uform::encoder;
namespace uc = unum::uform::core;
using namespace ue;

/** The bundle's bytes, linked in by the assembler stub `uform_gpu_bundle` writes. */
extern "C" uc::b8_t const uform_encoder_nvidia_bundle_start[], uform_encoder_nvidia_bundle_end[];

/** The bundle's kernels, in the order of @ref kernel_names_k . */
enum class kernel_t : std::size_t {
    fill_layout_k,
    fill_coords_k,
    gather_bytes_k,
    stage_patches_k,
    stage_frames_k,
    mrope_k,
    quantize_bf16_tokens_k,
    quantize_f32_tokens_k,
    rescale_tokens_k,
    count_k,
};

static char const *const kernel_names_k[] = {
    "uform_fill_layout_kernel",          "uform_fill_coords_kernel",         "uform_gather_bytes_kernel",
    "uform_stage_patches_kernel",        "uform_stage_frames_kernel",        "uform_mrope_kernel",
    "uform_quantize_bf16_tokens_kernel", "uform_quantize_f32_tokens_kernel", "uform_rescale_tokens_kernel",
};

/** Most blocks a grid-stride kernel launches. */
static constexpr std::size_t grid_blocks_max_k = 1024;

/** How far a device got loading the bundle; whichever thread leaves @c unloaded_k loads it. */
enum class load_stage_t : uc::u8_t {
    unloaded_k,
    loading_k,
    loaded_k,
};

/** The bundle as one device loaded it: the context it lives in and every kernel's function. The
 *  stage is touched through the standard reference, since the entries name no CPU class. */
struct device_module_t {
    load_stage_t stage = load_stage_t::unloaded_k;
    CUresult status = CUDA_ERROR_NOT_INITIALIZED;
    CUcontext context = nullptr;
    CUfunction functions[std::size_t(kernel_t::count_k)] = {};
};

static device_module_t device_modules_[uc::max_gpus_per_host_k];

[[nodiscard]] static CUstream native_(nk_stream_t stream) noexcept {
    return static_cast<CUstream>(static_cast<void *>(stream));
}

/** Run @p enqueue in the context of @p stream , the current one for a null stream, and report its
 *  result. */
template <typename enqueue_type_>
[[nodiscard]] static nk_status_t in_context_(nk_stream_t stream, enqueue_type_ const &enqueue) noexcept {
    CUcontext context = nullptr;
    if (stream &&
        (cuStreamGetCtx(native_(stream), &context) != CUDA_SUCCESS || cuCtxPushCurrent(context) != CUDA_SUCCESS))
        return nk_device_code_mismatch_k;
    CUresult const result = enqueue(native_(stream));
    if (stream) cuCtxPopCurrent(&context);
    return result == CUDA_SUCCESS ? nk_success_k : nk_device_code_mismatch_k;
}

/** The function of @p kernel in the current context's device, loading the bundle there on first
 *  use. */
[[nodiscard]] static CUresult function_of_(kernel_t kernel, CUfunction &function) noexcept {
    CUcontext context = nullptr;
    CUdevice device = 0;
    if (CUresult const result = cuCtxGetCurrent(&context); result != CUDA_SUCCESS) return result;
    if (CUresult const result = cuCtxGetDevice(&device); result != CUDA_SUCCESS) return result;
    if (device < 0 || std::size_t(device) >= uc::max_gpus_per_host_k) return CUDA_ERROR_INVALID_DEVICE;
    device_module_t &module = device_modules_[device];
    fu::standard_atomic_ref<load_stage_t> const stage_ref {module.stage};
    load_stage_t stage = load_stage_t::unloaded_k;
    if (stage_ref.compare_exchange_strong(stage, load_stage_t::loading_k, std::memory_order_acquire)) {
        module.context = context;
        if (uform_encoder_nvidia_bundle_end - uform_encoder_nvidia_bundle_start == 0)
            module.status = CUDA_ERROR_NO_BINARY_FOR_GPU;
        else {
            CUmodule loaded = nullptr;
            module.status = cuModuleLoadData(&loaded, uform_encoder_nvidia_bundle_start);
            for (std::size_t index = 0; module.status == CUDA_SUCCESS && index != std::size_t(kernel_t::count_k);
                 ++index)
                module.status = cuModuleGetFunction(&module.functions[index], loaded, kernel_names_k[index]);
        }
        stage_ref.store(load_stage_t::loaded_k, std::memory_order_release);
    }
    while (stage_ref.load(std::memory_order_acquire) != load_stage_t::loaded_k)
        fu::standard_yield_t {}(&module.stage, load_stage_t::loading_k, std::size_t {0}, fu::wait_capped_k);
    if (module.status != CUDA_SUCCESS) return module.status;
    if (module.context != context) return CUDA_ERROR_INVALID_CONTEXT;
    function = module.functions[std::size_t(kernel)];
    return CUDA_SUCCESS;
}

/** Launch @p kernel over @p blocks blocks of @p threads on @p stream ; each argument is passed as
 *  the exact type of the kernel's parameter, as the driver copies its bytes. */
template <typename... argument_types_>
[[nodiscard]] static nk_status_t launch_(kernel_t kernel, unsigned blocks, unsigned threads, nk_stream_t stream,
                                         argument_types_... arguments) noexcept {
    void *addresses[] = {static_cast<void *>(&arguments)...};
    return in_context_(stream, [&](CUstream native) {
        CUfunction function = nullptr;
        if (CUresult const result = function_of_(kernel, function); result != CUDA_SUCCESS) return result;
        return cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, native, addresses, nullptr);
    });
}

/** Blocks for a grid-stride walk over @p total items. */
[[nodiscard]] static unsigned stride_blocks_(std::size_t total) noexcept {
    std::size_t const blocks = uc::divide_round_up(total, entries_block_threads_k);
    return static_cast<unsigned>(blocks == 0 ? 1 : blocks < grid_blocks_max_k ? blocks : grid_blocks_max_k);
}

nk_status_t ue::vram_nvidia_entries_kit_t::describe(std::size_t vendor_ordinal,
                                                    device_description_t &description) noexcept {
    CUdevice device = 0;
    int compute_units = 0;
    std::size_t memory_bytes = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, static_cast<int>(vendor_ordinal)) != CUDA_SUCCESS ||
        cuDeviceGetName(description.name, sizeof(description.name), device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&compute_units, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) != CUDA_SUCCESS ||
        cuDeviceTotalMem(&memory_bytes, device) != CUDA_SUCCESS)
        return nk_device_code_mismatch_k;
    description.compute_units = static_cast<u32_t>(compute_units);
    description.memory_bytes = memory_bytes;
    return nk_success_k;
}

nk_status_t ue::vram_nvidia_entries_kit_t::copy(uc::bytes_span_t target, uc::bytes_view_t source,
                                                nk_stream_t stream) noexcept {
    return in_context_(stream, [&](CUstream native) {
        return cuMemcpyAsync(std::bit_cast<CUdeviceptr>(target.data()), std::bit_cast<CUdeviceptr>(source.data()),
                             source.size(), native);
    });
}

nk_status_t ue::vram_nvidia_entries_kit_t::zero(uc::bytes_span_t target, nk_stream_t stream) noexcept {
    return in_context_(stream, [&](CUstream native) {
        return cuMemsetD8Async(std::bit_cast<CUdeviceptr>(target.data()), 0, target.size(), native);
    });
}

nk_status_t ue::vram_nvidia_entries_kit_t::fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                                       u32r1_span_t pool_offsets, nk_stream_t stream) noexcept {
    return launch_(kernel_t::fill_layout_k, 1, 1, stream, lengths, packed_offsets, pool_offsets);
}

nk_status_t ue::vram_nvidia_entries_kit_t::fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                                       nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                                       u32_t frame_stride, u32_t positions, i16r2_span_t coords,
                                                       nk_stream_t stream) noexcept {
    return launch_(kernel_t::fill_coords_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, image_shapes, video_shapes, tube_frames, frame_stride, positions, coords);
}

nk_status_t ue::vram_nvidia_entries_kit_t::gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens,
                                                        bf16r2_view_t table, bf16r2_span_t rows,
                                                        nk_stream_t stream) noexcept {
    return launch_(kernel_t::gather_bytes_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, tokens, table, rows);
}

nk_status_t ue::vram_nvidia_entries_kit_t::stage_patches(segment_layout_t layout, modality_t target,
                                                         nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                                         f32r1_view_t inverse_std, bf16r2_span_t patches,
                                                         nk_stream_t stream) noexcept {
    nk::tensor_span<nk_bf16_t, 3> const cells {patches.data(), {patches.extent(0), pixels.extent(1), pixels.extent(2)}};
    return launch_(kernel_t::stage_patches_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, target, pixels, mean, inverse_std, cells);
}

nk_status_t ue::vram_nvidia_entries_kit_t::stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples,
                                                        bf16r2_span_t frames, nk_stream_t stream) noexcept {
    return launch_(kernel_t::stage_frames_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, samples, frames);
}

nk_status_t ue::vram_nvidia_entries_kit_t::mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                                 i16r2_view_t coords, rope_grids_t angles,
                                                 nk_stream_t stream) noexcept {
    return launch_(kernel_t::mrope_k, stride_blocks_(values.extent(0) * (head_width / 2u)), entries_block_threads_k,
                   stream, values, angles, coords, heads, head_width, role_offset);
}

nk_status_t ue::vram_nvidia_entries_kit_t::quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow,
                                                                f32r1_span_t token_scales,
                                                                nk_stream_t stream) noexcept {
    return launch_(kernel_t::quantize_bf16_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_nvidia_entries_kit_t::quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow,
                                                                f32r1_span_t token_scales,
                                                                nk_stream_t stream) noexcept {
    return launch_(kernel_t::quantize_f32_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_nvidia_entries_kit_t::rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales,
                                                          f32_t weight_descale, bf16r2_view_t residual,
                                                          bf16r2_span_t values, nk_stream_t stream) noexcept {
    return launch_(kernel_t::rescale_tokens_k, stride_blocks_(values.extent(0) * values.extent(1)),
                   entries_block_threads_k, stream, products, token_scales, weight_descale, residual, values);
}
