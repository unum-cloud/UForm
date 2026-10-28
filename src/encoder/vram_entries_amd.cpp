/**
 *  @file src/encoder/vram_entries_amd.cpp
 *  @author Ash Vardanian
 *  @date October 26, 2026
 *  @brief The AMD entries kit @ref vram_amd_entries_kit_t : copies and fills through the HIP
 *      runtime, and launches by name of the kernels @c vram_kernels_amd.hip compiles device-only.
 *
 *  The twin of @c vram_entries_nvidia.cpp . Compiled host-only, so no HIP host pass registers a fat
 *  binary with the runtime, and every call goes through the runtime table this unit loads at its
 *  first entry, so nothing links @c libamdhip64 . Each device loads the bundle once, the first time
 *  a stream on it launches, and keeps it for the life of the process. Every entry runs on its
 *  stream's device and leaves the caller's current device as it was.
 */
#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the module's load stage
#include <forkunion/types.hpp>   // `fu::standard_yield_t`, `fu::wait_capped_k`

#include "uform/encoder/vram_kits.hpp"      // `vram_amd_entries_kit_t`, `segment_layout_t`, `entries_block_threads_k`
#include "uform/utilities/vram_drivers.hpp" // `hip_api_t`

namespace fu = ashvardanian::forkunion;
namespace ue = unum::uform::encoder;
namespace uu = unum::uform::utilities;
using namespace ue;
using uu::b8_t;
using uu::bytes_span_t;
using uu::bytes_view_t;
using uu::divide_round_up;
using uu::hip_api_t;
using uu::hip_function_handle_t;
using uu::hip_module_handle_t;
using uu::hip_result_t;
using uu::hip_stream_handle_t;
using uu::max_gpus_per_host_k;
using uu::u8_t;

/** The bundle's bytes, linked in by the assembler stub @c uform_gpu_bundle writes. */
extern "C" b8_t const uform_encoder_amd_bundle_start[], uform_encoder_amd_bundle_end[];

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
enum class load_stage_t : u8_t {
    unloaded_k,
    loading_k,
    loaded_k,
};

/** The bundle as one device loaded it: every kernel's function. The stage is touched through the
 *  standard reference, since the entries name no CPU class. */
struct device_module_t {
    load_stage_t stage = load_stage_t::unloaded_k;
    hip_result_t status = uu::hip_error_not_initialized_k;
    hip_function_handle_t functions[std::size_t(kernel_t::count_k)] = {};
};

static device_module_t device_modules_[max_gpus_per_host_k];

/** The runtime every entry calls through, opened at the first entry; empty without one. */
[[nodiscard]] static hip_api_t const &runtime_() noexcept {
    static hip_api_t const runtime = hip_api_t::load();
    return runtime;
}

[[nodiscard]] static hip_stream_handle_t native_(nk_stream_t stream) noexcept {
    return static_cast<hip_stream_handle_t>(static_cast<void *>(stream));
}

/** Run @p enqueue on the device of @p stream , the current one for a null stream, and report its
 *  result. */
template <typename enqueue_type_>
[[nodiscard]] static nk_status_t on_device_(nk_stream_t stream, enqueue_type_ const &enqueue) noexcept {
    hip_api_t const &runtime = runtime_();
    int caller = 0, device = 0;
    if (!runtime || runtime.get_device(&caller) != uu::hip_success_k) return nk_device_code_mismatch_k;
    device = caller;
    if (stream && runtime.stream_device(native_(stream), &device) != uu::hip_success_k)
        return nk_device_code_mismatch_k;
    if (device != caller && runtime.set_device(device) != uu::hip_success_k) return nk_device_code_mismatch_k;
    hip_result_t const result = enqueue(runtime, native_(stream), device);
    if (device != caller) (void)runtime.set_device(caller);
    return result == uu::hip_success_k ? nk_success_k : nk_device_code_mismatch_k;
}

/** The function of @p kernel on @p device , the current one, loading the bundle there on first
 *  use. */
[[nodiscard]] static hip_result_t function_of_(hip_api_t const &runtime, kernel_t kernel, int device,
                                               hip_function_handle_t &function) noexcept {
    if (device < 0 || std::size_t(device) >= max_gpus_per_host_k) return uu::hip_error_invalid_device_k;
    device_module_t &module = device_modules_[device];
    fu::standard_atomic_ref<load_stage_t> const stage_ref {module.stage};
    load_stage_t stage = load_stage_t::unloaded_k;
    if (stage_ref.compare_exchange_strong(stage, load_stage_t::loading_k, std::memory_order_acquire)) {
        if (uform_encoder_amd_bundle_end - uform_encoder_amd_bundle_start == 0)
            module.status = uu::hip_error_no_binary_for_gpu_k;
        else {
            hip_module_handle_t loaded = nullptr;
            module.status = runtime.module_load_data(&loaded, uform_encoder_amd_bundle_start);
            for (std::size_t index = 0; module.status == uu::hip_success_k && index != std::size_t(kernel_t::count_k);
                 ++index)
                module.status = runtime.module_get_function(&module.functions[index], loaded, kernel_names_k[index]);
        }
        stage_ref.store(load_stage_t::loaded_k, std::memory_order_release);
    }
    while (stage_ref.load(std::memory_order_acquire) != load_stage_t::loaded_k)
        fu::standard_yield_t {}(&module.stage, load_stage_t::loading_k, std::size_t {0}, fu::wait_capped_k);
    if (module.status != uu::hip_success_k) return module.status;
    function = module.functions[std::size_t(kernel)];
    return uu::hip_success_k;
}

/** Launch @p kernel over @p blocks blocks of @p threads on @p stream ; each argument is passed as
 *  the exact type of the kernel's parameter, as the runtime copies its bytes. */
template <typename... argument_types_>
[[nodiscard]] static nk_status_t launch_(kernel_t kernel, unsigned blocks, unsigned threads, nk_stream_t stream,
                                         argument_types_... arguments) noexcept {
    void *addresses[] = {static_cast<void *>(&arguments)...};
    return on_device_(stream, [&](hip_api_t const &runtime, hip_stream_handle_t native, int device) {
        hip_function_handle_t function = nullptr;
        if (hip_result_t const result = function_of_(runtime, kernel, device, function); result != uu::hip_success_k)
            return result;
        return runtime.module_launch_kernel(function, blocks, 1, 1, threads, 1, 1, 0, native, addresses, nullptr);
    });
}

/** Blocks for a grid-stride walk over @p total items. */
[[nodiscard]] static unsigned stride_blocks_(std::size_t total) noexcept {
    std::size_t const blocks = divide_round_up(total, entries_block_threads_k);
    return static_cast<unsigned>(blocks == 0 ? 1 : blocks < grid_blocks_max_k ? blocks : grid_blocks_max_k);
}

nk_status_t ue::vram_amd_entries_kit_t::describe(std::size_t vendor_ordinal,
                                                 device_description_t &description) noexcept {
    hip_api_t const &runtime = runtime_();
    int const device = static_cast<int>(vendor_ordinal);
    int compute_units = 0, architecture_major = 0, architecture_minor = 0;
    std::size_t memory_bytes = 0;
    if (!runtime || runtime.get_device_name(description.name, sizeof(description.name), device) != uu::hip_success_k ||
        runtime.get_device_attribute(&compute_units, uu::hip_attribute_multiprocessor_count_k, device) !=
            uu::hip_success_k ||
        runtime.get_device_attribute(&architecture_major, uu::hip_attribute_compute_capability_major_k, device) !=
            uu::hip_success_k ||
        runtime.get_device_attribute(&architecture_minor, uu::hip_attribute_compute_capability_minor_k, device) !=
            uu::hip_success_k ||
        runtime.get_device_memory(&memory_bytes, device) != uu::hip_success_k)
        return nk_device_code_mismatch_k;
    description.compute_units = static_cast<u32_t>(compute_units);
    description.memory_bytes = memory_bytes;
    description.architecture_major = static_cast<u32_t>(architecture_major);
    description.architecture_minor = static_cast<u32_t>(architecture_minor);
    return nk_success_k;
}

nk_status_t ue::vram_amd_entries_kit_t::copy(bytes_span_t target, bytes_view_t source, nk_stream_t stream) noexcept {
    return on_device_(stream, [&](hip_api_t const &runtime, hip_stream_handle_t native, int) {
        return runtime.copy_async(target.data(), source.data(), source.size(), uu::hip_copy_default_k, native);
    });
}

nk_status_t ue::vram_amd_entries_kit_t::zero(bytes_span_t target, nk_stream_t stream) noexcept {
    return on_device_(stream, [&](hip_api_t const &runtime, hip_stream_handle_t native, int) {
        return runtime.fill_async(target.data(), 0, target.size(), native);
    });
}

nk_status_t ue::vram_amd_entries_kit_t::fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                                    u32r1_span_t pool_offsets, nk_stream_t stream) noexcept {
    return launch_(kernel_t::fill_layout_k, 1, 1, stream, lengths, packed_offsets, pool_offsets);
}

nk_status_t ue::vram_amd_entries_kit_t::fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                                    nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                                    u32_t frame_stride, u32_t positions, i16r2_span_t coords,
                                                    nk_stream_t stream) noexcept {
    return launch_(kernel_t::fill_coords_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, image_shapes, video_shapes, tube_frames, frame_stride, positions, coords);
}

nk_status_t ue::vram_amd_entries_kit_t::gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens,
                                                     bf16r2_view_t table, bf16r2_span_t rows,
                                                     nk_stream_t stream) noexcept {
    return launch_(kernel_t::gather_bytes_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, tokens, table, rows);
}

nk_status_t ue::vram_amd_entries_kit_t::stage_patches(segment_layout_t layout, modality_t target,
                                                      nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                                      f32r1_view_t inverse_std, bf16r2_span_t patches,
                                                      nk_stream_t stream) noexcept {
    nk::tensor_span<nk_bf16_t, 3> const cells {patches.data(), {patches.extent(0), pixels.extent(1), pixels.extent(2)}};
    return launch_(kernel_t::stage_patches_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, target, pixels, mean, inverse_std, cells);
}

nk_status_t ue::vram_amd_entries_kit_t::stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples,
                                                     bf16r2_span_t frames, nk_stream_t stream) noexcept {
    return launch_(kernel_t::stage_frames_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, samples, frames);
}

nk_status_t ue::vram_amd_entries_kit_t::mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                              i16r2_view_t coords, rope_grids_t angles, nk_stream_t stream) noexcept {
    return launch_(kernel_t::mrope_k, stride_blocks_(values.extent(0) * (head_width / 2u)), entries_block_threads_k,
                   stream, values, angles, coords, heads, head_width, role_offset);
}

nk_status_t ue::vram_amd_entries_kit_t::quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow,
                                                             f32r1_span_t token_scales, nk_stream_t stream) noexcept {
    return launch_(kernel_t::quantize_bf16_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_amd_entries_kit_t::quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow,
                                                             f32r1_span_t token_scales, nk_stream_t stream) noexcept {
    return launch_(kernel_t::quantize_f32_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_amd_entries_kit_t::rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales,
                                                       f32_t weight_descale, bf16r2_view_t residual,
                                                       bf16r2_span_t values, nk_stream_t stream) noexcept {
    return launch_(kernel_t::rescale_tokens_k, stride_blocks_(values.extent(0) * values.extent(1)),
                   entries_block_threads_k, stream, products, token_scales, weight_descale, residual, values);
}
