/**
 *  @file src/encoder/vram_entries_nvidia.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The NVIDIA entries kit @ref vram_nvidia_entries_kit_t : copies and fills through the CUDA
 *      driver, and launches by name of the kernels @c vram_kernels_nvidia.cu compiles device-only.
 *
 *  Compiled host-only, so no NVCC host pass registers a fatbin with the CUDA runtime, and every
 *  call goes through the driver table this unit loads at its first entry, so nothing links
 *  @c libcuda . Each device loads the bundle once, into its primary context at the first launch on
 *  it, and keeps it for the life of the process, as the runtime keeps a registered fatbin. Every
 *  entry runs in the current context, the primary context of the device the executor selected,
 *  where the executor also opened or adopted its stream for that device, and every allocation the
 *  executor's policies make or free lands on that device too.
 */
#include <bit> // `std::bit_cast` — device addresses

#include <numkong/capabilities.h> // `nk_capabilities_detected_cuda`, `nk_capabilities_compiled_cuda`

#include "uform/encoder/vram_kits.hpp" // `vram_nvidia_entries_kit_t`, `segment_layout_t`, `entries_block_threads_k`

#include "lib.hpp" // `gpu_runtime` - the driver and the bundle's tables

namespace fu = ashvardanian::forkunion;
namespace ue = unum::uform::encoder;
namespace util = unum::uform::utilities;
using namespace ue;
using util::b8_t;
using util::bytes_span_t;
using util::bytes_view_t;
using util::cuda_api_t;
using util::cuda_context_handle_t;
using util::cuda_device_address_t;
using util::cuda_function_handle_t;
using util::cuda_result_t;
using util::cuda_stream_handle_t;
using util::divide_round_up;
using util::u8_t;

/** The bundle's bytes, linked in by the assembler stub @c uform_gpu_bundle writes. */
extern "C" b8_t const uform_encoder_nvidia_bundle_start[], uform_encoder_nvidia_bundle_end[];

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

static char const* const kernel_names_k[] = {
    "uform_fill_layout_kernel",          "uform_fill_coords_kernel",         "uform_gather_bytes_kernel",
    "uform_stage_patches_kernel",        "uform_stage_frames_kernel",        "uform_mrope_kernel",
    "uform_quantize_bf16_tokens_kernel", "uform_quantize_f32_tokens_kernel", "uform_rescale_tokens_kernel",
};

/** The driver every entry calls through, the GPU runtime's; empty without one. */
[[nodiscard]] static cuda_api_t const& driver_() noexcept { return uf::gpu_drivers().cuda; }

using stream_t = vram_nvidia_entries_kit_t::stream_t;

/** Run @p enqueue on @p stream in the current context, the bound device's, and report its
 *  result. */
template <typename enqueue_type_>
[[nodiscard]] static nk_status_t in_context_(stream_t& stream, enqueue_type_ const& enqueue) noexcept {
    cuda_api_t const& driver = stream.handle.api();
    if (!driver) return nk_device_code_mismatch_k;
    return enqueue(driver, static_cast<cuda_stream_handle_t>(stream.handle)) == util::cuda_success_k
               ? nk_success_k
               : nk_device_code_mismatch_k;
}

/** The functions of the bundle on @p stream 's device, which its kernel tables load there on first
 *  use. */
[[nodiscard]] static cuda_result_t functions_of_(stream_t& stream,
                                                 std::span<cuda_function_handle_t const>& functions) noexcept {
    cuda_api_t const& driver = stream.handle.api();
    int const device = static_cast<int>(stream.ordinal);
    auto const load = [&](util::cuda_module_t& module, std::span<cuda_function_handle_t> into) noexcept {
        bytes_view_t const bundle {uform_encoder_nvidia_bundle_start, uform_encoder_nvidia_bundle_end};
        auto [loaded, status] = util::cuda_module_t::load(driver, device, bundle);
        if (status.failed()) return util::status_t {status};
        module = std::move(loaded);
        for (std::size_t index = 0; index != std::size(kernel_names_k); ++index)
            if (!(into[index] = module.function(kernel_names_k[index])))
                return util::status_t {util::status_code_t::device_k, "The bundle lacks a kernel"};
        return util::status_t {};
    };
    if (!stream.kernels || stream.kernels->on(stream.ordinal, load, functions).failed())
        return util::cuda_error_no_binary_for_gpu_k;
    return util::cuda_success_k;
}

/** Launch @p kernel over @p blocks blocks of @p threads on @p stream ; each argument is passed as
 *  the exact type of the kernel's parameter, as the driver copies its bytes. */
template <typename... argument_types_>
[[nodiscard]] static nk_status_t launch_(kernel_t kernel, unsigned blocks, unsigned threads, stream_t& stream,
                                         argument_types_... arguments) noexcept {
    void* addresses[] = {static_cast<void*>(&arguments)...}; // vendor-boundary: the launch takes argument addresses
    return in_context_(stream, [&](cuda_api_t const& driver, cuda_stream_handle_t native) {
        std::span<cuda_function_handle_t const> functions;
        if (cuda_result_t const result = functions_of_(stream, functions); result != util::cuda_success_k)
            return result;
        util::cuda_launch_config_t config;
        config.grid_x = blocks, config.block_x = threads, config.stream = native;
        return driver.launch_kernel(&config, functions[std::size_t(kernel)], addresses, nullptr);
    });
}

/** Blocks for a grid-stride walk over @p total items. */
[[nodiscard]] static unsigned stride_blocks_(std::size_t total) noexcept {
    std::size_t const blocks = divide_round_up(total, entries_block_threads_k);
    return static_cast<unsigned>(blocks == 0                          ? 1
                                 : blocks < entries_grid_blocks_max_k ? blocks
                                                                      : entries_grid_blocks_max_k);
}

nk_allocator_t ue::vram_nvidia_entries_kit_t::policy(stream_t&) noexcept {
    if (!driver_()) return {};
    // Weights live as long as their model, on no stream: a blocking allocation with the device
    // bound, and the blocking free that pairs with it
    nk_allocator_t policy {};
    policy.allocate = [](nk_size_t bytes, void*, nk_stream_t) noexcept -> void* {
        cuda_device_address_t address = 0;
        return driver_().allocate(&address, bytes) == util::cuda_success_k ? std::bit_cast<void*>(address) : nullptr;
    };
    policy.free = [](void* pointer, nk_size_t, void*, nk_stream_t) noexcept {
        [[maybe_unused]] cuda_result_t const freed = driver_().free(std::bit_cast<cuda_device_address_t>(pointer));
    };
    return policy;
}

void ue::vram_nvidia_entries_kit_t::bind(std::size_t vendor_ordinal) noexcept {
    [[maybe_unused]] cuda_result_t const bound = util::cuda_set_device(driver_(), static_cast<int>(vendor_ordinal));
}

nk_capability_t ue::vram_nvidia_entries_kit_t::capabilities_compiled(util::gpu_device_t const& device) noexcept {
    nk_capability_t compiled = 0;
    return nk_capabilities_compiled_cuda(device.vendor_ordinal, &compiled) == nk_success_k ? compiled : 0;
}

nk_status_t ue::vram_nvidia_entries_kit_t::copy(bytes_span_t target, bytes_view_t source, stream_t& stream) noexcept {
    return in_context_(stream, [&](cuda_api_t const& driver, cuda_stream_handle_t native) {
        return driver.copy_async(std::bit_cast<cuda_device_address_t>(target.data()),
                                 std::bit_cast<cuda_device_address_t>(source.data()), source.size(), native);
    });
}

nk_status_t ue::vram_nvidia_entries_kit_t::zero(bytes_span_t target, stream_t& stream) noexcept {
    return in_context_(stream, [&](cuda_api_t const& driver, cuda_stream_handle_t native) {
        return driver.fill_async(std::bit_cast<cuda_device_address_t>(target.data()), 0, target.size(), native);
    });
}

nk_status_t ue::vram_nvidia_entries_kit_t::fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                                       u32r1_span_t pool_offsets, stream_t& stream) noexcept {
    return launch_(kernel_t::fill_layout_k, 1, 1, stream, lengths, packed_offsets, pool_offsets);
}

nk_status_t ue::vram_nvidia_entries_kit_t::fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                                       nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                                       u32_t frame_stride, u32_t positions, i16r2_span_t coords,
                                                       stream_t& stream) noexcept {
    return launch_(kernel_t::fill_coords_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, image_shapes, video_shapes, tube_frames, frame_stride, positions, coords);
}

nk_status_t ue::vram_nvidia_entries_kit_t::gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens,
                                                        bf16r2_view_t table, bf16r2_span_t rows,
                                                        stream_t& stream) noexcept {
    return launch_(kernel_t::gather_bytes_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, tokens, table, rows);
}

nk_status_t ue::vram_nvidia_entries_kit_t::stage_patches(segment_layout_t layout, modality_t target,
                                                         nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                                         f32r1_view_t inverse_std, bf16r2_span_t patches,
                                                         stream_t& stream) noexcept {
    nk::tensor_span<nk_bf16_t, 3> const cells {patches.data(), {patches.extent(0), pixels.extent(1), pixels.extent(2)}};
    return launch_(kernel_t::stage_patches_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, target, pixels, mean, inverse_std, cells);
}

nk_status_t ue::vram_nvidia_entries_kit_t::stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples,
                                                        bf16r2_span_t frames, stream_t& stream) noexcept {
    return launch_(kernel_t::stage_frames_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, samples, frames);
}

nk_status_t ue::vram_nvidia_entries_kit_t::mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                                 i16r2_view_t coords, rope_grids_t angles, stream_t& stream) noexcept {
    return launch_(kernel_t::mrope_k, stride_blocks_(values.extent(0) * (head_width / 2u)), entries_block_threads_k,
                   stream, values, angles, coords, heads, head_width, role_offset);
}

nk_status_t ue::vram_nvidia_entries_kit_t::quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow,
                                                                f32r1_span_t token_scales, stream_t& stream) noexcept {
    return launch_(kernel_t::quantize_bf16_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_nvidia_entries_kit_t::quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow,
                                                                f32r1_span_t token_scales, stream_t& stream) noexcept {
    return launch_(kernel_t::quantize_f32_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_nvidia_entries_kit_t::rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales,
                                                          f32_t weight_descale, bf16r2_view_t residual,
                                                          bf16r2_span_t values, stream_t& stream) noexcept {
    return launch_(kernel_t::rescale_tokens_k, stride_blocks_(values.extent(0) * values.extent(1)),
                   entries_block_threads_k, stream, products, token_scales, weight_descale, residual, values);
}
