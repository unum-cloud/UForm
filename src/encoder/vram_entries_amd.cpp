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
 *  a stream on it launches, and keeps it for the life of the process. Every entry runs on the
 *  current device, the one the executor selected, never one a stream names, and so does every
 *  allocation and free of the executor's policies.
 */
#include <numkong/capabilities.h> // `nk_capabilities_detected_rocm`, `nk_capabilities_compiled_rocm`

#include "uform/encoder/vram_kits.hpp" // `vram_amd_entries_kit_t`, `segment_layout_t`, `entries_block_threads_k`

#include "lib.hpp" // `gpu_runtime` - the runtime and the bundle's tables

namespace fu = ashvardanian::forkunion;
namespace ue = unum::uform::encoder;
namespace util = unum::uform::utilities;
using namespace ue;
using util::b8_t;
using util::bytes_span_t;
using util::bytes_view_t;
using util::divide_round_up;
using util::hip_api_t;
using util::hip_function_handle_t;
using util::hip_module_handle_t;
using util::hip_result_t;
using util::hip_stream_handle_t;
using util::u8_t;

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

static char const* const kernel_names_k[] = {
    "uform_fill_layout_kernel",          "uform_fill_coords_kernel",         "uform_gather_bytes_kernel",
    "uform_stage_patches_kernel",        "uform_stage_frames_kernel",        "uform_mrope_kernel",
    "uform_quantize_bf16_tokens_kernel", "uform_quantize_f32_tokens_kernel", "uform_rescale_tokens_kernel",
};

/** The runtime every entry calls through, the GPU runtime's; empty without one. */
[[nodiscard]] static hip_api_t const& runtime_() noexcept { return uf::gpu_drivers().hip; }

using stream_t = vram_amd_entries_kit_t::stream_t;

/** Run @p enqueue on @p stream , whose device is bound, and report its result. */
template <typename enqueue_type_>
[[nodiscard]] static nk_status_t on_device_(stream_t& stream, enqueue_type_ const& enqueue) noexcept {
    hip_api_t const& runtime = stream.handle.api();
    if (!runtime) return nk_device_code_mismatch_k;
    return enqueue(runtime, static_cast<hip_stream_handle_t>(stream.handle)) == util::hip_success_k
               ? nk_success_k
               : nk_device_code_mismatch_k;
}

/** The functions of the bundle on @p stream 's device, which its kernel tables load there on first
 *  use. */
[[nodiscard]] static hip_result_t functions_of_(stream_t& stream,
                                                std::span<hip_function_handle_t const>& functions) noexcept {
    hip_api_t const& runtime = stream.handle.api();
    int const device = static_cast<int>(stream.ordinal);
    auto const load = [&](util::hip_module_t& module, std::span<hip_function_handle_t> into) noexcept {
        bytes_view_t const bundle {uform_encoder_amd_bundle_start, uform_encoder_amd_bundle_end};
        auto [loaded, status] = util::hip_module_t::load(runtime, device, bundle);
        if (status.failed()) return util::status_t {status};
        module = std::move(loaded);
        for (std::size_t index = 0; index != std::size(kernel_names_k); ++index)
            if (!(into[index] = module.function(kernel_names_k[index])))
                return util::status_t {util::status_code_t::device_k, "The bundle lacks a kernel"};
        return util::status_t {};
    };
    if (!stream.kernels || stream.kernels->on(stream.ordinal, load, functions).failed())
        return util::hip_error_no_binary_for_gpu_k;
    return util::hip_success_k;
}

/** Launch @p kernel over @p blocks blocks of @p threads on @p stream ; each argument is passed as
 *  the exact type of the kernel's parameter, as the runtime copies its bytes. */
template <typename... argument_types_>
[[nodiscard]] static nk_status_t launch_(kernel_t kernel, unsigned blocks, unsigned threads, stream_t& stream,
                                         argument_types_... arguments) noexcept {
    void* addresses[] = {static_cast<void*>(&arguments)...}; // vendor-boundary: the launch takes argument addresses
    return on_device_(stream, [&](hip_api_t const& runtime, hip_stream_handle_t native) {
        std::span<hip_function_handle_t const> functions;
        if (hip_result_t const result = functions_of_(stream, functions); result != util::hip_success_k) return result;
        return runtime.module_launch_kernel(functions[std::size_t(kernel)], blocks, 1, 1, threads, 1, 1, 0, native,
                                            addresses, nullptr);
    });
}

/** Blocks for a grid-stride walk over @p total items. */
[[nodiscard]] static unsigned stride_blocks_(std::size_t total) noexcept {
    std::size_t const blocks = divide_round_up(total, entries_block_threads_k);
    return static_cast<unsigned>(blocks == 0                          ? 1
                                 : blocks < entries_grid_blocks_max_k ? blocks
                                                                      : entries_grid_blocks_max_k);
}

nk_allocator_t ue::vram_amd_entries_kit_t::policy(stream_t&) noexcept {
    if (!runtime_()) return {};
    // Weights live as long as their model, on no stream: a blocking allocation with the device
    // bound, and the blocking free that pairs with it
    nk_allocator_t policy {};
    policy.allocate = [](nk_size_t bytes, void*, nk_stream_t) noexcept -> void* {
        void* pointer = nullptr;
        return runtime_().malloc(&pointer, bytes) == util::hip_success_k ? pointer : nullptr;
    };
    policy.free = [](void* pointer, nk_size_t, void*, nk_stream_t) noexcept {
        [[maybe_unused]] hip_result_t const freed = runtime_().free(pointer);
    };
    return policy;
}

void ue::vram_amd_entries_kit_t::bind(std::size_t vendor_ordinal) noexcept {
    [[maybe_unused]] hip_result_t const bound = util::hip_set_device(runtime_(), static_cast<int>(vendor_ordinal));
}

nk_capability_t ue::vram_amd_entries_kit_t::capabilities_compiled(util::gpu_device_t const& device) noexcept {
    nk_capability_t compiled = 0;
    return nk_capabilities_compiled_rocm(device.vendor_ordinal, &compiled) == nk_success_k ? compiled : 0;
}

nk_status_t ue::vram_amd_entries_kit_t::copy(bytes_span_t target, bytes_view_t source, stream_t& stream) noexcept {
    return on_device_(stream, [&](hip_api_t const& runtime, hip_stream_handle_t native) {
        return runtime.copy_async(target.data(), source.data(), source.size(), util::hip_copy_default_k, native);
    });
}

nk_status_t ue::vram_amd_entries_kit_t::zero(bytes_span_t target, stream_t& stream) noexcept {
    return on_device_(stream, [&](hip_api_t const& runtime, hip_stream_handle_t native) {
        return runtime.fill_async(target.data(), 0, target.size(), native);
    });
}

nk_status_t ue::vram_amd_entries_kit_t::fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets,
                                                    u32r1_span_t pool_offsets, stream_t& stream) noexcept {
    return launch_(kernel_t::fill_layout_k, 1, 1, stream, lengths, packed_offsets, pool_offsets);
}

nk_status_t ue::vram_amd_entries_kit_t::fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                                    nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames,
                                                    u32_t frame_stride, u32_t positions, i16r2_span_t coords,
                                                    stream_t& stream) noexcept {
    return launch_(kernel_t::fill_coords_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, image_shapes, video_shapes, tube_frames, frame_stride, positions, coords);
}

nk_status_t ue::vram_amd_entries_kit_t::gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> tokens,
                                                     bf16r2_view_t table, bf16r2_span_t rows,
                                                     stream_t& stream) noexcept {
    return launch_(kernel_t::gather_bytes_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, tokens, table, rows);
}

nk_status_t ue::vram_amd_entries_kit_t::stage_patches(segment_layout_t layout, modality_t target,
                                                      nk::tensor_view<pixel_t, 3> pixels, f32r1_view_t mean,
                                                      f32r1_view_t inverse_std, bf16r2_span_t patches,
                                                      stream_t& stream) noexcept {
    nk::tensor_span<nk_bf16_t, 3> const cells {patches.data(), {patches.extent(0), pixels.extent(1), pixels.extent(2)}};
    return launch_(kernel_t::stage_patches_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, target, pixels, mean, inverse_std, cells);
}

nk_status_t ue::vram_amd_entries_kit_t::stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> samples,
                                                     bf16r2_span_t frames, stream_t& stream) noexcept {
    return launch_(kernel_t::stage_frames_k, static_cast<unsigned>(layout.lengths.extent(0)), entries_block_threads_k,
                   stream, layout, samples, frames);
}

nk_status_t ue::vram_amd_entries_kit_t::mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                                              i16r2_view_t coords, rope_grids_t angles, stream_t& stream) noexcept {
    return launch_(kernel_t::mrope_k, stride_blocks_(values.extent(0) * (head_width / 2u)), entries_block_threads_k,
                   stream, values, angles, coords, heads, head_width, role_offset);
}

nk_status_t ue::vram_amd_entries_kit_t::quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow,
                                                             f32r1_span_t token_scales, stream_t& stream) noexcept {
    return launch_(kernel_t::quantize_bf16_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_amd_entries_kit_t::quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow,
                                                             f32r1_span_t token_scales, stream_t& stream) noexcept {
    return launch_(kernel_t::quantize_f32_tokens_k, static_cast<unsigned>(values.extent(0)), entries_block_threads_k,
                   stream, values, shadow, token_scales);
}

nk_status_t ue::vram_amd_entries_kit_t::rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales,
                                                       f32_t weight_descale, bf16r2_view_t residual,
                                                       bf16r2_span_t values, stream_t& stream) noexcept {
    return launch_(kernel_t::rescale_tokens_k, stride_blocks_(values.extent(0) * values.extent(1)),
                   entries_block_threads_k, stream, products, token_scales, weight_descale, residual, values);
}
