/**
 *  @file include/uform/encoder/vram_kits.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The GPU kits the encoder tower runs on: NumKong's packed products, packed attention and
 *      element-wise steps per NVIDIA, AMD and Apple generation, and a ladder per family whose floor
 *      is the vendor's generic kit.
 *
 *  Every kernel a kit names queues on the stream it is handed. Where a generation has only some
 *  of a family's kernels, its kit takes the rest from its vendor's earlier generations, and
 *  @c capability_k is the union of every bit the kit calls, as in @ref ram_kits.hpp . NVIDIA, AMD
 *  and Apple masks never overlap, so one ladder per family serves every vendor.
 *
 *  UForm's own steps come from one entries kit per vendor, declared here and defined by the
 *  vendor's unit beside the bundle of kernels it launches by name, so every class of a vendor
 *  shares one copy. The Metal kits take only NumKong's pack and workspace sizes, which are host
 *  arithmetic, and declare the rest, which @c vram_entries_apple.cpp defines by dispatching
 *  NumKong's MSL kernels on UForm's own buffers.
 */
#pragma once

#include <type_traits> // `std::conditional_t`

#include <numkong/attention.h> // `nk_attention_pack_*`, `nk_attention_packed_*`
#include <numkong/cast.h>      // `nk_cast_*`
#include <numkong/dots.h>      // `nk_dots_pack_*`, `nk_dots_packed_*`
#include <numkong/each.h>      // `nk_each_*`
#include <numkong/memory.h>    // `nk_allocator_t`

#include "uform/encoder/entries.hpp"  // `segment_layout_t`, `rope_grids_t`
#include "uform/encoder/ram_kits.hpp" // `highest_kit`

namespace unum::uform::encoder {

#pragma region Entries

/** What a vendor's runtime reports about one of its devices, its architecture numbers included. */
struct device_description_t {

    /** The vendor's own name for the device, NUL-terminated and truncated to fit. */
    char name[64] = {};

    /** Streaming multiprocessors or compute units. */
    u32_t compute_units = 0;

    /** Device memory in bytes. */
    u64_t memory_bytes = 0;

    /** The architecture's major number: NVIDIA's compute capability major, AMD's gfx major, the
     *  newest Apple GPU family. */
    u32_t architecture_major = 0;

    /** The architecture's minor number: NVIDIA's compute capability minor, AMD's gfx minor, zero
     *  on Apple. */
    u32_t architecture_minor = 0;
};

/**
 *  @brief NVIDIA's own encoder steps, the copies weights move with, and the device listing, which
 *      @c vram_entries_nvidia.cpp defines over the kernels it launches by name.
 *
 *  Every step queues on @p stream over device-addressable views and returns without waiting,
 *  @c nk_device_code_mismatch_k when the launch fails. Each mirrors the host entry of
 *  @ref ram_kits.hpp of the same name.
 */
struct vram_nvidia_entries_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;

    /** The allocation policy of the device @p stream submits to: its memory for a @p residency of
     *  VRAM, or pinned host pages the device addresses for RAM, through NumKong's allocators of the
     *  vendor; an empty policy when the build or the machine lacks them. */
    static nk_allocator_t policy(nk_stream_t stream, uform_residency_t residency) noexcept;

    /** Makes the vendor's device @p vendor_ordinal current on the calling thread, so the per-thread
     *  default stream, every allocation and free, and every step after it land there. */
    static nk_status_t select(std::size_t vendor_ordinal) noexcept;

    /** Waits for everything queued on @p stream . */
    static nk_status_t synchronize(nk_stream_t stream) noexcept;

    /** Reads the name, compute units, memory and compute capability of the vendor's device
     *  @p vendor_ordinal . */
    static nk_status_t describe(std::size_t vendor_ordinal, device_description_t& description) noexcept;

    /** @p source into @p target , as long, either side in host or device memory. */
    static nk_status_t copy(uu::bytes_span_t target, uu::bytes_view_t source, nk_stream_t stream) noexcept;

    /** Zeros into device memory at @p target . */
    static nk_status_t zero(uu::bytes_span_t target, nk_stream_t stream) noexcept;

    /** Prefix sums of @p lengths into @p packed_offsets @b [segments+1] , and @c pool_offsets[s]=s
     *  over @b [segments+1] , the one query row per item the pool attends with. */
    static nk_status_t fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets, u32r1_span_t pool_offsets,
                                   nk_stream_t stream) noexcept;

    /** Coordinates clamp below @p positions , so @c mrope reads no grid out of range. */
    static nk_status_t fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                   nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                   u32_t positions, i16r2_span_t coords, nk_stream_t stream) noexcept;
    static nk_status_t gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> grid_tokens,
                                    bf16r2_view_t table, bf16r2_span_t rows, nk_stream_t stream) noexcept;
    static nk_status_t stage_patches(segment_layout_t layout, modality_t target,
                                     nk::tensor_view<pixel_t, 3> grid_pixels, f32r1_view_t mean,
                                     f32r1_view_t inverse_std, bf16r2_span_t patches, nk_stream_t stream) noexcept;
    static nk_status_t stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> grid_samples,
                                    bf16r2_span_t frames, nk_stream_t stream) noexcept;
    static nk_status_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                             i16r2_view_t coords, rope_grids_t angles, nk_stream_t stream) noexcept;
    static nk_status_t quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                            nk_stream_t stream) noexcept;
    static nk_status_t quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                            nk_stream_t stream) noexcept;
    static nk_status_t rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                      bf16r2_view_t residual, bf16r2_span_t values, nk_stream_t stream) noexcept;
};

/** AMD's twin of @ref vram_nvidia_entries_kit_t , which @c vram_entries_amd.cpp defines. */
struct vram_amd_entries_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static nk_allocator_t policy(nk_stream_t stream, uform_residency_t residency) noexcept;
    static nk_status_t select(std::size_t vendor_ordinal) noexcept;
    static nk_status_t synchronize(nk_stream_t stream) noexcept;
    static nk_status_t describe(std::size_t vendor_ordinal, device_description_t& description) noexcept;
    static nk_status_t copy(uu::bytes_span_t target, uu::bytes_view_t source, nk_stream_t stream) noexcept;
    static nk_status_t zero(uu::bytes_span_t target, nk_stream_t stream) noexcept;
    static nk_status_t fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets, u32r1_span_t pool_offsets,
                                   nk_stream_t stream) noexcept;
    static nk_status_t fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                   nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                   u32_t positions, i16r2_span_t coords, nk_stream_t stream) noexcept;
    static nk_status_t gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> grid_tokens,
                                    bf16r2_view_t table, bf16r2_span_t rows, nk_stream_t stream) noexcept;
    static nk_status_t stage_patches(segment_layout_t layout, modality_t target,
                                     nk::tensor_view<pixel_t, 3> grid_pixels, f32r1_view_t mean,
                                     f32r1_view_t inverse_std, bf16r2_span_t patches, nk_stream_t stream) noexcept;
    static nk_status_t stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> grid_samples,
                                    bf16r2_span_t frames, nk_stream_t stream) noexcept;
    static nk_status_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                             i16r2_view_t coords, rope_grids_t angles, nk_stream_t stream) noexcept;
    static nk_status_t quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                            nk_stream_t stream) noexcept;
    static nk_status_t quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                            nk_stream_t stream) noexcept;
    static nk_status_t rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                      bf16r2_view_t residual, bf16r2_span_t values, nk_stream_t stream) noexcept;
};

/**
 *  @brief Apple's twin of @ref vram_nvidia_entries_kit_t , which @c vram_entries_apple.cpp defines.
 *
 *  Its memory is shared buffers the device's context records, which a launch resolves its operands
 *  through and makes resident: device memory by GPU address, as CUDA hands out device pointers,
 *  and pinned pages by host address. It waits through Metal, as UForm commits its command buffers.
 */
struct vram_apple_entries_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_metal_k;
    static nk_allocator_t policy(nk_stream_t stream, uform_residency_t residency) noexcept;

    /** Succeeds without a call, as every Metal queue names its own device. */
    static nk_status_t select(std::size_t vendor_ordinal) noexcept;
    static nk_status_t synchronize(nk_stream_t stream) noexcept;

    static nk_status_t describe(std::size_t vendor_ordinal, device_description_t& description) noexcept;
    static nk_status_t copy(uu::bytes_span_t target, uu::bytes_view_t source, nk_stream_t stream) noexcept;
    static nk_status_t zero(uu::bytes_span_t target, nk_stream_t stream) noexcept;
    static nk_status_t fill_layout(u32r1_view_t lengths, u32r1_span_t packed_offsets, u32r1_span_t pool_offsets,
                                   nk_stream_t stream) noexcept;
    static nk_status_t fill_coords(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                                   nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                                   u32_t positions, i16r2_span_t coords, nk_stream_t stream) noexcept;
    static nk_status_t gather_bytes(segment_layout_t layout, nk::tensor_view<token_id_t, 1> grid_tokens,
                                    bf16r2_view_t table, bf16r2_span_t rows, nk_stream_t stream) noexcept;
    static nk_status_t stage_patches(segment_layout_t layout, modality_t target,
                                     nk::tensor_view<pixel_t, 3> grid_pixels, f32r1_view_t mean,
                                     f32r1_view_t inverse_std, bf16r2_span_t patches, nk_stream_t stream) noexcept;
    static nk_status_t stage_frames(segment_layout_t layout, nk::tensor_view<f32_t, 2> grid_samples,
                                    bf16r2_span_t frames, nk_stream_t stream) noexcept;
    static nk_status_t mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width,
                             i16r2_view_t coords, rope_grids_t angles, nk_stream_t stream) noexcept;
    static nk_status_t quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                            nk_stream_t stream) noexcept;
    static nk_status_t quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                            nk_stream_t stream) noexcept;
    static nk_status_t rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                      bf16r2_view_t residual, bf16r2_span_t values, nk_stream_t stream) noexcept;
};

#pragma endregion Entries

#pragma region Products

#if NUMKONG_TARGET_CUDA
struct cuda_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cuda;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cuda;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cuda;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_cuda;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_cuda;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cuda;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cuda;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cuda;
};
#else
struct cuda_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA

#if NUMKONG_TARGET_AMPERE
struct ampere_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_ampere;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_ampere;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_ampere;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_ampere;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_ampere;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_ampere;
};
#else
struct ampere_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_HOPPER

/** Hopper's BFloat16 products beside Ampere's FP8 ones. */
struct hopper_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_hopper_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_hopper;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_hopper;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_hopper;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_hopper;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_hopper;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_ampere;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_ampere;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_ampere;
};
#else
struct hopper_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_hopper_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
struct blackwell_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_blackwell;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_blackwell;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_blackwell;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_blackwell;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_blackwell;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_blackwell;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_blackwell;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_blackwell;
};
#else
struct blackwell_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

/** Consumer Blackwell's FP8 products beside Ampere's BFloat16 ones. */
struct blackwellrtx_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_ampere;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_ampere;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_ampere;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_blackwellrtx;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_blackwellrtx;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_blackwellrtx;
};
#else
struct blackwellrtx_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_ROCM
struct rocm_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_rocm;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_rocm;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_rocm;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_rocm;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_rocm;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_rocm;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_rocm;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_rocm;
};
#else
struct rocm_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM

#if NUMKONG_TARGET_CDNA3
struct cdna3_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cdna3;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cdna3;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cdna3;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_cdna3;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_cdna3;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cdna3;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cdna3;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cdna3;
};
#else
struct cdna3_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA3

#if NUMKONG_TARGET_CDNA4
struct cdna4_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cdna4;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cdna4;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cdna4;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_cdna4;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_cdna4;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cdna4;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cdna4;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cdna4;
};
#else
struct cdna4_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5
struct cdna5_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_cdna5;
    static constexpr auto pack_bf16 = nk_dots_pack_bf16_cdna5;
    static constexpr auto multiply_bf16 = nk_dots_packed_bf16_cdna5;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_cdna5;
    static constexpr auto symmetric_bf16 = nk_dots_symmetric_bf16_cdna5;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_cdna5;
    static constexpr auto pack_e4m3 = nk_dots_pack_e4m3_cdna5;
    static constexpr auto multiply_e4m3 = nk_dots_packed_e4m3_cdna5;
};
#else
struct cdna5_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA5

#if NUMKONG_TARGET_METAL

/** The generic Metal products, from Apple7 on: NumKong's pack sizes, and the packs and tiles of its
 *  kernels that @c vram_entries_apple.cpp dispatches on UForm's own buffers. */
struct metal_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_metal;
    static decltype(nk_dots_pack_bf16_metal) pack_bf16;
    static decltype(nk_dots_packed_bf16_metal) multiply_bf16;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_metal;
    static decltype(nk_dots_symmetric_bf16_metal) symmetric_bf16;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_metal;
    static decltype(nk_dots_pack_e4m3_metal) pack_e4m3;
    static decltype(nk_dots_packed_e4m3_metal) multiply_e4m3;
};
#else
struct metal_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_METAL

#if NUMKONG_TARGET_APPLE9

/** The Apple9 products of the M3 and M4 GPUs, dispatched as the generic Metal ones are. */
struct apple9_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple9_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_apple9;
    static decltype(nk_dots_pack_bf16_apple9) pack_bf16;
    static decltype(nk_dots_packed_bf16_apple9) multiply_bf16;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_apple9;
    static decltype(nk_dots_symmetric_bf16_apple9) symmetric_bf16;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_apple9;
    static decltype(nk_dots_pack_e4m3_apple9) pack_e4m3;
    static decltype(nk_dots_packed_e4m3_apple9) multiply_e4m3;
};
#else
struct apple9_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple9_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_APPLE9

#if NUMKONG_TARGET_APPLE10

/** The Apple10 products of the M5 GPUs on Metal 4 tensors, dispatched as the generic ones are. */
struct apple10_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple10_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_dots_pack_size_bf16_apple10;
    static decltype(nk_dots_pack_bf16_apple10) pack_bf16;
    static decltype(nk_dots_packed_bf16_apple10) multiply_bf16;
    static constexpr auto multiply_workspace_size_bf16 = nk_dots_packed_workspace_size_bf16_apple10;
    static decltype(nk_dots_symmetric_bf16_apple10) symmetric_bf16;
    static constexpr auto pack_size_e4m3 = nk_dots_pack_size_e4m3_apple10;
    static decltype(nk_dots_pack_e4m3_apple10) pack_e4m3;
    static decltype(nk_dots_packed_e4m3_apple10) multiply_e4m3;
};
#else
struct apple10_products_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple10_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_APPLE10

/** The kit of the vendor whose group bit @p caps_ sets, NVIDIA's where it sets neither of the
 *  others. */
template <nk_capability_t caps_, typename rocm_kit_type_, typename metal_kit_type_, typename cuda_kit_type_>
using vendor_floor_kit =
    std::conditional_t<(caps_ & nk_cap_rocm_k) != 0, rocm_kit_type_,
                       std::conditional_t<(caps_ & nk_cap_metal_k) != 0, metal_kit_type_, cuda_kit_type_>>;

/** The products kit for one device mask, its vendor's generic kit the floor. */
template <nk_capability_t caps_>
struct vram_products_ladder {
    using floor_t = vendor_floor_kit<caps_, rocm_products_kit_t, metal_products_kit_t, cuda_products_kit_t>;
    using kit_t =
        typename highest_kit<caps_, floor_t, blackwell_products_kit_t, blackwellrtx_products_kit_t,
                             hopper_products_kit_t, ampere_products_kit_t, cdna5_products_kit_t, cdna4_products_kit_t,
                             cdna3_products_kit_t, apple10_products_kit_t, apple9_products_kit_t>::type;
};

#pragma endregion Products

#pragma region Attention

#if NUMKONG_TARGET_CUDA
struct cuda_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cuda;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cuda;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cuda;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_cuda;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_cuda;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_cuda;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_cuda;
};
#else
struct cuda_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA

#if NUMKONG_TARGET_AMPERE
struct ampere_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_ampere;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_ampere;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_ampere;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_ampere;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_ampere;
};
#else
struct ampere_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_HOPPER
struct hopper_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_hopper_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_hopper;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_hopper;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_hopper;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_hopper;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_hopper;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_hopper;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_hopper;
};
#else
struct hopper_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_hopper_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
struct blackwell_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_blackwell;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_blackwell;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_blackwell;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_blackwell;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_blackwell;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_blackwell;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_blackwell;
};
#else
struct blackwell_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

/** Consumer Blackwell's FP8 attention beside Ampere's BFloat16 one. */
struct blackwellrtx_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_ampere;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_ampere;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_ampere;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_ampere;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_blackwellrtx;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_blackwellrtx;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_blackwellrtx;
};
#else
struct blackwellrtx_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_ampere_k | nk_cap_blackwellrtx_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_ROCM
struct rocm_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_rocm;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_rocm;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_rocm;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_rocm;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_rocm;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_rocm;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_rocm;
};
#else
struct rocm_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM

#if NUMKONG_TARGET_ROCM && NUMKONG_TARGET_CDNA3

/** CDNA3's BFloat16 attention beside the generic ROCm FP8 one. */
struct cdna3_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k | nk_cap_cdna3_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cdna3;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cdna3;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cdna3;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_cdna3;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_rocm;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_rocm;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_rocm;
};
#else
struct cdna3_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k | nk_cap_cdna3_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM && NUMKONG_TARGET_CDNA3

#if NUMKONG_TARGET_CDNA4
struct cdna4_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cdna4;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cdna4;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cdna4;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_cdna4;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_cdna4;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_cdna4;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_cdna4;
};
#else
struct cdna4_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5
struct cdna5_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_cdna5;
    static constexpr auto pack_bf16 = nk_attention_pack_bf16_cdna5;
    static constexpr auto attend_bf16 = nk_attention_packed_bf16_cdna5;
    static constexpr auto attend_gradients_bf16 = nk_attention_packed_gradients_bf16_cdna5;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_cdna5;
    static constexpr auto pack_e4m3 = nk_attention_pack_e4m3_cdna5;
    static constexpr auto attend_e4m3 = nk_attention_packed_e4m3_cdna5;
};
#else
struct cdna5_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CDNA5

#if NUMKONG_TARGET_METAL

/** The generic Metal attention, from Apple7 on: NumKong's pack sizes, and the packs and passes
 *  of its kernels that @c vram_entries_apple.cpp dispatches on UForm's own buffers. */
struct metal_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_metal;
    static decltype(nk_attention_pack_bf16_metal) pack_bf16;
    static decltype(nk_attention_packed_bf16_metal) attend_bf16;
    static decltype(nk_attention_packed_gradients_bf16_metal) attend_gradients_bf16;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_metal;
    static decltype(nk_attention_pack_e4m3_metal) pack_e4m3;
    static decltype(nk_attention_packed_e4m3_metal) attend_e4m3;
};
#else
struct metal_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_METAL

#if NUMKONG_TARGET_APPLE9

/** The Apple9 attention of the M3 and M4 GPUs, dispatched as the generic Metal one is. */
struct apple9_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple9_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_apple9;
    static decltype(nk_attention_pack_bf16_apple9) pack_bf16;
    static decltype(nk_attention_packed_bf16_apple9) attend_bf16;
    static decltype(nk_attention_packed_gradients_bf16_apple9) attend_gradients_bf16;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_apple9;
    static decltype(nk_attention_pack_e4m3_apple9) pack_e4m3;
    static decltype(nk_attention_packed_e4m3_apple9) attend_e4m3;
};
#else
struct apple9_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple9_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_APPLE9

#if NUMKONG_TARGET_APPLE10

/** The Apple10 attention of the M5 GPUs on Metal 4 tensors, dispatched as the generic one is. */
struct apple10_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple10_k;
    static constexpr bool compiled_k = true;
    static constexpr auto pack_size_bf16 = nk_attention_pack_size_bf16_apple10;
    static decltype(nk_attention_pack_bf16_apple10) pack_bf16;
    static decltype(nk_attention_packed_bf16_apple10) attend_bf16;
    static decltype(nk_attention_packed_gradients_bf16_apple10) attend_gradients_bf16;
    static constexpr auto pack_size_e4m3 = nk_attention_pack_size_e4m3_apple10;
    static decltype(nk_attention_pack_e4m3_apple10) pack_e4m3;
    static decltype(nk_attention_packed_e4m3_apple10) attend_e4m3;
};
#else
struct apple10_attention_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_apple10_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_APPLE10

/** The attention kit for one device mask, its vendor's generic kit the floor. */
template <nk_capability_t caps_>
struct vram_attention_ladder {
    using floor_t = vendor_floor_kit<caps_, rocm_attention_kit_t, metal_attention_kit_t, cuda_attention_kit_t>;
    using kit_t = typename highest_kit<caps_, floor_t, blackwell_attention_kit_t, blackwellrtx_attention_kit_t,
                                       hopper_attention_kit_t, ampere_attention_kit_t, cdna5_attention_kit_t,
                                       cdna4_attention_kit_t, cdna3_attention_kit_t, apple10_attention_kit_t,
                                       apple9_attention_kit_t>::type;
};

#pragma endregion Attention

#pragma region Elementwise

#if NUMKONG_TARGET_CUDA
struct cuda_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_cuda;
    static constexpr auto sum_f32 = nk_each_sum_f32_cuda;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_cuda;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_cuda;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_cuda;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_cuda;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_cuda;
};
#else
struct cuda_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA

#if NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE

/** Ampere's casts, norms and BFloat16 gates over the generic CUDA sums, Float32
 *  gates and rotation. */
struct ampere_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_ampere;
    static constexpr auto sum_f32 = nk_each_sum_f32_cuda;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_ampere;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_ampere;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_cuda;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_ampere;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_cuda;
};
#else
struct ampere_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_ADA

/** Ada's FP8 casts over Ampere's norms and gates. */
struct ada_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k | nk_cap_ada_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_ada;
    static constexpr auto sum_f32 = nk_each_sum_f32_cuda;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_ampere;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_ampere;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_cuda;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_ampere;
    static constexpr auto rope_bf16 = nk_attention_rope_bf16_cuda;
};
#else
struct ada_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_cuda_k | nk_cap_ampere_k | nk_cap_ada_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_CUDA && NUMKONG_TARGET_AMPERE && NUMKONG_TARGET_ADA

#if NUMKONG_TARGET_ROCM

/** The generic ROCm kernels, without a rotation, as NumKong ships no RoPE for ROCm. */
struct rocm_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = true;
    static constexpr auto cast = nk_cast_rocm;
    static constexpr auto sum_f32 = nk_each_sum_f32_rocm;
    static constexpr auto rmscast_bf16 = nk_each_rmscast_bf16_rocm;
    static constexpr auto rmsnorm_bf16 = nk_each_rmsnorm_bf16_rocm;
    static constexpr auto swiglu_f32 = nk_each_swiglu_f32_rocm;
    static constexpr auto swiglu_bf16 = nk_each_swiglu_bf16_rocm;
};
#else
struct rocm_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_ROCM

#if NUMKONG_TARGET_METAL

/** The generic Metal kernels, which every later Apple family runs as they are, dispatched by
 *  @c vram_entries_apple.cpp on UForm's own buffers. */
struct metal_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr bool compiled_k = true;
    static decltype(nk_cast_metal) cast;
    static decltype(nk_each_sum_f32_metal) sum_f32;
    static decltype(nk_each_rmscast_bf16_metal) rmscast_bf16;
    static decltype(nk_each_rmsnorm_bf16_metal) rmsnorm_bf16;
    static decltype(nk_each_swiglu_f32_metal) swiglu_f32;
    static decltype(nk_each_swiglu_bf16_metal) swiglu_bf16;
    static decltype(nk_attention_rope_bf16_metal) rope_bf16;
};
#else
struct metal_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_METAL

/** The element-wise kit for one device mask, its vendor's generic kit the floor. */
template <nk_capability_t caps_>
struct vram_elementwise_ladder {
    using floor_t = vendor_floor_kit<caps_, rocm_elementwise_kit_t, metal_elementwise_kit_t, cuda_elementwise_kit_t>;
    using kit_t = typename highest_kit<caps_, floor_t, ada_elementwise_kit_t, ampere_elementwise_kit_t>::type;
};

#pragma endregion Elementwise

} // namespace unum::uform::encoder
