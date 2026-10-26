/**
 *  @file include/uform/encoder/entries.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The tower steps NumKong has no kernel for, as host loops split across a thread pool:
 *      the per-modality entries, the coordinate fill, M-RoPE, and the per-token E4M3 scaling
 *      around each quantized product.
 *
 *  Each step mirrors one NVIDIA kernel of the training tower element for element, so a checkpoint encodes the same
 *  on every residency:
 *
 *      gather_bytes          tower_text_entry_kernel_
 *      stage_patches         tower_normalize_segment_kernel_
 *      stage_frames          tower_audio_segment_kernel_
 *      fill_coords           fill_coords_kernel_, with the temporal coordinate scaled by the frame stride
 *      mrope                 transformer_rope_kernel_ and transformer_rope_angle_
 *      quantize_tokens_e4m3  transformer_token_quantize_kernel_
 *      rescale_tokens        transformer_token_rescale_kernel_
 *
 *  The batch grid keeps an item's cells at its segment offset; the tower runs on @b packed rows instead, every
 *  segment's live cells back to back, so the zero cells of short items cost nothing. @ref segment_layout_t maps one
 *  onto the other, and each step reads the grid and writes the packed rows.
 */
#pragma once

#include <span> // `std::span` — the angles a tower copies

#include <forkunion.hpp> // vendor-boundary: `fu::flat_pool_t`, the threads every step splits across

#include <numkong/capabilities.h> // `nk_capability_t`, `nk_stream_t`
#include <numkong/types.h>        // `nk_bf16_t`, `nk_e4m3_t`

#include "uform/core/multimodal_batch.hpp" // `modality_t`, the shape views
#include "uform/core/types.hpp"            // the scalar aliases and the rank-tagged views, `st::basic_vector`

namespace unum::uform::encoder {

namespace uc = unum::uform::core;
namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;
namespace st = ashvardanian::smashtable;

using uc::expected;
using uc::f32_t;
using uc::i16_t;
using uc::modality_t;
using uc::pixel_t;
using uc::token_id_t;
using uc::u16_t;
using uc::u32_t;
using uc::u64_t;

using uc::f32r1_span_t;
using uc::f32r1_view_t;
using uc::f32r2_span_t;
using uc::f32r2_view_t;
using uc::i16r2_span_t;
using uc::i16r2_view_t;
using uc::u32r1_view_t;

/** BFloat16 and E4M3 borrows, in NumKong's vendor-neutral scalars. */
using bf16r1_view_t = nk::tensor_view<nk_bf16_t, 1>;
using bf16r2_view_t = nk::tensor_view<nk_bf16_t, 2>;
using bf16r2_span_t = nk::tensor_span<nk_bf16_t, 2>;
using e4m3r2_view_t = nk::tensor_view<nk_e4m3_t, 2>;
using e4m3r2_span_t = nk::tensor_span<nk_e4m3_t, 2>;

/** E4M3's largest normal magnitude, which every quantized row's @c max_abs maps onto. */
inline constexpr f32_t e4m3_range_k = 240.0f;

/** The scale floor keeping an all-zero row representable. */
inline constexpr f32_t e4m3_descale_floor_k = 1.0e-12f;

/** Where every segment's live cells sit in the batch grid, and where they land in the packed rows. */
struct segment_layout_t {

    /** @b [segments] the first grid cell of each segment, the batch's own @c segment_offsets . */
    u32r1_view_t grid_offsets {};

    /** @b [segments] live cells per segment, zero for PAD. */
    u32r1_view_t lengths {};

    /** @b [segments+1] the first packed row of each segment, prefix sums of @ref lengths . */
    u32r1_view_t packed_offsets {};

    /** @b [segments] each segment's modality. */
    nk::tensor_view<modality_t, 1> modality {};
};

/** The three rotary angle grids @b [positions,pairs] one call reads, temporal first, in any memory. */
struct rope_grids_t {
    f32r2_view_t temporal {}, height {}, width {};
};

/** The three rotary angle grids, @c angle[position,pair]=position·base^(−2·pair/denominator) , temporal pairs
 *  first; the temporal denominator is the whole head width, so text coordinates match 1-D RoPE. */
struct rope_angles_t {

    /** Fill the grids for @p positions positions of each axis, @p temporal / @p height / @p width channels. */
    [[nodiscard]] static expected<rope_angles_t> make(u32_t positions, u32_t temporal, u32_t height, u32_t width,
                                                      f32_t base) noexcept;

    /** Every angle, the three grids back to back, for a copy into the memory a tower runs from. */
    [[nodiscard]] std::span<f32_t const> scalars() const noexcept { return angles_; }

    /** The three grids over @p copy , a copy of @ref scalars in any memory. */
    [[nodiscard]] rope_grids_t grids(f32_t const *copy) const noexcept {
        std::size_t const temporal = static_cast<std::size_t>(positions_) * pairs_[0];
        std::size_t const height = static_cast<std::size_t>(positions_) * pairs_[1];
        return {{copy, positions_, pairs_[0]},
                {copy + temporal, positions_, pairs_[1]},
                {copy + temporal + height, positions_, pairs_[2]}};
    }

  private:
    st::basic_vector<f32_t> angles_ {};
    u32_t positions_ = 0;
    u32_t pairs_[3] = {};
};

/** Rows @c [t,:] of @p rows for each text segment's cells, gathered @c table[token] ; other rows stay as they are.
 *  Token ids are trusted: packers write bytes only. */
void gather_bytes(segment_layout_t const &layout, nk::tensor_view<token_id_t, 1> grid_tokens, bf16r2_view_t table,
                  bf16r2_span_t rows, fu::flat_pool_t &pool) noexcept;

/** Each @p target segment's pixel cells @b [cells,spatial,channels] into patch rows @b [rows,spatial·channels] as
 *  @c (byte/255−mean)·inverse_std , channels innermost; other rows stay as they are. */
void stage_patches(segment_layout_t const &layout, modality_t target, nk::tensor_view<pixel_t, 3> grid_pixels,
                   f32r1_view_t mean, f32r1_view_t inverse_std, bf16r2_span_t patches, fu::flat_pool_t &pool) noexcept;

/** Each audio segment's raw frames @b [cells,chunk_samples] narrowed into rows; other rows stay as they are. */
void stage_frames(segment_layout_t const &layout, nk::tensor_view<f32_t, 2> grid_samples, bf16r2_span_t frames,
                  fu::flat_pool_t &pool) noexcept;

/** Every packed row's @c (t,h,w) into @p coords @b [rows,3] : text and audio @c (k,0,0) , image
 *  @c (0,k/cols,k%cols) , video @c (group·tube_frames·frame_stride,r,c) , PAD rows untouched. */
void fill_coords(segment_layout_t const &layout, nk::tensor_view<u16_t, 2> image_shapes,
                 nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride, i16r2_span_t coords,
                 fu::flat_pool_t &pool) noexcept;

/** Rotate the @p heads head vectors starting at column @p role_offset of every row in place: the NeoX split-half
 *  pair @c (k,k+head_width/2) turns by the angle its index selects, each coordinate clamped into the grids. */
void mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width, i16r2_view_t coords,
           rope_grids_t angles, fu::flat_pool_t &pool) noexcept;

/** Quantize each row of @p values by its own @c max_abs/240 into @p shadow , keeping the scale in @p token_scales . */
void quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                          fu::flat_pool_t &pool) noexcept;

/** @copydoc quantize_tokens_e4m3 */
void quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                          fu::flat_pool_t &pool) noexcept;

/** @c values[t,o]=products[t,o]·weight_descale·token_scales[t]+residual[t,o] , narrowed; an empty @p residual
 *  drops its term. */
void rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale, bf16r2_view_t residual,
                    bf16r2_span_t values, fu::flat_pool_t &pool) noexcept;

/** What a vendor's runtime reports about one of its devices beside its capabilities. */
struct device_description_t {

    /** The vendor's own name for the device, NUL-terminated and truncated to fit. */
    char name[64] = {};

    /** Streaming multiprocessors or compute units. */
    u32_t compute_units = 0;

    /** Device memory in bytes. */
    u64_t memory_bytes = 0;
};

/**
 *  @brief The same steps for one GPU vendor, queued on a stream over device-addressable views, plus the copies the
 *      tower moves weights with. Each vendor's unit defines one table with the same signatures, so the tower picks
 *      a vendor by the capability mask alone.
 *
 *  Every entry returns without waiting, @c nk_device_code_mismatch_k when the launch fails.
 */
struct device_kernels_t {

    /** The vendor's capability group, @c nk_cap_cuda_k or @c nk_cap_rocm_k . */
    nk_capability_t vendor = 0;

    /** Reads the name, compute units and memory of the vendor's device @p ordinal into
     *  @p description . */
    nk_status_t (*describe)(std::size_t ordinal, device_description_t &description) noexcept = nullptr;

    /** @p bytes from @p source to @p target , either side in host or device memory. */
    nk_status_t (*copy)(void *target, void const *source, std::size_t bytes, nk_stream_t stream) noexcept = nullptr;

    /** @p bytes zeros into device memory at @p target . */
    nk_status_t (*zero)(void *target, std::size_t bytes, nk_stream_t stream) noexcept = nullptr;

    /** Prefix sums of @p lengths into @p packed_offsets @b [segments+1] , and @c pool_offsets[s]=s over
     *  @b [segments+1] , the one query row per item the pool attends with. */
    nk_status_t (*fill_layout)(u32r1_view_t lengths, u32_t *packed_offsets, u32_t *pool_offsets,
                               nk_stream_t stream) noexcept = nullptr;

    /** @copydoc fill_coords ; coordinates clamp below @p positions , so @ref mrope reads no grid out of range. */
    nk_status_t (*fill_coords)(segment_layout_t layout, nk::tensor_view<u16_t, 2> image_shapes,
                               nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride,
                               u32_t positions, i16r2_span_t coords, nk_stream_t stream) noexcept = nullptr;

    /** @copydoc gather_bytes */
    nk_status_t (*gather_bytes)(segment_layout_t layout, nk::tensor_view<token_id_t, 1> grid_tokens,
                                bf16r2_view_t table, bf16r2_span_t rows, nk_stream_t stream) noexcept = nullptr;

    /** @copydoc stage_patches */
    nk_status_t (*stage_patches)(segment_layout_t layout, modality_t target, nk::tensor_view<pixel_t, 3> grid_pixels,
                                 f32r1_view_t mean, f32r1_view_t inverse_std, bf16r2_span_t patches,
                                 nk_stream_t stream) noexcept = nullptr;

    /** @copydoc stage_frames */
    nk_status_t (*stage_frames)(segment_layout_t layout, nk::tensor_view<f32_t, 2> grid_samples, bf16r2_span_t frames,
                                nk_stream_t stream) noexcept = nullptr;

    /** @copydoc mrope */
    nk_status_t (*mrope)(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width, i16r2_view_t coords,
                         rope_grids_t angles, nk_stream_t stream) noexcept = nullptr;

    /** @copydoc quantize_tokens_e4m3 */
    nk_status_t (*quantize_bf16_tokens_e4m3)(bf16r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                             nk_stream_t stream) noexcept = nullptr;

    /** @copydoc quantize_tokens_e4m3 */
    nk_status_t (*quantize_f32_tokens_e4m3)(f32r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                                            nk_stream_t stream) noexcept = nullptr;

    /** @copydoc rescale_tokens */
    nk_status_t (*rescale_tokens)(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale,
                                  bf16r2_view_t residual, bf16r2_span_t values, nk_stream_t stream) noexcept = nullptr;
};

/** Threads per block of every vendor kernel, a power of two the quantize reduction halves. */
inline constexpr unsigned entries_block_threads_k = 256;

/** The NVIDIA and AMD tables, each defined only by its vendor's unit, which a build without that vendor omits. */
extern device_kernels_t const vram_nvidia_kernels_k, vram_amd_kernels_k;

/** The tables of the vendors this build compiles, null for the others; CMake generates its one definition, so no
 *  source names a vendor the build lacks. */
extern device_kernels_t const *const device_kernels_k[2];

/** The vendor table whose capability group @p capabilities names, or null for a CPU mask or a vendor this build
 *  does not compile. */
[[nodiscard]] inline device_kernels_t const *device_kernels_of(nk_capability_t capabilities) noexcept {
    for (device_kernels_t const *kernels : device_kernels_k)
        if (kernels && (capabilities & kernels->vendor)) return kernels;
    return nullptr;
}

} // namespace unum::uform::encoder
