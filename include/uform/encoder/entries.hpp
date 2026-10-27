/**
 *  @file include/uform/encoder/entries.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief What the tower steps NumKong has no kernel for share across residencies: the packed
 *      layout of a batch, the rotary angle grids, and the E4M3 scaling constants.
 *
 *  The steps themselves are the host entries of @ref ram_kits.hpp and the vendor entries kits of
 *  @ref vram_kits.hpp . Each mirrors one NVIDIA kernel of the training tower element for element,
 *  so a checkpoint encodes the same on every residency:
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

#include <numkong/capabilities.h> // `nk_capability_t`, `nk_stream_t`
#include <numkong/types.h>        // `nk_bf16_t`, `nk_e4m3_t`

#include "uform/encoder/multimodal_batch.hpp" // `modality_t`, the shape views
#include "uform/encoder/tower_kernels.cuh"    // `e4m3_range_k`, `e4m3_descale_floor_k`
#include "uform/utilities/types.hpp"          // the scalar aliases and the rank-tagged views, `st::basic_vector`

namespace unum::uform::encoder {

namespace uu = unum::uform::utilities;
namespace nk = ashvardanian::numkong;
namespace st = ashvardanian::smashtable;

using uu::expected;
using uu::f32_t;
using uu::i16_t;
using uu::pixel_t;
using uu::token_id_t;
using uu::u16_t;
using uu::u32_t;
using uu::u64_t;

using uu::f32r1_span_t;
using uu::f32r1_view_t;
using uu::f32r2_span_t;
using uu::f32r2_view_t;
using uu::i16r2_span_t;
using uu::i16r2_view_t;
using uu::u32r1_span_t;
using uu::u32r1_view_t;

/** BFloat16 and E4M3 borrows, in NumKong's vendor-neutral scalars. */
using bf16r1_view_t = nk::tensor_view<nk_bf16_t, 1>;
using bf16r1_span_t = nk::tensor_span<nk_bf16_t, 1>;
using bf16r2_view_t = nk::tensor_view<nk_bf16_t, 2>;
using bf16r2_span_t = nk::tensor_span<nk_bf16_t, 2>;
using e4m3r1_span_t = nk::tensor_span<nk_e4m3_t, 1>;
using e4m3r2_view_t = nk::tensor_view<nk_e4m3_t, 2>;
using e4m3r2_span_t = nk::tensor_span<nk_e4m3_t, 2>;

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

/** Threads per block of every vendor kernel, a power of two the quantize reduction halves. */
inline constexpr unsigned entries_block_threads_k = 256;

} // namespace unum::uform::encoder
