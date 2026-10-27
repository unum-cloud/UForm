/**
 *  @file include/uform/encoder/geometry.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The shape and numerics of one encoder tower, read back from a checkpoint alone.
 *
 *  Every extent comes from two places that must agree: the tensor shapes and the metadata keys training
 *  writes at save. A media modality whose patch side or chunk is zero is absent, and so is its projection.
 */
#pragma once

#include "uform/utilities/safetensors_format.hpp" // `checkpoint_to_load_t`
#include "uform/utilities/types.hpp"              // `expected`, the scalar aliases

namespace unum::uform::encoder {

namespace uu = unum::uform::utilities;
namespace nk = ashvardanian::numkong;
namespace sz = ashvardanian::stringzilla;

using uu::expected;
using uu::f32_t;
using uu::u32_t;

/** Most channels a pixel carries, the extent of the per-channel statistics. */
inline constexpr u32_t pixel_channels_max_k = 4;

/** The tower's geometry: block extents, rotary split, context, and each modality's entry. */
struct geometry_t {
    u32_t depth = 0;
    u32_t hidden = 0;
    u32_t ffn_expansion = 0;
    u32_t head_width = 0;
    f32_t rms_epsilon = 0;

    /** Channels of each head the three rotary axes take, summing to @c head_width . */
    u32_t rope_temporal = 0, rope_height = 0, rope_width = 0;
    f32_t rope_base = 0;

    /** The extent of every rotary angle grid. */
    u32_t context = 0;

    /** The training row width: the longest text, and the row media items take their share of. */
    u32_t columns = 0;

    /** Rows of the byte embedding table. */
    u32_t vocabulary = 0;

    u32_t image_patch_side = 0, image_channels = 0;
    f32_t image_pixel_mean[pixel_channels_max_k] = {}, image_pixel_std[pixel_channels_max_k] = {};

    u32_t video_patch_side = 0, video_tube_frames = 0, video_frame_stride = 0, video_channels = 0;
    f32_t video_pixel_mean[pixel_channels_max_k] = {}, video_pixel_std[pixel_channels_max_k] = {};

    u32_t audio_chunk_samples = 0, audio_sample_rate = 0;

    /** Scalars per token vector of the late-interaction head, from its projection's rows; zero
     *  without one. */
    u32_t token_width = 0;

    /** Rows of the byte decoder @c masked_tokens , the choices a masked position scores; zero
     *  without one. */
    u32_t masked_vocabulary = 0;

    /** Question types the choice temperatures were calibrated over: zero without a decoder, one
     *  when uncalibrated. */
    u32_t choice_temperature_groups = 0;

    [[nodiscard]] u32_t heads() const noexcept { return hidden / head_width; }
    [[nodiscard]] u32_t ffn() const noexcept { return ffn_expansion * hidden; }
    [[nodiscard]] u32_t image_scalars() const noexcept { return image_patch_side * image_patch_side * image_channels; }
    [[nodiscard]] u32_t video_scalars() const noexcept {
        return video_tube_frames * video_patch_side * video_patch_side * video_channels;
    }

    /**
     *  @brief Read the geometry from @p checkpoint 's metadata and cross-check it against its
     *      tensor shapes.
     *
     *  @par Statuses:
     *  - @c status_t::invalid_argument_k A key is missing or unparsable, or a shape disagrees;
     *    the C layer reports it as @c uform_status_malformed_k .
     */
    [[nodiscard]] static expected<geometry_t> from(uu::checkpoint_to_load_t const &checkpoint) noexcept;
};

} // namespace unum::uform::encoder
