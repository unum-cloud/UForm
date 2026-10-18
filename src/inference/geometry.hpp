/**
 *  @brief The shape and numerics of one encoder tower, read back from a checkpoint alone.
 *  @file src/inference/geometry.hpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *
 *  Every extent comes from two places that must agree: the tensor shapes and the metadata keys training
 *  writes at save. A media modality whose patch side or chunk is zero is absent, and so is its projection.
 */
#pragma once

#include "core/types.hpp"      // `expected`, the scalar aliases
#include "core/checkpoint.hpp" // `checkpoint_to_load_t`

namespace unum::uform::inference {

namespace uc = unum::uform::core;
namespace nk = ashvardanian::numkong;
namespace sz = ashvardanian::stringzilla;

using uc::expected;
using uc::f32_t;
using uc::u32_t;

/** Most channels a pixel carries, the extent of the per-channel statistics. */
inline constexpr u32_t pixel_channels_max_k = 4;

/** The tower's geometry: block extents, rotary split, context, and each modality's entry. */
struct geometry_t {
    u32_t depth = 0;
    u32_t hidden = 0;
    u32_t ffn_expansion = 0;
    u32_t head_width = 0;
    f32_t rms_epsilon = 0;

    /** Channels of each head the three rotary axes take, summing to @ref head_width . */
    u32_t rope_temporal = 0, rope_height = 0, rope_width = 0;
    f32_t rope_base = 0;

    /** The longest item, and the extent of every rotary angle grid. */
    u32_t context = 0;

    /** Rows of the byte embedding table. */
    u32_t vocabulary = 0;

    u32_t image_patch_side = 0, image_channels = 0;
    f32_t image_pixel_mean[pixel_channels_max_k] = {}, image_pixel_std[pixel_channels_max_k] = {};

    u32_t video_patch_side = 0, video_tube_frames = 0, video_frame_stride = 0, video_channels = 0;
    f32_t video_pixel_mean[pixel_channels_max_k] = {}, video_pixel_std[pixel_channels_max_k] = {};

    u32_t audio_chunk_samples = 0, audio_sample_rate = 0;

    [[nodiscard]] u32_t heads() const noexcept { return hidden / head_width; }
    [[nodiscard]] u32_t ffn() const noexcept { return ffn_expansion * hidden; }
    [[nodiscard]] u32_t image_scalars() const noexcept { return image_patch_side * image_patch_side * image_channels; }
    [[nodiscard]] u32_t video_scalars() const noexcept {
        return video_tube_frames * video_patch_side * video_patch_side * video_channels;
    }

    /**
     *  @brief Read the geometry from @p checkpoint 's metadata and cross-check it against its tensor shapes.
     *  @retval status_t::invalid_input_k A key is missing or unparsable, or a shape disagrees; the C layer
     *      reports it as @c uform_status_malformed_k .
     */
    [[nodiscard]] static expected<geometry_t> from(uc::checkpoint_to_load_t const &checkpoint) noexcept;
};

} // namespace unum::uform::inference
