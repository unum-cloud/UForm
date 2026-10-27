/**
 *  @file src/encoder/geometry.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Parsing the tower geometry out of checkpoint metadata, and the rotary angle grids it
 *      implies — see the headers for the contracts. Load-time work, compiled once at the baseline.
 */
#include <cmath> // `std::pow`

#include <charconv>     // `std::from_chars`
#include <system_error> // `std::errc`

#include "uform/encoder/entries.hpp" // `rope_angles_t`
#include "uform/encoder/geometry.hpp"

namespace uu = unum::uform::utilities;
using namespace unum::uform::encoder;
using uu::expected_t;
using uu::status_t;

/** Parse @p text whole into @p value , integral or floating. */
template <typename value_type_>
[[nodiscard]] static bool parse_(sz::string_view_t text, value_type_ &value) noexcept {
    char const *const end = text.data() + text.size();
    std::from_chars_result const parsed = std::from_chars(text.data(), end, value);
    return !text.empty() && parsed.ec == std::errc {} && parsed.ptr == end;
}

/** Read the required metadata @p key into @p value . */
template <typename value_type_>
[[nodiscard]] static expected_t read_(uu::checkpoint_to_load_t const &checkpoint, sz::string_view_t key,
                                      value_type_ &value) noexcept {
    sz::string_view_t const text = checkpoint.metadata(key);
    if (text.empty()) return expected_t::fail(status_t::invalid_input_k, "the checkpoint misses the {} key", key);
    if (!parse_(text, value))
        return expected_t::fail(status_t::invalid_input_k, "the {} key holds an unparsable {}", key, text);
    return {};
}

/** Read the required comma-separated metadata @p key into exactly as many floats as @p values
 *  holds. */
[[nodiscard]] static expected_t read_list_(uu::checkpoint_to_load_t const &checkpoint, sz::string_view_t key,
                                           std::span<f32_t> values) noexcept {
    sz::string_view_t remaining = checkpoint.metadata(key);
    if (remaining.empty()) return expected_t::fail(status_t::invalid_input_k, "the checkpoint misses the {} key", key);
    std::size_t const count = values.size();
    std::size_t parsed = 0;
    for (sz::string_view_t separator = ","; parsed != count && !separator.empty(); ++parsed) {
        auto [item, match, after] = remaining.partition(',');
        if (!parse_(item, values[parsed]))
            return expected_t::fail(status_t::invalid_input_k, "the {} key holds an unparsable {}", key, item);
        separator = match, remaining = after;
    }
    if (parsed != count || !remaining.empty())
        return expected_t::fail(status_t::invalid_input_k, "the {} key must list {} values", key, count);
    return {};
}

expected<geometry_t> geometry_t::from(uu::checkpoint_to_load_t const &checkpoint) noexcept {
    using result_t = expected<geometry_t>;
    geometry_t geometry;
    expected_t read = read_(checkpoint, "depth", geometry.depth);
    if (read) read = read_(checkpoint, "hidden", geometry.hidden);
    if (read) read = read_(checkpoint, "ffn_expansion", geometry.ffn_expansion);
    if (read) read = read_(checkpoint, "head_width", geometry.head_width);
    if (read) read = read_(checkpoint, "rms_epsilon", geometry.rms_epsilon);
    if (read) read = read_(checkpoint, "rope_temporal", geometry.rope_temporal);
    if (read) read = read_(checkpoint, "rope_height", geometry.rope_height);
    if (read) read = read_(checkpoint, "rope_width", geometry.rope_width);
    if (read) read = read_(checkpoint, "rope_base", geometry.rope_base);
    if (read) read = read_(checkpoint, "context", geometry.context);
    // Checkpoints from before the key trained one row as wide as their rotary grid.
    if (read)
        read = checkpoint.metadata("columns").empty() ? (geometry.columns = geometry.context, expected_t {})
                                                      : read_(checkpoint, "columns", geometry.columns);
    if (read) read = read_(checkpoint, "image_patch_side", geometry.image_patch_side);
    if (read) read = read_(checkpoint, "image_channels", geometry.image_channels);
    if (read) read = read_(checkpoint, "video_patch_side", geometry.video_patch_side);
    if (read) read = read_(checkpoint, "video_tube_frames", geometry.video_tube_frames);
    if (read) read = read_(checkpoint, "video_frame_stride", geometry.video_frame_stride);
    if (read) read = read_(checkpoint, "video_channels", geometry.video_channels);
    if (read) read = read_(checkpoint, "audio_chunk_samples", geometry.audio_chunk_samples);
    if (read) read = read_(checkpoint, "audio_sample_rate", geometry.audio_sample_rate);
    if (!read) return result_t::fail(std::move(read));
    if (geometry.image_channels > pixel_channels_max_k || geometry.video_channels > pixel_channels_max_k)
        return result_t::fail(status_t::invalid_input_k, "pixels carry at most {} channels (image {}, video {})",
                              pixel_channels_max_k, geometry.image_channels, geometry.video_channels);
    std::span<f32_t> const image_mean = std::span {geometry.image_pixel_mean}.first(geometry.image_channels);
    std::span<f32_t> const image_std = std::span {geometry.image_pixel_std}.first(geometry.image_channels);
    std::span<f32_t> const video_mean = std::span {geometry.video_pixel_mean}.first(geometry.video_channels);
    std::span<f32_t> const video_std = std::span {geometry.video_pixel_std}.first(geometry.video_channels);
    read = read_list_(checkpoint, "image_pixel_mean", image_mean);
    if (read) read = read_list_(checkpoint, "image_pixel_std", image_std);
    if (read) read = read_list_(checkpoint, "video_pixel_mean", video_mean);
    if (read) read = read_list_(checkpoint, "video_pixel_std", video_std);
    if (!read) return result_t::fail(std::move(read));

    if (geometry.depth == 0 || geometry.head_width == 0 || geometry.hidden % geometry.head_width != 0 ||
        geometry.context == 0 || geometry.columns == 0)
        return result_t::fail(status_t::invalid_input_k, "depth {} hidden {} head_width {} context {} build no tower",
                              geometry.depth, geometry.hidden, geometry.head_width, geometry.context);
    if (geometry.rope_temporal + geometry.rope_height + geometry.rope_width != geometry.head_width ||
        geometry.rope_temporal % 2 != 0 || geometry.rope_height % 2 != 0 || geometry.rope_width % 2 != 0)
        return result_t::fail(status_t::invalid_input_k, "the rotary split {}+{}+{} must pair up to the head width {}",
                              geometry.rope_temporal, geometry.rope_height, geometry.rope_width, geometry.head_width);

    nk::tensor_view<nk_bf16_t, 2> const embedding = checkpoint["model.embedding"];
    nk::tensor_view<nk_bf16_t, 2> const gate_up = checkpoint["model.blocks.0.gate_up"];
    if (embedding.empty() || embedding.extent(1) != geometry.hidden)
        return result_t::fail(status_t::invalid_input_k, "the embedding must be [vocabulary,{}]", geometry.hidden);
    if (gate_up.empty() || gate_up.extent(0) != 2u * geometry.ffn())
        return result_t::fail(status_t::invalid_input_k, "the first gate_up must have {} rows", 2u * geometry.ffn());
    geometry.vocabulary = static_cast<u32_t>(embedding.extent(0));

    nk::tensor_view<nk_bf16_t, 2> const projection = checkpoint["model.token_head.projection"];
    if (!projection.empty() && projection.extent(1) != geometry.hidden)
        return result_t::fail(status_t::invalid_input_k, "the token head must project [width,{}]", geometry.hidden);
    geometry.token_width = static_cast<u32_t>(projection.empty() ? 0 : projection.extent(0));
    nk::tensor_view<nk_bf16_t, 2> const decoder = checkpoint["model.masked_tokens"];
    if (!decoder.empty() && decoder.extent(1) != geometry.hidden)
        return result_t::fail(status_t::invalid_input_k, "the byte decoder must be [vocabulary,{}]", geometry.hidden);
    geometry.masked_vocabulary = static_cast<u32_t>(decoder.empty() ? 0 : decoder.extent(0));
    nk::tensor_view<f32_t, 2> const temperatures = checkpoint.float32("choices.inverse_temperatures");
    geometry.choice_temperature_groups = static_cast<u32_t>(decoder.empty()        ? 0
                                                            : temperatures.empty() ? 1
                                                                                   : temperatures.numel());
    return result_t {geometry};
}

expected<rope_angles_t> rope_angles_t::make(u32_t positions, u32_t temporal, u32_t height, u32_t width,
                                            f32_t base) noexcept {
    using result_t = expected<rope_angles_t>;
    rope_angles_t angles;
    angles.positions_ = positions;
    angles.pairs_[0] = temporal / 2u, angles.pairs_[1] = height / 2u, angles.pairs_[2] = width / 2u;
    u32_t const denominators[3] = {temporal + height + width, height, width};
    std::size_t const pairs = angles.pairs_[0] + angles.pairs_[1] + angles.pairs_[2];
    if (st::failed(angles.angles_.resize(static_cast<std::size_t>(positions) * pairs)))
        return result_t::fail(status_t::bad_alloc_k, "angle grid allocation failed ({} positions)", positions);
    std::size_t offset = 0;
    for (u32_t axis = 0; axis != 3; ++axis)
        for (u32_t position = 0; position != positions; ++position)
            for (u32_t pair = 0; pair != angles.pairs_[axis]; ++pair)
                angles.angles_[offset++] = static_cast<f32_t>(position) *
                                           static_cast<f32_t>(std::pow(static_cast<uu::f64_t>(base),
                                                                       -2.0 * pair / denominators[axis]));
    return result_t {std::move(angles)};
}
