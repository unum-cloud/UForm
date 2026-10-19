/**
 *  @brief The host Lanczos resize of @ref unum::uform::media::resize_image , over decoded pixels and no codec, so
 *      a build without the codecs resizes the pixels its caller decoded.
 *  @file src/media/resize.cpp
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include <cmath> // `std::sin`, `std::floor`, `std::lround`

#include <algorithm> // `std::clamp`
#include <numbers>   // `std::numbers::pi`

#include "core/array.hpp" // `array` — the resize scratch
#include "media/image.hpp"

namespace um = unum::uform::media;
using namespace um;
using uc::array;
using uc::f32_t;
using uc::i64_t;

/** The lobes of the Lanczos window, and the taps one output pixel reads along an axis. */
inline constexpr i64_t lanczos_lobes_k = 3;
inline constexpr u32_t lanczos_taps_k = 6;

/** One output coordinate's Lanczos taps along an axis: the clamped source indices and their normalized weights. */
struct lanczos_tap_t {

    /** The source indices, clamped into the crop so its edge pixels replicate. */
    u32_t index[lanczos_taps_k] = {};

    /** The window weights, summing to one. */
    f32_t weight[lanczos_taps_k] = {};
};

/** The Lanczos-3 window @c sinc(x)·sinc(x/3) over @c |x|<3 . */
[[nodiscard]] static f64_t lanczos_(f64_t x) noexcept {
    if (x == 0.0) return 1.0;
    if (x <= -lanczos_lobes_k || x >= lanczos_lobes_k) return 0.0;
    f64_t const pi_x = std::numbers::pi * x;
    return lanczos_lobes_k * std::sin(pi_x) * std::sin(pi_x / lanczos_lobes_k) / (pi_x * pi_x);
}

/** Fill @p taps with one entry per output coordinate of a @p source_length → @p target_length resample. Output
 *  pixel centers map onto source pixel centers, @c (target+0.5)·scale−0.5 , and the window keeps its six-tap
 *  support whatever the scale. */
[[nodiscard]] static status_t lanczos_taps_(u32_t source_length, u32_t target_length,
                                            array<lanczos_tap_t> &taps) noexcept {
    if (taps.resize(target_length) != status_t::success_k) return status_t::bad_alloc_k;
    f64_t const scale = static_cast<f64_t>(source_length) / static_cast<f64_t>(target_length);
    i64_t const last = static_cast<i64_t>(source_length) - 1;
    for (u32_t target = 0; target != target_length; ++target) {
        f64_t const center = (target + 0.5) * scale - 0.5;
        i64_t const first = static_cast<i64_t>(std::floor(center)) - (lanczos_lobes_k - 1);
        lanczos_tap_t &tap = taps[target];
        f64_t weights[lanczos_taps_k] = {}, total = 0.0;
        for (u32_t k = 0; k != lanczos_taps_k; ++k) {
            i64_t const position = first + static_cast<i64_t>(k);
            weights[k] = lanczos_(center - static_cast<f64_t>(position));
            total += weights[k];
            tap.index[k] = static_cast<u32_t>(std::clamp<i64_t>(position, 0, last));
        }
        for (u32_t k = 0; k != lanczos_taps_k; ++k) tap.weight[k] = static_cast<f32_t>(weights[k] / total);
    }
    return status_t::success_k;
}

expected_t um::resize_image(pixel_image_view_t source, pixel_patches_span_t out) noexcept {
    u32_t const height = static_cast<u32_t>(source.extent(0)), width = static_cast<u32_t>(source.extent(1));
    u32_t const rows = static_cast<u32_t>(out.extent(0)), columns = static_cast<u32_t>(out.extent(1));
    u32_t const patch_side = static_cast<u32_t>(out.extent(2));
    if (source.extent(2) != image_channels_k || out.extent(4) != image_channels_k || out.extent(3) != patch_side)
        return expected_t::fail(status_t::shape_unsupported_k, "resize takes RGBA into square RGBA patches");
    if (width == 0 || height == 0 || rows == 0 || columns == 0 || patch_side == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "resize from {}x{} into {}x{} patches of {}", width,
                                height, columns, rows, patch_side);
    image_extent_t const target {columns * patch_side, rows * patch_side};
    crop_t const crop = image_center_crop(image_extent_t {width, height}, target);
    if (crop.width == 0 || crop.height == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "a {}x{} image has no {}x{} crop", width, height,
                                target.width, target.height);

    array<lanczos_tap_t> horizontal, vertical;
    array<f32_t> staged; // the crop's rows resampled to the target width, before the vertical pass
    if (lanczos_taps_(crop.width, target.width, horizontal) != status_t::success_k ||
        lanczos_taps_(crop.height, target.height, vertical) != status_t::success_k ||
        staged.resize_uninitialized(static_cast<std::size_t>(crop.height) * target.width * image_channels_k) !=
            status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "resize scratch allocation failed ({}x{})", target.width,
                                crop.height);

    nk::tensor_span<f32_t, 3> const staged_rows {staged.data(), {crop.height, target.width, image_channels_k}};
    for (u32_t y = 0; y != crop.height; ++y)
        for (u32_t x = 0; x != target.width; ++x) {
            lanczos_tap_t const &tap = horizontal[x];
            for (u32_t channel = 0; channel != image_channels_k; ++channel) staged_rows[y, x, channel] = 0.0f;
            for (u32_t k = 0; k != lanczos_taps_k; ++k)
                for (u32_t channel = 0; channel != image_channels_k; ++channel)
                    staged_rows[y, x, channel] += tap.weight[k] * source[crop.y + y, crop.x + tap.index[k], channel];
        }

    for (u32_t y = 0; y != target.height; ++y) {
        lanczos_tap_t const &tap = vertical[y];
        for (u32_t x = 0; x != target.width; ++x)
            for (u32_t channel = 0; channel != image_channels_k; ++channel) {
                f32_t sum = 0.0f;
                for (u32_t k = 0; k != lanczos_taps_k; ++k)
                    sum += tap.weight[k] * staged_rows[tap.index[k], x, channel];
                out[y / patch_side, x / patch_side, y % patch_side, x % patch_side, channel] = static_cast<pixel_t>(
                    std::lround(std::clamp(sum, 0.0f, 255.0f)));
            }
    }
    return {};
}
