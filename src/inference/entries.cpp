/**
 *  @brief Host implementations of the tower steps NumKong has no kernel for — see the header for the contract.
 *  @file src/inference/entries.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 */
#include <cmath> // `std::pow`, `std::sin`, `std::cos`, `std::fabs`

#include <numkong/cast.h> // `nk_bf16_to_f32_serial`, `nk_f32_to_bf16_serial`, `nk_f32_to_e4m3_serial`

#include "inference/entries.hpp"

namespace ui = unum::uform::inference;
using namespace ui;
using uc::f64_t;
using uc::status_t;

[[nodiscard]] static f32_t widen_(nk_bf16_t value) noexcept {
    f32_t wide = 0;
    nk_bf16_to_f32_serial(&value, &wide);
    return wide;
}

[[nodiscard]] static nk_bf16_t narrow_(f32_t value) noexcept {
    nk_bf16_t narrow {};
    nk_f32_to_bf16_serial(&value, &narrow);
    return narrow;
}

expected<rope_angles_t> rope_angles_t::make(u32_t positions, u32_t temporal, u32_t height, u32_t width,
                                            f32_t base) noexcept {
    using result_t = expected<rope_angles_t>;
    rope_angles_t angles;
    angles.positions_ = positions;
    angles.pairs_[0] = temporal / 2u, angles.pairs_[1] = height / 2u, angles.pairs_[2] = width / 2u;
    u32_t const denominators[3] = {temporal + height + width, height, width};
    std::size_t const pairs = angles.pairs_[0] + angles.pairs_[1] + angles.pairs_[2];
    if (angles.angles_.resize(static_cast<std::size_t>(positions) * pairs) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "angle grid allocation failed ({} positions)", positions);
    std::size_t offset = 0;
    for (u32_t axis = 0; axis != 3; ++axis)
        for (u32_t position = 0; position != positions; ++position)
            for (u32_t pair = 0; pair != angles.pairs_[axis]; ++pair)
                angles.angles_[offset++] = static_cast<f32_t>(position) *
                                           static_cast<f32_t>(
                                               std::pow(static_cast<f64_t>(base), -2.0 * pair / denominators[axis]));
    return result_t {std::move(angles)};
}

/** The angle of within-head pair @p pair at @p coordinate , each axis clamped into its grid. */
[[nodiscard]] static f32_t rope_angle_(rope_grids_t const &angles, u32_t pair, i16_t const *coordinate) noexcept {
    f32r2_view_t const grids[3] = {angles.temporal, angles.height, angles.width};
    u32_t axis = 0;
    for (; axis != 2 && pair >= grids[axis].extent(1); ++axis) pair -= static_cast<u32_t>(grids[axis].extent(1));
    std::size_t position = coordinate[axis] > 0 ? static_cast<std::size_t>(coordinate[axis]) : 0;
    if (position >= grids[axis].extent(0)) position = grids[axis].extent(0) - 1u;
    return grids[axis][position, pair];
}

void ui::gather_bytes(segment_layout_t const &layout, nk::tensor_view<token_id_t, 1> grid_tokens, bf16r2_view_t table,
                      bf16r2_span_t rows, fu::flat_pool_t &pool) noexcept {
    std::size_t const hidden = rows.extent(1);
    pool.for_n(layout.lengths.extent(0), [&](std::size_t segment, auto) noexcept {
        if (layout.modality[segment] != modality_t::text_k) return;
        for (u32_t cell = 0; cell != layout.lengths[segment]; ++cell) {
            std::size_t const token = grid_tokens[layout.grid_offsets[segment] + cell];
            std::size_t const row = layout.packed_offsets[segment] + cell;
            for (std::size_t channel = 0; channel != hidden; ++channel) rows[row, channel] = table[token, channel];
        }
    });
}

void ui::stage_patches(segment_layout_t const &layout, modality_t target, nk::tensor_view<pixel_t, 3> grid_pixels,
                       f32r1_view_t mean, f32r1_view_t inverse_std, bf16r2_span_t patches,
                       fu::flat_pool_t &pool) noexcept {
    std::size_t const spatial = grid_pixels.extent(1), channels = grid_pixels.extent(2);
    f32_t const inverse_byte_max = 1.0f / 255.0f;
    pool.for_n(layout.lengths.extent(0), [&](std::size_t segment, auto) noexcept {
        if (layout.modality[segment] != target) return;
        for (u32_t cell = 0; cell != layout.lengths[segment]; ++cell) {
            std::size_t const source = layout.grid_offsets[segment] + cell;
            std::size_t const row = layout.packed_offsets[segment] + cell;
            for (std::size_t position = 0; position != spatial; ++position)
                for (std::size_t channel = 0; channel != channels; ++channel) {
                    f32_t const byte = static_cast<f32_t>(grid_pixels[source, position, channel]);
                    f32_t const value = (byte * inverse_byte_max - mean[channel]) * inverse_std[channel];
                    patches[row, position * channels + channel] = narrow_(value);
                }
        }
    });
}

void ui::stage_frames(segment_layout_t const &layout, nk::tensor_view<f32_t, 2> grid_samples, bf16r2_span_t frames,
                      fu::flat_pool_t &pool) noexcept {
    std::size_t const samples = grid_samples.extent(1);
    pool.for_n(layout.lengths.extent(0), [&](std::size_t segment, auto) noexcept {
        if (layout.modality[segment] != modality_t::audio_k) return;
        for (u32_t cell = 0; cell != layout.lengths[segment]; ++cell) {
            std::size_t const source = layout.grid_offsets[segment] + cell;
            std::size_t const row = layout.packed_offsets[segment] + cell;
            for (std::size_t sample = 0; sample != samples; ++sample)
                frames[row, sample] = narrow_(grid_samples[source, sample]);
        }
    });
}

void ui::fill_coords(segment_layout_t const &layout, nk::tensor_view<u16_t, 2> image_shapes,
                     nk::tensor_view<u16_t, 2> video_shapes, u32_t tube_frames, u32_t frame_stride, i16r2_span_t coords,
                     fu::flat_pool_t &pool) noexcept {
    pool.for_n(layout.lengths.extent(0), [&](std::size_t segment, auto) noexcept {
        modality_t const modality = layout.modality[segment];
        for (u32_t cell = 0; cell != layout.lengths[segment]; ++cell) {
            u32_t temporal = 0, height = 0, width = 0;
            switch (modality) {
            case modality_t::text_k:
            case modality_t::audio_k: temporal = cell; break;
            case modality_t::image_k: {
                u32_t const columns = image_shapes[segment, 1];
                if (columns == 0u) continue;
                height = cell / columns, width = cell % columns;
                break;
            }
            case modality_t::video_k: {
                u32_t const columns = video_shapes[segment, 2];
                u32_t const groups_per_frame = static_cast<u32_t>(video_shapes[segment, 1]) * columns;
                if (groups_per_frame == 0u) continue;
                u32_t const planar = cell % groups_per_frame;
                temporal = cell / groups_per_frame * tube_frames * frame_stride;
                height = planar / columns, width = planar % columns;
                break;
            }
            }
            std::size_t const row = layout.packed_offsets[segment] + cell;
            coords[row, 0] = static_cast<i16_t>(temporal);
            coords[row, 1] = static_cast<i16_t>(height);
            coords[row, 2] = static_cast<i16_t>(width);
        }
    });
}

void ui::mrope(bf16r2_span_t values, u32_t role_offset, u32_t heads, u32_t head_width, i16r2_view_t coords,
               rope_grids_t angles, fu::flat_pool_t &pool) noexcept {
    u32_t const pairs = head_width / 2u;
    pool.for_slices(values.extent(0), [&](auto rows, auto) noexcept {
        for (std::size_t row : rows) {
            i16_t const coordinate[3] = {coords[row, 0], coords[row, 1], coords[row, 2]};
            for (u32_t pair = 0; pair != pairs; ++pair) {
                f32_t const angle = rope_angle_(angles, pair, coordinate);
                f32_t const sine = std::sin(angle), cosine = std::cos(angle);
                for (u32_t head = 0; head != heads; ++head) {
                    std::size_t const channel = role_offset + head * head_width + pair;
                    f32_t const real = widen_(values[row, channel]);
                    f32_t const imaginary = widen_(values[row, channel + pairs]);
                    values[row, channel] = narrow_(real * cosine - imaginary * sine);
                    values[row, channel + pairs] = narrow_(real * sine + imaginary * cosine);
                }
            }
        }
    });
}

/** The per-row quantization both input types share; @p widen turns one input scalar into Float32. */
template <typename view_type_, typename widen_type_>
static void quantize_rows_(view_type_ values, e4m3r2_span_t shadow, f32r1_span_t token_scales, widen_type_ widen,
                           fu::flat_pool_t &pool) noexcept {
    std::size_t const width = values.extent(1);
    pool.for_slices(values.extent(0), [&](auto rows, auto) noexcept {
        for (std::size_t row : rows) {
            f32_t largest = 0.0f;
            for (std::size_t channel = 0; channel != width; ++channel)
                largest = std::fmax(largest, std::fabs(widen(values[row, channel])));
            f32_t const scale = std::fmax(largest / e4m3_range_k, e4m3_descale_floor_k);
            f32_t const inverse = 1.0f / scale;
            token_scales[row] = scale;
            for (std::size_t channel = 0; channel != width; ++channel) {
                f32_t const scaled = widen(values[row, channel]) * inverse;
                nk_f32_to_e4m3_serial(&scaled, &shadow[row, channel]);
            }
        }
    });
}

void ui::quantize_tokens_e4m3(bf16r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                              fu::flat_pool_t &pool) noexcept {
    quantize_rows_(values, shadow, token_scales, widen_, pool);
}

void ui::quantize_tokens_e4m3(f32r2_view_t values, e4m3r2_span_t shadow, f32r1_span_t token_scales,
                              fu::flat_pool_t &pool) noexcept {
    quantize_rows_(values, shadow, token_scales, [](f32_t value) noexcept { return value; }, pool);
}

void ui::rescale_tokens(f32r2_view_t products, f32r1_view_t token_scales, f32_t weight_descale, bf16r2_view_t residual,
                        bf16r2_span_t values, fu::flat_pool_t &pool) noexcept {
    std::size_t const width = values.extent(1);
    pool.for_slices(values.extent(0), [&](auto rows, auto) noexcept {
        for (std::size_t row : rows)
            for (std::size_t channel = 0; channel != width; ++channel) {
                f32_t value = products[row, channel] * weight_descale;
                value *= token_scales[row];
                if (!residual.empty()) value += widen_(residual[row, channel]);
                values[row, channel] = narrow_(value);
            }
    });
}
