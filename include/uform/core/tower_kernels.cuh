/**
 *  @file include/uform/core/tower_kernels.cuh
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The element-wise bodies of the encoder tower's entries, rotary and E4M3 scaling steps,
 *      shared by the NVIDIA training kernels and every vendor's inference kernels.
 *
 *  Each body is a @c constexpr template over its views, so a host compiler reads this header too, and device
 *  compilers call it from any kernel through relaxed @c constexpr . A kernel passes its thread's place in the
 *  stripe, and a @p scalars object carrying its vendor's conversions:
 *
 *  - @c widen(x) Float32 from a stored scalar, BFloat16 or Float32;
 *  - @c narrow(x) BFloat16 from Float32, rounding to nearest even;
 *  - @c narrow_e4m3(x) E4M3 from Float32, saturating;
 *  - @c sincos(x,&sine,&cosine) .
 *
 *  Bodies read a source row and write a target row separately: training runs on the batch grid, where the two
 *  coincide, and inference on packed rows, where every segment's live cells sit back to back.
 */
#pragma once

#include <cmath>   // `std::fmax`, `std::fabs`
#include <cstddef> // `std::size_t`

#include "uform/core/multimodal_batch.hpp" // `modality_t`
#include "uform/core/types.hpp"            // the scalar aliases and the rank-tagged views

namespace unum::uform::core {

/** E4M3's largest normal magnitude — the quantisation range max_abs maps onto. */
inline constexpr f32_t e4m3_range_k = 240.0f;

/** The descale floor keeping a zero tensor representable — @c max_abs/240 clamps to this. */
inline constexpr f32_t e4m3_descale_floor_k = 1.0e-12f;

/** One text segment's rows @c target[target_first+k,:]=table[tokens[source_first+k],:] , channels striped. */
template <typename tokens_type_, typename table_type_, typename rows_type_>
constexpr void gather_bytes_segment(tokens_type_ tokens, table_type_ table, std::size_t source_first,
                                    std::size_t target_first, u32_t length, rows_type_ target, u32_t thread,
                                    u32_t threads) noexcept {
    u32_t const hidden = static_cast<u32_t>(target.extent(1));
    for (u32_t cell = 0; cell != length; ++cell) {
        std::size_t const token = static_cast<std::size_t>(tokens[source_first + cell]);
        for (u32_t channel = thread; channel < hidden; channel += threads)
            target[target_first + cell, channel] = table[token, channel];
    }
}

/** One pixel segment's cells @b [length,spatial,channels] as @c (byte/255−mean)·inverse_std , the
 *  @c length·spatial positions striped. */
template <typename pixels_type_, typename statistics_type_, typename patches_type_, typename scalars_type_>
constexpr void stage_patches_segment(pixels_type_ pixels, std::size_t source_first, std::size_t target_first,
                                     u32_t length, statistics_type_ mean, statistics_type_ inverse_std,
                                     patches_type_ patches, u32_t thread, u32_t threads,
                                     scalars_type_ const &scalars) noexcept {
    u32_t const spatial = static_cast<u32_t>(pixels.extent(1));
    u32_t const channels = static_cast<u32_t>(pixels.extent(2));
    std::size_t const cells = static_cast<std::size_t>(length) * spatial;
    f32_t const inverse_byte_max = 1.0f / 255.0f;
    for (std::size_t cell = thread; cell < cells; cell += threads) {
        u32_t const position = static_cast<u32_t>(cell % spatial);
        std::size_t const row = cell / spatial;
        for (u32_t channel = 0; channel != channels; ++channel) {
            f32_t const value = (static_cast<f32_t>(pixels[source_first + row, position, channel]) * inverse_byte_max -
                                 mean[channel]) *
                                inverse_std[channel];
            patches[target_first + row, position, channel] = scalars.narrow(value);
        }
    }
}

/** One audio segment's raw frames @b [length,samples] narrowed, the @c length·samples cells striped. */
template <typename samples_type_, typename frames_type_, typename scalars_type_>
constexpr void stage_frames_segment(samples_type_ samples, std::size_t source_first, std::size_t target_first,
                                    u32_t length, frames_type_ frames, u32_t thread, u32_t threads,
                                    scalars_type_ const &scalars) noexcept {
    u32_t const width = static_cast<u32_t>(samples.extent(1));
    std::size_t const cells = static_cast<std::size_t>(length) * width;
    for (std::size_t cell = thread; cell < cells; cell += threads) {
        u32_t const sample = static_cast<u32_t>(cell % width);
        std::size_t const row = cell / width;
        frames[target_first + row, sample] = scalars.narrow(samples[source_first + row, sample]);
    }
}

/** The angle of within-head pair @p pair for row @p token , from @p angles ' temporal, height and width grids
 *  @b [positions,pairs] , temporal pairs first. */
template <typename coords_type_, typename angles_type_>
[[nodiscard]] constexpr f32_t rope_angle(u32_t pair, coords_type_ coords, std::size_t token,
                                         angles_type_ const &angles) noexcept {
    u32_t const temporal_pairs = static_cast<u32_t>(angles.temporal.extent(1));
    u32_t const height_pairs = static_cast<u32_t>(angles.height.extent(1));
    if (pair < temporal_pairs) return angles.temporal[static_cast<std::size_t>(coords[token, 0]), pair];
    if (pair < temporal_pairs + height_pairs)
        return angles.height[static_cast<std::size_t>(coords[token, 1]), pair - temporal_pairs];
    return angles.width[static_cast<std::size_t>(coords[token, 2]), pair - temporal_pairs - height_pairs];
}

/** Rotate the channel pairs @c [first,…) every @p stride of the @b [tokens,head_width/2] grid in
 *  place: each pair's sine and cosine, of @p angle_sign times its angle, turn the NeoX split-half
 *  pair @c (k,k+head_width/2) of all @p heads heads from @p role_offset ; a sign of @c -1 is the
 *  inverse rotation. */
template <typename values_type_, typename coords_type_, typename angles_type_, typename scalars_type_>
constexpr void rope_rotate(values_type_ values, angles_type_ const &angles, coords_type_ coords, u32_t heads,
                           u32_t head_width, u32_t role_offset, f32_t angle_sign, std::size_t first, std::size_t stride,
                           scalars_type_ const &scalars) noexcept {
    u32_t const pairs_per_head = head_width >> 1u;
    std::size_t const total_pairs = static_cast<std::size_t>(pairs_per_head) * values.extent(0);
    for (std::size_t pair_index = first; pair_index < total_pairs; pair_index += stride) {
        std::size_t const token = pair_index / pairs_per_head;
        u32_t const pair = static_cast<u32_t>(pair_index % pairs_per_head);
        f32_t sine = 0, cosine = 0;
        scalars.sincos(angle_sign * rope_angle(pair, coords, token, angles), &sine, &cosine);
        for (u32_t head = 0; head != heads; ++head) {
            u32_t const channel = role_offset + head * head_width + pair;
            f32_t const real = scalars.widen(values[token, channel]);
            f32_t const imaginary = scalars.widen(values[token, channel + pairs_per_head]);
            values[token, channel] = scalars.narrow(real * cosine - imaginary * sine);
            values[token, channel + pairs_per_head] = scalars.narrow(real * sine + imaginary * cosine);
        }
    }
}

/** This thread's share of row @p token 's largest magnitude, over its striped channels. */
template <typename values_type_, typename scalars_type_>
[[nodiscard]] constexpr f32_t token_largest(values_type_ values, std::size_t token, u32_t thread, u32_t threads,
                                            scalars_type_ const &scalars) noexcept {
    u32_t const width = static_cast<u32_t>(values.extent(1));
    f32_t largest = 0.0f;
    for (u32_t channel = thread; channel < width; channel += threads)
        largest = std::fmax(largest, std::fabs(scalars.widen(values[token, channel])));
    return largest;
}

/** Quantize row @p token by its own @c max(largest/240,floor) into @p shadow , channels striped, and return
 *  that scale. */
template <typename values_type_, typename shadow_type_, typename scalars_type_>
constexpr f32_t quantize_token(values_type_ values, shadow_type_ shadow, std::size_t token, f32_t largest, u32_t thread,
                               u32_t threads, scalars_type_ const &scalars) noexcept {
    u32_t const width = static_cast<u32_t>(values.extent(1));
    f32_t const scale = std::fmax(largest / e4m3_range_k, e4m3_descale_floor_k);
    f32_t const inverse = 1.0f / scale;
    for (u32_t channel = thread; channel < width; channel += threads)
        shadow[token, channel] = scalars.narrow_e4m3(scalars.widen(values[token, channel]) * inverse);
    return scale;
}

/** The cells @c [first,…) every @p stride of @c values=products·descale·token_scales+residual , an empty
 *  @p token_scales or @p residual dropping its term. */
template <typename products_type_, typename scales_type_, typename residual_type_, typename values_type_,
          typename scalars_type_>
constexpr void rescale_tokens_cells(products_type_ products, scales_type_ token_scales, f32_t descale,
                                    residual_type_ residual, values_type_ values, std::size_t first, std::size_t stride,
                                    scalars_type_ const &scalars) noexcept {
    u32_t const width = static_cast<u32_t>(values.extent(1));
    std::size_t const total = values.extent(0) * width;
    for (std::size_t cell = first; cell < total; cell += stride) {
        std::size_t const token = cell / width;
        u32_t const channel = static_cast<u32_t>(cell % width);
        f32_t value = products[token, channel] * descale;
        if (!token_scales.empty()) value *= token_scales[token];
        if (!residual.empty()) value += scalars.widen(residual[token, channel]);
        values[token, channel] = scalars.narrow(value);
    }
}

/** The packed rows' layout, serially: @c packed_offsets prefix sums of @p lengths , and @c pool_offsets[s]=s , both
 *  @b [segments+1] . */
template <typename lengths_type_, typename offsets_type_>
constexpr void fill_packed_layout(lengths_type_ lengths, offsets_type_ packed_offsets,
                                  offsets_type_ pool_offsets) noexcept {
    std::size_t const segments = lengths.extent(0);
    u32_t offset = 0;
    for (std::size_t segment = 0; segment != segments; ++segment) {
        packed_offsets[segment] = offset, pool_offsets[segment] = static_cast<u32_t>(segment);
        offset += lengths[segment];
    }
    packed_offsets[segments] = offset, pool_offsets[segments] = static_cast<u32_t>(segments);
}

/** One segment's packed coordinates @c (t,h,w) , cells striped: text and audio @c (k,0,0) , image
 *  @c (0,k/cols,k%cols) , video @c (group·tube_frames·frame_stride,r,c) , each clamped below @p positions . */
template <typename shapes_type_, typename coords_type_>
constexpr void fill_coords_segment(modality_t modality, std::size_t segment, std::size_t target_first, u32_t length,
                                   shapes_type_ image_shapes, shapes_type_ video_shapes, u32_t tube_frames,
                                   u32_t frame_stride, u32_t positions, coords_type_ coords, u32_t thread,
                                   u32_t threads) noexcept {
    u32_t const last = positions - 1u;
    for (u32_t cell = thread; cell < length; cell += threads) {
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
        std::size_t const row = target_first + cell;
        coords[row, 0] = static_cast<i16_t>(temporal < last ? temporal : last);
        coords[row, 1] = static_cast<i16_t>(height < last ? height : last);
        coords[row, 2] = static_cast<i16_t>(width < last ? width : last);
    }
}

} // namespace unum::uform::core
