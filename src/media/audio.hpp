/**
 *  @brief Host audio decoding: any container libav reads, to one mono Float32 clip at the model's rate.
 *  @file src/media/audio.hpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  No GPU audio codec exists, so every residency decodes here, on its executor's pool, and uploads the waveform.
 *  The output is the raw signal: the spectral frontend is a learnable layer of the encoder, not a decode step.
 *  Every clip fills its destination exactly, truncated when long and padded with silence when short.
 */
#pragma once

#include <span> // `std::span`

#include "core/types.hpp" // `expected_t`, `status_t`, the scalar aliases

namespace unum::uform::media {

namespace uc = unum::uform::core;

using uc::b8_t;
using uc::expected_t;
using uc::f32_t;
using uc::f32r1_span_t;
using uc::status_t;
using uc::u32_t;

/**
 *  @brief Decode one encoded clip and resample it to mono Float32 at @p sample_rate , filling all of @p out .
 *
 *  Demuxes in memory through a custom AVIO context, decodes the first audio stream, and downmixes and resamples
 *  through libswresample, draining its delay line so codecs with priming samples lose none. A failed clip leaves
 *  @p out silent, never half-written.
 *
 *  @param[in] encoded The clip's container bytes.
 *  @param[in] sample_rate The mono rate to resample to.
 *  @param[out] out The clip's samples, exactly @c out.extent(0) of them.
 *  @retval status_t::invalid_input_k An empty clip or one without an audio stream.
 *  @retval status_t::shape_unsupported_k No decoder for the stream's codec.
 *  @retval status_t::io_error_k The container, decoder or resampler failed.
 */
[[nodiscard]] expected_t decode_audio(std::span<b8_t const> encoded, u32_t sample_rate, f32r1_span_t out) noexcept;

/**
 *  @brief Resample an already decoded mono signal from @p from_rate to @p to_rate , filling all of @p out .
 *
 *  The same libswresample conversion @ref decode_audio ends with, for callers that decoded elsewhere, like a
 *  browser's WebAudio.
 *
 *  @param[in] samples The mono signal.
 *  @param[in] from_rate The signal's rate.
 *  @param[in] to_rate The rate to resample to.
 *  @param[out] out The resampled samples, exactly @c out.extent(0) of them.
 *  @retval status_t::invalid_input_k A zero rate.
 *  @retval status_t::io_error_k The resampler failed.
 */
[[nodiscard]] expected_t resample_mono(std::span<f32_t const> samples, u32_t from_rate, u32_t to_rate,
                                       f32r1_span_t out) noexcept;

} // namespace unum::uform::media
