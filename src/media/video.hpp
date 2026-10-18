/**
 *  @brief Host video ingest: demux a container to its elementary stream, and decode it in software into a
 *      segment's tube cells.
 *  @file src/media/video.hpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  @ref demux_video serves every residency: the NVIDIA loader feeds its Annex-B bytes to NVDEC, while the RAM and
 *  AMD executors hand the same @ref demuxed_t to @ref decode_video_frames , a libavcodec decode that scales each
 *  kept frame to the planned grid with libswscale.
 *
 *  Both paths convert BT.601 limited-range YCbCr to full-range RGB, whatever the stream signals. They differ in
 *  the details: NVDEC's scaler resizes NV12 and the conversion kernel reads one chroma sample per 2×2 block, while
 *  libswscale scales bilinearly from the decoder's own format, interpolates chroma, and rounds in fixed point, so
 *  pixels agree to within a few levels rather than exactly.
 */
#pragma once

#include <span> // `std::span`

#include "core/types.hpp"            // `expected`, `status_t`, the scalar aliases
#include "core/array.hpp"            // `array` — the demuxed stream
#include "core/multimodal_batch.hpp" // `video_batch_shape_t`
#include "media/planning.hpp"        // `image_extent_t`, `video_grid_t`

namespace unum::uform::media {

using uc::array;
using uc::b8_t;
using uc::expected;
using uc::expected_t;
using uc::pixel_t;
using uc::status_t;
using uc::u8_t;

/** The channels every decoded video pixel carries — interleaved RGB. */
inline constexpr u32_t video_channels_k = 3u;

/** A segment's tube cells @b [temporal_groups,patch_rows·patch_columns,tube_frames,patch_side,patch_side,3] , which
 *  sit contiguously in a batch row in exactly this order. */
using pixel_tubes_span_t = nk::tensor_span<pixel_t, 6>;

/** The elementary-stream codecs a container can carry; contiguous from @c h264_k so a cast indexes a per-codec
 *  table. */
enum class video_codec_t : u8_t { h264_k = 0, h265_k, vp9_k, av1_k, mpeg2_k, mpeg4_k, vc1_k };

/** Count of @ref video_codec_t enumerators. */
inline constexpr std::size_t video_codec_count_k = 7;

/** One demuxed clip: its elementary stream, where each packet starts in it, its codec, and its native extent and
 *  frame count. H.264 and H.265 arrive as Annex-B; the others keep their packets as the container stored them. */
struct demuxed_t {

    /** The elementary stream, every packet back to back. */
    array<b8_t> annex_b {};

    /** @b [packets+1] packet boundaries into @ref annex_b ; a raw Annex-B input is one packet. */
    array<u64_t> packet_offsets {};

    /** The elementary-stream codec. */
    video_codec_t codec = video_codec_t::h264_k;

    /** The native frame extent, zero when the input carried none. */
    image_extent_t extent {};

    /** The native frame count, indexed or estimated from the duration, zero when unknown. */
    u32_t frames = 0;
};

/**
 *  @brief Demux one container's bytes in memory, through a custom AVIO context, into its video stream's packets,
 *      passing H.264 and H.265 through the @c *_mp4toannexb bitstream filters.
 *
 *  A raw Annex-B input passes through as one packet with its codec sniffed from the first NAL header.
 *
 *  @param[in] container The clip's container or raw elementary-stream bytes.
 *  @retval status_t::invalid_input_k No video stream, an unsupported codec, or a stream that yielded no bytes.
 *  @retval status_t::bad_alloc_k A libav or stream buffer allocation failed.
 */
[[nodiscard]] expected<demuxed_t> demux_video(std::span<b8_t const> container) noexcept;

/**
 *  @brief Decode @p clip in software and scatter its sampled frames into a segment's tube cells.
 *
 *  Displayed frame @c (tg·tube_frames+t)·frame_stride becomes tube frame @c t of temporal group @c tg , as the
 *  NVIDIA tile kernel samples, each scaled to @c (patch_columns·P,patch_rows·P) and converted to RGB. Frames a short
 *  clip lacks stay zero.
 *
 *  @param[in] clip The demuxed clip.
 *  @param[in] grid The segment's factorization, from @ref video_tube_grid .
 *  @param[in] shape The video modality's geometry: patch side, tube depth and frame stride.
 *  @param[out] out The segment's cells, shaped as @ref pixel_tubes_span_t says.
 *  @retval status_t::shape_unsupported_k @p out disagrees with @p grid and @p shape .
 *  @retval status_t::invalid_input_k The stream does not decode, or decodes no frame.
 */
[[nodiscard]] expected_t decode_video_frames(demuxed_t const &clip, video_grid_t grid, video_batch_shape_t const &shape,
                                             pixel_tubes_span_t out) noexcept;

} // namespace unum::uform::media
