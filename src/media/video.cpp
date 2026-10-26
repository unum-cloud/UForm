/**
 *  @brief Host video ingest — see the header for the contract.
 *  @file src/media/video.cpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  Every libav handle is checked and folded into @ref expected , so no libav failure leaves this file any other way.
 */
#include <algorithm> // `std::min`, `std::max`

extern "C" {
#include <libavcodec/avcodec.h>   // vendor-boundary: `avcodec_*`, `av_parser_*`, `AVPacket`, `AVFrame`
#include <libavcodec/bsf.h>       // vendor-boundary: `AVBSFContext`, `av_bsf_*`
#include <libavformat/avformat.h> // vendor-boundary: `AVFormatContext`, `av_read_frame`
#include <libavutil/mem.h>        // vendor-boundary: `av_malloc`, `av_free`
#include <libswscale/swscale.h>   // vendor-boundary: `sws_*`, `SwsContext`
}

#include "uform/media/video.hpp"

namespace um = unum::uform::media;
using namespace um;
using uc::i64_t;

/** Working buffer size of the custom @c AVIOContext ; FFmpeg's own 32 KB default. */
inline constexpr std::size_t video_avio_buffer_bytes_k = 32u * 1024u;

/** Backing state for the in-memory @c AVIOContext read and seek callbacks. */
struct video_memory_reader_t {

    /** The clip's container bytes — borrowed, not owned. */
    b8_t const *data = nullptr;

    /** Total size of @c data in bytes — the seek's end bound. */
    std::size_t size = 0;

    /** The read cursor, advanced by every read and moved by every seek. */
    std::size_t position = 0;
};

/** The @c uint8_t pointer libavcodec's parser reads elementary-stream @p bytes through. */
[[nodiscard]] static u8_t const *av_bytes_(b8_t const *bytes) noexcept { return reinterpret_cast<u8_t const *>(bytes); }

/** @c AVIOContext read callback: copy up to @p buffer_size bytes, @c AVERROR_EOF at the end. */
static int video_memory_read_(void *opaque, u8_t *buffer, int buffer_size) noexcept {
    auto *reader = static_cast<video_memory_reader_t *>(opaque);
    std::size_t const remaining = reader->size - reader->position;
    std::size_t const to_read = std::min<std::size_t>(static_cast<std::size_t>(buffer_size), remaining);
    if (to_read == 0) return AVERROR_EOF;
    for (std::size_t index = 0; index != to_read; ++index)
        buffer[index] = std::to_integer<u8_t>(reader->data[reader->position + index]);
    reader->position += to_read;
    return static_cast<int>(to_read);
}

/** @c AVIOContext seek callback: honors @c SEEK_SET/CUR/END and the @c AVSEEK_SIZE query, @c -1 out of range. */
static i64_t video_memory_seek_(void *opaque, i64_t offset, int whence) noexcept {
    auto *reader = static_cast<video_memory_reader_t *>(opaque);
    if (whence == AVSEEK_SIZE) return static_cast<i64_t>(reader->size);
    i64_t position = 0;
    if (whence == SEEK_SET) position = offset;
    else if (whence == SEEK_CUR) position = static_cast<i64_t>(reader->position) + offset;
    else if (whence == SEEK_END) position = static_cast<i64_t>(reader->size) + offset;
    else return -1;
    if (position < 0 || position > static_cast<i64_t>(reader->size)) return -1;
    reader->position = static_cast<std::size_t>(position);
    return position;
}

/** Whether @p bytes opens with a 3- or 4-byte Annex-B NAL start code. */
[[nodiscard]] static bool looks_like_annex_b_(std::span<b8_t const> bytes) noexcept {
    if (bytes.size() < 4u) return false;
    auto byte_at = [&](std::size_t index) { return std::to_integer<u8_t>(bytes[index]); };
    if (byte_at(0) == 0u && byte_at(1) == 0u && byte_at(2) == 0u && byte_at(3) == 1u) return true;
    return byte_at(0) == 0u && byte_at(1) == 0u && byte_at(2) == 1u;
}

/** Best-effort H.264 vs H.265 sniff from the first Annex-B NAL header. */
[[nodiscard]] static video_codec_t sniff_annex_b_codec_(std::span<b8_t const> bytes) noexcept {
    std::size_t index = 0;
    while (index + 3u < bytes.size()) {
        auto byte_at = [&](std::size_t offset) { return std::to_integer<u8_t>(bytes[index + offset]); };
        if (byte_at(0) == 0u && byte_at(1) == 0u && byte_at(2) == 1u) {
            index += 3u;
            break;
        }
        if (byte_at(0) == 0u && byte_at(1) == 0u && byte_at(2) == 0u && byte_at(3) == 1u) {
            index += 4u;
            break;
        }
        ++index;
    }
    if (index >= bytes.size()) return video_codec_t::h264_k;
    u8_t const hevc_type = (std::to_integer<u8_t>(bytes[index]) >> 1u) & 0x3Fu;
    return hevc_type >= 32u && hevc_type <= 34u ? video_codec_t::h265_k : video_codec_t::h264_k;
}

/** Append @p packet to @p clip 's stream and record where the next one starts. */
[[nodiscard]] static status_t append_packet_(demuxed_t &clip, std::span<b8_t const> packet) noexcept {
    std::size_t const previous = clip.annex_b.size();
    if (st::failed(clip.annex_b.resize(st::assume_uninitialized, previous + packet.size())))
        return status_t::bad_alloc_k;
    std::copy(packet.begin(), packet.end(), clip.annex_b.data() + previous);
    return st::failed(clip.packet_offsets.push_back(static_cast<u64_t>(clip.annex_b.size()))) ? status_t::bad_alloc_k
                                                                                              : status_t::success_k;
}

/** The payload bytes of the libav @p packet . */
[[nodiscard]] static std::span<b8_t const> packet_bytes_(AVPacket const *packet) noexcept {
    return std::as_bytes(std::span<u8_t const> {packet->data, static_cast<std::size_t>(packet->size)});
}

/** Map an @c AVCodecID onto @ref video_codec_t ; @c false leaves @p codec untouched. */
[[nodiscard]] static bool from_av_codec_(AVCodecID av_codec_id, video_codec_t &codec) noexcept {
    switch (av_codec_id) {
    case AV_CODEC_ID_H264: codec = video_codec_t::h264_k; return true;
    case AV_CODEC_ID_HEVC: codec = video_codec_t::h265_k; return true;
    case AV_CODEC_ID_VP9: codec = video_codec_t::vp9_k; return true;
    case AV_CODEC_ID_AV1: codec = video_codec_t::av1_k; return true;
    case AV_CODEC_ID_MPEG2VIDEO: codec = video_codec_t::mpeg2_k; return true;
    case AV_CODEC_ID_MPEG4: codec = video_codec_t::mpeg4_k; return true;
    case AV_CODEC_ID_VC1: codec = video_codec_t::vc1_k; return true;
    default: return false;
    }
}

/** The @c AVCodecID libavcodec decodes @p codec with. */
[[nodiscard]] static AVCodecID to_av_codec_(video_codec_t codec) noexcept {
    switch (codec) {
    case video_codec_t::h264_k: return AV_CODEC_ID_H264;
    case video_codec_t::h265_k: return AV_CODEC_ID_HEVC;
    case video_codec_t::vp9_k: return AV_CODEC_ID_VP9;
    case video_codec_t::av1_k: return AV_CODEC_ID_AV1;
    case video_codec_t::mpeg2_k: return AV_CODEC_ID_MPEG2VIDEO;
    case video_codec_t::mpeg4_k: return AV_CODEC_ID_MPEG4;
    case video_codec_t::vc1_k: return AV_CODEC_ID_VC1;
    }
    return AV_CODEC_ID_NONE;
}

/** The bitstream filter converting a container-wrapped codec to Annex-B; @c nullptr passes packets through. */
[[nodiscard]] static char const *bitstream_filter_name_(video_codec_t codec) noexcept {
    if (codec == video_codec_t::h264_k) return "h264_mp4toannexb";
    if (codec == video_codec_t::h265_k) return "hevc_mp4toannexb";
    // MP4 keeps MPEG-4 Part 2's VOL header in extradata, which NVDEC's parser needs in the stream.
    if (codec == video_codec_t::mpeg4_k) return "dump_extra";
    return nullptr;
}

expected<demuxed_t> um::demux_video(std::span<b8_t const> container) noexcept {
    using result_t = expected<demuxed_t>;
    demuxed_t clip;
    if (container.empty()) return result_t::fail(status_t::invalid_input_k, "empty video container");
    if (st::failed(clip.packet_offsets.push_back(0u)))
        return result_t::fail(status_t::bad_alloc_k, "packet table allocation failed");
    if (looks_like_annex_b_(container)) {
        // A raw elementary stream carries no dimensions until decoded.
        if (append_packet_(clip, container) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "elementary stream copy failed ({} bytes)", container.size());
        clip.codec = sniff_annex_b_codec_(container);
        return clip;
    }

    video_memory_reader_t reader {container.data(), container.size(), 0u};
    // Every libav handle is declared up front and freed exactly once by `release_libav`. `avformat_open_input` frees
    // and nulls `format` itself on failure while leaving our custom-IO `avio` ours to free (AVFMT_FLAG_CUSTOM_IO).
    u8_t *avio_buffer = static_cast<u8_t *>(av_malloc(video_avio_buffer_bytes_k));
    AVIOContext *avio = nullptr;
    AVFormatContext *format = nullptr;
    AVBSFContext *bitstream_filter = nullptr;
    AVPacket *raw_packet = nullptr;
    AVPacket *filtered_packet = nullptr;
    auto release_libav = [&]() noexcept {
        if (raw_packet != nullptr) av_packet_free(&raw_packet);
        if (filtered_packet != nullptr) av_packet_free(&filtered_packet);
        if (bitstream_filter != nullptr) av_bsf_free(&bitstream_filter);
        if (format != nullptr) avformat_close_input(&format);
        if (avio != nullptr) {
            if (avio->buffer != nullptr) av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
        else if (avio_buffer != nullptr) av_free(avio_buffer);
    };
    auto fail = [&](status_t status, char const *reason) noexcept {
        release_libav();
        return result_t::fail(status, "video demux: {}", reason);
    };
    if (avio_buffer == nullptr) return fail(status_t::bad_alloc_k, "AVIO buffer allocation failed");
    avio = avio_alloc_context(avio_buffer, static_cast<int>(video_avio_buffer_bytes_k), 0, &reader, &video_memory_read_,
                              nullptr, &video_memory_seek_);
    if (avio == nullptr) return fail(status_t::bad_alloc_k, "AVIO context allocation failed");
    format = avformat_alloc_context();
    if (format == nullptr) return fail(status_t::bad_alloc_k, "format context allocation failed");
    format->pb = avio;
    format->flags |= AVFMT_FLAG_CUSTOM_IO;
    if (avformat_open_input(&format, nullptr, nullptr, nullptr) < 0)
        return fail(status_t::invalid_input_k, "unrecognized container");
    if (avformat_find_stream_info(format, nullptr) < 0)
        return fail(status_t::invalid_input_k, "unreadable stream info");
    int video_stream_index = -1;
    for (unsigned stream_index = 0; stream_index != format->nb_streams; ++stream_index)
        if (format->streams[stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = static_cast<int>(stream_index);
            break;
        }
    if (video_stream_index < 0) return fail(status_t::invalid_input_k, "no video stream");
    AVStream *video_stream = format->streams[video_stream_index];
    if (!from_av_codec_(video_stream->codecpar->codec_id, clip.codec))
        return fail(status_t::invalid_input_k, "unsupported video codec");
    clip.extent.width = static_cast<u32_t>(std::max(video_stream->codecpar->width, 0));
    clip.extent.height = static_cast<u32_t>(std::max(video_stream->codecpar->height, 0));
    // Many mp4 / webm containers carry no frame index (`nb_frames == 0`); estimate the count from duration × average
    // frame rate, in integers since `duration` is in AV_TIME_BASE units, so sampled frames still spread over the clip.
    if (video_stream->nb_frames > 0) clip.frames = static_cast<u32_t>(video_stream->nb_frames);
    else {
        AVRational const rate = video_stream->avg_frame_rate;
        if (rate.num > 0 && rate.den > 0 && format->duration > 0)
            clip.frames = static_cast<u32_t>(static_cast<i64_t>(format->duration) * rate.num /
                                             (static_cast<i64_t>(rate.den) * AV_TIME_BASE));
    }

    if (char const *filter_name = bitstream_filter_name_(clip.codec); filter_name != nullptr) {
        AVBitStreamFilter const *descriptor = av_bsf_get_by_name(filter_name);
        if (descriptor == nullptr || av_bsf_alloc(descriptor, &bitstream_filter) < 0 ||
            avcodec_parameters_copy(bitstream_filter->par_in, video_stream->codecpar) < 0)
            return fail(status_t::bad_alloc_k, "Annex-B filter setup failed");
        bitstream_filter->time_base_in = video_stream->time_base;
        if (av_bsf_init(bitstream_filter) < 0) return fail(status_t::invalid_input_k, "Annex-B filter rejected");
    }

    raw_packet = av_packet_alloc();
    filtered_packet = av_packet_alloc();
    status_t status = raw_packet != nullptr && filtered_packet != nullptr ? status_t::success_k : status_t::bad_alloc_k;
    auto drain_filter = [&]() noexcept {
        while (status == status_t::success_k && av_bsf_receive_packet(bitstream_filter, filtered_packet) == 0) {
            status = append_packet_(clip, packet_bytes_(filtered_packet));
            av_packet_unref(filtered_packet);
        }
    };
    while (status == status_t::success_k && av_read_frame(format, raw_packet) >= 0) {
        if (raw_packet->stream_index != video_stream_index) {
            av_packet_unref(raw_packet);
            continue;
        }
        if (bitstream_filter == nullptr) status = append_packet_(clip, packet_bytes_(raw_packet));
        else if (av_bsf_send_packet(bitstream_filter, raw_packet) < 0) status = status_t::invalid_input_k;
        else drain_filter();
        av_packet_unref(raw_packet);
    }
    if (status == status_t::success_k && bitstream_filter != nullptr) {
        av_bsf_send_packet(bitstream_filter, nullptr);
        drain_filter();
    }
    if (status != status_t::success_k) return fail(status, "packet extraction failed");
    if (clip.annex_b.empty()) return fail(status_t::invalid_input_k, "the video stream holds no packets");
    release_libav();
    return clip;
}

/** One software decode's libav chain, owned as a unit so its destructor is the single unwind on every return. */
struct video_decoder_ {

    /** The decoder context, single-threaded since executors decode clips in parallel. */
    AVCodecContext *context = nullptr;

    /** The parser that splits the elementary stream into the decoder's packets. */
    AVCodecParserContext *parser = nullptr;

    /** The reusable packet the parser fills. */
    AVPacket *packet = nullptr;

    /** The reusable frame the decoder emits into. */
    AVFrame *frame = nullptr;

    /** The scaler to RGB24 at the planned grid size, rebuilt only when the decoded format changes. */
    SwsContext *scaler = nullptr;

    ~video_decoder_() noexcept {
        if (scaler != nullptr) sws_freeContext(scaler);
        if (frame != nullptr) av_frame_free(&frame);
        if (packet != nullptr) av_packet_free(&packet);
        if (parser != nullptr) av_parser_close(parser);
        if (context != nullptr) avcodec_free_context(&context);
    }
};

/** The sampling state threaded through the receive loop: which displayed frame is next, and the RGB staging of
 *  the one being scattered. */
struct frame_sampler_ {

    /** Displayed frames seen so far — drives the @c frame_stride subsampling. */
    std::size_t displayed = 0;

    /** Frames scattered so far, also the compact index of the next one. */
    std::size_t kept = 0;

    /** The compact frames the segment holds, @c temporal_groups·tube_frames . */
    std::size_t wanted = 0;

    /** Keep every @c stride -th displayed frame. */
    std::size_t stride = 1;

    /** The planned frame width, @c patch_columns·P . */
    u32_t width = 0;

    /** The planned frame height, @c patch_rows·P . */
    u32_t height = 0;

    /** One planned frame as packed @b [height,width,3] RGB. */
    st::basic_vector<pixel_t> rgb {};

    /** The first failure, or @c success_k while decoding continues. */
    status_t status = status_t::success_k;
};

/** Convert @p frame to RGB at the planned size and scatter it as compact frame @c sampler.kept into @p out . */
static void keep_frame_(video_decoder_ &decoder, frame_sampler_ &sampler, AVFrame const *frame,
                        pixel_tubes_span_t out) noexcept {
    SwsContext *const scaler = sws_getCachedContext(
        decoder.scaler, frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
        static_cast<int>(sampler.width), static_cast<int>(sampler.height), AV_PIX_FMT_RGB24,
        SWS_BILINEAR | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT, nullptr, nullptr, nullptr);
    if (scaler == nullptr) {
        sampler.status = status_t::invalid_input_k;
        return;
    }
    decoder.scaler = scaler;
    sws_setColorspaceDetails(scaler, sws_getCoefficients(SWS_CS_ITU601), 0, sws_getCoefficients(SWS_CS_ITU601), 1, 0,
                             1 << 16, 1 << 16); // BT.601 limited to full range whatever the stream signals, as on GPUs
    u8_t *const planes[4] = {sampler.rgb.data(), nullptr, nullptr, nullptr};
    int const strides[4] = {static_cast<int>(sampler.width * video_channels_k), 0, 0, 0};
    if (sws_scale(scaler, frame->data, frame->linesize, 0, frame->height, planes, strides) !=
        static_cast<int>(sampler.height)) {
        sampler.status = status_t::invalid_input_k;
        return;
    }
    std::size_t const tube_frames = out.extent(2), patch_side = out.extent(3);
    std::size_t const patch_columns = sampler.width / patch_side, cells = out.extent(1);
    std::size_t const temporal_group = sampler.kept / tube_frames, tube_frame = sampler.kept % tube_frames;
    nk::tensor_view<pixel_t, 5> const patches {
        sampler.rgb.data(),
        {sampler.height / patch_side, patch_side, patch_columns, patch_side, std::size_t {video_channels_k}}};
    for (std::size_t cell = 0; cell != cells; ++cell) {
        std::size_t const patch_row = cell / patch_columns, patch_column = cell % patch_columns;
        for (std::size_t y = 0; y != patch_side; ++y)
            for (std::size_t x = 0; x != patch_side; ++x)
                for (std::size_t channel = 0; channel != video_channels_k; ++channel)
                    out[temporal_group, cell, tube_frame, y, x, channel] =
                        patches[patch_row, y, patch_column, x, channel];
    }
    ++sampler.kept;
}

/** Pull every frame the decoder holds, keeping the sampled ones, until it asks for input or the segment is full. */
static void receive_frames_(video_decoder_ &decoder, frame_sampler_ &sampler, pixel_tubes_span_t out) noexcept {
    while (sampler.status == status_t::success_k && sampler.kept != sampler.wanted) {
        int const received = avcodec_receive_frame(decoder.context, decoder.frame);
        if (received == AVERROR(EAGAIN) || received == AVERROR_EOF) return;
        if (received < 0) {
            sampler.status = status_t::invalid_input_k;
            return;
        }
        if (sampler.displayed++ % sampler.stride == 0) keep_frame_(decoder, sampler, decoder.frame, out);
        av_frame_unref(decoder.frame);
    }
}

/** Send the parser's current packet, or the flush when it is empty, and pull what it decodes. */
static void submit_packet_(video_decoder_ &decoder, frame_sampler_ &sampler, AVPacket *packet,
                           pixel_tubes_span_t out) noexcept {
    int const sent = avcodec_send_packet(decoder.context, packet);
    if (sent < 0 && sent != AVERROR(EAGAIN) && sent != AVERROR_EOF) sampler.status = status_t::invalid_input_k;
    receive_frames_(decoder, sampler, out);
}

expected_t um::decode_video_frames(demuxed_t const &clip, video_grid_t grid, video_batch_shape_t const &shape,
                                   pixel_tubes_span_t out) noexcept {
    std::size_t const cells = static_cast<std::size_t>(grid.patch_rows) * grid.patch_columns;
    if (out.extent(0) != grid.temporal_groups || out.extent(1) != cells || out.extent(2) != shape.tube_frames ||
        out.extent(3) != shape.patch_side || out.extent(4) != shape.patch_side || out.extent(5) != video_channels_k ||
        shape.tube_frames == 0 || shape.patch_side == 0)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "tube cells [{},{},{},{},{},{}] disagree with the grid ({},{},{}) and the shape", //
                                out.extent(0), out.extent(1), out.extent(2), out.extent(3), out.extent(4),
                                out.extent(5), grid.temporal_groups, grid.patch_rows, grid.patch_columns);
    if (clip.annex_b.empty() || clip.packet_offsets.size() < 2u)
        return expected_t::fail(status_t::invalid_input_k, "the clip holds no packets");
    std::size_t const elements = out.extent(0) * cells * out.extent(2) * out.extent(3) * out.extent(4) * out.extent(5);
    pixel_t *const cells_begin = out.data();
    std::fill(cells_begin, cells_begin + elements, pixel_t {0});

    frame_sampler_ sampler;
    sampler.wanted = static_cast<std::size_t>(grid.temporal_groups) * shape.tube_frames;
    sampler.stride = std::max<u32_t>(shape.frame_stride, 1u);
    sampler.width = grid.patch_columns * shape.patch_side;
    sampler.height = grid.patch_rows * shape.patch_side;
    if (st::failed(sampler.rgb.resize(st::assume_uninitialized,
                                      static_cast<std::size_t>(sampler.width) * sampler.height * video_channels_k)))
        return expected_t::fail(status_t::bad_alloc_k, "RGB frame allocation failed ({}x{})", sampler.width,
                                sampler.height);

    AVCodecID const codec_id = to_av_codec_(clip.codec);
    AVCodec const *const codec = avcodec_find_decoder(codec_id);
    if (codec == nullptr)
        return expected_t::fail(status_t::shape_unsupported_k, "no software decoder for codec {}",
                                static_cast<int>(clip.codec));
    video_decoder_ decoder;
    decoder.context = avcodec_alloc_context3(codec);
    decoder.parser = av_parser_init(codec_id);
    decoder.packet = av_packet_alloc();
    decoder.frame = av_frame_alloc();
    if (decoder.context == nullptr || decoder.parser == nullptr || decoder.packet == nullptr ||
        decoder.frame == nullptr)
        return expected_t::fail(status_t::bad_alloc_k, "video decoder setup failed (codec {})",
                                static_cast<int>(clip.codec));
    decoder.context->thread_count = 1;
    if (avcodec_open2(decoder.context, codec, nullptr) < 0)
        return expected_t::fail(status_t::invalid_input_k, "video decoder open failed (codec {})",
                                static_cast<int>(clip.codec));

    std::size_t const packets = clip.packet_offsets.size() - 1u;
    for (std::size_t index = 0; index != packets && sampler.status == status_t::success_k; ++index) {
        u8_t const *data = av_bytes_(clip.annex_b.data() + clip.packet_offsets[index]);
        int remaining = static_cast<int>(clip.packet_offsets[index + 1] - clip.packet_offsets[index]);
        while (remaining > 0 && sampler.status == status_t::success_k && sampler.kept != sampler.wanted) {
            int const used = av_parser_parse2(decoder.parser, decoder.context, &decoder.packet->data,
                                              &decoder.packet->size, data, remaining, AV_NOPTS_VALUE, AV_NOPTS_VALUE,
                                              0);
            if (used < 0) {
                sampler.status = status_t::invalid_input_k;
                break;
            }
            data += used, remaining -= used; // raw Annex-B splits into access units, container packets pass whole
            if (decoder.packet->size != 0) submit_packet_(decoder, sampler, decoder.packet, out);
        }
    }
    if (sampler.status == status_t::success_k && sampler.kept != sampler.wanted) {
        av_parser_parse2(decoder.parser, decoder.context, &decoder.packet->data, &decoder.packet->size, nullptr, 0,
                         AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0); // flushes the access unit the parser still buffers
        if (decoder.packet->size != 0) submit_packet_(decoder, sampler, decoder.packet, out);
        submit_packet_(decoder, sampler, nullptr, out);
    }
    if (sampler.status != status_t::success_k)
        return expected_t::fail(sampler.status, "video decode failed after {} frames", sampler.kept);
    if (sampler.kept == 0) return expected_t::fail(status_t::invalid_input_k, "the clip decoded no frames");
    return {};
}
