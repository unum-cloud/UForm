/**
 *  @brief Host audio decoding — see the header for the contract.
 *  @file src/media/audio.cpp
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  A firewall over the libav C API: every libav and libswresample error is wrapped and can leave only as an
 *  @ref expected_t .
 */
#include <cstring> // `std::memcpy` — the in-memory AVIO read callback

#include <algorithm> // `std::fill`

extern "C" {
#include <libavcodec/avcodec.h>       // vendor-boundary: `avcodec_*`, `AVPacket`, `AVFrame`
#include <libavformat/avformat.h>     // vendor-boundary: `avformat_*`, `AVFormatContext`, `AVIOContext`
#include <libavutil/channel_layout.h> // vendor-boundary: `AVChannelLayout`, `av_channel_layout_*`
#include <libavutil/mem.h>            // vendor-boundary: `av_malloc`, `av_free`, `av_freep`
#include <libavutil/samplefmt.h>      // vendor-boundary: `AV_SAMPLE_FMT_FLT`
#include <libswresample/swresample.h> // vendor-boundary: `swr_*`, `SwrContext`
}

#include "uform/media/audio.hpp"

namespace um = unum::uform::media;
using namespace um;
using uc::i64_t;
using uc::u8_t;

/** Working buffer size of the custom @c AVIOContext ; FFmpeg's own 32 KB default. */
inline constexpr std::size_t audio_avio_buffer_bytes_k = 32u * 1024u;

/** Backing state for the in-memory AVIO read and seek callbacks over one clip's bytes. */
struct audio_memory_reader_t {

    /** The clip's bytes — borrowed, not owned. */
    b8_t const *data = nullptr;

    /** Total size of @c data in bytes — the seek's end bound. */
    std::size_t size = 0;

    /** The read cursor, advanced by every read and moved by every seek. */
    std::size_t position = 0;
};

/** @c AVIOContext read callback: copy up to @p buffer_size bytes, @c AVERROR_EOF at the end. */
static int audio_memory_read_(void *opaque, u8_t *buffer, int buffer_size) noexcept {
    auto *reader = static_cast<audio_memory_reader_t *>(opaque);
    std::size_t const remaining = reader->size - reader->position;
    std::size_t const requested = static_cast<std::size_t>(buffer_size);
    int const to_read = static_cast<int>(requested < remaining ? requested : remaining);
    if (to_read == 0) return AVERROR_EOF;
    std::memcpy(buffer, reader->data + reader->position, static_cast<std::size_t>(to_read));
    reader->position += static_cast<std::size_t>(to_read);
    return to_read;
}

/** @c AVIOContext seek callback: honors @c SEEK_SET/CUR/END and the @c AVSEEK_SIZE query, @c -1 out of range. */
static i64_t audio_memory_seek_(void *opaque, i64_t offset, int whence) noexcept {
    auto *reader = static_cast<audio_memory_reader_t *>(opaque);
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

/** The untyped planes libswresample writes and reads Float32 samples through. */
[[nodiscard]] static u8_t *swr_plane_(f32_t *samples) noexcept { return reinterpret_cast<u8_t *>(samples); }
[[nodiscard]] static u8_t const *swr_plane_(f32_t const *samples) noexcept {
    return reinterpret_cast<u8_t const *>(samples);
}

/** One clip's demux + decode + resample chain, owned as a unit so its destructor is the single unwind on every
 *  return. @c format holds a context only once @c avformat_open_input succeeded, so one teardown fits every state. */
struct audio_decoder_ {

    /** Backing bytes for the in-memory AVIO callbacks; @c avio holds its address, so it lives here. */
    audio_memory_reader_t reader {};

    /** The custom in-memory @c AVIOContext feeding the demuxer the clip's bytes. */
    AVIOContext *avio = nullptr;

    /** The opened demuxer context. */
    AVFormatContext *format = nullptr;

    /** The audio decoder context bound to the picked stream's codec parameters. */
    AVCodecContext *decoder = nullptr;

    /** The libswresample context converting the decoded audio to mono Float32 at the target rate. */
    SwrContext *resampler = nullptr;

    /** The reusable packet the demuxer reads each compressed unit into. */
    AVPacket *packet = nullptr;

    /** The reusable frame the decoder emits each PCM block into. */
    AVFrame *frame = nullptr;

    /** The picked audio stream's index in @c format->streams . */
    int audio_stream_index = -1;

    ~audio_decoder_() noexcept {
        if (frame != nullptr) av_frame_free(&frame);
        if (packet != nullptr) av_packet_free(&packet);
        if (resampler != nullptr) swr_free(&resampler);
        if (decoder != nullptr) avcodec_free_context(&decoder);
        if (format != nullptr) avformat_close_input(&format);
        if (avio != nullptr) {
            if (avio->buffer != nullptr) av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
    }
};

/**
 *  @brief Build the demux + decode + resample chain for one clip's bytes into @p session .
 *
 *  Stands up its own libav stack over the in-memory bytes, so several run concurrently on a pool. On any failure
 *  the partially built @p session is left for its destructor to unwind.
 *
 *  @param[in] encoded The clip's bytes, kept alive by the caller for @p session 's whole lifetime.
 *  @param[in] sample_rate The mono rate the resampler converts to.
 *  @param[out] session The session the libav chain is built into.
 */
[[nodiscard]] static status_t open_audio_(std::span<b8_t const> encoded, u32_t sample_rate,
                                          audio_decoder_ &session) noexcept {
    session.reader = audio_memory_reader_t {encoded.data(), encoded.size(), 0u};
    u8_t *avio_buffer = static_cast<u8_t *>(av_malloc(audio_avio_buffer_bytes_k));
    if (avio_buffer == nullptr) return status_t::bad_alloc_k;
    session.avio = avio_alloc_context(avio_buffer, static_cast<int>(audio_avio_buffer_bytes_k), 0, &session.reader,
                                      &audio_memory_read_, nullptr, &audio_memory_seek_);
    if (session.avio == nullptr) {
        av_free(avio_buffer);
        return status_t::bad_alloc_k;
    }
    session.avio->seekable = AVIO_SEEKABLE_NORMAL; // advertise the seek callback so probing demuxers can rewind
    AVFormatContext *format = avformat_alloc_context();
    if (format == nullptr) return status_t::bad_alloc_k;
    format->pb = session.avio;
    format->flags |= AVFMT_FLAG_CUSTOM_IO;
    // On failure `avformat_open_input` frees `format` itself, so the session adopts it only once it is open.
    if (avformat_open_input(&format, nullptr, nullptr, nullptr) < 0) return status_t::io_error_k;
    session.format = format;
    if (avformat_find_stream_info(session.format, nullptr) < 0) return status_t::io_error_k;

    for (unsigned stream_index = 0; stream_index != session.format->nb_streams; ++stream_index)
        if (session.format->streams[stream_index]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            session.audio_stream_index = static_cast<int>(stream_index);
            break;
        }
    if (session.audio_stream_index < 0) return status_t::invalid_input_k;

    AVStream *audio_stream = session.format->streams[session.audio_stream_index];
    AVCodec const *codec = avcodec_find_decoder(audio_stream->codecpar->codec_id);
    if (codec == nullptr) return status_t::shape_unsupported_k;
    session.decoder = avcodec_alloc_context3(codec);
    if (session.decoder == nullptr) return status_t::bad_alloc_k;
    if (avcodec_parameters_to_context(session.decoder, audio_stream->codecpar) < 0 ||
        avcodec_open2(session.decoder, codec, nullptr) < 0)
        return status_t::io_error_k;

    AVChannelLayout output_layout {};
    av_channel_layout_default(&output_layout, 1);
    AVChannelLayout input_layout {}; // the copy below uninitializes its destination first, so it must not be garbage
    if (session.decoder->ch_layout.nb_channels > 0) {
        if (av_channel_layout_copy(&input_layout, &session.decoder->ch_layout) < 0) {
            av_channel_layout_uninit(&output_layout);
            return status_t::io_error_k;
        }
    }
    else
        av_channel_layout_default(&input_layout, audio_stream->codecpar->ch_layout.nb_channels > 0
                                                     ? audio_stream->codecpar->ch_layout.nb_channels
                                                     : 1);

    int const resampler_built = swr_alloc_set_opts2(
        &session.resampler, &output_layout, AV_SAMPLE_FMT_FLT, static_cast<int>(sample_rate), &input_layout,
        session.decoder->sample_fmt, session.decoder->sample_rate, 0, nullptr);
    av_channel_layout_uninit(&input_layout);
    av_channel_layout_uninit(&output_layout);
    if (resampler_built < 0 || swr_init(session.resampler) < 0) return status_t::io_error_k;

    session.packet = av_packet_alloc();
    session.frame = av_frame_alloc();
    if (session.packet == nullptr || session.frame == nullptr) return status_t::bad_alloc_k;
    return status_t::success_k;
}

/** The state threaded through the resample phases. */
struct resample_cursor_t {

    /** Samples already written into the fixed-length clip. */
    std::size_t written = 0;

    /** The first error seen, or @c success_k while the phases keep running. */
    status_t outcome = status_t::success_k;
};

/**
 *  @brief Resample one step into the fixed-length clip and advance @p cursor : real input emits its samples, a
 *      null @p input pumps the resampler's delay line, so conversion and drain share this one call.
 *
 *  @param[inout] resampler The libswresample context to convert through.
 *  @param[in] input The input planes, or @c nullptr to drain the resampler.
 *  @param[in] input_samples The samples per input plane.
 *  @param[out] out The fixed-length clip, already silent.
 *  @param[inout] cursor The write position and outcome, advanced in place.
 *  @return Whether the resampler may still yield output: always for real input, and for a drain only while it
 *      emitted samples.
 */
static bool feed_(SwrContext *resampler, u8_t const *const *input, int input_samples, f32r1_span_t out,
                  resample_cursor_t &cursor) noexcept {
    std::size_t const capacity = out.extent(0);
    if (cursor.outcome != status_t::success_k || cursor.written >= capacity) return false;
    int const room = static_cast<int>(capacity - cursor.written);
    u8_t *output_planes[1] = {swr_plane_(out.data() + cursor.written)};
    int const converted = swr_convert(resampler, output_planes, room, input, input != nullptr ? input_samples : 0);
    if (converted < 0) {
        cursor.outcome = status_t::io_error_k;
        return false;
    }
    cursor.written += static_cast<std::size_t>(converted);
    return input != nullptr || converted != 0;
}

/** Pull every frame the decoder holds and resample each into the clip, until it asks for input or the clip is
 *  full. */
static void receive_frames_(audio_decoder_ &session, f32r1_span_t out, resample_cursor_t &cursor) noexcept {
    while (cursor.outcome == status_t::success_k && cursor.written < out.extent(0)) {
        int const received = avcodec_receive_frame(session.decoder, session.frame);
        if (received == AVERROR(EAGAIN) || received == AVERROR_EOF) break;
        if (received < 0) {
            cursor.outcome = status_t::io_error_k;
            break;
        }
        feed_(session.resampler, session.frame->extended_data, session.frame->nb_samples, out, cursor);
        av_frame_unref(session.frame);
    }
}

expected_t um::decode_audio(std::span<b8_t const> encoded, u32_t sample_rate, f32r1_span_t out) noexcept {
    std::size_t const samples = out.extent(0);
    std::fill(out.data(), out.data() + samples, 0.0f);
    if (encoded.empty()) return expected_t::fail(status_t::invalid_input_k, "empty audio clip");
    if (sample_rate == 0) return expected_t::fail(status_t::invalid_input_k, "audio sample rate must be nonzero");

    audio_decoder_ session;
    if (status_t const opened = open_audio_(encoded, sample_rate, session); opened != status_t::success_k)
        return expected_t::fail(opened, "audio clip of {} bytes failed to open", encoded.size());

    resample_cursor_t cursor;
    while (cursor.outcome == status_t::success_k && cursor.written < samples) {
        int const read = av_read_frame(session.format, session.packet);
        if (read == AVERROR_EOF) break;
        if (read < 0) {
            cursor.outcome = status_t::io_error_k;
            break;
        }
        if (session.packet->stream_index != session.audio_stream_index) {
            av_packet_unref(session.packet);
            continue;
        }
        int const sent = avcodec_send_packet(session.decoder, session.packet);
        av_packet_unref(session.packet);
        if (sent < 0 && sent != AVERROR(EAGAIN)) {
            cursor.outcome = status_t::io_error_k;
            break;
        }
        receive_frames_(session, out, cursor);
    }

    // Flush the decoder's tail frames, then drain the resampler's delay line — mandatory for codecs (MP3, AAC) that
    // hold priming or look-ahead samples.
    if (cursor.outcome == status_t::success_k) {
        avcodec_send_packet(session.decoder, nullptr);
        receive_frames_(session, out, cursor);
    }
    while (feed_(session.resampler, nullptr, 0, out, cursor)) {}
    if (cursor.outcome != status_t::success_k) {
        std::fill(out.data(), out.data() + samples, 0.0f);
        return expected_t::fail(cursor.outcome, "audio decode failed after {} samples", cursor.written);
    }
    return {};
}

expected_t um::resample_mono(std::span<f32_t const> samples, u32_t from_rate, u32_t to_rate,
                             f32r1_span_t out) noexcept {
    std::fill(out.data(), out.data() + out.extent(0), 0.0f);
    if (from_rate == 0 || to_rate == 0)
        return expected_t::fail(status_t::invalid_input_k, "resample rates must be nonzero ({} → {})", from_rate,
                                to_rate);
    AVChannelLayout mono {};
    av_channel_layout_default(&mono, 1);
    SwrContext *resampler = nullptr;
    int const built = swr_alloc_set_opts2(&resampler, &mono, AV_SAMPLE_FMT_FLT, static_cast<int>(to_rate), &mono,
                                          AV_SAMPLE_FMT_FLT, static_cast<int>(from_rate), 0, nullptr);
    av_channel_layout_uninit(&mono);
    if (built < 0 || swr_init(resampler) < 0) {
        swr_free(&resampler);
        return expected_t::fail(status_t::io_error_k, "resampler setup failed ({} → {})", from_rate, to_rate);
    }
    resample_cursor_t cursor;
    u8_t const *const input[1] = {swr_plane_(samples.data())};
    feed_(resampler, input, static_cast<int>(samples.size()), out, cursor);
    while (feed_(resampler, nullptr, 0, out, cursor)) {}
    swr_free(&resampler);
    if (cursor.outcome != status_t::success_k)
        return expected_t::fail(cursor.outcome, "resample failed after {} samples", cursor.written);
    return {};
}
