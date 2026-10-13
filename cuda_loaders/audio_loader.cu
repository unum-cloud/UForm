/**
 *  @brief The audio loader's bodies: the libav + ForkUnion decoder session and the draw → stage → decode →
 *      chunk batch path — see the header for the contract. Host decode is a firewall over the libav C API
 *      (like @c dataset_shards.cpp firewalls cuDF): every libav / libswresample error is wrapped and can leave only
 *      as an @ref expected_t . The decode lands one mono Float32 clip; the GPU then chunks it into frames of
 *      @c chunk_samples raw samples and scatters each clip into the batch's @c audio_samples as one sub-row
 *      segment. No cuFFT, no mel — the spectral frontend is a learnable layer of the encoder.
 *  @file cuda_loaders/audio_loader.cu
 */
#include <cstring> // `std::memcpy` — the in-memory AVIO read callback (vendor-boundary)

#include <algorithm> // `std::min`, `std::find`, `std::upper_bound` — the weighted file draw
#include <utility>   // `std::move` — expected-channel propagation

extern "C" {
#include <libavcodec/avcodec.h>       // vendor-boundary: `avcodec_*`, `AVPacket`, `AVFrame`
#include <libavformat/avformat.h>     // vendor-boundary: `avformat_*`, `AVFormatContext`, `AVIOContext`
#include <libavutil/channel_layout.h> // vendor-boundary: `AVChannelLayout`, `av_channel_layout_*`
#include <libavutil/mem.h>            // vendor-boundary: `av_malloc`, `av_free`, `av_freep`
#include <libavutil/samplefmt.h>      // vendor-boundary: `AV_SAMPLE_FMT_FLT`
#include <libswresample/swresample.h> // vendor-boundary: `swr_*`, `SwrContext`
}

#include <forkunion.hpp> // vendor-boundary: the host-CPU decode worker pool

#include "cuda_loaders/audio_loader.cuh"

namespace uc = unum::uform::core;
namespace ul = unum::uform::loaders;
using namespace uc;
using namespace ul;

namespace fu = ashvardanian::forkunion;

/**
 *  @brief Write one batch's audio-segment geometry with the serial cursor model (the tokens loader's walk):
 *      thread 0 packs each clip's @p columns_per_row frames into rows (one clip fills a whole row), jumping the
 *      cursor past a row boundary a segment would cross, and records @c segment_offsets / @c segment_lengths and
 *      @c audio_k into @p segment_modality . Audio is 1-D, so there is no shape descriptor. A segment that
 *      overflows the rectangle is dropped loudly (zero length); PAD segments carry zero length, the repeated
 *      cursor, and @c text_k .
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void audio_segments_kernel_(u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                              modality_span_t segment_modality, u32_t live_clips, u32_t columns_per_row,
                                              u32_t rows) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    u32_t const total_segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const capacity = columns_per_row * rows;
    u32_t const live = live_clips < total_segments ? live_clips : total_segments;
    u32_t cursor = 0;
    for (u32_t segment = 0; segment != total_segments; ++segment) {
        u32_t cells = segment < live ? columns_per_row : 0u;
        u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (cells > room_in_row) cursor += room_in_row;
        if (cursor + cells > capacity) {
            printf("uform: loaded audio batch overflows at segment %u — segment dropped\n", segment);
            cells = 0u;
        }
        segment_offsets[segment] = cursor;
        segment_lengths[segment] = cells;
        bool const filled = segment < live && cells != 0u;
        segment_modality[segment] = filled ? modality_t::audio_k : modality_t::text_k;
        cursor += cells;
    }
    segment_offsets[total_segments] = cursor;
}

/** @brief Backing state for the in-memory AVIO read / seek callbacks over one clip's bytes. */
struct memory_reader_t {
    b8_t const *data = nullptr;
    std::size_t size = 0;
    std::size_t position = 0;
};

/** @brief Working buffer size of the custom @c AVIOContext ; FFmpeg's own 32 KB default. */
constexpr std::size_t avio_buffer_bytes_k = 32u * 1024u;

/** @brief Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *      decorrelated under one @c (seed,step,rank) — the same salts the tokens and image loaders use. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/** @brief @c AVIOContext read callback: copy up to @p buffer_size bytes from the in-memory clip, advancing the
 *      reader; @c AVERROR_EOF at the end. */
static int memory_read_(void *opaque, u8_t *buffer, int buffer_size) noexcept {
    auto *reader = static_cast<memory_reader_t *>(opaque);
    std::size_t const remaining = reader->size - reader->position;
    std::size_t const requested = static_cast<std::size_t>(buffer_size);
    int const to_read = static_cast<int>(requested < remaining ? requested : remaining);
    if (to_read == 0) return AVERROR_EOF;
    std::memcpy(buffer, reader->data + reader->position, static_cast<std::size_t>(to_read));
    reader->position += static_cast<std::size_t>(to_read);
    return to_read;
}

/** @brief @c AVIOContext seek callback over the in-memory clip: honors @c SEEK_SET/CUR/END and the
 *      @c AVSEEK_SIZE size query; @c -1 on an out-of-range target. */
static i64_t memory_seek_(void *opaque, i64_t offset, int whence) noexcept {
    auto *reader = static_cast<memory_reader_t *>(opaque);
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

/**
 *  @brief One clip's libav demux + decode + resample chain, owned as a unit so that a single @ref release
 *      unwinds every handle exactly once on every exit — no per-failure teardown, no double-free. @ref av_open_
 *      builds it; @ref decode_one_clip_ holds it as a local, and its destructor is the one and only unwind.
 */
struct decode_session_ {
    /** @brief Backing bytes for the in-memory AVIO callbacks; @c avio holds its address, so it lives here. */
    memory_reader_t reader {};
    /** @brief The custom in-memory @c AVIOContext feeding the demuxer the clip's bytes (no filesystem touch). */
    AVIOContext *avio = nullptr;
    /** @brief The demuxer context — freed by @c avformat_close_input once opened, else @c avformat_free_context ;
     *      @c av_open_ nulls it on an @c avformat_open_input failure (which frees it itself) so @ref release skips
     *      it and never double-frees. */
    AVFormatContext *format = nullptr;
    /** @brief Whether @c avformat_open_input succeeded — selects which of the two demuxer teardowns @ref release
     *      runs. */
    bool format_opened = false;
    /** @brief The audio decoder context bound to the picked stream's codec parameters. */
    AVCodecContext *decoder = nullptr;
    /** @brief The libswresample context converting the decoded audio to mono Float32 at the target rate. */
    SwrContext *resampler = nullptr;
    /** @brief The reusable packet the demuxer reads each compressed unit into. */
    AVPacket *packet = nullptr;
    /** @brief The reusable frame the decoder emits each PCM block into. */
    AVFrame *frame = nullptr;
    /** @brief The picked audio stream's index in @c format->streams , or negative before one is found. */
    int audio_stream_index = -1;

    /** @brief Free every held libav handle exactly once; correct in any partially-built state, so every exit —
     *      success or failure — unwinds through this single call (invoked by the destructor). */
    void release() noexcept {
        if (frame != nullptr) av_frame_free(&frame);
        if (packet != nullptr) av_packet_free(&packet);
        if (resampler != nullptr) swr_free(&resampler);
        if (decoder != nullptr) avcodec_free_context(&decoder);
        if (format_opened) { avformat_close_input(&format); }
        else if (format != nullptr) { avformat_free_context(format); }
        if (avio != nullptr) {
            if (avio->buffer != nullptr) av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
    }

    /** @brief RAII: the single unwind for the whole chain on every return path. */
    ~decode_session_() noexcept { release(); }
};

/**
 *  @brief Build the demux + decode + resample chain for one clip's bytes into @p session — the setup half of
 *      @ref decode_one_clip_ . Stands up its own libav stack over the in-memory bytes (no filesystem touch), so
 *      several run concurrently on the worker pool. Targets one interleaved mono Float32 channel at
 *      @p sample_rate ; the input layout / rate / format come from the decoder. On any failure the partially
 *      built @p session is left in a state @ref decode_session_::release unwinds cleanly, so the caller returns
 *      the status and the session's destructor owns the teardown.
 *
 *  @param[in] encoded The clip's encoded bytes, kept alive by the caller for @p session 's whole lifetime.
 *  @param[in] sample_rate The mono rate the resampler converts to.
 *  @param[out] session The session the libav chain is built into.
 */
[[nodiscard]] static status_t av_open_(std::span<b8_t const> encoded, u32_t sample_rate,
                                       decode_session_ &session) noexcept {
    session.reader = memory_reader_t {encoded.data(), encoded.size(), 0u};
    u8_t *avio_buffer = static_cast<u8_t *>(av_malloc(avio_buffer_bytes_k));
    if (avio_buffer == nullptr) return status_t::bad_alloc_k;
    session.avio = avio_alloc_context(avio_buffer, static_cast<int>(avio_buffer_bytes_k), /*write=*/0, &session.reader,
                                      &memory_read_, /*write_packet=*/nullptr, &memory_seek_);
    if (session.avio == nullptr) {
        av_free(avio_buffer);
        return status_t::bad_alloc_k;
    }
    session.avio->seekable = AVIO_SEEKABLE_NORMAL; // advertise the seek callback so probing demuxers can rewind
    session.format = avformat_alloc_context();
    if (session.format == nullptr) return status_t::bad_alloc_k;
    session.format->pb = session.avio;
    session.format->flags |= AVFMT_FLAG_CUSTOM_IO;

    if (avformat_open_input(&session.format, /*url=*/nullptr, /*fmt=*/nullptr, /*options=*/nullptr) < 0) {
        // On failure `avformat_open_input` frees `format` itself; null it so `release` skips it (the AVIO
        // buffer, still ours, is the one thing left for the single teardown to reclaim).
        session.format = nullptr;
        return status_t::io_error_k;
    }
    session.format_opened = true;
    if (avformat_find_stream_info(session.format, /*options=*/nullptr) < 0) return status_t::io_error_k;

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
        avcodec_open2(session.decoder, codec, /*options=*/nullptr) < 0)
        return status_t::io_error_k;

    // Output is one interleaved Float32 mono channel at the target rate — the raw waveform the encoder's
    // learnable frontend consumes; input layout / rate / format come from the decoder. Both layouts are
    // value-initialized: `av_channel_layout_copy` uninitializes its destination first, and on a garbage
    // stack struct that free()s a garbage custom-map pointer — a residue-dependent worker-thread crash.
    AVChannelLayout output_layout {};
    av_channel_layout_default(&output_layout, /*channels=*/1);
    AVChannelLayout input_layout {};
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

    int const resampler_built = swr_alloc_set_opts2(&session.resampler, &output_layout, AV_SAMPLE_FMT_FLT,
                                                    static_cast<int>(sample_rate), &input_layout,
                                                    session.decoder->sample_fmt, session.decoder->sample_rate,
                                                    /*log_offset=*/0, /*log_ctx=*/nullptr);
    av_channel_layout_uninit(&input_layout);
    av_channel_layout_uninit(&output_layout);
    if (resampler_built < 0 || swr_init(session.resampler) < 0) return status_t::io_error_k;

    session.packet = av_packet_alloc();
    session.frame = av_frame_alloc();
    if (session.packet == nullptr || session.frame == nullptr) return status_t::bad_alloc_k;
    return status_t::success_k;
}

/** @brief The mutable state threaded through the resample phases, so the outcome / full pair is spelled once
 *      instead of re-declared beside each of the decode, flush, and drain loops. */
struct resample_cursor_t {
    /** @brief Samples already written into the fixed-length row. */
    std::size_t written = 0;
    /** @brief Whether the row reached @c clip_samples — the signal to stop pulling. */
    bool full = false;
    /** @brief The first error seen, or @c success_k while the phases keep running. */
    status_t outcome = status_t::success_k;
};

/**
 *  @brief Resample one step into the fixed-length mono row and advance @p cursor : a real @p decoded frame emits
 *      its samples, a null @p decoded pumps the resampler's delay line, so flush and drain share this one call.
 *      Mono Float32 is one plane, so the write pointer advances by whole samples.
 *
 *  @param[inout] resampler The libswresample context to convert through.
 *  @param[in] decoded The decoded frame to resample, or @c nullptr to drain the resampler to EOF.
 *  @param[out] waveform The fixed-length row samples land into (already pre-zeroed by the caller).
 *  @param[in] clip_samples The row's capacity — output is truncated to it.
 *  @param[inout] cursor The write position / full / outcome state advanced in place.
 *  @return Whether the resampler may still yield output: a real frame — always (the caller loops on the
 *      decoder, not on this return); a null drain feed — only while it emitted samples, @c false at its EOF.
 */
static bool feed_(SwrContext *resampler, AVFrame *decoded, std::span<f32_t> waveform, u32_t clip_samples,
                  resample_cursor_t &cursor) noexcept {
    if (cursor.full || cursor.outcome != status_t::success_k) return false;
    if (cursor.written >= clip_samples) {
        cursor.full = true;
        return false;
    }
    int const room = static_cast<int>(clip_samples - cursor.written);
    u8_t *output_planes[1] = {reinterpret_cast<u8_t *>(waveform.data() + cursor.written)};
    // `swr_convert` wants `uint8_t const *const *`; `AVFrame::extended_data` is `uint8_t **` — the canonical
    // FFmpeg-bridge cast through `void *` reconciles the cv-qualifier levels.
    u8_t const **input_planes = decoded ? (u8_t const **)(void *)decoded->extended_data : nullptr;
    int const input_samples = decoded ? decoded->nb_samples : 0;
    int const converted = swr_convert(resampler, output_planes, room, input_planes, input_samples);
    if (converted < 0) {
        cursor.outcome = status_t::io_error_k;
        return false;
    }
    cursor.written += static_cast<std::size_t>(converted);
    if (cursor.written >= clip_samples) cursor.full = true;
    return decoded != nullptr || converted != 0;
}

/** @brief Pull every frame the decoder currently holds and @ref feed_ each into the row, advancing @p cursor .
 *      Shared by the main decode and the post-flush tail — both just receive-then-resample until the decoder
 *      yields @c EAGAIN or @c EOF .
 *  @param[inout] session The open session whose decoder is drained and whose reusable frame is reused.
 *  @param[out] waveform The fixed-length row the resampled samples land into.
 *  @param[in] clip_samples The row's capacity.
 *  @param[inout] cursor The write position / full / outcome state advanced in place. */
static void receive_frames_(decode_session_ &session, std::span<f32_t> waveform, u32_t clip_samples,
                            resample_cursor_t &cursor) noexcept {
    while (cursor.outcome == status_t::success_k && !cursor.full) {
        int const received = avcodec_receive_frame(session.decoder, session.frame);
        if (received == AVERROR(EAGAIN) || received == AVERROR_EOF) break;
        if (received < 0) {
            cursor.outcome = status_t::io_error_k;
            break;
        }
        feed_(session.resampler, session.frame, waveform, clip_samples, cursor);
        av_frame_unref(session.frame);
    }
}

/**
 *  @brief Decode one encoded clip and resample it to mono Float32 at @p target_sample_rate , writing exactly
 *      @p clip_samples samples into @p waveform — padded with silence when the clip is short, truncated when it
 *      is long. @ref av_open_ builds the demux + decode + resample chain into a local @ref decode_session_ whose
 *      destructor is the one and only unwind for every return; this body drives the pull / decode / resample
 *      loop over it. No filesystem touch, so several run concurrently on the worker pool.
 */
[[nodiscard]] static status_t decode_one_clip_(std::span<b8_t const> encoded, u32_t target_sample_rate,
                                               u32_t clip_samples, std::span<f32_t> waveform) noexcept {
    // Silence is the well-defined empty / failure output — a bad clip becomes a padded row, never a
    // half-written one, so a caller that keeps the batch sees a coherent tensor.
    for (u32_t sample = 0; sample != clip_samples; ++sample) waveform[sample] = 0.0f;
    if (encoded.empty()) return status_t::invalid_input_k;

    decode_session_ session;
    if (status_t const opened = av_open_(encoded, target_sample_rate, session); opened != status_t::success_k)
        return opened;

    // Pull packets, decode frames, resample each into the fixed-length row until it fills or the container ends.
    resample_cursor_t cursor;
    while (cursor.outcome == status_t::success_k && !cursor.full) {
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
        receive_frames_(session, waveform, clip_samples, cursor);
    }

    // Flush the decoder's tail frames, then drain the resampler's delay line — mandatory for codecs (MP3, AAC)
    // that hold priming / look-ahead samples. Both phases just pump to EOF through the shared `feed_` step.
    if (cursor.outcome == status_t::success_k) {
        avcodec_send_packet(session.decoder, nullptr);
        receive_frames_(session, waveform, clip_samples, cursor);
    }
    while (feed_(session.resampler, nullptr, waveform, clip_samples, cursor)) {}

    // The tail past the decoded samples stays the pre-zeroed silence — the pad half of pad / truncate; the
    // session's destructor is the single unwind as this returns.
    return cursor.outcome;
}

expected_t ul::decode_audios(cpu_decode_pool_t &pool, column_view_t tape, std::span<u64_t const> drawn_rows,
                             std::span<audio_batch_shape_t const> representations, audio_samples_span_t audio_samples,
                             u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                             modality_span_t segment_modality, gpu_stream_t &stream) noexcept {
    std::size_t const clips = drawn_rows.size();
    if (clips == 0) return {};
    if (representations.empty())
        return expected_t::fail(status_t::invalid_input_k, "decode_audios requires a non-empty representations span");
    u32_t const rows = static_cast<u32_t>(audio_samples.extent(0));
    u32_t const columns = static_cast<u32_t>(audio_samples.extent(1));
    u32_t const chunk_samples = static_cast<u32_t>(audio_samples.extent(2));
    if (columns == 0 || chunk_samples == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "the audio grid is empty (columns={} chunk={})", columns,
                                chunk_samples);
    // The shape is the single source of geometry; each representation must match the grid's chunk width.
    for (audio_batch_shape_t const &representation : representations)
        if (representation.chunk_samples != chunk_samples)
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "representation chunk_samples {} must match the audio grid's {}",
                                    representation.chunk_samples, chunk_samples);
    std::size_t const reps_count = representations.size();
    std::size_t const job_count = clips * reps_count; // one segment per (clip, representation)
    if (segment_lengths.extent(0) < job_count || job_count > rows)
        return expected_t::fail(status_t::shape_unsupported_k, "audio geometry cannot hold {} segments", job_count);
    // One clip fills a whole row: `columns` frames, each of `chunk_samples` raw samples.
    u32_t const clip_samples = columns * chunk_samples;

    // Hop each drawn clip's encoded bytes device→pinned back to back — the decoder's host-side source. A pinned
    // `column_view_t` (packed `encoded` buffer + `[clips+1]` offsets) tapes them, so a per-clip byte range is one
    // indexed `staged_tape[clip]` read whose length is the adjacent-offset difference — no side table to disagree.
    array<b8_t, pinned_alloc<b8_t>> encoded;
    array<u64_t> encoded_offsets;
    if (encoded_offsets.try_resize_uninitialized(clips + 1) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "clip offset staging allocation failed ({} clips)", clips);
    std::size_t total_bytes = 0;
    for (std::size_t index = 0; index != clips; ++index) total_bytes += tape[drawn_rows[index]].size();
    if (encoded.try_resize_uninitialized(total_bytes) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "clip staging allocation failed ({} bytes)", total_bytes);
    std::size_t cursor = 0;
    for (std::size_t index = 0; index != clips; ++index) {
        auto const blob = tape[drawn_rows[index]];
        encoded_offsets[index] = cursor;
        if (blob.size() != 0 && cuMemcpyDtoHAsync(encoded.data() + cursor, reinterpret_cast<CUdeviceptr>(blob.data()),
                                                  blob.size(), stream.get()) != CUDA_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "clip byte staging copy failed at clip {}", index);
        cursor += blob.size();
    }
    encoded_offsets[clips] = cursor;
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "clip byte staging synchronization failed");
    column_view_t const staged_tape {std::span<b8_t const> {encoded.data(), total_bytes},
                                     std::span<u64_t const> {encoded_offsets.data(), clips + 1}};

    // Decode + resample to mono Float32 across the borrowed pool, into a pinned waveform landing already carrying
    // the grid's `[clips,columns,chunk_samples]` layout: clip `c` fills row `c` (a clip is exactly `clip_samples`).
    std::size_t const waveform_samples = job_count * static_cast<std::size_t>(clip_samples);
    array<f32_t, pinned_alloc<f32_t>> waveforms;
    array<i32_t> statuses;
    array<u32_t> sample_rates; // each representation's resample rate, captured for the pool workers
    if (waveforms.try_resize_uninitialized(waveform_samples) != status_t::success_k ||
        statuses.try_resize_uninitialized(job_count) != status_t::success_k ||
        sample_rates.try_resize(reps_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "waveform staging allocation failed ({} samples)",
                                waveform_samples);
    for (std::size_t rep = 0; rep != reps_count; ++rep) sample_rates[rep] = representations[rep].sample_rate;
    f32_t *const waveform_base = waveforms.data();
    i32_t *const status_base = statuses.data();
    u32_t const *const rate_base = sample_rates.data();
    std::size_t const reps = reps_count;
    // A job is (clip, representation): job j = (clip j/reps, rep j%reps), resampled to that representation's rate and
    // chunked into row j. A clip is decoded once per representation (one decode for the single-representation case).
    pool.for_n(job_count, [=](std::size_t job, fu::thread_in_domain_t) noexcept {
        std::size_t const clip = job / reps;
        std::span<b8_t const> const encoded_clip = staged_tape[clip];
        std::span<b8_t const> const bytes {encoded_clip.data(), encoded_clip.size()};
        std::span<f32_t> const job_waveform {waveform_base + job * static_cast<std::size_t>(clip_samples),
                                             static_cast<std::size_t>(clip_samples)};
        status_base[job] = static_cast<i32_t>(
            decode_one_clip_(bytes, rate_base[job % reps], clip_samples, job_waveform));
    });
    // Mark-and-continue: a clip that fails to decode is left silence (decode_one_clip_ pre-zeros its row), matching
    // the batch's "existence is read from the data" contract and the video loader's per-clip fault isolation. Only a
    // wholly failed batch is a systematic error (a bad codec / driver), so fail loud only then.
    std::size_t failures = 0;
    for (std::size_t job = 0; job != job_count; ++job)
        if (statuses[job] != static_cast<i32_t>(status_t::success_k)) ++failures;
    if (failures == job_count)
        return expected_t::fail(static_cast<status_t>(statuses[0]), "every one of {} segments failed to decode",
                                job_count);

    // Climb onto the GPU: one contiguous copy lands the decoded PCM into the grid's leading `clips` rows — the
    // decode buffer already matches that `[clips,columns,chunk_samples]` sub-span, so there is no reshape or
    // scatter — then the geometry kernel writes the segment table. Both retire at the closing synchronize.
    audio_samples_span_t const landing = audio_samples(nk::range(0, job_count), nk::slice);
    if (expected_t hopped = copy_host_to_device(f32r1_view_t {waveforms.data(), waveform_samples}, landing.flatten<1>(),
                                                stream);
        !hopped)
        return hopped;

    expected_t staged = launch_kernel(reinterpret_cast<void const *>(&audio_segments_kernel_), dim3 {1u, 1u, 1u},
                                      dim3 {1u, 1u, 1u}, stream, segment_offsets, segment_lengths, segment_modality,
                                      static_cast<u32_t>(job_count), columns, rows);
    // Retire the whole stream-ordered chain before the pinned decode landing falls out of scope.
    if (staged && cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        staged = expected_t::fail(status_t::cuda_error_k, "audio staging synchronization failed");
    return staged;
}

expected<audio_loader_parquet_t> audio_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                              sz::string_view_t path_or_glob, u64_t seed,
                                                              cpu_decode_pool_t &pool,
                                                              audio_batch_shape_t audio) noexcept {
    using result_t = expected<audio_loader_parquet_t>;

    if (audio.chunk_samples == 0 || audio.sample_rate == 0)
        return result_t::fail(status_t::invalid_input_k, "audio geometry is empty (chunk_samples={} sample_rate={})",
                              audio.chunk_samples, audio.sample_rate);

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths — POSIX glob(3).
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    audio_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.pool_ = &pool;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;
    loader.audio_ = audio;

    // Footer only — host-side, no GPU: a running prefix of each file's row count, the weighted file draw's
    // search table; its back holds the dataset's total rows.
    if (loader.row_prefix_.try_resize(loader.absolute_paths_.size() + 1) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "shard prefix allocation failed ({} shards)",
                              loader.absolute_paths_.size());
    loader.row_prefix_[0] = 0;
    for (std::size_t index = 0; index != loader.absolute_paths_.size(); ++index) {
        auto metadata = parquet_file_t::table_metadata(session, loader.absolute_paths_[index]);
        if (!metadata) return result_t::fail(std::move(metadata));
        if (metadata->rows == 0)
            return result_t::fail(status_t::invalid_input_k, "shard {} is empty", loader.absolute_paths_[index]);
        loader.row_prefix_[index + 1] = loader.row_prefix_[index] + metadata->rows;
    }
    return result_t {std::move(loader)};
}

expected_t audio_loader_parquet_t::fill(gpu_stream_t &gpu_stream, audio_loader_parquet_config_t config, u32_t gpu_rank,
                                        u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    audio_samples_view_t const first_grid = batches[0].audio_samples.view();
    std::size_t const rows = first_grid.extent(0);
    std::size_t const columns = first_grid.extent(1);
    if (rows == 0 || columns == 0) return expected_t::fail(status_t::shape_unsupported_k, "the batch carries no audio");

    // The loader's single flat `audio_batch_shape_t` is the one representation (the shape is the single source of
    // geometry) — a size-1 span today, cross-product-ready for a future multi-rate config. Each clip emits one
    // whole-row segment per representation, so a clip costs `representations.size()` rows; the clip count is one
    // representation-set per row, capped by the segment capacity, so the serial cursor packs without overflowing.
    std::span<audio_batch_shape_t const> const representations {&audio_, 1};
    std::size_t const max_segments = batches[0].segment_lengths.extent(0);
    std::size_t const clips_per_batch = std::min<std::size_t>(max_segments, rows) / representations.size();
    if (clips_per_batch == 0) return expected_t::fail(status_t::shape_unsupported_k, "the batch holds no audio clips");
    std::size_t const total_clips = unroll * clips_per_batch;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_rows = row_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};

    // Open distinct files, drawn with probability proportional to their row count, until the pool covers the
    // fill's clips; never fewer than two, unless the whole dataset is one file.
    array<sz::string_view_t> picked_paths;
    array<std::size_t> picked_shards;
    if (picked_paths.try_resize(shard_count) != status_t::success_k ||
        picked_shards.try_resize(shard_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
    std::size_t picked_count = 0, pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < total_clips || picked_count < floor_files) && picked_count < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        std::span<std::size_t const> const chosen {picked_shards.data(), picked_count};
        if (std::find(chosen.begin(), chosen.end(), file) != chosen.end()) continue;
        picked_shards[picked_count] = file;
        picked_paths[picked_count] = absolute_paths_[file].view();
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
        ++picked_count;
    }

    // Decode the picked files' clip column into one device reservoir; the true pool is the decoded rows.
    auto reservoir_read = parquet_file_t::make(*storage_, picked_paths.view().subspan(0, picked_count),
                                               {&config.column, 1}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_clips)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files cover fewer than the {} clips this fill needs",
                                pooled_rows, picked_count, total_clips);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");

    // The reservoir's clip offsets to the host once — every per-row byte bound is then host arithmetic.
    column_view_t const column = reservoir.column(0);
    array<u64_t> reservoir_offsets;
    if (reservoir_offsets.try_resize_uninitialized(column.offsets_.size()) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "reservoir offsets landing failed ({} rows)", pooled_rows);
    if (cuMemcpyDtoHAsync(reservoir_offsets.data(), reinterpret_cast<CUdeviceptr>(column.offsets_.data()),
                          column.offsets_.size() * sizeof(u64_t), gpu_stream.get()) != CUDA_SUCCESS ||
        cuStreamSynchronize(gpu_stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "reservoir offsets readback failed");

    // Draw the whole fill's rows on the host — a keyed bijection over the pool.
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    array<u64_t> record_indices;
    if (record_indices.try_resize_uninitialized(total_clips) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_clips);
    for (std::size_t segment = 0; segment != total_clips; ++segment)
        record_indices[segment] = epoch_permutation(row_key, data_epoch_t {training_step}, segment, pooled_rows);

    std::span<u64_t const> const host_offsets {reservoir_offsets.data(), reservoir_offsets.size()};
    column_view_t const tape {column.buffer_, host_offsets};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        audio_samples_view_t const grid_view = loaded.audio_samples.view();
        if (grid_view.extent(0) != rows || grid_view.extent(1) != columns ||
            clips_per_batch > loaded.segment_lengths.extent(0))
            return expected_t::fail(status_t::shape_unsupported_k, "batch {} cannot hold {} clips", batch,
                                    clips_per_batch);
        std::span<u64_t const> const drawn {record_indices.data() + batch * clips_per_batch, clips_per_batch};
        audio_samples_span_t const audio_samples = loaded.audio_samples.span();
        u32r1_span_t const segment_offsets = loaded.segment_offsets.span();
        u32r1_span_t const segment_lengths = loaded.segment_lengths.span();
        modality_span_t const segment_modality = loaded.segment_modality.span();
        if (expected_t decoded = decode_audios(*pool_, tape, drawn, representations, audio_samples, segment_offsets,
                                               segment_lengths, segment_modality, gpu_stream);
            !decoded)
            return decoded;
    }
    return {};
}
