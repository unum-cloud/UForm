/**
 *  @file src/loaders/audio_loader.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *  @brief The audio loader's bodies: the libav + ForkUnion decoder session and the draw → stage →
 *      decode → chunk batch path — see the header for the contract.
 *
 *  Host decode is @ref unum::uform::media::decode_audio , one mono Float32 clip per job across the
 *  pool; the GPU then chunks it into frames of @c chunk_samples raw samples and scatters each clip
 *  into the batch's @c audio_samples as one sub-row segment. No cuFFT, no mel — the spectral
 *  frontend is a learnable layer of the encoder.
 */
#include <algorithm> // `std::find`, `std::upper_bound` — the weighted file draw
#include <utility>   // `std::move` — expected-channel propagation

#include "uform/loaders/audio_loader.cuh"
#include "uform/media/audio.hpp" // `decode_audio` — the libav decode and resample to mono

namespace ue = unum::uform::encoder;
namespace ul = unum::uform::loaders;
namespace um = unum::uform::media;
using namespace ue;
using namespace unum::uform::utilities;
using namespace ul;

/**
 *  @brief Write one batch's audio-segment geometry with the serial cursor model (the
 *      tokens loader's walk).
 *
 *  Thread 0 packs each clip's @ref media_cells frames into half rows (two clips share a row, at its
 *  start and its middle), and records @c segment_offsets / @c segment_lengths and @c audio_k into
 *  @p segment_modality . Audio is 1-D, so there is no shape descriptor. A segment that overflows
 *  the rectangle is dropped (zero length) and counted into @p dropped ; PAD segments carry zero
 *  length, the repeated cursor, and @c text_k .
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void audio_segments_kernel_(u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                              modality_span_t segment_modality, u32r1_span_t dropped, u32_t live_clips,
                                              u32_t columns_per_row, u32_t rows) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    u32_t const total_segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const capacity = columns_per_row * rows;
    u32_t const live = live_clips < total_segments ? live_clips : total_segments;
    u32_t const clip_cells = media_cells(modality_t::audio_k, columns_per_row);
    u32_t cursor = 0, overflowed = 0;
    for (u32_t segment = 0; segment != total_segments; ++segment) {
        u32_t cells = segment < live ? clip_cells : 0u;
        if (cursor + cells > capacity) ++overflowed, cells = 0u;
        segment_offsets[segment] = cursor;
        segment_lengths[segment] = cells;
        bool const filled = segment < live && cells != 0u;
        segment_modality[segment] = filled ? modality_t::audio_k : modality_t::text_k;
        cursor += cells;
    }
    segment_offsets[total_segments] = cursor;
    dropped[0] += overflowed;
}

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *  decorrelated under one @c (seed,step,rank) — the same salts the tokens and image loaders use. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

expected_t ul::decode_audios(pool_view_t pool, column_view_t tape, u64r1_view_t drawn_rows,
                             std::span<audio_batch_shape_t const> representations, audio_samples_span_t audio_samples,
                             u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                             modality_span_t segment_modality, u32r1_span_t dropped, gpu_stream_t &stream) noexcept {
    std::size_t const clips = drawn_rows.size();
    if (clips == 0) return {};
    if (representations.empty())
        return expected_t::fail(status_t::invalid_argument_k,
                                "decode_audios requires a non-empty representations span");
    u32_t const rows = static_cast<u32_t>(audio_samples.extent(0));
    u32_t const columns = static_cast<u32_t>(audio_samples.extent(1));
    u32_t const chunk_samples = static_cast<u32_t>(audio_samples.extent(2));
    if (columns == 0 || chunk_samples == 0)
        return expected_t::fail(status_t::unsupported_k, "the audio grid is empty (columns={} chunk={})", columns,
                                chunk_samples);
    // The shape is the single source of geometry; each representation must match the
    // grid's chunk width.
    for (audio_batch_shape_t const &representation : representations)
        if (representation.chunk_samples != chunk_samples)
            return expected_t::fail(status_t::unsupported_k,
                                    "representation chunk_samples {} must match the audio grid's {}",
                                    representation.chunk_samples, chunk_samples);
    std::size_t const reps_count = representations.size();
    std::size_t const job_count = clips * reps_count; // one segment per (clip, representation)
    u32_t const clip_cells = media_cells(modality_t::audio_k, columns);
    if (segment_lengths.extent(0) < job_count || job_count * clip_cells > std::size_t {rows} * columns)
        return expected_t::fail(status_t::unsupported_k, "audio geometry cannot hold {} segments", job_count);
    // One clip fills half a row: `clip_cells` frames, each of `chunk_samples` raw samples.
    u32_t const clip_samples = clip_cells * chunk_samples;

    // Hop each drawn clip's encoded bytes device → pinned back to back — the decoder's host-side
    // source. A pinned `column_view_t` (packed `encoded` buffer + `[clips+1]` offsets) tapes them,
    // so a per-clip byte range is one indexed `staged_tape[clip]` read whose length is the
    // adjacent-offset difference — no side table to disagree.
    st::basic_vector<b8_t, pinned_alloc<b8_t>> encoded;
    st::basic_vector<u64_t> encoded_offsets;
    if (st::failed(encoded_offsets.resize(st::assume_uninitialized, clips + 1)))
        return expected_t::fail(status_t::bad_alloc_k, "clip offset staging allocation failed ({} clips)", clips);
    std::size_t total_bytes = 0;
    for (std::size_t index = 0; index != clips; ++index) total_bytes += tape[drawn_rows[index]].size();
    if (st::failed(encoded.resize(st::assume_uninitialized, total_bytes)))
        return expected_t::fail(status_t::bad_alloc_k, "clip staging allocation failed ({} bytes)", total_bytes);
    std::size_t cursor = 0;
    for (std::size_t index = 0; index != clips; ++index) {
        auto const blob = tape[drawn_rows[index]];
        encoded_offsets[index] = cursor;
        if (expected_t hopped = copy_device_to_host(b8r1_view_t {blob},
                                                    bytes_span_t {encoded}.subspan(cursor, blob.size()), stream);
            !hopped)
            return hopped;
        cursor += blob.size();
    }
    encoded_offsets[clips] = cursor;
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::device_error_k, "clip byte staging synchronization failed");
    column_view_t const staged_tape {bytes_view_t {encoded}, std::span<u64_t const> {encoded_offsets}};

    // Decode + resample to mono Float32 across the borrowed pool, into a pinned waveform landing
    // already carrying the grid's cell layout: clip `c` fills the half row at cell `c·clip_cells`
    // (a clip is exactly `clip_samples`).
    std::size_t const waveform_samples = job_count * static_cast<std::size_t>(clip_samples);
    st::basic_vector<f32_t, pinned_alloc<f32_t>> waveforms;
    st::basic_vector<i32_t> statuses;
    st::basic_vector<u32_t> sample_rates; // each representation's resample rate, captured for the pool workers
    if (st::failed(waveforms.resize(st::assume_uninitialized, waveform_samples)) ||
        st::failed(statuses.resize(st::assume_uninitialized, job_count)) || st::failed(sample_rates.resize(reps_count)))
        return expected_t::fail(status_t::bad_alloc_k, "waveform staging allocation failed ({} samples)",
                                waveform_samples);
    for (std::size_t rep = 0; rep != reps_count; ++rep) sample_rates[rep] = representations[rep].sample_rate;
    f32r2_span_t const job_waveforms {waveforms.data(), job_count, clip_samples};
    i32_t *const status_base = statuses.data();
    u32_t const *const rate_base = sample_rates.data();
    std::size_t const reps = reps_count;
    // A job is (clip, representation): job j = (clip j/reps, rep j%reps), resampled to that
    // representation's rate and chunked into row j. A clip is decoded once per representation (one
    // decode for the single-representation case).
    status_t const dispatched = pool.for_n(job_count, [=](std::size_t job, std::size_t) noexcept {
        std::size_t const clip = job / reps;
        bytes_view_t const bytes = staged_tape[clip];
        f32r1_span_t const job_waveform = job_waveforms[job, nk::slice];
        status_base[job] = static_cast<i32_t>(um::decode_audio(bytes, rate_base[job % reps], job_waveform).status_);
    });
    if (dispatched != status_t::success_k)
        return expected_t::fail(dispatched, "the decode pool refused {} audio jobs", job_count);
    // Mark-and-continue: a clip that fails to decode is left silence (decode_audio zeros its row),
    // matching the batch's "existence is read from the data" contract and the video loader's
    // per-clip fault isolation. Only a wholly failed batch is a systematic error (a bad codec /
    // driver), so fail loud only then.
    std::size_t failures = 0;
    for (std::size_t job = 0; job != job_count; ++job)
        if (statuses[job] != static_cast<i32_t>(status_t::success_k)) ++failures;
    if (failures == job_count)
        return expected_t::fail(static_cast<status_t>(statuses[0]), "every one of {} segments failed to decode",
                                job_count);

    // Climb onto the GPU: one contiguous copy lands the decoded PCM into the grid's leading
    // `job_count` half rows — the decode buffer already matches those cells, so there is no
    // reshape or scatter — then the geometry kernel writes the segment table. Both retire at
    // the closing synchronize.
    f32r1_span_t const landing {audio_samples.data(), waveform_samples};
    if (expected_t hopped = copy_host_to_device(std::span<f32_t const> {waveforms}, landing, stream); !hopped)
        return hopped;

    expected_t staged = launch_kernel(&audio_segments_kernel_, dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u}, stream,
                                      segment_offsets, segment_lengths, segment_modality, dropped,
                                      static_cast<u32_t>(job_count), columns, rows);
    // Retire the whole stream-ordered chain before the pinned decode landing falls out of scope.
    if (staged && cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        staged = expected_t::fail(status_t::device_error_k, "audio staging synchronization failed");
    return staged;
}

expected<audio_loader_parquet_t> audio_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                              sz::string_view_t path_or_glob, u64_t seed,
                                                              pool_view_t pool, audio_batch_shape_t audio) noexcept {
    using result_t = expected<audio_loader_parquet_t>;

    if (audio.chunk_samples == 0 || audio.sample_rate == 0)
        return result_t::fail(status_t::invalid_argument_k, "audio geometry is empty (chunk_samples={} sample_rate={})",
                              audio.chunk_samples, audio.sample_rate);

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths
    // — POSIX glob(3).
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    audio_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.pool_ = pool;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;
    loader.audio_ = audio;

    // Footer only — host-side, no GPU: a running prefix of each file's row count, the weighted file
    // draw's search table; its back holds the dataset's total rows.
    if (st::failed(loader.row_prefix_.resize(loader.absolute_paths_.size() + 1)))
        return result_t::fail(status_t::bad_alloc_k, "shard prefix allocation failed ({} shards)",
                              loader.absolute_paths_.size());
    loader.row_prefix_[0] = 0;
    for (std::size_t index = 0; index != loader.absolute_paths_.size(); ++index) {
        auto metadata = parquet_file_t::table_metadata(session, loader.absolute_paths_[index]);
        if (!metadata) return result_t::fail(std::move(metadata));
        if (metadata->rows == 0)
            return result_t::fail(status_t::invalid_argument_k, "shard {} is empty", loader.absolute_paths_[index]);
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
    if (rows == 0 || columns == 0) return expected_t::fail(status_t::unsupported_k, "the batch carries no audio");

    // The loader's single flat `audio_batch_shape_t` is the one representation (the shape is the
    // single source of geometry) — a size-1 span today, cross-product-ready for a future multi-rate
    // config. Each clip emits one half-row segment per representation; the clip count is one
    // representation-set per half row, capped by the segment capacity, so the serial cursor packs
    // without overflowing.
    std::span<audio_batch_shape_t const> const representations {&audio_, 1};
    std::size_t const max_segments = batches[0].segment_lengths.extent(0);
    std::size_t const per_row = columns / media_cells(modality_t::audio_k, static_cast<u32_t>(columns));
    std::size_t const clips_per_batch = fu::min_of<std::size_t>(max_segments, rows * per_row) / representations.size();
    if (clips_per_batch == 0) return expected_t::fail(status_t::unsupported_k, "the batch holds no audio clips");
    std::size_t const total_clips = unroll * clips_per_batch;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = fu::min_of<std::size_t>(2, shard_count);
    std::size_t const total_rows = row_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};

    // Open distinct files, drawn with probability proportional to their row count, until the pool
    // covers the fill's clips; never fewer than two, unless the whole dataset is one file.
    st::basic_vector<sz::string_view_t> picked_paths;
    st::basic_vector<std::size_t> picked_shards;
    std::size_t pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < total_clips || picked_shards.size() < floor_files) &&
           picked_shards.size() < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        if (std::find(picked_shards.begin(), picked_shards.end(), file) != picked_shards.end()) continue;
        if (st::failed(picked_shards.push_back(std::size_t {file})) ||
            st::failed(picked_paths.push_back(absolute_paths_[file].view())))
            return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
    }

    // Decode the picked files' clip column into one device reservoir; the true pool is
    // the decoded rows.
    auto reservoir_read = parquet_file_t::make(*storage_, std::span<sz::string_view_t const> {picked_paths},
                                               std::span<sz::string_view_t const> {&config.column, 1}, {}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_clips)
        return expected_t::fail(status_t::unsupported_k,
                                "{} pooled rows across {} files cover fewer than the {} clips this fill needs",
                                pooled_rows, picked_shards.size(), total_clips);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok())
        return expected_t::fail(status_t::device_error_k, "context entry failed for the loader stream");

    // The reservoir's clip offsets to the host once — every per-row byte bound is
    // then host arithmetic.
    column_view_t const column = reservoir.column(0);
    std::size_t const offsets_count = column.size() + 1u;
    st::basic_vector<u64_t> reservoir_offsets;
    if (st::failed(reservoir_offsets.resize(st::assume_uninitialized, offsets_count)))
        return expected_t::fail(status_t::bad_alloc_k, "reservoir offsets landing failed ({} rows)", pooled_rows);
    if (expected_t landed = copy_device_to_host(column.offsets, std::span<u64_t> {reservoir_offsets}, gpu_stream);
        !landed)
        return landed;
    if (cuStreamSynchronize(gpu_stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::device_error_k, "reservoir offsets readback failed");

    // Draw the whole fill's rows on the host — a keyed bijection over the pool.
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    st::basic_vector<u64_t> record_indices;
    if (st::failed(record_indices.resize(st::assume_uninitialized, total_clips)))
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_clips);
    for (std::size_t segment = 0; segment != total_clips; ++segment)
        record_indices[segment] = epoch_permutation(row_key, data_epoch_t {training_step}, segment, pooled_rows);

    column_view_t const tape {column.payload, std::span<u64_t const> {reservoir_offsets}};
    u64r2_view_t const batch_records {record_indices.data(), unroll, clips_per_batch};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        audio_samples_view_t const grid_view = loaded.audio_samples.view();
        if (grid_view.extent(0) != rows || grid_view.extent(1) != columns ||
            clips_per_batch > loaded.segment_lengths.extent(0))
            return expected_t::fail(status_t::unsupported_k, "batch {} cannot hold {} clips", batch, clips_per_batch);
        if (expected_t decoded = decode_audios(pool_, tape, batch_records[batch, nk::slice], representations,
                                               loaded.audio_samples.span(), loaded.segment_offsets.span(),
                                               loaded.segment_lengths.span(), loaded.segment_modality.span(),
                                               loaded.dropped.span(), gpu_stream);
            !decoded)
            return decoded;
    }
    return {};
}
