/**
 *  @brief The audio loader's bodies: the libav + ForkUnion decoder session and the draw → stage → decode →
 *      chunk batch path — see the header for the contract.
 *  @file src/loaders/audio_loader.cu
 *
 *  Host decode is @ref unum::uform::media::decode_audio , one mono Float32 clip per job across the pool; the GPU then
 *  chunks it into frames of @c chunk_samples raw samples and scatters each clip into the batch's @c audio_samples
 *  as one sub-row segment. No cuFFT, no mel — the spectral frontend is a learnable layer of the encoder.
 */
#include <algorithm> // `std::min`, `std::find`, `std::upper_bound` — the weighted file draw
#include <utility>   // `std::move` — expected-channel propagation

#include <forkunion.hpp> // vendor-boundary: the host-CPU decode worker pool

#include "media/audio.hpp" // `decode_audio` — the libav decode and resample to mono
#include "loaders/audio_loader.cuh"

namespace uc = unum::uform::core;
namespace ul = unum::uform::loaders;
namespace um = unum::uform::media;
using namespace uc;
using namespace ul;

namespace fu = ashvardanian::forkunion;

/**
 *  @brief Write one batch's audio-segment geometry with the serial cursor model (the tokens loader's walk).
 *
 *  Thread 0 packs each clip's @p columns_per_row frames into rows (one clip fills a whole row), jumping the
 *  cursor past a row boundary a segment would cross, and records @c segment_offsets / @c segment_lengths and
 *  @c audio_k into @p segment_modality . Audio is 1-D, so there is no shape descriptor. A segment that
 *  overflows the rectangle is dropped loudly (zero length); PAD segments carry zero length, the repeated
 *  cursor, and @c text_k .
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

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *  decorrelated under one @c (seed,step,rank) — the same salts the tokens and image loaders use. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

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

    // Hop each drawn clip's encoded bytes device → pinned back to back — the decoder's host-side source. A pinned
    // `column_view_t` (packed `encoded` buffer + `[clips+1]` offsets) tapes them, so a per-clip byte range is one
    // indexed `staged_tape[clip]` read whose length is the adjacent-offset difference — no side table to disagree.
    array<b8_t, pinned_alloc<b8_t>> encoded;
    array<u64_t> encoded_offsets;
    if (encoded_offsets.resize_uninitialized(clips + 1) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "clip offset staging allocation failed ({} clips)", clips);
    std::size_t total_bytes = 0;
    for (std::size_t index = 0; index != clips; ++index) total_bytes += tape[drawn_rows[index]].size();
    if (encoded.resize_uninitialized(total_bytes) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "clip staging allocation failed ({} bytes)", total_bytes);
    std::size_t cursor = 0;
    for (std::size_t index = 0; index != clips; ++index) {
        auto const blob = tape[drawn_rows[index]];
        encoded_offsets[index] = cursor;
        if (expected_t hopped = copy_device_to_host(b8r1_view_t {blob},
                                                    b8r1_span_t {encoded.data() + cursor, blob.size()}, stream);
            !hopped)
            return hopped;
        cursor += blob.size();
    }
    encoded_offsets[clips] = cursor;
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "clip byte staging synchronization failed");
    column_view_t const staged_tape {{encoded.data(), total_bytes}, {encoded_offsets.data(), clips + 1}};

    // Decode + resample to mono Float32 across the borrowed pool, into a pinned waveform landing already carrying
    // the grid's `[clips,columns,chunk_samples]` layout: clip `c` fills row `c` (a clip is exactly `clip_samples`).
    std::size_t const waveform_samples = job_count * static_cast<std::size_t>(clip_samples);
    array<f32_t, pinned_alloc<f32_t>> waveforms;
    array<i32_t> statuses;
    array<u32_t> sample_rates; // each representation's resample rate, captured for the pool workers
    if (waveforms.resize_uninitialized(waveform_samples) != status_t::success_k ||
        statuses.resize_uninitialized(job_count) != status_t::success_k ||
        sample_rates.resize(reps_count) != status_t::success_k)
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
        std::span<b8_t const> const bytes = staged_tape[clip];
        f32r1_span_t const job_waveform {waveform_base + job * static_cast<std::size_t>(clip_samples),
                                         static_cast<std::size_t>(clip_samples)};
        status_base[job] = static_cast<i32_t>(um::decode_audio(bytes, rate_base[job % reps], job_waveform).status_);
    });
    // Mark-and-continue: a clip that fails to decode is left silence (decode_audio zeros its row), matching
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

    expected_t staged = launch_kernel(&audio_segments_kernel_, dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u}, stream,
                                      segment_offsets, segment_lengths, segment_modality, static_cast<u32_t>(job_count),
                                      columns, rows);
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
    if (loader.row_prefix_.resize(loader.absolute_paths_.size() + 1) != status_t::success_k)
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
    if (picked_paths.resize(shard_count) != status_t::success_k ||
        picked_shards.resize(shard_count) != status_t::success_k)
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
    std::size_t const offsets_count = column.size() + 1u;
    array<u64_t> reservoir_offsets;
    if (reservoir_offsets.resize_uninitialized(offsets_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "reservoir offsets landing failed ({} rows)", pooled_rows);
    if (expected_t landed = copy_device_to_host(column.offsets, {reservoir_offsets.data(), offsets_count}, gpu_stream);
        !landed)
        return landed;
    if (cuStreamSynchronize(gpu_stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "reservoir offsets readback failed");

    // Draw the whole fill's rows on the host — a keyed bijection over the pool.
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    array<u64_t> record_indices;
    if (record_indices.resize_uninitialized(total_clips) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_clips);
    for (std::size_t segment = 0; segment != total_clips; ++segment)
        record_indices[segment] = epoch_permutation(row_key, data_epoch_t {training_step}, segment, pooled_rows);

    column_view_t const tape {column.payload, {reservoir_offsets.data(), offsets_count}};
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
