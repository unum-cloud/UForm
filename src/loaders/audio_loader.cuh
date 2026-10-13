/**
 *  @brief The Parquet audio loader: tied to a @b dataset, not a GPU — the same shape as
 *      @ref tokens_loader_parquet_t and @ref image_loader_parquet_t .
 *  @file src/loaders/audio_loader.cuh
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  @ref make reads only each shard's Parquet footer once, and every @ref fill takes the caller's stream,
 *  rank, and step, so one loader serves every GPU on every step. Each @ref fill opens @b as many randomly
 *  chosen shards as it takes to pool the clips it needs — at least two unless the dataset is one file —
 *  decodes their blob column into one device reservoir over GDS, draws rows by a keyed bijection over the
 *  whole pool, and decodes the drawn clips into the batch's @c audio_samples frame grid with
 *  @ref decode_audios .
 *
 *  @section audio_raw The loader delivers raw waveform frames — the spectral frontend is learned
 *
 *  A waveform is intrinsically @b 1-D, so audio is not patched (patching only serializes the extra axes of
 *  2-D images and 3-D video). The loader host-decodes each clip to one mono Float32 signal, resamples it,
 *  and chunks it into @b frames of @c chunk_samples raw samples — one frame per column. Each clip becomes one
 *  sub-row segment of frames packed into @c audio_samples ( @b [rows,columns,chunk_samples] Float32), tagged
 *  @c segment_modality[seg]=audio_k ; a frame's coordinate is @c (frame,0,0) , the same position axis text
 *  walks, so audio needs no shape descriptor. The mel / STFT frontend is @b not here: it is a learnable first
 *  layer of the encoder, so this cell carries no cuFFT, no filterbank, and no learnable weights.
 *
 *  @section audio_host_hop The host hop, and why
 *
 *  No GPU audio codec exists, so decode is host-side libav — the one place STYLE sanctions a thread pool,
 *  since decoding independent clips is embarrassingly parallel. @ref decode_audios hops the drawn clips'
 *  encoded bytes device-to-pinned (cuDF keeps the reservoir device-resident), fans them across the borrowed
 *  pool to decode + resample to mono Float32 at @c audio_batch_shape_t::sample_rate , and only then climbs
 *  back onto the GPU to stage the frames into the grid.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span`

#include <forkunion.hpp> // vendor-boundary: `fu::flat_pool_t` — the borrowed CPU-decode pool

#include "core/types.hpp"             // `expected`, `status_t`, `data_epoch_t`, `epoch_permutation`
#include "core/array.hpp"             // `array` — the per-fill index scratch
#include "core/gpu_allocators.hpp"    // `pinned_alloc` — the decode's host tapes
#include "core/schedule.cuh"          // `gpu_stream_t`, `switch_to_gpu_t`, `launch_kernel`
#include "core/multimodal_batch.hpp"  // `multimodal_batch_on_gpu_t` — the decode target
#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "loaders/dataset_shards.hpp" // `parquet_file_t`, `expand_glob`, `column_view_t`

namespace unum::uform::loaders {

/** The host-side worker pool the media decoders fan their host-CPU work across — one per run, owned at
 *  the composition root (the @b only sanctioned host-thread pool), borrowed here. */
using cpu_decode_pool_t = ashvardanian::forkunion::flat_pool_t;

/**
 *  @brief Decode a batch's drawn clip rows into its @c audio_samples frame grid — the audio decode every audio
 *      loader shares (Parquet today, tarball next).
 *
 *  Hops each drawn clip's encoded bytes off the device @p tape to a pinned tape, then for every
 *  @c (clip,representation) pair fans a libav decode + resample to mono Float32 at that representation's
 *  @c sample_rate across @p pool , climbs onto the GPU, and chunks the waveform into frames of
 *  @c chunk_samples raw samples (the grid's @c extent(2) , which every representation must match),
 *  scattering each into its own whole-row sub-row segment. Writes the segment geometry
 *  ( @p segment_offsets / @p segment_lengths ) and @p segment_modality — @c audio_k , 1-D, so no shape
 *  descriptor.
 *
 *  @param[inout] pool The run's CPU-decode worker pool, borrowed for the per-clip libav decode.
 *  @param[in] tape The pooled clip column with its offsets on host — device bytes, host-indexable via
 *      @c tape[row] .
 *  @param[in] drawn_rows The tape rows to decode, one clip each; each clip emits one segment per representation.
 *  @param[in] representations The flat @ref audio_batch_shape_t representations ( @c chunk_samples , @c sample_rate );
 *      the shape is the single source of geometry. Must be non-empty (the loader defaults it to one).
 *  @param[out] audio_samples The @b [rows,columns,chunk_samples] frame grid to chunk each clip's waveform into.
 *  @param[out] segment_offsets The @b [max_segments+1] segment START positions (serial cursor, terminal entry).
 *  @param[out] segment_lengths The @b [max_segments] live frame counts (one clip fills @c columns ; PAD 0).
 *  @param[out] segment_modality The @b [max_segments] per-segment tag — @c audio_k live, @c text_k on PAD.
 *  @param[inout] stream The stream the staging, the copy, and every kernel enqueue on.
 */
[[nodiscard]] expected_t decode_audios(cpu_decode_pool_t &pool, column_view_t tape, std::span<u64_t const> drawn_rows,
                                       std::span<audio_batch_shape_t const> representations,
                                       audio_samples_span_t audio_samples, u32r1_span_t segment_offsets,
                                       u32r1_span_t segment_lengths, modality_span_t segment_modality,
                                       gpu_stream_t &stream) noexcept;

/** The Parquet audio loader's baked knob — just which column; the audio geometry travels on the loader's
 *  @c audio_batch_shape_t (fixed at @ref audio_loader_parquet_t::make ), not here. */
struct audio_loader_parquet_config_t {

    /** The Parquet column holding the encoded clips. Audio datasets store the clip as a struct
     *  @c {bytes,path} , so this is the dotted nested field @c "<column>.bytes" cuDF selects natively — it
     *  arrives as a flat @c list<uint8> , one encoded clip per row. */
    sz::string_view_t column = sz::string_view_t {"audio.bytes", 11};
};

/**
 *  @brief The loader object: dataset-scoped, rank-agnostic, GPU-accelerated — the audio sibling of
 *      @ref tokens_loader_parquet_t and @ref image_loader_parquet_t .
 *
 *  @warning Holds one dataset's shard footers; keep one per input dataset. @ref fill allocates its scratch on
 *      demand and is safe to call concurrently for different GPU ranks on different streams.
 */
struct audio_loader_parquet_t {

    /**
     *  @brief Failure-safe exception-free constructor that indexes all the shards' footers (host, once).
     *  @param[inout] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's Parquet paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed.
     *  @param[inout] pool The run's CPU-decode worker pool, owned at the composition root and borrowed here;
     *      the loader never spawns threads of its own.
     *  @param[in] audio The audio modality's own geometry slice ( @c chunk_samples , @c sample_rate ) — the
     *      single source, and all this loader needs; it never sees the sibling modalities' shapes.
     */
    [[nodiscard]] static expected<audio_loader_parquet_t> make(gpu_direct_storage_session_t &session,
                                                               sz::string_view_t path_or_glob, u64_t seed,
                                                               cpu_decode_pool_t &pool,
                                                               audio_batch_shape_t audio) noexcept;

    /**
     *  @brief Fill the next few @p batches ' @c audio_samples grid on one GPU rank, one decoded clip per segment.
     *  @sa multimodal_batch_mixer_t for the subsequent cross-GPU mixing.
     *
     *  Opens ≥2 randomly chosen shards (keyed by @c (seed,step,rank) ), pools their clip column into one
     *  reservoir, draws a scatter of rows across it, and decodes each batch's drawn clips into its frame grid —
     *  each clip one sub-row segment of @c columns raw-sample frames — writing the segment geometry and
     *  @c segment_modality .
     *
     *  @param[inout] gpu_stream The GPU-specific stream the staging, decode, and chunking run on.
     *  @param[in] config Which Parquet column holds the clips.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches Output batches whose @c audio_samples grid and segment geometry get populated.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, audio_loader_parquet_config_t config, u32_t gpu_rank,
                                  u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept;

  private:

    /** The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;

    /** The run's CPU-decode worker pool — borrowed from the composition root, handed to the decode;
     *  never owned, never spawned here. */
    cpu_decode_pool_t *pool_ = nullptr;

    /** The dataset's Parquet shard paths, sorted by the glob — owned. */
    array<sz::string_t> absolute_paths_ {};

    /** The run seed keying the random file and row draws. */
    u64_t seed_ = 0;

    /** @b [shards+1] cumulative footer row counts — the weighted file draw's search table (an
     *  @c upper_bound maps a keyed row position to its owning file) and, at its back, the dataset's total
     *  rows. Read once per dataset; shared across every GPU the loader serves. */
    array<std::size_t> row_prefix_ {};

    /** The audio modality's geometry ( @c chunk_samples , @c sample_rate ) every @ref fill hands the
     *  decode — fixed at @ref make . */
    audio_batch_shape_t audio_ {};
};

} // namespace unum::uform::loaders
