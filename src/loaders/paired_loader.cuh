/**
 *  @file src/loaders/paired_loader.cuh
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The Parquet paired loader: an anchor text and its positive, an image, clip, recording
 *      or another text read from the same rows, each pair one bag, so a single fill carries both
 *      halves and the relation between them, and optionally a hard negative per pair.
 *
 *  Tied to a @b dataset like the single-modality loaders: @ref make reads only the shard footers,
 *  and every @ref fill takes the caller's stream, rank and step. A fill opens at least two shards
 *  unless the dataset is one file, decodes the anchor, positive and negative columns of the same
 *  rows into one reservoir, and draws rows by a keyed bijection over the pool. A row whose anchor
 *  or text positive is a list contributes one of its elements, drawn by a key of its own, so every
 *  element of a row recurs across steps. A row's negatives are a list too: strings for a text
 *  positive, and shard-local row indices into the positive column for media.
 *
 *  @section paired_layout One bag per pair, a negative alone
 *
 *  A batch of @c rows rows and @c segments segments takes @c min(rows,segments)/2 pairs, or a third
 *  of that with negatives. Pair @c i 's segments are its positive, then its anchor, both tagged bag
 *  @c i , the contiguous run the mixer routes whole, then its negative tagged a bag of its own past
 *  the pairs. A negative attracts nothing and repels through the ambient weight. Every segment
 *  spans at most one row, so any order of those bags packs into the rows a destination has. Inside
 *  the loaded batch the positives take the leading rows, the negatives the next, and the anchors
 *  the trailing rows they need, each through the decode its modality already has:
 *  @ref decode_images , @ref decode_audios or @ref decode_videos , and @ref pack_tokens for text.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span`

#include "core/types.hpp"             // `expected`, `status_t`, `epoch_permutation`
#include "core/array.hpp"             // `array`
#include "core/multimodal_batch.cuh"  // `multimodal_batch_on_gpu_t`, `multimodal_batch_shape_t`
#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "loaders/dataset_shards.hpp" // `parquet_file_t`, `expand_glob`, `column_view_t`
#include "loaders/tokens_loader.cuh"  // `pack_tokens`
#include "loaders/image_loader.cuh"   // `decode_images`, `nvjpeg_session_t`, `cpu_decode_pool_t`
#include "loaders/audio_loader.cuh"   // `decode_audios`
#include "loaders/video_loader.cuh"   // `decode_videos`, `nvdec_session_t`

namespace unum::uform::loaders {

/** The columns a paired loader reads — a different config is a different loader. */
struct paired_loader_parquet_config_t {

    /** The caption column: a string, or a list of strings of which a fill draws one per row. */
    sz::string_view_t anchor = sz::string_view_t {"caption", 7};

    /** The positive column: the encoded blob, the @c {bytes,path} struct whose @c bytes leaf holds
     *  it, or for a text pair a string or a list of strings. */
    sz::string_view_t positive = sz::string_view_t {"image", 5};

    /** The hard-negative column, empty for none: a list of strings for a text pair, and a list of
     *  shard-local rows of the positive column for media. */
    sz::string_view_t negative {};

    /** The bytes every anchor starts with, and every positive and negative text. */
    sz::string_view_t anchor_prefix {}, positive_prefix {};
};

/**
 *  @brief The loader object: dataset-scoped, rank-agnostic, one media modality per dataset.
 *
 *  @warning Holds one dataset's shard footers; keep one per input dataset. @ref fill allocates its
 *      scratch on demand and is safe to call concurrently for different GPU ranks on different
 *      streams.
 */
struct paired_loader_parquet_t {

    /**
     *  @brief Index the shards' footers and resolve the media column to its blob leaf.
     *  @param[inout] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's Parquet paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed.
     *  @param[inout] pool The run's CPU-decode worker pool, borrowed for the host decodes.
     *  @param[in] config The anchor, positive and negative columns and the text prefixes.
     *  @param[in] modality The positive's modality: @c image_k , @c audio_k , @c video_k or
     *      @c text_k .
     *  @param[in] shape The run's batch shape, whose slice of @p modality is the decode's geometry.
     */
    [[nodiscard]] static expected<paired_loader_parquet_t> make(
        gpu_direct_storage_session_t &session, sz::string_view_t path_or_glob, u64_t seed, cpu_decode_pool_t &pool,
        paired_loader_parquet_config_t config, modality_t modality, multimodal_batch_shape_t const &shape) noexcept;

    /**
     *  @brief Fill @p batches with caption-image pairs on one GPU rank.
     *  @param[inout] gpu_stream The stream the decode, the packing and the pairing run on.
     *  @param[inout] decoder This GPU's nvJPEG session.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches The batches whose segment table, bags, tokens and images get populated.
     *  @retval status_t::invalid_input_k The loader's media are not images.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, nvjpeg_session_t &decoder, u32_t gpu_rank,
                                  u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept;

    /**
     *  @brief Fill @p batches with caption-clip pairs on one GPU rank.
     *  @param[inout] gpu_stream The stream the decode, the packing and the pairing run on.
     *  @param[inout] decoder This GPU's NVDEC session.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches The batches whose segment table, bags, tokens and videos get populated.
     *  @retval status_t::invalid_input_k The loader's media are not videos.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, nvdec_session_t &decoder, u32_t gpu_rank,
                                  u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept;

    /**
     *  @brief Fill @p batches with caption-recording or text-text pairs on one GPU rank.
     *  @param[inout] gpu_stream The stream the decode, the packing and the pairing run on.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches The batches whose segment table, bags, tokens and audio samples get
     *      populated.
     *  @retval status_t::invalid_input_k The loader's positives are images or videos.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                  std::span<multimodal_batch_on_gpu_t> batches) noexcept;

    /** The modality of the loader's positives. */
    [[nodiscard]] modality_t modality() const noexcept { return modality_; }

    /** The device bytes a fill's reservoir can take outside the arena. */
    [[nodiscard]] reservoir_bound_t reservoir() const noexcept { return reservoir_; }

  private:

    /** Read and draw the pairs, decode the positives and negatives through @p decode , pack the
     *  anchors, and pair the segments. */
    template <typename decode_type_>
    [[nodiscard]] expected_t fill_(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                   std::span<multimodal_batch_on_gpu_t> batches, decode_type_ &&decode) noexcept;

    /** The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;

    /** The run's CPU-decode worker pool — borrowed, never spawned here. */
    cpu_decode_pool_t *pool_ = nullptr;

    /** The dataset's Parquet shard paths, sorted by the glob. */
    array<sz::string_t> absolute_paths_ {};

    /** The caption column, the positive column resolved to its blob leaf, and the negative column,
     *  empty for none. */
    sz::string_t anchor_ {}, positive_ {}, negative_ {};

    /** The bytes every anchor starts with, and every positive and negative text. */
    sz::string_t anchor_prefix_ {}, positive_prefix_ {};

    /** The run seed keying the file, row and caption draws. */
    u64_t seed_ = 0;

    /** @b [shards+1] cumulative footer row counts, the weighted file draw's search table. */
    array<std::size_t> row_prefix_ {};

    /** The device bytes a fill's reservoir can take, from the footers. */
    reservoir_bound_t reservoir_ {};

    /** The modality of the positive column. */
    modality_t modality_ = modality_t::image_k;

    /** The run's batch geometry, read for the slice of @ref modality_ . */
    multimodal_batch_shape_t shape_ {};
};

} // namespace unum::uform::loaders
