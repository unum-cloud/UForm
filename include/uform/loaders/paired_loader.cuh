/**
 *  @file include/uform/loaders/paired_loader.cuh
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
 *  positive, and shard-local row indices into the positive column for media, whose row passes over
 *  when the fill drew no row group holding its negative.
 *
 *  @section paired_layout One bag per pair or triplet
 *
 *  A batch of @c segments segments draws @c segments/2 pairs, or a third of that with negatives,
 *  and admits them whole in keyed order while their rows fit: captions pack back to back, and a
 *  media positive or negative takes at most one row. Pair @c i 's segments are its positive, its
 *  anchor and any negative, all tagged bag @c i , the contiguous run the mixer routes whole; the
 *  segments past the admitted pairs PAD. Every segment spans at most one row. Inside the loaded
 *  batch the positives take the leading rows, the negatives the next, and the anchors the trailing
 *  rows they need, each through the decode its modality already has:
 *  @ref decode_images , @ref decode_audios or @ref decode_videos , and @ref pack_tokens for text.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span`

#include "uform/encoder/multimodal_batch.cuh" // `multimodal_batch_on_gpu_t`, `multimodal_batch_shape_t`
#include "uform/utilities/types.hpp"          // `expected`, `status_t`, `epoch_permutation`
#include "uform/loaders/audio_loader.cuh"   // `decode_audios`
#include "uform/loaders/dataset_shards.hpp" // `parquet_file_t`, `expand_glob`, `column_view_t`
#include "uform/loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "uform/loaders/image_loader.cuh"   // `decode_images`, `nvjpeg_session_t`
#include "uform/loaders/tokens_loader.cuh"  // `pack_tokens`
#include "uform/loaders/video_loader.cuh"   // `decode_videos`, `nvdec_session_t`

namespace unum::uform::loaders {
using namespace unum::uform::utilities;
using namespace unum::uform::encoder;

/** The late-interaction side each role of a bag plays, from the mixture entry's @c query key. */
struct role_sides_t {
    late_interaction_side_t anchor = late_interaction_side_t::none_k;
    late_interaction_side_t positive = late_interaction_side_t::none_k;
    late_interaction_side_t negative = late_interaction_side_t::none_k;
};

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

    /** The late-interaction side of each role; all @c none_k when the entry names no @c query . */
    role_sides_t sides {};
};

/**
 *  @brief The loader object: dataset-scoped, rank-agnostic, one media modality per dataset.
 *
 *  @warning Holds one dataset's shard footers; keep one per input dataset. @ref fill draws into the
 *      loader's own pick storage, so one GPU rank fills at a time.
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
    [[nodiscard]] static expected<paired_loader_parquet_t> make(gpu_direct_storage_session_t &session,
                                                                sz::string_view_t path_or_glob, u64_t seed,
                                                                pool_view_t pool, paired_loader_parquet_config_t config,
                                                                modality_t modality,
                                                                multimodal_batch_shape_t const &shape) noexcept;

    /**
     *  @brief Fill @p batches with caption-image pairs on one GPU rank.
     *  @param[inout] gpu_stream The stream the decode, the packing and the pairing run on.
     *  @param[inout] decoder This GPU's nvJPEG session.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches The batches whose segment table, bags, tokens and images get populated.
     *  @retval status_t::invalid_argument_k The loader's media are not images.
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
     *  @retval status_t::invalid_argument_k The loader's media are not videos.
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
     *  @retval status_t::invalid_argument_k The loader's positives are images or videos.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                  std::span<multimodal_batch_on_gpu_t> batches) noexcept;

    /** The modality of the loader's positives. */
    [[nodiscard]] modality_t modality() const noexcept { return modality_; }

    /** The device bytes a fill's reservoir can take outside the arena. */
    [[nodiscard]] reservoir_bound_t reservoir() const noexcept { return reservoir_; }

    /** The pairs one batch row of @p columns holds, which bounds a fill's draws and the reservoir
     *  charged for them: as many media items as @ref media_cells places, a media negative taking
     *  rows of its own, or texts at the widest row group's mean length. */
    [[nodiscard]] f64_t pairs_per_row(std::size_t columns) const noexcept {
        if (u32_t const cells = media_cells(modality_, static_cast<u32_t>(columns)); cells != 0u)
            return static_cast<f64_t>(columns / cells) / (negative_.empty() ? 1.0 : 2.0);
        std::size_t const pair_cells = reservoir_.row_bytes / reservoir_bound_t::copies_k;
        return pair_cells != 0 ? static_cast<f64_t>(columns) / static_cast<f64_t>(pair_cells) : 1.0;
    }

    /** The rows holding every role column across the shards, the ones a fill draws. */
    [[nodiscard]] std::size_t rows() const noexcept { return groups_.rows(); }

  private:

    /** Read and draw the pairs, decode the positives and negatives through @p decode , pack the
     *  anchors, and pair the segments. */
    template <typename decode_type_>
    [[nodiscard]] expected_t fill_(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                   std::span<multimodal_batch_on_gpu_t> batches, decode_type_ &&decode) noexcept;

    /** The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;

    /** The run's CPU-decode worker pool — borrowed, never spawned here. */
    pool_view_t pool_ {nullptr};

    /** The caption column, the positive column resolved to its blob leaf, and the negative column,
     *  empty for none. */
    sz::string_t anchor_ {}, positive_ {}, negative_ {};

    /** The bytes every anchor starts with, and every positive and negative text. */
    sz::string_t anchor_prefix_ {}, positive_prefix_ {};

    /** The late-interaction side each role's segments carry. */
    role_sides_t sides_ {};

    /** The sample-key column, which names one media across rows, empty when shards carry none. */
    sz::string_t key_ {};

    /** The run seed keying the file, row and caption draws. */
    u64_t seed_ = 0;

    /** The shards' row groups holding every role column, and each fill's pick of them. */
    row_group_draw_t groups_ {};

    /** Per drawable shard, each row group's first shard-local row and then the shard's rows, which
     *  place a drawn group's rows and a media negative's row. */
    st::basic_vector<std::size_t> group_starts_ {};

    /** @b [shards+1] each drawable shard's first entry in @ref group_starts_ . */
    st::basic_vector<std::size_t> shard_starts_ {};

    /** @b [shards] each drawable shard path's hash, which a row's sample key folds with its
     *  shard-local row, so every entry that reads one shard keys its rows alike. */
    st::basic_vector<u64_t> shard_keys_ {};

    /** The device bytes a fill's reservoir can take, from the footers. */
    reservoir_bound_t reservoir_ {};

    /** The modality of the positive column. */
    modality_t modality_ = modality_t::image_k;

    /** The run's batch geometry, read for the slice of @ref modality_ . */
    multimodal_batch_shape_t shape_ {};
};

} // namespace unum::uform::loaders
