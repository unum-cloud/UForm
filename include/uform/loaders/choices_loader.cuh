/**
 *  @file include/uform/loaders/choices_loader.cuh
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The Parquet choices loader: a text state, the single-byte labels legal after it, and each
 *      label's target probability, read from the same rows, one choice segment per row.
 *
 *  Tied to a @b dataset like the paired loader:
 *  @ref unum::uform::loaders::choices_loader_parquet_t::make reads only the shard footers, and
 *  every @ref unum::uform::loaders::choices_loader_parquet_t::fill takes the caller's stream, rank
 *  and step. A fill opens at least two shards unless the dataset is one file, decodes the state,
 *  choices and targets columns of the same rows into one reservoir, lands the choices and targets
 *  on the host, and draws rows by a keyed bijection over the pool, passing over a row with no
 *  choice, more choices than a segment holds, a byte offered twice, or targets of another count.
 *
 *  @section choices_layout One segment per choice
 *
 *  A drawn row becomes one text segment: its state followed by @c byte_mask_k , the position the
 *  byte decoder answers at, a state longer than the row losing its head and never its end. The
 *  segment's row of @c segment_choices holds the row's bytes, padded with @c byte_pad_k , and its
 *  row of @c segment_choice_targets their probabilities. Segments pack back to back and never cross
 *  a row; the segments past the rows' room PAD and make no choice.
 *
 *  The columns hold the state as a string, the choices as a string or binary of one byte per label,
 *  and the targets as a binary of one little-endian Float32 per label, the forms the Parquet reader
 *  decodes.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span`

#include "uform/utilities/vram_nvidia_device.hpp"   // `gpu_alloc` — the per-fill scratch allocator
#include "uform/encoder/multimodal_batch.cuh"       // `multimodal_batch_on_gpu_t`, `byte_pad_k`, `byte_mask_k`
#include "uform/utilities/vram_nvidia_schedule.cuh" // `gpu_stream_t`, `switch_to_gpu_t`, `launch_kernel`
#include "uform/utilities/types.hpp"                // `expected`, `status_t`, `token_id_t`, `epoch_permutation`
#include "uform/loaders/dataset_shards.hpp"         // `parquet_file_t`, `row_group_draw_t`, `expand_glob`
#include "uform/loaders/direct_storage.hpp"         // `gpu_direct_storage_session_t`

namespace unum::uform::loaders {
using namespace unum::uform::utilities;
using namespace unum::uform::encoder;

/** The columns a choices loader reads — a different config is a different loader. */
struct choices_loader_parquet_config_t {

    /** The state column, the text each answer follows. */
    sz::string_view_t state = sz::string_view_t {"state", 5};

    /** The choices column, one legal byte per label. */
    sz::string_view_t choices = sz::string_view_t {"choices", 7};

    /** The targets column, one little-endian Float32 probability per label. */
    sz::string_view_t targets = sz::string_view_t {"targets", 7};
};

/**
 *  @brief The loader object: dataset-scoped and rank-agnostic.
 *
 *  @warning Holds one dataset's shard footers; keep one per input dataset.
 *      @ref unum::uform::loaders::choices_loader_parquet_t::fill draws into the loader's own pick
 *      storage, so one GPU rank fills at a time.
 */
struct choices_loader_parquet_t {

    /**
     *  @brief Index the shards' footers.
     *  @param[in,out] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's Parquet paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed.
     *  @param[in] config The state, choices and targets columns.
     *
     *  @par Statuses:
     *  - @c status_t::invalid_argument_k A shard is empty, or no row holds all three columns.
     */
    [[nodiscard]] static expected<choices_loader_parquet_t> make(gpu_direct_storage_session_t &session,
                                                                 sz::string_view_t path_or_glob, u64_t seed,
                                                                 choices_loader_parquet_config_t config) noexcept;

    /**
     *  @brief Fill @p batches with choice segments on one GPU rank.
     *  @param[in,out] gpu_stream The stream the decode and the packing run on.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches The batches whose segment table, tokens, choices and targets get
     *      populated.
     *
     *  @par Statuses:
     *  - @c status_t::unsupported_k The batches carry no choice columns, or the pool holds
     *    no well-formed row.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, u32_t gpu_rank, u32_t training_step,
                                  std::span<multimodal_batch_on_gpu_t> batches) noexcept;

    /** The device bytes a fill's reservoir can take outside the arena. */
    [[nodiscard]] reservoir_bound_t reservoir() const noexcept { return reservoir_; }

    /** The rows holding every column across the shards, the ones a fill draws. */
    [[nodiscard]] std::size_t rows() const noexcept { return groups_.rows(); }

  private:

    /** The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;

    /** The state, choices and targets columns. */
    sz::string_t state_ {}, choices_ {}, targets_ {};

    /** The run seed keying the file and row draws. */
    u64_t seed_ = 0;

    /** The shards' row groups holding all three columns, and each fill's pick of them. */
    row_group_draw_t groups_ {};

    /** The device bytes a fill's reservoir can take, from the footers. */
    reservoir_bound_t reservoir_ {};
};

} // namespace unum::uform::loaders
