/**
 *  @brief The byte-level Parquet tokens loader: tied to a @b dataset, not a GPU. @ref make reads only
 *      each shard's Parquet footer (host, once per dataset), so the shard metadata is never replicated
 *      per GPU, and every @ref fill takes the caller's stream and rank, so the @b same loader serves a
 *      different GPU on a different step.
 *  @file src/loaders/tokens_loader.cuh
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  Each @ref fill opens @b as many randomly chosen files as it takes to pool at least as many rows as
 *  every batch's segments combined — at least two (unless the whole dataset is one file), so a batch is
 *  never drawn from a single homogeneous shard. The files are drawn by a seeded hash keyed by
 *  @c (seed, step, rank) with probability @b proportional to each file's row count, so a bigger file is
 *  pooled more often, different GPUs tend to work on different files (occasional overlap is fine), and a
 *  restart reproduces the exact draw. The pooled files decode into @b one device reservoir; a keyed
 *  permutation over the whole pool then hands every batch a scatter of rows across those files.
 *
 *  A big part of the work is on the GPU: the decode (cuDF over GDS), the row-index permutation, and the
 *  geometry + gather packing are all kernels on the caller's stream; only choosing how many files to
 *  open is host arithmetic over the footer row counts. The reservoir and index scratch are re-allocated
 *  per fill and reclaimed once the batches are packed — never hoisted, since shard sizes vary wildly.
 *  The loader fills only this call's batches; the shuffle across batches and across the multimem /
 *  colocated domains is @ref multimodal_batch_mixer_t 's job, downstream.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <span>    // `std::span`

#include "core/types.hpp"             // `expected`, `status_t`, `token_id_t`, `data_epoch_t`, `epoch_permutation`
#include "core/array.hpp"             // `array` — the loader's footer metadata
#include "core/gpu_allocators.hpp"    // `gpu_alloc` — the per-fill scratch allocator
#include "core/schedule.cuh"          // `gpu_stream_t`, `switch_to_gpu_t`, `launch_kernel`, `grid_size_for`
#include "core/multimodal_batch.hpp"  // `multimodal_batch_on_gpu_t` — the packing target
#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "loaders/dataset_shards.hpp" // `parquet_file_t`, `jsonl_file_t`, `expand_glob`, `column_view_t`

namespace unum::uform::loaders {

/** @brief The Parquet tokens loader's baked knobs — a different config is a different loader. */
struct tokens_base_loader_parquet_config_t {
    /** @brief The Parquet's column name holding the UTF-8 documents. */
    sz::string_view_t column = sz::string_view_t {"text", 4};
};

/**
 *  @brief The loader object: dataset-scoped, rank-agnostic, GPU-accelerated. 
 * 
 *  @warning This class manages a considerable amount of metadata per dataset and you should only have one of them
 *      per input dataset. The "fill" methods allocate memory on-demand and are safe to call concurrently for different
 *      GPU ranks with different streams.
 */
struct tokens_loader_parquet_t {

    /**
     *  @brief Failure-safe exception-free constructor, that indexes all the shards.
     *  @note Synchronously reads metadata of a huge pool of files, may be slow.
     * 
     *  @param[in] path_or_glob Dataset's Parquet paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed.
     *  @param[inout] session GPU Direct Storage session for IO.
     */
    [[nodiscard]] static expected<tokens_loader_parquet_t> make( //
        gpu_direct_storage_session_t &session, sz::string_view_t path_or_glob, u64_t seed) noexcept;

    /** 
     *  @brief Fills the next few @p batches of training data for the MLM objective on one GPU rank.
     *  @sa multimodal_batch_mixer_t for subsequent cross-GPU mixing.
     * 
     *  This method opens enough randomly chosen files to sample texts segments to fill all the rows in @p batches
     *  and packs them efficiently, with many segments per row in each matrix. Unless the dataset contains just a
     *  single shard, it takes ≥2 shards to ensure we don't samplehighly uniform data from the same shard.
     * 
     *  @param[in] config Describes which columns we need to take from Parquet tables.
     *  @param[inout] gpu_stream The GPU-specific stream on which the staging and packing should happen.
     *  @param[in] gpu_rank Ensures different GPUs won't take the same rows when sampling the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches Output matrices to be populated with token segments, but not the media modalities.
     */
    [[nodiscard]] expected_t fill(                                            //
        gpu_stream_t &gpu_stream, tokens_base_loader_parquet_config_t config, //
        u32_t gpu_rank, u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept;

    /** @brief The distinct shard paths this rank draws for @p training_step — the keyed weighted file pick,
     *      pure over the loader's footer table (no IO), exposed for observability and cross-rank sampling
     *      conformance. @p target_segments is the pool target ( @c unroll·segments_per_batch ) a @ref fill would
     *      cover; the returned views borrow the loader's owned paths, valid for the loader's lifetime.
     *  @param[in] gpu_rank The GPU rank the draw is keyed by — different ranks draw different shard sets.
     *  @param[in] training_step The step the draw is keyed by.
     *  @param[in] target_segments The number of rows the pooled shards must cover. */
    [[nodiscard]] expected<array<sz::string_view_t>> drawn_shards(u32_t gpu_rank, u32_t training_step,
                                                                  std::size_t target_segments) const noexcept;

  private:
    /** @brief The keyed weighted file pick shared by @ref fill and @ref drawn_shards : draw distinct shards
     *      with probability proportional to their row count, keyed by @c (seed,step,rank) , until the pool
     *      covers @p target_segments rows (≥2 shards unless the dataset is one). Pure over the footer table. */
    [[nodiscard]] expected<array<sz::string_view_t>> pick_shards_(u32_t gpu_rank, u32_t training_step,
                                                                  std::size_t target_segments) const noexcept;
    /** @brief The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;
    /** @brief The dataset's Parquet shard paths — borrowed from the composition root. */
    array<sz::string_t> absolute_paths_ {};
    /** @brief The run seed keying the random file and row draws. */
    u64_t seed_ = 0;
    /** @brief @b [shards+1] cumulative footer row counts — the weighted file draw's search table (an
     *      @c upper_bound maps a keyed row position to its owning file) and, at its back, the dataset's
     *      total rows. Read once per dataset; shared across every GPU the loader serves. */
    array<std::size_t> row_prefix_ {};
};

/** @brief The JSONL tokens loader's baked knobs — a different config is a different loader. */
struct tokens_base_loader_jsonl_config_t {
    /** @brief The JSON field (one object per line) holding the UTF-8 documents. */
    sz::string_view_t field = sz::string_view_t {"text", 4};
};

/**
 *  @brief The JSONL sibling of @ref tokens_loader_parquet_t — a @b separate class (no format punning),
 *      same dataset-scoped, rank-agnostic shape. The one difference is footerless: JSONL carries no row
 *      count, so @ref make weighs files by @b byte size and @ref fill decodes files one at a time until
 *      the reservoir covers the batch, rather than counting rows from a footer up front.
 *
 *  @warning Holds one dataset's shard sizes; keep one per input dataset. @ref fill allocates on demand
 *      and is safe to call concurrently for different GPU ranks on different streams.
 */
struct tokens_loader_jsonl_t {

    /** @brief Index all the shards' byte sizes (POSIX @c stat , host, once — no decode).
     *  @param[inout] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's JSONL paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed. */
    [[nodiscard]] static expected<tokens_loader_jsonl_t> make( //
        gpu_direct_storage_session_t &session, sz::string_view_t path_or_glob, u64_t seed) noexcept;

    /** @brief Fill the next few @p batches for the MLM objective on one GPU rank — @sa
     *      multimodal_batch_mixer_t for the cross-GPU mixing. Draws files weighted by byte size and keyed
     *      by @c (seed, step, rank) , decodes them until the pooled rows cover @p batches (≥2 files unless
     *      the dataset is one), then packs many documents per row.
     *  @param[in] config Which JSON field to take.
     *  @param[inout] gpu_stream The GPU-specific stream the decode and packing run on.
     *  @param[in] gpu_rank Ensures different GPUs draw different files from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches Output matrices populated with token segments. */
    [[nodiscard]] expected_t fill(                                          //
        gpu_stream_t &gpu_stream, tokens_base_loader_jsonl_config_t config, //
        u32_t gpu_rank, u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept;

  private:
    /** @brief The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;
    /** @brief The dataset's JSONL shard paths — owned, expanded from the glob at @ref make . */
    array<sz::string_t> absolute_paths_ {};
    /** @brief The run seed keying the random file and row draws. */
    u64_t seed_ = 0;
    /** @brief @b [shards+1] cumulative @b byte sizes — the weighted file draw's search table. JSONL has
     *      no footer, so bytes (not rows) weigh the draw; the true row counts come from the decode. */
    array<std::size_t> byte_prefix_ {};
};

/** @brief The CSV tokens loader's baked knobs — a different config is a different loader. */
struct tokens_base_loader_csv_config_t {

    /** The CSV column holding the UTF-8 documents: a header name, or a 0-based index string when
     *  @ref header is positional. */
    sz::string_view_t column = sz::string_view_t {"text", 4};

    /** The field separator. */
    char delimiter = ',';

    /** Whether row 0 names the columns. */
    csv_header_t header = csv_header_t::named_k;
};

/**
 *  @brief The CSV sibling of @ref tokens_loader_parquet_t and @ref tokens_loader_jsonl_t — a @b separate
 *      class (no format punning), same dataset-scoped, rank-agnostic shape. Footerless like JSONL: @ref
 *      make weighs files by @b byte size and @ref fill decodes files one at a time until the reservoir
 *      covers the batch.
 *
 *  @warning Holds one dataset's shard sizes; keep one per input dataset. @ref fill allocates on demand
 *      and is safe to call concurrently for different GPU ranks on different streams.
 */
struct tokens_loader_csv_t {

    /** @brief Index all the shards' byte sizes (POSIX @c stat , host, once — no decode).
     *  @param[inout] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's CSV paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed. */
    [[nodiscard]] static expected<tokens_loader_csv_t> make( //
        gpu_direct_storage_session_t &session, sz::string_view_t path_or_glob, u64_t seed) noexcept;

    /** @brief Fill the next few @p batches for the MLM objective on one GPU rank — @sa
     *      multimodal_batch_mixer_t for the cross-GPU mixing. Draws files weighted by byte size and keyed
     *      by @c (seed, step, rank) , decodes them until the pooled rows cover @p batches (≥2 files unless
     *      the dataset is one), then packs many documents per row.
     *  @param[in] config Which CSV column to take, its delimiter, and whether it has a header.
     *  @param[inout] gpu_stream The GPU-specific stream the decode and packing run on.
     *  @param[in] gpu_rank Ensures different GPUs draw different files from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches Output matrices populated with token segments. */
    [[nodiscard]] expected_t fill(                                        //
        gpu_stream_t &gpu_stream, tokens_base_loader_csv_config_t config, //
        u32_t gpu_rank, u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept;

  private:
    /** @brief The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;
    /** @brief The dataset's CSV shard paths — owned, expanded from the glob at @ref make . */
    array<sz::string_t> absolute_paths_ {};
    /** @brief The run seed keying the random file and row draws. */
    u64_t seed_ = 0;
    /** @brief @b [shards+1] cumulative @b byte sizes — the weighted file draw's search table. CSV has no
     *      footer, so bytes (not rows) weigh the draw; the true row counts come from the decode. */
    array<std::size_t> byte_prefix_ {};
};

} // namespace unum::uform::loaders
