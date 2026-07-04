/**
 *  @brief The byte-level Parquet tokens loader: tied to a @b dataset, not a GPU. @ref make reads only
 *      each shard's Parquet footer (host, once per dataset), so the shard metadata is never replicated
 *      per GPU, and every @ref fill takes the caller's stream and rank, so the @b same loader serves a
 *      different GPU on a different step.
 *  @file cuda_loaders/tokens_loader.cuh
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  Each @ref fill opens @b as many randomly chosen files as it takes to pool at least as many rows as
 *  every batch's segments combined — at least two (unless the whole dataset is one file), so a batch is
 *  never drawn from a single homogeneous shard. The files are drawn by a seeded PRNG keyed by
 *  @c (seed, epoch, rank) , so different GPUs tend to work on different files (occasional overlap is
 *  fine) and a restart reproduces the exact draw. The pooled files decode into @b one device reservoir;
 *  a keyed permutation over the whole pool then hands every batch a scatter of rows across those files —
 *  the pool's row-uniform draw already gives a bigger file proportionally more rows, so the file pick
 *  itself needs no weighting.
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

#include "cuda_core/types.cuh"            // `expected`, `status_t`, `token_id_t`, `epoch_t`, `epoch_permutation`
#include "cuda_core/array.cuh"            // `array` — the loader's footer metadata
#include "cuda_core/multimodal_batch.cuh" // `multimodal_batch_on_gpu_t` — the packing target
#include "cuda_core/schedule.cuh"         // `gpu_stream_t`, `switch_to_gpu_t`, `launch_kernel`, `grid_size_for`
#include "cuda_core/gpu_allocators.cuh"   // `gpu_alloc` — the per-fill scratch allocator

#include "cuda_loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "cuda_loaders/parquet.hpp"        // `parquet_file_t`, `parquet_column_metadata`, `column_view_t`

namespace unum::uform::loaders {

/** @brief The Parquet tokens loader's baked knobs — a different config is a different loader. */
struct tokens_loader_parquet_config_t {
    /** @brief The Parquet column holding the UTF-8 documents. */
    sz::string_view column = sz::string_view {"text", 4};
};

/**
 *  @brief The loader object: dataset-scoped, GPU-agnostic. Borrows the storage session and the shard
 *      paths, and owns only the host footer metadata — no stream, no device buffers, no event. Every
 *      GPU-touching resource is a per- @ref fill local on the passed stream.
 */
struct tokens_loader_parquet_t {

    /** @brief Bind the loader to a dataset and read every shard's footer — once, host-side, no GPU.
     *      @p shards are the dataset's Parquet paths; @p seed keys the random file and row draws. */
    [[nodiscard]] static expected<tokens_loader_parquet_t> make(gpu_direct_storage_session_t const &storage,
                                                                std::span<sz::string_view const> shards, u64_t seed,
                                                                tokens_loader_parquet_config_t config) noexcept;

    /** @brief On @p stream (hence its GPU), open enough randomly chosen files to pool at least this
     *      call's total segments (≥2 files unless the dataset is one), decode them into one reservoir,
     *      and pack @b every batch in @p batches from a keyed permutation of the pool. @p rank keys the
     *      draw so different GPUs diverge onto different files; @p epoch advances the epoch. Nothing about
     *      the shape is configured — every count is read from @p batches . A pure enqueue on @p stream ;
     *      the caller orders on that same stream. */
    [[nodiscard]] expected_t fill(gpu_stream_t const &stream, u32_t rank, epoch_t epoch,
                                  std::span<multimodal_batch_on_gpu_t> batches) noexcept;

  private:
    /** @brief The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t const *storage_ = nullptr;
    /** @brief The dataset's Parquet shard paths — borrowed from the composition root. */
    std::span<sz::string_view const> shards_ {};
    /** @brief The run seed keying the random file and row draws. */
    u64_t seed_ = 0;
    /** @brief The baked knobs. */
    tokens_loader_parquet_config_t config_ {};

    /** @brief @b [shards] each file's footer row count — the coverage accumulation that decides how many
     *      files a fill opens. Read once per dataset; shared across every GPU the loader serves. */
    array<std::size_t> shard_rows_ {};
    /** @brief @b [shards] the document column's footer byte size per shard — the memory-budget signal. */
    array<std::size_t> shard_bytes_ {};
};

} // namespace unum::uform::loaders
