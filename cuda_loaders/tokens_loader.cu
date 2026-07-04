/**
 *  @brief The Parquet tokens loader's bodies and kernels — see the header for the contract. The pack
 *      pair mirrors the mixer's ( @c multimodal_batch.cu ): a serial geometry walk writing offsets +
 *      lengths under the segments-never-cross-rows rule, then a CTA-per-segment gather. The row draw is
 *      a third kernel — a keyed permutation of the pooled files' rows, without replacement.
 *  @file cuda_loaders/tokens_loader.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include <algorithm> // `std::min` (floor), `std::upper_bound` (weighted file pick), `std::find` (dedup)

#include "cuda_loaders/tokens_loader.cuh"

using namespace unum::uform::core;
using namespace unum::uform::loaders;

using u64r1_view_t = nk::tensor_view<u64_t, 1>;
using u64r1_span_t = nk::tensor_span<u64_t, 1>;
/** @brief Rank-2 token span — a batch's [rows, columns] token rectangle; its extents ARE the shape. */
using tokens_matrix_span_t = nk::tensor_span<token_id_t, 2>;

/** @brief Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row permutation
 *      stay decorrelated under one @c (seed, epoch, rank) . */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/**
 *  @brief Draw the whole fill's document rows: a keyed permutation of @c [0, pooled_rows) into
 *      @p record_indices , without replacement (a Feistel bijection, so distinct slots map to distinct
 *      rows). Because the pool is the picked files concatenated, a batch's slice of this permutation is
 *      a scatter of rows across those files.
 *
 *  __grid_size__ grid-stride over @c record_indices ; no dynamic shared memory.
 */
static __global__ void sample_rows_kernel_(u64r1_span_t record_indices, u64_t seed, epoch_t epoch,
                                           std::size_t pooled_rows) {
    std::size_t const total = record_indices.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; slot < total;
         slot += stride)
        record_indices[slot] = epoch_permutation(seed, epoch, slot, pooled_rows);
}

/**
 *  @brief Write one batch's geometry: thread 0 walks the destination slots serially — the repack cursor
 *      is a strict scan — reading each drawn document's byte count through its reservoir row index,
 *      truncating to the row width, and jumping the cursor past any row boundary the segment would
 *      cross. A segment that would overflow the rectangle is dropped loudly (zero length). Slots past
 *      the drawn documents PAD: zero length, repeated offset.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void tokens_geometry_kernel_(u64r1_view_t reservoir_offsets, u64r1_view_t record_indices,
                                               u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                               tokens_matrix_span_t tokens) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    u32_t const columns_per_row = static_cast<u32_t>(tokens.extent(1));
    u32_t const capacity = static_cast<u32_t>(tokens.extent(0) * tokens.extent(1));
    u32_t const live_segments = static_cast<u32_t>(record_indices.extent(0));
    u32_t const total_segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t cursor = 0;
    for (u32_t slot = 0; slot != total_segments; ++slot) {
        u32_t columns = 0;
        if (slot < live_segments) {
            u64_t const source = record_indices[slot];
            u64_t const bytes = reservoir_offsets[source + 1] - reservoir_offsets[source];
            columns = static_cast<u32_t>(std::min<u64_t>(bytes, columns_per_row));
        }
        // A segment never crosses its row: jump the cursor to the next row when it would.
        u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (columns > room_in_row) cursor += room_in_row;
        if (cursor + columns > capacity) {
            printf("uform: loaded batch overflows at slot %u — segment dropped\n", slot);
            columns = 0;
        }
        segment_offsets[slot] = cursor;
        segment_lengths[slot] = columns;
        cursor += columns;
    }
    segment_offsets[total_segments] = cursor;
}

/**
 *  @brief The gather-pack: CTA @c slot copies its document's bytes as token ids — a byte @b is a token
 *      in the 256-entry vocabulary — from the reservoir column into the batch's token rectangle at the
 *      geometry's placement, threads striding the columns.
 *
 *  __grid_size__ @c [documents,1,1] ; a PAD slot (zero length) exits immediately.
 */
static __global__ void tokens_gather_kernel_(b8r1_view_t reservoir_payload, u64r1_view_t reservoir_offsets,
                                             u64r1_view_t record_indices, u32r1_view_t segment_offsets,
                                             u32r1_view_t segment_lengths, tokens_matrix_span_t tokens) {
    u32_t const columns_per_row = static_cast<u32_t>(tokens.extent(1));
    u32_t const slot = blockIdx.x;
    u32_t const columns = segment_lengths[slot];
    if (columns == 0u) return;
    u64_t const source_begin = reservoir_offsets[record_indices[slot]];
    u32_t const destination = segment_offsets[slot];
    u32_t const row = destination / columns_per_row;
    u32_t const column_begin = destination % columns_per_row;
    for (u32_t column = threadIdx.x; column < columns; column += blockDim.x)
        tokens[row, column_begin + column] = static_cast<token_id_t>(
            static_cast<u8_t>(reservoir_payload[source_begin + column]));
}

expected<tokens_loader_parquet_t> tokens_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                                sz::string_view path_or_glob, u64_t seed) noexcept {
    using result_t = expected<tokens_loader_parquet_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths — POSIX glob(3).
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    tokens_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;

    // Footer only — host-side, no GPU, column-agnostic: a running prefix of each file's row count, the
    // weighted file draw's search table. The column is chosen per fill, so this reads the whole-table
    // footer, not a per-column one.
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

expected_t tokens_loader_parquet_t::fill(gpu_stream_t &gpu_stream, tokens_base_loader_parquet_config_t config,
                                         u32_t gpu_rank, u32_t training_step,
                                         std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    if (documents == 0u) return expected_t::fail(status_t::shape_unsupported_k, "the batch carries no segment slots");
    std::size_t const total_segments = unroll * documents;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_rows = row_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};

    // Open distinct files, drawn with probability proportional to their row count — a keyed row position
    // over the whole dataset binary-searches the prefix to its owning file — until the pool covers the
    // fill's segments; never fewer than two, unless the whole dataset is one file.
    array<sz::string_view> picked_paths;
    array<std::size_t> picked_shards;
    if (picked_paths.try_resize(shard_count) != status_t::success_k ||
        picked_shards.try_resize(shard_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
    std::size_t picked_count = 0;
    std::size_t pooled_estimate = 0;
    std::size_t attempt = 0;
    while ((pooled_estimate < total_segments || picked_count < floor_files) && picked_count < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        std::span<std::size_t const> const chosen {picked_shards.data(), picked_count};
        if (std::find(chosen.begin(), chosen.end(), file) != chosen.end()) continue;
        picked_shards[picked_count] = file;
        picked_paths[picked_count] = absolute_paths_[file];
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
        ++picked_count;
    }

    // Decode the picked files into one device reservoir; the true pool size is the decoded row count.
    auto reservoir_read = parquet_file_t::make(*storage_, picked_paths.view().subspan(0, picked_count),
                                               {&config.column, 1}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_segments)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files cover fewer than the {} segments this fill needs",
                                pooled_rows, picked_count, total_segments);

    // Per-fill device index buffer, drawn on the GPU by the keyed permutation of the whole pool.
    array<u64_t, gpu_alloc<u64_t>> record_indices {gpu_alloc<u64_t> {gpu_stream.get()}};
    if (record_indices.try_resize_uninitialized(total_segments) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_segments);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    if (expected_t drawn = launch_kernel(
            reinterpret_cast<void const *>(&sample_rows_kernel_), dim3 {grid_size_for(total_segments), 1u, 1u},
            dim3 {kernel_block_size_default_k, 1u, 1u}, gpu_stream,
            u64r1_span_t {record_indices.data(), total_segments}, row_key, epoch_t {training_step}, pooled_rows);
        !drawn)
        return drawn;

    column_view_t const column = reservoir.column(0);
    u64r1_view_t const reservoir_offsets {column.offsets_.data(), column.offsets_.size()};
    b8r1_view_t const reservoir_payload {column.buffer_.data(), column.buffer_.size()};
    dim3 const one_block {1u, 1u, 1u};
    dim3 const block_shape {kernel_block_size_default_k, 1u, 1u};
    dim3 const document_grid {static_cast<u32_t>(documents), 1u, 1u};

    // Each batch packs its own document slice of the permutation; the rectangle carries its own shape.
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (documents > loaded.segment_lengths.extent(0))
            return expected_t::fail(status_t::shape_unsupported_k, "{} documents exceed batch {}'s segment slots",
                                    documents, batch);
        u64r1_view_t const batch_records {record_indices.data() + batch * documents, documents};
        u32r1_span_t const segment_offsets {loaded.segment_offsets.data(), loaded.segment_offsets.extent(0)};
        u32r1_span_t const segment_lengths {loaded.segment_lengths.data(), loaded.segment_lengths.extent(0)};
        tokens_matrix_span_t const tokens {loaded.tokens.data(), loaded.tokens.extent(0), loaded.tokens.extent(1)};
        expected_t staged = launch_kernel(reinterpret_cast<void const *>(&tokens_geometry_kernel_), one_block,
                                          one_block, gpu_stream, reservoir_offsets, batch_records, segment_offsets,
                                          segment_lengths, tokens);
        if (staged)
            staged = launch_kernel(reinterpret_cast<void const *>(&tokens_gather_kernel_), document_grid, block_shape,
                                   gpu_stream, reservoir_payload, reservoir_offsets, batch_records,
                                   u32r1_view_t {segment_offsets.data(), segment_offsets.extent(0)},
                                   u32r1_view_t {segment_lengths.data(), segment_lengths.extent(0)}, tokens);
        if (!staged) return staged;
    }
    return {};
}
