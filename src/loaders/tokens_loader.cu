/**
 *  @brief The Parquet tokens loader's bodies and kernels — see the header for the contract.
 *  @file src/loaders/tokens_loader.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  The pack pair mirrors the mixer's ( @c multimodal_batch.cu ): a serial geometry walk writing offsets +
 *  lengths under the segments-never-cross-rows rule, then a CTA-per-segment gather. The row draw is a third
 *  kernel — a keyed permutation of the pooled files' rows, without replacement.
 */
#include <algorithm> // `std::min` (floor), `std::upper_bound` (weighted group pick), `std::find` (dedup), `std::sort`

#include "uform/loaders/tokens_loader.cuh"

using namespace unum::uform::encoder;
using namespace unum::uform::utilities;
using namespace unum::uform::loaders;
namespace ul = unum::uform::loaders;

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row permutation
 *  stay decorrelated under one @c (seed, epoch, rank) . */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/**
 *  @brief Draw the whole fill's document rows: a keyed permutation of @c [0, pooled_rows) into
 *      @p record_indices , without replacement (a Feistel bijection, so distinct draw positions map to
 *      distinct rows).
 *
 *  Because the pool is the picked files concatenated, a batch's slice of this permutation is a scatter of
 *  rows across those files.
 *
 *  __grid_size__ grid-stride over @c record_indices ; no dynamic shared memory.
 */
static __global__ void sample_rows_kernel_(u64r1_span_t record_indices, u64_t seed, data_epoch_t data_epoch,
                                           std::size_t pooled_rows) {
    std::size_t const total = record_indices.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t draw = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; draw < total;
         draw += stride)
        record_indices[draw] = epoch_permutation(seed, data_epoch, draw, pooled_rows);
}

/**
 *  @brief Write one batch's geometry: thread 0 walks the destination segments serially — the repack cursor
 *      is a strict scan.
 *
 *  It reads each drawn document's byte count through its reservoir row index, truncates to the row width,
 *  and jumps the cursor past any row boundary the segment would cross. A segment that would
 *  overflow the rectangle is dropped (zero length) and counted into @p dropped . Segments past the
 *  drawn documents PAD: zero length, repeated offset.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void tokens_geometry_kernel_(u64r1_view_t reservoir_offsets, u64r1_view_t record_indices,
                                               std::size_t prefix_bytes, u32r1_span_t segment_offsets,
                                               u32r1_span_t segment_lengths, modality_span_t segment_modality,
                                               u32r1_span_t dropped, tokens_matrix_span_t tokens) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    u32_t const columns_per_row = static_cast<u32_t>(tokens.extent(1));
    u32_t const capacity = static_cast<u32_t>(tokens.extent(0) * tokens.extent(1));
    u32_t const live_segments = static_cast<u32_t>(record_indices.extent(0));
    u32_t const total_segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t cursor = 0, overflowed = 0;
    for (u32_t segment_index = 0; segment_index != total_segments; ++segment_index) {
        u32_t columns = 0;
        if (segment_index < live_segments) {
            u64_t const source = record_indices[segment_index];
            u64_t const bytes = prefix_bytes + reservoir_offsets[source + 1] - reservoir_offsets[source];
            columns = static_cast<u32_t>(std::min<u64_t>(bytes, columns_per_row));
        }
        // A segment never crosses its row: jump the cursor to the next row when it would.
        u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (columns > room_in_row) cursor += room_in_row;
        if (cursor + columns > capacity) ++overflowed, columns = 0;
        segment_offsets[segment_index] = cursor;
        segment_lengths[segment_index] = columns;
        // Every fill owns the whole segment table — a reused batch may hold another modality's tags, and
        // a stale tag routes token bytes into a media head downstream (text_k doubles as PAD at length 0).
        segment_modality[segment_index] = modality_t::text_k;
        cursor += columns;
    }
    segment_offsets[total_segments] = cursor;
    dropped[0] += overflowed;
}

/**
 *  @brief The gather-pack: each CTA copies its document's bytes as token ids — a byte @b is a token
 *      in the 256-entry vocabulary — from the reservoir column into the batch's token rectangle at the
 *      geometry's placement, threads striding the columns.
 *
 *  __grid_size__ @c [documents,1,1] ; a PAD segment (zero length) exits immediately.
 */
static __global__ void tokens_gather_kernel_(b8r1_view_t reservoir_payload, u64r1_view_t reservoir_offsets,
                                             u64r1_view_t record_indices, b8r1_view_t prefix,
                                             u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                             tokens_matrix_span_t tokens) {
    u32_t const columns_per_row = static_cast<u32_t>(tokens.extent(1));
    u32_t const segment_index = blockIdx.x;
    u32_t const columns = segment_lengths[segment_index];
    if (columns == 0u) return;
    u64_t const source_begin = reservoir_offsets[record_indices[segment_index]];
    u32_t const destination = segment_offsets[segment_index];
    u32_t const row = destination / columns_per_row;
    u32_t const column_begin = destination % columns_per_row;
    std::size_t const prefix_bytes = prefix.extent(0);
    for (u32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        b8_t const byte = column < prefix_bytes ? prefix[column]
                                                : reservoir_payload[source_begin + column - prefix_bytes];
        tokens[row, column_begin + column] = static_cast<token_id_t>(static_cast<u8_t>(byte));
    }
}

expected_t ul::pack_tokens(column_view_t texts, u64r1_view_t elements, b8r1_view_t prefix, tokens_matrix_span_t tokens,
                           u32r1_span_t segment_offsets, u32r1_span_t segment_lengths, modality_span_t segment_modality,
                           u32r1_span_t dropped, gpu_stream_t &gpu_stream) noexcept {
    if (elements.extent(0) > segment_lengths.extent(0) || segment_offsets.extent(0) != segment_lengths.extent(0) + 1u ||
        segment_modality.extent(0) != segment_lengths.extent(0))
        return expected_t::fail(status_t::shape_unsupported_k, "{} documents exceed a table of {} segments",
                                elements.extent(0), segment_lengths.extent(0));
    if (elements.empty()) return {};
    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");
    dim3 const one_block {1u, 1u, 1u};
    expected_t packed = launch_kernel(&tokens_geometry_kernel_, one_block, one_block, gpu_stream, texts.offsets,
                                      elements, prefix.extent(0), segment_offsets, segment_lengths, segment_modality,
                                      dropped, tokens);
    if (packed)
        packed = launch_kernel(&tokens_gather_kernel_, dim3 {static_cast<u32_t>(elements.extent(0)), 1u, 1u},
                               dim3 {kernel_block_size_default_k, 1u, 1u}, gpu_stream, texts.payload, texts.offsets,
                               elements, prefix, segment_offsets, segment_lengths, tokens);
    return packed;
}

/** The format-agnostic pack: draw the whole fill's rows over @p reservoir_column on the GPU
 *  (a keyed permutation of @p pool_rows , without replacement) and pack @b every batch from it. Both
 *  the Parquet and JSONL loaders share this once they hold a decoded reservoir — only how the
 *  reservoir is obtained differs (a footer-known Parquet read vs a JSONL decode + concat). A pure
 *  enqueue on @p gpu_stream . */
[[nodiscard]] static expected_t pack_reservoir_(column_view_t reservoir_column, std::size_t pool_rows, u64_t row_key,
                                                data_epoch_t data_epoch, std::span<multimodal_batch_on_gpu_t> batches,
                                                gpu_stream_t &gpu_stream) noexcept {
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    std::size_t const total_segments = unroll * documents;

    // Per-fill device index buffer, drawn on the GPU by the keyed permutation of the whole pool.
    st::basic_vector<u64_t, gpu_alloc<u64_t>> record_indices {gpu_alloc<u64_t> {gpu_stream.get()}};
    if (st::failed(record_indices.resize(st::assume_uninitialized, total_segments)))
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_segments);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");
    nk::tensor_span<u64_t, 2> const batch_records {record_indices.data(), unroll, documents};
    if (expected_t drawn = launch_kernel(&sample_rows_kernel_, dim3 {grid_size_for(total_segments), 1u, 1u},
                                         dim3 {kernel_block_size_default_k, 1u, 1u}, gpu_stream,
                                         batch_records.flatten<1>(), row_key, data_epoch, pool_rows);
        !drawn)
        return drawn;

    // Each batch packs its own row of the permutation; the rectangle carries its shape.
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (expected_t packed = pack_tokens(reservoir_column, batch_records[batch, nk::slice], {}, loaded.tokens.span(),
                                            loaded.segment_offsets.span(), loaded.segment_lengths.span(),
                                            loaded.segment_modality.span(), loaded.dropped.span(), gpu_stream);
            !packed)
            return packed;
        // A document names no role, key, prefix or late-interaction side; none keeps stale bytes.
        expected_t cleared = zero_async(loaded.segment_side.span(), gpu_stream);
        if (cleared) cleared = zero_async(loaded.segment_role.span(), gpu_stream);
        if (cleared) cleared = zero_async(loaded.segment_key.span(), gpu_stream);
        if (cleared) cleared = zero_async(loaded.segment_prefix_length.span(), gpu_stream);
        if (!cleared) return cleared;
    }
    return {};
}

expected<tokens_loader_parquet_t> tokens_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                                sz::string_view_t path_or_glob, u64_t seed) noexcept {
    using result_t = expected<tokens_loader_parquet_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths — POSIX glob(3).
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    tokens_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;

    // Footer only — host-side, no GPU, column-agnostic: a running prefix of each row group's row
    // count, the weighted draw's search table. The column is chosen per fill, so this reads the
    // whole-table footer, not a per-column one.
    if (st::failed(loader.group_prefix_.push_back(0)))
        return result_t::fail(status_t::bad_alloc_k, "row-group prefix allocation failed");
    for (std::size_t index = 0; index != loader.absolute_paths_.size(); ++index) {
        auto metadata = parquet_file_t::table_metadata(session, loader.absolute_paths_[index]);
        if (!metadata) return result_t::fail(std::move(metadata));
        if (metadata->rows == 0)
            return result_t::fail(status_t::invalid_input_k, "shard {} is empty", loader.absolute_paths_[index]);
        for (std::size_t group = 0; group != metadata->group_rows.size(); ++group) {
            if (metadata->group_rows[group] == 0) continue;
            if (st::failed(loader.group_prefix_.push_back(loader.group_prefix_[loader.group_shard_.size()] +
                                                          metadata->group_rows[group])) ||
                st::failed(loader.group_shard_.push_back(static_cast<u32_t>(index))) ||
                st::failed(loader.group_index_.push_back(static_cast<u32_t>(group))))
                return result_t::fail(status_t::bad_alloc_k, "row-group table append failed");
        }
        loader.reservoir_.charge_groups(*metadata);
    }
    return result_t {std::move(loader)};
}

expected_t tokens_loader_parquet_t::fill(gpu_stream_t &gpu_stream, tokens_base_loader_parquet_config_t config,
                                         u32_t gpu_rank, u32_t training_step,
                                         std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    if (documents == 0u)
        return expected_t::fail(status_t::shape_unsupported_k, "the batch carries no segment capacity");
    std::size_t const total_segments = unroll * documents;

    // Pick row groups with the keyed weighted draw shared with `drawn_shards`, then decode only
    // them into one device reservoir; the true pool size is the decoded row count.
    auto picked = pick_groups_(gpu_rank, training_step, total_segments);
    if (!picked) return expected_t::fail(std::move(picked));
    constexpr splitmix64_hash_t hash {};

    auto reservoir_read = parquet_file_t::make(*storage_, picked->paths, {&config.column, 1}, picked->groups,
                                               gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_segments)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files cover fewer than the {} segments this fill needs",
                                pooled_rows, picked->paths.size(), total_segments);

    // Draw the pool's rows and pack every batch — shared with the JSONL loader.
    return pack_reservoir_(reservoir.column(0), pooled_rows, hash(seed_, gpu_rank, row_draw_salt_k),
                           data_epoch_t {training_step}, batches, gpu_stream);
}

expected<tokens_loader_parquet_t::picked_groups_t> tokens_loader_parquet_t::pick_groups_(
    u32_t gpu_rank, u32_t training_step, std::size_t target_segments) const noexcept {
    using result_t = expected<picked_groups_t>;
    std::size_t const group_count = group_shard_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, absolute_paths_.size());
    std::size_t const total_rows = group_prefix_[group_count];
    constexpr splitmix64_hash_t hash {};

    // A keyed row position over the whole dataset binary-searches the prefix to its owning group;
    // distinct groups accumulate until the pool covers the target, then only a new shard's group
    // meets the floor of two, so a fill reads at most one group past its rows and one more.
    picked_groups_t picked;
    st::basic_vector<std::size_t> picked_ids;
    st::basic_vector<u32_t> picked_shards;
    std::size_t pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < target_segments || picked.paths.size() < floor_files) &&
           picked_ids.size() < group_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const group = static_cast<std::size_t>(
            std::upper_bound(group_prefix_.begin(), group_prefix_.end(), position) - group_prefix_.begin() - 1);
        if (std::find(picked_ids.begin(), picked_ids.end(), group) != picked_ids.end()) continue;
        u32_t const shard = group_shard_[group];
        auto const found = std::find(picked_shards.begin(), picked_shards.end(), shard);
        if (found != picked_shards.end() && pooled_estimate >= target_segments) continue;
        std::size_t const pick = static_cast<std::size_t>(found - picked_shards.begin());
        if (found == picked_shards.end() && (st::failed(picked_shards.push_back(u32_t {shard})) ||
                                             st::failed(picked.paths.push_back(absolute_paths_[shard])) ||
                                             st::failed(picked.groups.push_back(st::basic_vector<u32_t> {}))))
            return result_t::fail(status_t::bad_alloc_k, "shard pick append failed");
        if (st::failed(picked_ids.push_back(std::size_t {group})) ||
            st::failed(picked.groups[pick].push_back(u32_t {group_index_[group]})))
            return result_t::fail(status_t::bad_alloc_k, "row-group pick append failed");
        pooled_estimate += group_prefix_[group + 1] - group_prefix_[group];
    }
    for (st::basic_vector<u32_t> &groups : picked.groups)
        std::sort(groups.begin(), groups.end()); // cuDF reads them ascending
    return result_t {std::move(picked)};
}

expected<st::basic_vector<sz::string_view_t>> tokens_loader_parquet_t::drawn_shards(
    u32_t gpu_rank, u32_t training_step, std::size_t target_segments) const noexcept {
    auto picked = pick_groups_(gpu_rank, training_step, target_segments);
    if (!picked) return expected<st::basic_vector<sz::string_view_t>>::fail(std::move(picked));
    return expected<st::basic_vector<sz::string_view_t>> {std::move(picked->paths)};
}

/** Weigh footerless @p paths by byte size, POSIX @c stat , into @p byte_prefix , the weighted file
 *  draw's running prefix, and charge each file's bytes to @p reservoir . */
[[nodiscard]] static expected_t weigh_by_bytes_(std::span<sz::string_t const> paths,
                                                st::basic_vector<std::size_t> &byte_prefix,
                                                reservoir_bound_t &reservoir) noexcept {
    if (st::failed(byte_prefix.resize(paths.size() + 1)))
        return expected_t::fail(status_t::bad_alloc_k, "shard prefix allocation failed ({} shards)", paths.size());
    byte_prefix[0] = 0;
    for (std::size_t index = 0; index != paths.size(); ++index) {
        auto bytes = file_size_bytes(paths[index]);
        if (!bytes) return expected_t::fail(std::move(bytes));
        if (*bytes == 0) return expected_t::fail(status_t::invalid_input_k, "shard {} is empty", paths[index]);
        byte_prefix[index + 1] = byte_prefix[index] + *bytes;
        reservoir.charge(parquet_length_metadata_t {0, *bytes});
    }
    return {};
}

expected<tokens_loader_jsonl_t> tokens_loader_jsonl_t::make(gpu_direct_storage_session_t &session,
                                                            sz::string_view_t path_or_glob, u64_t seed) noexcept {
    using result_t = expected<tokens_loader_jsonl_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths — POSIX readdir.
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    tokens_loader_jsonl_t loader;
    loader.storage_ = &session;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;

    // No footer — weigh files by byte size (POSIX stat), a running prefix for the weighted file draw.
    if (expected_t weighed = weigh_by_bytes_(loader.absolute_paths_, loader.byte_prefix_, loader.reservoir_); !weighed)
        return result_t::fail(std::move(weighed));
    return result_t {std::move(loader)};
}

expected_t tokens_loader_jsonl_t::fill(gpu_stream_t &gpu_stream, tokens_base_loader_jsonl_config_t config,
                                       u32_t gpu_rank, u32_t training_step,
                                       std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    if (documents == 0u)
        return expected_t::fail(status_t::shape_unsupported_k, "the batch carries no segment capacity");
    std::size_t const total_segments = unroll * documents;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_bytes = byte_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};

    // Pick distinct files weighted by BYTE size (JSONL has no footer), decoding each until the pooled
    // rows cover the fill — never fewer than two, unless the whole dataset is one file.
    st::basic_vector<jsonl_file_t> parts;
    st::basic_vector<std::size_t> picked_shards;
    std::size_t pooled_rows = 0;
    std::size_t attempt = 0;
    while ((pooled_rows < total_segments || parts.size() < floor_files) && parts.size() < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_bytes;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(byte_prefix_.begin(), byte_prefix_.end(), position) - byte_prefix_.begin() - 1);
        if (std::find(picked_shards.begin(), picked_shards.end(), file) != picked_shards.end()) continue;
        // Decode this file to learn its rows (no footer), and keep it alive for the concat.
        auto shard = jsonl_file_t::make(*storage_, absolute_paths_[file], config.field, gpu_stream);
        if (!shard) return expected_t::fail(std::move(shard));
        pooled_rows += shard->rows();
        if (st::failed(picked_shards.push_back(std::size_t {file})) ||
            st::failed(parts.push_back(std::move(shard.value_))))
            return expected_t::fail(status_t::bad_alloc_k, "reservoir parts allocation failed");
    }
    if (pooled_rows < total_segments)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} JSONL files cover fewer than the {} segments this fill needs",
                                pooled_rows, parts.size(), total_segments);

    // Merge the decoded shards into one reservoir field, then the shared pack.
    auto reservoir_read = jsonl_file_t::concat(parts, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    jsonl_file_t const reservoir = std::move(reservoir_read.value_);
    return pack_reservoir_(reservoir.column(), reservoir.rows(), hash(seed_, gpu_rank, row_draw_salt_k),
                           data_epoch_t {training_step}, batches, gpu_stream);
}

expected<tokens_loader_csv_t> tokens_loader_csv_t::make(gpu_direct_storage_session_t &session,
                                                        sz::string_view_t path_or_glob, u64_t seed) noexcept {
    using result_t = expected<tokens_loader_csv_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths — POSIX readdir.
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    tokens_loader_csv_t loader;
    loader.storage_ = &session;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;

    // No footer — weigh files by byte size (POSIX stat), a running prefix for the weighted file draw.
    if (expected_t weighed = weigh_by_bytes_(loader.absolute_paths_, loader.byte_prefix_, loader.reservoir_); !weighed)
        return result_t::fail(std::move(weighed));
    return result_t {std::move(loader)};
}

expected_t tokens_loader_csv_t::fill(gpu_stream_t &gpu_stream, tokens_base_loader_csv_config_t config, u32_t gpu_rank,
                                     u32_t training_step, std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    if (documents == 0u)
        return expected_t::fail(status_t::shape_unsupported_k, "the batch carries no segment capacity");
    std::size_t const total_segments = unroll * documents;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_bytes = byte_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};

    // Pick distinct files weighted by BYTE size (CSV has no footer), decoding each until the pooled rows
    // cover the fill — never fewer than two, unless the whole dataset is one file.
    st::basic_vector<csv_file_t> parts;
    st::basic_vector<std::size_t> picked_shards;
    std::size_t pooled_rows = 0;
    std::size_t attempt = 0;
    while ((pooled_rows < total_segments || parts.size() < floor_files) && parts.size() < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_bytes;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(byte_prefix_.begin(), byte_prefix_.end(), position) - byte_prefix_.begin() - 1);
        if (std::find(picked_shards.begin(), picked_shards.end(), file) != picked_shards.end()) continue;
        // Decode this file to learn its rows (no footer), and keep it alive for the concat.
        auto shard = csv_file_t::make(*storage_, absolute_paths_[file], config.column, config.delimiter, config.header,
                                      gpu_stream);
        if (!shard) return expected_t::fail(std::move(shard));
        pooled_rows += shard->rows();
        if (st::failed(picked_shards.push_back(std::size_t {file})) ||
            st::failed(parts.push_back(std::move(shard.value_))))
            return expected_t::fail(status_t::bad_alloc_k, "reservoir parts allocation failed");
    }
    if (pooled_rows < total_segments)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} CSV files cover fewer than the {} segments this fill needs",
                                pooled_rows, parts.size(), total_segments);

    // Merge the decoded shards into one reservoir column, then the shared pack.
    auto reservoir_read = csv_file_t::concat(parts, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    csv_file_t const reservoir = std::move(reservoir_read.value_);
    return pack_reservoir_(reservoir.column(), reservoir.rows(), hash(seed_, gpu_rank, row_draw_salt_k),
                           data_epoch_t {training_step}, batches, gpu_stream);
}
