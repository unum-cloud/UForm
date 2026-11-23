/**
 *  @file src/loaders/tokens_loader.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *  @brief The Parquet tokens loader's bodies and kernels — see the header for the contract.
 *
 *  The pack pair mirrors the mixer's ( @c multimodal_batch.cu ): a serial geometry walk writing
 *  offsets + lengths under the segments-never-cross-rows rule, then a CTA-per-segment gather. The
 *  row draw is a third kernel — a keyed permutation of the pooled files' rows, without replacement.
 */
#include <algorithm> // `std::upper_bound` (weighted file pick), `std::find` (dedup)

#include "uform/loaders/tokens_loader.cuh"

using namespace unum::uform::encoder;
namespace util = unum::uform::utilities;
using namespace unum::uform::loaders;
namespace ul = unum::uform::loaders;

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row permutation
 *  stay decorrelated under one @c (seed, epoch, rank) . */
inline constexpr util::u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr util::u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/**
 *  @brief Draw the whole fill's document rows: a keyed permutation of @c [0, pooled_rows) into
 *      @p record_indices , without replacement (a Feistel bijection, so distinct draw positions map
 *      to distinct rows).
 *
 *  Because the pool is the picked files concatenated, a batch's slice of this permutation is a
 *  scatter of rows across those files.
 *
 *  __grid_size__ grid-stride over @c record_indices ; no dynamic shared memory.
 */
static __global__ void sample_rows_kernel_(util::u64r1_span_t record_indices, util::u64_t seed,
                                           util::data_epoch_t data_epoch, std::size_t pooled_rows) {
    std::size_t const total = record_indices.extent(0);
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t draw = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; draw < total;
         draw += stride)
        record_indices[draw] = util::epoch_permutation(seed, data_epoch, draw, pooled_rows);
}

/**
 *  @brief Write one batch's geometry: thread 0 walks the destination segments serially — the repack
 *      cursor is a strict scan.
 *
 *  It reads each drawn document's byte count through its reservoir row index, truncates to the row
 *  width, and jumps the cursor past any row boundary the segment would cross. A segment that would
 *  overflow the rectangle is dropped (zero length) and counted into @p dropped . Segments past the
 *  drawn documents PAD: zero length, repeated offset.
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void tokens_geometry_kernel_(util::u64r1_view_t reservoir_offsets, util::u64r1_view_t record_indices,
                                               std::size_t prefix_bytes, util::u32r1_span_t segment_offsets,
                                               util::u32r1_span_t segment_lengths, modality_span_t segment_modality,
                                               util::u32r1_span_t dropped, tokens_matrix_span_t tokens) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    util::u32_t const columns_per_row = static_cast<util::u32_t>(tokens.extent(1));
    util::u32_t const capacity = static_cast<util::u32_t>(tokens.extent(0) * tokens.extent(1));
    util::u32_t const live_segments = static_cast<util::u32_t>(record_indices.extent(0));
    util::u32_t const total_segments = static_cast<util::u32_t>(segment_lengths.extent(0));
    util::u32_t cursor = 0, overflowed = 0;
    for (util::u32_t segment_index = 0; segment_index != total_segments; ++segment_index) {
        util::u32_t columns = 0;
        if (segment_index < live_segments) {
            util::u64_t const source = record_indices[segment_index];
            util::u64_t const bytes = prefix_bytes + reservoir_offsets[source + 1] - reservoir_offsets[source];
            columns = static_cast<util::u32_t>(fu::min_of<util::u64_t>(bytes, columns_per_row));
        }
        // A segment never crosses its row: jump the cursor to the next row when it would.
        util::u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (columns > room_in_row) cursor += room_in_row;
        if (cursor + columns > capacity) ++overflowed, columns = 0;
        segment_offsets[segment_index] = cursor;
        segment_lengths[segment_index] = columns;
        // Every fill owns the whole segment table — a reused batch may hold another modality's
        // tags, and a stale tag routes token bytes into a media head downstream (text_k doubles as
        // PAD at length 0).
        segment_modality[segment_index] = modality_t::text_k;
        cursor += columns;
    }
    segment_offsets[total_segments] = cursor;
    dropped[0] += overflowed;
}

/**
 *  @brief The gather-pack: each CTA copies its document's bytes as token ids — a byte @b is a token
 *      in the 256-entry vocabulary — from the reservoir column into the batch's token rectangle at
 *      the geometry's placement, threads striding the columns.
 *
 *  __grid_size__ @c [documents,1,1] ; a PAD segment (zero length) exits immediately.
 */
static __global__ void tokens_gather_kernel_(util::b8r1_view_t reservoir_payload, util::u64r1_view_t reservoir_offsets,
                                             util::u64r1_view_t record_indices, util::b8r1_view_t prefix,
                                             util::u32r1_view_t segment_offsets, util::u32r1_view_t segment_lengths,
                                             tokens_matrix_span_t tokens) {
    util::u32_t const columns_per_row = static_cast<util::u32_t>(tokens.extent(1));
    util::u32_t const segment_index = blockIdx.x;
    util::u32_t const columns = segment_lengths[segment_index];
    if (columns == 0u) return;
    util::u64_t const source_begin = reservoir_offsets[record_indices[segment_index]];
    util::u32_t const destination = segment_offsets[segment_index];
    util::u32_t const row = destination / columns_per_row;
    util::u32_t const column_begin = destination % columns_per_row;
    std::size_t const prefix_bytes = prefix.extent(0);
    for (util::u32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        util::b8_t const byte = column < prefix_bytes ? prefix[column]
                                                      : reservoir_payload[source_begin + column - prefix_bytes];
        tokens[row, column_begin + column] = static_cast<util::token_id_t>(static_cast<util::u8_t>(byte));
    }
}

util::expected_t ul::pack_tokens(column_view_t texts, util::u64r1_view_t elements, util::b8r1_view_t prefix,
                                 tokens_matrix_span_t tokens, util::u32r1_span_t segment_offsets,
                                 util::u32r1_span_t segment_lengths, modality_span_t segment_modality,
                                 util::u32r1_span_t dropped, util::gpu_stream_t& gpu_stream) noexcept {
    if (elements.extent(0) > segment_lengths.extent(0) || segment_offsets.extent(0) != segment_lengths.extent(0) + 1u ||
        segment_modality.extent(0) != segment_lengths.extent(0))
        return util::expected_t::fail(util::status_code_t::unsupported_k, "{} documents exceed a table of {} segments",
                                      elements.extent(0), segment_lengths.extent(0));
    if (elements.empty()) return {};
    util::switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok())
        return util::expected_t::fail(util::status_code_t::device_k, "Context entry failed for the loader stream");
    dim3 const one_block {1u, 1u, 1u};
    util::expected_t packed = util::launch_kernel(&tokens_geometry_kernel_, one_block, one_block, gpu_stream,
                                                  texts.offsets, elements, prefix.extent(0), segment_offsets,
                                                  segment_lengths, segment_modality, dropped, tokens);
    if (packed)
        packed = util::launch_kernel(&tokens_gather_kernel_,
                                     dim3 {static_cast<util::u32_t>(elements.extent(0)), 1u, 1u},
                                     dim3 {util::kernel_block_size_default_k, 1u, 1u}, gpu_stream, texts.payload,
                                     texts.offsets, elements, prefix, segment_offsets, segment_lengths, tokens);
    return packed;
}

/** The format-agnostic pack: draw the whole fill's rows over @p reservoir_column on the GPU (a
 *  keyed permutation of @p pool_rows , without replacement) and pack @b every batch from it. Both
 *  the Parquet and JSONL loaders share this once they hold a decoded reservoir — only how the
 *  reservoir is obtained differs (a footer-known Parquet read vs a JSONL decode + concat). A pure
 *  enqueue on @p gpu_stream . */
[[nodiscard]] static util::expected_t pack_reservoir_(column_view_t reservoir_column, std::size_t pool_rows,
                                                      util::u64_t row_key, util::data_epoch_t data_epoch,
                                                      std::span<multimodal_batch_on_gpu_t> batches,
                                                      util::gpu_stream_t& gpu_stream) noexcept {
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    std::size_t const total_segments = unroll * documents;

    // Per-fill device index buffer, drawn on the GPU by the keyed permutation of the whole pool.
    st::basic_vector<util::u64_t, util::gpu_alloc<util::u64_t>> record_indices {
        util::gpu_alloc<util::u64_t> {gpu_stream.get()}};
    if (st::failed(record_indices.resize(st::assume_uninitialized, total_segments)))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k, "row index allocation failed ({} draws)",
                                      total_segments);

    util::switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok())
        return util::expected_t::fail(util::status_code_t::device_k, "Context entry failed for the loader stream");
    nk::tensor_span<util::u64_t, 2> const batch_records {record_indices.data(), unroll, documents};
    if (util::expected_t drawn = util::launch_kernel(&sample_rows_kernel_,
                                                     dim3 {util::grid_size_for(total_segments), 1u, 1u},
                                                     dim3 {util::kernel_block_size_default_k, 1u, 1u}, gpu_stream,
                                                     batch_records.flatten<1>(), row_key, data_epoch, pool_rows);
        !drawn)
        return drawn;

    // Each batch packs its own row of the permutation; the rectangle carries its shape.
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t& loaded = batches[batch];
        if (util::expected_t packed = pack_tokens(reservoir_column, batch_records[batch, nk::slice], {},
                                                  loaded.tokens.span(), loaded.segment_offsets.span(),
                                                  loaded.segment_lengths.span(), loaded.segment_modality.span(),
                                                  loaded.dropped.span(), gpu_stream);
            !packed)
            return packed;
        // A document names no role, key, prefix or late-interaction side; none keeps stale bytes.
        util::expected_t cleared = util::zero_async(loaded.segment_side.span(), gpu_stream);
        if (cleared) cleared = util::zero_async(loaded.segment_role.span(), gpu_stream);
        if (cleared) cleared = util::zero_async(loaded.segment_key.span(), gpu_stream);
        if (cleared) cleared = util::zero_async(loaded.segment_prefix_length.span(), gpu_stream);
        if (!cleared) return cleared;
    }
    return {};
}

util::expected<tokens_loader_parquet_t> tokens_loader_parquet_t::make(gpu_direct_storage_session_t& session,
                                                                      sz::string_view_t path_or_glob,
                                                                      util::u64_t seed) noexcept {
    using result_t = util::expected<tokens_loader_parquet_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths
    // — POSIX glob(3).
    auto [paths, status] = expand_glob(path_or_glob);
    if (!status) return result_t::fail(std::move(status));

    tokens_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.seed_ = seed;

    // Footer only — host-side, no GPU, column-agnostic: the column is chosen per fill, so a footer
    // naming no column counts every row of a group complete.
    std::size_t groups = 0;
    for (sz::string_t const& path : paths) {
        auto [metadata, read_status] = parquet_file_t::table_metadata(session, path);
        if (!read_status) return result_t::fail(std::move(read_status));
        if (metadata.rows == 0)
            return result_t::fail(util::status_code_t::invalid_argument_k, "shard {} is empty", path);
        if (util::expected_t appended = loader.groups_.append_shard(path, metadata); !appended)
            return result_t::fail(std::move(appended));
        groups += metadata.group_rows.size();
        loader.reservoir_.charge(metadata);
    }
    if (util::expected_t reserved = loader.groups_.reserve(groups); !reserved)
        return result_t::fail(std::move(reserved));
    return result_t {std::move(loader)};
}

util::expected_t tokens_loader_parquet_t::fill(util::gpu_stream_t& gpu_stream,
                                               tokens_base_loader_parquet_config_t config, util::u32_t gpu_rank,
                                               util::u32_t training_step,
                                               std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    if (documents == 0u)
        return util::expected_t::fail(util::status_code_t::unsupported_k, "The batch carries no segment capacity");
    std::size_t const total_segments = unroll * documents;

    // Draw row groups by the keyed weighted pick shared with `drawn_shards`, then decode only them
    // into one device reservoir; the true pool size is the decoded row count.
    if (util::expected_t picked = groups_.draw(seed_, training_step, gpu_rank, file_draw_salt_k, total_segments, 2);
        !picked)
        return picked;
    constexpr util::splitmix64_hash_t hash {};

    auto [reservoir, status] = parquet_file_t::make(*storage_, groups_.picked_paths(),
                                                    std::span<sz::string_view_t const> {&config.column, 1},
                                                    groups_.picked_groups(), gpu_stream);
    if (!status) return util::expected_t::fail(std::move(status));
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_segments)
        return util::expected_t::fail(util::status_code_t::unsupported_k,
                                      "{} pooled rows across {} files cover fewer than the {} segments this fill needs",
                                      pooled_rows, groups_.picked_paths().size(), total_segments);

    // Draw the pool's rows and pack every batch — shared with the JSONL loader.
    return pack_reservoir_(reservoir.column(0), pooled_rows, hash(seed_, gpu_rank, row_draw_salt_k),
                           util::data_epoch_t {training_step}, batches, gpu_stream);
}

util::expected<sz::strings_view_t> tokens_loader_parquet_t::drawn_shards(util::u32_t gpu_rank,
                                                                         util::u32_t training_step,
                                                                         std::size_t target_segments) noexcept {
    using result_t = util::expected<sz::strings_view_t>;
    if (util::expected_t picked = groups_.draw(seed_, training_step, gpu_rank, file_draw_salt_k, target_segments, 2);
        !picked)
        return result_t::fail(std::move(picked));
    return result_t {groups_.picked_paths()};
}

/** Weigh footerless @p paths by byte size, POSIX @c stat , into @p byte_prefix , the weighted file
 *  draw's running prefix, and charge each file's bytes to @p reservoir . */
[[nodiscard]] static util::expected_t weigh_by_bytes_(std::span<sz::string_t const> paths,
                                                      st::basic_vector<std::size_t>& byte_prefix,
                                                      reservoir_bound_t& reservoir) noexcept {
    if (st::failed(byte_prefix.resize(paths.size() + 1)))
        return util::expected_t::fail(util::status_code_t::out_of_memory_k,
                                      "shard prefix allocation failed ({} shards)", paths.size());
    byte_prefix[0] = 0;
    for (std::size_t index = 0; index != paths.size(); ++index) {
        auto [bytes, status] = file_size_bytes(paths[index]);
        if (!status) return util::expected_t::fail(std::move(status));
        if (bytes == 0)
            return util::expected_t::fail(util::status_code_t::invalid_argument_k, "shard {} is empty", paths[index]);
        byte_prefix[index + 1] = byte_prefix[index] + bytes;
        reservoir.charge(parquet_length_metadata_t {0, bytes});
    }
    return {};
}

util::expected<tokens_loader_jsonl_t> tokens_loader_jsonl_t::make(gpu_direct_storage_session_t& session,
                                                                  sz::string_view_t path_or_glob,
                                                                  util::u64_t seed) noexcept {
    using result_t = util::expected<tokens_loader_jsonl_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths
    // — POSIX readdir.
    auto [paths, status] = expand_glob(path_or_glob);
    if (!status) return result_t::fail(std::move(status));

    tokens_loader_jsonl_t loader;
    loader.storage_ = &session;
    loader.absolute_paths_ = std::move(paths);
    loader.seed_ = seed;

    // No footer — weigh files by byte size (POSIX stat), a running prefix for the
    // weighted file draw.
    if (util::expected_t weighed = weigh_by_bytes_(loader.absolute_paths_, loader.byte_prefix_, loader.reservoir_);
        !weighed)
        return result_t::fail(std::move(weighed));
    return result_t {std::move(loader)};
}

util::expected_t tokens_loader_jsonl_t::fill(util::gpu_stream_t& gpu_stream, tokens_base_loader_jsonl_config_t config,
                                             util::u32_t gpu_rank, util::u32_t training_step,
                                             std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    if (documents == 0u)
        return util::expected_t::fail(util::status_code_t::unsupported_k, "The batch carries no segment capacity");
    std::size_t const total_segments = unroll * documents;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = fu::min_of<std::size_t>(2, shard_count);
    std::size_t const total_bytes = byte_prefix_[shard_count];
    constexpr util::splitmix64_hash_t hash {};

    // Pick distinct files weighted by BYTE size (JSONL has no footer), decoding each until the
    // pooled rows cover the fill — never fewer than two, unless the whole dataset is one file.
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
        auto [shard, status] = jsonl_file_t::make(*storage_, absolute_paths_[file], config.field, gpu_stream);
        if (!status) return util::expected_t::fail(std::move(status));
        pooled_rows += shard.rows();
        if (st::failed(picked_shards.push_back(std::size_t {file})) || st::failed(parts.push_back(std::move(shard))))
            return util::expected_t::fail(util::status_code_t::out_of_memory_k, "No room for the reservoir parts");
    }
    if (pooled_rows < total_segments)
        return util::expected_t::fail(
            util::status_code_t::unsupported_k,
            "{} pooled rows across {} JSONL files cover fewer than the {} segments this fill needs", pooled_rows,
            parts.size(), total_segments);

    // Merge the decoded shards into one reservoir field, then the shared pack.
    auto [reservoir, status] = jsonl_file_t::concat(parts, gpu_stream);
    if (!status) return util::expected_t::fail(std::move(status));
    return pack_reservoir_(reservoir.column(), reservoir.rows(), hash(seed_, gpu_rank, row_draw_salt_k),
                           util::data_epoch_t {training_step}, batches, gpu_stream);
}

util::expected<tokens_loader_csv_t> tokens_loader_csv_t::make(gpu_direct_storage_session_t& session,
                                                              sz::string_view_t path_or_glob,
                                                              util::u64_t seed) noexcept {
    using result_t = util::expected<tokens_loader_csv_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths
    // — POSIX readdir.
    auto [paths, status] = expand_glob(path_or_glob);
    if (!status) return result_t::fail(std::move(status));

    tokens_loader_csv_t loader;
    loader.storage_ = &session;
    loader.absolute_paths_ = std::move(paths);
    loader.seed_ = seed;

    // No footer — weigh files by byte size (POSIX stat), a running prefix for the
    // weighted file draw.
    if (util::expected_t weighed = weigh_by_bytes_(loader.absolute_paths_, loader.byte_prefix_, loader.reservoir_);
        !weighed)
        return result_t::fail(std::move(weighed));
    return result_t {std::move(loader)};
}

util::expected_t tokens_loader_csv_t::fill(util::gpu_stream_t& gpu_stream, tokens_base_loader_csv_config_t config,
                                           util::u32_t gpu_rank, util::u32_t training_step,
                                           std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const documents = batches[0].segment_lengths.extent(0);
    if (documents == 0u)
        return util::expected_t::fail(util::status_code_t::unsupported_k, "The batch carries no segment capacity");
    std::size_t const total_segments = unroll * documents;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = fu::min_of<std::size_t>(2, shard_count);
    std::size_t const total_bytes = byte_prefix_[shard_count];
    constexpr util::splitmix64_hash_t hash {};

    // Pick distinct files weighted by BYTE size (CSV has no footer), decoding each until the pooled
    // rows cover the fill — never fewer than two, unless the whole dataset is one file.
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
        auto [shard, status] = csv_file_t::make(*storage_, absolute_paths_[file], config.column, config.delimiter,
                                                config.header, gpu_stream);
        if (!status) return util::expected_t::fail(std::move(status));
        pooled_rows += shard.rows();
        if (st::failed(picked_shards.push_back(std::size_t {file})) || st::failed(parts.push_back(std::move(shard))))
            return util::expected_t::fail(util::status_code_t::out_of_memory_k, "No room for the reservoir parts");
    }
    if (pooled_rows < total_segments)
        return util::expected_t::fail(
            util::status_code_t::unsupported_k,
            "{} pooled rows across {} CSV files cover fewer than the {} segments this fill needs", pooled_rows,
            parts.size(), total_segments);

    // Merge the decoded shards into one reservoir column, then the shared pack.
    auto [reservoir, status] = csv_file_t::concat(parts, gpu_stream);
    if (!status) return util::expected_t::fail(std::move(status));
    return pack_reservoir_(reservoir.column(), reservoir.rows(), hash(seed_, gpu_rank, row_draw_salt_k),
                           util::data_epoch_t {training_step}, batches, gpu_stream);
}
