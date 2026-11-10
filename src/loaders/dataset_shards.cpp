/**
 *  @file src/loaders/dataset_shards.cpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The dataset-shard readers' one firewalled body: @ref parquet_file_t and @ref jsonl_file_t .
 *      cuDF and RMM throw and allocate freely, so every call to them lives inside this translation unit's
 *      @c try blocks and their types stay behind the readers' pimpls. See the header for the contract.
 */

#include <cstring> // `std::memcpy` — a footer statistic's bytes

#include <dirent.h>   // vendor-boundary: POSIX `opendir` / `readdir` — the raw directory scan, no glob to link
#include <sys/stat.h> // vendor-boundary: POSIX `stat` — the footerless formats' byte-size file weight

#include <algorithm> // `std::sort`, `std::upper_bound` — the shard order and the row-group draw, `std::find_if`
#include <limits>    // `std::numeric_limits` — a statistic-less group's unbounded span
#include <span>      // `std::span` — the borrowed payload / offsets
#include <string>
#include <vector>

#include <fnmatch.h> // vendor-boundary: POSIX `fnmatch` — the shard filename pattern match
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>         // `cudf::concatenate` — merge decoded shards into one table
#include <cudf/io/csv.hpp>              // `cudf::io::read_csv`, `csv_reader_options`
#include <cudf/io/json.hpp>             // `cudf::io::read_json`, `json_reader_options`
#include <cudf/io/parquet.hpp>          // `cudf::io::read_parquet`, `parquet_reader_options`
#include <cudf/io/parquet_metadata.hpp> // `read_parquet_metadata`, `read_parquet_footers`
#include <cudf/io/types.hpp>            // `cudf::io::source_info`
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>         // `cudf::cast` — the one offset widening
#include <cudf/io/datasource.hpp> // `cudf::io::datasource` — a sidecar's tar, read to the device whole
#include <cuda/stream_ref>        // `cuda::stream_ref` — the stream every cuDF and RMM call takes
#include <rmm/device_buffer.hpp>  // `rmm::device_buffer` — the gathered tar members

#include "uform/utilities/vram_nvidia_device.hpp" // `device_address` — the tar gather's copy endpoints
#include "uform/utilities/types.hpp"              // `st::basic_vector` — the owned per-column table
#include "uform/loaders/dataset_shards.hpp"

namespace ul = unum::uform::loaders;
using namespace unum::uform::utilities;
using namespace ul;

/** One decoded column's device-resident spans and the cuDF columns that back them —
 *  everything RAII, nothing hand-freed. A shared @b representation (not a shared type) behind
 *  both the Parquet and JSONL readers: the payload and offsets are @c std::span borrows into
 *  an owned cuDF buffer. */
struct column_span_t {

    /** cuDF's widened leaf-offset column, owned; the leaf offsets are always cast to @c u64 . */
    std::unique_ptr<cudf::column> widened_offsets;

    /** cuDF's widened group-offset column, owned; non-null only for a nested @c list<string> . */
    std::unique_ptr<cudf::column> widened_groups;

    /** The column's leaf bytes, back to back — borrowed from the owning table / column. */
    bytes_view_t payload;

    /** @b [elements+1] leaf offsets — owned by @ref widened_offsets . */
    std::span<u64_t const> offsets;

    /** @b [rows+1] row → element offsets for a nested column — owned by @ref widened_groups ;
     *  empty for a flat column. */
    std::span<u64_t const> groups;
};

/** The @ref column_view_t over @p column 's leaf payload and its @b [elements+1] offsets. */
[[nodiscard]] static column_view_t view_of_(column_span_t const &column) noexcept {
    return {column.payload, column.offsets};
}

/** What a decoded Parquet file owns and lends: the cuDF table (every column's payload + native
 *  offsets) and one @ref column_span_t per requested column, in request order. */
struct parquet_file_t::state_t {
    std::unique_ptr<cudf::table> table;
    st::basic_vector<column_span_t> columns;
    rmm::device_buffer tar_payload; // the members a tar-backed column gathers, back to back
};

parquet_file_t::parquet_file_t() noexcept = default;
parquet_file_t::parquet_file_t(parquet_file_t &&) noexcept = default;
parquet_file_t &parquet_file_t::operator=(parquet_file_t &&) noexcept = default;
parquet_file_t::~parquet_file_t() = default;
parquet_file_t::parquet_file_t(std::unique_ptr<state_t> state) noexcept : state_(std::move(state)) {}

std::size_t parquet_file_t::rows() const noexcept {
    return state_ ? static_cast<std::size_t>(state_->table->num_rows()) : 0;
}
std::size_t parquet_file_t::columns() const noexcept { return state_ ? state_->columns.size() : 0; }
column_view_t parquet_file_t::column(std::size_t index) const noexcept { return view_of_(state_->columns[index]); }
std::span<u64_t const> parquet_file_t::groups(std::size_t index) const noexcept {
    return state_ ? state_->columns[index].groups : std::span<u64_t const> {};
}

/** A decoded column flattened to its leaf payload, the leaf offsets child, and — for a nested
 *  @c list<string> column — the outer row → element offsets child. A @c string and a
 *  @c list<uint8> column differ only in which child holds the payload and carry no groups; a
 *  @c list<string> column carries both offset levels. */
struct flat_column_t {

    /** The column's leaf bytes, back to back. */
    bytes_view_t payload;

    /** The leaf offsets child — @c elements+1 entries, 32- or 64-bit. */
    cudf::column_view offsets;

    /** The outer row → element offsets child for a nested column; a default (empty-type)
     *  column_view for a flat one. */
    cudf::column_view groups;
};

/** The character bytes of the cuDF strings column @p strings — where cuDF's @c char
 *  tape becomes bytes. */
[[nodiscard]] static bytes_view_t chars_of_(cudf::strings_column_view const &strings,
                                            cuda::stream_ref stream) noexcept {
    return std::as_bytes(
        std::span<char const> {strings.chars_begin(stream), static_cast<std::size_t>(strings.chars_size(stream))});
}

/** Flatten the @p name column @p decoded to its payload and offset levels, or a
 *  @c unsupported_k failure naming it when it is not one of @c string , @c list<uint8> , or
 *  @c list<string> . */
[[nodiscard]] static expected<flat_column_t> flatten_column_(cudf::column_view const &decoded, sz::string_view_t name,
                                                             cuda::stream_ref stream) noexcept {
    using result_t = expected<flat_column_t>;
    if (decoded.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view const strings {decoded};
        return result_t {flat_column_t {chars_of_(strings, stream), strings.offsets(), {}}};
    }
    if (decoded.type().id() == cudf::type_id::LIST) {
        cudf::lists_column_view const list {decoded};
        cudf::column_view const child = list.child();
        // list<uint8> — a flat media blob: the child bytes are the payload, the list
        // offsets the leaf.
        if (child.type().id() == cudf::type_id::UINT8 || child.type().id() == cudf::type_id::INT8) {
            auto const *payload = static_cast<b8_t const *>(child.head());
            return result_t {flat_column_t {{payload, static_cast<std::size_t>(child.size())}, list.offsets(), {}}};
        }
        // list<int32> — row indices, the mined hard negatives: the child values are the payload,
        // and the list offsets count values, not bytes.
        if (child.type().id() == cudf::type_id::INT32) {
            auto const *payload = static_cast<b8_t const *>(child.head());
            return result_t {
                flat_column_t {{payload, static_cast<std::size_t>(child.size()) * sizeof(i32_t)}, list.offsets(), {}}};
        }
        // list<string> — a nested column (K captions / negatives per row): the string child's chars
        // are the leaf payload, its offsets the element → bytes level, and the list offsets the row
        // → element groups.
        if (child.type().id() == cudf::type_id::STRING) {
            cudf::strings_column_view const strings {child};
            return result_t {flat_column_t {chars_of_(strings, stream), strings.offsets(), list.offsets()}};
        }
    }
    // A dotted nested selection ("audio.bytes") arrives as its parent struct pruned to the one
    // requested child — unwrap and flatten the child itself.
    if (decoded.type().id() == cudf::type_id::STRUCT && decoded.num_children() == 1)
        return flatten_column_(decoded.child(0), name, stream);
    return result_t::fail(status_t::unsupported_k, "column {} is not string, list<uint8>, list<int32>, or list<string>",
                          name);
}

/** Own a @c u64_t copy of @p offsets — cuDF decodes 32- or 64-bit @b signed offsets, and byte
 *  positions are unsigned sizes, so the one @c cudf::cast to @c UINT64 both widens and unsigns.
 *  Writes the borrow into @p span and the owning column into @p owner . */
static void widen_offsets_(cudf::column_view const &offsets, cuda::stream_ref stream, std::span<u64_t const> &span,
                           std::unique_ptr<cudf::column> &owner) { // vendor-boundary: cuDF throws into the caller's try
    owner = cudf::cast(offsets, cudf::data_type {cudf::type_id::UINT64}, stream);
    span = {owner->view().data<u64_t>(), static_cast<std::size_t>(offsets.size())};
}

/** Widen a flattened column's offsets into an owned @ref column_span_t — the decode tail every
 *  reader shares once cuDF hands back a @ref flat_column_t . */
[[nodiscard]] static expected<column_span_t> adopt_flat_(
    expected<flat_column_t> flat, cuda::stream_ref stream) { // vendor-boundary: cuDF throws into the caller's try
    if (!flat) return expected<column_span_t>::fail(std::move(flat));
    column_span_t column;
    column.payload = flat->payload;
    widen_offsets_(flat->offsets, stream, column.offsets, column.widened_offsets);
    if (flat->groups.type().id() != cudf::type_id::EMPTY)
        widen_offsets_(flat->groups, stream, column.groups, column.widened_groups);
    return expected<column_span_t> {std::move(column)};
}

/** The sidecar's byte index of each row's member in the sibling tar: its offset and its length. */
inline constexpr char tar_offset_column_k[] = "tar_byte_offset";
inline constexpr char tar_length_column_k[] = "tar_byte_length";

/** Decode a footer statistic's little-endian 64-bit integer into @p value ; false when the
 *  writer kept none. */
[[nodiscard]] static bool statistic_(auto const &bytes, i64_t &value) noexcept {
    if (!bytes || bytes->size() != sizeof(i64_t)) return false;
    std::memcpy(&value, bytes->data(), sizeof(i64_t));
    return true;
}

/** The tar beside the Parquet sidecar at @p path : the same stem with a @c .tar suffix. */
[[nodiscard]] static bool tar_path_of_(sz::string_view_t path, sz::string_t &tar) noexcept {
    sz::string_view_t const sidecar_suffix {".parquet"};
    if (path.ends_with(sidecar_suffix)) path.remove_suffix(sidecar_suffix.size());
    return format_into(tar, "{}.tar", path);
}

/** Gather every row's @p extension member into @p payload back to back, one shard at a
 *  time: its sibling tar is read to the device whole, and one batch copies out the members
 *  its rows index. Rows follow the shards in @p paths order, @p rows_per_shard of each, as
 *  cuDF concatenates them. */
[[nodiscard]] static expected<column_span_t> read_tar_members_(
    cudf::column_view byte_offsets, cudf::column_view byte_lengths, sz::string_view_t extension,
    sz::strings_view_t paths, std::span<std::size_t const> rows_per_shard, cuda::stream_ref stream,
    rmm::device_buffer &payload) { // vendor-boundary: cuDF throws into the caller's try
    using result_t = expected<column_span_t>;
    std::size_t const rows = static_cast<std::size_t>(byte_offsets.size());
    st::basic_vector<i64_t> offsets, lengths;
    st::basic_vector<u64_t> bounds;
    st::basic_vector<CUdeviceptr> sources, destinations;
    st::basic_vector<std::size_t> sizes, row_runs, run_starts, run_ends, run_members, run_bases;
    if (st::failed(offsets.resize(rows)) || st::failed(lengths.resize(rows)) || st::failed(bounds.resize(rows + 1)) ||
        st::failed(sources.resize(rows)) || st::failed(destinations.resize(rows)) || st::failed(sizes.resize(rows)) ||
        st::failed(row_runs.resize(rows)))
        return result_t::fail(status_t::bad_alloc_k, "tar index allocation failed ({} rows)", rows);
    if (cuMemcpyDtoHAsync(offsets.data(), device_address(byte_offsets.data<i64_t>()), rows * sizeof(i64_t),
                          stream.get()) != CUDA_SUCCESS ||
        cuMemcpyDtoHAsync(lengths.data(), device_address(byte_lengths.data<i64_t>()), rows * sizeof(i64_t),
                          stream.get()) != CUDA_SUCCESS ||
        cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return result_t::fail(status_t::device_error_k, "tar index landing failed ({} rows)", rows);

    // The members land back to back in row order, so each one starts at the running sum of lengths.
    bounds[0] = 0;
    for (std::size_t row = 0; row != rows; ++row) bounds[row + 1] = bounds[row] + static_cast<u64_t>(lengths[row]);
    payload = rmm::device_buffer {bounds[rows], stream};

    CUmemcpyAttributes attributes {};
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    std::size_t attribute_index = 0;
    std::size_t row_begin = 0;
    for (std::size_t shard = 0; shard != paths.size(); ++shard) {
        std::size_t const row_end = row_begin + rows_per_shard[shard];
        sz::string_t tar;
        if (!tar_path_of_(paths[shard], tar))
            return result_t::fail(status_t::bad_alloc_k, "tar path allocation failed for {}", paths[shard]);
        auto source = cudf::io::datasource::create(std::string {tar.c_str()}); // vendor-boundary
        if (!source->supports_device_read())
            return result_t::fail(status_t::io_k, "the tar beside {} cannot be read to the device", paths[shard]);
        // Members are named `<key>.<column>`, as in WebDataset; a tar of other members means the
        // shard simply lacks the column, which must fail rather than read image bytes as text.
        std::uint8_t member_name[100] = {}; // the USTAR header's name field
        source->host_read(0, sizeof(member_name), member_name);
        std::uint8_t const *const name_end = std::find(member_name, member_name + sizeof(member_name), 0);
        std::size_t const name_bytes = static_cast<std::size_t>(name_end - member_name);
        if (name_bytes <= extension.size() || member_name[name_bytes - extension.size() - 1] != '.' ||
            !std::equal(extension.begin(), extension.end(), name_end - extension.size()))
            return result_t::fail(status_t::invalid_argument_k, "{} has no column {}, nor tar members named for it",
                                  paths[shard], extension);
        // Members stage in runs of the tar, and a run ends where the gap to the next member exceeds
        // the member bytes it holds, so staging stays under twice the members gathered.
        run_starts.clear();
        run_ends.clear();
        run_members.clear();
        run_bases.clear();
        for (std::size_t row = row_begin; row != row_end; ++row) {
            if (offsets[row] < 0 || lengths[row] <= 0 ||
                static_cast<std::size_t>(offsets[row] + lengths[row]) > source->size())
                return result_t::fail(status_t::invalid_argument_k, "row {} of {} names no member of its {}-byte tar",
                                      row - row_begin, paths[shard], source->size());
            std::size_t const member_begin = static_cast<std::size_t>(offsets[row]);
            std::size_t const member_end = member_begin + static_cast<std::size_t>(lengths[row]);
            std::size_t const run = run_ends.size() - 1;
            bool const extends = !run_ends.empty() && member_begin >= run_ends[run] &&
                                 member_begin - run_ends[run] <= run_members[run];
            if (extends) {
                run_ends[run] = member_end;
                run_members[run] += member_end - member_begin;
            }
            else if (st::failed(run_starts.push_back(std::size_t {member_begin})) ||
                     st::failed(run_ends.push_back(std::size_t {member_end})) ||
                     st::failed(run_members.push_back(member_end - member_begin)))
                return result_t::fail(status_t::bad_alloc_k, "tar run allocation failed for {}", paths[shard]);
            row_runs[row] = run_ends.size() - 1;
        }
        std::size_t staged_bytes = 0;
        for (std::size_t run = 0; run != run_ends.size(); ++run) {
            if (st::failed(run_bases.push_back(std::size_t {staged_bytes})))
                return result_t::fail(status_t::bad_alloc_k, "tar run allocation failed for {}", paths[shard]);
            staged_bytes += run_ends[run] - run_starts[run];
        }
        rmm::device_buffer staging {staged_bytes, stream};
        std::uint8_t *const staged = static_cast<std::uint8_t *>(staging.data());
        for (std::size_t run = 0; run != run_ends.size(); ++run)
            source->device_read(run_starts[run], run_ends[run] - run_starts[run], staged + run_bases[run], stream);
        for (std::size_t row = row_begin; row != row_end; ++row) {
            std::size_t const run = row_runs[row];
            sources[row] = device_address(staged + run_bases[run] + (offsets[row] - run_starts[run]));
            destinations[row] = device_address(static_cast<b8_t const *>(payload.data()) + bounds[row]);
            sizes[row] = static_cast<std::size_t>(lengths[row]);
        }
        // KvikIO writes the next tar from the host, out of stream order, so these copies finish
        // before the staging they read is freed for it.
        if ((row_end != row_begin &&
             cuMemcpyBatchAsync(destinations.data() + row_begin, sources.data() + row_begin, sizes.data() + row_begin,
                                row_end - row_begin, &attributes, &attribute_index, 1, stream.get()) != CUDA_SUCCESS) ||
            cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
            return result_t::fail(status_t::device_error_k, "tar member gather failed for {}", paths[shard]);
        row_begin = row_end;
    }

    column_span_t column;
    column.widened_offsets = std::make_unique<cudf::column>(
        cudf::data_type {cudf::type_id::UINT64}, static_cast<cudf::size_type>(rows + 1),
        rmm::device_buffer {bounds.data(), (rows + 1) * sizeof(u64_t), stream}, rmm::device_buffer {}, 0);
    column.offsets = {column.widened_offsets->view().data<u64_t>(), rows + 1};
    column.payload = {static_cast<b8_t const *>(payload.data()), payload.size()};
    return result_t {std::move(column)};
}

expected_t row_group_draw_t::append_shard(sz::string_view_t path, parquet_length_metadata_t const &footer) noexcept {
    if ((shard_offsets_.empty() && st::failed(shard_offsets_.push_back(0))) ||
        (group_prefix_.empty() && st::failed(group_prefix_.push_back(0))))
        return expected_t::fail(status_t::bad_alloc_k, "row-group table allocation failed");
    std::size_t const groups_before = drawable_groups_.size();
    for (std::size_t group = 0; group != footer.group_complete_rows.size(); ++group) {
        std::size_t const complete_rows = footer.group_complete_rows[group];
        if (complete_rows == 0) continue;
        if (st::failed(drawable_groups_.push_back(static_cast<u32_t>(group))) ||
            st::failed(group_prefix_.push_back(group_prefix_.back() + complete_rows)))
            return expected_t::fail(status_t::bad_alloc_k, "row-group table append failed for {}", path);
    }
    if (drawable_groups_.size() == groups_before) return {};
    sz::string_t owned;
    if (sz::failed(owned.assign(path)) || st::failed(paths_.push_back(std::move(owned))) ||
        st::failed(shard_offsets_.push_back(drawable_groups_.size())))
        return expected_t::fail(status_t::bad_alloc_k, "shard table append failed for {}", path);
    return {};
}

expected_t row_group_draw_t::reserve(std::size_t groups) noexcept {
    groups = fu::min_of(groups, drawable_groups_.size());
    std::size_t const shards = fu::min_of(groups, paths_.size());
    if (st::failed(drawn_.reserve(groups)) || st::failed(picked_groups_.reserve(groups)) ||
        st::failed(picked_shards_.reserve(shards)) || st::failed(picked_paths_.reserve(shards)) ||
        st::failed(picked_offsets_.reserve(shards + 1)))
        return expected_t::fail(status_t::bad_alloc_k, "pick storage allocation failed ({} groups)", groups);
    return {};
}

expected_t row_group_draw_t::draw(u64_t seed, u64_t step, std::size_t rank, u64_t salt, std::size_t rows,
                                  std::size_t floor_shards) noexcept {
    std::size_t const groups = drawable_groups_.size(), total_rows = this->rows();
    if (total_rows == 0) return expected_t::fail(status_t::invalid_argument_k, "no shard holds a complete row");
    auto const shard_of = [&](u32_t group) noexcept {
        return static_cast<std::size_t>(std::upper_bound(shard_offsets_.begin(), shard_offsets_.end(), group) -
                                        shard_offsets_.begin() - 1);
    };
    drawn_.clear(), picked_shards_.clear(), picked_paths_.clear(), picked_offsets_.clear(), picked_groups_.clear();

    // Distinct groups accumulate until they hold the rows, then only a fresh shard's group
    // meets the floor.
    constexpr splitmix64_hash_t hash {};
    std::size_t const floor = fu::min_of(floor_shards, paths_.size());
    std::size_t pooled = 0, shards = 0;
    for (std::size_t attempt = 0; (pooled < rows || shards < floor) && drawn_.size() != groups; ++attempt) {
        std::size_t const position = hash(seed, step, rank, salt, attempt) % total_rows;
        u32_t const group = static_cast<u32_t>(std::upper_bound(group_prefix_.begin(), group_prefix_.end(), position) -
                                               group_prefix_.begin() - 1);
        std::size_t const shard = shard_of(group);
        bool repeated = false, fresh = true;
        for (u32_t const earlier : drawn_) {
            repeated |= earlier == group;
            fresh &= earlier < shard_offsets_[shard] || earlier >= shard_offsets_[shard + 1];
        }
        if (repeated || (!fresh && pooled >= rows)) continue;
        if (drawn_.size() == drawn_.capacity())
            return expected_t::fail(status_t::bad_alloc_k, "a draw outgrew its {} reserved row groups", drawn_.size());
        (void)drawn_.push_back(st::assume_reserved, u32_t {group});
        pooled += group_prefix_[group + 1] - group_prefix_[group];
        shards += fresh;
    }

    // Drawable groups number shard-major, so sorted IDs list the shards and each
    // shard's groups ascending.
    std::sort(drawn_.begin(), drawn_.end());
    for (u32_t const group : drawn_) {
        std::size_t const shard = shard_of(group);
        if (picked_shards_.empty() || picked_shards_.back() != shard) {
            (void)picked_shards_.push_back(st::assume_reserved, static_cast<u32_t>(shard));
            (void)picked_paths_.push_back(st::assume_reserved, paths_[shard].view());
            (void)picked_offsets_.push_back(st::assume_reserved, picked_groups_.size());
        }
        (void)picked_groups_.push_back(st::assume_reserved, u32_t {drawable_groups_[group]});
    }
    (void)picked_offsets_.push_back(st::assume_reserved, picked_groups_.size());
    return {};
}

sz::strings_view_t row_group_draw_t::picked_paths() const noexcept {
    return std::span<sz::string_view_t const> {picked_paths_};
}

std::span<u32_t const> row_group_draw_t::picked_shards() const noexcept { return picked_shards_; }

tape_view<u32_t const> row_group_draw_t::picked_groups() const noexcept { return {picked_groups_, picked_offsets_}; }

std::size_t row_group_draw_t::rows() const noexcept { return group_prefix_.empty() ? 0 : group_prefix_.back(); }

expected<parquet_file_t> parquet_file_t::make(gpu_direct_storage_session_t const &storage, sz::string_view_t path,
                                              sz::strings_view_t columns, gpu_stream_t &stream) noexcept {
    // The single-shard convenience over the concatenating read.
    return make(storage, std::span<sz::string_view_t const> {&path, 1}, columns, {}, stream);
}

expected<parquet_file_t> parquet_file_t::make(gpu_direct_storage_session_t const &, sz::strings_view_t paths,
                                              sz::strings_view_t columns, tape_view<u32_t const> row_groups,
                                              gpu_stream_t &stream) noexcept {
    using result_t = expected<parquet_file_t>;
    if (paths.empty() || columns.empty())
        return result_t::fail(status_t::invalid_argument_k, "a read needs at least one shard path and one column");
    if (!row_groups.empty() && row_groups.size() != paths.size())
        return result_t::fail(status_t::invalid_argument_k, "{} row-group lists for {} shards", row_groups.size(),
                              paths.size());
    for (sz::string_view_t path : paths)
        if (path.contains('\0')) return result_t::fail(status_t::invalid_argument_k, "a shard path holds a NUL byte");
    try {
        std::vector<std::string> path_texts; // vendor-boundary: cuDF's source_info takes std::strings
        std::vector<std::string> names;      // vendor-boundary: cuDF's column selector takes std::strings
        path_texts.reserve(paths.size());
        names.reserve(columns.size());
        for (sz::string_view_t path : paths) path_texts.emplace_back(path.data(), path.size());
        for (sz::string_view_t column : columns) names.emplace_back(column.data(), column.size());
        switch_to_gpu_t gpu_scope {stream}; // cuDF allocates and launches in the current context
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed for the read");
        // GPU Direct reads land out of stream order, in pool memory a queued kernel may still read.
        if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
            return result_t::fail(status_t::device_error_k, "stream drain before the read failed");
        cuda::stream_ref const cudf_stream {stream.get()};

        // A column the sidecar lacks is each row's member of its sibling tar, read through the byte
        // index, whose two columns take its place and the end of the read.
        auto const footer = cudf::io::read_parquet_metadata(cudf::io::source_info {path_texts.front()});
        auto const &fields = footer.schema().root().children();
        auto const has_field = [&](std::string_view name) {
            std::string_view const top = name.substr(0, name.find('.')); // a dotted name selects a struct's leaf
            return std::any_of(fields.begin(), fields.end(), [&](auto const &field) { return field.name() == top; });
        };
        auto const missing = std::find_if(names.begin(), names.end(),
                                          [&](auto const &name) { return !has_field(name); });
        std::size_t const tar_column = static_cast<std::size_t>(missing - names.begin());
        if (missing != names.end() && has_field(tar_offset_column_k)) {
            *missing = tar_offset_column_k;
            names.emplace_back(tar_length_column_k);
        }

        // KvikIO / GDS reads only the named columns' chunks straight to device, and decode is on
        // the GPU. Each shard reads alone, since cuDF reads several only when every column matches,
        // and the shards' tables concatenate.
        std::vector<std::unique_ptr<cudf::table>> parts; // vendor-boundary
        std::vector<cudf::table_view> part_views;        // vendor-boundary
        std::vector<std::size_t> rows_per_shard;         // vendor-boundary
        for (std::size_t path = 0; path != path_texts.size(); ++path) {
            auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info {path_texts[path]})
                               .column_names(names)
                               .build();
            if (!row_groups.empty()) {
                std::vector<cudf::size_type> groups; // vendor-boundary
                groups.assign(row_groups[path].begin(), row_groups[path].end());
                options.set_row_groups({std::move(groups)});
            }
            auto loaded = cudf::io::read_parquet(options, cudf_stream);
            if (loaded.tbl == nullptr || static_cast<std::size_t>(loaded.tbl->num_columns()) != names.size())
                return result_t::fail(status_t::invalid_argument_k, "read of {} returned {} of {} requested columns",
                                      paths[path], loaded.tbl == nullptr ? 0 : loaded.tbl->num_columns(), names.size());
            rows_per_shard.push_back(static_cast<std::size_t>(loaded.tbl->num_rows()));
            part_views.push_back(loaded.tbl->view());
            parts.push_back(std::move(loaded.tbl));
        }
        std::unique_ptr<cudf::table> whole = parts.size() == 1 ? std::move(parts.front())
                                                               : cudf::concatenate(part_views, cudf_stream);
        parts.clear();

        auto state = std::make_unique<state_t>();
        if (st::failed(state->columns.resize(columns.size())))
            return result_t::fail(status_t::bad_alloc_k, "column table allocation failed ({} columns)", columns.size());
        cudf::table_view const table = whole->view();
        bool const tar_backed = names.size() != columns.size();
        for (std::size_t index = 0; index != columns.size(); ++index) {
            cudf::column_view const read = table.column(static_cast<cudf::size_type>(index));
            expected<column_span_t> column =
                tar_backed && index == tar_column
                    ? read_tar_members_(read, table.column(static_cast<cudf::size_type>(columns.size())),
                                        columns[index], paths, rows_per_shard, cudf_stream, state->tar_payload)
                    : adopt_flat_(flatten_column_(read, columns[index], cudf_stream), cudf_stream);
            if (!column) return result_t::fail(std::move(column));
            state->columns[index] = std::move(column.value_);
        }
        // Adopt the table: the borrowed payload / native-offset spans stay valid, now owned by the
        // file, and free stream-ordered on destruction.
        state->table = std::move(whole);
        return result_t {parquet_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::internal_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::internal_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<parquet_length_metadata_t> parquet_file_t::table_metadata(gpu_direct_storage_session_t const &,
                                                                   sz::string_view_t path,
                                                                   sz::strings_view_t complete_columns) noexcept {
    using result_t = expected<parquet_length_metadata_t>;
    try {
        // Footer only — no page decode: every row group's rows and uncompressed bytes.
        std::unique_ptr<cudf::io::datasource> const sources[1] = {
            cudf::io::datasource::create(std::string {std::string_view {path}})}; // vendor-boundary
        std::vector<cudf::io::parquet::FileMetaData> footers;                     // vendor-boundary
        footers = cudf::io::read_parquet_footers(sources);
        cudf::io::parquet::FileMetaData const &footer = footers.front();
        std::size_t const groups = footer.row_groups.size();
        parquet_length_metadata_t length {static_cast<std::size_t>(footer.num_rows), 0, {}, {}};
        st::basic_vector<std::size_t> spans;
        if (st::failed(length.group_rows.resize(groups)) || st::failed(length.group_bytes.resize(groups)) ||
            st::failed(length.group_complete_rows.resize(groups)) || st::failed(spans.resize(groups)))
            return result_t::fail(status_t::bad_alloc_k, "row-group table allocation failed for {}", path);
        bool tar_backed = false;
        // A named column the shard lacks leaves every row incomplete.
        bool const complete = std::all_of(
            complete_columns.begin(), complete_columns.end(), [&](sz::string_view_t column) {
                return column.empty() ||
                       std::any_of(footer.schema.begin(), footer.schema.end(),
                                   [&](auto const &element) { return std::string_view {column} == element.name; });
            });
        for (std::size_t group = 0; group != groups; ++group) {
            cudf::io::parquet::RowGroup const &row_group = footer.row_groups[group];
            length.group_rows[group] = static_cast<std::size_t>(row_group.num_rows);
            length.group_bytes[group] = 0;
            std::size_t incomplete_rows = 0;
            // A sidecar group's members span from its least offset to its greatest plus its longest
            // member, or the whole tar when the writer kept no statistics.
            i64_t first_member = 0, last_member = 0, longest_member = 0;
            std::size_t statistics = 0;
            for (cudf::io::parquet::ColumnChunk const &chunk : row_group.columns) {
                cudf::io::parquet::ColumnChunkMetaData const &chunk_metadata = chunk.meta_data;
                length.group_bytes[group] += static_cast<std::size_t>(chunk_metadata.total_uncompressed_size);
                if (chunk_metadata.path_in_schema.empty()) continue;
                std::string const &name = chunk_metadata.path_in_schema.front(); // vendor-boundary
                bool const named = std::any_of(
                    complete_columns.begin(), complete_columns.end(),
                    [&](sz::string_view_t column) { return std::string_view {column} == name; });
                if (named && chunk_metadata.statistics.null_count)
                    incomplete_rows += static_cast<std::size_t>(*chunk_metadata.statistics.null_count);
                if (chunk_metadata.path_in_schema.size() != 1) continue;
                if (name == tar_offset_column_k) {
                    tar_backed = true;
                    statistics += statistic_(chunk_metadata.statistics.min_value, first_member);
                    statistics += statistic_(chunk_metadata.statistics.max_value, last_member);
                }
                else if (name == tar_length_column_k)
                    statistics += statistic_(chunk_metadata.statistics.max_value, longest_member);
            }
            length.group_complete_rows[group] =
                complete ? length.group_rows[group] - fu::min_of(incomplete_rows, length.group_rows[group]) : 0;
            spans[group] = statistics == 3 ? static_cast<std::size_t>(last_member + longest_member - first_member)
                                           : (std::numeric_limits<std::size_t>::max)();
            length.bytes += length.group_bytes[group];
        }
        if (!tar_backed) return result_t {std::move(length)};

        // A whole-shard read stages the whole tar, and a group's read at most its span.
        sz::string_t tar;
        if (!tar_path_of_(path, tar))
            return result_t::fail(status_t::bad_alloc_k, "tar path allocation failed for {}", path);
        expected<std::size_t> tar_bytes = file_size_bytes(tar.view());
        if (!tar_bytes) return result_t::fail(std::move(tar_bytes));
        length.bytes += *tar_bytes;
        for (std::size_t group = 0; group != groups; ++group)
            length.group_bytes[group] += fu::min_of(spans[group], *tar_bytes);
        return result_t {std::move(length)};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::internal_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::internal_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<parquet_length_metadata_t> parquet_file_t::column_metadata(gpu_direct_storage_session_t const &,
                                                                    sz::string_view_t path,
                                                                    sz::string_view_t column) noexcept {
    using result_t = expected<parquet_length_metadata_t>;
    try {
        // Footer only — no row-group decode: the row count and the column's total
        // uncompressed byte size.
        cudf::io::parquet_metadata const meta = cudf::io::read_parquet_metadata(
            cudf::io::source_info {std::string {path}}); // vendor-boundary
        std::string const name {column};                 // vendor-boundary
        auto const &sizes = meta.columnchunk_metadata();
        auto const found = sizes.find(name);
        if (found == sizes.end())
            return result_t::fail(status_t::invalid_argument_k, "no column named {} in {}", column, path);
        std::size_t bytes = 0;
        for (auto const chunk_bytes : found->second) bytes += static_cast<std::size_t>(chunk_bytes);
        return result_t {parquet_length_metadata_t {static_cast<std::size_t>(meta.num_rows()), bytes}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::internal_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::internal_k, "cuDF raised a non-standard exception at the firewall");
    }
}

/** A decoded JSONL field: the owned cuDF column — extracted from a shard's table, or the
 *  concatenation of several — and its flattened device spans. */
struct jsonl_file_t::state_t {
    std::unique_ptr<cudf::column> field;
    column_span_t span;
};

jsonl_file_t::jsonl_file_t() noexcept = default;
jsonl_file_t::jsonl_file_t(jsonl_file_t &&) noexcept = default;
jsonl_file_t &jsonl_file_t::operator=(jsonl_file_t &&) noexcept = default;
jsonl_file_t::~jsonl_file_t() = default;
jsonl_file_t::jsonl_file_t(std::unique_ptr<state_t> state) noexcept : state_(std::move(state)) {}

std::size_t jsonl_file_t::rows() const noexcept {
    if (!state_) return 0;
    if (!state_->span.groups.empty()) return state_->span.groups.size() - 1;   // list<string>: rows+1 groups
    return state_->span.offsets.empty() ? 0 : state_->span.offsets.size() - 1; // flat string: rows+1 offsets
}
column_view_t jsonl_file_t::column() const noexcept { return view_of_(state_->span); }
std::span<u64_t const> jsonl_file_t::groups() const noexcept {
    return state_ ? state_->span.groups : std::span<u64_t const> {};
}

expected<jsonl_file_t> jsonl_file_t::make(gpu_direct_storage_session_t const &, sz::string_view_t path,
                                          sz::string_view_t field, gpu_stream_t &stream) noexcept {
    using result_t = expected<jsonl_file_t>;
    try {
        switch_to_gpu_t gpu_scope {stream}; // cuDF allocates and launches in the current context
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed for the read");
        // GPU Direct reads land out of stream order, in pool memory a queued kernel may still read.
        if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
            return result_t::fail(status_t::device_error_k, "stream drain before the read failed");
        cuda::stream_ref const cudf_stream {stream.get()};
        // One JSON object per line; KvikIO / GDS streams the bytes, cuDF decodes on the GPU.
        auto options = cudf::io::json_reader_options::builder(
                           cudf::io::source_info {std::string {path}}) // vendor-boundary
                           .lines(true)
                           .build();
        cudf::io::table_with_metadata loaded = cudf::io::read_json(options, cudf_stream);
        if (loaded.tbl == nullptr) return result_t::fail(status_t::io_k, "empty JSONL read of {}", path);

        // Select `field` among the decoded top-level columns by name.
        std::string const wanted {field}; // vendor-boundary
        std::size_t const column_count = loaded.metadata.schema_info.size();
        std::size_t field_index = column_count;
        for (std::size_t index = 0; index != column_count; ++index) // vendor-boundary
            if (loaded.metadata.schema_info[index].name == wanted) {
                field_index = index;
                break;
            }
        if (field_index == column_count)
            return result_t::fail(status_t::invalid_argument_k, "no field {} in JSONL {}", field, path);

        // Own just that field's column and drop the rest of the decoded table.
        std::vector<std::unique_ptr<cudf::column>> released = loaded.tbl->release(); // vendor-boundary
        auto state = std::make_unique<state_t>();
        state->field = std::move(released[field_index]);
        auto span = adopt_flat_(flatten_column_(state->field->view(), field, cudf_stream), cudf_stream);
        if (!span) return result_t::fail(std::move(span));
        state->span = std::move(span.value_);
        return result_t {jsonl_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::internal_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::internal_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<jsonl_file_t> jsonl_file_t::concat(std::span<jsonl_file_t const> parts, gpu_stream_t &stream) noexcept {
    using result_t = expected<jsonl_file_t>;
    if (parts.empty()) return result_t::fail(status_t::invalid_argument_k, "JSONL concat needs at least one shard");
    try {
        switch_to_gpu_t gpu_scope {stream}; // cuDF allocates and launches in the current context
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed for the read");
        cuda::stream_ref const cudf_stream {stream.get()};
        std::vector<cudf::column_view> fields; // vendor-boundary: cudf::concatenate takes a column-view list
        fields.reserve(parts.size());
        for (jsonl_file_t const &part : parts) fields.push_back(part.state_->field->view());
        auto state = std::make_unique<state_t>();
        state->field = cudf::concatenate(fields, cudf_stream); // one copy into a single owned field column
        auto span = adopt_flat_(flatten_column_(state->field->view(), sz::string_view_t {"jsonl", 5}, cudf_stream),
                                cudf_stream);
        if (!span) return result_t::fail(std::move(span));
        state->span = std::move(span.value_);
        return result_t {jsonl_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::internal_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::internal_k, "cuDF raised a non-standard exception at the firewall");
    }
}

struct csv_file_t::state_t {
    std::unique_ptr<cudf::column> field;
    column_span_t span;
};

csv_file_t::csv_file_t() noexcept = default;
csv_file_t::csv_file_t(csv_file_t &&) noexcept = default;
csv_file_t &csv_file_t::operator=(csv_file_t &&) noexcept = default;
csv_file_t::~csv_file_t() = default;
csv_file_t::csv_file_t(std::unique_ptr<state_t> state) noexcept : state_(std::move(state)) {}

std::size_t csv_file_t::rows() const noexcept {
    if (!state_) return 0;
    return state_->span.offsets.empty() ? 0 : state_->span.offsets.size() - 1; // flat string: rows+1 offsets
}
column_view_t csv_file_t::column() const noexcept { return view_of_(state_->span); }
std::span<u64_t const> csv_file_t::groups() const noexcept {
    return state_ ? state_->span.groups : std::span<u64_t const> {};
}

expected<csv_file_t> csv_file_t::make(gpu_direct_storage_session_t const &, sz::string_view_t path,
                                      sz::string_view_t column, char delimiter, csv_header_t header,
                                      gpu_stream_t &stream) noexcept {
    using result_t = expected<csv_file_t>;
    try {
        switch_to_gpu_t gpu_scope {stream}; // cuDF allocates and launches in the current context
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed for the read");
        // GPU Direct reads land out of stream order, in pool memory a queued kernel may still read.
        if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
            return result_t::fail(status_t::device_error_k, "stream drain before the read failed");
        cuda::stream_ref const cudf_stream {stream.get()};
        std::string const name {column}; // vendor-boundary
        // Read only the wanted column, forced to string so type inference cannot surprise us;
        // KvikIO / GDS streams the bytes, cuDF decodes on the GPU.
        auto options =
            cudf::io::csv_reader_options::builder(cudf::io::source_info {std::string {path}}) // vendor-boundary
                .delimiter(delimiter)
                .header(header == csv_header_t::named_k ? 0 : -1)
                .use_cols_names({name})                                                         // vendor-boundary
                .dtypes(std::vector<cudf::data_type> {cudf::data_type {cudf::type_id::STRING}}) // vendor-boundary
                .build();
        cudf::io::table_with_metadata loaded = cudf::io::read_csv(options, cudf_stream);
        if (loaded.tbl == nullptr || loaded.tbl->num_columns() == 0)
            return result_t::fail(status_t::invalid_argument_k, "no column {} in CSV {}", column, path);

        // Own the one decoded column and drop the rest of the table.
        std::vector<std::unique_ptr<cudf::column>> released = loaded.tbl->release(); // vendor-boundary
        auto state = std::make_unique<state_t>();
        state->field = std::move(released[0]);
        auto span = adopt_flat_(flatten_column_(state->field->view(), column, cudf_stream), cudf_stream);
        if (!span) return result_t::fail(std::move(span));
        state->span = std::move(span.value_);
        return result_t {csv_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::internal_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::internal_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<csv_file_t> csv_file_t::concat(std::span<csv_file_t const> parts, gpu_stream_t &stream) noexcept {
    using result_t = expected<csv_file_t>;
    if (parts.empty()) return result_t::fail(status_t::invalid_argument_k, "CSV concat needs at least one shard");
    try {
        switch_to_gpu_t gpu_scope {stream}; // cuDF allocates and launches in the current context
        if (!gpu_scope.ok()) return result_t::fail(status_t::device_error_k, "context entry failed for the read");
        cuda::stream_ref const cudf_stream {stream.get()};
        std::vector<cudf::column_view> columns; // vendor-boundary: cudf::concatenate takes a column-view list
        columns.reserve(parts.size());
        for (csv_file_t const &part : parts) columns.push_back(part.state_->field->view());
        auto state = std::make_unique<state_t>();
        state->field = cudf::concatenate(columns, cudf_stream); // one copy into a single owned column
        auto span = adopt_flat_(flatten_column_(state->field->view(), sz::string_view_t {"csv", 3}, cudf_stream),
                                cudf_stream);
        if (!span) return result_t::fail(std::move(span));
        state->span = std::move(span.value_);
        return result_t {csv_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::internal_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::internal_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<std::size_t> ul::file_size_bytes(sz::string_view_t path) noexcept {
    using result_t = expected<std::size_t>;
    sz::string_t text; // null-terminated for the C stat call
    if (sz::failed(text.assign(path))) return result_t::fail(status_t::bad_alloc_k, "path allocation failed");
    struct ::stat info {}; // vendor-boundary: POSIX stat
    if (::stat(text.c_str(), &info) != 0) return result_t::fail(status_t::io_k, "stat failed for {}", path);
    return result_t {static_cast<std::size_t>(info.st_size)};
}

/** Whether @p path names a directory once every symlink resolves. */
[[nodiscard]] static bool is_directory_(sz::string_t const &path) noexcept {
    struct ::stat info {}; // vendor-boundary: POSIX stat
    return ::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

expected<st::basic_vector<sz::string_t>> ul::expand_glob(sz::string_view_t path_or_glob) noexcept {
    using result_t = expected<st::basic_vector<sz::string_t>>;

    // Wildcards may sit in ANY component — the mixture manifests spell mid-path patterns like
    // `data/*/train/*.parquet` — so the expansion walks the path left to right, forking the
    // running directory prefixes at every patterned component and matching files at the last one.
    // A forked branch that lacks a later component simply contributes nothing; only a wholly
    // empty match fails.
    st::basic_vector<sz::string_view_t> components;
    for (sz::string_view_t component : path_or_glob.split(sz::byteset_t {"/", 1}))
        if (!component.empty() && st::failed(components.push_back(std::move(component))))
            return result_t::fail(status_t::bad_alloc_k, "glob component table allocation failed");
    if (components.size() == 0) return result_t::fail(status_t::invalid_argument_k, "no files match {}", path_or_glob);

    bool const absolute = path_or_glob.starts_with(sz::string_view_t {"/", 1});
    st::basic_vector<sz::string_t> prefixes;
    {
        sz::string_t seed_prefix;
        if (sz::failed(seed_prefix.assign(absolute ? sz::string_view_t {"/", 1} : sz::string_view_t {".", 1})) ||
            st::failed(prefixes.push_back(std::move(seed_prefix))))
            return result_t::fail(status_t::bad_alloc_k, "glob prefix allocation failed");
    }

    st::basic_vector<sz::string_t> paths;
    sz::byteset_t const wildcards {"*?[", 3};
    for (std::size_t level = 0; level != components.size(); ++level) {
        sz::string_view_t const component = components[level];
        bool const last = level + 1 == components.size();
        bool const patterned = component.find_first_of(wildcards) != sz::string_view_t::npos;
        bool const recursive = component == sz::string_view_t {"**", 2};
        if (recursive && last)
            return result_t::fail(status_t::invalid_argument_k, "{} ends in **, not in a file pattern", path_or_glob);

        // A literal directory component descends every prefix without touching the filesystem — the
        // next level's opendir discovers a missing branch and drops it.
        if (!patterned && !last) {
            for (std::size_t index = 0; index != prefixes.size(); ++index)
                if (sz::failed(prefixes[index].append(sz::string_view_t {"/", 1})) ||
                    sz::failed(prefixes[index].append(component)))
                    return result_t::fail(status_t::bad_alloc_k, "glob prefix growth failed");
            continue;
        }

        sz::string_t pattern_text; // null-terminated for fnmatch
        if (sz::failed(pattern_text.assign(component)))
            return result_t::fail(status_t::bad_alloc_k, "glob pattern allocation failed");
        // `**` keeps each prefix and adds every real directory below, as a symlink could loop.
        st::basic_vector<sz::string_t> next_prefixes;
        if (recursive) next_prefixes = std::move(prefixes);
        st::basic_vector<sz::string_t> const &walked = recursive ? next_prefixes : prefixes;
        for (std::size_t index = 0; index != walked.size(); ++index) {
            ::DIR *const handle = ::opendir(walked[index].c_str());
            if (handle == nullptr) continue; // a forked branch without this component matches nothing
            for (::dirent const *entry = ::readdir(handle); entry != nullptr; entry = ::readdir(handle)) {
                if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) continue;
                if (!recursive && ::fnmatch(pattern_text.c_str(), entry->d_name, 0) != 0) continue;
                sz::string_t full;
                if (sz::failed(full.assign(walked[index].view())) ||
                    sz::failed(full.append(sz::string_view_t {"/", 1})) ||
                    sz::failed(full.append(sz::string_view_t {entry->d_name}))) {
                    ::closedir(handle);
                    return result_t::fail(status_t::bad_alloc_k, "path string allocation failed");
                }
                // Shared filesystems report DT_UNKNOWN, and dataset folders are often symlinks.
                bool const resolves = entry->d_type == DT_LNK || entry->d_type == DT_UNKNOWN;
                bool const is_directory = entry->d_type == DT_DIR || (!recursive && resolves && is_directory_(full));
                if (last == is_directory) continue; // files close the walk; directories continue it
                if (st::failed(last ? paths.push_back(std::move(full)) : next_prefixes.push_back(std::move(full)))) {
                    ::closedir(handle);
                    return result_t::fail(status_t::bad_alloc_k, "path string allocation failed");
                }
            }
            ::closedir(handle);
        }
        if (!last) prefixes = std::move(next_prefixes);
    }
    if (paths.empty()) return result_t::fail(status_t::invalid_argument_k, "no files match {}", path_or_glob);

    // readdir order is filesystem-dependent; sort so the shard order is deterministic
    // across every rank.
    std::sort(paths.begin(), paths.end());
    return result_t {std::move(paths)};
}
