/**
 *  @brief The dataset-shard readers' one firewalled body: @ref parquet_file_t and @ref jsonl_file_t .
 *      cuDF and RMM throw and allocate freely, so every call to them lives inside this translation unit's
 *      @c try blocks and their types stay behind the readers' pimpls. See the header for the contract.
 *  @file cuda_loaders/dataset_shards.cpp
 */

#include <algorithm> // `std::sort` — the deterministic shard order
#include <span>      // `std::span` — the borrowed payload / offsets
#include <string>
#include <vector>

#include <dirent.h>   // vendor-boundary: POSIX `opendir` / `readdir` — the raw directory scan, no glob to link
#include <fnmatch.h>  // vendor-boundary: POSIX `fnmatch` — the shard filename pattern match
#include <sys/stat.h> // vendor-boundary: POSIX `stat` — the footerless formats' byte-size file weight

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>         // `cudf::concatenate` — merge decoded JSONL shards into one field
#include <cudf/io/csv.hpp>              // `cudf::io::read_csv`, `csv_reader_options`
#include <cudf/io/json.hpp>             // `cudf::io::read_json`, `json_reader_options`
#include <cudf/io/parquet.hpp>          // `cudf::io::read_parquet`, `parquet_reader_options`
#include <cudf/io/parquet_metadata.hpp> // `cudf::io::read_parquet_metadata`, `parquet_metadata`
#include <cudf/io/types.hpp>            // `cudf::io::source_info`
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp> // `cudf::cast` — the one offset widening
#include <rmm/cuda_stream_view.hpp>

#include "cuda_loaders/dataset_shards.hpp"

#include "cuda_core/array.hpp" // `array` — the owned per-column table

namespace uc = unum::uform::core;
namespace ul = unum::uform::loaders;
using namespace uc;
using namespace ul;

/** @brief One decoded column's device-resident spans and the cuDF columns that back them — everything
 *      RAII, nothing hand-freed. A shared @b representation (not a shared type) behind both the Parquet
 *      and JSONL readers: the payload and offsets are @c std::span borrows into an owned cuDF buffer. */
struct column_span_t {
    /** @brief cuDF's widened leaf-offset column, owned; the leaf offsets are always cast to @c u64 . */
    std::unique_ptr<cudf::column> widened_offsets;
    /** @brief cuDF's widened group-offset column, owned; non-null only for a nested @c list<string> . */
    std::unique_ptr<cudf::column> widened_groups;
    /** @brief The column's leaf bytes, back to back — borrowed from the owning table / column. */
    std::span<b8_t const> payload;
    /** @brief @b [elements+1] leaf offsets — owned by @ref widened_offsets . */
    std::span<u64_t const> offsets;
    /** @brief @b [rows+1] row → element offsets for a nested column — owned by @ref widened_groups ;
     *      empty for a flat column. */
    std::span<u64_t const> groups;
};

/** @brief What a decoded Parquet file owns and lends: the cuDF table (every column's payload + native
 *      offsets) and one @ref column_span_t per requested column, in request order. */
struct parquet_file_t::state_t {
    std::unique_ptr<cudf::table> table;
    array<column_span_t> columns;
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
column_view_t parquet_file_t::column(std::size_t index) const noexcept {
    column_span_t const &column = state_->columns[index];
    return {{column.payload.data(), column.payload.size()}, {column.offsets.data(), column.offsets.size()}};
}
std::span<u64_t const> parquet_file_t::groups(std::size_t index) const noexcept {
    return state_ ? state_->columns[index].groups : std::span<u64_t const> {};
}

/** @brief A decoded column flattened to its leaf payload, the leaf offsets child, and — for a nested
 *      @c list<string> column — the outer row → element offsets child. A @c string and a
 *      @c list<uint8> column differ only in which child holds the payload and carry no groups; a
 *      @c list<string> column carries both offset levels. */
struct flat_column_t {
    /** @brief The column's leaf bytes, back to back. */
    std::span<b8_t const> payload;
    /** @brief The leaf offsets child — @c elements+1 entries, 32- or 64-bit. */
    cudf::column_view offsets;
    /** @brief The outer row → element offsets child for a nested column; a default (empty-type)
     *      column_view for a flat one. */
    cudf::column_view groups;
};

/** @brief Flatten the @p name column @p decoded to its payload and offset levels, or a
 *      @c shape_unsupported_k failure naming it when it is not one of @c string , @c list<uint8> , or
 *      @c list<string> . */
[[nodiscard]] static expected<flat_column_t> flatten_column_(cudf::column_view const &decoded, sz::string_view name,
                                                             rmm::cuda_stream_view stream) noexcept {
    using result_t = expected<flat_column_t>;
    if (decoded.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view const strings {decoded};
        auto const *payload = reinterpret_cast<b8_t const *>(strings.chars_begin(stream));
        return result_t {
            flat_column_t {{payload, static_cast<std::size_t>(strings.chars_size(stream))}, strings.offsets(), {}}};
    }
    if (decoded.type().id() == cudf::type_id::LIST) {
        cudf::lists_column_view const list {decoded};
        cudf::column_view const child = list.child();
        // list<uint8> — a flat media blob: the child bytes are the payload, the list offsets the leaf.
        if (child.type().id() == cudf::type_id::UINT8 || child.type().id() == cudf::type_id::INT8) {
            auto const *payload = static_cast<b8_t const *>(child.head());
            return result_t {flat_column_t {{payload, static_cast<std::size_t>(child.size())}, list.offsets(), {}}};
        }
        // list<string> — a nested column (K captions / negatives per row): the string child's chars are
        // the leaf payload, its offsets the element → bytes level, and the list offsets the row → element
        // groups.
        if (child.type().id() == cudf::type_id::STRING) {
            cudf::strings_column_view const strings {child};
            auto const *payload = reinterpret_cast<b8_t const *>(strings.chars_begin(stream));
            return result_t {flat_column_t {{payload, static_cast<std::size_t>(strings.chars_size(stream))},
                                            strings.offsets(),
                                            list.offsets()}};
        }
    }
    return result_t::fail(status_t::shape_unsupported_k, "column {} is not string, list<uint8>, or list<string>", name);
}

/** @brief Own a @c u64_t copy of @p offsets — cuDF decodes 32- or 64-bit @b signed offsets, and byte
 *      positions are unsigned sizes, so the one @c cudf::cast to @c UINT64 both widens and unsigns.
 *      Writes the borrow into @p span and the owning column into @p owner . */
static void widen_offsets_(cudf::column_view const &offsets, rmm::cuda_stream_view stream, std::span<u64_t const> &span,
                           std::unique_ptr<cudf::column> &owner) {
    owner = cudf::cast(offsets, cudf::data_type {cudf::type_id::UINT64}, stream);
    span = {owner->view().data<u64_t>(), static_cast<std::size_t>(offsets.size())};
}

expected<parquet_file_t> parquet_file_t::make(gpu_direct_storage_session_t const &storage, sz::string_view path,
                                              std::span<sz::string_view const> columns,
                                              gpu_stream_t const &stream) noexcept {
    return make(storage, {&path, 1}, columns, stream); // The single-shard convenience over the concatenating read.
}

expected<parquet_file_t> parquet_file_t::make(gpu_direct_storage_session_t const &,
                                              std::span<sz::string_view const> paths,
                                              std::span<sz::string_view const> columns,
                                              gpu_stream_t const &stream) noexcept {
    using result_t = expected<parquet_file_t>;
    if (paths.empty() || columns.empty())
        return result_t::fail(status_t::invalid_input_k, "a read needs at least one shard path and one column");
    try {
        std::vector<std::string> path_texts; // vendor-boundary: cuDF's source_info takes std::strings
        std::vector<std::string> names;      // vendor-boundary: cuDF's column selector takes std::strings
        path_texts.reserve(paths.size());
        names.reserve(columns.size());
        for (sz::string_view path : paths) path_texts.emplace_back(path.data(), path.size());
        for (sz::string_view column : columns) names.emplace_back(column.data(), column.size());
        rmm::cuda_stream_view const cudf_stream {reinterpret_cast<cudaStream_t>(stream.get())};

        // KvikIO / GDS reads only the named columns' chunks straight to device; multiple shards decode
        // into one concatenated table, and decode is on the GPU.
        auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info {std::move(path_texts)})
                           .column_names(names)
                           .build();
        auto loaded = cudf::io::read_parquet(options, cudf_stream);
        if (loaded.tbl == nullptr || static_cast<std::size_t>(loaded.tbl->num_columns()) != columns.size())
            return result_t::fail(status_t::invalid_input_k, "read returned {} of {} requested columns",
                                  loaded.tbl == nullptr ? 0 : loaded.tbl->num_columns(), columns.size());

        auto state = std::make_unique<state_t>();
        if (state->columns.try_resize(columns.size()) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "column table allocation failed ({} columns)", columns.size());
        for (std::size_t index = 0; index != columns.size(); ++index) {
            auto flat = flatten_column_(loaded.tbl->view().column(static_cast<cudf::size_type>(index)), columns[index],
                                        cudf_stream);
            if (!flat) return result_t::fail(std::move(flat));

            // 64-bit offsets everywhere, both levels: borrow cuDF's when already i64, else keep the
            // column cuDF's own `cast` produced — the offsets stay in a cuDF-owned buffer, no copy.
            column_span_t &column = state->columns[index];
            column.payload = flat->payload;
            widen_offsets_(flat->offsets, cudf_stream, column.offsets, column.widened_offsets);
            if (flat->groups.type().id() != cudf::type_id::EMPTY)
                widen_offsets_(flat->groups, cudf_stream, column.groups, column.widened_groups);
        }
        // Adopt the table: the borrowed payload / native-offset spans stay valid, now owned by the file,
        // and free stream-ordered on destruction.
        state->table = std::move(loaded.tbl);
        return result_t {parquet_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::other_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::other_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<parquet_length_metadata_t> parquet_file_t::table_metadata(gpu_direct_storage_session_t const &,
                                                                   sz::string_view path) noexcept {
    using result_t = expected<parquet_length_metadata_t>;
    try {
        // Footer only — no row-group decode: the row count and the whole table's uncompressed byte size.
        cudf::io::parquet_metadata const meta = cudf::io::read_parquet_metadata(
            cudf::io::source_info {std::string {path.data(), path.size()}}); // vendor-boundary
        std::size_t bytes = 0;
        for (auto const &column : meta.columnchunk_metadata()) // vendor-boundary
            for (auto const chunk_bytes : column.second) bytes += static_cast<std::size_t>(chunk_bytes);
        return result_t {parquet_length_metadata_t {static_cast<std::size_t>(meta.num_rows()), bytes}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::other_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::other_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<parquet_length_metadata_t> parquet_file_t::column_metadata(gpu_direct_storage_session_t const &,
                                                                    sz::string_view path,
                                                                    sz::string_view column) noexcept {
    using result_t = expected<parquet_length_metadata_t>;
    try {
        // Footer only — no row-group decode: the row count and the column's total uncompressed byte size.
        cudf::io::parquet_metadata const meta = cudf::io::read_parquet_metadata(
            cudf::io::source_info {std::string {path.data(), path.size()}}); // vendor-boundary
        std::string const name {column.data(), column.size()};               // vendor-boundary
        auto const &sizes = meta.columnchunk_metadata();
        auto const found = sizes.find(name);
        if (found == sizes.end())
            return result_t::fail(status_t::invalid_input_k, "no column named {} in {}", column, path);
        std::size_t bytes = 0;
        for (auto const chunk_bytes : found->second) bytes += static_cast<std::size_t>(chunk_bytes);
        return result_t {parquet_length_metadata_t {static_cast<std::size_t>(meta.num_rows()), bytes}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::other_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::other_k, "cuDF raised a non-standard exception at the firewall");
    }
}

/** @brief Widen a flattened column's offsets into an owned @ref column_span_t — the decode tail both
 *      readers share once cuDF hands back a @ref flat_column_t . */
static column_span_t adopt_flat_(flat_column_t const &flat, rmm::cuda_stream_view stream) {
    column_span_t column;
    column.payload = flat.payload;
    widen_offsets_(flat.offsets, stream, column.offsets, column.widened_offsets);
    if (flat.groups.type().id() != cudf::type_id::EMPTY)
        widen_offsets_(flat.groups, stream, column.groups, column.widened_groups);
    return column;
}

/** @brief A decoded JSONL field: the owned cuDF column — extracted from a shard's table, or the
 *      concatenation of several — and its flattened device spans. */
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
column_view_t jsonl_file_t::column() const noexcept {
    column_span_t const &field = state_->span;
    return {{field.payload.data(), field.payload.size()}, {field.offsets.data(), field.offsets.size()}};
}
std::span<u64_t const> jsonl_file_t::groups() const noexcept {
    return state_ ? state_->span.groups : std::span<u64_t const> {};
}

expected<jsonl_file_t> jsonl_file_t::make(gpu_direct_storage_session_t const &, sz::string_view path,
                                          sz::string_view field, gpu_stream_t const &stream) noexcept {
    using result_t = expected<jsonl_file_t>;
    try {
        rmm::cuda_stream_view const cudf_stream {reinterpret_cast<cudaStream_t>(stream.get())};
        // One JSON object per line; KvikIO / GDS streams the bytes, cuDF decodes on the GPU.
        auto options = cudf::io::json_reader_options::builder(
                           cudf::io::source_info {std::string {path.data(), path.size()}}) // vendor-boundary
                           .lines(true)
                           .build();
        cudf::io::table_with_metadata loaded = cudf::io::read_json(options, cudf_stream);
        if (loaded.tbl == nullptr) return result_t::fail(status_t::io_error_k, "empty JSONL read of {}", path);

        // Select `field` among the decoded top-level columns by name.
        std::string const wanted {field.data(), field.size()}; // vendor-boundary
        std::size_t const column_count = loaded.metadata.schema_info.size();
        std::size_t field_index = column_count;
        for (std::size_t index = 0; index != column_count; ++index) // vendor-boundary
            if (loaded.metadata.schema_info[index].name == wanted) {
                field_index = index;
                break;
            }
        if (field_index == column_count)
            return result_t::fail(status_t::invalid_input_k, "no field {} in JSONL {}", field, path);

        // Own just that field's column and drop the rest of the decoded table.
        std::vector<std::unique_ptr<cudf::column>> released = loaded.tbl->release(); // vendor-boundary
        auto state = std::make_unique<state_t>();
        state->field = std::move(released[field_index]);
        auto flat = flatten_column_(state->field->view(), field, cudf_stream);
        if (!flat) return result_t::fail(std::move(flat));
        state->span = adopt_flat_(*flat, cudf_stream);
        return result_t {jsonl_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::other_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::other_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<jsonl_file_t> jsonl_file_t::concat(std::span<jsonl_file_t const> parts, gpu_stream_t const &stream) noexcept {
    using result_t = expected<jsonl_file_t>;
    if (parts.empty()) return result_t::fail(status_t::invalid_input_k, "JSONL concat needs at least one shard");
    try {
        rmm::cuda_stream_view const cudf_stream {reinterpret_cast<cudaStream_t>(stream.get())};
        std::vector<cudf::column_view> fields; // vendor-boundary: cudf::concatenate takes a column-view list
        fields.reserve(parts.size());
        for (jsonl_file_t const &part : parts) fields.push_back(part.state_->field->view());
        auto state = std::make_unique<state_t>();
        state->field = cudf::concatenate(fields, cudf_stream); // one copy into a single owned field column
        auto flat = flatten_column_(state->field->view(), sz::string_view {"jsonl", 5}, cudf_stream);
        if (!flat) return result_t::fail(std::move(flat));
        state->span = adopt_flat_(*flat, cudf_stream);
        return result_t {jsonl_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::other_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::other_k, "cuDF raised a non-standard exception at the firewall");
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
column_view_t csv_file_t::column() const noexcept {
    column_span_t const &field = state_->span;
    return {{field.payload.data(), field.payload.size()}, {field.offsets.data(), field.offsets.size()}};
}
std::span<u64_t const> csv_file_t::groups() const noexcept {
    return state_ ? state_->span.groups : std::span<u64_t const> {};
}

expected<csv_file_t> csv_file_t::make(gpu_direct_storage_session_t const &, sz::string_view path,
                                      sz::string_view column, char delimiter, bool has_header,
                                      gpu_stream_t const &stream) noexcept {
    using result_t = expected<csv_file_t>;
    try {
        rmm::cuda_stream_view const cudf_stream {reinterpret_cast<cudaStream_t>(stream.get())};
        std::string const name {column.data(), column.size()}; // vendor-boundary
        // Read only the wanted column, forced to string so type inference cannot surprise us; KvikIO /
        // GDS streams the bytes, cuDF decodes on the GPU.
        auto options =
            cudf::io::csv_reader_options::builder(
                cudf::io::source_info {std::string {path.data(), path.size()}}) // vendor-boundary
                .delimiter(delimiter)
                .header(has_header ? 0 : -1)
                .use_cols_names({name})                                                         // vendor-boundary
                .dtypes(std::vector<cudf::data_type> {cudf::data_type {cudf::type_id::STRING}}) // vendor-boundary
                .build();
        cudf::io::table_with_metadata loaded = cudf::io::read_csv(options, cudf_stream);
        if (loaded.tbl == nullptr || loaded.tbl->num_columns() == 0)
            return result_t::fail(status_t::invalid_input_k, "no column {} in CSV {}", column, path);

        // Own the one decoded column and drop the rest of the table.
        std::vector<std::unique_ptr<cudf::column>> released = loaded.tbl->release(); // vendor-boundary
        auto state = std::make_unique<state_t>();
        state->field = std::move(released[0]);
        auto flat = flatten_column_(state->field->view(), column, cudf_stream);
        if (!flat) return result_t::fail(std::move(flat));
        state->span = adopt_flat_(*flat, cudf_stream);
        return result_t {csv_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::other_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::other_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<csv_file_t> csv_file_t::concat(std::span<csv_file_t const> parts, gpu_stream_t const &stream) noexcept {
    using result_t = expected<csv_file_t>;
    if (parts.empty()) return result_t::fail(status_t::invalid_input_k, "CSV concat needs at least one shard");
    try {
        rmm::cuda_stream_view const cudf_stream {reinterpret_cast<cudaStream_t>(stream.get())};
        std::vector<cudf::column_view> columns; // vendor-boundary: cudf::concatenate takes a column-view list
        columns.reserve(parts.size());
        for (csv_file_t const &part : parts) columns.push_back(part.state_->field->view());
        auto state = std::make_unique<state_t>();
        state->field = cudf::concatenate(columns, cudf_stream); // one copy into a single owned column
        auto flat = flatten_column_(state->field->view(), sz::string_view {"csv", 3}, cudf_stream);
        if (!flat) return result_t::fail(std::move(flat));
        state->span = adopt_flat_(*flat, cudf_stream);
        return result_t {csv_file_t {std::move(state)}};
    }
    catch (std::exception const &error) {
        return result_t::fail(status_t::other_k, "cuDF raised at the firewall: {}", error.what());
    }
    catch (...) {
        return result_t::fail(status_t::other_k, "cuDF raised a non-standard exception at the firewall");
    }
}

expected<std::size_t> ul::file_size_bytes(sz::string_view path) noexcept {
    using result_t = expected<std::size_t>;
    sz::string text; // null-terminated for the C stat call
    if (!text.try_assign(path)) return result_t::fail(status_t::bad_alloc_k, "path allocation failed");
    struct ::stat info {}; // vendor-boundary: POSIX stat
    if (::stat(text.c_str(), &info) != 0) return result_t::fail(status_t::io_error_k, "stat failed for {}", path);
    return result_t {static_cast<std::size_t>(info.st_size)};
}

expected<array<sz::string>> ul::expand_glob(sz::string_view path_or_glob) noexcept {
    using result_t = expected<array<sz::string>>;

    // Split "directory/pattern" at the last '/' with StringZilla's rpartition; a bare filename has no
    // slash, so the match comes back empty and the directory defaults to ".".
    auto const parts = path_or_glob.rpartition('/');
    sz::string_view const directory = parts.match.empty() ? sz::string_view {".", 1} : parts.before;
    sz::string_view const pattern = parts.match.empty() ? parts.before : parts.after;

    sz::string directory_text, pattern_text; // null-terminated for opendir / fnmatch
    if (!directory_text.try_assign(directory) || !pattern_text.try_assign(pattern))
        return result_t::fail(status_t::bad_alloc_k, "glob split allocation failed");

    ::DIR *const handle = ::opendir(directory_text.c_str());
    if (handle == nullptr) return result_t::fail(status_t::io_error_k, "cannot open directory {}", directory);

    array<sz::string> paths;
    for (::dirent const *entry = ::readdir(handle); entry != nullptr; entry = ::readdir(handle)) {
        if (entry->d_type == DT_DIR) continue;
        if (::fnmatch(pattern_text.c_str(), entry->d_name, 0) != 0) continue;
        sz::string full;
        if (!full.try_assign(directory) || !full.try_append(sz::string_view {"/", 1}) ||
            !full.try_append(sz::string_view {entry->d_name}) ||
            paths.try_push_back(std::move(full)) != status_t::success_k) {
            ::closedir(handle);
            return result_t::fail(status_t::bad_alloc_k, "path string allocation failed");
        }
    }
    ::closedir(handle);
    if (paths.empty()) return result_t::fail(status_t::invalid_input_k, "no files match {}", path_or_glob);

    // readdir order is filesystem-dependent; sort so the shard order is deterministic across every rank.
    std::sort(paths.begin(), paths.end());
    return result_t {std::move(paths)};
}
