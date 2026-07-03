/**
 *  @brief The Parquet cell's one firewalled body: @ref parquet_file_t::make . cuDF and RMM throw and
 *      allocate freely, so every call to them lives inside this translation unit's one @c try and
 *      their types stay behind @ref parquet_file_t::state_t . See the header for the contract.
 *  @file cuda_loaders/parquet.cpp
 */
#include "parquet.hpp"

#include <optional> // `std::optional` — the flatten result, in place of out-parameters
#include <span>     // `std::span` — the borrowed payload / offsets
#include <string>
#include <vector>

#include "array.cuh" // `array` — the owned per-column table

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp> // `cudf::io::read_parquet`, `parquet_reader_options`
#include <cudf/io/types.hpp>   // `cudf::io::source_info`
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp> // `cudf::cast` — the one offset widening
#include <rmm/cuda_stream_view.hpp>

namespace unum::uform::loaders {

/** @brief What a decoded file owns and lends: the cuDF table (payloads and native offsets) and, per
 *      column, cuDF's own widened-offset column when one was needed — everything RAII, nothing
 *      hand-freed. The payload and offsets each column lends out are @c std::span borrows into one of
 *      those owned buffers. */
struct parquet_file_t::state_t {
    struct column_span_t {
        /** @brief cuDF's widened-offset column, owned; non-null only when the decoded offsets were 32-bit. */
        std::unique_ptr<cudf::column> widened_offsets;
        /** @brief The column's row bytes, back to back — borrowed from @ref table . */
        std::span<b8_t const> payload;
        /** @brief @b [rows+1] prefix offsets — borrowed from @ref table or @ref widened_offsets . */
        std::span<i64_t const> offsets;
    };
    /** @brief The decoded table — owns every column's payload and its native offsets. */
    std::unique_ptr<cudf::table> table;
    /** @brief One entry per requested column, in request order. */
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
    state_t::column_span_t const &column = state_->columns[index];
    return {{column.payload.data(), column.payload.size()}, {column.offsets.data(), column.offsets.size()}};
}

/** @brief A decoded column flattened to its payload span and its offsets child — a @c string and a
 *      @c list<uint8> column differ only in which child holds each. */
struct flat_column_t {
    /** @brief The column's row bytes, back to back. */
    std::span<b8_t const> payload;
    /** @brief The offsets child — @c rows+1 entries, 32- or 64-bit. */
    cudf::column_view offsets;
};

/** @brief Flatten @p decoded , or @c nullopt when it is neither a @c string nor a @c list<uint8> . */
[[nodiscard]] static std::optional<flat_column_t> flatten_column_(cudf::column_view const &decoded,
                                                                  rmm::cuda_stream_view stream) noexcept {
    if (decoded.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view const strings {decoded};
        auto const *payload = reinterpret_cast<b8_t const *>(strings.chars_begin(stream));
        return flat_column_t {{payload, static_cast<std::size_t>(strings.chars_size(stream))}, strings.offsets()};
    }
    if (decoded.type().id() == cudf::type_id::LIST) {
        cudf::lists_column_view const list {decoded};
        cudf::column_view const child = list.child();
        if (child.type().id() != cudf::type_id::UINT8 && child.type().id() != cudf::type_id::INT8) return std::nullopt;
        auto const *payload = static_cast<b8_t const *>(child.head());
        return flat_column_t {{payload, static_cast<std::size_t>(child.size())}, list.offsets()};
    }
    return std::nullopt;
}

expected<parquet_file_t> parquet_file_t::make(gpu_direct_storage_session_t const &, sz::string_view path,
                                              std::span<sz::string_view const> columns,
                                              gpu_stream_t const &stream) noexcept {
    using result_t = expected<parquet_file_t>;
    if (path.empty() || columns.empty())
        return result_t::fail(status_t::invalid_input_k, "a read needs a shard path and at least one column");
    try {
        std::vector<std::string> names; // vendor-boundary: cuDF's column selector takes std::strings
        names.reserve(columns.size());
        for (sz::string_view column : columns) names.emplace_back(column.data(), column.size());
        rmm::cuda_stream_view const cudf_stream {reinterpret_cast<cudaStream_t>(stream.get())};

        // KvikIO / GDS reads only the named columns' chunks straight to device; decode is on the GPU.
        auto options = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info {std::string {path.data(), path.size()}}) // vendor-boundary
                           .column_names(names)
                           .build();
        auto loaded = cudf::io::read_parquet(options, cudf_stream);
        if (loaded.tbl == nullptr || static_cast<std::size_t>(loaded.tbl->num_columns()) != columns.size())
            return result_t::fail(status_t::invalid_input_k, "read returned {} of {} requested columns from {}",
                                  loaded.tbl == nullptr ? 0 : loaded.tbl->num_columns(), columns.size(), path);

        auto state = std::make_unique<state_t>();
        if (state->columns.try_resize(columns.size()) != status_t::success_k)
            return result_t::fail(status_t::bad_alloc_k, "column table allocation failed ({} columns)", columns.size());
        for (std::size_t index = 0; index != columns.size(); ++index) {
            std::optional<flat_column_t> const flat = flatten_column_(
                loaded.tbl->view().column(static_cast<cudf::size_type>(index)), cudf_stream);
            if (!flat)
                return result_t::fail(status_t::shape_unsupported_k, "column {} is not string or list<uint8>",
                                      columns[index]);

            std::size_t const offsets_count = static_cast<std::size_t>(flat->offsets.size()); // rows + 1
            state_t::column_span_t &column = state->columns[index];
            column.payload = flat->payload;
            // 64-bit offsets everywhere: borrow cuDF's when already i64, else keep the column cuDF's own
            // `cast` produced — either way the offsets stay in a cuDF-owned buffer, no copy of our own.
            if (flat->offsets.type().id() == cudf::type_id::INT64)
                column.offsets = {flat->offsets.data<i64_t>(), offsets_count};
            else {
                column.widened_offsets = cudf::cast(flat->offsets, cudf::data_type {cudf::type_id::INT64}, cudf_stream);
                column.offsets = {column.widened_offsets->view().data<i64_t>(), offsets_count};
            }
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

} // namespace unum::uform::loaders
