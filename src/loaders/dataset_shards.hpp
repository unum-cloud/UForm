/**
 *  @brief The dataset-shard reader cell: the loader tier's GPU-native readers.
 *  @file src/loaders/dataset_shards.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  @ref parquet_file_t and @ref jsonl_file_t decode one shard — Parquet columns or a newline-delimited JSON
 *  field — straight into device memory over KvikIO / GPU Direct Storage, own the decoded columns for their
 *  lifetime, and hand out @b zero-copy @ref column_view_t borrows. No host bounce, no Apache Arrow, no bespoke
 *  tape — cuDF is a required dependency, and its owned columns are the tape.
 *
 *  A file's payload — the concatenated bytes of every row of a column — is borrowed in place from the
 *  owned cuDF table, never copied out. cuDF emits 32- or 64-bit @b signed offsets; the reader casts them
 *  once to unsigned @c u64_t and keeps that column alive alongside the table, so every @ref column view
 *  carries 64-bit unsigned offsets and nothing downstream meets a 32-bit or a signedness boundary.
 *  @c string and @c list<uint8> columns both flatten to the same @c (bytes, offsets) shape, so text
 *  documents and media blobs come back through one interface.
 *
 *  cuDF and RMM throw and allocate freely; that is confined to @c dataset_shards.cpp behind an exception
 *  firewall, and the cuDF types never cross this header — they live behind a @c std::unique_ptr pimpl,
 *  so ownership is RAII end to end with no hand-rolled @c new / @c delete .
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <algorithm> // `std::max`
#include <memory>    // `std::unique_ptr` — the pimpl owning the cuDF table
#include <span>      // `std::span`

#include "core/types.hpp"    // `expected`, `status_t`, the scalar aliases, `b8r1_view_t`, `sz::string_t`
#include "core/array.hpp"    // `array` — the glob expansion result and the row-group lists
#include "core/schedule.cuh" // `gpu_stream_t`
#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t` — the reader's GDS precondition

namespace unum::uform::loaders {

using namespace unum::uform::core;

/** A column of byte strings: every element's bytes back to back with @c elements+1 terminator-free
 *  @c u64_t offsets — element @c i is the full @c [offsets[i],offsets[i+1]) span. Both text ( @c string )
 *  and media ( @c list<uint8> ) columns arrive as this, the payload device-resident; the loaders re-view
 *  it with host offsets to slice it on the host. */
struct column_view_t {

    /** Every element's bytes, back to back. */
    b8r1_view_t payload;

    /** @b [elements+1] byte offsets into @ref payload . */
    u64r1_view_t offsets;

    /** The element count, one less than the offsets. */
    [[nodiscard]] std::size_t size() const noexcept { return offsets.empty() ? 0 : offsets.extent(0) - 1; }

    /** Element @p index 's bytes, sliced on the host, so @ref offsets must be host-resident. */
    [[nodiscard]] std::span<b8_t const> operator[](std::size_t index) const noexcept {
        u64_t const *const bounds = offsets.data();
        return {payload.data() + bounds[index], static_cast<std::size_t>(bounds[index + 1] - bounds[index])};
    }
};

/** One shard's footer facts — the row count and a summed uncompressed byte size (the whole
 *  table, or one column) — read without decoding, to size the loader's file pool. */
struct parquet_length_metadata_t {
    std::size_t rows = 0;
    std::size_t bytes = 0;

    /** @b [groups] each row group's rows and summed uncompressed bytes, for partial reads. */
    array<std::size_t> group_rows {}, group_bytes {};

    /** @b [groups] each row group's rows that no null leaf of the named columns leaves incomplete,
     *  at least; every row when no column is named. */
    array<std::size_t> group_complete_rows {};
};

/** The device bytes a fill's reservoir can take outside the arena, @c fixed_bytes+rows·row_bytes
 *  for a fill of @c rows rows. A fill reads at least two shards or row groups, and more only while
 *  its rows fall short, so it reads at most one past what its rows need. Every byte counts thrice,
 *  as cuDF holds the compressed chunks, the decompressed pages and the decoded columns at once. */
struct reservoir_bound_t {
    std::size_t fixed_bytes = 0, row_bytes = 0;
    static constexpr std::size_t copies_k = 3;

    /** Widen the bound to cover a whole-shard read of @p shard 's footer, or of @c shard.bytes
     *  alone when it tells no rows. */
    void charge(parquet_length_metadata_t const &shard) noexcept {
        fixed_bytes = std::max(fixed_bytes, 2 * copies_k * shard.bytes);
        if (shard.rows != 0) row_bytes = std::max(row_bytes, copies_k * divide_round_up(shard.bytes, shard.rows));
    }

    /** Widen the bound to cover a read of only some row groups of @p shard 's footer. */
    void charge_groups(parquet_length_metadata_t const &shard) noexcept {
        for (std::size_t group = 0; group != shard.group_rows.size(); ++group) {
            fixed_bytes = std::max(fixed_bytes, 2 * copies_k * shard.group_bytes[group]);
            if (shard.group_rows[group] != 0)
                row_bytes = std::max(row_bytes,
                                     copies_k * divide_round_up(shard.group_bytes[group], shard.group_rows[group]));
        }
    }

    /** Widen the bound to cover @p other as well. */
    void charge(reservoir_bound_t other) noexcept {
        fixed_bytes = std::max(fixed_bytes, other.fixed_bytes);
        row_bytes = std::max(row_bytes, other.row_bytes);
    }
};

/** One Parquet shard's requested columns, decoded and owned on device. Move-only; the cuDF
 *  table (and any widened-offset columns) it owns free through RAII when the file is destroyed, so
 *  a file outlives the @ref make call that produced it for as long as its @ref column views are
 *  read. A default-constructed / moved-from file is empty ( @c columns()==0 ). */
struct parquet_file_t {

    parquet_file_t() noexcept;
    parquet_file_t(parquet_file_t const &) = delete;
    parquet_file_t &operator=(parquet_file_t const &) = delete;
    parquet_file_t(parquet_file_t &&) noexcept;
    parquet_file_t &operator=(parquet_file_t &&) noexcept;
    ~parquet_file_t();

    /**
     *  @brief Decode @p columns of the Parquet shard at @p path into device memory on @p stream — cuDF
     *      reads only those columns' chunks over KvikIO / GDS and decodes on the GPU. @p storage is the
     *      configured storage session, proof the GDS path is set up.
     *
     *  @param[in] columns Column names to decode, in the order @ref column indexes them.
     */
    [[nodiscard]] static expected<parquet_file_t> make(gpu_direct_storage_session_t const &storage,
                                                       sz::string_view_t path,
                                                       std::span<sz::string_view_t const> columns,
                                                       gpu_stream_t &stream) noexcept;

    /** Decode @p columns of @b several shards at once into @b one concatenated file — cuDF reads
     *  all @p paths and returns a single table whose rows are the shards' rows back to back. The
     *  row draw's reservoir. */
    [[nodiscard]] static expected<parquet_file_t> make(gpu_direct_storage_session_t const &storage,
                                                       std::span<sz::string_view_t const> paths,
                                                       std::span<sz::string_view_t const> columns,
                                                       gpu_stream_t &stream) noexcept;

    /** Decode only @p row_groups of @p paths , one ascending list per path, into one concatenated
     *  file whose rows follow the paths, then the groups, back to back. */
    [[nodiscard]] static expected<parquet_file_t> make(gpu_direct_storage_session_t const &storage,
                                                       std::span<sz::string_view_t const> paths,
                                                       std::span<sz::string_view_t const> columns,
                                                       std::span<array<u32_t> const> row_groups,
                                                       gpu_stream_t &stream) noexcept;

    /** Read only the shard's Parquet footer at @p path — @c num_rows , the @b whole table's summed
     *  column-chunk byte size, and each row group's rows, bytes and rows @p complete_columns all
     *  fill — never touching the row groups' pages or the GPU. */
    [[nodiscard]] static expected<parquet_length_metadata_t> table_metadata(
        gpu_direct_storage_session_t const &storage, sz::string_view_t path,
        std::span<sz::string_view_t const> complete_columns = {}) noexcept;

    /** Read only the shard's Parquet footer at @p path — @c num_rows and @p column 's summed
     *  column-chunk byte size — never touching the row groups or the GPU. */
    [[nodiscard]] static expected<parquet_length_metadata_t> column_metadata(
        gpu_direct_storage_session_t const &storage, sz::string_view_t path, sz::string_view_t column) noexcept;

    /** Rows decoded — the same across every column. */
    [[nodiscard]] std::size_t rows() const noexcept;

    /** How many columns were decoded. */
    [[nodiscard]] std::size_t columns() const noexcept;

    /** Column @p index 's leaf bytes: element → bytes. A flat @c string / @c list<uint8> column
     *  has one element per row (element @c i is row @c i ); a nested @c list<string> column lays
     *  every row's strings back to back, so it has more elements than rows — @ref groups maps rows
     *  to them. */
    [[nodiscard]] column_view_t column(std::size_t index) const noexcept;

    /** Column @p index 's row → element grouping: row @c r owns the leaf elements in
     *  @c [groups()[r], groups()[r+1]) . @b Empty for a flat column (row @c r @b is element @c r );
     *  @b [rows+1] for a nested @c list<string> column (multiple captions / negatives per row). All
     *  a consumer needs to see multi-positive / multi-negative structure and lay out segments. */
    [[nodiscard]] std::span<u64_t const> groups(std::size_t index) const noexcept;

  private:
    struct state_t; // The owned cuDF table plus its column views — defined in `dataset_shards.cpp`.
    explicit parquet_file_t(std::unique_ptr<state_t> state) noexcept;
    std::unique_ptr<state_t> state_;
};

/** One newline-delimited JSON shard's chosen field, decoded and owned on device — the JSONL
 *  sibling of @ref parquet_file_t , a @b separate type (no format punning). cuDF's line-JSON reader
 *  decodes each line's @p field into the same device @ref column_view_t shape, so a consumer reads
 *  it exactly like a Parquet column; only the decode entry and the @b absent footer differ. Holds
 *  a single field, not a column set. Move-only, RAII; a default / moved-from file is empty. */
struct jsonl_file_t {

    jsonl_file_t() noexcept;
    jsonl_file_t(jsonl_file_t const &) = delete;
    jsonl_file_t &operator=(jsonl_file_t const &) = delete;
    jsonl_file_t(jsonl_file_t &&) noexcept;
    jsonl_file_t &operator=(jsonl_file_t &&) noexcept;
    ~jsonl_file_t();

    /** Decode @p field of the JSONL shard at @p path — one JSON object per line — into device
     *  memory on @p stream over KvikIO / GDS. JSONL carries no footer, so a shard is always decoded
     *  to learn its rows; @ref file_size_bytes is the cheap size the loader draws files by instead. */
    [[nodiscard]] static expected<jsonl_file_t> make(gpu_direct_storage_session_t const &storage,
                                                     sz::string_view_t path, sz::string_view_t field,
                                                     gpu_stream_t &stream) noexcept;

    /** Concatenate several decoded shards' fields into one reservoir — the footerless coverage
     *  loop decodes one file at a time, then merges them here (a cuDF concatenate, one copy). */
    [[nodiscard]] static expected<jsonl_file_t> concat(std::span<jsonl_file_t const> parts,
                                                       gpu_stream_t &stream) noexcept;

    /** Rows decoded — one per JSON line. */
    [[nodiscard]] std::size_t rows() const noexcept;

    /** The chosen field's leaf bytes and offsets — the same shape @ref parquet_file_t::column
     *  hands out, so the pack kernels are identical. */
    [[nodiscard]] column_view_t column() const noexcept;

    /** Row → element grouping when @p field is a JSON array of strings; empty for a scalar
     *  string field (row @c r @b is element @c r ). */
    [[nodiscard]] std::span<u64_t const> groups() const noexcept;

  private:
    struct state_t; // The owned cuDF column plus its flattened view — defined in `dataset_shards.cpp`.
    explicit jsonl_file_t(std::unique_ptr<state_t> state) noexcept;
    std::unique_ptr<state_t> state_;
};

/** Whether a CSV shard's row 0 names its columns, or every row is data and columns are positional. */
enum class csv_header_t : bool { positional_k, named_k };

/** One CSV shard's chosen column, decoded and owned on device — the CSV sibling of
 *  @ref parquet_file_t and @ref jsonl_file_t , a @b separate type (no format punning). cuDF's CSV
 *  reader decodes @p column into the same device @ref column_view_t shape, so a consumer reads it
 *  exactly like a Parquet column; only the decode entry (delimiter, header) and the @b absent footer
 *  differ. Holds a single column. Move-only, RAII; a default / moved-from file is empty. */
struct csv_file_t {

    csv_file_t() noexcept;
    csv_file_t(csv_file_t const &) = delete;
    csv_file_t &operator=(csv_file_t const &) = delete;
    csv_file_t(csv_file_t &&) noexcept;
    csv_file_t &operator=(csv_file_t &&) noexcept;
    ~csv_file_t();

    /** Decode @p column of the CSV shard at @p path into device memory on @p stream over KvikIO / GDS.
     *  @p delimiter is the field separator; @p header says whether row 0 names the columns, and when
     *  positional, @p column is a 0-based index string. CSV carries no footer, so a shard is always decoded
     *  to learn its rows; @ref file_size_bytes is the loader's cheap weight. */
    [[nodiscard]] static expected<csv_file_t> make(gpu_direct_storage_session_t const &storage, sz::string_view_t path,
                                                   sz::string_view_t column, char delimiter, csv_header_t header,
                                                   gpu_stream_t &stream) noexcept;

    /** Concatenate several decoded shards' columns into one reservoir — the footerless coverage
     *  loop decodes one file at a time, then merges them here (a cuDF concatenate, one copy). */
    [[nodiscard]] static expected<csv_file_t> concat(std::span<csv_file_t const> parts, gpu_stream_t &stream) noexcept;

    /** Rows decoded — one per CSV record. */
    [[nodiscard]] std::size_t rows() const noexcept;

    /** The chosen column's leaf bytes and offsets — the same shape @ref parquet_file_t::column
     *  hands out, so the pack kernels are identical. */
    [[nodiscard]] column_view_t column() const noexcept;

    /** Empty — a CSV column is a flat string, so row @c r @b is element @c r . */
    [[nodiscard]] std::span<u64_t const> groups() const noexcept;

  private:
    struct state_t; // The owned cuDF column plus its flattened view — defined in `dataset_shards.cpp`.
    explicit csv_file_t(std::unique_ptr<state_t> state) noexcept;
    std::unique_ptr<state_t> state_;
};

/** The byte size of @p path via POSIX @c stat — the cheap file-draw weight for the footerless
 *  formats (JSONL, CSV), which cannot know a row count without decoding. */
[[nodiscard]] expected<std::size_t> file_size_bytes(sz::string_view_t path) noexcept;

/** Expand a @c directory/pattern (or a plain path) into the sorted paths it matches, via POSIX
 *  @c opendir / @c readdir + @c fnmatch — no glob library to link — sorted so the shard order is
 *  deterministic across every rank. An empty match is a failure, not an empty list. */
[[nodiscard]] expected<array<sz::string_t>> expand_glob(sz::string_view_t path_or_glob) noexcept;

} // namespace unum::uform::loaders
