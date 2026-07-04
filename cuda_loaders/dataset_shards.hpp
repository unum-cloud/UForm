/**
 *  @brief The dataset-shard reader cell: the loader tier's GPU-native readers. @ref parquet_file_t and
 *      @ref jsonl_file_t decode one shard — Parquet columns or a newline-delimited JSON field — straight
 *      into device memory over KvikIO / GPU Direct Storage, own the decoded columns for their lifetime,
 *      and hand out @b zero-copy StringZilla views. No host bounce, no Apache Arrow, no bespoke tape —
 *      cuDF is a required dependency, and its owned columns are the tape.
 *  @file cuda_loaders/dataset_shards.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
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
#include <memory>  // `std::unique_ptr` — the pimpl owning the cuDF table
#include <span>    // `std::span`

#include "cuda_core/types.cuh"    // `expected`, `status_t`, the scalar aliases, `sz::arrow_packed_view`, `sz::string`
#include "cuda_core/array.cuh"    // `array` — the glob expansion result
#include "cuda_core/schedule.cuh" // `gpu_stream_t`
#include "cuda_loaders/direct_storage.hpp" // `gpu_direct_storage_session_t` — the reader's GDS precondition

namespace unum::uform::loaders {

using namespace unum::uform::core;

/** @brief A decoded column, device-resident: every row's bytes back to back with @c rows+1
 *      terminator-free @c u64_t offsets (byte positions are unsigned sizes) — element @c i is the full
 *      @c [offsets[i], offsets[i+1]) span. Both text ( @c string ) and media ( @c list<uint8> )
 *      columns arrive as this. */
using column_view_t = sz::arrow_packed_view<b8_t, u64_t>;

/** @brief One shard's footer facts — the row count and a summed uncompressed byte size (the whole
 *      table, or one column) — read without decoding, to size the loader's file pool. */
struct parquet_length_metadata_t {
    std::size_t rows = 0;
    std::size_t bytes = 0;
};

/**
 *  @brief One Parquet shard's requested columns, decoded and owned on device. Move-only; the cuDF
 *      table (and any widened-offset columns) it owns free through RAII when the file is destroyed, so
 *      a file outlives the @ref make call that produced it for as long as its @ref column views are
 *      read. A default-constructed / moved-from file is empty ( @c columns()==0 ).
 */
struct parquet_file_t {

    parquet_file_t() noexcept;
    parquet_file_t(parquet_file_t const &) = delete;
    parquet_file_t &operator=(parquet_file_t const &) = delete;
    parquet_file_t(parquet_file_t &&) noexcept;
    parquet_file_t &operator=(parquet_file_t &&) noexcept;
    ~parquet_file_t();

    /** @brief Decode @p columns of the Parquet shard at @p path into device memory on @p stream — cuDF
     *      reads only those columns' chunks over KvikIO / GDS and decodes on the GPU. @p storage is the
     *      configured storage session, proof the GDS path is set up.
     *
     *  @param[in] columns Column names to decode, in the order @ref column indexes them. */
    [[nodiscard]] static expected<parquet_file_t> make(gpu_direct_storage_session_t const &storage,
                                                       sz::string_view path, std::span<sz::string_view const> columns,
                                                       gpu_stream_t const &stream) noexcept;

    /** @brief Decode @p columns of @b several shards at once into @b one concatenated file — cuDF reads
     *      all @p paths and returns a single table whose rows are the shards' rows back to back. The
     *      row draw's reservoir. */
    [[nodiscard]] static expected<parquet_file_t> make(gpu_direct_storage_session_t const &storage,
                                                       std::span<sz::string_view const> paths,
                                                       std::span<sz::string_view const> columns,
                                                       gpu_stream_t const &stream) noexcept;

    /** @brief Read only the shard's Parquet footer at @p path — @c num_rows and the @b whole table's
     *      summed column-chunk byte size — never touching the row groups or the GPU. */
    [[nodiscard]] static expected<parquet_length_metadata_t> table_metadata(gpu_direct_storage_session_t const &storage,
                                                                            sz::string_view path) noexcept;

    /** @brief Read only the shard's Parquet footer at @p path — @c num_rows and @p column 's summed
     *      column-chunk byte size — never touching the row groups or the GPU. */
    [[nodiscard]] static expected<parquet_length_metadata_t> column_metadata(
        gpu_direct_storage_session_t const &storage, sz::string_view path, sz::string_view column) noexcept;

    /** @brief Rows decoded — the same across every column. */
    [[nodiscard]] std::size_t rows() const noexcept;
    /** @brief How many columns were decoded. */
    [[nodiscard]] std::size_t columns() const noexcept;
    /** @brief Column @p index 's leaf bytes: element → bytes. A flat @c string / @c list<uint8> column
     *      has one element per row (element @c i is row @c i ); a nested @c list<string> column lays
     *      every row's strings back to back, so it has more elements than rows — @ref groups maps rows
     *      to them. */
    [[nodiscard]] column_view_t column(std::size_t index) const noexcept;
    /** @brief Column @p index 's row → element grouping: row @c r owns the leaf elements in
     *      @c [groups()[r], groups()[r+1]) . @b Empty for a flat column (row @c r @b is element @c r );
     *      @b [rows+1] for a nested @c list<string> column (multiple captions / negatives per row). All
     *      a consumer needs to see multi-positive / multi-negative structure and lay out segments. */
    [[nodiscard]] std::span<u64_t const> groups(std::size_t index) const noexcept;

  private:
    struct state_t; // The owned cuDF table plus its column views — defined in `dataset_shards.cpp`.
    explicit parquet_file_t(std::unique_ptr<state_t> state) noexcept;
    std::unique_ptr<state_t> state_;
};

/**
 *  @brief One newline-delimited JSON shard's chosen field, decoded and owned on device — the JSONL
 *      sibling of @ref parquet_file_t , a @b separate type (no format punning). cuDF's line-JSON reader
 *      decodes each line's @p field into the same device @ref column_view_t shape, so a consumer reads
 *      it exactly like a Parquet column; only the decode entry and the @b absent footer differ. Holds
 *      a single field, not a column set. Move-only, RAII; a default / moved-from file is empty.
 */
struct jsonl_file_t {

    jsonl_file_t() noexcept;
    jsonl_file_t(jsonl_file_t const &) = delete;
    jsonl_file_t &operator=(jsonl_file_t const &) = delete;
    jsonl_file_t(jsonl_file_t &&) noexcept;
    jsonl_file_t &operator=(jsonl_file_t &&) noexcept;
    ~jsonl_file_t();

    /** @brief Decode @p field of the JSONL shard at @p path — one JSON object per line — into device
     *      memory on @p stream over KvikIO / GDS. JSONL carries no footer, so a shard is always decoded
     *      to learn its rows; @ref file_size_bytes is the cheap size the loader draws files by instead. */
    [[nodiscard]] static expected<jsonl_file_t> make(gpu_direct_storage_session_t const &storage, sz::string_view path,
                                                     sz::string_view field, gpu_stream_t const &stream) noexcept;

    /** @brief Concatenate several decoded shards' fields into one reservoir — the footerless coverage
     *      loop decodes one file at a time, then merges them here (a cuDF concatenate, one copy). */
    [[nodiscard]] static expected<jsonl_file_t> concat(std::span<jsonl_file_t const> parts,
                                                       gpu_stream_t const &stream) noexcept;

    /** @brief Rows decoded — one per JSON line. */
    [[nodiscard]] std::size_t rows() const noexcept;
    /** @brief The chosen field's leaf bytes and offsets — the same shape @ref parquet_file_t::column
     *      hands out, so the pack kernels are identical. */
    [[nodiscard]] column_view_t column() const noexcept;
    /** @brief Row → element grouping when @p field is a JSON array of strings; empty for a scalar
     *      string field (row @c r @b is element @c r ). */
    [[nodiscard]] std::span<u64_t const> groups() const noexcept;

  private:
    struct state_t; // The owned cuDF column plus its flattened view — defined in `dataset_shards.cpp`.
    explicit jsonl_file_t(std::unique_ptr<state_t> state) noexcept;
    std::unique_ptr<state_t> state_;
};

/**
 *  @brief One CSV shard's chosen column, decoded and owned on device — the CSV sibling of
 *      @ref parquet_file_t and @ref jsonl_file_t , a @b separate type (no format punning). cuDF's CSV
 *      reader decodes @p column into the same device @ref column_view_t shape, so a consumer reads it
 *      exactly like a Parquet column; only the decode entry (delimiter, header) and the @b absent footer
 *      differ. Holds a single column. Move-only, RAII; a default / moved-from file is empty.
 */
struct csv_file_t {

    csv_file_t() noexcept;
    csv_file_t(csv_file_t const &) = delete;
    csv_file_t &operator=(csv_file_t const &) = delete;
    csv_file_t(csv_file_t &&) noexcept;
    csv_file_t &operator=(csv_file_t &&) noexcept;
    ~csv_file_t();

    /** @brief Decode @p column of the CSV shard at @p path into device memory on @p stream over KvikIO /
     *      GDS. @p delimiter is the field separator; @p has_header selects whether row 0 names the columns
     *      (else they are positional and @p column is a 0-based index string). CSV carries no footer, so a
     *      shard is always decoded to learn its rows; @ref file_size_bytes is the loader's cheap weight. */
    [[nodiscard]] static expected<csv_file_t> make(gpu_direct_storage_session_t const &storage, sz::string_view path,
                                                   sz::string_view column, char delimiter, bool has_header,
                                                   gpu_stream_t const &stream) noexcept;

    /** @brief Concatenate several decoded shards' columns into one reservoir — the footerless coverage
     *      loop decodes one file at a time, then merges them here (a cuDF concatenate, one copy). */
    [[nodiscard]] static expected<csv_file_t> concat(std::span<csv_file_t const> parts,
                                                     gpu_stream_t const &stream) noexcept;

    /** @brief Rows decoded — one per CSV record. */
    [[nodiscard]] std::size_t rows() const noexcept;
    /** @brief The chosen column's leaf bytes and offsets — the same shape @ref parquet_file_t::column
     *      hands out, so the pack kernels are identical. */
    [[nodiscard]] column_view_t column() const noexcept;
    /** @brief Empty — a CSV column is a flat string, so row @c r @b is element @c r . */
    [[nodiscard]] std::span<u64_t const> groups() const noexcept;

  private:
    struct state_t; // The owned cuDF column plus its flattened view — defined in `dataset_shards.cpp`.
    explicit csv_file_t(std::unique_ptr<state_t> state) noexcept;
    std::unique_ptr<state_t> state_;
};

/** @brief The byte size of @p path via POSIX @c stat — the cheap file-draw weight for the footerless
 *      formats (JSONL, CSV), which cannot know a row count without decoding. */
[[nodiscard]] expected<std::size_t> file_size_bytes(sz::string_view path) noexcept;

/** @brief Expand a @c directory/pattern (or a plain path) into the sorted paths it matches, via POSIX
 *      @c opendir / @c readdir + @c fnmatch — no glob library to link — sorted so the shard order is
 *      deterministic across every rank. An empty match is a failure, not an empty list. */
[[nodiscard]] expected<array<sz::string>> expand_glob(sz::string_view path_or_glob) noexcept;

} // namespace unum::uform::loaders
