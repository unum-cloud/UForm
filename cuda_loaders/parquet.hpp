/**
 *  @brief The Parquet cell: the loader tier's single GPU-native reader. @ref parquet_file_t::make
 *      decodes the requested columns of one Parquet shard straight into device memory over KvikIO /
 *      GPU Direct Storage, then owns that decoded table for its lifetime and hands out @b zero-copy
 *      StringZilla views into it. No host bounce, no Apache Arrow, no bespoke tape — cuDF is a
 *      required dependency, and its owned columns are the tape.
 *  @file cuda_loaders/parquet.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  A file's payload — the concatenated bytes of every row of a column — is borrowed in place from the
 *  owned cuDF table, never copied out. cuDF emits 32-bit offsets below 2 GiB; where it does, the
 *  reader keeps cuDF's own widened @c i64_t offsets column alive alongside the table (again no copy),
 *  so every @ref column view carries 64-bit offsets and nothing downstream meets a 32-bit boundary.
 *  @c string and @c list<uint8> columns both flatten to the same @c (bytes, offsets) shape, so text
 *  documents and media blobs come back through one interface.
 *
 *  cuDF and RMM throw and allocate freely; that is confined to @c parquet.cpp behind an exception
 *  firewall, and the cuDF types never cross this header — they live behind a @c std::unique_ptr pimpl,
 *  so ownership is RAII end to end with no hand-rolled @c new / @c delete .
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <memory>  // `std::unique_ptr` — the pimpl owning the cuDF table
#include <span>    // `std::span`

#include "types.cuh"    // `expected`, `status_t`, the scalar aliases, `sz::arrow_packed_view`
#include "schedule.cuh" // `gpu_stream_t`

#include "direct_storage.hpp" // `gpu_direct_storage_session_t` — the reader's GDS precondition

namespace unum::uform::loaders {

using namespace unum::uform::core;

/** @brief A decoded column, device-resident: every row's bytes back to back with @c rows+1
 *      terminator-free @c i64_t offsets — the raw Apache Arrow convention, so element @c i is the full
 *      @c [offsets[i], offsets[i+1]) span. Both text ( @c string ) and media ( @c list<uint8> )
 *      columns arrive as this. */
using column_view_t = sz::arrow_packed_view<b8_t, i64_t>;

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

    /** @brief Rows decoded — the same across every column. */
    [[nodiscard]] std::size_t rows() const noexcept;
    /** @brief How many columns were decoded. */
    [[nodiscard]] std::size_t columns() const noexcept;
    /** @brief Column @p index as a zero-copy device view into the owned table. */
    [[nodiscard]] column_view_t column(std::size_t index) const noexcept;

  private:
    struct state_t; // The owned cuDF table plus its column views — defined in `parquet.cpp`.
    explicit parquet_file_t(std::unique_ptr<state_t> state) noexcept;
    std::unique_ptr<state_t> state_;
};

} // namespace unum::uform::loaders
