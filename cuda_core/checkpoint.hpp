/**
 *  @brief Safetensors checkpoints: one file, a yyjson-built JSON header of names / dtypes / shapes /
 *      offsets, then the raw little-endian tensor bytes — readable by the whole Python ecosystem,
 *      written and read here through NumKong views with no exceptions and no vendor containers.
 *  @file cuda_core/checkpoint.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  @section checkpoint_containers Two containers, two directions
 *
 *  The checkpoint types are the containers — no caller-side table of views to materialize.
 *  @ref checkpoint_to_save_t snapshots each inserted tensor into its pinned payload immediately, so
 *  training may keep mutating the source; @ref checkpoint_to_save_t::save is then pure host I/O.
 *  @ref checkpoint_to_load_t reads a whole file once and hands out by-name views into its pinned
 *  payload — the caller uploads what it wants and decides what a missing name means, which is the
 *  whole warm-start policy: a masked-pretraining checkpoint fills the shared encoder, the objective
 *  heads come back empty, and the caller re-seeds them.
 *
 *  The caller (main — the composition root) spells the naming contract @c model.<section>.<part> at
 *  both ends. Shapes are the true @b [rows,cols] extents of the rank-2 views; BFloat16 is the only
 *  training dtype; weights only — optimizer moments restart fresh on resume.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include "cuda_core/types.hpp"          // `expected`, `expected_t`, `status_t`, `sz::string`
#include "cuda_core/array.hpp"          // `array` — the entry table and the payload
#include "cuda_core/gpu_allocators.hpp" // `pinned_alloc` — page-locked staging the DMA engines read directly
#include "cuda_core/schedule.cuh"       // `gpu_stream_t` — orders the snapshot after the training kernels

namespace unum::uform::core {

/** @brief One indexed checkpoint entry: the owned name, the logical @b [rows,cols] extents (vectors
 *      are @b [1,cols] ), and where the tensor's scalars start inside the owner's payload. */
struct checkpoint_entry_t {
    /** @brief The owned name text — short names ride StringZilla's inline buffer. */
    sz::string name {};
    /** @brief Leading extent; @c 1 for vector parameters. */
    u64_t rows = 0;
    /** @brief Trailing extent. */
    u64_t cols = 0;
    /** @brief Scalar (not byte) offset inside the owner's payload. */
    u64_t scalar_offset = 0;
};

/**
 *  @brief The outbound checkpoint container: @ref try_insert snapshots one device tensor into the
 *      pinned payload — stream-ordered after the kernels already enqueued, drained by a bounded
 *      synchronize — so the weights are free to mutate the moment it returns, and @ref save is pure
 *      host I/O. All buffers grow lazily and keep their backing across cadences; @ref try_reserve
 *      hoists the growth entirely. Lead-rank-only by the caller's discipline; must not outlive the
 *      streams' contexts — releasing pinned pages needs the driver alive.
 */
struct checkpoint_to_save_t {

    checkpoint_to_save_t() noexcept = default;
    checkpoint_to_save_t(checkpoint_to_save_t const &) = delete;
    checkpoint_to_save_t &operator=(checkpoint_to_save_t const &) = delete;
    checkpoint_to_save_t(checkpoint_to_save_t &&) noexcept = default;
    checkpoint_to_save_t &operator=(checkpoint_to_save_t &&) noexcept = default;
    ~checkpoint_to_save_t() noexcept = default;

    /** @brief Adopts @p allocator — the byte-typed pinned handle the DMA-staged payload rebinds —
     *      and allocates nothing until the first @ref try_reserve or @ref try_insert . */
    [[nodiscard]] static expected<checkpoint_to_save_t> make(pinned_alloc_t allocator) noexcept;

    /** @brief Optional hoist: pre-size the payload to @p total_scalars and the entry table to
     *      @p tensor_count , so no @ref try_insert ever reallocates. Needs a current context at the
     *      call, like every driver allocation here. */
    [[nodiscard]] expected_t try_reserve(u64_t total_scalars, u64_t tensor_count) noexcept;

    /** @brief Append one entry, snapshotting @p data into the pinned payload @b now : the copy is
     *      enqueued on @p stream — after every kernel already there — and drained by a bounded
     *      synchronize, so the caller may keep updating the weights the moment this returns.
     *      @p name is copied into owned text; duplicates are rejected loudly. */
    [[nodiscard]] expected_t try_insert(sz::string_view name, nk::tensor_view<bf16_t, 2> data,
                                        gpu_stream_t &stream) noexcept;

    /** @brief Write the accumulated entries to @p path in the safetensors layout, with @c {"step": …}
     *      metadata — the JSON header space-padded to an 8-byte boundary like Hugging Face's own
     *      writer, then the whole contiguous payload in one shot. Pure host work: no stream, no GPU
     *      calls. */
    [[nodiscard]] expected_t save(sz::string_view path, u64_t step) noexcept;

    /** @brief Drop the entries, keep every buffer's backing — the next cadence re-inserts and
     *      re-saves allocation-free. */
    void clear() noexcept;

    /** @brief How many tensors the container currently holds. */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

  private:
    /** @brief Every inserted tensor's scalars back-to-back in insertion order — page-locked pages the
     *      DMA engines write directly, and `save` streams to disk in a single write. */
    array<bf16_t, pinned_alloc<bf16_t>> payload_ {};
    /** @brief The name / shape / offset record per inserted tensor — pageable on purpose, no DMA
     *      engine ever touches it. */
    array<checkpoint_entry_t> entries_ {};
    /** @brief The yyjson pool `save` serializes the header in — grown lazily, settled after the
     *      first save. */
    array<char> scratch_ {};
};

/**
 *  @brief The inbound checkpoint container: @ref make reads a whole safetensors file once — header
 *      parsed and validated, payload staged into pinned pages — and @ref operator[] hands out
 *      read-only views by name. The caller uploads what it wants with @ref copy_host_to_device (true
 *      async DMA off the pinned backing); strictness is the caller's policy, not this container's.
 *      Must not outlive the streams' contexts — releasing pinned pages needs the driver alive.
 */
struct checkpoint_to_load_t {

    checkpoint_to_load_t() noexcept = default;
    checkpoint_to_load_t(checkpoint_to_load_t const &) = delete;
    checkpoint_to_load_t &operator=(checkpoint_to_load_t const &) = delete;
    checkpoint_to_load_t(checkpoint_to_load_t &&) noexcept = default;
    checkpoint_to_load_t &operator=(checkpoint_to_load_t &&) noexcept = default;
    ~checkpoint_to_load_t() noexcept = default;

    /** @brief Read @p path in full: parse the header — its declared length bounded by the file's own
     *      extent, every entry's @c data_offsets validated against its shape and the payload — and
     *      stage the payload into pages from @p allocator , the byte-typed pinned handle it rebinds.
     *      Needs a current context at the call. Synchronous: the file is closed and everything
     *      resident when this returns. */
    [[nodiscard]] static expected<checkpoint_to_load_t> make(sz::string_view path, pinned_alloc_t allocator) noexcept;

    /** @brief By-name lookup: a read-only rank-2 view into the pinned payload, or an @b empty view
     *      when the file misses @p layer_name (or holds it in a dtype this cell doesn't train in) —
     *      check @c .empty() and decide. */
    [[nodiscard]] nk::tensor_view<bf16_t, 2> operator[](sz::string_view layer_name) const noexcept;

    /** @brief The @c {"__metadata__":{"step": …}} value the file was saved at; @c 0 when absent. */
    [[nodiscard]] u64_t step() const noexcept { return step_; }

    /** @brief How many tensors the container indexed. */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

  private:
    /** @brief The file's whole tensor payload — page-locked, so the caller's uploads are true async
     *      DMA. */
    array<bf16_t, pinned_alloc<bf16_t>> payload_ {};
    /** @brief The name / shape / offset record per indexed BF16 tensor — pageable on purpose, no DMA
     *      engine ever touches it. */
    array<checkpoint_entry_t> entries_ {};
    /** @brief The hoisted parse scratch: the file's header bytes up front, the yyjson pool behind
     *      them — one allocation for the whole parse. */
    array<char> scratch_ {};
    /** @brief The parsed step metadata; @c 0 when the file carries none. */
    u64_t step_ = 0;
};

} // namespace unum::uform::core
