/**
 *  @file include/uform/utilities/ram_device.hpp
 *  @author Ash Vardanian
 *  @date October 27, 2026
 *  @brief Host memory domains, pool dispatch and statuses, over ForkUnion's C ABI.
 *
 *  Engines, loaders, tools and tests dispatch on a ForkUnion pool the library spawns, or on an
 *  empty one, which runs every dispatch on the calling thread.
 */
#pragma once

#include <cassert> // `assert` - an arena dies with no chunk left
#include <cstddef> // `std::size_t`

#include <limits>      // `std::numeric_limits` - an allocation's count overflowing
#include <new>         // `std::align_val_t`, `std::nothrow` - host chunks
#include <span>        // `std::span` - the chunks an owner frees
#include <type_traits> // `std::is_trivially_copyable_v`, `std::remove_reference_t`

#include <forkunion/types.hpp> // `fu::status_t`, `fu::dispatched_status`

#include "uform/utilities/types.hpp" // `status_t`, `expected`, `bytes_span_t`

namespace unum::uform::utilities {

#pragma region Executor

/** Translates a ForkUnion failure into the code every UForm call reports. */
constexpr status_code_t status_from_(fu_status_t status) noexcept {
    switch (status) {
    case fu_success_k: return status_code_t::success_k;
    case fu_bad_alloc_k: return status_code_t::out_of_memory_k;
    case fu_invalid_argument_k: return status_code_t::invalid_argument_k;
    case fu_unsupported_k: return status_code_t::unsupported_k;
    case fu_contended_k: return status_code_t::contended_k;
    default: return status_code_t::internal_k;
    }
}

/** Translates a failure of a pool dispatch, which spells the same codes. */
constexpr status_code_t status_from_(fu::status_t status) noexcept {
    return status_from_(static_cast<fu_status_t>(status));
}

/** Runs @p task on each of @p n similar-cost tasks across @p pool , blocking until all finish; an
 *  empty pool runs them on the calling thread. */
template <typename task_type_>
fu::status_t for_n(fu_pool_t pool, std::size_t n, task_type_&& task) noexcept {
    using task_t = std::remove_reference_t<task_type_>;
    if (!pool) {
        for (std::size_t index = 0; index != n; ++index) task(index, std::size_t {0});
        return fu::status_t::success_k;
    }
    auto const run = [](fu_lambda_context_t context, std::size_t index, std::size_t thread, std::size_t) noexcept {
        (*static_cast<task_t*>(context))(index, thread);
    };
    return static_cast<fu::status_t>(fu_pool_for_n(pool, n, run, static_cast<void*>(&task)));
}

/** The memory domain naming none, where an allocation takes the plain heap. */
inline constexpr fu_memory_domain_id_t no_memory_domain_k = -1;

/** The label the threads of every UForm pool have in the OS. */
inline constexpr char const pool_name_k[] = "uform";

/**
 *  @brief Spawns @p threads threads of a @p threads_type_ pool across the compute domains of
 *      @p topology , or only those local to @p memory_domain unless it is @ref no_memory_domain_k .
 *  @param[in] exclusivity Whether the caller is one of the @p threads or only coordinates them.
 *
 *  Uses only the ForkUnion facilities @p allowed names; a build without OS threads, such as
 *  WebAssembly's, runs one thread on the caller and refuses more as unsupported.
 */
template <typename threads_type_, typename topology_type_>
[[nodiscard]] expected<threads_type_> spawn_pool(topology_type_ const& topology, std::size_t threads,
                                                 fu::caller_exclusivity_t exclusivity,
                                                 fu_capabilities_t allowed = fu_capabilities_all_k,
                                                 fu_memory_domain_id_t memory_domain = no_memory_domain_k) noexcept {
    threads_type_ pool;
    fu::status_t const status = memory_domain < 0 ? pool.spawn(topology, threads, exclusivity, allowed, pool_name_k)
                                                  : pool.spawn_near(topology, memory_domain, threads, exclusivity,
                                                                    allowed, pool_name_k);
    if (fu::failed(status))
        return expected<threads_type_>::fail(status_from_(status), "spawning {} threads failed", threads);
    return pool;
}

/** The memory domain nearest the PCI function at @p bus_id, or @c no_memory_domain_k where the
 *  topology names none. */
inline fu_memory_domain_id_t memory_domain_of_pci(char const* bus_id) noexcept { // vendor-boundary: a C string
    fu_memory_domain_id_t domain = no_memory_domain_k;
    return fu_memory_domain_id_of_pci(bus_id, &domain) == fu_success_k ? domain : no_memory_domain_k;
}

/** Allocates @p bytes on the memory domain @p domain , answering an empty span on failure. */
[[nodiscard]] inline bytes_span_t allocate_on_memory_domain(fu_memory_domain_id_t domain, std::size_t bytes) noexcept {
    void* memory = nullptr;
    if (fu_allocate_on_domain_id(domain, bytes, &memory) != fu_success_k) return {};
    return {static_cast<b8_t*>(memory), bytes};
}

/** Frees @p memory , which @ref allocate_on_memory_domain placed on @p domain . */
inline void free_on_memory_domain(fu_memory_domain_id_t domain, bytes_span_t memory) noexcept {
    fu_free_on_domain_id(domain, memory.data(), memory.size());
}

/** How page-locked host pages were made, which names the verb that frees them. */
enum class pinned_origin_t : u32_t {

    /** Allocated and page-locked by the GPU runtime. */
    runtime_k,

    /** Placed on a memory domain, then registered with the GPU runtime. */
    memory_domain_k,
};

#pragma endregion Executor

#pragma region Arenas

/**
 *  @brief Memory one call at a time rewinds and allocates from: chunks its owner made and
 *      appended, which never move, so every view a call hands out stays put until the next rewind.
 *
 *  The arena allocates and frees nothing. Its owner reserves before a call allocates, appending a
 *  chunk at least as wide as the call's whole footprint when @ref room falls short, and frees each
 *  chunk through the verb that made it before the arena dies: the stream's for device and pinned
 *  memory, @ref free_on_memory_domain for host pages. Movable so a record can hold it, never
 *  move-assigned, since that would drop the destination's chunks.
 */
class chunked_arena_t {
  public:
    /** Every allocation starts on this boundary, which every kernel's operands accept. */
    inline static constexpr std::size_t alignment_k = 64;

    chunked_arena_t() noexcept = default;
    chunked_arena_t(chunked_arena_t&&) noexcept = default;
    chunked_arena_t& operator=(chunked_arena_t&&) = delete;
    ~chunked_arena_t() noexcept { assert(chunks_.empty() && "The owner frees the chunks it appended"); }

    /** The bytes an allocation of @p bytes spans, its padding included: what a reservation sums. */
    [[nodiscard]] static constexpr std::size_t footprint(std::size_t bytes) noexcept {
        return round_up_to_multiple(bytes, alignment_k);
    }

    /** The bytes an allocation of @p count values of @p value_type_ spans. */
    template <typename value_type_>
    [[nodiscard]] static constexpr std::size_t footprint(std::size_t count) noexcept {
        return footprint(count * sizeof(value_type_));
    }

    /** The widest run allocations may take whole from here on: the rest of the chunk being filled,
     *  or a later chunk, all untouched. Allocations summing to no more than this all land. */
    [[nodiscard]] std::size_t room() const noexcept {
        std::size_t widest = current_ != chunks_.size() ? chunks_[current_].size() - used_ : 0;
        for (std::size_t index = current_ + 1; index < chunks_.size(); ++index)
            widest = chunks_[index].size() > widest ? chunks_[index].size() : widest;
        return widest;
    }

    /** Bytes every chunk spans together, which the owner doubles when it appends. */
    [[nodiscard]] std::size_t held() const noexcept {
        std::size_t held = 0;
        for (bytes_span_t const chunk : chunks_) held += chunk.size();
        return held;
    }

    /**
     *  @brief Takes @p chunk , which the owner allocated and frees, as room for later allocations.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The chunk table could not grow; the owner still holds @p chunk .
     */
    status_t append(bytes_span_t chunk) noexcept {
        if (st::failed(chunks_.push_back(bytes_span_t {chunk})))
            return {status_code_t::out_of_memory_k, "The chunk table is full"};
        return {};
    }

    /** The chunks the owner appended, which it frees through the verb that made each. */
    [[nodiscard]] std::span<bytes_span_t const> chunks() const noexcept { return chunks_; }

    /** Drops every chunk once the owner freed them. */
    void forget() noexcept { chunks_.clear(), current_ = 0, used_ = 0; }

    /** Forgets every allocation, keeping the chunks, which invalidates every view the previous call
     *  handed out. */
    void rewind() noexcept { current_ = 0, used_ = 0; }

    /** Whether @p address lies in memory this arena hands out, which its next rewind reuses. */
    [[nodiscard]] bool contains(void const* address) const noexcept {
        for (bytes_span_t const chunk : chunks_)
            if (address >= chunk.data() && address < chunk.data() + chunk.size()) return true;
        return false;
    }

    /** Hands out @p bytes on an @ref alignment_k boundary from the reserved chunks, or null when
     *  they are full, which the caller reports as out of memory. */
    [[nodiscard]] void* allocate(std::size_t bytes) noexcept {
        std::size_t const aligned = footprint(bytes);
        if (aligned < bytes) return nullptr; // saturated within an alignment of the address space
        while (current_ != chunks_.size() && chunks_[current_].size() - used_ < aligned) ++current_, used_ = 0;
        if (current_ == chunks_.size()) return nullptr;
        b8_t* const landed = chunks_[current_].data() + used_;
        used_ += aligned;
        return landed;
    }

    /** Hands out @p count values of @p value_type_, or null when they do not fit. */
    template <typename value_type_>
    [[nodiscard]] value_type_* allocate(std::size_t count) noexcept {
        static_assert(std::is_trivially_copyable_v<value_type_>, "Arenas hold plain outputs only");
        if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(value_type_)) return nullptr;
        return static_cast<value_type_*>(allocate(count * sizeof(value_type_)));
    }

  private:
    st::basic_vector<bytes_span_t> chunks_;

    /** The chunk being filled, and the bytes of it already handed out since the last rewind. */
    std::size_t current_ = 0;
    std::size_t used_ = 0;
};

/**
 *  @brief Makes room for allocations of @p bytes in host @p arena , appending a heap chunk at
 *      least that wide and as wide as everything held.
 *
 *  @par Statuses:
 *  - @c out_of_memory_k The heap refused the chunk, or the chunk table could not grow.
 */
inline status_t reserve_host_pages(chunked_arena_t& arena, std::size_t bytes) noexcept {
    if (arena.room() >= bytes) return {};
    std::size_t const held = arena.held(), width = held > bytes ? held : bytes;
    auto* const data = static_cast<b8_t*>(
        ::operator new(width, std::align_val_t {chunked_arena_t::alignment_k}, std::nothrow));
    if (!data) return {status_code_t::out_of_memory_k, "The host refused an arena chunk"};
    if (status_t const appended = arena.append({data, width}); appended.failed())
        return ::operator delete(data, std::align_val_t {chunked_arena_t::alignment_k}), appended;
    return {};
}

/** Frees every chunk of host @p arena , which @ref reserve_host_pages took from the heap. */
inline void release_host_pages(chunked_arena_t& arena) noexcept {
    for (bytes_span_t const chunk : arena.chunks())
        ::operator delete(chunk.data(), std::align_val_t {chunked_arena_t::alignment_k});
    arena.forget();
}

#pragma endregion Arenas

} // namespace unum::uform::utilities
