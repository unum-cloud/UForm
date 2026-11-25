/**
 *  @file include/uform/utilities/ram_device.hpp
 *  @author Ash Vardanian
 *  @date October 27, 2026
 *  @brief Host topology and thread spawning, over ForkUnion's C ABI and its @c fu::pool_t .
 *
 *  Engines, loaders, tools and tests dispatch on a @c fu::pool_t that @ref spawn_pool fills, or on
 *  an empty one, which runs every dispatch on the calling thread.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <forkunion/forkunion.hpp> // `fu::pool_t`, `fu::topology_t`, `fu::default_threads_count`

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

/** Translates a failure of a @c fu::pool_t dispatch, which spells the same codes. */
constexpr status_code_t status_from_(fu::status_t status) noexcept {
    return status_from_(static_cast<fu_status_t>(status));
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

/** The label the threads of every UForm pool have in the OS. */
inline constexpr char const pool_name_k[] = "uform";

/**
 *  @brief Spawns @p threads threads into @p pool across the compute domains of @p topology , or
 *      only those local to @p memory_domain unless it is -1, using only the ForkUnion facilities
 *      @p allowed names; a build without OS threads, such as WebAssembly's, runs one thread on the
 *      caller and refuses more as unsupported.
 *  @param[in] exclusivity Whether the caller is one of the @p threads or only coordinates them.
 */
[[nodiscard]] inline expected_t spawn_pool(fu::pool_t& pool, fu::topology_t const& topology, std::size_t threads,
                                           fu::caller_exclusivity_t exclusivity, fu_capabilities_t allowed,
                                           fu_memory_domain_id_t memory_domain = -1) noexcept {
    fu::status_t const status = memory_domain < 0 ? pool.spawn(topology, threads, exclusivity, allowed, pool_name_k)
                                                  : pool.spawn_near(topology, memory_domain, threads, exclusivity,
                                                                    allowed, pool_name_k);
    if (fu::failed(status)) return expected_t::fail(status_from_(status), "spawning {} threads failed", threads);
    return {};
}

#pragma endregion Executor

} // namespace unum::uform::utilities
