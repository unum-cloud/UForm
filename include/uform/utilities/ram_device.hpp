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

#include <utility> // `std::exchange`, `std::swap`

#include <forkunion/forkunion.hpp> // `fu::pool_t`, `fu::default_threads_count`, `fu_topology_t`

#include "uform/utilities/types.hpp" // `status_t`, `expected`, `bytes_span_t`

namespace unum::uform::utilities {

#pragma region Executor

/** Translates a ForkUnion failure into the code every UForm call reports. */
constexpr status_code_t status_from_(fu_status_t status) noexcept {
    switch (status) {
    case fu_success_k: return status_code_t::success_k;
    case fu_bad_alloc_k: return status_code_t::out_of_memory_k;
    case fu_invalid_argument_k: return status_code_t::invalid_argument_k;
    case fu_contended_k: return status_code_t::contended_k;
    default: return status_code_t::internal_k;
    }
}

/** Translates a failure of a @c fu::pool_t dispatch, which spells the same codes. */
constexpr status_code_t status_from_(fu::status_t status) noexcept {
    return status_from_(static_cast<fu_status_t>(status));
}

/** The machine as ForkUnion harvests it, its cores, compute domains and memory domains, owned
 *  and deleted here. */
class topology_t {
  public:
    /** A topology holding nothing, as a moved-from one does, until @ref make harvests one. */
    constexpr topology_t() noexcept = default;

    topology_t(topology_t&& other) noexcept : topology_(std::exchange(other.topology_, nullptr)) {}

    topology_t& operator=(topology_t&& other) noexcept {
        std::swap(topology_, other.topology_);
        return *this;
    }

    topology_t(topology_t const&) = delete;
    topology_t& operator=(topology_t const&) = delete;

    ~topology_t() noexcept { fu_topology_delete(topology_); }

    /** Harvests the topology of the machine the process runs on. */
    [[nodiscard]] static expected<topology_t> make() noexcept {
        topology_t topology;
        if (fu_status_t const status = fu_topology_new(&topology.topology_); status != fu_success_k)
            return expected<topology_t>::fail(status_from_(status), "Topology discovery failed");
        return topology;
    }

    /** Counts the memory domains the process may allocate on. */
    [[nodiscard]] std::size_t memory_domains_count() const noexcept {
        std::size_t count = 0;
        return fu_memory_domains_count(topology_, &count) == fu_success_k ? count : 0;
    }

    /** The identifier the operating system gives memory domain @p index , or -1 past
     *  the last one. */
    [[nodiscard]] fu_memory_domain_id_t memory_domain(std::size_t index) const noexcept {
        fu_memory_domain_id_t identifier = -1;
        return fu_memory_domain_id_at_index(topology_, index, &identifier) == fu_success_k ? identifier : -1;
    }

    /** Counts the compute domains, the clusters of cores a pool spreads its threads over. */
    [[nodiscard]] std::size_t compute_domains_count() const noexcept {
        std::size_t count = 0;
        return fu_compute_domains_count(topology_, &count) == fu_success_k ? count : 0;
    }

    /** The identifier of the memory domain local to compute domain @p compute_domain . */
    [[nodiscard]] fu_memory_domain_id_t memory_domain_near(std::size_t compute_domain) const noexcept {
        std::size_t index = 0;
        if (fu_local_memory_of(topology_, compute_domain, &index) != fu_success_k) return -1;
        return memory_domain(index);
    }

    /** Counts the logical cores of compute domain @p compute_domain . */
    [[nodiscard]] std::size_t cores_in(std::size_t compute_domain) const noexcept {
        std::size_t cores = 0;
        return fu_logical_cores_count_in(topology_, compute_domain, &cores) == fu_success_k ? cores : 0;
    }

    /** Counts the logical cores this process may run on. */
    [[nodiscard]] std::size_t logical_cores_count() const noexcept {
        std::size_t cores = 0;
        return fu_logical_cores_count(topology_, &cores) == fu_success_k ? cores : 0;
    }

    /** The ForkUnion topology itself, which pools spawn over. */
    [[nodiscard]] fu_topology_t get() const noexcept { return topology_; }

  private:
    /** The harvested topology, owned and deleted here. */
    fu_topology_t topology_ = nullptr;
};

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
 *      @p allowed names; a build without OS threads, such as WebAssembly's, spawns only the
 *      caller, as a worker.
 *  @param[in] exclusivity Whether the caller is one of the @p threads or only coordinates them.
 */
[[nodiscard]] inline expected_t spawn_pool(fu::pool_t& pool, topology_t const& topology, std::size_t threads,
                                           fu::caller_exclusivity_t exclusivity, fu_capabilities_t allowed,
                                           fu_memory_domain_id_t memory_domain = -1) noexcept {
    if ((fu_comptime_capabilities() & fu_capability_os_threads_k) == 0)
        threads = 1, exclusivity = fu::caller_inclusive_k;
    fu::status_t const status = memory_domain < 0
                                    ? pool.spawn(topology.get(), threads, exclusivity, allowed, pool_name_k)
                                    : pool.spawn_near(topology.get(), memory_domain, threads, exclusivity, allowed,
                                                      pool_name_k);
    if (fu::failed(status)) return expected_t::fail(status_from_(status), "spawning {} threads failed", threads);
    return {};
}

#pragma endregion Executor

} // namespace unum::uform::utilities
