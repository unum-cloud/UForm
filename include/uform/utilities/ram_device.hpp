/**
 *  @file include/uform/utilities/ram_device.hpp
 *  @author Ash Vardanian
 *  @date October 27, 2026
 *  @brief The one owner of host threads and topology, over ForkUnion's C ABI.
 *
 *  Engines, loaders, tools and tests dispatch through a @c pool_view_t of either an
 *  @c executor_t, a pool spawned here, or a @c lent_pool_t, a pool another library lends; nothing
 *  else calls @c fu_* .
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <type_traits> // `std::is_nothrow_invocable_v`, `std::remove_reference_t`
#include <utility>     // `std::exchange`, `std::swap`

#include <forkunion/forkunion.h> // `fu_pool_t`, `fu_topology_t`, `fu_pool_for_n`

#include "uform/utilities/types.hpp" // `status_t`, `expected`

namespace unum::uform::utilities {

#pragma region Executor

/** Translates a ForkUnion failure into the status every UForm call reports. */
constexpr status_t status_from_(fu_status_t status) noexcept {
    switch (status) {
    case fu_success_k: return status_t::success_k;
    case fu_bad_alloc_k: return status_t::bad_alloc_k;
    case fu_invalid_argument_k: return status_t::invalid_argument_k;
    case fu_contended_k: return status_t::contended_k;
    default: return status_t::internal_k;
    }
}

/** A callable run once per task, with the task index and the index of the thread running it. */
template <typename callback_>
concept task_callback = std::is_nothrow_invocable_v<callback_ &, std::size_t, std::size_t>;

/** What the thread that spawns a pool does on every dispatch it makes afterwards. */
enum class caller_exclusivity_t : std::uint8_t {

    /** Counts as one of the threads, runs its share of every dispatch, and is pinned like the
     *  rest. */
    caller_inclusive_k,

    /** Only dispatches and waits, keeping its own affinity and adding nothing to the thread
     *  count. */
    caller_exclusive_k,
};

/**
 *  @brief Dispatches on a ForkUnion pool owned elsewhere; copying it copies the right to dispatch,
 *      never the pool.
 *
 *  A dispatch claims the pool itself, so one finding the pool running another dispatch, its own or
 *  another caller's, runs nothing and answers @c status_t::contended_k .
 */
class pool_view_t {
  public:
    explicit constexpr pool_view_t(fu_pool_t pool) noexcept : pool_(pool) {}

    /** Runs @p callback over @p count similar-cost tasks split evenly across the threads. */
    template <task_callback callback_>
    [[nodiscard]] status_t for_n(std::size_t count, callback_ &&callback) const noexcept {
        fu_for_task_t const trampoline = [](fu_lambda_context_t context, std::size_t task, std::size_t thread,
                                            std::size_t) noexcept {
            (*static_cast<std::remove_reference_t<callback_> *>(context))(task, thread);
        };
        return status_from_(fu_pool_for_n(pool_, count, trampoline, &callback));
    }

    /** Runs @p callback over @p count tasks of uneven cost, each thread taking the next task as it
     *  finishes one. */
    template <task_callback callback_>
    [[nodiscard]] status_t for_n_dynamic(std::size_t count, callback_ &&callback) const noexcept {
        fu_for_task_t const trampoline = [](fu_lambda_context_t context, std::size_t task, std::size_t thread,
                                            std::size_t) noexcept {
            (*static_cast<std::remove_reference_t<callback_> *>(context))(task, thread);
        };
        return status_from_(fu_pool_for_n_dynamic(pool_, count, trampoline, &callback));
    }

    /** The number of threads every dispatch runs on, the caller included when it is a worker. */
    [[nodiscard]] std::size_t threads_count() const noexcept {
        std::size_t threads = 0;
        return fu_pool_threads_count(pool_, &threads) == fu_success_k ? threads : 0;
    }

  private:

    /** The pool every dispatch runs on, owned elsewhere. */
    fu_pool_t pool_;
};

/** The machine as ForkUnion harvests it, its cores, compute domains and memory domains, owned and
 *  deleted here. */
class topology_t {
  public:
    topology_t(topology_t &&other) noexcept : topology_(std::exchange(other.topology_, nullptr)) {}

    topology_t &operator=(topology_t &&other) noexcept {
        std::swap(topology_, other.topology_);
        return *this;
    }

    topology_t(topology_t const &) = delete;
    topology_t &operator=(topology_t const &) = delete;

    ~topology_t() noexcept { fu_topology_delete(topology_); }

    /** Harvests the topology of the machine the process runs on. */
    [[nodiscard]] static expected<topology_t> make() noexcept {
        topology_t topology;
        if (fu_status_t const status = fu_topology_new(&topology.topology_); status != fu_success_k)
            return expected<topology_t>::fail(status_from_(status), "topology discovery failed");
        return topology;
    }

    /** Counts the memory domains the process may allocate on. */
    [[nodiscard]] std::size_t memory_domains_count() const noexcept {
        std::size_t count = 0;
        return fu_memory_domains_count(topology_, &count) == fu_success_k ? count : 0;
    }

    /** The identifier the operating system gives memory domain @p index , or -1 past the last
     *  one. */
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
    constexpr topology_t() noexcept = default;

    /** The harvested topology, owned and deleted here. */
    fu_topology_t topology_ = nullptr;
};

/**
 *  @brief A ForkUnion pool spawned and deleted here.
 *
 *  A worker caller counts as one of the threads and runs its share of every dispatch, while a
 *  coordinator only waits. Every dispatch blocks until all threads return, and refuses while
 *  another one runs.
 */
class executor_t {
  public:

    /** The label the pool's threads carry in the OS. */
    static constexpr char const pool_name_k[] = "uform";

    /** An executor holding no pool, as a moved-from one does, until @ref make assigns one. */
    constexpr executor_t() noexcept = default;

    executor_t(executor_t &&other) noexcept : pool_(std::exchange(other.pool_, nullptr)) {}

    executor_t &operator=(executor_t &&other) noexcept {
        std::swap(pool_, other.pool_);
        return *this;
    }

    executor_t(executor_t const &) = delete;
    executor_t &operator=(executor_t const &) = delete;

    ~executor_t() noexcept { fu_pool_delete(pool_); }

    /**
     *  @brief Harvests the topology and spawns @p threads threads across the whole machine, the
     *      caller among them for a worker.
     *  @param[in] threads Thread count, at least one.
     *  @param[in] exclusivity Whether the caller is one of the @p threads or only coordinates them.
     */
    [[nodiscard]] static expected<executor_t> make(std::size_t threads, caller_exclusivity_t exclusivity) noexcept {
        expected<topology_t> topology = topology_t::make();
        if (!topology) return expected<executor_t>::fail(std::move(topology));
        return make(*topology, threads, exclusivity);
    }

    /** Spawns @p threads threads interleaved across every compute domain of @p topology , or only
     *  the caller in a build without OS threads, such as WebAssembly's. */
    [[nodiscard]] static expected<executor_t> make(topology_t const &topology, std::size_t threads,
                                                   caller_exclusivity_t exclusivity) noexcept {
        if ((fu_comptime_capabilities() & fu_capability_os_threads_k) == 0) threads = 1;
        executor_t executor;
        if (fu_status_t const status = fu_pool_new(pool_name_k, fu_capabilities_all_k, &executor.pool_);
            status != fu_success_k)
            return expected<executor_t>::fail(status_from_(status), "pool creation failed");
        if (fu_status_t const status = fu_pool_spawn(topology.get(), executor.pool_, threads, spawned_(exclusivity));
            status != fu_success_k)
            return expected<executor_t>::fail(status_from_(status), "spawning {} threads failed", threads);
        return executor;
    }

    /** The number of threads every dispatch runs on, the caller included when it is a worker. */
    [[nodiscard]] std::size_t threads_count() const noexcept { return view().threads_count(); }

    /** The right to dispatch on the pool, which stays owned here. */
    [[nodiscard]] pool_view_t view() const noexcept { return pool_view_t {pool_}; }

  private:

    /** ForkUnion's spelling of @p exclusivity . */
    static constexpr fu_caller_exclusivity_t spawned_(caller_exclusivity_t exclusivity) noexcept {
        return exclusivity == caller_exclusivity_t::caller_inclusive_k ? fu_caller_inclusive_k : fu_caller_exclusive_k;
    }

    /** The pool every dispatch runs on, owned and deleted here. */
    fu_pool_t pool_ = nullptr;
};

/** A pool lent by another library: dispatched on while its lender's own dispatch does not run,
 *  never deleted. */
class lent_pool_t {
  public:

    /** Borrows @p pool , refusing a null pool or one holding no threads. */
    [[nodiscard]] static expected<lent_pool_t> borrow(fu_pool_t pool) noexcept {
        if (!pool || pool_view_t {pool}.threads_count() == 0)
            return expected<lent_pool_t>::fail(status_t::invalid_argument_k, "the lent pool holds no threads");
        return lent_pool_t {pool};
    }

    /** The right to dispatch on the pool, which stays with its lender. */
    [[nodiscard]] pool_view_t view() const noexcept { return pool_view_t {pool_}; }

  private:
    explicit constexpr lent_pool_t(fu_pool_t pool) noexcept : pool_(pool) {}

    /** The pool every dispatch runs on, which its lender deletes. */
    fu_pool_t pool_;
};

#pragma endregion Executor

} // namespace unum::uform::utilities
