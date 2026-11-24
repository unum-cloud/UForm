/**
 *  @file include/uform/utilities/ram_device.hpp
 *  @author Ash Vardanian
 *  @date October 27, 2026
 *  @brief The one owner of host threads and topology, over ForkUnion's C ABI.
 *
 *  Engines, loaders, tools and tests dispatch through a @c pool_view_t of a @c ram_executor_t,
 *  whose threads are a pool spawned and deleted here or the calling thread alone; nothing else
 *  calls @c fu_* .
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <type_traits> // `std::is_nothrow_invocable_v`, `std::remove_reference_t`
#include <utility>     // `std::exchange`, `std::swap`
#include <variant>     // `std::variant`, `std::get_if`, `std::get`, a pool or the calling thread

#include <forkunion/forkunion.h> // `fu_pool_t`, `fu_topology_t`, `fu_pool_for_n`

#include "uform/utilities/types.hpp" // `status_t`, `expected`, `bytes_span_t`, `fu::dummy_pool_t`

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

/** A callable run once per task, with the task index and the index of the thread running it. */
template <typename callback_>
concept task_callback = std::is_nothrow_invocable_v<callback_&, std::size_t, std::size_t>;

/** What the thread that spawns a pool does on every dispatch it makes afterwards. */
enum class caller_exclusivity_t : std::uint8_t {

    /** Counts as one of the threads, runs its share of every dispatch, and is pinned
     *  like the rest. */
    caller_inclusive_k,

    /** Only dispatches and waits, keeping its own affinity and adding nothing to
     *  the thread count. */
    caller_exclusive_k,
};

/** The threads an executor dispatches on: a ForkUnion pool it spawned and deletes, or the calling
 *  thread alone. */
using threads_t = std::variant<fu_pool_t, fu::dummy_pool_t>;

/**
 *  @brief Dispatches on the threads of an executor owned elsewhere; copying it copies the right to
 *      dispatch, never the pool.
 *
 *  A dispatch claims the pool itself, so one finding the pool running another dispatch, its own or
 *  another caller's, runs nothing and answers @c status_code_t::contended_k . On the calling thread
 *  every dispatch runs its tasks in order, as thread 0.
 */
class pool_view_t {
  public:
    explicit constexpr pool_view_t(threads_t threads) noexcept : threads_(threads) {}

    /** Runs @p callback over @p count similar-cost tasks split evenly across the threads. */
    template <task_callback callback_>
    [[nodiscard]] status_t for_n(std::size_t count, callback_&& callback) const noexcept {
        if (fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_))
            return status_from_(fu_pool_for_n(*pool, count, &task_trampoline_<callback_>, &callback));
        fu::dummy_pool_t serial = std::get<fu::dummy_pool_t>(threads_); // a copy, since its `for_n` is not `const`
        serial.for_n(count, calling_thread_task_<callback_> {callback});
        return {};
    }

    /** Runs @p callback over @p count tasks of uneven cost, each thread taking the next task as
     *  it finishes one. */
    template <task_callback callback_>
    [[nodiscard]] status_t for_n_dynamic(std::size_t count, callback_&& callback) const noexcept {
        if (fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_))
            return status_from_(fu_pool_for_n_dynamic(*pool, count, &task_trampoline_<callback_>, &callback));
        fu::dummy_pool_t serial = std::get<fu::dummy_pool_t>(threads_); // a copy, since its `for_n` is not `const`
        serial.for_n_dynamic(count, calling_thread_task_<callback_> {callback});
        return {};
    }

    /** Runs @p fork once per thread inside a dispatch the caller joins with @ref unsafe_join ,
     *  answering whether the pool took it; the calling thread runs @p fork before it returns. */
    [[nodiscard]] bool unsafe_for_threads(fu_for_threads_t fork, fu_lambda_context_t context) const noexcept {
        // The token stays local: thread 0 may end its work before this thread stores it
        fu_generation_t generation = 0;
        if (fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_))
            return fu_pool_unsafe_for_threads(*pool, fork, context, &generation) == fu_success_k;
        fork(context, 0, 0);
        return true;
    }

    /** Waits until the threads leave the dispatch @ref unsafe_for_threads started; the calling
     *  thread has already left it. */
    void unsafe_join() const noexcept {
        if (fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_)) fu_pool_unsafe_join(*pool, 0);
    }

    /** Parks the threads in the kernel until the next dispatch; the calling thread has none to
     *  park. */
    void park() const noexcept {
        if (fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_)) fu_pool_sleep(*pool, 0);
    }

    /** Whether the thread that spawned the pool runs its share of every dispatch; the calling
     *  thread always does. */
    [[nodiscard]] caller_exclusivity_t caller_exclusivity() const noexcept {
        fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_);
        if (!pool) return caller_exclusivity_t::caller_inclusive_k;
        fu_caller_exclusivity_t exclusivity = fu_caller_inclusive_k;
        if (fu_pool_caller_exclusivity(*pool, &exclusivity) != fu_success_k)
            return caller_exclusivity_t::caller_inclusive_k;
        return exclusivity == fu_caller_exclusive_k ? caller_exclusivity_t::caller_exclusive_k
                                                    : caller_exclusivity_t::caller_inclusive_k;
    }

    /** The number of threads every dispatch runs on, the caller included when it is a worker. */
    [[nodiscard]] std::size_t threads_count() const noexcept {
        fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_);
        if (!pool) return 1;
        std::size_t threads = 0;
        return fu_pool_threads_count(*pool, &threads) == fu_success_k ? threads : 0;
    }

  private:

    /** A task of @p callback_ as ForkUnion's C pool runs it, its context addressing the
     *  callback. */
    template <typename callback_>
    static void task_trampoline_(fu_lambda_context_t context, std::size_t task, std::size_t thread,
                                 std::size_t) noexcept {
        (*static_cast<std::remove_reference_t<callback_>*>(context))(task, thread);
    }

    /** A task of @p callback_ as the calling thread runs it, as thread 0. */
    template <typename callback_>
    struct calling_thread_task_ {
        std::remove_reference_t<callback_>& callback;
        void operator()(std::size_t task, fu::dummy_pool_t::thread_in_domain_t) const noexcept { callback(task, 0); }
    };

    /** The threads every dispatch runs on, owned elsewhere. */
    threads_t threads_;
};

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

/**
 *  @brief The threads of an executor: a ForkUnion pool spawned here, or the calling thread.
 *
 *  The calling-thread executor, which @ref on_calling_thread and a default or moved-from one hold,
 *  creates no pool, no thread and no topology probe, and runs every dispatch in order on its
 *  caller. Of a spawned pool, a worker caller counts as one of the threads and runs its share of
 *  every dispatch, while a coordinator only waits. Every dispatch blocks until all threads return,
 *  and refuses while another one runs.
 */
class ram_executor_t {
  public:

    /** The label the pool's threads have in the OS. */
    static constexpr char const pool_name_k[] = "uform";

    /** The calling thread alone, as a moved-from executor is, until @ref make spawns a pool. */
    constexpr ram_executor_t() noexcept = default;

    ram_executor_t(ram_executor_t&& other) noexcept
        : threads_(std::exchange(other.threads_, threads_t {std::in_place_type<fu::dummy_pool_t>})) {}

    ram_executor_t& operator=(ram_executor_t&& other) noexcept {
        std::swap(threads_, other.threads_);
        return *this;
    }

    ram_executor_t(ram_executor_t const&) = delete;
    ram_executor_t& operator=(ram_executor_t const&) = delete;

    ~ram_executor_t() noexcept {
        if (fu_pool_t const* const pool = std::get_if<fu_pool_t>(&threads_); pool && *pool) fu_pool_delete(*pool);
    }

    /** The calling thread alone: no pool, no thread, no topology probe. */
    [[nodiscard]] static ram_executor_t on_calling_thread() noexcept { return ram_executor_t {}; }

    /** How a pool runs deferred calls: on its own threads, the caller only coordinating, where the
     *  build has OS threads, and with its caller among them otherwise, as on WebAssembly. */
    [[nodiscard]] static caller_exclusivity_t deferring_exclusivity() noexcept {
        return (fu_comptime_capabilities() & fu_capability_os_threads_k) ? caller_exclusivity_t::caller_exclusive_k
                                                                         : caller_exclusivity_t::caller_inclusive_k;
    }

    /**
     *  @brief Harvests the topology and spawns @p threads threads across the whole machine, the
     *      caller among them for a worker.
     *  @param[in] threads Thread count, at least one.
     *  @param[in] exclusivity Whether the caller is one of the @p threads or only coordinates them.
     */
    [[nodiscard]] static expected<ram_executor_t> make(std::size_t threads, caller_exclusivity_t exclusivity) noexcept {
        auto [topology, status] = topology_t::make();
        if (!status) return expected<ram_executor_t>::fail(std::move(status));
        return make(topology, threads, exclusivity);
    }

    /** Spawns @p threads threads interleaved across every compute domain of @p topology , or only
     *  the caller in a build without OS threads, such as WebAssembly's. */
    [[nodiscard]] static expected<ram_executor_t> make(topology_t const& topology, std::size_t threads,
                                                       caller_exclusivity_t exclusivity) noexcept {
        if ((fu_comptime_capabilities() & fu_capability_os_threads_k) == 0) threads = 1;
        fu_pool_t pool = nullptr;
        if (fu_status_t const status = fu_pool_new(pool_name_k, fu_capabilities_all_k, &pool); status != fu_success_k)
            return expected<ram_executor_t>::fail(status_from_(status), "Pool creation failed");
        ram_executor_t executor;
        executor.threads_ = pool;
        if (fu_status_t const status = fu_pool_spawn(topology.get(), pool, threads, spawned_(exclusivity));
            status != fu_success_k)
            return expected<ram_executor_t>::fail(status_from_(status), "spawning {} threads failed", threads);
        return executor;
    }

    /** Spawns @p threads threads over the compute domains local to @p memory_domain , or only the
     *  caller in a build without OS threads. */
    [[nodiscard]] static expected<ram_executor_t> spawn_near(topology_t const& topology,
                                                             fu_memory_domain_id_t memory_domain, std::size_t threads,
                                                             caller_exclusivity_t exclusivity) noexcept {
        if ((fu_comptime_capabilities() & fu_capability_os_threads_k) == 0) threads = 1;
        fu_pool_t pool = nullptr;
        if (fu_status_t const status = fu_pool_new(pool_name_k, fu_capabilities_all_k, &pool); status != fu_success_k)
            return expected<ram_executor_t>::fail(status_from_(status), "Pool creation failed");
        ram_executor_t executor;
        executor.threads_ = pool;
        if (fu_status_t const status = fu_pool_spawn_near_memory_domain(topology.get(), pool, memory_domain, threads,
                                                                        spawned_(exclusivity));
            status != fu_success_k)
            return expected<ram_executor_t>::fail(
                status_from_(status), "spawning {} threads near memory domain {} failed", threads, memory_domain);
        return executor;
    }

    /** The number of threads every dispatch runs on, the caller included when it is a worker. */
    [[nodiscard]] std::size_t threads_count() const noexcept { return view().threads_count(); }

    /** The right to dispatch on the threads, which stay owned here. */
    [[nodiscard]] pool_view_t view() const noexcept { return pool_view_t {threads_}; }

  private:

    /** ForkUnion's spelling of @p exclusivity . */
    static constexpr fu_caller_exclusivity_t spawned_(caller_exclusivity_t exclusivity) noexcept {
        return exclusivity == caller_exclusivity_t::caller_inclusive_k ? fu_caller_inclusive_k : fu_caller_exclusive_k;
    }

    /** The threads every dispatch runs on, a pool owned and deleted here or the calling thread. */
    threads_t threads_ {std::in_place_type<fu::dummy_pool_t>};
};

#pragma endregion Executor

} // namespace unum::uform::utilities
