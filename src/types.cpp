/**
 *  @file src/types.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The verbs of @c types.h: outcomes, executors, arenas, the name tables and the
 *      library-wide probes, and the one call each executor holds at a time.
 *
 *  A deferred call on a caller-exclusive pool runs whole on the pool's thread 0, which forks the
 *  rest of the pool from inside its slice and ends the call there; every other call runs on its
 *  caller. Nothing waits in a table, so a wait, a drain or a free only waits.
 */
#include <cassert> // `assert` - a callback freeing the executor running it
#include <cstring> // `std::memcpy` - a listed device's name

#include <iterator> // `std::size` - the GPU vendors
#include <memory>   // `std::unique_ptr` - a handle under construction
#include <new>      // `std::nothrow`
#include <span>     // `std::span` - the devices a VRAM executor names

#include "encoder_punned.hpp" // `entries_kit_of`, `vendor_of` - a listed device's description

namespace ue = unum::uform::encoder;
using namespace unum::uform;

#pragma region Executor Calls

/** Holds @p arena for the call whose published outcome is @p call, answering whether it was free or
 *  already that call's; a null arena holds nothing, left for the verb to refuse. */
static bool hold_arena(uform_arena_t arena, uform_outcome_t const* call) noexcept {
    if (!arena) return true;
    uform_outcome_t const* holder = nullptr;
    // Sequentially consistent, as an export opening re-reads the holder after its own count
    return fu::standard_atomic_ref<uform_outcome_t const*> {arena->holder}.compare_exchange_strong(holder, call) ||
           holder == call;
}

void uf::release_arena(uform_arena_t arena, uform_outcome_t const* call) noexcept {
    if (arena)
        fu::standard_atomic_ref<uform_outcome_t const*> {arena->holder}.compare_exchange_strong(
            call, nullptr, std::memory_order_release);
}

bool uf::take_call_claim(uform_executor_handle_t& executor) noexcept {
    fu::standard_atomic_ref<call_claim_t> claim {executor.holding_call.claim};
    if (claim.exchange(call_claim_t::held_k, std::memory_order_acquire) != call_claim_t::free_k) return false;
    // The last call's thread 0 may still be leaving the slice it ended that call in
    if (delegates(executor)) executor.threads.unsafe_join(0);
    return true;
}

void uf::record_status(uform_executor_t executor, uform_outcome_t const& recorded) noexcept {
    executor->holding_call.recorded = recorded;
}

bool uf::delegates(uform_executor_handle_t& executor) noexcept {
    return executor.threads.caller_exclusivity() == fu::caller_exclusive_k;
}

void uf::end_call(uform_executor_handle_t& executor) noexcept {
    holding_call_t& call = executor.holding_call;
    uform_outcome_t* const outcome = call.outcome;
    uform_outcome_t const recorded = call.recorded;
    for (util::u32_t* const named : call.named_calls)
        if (named) fu::standard_atomic_ref<util::u32_t> {*named}.fetch_sub(1, std::memory_order_release);
    call.named_calls = {};
    fu::standard_atomic_ref<void const*>(call.arguments).store(nullptr, std::memory_order_relaxed);
    fu::standard_atomic_ref<util::u64_t>(call.body_thread).store(0, std::memory_order_relaxed);
    call.delegated_arguments.emplace<std::monostate>();
    // ! Past this store a waiter may free the executor; only the caller's outcome is touched after
    fu::standard_atomic_ref<call_claim_t> {call.claim}.store(call_claim_t::free_k, std::memory_order_release);
    propagate_error(recorded.status, outcome, recorded.message);
}

host_verbs_t const& uf::host_verbs() noexcept {
    static host_verbs_t const* const picked = [] noexcept {
        auto [ceiling, status] = pick_cpu_ceiling(uform_cpu_class_unknown_k);
        return status ? pick_host_verbs(cpu_ladder_t {}, ceiling) : host_class_tables<cpu_ladder_floor_k>::find();
    }();
    return *picked;
}

bool uf::runs_holding_call(uform_executor_handle_t& executor) noexcept {
    return fu::standard_atomic_ref<util::u64_t> {executor.holding_call.body_thread}.load(std::memory_order_relaxed) ==
           fu::current_thread_id();
}

void uf::take_idle_claim(uform_executor_handle_t& executor) noexcept {
    while (!take_call_claim(executor)) host_verbs().wait_while_claimed(&executor.holding_call.claim);
}

uform_arena_t uf::claim_arena(uform_arena_t arena, uform_executor_t executor, std::initializer_list<void const*> inputs,
                              call_outcome_t& outcome) noexcept {
    if (!arena) return propagate_error(uform_status_invalid_argument_k, outcome, "The arena must not be null"), nullptr;
    bool const arena_on_device = arena->executor && !std::holds_alternative<std::monostate>(arena->executor->device);
    if (arena_on_device && arena->executor != executor)
        return propagate_error(uform_status_device_k, outcome,
                               "A device arena serves calls on the executor that made it"),
               nullptr;
    if (!arena_on_device && executor && !std::holds_alternative<std::monostate>(executor->device))
        return propagate_error(uform_status_device_k, outcome, "Host pages serve no call computing on a GPU"), nullptr;
    if (!hold_arena(arena, outcome.published()))
        return propagate_error(uform_status_contended_k, outcome, "The arena serves another call"), nullptr;
    // An export opened between the entry's check and the hold above sees the hold and backs out,
    // or this load sees its count
    if (fu::standard_atomic_ref<util::u32_t> {arena->exports}.load())
        return propagate_error(uform_status_contended_k, outcome, "An export of the arena's outputs is open"), nullptr;
    for (void const* input : inputs)
        if (input && arena->chunks.contains(input))
            return propagate_error(uform_status_invalid_argument_k, outcome,
                                   "An input lies inside the arena it rewinds"),
                   nullptr;
    arena->chunks.rewind();
    return arena;
}

util::status_t uf::reserve_arena(uform_arena_handle_t& arena, std::size_t bytes) noexcept {
    chunked_arena_t& chunks = arena.chunks;
    if (chunks.room() >= bytes) return {};
    if (!arena.executor || std::holds_alternative<std::monostate>(arena.executor->device))
        return util::reserve_host_pages(chunks, bytes);
    std::size_t const held = chunks.held(), width = held > bytes ? held : bytes;
    return std::visit(
        [&]<typename executor_type_>(executor_type_& gpus) noexcept -> util::status_t {
            if constexpr (std::is_same_v<executor_type_, std::monostate>)
                return {util::status_code_t::device_k, "The arena's executor drives no GPU"};
            else {
                auto& first = gpus.stream(0);
                if (util::status_t const bound = first.bind(); bound.failed()) return bound;
                auto [address, allocate_status] = first.allocate_async(width);
                if (allocate_status.failed()) return allocate_status;
                if (util::status_t const appended = chunks.append({reinterpret_cast<util::b8_t*>(address), width});
                    appended.failed())
                    return first.free_async(address), appended;
                // Other streams copy into the arena and the trainer writes it synchronously, so the
                // rare growth completes on the first stream before the memory leaves it
                return first.wait();
            }
        },
        arena.executor->device);
}

#pragma endregion Executor Calls

#pragma region Executors

/** Whether the OS numbers a NUMA node of @p topology @p id. */
static bool has_memory_domain(fu::topology_t const& topology, util::i32_t id) noexcept {
    for (std::size_t index = 0; index != topology.memory_domains_count(); ++index)
        if (topology.memory_domain(index) == id) return true;
    return false;
}

/** Copies the NUMA nodes @p ram names into @p domains, every node of @p topology when it names
 *  none. False once @p outcome records why not. */
static bool adopt_memory_domains(st::basic_vector<util::i32_t>& domains, uform_ram_options_t const& ram,
                                 fu::topology_t const& topology, uform_outcome_t* outcome) noexcept {
    if (ram.memory_domains_count && !ram.memory_domains)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The memory domains must not be null");
    std::size_t const count = ram.memory_domains_count ? ram.memory_domains_count : topology.memory_domains_count();
    if (st::failed(domains.resize(count)))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to list the memory domains");
    for (std::size_t index = 0; index != count; ++index) {
        domains[index] = ram.memory_domains_count ? ram.memory_domains[index] : topology.memory_domain(index);
        if (!has_memory_domain(topology, domains[index]))
            return propagate_error(uform_status_invalid_argument_k, outcome,
                                   "A memory domain names no NUMA node of this machine");
    }
    return true;
}

/** Cores of the compute domains local to the NUMA nodes in @p domains . */
static std::size_t cores_of(fu::topology_t const& topology, st::basic_vector<util::i32_t> const& domains) noexcept {
    std::size_t cores = 0;
    for (std::size_t compute = 0; compute != topology.compute_domains_count(); ++compute)
        for (util::i32_t const domain : domains)
            if (topology.memory_domain_near(compute) == domain) cores += topology.cores_in(compute);
    return cores;
}

/** The NumKong vendor bit of @p vendor , which picks its residency, its ladder and its kit. */
static nk_capability_t numkong_vendor_of(util::gpu_vendor_t vendor) noexcept {
    switch (vendor) {
    case util::gpu_vendor_t::nvidia_k: return nk_cap_cuda_k;
    case util::gpu_vendor_t::amd_k: return nk_cap_rocm_k;
    case util::gpu_vendor_t::apple_k: return nk_cap_metal_k;
    case util::gpu_vendor_t::unknown_k: return 0;
    }
    return 0;
}

uf::gpu_runtime_t::gpu_runtime_t() noexcept {
    devices_count = util::list_gpu_devices(drivers, devices);
    // Every class each device loads, by what its vendor's loader runs there
    for (std::size_t index = 0; index != devices_count; ++index) {
        util::gpu_device_t const& device = devices[index];
        visit_vram_residency(
            numkong_vendor_of(device.vendor), [&]<vendored_residency_t residency_, typename ladder_type_>() noexcept {
                if constexpr (gpu_vendor_compiled(vendor_of(residency_)))
                    loaded[index] = gpu_classes_loading(ladder_type_ {}, gpu_vendor_of(residency_),
                                                        entries_kit_of<residency_>::capabilities_compiled(device));
                return true;
            });
    }
}

/** The newest class device @p index of the listing loads under @p ceiling , over its vendor's
 *  ladder, or @c unknown_k where it loads none. */
static gpu_class_t best_gpu_class(std::size_t index, gpu_class_mask_t loaded, uform_gpu_class_t ceiling) noexcept {
    gpu_class_t picked = gpu_class_t::unknown_k;
    std::span<util::gpu_device_t const> const listed = gpu_devices();
    visit_vram_residency(numkong_vendor_of(listed[index].vendor),
                         [&]<vendored_residency_t, typename ladder_type_>() noexcept {
                             picked = pick_gpu_class(ladder_type_ {}, loaded, ceiling);
                             return true;
                         });
    return picked;
}

/** What an executor naming no devices drives, by position in the listing, into @p every : the first
 *  listed device loading a class under @p ceiling , then every device of its vendor that still
 *  shares one under it with all taken before, so the devices taken always share a class. */
static void every_known_gpu(util::per_gpu_array<util::i32_t>& every, uform_gpu_class_t ceiling) noexcept {
    gpu_runtime_t const& runtime = gpu_runtime();
    gpu_class_mask_t shared = 0;
    for (std::size_t index = 0; index != runtime.devices_count; ++index) {
        gpu_class_mask_t const together = every.size() ? shared & runtime.loaded[index] : runtime.loaded[index];
        if (best_gpu_class(index, together, ceiling) == gpu_class_t::unknown_k) continue;
        if (every.size() && runtime.devices[index].vendor != runtime.devices[every[0]].vendor) continue;
        if (every.push_back(static_cast<util::i32_t>(index)) != util::status_code_t::success_k) return;
        shared = together;
    }
}

/** Whether @p vram names one device twice, which adopted streams refuse, one stream per device. */
static bool names_a_device_twice(uform_vram_options_t const& vram) noexcept {
    for (std::size_t index = 0; index != vram.devices_count; ++index)
        for (std::size_t earlier = 0; earlier != index; ++earlier)
            if (vram.devices[earlier] == vram.devices[index]) return true;
    return false;
}

/** The listing rows the positions @p listed name into @p rows , and into @p picked the best class
 *  all of them load, never above @p ceiling , as one model's replicas run one class. False once
 *  @p outcome records why not. */
static bool resolve_placement(std::span<util::i32_t const> listed, uform_gpu_class_t ceiling,
                              util::gpu_device_t (&rows)[util::gpu_devices_capacity_k], gpu_class_t& picked,
                              uform_outcome_t* outcome) noexcept {
    if (listed.size() > util::gpu_devices_capacity_k)
        return propagate_error(uform_status_invalid_argument_k, outcome, "More devices than one process drives");
    gpu_runtime_t const& runtime = gpu_runtime();
    gpu_class_mask_t shared = ~gpu_class_mask_t {0};
    for (std::size_t index = 0; index != listed.size(); ++index) {
        if (listed[index] < 0 || static_cast<std::size_t>(listed[index]) >= runtime.devices_count)
            return propagate_error(uform_status_device_k, outcome, "A device ordinal names no GPU this build drives");
        rows[index] = runtime.devices[listed[index]];
        shared &= runtime.loaded[listed[index]];
        if (rows[index].vendor != rows[0].vendor)
            return propagate_error(uform_status_config_mismatch_k, outcome, "The devices span vendors");
    }
    picked = best_gpu_class(static_cast<std::size_t>(listed[0]), shared, ceiling);
    if (picked == gpu_class_t::unknown_k)
        return propagate_error(uform_status_unsupported_k, outcome,
                               "The GPUs share no compiled class under the ceiling");
    return true;
}

#pragma endregion Executors

/** Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
static bool named(uform_bytes_view_t name, sz::string_view_t spelling) noexcept { return text_of(name) == spelling; }

/** Writes to @p output the one of @p values that @p spell spells as @p name, or fails with
 *  @c uform_status_invalid_argument_k and writes nothing. */
template <typename value_type_, std::size_t count_>
static uform_status_t read_name(uform_bytes_view_t name, value_type_ const (&values)[count_],
                                char const* (*spell)(value_type_), value_type_* output) noexcept {
    if (!output) return uform_status_invalid_argument_k;
    for (value_type_ const value : values)
        if (named(name, spell(value))) return *output = value, uform_status_success_k;
    return uform_status_invalid_argument_k;
}

/** Frees every chunk of @p memory , device memory @p gpus made, in the order of its first stream
 *  with that device bound; a stream that cannot bind leaks them rather than free on another. */
template <typename executor_type_>
static void release_device_chunks(executor_type_& gpus, chunked_arena_t& memory) noexcept {
    if constexpr (!std::is_same_v<executor_type_, std::monostate>) {
        auto& first = gpus.stream(0);
        if (first.bind().failed()) return;
        for (util::bytes_span_t const chunk : memory.chunks())
            first.free_async(reinterpret_cast<std::uintptr_t>(chunk.data()));
    }
}

extern "C" {

#pragma region Outcomes Executors and Arenas

UFORM_EXPORT uform_status_t uform_outcome_wait(uform_outcome_t* outcome) { return host_verbs().wait(outcome); }

UFORM_EXPORT uform_status_t uform_outcome_status(uform_outcome_t* outcome) {
    return fu::standard_atomic_ref<uform_status_t> {outcome->status}.load(std::memory_order_acquire);
}

UFORM_EXPORT void uform_gpu_devices(uform_gpu_devices_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->devices_count)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The count output must not be null"), void();
    if (arguments->devices_capacity && !arguments->devices)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The devices must not be null"), void();
    gpu_runtime_t const& runtime = gpu_runtime(); // a vendor this build or machine lacks lists nothing
    std::size_t const listed = runtime.devices_count;
    for (std::size_t index = 0; index != listed && index != arguments->devices_capacity; ++index) {
        util::gpu_device_t const& device = runtime.devices[index];
        uform_gpu_device_t& entry = arguments->devices[index];
        entry = {};
        entry.vendor = static_cast<uform_gpu_vendor_t>(device.vendor);
        entry.vendor_ordinal = device.vendor_ordinal;
        entry.gpu_class = static_cast<uform_gpu_class_t>(
            best_gpu_class(index, runtime.loaded[index], uform_gpu_class_unknown_k));
        entry.compute_units = device.compute_units;
        entry.memory_bytes = device.memory_bytes;
        std::memcpy(entry.name, device.name, sizeof(entry.name));
    }
    *arguments->devices_count = listed;
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_executor_for_ram(uform_executor_for_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    auto [ceiling, pick_status] = pick_cpu_ceiling(arguments->ram.cpu_class_ceiling);
    if (!pick_status)
        return propagate_error(pick_status.status_, outcome, "No compiled class fits the ceiling"), void();
    fu::topology_t topology;
    if (fu::status_t const harvested = topology.harvest(); fu::failed(harvested))
        return propagate_error(util::status_from_(harvested), outcome, "Topology discovery failed"), void();
    std::unique_ptr<uform_executor_handle_t> executor {new (std::nothrow) uform_executor_handle_t {}};
    if (!executor)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    executor->cpu_ceiling = ceiling;
    st::basic_vector<util::i32_t> memory_domains;
    if (!adopt_memory_domains(memory_domains, arguments->ram, topology, outcome)) return;
    // Deferred calls run on the pool's own threads, the caller only coordinating
    std::size_t const threads = arguments->threads ? arguments->threads
                                                   : fu::default_threads_count(cores_of(topology, memory_domains));
    if (util::expected_t spawned = util::spawn_pool(executor->threads, topology, threads, fu::caller_exclusive_k,
                                                    forkunion_allowed_of(ceiling),
                                                    arguments->ram.memory_domains_count == 1 ? memory_domains[0] : -1);
        !spawned)
        return propagate_error(spawned.status_, outcome, "Failed to spawn the threads"), void();
    *arguments->executor = executor.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_executor_for_vram(uform_executor_for_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    uform_vram_options_t const& requested = arguments->vram;
    if (arguments->streams && !requested.devices_count)
        return propagate_error(uform_status_invalid_argument_k, outcome,
                               "Streams pair with the devices named beside them"),
               void();
    if (requested.devices_count && !requested.devices)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The devices must not be null"), void();
    if (arguments->streams && names_a_device_twice(requested))
        return propagate_error(uform_status_invalid_argument_k, outcome, "Each adopted stream needs its own device"),
               void();
    auto [ceiling, pick_status] = pick_cpu_ceiling(arguments->ram.cpu_class_ceiling);
    if (!pick_status)
        return propagate_error(pick_status.status_, outcome, "No compiled class fits the ceiling"), void();
    if (!gpu_runtime().devices_count)
        return propagate_error(uform_status_device_k, outcome, "No GPU this build drives answers"), void();
    if (requested.gpu_class_ceiling != uform_gpu_class_unknown_k &&
        !listed_in(gpu_classes_k, requested.gpu_class_ceiling))
        return propagate_error(uform_status_invalid_argument_k, outcome, "No GPU class has that value"), void();
    util::per_gpu_array<util::i32_t> every;
    if (!requested.devices_count) every_known_gpu(every, requested.gpu_class_ceiling);
    std::span<util::i32_t const> const listed = requested.devices_count
                                                    ? std::span<util::i32_t const> {requested.devices,
                                                                                    requested.devices_count}
                                                    : every.view();
    if (listed.empty())
        return propagate_error(uform_status_unsupported_k, outcome, "No GPU runs a class this build compiled"), void();
    util::gpu_device_t rows[util::gpu_devices_capacity_k];
    gpu_class_t picked = gpu_class_t::unknown_k;
    if (!resolve_placement(listed, requested.gpu_class_ceiling, rows, picked, outcome)) return;
    fu::topology_t topology;
    if (fu::status_t const harvested = topology.harvest(); fu::failed(harvested))
        return propagate_error(util::status_from_(harvested), outcome, "Topology discovery failed"), void();
    std::unique_ptr<uform_executor_handle_t> executor {new (std::nothrow) uform_executor_handle_t {}};
    if (!executor)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    executor->cpu_ceiling = ceiling;
    executor->gpu_class = picked;
    st::basic_vector<util::i32_t> memory_domains;
    if (!adopt_memory_domains(memory_domains, arguments->ram, topology, outcome)) return;
    // The caller is decoder 0: it stages, feeds the devices and waits, so calls finish on return
    std::size_t const decoders = fu::default_threads_count(cores_of(topology, memory_domains));
    if (util::expected_t spawned = util::spawn_pool(executor->threads, topology, decoders, fu::caller_inclusive_k,
                                                    forkunion_allowed_of(ceiling));
        !spawned)
        return propagate_error(spawned.status_, outcome, "Failed to spawn the decoders"), void();
    // Last, as it adopts the caller's streams only once every fallible step is behind
    std::size_t scratch_limit = arguments->scratch_limit;
    std::span<void* const> const streams {arguments->streams, arguments->streams ? listed.size() : 0};
    gpu_runtime_t& runtime = gpu_runtime();
    auto [device, spawn_status] = util::spawn_gpu_executor(
        runtime.drivers, std::span<util::gpu_device_t const> {rows, listed.size()}, streams, runtime, scratch_limit);
    if (spawn_status.failed())
        return propagate_error(spawn_status, outcome, "The devices refused their streams"), void();
    executor->device = std::move(device);
    *arguments->executor = executor.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_executor_drain(uform_executor_drain_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->executor) // runs every call to completion before returning, so nothing waits behind it
        return propagate_error(uform_status_success_k, outcome, nullptr), void();
    uform_executor_handle_t& executor = *arguments->executor;
    if (arguments->mood != uform_mood_parked_k && arguments->mood != uform_mood_spinning_k)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The mood names no value"), void();
    if (runs_holding_call(executor))
        return propagate_error(uform_status_contended_k, outcome, "A callback cannot drain the executor running it"),
               void();
    take_idle_claim(executor);
    util::status_t const drained = std::visit(
        []<typename executor_type_>(executor_type_& gpus) noexcept -> util::status_t {
            if constexpr (std::is_same_v<executor_type_, std::monostate>) return {};
            else return gpus.wait();
        },
        executor.device);
    // Only the claim holder parks the pool, so no submitter's dispatch races the sleep
    if (drained.succeeded() && arguments->mood == uform_mood_parked_k) executor.threads.sleep();
    fu::standard_atomic_ref<call_claim_t> {executor.holding_call.claim}.store(call_claim_t::free_k,
                                                                              std::memory_order_release);
    if (drained.failed()) return propagate_error(drained, outcome, "A device failed the work queued on it"), void();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT uform_status_t uform_executor_free(uform_executor_t executor) {
    if (!executor) return uform_status_success_k;
    assert(!runs_holding_call(*executor) && "A callback cannot free the executor running it");
    // A refusal never waits: the count is read before the claim
    if (fu::standard_atomic_ref<util::u32_t> {executor->arenas}.load(std::memory_order_acquire) != 0)
        return uform_status_contended_k;
    take_idle_claim(*executor); // never released: the executor dies holding it
    // Each stream record drains its stream, frees its scratch with its device bound, then closes
    delete executor;
    return uform_status_success_k;
}

/** The native handle of @p stream : the CUDA or HIP stream it queues on, or the Metal queue it
 *  commits to. */
static void* native_stream_(util::cuda_executor_t::stream_t const& stream) noexcept {
    return static_cast<util::cuda_stream_handle_t>(stream.handle);
}
static void* native_stream_(util::hip_executor_t::stream_t const& stream) noexcept {
    return static_cast<util::hip_stream_handle_t>(stream.handle);
}
static void* native_stream_(util::metal_executor_t::stream_t const& stream) noexcept { return stream.handle.queue(); }

/** Whether the foreign @p foreign stream sits on @p stream 's device: @c config_mismatch_k when it
 *  sits on another, @c device_k when the driver cannot name it; Metal names no foreign stream. */
static uform_status_t foreign_stream_on_device(util::cuda_executor_t::stream_t& stream, void* foreign) noexcept {
    // Handles 0, 1 and 2 name the current device's default streams, meant as this one's
    if (reinterpret_cast<std::uintptr_t>(foreign) <= 2) return uform_status_success_k;
    auto const& api = stream.handle.api();
    util::cuda_context_handle_t context = nullptr;
    if (api.stream_get_context(static_cast<util::cuda_stream_handle_t>(foreign), &context) != util::cuda_success_k)
        return uform_status_device_k;
    return context == api.primary_contexts[stream.ordinal] ? uform_status_success_k : uform_status_config_mismatch_k;
}
static uform_status_t foreign_stream_on_device(util::hip_executor_t::stream_t& stream, void* foreign) noexcept {
    // Handles 0, 1 and 2 name the current device's default streams, meant as this one's
    if (reinterpret_cast<std::uintptr_t>(foreign) <= 2) return uform_status_success_k;
    int device = -1;
    if (stream.handle.api().stream_get_device(static_cast<util::hip_stream_handle_t>(foreign), &device) !=
        util::hip_success_k)
        return uform_status_device_k;
    return device == static_cast<int>(stream.ordinal) ? uform_status_success_k : uform_status_config_mismatch_k;
}
static uform_status_t foreign_stream_on_device(util::metal_executor_t::stream_t&, void*) noexcept {
    return uform_status_unsupported_k;
}

/** Queues on @p stream a wait for the foreign event @p event points at, or, when null, for all
 *  queued so far on the @p foreign stream, which must sit on @p stream 's device, as its own event
 *  marks it; a null event behind @p event leaves nothing to wait for, and Metal has none. */
static uform_status_t follow_foreign_(util::cuda_executor_t::stream_t& stream, void* event, void* foreign) noexcept {
    if (stream.bind().failed()) return uform_status_device_k;
    if (uform_status_t const placed = foreign_stream_on_device(stream, foreign); placed != uform_status_success_k)
        return placed;
    auto const& api = stream.handle.api();
    util::cuda_event_handle_t const awaited = event ? *static_cast<util::cuda_event_handle_t const*>(event)
                                                    : stream.event;
    if (!awaited) return uform_status_success_k; // Arrow's producer had nothing left to wait for
    if (!event && api.event_record(awaited, static_cast<util::cuda_stream_handle_t>(foreign)) != util::cuda_success_k)
        return uform_status_device_k;
    return api.stream_wait_event(stream.handle, awaited, 0) == util::cuda_success_k ? uform_status_success_k
                                                                                    : uform_status_device_k;
}
static uform_status_t follow_foreign_(util::hip_executor_t::stream_t& stream, void* event, void* foreign) noexcept {
    if (stream.bind().failed()) return uform_status_device_k;
    if (uform_status_t const placed = foreign_stream_on_device(stream, foreign); placed != uform_status_success_k)
        return placed;
    auto const& api = stream.handle.api();
    util::hip_event_handle_t const awaited = event ? *static_cast<util::hip_event_handle_t const*>(event)
                                                   : stream.event;
    if (!awaited) return uform_status_success_k; // Arrow's producer had nothing left to wait for
    if (!event && api.event_record(awaited, static_cast<util::hip_stream_handle_t>(foreign)) != util::hip_success_k)
        return uform_status_device_k;
    return api.stream_wait_event(stream.handle, awaited, 0) == util::hip_success_k ? uform_status_success_k
                                                                                   : uform_status_device_k;
}
static uform_status_t follow_foreign_(util::metal_executor_t::stream_t&, void*, void*) noexcept {
    return uform_status_unsupported_k;
}

UFORM_EXPORT void uform_executor_stream(uform_executor_stream_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->executor || !arguments->stream)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The executor and output must not be null"),
               void();
    *arguments->stream = nullptr;
    std::visit(
        [&]<typename executor_type_>(executor_type_& gpus) noexcept {
            if constexpr (std::is_same_v<executor_type_, std::monostate>)
                propagate_error(uform_status_unsupported_k, outcome, "A RAM executor drives no stream");
            else if (arguments->position >= gpus.size())
                propagate_error(uform_status_invalid_argument_k, outcome, "The position is past the streams");
            else
                *arguments->stream = native_stream_(gpus.stream(arguments->position)),
                propagate_error(uform_status_success_k, outcome, nullptr);
        },
        arguments->executor->device);
}

UFORM_EXPORT void uform_executor_follow(uform_executor_follow_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->executor || !arguments->event == !arguments->stream)
        return propagate_error(uform_status_invalid_argument_k, outcome,
                               "A non-null executor and exactly one of an event and a stream"),
               void();
    uform_executor_handle_t& executor = *arguments->executor;
    // The wait queues on a stream a call may drive, so it holds the executor as a call does
    if (!take_call_claim(executor))
        return propagate_error(uform_status_contended_k, outcome, "Another call holds the executor"), void();
    uform_status_t const followed = std::visit(
        [&]<typename executor_type_>(executor_type_& gpus) noexcept {
            if constexpr (std::is_same_v<executor_type_, std::monostate>) return uform_status_unsupported_k;
            else {
                if (arguments->position >= gpus.size()) return uform_status_invalid_argument_k;
                // Queueing on the executor's stream drives it, as a call does
                auto [claim, claim_status] = gpus.claim();
                if (claim_status.failed()) return uform_status_contended_k;
                return follow_foreign_(gpus.stream(arguments->position), arguments->event, arguments->stream);
            }
        },
        executor.device);
    fu::standard_atomic_ref<call_claim_t> {executor.holding_call.claim}.store(call_claim_t::free_k,
                                                                              std::memory_order_release);
    propagate_error(followed, outcome, followed == uform_status_success_k ? nullptr : "The stream could not follow");
}

UFORM_EXPORT void uform_executor_accepts(uform_executor_accepts_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->executor)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The executor must not be null"), void();
    uform_status_t const accepted = std::visit(
        [&]<typename executor_type_>(executor_type_& gpus) noexcept {
            if constexpr (std::is_same_v<executor_type_, std::monostate>) return uform_status_unsupported_k;
            else if (arguments->position >= gpus.size()) return uform_status_invalid_argument_k;
            else return foreign_stream_on_device(gpus.stream(arguments->position), arguments->stream);
        },
        arguments->executor->device);
    propagate_error(accepted, outcome, accepted == uform_status_success_k ? nullptr : "The stream is not accepted");
}

UFORM_EXPORT void uform_arena_init(uform_arena_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->arena = nullptr;
    // A VRAM executor lands outputs on its first stream's device, any other on the host
    uform_executor_t const executor = arguments->executor;
    uform_arena_t const arena = new (std::nothrow) uform_arena_handle_t {{}, executor};
    if (!arena) return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the arena"), void();
    // Every arena made with an executor counts, RAM included
    util::u32_t constexpr most_k = (std::numeric_limits<util::u32_t>::max)();
    if (executor && fu::atomic_fetch_add_if_at_most(fu::standard_atomic_ref<util::u32_t> {executor->arenas}, 1u, most_k,
                                                    std::memory_order_acq_rel) == most_k)
        return delete arena, propagate_error(uform_status_overflow_k, outcome, "The executor counts too many arenas"),
               void();
    *arguments->arena = arena;
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT uform_status_t uform_arena_free(uform_arena_t arena) {
    if (!arena) return uform_status_success_k;
    host_verbs().wait_named_calls(&arena->named_calls);
    if (fu::standard_atomic_ref<util::u32_t> {arena->exports}.load()) return uform_status_contended_k;
    uform_executor_t const executor = arena->executor;
    if (!executor || std::holds_alternative<std::monostate>(executor->device)) util::release_host_pages(arena->chunks);
    else std::visit([&](auto& gpus) noexcept { release_device_chunks(gpus, arena->chunks); }, executor->device);
    arena->chunks.forget();
    delete arena;
    // Last, as the memory went back through the executor
    if (executor) fu::standard_atomic_ref<util::u32_t> {executor->arenas}.fetch_sub(1, std::memory_order_release);
    return uform_status_success_k;
}

UFORM_EXPORT uform_status_t uform_arena_export_open(uform_arena_t arena) {
    if (!arena) return uform_status_invalid_argument_k;
    fu::standard_atomic_ref<util::u32_t> exports {arena->exports};
    util::u32_t open = exports.load();
    do
        if (open == (std::numeric_limits<util::u32_t>::max)()) return uform_status_overflow_k;
    while (!exports.compare_exchange_weak(open, open + 1));
    // A call holding the arena may rewind it at any moment, so the count backs out
    if (fu::standard_atomic_ref<uform_outcome_t const*> {arena->holder}.load())
        return exports.fetch_sub(1), uform_status_contended_k;
    return uform_status_success_k;
}

UFORM_EXPORT void uform_arena_export_close(uform_arena_t arena) {
    if (!arena) return;
    fu::standard_atomic_ref<util::u32_t> exports {arena->exports};
    util::u32_t open = exports.load(std::memory_order_relaxed);
    while (open &&
           !exports.compare_exchange_weak(open, open - 1, std::memory_order_release, std::memory_order_relaxed));
}

UFORM_EXPORT void uform_arena_describe(uform_arena_describe_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->arena)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The arena must not be null"), void();
    if (!arguments->description)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The description must not be null"), void();
    *arguments->description = {uform_residency_ram_k, -1, uform_gpu_vendor_unknown_k, -1, uform_host_access_readable_k};
    if (arguments->arena->executor)
        std::visit(
            [&]<typename executor_type_>(executor_type_ const& gpus) noexcept {
                if constexpr (!std::is_same_v<executor_type_, std::monostate>) {
                    // Apple's GPU and CPU share one memory, so its device outputs stay readable
                    constexpr uform_host_access_t host_access = executor_type_::vendor_k == util::gpu_vendor_t::apple_k
                                                                    ? uform_host_access_readable_k
                                                                    : uform_host_access_refused_k;
                    *arguments->description = {uform_residency_vram_k,
                                               listed_ordinal_of<executor_type_>(gpus.stream(0)),
                                               static_cast<uform_gpu_vendor_t>(executor_type_::vendor_k),
                                               static_cast<int32_t>(gpus.stream(0).ordinal), host_access};
                }
            },
            arguments->arena->executor->device);
    propagate_error(uform_status_success_k, outcome, nullptr);
}

#pragma endregion Outcomes Executors and Arenas

#pragma region Names

UFORM_EXPORT char const* uform_status_name(uform_status_t status) {
    switch (status) {
    case uform_status_success_k: return "success";
    case uform_status_deferred_k: return "deferred";
    case uform_status_out_of_memory_k: return "out-of-memory";
    case uform_status_shape_mismatch_k: return "shape-mismatch";
    case uform_status_unsupported_layout_k: return "unsupported-layout";
    case uform_status_unknown_kind_k: return "unknown-kind";
    case uform_status_io_k: return "io";
    case uform_status_device_k: return "device";
    case uform_status_malformed_k: return "malformed";
    case uform_status_internal_k: return "internal";
    case uform_status_overflow_k: return "overflow";
    case uform_status_uninitialized_k: return "uninitialized";
    case uform_status_config_mismatch_k: return "config-mismatch";
    case uform_status_contended_k: return "contended";
    case uform_status_cancelled_k: return "cancelled";
    case uform_status_unsupported_k: return "unsupported";
    case uform_status_invalid_argument_k: return "invalid-argument";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_status_from_name(uform_bytes_view_t name, uform_status_t* status) {
    return read_name<uform_status_t>(
        name,
        {uform_status_success_k, uform_status_deferred_k, uform_status_out_of_memory_k, uform_status_shape_mismatch_k,
         uform_status_unsupported_layout_k, uform_status_unknown_kind_k, uform_status_io_k, uform_status_device_k,
         uform_status_malformed_k, uform_status_internal_k, uform_status_overflow_k, uform_status_uninitialized_k,
         uform_status_config_mismatch_k, uform_status_contended_k, uform_status_cancelled_k, uform_status_unsupported_k,
         uform_status_invalid_argument_k},
        uform_status_name, status);
}

UFORM_EXPORT char const* uform_scalar_name(uform_scalar_t scalar) {
    switch (scalar) {
    case uform_scalar_unknown_k: return "unknown";
    case uform_scalar_f64_k: return "f64";
    case uform_scalar_f32_k: return "f32";
    case uform_scalar_bf16_k: return "bf16";
    case uform_scalar_f16_k: return "f16";
    case uform_scalar_e5m2_k: return "e5m2";
    case uform_scalar_e4m3_k: return "e4m3";
    case uform_scalar_e3m2_k: return "e3m2";
    case uform_scalar_e2m3_k: return "e2m3";
    case uform_scalar_i8_k: return "i8";
    case uform_scalar_i4x2_k: return "i4x2";
    case uform_scalar_i32_k: return "i32";
    case uform_scalar_u8_k: return "u8";
    case uform_scalar_u4x2_k: return "u4x2";
    case uform_scalar_u1x8_k: return "u1x8";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_scalar_from_name(uform_bytes_view_t name, uform_scalar_t* scalar) {
    return read_name<uform_scalar_t>(
        name,
        {uform_scalar_unknown_k, uform_scalar_f64_k, uform_scalar_f32_k, uform_scalar_bf16_k, uform_scalar_f16_k,
         uform_scalar_e5m2_k, uform_scalar_e4m3_k, uform_scalar_e3m2_k, uform_scalar_e2m3_k, uform_scalar_i8_k,
         uform_scalar_i4x2_k, uform_scalar_i32_k, uform_scalar_u8_k, uform_scalar_u4x2_k, uform_scalar_u1x8_k},
        uform_scalar_name, scalar);
}

UFORM_EXPORT char const* uform_mood_name(uform_mood_t mood) {
    switch (mood) {
    case uform_mood_parked_k: return "parked";
    case uform_mood_spinning_k: return "spinning";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_mood_from_name(uform_bytes_view_t name, uform_mood_t* mood) {
    return read_name<uform_mood_t>(name, {uform_mood_parked_k, uform_mood_spinning_k}, uform_mood_name, mood);
}

UFORM_EXPORT char const* uform_cpu_class_name(uform_cpu_class_t cpu_class) {
    switch (cpu_class) {
    case uform_cpu_class_unknown_k: return "unknown";
    case uform_cpu_class_serial_k: return "serial";
    case uform_cpu_class_intel_haswell_k: return "intel-haswell";
    case uform_cpu_class_intel_sierra_k: return "intel-sierra";
    case uform_cpu_class_intel_skylake_k: return "intel-skylake";
    case uform_cpu_class_intel_icelake_k: return "intel-icelake";
    case uform_cpu_class_amd_genoa_k: return "amd-genoa";
    case uform_cpu_class_intel_sapphire_k: return "intel-sapphire";
    case uform_cpu_class_intel_granite_k: return "intel-granite";
    case uform_cpu_class_intel_diamond_k: return "intel-diamond";
    case uform_cpu_class_aws_g2_k: return "aws-g2";
    case uform_cpu_class_apple_m2_k: return "apple-m2";
    case uform_cpu_class_apple_m4_k: return "apple-m4";
    case uform_cpu_class_aws_g5_k: return "aws-g5";
    case uform_cpu_class_nvidia_vera_k: return "nvidia-vera";
    case uform_cpu_class_alibaba_c930_k: return "alibaba-c930";
    case uform_cpu_class_ibm_power10_k: return "ibm-power10";
    case uform_cpu_class_wasm32_v128_k: return "wasm32-v128";
    case uform_cpu_class_wasm64_extended_k: return "wasm64-extended";
    case uform_cpu_class_monoversioned_k: return "monoversioned";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_cpu_class_from_name(uform_bytes_view_t name, uform_cpu_class_t* cpu_class) {
    if (cpu_class && named(name, uform_cpu_class_name(uform_cpu_class_unknown_k)))
        return *cpu_class = uform_cpu_class_unknown_k, uform_status_success_k;
    return read_name<uform_cpu_class_t>(name, cpu_classes_k, uform_cpu_class_name, cpu_class);
}

UFORM_EXPORT char const* uform_gpu_class_name(uform_gpu_class_t gpu_class) {
    switch (gpu_class) {
    case uform_gpu_class_unknown_k: return "unknown";
    case uform_gpu_class_nvidia_ampere_k: return "nvidia-ampere";
    case uform_gpu_class_nvidia_ada_k: return "nvidia-ada";
    case uform_gpu_class_nvidia_hopper_k: return "nvidia-hopper";
    case uform_gpu_class_nvidia_blackwell_k: return "nvidia-blackwell";
    case uform_gpu_class_nvidia_blackwellrtx_k: return "nvidia-blackwellrtx";
    case uform_gpu_class_nvidia_rubin_k: return "nvidia-rubin";
    case uform_gpu_class_amd_cdna3_k: return "amd-cdna3";
    case uform_gpu_class_amd_cdna4_k: return "amd-cdna4";
    case uform_gpu_class_amd_cdna5_k: return "amd-cdna5";
    case uform_gpu_class_apple_metal7_k: return "apple-metal7";
    case uform_gpu_class_apple_metal9_k: return "apple-metal9";
    case uform_gpu_class_apple_metal10_k: return "apple-metal10";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_gpu_class_from_name(uform_bytes_view_t name, uform_gpu_class_t* gpu_class) {
    if (gpu_class && named(name, uform_gpu_class_name(uform_gpu_class_unknown_k)))
        return *gpu_class = uform_gpu_class_unknown_k, uform_status_success_k;
    return read_name<uform_gpu_class_t>(name, gpu_classes_k, uform_gpu_class_name, gpu_class);
}

UFORM_EXPORT char const* uform_gpu_vendor_name(uform_gpu_vendor_t vendor) {
    switch (vendor) {
    case uform_gpu_vendor_unknown_k: return "unknown";
    case uform_gpu_vendor_nvidia_k: return "nvidia";
    case uform_gpu_vendor_amd_k: return "amd";
    case uform_gpu_vendor_apple_k: return "apple";
    }
    return "unrecognized";
}

UFORM_EXPORT char const* uform_host_access_name(uform_host_access_t host_access) {
    switch (host_access) {
    case uform_host_access_refused_k: return "refused";
    case uform_host_access_readable_k: return "readable";
    }
    return "unrecognized";
}

UFORM_EXPORT char const* uform_residency_name(uform_residency_t residency) {
    switch (residency) {
    case uform_residency_unknown_k: return "unknown";
    case uform_residency_ram_k: return "ram";
    case uform_residency_vram_k: return "vram";
    }
    return "unrecognized";
}

UFORM_EXPORT char const* uform_modality_name(uform_modality_t modality) {
    switch (modality) {
    case uform_modality_text_k: return "text";
    case uform_modality_image_k: return "image";
    case uform_modality_video_k: return "video";
    case uform_modality_audio_k: return "audio";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_modality_from_name(uform_bytes_view_t name, uform_modality_t* modality) {
    return read_name<uform_modality_t>(
        name, {uform_modality_text_k, uform_modality_image_k, uform_modality_video_k, uform_modality_audio_k},
        uform_modality_name, modality);
}

UFORM_EXPORT char const* uform_objective_name(uform_objective_t objective) {
    switch (objective) {
    case uform_objective_masked_k: return "masked";
    case uform_objective_contrastive_k: return "contrastive";
    case uform_objective_joint_k: return "joint";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_objective_from_name(uform_bytes_view_t name, uform_objective_t* objective) {
    return read_name<uform_objective_t>(
        name, {uform_objective_masked_k, uform_objective_contrastive_k, uform_objective_joint_k}, uform_objective_name,
        objective);
}

UFORM_EXPORT char const* uform_negatives_name(uform_negatives_t negatives) {
    switch (negatives) {
    case uform_negatives_ambient_k: return "ambient";
    case uform_negatives_named_k: return "named";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_negatives_from_name(uform_bytes_view_t name, uform_negatives_t* negatives) {
    return read_name<uform_negatives_t>(name, {uform_negatives_ambient_k, uform_negatives_named_k},
                                        uform_negatives_name, negatives);
}

UFORM_EXPORT char const* uform_multimem_tier_name(uform_multimem_tier_t tier) {
    switch (tier) {
    case uform_multimem_tier_switch_reduced_k: return "switch-reduced";
    case uform_multimem_tier_peer_read_k: return "peer-read";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_multimem_tier_from_name(uform_bytes_view_t name, uform_multimem_tier_t* tier) {
    return read_name<uform_multimem_tier_t>(
        name, {uform_multimem_tier_switch_reduced_k, uform_multimem_tier_peer_read_k}, uform_multimem_tier_name, tier);
}

UFORM_EXPORT char const* uform_pair_loss_form_name(uform_pair_loss_form_t form) {
    switch (form) {
    case uform_pair_loss_form_derived_k: return "derived";
    case uform_pair_loss_form_balanced_k: return "balanced";
    case uform_pair_loss_form_ranked_k: return "ranked";
    case uform_pair_loss_form_sigmoid_k: return "sigmoid";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_pair_loss_form_from_name(uform_bytes_view_t name, uform_pair_loss_form_t* form) {
    return read_name<uform_pair_loss_form_t>(name,
                                             {uform_pair_loss_form_derived_k, uform_pair_loss_form_balanced_k,
                                              uform_pair_loss_form_ranked_k, uform_pair_loss_form_sigmoid_k},
                                             uform_pair_loss_form_name, form);
}

UFORM_EXPORT char const* uform_late_interaction_side_name(uform_late_interaction_side_t side) {
    switch (side) {
    case uform_late_interaction_side_none_k: return "none";
    case uform_late_interaction_side_query_k: return "query";
    case uform_late_interaction_side_document_k: return "document";
    case uform_late_interaction_side_query_and_document_k: return "query-and-document";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_late_interaction_side_from_name(uform_bytes_view_t name,
                                                                  uform_late_interaction_side_t* side) {
    return read_name<uform_late_interaction_side_t>(
        name,
        {uform_late_interaction_side_none_k, uform_late_interaction_side_query_k,
         uform_late_interaction_side_document_k, uform_late_interaction_side_query_and_document_k},
        uform_late_interaction_side_name, side);
}

#pragma endregion Names

#pragma region Probes

UFORM_EXPORT void uform_version(int* major, int* minor, int* patch) {
    *major = UFORM_VERSION_MAJOR, *minor = UFORM_VERSION_MINOR, *patch = UFORM_VERSION_PATCH;
}

/** The build reads @c VERSION whole, pre-release suffix included, which the macros cannot spell. */
UFORM_EXPORT char const* uform_version_name(void) { return UFORM_VERSION_NAME; }

UFORM_EXPORT char const* uform_capabilities_compiled(void) {
    static char const* const rendered = [] {
        static char text[NUMKONG_CAPABILITIES_NAME_CAPACITY] = {};
        nk_capability_t compiled = 0;
        [[maybe_unused]] nk_status_t const cpu_status = nk_capabilities_compiled_cpu(&compiled);
        for (nk_capability_t const vendor : {nk_cap_cuda_k, nk_cap_rocm_k, nk_cap_metal_k})
            if (gpu_vendor_compiled(vendor)) compiled |= vendor;
        for (uform_gpu_class_t const gpu_class : gpu_classes_k)
            if (gpu_class_compiled(static_cast<gpu_class_t>(gpu_class)))
                compiled |= numkong_capabilities_of(static_cast<gpu_class_t>(gpu_class));
        nk_capabilities_name(compiled, text, sizeof(text));
        return text;
    }();
    return rendered;
}

UFORM_EXPORT char const* uform_capabilities_available(void) {
    static char const* const rendered = [] {
        static char text[NUMKONG_CAPABILITIES_NAME_CAPACITY] = {};
        auto [ceiling, status] = pick_cpu_ceiling(uform_cpu_class_unknown_k);
        nk_capability_t available = status ? ceiling.compute : nk_cap_serial_k;
        gpu_runtime_t const& runtime = gpu_runtime();
        for (std::size_t index = 0; index != runtime.devices_count; ++index)
            available |= numkong_capabilities_of(
                best_gpu_class(index, runtime.loaded[index], uform_gpu_class_unknown_k));
        nk_capabilities_name(available, text, sizeof(text));
        return text;
    }();
    return rendered;
}

#pragma endregion Probes

} // extern "C"
