/**
 *  @file src/types.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The verbs of @c types.h: outcomes, executors, arenas, the name tables and the library-wide probes, and the
 *      registry of deferred calls behind every verb.
 *
 *  A deferred call waits in its executor's table until something needs its result: a wait on its outcome, a drain,
 *  a blocking call on the same executor, or a free of a handle it names. Whichever thread asks runs it, and every
 *  call held before it on that executor, in submission order.
 */
#include <cstring> // `std::memcpy` - a listed device's name

#include <memory> // `std::unique_ptr` - a handle under construction
#include <new>    // `std::nothrow`

#include "encoder_punned.hpp" // `entries_kit_of`, `vendor_of` - a listed device's description

namespace ue = unum::uform::encoder;
using namespace unum::uform;

deferred_registry_t uf::deferred_registry;
thread_local void const *uf::resuming_call = nullptr;

#pragma region Deferred Calls

/** Whether the call in @p call reads or writes @p handle: its outcome, its executor, or a model, arena, mesh or
 *  trainer it names. An output for a handle the call creates holds nothing yet. */
static bool deferred_uses(deferred_call_t const &call, void const *handle) noexcept {
    return std::visit(
        [&](auto const &arguments) noexcept {
            auto const names = [&](auto named) noexcept {
                if constexpr (std::is_pointer_v<std::remove_pointer_t<decltype(named)>>) return false;
                else return static_cast<void const *>(named) == handle;
            };
            bool used = static_cast<void const *>(arguments.outcome) == handle ||
                        static_cast<void const *>(arguments.executor) == handle;
            if constexpr (requires { arguments.model; }) used = used || names(arguments.model);
            if constexpr (requires { arguments.arena; }) used = used || names(arguments.arena);
            if constexpr (requires { arguments.mesh; }) used = used || names(arguments.mesh);
            if constexpr (requires { arguments.trainer; }) used = used || names(arguments.trainer);
            return used;
        },
        call.arguments);
}

/** Holds @p arena for the call whose published outcome is @p call, answering whether it was free or already that
 *  call's; a null arena holds nothing, left for the verb to refuse. */
static bool hold_arena(uform_arena_t arena, uform_outcome_t const *call) noexcept {
    if (!arena) return true;
    uform_outcome_t const *holder = nullptr;
    return fu::standard_atomic_ref<uform_outcome_t const *> {arena->holder}.compare_exchange_strong(
               holder, call, std::memory_order_acquire) ||
           holder == call;
}

void uf::release_arena(uform_arena_t arena, uform_outcome_t const *call) noexcept {
    if (arena)
        fu::standard_atomic_ref<uform_outcome_t const *> {arena->holder}.compare_exchange_strong(
            call, nullptr, std::memory_order_release);
}

void uf::run_held(uform_executor_t executor, u64_t last) noexcept {
    for (;;) {
        deferred_call_t *oldest = nullptr;
        {
            st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
            for (deferred_call_t &call : executor->deferred)
                if (call.stage == deferred_stage_t::held_k && call.ticket <= last &&
                    (!oldest || call.ticket < oldest->ticket))
                    oldest = &call;
            if (!oldest) return;
            oldest->stage = deferred_stage_t::running_k;
        }
        std::visit(
            [&](auto &arguments) noexcept {
                resuming_call = &arguments;
                oldest->verb(&arguments);
            },
            oldest->arguments);
        {
            st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
            oldest->stage = deferred_stage_t::free_k;
            fu::standard_atomic_ref<u64_t> {deferred_registry.finished}.fetch_add(1, std::memory_order_release);
        }
    }
}

/** Waits until no call @p executor took with a ticket up to @p last still runs on another thread. */
static void wait_running(uform_executor_handle_t &executor, u64_t last) noexcept {
    for (;;) {
        u64_t seen = 0;
        {
            st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
            std::size_t running = 0;
            for (deferred_call_t const &call : executor.deferred)
                running += call.stage == deferred_stage_t::running_k && call.ticket <= last;
            if (!running) return;
            seen = fu::standard_atomic_ref<u64_t> {deferred_registry.finished}.load(std::memory_order_relaxed);
        }
        executor.host_verbs->wait(deferred_registry.finished, seen);
    }
}

void uf::finish_calls_using(void const *handle) noexcept {
    if (!handle) return;
    for (;;) {
        uform_executor_handle_t *holding = nullptr;
        host_verbs_t const *running = nullptr; // a static table, still valid once the lock lets the executor go
        u64_t last = 0, seen = 0;
        {
            st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
            for (uform_executor_handle_t *executor = deferred_registry.executors; executor && !holding;
                 executor = executor->next_listed)
                for (deferred_call_t const &call : executor->deferred)
                    if (call.stage == deferred_stage_t::held_k && deferred_uses(call, handle))
                        holding = executor, last = last > call.ticket ? last : call.ticket;
                    else if (call.stage == deferred_stage_t::running_k && deferred_uses(call, handle))
                        running = executor->host_verbs;
            seen = fu::standard_atomic_ref<u64_t> {deferred_registry.finished}.load(std::memory_order_relaxed);
        }
        if (holding) run_held(holding, last); // takes the lock itself, so it runs outside it
        else if (running) running->wait(deferred_registry.finished, seen);
        else return;
    }
}

void uf::hold_deferred(uform_executor_t executor, deferred_arguments_t const &arguments, void (*verb)(void *),
                       uform_arena_t arena, uform_outcome_t *outcome) noexcept {
    propagate_pending(outcome);
    if (!hold_arena(arena, outcome))
        return propagate_error(uform_status_contended_k, outcome, "The arena serves another call"), void();
    {
        st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
        for (deferred_call_t &call : executor->deferred)
            if (call.stage == deferred_stage_t::free_k) {
                call.arguments = arguments;
                call.verb = verb;
                call.ticket = ++deferred_registry.tickets;
                call.stage = deferred_stage_t::held_k;
                return;
            }
    }
    release_arena(arena, outcome); // a refused submission never runs, so nothing else would let go of its arena
    propagate_error(uform_status_contended_k, outcome, "The deferred table is full");
}

uform_arena_t uf::claim_arena(uform_arena_t arena, uform_executor_t executor,
                              std::initializer_list<void const *> inputs, call_outcome_t &outcome) noexcept {
    if (!arena) return propagate_error(uform_status_invalid_argument_k, outcome, "The arena must not be null"), nullptr;
    if (std::holds_alternative<device_arena_t>(arena->memory) && arena->executor != executor)
        return propagate_error(uform_status_device_k, outcome,
                               "A device arena serves calls on the executor that made it"),
               nullptr;
    if (std::holds_alternative<host_arena_t>(arena->memory) && executor &&
        std::holds_alternative<vram_executor_t>(executor->executor))
        return propagate_error(uform_status_device_k, outcome, "Host pages serve no call computing on a GPU"), nullptr;
    if (!hold_arena(arena, outcome.published()))
        return propagate_error(uform_status_contended_k, outcome, "The arena serves another call"), nullptr;
    chunked_arena_t &memory = chunks_of(arena->memory);
    for (void const *input : inputs)
        if (input && memory.contains(input))
            return propagate_error(uform_status_invalid_argument_k, outcome,
                                   "An input lies inside the arena it rewinds"),
                   nullptr;
    memory.rewind();
    return arena;
}

#pragma endregion Deferred Calls

#pragma region Executors

/** Threads an executor spawns over @p cores when its caller names none: 7/8 of them, and at least one. */
static std::size_t derived_threads(std::size_t cores) noexcept {
    std::size_t const threads = cores - cores / idle_cores_denominator_k;
    return threads ? threads : 1;
}

/** The NUMA node of @p topology at position @p index. */
static fu::memory_domain_t const &node_at(fu::machine_topology_t const &topology, std::size_t index) noexcept {
    return topology.memory_domain_at(static_cast<fu::memory_domain_index_t>(index));
}

/** The NUMA node of @p topology the OS numbers @p id, or null. */
static fu::memory_domain_t const *node_of(fu::machine_topology_t const &topology, i32_t id) noexcept {
    for (std::size_t index = 0; index != topology.memory_domains_count(); ++index)
        if (node_at(topology, index).memory_domain_id == id) return &node_at(topology, index);
    return nullptr;
}

/** Copies the NUMA nodes @p ram names into @p domains, every node of @p topology when it names none. False once
 *  @p outcome records why not. */
static bool adopt_memory_domains(st::basic_vector<i32_t> &domains, uform_ram_options_t const &ram,
                                 fu::machine_topology_t const &topology, uform_outcome_t *outcome) noexcept {
    if (ram.memory_domains_count && !ram.memory_domains)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The memory domains must not be null");
    std::size_t const count = ram.memory_domains_count ? ram.memory_domains_count : topology.memory_domains_count();
    if (st::failed(domains.resize(count)))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to list the memory domains");
    for (std::size_t index = 0; index != count; ++index) {
        domains[index] = ram.memory_domains_count ? ram.memory_domains[index]
                                                  : node_at(topology, index).memory_domain_id;
        if (!node_of(topology, domains[index]))
            return propagate_error(uform_status_invalid_argument_k, outcome,
                                   "A memory domain names no NUMA node of this machine");
    }
    return true;
}

/** Cores the NUMA nodes in @p domains carry, every one of them a node of @p topology. */
static std::size_t cores_of(fu::machine_topology_t const &topology, st::basic_vector<i32_t> const &domains) noexcept {
    std::size_t cores = 0;
    for (i32_t const domain : domains) cores += node_of(topology, domain)->core_ids.size();
    return cores;
}

/** Sizes @p executor 's deferred table at @p limit entries, zero taking the default, gives it the
 *  host verbs of @p cpu_class , lists it, and hands it out through @p output; frees it once
 *  @p outcome records why not. */
static void hand_out_executor(std::unique_ptr<uform_executor_handle_t> executor, cpu_class_t cpu_class,
                              std::size_t limit, uform_executor_t *output, uform_outcome_t *outcome) noexcept {
    if (st::failed(executor->deferred.resize(limit ? limit : deferred_limit_default_k)))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the deferred table"), void();
    executor->host_verbs = pick_host_verbs(cpu_ladder_t {}, cpu_class);
    {
        st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
        executor->next_listed = deferred_registry.executors;
        deferred_registry.executors = executor.get();
    }
    *output = executor.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** Takes @p executor off the list; the caller already finished every call it held. */
static void delist_executor(uform_executor_handle_t &executor) noexcept {
    st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
    for (uform_executor_handle_t **link = &deferred_registry.executors; *link; link = &(*link)->next_listed)
        if (*link == &executor) {
            *link = executor.next_listed;
            return;
        }
}

/** The capabilities a device of @p detected capabilities runs under @p pinned : its class's, a
 *  pinned class's it holds, or, for a device no class covers and nothing pins, its own; zero when
 *  the pin refuses it. */
static nk_capability_t device_class_capabilities(nk_capability_t detected, uform_gpu_class_t pinned) noexcept {
    gpu_class_t const device_class = gpu_class_of(detected);
    if (device_class == gpu_class_t::unknown_k && pinned == uform_gpu_class_unknown_k) return detected;
    return gpu_pin_capabilities(device_class, pinned);
}

/** Reads the name, compute units and memory of @p home_ 's device @p vendor_ordinal through its
 *  vendor's entries kit; false for a vendor this build lacks or a runtime that refuses. */
template <encoder_home_t home_>
static bool describe_device(std::size_t vendor_ordinal, ue::device_description_t &description) noexcept {
    if constexpr (!gpu_vendor_compiled(vendor_of(home_))) return false;
    else return entries_kit_of<home_>::describe(vendor_ordinal, description) == nk_success_k;
}

/** Opens a stream and a scratch arena on the GPU the process-wide @p listed ordinal names, CUDA's
 *  devices first, then ROCm's, dispatching as @p pinned , and appends it to @p devices. False once
 *  @p outcome records why not. */
static bool open_device(per_gpu_array<vram_device_t> &devices, i32_t listed, nk_size_t cuda_devices,
                        nk_size_t rocm_devices, uform_gpu_class_t pinned, std::size_t scratch_limit,
                        uform_outcome_t *outcome) noexcept {
    if (listed < 0 || static_cast<nk_size_t>(listed) >= cuda_devices + rocm_devices)
        return propagate_error(uform_status_device_k, outcome, "A device ordinal names no GPU this build drives");
    nk_size_t const flat_ordinal = static_cast<nk_size_t>(listed);
    nk::device_kind_t const kind = flat_ordinal < cuda_devices ? nk::device_kind_t::cuda_k : nk::device_kind_t::rocm_k;
    nk_size_t const vendor_ordinal = kind == nk::device_kind_t::cuda_k ? flat_ordinal : flat_ordinal - cuda_devices;
    nk::expected<nk::device_t> const device = nk::device_t::make(kind, vendor_ordinal);
    if (!device) return propagate_error(status_of(static_cast<nk_status_t>(device.status)), outcome, "No such GPU");
    nk::expected<nk_capability_t> const detected = device.value.capabilities_enabled();
    if (!detected || !detected.value)
        return propagate_error(uform_status_device_k, outcome, "This build holds no kernels for the GPU");
    nk_capability_t const capabilities = device_class_capabilities(detected.value, pinned);
    if (!capabilities) return propagate_error(uform_status_unsupported_k, outcome, "The GPU does not admit the pin");
    nk::expected<owned_stream_t> stream = owned_stream_t::make(device.value);
    if (!stream)
        return propagate_error(status_of(static_cast<nk_status_t>(stream.status)), outcome, "Failed to open a stream");
    nk_allocator_t policy {}, pages {};
    if (nk_status_t const status = nk_allocator_init_device_best(&policy, capabilities); status != nk_success_k)
        return propagate_error(status_of(status), outcome, "The GPU has no device allocator");
    if (nk_status_t const status = nk_allocator_init_pinned_best(&pages, capabilities); status != nk_success_k)
        return propagate_error(status_of(status), outcome, "The GPU has no pinned host allocator");
    nk_stream_t const handle = stream.value.get();
    vram_device_t opened {listed,
                          static_cast<i32_t>(vendor_ordinal),
                          capabilities,
                          std::move(stream.value),
                          device_arena_t {chunked_arena_t(policy, handle, scratch_limit)},
                          chunked_arena_t(pages, handle, 0)};
    if (devices.push_back(std::move(opened)) != core::status_t::success_k)
        return propagate_error(uform_status_invalid_argument_k, outcome, "More devices than one process drives");
    return true;
}

#pragma endregion Executors

extern "C" {

#pragma region Outcomes Executors and Arenas

UFORM_EXPORT uform_status_t uform_outcome_wait(uform_outcome_t *outcome) {
    finish_calls_using(outcome); // waits on a deferred call through its executor's class waiter
    fu::standard_atomic_ref<uform_status_t> status {outcome->status};
    uform_status_t seen = status.load(std::memory_order_acquire);
    for (; seen == uform_status_deferred_k; seen = status.load(std::memory_order_acquire))
        fu::standard_yield_t {}(&outcome->status, seen, std::size_t {0}, fu::wait_capped_k); // no executor owns it
    return seen;
}

UFORM_EXPORT uform_status_t uform_outcome_status(uform_outcome_t *outcome) {
    return fu::standard_atomic_ref<uform_status_t> {outcome->status}.load(std::memory_order_acquire);
}

UFORM_EXPORT void uform_gpu_devices(uform_gpu_devices_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->devices_count)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The count output must not be null"), void();
    if (arguments->devices_capacity && !arguments->devices)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The devices must not be null"), void();
    std::size_t listed = 0;
    for (nk::device_kind_t const kind : {nk::device_kind_t::cuda_k, nk::device_kind_t::rocm_k}) {
        nk::expected<std::size_t> const count = nk::device_t::count(kind);
        if (!count) continue; // a vendor this build or machine lacks lists nothing
        for (std::size_t ordinal = 0; ordinal != count.value; ++ordinal, ++listed) {
            if (listed >= arguments->devices_capacity) continue;
            nk::expected<nk::device_t> const device = nk::device_t::make(kind, ordinal);
            nk::expected<nk_capability_t> const detected = device ? device.value.capabilities_enabled()
                                                                  : nk::expected<nk_capability_t> {};
            nk_capability_t const capabilities = detected ? detected.value : 0;
            uform_gpu_device_t &entry = arguments->devices[listed];
            entry = {};
            entry.vendor = kind == nk::device_kind_t::cuda_k ? uform_gpu_vendor_nvidia_k : uform_gpu_vendor_amd_k;
            entry.vendor_ordinal = static_cast<u32_t>(ordinal);
            entry.gpu_class = static_cast<uform_gpu_class_t>(gpu_class_of(capabilities));
            ue::device_description_t description;
            bool const described = kind == nk::device_kind_t::cuda_k
                                       ? describe_device<encoder_home_t::vram_nvidia_k>(ordinal, description)
                                       : describe_device<encoder_home_t::vram_amd_k>(ordinal, description);
            if (!described) continue;
            std::memcpy(entry.name, description.name, sizeof(entry.name));
            entry.compute_units = description.compute_units;
            entry.memory_bytes = description.memory_bytes;
        }
    }
    *arguments->devices_count = listed;
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_executor_for_ram(uform_executor_for_ram_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest()))
        return propagate_error(uform_status_device_k, outcome, "Topology discovery failed"), void();
    std::unique_ptr<uform_executor_handle_t> executor {new (std::nothrow) uform_executor_handle_t {}};
    if (!executor)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    ram_executor_t &ram = std::get<ram_executor_t>(executor->executor);
    if (!adopt_memory_domains(ram.memory_domains, arguments->ram, topology, outcome)) return;
    std::size_t const threads = arguments->threads ? arguments->threads
                                                   : derived_threads(cores_of(topology, ram.memory_domains));
    if (fu::failed(ram.pool.spawn(threads)))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to spawn the threads"), void();
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k)
        return propagate_error(uform_status_device_k, outcome, "The CPU capabilities could not be read"), void();
    if (!cpu_pin_admitted(arguments->ram.cpu_class, available))
        return propagate_error(uform_status_unsupported_k, outcome, "The CPU does not admit the pinned class"), void();
    cpu_class_t const cpu_class = pick_cpu_class(cpu_ladder_t {}, arguments->ram.cpu_class, available);
    ram.capabilities = numkong_capabilities_of(cpu_class);
    hand_out_executor(std::move(executor), cpu_class, arguments->deferred_limit, arguments->executor, outcome);
}

UFORM_EXPORT void uform_executor_on_pool(uform_executor_on_pool_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    propagate_error(uform_status_unsupported_k, outcome, "Borrowed pools are not wired yet");
}

UFORM_EXPORT void uform_executor_for_vram(uform_executor_for_vram_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    uform_vram_options_t const &vram = arguments->vram;
    if (arguments->streams)
        return propagate_error(uform_status_unsupported_k, outcome, "Borrowed streams are not wired yet"), void();
    if (vram.devices_count && !vram.devices)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The devices must not be null"), void();
    nk_size_t cuda_devices = 0, rocm_devices = 0;
    if (nk_cuda_count_devices(&cuda_devices) != nk_success_k) cuda_devices = 0;
    if (nk_rocm_count_devices(&rocm_devices) != nk_success_k) rocm_devices = 0;
    if (!cuda_devices && !rocm_devices)
        return propagate_error(uform_status_device_k, outcome, "No GPU this build drives answers"), void();
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest()))
        return propagate_error(uform_status_device_k, outcome, "Topology discovery failed"), void();
    std::unique_ptr<uform_executor_handle_t> executor {new (std::nothrow) uform_executor_handle_t {}};
    if (!executor)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    vram_executor_t &devices = executor->executor.emplace<vram_executor_t>();
    std::size_t const count = vram.devices_count ? vram.devices_count : 1;
    for (std::size_t index = 0; index != count; ++index) {
        i32_t const listed = vram.devices_count ? vram.devices[index] : 0;
        if (!open_device(devices.devices, listed, cuda_devices, rocm_devices, vram.gpu_class, arguments->scratch_limit,
                         outcome))
            return;
    }
    for (vram_device_t const &device : devices.devices)
        if (gpu_class_of(device.capabilities) != gpu_class_of(devices.devices[0].capabilities))
            return propagate_error(uform_status_config_mismatch_k, outcome, "The devices differ in class"), void();
    if (fu::failed(devices.decoders.spawn(derived_threads(topology.logical_cores_count()))))
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to spawn the decoders"), void();
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k)
        return propagate_error(uform_status_device_k, outcome, "The CPU capabilities could not be read"), void();
    cpu_class_t const cpu_class = pick_cpu_class(cpu_ladder_t {}, uform_cpu_class_unknown_k, available);
    hand_out_executor(std::move(executor), cpu_class, arguments->deferred_limit, arguments->executor, outcome);
}

UFORM_EXPORT void uform_executor_drain(uform_executor_drain_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->executor) // runs every call to completion before returning, so nothing waits behind it
        return propagate_error(uform_status_success_k, outcome, nullptr), void();
    uform_executor_handle_t &executor = *arguments->executor;
    u64_t last = 0;
    {
        st::unique_lock<fu::spin_mutex_t> lock(deferred_registry.guard);
        last = deferred_registry.tickets;
    }
    run_held(&executor, last);
    wait_running(executor, last);
    if (vram_executor_t *const devices = std::get_if<vram_executor_t>(&executor.executor))
        for (vram_device_t &device : devices->devices)
            if (nk_status_t const status = nk_stream_synchronize_best(device.capabilities, device.stream.get());
                status != nk_success_k)
                return propagate_error(status_of(status), outcome, "A device failed the work queued on it"), void();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_executor_free(uform_executor_t executor) {
    if (!executor) return;
    finish_calls_using(executor);
    delist_executor(*executor);
    delete executor;
}

UFORM_EXPORT void uform_arena_init(uform_arena_init_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->arena = nullptr;
    std::unique_ptr<uform_arena_handle_t> arena {new (std::nothrow) uform_arena_handle_t {}};
    if (!arena) return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the arena"), void();
    if (arguments->executor) // a VRAM executor lands outputs on its first device's stream, any other on the host
        if (vram_executor_t *const devices = std::get_if<vram_executor_t>(&arguments->executor->executor)) {
            vram_device_t &first = devices->devices[0];
            nk_allocator_t policy {};
            if (nk_status_t const status = nk_allocator_init_device_best(&policy, first.capabilities);
                status != nk_success_k)
                return propagate_error(status_of(status), outcome, "The GPU has no device allocator"), void();
            arena->memory.emplace<device_arena_t>(chunked_arena_t(policy, first.stream.get(), 0));
            arena->executor = arguments->executor;
        }
    *arguments->arena = arena.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_arena_free(uform_arena_t arena) {
    finish_calls_using(arena);
    delete arena;
}

#pragma endregion Outcomes Executors and Arenas

#pragma region Names

/** Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
static bool named(uform_bytes_view_t name, sz::string_view_t spelling) noexcept { return text_of(name) == spelling; }

UFORM_EXPORT char const *uform_status_name(uform_status_t status) {
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
    case uform_status_config_mismatch_k: return "config-mismatch";
    case uform_status_contended_k: return "contended";
    case uform_status_cancelled_k: return "cancelled";
    case uform_status_unsupported_k: return "unsupported";
    case uform_status_invalid_argument_k: return "invalid-argument";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_status_from_name(uform_bytes_view_t name) {
    for (uform_status_t const status :
         {uform_status_success_k, uform_status_deferred_k, uform_status_out_of_memory_k, uform_status_shape_mismatch_k,
          uform_status_unsupported_layout_k, uform_status_unknown_kind_k, uform_status_io_k, uform_status_device_k,
          uform_status_malformed_k, uform_status_internal_k, uform_status_config_mismatch_k, uform_status_contended_k,
          uform_status_cancelled_k, uform_status_unsupported_k, uform_status_invalid_argument_k})
        if (named(name, uform_status_name(status))) return status;
    return uform_status_invalid_argument_k;
}

UFORM_EXPORT char const *uform_scalar_name(uform_scalar_t scalar) {
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

UFORM_EXPORT uform_scalar_t uform_scalar_from_name(uform_bytes_view_t name) {
    for (uform_scalar_t const scalar :
         {uform_scalar_f64_k, uform_scalar_f32_k, uform_scalar_bf16_k, uform_scalar_f16_k, uform_scalar_e5m2_k,
          uform_scalar_e4m3_k, uform_scalar_e3m2_k, uform_scalar_e2m3_k, uform_scalar_i8_k, uform_scalar_i4x2_k,
          uform_scalar_i32_k, uform_scalar_u8_k, uform_scalar_u4x2_k, uform_scalar_u1x8_k})
        if (named(name, uform_scalar_name(scalar))) return scalar;
    return uform_scalar_unknown_k;
}

UFORM_EXPORT char const *uform_mood_name(uform_mood_t mood) {
    switch (mood) {
    case uform_mood_parked_k: return "parked";
    case uform_mood_spinning_k: return "spinning";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_mood_t uform_mood_from_name(uform_bytes_view_t name) {
    return named(name, "spinning") ? uform_mood_spinning_k : uform_mood_parked_k;
}

UFORM_EXPORT char const *uform_cpu_class_name(uform_cpu_class_t cpu_class) {
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
    case uform_cpu_class_wasm32_v128_k: return "wasm32-v128";
    case uform_cpu_class_wasm64_extended_k: return "wasm64-extended";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_cpu_class_t uform_cpu_class_from_name(uform_bytes_view_t name) {
    for (uform_cpu_class_t const cpu_class :
         {uform_cpu_class_serial_k, uform_cpu_class_intel_haswell_k, uform_cpu_class_intel_sierra_k,
          uform_cpu_class_intel_skylake_k, uform_cpu_class_intel_icelake_k, uform_cpu_class_amd_genoa_k,
          uform_cpu_class_intel_sapphire_k, uform_cpu_class_intel_granite_k, uform_cpu_class_intel_diamond_k,
          uform_cpu_class_aws_g2_k, uform_cpu_class_apple_m2_k, uform_cpu_class_apple_m4_k, uform_cpu_class_aws_g5_k,
          uform_cpu_class_nvidia_vera_k, uform_cpu_class_alibaba_c930_k, uform_cpu_class_wasm32_v128_k,
          uform_cpu_class_wasm64_extended_k})
        if (named(name, uform_cpu_class_name(cpu_class))) return cpu_class;
    return uform_cpu_class_unknown_k;
}

UFORM_EXPORT char const *uform_gpu_class_name(uform_gpu_class_t gpu_class) {
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

UFORM_EXPORT uform_gpu_class_t uform_gpu_class_from_name(uform_bytes_view_t name) {
    for (uform_gpu_class_t const gpu_class :
         {uform_gpu_class_nvidia_ampere_k, uform_gpu_class_nvidia_ada_k, uform_gpu_class_nvidia_hopper_k,
          uform_gpu_class_nvidia_blackwell_k, uform_gpu_class_nvidia_blackwellrtx_k, uform_gpu_class_nvidia_rubin_k,
          uform_gpu_class_amd_cdna3_k, uform_gpu_class_amd_cdna4_k, uform_gpu_class_amd_cdna5_k,
          uform_gpu_class_apple_metal7_k, uform_gpu_class_apple_metal9_k, uform_gpu_class_apple_metal10_k})
        if (named(name, uform_gpu_class_name(gpu_class))) return gpu_class;
    return uform_gpu_class_unknown_k;
}

UFORM_EXPORT char const *uform_modality_name(uform_modality_t modality) {
    switch (modality) {
    case uform_modality_text_k: return "text";
    case uform_modality_image_k: return "image";
    case uform_modality_video_k: return "video";
    case uform_modality_audio_k: return "audio";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_modality_t uform_modality_from_name(uform_bytes_view_t name) {
    for (uform_modality_t const modality : {uform_modality_image_k, uform_modality_video_k, uform_modality_audio_k})
        if (named(name, uform_modality_name(modality))) return modality;
    return uform_modality_text_k;
}

UFORM_EXPORT char const *uform_objective_name(uform_objective_t objective) {
    switch (objective) {
    case uform_objective_masked_k: return "masked";
    case uform_objective_contrastive_k: return "contrastive";
    case uform_objective_joint_k: return "joint";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_objective_t uform_objective_from_name(uform_bytes_view_t name) {
    for (uform_objective_t const objective : {uform_objective_contrastive_k, uform_objective_joint_k})
        if (named(name, uform_objective_name(objective))) return objective;
    return uform_objective_masked_k;
}

UFORM_EXPORT char const *uform_negatives_name(uform_negatives_t negatives) {
    switch (negatives) {
    case uform_negatives_ambient_k: return "ambient";
    case uform_negatives_named_k: return "named";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_negatives_t uform_negatives_from_name(uform_bytes_view_t name) {
    return named(name, "named") ? uform_negatives_named_k : uform_negatives_ambient_k;
}

UFORM_EXPORT char const *uform_multimem_tier_name(uform_multimem_tier_t tier) {
    switch (tier) {
    case uform_multimem_tier_switch_reduced_k: return "switch-reduced";
    case uform_multimem_tier_peer_read_k: return "peer-read";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_multimem_tier_t uform_multimem_tier_from_name(uform_bytes_view_t name) {
    return named(name, "peer-read") ? uform_multimem_tier_peer_read_k : uform_multimem_tier_switch_reduced_k;
}

#pragma endregion Names

#pragma region Probes

UFORM_EXPORT void uform_version(int *major, int *minor, int *patch) {
    *major = UFORM_VERSION_MAJOR, *minor = UFORM_VERSION_MINOR, *patch = UFORM_VERSION_PATCH;
}

/** The release tooling rewrites the literal in place, pre-release suffix included, as it does the macros. */
UFORM_EXPORT char const *uform_version_name(void) { return "4.0.0-rc.1"; }

UFORM_EXPORT char const *uform_capabilities_compiled(void) {
    static char const *const rendered = [] {
        static char text[NUMKONG_CAPABILITIES_NAME_CAPACITY] = {};
        nk_capability_t cpu = 0, cuda = 0, rocm = 0;
        [[maybe_unused]] nk_status_t const cpu_status = nk_cpu_capabilities_compiled(&cpu);
        [[maybe_unused]] nk_status_t const cuda_status = nk_cuda_capabilities_compiled(&cuda);
        [[maybe_unused]] nk_status_t const rocm_status = nk_rocm_capabilities_compiled(&rocm);
        nk_capabilities_name(cpu | cuda | rocm, text, sizeof(text));
        return text;
    }();
    return rendered;
}

UFORM_EXPORT char const *uform_capabilities_available(void) {
    static char const *const rendered = [] {
        static char text[NUMKONG_CAPABILITIES_NAME_CAPACITY] = {};
        nk_capability_t available = 0;
        if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = 0;
        nk_size_t cuda_devices = 0, rocm_devices = 0;
        if (nk_cuda_count_devices(&cuda_devices) != nk_success_k) cuda_devices = 0;
        if (nk_rocm_count_devices(&rocm_devices) != nk_success_k) rocm_devices = 0;
        for (nk_size_t ordinal = 0; ordinal != cuda_devices; ++ordinal) {
            nk_capability_t device = 0;
            if (nk_cuda_capabilities_enabled(ordinal, &device) == nk_success_k) available |= device;
        }
        for (nk_size_t ordinal = 0; ordinal != rocm_devices; ++ordinal) {
            nk_capability_t device = 0;
            if (nk_rocm_capabilities_enabled(ordinal, &device) == nk_success_k) available |= device;
        }
        nk_capabilities_name(available, text, sizeof(text));
        return text;
    }();
    return rendered;
}

#pragma endregion Probes

} // extern "C"
