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
#include <cstring> // `std::memcpy` - a listed device's name

#include <memory> // `std::unique_ptr` - a handle under construction
#include <new>    // `std::nothrow`

#include <forkunion/topology.hpp> // `fu::current_thread_id` - the thread running an executor's call

#include "encoder_punned.hpp" // `entries_kit_of`, `vendor_of` - a listed device's description

namespace ue = unum::uform::encoder;
using namespace unum::uform;

#pragma region Executor Calls

/** Holds @p arena for the call whose published outcome is @p call, answering whether it was free or
 *  already that call's; a null arena holds nothing, left for the verb to refuse. */
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

/** The pool a call of @p executor forks on: the RAM threads, or a VRAM executor's decoders. */
static fu_pool_t pool_of(uform_executor_handle_t &executor) noexcept {
    if (ram_executor_t *const ram = std::get_if<ram_executor_t>(&executor.executor)) return view_of(ram->pool).pool();
    return std::get<vram_executor_t>(executor.executor).decoders.view().pool();
}

bool uf::take_call_claim(uform_executor_handle_t &executor) noexcept {
    fu::standard_atomic_ref<call_claim_t> claim {executor.holding_call.claim};
    if (claim.exchange(call_claim_t::held_k, std::memory_order_acquire) != call_claim_t::free_k) return false;
    // The last call's thread 0 may still be leaving the slice it ended that call in
    if (delegates(executor)) fu_pool_unsafe_join(pool_of(executor), 0);
    fu::standard_atomic_ref<u64_t> {executor.holding_call.body_thread}.store(fu::current_thread_id(),
                                                                             std::memory_order_relaxed);
    return true;
}

void uf::record_status(uform_executor_t executor, uform_outcome_t const &recorded) noexcept {
    executor->holding_call.recorded = recorded;
}

bool uf::delegates(uform_executor_handle_t &executor) noexcept {
    fu_caller_exclusivity_t exclusivity = fu_caller_inclusive_k;
    return fu_pool_caller_exclusivity(pool_of(executor), &exclusivity) == fu_success_k &&
           exclusivity == fu_caller_exclusive_k;
}

uform_status_t uf::delegate_call(uform_executor_handle_t &executor, void (*body)(void *)) noexcept {
    executor.holding_call.delegated_body = body;
    // The caller is not the body's thread: a drain it issues meanwhile must wait, not refuse
    fu::standard_atomic_ref<u64_t> {executor.holding_call.body_thread}.store(0, std::memory_order_relaxed);
    // Thread 0 runs the whole call in its slice, forking the rest of the pool as it goes
    fu_for_threads_t const run_on_thread_zero = [](fu_lambda_context_t context, std::size_t thread,
                                                   std::size_t) noexcept {
        if (thread != 0) return;
        uform_executor_handle_t &held = *static_cast<uform_executor_handle_t *>(context);
        fu::standard_atomic_ref<u64_t> {held.holding_call.body_thread}.store(fu::current_thread_id(),
                                                                             std::memory_order_relaxed);
        held.holding_call.delegated_body(context);
        end_call(held);
    };
    // The token stays local: thread 0 may end the call before this thread stores it
    fu_generation_t generation = 0;
    fu_status_t const status = fu_pool_unsafe_for_threads(pool_of(executor), run_on_thread_zero, &executor,
                                                          &generation);
    return status == fu_success_k ? uform_status_success_k : uform_status_contended_k;
}

void uf::end_call(uform_executor_handle_t &executor) noexcept {
    holding_call_t &call = executor.holding_call;
    uform_outcome_t *const outcome = call.outcome;
    uform_outcome_t const recorded = call.recorded;
    for (u64_t *&named : call.named_calls)
        if (named) fu::standard_atomic_ref<u64_t> {*named}.fetch_sub(1, std::memory_order_release), named = nullptr;
    call.arguments = nullptr;
    call.delegated_body = nullptr;
    call.delegated_arguments.emplace<std::monostate>();
    // ! Past this store a waiter may free the executor; only the caller's outcome is touched after
    fu::standard_atomic_ref<call_claim_t> {call.claim}.store(call_claim_t::free_k, std::memory_order_release);
    propagate_error(recorded.status, outcome, recorded.message);
}

void uf::wait_named_calls(u64_t &named_calls) noexcept {
    fu::standard_atomic_ref<u64_t> count {named_calls};
    for (u64_t seen = count.load(std::memory_order_acquire); seen; seen = count.load(std::memory_order_acquire))
        // No executor owns a handle
        fu::standard_yield_t {}(&named_calls, seen, std::size_t {0}, fu::wait_capped_k);
}

bool uf::runs_holding_call(uform_executor_handle_t &executor) noexcept {
    holding_call_t &call = executor.holding_call;
    if (fu::standard_atomic_ref<call_claim_t> {call.claim}.load(std::memory_order_acquire) == call_claim_t::free_k)
        return false;
    return fu::standard_atomic_ref<u64_t> {call.body_thread}.load(std::memory_order_relaxed) == fu::current_thread_id();
}

void uf::wait_while_claimed(uform_executor_handle_t &executor) noexcept {
    fu::standard_atomic_ref<call_claim_t> claim {executor.holding_call.claim};
    while (claim.load(std::memory_order_acquire) == call_claim_t::held_k)
        fu::standard_yield_t {}(&executor.holding_call.claim, call_claim_t::held_k, std::size_t {0}, fu::wait_capped_k);
    if (delegates(executor)) fu_pool_unsafe_join(pool_of(executor), 0); // its thread 0 may still be leaving its slice
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

#pragma endregion Executor Calls

#pragma region Executors

/** Threads an executor spawns over @p cores when its caller names none: 7/8 of them, and
 *  at least one. */
static std::size_t derived_threads(std::size_t cores) noexcept {
    std::size_t const threads = cores - cores / idle_cores_denominator_k;
    return threads ? threads : 1;
}

/** Whether the OS numbers a NUMA node of @p topology @p id. */
static bool has_memory_domain(topology_t const &topology, i32_t id) noexcept {
    for (std::size_t index = 0; index != topology.memory_domains_count(); ++index)
        if (topology.memory_domain(index) == id) return true;
    return false;
}

/** Copies the NUMA nodes @p ram names into @p domains, every node of @p topology when it names
 *  none. False once @p outcome records why not. */
static bool adopt_memory_domains(st::basic_vector<i32_t> &domains, uform_ram_options_t const &ram,
                                 topology_t const &topology, uform_outcome_t *outcome) noexcept {
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
static std::size_t cores_of(topology_t const &topology, st::basic_vector<i32_t> const &domains) noexcept {
    std::size_t cores = 0;
    for (std::size_t compute = 0; compute != topology.compute_domains_count(); ++compute)
        for (i32_t const domain : domains)
            if (topology.memory_domain_near(compute) == domain) cores += topology.cores_in(compute);
    return cores;
}

/** Gives @p executor the host verbs of the newest class under @p ceiling and hands it out through
 *  @p output. */
static void hand_out_executor(std::unique_ptr<uform_executor_handle_t> executor, cpu_ceiling_t ceiling,
                              uform_executor_t *output, uform_outcome_t *outcome) noexcept {
    executor->host_verbs = pick_host_verbs(cpu_ladder_t {}, ceiling);
    *output = executor.release();
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** Pools that run deferred calls on their own threads where the build has threads, and include
 *  their caller otherwise, as WebAssembly's single thread does. */
static caller_exclusivity_t spawned_exclusivity() noexcept {
    return (fu_comptime_capabilities() & fu_capability_os_threads_k) ? caller_exclusivity_t::caller_exclusive_k
                                                                     : caller_exclusivity_t::caller_inclusive_k;
}

/** Reads the name, compute units and memory of @p home_ 's device @p vendor_ordinal through its
 *  vendor's entries kit; false for a vendor this build lacks or a runtime that refuses. */
template <encoder_home_t home_>
static bool describe_device(std::size_t vendor_ordinal, ue::device_description_t &description) noexcept {
    if constexpr (!gpu_vendor_compiled(vendor_of(home_))) return false;
    else return entries_kit_of<home_>::describe(vendor_ordinal, description) == nk_success_k;
}

/** The vendor's name in the C interface for a NumKong device kind. */
static uform_gpu_vendor_t gpu_vendor_of(nk::device_kind_t kind) noexcept {
    return kind == nk::device_kind_t::cuda_k ? uform_gpu_vendor_nvidia_k : uform_gpu_vendor_amd_k;
}

/** Fills @p description of @p kind 's device @p vendor_ordinal from its driver and answers
 *  the class its architecture numbers name; unknown for a vendor this build lacks or a
 *  runtime that refuses. */
static gpu_class_t describe_gpu(nk::device_kind_t kind, std::size_t vendor_ordinal,
                                ue::device_description_t &description) noexcept {
    bool const described = kind == nk::device_kind_t::cuda_k
                               ? describe_device<encoder_home_t::vram_nvidia_k>(vendor_ordinal, description)
                               : describe_device<encoder_home_t::vram_amd_k>(vendor_ordinal, description);
    if (!described) return gpu_class_t::unknown_k;
    return gpu_class_of(gpu_vendor_of(kind), description.architecture_major, description.architecture_minor);
}

/** The NumKong kits this build compiled for @p kind 's devices, its vendor's group bit included. */
static nk_capability_t compiled_for(nk::device_kind_t kind) noexcept {
    nk_capability_t compiled = 0;
    nk_status_t const status = kind == nk::device_kind_t::cuda_k ? nk_cuda_capabilities_compiled(&compiled)
                                                                 : nk_rocm_capabilities_compiled(&compiled);
    return status == nk_success_k ? compiled : 0;
}

/** The capabilities a device of @p device_class , from @p kind 's vendor, runs under @p pinned :
 *  its class's, a pinned class's it holds, or the vendor floor for a device no class covers and
 *  nothing pins; only what this build compiled, and zero when the pin refuses it. */
static nk_capability_t device_capabilities(nk::device_kind_t kind, gpu_class_t device_class,
                                           uform_gpu_class_t pinned) noexcept {
    nk_capability_t const vendor = kind == nk::device_kind_t::cuda_k ? nk_cap_cuda_k : nk_cap_rocm_k;
    nk_capability_t const held = device_class == gpu_class_t::unknown_k && pinned == uform_gpu_class_unknown_k
                                     ? vendor
                                     : gpu_pin_capabilities(device_class, pinned);
    return held ? (held | vendor) & compiled_for(kind) : 0;
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
    if (!compiled_for(kind))
        return propagate_error(uform_status_device_k, outcome, "This build holds no kernels for the GPU");
    ue::device_description_t description;
    gpu_class_t const device_class = describe_gpu(kind, vendor_ordinal, description);
    nk_capability_t const capabilities = device_capabilities(kind, device_class, pinned);
    if (!capabilities) return propagate_error(uform_status_unsupported_k, outcome, "The GPU does not admit the pin");
    gpu_class_t const gpu_class = pinned == uform_gpu_class_unknown_k ? device_class : static_cast<gpu_class_t>(pinned);
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
                          gpu_class,
                          std::move(stream.value),
                          device_arena_t {chunked_arena_t(policy, handle, scratch_limit)},
                          chunked_arena_t(pages, handle, 0)};
    if (devices.push_back(std::move(opened)) != status_t::success_k)
        return propagate_error(uform_status_invalid_argument_k, outcome, "More devices than one process drives");
    return true;
}

#pragma endregion Executors

/** Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
static bool named(uform_bytes_view_t name, sz::string_view_t spelling) noexcept { return text_of(name) == spelling; }

/** Writes to @p output the one of @p values that @p spell spells as @p name, or fails with
 *  @c uform_status_invalid_argument_k and writes nothing. */
template <typename value_type_>
static uform_status_t read_name(uform_bytes_view_t name, std::initializer_list<value_type_> values,
                                char const *(*spell)(value_type_), value_type_ *output) noexcept {
    if (!output) return uform_status_invalid_argument_k;
    for (value_type_ const value : values)
        if (named(name, spell(value))) return *output = value, uform_status_success_k;
    return uform_status_invalid_argument_k;
}

extern "C" {

#pragma region Outcomes Executors and Arenas

UFORM_EXPORT uform_status_t uform_outcome_wait(uform_outcome_t *outcome) {
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
            uform_gpu_device_t &entry = arguments->devices[listed];
            entry = {};
            entry.vendor = gpu_vendor_of(kind);
            entry.vendor_ordinal = static_cast<u32_t>(ordinal);
            ue::device_description_t description;
            entry.gpu_class = static_cast<uform_gpu_class_t>(describe_gpu(kind, ordinal, description));
            std::memcpy(entry.name, description.name, sizeof(entry.name));
            entry.compute_units = description.compute_units;
            entry.memory_bytes = description.memory_bytes;
        }
    }
    *arguments->devices_count = listed;
    propagate_error(uform_status_success_k, outcome, nullptr);
}

/** Sets the ceiling @p ram pins, or the machine's, then hands out @p executor , whose pool and
 *  memory domains are set. */
static void hand_out_ram_executor(std::unique_ptr<uform_executor_handle_t> executor, uform_ram_options_t const &ram,
                                  uform_executor_t *output, uform_outcome_t *outcome) noexcept {
    expected<cpu_ceiling_t> ceiling = pick_cpu_ceiling(ram.cpu_class);
    if (!ceiling)
        return propagate_error(uform_status_unsupported_k, outcome, "The CPU does not admit the pinned class"), void();
    std::get<ram_executor_t>(executor->executor).ceiling = *ceiling;
    hand_out_executor(std::move(executor), *ceiling, output, outcome);
}

UFORM_EXPORT void uform_executor_for_ram(uform_executor_for_ram_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    expected<topology_t> topology = topology_t::make();
    if (!topology) return propagate_error(status_of(topology.status_), outcome, "Topology discovery failed"), void();
    std::unique_ptr<uform_executor_handle_t> executor {new (std::nothrow) uform_executor_handle_t {}};
    if (!executor)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    ram_executor_t &ram = std::get<ram_executor_t>(executor->executor);
    if (!adopt_memory_domains(ram.memory_domains, arguments->ram, *topology, outcome)) return;
    std::size_t const threads = arguments->threads ? arguments->threads
                                                   : derived_threads(cores_of(*topology, ram.memory_domains));
    expected<executor_t> pool = executor_t::make(*topology, threads, spawned_exclusivity());
    if (!pool) return propagate_error(status_of(pool.status_), outcome, "Failed to spawn the threads"), void();
    ram.pool = std::move(*pool);
    hand_out_ram_executor(std::move(executor), arguments->ram, arguments->executor, outcome);
}

UFORM_EXPORT void uform_executor_on_pool(uform_executor_on_pool_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    expected<lent_pool_t> pool = lent_pool_t::borrow(static_cast<fu_pool_t>(arguments->pool));
    if (!pool) return propagate_error(status_of(pool.status_), outcome, "The lent pool holds no threads"), void();
    expected<topology_t> topology = topology_t::make();
    if (!topology) return propagate_error(status_of(topology.status_), outcome, "Topology discovery failed"), void();
    std::unique_ptr<uform_executor_handle_t> executor {new (std::nothrow) uform_executor_handle_t {}};
    if (!executor)
        return propagate_error(uform_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    ram_executor_t &ram = std::get<ram_executor_t>(executor->executor);
    if (!adopt_memory_domains(ram.memory_domains, arguments->ram, *topology, outcome)) return;
    ram.pool = std::move(*pool);
    hand_out_ram_executor(std::move(executor), arguments->ram, arguments->executor, outcome);
}

UFORM_EXPORT void uform_executor_for_vram(uform_executor_for_vram_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    uform_vram_options_t const &vram = arguments->vram;
    if (arguments->streams)
        return propagate_error(uform_status_unsupported_k, outcome, "Borrowed streams are not wired yet"), void();
    if (vram.devices_count && !vram.devices)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The devices must not be null"), void();
    expected<cpu_ceiling_t> ceiling = pick_cpu_ceiling(arguments->ram.cpu_class);
    if (!ceiling)
        return propagate_error(uform_status_unsupported_k, outcome, "The CPU does not admit the pinned class"), void();
    nk_size_t cuda_devices = 0, rocm_devices = 0;
    if (nk_cuda_count_devices(&cuda_devices) != nk_success_k) cuda_devices = 0;
    if (nk_rocm_count_devices(&rocm_devices) != nk_success_k) rocm_devices = 0;
    if (!cuda_devices && !rocm_devices)
        return propagate_error(uform_status_device_k, outcome, "No GPU this build drives answers"), void();
    expected<topology_t> topology = topology_t::make();
    if (!topology) return propagate_error(status_of(topology.status_), outcome, "Topology discovery failed"), void();
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
        if (device.gpu_class != devices.devices[0].gpu_class)
            return propagate_error(uform_status_config_mismatch_k, outcome, "The devices differ in class"), void();
    st::basic_vector<i32_t> memory_domains;
    if (!adopt_memory_domains(memory_domains, arguments->ram, *topology, outcome)) return;
    // The caller is decoder 0: it stages, feeds the devices and waits, so calls finish on return
    expected<executor_t> decoders = executor_t::make(*topology, derived_threads(cores_of(*topology, memory_domains)),
                                                     caller_exclusivity_t::caller_inclusive_k);
    if (!decoders) return propagate_error(status_of(decoders.status_), outcome, "Failed to spawn the decoders"), void();
    devices.decoders = std::move(*decoders);
    hand_out_executor(std::move(executor), *ceiling, arguments->executor, outcome);
}

UFORM_EXPORT void uform_executor_drain(uform_executor_drain_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->executor) // runs every call to completion before returning, so nothing waits behind it
        return propagate_error(uform_status_success_k, outcome, nullptr), void();
    uform_executor_handle_t &executor = *arguments->executor;
    if (arguments->mood != uform_mood_parked_k && arguments->mood != uform_mood_spinning_k)
        return propagate_error(uform_status_invalid_argument_k, outcome, "The mood names no value"), void();
    if (runs_holding_call(executor))
        return propagate_error(uform_status_contended_k, outcome, "A callback cannot drain the executor running it"),
               void();
    wait_while_claimed(executor);
    if (vram_executor_t *const devices = std::get_if<vram_executor_t>(&executor.executor))
        for (vram_device_t &device : devices->devices)
            if (nk_status_t const status = nk_stream_synchronize_best(device.capabilities, device.stream.get());
                status != nk_success_k)
                return propagate_error(status_of(status), outcome, "A device failed the work queued on it"), void();
    // A parked pool sleeps until its next dispatch, while a lent one keeps its lender's mood
    if (arguments->mood == uform_mood_parked_k) {
        if (ram_executor_t *const ram = std::get_if<ram_executor_t>(&executor.executor)) {
            if (executor_t *const owned = std::get_if<executor_t>(&ram->pool)) fu_pool_sleep(owned->view().pool(), 0);
        }
        else fu_pool_sleep(std::get<vram_executor_t>(executor.executor).decoders.view().pool(), 0);
    }
    propagate_error(uform_status_success_k, outcome, nullptr);
}

UFORM_EXPORT void uform_executor_free(uform_executor_t executor) {
    if (!executor) return;
    wait_while_claimed(*executor);
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
    if (!arena) return;
    wait_named_calls(arena->named_calls);
    delete arena;
}

UFORM_EXPORT void uform_arena_describe(uform_arena_describe_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->arena || !arguments->description)
        return propagate_error(uform_status_invalid_argument_k, outcome,
                               "The arena and its description must not be null"),
               void();
    *arguments->description = {uform_residency_ram_k, -1};
    if (std::holds_alternative<device_arena_t>(arguments->arena->memory))
        *arguments->description = {uform_residency_vram_k,
                                   std::get<vram_executor_t>(arguments->arena->executor->executor).devices[0].ordinal};
    propagate_error(uform_status_success_k, outcome, nullptr);
}

#pragma endregion Outcomes Executors and Arenas

#pragma region Names

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

UFORM_EXPORT uform_status_t uform_status_from_name(uform_bytes_view_t name, uform_status_t *status) {
    return read_name<uform_status_t>(
        name,
        {uform_status_success_k, uform_status_deferred_k, uform_status_out_of_memory_k, uform_status_shape_mismatch_k,
         uform_status_unsupported_layout_k, uform_status_unknown_kind_k, uform_status_io_k, uform_status_device_k,
         uform_status_malformed_k, uform_status_internal_k, uform_status_config_mismatch_k, uform_status_contended_k,
         uform_status_cancelled_k, uform_status_unsupported_k, uform_status_invalid_argument_k},
        uform_status_name, status);
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

UFORM_EXPORT uform_status_t uform_scalar_from_name(uform_bytes_view_t name, uform_scalar_t *scalar) {
    return read_name<uform_scalar_t>(
        name,
        {uform_scalar_f64_k, uform_scalar_f32_k, uform_scalar_bf16_k, uform_scalar_f16_k, uform_scalar_e5m2_k,
         uform_scalar_e4m3_k, uform_scalar_e3m2_k, uform_scalar_e2m3_k, uform_scalar_i8_k, uform_scalar_i4x2_k,
         uform_scalar_i32_k, uform_scalar_u8_k, uform_scalar_u4x2_k, uform_scalar_u1x8_k},
        uform_scalar_name, scalar);
}

UFORM_EXPORT char const *uform_mood_name(uform_mood_t mood) {
    switch (mood) {
    case uform_mood_parked_k: return "parked";
    case uform_mood_spinning_k: return "spinning";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_mood_from_name(uform_bytes_view_t name, uform_mood_t *mood) {
    return read_name<uform_mood_t>(name, {uform_mood_parked_k, uform_mood_spinning_k}, uform_mood_name, mood);
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
    case uform_cpu_class_monoversioned_k: return "monoversioned";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_cpu_class_from_name(uform_bytes_view_t name, uform_cpu_class_t *cpu_class) {
    return read_name<uform_cpu_class_t>(
        name,
        {uform_cpu_class_serial_k, uform_cpu_class_intel_haswell_k, uform_cpu_class_intel_sierra_k,
         uform_cpu_class_intel_skylake_k, uform_cpu_class_intel_icelake_k, uform_cpu_class_amd_genoa_k,
         uform_cpu_class_intel_sapphire_k, uform_cpu_class_intel_granite_k, uform_cpu_class_intel_diamond_k,
         uform_cpu_class_aws_g2_k, uform_cpu_class_apple_m2_k, uform_cpu_class_apple_m4_k, uform_cpu_class_aws_g5_k,
         uform_cpu_class_nvidia_vera_k, uform_cpu_class_alibaba_c930_k, uform_cpu_class_wasm32_v128_k,
         uform_cpu_class_wasm64_extended_k, uform_cpu_class_monoversioned_k},
        uform_cpu_class_name, cpu_class);
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

UFORM_EXPORT uform_status_t uform_gpu_class_from_name(uform_bytes_view_t name, uform_gpu_class_t *gpu_class) {
    return read_name<uform_gpu_class_t>(
        name,
        {uform_gpu_class_nvidia_ampere_k, uform_gpu_class_nvidia_ada_k, uform_gpu_class_nvidia_hopper_k,
         uform_gpu_class_nvidia_blackwell_k, uform_gpu_class_nvidia_blackwellrtx_k, uform_gpu_class_nvidia_rubin_k,
         uform_gpu_class_amd_cdna3_k, uform_gpu_class_amd_cdna4_k, uform_gpu_class_amd_cdna5_k,
         uform_gpu_class_apple_metal7_k, uform_gpu_class_apple_metal9_k, uform_gpu_class_apple_metal10_k},
        uform_gpu_class_name, gpu_class);
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

UFORM_EXPORT uform_status_t uform_modality_from_name(uform_bytes_view_t name, uform_modality_t *modality) {
    return read_name<uform_modality_t>(
        name, {uform_modality_text_k, uform_modality_image_k, uform_modality_video_k, uform_modality_audio_k},
        uform_modality_name, modality);
}

UFORM_EXPORT char const *uform_objective_name(uform_objective_t objective) {
    switch (objective) {
    case uform_objective_masked_k: return "masked";
    case uform_objective_contrastive_k: return "contrastive";
    case uform_objective_joint_k: return "joint";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_objective_from_name(uform_bytes_view_t name, uform_objective_t *objective) {
    return read_name<uform_objective_t>(
        name, {uform_objective_masked_k, uform_objective_contrastive_k, uform_objective_joint_k}, uform_objective_name,
        objective);
}

UFORM_EXPORT char const *uform_negatives_name(uform_negatives_t negatives) {
    switch (negatives) {
    case uform_negatives_ambient_k: return "ambient";
    case uform_negatives_named_k: return "named";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_negatives_from_name(uform_bytes_view_t name, uform_negatives_t *negatives) {
    return read_name<uform_negatives_t>(name, {uform_negatives_ambient_k, uform_negatives_named_k},
                                        uform_negatives_name, negatives);
}

UFORM_EXPORT char const *uform_multimem_tier_name(uform_multimem_tier_t tier) {
    switch (tier) {
    case uform_multimem_tier_switch_reduced_k: return "switch-reduced";
    case uform_multimem_tier_peer_read_k: return "peer-read";
    }
    return "unrecognized";
}

UFORM_EXPORT uform_status_t uform_multimem_tier_from_name(uform_bytes_view_t name, uform_multimem_tier_t *tier) {
    return read_name<uform_multimem_tier_t>(
        name, {uform_multimem_tier_switch_reduced_k, uform_multimem_tier_peer_read_k}, uform_multimem_tier_name, tier);
}

#pragma endregion Names

#pragma region Probes

UFORM_EXPORT void uform_version(int *major, int *minor, int *patch) {
    *major = UFORM_VERSION_MAJOR, *minor = UFORM_VERSION_MINOR, *patch = UFORM_VERSION_PATCH;
}

/** The release tooling rewrites the literal in place, pre-release suffix included, as it
 *  does the macros. */
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
        expected<cpu_ceiling_t> const ceiling = pick_cpu_ceiling(uform_cpu_class_unknown_k);
        nk_capability_t available = ceiling ? ceiling->compute : nk_cap_serial_k;
        for (nk::device_kind_t const kind : {nk::device_kind_t::cuda_k, nk::device_kind_t::rocm_k}) {
            nk::expected<std::size_t> const count = nk::device_t::count(kind);
            for (std::size_t ordinal = 0; count && ordinal != count.value; ++ordinal) {
                ue::device_description_t description;
                gpu_class_t const device_class = describe_gpu(kind, ordinal, description);
                available |= device_capabilities(kind, device_class, uform_gpu_class_unknown_k);
            }
        }
        nk_capabilities_name(available, text, sizeof(text));
        return text;
    }();
    return rendered;
}

#pragma endregion Probes

} // extern "C"
