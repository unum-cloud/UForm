/**
 *  @file include/uform/utilities/vram_amd_device.hpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief The HIP runtime's entry points, resolved into one table from a library opened at first
 *      use, and the stream, event and module owners every AMD executor queues through.
 *
 *  The inference library never links @c libamdhip64 , so it loads on a machine without ROCm:
 *  @ref unum::uform::utilities::hip_api_t::load opens it, and a machine without it keeps an empty
 *  table. Plain C++ needing no SDK: the types, constants and prototypes the table calls are spelled
 *  here, copied from the runtime's stable ABI, so any host compiler builds the launches. Every
 *  sibling engine carries the @c Shared @c VRAM regions word for word, which @c selftest checks.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - the runtime's default stream handles

#include <memory>  // `std::unique_ptr` - an executor's stream records
#include <span>    // `std::span` - the devices an executor opens over, a code object's bytes
#include <utility> // `std::exchange`, `std::swap`, `std::move`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the executor's claim word

#include "uform/utilities/blocking_files.hpp" // `shared_library_t` - the runtime opened at first use
#include "uform/utilities/ram_device.hpp"     // `memory_domain_of_pci` - pinned placement
#include "uform/utilities/types.hpp"          // `status_t`, `byte_t`, `gpu_vendor_t`, `gpu_devices_capacity_k`

namespace unum::uform::utilities {

#pragma region Runtime ABI

/** The runtime's status word, @c hipError_t , naming the values UForm tells apart; any other
 *  is a failure. */
enum hip_result_t : int {
    hip_success_k = 0,
    hip_error_not_initialized_k = 3,
    hip_error_invalid_device_k = 101,
    hip_error_no_binary_for_gpu_k = 209,
    hip_error_not_ready_k = 600,
};

/** The runtime's opaque handles: @c hipStream_t , @c hipEvent_t , @c hipModule_t and
 *  @c hipFunction_t . */
using hip_stream_handle_t = struct hip_stream_opaque_t*;
using hip_event_handle_t = struct hip_event_opaque_t*;
using hip_module_handle_t = struct hip_module_opaque_t*;
using hip_function_handle_t = struct hip_function_opaque_t*;

/** The @c hipMemcpyKind values; UForm copies with the default, which unified addressing
 *  resolves. */
enum hip_copy_kind_t : int {
    hip_copy_host_to_device_k = 1,
    hip_copy_device_to_host_k = 2,
    hip_copy_device_to_device_k = 3,
    hip_copy_default_k = 4,
};

/** The @c hipDeviceAttribute_t values UForm queries; on AMD the compute capability is the @c gfx
 *  processor's major and minor version. */
enum hip_device_attribute_t : int {
    hip_attribute_compute_capability_major_k = 23,
    hip_attribute_compute_capability_minor_k = 61,
    hip_attribute_multiprocessor_count_k = 63,
    hip_attribute_warp_size_k = 87,
};

/** The @c hipMemoryAdvise values the table's managed memory takes. */
enum hip_memory_advice_t : int {
    hip_advise_preferred_location_k = 3,
    hip_advise_accessed_by_k = 5,
};

/** The flags UForm passes: @c hipStreamNonBlocking , @c hipEventBlockingSync ,
 *  @c hipEventDisableTiming and @c hipHostMallocPortable . */
inline constexpr unsigned hip_stream_non_blocking_k = 1;
inline constexpr unsigned hip_event_blocking_sync_k = 1;
inline constexpr unsigned hip_event_disable_timing_k = 2;
inline constexpr unsigned hip_host_portable_k = 1;

/** The runtime's legacy and per-thread default streams, which no executor adopts. */
inline hip_stream_handle_t const hip_stream_legacy_handle_k = reinterpret_cast<hip_stream_handle_t>(
    std::uintptr_t {0x1});
inline hip_stream_handle_t const hip_stream_per_thread_handle_k = reinterpret_cast<hip_stream_handle_t>(
    std::uintptr_t {0x2});

#pragma endregion Runtime ABI

#pragma region Shared VRAM Driver

/** What the runtime calls in stream order once the work queued before it finished, with that
 *  work's status and the pointer given beside it; it must make no runtime call. */
using hip_stream_callback_t = void (*)(hip_stream_handle_t, hip_result_t, void*);

/**
 *  @brief The HIP entry points every engine calls, resolved from @c libamdhip64 once per process.
 *
 *  Every entry is null when the library is absent or lacks any of them, which leaves AMD with no
 *  devices; each spells the C prototype of the exported symbol it is resolved by. An engine's
 *  optional entries resolve apart from these, so a runtime lacking one keeps this table whole.
 */
struct hip_api_t {
    hip_result_t (*get_device_count)(int*) = nullptr;
    hip_result_t (*get_device)(int*, int) = nullptr;
    hip_result_t (*get_device_attribute)(int*, hip_device_attribute_t, int) = nullptr;
    hip_result_t (*get_device_name)(char*, int, int) = nullptr;
    hip_result_t (*get_pci_bus_id)(char*, int, int) = nullptr;
    hip_result_t (*get_device_memory)(std::size_t*, int) = nullptr;
    hip_result_t (*can_access_peer)(int*, int, int) = nullptr;
    hip_result_t (*set_device)(int) = nullptr;
    hip_result_t (*pointer_attribute)(void*, int, void const*) = nullptr;
    hip_result_t (*malloc_async)(void**, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*free_async)(void*, hip_stream_handle_t) = nullptr;
    hip_result_t (*malloc_managed)(void**, std::size_t, unsigned) = nullptr;
    hip_result_t (*malloc)(void**, std::size_t) = nullptr;
    hip_result_t (*memory_advise)(void const*, std::size_t, hip_memory_advice_t, int) = nullptr;
    hip_result_t (*free)(void*) = nullptr;
    hip_result_t (*host_malloc)(void**, std::size_t, unsigned) = nullptr;
    hip_result_t (*host_free)(void*) = nullptr;
    hip_result_t (*host_register)(void*, std::size_t, unsigned) = nullptr;
    hip_result_t (*host_unregister)(void*) = nullptr;
    hip_result_t (*copy)(void*, void const*, std::size_t, hip_copy_kind_t) = nullptr;
    hip_result_t (*copy_async)(void*, void const*, std::size_t, hip_copy_kind_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*copy_2d_async)(void*, std::size_t, void const*, std::size_t, std::size_t, std::size_t,
                                  hip_copy_kind_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*fill)(void*, int, std::size_t) = nullptr;
    hip_result_t (*fill_async)(void*, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*fill_words_async)(void*, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*stream_create_with_flags)(hip_stream_handle_t*, unsigned) = nullptr;
    hip_result_t (*stream_destroy)(hip_stream_handle_t) = nullptr;
    hip_result_t (*stream_synchronize)(hip_stream_handle_t) = nullptr;
    hip_result_t (*stream_query)(hip_stream_handle_t) = nullptr;
    hip_result_t (*stream_get_device)(hip_stream_handle_t, int*) = nullptr;
    hip_result_t (*stream_wait_event)(hip_stream_handle_t, hip_event_handle_t, unsigned) = nullptr;
    hip_result_t (*stream_add_callback)(hip_stream_handle_t, hip_stream_callback_t, void*, unsigned) = nullptr;
    hip_result_t (*event_create_with_flags)(hip_event_handle_t*, unsigned) = nullptr;
    hip_result_t (*event_record)(hip_event_handle_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*event_query)(hip_event_handle_t) = nullptr;
    hip_result_t (*event_synchronize)(hip_event_handle_t) = nullptr;
    hip_result_t (*event_destroy)(hip_event_handle_t) = nullptr;
    hip_result_t (*module_load_data)(hip_module_handle_t*, void const*) = nullptr;
    hip_result_t (*module_unload)(hip_module_handle_t) = nullptr;
    hip_result_t (*module_get_function)(hip_function_handle_t*, hip_module_handle_t,
                                        char const*) = nullptr; // vendor-boundary
    hip_result_t (*function_get_attribute)(int*, int, hip_function_handle_t) = nullptr;
    hip_result_t (*function_set_attribute)(void const*, int, int) = nullptr;
    hip_result_t (*module_launch_kernel)(hip_function_handle_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                         unsigned, unsigned, hip_stream_handle_t, void**, void**) = nullptr;

    /** The memory domain nearest each device, where its pinned host bytes go, or
     *  @c no_memory_domain_k where none is known. */
    fu_memory_domain_id_t memory_domains[gpu_devices_capacity_k] = {};

    /** Whether the runtime loaded with every entry resolved. */
    explicit operator bool() const noexcept { return module_launch_kernel != nullptr; }

    /** Opens the runtime and resolves every entry, or answers the empty table. The library stays
     *  open for the life of the process, as the table points into it. */
    static hip_api_t load() noexcept {
        // Each system names the runtime its own way, and opening another's name merely fails.
        shared_library_t library;
        for (char const* name : // vendor-boundary
             {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "amdhip64_7.dll", "amdhip64_6.dll"})
            if ((library = shared_library_t::open(name))) break;
        if (!library) return {};
        hip_api_t runtime;
        bool const resolved = //
            library.resolve("hipGetDeviceCount", runtime.get_device_count) &&
            library.resolve("hipDeviceGet", runtime.get_device) &&
            library.resolve("hipDeviceGetAttribute", runtime.get_device_attribute) &&
            library.resolve("hipDeviceGetName", runtime.get_device_name) &&
            library.resolve("hipDeviceGetPCIBusId", runtime.get_pci_bus_id) &&
            library.resolve("hipDeviceTotalMem", runtime.get_device_memory) &&
            library.resolve("hipDeviceCanAccessPeer", runtime.can_access_peer) &&
            library.resolve("hipSetDevice", runtime.set_device) &&
            library.resolve("hipPointerGetAttribute", runtime.pointer_attribute) &&
            library.resolve("hipMallocAsync", runtime.malloc_async) &&
            library.resolve("hipFreeAsync", runtime.free_async) &&
            library.resolve("hipMallocManaged", runtime.malloc_managed) &&
            library.resolve("hipMalloc", runtime.malloc) && library.resolve("hipMemAdvise", runtime.memory_advise) &&
            library.resolve("hipFree", runtime.free) && library.resolve("hipHostMalloc", runtime.host_malloc) &&
            library.resolve("hipHostFree", runtime.host_free) &&
            library.resolve("hipHostRegister", runtime.host_register) &&
            library.resolve("hipHostUnregister", runtime.host_unregister) &&
            library.resolve("hipMemcpy", runtime.copy) && library.resolve("hipMemcpyAsync", runtime.copy_async) &&
            library.resolve("hipMemcpy2DAsync", runtime.copy_2d_async) && library.resolve("hipMemset", runtime.fill) &&
            library.resolve("hipMemsetAsync", runtime.fill_async) &&
            library.resolve("hipMemsetD32Async", runtime.fill_words_async) &&
            library.resolve("hipStreamCreateWithFlags", runtime.stream_create_with_flags) &&
            library.resolve("hipStreamDestroy", runtime.stream_destroy) &&
            library.resolve("hipStreamSynchronize", runtime.stream_synchronize) &&
            library.resolve("hipStreamQuery", runtime.stream_query) &&
            library.resolve("hipStreamGetDevice", runtime.stream_get_device) &&
            library.resolve("hipStreamWaitEvent", runtime.stream_wait_event) &&
            library.resolve("hipStreamAddCallback", runtime.stream_add_callback) &&
            library.resolve("hipEventCreateWithFlags", runtime.event_create_with_flags) &&
            library.resolve("hipEventRecord", runtime.event_record) &&
            library.resolve("hipEventQuery", runtime.event_query) &&
            library.resolve("hipEventSynchronize", runtime.event_synchronize) &&
            library.resolve("hipEventDestroy", runtime.event_destroy) &&
            library.resolve("hipModuleLoadData", runtime.module_load_data) &&
            library.resolve("hipModuleUnload", runtime.module_unload) &&
            library.resolve("hipModuleGetFunction", runtime.module_get_function) &&
            library.resolve("hipFuncGetAttribute", runtime.function_get_attribute) &&
            library.resolve("hipFuncSetAttribute", runtime.function_set_attribute) &&
            library.resolve("hipModuleLaunchKernel", runtime.module_launch_kernel);
        if (!resolved) return {};
        // A bus that names no memory domain leaves the device's pinned bytes on the heap.
        for (int device = 0; static_cast<std::size_t>(device) != gpu_devices_capacity_k; ++device) {
            char bus_id[32] = {};
            [[maybe_unused]] hip_result_t const named = runtime.get_pci_bus_id(bus_id, sizeof(bus_id), device);
            runtime.memory_domains[device] = memory_domain_of_pci(bus_id);
        }
        return runtime;
    }
};

/** Makes HIP ordinal @p ordinal current on the calling thread, which every stream, allocation and
 *  launch after it runs on - the twin of @c cuda_set_device. Asked on every entry, so a caller
 *  switching devices between calls is rebound. */
inline hip_result_t hip_set_device(hip_api_t const& api, int ordinal) noexcept {
    if (!api) return hip_error_not_initialized_k;
    return api.set_device(ordinal);
}

/** One device's ordering identity: a non-blocking stream an executor opens or adopts and drains and
 *  destroys when it dies, or the device's null legacy stream, which owns nothing. */
struct hip_stream_t {

    /**
     *  @brief Opens a non-blocking stream on the current device, owned from now on.
     *
     *  @par Statuses:
     *  - @c device_k The runtime refused the stream.
     */
    static expected<hip_stream_t> open(hip_api_t const& api) noexcept {
        hip_stream_t stream;
        stream.api_ = &api;
        if (!api || api.stream_create_with_flags(&stream.handle_, hip_stream_non_blocking_k) != hip_success_k)
            return {{}, {status_code_t::device_k, "Runtime refused a stream"}};
        return {std::move(stream), status_code_t::success_k};
    }

    /**
     *  @brief Checks that @p handle, a caller's stream, may be adopted as the stream of HIP ordinal
     *      @p ordinal, without taking it.
     *
     *  @par Statuses:
     *  - @c device_k There is no runtime.
     *  - @c invalid_argument_k The @p handle is null or a default stream, which no owner destroys.
     *  - @c config_mismatch_k The stream sits on another device than @p ordinal.
     */
    static status_t adoptable(hip_api_t const& api, std::size_t ordinal, void* handle) noexcept {
        if (!api) return {status_code_t::device_k, "No HIP runtime"};
        auto const stream = static_cast<hip_stream_handle_t>(handle);
        if (!stream || stream == hip_stream_legacy_handle_k || stream == hip_stream_per_thread_handle_k)
            return {status_code_t::invalid_argument_k, "A default stream cannot be adopted"};
        int device = -1;
        if (api.stream_get_device(stream, &device) != hip_success_k || device < 0 ||
            static_cast<std::size_t>(device) != ordinal)
            return {status_code_t::config_mismatch_k, "A stream sits on another device"};
        return {};
    }

    /** Adopts @p handle, a stream the caller created, which it synchronizes and destroys. */
    explicit hip_stream_t(hip_api_t const& api, hip_stream_handle_t handle) noexcept : api_(&api), handle_(handle) {}

    hip_stream_t() noexcept = default;
    hip_stream_t(hip_stream_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    hip_stream_t& operator=(hip_stream_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(handle_, other.handle_);
        return *this;
    }
    hip_stream_t(hip_stream_t const&) = delete;
    hip_stream_t& operator=(hip_stream_t const&) = delete;
    ~hip_stream_t() noexcept {
        if (handle_) api_->stream_synchronize(handle_), api_->stream_destroy(handle_);
    }

    /** The runtime table every call on this stream goes through. */
    hip_api_t const& api() const noexcept { return *api_; }

    operator hip_stream_handle_t() const noexcept { return handle_; }

  private:
    hip_api_t const* api_ = nullptr;
    hip_stream_handle_t handle_ = nullptr;
};

/** Owned event, born on the current device, marking a point on a stream that the host polls or
 *  awaits. It blocks on sync, so a drain sleeps instead of spinning a core, and destroying it
 *  never waits for the work it marked. */
struct hip_event_t {

    /** An empty event, which records nothing. */
    hip_event_t() noexcept = default;

    explicit hip_event_t(hip_api_t const& api) noexcept : api_(&api) {
        if (api) api.event_create_with_flags(&handle_, hip_event_blocking_sync_k | hip_event_disable_timing_k);
    }
    hip_event_t(hip_event_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    hip_event_t& operator=(hip_event_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(handle_, other.handle_);
        return *this;
    }
    hip_event_t(hip_event_t const&) = delete;
    hip_event_t& operator=(hip_event_t const&) = delete;
    ~hip_event_t() noexcept {
        if (handle_) api_->event_destroy(handle_);
    }

    operator hip_event_handle_t() const noexcept { return handle_; }

    /**
     *  @brief Marks everything queued on @p stream so far, which @ref finished and @ref wait
     *      then await in order.
     *
     *  @par Statuses:
     *  - @c device_k The event was never created.
     */
    status_t record(hip_stream_handle_t stream) noexcept {
        if (!handle_ || api_->event_record(handle_, stream) != hip_success_k)
            return {status_code_t::device_k, "The event recorded nothing"};
        return {};
    }

    /**
     *  @brief Polls the recorded work without blocking; nothing recorded has nothing to run.
     *
     *  @par Statuses:
     *  - @c deferred_k The recorded work still runs.
     *  - @c device_k The event was never created.
     */
    status_t finished() const noexcept {
        if (!handle_) return {status_code_t::device_k, "The event was never created"};
        hip_result_t const queried = api_->event_query(handle_);
        if (queried == hip_error_not_ready_k) return {status_code_t::deferred_k};
        if (queried != hip_success_k) return {status_code_t::device_k, "The device failed the recorded work"};
        return {};
    }

    /**
     *  @brief Sleeps until the recorded work finishes and answers whether the device finished it.
     *
     *  @par Statuses:
     *  - @c device_k The event was never created.
     */
    status_t wait() noexcept {
        if (!handle_ || api_->event_synchronize(handle_) != hip_success_k)
            return {status_code_t::device_k, "The device failed the recorded work"};
        return {};
    }

  private:
    hip_api_t const* api_ = nullptr;
    hip_event_handle_t handle_ = nullptr;
};

/** One code object loaded onto one GPU and unloaded with this owner; kernels are found by name. */
struct hip_module_t {
    using function_t = hip_function_handle_t;

    hip_module_t() noexcept = default;
    hip_module_t(hip_module_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    hip_module_t& operator=(hip_module_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(handle_, other.handle_);
        return *this;
    }
    hip_module_t(hip_module_t const&) = delete;
    hip_module_t& operator=(hip_module_t const&) = delete;
    ~hip_module_t() noexcept {
        if (handle_) api_->module_unload(handle_);
    }

    /**
     *  @brief Loads @p code_object onto HIP ordinal @p device, binding it first, since a module
     *      loads into the current device; an offload bundle loads its entry for that device.
     *
     *  @par Statuses:
     *  - @c device_k The @p code_object is empty, or the runtime refused it.
     */
    static expected<hip_module_t> load(hip_api_t const& api, int device, bytes_view_t code_object) noexcept {
        hip_module_t module;
        module.api_ = &api;
        if (code_object.empty() || hip_set_device(api, device) != hip_success_k ||
            api.module_load_data(&module.handle_, code_object.data()) != hip_success_k)
            return {{}, {status_code_t::device_k, "The runtime refused a code object"}};
        return {std::move(module), status_code_t::success_k};
    }

    /** The runtime table the code object was loaded through, which launches of its kernels go
     *  through. */
    hip_api_t const& api() const noexcept { return *api_; }

    /** The kernel exported as @p name, or null when the code object has none. */
    function_t function(char const* name) const noexcept { // vendor-boundary
        function_t function = nullptr;
        return handle_ && api_->module_get_function(&function, handle_, name) == hip_success_k ? function : nullptr;
    }

  private:
    hip_api_t const* api_ = nullptr;
    hip_module_handle_t handle_ = nullptr;
};

#pragma endregion Shared VRAM Driver

#pragma region Listing and Scratch

/** Writes every HIP device @p api reports into @p into , up to its size, in the driver's order,
 *  and answers how many it wrote, skipping any device the runtime will not describe, and none for
 *  an empty table. */
inline std::size_t list_amd_gpu_devices(hip_api_t const& api, std::span<gpu_device_t> into) noexcept {
    std::size_t count = 0;
    int devices_count = 0;
    if (!api || api.get_device_count(&devices_count) != hip_success_k) return count;
    for (int ordinal = 0; ordinal != devices_count && count != into.size(); ++ordinal) {
        int device = 0, warp_lanes = 0, compute_units = 0, major = 0, minor = 0;
        std::size_t memory_bytes = 0;
        gpu_device_t listed {gpu_vendor_t::amd_k, static_cast<u32_t>(ordinal)};
        if (api.get_device(&device, ordinal) != hip_success_k ||
            api.get_device_name(listed.name, static_cast<int>(sizeof(listed.name)), device) != hip_success_k ||
            api.get_device_attribute(&warp_lanes, hip_attribute_warp_size_k, device) != hip_success_k ||
            api.get_device_attribute(&compute_units, hip_attribute_multiprocessor_count_k, device) != hip_success_k ||
            api.get_device_attribute(&major, hip_attribute_compute_capability_major_k, device) != hip_success_k ||
            api.get_device_attribute(&minor, hip_attribute_compute_capability_minor_k, device) != hip_success_k ||
            api.get_device_memory(&memory_bytes, device) != hip_success_k)
            continue;
        listed.architecture_major = static_cast<u32_t>(major), listed.architecture_minor = static_cast<u32_t>(minor);
        listed.warp_lanes = static_cast<u32_t>(warp_lanes), listed.compute_units = static_cast<u32_t>(compute_units);
        listed.memory_bytes = memory_bytes;
        into[count++] = listed;
    }
    return count;
}

/** Where a hip stream's launches find their functions: the process's one table of UForm's bundle
 *  per device, which the GPU runtime owns. */
using hip_kernels_t = kernel_tables<hip_module_t>*;

/**
 *  @brief UForm's queue-ordered memory on one HIP stream: the activations a call computes in, on
 *      the device, and the pinned transfers it packs what the device reads into.
 *
 *  Both only hand out what @ref reserve grew them by, with the stream's device bound, before a call
 *  allocates, and @ref release frees them after the stream drained, each chunk through the verb
 *  that made it.
 */
struct hip_scratch_t {

    /** Device memory the call's layers compute in, grown in the stream's order. */
    chunked_arena_t activations;

    /** Pinned host pages both the host and the device address, where a call packs what the device
     *  reads. */
    chunked_arena_t transfers;

    /** The ceiling on @ref activations , zero for none, which every @ref reserve checks. */
    std::size_t limit = 0;

    /** Points @p record at the bundle table of @p runtime its launches find their functions in,
     *  and takes the executor's ceiling on activations, zero for none; the arenas start empty and
     *  grow at the first @ref reserve . */
    template <typename record_type_, typename runtime_type_>
    status_t open(record_type_& record, runtime_type_& runtime, std::size_t& scratch_limit) noexcept {
        record.kernels = &runtime.hip_kernels;
        limit = scratch_limit;
        return {};
    }

    /**
     *  @brief Rewinds both arenas and makes each hold a call's @p activation_bytes and
     *      @p transfer_bytes , as the allocations' @ref chunked_arena_t::footprint values sum.
     *
     *  After the rewind nothing either arena holds is live, so one that falls short frees its
     *  chunks and takes one chunk sized to the call: the activations never hold more than the
     *  @ref limit .
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The activations pass the @ref limit , the device or the host refused a
     *    chunk, or a chunk table could not grow.
     */
    template <typename stream_type_>
    status_t reserve(stream_type_& stream, std::size_t activation_bytes, std::size_t transfer_bytes = 0) noexcept {
        if (limit && activation_bytes > limit)
            return {status_code_t::out_of_memory_k, "The executor's scratch limit is too low"};
        activations.rewind(), transfers.rewind();
        if (activations.room() < activation_bytes) {
            free_activations_(stream);
            auto [address, allocate_status] = stream.allocate_async(activation_bytes);
            if (allocate_status.failed()) return allocate_status;
            if (status_t const appended = activations.append({reinterpret_cast<b8_t*>(address), activation_bytes});
                appended.failed())
                return stream.free_async(address), appended;
        }
        if (transfers.room() < transfer_bytes) {
            free_transfers_(stream);
            void* pages = nullptr;
            if (stream.handle.api().host_malloc(&pages, transfer_bytes, hip_host_portable_k) != hip_success_k)
                return {status_code_t::out_of_memory_k, "The host refused pinned pages"};
            if (status_t const appended = transfers.append({static_cast<b8_t*>(pages), transfer_bytes});
                appended.failed())
                return stream.handle.api().host_free(pages), appended;
        }
        return {};
    }

    /** Frees every chunk of both arenas once @p stream drained, with its device bound. */
    template <typename stream_type_>
    void release(stream_type_& stream) noexcept {
        if (stream.bind().failed()) return activations.forget(), transfers.forget();
        free_activations_(stream), free_transfers_(stream);
    }

  private:
    /** Frees the activations in @p stream 's order, which drained what read them. */
    template <typename stream_type_>
    void free_activations_(stream_type_& stream) noexcept {
        for (bytes_span_t const chunk : activations.chunks())
            stream.free_async(reinterpret_cast<std::uintptr_t>(chunk.data()));
        activations.forget();
    }

    /** Frees the pinned transfers, which the last call's wait left unread. */
    template <typename stream_type_>
    void free_transfers_(stream_type_& stream) noexcept {
        for (bytes_span_t const chunk : transfers.chunks()) stream.handle.api().host_free(chunk.data());
        transfers.forget();
    }
};

#pragma endregion Listing and Scratch

#pragma region Shared VRAM Executor

/**
 *  @brief The streams a call queues on, one per entry of the devices it was spawned over, in the
 *      caller's order, each with the event its waits sleep on, its kernels and the engine's memory.
 *
 *  Work is queued, never waited on, so one host thread drives every device. Each stream record is
 *  built in place on the heap and never moves, so the memory may point at its stream owner; it
 *  dies draining its stream, then releasing its memory with its device bound, then closing its
 *  event and its stream.
 */
struct hip_executor_t {

    /** The engine's queue-ordered memory on one stream. */
    using scratch_t = hip_scratch_t;

    /** One call's hold on the executor: the word it set, which it clears when it dies. Move-only,
     *  so exactly one owner releases the hold. */
    class claim_t {
      public:
        claim_t() noexcept = default;
        explicit claim_t(u32_t& held_word) noexcept : held_word_(&held_word) {}
        claim_t(claim_t&& other) noexcept : held_word_(std::exchange(other.held_word_, nullptr)) {}
        claim_t& operator=(claim_t&& other) noexcept {
            std::swap(held_word_, other.held_word_);
            return *this;
        }
        claim_t(claim_t const&) = delete;
        claim_t& operator=(claim_t const&) = delete;
        ~claim_t() noexcept {
            if (held_word_) fu::standard_atomic_ref<u32_t>(*held_word_).store(0, std::memory_order_release);
        }

      private:
        u32_t* held_word_ = nullptr;
    };

    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::amd_k;

    /** One device's stream: what it queues on, the event its waits sleep on, the kernels it
     *  launches, and the engine's memory, declared last so it dies first. */
    struct stream_t {

        /** The HIP ordinal of the device this stream queues on, as the GPU runtime lists it. */
        std::size_t ordinal = 0;

        /** Opened by the executor, adopted from the caller, or the device's null legacy stream. */
        hip_stream_t handle;

        /** What @ref wait records and sleeps on, so a wait frees its core. */
        hip_event_t event;

        /** The kernels this stream launches, as the engine resolves them when the record opens. */
        hip_kernels_t kernels = {};

        /** The engine's memory, safe to reuse only in this stream's queue order, opened once this
         *  record sits in the executor. */
        scratch_t memory;

        /** Takes @p opened as the stream of device @p device, which is current, as the event is
         *  born there too. */
        stream_t(std::size_t device, hip_stream_t&& opened) noexcept
            : ordinal(device), handle(std::move(opened)), event(handle.api()) {}
        stream_t(stream_t const&) = delete;
        stream_t& operator=(stream_t const&) = delete;

        /** Drains an owned stream, then releases the memory in its order with the device bound;
         *  the event and the stream close after. A record owning none, a placeholder or the
         *  device's legacy stream, has nothing of its own to drain. */
        ~stream_t() noexcept {
            [[maybe_unused]] status_t const ready = handle ? wait() : bind();
            memory.release(*this);
        }

        /**
         *  @brief Makes this stream's device current on the calling thread.
         *
         *  @par Statuses:
         *  - @c device_k The device could not be made current.
         */
        status_t bind() const noexcept {
            if (hip_set_device(handle.api(), static_cast<int>(ordinal)) != hip_success_k)
                return {status_code_t::device_k, "Device selection failed"};
            return {};
        }

        /**
         *  @brief Waits for everything queued on this stream so far, recording the event with the
         *      device bound and sleeping on it.
         *
         *  @par Statuses:
         *  - @c device_k The device could not be bound, or it failed the work.
         */
        status_t wait() noexcept {
            if (status_t const bound = bind(); bound.failed()) return bound;
            if (status_t const recorded = event.record(handle); recorded.failed()) return recorded;
            return event.wait();
        }

        /**
         *  @brief Queues an allocation of @p bytes in this stream's order, its device current,
         *      answering its address.
         *
         *  @par Statuses:
         *  - @c out_of_memory_k The device refused the allocation.
         */
        expected<std::uintptr_t> allocate_async(std::size_t bytes) noexcept {
            void* address = nullptr;
            if (handle.api().malloc_async(&address, bytes, handle) != hip_success_k)
                return {0, {status_code_t::out_of_memory_k, "The device refused an allocation"}};
            return {reinterpret_cast<std::uintptr_t>(address), status_code_t::success_k};
        }

        /** Queues freeing @p address, which @ref allocate_async answered, in this stream's
         *  order. */
        void free_async(std::uintptr_t address) noexcept {
            handle.api().free_async(reinterpret_cast<void*>(address), handle);
        }

        /**
         *  @brief Queues a copy of @p bytes from @p source to @p target, either side the host's
         *      or a device's, as unified addressing tells apart; the device is current.
         *
         *  @par Statuses:
         *  - @c device_k The runtime refused the copy.
         */
        status_t copy_async(std::uintptr_t target, std::uintptr_t source, std::size_t bytes) noexcept {
            if (handle.api().copy_async(reinterpret_cast<void*>(target), reinterpret_cast<void const*>(source), bytes,
                                        hip_copy_default_k, handle) != hip_success_k)
                return {status_code_t::device_k, "The runtime refused a copy"};
            return {};
        }

        /** Queues a copy of @p source into device memory at @p target, as @ref copy_async does. */
        status_t upload_async(std::uintptr_t target, bytes_view_t source) noexcept {
            return copy_async(target, reinterpret_cast<std::uintptr_t>(source.data()), source.size());
        }

        /** Queues a copy of device memory at @p source into @p target, as @ref copy_async does. */
        status_t download_async(bytes_span_t target, std::uintptr_t source) noexcept {
            return copy_async(reinterpret_cast<std::uintptr_t>(target.data()), source, target.size());
        }

        /**
         *  @brief Queues setting @p bytes at @p target to @p value; the device is current.
         *
         *  @par Statuses:
         *  - @c device_k The runtime refused the fill.
         */
        status_t fill_async(std::uintptr_t target, byte_t value, std::size_t bytes) noexcept {
            if (handle.api().fill_async(reinterpret_cast<void*>(target), static_cast<int>(value), bytes, handle) !=
                hip_success_k)
                return {status_code_t::device_k, "The runtime refused a fill"};
            return {};
        }
    };

    hip_executor_t() noexcept = default;
    hip_executor_t(hip_executor_t&& other) noexcept { swap(other); }
    hip_executor_t& operator=(hip_executor_t&& other) noexcept { return swap(other), *this; }
    hip_executor_t(hip_executor_t const&) = delete;
    hip_executor_t& operator=(hip_executor_t const&) = delete;

    /** Exchanges the stream records and the ceiling, none of which moves in memory; the hold
     *  stays, since a move with a call in flight is misuse. */
    void swap(hip_executor_t& other) noexcept {
        std::swap(streams_, other.streams_);
        std::swap(streams_count_, other.streams_count_);
    }

    /**
     *  @brief Claims the executor for one call, answering the hold, which releases it when it dies.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     */
    expected<claim_t> claim() noexcept {
        u32_t idle = 0;
        if (!fu::standard_atomic_ref<u32_t>(held_).compare_exchange_strong(idle, 1, std::memory_order_acquire,
                                                                           std::memory_order_relaxed))
            return {{}, {status_code_t::contended_k, "Another call holds the executor"}};
        return {claim_t {held_}, status_code_t::success_k};
    }

    /**
     *  @brief Spawns one non-blocking stream per entry of @p devices through @p api, or adopts
     *      @p streams, one per entry and on it.
     *
     *  Each stream's memory and kernels open with @p memory_arguments. An adopted stream becomes
     *  the executor's, which synchronizes and destroys it when it dies. Adoption comes after every
     *  step that can fail, so a refused spawn leaves every one of @p streams the caller's.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k No device is named, more than a listing holds, the @p streams count
     *    differs from theirs, or a stream is null or a default one.
     *  - @c config_mismatch_k A stream sits on another device.
     *  - @c device_k There is no driver, or a device refused its stream or its event.
     *  - @c out_of_memory_k A stream record or its memory could not be allocated.
     */
    template <typename... memory_arguments_>
    static expected<hip_executor_t> spawn(hip_api_t const& api, std::span<int const> devices,
                                          std::span<void* const> streams,
                                          memory_arguments_&... memory_arguments) noexcept {
        if (!streams.empty() && streams.size() != devices.size())
            return {{}, {status_code_t::invalid_argument_k, "One stream per device"}};
        for (std::size_t index = 0; index != streams.size(); ++index)
            if (status_t const checked = hip_stream_t::adoptable(api, static_cast<std::size_t>(devices[index]),
                                                                 streams[index]);
                checked.failed())
                return {{}, checked};
        // Adopted streams wait on placeholders owning nothing until every fallible step is behind.
        auto const placeholder = [](hip_api_t const& table) noexcept -> expected<hip_stream_t> {
            return {hip_stream_t {table, nullptr}, status_code_t::success_k};
        };
        auto [executor, build_status] = build(api, devices, streams.empty() ? &hip_stream_t::open : +placeholder,
                                              memory_arguments...);
        if (build_status.failed()) return {{}, build_status};
        // Every fallible step is behind; each record takes its adopted stream into the same owner,
        // which its memory already points at.
        for (std::size_t index = 0; index != streams.size(); ++index)
            executor.streams_[index]->handle = hip_stream_t {api, static_cast<hip_stream_handle_t>(streams[index])};
        return {std::move(executor), status_code_t::success_k};
    }

    /** Waits for everything queued on every stream - the fleet's completion - marking each stream
     *  with its device bound, then sleeping on each mark in turn. */
    status_t wait() noexcept {
        for (std::size_t index = 0; index != streams_count_; ++index) {
            stream_t& opened = *streams_[index];
            if (status_t const bound = opened.bind(); bound.failed()) return bound;
            if (status_t const recorded = opened.event.record(opened.handle); recorded.failed()) return recorded;
        }
        for (std::size_t index = 0; index != streams_count_; ++index)
            if (status_t const drained = streams_[index]->event.wait(); drained.failed()) return drained;
        return {};
    }

    /** The stream at @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return *streams_[index]; }
    stream_t const& stream(std::size_t index) const noexcept { return *streams_[index]; }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_count_; }

    /** Builds one stream record per entry of @p devices over the stream @p open answers there,
     *  with that device current, and opens its memory and kernels in place. */
    template <typename... memory_arguments_>
    static expected<hip_executor_t> build(hip_api_t const& api, std::span<int const> devices,
                                          expected<hip_stream_t> (*open)(hip_api_t const&),
                                          memory_arguments_&... memory_arguments) noexcept {
        if (devices.empty() || devices.size() > gpu_devices_capacity_k)
            return {{}, {status_code_t::invalid_argument_k, "An executor names one to a listing's devices"}};
        hip_executor_t executor;
        for (int const device : devices) {
            if (hip_set_device(api, device) != hip_success_k)
                return {{}, {status_code_t::device_k, "Device selection failed"}};
            auto [opened, open_status] = open(api);
            if (open_status.failed()) return {{}, open_status};
            std::unique_ptr<stream_t>& record = executor.streams_[executor.streams_count_];
            record.reset(new (std::nothrow) stream_t {static_cast<std::size_t>(device), std::move(opened)});
            if (!record) return {{}, {status_code_t::out_of_memory_k, "No room for a stream record"}};
            ++executor.streams_count_;
            if (!record->event) return {{}, {status_code_t::device_k, "Runtime refused a stream event"}};
            if (status_t const opened_memory = record->memory.open(*record, memory_arguments...);
                opened_memory.failed())
                return {{}, opened_memory};
        }
        return {std::move(executor), status_code_t::success_k};
    }

  private:
    /** One record per named entry, in the caller's order, each on the heap so it never moves. */
    std::unique_ptr<stream_t> streams_[gpu_devices_capacity_k];
    std::size_t streams_count_ = 0;

    /** One while a caller drives the streams, set and cleared through the class's atomic
     *  reference; never swapped. */
    u32_t held_ = 0;
};

#pragma endregion Shared VRAM Executor

} // namespace unum::uform::utilities
