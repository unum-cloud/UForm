/**
 *  @file include/uform/utilities/vram_nvidia_device.hpp
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief The CUDA driver's entry points, resolved into one table from a library opened at first
 *      use, and the stream, event and module owners every NVIDIA executor queues through.
 *
 *  The inference library never links @c libcuda : @ref util::cuda_api_t::load opens it at first
 *  use and resolves the entry points into one table, which stays empty on a machine without the
 *  driver. Plain C++ needing no SDK: the table spells its own types, constants and prototypes, so
 *  any host compiler builds the launches, and the training allocators that link the driver live
 *  in @c vram_nvidia_memory.hpp . The @c Shared @c VRAM regions are text every sibling engine
 *  carries word for word, which @c selftest @c --siblings compares.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - the driver's default stream handles

#include <memory>  // `std::unique_ptr` - an executor's stream records
#include <span>    // `std::span` - the devices an executor opens over, a fatbin's bytes
#include <utility> // `std::exchange`, `std::swap`, `std::move`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the executor's claim word

#include "uform/utilities/blocking_files.hpp" // `shared_library_t` - the driver opened at first use
#include "uform/utilities/ram_device.hpp"     // `memory_domain_of_pci` - pinned placement
#include "uform/utilities/types.hpp"          // `status_t`, `byte_t`, `gpu_vendor_t`, `gpu_devices_capacity_k`

namespace unum::uform::utilities {

#pragma region Driver ABI

/** The driver's status word, @c CUresult , naming the values UForm tells apart; any other
 *  is a failure. */
enum cuda_result_t : int {
    cuda_success_k = 0,
    cuda_error_not_initialized_k = 3,
    cuda_error_invalid_device_k = 101,
    cuda_error_invalid_context_k = 201,
    cuda_error_no_binary_for_gpu_k = 209,
    cuda_error_not_ready_k = 600,
};

/** A device address, @c CUdeviceptr , which unified addressing shares with the host. */
using cuda_device_address_t = unsigned long long;

/** The driver's opaque handles: @c CUcontext , @c CUstream , @c CUevent , @c CUmodule and
 *  @c CUfunction . */
using cuda_context_handle_t = struct cuda_context_opaque_t*;
using cuda_stream_handle_t = struct cuda_stream_opaque_t*;
using cuda_event_handle_t = struct cuda_event_opaque_t*;
using cuda_module_handle_t = struct cuda_module_opaque_t*;
using cuda_function_handle_t = struct cuda_function_opaque_t*;

/** The @c CUdevice_attribute values UForm queries. */
enum cuda_device_attribute_t : int {
    cuda_attribute_warp_size_k = 10,
    cuda_attribute_multiprocessor_count_k = 16,
    cuda_attribute_compute_capability_major_k = 75,
    cuda_attribute_compute_capability_minor_k = 76,
};

/** The @c CUmem_advise values the table's managed memory takes. */
enum cuda_memory_advice_t : int {
    cuda_advise_read_mostly_k = 1,
    cuda_advise_preferred_location_k = 3,
    cuda_advise_accessed_by_k = 5,
};

/** Where memory lives, @c CUmemLocation : a device ordinal, or the host. */
struct cuda_memory_location_t {
    inline static constexpr int device_k = 1, host_k = 2;
    int type = 0;
    int id = 0;
};

/** One pitched copy, @c CUDA_MEMCPY2D , both sides named by unified address. */
struct cuda_copy_2d_t {
    inline static constexpr int unified_k = 4;
    std::size_t source_x_bytes = 0, source_y = 0;
    int source_memory_type = 0;
    void const* source_host = nullptr;
    cuda_device_address_t source_device = 0;
    void* source_array = nullptr;
    std::size_t source_pitch = 0;
    std::size_t destination_x_bytes = 0, destination_y = 0;
    int destination_memory_type = 0;
    void* destination_host = nullptr;
    cuda_device_address_t destination_device = 0;
    void* destination_array = nullptr;
    std::size_t destination_pitch = 0;
    std::size_t width_bytes = 0, height = 0;
};

/** One launch attribute, @c CUlaunchAttribute : its ID, then the driver's 64-byte value union. */
struct cuda_launch_attribute_t {
    int id = 0;
    alignas(8) int value[16] = {};
};

/** One launch's geometry, @c CUlaunchConfig , and the attributes it holds. */
struct cuda_launch_config_t {
    unsigned grid_x = 1, grid_y = 1, grid_z = 1;
    unsigned block_x = 1, block_y = 1, block_z = 1;
    unsigned shared_bytes = 0;
    cuda_stream_handle_t stream = nullptr;
    cuda_launch_attribute_t* attributes = nullptr;
    unsigned attributes_count = 0;
};

/** The flags UForm passes: @c CU_STREAM_NON_BLOCKING , @c CU_EVENT_BLOCKING_SYNC ,
 *  @c CU_EVENT_DISABLE_TIMING and @c CU_MEMHOSTALLOC_PORTABLE . */
inline constexpr unsigned cuda_stream_non_blocking_k = 1;
inline constexpr unsigned cuda_event_blocking_sync_k = 1;
inline constexpr unsigned cuda_event_disable_timing_k = 2;
inline constexpr unsigned cuda_host_portable_k = 1;

/** The driver's legacy and per-thread default streams, which no executor adopts. */
inline cuda_stream_handle_t const cuda_stream_legacy_handle_k = reinterpret_cast<cuda_stream_handle_t>(
    std::uintptr_t {0x1});
inline cuda_stream_handle_t const cuda_stream_per_thread_handle_k = reinterpret_cast<cuda_stream_handle_t>(
    std::uintptr_t {0x2});

#pragma endregion Driver ABI

#pragma region Shared VRAM Driver

/** What the driver calls in stream order once the work queued before it finished, with that work's
 *  status and the pointer given beside it; it must make no driver call. */
using cuda_stream_callback_t = void (*)(cuda_stream_handle_t, cuda_result_t, void*);

/**
 *  @brief The driver entry points every engine calls, resolved from @c libcuda once per process.
 *
 *  Every entry is null when the driver is absent or lacks any of them, which leaves NVIDIA with no
 *  devices; each spells the prototype of the exported symbol it is resolved by, the @c _v2 one
 *  where @c <cuda.h> maps the name onto it. An engine's optional entries resolve apart from these,
 *  so a driver lacking one keeps this table whole.
 */
struct cuda_api_t {
    cuda_result_t (*init)(unsigned) = nullptr;
    cuda_result_t (*get_device_count)(int*) = nullptr;
    cuda_result_t (*get_device)(int*, int) = nullptr;
    cuda_result_t (*get_device_attribute)(int*, cuda_device_attribute_t, int) = nullptr;
    cuda_result_t (*get_device_name)(char*, int, int) = nullptr;
    cuda_result_t (*get_pci_bus_id)(char*, int, int) = nullptr;
    cuda_result_t (*get_device_memory)(std::size_t*, int) = nullptr;
    cuda_result_t (*can_access_peer)(int*, int, int) = nullptr;
    cuda_result_t (*retain_primary_context)(cuda_context_handle_t*, int) = nullptr;
    cuda_result_t (*set_current_context)(cuda_context_handle_t) = nullptr;
    cuda_result_t (*pointer_attribute)(void*, int, cuda_device_address_t) = nullptr;
    cuda_result_t (*allocate_async)(cuda_device_address_t*, std::size_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*free_async)(cuda_device_address_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*allocate_managed)(cuda_device_address_t*, std::size_t, unsigned) = nullptr;
    cuda_result_t (*allocate)(cuda_device_address_t*, std::size_t) = nullptr;
    cuda_result_t (*memory_advise)(cuda_device_address_t, std::size_t, cuda_memory_advice_t,
                                   cuda_memory_location_t) = nullptr;
    cuda_result_t (*free)(cuda_device_address_t) = nullptr;
    cuda_result_t (*allocate_host)(void**, std::size_t, unsigned) = nullptr;
    cuda_result_t (*free_host)(void*) = nullptr;
    cuda_result_t (*register_host)(void const*, std::size_t, unsigned) = nullptr;
    cuda_result_t (*unregister_host)(void const*) = nullptr;
    cuda_result_t (*copy)(cuda_device_address_t, cuda_device_address_t, std::size_t) = nullptr;
    cuda_result_t (*copy_async)(cuda_device_address_t, cuda_device_address_t, std::size_t,
                                cuda_stream_handle_t) = nullptr;
    cuda_result_t (*copy_2d_async)(cuda_copy_2d_t const*, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*fill)(cuda_device_address_t, unsigned char, std::size_t) = nullptr;
    cuda_result_t (*fill_async)(cuda_device_address_t, unsigned char, std::size_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*fill_words_async)(cuda_device_address_t, unsigned, std::size_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*stream_create)(cuda_stream_handle_t*, unsigned) = nullptr;
    cuda_result_t (*stream_destroy)(cuda_stream_handle_t) = nullptr;
    cuda_result_t (*stream_synchronize)(cuda_stream_handle_t) = nullptr;
    cuda_result_t (*stream_query)(cuda_stream_handle_t) = nullptr;
    cuda_result_t (*stream_get_context)(cuda_stream_handle_t, cuda_context_handle_t*) = nullptr;
    cuda_result_t (*stream_wait_event)(cuda_stream_handle_t, cuda_event_handle_t, unsigned) = nullptr;
    cuda_result_t (*stream_add_callback)(cuda_stream_handle_t, cuda_stream_callback_t, void*, unsigned) = nullptr;
    cuda_result_t (*event_create)(cuda_event_handle_t*, unsigned) = nullptr;
    cuda_result_t (*event_record)(cuda_event_handle_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*event_query)(cuda_event_handle_t) = nullptr;
    cuda_result_t (*event_synchronize)(cuda_event_handle_t) = nullptr;
    cuda_result_t (*event_destroy)(cuda_event_handle_t) = nullptr;
    cuda_result_t (*module_load_data)(cuda_module_handle_t*, void const*) = nullptr;
    cuda_result_t (*module_unload)(cuda_module_handle_t) = nullptr;
    cuda_result_t (*module_get_function)(cuda_function_handle_t*, cuda_module_handle_t,
                                         char const*) = nullptr; // vendor-boundary
    cuda_result_t (*function_get_attribute)(int*, int, cuda_function_handle_t) = nullptr;
    cuda_result_t (*function_set_attribute)(cuda_function_handle_t, int, int) = nullptr;
    cuda_result_t (*launch_kernel)(cuda_launch_config_t const*, cuda_function_handle_t, void**, void**) = nullptr;

    /** Every listed device's primary context by ordinal, retained once as the table loads and never
     *  released, as the driver stays open; null past the listed devices. */
    cuda_context_handle_t primary_contexts[gpu_devices_capacity_k] = {};

    /** The memory domain nearest each listed device, where its pinned host bytes go, or
     *  @c no_memory_domain_k where none is known. */
    fu_memory_domain_id_t memory_domains[gpu_devices_capacity_k] = {};

    /** Whether the driver loaded and initialized with every entry resolved. */
    explicit operator bool() const noexcept { return launch_kernel != nullptr; }

    /** Opens the driver, resolves every entry and initializes it, or answers the empty table. The
     *  library stays open for the life of the process, as the table points into it. */
    static cuda_api_t load() noexcept {
        // Each system names the driver its own way, and opening another's name merely fails.
        shared_library_t library;
        for (char const* name : {"libcuda.so.1", "nvcuda.dll"}) // vendor-boundary
            if ((library = shared_library_t::open(name))) break;
        if (!library) return {};
        cuda_api_t api;
        bool const resolved = //
            library.resolve("cuInit", api.init) && library.resolve("cuDeviceGetCount", api.get_device_count) &&
            library.resolve("cuDeviceGet", api.get_device) &&
            library.resolve("cuDeviceGetAttribute", api.get_device_attribute) &&
            library.resolve("cuDeviceGetName", api.get_device_name) &&
            library.resolve("cuDeviceGetPCIBusId", api.get_pci_bus_id) &&
            library.resolve("cuDeviceTotalMem_v2", api.get_device_memory) &&
            library.resolve("cuDeviceCanAccessPeer", api.can_access_peer) &&
            library.resolve("cuDevicePrimaryCtxRetain", api.retain_primary_context) &&
            library.resolve("cuCtxSetCurrent", api.set_current_context) &&
            library.resolve("cuPointerGetAttribute", api.pointer_attribute) &&
            library.resolve("cuMemAllocAsync", api.allocate_async) &&
            library.resolve("cuMemFreeAsync", api.free_async) &&
            library.resolve("cuMemAllocManaged", api.allocate_managed) &&
            library.resolve("cuMemAlloc_v2", api.allocate) && library.resolve("cuMemAdvise_v2", api.memory_advise) &&
            library.resolve("cuMemFree_v2", api.free) && library.resolve("cuMemHostAlloc", api.allocate_host) &&
            library.resolve("cuMemFreeHost", api.free_host) &&
            library.resolve("cuMemHostRegister_v2", api.register_host) &&
            library.resolve("cuMemHostUnregister", api.unregister_host) && library.resolve("cuMemcpy", api.copy) &&
            library.resolve("cuMemcpyAsync", api.copy_async) &&
            library.resolve("cuMemcpy2DAsync_v2", api.copy_2d_async) && library.resolve("cuMemsetD8_v2", api.fill) &&
            library.resolve("cuMemsetD8Async", api.fill_async) &&
            library.resolve("cuMemsetD32Async", api.fill_words_async) &&
            library.resolve("cuStreamCreate", api.stream_create) &&
            library.resolve("cuStreamDestroy_v2", api.stream_destroy) &&
            library.resolve("cuStreamSynchronize", api.stream_synchronize) &&
            library.resolve("cuStreamQuery", api.stream_query) &&
            library.resolve("cuStreamGetCtx", api.stream_get_context) &&
            library.resolve("cuStreamWaitEvent", api.stream_wait_event) &&
            library.resolve("cuStreamAddCallback", api.stream_add_callback) &&
            library.resolve("cuEventCreate", api.event_create) && library.resolve("cuEventRecord", api.event_record) &&
            library.resolve("cuEventQuery", api.event_query) &&
            library.resolve("cuEventSynchronize", api.event_synchronize) &&
            library.resolve("cuEventDestroy_v2", api.event_destroy) &&
            library.resolve("cuModuleLoadData", api.module_load_data) &&
            library.resolve("cuModuleUnload", api.module_unload) &&
            library.resolve("cuModuleGetFunction", api.module_get_function) &&
            library.resolve("cuFuncGetAttribute", api.function_get_attribute) &&
            library.resolve("cuFuncSetAttribute", api.function_set_attribute) &&
            library.resolve("cuLaunchKernelEx", api.launch_kernel);
        // A driver without a usable GPU fails its initialization, which lists no devices either.
        if (!resolved || api.init(0) != cuda_success_k) return {};
        int devices_count = 0;
        if (api.get_device_count(&devices_count) != cuda_success_k) return {};
        for (int ordinal = 0; ordinal != devices_count && static_cast<std::size_t>(ordinal) != gpu_devices_capacity_k;
             ++ordinal) {
            // A device with no retained context refuses at its own `cuda_set_device`.
            int device = 0;
            cuda_context_handle_t primary = nullptr;
            if (api.get_device(&device, ordinal) == cuda_success_k &&
                api.retain_primary_context(&primary, device) == cuda_success_k)
                api.primary_contexts[ordinal] = primary;
            // A bus that names no memory domain leaves the device's pinned bytes on the heap.
            char bus_id[32] = {};
            [[maybe_unused]] cuda_result_t const named = api.get_pci_bus_id(bus_id, sizeof(bus_id), device);
            api.memory_domains[ordinal] = memory_domain_of_pci(bus_id);
        }
        return api;
    }
};

/** Makes the primary context of device @p ordinal, which @p api retained as it loaded, current on
 *  the calling thread: the driver twin of selecting a device, which every stream, allocation and
 *  launch after it runs on. Every entry asks again, rebinding a caller that switched devices. */
inline cuda_result_t cuda_set_device(cuda_api_t const& api, int ordinal) noexcept {
    if (!api) return cuda_error_not_initialized_k;
    if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= gpu_devices_capacity_k || !api.primary_contexts[ordinal])
        return cuda_error_invalid_device_k;
    return api.set_current_context(api.primary_contexts[ordinal]);
}

/** One device's ordering identity: a non-blocking stream an executor opens or adopts and drains and
 *  destroys when it dies, or the device's null legacy stream, which owns nothing. */
struct cuda_stream_t {

    /**
     *  @brief Opens a non-blocking stream in the current context, owned from now on.
     *
     *  @par Statuses:
     *  - @c device_k The driver refused the stream.
     */
    static expected<cuda_stream_t> open(cuda_api_t const& api) noexcept {
        cuda_stream_t stream;
        stream.api_ = &api;
        if (!api || api.stream_create(&stream.handle_, cuda_stream_non_blocking_k) != cuda_success_k)
            return {{}, {status_code_t::device_k, "Driver refused a stream"}};
        return {std::move(stream), status_code_t::success_k};
    }

    /**
     *  @brief Checks that @p handle, a caller's stream, may be adopted as the stream of device
     *      @p ordinal, without taking it.
     *
     *  @par Statuses:
     *  - @c device_k There is no driver.
     *  - @c invalid_argument_k The @p handle is null or a default stream, which no owner destroys.
     *  - @c config_mismatch_k The stream sits off the primary context of @p ordinal.
     */
    static status_t adoptable(cuda_api_t const& api, std::size_t ordinal, void* handle) noexcept {
        if (!api) return {status_code_t::device_k, "No CUDA driver"};
        auto const stream = static_cast<cuda_stream_handle_t>(handle);
        if (!stream || stream == cuda_stream_legacy_handle_k || stream == cuda_stream_per_thread_handle_k)
            return {status_code_t::invalid_argument_k, "A default stream cannot be adopted"};
        cuda_context_handle_t context = nullptr;
        if (ordinal >= gpu_devices_capacity_k || api.stream_get_context(stream, &context) != cuda_success_k ||
            context != api.primary_contexts[ordinal])
            return {status_code_t::config_mismatch_k, "A stream sits off its device's primary context"};
        return {};
    }

    /** Adopts @p handle, a stream the caller created, which it synchronizes and destroys. */
    explicit cuda_stream_t(cuda_api_t const& api, cuda_stream_handle_t handle) noexcept : api_(&api), handle_(handle) {}

    cuda_stream_t() noexcept = default;
    cuda_stream_t(cuda_stream_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    cuda_stream_t& operator=(cuda_stream_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(handle_, other.handle_);
        return *this;
    }
    cuda_stream_t(cuda_stream_t const&) = delete;
    cuda_stream_t& operator=(cuda_stream_t const&) = delete;
    ~cuda_stream_t() noexcept {
        if (handle_) api_->stream_synchronize(handle_), api_->stream_destroy(handle_);
    }

    /** The driver table every call on this stream goes through. */
    cuda_api_t const& api() const noexcept { return *api_; }

    operator cuda_stream_handle_t() const noexcept { return handle_; }

  private:
    cuda_api_t const* api_ = nullptr;
    cuda_stream_handle_t handle_ = nullptr;
};

/** Owned event, born in the current context, marking a point on a stream that the host polls or
 *  awaits. It blocks on sync, so a drain sleeps instead of spinning a core, and destroying it
 *  never waits for the work it marked. */
struct cuda_event_t {

    /** An empty event, which records nothing. */
    cuda_event_t() noexcept = default;

    explicit cuda_event_t(cuda_api_t const& api) noexcept : api_(&api) {
        if (api) api.event_create(&handle_, cuda_event_blocking_sync_k | cuda_event_disable_timing_k);
    }
    cuda_event_t(cuda_event_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    cuda_event_t& operator=(cuda_event_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(handle_, other.handle_);
        return *this;
    }
    cuda_event_t(cuda_event_t const&) = delete;
    cuda_event_t& operator=(cuda_event_t const&) = delete;
    ~cuda_event_t() noexcept {
        if (handle_) api_->event_destroy(handle_);
    }

    operator cuda_event_handle_t() const noexcept { return handle_; }

    /**
     *  @brief Marks everything queued on @p stream so far, which @ref finished and @ref wait
     *      then await in order.
     *
     *  @par Statuses:
     *  - @c device_k The event was never created.
     */
    status_t record(cuda_stream_handle_t stream) noexcept {
        if (!handle_ || api_->event_record(handle_, stream) != cuda_success_k)
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
        cuda_result_t const queried = api_->event_query(handle_);
        if (queried == cuda_error_not_ready_k) return {status_code_t::deferred_k};
        if (queried != cuda_success_k) return {status_code_t::device_k, "The device failed the recorded work"};
        return {};
    }

    /**
     *  @brief Sleeps until the recorded work finishes and answers whether the device finished it.
     *
     *  @par Statuses:
     *  - @c device_k The event was never created.
     */
    status_t wait() noexcept {
        if (!handle_ || api_->event_synchronize(handle_) != cuda_success_k)
            return {status_code_t::device_k, "The device failed the recorded work"};
        return {};
    }

  private:
    cuda_api_t const* api_ = nullptr;
    cuda_event_handle_t handle_ = nullptr;
};

/** One fatbin loaded into one GPU's primary context and unloaded with this owner; kernels are found
 *  by name. */
struct cuda_module_t {
    using function_t = cuda_function_handle_t;

    cuda_module_t() noexcept = default;
    cuda_module_t(cuda_module_t&& other) noexcept
        : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)), device_(other.device_) {}
    cuda_module_t& operator=(cuda_module_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(handle_, other.handle_), std::swap(device_, other.device_);
        return *this;
    }
    cuda_module_t(cuda_module_t const&) = delete;
    cuda_module_t& operator=(cuda_module_t const&) = delete;
    ~cuda_module_t() noexcept {
        if (handle_ && cuda_set_device(*api_, device_) == cuda_success_k) api_->module_unload(handle_);
    }

    /**
     *  @brief Loads @p fatbin onto device @p device through @p api, binding it first, since
     *      a module loads into the current context; the driver picks the cubin matching the
     *      device, or JIT-compiles its PTX.
     *
     *  @par Statuses:
     *  - @c device_k The @p fatbin is empty, or the driver refused it.
     */
    static expected<cuda_module_t> load(cuda_api_t const& api, int device, bytes_view_t fatbin) noexcept {
        cuda_module_t module;
        module.api_ = &api;
        module.device_ = device;
        if (fatbin.empty() || cuda_set_device(api, device) != cuda_success_k ||
            api.module_load_data(&module.handle_, fatbin.data()) != cuda_success_k)
            return {{}, {status_code_t::device_k, "The driver refused a fatbin"}};
        return {std::move(module), status_code_t::success_k};
    }

    /** The driver table the fatbin was loaded through, which launches of its kernels go through. */
    cuda_api_t const& api() const noexcept { return *api_; }

    /** The kernel exported as @p name, or null when the fatbin has none. */
    function_t function(char const* name) const noexcept { // vendor-boundary
        function_t function = nullptr;
        return handle_ && api_->module_get_function(&function, handle_, name) == cuda_success_k ? function : nullptr;
    }

  private:
    cuda_api_t const* api_ = nullptr;
    cuda_module_handle_t handle_ = nullptr;
    int device_ = 0;
};

#pragma endregion Shared VRAM Driver

#pragma region Listing and Scratch

/** Writes every CUDA device @p api reports into @p into , up to its size, in the driver's order,
 *  and answers how many it wrote, skipping any device the driver will not describe, and none for
 *  an empty table. */
inline std::size_t list_nvidia_gpu_devices(cuda_api_t const& api, std::span<gpu_device_t> into) noexcept {
    std::size_t count = 0;
    int devices_count = 0;
    if (!api || api.get_device_count(&devices_count) != cuda_success_k) return count;
    for (int ordinal = 0; ordinal != devices_count && count != into.size(); ++ordinal) {
        int device = 0, warp_lanes = 0, compute_units = 0, major = 0, minor = 0;
        std::size_t memory_bytes = 0;
        gpu_device_t listed {gpu_vendor_t::nvidia_k, static_cast<u32_t>(ordinal)};
        if (api.get_device(&device, ordinal) != cuda_success_k ||
            api.get_device_name(listed.name, static_cast<int>(sizeof(listed.name)), device) != cuda_success_k ||
            api.get_device_attribute(&warp_lanes, cuda_attribute_warp_size_k, device) != cuda_success_k ||
            api.get_device_attribute(&compute_units, cuda_attribute_multiprocessor_count_k, device) != cuda_success_k ||
            api.get_device_attribute(&major, cuda_attribute_compute_capability_major_k, device) != cuda_success_k ||
            api.get_device_attribute(&minor, cuda_attribute_compute_capability_minor_k, device) != cuda_success_k ||
            api.get_device_memory(&memory_bytes, device) != cuda_success_k)
            continue;
        listed.architecture_major = static_cast<u32_t>(major), listed.architecture_minor = static_cast<u32_t>(minor);
        listed.warp_lanes = static_cast<u32_t>(warp_lanes), listed.compute_units = static_cast<u32_t>(compute_units);
        listed.memory_bytes = memory_bytes;
        into[count++] = listed;
    }
    return count;
}

/** Where a cuda stream's launches find their functions: the process's one table of UForm's bundle
 *  per device, which the GPU runtime owns. */
using cuda_kernels_t = kernel_tables<cuda_module_t>*;

/**
 *  @brief UForm's queue-ordered memory on one CUDA stream: the activations a call computes in, on
 *      the device, and the pinned transfers it packs what the device reads into.
 *
 *  Both only hand out what @ref reserve grew them by, with the stream's device bound, before a call
 *  allocates, and @ref release frees them after the stream drained, each chunk through the verb
 *  that made it.
 */
struct cuda_scratch_t {

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
        record.kernels = &runtime.cuda_kernels;
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
            if (stream.handle.api().allocate_host(&pages, transfer_bytes, cuda_host_portable_k) != cuda_success_k)
                return {status_code_t::out_of_memory_k, "The host refused pinned pages"};
            if (status_t const appended = transfers.append({static_cast<b8_t*>(pages), transfer_bytes});
                appended.failed())
                return stream.handle.api().free_host(pages), appended;
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
        for (bytes_span_t const chunk : transfers.chunks()) stream.handle.api().free_host(chunk.data());
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
struct cuda_executor_t {

    /** The engine's queue-ordered memory on one stream. */
    using scratch_t = cuda_scratch_t;

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

    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;

    /** One device's stream: what it queues on, the event its waits sleep on, the kernels it
     *  launches, and the engine's memory, declared last so it dies first. */
    struct stream_t {

        /** The CUDA ordinal of the device this stream queues on, as the GPU runtime lists it. */
        std::size_t ordinal = 0;

        /** Opened by the executor, adopted from the caller, or the device's null legacy stream. */
        cuda_stream_t handle;

        /** What @ref wait records and sleeps on, so a wait frees its core. */
        cuda_event_t event;

        /** The kernels this stream launches, as the engine resolves them when the record opens. */
        cuda_kernels_t kernels = {};

        /** The engine's memory, safe to reuse only in this stream's queue order, opened once this
         *  record sits in the executor. */
        scratch_t memory;

        /** Takes @p opened as the stream of device @p device, which is current, as the event is
         *  born there too. */
        stream_t(std::size_t device, cuda_stream_t&& opened) noexcept
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
            if (cuda_set_device(handle.api(), static_cast<int>(ordinal)) != cuda_success_k)
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
            cuda_device_address_t address = 0;
            if (handle.api().allocate_async(&address, bytes, handle) != cuda_success_k)
                return {0, {status_code_t::out_of_memory_k, "The device refused an allocation"}};
            return {static_cast<std::uintptr_t>(address), status_code_t::success_k};
        }

        /** Queues freeing @p address, which @ref allocate_async answered, in this stream's
         *  order. */
        void free_async(std::uintptr_t address) noexcept { handle.api().free_async(address, handle); }

        /**
         *  @brief Queues a copy of @p bytes from @p source to @p target, either side the host's
         *      or a device's, as unified addressing tells apart; the device is current.
         *
         *  @par Statuses:
         *  - @c device_k The driver refused the copy.
         */
        status_t copy_async(std::uintptr_t target, std::uintptr_t source, std::size_t bytes) noexcept {
            if (handle.api().copy_async(target, source, bytes, handle) != cuda_success_k)
                return {status_code_t::device_k, "The driver refused a copy"};
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
         *  - @c device_k The driver refused the fill.
         */
        status_t fill_async(std::uintptr_t target, byte_t value, std::size_t bytes) noexcept {
            if (handle.api().fill_async(target, static_cast<unsigned char>(value), bytes, handle) != cuda_success_k)
                return {status_code_t::device_k, "The driver refused a fill"};
            return {};
        }
    };

    cuda_executor_t() noexcept = default;
    cuda_executor_t(cuda_executor_t&& other) noexcept { swap(other); }
    cuda_executor_t& operator=(cuda_executor_t&& other) noexcept { return swap(other), *this; }
    cuda_executor_t(cuda_executor_t const&) = delete;
    cuda_executor_t& operator=(cuda_executor_t const&) = delete;

    /** Exchanges the stream records and the ceiling, none of which moves in memory; the hold
     *  stays, since a move with a call in flight is misuse. */
    void swap(cuda_executor_t& other) noexcept {
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
     *  - @c config_mismatch_k A stream sits off its device's primary context.
     *  - @c device_k There is no driver, or a device refused its stream or its event.
     *  - @c out_of_memory_k A stream record or its memory could not be allocated.
     */
    template <typename... memory_arguments_>
    static expected<cuda_executor_t> spawn(cuda_api_t const& api, std::span<int const> devices,
                                           std::span<void* const> streams,
                                           memory_arguments_&... memory_arguments) noexcept {
        if (!streams.empty() && streams.size() != devices.size())
            return {{}, {status_code_t::invalid_argument_k, "One stream per device"}};
        for (std::size_t index = 0; index != streams.size(); ++index)
            if (status_t const checked = cuda_stream_t::adoptable(api, static_cast<std::size_t>(devices[index]),
                                                                  streams[index]);
                checked.failed())
                return {{}, checked};
        // Adopted streams wait on placeholders owning nothing until every fallible step is behind.
        auto const placeholder = [](cuda_api_t const& table) noexcept -> expected<cuda_stream_t> {
            return {cuda_stream_t {table, nullptr}, status_code_t::success_k};
        };
        auto [executor, build_status] = build(api, devices, streams.empty() ? &cuda_stream_t::open : +placeholder,
                                              memory_arguments...);
        if (build_status.failed()) return {{}, build_status};
        // Every fallible step is behind; each record takes its adopted stream into the same owner,
        // which its memory already points at.
        for (std::size_t index = 0; index != streams.size(); ++index)
            executor.streams_[index]->handle = cuda_stream_t {api, static_cast<cuda_stream_handle_t>(streams[index])};
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
    static expected<cuda_executor_t> build(cuda_api_t const& api, std::span<int const> devices,
                                           expected<cuda_stream_t> (*open)(cuda_api_t const&),
                                           memory_arguments_&... memory_arguments) noexcept {
        if (devices.empty() || devices.size() > gpu_devices_capacity_k)
            return {{}, {status_code_t::invalid_argument_k, "An executor names one to a listing's devices"}};
        cuda_executor_t executor;
        for (int const device : devices) {
            if (cuda_set_device(api, device) != cuda_success_k)
                return {{}, {status_code_t::device_k, "Device selection failed"}};
            auto [opened, open_status] = open(api);
            if (open_status.failed()) return {{}, open_status};
            std::unique_ptr<stream_t>& record = executor.streams_[executor.streams_count_];
            record.reset(new (std::nothrow) stream_t {static_cast<std::size_t>(device), std::move(opened)});
            if (!record) return {{}, {status_code_t::out_of_memory_k, "No room for a stream record"}};
            ++executor.streams_count_;
            if (!record->event) return {{}, {status_code_t::device_k, "Driver refused a stream event"}};
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
