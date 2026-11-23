/**
 *  @file include/uform/utilities/vram_apple_device.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Host-side vocabulary for driving Apple GPUs: Metal opened at first use, the device
 *      contexts and their pipelines, shared buffers, launches, streams and the executor.
 *
 *  Metal's API is Objective-C, reached here through @c objc_msgSend cast to each call's exact
 *  prototype, so a plain C++ unit drives it with no SDK header. Nothing links Metal, Foundation,
 *  I/O Kit or the Objective-C runtime: @ref unum::uform::utilities::metal_api_t::load opens them
 *  and resolves the few C entry points UForm calls into one table, which stays empty where any is
 *  absent, so Apple then lists no devices and the binary still loads. Kernels travel as MSL source
 *  and compile on the device at first use, so no Metal toolchain is needed to build. Memory is
 *  unified and every buffer is shared, so the host reads and writes it in place. A device context
 *  records every buffer UForm allocates on it, and a launch makes the ones it touches resident.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::strcmp`, `std::strncpy`

#include <span>    // `std::span` - the device list an executor opens over
#include <utility> // `std::exchange`, `std::move`, `std::swap`

#include <smashtable/basic_growing_pool.hpp> // `st::basic_growing_pool` - every buffer a device holds

#include "uform/utilities/array.hpp"          // `per_gpu_array`, `max_gpus_per_host_k`
#include "uform/utilities/blocking_files.hpp" // `shared_library_t` - Metal opened at first use
#include "uform/utilities/types.hpp"          // `expected`, `status_t`, `st::basic_vector`, `fu::spin_mutex_t`

namespace unum::uform::utilities {

/** The Objective-C runtime's selector, @c SEL , opaque to everyone but the runtime. */
using metal_selector_t = struct metal_selector_opaque_t*;

/**
 *  @brief The Metal, Objective-C runtime, I/O Kit and Core Foundation entry points UForm calls,
 *      resolved on first use from the system's own frameworks and @c libobjc .
 *
 *  Every entry is null when any library is absent or lacks any of them, which leaves Apple with no
 *  devices. Foundation is opened for @c NSString alone, which no entry names, and I/O Kit for the
 *  GPU core count, which Metal does not report. Each entry spells the C prototype of its symbol,
 *  with Core Foundation's references as plain pointers and I/O Kit's objects as their 32-bit Mach
 *  ports; @c objc_msgSend takes none of its own, so every send casts it to the call's exact one.
 */
struct metal_api_t {
    void* (*copy_all_devices)() = nullptr;
    void (*send)() = nullptr;
    metal_selector_t (*register_selector)(char const*) = nullptr; // vendor-boundary: the runtime's prototype
    void* (*get_class)(char const*) = nullptr;                    // vendor-boundary: the runtime's prototype
    void* (*registry_entry_id_matching)(u64_t) = nullptr;
    u32_t (*get_matching_service)(u32_t, void*) = nullptr;
    void* (*search_registry_property)(u32_t, char const*, void const*, void const*, u32_t) = nullptr; // vendor-boundary
    int (*release_registry_object)(u32_t) = nullptr;
    void* (*create_string)(void const*, char const*, u32_t) = nullptr; // vendor-boundary: Core Foundation's prototype
    unsigned char (*get_number_value)(void const*, long, void*) = nullptr;
    void (*release_reference)(void const*) = nullptr;
    void* (*autorelease_pool_push)() = nullptr;
    void (*autorelease_pool_pop)(void*) = nullptr;

    /** Whether every library loaded with every entry resolved. */
    explicit operator bool() const noexcept { return autorelease_pool_pop != nullptr; }

    /** Opens the libraries and resolves every entry, or answers the empty table. The libraries stay
     *  open for the life of the process, as the table points into them. */
    static metal_api_t load() noexcept {
        shared_library_t const metal = shared_library_t::open("/System/Library/Frameworks/Metal.framework/Metal");
        shared_library_t const foundation = shared_library_t::open(
            "/System/Library/Frameworks/Foundation.framework/Foundation");
        shared_library_t const runtime = shared_library_t::open("/usr/lib/libobjc.A.dylib");
        shared_library_t const registry = shared_library_t::open("/System/Library/Frameworks/IOKit.framework/IOKit");
        shared_library_t const core = shared_library_t::open(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation");
        metal_api_t api;
        bool const resolved = //
            foundation && metal.resolve("MTLCopyAllDevices", api.copy_all_devices) &&
            runtime.resolve("objc_msgSend", api.send) && runtime.resolve("sel_registerName", api.register_selector) &&
            runtime.resolve("objc_getClass", api.get_class) &&
            registry.resolve("IORegistryEntryIDMatching", api.registry_entry_id_matching) &&
            registry.resolve("IOServiceGetMatchingService", api.get_matching_service) &&
            registry.resolve("IORegistryEntrySearchCFProperty", api.search_registry_property) &&
            registry.resolve("IOObjectRelease", api.release_registry_object) &&
            core.resolve("CFStringCreateWithCString", api.create_string) &&
            core.resolve("CFNumberGetValue", api.get_number_value) &&
            core.resolve("CFRelease", api.release_reference) &&
            runtime.resolve("objc_autoreleasePoolPush", api.autorelease_pool_push) &&
            runtime.resolve("objc_autoreleasePoolPop", api.autorelease_pool_pop);
        return resolved ? api : metal_api_t {};
    }
};

/** Sends @p selector to @p target with @p arguments through @p api , cast to the exact prototype
 *  the call has. */
template <typename result_type_ = void*, typename... argument_types_>
inline result_type_ metal_send(metal_api_t const& api, void* target,
                               char const* selector, // vendor-boundary: the runtime names selectors by C strings
                               argument_types_... arguments) noexcept {
    using function_t = result_type_ (*)(void*, metal_selector_t, argument_types_...);
    return reinterpret_cast<function_t>(api.send)( // vendor-boundary: `objc_msgSend` takes the callee's prototype
        target, api.register_selector(selector), arguments...);
}

/** An @c NSString over @p text , autoreleased. */
inline void* metal_string(metal_api_t const& api, char const* text) noexcept { // vendor-boundary: a C string
    return metal_send(api, api.get_class("NSString"), "stringWithUTF8String:", text);
}

/** A launch extent, laid out as @c MTLSize so it passes to Metal by value. */
struct metal_size_t {
    std::size_t width = 1, height = 1, depth = 1;
};

/** @c MTLResourceUsageRead with @c MTLResourceUsageWrite, as a launch may do either to every buffer
 *  it makes resident. */
inline constexpr std::size_t metal_usage_read_write_k = 3;

/** One shared-storage @c MTLBuffer : the table it is released through, the handle a launch binds
 *  and the host address the same bytes sit at. Move-only, and released when it dies. */
struct metal_buffer_t {
    metal_api_t const* api = nullptr;
    void* handle = nullptr;
    b8_t* host = nullptr; // vendor-boundary: Metal answers the contents as one address
    std::size_t bytes = 0;

    metal_buffer_t() noexcept = default;
    metal_buffer_t(metal_buffer_t&& other) noexcept
        : api(other.api), handle(std::exchange(other.handle, nullptr)), host(std::exchange(other.host, nullptr)),
          bytes(std::exchange(other.bytes, 0)) {}
    metal_buffer_t& operator=(metal_buffer_t&& other) noexcept {
        metal_buffer_t dying {std::move(*this)};
        api = other.api, handle = std::exchange(other.handle, nullptr), host = std::exchange(other.host, nullptr);
        bytes = std::exchange(other.bytes, 0);
        return *this;
    }
    metal_buffer_t(metal_buffer_t const&) = delete;
    metal_buffer_t& operator=(metal_buffer_t const&) = delete;
    ~metal_buffer_t() noexcept {
        if (handle) metal_send<void>(*api, handle, "release");
    }

    explicit operator bool() const noexcept { return handle != nullptr; }
};

/** One built entry point as a launch reads it: the pipeline state, and the threads one threadgroup
 *  of it holds at most, its @c maxTotalThreadsPerThreadgroup , which its register use bounds. */
struct metal_pipeline_state_t {
    void* pipeline = nullptr;
    u32_t max_threads = 0;
};

/** One compiled entry point and the source library it shares with the other entry points; the name
 *  is copied, as callers spell it into their own buffers. */
struct metal_pipeline_t {
    char const* source = nullptr;
    char name[64] = {};
    void* library = nullptr;
    metal_pipeline_state_t state;
};

/** One listed Apple GPU's process-wide state: the table its calls go through, the device, its
 *  default queue, its family and cores, and every pipeline built for it, guarded by one lock. */
struct metal_context_t {
    metal_api_t const* api = nullptr;
    void* device = nullptr;
    void* queue = nullptr;

    /** The newest Apple GPU family the device supports, such as 9 for @c MTLGPUFamilyApple9 ; zero
     *  for a GPU older than Apple7 or not Apple's own. */
    u32_t family = 0;

    /** The GPU cores the I/O Registry lists for the device, zero when it lists none. */
    u32_t multiprocessor_count = 0;

    fu::spin_mutex_t lock;
    st::basic_vector<metal_pipeline_t> pipelines;

    /** Every shared buffer UForm allocated on the device, guarded by @ref lock : a free finds the
     *  buffer its address names, a launch turns each host address it binds into the buffer's GPU
     *  address and makes that buffer resident, and a copy tells device memory from host memory. */
    st::basic_growing_pool<metal_buffer_t> buffers;

    /**
     *  @brief Kernel @p name 's pipeline from the library built out of @p source under the
     *      @c MTLLanguageVersion @p language_version , built on first use.
     *
     *  A caller meeting another's build waits until it ends, as a module load waits in the CUDA and
     *  HIP drivers. The 32 lanes checked are the simdgroup width every MSL source is written for.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The registry cannot grow.
     *  - @c device_k The device refused the source or has no such kernel.
     *  - @c unsupported_k The pipeline's simdgroups are not 32 lanes wide.
     */
    expected<metal_pipeline_state_t> pipeline(char const* source, // vendor-boundary: Metal reads C strings
                                              char const* name,   // vendor-boundary: Metal reads C strings
                                              std::size_t language_version) noexcept {
        constexpr std::size_t simdgroup_lanes_k = 32;
        st::unique_lock<fu::spin_mutex_t> guard {lock};
        void* library = nullptr;
        for (metal_pipeline_t const& entry : pipelines) {
            if (entry.source != source) continue;
            library = entry.library;
            if (std::strcmp(entry.name, name) == 0) return entry.state;
        }
        if (st::failed(pipelines.reserve(pipelines.size() + 1)))
            return expected<metal_pipeline_state_t>::fail(status_code_t::out_of_memory_k, "Metal pipeline registry");
        metal_api_t const& api = *this->api;
        void* const pool = api.autorelease_pool_push();
        if (!library) {
            void* const options = metal_send(api, metal_send(api, api.get_class("MTLCompileOptions"), "alloc"), "init");
            metal_send<void>(api, options, "setLanguageVersion:", language_version);
            void* error = nullptr;
            library = metal_send(api, device, "newLibraryWithSource:options:error:", metal_string(api, source), options,
                                 &error);
            metal_send<void>(api, options, "release");
        }
        void* pipeline = nullptr;
        if (library) {
            void* const function = metal_send(api, library, "newFunctionWithName:", metal_string(api, name));
            if (function) {
                void* error = nullptr;
                pipeline = metal_send(api, device, "newComputePipelineStateWithFunction:error:", function, &error);
                metal_send<void>(api, function, "release");
            }
        }
        api.autorelease_pool_pop(pool);
        if (!pipeline) return expected<metal_pipeline_state_t>::fail(status_code_t::device_k, "Metal kernel {}", name);
        if (metal_send<std::size_t>(api, pipeline, "threadExecutionWidth") != simdgroup_lanes_k) {
            metal_send<void>(api, pipeline, "release");
            return expected<metal_pipeline_state_t>::fail(status_code_t::unsupported_k, "Metal simdgroups of {}", name);
        }
        u32_t const max_threads = static_cast<u32_t>(
            metal_send<std::size_t>(api, pipeline, "maxTotalThreadsPerThreadgroup"));
        metal_pipeline_t entry {source, {}, library, {pipeline, max_threads}};
        std::strncpy(entry.name, name, sizeof(entry.name) - 1);
        metal_pipeline_state_t const state = entry.state;
        [[maybe_unused]] st::status_t const kept = pipelines.push_back(st::assume_reserved, std::move(entry));
        return state;
    }
};

/** Every Apple GPU Metal lists, in Metal's own order, opened once per process by its caller and
 *  kept for it, as Metal's own device objects are; a device's Apple ordinal is its position here,
 *  NVIDIA and AMD devices coming before Apple ones in every UForm listing. */
struct metal_contexts_t {
    metal_context_t contexts[max_gpus_per_host_k];
    std::size_t count = 0;

    /** Opens every device @p api lists, none when the table is empty, reading each one's newest
     *  Apple family and, from the I/O Registry, its GPU cores, which Metal does not report. */
    explicit metal_contexts_t(metal_api_t const& api) noexcept {
        // `MTLGPUFamilyApple10` down to `Apple7`, numbered past `MTLGPUFamilyApple1` less one
        constexpr std::size_t families_k[] = {1010, 1009, 1008, 1007}, family_base_k = 1000;
        // `kIOMainPortDefault`, `kCFStringEncodingUTF8` and recursive parent iteration
        constexpr u32_t main_port_k = 0, utf8_k = 0x08000100, search_options_k = 1 | 2;
        // `kCFNumberSInt64Type`
        constexpr long signed_64_bits_k = 4;
        // Allocations one device holds at once, far past what a tower and its arenas reach
        constexpr std::size_t buffers_limit_k = std::size_t {1} << 20;
        void* const listed = api ? api.copy_all_devices() : nullptr;
        if (!listed) return;
        void* const pool = api.autorelease_pool_push();
        std::size_t const listed_count = metal_send<std::size_t>(api, listed, "count");
        for (std::size_t position = 0; position != listed_count && count != max_gpus_per_host_k; ++position) {
            void* const device = metal_send(api, listed, "objectAtIndex:", position);
            void* const queue = metal_send(api, device, "newCommandQueue");
            if (!queue) continue;
            metal_context_t& context = contexts[count];
            context.api = &api;
            context.device = metal_send(api, device, "retain");
            context.queue = queue;
            for (std::size_t const family : families_k)
                if (!context.family && metal_send<bool>(api, device, "supportsFamily:", family))
                    context.family = static_cast<u32_t>(family - family_base_k);
            // The lookup consumes the matching dictionary, so only the service is released here
            u32_t const service = api.get_matching_service(
                main_port_k, api.registry_entry_id_matching(metal_send<u64_t>(api, device, "registryID")));
            void* const key = service ? api.create_string(nullptr, "gpu-core-count", utf8_k) : nullptr;
            void* const cores = key ? api.search_registry_property(service, "IOService", key, nullptr, search_options_k)
                                    : nullptr;
            i64_t listed_cores = 0;
            if (cores && !api.get_number_value(cores, signed_64_bits_k, &listed_cores)) listed_cores = 0;
            if (cores) api.release_reference(cores);
            if (key) api.release_reference(key);
            if (service) api.release_registry_object(service);
            context.multiprocessor_count = listed_cores > 0 ? static_cast<u32_t>(listed_cores) : 0;
            auto made = st::basic_growing_pool<metal_buffer_t>::make(0, buffers_limit_k);
            if (made) context.buffers = std::move(*made);
            ++count;
        }
        api.autorelease_pool_pop(pool);
        metal_send<void>(api, listed, "release");
    }
    metal_contexts_t(metal_contexts_t const&) = delete;
    metal_contexts_t& operator=(metal_contexts_t const&) = delete;

    /** The context of the Apple GPU listed at @p device , or null past the listing. */
    metal_context_t* context(int device) noexcept {
        return device >= 0 && static_cast<std::size_t>(device) < count ? &contexts[device] : nullptr;
    }
};

/** Allocates @p bytes of zeroed shared storage on @p context 's device; an empty buffer on
 *  exhaustion. Metal commits the pages lazily, so the bytes cost nothing until touched. */
inline metal_buffer_t metal_allocate(metal_context_t& context, std::size_t bytes) noexcept {
    metal_buffer_t buffer;
    if (!bytes) return buffer;
    // `MTLResourceStorageModeShared`
    constexpr std::size_t shared_storage_k = 0;
    metal_api_t const& api = *context.api;
    buffer.api = &api;
    buffer.handle = metal_send(api, context.device, "newBufferWithLength:options:", bytes, shared_storage_k);
    if (!buffer.handle) return buffer;
    buffer.host = static_cast<b8_t*>(metal_send(api, buffer.handle, "contents"));
    buffer.bytes = bytes;
    return buffer;
}

/** Waits for everything committed to @p queue through @p api : one empty command buffer, which
 *  completes only after every one committed before it, answering whether the device ran them. */
inline bool metal_synchronize(metal_api_t const& api, void* queue) noexcept {
    // `MTLCommandBufferStatusCompleted`
    constexpr std::size_t completed_k = 4;
    void* const pool = api.autorelease_pool_push();
    void* const commands = metal_send(api, queue, "commandBuffer");
    std::size_t state = 0;
    if (commands) {
        metal_send<void>(api, commands, "commit");
        metal_send<void>(api, commands, "waitUntilCompleted");
        state = metal_send<std::size_t>(api, commands, "status");
    }
    api.autorelease_pool_pop(pool);
    return state == completed_k;
}

/** Owned @c id<MTLCommandQueue> on one Apple GPU, the twin of a CUDA stream; null rides the
 *  device's default queue, which its @ref metal_context_t keeps, so the stream names its device
 *  and context. Move-only, and released when it dies. */
struct metal_stream_t {

    /** The Apple ordinal this stream submits to. */
    int device = 0;

    /** A stream on no device, which only fills the default of a table built later. */
    metal_stream_t() noexcept = default;

    /** Takes @p handle , owned and opened on @p device , whose context is @p context , or rides
     *  that device's default queue when null. */
    metal_stream_t(metal_context_t& context, int device, void* handle = nullptr) noexcept
        : device(device), context_(&context), handle_(handle) {}
    metal_stream_t(metal_stream_t&& other) noexcept
        : device(other.device), context_(other.context_), handle_(std::exchange(other.handle_, nullptr)) {}
    metal_stream_t(metal_stream_t const&) = delete;
    metal_stream_t& operator=(metal_stream_t const&) = delete;
    ~metal_stream_t() noexcept {
        if (handle_) metal_send<void>(*context_->api, handle_, "release");
    }

    /** The owned queue, or null for the device's default one. */
    void* handle() const noexcept { return handle_; }

    /** The context of the stream's device, or null for a stream on no device. */
    metal_context_t* context() const noexcept { return context_; }

  private:
    metal_context_t* context_ = nullptr;
    void* handle_ = nullptr;
};

/** One command buffer being encoded on one queue: binds buffers and bytes, dispatches, and commits
 *  without waiting. Every call to a device runs through one of these, so encoders live on the
 *  caller's stack and any number of threads encode at once; a committed one stays retained, and as
 *  a queue runs its command buffers in commit order, its finishing means every earlier one's. */
struct metal_launch_t {
    metal_api_t const* api = nullptr;
    void* commands = nullptr;
    void* encoder = nullptr;

    /** No command buffer - one with nothing left to run. */
    metal_launch_t() noexcept = default;

    /** Opens one command buffer on @p queue with a compute encoder for @p pipeline , through
     *  @p api . */
    metal_launch_t(metal_api_t const& api, void* queue, void* pipeline) noexcept : api(&api) {
        void* const pool = api.autorelease_pool_push();
        commands = metal_send(api, queue, "commandBuffer");
        if (commands) metal_send<void>(api, commands, "retain");
        if (commands) encoder = metal_send(api, commands, "computeCommandEncoder");
        if (encoder) {
            metal_send<void>(api, encoder, "retain");
            metal_send<void>(api, encoder, "setComputePipelineState:", pipeline);
        }
        // A command buffer without its encoder is never committed, so none is kept to await
        else if (commands) metal_send<void>(api, std::exchange(commands, nullptr), "release");
        api.autorelease_pool_pop(pool);
    }
    metal_launch_t(metal_launch_t&& other) noexcept
        : api(other.api), commands(std::exchange(other.commands, nullptr)),
          encoder(std::exchange(other.encoder, nullptr)) {}
    metal_launch_t& operator=(metal_launch_t&& other) noexcept {
        metal_launch_t dying {std::move(*this)};
        api = other.api;
        commands = std::exchange(other.commands, nullptr), encoder = std::exchange(other.encoder, nullptr);
        return *this;
    }
    metal_launch_t(metal_launch_t const&) = delete;
    metal_launch_t& operator=(metal_launch_t const&) = delete;
    ~metal_launch_t() noexcept {
        if (encoder) metal_send<void>(*api, encoder, "endEncoding"), metal_send<void>(*api, encoder, "release");
        if (commands) metal_send<void>(*api, commands, "release");
    }

    /** Whether the encoder is open, so the launch can still bind and dispatch. */
    explicit operator bool() const noexcept { return encoder != nullptr; }

    /** Binds @p buffer from @p offset at the kernel's buffer argument @p index . */
    void bind(metal_buffer_t const& buffer, std::size_t index, std::size_t offset = 0) noexcept {
        metal_send<void>(*api, encoder, "setBuffer:offset:atIndex:", buffer.handle, offset, index);
    }

    /** Copies @p bytes of @p arguments into the kernel's buffer argument @p index . */
    void bind_bytes(void const* arguments, // vendor-boundary: Metal copies a record by address and length
                    std::size_t bytes, std::size_t index) noexcept {
        metal_send<void>(*api, encoder, "setBytes:length:atIndex:", arguments, bytes, index);
    }

    /**
     *  @brief Dispatches @p groups threadgroups of @p threads each and commits, without waiting.
     *
     *  @par Statuses:
     *  - @c device_k The encoder is not open.
     */
    status_t submit(std::size_t groups, std::size_t threads) noexcept {
        return submit(metal_size_t {groups}, metal_size_t {threads});
    }

    /** Dispatches a grid of @p groups threadgroups of @p threads each and commits, without
     *  waiting; @c device_k when the encoder is not open. */
    status_t submit(metal_size_t groups, metal_size_t threads) noexcept {
        if (!encoder) return status_code_t::device_k;
        metal_send<void, metal_size_t, metal_size_t>(*api, encoder,
                                                     "dispatchThreadgroups:threadsPerThreadgroup:", groups, threads);
        metal_send<void>(*api, encoder, "endEncoding");
        metal_send<void>(*api, encoder, "release");
        encoder = nullptr;
        metal_send<void>(*api, commands, "commit");
        return status_code_t::success_k;
    }

    /** Waits for the committed command buffer and answers whether the device finished it; nothing
     *  committed has nothing to run. */
    status_t wait() noexcept {
        if (!commands || encoder) return status_code_t::success_k;
        // `MTLCommandBufferStatusCompleted`
        constexpr std::size_t completed_k = 4;
        metal_send<void>(*api, commands, "waitUntilCompleted");
        return metal_send<std::size_t>(*api, commands, "status") == completed_k ? status_code_t::success_k
                                                                                : status_code_t::device_k;
    }
};

/** The streams a call submits to and the scratch they run with, one of each per device. Work is
 *  committed, never waited on but by @ref wait , so one host thread drives every device; calls are
 *  claimed one at a time by the executor handle that owns it, as on every vendor. */
struct metal_executor_t {

    /** One device's run of scratch: one shared buffer, sized by @ref reserve . */
    using arena_t = metal_buffer_t;

    /** One device's stream and run. */
    struct stream_t {

        /** The ordering identity every launch on this device rides, naming the device. */
        metal_stream_t handle;

        /** The scratch a call reserves its activations in, sized by @ref reserve . */
        arena_t arena;

        /** Takes @p handle , whose run waits for @ref reserve . */
        explicit stream_t(metal_stream_t&& handle) noexcept : handle(std::move(handle)) {}
    };

    metal_executor_t() = default;
    metal_executor_t(metal_executor_t&& other) noexcept { swap(other); }
    metal_executor_t& operator=(metal_executor_t&& other) noexcept { return swap(other), *this; }
    metal_executor_t(metal_executor_t const&) = delete;
    metal_executor_t& operator=(metal_executor_t const&) = delete;

    void swap(metal_executor_t& other) noexcept {
        std::swap(streams_, other.streams_);
        std::swap(scratch_limit_, other.scratch_limit_);
    }

    /**
     *  @brief Opens one stream per entry of @p devices among @p contexts , whose scratch never
     *      grows past @p scratch_limit bytes, zero for no ceiling.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k The @p devices are empty.
     *  - @c out_of_memory_k The stream table cannot grow.
     *  - @c device_k A device is not listed or refused a queue.
     */
    static expected<metal_executor_t> spawn(metal_contexts_t& contexts, std::span<int const> devices,
                                            std::size_t scratch_limit = 0) noexcept {
        if (devices.empty()) return expected<metal_executor_t>::fail(status_code_t::invalid_argument_k, "No Apple GPU");
        metal_executor_t executor;
        if (st::failed(executor.streams_.reserve(devices.size())))
            return expected<metal_executor_t>::fail(status_code_t::out_of_memory_k, "Metal stream table");
        for (int const device : devices) {
            metal_context_t* const context = contexts.context(device);
            if (!context) return expected<metal_executor_t>::fail(status_code_t::device_k, "No Apple GPU {}", device);
            void* const queue = metal_send(*context->api, context->device, "newCommandQueue");
            if (!queue) return expected<metal_executor_t>::fail(status_code_t::device_k, "Apple GPU {} queue", device);
            [[maybe_unused]] st::status_t const kept = executor.streams_.push_back(
                st::assume_reserved, stream_t {metal_stream_t {*context, device, queue}});
        }
        executor.scratch_limit_ = scratch_limit;
        return executor;
    }

    /**
     *  @brief Reserves @p bytes_per_stream of scratch on every stream; the old run lives on in
     *      whichever command buffer still binds it.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k The executor has no streams.
     *  - @c out_of_memory_k @p bytes_per_stream pass the scratch ceiling or the device's memory.
     *  - @c device_k A stream's device is not listed.
     */
    status_t reserve(std::size_t bytes_per_stream) noexcept {
        if (streams_.empty()) return status_code_t::invalid_argument_k;
        if (scratch_limit_ && bytes_per_stream > scratch_limit_)
            return {status_code_t::out_of_memory_k, "Request passes the scratch ceiling"};
        for (stream_t& opened : streams_) {
            metal_context_t* const context = opened.handle.context();
            if (!context) return status_code_t::device_k;
            opened.arena = metal_allocate(*context, bytes_per_stream);
            if (bytes_per_stream && !opened.arena)
                return {status_code_t::out_of_memory_k, "No room for the stream arena"};
        }
        return status_code_t::success_k;
    }

    /**
     *  @brief Waits for every command buffer committed to every stream: one empty command buffer
     *      per stream, which completes only after every one committed before it.
     *
     *  @par Statuses:
     *  - @c device_k A stream's device is not listed, or failed its work.
     */
    status_t wait() noexcept {
        for (stream_t const& opened : streams_) {
            metal_stream_t const& stream = opened.handle;
            metal_context_t* const context = stream.context();
            if (!context || !metal_synchronize(*context->api, stream.handle() ? stream.handle() : context->queue))
                return status_code_t::device_k;
        }
        return status_code_t::success_k;
    }

    /** The stream at @p index , in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }

    /** The Apple ordinal stream @p index submits to. */
    int device(std::size_t index) const noexcept { return streams_[index].handle.device; }

    /** The first stream on @p device , or null when the executor drives none there. */
    stream_t* stream_on(int device) noexcept {
        for (stream_t& opened : streams_)
            if (opened.handle.device == device) return &opened;
        return nullptr;
    }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_.size(); }

  private:
    st::basic_vector<stream_t> streams_;

    /** The ceiling on every stream's scratch bytes; zero lets scratch grow as a call asks. */
    std::size_t scratch_limit_ = 0;
};

} // namespace unum::uform::utilities
