/**
 *  @file include/uform/utilities/vram_apple_device.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Host-side vocabulary for driving Apple GPUs: Metal opened at first use, the device
 *      contexts, shared buffers, launches, streams and the executor.
 *
 *  Metal's API is Objective-C, reached here through @c objc_msgSend cast to each call's exact
 *  prototype, so a plain C++ unit drives it with no SDK header. Nothing links Metal, Foundation,
 *  I/O Kit or the Objective-C runtime: @ref util::metal_api_t::load opens them and resolves the
 *  few C entry points UForm calls into one table, which stays empty where any is absent, so Apple
 *  then lists no devices and the binary still loads. Kernels travel as MSL source and compile on
 *  the device at first use, so no Metal toolchain is needed to build. Memory is unified and every
 *  buffer is shared, so the host reads and writes it in place. A device context records every
 *  buffer UForm allocates on it, and a launch makes the ones it touches resident.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - a block's address
#include <cstring> // `std::strcmp`, `std::strncpy`, `std::memcpy`, `std::memset`

#include <memory>  // `std::unique_ptr` - an executor's stream records
#include <span>    // `std::span` - the devices an executor opens over, a listing written
#include <utility> // `std::exchange`, `std::move`, `std::swap` - the owners' moves

#include <forkunion/atomics.hpp>             // `fu::standard_atomic_ref` - the executor's claim word
#include <forkunion/locks.hpp>               // `fu::fair_mutex_t` - a context's lock
#include <smashtable/basic_growing_pool.hpp> // `st::basic_growing_pool` - every buffer a device holds

#include "uform/utilities/blocking_files.hpp" // `shared_library_t` - Metal opened at first use
#include "uform/utilities/ram_device.hpp"     // `chunked_arena_t` - a stream's scratch
#include "uform/utilities/types.hpp"          // `expected`, `status_t`, `st::basic_vector`

namespace unum::uform::utilities {

/** The Objective-C runtime's selector, @c SEL , opaque to everyone but the runtime. */
using metal_selector_t = struct metal_selector_opaque_t*;

/** The simdgroup width every MSL source is written for; a pipeline of another width is refused. */
inline constexpr std::size_t metal_simdgroup_width_k = 32;

#pragma region Shared VRAM Driver

/** The Metal Shading Language versions an engine compiles its sources under, each spelled as its
 *  @c MTLLanguageVersion value. */
enum class metal_language_t : std::size_t {
    msl_3_1_k = (3u << 16) | 1u,
    msl_3_2_k = (3u << 16) | 2u,
    msl_4_0_k = (4u << 16) | 0u,
};

/**
 *  @brief The Metal, Objective-C runtime, I/O Kit and Core Foundation entry points every engine
 *      calls, resolved on first use from the system's own frameworks and @c libobjc.
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
    metal_selector_t (*register_selector)(char const*) = nullptr; // vendor-boundary
    void* (*get_class)(char const*) = nullptr;                    // vendor-boundary
    void* (*registry_entry_id_matching)(u64_t) = nullptr;
    u32_t (*get_matching_service)(u32_t, void*) = nullptr;
    void* (*search_registry_property)(u32_t, char const*, void const*, void const*, u32_t) = nullptr; // vendor-boundary
    int (*release_registry_object)(u32_t) = nullptr;
    void* (*create_string)(void const*, char const*, u32_t) = nullptr; // vendor-boundary
    unsigned char (*get_number_value)(void const*, long, void*) = nullptr;
    void (*release_reference)(void const*) = nullptr;
    void* (*autorelease_push)() = nullptr;
    void (*autorelease_pop)(void*) = nullptr;

    /** Whether every library loaded with every entry resolved. */
    explicit operator bool() const noexcept { return autorelease_pop != nullptr; }

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
            runtime.resolve("objc_autoreleasePoolPush", api.autorelease_push) &&
            runtime.resolve("objc_autoreleasePoolPop", api.autorelease_pop);
        return resolved ? api : metal_api_t {};
    }
};

/** Sends @p selector to @p target with @p arguments through @p api, cast to the exact prototype the
 *  call has. */
template <typename result_type_ = void*, typename... argument_types_>
inline result_type_ metal_send(metal_api_t const& api, void* target, char const* selector, // vendor-boundary
                               argument_types_... arguments) noexcept {
    using function_t = result_type_ (*)(void*, metal_selector_t, argument_types_...);
    return reinterpret_cast<function_t>(api.send)(target, api.register_selector(selector), arguments...);
}

/** An @c NSString over @p text, autoreleased. */
inline void* metal_string(metal_api_t const& api, char const* text) noexcept { // vendor-boundary
    return metal_send(api, reinterpret_cast<void*>(api.get_class("NSString")), "stringWithUTF8String:", text);
}

/** One library of MSL source compiled for one Apple device and released with this owner; kernels
 *  are found by name, each built into a pipeline. */
struct metal_module_t {
    using function_t = void*;

    metal_module_t() noexcept = default;
    metal_module_t(metal_module_t&& other) noexcept
        : api_(other.api_), device_(other.device_), library_(std::exchange(other.library_, nullptr)) {}
    metal_module_t& operator=(metal_module_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(device_, other.device_), std::swap(library_, other.library_);
        return *this;
    }
    metal_module_t(metal_module_t const&) = delete;
    metal_module_t& operator=(metal_module_t const&) = delete;
    ~metal_module_t() noexcept {
        if (library_) metal_send<void>(*api_, library_, "release");
    }

    /**
     *  @brief Compiles @p source, NUL-terminated MSL text, under @p language on @p device, which
     *      is the @c id<MTLDevice> the library runs on.
     *
     *  @par Statuses:
     *  - @c device_k The @p source is empty or unterminated, or the compiler rejected it.
     */
    static expected<metal_module_t> load(metal_api_t const& api, void* device, bytes_view_t source,
                                         metal_language_t language) noexcept {
        if (source.empty() || source.back() != byte_t {0} || !api || !device)
            return {{}, {status_code_t::device_k, "The device refused the MSL source"}};
        metal_module_t module;
        module.api_ = &api;
        module.device_ = device;
        void* const autoreleased = api.autorelease_push();
        void* const options = metal_send(api, metal_send(api, api.get_class("MTLCompileOptions"), "alloc"), "init");
        metal_send<void>(api, options, "setLanguageVersion:", static_cast<std::size_t>(language));
        void* error = nullptr;
        module.library_ = metal_send(api, device, "newLibraryWithSource:options:error:",
                                     metal_string(api, reinterpret_cast<char const*>(source.data())), options, &error);
        metal_send<void>(api, options, "release");
        api.autorelease_pop(autoreleased);
        if (!module.library_) return {{}, {status_code_t::device_k, "The device refused the MSL source"}};
        return {std::move(module), status_code_t::success_k};
    }

    /** The table the library was compiled through, which launches of its kernels go through. */
    metal_api_t const& api() const noexcept { return *api_; }

    /** The pipeline of the kernel exported as @p name, owned by the caller, or null when the
     *  library has none or its simdgroups are not the @c metal_simdgroup_width_k lanes every MSL
     *  source is written against. */
    function_t function(char const* name) const noexcept { // vendor-boundary
        if (!library_) return nullptr;
        metal_api_t const& api = *api_;
        void* const autoreleased = api.autorelease_push();
        void* const function = metal_send(api, library_, "newFunctionWithName:", metal_string(api, name));
        void* pipeline = nullptr;
        if (function) {
            void* error = nullptr;
            pipeline = metal_send(api, device_, "newComputePipelineStateWithFunction:error:", function, &error);
            metal_send<void>(api, function, "release");
        }
        api.autorelease_pop(autoreleased);
        if (pipeline && metal_send<std::size_t>(api, pipeline, "threadExecutionWidth") != metal_simdgroup_width_k)
            metal_send<void>(api, std::exchange(pipeline, nullptr), "release");
        return pipeline;
    }

  private:
    metal_api_t const* api_ = nullptr;
    void* device_ = nullptr;
    void* library_ = nullptr;
};

#pragma endregion Shared VRAM Driver

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

/** One listed Apple GPU's process-wide state: the table its calls go through, the device, its
 *  default queue, its family and cores, and the shared buffers allocated on it; its pipelines live
 *  in the GPU runtime's Metal kernel tables. */
struct metal_context_t {
    metal_api_t const* api = nullptr;
    void* device = nullptr;
    void* queue = nullptr;

    /** The newest Apple GPU family the device supports, such as 9 for @c MTLGPUFamilyApple9 ; zero
     *  for a GPU older than Apple7 or not Apple's own. */
    u32_t family = 0;

    /** The GPU cores the I/O Registry lists for the device, zero when it lists none. */
    u32_t multiprocessor_count = 0;

    fu::fair_mutex_t lock;

    /** Every shared buffer UForm allocated on the device, guarded by @ref lock : a free finds the
     *  buffer its address names, a launch turns each host address it binds into the buffer's GPU
     *  address and makes that buffer resident, and a copy tells device memory from host memory. */
    st::basic_growing_pool<metal_buffer_t> buffers;
};

/** Every Apple GPU Metal lists, in Metal's own order, opened once per process by its caller and
 *  kept for it, as Metal's own device objects are; a device's Apple ordinal is its position here,
 *  NVIDIA and AMD devices coming before Apple ones in every UForm listing. */
struct metal_contexts_t {
    metal_context_t contexts[gpu_devices_capacity_k];
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
        void* const autoreleased = api.autorelease_push();
        std::size_t const listed_count = metal_send<std::size_t>(api, listed, "count");
        for (std::size_t position = 0; position != listed_count && count != gpu_devices_capacity_k; ++position) {
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
        api.autorelease_pop(autoreleased);
        metal_send<void>(api, listed, "release");
    }
    metal_contexts_t(metal_contexts_t const&) = delete;
    metal_contexts_t& operator=(metal_contexts_t const&) = delete;

    /** The context of the Apple GPU listed at @p device , or null past the listing. */
    metal_context_t* context(int device) noexcept {
        return device >= 0 && static_cast<std::size_t>(device) < count ? &contexts[device] : nullptr;
    }
};

/** Writes every Apple GPU @p opened holds into @p into , up to its size, in Metal's order, and
 *  answers how many it wrote; simdgroup lanes stay zero, as only a built pipeline reports them. */
inline std::size_t list_apple_gpu_devices(metal_contexts_t const& opened, std::span<gpu_device_t> into) noexcept {
    std::size_t count = 0;
    for (std::size_t ordinal = 0; ordinal != opened.count && count != into.size(); ++ordinal) {
        metal_context_t const& context = opened.contexts[ordinal];
        metal_api_t const& api = *context.api;
        gpu_device_t& listed = into[count++];
        listed = {gpu_vendor_t::apple_k, static_cast<u32_t>(ordinal), context.family};
        listed.compute_units = context.multiprocessor_count;
        listed.memory_bytes = metal_send<u64_t>(api, context.device, "recommendedMaxWorkingSetSize");
        void* const autoreleased = api.autorelease_push();
        auto const name = metal_send<char const*>(api, metal_send(api, context.device, "name"), "UTF8String");
        if (name) std::strncpy(listed.name, name, sizeof(listed.name) - 1);
        api.autorelease_pop(autoreleased);
    }
    return count;
}

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
    void* const autoreleased = api.autorelease_push();
    void* const commands = metal_send(api, queue, "commandBuffer");
    std::size_t state = 0;
    if (commands) {
        metal_send<void>(api, commands, "commit");
        metal_send<void>(api, commands, "waitUntilCompleted");
        state = metal_send<std::size_t>(api, commands, "status");
    }
    api.autorelease_pop(autoreleased);
    return state == completed_k;
}

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
        void* const autoreleased = api.autorelease_push();
        commands = metal_send(api, queue, "commandBuffer");
        if (commands) metal_send<void>(api, commands, "retain");
        if (commands) encoder = metal_send(api, commands, "computeCommandEncoder");
        if (encoder) {
            metal_send<void>(api, encoder, "retain");
            metal_send<void>(api, encoder, "setComputePipelineState:", pipeline);
        }
        // A command buffer without its encoder is never committed, so none is kept to await
        else if (commands) metal_send<void>(api, std::exchange(commands, nullptr), "release");
        api.autorelease_pop(autoreleased);
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

#pragma region Scratch

/** Where a metal stream's launches find their functions: the process's one table of UForm's bundle
 *  per device, which the GPU runtime owns. */
using metal_kernels_t = kernel_tables<metal_module_t>*;

/** Registers @p buffer with @p context , so launches resolve its addresses and a free finds it.
 *  @c out_of_memory_k when the registry is full, which releases the buffer. */
inline status_t metal_register_(metal_context_t& context, metal_buffer_t&& buffer) noexcept {
    st::unique_lock<fu::fair_mutex_t> guard {context.lock};
    auto const index = context.buffers.take();
    if (index == context.buffers.absent_k) return {status_code_t::out_of_memory_k, "The Metal buffer registry is full"};
    context.buffers[index] = std::move(buffer);
    return {};
}

/**
 *  @brief Allocates @p bytes of shared storage on @p context 's device and registers it, answering
 *      its @c contents pointer, which the host reads and writes in place and a launch resolves to
 *      the buffer's GPU address through the registry.
 *
 *  @par Statuses:
 *  - @c out_of_memory_k The device refused the buffer, or the registry is full.
 */
inline expected<std::uintptr_t> metal_allocate_block(metal_context_t& context, std::size_t bytes) noexcept {
    metal_buffer_t buffer = metal_allocate(context, bytes);
    if (!buffer) return {{}, {status_code_t::out_of_memory_k, "The device refused a buffer"}};
    std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(buffer.host);
    if (status_t const registered = metal_register_(context, std::move(buffer)); registered.failed())
        return {{}, registered};
    return {std::uintptr_t {address}, status_code_t::success_k};
}

/** Releases the registered buffer @p address starts, its GPU or its host address; committed work
 *  that binds it keeps it alive until it completes. */
inline void metal_free_block(metal_context_t& context, std::uintptr_t address) noexcept {
    metal_buffer_t dying;
    st::unique_lock<fu::fair_mutex_t> guard {context.lock};
    for (metal_buffer_t& buffer : context.buffers.records()) {
        if (!buffer) continue;
        if (reinterpret_cast<std::uintptr_t>(buffer.host) != address &&
            metal_send<u64_t>(*buffer.api, buffer.handle, "gpuAddress") != address)
            continue;
        context.buffers.vacate(context.buffers.index_of(buffer));
        dying = std::move(buffer);
        return;
    }
}

/**
 *  @brief UForm's memory on one Apple stream: the activations a call computes in and the transfers
 *      it packs what the device reads into, both shared buffers of the stream's device handed out
 *      by their @c contents pointers.
 *
 *  Both only hand out what @ref reserve grew them by before a call allocates, and @ref release
 *  frees them after the stream drained.
 */
struct metal_scratch_t {

    /** Shared buffers the call's layers compute in, handed out by their @c contents pointers. */
    chunked_arena_t activations;

    /** Shared buffers the host packs what the device reads into, handed out by host address. */
    chunked_arena_t transfers;

    /** The ceiling on @ref activations , zero for none, which every @ref reserve checks. */
    std::size_t limit = 0;

    /** Points @p record at the bundle table of @p runtime its launches find their functions in,
     *  and takes the executor's ceiling on activations, zero for none; the arenas start empty and
     *  grow at the first @ref reserve . */
    template <typename record_type_, typename runtime_type_>
    status_t open(record_type_& record, runtime_type_& runtime, std::size_t& scratch_limit) noexcept {
        record.kernels = &runtime.metal_kernels;
        limit = scratch_limit;
        return {};
    }

    /**
     *  @brief Rewinds both arenas and makes each hold a call's @p activation_bytes and
     *      @p transfer_bytes , as the allocations' @ref chunked_arena_t::footprint values sum.
     *
     *  After the rewind nothing either arena holds is live, so one that falls short frees its
     *  buffers and takes one sized to the call: the activations never hold more than the
     *  @ref limit .
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The activations pass the @ref limit , the device refused a buffer, or a
     *    chunk table or the registry could not grow.
     */
    template <typename stream_type_>
    status_t reserve(stream_type_& stream, std::size_t activation_bytes, std::size_t transfer_bytes = 0) noexcept {
        if (limit && activation_bytes > limit)
            return {status_code_t::out_of_memory_k, "The executor's scratch limit is too low"};
        activations.rewind(), transfers.rewind();
        metal_context_t* const context = stream.handle.context();
        if (!context) return {status_code_t::device_k, "The stream names no device"};
        if (activations.room() < activation_bytes) {
            free_buffers_(*context, activations);
            auto [address, allocate_status] = stream.allocate_async(activation_bytes);
            if (allocate_status.failed()) return allocate_status;
            if (status_t const appended = activations.append({reinterpret_cast<b8_t*>(address), activation_bytes});
                appended.failed())
                return stream.free_async(address), appended;
        }
        if (transfers.room() < transfer_bytes) {
            free_buffers_(*context, transfers);
            auto [address, allocate_status] = metal_allocate_block(*context, transfer_bytes);
            if (allocate_status.failed()) return allocate_status;
            if (status_t const appended = transfers.append({reinterpret_cast<b8_t*>(address), transfer_bytes});
                appended.failed())
                return metal_free_block(*context, address), appended;
        }
        return {};
    }

    /** Frees every buffer of both arenas once @p stream drained. */
    template <typename stream_type_>
    void release(stream_type_& stream) noexcept {
        if (metal_context_t* const context = stream.handle.context())
            free_buffers_(*context, activations), free_buffers_(*context, transfers);
        activations.forget(), transfers.forget();
    }

  private:
    /** Frees every buffer of @p arena on @p context 's device, which the last call's wait left
     *  unread. */
    static void free_buffers_(metal_context_t& context, chunked_arena_t& arena) noexcept {
        for (bytes_span_t const chunk : arena.chunks())
            metal_free_block(context, reinterpret_cast<std::uintptr_t>(chunk.data()));
        arena.forget();
    }
};

#pragma endregion Scratch

#pragma region Shared VRAM Executor

/** Owned @c id<MTLCommandQueue> on one Apple GPU, the twin of @c cuda_stream_t; null rides the
 *  device's default queue, which its @ref metal_context_t keeps, so the stream names its device
 *  and context. Move-only, and released when it dies. */
struct metal_stream_t {

    /** A stream on no device, which only fills the default of a record built later. */
    metal_stream_t() noexcept = default;

    /** Takes @p handle, owned and opened on the device of @p context, or rides that device's
     *  default queue when null. */
    explicit metal_stream_t(metal_context_t& context, void* handle = nullptr) noexcept
        : context_(&context), handle_(handle) {}
    metal_stream_t(metal_stream_t&& other) noexcept
        : context_(other.context_), handle_(std::exchange(other.handle_, nullptr)) {}
    metal_stream_t& operator=(metal_stream_t&& other) noexcept {
        std::swap(context_, other.context_), std::swap(handle_, other.handle_);
        return *this;
    }
    metal_stream_t(metal_stream_t const&) = delete;
    metal_stream_t& operator=(metal_stream_t const&) = delete;
    ~metal_stream_t() noexcept {
        if (handle_) metal_send<void>(*context_->api, handle_, "release");
    }

    /**
     *  @brief Opens a command queue on the device of @p context, owned from now on.
     *
     *  @par Statuses:
     *  - @c device_k The device refused the queue.
     */
    static expected<metal_stream_t> open(metal_context_t& context) noexcept {
        void* const queue = metal_send(*context.api, context.device, "newCommandQueue");
        if (!queue) return {{}, {status_code_t::device_k, "Device refused a command queue"}};
        return {metal_stream_t {context, queue}, status_code_t::success_k};
    }

    /**
     *  @brief Checks that @p handle, a caller's @c id<MTLCommandQueue>, may be adopted on the
     *      device of @p context, without retaining it.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k The @p handle is null.
     *  - @c config_mismatch_k The queue sits on another device.
     */
    static status_t adoptable(metal_context_t const& context, void* handle) noexcept {
        if (!handle) return {status_code_t::invalid_argument_k, "A default stream cannot be adopted"};
        if (metal_send(*context.api, handle, "device") != context.device)
            return {status_code_t::config_mismatch_k, "A stream sits on another device"};
        return {};
    }

    /** The owned queue, or null for the device's default one. */
    void* owned() const noexcept { return handle_; }

    /** The queue work is committed to: the owned one, or the device's default one. */
    void* queue() const noexcept { return handle_ ? handle_ : context_->queue; }

    /** The context of the stream's device, or null for a stream on no device. */
    metal_context_t* context() const noexcept { return context_; }

  private:
    metal_context_t* context_ = nullptr;
    void* handle_ = nullptr;
};

/**
 *  @brief The queues a call commits to, one per entry of the devices it was spawned over, in the
 *      caller's order, each with its kernels and the engine's memory.
 *
 *  Work is committed, never waited on, so one host thread drives every device. Each stream record
 *  is built in place on the heap and never moves; it dies waiting for its queue, then releasing
 *  its memory, then releasing its queue. Apple's memory is unified, so a copy or a fill is a host
 *  one, done when it returns.
 */
struct metal_executor_t {

    /** The engine's memory on one stream. */
    using scratch_t = metal_scratch_t;

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

    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::apple_k;

    /** One device's queue, the kernels it launches, and the engine's memory, declared last so it
     *  dies first. */
    struct stream_t {

        /** The Apple ordinal of the device this stream submits to, as the GPU runtime lists it. */
        std::size_t ordinal = 0;

        /** Opened by the executor, adopted from the caller, or the device's default queue. */
        metal_stream_t handle;

        /** The kernels this stream launches, as the engine resolves them when the record opens. */
        metal_kernels_t kernels = {};

        /** The engine's memory, safe to reuse only in this stream's queue order, opened once this
         *  record sits in the executor. */
        scratch_t memory;

        /** Takes @p opened as the queue of device @p device. */
        stream_t(std::size_t device, metal_stream_t&& opened) noexcept : ordinal(device), handle(std::move(opened)) {}
        stream_t(stream_t const&) = delete;
        stream_t& operator=(stream_t const&) = delete;

        /** Waits for an owned queue, then releases the memory; the queue is released after. A
         *  record owning none, a placeholder or the device's default queue, waits for nothing. */
        ~stream_t() noexcept {
            [[maybe_unused]] status_t const ready = handle.owned() ? wait() : bind();
            memory.release(*this);
        }

        /**
         *  @brief Checks the stream names a device, which Metal needs no thread binding for.
         *
         *  @par Statuses:
         *  - @c device_k The stream names no device.
         */
        status_t bind() const noexcept {
            if (!handle.context()) return {status_code_t::device_k, "The stream names no device"};
            return {};
        }

        /**
         *  @brief Waits for every command buffer committed so far, committing an empty one, which
         *      a queue completes only after every one committed before it.
         *
         *  @par Statuses:
         *  - @c device_k The stream names no device, or the device failed the work.
         */
        status_t wait() noexcept {
            // The state of a command buffer the device ran, `MTLCommandBufferStatusCompleted`.
            constexpr std::size_t completed_k = 4;
            metal_context_t const* const context = handle.context();
            if (!context) return {status_code_t::device_k, "The stream names no device"};
            metal_api_t const& api = *context->api;
            void* const autoreleased = api.autorelease_push();
            void* const commands = metal_send(api, handle.queue(), "commandBuffer");
            std::size_t state = 0;
            if (commands) {
                metal_send<void>(api, commands, "commit");
                metal_send<void>(api, commands, "waitUntilCompleted");
                state = metal_send<std::size_t>(api, commands, "status");
            }
            api.autorelease_pop(autoreleased);
            if (state != completed_k) return {status_code_t::device_k, "The device failed the committed work"};
            return {};
        }

        /**
         *  @brief Orders this stream after everything @p other committed so far. Metal has no
         *      event entry here, so it waits on @p other from the host, and the follow is done
         *      once it returns.
         *
         *  @par Statuses:
         *  - @c device_k @p other names no device, or the device failed its work.
         */
        status_t follow(stream_t& other) noexcept { return other.wait(); }

        /**
         *  @brief Allocates @p bytes of shared storage on this stream's device through its buffer
         *      registry, answering their host address, which kernels reach too.
         *
         *  @par Statuses:
         *  - @c out_of_memory_k The device refused the allocation.
         *  - @c unsupported_k The engine keeps no buffer registry.
         */
        expected<std::uintptr_t> allocate_async(std::size_t bytes) noexcept {
            return metal_allocate_block(*handle.context(), bytes);
        }

        /** Releases @p address, which @ref allocate_async answered; work still committed on it
         *  keeps it alive until it finishes. */
        void free_async(std::uintptr_t address) noexcept { metal_free_block(*handle.context(), address); }

        /** Copies @p bytes from @p source to @p target in unified memory; done when it returns. */
        status_t copy_async(std::uintptr_t target, std::uintptr_t source, std::size_t bytes) noexcept {
            std::memcpy(reinterpret_cast<void*>(target), reinterpret_cast<void const*>(source), bytes);
            return {};
        }

        /** Copies @p source into memory at @p target, as @ref copy_async does. */
        status_t upload_async(std::uintptr_t target, bytes_view_t source) noexcept {
            return copy_async(target, reinterpret_cast<std::uintptr_t>(source.data()), source.size());
        }

        /** Copies memory at @p source into @p target, as @ref copy_async does. */
        status_t download_async(bytes_span_t target, std::uintptr_t source) noexcept {
            return copy_async(reinterpret_cast<std::uintptr_t>(target.data()), source, target.size());
        }

        /** Sets @p bytes at @p target to @p value in unified memory; done when it returns. */
        status_t fill_async(std::uintptr_t target, byte_t value, std::size_t bytes) noexcept {
            std::memset(reinterpret_cast<void*>(target), static_cast<int>(value), bytes);
            return {};
        }
    };

    metal_executor_t() noexcept = default;
    metal_executor_t(metal_executor_t&& other) noexcept { swap(other); }
    metal_executor_t& operator=(metal_executor_t&& other) noexcept { return swap(other), *this; }
    metal_executor_t(metal_executor_t const&) = delete;
    metal_executor_t& operator=(metal_executor_t const&) = delete;

    /** Exchanges the stream records and the ceiling, none of which moves in memory; the hold
     *  stays, since a move with a call in flight is misuse. */
    void swap(metal_executor_t& other) noexcept {
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
     *  @brief Spawns one command queue per entry of @p devices among @p contexts, or retains the
     *      @c id<MTLCommandQueue> of @p streams beside it.
     *
     *  Each stream's memory and kernels open with @p memory_arguments. The executor releases its
     *  own reference to a retained queue when it dies, while the caller keeps and releases theirs.
     *  Adoption follows every fallible step, so a refused spawn retains none of @p streams.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k No device is named, more than a listing holds, the @p streams count
     *    differs from theirs, or a queue is null.
     *  - @c config_mismatch_k A queue sits on another device.
     *  - @c device_k A device is not listed, or refused its queue.
     *  - @c out_of_memory_k A stream record or its memory could not be allocated.
     */
    template <typename... memory_arguments_>
    static expected<metal_executor_t> spawn(metal_contexts_t& contexts, std::span<int const> devices,
                                            std::span<void* const> streams,
                                            memory_arguments_&... memory_arguments) noexcept {
        if (!streams.empty() && streams.size() != devices.size())
            return {{}, {status_code_t::invalid_argument_k, "One stream per device"}};
        for (std::size_t index = 0; index != streams.size(); ++index) {
            metal_context_t* const context = contexts.context(devices[index]);
            if (!context) return {{}, {status_code_t::device_k, "Device is not listed"}};
            if (status_t const checked = metal_stream_t::adoptable(*context, streams[index]); checked.failed())
                return {{}, checked};
        }
        // Adopted queues wait on placeholders that own nothing until every fallible step is behind.
        auto const placeholder = [](metal_context_t& context) noexcept -> expected<metal_stream_t> {
            return {metal_stream_t {context}, status_code_t::success_k};
        };
        auto [executor, build_status] = build(contexts, devices, streams.empty() ? &metal_stream_t::open : +placeholder,
                                              memory_arguments...);
        if (build_status.failed()) return {{}, build_status};
        // Every fallible step is behind, so a refusal above retained none of the caller's queues.
        for (std::size_t index = 0; index != streams.size(); ++index) {
            metal_context_t& context = *contexts.context(devices[index]);
            executor.streams_[index]->handle = metal_stream_t {context,
                                                               metal_send(*context.api, streams[index], "retain")};
        }
        return {std::move(executor), status_code_t::success_k};
    }

    /** Waits for every command buffer committed to every stream, the fleet's completion. */
    status_t wait() noexcept {
        for (std::size_t index = 0; index != streams_count_; ++index)
            if (status_t const drained = streams_[index]->wait(); drained.failed()) return drained;
        return {};
    }

    /** The stream at @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return *streams_[index]; }
    stream_t const& stream(std::size_t index) const noexcept { return *streams_[index]; }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_count_; }

    /** Builds one stream record per entry of @p devices over the queue @p open answers there,
     *  and opens its memory and kernels in place. */
    template <typename... memory_arguments_>
    static expected<metal_executor_t> build(metal_contexts_t& contexts, std::span<int const> devices,
                                            expected<metal_stream_t> (*open)(metal_context_t&),
                                            memory_arguments_&... memory_arguments) noexcept {
        if (devices.empty() || devices.size() > gpu_devices_capacity_k)
            return {{}, {status_code_t::invalid_argument_k, "An executor names one to a listing's devices"}};
        metal_executor_t executor;
        for (int const device : devices) {
            metal_context_t* const context = contexts.context(device);
            if (!context) return {{}, {status_code_t::device_k, "Device is not listed"}};
            auto [opened, open_status] = open(*context);
            if (open_status.failed()) return {{}, open_status};
            std::unique_ptr<stream_t>& record = executor.streams_[executor.streams_count_];
            record.reset(new (std::nothrow) stream_t {static_cast<std::size_t>(device), std::move(opened)});
            if (!record) return {{}, {status_code_t::out_of_memory_k, "No room for a stream record"}};
            ++executor.streams_count_;
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
