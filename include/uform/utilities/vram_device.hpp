/**
 *  @file include/uform/utilities/vram_device.hpp
 *  @author Ash Vardanian
 *  @date November 26, 2026
 *  @brief The vendor-neutral face of the GPU residency: every vendor's driver, the device
 *      listing, and one executor over every vendor.
 *
 *  One binary drives NVIDIA, AMD and Apple GPUs at once, each vendor through a runtime opened once
 *  and declared without its SDK: NVIDIA devices are listed first, then AMD, then Apple, and a
 *  vendor whose runtime is absent contributes no devices. The driver tables, the stream owners, the
 *  executors, the concepts and the spawn sit in the Shared VRAM regions, which every sibling engine
 *  carries word for word, and @c selftest @c --siblings compares.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - a stream's addresses
#include <cstring> // `std::strlen`, `std::strcmp` - a Metal kernel's source and name

#include <concepts>    // `std::same_as`, `std::convertible_to` - the stream and executor concepts
#include <span>        // `std::span` - the listing written
#include <type_traits> // `std::type_identity` - the driver of one vendor's executor
#include <variant>     // `std::variant`, `std::monostate` - one alternative per vendor

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - a device's load stage
#include <forkunion/locks.hpp>   // `fu::fair_mutex_t` - a Metal compile
#include <forkunion/types.hpp>   // `fu::standard_yield_t`, `fu::wait_capped_k`

#include "uform/utilities/types.hpp"              // `gpu_device_t`
#include "uform/utilities/vram_amd_device.hpp"    // `list_amd_gpu_devices`, `hip_executor_t`
#include "uform/utilities/vram_apple_device.hpp"  // `list_apple_gpu_devices`, `metal_executor_t`
#include "uform/utilities/vram_nvidia_device.hpp" // `list_nvidia_gpu_devices`, `cuda_executor_t`

namespace unum::uform::utilities {

#pragma region Kernels

/** How far one device got loading one vendor's bundle; whichever thread leaves @c unloaded_k
 *  loads it. */
enum class module_stage_t : u8_t {
    unloaded_k,
    loading_k,
    loaded_k,
};

/** Kernels one bundle exports at most, so a device's table needs no heap. */
inline constexpr std::size_t bundle_kernels_capacity_k = 16;

/**
 *  @brief One vendor's bundle as every device of the process loaded it, at the first launch there,
 *      and kept for the life of the process, as the runtime keeps a registered fatbin.
 *
 *  Each device's stage is touched through the standard reference, since its readers name no CPU
 *  class; a caller meeting another's load waits until it ends.
 */
template <typename module_type_>
struct kernel_tables {
    using function_t = typename module_type_::function_t;

    /** One device's load: its stage, its outcome, the module and every kernel's function. */
    struct device_t {
        module_stage_t stage = module_stage_t::unloaded_k;
        status_t status {status_code_t::uninitialized_k, "The bundle never loaded"};
        module_type_ module;
        function_t functions[bundle_kernels_capacity_k] = {};
    };

    device_t devices[gpu_devices_capacity_k];

    /**
     *  @brief The functions of the bundle on device @p ordinal into @p into , which @p load fills
     *      once into the module and the functions it is handed.
     *
     *  @par Statuses:
     *  - @c device_k The ordinal passes the listing, or the bundle would not load there.
     */
    template <typename load_type_>
    status_t on(std::size_t ordinal, load_type_ const& load, std::span<function_t const>& into) noexcept {
        if (ordinal >= gpu_devices_capacity_k) return {status_code_t::device_k, "No such device"};
        device_t& device = devices[ordinal];
        fu::standard_atomic_ref<module_stage_t> const stage {device.stage};
        module_stage_t unloaded = module_stage_t::unloaded_k;
        if (stage.compare_exchange_strong(unloaded, module_stage_t::loading_k, std::memory_order_acquire)) {
            device.status = load(device.module, std::span<function_t> {device.functions});
            stage.store(module_stage_t::loaded_k, std::memory_order_release);
        }
        while (stage.load(std::memory_order_acquire) != module_stage_t::loaded_k)
            fu::standard_yield_t {}(&device.stage, module_stage_t::loading_k, std::size_t {0}, fu::wait_capped_k);
        if (device.status.failed()) return device.status;
        into = device.functions;
        return {};
    }
};

/**
 *  @brief The Metal pipelines every Apple device of the process built, by name, under each MSL
 *      version, compiled at the first launch that needs one and kept for the life of the process.
 *
 *  A lookup reads the published pipelines with no lock; a miss takes the fair mutex, builds the
 *  library of its device, source and version once, builds the pipeline and publishes it with a
 *  release store of the count, as a module load serializes in the CUDA and HIP drivers.
 */
template <>
struct kernel_tables<metal_module_t> {

    /** One built entry point: where it was built, its name, and its state as launches read it. */
    struct pipeline_t {
        std::size_t device = 0;
        char const* source = nullptr; // vendor-boundary: the MSL source compiled
        metal_language_t language = metal_language_t::msl_3_1_k;
        char const* name = nullptr; // vendor-boundary: the literal the launch spelled, compared whole
        metal_pipeline_state_t state;
    };

    /** One compiled library: the device, the source and the version it was built under. */
    struct library_t {
        std::size_t device = 0;
        char const* source = nullptr; // vendor-boundary: the MSL source compiled
        metal_language_t language = metal_language_t::msl_3_1_k;
        metal_module_t module;
    };

    /** Pipelines and libraries one process holds at most, far past a few devices' kernels. */
    inline static constexpr std::size_t pipelines_capacity_k = 1024, libraries_capacity_k = 64;

    /**
     *  @brief Kernel @p name 's pipeline on device @p device of @p context from @p source under
     *      @p language into @p into , built on first use.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The tables are full.
     *  - @c device_k The device refused the source or has no such kernel.
     *  - @c unsupported_k The device's simdgroups are not @c metal_simdgroup_width_k lanes wide.
     */
    status_t pipeline(metal_context_t const& context, std::size_t device,
                      char const* source, // vendor-boundary: Metal reads C strings
                      char const* name,   // vendor-boundary: Metal reads C strings
                      metal_language_t language, metal_pipeline_state_t& into) noexcept {
        if (find_(device, source, name, language, into)) return {};
        st::unique_lock<fu::fair_mutex_t> guard {compiling_};
        if (find_(device, source, name, language, into)) return {};
        std::size_t const published = pipelines_count_;
        if (published == pipelines_capacity_k) return {status_code_t::out_of_memory_k, "The Metal pipelines are full"};
        metal_module_t const* library = nullptr;
        for (std::size_t index = 0; index != libraries_count_ && !library; ++index)
            if (libraries_[index].device == device && libraries_[index].source == source &&
                libraries_[index].language == language)
                library = &libraries_[index].module;
        if (!library) {
            if (libraries_count_ == libraries_capacity_k)
                return {status_code_t::out_of_memory_k, "The Metal libraries are full"};
            library_t& built = libraries_[libraries_count_];
            bytes_view_t const text {reinterpret_cast<b8_t const*>(source), std::strlen(source) + 1};
            auto [module, load_status] = metal_module_t::load(*context.api, context.device, text, language);
            if (load_status.failed()) return load_status;
            built.module = std::move(module);
            built.device = device, built.source = source, built.language = language;
            library = &built.module;
            ++libraries_count_;
        }
        // An Apple GPU's simdgroups are 32 lanes, so its null names a missing kernel; another
        // vendor's GPU under Metal answers null for its simdgroup width
        void* const built = library->function(name);
        if (!built && context.family) return {status_code_t::device_k, "The device has no such Metal kernel"};
        if (!built) return {status_code_t::unsupported_k, "The device's simdgroups are not 32 lanes wide"};
        pipeline_t& entry = pipelines_[published];
        entry.device = device, entry.source = source, entry.language = language, entry.name = name;
        entry.state = {
            built, static_cast<u32_t>(metal_send<std::size_t>(*context.api, built, "maxTotalThreadsPerThreadgroup"))};
        fu::standard_atomic_ref<std::size_t> {pipelines_count_}.store(published + 1, std::memory_order_release);
        into = entry.state;
        return {};
    }

  private:
    /** Whether a published pipeline matches, which it then writes to @p into . */
    bool find_(std::size_t device, char const* source, char const* name, // vendor-boundary: C strings
               metal_language_t language, metal_pipeline_state_t& into) const noexcept {
        std::size_t const published = fu::standard_atomic_ref<std::size_t const> {pipelines_count_}.load(
            std::memory_order_acquire);
        for (std::size_t index = 0; index != published; ++index) {
            pipeline_t const& entry = pipelines_[index];
            if (entry.device == device && entry.source == source && entry.language == language &&
                std::strcmp(entry.name, name) == 0)
                return into = entry.state, true;
        }
        return false;
    }

    pipeline_t pipelines_[pipelines_capacity_k];
    std::size_t pipelines_count_ = 0;
    library_t libraries_[libraries_capacity_k];
    std::size_t libraries_count_ = 0;
    fu::fair_mutex_t compiling_;
};

#pragma endregion Kernels

/** UForm's optional driver entries beyond the shared tables: none, as UForm calls only core
 *  entries. */
struct gpu_driver_extensions_t {};

#pragma region Shared VRAM

/** Every vendor's driver table, Apple's device contexts and the engine's optional entries, which a
 *  caller builds once per process and hands to every GPU call; a vendor whose runtime is absent
 *  keeps an empty table. */
struct gpu_drivers_t {
    cuda_api_t cuda;
    hip_api_t hip;
    metal_api_t metal;
    metal_contexts_t metal_contexts;

    /** The engine's optional entries, resolved apart from the tables above. */
    gpu_driver_extensions_t extensions;

    gpu_drivers_t() noexcept
        : cuda(cuda_api_t::load()), hip(hip_api_t::load()), metal(metal_api_t::load()), metal_contexts(metal) {}
    gpu_drivers_t(gpu_drivers_t const&) = delete;
    gpu_drivers_t& operator=(gpu_drivers_t const&) = delete;
};

/** The driver each vendor's executor spawns over: its runtime's table, or Apple's contexts. */
inline cuda_api_t const& driver_of(gpu_drivers_t& drivers, std::type_identity<cuda_executor_t>) noexcept {
    return drivers.cuda;
}
inline hip_api_t const& driver_of(gpu_drivers_t& drivers, std::type_identity<hip_executor_t>) noexcept {
    return drivers.hip;
}
inline metal_contexts_t& driver_of(gpu_drivers_t& drivers, std::type_identity<metal_executor_t>) noexcept {
    return drivers.metal_contexts;
}

/** Writes into @p into every device @p drivers reach, NVIDIA first, then AMD, then Apple, as each
 *  driver reports it, and answers how many it wrote. */
inline std::size_t list_gpu_devices(gpu_drivers_t const& drivers, std::span<gpu_device_t> into) noexcept {
    std::size_t count = list_nvidia_gpu_devices(drivers.cuda, into);
    count += list_amd_gpu_devices(drivers.hip, into.subspan(count));
    count += list_apple_gpu_devices(drivers.metal_contexts, into.subspan(count));
    return count;
}

/** What one vendor's stream answers: its device, binding it, waiting for everything queued, and
 *  the memory operations queued in its order, each with that device bound. */
template <typename stream_type_>
concept a_gpu_stream = requires(stream_type_& stream, stream_type_ const& spawned, std::size_t bytes,
                                std::uintptr_t address, bytes_span_t target) {
    { spawned.ordinal } -> std::convertible_to<std::size_t>;
    { spawned.bind() } -> std::same_as<status_t>;
    { stream.wait() } -> std::same_as<status_t>;
    { stream.allocate_async(bytes) } -> std::same_as<expected<std::uintptr_t>>;
    { stream.free_async(address) };
    { stream.copy_async(address, address, bytes) } -> std::same_as<status_t>;
    { stream.upload_async(address, bytes_view_t {target}) } -> std::same_as<status_t>;
    { stream.download_async(target, address) } -> std::same_as<status_t>;
    { stream.fill_async(address, byte_t {}, bytes) } -> std::same_as<status_t>;
};

/** What every vendor's executor answers: its vendor, and its streams, one per device in the order
 *  the caller named them, and the wait for all of them. */
template <typename executor_type_>
concept a_gpu_executor = a_gpu_stream<typename executor_type_::stream_t> &&
                         requires(executor_type_& executor, executor_type_ const& spawned, std::size_t index) {
                             { executor_type_::vendor_k } -> std::convertible_to<gpu_vendor_t>;
                             { spawned.size() } -> std::same_as<std::size_t>;
                             { executor.stream(index) } -> std::same_as<typename executor_type_::stream_t&>;
                             { executor.wait() } -> std::same_as<status_t>;
                         };

static_assert(a_gpu_executor<cuda_executor_t>);
static_assert(a_gpu_executor<hip_executor_t>);
static_assert(a_gpu_executor<metal_executor_t>);

/** Whichever vendor's executor a VRAM call runs on, or none. */
using gpu_executor_t = std::variant<std::monostate, cuda_executor_t, hip_executor_t, metal_executor_t>;

/**
 *  @brief Spawns the executor of the one vendor every device in @p devices belongs to, over that
 *      vendor's driver in @p drivers, opening each stream's memory and kernels with
 *      @p memory_arguments.
 *
 *  One stream per distinct device over their vendor ordinals, which every shard there shares: the
 *  executor opens it, or adopts the entry of @p streams beside the device, a refused spawn leaving
 *  every one of them the caller's.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k No device is named, one repeats beside @p streams, or a stream is null
 *    or a default one.
 *  - @c config_mismatch_k The @p devices span vendors, or a stream sits on another device than
 *    its entry names.
 *  - @c device_k The build drives no executor of their vendor, or a device refused its stream.
 */
template <typename... memory_arguments_>
expected<gpu_executor_t> spawn_gpu_executor(gpu_drivers_t& drivers, std::span<gpu_device_t const> devices,
                                            std::span<void* const> streams,
                                            memory_arguments_&... memory_arguments) noexcept {
    if (devices.empty() || devices.size() > gpu_devices_capacity_k)
        return {{}, {status_code_t::invalid_argument_k, "An executor names one to a listing's devices"}};
    int ordinals[gpu_devices_capacity_k] = {};
    std::size_t ordinals_count = 0;
    for (gpu_device_t const& device : devices) {
        if (device.vendor != devices[0].vendor)
            return {{}, {status_code_t::config_mismatch_k, "One executor drives one GPU vendor"}};
        int const ordinal = static_cast<int>(device.vendor_ordinal);
        std::size_t earlier = 0;
        while (earlier != ordinals_count && ordinals[earlier] != ordinal) ++earlier;
        // Two adopted streams on one device would leave one of them unowned.
        if (earlier != ordinals_count && !streams.empty())
            return {{}, {status_code_t::invalid_argument_k, "Each adopted stream needs its own device"}};
        if (earlier == ordinals_count) ordinals[ordinals_count++] = ordinal;
    }
    gpu_executor_t executor;
    status_t spawning {status_code_t::device_k, "The build drives no executor of the vendor"};
    auto const spawn = [&]<typename executor_type_>(std::type_identity<executor_type_> vendor) noexcept {
        if (executor_type_::vendor_k != devices[0].vendor) return false;
        auto [spawned, status] = executor_type_::spawn(
            driver_of(drivers, vendor), std::span<int const> {ordinals, ordinals_count}, streams, memory_arguments...);
        if (status.succeeded()) executor.template emplace<executor_type_>(std::move(spawned));
        spawning = status;
        return true;
    };
    spawn(std::type_identity<cuda_executor_t> {}) || spawn(std::type_identity<hip_executor_t> {}) ||
        spawn(std::type_identity<metal_executor_t> {});
    return {std::move(executor), spawning};
}

#pragma endregion Shared VRAM

} // namespace unum::uform::utilities
