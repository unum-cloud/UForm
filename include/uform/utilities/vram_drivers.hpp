/**
 *  @file include/uform/utilities/vram_drivers.hpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief The GPU drivers UForm's own kernels launch through: the CUDA driver's and the HIP
 *      runtime's entry points, each resolved into one table from a library opened at first use.
 *
 *  The inference library never links @c libcuda or @c libamdhip64, so it loads on a machine with
 *  either vendor, both or none: @ref unum::uform::utilities::cuda_api_t::load and
 *  @ref unum::uform::utilities::hip_api_t::load open them, and a vendor whose library is absent
 *  keeps an empty table. Plain C++ needing neither SDK: the few types, constants and prototypes
 *  the tables call are spelled here, copied from each vendor's stable ABI, so any host compiler
 *  builds the units that launch.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <dlfcn.h> // `dlopen`, `dlsym` - each driver opened at first use

namespace unum::uform::utilities {

/** A shared library opened by name and kept open for the process's life, as the entry points
 *  resolved from it are never unloaded; empty where the system has no such library. */
struct shared_library_t {
    void *handle = nullptr;

    /** Opens the library named @p name , or answers the empty one. */
    static shared_library_t open(char const *name) noexcept { // vendor-boundary: `dlopen` takes a C string
        return {dlopen(name, RTLD_NOW | RTLD_LOCAL)};
    }

    /** Points @p entry at the symbol exported as @p name , answering whether there is one. */
    template <typename function_type_>
    bool resolve(char const *name, function_type_ &entry) const noexcept { // vendor-boundary: `dlsym` takes one
        if (!handle) return false;
        entry = reinterpret_cast<function_type_>(dlsym(handle, name));
        return entry != nullptr;
    }

    explicit operator bool() const noexcept { return handle != nullptr; }
};

#pragma region CUDA Driver

/** The driver's status word, @c CUresult , naming the values UForm tells apart; any other
 *  is a failure. */
enum cuda_result_t : int {
    cuda_success_k = 0,
    cuda_error_not_initialized_k = 3,
    cuda_error_invalid_device_k = 101,
    cuda_error_invalid_context_k = 201,
    cuda_error_no_binary_for_gpu_k = 209,
};

/** The driver's opaque handles: @c CUcontext , @c CUstream , @c CUmodule and @c CUfunction . */
using cuda_context_handle_t = struct cuda_context_opaque_t *;
using cuda_stream_handle_t = struct cuda_stream_opaque_t *;
using cuda_module_handle_t = struct cuda_module_opaque_t *;
using cuda_function_handle_t = struct cuda_function_opaque_t *;

/** A device address, @c CUdeviceptr . */
using cuda_device_address_t = unsigned long long;

/** The @c CUdevice_attribute values UForm queries. */
enum cuda_device_attribute_t : int {
    cuda_attribute_multiprocessor_count_k = 16,
    cuda_attribute_compute_capability_major_k = 75,
    cuda_attribute_compute_capability_minor_k = 76,
};

/**
 *  @brief The driver entry points UForm's kernels launch through, resolved from @c libcuda .
 *
 *  Every entry is null when the driver is absent or lacks any of them; each spells the prototype of
 *  the exported symbol it is resolved by, the @c _v2 one where @c <cuda.h> maps the name onto it.
 */
struct cuda_api_t {
    cuda_result_t (*init)(unsigned) = nullptr;
    cuda_result_t (*get_device)(int *, int) = nullptr;
    cuda_result_t (*get_device_attribute)(int *, cuda_device_attribute_t, int) = nullptr;
    cuda_result_t (*get_device_name)(char *, int, int) = nullptr;
    cuda_result_t (*get_device_memory)(std::size_t *, int) = nullptr;
    cuda_result_t (*get_current_context)(cuda_context_handle_t *) = nullptr;
    cuda_result_t (*get_context_device)(int *) = nullptr;
    cuda_result_t (*push_context)(cuda_context_handle_t) = nullptr;
    cuda_result_t (*pop_context)(cuda_context_handle_t *) = nullptr;
    cuda_result_t (*stream_context)(cuda_stream_handle_t, cuda_context_handle_t *) = nullptr;
    cuda_result_t (*copy_async)(cuda_device_address_t, cuda_device_address_t, std::size_t,
                                cuda_stream_handle_t) = nullptr;
    cuda_result_t (*fill_async)(cuda_device_address_t, unsigned char, std::size_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*module_load_data)(cuda_module_handle_t *, void const *) = nullptr;
    cuda_result_t (*module_get_function)(cuda_function_handle_t *, cuda_module_handle_t,
                                         char const *) = nullptr; // vendor-boundary: the driver's prototype
    cuda_result_t (*launch_kernel)(cuda_function_handle_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   unsigned, cuda_stream_handle_t, void **, void **) = nullptr;

    /** Whether the driver loaded and initialized with every entry resolved. */
    explicit operator bool() const noexcept { return launch_kernel != nullptr; }

    /** Opens the driver, resolves every entry and initializes it, or answers the empty table. The
     *  library stays open for the life of the process, as the table points into it. */
    static cuda_api_t load() noexcept {
        shared_library_t const library = shared_library_t::open("libcuda.so.1");
        cuda_api_t api;
        bool const resolved = //
            library.resolve("cuInit", api.init) && library.resolve("cuDeviceGet", api.get_device) &&
            library.resolve("cuDeviceGetAttribute", api.get_device_attribute) &&
            library.resolve("cuDeviceGetName", api.get_device_name) &&
            library.resolve("cuDeviceTotalMem_v2", api.get_device_memory) &&
            library.resolve("cuCtxGetCurrent", api.get_current_context) &&
            library.resolve("cuCtxGetDevice", api.get_context_device) &&
            library.resolve("cuCtxPushCurrent_v2", api.push_context) &&
            library.resolve("cuCtxPopCurrent_v2", api.pop_context) &&
            library.resolve("cuStreamGetCtx", api.stream_context) && library.resolve("cuMemcpyAsync", api.copy_async) &&
            library.resolve("cuMemsetD8Async", api.fill_async) &&
            library.resolve("cuModuleLoadData", api.module_load_data) &&
            library.resolve("cuModuleGetFunction", api.module_get_function) &&
            library.resolve("cuLaunchKernel", api.launch_kernel);
        return resolved && api.init(0) == cuda_success_k ? api : cuda_api_t {};
    }
};

#pragma endregion CUDA Driver

#pragma region HIP Runtime

/** The runtime's status word, @c hipError_t , naming the values UForm tells apart; any other
 *  is a failure. */
enum hip_result_t : int {
    hip_success_k = 0,
    hip_error_not_initialized_k = 3,
    hip_error_invalid_device_k = 101,
    hip_error_no_binary_for_gpu_k = 209,
};

/** The runtime's opaque handles: @c hipStream_t , @c hipModule_t and @c hipFunction_t . */
using hip_stream_handle_t = struct hip_stream_opaque_t *;
using hip_module_handle_t = struct hip_module_opaque_t *;
using hip_function_handle_t = struct hip_function_opaque_t *;

/** The @c hipMemcpyKind UForm copies with: unified addressing tells the sides apart. */
inline constexpr int hip_copy_default_k = 4;

/** The @c hipDeviceAttribute_t values UForm queries; on AMD the compute capability is the @c gfx
 *  processor's major and minor version. */
enum hip_device_attribute_t : int {
    hip_attribute_compute_capability_major_k = 23,
    hip_attribute_compute_capability_minor_k = 61,
    hip_attribute_multiprocessor_count_k = 63,
};

/**
 *  @brief The HIP entry points UForm's kernels launch through, resolved from @c libamdhip64 .
 *
 *  Every entry is null when the library is absent or lacks any of them; each spells the C
 *  prototype of the exported symbol it is resolved by.
 */
struct hip_api_t {
    hip_result_t (*get_device_attribute)(int *, hip_device_attribute_t, int) = nullptr;
    hip_result_t (*get_device_name)(char *, int, int) = nullptr;
    hip_result_t (*get_device_memory)(std::size_t *, int) = nullptr;
    hip_result_t (*get_device)(int *) = nullptr;
    hip_result_t (*set_device)(int) = nullptr;
    hip_result_t (*stream_device)(hip_stream_handle_t, int *) = nullptr;
    hip_result_t (*copy_async)(void *, void const *, std::size_t, int, hip_stream_handle_t) = nullptr;
    hip_result_t (*fill_async)(void *, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*module_load_data)(hip_module_handle_t *, void const *) = nullptr;
    hip_result_t (*module_get_function)(hip_function_handle_t *, hip_module_handle_t,
                                        char const *) = nullptr; // vendor-boundary: the runtime's prototype
    hip_result_t (*module_launch_kernel)(hip_function_handle_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                         unsigned, unsigned, hip_stream_handle_t, void **, void **) = nullptr;

    /** Whether the runtime loaded with every entry resolved. */
    explicit operator bool() const noexcept { return module_launch_kernel != nullptr; }

    /** Opens the runtime under the first name a ROCm release gives it and resolves every entry, or
     *  answers the empty table. The library stays open for the life of the process, as the table
     *  points into it. */
    static hip_api_t load() noexcept {
        shared_library_t library;
        for (char const *name : {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"}) // vendor-boundary: sonames
            if ((library = shared_library_t::open(name))) break;
        hip_api_t api;
        bool const resolved = //
            library.resolve("hipDeviceGetAttribute", api.get_device_attribute) &&
            library.resolve("hipDeviceGetName", api.get_device_name) &&
            library.resolve("hipDeviceTotalMem", api.get_device_memory) &&
            library.resolve("hipGetDevice", api.get_device) && library.resolve("hipSetDevice", api.set_device) &&
            library.resolve("hipStreamGetDevice", api.stream_device) &&
            library.resolve("hipMemcpyAsync", api.copy_async) && library.resolve("hipMemsetAsync", api.fill_async) &&
            library.resolve("hipModuleLoadData", api.module_load_data) &&
            library.resolve("hipModuleGetFunction", api.module_get_function) &&
            library.resolve("hipModuleLaunchKernel", api.module_launch_kernel);
        return resolved ? api : hip_api_t {};
    }
};

#pragma endregion HIP Runtime

} // namespace unum::uform::utilities
