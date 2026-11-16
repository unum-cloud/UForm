/**
 *  @file include/uform/utilities/vram_amd_device.hpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief The HIP runtime's entry points the encoder's own kernels launch through on AMD GPUs,
 *      resolved into one table from a library opened at first use.
 *
 *  The inference library never links @c libamdhip64 , so it loads on a machine without ROCm:
 *  @ref unum::uform::utilities::hip_api_t::load opens it, and a machine without it keeps an empty
 *  table. Plain C++ needing no SDK: the few types, constants and prototypes the table calls are
 *  spelled here, copied from the runtime's stable ABI, so any host compiler builds the launches.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include "uform/utilities/blocking_files.hpp" // `shared_library_t` - the runtime opened at first use

namespace unum::uform::utilities {

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
using hip_stream_handle_t = struct hip_stream_opaque_t*;
using hip_module_handle_t = struct hip_module_opaque_t*;
using hip_function_handle_t = struct hip_function_opaque_t*;

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
    hip_result_t (*get_device_attribute)(int*, hip_device_attribute_t, int) = nullptr;
    hip_result_t (*get_device_name)(char*, int, int) = nullptr;
    hip_result_t (*get_device_memory)(std::size_t*, int) = nullptr;
    hip_result_t (*get_device)(int*) = nullptr;
    hip_result_t (*set_device)(int) = nullptr;
    hip_result_t (*stream_device)(hip_stream_handle_t, int*) = nullptr;
    hip_result_t (*copy_async)(void*, void const*, std::size_t, int, hip_stream_handle_t) = nullptr;
    hip_result_t (*fill_async)(void*, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*module_load_data)(hip_module_handle_t*, void const*) = nullptr;
    hip_result_t (*module_get_function)(hip_function_handle_t*, hip_module_handle_t,
                                        char const*) = nullptr; // vendor-boundary: the runtime's prototype
    hip_result_t (*module_launch_kernel)(hip_function_handle_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                         unsigned, unsigned, hip_stream_handle_t, void**, void**) = nullptr;

    /** Whether the runtime loaded with every entry resolved. */
    explicit operator bool() const noexcept { return module_launch_kernel != nullptr; }

    /** Opens the runtime under the first name a ROCm release gives it and resolves every entry, or
     *  answers the empty table. The library stays open for the life of the process, as the table
     *  points into it. */
    static hip_api_t load() noexcept {
        shared_library_t library;
        for (char const* name : {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"}) // vendor-boundary: sonames
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
