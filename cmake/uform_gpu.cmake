# GPU languages, and the C++23 dialect every CUDA unit compiles at.

if (UFORM_BUILD_CUDA)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
        message(FATAL_ERROR "UForm needs CUDA 13 or newer, found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif ()
    # The assembler embeds device-only bundles, see `uform_gpu_bundle`.
    enable_language(ASM)
    find_package(CUDAToolkit 13 REQUIRED)
endif ()
if (UFORM_BUILD_ROCM)
    enable_language(HIP)
endif ()

# C++23: the multidimensional subscript `tensor[row, column]` is the project's indexing default — device code included,
# which needs `nvcc` >= 13.1 (`-std=c++23` first shipped there; 13.0 rejects it). Host targets declare `cxx_std_23`
# themselves, while CMake's own `nvcc` dialect table still stops at 20 (`Modules/Compiler/NVIDIA.cmake` as of CMake 4.3),
# so no CUDA target can declare `cuda_std_23`, and the dialect and its flag mapping stay here until upstream catches up.
set(CMAKE_CUDA_STANDARD 23)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1 AND NOT CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
    set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif ()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
