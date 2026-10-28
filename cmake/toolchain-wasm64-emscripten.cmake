# WebAssembly via Emscripten, the extended module.
#
# ~~~
#   source /path/to/emsdk/emsdk_env.sh
#   cmake --preset wasm64_emscripten
#   cmake --build --preset wasm64_emscripten --target uform_static
# ~~~
#
# Relaxed SIMD, shared memory and 64-bit addressing are whole-module choices, so one file fixes all three: the pools
# spawn Web Workers, and the memory passes 4 GB. A page loads it only when cross-origin isolated. Emscripten's own file
# reads the width off `CMAKE_C_FLAGS` as it loads, so the flags precede the include.
if (NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK is unset; source the SDK's `emsdk_env.sh` before configuring.")
endif ()

set(CMAKE_C_FLAGS
    "-pthread --target=wasm64 -msimd128 -mrelaxed-simd"
    CACHE STRING "Flags used by the C compiler during all build types."
)
set(CMAKE_CXX_FLAGS
    "-pthread --target=wasm64 -msimd128 -mrelaxed-simd"
    CACHE STRING "Flags used by the CXX compiler during all build types."
)
set(EMSCRIPTEN_SYSTEM_PROCESSOR wasm64)
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-pthread --target=wasm64 -sPROXY_TO_PTHREAD -sALLOW_MEMORY_GROWTH -sNODE_HOST_ENV")
