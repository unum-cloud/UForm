# WebAssembly via the WASI SDK, for standalone runtimes and the Rust crate.
#
# ~~~
#   export WASI_SDK_PATH=/opt/wasi-sdk
#   cmake -B build_wasm32_wasi_release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake -DUFORM_BUILD_LIB_C=1
#   cmake --build build_wasm32_wasi_release --target uform_static
# ~~~
#
# `wasm32-wasip1`, one thread, the fixed-width `simd128` tier, self-contained memory: the same module runs under
# Wasmtime, Wasmer and Node. The SDK's own file names the compilers and the sysroot; this one adds the tier and the
# runtime `ctest` drives. Exceptions stay off, which is the `noeh` multilib `build.rs` links against.
if (NOT DEFINED WASI_SDK_PATH)
    if (DEFINED ENV{WASI_SDK_PATH})
        set(WASI_SDK_PATH "$ENV{WASI_SDK_PATH}")
    else ()
        message(FATAL_ERROR "WASI_SDK_PATH is unset; point it at an unpacked wasi-sdk release before configuring.")
    endif ()
endif ()

# Nested try-compile projects reread this file and inherit the environment, but not this cache variable.
set(ENV{WASI_SDK_PATH} "${WASI_SDK_PATH}")

set(WASI_SDK_PREFIX "${WASI_SDK_PATH}")
include("${WASI_SDK_PATH}/share/cmake/wasi-sdk-p1.cmake")

set(CMAKE_C_FLAGS_INIT "-msimd128")
set(CMAKE_CXX_FLAGS_INIT "-msimd128 -fno-exceptions")

find_program(UFORM_WASMTIME wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
if (UFORM_WASMTIME)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${UFORM_WASMTIME};run;-S;inherit-env=y;--dir;.")
endif ()
