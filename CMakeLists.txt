# UForm — CUDA-first multimodal training stack.
#
# The core cell is a static library of `.cuh` interfaces + `.cu` kernel translation units; this file wires its source
# dependencies and `src/CMakeLists.txt` exposes the `uform::core` / `uform::loaders` / `uform::train` targets every tool
# and test links.
#
# Prerequisites — native libraries (apt).
# The CUDA toolkit must be 13.3+ on `PATH`: `nvcc` needs to be >= 13.1 for `-std=c++23` (see the standard block below).
# The machine default lives in `/etc/profile.d/cuda.sh`, pointing `PATH` / `LD_LIBRARY_PATH` at the `/usr/local/cuda`
# symlink; repoint that symlink to switch toolkits globally. Then the once-per-machine native set:
#
#   sudo apt install -y \
#     libcudnn9-dev-cuda-13 libnpp-dev-13-3 libnvjpeg-dev-13-3 libcufile-dev-13-3 \
#     libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
#     libjpeg-turbo8-dev libturbojpeg0-dev libpng-dev libwebp-dev \
#     libibverbs-dev ccache
#
# Prerequisites — RAPIDS cuDF (pip wheel into a repo-local venv).
# libcudf ships as a self-contained cu13 wheel (deliberately not `find_package(cudf)` — see the discovery block):
#
#   uv venv .venv && uv pip install --python .venv --extra-index-url https://pypi.nvidia.com libcudf-cu13
#
# CMake auto-discovers `$VIRTUAL_ENV`, then the repo `.venv`, then the interpreter's site-packages; override the search
# root with `-DUFORM_SITE_PACKAGES=<dir>`, or point straight at the libraries with `-DUFORM_CUDF_ROOT=<prefix>`.
#
# Configure, build, test. `CMakePresets.json` also holds `debug`, `check` and `relwithdebinfo`, each building into
# `build_<preset>`; machine specifics such as the compiler path go into a gitignored `CMakeUserPresets.json`.
#
#   cmake --preset release
#   cmake --build --preset release
#   ctest --preset release
#
# Local dependency checkouts (optional).
#
#   cmake --preset release -DFETCHCONTENT_SOURCE_DIR_NUMKONG="$PWD/../NumKong"   # likewise any other dependency
#
# Feature gates.
#
# `-DUFORM_BUILD_CUDA=ON` adds the NVIDIA residency, the trainer, the mesh and the loaders, which every training preset
# sets; `-DUFORM_BUILD_ROCM=ON` adds the AMD residency. Without them only the host libraries and the media codecs are
# needed, and on WebAssembly not even the codecs. Every native dependency above is required and fails the configure
# loudly if absent — each is available on every target, including consumer / workstation GPUs. InfiniBand is the sole
# optional path (absent on those cards): `-DUFORM_WITH_INFINIBAND=AUTO|ON|OFF` (default `AUTO` — probe and enable when
# present). `-DUFORM_BUILD_TEST=OFF` and `-DUFORM_BUILD_TOOLS=OFF` skip `test/` and `tools/`, which only a top-level
# build compiles by default.
cmake_minimum_required(VERSION 3.27)

# Must precede `project()` so `enable_language(CUDA)` uses our value instead of CMake's minimum-supported fallback.
# Dual-arch by default: Hopper (sm_90a) and Blackwell (sm_100a).
if (NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES
        "90a;100a"
        CACHE STRING "CUDA target architectures: 90a (Hopper), 100a (Blackwell)" FORCE
    )
endif ()

# Leverage `ccache` for incremental compilation
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif ()

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any
# cloned repo execute code), so point `core.hooksPath` at `.githooks` here — configuring the build arms
# the `pre-commit` / `commit-msg` checks. Skipped when there is no `.git` (tarball / export builds).
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _uform_hooks_status
        OUTPUT_QUIET ERROR_QUIET
    )
    if (_uform_hooks_status EQUAL 0)
        message(STATUS "uform: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _uform_version)
# CMake versions are numeric, so the pre-release suffix of VERSION stays out of PROJECT_VERSION.
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _uform_version_number "${_uform_version}")
project(
    uform
    VERSION ${_uform_version_number}
    LANGUAGES C CXX
    DESCRIPTION "Pocket-sized multimodal AI for content understanding and generation"
    HOMEPAGE_URL "https://github.com/unum-cloud/UForm"
)

# Determine if UForm is built as a subproject (using `add_subdirectory`) or if it is the main project
set(UFORM_IS_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(UFORM_IS_MAIN_PROJECT ON)
endif ()

option(UFORM_BUILD_CUDA "Add the NVIDIA residency, the trainer, the mesh and the loaders" OFF)
option(UFORM_BUILD_ROCM "Add the AMD residency" OFF)
option(UFORM_BUILD_LIB_C "Build uform_static and uform_shared" ${UFORM_IS_MAIN_PROJECT})
option(UFORM_BUILD_LIB_C_STUB "Build uform_static_stub and uform_shared_stub" OFF)
option(UFORM_BUILD_LIB_PYTHON "Build the CPython extension, set by scikit-build" OFF)
option(UFORM_BUILD_LIB_NODE "Build the Node addon, set by cmake-js" OFF)
option(UFORM_BUILD_TEST "Compile the conformance suites in test/" ${UFORM_IS_MAIN_PROJECT})
option(UFORM_BUILD_TOOLS "Compile the trainer in tools/" ${UFORM_IS_MAIN_PROJECT})

if (UFORM_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif ()
if (UFORM_BUILD_ROCM)
    enable_language(HIP)
endif ()

# C++23: the multidimensional subscript `tensor[row, column]` is the project's indexing default — device code included,
# which needs `nvcc` >= 13.1 (`-std=c++23` first shipped there; 13.0 rejects it). CMake's own `nvcc` dialect table still
# stops at 20 (`Modules/Compiler/NVIDIA.cmake` as of CMake 4.3), so the flag mapping is declared here until upstream
# catches up.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1 AND NOT CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
    set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

include(FetchContent)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_library.cmake")

# Source dependencies. NumKong spans the tensor substrate, StringZilla parses exchange payloads, ForkUnion is the
# host-side demux and decode thread pool, yyjson reads checkpoint headers, cudnn-frontend is the fused-attention graph
# DSL. The first three are submodules: a recursive clone builds them in place, any other clone fetches the pins below,
# which match the submodule commits. `FETCHCONTENT_SOURCE_DIR_<NAME>` points any of them at a local checkout.
function (uform_sibling NAME REPOSITORY TAG)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/CMakeLists.txt")
        add_subdirectory(${NAME})
    else ()
        message(STATUS "uform: ${NAME} submodule absent, fetching ${TAG}")
        FetchContent_Declare(
            ${NAME}
            GIT_REPOSITORY ${REPOSITORY}
            GIT_TAG ${TAG}
        )
        FetchContent_MakeAvailable(${NAME})
    endif ()
endfunction ()

# The header-only NumKong target stubs out the `_best` dispatch the inference products and attention run through, so
# UForm builds the compiled library, with the GPU vendors its own switches add, for its own architectures.
set(NUMKONG_BUILD_CUDA ${UFORM_BUILD_CUDA})
set(NUMKONG_BUILD_ROCM ${UFORM_BUILD_ROCM})
set(NUMKONG_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set(NUMKONG_BUILD_TEST OFF)
set(NUMKONG_BUILD_BENCH OFF)
uform_sibling(numkong https://github.com/ashvardanian/NumKong df1ab982a00b8aa97f39821cb5f371bd98caa43f)
uform_sibling(stringzilla https://github.com/ashvardanian/StringZilla afac5997e3fb15fd1586eca6724f9dacea152cb0)
uform_sibling(forkunion https://github.com/ashvardanian/ForkUnion 606eb35bad36e720116f9ab0b83299543b50c3b1)

FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.12.0
)
FetchContent_MakeAvailable(yyjson)

# Host codecs: libjpeg-turbo / libpng / libwebp decode images on the CPU, libav* demuxes video and decodes audio (no GPU
# audio codec exists). WebAssembly builds take already-decoded media, so they discover none of them.
if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm")
    find_package(JPEG REQUIRED)
    find_package(PNG REQUIRED)
    find_library(UFORM_WEBP_LIBRARY webp REQUIRED)
    uform_find_library(UFORM_AVFORMAT_LIBRARY avformat)
    uform_find_library(UFORM_AVCODEC_LIBRARY avcodec)
    uform_find_library(UFORM_AVUTIL_LIBRARY avutil)
    uform_find_library(UFORM_SWRESAMPLE_LIBRARY swresample)
    if (NOT
        (UFORM_AVFORMAT_LIBRARY
         AND UFORM_AVCODEC_LIBRARY
         AND UFORM_AVUTIL_LIBRARY
         AND UFORM_SWRESAMPLE_LIBRARY)
    )
        message(FATAL_ERROR "uform: libav* not found — install the ffmpeg dev packages: "
                            "sudo apt install libavformat-dev libavcodec-dev libavutil-dev libswresample-dev."
        )
    endif ()
endif ()

# InfiniBand: the one optional transport (absent on consumer / workstation cards). Tri-state probe, see the module.
set(UFORM_WITH_INFINIBAND
    "AUTO"
    CACHE STRING "InfiniBand transport: ON (require), OFF (disable), AUTO (probe and enable when present)"
)
set_property(CACHE UFORM_WITH_INFINIBAND PROPERTY STRINGS ON OFF AUTO)

if (UFORM_BUILD_CUDA)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe_infiniband.cmake")
    uform_probe_infiniband()

    # Header-only: a `SOURCE_SUBDIR` without a `CMakeLists.txt` populates the sources without configuring them.
    FetchContent_Declare(
        cudnn_frontend
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG v1.19.1
        SOURCE_SUBDIR include
    )
    FetchContent_MakeAvailable(cudnn_frontend)

    # cuDNN powers the fused scaled-dot-product-attention graphs of the transformer blocks — available on every NVIDIA
    # GPU, so hard-required alongside the frontend header DSL fetched above.
    find_library(UFORM_CUDNN_LIBRARY cudnn REQUIRED)

    # The loaders' vendor dependencies below are available on every NVIDIA target and are hard-required, discovered
    # through the layered `uform_find_library` search (the RAPIDS / NVIDIA cu13 Python wheels, then system paths). A
    # missing dependency fails the configure loudly naming the install.

    # libcudf: GPU-side Parquet reads on explicit per-loader streams. Deliberately not `find_package(cudf)` — its config
    # drags a CCCL that conflicts with the CUDA toolkit's; the wheels ship self-contained `.so`s and headers.
    set(UFORM_CUDF_ROOT
        ""
        CACHE PATH "Directory holding libcudf/lib64, librmm/lib64, ... overriding the wheel discovery"
    )
    if (UFORM_CUDF_ROOT)
        set(_uform_cudf_prefix "${UFORM_CUDF_ROOT}")
    else ()
        set(_uform_cudf_prefix "${UFORM_SITE_PACKAGES}")
    endif ()
    if (NOT (_uform_cudf_prefix AND EXISTS "${_uform_cudf_prefix}/libcudf/lib64/libcudf.so"))
        message(
            FATAL_ERROR
                "uform: libcudf not found under '${_uform_cudf_prefix}'. Install the RAPIDS cu13 wheel:\n"
                "  uv venv .venv && uv pip install --python .venv --extra-index-url https://pypi.nvidia.com libcudf-cu13\n"
                "or pass -DUFORM_CUDF_ROOT=<prefix holding libcudf/lib64>."
        )
    endif ()
    add_library(uform_cudf INTERFACE)
    # `--no-as-needed` keeps the transitively-dlopened RAPIDS runtime libraries in the link closure.
    target_link_options(uform_cudf INTERFACE "LINKER:--no-as-needed" "LINKER:--disable-new-dtags")
    target_link_libraries(
        uform_cudf
        INTERFACE "${_uform_cudf_prefix}/libcudf/lib64/libcudf.so"
                  "${_uform_cudf_prefix}/librmm/lib64/librmm.so"
                  "${_uform_cudf_prefix}/libkvikio/lib64/libkvikio.so"
                  "${_uform_cudf_prefix}/nvidia/libnvcomp/lib64/libnvcomp.so.5"
                  "${_uform_cudf_prefix}/rapids_logger/lib64/librapids_logger.so"
    )
    message(STATUS "uform: libcudf loaders from ${_uform_cudf_prefix}/libcudf")

    # cuFile: GPU Direct Storage — the byte tape's primary fill path. Ships with the CUDA toolkit, so hard-required; the
    # POSIX fallback is compiled either way and the runtime drops to it when GDS is unavailable on the filesystem.
    uform_find_library(UFORM_CUFILE_LIBRARY cufile)
    if (NOT UFORM_CUFILE_LIBRARY)
        message(
            FATAL_ERROR "uform: libcufile not found — it ships with the CUDA toolkit; check the CUDA install / PATH."
        )
    endif ()

    # NVDEC: nv-codec-headers carry the `cuvid*` declarations without the Video Codec SDK EULA download; libnvcuvid is
    # part of the NVIDIA driver, present on every target. (Loader wired now; video sources consumed later.) nvJPEG and
    # NPP, the image half, come with the CUDA toolkit as `CUDA::` targets.
    uform_find_library(UFORM_NVCUVID_LIBRARY nvcuvid EXTRA_NAMES libnvcuvid.so.1)
    if (NOT UFORM_NVCUVID_LIBRARY)
        message(
            FATAL_ERROR "uform: libnvcuvid not found — it is part of the NVIDIA driver; ensure the driver is installed."
        )
    endif ()
    FetchContent_Declare(
        ffnvcodec
        GIT_REPOSITORY https://github.com/FFmpeg/nv-codec-headers.git
        GIT_TAG n13.0.19.0
        SOURCE_SUBDIR include
    )
    FetchContent_MakeAvailable(ffnvcodec)
endif ()

# `ctest` reads the tests from the build root, so testing is enabled here for every subdirectory that adds some: the
# conformance suites, and the C test beside the C libraries.
if (UFORM_BUILD_TEST OR UFORM_BUILD_LIB_C)
    enable_testing()
endif ()

add_subdirectory(src)
if (UFORM_BUILD_LIB_C_STUB AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src_stub/CMakeLists.txt")
    add_subdirectory(src_stub)
endif ()
if (UFORM_BUILD_LIB_C AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/c/CMakeLists.txt")
    add_subdirectory(c)
endif ()
if (UFORM_BUILD_LIB_PYTHON AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
    add_subdirectory(python)
endif ()
if (UFORM_BUILD_LIB_NODE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/javascript/CMakeLists.txt")
    add_subdirectory(javascript)
endif ()
