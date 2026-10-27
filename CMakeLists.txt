# UForm — CUDA-first multimodal training stack.
#
# The core cell is a static library of `.cuh` interfaces + `.cu` kernel translation units; this file wires its source
# dependencies and `src/CMakeLists.txt` exposes the `uform::loaders` / `uform::train` targets every tool
# and test links.
#
# Prerequisites — native libraries (apt).
# The CUDA toolkit must be 13.3+ on `PATH`: `nvcc` needs to be >= 13.1 for `-std=c++23` (see the standard block below).
# The machine default lives in `/etc/profile.d/cuda.sh`, pointing `PATH` / `LD_LIBRARY_PATH` at the `/usr/local/cuda`
# symlink; repoint that symlink to switch toolkits globally. Then the once-per-machine native set:
#
#   sudo apt install -y \
#     libnpp-dev-13-3 libnvjpeg-dev-13-3 libcufile-dev-13-3 \
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
# Feature gates.
#
# `-DUFORM_BUILD_CUDA=ON` adds the NVIDIA residency, the trainer, the mesh and the loaders, which every training preset
# sets; `-DUFORM_BUILD_ROCM=ON` adds the AMD residency. Without them only the host libraries and the media codecs are
# needed, and on WebAssembly not even the codecs. Every native dependency above is required and fails the configure
# loudly if absent — each is available on every target, including consumer / workstation GPUs. InfiniBand is the sole
# optional path (absent on those cards): `-DUFORM_WITH_INFINIBAND=AUTO|ON|OFF` (default `AUTO` — probe and enable when
# present). `-DUFORM_BUILD_TEST=ON`, `-DUFORM_BUILD_TOOLS=ON` and `-DUFORM_BUILD_BENCH=ON` add the suites in `test/`, the
# trainer in `tools/` and the benchmarks in `bench/`, which every build leaves out unless asked.
cmake_minimum_required(VERSION 3.28)

# Every CUDA unit compiles one cubin per generation, as USearch's do: sm_86 runs Ampere's and B300 runs "100f", and PTX
# would double every object for GPUs no build meets. Iterate on one card with its own code, like "89-real". Only the
# Hopper and Blackwell codes reach the switch bundle, see `src/CMakeLists.txt`. Must precede `project()`, so
# `enable_language(CUDA)` takes it over CMake's fallback.
set(UFORM_CUDA_ARCHITECTURES
    "80-real;89-real;90a-real;100f-real;120f-real"
    CACHE STRING "GPU architectures every CUDA unit, NumKong's included, compiles for"
)
set(CMAKE_CUDA_ARCHITECTURES "${UFORM_CUDA_ARCHITECTURES}")

# Leverage `ccache` for incremental compilation
find_program(UFORM_CCACHE_PROGRAM ccache)
if (UFORM_CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${UFORM_CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${UFORM_CCACHE_PROGRAM}")
endif ()

project(
    uform
    VERSION 4.0.0
    LANGUAGES C CXX
    DESCRIPTION "Pocket-sized multimodal AI for content understanding and generation"
    HOMEPAGE_URL "https://github.com/unum-cloud/UForm"
)

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any
# cloned repo execute code), so point `core.hooksPath` at `.githooks` here — configuring the build arms
# the `pre-commit` / `commit-msg` checks. Skipped when there is no `.git` (tarball / export builds).
if (PROJECT_IS_TOP_LEVEL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE uform_hooks_status_
        OUTPUT_QUIET ERROR_QUIET
    )
    if (uform_hooks_status_ EQUAL 0)
        message(STATUS "uform: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

option(UFORM_INSTALL "Install the C headers, libraries and CMake package" ${PROJECT_IS_TOP_LEVEL})
option(UFORM_BUILD_CUDA "Add the NVIDIA residency, the trainer, the mesh and the loaders" OFF)
option(UFORM_BUILD_ROCM "Add the AMD residency" OFF)
option(UFORM_BUILD_SHARED "Compile uform_static and uform_shared in the default build and install them" OFF)
option(UFORM_BUILD_STUB "Build uform_static_stub and uform_shared_stub" OFF)
option(UFORM_BUILD_PYTHON "Build the CPython extension, set by scikit-build" OFF)
option(UFORM_BUILD_NODE "Build the Node addon, set by cmake-js" OFF)
option(UFORM_BUILD_SWIFT "Package the static library for SwiftPM, as an XCFramework or an artifact bundle" OFF)
option(UFORM_BUILD_TEST "Compile the conformance suites in test/" OFF)
option(UFORM_BUILD_TOOLS "Compile the trainer in tools/" OFF)
option(UFORM_BUILD_BENCH "Compile the benchmarks in bench/" OFF)
option(UFORM_WERROR "Treat compiler warnings as errors" OFF)

if (UFORM_BUILD_CUDA)
    enable_language(CUDA)
    # The assembler embeds device-only bundles, see `uform_gpu_bundle`.
    enable_language(ASM)
    find_package(CUDAToolkit REQUIRED)
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
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings apply to UForm's own targets only, so dependencies build with their own flags, and CUDA units with nvcc's.
add_library(uform_warnings INTERFACE)
if (MSVC)
    target_compile_options(uform_warnings INTERFACE /W4 /permissive- /Zc:__cplusplus)
    if (UFORM_WERROR)
        target_compile_options(uform_warnings INTERFACE /WX)
    endif ()
else ()
    target_compile_options(
        uform_warnings
        INTERFACE
            "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic;-Wno-unknown-pragmas;-Wno-missing-field-initializers>"
            "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wdocumentation>"
    )
    if (UFORM_WERROR)
        target_compile_options(uform_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
    endif ()
endif ()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

include(FetchContent)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_library.cmake")

# Source dependencies. NumKong spans the tensor substrate, StringZilla parses exchange payloads, ForkUnion is the
# host-side demux and decode thread pool, yyjson reads checkpoint headers. The first three are submodules: a recursive
# clone builds them in place, any other clone fetches the pins below, which match the submodule commits. `FETCHCONTENT_SOURCE_DIR_<NAME>` points any of them at a local checkout.
function (uform_sibling name_ repository_ tag_)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name_}/CMakeLists.txt")
        add_subdirectory(${name_} SYSTEM)
    else ()
        message(STATUS "uform: ${name_} submodule absent, fetching ${tag_}")
        FetchContent_Declare(
            ${name_}
            GIT_REPOSITORY ${repository_}
            GIT_TAG ${tag_}
            SYSTEM
        )
        FetchContent_MakeAvailable(${name_})
    endif ()
endfunction ()

# The header-only NumKong target stubs out the `_best` dispatch the inference products and attention run through, so
# UForm builds the compiled library, with the GPU vendors its own switches add, for its own architectures.
set(NUMKONG_BUILD_CUDA ${UFORM_BUILD_CUDA})
set(NUMKONG_BUILD_ROCM ${UFORM_BUILD_ROCM})
set(NUMKONG_CUDA_ARCHITECTURES "${UFORM_CUDA_ARCHITECTURES}")
uform_sibling(numkong https://github.com/ashvardanian/NumKong 49b5140981945527e0ffca4d8037285022b98ad3)
uform_sibling(stringzilla https://github.com/ashvardanian/StringZilla e169620f92db7c5df3a71baf2e035bc43ecf6a39)
uform_sibling(forkunion https://github.com/ashvardanian/ForkUnion 0c87b0bb539ccf6d945711a12e73b731530c9fa8)

# SmashTable's containers, header-only
uform_sibling(smashtable https://github.com/ashvardanian/SmashTable 07a8cb3800de52a4c0f90e80014b3937b39121dc)

set(YYJSON_INSTALL OFF)
FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.12.0
    SYSTEM
)
FetchContent_MakeAvailable(yyjson)
if (UFORM_BUILD_TEST
    OR UFORM_BUILD_TOOLS
    OR UFORM_BUILD_BENCH
)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/fmt.cmake")
endif ()

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
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/infiniband.cmake")
    uform_find_infiniband(uform_infiniband_enabled_)

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
        set(uform_cudf_prefix_ "${UFORM_CUDF_ROOT}")
    else ()
        set(uform_cudf_prefix_ "${uform_wheel_site_packages_}")
    endif ()
    if (NOT (uform_cudf_prefix_ AND EXISTS "${uform_cudf_prefix_}/libcudf/lib64/libcudf.so"))
        message(
            FATAL_ERROR
                "uform: libcudf not found under '${uform_cudf_prefix_}'. Install the RAPIDS cu13 wheel:\n"
                "  uv venv .venv && uv pip install --python .venv --extra-index-url https://pypi.nvidia.com libcudf-cu13\n"
                "or pass -DUFORM_CUDF_ROOT=<prefix holding libcudf/lib64>."
        )
    endif ()
    add_library(uform_cudf INTERFACE)
    # `--no-as-needed` keeps the transitively-dlopened RAPIDS runtime libraries in the link closure.
    target_link_options(uform_cudf INTERFACE "LINKER:--no-as-needed" "LINKER:--disable-new-dtags")
    target_link_libraries(
        uform_cudf
        INTERFACE "${uform_cudf_prefix_}/libcudf/lib64/libcudf.so"
                  "${uform_cudf_prefix_}/librmm/lib64/librmm.so"
                  "${uform_cudf_prefix_}/libkvikio/lib64/libkvikio.so"
                  "${uform_cudf_prefix_}/nvidia/libnvcomp/lib64/libnvcomp.so.5"
                  "${uform_cudf_prefix_}/rapids_logger/lib64/librapids_logger.so"
    )
    message(STATUS "uform: libcudf loaders from ${uform_cudf_prefix_}/libcudf")

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
# conformance suites, and the C tests beside the C libraries and the stand-in.
if (UFORM_BUILD_TEST)
    enable_testing()
endif ()

# The heaviest CUDA units peak near 4 GB each, so those of `src/`, `test/` and `tools/` queue in a pool of three, which
# bounds a build's memory whatever `-j` asks.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS uform_cuda_units=3)

# The suites precede `c/`, whose C tests load the checkpoint fixture `test/` writes.
set(uform_bundled_files_)
link_libraries($<BUILD_INTERFACE:uform_warnings>)
add_subdirectory(src)
if (UFORM_BUILD_TEST)
    add_subdirectory(test)
endif ()
if (UFORM_BUILD_TOOLS AND UFORM_BUILD_CUDA)
    add_subdirectory(tools)
endif ()
if (UFORM_BUILD_BENCH)
    add_subdirectory(bench)
endif ()
# `uform_bundle` → a self-contained runtime tree (executables + their shared-library closure, glibc excepted) under
# `runtime-bundle/`, for Slurm or K8s clusters where nodes are not package-identical: deploy the tree to the shared
# filesystem and every node runs it, whatever it has installed natively. Each executable travels as its own argument.
if (uform_bundled_files_)
    add_custom_target(
        uform_bundle
        COMMAND ${CMAKE_COMMAND} "-DUFORM_BUNDLE_DIR=${PROJECT_BINARY_DIR}/runtime-bundle" -P
                "${PROJECT_SOURCE_DIR}/cmake/bundle_runtime.cmake" -- ${uform_bundled_files_}
        COMMENT "Bundling executables with their runtime closure into runtime-bundle/"
        VERBATIM
    )
endif ()
if (UFORM_BUILD_STUB)
    add_subdirectory(src_stub)
endif ()
add_subdirectory(c)
# The C library as SwiftPM links it: an XCFramework on Apple platforms and an SE-0482 artifact bundle elsewhere, packed
# by NumKong's script beside the C headers and a module map over their umbrella. Each build adds its target to the
# artifact in UFORM_SWIFT_DIRECTORY, so a release fills one artifact from a build per target. With UFORM_BUILD_STUB the
# artifact packs the stand-in, which the Swift suite runs against.
if (UFORM_BUILD_SWIFT)
    set(UFORM_SWIFT_DIRECTORY
        "${PROJECT_BINARY_DIR}"
        CACHE PATH "Where the Swift artifact lands"
    )
    set(UFORM_SWIFT_SDK
        ""
        CACHE STRING "The Apple SDK the archive targets, like `macosx` or `iphonesimulator`"
    )
    set(UFORM_SWIFT_TRIPLES
        "${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu"
        CACHE STRING "Swift target triples the artifact bundle serves"
    )
    if (APPLE)
        set(uform_swift_target_ -D "NUMKONG_SWIFT_SDK=${UFORM_SWIFT_SDK}" -D
                                "NUMKONG_SWIFT_ARCHITECTURE=${CMAKE_OSX_ARCHITECTURES}"
        )
    else ()
        set(uform_swift_target_ -D "NUMKONG_SWIFT_TRIPLES=$<JOIN:${UFORM_SWIFT_TRIPLES},,>" -D
                                "NUMKONG_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    if (UFORM_BUILD_STUB)
        set(uform_swift_archive_ uform_static_stub)
    else ()
        set(uform_swift_archive_ uform_static)
    endif ()
    set(uform_swift_headers_ "${PROJECT_BINARY_DIR}/swift_headers")
    file(WRITE "${uform_swift_headers_}/module.modulemap"
         "module UFormC {\n    header \"uform/uform.h\"\n    export *\n}\n"
    )
    add_custom_target(
        uform_swift ALL
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${PROJECT_SOURCE_DIR}/c/uform" "${uform_swift_headers_}/uform"
        COMMAND
            "${CMAKE_COMMAND}" -D NUMKONG_SWIFT_MODULE=UFormC -D
            "NUMKONG_SWIFT_ARCHIVE=$<TARGET_FILE:${uform_swift_archive_}>" -D
            "NUMKONG_SWIFT_HEADERS=${uform_swift_headers_}" -D "NUMKONG_SWIFT_DIRECTORY=${UFORM_SWIFT_DIRECTORY}"
            ${uform_swift_target_} -P "${numkong_SOURCE_DIR}/cmake/numkong_swift_artifact.cmake"
        DEPENDS ${uform_swift_archive_}
        VERBATIM
    )
endif ()
if (UFORM_BUILD_PYTHON)
    add_subdirectory(python)
endif ()
if (UFORM_BUILD_NODE)
    add_subdirectory(javascript)
endif ()
