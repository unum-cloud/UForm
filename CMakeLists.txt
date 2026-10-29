# UForm — CUDA-first multimodal training stack.
#
# The core cell is a static library of `.cuh` interfaces + `.cu` kernel translation units; this file wires its source
# dependencies and `src/CMakeLists.txt` exposes the `uform::loaders` / `uform::train` targets every tool
# and test links.
#
# Prerequisites — native libraries (apt).
# The CUDA toolkit must be 13.3+ on `PATH`: `nvcc` needs to be >= 13.1 for `-std=c++23` (see `cmake/uform_gpu.cmake`).
# The machine default lives in `/etc/profile.d/cuda.sh`, pointing `PATH` / `LD_LIBRARY_PATH` at the `/usr/local/cuda`
# symlink; repoint that symlink to switch toolkits globally. Then the once-per-machine native set:
#
#   sudo apt install -y \
#     libnpp-dev-13-3 libnvjpeg-dev-13-3 \
#     libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
#     libturbojpeg0-dev libpng-dev libwebp-dev \
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
# Configure, build, test. `CMakePresets.json` also holds `debug`, `check`, `relwithdebinfo` and the GPU presets `cuda`
# and `rocm`, each building into `build_<preset>`; machine specifics such as the compiler path go into a gitignored
# `CMakeUserPresets.json`. Export `CMAKE_CXX_COMPILER_LAUNCHER=ccache` and `CMAKE_CUDA_COMPILER_LAUNCHER=ccache` to
# cache objects across trees.
#
#   cmake --preset cuda
#   cmake --build --preset cuda
#   ctest --preset cuda
#
# Feature gates.
#
# `-DUFORM_CUDA_ARCH=all` adds the NVIDIA residency and `-DUFORM_ROCM_ARCH=all` the AMD one; `-DUFORM_BUILD_TRAIN=ON`
# adds the trainer, the mesh and the loaders over CUDA, which the `cuda` preset and the machine profiles set. Without
# them only the host libraries and the
# media codecs are needed, and with `-DUFORM_WITH_CODECS=OFF`, the default on WebAssembly, not even the codecs. Every
# native dependency above is required and fails the configure loudly if absent — each is available on every target,
# including consumer / workstation GPUs. InfiniBand is the sole optional path (absent on those cards):
# `-DUFORM_WITH_INFINIBAND=AUTO|ON|OFF` (default `AUTO` — probe and enable when present). `-DUFORM_BUILD_TEST=ON`,
# `-DUFORM_BUILD_TOOLS=ON` and `-DUFORM_BUILD_BENCH=ON` add the suites in `test/`, the trainer in `tools/` and the
# benchmarks in `bench/`, which every build leaves out unless asked.
#
# Targets. `-DUFORM_MARCH=native`, an ISA level like `x86-64-v3` or a CPU name like `graniterapids` raises the floor of
# UForm's own units and its libraries, and on WebAssembly picks the module's `serial`, `v128` or `v128relaxed`
# capability. `-DUFORM_MULTIVERSIONED=OFF` builds for that floor alone: only the CPU classes it implies exist, and the
# newest of each lineage replaces its parents, `serial` included. `-DUFORM_CUDA_ARCH=100f` compiles one GPU code as
# SASS, and with it only the GPU classes it runs, `native` the family code of each visible GPU, and `all` the release
# set. The `xeon4_hopper`, `xeon6_blackwell`, `grace_hopper` and `vera_rubin` presets set all three for one machine.
cmake_minimum_required(VERSION 3.28)

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
set(UFORM_CUDA_ARCH
    ""
    CACHE STRING "NVIDIA code the residency and every CUDA unit build for: empty for none, all, native or one"
)
set(UFORM_ROCM_ARCH
    ""
    CACHE STRING "AMD target the residency builds for: empty for none, all, native or one"
)
option(UFORM_BUILD_TRAIN "Add the NVIDIA trainer, the mesh and the loaders, which link the CUDA libraries" OFF)
option(UFORM_BUILD_SHARED "Compile uform_static and uform_shared in the default build and install them" OFF)
option(UFORM_BUILD_STUB "Build uform_static_stub and uform_shared_stub" OFF)
option(UFORM_BUILD_PYTHON "Build the CPython extension, set by scikit-build" OFF)
option(UFORM_BUILD_NODE "Build the Node addon, set by cmake-js" OFF)
option(UFORM_BUILD_SWIFT "Package the static library for SwiftPM, as an XCFramework or an artifact bundle" OFF)
option(UFORM_BUILD_TEST "Compile the conformance suites in test/" OFF)
option(UFORM_BUILD_TOOLS "Compile the trainer in tools/" OFF)
option(UFORM_BUILD_BENCH "Compile the benchmarks in bench/" OFF)
option(UFORM_WERROR "Treat compiler warnings as errors" OFF)
# WebAssembly takes already-decoded media, so only native builds default to the host codecs.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm")
    set(uform_codecs_default_ OFF)
else ()
    set(uform_codecs_default_ ON)
endif ()
option(UFORM_WITH_CODECS "Decode images, video and audio through the host codecs the verbs over files need"
       ${uform_codecs_default_}
)
# Seeded from the environment at the first configure, so `pip`, `cargo` and other wrappers honour it.
set(UFORM_MARCH
    "$ENV{UFORM_MARCH}"
    CACHE STRING "The `-march` floor of UForm and its libraries: empty, native, a CPU, or serial, v128 or v128relaxed"
)
option(UFORM_MULTIVERSIONED "Keep every CPU class above the floor, or only the best ones UFORM_MARCH implies" ON)

if (UFORM_BUILD_TRAIN AND NOT UFORM_CUDA_ARCH)
    message(FATAL_ERROR "UFORM_BUILD_TRAIN needs UFORM_CUDA_ARCH, as the trainer runs on NVIDIA alone")
endif ()
if (UFORM_BUILD_TRAIN AND NOT UFORM_WITH_CODECS)
    message(FATAL_ERROR "UFORM_BUILD_TRAIN needs UFORM_WITH_CODECS, as the loaders decode on the host too")
endif ()

include(cmake/uform_gpu.cmake)

set(CMAKE_CXX_EXTENSIONS OFF)
# No target uses C++20 modules, and a Clang installed without `clang-scan-deps` would fail every C++ unit.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include(cmake/uform_flags.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

include(FetchContent)
include(cmake/find_library.cmake)

include(cmake/uform_siblings.cmake)

set(YYJSON_INSTALL OFF)
FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.13.0
    EXCLUDE_FROM_ALL SYSTEM
)
FetchContent_MakeAvailable(yyjson)
if (UFORM_BUILD_TEST
    OR UFORM_BUILD_TOOLS
    OR UFORM_BUILD_BENCH
)
    include(cmake/fmt.cmake)
endif ()

include(cmake/uform_platform.cmake)

# UForm's CUDA units take their own copy of the architecture list after the libraries resolved theirs. `all` is the
# release set, one cubin per generation plus an Ampere PTX tail, from Ampere up since UForm's kernels need it. "native"
# maps each visible GPU to the code its capability rows need, family suffix included, where CMake's own resolution would
# drop it; eight cards of one kind still compile one code. A native or named code compiles as SASS alone.
if (UFORM_CUDA_ARCH)
    set(uform_cuda_native_codes_
        75-real=75-real
        80-real=80-real
        86-real=86-real
        89-real=89-real
        90-real=90a-real
        100-real=100f-real
        103-real=103f-real
        110-real=110f-real
        120-real=120f-real
        121-real=121f-real
    )
    if (UFORM_CUDA_ARCH STREQUAL "all")
        set(uform_cuda_architectures_ 80-real 89-real 90a-real 100f-real 120f-real 80-virtual)
    elseif (UFORM_CUDA_ARCH STREQUAL "native")
        set(uform_cuda_architectures_)
        foreach (visible_ IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
            set(native_code_)
            foreach (pair_ IN LISTS uform_cuda_native_codes_)
                if (pair_ MATCHES "^${visible_}=(.+)$")
                    set(native_code_ ${CMAKE_MATCH_1})
                endif ()
            endforeach ()
            if (NOT native_code_)
                message(FATAL_ERROR "UFORM_CUDA_ARCH=native has no code for the visible ${visible_} GPU; "
                                    "name its code instead"
                )
            endif ()
            if (NOT native_code_ IN_LIST uform_cuda_architectures_)
                list(APPEND uform_cuda_architectures_ ${native_code_})
            endif ()
        endforeach ()
    elseif (UFORM_CUDA_ARCH MATCHES "^[0-9]+[af]?$")
        set(uform_cuda_architectures_ ${UFORM_CUDA_ARCH}-real)
    else ()
        message(FATAL_ERROR "UFORM_CUDA_ARCH takes all, native or one code like 100f, not `${UFORM_CUDA_ARCH}`")
    endif ()
    if (CMAKE_VERSION VERSION_LESS 3.31 AND uform_cuda_architectures_ MATCHES "1[0-9][0-9][af]")
        message(FATAL_ERROR "Three-digit \"a\" and \"f\" CUDA codes need CMake 3.31, found ${CMAKE_VERSION}")
    endif ()
    set(CMAKE_CUDA_ARCHITECTURES "${uform_cuda_architectures_}")
    # NVCC's default leaves Turing and Ampere cubins uncompressed, ten times their compressed size.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>)
endif ()

include(cmake/uform_dependencies.cmake)

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
if (UFORM_BUILD_TOOLS AND UFORM_BUILD_TRAIN)
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

include(cmake/uform_install.cmake)
