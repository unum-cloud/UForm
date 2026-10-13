# UForm — CUDA-first multimodal training stack.
#
# The core cell is a static library of `.cuh` interfaces + `.cu` kernel translation units; this file wires its source
# dependencies and exposes the `uform::core` / `uform::loaders` / `uform::train` targets every tool and test links.
#
# Prerequisites — native libraries (apt).
# The CUDA toolkit must be 13.3+ on `PATH`: `nvcc` needs to be >= 13.1 for `-std=c++23` (see the standard block below).
# The machine default lives in `/etc/profile.d/cuda.sh`, pointing `PATH` / `LD_LIBRARY_PATH` at the `/usr/local/cuda`
# symlink; repoint that symlink to switch toolkits globally. Then the once-per-machine native set:
#
#   sudo apt install -y \
#     libcudnn9-dev-cuda-13 libnpp-dev-13-3 libnvjpeg-dev-13-3 libcufile-dev-13-3 \
#     libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
#     libjpeg-turbo8-dev libpng-dev libwebp-dev \
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
# Every native dependency above is required and fails the configure loudly if absent — each is available on every
# target, including consumer / workstation GPUs. InfiniBand is the sole optional path (absent on those cards):
# `-DUFORM_WITH_INFINIBAND=AUTO|ON|OFF` (default `AUTO` — probe and enable when present). `-DUFORM_BUILD_TEST=OFF` and
# `-DUFORM_BUILD_TOOLS=OFF` skip `test/` and `tools/`, which only a top-level build compiles by default.
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
    LANGUAGES C CXX CUDA
    DESCRIPTION "Pocket-sized multimodal AI for content understanding and generation"
    HOMEPAGE_URL "https://github.com/unum-cloud/UForm"
)

option(UFORM_BUILD_TEST "Compile the conformance suites in test/" ${PROJECT_IS_TOP_LEVEL})
option(UFORM_BUILD_TOOLS "Compile the trainer in tools/" ${PROJECT_IS_TOP_LEVEL})

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

find_package(CUDAToolkit REQUIRED)
include(FetchContent)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_library.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe_infiniband.cmake")

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
# UForm builds the compiled library with CUDA for its own architectures.
set(NUMKONG_BUILD_CUDA ON)
set(NUMKONG_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set(NUMKONG_BUILD_TEST OFF)
set(NUMKONG_BUILD_BENCH OFF)
uform_sibling(numkong https://github.com/ashvardanian/NumKong c450acffcc001cbd10169516b79fe33f6f8ed154)
uform_sibling(stringzilla https://github.com/ashvardanian/StringZilla 7826f358a51df3871fba5cc825ebab3e1163174a)
uform_sibling(forkunion https://github.com/ashvardanian/ForkUnion 075d14137420f204ff175acd2d29c22e163e241c)

FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.12.0
)
# Header-only: a `SOURCE_SUBDIR` without a `CMakeLists.txt` populates the sources without configuring them.
FetchContent_Declare(
    cudnn_frontend
    GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
    GIT_TAG v1.19.1
    SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(yyjson cudnn_frontend)

# cuDNN powers the fused scaled-dot-product-attention graphs of the transformer blocks — available on every NVIDIA GPU,
# so hard-required alongside the frontend header DSL fetched above.
find_library(UFORM_CUDNN_LIBRARY cudnn REQUIRED)

# InfiniBand: the one optional transport (absent on consumer / workstation cards). Tri-state probe, see the module.
set(UFORM_WITH_INFINIBAND
    "AUTO"
    CACHE STRING "InfiniBand transport: ON (require), OFF (disable), AUTO (probe and enable when present)"
)
set_property(CACHE UFORM_WITH_INFINIBAND PROPERTY STRINGS ON OFF AUTO)
uform_probe_infiniband()

# `src/core` → `uform::core`: one static library for the whole core cell, compiled once, never per consumer.
# Dependencies stay lean: NumKong, StringZilla, yyjson, the CUDA Driver API (no NCCL, no NVSHMEM; the runtime arrives
# only through NumKong's kernel units), plus the fused-attention tier (nvrtc + cuDNN + cudnn-frontend) the transformer /
# tower cells need. `src/` is the one include root of all three libraries, so sources include `"core/types.hpp"`.
add_library(
    uform_core STATIC
    src/core/schedule.cu
    src/core/introspect.cu
    src/core/multimem_collectives.cu
    src/core/infiniband_collectives.cu
    src/core/multimodal_masking.cu
    src/core/checkpoint.cpp
    src/core/multimodal_batch.cu
    src/core/contrastive_segments.cu
    src/core/optimizer.cu
    src/core/transformer_block.cu
    src/core/encoder_tower.cu
)
add_library(uform::core ALIAS uform_core)
target_include_directories(uform_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" "${cudnn_frontend_SOURCE_DIR}/include")
target_link_libraries(
    uform_core PUBLIC numkong::static stringzilla::static yyjson CUDA::cuda_driver CUDA::nvrtc "${UFORM_CUDNN_LIBRARY}"
)
target_compile_features(uform_core PUBLIC cxx_std_23)
# NumKong accessors are `constexpr` for device-callability; `nvcc` needs the (long-stable) flag to allow it.
target_compile_options(uform_core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
if (UFORM_INFINIBAND_ENABLED)
    target_compile_definitions(uform_core PUBLIC UFORM_WITH_INFINIBAND=1)
    target_include_directories(uform_core PUBLIC "${UFORM_IBVERBS_INCLUDE_DIR}")
    target_link_libraries(uform_core PUBLIC "${UFORM_IBVERBS_LIBRARY}" "${UFORM_MLX5_LIBRARY}")
endif ()

# `src/train` → `uform::train`: the training stack compiled once, linked by the `uform_base` tool and the
# conformance suites alike.
add_library(uform_train STATIC src/train/base.cu)
add_library(uform::train ALIAS uform_train)
target_link_libraries(uform_train PUBLIC uform::core)

# `src/loaders` → `uform::loaders`: GPU-native data ingestion, one translation unit per format / modality cell. Every
# vendor dependency below is available on every target and is hard-required, discovered through the layered
# `uform_find_library` search (the RAPIDS / NVIDIA cu13 Python wheels, then system paths). A missing dependency fails
# the configure loudly naming the install.

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
    message(FATAL_ERROR "uform: libcufile not found — it ships with the CUDA toolkit; check the CUDA install / PATH.")
endif ()

# Image decode: nvJPEG decodes JPEG device-native from the reservoir tape and NPP (nppig) batch-crops/resizes, both from
# the CUDA toolkit; libjpeg-turbo / libpng / libwebp are the CPU fallback for PNG, WebP, and progressive/rejected JPEG.
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_library(UFORM_WEBP_LIBRARY webp REQUIRED)

# NVDEC: nv-codec-headers carry the `cuvid*` declarations without the Video Codec SDK EULA download; libnvcuvid is part
# of the NVIDIA driver, present on every target. (Loader wired now; video sources consumed later.)
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

# libav*: host-side demux for video, decode for audio (no GPU audio codec exists). Wired now, consumed later.
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

add_library(
    uform_loaders STATIC src/loaders/dataset_shards.cpp src/loaders/direct_storage.cpp src/loaders/tokens_loader.cu
                         src/loaders/image_loader.cu src/loaders/video_loader.cu src/loaders/audio_loader.cu
)
add_library(uform::loaders ALIAS uform_loaders)
target_link_libraries(uform_loaders PUBLIC uform::core)

# libcudf: `BEFORE`, and on the consumer directly — the wheels bundle their own CCCL 3.3 under `include/rapids`, which
# must precede the CUDA toolkit's older copy (propagated via the `CUDA::` imported targets) or every RMM header fails
# its version gate.
target_link_libraries(uform_loaders PRIVATE uform_cudf)
target_include_directories(
    uform_loaders SYSTEM BEFORE
    PRIVATE "${_uform_cudf_prefix}/libcudf/include/rapids" "${_uform_cudf_prefix}/libcudf/include"
            "${_uform_cudf_prefix}/librmm/include" "${_uform_cudf_prefix}/libkvikio/include"
            "${_uform_cudf_prefix}/rapids_logger/include"
)

# cuFile GPU Direct Storage.
target_link_libraries(uform_loaders PRIVATE "${UFORM_CUFILE_LIBRARY}")

# Image decode: nvJPEG + NPP on the GPU, the three CPU codecs for the fallback.
target_link_libraries(
    uform_loaders PRIVATE CUDA::nvjpeg CUDA::nppig CUDA::nppc JPEG::JPEG PNG::PNG "${UFORM_WEBP_LIBRARY}"
)

# Video: NVDEC decode + libavformat/avcodec host demux to Annex-B (the nv-codec-headers carry the `cuvid*` decls). The
# libav headers live under the multiarch `/usr/include/<triple>` prefix.
target_include_directories(uform_loaders SYSTEM PRIVATE "${ffnvcodec_SOURCE_DIR}/include")
target_link_libraries(
    uform_loaders PRIVATE "${UFORM_NVCUVID_LIBRARY}" "${UFORM_AVFORMAT_LIBRARY}" "${UFORM_AVCODEC_LIBRARY}"
                          "${UFORM_AVUTIL_LIBRARY}"
)

# Audio: libavcodec decode + libswresample resample to mono float (no GPU audio codec exists). Shares the libav* demux
# targets already linked above for video; adds the resampler.
target_link_libraries(uform_loaders PRIVATE "${UFORM_SWRESAMPLE_LIBRARY}")

# The host-side media demux and decode thread pool, public because the loader headers expose `cpu_decode_pool_t`.
target_link_libraries(uform_loaders PUBLIC forkunion::header)

# `tools/` → `uform_base`, the trainer binary.
if (UFORM_BUILD_TOOLS)
    add_executable(uform_base tools/base.cu)
    target_link_libraries(uform_base PRIVATE uform::train)
    list(APPEND _uform_executables uform_base)
endif ()

# `test/` → conformance binaries, one per cell.
if (UFORM_BUILD_TEST)
    enable_testing()
    add_executable(uform_test_mesh test/mesh.cpp)
    target_link_libraries(uform_test_mesh PRIVATE uform::core)
    add_executable(uform_test_mixer test/mixer.cu)
    target_link_libraries(uform_test_mixer PRIVATE uform::core)
    add_executable(uform_test_contrastive test/contrastive.cu)
    target_link_libraries(uform_test_contrastive PRIVATE uform::core)
    add_executable(uform_test_loaders test/loaders.cu)
    target_link_libraries(uform_test_loaders PRIVATE uform::loaders)
    add_executable(uform_test_base test/base.cu)
    target_link_libraries(uform_test_base PRIVATE uform::train)
    # The preflight probe every multi-node sbatch runs first: links the full training stack so merely starting
    # on a node proves the runtime closure resolves there, then checks GPUDirect registration per GPU-NIC pair
    # and the rendezvous directory — one named line per environment gap, before the real job spends a minute
    # discovering it mid-suite.
    add_executable(uform_test_env test/env.cpp)
    target_link_libraries(uform_test_env PRIVATE uform::train)
    foreach (test_name IN ITEMS mesh mixer contrastive loaders base env)
        add_test(NAME uform_test_${test_name} COMMAND uform_test_${test_name})
        list(APPEND _uform_executables uform_test_${test_name})
    endforeach ()
endif ()

if (_uform_executables)
    # Every runnable resolves shared libraries beside itself first (`$ORIGIN`), so a `uform_bundle` tree on a
    # shared filesystem runs on nodes whose native installs diverge from the build node's.
    set_target_properties(${_uform_executables} PROPERTIES BUILD_RPATH "$ORIGIN")

    # `uform_bundle` → a self-contained runtime tree (executables + their shared-library closure, glibc
    # excepted) under `runtime-bundle/`, for Slurm or K8s clusters where nodes are not package-identical:
    # deploy the tree to the shared filesystem and every node runs it, whatever it has installed natively.
    # The executable list travels |-separated — a raw `;` cannot survive the custom-command argument path.
    list(TRANSFORM _uform_executables REPLACE "^(.+)$" "$<TARGET_FILE:\\1>" OUTPUT_VARIABLE _uform_bundle_files)
    list(JOIN _uform_bundle_files "|" _uform_bundle_files)
    add_custom_target(
        uform_bundle
        COMMAND ${CMAKE_COMMAND} "-DBUNDLE_EXECUTABLES=${_uform_bundle_files}"
                "-DBUNDLE_DIR=${CMAKE_BINARY_DIR}/runtime-bundle" -P "${CMAKE_SOURCE_DIR}/cmake/bundle_runtime.cmake"
        DEPENDS ${_uform_executables}
        COMMENT "Bundling executables with their runtime closure into runtime-bundle/"
        VERBATIM
    )
endif ()
