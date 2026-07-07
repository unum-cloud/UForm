# UForm — CUDA-first multimodal training stack.
#
# The core cell is a static library of `.cuh` interfaces + `.cu` kernel translation units; this file wires its source
# dependencies and exposes the `uform` / `uform_loaders` targets every consumer, test, and bench links.
#
# Prerequisites — native libraries (apt).
# The CUDA toolkit must be 13.3+ on `PATH`: `nvcc` needs to be >= 13.1 for `-std=c++23` (see the standard block below).
# The machine default lives in `/etc/profile.d/cuda.sh`, pointing `PATH` / `LD_LIBRARY_PATH` at the `/usr/local/cuda`
# symlink; repoint that symlink to switch toolkits globally. Then the once-per-machine native set:
#
#   sudo apt install -y \
#     libcudnn9-dev-cuda-13 libnpp-dev-13-3 libnvjpeg-dev-13-3 libcufile-dev-13-3 \
#     libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
#     libjpeg-turbo8-dev libpng-dev libwebp-dev libarrow-dev libparquet-dev \
#     libibverbs-dev ccache
#
# The Arrow dev packages come from the Apache Arrow apt repository (apache-arrow-apt-source), not the base archive.
#
# Prerequisites — RAPIDS cuDF (pip wheel into a repo-local venv).
# libcudf ships as a self-contained cu13 wheel (deliberately not `find_package(cudf)` — see the discovery block):
#
#   uv venv .venv && uv pip install --python .venv --extra-index-url https://pypi.nvidia.com libcudf-cu13
#
# CMake auto-discovers `$VIRTUAL_ENV`, then the repo `.venv`, then the interpreter's site-packages; override the search
# root with `-DUFORM_SITE_PACKAGES=<dir>`, or point straight at the libraries with `-DUFORM_CUDF_ROOT=<prefix>`.
#
# Configure, build, test.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --parallel
#   ctest --test-dir build --output-on-failure
#
# Local dependency forks (optional).
#
#   cp cmake/local.cmake.example cmake/local.cmake   # then point the UFORM_*_ROOT variables at working checkouts
#
# Feature gates.
#
# Every native dependency above is required and fails the configure loudly if absent — each is available on every
# target, including consumer / workstation GPUs. InfiniBand is the sole optional path (absent on those cards):
# `-DUFORM_WITH_INFINIBAND=AUTO|ON|OFF` (default `AUTO` — probe and enable when present).
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

project(
    uform
    VERSION 4.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Pocket-sized multimodal AI for content understanding and generation"
    HOMEPAGE_URL "https://github.com/unum-cloud/uform"
)

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
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependency.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_library.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe_infiniband.cmake")

# Per-checkout dependency overrides live in `cmake/local.cmake` — the analog of a Rust `.cargo/config.toml`: explicit,
# git-ignored, never inferred from sibling directories. Copy `cmake/local.cmake.example` and point the `UFORM_*_ROOT`
# variables at your working checkouts; without the file every dependency is fetched pinned from its upstream. Included
# before any resolution, so a `set(... CACHE ... FORCE)` there wins over stale caches.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/local.cmake" OPTIONAL)

# Source dependencies — a local checkout wins, otherwise a pinned download (see `cmake/dependency.cmake`). Lean by
# design: NumKong spans the tensor substrate, StringZilla parses exchange payloads, yyjson reads checkpoint headers,
# cudnn-frontend is the fused-attention graph DSL, fork_union is the host-side demux/decode thread pool.
uform_dependency(
    NAME numkong
    MODE SUBDIR
    ROOT_VAR UFORM_NUMKONG_ROOT
    REPO https://github.com/ashvardanian/NumKong.git
    TAG main-dev
)
uform_dependency(
    NAME stringzilla
    MODE SUBDIR
    ROOT_VAR UFORM_STRINGZILLA_ROOT
    REPO https://github.com/ashvardanian/StringZilla.git
    TAG main-dev
)
uform_dependency(
    NAME yyjson
    MODE SUBDIR
    ROOT_VAR UFORM_YYJSON_ROOT
    REPO https://github.com/ibireme/yyjson.git
    TAG 0.12.0
)
uform_dependency(
    NAME cudnn_frontend
    MODE HEADER
    ROOT_VAR UFORM_CUDNN_FRONTEND_ROOT
    INCLUDE_VAR UFORM_CUDNN_FRONTEND_INCLUDE
    REPO https://github.com/NVIDIA/cudnn-frontend.git
    TAG v1.19.1
)
uform_dependency(
    NAME fork_union
    MODE HEADER
    ROOT_VAR UFORM_FORKUNION_ROOT
    INCLUDE_VAR UFORM_FORKUNION_INCLUDE
    REPO https://github.com/ashvardanian/fork_union.git
    TAG v2.3.1
)

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

# `cuda_core` → `uform`: one static library for the whole core cell, compiled once, never per consumer.
# Dependencies stay lean: NumKong, StringZilla, yyjson, the CUDA Driver API (no runtime, no NCCL, no NVSHMEM),
# cuBLASLt, plus the fused-attention tier (nvrtc + cuDNN + cudnn-frontend) the transformer / tower cells need.
add_library(
    uform STATIC
    cuda_core/schedule.cu
    cuda_core/introspect.cu
    cuda_core/multimem_collectives.cu
    cuda_core/infiniband_collectives.cu
    cuda_core/multimodal_masking.cu
    cuda_core/checkpoint.cpp
    cuda_core/multimodal_batch.cu
    cuda_core/contrastive_segments.cu
    cuda_core/optimizer.cu
    cuda_core/transformer_block.cu
    cuda_core/encoder_tower.cu
)
target_include_directories(uform PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${UFORM_CUDNN_FRONTEND_INCLUDE}")
target_link_libraries(
    uform
    PUBLIC numkong
           stringzilla_header
           yyjson
           CUDA::cuda_driver
           CUDA::cublasLt
           CUDA::nvrtc
           "${UFORM_CUDNN_LIBRARY}"
)
target_compile_features(uform PUBLIC cxx_std_23)
# NumKong accessors are `constexpr` for device-callability; `nvcc` needs the (long-stable) flag to allow it.
target_compile_options(uform PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
if (UFORM_INFINIBAND_ENABLED)
    target_compile_definitions(uform PUBLIC UFORM_WITH_INFINIBAND=1)
    target_include_directories(uform PUBLIC "${UFORM_IBVERBS_INCLUDE_DIR}")
    target_link_libraries(uform PUBLIC "${UFORM_IBVERBS_LIBRARY}" "${UFORM_MLX5_LIBRARY}")
endif ()

# `cuda_train` → `uform_train`: the training stack compiled once, linked by the `uform_base` binary and the
# conformance test alike.
add_library(uform_train STATIC cuda_train/base.cu)
target_link_libraries(uform_train PUBLIC uform)
add_executable(uform_base cuda_train/base_main.cu)
target_link_libraries(uform_base PRIVATE uform_train)

# `cuda_test` → conformance binaries, one per cell; the contrastive and transformer suites rejoin with the
# contrastive rework.
enable_testing()
add_executable(cuda_test_mesh cuda_test/mesh.cpp)
target_link_libraries(cuda_test_mesh PRIVATE uform)
add_test(NAME mesh COMMAND cuda_test_mesh)

# The preflight probe every multi-node sbatch runs first: links the full training stack so merely starting
# on a node proves the runtime closure resolves there, then checks GPUDirect registration per GPU-NIC pair
# and the rendezvous directory — one named line per environment gap, before the real job spends a minute
# discovering it mid-suite.
add_executable(cuda_test_env cuda_test/env.cpp)
target_link_libraries(cuda_test_env PRIVATE uform_train)
add_test(NAME env COMMAND cuda_test_env)

add_executable(cuda_test_base cuda_test/base.cu)
target_link_libraries(cuda_test_base PRIVATE uform_train)
add_test(NAME base COMMAND cuda_test_base)

add_executable(cuda_test_mixer cuda_test/mixer.cu)
target_link_libraries(cuda_test_mixer PRIVATE uform)
add_test(NAME mixer COMMAND cuda_test_mixer)

add_executable(cuda_test_contrastive cuda_test/contrastive.cu)
target_link_libraries(cuda_test_contrastive PRIVATE uform)
add_test(NAME contrastive COMMAND cuda_test_contrastive)

# Every runnable resolves shared libraries beside itself first (`$ORIGIN`), so a `uform_bundle` tree on a
# shared filesystem runs on nodes whose native installs diverge from the build node's.
set_target_properties(
    uform_base cuda_test_env cuda_test_mesh cuda_test_base cuda_test_mixer cuda_test_contrastive PROPERTIES BUILD_RPATH
                                                                                                            "$ORIGIN"
)

# `uform_bundle` → a self-contained runtime tree (executables + their shared-library closure, glibc
# excepted) under `runtime-bundle/`, for Slurm or K8s clusters where nodes are not package-identical:
# deploy the tree to the shared filesystem and every node runs it, whatever it has installed natively.
# The executable list travels |-separated — a raw `;` cannot survive the custom-command argument path.
add_custom_target(
    uform_bundle
    COMMAND
        ${CMAKE_COMMAND}
        "-DBUNDLE_EXECUTABLES=$<TARGET_FILE:uform_base>|$<TARGET_FILE:cuda_test_env>|$<TARGET_FILE:cuda_test_mesh>|$<TARGET_FILE:cuda_test_base>|$<TARGET_FILE:cuda_test_mixer>|$<TARGET_FILE:cuda_test_contrastive>"
        "-DBUNDLE_DIR=${CMAKE_BINARY_DIR}/runtime-bundle" -P "${CMAKE_SOURCE_DIR}/cmake/bundle_runtime.cmake"
    DEPENDS uform_base cuda_test_env cuda_test_mesh cuda_test_base cuda_test_mixer cuda_test_contrastive
    COMMENT "Bundling executables with their runtime closure into runtime-bundle/"
    VERBATIM
)

# `cuda_loaders` → `uform_loaders`: GPU-native data ingestion, one translation unit per format / modality cell. Every
# vendor dependency below is available on every target and is hard-required, discovered through the layered
# `uform_find_library` search (`UFORM_*_ROOT` from `cmake/local.cmake`, then the RAPIDS / NVIDIA cu13 Python wheels,
# then system paths). A missing dependency fails the configure loudly naming the install.

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
uform_dependency(
    NAME ffnvcodec
    MODE HEADER
    INCLUDE_VAR UFORM_FFNVCODEC_INCLUDE
    REPO https://github.com/FFmpeg/nv-codec-headers.git
    TAG n13.0.19.0
)

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

# The CPU Arrow fallback inside `parquet.cpp` — the one cell where Apache Arrow lives, behind an exception firewall.
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
message(STATUS "uform: Arrow fallback (Arrow ${Arrow_VERSION})")

add_library(
    uform_loaders STATIC cuda_loaders/dataset_shards.cpp cuda_loaders/direct_storage.cpp cuda_loaders/tokens_loader.cu
                         cuda_loaders/image_loader.cu cuda_loaders/video_loader.cu cuda_loaders/audio_loader.cu
)
target_include_directories(uform_loaders PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(uform_loaders PUBLIC uform)

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

# Image decode: nvJPEG + NPP on the GPU, the three CPU codecs for the fallback. cuFFT drives the audio loader's
# GPU STFT (the log-mel frontend), so it links on the same GPU-vendor line.
target_link_libraries(
    uform_loaders PRIVATE CUDA::nvjpeg CUDA::nppig CUDA::nppc JPEG::JPEG PNG::PNG "${UFORM_WEBP_LIBRARY}"
)

# Video: NVDEC decode + libavformat/avcodec host demux to Annex-B (the nv-codec-headers carry the `cuvid*` decls). The
# libav headers live under the multiarch `/usr/include/<triple>` prefix.
target_include_directories(uform_loaders SYSTEM PRIVATE "${UFORM_FFNVCODEC_INCLUDE}")
target_link_libraries(
    uform_loaders PRIVATE "${UFORM_NVCUVID_LIBRARY}" "${UFORM_AVFORMAT_LIBRARY}" "${UFORM_AVCODEC_LIBRARY}"
                          "${UFORM_AVUTIL_LIBRARY}"
)

# Audio: libavcodec decode + libswresample resample to mono float (no GPU audio codec exists). Shares the libav* demux
# targets already linked above for video; adds the resampler.
target_link_libraries(uform_loaders PRIVATE "${UFORM_SWRESAMPLE_LIBRARY}")

# The host-side media demux / decode thread pool — header-only, used by the video and audio cells.
target_include_directories(uform_loaders SYSTEM PRIVATE "${UFORM_FORKUNION_INCLUDE}")

# Arrow CPU Parquet fallback, behind the exception firewall.
target_link_libraries(uform_loaders PRIVATE Arrow::arrow_shared Parquet::parquet_shared)

# Loader conformance binaries. A datasets suite rejoins when the loader tier stabilizes.
# `loaders` includes the media loader headers, which expose `cpu_decode_pool_t` (= `fu::basic_pool_t`), so it needs
# fork_union on its own include path — `uform_loaders` keeps that dependency PRIVATE.
add_executable(cuda_test_loaders cuda_test/loaders.cu)
target_link_libraries(cuda_test_loaders PRIVATE uform_loaders)
target_include_directories(cuda_test_loaders SYSTEM PRIVATE "${UFORM_FORKUNION_INCLUDE}")
add_test(NAME loaders COMMAND cuda_test_loaders)
