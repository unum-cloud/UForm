# `uform_host` → the verbs and handles every build carries, compiled once into objects that `c/` assembles into
# `uform_static` and `uform_shared`. `src/` and `c/` are its include roots, so sources include `"core/types.hpp"` and
# `"uform/types.h"`. Only `UFORM_EXPORT` symbols leave the shared library.
add_library(
    uform_host OBJECT
    types.cpp
    model.cpp
    encode.cpp
    core/checkpoint.cpp
    inference/geometry.cpp
    inference/entries.cpp
    inference/tower.cpp
    media/planning.cpp
    media/resize.cpp
)
target_include_directories(uform_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/c")
# Training runs on NVIDIA alone: without CUDA its verbs refuse, and with it `uform_nvidia_train` below serves them.
if (NOT UFORM_BUILD_CUDA)
    target_sources(uform_host PRIVATE train_absent.cpp)
endif ()
target_link_libraries(uform_host PUBLIC numkong::static forkunion::header stringzilla::static yyjson)

# The host decoders and the verbs over encoded files, over the codecs the top-level file discovers; WebAssembly takes
# already-decoded media instead, and its verbs over files report themselves unsupported.
if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm")
    uform_find_library(UFORM_SWSCALE_LIBRARY swscale)
    if (NOT UFORM_SWSCALE_LIBRARY)
        message(FATAL_ERROR "uform: libswscale not found — install the ffmpeg dev package: "
                            "sudo apt install libswscale-dev."
        )
    endif ()
    uform_find_library(UFORM_TURBOJPEG_LIBRARY turbojpeg)
    if (NOT UFORM_TURBOJPEG_LIBRARY)
        message(FATAL_ERROR "uform: libturbojpeg not found — install the TurboJPEG dev package: "
                            "sudo apt install libturbojpeg0-dev."
        )
    endif ()
    target_sources(uform_host PRIVATE encode_media.cpp media/image.cpp media/video.cpp media/audio.cpp)
    target_link_libraries(
        uform_host
        PUBLIC "${UFORM_TURBOJPEG_LIBRARY}"
               PNG::PNG
               "${UFORM_WEBP_LIBRARY}"
               "${UFORM_AVFORMAT_LIBRARY}"
               "${UFORM_AVCODEC_LIBRARY}"
               "${UFORM_AVUTIL_LIBRARY}"
               "${UFORM_SWRESAMPLE_LIBRARY}"
               "${UFORM_SWSCALE_LIBRARY}"
    )
else ()
    target_sources(uform_host PRIVATE encode_media_absent.cpp)
endif ()
target_compile_features(uform_host PUBLIC cxx_std_23)
target_compile_definitions(uform_host PRIVATE UFORM_EXPORTING)
set_target_properties(
    uform_host PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
)

# `test/` → the host tower's suites: `uform_tiny_checkpoint` writes the seeded checkpoint once per `ctest` run as a
# fixture, which the parity test here and the C test in `c/` load.
if (UFORM_BUILD_TEST)
    add_executable(uform_tiny_checkpoint "${PROJECT_SOURCE_DIR}/test/tiny_checkpoint.cpp")
    add_test(NAME uform_tiny_checkpoint COMMAND uform_tiny_checkpoint "${PROJECT_BINARY_DIR}/tiny.safetensors")
    set_tests_properties(uform_tiny_checkpoint PROPERTIES FIXTURES_SETUP uform_tiny_checkpoint)
    add_executable(uform_test_inference "${PROJECT_SOURCE_DIR}/test/inference.cpp")
    target_link_libraries(uform_test_inference PRIVATE uform_host)
    add_test(NAME uform_test_inference COMMAND uform_test_inference "${PROJECT_BINARY_DIR}/tiny.safetensors")
    set_tests_properties(uform_test_inference PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint)
endif ()

if (UFORM_BUILD_CUDA)
    # `src/core` → `uform::core`: one static library for the whole core cell, compiled once, never per consumer.
    # Dependencies stay lean: NumKong, StringZilla, yyjson, the CUDA Driver API (no NCCL, no NVSHMEM; the runtime
    # arrives only through NumKong's kernel units), plus the fused-attention tier (nvrtc + cuDNN + cudnn-frontend) the
    # transformer / tower cells need. `src/` is the one include root of all three libraries, so sources include
    # `"core/types.hpp"`.
    add_library(
        uform_core STATIC
        core/schedule.cu
        core/introspect.cu
        core/multimem_collectives.cu
        core/infiniband_collectives.cu
        core/multimodal_masking.cu
        core/checkpoint.cpp
        core/multimodal_batch.cu
        core/contrastive_segments.cu
        core/optimizer.cu
        core/transformer_block.cu
        core/encoder_tower.cu
    )
    add_library(uform::core ALIAS uform_core)
    target_include_directories(uform_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${cudnn_frontend_SOURCE_DIR}/include")
    target_link_libraries(
        uform_core PUBLIC numkong::static stringzilla::static yyjson CUDA::cuda_driver CUDA::nvrtc
                          "${UFORM_CUDNN_LIBRARY}"
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
    add_library(uform_train STATIC train/base.cu train/trainer.cu)
    add_library(uform::train ALIAS uform_train)
    target_link_libraries(uform_train PUBLIC uform::core)
    # The items source plans and decodes caller media through the host decoders `uform_host` compiled.
    target_link_libraries(uform_train PRIVATE uform_host)

    # `train.cpp` → the training verbs, which `c/` links into both library flavors. They hold the composition by
    # value, so they compile as CUDA like the composition itself.
    add_library(uform_nvidia_train OBJECT train.cpp)
    set_source_files_properties(train.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(uform_nvidia_train PRIVATE uform_host uform::train)
    target_compile_definitions(uform_nvidia_train PRIVATE UFORM_EXPORTING)
    set_target_properties(uform_nvidia_train PROPERTIES CUDA_VISIBILITY_PRESET hidden)

    # `src/loaders` → `uform::loaders`: GPU-native data ingestion, one translation unit per format / modality cell,
    # over the vendor libraries the top-level file discovers.
    add_library(
        uform_loaders STATIC loaders/dataset_shards.cpp loaders/direct_storage.cpp loaders/tokens_loader.cu
                             loaders/image_loader.cu loaders/video_loader.cu loaders/audio_loader.cu
    )
    add_library(uform::loaders ALIAS uform_loaders)
    target_link_libraries(uform_loaders PUBLIC uform::core)
    # The loaders call the host decoders, so the archive takes the objects `uform_host` compiled rather than its own.
    target_link_libraries(uform_loaders PRIVATE uform_host)

    # libcudf: `BEFORE`, and on the consumer directly — the wheels bundle their own CCCL 3.3 under `include/rapids`,
    # which must precede the CUDA toolkit's older copy (propagated via the `CUDA::` imported targets) or every RMM
    # header fails its version gate.
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

    # Video: NVDEC decode + libavformat/avcodec host demux to Annex-B (the nv-codec-headers carry the `cuvid*` decls).
    # The libav headers live under the multiarch `/usr/include/<triple>` prefix.
    target_include_directories(uform_loaders SYSTEM PRIVATE "${ffnvcodec_SOURCE_DIR}/include")
    target_link_libraries(
        uform_loaders PRIVATE "${UFORM_NVCUVID_LIBRARY}" "${UFORM_AVFORMAT_LIBRARY}" "${UFORM_AVCODEC_LIBRARY}"
                              "${UFORM_AVUTIL_LIBRARY}"
    )

    # Audio: libavcodec decode + libswresample resample to mono float (no GPU audio codec exists). Shares the libav*
    # demux targets already linked above for video; adds the resampler, and the scaler the host video decode uses.
    target_link_libraries(uform_loaders PRIVATE "${UFORM_SWRESAMPLE_LIBRARY}" "${UFORM_SWSCALE_LIBRARY}")

    # The host-side media demux and decode thread pool, public because the loader headers expose `cpu_decode_pool_t`.
    target_link_libraries(uform_loaders PUBLIC forkunion::header)

    # `tools/` → `uform_base`, the trainer binary.
    if (UFORM_BUILD_TOOLS)
        add_executable(uform_base "${PROJECT_SOURCE_DIR}/tools/base.cu")
        target_link_libraries(uform_base PRIVATE uform::train)
        list(APPEND _uform_executables uform_base)
    endif ()

    # `test/` → conformance binaries, one per cell.
    if (UFORM_BUILD_TEST)
        add_executable(uform_test_mesh "${PROJECT_SOURCE_DIR}/test/mesh.cpp")
        target_link_libraries(uform_test_mesh PRIVATE uform::core)
        add_executable(uform_test_mixer "${PROJECT_SOURCE_DIR}/test/mixer.cu")
        target_link_libraries(uform_test_mixer PRIVATE uform::core)
        add_executable(uform_test_contrastive "${PROJECT_SOURCE_DIR}/test/contrastive.cu")
        target_link_libraries(uform_test_contrastive PRIVATE uform::core)
        add_executable(uform_test_loaders "${PROJECT_SOURCE_DIR}/test/loaders.cu")
        target_link_libraries(uform_test_loaders PRIVATE uform::loaders)
        add_executable(uform_test_base "${PROJECT_SOURCE_DIR}/test/base.cu")
        target_link_libraries(uform_test_base PRIVATE uform::train)
        # The preflight probe every multi-node sbatch runs first: links the full training stack so merely starting
        # on a node proves the runtime closure resolves there, then checks GPUDirect registration per GPU-NIC pair
        # and the rendezvous directory — one named line per environment gap, before the real job spends a minute
        # discovering it mid-suite.
        add_executable(uform_test_env "${PROJECT_SOURCE_DIR}/test/env.cpp")
        target_link_libraries(uform_test_env PRIVATE uform::train)
        foreach (test_name IN ITEMS mesh mixer contrastive loaders base env)
            add_test(NAME uform_test_${test_name} COMMAND uform_test_${test_name})
            list(APPEND _uform_executables uform_test_${test_name})
        endforeach ()
    endif ()

    if (_uform_executables)
        # Every runnable resolves shared libraries beside itself first (`$ORIGIN`), so a `uform_bundle` tree on a
        # shared filesystem runs on nodes whose native installs diverge from the build node's. The runnables stay
        # at the build root, where the sbatch scripts and the launch configurations look for them.
        set_target_properties(
            ${_uform_executables} PROPERTIES BUILD_RPATH "$ORIGIN" RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
        )

        # `uform_bundle` → a self-contained runtime tree (executables + their shared-library closure, glibc
        # excepted) under `runtime-bundle/`, for Slurm or K8s clusters where nodes are not package-identical:
        # deploy the tree to the shared filesystem and every node runs it, whatever it has installed natively.
        # The executable list travels |-separated — a raw `;` cannot survive the custom-command argument path.
        list(TRANSFORM _uform_executables REPLACE "^(.+)$" "$<TARGET_FILE:\\1>" OUTPUT_VARIABLE _uform_bundle_files)
        list(JOIN _uform_bundle_files "|" _uform_bundle_files)
        add_custom_target(
            uform_bundle
            COMMAND
                ${CMAKE_COMMAND} "-DBUNDLE_EXECUTABLES=${_uform_bundle_files}"
                "-DBUNDLE_DIR=${PROJECT_BINARY_DIR}/runtime-bundle" -P
                "${PROJECT_SOURCE_DIR}/cmake/bundle_runtime.cmake"
            DEPENDS ${_uform_executables}
            COMMENT "Bundling executables with their runtime closure into runtime-bundle/"
            VERBATIM
        )
    endif ()
endif ()

# Each GPU vendor's inference kernels, one object library per vendor, which `c/` links into both library flavors beside
# `uform_host`. The vendor table the tower dispatches through is generated, so it names exactly the vendors this build
# compiles, and keeps their objects referenced when a consumer links the archive.
set(UFORM_NVIDIA_INFERENCE_KERNELS nullptr)
set(UFORM_AMD_INFERENCE_KERNELS nullptr)
if (UFORM_BUILD_CUDA)
    add_library(uform_nvidia_inference OBJECT inference/vram_nvidia_kernels.cu)
    target_include_directories(uform_nvidia_inference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(uform_nvidia_inference PRIVATE numkong::static forkunion::header stringzilla::static)
    target_link_libraries(uform_nvidia_inference PUBLIC CUDA::cuda_driver)
    target_compile_options(uform_nvidia_inference PRIVATE --expt-relaxed-constexpr)
    set_target_properties(uform_nvidia_inference PROPERTIES CUDA_VISIBILITY_PRESET hidden)
    set(UFORM_NVIDIA_INFERENCE_KERNELS &vram_nvidia_kernels_k)
    if (UFORM_BUILD_TEST)
        target_link_libraries(uform_test_inference PRIVATE uform_nvidia_inference)
    endif ()
endif ()
if (UFORM_BUILD_ROCM)
    add_library(uform_amd_inference OBJECT inference/vram_amd_kernels.hip)
    target_include_directories(uform_amd_inference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(uform_amd_inference PRIVATE numkong::static forkunion::header stringzilla::static)
    set_target_properties(uform_amd_inference PROPERTIES HIP_VISIBILITY_PRESET hidden)
    set(UFORM_AMD_INFERENCE_KERNELS &vram_amd_kernels_k)
    if (UFORM_BUILD_TEST)
        target_link_libraries(uform_test_inference PRIVATE uform_amd_inference)
    endif ()
endif ()
file(
    CONFIGURE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/inference/vram_vendors.cpp"
    CONTENT
        [=[#include "inference/entries.hpp"

namespace ui = unum::uform::inference;
ui::device_kernels_t const *const ui::device_kernels_k[2] = {@UFORM_NVIDIA_INFERENCE_KERNELS@, @UFORM_AMD_INFERENCE_KERNELS@};
]=]
    @ONLY
)
target_sources(uform_host PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/inference/vram_vendors.cpp")
