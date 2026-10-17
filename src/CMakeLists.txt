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
    add_library(uform_train STATIC train/base.cu)
    add_library(uform::train ALIAS uform_train)
    target_link_libraries(uform_train PUBLIC uform::core)

    # `src/loaders` → `uform::loaders`: GPU-native data ingestion, one translation unit per format / modality cell,
    # over the vendor libraries the top-level file discovers.
    add_library(
        uform_loaders STATIC loaders/dataset_shards.cpp loaders/direct_storage.cpp loaders/tokens_loader.cu
                             loaders/image_loader.cu loaders/video_loader.cu loaders/audio_loader.cu
    )
    add_library(uform::loaders ALIAS uform_loaders)
    target_link_libraries(uform_loaders PUBLIC uform::core)

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
    # demux targets already linked above for video; adds the resampler.
    target_link_libraries(uform_loaders PRIVATE "${UFORM_SWRESAMPLE_LIBRARY}")

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
