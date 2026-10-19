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
    uform_host
    PROPERTIES C_VISIBILITY_PRESET hidden
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
)

if (UFORM_BUILD_CUDA)
    set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)

    # Compiles @p source device-only, with @p target's includes, definitions and options, into a fatbin of the CUDA
    # architectures `CODES` lists, and links it into @p target in a read-only section through an assembler stub, under
    # `<name>_bundle_start` and `<name>_bundle_end`, so it travels into archives too. With no codes the bundle is empty,
    # and every device refuses to load it.
    function (uform_gpu_bundle target name source)
        cmake_parse_arguments(arg "" "" "CODES" ${ARGN})
        set(bundle_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${name}.fatbin)
        if (arg_CODES)
            add_library(${name}_device OBJECT ${source})
            set_target_properties(
                ${name}_device PROPERTIES CUDA_ARCHITECTURES "${arg_CODES}" CUDA_FATBIN_COMPILATION ON
            )
            target_include_directories(${name}_device PRIVATE $<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>)
            target_compile_definitions(${name}_device PRIVATE $<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>)
            target_compile_options(${name}_device PRIVATE $<TARGET_PROPERTY:${target},COMPILE_OPTIONS>)
            add_dependencies(${target} ${name}_device)
            # The fatbin lands at the one path the stub includes and depends on, as neither takes a generator
            # expression.
            add_custom_command(
                OUTPUT ${bundle_}
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${name}_device> ${bundle_}
                DEPENDS $<TARGET_OBJECTS:${name}_device>
                VERBATIM
            )
        else ()
            file(WRITE ${bundle_} "")
        endif ()
        set(stub_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${name}_bundle.S)
        file(
            WRITE ${stub_}
            ".section .rodata\n.balign 4096\n.globl ${name}_bundle_start\n${name}_bundle_start:\n"
            ".incbin \"${bundle_}\"\n.globl ${name}_bundle_end\n${name}_bundle_end:\n"
            ".section .note.GNU-stack,\"\",@progbits\n"
        )
        set_source_files_properties(${stub_} PROPERTIES LANGUAGE ASM OBJECT_DEPENDS ${bundle_})
        target_sources(${target} PRIVATE ${stub_})
    endfunction ()

    # The configured generations that run `multimem`, Hopper and the Blackwell datacenter parts, each compiled for its own
    # code alone, and both under "native".
    set(uform_multicast_codes_)
    if ("90a-real" IN_LIST UFORM_CUDA_ARCHITECTURES OR "native" IN_LIST UFORM_CUDA_ARCHITECTURES)
        list(APPEND uform_multicast_codes_ 90a-real)
    endif ()
    if ("100f-real" IN_LIST UFORM_CUDA_ARCHITECTURES OR "native" IN_LIST UFORM_CUDA_ARCHITECTURES)
        list(APPEND uform_multicast_codes_ 100f-real)
    endif ()

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
    # The `multimem` kernels compile for the multicast-capable codes alone, and a domain loads them by name.
    uform_gpu_bundle(uform_core uform_multimem_switch core/multimem_switch.cu CODES ${uform_multicast_codes_})

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
        uform_loaders STATIC
        loaders/dataset_shards.cpp
        loaders/direct_storage.cpp
        loaders/tokens_loader.cu
        loaders/image_loader.cu
        loaders/video_loader.cu
        loaders/audio_loader.cu
        loaders/paired_loader.cu
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

    # `train/datasets.cu` → `uform::datasets`: the mixtures and the batch source over the loaders, kept out of
    # `uform::train`, which the C library links without the loaders or cuDF.
    add_library(uform_datasets STATIC train/datasets.cu)
    add_library(uform::datasets ALIAS uform_datasets)
    target_link_libraries(uform_datasets PUBLIC uform::train uform::loaders)
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
endif ()
if (UFORM_BUILD_ROCM)
    add_library(uform_amd_inference OBJECT inference/vram_amd_kernels.hip)
    target_include_directories(uform_amd_inference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(uform_amd_inference PRIVATE numkong::static forkunion::header stringzilla::static)
    set_target_properties(uform_amd_inference PROPERTIES HIP_VISIBILITY_PRESET hidden)
    set(UFORM_AMD_INFERENCE_KERNELS &vram_amd_kernels_k)
endif ()
file(
    CONFIGURE
    OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/inference/vram_vendors.cpp"
    CONTENT
    [=[#include "inference/entries.hpp"

namespace ui = unum::uform::inference;
ui::device_kernels_t const *const ui::device_kernels_k[2] = {@UFORM_NVIDIA_INFERENCE_KERNELS@, @UFORM_AMD_INFERENCE_KERNELS@};
]=]
    @ONLY
)
target_sources(uform_host PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/inference/vram_vendors.cpp")
