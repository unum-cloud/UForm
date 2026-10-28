# Host codecs, InfiniBand, and the vendor libraries the loaders link: cuDF and NVDEC.

# Host codecs: libjpeg-turbo / libpng / libwebp decode images on the CPU, libav* demuxes video and decodes audio (no GPU
# audio codec exists). Without UFORM_WITH_CODECS, as on WebAssembly, a build takes already-decoded media and discovers
# none of them.
if (UFORM_WITH_CODECS)
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

if (UFORM_BUILD_TRAIN)
    include(cmake/infiniband.cmake)
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
