# `uform_host` → the verbs and handles every build carries, compiled once into objects that `c/` assembles into
# `uform_static` and `uform_shared`. `src/`, `include/` and `c/` are its include roots, so sources include
# `"uform/utilities/vram_nvidia_schedule.cuh"`, `"uform/types.h"` and the flat `"lib.hpp"`. Only `UFORM_EXPORT` symbols leave the shared
# library.
add_library(
    uform_host OBJECT
    types.cpp
    model.cpp
    encode.cpp
    utilities/safetensors_format.cpp
    encoder/geometry.cpp
    media/planning.cpp
    media/resize.cpp
)
target_include_directories(
    uform_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/c"
)

# Compiles @p source_ device-only into a bundle for @p vendor_ - a CUDA fatbin through NVCC for "nvidia", a HIP offload
# bundle through HIP-Clang for "amd" - of the codes after `CODES`, with @p target_'s includes, definitions and options,
# and links it into @p target_ in a read-only section through an assembler stub, under `<name>_bundle_start` and
# `<name>_bundle_end`, so it travels into archives too. No pass registers it with a GPU runtime: the host unit loads it
# by name. With no codes the bundle is empty, and every device refuses to load it.
function (uform_gpu_bundle target_ vendor_ name_ source_)
    cmake_parse_arguments(PARSE_ARGV 4 arg "" "" "CODES")
    set(bundle_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${name_}.bundle)
    if (arg_CODES AND vendor_ STREQUAL "nvidia")
        add_library(${name_}_device OBJECT ${source_})
        set_target_properties(${name_}_device PROPERTIES CUDA_ARCHITECTURES "${arg_CODES}" CUDA_FATBIN_COMPILATION ON)
        target_include_directories(${name_}_device PRIVATE $<TARGET_PROPERTY:${target_},INCLUDE_DIRECTORIES>)
        target_compile_definitions(${name_}_device PRIVATE $<TARGET_PROPERTY:${target_},COMPILE_DEFINITIONS>)
        target_compile_options(${name_}_device PRIVATE $<TARGET_PROPERTY:${target_},COMPILE_OPTIONS>)
        add_dependencies(${target_} ${name_}_device)
        # The fatbin lands at the one path the stub includes and depends on, as neither takes a generator expression.
        add_custom_command(
            OUTPUT ${bundle_}
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${name_}_device> ${bundle_}
            DEPENDS $<TARGET_OBJECTS:${name_}_device>
            VERBATIM
        )
    elseif (arg_CODES)
        # Never through CMake's HIP language, whose fat binaries would register with `libamdhip64`.
        set(offload_)
        foreach (code_ IN LISTS arg_CODES)
            list(APPEND offload_ --offload-arch=${code_})
        endforeach ()
        add_custom_command(
            OUTPUT ${bundle_}
            COMMAND
                ${CMAKE_HIP_COMPILER} -x hip --cuda-device-only -c -std=c++23 -O3 ${offload_}
                "-I$<JOIN:$<TARGET_PROPERTY:${target_},INCLUDE_DIRECTORIES>,;-I>"
                "-D$<JOIN:$<TARGET_PROPERTY:${target_},COMPILE_DEFINITIONS>,;-D>" -MD -MF ${bundle_}.d
                ${CMAKE_CURRENT_SOURCE_DIR}/${source_} -o ${bundle_}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source_}
            DEPFILE ${bundle_}.d
            COMMENT "Compiling the ${name_} kernels"
            COMMAND_EXPAND_LISTS VERBATIM
        )
    else ()
        file(WRITE ${bundle_} "")
    endif ()
    set(stub_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${name_}_bundle.S)
    file(
        WRITE ${stub_}
        ".section .rodata\n.balign 4096\n.globl ${name_}_bundle_start\n${name_}_bundle_start:\n"
        ".incbin \"${bundle_}\"\n.globl ${name_}_bundle_end\n${name_}_bundle_end:\n"
        ".section .note.GNU-stack,\"\",@progbits\n"
    )
    set_source_files_properties(${stub_} PROPERTIES LANGUAGE ASM OBJECT_DEPENDS ${bundle_})
    target_sources(${target_} PRIVATE ${stub_})
endfunction ()

# The encoder's RAM variants compile per CPU class. A variant inlines its class's kits and host entries and runs on
# nothing older, so its unit takes the class's instruction set outright. A class is its parent plus a delta, the NumKong
# capabilities `numkong_capabilities_of` adds over `parent_class_of`, restated here row for row, and after `FORKUNION`
# the ForkUnion capabilities its waiter or atomic reference adds: it inherits its parent's flags, adds the flags of each
# NumKong delta, and exists only when its parent does and every delta is enabled, the switches after the probes
# included. Each NumKong capability it holds also reaches the unit as `NUMKONG_TARGET_<CAPABILITY>=1`, so its kits never
# depend on which macros a flag set implies; capabilities follow `OVER` as `name MACRO` pairs. An Arm or RISC-V compiler
# keeps only the last `-march=`, so those rows spell the union of their lineage after `MARCH`, which lands last. The
# `serial` class is the floor every CPU runs.
#
# UFORM_TARGET_CLASSES keeps every existing class when empty, `serial` and the named ones for a list, and under `native`
# the newest class of each lineage whose capabilities this host runs, as NumKong's and ForkUnion's probes found.
# `uform_host` reads each declared class as `UFORM_TARGET_<CLASS>`, 1 for a kept one and 0 otherwise.
set(uform_class_serial_exists_ TRUE)
set(uform_class_serial_runs_ TRUE)
set(uform_cpu_rows_)

# A WebAssembly module carries the one SIMD capability its toolchain declares, so its verdicts read the compiler's own
# macros rather than a NumKong probe.
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR CMAKE_SYSTEM_NAME STREQUAL "WASI")
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
        "#if !defined(__wasm_simd128__)\n#error\n#endif\nint main() { return 0; }" uform_target_v128_compiles_
    )
    check_cxx_source_compiles(
        "#if !defined(__wasm_relaxed_simd__)\n#error\n#endif\nint main() { return 0; }"
        uform_target_v128relaxed_compiles_
    )
endif ()

# Declares the CPU class @p class_, gated as `UFORM_TARGET_<target_macro>`, over the class named after `OVER`.
function (uform_cpu_class class_ target_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "OVER;MARCH" "FORKUNION")
    set(flags_ ${uform_class_${arg_OVER}_flags_})
    set(definitions_ ${uform_class_${arg_OVER}_definitions_})
    set(exists_ ${uform_class_${arg_OVER}_exists_})
    set(runs_ ${uform_class_${arg_OVER}_runs_})
    list(LENGTH arg_UNPARSED_ARGUMENTS entries_)
    if (entries_)
        math(EXPR last_ "${entries_} - 1")
        foreach (index_ RANGE 0 ${last_} 2)
            math(EXPR macro_index_ "${index_} + 1")
            list(GET arg_UNPARSED_ARGUMENTS ${index_} capability_)
            list(GET arg_UNPARSED_ARGUMENTS ${macro_index_} capability_macro_)
            if (numkong_target_${capability_}_enabled OR uform_target_${capability_}_compiles_)
                list(APPEND flags_ ${numkong_target_${capability_}_flags})
                list(APPEND definitions_ NUMKONG_TARGET_${capability_macro_}=1)
            else ()
                set(exists_ FALSE)
            endif ()
            if (NOT numkong_target_${capability_}_runs)
                set(runs_ FALSE)
            endif ()
        endforeach ()
    endif ()
    # ForkUnion's verdicts reach every class unit through `forkunion::cpu_capabilities_compiled`, so they add no flags.
    foreach (capability_ IN LISTS arg_FORKUNION)
        if (NOT forkunion_target_${capability_}_enabled)
            set(exists_ FALSE)
        endif ()
        if (NOT forkunion_target_${capability_}_runs)
            set(runs_ FALSE)
        endif ()
    endforeach ()
    if (arg_MARCH)
        list(APPEND flags_ -march=${arg_MARCH})
    endif ()
    if (UFORM_TARGET_CLASSES STREQUAL "native")
        set(kept_ ${runs_})
        # The newest class of a lineage stands in for its parent, whose unit then goes.
        if (runs_)
            set(uform_class_${arg_OVER}_kept_
                FALSE
                PARENT_SCOPE
            )
        endif ()
    elseif (UFORM_TARGET_CLASSES AND NOT class_ IN_LIST UFORM_TARGET_CLASSES)
        set(kept_ FALSE)
    else ()
        set(kept_ ${exists_})
    endif ()
    set(uform_class_${class_}_flags_
        "${flags_}"
        PARENT_SCOPE
    )
    set(uform_class_${class_}_definitions_
        "${definitions_}"
        PARENT_SCOPE
    )
    set(uform_class_${class_}_exists_
        ${exists_}
        PARENT_SCOPE
    )
    set(uform_class_${class_}_runs_
        ${runs_}
        PARENT_SCOPE
    )
    set(uform_class_${class_}_kept_
        ${kept_}
        PARENT_SCOPE
    )
    set(uform_class_${class_}_macro_
        ${target_macro_}
        PARENT_SCOPE
    )
    set(uform_cpu_rows_
        ${uform_cpu_rows_} ${class_}
        PARENT_SCOPE
    )
endfunction ()

uform_cpu_class(intel_haswell INTEL_HASWELL OVER serial haswell HASWELL)
uform_cpu_class(
    intel_sierra
    INTEL_SIERRA
    OVER
    intel_haswell
    alder
    ALDER
    sierra
    SIERRA
)
uform_cpu_class(intel_skylake INTEL_SKYLAKE OVER intel_haswell skylake SKYLAKE)
uform_cpu_class(intel_icelake INTEL_ICELAKE OVER intel_skylake icelake ICELAKE)
uform_cpu_class(amd_genoa AMD_GENOA OVER intel_icelake genoa GENOA)
uform_cpu_class(
    intel_sapphire
    INTEL_SAPPHIRE
    OVER
    amd_genoa
    sapphire
    SAPPHIRE
    sapphireamx
    SAPPHIREAMX
)
uform_cpu_class(intel_granite INTEL_GRANITE OVER intel_sapphire graniteamx GRANITEAMX)
uform_cpu_class(
    intel_diamond
    INTEL_DIAMOND
    OVER
    intel_granite
    alder
    ALDER
    sierra
    SIERRA
    diamond
    DIAMOND
)
uform_cpu_class(aws_g2 AWS_G2 OVER serial neon NEON)
uform_cpu_class(
    apple_m2
    APPLE_M2
    OVER
    aws_g2
    MARCH
    armv8.6-a+simd+dotprod+bf16+fp16+fp16fml
    neonsdot
    NEONSDOT
    neonbfdot
    NEONBFDOT
    neonfhm
    NEONFHM
)
uform_cpu_class(
    apple_m4
    APPLE_M4
    OVER
    apple_m2
    MARCH
    armv8.6-a+simd+dotprod+bf16+fp16+fp16fml+sme+sme2
    sme
    SME
    smebi32
    SMEBI32
)
uform_cpu_class(aws_g5 AWS_G5 OVER apple_m2 FORKUNION arm64_wfet)
uform_cpu_class(
    nvidia_vera
    NVIDIA_VERA
    OVER
    aws_g5
    MARCH
    armv8.6-a+simd+dotprod+bf16+fp16+fp16fml+fp8dot4
    neonfp8
    NEONFP8
)
uform_cpu_class(
    alibaba_c930
    ALIBABA_C930
    OVER
    serial
    MARCH
    rv64gcv_zvbb
    rvv
    RVV
    rvvbb
    RVVBB
)
uform_cpu_class(wasm32_v128 WASM32_V128 OVER serial v128 V128)
uform_cpu_class(wasm64_extended WASM64_EXTENDED OVER wasm32_v128 v128relaxed V128RELAXED)

set(uform_cpu_classes_ serial)
set(uform_cpu_gates_)
foreach (class_ IN LISTS uform_cpu_rows_)
    if (uform_class_${class_}_kept_)
        list(APPEND uform_cpu_classes_ ${class_})
    endif ()
    list(APPEND uform_cpu_gates_ "UFORM_TARGET_${uform_class_${class_}_macro_}=$<BOOL:${uform_class_${class_}_kept_}>")
endforeach ()
if (NOT UFORM_TARGET_CLASSES STREQUAL "native")
    foreach (class_ IN LISTS UFORM_TARGET_CLASSES)
        if (NOT class_ STREQUAL "serial" AND NOT uform_class_${class_}_kept_)
            message(
                FATAL_ERROR "UFORM_TARGET_CLASSES names ${class_}, which no row declares or this toolchain cannot build"
            )
        endif ()
    endforeach ()
endif ()

# The GPU classes compile no device code of their own: NumKong's libraries carry every generation's kernels, and one
# bundle per vendor carries UForm's entries. So a class is a host unit picking its generation's kits, every class of a
# built vendor exists, and the vendor-wide class `nvidia` or `amd`, on NumKong's generic kits, runs every device of the
# vendor without a class of its own. Their units see `NUMKONG_ARCH_<VENDOR>_` and NumKong's
# `numkong_<vendor>_capabilities_compiled_`, which declare the generations its library compiled, and `uform_host` reads
# each vendor as `UFORM_ARCH_CUDA_` or `UFORM_ARCH_ROCM_`, 1 when built and 0 otherwise. Rubin runs the Blackwell class
# through `parent_class_of`, so it compiles nothing of its own.
set(uform_gpu_gates_ "UFORM_ARCH_CUDA_=$<BOOL:${UFORM_BUILD_CUDA}>" "UFORM_ARCH_ROCM_=$<BOOL:${UFORM_BUILD_ROCM}>")
function (uform_nvidia_class class_)
    set(uform_class_${class_}_exists_
        TRUE
        PARENT_SCOPE
    )
    set(uform_nvidia_classes_
        ${uform_nvidia_classes_} ${class_}
        PARENT_SCOPE
    )
endfunction ()
function (uform_amd_class class_)
    set(uform_class_${class_}_exists_
        TRUE
        PARENT_SCOPE
    )
    set(uform_amd_classes_
        ${uform_amd_classes_} ${class_}
        PARENT_SCOPE
    )
endfunction ()
if (UFORM_BUILD_CUDA)
    set(uform_nvidia_classes_)
    set(uform_nvidia_definitions_ "NUMKONG_ARCH_CUDA_=1")
    uform_nvidia_class(nvidia)
    uform_nvidia_class(nvidia_ampere)
    uform_nvidia_class(nvidia_ada)
    uform_nvidia_class(nvidia_hopper)
    uform_nvidia_class(nvidia_blackwell)
    uform_nvidia_class(nvidia_blackwellrtx)
endif ()
if (UFORM_BUILD_ROCM)
    enable_language(ASM)
    set(uform_amd_classes_)
    set(uform_amd_definitions_ "NUMKONG_ARCH_ROCM_=1")
    uform_amd_class(amd)
    uform_amd_class(amd_cdna3)
    uform_amd_class(amd_cdna4)
    uform_amd_class(amd_cdna5)
endif ()

# Declares the @p modality_ variants resident in @p residency_ on @p backend_, over the classes after it that exist, as
# one object library of a unit per class, `<modality>/<residency>_variants_<class>.cpp`, compiled with its class's
# flags and definitions: `uform_<modality>_<residency>_variants` on the CPU, `uform_<modality>_<residency>_<backend>_variants`
# on a GPU vendor, whose classes share that vendor's bundle. The library joins `uform_<modality>_variants_`, which every
# consumer of `uform_host` links after it, so the baseline's copy of any shared inline function comes first.
function (uform_variants modality_ residency_ backend_)
    set(library_ uform_${modality_}_${residency_}_variants)
    if (NOT backend_ STREQUAL "cpu")
        set(library_ uform_${modality_}_${residency_}_${backend_}_variants)
    endif ()
    add_library(${library_} OBJECT)
    foreach (class_ IN LISTS ARGN)
        if (uform_class_${class_}_exists_)
            set(unit_ ${modality_}/${residency_}_variants_${class_}.cpp)
            target_sources(${library_} PRIVATE ${unit_})
            set_source_files_properties(
                ${unit_} PROPERTIES COMPILE_OPTIONS "${uform_class_${class_}_flags_}"
                                    COMPILE_DEFINITIONS "${uform_class_${class_}_definitions_}"
            )
        endif ()
    endforeach ()
    target_include_directories(
        ${library_} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/c"
    )
    target_compile_definitions(${library_} PRIVATE ${uform_${backend_}_definitions_})
    target_link_libraries(
        ${library_} PRIVATE numkong::static forkunion::header stringzilla::static smashtable::header yyjson
                            $<BUILD_INTERFACE:forkunion::cpu_capabilities_compiled>
    )
    target_compile_features(${library_} PRIVATE cxx_std_23)
    set_target_properties(${library_} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    set(uform_${modality_}_variants_
        ${uform_${modality_}_variants_} ${library_}
        PARENT_SCOPE
    )
endfunction ()

# The encoder's variant libraries, and each GPU vendor's entries: one host unit launching by name the kernels its
# bundle carries for every configured architecture, through the driver it opens at first use rather than links.
set(uform_encoder_variants_)
uform_variants(encoder ram cpu ${uform_cpu_classes_})
if (UFORM_BUILD_CUDA)
    uform_variants(encoder vram nvidia ${uform_nvidia_classes_})
    target_sources(uform_encoder_vram_nvidia_variants PRIVATE encoder/vram_entries_nvidia.cpp)
    target_link_libraries(
        uform_encoder_vram_nvidia_variants
        PRIVATE $<BUILD_INTERFACE:numkong_cuda_capabilities_compiled_>
        PUBLIC ${CMAKE_DL_LIBS}
    )
    uform_gpu_bundle(
        uform_encoder_vram_nvidia_variants nvidia uform_encoder_nvidia encoder/vram_kernels_nvidia.cu CODES
        ${CMAKE_CUDA_ARCHITECTURES}
    )
    target_link_libraries(
        uform_encoder_nvidia_device PRIVATE numkong::static forkunion::header stringzilla::static smashtable::header
    )
    target_compile_options(uform_encoder_nvidia_device PRIVATE --expt-relaxed-constexpr)
endif ()
if (UFORM_BUILD_ROCM)
    uform_variants(encoder vram amd ${uform_amd_classes_})
    target_sources(uform_encoder_vram_amd_variants PRIVATE encoder/vram_entries_amd.cpp)
    target_link_libraries(
        uform_encoder_vram_amd_variants
        PRIVATE $<BUILD_INTERFACE:numkong_rocm_capabilities_compiled_>
        PUBLIC ${CMAKE_DL_LIBS}
    )
    uform_gpu_bundle(
        uform_encoder_vram_amd_variants amd uform_encoder_amd encoder/vram_kernels_amd.hip CODES
        ${NUMKONG_ROCM_ARCHITECTURES}
    )
endif ()
target_compile_definitions(uform_host PUBLIC ${uform_cpu_gates_} ${uform_gpu_gates_})
# Training runs on NVIDIA alone: without UFORM_BUILD_TRAIN its verbs refuse, and with it `uform_nvidia_train` below
# serves them.
if (NOT UFORM_BUILD_TRAIN)
    target_sources(uform_host PRIVATE train_absent.cpp)
endif ()
target_link_libraries(
    uform_host PUBLIC numkong::static forkunion::header forkunion::static stringzilla::static smashtable::header yyjson
)
# Every unit naming a class's waiter or atomic reference reads ForkUnion's probe verdicts rather than its own flags, so
# the class units and the dispatch unit name the same rungs, and the dispatch unit admits them at run time.
target_link_libraries(uform_host PUBLIC $<BUILD_INTERFACE:forkunion::cpu_capabilities_compiled>)

# The host decoders and the verbs over encoded files, over the codecs the top-level file discovers; without
# UFORM_WITH_CODECS a build takes already-decoded media instead, and its verbs over files report themselves unsupported.
if (UFORM_WITH_CODECS)
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

# The training stack is the one part exempt from the vendor boundary: the trainer, the mesh and the loaders link the
# CUDA driver, NVML, nvJPEG, NPP, NVDEC and cuDF outright, so they build only under UFORM_BUILD_TRAIN, and the
# inference library, whose encoder variants open their drivers at first use, links them only then.
if (UFORM_BUILD_TRAIN)
    set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)

    # The resolved codes that run `multimem`: those NumKong's cached Hopper and datacenter-Blackwell capabilities list,
    # with or without CMake's "-real" or "-virtual", each compiled for its own code alone.
    set(uform_multicast_codes_)
    foreach (multicast_architecture_ IN LISTS numkong_target_hopper_architectures
                                              numkong_target_blackwell_architectures
    )
        foreach (architecture_ IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if (architecture_ MATCHES "^${multicast_architecture_}(-real|-virtual)?$"
                AND NOT "${multicast_architecture_}-real" IN_LIST uform_multicast_codes_
            )
                list(APPEND uform_multicast_codes_ ${multicast_architecture_}-real)
            endif ()
        endforeach ()
    endforeach ()

    # `src/train` → `uform::train`: the training stack compiled once, never per consumer, linked by the `uform_base`
    # tool and the conformance suites alike.
    # Dependencies stay lean: NumKong, StringZilla, yyjson and the CUDA Driver API, with no NCCL or NVSHMEM; the runtime
    # arrives only through NumKong's kernel units. `src/` and `include/` are the include roots of all three libraries,
    # so sources include `"uform/utilities/vram_nvidia_schedule.cuh"` and `"uform/utilities/types.hpp"`.
    add_library(
        uform_train STATIC
        utilities/vram_nvidia_schedule.cu
        train/introspect.cu
        utilities/multimem_collectives.cu
        utilities/infiniband_collectives.cu
        train/multimodal_masking.cu
        utilities/safetensors_format.cpp
        encoder/multimodal_batch.cu
        train/contrastive_segments.cu
        train/contrastive_tokens.cu
        train/pooled_spectrum.cu
        train/optimizer.cu
        train/transformer_block.cu
        train/encoder_tower.cu
        train/base.cu
        train/trainer.cu
    )
    add_library(uform::train ALIAS uform_train)
    target_include_directories(
        uform_train PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/c"
    )
    target_link_libraries(
        uform_train PUBLIC numkong::static stringzilla::static smashtable::header yyjson CUDA::cuda_driver
    )
    target_compile_features(uform_train PUBLIC cxx_std_23)
    # NumKong accessors are `constexpr` for device-callability; `nvcc` needs the (long-stable) flag to allow it.
    target_compile_options(uform_train PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if (uform_infiniband_enabled_)
        target_compile_definitions(uform_train PUBLIC UFORM_WITH_INFINIBAND=1)
        target_include_directories(uform_train PUBLIC "${UFORM_IBVERBS_INCLUDE_DIR}")
        target_link_libraries(uform_train PUBLIC "${UFORM_IBVERBS_LIBRARY}" "${UFORM_MLX5_LIBRARY}")
    endif ()
    # The `multimem` kernels compile for the multicast-capable codes alone, and a domain loads them by name.
    uform_gpu_bundle(
        uform_train nvidia uform_multimem_switch utilities/multimem_switch.cu CODES ${uform_multicast_codes_}
    )
    # The envelope probe reads this process's device bytes from NVML, which ships with the driver like libcuda.
    target_link_libraries(uform_train PRIVATE CUDA::nvml)
    # `gpu_numerics_of` fills each GPU class's table from the rungs NumKong compiled.
    target_link_libraries(uform_train PRIVATE $<BUILD_INTERFACE:numkong_cuda_capabilities_compiled_>)
    # The items source plans and decodes caller media through the host decoders `uform_host` compiled, and those
    # objects name the encoder's variants, so the archive carries them too.
    target_link_libraries(uform_train PRIVATE uform_host ${uform_encoder_variants_})

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
        loaders/choices_loader.cu
    )
    add_library(uform::loaders ALIAS uform_loaders)
    target_link_libraries(uform_loaders PUBLIC uform::train)
    # The loaders call the host decoders, so the archive takes the objects `uform_host` compiled rather than its own,
    # with the encoder's variants they name.
    target_link_libraries(uform_loaders PRIVATE uform_host ${uform_encoder_variants_})

    # libcudf: `BEFORE`, and on the consumer directly — the wheels bundle their own CCCL 3.3 under `include/rapids`,
    # which must precede the CUDA toolkit's older copy (propagated via the `CUDA::` imported targets) or every RMM
    # header fails its version gate.
    target_link_libraries(uform_loaders PRIVATE uform_cudf)
    target_include_directories(
        uform_loaders SYSTEM BEFORE
        PRIVATE "${uform_cudf_prefix_}/libcudf/include/rapids" "${uform_cudf_prefix_}/libcudf/include"
                "${uform_cudf_prefix_}/librmm/include" "${uform_cudf_prefix_}/libkvikio/include"
                "${uform_cudf_prefix_}/rapids_logger/include"
    )

    # Image decode: nvJPEG + NPP on the GPU, the three CPU codecs for the fallback.
    target_link_libraries(uform_loaders PRIVATE CUDA::nvjpeg CUDA::nppig CUDA::nppc PNG::PNG "${UFORM_WEBP_LIBRARY}")

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

    # The host-side media demux and decode run on ForkUnion pools, public because the loader headers take `pool_view_t`.
    target_link_libraries(uform_loaders PUBLIC forkunion::header forkunion::static)

    # `train/datasets.cu` → `uform::datasets`: the mixtures and the batch source over the loaders, kept out of
    # `uform::train`, which the C library links without the loaders or cuDF.
    add_library(uform_datasets STATIC train/datasets.cu)
    add_library(uform::datasets ALIAS uform_datasets)
    target_link_libraries(uform_datasets PUBLIC uform::train uform::loaders)
endif ()

# Objects never travel through another object library, so every consumer of `uform_host` names the variants too.
return(PROPAGATE uform_encoder_variants_)
