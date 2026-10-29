# The C interface lives in `c/uform/*.h` and its implementation in `src/`, so this directory holds only the headers and
# the test that proves the two agree. Both library flavors take the objects `src/` compiled rather than compiling a
# boundary of their own.
#
# The shared flavor links the object libraries and inherits their dependencies; the archive takes the objects
# themselves, since an archive links nothing - and linking a library instead would drop every symbol nothing inside
# referenced, which is the entire C interface. For the same reason the archive carries its dependencies' objects, so
# no consumer hunts yyjson, NumKong or StringZilla archives out of the dependency tree to link one of ours.
include(GNUInstallDirs)
# Without `UFORM_BUILD_SHARED`, the libraries build only for the targets that link them
if (NOT UFORM_BUILD_SHARED)
    set(uform_library_exclusion_ EXCLUDE_FROM_ALL)
endif ()
add_library(uform_shared SHARED ${uform_library_exclusion_})
add_library(uform::shared ALIAS uform_shared)
# The encoder's variants follow `uform_host`, so the baseline's copy of an inline function they share comes first.
target_link_libraries(uform_shared PRIVATE uform_host ${uform_encoder_variants_})
# The archives it links, NumKong's, ForkUnion's and yyjson's among them, were compiled with default visibility, and the
# standard library headers declare their templates visible, so only the linker can keep those symbols in.
if (APPLE)
    target_link_options(uform_shared PRIVATE "LINKER:-exported_symbol,_uform_*")
elseif (CMAKE_SYSTEM_NAME MATCHES "^(Linux|Android|FreeBSD)$")
    file(WRITE "${PROJECT_BINARY_DIR}/uform_shared.map" "{\n  global: uform_*;\n  local: *;\n};\n")
    target_link_options(uform_shared PRIVATE "LINKER:--version-script=${PROJECT_BINARY_DIR}/uform_shared.map")
endif ()
# `SOVERSION` carries the major alone, the only break the C ABI admits.
set_target_properties(uform_shared PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
set(uform_variant_objects_)
foreach (library_ IN LISTS uform_encoder_variants_)
    list(APPEND uform_variant_objects_ $<TARGET_OBJECTS:${library_}>)
endforeach ()
add_library(
    uform_static STATIC
    ${uform_library_exclusion_}
    $<TARGET_OBJECTS:uform_host>
    ${uform_variant_objects_}
    $<TARGET_OBJECTS:yyjson>
    $<TARGET_OBJECTS:numkong_static>
    $<TARGET_OBJECTS:stringzilla_static>
    $<TARGET_OBJECTS:forkunion_static>
)
add_library(uform::static ALIAS uform_static)
add_library(uform_header INTERFACE)
add_library(uform::header ALIAS uform_header)
set_target_properties(uform_static PROPERTIES EXPORT_NAME static)
set_target_properties(uform_shared PROPERTIES EXPORT_NAME shared)
set_target_properties(uform_header PROPERTIES EXPORT_NAME header)
target_include_directories(
    uform_header INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(uform_shared PUBLIC uform_header)
target_link_libraries(uform_static PUBLIC uform_header)
# The system codecs the host decoders call stay shared libraries, which an archive can only name to its consumers.
if (UFORM_WITH_CODECS)
    target_link_libraries(
        uform_static
        INTERFACE "${UFORM_TURBOJPEG_LIBRARY}"
                  PNG::PNG
                  "${UFORM_WEBP_LIBRARY}"
                  "${UFORM_AVFORMAT_LIBRARY}"
                  "${UFORM_AVCODEC_LIBRARY}"
                  "${UFORM_AVUTIL_LIBRARY}"
                  "${UFORM_SWRESAMPLE_LIBRARY}"
                  "${UFORM_SWSCALE_LIBRARY}"
    )
endif ()
# NumKong compiles each GPU generation into an object library of its own beside its archive's objects, which the
# archive therefore takes one by one, naming to its consumers the vendor runtime those units launch through.
foreach (
    numkong_unit_ IN
    ITEMS cuda
          ampere
          ada
          hopper
          blackwell
          blackwellrtx
          blackwellultra
          rocm
          cdna3
          cdna4
          cdna5
)
    if (TARGET numkong_${numkong_unit_}_)
        target_sources(uform_static PRIVATE $<TARGET_OBJECTS:numkong_${numkong_unit_}_>)
    endif ()
endforeach ()
if (TARGET numkong_cuda_)
    target_link_libraries(uform_static INTERFACE CUDA::cudart)
endif ()
if (TARGET numkong_rocm_)
    target_link_libraries(uform_static INTERFACE hip::host)
endif ()
# Each GPU vendor's variants join both flavors above and open that vendor's driver at first use, linking none.
if (TARGET uform_encoder_vram_nvidia_variants OR TARGET uform_encoder_vram_amd_variants)
    target_link_libraries(uform_static INTERFACE ${CMAKE_DL_LIBS})
endif ()
# The training verbs join both flavors with UFORM_BUILD_TRAIN, and the archive names the training stack to its
# consumers.
if (TARGET uform_nvidia_train)
    target_link_libraries(uform_shared PRIVATE uform_nvidia_train uform::train)
    target_sources(uform_static PRIVATE $<TARGET_OBJECTS:uform_nvidia_train>)
    target_link_libraries(uform_static INTERFACE $<BUILD_INTERFACE:uform::train>)
endif ()
set_target_properties(
    uform_shared uform_static
    PROPERTIES LINKER_LANGUAGE CXX
               C_VISIBILITY_PRESET hidden
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
)

# The test is C, so it reaches the C++ runtime and the host's thread library through the archive's link language.
if (UFORM_BUILD_TEST)
    add_executable(uform_test_c test.c)
    target_link_libraries(uform_test_c PRIVATE uform_static forkunion::header)
    set_target_properties(uform_test_c PROPERTIES LINKER_LANGUAGE CXX)
    if (EMSCRIPTEN)
        target_link_options(uform_test_c PRIVATE "-sNODERAWFS")
    endif ()
    # The engine loads only a real checkpoint, so the test takes the seeded one whenever the suites are built.
    if (TARGET uform_tiny_checkpoint)
        add_test(NAME uform_test_c COMMAND uform_test_c "${PROJECT_BINARY_DIR}/tiny.safetensors")
        set_tests_properties(uform_test_c PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint)
    else ()
        add_test(NAME uform_test_c COMMAND uform_test_c)
    endif ()
endif ()

# The same checkpoint on an NVIDIA GPU against the host, which only a build with the NVIDIA kernels and the suites runs.
if (TARGET uform_encoder_vram_nvidia_variants AND TARGET uform_tiny_checkpoint)
    add_executable(uform_test_vram "${PROJECT_SOURCE_DIR}/test/vram.c")
    target_link_libraries(uform_test_vram PRIVATE uform_static CUDA::cuda_driver)
    set_target_properties(uform_test_vram PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME uform_test_vram COMMAND uform_test_vram "${PROJECT_BINARY_DIR}/tiny.safetensors")
    set_tests_properties(uform_test_vram PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint)
endif ()
