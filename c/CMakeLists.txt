# The C interface lives in `c/uform/*.h` and its implementation in `src/`, so this directory holds only the headers and
# the test that proves the two agree. Both library flavors take the objects `src/` compiled rather than compiling a
# boundary of their own.
#
# The shared flavor links the object libraries and inherits their dependencies; the archive takes the objects
# themselves, since an archive links nothing - and linking a library instead would drop every symbol nothing inside
# referenced, which is the entire C interface. For the same reason the archive carries its dependencies' objects, so
# no consumer hunts yyjson, NumKong or StringZilla archives out of the dependency tree to link one of ours.
add_library(uform_shared SHARED)
add_library(uform::shared ALIAS uform_shared)
target_link_libraries(uform_shared PRIVATE uform_host)
# The archives it links were compiled with default visibility, so only the linker can keep their symbols in.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(uform_shared PRIVATE "LINKER:--exclude-libs,ALL")
endif ()
add_library(
    uform_static STATIC $<TARGET_OBJECTS:uform_host> $<TARGET_OBJECTS:yyjson> $<TARGET_OBJECTS:numkong_static>
                        $<TARGET_OBJECTS:stringzilla_static>
)
add_library(uform::static ALIAS uform_static)
target_include_directories(uform_shared PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(uform_static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
# The system codecs the host decoders call stay shared libraries, which an archive can only name to its consumers.
if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm")
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
# archive therefore takes one by one.
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
# Each GPU vendor's inference kernels join both flavors, and the archive names that vendor's runtime to its consumers.
if (TARGET uform_nvidia_inference)
    target_link_libraries(uform_shared PRIVATE uform_nvidia_inference)
    target_sources(uform_static PRIVATE $<TARGET_OBJECTS:uform_nvidia_inference>)
    target_link_libraries(uform_static INTERFACE CUDA::cudart CUDA::cuda_driver)
endif ()
# The training verbs join both flavors with the NVIDIA build, and the archive names the training stack to its consumers.
if (TARGET uform_nvidia_train)
    target_link_libraries(uform_shared PRIVATE uform_nvidia_train uform::train)
    target_sources(uform_static PRIVATE $<TARGET_OBJECTS:uform_nvidia_train>)
    target_link_libraries(uform_static INTERFACE uform::train)
endif ()
if (TARGET uform_amd_inference)
    target_link_libraries(uform_shared PRIVATE uform_amd_inference)
    target_sources(uform_static PRIVATE $<TARGET_OBJECTS:uform_amd_inference>)
    target_link_libraries(uform_static INTERFACE hip::host)
endif ()
set_target_properties(
    uform_shared uform_static
    PROPERTIES LINKER_LANGUAGE CXX
               C_VISIBILITY_PRESET hidden
               CXX_VISIBILITY_PRESET hidden
)

# The test is C, so it reaches the C++ runtime and the host's thread library through the archive's link language.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test.c")
    add_executable(uform_test_c test.c)
    target_link_libraries(uform_test_c PRIVATE uform_static)
    set_target_properties(uform_test_c PROPERTIES LINKER_LANGUAGE CXX)
    # The engine loads only a real checkpoint, so the test takes the seeded one whenever the suites are built.
    if (TARGET uform_tiny_checkpoint)
        add_test(NAME uform_test_c COMMAND uform_test_c "${PROJECT_BINARY_DIR}/tiny.safetensors")
        set_tests_properties(uform_test_c PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint)
    else ()
        add_test(NAME uform_test_c COMMAND uform_test_c)
    endif ()
endif ()

# The same checkpoint on an NVIDIA GPU against the host, which only a build with the NVIDIA kernels and the suites runs.
if (TARGET uform_nvidia_inference AND TARGET uform_tiny_checkpoint)
    add_executable(uform_test_vram "${PROJECT_SOURCE_DIR}/test/vram.c")
    target_link_libraries(uform_test_vram PRIVATE uform_static CUDA::cuda_driver)
    set_target_properties(uform_test_vram PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME uform_test_vram COMMAND uform_test_vram "${PROJECT_BINARY_DIR}/tiny.safetensors")
    set_tests_properties(uform_test_vram PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint)
endif ()
