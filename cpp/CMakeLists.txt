# The header-only C++ SDK over the C interface: `cpp/uform/*.hpp` include only the C headers `uform::abi` carries
# and the standard library, which the interface target enforces, so a consumer linking `uform::cpp` gets the archive
# and sees no engine header.
add_library(uform_cpp INTERFACE)
add_library(uform::cpp ALIAS uform_cpp)
set_target_properties(uform_cpp PROPERTIES EXPORT_NAME cpp)
target_include_directories(
    uform_cpp INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(uform_cpp INTERFACE uform::abi uform::static)
target_compile_features(uform_cpp INTERFACE cxx_std_20)

# The smoke test against the engine loads the checkpoint the suites seed, as the C test does; the stand-in's loads any
# bytes.
if (UFORM_BUILD_TEST)
    add_executable(uform_test_cpp test.cpp)
    target_link_libraries(uform_test_cpp PRIVATE uform::cpp uform_static fmt::fmt uform_warnings uform_sanitizers)
    add_test(NAME uform_test_cpp COMMAND uform_test_cpp "${PROJECT_BINARY_DIR}/tiny.safetensors")
    set_tests_properties(uform_test_cpp PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint)
endif ()

# The same test against the stand-in, which builds without the engine, so an SDK header reaching past `c/` fails to
# compile here. It takes the SDK's headers rather than `uform::cpp`, which would link the engine's archive.
if (UFORM_BUILD_TEST AND UFORM_BUILD_STUB)
    add_executable(uform_test_cpp_stub test.cpp)
    target_include_directories(uform_test_cpp_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(uform_test_cpp_stub PRIVATE cxx_std_20)
    target_link_libraries(
        uform_test_cpp_stub PRIVATE uform::abi uform_static_stub fmt::fmt uform_warnings uform_sanitizers
    )
    set_target_properties(uform_test_cpp_stub PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub")
    add_test(NAME uform_test_cpp_stub COMMAND uform_test_cpp_stub)
endif ()
