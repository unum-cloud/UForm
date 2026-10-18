# The synchronous stand-in for the C interface: a separate library defining the same symbols a different way, so any
# consumer - a binding suite, a downstream SDK, a bisect - runs in seconds against the ABI instead of waiting on a build
# of the engine. It needs the C headers and the C++ standard library and nothing else - no NumKong, no ForkUnion, no
# CUDA - so it sets by hand the few properties consumers depend on. The one runtime it resolves is the platform's
# threads, since a caller may wait on another thread's deferred call.
#
# The file names match what `c/` produces - `libuform_shared` and `libuform_static` - and land in a directory of their
# own, so a suite chooses an implementation with one `-L` and the two libraries can never shadow one another.
set(UFORM_STUB_SOURCES types.cpp model.cpp encode.cpp train.cpp)

find_package(Threads REQUIRED)

add_library(uform_shared_stub SHARED ${UFORM_STUB_SOURCES})
add_library(uform_static_stub STATIC ${UFORM_STUB_SOURCES})
add_library(${PROJECT_NAME}::shared_stub ALIAS uform_shared_stub)
add_library(${PROJECT_NAME}::static_stub ALIAS uform_static_stub)

foreach (TARGET_NAME uform_shared_stub uform_static_stub)
    target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)
    target_compile_definitions(${TARGET_NAME} PRIVATE UFORM_EXPORTING)
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES PREFIX "lib"
                   POSITION_INDEPENDENT_CODE ON
                   CXX_VISIBILITY_PRESET hidden
                   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub"
                   ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub"
                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub"
    )
    # A root may pin every configuration's directory too, and those outrank the plain ones above.
    foreach (STUB_CONFIGURATION DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(
            ${TARGET_NAME}
            PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${STUB_CONFIGURATION} "${CMAKE_BINARY_DIR}/stub"
                       ARCHIVE_OUTPUT_DIRECTORY_${STUB_CONFIGURATION} "${CMAKE_BINARY_DIR}/stub"
                       RUNTIME_OUTPUT_DIRECTORY_${STUB_CONFIGURATION} "${CMAKE_BINARY_DIR}/stub"
        )
    endforeach ()
endforeach ()

set_target_properties(uform_shared_stub PROPERTIES OUTPUT_NAME "uform_shared")
set_target_properties(uform_static_stub PROPERTIES OUTPUT_NAME "uform_static")

# The C suite the engine's library runs too, linked against the static stand-in. It is written in C, but the library
# it links is C++, so the C++ driver links it and brings the standard library along.
add_executable(uform_test_c_stub ${PROJECT_SOURCE_DIR}/c/test.c)
target_link_libraries(uform_test_c_stub PRIVATE uform_static_stub)
set_target_properties(
    uform_test_c_stub
    PROPERTIES C_STANDARD 99
               C_STANDARD_REQUIRED ON
               LINKER_LANGUAGE CXX
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub"
)
add_test(NAME uform_test_c_stub COMMAND uform_test_c_stub)
