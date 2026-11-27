# `uform_base`, the trainer, and `uform_eval` and `uform_grow`, which score and grow its checkpoints, at the build root
# beside the suites and bundled with them by `uform_bundle`. `uform_base` reads its enumerators through the C names.
set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_BUILD_RPATH "$ORIGIN")
add_executable(uform_base base.cu)
target_link_libraries(uform_base PRIVATE uform::static uform::datasets fmt::fmt)
add_executable(uform_eval eval.cpp)
target_link_libraries(
    uform_eval
    PRIVATE uform::static
            uform::train
            numkong::types
            numkong::static
            forkunion::cpp
            smashtable::header
            yyjson
            fmt::fmt
)
add_executable(uform_grow grow.cpp)
target_link_libraries(uform_grow PRIVATE uform::static uform::train fmt::fmt)
set(uform_bundled_files_
    ${uform_bundled_files_} $<TARGET_FILE:uform_base> $<TARGET_FILE:uform_eval> $<TARGET_FILE:uform_grow>
    PARENT_SCOPE
)
# Each binary's flags, variables and stop, through the emulator a cross build runs it in, labelled `cuda` as each
# loads the CUDA driver; the calibration stop runs against the tiny checkpoint the suites write.
if (UFORM_BUILD_TEST)
    add_executable(uform_test_tools ${PROJECT_SOURCE_DIR}/test/tools.cpp)
    target_link_libraries(uform_test_tools PRIVATE fmt::fmt)
    target_compile_features(uform_test_tools PRIVATE cxx_std_23)
    foreach (tool_ IN ITEMS base eval grow)
        add_test(NAME uform_test_tools_${tool_} COMMAND uform_test_tools -- ${CMAKE_CROSSCOMPILING_EMULATOR}
                                                        $<TARGET_FILE:uform_${tool_}>
        )
        set_tests_properties(uform_test_tools_${tool_} PROPERTIES LABELS cuda)
    endforeach ()
    set_tests_properties(
        uform_test_tools_eval PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint ENVIRONMENT
                                         "UFORM_TEST_CHECKPOINT=${PROJECT_BINARY_DIR}/tiny.safetensors"
    )
endif ()
