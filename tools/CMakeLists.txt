# `uform_base`, the trainer binary, at the build root beside the suites and bundled with them by `uform_bundle`.
set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_BUILD_RPATH "$ORIGIN")
add_executable(uform_base base.cu)
target_link_libraries(uform_base PRIVATE uform::datasets fmt::fmt)
add_executable(uform_eval eval.cpp)
target_link_libraries(uform_eval PRIVATE uform::static smashtable::header yyjson fmt::fmt)
set(uform_bundled_files_
    ${uform_bundled_files_} $<TARGET_FILE:uform_base> $<TARGET_FILE:uform_eval>
    PARENT_SCOPE
)
