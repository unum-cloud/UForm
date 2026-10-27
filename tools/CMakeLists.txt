# `uform_base`, the trainer binary, at the build root beside the suites and bundled with them by `uform_bundle`.
set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_BUILD_RPATH "$ORIGIN")
add_executable(uform_base base.cu)
target_link_libraries(uform_base PRIVATE uform::datasets fmt::fmt)
add_executable(uform_eval eval.cpp)
target_link_libraries(
    uform_eval
    PRIVATE uform::static
            uform::train
            numkong::static
            forkunion::header
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
