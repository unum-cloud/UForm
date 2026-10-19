# `uform_bench_core`, the core cell's GPU benchmarks, at the build root beside the suites. It prints the version and the
# capabilities through the C interface, whose host objects the training archive carries.
set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_BUILD_RPATH "$ORIGIN")
add_executable(uform_bench_core core.cu)
target_link_libraries(uform_bench_core PRIVATE uform::train fmt::fmt)
target_include_directories(uform_bench_core PRIVATE "${PROJECT_SOURCE_DIR}/c")
