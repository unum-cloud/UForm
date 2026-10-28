# The benchmarks, at the build root beside the suites, each printing the version and the capabilities through the C
# interface. `uform_bench_encoder` reaches the encoder through the shared C library alone, so it builds on every tree;
# `uform_bench_core` times the core cell's GPU kernels, whose host objects the training archive carries.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_BUILD_RPATH "$ORIGIN")
add_executable(uform_bench_encoder encoder.cpp)
target_link_libraries(uform_bench_encoder PRIVATE uform::shared fmt::fmt)
target_compile_features(uform_bench_encoder PRIVATE cxx_std_23)
if (UFORM_BUILD_TRAIN)
    set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)
    add_executable(uform_bench_core core.cu)
    target_link_libraries(uform_bench_core PRIVATE uform::train fmt::fmt)
    target_include_directories(uform_bench_core PRIVATE "${PROJECT_SOURCE_DIR}/c")
endif ()
