# The host tower's suites: `uform_tiny_checkpoint` writes the seeded checkpoint once per `ctest` run as a fixture, which
# the parity test here and the C tests in `c/` load.
add_executable(uform_tiny_checkpoint tiny_checkpoint.cpp)
add_test(NAME uform_tiny_checkpoint COMMAND uform_tiny_checkpoint "${PROJECT_BINARY_DIR}/tiny.safetensors")
set_tests_properties(uform_tiny_checkpoint PROPERTIES FIXTURES_SETUP uform_tiny_checkpoint)
add_executable(uform_test_inference inference.cpp)
target_link_libraries(uform_test_inference PRIVATE uform_host)
if (UFORM_BUILD_CUDA)
    target_link_libraries(uform_test_inference PRIVATE uform_nvidia_inference)
endif ()
if (UFORM_BUILD_ROCM)
    target_link_libraries(uform_test_inference PRIVATE uform_amd_inference)
endif ()
add_test(NAME uform_test_inference COMMAND uform_test_inference "${PROJECT_BINARY_DIR}/tiny.safetensors")
set_tests_properties(uform_test_inference PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint)

# The conformance binaries, one per cell, which `uform_bundle` ships with their runtime closure. Each sits at the build
# root, where the sbatch scripts and the launch configurations look for it, and resolves shared libraries beside itself
# first, so the tree runs on nodes whose native installs diverge from the build node's.
if (UFORM_BUILD_CUDA)
    set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    add_executable(uform_test_mesh mesh.cpp)
    target_link_libraries(uform_test_mesh PRIVATE uform::core)
    add_executable(uform_test_mixer mixer.cu)
    target_link_libraries(uform_test_mixer PRIVATE uform::core)
    add_executable(uform_test_contrastive contrastive.cu)
    target_link_libraries(uform_test_contrastive PRIVATE uform::core)
    add_executable(uform_test_loaders loaders.cu)
    target_link_libraries(uform_test_loaders PRIVATE uform::loaders)
    add_executable(uform_test_base base.cu)
    target_link_libraries(uform_test_base PRIVATE uform::train)
    # The preflight probe every multi-node sbatch runs first: links the full training stack so merely starting on a
    # node proves the runtime closure resolves there, then checks GPUDirect registration per GPU-NIC pair and the
    # rendezvous directory — one named line per environment gap, before the real job spends a minute discovering it
    # mid-suite.
    add_executable(uform_test_env env.cpp)
    target_link_libraries(uform_test_env PRIVATE uform::train)
    foreach (test_name IN ITEMS mesh mixer contrastive loaders base env)
        add_test(NAME uform_test_${test_name} COMMAND uform_test_${test_name})
        list(APPEND uform_bundled_files_ $<TARGET_FILE:uform_test_${test_name}>)
    endforeach ()
    set(uform_bundled_files_
        ${uform_bundled_files_}
        PARENT_SCOPE
    )
endif ()
