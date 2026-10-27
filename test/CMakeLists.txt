# The host tower's suites: `uform_tiny_checkpoint` writes the seeded checkpoint once per `ctest` run as a fixture, which
# the parity test here and the C tests in `c/` load. Every suite prints through {fmt} and reads the version and the
# capabilities through the C interface. Labels sort the suites by what they need: `host` runs anywhere, `cuda` wants a
# GPU, and `cluster` names the preflight every multi-node sbatch runs first.
add_executable(uform_tiny_checkpoint tiny_checkpoint.cpp)
target_link_libraries(uform_tiny_checkpoint PRIVATE fmt::fmt)
target_compile_features(uform_tiny_checkpoint PRIVATE cxx_std_23)
add_test(NAME uform_tiny_checkpoint COMMAND uform_tiny_checkpoint "${PROJECT_BINARY_DIR}/tiny.safetensors")
set_tests_properties(uform_tiny_checkpoint PROPERTIES FIXTURES_SETUP uform_tiny_checkpoint LABELS host)
add_executable(uform_test_inference inference.cpp)
target_link_libraries(uform_test_inference PRIVATE uform_host ${uform_encoder_variants_} fmt::fmt)
add_test(NAME uform_test_inference COMMAND uform_test_inference "${PROJECT_BINARY_DIR}/tiny.safetensors")
set_tests_properties(uform_test_inference PROPERTIES FIXTURES_REQUIRED uform_tiny_checkpoint LABELS host)
# Under Emscripten the suites run in Node, reading and writing the build tree's real files.
if (EMSCRIPTEN)
    target_link_options(uform_tiny_checkpoint PRIVATE "-sNODERAWFS")
    target_link_options(uform_test_inference PRIVATE "-sNODERAWFS")
endif ()

# The conformance binaries, one per cell, which `uform_bundle` ships with their runtime closure. Each sits at the build
# root, where the sbatch scripts and the launch configurations look for it, and resolves shared libraries beside itself
# first, so the tree runs on nodes whose native installs diverge from the build node's. The training and loader
# archives carry the host objects behind the C interface's probes.
if (UFORM_BUILD_CUDA)
    set(CMAKE_JOB_POOL_COMPILE uform_cuda_units)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    add_executable(uform_test_mesh mesh.cpp)
    target_link_libraries(uform_test_mesh PRIVATE uform::train)
    add_executable(uform_test_mixer mixer.cu)
    target_link_libraries(uform_test_mixer PRIVATE uform::train)
    add_executable(uform_test_contrastive contrastive.cu)
    target_link_libraries(uform_test_contrastive PRIVATE uform::train)
    add_executable(uform_test_loaders loaders.cu)
    target_link_libraries(uform_test_loaders PRIVATE uform::datasets)
    add_executable(uform_test_base base.cu)
    target_link_libraries(uform_test_base PRIVATE uform::train)
    # The preflight probe every multi-node sbatch runs first: links the full training stack so merely starting on a
    # node proves the runtime closure resolves there, then checks GPUDirect registration per GPU-NIC pair and the
    # rendezvous directory — one named line per environment gap, before the real job spends a minute discovering it
    # mid-suite.
    add_executable(uform_test_env env.cpp)
    target_link_libraries(uform_test_env PRIVATE uform::train)
    foreach (test_name_ IN ITEMS mesh mixer contrastive loaders base env)
        target_link_libraries(uform_test_${test_name_} PRIVATE fmt::fmt)
        target_include_directories(uform_test_${test_name_} PRIVATE "${PROJECT_SOURCE_DIR}/c")
        add_test(NAME uform_test_${test_name_} COMMAND uform_test_${test_name_})
        set_tests_properties(uform_test_${test_name_} PROPERTIES LABELS cuda)
        list(APPEND uform_bundled_files_ $<TARGET_FILE:uform_test_${test_name_}>)
    endforeach ()
    set_tests_properties(uform_test_env PROPERTIES LABELS "cuda;cluster")
    # The world relations, every process's slice against the single-process construction, on the host alone.
    add_executable(uform_test_relations relations.cpp)
    target_link_libraries(uform_test_relations PRIVATE uform::train fmt::fmt)
    target_include_directories(uform_test_relations PRIVATE "${PROJECT_SOURCE_DIR}/c")
    add_test(NAME uform_test_relations COMMAND uform_test_relations)
    set_tests_properties(uform_test_relations PROPERTIES LABELS host)
    set(uform_bundled_files_
        ${uform_bundled_files_}
        PARENT_SCOPE
    )
endif ()

# The backpropagation suite: finite differences against every analytic gradient of the masked replica, the masking
# and initialization invariants, and Adafactor against a host replay.
if (UFORM_BUILD_CUDA)
    add_executable(uform_test_gradients gradients.cu)
    target_link_libraries(uform_test_gradients PRIVATE uform::train fmt::fmt)
    target_include_directories(uform_test_gradients PRIVATE "${PROJECT_SOURCE_DIR}/c")
    add_test(NAME uform_test_gradients COMMAND uform_test_gradients)
    set_tests_properties(uform_test_gradients PROPERTIES LABELS cuda)
endif ()
