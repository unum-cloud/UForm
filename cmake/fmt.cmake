# The pin for what only the suites, benchmarks and tools link: {fmt} for their output, as libstdc++ doesn't ship
# `std::print` yet on our toolchain. `FIND_PACKAGE_ARGS` lets a system-provided {fmt} satisfy the dependency without
# touching the network, but only from the pinned version up, since older copies predate `fmt::println`.
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.2.0
    SYSTEM FIND_PACKAGE_ARGS 12.2.0
)
FetchContent_MakeAvailable(fmt)
