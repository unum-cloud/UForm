# bundle_runtime.cmake — build a self-contained runtime tree for heterogeneous clusters.
#
# Invoked at build time by the `uform_bundle` custom target as
# `cmake -DUFORM_BUNDLE_DIR=<dir> -P bundle_runtime.cmake -- <executable>...`, where the destination tree takes every
# executable beside its shared-library closure.
#
# Every executable is copied together with its resolved shared-library closure — everything except the
# glibc core, which must come from the running node's own loader, and the NVIDIA driver's user-mode
# libraries, which must match the node's kernel module. Paired with the executables' `$ORIGIN`
# build RPATH, the tree runs from a shared filesystem on any node regardless of what that node has
# installed natively. Unresolved dependencies fail the bundle loudly, naming each library.

set(executables_)
set(listing_ OFF)
math(EXPR last_argument_ "${CMAKE_ARGC} - 1")
foreach (argument_ RANGE ${last_argument_})
    if (listing_)
        list(APPEND executables_ "${CMAKE_ARGV${argument_}}")
    elseif (CMAKE_ARGV${argument_} STREQUAL "--")
        set(listing_ ON)
    endif ()
endforeach ()
if (NOT executables_ OR NOT DEFINED UFORM_BUNDLE_DIR)
    message(FATAL_ERROR "bundle_runtime.cmake needs -DUFORM_BUNDLE_DIR=<dir> and the executables after `--`")
endif ()

# A tree reused across builds would keep libraries the closure no longer names.
file(REMOVE_RECURSE "${UFORM_BUNDLE_DIR}")
file(MAKE_DIRECTORY "${UFORM_BUNDLE_DIR}")

file(
    GET_RUNTIME_DEPENDENCIES
    EXECUTABLES
    ${executables_}
    RESOLVED_DEPENDENCIES_VAR
    resolved_
    UNRESOLVED_DEPENDENCIES_VAR
    unresolved_
    # The glibc core travels with the node's own loader — bundling it breaks more than it fixes.
    # A driver library from the build machine fails `cuInit` against any other kernel module.
    PRE_EXCLUDE_REGEXES
    "^ld-linux"
    "^libc\\.so"
    "^libm\\.so"
    "^libmvec\\.so"
    "^libdl\\.so"
    "^libpthread\\.so"
    "^librt\\.so"
    "^libresolv\\.so"
    "^libgcc_s\\.so"
    "^libcuda\\.so"
    "^libnvcuvid\\.so"
    "^libnvidia-"
)

if (unresolved_)
    message(FATAL_ERROR "unresolved runtime dependencies: ${unresolved_}")
endif ()

foreach (library_ IN LISTS resolved_)
    file(
        COPY "${library_}"
        DESTINATION "${UFORM_BUNDLE_DIR}"
        FOLLOW_SYMLINK_CHAIN
    )
endforeach ()
foreach (executable_ IN LISTS executables_)
    file(
        COPY "${executable_}"
        DESTINATION "${UFORM_BUNDLE_DIR}"
        FOLLOW_SYMLINK_CHAIN
    )
endforeach ()

list(LENGTH resolved_ library_count_)
list(LENGTH executables_ executable_count_)
message(STATUS "bundled ${executable_count_} executable(s) + ${library_count_} libraries into ${UFORM_BUNDLE_DIR}")
