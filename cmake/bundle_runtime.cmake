# bundle_runtime.cmake — build a self-contained runtime tree for heterogeneous clusters.
#
# Invoked at build time by the `uform_bundle` custom target (`cmake -P` script mode) with:
#
#   BUNDLE_EXECUTABLES — ;-list of built executables to bundle
#   BUNDLE_DIR         — destination tree; executables land beside their shared-library closure
#
# Every executable is copied together with its resolved shared-library closure — everything except the
# glibc core, which must come from the running node's own loader. Paired with the executables' `$ORIGIN`
# build RPATH, the tree runs from a shared filesystem on any node regardless of what that node has
# installed natively (the gap that once meant "libcudnn.so.9: cannot open shared object file" on the one
# node lacking the apt package). Unresolved dependencies fail the bundle loudly, naming each library.

if (NOT DEFINED BUNDLE_EXECUTABLES OR NOT DEFINED BUNDLE_DIR)
    message(FATAL_ERROR "bundle_runtime.cmake needs -DBUNDLE_EXECUTABLES=<|-list> and -DBUNDLE_DIR=<dir>")
endif ()

# The executable list arrives |-separated — a raw `;` cannot survive the custom-command argument path.
string(REPLACE "|" ";" BUNDLE_EXECUTABLES "${BUNDLE_EXECUTABLES}")

file(MAKE_DIRECTORY "${BUNDLE_DIR}")

file(
    GET_RUNTIME_DEPENDENCIES
    EXECUTABLES
    ${BUNDLE_EXECUTABLES}
    RESOLVED_DEPENDENCIES_VAR
    bundle_resolved
    UNRESOLVED_DEPENDENCIES_VAR
    bundle_unresolved
    # The glibc core travels with the node's own loader — bundling it breaks more than it fixes.
    PRE_EXCLUDE_REGEXES
    "^ld-linux"
    "^libc\\.so"
    "^libm\\.so"
    "^libdl\\.so"
    "^libpthread\\.so"
    "^librt\\.so"
    "^libresolv\\.so"
    "^libgcc_s\\.so"
)

if (bundle_unresolved)
    message(FATAL_ERROR "unresolved runtime dependencies: ${bundle_unresolved}")
endif ()

foreach (bundle_library IN LISTS bundle_resolved)
    file(
        COPY "${bundle_library}"
        DESTINATION "${BUNDLE_DIR}"
        FOLLOW_SYMLINK_CHAIN
    )
endforeach ()
foreach (bundle_executable IN LISTS BUNDLE_EXECUTABLES)
    file(
        COPY "${bundle_executable}"
        DESTINATION "${BUNDLE_DIR}"
        FOLLOW_SYMLINK_CHAIN
    )
endforeach ()

list(LENGTH bundle_resolved bundle_library_count)
list(LENGTH BUNDLE_EXECUTABLES bundle_executable_count)
message(
    STATUS "bundled ${bundle_executable_count} executable(s) + ${bundle_library_count} librarie(s) into ${BUNDLE_DIR}"
)
