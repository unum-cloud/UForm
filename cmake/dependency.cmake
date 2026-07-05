# Resolve one source dependency: a local checkout wins, otherwise a pinned FetchContent download. A single
# call replaces the near-identical fetch-or-local blocks that would otherwise repeat per dependency.
#
# Usage:  uform_dependency(NAME <name> MODE SUBDIR|HEADER REPO <url> TAG <tag>
#                          [ROOT_VAR <VAR>] [INCLUDE_VAR <VAR>])
#
#   MODE SUBDIR   `add_subdirectory` the checkout, or `FetchContent` it; targets then link by NAME.
#   MODE HEADER   populate a header-only dep without configuring it (a sentinel `SOURCE_SUBDIR` replaces the
#                 deprecated single-arg `FetchContent_Populate`) and return its include dir via INCLUDE_VAR.
#   ROOT_VAR      name of a `UFORM_*_ROOT` cache PATH; when set, its checkout overrides the download. Declared
#                 here so callers skip the `set(... CACHE ...)` boilerplate; a `FORCE` set in `cmake/local.cmake`
#                 still wins, since this plain cache set defers to an existing value.
#   INCLUDE_VAR   (HEADER only) receives <source>/include in the caller's scope.
#
# Naming: `UFORM_*` are public cache variables, `_uform_*` are internal (locals and parsed arguments alike).

function (uform_dependency)
    cmake_parse_arguments(_uform "" "NAME;MODE;REPO;TAG;ROOT_VAR;INCLUDE_VAR" "" ${ARGN})

    set(_uform_root "")
    if (_uform_ROOT_VAR)
        set(${_uform_ROOT_VAR}
            ""
            CACHE PATH "Local ${_uform_NAME} checkout overriding the FetchContent fallback"
        )
        set(_uform_root "${${_uform_ROOT_VAR}}")
    endif ()

    if (_uform_MODE STREQUAL "SUBDIR")
        if (_uform_root)
            message(STATUS "uform: ${_uform_NAME} from ${_uform_root}")
            add_subdirectory("${_uform_root}" "${CMAKE_CURRENT_BINARY_DIR}/${_uform_NAME}" EXCLUDE_FROM_ALL)
        else ()
            FetchContent_Declare(
                ${_uform_NAME}
                GIT_REPOSITORY ${_uform_REPO}
                GIT_TAG ${_uform_TAG}
            )
            FetchContent_MakeAvailable(${_uform_NAME})
        endif ()
    elseif (_uform_MODE STREQUAL "HEADER")
        if (_uform_root)
            set(${_uform_INCLUDE_VAR}
                "${_uform_root}/include"
                PARENT_SCOPE
            )
        else ()
            # `SOURCE_SUBDIR` points at a directory with no `CMakeLists.txt`, so `FetchContent_MakeAvailable`
            # populates the sources but skips `add_subdirectory` — the replacement for `FetchContent_Populate`.
            FetchContent_Declare(
                ${_uform_NAME}
                GIT_REPOSITORY ${_uform_REPO}
                GIT_TAG ${_uform_TAG}
                SOURCE_SUBDIR _uform_no_configure
            )
            FetchContent_MakeAvailable(${_uform_NAME})
            set(${_uform_INCLUDE_VAR}
                "${${_uform_NAME}_SOURCE_DIR}/include"
                PARENT_SCOPE
            )
        endif ()
    else ()
        message(FATAL_ERROR "uform_dependency(${_uform_NAME}): MODE must be SUBDIR or HEADER, got '${_uform_MODE}'")
    endif ()
endfunction ()
