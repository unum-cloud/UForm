# The sibling helper, the switches forwarded to the siblings, and the siblings themselves.

include(FetchContent)

# A sibling builds in place from its submodule when a recursive clone checked one out, or else from the commit that
# submodule pins, fetched, so a plain clone or a `FetchContent` of UForm still configures.
# The caller names the sibling's `FETCHCONTENT_SOURCE_DIR_<NAME>`, so nothing derives it.
# A non-empty one points the sibling at a local checkout, as in a workspace keeping the siblings side by side, and
# outranks the submodule.
# A parent project that already added the sibling keeps its copy.
# Every sibling is a system dependency that builds only what UForm's targets link, so its own warnings never fail a
# build under UFORM_WERROR.
function (uform_sibling name_ override_ repository_ tag_)
    if (TARGET ${name_}::header)
        return()
    endif ()
    if (EXISTS "${PROJECT_SOURCE_DIR}/${name_}/CMakeLists.txt" AND NOT ${override_})
        add_subdirectory("${PROJECT_SOURCE_DIR}/${name_}" "${PROJECT_BINARY_DIR}/${name_}" EXCLUDE_FROM_ALL SYSTEM)
        return()
    endif ()
    message(STATUS "uform: ${name_} from its local checkout, or else fetched at ${tag_}")
    FetchContent_Declare(
        ${name_}
        GIT_REPOSITORY ${repository_}
        GIT_TAG ${tag_}
        EXCLUDE_FROM_ALL SYSTEM
    )
    FetchContent_MakeAvailable(${name_})
endfunction ()

# The siblings build for the floor, the build kind and the sanitizer UForm builds for.
# ThreadSanitizer then sees their barriers, and ForkUnion leaves out the instructions it cannot see.
# A sibling this project never adds leaves its switches unread.
foreach (sibling_ IN ITEMS NUMKONG STRINGZILLA FORKUNION SMASHTABLE)
    set(${sibling_}_MARCH "${UFORM_MARCH}")
    set(${sibling_}_MULTIVERSIONED "${UFORM_MULTIVERSIONED}")
    set(${sibling_}_BUILD_SANITIZERS "${UFORM_BUILD_SANITIZERS}")
endforeach ()

# Everything below is this project's own: the siblings it builds on, at the commits their submodules pin.

# The header-only NumKong target stubs out the `_best` dispatch the inference products and attention run through, so
# UForm builds the compiled library, with the GPU vendors its own switches add, for its own architectures. No other
# library's GPU code runs in UForm.
set(NUMKONG_CUDA_ARCH "${UFORM_CUDA_ARCH}")
set(NUMKONG_ROCM_ARCH "${UFORM_ROCM_ARCH}")
set(NUMKONG_METAL_ARCH "${UFORM_METAL_ARCH}")
uform_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG https://github.com/ashvardanian/NumKong
    f58322c4bfc83a246bed679b4ba2e57eb1170ec8
)
uform_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA https://github.com/ashvardanian/StringZilla
    a01d3e34da6049b871e3d647333688e216325f1b
)
uform_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION https://github.com/ashvardanian/ForkUnion
    7fe655aa759804b6c3a5ceef517650a5fceee02e
)

# SmashTable's containers, header-only
uform_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE https://github.com/ashvardanian/SmashTable
    09caec7b2084dd20a6563c8f0e87448eebf90d43
)
