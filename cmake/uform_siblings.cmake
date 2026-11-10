# The sibling libraries, and the switches UForm forwards to them.

# Source dependencies. NumKong spans the tensor substrate, StringZilla parses exchange payloads, ForkUnion is the
# host-side demux and decode thread pool, yyjson reads checkpoint headers. The first three are submodules: a recursive
# clone builds them in place, any other clone fetches the pins below, which match the submodule commits. The caller
# names each sibling's `FETCHCONTENT_SOURCE_DIR_<NAME>`, which points it at a local checkout, and a parent project that
# already added a sibling keeps its own.
function (uform_sibling name_ override_ repository_ tag_)
    if (TARGET ${name_}::header)
        return()
    endif ()
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name_}/CMakeLists.txt" AND NOT DEFINED ${override_})
        add_subdirectory(${name_} EXCLUDE_FROM_ALL SYSTEM)
    else ()
        message(STATUS "uform: ${name_} submodule absent, fetching ${tag_}")
        FetchContent_Declare(
            ${name_}
            GIT_REPOSITORY ${repository_}
            GIT_TAG ${tag_}
            EXCLUDE_FROM_ALL SYSTEM
        )
        FetchContent_MakeAvailable(${name_})
    endif ()
endfunction ()

# The header-only NumKong target stubs out the `_best` dispatch the inference products and attention run through, so
# UForm builds the compiled library, with the GPU vendors its own switches add, for its own architectures. No other
# library's GPU code runs in UForm. Every library compiles at the floor and build kind UForm names.
set(NUMKONG_CUDA_ARCH "${UFORM_CUDA_ARCH}")
set(NUMKONG_ROCM_ARCH "${UFORM_ROCM_ARCH}")
set(NUMKONG_MARCH "${UFORM_MARCH}")
set(STRINGZILLA_MARCH "${UFORM_MARCH}")
set(FORKUNION_MARCH "${UFORM_MARCH}")
set(NUMKONG_MULTIVERSIONED ${UFORM_MULTIVERSIONED})
set(STRINGZILLA_MULTIVERSIONED ${UFORM_MULTIVERSIONED})
set(FORKUNION_MULTIVERSIONED ${UFORM_MULTIVERSIONED})
uform_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG https://github.com/ashvardanian/NumKong
    0a027dc6ab79fed17f08e20a7cd3bdc94e59f655
)
uform_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA https://github.com/ashvardanian/StringZilla
    8b16eea7891d5df9f44706972112d739917b5bf3
)
uform_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION https://github.com/ashvardanian/ForkUnion
    9a69946aa5e9a8cbe56b51dab1dfae7f2281537d
)

# SmashTable's containers, header-only
uform_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE https://github.com/ashvardanian/SmashTable
    788a94f7baef9eae8923009a444fb6b3dae5f8c4
)
