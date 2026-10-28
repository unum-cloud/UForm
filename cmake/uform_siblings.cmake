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
# UForm builds the compiled library, with the GPU vendors its own switches add, for its own architectures. Every
# library tunes its units for the CPU UForm targets, when it names one.
set(NUMKONG_BUILD_CUDA ${UFORM_BUILD_CUDA})
set(NUMKONG_BUILD_ROCM ${UFORM_BUILD_ROCM})
set(NUMKONG_CUDA_ARCHITECTURES "${UFORM_CUDA_ARCHITECTURES}")
if (UFORM_TARGET_ARCH)
    set(NUMKONG_TARGET_ARCH "${UFORM_TARGET_ARCH}")
    set(STRINGZILLA_TARGET_ARCH "${UFORM_TARGET_ARCH}")
    set(FORKUNION_TARGET_ARCH "${UFORM_TARGET_ARCH}")
endif ()
uform_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG https://github.com/ashvardanian/NumKong
    3fecca8d625ccc89ba2605e195e621e48446c795
)
uform_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA https://github.com/ashvardanian/StringZilla
    53fb6a67b188561ffcab506ee3d03901716b1b49
)
uform_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION https://github.com/ashvardanian/ForkUnion
    0b9296ebe0dc3886f4f850f0dc97d6ce382ac909
)

# SmashTable's containers, header-only
uform_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE https://github.com/ashvardanian/SmashTable
    e45080c7ea61a5a9c7fc90ac8b9595b49b341574
)
