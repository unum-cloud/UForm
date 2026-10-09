# Resolve a vendor library for the loader tier (libcudf, cuFile, NVDEC, libav*, ...). One layered search:
# the RAPIDS / NVIDIA cu13 Python-wheel layouts
# (`site-packages/<wheel>/lib64`, `site-packages/nvidia/cu13/lib`), then the system CUDA toolkit and the Debian
# multiarch directories. `UFORM_LIBRARY_HINTS` holds the shared defaults; a call's `EXTRA_HINTS` are searched
# before them so a site-specific install wins.
#
# Naming: `UFORM_*` are public cache variables, `_uform_*` are internal (locals and parsed arguments alike).

# Wheel discovery: an explicit site-packages override, then — in priority order — an activated venv
# ($VIRTUAL_ENV), the repo-local .venv, or the interpreter's own site-packages. The first directory that
# actually holds the cu13 wheel layout wins.
set(UFORM_SITE_PACKAGES
    ""
    CACHE PATH "Python site-packages directory holding the RAPIDS / NVIDIA cu13 wheels"
)
if (NOT UFORM_SITE_PACKAGES)
    set(_uform_candidate_site_packages "")
    if (DEFINED ENV{VIRTUAL_ENV})
        file(GLOB _uform_venv_env "$ENV{VIRTUAL_ENV}/lib/python*/site-packages")
        list(APPEND _uform_candidate_site_packages ${_uform_venv_env})
    endif ()
    file(GLOB _uform_venv_repo "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python*/site-packages")
    list(APPEND _uform_candidate_site_packages ${_uform_venv_repo})
    find_package(Python QUIET COMPONENTS Interpreter)
    if (Python_SITELIB)
        list(APPEND _uform_candidate_site_packages "${Python_SITELIB}")
    endif ()
    foreach (_uform_site_packages IN LISTS _uform_candidate_site_packages)
        if (EXISTS "${_uform_site_packages}/nvidia" OR EXISTS "${_uform_site_packages}/libcudf")
            set(UFORM_SITE_PACKAGES "${_uform_site_packages}")
            break()
        endif ()
    endforeach ()
endif ()
if (UFORM_SITE_PACKAGES)
    message(STATUS "uform: wheel libraries from ${UFORM_SITE_PACKAGES}")
endif ()

set(UFORM_LIBRARY_HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib64)
if (UFORM_SITE_PACKAGES)
    list(
        PREPEND
        UFORM_LIBRARY_HINTS
        "${UFORM_SITE_PACKAGES}/nvidia/cu13/lib"
        "${UFORM_SITE_PACKAGES}/libcudf/lib64"
        "${UFORM_SITE_PACKAGES}/librmm/lib64"
        "${UFORM_SITE_PACKAGES}/libkvikio/lib64"
        "${UFORM_SITE_PACKAGES}/nvidia/libnvcomp/lib64"
        "${UFORM_SITE_PACKAGES}/rapids_logger/lib64"
        "${UFORM_SITE_PACKAGES}/nvidia/nvimgcodec/lib64"
    )
endif ()

# Usage:  uform_find_library(<OUT_VAR> <LIBNAME> [EXTRA_NAMES <name>...] [EXTRA_HINTS <dir>...])
#
# EXTRA_NAMES are tried before <LIBNAME>: the NVIDIA wheels ship only versioned sonames (e.g.
# `libnvimgcodec.so.0`), and `find_library` is name-major, so probing the versioned soname first lets the
# wheel copy win over a stray unversioned system library.
function (uform_find_library OUT_VAR LIBNAME)
    cmake_parse_arguments(PARSE_ARGV 2 _uform "" "" "EXTRA_NAMES;EXTRA_HINTS")
    find_library(
        ${OUT_VAR}
        NAMES ${_uform_EXTRA_NAMES} ${LIBNAME}
        HINTS ${_uform_EXTRA_HINTS} ${UFORM_LIBRARY_HINTS}
    )
    if (${OUT_VAR})
        message(STATUS "uform: found ${LIBNAME} -> ${${OUT_VAR}}")
    endif ()
    set(${OUT_VAR}
        "${${OUT_VAR}}"
        PARENT_SCOPE
    )
endfunction ()
