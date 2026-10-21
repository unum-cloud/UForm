# Resolve a vendor library for the loader tier (libcudf, cuFile, NVDEC, libav*, ...). One layered search:
# the RAPIDS / NVIDIA cu13 Python-wheel layouts
# (`site-packages/<wheel>/lib64`, `site-packages/nvidia/cu13/lib`), then the system CUDA toolkit and the Debian
# multiarch directories. `uform_library_hints_` holds the shared defaults; a call's `EXTRA_HINTS` are searched
# before them so a site-specific install wins.
#
# Naming: `UFORM_*` are public cache variables, `uform_*_` are internal, parameters and locals end in `_`, and parsed
# arguments are `arg_*`.

# Wheel discovery: an explicit site-packages override, then — in priority order — an activated venv
# ($VIRTUAL_ENV), the repo-local .venv, or the interpreter's own site-packages. The first directory that
# actually holds the cu13 wheel layout wins.
set(UFORM_SITE_PACKAGES
    ""
    CACHE PATH "Python site-packages directory holding the RAPIDS / NVIDIA cu13 wheels"
)
set(uform_wheel_site_packages_ "${UFORM_SITE_PACKAGES}")
if (NOT uform_wheel_site_packages_)
    set(uform_candidate_site_packages_ "")
    if (DEFINED ENV{VIRTUAL_ENV})
        file(GLOB uform_venv_env_ "$ENV{VIRTUAL_ENV}/lib/python*/site-packages")
        list(APPEND uform_candidate_site_packages_ ${uform_venv_env_})
    endif ()
    file(GLOB uform_venv_repo_ "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python*/site-packages")
    list(APPEND uform_candidate_site_packages_ ${uform_venv_repo_})
    find_package(Python QUIET COMPONENTS Interpreter)
    if (Python_SITELIB)
        list(APPEND uform_candidate_site_packages_ "${Python_SITELIB}")
    endif ()
    foreach (site_packages_ IN LISTS uform_candidate_site_packages_)
        if (EXISTS "${site_packages_}/nvidia" OR EXISTS "${site_packages_}/libcudf")
            set(uform_wheel_site_packages_ "${site_packages_}")
            break()
        endif ()
    endforeach ()
endif ()
if (uform_wheel_site_packages_)
    message(STATUS "uform: wheel libraries from ${uform_wheel_site_packages_}")
endif ()

set(uform_library_hints_ ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib64)
if (uform_wheel_site_packages_)
    list(
        PREPEND
        uform_library_hints_
        "${uform_wheel_site_packages_}/nvidia/cu13/lib"
        "${uform_wheel_site_packages_}/libcudf/lib64"
        "${uform_wheel_site_packages_}/librmm/lib64"
        "${uform_wheel_site_packages_}/libkvikio/lib64"
        "${uform_wheel_site_packages_}/nvidia/libnvcomp/lib64"
        "${uform_wheel_site_packages_}/rapids_logger/lib64"
        "${uform_wheel_site_packages_}/nvidia/nvimgcodec/lib64"
    )
endif ()

# Usage:  uform_find_library(<output_> <library_name_> [EXTRA_NAMES <name>...] [EXTRA_HINTS <dir>...])
#
# EXTRA_NAMES are tried before <library_name_>: the NVIDIA wheels ship only versioned sonames (e.g.
# `libnvimgcodec.so.0`), and `find_library` is name-major, so probing the versioned soname first lets the
# wheel copy win over a stray unversioned system library. The result is the cache entry `find_library` writes.
function (uform_find_library output_ library_name_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "EXTRA_NAMES;EXTRA_HINTS")
    find_library(
        ${output_}
        NAMES ${arg_EXTRA_NAMES} ${library_name_}
        HINTS ${arg_EXTRA_HINTS} ${uform_library_hints_}
    )
    get_property(
        found_
        CACHE ${output_}
        PROPERTY VALUE
    )
    if (found_)
        message(STATUS "uform: found ${library_name_} -> ${found_}")
    endif ()
endfunction ()
