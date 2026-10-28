# The CPU UForm's own units are tuned for.

# UFORM_MARCH raises the floor of UForm's own C and C++ units through `uform_warnings`, which every one of them links:
# an ISA level, `native` for this host, or a CPU name, also when cross-compiling. It lands after the libraries, so MSVC
# reads which capabilities NumKong found this host runs. A WebAssembly module keeps its toolchain's flags, as every unit
# of one module must agree.
if (UFORM_MARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "UFORM_MARCH=native needs the host as the target; name a CPU instead")
elseif (UFORM_MARCH AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm")
    if (MSVC AND UFORM_MARCH STREQUAL "native")
        # MSVC has no `/arch:native`, so it takes the `/arch:` of the widest capability this host runs.
        if (numkong_target_diamond_runs)
            set(uform_native_msvc_flags_ ${numkong_target_diamond_flags})
        elseif (numkong_target_skylake_runs)
            set(uform_native_msvc_flags_ ${numkong_target_skylake_flags})
        elseif (numkong_target_haswell_runs)
            set(uform_native_msvc_flags_ ${numkong_target_haswell_flags})
        elseif (numkong_target_neonbfdot_runs)
            set(uform_native_msvc_flags_ ${numkong_target_neonbfdot_flags})
        elseif (numkong_target_neonsdot_runs)
            set(uform_native_msvc_flags_ ${numkong_target_neonsdot_flags})
        elseif (numkong_target_neonhalf_runs)
            set(uform_native_msvc_flags_ ${numkong_target_neonhalf_flags})
        endif ()
        target_compile_options(uform_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:${uform_native_msvc_flags_}>")
    elseif (MSVC)
        message(FATAL_ERROR "MSVC takes UFORM_MARCH=native only; pass an /arch: value such as AVX2, AVX512 or AVX10.2 "
                            "through CMAKE_CXX_FLAGS instead of `${UFORM_MARCH}`"
        )
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64)$")
        # AArch64 names CPUs through `-mcpu`, like `neoverse-v2`, as `-march` takes only architecture versions.
        target_compile_options(uform_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-mcpu=${UFORM_MARCH}>")
    else ()
        target_compile_options(uform_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-march=${UFORM_MARCH}>")
    endif ()
endif ()
