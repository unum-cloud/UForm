# The warning flags UForm's own targets take through `uform_warnings`.

# Warnings apply to UForm's own targets only, so dependencies build with their own flags, and CUDA units with nvcc's.
add_library(uform_warnings INTERFACE)
if (MSVC)
    target_compile_options(uform_warnings INTERFACE /W4 /permissive- /Zc:__cplusplus)
    if (UFORM_WERROR)
        target_compile_options(uform_warnings INTERFACE /WX)
    endif ()
else ()
    target_compile_options(
        uform_warnings
        INTERFACE
            "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic;-Wno-unknown-pragmas;-Wno-missing-field-initializers>"
            "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wdocumentation>"
    )
    if (UFORM_WERROR)
        target_compile_options(uform_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
    endif ()
endif ()
