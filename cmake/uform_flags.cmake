# The warning flags UForm's own targets take through `uform_warnings`.

# Warnings apply to UForm's own targets only, so dependencies build with their own flags, and CUDA units with nvcc's.
add_library(uform_warnings INTERFACE)
if (MSVC)
    # The conformance switches MSVC leaves off by default, which every other supported compiler behaves as.
    target_compile_options(
        uform_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:/W4;/Zc:preprocessor;/utf-8;/bigobj>"
                                 "$<$<COMPILE_LANGUAGE:CXX>:/permissive-;/Zc:__cplusplus>"
    )
    if (UFORM_WERROR)
        target_compile_options(uform_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>")
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
