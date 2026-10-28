# The CPython boundary lives in `python/*.c` and links the C interface as an archive, so the wheel carries one shared
# object. `scikit-build-core` drives this from `pyproject.toml`.

set(uform_python_library_ uform_static)
if (UFORM_BUILD_STUB)
    set(uform_python_library_ uform_static_stub)
endif ()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

# Module glue first, then the shared vocabulary, then one file per header of the C interface.
python_add_library(
    uform_python
    MODULE
    WITH_SOABI
    lib.c
    types.c
    model.c
    encode.c
    train.c
)

# `uform/__init__.py` imports it as `uform.lib`, so the file name is part of the interface.
set_target_properties(
    uform_python
    PROPERTIES OUTPUT_NAME "lib"
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
)

# The library is C++, so the C++ driver links the extension and brings the standard library along.
set_target_properties(uform_python PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(uform_python PRIVATE ${PROJECT_SOURCE_DIR}/c ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(uform_python PRIVATE ${uform_python_library_})

# CPython's own symbols come from the interpreter, so they are absent at link time and the linker must not insist. The
# module's one export is its init function, and the archive's symbols, NumKong's kernels among them, stay inside it.
set_target_properties(
    uform_python
    PROPERTIES C_VISIBILITY_PRESET hidden
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
)
if (APPLE)
    target_link_options(uform_python PRIVATE -undefined dynamic_lookup "LINKER:-exported_symbol,_PyInit_lib")
elseif (MSVC)
    target_link_options(uform_python PRIVATE /FORCE)
else ()
    target_link_options(uform_python PRIVATE -Wl,--unresolved-symbols=ignore-all "LINKER:--exclude-libs,ALL")
endif ()

# Named, or every dependency's own install rules land in the wheel beside the extension.
install(
    TARGETS uform_python
    DESTINATION uform
    COMPONENT uform_python
)
