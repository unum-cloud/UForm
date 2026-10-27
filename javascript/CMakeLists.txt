# The JavaScript SDK has two backends over one C interface. Under Emscripten this directory links the C library into
# `uform.mjs` and `uform.wasm`, an ES module factory the browser and Node load alike; everywhere else it builds the
# Node-API addon `uform.node`, which `cmake-js` drives from `package.json`. Both link the archive, so the engine or the
# stand-in travels inside them.

set(uform_javascript_library_ uform_static)
if (UFORM_BUILD_STUB)
    set(uform_javascript_library_ uform_static_stub)
endif ()

if (EMSCRIPTEN)
    # Every verb the TypeScript side calls is a root of the link, so the archive members defining them are kept. The
    # names are Emscripten's, with the leading underscore of a C symbol.
    set(uform_wasm_exports_
        _malloc
        _free
        _uform_outcome_status
        _uform_executor_free
        _uform_arena_init
        _uform_arena_free
        _uform_status_name
        _uform_scalar_name
        _uform_modality_name
        _uform_objective_name
        _uform_negatives_name
        _uform_multimem_tier_name
        _uform_cpu_class_name
        _uform_gpu_class_name
        _uform_version_name
        _uform_capabilities_compiled
        _uform_capabilities_available
        _uform_model_load_ram_buffer
        _uform_model_describe
        _uform_model_free
        _uform_text_encode
        _uform_image_encode_pixels
        _uform_audio_encode_samples
        _uform_text_choose
    )

    # An executable needs a source of its own; the module's whole content comes from the archive.
    file(
        GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/uform_wasm.c"
        CONTENT "/* Every symbol comes from the archive. */\n"
    )
    add_executable(uform_wasm "${CMAKE_CURRENT_BINARY_DIR}/uform_wasm.c")
    target_link_libraries(uform_wasm PRIVATE ${uform_javascript_library_})
    set_target_properties(
        uform_wasm
        PROPERTIES OUTPUT_NAME "uform"
                   SUFFIX ".mjs"
                   LINKER_LANGUAGE CXX
    )
    target_link_options(
        uform_wasm
        PRIVATE
        "-sMODULARIZE"
        "-sEXPORT_ES6"
        "-sEXPORT_NAME=createUForm"
        "-sALLOW_MEMORY_GROWTH"
        "-sALLOW_TABLE_GROWTH"
        "-sENVIRONMENT=web,worker,node"
        "-sEXPORTED_FUNCTIONS=$<JOIN:${uform_wasm_exports_},$<COMMA>>"
        "-sEXPORTED_RUNTIME_METHODS=HEAPU8,UTF8ToString,addFunction,removeFunction"
    )
    return()
endif ()

if (NOT CMAKE_JS_INC)
    message(FATAL_ERROR "The Node addon is built by `cmake-js`, which supplies the headers and the import library.")
endif ()

# `CMAKE_JS_SRC` is the Windows delay-load hook and empty everywhere else.
add_library(uform_node SHARED lib.c types.c model.c encode.c train.c ${CMAKE_JS_SRC})

# The loader resolves the addon by file name, so the `.node` suffix and the bare stem are part of the interface rather
# than a platform convention.
set_target_properties(
    uform_node
    PROPERTIES PREFIX ""
               SUFFIX ".node"
               OUTPUT_NAME "uform"
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
               LINKER_LANGUAGE CXX
)

target_include_directories(uform_node PRIVATE ${CMAKE_JS_INC} ${PROJECT_SOURCE_DIR}/c ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(uform_node PRIVATE NAPI_VERSION=8)
target_link_libraries(uform_node PRIVATE ${CMAKE_JS_LIB} ${uform_javascript_library_})

if (MSVC)
    target_compile_options(uform_node PRIVATE /Zc:preprocessor)
endif ()
