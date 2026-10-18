# The JavaScript SDK has two backends over one C interface. Under Emscripten this directory links the C library into
# `uform.mjs` and `uform.wasm`, an ES module factory the browser and Node load alike; everywhere else it builds the
# Node-API addon `uform.node`, which `cmake-js` drives from `package.json`. Both link the archive, so the engine or the
# stand-in travels inside them.

if (TARGET uform_static)
    set(UFORM_JAVASCRIPT_LIBRARY uform_static)
elseif (TARGET uform_static_stub)
    set(UFORM_JAVASCRIPT_LIBRARY uform_static_stub)
else ()
    message(FATAL_ERROR "The JavaScript SDK links the C interface; configure with `-DUFORM_BUILD_LIB_C=1`.")
endif ()

if (EMSCRIPTEN)
    # Every verb the TypeScript side calls is a root of the link, so the archive members defining them are kept.
    set(UFORM_WASM_EXPORTS
        malloc
        free
        uform_outcome_status
        uform_executor_free
        uform_arena_init
        uform_arena_free
        uform_status_name
        uform_scalar_name
        uform_modality_name
        uform_objective_name
        uform_negatives_name
        uform_multimem_tier_name
        uform_version_name
        uform_capabilities_compiled
        uform_capabilities_available
        uform_model_load_ram_buffer
        uform_model_describe
        uform_model_free
        uform_text_encode
        uform_image_encode_pixels
        uform_audio_encode_samples
    )
    list(TRANSFORM UFORM_WASM_EXPORTS PREPEND "_")
    list(JOIN UFORM_WASM_EXPORTS "," UFORM_WASM_EXPORTS_JOINED)

    # An executable needs a source of its own; the module's whole content comes from the archive.
    file(
        GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/uform_wasm.c"
        CONTENT "/* Every symbol comes from the archive. */\n"
    )
    add_executable(uform_wasm "${CMAKE_CURRENT_BINARY_DIR}/uform_wasm.c")
    target_link_libraries(uform_wasm PRIVATE ${UFORM_JAVASCRIPT_LIBRARY})
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
        "-sEXPORTED_FUNCTIONS=${UFORM_WASM_EXPORTS_JOINED}"
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
target_link_libraries(uform_node PRIVATE ${CMAKE_JS_LIB} ${UFORM_JAVASCRIPT_LIBRARY})

if (MSVC)
    target_compile_options(uform_node PRIVATE /Zc:preprocessor)
endif ()
