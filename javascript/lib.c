/**
 *  @file javascript/lib.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Module glue of the UForm Node-API addon: the classes and functions it publishes, and the
 *      per-environment state its verbs type-check against.
 *
 *  Only @c <node_api.h> - no C++ wrapper, no exceptions, and nothing of the library beyond its C surface, so one
 *  prebuilt shared library serves every Node release the ABI covers.
 */
#include <stdlib.h> // `calloc`, `free`

#include "lib.h"

#pragma region Registration

static void uform_js_addon_finalize(napi_env env, void *data, void *hint) {
    uform_js_addon_t *const addon = (uform_js_addon_t *)data;
    (void)hint;
    if (!addon) return;
    napi_ref const references[] = {addon->executor_class, addon->arena_class,    addon->model_class, addon->mesh_class,
                                   addon->trainer_class,  addon->deferred_class, addon->seal};
    for (size_t position = 0; position != sizeof(references) / sizeof(references[0]); ++position)
        if (references[position]) napi_delete_reference(env, references[position]);
    free(addon);
}

NAPI_MODULE_INIT() {
    static napi_property_descriptor const module_properties[] = {
        {"version", NULL, uform_js_version, NULL, NULL, NULL, napi_default, NULL},
        {"capabilitiesCompiled", NULL, uform_js_capabilities_compiled, NULL, NULL, NULL, napi_default, NULL},
        {"capabilitiesAvailable", NULL, uform_js_capabilities_available, NULL, NULL, NULL, napi_default, NULL},
    };

    uform_js_addon_t *const addon = (uform_js_addon_t *)calloc(1, sizeof(uform_js_addon_t));
    napi_value seal = NULL, scalars = NULL, modalities = NULL, objectives = NULL, negatives = NULL;
    if (!addon) { // the state goes first, since instantiating a class checks the seal
        uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the addon state");
        return NULL;
    }
    if (napi_set_instance_data(env, addon, uform_js_addon_finalize, NULL) != napi_ok) {
        uform_js_addon_finalize(env, addon, NULL);
        uform_js_throw(env, uform_status_internal_k, "Could not register the addon state");
        return NULL;
    }
    if (napi_create_external(env, addon, NULL, NULL, &seal) != napi_ok ||
        napi_create_reference(env, seal, 1, &addon->seal) != napi_ok) {
        uform_js_throw(env, uform_status_internal_k, "Could not create the addon seal");
        return NULL;
    }

    napi_value const classes[] = {
        uform_js_executor_define(env), uform_js_arena_define(env),   uform_js_model_define(env),
        uform_js_mesh_define(env),     uform_js_trainer_define(env), uform_js_deferred_define(env),
    };
    napi_ref *const references[] = {&addon->executor_class, &addon->arena_class,   &addon->model_class,
                                    &addon->mesh_class,     &addon->trainer_class, &addon->deferred_class};
    static char const *const names[] = {"Executor",     "Arena",           "CompiledModel",
                                        "CompiledMesh", "CompiledTrainer", "CompiledDeferred"};
    for (size_t position = 0; position != sizeof(classes) / sizeof(classes[0]); ++position) {
        if (!classes[position]) return NULL;
        if (napi_create_reference(env, classes[position], 1, references[position]) != napi_ok ||
            napi_set_named_property(env, exports, names[position], classes[position]) != napi_ok) {
            uform_js_throw(env, uform_status_internal_k, "Could not register the addon classes");
            return NULL;
        }
    }

    if (napi_define_properties(env, exports, sizeof(module_properties) / sizeof(module_properties[0]),
                               module_properties) != napi_ok ||
        !(scalars = uform_js_scalars(env)) || napi_set_named_property(env, exports, "scalars", scalars) != napi_ok ||
        !(modalities = uform_js_modalities(env)) ||
        napi_set_named_property(env, exports, "modalities", modalities) != napi_ok ||
        !(objectives = uform_js_objectives(env)) ||
        napi_set_named_property(env, exports, "objectives", objectives) != napi_ok ||
        !(negatives = uform_js_negatives(env)) ||
        napi_set_named_property(env, exports, "negatives", negatives) != napi_ok) {
        uform_js_throw(env, uform_status_internal_k, "Could not publish the addon exports");
        return NULL;
    }
    return exports;
}

#pragma endregion Registration
