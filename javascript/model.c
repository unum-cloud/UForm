/**
 *  @file javascript/model.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The Node-API model lifecycle: loads into RAM or VRAM, the description, and the release.
 *
 *  The @c CompiledModel class carries the encode verbs of @c encode.c beside these, so one handle serves every
 *  modality the model was trained on.
 */
#include <stdlib.h> // `calloc`, `free`
#include <string.h> // `memcpy`

#include "lib.h"

#pragma region Lifecycle

/** A describe call: the shared header, then the report the library fills. */
typedef struct uform_js_describe_call_t {

    /** The outcome, the answer builder and the held values. */
    uform_js_call_t call;

    /** The report @c uform_model_describe fills. */
    uform_model_description_t description;
} uform_js_describe_call_t;

static void uform_js_model_finalize(napi_env env, void *data, void *hint) {
    uform_js_model_t *const model = (uform_js_model_t *)data;
    (void)env;
    (void)hint;
    if (!model) return;
    uform_model_free(model->handle);
    free(model);
}

/** Wraps a handle a load just made, or throws its outcome and frees the handle. */
static napi_value uform_js_model_adopt(napi_env env, uform_outcome_t const *outcome, uform_model_t handle) {
    if (!uform_js_succeeded(env, outcome)) return NULL;
    uform_js_addon_t *const addon = uform_js_addon(env);
    napi_value const instance = addon ? uform_js_instantiate(env, addon->model_class) : NULL;
    uform_js_model_t *const model = (uform_js_model_t *)calloc(1, sizeof(uform_js_model_t));
    if (!instance || !model || napi_wrap(env, instance, model, uform_js_model_finalize, NULL, NULL) != napi_ok) {
        free(model);
        uform_model_free(handle);
        uform_js_throw(env, uform_status_internal_k, "Could not instantiate the CompiledModel");
        return NULL;
    }
    model->handle = handle;
    return instance;
}

/** Where a load reads its checkpoint from, and where it places the weights. */
typedef enum uform_js_load_t {
    uform_js_load_ram_k,
    uform_js_load_ram_buffer_k,
    uform_js_load_vram_k,
    uform_js_load_vram_buffer_k,
} uform_js_load_t;

/** @c loadRam(path, executor, memoryDomains, quantization, cpuClass), @c loadVram(path, executor,
 *  devices, quantization, gpuClass), and their @c Buffer twins taking any @c TypedArray → a model,
 *  the load blocking on @p executor. */
static napi_value uform_js_model_load(napi_env env, napi_callback_info info, uform_js_load_t load) {
    napi_value arguments[5];
    if (!uform_js_arguments(env, info, 5, arguments, NULL)) return NULL;
    bool const from_path = load == uform_js_load_ram_k || load == uform_js_load_vram_k;
    char *path = NULL;
    void *buffer = NULL;
    uform_bytes_view_t bytes = {NULL, 0};
    uform_executor_t executor = NULL;
    uform_scalar_t quantization = uform_scalar_unknown_k;
    int32_t *ordinals = NULL;
    size_t ordinals_count = 0;
    char const *const field = load >= uform_js_load_vram_k ? "devices" : "memoryDomains";
    if (from_path) path = uform_js_string_from_value(env, arguments[0], "path", &bytes);
    else if (uform_js_bytes_from_value(env, arguments[0], "buffer", &buffer, &bytes.length))
        bytes.data = (uint8_t const *)buffer;
    uform_cpu_class_t cpu_class = uform_cpu_class_unknown_k;
    uform_gpu_class_t gpu_class = uform_gpu_class_unknown_k;
    bool const read = from_path ? path != NULL : bytes.data != NULL || !bytes.length;
    bool const pinned = load >= uform_js_load_vram_k ? uform_js_gpu_class_from_value(env, arguments[4], &gpu_class)
                                                     : uform_js_cpu_class_from_value(env, arguments[4], &cpu_class);
    if (!read || !pinned || !uform_js_executor_from_value(env, arguments[1], &executor) ||
        !uform_js_scalar_from_value(env, arguments[3], "quantization", &quantization) ||
        !uform_js_ordinals_from_value(env, arguments[2], field, &ordinals, &ordinals_count)) {
        free(path);
        return NULL;
    }

    uform_model_t handle = NULL;
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_ram_options_t const ram = {ordinals, ordinals_count, cpu_class};
    uform_vram_options_t const vram = {ordinals, ordinals_count, gpu_class};
    switch (load) {
    case uform_js_load_ram_k: {
        uform_model_load_ram_t arguments = {.model = &handle,
                                            .executor = executor,
                                            .outcome = &outcome,
                                            .path = bytes,
                                            .ram = ram,
                                            .quantization = quantization};
        uform_model_load_ram(&arguments);
    } break;
    case uform_js_load_ram_buffer_k: {
        uform_model_load_ram_buffer_t arguments = {.model = &handle,
                                                   .executor = executor,
                                                   .outcome = &outcome,
                                                   .buffer = bytes,
                                                   .ram = ram,
                                                   .quantization = quantization};
        uform_model_load_ram_buffer(&arguments);
    } break;
    case uform_js_load_vram_k: {
        uform_model_load_vram_t arguments = {.model = &handle,
                                             .executor = executor,
                                             .outcome = &outcome,
                                             .path = bytes,
                                             .vram = vram,
                                             .quantization = quantization};
        uform_model_load_vram(&arguments);
    } break;
    case uform_js_load_vram_buffer_k: {
        uform_model_load_vram_buffer_t arguments = {.model = &handle,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .buffer = bytes,
                                                    .vram = vram,
                                                    .quantization = quantization};
        uform_model_load_vram_buffer(&arguments);
    } break;
    }
    free(ordinals);
    free(path);
    return uform_js_model_adopt(env, &outcome, handle);
}

static napi_value uform_js_model_load_ram(napi_env env, napi_callback_info info) {
    return uform_js_model_load(env, info, uform_js_load_ram_k);
}

static napi_value uform_js_model_load_ram_buffer(napi_env env, napi_callback_info info) {
    return uform_js_model_load(env, info, uform_js_load_ram_buffer_k);
}

static napi_value uform_js_model_load_vram(napi_env env, napi_callback_info info) {
    return uform_js_model_load(env, info, uform_js_load_vram_k);
}

static napi_value uform_js_model_load_vram_buffer(napi_env env, napi_callback_info info) {
    return uform_js_model_load(env, info, uform_js_load_vram_buffer_k);
}

/** @c free() - releases the handle now, waiting for every deferred call naming it; idempotent. */
static napi_value uform_js_model_release(napi_env env, napi_callback_info info) {
    uform_js_model_t *const model = (uform_js_model_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!model) return NULL;
    uform_model_free(model->handle);
    model->handle = NULL;
    return uform_js_undefined(env);
}

static napi_value uform_js_finish_describe(napi_env env, uform_js_call_t *call, napi_value const *slots) {
    uform_model_description_t const *const description = &((uform_js_describe_call_t *)call)->description;
    static char const *const names[] = {"dimensions", "context",  "quantization", "memoryUsage",
                                        "cpuClass",   "gpuClass", "tokenWidth",   "choiceTemperatureGroups"};
    napi_value const values[] = {uform_js_number(env, description->dimensions),
                                 uform_js_number(env, description->context),
                                 uform_js_static_string(env, uform_scalar_name(description->quantization)),
                                 uform_js_number(env, description->memory_usage),
                                 uform_js_class_name(env, uform_cpu_class_name(description->cpu_class)),
                                 uform_js_class_name(env, uform_gpu_class_name(description->gpu_class)),
                                 uform_js_number(env, description->token_width),
                                 uform_js_number(env, description->choice_temperature_groups)};
    (void)slots;
    return uform_js_object(env, sizeof(names) / sizeof(names[0]), names, values);
}

/** @c describe(executor, completion) → the geometry and placement, each read once. */
static napi_value uform_js_model_describe(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    uform_js_model_t *const model = (uform_js_model_t *)uform_js_self(env, info, 2, arguments, &self);
    if (model && !model->handle) uform_js_throw(env, uform_status_invalid_argument_k, "The model was freed");
    if (!model || !model->handle) return NULL;
    uform_js_describe_call_t local;
    uform_model_describe_t describe = {.model = model->handle};
    uform_js_describe_call_t *const call = (uform_js_describe_call_t *)uform_js_call_open_on(
        env, arguments, &local.call, sizeof(local), &describe.executor, &describe.completion);
    if (!call) return NULL;
    napi_value const slots[uform_js_slots_k] = {self, arguments[0]};
    call->call.finish = uform_js_finish_describe;
    describe.outcome = &call->call.outcome;
    describe.description = &call->description;
    if (!uform_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    uform_model_describe(&describe);
    return uform_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

#pragma endregion Lifecycle

napi_value uform_js_model_define(napi_env env) {
    static napi_property_descriptor const lifecycle[] = {
        {"free", NULL, uform_js_model_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"describe", NULL, uform_js_model_describe, NULL, NULL, NULL, napi_default_method, NULL},
        {"loadRam", NULL, uform_js_model_load_ram, NULL, NULL, NULL, napi_static, NULL},
        {"loadRamBuffer", NULL, uform_js_model_load_ram_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"loadVram", NULL, uform_js_model_load_vram, NULL, NULL, NULL, napi_static, NULL},
        {"loadVramBuffer", NULL, uform_js_model_load_vram_buffer, NULL, NULL, NULL, napi_static, NULL},
    };
    size_t const lifecycle_count = sizeof(lifecycle) / sizeof(lifecycle[0]);
    size_t encode_count = 0;
    napi_property_descriptor const *const encode = uform_js_encode_properties(&encode_count);
    napi_property_descriptor properties[16];
    if (lifecycle_count + encode_count > sizeof(properties) / sizeof(properties[0])) {
        uform_js_throw(env, uform_status_internal_k, "The CompiledModel has more verbs than room");
        return NULL;
    }
    memcpy(properties, lifecycle, sizeof(lifecycle));
    memcpy(properties + lifecycle_count, encode, encode_count * sizeof(napi_property_descriptor));
    return uform_js_class_define(env, "CompiledModel", lifecycle_count + encode_count, properties);
}
