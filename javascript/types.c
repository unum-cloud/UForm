/**
 *  @file javascript/types.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The Node-API vocabulary every verb shares.
 *
 *  Closed-set naming, argument parsing, tapes, the @c Executor whose threads or streams every verb runs on, the
 *  @c Arena every scaling output lands in, and the @c CompiledDeferred a deferred call answers.
 */
#include <stdio.h>  // `snprintf`
#include <stdlib.h> // `malloc`, `calloc`, `realloc`, `free`
#include <string.h> // `memset`, `strcmp`, `strlen`

#include "lib.h"

#pragma region Naming

static uform_bytes_view_t uform_js_name_view(char const *name) {
    return (uform_bytes_view_t) {(uint8_t const *)name, strlen(name)};
}

/**
 *  @brief Reads a closed-set name into @p name, which the caller reads through the C name tables.
 *
 *  The C readers answer the default for a name they do not know, so a name counts as read only
 *  when the C writer spells the answer back the same way, which refuses a misspelling rather than
 *  reading it as the default.
 */
static bool uform_js_name_from_value(napi_env env, napi_value value, char const *field,
                                     char name[uform_js_name_capacity_k]) {
    size_t length = 0;
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok || kind != napi_string ||
        napi_get_value_string_utf8(env, value, name, uform_js_name_capacity_k, &length) != napi_ok)
        return uform_js_throw_field(env, field, "must be a string");
    return true;
}

static bool uform_js_mood_from_value(napi_env env, napi_value value, uform_mood_t *result) {
    char name[uform_js_name_capacity_k];
    *result = uform_mood_parked_k;
    if (uform_js_value_absent(env, value)) return true;
    if (!uform_js_name_from_value(env, value, "mood", name)) return false;
    *result = uform_mood_from_name(uform_js_name_view(name));
    if (strcmp(uform_mood_name(*result), name) == 0) return true;
    return uform_js_throw_field(env, "mood", "must name a mood");
}

bool uform_js_cpu_class_from_value(napi_env env, napi_value value, uform_cpu_class_t *result) {
    char name[uform_js_name_capacity_k];
    *result = uform_cpu_class_unknown_k;
    if (uform_js_value_absent(env, value)) return true;
    if (!uform_js_name_from_value(env, value, "cpuClass", name)) return false;
    *result = uform_cpu_class_from_name(uform_js_name_view(name));
    if (*result != uform_cpu_class_unknown_k && strcmp(uform_cpu_class_name(*result), name) == 0) return true;
    return uform_js_throw(env, uform_status_unknown_kind_k, "The 'cpuClass' argument names no CPU class");
}

bool uform_js_gpu_class_from_value(napi_env env, napi_value value, uform_gpu_class_t *result) {
    char name[uform_js_name_capacity_k];
    *result = uform_gpu_class_unknown_k;
    if (uform_js_value_absent(env, value)) return true;
    if (!uform_js_name_from_value(env, value, "gpuClass", name)) return false;
    *result = uform_gpu_class_from_name(uform_js_name_view(name));
    if (*result != uform_gpu_class_unknown_k && strcmp(uform_gpu_class_name(*result), name) == 0) return true;
    return uform_js_throw(env, uform_status_unknown_kind_k, "The 'gpuClass' argument names no GPU class");
}

napi_value uform_js_class_name(napi_env env, char const *name) {
    napi_value spelled = NULL;
    if (!name || strcmp(name, "unknown") == 0) napi_get_null(env, &spelled);
    else napi_create_string_utf8(env, name, NAPI_AUTO_LENGTH, &spelled);
    return spelled;
}

bool uform_js_scalar_from_value(napi_env env, napi_value value, char const *field, uform_scalar_t *result) {
    uint32_t code = 0;
    if (napi_get_value_uint32(env, value, &code) != napi_ok || !uform_js_row_bytes((uform_scalar_t)code, 1))
        return uform_js_throw_field(env, field, "must name a scalar kind");
    *result = (uform_scalar_t)code;
    return true;
}

/** Maps each of @p codes to the canonical spelling @p spell gives it, frozen. */
static napi_value uform_js_codes(napi_env env, int const *codes, size_t count, char const *(*spell)(int)) {
    napi_value table = NULL;
    bool mapped = napi_create_object(env, &table) == napi_ok;
    for (size_t position = 0; mapped && position != count; ++position)
        mapped = napi_set_named_property(env, table, spell(codes[position]),
                                         uform_js_number(env, (size_t)codes[position])) == napi_ok;
    if (!mapped || napi_object_freeze(env, table) != napi_ok) {
        uform_js_throw(env, uform_status_internal_k, "Could not map the closed set");
        return NULL;
    }
    return table;
}

static char const *uform_js_scalar_spell(int scalar) { return uform_scalar_name((uform_scalar_t)scalar); }

static char const *uform_js_modality_spell(int modality) { return uform_modality_name((uform_modality_t)modality); }

static char const *uform_js_objective_spell(int objective) {
    return uform_objective_name((uform_objective_t)objective);
}

static char const *uform_js_negatives_spell(int negatives) {
    return uform_negatives_name((uform_negatives_t)negatives);
}

napi_value uform_js_scalars(napi_env env) {
    static int const scalars[] = {
        uform_scalar_f64_k,  uform_scalar_f32_k,  uform_scalar_bf16_k, uform_scalar_f16_k,  uform_scalar_e5m2_k,
        uform_scalar_e4m3_k, uform_scalar_e3m2_k, uform_scalar_e2m3_k, uform_scalar_i8_k,   uform_scalar_i4x2_k,
        uform_scalar_i32_k,  uform_scalar_u8_k,   uform_scalar_u4x2_k, uform_scalar_u1x8_k,
    };
    return uform_js_codes(env, scalars, sizeof(scalars) / sizeof(scalars[0]), uform_js_scalar_spell);
}

napi_value uform_js_modalities(napi_env env) {
    static int const modalities[] = {uform_modality_text_k, uform_modality_image_k, uform_modality_video_k,
                                     uform_modality_audio_k};
    return uform_js_codes(env, modalities, sizeof(modalities) / sizeof(modalities[0]), uform_js_modality_spell);
}

napi_value uform_js_objectives(napi_env env) {
    static int const objectives[] = {uform_objective_masked_k, uform_objective_contrastive_k, uform_objective_joint_k};
    return uform_js_codes(env, objectives, sizeof(objectives) / sizeof(objectives[0]), uform_js_objective_spell);
}

napi_value uform_js_negatives(napi_env env) {
    static int const negatives[] = {uform_negatives_ambient_k, uform_negatives_named_k};
    return uform_js_codes(env, negatives, sizeof(negatives) / sizeof(negatives[0]), uform_js_negatives_spell);
}

size_t uform_js_row_bytes(uform_scalar_t scalar, size_t columns) {
    switch (scalar) {
    case uform_scalar_f64_k: return columns * 8;
    case uform_scalar_f32_k:
    case uform_scalar_i32_k: return columns * 4;
    case uform_scalar_bf16_k:
    case uform_scalar_f16_k: return columns * 2;
    case uform_scalar_e5m2_k:
    case uform_scalar_e4m3_k:
    case uform_scalar_e3m2_k:
    case uform_scalar_e2m3_k:
    case uform_scalar_i8_k:
    case uform_scalar_u8_k: return columns;
    case uform_scalar_i4x2_k:
    case uform_scalar_u4x2_k: return columns / 2 + columns % 2;
    case uform_scalar_u1x8_k: return columns / 8 + (columns % 8 != 0);
    default: return 0;
    }
}

#pragma endregion Naming

#pragma region Argument Parsing

bool uform_js_throw(napi_env env, uform_status_t status, char const *message) {
    napi_throw_error(env, uform_status_name(status), message);
    return false;
}

bool uform_js_throw_field(napi_env env, char const *field, char const *expectation) {
    char message[uform_js_message_capacity_k];
    snprintf(message, sizeof(message), "The '%s' argument %s", field, expectation);
    return uform_js_throw(env, uform_status_invalid_argument_k, message);
}

bool uform_js_succeeded(napi_env env, uform_outcome_t const *outcome) {
    if (outcome->status == uform_status_success_k) return true;
    return uform_js_throw(env, outcome->status, outcome->message ? outcome->message : "The operation failed");
}

bool uform_js_value_absent(napi_env env, napi_value value) {
    napi_valuetype kind = napi_undefined;
    if (!value || napi_typeof(env, value, &kind) != napi_ok) return true;
    return kind == napi_undefined || kind == napi_null;
}

napi_value uform_js_property(napi_env env, napi_value object, char const *name) {
    napi_value property = NULL;
    napi_valuetype kind = napi_undefined;
    if (!object || napi_typeof(env, object, &kind) != napi_ok || kind != napi_object ||
        napi_get_named_property(env, object, name, &property) != napi_ok)
        return uform_js_undefined(env);
    return property;
}

bool uform_js_double_from_value(napi_env env, napi_value value, char const *field, double *result) {
    *result = 0;
    if (uform_js_value_absent(env, value)) return true;
    if (napi_get_value_double(env, value, result) != napi_ok)
        return uform_js_throw_field(env, field, "must be a finite real");
    return true;
}

bool uform_js_size_from_value(napi_env env, napi_value value, char const *field, size_t *result) {
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok) return uform_js_throw_field(env, field, "could not be inspected");
    if (kind == napi_undefined || kind == napi_null) {
        *result = 0;
        return true;
    }
    if (kind == napi_bigint) {
        uint64_t wide = 0;
        bool lossless = false;
        if (napi_get_value_bigint_uint64(env, value, &wide, &lossless) != napi_ok || !lossless || wide > SIZE_MAX)
            return uform_js_throw_field(env, field, "must be a non-negative integer");
        *result = (size_t)wide;
        return true;
    }
    double number = 0;
    if (kind != napi_number || napi_get_value_double(env, value, &number) != napi_ok)
        return uform_js_throw_field(env, field, "must be a number or a BigInt");
    if (!(number >= 0 && number < 18446744073709551616.0) || (double)(uint64_t)number != number ||
        (uint64_t)number > SIZE_MAX) // ranged before the casts, which are undefined outside it
        return uform_js_throw_field(env, field, "must be a non-negative integer");
    *result = (size_t)number;
    return true;
}

char *uform_js_string_from_value(napi_env env, napi_value value, char const *field, uform_bytes_view_t *view) {
    napi_valuetype kind = napi_undefined;
    size_t length = 0;
    if (napi_typeof(env, value, &kind) != napi_ok || kind != napi_string ||
        napi_get_value_string_utf8(env, value, NULL, 0, &length) != napi_ok) {
        uform_js_throw_field(env, field, "must be a string");
        return NULL;
    }
    char *const text = (char *)malloc(length + 1); // measured first, as a short buffer truncates silently
    if (!text) {
        uform_js_throw(env, uform_status_out_of_memory_k, "Could not copy the string");
        return NULL;
    }
    if (napi_get_value_string_utf8(env, value, text, length + 1, &length) != napi_ok) {
        free(text);
        uform_js_throw_field(env, field, "must be a string");
        return NULL;
    }
    view->data = (uint8_t const *)text;
    view->length = length;
    return text;
}

bool uform_js_completion_from_value(napi_env env, napi_value value, uform_completion_t *result) {
    size_t completion = 0;
    if (!uform_js_size_from_value(env, value, "completion", &completion)) return false;
    if (completion != uform_completion_blocking_k && completion != uform_completion_deferred_k)
        return uform_js_throw_field(env, "completion", "must be blocking or deferred");
    *result = (uform_completion_t)completion;
    return true;
}

bool uform_js_ordinals_from_value(napi_env env, napi_value value, char const *field, int32_t **ordinals,
                                  size_t *count) {
    *ordinals = NULL;
    *count = 0;
    if (uform_js_value_absent(env, value)) return true;

    bool is_array = false;
    uint32_t length = 0;
    if (napi_is_array(env, value, &is_array) != napi_ok || !is_array ||
        napi_get_array_length(env, value, &length) != napi_ok)
        return uform_js_throw_field(env, field, "must be an array of integers");
    if (!length) return true;

    int32_t *const claimed = (int32_t *)malloc((size_t)length * sizeof(int32_t));
    if (!claimed) return uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the residency arm");
    for (uint32_t position = 0; position != length; ++position) {
        napi_value entry = NULL;
        if (napi_get_element(env, value, position, &entry) != napi_ok ||
            napi_get_value_int32(env, entry, &claimed[position]) != napi_ok) {
            free(claimed);
            return uform_js_throw_field(env, field, "must hold integers");
        }
    }
    *ordinals = claimed;
    *count = length;
    return true;
}

/** Bytes one element of a @c TypedArray occupies; zero for a type this binding does not know. */
static size_t uform_js_typedarray_unit_bytes(napi_typedarray_type type) {
    switch (type) {
    case napi_int8_array:
    case napi_uint8_array:
    case napi_uint8_clamped_array: return 1;
#ifdef NODE_API_HAS_FLOAT16_ARRAY
    case napi_float16_array:
#endif
    case napi_int16_array:
    case napi_uint16_array: return 2;
    case napi_int32_array:
    case napi_uint32_array:
    case napi_float32_array: return 4;
    case napi_float64_array:
    case napi_bigint64_array:
    case napi_biguint64_array: return 8;
    default: return 0;
    }
}

bool uform_js_bytes_from_value(napi_env env, napi_value value, char const *field, void **data, size_t *bytes) {
    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0, elements = 0;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, data, &arraybuffer, &offset) != napi_ok ||
        !uform_js_typedarray_unit_bytes(type))
        return uform_js_throw_field(env, field, "must be a numeric TypedArray");
    *bytes = elements * uform_js_typedarray_unit_bytes(type);
    return true;
}

bool uform_js_array_from_value(napi_env env, napi_value value, char const *field, napi_typedarray_type type,
                               void **data, size_t *elements) {
    napi_typedarray_type found = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &found, elements, data, &arraybuffer, &offset) != napi_ok ||
        (found != type && !(type == napi_uint8_array && found == napi_uint8_clamped_array)))
        return uform_js_throw_field(env, field, "is a TypedArray of another element type");
    return true;
}

/** Hands @p call a heap block of @p bytes it frees once final; @c NULL with an exception pending. */
static void *uform_js_call_own(napi_env env, uform_js_call_t *call, size_t bytes) {
    for (size_t block = 0; block != uform_js_owned_k; ++block) {
        if (call->owned[block]) continue;
        call->owned[block] = malloc(bytes ? bytes : 1);
        if (!call->owned[block]) uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the offsets");
        return call->owned[block];
    }
    uform_js_throw(env, uform_status_internal_k, "The call owns too many blocks");
    return NULL;
}

bool uform_js_offsets_from_value(napi_env env, napi_value value, char const *field, uform_js_call_t *call,
                                 size_t const **offsets, size_t *count) {
    *offsets = NULL;
    *count = 0;
    if (uform_js_value_absent(env, value)) return true;
    void *data = NULL;
    size_t elements = 0;
    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, &data, &arraybuffer, &offset) != napi_ok ||
        (type != napi_biguint64_array && type != napi_uint32_array))
        return uform_js_throw_field(env, field, "must be a BigUint64Array or a Uint32Array");
    if (!elements) return uform_js_throw_field(env, field, "must hold one more offset than entries");
    if (uform_js_typedarray_unit_bytes(type) == sizeof(size_t)) { // as wide as `size_t`, so lent as it is
        *offsets = (size_t const *)data;
        *count = elements - 1;
        return true;
    }
    if (type != napi_uint32_array) return uform_js_throw_field(env, field, "is wider than this platform's size_t");
    size_t *const widened = (size_t *)uform_js_call_own(env, call, elements * sizeof(size_t));
    if (!widened) return false;
    for (size_t position = 0; position != elements; ++position) widened[position] = ((uint32_t const *)data)[position];
    *offsets = widened;
    *count = elements - 1;
    return true;
}

bool uform_js_tape_from_value(napi_env env, napi_value data_value, napi_value offsets_value, char const *field,
                              napi_typedarray_type element_type, uform_js_call_t *call, size_t *count,
                              uform_tape_view_t *tape) {
    void *data = NULL;
    size_t elements = 0;
    if (!uform_js_array_from_value(env, data_value, field, element_type, &data, &elements) ||
        !uform_js_offsets_from_value(env, offsets_value, "offsets", call, &tape->offsets, count))
        return false;
    if (!tape->offsets) return uform_js_throw_field(env, "offsets", "must be a BigUint64Array or a Uint32Array");
    if (tape->offsets[*count] > elements) // the library reads as far as the last offset says
        return uform_js_throw(env, uform_status_shape_mismatch_k, "The offsets reach past the end of the tape");
    tape->data = data;
    tape->scalar = element_type == napi_float32_array ? uform_scalar_f32_k : uform_scalar_u8_k;
    return true;
}

bool uform_js_per_item_from_value(napi_env env, napi_value value, char const *field, size_t count,
                                  uint32_t const **result) {
    void *data = NULL;
    size_t elements = 0;
    *result = NULL;
    if (uform_js_value_absent(env, value)) return true;
    if (!uform_js_array_from_value(env, value, field, napi_uint32_array, &data, &elements)) return false;
    if (elements != count) return uform_js_throw_field(env, field, "must hold one entry per item");
    *result = (uint32_t const *)data;
    return true;
}

/** Calls the JavaScript progress handler in place: @c false from it cancels, and so does a throw, which the call
 *  rethrows once it returns. */
static uform_continuation_t uform_js_progress_relay(size_t completed, size_t total, void *state) {
    uform_js_call_t *const call = (uform_js_call_t *)state;
    napi_env const env = call->env;
    napi_value global = NULL, answer = NULL, exception = NULL;
    napi_value const arguments[2] = {uform_js_number(env, completed), uform_js_number(env, total)};
    if (call->failure) return uform_cancel_k;
    if (napi_get_global(env, &global) != napi_ok ||
        napi_call_function(env, global, call->progress, 2, arguments, &answer) != napi_ok) {
        if (napi_get_and_clear_last_exception(env, &exception) == napi_ok && exception)
            napi_create_reference(env, exception, 1, &call->failure);
        return uform_cancel_k;
    }
    napi_valuetype kind = napi_undefined;
    bool proceed = true;
    if (napi_typeof(env, answer, &kind) == napi_ok && kind == napi_boolean) napi_get_value_bool(env, answer, &proceed);
    return proceed ? uform_continue_k : uform_cancel_k;
}

bool uform_js_progress_from_value(napi_env env, napi_value value, uform_executor_t executor, uform_js_call_t *call,
                                  uform_progress_t *progress) {
    memset(progress, 0, sizeof(*progress));
    if (uform_js_value_absent(env, value)) return true;
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok || kind != napi_function)
        return uform_js_throw_field(env, "progress", "must be a function");
    if (executor)
        return uform_js_throw_field(env, "progress",
                                    "runs on the calling thread, so only a model's own verbs take one, not a receiver");
    call->env = env;
    call->progress = value;
    progress->callback = uform_js_progress_relay;
    progress->state = call;
    return true;
}

napi_value uform_js_number(napi_env env, size_t value) {
    napi_value number = NULL;
    napi_create_double(env, (double)value, &number);
    return number;
}

napi_value uform_js_static_string(napi_env env, char const *text) {
    napi_value string = NULL;
    napi_create_string_utf8(env, text ? text : "", NAPI_AUTO_LENGTH, &string);
    return string;
}

napi_value uform_js_undefined(napi_env env) {
    napi_value undefined = NULL;
    napi_get_undefined(env, &undefined);
    return undefined;
}

napi_value uform_js_object(napi_env env, size_t count, char const *const *names, napi_value const *values) {
    napi_value object = NULL;
    bool built = napi_create_object(env, &object) == napi_ok;
    for (size_t position = 0; built && position != count; ++position)
        built = napi_set_named_property(env, object, names[position], values[position]) == napi_ok;
    if (built) return object;
    uform_js_throw(env, uform_status_internal_k, "Could not publish the outputs");
    return NULL;
}

bool uform_js_arguments(napi_env env, napi_callback_info info, size_t wanted, napi_value *arguments, napi_value *self) {
    size_t arguments_count = wanted;
    if (napi_get_cb_info(env, info, &arguments_count, arguments, self, NULL) != napi_ok)
        return uform_js_throw(env, uform_status_internal_k, "Could not read the arguments");
    for (size_t position = arguments_count; position < wanted; ++position)
        arguments[position] = uform_js_undefined(env);
    return true;
}

#pragma endregion Argument Parsing

#pragma region Classes

uform_js_addon_t *uform_js_addon(napi_env env) {
    uform_js_addon_t *addon = NULL;
    if (napi_get_instance_data(env, (void **)&addon) != napi_ok || !addon) {
        uform_js_throw(env, uform_status_internal_k, "The addon state is missing");
        return NULL;
    }
    return addon;
}

bool uform_js_sealed(napi_env env, napi_callback_info info, napi_value *self) {
    size_t arguments_count = 1;
    napi_value argument = NULL, seal = NULL;
    bool equal = false;
    uform_js_addon_t *const addon = uform_js_addon(env);
    return addon && napi_get_cb_info(env, info, &arguments_count, &argument, self, NULL) == napi_ok &&
           arguments_count == 1 && napi_get_reference_value(env, addon->seal, &seal) == napi_ok &&
           napi_strict_equals(env, argument, seal, &equal) == napi_ok && equal;
}

napi_value uform_js_instantiate(napi_env env, napi_ref constructor) {
    napi_value constructor_value = NULL, seal = NULL, instance = NULL;
    uform_js_addon_t *const addon = uform_js_addon(env);
    if (!addon || napi_get_reference_value(env, constructor, &constructor_value) != napi_ok ||
        napi_get_reference_value(env, addon->seal, &seal) != napi_ok ||
        napi_new_instance(env, constructor_value, 1, &seal, &instance) != napi_ok)
        return NULL;
    return instance;
}

/** A sealed constructor: refused unless one of the addon's own factories calls it with the seal. */
static napi_value uform_js_sealed_construct(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    if (!uform_js_sealed(env, info, &self)) {
        uform_js_throw(env, uform_status_invalid_argument_k, "This class is made by its factories, never directly");
        return NULL;
    }
    return self;
}

/** Gives the instances of @p constructor a @c [Symbol.dispose] calling their @c free, as @c using expects. */
static bool uform_js_class_disposable(napi_env env, napi_value constructor) {
    napi_value global = NULL, symbols = NULL, dispose = NULL, prototype = NULL, function = NULL;
    if (napi_get_global(env, &global) != napi_ok ||
        napi_get_named_property(env, global, "Symbol", &symbols) != napi_ok ||
        napi_get_named_property(env, symbols, "dispose", &dispose) != napi_ok)
        return false;
    if (uform_js_value_absent(env, dispose)) return true; // a runtime without explicit resource management
    return napi_get_named_property(env, constructor, "prototype", &prototype) == napi_ok &&
           napi_get_named_property(env, prototype, "free", &function) == napi_ok &&
           napi_set_property(env, prototype, dispose, function) == napi_ok;
}

napi_value uform_js_class_define(napi_env env, char const *name, size_t count,
                                 napi_property_descriptor const *properties) {
    napi_value defined = NULL;
    if (napi_define_class(env, name, NAPI_AUTO_LENGTH, uform_js_sealed_construct, NULL, count, properties, &defined) !=
            napi_ok ||
        !uform_js_class_disposable(env, defined)) {
        uform_js_throw(env, uform_status_internal_k, "Could not define a class");
        return NULL;
    }
    return defined;
}

void *uform_js_unwrap_instance(napi_env env, napi_value value, napi_ref reference, char const *field,
                               char const *expectation) {
    napi_value constructor = NULL;
    bool is_instance = false;
    void *wrapped = NULL;
    if (napi_get_reference_value(env, reference, &constructor) != napi_ok ||
        napi_instanceof(env, value, constructor, &is_instance) != napi_ok || !is_instance ||
        napi_unwrap(env, value, &wrapped) != napi_ok || !wrapped) {
        uform_js_throw_field(env, field, expectation);
        return NULL;
    }
    return wrapped;
}

void *uform_js_self(napi_env env, napi_callback_info info, size_t wanted, napi_value *arguments, napi_value *self) {
    napi_value receiver = NULL;
    void *wrapped = NULL;
    if (!uform_js_arguments(env, info, wanted, arguments, &receiver)) return NULL;
    if (napi_unwrap(env, receiver, &wrapped) != napi_ok || !wrapped) {
        uform_js_throw(env, uform_status_internal_k, "The receiver is not attached");
        return NULL;
    }
    if (self) *self = receiver;
    return wrapped;
}

#pragma endregion Classes

#pragma region Executor

static void uform_js_executor_finalize(napi_env env, void *data, void *hint) {
    uform_js_executor_t *const executor = (uform_js_executor_t *)data;
    (void)env;
    (void)hint;
    if (!executor) return;
    uform_executor_free(executor->handle);
    free(executor);
}

/** Wraps the handle a factory just made, or throws its outcome; frees the handle on failure. */
static napi_value uform_js_executor_adopt(napi_env env, uform_outcome_t const *outcome, uform_executor_t handle) {
    if (!uform_js_succeeded(env, outcome)) return NULL;
    uform_js_addon_t *const addon = uform_js_addon(env);
    napi_value const instance = addon ? uform_js_instantiate(env, addon->executor_class) : NULL;
    uform_js_executor_t *const executor = (uform_js_executor_t *)calloc(1, sizeof(uform_js_executor_t));
    if (!instance || !executor ||
        napi_wrap(env, instance, executor, uform_js_executor_finalize, NULL, NULL) != napi_ok) {
        free(executor);
        uform_executor_free(handle);
        uform_js_throw(env, uform_status_internal_k, "Could not instantiate the Executor");
        return NULL;
    }
    executor->handle = handle;
    return instance;
}

/** Reads a foreign pointer, crossing as the @c BigInt of its address; absent reads as @c NULL. */
static bool uform_js_pointer_from_value(napi_env env, napi_value value, char const *field, void **pointer) {
    *pointer = NULL;
    if (uform_js_value_absent(env, value)) return true;
    uint64_t address = 0;
    bool lossless = false;
    if (napi_get_value_bigint_uint64(env, value, &address, &lossless) != napi_ok || !lossless)
        return uform_js_throw_field(env, field, "must be a BigInt address");
    *pointer = (void *)(uintptr_t)address;
    return true;
}

/** @c forRam({memoryDomains, cpuClass, threads, deferredLimit}) → host threads over NUMA nodes. */
static napi_value uform_js_executor_for_ram(napi_env env, napi_callback_info info) {
    napi_value config = NULL;
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_executor_t handle = NULL;
    uform_executor_for_ram_t for_ram = {.executor = &handle, .outcome = &outcome};
    int32_t *ordinals = NULL;
    if (!uform_js_arguments(env, info, 1, &config, NULL) ||
        !uform_js_cpu_class_from_value(env, uform_js_property(env, config, "cpuClass"), &for_ram.ram.cpu_class) ||
        !uform_js_size_from_value(env, uform_js_property(env, config, "threads"), "threads", &for_ram.threads) ||
        !uform_js_size_from_value(env, uform_js_property(env, config, "deferredLimit"), "deferredLimit",
                                  &for_ram.deferred_limit) ||
        !uform_js_ordinals_from_value(env, uform_js_property(env, config, "memoryDomains"), "memoryDomains", &ordinals,
                                      &for_ram.ram.memory_domains_count))
        return NULL;
    for_ram.ram.memory_domains = ordinals;
    uform_executor_for_ram(&for_ram);
    free(ordinals);
    return uform_js_executor_adopt(env, &outcome, handle);
}

/** @c onPool(pool, {memoryDomains, cpuClass, deferredLimit}) → an executor borrowing a ForkUnion
 *  pool. */
static napi_value uform_js_executor_on_pool(napi_env env, napi_callback_info info) {
    napi_value arguments[2];
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_executor_t handle = NULL;
    uform_executor_on_pool_t on_pool = {.executor = &handle, .outcome = &outcome};
    int32_t *ordinals = NULL;
    if (!uform_js_arguments(env, info, 2, arguments, NULL) ||
        !uform_js_pointer_from_value(env, arguments[0], "pool", &on_pool.pool) ||
        !uform_js_cpu_class_from_value(env, uform_js_property(env, arguments[1], "cpuClass"), &on_pool.ram.cpu_class) ||
        !uform_js_size_from_value(env, uform_js_property(env, arguments[1], "deferredLimit"), "deferredLimit",
                                  &on_pool.deferred_limit) ||
        !uform_js_ordinals_from_value(env, uform_js_property(env, arguments[1], "memoryDomains"), "memoryDomains",
                                      &ordinals, &on_pool.ram.memory_domains_count))
        return NULL;
    on_pool.ram.memory_domains = ordinals;
    uform_executor_on_pool(&on_pool);
    free(ordinals);
    return uform_js_executor_adopt(env, &outcome, handle);
}

/** @c forVram({devices, gpuClass, memoryDomains, cpuClass, streams, scratchLimit, deferredLimit})
 *  → a stream per device and host decoders over NUMA nodes. */
static napi_value uform_js_executor_for_vram(napi_env env, napi_callback_info info) {
    napi_value config = NULL;
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_executor_t handle = NULL;
    uform_executor_for_vram_t for_vram = {.executor = &handle, .outcome = &outcome};
    int32_t *ordinals = NULL, *memory_domains = NULL;
    if (!uform_js_arguments(env, info, 1, &config, NULL) ||
        !uform_js_gpu_class_from_value(env, uform_js_property(env, config, "gpuClass"), &for_vram.vram.gpu_class) ||
        !uform_js_cpu_class_from_value(env, uform_js_property(env, config, "cpuClass"), &for_vram.ram.cpu_class) ||
        !uform_js_size_from_value(env, uform_js_property(env, config, "scratchLimit"), "scratchLimit",
                                  &for_vram.scratch_limit) ||
        !uform_js_size_from_value(env, uform_js_property(env, config, "deferredLimit"), "deferredLimit",
                                  &for_vram.deferred_limit) ||
        !uform_js_ordinals_from_value(env, uform_js_property(env, config, "devices"), "devices", &ordinals,
                                      &for_vram.vram.devices_count))
        return NULL;
    for_vram.vram.devices = ordinals;
    if (!uform_js_ordinals_from_value(env, uform_js_property(env, config, "memoryDomains"), "memoryDomains",
                                      &memory_domains, &for_vram.ram.memory_domains_count)) {
        free(ordinals);
        return NULL;
    }
    for_vram.ram.memory_domains = memory_domains;

    napi_value const streams_value = uform_js_property(env, config, "streams");
    void **streams = NULL;
    uint32_t streams_count = 0; // the library reads one stream per device, which only this side can check
    if (!uform_js_value_absent(env, streams_value)) {
        bool is_array = false;
        if (napi_is_array(env, streams_value, &is_array) != napi_ok || !is_array ||
            napi_get_array_length(env, streams_value, &streams_count) != napi_ok ||
            streams_count != for_vram.vram.devices_count) {
            free(memory_domains);
            free(ordinals);
            uform_js_throw_field(env, "streams", "must be an array of BigInt addresses, one per device");
            return NULL;
        }
        streams = (void **)calloc(streams_count ? streams_count : 1, sizeof(void *));
        bool read = streams != NULL;
        for (uint32_t position = 0; read && position != streams_count; ++position) {
            napi_value entry = NULL;
            read = napi_get_element(env, streams_value, position, &entry) == napi_ok &&
                   uform_js_pointer_from_value(env, entry, "streams", &streams[position]);
        }
        if (!read) {
            free(streams);
            free(memory_domains);
            free(ordinals);
            if (!streams) uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the streams");
            return NULL;
        }
        for_vram.streams = (void *const *)streams;
    }
    uform_executor_for_vram(&for_vram);
    free(streams);
    free(memory_domains);
    free(ordinals);
    return uform_js_executor_adopt(env, &outcome, handle);
}

/** @c drain(mood) - waits for every call submitted so far, then idles in @p mood; absent parks. */
static napi_value uform_js_executor_drain(napi_env env, napi_callback_info info) {
    napi_value argument = NULL;
    uform_js_executor_t *const executor = (uform_js_executor_t *)uform_js_self(env, info, 1, &argument, NULL);
    uform_mood_t mood = uform_mood_parked_k;
    if (!executor || !uform_js_mood_from_value(env, argument, &mood)) return NULL;
    if (!executor->handle) {
        uform_js_throw(env, uform_status_invalid_argument_k, "The Executor was already freed");
        return NULL;
    }
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_executor_drain_t drain = {.executor = executor->handle, .outcome = &outcome, .mood = mood};
    uform_executor_drain(&drain);
    if (!uform_js_succeeded(env, &outcome)) return NULL;
    return uform_js_undefined(env);
}

/** @c free() - runs every deferred call and joins the threads now; a second call is a no-op. */
static napi_value uform_js_executor_release(napi_env env, napi_callback_info info) {
    uform_js_executor_t *const executor = (uform_js_executor_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!executor) return NULL;
    if (executor->arenas) {
        uform_js_throw(env, uform_status_contended_k, "Free the GPU arenas made on the Executor before it");
        return NULL;
    }
    uform_executor_free(executor->handle);
    executor->handle = NULL;
    return uform_js_undefined(env);
}

/** Unwraps an @c Executor that was not freed yet; @c NULL with an exception pending. */
static uform_js_executor_t *uform_js_executor_live(napi_env env, napi_value value) {
    uform_js_addon_t *const addon = uform_js_addon(env);
    uform_js_executor_t *const executor = addon ? (uform_js_executor_t *)uform_js_unwrap_instance(
                                                      env, value, addon->executor_class, "executor",
                                                      "must be an Executor")
                                                : NULL;
    if (executor && !executor->handle) { // a freed executor would read as NULL, silently running on this thread
        uform_js_throw(env, uform_status_invalid_argument_k, "The Executor was already freed");
        return NULL;
    }
    return executor;
}

bool uform_js_executor_from_value(napi_env env, napi_value value, uform_executor_t *handle) {
    *handle = NULL;
    if (uform_js_value_absent(env, value)) return true;
    uform_js_executor_t const *const executor = uform_js_executor_live(env, value);
    if (!executor) return false;
    *handle = executor->handle;
    return true;
}

napi_value uform_js_executor_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"drain", NULL, uform_js_executor_drain, NULL, NULL, NULL, napi_default_method, NULL},
        {"free", NULL, uform_js_executor_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"forRam", NULL, uform_js_executor_for_ram, NULL, NULL, NULL, napi_static, NULL},
        {"forVram", NULL, uform_js_executor_for_vram, NULL, NULL, NULL, napi_static, NULL},
        {"onPool", NULL, uform_js_executor_on_pool, NULL, NULL, NULL, napi_static, NULL},
    };
    return uform_js_class_define(env, "Executor", sizeof(properties) / sizeof(properties[0]), properties);
}

#pragma endregion Executor

#pragma region Arena

/** Forgets every view handed out so far, detaching the ones still alive when @p detach asks. */
static void uform_js_arena_forget(napi_env env, uform_js_arena_t *arena, bool detach) {
    for (size_t view_index = 0; view_index != arena->views_count; ++view_index) {
        napi_value buffer = NULL;
        if (detach && napi_get_reference_value(env, arena->views[view_index], &buffer) == napi_ok && buffer)
            napi_detach_arraybuffer(env, buffer);
        napi_delete_reference(env, arena->views[view_index]);
    }
    arena->views_count = 0;
}

/** Lets go of the executor a GPU arena held, once the arena's memory is freed. */
static void uform_js_arena_unbind(napi_env env, uform_js_arena_t *arena) {
    if (!arena->executor) return;
    --arena->executor->arenas;
    napi_delete_reference(env, arena->executor_reference);
    arena->executor = NULL;
}

static void uform_js_arena_finalize(napi_env env, void *data, void *hint) {
    uform_js_arena_t *const arena = (uform_js_arena_t *)data;
    (void)hint;
    if (!arena) return;
    uform_js_arena_forget(env, arena, false);
    free(arena->views);
    uform_arena_free(arena->handle);
    uform_js_arena_unbind(env, arena);
    free(arena);
}

/** Where the outputs of @p arena live, as the library reports it; @c uform_residency_unknown_k
 *  with an exception pending when it refuses. */
static uform_residency_t uform_js_arena_described(napi_env env, uform_js_arena_t const *arena) {
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_arena_description_t description = {uform_residency_unknown_k, -1};
    uform_arena_describe_t describe = {.arena = arena->handle, .outcome = &outcome, .description = &description};
    uform_arena_describe(&describe);
    return uform_js_succeeded(env, &outcome) ? description.residency : uform_residency_unknown_k;
}

/** @c Arena(executor) - empty memory where @p executor computes; host pages for the calling thread when absent. */
static napi_value uform_js_arena_construct(napi_env env, napi_callback_info info) {
    napi_value self = NULL, argument = NULL;
    if (!uform_js_arguments(env, info, 1, &argument, &self)) return NULL;
    uform_js_executor_t *executor = NULL;
    if (!uform_js_value_absent(env, argument) && !(executor = uform_js_executor_live(env, argument))) return NULL;
    uform_js_arena_t *const arena = (uform_js_arena_t *)calloc(1, sizeof(uform_js_arena_t));
    if (!arena) {
        uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the Arena");
        return NULL;
    }
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    uform_arena_init_t init = {.arena = &arena->handle,
                               .executor = executor ? executor->handle : NULL,
                               .outcome = &outcome};
    uform_arena_init(&init);
    if (!uform_js_succeeded(env, &outcome)) {
        free(arena);
        return NULL;
    }
    uform_residency_t const residency = uform_js_arena_described(env, arena);
    if (residency == uform_residency_unknown_k) {
        uform_arena_free(arena->handle);
        free(arena);
        return NULL;
    }
    if (residency == uform_residency_vram_k) { // a GPU arena frees on its executor's first stream
        if (napi_create_reference(env, argument, 1, &arena->executor_reference) != napi_ok) {
            uform_arena_free(arena->handle);
            free(arena);
            uform_js_throw(env, uform_status_internal_k, "Could not hold the Executor");
            return NULL;
        }
        arena->executor = executor;
        ++executor->arenas;
    }
    if (napi_wrap(env, self, arena, uform_js_arena_finalize, NULL, NULL) != napi_ok) {
        uform_arena_free(arena->handle);
        uform_js_arena_unbind(env, arena);
        free(arena);
        uform_js_throw(env, uform_status_internal_k, "Could not attach the Arena");
        return NULL;
    }
    return self;
}

/** @c free() - releases the memory now, once every deferred call writing it is final, detaching every view still
 *  held into it; a second call is a no-op. */
static napi_value uform_js_arena_release(napi_env env, napi_callback_info info) {
    uform_js_arena_t *const arena = (uform_js_arena_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!arena) return NULL;
    if (!arena->handle) return uform_js_undefined(env);
    uform_arena_free(arena->handle);
    uform_js_arena_forget(env, arena, true);
    uform_js_arena_unbind(env, arena);
    arena->handle = NULL;
    arena->holder = NULL;
    return uform_js_undefined(env);
}

bool uform_js_arena_from_value(napi_env env, napi_value value, uform_js_call_t *call, uform_arena_t *handle) {
    uform_js_addon_t *const addon = uform_js_addon(env);
    if (!addon) return false;
    uform_js_arena_t *const arena = (uform_js_arena_t *)uform_js_unwrap_instance(env, value, addon->arena_class,
                                                                                 "arena", "must be an Arena");
    if (!arena) return false;
    if (!arena->handle) return uform_js_throw(env, uform_status_invalid_argument_k, "The Arena was already freed");
    call->arena = arena;
    *handle = arena->handle;
    return true;
}

/** Views @p buffer as @p elements of @p type; half-precision rows read as their bits on a runtime without
 *  @c Float16Array, which a build against newer headers can still be loaded into. */
static napi_status uform_js_typedarray(napi_env env, napi_typedarray_type type, size_t elements, napi_value buffer,
                                       napi_value *array) {
    napi_status const created = napi_create_typedarray(env, type, elements, buffer, 0, array);
#ifdef NODE_API_HAS_FLOAT16_ARRAY
    if (created != napi_ok && type == napi_float16_array)
        return napi_create_typedarray(env, napi_uint16_array, elements, buffer, 0, array);
#endif
    return created;
}

napi_value uform_js_arena_view(napi_env env, napi_value arena_value, napi_typedarray_type type, void const *data,
                               size_t bytes, size_t elements) {
    napi_value buffer = NULL, array = NULL;
    uform_js_arena_t *arena = NULL;
    if (napi_unwrap(env, arena_value, (void **)&arena) != napi_ok || !arena) {
        uform_js_throw(env, uform_status_internal_k, "The Arena is not attached");
        return NULL;
    }
    uform_residency_t const residency = uform_js_arena_described(env, arena);
    if (residency == uform_residency_unknown_k) return NULL;
    if (residency == uform_residency_vram_k) {
        napi_get_null(env, &array);
        return array;
    }
    if (arena->views_count == arena->views_capacity) {
        size_t const capacity = arena->views_capacity ? arena->views_capacity * 2 : 4;
        napi_ref *const views = (napi_ref *)realloc(arena->views, capacity * sizeof(napi_ref));
        if (!views) {
            uform_js_throw(env, uform_status_out_of_memory_k, "Could not track the arena view");
            return NULL;
        }
        arena->views = views;
        arena->views_capacity = capacity;
    }
    napi_property_descriptor const owner = {"arena", NULL, NULL, NULL, NULL, arena_value, napi_default, NULL};
    bool const empty = !data || !bytes;
    napi_status const created = empty ? napi_create_arraybuffer(env, 0, NULL, &buffer)
                                      : napi_create_external_arraybuffer(env, (void *)data, bytes, NULL, NULL, &buffer);
    if (created != napi_ok || napi_define_properties(env, buffer, 1, &owner) != napi_ok ||
        uform_js_typedarray(env, type, empty ? 0 : elements, buffer, &array) != napi_ok ||
        napi_create_reference(env, buffer, 0, &arena->views[arena->views_count]) != napi_ok) {
        uform_js_throw(env, uform_status_internal_k, "Could not view the arena");
        return NULL;
    }
    ++arena->views_count;
    return array;
}

/** The @c TypedArray a scalar reads back as; kinds JavaScript spells no element for stay bytes. */
static napi_typedarray_type uform_js_typedarray_of(uform_scalar_t scalar, size_t *unit_bytes) {
    switch (scalar) {
    case uform_scalar_f64_k: *unit_bytes = 8; return napi_float64_array;
    case uform_scalar_f32_k: *unit_bytes = 4; return napi_float32_array;
#ifdef NODE_API_HAS_FLOAT16_ARRAY
    case uform_scalar_f16_k: *unit_bytes = 2; return napi_float16_array;
#else
    case uform_scalar_f16_k: *unit_bytes = 2; return napi_uint16_array;
#endif
    case uform_scalar_bf16_k: *unit_bytes = 2; return napi_uint16_array;
    case uform_scalar_i8_k: *unit_bytes = 1; return napi_int8_array;
    default: *unit_bytes = 1; return napi_uint8_array;
    }
}

napi_value uform_js_rows_object(napi_env env, napi_value arena, uform_rows_view_t const *rows, size_t count) {
    size_t unit_bytes = 1;
    napi_typedarray_type const type = uform_js_typedarray_of(rows->scalar, &unit_bytes);
    size_t const bytes = count * rows->stride;
    napi_value const data = uform_js_arena_view(env, arena, type, rows->data, bytes, bytes / unit_bytes);
    if (!data) return NULL;
    static char const *const names[] = {"data", "rows", "columns", "stride", "scalar"};
    napi_value const values[] = {data, uform_js_number(env, count), uform_js_number(env, rows->columns),
                                 uform_js_number(env, rows->stride),
                                 uform_js_static_string(env, uform_scalar_name(rows->scalar))};
    return uform_js_object(env, sizeof(names) / sizeof(names[0]), names, values);
}

napi_value uform_js_offsets_view(napi_env env, napi_value arena, size_t const *offsets, size_t count) {
    bool const wide = sizeof(size_t) == 8;
    return uform_js_arena_view(env, arena, wide ? napi_biguint64_array : napi_uint32_array, offsets,
                               (count + 1) * sizeof(size_t), count + 1);
}

/** @c residency → "ram" or "vram", where the executor that made it computes. */
static napi_value uform_js_arena_residency(napi_env env, napi_callback_info info) {
    uform_js_arena_t const *const arena = (uform_js_arena_t const *)uform_js_self(env, info, 0, NULL, NULL);
    if (!arena) return NULL;
    uform_residency_t const residency = uform_js_arena_described(env, arena);
    if (residency == uform_residency_unknown_k) return NULL;
    return uform_js_static_string(env, residency == uform_residency_vram_k ? "vram" : "ram");
}

napi_value uform_js_arena_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"residency", NULL, NULL, uform_js_arena_residency, NULL, NULL, napi_default, NULL},
        {"free", NULL, uform_js_arena_release, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value arena_class = NULL;
    if (napi_define_class(env, "Arena", NAPI_AUTO_LENGTH, uform_js_arena_construct, NULL,
                          sizeof(properties) / sizeof(properties[0]), properties, &arena_class) != napi_ok ||
        !uform_js_class_disposable(env, arena_class)) {
        uform_js_throw(env, uform_status_internal_k, "Could not define the Arena class");
        return NULL;
    }
    return arena_class;
}

#pragma endregion Arena

#pragma region Deferred Calls and Library Probes

/** Drops what a final call holds and owns, keeping its outcome readable; safe to repeat. */
static void uform_js_call_release(napi_env env, uform_js_call_t *call) {
    if (call->failure) napi_delete_reference(env, call->failure);
    call->failure = NULL;
    for (size_t slot = 0; slot != uform_js_slots_k; ++slot) {
        if (call->held[slot]) napi_delete_reference(env, call->held[slot]);
        call->held[slot] = NULL;
    }
    for (size_t block = 0; block != uform_js_owned_k; ++block) {
        free(call->owned[block]);
        call->owned[block] = NULL;
    }
}

/**
 *  @brief Frees a deferred call once its wrapper is collected.
 *
 *  The wrapper stays reachable until its call is final - the TypeScript @c Deferred re-registers it with the
 *  collector until then - so an unfinished call here is a contract breach, and it is left to leak rather than be
 *  freed under the library's feet.
 */
static void uform_js_deferred_finalize(napi_env env, void *data, void *hint) {
    uform_js_call_t *const call = (uform_js_call_t *)data;
    (void)hint;
    if (!call || uform_outcome_status(&call->outcome) == uform_status_deferred_k) return;
    uform_js_call_release(env, call);
    free(call);
}

/** @c status() → the canonical name of the call's status, @c deferred until it is final. */
static napi_value uform_js_deferred_status(napi_env env, napi_callback_info info) {
    uform_js_call_t *const call = (uform_js_call_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!call) return NULL;
    return uform_js_static_string(env, uform_status_name(uform_outcome_status(&call->outcome)));
}

/** An outcome wait handed to the libuv pool, which a promise resolves once it returns. */
typedef struct uform_js_settling_t {

    /** The work item the pool runs. */
    napi_async_work work;

    /** The promise the loop resolves once the wait returns. */
    napi_deferred promise;

    /** A strong reference keeping the call alive through the wait. */
    napi_ref wrapper;

    /** The outcome the pool waits on. */
    uform_outcome_t *outcome;
} uform_js_settling_t;

static void uform_js_settling_execute(napi_env env, void *data) {
    (void)env;
    uform_outcome_wait(((uform_js_settling_t *)data)->outcome);
}

static void uform_js_settling_complete(napi_env env, napi_status status, void *data) {
    uform_js_settling_t *const settling = (uform_js_settling_t *)data;
    (void)status;
    napi_resolve_deferred(env, settling->promise, uform_js_undefined(env));
    napi_delete_reference(env, settling->wrapper);
    napi_delete_async_work(env, settling->work);
    free(settling);
}

/** @c wait() - blocks the calling thread until the call is final. */
static napi_value uform_js_deferred_wait(napi_env env, napi_callback_info info) {
    uform_js_call_t *const call = (uform_js_call_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!call) return NULL;
    uform_outcome_wait(&call->outcome);
    return uform_js_undefined(env);
}

/** @c settled() → a promise resolved once the call is final, its wait run on the libuv pool. */
static napi_value uform_js_deferred_settled(napi_env env, napi_callback_info info) {
    napi_value self = NULL, promise = NULL, name = NULL;
    uform_js_call_t *const call = (uform_js_call_t *)uform_js_self(env, info, 0, NULL, &self);
    if (!call) return NULL;
    uform_js_settling_t *const settling = (uform_js_settling_t *)calloc(1, sizeof(uform_js_settling_t));
    if (!settling) {
        uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the wait");
        return NULL;
    }
    settling->outcome = &call->outcome;
    if (napi_create_string_utf8(env, "uform:deferred", NAPI_AUTO_LENGTH, &name) != napi_ok ||
        napi_create_reference(env, self, 1, &settling->wrapper) != napi_ok ||
        napi_create_async_work(env, NULL, name, uform_js_settling_execute, uform_js_settling_complete, settling,
                               &settling->work) != napi_ok ||
        napi_create_promise(env, &settling->promise, &promise) != napi_ok ||
        napi_queue_async_work(env, settling->work) != napi_ok) {
        if (settling->work) napi_delete_async_work(env, settling->work);
        if (settling->wrapper) napi_delete_reference(env, settling->wrapper);
        free(settling);
        uform_js_throw(env, uform_status_internal_k, "Could not queue the wait");
        return NULL;
    }
    return promise;
}

/** Builds a final call's answer, or throws its handler's exception, failure or stale arena. */
static napi_value uform_js_call_finish(napi_env env, uform_js_call_t *call, napi_value const *slots) {
    napi_value failure = NULL;
    if (call->failure && napi_get_reference_value(env, call->failure, &failure) == napi_ok) {
        napi_throw(env, failure);
        return NULL;
    }
    if (!uform_js_succeeded(env, &call->outcome)) return NULL;
    if (call->arena && call->arena->holder != call) {
        uform_js_throw(env, uform_status_invalid_argument_k,
                       "The arena was rewound or freed since this call wrote into it");
        return NULL;
    }
    return call->finish(env, call, slots);
}

/** @c result() → the verb's answer once the call is final, or its failure thrown; answers once. */
static napi_value uform_js_deferred_result(napi_env env, napi_callback_info info) {
    uform_js_call_t *const call = (uform_js_call_t *)uform_js_self(env, info, 0, NULL, NULL);
    if (!call) return NULL;
    if (uform_outcome_status(&call->outcome) == uform_status_deferred_k) {
        uform_js_throw(env, uform_status_internal_k, "The call is not final yet");
        return NULL;
    }
    if (!call->finish) {
        uform_js_throw(env, uform_status_internal_k, "The call already answered");
        return NULL;
    }
    napi_value slots[uform_js_slots_k] = {NULL};
    for (size_t slot = 0; slot != uform_js_slots_k; ++slot)
        if (call->held[slot]) napi_get_reference_value(env, call->held[slot], &slots[slot]);
    napi_value const result = uform_js_call_finish(env, call, slots);
    call->finish = NULL;
    uform_js_call_release(env, call);
    return result;
}

napi_value uform_js_deferred_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"status", NULL, uform_js_deferred_status, NULL, NULL, NULL, napi_default_method, NULL},
        {"wait", NULL, uform_js_deferred_wait, NULL, NULL, NULL, napi_default_method, NULL},
        {"settled", NULL, uform_js_deferred_settled, NULL, NULL, NULL, napi_default_method, NULL},
        {"result", NULL, uform_js_deferred_result, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value deferred_class = NULL;
    if (napi_define_class(env, "CompiledDeferred", NAPI_AUTO_LENGTH, uform_js_sealed_construct, NULL,
                          sizeof(properties) / sizeof(properties[0]), properties, &deferred_class) != napi_ok) {
        uform_js_throw(env, uform_status_internal_k, "Could not define the CompiledDeferred class");
        return NULL;
    }
    return deferred_class;
}

uform_js_call_t *uform_js_call_open(napi_env env, uform_completion_t completion, uform_js_call_t *local, size_t bytes) {
    memset(local, 0, bytes);
    if (completion == uform_completion_blocking_k) return local;
    uform_js_call_t *const call = (uform_js_call_t *)calloc(1, bytes);
    if (!call) uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the deferred call");
    return call;
}

uform_js_call_t *uform_js_call_open_on(napi_env env, napi_value const *arguments, uform_js_call_t *local, size_t bytes,
                                       uform_executor_t *executor, uform_completion_t *completion) {
    if (!uform_js_executor_from_value(env, arguments[0], executor) ||
        !uform_js_completion_from_value(env, arguments[1], completion))
        return NULL;
    if (*completion == uform_completion_deferred_k && !*executor) {
        uform_js_throw_field(env, "executor", "must be given to a deferred call");
        return NULL;
    }
    return uform_js_call_open(env, *completion, local, bytes);
}

bool uform_js_call_hold(napi_env env, uform_js_call_t *call, uform_js_call_t const *local, napi_value const *slots,
                        napi_value *deferred) {
    if (call == local) return true;
    uform_js_addon_t *const addon = uform_js_addon(env);
    napi_value const instance = addon ? uform_js_instantiate(env, addon->deferred_class) : NULL;
    bool held = instance != NULL;
    for (size_t slot = 0; held && slot != uform_js_slots_k; ++slot)
        if (!uform_js_value_absent(env, slots[slot]))
            held = napi_create_reference(env, slots[slot], 1, &call->held[slot]) == napi_ok;
    if (held && napi_wrap(env, instance, call, uform_js_deferred_finalize, NULL, NULL) == napi_ok) {
        *deferred = instance;
        return true;
    }
    uform_js_call_release(env, call);
    free(call);
    uform_js_throw(env, uform_status_internal_k, "Could not hold the deferred call");
    return false;
}

napi_value uform_js_call_answer(napi_env env, uform_js_call_t *call, uform_js_call_t const *local,
                                napi_value const *slots, napi_value deferred) {
    uform_js_arena_t *const arena = call->arena;
    if (arena && uform_outcome_status(&call->outcome) != uform_status_contended_k) {
        uform_js_arena_forget(env, arena, true);
        arena->holder = call;
    }
    if (call != local) return deferred;
    napi_value const result = uform_js_call_finish(env, call, slots);
    uform_js_call_release(env, call);
    return result;
}

void uform_js_call_abandon(uform_js_call_t *call, uform_js_call_t const *local) {
    if (!call) return;
    for (size_t block = 0; block != uform_js_owned_k; ++block) free(call->owned[block]);
    if (call != local) free(call);
}

napi_value uform_js_version(napi_env env, napi_callback_info info) {
    (void)info;
    return uform_js_static_string(env, uform_version_name());
}

napi_value uform_js_capabilities_compiled(napi_env env, napi_callback_info info) {
    (void)info;
    return uform_js_static_string(env, uform_capabilities_compiled());
}

napi_value uform_js_capabilities_available(napi_env env, napi_callback_info info) {
    (void)info;
    return uform_js_static_string(env, uform_capabilities_available());
}

/** @c gpuDevices() → @c [{vendor, vendorOrdinal, gpuClass, computeUnits, memoryBytes, name}],
 *  counted first and then filled. */
napi_value uform_js_gpu_devices(napi_env env, napi_callback_info info) {
    (void)info;
    uform_outcome_t outcome = {uform_status_success_k, NULL};
    size_t count = 0;
    uform_gpu_devices(&(uform_gpu_devices_t) {.outcome = &outcome, .devices_count = &count});
    if (!uform_js_succeeded(env, &outcome)) return NULL;
    uform_gpu_device_t *const devices = (uform_gpu_device_t *)calloc(count ? count : 1, sizeof(uform_gpu_device_t));
    if (!devices) {
        uform_js_throw(env, uform_status_out_of_memory_k, "Could not allocate the device listing");
        return NULL;
    }
    size_t const capacity = count;
    uform_gpu_devices(&(uform_gpu_devices_t) {.outcome = &outcome,
                                              .devices = devices,
                                              .devices_capacity = capacity,
                                              .devices_count = &count});
    napi_value listing = NULL;
    bool listed = uform_js_succeeded(env, &outcome) && napi_create_array(env, &listing) == napi_ok;
    if (count > capacity) count = capacity;
    static char const *const names[] = {"vendor", "vendorOrdinal", "gpuClass", "computeUnits", "memoryBytes", "name"};
    for (size_t position = 0; listed && position != count; ++position) {
        uform_gpu_device_t const *const device = &devices[position];
        napi_value const values[] = {uform_js_number(env, (size_t)device->vendor),
                                     uform_js_number(env, device->vendor_ordinal),
                                     uform_js_class_name(env, uform_gpu_class_name(device->gpu_class)),
                                     uform_js_number(env, device->compute_units),
                                     uform_js_number(env, (size_t)device->memory_bytes),
                                     uform_js_static_string(env, device->name)};
        napi_value const entry = uform_js_object(env, sizeof(names) / sizeof(names[0]), names, values);
        listed = entry && napi_set_element(env, listing, (uint32_t)position, entry) == napi_ok;
    }
    free(devices);
    return listed ? listing : NULL;
}

#pragma endregion Deferred Calls and Library Probes
