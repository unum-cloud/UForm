/**
 *  @file python/types.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The CPython vocabulary every header shares: failures, enums, keyword-built structures,
 *      calls, receivers, executors, arenas and the tapes inputs gather into.
 *
 *  Nothing here belongs to one header of the C interface: an @c Arena an encoding writes into is
 *  the very object a training step accepts, because there is one definition and one address.
 */
#include "lib.h"

#pragma region Failures

PyObject* ErrorType = NULL;

/** The class each failing status raises, indexed by status; success and deferred never raise. */
static PyObject* status_error_types[uform_status_invalid_argument_k + 1];

/** One failing status, the dotted name of the class it raises, the built-in it also subclasses,
 *  and its doc. */
static struct {
    uform_status_t status;
    char const* name;

    /** NULL when no built-in names the fault. */
    PyObject** builtin;
    char const* doc;
} const status_errors[] = {
    {uform_status_out_of_memory_k, "uform.OutOfMemoryError", &PyExc_MemoryError,
     "Device, pinned, or host memory could not be allocated, or the scratch ceiling is too low."},
    {uform_status_shape_mismatch_k, "uform.ShapeMismatchError", &PyExc_ValueError,
     "Item counts, a pixel shape, or a batch envelope disagree with the call or the model."},
    {uform_status_unsupported_layout_k, "uform.UnsupportedLayoutError", &PyExc_ValueError,
     "A stride or packing the model cannot read."},
    {uform_status_unknown_kind_k, "uform.UnknownKindError", &PyExc_ValueError,
     "A scalar, modality, or name this build does not know."},
    {uform_status_io_k, "uform.InputOutputError", &PyExc_OSError,
     "A checkpoint file could not be opened, read, or written."},
    {uform_status_device_k, "uform.DeviceError", NULL,
     "The driver refused or the device faulted; after a fault the executor refuses every later call."},
    {uform_status_malformed_k, "uform.MalformedError", &PyExc_ValueError,
     "Supplied bytes could not be decoded: a checkpoint, an image, a clip, or a recording."},
    {uform_status_internal_k, "uform.InternalError", NULL,
     "The library reported neither success nor a failure - a defect."},
    {uform_status_overflow_k, "uform.CapacityOverflowError", &PyExc_OverflowError,
     "A count would pass the largest value its type holds."},
    {uform_status_uninitialized_k, "uform.UninitializedError", NULL,
     "The handle the call operates on is missing or closed, refused before any other check."},
    {uform_status_config_mismatch_k, "uform.ConfigMismatchError", &PyExc_ValueError,
     "The request contradicts a frozen configuration, like resuming a checkpoint of another geometry."},
    {uform_status_contended_k, "uform.ContendedError", &PyExc_BufferError,
     "Another call holds the executor or the arena, an export of the arena is alive, or an arena of the executor is."},
    {uform_status_cancelled_k, "uform.CancelledError", NULL,
     "A progress handler stopped the call, which published nothing."},
    {uform_status_unsupported_k, "uform.UnsupportedError", &PyExc_NotImplementedError,
     "The verb is declared, but this residency or model does not implement it."},
    {uform_status_invalid_argument_k, "uform.InvalidArgumentError", &PyExc_ValueError,
     "An argument breaks the verb's contract."},
};

int uform_py_raise(uform_status_t status, char const* format, ...) {
    PyErr_Clear(); // a status this specific supersedes whatever the caller was carrying
    size_t const types_count = sizeof(status_error_types) / sizeof(*status_error_types);
    PyObject* type = (size_t)status < types_count ? status_error_types[status] : NULL;
    if (!type) {
        status = uform_status_internal_k;
        type = status_error_types[status];
    }
    va_list arguments;
    va_start(arguments, format);
    PyObject* const message = PyUnicode_FromFormatV(format ? format : "", arguments);
    va_end(arguments);
    if (!message) return -1;

    PyObject* const instance = PyObject_CallOneArg(type, message);
    Py_DECREF(message);
    if (!instance) return -1;
    PyObject* const member = uform_py_enum_to_object(&uform_py_status, (int)status);
    int const attached = member ? PyObject_SetAttrString(instance, "status", member) : -1;
    Py_XDECREF(member);
    if (attached != 0) {
        Py_DECREF(instance);
        return -1;
    }
    PyErr_SetRaisedException(instance);
    return -1;
}

int uform_py_raise_on_outcome(uform_outcome_t const* outcome) {
    if (outcome->status == uform_status_success_k) return 0;
    return uform_py_raise(outcome->status, "%s", outcome->message ? outcome->message : "");
}

int uform_py_dict_steal_item(PyObject* dict, char const* key, PyObject* value) {
    if (!value) return -1;
    int const stored = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return stored;
}

static char const doc_Error[] =                                                                //
    "Error(message)\n"                                                                         //
    "\n"                                                                                       //
    "The base of every failure the library reports. Each status shared with the C API\n"       //
    "raises a subclass of its own, such as `ShapeMismatchError`, so a caller catches one\n"    //
    "fault by its type or every fault here. A subclass also derives the built-in that names\n" //
    "the same fault, so `except ValueError` still catches a `ShapeMismatchError`. A failure\n" //
    "caught before the boundary raises the class the library would have raised for it.\n"      //
    "\n"                                                                                       //
    "Attributes:\n"                                                                            //
    "  status (Status): The status the raised subclass mirrors, such as\n"                     //
    "    `Status.SHAPE_MISMATCH`.\n"                                                           //
    "  args (tuple): The detail the status cannot express, as on any exception.";

int uform_py_register_error_types(PyObject* module) {
    ErrorType = PyErr_NewExceptionWithDoc("uform.Error", (char*)doc_Error, PyExc_RuntimeError, NULL);
    if (!ErrorType || PyModule_AddObjectRef(module, "Error", ErrorType) < 0) return -1;
    for (size_t row = 0; row != sizeof(status_errors) / sizeof(*status_errors); ++row) {
        PyObject* const bases = status_errors[row].builtin ? PyTuple_Pack(2, ErrorType, *status_errors[row].builtin)
                                                           : Py_NewRef(ErrorType);
        if (!bases) return -1;
        PyObject* const type = PyErr_NewExceptionWithDoc(status_errors[row].name, (char*)status_errors[row].doc, bases,
                                                         NULL);
        Py_DECREF(bases);
        if (!type) return -1;
        status_error_types[status_errors[row].status] = type;
        if (PyModule_AddObjectRef(module, status_errors[row].name + sizeof("uform.") - 1, type) < 0) return -1;
    }
    return 0;
}

#pragma endregion Failures

#pragma region Enums

static char const* spell_status(int value) { return uform_status_name((uform_status_t)value); }
static char const* spell_scalar(int value) { return uform_scalar_name((uform_scalar_t)value); }
static char const* spell_mood(int value) { return uform_mood_name((uform_mood_t)value); }
static char const* spell_cpu_class(int value) { return uform_cpu_class_name((uform_cpu_class_t)value); }
static char const* spell_gpu_class(int value) { return uform_gpu_class_name((uform_gpu_class_t)value); }

static char const* spell_gpu_vendor(int value) { return uform_gpu_vendor_name((uform_gpu_vendor_t)value); }

uform_py_enum_t uform_py_status = {.name = "Status",
                                   .base = "IntEnum",
                                   .first = uform_status_success_k,
                                   .last = uform_status_invalid_argument_k,
                                   .spell = spell_status};
uform_py_enum_t uform_py_scalar = {.name = "Scalar",
                                   .base = "IntEnum",
                                   .first = uform_scalar_f64_k,
                                   .last = uform_scalar_u1x8_k,
                                   .spell = spell_scalar};
uform_py_enum_t uform_py_mood = {.name = "Mood",
                                 .base = "IntEnum",
                                 .first = uform_mood_parked_k,
                                 .last = uform_mood_spinning_k,
                                 .spell = spell_mood};

/** Classes fit a byte, and the registry values this build lacks spell no member. */
uform_py_enum_t uform_py_cpu_class = {.name = "CPUClass",
                                      .base = "IntEnum",
                                      .first = uform_cpu_class_serial_k,
                                      .last = UINT8_MAX,
                                      .spell = spell_cpu_class};
uform_py_enum_t uform_py_gpu_class = {.name = "GPUClass",
                                      .base = "IntEnum",
                                      .first = uform_gpu_class_nvidia_ampere_k,
                                      .last = UINT8_MAX,
                                      .spell = spell_gpu_class};
uform_py_enum_t uform_py_gpu_vendor = {.name = "GPUVendor",
                                       .base = "IntEnum",
                                       .first = uform_gpu_vendor_nvidia_k,
                                       .last = uform_gpu_vendor_apple_k,
                                       .spell = spell_gpu_vendor};

/** The spelling of @p value, or NULL when no member holds it. */
static char const* enum_spelling(uform_py_enum_t const* mirrored, long value) {
    if (value < mirrored->first || value > mirrored->last) return NULL;
    char const* const spelled = mirrored->spell((int)value);
    if (!spelled || strcmp(spelled, "unknown") == 0 || strcmp(spelled, "unrecognized") == 0) return NULL;
    return spelled;
}

/** Builds the class of @p mirrored by calling its base from @p enum_module with @p keywords,
 *  in value order. */
static int enum_register(PyObject* module, PyObject* enum_module, PyObject* keywords, uform_py_enum_t* mirrored) {
    PyObject* members = PyDict_New();
    for (long value = mirrored->first; members && value <= mirrored->last; ++value) {
        char const* const spelled = enum_spelling(mirrored, value);
        if (!spelled) continue;
        char member_name[32];
        size_t length = 0;
        for (; spelled[length] && length + 1 != sizeof(member_name); ++length) {
            char const letter = spelled[length];
            member_name[length] = letter == '-'                    ? '_'
                                  : letter >= 'a' && letter <= 'z' ? (char)(letter - 'a' + 'A')
                                                                   : letter;
        }
        member_name[length] = '\0';
        if (uform_py_dict_steal_item(members, member_name, PyLong_FromLong(value)) != 0) Py_CLEAR(members);
    }
    PyObject* const base = members ? PyObject_GetAttrString(enum_module, mirrored->base) : NULL;
    PyObject* const name = base ? PyUnicode_FromString(mirrored->name) : NULL;
    PyObject* const arguments = name ? PyTuple_Pack(2, name, members) : NULL;
    mirrored->type = arguments ? PyObject_Call(base, arguments, keywords) : NULL;
    Py_XDECREF(arguments);
    Py_XDECREF(name);
    Py_XDECREF(base);
    Py_XDECREF(members);
    return mirrored->type ? PyModule_AddObjectRef(module, mirrored->name, mirrored->type) : -1;
}

int uform_py_register_enums(PyObject* module) {
    uform_py_enum_t* const mirrored[] = {
        &uform_py_status,       &uform_py_scalar,     &uform_py_mood,           &uform_py_cpu_class,
        &uform_py_gpu_class,    &uform_py_gpu_vendor, &uform_py_outputs,        &uform_py_modality,
        &uform_py_objective,    &uform_py_negatives,  &uform_py_pair_loss_form, &uform_py_late_interaction_side,
        &uform_py_multimem_tier};
    PyObject* const enum_module = PyImport_ImportModule("enum");
    PyObject* keywords = enum_module ? PyDict_New() : NULL;
    if (keywords && uform_py_dict_steal_item(keywords, "module", PyUnicode_FromString("uform")) != 0)
        Py_CLEAR(keywords);
    int registered = keywords ? 0 : -1;
    for (size_t row = 0; registered == 0 && row != sizeof(mirrored) / sizeof(*mirrored); ++row)
        registered = enum_register(module, enum_module, keywords, mirrored[row]);
    Py_XDECREF(keywords);
    Py_XDECREF(enum_module);
    return registered;
}

int uform_py_enum_from_object(PyObject* object, char const* field, uform_py_enum_t const* mirrored, int* value) {
    if (!PyLong_Check(object) || PyBool_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' option takes a uform.%s, got %S", field, mirrored->name,
                     Py_TYPE(object));
        return -1;
    }
    int overflow = 0;
    long const read = PyLong_AsLongAndOverflow(object, &overflow);
    if (read == -1 && PyErr_Occurred()) return -1;
    if (!overflow && enum_spelling(mirrored, read)) {
        *value = (int)read;
        return 0;
    }
    return uform_py_raise(uform_status_unknown_kind_k, "the '%s' option takes a uform.%s, and %R is none", field,
                          mirrored->name, object);
}

PyObject* uform_py_enum_to_object(uform_py_enum_t const* mirrored, int value) {
    PyObject* const integer = PyLong_FromLong(value);
    PyObject* const member = integer ? PyObject_CallOneArg(mirrored->type, integer) : NULL;
    Py_XDECREF(integer);
    return member;
}

int uform_py_class_from_object(PyObject* object, char const* field, uform_py_enum_t const* classes, int* value) {
    *value = 0;
    if (!object || object == Py_None) return 0;
    return uform_py_enum_from_object(object, field, classes, value);
}

PyObject* uform_py_class_to_object(uform_py_enum_t const* classes, int value) {
    return value ? uform_py_enum_to_object(classes, value) : Py_NewRef(Py_None);
}

#pragma endregion Enums

#pragma region Arguments

int uform_py_arguments_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                             PyObject* args_names_tuple, Py_ssize_t positional_only, Py_ssize_t positional_max,
                             uform_py_argument_t const* arguments, size_t count) {
    if (positional_args_count > positional_max) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most %zd positional arguments, received %zd", verb, positional_max,
                     positional_args_count);
        return -1;
    }
    for (size_t position = 0; position != count; ++position) *arguments[position].value = NULL;
    for (Py_ssize_t position = 0; position != positional_args_count; ++position)
        *arguments[position].value = args[position];

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t name_index = 0; name_index != args_names_count; ++name_index) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, name_index);
        size_t position = (size_t)positional_only;
        while (position != count && PyUnicode_CompareWithASCIIString(name, arguments[position].name) != 0) ++position;
        if (position == count) {
            PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument %R", verb, name);
            return -1;
        }
        if (*arguments[position].value) {
            PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%s'", verb, arguments[position].name);
            return -1;
        }
        *arguments[position].value = args[positional_args_count + name_index];
    }
    for (Py_ssize_t position = 0; position != positional_only; ++position)
        if (!*arguments[position].value) {
            PyErr_Format(PyExc_TypeError, "%s() missing required argument '%s'", verb, arguments[position].name);
            return -1;
        }
    return 0;
}

int uform_py_size_from_object(PyObject* object, char const* field, size_t* value) {
    if (!object) return 0;
    Py_ssize_t const signed_value = PyNumber_AsSsize_t(object, PyExc_OverflowError);
    if (signed_value == -1 && PyErr_Occurred()) return -1;
    if (signed_value < 0)
        return uform_py_raise(uform_status_invalid_argument_k, "the '%s' option must not be negative", field);
    *value = (size_t)signed_value;
    return 0;
}

int uform_py_scalar_from_object(PyObject* object, char const* field, uform_scalar_t* kind) {
    if (!object) return 0;
    int value = uform_scalar_unknown_k;
    if (uform_py_enum_from_object(object, field, &uform_py_scalar, &value) != 0) return -1;
    *kind = (uform_scalar_t)value;
    return 0;
}

int uform_py_ordinals_from_object(PyObject* object, char const* field, int32_t** ordinals, size_t* count) {
    *ordinals = NULL;
    *count = 0;
    if (!object || object == Py_None) return 0;
    PyObject* const sequence = PySequence_Fast(object, "expected a sequence of integers");
    if (!sequence) return -1;
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    int32_t* const claimed = length ? (int32_t*)PyMem_Malloc((size_t)length * sizeof(int32_t)) : NULL;
    if (length && !claimed) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t position = 0; position != length; ++position) {
        long const ordinal = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, position));
        if ((ordinal == -1 && PyErr_Occurred()) || ordinal < INT32_MIN || ordinal > INT32_MAX) {
            PyMem_Free(claimed);
            Py_DECREF(sequence);
            if (PyErr_Occurred()) return -1;
            return uform_py_raise(uform_status_invalid_argument_k,
                                  "the '%s' option holds %ld, outside the 32-bit range", field, ordinal);
        }
        claimed[position] = (int32_t)ordinal;
    }
    Py_DECREF(sequence);
    *ordinals = claimed;
    *count = (size_t)length;
    return 0;
}

int uform_py_path_from_object(PyObject* object, char const* field, uform_bytes_view_t* path, PyObject** decoded) {
    PyObject* bytes_object = NULL;
    if (!PyUnicode_FSConverter(object, &bytes_object)) return -1;
    path->data = (uint8_t const*)PyBytes_AS_STRING(bytes_object);
    path->length = (size_t)PyBytes_GET_SIZE(bytes_object);
    *decoded = bytes_object;
    return 0;
}

PyObject* uform_py_struct_sequence_steal(PyTypeObject* type, PyObject** fields, Py_ssize_t count) {
    PyObject* result = NULL;
    Py_ssize_t filled = 0;
    while (filled != count && fields[filled]) ++filled;
    if (filled == count) result = PyStructSequence_New(type);
    if (!result) {
        for (Py_ssize_t field = 0; field != count; ++field) Py_XDECREF(fields[field]);
        return NULL;
    }
    for (Py_ssize_t field = 0; field != count; ++field) PyStructSequence_SET_ITEM(result, field, fields[field]);
    return result;
}

PyObject* uform_py_size_getter(PyObject* self, void* closure) {
    return PyLong_FromSize_t(*(size_t const*)(void const*)((char const*)self + (size_t)closure));
}

#pragma endregion Arguments

#pragma region Keyword Structures

/**
 *  @brief Checks and stores @p value as field @p field of a new structure, a tuple building a
 *      nested structure from its items in order.
 *  @return A new reference to what the structure keeps, or NULL with an exception set.
 */
static PyObject* field_accept(uform_py_field_t const* field, PyObject* value) {
    switch (field->kind) {
    case uform_py_field_size_k: {
        size_t checked = 0;
        return uform_py_size_from_object(value, field->name, &checked) == 0 ? Py_NewRef(value) : NULL;
    }
    case uform_py_field_float_k:
    case uform_py_field_double_k:
        if (PyNumber_Check(value)) return PyNumber_Float(value); // `PyNumber_Float` alone would parse a string
        PyErr_Format(PyExc_TypeError, "the '%s' field takes a float, got %S", field->name, Py_TYPE(value));
        return NULL;
    case uform_py_field_enum_k: {
        int member = 0;
        if (uform_py_enum_from_object(value, field->name, (uform_py_enum_t const*)field->detail, &member) != 0)
            return NULL;
        return uform_py_enum_to_object((uform_py_enum_t const*)field->detail, member);
    }
    case uform_py_field_struct_k: {
        PyTypeObject* const nested = (PyTypeObject*)field->detail;
        if (value == Py_None || Py_IS_TYPE(value, nested)) return Py_NewRef(value);
        if (!PyTuple_Check(value)) {
            PyErr_Format(PyExc_TypeError, "the '%s' field takes a uform.%s or a tuple of its fields, got %S",
                         field->name, nested->tp_name + sizeof("uform.") - 1, Py_TYPE(value));
            return NULL;
        }
        uform_py_fields_t const* const schema = (uform_py_fields_t const*)nested;
        Py_ssize_t const given = PyTuple_GET_SIZE(value);
        if ((size_t)given > schema->count) {
            PyErr_Format(PyExc_TypeError, "the '%s' field takes at most %zu values, got %zd", field->name,
                         schema->count, given);
            return NULL;
        }
        PyObject* keywords = PyDict_New();
        for (Py_ssize_t position = 0; keywords && position != given; ++position)
            if (PyDict_SetItemString(keywords, schema->fields[position].name, PyTuple_GET_ITEM(value, position)) != 0)
                Py_CLEAR(keywords);
        PyObject* const empty = keywords ? PyTuple_New(0) : NULL;
        PyObject* const built = empty ? PyObject_Call((PyObject*)nested, empty, keywords) : NULL;
        Py_XDECREF(empty);
        Py_XDECREF(keywords);
        return built;
    }
    case uform_py_field_object_k: return Py_NewRef(value);
    }
    return NULL;
}

/** The value a field the caller left out holds: zero for a number, the zero member of an
 *  enum, None otherwise. */
static PyObject* field_default(uform_py_field_t const* field) {
    switch (field->kind) {
    case uform_py_field_size_k: return PyLong_FromLong(0);
    case uform_py_field_float_k:
    case uform_py_field_double_k: return PyFloat_FromDouble(0.0);
    case uform_py_field_enum_k: return uform_py_enum_to_object((uform_py_enum_t const*)field->detail, 0);
    default: return Py_NewRef(Py_None);
    }
}

static PyObject* Fields_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    uform_py_fields_t const* const schema = (uform_py_fields_t const*)type;
    if (PyTuple_GET_SIZE(args) != 0) {
        PyErr_Format(PyExc_TypeError, "%s() takes keyword arguments only", type->tp_name + sizeof("uform.") - 1);
        return NULL;
    }
    Py_ssize_t const given = keywords ? PyDict_GET_SIZE(keywords) : 0; // owned by this call, so its values stay valid
    Fields* const instance = (Fields*)type->tp_alloc(type, 0);
    if (!instance) return NULL;
    Py_ssize_t matched = 0;
    for (size_t index = 0; index != schema->count; ++index) {
        uform_py_field_t const* const field = &schema->fields[index];
        PyObject* const value = keywords ? PyDict_GetItemString(keywords, field->name) : NULL;
        instance->values[index] = value ? field_accept(field, value) : field_default(field);
        matched += value != NULL;
        if (!instance->values[index]) goto failed;
    }
    if (matched != given) {
        PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument among %R",
                     type->tp_name + sizeof("uform.") - 1, keywords);
        goto failed;
    }
    return (PyObject*)instance;

failed:
    Py_DECREF(instance);
    return NULL;
}

static void Fields_dealloc(Fields* self) {
    for (size_t index = 0; index != uform_py_fields_max_k; ++index) Py_XDECREF(self->values[index]);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Fields_repr(Fields* self) {
    uform_py_fields_t const* const schema = (uform_py_fields_t const*)Py_TYPE(self);
    PyObject* const parts = PyList_New(0);
    for (size_t index = 0; parts && index != schema->count; ++index) {
        PyObject* const part = PyUnicode_FromFormat("%s=%R", schema->fields[index].name, self->values[index]);
        if (!part || PyList_Append(parts, part) != 0) {
            Py_XDECREF(part);
            Py_DECREF(parts);
            return NULL;
        }
        Py_DECREF(part);
    }
    PyObject* const separator = parts ? PyUnicode_FromString(", ") : NULL;
    PyObject* const joined = separator ? PyUnicode_Join(separator, parts) : NULL;
    PyObject* const repr = joined
                               ? PyUnicode_FromFormat("%s(%U)", Py_TYPE(self)->tp_name + sizeof("uform.") - 1, joined)
                               : NULL;
    Py_XDECREF(joined);
    Py_XDECREF(separator);
    Py_XDECREF(parts);
    return repr;
}

int uform_py_fields_ready(uform_py_fields_t* fields) {
    PyMemberDef* const members = (PyMemberDef*)PyMem_Calloc(fields->count + 1, sizeof(PyMemberDef));
    if (!members) {
        PyErr_NoMemory();
        return -1;
    }
    for (size_t index = 0; index != fields->count; ++index)
        members[index] = (PyMemberDef) {fields->fields[index].name, Py_T_OBJECT_EX,
                                        (Py_ssize_t)(offsetof(Fields, values) + index * sizeof(PyObject*)), Py_READONLY,
                                        fields->fields[index].doc};
    fields->type.tp_basicsize = sizeof(Fields);
    fields->type.tp_flags = Py_TPFLAGS_DEFAULT;
    fields->type.tp_new = Fields_tp_new;
    fields->type.tp_dealloc = (destructor)Fields_dealloc;
    fields->type.tp_repr = (reprfunc)Fields_repr;
    fields->type.tp_members = members;
    return PyType_Ready(&fields->type);
}

int uform_py_fields_write(PyObject* object, char const* field, uform_py_fields_t const* fields, void* target) {
    if (!object || object == Py_None) return 0;
    if (!Py_IS_TYPE(object, (PyTypeObject*)&fields->type)) {
        PyErr_Format(PyExc_TypeError, "the '%s' option takes a uform.%s, got %S", field,
                     fields->type.tp_name + sizeof("uform.") - 1, Py_TYPE(object));
        return -1;
    }
    for (size_t index = 0; index != fields->count; ++index) {
        uform_py_field_t const* const described = &fields->fields[index];
        PyObject* const value = ((Fields*)object)->values[index];
        char* const at = (char*)target + described->offset;
        switch (described->kind) {
        case uform_py_field_size_k:
            if (uform_py_size_from_object(value, described->name, (size_t*)(void*)at) != 0) return -1;
            break;
        case uform_py_field_float_k: *(float*)(void*)at = (float)PyFloat_AS_DOUBLE(value); break;
        case uform_py_field_double_k: *(double*)(void*)at = PyFloat_AS_DOUBLE(value); break;
        case uform_py_field_enum_k: *(int*)(void*)at = (int)PyLong_AsLong(value); break;
        case uform_py_field_struct_k:
            if (uform_py_fields_write(value, described->name, (uform_py_fields_t const*)described->detail, at) != 0)
                return -1;
            break;
        case uform_py_field_object_k: break;
        }
    }
    return 0;
}

#pragma endregion Keyword Structures

#pragma region Calls

/** The @c Deferred whose trailing storage holds @p call. */
static Deferred* deferred_of(void* call) { return (Deferred*)(void*)((char*)call - offsetof(Deferred, call)); }

void* uform_py_call_open(uform_py_receiver_t const* receiver, uform_py_verb_t const* verb, void* stack) {
    void* call = stack;
    if (receiver->completion == uform_completion_deferred_k) {
        Deferred* const deferred = PyObject_GC_NewVar(Deferred, &DeferredType, (Py_ssize_t)verb->call_bytes);
        if (!deferred) return NULL;
        deferred->verb = verb;
        deferred->origin = Py_NewRef(receiver->origin);
        atomic_init(&deferred->answering, 0);
        atomic_init(&deferred->answer, NULL);
        atomic_init(&deferred->raised, NULL);
        call = deferred->call;
        memset(call, 0, verb->call_bytes);
        PyObject_GC_Track(deferred);
        return call;
    }
    memset(call, 0, verb->call_bytes);
    return call;
}

/** Raises the ValueError of a result a later call made stale; always -1. */
static int result_stale(void) {
    PyErr_SetString(PyExc_ValueError, "the arena was reused by a later call");
    return -1;
}

/** Answers a final @p call: the outcome's failure, else a @c ValueError when a later call
 *  rewound its arena, else the verb's answer. */
static PyObject* call_answer(uform_py_verb_t const* verb, void* call) {
    uform_py_call_t* const header = (uform_py_call_t*)call;
    if (uform_py_raise_on_outcome(&header->outcome) != 0) return NULL; // the call's own failure comes first
    if (!header->arena) return verb->answer(call);
    // An answer may copy out of the arena, so it reads under an export, never after a later rewind
    uform_arena_t const arena = ((Arena*)header->arena)->handle;
    uform_status_t const opened = uform_arena_export_open(arena);
    if (opened == uform_status_contended_k) return result_stale(), NULL;
    if (opened != uform_status_success_k)
        return uform_py_raise(opened, "the arena could not count another export"), NULL;
    PyObject* const answer = header->generation == uform_arena_generation(arena) ? verb->answer(call)
                                                                                 : (result_stale(), NULL);
    uform_arena_export_close(arena);
    return answer;
}

/** Lets go of what @p call gathered, keeping whatever exception is set. */
static void call_release(uform_py_verb_t const* verb, void* call) {
    PyObject* const raised = PyErr_GetRaisedException();
    verb->release(call);
    PyErr_SetRaisedException(raised);
}

PyObject* uform_py_call_run(uform_py_receiver_t const* receiver, uform_py_verb_t const* verb, void* call) {
    // The call's take makes the next generation; a racing take can only make the result read stale
    uform_py_call_t* const header = (uform_py_call_t*)call;
    if (header->arena) header->generation = uform_arena_generation(((Arena*)header->arena)->handle) + 1;
    Py_BEGIN_ALLOW_THREADS;
    verb->submit(call);
    Py_END_ALLOW_THREADS;
    if (receiver->completion == uform_completion_deferred_k) return (PyObject*)deferred_of(call);
    PyObject* const answer = call_answer(verb, call);
    call_release(verb, call);
    return answer;
}

PyObject* uform_py_call_discard(uform_py_receiver_t const* receiver, uform_py_verb_t const* verb, void* call) {
    call_release(verb, call);
    if (receiver->completion == uform_completion_blocking_k) return NULL;
    Deferred* const deferred = deferred_of(call);
    atomic_store(&deferred->answering, 1); // never submitted, so nothing is left to answer or let go
    PyObject* const raised = PyErr_GetRaisedException();
    Py_DECREF(deferred);
    PyErr_SetRaisedException(raised);
    return NULL;
}

void uform_py_call_release(void* call) {
    uform_py_call_t* const header = (uform_py_call_t*)call;
    Py_CLEAR(header->arena);
    Py_CLEAR(header->result);
}

void* uform_py_call_result(uform_py_call_t* call, PyTypeObject* type) {
    call->result = type->tp_alloc(type, 0);
    return call->result;
}

PyObject* uform_py_call_answer_result(void* call) {
    PyObject* const result = ((uform_py_call_t*)call)->result;
    return Py_NewRef(result ? result : Py_None);
}

int uform_py_callable_from_object(PyObject* object, char const* field, PyObject** callable) {
    *callable = NULL;
    if (!object || object == Py_None) return 0;
    if (!PyCallable_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' handler must be callable, got %S", field, Py_TYPE(object));
        return -1;
    }
    *callable = Py_NewRef(object);
    return 0;
}

/**
 *  @brief Re-enters Python for one progress report, from the thread running the call.
 *
 *  Reports are serialized, but a deferred call runs on the executor's thread 0, not on its caller,
 *  so the GIL is taken here rather than assumed. A handler answering None continues and a falsy
 *  one cancels; one that raises reports through the unraisable hook and cancels.
 */
static uform_continuation_t progress_trampoline(size_t completed, size_t total, void* state) {
    PyObject* const callable = (PyObject*)state;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    PyObject* const completed_object = PyLong_FromSize_t(completed);
    PyObject* const total_object = completed_object ? PyLong_FromSize_t(total) : NULL;
    PyObject* const arguments[2] = {completed_object, total_object};
    PyObject* const verdict = total_object ? PyObject_Vectorcall(callable, arguments, 2, NULL) : NULL;
    int const truth = !verdict ? -1 : verdict == Py_None ? 1 : PyObject_IsTrue(verdict);
    Py_XDECREF(verdict);
    Py_XDECREF(total_object);
    Py_XDECREF(completed_object);
    if (truth < 0) PyErr_WriteUnraisable(callable);
    PyGILState_Release(gil_state);
    return truth > 0 ? uform_continue_k : uform_cancel_k;
}

uform_progress_t uform_py_progress_registered(PyObject* callable, size_t steps) {
    return (uform_progress_t) {.callback = callable ? progress_trampoline : NULL, .state = callable, .steps = steps};
}

#pragma endregion Calls

#pragma region Receivers and Deferred Calls

/** A model or trainer bound to the executor its verbs run on; its type alone chooses
 *  how they complete. */
typedef struct On {
    PyObject ob_base;

    /** The model or trainer the verbs address, held. */
    PyObject* target;

    /** The @c Executor the calls run on, held. */
    PyObject* executor;
} On;

static char const doc_On[] =                                                                  //
    "On[Model | Trainer]\n"                                                                   //
    "\n"                                                                                      //
    "A model or trainer bound to the executor its verbs block on, as `model.on(executor)`\n"  //
    "answers it.\n"                                                                           //
    "\n"                                                                                      //
    "Offers the verbs of its target with the same arguments and answers; each call runs on\n" //
    "the executor, and one that finds another call holding it raises `ContendedError`.\n"     //
    "It holds the target and the executor, so it may outlive the names they were bound to.\n" //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> executor = uform.Executor.for_ram()\n"                                             //
    "  >>> model = uform.Model.load_ram_buffer(b'any bytes the stub accepts')\n"              //
    "  >>> model.on(executor).describe().quantization\n"                                      //
    "  <Scalar.BF16: 3>";

static char const doc_DeferredOn[] =                                                                //
    "DeferredOn[Model | Trainer]\n"                                                                 //
    "\n"                                                                                            //
    "A model or trainer bound to the executor its verbs submit to, as `model.deferred(executor)`\n" //
    "answers it.\n"                                                                                 //
    "\n"                                                                                            //
    "Offers the verbs of its target with the same arguments, each answering a `Deferred` at\n"      //
    "once. Arguments the binding refuses raise at the call, while the library's refusals,\n"        //
    "`ContendedError` from an executor another call holds included, raise at `await` or `wait()`.";

int uform_py_receiver_from_object(PyObject* self, PyTypeObject* type, uform_py_receiver_t* receiver) {
    int const deferred = Py_IS_TYPE(self, &DeferredOnType);
    receiver->origin = self;
    receiver->target = self;
    receiver->executor = NULL;
    receiver->completion = deferred ? uform_completion_deferred_k : uform_completion_blocking_k;
    if (deferred || Py_IS_TYPE(self, &OnType)) receiver->target = ((On*)self)->target;
    if (!Py_IS_TYPE(receiver->target, type)) {
        PyErr_Format(PyExc_TypeError, "the verb addresses a uform.%s, got a receiver over %S",
                     type->tp_name + sizeof("uform.") - 1, Py_TYPE(receiver->target));
        return -1;
    }
    if (receiver->target == self) return 0;
    return uform_py_executor_from_object(((On*)self)->executor, &receiver->executor);
}

PyObject* uform_py_on_new(PyTypeObject* type, PyObject* target, PyObject* executor) {
    uform_executor_t handle = NULL;
    if (uform_py_executor_from_object(executor, &handle) != 0) return NULL;
    On* const on = PyObject_New(On, type);
    if (!on) return NULL;
    on->target = Py_NewRef(target);
    on->executor = Py_NewRef(executor);
    return (PyObject*)on;
}

static PyObject* On_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError, "receivers are built through model.on(executor) or model.deferred(executor)");
    return NULL;
}

static void On_dealloc(On* self) {
    Py_XDECREF(self->executor);
    Py_XDECREF(self->target);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* On_repr(On* self) {
    return PyUnicode_FromFormat("%s(%R, %R)", Py_TYPE(self)->tp_name + sizeof("uform.") - 1, self->target,
                                self->executor);
}

/** Binds the verb @p name of the target the receiver holds, from the tables its type shares with
 *  its receivers; anything else resolves as on any object. */
static PyObject* On_getattro(PyObject* self, PyObject* name) {
    PyMethodDef* const model_tables[] = {uform_py_model_verbs, uform_py_encode_verbs, NULL};
    PyMethodDef* const trainer_tables[] = {uform_py_trainer_verbs, NULL};
    PyMethodDef* const* tables = Py_IS_TYPE(((On*)self)->target, &ModelType) ? model_tables : trainer_tables;
    for (; PyUnicode_Check(name) && *tables; ++tables)
        for (PyMethodDef* verb = *tables; verb->ml_name; ++verb)
            if (PyUnicode_CompareWithASCIIString(name, verb->ml_name) == 0) return PyCFunction_New(verb, self);
    return PyObject_GenericGetAttr(self, name);
}

static PyMethodDef On_methods[] = {
    {"__class_getitem__", Py_GenericAlias, METH_O | METH_CLASS,
     "Names the target the receiver addresses, as `On[Model]` or `DeferredOn[Trainer]`."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject OnType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.On",
    .tp_doc = doc_On,
    .tp_basicsize = sizeof(On),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = On_tp_new,
    .tp_dealloc = (destructor)On_dealloc,
    .tp_repr = (reprfunc)On_repr,
    .tp_getattro = On_getattro,
    .tp_methods = On_methods,
};

PyTypeObject DeferredOnType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.DeferredOn",
    .tp_doc = doc_DeferredOn,
    .tp_basicsize = sizeof(On),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = On_tp_new,
    .tp_dealloc = (destructor)On_dealloc,
    .tp_repr = (reprfunc)On_repr,
    .tp_getattro = On_getattro,
    .tp_methods = On_methods,
};

static char const doc_Deferred[] =                                                              //
    "Deferred[Answer]\n"                                                                        //
    "\n"                                                                                        //
    "A call a `DeferredOn` submitted, which answers once it is final.\n"                        //
    "\n"                                                                                        //
    "`await` suspends the coroutine until the call is final, and `wait()` blocks the thread\n"  //
    "instead. Either answers what the blocking verb would have answered, or raises what it\n"   //
    "would have raised, as many times as it is asked; `status` polls without waiting. It\n"     //
    "holds the call and the buffers it reads until the call is final, so dropping it waits,\n"  //
    "and holds its arena as the blocking verb would. `await` polls `status` between turns\n"    //
    "of the running event loop, so no thread waits for it.\n"                                   //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> import asyncio\n"                                                                    //
    "  >>> executor = uform.Executor.for_ram()\n"                                               //
    "  >>> model = uform.Model.load_ram_buffer(b'any bytes the stub accepts')\n"                //
    "  >>> async def encode():\n"                                                               //
    "  ...     call = model.deferred(executor).text_encode(['hello'], uform.Arena(executor))\n" //
    "  ...     return await call  # or: call.wait()\n"                                          //
    "  >>> len(asyncio.run(encode()).embeddings)\n"                                             //
    "  1";

/** The outcome of the call @p deferred holds. */
static uform_outcome_t* deferred_outcome(Deferred* deferred) {
    return &((uform_py_call_t*)(void*)deferred->call)->outcome;
}

/** The status of the call @p deferred holds, as the library finishes it elsewhere. */
static uform_status_t deferred_status(Deferred* deferred) { return uform_outcome_status(deferred_outcome(deferred)); }

/** Blocks with the thread state detached until the call @p deferred holds is final. */
static void deferred_block(Deferred* deferred) {
    if (deferred_status(deferred) != uform_status_deferred_k) return;
    uform_outcome_t* const outcome = deferred_outcome(deferred);
    Py_BEGIN_ALLOW_THREADS;
    uform_outcome_wait(outcome);
    Py_END_ALLOW_THREADS;
}

/**
 *  @brief Answers what a final @p self finished with.
 *
 *  The one thread taking the call answers it, lets go of its inputs and publishes; a thread asking
 *  while that one still answers is refused as contended rather than made to wait.
 */
static PyObject* deferred_answer(Deferred* self) {
    if (!atomic_exchange(&self->answering, 1)) {
        PyObject* const answer = call_answer(self->verb, self->call);
        PyObject* const raised = answer ? NULL : PyErr_GetRaisedException();
        call_release(self->verb, self->call);
        if (answer) atomic_store_explicit(&self->answer, answer, memory_order_release);
        else atomic_store_explicit(&self->raised, raised, memory_order_release);
    }
    PyObject* const raised = atomic_load_explicit(&self->raised, memory_order_acquire);
    if (raised) {
        PyErr_SetRaisedException(Py_NewRef(raised));
        return NULL;
    }
    PyObject* const answer = atomic_load_explicit(&self->answer, memory_order_acquire);
    if (answer) return Py_NewRef(answer);
    return uform_py_raise(uform_status_contended_k, "another thread is answering this call"), NULL;
}

static char const doc_Deferred_status[] =                                                        //
    "The status of the call: `Status.DEFERRED` until it is final, then the final one, such as\n" //
    "`Status.SUCCESS`. Calls `uform_outcome_status`.";

static PyObject* Deferred_get_status(Deferred* self, void* closure) {
    return uform_py_enum_to_object(&uform_py_status, (int)deferred_status(self));
}

static char const doc_Deferred_wait[] =                                                      //
    "wait() -> object\n"                                                                     //
    "\n"                                                                                     //
    "Blocks the calling thread, letting other threads run, until the call is final,\n"       //
    "then answers what the blocking verb would have answered; `await` is the native form.\n" //
    "Calls `uform_outcome_wait`.\n"                                                          //
    "\n"                                                                                     //
    "Raises:\n"                                                                              //
    "  Error: What the blocking verb would have raised, `ContendedError` included when\n"    //
    "    another call held the executor or another thread is answering this one.";

static PyObject* Deferred_wait(Deferred* self, PyObject* unused) {
    deferred_block(self);
    return deferred_answer(self);
}

/** Answers the iterator of <tt>uform._settled(self)</tt>, which re-polls the status once per turn of
 *  the running loop, so no thread is parked on the call. */
static PyObject* Deferred_await(PyObject* self) {
    PyObject* const package = PyImport_ImportModule("uform");
    PyObject* const settled = package ? PyObject_GetAttrString(package, "_settled") : NULL;
    Py_XDECREF(package);
    PyObject* const coroutine = settled ? PyObject_CallOneArg(settled, self) : NULL;
    Py_XDECREF(settled);
    PyObject* const awaiting = coroutine ? PyObject_CallMethod(coroutine, "__await__", NULL) : NULL;
    Py_XDECREF(coroutine);
    return awaiting;
}

static void Deferred_dealloc(Deferred* self) {
    PyObject_GC_UnTrack(self);
    // The library writes into the held call and reads its inputs until the call is final
    if (!atomic_load(&self->answering)) {
        deferred_block(self);
        call_release(self->verb, self->call);
    }
    Py_XDECREF(atomic_load(&self->answer));
    Py_XDECREF(atomic_load(&self->raised));
    Py_XDECREF(self->origin);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Visits what @p self holds; an unfinished call's own objects stay unvisited, so every cycle
 *  through them waits for a later collection rather than being cleared while the library
 *  still reads them. */
static int Deferred_traverse(Deferred* self, visitproc visit, void* arg) {
    PyObject* const answer = atomic_load_explicit(&self->answer, memory_order_relaxed);
    PyObject* const raised = atomic_load_explicit(&self->raised, memory_order_relaxed);
    Py_VISIT(self->origin);
    Py_VISIT(answer);
    Py_VISIT(raised);
    if (atomic_load(&self->answering) || deferred_status(self) == uform_status_deferred_k) return 0;
    uform_py_call_t const* const header = (uform_py_call_t const*)(void const*)self->call;
    Py_VISIT(header->arena);
    Py_VISIT(header->result);
    return self->verb->traverse ? self->verb->traverse(self->call, visit, arg) : 0;
}

/** Breaks a garbage cycle through a final @p self; an unfinished one is left whole, as its
 *  traversal hid the cycle. */
static int Deferred_clear(Deferred* self) {
    if (deferred_status(self) == uform_status_deferred_k) return 0;
    if (!atomic_exchange(&self->answering, 1)) call_release(self->verb, self->call);
    Py_XDECREF(atomic_exchange(&self->answer, NULL));
    Py_XDECREF(atomic_exchange(&self->raised, NULL));
    Py_CLEAR(self->origin);
    return 0;
}

static PyObject* Deferred_repr(Deferred* self) {
    return PyUnicode_FromFormat("Deferred(status=%s)", uform_status_name(deferred_status(self)));
}

static PyMethodDef Deferred_methods[] = {
    {"wait", (PyCFunction)Deferred_wait, METH_NOARGS, doc_Deferred_wait},
    {"__class_getitem__", Py_GenericAlias, METH_O | METH_CLASS, "Names what `wait()` answers, as `Deferred[Encoded]`."},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef Deferred_getset[] = {
    {"status", (getter)Deferred_get_status, NULL, doc_Deferred_status, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyAsyncMethods Deferred_as_async = {.am_await = Deferred_await};

PyTypeObject DeferredType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.Deferred",
    .tp_doc = doc_Deferred,
    .tp_basicsize = sizeof(Deferred),
    .tp_itemsize = 1,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_new = NULL, // Answered by the verbs of a deferred receiver, never constructed by name.
    .tp_dealloc = (destructor)Deferred_dealloc,
    .tp_traverse = (traverseproc)Deferred_traverse,
    .tp_clear = (inquiry)Deferred_clear,
    .tp_repr = (reprfunc)Deferred_repr,
    .tp_as_async = &Deferred_as_async,
    .tp_methods = Deferred_methods,
    .tp_getset = Deferred_getset,
};

#pragma endregion Receivers and Deferred Calls

#pragma region Executors

static char const doc_Executor[] =                                                              //
    "Executor\n"                                                                                //
    "\n"                                                                                        //
    "The threads or streams a verb runs on, the activation scratch and the staging, shared\n"   //
    "across models.\n"                                                                          //
    "\n"                                                                                        //
    "An executor owns host threads or device streams; a model owns weights. Keeping them\n"     //
    "apart is what lets one executor serve several models, and what makes the cost of\n"        //
    "spawning visible at the line that spawns rather than hidden in a constructor.\n"           //
    "\n"                                                                                        //
    "Construct one through the residency it drives - `for_ram` or `for_vram` - since each\n"    //
    "spawns something different and takes different arguments. A receiver names it:\n"          //
    "`model.on(executor)` blocks on it, `model.deferred(executor)` submits to it. It runs\n"    //
    "one call at a time, and a call finding another one holding it raises a `ContendedError`\n" //
    "once waited for. In a `with` block, it closes on exit, refusing further verbs; its\n"      //
    "threads are freed once every arena, result and receiver holding it is gone.\n"             //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> with uform.Executor.for_ram() as executor:\n"                                        //
    "  ...     executor.drain(uform.Mood.PARKED)";

/** Wraps a fresh handle, or frees it when the wrapper cannot be allocated; a new
 *  reference, or NULL. */
static PyObject* executor_adopt(PyTypeObject* type, uform_executor_t handle) {
    Executor* const executor = (Executor*)type->tp_alloc(type, 0);
    if (!executor) {
        uform_executor_free(handle);
        return NULL;
    }
    atomic_init(&executor->handle, handle);
    executor->owned = handle;
    return (PyObject*)executor;
}

static PyObject* Executor_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError, "Executor is built through Executor.for_ram() or .for_vram()");
    return NULL;
}

static void Executor_dealloc(Executor* executor) {
    uform_status_t freed = uform_status_success_k;
    Py_BEGIN_ALLOW_THREADS;
    freed = uform_executor_free(executor->owned);
    Py_END_ALLOW_THREADS;
    // Every Arena holds its Executor, so the free never refuses; were it to, the threads leak
    if (freed != uform_status_success_k) {
        PyObject* const pending = PyErr_GetRaisedException();
        uform_py_raise(freed, "an Executor died while an Arena made with it lived, so its threads leak");
        PyErr_WriteUnraisable(NULL);
        PyErr_SetRaisedException(pending);
    }
    Py_TYPE(executor)->tp_free((PyObject*)executor);
}

static PyObject* Executor_enter(PyObject* self, PyObject* unused) { return Py_NewRef(self); }

static PyObject* Executor_exit(PyObject* self, PyObject* const* args, Py_ssize_t count) {
    atomic_store(&((Executor*)self)->handle, NULL);
    Py_RETURN_NONE;
}

int uform_py_executor_from_object(PyObject* object, uform_executor_t* executor) {
    if (!PyObject_TypeCheck(object, &ExecutorType)) {
        PyErr_Format(PyExc_TypeError, "expected an Executor, got %S", Py_TYPE(object));
        return -1;
    }
    // A null executor would quietly run the call on the calling thread
    if (!atomic_load(&((Executor*)object)->handle))
        return uform_py_raise(uform_status_uninitialized_k, "the executor is closed");
    // A close racing the verb past this check stops only later verbs, as dealloc frees the executor
    *executor = ((Executor*)object)->owned;
    return 0;
}

int uform_py_executor_optional(PyObject* object, uform_executor_t* executor) {
    *executor = NULL;
    return !object || object == Py_None ? 0 : uform_py_executor_from_object(object, executor);
}

static char const doc_Executor_for_ram[] =                                                        //
    "for_ram(threads=0, *, memory_domains=None, cpu_class_ceiling=None) -> Executor\n"            //
    "\n"                                                                                          //
    "Creates an executor of host threads for RAM-resident models, interleaved across the\n"       //
    "compute domains local to the named memory domains.\n"                                        //
    "Calls `uform_executor_for_ram`.\n"                                                           //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  threads (int, optional): Threads to spawn; 0 takes 7/8 of the cores the domains\n"         //
    "    span.\n"                                                                                 //
    "  memory_domains (Sequence[int], optional): NUMA node IDs of the models it will\n"           //
    "    drive; None derives every node the process may allocate on.\n"                           //
    "  cpu_class_ceiling (CPUClass, optional): The newest class to dispatch as, capped by what\n" //
    "    the CPU runs; None sets no ceiling.\n"                                                   //
    "\n"                                                                                          //
    "Example:\n"                                                                                  //
    "  >>> with uform.Executor.for_ram(threads=8) as executor:\n"                                 //
    "  ...     executor.drain()";

static PyObject* Executor_for_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    PyObject *threads_object, *memory_domains_object, *cpu_class_object;
    uform_py_argument_t const arguments[] = {
        {"threads", &threads_object},
        {"memory_domains", &memory_domains_object},
        {"cpu_class_ceiling", &cpu_class_object},
    };
    size_t threads = 0;
    int cpu_class_ceiling = 0;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (uform_py_arguments_parse("for_ram", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_size_from_object(threads_object, "threads", &threads) != 0 ||
        uform_py_class_from_object(cpu_class_object, "cpu_class_ceiling", &uform_py_cpu_class, &cpu_class_ceiling) !=
            0 ||
        uform_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                      &memory_domains_count) != 0)
        return NULL;

    uform_executor_t handle = NULL;
    uform_outcome_t outcome = {0};
    uform_executor_for_ram_t executor_arguments = {
        .executor = &handle,
        .outcome = &outcome,
        .ram = {memory_domains, memory_domains_count, (uform_cpu_class_t)cpu_class_ceiling},
        .threads = threads};
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_for_ram(&executor_arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);

    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

static char const doc_Executor_for_vram[] =                                                        //
    "for_vram(devices=None, *, scratch_limit=0, gpu_class_ceiling=None, "                          //
    "memory_domains=None, cpu_class_ceiling=None) -> Executor\n"                                   //
    "\n"                                                                                           //
    "Opens one submission stream on every named device, all of them driven by the host\n"          //
    "thread running the call, and spawns the host threads that decode what the devices\n"          //
    "cannot.\n"                                                                                    //
    "Calls `uform_executor_for_vram`.\n"                                                           //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  devices (Sequence[int], optional): Positions in `gpu_devices()` to drive; None\n"           //
    "    derives one free device.\n"                                                               //
    "  scratch_limit (int, optional): The ceiling on scratch bytes per device; 0 lets\n"           //
    "    scratch grow as batches ask, and a batch larger than the ceiling runs in passes.\n"       //
    "  gpu_class_ceiling (GPUClass, optional): The newest class every device dispatches as, the\n" //
    "    best class all of them load under it; None sets no ceiling.\n"                            //
    "  memory_domains (Sequence[int], optional): NUMA node IDs whose cores the host\n"             //
    "    decoders take; None derives every node the process may allocate on.\n"                    //
    "  cpu_class_ceiling (CPUClass, optional): The newest class the host decoders dispatch as;\n"  //
    "    None sets no ceiling.\n"                                                                  //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "  >>> executor = uform.Executor.for_vram([0, 1])   # doctest: +SKIP";

static PyObject* Executor_for_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *devices_object, *scratch_limit_object, *gpu_class_object, *memory_domains_object, *cpu_class_object;
    uform_py_argument_t const arguments[] = {
        {"devices", &devices_object},
        {"scratch_limit", &scratch_limit_object},
        {"gpu_class_ceiling", &gpu_class_object},
        {"memory_domains", &memory_domains_object},
        {"cpu_class_ceiling", &cpu_class_object},
    };
    size_t scratch_limit = 0;
    int gpu_class_ceiling = 0, cpu_class_ceiling = 0;
    int32_t *devices = NULL, *memory_domains = NULL;
    size_t devices_count = 0, memory_domains_count = 0;
    if (uform_py_arguments_parse("for_vram", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_size_from_object(scratch_limit_object, "scratch_limit", &scratch_limit) != 0 ||
        uform_py_class_from_object(gpu_class_object, "gpu_class_ceiling", &uform_py_gpu_class, &gpu_class_ceiling) !=
            0 ||
        uform_py_class_from_object(cpu_class_object, "cpu_class_ceiling", &uform_py_cpu_class, &cpu_class_ceiling) != 0)
        return NULL;
    if (uform_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0) return NULL;
    if (uform_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                      &memory_domains_count) != 0) {
        PyMem_Free(devices);
        return NULL;
    }

    uform_executor_t handle = NULL;
    uform_outcome_t outcome = {0};
    uform_executor_for_vram_t executor_arguments = {
        .executor = &handle,
        .outcome = &outcome,
        .vram = {devices, devices_count, (uform_gpu_class_t)gpu_class_ceiling},
        .ram = {memory_domains, memory_domains_count, (uform_cpu_class_t)cpu_class_ceiling},
        .scratch_limit = scratch_limit};
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_for_vram(&executor_arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);
    PyMem_Free(memory_domains);

    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

static char const doc_Executor_drain[] =                                                  //
    "drain(mood=Mood.PARKED) -> None\n"                                                   //
    "\n"                                                                                  //
    "Waits until no call holds this executor, then sets what its threads or streams do\n" //
    "until its next call.\n"                                                              //
    "Calls `uform_executor_drain`.\n"                                                     //
    "\n"                                                                                  //
    "Args:\n"                                                                             //
    "  mood (Mood, optional): `Mood.PARKED` frees the core, so the next call pays a\n"    //
    "    wake-up; `Mood.SPINNING` keeps polling, trading a core for the next call's\n"    //
    "    latency.\n"                                                                      //
    "\n"                                                                                  //
    "Raises:\n"                                                                           //
    "  Error: What befell the executor itself, such as a lost device; every call\n"       //
    "    reports its own failure through its own verb.";

static PyObject* Executor_drain(Executor* executor, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    PyObject* mood_object;
    uform_py_argument_t const arguments[] = {{"mood", &mood_object}};
    int mood = uform_mood_parked_k;
    uform_executor_t handle = NULL;
    if (uform_py_arguments_parse("drain", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        (mood_object && uform_py_enum_from_object(mood_object, "mood", &uform_py_mood, &mood) != 0) ||
        uform_py_executor_from_object((PyObject*)executor, &handle) != 0)
        return NULL;

    uform_outcome_t outcome = {0};
    uform_executor_drain_t drain_arguments = {.executor = handle, .outcome = &outcome, .mood = (uform_mood_t)mood};
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_drain(&drain_arguments);
    Py_END_ALLOW_THREADS;
    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject* Executor_repr(Executor* executor) {
    return PyUnicode_FromFormat("uform.Executor(at=%p)", (void*)atomic_load(&executor->handle));
}

static PyMethodDef Executor_methods[] = {
    {"for_ram", (PyCFunction)Executor_for_ram, uform_py_method_flags_k | METH_CLASS, doc_Executor_for_ram},
    {"for_vram", (PyCFunction)Executor_for_vram, uform_py_method_flags_k | METH_CLASS, doc_Executor_for_vram},
    {"drain", (PyCFunction)Executor_drain, uform_py_method_flags_k, doc_Executor_drain},
    {"__enter__", (PyCFunction)Executor_enter, METH_NOARGS, NULL},
    {"__exit__", (PyCFunction)Executor_exit, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL},
};

PyTypeObject ExecutorType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.Executor",
    .tp_doc = doc_Executor,
    .tp_basicsize = sizeof(Executor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Executor_tp_new,
    .tp_dealloc = (destructor)Executor_dealloc,
    .tp_methods = Executor_methods,
    .tp_repr = (reprfunc)Executor_repr,
};

#pragma endregion Executors

#pragma region DLPack

/** The buffer formats a DLPack tensor crosses as, and the element type each stands for. */
static struct {
    char const* format;
    DLDataType dtype;
} const dlpack_formats[] = {
    {"B", {kDLUInt, 8, 1}},   {"b", {kDLInt, 8, 1}},    {"I", {kDLUInt, 32, 1}},
    {"i", {kDLInt, 32, 1}},   {"Q", {kDLUInt, 64, 1}},  {"e", {kDLFloat, 16, 1}},
    {"f", {kDLFloat, 32, 1}}, {"d", {kDLFloat, 64, 1}}, {"N", {kDLUInt, sizeof(size_t) * 8, 1}},
};

/** The DLPack device of the memory @p description names, derived as its type documents: VRAM the
 *  host may not read is device memory, VRAM it may read is CUDA managed memory, RAM placed for a
 *  device is page-locked for it, and other RAM is host memory. Apple's shared memory crosses as
 *  host memory, since no Metal exchange is supported yet. */
static DLDevice uform_py_dlpack_device(uform_arena_description_t const* description) {
    int32_t const ordinal = description->vendor_ordinal;
    int const readable = description->host_access == uform_host_access_readable_k;
    if (description->residency == uform_residency_vram_k && description->vendor == uform_gpu_vendor_apple_k)
        return (DLDevice) {kDLCPU, 0};
    if (description->residency == uform_residency_vram_k && description->vendor == uform_gpu_vendor_amd_k)
        return (DLDevice) {kDLROCM, ordinal};
    if (description->residency == uform_residency_vram_k)
        return (DLDevice) {readable ? kDLCUDAManaged : kDLCUDA, ordinal};
    if (description->device >= 0 && description->vendor == uform_gpu_vendor_nvidia_k)
        return (DLDevice) {kDLCUDAHost, ordinal};
    if (description->device >= 0 && description->vendor == uform_gpu_vendor_amd_k)
        return (DLDevice) {kDLROCMHost, ordinal};
    return (DLDevice) {kDLCPU, 0};
}

/**
 *  @brief Finds the device whose memory a call on @p arena may read: the GPU a VRAM arena lives on.
 *  @return 1 with @p device set, 0 for no arena or host pages, or -1 with an exception set.
 */
static int arena_device(PyObject* arena, DLDevice* device) {
    if (!arena || !PyObject_TypeCheck(arena, &ArenaType)) return 0; // a stranger is refused once the call holds it
    uform_arena_description_t description = {0};
    uform_outcome_t outcome = {0};
    uform_arena_describe(
        &(uform_arena_describe_t) {.arena = ((Arena*)arena)->handle, .outcome = &outcome, .description = &description});
    if (uform_py_raise_on_outcome(&outcome) != 0) return -1;
    if (description.residency != uform_residency_vram_k) return 0;
    *device = uform_py_dlpack_device(&description);
    return 1;
}

/** The executor @p arena serves, which it holds, or NULL for an arena of the calling thread. */
static uform_executor_t arena_executor(PyObject* arena) {
    PyObject* const executor = ((Arena*)arena)->executor;
    return executor ? ((Executor*)executor)->owned : NULL;
}

/** Orders the first stream of @p arena 's executor after a foreign @p event or @p stream. */
static uform_status_t arena_follow(Arena* arena, void* event, void* stream) {
    if (!event && !stream) return uform_status_success_k;
    uform_outcome_t outcome = {0};
    uform_executor_follow(&(uform_executor_follow_t) {.executor = arena_executor((PyObject*)arena),
                                                      .position = 0,
                                                      .event = event,
                                                      .stream = stream,
                                                      .outcome = &outcome});
    return outcome.status;
}

/** Makes the host wait for a foreign @p event before reading what its producer wrote: the GPU
 *  @p arena's first stream follows it and a drain waits for that stream; 0, or -1 with an error. */
static int arena_wait(PyObject* arena, void* event) {
    DLDevice device;
    int const on_device = arena_device(arena, &device);
    if (on_device <= 0)
        return on_device ? -1
                         : uform_py_raise(uform_status_device_k, "only a call on a GPU arena waits for a producer");
    uform_status_t const followed = arena_follow((Arena*)arena, event, NULL);
    if (followed != uform_status_success_k)
        return uform_py_raise(followed, "the arena's stream could not follow the Arrow array's producer");
    uform_executor_t const executor = arena_executor(arena);
    uform_outcome_t outcome = {0};
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_drain(
        &(uform_executor_drain_t) {.executor = executor, .outcome = &outcome, .mood = uform_mood_parked_k});
    Py_END_ALLOW_THREADS;
    return uform_py_raise_on_outcome(&outcome);
}

/**
 *  @brief The @c stream a DLPack consumer on @p arena passes @p object 's @c __dlpack__, as the
 *      standard spells it: the arena's first stream when the producer's memory is on that GPU's
 *      vendor, CUDA's legacy stream as 1, and @c None otherwise.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* dlpack_stream_argument(PyObject* object, PyObject* arena) {
    DLDevice device;
    int const on_device = arena_device(arena, &device);
    if (on_device <= 0) return on_device ? NULL : Py_NewRef(Py_None);
    int const cuda = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
    if (!cuda && device.device_type != kDLROCM) return Py_NewRef(Py_None); // Apple arenas cross as host memory
    PyObject* const placement = PyObject_HasAttrString(object, "__dlpack_device__")
                                    ? PyObject_CallMethod(object, "__dlpack_device__", NULL)
                                    : Py_NewRef(Py_None);
    if (!placement) return NULL;
    long const producer = PyTuple_Check(placement) && PyTuple_GET_SIZE(placement) == 2
                              ? PyLong_AsLong(PyTuple_GET_ITEM(placement, 0))
                              : -1;
    Py_DECREF(placement);
    if (producer == -1 && PyErr_Occurred()) return NULL;
    int const same_vendor = cuda ? producer == kDLCUDA || producer == kDLCUDAManaged : producer == kDLROCM;
    if (!same_vendor) return Py_NewRef(Py_None);
    void* stream = NULL;
    uform_outcome_t outcome = {0};
    uform_executor_stream(&(uform_executor_stream_t) {.executor = arena_executor(arena),
                                                      .position = 0,
                                                      .outcome = &outcome,
                                                      .stream = &stream});
    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    return stream ? PyLong_FromVoidPtr(stream) : PyLong_FromLong(cuda ? 1 : 0);
}

/** Hands the tensor a consumed capsule holds back to its producer, and frees the extents
 *  its view pointed at. */
static void dlpack_capsule_release(PyObject* capsule) {
    PyMem_Free(PyCapsule_GetContext(capsule));
    if (PyCapsule_IsValid(capsule, "used_dltensor_versioned")) {
        DLManagedTensorVersioned* const managed = PyCapsule_GetPointer(capsule, "used_dltensor_versioned");
        if (managed->deleter) managed->deleter(managed);
    }
    else if (PyCapsule_IsValid(capsule, "used_dltensor")) {
        DLManagedTensor* const managed = PyCapsule_GetPointer(capsule, "used_dltensor");
        if (managed->deleter) managed->deleter(managed);
    }
}

/**
 *  @brief Asks @p exporter for a DLPack capsule, versioned first, and a legacy one from a producer
 *      that predates @c max_version, passing @p stream for it to order its writes before.
 *  @return A new capsule, or NULL with the producer's exception set.
 */
static PyObject* dlpack_capsule_request(PyObject* exporter, PyObject* stream) {
    PyObject* const keywords = PyDict_New();
    PyObject* const version = keywords ? PyTuple_New(2) : NULL;
    if (version) {
        PyTuple_SET_ITEM(version, 0, PyLong_FromLong(DLPACK_MAJOR_VERSION));
        PyTuple_SET_ITEM(version, 1, PyLong_FromLong(DLPACK_MINOR_VERSION));
    }
    int const packed = version && PyTuple_GET_ITEM(version, 0) && PyTuple_GET_ITEM(version, 1) &&
                       uform_py_dict_steal_item(keywords, "stream", Py_NewRef(stream)) == 0 &&
                       uform_py_dict_steal_item(keywords, "max_version", Py_NewRef(version)) == 0;
    Py_XDECREF(version);
    PyObject* const arguments = packed ? PyTuple_New(0) : NULL;
    PyObject* capsule = arguments ? PyObject_Call(exporter, arguments, keywords) : NULL;
    if (!capsule && arguments && PyErr_ExceptionMatches(PyExc_TypeError)) {
        PyErr_Clear();
        if (PyDict_DelItemString(keywords, "max_version") == 0) capsule = PyObject_Call(exporter, arguments, keywords);
    }
    Py_XDECREF(arguments);
    Py_XDECREF(keywords);
    return capsule;
}

/**
 *  @brief Consumes the DLPack export of @p object into @p view, whose @c obj becomes the capsule,
 *      so releasing the view hands the tensor back to its producer.
 *  @return 1 when it does, 0 when @p object exports no DLPack, or -1 with an exception set.
 *
 *  Only C-contiguous tensors of DLPack 1.x cross. Memory the host may read always does, and device
 *  memory only on the GPU the VRAM @p arena lives on, since the binding reads the other inputs
 *  itself; a NULL @p arena takes host memory alone. The view is read-only whatever the read-only
 *  flag says, as no input is written.
 */
static int dlpack_acquire(PyObject* object, PyObject* arena, Py_buffer* view) {
    PyObject* const exporter = PyObject_GetAttrString(object, "__dlpack__");
    if (!exporter) {
        PyErr_Clear();
        return 0;
    }
    PyObject* const stream = dlpack_stream_argument(object, arena);
    PyObject* const capsule = stream ? dlpack_capsule_request(exporter, stream) : NULL;
    Py_XDECREF(stream);
    Py_DECREF(exporter);
    // A producer may refuse this export and still offer the buffer protocol, as NumPy before 2.1
    // does for read-only arrays
    if (!capsule && PyErr_ExceptionMatches(PyExc_BufferError)) {
        PyErr_Clear();
        return 0;
    }
    if (!capsule) return -1;
    int const versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    void* const managed = PyCapsule_GetPointer(capsule, versioned ? "dltensor_versioned" : "dltensor");
    if (!managed) {
        Py_DECREF(capsule);
        return -1;
    }
    if (versioned && ((DLManagedTensorVersioned*)managed)->version.major != DLPACK_MAJOR_VERSION) {
        // Another major version may lay its tensor out differently, so only its deleter is safe
        DLManagedTensorVersioned* const foreign = (DLManagedTensorVersioned*)managed;
        uint32_t const major = foreign->version.major;
        if (PyCapsule_SetName(capsule, "used_dltensor_versioned") == 0 && foreign->deleter) foreign->deleter(foreign);
        Py_DECREF(capsule);
        PyErr_Format(PyExc_BufferError, "DLPack %u.x is not 1.x", (unsigned)major);
        return -1;
    }
    DLTensor const* const tensor = versioned ? &((DLManagedTensorVersioned*)managed)->dl_tensor
                                             : &((DLManagedTensor*)managed)->dl_tensor;
    DLDeviceType const device = tensor->device.device_type;
    int const host_readable = device == kDLCPU || device == kDLCUDAHost || device == kDLROCMHost ||
                              device == kDLCUDAManaged;
    DLDevice accepted = {kDLCPU, 0};
    int const on_device = host_readable ? 0 : arena_device(arena, &accepted);
    if (on_device < 0) {
        Py_DECREF(capsule);
        return -1;
    }
    char const* format = NULL;
    for (size_t row = 0; row != sizeof(dlpack_formats) / sizeof(*dlpack_formats) && !format; ++row)
        if (dlpack_formats[row].dtype.code == tensor->dtype.code &&
            dlpack_formats[row].dtype.bits == tensor->dtype.bits && tensor->dtype.lanes == 1)
            format = dlpack_formats[row].format;
    Py_ssize_t items = 1;
    int contiguous = tensor->ndim >= 0;
    for (int axis = tensor->ndim - 1; axis >= 0; --axis) {
        contiguous = contiguous && tensor->shape[axis] >= 0 &&
                     (!tensor->strides || tensor->shape[axis] < 2 || tensor->strides[axis] == items);
        items *= (Py_ssize_t)tensor->shape[axis];
    }
    int failed = 0;
    int const same_vendor = device == accepted.device_type ||
                            (device == kDLCUDA && accepted.device_type == kDLCUDAManaged); // managed reads either
    if (device == kDLMetal)
        failed = uform_py_raise(uform_status_unsupported_k, "Metal tensors don't cross DLPack yet; pass host memory");
    else if (!host_readable && !(on_device && same_vendor && tensor->device.device_id == accepted.device_id))
        failed = uform_py_raise(uform_status_device_k, "this input cannot take memory of DLPack device (%d, %d)",
                                (int)device, (int)tensor->device.device_id);
    else if (!format)
        failed = uform_py_raise(uform_status_unsupported_layout_k, "the DLPack tensor holds items no format names");
    else if (!contiguous)
        failed = uform_py_raise(uform_status_unsupported_layout_k, "the DLPack tensor must be C-contiguous");
    Py_ssize_t* const extents = failed ? NULL
                                       : (Py_ssize_t*)PyMem_Malloc((size_t)(tensor->ndim + 1) * sizeof(Py_ssize_t));
    if (!failed && !extents) failed = (PyErr_NoMemory(), -1);
    if (!failed && (PyCapsule_SetContext(capsule, extents) != 0 ||
                    PyCapsule_SetName(capsule, versioned ? "used_dltensor_versioned" : "used_dltensor") != 0 ||
                    PyCapsule_SetDestructor(capsule, dlpack_capsule_release) != 0)) {
        PyMem_Free(extents);
        failed = -1;
    }
    if (failed) {
        Py_DECREF(capsule);
        return -1;
    }
    for (int axis = 0; axis != tensor->ndim; ++axis) extents[axis] = (Py_ssize_t)tensor->shape[axis];
    Py_ssize_t const item_bytes = tensor->dtype.bits / 8;
    *view = (Py_buffer) {.buf = (char*)tensor->data + tensor->byte_offset,
                         .obj = capsule,
                         .len = items * item_bytes,
                         .itemsize = item_bytes,
                         .readonly = 1,
                         .ndim = tensor->ndim,
                         .format = (char*)format,
                         .shape = extents};
    return 1;
}

/**
 *  @brief Reads the CUDA Array Interface of @p object into @p view, for a call on a CUDA @p arena:
 *      version 3, C-contiguous and unmasked, the arena's first stream waiting on the one it names.
 *  @return 1 when it does, 0 when @p object or @p arena takes no part in it, or -1 with an error.
 */
static int cuda_array_interface_acquire(PyObject* object, PyObject* arena, Py_buffer* view) {
    DLDevice accepted = {kDLCPU, 0};
    int const on_device = arena_device(arena, &accepted);
    if (on_device <= 0 || (accepted.device_type != kDLCUDA && accepted.device_type != kDLCUDAManaged)) return on_device;
    if (!PyObject_HasAttrString(object, "__cuda_array_interface__")) return 0;
    PyObject* const exported = PyObject_GetAttrString(object, "__cuda_array_interface__");
    if (!exported) return -1;
    PyObject* const data = PyDict_Check(exported) ? PyDict_GetItemString(exported, "data") : NULL;
    PyObject* const shape = PyDict_Check(exported) ? PyDict_GetItemString(exported, "shape") : NULL;
    PyObject* const typestr = PyDict_Check(exported) ? PyDict_GetItemString(exported, "typestr") : NULL;
    PyObject* const strides = PyDict_Check(exported) ? PyDict_GetItemString(exported, "strides") : NULL;
    PyObject* const mask = PyDict_Check(exported) ? PyDict_GetItemString(exported, "mask") : NULL;
    PyObject* const stream = PyDict_Check(exported) ? PyDict_GetItemString(exported, "stream") : NULL;
    char const* const type = typestr && PyUnicode_Check(typestr) ? PyUnicode_AsUTF8(typestr) : NULL;
    int failed = 0;
    if (!data || !PyTuple_Check(data) || PyTuple_GET_SIZE(data) != 2 || !shape || !PyTuple_Check(shape) || !type)
        failed = uform_py_raise(uform_status_invalid_argument_k,
                                "the CUDA Array Interface lacks data, shape or typestr");
    else if ((strides && strides != Py_None) || (mask && mask != Py_None))
        failed = uform_py_raise(uform_status_unsupported_layout_k, "the CUDA array must be C-contiguous and unmasked");
    // The interface names a stream as an integer: a handle, or 1 and 2 for default streams, never 0
    void* const producer = stream && stream != Py_None && !failed ? PyLong_AsVoidPtr(stream) : NULL;
    if (!failed && stream && stream != Py_None && !producer)
        failed = PyErr_Occurred() ? -1 : uform_py_raise(uform_status_invalid_argument_k, "a CUDA array stream of 0");
    char const* format = NULL;
    if (!failed && strlen(type) == 3 && (type[0] == '<' || type[0] == '|')) {
        uint8_t const code = type[1] == 'u' ? kDLUInt : type[1] == 'i' ? kDLInt : type[1] == 'f' ? kDLFloat : 0xFF;
        for (size_t row = 0; row != sizeof(dlpack_formats) / sizeof(*dlpack_formats) && !format; ++row)
            if (dlpack_formats[row].dtype.code == code && dlpack_formats[row].dtype.bits == (type[2] - '0') * 8)
                format = dlpack_formats[row].format;
    }
    if (!failed && !format)
        failed = uform_py_raise(uform_status_unsupported_layout_k, "the CUDA array holds items no format names");
    Py_ssize_t const item_bytes = failed ? 0 : (Py_ssize_t)(type[2] - '0');
    Py_ssize_t items = 1;
    for (Py_ssize_t axis = 0; !failed && axis != PyTuple_GET_SIZE(shape); ++axis) {
        Py_ssize_t const extent = PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, axis));
        failed = extent < 0
                     ? (PyErr_Occurred() ? -1 : uform_py_raise(uform_status_shape_mismatch_k, "a negative extent"))
                     : 0;
        items *= extent;
    }
    void* const address = failed ? NULL : PyLong_AsVoidPtr(PyTuple_GET_ITEM(data, 0));
    if (!failed && PyErr_Occurred()) failed = -1;
    if (!failed && producer > (void*)2) { // a handle names its device, unlike the default streams 1 and 2
        uform_outcome_t outcome = {0};
        uform_executor_accepts(&(uform_executor_accepts_t) {.executor = arena_executor(arena),
                                                            .position = 0,
                                                            .stream = producer,
                                                            .outcome = &outcome});
        failed = outcome.status == uform_status_config_mismatch_k
                     ? uform_py_raise(uform_status_device_k, "the CUDA array's stream sits on another device")
                     : uform_py_raise_on_outcome(&outcome);
    }
    uform_status_t const followed = failed ? uform_status_success_k : arena_follow((Arena*)arena, NULL, producer);
    if (followed != uform_status_success_k) // the call reads after the producer
        failed = uform_py_raise(followed, "the arena's stream could not follow the CUDA array's producer");
    // A tuple owns the exporter, so releasing the view calls no buffer hook it never armed
    PyObject* const owner = failed ? NULL : PyTuple_Pack(1, object);
    Py_DECREF(exported);
    if (failed || !owner) return -1;
    *view = (Py_buffer) {.buf = address,
                         .obj = owner,
                         .len = items * item_bytes,
                         .itemsize = item_bytes,
                         .readonly = 1,
                         .ndim = 1,
                         .format = (char*)format};
    return 1;
}

#pragma endregion DLPack

#pragma region Arenas

static char const doc_Arena[] =                                                                //
    "Arena(executor=None, /)\n"                                                                //
    "\n"                                                                                       //
    "The memory every output whose length scales with a call lands in - embeddings, states\n"  //
    "and their offsets - which a verb makes afresh when given none, and which each call\n"     //
    "naming it rewinds, so a steady loop allocates once.\n"                                    //
    "\n"                                                                                       //
    "The executor makes it where its calls compute, host pages or the memory of its GPU,\n"    //
    "and none makes host pages for calls on the calling thread. Results are zero-copy views\n" //
    "over it until the next call handed it rewinds it, after which they raise `ValueError`.\n" //
    "It serves one call at a time, and a call finding another one holding it, or a view of\n"  //
    "a result still alive, raises `ContendedError`, a `BufferError`. A deferred call holds\n"  //
    "it until it is final. Results over memory the host may not read export `__dlpack__`,\n"   //
    "and host reads of them raise `BufferError`. A GPU arena serves only calls on the\n"       //
    "executor that made it, which the arena holds, so that executor is freed only once the\n"  //
    "arena and every result over it are gone. Calls `uform_arena_init`.\n"                     //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor, optional): The executor whose calls it serves; None serves the\n"   //
    "    calling thread.\n"                                                                    //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> model = uform.Model.load_ram_buffer(b'any bytes the stub accepts')\n"               //
    "  >>> arena = uform.Arena()\n"                                                            //
    "  >>> first = model.text_encode(['first'], arena)  # stale once the next call rewinds\n"  //
    "  >>> memoryview(model.text_encode(['second'], arena).embeddings).shape\n"                //
    "  (1, 64)";

static PyObject* Arena_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    Py_ssize_t const positional_args_count = PyTuple_GET_SIZE(args);
    if (positional_args_count > 1 || (keywords && PyDict_GET_SIZE(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Arena() takes one optional positional argument, the executor");
        return NULL;
    }
    PyObject* const executor_object = positional_args_count ? PyTuple_GET_ITEM(args, 0) : Py_None;
    uform_executor_t executor = NULL;
    if (uform_py_executor_optional(executor_object, &executor) != 0) return NULL;

    uform_arena_t handle = NULL;
    uform_outcome_t outcome = {0};
    uform_arena_init_t arguments = {.arena = &handle, .executor = executor, .outcome = &outcome};
    uform_arena_init(&arguments);
    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    Arena* const arena = (Arena*)type->tp_alloc(type, 0);
    if (!arena) {
        uform_arena_free(handle); // a new arena has no export open, so it frees
        return NULL;
    }
    arena->handle = handle;
    if (executor_object != Py_None) arena->executor = Py_NewRef(executor_object); // freed after the arena
    return (PyObject*)arena;
}

static void Arena_dealloc(Arena* self);
static uform_status_t arena_follow_unfollowed(Arena* arena);

PyTypeObject ArenaType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.Arena",
    .tp_doc = doc_Arena,
    .tp_basicsize = sizeof(Arena),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Arena_tp_new,
    .tp_dealloc = (destructor)Arena_dealloc,
};

int uform_py_call_hold_arena(uform_py_call_t* call, uform_py_receiver_t const* receiver, PyObject* object) {
    if (!object || object == Py_None) {
        int const bound = receiver->origin != receiver->target; // an `On` or `DeferredOn` names its executor
        call->arena = bound ? PyObject_CallOneArg((PyObject*)&ArenaType, ((On*)receiver->origin)->executor)
                            : PyObject_CallNoArgs((PyObject*)&ArenaType);
        return call->arena ? 0 : -1;
    }
    if (!PyObject_TypeCheck(object, &ArenaType)) {
        PyErr_Format(PyExc_TypeError, "the 'arena' argument must be an Arena, got %S", Py_TYPE(object));
        return -1;
    }
    // Dropped exports still open wait for their consumers' kernels, which may still read the memory
    uform_status_t const followed = arena_follow_unfollowed((Arena*)object);
    if (followed != uform_status_success_k)
        return uform_py_raise(followed, "the arena's stream could not wait for a dropped export's consumer");
    call->arena = Py_NewRef(object);
    return 0;
}

/** The rank of every output in an arena: rows by columns. */
enum { arena_buffer_max_rank_k = 2 };

/** One output a call left in an arena, exported zero-copy, and the base of every result
 *  that is one. */
typedef struct ArenaBuffer {
    PyObject ob_base;

    /** The arena the output lives in, kept alive so the memory outlives every other name for it. */
    PyObject* arena;

    /** The arena's generation the output was made under; a later one has rewound it. */
    uint64_t generation;

    /** Exports of the arena this object opened, one per read of @c __cuda_array_interface__, which
     *  has no release of its own, so they close as the object dies. */
    size_t cuda_array_interface_reads;

    void const* data;
    char const* format;
    Py_ssize_t item_bytes;
    int rank;
    Py_ssize_t shape[arena_buffer_max_rank_k];
    Py_ssize_t strides[arena_buffer_max_rank_k];
} ArenaBuffer;

/** The format of rows whose scalar no buffer format names, @c bf16 among them, read as their bytes:
 *  told apart from @c u8 rows by its address, since only DLPack and buffers can take raw bytes. */
static char const opaque_rows_format[] = "B";

/** A ragged run of rows in an arena, and the offsets around its items. */
typedef struct RowsTape {
    ArenaBuffer rows;

    /** The @c (count + 1,) @c ArenaBuffer of offsets into the rows. */
    PyObject* offsets;
} RowsTape;

/** Where the outputs of @p buffer live, as the engine reports its arena; 0, or -1 with
 *  an error set. */
static int arena_buffer_describe(ArenaBuffer const* buffer, uform_arena_description_t* description) {
    uform_outcome_t outcome = {0};
    uform_arena_describe(&(uform_arena_describe_t) {.arena = ((Arena*)buffer->arena)->handle,
                                                    .outcome = &outcome,
                                                    .description = description});
    return uform_py_raise_on_outcome(&outcome);
}

/**
 *  @brief Opens one export of the arena the output @p buffer reads, then checks @p buffer is of
 *      its current generation, so no call rewinds the memory until @c uform_py_view_export_close.
 *      Every export of an output goes through it.
 *  @return 0, or -1 raising ValueError for a result a later call made stale.
 */
static int uform_py_view_export_open(ArenaBuffer const* buffer) {
    uform_arena_t const arena = ((Arena*)buffer->arena)->handle;
    uform_status_t const opened = uform_arena_export_open(arena);
    // A call holding the arena has already moved it past every earlier result
    if (opened == uform_status_contended_k) return result_stale();
    if (opened != uform_status_success_k) return uform_py_raise(opened, "the arena could not count another export");
    if (buffer->generation == uform_arena_generation(arena)) return 0;
    uform_arena_export_close(arena);
    return result_stale();
}

/** Closes one export @c uform_py_view_export_open opened on the arena @p buffer reads. */
static void uform_py_view_export_close(ArenaBuffer const* buffer) {
    uform_arena_export_close(((Arena*)buffer->arena)->handle);
}

/** Whether host code may read the pages of the arena behind @p buffer: 1 or 0, or -1 with an
 *  exception set. */
static int uform_py_host_readable(ArenaBuffer const* buffer) {
    uform_arena_description_t description = {0};
    if (arena_buffer_describe(buffer, &description) != 0) return -1;
    return description.host_access == uform_host_access_readable_k;
}

static int ArenaBuffer_getbuffer(ArenaBuffer* self, Py_buffer* view, int flags) {
    view->obj = NULL;
    int const readable = uform_py_host_readable(self);
    if (readable < 0) return -1;
    if (!readable) {
        PyErr_SetString(PyExc_BufferError, "the host may not read the output's memory; read `__dlpack__`");
        return -1;
    }
    if (uform_py_view_export_open(self) != 0) return -1;
    Py_ssize_t bytes = self->item_bytes;
    for (int axis = 0; axis != self->rank; ++axis) bytes *= self->shape[axis];
    view->obj = Py_NewRef((PyObject*)self);
    view->buf = (void*)self->data;
    view->len = bytes;
    view->readonly = 0;
    view->itemsize = self->item_bytes; // describes the cells the shape counts, even when no format was asked for
    view->format = (flags & PyBUF_FORMAT) == PyBUF_FORMAT ? (char*)self->format : NULL;
    view->ndim = self->rank;
    view->shape = (flags & PyBUF_ND) == PyBUF_ND ? self->shape : NULL;
    view->strides = (flags & PyBUF_STRIDES) == PyBUF_STRIDES ? self->strides : NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    return 0;
}

/** A tuple of the first @p rank of @p extents; a new reference, or NULL. */
static PyObject* extents_to_tuple(Py_ssize_t const* extents, int rank) {
    PyObject* const tuple = PyTuple_New(rank);
    for (int axis = 0; tuple && axis != rank; ++axis) {
        PyObject* const extent = PyLong_FromSsize_t(extents[axis]);
        if (!extent) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, axis, extent);
    }
    return tuple;
}

/** The DLPack device the outputs of @p buffer live on; 0, or -1 with an exception set. */
static int arena_buffer_device(ArenaBuffer const* buffer, DLDevice* device) {
    uform_arena_description_t description = {0};
    if (arena_buffer_describe(buffer, &description) != 0) return -1;
    *device = uform_py_dlpack_device(&description);
    return 0;
}

static PyObject* ArenaBuffer_get_cuda_array_interface(ArenaBuffer* self, void* closure) {
    DLDevice device;
    if (arena_buffer_device(self, &device) != 0) return NULL;
    if (device.device_type != kDLCUDA && device.device_type != kDLCUDAManaged) {
        PyErr_SetString(PyExc_AttributeError, "only CUDA device and managed memory cross the CUDA Array Interface");
        return NULL;
    }
    if (uform_py_view_export_open(self) != 0) return NULL; // a consumer reads the pointer while it holds this object
    ++self->cuda_array_interface_reads;
    char const kind = strchr("bi", *self->format) ? 'i' : strchr("efd", *self->format) ? 'f' : 'u';
    char const typestr[3] = {self->item_bytes == 1 ? '|' : '<', kind, (char)('0' + self->item_bytes)};
    PyObject* const address = PyLong_FromVoidPtr((void*)self->data);
    PyObject* const data = address ? PyTuple_Pack(2, address, Py_False) : NULL;
    Py_XDECREF(address);
    PyObject* const array_interface = data ? PyDict_New() : NULL;
    if (!array_interface) {
        Py_XDECREF(data);
        return NULL;
    }
    if (uform_py_dict_steal_item(array_interface, "data", data) != 0 ||
        uform_py_dict_steal_item(array_interface, "shape", extents_to_tuple(self->shape, self->rank)) != 0 ||
        uform_py_dict_steal_item(array_interface, "strides", extents_to_tuple(self->strides, self->rank)) != 0 ||
        uform_py_dict_steal_item(array_interface, "typestr", PyUnicode_FromStringAndSize(typestr, 3)) != 0 ||
        uform_py_dict_steal_item(array_interface, "stream", Py_NewRef(Py_None)) != 0 || // the writing call is final
        uform_py_dict_steal_item(array_interface, "version", PyLong_FromLong(3)) != 0) {
        Py_DECREF(array_interface);
        return NULL;
    }
    return array_interface;
}

/** What a DLPack capsule over an @c ArenaBuffer owns: the managed tensor, its extents and strides,
 *  and the buffer, whose arena counts the export until the consumer lets go. */
typedef struct dlpack_export_t {
    union {
        DLManagedTensor legacy;
        DLManagedTensorVersioned versioned;
    } managed;
    ArenaBuffer* buffer;

    /** The stream the consumer's kernels read the tensor on, which the arena's stream follows
     *  before the export closes, or NULL when the consumer named none. */
    void* consumer_stream;

    /** The next export on the arena's @c unfollowed stack, once this one waits there. */
    struct dlpack_export_t* next;

    int64_t shape[arena_buffer_max_rank_k];
    int64_t strides[arena_buffer_max_rank_k];
} dlpack_export_t;

/** Pushes the exports @p first through @p last onto the @c unfollowed stack of @p arena. */
static void arena_push_unfollowed(Arena* arena, dlpack_export_t* first, dlpack_export_t* last) {
    last->next = atomic_load(&arena->unfollowed);
    while (!atomic_compare_exchange_weak(&arena->unfollowed, &last->next, first)) {}
}

/** Follows and closes every export on the @c unfollowed stack of @p arena, answering the first
 *  failed follow's status with that export and the rest pushed back. */
static uform_status_t arena_follow_unfollowed(Arena* arena) {
    dlpack_export_t* exported = atomic_exchange(&arena->unfollowed, NULL);
    while (exported) {
        uform_status_t const followed = arena_follow(arena, NULL, exported->consumer_stream);
        if (followed != uform_status_success_k) {
            dlpack_export_t* last = exported;
            while (last->next) last = last->next;
            arena_push_unfollowed(arena, exported, last);
            return followed;
        }
        dlpack_export_t* const next = exported->next;
        uform_arena_export_close(arena->handle);
        PyMem_Free(exported);
        exported = next;
    }
    return uform_status_success_k;
}

static void Arena_dealloc(Arena* self) {
    uform_status_t const followed = arena_follow_unfollowed(self);
    if (followed != uform_status_success_k) {
        // The consumers' streams can't be followed, so the arena waits for its own work instead
        uform_outcome_t outcome = {0};
        uform_executor_t const executor = arena_executor((PyObject*)self);
        Py_BEGIN_ALLOW_THREADS;
        if (executor)
            uform_executor_drain(
                &(uform_executor_drain_t) {.executor = executor, .outcome = &outcome, .mood = uform_mood_parked_k});
        Py_END_ALLOW_THREADS;
        for (dlpack_export_t* exported = atomic_exchange(&self->unfollowed, NULL); exported;) {
            dlpack_export_t* const next = exported->next;
            uform_arena_export_close(self->handle);
            PyMem_Free(exported);
            exported = next;
        }
        PyObject* const pending = PyErr_GetRaisedException();
        uform_py_raise(followed, "the arena was freed before its stream could wait for a dropped export's consumer");
        PyErr_WriteUnraisable(NULL);
        PyErr_SetRaisedException(pending);
    }
    Py_BEGIN_ALLOW_THREADS;
    uform_arena_free(self->handle); // every export holds this object or waits on its stack, so none is open
    Py_END_ALLOW_THREADS;
    Py_XDECREF(self->executor);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Closes the export, or leaves it for the arena's next user to follow and close, taking the
 *  interpreter, as a consumer may delete from a thread of its own. */
static void dlpack_export_release(dlpack_export_t* exported) {
    PyGILState_STATE const state = PyGILState_Ensure();
    // The consumer's kernels may still read the memory, so the arena's stream waits for them first
    ArenaBuffer* const buffer = exported->buffer;
    Arena* const arena = (Arena*)buffer->arena;
    if (arena_follow(arena, NULL, exported->consumer_stream) == uform_status_contended_k)
        // A call on another arena holds the executor: the export stays open until the arena's next
        // user follows the stream, and the buffer goes, as the stack must not keep the arena alive
        arena_push_unfollowed(arena, exported, exported);
    else {
        uform_arena_export_close(arena->handle);
        PyMem_Free(exported);
    }
    Py_DECREF(buffer);
    PyGILState_Release(state);
}

static void dlpack_legacy_deleter(DLManagedTensor* self) { dlpack_export_release(self->manager_ctx); }

static void dlpack_versioned_deleter(DLManagedTensorVersioned* self) { dlpack_export_release(self->manager_ctx); }

/** Releases an export no consumer took; a consumer renames the capsule and owns it. */
static void dlpack_export_capsule_release(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        DLManagedTensor* const managed = PyCapsule_GetPointer(capsule, "dltensor");
        managed->deleter(managed);
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        DLManagedTensorVersioned* const managed = PyCapsule_GetPointer(capsule, "dltensor_versioned");
        managed->deleter(managed);
    }
}

static PyObject* ArenaBuffer_dlpack(ArenaBuffer* self, PyObject* const* args, Py_ssize_t positional, PyObject* names) {
    // The call that wrote the output is final, so `stream` names no work left to order after.
    PyObject *stream = Py_None, *max_version = Py_None, *dl_device = Py_None, *copy = Py_None;
    if (positional) return PyErr_Format(PyExc_TypeError, "__dlpack__() takes keyword arguments only");
    for (Py_ssize_t keyword = 0; names && keyword != PyTuple_GET_SIZE(names); ++keyword) {
        PyObject* const name = PyTuple_GET_ITEM(names, keyword);
        PyObject** const slot = PyUnicode_CompareWithASCIIString(name, "stream") == 0        ? &stream
                                : PyUnicode_CompareWithASCIIString(name, "max_version") == 0 ? &max_version
                                : PyUnicode_CompareWithASCIIString(name, "dl_device") == 0   ? &dl_device
                                : PyUnicode_CompareWithASCIIString(name, "copy") == 0        ? &copy
                                                                                             : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "__dlpack__() got an unexpected keyword argument '%S'", name);
        *slot = args[keyword];
    }
    int const copied = copy == Py_None ? 0 : PyObject_IsTrue(copy);
    if (copied < 0) return NULL;
    if (copied) return PyErr_Format(PyExc_BufferError, "an arena output exports only in place, never a copy");
    DLDevice device;
    if (arena_buffer_device(self, &device) != 0) return NULL;
    void* consumer_stream = NULL;
    if (stream != Py_None) { // checked as the standard asks, and followed once the consumer is done
        if (!PyLong_Check(stream) || PyBool_Check(stream))
            return PyErr_Format(PyExc_TypeError, "stream must be an integer or None, got %S", Py_TYPE(stream));
        long long const handle = PyLong_AsLongLong(stream);
        if (handle == -1 && PyErr_Occurred()) return NULL;
        int const cuda = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
        int const rocm = device.device_type == kDLROCM;
        if (handle != -1 && !(cuda && handle > 0) && !(rocm && handle >= 0))
            return PyErr_Format(PyExc_ValueError, "stream %lld names no stream on DLPack device (%d, %d)", handle,
                                (int)device.device_type, (int)device.device_id);
        if (handle != -1 && !arena_executor(self->arena))
            return uform_py_raise(uform_status_uninitialized_k, "the executor is closed"), NULL;
        if ((cuda && handle > 2) || (rocm && handle > 0)) { // a handle rather than a default stream
            uform_outcome_t outcome = {0};
            uform_executor_accepts(&(uform_executor_accepts_t) {.executor = arena_executor(self->arena),
                                                                .position = 0,
                                                                .stream = PyLong_AsVoidPtr(stream),
                                                                .outcome = &outcome});
            if (outcome.status == uform_status_config_mismatch_k)
                return PyErr_Format(PyExc_ValueError, "stream %lld sits on another device than the output", handle);
            if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
        }
        if (handle != -1) consumer_stream = PyLong_AsVoidPtr(stream);
    }
    if (dl_device != Py_None) {
        if (!PyTuple_Check(dl_device) || PyTuple_GET_SIZE(dl_device) != 2)
            return PyErr_Format(PyExc_TypeError, "dl_device must be a (device_type, device_id) tuple");
        long const requested_type = PyLong_AsLong(PyTuple_GET_ITEM(dl_device, 0));
        long const requested_id = PyLong_AsLong(PyTuple_GET_ITEM(dl_device, 1));
        if (PyErr_Occurred()) return NULL;
        if (requested_type != (long)device.device_type || requested_id != (long)device.device_id)
            return PyErr_Format(PyExc_BufferError, "the output lives on DLPack device (%d, %d)",
                                (int)device.device_type, (int)device.device_id);
    }
    long major = 0;
    if (max_version != Py_None) {
        if (!PyTuple_Check(max_version) || PyTuple_GET_SIZE(max_version) != 2)
            return PyErr_Format(PyExc_TypeError, "max_version must be a (major, minor) tuple");
        major = PyLong_AsLong(PyTuple_GET_ITEM(max_version, 0));
        if (PyErr_Occurred()) return NULL;
    }
    DLDataType dtype = {0, 0, 0};
    for (size_t row = 0; row != sizeof(dlpack_formats) / sizeof(*dlpack_formats); ++row)
        if (*dlpack_formats[row].format == *self->format) dtype = dlpack_formats[row].dtype;
    if (!dtype.bits) return PyErr_Format(PyExc_BufferError, "format '%s' has no DLPack type", self->format);

    dlpack_export_t* const exported = (dlpack_export_t*)PyMem_Calloc(1, sizeof(dlpack_export_t));
    if (!exported) return PyErr_NoMemory();
    if (uform_py_view_export_open(self) != 0) {
        PyMem_Free(exported);
        return NULL;
    }
    exported->buffer = (ArenaBuffer*)Py_NewRef((PyObject*)self);
    exported->consumer_stream = consumer_stream;
    for (int axis = 0; axis != self->rank; ++axis) {
        exported->shape[axis] = self->shape[axis];
        exported->strides[axis] = self->strides[axis] / self->item_bytes;
    }
    DLTensor const tensor = {(void*)self->data, device, self->rank, dtype, exported->shape, exported->strides, 0};
    void* managed = NULL;
    if (major >= 1) {
        exported->managed.versioned = (DLManagedTensorVersioned) {{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION},
                                                                  exported,
                                                                  dlpack_versioned_deleter,
                                                                  0,
                                                                  tensor};
        managed = &exported->managed.versioned;
    }
    else {
        exported->managed.legacy = (DLManagedTensor) {tensor, exported, dlpack_legacy_deleter};
        managed = &exported->managed.legacy;
    }
    PyObject* const capsule = PyCapsule_New(managed, major >= 1 ? "dltensor_versioned" : "dltensor",
                                            dlpack_export_capsule_release);
    if (!capsule) dlpack_export_release(exported);
    return capsule;
}

static PyObject* ArenaBuffer_dlpack_device(ArenaBuffer* self, PyObject* unused) {
    DLDevice device;
    if (self->generation != uform_arena_generation(((Arena*)self->arena)->handle)) return result_stale(), NULL;
    if (arena_buffer_device(self, &device) != 0) return NULL;
    PyObject* const pair = PyTuple_New(2);
    if (!pair) return NULL;
    PyTuple_SET_ITEM(pair, 0, PyLong_FromLong((long)device.device_type));
    PyTuple_SET_ITEM(pair, 1, PyLong_FromLong((long)device.device_id));
    if (PyTuple_GET_ITEM(pair, 0) && PyTuple_GET_ITEM(pair, 1)) return pair;
    Py_DECREF(pair);
    return NULL;
}

/** The levels an Arrow export of an output nests: offsets around rows, rows of columns, values. */
enum { arrow_export_max_levels_k = 3 };

/**
 *  @brief What one Arrow export over an @c ArenaBuffer owns: its arrays and schemas level by level,
 *      outermost first, and the export it holds on the buffer's arena while any array is alive.
 *
 *  Every array and schema counts in @c alive, as a consumer may move a child out and release it
 *  after its parent, and the last release frees the block.
 */
typedef struct arrow_export_t {
    ArenaBuffer* buffer;
    int levels;
    atomic_int alive;
    struct ArrowArray arrays[arrow_export_max_levels_k];
    struct ArrowArray* array_children[arrow_export_max_levels_k];
    void const* array_buffers[arrow_export_max_levels_k][2];
    struct ArrowSchema schemas[arrow_export_max_levels_k];
    struct ArrowSchema* schema_children[arrow_export_max_levels_k];
    char formats[arrow_export_max_levels_k][24];
} arrow_export_t;

/** Drops one of the arrays and schemas of @p exported, freeing it after the last, from whichever
 *  thread its consumer releases on. */
static void arrow_export_drop(arrow_export_t* exported) {
    if (atomic_fetch_sub(&exported->alive, 1) != 1) return;
    PyGILState_STATE const state = PyGILState_Ensure();
    if (exported->buffer) {
        uform_arena_export_close(((Arena*)exported->buffer->arena)->handle);
        Py_DECREF(exported->buffer);
    }
    PyMem_RawFree(exported);
    PyGILState_Release(state);
}

static void arrow_export_array_release(struct ArrowArray* array) {
    for (int64_t child = 0; child != array->n_children; ++child)
        if (array->children[child]->release) array->children[child]->release(array->children[child]);
    array->release = NULL;
    arrow_export_drop((arrow_export_t*)array->private_data);
}

static void arrow_export_schema_release(struct ArrowSchema* schema) {
    for (int64_t child = 0; child != schema->n_children; ++child)
        if (schema->children[child]->release) schema->children[child]->release(schema->children[child]);
    schema->release = NULL;
    arrow_export_drop((arrow_export_t*)schema->private_data);
}

/** The Arrow format of the values of @p buffer, or NULL for rows no Arrow type names, like
 *  @c bf16 ones. */
static char const* arrow_value_format(ArenaBuffer const* buffer) {
    if (buffer->format == opaque_rows_format) return NULL;
    switch (*buffer->format) {
    case 'B': return "C";
    case 'b': return "c";
    case 'I': return "I";
    case 'i': return "i";
    case 'Q':
    case 'N': return "L";
    case 'e': return "e";
    case 'f': return "f";
    case 'd': return "g";
    default: return NULL;
    }
}

/**
 *  @brief Lays @p buffer out as Arrow: a run of values, rows as fixed-size lists of their
 *      columns, and a @c RowsTape as a large list of those, its @c size_t offsets read as
 *      @c int64 in place.
 *  @return A new export holding @p buffer and one export on its arena, or NULL with an error.
 */
static arrow_export_t* arrow_export_new(ArenaBuffer* buffer) {
    char const* const value_format = arrow_value_format(buffer);
    if (!value_format) {
        PyErr_Format(PyExc_BufferError, "the output's values have no Arrow type");
        return NULL;
    }
    ArenaBuffer const* const offsets = PyObject_TypeCheck((PyObject*)buffer, &RowsTapeType)
                                           ? (ArenaBuffer const*)((RowsTape*)buffer)->offsets
                                           : NULL;
    arrow_export_t* const exported = (arrow_export_t*)PyMem_RawCalloc(1, sizeof(arrow_export_t));
    if (!exported) return (arrow_export_t*)PyErr_NoMemory();
    if (uform_py_view_export_open(buffer) != 0) {
        PyMem_RawFree(exported);
        return NULL;
    }
    exported->buffer = (ArenaBuffer*)Py_NewRef((PyObject*)buffer);

    // Outermost first: the large list of a tape's items, each row's fixed-size list, the values
    int64_t const rows = buffer->shape[0], columns = buffer->rank == 2 ? buffer->shape[1] : 1;
    int level = 0;
    if (offsets) {
        snprintf(exported->formats[level], sizeof(exported->formats[level]), "+L");
        exported->array_buffers[level][1] = offsets->data;
        exported->arrays[level].length = offsets->shape[0] - 1;
        ++level;
    }
    if (buffer->rank == 2) {
        snprintf(exported->formats[level], sizeof(exported->formats[level]), "+w:%lld", (long long)columns);
        exported->arrays[level].length = rows;
        ++level;
    }
    snprintf(exported->formats[level], sizeof(exported->formats[level]), "%s", value_format);
    exported->array_buffers[level][1] = buffer->data;
    exported->arrays[level].length = rows * columns;
    exported->levels = ++level;
    atomic_store(&exported->alive, 2 * level);
    for (int at = 0; at != level; ++at) {
        int const nested = at + 1 != level;
        int const values = !nested;
        exported->array_children[at] = nested ? &exported->arrays[at + 1] : NULL;
        exported->schema_children[at] = nested ? &exported->schemas[at + 1] : NULL;
        struct ArrowArray* const array = &exported->arrays[at];
        array->n_buffers = values || exported->array_buffers[at][1] ? 2 : 1;
        array->buffers = exported->array_buffers[at];
        array->n_children = nested;
        array->children = nested ? &exported->array_children[at] : NULL;
        array->release = arrow_export_array_release;
        array->private_data = exported;
        exported->schemas[at] = (struct ArrowSchema) {.format = exported->formats[at],
                                                      .name = nested || !at ? "" : "item",
                                                      .n_children = nested,
                                                      .children = nested ? &exported->schema_children[at] : NULL,
                                                      .release = arrow_export_schema_release,
                                                      .private_data = exported};
    }
    return exported;
}

/** Releases a schema or array capsule no consumer took; a consumer moves the struct out and
 *  leaves its release NULL. */
static void arrow_schema_capsule_release(PyObject* capsule) {
    struct ArrowSchema* const schema = (struct ArrowSchema*)PyCapsule_GetPointer(capsule, "arrow_schema");
    if (schema->release) schema->release(schema);
    PyMem_Free(schema);
}

static void arrow_array_capsule_release(PyObject* capsule) {
    struct ArrowArray* const array = (struct ArrowArray*)PyCapsule_GetPointer(capsule, "arrow_array");
    if (array->release) array->release(array);
    PyMem_Free(array);
}

static void arrow_device_array_capsule_release(PyObject* capsule) {
    struct ArrowDeviceArray* const device_array = (struct ArrowDeviceArray*)PyCapsule_GetPointer(capsule,
                                                                                                 "arrow_device_array");
    if (device_array->array.release) device_array->array.release(&device_array->array);
    PyMem_Free(device_array);
}

/** A capsule moving @p schema out of an export; a new reference, or NULL with it released. */
static PyObject* arrow_schema_capsule(struct ArrowSchema* schema) {
    struct ArrowSchema* const moved = (struct ArrowSchema*)PyMem_Malloc(sizeof(*moved));
    PyObject* const capsule = moved ? PyCapsule_New(moved, "arrow_schema", arrow_schema_capsule_release) : NULL;
    if (!capsule) {
        PyMem_Free(moved);
        schema->release(schema);
        return moved ? NULL : PyErr_NoMemory();
    }
    *moved = *schema;
    schema->release = NULL;
    return capsule;
}

/**
 *  @brief Exports @p buffer as an Arrow schema capsule and array capsule, on a device when
 *      @p device names one, after the host check the caller made.
 *  @return A new tuple, or NULL with an exception set.
 */
static PyObject* arrow_export_pair(ArenaBuffer* buffer, DLDevice const* device) {
    arrow_export_t* const exported = arrow_export_new(buffer);
    if (!exported) return NULL;
    PyObject* const schema = arrow_schema_capsule(&exported->schemas[0]);
    struct ArrowArray* const top = &exported->arrays[0];
    PyObject* array = NULL;
    if (!schema) top->release(top);
    else if (device) {
        struct ArrowDeviceArray* const moved = (struct ArrowDeviceArray*)PyMem_Calloc(1, sizeof(*moved));
        array = moved ? PyCapsule_New(moved, "arrow_device_array", arrow_device_array_capsule_release) : NULL;
        if (array) {
            // The call that wrote the output is final, so a consumer waits on no event.
            *moved = (struct ArrowDeviceArray) {.array = *top,
                                                .device_id = device->device_type == kDLCPU ? -1 : device->device_id,
                                                .device_type = (ArrowDeviceType)device->device_type};
            top->release = NULL;
        }
        else {
            if (!moved) PyErr_NoMemory();
            PyMem_Free(moved);
            top->release(top);
        }
    }
    else {
        struct ArrowArray* const moved = (struct ArrowArray*)PyMem_Malloc(sizeof(*moved));
        array = moved ? PyCapsule_New(moved, "arrow_array", arrow_array_capsule_release) : NULL;
        if (array) *moved = *top, top->release = NULL;
        else {
            if (!moved) PyErr_NoMemory();
            PyMem_Free(moved);
            top->release(top);
        }
    }
    PyObject* const pair = array ? PyTuple_Pack(2, schema, array) : NULL;
    Py_XDECREF(schema);
    Py_XDECREF(array);
    return pair;
}

/**
 *  @brief Reads the arguments of an Arrow export: a @c requested_schema, which a producer may
 *      ignore and the consumer casts after, and, where @p extra_allowed, keywords a later
 *      version may add, refused unless @c None.
 *  @return 0, or -1 with an exception set.
 */
static int arrow_export_arguments(char const* method, PyObject* const* args, Py_ssize_t positional, PyObject* names,
                                  int extra_allowed) {
    if (positional > 1) return PyErr_Format(PyExc_TypeError, "%s() takes at most one positional argument", method), -1;
    for (Py_ssize_t keyword = 0; names && keyword != PyTuple_GET_SIZE(names); ++keyword) {
        PyObject* const name = PyTuple_GET_ITEM(names, keyword);
        if (PyUnicode_CompareWithASCIIString(name, "requested_schema") == 0 && !positional) continue;
        if (!extra_allowed)
            return PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%S'", method, name), -1;
        if (args[positional + keyword] != Py_None)
            return PyErr_Format(PyExc_NotImplementedError, "%s() does not support '%S'", method, name), -1;
    }
    return 0;
}

/** Refuses an Arrow export the host would read when it may not read the output; 0, or -1 with an
 *  exception set. */
static int arrow_export_host_check(ArenaBuffer* buffer) {
    int const readable = uform_py_host_readable(buffer);
    if (readable < 0) return -1;
    if (!readable)
        return uform_py_raise(uform_status_device_k,
                              "the host may not read the output; read `__arrow_c_device_array__`");
    return 0;
}

static PyObject* ArenaBuffer_arrow_c_array(ArenaBuffer* self, PyObject* const* args, Py_ssize_t positional,
                                           PyObject* names) {
    if (arrow_export_arguments("__arrow_c_array__", args, positional, names, 0) != 0 ||
        arrow_export_host_check(self) != 0)
        return NULL;
    return arrow_export_pair(self, NULL);
}

static PyObject* ArenaBuffer_arrow_c_device_array(ArenaBuffer* self, PyObject* const* args, Py_ssize_t positional,
                                                  PyObject* names) {
    DLDevice device;
    if (arrow_export_arguments("__arrow_c_device_array__", args, positional, names, 1) != 0 ||
        arena_buffer_device(self, &device) != 0)
        return NULL;
    return arrow_export_pair(self, &device);
}

/** What an Arrow stream over a @c RowsTape owns: the tape, whose arena counts the stream as an
 *  export until it is released, and whether its one batch was handed out. */
typedef struct arrow_stream_t {
    ArenaBuffer* buffer;
    int exhausted;
} arrow_stream_t;

static int arrow_stream_get_schema(struct ArrowArrayStream* stream, struct ArrowSchema* out) {
    arrow_stream_t* const state = (arrow_stream_t*)stream->private_data;
    PyGILState_STATE const gil = PyGILState_Ensure();
    arrow_export_t* const exported = arrow_export_new(state->buffer);
    if (exported) {
        *out = exported->schemas[0];
        exported->schemas[0].release = NULL;
        exported->arrays[0].release(&exported->arrays[0]);
    }
    else PyErr_Clear();
    PyGILState_Release(gil);
    return exported ? 0 : EIO;
}

static int arrow_stream_get_next(struct ArrowArrayStream* stream, struct ArrowArray* out) {
    arrow_stream_t* const state = (arrow_stream_t*)stream->private_data;
    out->release = NULL;
    if (state->exhausted) return 0;
    PyGILState_STATE const gil = PyGILState_Ensure();
    arrow_export_t* const exported = arrow_export_new(state->buffer);
    if (exported) {
        *out = exported->arrays[0];
        exported->arrays[0].release = NULL;
        exported->schemas[0].release(&exported->schemas[0]);
        state->exhausted = 1;
    }
    else PyErr_Clear();
    PyGILState_Release(gil);
    return exported ? 0 : EIO;
}

static char const* arrow_stream_get_last_error(struct ArrowArrayStream* stream) {
    return "the output's arena could not count another export, or memory ran out";
}

static void arrow_stream_release(struct ArrowArrayStream* stream) {
    arrow_stream_t* const state = (arrow_stream_t*)stream->private_data;
    PyGILState_STATE const gil = PyGILState_Ensure();
    uform_arena_export_close(((Arena*)state->buffer->arena)->handle);
    Py_DECREF(state->buffer);
    PyMem_RawFree(state);
    PyGILState_Release(gil);
    stream->release = NULL;
}

static void arrow_stream_capsule_release(PyObject* capsule) {
    struct ArrowArrayStream* const stream = (struct ArrowArrayStream*)PyCapsule_GetPointer(capsule,
                                                                                           "arrow_array_stream");
    if (stream->release) stream->release(stream);
    PyMem_Free(stream);
}

static PyObject* RowsTape_arrow_c_stream(ArenaBuffer* self, PyObject* const* args, Py_ssize_t positional,
                                         PyObject* names) {
    if (arrow_export_arguments("__arrow_c_stream__", args, positional, names, 0) != 0 ||
        arrow_export_host_check(self) != 0)
        return NULL;
    arrow_stream_t* const state = (arrow_stream_t*)PyMem_RawCalloc(1, sizeof(arrow_stream_t));
    struct ArrowArrayStream* const stream = state ? (struct ArrowArrayStream*)PyMem_Calloc(1, sizeof(*stream)) : NULL;
    if (!stream) {
        PyMem_RawFree(state);
        return PyErr_NoMemory();
    }
    int const opened = uform_py_view_export_open(self) == 0;
    PyObject* const capsule = opened ? PyCapsule_New(stream, "arrow_array_stream", arrow_stream_capsule_release) : NULL;
    if (!capsule) {
        if (opened) uform_py_view_export_close(self);
        PyMem_RawFree(state);
        PyMem_Free(stream);
        return NULL;
    }
    state->buffer = (ArenaBuffer*)Py_NewRef((PyObject*)self);
    *stream = (struct ArrowArrayStream) {.get_schema = arrow_stream_get_schema,
                                         .get_next = arrow_stream_get_next,
                                         .get_last_error = arrow_stream_get_last_error,
                                         .release = arrow_stream_release,
                                         .private_data = state};
    return capsule;
}

static char const doc_ArenaBuffer_dlpack[] =                                                 //
    "__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None) -> PyCapsule\n" //
    "\n"                                                                                     //
    "Exports the output in place as a DLPack capsule, on host pages or the GPU its arena\n"  //
    "lives on, valid until the next call handed the arena rewinds it. A `max_version` of\n"  //
    "1 or more asks for a versioned capsule. A `stream` handle must sit on that GPU, and\n"  //
    "once the consumer deletes the capsule, the arena's next call waits for the work\n"      //
    "queued on it before it rewinds the memory.";

static char const doc_ArenaBuffer_arrow_c_array[] =                                              //
    "__arrow_c_array__(requested_schema=None) -> tuple[PyCapsule, PyCapsule]\n"                  //
    "\n"                                                                                         //
    "Exports the output in place as an Arrow array: values, rows as fixed-size lists of their\n" //
    "columns, and a `RowsTape` as a large list of those. Memory the host may not read raises\n"  //
    "`DeviceError`; `bf16` rows have no Arrow type and raise `BufferError`.";

static char const doc_ArenaBuffer_arrow_c_device_array[] =                                       //
    "__arrow_c_device_array__(requested_schema=None, **kwargs) -> tuple[PyCapsule, PyCapsule]\n" //
    "\n"                                                                                         //
    "Exports the output in place as an Arrow device array on the device its arena lives on,\n"   //
    "laid out as `__arrow_c_array__` lays it out, with no event to wait on.";

static char const doc_ArenaBuffer_dlpack_device[] = //
    "__dlpack_device__() -> tuple[int, int]\n"      //
    "\n"                                            //
    "The DLPack device type and ID the output lives on.";

static PyMethodDef ArenaBuffer_methods[] = {
    {"__dlpack__", (PyCFunction)(void (*)(void))ArenaBuffer_dlpack, METH_FASTCALL | METH_KEYWORDS,
     doc_ArenaBuffer_dlpack},
    {"__dlpack_device__", (PyCFunction)ArenaBuffer_dlpack_device, METH_NOARGS, doc_ArenaBuffer_dlpack_device},
    {"__arrow_c_array__", (PyCFunction)(void (*)(void))ArenaBuffer_arrow_c_array, METH_FASTCALL | METH_KEYWORDS,
     doc_ArenaBuffer_arrow_c_array},
    {"__arrow_c_device_array__", (PyCFunction)(void (*)(void))ArenaBuffer_arrow_c_device_array,
     METH_FASTCALL | METH_KEYWORDS, doc_ArenaBuffer_arrow_c_device_array},
    {NULL, NULL, 0, NULL},
};

static void ArenaBuffer_dealloc(ArenaBuffer* self) {
    for (; self->cuda_array_interface_reads; --self->cuda_array_interface_reads)
        uform_arena_export_close(((Arena*)self->arena)->handle);
    Py_DECREF(self->arena);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static Py_ssize_t ArenaBuffer_length(ArenaBuffer* self) { return self->shape[0]; }

static PySequenceMethods ArenaBuffer_as_sequence = {.sq_length = (lenfunc)ArenaBuffer_length};

static void ArenaBuffer_releasebuffer(ArenaBuffer* self, Py_buffer* view) {
    uform_arena_export_close(((Arena*)self->arena)->handle);
}

static PyBufferProcs ArenaBuffer_as_buffer = {
    .bf_getbuffer = (getbufferproc)ArenaBuffer_getbuffer,
    .bf_releasebuffer = (releasebufferproc)ArenaBuffer_releasebuffer,
};

static PyGetSetDef ArenaBuffer_getset[] = {
    {"__cuda_array_interface__", (getter)ArenaBuffer_get_cuda_array_interface, NULL,
     "Version 3 of the CUDA Array Interface, present only over CUDA device and managed memory. Python-only.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static char const doc_ArenaBuffer[] =                                                        //
    "ArenaBuffer\n"                                                                          //
    "\n"                                                                                     //
    "One output a call left in an `Arena`, viewing it until it rewinds: a (rows, columns)\n" //
    "matrix for the scalars a buffer can format, and a (rows, stride) run of 'B' for the\n"  //
    "rest, `bf16` among them. Every output exports `__dlpack__` and the Arrow arrays;\n"     //
    "outputs the host may read also export the writable buffer protocol, and CUDA device\n"  //
    "and managed ones `__cuda_array_interface__`.";

PyTypeObject ArenaBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.ArenaBuffer",
    .tp_doc = doc_ArenaBuffer,
    .tp_basicsize = sizeof(ArenaBuffer),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, // Behind the results a verb returns, never constructed by name.
    .tp_dealloc = (destructor)ArenaBuffer_dealloc,
    .tp_as_sequence = &ArenaBuffer_as_sequence,
    .tp_as_buffer = &ArenaBuffer_as_buffer,
    .tp_getset = ArenaBuffer_getset,
    .tp_methods = ArenaBuffer_methods,
};

static void RowsTape_dealloc(RowsTape* self) {
    Py_XDECREF(self->offsets);
    ArenaBuffer_dealloc(&self->rows);
}

static PyMemberDef RowsTape_members[] = {
    {"offsets", Py_T_OBJECT_EX, offsetof(RowsTape, offsets), Py_READONLY,
     "(count + 1,) Apache Arrow offsets into the rows, item `i` spanning rows `offsets[i]` up to `offsets[i + 1]`."},
    {NULL, 0, 0, 0, NULL},
};

static char const doc_RowsTape_arrow_c_stream[] =                                           //
    "__arrow_c_stream__(requested_schema=None) -> PyCapsule\n"                              //
    "\n"                                                                                    //
    "Exports the tape in place as an Arrow stream of one batch, its `__arrow_c_array__`,\n" //
    "for consumers that read batches of rows, like `pyarrow.chunked_array`.";

static PyMethodDef RowsTape_methods[] = {
    {"__arrow_c_stream__", (PyCFunction)(void (*)(void))RowsTape_arrow_c_stream, METH_FASTCALL | METH_KEYWORDS,
     doc_RowsTape_arrow_c_stream},
    {NULL, NULL, 0, NULL},
};

static char const doc_RowsTape[] =                                                          //
    "RowsTape\n"                                                                            //
    "\n"                                                                                    //
    "A ragged run of rows a call wrote into an `Arena`, one row per token each item kept\n" //
    "after truncation, viewing it until it rewinds. The rows read as an `ArenaBuffer` of\n" //
    "(tokens, columns), and `offsets`, (count + 1,) unsigned words, splits them by item.\n" //
    "A run of one value per row, like the probabilities of `text_choose`, reads as a\n"     //
    "flat (values,) buffer instead.";

PyTypeObject RowsTapeType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.RowsTape",
    .tp_doc = doc_RowsTape,
    .tp_basicsize = sizeof(RowsTape),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_base = &ArenaBufferType,
    .tp_new = NULL, // Returned as the states of an encoding, never constructed by name.
    .tp_dealloc = (destructor)RowsTape_dealloc,
    .tp_members = RowsTape_members,
    .tp_methods = RowsTape_methods,
};

/** The buffer format of the scalars a @c memoryview can format, and the bytes one value takes. */
static struct {
    uform_scalar_t kind;
    char const* format;
    Py_ssize_t bytes;
} const scalar_formats[] = {
    {uform_scalar_f64_k, "d", 8}, {uform_scalar_f32_k, "f", 4}, {uform_scalar_f16_k, "e", 2},
    {uform_scalar_i32_k, "i", 4}, {uform_scalar_i8_k, "b", 1},  {uform_scalar_u8_k, "B", 1},
};

/** A new @c ArenaBuffer of @p type over @p data inside the arena @p call held, C-contiguous and
 *  of the call's generation; NULL on failure. */
static ArenaBuffer* arena_buffer_new(PyTypeObject* type, uform_py_call_t const* call, void const* data,
                                     char const* format, Py_ssize_t item_bytes, Py_ssize_t const* shape, int rank) {
    static char const empty_output = 0; // stands in for a null output, which no exporter may hand out
    ArenaBuffer* const exporter = (ArenaBuffer*)type->tp_alloc(type, 0);
    if (!exporter) return NULL;
    exporter->arena = Py_NewRef(call->arena);
    exporter->generation = call->generation;
    exporter->data = data ? data : &empty_output;
    exporter->format = format;
    exporter->item_bytes = item_bytes;
    exporter->rank = rank;
    Py_ssize_t stride = item_bytes;
    for (int axis = rank - 1; axis >= 0; --axis) {
        exporter->shape[axis] = shape[axis];
        exporter->strides[axis] = stride;
        stride *= shape[axis];
    }
    return exporter;
}

PyObject* uform_py_rows_answer(PyTypeObject* type, uform_py_call_t const* call, uform_rows_view_t const* rows,
                               size_t count) {
    char const* format = opaque_rows_format;
    Py_ssize_t item_bytes = 1;
    Py_ssize_t shape[2] = {(Py_ssize_t)count, (Py_ssize_t)rows->stride};
    for (size_t row = 0; row != sizeof(scalar_formats) / sizeof(*scalar_formats); ++row)
        if (scalar_formats[row].kind == rows->scalar &&
            rows->stride == rows->columns * (size_t)scalar_formats[row].bytes) {
            format = scalar_formats[row].format;
            item_bytes = scalar_formats[row].bytes;
            shape[1] = (Py_ssize_t)rows->columns;
        }
    return (PyObject*)arena_buffer_new(type, call, rows->data, format, item_bytes, shape, 2);
}

PyObject* uform_py_rows_tape_answer(uform_py_call_t const* call, uform_rows_tape_view_t const* states, size_t count,
                                    size_t tokens) {
    Py_ssize_t const offsets_shape[1] = {(Py_ssize_t)count + 1};
    RowsTape* rows_tape = (RowsTape*)uform_py_rows_answer(&RowsTapeType, call, &states->rows, tokens);
    if (!rows_tape) return NULL;
    rows_tape->offsets = (PyObject*)arena_buffer_new(&ArenaBufferType, call, states->offsets, "N", sizeof(size_t),
                                                     offsets_shape, 1);
    if (!rows_tape->offsets) Py_CLEAR(rows_tape);
    return (PyObject*)rows_tape;
}

PyObject* uform_py_tape_answer(uform_py_call_t const* call, uform_tape_view_t const* tape, size_t count,
                               size_t elements) {
    Py_ssize_t const shape[1] = {(Py_ssize_t)elements}, offsets_shape[1] = {(Py_ssize_t)count + 1};
    RowsTape* rows_tape = (RowsTape*)arena_buffer_new(&RowsTapeType, call, tape->data, "f", sizeof(float), shape, 1);
    if (!rows_tape) return NULL;
    rows_tape->offsets = (PyObject*)arena_buffer_new(&ArenaBufferType, call, tape->offsets, "N", sizeof(size_t),
                                                     offsets_shape, 1);
    if (!rows_tape->offsets) Py_CLEAR(rows_tape);
    return (PyObject*)rows_tape;
}

#pragma endregion Arenas

#pragma region Tapes

void uform_py_tape_release(uform_py_tape_t* tape) {
    PyMem_Free(tape->gathered);
    if (tape->data.obj) PyBuffer_Release(&tape->data);
    if (tape->offsets.obj) PyBuffer_Release(&tape->offsets);
    memset(tape, 0, sizeof(*tape));
}

/** Bytes one element of @p element takes. */
static size_t element_bytes(uform_py_element_t element) {
    return element == uform_py_element_samples_k ? sizeof(float) : 1;
}

/** The format character a buffer of @p element must report, its byte-order prefix stripped;
 *  NULL takes any. */
static char const* element_format(uform_py_element_t element) {
    return element == uform_py_element_samples_k ? "f" : element == uform_py_element_pixels_k ? "B" : NULL;
}

/** The format character of @p buffer, its byte-order prefix skipped; "B" when the
 *  exporter named none. */
static char buffer_format(Py_buffer const* buffer) {
    char const* format = buffer->format ? buffer->format : "B";
    while (*format == '@' || *format == '=' || *format == '<' || *format == '>' || *format == '!') ++format;
    return *format;
}

/** Releases the Arrow array a capsule owns, the owner of a tape read in place from Arrow memory. */
static void arrow_import_capsule_release(PyObject* capsule) {
    struct ArrowDeviceArray* const imported = (struct ArrowDeviceArray*)PyCapsule_GetPointer(capsule, NULL);
    if (imported->array.release) imported->array.release(&imported->array);
    PyMem_Free(imported);
}

/** Whether the host may read memory Arrow places on @p device_type, as the device-type table in
 *  @c c/uform/types.h has it. */
static int arrow_device_host_readable(ArrowDeviceType device_type) {
    return device_type == ARROW_DEVICE_CPU || device_type == ARROW_DEVICE_CUDA_HOST ||
           device_type == ARROW_DEVICE_ROCM_HOST || device_type == ARROW_DEVICE_CUDA_MANAGED;
}

/** Calls the Arrow export @p name of @p object with no requested schema; a new reference, or NULL
 *  with an exception set. */
static PyObject* arrow_export_call(PyObject* object, char const* name) {
    PyObject* const method = PyObject_GetAttrString(object, name);
    if (!method) return NULL;
    PyObject* const exported = PyObject_CallNoArgs(method);
    Py_DECREF(method);
    return exported;
}

/** The buffer formats a primitive Arrow array crosses as, and the bytes of each item. */
static struct {
    char arrow;
    char const* format;
    Py_ssize_t bytes;
} const arrow_formats[] = {
    {'C', "B", 1}, {'c', "b", 1}, {'I', "I", 4}, {'i', "i", 4},
    {'L', "Q", 8}, {'e', "e", 2}, {'f', "f", 4}, {'g', "d", 8},
};

/**
 *  @brief Reads the one primitive Arrow array @p object exports, through
 *      @c __arrow_c_device_array__ first and @c __arrow_c_array__ otherwise, into @p view: host
 *      memory, or memory of the GPU @p arena lives on, as the DLPack route takes.
 *  @return 1 when it does, 0 when @p object exports no Arrow array, or -1 with an exception set.
 */
static int arrow_buffer_acquire(PyObject* object, PyObject* arena, Py_buffer* view) {
    int const device = PyObject_HasAttrString(object, "__arrow_c_device_array__");
    if (!device && !PyObject_HasAttrString(object, "__arrow_c_array__")) return 0;
    PyObject* const pair = arrow_export_call(object, device ? "__arrow_c_device_array__" : "__arrow_c_array__");
    if (!pair) return -1;
    struct ArrowSchema* const schema = PyTuple_Check(pair) && PyTuple_GET_SIZE(pair) == 2
                                           ? (struct ArrowSchema*)PyCapsule_GetPointer(PyTuple_GET_ITEM(pair, 0),
                                                                                       "arrow_schema")
                                           : NULL;
    void* const array =
        schema ? PyCapsule_GetPointer(PyTuple_GET_ITEM(pair, 1), device ? "arrow_device_array" : "arrow_array") : NULL;
    struct ArrowDeviceArray* const imported = array ? (struct ArrowDeviceArray*)PyMem_Calloc(1, sizeof(*imported))
                                                    : NULL;
    PyObject* const owner = imported ? PyCapsule_New(imported, NULL, arrow_import_capsule_release) : NULL;
    if (!owner) {
        if (array && !imported) PyErr_NoMemory();
        else if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, "the Arrow export is no pair of a schema and an array");
        PyMem_Free(imported);
        Py_DECREF(pair);
        return -1;
    }
    // The array moves into the owner, and the capsule, seeing its release cleared, leaves it alone
    if (device) *imported = *(struct ArrowDeviceArray*)array;
    else
        *imported = (struct ArrowDeviceArray) {.array = *(struct ArrowArray*)array,
                                               .device_id = -1,
                                               .device_type = ARROW_DEVICE_CPU};
    ((struct ArrowArray*)array)->release = NULL;
    char const* format = NULL;
    Py_ssize_t item_bytes = 0;
    for (size_t row = 0; row != sizeof(arrow_formats) / sizeof(*arrow_formats); ++row)
        if (schema->format[0] == arrow_formats[row].arrow && !schema->format[1])
            format = arrow_formats[row].format, item_bytes = arrow_formats[row].bytes;
    struct ArrowArray const* const values = &imported->array;
    int const host_readable = arrow_device_host_readable(imported->device_type);
    DLDevice accepted = {kDLCPU, 0};
    int const on_device = host_readable ? 0 : arena_device(arena, &accepted);
    int const same_vendor = (int)imported->device_type == (int)accepted.device_type ||
                            (imported->device_type == ARROW_DEVICE_CUDA && accepted.device_type == kDLCUDAManaged);
    int failed = on_device < 0 ? -1 : 0;
    if (!failed && imported->device_type == ARROW_DEVICE_METAL)
        failed = uform_py_raise(uform_status_unsupported_k, "Metal arrays don't cross Arrow yet; pass host memory");
    else if (!failed && !host_readable && !(on_device && same_vendor && imported->device_id == accepted.device_id))
        failed = uform_py_raise(uform_status_device_k, "this input cannot take memory of Arrow device (%d, %d)",
                                (int)imported->device_type, (int)imported->device_id);
    else if (!failed && !format)
        failed = uform_py_raise(uform_status_unsupported_layout_k, "the Arrow array of format '%s' holds no scalars",
                                schema->format);
    else if (!failed && values->null_count != 0 && values->buffers[0])
        failed = uform_py_raise(uform_status_invalid_argument_k, "the Arrow array holds nulls");
    // The call or the host reads the values, so the host waits for a producer still writing them
    if (!failed && imported->sync_event && arena_wait(arena, imported->sync_event) != 0) failed = -1;
    if (!failed)
        *view = (Py_buffer) {.buf = (char*)values->buffers[1] + values->offset * item_bytes,
                             .obj = Py_NewRef(owner),
                             .len = (Py_ssize_t)values->length * item_bytes,
                             .itemsize = item_bytes,
                             .readonly = 1,
                             .ndim = 1,
                             .format = (char*)format};
    Py_DECREF(owner);
    Py_DECREF(pair);
    return failed ? -1 : 1;
}

/** Acquires @p object into @p view through DLPack, versioned then legacy, then Arrow, device then
 *  host, then the CUDA Array Interface on a CUDA @p arena, then the buffer protocol under
 *  @p flags; 0, or -1 with an exception set. */
static int uform_py_buffer_from_object(PyObject* object, PyObject* arena, int flags, Py_buffer* view) {
    int consumed = dlpack_acquire(object, arena, view);
    if (!consumed) consumed = arrow_buffer_acquire(object, arena, view);
    if (!consumed) consumed = cuda_array_interface_acquire(object, arena, view);
    return consumed ? (consumed > 0 ? 0 : -1) : PyObject_GetBuffer(object, view, flags);
}

/**
 *  @brief Acquires the offsets of a @c (data, offsets) pair into @p tape when @p candidate exports
 *      a C-contiguous run of unsigned words as wide as a @c size_t.
 *  @return 1 when it does, 0 when it is no offsets buffer at all, with no exception set.
 */
static int tape_offsets_acquire(PyObject* candidate, uform_py_tape_t* tape) {
    if (uform_py_buffer_from_object(candidate, NULL, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT, &tape->offsets) != 0) {
        PyErr_Clear();
        return 0;
    }
    char const format = buffer_format(&tape->offsets);
    if (tape->offsets.itemsize == sizeof(size_t) && tape->offsets.len >= (Py_ssize_t)sizeof(size_t) &&
        (format == 'N' || format == 'Q' || (format == 'L' && sizeof(unsigned long) == sizeof(size_t))))
        return 1;
    PyBuffer_Release(&tape->offsets);
    memset(&tape->offsets, 0, sizeof(tape->offsets));
    return 0;
}

/** Acquires the data of a pair whose offsets @p tape already holds; returns 0 or -1 with
 *  an exception set. */
static int tape_data_acquire(PyObject* data, char const* field, uform_py_element_t element, PyObject* arena,
                             uform_py_tape_t* tape) {
    if (uform_py_buffer_from_object(data, arena, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT, &tape->data) != 0) return -1;
    char const* const expected = element_format(element);
    if ((size_t)tape->data.itemsize != element_bytes(element) || (expected && buffer_format(&tape->data) != *expected))
        return uform_py_raise(uform_status_unsupported_layout_k, "the '%s' data holds items the verb cannot read",
                              field);
    size_t const* const offsets = (size_t const*)tape->offsets.buf;
    tape->count = (size_t)tape->offsets.len / sizeof(size_t) - 1;
    size_t const elements = (size_t)tape->data.len / element_bytes(element);
    if (offsets[tape->count] > elements)
        return uform_py_raise(uform_status_shape_mismatch_k, "the '%s' offsets end at %zu, past its %zu elements",
                              field, offsets[tape->count], elements);
    tape->view = (uform_tape_view_t) {tape->data.buf, offsets,
                                      element == uform_py_element_samples_k ? uform_scalar_f32_k : uform_scalar_u8_k};
    return 0;
}

/** Acquires one entry of a gathered batch into @p view, checking it holds @p element; returns
 *  0 or -1. */
static int tape_entry_acquire(PyObject* entry, char const* field, Py_ssize_t position, uform_py_element_t element,
                              Py_buffer* view) {
    if (element == uform_py_element_text_k && PyUnicode_Check(entry)) {
        Py_ssize_t length = 0;
        char const* const text = PyUnicode_AsUTF8AndSize(entry, &length); // lives as long as the string the view holds
        return text ? PyBuffer_FillInfo(view, entry, (void*)text, length, 1, PyBUF_SIMPLE) : -1;
    }
    int const flags = element == uform_py_element_pixels_k || element == uform_py_element_samples_k
                          ? PyBUF_C_CONTIGUOUS | PyBUF_FORMAT
                          : PyBUF_C_CONTIGUOUS;
    if (uform_py_buffer_from_object(entry, NULL, flags, view) != 0) {
        if (!PyErr_ExceptionMatches(PyExc_TypeError)) return -1; // a producer's own refusal says more
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "entry %zd of '%s' must be a %s, got %S", position, field,
                     element == uform_py_element_text_k ? "str or a bytes-like object" : "contiguous buffer",
                     Py_TYPE(entry));
        return -1;
    }
    char const* const expected = element_format(element);
    int const shaped = element == uform_py_element_pixels_k
                           ? view->ndim == 3 && view->shape[2] == 3 && view->shape[0] <= UINT32_MAX &&
                                 view->shape[1] <= UINT32_MAX
                           : element != uform_py_element_samples_k || view->ndim == 1;
    if ((size_t)view->itemsize == element_bytes(element) && (!expected || buffer_format(view) == *expected) && shaped)
        return 0;
    PyBuffer_Release(view);
    return uform_py_raise(uform_status_shape_mismatch_k, "entry %zd of '%s' must be %s", position, field,
                          element == uform_py_element_pixels_k    ? "a [height, width, 3] buffer of uint8"
                          : element == uform_py_element_samples_k ? "a 1-D buffer of float32"
                                                                  : "a run of bytes");
}

/** Gathers the entries of @p sequence into one block @p tape owns; returns 0 or -1 with
 *  an exception set. */
static int tape_gather(PyObject* sequence, char const* field, uform_py_element_t element, uform_py_tape_t* tape) {
    Py_ssize_t const count = PySequence_Fast_GET_SIZE(sequence);
    Py_buffer* const views = (Py_buffer*)PyMem_Calloc(count ? (size_t)count : 1, sizeof(Py_buffer));
    if (!views) {
        PyErr_NoMemory();
        return -1;
    }
    Py_ssize_t acquired = 0;
    size_t total = 0;
    int failed = 0;
    for (; acquired != count; ++acquired) {
        if (tape_entry_acquire(PySequence_Fast_GET_ITEM(sequence, acquired), field, acquired, element,
                               &views[acquired]) != 0) {
            failed = 1;
            break;
        }
        total += (size_t)views[acquired].len;
    }

    size_t const extents_count = element == uform_py_element_pixels_k ? 2 * (size_t)count : 0;
    size_t const head = ((size_t)count + 1) * sizeof(size_t) + extents_count * sizeof(uint32_t);
    if (!failed && !(tape->gathered = PyMem_Malloc(head + total))) {
        PyErr_NoMemory();
        failed = 1;
    }
    if (!failed) {
        size_t* const offsets = (size_t*)tape->gathered;
        uint32_t* const extents = extents_count ? (uint32_t*)(offsets + count + 1) : NULL;
        char* const data = (char*)tape->gathered + head;
        size_t written = 0;
        offsets[0] = 0;
        for (Py_ssize_t position = 0; position != count; ++position) {
            memcpy(data + written, views[position].buf, (size_t)views[position].len);
            written += (size_t)views[position].len;
            offsets[position + 1] = written / element_bytes(element);
            if (extents) {
                extents[position] = (uint32_t)views[position].shape[0];
                extents[count + position] = (uint32_t)views[position].shape[1];
            }
        }
        tape->count = (size_t)count;
        tape->extents = extents;
        tape->view = (uform_tape_view_t) {
            data, offsets, element == uform_py_element_samples_k ? uform_scalar_f32_k : uform_scalar_u8_k};
    }
    while (acquired-- != 0) PyBuffer_Release(&views[acquired]);
    PyMem_Free(views);
    return failed ? -1 : 0;
}

/** One chunk of an Arrow import, as a tape reads it: the elements, and the offsets around them,
 *  either 32 or 64 bits wide. */
typedef struct arrow_chunk_t {
    void const* data;
    void const* offsets;
    size_t offset_bytes;
    size_t count;
} arrow_chunk_t;

/**
 *  @brief Checks @p schema names a batch of @p element entries, strings, binaries or lists of
 *      @c float32, and finds the width of its offsets.
 *  @return The bytes one offset takes, or 0 with an exception set.
 */
static size_t arrow_schema_offset_bytes(struct ArrowSchema const* schema, char const* field,
                                        uform_py_element_t element) {
    char const* const format = schema->format;
    int const bytes_like = element == uform_py_element_text_k || element == uform_py_element_bytes_k;
    if (bytes_like && (!strcmp(format, "u") || !strcmp(format, "z"))) return sizeof(int32_t);
    if (bytes_like && (!strcmp(format, "U") || !strcmp(format, "Z"))) return sizeof(int64_t);
    int const floats = element == uform_py_element_samples_k && schema->n_children == 1 &&
                       !strcmp(schema->children[0]->format, "f");
    if (floats && !strcmp(format, "+l")) return sizeof(int32_t);
    if (floats && !strcmp(format, "+L")) return sizeof(int64_t);
    uform_py_raise(uform_status_unsupported_layout_k, "the '%s' Arrow array of format '%s' holds no %s", field, format,
                   bytes_like ? "strings or binaries" : "lists of float32");
    return 0;
}

/** Reads one Arrow array into @p chunk after @p schema checked its layout; 0, or -1 with an
 *  exception set. */
static int arrow_chunk_read(struct ArrowDeviceArray const* imported, size_t offset_bytes, char const* field,
                            PyObject* arena, arrow_chunk_t* chunk) {
    struct ArrowArray const* const array = &imported->array;
    if (imported->device_type == ARROW_DEVICE_METAL)
        return uform_py_raise(uform_status_unsupported_k, "Metal arrays don't cross Arrow yet; pass host memory");
    if (!arrow_device_host_readable(imported->device_type))
        return uform_py_raise(uform_status_device_k, "the '%s' Arrow offsets live where the host may not read them",
                              field);
    // The host reads the offsets while dispatching the call, so it waits for the producer itself
    if (imported->sync_event && arena_wait(arena, imported->sync_event) != 0) return -1;
    int const nested = array->n_children == 1;
    struct ArrowArray const* const values = nested ? array->children[0] : array;
    if ((array->null_count != 0 && array->buffers[0]) || (values->null_count != 0 && values->buffers[0]))
        return uform_py_raise(uform_status_invalid_argument_k, "the '%s' Arrow array holds nulls", field);
    chunk->data = nested ? (char const*)values->buffers[1] + values->offset * (int64_t)sizeof(float)
                         : array->buffers[2];
    chunk->offsets = (char const*)array->buffers[1] + array->offset * (int64_t)offset_bytes;
    chunk->offset_bytes = offset_bytes;
    chunk->count = (size_t)array->length;
    return 0;
}

/** Offset @p position of @p chunk, whatever its width. */
static size_t arrow_chunk_offset(arrow_chunk_t const* chunk, size_t position) {
    int64_t const* const wide = (int64_t const*)chunk->offsets;
    int32_t const* const narrow = (int32_t const*)chunk->offsets;
    return chunk->offset_bytes == sizeof(int64_t) ? (size_t)wide[position] : (size_t)narrow[position];
}

/**
 *  @brief Points @p tape at one Arrow array in place, its 64-bit offsets included; 32-bit
 *      offsets alone are widened into a block the tape owns. @p owner keeps the array alive.
 *  @return 0, or -1 with an exception set.
 */
static int arrow_tape_in_place(PyObject* owner, arrow_chunk_t const* chunk, uform_py_element_t element,
                               uform_py_tape_t* tape) {
    size_t const bytes = element == uform_py_element_samples_k ? sizeof(float) : 1;
    size_t const elements = chunk->count ? arrow_chunk_offset(chunk, chunk->count) : 0;
    if (PyBuffer_FillInfo(&tape->data, owner, (void*)chunk->data, (Py_ssize_t)(elements * bytes), 1, PyBUF_SIMPLE) != 0)
        return -1;
    size_t const* offsets = (size_t const*)chunk->offsets;
    if (chunk->offset_bytes == sizeof(size_t) && chunk->count) { // an empty array may carry no offsets at all
        if (PyBuffer_FillInfo(&tape->offsets, owner, (void*)chunk->offsets,
                              (Py_ssize_t)((chunk->count + 1) * sizeof(size_t)), 1, PyBUF_SIMPLE) != 0)
            return -1;
    }
    else {
        size_t* const widened = (size_t*)PyMem_Malloc((chunk->count + 1) * sizeof(size_t));
        if (!(tape->gathered = widened)) return PyErr_NoMemory(), -1;
        for (size_t position = 0; position <= chunk->count; ++position)
            widened[position] = chunk->count ? arrow_chunk_offset(chunk, position) : 0;
        offsets = widened;
    }
    tape->count = chunk->count;
    tape->view = (uform_tape_view_t) {chunk->data, offsets,
                                      element == uform_py_element_samples_k ? uform_scalar_f32_k : uform_scalar_u8_k};
    return 0;
}

/** Gathers @p count Arrow chunks into one block @p tape owns, with one copy; 0, or -1 with an
 *  exception set. */
static int arrow_tape_gather(arrow_chunk_t const* chunks, size_t count, uform_py_element_t element,
                             uform_py_tape_t* tape) {
    size_t const bytes = element == uform_py_element_samples_k ? sizeof(float) : 1;
    size_t entries = 0, elements = 0;
    for (size_t chunk = 0; chunk != count; ++chunk) {
        entries += chunks[chunk].count;
        if (chunks[chunk].count)
            elements += arrow_chunk_offset(&chunks[chunk], chunks[chunk].count) - arrow_chunk_offset(&chunks[chunk], 0);
    }
    size_t const head = (entries + 1) * sizeof(size_t);
    if (!(tape->gathered = PyMem_Malloc(head + elements * bytes))) return PyErr_NoMemory(), -1;
    size_t* const offsets = (size_t*)tape->gathered;
    char* const data = (char*)tape->gathered + head;
    size_t entry = 0, written = 0;
    offsets[0] = 0;
    for (size_t chunk = 0; chunk != count; ++chunk) {
        if (!chunks[chunk].count) continue;
        size_t const first = arrow_chunk_offset(&chunks[chunk], 0);
        size_t const last = arrow_chunk_offset(&chunks[chunk], chunks[chunk].count);
        memcpy(data + written * bytes, (char const*)chunks[chunk].data + first * bytes, (last - first) * bytes);
        for (size_t position = 1; position <= chunks[chunk].count; ++position)
            offsets[++entry] = written + arrow_chunk_offset(&chunks[chunk], position) - first;
        written += last - first;
    }
    tape->count = entries;
    tape->view = (uform_tape_view_t) {data, offsets,
                                      element == uform_py_element_samples_k ? uform_scalar_f32_k : uform_scalar_u8_k};
    return 0;
}

/**
 *  @brief Reads an object exporting one Arrow array, on a device or in host memory, into @p tape.
 *  @return 0, or -1 with an exception set.
 */
static int arrow_tape_from_array(PyObject* object, int device, char const* field, uform_py_element_t element,
                                 PyObject* arena, uform_py_tape_t* tape) {
    PyObject* const pair = arrow_export_call(object, device ? "__arrow_c_device_array__" : "__arrow_c_array__");
    if (!pair) return -1;
    struct ArrowSchema* const schema = PyTuple_Check(pair) && PyTuple_GET_SIZE(pair) == 2
                                           ? (struct ArrowSchema*)PyCapsule_GetPointer(PyTuple_GET_ITEM(pair, 0),
                                                                                       "arrow_schema")
                                           : NULL;
    void* const array =
        schema ? PyCapsule_GetPointer(PyTuple_GET_ITEM(pair, 1), device ? "arrow_device_array" : "arrow_array") : NULL;
    if (!array) {
        if (!PyErr_Occurred())
            PyErr_Format(PyExc_TypeError, "the '%s' Arrow export is no pair of a schema and an array", field);
        Py_DECREF(pair);
        return -1;
    }
    struct ArrowDeviceArray* const imported = (struct ArrowDeviceArray*)PyMem_Calloc(1, sizeof(*imported));
    PyObject* const owner = imported ? PyCapsule_New(imported, NULL, arrow_import_capsule_release) : NULL;
    if (!owner) {
        PyMem_Free(imported);
        Py_DECREF(pair);
        return imported ? -1 : (PyErr_NoMemory(), -1);
    }
    // The array moves into the owner, and the capsule, seeing its release cleared, leaves it alone
    if (device) *imported = *(struct ArrowDeviceArray*)array;
    else
        *imported = (struct ArrowDeviceArray) {.array = *(struct ArrowArray*)array,
                                               .device_id = -1,
                                               .device_type = ARROW_DEVICE_CPU};
    ((struct ArrowArray*)array)->release = NULL;
    arrow_chunk_t chunk;
    size_t const offset_bytes = arrow_schema_offset_bytes(schema, field, element);
    int const failed = !offset_bytes || arrow_chunk_read(imported, offset_bytes, field, arena, &chunk) != 0 ||
                       arrow_tape_in_place(owner, &chunk, element, tape) != 0;
    Py_DECREF(owner);
    Py_DECREF(pair);
    return failed ? -1 : 0;
}

/**
 *  @brief Reads an object exporting a stream of Arrow arrays, on a device or in host memory, into
 *      @p tape: one chunk in place, several gathered with one copy.
 *  @return 0, or -1 with an exception set.
 */
static int arrow_tape_from_stream(PyObject* object, int device, char const* field, uform_py_element_t element,
                                  PyObject* arena, uform_py_tape_t* tape) {
    PyObject* const capsule = arrow_export_call(object, device ? "__arrow_c_device_stream__" : "__arrow_c_stream__");
    if (!capsule) return -1;
    void* const stream = PyCapsule_GetPointer(capsule, device ? "arrow_device_array_stream" : "arrow_array_stream");
    if (!stream) {
        Py_DECREF(capsule);
        return -1;
    }
    struct ArrowDeviceArrayStream* const device_stream = (struct ArrowDeviceArrayStream*)stream;
    struct ArrowArrayStream* const host_stream = (struct ArrowArrayStream*)stream;
    struct ArrowSchema schema = {0};
    int failed = (device ? device_stream->get_schema(device_stream, &schema)
                         : host_stream->get_schema(host_stream, &schema))
                     ? uform_py_raise(uform_status_invalid_argument_k, "the '%s' Arrow stream has no schema", field)
                     : 0;
    size_t const offset_bytes = failed ? 0 : arrow_schema_offset_bytes(&schema, field, element);
    if (schema.release) schema.release(&schema);
    failed = failed || !offset_bytes;

    // Every chunk stays owned until the tape is built, so a single one is read in place.
    PyObject** owners = NULL;
    arrow_chunk_t* chunks = NULL;
    size_t count = 0, capacity = 0;
    while (!failed) {
        struct ArrowDeviceArray next = {0};
        int const code = device ? device_stream->get_next(device_stream, &next)
                                : host_stream->get_next(host_stream, &next.array);
        if (code) {
            failed = uform_py_raise(uform_status_invalid_argument_k, "the '%s' Arrow stream failed with code %d", field,
                                    code);
            break;
        }
        if (!next.array.release) break;
        if (!device) next.device_type = ARROW_DEVICE_CPU, next.device_id = -1;
        if (count == capacity) {
            capacity = capacity ? capacity * 2 : 4;
            PyObject** const grown_owners = (PyObject**)PyMem_Realloc(owners, capacity * sizeof(*owners));
            if (grown_owners) owners = grown_owners;
            arrow_chunk_t* const grown_chunks = (arrow_chunk_t*)PyMem_Realloc(chunks, capacity * sizeof(*chunks));
            if (grown_chunks) chunks = grown_chunks;
            if (!grown_owners || !grown_chunks) {
                next.array.release(&next.array);
                failed = (PyErr_NoMemory(), -1);
                break;
            }
        }
        struct ArrowDeviceArray* const imported = (struct ArrowDeviceArray*)PyMem_Malloc(sizeof(*imported));
        owners[count] = imported ? PyCapsule_New(imported, NULL, arrow_import_capsule_release) : NULL;
        if (!owners[count]) {
            PyMem_Free(imported);
            next.array.release(&next.array);
            failed = imported ? -1 : (PyErr_NoMemory(), -1);
            break;
        }
        *imported = next;
        failed = arrow_chunk_read(imported, offset_bytes, field, arena, &chunks[count++]) != 0;
    }
    if (!failed)
        failed = count == 1 ? arrow_tape_in_place(owners[0], &chunks[0], element, tape) != 0
                            : arrow_tape_gather(chunks, count, element, tape) != 0;
    while (count) Py_DECREF(owners[--count]);
    PyMem_Free(owners);
    PyMem_Free(chunks);
    Py_DECREF(capsule); // releases the stream
    return failed ? -1 : 0;
}

/**
 *  @brief Reads @p object into @p tape through the Arrow PyCapsule interface, device exports first.
 *  @return 1 when it does, 0 when @p object exports no Arrow, or -1 with an exception set.
 */
static int arrow_tape_acquire(PyObject* object, char const* field, uform_py_element_t element, PyObject* arena,
                              uform_py_tape_t* tape) {
    static char const* const protocols[] = {"__arrow_c_device_stream__", "__arrow_c_device_array__",
                                            "__arrow_c_stream__", "__arrow_c_array__"};
    for (int protocol = 0; protocol != 4; ++protocol) {
        if (!PyObject_HasAttrString(object, protocols[protocol])) continue;
        int const device = protocol < 2, streamed = protocol % 2 == 0;
        int const read = streamed ? arrow_tape_from_stream(object, device, field, element, arena, tape)
                                  : arrow_tape_from_array(object, device, field, element, arena, tape);
        return read == 0 ? 1 : -1;
    }
    return 0;
}

int uform_py_tape_from_object(PyObject* object, char const* field, uform_py_element_t element, PyObject* arena,
                              uform_py_tape_t* tape) {
    memset(tape, 0, sizeof(*tape));
    if (PyUnicode_Check(object) || PyBytes_Check(object) || PyByteArray_Check(object) || PyMemoryView_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' argument takes a sequence or a (data, offsets) pair, got one %S", field,
                     Py_TYPE(object));
        return -1;
    }
    if (element != uform_py_element_pixels_k && PyTuple_Check(object) && PyTuple_GET_SIZE(object) == 2 &&
        tape_offsets_acquire(PyTuple_GET_ITEM(object, 1), tape))
        return tape_data_acquire(PyTuple_GET_ITEM(object, 0), field, element, arena, tape);
    if (element != uform_py_element_pixels_k) {
        int const imported = arrow_tape_acquire(object, field, element, arena, tape);
        if (imported) return imported > 0 ? 0 : -1;
    }
    PyObject* const sequence = PySequence_Fast(object, "expected a sequence");
    if (!sequence) return -1;
    int const gathered = tape_gather(sequence, field, element, tape);
    Py_DECREF(sequence);
    return gathered;
}

void uform_py_words_release(uform_py_words_t* words) {
    PyMem_Free(words->gathered);
    if (words->buffer.obj) PyBuffer_Release(&words->buffer);
    memset(words, 0, sizeof(*words));
}

int uform_py_words_from_object(PyObject* object, char const* field, size_t count, uform_py_words_t* words) {
    if (!object || object == Py_None) return 0;
    if (uform_py_buffer_from_object(object, NULL, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT, &words->buffer) == 0) {
        if (words->buffer.itemsize != sizeof(uint32_t) || buffer_format(&words->buffer) != 'I')
            return uform_py_raise(uform_status_unsupported_layout_k, "the '%s' buffer must hold uint32 words", field);
        if ((size_t)words->buffer.len != count * sizeof(uint32_t))
            return uform_py_raise(uform_status_shape_mismatch_k, "the '%s' buffer must hold %zu words", field, count);
        words->data = (uint32_t const*)words->buffer.buf;
        return 0;
    }
    PyErr_Clear();
    PyObject* const sequence = PySequence_Fast(object, "expected a buffer of uint32 or a sequence of integers");
    if (!sequence) return -1;
    if ((size_t)PySequence_Fast_GET_SIZE(sequence) != count) {
        Py_DECREF(sequence);
        return uform_py_raise(uform_status_shape_mismatch_k, "the '%s' option must hold %zu entries", field, count);
    }
    words->gathered = (uint32_t*)PyMem_Malloc((count ? count : 1) * sizeof(uint32_t));
    int failed = !words->gathered;
    if (failed) PyErr_NoMemory();
    for (size_t position = 0; !failed && position != count; ++position) {
        size_t value = 0;
        failed = uform_py_size_from_object(PySequence_Fast_GET_ITEM(sequence, position), field, &value) != 0;
        if (!failed && value > UINT32_MAX)
            failed = uform_py_raise(uform_status_invalid_argument_k, "the '%s' option exceeds 32 bits", field);
        if (!failed) words->gathered[position] = (uint32_t)value;
    }
    Py_DECREF(sequence);
    words->data = words->gathered;
    return failed ? -1 : 0;
}

#pragma endregion Tapes

#pragma region Library Probes

char const doc_version[] =              //
    "version() -> str\n"                //
    "\n"                                //
    "The library's semantic version.\n" //
    "Calls `uform_version_name`.";

char const doc_capabilities_compiled[] =                                          //
    "capabilities_compiled() -> str\n"                                            //
    "\n"                                                                          //
    "The hardware capabilities this binary was compiled with, comma-separated.\n" //
    "Calls `uform_capabilities_compiled`.";

char const doc_capabilities_available[] =                                          //
    "capabilities_available() -> str\n"                                            //
    "\n"                                                                           //
    "The hardware capabilities this machine offers at runtime, comma-separated.\n" //
    "Calls `uform_capabilities_available`.";

PyObject* api_version(PyObject* self, PyObject* unused) { return PyUnicode_FromString(uform_version_name()); }

PyObject* api_capabilities_compiled(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(uform_capabilities_compiled());
}

PyObject* api_capabilities_available(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(uform_capabilities_available());
}

/** One GPU this process can drive, as @c uform_gpu_devices listed it. */
typedef struct GPUDevice {
    PyObject ob_base;
    uform_gpu_device_t device;
} GPUDevice;

static PyObject* GPUDevice_get_vendor(GPUDevice* self, void* closure) {
    return uform_py_enum_to_object(&uform_py_gpu_vendor, (int)self->device.vendor);
}

static PyObject* GPUDevice_get_gpu_class(GPUDevice* self, void* closure) {
    return uform_py_class_to_object(&uform_py_gpu_class, (int)self->device.gpu_class);
}

static PyObject* GPUDevice_get_name(GPUDevice* self, void* closure) { return PyUnicode_FromString(self->device.name); }

static PyMemberDef GPUDevice_members[] = {
    {"vendor_ordinal", Py_T_UINT, offsetof(GPUDevice, device.vendor_ordinal), Py_READONLY,
     "The device's ordinal in its vendor's runtime, as nvidia-smi or rocm-smi list it."},
    {"compute_units", Py_T_UINT, offsetof(GPUDevice, device.compute_units), Py_READONLY,
     "Streaming multiprocessors or compute units."},
    {"memory_bytes", Py_T_ULONGLONG, offsetof(GPUDevice, device.memory_bytes), Py_READONLY, "Device memory in bytes."},
    {NULL, 0, 0, 0, NULL},
};

static PyGetSetDef GPUDevice_getset[] = {
    {"vendor", (getter)GPUDevice_get_vendor, NULL, "The `GPUVendor` whose runtime drives the device.", NULL},
    {"gpu_class", (getter)GPUDevice_get_gpu_class, NULL, "The `GPUClass` whose kernels run on it, or None.", NULL},
    {"name", (getter)GPUDevice_get_name, NULL, "The vendor's own name for the device.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject* GPUDevice_repr(GPUDevice* self) {
    return PyUnicode_FromFormat("GPUDevice(name='%s', vendor_ordinal=%u, memory_bytes=%llu)", self->device.name,
                                (unsigned)self->device.vendor_ordinal, (unsigned long long)self->device.memory_bytes);
}

PyTypeObject GPUDeviceType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.GPUDevice",
    .tp_doc = "One GPU this process can drive, its position in `gpu_devices()` being its ordinal.",
    .tp_basicsize = sizeof(GPUDevice),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_repr = (reprfunc)GPUDevice_repr,
    .tp_members = GPUDevice_members,
    .tp_getset = GPUDevice_getset,
};

char const doc_gpu_devices[] =                                                                      //
    "gpu_devices() -> tuple[GPUDevice, ...]\n"                                                      //
    "\n"                                                                                            //
    "Every GPU this process can drive, NVIDIA devices first, then AMD, then Apple, so a device's\n" //
    "position is the ordinal `Executor.for_vram` and `Model.load_vram` name it by. A vendor this\n" //
    "build does not compile, or whose runtime is absent, lists none.\n"                             //
    "Calls `uform_gpu_devices`.";

/** Lists up to @p capacity devices into @p devices; returns how many there are, or -1. */
static Py_ssize_t gpu_devices_list(uform_gpu_device_t* devices, size_t capacity) {
    size_t devices_count = 0;
    uform_outcome_t outcome = {0};
    uform_gpu_devices_t arguments = {.outcome = &outcome,
                                     .devices = devices,
                                     .devices_capacity = capacity,
                                     .devices_count = &devices_count};
    Py_BEGIN_ALLOW_THREADS;
    uform_gpu_devices(&arguments);
    Py_END_ALLOW_THREADS;
    return uform_py_raise_on_outcome(&outcome) == 0 ? (Py_ssize_t)devices_count : -1;
}

PyObject* api_gpu_devices(PyObject* self, PyObject* unused) {
    Py_ssize_t const counted = gpu_devices_list(NULL, 0);
    if (counted < 0) return NULL;
    uform_gpu_device_t* const devices = (uform_gpu_device_t*)PyMem_Calloc(counted ? (size_t)counted : 1,
                                                                          sizeof(uform_gpu_device_t));
    if (!devices) return PyErr_NoMemory();
    Py_ssize_t const listed = gpu_devices_list(devices, (size_t)counted);
    PyObject* tuple = listed < 0 ? NULL : PyTuple_New(listed < counted ? listed : counted);
    for (Py_ssize_t index = 0; tuple && index != PyTuple_GET_SIZE(tuple); ++index) {
        GPUDevice* const entry = (GPUDevice*)GPUDeviceType.tp_alloc(&GPUDeviceType, 0);
        if (entry) entry->device = devices[index];
        if (!entry) Py_CLEAR(tuple);
        else PyTuple_SET_ITEM(tuple, index, (PyObject*)entry);
    }
    PyMem_Free(devices);
    return tuple;
}

#pragma endregion Library Probes
