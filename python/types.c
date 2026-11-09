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

PyObject *ErrorType = NULL;

/** The class each failing status raises, indexed by status; success and deferred never raise. */
static PyObject *status_error_types[uform_status_invalid_argument_k + 1];

/** One failing status, the dotted name of the class it raises, the built-in it also subclasses,
 *  and its doc. */
static struct {
    uform_status_t status;
    char const *name;

    /** NULL when no built-in names the fault. */
    PyObject **builtin;
    char const *doc;
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
    {uform_status_config_mismatch_k, "uform.ConfigMismatchError", &PyExc_ValueError,
     "The request contradicts a frozen configuration, like resuming a checkpoint of another geometry."},
    {uform_status_contended_k, "uform.ContendedError", NULL, "Another call already holds the executor or the arena."},
    {uform_status_cancelled_k, "uform.CancelledError", NULL,
     "A progress handler stopped the call, which published nothing."},
    {uform_status_unsupported_k, "uform.UnsupportedError", &PyExc_NotImplementedError,
     "The verb is declared, but this residency or model does not implement it."},
    {uform_status_invalid_argument_k, "uform.InvalidArgumentError", &PyExc_ValueError,
     "An argument breaks the verb's contract."},
};

int uform_py_raise(uform_status_t status, char const *format, ...) {
    PyErr_Clear(); // a status this specific supersedes whatever the caller was carrying
    size_t const types_count = sizeof(status_error_types) / sizeof(*status_error_types);
    PyObject *type = (size_t)status < types_count ? status_error_types[status] : NULL;
    if (!type) {
        status = uform_status_internal_k;
        type = status_error_types[status];
    }
    va_list arguments;
    va_start(arguments, format);
    PyObject *const message = PyUnicode_FromFormatV(format ? format : "", arguments);
    va_end(arguments);
    if (!message) return -1;

    PyObject *const instance = PyObject_CallOneArg(type, message);
    Py_DECREF(message);
    if (!instance) return -1;
    PyObject *const member = uform_py_enum_to_object(&uform_py_status, (int)status);
    int const attached = member ? PyObject_SetAttrString(instance, "status", member) : -1;
    Py_XDECREF(member);
    if (attached != 0) {
        Py_DECREF(instance);
        return -1;
    }
    PyErr_SetRaisedException(instance);
    return -1;
}

int uform_py_raise_on_outcome(uform_outcome_t const *outcome) {
    if (outcome->status == uform_status_success_k) return 0;
    return uform_py_raise(outcome->status, "%s", outcome->message ? outcome->message : "");
}

int uform_py_dict_steal_item(PyObject *dict, char const *key, PyObject *value) {
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
    "  args (tuple): The detail the status cannot carry, as on any exception.";

int uform_py_register_error_types(PyObject *module) {
    ErrorType = PyErr_NewExceptionWithDoc("uform.Error", (char *)doc_Error, PyExc_RuntimeError, NULL);
    if (!ErrorType || PyModule_AddObjectRef(module, "Error", ErrorType) < 0) return -1;
    for (size_t row = 0; row != sizeof(status_errors) / sizeof(*status_errors); ++row) {
        PyObject *const bases = status_errors[row].builtin ? PyTuple_Pack(2, ErrorType, *status_errors[row].builtin)
                                                           : Py_NewRef(ErrorType);
        if (!bases) return -1;
        PyObject *const type = PyErr_NewExceptionWithDoc(status_errors[row].name, (char *)status_errors[row].doc, bases,
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

static char const *spell_status(int value) { return uform_status_name((uform_status_t)value); }
static char const *spell_scalar(int value) { return uform_scalar_name((uform_scalar_t)value); }
static char const *spell_mood(int value) { return uform_mood_name((uform_mood_t)value); }
static char const *spell_cpu_class(int value) { return uform_cpu_class_name((uform_cpu_class_t)value); }
static char const *spell_gpu_class(int value) { return uform_gpu_class_name((uform_gpu_class_t)value); }

/** The C interface spells no vendor, so the binding names the three it lists. */
static char const *spell_gpu_vendor(int value) {
    switch (value) {
    case uform_gpu_vendor_nvidia_k: return "nvidia";
    case uform_gpu_vendor_amd_k: return "amd";
    case uform_gpu_vendor_apple_k: return "apple";
    default: return NULL;
    }
}

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
uform_py_enum_t uform_py_cpu_class = {.name = "CPUClass",
                                      .base = "IntEnum",
                                      .first = uform_cpu_class_serial_k,
                                      .last = uform_cpu_class_monoversioned_k,
                                      .spell = spell_cpu_class};
uform_py_enum_t uform_py_gpu_class = {.name = "GPUClass",
                                      .base = "IntEnum",
                                      .first = uform_gpu_class_nvidia_ampere_k,
                                      .last = uform_gpu_class_apple_metal10_k,
                                      .spell = spell_gpu_class};
uform_py_enum_t uform_py_gpu_vendor = {.name = "GPUVendor",
                                       .base = "IntEnum",
                                       .first = uform_gpu_vendor_nvidia_k,
                                       .last = uform_gpu_vendor_apple_k,
                                       .spell = spell_gpu_vendor};

/** The spelling of @p value, or NULL when no member carries it. */
static char const *enum_spelling(uform_py_enum_t const *mirrored, long value) {
    if (value < mirrored->first || value > mirrored->last) return NULL;
    char const *const spelled = mirrored->spell((int)value);
    if (!spelled || strcmp(spelled, "unknown") == 0 || strcmp(spelled, "unrecognized") == 0) return NULL;
    return spelled;
}

/** Builds the class of @p mirrored by calling its base from @p enum_module with @p keywords,
 *  in value order. */
static int enum_register(PyObject *module, PyObject *enum_module, PyObject *keywords, uform_py_enum_t *mirrored) {
    PyObject *members = PyDict_New();
    for (long value = mirrored->first; members && value <= mirrored->last; ++value) {
        char const *const spelled = enum_spelling(mirrored, value);
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
    PyObject *const base = members ? PyObject_GetAttrString(enum_module, mirrored->base) : NULL;
    PyObject *const name = base ? PyUnicode_FromString(mirrored->name) : NULL;
    PyObject *const arguments = name ? PyTuple_Pack(2, name, members) : NULL;
    mirrored->type = arguments ? PyObject_Call(base, arguments, keywords) : NULL;
    Py_XDECREF(arguments);
    Py_XDECREF(name);
    Py_XDECREF(base);
    Py_XDECREF(members);
    return mirrored->type ? PyModule_AddObjectRef(module, mirrored->name, mirrored->type) : -1;
}

int uform_py_register_enums(PyObject *module) {
    uform_py_enum_t *const mirrored[] = {
        &uform_py_status,       &uform_py_scalar,     &uform_py_mood,           &uform_py_cpu_class,
        &uform_py_gpu_class,    &uform_py_gpu_vendor, &uform_py_outputs,        &uform_py_modality,
        &uform_py_objective,    &uform_py_negatives,  &uform_py_pair_loss_form, &uform_py_late_interaction_side,
        &uform_py_multimem_tier};
    PyObject *const enum_module = PyImport_ImportModule("enum");
    PyObject *keywords = enum_module ? PyDict_New() : NULL;
    if (keywords && uform_py_dict_steal_item(keywords, "module", PyUnicode_FromString("uform")) != 0)
        Py_CLEAR(keywords);
    int registered = keywords ? 0 : -1;
    for (size_t row = 0; registered == 0 && row != sizeof(mirrored) / sizeof(*mirrored); ++row)
        registered = enum_register(module, enum_module, keywords, mirrored[row]);
    Py_XDECREF(keywords);
    Py_XDECREF(enum_module);
    return registered;
}

int uform_py_enum_from_object(PyObject *object, char const *field, uform_py_enum_t const *mirrored, int *value) {
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

PyObject *uform_py_enum_to_object(uform_py_enum_t const *mirrored, int value) {
    PyObject *const integer = PyLong_FromLong(value);
    PyObject *const member = integer ? PyObject_CallOneArg(mirrored->type, integer) : NULL;
    Py_XDECREF(integer);
    return member;
}

int uform_py_class_from_object(PyObject *object, char const *field, uform_py_enum_t const *classes, int *value) {
    *value = 0;
    if (!object || object == Py_None) return 0;
    return uform_py_enum_from_object(object, field, classes, value);
}

PyObject *uform_py_class_to_object(uform_py_enum_t const *classes, int value) {
    return value ? uform_py_enum_to_object(classes, value) : Py_NewRef(Py_None);
}

#pragma endregion Enums

#pragma region Arguments

int uform_py_arguments_parse(char const *verb, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple, Py_ssize_t positional_only, Py_ssize_t positional_max,
                             uform_py_argument_t const *arguments, size_t count) {
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
        PyObject *const name = PyTuple_GET_ITEM(args_names_tuple, name_index);
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

int uform_py_size_from_object(PyObject *object, char const *field, size_t *value) {
    if (!object) return 0;
    Py_ssize_t const signed_value = PyNumber_AsSsize_t(object, PyExc_OverflowError);
    if (signed_value == -1 && PyErr_Occurred()) return -1;
    if (signed_value < 0)
        return uform_py_raise(uform_status_invalid_argument_k, "the '%s' option must not be negative", field);
    *value = (size_t)signed_value;
    return 0;
}

int uform_py_scalar_from_object(PyObject *object, char const *field, uform_scalar_t *kind) {
    if (!object) return 0;
    int value = uform_scalar_unknown_k;
    if (uform_py_enum_from_object(object, field, &uform_py_scalar, &value) != 0) return -1;
    *kind = (uform_scalar_t)value;
    return 0;
}

int uform_py_ordinals_from_object(PyObject *object, char const *field, int32_t **ordinals, size_t *count) {
    *ordinals = NULL;
    *count = 0;
    if (!object || object == Py_None) return 0;
    PyObject *const sequence = PySequence_Fast(object, "expected a sequence of integers");
    if (!sequence) return -1;
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    int32_t *const claimed = length ? (int32_t *)PyMem_Malloc((size_t)length * sizeof(int32_t)) : NULL;
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

int uform_py_path_from_object(PyObject *object, char const *field, uform_bytes_view_t *path, PyObject **decoded) {
    PyObject *bytes_object = NULL;
    if (!PyUnicode_FSConverter(object, &bytes_object)) return -1;
    path->data = (uint8_t const *)PyBytes_AS_STRING(bytes_object);
    path->length = (size_t)PyBytes_GET_SIZE(bytes_object);
    *decoded = bytes_object;
    return 0;
}

PyObject *uform_py_struct_sequence_steal(PyTypeObject *type, PyObject **fields, Py_ssize_t count) {
    PyObject *result = NULL;
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

#pragma endregion Arguments

#pragma region Keyword Structures

/**
 *  @brief Checks and stores @p value as field @p field of a new structure, a tuple building a
 *      nested structure from its items in order.
 *  @return A new reference to what the structure keeps, or NULL with an exception set.
 */
static PyObject *field_accept(uform_py_field_t const *field, PyObject *value) {
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
        if (uform_py_enum_from_object(value, field->name, (uform_py_enum_t const *)field->detail, &member) != 0)
            return NULL;
        return uform_py_enum_to_object((uform_py_enum_t const *)field->detail, member);
    }
    case uform_py_field_struct_k: {
        PyTypeObject *const nested = (PyTypeObject *)field->detail;
        if (value == Py_None || Py_IS_TYPE(value, nested)) return Py_NewRef(value);
        if (!PyTuple_Check(value)) {
            PyErr_Format(PyExc_TypeError, "the '%s' field takes a uform.%s or a tuple of its fields, got %S",
                         field->name, nested->tp_name + sizeof("uform.") - 1, Py_TYPE(value));
            return NULL;
        }
        uform_py_fields_t const *const schema = (uform_py_fields_t const *)nested;
        Py_ssize_t const given = PyTuple_GET_SIZE(value);
        if ((size_t)given > schema->count) {
            PyErr_Format(PyExc_TypeError, "the '%s' field takes at most %zu values, got %zd", field->name,
                         schema->count, given);
            return NULL;
        }
        PyObject *keywords = PyDict_New();
        for (Py_ssize_t position = 0; keywords && position != given; ++position)
            if (PyDict_SetItemString(keywords, schema->fields[position].name, PyTuple_GET_ITEM(value, position)) != 0)
                Py_CLEAR(keywords);
        PyObject *const empty = keywords ? PyTuple_New(0) : NULL;
        PyObject *const built = empty ? PyObject_Call((PyObject *)nested, empty, keywords) : NULL;
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
static PyObject *field_default(uform_py_field_t const *field) {
    switch (field->kind) {
    case uform_py_field_size_k: return PyLong_FromLong(0);
    case uform_py_field_float_k:
    case uform_py_field_double_k: return PyFloat_FromDouble(0.0);
    case uform_py_field_enum_k: return uform_py_enum_to_object((uform_py_enum_t const *)field->detail, 0);
    default: return Py_NewRef(Py_None);
    }
}

static PyObject *Fields_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    uform_py_fields_t const *const schema = (uform_py_fields_t const *)type;
    if (PyTuple_GET_SIZE(args) != 0) {
        PyErr_Format(PyExc_TypeError, "%s() takes keyword arguments only", type->tp_name + sizeof("uform.") - 1);
        return NULL;
    }
    Py_ssize_t const given = keywords ? PyDict_GET_SIZE(keywords) : 0; // owned by this call, so borrowed values stay
    Fields *const instance = (Fields *)type->tp_alloc(type, 0);
    if (!instance) return NULL;
    Py_ssize_t matched = 0;
    for (size_t index = 0; index != schema->count; ++index) {
        uform_py_field_t const *const field = &schema->fields[index];
        PyObject *const value = keywords ? PyDict_GetItemString(keywords, field->name) : NULL;
        instance->values[index] = value ? field_accept(field, value) : field_default(field);
        matched += value != NULL;
        if (!instance->values[index]) goto failed;
    }
    if (matched != given) {
        PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument among %R",
                     type->tp_name + sizeof("uform.") - 1, keywords);
        goto failed;
    }
    return (PyObject *)instance;

failed:
    Py_DECREF(instance);
    return NULL;
}

static void Fields_dealloc(Fields *self) {
    for (size_t index = 0; index != uform_py_fields_max_k; ++index) Py_XDECREF(self->values[index]);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Fields_repr(Fields *self) {
    uform_py_fields_t const *const schema = (uform_py_fields_t const *)Py_TYPE(self);
    PyObject *const parts = PyList_New(0);
    for (size_t index = 0; parts && index != schema->count; ++index) {
        PyObject *const part = PyUnicode_FromFormat("%s=%R", schema->fields[index].name, self->values[index]);
        if (!part || PyList_Append(parts, part) != 0) {
            Py_XDECREF(part);
            Py_DECREF(parts);
            return NULL;
        }
        Py_DECREF(part);
    }
    PyObject *const separator = parts ? PyUnicode_FromString(", ") : NULL;
    PyObject *const joined = separator ? PyUnicode_Join(separator, parts) : NULL;
    PyObject *const repr = joined
                               ? PyUnicode_FromFormat("%s(%U)", Py_TYPE(self)->tp_name + sizeof("uform.") - 1, joined)
                               : NULL;
    Py_XDECREF(joined);
    Py_XDECREF(separator);
    Py_XDECREF(parts);
    return repr;
}

int uform_py_fields_ready(uform_py_fields_t *fields) {
    PyMemberDef *const members = (PyMemberDef *)PyMem_Calloc(fields->count + 1, sizeof(PyMemberDef));
    if (!members) {
        PyErr_NoMemory();
        return -1;
    }
    for (size_t index = 0; index != fields->count; ++index)
        members[index] = (PyMemberDef) {fields->fields[index].name, Py_T_OBJECT_EX,
                                        (Py_ssize_t)(offsetof(Fields, values) + index * sizeof(PyObject *)),
                                        Py_READONLY, fields->fields[index].doc};
    fields->type.tp_basicsize = sizeof(Fields);
    fields->type.tp_flags = Py_TPFLAGS_DEFAULT;
    fields->type.tp_new = Fields_tp_new;
    fields->type.tp_dealloc = (destructor)Fields_dealloc;
    fields->type.tp_repr = (reprfunc)Fields_repr;
    fields->type.tp_members = members;
    return PyType_Ready(&fields->type);
}

int uform_py_fields_write(PyObject *object, char const *field, uform_py_fields_t const *fields, void *target) {
    if (!object || object == Py_None) return 0;
    if (!Py_IS_TYPE(object, (PyTypeObject *)&fields->type)) {
        PyErr_Format(PyExc_TypeError, "the '%s' option takes a uform.%s, got %S", field,
                     fields->type.tp_name + sizeof("uform.") - 1, Py_TYPE(object));
        return -1;
    }
    for (size_t index = 0; index != fields->count; ++index) {
        uform_py_field_t const *const described = &fields->fields[index];
        PyObject *const value = ((Fields *)object)->values[index];
        char *const at = (char *)target + described->offset;
        switch (described->kind) {
        case uform_py_field_size_k:
            if (uform_py_size_from_object(value, described->name, (size_t *)(void *)at) != 0) return -1;
            break;
        case uform_py_field_float_k: *(float *)(void *)at = (float)PyFloat_AS_DOUBLE(value); break;
        case uform_py_field_double_k: *(double *)(void *)at = PyFloat_AS_DOUBLE(value); break;
        case uform_py_field_enum_k: *(int *)(void *)at = (int)PyLong_AsLong(value); break;
        case uform_py_field_struct_k:
            if (uform_py_fields_write(value, described->name, (uform_py_fields_t const *)described->detail, at) != 0)
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
static Deferred *deferred_of(void *call) { return (Deferred *)(void *)((char *)call - offsetof(Deferred, call)); }

void *uform_py_call_open(uform_py_receiver_t const *receiver, uform_py_verb_t const *verb, void *stack) {
    void *call = stack;
    if (receiver->completion == uform_completion_deferred_k) {
        Deferred *const deferred = PyObject_GC_NewVar(Deferred, &DeferredType, (Py_ssize_t)verb->call_bytes);
        if (!deferred) return NULL;
        deferred->verb = verb;
        deferred->origin = Py_NewRef(receiver->origin);
        deferred->answer = NULL;
        deferred->raised = NULL;
        call = deferred->call;
        memset(call, 0, verb->call_bytes);
        PyObject_GC_Track(deferred);
        return call;
    }
    memset(call, 0, verb->call_bytes);
    return call;
}

/** Answers a final @p call: a callback's first exception, else the outcome's failure, else
 *  the verb's answer. */
static PyObject *call_answer(uform_py_verb_t const *verb, void *call) {
    uform_py_call_t *const header = (uform_py_call_t *)call;
    if (header->raised) {
        PyErr_SetRaisedException(header->raised);
        header->raised = NULL;
        return NULL;
    }
    return uform_py_raise_on_outcome(&header->outcome) == 0 ? verb->answer(call) : NULL;
}

/** Lets go of what @p call gathered, keeping whatever exception is set. */
static void call_release(uform_py_verb_t const *verb, void *call) {
    PyObject *const raised = PyErr_GetRaisedException();
    verb->release(call);
    PyErr_SetRaisedException(raised);
}

PyObject *uform_py_call_run(uform_py_receiver_t const *receiver, uform_py_verb_t const *verb, void *call) {
    Py_BEGIN_ALLOW_THREADS;
    verb->submit(call);
    Py_END_ALLOW_THREADS;
    if (receiver->completion == uform_completion_deferred_k) return (PyObject *)deferred_of(call);
    PyObject *const answer = call_answer(verb, call);
    call_release(verb, call);
    return answer;
}

PyObject *uform_py_call_discard(uform_py_receiver_t const *receiver, uform_py_verb_t const *verb, void *call) {
    call_release(verb, call);
    if (receiver->completion == uform_completion_blocking_k) return NULL;
    Deferred *const deferred = deferred_of(call);
    deferred->verb = NULL;
    PyObject *const raised = PyErr_GetRaisedException();
    Py_DECREF(deferred);
    PyErr_SetRaisedException(raised);
    return NULL;
}

void uform_py_call_release(void *call) {
    uform_py_call_t *const header = (uform_py_call_t *)call;
    Py_CLEAR(header->raised);
    if (header->arena) uform_py_arena_unclaim(header->arena);
    header->arena = NULL;
}

PyObject *uform_py_call_answer_none(void *call) { Py_RETURN_NONE; }

int uform_py_callback_from_object(PyObject *object, char const *field, uform_py_call_t *call,
                                  uform_py_callback_t *callback) {
    callback->call = call;
    callback->callable = NULL;
    if (!object || object == Py_None) return 0;
    if (!PyCallable_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' handler must be callable, got %S", field, Py_TYPE(object));
        return -1;
    }
    callback->callable = Py_NewRef(object);
    return 0;
}

/** Moves the exception the calling thread holds into @p call unless an earlier one is there, a
 *  store that is the fence a later report reads; the GIL is held. */
static void callback_latch(uform_py_call_t *call) {
    PyObject *const raised = PyErr_GetRaisedException();
    PyObject *expected = NULL;
    if (raised && !atomic_compare_exchange_strong((_Atomic(PyObject *) *)&call->raised, &expected, raised))
        Py_DECREF(raised);
}

/** Whether a callback of @p call already raised, so a later one returns at once. */
static int callback_raised(uform_py_call_t *call) {
    return atomic_load_explicit((_Atomic(PyObject *) *)&call->raised, memory_order_acquire) != NULL;
}

/**
 *  @brief Re-enters Python for one progress report, from the thread running the call.
 *
 *  Reports are serialized, but a deferred call runs on the executor's thread 0, not on its caller,
 *  so the GIL is taken here rather than assumed. A handler answering @c None carries on, and one
 *  that raises also stops the call, so it unwinds at the next slice instead of running on.
 */
static uform_continuation_t progress_trampoline(size_t completed, size_t total, void *state) {
    uform_py_callback_t const *const callback = (uform_py_callback_t const *)state;
    if (callback_raised(callback->call)) return uform_cancel_k;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    uform_continuation_t continuation = uform_cancel_k;
    PyObject *const completed_object = PyLong_FromSize_t(completed);
    PyObject *const total_object = completed_object ? PyLong_FromSize_t(total) : NULL;
    PyObject *const arguments[2] = {completed_object, total_object};
    PyObject *const verdict = total_object ? PyObject_Vectorcall(callback->callable, arguments, 2, NULL) : NULL;
    if (verdict) {
        int const truth = verdict == Py_None ? 1 : PyObject_IsTrue(verdict);
        continuation = truth > 0 ? uform_continue_k : uform_cancel_k;
        Py_DECREF(verdict);
    }
    Py_XDECREF(total_object);
    Py_XDECREF(completed_object);
    if (PyErr_Occurred()) callback_latch(callback->call);
    PyGILState_Release(gil_state);
    return continuation;
}

uform_progress_t uform_py_progress_registered(uform_py_callback_t *callback, size_t steps) {
    return (uform_progress_t) {.callback = callback->callable ? progress_trampoline : NULL,
                               .state = callback,
                               .steps = steps};
}

#pragma endregion Calls

#pragma region Receivers and Deferred Calls

/** A model or trainer bound to the executor its verbs run on; its type alone chooses
 *  how they complete. */
typedef struct On {
    PyObject ob_base;

    /** The model or trainer the verbs address, held. */
    PyObject *target;

    /** The @c Executor the calls run on, held. */
    PyObject *executor;
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

int uform_py_receiver_from_object(PyObject *self, PyTypeObject *type, uform_py_receiver_t *receiver) {
    int const deferred = Py_IS_TYPE(self, &DeferredOnType);
    receiver->origin = self;
    receiver->target = self;
    receiver->executor = NULL;
    receiver->completion = deferred ? uform_completion_deferred_k : uform_completion_blocking_k;
    if (deferred || Py_IS_TYPE(self, &OnType)) receiver->target = ((On *)self)->target;
    if (!Py_IS_TYPE(receiver->target, type)) {
        PyErr_Format(PyExc_TypeError, "the verb addresses a uform.%s, got a receiver over %S",
                     type->tp_name + sizeof("uform.") - 1, Py_TYPE(receiver->target));
        return -1;
    }
    if (receiver->target == self) return 0;
    return uform_py_executor_from_object(((On *)self)->executor, &receiver->executor);
}

PyObject *uform_py_on_new(PyTypeObject *type, PyObject *target, PyObject *executor) {
    uform_executor_t handle = NULL;
    if (uform_py_executor_from_object(executor, &handle) != 0) return NULL;
    On *const on = PyObject_New(On, type);
    if (!on) return NULL;
    on->target = Py_NewRef(target);
    on->executor = Py_NewRef(executor);
    return (PyObject *)on;
}

static PyObject *On_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    PyErr_SetString(PyExc_TypeError, "receivers are built through model.on(executor) or model.deferred(executor)");
    return NULL;
}

static void On_dealloc(On *self) {
    Py_XDECREF(self->executor);
    Py_XDECREF(self->target);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *On_repr(On *self) {
    return PyUnicode_FromFormat("%s(%R, %R)", Py_TYPE(self)->tp_name + sizeof("uform.") - 1, self->target,
                                self->executor);
}

PyTypeObject OnType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.On",
    .tp_doc = doc_On,
    .tp_basicsize = sizeof(On),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = On_tp_new,
    .tp_dealloc = (destructor)On_dealloc,
    .tp_repr = (reprfunc)On_repr,
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
};

static char const doc_Deferred[] =                                                              //
    "Deferred[Answer]\n"                                                                        //
    "\n"                                                                                        //
    "A call a `DeferredOn` submitted, which answers once it is final.\n"                        //
    "\n"                                                                                        //
    "`await` suspends the coroutine until the call is final, and `wait()` blocks the thread\n"  //
    "instead. Either answers what the blocking verb would have answered, or raises what it\n"   //
    "would have raised, as many times as it is asked; `status` polls without waiting. It\n"     //
    "holds the call and the buffers it reads until the call is final, so dropping it\n"         //
    "waits, and holds its arena as the blocking verb would.\n"                                  //
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
static uform_outcome_t *deferred_outcome(Deferred *deferred) {
    return &((uform_py_call_t *)(void *)deferred->call)->outcome;
}

/** The status of the call @p deferred holds, as the library finishes it elsewhere. */
static uform_status_t deferred_status(Deferred *deferred) { return uform_outcome_status(deferred_outcome(deferred)); }

/** Blocks with the thread state detached until the call @p deferred holds is final. */
static void deferred_block(Deferred *deferred) {
    if (deferred_status(deferred) != uform_status_deferred_k) return;
    uform_outcome_t *const outcome = deferred_outcome(deferred);
    Py_BEGIN_ALLOW_THREADS;
    uform_outcome_wait(outcome);
    Py_END_ALLOW_THREADS;
}

/**
 *  @brief Finishes a final @p self exactly once: the thread taking its verb answers, publishes,
 *      then lets go of its inputs.
 *
 *  Answering may run Python, which lets another thread in even under the GIL, so the verb is taken
 *  before answering and the answer published after it, never one check spanning both.
 */
static void deferred_finish(Deferred *self) {
    uform_py_verb_t const *finishing = NULL;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(self);
#endif
    finishing = self->verb;
    self->verb = NULL;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    if (!finishing) return;
    PyObject *const answer = call_answer(finishing, self->call);
    PyObject *const raised = answer ? NULL : PyErr_GetRaisedException();
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(self);
#endif
    self->answer = answer;
    self->raised = raised;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    call_release(finishing, self->call);
}

/** Answers what a final @p self finished with, finishing it first when nobody has. */
static PyObject *deferred_answer(Deferred *self) {
    deferred_finish(self);
    for (;;) {
        PyObject *answer = NULL;
        int published = 0;
#if PY_VERSION_HEX >= 0x030D0000
        Py_BEGIN_CRITICAL_SECTION(self);
#endif
        published = self->answer || self->raised;
        if (self->raised) PyErr_SetRaisedException(Py_NewRef(self->raised));
        else if (self->answer) answer = Py_NewRef(self->answer);
#if PY_VERSION_HEX >= 0x030D0000
        Py_END_CRITICAL_SECTION();
#endif
        if (published) return answer;
        Py_BEGIN_ALLOW_THREADS; // another thread is still answering, so this one yields until it publishes
        Py_END_ALLOW_THREADS;
    }
}

static char const doc_Deferred_status[] =                                                        //
    "The status of the call: `Status.DEFERRED` until it is final, then the final one, such as\n" //
    "`Status.SUCCESS`. Calls `uform_outcome_status`.";

static PyObject *Deferred_get_status(Deferred *self, void *closure) {
    uform_status_t const status = deferred_status(self);
    if (status != uform_status_deferred_k) deferred_finish(self); // lets go of the inputs, as a poller never waits
    return uform_py_enum_to_object(&uform_py_status, (int)status);
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
    "    another call held the executor.";

static PyObject *Deferred_wait(Deferred *self, PyObject *unused) {
    deferred_block(self);
    return deferred_answer(self);
}

/** Calls the method @p name of @p self with up to two positional arguments, NULL ones left out. */
static PyObject *method_call(PyObject *self, char const *name, PyObject *first, PyObject *second) {
    PyObject *const method = PyUnicode_InternFromString(name);
    if (!method) return NULL;
    PyObject *const arguments[3] = {self, first, second};
    size_t const count = 1 + (first != NULL) + (second != NULL);
    PyObject *const result = PyObject_VectorcallMethod(method, arguments, count, NULL);
    Py_DECREF(method);
    return result;
}

/** Answers the iterator of an `asyncio.Future` that `wait()` resolves on the running
 *  loop's default executor. */
static PyObject *Deferred_await(PyObject *self) {
    PyObject *const asyncio = PyImport_ImportModule("asyncio");
    PyObject *const loop = asyncio ? method_call(asyncio, "get_running_loop", NULL, NULL) : NULL;
    Py_XDECREF(asyncio);
    PyObject *const wait = loop ? PyObject_GetAttrString(self, "wait") : NULL;
    PyObject *const future = wait ? method_call(loop, "run_in_executor", Py_None, wait) : NULL;
    Py_XDECREF(wait);
    Py_XDECREF(loop);
    PyObject *const awaiting = future ? method_call(future, "__await__", NULL, NULL) : NULL;
    Py_XDECREF(future);
    return awaiting;
}

static void Deferred_dealloc(Deferred *self) {
    PyObject_GC_UnTrack(self);
    if (self->verb) { // the library writes into the call and reads its borrowed inputs until it is final
        deferred_block(self);
        call_release(self->verb, self->call);
    }
    Py_XDECREF(self->answer);
    Py_XDECREF(self->raised);
    Py_XDECREF(self->origin);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/** Visits what @p self holds; an unfinished call's own objects stay unvisited, so every cycle
 *  through them waits for a later collection rather than being cleared while the library
 *  still reads them. */
static int Deferred_traverse(Deferred *self, visitproc visit, void *arg) {
    Py_VISIT(self->origin);
    Py_VISIT(self->answer);
    Py_VISIT(self->raised);
    if (!self->verb || deferred_status(self) == uform_status_deferred_k) return 0;
    uform_py_call_t const *const header = (uform_py_call_t const *)(void const *)self->call;
    Py_VISIT(header->raised);
    Py_VISIT(header->arena);
    return self->verb->traverse ? self->verb->traverse(self->call, visit, arg) : 0;
}

/** Breaks a garbage cycle through a final @p self; an unfinished one is left whole, as its
 *  traversal hid the cycle. */
static int Deferred_clear(Deferred *self) {
    if (deferred_status(self) == uform_status_deferred_k) return 0;
    uform_py_verb_t const *const releasing = self->verb;
    self->verb = NULL;
    if (releasing) call_release(releasing, self->call);
    Py_CLEAR(self->answer);
    Py_CLEAR(self->raised);
    Py_CLEAR(self->origin);
    return 0;
}

static PyObject *Deferred_repr(Deferred *self) {
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

static char const doc_Executor[] =                                                            //
    "Executor\n"                                                                              //
    "\n"                                                                                      //
    "The threads or streams a verb runs on, the activation scratch and the staging, shared\n" //
    "across models.\n"                                                                        //
    "\n"                                                                                      //
    "An executor owns host threads or device streams; a model owns weights. Keeping them\n"   //
    "apart is what lets one executor serve several models, and what makes the cost of\n"      //
    "spawning visible at the line that spawns rather than hidden in a constructor.\n"         //
    "\n"                                                                                      //
    "Construct one through the residency it drives - `for_ram`, `on_pool` or `for_vram` -\n"  //
    "since each spawns something different and takes different arguments. A receiver names\n" //
    "it: `model.on(executor)` blocks on it, `model.deferred(executor)` submits to it. It\n"   //
    "runs one call at a time, and a call finding another one holding it raises a\n"           //
    "`ContendedError` once waited for. In a `with` block, it waits for its calls and\n"       //
    "closes on exit, or leaves both to the last GPU arena it made that is alive.\n"           //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> with uform.Executor.for_ram() as executor:\n"                                      //
    "  ...     executor.drain(uform.Mood.PARKED)";

/** Wraps a fresh handle, or frees it when the wrapper cannot be allocated; a new
 *  reference, or NULL. */
static PyObject *executor_adopt(PyTypeObject *type, uform_executor_t handle) {
    Executor *const executor = (Executor *)type->tp_alloc(type, 0);
    if (!executor) {
        uform_executor_free(handle);
        return NULL;
    }
    executor->handle = handle;
    return (PyObject *)executor;
}

static PyObject *Executor_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    PyErr_SetString(PyExc_TypeError, "Executor is built through Executor.for_ram(), .on_pool() or .for_vram()");
    return NULL;
}

/** Frees @p handle, waiting in C for every deferred call naming it; NULL is safe. */
static void executor_free(uform_executor_t handle) {
    if (!handle) return;
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_free(handle);
    Py_END_ALLOW_THREADS;
}

/** Closes @p executor once, fenced against a concurrent close; while a GPU arena it made is alive,
 *  the handle is left for the last of them to free. */
static void executor_close(Executor *executor) {
    uform_executor_t handle = NULL;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(executor);
#endif
    handle = executor->handle;
    executor->handle = NULL;
    if (handle && executor->arenas) {
        executor->retained = handle;
        handle = NULL;
    }
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    executor_free(handle);
}

/** Takes a GPU arena's hold on @p executor, whose handle must outlive the arena. */
static PyObject *executor_hold(PyObject *executor) {
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(executor);
#endif
    ++((Executor *)executor)->arenas;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    return Py_NewRef(executor);
}

/** Drops a GPU arena's hold on @p executor, freeing the handle a close left to the last of them. */
static void executor_release(PyObject *executor) {
    uform_executor_t retained = NULL;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(executor);
#endif
    if (!--((Executor *)executor)->arenas) {
        retained = ((Executor *)executor)->retained;
        ((Executor *)executor)->retained = NULL;
    }
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    executor_free(retained);
    Py_DECREF(executor);
}

static void Executor_dealloc(Executor *executor) {
    executor_close(executor);
    Py_TYPE(executor)->tp_free((PyObject *)executor);
}

static PyObject *Executor_enter(PyObject *self, PyObject *unused) { return Py_NewRef(self); }

static PyObject *Executor_exit(PyObject *self, PyObject *const *args, Py_ssize_t count) {
    executor_close((Executor *)self);
    Py_RETURN_NONE;
}

int uform_py_executor_from_object(PyObject *object, uform_executor_t *executor) {
    if (!PyObject_TypeCheck(object, &ExecutorType)) {
        PyErr_Format(PyExc_TypeError, "expected an Executor, got %S", Py_TYPE(object));
        return -1;
    }
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(object);
#endif
    *executor = ((Executor *)object)->handle;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    if (!*executor) // a null executor would quietly run the call on the calling thread
        return uform_py_raise(uform_status_invalid_argument_k, "the executor was closed");
    return 0;
}

int uform_py_executor_optional(PyObject *object, uform_executor_t *executor) {
    *executor = NULL;
    return !object || object == Py_None ? 0 : uform_py_executor_from_object(object, executor);
}

static char const doc_Executor_for_ram[] =                                                      //
    "for_ram(threads=0, *, memory_domains=None, cpu_class=None) -> Executor\n"                  //
    "\n"                                                                                        //
    "Creates an executor of host threads for RAM-resident models, interleaved across the\n"     //
    "compute domains local to the named memory domains.\n"                                      //
    "Calls `uform_executor_for_ram`.\n"                                                         //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  threads (int, optional): Threads to spawn; 0 takes 7/8 of the cores the domains\n"       //
    "    carry.\n"                                                                              //
    "  memory_domains (Sequence[int], optional): NUMA node IDs of the models it will\n"         //
    "    drive; None derives every node the process may allocate on.\n"                         //
    "  cpu_class (CPUClass, optional): The class to dispatch as; None derives the CPU's own.\n" //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> with uform.Executor.for_ram(threads=8) as executor:\n"                               //
    "  ...     executor.drain()";

static PyObject *Executor_for_ram(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    PyObject *threads_object, *memory_domains_object, *cpu_class_object;
    uform_py_argument_t const arguments[] = {
        {"threads", &threads_object},
        {"memory_domains", &memory_domains_object},
        {"cpu_class", &cpu_class_object},
    };
    size_t threads = 0;
    int cpu_class = 0;
    int32_t *memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (uform_py_arguments_parse("for_ram", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_size_from_object(threads_object, "threads", &threads) != 0 ||
        uform_py_class_from_object(cpu_class_object, "cpu_class", &uform_py_cpu_class, &cpu_class) != 0 ||
        uform_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                      &memory_domains_count) != 0)
        return NULL;

    uform_executor_t handle = NULL;
    uform_outcome_t outcome = {0};
    uform_executor_for_ram_t executor_arguments = {
        .executor = &handle,
        .outcome = &outcome,
        .ram = {memory_domains, memory_domains_count, (uform_cpu_class_t)cpu_class},
        .threads = threads};
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_for_ram(&executor_arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);

    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

static char const doc_Executor_on_pool[] =                                                     //
    "on_pool(pool, /, *, memory_domains=None, cpu_class=None) -> Executor\n"                   //
    "\n"                                                                                       //
    "Drives RAM-resident models on a ForkUnion pool another library already spawned, so\n"     //
    "UForm, USearch and UStore in one process share one set of threads while each keeps its\n" //
    "own scratch. Borrowing is not designed across projects yet, so it raises\n"               //
    "`UnsupportedError` until it is.\n"                                                        //
    "Calls `uform_executor_on_pool`.\n"                                                        //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  pool (int): The address of the `fu_pool_t` to borrow, as `ctypes` and `cffi` hand\n"    //
    "    pointers over; the pool must outlive the executor.\n"                                 //
    "  memory_domains (Sequence[int], optional): NUMA node IDs of the models it will\n"        //
    "    drive; None derives every node the process may allocate on.\n"                        //
    "  cpu_class (CPUClass, optional): The class to dispatch as; None derives the CPU's own.";

static PyObject *Executor_on_pool(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    PyObject *pool_object, *memory_domains_object, *cpu_class_object;
    uform_py_argument_t const arguments[] = {
        {"pool", &pool_object},
        {"memory_domains", &memory_domains_object},
        {"cpu_class", &cpu_class_object},
    };
    int cpu_class = 0;
    if (uform_py_arguments_parse("on_pool", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_class_from_object(cpu_class_object, "cpu_class", &uform_py_cpu_class, &cpu_class) != 0)
        return NULL;

    PyObject *const address = PyNumber_Index(pool_object);
    if (!address) return NULL;
    void *const pool = PyLong_AsVoidPtr(address);
    Py_DECREF(address);
    if (!pool && PyErr_Occurred()) return NULL;

    int32_t *memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (uform_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                      &memory_domains_count) != 0)
        return NULL;

    uform_executor_t handle = NULL;
    uform_outcome_t outcome = {0};
    uform_executor_on_pool_t executor_arguments = {
        .executor = &handle,
        .outcome = &outcome,
        .ram = {memory_domains, memory_domains_count, (uform_cpu_class_t)cpu_class},
        .pool = pool};
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_on_pool(&executor_arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);

    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

static char const doc_Executor_for_vram[] =                                                     //
    "for_vram(devices=None, *, scratch_limit=0, gpu_class=None, "                               //
    "memory_domains=None, cpu_class=None) -> Executor\n"                                        //
    "\n"                                                                                        //
    "Opens one submission stream on every named device, all of them driven by the host\n"       //
    "thread running the call, and spawns the host threads that decode what the devices\n"       //
    "cannot.\n"                                                                                 //
    "Calls `uform_executor_for_vram`.\n"                                                        //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  devices (Sequence[int], optional): Positions in `gpu_devices()` to drive; None\n"        //
    "    derives one free device.\n"                                                            //
    "  scratch_limit (int, optional): The ceiling on scratch bytes per device; 0 lets\n"        //
    "    scratch grow as batches ask, and a batch larger than the ceiling runs in passes.\n"    //
    "  gpu_class (GPUClass, optional): The class every device dispatches as; None derives\n"    //
    "    the devices' own, which must agree.\n"                                                 //
    "  memory_domains (Sequence[int], optional): NUMA node IDs whose cores the host\n"          //
    "    decoders take; None derives every node the process may allocate on.\n"                 //
    "  cpu_class (CPUClass, optional): The class the host decoders dispatch as; None derives\n" //
    "    the CPU's own.\n"                                                                      //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> executor = uform.Executor.for_vram([0, 1])   # doctest: +SKIP";

static PyObject *Executor_for_vram(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {
    PyObject *devices_object, *scratch_limit_object, *gpu_class_object, *memory_domains_object, *cpu_class_object;
    uform_py_argument_t const arguments[] = {
        {"devices", &devices_object},     {"scratch_limit", &scratch_limit_object},
        {"gpu_class", &gpu_class_object}, {"memory_domains", &memory_domains_object},
        {"cpu_class", &cpu_class_object},
    };
    size_t scratch_limit = 0;
    int gpu_class = 0, cpu_class = 0;
    int32_t *devices = NULL, *memory_domains = NULL;
    size_t devices_count = 0, memory_domains_count = 0;
    if (uform_py_arguments_parse("for_vram", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_size_from_object(scratch_limit_object, "scratch_limit", &scratch_limit) != 0 ||
        uform_py_class_from_object(gpu_class_object, "gpu_class", &uform_py_gpu_class, &gpu_class) != 0 ||
        uform_py_class_from_object(cpu_class_object, "cpu_class", &uform_py_cpu_class, &cpu_class) != 0)
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
        .vram = {devices, devices_count, (uform_gpu_class_t)gpu_class},
        .ram = {memory_domains, memory_domains_count, (uform_cpu_class_t)cpu_class},
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

static PyObject *Executor_drain(Executor *executor, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    PyObject *mood_object;
    uform_py_argument_t const arguments[] = {{"mood", &mood_object}};
    int mood = uform_mood_parked_k;
    uform_executor_t handle = NULL;
    if (uform_py_arguments_parse("drain", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        (mood_object && uform_py_enum_from_object(mood_object, "mood", &uform_py_mood, &mood) != 0) ||
        uform_py_executor_from_object((PyObject *)executor, &handle) != 0)
        return NULL;

    uform_outcome_t outcome = {0};
    uform_executor_drain_t drain_arguments = {.executor = handle, .outcome = &outcome, .mood = (uform_mood_t)mood};
    Py_BEGIN_ALLOW_THREADS;
    uform_executor_drain(&drain_arguments);
    Py_END_ALLOW_THREADS;
    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *Executor_repr(Executor *executor) {
    return PyUnicode_FromFormat("uform.Executor(at=%p)", (void *)executor->handle);
}

static PyMethodDef Executor_methods[] = {
    {"for_ram", (PyCFunction)Executor_for_ram, uform_py_method_flags_k | METH_CLASS, doc_Executor_for_ram},
    {"on_pool", (PyCFunction)Executor_on_pool, uform_py_method_flags_k | METH_CLASS, doc_Executor_on_pool},
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

/** The parts of the DLPack 1.x ABI this binding exchanges, spelled as @c dlpack.h spells them. */
typedef enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLMetal = 8,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLCUDAManaged = 13,
} DLDeviceType;

typedef enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
} DLDataTypeCode;

typedef struct DLPackVersion {
    uint32_t major;
    uint32_t minor;
} DLPackVersion;

typedef struct DLDevice {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

typedef struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct DLTensor {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;

typedef struct DLManagedTensorVersioned {
    DLPackVersion version;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensorVersioned *self);
    uint64_t flags;
    DLTensor dl_tensor;
} DLManagedTensorVersioned;

/** The buffer formats a DLPack tensor crosses as, and the element type each stands for. */
static struct {
    char const *format;
    DLDataType dtype;
} const dlpack_formats[] = {
    {"B", {kDLUInt, 8, 1}},   {"b", {kDLInt, 8, 1}},    {"I", {kDLUInt, 32, 1}},
    {"i", {kDLInt, 32, 1}},   {"Q", {kDLUInt, 64, 1}},  {"e", {kDLFloat, 16, 1}},
    {"f", {kDLFloat, 32, 1}}, {"d", {kDLFloat, 64, 1}}, {"N", {kDLUInt, sizeof(size_t) * 8, 1}},
};

/** Hands the tensor a consumed capsule holds back to its producer, and frees the extents
 *  its view borrowed. */
static void dlpack_capsule_release(PyObject *capsule) {
    PyMem_Free(PyCapsule_GetContext(capsule));
    if (PyCapsule_IsValid(capsule, "used_dltensor_versioned")) {
        DLManagedTensorVersioned *const managed = PyCapsule_GetPointer(capsule, "used_dltensor_versioned");
        if (managed->deleter) managed->deleter(managed);
    }
    else if (PyCapsule_IsValid(capsule, "used_dltensor")) {
        DLManagedTensor *const managed = PyCapsule_GetPointer(capsule, "used_dltensor");
        if (managed->deleter) managed->deleter(managed);
    }
}

/**
 *  @brief Consumes the DLPack export of @p object into @p view, whose @c obj becomes the capsule,
 *      so releasing the view hands the tensor back to its producer.
 *  @return 1 when it does, 0 when @p object exports no DLPack, or -1 with an exception set.
 *
 *  Only C-contiguous tensors cross. Host-readable memory always does, and CUDA and ROCm memory only
 *  where @p device_allowed, since the binding reads the other inputs itself.
 */
static int dlpack_acquire(PyObject *object, int device_allowed, Py_buffer *view) {
    PyObject *const exporter = PyObject_GetAttrString(object, "__dlpack__");
    if (!exporter) {
        PyErr_Clear();
        return 0;
    }
    PyObject *const arguments = PyTuple_New(0);
    PyObject *const keywords = PyDict_New();
    PyObject *const version = PyTuple_New(2);
    if (version) {
        PyTuple_SET_ITEM(version, 0, PyLong_FromLong(1));
        PyTuple_SET_ITEM(version, 1, PyLong_FromLong(0));
    }
    int const packed = arguments && keywords && version && PyTuple_GET_ITEM(version, 0) &&
                       PyTuple_GET_ITEM(version, 1) && PyDict_SetItemString(keywords, "stream", Py_None) == 0 &&
                       PyDict_SetItemString(keywords, "max_version", version) == 0;
    PyObject *const capsule = packed ? PyObject_Call(exporter, arguments, keywords) : NULL;
    Py_XDECREF(version);
    Py_XDECREF(keywords);
    Py_XDECREF(arguments);
    Py_DECREF(exporter);
    if (!capsule) { // a producer refusing this export may still lend the buffer protocol
        PyErr_Clear();
        return 0;
    }
    int const versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    void *const managed = PyCapsule_GetPointer(capsule, versioned ? "dltensor_versioned" : "dltensor");
    if (!managed) {
        Py_DECREF(capsule);
        return -1;
    }
    DLTensor const *const tensor = versioned ? &((DLManagedTensorVersioned *)managed)->dl_tensor
                                             : &((DLManagedTensor *)managed)->dl_tensor;
    DLDeviceType const device = tensor->device.device_type;
    int const host_readable = device == kDLCPU || device == kDLCUDAHost || device == kDLROCMHost ||
                              device == kDLCUDAManaged;
    char const *format = NULL;
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
    if (!host_readable && !(device_allowed && (device == kDLCUDA || device == kDLROCM)))
        failed = uform_py_raise(uform_status_device_k, "this input cannot take memory of DLPack device type %d",
                                (int)device);
    else if (!format)
        failed = uform_py_raise(uform_status_unsupported_layout_k, "the DLPack tensor holds items no format names");
    else if (!contiguous)
        failed = uform_py_raise(uform_status_unsupported_layout_k, "the DLPack tensor must be C-contiguous");
    Py_ssize_t *const extents = failed ? NULL
                                       : (Py_ssize_t *)PyMem_Malloc((size_t)(tensor->ndim + 1) * sizeof(Py_ssize_t));
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
    *view = (Py_buffer) {.buf = (char *)tensor->data + tensor->byte_offset,
                         .obj = capsule,
                         .len = items * item_bytes,
                         .itemsize = item_bytes,
                         .readonly = 1,
                         .ndim = tensor->ndim,
                         .format = (char *)format,
                         .shape = extents};
    return 1;
}

/** Acquires @p object into @p view through DLPack first, then the buffer protocol under @p flags;
 *  0, or -1 with an exception set. */
static int buffer_acquire(PyObject *object, int device_allowed, int flags, Py_buffer *view) {
    int const consumed = dlpack_acquire(object, device_allowed, view);
    return consumed ? (consumed > 0 ? 0 : -1) : PyObject_GetBuffer(object, view, flags);
}

#pragma endregion DLPack

#pragma region Arenas

static char const doc_Arena[] =                                                                //
    "Arena(executor=None, /)\n"                                                                //
    "\n"                                                                                       //
    "The memory every output whose length scales with a call lands in - embeddings, states\n"  //
    "and their offsets - which every such verb requires and each call rewinds, so a steady\n"  //
    "loop allocates once.\n"                                                                   //
    "\n"                                                                                       //
    "The executor makes it where its calls compute, host pages or the memory of its GPU,\n"    //
    "and none makes host pages for calls on the calling thread. Results are zero-copy views\n" //
    "over it. Like a `bytearray`, it refuses to rewind with `ContendedError` while a buffer\n" //
    "over its outputs is exported, and a result read after a later call rewound it raises\n"   //
    "`InvalidArgumentError`. A deferred call holds it as a blocking one does, until it is\n"   //
    "waited on or dropped. Results over device memory export `__dlpack__`, and host reads\n"   //
    "of them raise `DeviceError`. A GPU arena serves only calls on the executor that made\n"   //
    "it and holds that executor until it is freed, so an executor closed first keeps its\n"    //
    "streams until then.\n"                                                                    //
    "Calls `uform_arena_init`.\n"                                                              //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor, optional): The executor whose calls it serves; None serves the\n"   //
    "    calling thread.\n"                                                                    //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> model = uform.Model.load_ram_buffer(b'any bytes the stub accepts')\n"               //
    "  >>> arena = uform.Arena()\n"                                                            //
    "  >>> first = memoryview(model.text_encode(['first'], arena).embeddings)\n"               //
    "  >>> first.release()\n"                                                                  //
    "  >>> memoryview(model.text_encode(['second'], arena).embeddings).shape\n"                //
    "  (1, 64)";

static PyObject *Arena_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    Py_ssize_t const positional_args_count = PyTuple_GET_SIZE(args);
    if (positional_args_count > 1 || (keywords && PyDict_GET_SIZE(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Arena() takes one optional positional argument, the executor");
        return NULL;
    }
    PyObject *const executor_object = positional_args_count ? PyTuple_GET_ITEM(args, 0) : Py_None;
    uform_executor_t executor = NULL;
    if (uform_py_executor_optional(executor_object, &executor) != 0) return NULL;

    uform_arena_t handle = NULL;
    uform_outcome_t outcome = {0};
    uform_arena_init_t arguments = {.arena = &handle, .executor = executor, .outcome = &outcome};
    uform_arena_init(&arguments);
    if (uform_py_raise_on_outcome(&outcome) != 0) return NULL;
    Arena *const arena = (Arena *)type->tp_alloc(type, 0);
    if (!arena) {
        uform_arena_free(handle);
        return NULL;
    }
    arena->handle = handle;
    uform_arena_description_t description = {0};
    uform_arena_describe(&(uform_arena_describe_t) {.arena = handle, .outcome = &outcome, .description = &description});
    if (uform_py_raise_on_outcome(&outcome) != 0) {
        Py_DECREF(arena);
        return NULL;
    }
    if (description.residency == uform_residency_vram_k) // a GPU arena frees on its executor's first stream
        arena->executor = executor_hold(executor_object);
    return (PyObject *)arena;
}

static void Arena_dealloc(Arena *self) {
    Py_BEGIN_ALLOW_THREADS;
    uform_arena_free(self->handle);
    Py_END_ALLOW_THREADS;
    if (self->executor) executor_release(self->executor);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyTypeObject ArenaType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.Arena",
    .tp_doc = doc_Arena,
    .tp_basicsize = sizeof(Arena),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Arena_tp_new,
    .tp_dealloc = (destructor)Arena_dealloc,
};

int uform_py_arena_claim(PyObject *object, PyObject **arena) {
    if (!PyObject_TypeCheck(object, &ArenaType)) {
        PyErr_Format(PyExc_TypeError, "the 'arena' argument must be an Arena, got %S", Py_TYPE(object));
        return -1;
    }
    Arena *const claimed = (Arena *)object;
    Py_ssize_t held = 0;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(object);
#endif
    held = claimed->exports;
    if (!held) {
        claimed->exports = 1;
        ++claimed->generation;
    }
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    if (held)
        return uform_py_raise(uform_status_contended_k,
                              "the arena serves a call or exports a buffer over its outputs; release it first");
    *arena = Py_NewRef(object);
    return 0;
}

void uform_py_arena_unclaim(PyObject *arena) {
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(arena);
#endif
    --((Arena *)arena)->exports;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    Py_DECREF(arena);
}

/** The rank of every output in an arena: rows by columns. */
enum { arena_buffer_max_rank_k = 2 };

/** One output a call left in an arena, exported zero-copy, and the base of every result
 *  that is one. */
typedef struct ArenaBuffer {
    PyObject ob_base;

    /** The arena the output lives in, kept alive so the memory outlives every other name for it. */
    PyObject *arena;

    /** The arena's generation when the call wrote the output; any later one means a
     *  call rewound it. */
    size_t generation;

    void const *data;
    char const *format;
    Py_ssize_t item_bytes;
    int rank;
    Py_ssize_t shape[arena_buffer_max_rank_k];
    Py_ssize_t strides[arena_buffer_max_rank_k];
} ArenaBuffer;

/** A ragged run of rows in an arena, and the offsets around its items. */
typedef struct RowsTape {
    ArenaBuffer rows;

    /** The @c (count + 1,) @c ArenaBuffer of offsets into the rows. */
    PyObject *offsets;
} RowsTape;

/** Where the outputs of @p buffer live, as the engine reports its arena; 0, or -1 with
 *  an error set. */
static int arena_buffer_describe(ArenaBuffer const *buffer, uform_arena_description_t *description) {
    uform_outcome_t outcome = {0};
    uform_arena_describe(&(uform_arena_describe_t) {.arena = ((Arena *)buffer->arena)->handle,
                                                    .outcome = &outcome,
                                                    .description = description});
    return uform_py_raise_on_outcome(&outcome);
}

/** Raises @c InvalidArgumentError unless the call that wrote @p buffer is still the last to claim
 *  its arena; returns 0 or -1. */
static int arena_buffer_fresh(ArenaBuffer const *buffer, size_t generation) {
    if (generation == buffer->generation) return 0;
    return uform_py_raise(uform_status_invalid_argument_k, "a later call rewound the arena this output lived in");
}

/** Holds the arena of @p buffer against a rewind as one more export; 0, or -1 with an
 *  @c InvalidArgumentError set when a later call rewound it. */
static int arena_buffer_hold(ArenaBuffer const *buffer) {
    Arena *const arena = (Arena *)buffer->arena;
    size_t generation = 0;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(arena);
#endif
    generation = arena->generation;
    if (generation == buffer->generation) ++arena->exports;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    return arena_buffer_fresh(buffer, generation);
}

/** Lets go of one export @c arena_buffer_hold took on the arena of @p buffer. */
static void arena_buffer_unhold(ArenaBuffer const *buffer) {
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(buffer->arena);
#endif
    --((Arena *)buffer->arena)->exports;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
}

/**
 *  @brief Holds the arena of @p buffer against a rewind while the host reads it, as one export.
 *  @return 0 on success, -1 with a @c DeviceError set when the output is on a device, or an
 *      @c InvalidArgumentError when a later call rewound it.
 */
static int arena_buffer_pin(ArenaBuffer const *buffer) {
    uform_arena_description_t description = {0};
    if (arena_buffer_describe(buffer, &description) != 0) return -1;
    if (description.residency == uform_residency_vram_k)
        return uform_py_raise(uform_status_device_k, "the output lives in device memory; read `__dlpack__`");
    return arena_buffer_hold(buffer);
}

static int ArenaBuffer_getbuffer(ArenaBuffer *self, Py_buffer *view, int flags) {
    view->obj = NULL;
    if (arena_buffer_pin(self) != 0) return -1;
    Py_ssize_t bytes = self->item_bytes;
    for (int axis = 0; axis != self->rank; ++axis) bytes *= self->shape[axis];
    view->obj = Py_NewRef((PyObject *)self);
    view->buf = (void *)self->data;
    view->len = bytes;
    view->readonly = 0;
    view->itemsize = self->item_bytes; // describes the cells the shape counts, even when no format was asked for
    view->format = (flags & PyBUF_FORMAT) == PyBUF_FORMAT ? (char *)self->format : NULL;
    view->ndim = self->rank;
    view->shape = (flags & PyBUF_ND) == PyBUF_ND ? self->shape : NULL;
    view->strides = (flags & PyBUF_STRIDES) == PyBUF_STRIDES ? self->strides : NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    return 0;
}

static void ArenaBuffer_releasebuffer(ArenaBuffer *self, Py_buffer *view) { arena_buffer_unhold(self); }

/** A tuple of the first @p rank of @p extents; a new reference, or NULL. */
static PyObject *extents_to_tuple(Py_ssize_t const *extents, int rank) {
    PyObject *const tuple = PyTuple_New(rank);
    for (int axis = 0; tuple && axis != rank; ++axis) {
        PyObject *const extent = PyLong_FromSsize_t(extents[axis]);
        if (!extent) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, axis, extent);
    }
    return tuple;
}

static PyObject *ArenaBuffer_get_cuda_array_interface(ArenaBuffer *self, void *closure) {
    uform_arena_description_t description = {0};
    if (arena_buffer_describe(self, &description) != 0) return NULL;
    if (description.residency != uform_residency_vram_k) {
        PyErr_SetString(PyExc_AttributeError, "the output lives in host memory; read it through the buffer protocol");
        return NULL;
    }
    size_t generation = 0;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(self->arena);
#endif
    generation = ((Arena *)self->arena)->generation;
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    if (arena_buffer_fresh(self, generation) != 0) return NULL;
    char const kind = *self->format == 'b' ? 'i' : strchr("efd", *self->format) ? 'f' : 'u';
    char const typestr[3] = {self->item_bytes == 1 ? '|' : '<', kind, (char)('0' + self->item_bytes)};
    PyObject *const address = PyLong_FromVoidPtr((void *)self->data);
    PyObject *const data = address ? PyTuple_Pack(2, address, Py_False) : NULL;
    Py_XDECREF(address);
    PyObject *const interface = data ? PyDict_New() : NULL;
    if (!interface) {
        Py_XDECREF(data);
        return NULL;
    }
    if (uform_py_dict_steal_item(interface, "data", data) != 0 ||
        uform_py_dict_steal_item(interface, "shape", extents_to_tuple(self->shape, self->rank)) != 0 ||
        uform_py_dict_steal_item(interface, "strides", extents_to_tuple(self->strides, self->rank)) != 0 ||
        uform_py_dict_steal_item(interface, "typestr", PyUnicode_FromStringAndSize(typestr, 3)) != 0 ||
        uform_py_dict_steal_item(interface, "version", PyLong_FromLong(3)) != 0) {
        Py_DECREF(interface);
        return NULL;
    }
    return interface;
}

/** The DLPack device the outputs of @p buffer live on, from the engine's description of its arena;
 *  0, or -1 with an exception set. */
static int arena_buffer_device(ArenaBuffer const *buffer, DLDevice *device) {
    uform_arena_description_t description = {0};
    if (arena_buffer_describe(buffer, &description) != 0) return -1;
    *device = (DLDevice) {kDLCPU, 0};
    if (description.residency != uform_residency_vram_k) return 0;
    size_t const capacity = (size_t)description.device + 1;
    size_t count = 0;
    uform_gpu_device_t *const listed = (uform_gpu_device_t *)PyMem_Calloc(capacity, sizeof(uform_gpu_device_t));
    if (!listed) return PyErr_NoMemory(), -1;
    uform_outcome_t outcome = {0};
    uform_gpu_devices(&(uform_gpu_devices_t) {.outcome = &outcome,
                                              .devices = listed,
                                              .devices_capacity = capacity,
                                              .devices_count = &count});
    int failed = uform_py_raise_on_outcome(&outcome);
    if (!failed && count < capacity) failed = uform_py_raise(uform_status_device_k, "the arena's GPU is not listed");
    uform_gpu_device_t const *const gpu = &listed[description.device];
    if (!failed)
        *device = (DLDevice) {gpu->vendor == uform_gpu_vendor_amd_k     ? kDLROCM
                              : gpu->vendor == uform_gpu_vendor_apple_k ? kDLMetal
                                                                        : kDLCUDA,
                              (int32_t)gpu->vendor_ordinal};
    PyMem_Free(listed);
    return failed ? -1 : 0;
}

/** What a DLPack capsule over an @c ArenaBuffer owns: the managed tensor, its extents and strides,
 *  and the export it holds on the buffer's arena until the consumer lets go. */
typedef struct dlpack_export_t {
    union {
        DLManagedTensor legacy;
        DLManagedTensorVersioned versioned;
    } managed;
    ArenaBuffer *buffer;
    int64_t shape[arena_buffer_max_rank_k];
    int64_t strides[arena_buffer_max_rank_k];
} dlpack_export_t;

/** Releases an export from whichever thread its consumer drops it on, taking the GIL itself. */
static void dlpack_export_release(dlpack_export_t *exported) {
    PyGILState_STATE const state = PyGILState_Ensure();
    arena_buffer_unhold(exported->buffer);
    Py_DECREF(exported->buffer);
    PyMem_Free(exported);
    PyGILState_Release(state);
}

static void dlpack_legacy_deleter(DLManagedTensor *self) { dlpack_export_release(self->manager_ctx); }

static void dlpack_versioned_deleter(DLManagedTensorVersioned *self) { dlpack_export_release(self->manager_ctx); }

/** Releases an export no consumer took; a consumer renames the capsule and owns it. */
static void dlpack_export_capsule_release(PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        DLManagedTensor *const managed = PyCapsule_GetPointer(capsule, "dltensor");
        managed->deleter(managed);
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        DLManagedTensorVersioned *const managed = PyCapsule_GetPointer(capsule, "dltensor_versioned");
        managed->deleter(managed);
    }
}

static PyObject *ArenaBuffer_dlpack(ArenaBuffer *self, PyObject *const *args, Py_ssize_t positional, PyObject *names) {
    // The call that wrote the output is final, so `stream` names no work left to order after.
    PyObject *stream = Py_None, *max_version = Py_None, *dl_device = Py_None, *copy = Py_None;
    if (positional) return PyErr_Format(PyExc_TypeError, "__dlpack__() takes keyword arguments only");
    for (Py_ssize_t keyword = 0; names && keyword != PyTuple_GET_SIZE(names); ++keyword) {
        PyObject *const name = PyTuple_GET_ITEM(names, keyword);
        PyObject **const slot = PyUnicode_CompareWithASCIIString(name, "stream") == 0        ? &stream
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

    dlpack_export_t *const exported = (dlpack_export_t *)PyMem_Calloc(1, sizeof(dlpack_export_t));
    if (!exported) return PyErr_NoMemory();
    if (arena_buffer_hold(self) != 0) {
        PyMem_Free(exported);
        return NULL;
    }
    exported->buffer = (ArenaBuffer *)Py_NewRef((PyObject *)self);
    for (int axis = 0; axis != self->rank; ++axis) {
        exported->shape[axis] = self->shape[axis];
        exported->strides[axis] = self->strides[axis] / self->item_bytes;
    }
    DLTensor const tensor = {(void *)self->data, device, self->rank, dtype, exported->shape, exported->strides, 0};
    void *managed = NULL;
    if (major >= 1) {
        exported->managed.versioned =
            (DLManagedTensorVersioned) {{1, 0}, exported, dlpack_versioned_deleter, 0, tensor};
        managed = &exported->managed.versioned;
    }
    else {
        exported->managed.legacy = (DLManagedTensor) {tensor, exported, dlpack_legacy_deleter};
        managed = &exported->managed.legacy;
    }
    PyObject *const capsule = PyCapsule_New(managed, major >= 1 ? "dltensor_versioned" : "dltensor",
                                            dlpack_export_capsule_release);
    if (!capsule) dlpack_export_release(exported);
    return capsule;
}

static PyObject *ArenaBuffer_dlpack_device(ArenaBuffer *self, PyObject *unused) {
    DLDevice device;
    if (arena_buffer_device(self, &device) != 0) return NULL;
    PyObject *const pair = PyTuple_New(2);
    if (!pair) return NULL;
    PyTuple_SET_ITEM(pair, 0, PyLong_FromLong((long)device.device_type));
    PyTuple_SET_ITEM(pair, 1, PyLong_FromLong((long)device.device_id));
    if (PyTuple_GET_ITEM(pair, 0) && PyTuple_GET_ITEM(pair, 1)) return pair;
    Py_DECREF(pair);
    return NULL;
}

static char const doc_ArenaBuffer_dlpack[] =                                                 //
    "__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None) -> PyCapsule\n" //
    "\n"                                                                                     //
    "Exports the output in place as a DLPack capsule, on host pages or the GPU its arena\n"  //
    "lives on; the arena refuses to rewind until the consumer lets go. A `max_version` of\n" //
    "1 or more asks for a versioned capsule.";

static char const doc_ArenaBuffer_dlpack_device[] = //
    "__dlpack_device__() -> tuple[int, int]\n"      //
    "\n"                                            //
    "The DLPack device type and ID the output lives on.";

static PyMethodDef ArenaBuffer_methods[] = {
    {"__dlpack__", (PyCFunction)(void (*)(void))ArenaBuffer_dlpack, METH_FASTCALL | METH_KEYWORDS,
     doc_ArenaBuffer_dlpack},
    {"__dlpack_device__", (PyCFunction)ArenaBuffer_dlpack_device, METH_NOARGS, doc_ArenaBuffer_dlpack_device},
    {NULL, NULL, 0, NULL},
};

static void ArenaBuffer_dealloc(ArenaBuffer *self) {
    Py_DECREF(self->arena);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static Py_ssize_t ArenaBuffer_length(ArenaBuffer *self) { return self->shape[0]; }

static PySequenceMethods ArenaBuffer_as_sequence = {.sq_length = (lenfunc)ArenaBuffer_length};

static PyBufferProcs ArenaBuffer_as_buffer = {
    .bf_getbuffer = (getbufferproc)ArenaBuffer_getbuffer,
    .bf_releasebuffer = (releasebufferproc)ArenaBuffer_releasebuffer,
};

static PyGetSetDef ArenaBuffer_getset[] = {
    {"__cuda_array_interface__", (getter)ArenaBuffer_get_cuda_array_interface, NULL,
     "Version 3 of the CUDA Array Interface, present only over device memory. Python-only.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static char const doc_ArenaBuffer[] =                                                        //
    "ArenaBuffer\n"                                                                          //
    "\n"                                                                                     //
    "One output a call left in an `Arena`, viewing it until it rewinds: a (rows, columns)\n" //
    "matrix for the scalars a buffer can format, and a (rows, stride) run of 'B' for the\n"  //
    "rest, `bf16` among them. Every output exports `__dlpack__`; host outputs also export\n" //
    "the writable buffer protocol, and device ones `__cuda_array_interface__`.";

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

static void RowsTape_dealloc(RowsTape *self) {
    Py_XDECREF(self->offsets);
    ArenaBuffer_dealloc(&self->rows);
}

static PyMemberDef RowsTape_members[] = {
    {"offsets", Py_T_OBJECT_EX, offsetof(RowsTape, offsets), Py_READONLY,
     "(count + 1,) Apache Arrow offsets into the rows, item `i` spanning rows `offsets[i]` up to `offsets[i + 1]`."},
    {NULL, 0, 0, 0, NULL},
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
};

/** The buffer format of the scalars a @c memoryview can format, and the bytes one value takes. */
static struct {
    uform_scalar_t kind;
    char const *format;
    Py_ssize_t bytes;
} const scalar_formats[] = {
    {uform_scalar_f64_k, "d", 8}, {uform_scalar_f32_k, "f", 4}, {uform_scalar_f16_k, "e", 2},
    {uform_scalar_i32_k, "i", 4}, {uform_scalar_i8_k, "b", 1},  {uform_scalar_u8_k, "B", 1},
};

/** A new @c ArenaBuffer of @p type over @p data inside the claimed @p arena, C-contiguous;
 *  NULL on failure. */
static ArenaBuffer *arena_buffer_new(PyTypeObject *type, PyObject *arena, void const *data, char const *format,
                                     Py_ssize_t item_bytes, Py_ssize_t const *shape, int rank) {
    static char const empty_output = 0; // stands in for a null output, which no exporter may hand out
    ArenaBuffer *const exporter = (ArenaBuffer *)type->tp_alloc(type, 0);
    if (!exporter) return NULL;
    exporter->arena = Py_NewRef(arena);
    exporter->generation = ((Arena *)arena)->generation; // built while the writing call still holds the arena
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

PyObject *uform_py_rows_answer(PyTypeObject *type, PyObject *arena, uform_rows_view_t const *rows, size_t count) {
    char const *format = "B";
    Py_ssize_t item_bytes = 1;
    Py_ssize_t shape[2] = {(Py_ssize_t)count, (Py_ssize_t)rows->stride};
    for (size_t row = 0; row != sizeof(scalar_formats) / sizeof(*scalar_formats); ++row)
        if (scalar_formats[row].kind == rows->scalar &&
            rows->stride == rows->columns * (size_t)scalar_formats[row].bytes) {
            format = scalar_formats[row].format;
            item_bytes = scalar_formats[row].bytes;
            shape[1] = (Py_ssize_t)rows->columns;
        }
    return (PyObject *)arena_buffer_new(type, arena, rows->data, format, item_bytes, shape, 2);
}

PyObject *uform_py_rows_tape_answer(PyObject *arena, uform_rows_tape_view_t const *states, size_t count,
                                    size_t tokens) {
    Py_ssize_t const offsets_shape[1] = {(Py_ssize_t)count + 1};
    RowsTape *rows_tape = (RowsTape *)uform_py_rows_answer(&RowsTapeType, arena, &states->rows, tokens);
    if (!rows_tape) return NULL;
    rows_tape->offsets = (PyObject *)arena_buffer_new(&ArenaBufferType, arena, states->offsets, "N", sizeof(size_t),
                                                      offsets_shape, 1);
    if (!rows_tape->offsets) Py_CLEAR(rows_tape);
    return (PyObject *)rows_tape;
}

PyObject *uform_py_tape_answer(PyObject *arena, uform_tape_view_t const *tape, size_t count, size_t elements) {
    Py_ssize_t const shape[1] = {(Py_ssize_t)elements}, offsets_shape[1] = {(Py_ssize_t)count + 1};
    RowsTape *rows_tape = (RowsTape *)arena_buffer_new(&RowsTapeType, arena, tape->data, "f", sizeof(float), shape, 1);
    if (!rows_tape) return NULL;
    rows_tape->offsets = (PyObject *)arena_buffer_new(&ArenaBufferType, arena, tape->offsets, "N", sizeof(size_t),
                                                      offsets_shape, 1);
    if (!rows_tape->offsets) Py_CLEAR(rows_tape);
    return (PyObject *)rows_tape;
}

#pragma endregion Arenas

#pragma region Tapes

void uform_py_tape_release(uform_py_tape_t *tape) {
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
static char const *element_format(uform_py_element_t element) {
    return element == uform_py_element_samples_k ? "f" : element == uform_py_element_pixels_k ? "B" : NULL;
}

/** The format character of @p buffer, its byte-order prefix skipped; "B" when the
 *  exporter named none. */
static char buffer_format(Py_buffer const *buffer) {
    char const *format = buffer->format ? buffer->format : "B";
    while (*format == '@' || *format == '=' || *format == '<' || *format == '>' || *format == '!') ++format;
    return *format;
}

/**
 *  @brief Acquires the offsets of a @c (data, offsets) pair into @p tape when @p candidate exports
 *      a C-contiguous run of unsigned words as wide as a @c size_t.
 *  @return 1 when it does, 0 when it is no offsets buffer at all, with no exception set.
 */
static int tape_offsets_acquire(PyObject *candidate, uform_py_tape_t *tape) {
    if (buffer_acquire(candidate, 0, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT, &tape->offsets) != 0) {
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

/** Lends the data of a pair whose offsets @p tape already holds; returns 0 or -1 with
 *  an exception set. */
static int tape_lend(PyObject *data, char const *field, uform_py_element_t element, uform_py_tape_t *tape) {
    if (buffer_acquire(data, 1, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT, &tape->data) != 0) return -1;
    char const *const expected = element_format(element);
    if ((size_t)tape->data.itemsize != element_bytes(element) || (expected && buffer_format(&tape->data) != *expected))
        return uform_py_raise(uform_status_unsupported_layout_k, "the '%s' data holds items the verb cannot read",
                              field);
    size_t const *const offsets = (size_t const *)tape->offsets.buf;
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
static int tape_entry_acquire(PyObject *entry, char const *field, Py_ssize_t position, uform_py_element_t element,
                              Py_buffer *view) {
    if (element == uform_py_element_text_k && PyUnicode_Check(entry)) {
        Py_ssize_t length = 0;
        char const *const text = PyUnicode_AsUTF8AndSize(entry, &length); // lives as long as the string the view holds
        return text ? PyBuffer_FillInfo(view, entry, (void *)text, length, 1, PyBUF_SIMPLE) : -1;
    }
    int const flags = element == uform_py_element_pixels_k || element == uform_py_element_samples_k
                          ? PyBUF_C_CONTIGUOUS | PyBUF_FORMAT
                          : PyBUF_C_CONTIGUOUS;
    if (buffer_acquire(entry, 0, flags, view) != 0) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "entry %zd of '%s' must be a %s, got %S", position, field,
                     element == uform_py_element_text_k ? "str or a bytes-like object" : "contiguous buffer",
                     Py_TYPE(entry));
        return -1;
    }
    char const *const expected = element_format(element);
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
static int tape_gather(PyObject *sequence, char const *field, uform_py_element_t element, uform_py_tape_t *tape) {
    Py_ssize_t const count = PySequence_Fast_GET_SIZE(sequence);
    Py_buffer *const views = (Py_buffer *)PyMem_Calloc(count ? (size_t)count : 1, sizeof(Py_buffer));
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
        size_t *const offsets = (size_t *)tape->gathered;
        uint32_t *const extents = extents_count ? (uint32_t *)(offsets + count + 1) : NULL;
        char *const data = (char *)tape->gathered + head;
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

int uform_py_tape_from_object(PyObject *object, char const *field, uform_py_element_t element, uform_py_tape_t *tape) {
    memset(tape, 0, sizeof(*tape));
    if (PyUnicode_Check(object) || PyBytes_Check(object) || PyByteArray_Check(object) || PyMemoryView_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' argument takes a sequence or a (data, offsets) pair, got one %S", field,
                     Py_TYPE(object));
        return -1;
    }
    if (element != uform_py_element_pixels_k && PyTuple_Check(object) && PyTuple_GET_SIZE(object) == 2 &&
        tape_offsets_acquire(PyTuple_GET_ITEM(object, 1), tape))
        return tape_lend(PyTuple_GET_ITEM(object, 0), field, element, tape);
    PyObject *const sequence = PySequence_Fast(object, "expected a sequence");
    if (!sequence) return -1;
    int const gathered = tape_gather(sequence, field, element, tape);
    Py_DECREF(sequence);
    return gathered;
}

void uform_py_words_release(uform_py_words_t *words) {
    PyMem_Free(words->gathered);
    if (words->buffer.obj) PyBuffer_Release(&words->buffer);
    memset(words, 0, sizeof(*words));
}

int uform_py_words_from_object(PyObject *object, char const *field, size_t count, uform_py_words_t *words) {
    if (!object || object == Py_None) return 0;
    if (buffer_acquire(object, 0, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT, &words->buffer) == 0) {
        if (words->buffer.itemsize != sizeof(uint32_t) || buffer_format(&words->buffer) != 'I')
            return uform_py_raise(uform_status_unsupported_layout_k, "the '%s' buffer must hold uint32 words", field);
        if ((size_t)words->buffer.len != count * sizeof(uint32_t))
            return uform_py_raise(uform_status_shape_mismatch_k, "the '%s' buffer must hold %zu words", field, count);
        words->data = (uint32_t const *)words->buffer.buf;
        return 0;
    }
    PyErr_Clear();
    PyObject *const sequence = PySequence_Fast(object, "expected a buffer of uint32 or a sequence of integers");
    if (!sequence) return -1;
    if ((size_t)PySequence_Fast_GET_SIZE(sequence) != count) {
        Py_DECREF(sequence);
        return uform_py_raise(uform_status_shape_mismatch_k, "the '%s' option must hold %zu entries", field, count);
    }
    words->gathered = (uint32_t *)PyMem_Malloc((count ? count : 1) * sizeof(uint32_t));
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

PyObject *api_version(PyObject *self, PyObject *unused) { return PyUnicode_FromString(uform_version_name()); }

PyObject *api_capabilities_compiled(PyObject *self, PyObject *unused) {
    return PyUnicode_FromString(uform_capabilities_compiled());
}

PyObject *api_capabilities_available(PyObject *self, PyObject *unused) {
    return PyUnicode_FromString(uform_capabilities_available());
}

static PyStructSequence_Field GPUDevice_fields[] = {
    {"vendor", "The `GPUVendor` whose runtime drives the device."},
    {"vendor_ordinal", "The device's ordinal in its vendor's runtime, as nvidia-smi or rocm-smi list it."},
    {"gpu_class", "The `GPUClass` whose kernels run on it, or None."},
    {"compute_units", "Streaming multiprocessors or compute units."},
    {"memory_bytes", "Device memory in bytes."},
    {"name", "The vendor's own name for the device."},
    {NULL, NULL},
};

PyStructSequence_Desc uform_py_gpu_device_desc = {
    "uform.GPUDevice", "One GPU this process can drive, its position in `gpu_devices()` being its ordinal.",
    GPUDevice_fields, 6};

PyTypeObject GPUDeviceType;

char const doc_gpu_devices[] =                                                                   //
    "gpu_devices() -> tuple[GPUDevice, ...]\n"                                                   //
    "\n"                                                                                         //
    "Every GPU this process can drive, NVIDIA devices first, then AMD, so a device's position\n" //
    "is the ordinal `Executor.for_vram` and `Model.load_vram` name it by. A vendor this build\n" //
    "does not compile, or whose runtime is absent, lists none.\n"                                //
    "Calls `uform_gpu_devices`.";

/** Lists up to @p capacity devices into @p devices; returns how many there are, or -1. */
static Py_ssize_t gpu_devices_list(uform_gpu_device_t *devices, size_t capacity) {
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

PyObject *api_gpu_devices(PyObject *self, PyObject *unused) {
    Py_ssize_t const counted = gpu_devices_list(NULL, 0);
    if (counted < 0) return NULL;
    uform_gpu_device_t *const devices = (uform_gpu_device_t *)PyMem_Calloc(counted ? (size_t)counted : 1,
                                                                           sizeof(uform_gpu_device_t));
    if (!devices) return PyErr_NoMemory();
    Py_ssize_t const listed = gpu_devices_list(devices, (size_t)counted);
    PyObject *tuple = listed < 0 ? NULL : PyTuple_New(listed < counted ? listed : counted);
    for (Py_ssize_t index = 0; tuple && index != PyTuple_GET_SIZE(tuple); ++index) {
        uform_gpu_device_t const *const device = &devices[index];
        PyObject *fields[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
        (void)((fields[0] = uform_py_enum_to_object(&uform_py_gpu_vendor, (int)device->vendor)) &&
               (fields[1] = PyLong_FromUnsignedLong(device->vendor_ordinal)) &&
               (fields[2] = uform_py_class_to_object(&uform_py_gpu_class, (int)device->gpu_class)) &&
               (fields[3] = PyLong_FromUnsignedLong(device->compute_units)) &&
               (fields[4] = PyLong_FromUnsignedLongLong(device->memory_bytes)) &&
               (fields[5] = PyUnicode_FromString(device->name)));
        PyObject *const entry = uform_py_struct_sequence_steal(&GPUDeviceType, fields, 6);
        if (!entry) Py_CLEAR(tuple);
        else PyTuple_SET_ITEM(tuple, index, entry);
    }
    PyMem_Free(devices);
    return tuple;
}

#pragma endregion Library Probes
