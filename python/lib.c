/**
 *  @file python/lib.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The module-init glue - `PyModuleDef`, method merging, type registration, and `PyInit_lib`.
 *
 *  Each header's file contributes its verbs as one table, and this file merges them into the methods of the
 *  model, the trainer and both receivers, so a verb is written once and every receiver offers it.
 */
#include "lib.h"

static char const doc_module[] =                                                                    //
    "Pocket-Sized Multimodal AI: CPU + GPU inference and GPU training for content understanding.\n" //
    "\n"                                                                                            //
    "Models, executors and trainers sitting directly on the stable C ABI. Every verb is a\n"        //
    "batch over buffer-protocol inputs, every output lands in an arena the caller reuses,\n"        //
    "and NumPy is never required.";

/** One type the module readies, the name it exports it under, and the layout of a struct sequence. */
static struct {
    char const *name;
    PyTypeObject *type;

    /** NULL for every type but a struct sequence. */
    PyStructSequence_Desc *layout;
} const module_types[] = {
    {"Executor", &ExecutorType, NULL},
    {"Arena", &ArenaType, NULL},
    {"ArenaBuffer", &ArenaBufferType, NULL},
    {"RowsTape", &RowsTapeType, NULL},
    {"Deferred", &DeferredType, NULL},
    {"On", &OnType, NULL},
    {"DeferredOn", &DeferredOnType, NULL},
    {"GPUDevice", &GPUDeviceType, &uform_py_gpu_device_desc},

    {"Model", &ModelType, NULL},
    {"ModelDescription", &ModelDescriptionType, &uform_py_model_description_desc},

    {"Encoded", &EncodedType, &uform_py_encoded_desc},
    {"TextEncodeStats", &TextEncodeStatsType, NULL},
    {"ImageEncodeStats", &ImageEncodeStatsType, NULL},
    {"VideoEncodeStats", &VideoEncodeStatsType, NULL},
    {"AudioEncodeStats", &AudioEncodeStatsType, NULL},

    {"Mesh", &MeshType, NULL},
    {"MeshDescription", &MeshDescriptionType, &uform_py_mesh_description_desc},
    {"Trainer", &TrainerType, NULL},
    {"TrainerStepStats", &TrainerStepStatsType, NULL},
};

/** The keyword-built structures, readied from their field tables; nested ones before the ones holding them. */
static uform_py_fields_t *const module_fields[] = {
    &ImageShapeType,      &VideoShapeType, &AudioShapeType, &BatchShapeType,  &MaskingType, &ContrastiveType,
    &LateInteractionType, &ChoicesType,    &ScheduleType,   &MeshOptionsType, &EdgesType,   &RelationsType,
};

/** Rows of the merged tables: a receiver offers ten verbs, a model fourteen, a trainer five. */
enum { merged_methods_max_k = 16 };

static PyMethodDef model_methods[merged_methods_max_k], trainer_methods[merged_methods_max_k],
    receiver_methods[merged_methods_max_k];

static PyMethodDef const receiver_class_getitem[] = {
    {"__class_getitem__", Py_GenericAlias, METH_O | METH_CLASS,
     "Names the target the receiver addresses, as `On[Model]` or `DeferredOn[Trainer]`."},
    {NULL, NULL, 0, NULL},
};

/** Copies the rows of the @p count NULL-terminated @p tables into @p merged, ending it with a sentinel. */
static void methods_merge(PyMethodDef *merged, PyMethodDef const *const *tables, size_t count) {
    size_t written = 0;
    for (size_t table = 0; table != count; ++table)
        for (PyMethodDef const *row = tables[table]; row->ml_name; ++row) merged[written++] = *row;
    merged[written] = (PyMethodDef) {NULL, NULL, 0, NULL};
}

static PyMethodDef module_methods[] = {
    {"version", (PyCFunction)api_version, METH_NOARGS, doc_version},
    {"capabilities_compiled", (PyCFunction)api_capabilities_compiled, METH_NOARGS, doc_capabilities_compiled},
    {"capabilities_available", (PyCFunction)api_capabilities_available, METH_NOARGS, doc_capabilities_available},
    {"gpu_devices", (PyCFunction)api_gpu_devices, METH_NOARGS, doc_gpu_devices},
    {NULL, NULL, 0, NULL},
};

/** Named with the dotted name the extension is built as, where the import machinery finds it, while the types keep
 *  the undotted `uform.` prefix in their `tp_name`, where the package re-exports them. */
static PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, .m_name = "uform.lib", .m_doc = doc_module, .m_size = -1, .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit_lib(void) {
    PyMethodDef const *const model_tables[] = {uform_py_model_methods, uform_py_model_verbs, uform_py_encode_verbs};
    PyMethodDef const *const trainer_tables[] = {uform_py_trainer_methods, uform_py_trainer_verbs};
    PyMethodDef const *const receiver_tables[] = {uform_py_model_verbs, uform_py_encode_verbs, uform_py_trainer_verbs,
                                                  receiver_class_getitem};
    methods_merge(model_methods, model_tables, sizeof(model_tables) / sizeof(*model_tables));
    methods_merge(trainer_methods, trainer_tables, sizeof(trainer_tables) / sizeof(*trainer_tables));
    methods_merge(receiver_methods, receiver_tables, sizeof(receiver_tables) / sizeof(*receiver_tables));
    ModelType.tp_methods = model_methods;
    TrainerType.tp_methods = trainer_methods;
    OnType.tp_methods = receiver_methods;
    DeferredOnType.tp_methods = receiver_methods;

    size_t const types_count = sizeof(module_types) / sizeof(*module_types);
    for (size_t type_index = 0; type_index != types_count; ++type_index) {
        PyTypeObject *const type = module_types[type_index].type;
        PyStructSequence_Desc *const layout = module_types[type_index].layout;
        if ((layout ? PyStructSequence_InitType2(type, layout) : PyType_Ready(type)) < 0) return NULL;
    }
    size_t const fields_count = sizeof(module_fields) / sizeof(*module_fields);
    for (size_t fields_index = 0; fields_index != fields_count; ++fields_index)
        if (uform_py_fields_ready(module_fields[fields_index]) < 0) return NULL;

    PyObject *const module = PyModule_Create(&module_definition);
    if (!module) return NULL;

#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED); // every call detaches its thread state around the library
#endif

    if (uform_py_register_error_types(module) < 0 || uform_py_register_enums(module) < 0 ||
        PyModule_AddStringConstant(module, "__version__", uform_version_name()) < 0)
        goto failed;
    for (size_t type_index = 0; type_index != types_count; ++type_index)
        if (PyModule_AddObjectRef(module, module_types[type_index].name, (PyObject *)module_types[type_index].type) < 0)
            goto failed;
    for (size_t fields_index = 0; fields_index != fields_count; ++fields_index) {
        PyTypeObject *const type = &module_fields[fields_index]->type;
        if (PyModule_AddObjectRef(module, type->tp_name + sizeof("uform.") - 1, (PyObject *)type) < 0) goto failed;
    }
    return module;

failed:
    Py_DECREF(module);
    return NULL;
}
