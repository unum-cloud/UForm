/**
 *  @file python/model.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief CPython model lifecycle: load an encoder into RAM or VRAM, describe it, and free it.
 *
 *  Residency is in the verb name, as in the C interface: @c load_vram places the weights on
 *  the devices it names, and @c load_ram in host memory. A load is a constructor, so it
 *  blocks, on the executor it is given or on the calling thread; @c describe is a verb,
 *  reached through every receiver.
 */
#include "lib.h"

#pragma region Construction

static char const doc_Model[] =                                                                    //
    "Model\n"                                                                                      //
    "\n"                                                                                           //
    "A loaded encoder: its weights, their packed copies, and on a GPU the session its attention\n" //
    "runs on. One model serves every modality it was trained on, and only infers.\n"               //
    "\n"                                                                                           //
    "Load one into the residency its weights live in - `load_ram` or `load_vram` - from a\n"       //
    "safetensors path, or from a buffer holding one with `load_ram_buffer` or\n"                   //
    "`load_vram_buffer`.\n"                                                                        //
    "\n"                                                                                           //
    "The receiver chooses where a verb runs, and the verb stays the same: the model runs it\n"     //
    "on the calling thread, `on(executor)` blocks on that executor, and `deferred(executor)`\n"    //
    "submits it there and answers a `Deferred`. See `uform.Executor`.\n"                           //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "  >>> model = uform.Model.load_ram_buffer(b'any bytes the stub accepts')\n"                   //
    "  >>> model.describe().dimensions\n"                                                          //
    "  64";

static PyObject *Model_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    PyErr_SetString(PyExc_TypeError,
                    "Model is built through load_ram(), load_ram_buffer(), load_vram() or load_vram_buffer()");
    return NULL;
}

static void Model_dealloc(Model *self) {
    Py_BEGIN_ALLOW_THREADS;
    uform_model_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/** Wraps the handle a load produced, a progress handler's own exception outranking the outcome,
 *  or frees it. */
static PyObject *model_loaded(PyTypeObject *type, uform_model_t handle, uform_py_call_t *call) {
    if (call->raised || call->outcome.status != uform_status_success_k) {
        uform_model_free(handle);
        if (!call->raised) return uform_py_raise_on_outcome(&call->outcome), NULL;
        PyErr_SetRaisedException(call->raised);
        call->raised = NULL;
        return NULL;
    }
    Model *const model = (Model *)type->tp_alloc(type, 0);
    if (!model) {
        uform_model_free(handle);
        return NULL;
    }
    model->handle = handle;
    return (PyObject *)model;
}

/** What every load takes beside its source and its residency's placement. */
typedef struct load_options_t {
    uform_executor_t executor;
    uform_scalar_t quantization;
    uform_py_call_t call;
    uform_py_callback_t progress;
    size_t progress_steps;
} load_options_t;

/**
 *  @brief Reads what every load takes: the source as @p source_name, the placement as
 *      @p arm_name, the pin of @p classes as @p pin_name, then `executor`, `quantization`,
 *      `progress` and `progress_steps` by name.
 *  @return 0 on success, -1 with an exception set otherwise; release
 *      @c options->progress either way.
 */
static int load_parse(char const *verb, PyObject *const *args, Py_ssize_t positional_args_count,
                      PyObject *args_names_tuple, char const *source_name, PyObject **source, char const *arm_name,
                      PyObject **arm, char const *pin_name, uform_py_enum_t const *classes, int *pin,
                      load_options_t *options) {
    PyObject *pin_object, *executor_object, *quantization_object, *progress_object, *progress_steps_object;
    uform_py_argument_t const arguments[] = {
        {source_name, source},
        {arm_name, arm},
        {pin_name, &pin_object},
        {"executor", &executor_object},
        {"quantization", &quantization_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
    };
    memset(options, 0, sizeof(*options));
    options->quantization = uform_scalar_bf16_k;
    if (uform_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_class_from_object(pin_object, pin_name, classes, pin) != 0 ||
        uform_py_executor_optional(executor_object, &options->executor) != 0 ||
        uform_py_scalar_from_object(quantization_object, "quantization", &options->quantization) != 0 ||
        uform_py_size_from_object(progress_steps_object, "progress_steps", &options->progress_steps) != 0)
        return -1;
    return uform_py_callback_from_object(progress_object, "progress", &options->call, &options->progress);
}

static char const doc_Model_load_ram[] =                                                           //
    "load_ram(path, /, *, memory_domains=None, cpu_class=None, executor=None, "                    //
    "quantization=Scalar.BF16, progress=None, progress_steps=0) -> Model\n"                        //
    "\n"                                                                                           //
    "Copies a safetensors checkpoint into host memory and packs its products for the CPU; the\n"   //
    "checkpoint carries the model's geometry.\n"                                                   //
    "Calls `uform_model_load_ram`.\n"                                                              //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  path (str | bytes | os.PathLike): The checkpoint, read in full and closed.\n"               //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place the weights in; None\n"    //
    "    derives every node.\n"                                                                    //
    "  cpu_class (CPUClass, optional): The class to pack for and dispatch as; None derives the\n"  //
    "    CPU's own.\n"                                                                             //
    "  executor (Executor, optional): The executor the load runs on; None loads on the\n"          //
    "    calling thread.\n"                                                                        //
    "  quantization (Scalar, optional): The scalar the transformer blocks run in, `Scalar.BF16`\n" //
    "    as stored or `Scalar.E4M3` quantized at load.\n"                                          //
    "  progress (callable, optional): Called with (completed, total) tensors placed; returning\n"  //
    "    False stops the load and raises `CancelledError`.\n"                                      //
    "  progress_steps (int, optional): Reports the load makes at most; 0 derives to ten.";

static PyObject *Model_load_ram(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    PyObject *path_object, *memory_domains_object, *decoded = NULL, *result = NULL;
    load_options_t options;
    uform_bytes_view_t path;
    int32_t *memory_domains = NULL;
    size_t memory_domains_count = 0;
    int cpu_class = 0;
    if (load_parse("load_ram", args, positional_args_count, args_names_tuple, "path", &path_object, "memory_domains",
                   &memory_domains_object, "cpu_class", &uform_py_cpu_class, &cpu_class, &options) != 0 ||
        uform_py_path_from_object(path_object, "path", &path, &decoded) != 0 ||
        uform_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                      &memory_domains_count) != 0)
        goto release;

    uform_model_t handle = NULL;
    uform_model_load_ram_t arguments = {
        .model = &handle,
        .executor = options.executor,
        .outcome = &options.call.outcome,
        .path = path,
        .ram = {memory_domains, memory_domains_count, (uform_cpu_class_t)cpu_class},
        .quantization = options.quantization,
        .progress = uform_py_progress_registered(&options.progress, options.progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    uform_model_load_ram(&arguments);
    Py_END_ALLOW_THREADS;
    result = model_loaded(type, handle, &options.call);

release:
    PyMem_Free(memory_domains);
    Py_XDECREF(decoded);
    Py_XDECREF(options.progress.callable);
    return result;
}

static char const doc_Model_load_ram_buffer[] =                                                  //
    "load_ram_buffer(buffer, /, *, memory_domains=None, cpu_class=None, executor=None, "         //
    "quantization=Scalar.BF16, progress=None, progress_steps=0) -> Model\n"                      //
    "\n"                                                                                         //
    "Copies a safetensors checkpoint held in memory into host memory, as `load_ram` does from\n" //
    "a path; nothing of the buffer is kept.\n"                                                   //
    "Calls `uform_model_load_ram_buffer`.\n"                                                     //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  buffer (buffer): The checkpoint itself, any contiguous bytes-like object.\n"              //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place the weights in.\n"       //
    "  cpu_class (CPUClass, optional): The class to pack for and dispatch as.\n"                 //
    "  executor (Executor, optional): The executor the load runs on.\n"                          //
    "  quantization (Scalar, optional): The scalar the transformer blocks run in.\n"             //
    "  progress (callable, optional): Called with (completed, total) tensors placed.\n"          //
    "  progress_steps (int, optional): Reports the load makes at most; 0 derives to ten.";

static PyObject *Model_load_ram_buffer(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    PyObject *buffer_object, *memory_domains_object, *result = NULL;
    load_options_t options;
    Py_buffer buffer = {0};
    int32_t *memory_domains = NULL;
    size_t memory_domains_count = 0;
    int cpu_class = 0;
    if (load_parse("load_ram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "memory_domains", &memory_domains_object, "cpu_class", &uform_py_cpu_class, &cpu_class,
                   &options) != 0 ||
        PyObject_GetBuffer(buffer_object, &buffer, PyBUF_C_CONTIGUOUS) != 0 ||
        uform_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                      &memory_domains_count) != 0)
        goto release;

    uform_model_t handle = NULL;
    uform_model_load_ram_buffer_t arguments = {
        .model = &handle,
        .executor = options.executor,
        .outcome = &options.call.outcome,
        .buffer = {(uint8_t const *)buffer.buf, (size_t)buffer.len},
        .ram = {memory_domains, memory_domains_count, (uform_cpu_class_t)cpu_class},
        .quantization = options.quantization,
        .progress = uform_py_progress_registered(&options.progress, options.progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    uform_model_load_ram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = model_loaded(type, handle, &options.call);

release:
    PyMem_Free(memory_domains);
    if (buffer.obj) PyBuffer_Release(&buffer);
    Py_XDECREF(options.progress.callable);
    return result;
}

static char const doc_Model_load_vram[] =                                                      //
    "load_vram(path, /, *, devices=None, gpu_class=None, executor=None, "                      //
    "quantization=Scalar.BF16, progress=None, progress_steps=0) -> Model\n"                    //
    "\n"                                                                                       //
    "Copies a safetensors checkpoint onto the named GPUs, replicated on each, and packs its\n" //
    "products there.\n"                                                                        //
    "Calls `uform_model_load_vram`.\n"                                                         //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  path (str | bytes | os.PathLike): The checkpoint, read in full and closed.\n"           //
    "  devices (Sequence[int], optional): Positions in `gpu_devices()` to place the weights\n" //
    "    on; None derives one free device.\n"                                                  //
    "  gpu_class (GPUClass, optional): The class every device dispatches as; None derives\n"   //
    "    the devices' own, which must agree.\n"                                                //
    "  executor (Executor, optional): The executor the upload runs on, one stream per\n"       //
    "    device.\n"                                                                            //
    "  quantization (Scalar, optional): The scalar the transformer blocks run in.\n"           //
    "  progress (callable, optional): Called with (completed, total) tensors placed.\n"        //
    "  progress_steps (int, optional): Reports the load makes at most; 0 derives to ten.\n"    //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> model = uform.Model.load_vram('model.safetensors', devices=[0])   # doctest: +SKIP";

static PyObject *Model_load_vram(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    PyObject *path_object, *devices_object, *decoded = NULL, *result = NULL;
    load_options_t options;
    uform_bytes_view_t path;
    int32_t *devices = NULL;
    size_t devices_count = 0;
    int gpu_class = 0;
    if (load_parse("load_vram", args, positional_args_count, args_names_tuple, "path", &path_object, "devices",
                   &devices_object, "gpu_class", &uform_py_gpu_class, &gpu_class, &options) != 0 ||
        uform_py_path_from_object(path_object, "path", &path, &decoded) != 0 ||
        uform_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    uform_model_t handle = NULL;
    uform_model_load_vram_t arguments = {
        .model = &handle,
        .executor = options.executor,
        .outcome = &options.call.outcome,
        .path = path,
        .vram = {devices, devices_count, (uform_gpu_class_t)gpu_class},
        .quantization = options.quantization,
        .progress = uform_py_progress_registered(&options.progress, options.progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    uform_model_load_vram(&arguments);
    Py_END_ALLOW_THREADS;
    result = model_loaded(type, handle, &options.call);

release:
    PyMem_Free(devices);
    Py_XDECREF(decoded);
    Py_XDECREF(options.progress.callable);
    return result;
}

static char const doc_Model_load_vram_buffer[] =                                          //
    "load_vram_buffer(buffer, /, *, devices=None, gpu_class=None, executor=None, "        //
    "quantization=Scalar.BF16, progress=None, progress_steps=0) -> Model\n"               //
    "\n"                                                                                  //
    "Copies a safetensors checkpoint held in host memory onto the named GPUs, as\n"       //
    "`load_vram` does from a path.\n"                                                     //
    "Calls `uform_model_load_vram_buffer`.\n"                                             //
    "\n"                                                                                  //
    "Args:\n"                                                                             //
    "  buffer (buffer): The checkpoint itself, any contiguous bytes-like object.\n"       //
    "  devices (Sequence[int], optional): Positions in `gpu_devices()` to place it on.\n" //
    "  gpu_class (GPUClass, optional): The class every device dispatches as.\n"           //
    "  executor (Executor, optional): The executor the upload runs on.\n"                 //
    "  quantization (Scalar, optional): The scalar the transformer blocks run in.\n"      //
    "  progress (callable, optional): Called with (completed, total) tensors placed.\n"   //
    "  progress_steps (int, optional): Reports the load makes at most; 0 derives to ten.";

static PyObject *Model_load_vram_buffer(PyTypeObject *type, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple) {
    PyObject *buffer_object, *devices_object, *result = NULL;
    load_options_t options;
    Py_buffer buffer = {0};
    int32_t *devices = NULL;
    size_t devices_count = 0;
    int gpu_class = 0;
    if (load_parse("load_vram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "devices", &devices_object, "gpu_class", &uform_py_gpu_class, &gpu_class, &options) != 0 ||
        PyObject_GetBuffer(buffer_object, &buffer, PyBUF_C_CONTIGUOUS) != 0 ||
        uform_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    uform_model_t handle = NULL;
    uform_model_load_vram_buffer_t arguments = {
        .model = &handle,
        .executor = options.executor,
        .outcome = &options.call.outcome,
        .buffer = {(uint8_t const *)buffer.buf, (size_t)buffer.len},
        .vram = {devices, devices_count, (uform_gpu_class_t)gpu_class},
        .quantization = options.quantization,
        .progress = uform_py_progress_registered(&options.progress, options.progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    uform_model_load_vram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = model_loaded(type, handle, &options.call);

release:
    PyMem_Free(devices);
    if (buffer.obj) PyBuffer_Release(&buffer);
    Py_XDECREF(options.progress.callable);
    return result;
}

static char const doc_Model_on[] =                                                             //
    "on(executor) -> On[Model]\n"                                                              //
    "\n"                                                                                       //
    "A receiver whose verbs block on `executor`. A verb that finds another call holding the\n" //
    "executor raises `ContendedError` without running.\n"                                      //
    "It offers every verb the model does, with the same arguments and answers. Python-only.\n" //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor): The executor the calls run on; its residency must match.";

static char const doc_Model_deferred[] =                                                        //
    "deferred(executor) -> DeferredOn[Model]\n"                                                 //
    "\n"                                                                                        //
    "A receiver whose verbs submit their call to `executor` and answer a `Deferred` at once,\n" //
    "with the same arguments the model takes. It holds the model and the executor until\n"      //
    "every `Deferred` it answered is gone. Python-only.\n"                                      //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  executor (Executor): The executor the calls run on; its residency must match.";

static PyObject *Model_on(PyObject *self, PyObject *executor) { return uform_py_on_new(&OnType, self, executor); }

static PyObject *Model_deferred(PyObject *self, PyObject *executor) {
    return uform_py_on_new(&DeferredOnType, self, executor);
}

static PyObject *Model_repr(Model *self) { return PyUnicode_FromFormat("uform.Model(at=%p)", (void *)self->handle); }

#pragma endregion Construction

#pragma region Introspection

static PyStructSequence_Field ModelDescription_fields[] = {
    {"dimensions", "Scalars per embedding row; Matryoshka prefixes of it are embeddings too."},
    {"context", "Tokens one item may span before it is truncated."},
    {"quantization", "The `Scalar` the transformer blocks run in."},
    {"memory_usage", "Bytes the model holds; activations are the executor's."},
    {"cpu_class", "The `CPUClass` a RAM model was packed for, or None for a VRAM model."},
    {"gpu_class", "The `GPUClass` every replica of a VRAM model runs as, or None for a RAM model."},
    {"token_width", "Scalars per token vector; 0 when the checkpoint carries no token head."},
    {"choice_temperature_groups", "Question types of `text_choose`: 0 without the byte decoder, 1 when uncalibrated."},
    {NULL, NULL},
};

PyStructSequence_Desc uform_py_model_description_desc = {
    "uform.ModelDescription", "What a loaded model reports about itself, as `uform_model_description_t` holds it.",
    ModelDescription_fields, 8};

PyTypeObject ModelDescriptionType;

static char const doc_Model_describe[] =                                         //
    "describe() -> ModelDescription\n"                                           //
    "\n"                                                                         //
    "Reports the model's geometry and placement in one call.\n"                  //
    "Calls `uform_model_describe`.\n"                                            //
    "\n"                                                                         //
    "Example:\n"                                                                 //
    "  >>> model = uform.Model.load_ram_buffer(b'any bytes the stub accepts')\n" //
    "  >>> model.describe().context\n"                                           //
    "  512";

/** A @c uform_model_describe call and the description it lands. */
typedef struct describe_call_t {
    uform_py_call_t call;
    uform_model_describe_t arguments;
    uform_model_description_t description;
} describe_call_t;

static void describe_submit(void *call) { uform_model_describe(&((describe_call_t *)call)->arguments); }

static PyObject *describe_answer(void *erased) {
    uform_model_description_t const *const description = &((describe_call_t *)erased)->description;
    PyObject *fields[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    (void)((fields[0] = PyLong_FromSize_t(description->dimensions)) &&
           (fields[1] = PyLong_FromSize_t(description->context)) &&
           (fields[2] = uform_py_enum_to_object(&uform_py_scalar, (int)description->quantization)) &&
           (fields[3] = PyLong_FromSize_t(description->memory_usage)) &&
           (fields[4] = uform_py_class_to_object(&uform_py_cpu_class, (int)description->cpu_class)) &&
           (fields[5] = uform_py_class_to_object(&uform_py_gpu_class, (int)description->gpu_class)) &&
           (fields[6] = PyLong_FromSize_t(description->token_width)) &&
           (fields[7] = PyLong_FromSize_t(description->choice_temperature_groups)));
    return uform_py_struct_sequence_steal(&ModelDescriptionType, fields, 8);
}

static uform_py_verb_t const describe_verb = {sizeof(describe_call_t), describe_submit, describe_answer,
                                              uform_py_call_release, NULL};

static PyObject *Model_describe(PyObject *self, PyObject *unused) {
    uform_py_receiver_t receiver;
    if (uform_py_receiver_from_object(self, &ModelType, &receiver) != 0) return NULL;
    describe_call_t stack;
    describe_call_t *const call = uform_py_call_open(&receiver, &describe_verb, &stack);
    if (!call) return NULL;
    call->arguments = (uform_model_describe_t) {.model = ((Model *)receiver.target)->handle,
                                                .executor = receiver.executor,
                                                .outcome = &call->call.outcome,
                                                .completion = receiver.completion,
                                                .description = &call->description};
    return uform_py_call_run(&receiver, &describe_verb, call);
}

#pragma endregion Introspection

#pragma region Registration

PyMethodDef uform_py_model_verbs[] = {
    {"describe", (PyCFunction)Model_describe, METH_NOARGS, doc_Model_describe},
    {NULL, NULL, 0, NULL},
};

/** The constructors and receivers only a model offers; `lib.c` appends the verbs it shares
 *  with its receivers. */
PyMethodDef uform_py_model_methods[] = {
    {"load_ram", (PyCFunction)Model_load_ram, uform_py_method_flags_k | METH_CLASS, doc_Model_load_ram},
    {"load_ram_buffer", (PyCFunction)Model_load_ram_buffer, uform_py_method_flags_k | METH_CLASS,
     doc_Model_load_ram_buffer},
    {"load_vram", (PyCFunction)Model_load_vram, uform_py_method_flags_k | METH_CLASS, doc_Model_load_vram},
    {"load_vram_buffer", (PyCFunction)Model_load_vram_buffer, uform_py_method_flags_k | METH_CLASS,
     doc_Model_load_vram_buffer},
    {"on", (PyCFunction)Model_on, METH_O, doc_Model_on},
    {"deferred", (PyCFunction)Model_deferred, METH_O, doc_Model_deferred},
    {NULL, NULL, 0, NULL},
};

PyTypeObject ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.Model",
    .tp_doc = doc_Model,
    .tp_basicsize = sizeof(Model),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Model_tp_new,
    .tp_dealloc = (destructor)Model_dealloc,
    .tp_repr = (reprfunc)Model_repr,
};

#pragma endregion Registration
