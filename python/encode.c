/**
 *  @file python/encode.c
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief CPython forward passes: text, images, video and audio in, pooled embeddings and per-token states out,
 *      one verb per input form.
 *
 *  Every verb runs the same tower and keeps whichever surfaces its `outputs` flags name, from one forward. The
 *  flags decide which of the C output pointers the binding passes, so a surface left out costs nothing past the
 *  final norm. Every verb is written once and reached through three receivers: the model runs it on the calling
 *  thread, an @c On blocks on its executor, and a @c DeferredOn submits it and answers a @c Deferred. The call
 *  owns the tape its inputs were gathered into, or holds the buffers they were lent from, until it is final.
 */
#include "lib.h"

#pragma region Outputs

/** The flags of `uform.Outputs`, named after the two output fields of every encode verb. */
enum {
    outputs_embeddings_k = 1,
    outputs_states_k = 2,
};

static char const *spell_outputs(int value) {
    return value == outputs_embeddings_k ? "embeddings" : value == outputs_states_k ? "states" : NULL;
}

uform_py_enum_t uform_py_outputs = {.name = "Outputs",
                                    .base = "IntFlag",
                                    .first = outputs_embeddings_k,
                                    .last = outputs_states_k,
                                    .spell = spell_outputs};

/** Reads an `Outputs` flag set, defaulting to embeddings alone; returns 0 or -1 with an exception set. */
static int outputs_from_object(PyObject *object, int *outputs) {
    *outputs = outputs_embeddings_k;
    if (!object) return 0;
    if (!PyLong_Check(object) || PyBool_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the 'outputs' option takes uform.Outputs flags, got %S", Py_TYPE(object));
        return -1;
    }
    long const value = PyLong_AsLong(object);
    if (value == -1 && PyErr_Occurred()) return -1;
    if (value < 0 || value > (outputs_embeddings_k | outputs_states_k))
        return uform_py_raise(uform_status_invalid_argument_k, "the 'outputs' option holds no such flags: %R", object);
    *outputs = (int)value;
    return 0;
}

static PyStructSequence_Field Encoded_fields[] = {
    {"embeddings", "The (count, dimensions) `ArenaBuffer` of pooled embeddings, or None when not asked for."},
    {"states", "The (tokens, dimensions) `RowsTape` of final-norm states and its offsets, or None when not asked for."},
    {NULL, NULL},
};

PyStructSequence_Desc uform_py_encoded_desc = {
    "uform.Encoded",
    "What an encoding wrote into its arena, unpacking as `embeddings, states = encoded`; each views the arena " //
    "until a later call rewinds it.",
    Encoded_fields, 2};

PyTypeObject EncodedType;

#pragma endregion Outputs

#pragma region Reports

/** What an encoding reports beside its outputs, one layout for every modality's own type. */
typedef struct EncodeStats {
    PyObject ob_base;
    union {
        uform_text_encode_stats_t text;
        uform_image_encode_stats_t image;
        uform_video_encode_stats_t video;
        uform_audio_encode_stats_t audio;
    } stats;
} EncodeStats;

static PyObject *EncodeStats_tp_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_Format(PyExc_TypeError, "%s() takes no arguments", type->tp_name + sizeof("uform.") - 1);
        return NULL;
    }
    return type->tp_alloc(type, 0); // Zeroed, so a fresh report has counted nothing.
}

static void EncodeStats_dealloc(EncodeStats *self) { Py_TYPE(self)->tp_free((PyObject *)self); }

static PyObject *EncodeStats_repr(EncodeStats *self) {
    return PyUnicode_FromFormat("%s(truncated=%zu, tokens=%zu)", Py_TYPE(self)->tp_name + sizeof("uform.") - 1,
                                self->stats.text.truncated, self->stats.text.tokens);
}

/** Reads the @c size_t counter at the byte offset @p closure names inside the report. */
static PyObject *EncodeStats_get_counter(EncodeStats *self, void *closure) {
    size_t counter = 0;
#if PY_VERSION_HEX >= 0x030D0000
    Py_BEGIN_CRITICAL_SECTION(self);
#endif
    counter = *(size_t const *)(void const *)((char const *)&self->stats + (size_t)closure);
#if PY_VERSION_HEX >= 0x030D0000
    Py_END_CRITICAL_SECTION();
#endif
    return PyLong_FromSize_t(counter);
}

static PyGetSetDef EncodeStats_getset[] = {
    {"truncated", (getter)EncodeStats_get_counter, NULL, "Items cut at the model's context.",
     (void *)offsetof(uform_text_encode_stats_t, truncated)},
    {"tokens", (getter)EncodeStats_get_counter, NULL, "Tokens the call encoded, after truncation.",
     (void *)offsetof(uform_text_encode_stats_t, tokens)},
    {NULL, NULL, NULL, NULL, NULL},
};

static char const doc_TextEncodeStats[] =                                      //
    "TextEncodeStats()\n"                                                      //
    "\n"                                                                       //
    "What `text_encode` reports, overwritten by every call it is passed to:\n" //
    "`truncated` items and `tokens` bytes encoded.";

PyTypeObject TextEncodeStatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.TextEncodeStats",
    .tp_doc = doc_TextEncodeStats,
    .tp_basicsize = sizeof(EncodeStats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = EncodeStats_tp_new,
    .tp_dealloc = (destructor)EncodeStats_dealloc,
    .tp_repr = (reprfunc)EncodeStats_repr,
    .tp_getset = EncodeStats_getset,
};

static char const doc_ImageEncodeStats[] =                                                               //
    "ImageEncodeStats()\n"                                                                               //
    "\n"                                                                                                 //
    "What `image_encode` and `image_encode_pixels` report, overwritten by every call it is passed to:\n" //
    "`truncated` items and `tokens` patches encoded.";

PyTypeObject ImageEncodeStatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.ImageEncodeStats",
    .tp_doc = doc_ImageEncodeStats,
    .tp_basicsize = sizeof(EncodeStats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = EncodeStats_tp_new,
    .tp_dealloc = (destructor)EncodeStats_dealloc,
    .tp_repr = (reprfunc)EncodeStats_repr,
    .tp_getset = EncodeStats_getset,
};

static char const doc_VideoEncodeStats[] =                                      //
    "VideoEncodeStats()\n"                                                      //
    "\n"                                                                        //
    "What `video_encode` reports, overwritten by every call it is passed to:\n" //
    "`truncated` items and `tokens` tubes encoded.";

PyTypeObject VideoEncodeStatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.VideoEncodeStats",
    .tp_doc = doc_VideoEncodeStats,
    .tp_basicsize = sizeof(EncodeStats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = EncodeStats_tp_new,
    .tp_dealloc = (destructor)EncodeStats_dealloc,
    .tp_repr = (reprfunc)EncodeStats_repr,
    .tp_getset = EncodeStats_getset,
};

static char const doc_AudioEncodeStats[] =                                                                //
    "AudioEncodeStats()\n"                                                                                //
    "\n"                                                                                                  //
    "What `audio_encode` and `audio_encode_samples` report, overwritten by every call it is passed to:\n" //
    "`truncated` items and `tokens` frames encoded.";

PyTypeObject AudioEncodeStatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "uform.AudioEncodeStats",
    .tp_doc = doc_AudioEncodeStats,
    .tp_basicsize = sizeof(EncodeStats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = EncodeStats_tp_new,
    .tp_dealloc = (destructor)EncodeStats_dealloc,
    .tp_repr = (reprfunc)EncodeStats_repr,
    .tp_getset = EncodeStats_getset,
};

#pragma endregion Reports

#pragma region Calls

/** One encode call, the inputs it owns or borrows until it is final, and the outputs it lands. */
typedef struct encode_call_t {
    uform_py_call_t call;
    union {
        uform_text_encode_t text;
        uform_image_encode_t image;
        uform_image_encode_pixels_t pixels;
        uform_video_encode_t video;
        uform_audio_encode_t audio;
        uform_audio_encode_samples_t samples;
    } arguments;
    uform_py_tape_t inputs;
    uform_rows_view_t embeddings;
    uform_rows_tape_view_t states;

    /** Always passed, since its `tokens` is the row count of the states, wherever the arena lives. */
    union {
        uform_text_encode_stats_t text;
        uform_image_encode_stats_t image;
        uform_video_encode_stats_t video;
        uform_audio_encode_stats_t audio;
    } stats;

    /** The caller's report the stats are copied into once the call succeeds, held, or NULL. */
    PyObject *stats_object;
    uform_py_callback_t progress;

    /** The `Outputs` flags the caller named. */
    int outputs;
} encode_call_t;

static PyObject *encode_answer(void *erased) {
    encode_call_t *const call = (encode_call_t *)erased;
    if (call->stats_object) {
#if PY_VERSION_HEX >= 0x030D0000
        Py_BEGIN_CRITICAL_SECTION(call->stats_object);
#endif
        ((EncodeStats *)call->stats_object)->stats.text = call->stats.text;
#if PY_VERSION_HEX >= 0x030D0000
        Py_END_CRITICAL_SECTION();
#endif
    }
    size_t const count = call->inputs.count;
    PyObject *fields[2] = {NULL, NULL};
    fields[0] = call->outputs & outputs_embeddings_k
                    ? uform_py_rows_answer(&ArenaBufferType, call->call.arena, &call->embeddings, count)
                    : Py_NewRef(Py_None);
    if (fields[0])
        fields[1] = call->outputs & outputs_states_k
                        ? uform_py_rows_tape_answer(call->call.arena, &call->states, count, call->stats.text.tokens)
                        : Py_NewRef(Py_None);
    return uform_py_struct_sequence_steal(&EncodedType, fields, 2);
}

static void encode_release(void *erased) {
    encode_call_t *const call = (encode_call_t *)erased;
    uform_py_tape_release(&call->inputs);
    Py_CLEAR(call->progress.callable);
    Py_CLEAR(call->stats_object);
    uform_py_call_release(&call->call);
}

static int encode_traverse(void *erased, visitproc visit, void *arg) {
    encode_call_t *const call = (encode_call_t *)erased;
    Py_VISIT(call->progress.callable);
    Py_VISIT(call->stats_object);
    return 0;
}

/** What tells one encode verb from another: how its inputs gather, the report it fills, and how it runs. */
typedef struct encode_verb_t {
    char const *input_name;
    uform_py_element_t element;
    PyTypeObject *stats_type;
    uform_py_verb_t verb;
} encode_verb_t;

/** The arguments every encode verb binds, `sample_rate` only for decoded samples. */
typedef struct encode_arguments_t {
    PyObject *inputs, *sample_rate, *arena, *outputs, *outputs_scalar, *progress, *progress_steps, *stats;
} encode_arguments_t;

/** The fields every encode verb's C structure shares, read once for all six. */
typedef struct encode_common_t {
    uform_model_t model;
    uform_executor_t executor;
    uform_completion_t completion;
    uform_rows_view_t *embeddings;
    uform_rows_tape_view_t *states;
    uform_arena_t arena;
    uform_scalar_t outputs_scalar;
    uform_progress_t progress;
} encode_common_t;

/** Binds `(inputs, arena, /, *, outputs, outputs_scalar, progress, progress_steps, stats)`; returns 0 or -1. */
static int encode_parse(char const *verb, encode_verb_t const *descriptor, PyObject *const *args,
                        Py_ssize_t positional_args_count, PyObject *args_names_tuple, encode_arguments_t *parsed) {
    uform_py_argument_t const arguments[] = {
        {descriptor->input_name, &parsed->inputs},
        {"arena", &parsed->arena},
        {"outputs", &parsed->outputs},
        {"outputs_scalar", &parsed->outputs_scalar},
        {"progress", &parsed->progress},
        {"progress_steps", &parsed->progress_steps},
        {"stats", &parsed->stats},
    };
    parsed->sample_rate = NULL;
    return uform_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                    sizeof(arguments) / sizeof(*arguments));
}

/**
 *  @brief Opens the call of @p descriptor on @p self, gathering its inputs, claiming its arena and reading
 *      every field the six C structures share into @p common.
 *  @return The call to fill, or NULL with an exception set.
 */
static encode_call_t *encode_open(PyObject *self, encode_verb_t const *descriptor, encode_arguments_t const *parsed,
                                  uform_py_receiver_t *receiver, encode_common_t *common, encode_call_t *stack) {
    int outputs = 0;
    size_t progress_steps = 0;
    uform_scalar_t outputs_scalar = uform_scalar_f32_k;
    if (uform_py_receiver_from_object(self, &ModelType, receiver) != 0 ||
        outputs_from_object(parsed->outputs, &outputs) != 0 ||
        uform_py_scalar_from_object(parsed->outputs_scalar, "outputs_scalar", &outputs_scalar) != 0 ||
        uform_py_size_from_object(parsed->progress_steps, "progress_steps", &progress_steps) != 0)
        return NULL;
    if (parsed->stats && parsed->stats != Py_None && !Py_IS_TYPE(parsed->stats, descriptor->stats_type)) {
        PyErr_Format(PyExc_TypeError, "the 'stats' option must be a %s or None, got %S",
                     descriptor->stats_type->tp_name + sizeof("uform.") - 1, Py_TYPE(parsed->stats));
        return NULL;
    }

    encode_call_t *const call = uform_py_call_open(receiver, &descriptor->verb, stack);
    if (!call) return NULL;
    if (uform_py_tape_from_object(parsed->inputs, descriptor->input_name, descriptor->element, &call->inputs) != 0 ||
        uform_py_callback_from_object(parsed->progress, "progress", &call->call, &call->progress) != 0 ||
        uform_py_arena_claim(parsed->arena, &call->call.arena) != 0) {
        uform_py_call_discard(receiver, &descriptor->verb, call);
        return NULL;
    }
    call->outputs = outputs;
    if (parsed->stats && parsed->stats != Py_None) call->stats_object = Py_NewRef(parsed->stats);
    *common = (encode_common_t) {
        .model = ((Model *)receiver->target)->handle,
        .executor = receiver->executor,
        .completion = receiver->completion,
        .embeddings = outputs & outputs_embeddings_k ? &call->embeddings : NULL,
        .states = outputs & outputs_states_k ? &call->states : NULL,
        .arena = ((Arena *)call->call.arena)->handle,
        .outputs_scalar = outputs_scalar,
        .progress = uform_py_progress_registered(&call->progress, progress_steps),
    };
    return call;
}

#pragma endregion Calls

#pragma region Verbs

static void text_encode_submit(void *call) { uform_text_encode(&((encode_call_t *)call)->arguments.text); }
static void image_encode_submit(void *call) { uform_image_encode(&((encode_call_t *)call)->arguments.image); }
static void image_encode_pixels_submit(void *call) {
    uform_image_encode_pixels(&((encode_call_t *)call)->arguments.pixels);
}
static void video_encode_submit(void *call) { uform_video_encode(&((encode_call_t *)call)->arguments.video); }
static void audio_encode_submit(void *call) { uform_audio_encode(&((encode_call_t *)call)->arguments.audio); }
static void audio_encode_samples_submit(void *call) {
    uform_audio_encode_samples(&((encode_call_t *)call)->arguments.samples);
}

static encode_verb_t const text_encode_verb = {
    "texts",
    uform_py_element_text_k,
    &TextEncodeStatsType,
    {sizeof(encode_call_t), text_encode_submit, encode_answer, encode_release, encode_traverse}};
static encode_verb_t const image_encode_verb = {
    "files",
    uform_py_element_bytes_k,
    &ImageEncodeStatsType,
    {sizeof(encode_call_t), image_encode_submit, encode_answer, encode_release, encode_traverse}};
static encode_verb_t const image_encode_pixels_verb = {
    "pixels",
    uform_py_element_pixels_k,
    &ImageEncodeStatsType,
    {sizeof(encode_call_t), image_encode_pixels_submit, encode_answer, encode_release, encode_traverse}};
static encode_verb_t const video_encode_verb = {
    "files",
    uform_py_element_bytes_k,
    &VideoEncodeStatsType,
    {sizeof(encode_call_t), video_encode_submit, encode_answer, encode_release, encode_traverse}};
static encode_verb_t const audio_encode_verb = {
    "files",
    uform_py_element_bytes_k,
    &AudioEncodeStatsType,
    {sizeof(encode_call_t), audio_encode_submit, encode_answer, encode_release, encode_traverse}};
static encode_verb_t const audio_encode_samples_verb = {
    "samples",
    uform_py_element_samples_k,
    &AudioEncodeStatsType,
    {sizeof(encode_call_t), audio_encode_samples_submit, encode_answer, encode_release, encode_traverse}};

static char const doc_Model_text_encode[] =                                                             //
    "text_encode(texts, arena, /, *, outputs=Outputs.EMBEDDINGS, outputs_scalar=Scalar.F32,\n"          //
    "    progress=None, progress_steps=0, stats=None) -> Encoded\n"                                     //
    "\n"                                                                                                //
    "Encodes UTF-8 texts through one forward, keeping the surfaces `outputs` names. Models are\n"       //
    "tokenizer-free, so every byte is a token, and a text longer than the context is cut.\n"            //
    "Calls `uform_text_encode`.\n"                                                                      //
    "\n"                                                                                                //
    "Args:\n"                                                                                           //
    "  texts (Sequence[str | bytes] | tuple[buffer, buffer]): The texts, gathered into one tape,\n"     //
    "    or a (data, offsets) pair of a byte buffer and count + 1 size_t offsets, lent as is.\n"        //
    "  arena (Arena): Where the outputs land, rewound by the call.\n"                                   //
    "  outputs (Outputs, optional): `Outputs.EMBEDDINGS`, `Outputs.STATES`, or both.\n"                 //
    "  outputs_scalar (Scalar, optional): The scalar every output lands as.\n"                          //
    "  progress (callable, optional): Called with (completed, total) items; returning False\n"          //
    "    stops the call and raises `CancelledError`.\n"                                                 //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"             //
    "  stats (TextEncodeStats, optional): Overwritten with this call's report.\n"                       //
    "\n"                                                                                                //
    "Returns:\n"                                                                                        //
    "  Encoded: `embeddings`, a (count, dimensions) `ArenaBuffer`, and `states`, a (tokens,\n"          //
    "    dimensions) `RowsTape` with `offsets`, each None when `outputs` leaves it out.\n"              //
    "\n"                                                                                                //
    "Example:\n"                                                                                        //
    "  >>> model = uform.Model.load_ram_buffer(b'any bytes the stub accepts')\n"                        //
    "  >>> encoded = model.text_encode(['hi', 'there'], uform.Arena(), outputs=uform.Outputs.STATES)\n" //
    "  >>> encoded.embeddings is None, len(encoded.states), len(encoded.states.offsets)\n"              //
    "  (True, 7, 3)";

static PyObject *Model_text_encode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {
    encode_arguments_t parsed;
    uform_py_receiver_t receiver;
    encode_common_t common;
    encode_call_t stack;
    if (encode_parse("text_encode", &text_encode_verb, args, positional_args_count, args_names_tuple, &parsed) != 0)
        return NULL;
    encode_call_t *const call = encode_open(self, &text_encode_verb, &parsed, &receiver, &common, &stack);
    if (!call) return NULL;
    call->arguments.text = (uform_text_encode_t) {
        .model = common.model,
        .executor = common.executor,
        .outcome = &call->call.outcome,
        .completion = common.completion,
        .count = call->inputs.count,
        .texts = call->inputs.view,
        .embeddings = common.embeddings,
        .states = common.states,
        .arena = common.arena,
        .outputs_scalar = common.outputs_scalar,
        .progress = common.progress,
        .stats = &call->stats.text,
    };
    return uform_py_call_run(&receiver, &text_encode_verb.verb, call);
}

static char const doc_Model_image_encode[] =                                                     //
    "image_encode(files, arena, /, *, outputs=Outputs.EMBEDDINGS, outputs_scalar=Scalar.F32,\n"  //
    "    progress=None, progress_steps=0, stats=None) -> Encoded\n"                              //
    "\n"                                                                                         //
    "Decodes and encodes JPEG, PNG or WebP files; an undecodable one raises `MalformedError`.\n" //
    "Calls `uform_image_encode`.\n"                                                              //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  files (Sequence[bytes-like] | tuple[buffer, buffer]): The encoded files, or a (data,\n"   //
    "    offsets) pair lent as is.\n"                                                            //
    "  arena (Arena): Where the outputs land, rewound by the call.\n"                            //
    "  outputs, outputs_scalar, progress, progress_steps: As for `text_encode`.\n"               //
    "  stats (ImageEncodeStats, optional): Overwritten with this call's report.";

static PyObject *Model_image_encode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    encode_arguments_t parsed;
    uform_py_receiver_t receiver;
    encode_common_t common;
    encode_call_t stack;
    if (encode_parse("image_encode", &image_encode_verb, args, positional_args_count, args_names_tuple, &parsed) != 0)
        return NULL;
    encode_call_t *const call = encode_open(self, &image_encode_verb, &parsed, &receiver, &common, &stack);
    if (!call) return NULL;
    call->arguments.image = (uform_image_encode_t) {
        .model = common.model,
        .executor = common.executor,
        .outcome = &call->call.outcome,
        .completion = common.completion,
        .count = call->inputs.count,
        .images = call->inputs.view,
        .embeddings = common.embeddings,
        .states = common.states,
        .arena = common.arena,
        .outputs_scalar = common.outputs_scalar,
        .progress = common.progress,
        .stats = &call->stats.image,
    };
    return uform_py_call_run(&receiver, &image_encode_verb.verb, call);
}

static char const doc_Model_image_encode_pixels[] =                                                     //
    "image_encode_pixels(pixels, arena, /, *, outputs=Outputs.EMBEDDINGS, outputs_scalar=Scalar.F32,\n" //
    "    progress=None, progress_steps=0, stats=None) -> Encoded\n"                                     //
    "\n"                                                                                                //
    "Encodes decoded images, resizing on the executor's side.\n"                                        //
    "Calls `uform_image_encode_pixels`.\n"                                                              //
    "\n"                                                                                                //
    "Args:\n"                                                                                           //
    "  pixels (Sequence[buffer]): C-contiguous [height, width, 3] buffers of uint8 RGB, gathered\n"     //
    "    into one tape with their heights and widths.\n"                                                //
    "  arena (Arena): Where the outputs land, rewound by the call.\n"                                   //
    "  outputs, outputs_scalar, progress, progress_steps: As for `text_encode`.\n"                      //
    "  stats (ImageEncodeStats, optional): Overwritten with this call's report.";

static PyObject *Model_image_encode_pixels(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                           PyObject *args_names_tuple) {
    encode_arguments_t parsed;
    uform_py_receiver_t receiver;
    encode_common_t common;
    encode_call_t stack;
    if (encode_parse("image_encode_pixels", &image_encode_pixels_verb, args, positional_args_count, args_names_tuple,
                     &parsed) != 0)
        return NULL;
    encode_call_t *const call = encode_open(self, &image_encode_pixels_verb, &parsed, &receiver, &common, &stack);
    if (!call) return NULL;
    size_t const count = call->inputs.count;
    call->arguments.pixels = (uform_image_encode_pixels_t) {
        .model = common.model,
        .executor = common.executor,
        .outcome = &call->call.outcome,
        .completion = common.completion,
        .count = count,
        .pixels = call->inputs.view,
        .heights = call->inputs.extents,
        .widths = call->inputs.extents ? call->inputs.extents + count : NULL,
        .embeddings = common.embeddings,
        .states = common.states,
        .arena = common.arena,
        .outputs_scalar = common.outputs_scalar,
        .progress = common.progress,
        .stats = &call->stats.image,
    };
    return uform_py_call_run(&receiver, &image_encode_pixels_verb.verb, call);
}

static char const doc_Model_video_encode[] =                                                    //
    "video_encode(files, arena, /, *, outputs=Outputs.EMBEDDINGS, outputs_scalar=Scalar.F32,\n" //
    "    progress=None, progress_steps=0, stats=None) -> Encoded\n"                             //
    "\n"                                                                                        //
    "Decodes and encodes MP4 or WebM clips.\n"                                                  //
    "Calls `uform_video_encode`.\n"                                                             //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  files (Sequence[bytes-like] | tuple[buffer, buffer]): The encoded clips, or a (data,\n"  //
    "    offsets) pair lent as is.\n"                                                           //
    "  arena (Arena): Where the outputs land, rewound by the call.\n"                           //
    "  outputs, outputs_scalar, progress, progress_steps: As for `text_encode`.\n"              //
    "  stats (VideoEncodeStats, optional): Overwritten with this call's report.";

static PyObject *Model_video_encode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    encode_arguments_t parsed;
    uform_py_receiver_t receiver;
    encode_common_t common;
    encode_call_t stack;
    if (encode_parse("video_encode", &video_encode_verb, args, positional_args_count, args_names_tuple, &parsed) != 0)
        return NULL;
    encode_call_t *const call = encode_open(self, &video_encode_verb, &parsed, &receiver, &common, &stack);
    if (!call) return NULL;
    call->arguments.video = (uform_video_encode_t) {
        .model = common.model,
        .executor = common.executor,
        .outcome = &call->call.outcome,
        .completion = common.completion,
        .count = call->inputs.count,
        .videos = call->inputs.view,
        .embeddings = common.embeddings,
        .states = common.states,
        .arena = common.arena,
        .outputs_scalar = common.outputs_scalar,
        .progress = common.progress,
        .stats = &call->stats.video,
    };
    return uform_py_call_run(&receiver, &video_encode_verb.verb, call);
}

static char const doc_Model_audio_encode[] =                                                    //
    "audio_encode(files, arena, /, *, outputs=Outputs.EMBEDDINGS, outputs_scalar=Scalar.F32,\n" //
    "    progress=None, progress_steps=0, stats=None) -> Encoded\n"                             //
    "\n"                                                                                        //
    "Decodes, resamples to mono and encodes WAV, FLAC, MP3 or Opus recordings.\n"               //
    "Calls `uform_audio_encode`.\n"                                                             //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  files (Sequence[bytes-like] | tuple[buffer, buffer]): The encoded recordings, or a\n"    //
    "    (data, offsets) pair lent as is.\n"                                                    //
    "  arena (Arena): Where the outputs land, rewound by the call.\n"                           //
    "  outputs, outputs_scalar, progress, progress_steps: As for `text_encode`.\n"              //
    "  stats (AudioEncodeStats, optional): Overwritten with this call's report.";

static PyObject *Model_audio_encode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    encode_arguments_t parsed;
    uform_py_receiver_t receiver;
    encode_common_t common;
    encode_call_t stack;
    if (encode_parse("audio_encode", &audio_encode_verb, args, positional_args_count, args_names_tuple, &parsed) != 0)
        return NULL;
    encode_call_t *const call = encode_open(self, &audio_encode_verb, &parsed, &receiver, &common, &stack);
    if (!call) return NULL;
    call->arguments.audio = (uform_audio_encode_t) {
        .model = common.model,
        .executor = common.executor,
        .outcome = &call->call.outcome,
        .completion = common.completion,
        .count = call->inputs.count,
        .recordings = call->inputs.view,
        .embeddings = common.embeddings,
        .states = common.states,
        .arena = common.arena,
        .outputs_scalar = common.outputs_scalar,
        .progress = common.progress,
        .stats = &call->stats.audio,
    };
    return uform_py_call_run(&receiver, &audio_encode_verb.verb, call);
}

static char const doc_Model_audio_encode_samples[] =                                                 //
    "audio_encode_samples(samples, sample_rate, arena, /, *, outputs=Outputs.EMBEDDINGS,\n"          //
    "    outputs_scalar=Scalar.F32, progress=None, progress_steps=0, stats=None) -> Encoded\n"       //
    "\n"                                                                                             //
    "Encodes decoded mono recordings, resampled to the model's rate where `sample_rate` differs.\n"  //
    "Calls `uform_audio_encode_samples`.\n"                                                          //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  samples (Sequence[buffer] | tuple[buffer, buffer]): 1-D float32 buffers, gathered into one\n" //
    "    tape, or a (data, offsets) pair lent as is, its offsets counted in samples.\n"              //
    "  sample_rate (int): Samples per second shared by every recording.\n"                           //
    "  arena (Arena): Where the outputs land, rewound by the call.\n"                                //
    "  outputs, outputs_scalar, progress, progress_steps: As for `text_encode`.\n"                   //
    "  stats (AudioEncodeStats, optional): Overwritten with this call's report.";

static PyObject *Model_audio_encode_samples(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple) {
    encode_arguments_t parsed;
    uform_py_receiver_t receiver;
    encode_common_t common;
    encode_call_t stack;
    uform_py_argument_t const arguments[] = {
        {"samples", &parsed.inputs},
        {"sample_rate", &parsed.sample_rate},
        {"arena", &parsed.arena},
        {"outputs", &parsed.outputs},
        {"outputs_scalar", &parsed.outputs_scalar},
        {"progress", &parsed.progress},
        {"progress_steps", &parsed.progress_steps},
        {"stats", &parsed.stats},
    };
    size_t sample_rate = 0;
    if (uform_py_arguments_parse("audio_encode_samples", args, positional_args_count, args_names_tuple, 3, 3, arguments,
                                 sizeof(arguments) / sizeof(*arguments)) != 0 ||
        uform_py_size_from_object(parsed.sample_rate, "sample_rate", &sample_rate) != 0)
        return NULL;
    if (sample_rate > UINT32_MAX)
        return uform_py_raise(uform_status_invalid_argument_k, "the 'sample_rate' option exceeds 32 bits"), NULL;
    encode_call_t *const call = encode_open(self, &audio_encode_samples_verb, &parsed, &receiver, &common, &stack);
    if (!call) return NULL;
    call->arguments.samples = (uform_audio_encode_samples_t) {
        .model = common.model,
        .executor = common.executor,
        .outcome = &call->call.outcome,
        .completion = common.completion,
        .count = call->inputs.count,
        .samples = call->inputs.view,
        .sample_rate = (uint32_t)sample_rate,
        .embeddings = common.embeddings,
        .states = common.states,
        .arena = common.arena,
        .outputs_scalar = common.outputs_scalar,
        .progress = common.progress,
        .stats = &call->stats.audio,
    };
    return uform_py_call_run(&receiver, &audio_encode_samples_verb.verb, call);
}

#pragma endregion Verbs

#pragma region Registration

PyMethodDef uform_py_encode_verbs[] = {
    {"text_encode", (PyCFunction)Model_text_encode, uform_py_method_flags_k, doc_Model_text_encode},
    {"image_encode", (PyCFunction)Model_image_encode, uform_py_method_flags_k, doc_Model_image_encode},
    {"image_encode_pixels", (PyCFunction)Model_image_encode_pixels, uform_py_method_flags_k,
     doc_Model_image_encode_pixels},
    {"video_encode", (PyCFunction)Model_video_encode, uform_py_method_flags_k, doc_Model_video_encode},
    {"audio_encode", (PyCFunction)Model_audio_encode, uform_py_method_flags_k, doc_Model_audio_encode},
    {"audio_encode_samples", (PyCFunction)Model_audio_encode_samples, uform_py_method_flags_k,
     doc_Model_audio_encode_samples},
    {NULL, NULL, 0, NULL},
};

#pragma endregion Registration
