/**
 *  @file python/lib.h
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Declarations shared by every UForm CPython translation unit.
 *
 *  One extension, one file per header of the C interface: @c types.c owns the vocabulary every header shares,
 *  @c model.c, @c encode.c and @c train.c own their verbs, and @c lib.c owns the module glue alone. Types are
 *  defined once here and readied once in @c lib.c, so an @c Arena made for an encoding is the very object a
 *  training step accepts.
 *
 *  @section python_calling_convention Latency, Quality, and Arguments Parsing
 *
 *  No PyBind11, NanoBind, or SWIG - only the CPython API, and never @c PyArg_ParseTupleAndKeywords, whose format
 *  specifiers are parsed at runtime and which cannot use the fast calling convention. Every verb is
 *  @c METH_FASTCALL | @c METH_KEYWORDS and binds its arguments through one table of names,
 *  @c uform_py_arguments_parse.
 *
 *  @section python_receivers Receivers and Deferred Calls
 *
 *  The type of the object a verb is called on chooses where its call runs and how it completes: a model runs it
 *  on the calling thread, a trainer blocks on the executor it was created on, an @c On blocks on its executor,
 *  and a @c DeferredOn submits it there and answers a @c Deferred. A verb fills one call structure, kept on the
 *  stack while it blocks and inside its @c Deferred once submitted, so one body serves every receiver and
 *  nothing the library borrows ever moves. The binding starts no threads and keeps two counts: an arena's
 *  @c exports, since a buffer it exported cannot be revoked before C rewinds the memory under it, to which a call
 *  adds one until it is waited on or dropped, and an executor's GPU @c arenas, since each frees on the
 *  executor's stream, so a close leaves its handle to the last of them. Dropping a @c Deferred waits on its
 *  outcome, and every other wait belongs to the C library.
 *
 *  @section python_buffers Buffer Protocol
 *
 *  Every input arrives through PEP 3118, so NumPy is never a build-time or a runtime dependency. A batch of
 *  separate objects is gathered into one tape the call owns, and a @c (data, offsets) pair of buffers is lent
 *  to the library as a tape without a copy. Every output whose length scales with the call lands in the
 *  @c Arena the caller passes and leaves as zero-copy results over it. An arena exporting a buffer refuses to
 *  rewind, as a @c bytearray refuses to resize, and a result whose arena a later call rewound raises
 *  @c InvalidArgumentError when read.
 *
 *  @section python_naming Naming
 *
 *  A Python object's structure takes its class name, @c Model for @c uform.Model, and its type object
 *  @c ModelType. A C helper shared through this header takes the @c uform_py_ prefix and the @c _t suffix,
 *  since C has no namespaces. A helper one file keeps to itself takes only @c _t.
 */
#ifndef UNUM_UFORM_PYTHON_LIB_H
#define UNUM_UFORM_PYTHON_LIB_H

/*  CPython defines preprocessor macros that reach into the standard headers, so it leads rather than joining the
 *  third-party group; every unit here includes this file before anything else. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdarg.h>    // `va_list` - the formatted raise
#include <stdatomic.h> // `atomic_load_explicit`, `atomic_compare_exchange_strong`
#include <stddef.h>    // `size_t`, `offsetof`
#include <stdint.h>    // `uint32_t`, `uint64_t`, `PTRDIFF_MAX`
#include <string.h>    // `memcpy`, `memset`

#include "uform/uform.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Calling Convention

/** Compile-time constants of the binding layer. */
enum {

    /** Fast-call plus keywords, the calling convention every verb uses. */
    uform_py_method_flags_k = METH_FASTCALL | METH_KEYWORDS,

    /** Fields the widest keyword-built structure holds, `Masking`. */
    uform_py_fields_max_k = 12,
};

/** One argument a verb takes and where its value lands, left NULL when the caller skipped it. */
typedef struct uform_py_argument_t {

    /** The keyword spelling, which also names the argument in every refusal. */
    char const *name;

    /** Where the borrowed value lands. */
    PyObject **value;
} uform_py_argument_t;

/**
 *  @brief Binds a fast-call's arguments to @p arguments, refusing unknown names, duplicates and missing ones.
 *
 *  The first @p positional_only arguments bind by position alone and must be given; the ones after them up to
 *  @p positional_max bind by position or by name; the rest bind by name alone.
 *
 *  @return 0 on success, -1 with a @c TypeError set otherwise.
 */
int uform_py_arguments_parse(char const *verb, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple, Py_ssize_t positional_only, Py_ssize_t positional_max,
                             uform_py_argument_t const *arguments, size_t count);

#pragma endregion Calling Convention

#pragma region Shared Types

extern PyTypeObject ExecutorType;
extern PyTypeObject ArenaType;
extern PyTypeObject ArenaBufferType;
extern PyTypeObject RowsTapeType;
extern PyTypeObject DeferredType;
extern PyTypeObject OnType;
extern PyTypeObject DeferredOnType;

extern PyTypeObject ModelType;
extern PyTypeObject ModelDescriptionType;

extern PyTypeObject EncodedType;
extern PyTypeObject TextEncodeStatsType;
extern PyTypeObject ImageEncodeStatsType;
extern PyTypeObject VideoEncodeStatsType;
extern PyTypeObject AudioEncodeStatsType;

extern PyTypeObject MeshType;
extern PyTypeObject MeshDescriptionType;
extern PyTypeObject TrainerType;
extern PyTypeObject TrainerStepStatsType;

/** The layouts of the struct sequences, which `lib.c` readies through their own initializer. */
extern PyStructSequence_Desc uform_py_model_description_desc;
extern PyStructSequence_Desc uform_py_encoded_desc;
extern PyStructSequence_Desc uform_py_mesh_description_desc;

/** The base of the one exception class per failing status, subclassing @c RuntimeError. */
extern PyObject *ErrorType;

/** The threads or streams a verb runs on, owned apart from any model so one executor serves many. */
typedef struct Executor {
    PyObject ob_base;

    /** The library's executor, NULL once a `with` block closed it. */
    uform_executor_t handle;

    /** Devices its streams run on, at least one for a GPU executor; zero for host threads. */
    size_t devices_count;

    /** The GPU arenas it made that are still alive, each holding it, since they free on its first stream. */
    Py_ssize_t arenas;

    /** The handle a close left to those arenas, freed with the last of them. */
    uform_executor_t retained;
} Executor;

/**
 *  @brief The memory every output whose length scales with a call lands in, rewound by the next call handed it.
 *
 *  A call is refused while a buffer over the previous outputs is still exported, and every result over them
 *  refuses to be read once a later call rewound it.
 */
typedef struct Arena {
    PyObject ob_base;
    uform_arena_t handle;

    /** Buffers exported over its outputs, plus one while a call holds it. */
    Py_ssize_t exports;

    /** Calls that claimed it so far, which a result compares against the one it was written by. */
    size_t generation;

    /** The @c Executor a GPU arena serves alone, held until the arena is freed; NULL for host pages. */
    PyObject *executor;
} Arena;

/** A loaded encoder, serving every modality it was trained on. */
typedef struct Model {
    PyObject ob_base;
    uform_model_t handle;
} Model;

/** A C enum mirrored as an `enum.IntEnum`, or a binding flag set as an `enum.IntFlag`, spelled by a name table. */
typedef struct uform_py_enum_t {

    /** The class name, as `uform` exports it, such as "Status". */
    char const *name;

    /** The base class in the `enum` module, "IntEnum" or "IntFlag". */
    char const *base;

    /** The lowest and highest values spelled; a gap spelling NULL, "unknown" or "unrecognized" is no member. */
    int first, last;

    /** A one-line wrapper over `uform_status_name` and its siblings. */
    char const *(*spell)(int);

    /** The class, built once at module init and never written again. */
    PyObject *type;
} uform_py_enum_t;

extern uform_py_enum_t uform_py_status, uform_py_scalar, uform_py_mood, uform_py_outputs, uform_py_modality,
    uform_py_objective, uform_py_negatives, uform_py_multimem_tier;

/**
 *  @brief Builds every mirrored enum's class and adds it to @p module, one member per value spelled,
 *      `out-of-memory` becoming `OUT_OF_MEMORY`.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int uform_py_register_enums(PyObject *module);

/**
 *  @brief Reads a member of @p mirrored from an `int`, refusing a `bool`, text and values no member carries.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int uform_py_enum_from_object(PyObject *object, char const *field, uform_py_enum_t const *mirrored, int *value);

/** The member of @p mirrored valued @p value; returns a new reference, or @c NULL with an exception set. */
PyObject *uform_py_enum_to_object(uform_py_enum_t const *mirrored, int value);

/** What one field of a keyword-built structure holds, and how it reaches the C structure. */
typedef enum uform_py_field_kind_t {

    /** A non-negative `int`, written as a @c size_t. */
    uform_py_field_size_k,

    /** A `float`, written as a @c float. */
    uform_py_field_float_k,

    /** A `float`, written as a @c double. */
    uform_py_field_double_k,

    /** A member of the @c uform_py_enum_t the field's detail names, written as an @c int. */
    uform_py_field_enum_k,

    /** Another keyword-built structure, of the type the field's detail names, written in place; a tuple fills
     *  its fields in order. */
    uform_py_field_struct_k,

    /** Any object, kept as given, which the verb reading the structure converts itself. */
    uform_py_field_object_k,
} uform_py_field_kind_t;

/** One field of a keyword-built structure. */
typedef struct uform_py_field_t {
    char const *name;
    uform_py_field_kind_t kind;

    /** Where the field lands inside the C structure; unused for an object field. */
    size_t offset;

    /** The @c uform_py_enum_t of an enum field, or the @c uform_py_fields_t of a structure field. */
    void const *detail;
    char const *doc;
} uform_py_field_t;

/** The type of a keyword-built structure, leading with the type object so an instance finds its fields. */
typedef struct uform_py_fields_t {
    PyTypeObject type;
    uform_py_field_t const *fields;
    size_t count;
} uform_py_fields_t;

/** An instance of a keyword-built structure: the value of each field, in the order its type lists them. */
typedef struct Fields {
    PyObject ob_base;
    PyObject *values[uform_py_fields_max_k];
} Fields;

extern uform_py_fields_t ImageShapeType, VideoShapeType, AudioShapeType, BatchShapeType, MaskingType, ContrastiveType,
    ScheduleType, MeshOptionsType, EdgesType, RelationsType;

/** Readies @p fields, its members generated from its field table and kept as long as the process, like the module;
 *  returns 0 or -1 with an exception set. */
int uform_py_fields_ready(uform_py_fields_t *fields);

/**
 *  @brief Writes every field of @p object but its object fields into the C structure at @p target, nested
 *      structures included; @c None or a NULL @p object leaves @p target zeroed.
 *  @return 0 on success, -1 with a @c TypeError set when @p object is not of @p fields.
 */
int uform_py_fields_write(PyObject *object, char const *field, uform_py_fields_t const *fields, void *target);

/** The value of field @p index of @p object, borrowed. */
static inline PyObject *uform_py_fields_value(PyObject *object, size_t index) {
    return ((Fields *)object)->values[index];
}

#pragma endregion Shared Types

#pragma region Calls

/** What a verb's receiver names: the handle, the executor the call runs on and how it completes. */
typedef struct uform_py_receiver_t {

    /** The receiver object, which a deferred call holds so its handle and executor outlive the call. */
    PyObject *origin;

    /** The model or trainer the verb addresses, borrowed from @c origin. */
    PyObject *target;

    /** The executor the call runs on; NULL runs it on the calling thread. */
    uform_executor_t executor;

    /** Blocking for a model, a trainer or an @c On, deferred for a @c DeferredOn. */
    uform_completion_t completion;
} uform_py_receiver_t;

/** The verbs a model or trainer shares with its receivers, which `lib.c` merges into each type's methods. */
extern PyMethodDef uform_py_model_verbs[];
extern PyMethodDef uform_py_encode_verbs[];
extern PyMethodDef uform_py_trainer_verbs[];

/** The constructors and receivers a model or trainer alone offers. */
extern PyMethodDef uform_py_model_methods[];
extern PyMethodDef uform_py_trainer_methods[];

/**
 *  @brief Reads what @p self names, a model or trainer on the calling thread, or an @c On or a @c DeferredOn on
 *      its executor, refusing a receiver over another type than @p type.
 *  @return 0 on success, -1 with an exception set when the receiver's executor was closed or names another type.
 */
int uform_py_receiver_from_object(PyObject *self, PyTypeObject *type, uform_py_receiver_t *receiver);

/** A new @c On or @c DeferredOn, as @p type names, holding @p target and the @c Executor @p executor; NULL on
 *  failure. */
PyObject *uform_py_on_new(PyTypeObject *type, PyObject *target, PyObject *executor);

/** What every call keeps until it is final, leading its verb's call structure: on the stack while it blocks,
 *  inside its @c Deferred once submitted. */
typedef struct uform_py_call_t {

    /** The outcome the library finishes. */
    uform_outcome_t outcome;

    /** The first exception a callback raised, which outranks the outcome it turned into; NULL while none has. */
    PyObject *raised;

    /** The claimed @c Arena the call's outputs land in, or NULL for a verb with none. */
    PyObject *arena;
} uform_py_call_t;

/** One Python callable a call re-enters from library threads, and the call its first failure lands in. */
typedef struct uform_py_callback_t {

    /** The callable, held, or NULL when the caller passed none. */
    PyObject *callable;

    /** The call whose @c raised field latches the first exception. */
    uform_py_call_t *call;
} uform_py_callback_t;

/** How a verb runs a call it filled: the bytes of its call, how it reaches the library, answers and lets go. */
typedef struct uform_py_verb_t {

    /** Bytes of the verb's call structure, which leads with a @c uform_py_call_t. */
    size_t call_bytes;

    /** Hands the call to the library, with the thread state detached around it. */
    void (*submit)(void *call);

    /** Builds the answer of a call whose outcome succeeded, over the arena it claimed. */
    PyObject *(*answer)(void *call);

    /** Lets go of whatever the fill gathered and the answer did not take. */
    void (*release)(void *call);

    /** Visits the callables and reports the call holds beyond its header, for the collector; NULL when none. */
    int (*traverse)(void *call, visitproc visit, void *arg);
} uform_py_verb_t;

/** A deferred call owning everything the library touches until it is final, the object being the allocation, so
 *  nothing it lends ever moves. */
typedef struct Deferred {

    /** The object header, sized in bytes of the verb's call. */
    PyVarObject ob_base;

    /** How the call answers and lets go; NULL once it let go of its inputs. */
    uform_py_verb_t const *verb;

    /** The receiver the call was made through, held so its handle and executor outlive the call. */
    PyObject *origin;

    /** The answer, once the call finished without raising. */
    PyObject *answer;

    /** The exception, once the call finished raising it. */
    PyObject *raised;

    /** The verb's call, led by a @c uform_py_call_t, which the outcome lives in. */
    uint64_t call[];
} Deferred;

/**
 *  @brief Opens the call of @p verb: the zeroed @p stack for a blocking receiver, or the storage of a new
 *      @c Deferred holding @p receiver for a deferred one.
 *  @return The call to fill, or NULL with an exception set.
 */
void *uform_py_call_open(uform_py_receiver_t const *receiver, uform_py_verb_t const *verb, void *stack);

/** Runs a filled @p call: blocks and answers for a blocking receiver, or submits and answers its @c Deferred. */
PyObject *uform_py_call_run(uform_py_receiver_t const *receiver, uform_py_verb_t const *verb, void *call);

/** Lets go of a call whose fill failed, keeping the exception set; always returns NULL. */
PyObject *uform_py_call_discard(uform_py_receiver_t const *receiver, uform_py_verb_t const *verb, void *call);

/** Lets go of the arena claim and the latched exception every call header holds. */
void uform_py_call_release(void *call);

/** The answer of a verb with nothing to report. */
PyObject *uform_py_call_answer_none(void *call);

/**
 *  @brief Reads an optional callable into @p callback, latching its failures into @p call.
 *  @return 0 on success, -1 with a @c TypeError set when @p object is neither callable nor None.
 */
int uform_py_callback_from_object(PyObject *object, char const *field, uform_py_call_t *call,
                                  uform_py_callback_t *callback);

/** The progress to register for @p callback and @p steps; its trampoline is NULL when the caller passed no
 *  handler. */
uform_progress_t uform_py_progress_registered(uform_py_callback_t *callback, size_t steps);

#pragma endregion Calls

#pragma region Shared Helpers

/**
 *  @brief Raises the class @p outcome 's status maps to when it carries a failure.
 *  @return -1 with an exception set when @p outcome failed, 0 otherwise.
 */
int uform_py_raise_on_outcome(uform_outcome_t const *outcome);

/** Raises the @c Error subclass @p status maps to, with a formatted message, its @c status a @c Status member;
 *  always returns -1. */
int uform_py_raise(uform_status_t status, char const *format, ...);

/** Readies and registers @c ErrorType and one subclass per failing status on @p module; returns 0 or -1. */
int uform_py_register_error_types(PyObject *module);

/**
 *  @brief Stores @p value under @p key in @p dict and drops the caller's reference to it, whether or not the
 *      store succeeds.
 *  @return -1 with an exception set when @p value is null or the store failed, 0 otherwise.
 */
int uform_py_dict_steal_item(PyObject *dict, char const *key, PyObject *value);

/** A new struct sequence of @p type taking every one of @p fields, or NULL dropping them all when any is NULL. */
PyObject *uform_py_struct_sequence_steal(PyTypeObject *type, PyObject **fields, Py_ssize_t count);

/**
 *  @brief Reads a @c Scalar member, leaving @p kind alone when @p object is NULL.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int uform_py_scalar_from_object(PyObject *object, char const *field, uform_scalar_t *kind);

/** Reads a non-negative count, leaving @p value alone when @p object is NULL; returns 0 or -1 with an exception
 *  set. */
int uform_py_size_from_object(PyObject *object, char const *field, size_t *value);

/**
 *  @brief Reads a sequence of NUMA node IDs or device ordinals into a caller-freed array.
 *  @param[out] ordinals Receives a @c PyMem_Free -able array; @c NULL when @p object is NULL or @c None.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int uform_py_ordinals_from_object(PyObject *object, char const *field, int32_t **ordinals, size_t *count);

/**
 *  @brief Borrows a filesystem path from a @c str, @c bytes, or @c os.PathLike.
 *  @param[out] decoded A new reference the caller must release once @p path is spent.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int uform_py_path_from_object(PyObject *object, char const *field, uform_bytes_view_t *path, PyObject **decoded);

/**
 *  @brief The handle of the @c Executor @p object, refusing anything else and a closed one.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int uform_py_executor_from_object(PyObject *object, uform_executor_t *executor);

/** Reads an optional `executor`: NULL or @c None runs on the calling thread. Returns 0 or -1 with an exception
 *  set. */
int uform_py_executor_optional(PyObject *object, uform_executor_t *executor);

/** What the elements of a tape are, and so which inputs gather into it. */
typedef enum uform_py_element_t {

    /** UTF-8 bytes: a `str` encodes, a bytes-like object is taken as it is. */
    uform_py_element_text_k,

    /** The bytes of an encoded file, from a bytes-like object. */
    uform_py_element_bytes_k,

    /** Mono `float32` samples, from a 1-D buffer of format 'f'. */
    uform_py_element_samples_k,

    /** Interleaved 8-bit RGB, from a C-contiguous `[height, width, 3]` buffer of format 'B'. */
    uform_py_element_pixels_k,
} uform_py_element_t;

/** A batch as the library reads it, gathered into one block the call owns, or lent from a pair of buffers. */
typedef struct uform_py_tape_t {

    /** What the library reads. */
    uform_tape_view_t view;

    /** Entries the tape holds, one fewer than its offsets. */
    size_t count;

    /** For pixels, the @c count heights followed by the @c count widths, inside @c gathered. */
    uint32_t *extents;

    /** The @c PyMem block holding the offsets, the extents and the elements; NULL when lent. */
    void *gathered;

    /** The lent buffers, zeroed when gathered. */
    Py_buffer data, offsets;
} uform_py_tape_t;

/**
 *  @brief Reads a batch of @p element entries: a sequence of separate objects, gathered with one copy, or, for
 *      every element but pixels, a @c (data, offsets) tuple whose offsets buffer holds @c size_t words, lent.
 *  @return 0 on success, -1 with an exception set otherwise; release @p tape either way.
 */
int uform_py_tape_from_object(PyObject *object, char const *field, uform_py_element_t element, uform_py_tape_t *tape);

/** Releases whatever @c uform_py_tape_from_object acquired; a zeroed tape is a no-op. */
void uform_py_tape_release(uform_py_tape_t *tape);

/**
 *  @brief Claims the @c Arena @p object for one call, rewinding it: every result over its previous outputs goes
 *      stale.
 *  @param[out] arena A new reference to the claimed @c Arena; hand it to @c uform_py_arena_unclaim.
 *  @return 0 on success, -1 with an exception set when @p object is no @c Arena, serves a call, or still exports.
 */
int uform_py_arena_claim(PyObject *object, PyObject **arena);

/** Ends the claim @c uform_py_arena_claim took, and drops its reference. */
void uform_py_arena_unclaim(PyObject *arena);

/**
 *  @brief An @c ArenaBuffer over the @p count rows a call wrote packed into the @c Arena @p arena it claimed.
 *
 *  Scalars a @c memoryview can format leave as a @c (count, columns) matrix, and the rest - @c bf16 and the
 *  mini-floats among them - as a @c (count, stride) run of @c 'B'.
 *
 *  @param[in] type @c ArenaBufferType or a subtype.
 *  @return A new reference, or NULL with an exception set.
 */
PyObject *uform_py_rows_answer(PyTypeObject *type, PyObject *arena, uform_rows_view_t const *rows, size_t count);

/**
 *  @brief A @c RowsTape over the @p tokens rows of @p states a call wrote into @p arena, its @c offsets an
 *      @c ArenaBuffer of the @p count + 1 @c 'N' words beside them.
 *  @return A new reference, or NULL with an exception set.
 */
PyObject *uform_py_rows_tape_answer(PyObject *arena, uform_rows_tape_view_t const *states, size_t count, size_t tokens);

extern char const doc_version[];
extern char const doc_capabilities_compiled[];
extern char const doc_capabilities_available[];

PyObject *api_version(PyObject *self, PyObject *unused);
PyObject *api_capabilities_compiled(PyObject *self, PyObject *unused);
PyObject *api_capabilities_available(PyObject *self, PyObject *unused);

#pragma endregion Shared Helpers

#ifdef __cplusplus
}
#endif

#endif // UNUM_UFORM_PYTHON_LIB_H
