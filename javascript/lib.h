/**
 *  @file javascript/lib.h
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Declarations shared by every UForm Node-API translation unit.
 *
 *  One addon, one file per header: @c types.c owns the vocabulary every verb shares, @c model.c the model
 *  lifecycle, @c encode.c the forward passes, @c train.c the mesh and the trainer, and @c lib.c the module glue.
 *
 *  @section javascript_threading Threading
 *
 *  Every verb is a batch and the library fans it out itself, so this layer spawns nothing: it borrows the
 *  caller's @c TypedArray storage, hands the library one tape, and views the outputs it lands in the caller's
 *  @c Arena without copying them. A verb handed no executor runs on the calling thread, as the C interface's
 *  @c NULL executor does.
 *
 *  Every verb takes the C @c completion beside its executor. A blocking call answers its result; a deferred one
 *  answers a @c CompiledDeferred holding the outcome and strong references to every JavaScript value the library
 *  borrows until the call is final. Its @c wait blocks the calling thread in @c uform_outcome_wait, and its
 *  @c settled hands that wait to the libuv pool through @c napi_async_work, the one thread this layer ever
 *  borrows, and only while a caller awaits.
 *
 *  A JavaScript function runs only on the thread owning its context, which no executor's thread is, so a
 *  progress handler is taken only by a call without an executor, where the library calls it in place.
 */
#ifndef UNUM_UFORM_JAVASCRIPT_LIB_H
#define UNUM_UFORM_JAVASCRIPT_LIB_H

#include <stdbool.h> // `bool`
#include <stddef.h>  // `size_t`
#include <stdint.h>  // `uint32_t`

#include <node_api.h>

#include "uform/uform.h"

#pragma region Vocabulary

/** Compile-time constants of the binding layer. */
enum {

    /** Bytes a closed-set name may occupy, terminator included. */
    uform_js_name_capacity_k = 64,

    /** Bytes a formatted diagnostic may occupy, terminator included. */
    uform_js_message_capacity_k = 256,

    /** Heap blocks one call may own, freed together once it is final: the offsets it widened to @c size_t. */
    uform_js_owned_k = 5,
};

/** The surfaces an encode call keeps, as the @c outputs flags of the TypeScript side spell them. */
typedef enum uform_js_outputs_t {

    /** One pooled embedding per item. */
    uform_js_outputs_embeddings_k = 1,

    /** The final-norm states, one row per token kept. */
    uform_js_outputs_states_k = 2,

    /** The token head's unit vectors, one row per token kept. */
    uform_js_outputs_token_vectors_k = 4,
} uform_js_outputs_t;

/** Where an executor computes, which every arena it makes inherits; device memory is never viewed. */
typedef enum uform_js_residency_t {

    /** Host threads over host pages. */
    uform_js_residency_ram_k = 0,

    /** Device streams over device memory. */
    uform_js_residency_vram_k = 1,
} uform_js_residency_t;

/** Everything the addon owns per environment: the constructors its verbs type-check against, reached through
 *  @c napi_get_instance_data rather than a namespace-scope global. */
typedef struct uform_js_addon_t {

    /** The @c Executor constructor, kept alive for @c napi_instanceof. */
    napi_ref executor_class;

    /** The @c Arena constructor, kept alive for @c napi_instanceof. */
    napi_ref arena_class;

    /** The @c CompiledModel constructor, which every load wraps its handle in. */
    napi_ref model_class;

    /** The @c CompiledMesh constructor, which a bootstrap wraps its handle in. */
    napi_ref mesh_class;

    /** The @c CompiledTrainer constructor, which a trainer's creation wraps its handle in. */
    napi_ref trainer_class;

    /** The @c CompiledDeferred constructor, which every deferred submission instantiates. */
    napi_ref deferred_class;

    /** The token only this addon holds, which a direct construction from JavaScript lacks. */
    napi_ref seal;
} uform_js_addon_t;

struct uform_js_call_t;

/** The threads or streams a verb runs on, owned apart from any model so one executor serves many. */
typedef struct uform_js_executor_t {

    /** The library handle, freed by @c free or when the JavaScript wrapper is collected. */
    uform_executor_t handle;

    /** Where it computes, which every arena it makes inherits. */
    uform_js_residency_t residency;

    /** Live GPU arenas it made, each of which must be freed before it is. */
    size_t arenas;
} uform_js_executor_t;

/** Memory every output whose length scales with a call lands in, and the views handed out over it. */
typedef struct uform_js_arena_t {

    /** The library handle, made with the wrapper and @c NULL once freed. */
    uform_arena_t handle;

    /** The residency of the executor that made it. */
    uform_js_residency_t residency;

    /** The executor a GPU arena grows, fills and frees on, held until the arena is freed; @c NULL for host
     *  pages, which may outlive theirs. */
    uform_js_executor_t *executor;

    /** The strong reference keeping @c executor's wrapper alive alongside it. */
    napi_ref executor_reference;

    /** The last call the library took naming it, compared and never read; @c NULL once freed. */
    struct uform_js_call_t const *holder;

    /** Weak references to every @c ArrayBuffer viewing the memory since the last rewind. */
    napi_ref *views;

    /** References held in @c views. */
    size_t views_count;

    /** References @c views has room for. */
    size_t views_capacity;
} uform_js_arena_t;

/** An owned model handle, one per @c CompiledModel instance. */
typedef struct uform_js_model_t {

    /** The library handle, freed by @c free or the collector. */
    uform_model_t handle;
} uform_js_model_t;

/** An owned mesh handle, one per @c CompiledMesh instance. */
typedef struct uform_js_mesh_t {

    /** The library handle, freed by @c free or the collector. */
    uform_mesh_t handle;
} uform_js_mesh_t;

/** An owned trainer handle, one per @c CompiledTrainer instance. */
typedef struct uform_js_trainer_t {

    /** The library handle, freed by @c free or the collector. */
    uform_trainer_t handle;

    /** The strong reference keeping the borrowed mesh's wrapper alive until the trainer is freed; @c NULL
     *  for a run without peers. */
    napi_ref mesh_reference;
} uform_js_trainer_t;

/** The JavaScript values a call reads or writes; a deferred call holds every one until it is final. */
typedef enum uform_js_slot_t {

    /** The handle the verb runs on. */
    uform_js_slot_receiver_k = 0,

    /** The @c Executor it runs on, absent for the calling thread. */
    uform_js_slot_executor_k,

    /** The @c Arena its outputs land in. */
    uform_js_slot_arena_k,

    /** The first borrowed input, every later slot holding one more. */
    uform_js_slot_inputs_k,

    /** Slots in every call: the three above and up to thirteen borrowed inputs. */
    uform_js_slots_k = 16,
} uform_js_slot_t;

/** Builds a verb's answer from a successful call's outputs; @c NULL with an exception pending. */
typedef napi_value (*uform_js_finish_t)(napi_env env, struct uform_js_call_t *call, napi_value const *slots);

/**
 *  @brief One call submitted to the library: its outcome, how its answer is built, what it holds.
 *
 *  Every verb family extends it with the outputs its verbs land, keeping this header first, so one
 *  @c CompiledDeferred serves them all. A blocking call lives on the stack of its verb and holds nothing, since
 *  its arguments are reachable for as long as it runs; a deferred one lives on the heap and holds strong
 *  references to every slot until it is final and settled.
 */
typedef struct uform_js_call_t {

    /** Where the library reports the call, final once its status leaves @c deferred. */
    uform_outcome_t outcome;

    /** Builds the answer of a call that succeeded; @c NULL once the answer was built. */
    uform_js_finish_t finish;

    /** Strong references to the slots a deferred call borrows; @c NULL for an empty slot. */
    napi_ref held[uform_js_slots_k];

    /** Heap blocks the call reads, freed once it is final. */
    void *owned[uform_js_owned_k];

    /** The arena the call lands its outputs in; @c NULL for none. */
    uform_js_arena_t *arena;

    /** The environment a progress handler is called in, set only beside @c progress. */
    napi_env env;

    /** The @b optional progress handler of a call on the calling thread. */
    napi_value progress;

    /** The first exception the progress handler threw, thrown instead of the cancellation it caused. */
    napi_ref failure;
} uform_js_call_t;

#pragma endregion Vocabulary

#pragma region Argument Parsing

/** Throws @p message with the canonical name of @p status as its @c code; always returns false, so a caller
 *  returns straight through. */
bool uform_js_throw(napi_env env, uform_status_t status, char const *message);

/** Throws an @c invalid-argument error saying what @p field expected; always returns false. */
bool uform_js_throw_field(napi_env env, char const *field, char const *expectation);

/**
 *  @brief Throws the mapped JavaScript error when @p outcome carries a failure.
 *  @return @c true when the call succeeded, @c false with an exception pending otherwise.
 */
bool uform_js_succeeded(napi_env env, uform_outcome_t const *outcome);

/** Whether @p value is @c undefined or @c null, which every optional argument derives from. */
bool uform_js_value_absent(napi_env env, napi_value value);

/** Reads a named property of an @b optional options object; an absent one reads @c undefined. */
napi_value uform_js_property(napi_env env, napi_value object, char const *name);

/** Reads a non-negative integer from a number or a @c BigInt; absent arguments read as zero. */
bool uform_js_size_from_value(napi_env env, napi_value value, char const *field, size_t *result);

/** Reads a finite real; absent arguments read as zero, which every knob derives from. */
bool uform_js_double_from_value(napi_env env, napi_value value, char const *field, double *result);

/**
 *  @brief Copies a UTF-8 string into a @c malloc-ed block, so a deferred call can borrow it.
 *  @param[out] view The bytes, without the terminator the block also carries.
 *  @return The block the caller frees, or @c NULL with an exception pending.
 */
char *uform_js_string_from_value(napi_env env, napi_value value, char const *field, uform_bytes_view_t *view);

/** Reads the C @c completion every verb takes beside its executor. */
bool uform_js_completion_from_value(napi_env env, napi_value value, uform_completion_t *result);

/** Reads a scalar kind from the code @c uform_js_scalars published for its canonical spelling. */
bool uform_js_scalar_from_value(napi_env env, napi_value value, char const *field, uform_scalar_t *result);

/** Maps every scalar kind's canonical spelling to its code once, through the C name tables. */
napi_value uform_js_scalars(napi_env env);

/** Reads an @b optional CPU class pin through the C name tables; absent derives the CPU's own. */
bool uform_js_cpu_class_from_value(napi_env env, napi_value value, uform_cpu_class_t *result);

/** Reads an @b optional GPU class pin by its C name; absent derives the devices' own. */
bool uform_js_gpu_class_from_value(napi_env env, napi_value value, uform_gpu_class_t *result);

/** Wraps a class's canonical spelling, or answers @c null for the unknown class. */
napi_value uform_js_class_name(napi_env env, char const *name);

/** Maps every modality's canonical spelling to its code once, through the C name tables. */
napi_value uform_js_modalities(napi_env env);
napi_value uform_js_objectives(napi_env env);
napi_value uform_js_negatives(napi_env env);

/** Bytes a row of @p columns values of @p scalar takes, sub-byte packing included; else zero. */
size_t uform_js_row_bytes(uform_scalar_t scalar, size_t columns);

/**
 *  @brief Reads NUMA node IDs or device ordinals from an array of numbers.
 *  @param[out] ordinals Receives a @c free -able array; @c NULL when the value is absent or empty.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool uform_js_ordinals_from_value(napi_env env, napi_value value, char const *field, int32_t **ordinals, size_t *count);

/**
 *  @brief Borrows the bytes of a numeric @c TypedArray, whatever its element type.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool uform_js_bytes_from_value(napi_env env, napi_value value, char const *field, void **data, size_t *bytes);

/**
 *  @brief Borrows a @c TypedArray of exactly @p type, reporting its elements.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool uform_js_array_from_value(napi_env env, napi_value value, char const *field, napi_typedarray_type type,
                               void **data, size_t *elements);

/**
 *  @brief Reads a ragged batch: its elements in @p data_value and @c count + 1 offsets in a @c Uint32Array,
 *      widened into a @c size_t block the call owns.
 *  @param[in] element_type The @c TypedArray the elements arrive in.
 *  @param[out] count The entries the tape holds, which every batch verb takes as its @c count.
 *  @param[out] tape The view the library reads; its offsets live in the block @c call owns.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool uform_js_tape_from_value(napi_env env, napi_value data_value, napi_value offsets_value, char const *field,
                              napi_typedarray_type element_type, uform_js_call_t *call, size_t *count,
                              uform_tape_view_t *tape);

/**
 *  @brief Widens a @c Uint32Array of offsets into a @c size_t block @p call owns.
 *  @param[out] offsets The widened run, @c NULL when the value is absent.
 *  @param[out] count One less than the offsets read, the entries they bound.
 */
bool uform_js_offsets_from_value(napi_env env, napi_value value, char const *field, uform_js_call_t *call,
                                 size_t const **offsets, size_t *count);

/** Borrows an @b optional @c Uint32Array of exactly @p count entries, one per item; @c NULL when
 *  absent. */
bool uform_js_per_item_from_value(napi_env env, napi_value value, char const *field, size_t count,
                                  uint32_t const **result);

/**
 *  @brief Reads an @b optional progress handler, refused beside an executor, whose threads never own a
 *      JavaScript context.
 *  @param[out] progress The C handler and its state, pointing at @p call; zeroed when absent.
 */
bool uform_js_progress_from_value(napi_env env, napi_value value, uform_executor_t executor, uform_js_call_t *call,
                                  uform_progress_t *progress);

/** Wraps a count the library reported, once its error has been checked. */
napi_value uform_js_number(napi_env env, size_t value);

/** Wraps a static null-terminated string the library owns. */
napi_value uform_js_static_string(napi_env env, char const *text);

/** The @c undefined every void verb hands back. */
napi_value uform_js_undefined(napi_env env);

/** Builds an object of @p count named values, or answers @c NULL with an exception pending. */
napi_value uform_js_object(napi_env env, size_t count, char const *const *names, napi_value const *values);

/** Reads up to @p wanted arguments of a function; missing ones read as @c undefined. */
bool uform_js_arguments(napi_env env, napi_callback_info info, size_t wanted, napi_value *arguments, napi_value *self);

#pragma endregion Argument Parsing

#pragma region Classes and Calls

/** The addon state holding every class constructor; @c NULL with an exception pending. */
uform_js_addon_t *uform_js_addon(napi_env env);

/** Defines the @c Executor class; the module glue keeps the reference. */
napi_value uform_js_executor_define(napi_env env);

/** Unwraps the @c executor argument every verb takes; an absent one reads as @c NULL. */
bool uform_js_executor_from_value(napi_env env, napi_value value, uform_executor_t *handle);

/** Defines the @c Arena class; the module glue keeps the reference. */
napi_value uform_js_arena_define(napi_env env);

/**
 *  @brief Unwraps the @c arena a verb's outputs land in for @p call.
 *
 *  A freed arena is refused; one a deferred call still writes is the library's to refuse with
 *  @c uform_status_contended_k. The views of the previous call are detached once the library took this one,
 *  in @c uform_js_call_answer.
 */
bool uform_js_arena_from_value(napi_env env, napi_value value, uform_js_call_t *call, uform_arena_t *handle);

/**
 *  @brief Views @p bytes an arena holds as a @c TypedArray of @p type, without copying.
 *
 *  The backing @c ArrayBuffer references @p arena, so the memory outlives every view of it, and the arena
 *  remembers the buffer, empty ones included, so its next call or its @c free detaches the view rather than
 *  leaving it dangling. A VRAM arena holds device memory, which is never viewed as host memory, so its views
 *  read as @c null.
 *
 *  @return The view, or @c NULL with an exception pending.
 */
napi_value uform_js_arena_view(napi_env env, napi_value arena, napi_typedarray_type type, void const *data,
                               size_t bytes, size_t elements);

/** Views @p count rows a call landed inside @p arena as @c {data, rows, columns, stride, scalar}. */
napi_value uform_js_rows_object(napi_env env, napi_value arena, uform_rows_view_t const *rows, size_t count);

/** Views @p count + 1 @c size_t offsets a call landed inside @p arena, as a @c BigUint64Array on
 *  64-bit hosts. */
napi_value uform_js_offsets_view(napi_env env, napi_value arena, size_t const *offsets, size_t count);

/** Defines the @c CompiledDeferred class; the module glue keeps the reference. */
napi_value uform_js_deferred_define(napi_env env);

/** Unwraps an instance of the class @p reference holds, or throws naming @p field and answers @c NULL. */
void *uform_js_unwrap_instance(napi_env env, napi_value value, napi_ref reference, char const *field,
                               char const *expectation);

/** Reads the native state behind the receiver of a method, and @p wanted arguments beside it. */
void *uform_js_self(napi_env env, napi_callback_info info, size_t wanted, napi_value *arguments, napi_value *self);

/** The token constructors check, so only this addon instantiates its own classes. */
bool uform_js_sealed(napi_env env, napi_callback_info info, napi_value *self);

/** Instantiates one of the addon's classes, handing its constructor the seal. */
napi_value uform_js_instantiate(napi_env env, napi_ref constructor);

/** Defines a class whose constructor only accepts the seal, and whose instances get a @c [Symbol.dispose]
 *  calling their @c free. */
napi_value uform_js_class_define(napi_env env, char const *name, size_t count,
                                 napi_property_descriptor const *properties);

/**
 *  @brief Opens a call: @p local when blocking, or a zeroed heap block of @p bytes when deferred.
 *  @return The call, or @c NULL with an exception pending.
 */
uform_js_call_t *uform_js_call_open(napi_env env, uform_completion_t completion, uform_js_call_t *local, size_t bytes);

/**
 *  @brief Readies a deferred call: holds @p slots and wraps the call in a @c CompiledDeferred.
 *  @param[out] deferred The wrapper the verb answers once submitted; untouched for a blocking call.
 *  @return @c true on success, @c false with an exception pending and the call released otherwise.
 *
 *  Runs before the submission, so a failure here leaves nothing in flight.
 */
bool uform_js_call_hold(napi_env env, uform_js_call_t *call, uform_js_call_t const *local, napi_value const *slots,
                        napi_value *deferred);

/**
 *  @brief Answers a submitted call: a blocking one's finished result, a deferred one's wrapper.
 *
 *  Unless the library refused it as contended, the call rewound its arena, so the views of the call before it
 *  are detached here and the arena remembers this one as its holder.
 */
napi_value uform_js_call_answer(napi_env env, uform_js_call_t *call, uform_js_call_t const *local,
                                napi_value const *slots, napi_value deferred);

/** Releases a call that never reached the library, after its arguments failed to parse. */
void uform_js_call_abandon(uform_js_call_t *call, uform_js_call_t const *local);

/** Opens the call every instance verb runs through, reading the executor and the completion from the first
 *  two arguments. */
uform_js_call_t *uform_js_call_open_on(napi_env env, napi_value const *arguments, uform_js_call_t *local, size_t bytes,
                                       uform_executor_t *executor, uform_completion_t *completion);

/** Defines the @c CompiledModel class, its lifecycle verbs and its encode verbs. */
napi_value uform_js_model_define(napi_env env);

/** Appends the encode verbs to the @c CompiledModel properties @c model.c defines. */
napi_property_descriptor const *uform_js_encode_properties(size_t *count);

/** Defines the @c CompiledMesh class and its bootstrap. */
napi_value uform_js_mesh_define(napi_env env);

/** Defines the @c CompiledTrainer class, its creation and its verbs. */
napi_value uform_js_trainer_define(napi_env env);

/** Answers with the library's semantic version. */
napi_value uform_js_version(napi_env env, napi_callback_info info);

/** Answers with the kernel families this build carries, whatever the machine offers. */
napi_value uform_js_capabilities_compiled(napi_env env, napi_callback_info info);

/** Answers with the kernel families this machine can run, and dispatch picks from. */
napi_value uform_js_capabilities_available(napi_env env, napi_callback_info info);

/** Answers with every GPU this process can drive, in the order VRAM placements number them. */
napi_value uform_js_gpu_devices(napi_env env, napi_callback_info info);

#pragma endregion Classes and Calls

#endif // UNUM_UFORM_JAVASCRIPT_LIB_H
