# UForm for C

The C ABI is the one versioned interface of UForm, and every binding is a layer over it.
It compiles as plain C11, holds no C++ type, and never logs.

## Headers

- `uform/uform.h`: the umbrella over the four headers below.
- `uform/types.h`: statuses and outcomes, scalars, rows, tapes, progress, CPU and GPU classes, devices, executors, arenas, the name tables and the version probes.
- `uform/model.h`: loading an encoder into RAM or VRAM, from a file or a buffer, describing it and freeing it.
- `uform/encode.h`: texts, files, pixels, clips and samples to embeddings, states and token vectors, and scored choices.
- `uform/train.h`: the mesh, trainers, their steps and checkpoints, on NVIDIA GPUs alone.

A caller only encoding may include `encode.h` alone, since each header pulls in `types.h`.
`find_package(uform)` reads an install as `uform::static`, `uform::shared` and `uform::header`.

## Quickstart

```c
#include <stdio.h>  // `printf`
#include <string.h> // `strlen`

#include <uform/uform.h>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    uform_outcome_t outcome = {0};
    uform_executor_t executor = NULL;
    uform_executor_for_ram(&(uform_executor_for_ram_t){.executor = &executor, .outcome = &outcome});
    if (outcome.status != uform_status_success_k) return printf("%s\n", outcome.message), 1;

    uform_model_t model = NULL;
    uform_model_load_ram(&(uform_model_load_ram_t){
        .model = &model, .executor = executor, .outcome = &outcome,
        .path = {(uint8_t const*)argv[1], strlen(argv[1])}, .quantization = uform_scalar_bf16_k});
    if (outcome.status != uform_status_success_k) return printf("%s\n", outcome.message), 1;

    uform_arena_t arena = NULL;
    uform_arena_init(&(uform_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});

    char const texts[] = "a cat on a matsunrise over the bay";
    size_t const offsets[] = {0, 14, 34};
    uform_rows_view_t embeddings = {0};
    uform_text_encode(&(uform_text_encode_t){
        .model = model, .executor = executor, .outcome = &outcome, .count = 2,
        .texts = {texts, offsets, uform_scalar_u8_k}, .embeddings = &embeddings, .arena = arena,
        .outputs_scalar = uform_scalar_f32_k});
    if (outcome.status == uform_status_success_k)
        printf("%zu embeddings of %zu dimensions\n", (size_t)2, embeddings.columns);

    uform_arena_free(arena);
    uform_model_free(model);
    uform_executor_free(executor);
    return outcome.status != uform_status_success_k;
}
```

## Argument Structures

Every verb that takes options, outputs or callbacks takes one structure named after it, as `uform_text_encode` takes a `uform_text_encode_t`.
Fields open in the same order everywhere: the handle, `executor`, `outcome` and `completion`, then inputs, then outputs and the `arena` they land in, then knobs, and last `progress` and `stats`.
Structures carry no size or version field, so initialize them by designator, which zeroes every field left out, and expect the major version to change whenever one grows.
Name lookups and the `_free` family stay positional, and a `_free` of `NULL` is safe.

## Executors

An executor owns what a call runs on but a model does not: host threads or device streams, activation scratch, and the pinned buffer staging passes through.

- `uform_executor_for_ram` spawns host threads across the compute domains local to the memory domains of its `uform_ram_options_t`, 7/8 of their cores when `threads` is zero.
- `uform_executor_on_pool` lends UForm a ForkUnion `fu_pool_t` another library spawned, so UForm, USearch and UStore in one process share one set of threads; the pool must outlive the executor.
- `uform_executor_for_vram` opens one stream on every device of its `uform_vram_options_t`, named by their position in `uform_gpu_devices`, beside host threads for the decoding the devices cannot do.
- `uform_executor_drain` waits for every call submitted before it and sets the `uform_mood_t` the executor adopts when idle, parked or spinning.
- `uform_executor_free` drains, releases what the executor owns and returns what it borrowed.

One executor serves any number of models of its residency, and a call pairing a model with an executor of another residency reports `uform_status_device_k`.
A `NULL` executor runs the call on the calling thread, final when it returns.
`cpu_class` and `gpu_class` in the options pin the class a call dispatches as, and `uform_status_unsupported_k` answers a class the machine cannot run.

## Outcomes and Completions

Every call stores its result in the `uform_outcome_t` it names: a status, and a static message that is `NULL` exactly on success.
`uform_completion_blocking_k`, the default, returns with the status final and every output readable.
`uform_completion_deferred_k` returns once submitted, and the status reads `uform_status_deferred_k` until the call is final.
`uform_outcome_status` polls with acquire ordering and `uform_outcome_wait` blocks until final, and these are the only ways to read a status another thread may still store.
Calls on one executor run in submission order, so a caller orders dependent calls by waiting on the first.
A submission finding the executor's deferred table full, sized by `deferred_limit` and 64 entries by default, finishes `uform_status_contended_k`.
A call stops early only when its `uform_progress_t` handler returns `uform_cancel_k`, and then finishes `uform_status_cancelled_k`.

## Arenas

Every output whose length scales with the call lands in an arena, while fixed-size reports such as stats and descriptions stay caller fields.
`uform_arena_init` places an arena where its executor computes, host pages or the memory of its GPU, and allocates nothing until a call writes into it.
A call rewinds the arena it is given, invalidating the previous call's outputs, and grows it only when the new outputs outsize everything it held, so a steady loop allocates once.

- A `NULL` arena, or inputs inside the arena the call rewinds, report `uform_status_invalid_argument_k`.
- An arena serves one call at a time, and a second call naming it finishes `uform_status_contended_k`.
- A call on another residency or device than the arena's reports `uform_status_device_k`.
- A GPU arena grows on the first stream of the executor that made it, so it serves only that executor and is freed before it.

## Rows and Tapes

Outputs are views the call writes packed into its arena, so none carries a capacity for the caller to pick.
`uform_rows_view_t` holds one row per item as `data`, `stride`, `columns` and `scalar`, field for field USearch's rows, so pooled embeddings feed an index as they are.
Ragged inputs are `uform_tape_view_t` tapes laid out as Apache Arrow lays them: elements end to end in `data`, and `count + 1` offsets in host memory around them.
Ragged outputs, such as per-token states, are `uform_rows_tape_view_t`: packed rows and one more offset than items.
On a VRAM executor every output is device memory, which the caller copies out itself or hands to a USearch VRAM index as it is.

## Statuses

Statuses are numbered as USearch numbers them, so a failure keeps its identity across the two libraries:

- `uform_status_out_of_memory_k`: host, pinned or device memory ran out, or the scratch ceiling is too low.
- `uform_status_shape_mismatch_k` and `uform_status_unsupported_layout_k`: counts, shapes or strides the model cannot read.
- `uform_status_malformed_k` and `uform_status_io_k`: bytes that do not decode, or a file that does not open, read or write.
- `uform_status_device_k`: the driver refused, a device faulted, or an executor and a model or arena disagree on residency.
- `uform_status_unknown_kind_k`: a scalar, modality or name this build does not know.
- `uform_status_unsupported_k`: a verb, modality or class this model or build does not serve.
- `uform_status_invalid_argument_k` and `uform_status_config_mismatch_k`: an argument breaking the verb's contract, or a request contradicting a frozen configuration.
- `uform_status_contended_k` and `uform_status_cancelled_k`: a call refused rather than waited for, or stopped by its progress handler.
- `uform_status_internal_k`: a defect, reported rather than hung on.

`uform_status_name` spells a status as lowercase words joined by dashes and `uform_status_from_name` reads it back, as the other closed sets' `_name` and `_from_name` pairs do.
`uform_version`, `uform_capabilities_compiled` and `uform_capabilities_available` report the library this process linked and the hardware it found.
