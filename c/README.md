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
Statuses, scalars, rows, tapes, progress, arenas and the executor contract mirror USearch's `types.h` value for value, so embeddings and failures cross between the two libraries unchanged.
`find_package(uform)` reads an install as `uform::static`, `uform::shared` and `uform::header`.
An install or the SDK archive also answers `pkg-config --libs uform` with the shared library, which exports the `uform_*` functions alone.

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
Submission copies the structure, while everything it points to stays borrowed until final.
Calls on one executor run in submission order, so a caller orders dependent calls by waiting on the first.
A submission finding the executor's deferred table full, sized by `deferred_limit` and 64 entries by default, finishes `uform_status_contended_k`.
A call stops early only when its `uform_progress_t` handler returns `uform_cancel_k`, and then finishes `uform_status_cancelled_k`.

```c
uform_outcome_t outcomes[4] = {0};
for (size_t shard = 0; shard != 4; ++shard) {
    encodes[shard].outcome = &outcomes[shard];
    encodes[shard].completion = uform_completion_deferred_k;
    uform_image_encode(&encodes[shard]);
}
for (size_t shard = 0; shard != 4; ++shard) uform_outcome_wait(&outcomes[shard]);
```

## Arenas

Callers lend memory, never a way to allocate it, and every byte has one owner that picks where it lives once:

- Weights and their packed copies belong to the model, placed by its load's residency.
- Weights, gradients and optimizer moments under training belong to the trainer, on its executor's devices.
- Activations, scratch and staging belong to the executor, grown to a high-water mark.

Every output whose length scales with the call lands in an arena, while fixed-size reports such as stats and descriptions stay caller fields.
`uform_arena_init` places an arena where its executor computes, host pages or the memory of its GPU, and allocates nothing until a call writes into it.
A call rewinds the arena it is given, invalidating the previous call's outputs, and grows it only when the new outputs outsize everything it held, so a steady loop allocates once.

- A `NULL` arena, or inputs inside the arena the call rewinds, report `uform_status_invalid_argument_k`.
- An arena serves one call at a time, and a second call naming it finishes `uform_status_contended_k`.
- A call on another residency or device than the arena's reports `uform_status_device_k`.
- A GPU arena grows on the first stream of the executor that made it, so it serves only that executor and is freed before it.

```c
uform_arena_t arena = NULL;
uform_arena_init(&(uform_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
for (size_t batch = 0; batch != batches_count; ++batch) {
    encode.texts = texts[batch], encode.arena = arena, encode.embeddings = &embeddings;
    uform_text_encode(&encode);
    if (encode.outcome->status != uform_status_success_k) break;
    consume(embeddings, encode.count); // valid until the next call given `arena`
}
uform_arena_free(arena);
```

## Building

CMake builds `uform_static` and `uform_shared` from the objects `src/` compiles, and the archive carries the NumKong, StringZilla and yyjson objects it needs, naming only the system codecs to its consumers:

```sh
cmake --preset library
cmake --build --preset library
```

The presets choose the residencies a build carries:

- `library` builds the engine for the host alone.
- `cuda_rocm_library` adds the NVIDIA and AMD residencies, the trainer and the mesh, with Clang hosting both vendors' compilers.
- `linux_cuda_rocm` adds both residencies without the trainer, and launches UForm's own kernels through drivers opened at first use.
- `wasm32_emscripten` and `wasm32_wasi` build for WebAssembly, which takes decoded media only.
- `stub_library` builds `src_stub/`, a stand-in answering the same verbs over the C++ standard library and ForkUnion.

The stand-in's archives carry the engine's file names in a `stub/` directory of their own, so a consumer swaps implementations with one `-L`.
`c/test.c` runs against both, with the tiny checkpoint `test/tiny_checkpoint.cpp` writes when the suites are built.

## Classes and Devices

Dispatch picks one tuned variant per CPU or GPU family, named by `uform_cpu_class_t` and `uform_gpu_class_t`, whose enumerators and ordinals are USearch's.
`uform_ram_options_t::cpu_class` and `uform_vram_options_t::gpu_class` pin one, and the `unknown` zero derives the hardware's own:

- A CPU pin dispatches as if the CPU were of that class, taking every rung whose features it holds, and a class the CPU cannot run reports `uform_status_unsupported_k`.
- A GPU pin dispatches every named device as that class, and a device that cannot run it reports `uform_status_unsupported_k`.
- Unpinned devices of one VRAM executor must share a class, or the call reports `uform_status_config_mismatch_k`.

`uform_gpu_devices` lists every GPU the process can drive, NVIDIA devices first, then AMD, each as a `uform_gpu_device_t` of its vendor, vendor ordinal, class, compute units, memory and name.
A device's position in that listing is the ordinal `uform_vram_options_t::devices` names it by, and a machine without GPUs lists none and succeeds.
`uform_cpu_class_name` and `uform_gpu_class_name` spell a class, like `intel-haswell` or `nvidia-hopper`, and the `_from_name` verbs read one back.

## Residencies

Residency is in the verb name, as in USearch and UStore, so a residency a build lacks has no verb to call:

- __RAM__: `uform_model_load_ram` and `uform_model_load_ram_buffer` place the weights in host memory, on the NUMA nodes `uform_ram_options_t` names, and every build carries them.
- __NVIDIA__: `uform_model_load_vram` and `uform_model_load_vram_buffer` place them on the CUDA devices `uform_vram_options_t` names, in builds with `UFORM_CUDA_ARCH`.
- __AMD__: the same VRAM verbs place them on ROCm devices, in builds with `UFORM_ROCM_ARCH`.

On NVIDIA, text and JPEG images never leave the device; container demux, PNG and WebP, audio, and every modality on AMD decode on host threads the executor spawns beside its streams.
`uform_capabilities_compiled` and `uform_capabilities_available` report what the binary was built with and what the machine offers.

## Encoding

`uform_model_describe` reports the model's `dimensions`, `context`, `quantization`, memory usage and the CPU or GPU class it dispatched to.
It also reports the `cpu_class` a RAM model was packed for or the `gpu_class` a VRAM model runs as, the `token_width` of its token head, and the `choice_temperature_groups` of its byte decoder, each zero when the checkpoint carries no such part.
Every encode verb runs the same tower, the modality's entry, the transformer blocks and the final norm, and keeps whichever surfaces the caller names from one forward:

- `embeddings` land one pooled row per item, past the attention pool and the projection head.
- `states` land one row per token each item kept, the surface rerankers and masked heads read.
- `token_vectors` land the token head's unit vectors, `token_width` wide, on the same offsets as the states, for late-interaction reranking.

A `NULL` output skips its surface, naming none is refused with `uform_status_invalid_argument_k`, and a model without a token head refuses `token_vectors` with `uform_status_unsupported_k`.
Every surface lands in the arena as `outputs_scalar`, and an item longer than the model's `context` is truncated and counted in the stats.
One verb per input form takes only the fields that form needs:

- `uform_text_encode` takes UTF-8 texts, a byte per token, since models are tokenizer-free.
- `uform_image_encode` takes JPEG, PNG or WebP files, and `uform_image_encode_pixels` interleaved 8-bit RGB.
- `uform_video_encode` takes MP4 or WebM clips.
- `uform_audio_encode` takes WAV, FLAC, MP3 or Opus files, and `uform_audio_encode_samples` mono samples at a rate.

```c
uform_rows_view_t embeddings = {0};
uform_rows_tape_view_t states = {0};
uform_text_encode(&(uform_text_encode_t){
    .model = model, .executor = executor, .outcome = &outcome, .count = n,
    .texts = {bytes, offsets, uform_scalar_u8_k}, .embeddings = &embeddings, .states = &states,
    .arena = arena, .outputs_scalar = uform_scalar_bf16_k});
```

## Calibrated Choices

`uform_text_choose` scores each text's legal choices, one byte each, through one forward of the byte decoder.
The call appends the mask byte to every text, and a text longer than the model's `context` loses its head, never its end.
Each item's probabilities are a softmax over its own choices alone, at the calibrated inverse temperature of its `temperature_groups` entry, and land in the arena as one `f32` tape in `choices` order.
A model whose `choice_temperature_groups` is zero carries no byte decoder and reports `uform_status_unsupported_k`.

```c
uform_tape_view_t probabilities = {0};
uform_text_choose(&(uform_text_choose_t){
    .model = model, .executor = executor, .outcome = &outcome, .count = n,
    .texts = {bytes, offsets, uform_scalar_u8_k}, .choices = {letters, letter_offsets, uform_scalar_u8_k},
    .probabilities = &probabilities, .arena = arena});
```

A checkpoint stores one inverse temperature per question type as `choices.inverse_temperatures`, and one without them chooses at an inverse temperature of one.
`uform_eval <checkpoint> --calibrate <choices.jsonl> --output <calibrated.safetensors>` fits them on held-out choices, prints each group's negative log-likelihood and the top-choice calibration error before and after, and writes the calibrated copy.

## Mesh and Trainer

Training is GPU-only.
`uform_mesh_bootstrap` joins this process to a run once, through files in a directory every host sees, and after it processes talk only over NVLink and InfiniBand.
The mesh is the process-wide half of the transport, which every trainer borrows; `uform_mesh_describe` reports the topology every process agreed on, its multimem tier, and the `gpu_class` every GPU of the run dispatches as, the executor's pin or the devices' own.
Every process bootstraps with the same count, multicast extent, directory, job and GPU class, or the call reports `uform_status_config_mismatch_k`.

`uform_trainer_init_vram` makes a trainer on a VRAM executor, owning one replica of the tower per device, its gradients, the Adafactor moments, the reconstruction heads and the learnable temperatures.
One `uform_trainer_step` is one optimizer step over this process's share of the global batch: forward, the losses, backward, the world gradient reduction, and the update.
Items arrive tagged with a `uform_modality_t`, grouped into bags that travel together, with attract, repel and ignore relations indexing the step's own items.
Beside the masked and pooled contrastive losses, a trainer relaxes two optional terms, each read when its weight is positive:

- `late_interaction` trains the token head with hard MaxSim scores between the items each step's `sides` names a query, a document or both.
- `choices` trains the byte decoder on the legal bytes each step's `choices` tape gives, the tape `uform_text_choose` reads, and their target probabilities in the `choice_targets` tape, shaped as the probabilities it lands, within the batch's `max_choices_per_item`.

`uform_pair_loss_form_t` picks how a contrastive term turns a row of scores into forces: `balanced`, `ranked` or SigLIP's `sigmoid`, where `derived` keeps sigmoid for pooled embeddings and ranked for token vectors.
The step's stats report every loss as its world mean, with `late_interaction_loss` and `choices_loss` beside the contrastive and masked terms, the items `truncated` at a row's width, and the segments `dropped` whole for want of room.
A training call that fails after a GPU recorded a fault, such as a peer missing a multimem barrier or an InfiniBand round past its deadline, carries that fault as its outcome message.
`uform_trainer_save` writes the safetensors checkpoint the model verbs load.
Every path, the mesh's `directory`, a `resume` checkpoint and a save's `path`, arrives as the operating system's own bytes, not necessarily UTF-8, and one holding a NUL byte is refused with `uform_status_invalid_argument_k`; the mesh's `job` is a UTF-8 name.

```c
uform_mesh_t mesh = NULL;
uform_mesh_bootstrap(&(uform_mesh_bootstrap_t){.mesh = &mesh, .executor = executor, .outcome = &outcome,
    .options = {.host_domain_index = node, .host_domain_count = nodes, .directory = {exchange, exchange_length},
                .job = {job, job_length}}});
uform_trainer_t trainer = NULL;
uform_trainer_init_vram(&(uform_trainer_init_vram_t){
    .trainer = &trainer, .executor = executor, .outcome = &outcome, .mesh = mesh, .depth = 12, .hidden = 512,
    .batch = {.max_rows_per_device = 64, .max_segments_per_device = 256, .max_columns_per_row = 1024,
              .max_bags_per_device = 128, .image = {.patch_side = 16}},
    .objective = uform_objective_joint_k, .schedule = {.steps = 100000}, .seed = 42});
```

## Rows and Tapes

A verb's `count` is its batch axis and nothing else.
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
