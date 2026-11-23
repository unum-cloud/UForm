# UForm for C

The C ABI is the one versioned interface of UForm, and every binding is a layer over it.
It compiles as plain C11, holds no C++ type, and never logs.

## Installation

```sh
cmake --preset library -D UFORM_BUILD_TEST=ON   # uform_static and uform_shared for the host, and the suites
cmake --build --preset library
ctest --test-dir build_library                   # c/test.c and the host suites, on the tiny checkpoint
```

`cuda_rocm_library` adds the NVIDIA and AMD residencies, the trainer and the mesh; `wasm32_emscripten` and `wasm32_wasi` build for WebAssembly, which takes decoded media only; `stub_library` builds `src_stub/`, a stand-in answering every verb, whose archives carry the engine's names so one `-L` swaps them.
`find_package(uform)` reads an install as `uform::static`, `uform::shared`, `uform::abi` for the C headers alone and `uform::cpp` for the C++ wrappers, and `pkg-config --libs uform` names the shared library.
`uform/uform.h` includes `types.h`, `model.h`, `encode.h` and `train.h`, and a caller only encoding may include `encode.h` alone.
Statuses, scalars, rows, tapes, arenas and the executor contract mirror USearch's `types.h` value for value, so embeddings and failures cross between the two libraries unchanged.

## Quickstart

Encode two captions and a photo into one space, then pick the caption closest to the photo:

```c
#include <math.h>   // `sqrtf`
#include <stdio.h>  // `fopen`, `fread`, `printf`
#include <string.h> // `strlen`

#include <uform/uform.h>

/** The cosine of row `i` of `texts` and the one row of `images`; embeddings come unnormalized. */
static float cosine(uform_rows_view_t texts, size_t i, uform_rows_view_t images) {
    float const* text = (float const*)((char const*)texts.data + i * texts.stride);
    float const* image = (float const*)images.data;
    float dot = 0, texts_norm = 0, images_norm = 0;
    for (size_t k = 0; k != texts.columns; ++k)
        dot += text[k] * image[k], texts_norm += text[k] * text[k], images_norm += image[k] * image[k];
    return dot / sqrtf(texts_norm * images_norm);
}

int main(void) {
    uform_outcome_t outcome = {0};
    uform_executor_t executor = NULL;
    uform_model_t model = NULL;
    uform_arena_t texts_arena = NULL, images_arena = NULL; // a call rewinds its arena, so two outputs need two
    char const path[] = "model.safetensors";
    uform_executor_for_ram(&(uform_executor_for_ram_t){.executor = &executor, .outcome = &outcome});
    if (!outcome.status) uform_model_load_ram(&(uform_model_load_ram_t){.model = &model, .executor = executor,
        .outcome = &outcome, .path = {(uint8_t const*)path, strlen(path)}, .quantization = uform_scalar_bf16_k});
    if (!outcome.status)
        uform_arena_init(&(uform_arena_init_t){.arena = &texts_arena, .executor = executor, .outcome = &outcome});
    if (!outcome.status)
        uform_arena_init(&(uform_arena_init_t){.arena = &images_arena, .executor = executor, .outcome = &outcome});

    char const* captions[] = {"a cat on a mat", "a sunrise over the bay"};
    char const caption_bytes[] = "a cat on a mata sunrise over the bay";
    size_t const caption_offsets[] = {0, 14, 36};
    static unsigned char photo[1 << 20];
    FILE* file = fopen("cat.png", "rb");
    size_t const photo_offsets[] = {0, file ? fread(photo, 1, sizeof(photo), file) : 0};
    if (file) fclose(file);

    uform_rows_view_t texts = {0}, images = {0};
    if (!outcome.status) uform_text_encode(&(uform_text_encode_t){.model = model, .executor = executor,
        .outcome = &outcome, .count = 2, .texts = {caption_bytes, caption_offsets, uform_scalar_u8_k},
        .embeddings = &texts, .arena = texts_arena, .outputs_scalar = uform_scalar_f32_k});
    if (!outcome.status) uform_image_encode(&(uform_image_encode_t){.model = model, .executor = executor,
        .outcome = &outcome, .count = 1, .images = {photo, photo_offsets, uform_scalar_u8_k},
        .embeddings = &images, .arena = images_arena, .outputs_scalar = uform_scalar_f32_k});

    if (outcome.status) return printf("%s\n", outcome.message), 1;
    size_t const best = cosine(texts, 1, images) > cosine(texts, 0, images);
    printf("'%s' describes the photo best\n", captions[best]);

    uform_arena_free(images_arena), uform_arena_free(texts_arena);
    uform_model_free(model), uform_executor_free(executor);
    return 0;
}
```

Every call stores its status in the `uform_outcome_t` it names, `uform_status_success_k` being zero, so a chain of calls proceeds while the status stays zero.

## Receivers & Deferred Calls

Every call stores its result in the `uform_outcome_t` it names: a status, and a static message that is `NULL` exactly on success.
`uform_completion_blocking_k`, the default, returns with the status final and every output readable.
`uform_completion_deferred_k` returns once the executor's own thread took the call, and the status reads `uform_status_deferred_k` until the call is final.
`uform_outcome_status` polls with acquire ordering and `uform_outcome_wait` blocks until final, and these are the only ways to read a status another thread may still store.
Submission copies the structure, while everything it points to stays borrowed until final.
An executor runs one call at a time: a call finding another one holding its executor finishes `uform_status_contended_k` without running, so a caller orders dependent calls by waiting on the first.
A call lets go of its arena, its handles and its executor before it stores the final status, so whoever reads it final may reuse all three at once.
A call stops early only when its `uform_progress_t` handler returns `uform_cancel_k`, and then finishes `uform_status_cancelled_k`.

```c
uform_outcome_t outcomes[4] = {0};
for (size_t shard = 0; shard != 4; ++shard) { // one executor per shard, each running one call at a time
    encodes[shard].outcome = &outcomes[shard];
    encodes[shard].executor = executors[shard];
    encodes[shard].completion = uform_completion_deferred_k;
    uform_image_encode(&encodes[shard]);
}
for (size_t shard = 0; shard != 4; ++shard) uform_outcome_wait(&outcomes[shard]);
```

### Inclusive and Exclusive Executors

The threads of an executor either include the thread that calls it or work without it.
A caller-inclusive pool needs its caller as one of its threads, so a deferred call on it runs at submission and is final when it returns.
A caller-exclusive pool has threads of its own: a deferred call returns as soon as the pool's thread 0 takes it, and that thread runs the call, forks included, and stores the status.
`uform_executor_for_ram` and `uform_executor_for_vram` spawn caller-exclusive pools wherever the build has threads, and WebAssembly's single thread is inclusive.
Because a deferred call needs nobody to drive it, one thread can keep several executors busy, here across UForm, USearch and UStore, by only reading status words:

```c
text_encode.completion = uform_completion_deferred_k;   // UForm, on a VRAM executor
dense_search.completion = usearch_completion_deferred_k; // USearch, on a RAM executor
blobs_read.completion = ustore_completion_deferred_k;    // UStore, on an NVMe executor
uform_text_encode(&text_encode);
usearch_dense_search(&dense_search);
ustore_blobs_read(&blobs_read);
while (uform_outcome_status(text_encode.outcome) == uform_status_deferred_k ||
       usearch_outcome_status(dense_search.outcome) == usearch_status_deferred_k ||
       ustore_outcome_status(blobs_read.outcome) == ustore_status_deferred_k)
    prepare_next_batch();
```

## Executors

An executor owns what a call runs on but a model does not: host threads or device streams, activation scratch, and the pinned buffer staging passes through.

- `uform_executor_for_ram` creates an executor of host threads across the compute domains local to the memory domains of its `uform_ram_options_t`, 7/8 of their cores when `threads` is zero.
- `uform_executor_on_pool` lends UForm a ForkUnion `fu_pool_t` another library spawned, so UForm, USearch and UStore in one process share one set of threads; the pool must outlive the executor.
- `uform_executor_for_vram` opens one stream on every device of its `uform_vram_options_t`, named by their position in `uform_gpu_devices`, beside host threads for the decoding the devices cannot do.
- `uform_executor_drain` waits until no call holds the executor, then applies the `uform_mood_t` its threads keep until the next call: parked sleeps in the kernel, spinning keeps polling.
- `uform_executor_free` waits the same way, releases what the executor owns and returns what it borrowed.

```c
int32_t const nodes[] = {0}, devices[] = {0, 1};
uform_executor_t host = NULL, gpus = NULL;
uform_executor_for_ram(&(uform_executor_for_ram_t){.executor = &host, .outcome = &outcome, .threads = 16,
    .ram = {.memory_domains = nodes, .memory_domains_count = 1}});
uform_executor_for_vram(&(uform_executor_for_vram_t){.executor = &gpus, .outcome = &outcome,
    .vram = {.devices = devices, .devices_count = 2}}); // one stream per device, plus host decoders
uform_executor_drain(&(uform_executor_drain_t){.executor = gpus, .outcome = &outcome, .mood = uform_mood_parked_k});
uform_executor_free(gpus), uform_executor_free(host);
```

One executor serves any number of models of its residency, and a call pairing a model with an executor of another residency reports `uform_status_device_k`.
A `NULL` executor runs the call on the calling thread, final when it returns.
`cpu_class_ceiling` and `gpu_class_ceiling` in the options cap the class a call dispatches as, and the call runs the best class under the cap the hardware runs.

### Classes and Devices

Dispatch picks one tuned variant per CPU or GPU family, named by `uform_cpu_class_t` and `uform_gpu_class_t`, whose enumerators and fixed values are USearch's.
`uform_ram_options_t::cpu_class_ceiling` and `uform_vram_options_t::gpu_class_ceiling` cap one, and the `unknown` zero sets no cap:

- A CPU ceiling intersects with what the CPU runs: the newest compiled class whose features both hold runs, so a ceiling above the CPU runs as the CPU's own.
- A GPU ceiling runs every named device as the best class all of them load at or under it, and devices sharing no such class report `uform_status_unsupported_k`.
- Devices of one VRAM executor must share a vendor, or the call reports `uform_status_config_mismatch_k`.

`uform_gpu_devices` lists every GPU the process can drive, NVIDIA devices first, then AMD, then Apple, each as a `uform_gpu_device_t` of its vendor, vendor ordinal, class, compute units, memory and name.
A device's position in that listing is the ordinal `uform_vram_options_t::devices` names it by, and a machine without GPUs lists none and succeeds.
`uform_cpu_class_name` and `uform_gpu_class_name` spell a class, like `intel-haswell` or `nvidia-hopper`, and the `_from_name` verbs read one back.

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

### Rows and Tapes

A verb's `count` is its batch axis and nothing else.
Outputs are views the call writes packed into its arena, so none carries a capacity for the caller to pick.
`uform_rows_view_t` holds one row per item as `data`, `stride`, `columns` and `scalar`, field for field USearch's rows, so pooled embeddings feed an index as they are.
Ragged inputs are `uform_tape_view_t` tapes laid out as Apache Arrow lays them: elements end to end in `data`, and `count + 1` offsets in host memory around them.
Ragged outputs, such as per-token states, are `uform_rows_tape_view_t`: packed rows and one more offset than items.
On a VRAM executor every output is device memory, which the caller copies out itself or hands to a USearch VRAM index as it is.

## Encoding

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

## Choices

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

## Models

Residency is in the verb name, as in USearch and UStore, so a residency a build lacks has no verb to call:

- __RAM__: `uform_model_load_ram` and `uform_model_load_ram_buffer` place the weights in host memory, on the NUMA nodes `uform_ram_options_t` names, and every build carries them.
- __NVIDIA__: `uform_model_load_vram` and `uform_model_load_vram_buffer` place them on the CUDA devices `uform_vram_options_t` names, in builds with `UFORM_CUDA_ARCH`.
- __AMD__: the same VRAM verbs place them on ROCm devices, in builds with `UFORM_ROCM_ARCH`.

On NVIDIA, text and JPEG images never leave the device; container demux, PNG and WebP, audio, and every modality on AMD decode on host threads the executor spawns beside its streams.
`uform_capabilities_compiled` and `uform_capabilities_available` report what the binary was built with and what the machine offers.

`uform_model_describe` reports the model's `dimensions`, `context`, `quantization`, memory usage and the CPU or GPU class it dispatched to.
It also reports the `cpu_class` a RAM model was packed for or the `gpu_class` a VRAM model runs as, the `token_width` of its token head, and the `choice_temperature_groups` of its byte decoder, each zero when the checkpoint carries no such part.

```c
uform_model_t model = NULL;
int32_t const devices[] = {0};
uform_model_load_vram(&(uform_model_load_vram_t){.model = &model, .executor = gpus, .outcome = &outcome,
    .path = {(uint8_t const*)path, strlen(path)}, .vram = {.devices = devices, .devices_count = 1},
    .quantization = uform_scalar_e4m3_k}); // Float8 blocks, quantized at load
uform_model_description_t description = {0};
uform_model_describe(&(uform_model_describe_t){.model = model, .executor = gpus, .outcome = &outcome,
    .description = &description});
printf("%zu dimensions on %s\n", description.dimensions, uform_gpu_class_name(description.gpu_class));
```

## Training

Training is GPU-only.
`uform_mesh_bootstrap` joins this process to a run once, through files in a directory every host sees, and after it processes talk only over NVLink and InfiniBand.
The mesh is the process-wide half of the transport, which every trainer borrows; `uform_mesh_describe` reports the topology every process agreed on, its multimem tier, and the `gpu_class` every GPU of the run dispatches as, the best class its devices load under the executor's ceiling.
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

## Errors

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

```c
uform_text_encode(&encode);
if (outcome.status == uform_status_unsupported_k) // a token head this checkpoint lacks: retry without it
    encode.token_vectors = NULL, uform_text_encode(&encode);
if (outcome.status != uform_status_success_k)
    fprintf(stderr, "%s: %s\n", uform_status_name(outcome.status), outcome.message);
```

## Design

Every verb that takes options, outputs or callbacks takes one structure named after it, as `uform_text_encode` takes a `uform_text_encode_t`.
Fields open in the same order everywhere: the handle, `executor`, `outcome` and `completion`, then inputs, then outputs and the `arena` they land in, then knobs, and last `progress` and `stats`.
Structures carry no size or version field, so initialize them by designator, which zeroes every field left out, and expect the major version to change whenever one grows.
Name lookups and the `_free` family stay positional, and a `_free` of `NULL` is safe.
