=====
C ABI
=====

The C interface in ``c/uform/`` is what every binding calls, and what any other language reaches through its foreign-function layer.
``uform/uform.h`` includes the four headers, and each of them also stands alone:

- ``uform/types.h`` holds the shared vocabulary: statuses, scalars, rows and tapes, outcomes, executors, arenas and introspection.
- ``uform/model.h`` loads an encoder into RAM or VRAM, describes it, and frees it.
- ``uform/encode.h`` runs the forward passes, one verb per input form.
- ``uform/train.h`` joins processes into a mesh and runs optimizer steps on a trainer.

Statuses, scalars, rows, tapes, progress, arenas and the executor contract mirror USearch's ``types.h`` value for value, so embeddings and failures cross between the two libraries unchanged.

========
Building
========

CMake builds ``uform_static`` and ``uform_shared`` from the objects ``src/`` compiles, and the archive carries the NumKong, StringZilla and yyjson objects it needs, naming only the system codecs to its consumers:

.. code-block:: sh

    cmake --preset cpu_library
    cmake --build --preset cpu_library

The presets choose the residencies a build carries:

- ``cpu_library`` builds the engine for the host alone.
- ``nvidia_amd_library`` adds the NVIDIA and AMD residencies, the trainer and the mesh, with Clang hosting both vendors' compilers.
- ``wasm32_emscripten_release`` and ``wasm32_wasi_release`` build for WebAssembly, which takes decoded media only.
- ``stub_library`` builds ``src_stub/``, a stand-in answering the same verbs over the C++ standard library alone.

The stand-in's archives carry the engine's file names in a ``stub/`` directory of their own, so a consumer swaps implementations with one ``-L``.
``c/test.c`` runs against both, with the tiny checkpoint ``test/tiny_checkpoint.cpp`` writes when the suites are built.

=====
Verbs
=====

Every verb that takes options, outputs or callbacks takes one argument structure named after it, so ``uform_text_encode`` takes a ``uform_text_encode_t``.
Optional arguments are skipped by name and their order never matters, while name lookups and the ``_free`` family stay positional.
Every structure opens in the same order: the handle, ``executor``, ``outcome``, then ``completion``.
Inputs follow, then outputs and the ``arena`` they land in, then knobs, and last ``progress`` and ``stats``.

Structures carry no size or version field, so they grow source-compatibly only.
Initialize them by designator, which zeroes every field left out, and a zero field derives its documented default.
The library never logs.

=========================
Outcomes and Completions
=========================

Every call names its ``completion``, and blocking and deferred calls share one shape:

- ``uform_completion_blocking_k``, the zero default, returns with the final status stored and every output readable.
- ``uform_completion_deferred_k`` returns once submitted, and the call finishes on its executor.
- A ``NULL`` executor runs the call on the calling thread, final when it returns.

``outcome->status`` reads ``uform_status_deferred_k`` until the call is final, even for a call refused at submission.
``uform_outcome_status`` polls it with acquire ordering, and ``uform_outcome_wait`` blocks until it is final.
Submission copies the structure, while everything it points to stays borrowed until final.
A call stops early only when its progress handler returns ``uform_cancel_k``, and finishes ``uform_status_cancelled_k``.

.. code-block:: c

    uform_outcome_t outcomes[4] = {0};
    for (size_t shard = 0; shard != 4; ++shard) {
        encodes[shard].outcome = &outcomes[shard];
        encodes[shard].completion = uform_completion_deferred_k;
        uform_image_encode(&encodes[shard]);
    }
    for (size_t shard = 0; shard != 4; ++shard) uform_outcome_wait(&outcomes[shard]);

Every status carries the number USearch gives it, and ``uform_status_name`` spells it, like ``shape-mismatch``.

=========
Executors
=========

An executor owns what a call needs to run but a model does not: the host threads or device streams that advance the work, the activation scratch, and the pinned buffer staging passes through.
One executor serves any number of models of its residency, and calls on one executor run in submission order.

- ``uform_executor_for_ram`` spawns host threads across NUMA nodes, 7/8 of their cores by default.
- ``uform_executor_for_vram`` opens one stream per named device, or borrows the caller's, driven by whichever host thread submits.
- ``uform_executor_on_pool`` borrows a ForkUnion pool's threads, and reports ``unsupported`` until borrowing is wired.
- ``uform_executor_drain`` waits for every earlier call and picks whether the threads park or spin once idle.

Each executor tracks its deferred calls in a table sized by ``deferred_limit`` at creation, so submitting never allocates, and a submission finding it full finishes ``uform_status_contended_k``.

======
Arenas
======

Callers lend memory, never a way to allocate it, and every byte has one owner that picks where it lives once:

- Weights and their packed copies belong to the model, placed by its load's residency.
- Weights, gradients and optimizer moments under training belong to the trainer, on its executor's devices.
- Activations, scratch and staging belong to the executor, grown to a high-water mark.
- Every output whose length scales with the call lands in an arena, while fixed-size reports such as stats and descriptions stay caller fields.

``uform_arena_init`` places an arena where its executor computes, in host pages or in the memory of its GPU.
A call rewinds the arena it is given, invalidating the previous call's outputs, and writes views into it, so a steady loop allocates once.
Blocking calls borrow the arena, and deferred calls hold it until final, so an arena serves one call at a time.
On a VRAM executor the outputs are device memory, which a USearch VRAM index reads as it is.

.. code-block:: c

    uform_arena_t arena = NULL;
    uform_arena_init(&(uform_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
    for (size_t batch = 0; batch != batches_count; ++batch) {
        encode.texts = texts[batch], encode.arena = arena, encode.embeddings = &embeddings;
        uform_text_encode(&encode);
        if (encode.outcome->status != uform_status_success_k) break;
        consume(embeddings, encode.count); // valid until the next call given `arena`
    }
    uform_arena_free(arena);

A verb's ``count`` is its batch axis and nothing else.
Inputs are views, where a zero stride repeats the first element as BLAS does, and ragged inputs and outputs are tapes with Apache Arrow offsets, one more of them than entries.

===========
Residencies
===========

Residency is in the verb name, as in USearch and UStore, so a residency a build lacks has no verb to call:

- **RAM**: ``uform_model_load_ram`` and ``uform_model_load_ram_buffer`` place the weights in host memory, on the NUMA nodes ``uform_ram_options_t`` names, and every build carries them.
- **NVIDIA**: ``uform_model_load_vram`` and ``uform_model_load_vram_buffer`` place them on the CUDA devices ``uform_vram_options_t`` names, in builds with ``UFORM_BUILD_CUDA``.
- **AMD**: the same VRAM verbs place them on ROCm devices, in builds with ``UFORM_BUILD_ROCM``.

On NVIDIA, text and JPEG images never leave the device; container demux, PNG and WebP, audio, and every modality on AMD decode on host threads the executor spawns beside its streams.
``uform_capabilities_compiled`` and ``uform_capabilities_available`` report what the binary was built with and what the machine offers.

========
Encoding
========

``uform_model_describe`` reports the model's ``dimensions``, ``context``, ``quantization``, memory usage and capabilities.
Every encode verb runs the same tower, the modality's entry, the transformer blocks and the final norm, and keeps whichever surfaces the caller names from one forward:

- ``embeddings`` land one pooled row per item, past the attention pool and the projection head.
- ``states`` land one row per token each item kept, the surface rerankers and masked heads read.

A ``NULL`` output skips its surface, and naming neither is refused with ``uform_status_invalid_argument_k``.
Both land in the arena as ``outputs_scalar``, and an item longer than the model's ``context`` is truncated and counted in the stats.
One verb per input form takes only the fields that form needs:

- ``uform_text_encode`` takes UTF-8 texts, a byte per token, since models are tokenizer-free.
- ``uform_image_encode`` takes JPEG, PNG or WebP files, and ``uform_image_encode_pixels`` interleaved 8-bit RGB.
- ``uform_video_encode`` takes MP4 or WebM clips.
- ``uform_audio_encode`` takes WAV, FLAC, MP3 or Opus files, and ``uform_audio_encode_samples`` mono samples at a rate.

.. code-block:: c

    uform_rows_view_t embeddings = {0};
    uform_rows_tape_view_t states = {0};
    uform_text_encode(&(uform_text_encode_t){
        .model = model, .executor = executor, .outcome = &outcome, .count = n,
        .texts = {bytes, offsets, uform_scalar_u8_k}, .embeddings = &embeddings, .states = &states,
        .arena = arena, .outputs_scalar = uform_scalar_bf16_k});

==================
Mesh and Trainer
==================

Training is GPU-only.
``uform_mesh_bootstrap`` joins this process to a run once, through files in a directory every host sees, and after it processes talk only over NVLink and InfiniBand.
The mesh is the process-wide half of the transport, which every trainer borrows; ``uform_mesh_describe`` reports the topology every process agreed on and its multimem tier.

``uform_trainer_init_vram`` makes a trainer on a VRAM executor, owning one replica of the tower per device, its gradients, the Adafactor moments, the reconstruction heads and the learnable temperatures.
One ``uform_trainer_step`` is one optimizer step over this process's share of the global batch: forward, the masked and contrastive losses, backward, the world gradient reduction, and the update.
Items arrive tagged with a ``uform_modality_t``, grouped into bags that travel together, with attract, repel and ignore relations indexing the step's own items.
``uform_trainer_save`` writes the safetensors checkpoint the model verbs load.

.. code-block:: c

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
