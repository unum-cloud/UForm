# UForm for Python

The bindings are pure CPython, with no PyBind11, no NanoBind, and no SWIG.
They need CPython 3.12 or newer on Linux, macOS or Windows, free-threaded builds included, and the extension never requires NumPy.
Every verb is a batch over buffer-protocol inputs, and every output whose length the call decides lands in an arena you reuse.

- `uform.Model` encodes text, images, video and audio into pooled embeddings, per-token states and token vectors, and scores calibrated choices over texts.
- `uform.Trainer` trains one on GPUs, joined across hosts by a `uform.Mesh`.
- `uform.Executor` owns the threads or streams every call runs on, apart from any model.

## Installation

```sh
pip install uform
```

The wheels carry the CPU library.
A build from source compiles the C library and the extension through `scikit-build-core`, and the suites run against the stub library alone:

```sh
pip install . -C cmake.define.UFORM_BUILD_LIB_C=OFF -C cmake.define.UFORM_BUILD_LIB_C_STUB=ON
uv run pytest test/python
```

## Quickstart

```python
import numpy as np
import uform

executor = uform.Executor.for_ram()  # The threads every receiver names
arena = uform.Arena(executor)  # The memory outputs land in, made where the executor computes
model = uform.Model.load_ram("model.safetensors", quantization=uform.Scalar.BF16)

encoded = model.on(executor).text_encode(["a cat on a mat", "a dog in the fog"], arena)
embeddings = np.asarray(encoded.embeddings)  # (2, dimensions) float32, zero-copy
assert embeddings.shape == (2, model.describe().dimensions)
```

## Receivers & Deferred Calls

A receiver carries what every C argument structure puts before its inputs: the executor and the completion.
The model runs a verb on the calling thread, `model.on(executor)` answers a `uform.On` whose verbs block on that executor, and `model.deferred(executor)` answers a `uform.DeferredOn` whose verbs submit there and answer a `uform.Deferred`.
The receiver's type alone chooses the completion, so all three offer the same verbs with the same arguments.
In a coroutine, `await` a deferred call; anywhere else, `wait()` blocks the thread with its state detached:

```python
call = model.deferred(executor).image_encode(files, uform.Arena(executor))
encoded = await call  # or: call.wait()
```

`call.status` polls without waiting, reading `Status.DEFERRED` until the call is final.
Dropping a `Deferred` waits for its call first, since the library still reads the inputs it holds.
A trainer runs `step` and `save` blocking on the executor it was created on, and `trainer.on(...)` and `trainer.deferred(...)` choose another completion.

## Executors

`uform.Executor.for_ram(threads=0, *, memory_domains=None, deferred_limit=0, cpu_class=None)` spawns host threads, and `uform.Executor.for_vram(devices=None, *, scratch_limit=0, deferred_limit=0, gpu_class=None)` opens one stream per device.
A device is named by its position in `uform.gpu_devices()`, which lists every GPU the process can drive as a `uform.GPUDevice`, NVIDIA devices first, then AMD.
A `uform.CPUClass` or `uform.GPUClass` pin dispatches as that class, and `None` derives the hardware's own.
`uform.Executor.on_pool(pool)` mirrors the C verb that borrows a ForkUnion pool, and raises `UnsupportedError` until borrowing is designed.
Calls on one executor run in submission order, and a deferred call finding its `deferred_limit` table full raises `ContendedError` once waited for.
`executor.drain(uform.Mood.PARKED)` waits for every earlier call and sets what the executor does once idle.
In a `with` block, it waits for its calls and closes on exit.

## Arenas

Every output whose length only the call knows, the embeddings, the states and their offsets, lands in a `uform.Arena` you pass, and the verb answers views over it without copying.
`uform.Arena(executor)` places it where that executor computes, decided once, and `uform.Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, pass that call another arena; an arena serves one call at a time, so each deferred call needs its own.
A NumPy array or `memoryview` over an output pins its arena as a `memoryview` pins a `bytearray`, so the next call naming that arena raises `ContendedError` until the export is released.
A stale output raises `InvalidArgumentError` when read.
An arena from a VRAM executor holds device memory: its views export `__cuda_array_interface__` for CuPy, PyTorch or Numba, and host reads raise `DeviceError`.

```python
arena = uform.Arena(executor)
for batch in batches:
    encoded = model.on(executor).text_encode(batch, arena)  # rewinds: the last batch's views are gone
    index.add(np.array(encoded.embeddings))  # a copy, so nothing pins the arena
```

## Encoding

One verb per input form, each spelled as in C:

- `text_encode(texts, arena)` takes UTF-8 texts, a byte per token, since models are tokenizer-free.
- `image_encode(files, arena)` takes JPEG, PNG or WebP files.
- `image_encode_pixels(pixels, arena)` takes C-contiguous `[height, width, 3]` `uint8` buffers.
- `video_encode(files, arena)` takes MP4 or WebM clips.
- `audio_encode(files, arena)` takes WAV, FLAC, MP3 or Opus recordings.
- `audio_encode_samples(samples, sample_rate, arena)` takes 1-D `float32` buffers.

A batch is a sequence of separate objects, `str` or bytes-like for texts and bytes-like for files, gathered into one tape with one copy.
A `(data, offsets)` tuple of buffers, the data and `count + 1` `size_t` offsets, is lent to the library as a tape without a copy:

```python
data = b"first documentthe second"
offsets = np.array([0, 14, 24], dtype=np.uint64)
encoded = model.text_encode((data, offsets), arena)
```

`outputs` names the surfaces a call keeps, and decides which C output pointers the binding passes:

```python
encoded = model.text_encode(texts, arena, outputs=uform.Outputs.EMBEDDINGS | uform.Outputs.STATES)
embeddings, states, token_vectors = encoded  # an `ArenaBuffer` of (count, dimensions), a `RowsTape`, and None
offsets = np.asarray(states.offsets)  # item `i` spans rows `offsets[i]` up to `offsets[i + 1]`
```

`Outputs.EMBEDDINGS` is the default, and a surface left out answers `None` and costs nothing past the final norm.
`Outputs.TOKEN_VECTORS` keeps the token head's unit vectors, a `RowsTape` of `(tokens, token_width)` on the states' offsets, for late-interaction reranking; a model whose `describe().token_width` is zero raises `UnsupportedError`.
`outputs_scalar` picks the scalar every output lands as, `Scalar.F32` by default; `Scalar.BF16` outputs export as raw bytes, a `(count, row_bytes)` run of `'B'`.
`stats=uform.TextEncodeStats()` and its siblings are overwritten with the truncated items and the encoded tokens.
`progress=handler` is called with `(completed, total)` items, and a handler returning `False` stops the call, which raises `uform.CancelledError`; one that raises stops it too, and its own exception reaches the caller.

## Choices

`text_choose(texts, choices, arena)` scores each text's legal single-byte choices through one forward, a softmax over them alone at the calibrated inverse temperature of the text's question type:

```python
probabilities = model.text_choose(["keep this candidate? ", "expand which neighbour? "], ["yn", "ABCD"], arena)
offsets = np.asarray(probabilities.offsets)  # [0, 2, 6]: one float32 per choice, in `choices` order
```

`temperature_groups` names each text's question type, below the model's `choice_temperature_groups`, and `None` puts every text in group zero.
A model without the byte decoder, its `choice_temperature_groups` zero, raises `UnsupportedError`.

## Models

`uform.Model.load_ram(path)` and `uform.Model.load_vram(path, devices=[0])` copy a safetensors checkpoint into host memory or onto GPUs, and `load_ram_buffer` and `load_vram_buffer` take the checkpoint's bytes instead.
Each takes `quantization`, `Scalar.BF16` as stored or `Scalar.E4M3` quantized at load, an optional `executor` the load runs on, and a `progress` handler.
The RAM loads take a `cpu_class` pin and the VRAM loads a `gpu_class` pin, and a class the hardware cannot run raises `UnsupportedError`.
`model.describe()` answers a `uform.ModelDescription` of `dimensions`, `context`, `quantization`, `memory_usage`, `capabilities`, `cpu_class`, `gpu_class`, `token_width` and `choice_temperature_groups`.

## Training

Training is GPU-only.
The configuration arrives as keyword-built structures mirroring the C ones field by field, so a field left out is zero, which the library derives:

```python
executor = uform.Executor.for_vram(devices=[0, 1])
mesh = uform.Mesh.bootstrap(executor, uform.MeshOptions(host_domain_index=rank, host_domain_count=nodes,
                                                        directory="/shared/exchange", job="run-17"))
trainer = uform.Trainer.init_vram(executor, mesh=mesh, depth=12, hidden=512, objective=uform.Objective.JOINT,
                                  batch=uform.BatchShape(max_rows_per_device=64, max_segments_per_device=256,
                                                         max_columns_per_row=1024, max_bags_per_device=128,
                                                         image=uform.ImageShape(patch_side=16)),
                                  schedule=uform.Schedule(steps=100_000), seed=42)
stats = uform.TrainerStepStats()
trainer.step(modalities, contents, bags=bags, relations=uform.Relations(attract=(offsets, ids)), stats=stats)
trainer.save("checkpoint.safetensors")
```

`BatchShape` nests an `ImageShape`, a `VideoShape` and an `AudioShape`, each also taken as a tuple of its fields in order.
`Masking`, `Contrastive`, `LateInteraction`, `Choices` and `Schedule` configure the objectives, and `Relations` holds three `Edges` sets of buffers: `size_t` offsets, `uint32` IDs and optional `float32` weights.
`Contrastive` and `LateInteraction` take a `uform.PairLossForm`, and `trainer.step` takes each item's `uform.LateInteractionSide` as `sides`.
A `ChoicesView` of `size_t` offsets, `uint8` bytes and `float32` targets trains the calibrated choices, within the batch's `max_choices_per_item`.
`trainer.step` answers the pooled embeddings the loss scored when given an `arena`, and `None` otherwise.
`mesh.describe()` answers a `uform.MeshDescription`, its `multimem_tier` a `uform.MultimemTier` and its `gpu_class` the `uform.GPUClass` every GPU of the run dispatches as.

## Errors

Every failure raises a subclass of `uform.Error`, itself a `RuntimeError`, and each carries the status the C ABI reported as its `status` attribute, a `uform.Status` member:

```python
try:
    model.on(executor).text_encode(texts, arena, progress=lambda completed, total: completed < 100)
except uform.CancelledError:
    ...  # The handler stopped the call, which published nothing
except uform.Error as error:
    print(error.status)  # Status.SHAPE_MISMATCH, Status.DEVICE, ...
```

`SUCCESS` and `DEFERRED` raise nothing, and every other member has its own class, which also derives the built-in naming the same fault where one exists:

- `OUT_OF_MEMORY` → `OutOfMemoryError`, a `MemoryError`
- `SHAPE_MISMATCH` → `ShapeMismatchError`, a `ValueError`
- `UNSUPPORTED_LAYOUT` → `UnsupportedLayoutError`, a `ValueError`
- `UNKNOWN_KIND` → `UnknownKindError`, a `ValueError`
- `IO` → `InputOutputError`, an `OSError`
- `DEVICE` → `DeviceError`
- `MALFORMED` → `MalformedError`, a `ValueError`
- `INTERNAL` → `InternalError`
- `CONFIG_MISMATCH` → `ConfigMismatchError`, a `ValueError`
- `CONTENDED` → `ContendedError`
- `CANCELLED` → `CancelledError`
- `UNSUPPORTED` → `UnsupportedError`, a `NotImplementedError`
- `INVALID_ARGUMENT` → `InvalidArgumentError`, a `ValueError`

A failure the binding catches before the boundary raises the class the library would have raised for the same fault.
Arguments the binding refuses raise at the call, while the library's refusals of a deferred call raise at `await` or `wait()`.

## Design

- __A thin layer over the C ABI.__
  One module per header, `types`, `model`, `encode` and `train`, one method per exported verb spelled as in C, and enums mirroring the C enums as `enum.IntEnum` classes built from the C name functions, never strings.
- __The receiver chooses, the verb stays.__
  The model runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `uform_outcome_status`, `wait` blocks through `uform_outcome_wait`, and `await` hands `wait()` to the running event loop's default executor.
- __No life of its own.__
  The binding starts no threads and keeps no queues or caches; releasing a `Deferred` rides reference counting, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output verb takes an arena and answers views borrowing it; an exported buffer pins its arena until released, and a view over a rewound one refuses to be read.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table entry, a surface `outputs` leaves out is never computed, and an arena allocates only when a call writes into it.
- __Blocking calls stay as simple as the language allows.__
  `model.text_encode(texts, arena)` takes no executor, completion or receiver argument and returns its result.
- __Suites run against the stub library__, and against the real one with the tiny checkpoint `UFORM_TEST_CHECKPOINT` names.
