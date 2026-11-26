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

The wheels include the CPU library.
A build from source compiles the C library and the extension through `scikit-build-core`, and the suites run against the stub library alone:

```sh
pip install . -C cmake.define.UFORM_BUILD_STUB=ON
uv run pytest test/python
```

## Quickstart

Encode two captions and a photo into one space, then pick the caption closest to the photo:

```python
import numpy as np
import uform

executor = uform.Executor.for_ram()  # the threads every receiver names
arena = uform.Arena(executor)  # where outputs land; each call rewinds it
model = uform.Model.load_ram("model.safetensors")

captions = ["a cat on a mat", "a sunrise over the bay"]
texts = np.array(model.on(executor).text_encode(captions, arena).embeddings)  # a copy outlives the next call
with open("cat.png", "rb") as photo:
    image = np.array(model.on(executor).image_encode([photo.read()], arena).embeddings)[0]

cosines = texts @ image / (np.linalg.norm(texts, axis=1) * np.linalg.norm(image))
print(f"{captions[cosines.argmax()]!r} describes the photo best")
```

## Receivers & Deferred Calls

A receiver holds what every C argument structure puts before its inputs: the executor and the completion.
The model runs a verb on the calling thread, `model.on(executor)` answers a `uform.On` whose verbs block on that executor, and `model.deferred(executor)` answers a `uform.DeferredOn` whose verbs submit there and answer a `uform.Deferred`.
The receiver's type alone chooses the completion, so all three offer the same verbs with the same arguments.
In a coroutine, `await` a deferred call; anywhere else, `wait()` blocks the thread with its state detached:

```python
call = model.deferred(executor).image_encode(files, uform.Arena(executor))
encoded = await call  # or: call.wait()
```

`call.status` polls without waiting, reading `Status.DEFERRED` until the call is final.
`await` re-polls it on the next turn of the running loop through `loop.call_soon`, never sleeping and parking no thread, so calls on two executors run together and `asyncio.gather` awaits both:

```python
async def fan_out(model, left, right, files, texts):
    images = model.deferred(left).image_encode(files, uform.Arena(left))
    captions = model.deferred(right).text_encode(texts, uform.Arena(right))
    return await asyncio.gather(images, captions)  # both submitted before either is awaited
```

Dropping a `Deferred` waits for its call first, since the library still reads the inputs it holds.
A trainer runs `step` and `save` blocking on the executor it was created on, and `trainer.on(...)` and `trainer.deferred(...)` choose another completion.

## Executors

`uform.Executor.for_ram(threads=0, *, memory_domains=None, cpu_class_ceiling=None)` creates an executor of host threads, and `uform.Executor.for_vram(devices=None, *, scratch_limit=0, gpu_class_ceiling=None)` opens one stream per device.
A device is named by its position in `uform.gpu_devices()`, which lists every GPU the process can drive as a `uform.GPUDevice`, NVIDIA devices first, then AMD, then Apple.
A `uform.CPUClass` or `uform.GPUClass` ceiling dispatches as the best class under it the hardware runs, and `None` sets no ceiling.
An executor runs one call at a time, and a call finding another one holding it raises `ContendedError` once waited for. A deferred call on the executor's own threads finishes while Python only awaits it, and its progress callbacks run on those threads, holding the GIL while they do.
`executor.drain(uform.Mood.PARKED)` waits until no call holds the executor and its devices are idle, then applies the mood.
In a `with` block, it waits for its calls and closes on exit.

## Arenas

Every output whose length only the call knows, the embeddings, the states and their offsets, lands in a `uform.Arena` you pass, and the verb answers views over it without copying.
`uform.Arena(executor)` places it where that executor computes, decided once, and `uform.Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, pass that call another arena; an arena serves one call at a time, so each deferred call needs its own.
A live `memoryview`, NumPy view, DLPack capsule, Arrow array or CUDA Array Interface consumer of a result keeps its arena from rewinding: a call naming it raises `ContendedError`, as does a call naming an arena another call holds.
An arena from a VRAM executor holds device memory: the buffer protocol raises `BufferError` wherever the host may not read the pages, and CUDA device and managed memory also export `__cuda_array_interface__` for CuPy or Numba.
An Apple GPU arena's outputs cross as host memory, through `kDLCPU`, plain Arrow arrays and the buffer protocol, and Metal tensors or Arrow arrays offered as inputs raise `UnsupportedError`, as no Metal exchange is supported yet.
An arena never outlives the executor that made it, so drop every arena before the executor's `with` block ends, and an arena from `uform.Arena()` lives until it is dropped.

```python
arena = uform.Arena(executor)
for batch in batches:
    encoded = model.on(executor).text_encode(batch, arena)  # rewinds: the last batch's views are gone
    index.add(np.array(encoded.embeddings))  # a copy, so it outlives the next rewind
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
A `(data, offsets)` tuple of buffers, the data and `count + 1` `size_t` offsets, is read in place as a tape without a copy, and both buffers must stay alive and unchanged until the call is final:

```python
data = b"first documentthe second"
offsets = np.array([0, 14, 24], dtype=np.uint64)
encoded = model.text_encode((data, offsets), arena)
```

Each member of the pair crosses through DLPack first, a versioned capsule before a legacy one, then a primitive Arrow array, a device export before a host one, then the CUDA Array Interface on a CUDA arena, then the buffer protocol; device memory crosses only onto the GPU the arena lives on.
On a GPU arena, a DLPack producer on that vendor's device is handed the arena's first stream to order its writes before, and the stream a CUDA Array Interface names is waited on from that first stream, so the host never blocks.
An Apache Arrow array or stream of strings, binaries or lists of `float32` is read the same way, as Arrow lays them out with offsets too: 64-bit offsets in place, 32-bit ones widened while the bytes stay in place, and a stream of several chunks gathered with one copy.
Arrow nulls are refused, and its offsets must be memory the host may read, as they drive host-side dispatch, so for an array carrying a `sync_event` the arena's first stream follows it and the host waits for that stream before reading:

```python
texts = pyarrow.array(["first document", "the second"], pyarrow.large_string())
encoded = model.text_encode(texts, arena)
```

Every output exports itself in place: `__dlpack__` for NumPy, PyTorch or CuPy, `__arrow_c_array__` and `__arrow_c_device_array__` for PyArrow, rows as fixed-size lists of their columns, and a `RowsTape` also as `__arrow_c_stream__`, a large list of each item's rows.
While any such export is alive, a call naming its arena raises `ContendedError`, so a consumer never reads rows a later call rewrote.
A `stream` handed to `__dlpack__` must sit on the output's device, and once the consumer drops the capsule, the arena's next call waits for the work queued on that stream before it rewinds.

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
`progress=handler` is called with `(completed, total)` items, and a handler returning `False` stops the call, which raises `uform.CancelledError`; one that raises is reported through `sys.unraisablehook` and stops the call, which raises `CancelledError` too.

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
The RAM loads take a `cpu_class_ceiling` and the VRAM loads a `gpu_class_ceiling`, and devices sharing no class under the ceiling raise `UnsupportedError`.
`model.describe()` answers a `uform.ModelDescription` of `dimensions`, `context`, `quantization`, `memory_usage`, `cpu_class`, `gpu_class`, `token_width` and `choice_temperature_groups`.

```python
model = uform.Model.load_ram("model.safetensors", quantization=uform.Scalar.E4M3)  # Float8 blocks, half the bytes
described = model.describe()
print(described.dimensions, described.context, described.cpu_class, described.memory_usage)
```

## Training

Training is GPU-only.
The configuration arrives as keyword-built structures mirroring the C ones field by field, so a field left out is zero, which the library derives:

```python
executor = uform.Executor.for_vram(devices=[0, 1])
mesh = uform.Mesh.bootstrap(
    executor,
    uform.MeshOptions(host_domain_index=rank, host_domain_count=nodes, directory="/shared/exchange", job="run-17"),
)
trainer = uform.Trainer.init_vram(
    executor,
    mesh=mesh,
    depth=12,
    hidden=512,
    objective=uform.Objective.JOINT,
    batch=uform.BatchShape(
        max_rows_per_device=64,
        max_segments_per_device=256,
        max_columns_per_row=1024,
        max_bags_per_device=128,
        image=uform.ImageShape(patch_side=16),
    ),
    schedule=uform.Schedule(steps=100_000),
    seed=42,
)
stats = uform.TrainerStepStats()
trainer.step(modalities, contents, bags=bags, relations=uform.Relations(attract=(offsets, ids)), stats=stats)
trainer.save("checkpoint.safetensors")
```

`BatchShape` nests an `ImageShape`, a `VideoShape` and an `AudioShape`, each also taken as a tuple of its fields in order.
`Masking`, `Contrastive`, `LateInteraction`, `Choices` and `Schedule` configure the objectives, and `Relations` holds three `Edges` sets of buffers: `size_t` offsets, `uint32` IDs and optional `float32` weights.
`Contrastive` and `LateInteraction` take a `uform.PairLossForm`, and `trainer.step` takes each item's `uform.LateInteractionSide` as `sides`.
`choices`, taken as `text_choose` takes them, and `choice_targets`, a `float32` tape shaped as the `RowsTape` it answers, train the calibrated choices within the batch's `max_choices_per_item`, so `(probabilities, probabilities.offsets)` from one model can train another.
Paths, like the mesh's `directory`, cross as `os.fsencode` gives their bytes, while its `job` is a `str` name.
`trainer.step` answers the pooled embeddings the loss scored when given an `arena`, and `None` otherwise.
`mesh.describe()` answers a `uform.MeshDescription`, its `multimem_tier` a `uform.MultimemTier` and its `gpu_class` the `uform.GPUClass` every GPU of the run dispatches as.

## Errors

Every failure raises a subclass of `uform.Error`, itself a `RuntimeError`, and each holds the status the C ABI reported as its `status` attribute, a `uform.Status` member:

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
- `OVERFLOW` → `CapacityOverflowError`, an `OverflowError`
- `UNINITIALIZED` → `UninitializedError`
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
  `status` polls through `uform_outcome_status`, `wait` blocks through `uform_outcome_wait`, and `await` re-polls `status` once per turn of the running loop through `loop.call_soon`, so no thread waits.
- __No life of its own.__
  The binding starts no threads and keeps no queues or caches; releasing a `Deferred` rides reference counting, which waits for the call first.
- __Arenas belong to the caller and are reused, never shared.__
  Every output verb takes an arena and answers views into it; a view lasts until the next call on that arena rewinds it, and the library refuses an arena another call holds.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table entry, a surface `outputs` leaves out is never computed, and an arena allocates only when a call writes into it.
- __Blocking calls stay as simple as the language allows.__
  `model.text_encode(texts, arena)` takes no executor, completion or receiver argument and returns its result.
- __Suites run against the stub library__, and against the real one with the tiny checkpoint `UFORM_TEST_CHECKPOINT` names.
