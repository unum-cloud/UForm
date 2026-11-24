# UForm for Mojo

The Mojo package binds the C ABI in `c/uform/`, opening the engine's shared library with `dlopen` rather than linking it, so a consumer builds with no linker flags of its own.
It needs Mojo 1.2 through pixi, on Linux x86-64, Linux AArch64 or macOS on Apple silicon.
It binds every exported verb: the model and its encode verbs, the mesh and the trainer, and the executors and arenas they run on and write into.

## Installation

Let Pixi build the package from source, with no channel to add and no linker flag to pass; the engine arrives as its run dependency, in the environment prefix `mojo build` bakes as runpath.

```toml
[workspace]
preview = ["pixi-build"]

[dependencies]
uform_mojo = { git = "https://github.com/unum-cloud/UForm", subdirectory = "mojo", tag = "v4.0.0-rc.1" }
```

From a checkout, `pixi run test` builds the engine through the tree's own CMake, with the host codecs from conda-forge, and runs `mojo/test.mojo` against it.
The model checks need a checkpoint: `uform_tiny_checkpoint <path>` writes a tiny one, and `UFORM_TEST_CHECKPOINT=<path> pixi run test` runs them; without it they are skipped.
On Linux the environment pins a glibc 2.34 sysroot, which the MAX runtime `mojo build` links against needs.

## Quickstart

Encode two captions and a photo into one space, then pick the caption closest to the photo:

```mojo
from std.math import sqrt
from uform import Arena, Executor, Library, Model, Outputs, TapeView


def cosine(text: Span[Float32, _], image: Span[Float32, _]) -> Float32:
    var dot: Float32 = 0  # embeddings come unnormalized
    var text_norm: Float32 = 0
    var image_norm: Float32 = 0
    for i in range(len(image)):
        dot += text[i] * image[i]
        text_norm += text[i] * text[i]
        image_norm += image[i] * image[i]
    return dot / sqrt(text_norm * image_norm)


def main() raises:
    var library = Library()  # opens the engine once; every handle keeps it mapped
    var executor = Executor.for_ram(library)
    var model = Model.load_ram(executor, "model.safetensors")  # BFloat16 blocks, as the checkpoint stores them
    var texts_arena = Arena(executor)  # a call rewinds its arena, so two outputs need two
    var images_arena = Arena(executor)

    var captions: List[String] = ["a cat on a mat", "a sunrise over the bay"]
    var caption_offsets: List[UInt] = [0, 14, 36]  # one more offset than captions, end to end
    var photo = open("cat.png", "r").read_bytes()
    var photo_offsets: List[UInt] = [0, UInt(len(photo))]
    var tape = TapeView.of("a cat on a mata sunrise over the bay".as_bytes(), caption_offsets)
    var texts = model.on(executor).text_encode(tape, Outputs.EMBEDDINGS, texts_arena).embeddings().value()
    var images = model.on(executor).image_encode(TapeView.of(photo, photo_offsets), Outputs.EMBEDDINGS, images_arena)
    var image = images.embeddings().value().row[DType.float32](0)
    var best = 1 if cosine(texts.row[DType.float32](1), image) > cosine(texts.row[DType.float32](0), image) else 0
    print("'" + captions[best] + "' describes the photo best")
```

## Receivers & Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
The model runs it on the calling thread, `model.on(executor)` blocks on an executor, and `model.deferred(executor)` submits it there and answers a `Deferred`:

```mojo
model.text_encode(texts, Outputs.EMBEDDINGS, arena)                          # the calling thread
model.on(executor).text_encode(texts, Outputs.EMBEDDINGS, arena)             # blocking; `CONTENDED` while another call holds it
var call = model.deferred(executor).text_encode(texts, Outputs.EMBEDDINGS, arena)  # Deferred[...]
print(call.wait().embeddings().value().row[DType.float32](0)[0])
```

`ModelOn` and `ModelDeferredOn` offer the same verbs with the same arguments, and the blocking verbs are written once, in the `ModelVerbs` trait the model and `ModelOn` conform to.
A trainer has only the executor receivers, `trainer.on(executor)` and `trainer.deferred(executor)`, since its verbs run on the devices it was made over.
Loads and `Mesh.bootstrap` address no handle yet, so they are constructors taking the executor they run on.

A `Deferred` owns its outcome and outputs until the call is final; the model, the executor, the arena and every input it was handed stay the caller's, and Mojo keeps them alive for as long as the `Deferred` is.
Mojo has no runtime of its own to await on, so `wait` is the whole interface: it blocks in `uform_outcome_wait` until the call is final, then answers the result or raises the failure, as many times as it is asked.
`status` reads the outcome without blocking, and destroying an unfinished `Deferred` waits for it.

### Progress

Every verb with a `progress` slot takes a `Progress`, made from a handler that is a compile-time parameter, since a capturing closure cannot cross into C; its state crosses beside it as a pointer that must outlive the call:

```mojo
def stop_halfway(completed: Int, total: Int, seen: Pointer[Int, MutAnyOrigin]) -> Continuation:
    seen[] = completed
    return Continuation.CANCEL if 2 * completed >= total else Continuation.CONTINUE

var seen = 0
var config = EncodeConfig(progress=Progress.of[stop_halfway](Pointer(to=seen), steps=10))
model.text_encode(texts, Outputs.EMBEDDINGS, arena, config)  # raises Status.CANCELLED halfway
```

Handlers returning a value are confirmed on Linux x86-64 with Mojo 1.2; Linux AArch64 and macOS rest on the CI matrix.

## Executors

One executor serves any number of models of its residency.
`Executor(library)` owns nothing and runs every call handed it on the calling thread.
There is a constructor per C one, and each fixes a kind that must match the residency of every model it drives:

```mojo
var threads = Executor.for_ram(library, RamOptions.on(domains), threads=8)  # zero takes 7/8 of the cores
var streams = Executor.for_vram(library, VramOptions.on(devices))             # one host thread, a stream per device
threads.drain(Mood.SPINNING)                                                # waits until no call holds it, then keeps spinning
```

An executor runs one call at a time, so threads that genuinely overlap take one executor apiece.
A call finding another one holding its executor finishes at once with `Status.CONTENDED`.

## Arenas

Every output whose length only the call knows, embeddings, states, token vectors and probabilities, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`Arena(executor)` places it where that executor computes, decided once, and `Arena(library)` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once; reading a stale one raises `Status.INVALID_ARGUMENT`.
To keep a result past the next call, pass that call another arena, and give every deferred call its own.
An arena from a VRAM executor holds device memory, which host accessors refuse with `Status.DEVICE`, and serves only that executor's calls, keeping the executor alive until the arena goes.

The engine classifies every input pointer through the driver, so `TapeView.of` takes a host span or a host or device pointer, such as `buffer.unsafe_ptr()` of a MAX `DeviceBuffer`, with its offsets in host memory.

## Encoding

Every encode verb takes its inputs as tapes, elements end to end with one more offset than items, then the `Outputs` to keep, the arena, and an optional `EncodeConfig` naming the scalar outputs land as and a report slot:

```mojo
var stats = EncodeStats()
var config = EncodeConfig(outputs_scalar=ScalarKind.BF16, stats=Pointer(to=stats))
var images = model.image_encode(files, Outputs.EMBEDDINGS | Outputs.TOKEN_VECTORS, arena, config)
var pixels = model.image_encode_pixels(rgb, heights, widths, Outputs.EMBEDDINGS, arena)
var clips = model.video_encode(videos, Outputs.EMBEDDINGS, arena)
var sounds = model.audio_encode_samples(samples, 16000, Outputs.EMBEDDINGS, arena)
```

## Choices

`text_choose` scores each text's legal choices, one byte each, through one forward of the byte decoder, a softmax over them alone at the calibrated inverse temperature of the text's question type:

```mojo
var states = "keep this candidate? expand which neighbour? ".as_bytes()
var state_offsets: List[UInt] = [0, 21, 45]
var letter_offsets: List[UInt] = [0, 2, 6]
var choices = TapeView.of("ynABCD".as_bytes(), letter_offsets)
var probabilities = model.text_choose(TapeView.of(states, state_offsets), choices, arena)
var expand = probabilities.item[DType.float32](1)  # the probabilities of 'A' through 'D'
```

`TextChooseConfig` names each text's question type in `temperature_groups`, below the model's `choice_temperature_groups`, and an empty span puts every text in group zero.
A model without the byte decoder, its `choice_temperature_groups` zero, raises `Status.UNSUPPORTED`.

## Models

`Model.load_ram` and `Model.load_vram` copy a safetensors checkpoint into host memory or onto the GPUs of a VRAM executor, and `load_ram_buffer` and `load_vram_buffer` take the checkpoint's bytes instead.
`quantization` keeps the blocks in `ScalarKind.BF16` as stored or quantizes them to `ScalarKind.E4M3` at load, and `RamOptions` and `VramOptions` name the memory domains or devices and an optional `cpu_class_ceiling` or `gpu_class_ceiling`, under which dispatch runs the best class the hardware runs.

```mojo
var model = Model.load_vram(streams, "model.safetensors", VramOptions.on(devices), quantization=ScalarKind.E4M3)
var described = model.describe()
print(described.dimensions, described.context, described.memory_usage, described.gpu_class)
```

`describe` reports the `dimensions`, `context`, `quantization`, `memory_usage`, the `cpu_class` a RAM model was packed for or the `gpu_class` a VRAM model runs as, the `token_width` of its token head and the `choice_temperature_groups` of its byte decoder.
A model frees its weights when Mojo destroys it, which waits for every deferred call naming it.

## Training

Training is GPU-only.
A run joins its processes with `Mesh.bootstrap` over a device executor, through files in a directory every host sees, then steps one `Trainer` per process.
Each `step` is one optimizer step over this process's share of the global batch: forward, the losses, backward, the world gradient reduction and the update.
Items arrive as a tape of contents tagged with their `Modality`, and `Relations` mark which items attract, repel or ignore each other:

```mojo
var executor = Executor.for_vram(library, VramOptions.on(devices))
var mesh = Mesh.bootstrap(executor, options)
var config = TrainerInitVramConfig[origin_of(mesh)]()
config.depth = 12
config.hidden = 768
config.mesh = Pointer(to=mesh)
var trainer = Trainer.init_vram(executor, config)
var stats = TrainerStepStats()
trainer.on(executor).step(modalities, contents, TrainerStepConfig(relations=relations, stats=Pointer(to=stats)))
print(stats.step, stats.loss, stats.contrastive_loss)  # world means
trainer.on(executor).save("checkpoint.safetensors")
```

`TrainerStepConfig` also holds the `sides` that make items queries or documents for the late-interaction term and the `choices` and `choice_targets` tapes the byte decoder trains on, each read when the trainer's weight for it is positive.
`save` writes the safetensors checkpoint the model loads read, and every process of the run bootstraps with the same count, directory and job, or `Mesh.bootstrap` raises `Status.CONFIG_MISMATCH`.

## Errors

Every fallible entry point raises `Error` and nothing else, since Mojo allows one error type per function.
An `Error` holds a `Status` shared with the C interface, so a fault this binding catches before the boundary reports the status the engine would have reported for the same fault:

```mojo
try:
    _ = Model.load_ram(executor, "missing.safetensors")
except error:
    if error.status == Status.IO:
        pass
    print(error)  # UForm: io [uform_model_load_ram]: ...
```

An `Error` keeps the engine mapped while it lives, since its name and message are the engine's own static text.

## Design

Blocking stays simple: a verb on the model or on `model.on(executor)` reads as it would if asynchrony did not exist.
Deferred calls are symmetric, the same verbs on another receiver, observed only through the outcome's status word, and release rides Mojo's own destructor.
Nothing costs anything until used: the binding starts no threads, keeps no counts and caches nothing, so a call is one argument structure built on the stack and one indirect call.
