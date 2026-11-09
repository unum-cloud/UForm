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

```mojo
from uform import Arena, Executor, Library, Model, Outputs, ScalarKind, TapeView, EncodeConfig

def main() raises:
    var library = Library()
    var executor = Executor.for_ram(library, threads=8)
    var model = Model.load_ram(executor, "model.safetensors")  # BFloat16 blocks, as the checkpoint stores them

    var texts = "a cata dog".as_bytes()  # two texts end to end
    var offsets: List[UInt] = [0, 5, 10]  # one more offset than texts
    var arena = Arena(executor)
    var encoded = model.on(executor).text_encode(TapeView.of(texts, offsets), Outputs.EMBEDDINGS, arena)
    var embeddings = encoded.embeddings().value()
    print(len(embeddings), embeddings.columns(), embeddings.row[DType.float32](0)[0])
```

A model owns its weights; an `Executor` owns the threads or streams a call runs on; an `Arena` owns the outputs a call writes.
`Library()` opens the engine once, and every handle created through it keeps it mapped.

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

A `Deferred` owns its outcome and outputs until the call is final, and borrows the model, the executor, the arena and every input it was handed, which Mojo keeps alive for as long as the `Deferred` is.
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
var shared = Executor.on_pool(library, fork_union_pool)                       # borrows a fu_pool_t
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
var probabilities = model.text_choose(states, choices, arena)  # one F32 per legal byte
```

## Training

A run joins its processes with `Mesh.bootstrap` over a device executor, then steps one `Trainer` per process:

```mojo
var executor = Executor.for_vram(library, VramOptions.on(devices))
var mesh = Mesh.bootstrap(executor, options)
var config = TrainerInitVramConfig[origin_of(mesh)]()
config.depth = 12
config.hidden = 768
config.mesh = Pointer(to=mesh)
var trainer = Trainer.init_vram(executor, config)
trainer.on(executor).step(modalities, contents, TrainerStepConfig(relations=relations))
trainer.on(executor).save("checkpoint.safetensors")
```

## Errors

Every fallible entry point raises `Error` and nothing else, since Mojo allows one error type per function.
An `Error` carries a `Status` shared with the C interface, so a fault this binding catches before the boundary reports the status the engine would have reported for the same fault:

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
