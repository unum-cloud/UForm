# UForm for Swift

The Swift package binds the C ABI in `c/uform/` through `UFormC`, a target exposing its headers, and links the `libuform_static` engine CMake builds.
It requires Swift 6.2 or newer, on macOS 15, iOS 18, tvOS 18, watchOS 11, visionOS 2, or Linux.
Models are tokenizer-free, so the package has no dependencies beyond the C library: text crosses as UTF-8 bytes, media as encoded files or decoded samples.

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/unum-cloud/UForm", .upToNextMajor(from: "4.0.0"))
]
```

The package compiles the Swift surface only.
The engine itself is `libuform_static`, built by CMake and linked by the package, so point the linker at the build directory or install the library system-wide.

```sh
cmake --preset cpu_library
cmake --build --preset cpu_library --target uform_static
swift build -Xlinker -Lbuild_cpu_library
```

The test suite runs in seconds against the stub library, which answers every verb with deterministic rows of the right shape:

```sh
cmake --preset stub_library
cmake --build --preset stub_library --target uform_static_stub
swift test -Xlinker -Lbuild_stub_library/stub
```

## Quickstart

```swift
import UForm

let executor = try Executor.forRam()
let arena = try Arena(on: executor)                                  // where the outputs land
let model = try Model.loadRam(path: "model.safetensors")
let encoded = try model.on(executor).textEncode(["A blue house"], outputs: .embeddings, arena: arena)
let embedding = Array(encoded.embeddings![0])                        // [Float], the model's dimensions wide
```

## Receivers & Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
Called on the model, a verb runs on the calling thread; on `model.on(executor)` it blocks on that executor, after every call submitted there before it; on `model.deferred(executor)` it submits there and answers a `Deferred` at once.

```swift
let here = try model.textEncode(texts, arena: arena)                  // the calling thread
let blocking = try model.on(executor).textEncode(texts, arena: arena) // blocking, on the executor
let call = try model.deferred(executor).imageEncode(files, arena: arena) // Deferred<Encoded>
let images = try await call.value  // or: try call.wait()
```

A `Deferred` owns its outcome, its outputs and copies of its inputs until the call is final, so the caller's arrays may go at once.
It borrows the model, the executor and the arena as a blocking call does, since releasing any of them waits for the calls naming it.
`status` reads `deferred` until then, through `uform_outcome_status`, and `wait()` blocks through `uform_outcome_wait` and answers the result or throws the failure, the same every time it is asked.
`value` waits on a Dispatch thread and then reads the result in the caller's isolation, so an `async` caller suspends rather than blocks.
Releasing an unfinished one waits for it first.

```swift
let call = try model.deferred(executor).audioEncode(recordings, arena: arena)
if call.status == .deferred {
    // other work
}
let encoded = try await call.value  // or: try call.wait()
```

A failure this binding catches before the call crosses, like heights and widths that do not name one per image, throws at submission, and one the library catches arrives through `status` and `wait`.
A submission finding the executor's deferred table full, sized by `deferredLimit`, finishes at once with `contended`.

## Executors

A model says where its weights live; an `Executor` says who runs the calls on them.
One executor serves any number of models of its residency, so several models share one machine's cores or devices.

```swift
let hostThreads = try Executor.forRam(memoryDomains: [0, 1], threads: 0, deferredLimit: 0)
let deviceStreams = try Executor.forVram(devices: [0, 1], streams: nil, scratchLimit: 1 << 30)
try hostThreads.drain(mood: .parked)
```

Each factory takes every argument its C constructor has, and only those.
A thread count of zero takes 7/8 of the cores the memory domains carry, and `forVram` takes no thread count at all, since the submitting thread drives one stream per device.
`Executor.onPool(memoryDomains:cpuClass:pool:deferredLimit:)` mirrors the C verb that borrows a ForkUnion pool, and reports `unsupported` until the library wires borrowed pools.
`drain(mood:)` waits for the executor's earlier calls and picks whether its threads park or spin once idle.

## Arenas

Every output whose length scales with the call - pooled embeddings, per-token states and their offsets - lands in an `Arena` you pass, and the verb answers views over it without copying.
There is one verb per operation: no allocating twin, no caller buffers, and no default arena.
`Arena(on: executor)` places it where that executor computes, decided once, and `Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once it reaches its high-water mark.
Reading a stale view traps, as an out-of-bounds index does; to keep a result past the next call, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own, and a call naming an arena another call holds finishes `contended`.
An arena from a VRAM executor holds device memory, which host accessors refuse with a trap.

```swift
let arena = try Arena(on: executor)
for batch in batches {
    let encoded = try model.on(executor).textEncode(batch, arena: arena) // rewinds: the last batch's rows are gone
    index.add(encoded.embeddings!)
}
let calls = try zip(shards, arenas).map { shard, arena in
    try model.deferred(executor).imageEncode(shard, arena: arena)        // one arena per submitted call
}
for call in calls { consume(try await call.value) }                      // or: try call.wait()
```

A result keeps the arena it reads alive, so it outlives the model and the caller's reference alike.

## Encoding

Every verb runs one forward and keeps the surfaces `outputs` names, deciding which C output pointers the binding passes:

- `.embeddings`, the default, lands one pooled row per item as `Rows`;
- `.states` lands the final-norm states, one row per token each item kept after truncation, as `RowsTape`;
- `.tokenVectors` lands the token head's unit vectors, `tokenWidth` wide, on the same offsets as the states, and a model without a token head refuses it with `unsupported`;
- naming none is refused with `invalidArgument`.

```swift
let encoded = try model.textEncode(texts, outputs: [.embeddings, .states], arena: arena)
let pooled = encoded.embeddings!                  // pooled[i] is item i's row
let states = encoded.states!                      // states[i] is item i's rows, one per byte of its text
print(pooled.columns, pooled.stride, pooled.scalar, states[0].count, encoded.stats.truncated)
```

The six verbs take one input form each:

```swift
try model.textEncode(["a caption"], arena: arena)                                     // UTF-8, a byte per token
try model.imageEncode([jpeg, png], arena: arena)                                      // JPEG, PNG or WebP files
try model.imageEncodePixels([rgb], heights: [480], widths: [640], arena: arena)       // interleaved 8-bit RGB
try model.videoEncode([mp4], arena: arena)                                            // MP4 or WebM files
try model.audioEncode([wav], arena: arena)                                            // WAV, FLAC, MP3 or Opus
try model.audioEncodeSamples([samples], sampleRate: 16_000, arena: arena)             // mono Float samples
```

`EncodeConfig` carries the rest: `outputsScalar`, `f32` by default, and a `progress` handler with its `progressSteps`.
Rows of `f32`, `bf16` and `f16` read as `Float`, and `withUnsafeBytes` hands any scalar on as stored, such as to a USearch index.

```swift
let config = EncodeConfig(outputsScalar: .bf16, progress: { done, total in done < limit ? .continue : .cancel })
let narrow = try model.textEncode(texts, arena: arena, config: config)
narrow.embeddings!.withUnsafeBytes { bytes in index.add(bytes) }
```

A handler never runs on two threads at once, but may run on whichever thread advanced the work, hence `@Sendable`; returning `.cancel` stops the call with `cancelled`.

## Calibrated Choices

`textChoose` scores each text's legal choices, one byte each, through one forward of the byte decoder, a softmax over that item's choices alone at its question type's calibrated inverse temperature.
The call appends the mask byte, and a text longer than the context loses its head, never its end.
A model whose `choiceTemperatureGroups` is zero carries no byte decoder and refuses with `unsupported`.

```swift
let chosen = try model.textChoose(
    ["keep this candidate? ", "expand which neighbor? "],
    choices: [Array("yn".utf8), Array("ABCD".utf8)],
    temperatureGroups: [0, 1],
    arena: arena)
let keep = chosen.probabilities[0][0]           // one Float per choice, in the order offered
```

## Models

Residency is the constructor's name, as in the C lifecycle verbs, so a residency a build lacks has no constructor to call.

```swift
let inRam = try Model.loadRam(path: "model.safetensors", memoryDomains: [0], quantization: .bf16)
let inVram = try Model.loadVram(path: "model.safetensors", executor: deviceStreams, devices: [0, 1])
let fromBytes = try Model.loadRamBuffer(checkpoint.span.bytes, quantization: .e4m3)
let description = try inRam.describe()
print(description.dimensions, description.context, description.quantization, description.memoryUsage)
print(description.cpuClass, description.tokenWidth, description.choiceTemperatureGroups)
```

`quantization` is the scalar the transformer blocks run in: `bf16` as the checkpoint stores it, or `e4m3` quantized at load.
Every RAM load and executor takes a `cpuClass` pin, and every VRAM one a `gpuClass`, dispatching as if the hardware were of that class; nil derives its own.
`describe()` reports the class the model runs as, nil on the other residency, the `tokenWidth` of its token head, and its `choiceTemperatureGroups`, each zero when the checkpoint carries no such head.
`gpuDevices()` lists every GPU the process can drive, and a device's position in it is the ordinal `devices` names it by.

```swift
let pinned = try Model.loadRam(path: "model.safetensors", cpuClass: CPUClass(name: "intel-haswell"))
for (ordinal, device) in try gpuDevices().enumerated() { print(ordinal, device.name, device.gpuClass!) }
```

## Training

Training is GPU-only, so every constructor takes a VRAM executor, and the trainer's verbs block on it.

```swift
let mesh = try Mesh.bootstrap(executor: deviceStreams, options: MeshOptions(
    hostDomainIndex: node, hostDomainCount: nodes, directory: "/shared/rendezvous", job: "run-42"))
let trainer = try Trainer.initVram(
    executor: deviceStreams, depth: 12, hidden: 512,
    batch: BatchShape(maxRowsPerDevice: 64, maxSegmentsPerDevice: 256, maxColumnsPerRow: 1024,
                      maxBagsPerDevice: 128, image: ImageShape(patchSide: 16)),
    objective: .joint, schedule: Schedule(steps: 100_000), mesh: mesh, seed: 42)
let step = try trainer.step(
    modalities: [.text, .image, .text, .image], contents: items, bags: [0, 0, 1, 1],
    relations: Relations(attract: Edges(offsets: [0, 1, 2, 3, 4], ids: [1, 0, 3, 2])))
print(step.stats.step, step.stats.loss, step.stats.contrastiveLoss)
try trainer.save(path: "checkpoint.safetensors")
```

`Modality`, `Objective`, `Negatives` and `MultimemTier` spell themselves through the library's own name tables.
`Contrastive` and `LateInteraction` each take a `PairLossForm`, and a positive `LateInteraction.weight` trains the token head over the items whose `sides` name them a query, a document or both.
A positive `Choices.weight` trains calibrated choices, which needs `BatchShape.maxChoicesPerItem`, over the legal bytes each step's `choices` names, as `textChoose` takes them, and their `choiceTargets`.
The step reports both terms as `lateInteractionLoss` and `choicesLoss`.
A step over arrays of items gathers each into one tape, its one copy, while a step over `(data, offsets)` tapes lends their storage to the library in place.
`Modality`, `Objective`, `Negatives` and `MultimemTier` read a spelling back with `init?(name:)`, answering nil for a name the library's table would quietly read as its default.

## Errors

`UFormError` is the only error the SDK throws, and every method says so with `throws(UFormError)`, so a `catch` reads `error.status` with no cast.
`Status` has one constant per `uform_status_*_k` of the C interface, and its `description` spells it as the library does, like `invalid-argument`.

```swift
do {
    _ = try Model.loadRam(path: "missing.safetensors")
} catch {
    print(error.status, error.message) // io The checkpoint could not be opened for reading
}
```

`Scalar`, `Mood` and every other closed set read a spelling back with `init?(name:)`, and nil for one the library does not spell.

## What This Binding Does Not Expose

- The constructors address no model or trainer yet, so no receiver offers them; each blocks, the model loads on an optional `executor` and the trainer and mesh on a required one.
- A trainer is bound to the executor it was made on, so its verbs block there and take no receiver.

## Design

- __A thin layer over the C ABI.__
  One Swift file per header - `Types`, `Model`, `Encode`, `Train` - and a C verb `uform_<noun>_<verb>` is a camel-cased method here, such as `imageEncodePixels`.
- __The receiver chooses, the verb stays.__
  The model runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `uform_outcome_status`, `wait()` blocks through `uform_outcome_wait`, and `try await call.value` runs that wait on a Dispatch thread, so an `async` caller suspends.
- __No life of its own.__
  The binding starts no threads, keeps no queues and caches nothing; releasing a `Deferred` rides its `deinit`, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output verb takes an arena and answers views borrowing it; a view remembers which call wrote it and traps once a later one did.
- __Statuses and modes are enums.__
  `Status`, `Scalar`, `Mood`, `CPUClass`, `GPUClass`, `Modality`, `Objective`, `Negatives` and `MultimemTier` carry their C enumerators and spell themselves through the C name functions, never strings.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table entry, a surface `outputs` leaves out is never computed, and an arena allocates only when a call writes into it.
