# UForm for JavaScript

UForm encodes text, images, video and audio into embeddings USearch indexes as they are.
The JavaScript SDK has two backends behind one TypeScript API: a Node-API addon over the C library, and a WebAssembly build of the same library for browsers.
The WebAssembly build is how the smallest models sit inside a page, computing on the visitor's CPU with no server round trip.

## Installation

```bash
npm install @unum-cloud/uform
```

The package carries prebuilt addons under `prebuilds/`, and the `uform.mjs` and `uform.wasm` pair for browsers under `javascript/dist/wasm/`.

## Node

Models are tokenizer-free: text arrives as UTF-8 and every byte is a token.
An executor owns the threads, a model owns the weights, and an arena owns the outputs, so one executor serves many models and one arena serves a steady loop without allocating.

```js
import { Executor, Arena, Model, Outputs, Scalar } from "@unum-cloud/uform";

const executor = Executor.forRam();
const arena = new Arena(executor);
const model = Model.loadRam("model.safetensors", { executor });

const encoded = await model.deferred(executor).textEncode(["a small red panda"], arena); // or: call.wait()
const embedding = encoded.embeddings.row(0); // a Float32Array of the model's dimensions
```

The same verbs run on the calling thread with no receiver, and block on an executor through `on`:

```js
const here = model.textEncode(["a small red panda"], arena);
const there = model.on(executor).textEncode(["a small red panda"], arena);
```

Every verb keeps the pooled embeddings by default.
Ask for the final-norm states too, one row per token kept, with the `outputs` flags:

```js
const { embeddings, states, stats } = model.textEncode(texts, arena, {
  outputs: Outputs.Embeddings | Outputs.States,
  outputsScalar: Scalar.BF16,
});
const tokensOfFirst = states.item(0); // rows from states.offsets[0] up to states.offsets[1]
console.log(stats.truncated, stats.tokens);
```

A checkpoint with a token head also lands unit token vectors for late-interaction reranking, `tokenWidth` wide and sharing the states' offsets:

```js
const { tokenWidth } = model.describe(); // zero without a token head, which answers `unsupported`
const { tokenVectors } = model.textEncode(documents, arena, { outputs: Outputs.TokenVectors });
```

A checkpoint with the byte decoder scores closed sets of single-byte choices in one forward, each item's probabilities summing to one:

```js
const { probabilities } = model.textChoose(["keep this candidate?", "expand which neighbour?"], ["yn", "ABCD"], arena);
const [keep, drop] = probabilities.item(0);
```

Images, clips and recordings arrive encoded, as JPEG, PNG, WebP, MP4, WebM, WAV, FLAC, MP3 or Opus files:

```js
const images = model.imageEncode([await fs.readFile("panda.jpg")], arena);
const clips = model.videoEncode([await fs.readFile("panda.mp4")], arena);
const recordings = model.audioEncode([await fs.readFile("panda.wav")], arena);
```

Or decoded, as RGB or RGBA pixels with their shape, and as mono samples with their rate:

```js
const pixels = model.imageEncodePixels([{ data: rgb, width: 640, height: 480 }], arena);
const samples = model.audioEncodeSamples([new Float32Array(16000)], 16000, arena);
```

A model on GPUs takes a VRAM executor and lands its outputs in device memory, which no host view reads:

```js
const gpus = Executor.forVram({ devices: [0, 1] });
const resident = Model.loadVram("model.safetensors", { executor: gpus, devices: [0, 1] });
```

Device ordinals are positions in `gpuDevices()`, NVIDIA devices first, then AMD.
Dispatch picks a CPU or GPU class on its own, and `cpuClass` or `gpuClass` pins one by its name:

```js
for (const { name, gpuClass, memoryBytes } of gpuDevices()) console.log(name, gpuClass, memoryBytes);
const portable = Model.loadRam("model.safetensors", { cpuClass: "intel-haswell" });
console.log(portable.describe().cpuClass); // "intel-haswell"
```

## Browser

The browser build computes on the calling thread, so every call blocks and a deferred receiver answers a call that is already final.
Decode media with the platform, through a canvas or the Web Audio API, then hand the pixels and samples to UForm.

```js
import { initialize, Arena, Model } from "@unum-cloud/uform/wasm";

await initialize(); // loads the uform.mjs and uform.wasm the package ships
const model = Model.loadRamBuffer(await (await fetch(url)).arrayBuffer());
const arena = new Arena();

const context = canvas.getContext("2d");
const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
const encoded = model.imageEncodePixels([pixels], arena);
const texts = model.textEncode(["a small red panda"], arena);
```

A bundler that relocates the module passes its factory or URL to `initialize`.
Encoded media, VRAM, the mesh and the trainer report `unsupported` in the browser build.

## Arenas and Views

Every output whose length scales with the call lands in an arena, and the verb answers views into it rather than copies.
Each call rewinds the arena it is given, detaching the buffers under the previous call's views, so a stale view refuses its reads with `invalid-argument` instead of reading another call's memory.
To keep one result past the next call, give that call another arena.
A deferred call holds its arena until final, and a second call naming it finishes `contended`.

```js
const first = model.textEncode(["first"], arena);
const second = model.textEncode(["second"], arena);
first.embeddings.data; // throws: the arena was rewound
```

## Training

Training runs on GPUs through the Node addon.
A run's processes join a mesh once, and a trainer borrows it until freed.

```js
import { Executor, Mesh, Trainer, Objective, Modality } from "@unum-cloud/uform";

const gpus = Executor.forVram();
const mesh = Mesh.bootstrap(gpus, { hostDomainIndex: 0, hostDomainCount: 2, directory: "/shared", job: "run-1" });
const trainer = Trainer.initVram(
  gpus,
  { depth: 12, hidden: 512, batch: { maxRowsPerDevice: 64 }, objective: Objective.Joint, schedule: { steps: 100000 } },
  mesh,
);
const { stats } = trainer.step({
  modalities: [Modality.Text, Modality.Image],
  contents: ["a small red panda", jpeg],
  relations: { attract: { offsets: [0, 1, 2], ids: [1, 0] } },
});
await trainer.deferred().save("checkpoint.safetensors"); // or: call.wait()
```

A `lateInteraction` term trains the token head over items whose `sides` name them queries or documents, and a `choices` term trains the byte decoder on each item's legal bytes and their target probabilities:

```js
const reranker = Trainer.initVram(gpus, {
  depth: 12,
  hidden: 512,
  objective: Objective.Contrastive,
  schedule: { steps: 100000 },
  batch: { maxRowsPerDevice: 64, maxChoicesPerItem: 16 },
  lateInteraction: { weight: 1, pairLoss: PairLossForm.Ranked },
  choices: { weight: 1 },
});
reranker.step({
  modalities: [Modality.Text, Modality.Text],
  contents: ["red panda diet", "Red pandas eat bamboo."],
  sides: [LateInteractionSide.Query, LateInteractionSide.Document],
  choices: { offsets: [0, 2, 2], bytes: [0x79, 0x6e], targets: [0.9, 0.1] },
  relations: { attract: { offsets: [0, 1, 2], ids: [1, 0] } },
});
```

## Errors

Every error carries a `code` from the `Status` enum, spelled as the C library names it, so a caller branches on the identity of the fault:

```js
try {
  Model.loadRam("missing.safetensors");
} catch (error) {
  if (error.code === Status.Io) console.log("no such checkpoint");
}
```

## Design

The SDK is a thin layer over the C interface, one module per header: `types`, `model`, `encode` and `train`.

- One method per exported verb, spelled as in C: `textEncode` calls `uform_text_encode`.
- The receiver chooses and the verb stays: `model.textEncode` runs on the calling thread, `model.on(executor).textEncode` blocks on an executor, and `model.deferred(executor).textEncode` answers a `Deferred`.
- A `Deferred` is `PromiseLike` and `Disposable`: `await` waits on the libuv pool, `wait` blocks the calling thread, `status` polls, and disposing one waits until it is final.
- Every output verb takes an arena and answers views borrowing it, detached when the arena rewinds or is freed.
- Statuses, scalars, moods, modalities and multimem tiers are enums read from the C name tables; objectives and negatives carry the C numbers.
- CPU and GPU classes cross as the strings the C name tables spell, and pair-loss forms, late-interaction sides and GPU vendors as numeric enums of the C numbers.
- The SDK starts no threads and keeps no queues or caches; the only thread it borrows is the libuv worker an `await` runs its wait on.
- A progress handler runs only on the calling thread, so only a model's own verbs take one.
- Blocking calls look as they would without asynchrony, deferred calls reach the same verbs through the language's own idioms, and neither costs anything until used.
- The browser build lays every argument structure into the module's heap at offsets computed once from the C headers, runs it with a `NULL` executor, and copies its outputs out before the next call can grow the heap.
- The suite runs against the stand-in library, and against the real library with the tiny checkpoint.
