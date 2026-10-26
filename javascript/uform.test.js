// The binding suite: the Node addon first, then the WebAssembly module under Node, both over whichever C library the
// build linked. The checkpoint is the one `UFORM_TEST_CHECKPOINT` names, or else stand-in bytes only the stub loads, so
// widths come from the model's description and only text, where every byte is a token, pins exact token counts.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const zlib = require("node:zlib");
const uform = require("./dist/cjs/uform.js");

const { Arena, Executor, Model, Outputs, Scalar, Status } = uform;
const named = process.env.UFORM_TEST_CHECKPOINT;
const checkpointPath = named || path.join(fs.mkdtempSync(path.join(os.tmpdir(), "uform-")), "model.safetensors");
if (!named) fs.writeFileSync(checkpointPath, Buffer.from("a stand-in checkpoint"));
const described = Model.loadRam(checkpointPath);
const { dimensions, context, tokenWidth, choiceTemperatureGroups } = described.describe();
described.free();

// Whether `verb` throws an error whose `code` is `status`.
function throwsStatus(verb, status) {
  assert.throws(verb, (error) => error.code === status, `expected ${status}`);
}

// An RGBA image of `width` x `height` pixels, shaped as a canvas `ImageData`, filled from `seed`.
function imageData(width, height, seed) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let position = 0; position < data.length; position++) data[position] = (position * 31 + seed) & 0xff;
  return { data, width, height };
}

// The same pixels as interleaved RGB, which the verb reads as the same image.
function rgbOf(image) {
  const rgb = new Uint8Array(image.width * image.height * 3);
  for (let pixel = 0; pixel < image.width * image.height; pixel++)
    rgb.set(image.data.subarray(pixel * 4, pixel * 4 + 3), pixel * 3);
  return { data: rgb, width: image.width, height: image.height };
}

// A valid 8 x 8 RGB PNG of a gradient.
function png() {
  const chunk = (kind, data) => {
    const body = Buffer.concat([Buffer.from(kind), data]);
    const framing = Buffer.alloc(8);
    framing.writeUInt32BE(data.length, 0);
    framing.writeUInt32BE(zlib.crc32(body), 4);
    return Buffer.concat([framing.subarray(0, 4), body, framing.subarray(4)]);
  };
  const rows = Buffer.alloc(8 * 25);
  for (let row = 0; row < 8; row++)
    for (let byte = 0; byte < 24; byte++) rows[row * 25 + 1 + byte] = (row * 32 + byte * 8) & 0xff;
  const header = Buffer.from([0, 0, 0, 8, 0, 0, 0, 8, 8, 2, 0, 0, 0]);
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const pixels = chunk("IDAT", zlib.deflateSync(rows));
  return Buffer.concat([signature, chunk("IHDR", header), pixels, chunk("IEND", Buffer.alloc(0))]);
}

// A valid 16 kHz mono 16-bit WAV of a sawtooth.
function wav(samples = 1600) {
  const wav = Buffer.alloc(44 + samples * 2);
  wav.write("RIFF", 0);
  wav.writeUInt32LE(36 + samples * 2, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(16000, 24);
  wav.writeUInt32LE(32000, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(samples * 2, 40);
  for (let sample = 0; sample < samples; sample++) wav.writeInt16LE((sample % 64) * 512 - 16384, 44 + sample * 2);
  return wav;
}

// Embeddings the addon computed, which the WebAssembly module must reproduce up to the rounding its own kernels add.
const reference = {};

function assertReproduces(row, expected) {
  assert.strictEqual(row.length, expected.length);
  for (let column = 0; column < row.length; column++)
    assert.ok(Math.abs(row[column] - expected[column]) <= 1e-5, `column ${column}`);
}

test("names come from the C tables", () => {
  assert.match(uform.version(), /^4\./);
  assert.strictEqual(typeof uform.capabilitiesCompiled(), "string");
  assert.strictEqual(Scalar.BF16, "bf16");
  assert.strictEqual(uform.Modality.Video, "video");
  assert.strictEqual(uform.MultimemTier.PeerRead, "peer-read");
  assert.strictEqual(uform.Objective.Joint, "joint");
});

test("executors: RAM spawns, VRAM and borrowed pools refuse with their statuses", () => {
  const executor = Executor.forRam({ threads: 2, deferredLimit: 4 });
  executor.drain(uform.Mood.Spinning);
  throwsStatus(() => executor.drain("dozing"), Status.InvalidArgument);
  executor.free();
  executor.free();
  throwsStatus(() => new Arena(executor), Status.InvalidArgument);
  throwsStatus(() => Executor.forVram({ devices: [0] }), Status.Device);
  throwsStatus(() => Executor.onPool(1n), Status.Unsupported);
  assert.throws(() => new Executor(Symbol(), {}), (error) => error.code === Status.InvalidArgument);
});

test("models load from paths and buffers and describe themselves", () => {
  const model = Model.loadRam(checkpointPath);
  const description = model.describe();
  assert.ok(description.dimensions > 0 && description.context > 0);
  assert.strictEqual(description.quantization, Scalar.BF16);
  const buffered = Model.loadRamBuffer(fs.readFileSync(checkpointPath), { quantization: Scalar.E4M3 });
  assert.strictEqual(buffered.describe().quantization, Scalar.E4M3);
  assert.strictEqual(buffered.describe().dimensions, dimensions);
  throwsStatus(() => Model.loadRam(checkpointPath + ".missing"), Status.Io);
  throwsStatus(() => Model.loadRam(checkpointPath, { quantization: Scalar.F32 }), Status.Unsupported);
  throwsStatus(() => Model.loadVram(checkpointPath), Status.Device);
  buffered.free();
  throwsStatus(() => buffered.describe(), Status.InvalidArgument);
  model.free();
});

test("text lands embeddings and states with Arrow offsets and stats", () => {
  using model = Model.loadRam(checkpointPath);
  using arena = new Arena();
  const texts = ["hello", "", "a".repeat(context + 88)];
  const encoded = model.textEncode(texts, arena, { outputs: Outputs.Embeddings | Outputs.States });
  assert.deepStrictEqual(encoded.stats, { truncated: 1, tokens: 5 + 0 + context });
  assert.strictEqual(encoded.embeddings.rows, 3);
  assert.strictEqual(encoded.embeddings.columns, dimensions);
  assert.ok(encoded.embeddings.data instanceof Float32Array);
  assert.strictEqual(encoded.embeddings.data.length, 3 * dimensions);
  assert.deepStrictEqual([...encoded.states.offsets], [0n, 5n, 5n, BigInt(5 + context)]);
  assert.strictEqual(encoded.states.rows.rows, 5 + context);
  assert.strictEqual(encoded.states.item(0).length, 5 * dimensions);
  assert.strictEqual(encoded.states.item(1).length, 0);
  reference.hello = Array.from(encoded.embeddings.row(0));
  const alone = model.textEncode(["hello"], new Arena());
  assert.strictEqual(alone.states, undefined);
  assert.deepStrictEqual(Array.from(alone.embeddings.row(0)), reference.hello);
  const packed = model.textEncode({ data: Buffer.from("hello"), offsets: Uint32Array.of(0, 5) }, new Arena());
  assert.deepStrictEqual(Array.from(packed.embeddings.row(0)), reference.hello);
  throwsStatus(() => model.textEncode({ data: Buffer.from("hello"), offsets: [0, 5] }, arena), Status.InvalidArgument);
  throwsStatus(() => model.textEncode(["hello"], arena, { outputs: 0 }), Status.InvalidArgument);
  throwsStatus(() => model.textEncode(["hello"], arena, { outputsScalar: Scalar.I8 }), Status.Unsupported);
  const halves = model.textEncode(["hello"], arena, { outputsScalar: Scalar.BF16 });
  assert.ok(halves.embeddings.data instanceof Uint16Array);
  assert.strictEqual(halves.embeddings.stride, dimensions * 2);
});

test("every call rewinds its arena, detaching the views of the call before it", () => {
  using model = Model.loadRam(checkpointPath);
  const arena = new Arena();
  const kept = new Arena();
  const first = model.textEncode(["first"], arena);
  const elsewhere = model.textEncode(["elsewhere"], kept);
  const view = first.embeddings.data;
  model.textEncode(["second"], arena);
  assert.strictEqual(view.length, 0);
  throwsStatus(() => first.embeddings.data, Status.InvalidArgument);
  assert.strictEqual(elsewhere.embeddings.data.length, dimensions);
  arena.free();
  arena.free();
  throwsStatus(() => model.textEncode(["third"], arena), Status.InvalidArgument);
  kept.free();
  throwsStatus(() => elsewhere.embeddings.data, Status.InvalidArgument);
});

test("receivers choose completion and the verb stays: blocking and deferred agree", async () => {
  using executor = Executor.forRam();
  using model = Model.loadRam(checkpointPath, { executor });
  using arena = new Arena(executor);
  const blocking = model.on(executor).textEncode(["hello"], arena);
  assert.deepStrictEqual(Array.from(blocking.embeddings.data), reference.hello);
  assert.strictEqual(model.on(executor).describe().dimensions, dimensions);

  const call = model.deferred(executor).textEncode(["hello"], arena);
  assert.strictEqual(call.status, Status.Deferred);
  const awaited = await call;
  assert.strictEqual(call.status, Status.Success);
  assert.deepStrictEqual(Array.from(awaited.embeddings.data), reference.hello);
  assert.strictEqual(call.wait(), awaited);
  // A packed tape whose offsets are as wide as `size_t` is lent, so the deferred call holds both arrays.
  const tape = { data: new TextEncoder().encode("hello"), offsets: BigUint64Array.of(0n, 5n) };
  const lent = await model.deferred(executor).textEncode(tape, arena);
  assert.deepStrictEqual(Array.from(lent.embeddings.data), reference.hello);

  const waited = model.deferred(executor).describe().wait();
  assert.strictEqual(waited.dimensions, dimensions);
  throwsStatus(() => model.deferred(undefined), Status.InvalidArgument);
});

test("a deferred call holds its arena until final, so a second one is contended", async () => {
  using executor = Executor.forRam();
  using model = Model.loadRam(checkpointPath);
  using arena = new Arena(executor);
  const first = model.deferred(executor).textEncode(["first"], arena);
  const second = model.deferred(executor).textEncode(["second"], arena);
  assert.strictEqual(second.status, Status.Contended);
  await assert.rejects(Promise.resolve(second), (error) => error.code === Status.Contended);
  const landed = await first;
  assert.strictEqual(landed.embeddings.rows, 1);
  executor.drain();
});

test("a full deferred table refuses the next submission", () => {
  using executor = Executor.forRam({ deferredLimit: 1 });
  using model = Model.loadRam(checkpointPath);
  const first = model.deferred(executor).textEncode(["one"], new Arena(executor));
  const second = model.deferred(executor).textEncode(["two"], new Arena(executor));
  assert.strictEqual(second.status, Status.Contended);
  assert.strictEqual(first.wait().embeddings.rows, 1);
});

test("pixels take ImageData and RGB alike", () => {
  using model = Model.loadRam(checkpointPath);
  using arena = new Arena();
  const rgba = imageData(17, 9, 3);
  const encoded = model.imageEncodePixels([rgba, rgbOf(rgba)], arena, { outputs: Outputs.States });
  const [, first, both] = encoded.states.offsets;
  assert.ok(first > 0n && both === 2n * first);
  assert.deepStrictEqual(encoded.stats, { truncated: 0, tokens: Number(both) });
  const embedded = model.imageEncodePixels([rgba, rgbOf(rgba)], arena);
  assert.deepStrictEqual(Array.from(embedded.embeddings.row(0)), Array.from(embedded.embeddings.row(1)));
  reference.pixels = Array.from(embedded.embeddings.row(0));
  throwsStatus(() => model.imageEncodePixels([{ data: new Uint8Array(5), width: 2, height: 2 }], arena), Status.ShapeMismatch);
});

test("samples and encoded files count their tokens", () => {
  using model = Model.loadRam(checkpointPath);
  using arena = new Arena();
  const samples = model.audioEncodeSamples([new Float32Array(1000), new Float32Array(256)], 16000, arena, {
    outputs: Outputs.Embeddings | Outputs.States,
  });
  const [, longer, both] = samples.states.offsets;
  assert.ok(longer >= both - longer && both - longer > 0n);
  assert.deepStrictEqual(samples.stats, { truncated: 0, tokens: Number(both) });
  reference.samples = Array.from(samples.embeddings.row(0));
  throwsStatus(() => model.audioEncodeSamples([new Float32Array(4)], 0, arena), Status.InvalidArgument);
  for (const [verb, file] of [["imageEncode", png()], ["audioEncode", wav()]]) {
    const encoded = model[verb]([file, file], arena, { outputs: Outputs.States });
    const [, first, twice] = encoded.states.offsets;
    assert.ok(first > 0n && twice === 2n * first, verb);
    assert.strictEqual(encoded.stats.tokens, Number(twice), verb);
  }
  // The engine fails an undecodable file whole as malformed; the stub decodes nothing, so it encodes any bytes.
  let video;
  try {
    video = model.videoEncode([new Uint8Array(33)], arena);
  } catch (error) {
    assert.strictEqual(error.code, Status.Malformed);
  }
  if (video) assert.strictEqual(video.embeddings.rows, 1);
});

test("progress runs on the calling thread, and stops or rethrows", () => {
  using model = Model.loadRam(checkpointPath);
  using arena = new Arena();
  const heard = [];
  model.textEncode(["a", "b", "c", "d"], arena, { progress: (completed, total) => void heard.push([completed, total]) });
  assert.deepStrictEqual(heard.at(-1), [4, 4]);
  throwsStatus(() => model.textEncode(["a", "b"], arena, { progress: () => false }), Status.Cancelled);
  assert.throws(
    () =>
      model.textEncode(["a", "b"], arena, {
        progress: () => {
          throw new RangeError("from the handler");
        },
      }),
    RangeError,
  );
  using executor = Executor.forRam();
  throwsStatus(() => model.on(executor).textEncode(["a"], arena, { progress: () => true }), Status.InvalidArgument);
});

test("classes pin by their C names, and the GPU listing answers", () => {
  for (const device of uform.gpuDevices()) assert.strictEqual(typeof device.gpuClass, "string");
  using model = Model.loadRam(checkpointPath, { cpuClass: "serial" });
  const description = model.describe();
  assert.strictEqual(description.cpuClass, "serial");
  assert.strictEqual(description.gpuClass, null);
  throwsStatus(() => Model.loadRam(checkpointPath, { cpuClass: "intel-pentium" }), Status.UnknownKind);
  Executor.forRam({ cpuClass: "serial" }).free();
});

test("token vectors are unit rows beside the states, or unsupported without a token head", () => {
  using model = Model.loadRam(checkpointPath);
  using arena = new Arena();
  const outputs = Outputs.States | Outputs.TokenVectors;
  if (!tokenWidth) return throwsStatus(() => model.textEncode(["hello"], arena, { outputs }), Status.Unsupported);
  const encoded = model.textEncode(["hello", "a".repeat(context + 4)], arena, { outputs });
  assert.strictEqual(encoded.tokenVectors.rows.columns, tokenWidth);
  assert.deepStrictEqual([...encoded.tokenVectors.offsets], [...encoded.states.offsets]);
  for (let row = 0; row < encoded.tokenVectors.rows.rows; row++) {
    const squares = encoded.tokenVectors.rows.row(row).reduce((sum, value) => sum + value * value, 0);
    assert.ok(squares > 0.98 && squares < 1.02, `row ${row}`);
  }
});

test("choices sum to one per item, or unsupported without the byte decoder", () => {
  using model = Model.loadRam(checkpointPath);
  using arena = new Arena();
  const texts = ["keep this candidate?", "expand which neighbour?", "a".repeat(context + 4)];
  const choices = ["yn", "ABC", "s"];
  if (!choiceTemperatureGroups)
    return throwsStatus(() => model.textChoose(texts, choices, arena), Status.Unsupported);
  const chosen = model.textChoose(texts, choices, arena, { temperatureGroups: [0, 0, 0] });
  assert.deepStrictEqual([...chosen.probabilities.offsets], [0n, 2n, 5n, 6n]);
  for (let item = 0; item < texts.length; item++) {
    const sum = chosen.probabilities.item(item).reduce((total, value) => total + value, 0);
    assert.ok(Math.abs(sum - 1) < 1e-3, `item ${item}`);
  }
  const past = { temperatureGroups: [0, choiceTemperatureGroups, 0] };
  throwsStatus(() => model.textChoose(texts, choices, arena, past), Status.InvalidArgument);
});

test("training reports device on a build without GPUs", () => {
  using executor = Executor.forRam();
  throwsStatus(() => uform.Mesh.bootstrap(executor), Status.Device);
  const config = { depth: 2, hidden: 128, batch: { maxRowsPerDevice: 4 }, objective: uform.Objective.Masked, schedule: { steps: 1 } };
  throwsStatus(() => uform.Trainer.initVram(executor, config), Status.Device);
  const heads = {
    ...config,
    batch: { maxRowsPerDevice: 4, maxChoicesPerItem: 16 },
    lateInteraction: { weight: 1, pairLoss: uform.PairLossForm.Ranked },
    choices: { weight: 1, brierWeight: 0.5 },
  };
  throwsStatus(() => uform.Trainer.initVram(executor, heads), Status.Device);
  throwsStatus(() => uform.Trainer.initVram(executor, { ...heads, choices: { weight: "heavy" } }), Status.InvalidArgument);
});

const wasmModule = path.join(__dirname, "dist", "wasm", "uform.mjs");
const wasmSkip = fs.existsSync(wasmModule) ? false : "run `npm run build-wasm` first";

test("wasm: text reproduces the addon's embeddings through heap-laid structures", { skip: wasmSkip }, async () => {
  const wasm = require("./dist/cjs/wasm.js");
  await wasm.initialize();
  assert.strictEqual(wasm.version(), uform.version());
  const model = wasm.Model.loadRamBuffer(fs.readFileSync(checkpointPath));
  assert.strictEqual(model.describe().dimensions, dimensions);
  const arena = new wasm.Arena();
  const encoded = model.textEncode(["hello", "", "a".repeat(context + 88)], arena, {
    outputs: wasm.Outputs.Embeddings | wasm.Outputs.States,
  });
  assert.deepStrictEqual(encoded.stats, { truncated: 1, tokens: 5 + context });
  assertReproduces(encoded.embeddings.row(0), reference.hello);
  assert.deepStrictEqual([...encoded.states.offsets], [0n, 5n, 5n, BigInt(5 + context)]);
  assert.strictEqual(encoded.states.item(2).length, context * dimensions);
  const view = encoded.embeddings.data;
  model.textEncode(["again"], arena);
  assert.strictEqual(view.byteLength, 0);
  throwsStatus(() => encoded.embeddings.data, Status.InvalidArgument);
});

test("wasm: pixels, samples, receivers and refusals", { skip: wasmSkip }, async () => {
  const wasm = require("./dist/cjs/wasm.js");
  const model = wasm.Model.loadRamBuffer(new Uint8Array(fs.readFileSync(checkpointPath)).buffer);
  const arena = new wasm.Arena();
  const rgba = imageData(17, 9, 3);
  const pixels = model.imageEncodePixels([rgba, rgbOf(rgba)], arena, { outputsScalar: Scalar.F32 });
  assertReproduces(pixels.embeddings.row(1), reference.pixels);
  const samples = model.audioEncodeSamples([new Float32Array(1000), new Float32Array(256)], 16000, arena);
  assertReproduces(samples.embeddings.row(0), reference.samples);

  const executor = wasm.Executor.forRam();
  const call = model.deferred(executor).textEncode(["hello"], arena);
  assert.strictEqual(call.status, Status.Success);
  assertReproduces((await call).embeddings.data, reference.hello);
  assert.strictEqual(model.on(executor).describe().context, context);

  const heard = [];
  model.textEncode(["a", "b", "c"], arena, { progress: (completed) => void heard.push(completed) });
  assert.deepStrictEqual(heard, [1, 2, 3]);
  throwsStatus(() => model.textEncode(["a", "b"], arena, { progress: () => false }), Status.Cancelled);
  throwsStatus(() => model.imageEncode([new Uint8Array(4)], arena), Status.Unsupported);
  throwsStatus(() => wasm.Model.loadRam(checkpointPath), Status.Unsupported);
  throwsStatus(() => wasm.Executor.forVram(), Status.Unsupported);
  throwsStatus(() => wasm.Trainer.initVram(executor, { objective: uform.Objective.Masked }), Status.Unsupported);
  throwsStatus(() => model.textEncode(["a"], arena, { outputs: 0 }), Status.InvalidArgument);
  const { tokenWidth: heads, choiceTemperatureGroups: groups } = model.describe();
  assert.deepStrictEqual([heads, groups], [tokenWidth, choiceTemperatureGroups]);
  if (!groups) throwsStatus(() => model.textChoose(["stop?"], ["yn"], arena), Status.Unsupported);
  else assert.strictEqual(model.textChoose(["stop?"], ["yn"], arena).probabilities.data.length, 2);
  arena.free();
  model.free();
  throwsStatus(() => model.describe(), Status.InvalidArgument);
});
