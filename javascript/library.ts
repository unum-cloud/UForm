/**
 * The boundary both backends satisfy: the shapes the Node addon and the WebAssembly module publish, and the search that
 * locates the prebuilt addon.
 *
 * Every signature here is positional and takes no defaults, so the wrapper classes decide what an omitted argument
 * means rather than a backend guessing. Every instance verb opens with the executor and the C `completion`, and answers
 * its result when blocking or a `CompiledDeferred` when deferred. Only the Node entry imports this module at runtime;
 * the browser entry imports its types alone, so no bundle pulls `node-gyp-build` into a page.
 */
import build from "node-gyp-build";
import * as path from "path";
import { existsSync } from "fs";
import { getFileName, getRoot } from "bindings";

/** The C `uform_completion_t` a verb is submitted with: `0` blocks, `1` defers. */
export type Completion = 0 | 1;

/** Hears a long call's advance; answering `false` stops it with `cancelled`. */
export type ProgressCallback = (completed: number, total: number) => boolean | void;

/**
 * The `count + 1` offsets of a tape: a `BigUint64Array`, which the Node addon lends as `size_t` without a copy, or a
 * `Uint32Array`, which it widens and the WebAssembly module, whose `size_t` is 32 bits wide, lends as it is.
 */
export type Offsets = BigUint64Array | Uint32Array;

/** The threads or streams a verb runs on. */
export interface CompiledExecutor {
  drain(mood?: string): void;
  free(): void;
}

/** Memory every output whose length scales with a call lands in, made where its executor computes. */
export interface CompiledArena {
  readonly residency: string;
  free(): void;
}

/** A call submitted deferred, owning its outcome until it is final. */
export interface CompiledDeferred {
  status(): string;
  wait(): void;
  settled(): Promise<void>;
  result(): unknown;
}

/** Rows a call landed inside its arena; `data` is `null` over device memory. */
export interface CompiledRows {
  data: ArrayBufferView | null;
  rows: number;
  columns: number;
  stride: number;
  scalar: string;
}

/** The final-norm states or the token vectors a call landed inside its arena, with `count + 1` row offsets. */
export interface CompiledStates extends CompiledRows {
  offsets: BigUint64Array | null;
}

/** What an encode verb reports beside its outputs. */
export interface CompiledEncodeStats {
  truncated: number;
  tokens: number;
}

/** The surfaces one encode call kept, present as they were asked for. */
export interface CompiledEncoded {
  embeddings?: CompiledRows;
  states?: CompiledStates;
  tokenVectors?: CompiledStates;
  stats: CompiledEncodeStats;
}

/** One probability per choice a call landed inside its arena, with `count + 1` offsets; `null` over device memory. */
export interface CompiledChosen {
  probabilities: Float32Array | null;
  offsets: BigUint64Array | null;
  stats: CompiledEncodeStats;
}

/** What a loaded model reports about itself, in the canonical spellings. */
export interface CompiledDescription {
  dimensions: number;
  context: number;
  quantization: string;
  memoryUsage: number;
  cpuClass: string | null;
  gpuClass: string | null;
  tokenWidth: number;
  choiceTemperatureGroups: number;
}

/** One GPU this process can drive, its position in the listing being the ordinal VRAM placements name it by. */
export interface CompiledGpuDevice {
  vendor: number;
  vendorOrdinal: number;
  gpuClass: string | null;
  computeUnits: number;
  memoryBytes: number;
  name: string;
}

/** What a mesh reports about itself. */
export interface CompiledMeshDescription {
  gpusPerHostDomain: number;
  gpusPerMultimemDomain: number;
  gpusPerColocatedDomain: number;
  hostDomainIndex: number;
  firstGpuIndex: number;
  multimemTier: string;
  gpuClass: string | null;
}

/** What one optimizer step reports, and the embeddings it scored when an arena was given. */
export interface CompiledStepped {
  stats: {
    step: number;
    learningRate: number;
    loss: number;
    contrastiveLoss: number;
    maskedTextLoss: number;
    maskedImageLoss: number;
    maskedVideoLoss: number;
    maskedAudioLoss: number;
    lateInteractionLoss: number;
    choicesLoss: number;
    truncated: number;
    dropped: number;
  };
  embeddings?: CompiledRows;
}

/** The leading arguments of every instance verb. */
type On = [executor: CompiledExecutor | undefined, completion: Completion];

/** The trailing arguments of every encode verb. */
type Landing = [arena: CompiledArena, outputs: number, outputsScalar: number, progress: ProgressCallback | undefined];

/** An encode verb's answer, or a `CompiledDeferred` of it. */
type Encoding = CompiledEncoded | CompiledDeferred;

/** One model handle; every verb answers its result, or a `CompiledDeferred` of it when deferred. */
export interface CompiledModel {
  free(): void;
  describe(...on: On): CompiledDescription | CompiledDeferred;
  textEncode(...arguments_: [...On, bytes: Uint8Array, offsets: Offsets, ...Landing]): Encoding;
  imageEncode(...arguments_: [...On, bytes: Uint8Array, offsets: Offsets, ...Landing]): Encoding;
  imageEncodePixels(
    ...arguments_: [...On, rgb: Uint8Array, offsets: Offsets, heights: Uint32Array, widths: Uint32Array, ...Landing]
  ): Encoding;
  videoEncode(...arguments_: [...On, bytes: Uint8Array, offsets: Offsets, ...Landing]): Encoding;
  audioEncode(...arguments_: [...On, bytes: Uint8Array, offsets: Offsets, ...Landing]): Encoding;
  audioEncodeSamples(
    ...arguments_: [...On, samples: Float32Array, offsets: Offsets, sampleRate: number, ...Landing]
  ): Encoding;
  textChoose(
    ...arguments_: [
      ...On,
      bytes: Uint8Array,
      offsets: Offsets,
      choices: Uint8Array,
      choiceOffsets: Offsets,
      temperatureGroups: Uint32Array | undefined,
      arena: CompiledArena,
      progress: ProgressCallback | undefined,
    ]
  ): CompiledChosen | CompiledDeferred;
}

/** The model class, whose loads are the only way to a live handle. */
export interface CompiledModelConstructor {
  loadRam(
    path: string,
    executor: CompiledExecutor | undefined,
    memoryDomains: number[] | undefined,
    quantization: number,
    cpuClass: string | undefined,
  ): CompiledModel;
  loadRamBuffer(
    buffer: ArrayBufferView,
    executor: CompiledExecutor | undefined,
    memoryDomains: number[] | undefined,
    quantization: number,
    cpuClass: string | undefined,
  ): CompiledModel;
  loadVram(
    path: string,
    executor: CompiledExecutor | undefined,
    devices: number[] | undefined,
    quantization: number,
    gpuClass: string | undefined,
  ): CompiledModel;
  loadVramBuffer(
    buffer: ArrayBufferView,
    executor: CompiledExecutor | undefined,
    devices: number[] | undefined,
    quantization: number,
    gpuClass: string | undefined,
  ): CompiledModel;
}

/** One process's mesh. */
export interface CompiledMesh {
  describe(): CompiledMeshDescription;
  free(): void;
}

/** The mesh class, whose bootstrap is the only way to a live handle. */
export interface CompiledMeshConstructor {
  bootstrap(executor: CompiledExecutor, options: object): CompiledMesh;
}

/** One trainer handle. */
export interface CompiledTrainer {
  free(): void;
  step(
    ...arguments_: [
      ...On,
      modalities: Int32Array,
      contents: Uint8Array,
      offsets: Offsets,
      bags: Uint32Array | undefined,
      temperatureGroups: Uint32Array | undefined,
      sides: Int32Array | undefined,
      choices: Uint8Array | undefined,
      choiceOffsets: Offsets | undefined,
      choiceTargets: Float32Array | undefined,
      choiceTargetOffsets: Offsets | undefined,
      attractOffsets: Uint32Array | undefined,
      attractIds: Uint32Array | undefined,
      attractWeights: Float32Array | undefined,
      repelOffsets: Uint32Array | undefined,
      repelIds: Uint32Array | undefined,
      repelWeights: Float32Array | undefined,
      ignoreOffsets: Uint32Array | undefined,
      ignoreIds: Uint32Array | undefined,
      arena: CompiledArena | undefined,
      outputsScalar: number | undefined,
    ]
  ): CompiledStepped | CompiledDeferred;
  save(...arguments_: [...On, path: string]): undefined | CompiledDeferred;
}

/** The trainer class, whose creation is the only way to a live handle. */
export interface CompiledTrainerConstructor {
  initVram(executor: CompiledExecutor, mesh: CompiledMesh | undefined, config: object): CompiledTrainer;
}

/** The executor class and its factories. */
export interface CompiledExecutorConstructor {
  forRam(config?: object): CompiledExecutor;
  forVram(config?: object): CompiledExecutor;
  onPool(pool: bigint, config?: object): CompiledExecutor;
}

/** Everything a backend publishes. */
export interface Compiled {
  Executor: CompiledExecutorConstructor;
  Arena: new (executor?: CompiledExecutor) => CompiledArena;
  CompiledModel: CompiledModelConstructor;
  CompiledMesh: CompiledMeshConstructor;
  CompiledTrainer: CompiledTrainerConstructor;
  scalars: Readonly<Record<string, number>>;
  modalities: Readonly<Record<string, number>>;
  objectives: Readonly<Record<string, number>>;
  negatives: Readonly<Record<string, number>>;
  version(): string;
  capabilitiesCompiled(): string;
  capabilitiesAvailable(): string;
  gpuDevices(): CompiledGpuDevice[];
}

/** Walks up from `directory` to the first ancestor holding a build or a prebuild tree. */
function findBuildDirectory(directory: string): string {
  if (existsSync(path.join(directory, "build"))) return directory;
  if (existsSync(path.join(directory, "prebuilds"))) return directory;
  if (directory === path.dirname(directory)) throw new Error("Could not find the native build of @unum-cloud/uform");
  return findBuildDirectory(path.dirname(directory));
}

/** The directory this module was loaded from, however the bundler spelled it. */
function currentDirectory(): string {
  try {
    if (__dirname) return __dirname;
  } catch (error) {}
  return getRoot(getFileName());
}

/** Loads the Node addon, the backend the package entry binds. */
export function loadAddon(): Compiled {
  return build(findBuildDirectory(currentDirectory())) as Compiled;
}
