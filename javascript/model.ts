/**
 *  @file javascript/model.ts
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The model lifecycle: loads into RAM or VRAM, the description, and the receivers every
 *      encode verb runs through.
 */
import type { CompiledDescription, CompiledModel } from "./library.js";
import {
  type ChooseConfig,
  type Chosen,
  type EncodeConfig,
  type Encoded,
  type Pixels,
  type Tape,
  modelAudioEncode,
  modelAudioEncodeSamples,
  modelImageEncode,
  modelImageEncodePixels,
  modelTextChoose,
  modelTextEncode,
  modelVideoEncode,
} from "./encode.js";
import {
  type Arena,
  type Call,
  type Deferred,
  Executor,
  Scalar,
  Status,
  backend,
  block,
  callOf,
  defer,
  fail,
  scalarCode,
} from "./types.js";

/** Where a checkpoint loaded into host memory lands. Mirrors `uform_model_load_ram_t`. */
export interface ModelLoadRamConfig {
  /** The executor the load runs on; absent loads on the calling thread. */
  executor?: Executor;
  /** NUMA node IDs to place the weights in; absent derives every node. */
  memoryDomains?: number[];
  /** The newest CPU class to pack the tower for, spelled like "intel-haswell": the load runs
   *  the newest compiled class the CPU runs at or under it; absent sets no ceiling. */
  cpuClassCeiling?: string;
  /** The scalar the transformer blocks run in: `bf16` as the checkpoint stores it, the
   *  default, or `e4m3`. */
  quantization?: Scalar;
}

/** Where a checkpoint loaded into device memory lands. Mirrors `uform_model_load_vram_t`. */
export interface ModelLoadVramConfig {
  /** The executor the upload runs on, one stream per device; absent uploads on
   *  the calling thread. */
  executor?: Executor;
  /** Device ordinals, positions in `gpuDevices()`, to replicate the weights on; absent takes
   *  one free device. */
  devices?: number[];
  /** The newest GPU class every replica runs as, spelled like "nvidia-hopper"; absent sets no
   *  ceiling. */
  gpuClassCeiling?: string;
  /** The scalar the transformer blocks run in: `bf16` as the checkpoint stores it, the
   *  default, or `e4m3`. */
  quantization?: Scalar;
}

/** What a loaded model reports about itself. Mirrors `uform_model_description_t`. */
export interface ModelDescription {
  /** Scalars per embedding row; Matryoshka prefixes of it are embeddings too. */
  dimensions: number;
  /** Tokens one item may span before it is truncated. */
  context: number;
  /** The scalar the transformer blocks run in. */
  quantization: Scalar;
  /** Bytes the model holds; activations are the executor's. */
  memoryUsage: number;
  /** The CPU class a RAM model's tower was packed for, like "intel-haswell"; `null` for
   *  a VRAM model. */
  cpuClass: string | null;
  /** The GPU class every replica of a VRAM model runs as, like "nvidia-hopper"; `null` for
   *  a RAM model. */
  gpuClass: string | null;
  /** Scalars per token vector; zero when the checkpoint has no token head. */
  tokenWidth: number;
  /** Question types the choice temperatures were calibrated over: zero without the byte
   *  decoder, one uncalibrated. */
  choiceTemperatureGroups: number;
}

/** The body of every receiver's `describe`. */
function modelDescribe(model: CompiledModel, executor: Executor | undefined): Call<ModelDescription> {
  return callOf(
    (completion) => model.describe(executor?.compiled, completion),
    (described: CompiledDescription) => ({ ...described, quantization: described.quantization as Scalar }),
  );
}

/** The verbs of one model, blocking on one executor: `model.on(executor)`. Every call returns
 *  with its outputs written, and one that finds another call holding the executor throws
 *  `contended` without running. */
export class ModelOn {
  readonly #model: CompiledModel;
  /** The executor every verb runs on. */
  readonly executor: Executor;

  /** Made by `model.on(executor)`, never directly. */
  constructor(model: CompiledModel, executor: Executor) {
    this.#model = model;
    this.executor = executor;
  }

  /** {@inheritDoc Model.describe} */
  describe(): ModelDescription {
    return block(modelDescribe(this.#model, this.executor));
  }

  /** {@inheritDoc Model.textEncode} */
  textEncode(texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelTextEncode(this.#model, this.executor, texts, arena, config));
  }

  /** {@inheritDoc Model.textChoose} */
  textChoose(
    texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    choices: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    arena: Arena,
    config: ChooseConfig = {},
  ): Chosen {
    return block(modelTextChoose(this.#model, this.executor, texts, choices, arena, config));
  }

  /** {@inheritDoc Model.imageEncode} */
  imageEncode(images: readonly Uint8Array[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelImageEncode(this.#model, this.executor, images, arena, config));
  }

  /** {@inheritDoc Model.imageEncodePixels} */
  imageEncodePixels(images: readonly Pixels[], arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelImageEncodePixels(this.#model, this.executor, images, arena, config));
  }

  /** {@inheritDoc Model.videoEncode} */
  videoEncode(videos: readonly Uint8Array[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelVideoEncode(this.#model, this.executor, videos, arena, config));
  }

  /** {@inheritDoc Model.audioEncode} */
  audioEncode(recordings: readonly Uint8Array[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelAudioEncode(this.#model, this.executor, recordings, arena, config));
  }

  /** {@inheritDoc Model.audioEncodeSamples} */
  audioEncodeSamples(
    recordings: readonly Float32Array[] | Tape<Float32Array>,
    sampleRate: number,
    arena: Arena,
    config: EncodeConfig = {},
  ): Encoded {
    return block(modelAudioEncodeSamples(this.#model, this.executor, recordings, sampleRate, arena, config));
  }
}

/** The verbs of one model, submitted to one executor and answered as {@link Deferred} calls:
 *  `model.deferred(executor)`. A verb writes into the caller's arena until it is final, and waits
 *  into the same views a blocking call answers. */
export class ModelDeferredOn {
  readonly #model: CompiledModel;
  /** The executor every verb is submitted to. */
  readonly executor: Executor;

  /** Made by `model.deferred(executor)`, never directly. */
  constructor(model: CompiledModel, executor: Executor) {
    this.#model = model;
    this.executor = executor;
  }

  /** {@inheritDoc Model.describe} */
  describe(): Deferred<ModelDescription> {
    return defer(modelDescribe(this.#model, this.executor));
  }

  /** {@inheritDoc Model.textEncode} */
  textEncode(
    texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Deferred<Encoded> {
    return defer(modelTextEncode(this.#model, this.executor, texts, arena, config));
  }

  /** {@inheritDoc Model.textChoose} */
  textChoose(
    texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    choices: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    arena: Arena,
    config: ChooseConfig = {},
  ): Deferred<Chosen> {
    return defer(modelTextChoose(this.#model, this.executor, texts, choices, arena, config));
  }

  /** {@inheritDoc Model.imageEncode} */
  imageEncode(
    images: readonly Uint8Array[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Deferred<Encoded> {
    return defer(modelImageEncode(this.#model, this.executor, images, arena, config));
  }

  /** {@inheritDoc Model.imageEncodePixels} */
  imageEncodePixels(images: readonly Pixels[], arena: Arena, config: EncodeConfig = {}): Deferred<Encoded> {
    return defer(modelImageEncodePixels(this.#model, this.executor, images, arena, config));
  }

  /** {@inheritDoc Model.videoEncode} */
  videoEncode(
    videos: readonly Uint8Array[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Deferred<Encoded> {
    return defer(modelVideoEncode(this.#model, this.executor, videos, arena, config));
  }

  /** {@inheritDoc Model.audioEncode} */
  audioEncode(
    recordings: readonly Uint8Array[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Deferred<Encoded> {
    return defer(modelAudioEncode(this.#model, this.executor, recordings, arena, config));
  }

  /** {@inheritDoc Model.audioEncodeSamples} */
  audioEncodeSamples(
    recordings: readonly Float32Array[] | Tape<Float32Array>,
    sampleRate: number,
    arena: Arena,
    config: EncodeConfig = {},
  ): Deferred<Encoded> {
    return defer(modelAudioEncodeSamples(this.#model, this.executor, recordings, sampleRate, arena, config));
  }
}

/** The token the loads hold, which is what a direct `new Model` cannot produce. */
const modelSeal = Symbol("Model");

/** Views a checkpoint held in memory as the bytes a backend reads. */
function bufferOf(buffer: ArrayBuffer | ArrayBufferView): ArrayBufferView {
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  if (ArrayBuffer.isView(buffer)) return buffer;
  throw fail(Status.InvalidArgument, "A checkpoint buffer is an ArrayBuffer or a TypedArray.");
}

/**
 *  A loaded encoder, running every verb on the calling thread. `on(executor)` and
 *  `deferred(executor)` reach the same verbs on an executor. One model serves every modality it
 *  was trained on.
 */
export class Model implements Disposable {
  readonly #model: CompiledModel;

  /** Reached through the loads. */
  private constructor(seal: symbol, compiledModel: CompiledModel) {
    if (seal !== modelSeal)
      throw fail(Status.InvalidArgument, "A Model is opened by Model.loadRam, loadRamBuffer, loadVram or loadVramBuffer.");
    this.#model = compiledModel;
  }

  /**
   *  The same verbs, each blocking on `executor`; one that finds another call holding it throws
   *  `contended`. Binding-only.
   *  @param executor - The threads or streams to run on; its residency must match the model's.
   */
  on(executor: Executor): ModelOn {
    return new ModelOn(this.#model, executor);
  }

  /**
   *  The same verbs, each submitted to `executor` and answered as a {@link Deferred}. Binding-only.
   *  @param executor - The threads or streams to run on; its residency must match the model's.
   */
  deferred(executor: Executor): ModelDeferredOn {
    if (!(executor instanceof Executor)) throw fail(Status.InvalidArgument, "A deferred receiver needs an Executor.");
    return new ModelDeferredOn(this.#model, executor);
  }

  /** Reports the model's geometry and placement in one call. Calls `uform_model_describe`. */
  describe(): ModelDescription {
    return block(modelDescribe(this.#model, undefined));
  }

  /**
   *  Encodes texts through one forward; every UTF-8 byte is a token. Calls `uform_text_encode`.
   *
   *  @param texts - Strings, UTF-8 bytes already encoded, or a tape of them.
   *  @param arena - The memory the outputs land in, rewound by the call.
   *  @param config - The surfaces, their scalar, and a progress handler.
   *  @returns The outputs, viewed inside `arena`.
   */
  textEncode(texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelTextEncode(this.#model, undefined, texts, arena, config));
  }

  /**
   *  Scores each item's legal choices through one forward, a softmax over them alone at the item's
   *  group's calibrated inverse temperature; a model without the byte decoder reports
   *  `unsupported`. Calls `uform_text_choose`.
   *
   *  @param texts - The states, each ending where its choice goes.
   *  @param choices - Each item's legal choices, one byte each, like "yn".
   *  @param arena - The memory the probabilities land in, rewound by the call.
   *  @param config - The question types and a progress handler.
   *  @returns The probabilities, viewed inside `arena`.
   */
  textChoose(
    texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    choices: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    arena: Arena,
    config: ChooseConfig = {},
  ): Chosen {
    return block(modelTextChoose(this.#model, undefined, texts, choices, arena, config));
  }

  /**
   *  Decodes and encodes JPEG, PNG or WebP files; the browser build reports
   *  `unsupported`. Calls `uform_image_encode`.
   *
   *  @param images - One encoded file per item, or a tape of them.
   *  @param arena - The memory the outputs land in, rewound by the call.
   *  @param config - The surfaces, their scalar, and a progress handler.
   */
  imageEncode(images: readonly Uint8Array[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelImageEncode(this.#model, undefined, images, arena, config));
  }

  /**
   *  Encodes decoded images, resized on the executor's side. Calls `uform_image_encode_pixels`.
   *
   *  @param images - `ImageData`, or RGB or RGBA bytes with their width and height.
   *  @param arena - The memory the outputs land in, rewound by the call.
   *  @param config - The surfaces, their scalar, and a progress handler.
   */
  imageEncodePixels(images: readonly Pixels[], arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelImageEncodePixels(this.#model, undefined, images, arena, config));
  }

  /**
   *  Decodes and encodes MP4 or WebM clips; the browser build reports
   *  `unsupported`. Calls `uform_video_encode`.
   *
   *  @param videos - One encoded file per item, or a tape of them.
   *  @param arena - The memory the outputs land in, rewound by the call.
   *  @param config - The surfaces, their scalar, and a progress handler.
   */
  videoEncode(videos: readonly Uint8Array[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelVideoEncode(this.#model, undefined, videos, arena, config));
  }

  /**
   *  Decodes and encodes WAV, FLAC, MP3 or Opus files; the browser build reports
   *  `unsupported`. Calls `uform_audio_encode`.
   *
   *  @param recordings - One encoded file per item, or a tape of them.
   *  @param arena - The memory the outputs land in, rewound by the call.
   *  @param config - The surfaces, their scalar, and a progress handler.
   */
  audioEncode(recordings: readonly Uint8Array[] | Tape<Uint8Array>, arena: Arena, config: EncodeConfig = {}): Encoded {
    return block(modelAudioEncode(this.#model, undefined, recordings, arena, config));
  }

  /**
   *  Encodes decoded mono recordings, resampled to the model's rate where `sampleRate`
   *  differs. Calls `uform_audio_encode_samples`.
   *
   *  @param recordings - One run of samples per item, or a tape of them.
   *  @param sampleRate - Samples per second shared by every recording.
   *  @param arena - The memory the outputs land in, rewound by the call.
   *  @param config - The surfaces, their scalar, and a progress handler.
   */
  audioEncodeSamples(
    recordings: readonly Float32Array[] | Tape<Float32Array>,
    sampleRate: number,
    arena: Arena,
    config: EncodeConfig = {},
  ): Encoded {
    return block(modelAudioEncodeSamples(this.#model, undefined, recordings, sampleRate, arena, config));
  }

  /** Releases the weights now, waiting for every deferred call naming the model; a second `free`
   *  does nothing. Calls `uform_model_free`. */
  free(): void {
    this.#model.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  /**
   *  Copies a safetensors checkpoint file into host memory and packs it for the
   *  CPU. Calls `uform_model_load_ram`.
   *  @param path - The checkpoint, read in full and closed before the load completes.
   *  @param config - The executor, the NUMA nodes, and the block scalar.
   */
  static loadRam(path: string, config: ModelLoadRamConfig = {}): Model {
    return new Model(
      modelSeal,
      backend().CompiledModel.loadRam(
        path,
        config.executor?.compiled,
        config.memoryDomains,
        scalarCode(config.quantization ?? Scalar.BF16),
        config.cpuClassCeiling,
      ),
    );
  }

  /**
   *  Copies a checkpoint held in memory into host memory, which is how a page loads one it
   *  fetched. Calls `uform_model_load_ram_buffer`.
   *  @param buffer - The checkpoint bytes, copied before the load completes.
   *  @param config - The executor, the NUMA nodes, and the block scalar.
   */
  static loadRamBuffer(buffer: ArrayBuffer | ArrayBufferView, config: ModelLoadRamConfig = {}): Model {
    return new Model(
      modelSeal,
      backend().CompiledModel.loadRamBuffer(
        bufferOf(buffer),
        config.executor?.compiled,
        config.memoryDomains,
        scalarCode(config.quantization ?? Scalar.BF16),
        config.cpuClassCeiling,
      ),
    );
  }

  /**
   *  Copies a checkpoint file onto the named GPUs and packs it
   *  there. Calls `uform_model_load_vram`.
   *  @param path - The checkpoint, read in full and closed before the load completes.
   *  @param config - The executor, the devices, and the block scalar.
   */
  static loadVram(path: string, config: ModelLoadVramConfig = {}): Model {
    return new Model(
      modelSeal,
      backend().CompiledModel.loadVram(
        path,
        config.executor?.compiled,
        config.devices,
        scalarCode(config.quantization ?? Scalar.BF16),
        config.gpuClassCeiling,
      ),
    );
  }

  /**
   *  Copies a checkpoint held in memory onto the named GPUs. Calls `uform_model_load_vram_buffer`.
   *  @param buffer - The checkpoint bytes, copied before the load completes.
   *  @param config - The executor, the devices, and the block scalar.
   */
  static loadVramBuffer(buffer: ArrayBuffer | ArrayBufferView, config: ModelLoadVramConfig = {}): Model {
    return new Model(
      modelSeal,
      backend().CompiledModel.loadVramBuffer(
        bufferOf(buffer),
        config.executor?.compiled,
        config.devices,
        scalarCode(config.quantization ?? Scalar.BF16),
        config.gpuClassCeiling,
      ),
    );
  }
}
