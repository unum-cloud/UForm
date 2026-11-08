/**
 *  @file javascript/model.ts
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The model lifecycle: loads into RAM or VRAM, the description, and the receivers every
 *      encode verb runs through.
 */
import type { CompiledDescription, CompiledModel } from "./library.js";
import { EncodeVerbs } from "./encode.js";
import { Answer, Executor, Scalar, Status, backend, fail, scalarCode } from "./types.js";

/** Where a checkpoint loaded into host memory lands. Mirrors `uform_model_load_ram_t`. */
export interface ModelLoadRamConfig {
  /** The executor the load runs on; absent loads on the calling thread. */
  executor?: Executor;
  /** NUMA node IDs to place the weights in; absent derives every node. */
  memoryDomains?: number[];
  /** The CPU class to pack the tower for, spelled like "intel-haswell"; absent derives
   *  the CPU's own. */
  cpuClass?: string;
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
  /** The GPU class every replica runs as, spelled like "nvidia-hopper"; absent derives
   *  the devices' own. */
  gpuClass?: string;
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
  /** Scalars per token vector; zero when the checkpoint carries no token head. */
  tokenWidth: number;
  /** Question types the choice temperatures were calibrated over: zero without the byte
   *  decoder, one uncalibrated. */
  choiceTemperatureGroups: number;
}

/** The verbs every receiver of a model shares: the encode verbs and the description. */
abstract class ModelVerbs<Receiving extends "blocking" | "deferred"> extends EncodeVerbs<Receiving> {
  /** Reports the model's geometry and placement in one call. Calls `uform_model_describe`. */
  describe(): Answer<Receiving, ModelDescription> {
    const executor = this.executor?.compiled;
    return this.submit(
      (completion) => this.compiled.describe(executor, completion),
      (described: CompiledDescription) => ({ ...described, quantization: described.quantization as Scalar }),
    );
  }
}

/**
 *  The verbs of one model, blocking on one executor: `model.on(executor)`. Every call starts after
 *  every earlier submission to that executor, and returns with its outputs written.
 */
export class ModelOn extends ModelVerbs<"blocking"> {
  /** Made by `model.on(executor)`, never directly. */
  constructor(compiled: CompiledModel, executor: Executor | undefined) {
    super(compiled, executor, "blocking");
  }
}

/**
 *  The verbs of one model, submitted to one executor and answered as {@link Deferred} calls:
 *  `model.deferred(executor)`. A verb borrows its arena until final, and waits into the same views
 *  a blocking call answers.
 */
export class ModelDeferredOn extends ModelVerbs<"deferred"> {
  /** Made by `model.deferred(executor)`, never directly. */
  constructor(compiled: CompiledModel, executor: Executor) {
    super(compiled, executor, "deferred");
  }
}

/** The token the loads hold, which is what a direct `new Model` cannot produce. */
const modelSeal = Symbol("Model");

/** Reads a checkpoint held in memory as the bytes a backend borrows. */
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
export class Model extends ModelOn implements Disposable {
  /** Reached through the loads. */
  private constructor(seal: symbol, compiledModel: CompiledModel) {
    if (seal !== modelSeal)
      throw fail(Status.InvalidArgument, "A Model is opened by Model.loadRam, loadRamBuffer, loadVram or loadVramBuffer.");
    super(compiledModel, undefined);
  }

  /**
   *  The same verbs, each blocking on `executor`, after every earlier submission
   *  to it. Binding-only.
   *  @param executor - The threads or streams to run on; its residency must match the model's.
   */
  on(executor: Executor): ModelOn {
    return new ModelOn(this.compiled, executor);
  }

  /**
   *  The same verbs, each submitted to `executor` and answered as a {@link Deferred}. Binding-only.
   *  @param executor - The threads or streams to run on; its residency must match the model's.
   */
  deferred(executor: Executor): ModelDeferredOn {
    if (!(executor instanceof Executor)) throw fail(Status.InvalidArgument, "A deferred receiver needs an Executor.");
    return new ModelDeferredOn(this.compiled, executor);
  }

  /** Releases the weights now, waiting for every deferred call naming the model; a second `free`
   *  does nothing. Calls `uform_model_free`. */
  free(): void {
    this.compiled.free();
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
        config.cpuClass,
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
        config.cpuClass,
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
        config.gpuClass,
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
        config.gpuClass,
      ),
    );
  }
}
