/**
 * The WebAssembly backend and the browser entry point: the same API as the Node addon, over the Emscripten build of the
 * C library that `uform.mjs` and `uform.wasm` carry.
 *
 * Calls run blocking on the calling thread with a `NULL` executor, which is all one single-threaded module offers, so
 * executors only order calls and a deferred receiver answers a call that is final once submitted. Argument structures
 * are laid into the module's heap at offsets computed once from the C headers' field order, as wasm32 lays them out:
 * pointers, `size_t` and enumerations are four bytes. Outputs are copied out of the arena before the call returns,
 * since a later heap growth would invalidate any view into it, and the copies are detached when the arena next
 * rewinds, as the addon's views are. Encoded media need host decoders this build does not carry, so those verbs, the
 * VRAM residency, the mesh and the trainer report `unsupported`; decode in the page and use the pixels and samples
 * verbs instead.
 *
 * ```js
 * import { initialize, Arena, Model } from "@unum-cloud/uform/wasm";
 * await initialize();
 * const model = Model.loadRamBuffer(await (await fetch(url)).arrayBuffer());
 * ```
 */
import type {
  Compiled,
  CompiledArena,
  CompiledDeferred,
  CompiledDescription,
  CompiledEncoded,
  CompiledExecutor,
  CompiledModel,
  CompiledRows,
  CompiledStates,
  ProgressCallback,
} from "./library.js";
import { Status, backend, bind, fail } from "./types.js";

/** The Emscripten module surface this backend calls: the heap, the runtime helpers, and the exported C verbs. */
export interface WasmModule {
  HEAPU8: Uint8Array;
  UTF8ToString(pointer: number): string;
  addFunction(callback: (...values: number[]) => number, signature: string): number;
  removeFunction(pointer: number): void;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _uform_arena_init(arguments_: number): void;
  _uform_arena_free(arena: number): void;
  _uform_status_name(status: number): number;
  _uform_scalar_name(scalar: number): number;
  _uform_modality_name(modality: number): number;
  _uform_objective_name(objective: number): number;
  _uform_negatives_name(negatives: number): number;
  _uform_version_name(): number;
  _uform_capabilities_compiled(): number;
  _uform_capabilities_available(): number;
  _uform_model_load_ram_buffer(arguments_: number): void;
  _uform_model_describe(arguments_: number): void;
  _uform_model_free(model: number): void;
  _uform_text_encode(arguments_: number): void;
  _uform_image_encode_pixels(arguments_: number): void;
  _uform_audio_encode_samples(arguments_: number): void;
}

/** The ES module factory `uform.mjs` exports by default. */
export type WasmFactory = (options?: object) => Promise<WasmModule>;

/** The C scalar types a wasm32 argument structure holds, every one of them four bytes wide and aligned. */
type Primitive = "pointer" | "size" | "enum" | "u32";

/** Bytes each primitive takes on wasm32, which is also its alignment. */
const primitiveBytes: Record<Primitive, number> = { pointer: 4, size: 4, enum: 4, u32: 4 };

/** A structure's size, alignment, and the offset of every field, nested ones spelled `outer.inner`. */
interface Layout {
  size: number;
  alignment: number;
  at: Record<string, number>;
}

/** Lays out the fields of a C structure in declaration order, padding each to its alignment as the C compiler does. */
function struct(fields: ReadonlyArray<readonly [string, Primitive | Layout]>): Layout {
  const at: Record<string, number> = {};
  let offset = 0;
  let alignment = 1;
  for (const [name, type] of fields) {
    const size = typeof type === "string" ? primitiveBytes[type] : type.size;
    const aligned = typeof type === "string" ? primitiveBytes[type] : type.alignment;
    offset += (aligned - (offset % aligned)) % aligned;
    at[name] = offset;
    if (typeof type !== "string") for (const [inner, place] of Object.entries(type.at)) at[`${name}.${inner}`] = offset + place;
    offset += size;
    alignment = Math.max(alignment, aligned);
  }
  return { size: offset + ((alignment - (offset % alignment)) % alignment), alignment, at };
}

const outcomeLayout = struct([["status", "enum"], ["message", "pointer"]]);
const bytesViewLayout = struct([["data", "pointer"], ["length", "size"]]);
const ramOptionsLayout = struct([["memory_domains", "pointer"], ["memory_domains_count", "size"]]);
const tapeLayout = struct([["data", "pointer"], ["offsets", "pointer"], ["scalar", "enum"]]);
const progressLayout = struct([["callback", "pointer"], ["state", "pointer"], ["steps", "size"]]);
const rowsLayout = struct([["data", "pointer"], ["stride", "size"], ["columns", "size"], ["scalar", "enum"]]);
const rowsTapeLayout = struct([["rows", rowsLayout], ["offsets", "pointer"]]);
const encodeStatsLayout = struct([["truncated", "size"], ["tokens", "size"]]);
const arenaInitLayout = struct([["arena", "pointer"], ["executor", "pointer"], ["outcome", "pointer"]]);
const loadRamBufferLayout = struct([
  ["model", "pointer"],
  ["executor", "pointer"],
  ["outcome", "pointer"],
  ["completion", "enum"],
  ["buffer", bytesViewLayout],
  ["ram", ramOptionsLayout],
  ["quantization", "enum"],
  ["progress", progressLayout],
]);
const descriptionLayout = struct([
  ["dimensions", "size"],
  ["context", "size"],
  ["quantization", "enum"],
  ["memory_usage", "size"],
  ["capabilities", "pointer"],
]);
const describeLayout = struct([
  ["model", "pointer"],
  ["executor", "pointer"],
  ["outcome", "pointer"],
  ["completion", "enum"],
  ["description", "pointer"],
]);

/** The fields every encode structure opens with, then its inputs, then the fields it closes with. */
const encodeHead = [
  ["model", "pointer"],
  ["executor", "pointer"],
  ["outcome", "pointer"],
  ["completion", "enum"],
  ["count", "size"],
] as const;
const encodeTail = [
  ["embeddings", "pointer"],
  ["states", "pointer"],
  ["arena", "pointer"],
  ["outputs_scalar", "enum"],
  ["progress", progressLayout],
  ["stats", "pointer"],
] as const;
const textEncodeLayout = struct([...encodeHead, ["texts", tapeLayout], ...encodeTail]);
const pixelsEncodeLayout = struct([
  ...encodeHead,
  ["pixels", tapeLayout],
  ["heights", "pointer"],
  ["widths", "pointer"],
  ...encodeTail,
]);
const samplesEncodeLayout = struct([...encodeHead, ["samples", tapeLayout], ["sample_rate", "u32"], ...encodeTail]);

/** The C scalar codes the name tables are asked about, as `uform_scalar_t` numbers them. */
const scalarCodes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 20, 21, 22];

/** The C modality codes, as `uform_modality_t` numbers them. */
const modalityCodes = [0, 1, 2, 3];

/** The C objective codes, as `uform_objective_t` numbers them. */
const objectiveCodes = [0, 1, 2];

/** The C negatives codes, as `uform_negatives_t` numbers them. */
const negativesCodes = [0, 1];

/** The heap blocks one call allocates, freed together once its outputs are copied out. */
class Frame {
  readonly #module: WasmModule;
  readonly #blocks: number[] = [];

  constructor(module: WasmModule) {
    this.#module = module;
  }

  /** A zeroed block of `bytes`. */
  zeroed(bytes: number): number {
    const address = this.#module._malloc(Math.max(bytes, 4));
    if (!address) throw fail(Status.OutOfMemory, "The WebAssembly heap could not grow.");
    this.#blocks.push(address);
    this.#module.HEAPU8.fill(0, address, address + Math.max(bytes, 4));
    return address;
  }

  /** A copy of `view` on the heap. */
  copy(view: ArrayBufferView): number {
    const address = this.zeroed(view.byteLength);
    this.#module.HEAPU8.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength), address);
    return address;
  }

  /** Stores a four-byte field. */
  store(address: number, value: number): void {
    new DataView(this.#module.HEAPU8.buffer).setUint32(address, value >>> 0, true);
  }

  /** Loads a four-byte field. */
  load(address: number): number {
    return new DataView(this.#module.HEAPU8.buffer).getUint32(address, true);
  }

  free(): void {
    for (const address of this.#blocks) this.#module._free(address);
    this.#blocks.length = 0;
  }
}

/** The array type a scalar reads back as; kinds JavaScript spells no element for stay bytes. */
function viewOf(scalar: string, buffer: ArrayBuffer): ArrayBufferView {
  const half = (globalThis as { Float16Array?: new (buffer: ArrayBuffer) => ArrayBufferView }).Float16Array;
  if (scalar === "f64") return new Float64Array(buffer);
  if (scalar === "f32") return new Float32Array(buffer);
  if (scalar === "f16") return half ? new half(buffer) : new Uint16Array(buffer);
  if (scalar === "bf16") return new Uint16Array(buffer);
  if (scalar === "i8") return new Int8Array(buffer);
  return new Uint8Array(buffer);
}

/** Builds the backend over an instantiated module. */
function wasmBackend(module: WasmModule): Compiled {
  const text = (pointer: number) => (pointer ? module.UTF8ToString(pointer) : "");
  const names = (codes: number[], spell: (code: number) => number) =>
    Object.freeze(Object.fromEntries(codes.map((code) => [text(spell(code)), code])));
  const scalars = names(scalarCodes, module._uform_scalar_name);

  /** Throws the failure an outcome at `address` carries. */
  function check(frame: Frame, address: number): void {
    const status = frame.load(address + outcomeLayout.at.status);
    if (status === 0) return;
    const message = text(frame.load(address + outcomeLayout.at.message)) || "The operation failed";
    throw fail(text(module._uform_status_name(status)) as Status, message);
  }

  /** Runs `call` now and answers a deferred call that is already final. */
  function settled(run: () => unknown): CompiledDeferred {
    let value: unknown;
    let failure: { error: unknown } | undefined;
    try {
      value = run();
    } catch (error) {
      failure = { error };
    }
    return {
      status: () => (failure ? ((failure.error as { code?: string }).code ?? Status.Internal) : Status.Success),
      wait: () => undefined,
      settled: () => Promise.resolve(),
      result: () => {
        if (failure) throw failure.error;
        return value;
      },
    };
  }

  class WasmExecutor implements CompiledExecutor {
    static forRam(): CompiledExecutor {
      return new WasmExecutor();
    }
    static forVram(): CompiledExecutor {
      throw fail(Status.Unsupported, "The WebAssembly build has no VRAM residency.");
    }
    static onPool(): CompiledExecutor {
      throw fail(Status.Unsupported, "The WebAssembly build computes on the calling thread alone.");
    }
    drain(): void {}
    free(): void {}
  }

  class WasmArena implements CompiledArena {
    readonly residency = "ram";
    handle = 0;
    readonly #views: ArrayBuffer[] = [];

    constructor(executor?: CompiledExecutor) {
      if (executor !== undefined && !(executor instanceof WasmExecutor))
        throw fail(Status.InvalidArgument, "The 'executor' argument must be an Executor of this backend.");
      const frame = new Frame(module);
      try {
        const slot = frame.zeroed(4);
        const outcome = frame.zeroed(outcomeLayout.size);
        const init = frame.zeroed(arenaInitLayout.size);
        frame.store(init + arenaInitLayout.at.arena, slot);
        frame.store(init + arenaInitLayout.at.outcome, outcome);
        module._uform_arena_init(init);
        check(frame, outcome);
        this.handle = frame.load(slot);
      } finally {
        frame.free();
      }
    }

    /** Detaches every copy handed out since the last rewind, as a call rewinding the arena invalidates them. */
    rewind(): void {
      for (const buffer of this.#views) buffer.transfer();
      this.#views.length = 0;
    }

    /** Remembers `buffer`, so the next rewind or free detaches it. */
    adopt(buffer: ArrayBuffer): ArrayBuffer {
      this.#views.push(buffer);
      return buffer;
    }

    /** Copies `bytes` bytes at `address` out of the heap into a buffer this arena detaches on its next rewind. */
    copy(address: number, bytes: number): ArrayBuffer {
      return this.adopt(module.HEAPU8.slice(address, address + bytes).buffer);
    }

    free(): void {
      if (!this.handle) return;
      this.rewind();
      module._uform_arena_free(this.handle);
      this.handle = 0;
    }
  }

  /** Reads the rows at `address`, `count` of them, into a copy owned by `arena`. */
  function rowsAt(frame: Frame, arena: WasmArena, address: number, count: number): CompiledRows {
    const stride = frame.load(address + rowsLayout.at.stride);
    const scalar = text(module._uform_scalar_name(frame.load(address + rowsLayout.at.scalar)));
    const data = arena.copy(frame.load(address + rowsLayout.at.data), count * stride);
    return { data: viewOf(scalar, data), rows: count, columns: frame.load(address + rowsLayout.at.columns), stride, scalar };
  }

  /** The tape fields of an encode structure: elements and `size_t` offsets, which a `Uint32Array` matches on wasm32. */
  type Inputs = { data: ArrayBufferView; offsets: Uint32Array; scalar: number };

  /** Lays one encode structure out, runs it blocking, and copies its outputs out of `arena`. */
  function encode(
    model: number,
    verb: (arguments_: number) => void,
    layout: Layout,
    tape: string,
    inputs: Inputs,
    extra: (frame: Frame, arguments_: number) => void,
    arenaValue: CompiledArena,
    outputs: number,
    outputsScalar: number,
    progress: ProgressCallback | undefined,
  ): CompiledEncoded {
    if (!(arenaValue instanceof WasmArena) || !arenaValue.handle)
      throw fail(Status.InvalidArgument, "The 'arena' argument must be a live Arena of this backend.");
    const arena = arenaValue;
    const count = inputs.offsets.length - 1;
    const frame = new Frame(module);
    let relay = 0;
    let thrown: { error: unknown } | undefined;
    try {
      const call = frame.zeroed(layout.size);
      const outcome = frame.zeroed(outcomeLayout.size);
      const embeddings = outputs & 1 ? frame.zeroed(rowsLayout.size) : 0;
      const states = outputs & 2 ? frame.zeroed(rowsTapeLayout.size) : 0;
      const stats = frame.zeroed(encodeStatsLayout.size);
      frame.store(call + layout.at.model, model);
      frame.store(call + layout.at.outcome, outcome);
      frame.store(call + layout.at.count, count);
      frame.store(call + layout.at[`${tape}.data`], frame.copy(inputs.data));
      frame.store(call + layout.at[`${tape}.offsets`], frame.copy(inputs.offsets));
      frame.store(call + layout.at[`${tape}.scalar`], inputs.scalar);
      frame.store(call + layout.at.embeddings, embeddings);
      frame.store(call + layout.at.states, states);
      frame.store(call + layout.at.arena, arena.handle);
      frame.store(call + layout.at.outputs_scalar, outputsScalar);
      frame.store(call + layout.at.stats, stats);
      extra(frame, call);
      if (progress) {
        relay = module.addFunction((completed: number, total: number) => {
          if (thrown) return 1;
          try {
            return progress(completed, total) === false ? 1 : 0;
          } catch (error) {
            thrown = { error };
            return 1;
          }
        }, "iiii");
        frame.store(call + layout.at["progress.callback"], relay);
      }
      arena.rewind();
      verb(call);
      if (thrown) throw thrown.error;
      check(frame, outcome);
      const landed: CompiledEncoded = {
        stats: {
          truncated: frame.load(stats + encodeStatsLayout.at.truncated),
          tokens: frame.load(stats + encodeStatsLayout.at.tokens),
        },
      };
      if (embeddings) landed.embeddings = rowsAt(frame, arena, embeddings, count);
      if (states) {
        const offsetsAddress = frame.load(states + rowsTapeLayout.at.offsets);
        const offsets = new BigUint64Array(arena.adopt(new ArrayBuffer((count + 1) * 8)));
        for (let position = 0; position <= count; position++)
          offsets[position] = BigInt(frame.load(offsetsAddress + position * 4));
        const rows: CompiledStates = { ...rowsAt(frame, arena, states, Number(offsets[count])), offsets };
        landed.states = rows;
      }
      return landed;
    } finally {
      if (relay) module.removeFunction(relay);
      frame.free();
    }
  }

  /** Refuses a verb that needs host decoders the browser build does not carry. */
  function undecodable(): never {
    throw fail(Status.Unsupported, "The WebAssembly build decodes no media; pass decoded pixels or samples instead.");
  }

  /** Runs a blocking call now, or answers it as a deferred call already final. */
  function complete<Result>(completion: number, run: () => Result): Result | CompiledDeferred {
    return completion === 1 ? settled(run) : run();
  }

  class WasmModel implements CompiledModel {
    handle: number;

    constructor(handle: number) {
      this.handle = handle;
    }

    static loadRam(): CompiledModel {
      throw fail(Status.Unsupported, "The WebAssembly build has no filesystem; load the checkpoint with loadRamBuffer.");
    }

    static loadVram(): CompiledModel {
      throw fail(Status.Unsupported, "The WebAssembly build has no VRAM residency.");
    }

    static loadVramBuffer(): CompiledModel {
      throw fail(Status.Unsupported, "The WebAssembly build has no VRAM residency.");
    }

    static loadRamBuffer(
      buffer: ArrayBufferView,
      executor: CompiledExecutor | undefined,
      memoryDomains: number[] | undefined,
      quantization: number,
    ): CompiledModel {
      const frame = new Frame(module);
      try {
        const slot = frame.zeroed(4);
        const outcome = frame.zeroed(outcomeLayout.size);
        const load = frame.zeroed(loadRamBufferLayout.size);
        frame.store(load + loadRamBufferLayout.at.model, slot);
        frame.store(load + loadRamBufferLayout.at.outcome, outcome);
        frame.store(load + loadRamBufferLayout.at["buffer.data"], frame.copy(buffer));
        frame.store(load + loadRamBufferLayout.at["buffer.length"], buffer.byteLength);
        frame.store(load + loadRamBufferLayout.at.quantization, quantization);
        module._uform_model_load_ram_buffer(load);
        check(frame, outcome);
        return new WasmModel(frame.load(slot));
      } finally {
        frame.free();
      }
    }

    #live(): number {
      if (!this.handle) throw fail(Status.InvalidArgument, "The model was freed");
      return this.handle;
    }

    free(): void {
      module._uform_model_free(this.handle);
      this.handle = 0;
    }

    describe(executor: CompiledExecutor | undefined, completion: number): CompiledDescription | CompiledDeferred {
      return complete(completion, () => {
        const frame = new Frame(module);
        try {
          const outcome = frame.zeroed(outcomeLayout.size);
          const description = frame.zeroed(descriptionLayout.size);
          const describe = frame.zeroed(describeLayout.size);
          frame.store(describe + describeLayout.at.model, this.#live());
          frame.store(describe + describeLayout.at.outcome, outcome);
          frame.store(describe + describeLayout.at.description, description);
          module._uform_model_describe(describe);
          check(frame, outcome);
          return {
            dimensions: frame.load(description + descriptionLayout.at.dimensions),
            context: frame.load(description + descriptionLayout.at.context),
            quantization: text(module._uform_scalar_name(frame.load(description + descriptionLayout.at.quantization))),
            memoryUsage: frame.load(description + descriptionLayout.at.memory_usage),
            capabilities: text(frame.load(description + descriptionLayout.at.capabilities)),
          };
        } finally {
          frame.free();
        }
      });
    }

    textEncode(
      executor: CompiledExecutor | undefined,
      completion: number,
      bytes: Uint8Array,
      offsets: Uint32Array,
      arena: CompiledArena,
      outputs: number,
      outputsScalar: number,
      progress: ProgressCallback | undefined,
    ): CompiledEncoded | CompiledDeferred {
      const model = this.#live();
      return complete(completion, () =>
        encode(
          model,
          module._uform_text_encode,
          textEncodeLayout,
          "texts",
          { data: bytes, offsets, scalar: scalars.u8 },
          () => undefined,
          arena,
          outputs,
          outputsScalar,
          progress,
        ),
      );
    }

    imageEncodePixels(
      executor: CompiledExecutor | undefined,
      completion: number,
      rgb: Uint8Array,
      offsets: Uint32Array,
      heights: Uint32Array,
      widths: Uint32Array,
      arena: CompiledArena,
      outputs: number,
      outputsScalar: number,
      progress: ProgressCallback | undefined,
    ): CompiledEncoded | CompiledDeferred {
      const model = this.#live();
      return complete(completion, () =>
        encode(
          model,
          module._uform_image_encode_pixels,
          pixelsEncodeLayout,
          "pixels",
          { data: rgb, offsets, scalar: scalars.u8 },
          (frame, call) => {
            frame.store(call + pixelsEncodeLayout.at.heights, frame.copy(heights));
            frame.store(call + pixelsEncodeLayout.at.widths, frame.copy(widths));
          },
          arena,
          outputs,
          outputsScalar,
          progress,
        ),
      );
    }

    audioEncodeSamples(
      executor: CompiledExecutor | undefined,
      completion: number,
      samples: Float32Array,
      offsets: Uint32Array,
      sampleRate: number,
      arena: CompiledArena,
      outputs: number,
      outputsScalar: number,
      progress: ProgressCallback | undefined,
    ): CompiledEncoded | CompiledDeferred {
      const model = this.#live();
      return complete(completion, () =>
        encode(
          model,
          module._uform_audio_encode_samples,
          samplesEncodeLayout,
          "samples",
          { data: samples, offsets, scalar: scalars.f32 },
          (frame, call) => frame.store(call + samplesEncodeLayout.at.sample_rate, sampleRate),
          arena,
          outputs,
          outputsScalar,
          progress,
        ),
      );
    }

    imageEncode(): CompiledEncoded {
      return undecodable();
    }

    videoEncode(): CompiledEncoded {
      return undecodable();
    }

    audioEncode(): CompiledEncoded {
      return undecodable();
    }
  }

  const unsupported = () => {
    throw fail(Status.Unsupported, "Training needs GPUs, which the WebAssembly build never reaches.");
  };

  return {
    Executor: WasmExecutor,
    Arena: WasmArena,
    CompiledModel: WasmModel,
    CompiledMesh: { bootstrap: unsupported },
    CompiledTrainer: { initVram: unsupported },
    scalars,
    modalities: names(modalityCodes, module._uform_modality_name),
    objectives: names(objectiveCodes, module._uform_objective_name),
    negatives: names(negativesCodes, module._uform_negatives_name),
    version: () => text(module._uform_version_name()),
    capabilitiesCompiled: () => text(module._uform_capabilities_compiled()),
    capabilitiesAvailable: () => text(module._uform_capabilities_available()),
  };
}

/**
 * Instantiates the WebAssembly module and binds it, so every class made afterwards runs on it. Binding-only.
 *
 * @param {WasmFactory|string} [factory] - The `uform.mjs` factory, or the URL to import it from; absent imports the
 *     copy the package ships beside this module.
 */
export async function initialize(factory?: WasmFactory | string): Promise<void> {
  let create = factory;
  if (typeof create !== "function") {
    const specifier: string = create ?? "../wasm/uform.mjs";
    create = ((await import(specifier)) as { default: WasmFactory }).default;
  }
  bind(wasmBackend(await create()));
}

export {
  Arena,
  ArenaView,
  Deferred,
  Executor,
  Modality,
  Mood,
  MultimemTier,
  Negatives,
  Objective,
  Scalar,
  Status,
} from "./types.js";
export type {
  Answer,
  ExecutorForRamConfig,
  ExecutorForVramConfig,
  ExecutorOnPoolConfig,
  UFormError,
} from "./types.js";
export { EncodeVerbs, Outputs, Rows, States } from "./encode.js";
export type { EncodeConfig, EncodeStats, Encoded, Pixels } from "./encode.js";
export { Model, ModelDeferredOn, ModelOn } from "./model.js";
export type { ModelDescription, ModelLoadRamConfig, ModelLoadVramConfig } from "./model.js";
export { Mesh, Trainer, TrainerDeferredOn } from "./train.js";
export type { ProgressCallback } from "./library.js";

/** The library's version, as the release tooling stamped it. Calls `uform_version_name`. */
export function version(): string {
  return backend().version();
}
