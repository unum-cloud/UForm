/**
 * The forward passes: text, images, video and audio in, pooled embeddings, per-token states and token vectors out, one
 * verb per input form, and the text choices, written once for every receiver.
 */
import type {
  CompiledChosen,
  CompiledEncoded,
  CompiledModel,
  CompiledRows,
  CompiledStates,
  Completion,
  Offsets,
  ProgressCallback,
} from "./library.js";
import { Answer, Arena, ArenaView, Executor, Scalar, Status, fail, scalarCode, submitAs } from "./types.js";

/**
 * The surfaces an encode call keeps, combined with `|`. The C verb fills whichever of `embeddings`, `states` and
 * `token_vectors` is non-`NULL`, and these flags decide which it is handed.
 * @enum {number}
 * @readonly
 */
export enum Outputs {
  /** One pooled embedding per item. */
  Embeddings = 1,
  /** The final-norm states, one row per token kept after truncation. */
  States = 2,
  /** The token head's unit vectors, one row per token kept, `tokenWidth` wide; `unsupported` without a token head. */
  TokenVectors = 4,
}

/** Optional per-call settings of every encode verb. */
export interface EncodeConfig {
  /** The surfaces to keep; absent keeps `Outputs.Embeddings` alone. */
  outputs?: number;
  /** The scalar every output lands as; absent lands `f32`, which a `Float32Array` reads. */
  outputsScalar?: Scalar;
  /** Hears the items encoded so far and may stop the call by answering `false`; only a model's own verbs, which run on the calling thread, take one. */
  progress?: ProgressCallback;
}

/** One decoded image: `ImageData` as a canvas hands it out, or interleaved RGB or RGBA bytes with their shape. */
export interface Pixels {
  /** Rows end to end, three or four bytes per pixel. */
  data: Uint8Array | Uint8ClampedArray;
  /** Pixels per row. */
  width: number;
  /** Rows. */
  height: number;
}

/** What an encode call reports beside its outputs. */
export interface EncodeStats {
  /** Items cut at the model's context. */
  truncated: number;
  /** Tokens the call encoded, after truncation. */
  tokens: number;
}

/**
 * Rows a call landed in its arena. Mirrors a `uform_rows_view_t` output.
 *
 * `data` is a `Float32Array` for `f32`, a `Float16Array` for `f16` where the runtime has one, a `Uint16Array` for
 * `bf16`, and raw bytes for the kinds JavaScript spells no element for.
 */
export class Rows extends ArenaView {
  /** Rows written. */
  readonly rows: number;
  /** Values in each row, the model's dimensions. */
  readonly columns: number;
  /** Bytes between consecutive rows. */
  readonly stride: number;
  /** The scalar kind the rows landed as. */
  readonly scalar: Scalar;
  readonly #data: ArrayBufferView | null;

  /** Made by a verb, never directly. */
  constructor(landed: CompiledRows) {
    super();
    this.#data = landed.data;
    this.rows = landed.rows;
    this.columns = landed.columns;
    this.stride = landed.stride;
    this.scalar = landed.scalar as Scalar;
  }

  /** Every row, laid `stride` bytes apart. */
  get data(): ArrayBufferView {
    return this.live(this.#data);
  }

  /** Borrows row `row` without copying. Binding-only. */
  row(row: number): ArrayBufferView {
    if (!Number.isInteger(row) || row < 0 || row >= this.rows)
      throw fail(Status.InvalidArgument, `No row ${row} in a run of ${this.rows}.`);
    const data = this.data as Uint8Array;
    const unit = data.BYTES_PER_ELEMENT;
    return data.subarray((row * this.stride) / unit, ((row + 1) * this.stride) / unit);
  }
}

/**
 * The final-norm states or the token vectors a call landed in its arena: rows end to end, item `i` spanning rows
 * `offsets[i]` up to `offsets[i + 1]`. Mirrors `uform_rows_tape_view_t`.
 */
export class States extends ArenaView {
  /** Every item's rows, end to end. */
  readonly rows: Rows;
  readonly #offsets: BigUint64Array | null;

  /** Made by a verb, never directly. */
  constructor(landed: CompiledStates) {
    super();
    this.rows = new Rows(landed);
    this.#offsets = landed.offsets;
  }

  /** The `count + 1` row offsets, the last of them the total. */
  get offsets(): BigUint64Array {
    return this.live(this.#offsets);
  }

  /** Borrows the rows of item `item` without copying. Binding-only. */
  item(item: number): ArrayBufferView {
    const offsets = this.offsets;
    if (!Number.isInteger(item) || item < 0 || item + 1 >= offsets.length)
      throw fail(Status.InvalidArgument, `No item ${item} in a run of ${offsets.length - 1}.`);
    const data = this.rows.data as Uint8Array;
    const unit = data.BYTES_PER_ELEMENT;
    const stride = this.rows.stride / unit;
    return data.subarray(Number(offsets[item]) * stride, Number(offsets[item + 1]) * stride);
  }
}

/** The surfaces one encode call kept, present as they were asked for. */
export interface Encoded {
  /** One pooled embedding per item. */
  embeddings?: Rows;
  /** The final-norm states of every item. */
  states?: States;
  /** The token head's unit vectors of every item, sharing the states' offsets. */
  tokenVectors?: States;
  /** Truncation and token counts. */
  stats: EncodeStats;
}

/** Lifts a backend's answer into the binding's views. */
function encodedOf(landed: CompiledEncoded): Encoded {
  return {
    embeddings: landed.embeddings && new Rows(landed.embeddings),
    states: landed.states && new States(landed.states),
    tokenVectors: landed.tokenVectors && new States(landed.tokenVectors),
    stats: { truncated: landed.stats.truncated, tokens: landed.stats.tokens },
  };
}

/**
 * One probability per choice a choose call landed in its arena, in the choices' order: item `i` spans
 * `offsets[i]` up to `offsets[i + 1]`. Mirrors the `probabilities` output of `uform_text_choose_t`.
 */
export class Probabilities extends ArenaView {
  readonly #data: Float32Array | null;
  readonly #offsets: BigUint64Array | null;

  /** Made by a verb, never directly. */
  constructor(landed: CompiledChosen) {
    super();
    this.#data = landed.probabilities;
    this.#offsets = landed.offsets;
  }

  /** Every item's probabilities, end to end. */
  get data(): Float32Array {
    return this.live(this.#data);
  }

  /** The `count + 1` offsets, the last of them the total. */
  get offsets(): BigUint64Array {
    return this.live(this.#offsets);
  }

  /** Borrows the probabilities of item `item` without copying. Binding-only. */
  item(item: number): Float32Array {
    const offsets = this.offsets;
    if (!Number.isInteger(item) || item < 0 || item + 1 >= offsets.length)
      throw fail(Status.InvalidArgument, `No item ${item} in a run of ${offsets.length - 1}.`);
    return this.data.subarray(Number(offsets[item]), Number(offsets[item + 1]));
  }
}

/** Optional per-call settings of `textChoose`. */
export interface ChooseConfig {
  /** The question type of each item, below the model's `choiceTemperatureGroups`; absent puts every item in group 0. */
  temperatureGroups?: Uint32Array | number[];
  /** Hears the items scored so far and may stop the call by answering `false`; only a model's own verbs take one. */
  progress?: ProgressCallback;
}

/** What one choose call landed. */
export interface Chosen {
  /** One probability per choice, summing to one per item that offers any. */
  probabilities: Probabilities;
  /** Truncation and token counts. */
  stats: EncodeStats;
}

/**
 * A ragged batch packed as the C interface reads it: elements end to end, and `count + 1` offsets into them. Mirrors
 * `uform_tape_view_t`.
 *
 * Every verb taking an array of entries also takes a packed tape, which the Node addon lends to the call as it is when
 * its offsets are a `BigUint64Array`; an array of entries is first gathered into one tape, the one copy that path makes.
 * A `Probabilities` answer is itself a tape, so it trains a step's `choiceTargets` as it lands.
 */
export interface Tape<Elements extends Uint8Array | Float32Array> {
  data: Elements;
  offsets: Offsets;
}

/** Lends a packed tape as it is, refusing anything that is neither one nor an array of entries. */
function packed<Elements extends Uint8Array | Float32Array>(tape: Tape<Elements>): Tape<Elements> {
  const offsets = tape?.offsets;
  if (!ArrayBuffer.isView(tape?.data) || !(offsets instanceof BigUint64Array || offsets instanceof Uint32Array))
    throw fail(Status.InvalidArgument, "A batch is an array of entries, or a tape of data and offsets.");
  return tape;
}

/** Lays `entries` end to end, each measured by `length` and copied by `place`. */
function tapeOf<Entry, Elements extends Uint8Array | Float32Array>(
  entries: readonly Entry[],
  length: (entry: Entry) => number,
  allocate: (elements: number) => Elements,
  place: (data: Elements, entry: Entry, at: number) => void,
): Tape<Elements> {
  if (!Array.isArray(entries)) throw fail(Status.InvalidArgument, "A batch is an array of entries.");
  const offsets = new BigUint64Array(entries.length + 1);
  let total = 0;
  for (let entry = 0; entry < entries.length; entry++) offsets[entry + 1] = BigInt((total += length(entries[entry])));
  const data = allocate(total);
  for (let entry = 0; entry < entries.length; entry++) place(data, entries[entry], Number(offsets[entry]));
  return { data, offsets };
}

/** Lays UTF-8 texts or byte runs end to end, so text and every encoded-file verb share one tape, or lends a packed
 *  tape as it is. */
export function bytesTape(entries: readonly (string | Uint8Array)[] | Tape<Uint8Array>): Tape<Uint8Array> {
  if (!Array.isArray(entries)) return packed(entries as Tape<Uint8Array>);
  const encoder = new TextEncoder();
  const encoded = entries.map((entry) => (typeof entry === "string" ? encoder.encode(entry) : entry));
  for (const entry of encoded)
    if (!(entry instanceof Uint8Array)) throw fail(Status.InvalidArgument, "Every entry is a string or a Uint8Array.");
  return tapeOf(
    encoded,
    (entry) => entry.byteLength,
    (elements) => new Uint8Array(elements),
    (data, entry, at) => data.set(entry, at),
  );
}

/** Lays decoded images end to end as interleaved RGB, dropping the alpha of RGBA entries such as `ImageData`. */
function pixelsTape(images: readonly Pixels[]): Tape<Uint8Array> & { heights: Uint32Array; widths: Uint32Array } {
  const heights = new Uint32Array(images.length);
  const widths = new Uint32Array(images.length);
  images.forEach((image, position) => {
    const pixels = image.width * image.height;
    if (image.data.length !== pixels * 3 && image.data.length !== pixels * 4)
      throw fail(Status.ShapeMismatch, `Image ${position} holds neither RGB nor RGBA of its width and height.`);
    heights[position] = image.height;
    widths[position] = image.width;
  });
  const tape = tapeOf(
    images,
    (image) => image.width * image.height * 3,
    (elements) => new Uint8Array(elements),
    (data, image, at) => {
      const pixels = image.width * image.height;
      if (image.data.length === pixels * 3) return data.set(image.data, at);
      for (let pixel = 0; pixel < pixels; pixel++) {
        data[at + pixel * 3] = image.data[pixel * 4];
        data[at + pixel * 3 + 1] = image.data[pixel * 4 + 1];
        data[at + pixel * 3 + 2] = image.data[pixel * 4 + 2];
      }
    },
  );
  return { ...tape, heights, widths };
}

/** Lays mono recordings or other runs of floats end to end, or lends a packed tape as it is. */
export function samplesTape(recordings: readonly Float32Array[] | Tape<Float32Array>): Tape<Float32Array> {
  if (!Array.isArray(recordings)) return packed(recordings as Tape<Float32Array>);
  for (const recording of recordings)
    if (!(recording instanceof Float32Array)) throw fail(Status.InvalidArgument, "Every recording is a Float32Array.");
  return tapeOf(
    recordings,
    (recording) => recording.length,
    (elements) => new Float32Array(elements),
    (data, recording, at) => data.set(recording, at),
  );
}

/**
 * The encode verbs every receiver of a model shares, written once.
 *
 * The receiver chooses where a verb runs and how it completes, and the verb stays the same: a `Model` runs it on the
 * calling thread, `model.on(executor)` blocks on an executor, and `model.deferred(executor)` submits it there and
 * answers a {@link Deferred}. Every verb lands its outputs in `arena` and answers views into it.
 */
export abstract class EncodeVerbs<Receiving extends "blocking" | "deferred"> {
  /** The backend handle every verb is called on; shared by every receiver of one model. */
  protected readonly compiled: CompiledModel;
  /** The executor the verbs run on; absent runs them on the calling thread. */
  readonly executor: Executor | undefined;
  /** How this receiver completes its calls. */
  protected readonly receiving: Receiving;

  protected constructor(compiled: CompiledModel, executor: Executor | undefined, receiving: Receiving) {
    this.compiled = compiled;
    this.executor = executor;
    this.receiving = receiving;
  }

  /** Submits one call with this receiver's completion, and hands its answer back as this receiver does. */
  protected submit<Raw, Answered>(
    call: (completion: Completion) => unknown,
    answer: (raw: Raw) => Answered,
  ): Answer<Receiving, Answered> {
    return submitAs(this.receiving, call, answer);
  }

  /** Submits one encode call, the tape already laid out by the verb. */
  #encode(
    call: (completion: Completion, ...landing: [Arena["compiled"], number, number, ProgressCallback | undefined]) => unknown,
    arena: Arena,
    config: EncodeConfig,
  ): Answer<Receiving, Encoded> {
    if (!(arena instanceof Arena)) throw fail(Status.InvalidArgument, "The 'arena' argument must be an Arena.");
    const outputs = config.outputs ?? Outputs.Embeddings;
    const scalar = scalarCode(config.outputsScalar ?? Scalar.F32);
    return this.submit(
      (completion) => call(completion, arena.compiled, outputs, scalar, config.progress),
      (landed: CompiledEncoded) => encodedOf(landed),
    );
  }

  /**
   * Encodes texts through one forward; every UTF-8 byte is a token. Calls `uform_text_encode`.
   *
   * @param {Array<string|Uint8Array>|Tape<Uint8Array>} texts - Strings, UTF-8 bytes already encoded, or a tape of them.
   * @param {Arena} arena - The memory the outputs land in, rewound by the call.
   * @param {EncodeConfig} [config] - The surfaces, their scalar, and a progress handler.
   * @return {Encoded} The outputs, viewed inside `arena`.
   */
  textEncode(
    texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Answer<Receiving, Encoded> {
    const tape = bytesTape(texts);
    const executor = this.executor?.compiled;
    return this.#encode(
      (completion, ...landing) => this.compiled.textEncode(executor, completion, tape.data, tape.offsets, ...landing),
      arena,
      config,
    );
  }

  /**
   * Scores each item's legal choices through one forward, a softmax over them alone at the item's group's calibrated
   * inverse temperature; a model without the byte decoder reports `unsupported`. Calls `uform_text_choose`.
   *
   * @param {Array<string|Uint8Array>|Tape<Uint8Array>} texts - The states, each ending where its choice goes.
   * @param {Array<string|Uint8Array>|Tape<Uint8Array>} choices - Each item's legal choices, one byte each, like "yn".
   * @param {Arena} arena - The memory the probabilities land in, rewound by the call.
   * @param {ChooseConfig} [config] - The question types and a progress handler.
   * @return {Chosen} The probabilities, viewed inside `arena`.
   */
  textChoose(
    texts: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    choices: readonly (string | Uint8Array)[] | Tape<Uint8Array>,
    arena: Arena,
    config: ChooseConfig = {},
  ): Answer<Receiving, Chosen> {
    if (!(arena instanceof Arena)) throw fail(Status.InvalidArgument, "The 'arena' argument must be an Arena.");
    const states = bytesTape(texts);
    const legal = bytesTape(choices);
    const groups = config.temperatureGroups;
    const temperatureGroups = groups === undefined || groups instanceof Uint32Array ? groups : Uint32Array.from(groups);
    const executor = this.executor?.compiled;
    return this.submit(
      (completion) =>
        this.compiled.textChoose(
          executor,
          completion,
          states.data,
          states.offsets,
          legal.data,
          legal.offsets,
          temperatureGroups,
          arena.compiled,
          config.progress,
        ),
      (landed: CompiledChosen) => ({
        probabilities: new Probabilities(landed),
        stats: { truncated: landed.stats.truncated, tokens: landed.stats.tokens },
      }),
    );
  }

  /**
   * Decodes and encodes JPEG, PNG or WebP files; the browser build reports `unsupported`. Calls `uform_image_encode`.
   *
   * @param {Uint8Array[]|Tape<Uint8Array>} images - One encoded file per item, or a tape of them.
   * @param {Arena} arena - The memory the outputs land in, rewound by the call.
   * @param {EncodeConfig} [config] - The surfaces, their scalar, and a progress handler.
   */
  imageEncode(
    images: readonly Uint8Array[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Answer<Receiving, Encoded> {
    const tape = bytesTape(images);
    const executor = this.executor?.compiled;
    return this.#encode(
      (completion, ...landing) => this.compiled.imageEncode(executor, completion, tape.data, tape.offsets, ...landing),
      arena,
      config,
    );
  }

  /**
   * Encodes decoded images, resized on the executor's side. Calls `uform_image_encode_pixels`.
   *
   * @param {Pixels[]} images - `ImageData`, or RGB or RGBA bytes with their width and height.
   * @param {Arena} arena - The memory the outputs land in, rewound by the call.
   * @param {EncodeConfig} [config] - The surfaces, their scalar, and a progress handler.
   */
  imageEncodePixels(images: readonly Pixels[], arena: Arena, config: EncodeConfig = {}): Answer<Receiving, Encoded> {
    const tape = pixelsTape(images);
    const executor = this.executor?.compiled;
    return this.#encode(
      (completion, ...landing) =>
        this.compiled.imageEncodePixels(
          executor,
          completion,
          tape.data,
          tape.offsets,
          tape.heights,
          tape.widths,
          ...landing,
        ),
      arena,
      config,
    );
  }

  /**
   * Decodes and encodes MP4 or WebM clips; the browser build reports `unsupported`. Calls `uform_video_encode`.
   *
   * @param {Uint8Array[]|Tape<Uint8Array>} videos - One encoded file per item, or a tape of them.
   * @param {Arena} arena - The memory the outputs land in, rewound by the call.
   * @param {EncodeConfig} [config] - The surfaces, their scalar, and a progress handler.
   */
  videoEncode(
    videos: readonly Uint8Array[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Answer<Receiving, Encoded> {
    const tape = bytesTape(videos);
    const executor = this.executor?.compiled;
    return this.#encode(
      (completion, ...landing) => this.compiled.videoEncode(executor, completion, tape.data, tape.offsets, ...landing),
      arena,
      config,
    );
  }

  /**
   * Decodes and encodes WAV, FLAC, MP3 or Opus files; the browser build reports `unsupported`. Calls `uform_audio_encode`.
   *
   * @param {Uint8Array[]|Tape<Uint8Array>} recordings - One encoded file per item, or a tape of them.
   * @param {Arena} arena - The memory the outputs land in, rewound by the call.
   * @param {EncodeConfig} [config] - The surfaces, their scalar, and a progress handler.
   */
  audioEncode(
    recordings: readonly Uint8Array[] | Tape<Uint8Array>,
    arena: Arena,
    config: EncodeConfig = {},
  ): Answer<Receiving, Encoded> {
    const tape = bytesTape(recordings);
    const executor = this.executor?.compiled;
    return this.#encode(
      (completion, ...landing) => this.compiled.audioEncode(executor, completion, tape.data, tape.offsets, ...landing),
      arena,
      config,
    );
  }

  /**
   * Encodes decoded mono recordings, resampled to the model's rate where `sampleRate` differs. Calls `uform_audio_encode_samples`.
   *
   * @param {Float32Array[]|Tape<Float32Array>} recordings - One run of samples per item, or a tape of them.
   * @param {number} sampleRate - Samples per second shared by every recording.
   * @param {Arena} arena - The memory the outputs land in, rewound by the call.
   * @param {EncodeConfig} [config] - The surfaces, their scalar, and a progress handler.
   */
  audioEncodeSamples(
    recordings: readonly Float32Array[] | Tape<Float32Array>,
    sampleRate: number,
    arena: Arena,
    config: EncodeConfig = {},
  ): Answer<Receiving, Encoded> {
    const tape = samplesTape(recordings);
    const executor = this.executor?.compiled;
    return this.#encode(
      (completion, ...landing) =>
        this.compiled.audioEncodeSamples(executor, completion, tape.data, tape.offsets, sampleRate, ...landing),
      arena,
      config,
    );
  }
}
