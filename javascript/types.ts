/**
 *  @file javascript/types.ts
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The vocabulary every verb shares: closed-set names, the backend the classes run on, the
 *      executor whose threads a verb runs on, the arena every scaling output lands in, and the
 *      deferred call a deferred receiver answers.
 */
import type { Compiled, CompiledArena, CompiledDeferred, CompiledExecutor, Completion } from "./library.js";

/**
 *  What went wrong, one member per `uform_status_*_k` of the C interface and spelled as its
 *  canonical name. Every error the library or this binding throws carries one as its `code`, so a
 *  caller branches on the identity of the fault rather than on which side of the boundary saw it. A
 *  {@link Deferred} also reports `Success`, and `Deferred` until its call is final.
 *  @enum {string}
 *  @readonly
 */
export enum Status {
  Success = "success",
  Deferred = "deferred",
  OutOfMemory = "out-of-memory",
  ShapeMismatch = "shape-mismatch",
  UnsupportedLayout = "unsupported-layout",
  UnknownKind = "unknown-kind",
  Io = "io",
  Device = "device",
  Malformed = "malformed",
  Internal = "internal",
  Uninitialized = "uninitialized",
  ConfigMismatch = "config-mismatch",
  Contended = "contended",
  Cancelled = "cancelled",
  Unsupported = "unsupported",
  InvalidArgument = "invalid-argument",
}

/** An error thrown by the library or this binding, its `code` naming the failure. */
export interface UFormError extends Error {
  code: Status;
}

/** Builds the error this binding throws when it refuses a call before it crosses into a backend. */
export function fail(status: Status, message: string): UFormError {
  return Object.assign(new Error(message), { code: status });
}

/**
 *  What an executor does with its threads or streams once it goes idle.
 *  @enum {string}
 *  @readonly
 */
export enum Mood {
  /** Idle threads sleep; the next call pays a wake-up. */
  Parked = "parked",
  /** Idle threads stay hot, trading power for the next call's latency. */
  Spinning = "spinning",
}

/**
 *  The scalar kinds a model computes in and its outputs land as, numbered as USearch numbers them.
 *  @enum {string}
 *  @readonly
 */
export enum Scalar {
  F64 = "f64",
  F32 = "f32",
  BF16 = "bf16",
  F16 = "f16",
  E5M2 = "e5m2",
  E4M3 = "e4m3",
  E3M2 = "e3m2",
  E2M3 = "e2m3",
  I8 = "i8",
  I4x2 = "i4x2",
  I32 = "i32",
  U8 = "u8",
  U4x2 = "u4x2",
  U1x8 = "u1x8",
}

/**
 *  What the bytes of one training item hold.
 *  @enum {string}
 *  @readonly
 */
export enum Modality {
  Text = "text",
  Image = "image",
  Video = "video",
  Audio = "audio",
}

/**
 *  Which losses a training step relaxes.
 *  @enum {string}
 *  @readonly
 */
export enum Objective {
  /** Masked reconstruction alone. */
  Masked = "masked",
  /** The pairwise contrastive loss alone. */
  Contrastive = "contrastive",
  /** Both, joined at the final-norm states. */
  Joint = "joint",
}

/**
 *  Which pairs the contrastive loss pushes apart.
 *  @enum {string}
 *  @readonly
 */
export enum Negatives {
  /** Repel edges, and every pair no edge names. */
  Ambient = "ambient",
  /** Repel edges alone. */
  Named = "named",
}

/** How a mesh's NVLink domains reduce, as `uform_multimem_tier_name` spells it. */
export type MultimemTier = "switch-reduced" | "peer-read";

/** How a contrastive term turns a row of scores into forces, one member per
 *  `uform_pair_loss_form_t`.
 *  @enum {string}
 *  @readonly */
export enum PairLossForm {
  /** The term's own form: sigmoid for pooled embeddings, ranked for token vectors. */
  Derived = "derived",
  /** The sigmoid cross-entropy of every cell, its attract and repel halves normalized apart. */
  Balanced = "balanced",
  /** The sigmoid cross-entropy of every positive's score over every negative's, where
   *  the bias cancels. */
  Ranked = "ranked",
  /** SigLIP's sigmoid cross-entropy of every cell, each repel weighed like an attract. */
  Sigmoid = "sigmoid",
}

/** Which side of a late-interaction score an item takes, one member per
 *  `uform_late_interaction_side_t`.
 *  @enum {string}
 *  @readonly */
export enum LateInteractionSide {
  /** The item takes no part in late interaction. */
  None = "none",
  /** The item's tokens score the documents it attracts and repels. */
  Query = "query",
  /** The item's tokens are scored by the queries naming it. */
  Document = "document",
  /** The item plays both sides. */
  QueryAndDocument = "query-and-document",
}

/** The runtime that drives a GPU, one name per `uform_gpu_vendor_t`. */
export type GpuVendor = "unknown" | "nvidia" | "amd" | "apple";

/** One GPU this process can drive. Mirrors `uform_gpu_device_t`. */
export interface GpuDevice {
  /** The vendor whose runtime drives the device. */
  vendor: GpuVendor;
  /** The device's ordinal in its vendor's runtime, as `nvidia-smi` or `rocm-smi` lists it. */
  vendorOrdinal: number;
  /** The class whose kernels run on it, as `uform_gpu_class_name` spells
   *  it, like "nvidia-hopper". */
  gpuClass: string | null;
  /** Streaming multiprocessors or compute units. */
  computeUnits: number;
  /** Device memory in bytes. */
  memoryBytes: number;
  /** The vendor's own name for the device. */
  name: string;
}

let bound: Compiled | undefined;

/** Installs the backend every class made from now on runs on: the Node addon, or
 *  the WebAssembly module. */
export function bind(compiled: Compiled): void {
  bound = compiled;
}

/** The backend classes are made on; refuses with {@link Status.Unsupported} before one is bound. */
export function backend(): Compiled {
  if (!bound)
    throw fail(
      Status.Unsupported,
      "No backend is bound: import '@unum-cloud/uform' under Node, or await initialize() from '@unum-cloud/uform/wasm'.",
    );
  return bound;
}

/** The code `scalar` crosses a backend as, mapped from the C name tables. */
export function scalarCode(scalar: Scalar): number {
  const code = backend().scalars[scalar];
  if (code === undefined) throw fail(Status.UnknownKind, `This build names no scalar ${scalar}.`);
  return code;
}

/** The code `modality` crosses a backend as, mapped from the C name tables. */
export function modalityCode(modality: Modality): number {
  const code = backend().modalities[modality];
  if (code === undefined) throw fail(Status.UnknownKind, `This build names no modality ${modality}.`);
  return code;
}

/** The code `objective` crosses a backend as, mapped from the C name tables. */
export function objectiveCode(objective: Objective): number {
  const code = backend().objectives[objective];
  if (code === undefined) throw fail(Status.UnknownKind, `This build names no objective ${objective}.`);
  return code;
}

/** The code `negatives` crosses a backend as, mapped from the C name tables. */
export function negativesCode(negatives: Negatives): number {
  const code = backend().negatives[negatives];
  if (code === undefined) throw fail(Status.UnknownKind, `This build names no negatives policy ${negatives}.`);
  return code;
}

/** Lists every GPU this process can drive, NVIDIA devices first, then AMD, then Apple, so a
 *  device's position is the ordinal VRAM placements name it by; a machine without GPUs lists none.
 *  Calls `uform_gpu_devices`. */
export function gpuDevices(): GpuDevice[] {
  return backend().gpuDevices() as GpuDevice[];
}

/** Where the host threads of a RAM executor are placed, how many there are, and how many deferred
 *  calls it tracks. */
export interface ExecutorForRamConfig {
  /** NUMA node IDs the threads are spread over; absent spreads them over every node. */
  memoryDomains?: number[];
  /** The newest CPU class to dispatch as, spelled like "intel-haswell", capped by what the CPU
   *  runs; absent sets no ceiling. */
  cpuClassCeiling?: string;
  /** Threads to spawn; absent takes 7/8 of the cores the domains carry. */
  threads?: number;
}

/** Where the borrowed pool of a RAM executor places its models. */
export interface ExecutorOnPoolConfig {
  /** NUMA node IDs of the models it will drive; absent derives every node. */
  memoryDomains?: number[];
  /** The newest CPU class to dispatch as, spelled like "intel-haswell", capped by what the CPU
   *  runs; absent sets no ceiling. */
  cpuClassCeiling?: string;
}

/** Which devices a VRAM executor drives, and on which streams. */
export interface ExecutorForVramConfig {
  /** Device ordinals, positions in {@link gpuDevices}, to drive; absent takes one free device. */
  devices?: number[];
  /** The newest GPU class every device dispatches as, spelled like "nvidia-hopper": the best
   *  class all of them load at or under it; absent sets no ceiling. */
  gpuClassCeiling?: string;
  /** NUMA node IDs whose cores the host decoders take; absent takes every node. */
  memoryDomains?: number[];
  /** The newest CPU class the host decoders dispatch as, spelled like "intel-haswell"; absent
   *  sets no ceiling. */
  cpuClassCeiling?: string;
  /** Borrowed CUDA stream addresses, one per device in `devices` order, which the library refuses
   *  as unsupported; absent creates its own. */
  streams?: bigint[];
  /** Ceiling on scratch bytes per device; absent lets scratch grow as batches ask. */
  scratchLimit?: number;
}

/** The token the factories hold, which is what a direct `new Executor` cannot produce. */
const executorSeal = Symbol("Executor");

/**
 *  The threads or streams a verb runs on, shared across models.
 *
 *  An executor owns host threads or device streams and the activation scratch; a model owns
 *  weights. Keeping them apart is what lets one executor serve several models, and what makes the
 *  cost of spawning visible at the line that spawns it. The WebAssembly backend computes on the
 *  calling thread alone, so its executors only order calls. The collector frees it eventually;
 *  `free` joins the threads now.
 */
export class Executor implements Disposable {
  /** The backend handle every verb is handed. */
  readonly compiled: CompiledExecutor;

  /** Reached through the factories. */
  private constructor(seal: symbol, compiled: CompiledExecutor) {
    if (seal !== executorSeal)
      throw fail(Status.InvalidArgument, "An Executor is made by Executor.forRam, Executor.forVram or Executor.onPool.");
    this.compiled = compiled;
  }

  /** Creates an executor of host threads for RAM-resident models.
   *  Calls `uform_executor_for_ram`. */
  static forRam(config: ExecutorForRamConfig = {}): Executor {
    return new Executor(executorSeal, backend().Executor.forRam(config));
  }

  /** Opens one stream per device for VRAM-resident models. Calls `uform_executor_for_vram`. */
  static forVram(config: ExecutorForVramConfig = {}): Executor {
    return new Executor(executorSeal, backend().Executor.forVram(config));
  }

  /** Borrows a ForkUnion `fu_pool_t` by address, which reports `unsupported` until borrowing is
   *  designed. Calls `uform_executor_on_pool`. */
  static onPool(pool: bigint, config: ExecutorOnPoolConfig = {}): Executor {
    return new Executor(executorSeal, backend().Executor.onPool(pool, config));
  }

  /**
   *  Waits until no call holds this executor, then keeps `mood` until its next
   *  call. Calls `uform_executor_drain`.
   *  @param [mood=Mood.Parked] - What the threads or streams do once they go idle.
   */
  drain(mood: Mood = Mood.Parked): void {
    this.compiled.drain(mood);
  }

  /** Runs every deferred call and joins the threads now; a second `free` does
   *  nothing. Calls `uform_executor_free`. */
  free(): void {
    this.compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/**
 *  Memory the library writes every output whose length scales with a call into,
 *  reused across calls.
 *
 *  Every verb with such an output takes one and answers a view into it rather than a copy, so a
 *  steady loop allocates once. Each call rewinds the arena it is handed and detaches every buffer
 *  over the previous call's outputs: a view refuses its reads with {@link Status.InvalidArgument}
 *  from then on. To keep one result past the next call, give that call another arena. A deferred
 *  call holds it until final: another call naming it before then finishes {@link Status.Contended}
 *  and leaves the first call's outputs alone, and `free` waits for it.
 */
export class Arena implements Disposable {
  /** The backend handle every verb is handed. */
  readonly compiled: CompiledArena;

  /** Makes an arena where `executor` computes, or host pages for calls on the calling
   *  thread. Calls `uform_arena_init`. */
  constructor(executor?: Executor) {
    this.compiled = new (backend().Arena)(executor?.compiled);
  }

  /** `"ram"` or `"vram"`, where the executor that made it computes; a VRAM arena's views
   *  refuse host reads. */
  get residency(): string {
    return this.compiled.residency;
  }

  /** Releases the memory now, once every deferred call writing it is final, detaching every view
   *  into it. Calls `uform_arena_free`. */
  free(): void {
    this.compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/**
 *  Outputs a call wrote into an arena, readable until the arena's next call rewinds it.
 *
 *  The rewind detaches the buffer under every array, so each accessor refuses once the arena moved
 *  on, and refuses device memory with {@link Status.Device}, so a view never reads memory its call
 *  did not write.
 */
export abstract class ArenaView {
  /** Answers `view` while its buffer still holds this call's outputs in host memory,
   *  and refuses otherwise. */
  protected live<View extends ArrayBufferView>(view: View | null): View {
    if (view === null) throw fail(Status.Device, "The outputs are device memory, which no host view reads.");
    if ((view.buffer as ArrayBuffer).detached)
      throw fail(Status.InvalidArgument, "The arena was rewound or freed since this view was written.");
    return view;
  }
}

/** Yields one turn of the event loop: `setImmediate` under Node, a zero timeout in a page. */
const yieldTurn: (next: () => void) => void = globalThis.setImmediate ?? ((next) => setTimeout(next, 0));

/** Resolves once `call` is final, polling its status with a doubling pause from 1 µs up to 1 ms:
 *  pauses under 1 ms only yield a turn of the event loop, and the rest sleep 1 ms. */
function polled(call: CompiledDeferred): Promise<void> {
  return new Promise((resolve) => {
    let pause = 0.001;
    const poll = (): void => {
      if (call.status() !== Status.Deferred) return resolve();
      if (pause < 1) yieldTurn(poll);
      else setTimeout(poll, 1);
      pause = Math.min(pause * 2, 1);
    };
    poll();
  });
}

/** Deferred calls dropped before they are final, each handed back to the collector until it is, so
 *  none is freed while the library still writes into it. */
const unfinished = new FinalizationRegistry<CompiledDeferred>((call) => {
  if (call.status() === Status.Deferred) unfinished.register({}, call);
});

/**
 *  A call submitted by a deferred receiver, final once its executor finishes it.
 *
 *  It keeps everything the library reads or writes alive until then - the outcome, the inputs, the
 *  outputs and every handle it names - so none of them is collected under the call. `status` polls
 *  without blocking, `wait` blocks and answers, and `await` polls the status between turns of the
 *  event loop, never parking a thread; both answer the same cached result, or throw the same
 *  failure, however often they are asked. On the
 *  WebAssembly backend the call is final once submitted.
 *
 *  ```js
 *  const encoded = await call; // or: call.wait()
 *  ```
 */
export class Deferred<T> implements PromiseLike<T>, Disposable {
  readonly #call: CompiledDeferred;
  readonly #answer: (raw: never) => T;
  #outcome: { value: T } | { error: unknown } | undefined;
  #settling: Promise<T> | undefined;

  /** Made by a verb of a deferred receiver, never directly. */
  constructor(call: CompiledDeferred, answer: (raw: never) => T) {
    this.#call = call;
    this.#answer = answer;
    unfinished.register(this, call, this);
  }

  /** `Status.Deferred` until the call is final, then `Status.Success` or its
   *  failure. Calls `uform_outcome_status`. */
  get status(): Status {
    return this.#call.status() as Status;
  }

  /** Blocks the calling thread until the call is final, then answers its result or throws its
   *  failure. Calls `uform_outcome_wait`. */
  wait(): T {
    if (!this.#outcome) {
      this.#call.wait();
      this.#settle();
    }
    return this.#unwrap();
  }

  /** Polls until the call is final without blocking the event loop, which is what `await` reaches
   *  for. Calls `uform_outcome_status` between turns, backing off from 1 µs up to 1 ms. */
  then<Fulfilled = T, Rejected = never>(
    onFulfilled?: ((value: T) => Fulfilled | PromiseLike<Fulfilled>) | null,
    onRejected?: ((reason: unknown) => Rejected | PromiseLike<Rejected>) | null,
  ): Promise<Fulfilled | Rejected> {
    this.#settling ??= this.#outcome
      ? Promise.resolve().then(() => this.#unwrap())
      : polled(this.#call).then(() => {
          this.#settle();
          return this.#unwrap();
        });
    return this.#settling.then(onFulfilled, onRejected);
  }

  /** Waits until the call is final, so `using` never leaves one running past its block. */
  [Symbol.dispose](): void {
    if (this.#outcome) return;
    this.#call.wait();
    this.#settle();
  }

  #settle(): void {
    if (this.#outcome) return;
    try {
      this.#outcome = { value: this.#answer(this.#call.result() as never) };
    } catch (error) {
      this.#outcome = { error };
    }
    unfinished.unregister(this);
  }

  #unwrap(): T {
    const outcome = this.#outcome as { value: T } | { error: unknown };
    if ("error" in outcome) throw outcome.error;
    return outcome.value;
  }
}

/** One verb's native call, given the completion its receiver picks, and how its answer lifts. */
export interface Call<Answered> {
  readonly run: (completion: Completion) => unknown;
  readonly answer: (raw: never) => Answered;
}

/** Pairs a verb's native call with the lift of its answer. */
export function callOf<Raw, Answered>(
  run: (completion: Completion) => unknown,
  answer: (raw: Raw) => Answered,
): Call<Answered> {
  return { run, answer: answer as (raw: never) => Answered };
}

/** Runs a call to its end on the receiver's thread or executor, answering its lifted result. */
export function block<Answered>(call: Call<Answered>): Answered {
  return (call.answer as (raw: unknown) => Answered)(call.run(0));
}

/** Submits a call to the receiver's executor, answering the {@link Deferred} that owns it. */
export function defer<Answered>(call: Call<Answered>): Deferred<Answered> {
  return new Deferred(call.run(1) as CompiledDeferred, call.answer);
}
