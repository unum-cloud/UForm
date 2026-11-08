/**
 *  @file javascript/types.ts
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The vocabulary every verb shares: closed-set names, the backend the classes run on, the
 *      executor whose threads a verb runs on, the arena every scaling output lands in, and the
 *      deferred call a deferred receiver answers.
 */
import type { Compiled, CompiledArena, CompiledDeferred, CompiledExecutor } from "./library.js";

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

/**
 *  How a mesh's NVLink domains reduce.
 *  @enum {string}
 *  @readonly
 */
export enum MultimemTier {
  SwitchReduced = "switch-reduced",
  PeerRead = "peer-read",
}

/**
 *  How a contrastive term turns a row of scores into forces, numbered as
 *  `uform_pair_loss_form_t` numbers them.
 *  @enum {number}
 *  @readonly
 */
export enum PairLossForm {
  /** The term's own form: sigmoid for pooled embeddings, ranked for token vectors. */
  Derived = 0,
  /** The sigmoid cross-entropy of every cell, its attract and repel halves normalized apart. */
  Balanced = 1,
  /** The sigmoid cross-entropy of every positive's score over every negative's, where
   *  the bias cancels. */
  Ranked = 2,
  /** SigLIP's sigmoid cross-entropy of every cell, each repel weighed like an attract. */
  Sigmoid = 3,
}

/**
 *  Which side of a late-interaction score an item takes, numbered as
 *  `uform_late_interaction_side_t` numbers them.
 *  @enum {number}
 *  @readonly
 */
export enum LateInteractionSide {
  /** The item takes no part in late interaction. */
  None = 0,
  /** The item's tokens score the documents it attracts and repels. */
  Query = 1,
  /** The item's tokens are scored by the queries naming it. */
  Document = 2,
  /** The item plays both sides. */
  QueryAndDocument = 3,
}

/**
 *  The runtime that drives a GPU, numbered as `uform_gpu_vendor_t` numbers them.
 *  @enum {number}
 *  @readonly
 */
export enum GpuVendor {
  Unknown = 0,
  Nvidia = 1,
  Amd = 2,
  Apple = 3,
}

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

/**
 *  Lists every GPU this process can drive, NVIDIA devices first, then AMD, so a device's
 *  position is the ordinal VRAM placements name it by; a machine without GPUs lists
 *  none. Calls `uform_gpu_devices`.
 */
export function gpuDevices(): GpuDevice[] {
  return backend().gpuDevices() as GpuDevice[];
}

/** Where the host threads of a RAM executor are placed, how many there are, and how many deferred
 *  calls it tracks. */
export interface ExecutorForRamConfig {
  /** NUMA node IDs the threads are spread over; absent spreads them over every node. */
  memoryDomains?: number[];
  /** The CPU class to dispatch as, spelled like "intel-haswell"; absent derives the CPU's own. */
  cpuClass?: string;
  /** Threads to spawn; absent takes 7/8 of the cores the domains carry. */
  threads?: number;
  /** Deferred calls tracked at once; absent takes a default. */
  deferredLimit?: number;
}

/** Where the borrowed pool of a RAM executor is placed, and how many deferred calls it tracks. */
export interface ExecutorOnPoolConfig {
  /** NUMA node IDs of the models it will drive; absent derives every node. */
  memoryDomains?: number[];
  /** The CPU class to dispatch as, spelled like "intel-haswell"; absent derives the CPU's own. */
  cpuClass?: string;
  /** Deferred calls tracked at once; absent takes a default. */
  deferredLimit?: number;
}

/** Which devices a VRAM executor drives, and on which streams. */
export interface ExecutorForVramConfig {
  /** Device ordinals, positions in {@link gpuDevices}, to drive; absent takes one free device. */
  devices?: number[];
  /** The GPU class every device dispatches as, spelled like "nvidia-hopper"; absent derives
   *  the devices' own. */
  gpuClass?: string;
  /** NUMA node IDs whose cores the host decoders take; absent takes every node. */
  memoryDomains?: number[];
  /** The CPU class the host decoders dispatch as, spelled like "intel-haswell"; absent derives
   *  the CPU's own. */
  cpuClass?: string;
  /** Borrowed CUDA stream addresses, one per device in `devices` order; absent creates its own. */
  streams?: bigint[];
  /** Ceiling on scratch bytes per device; absent lets scratch grow as batches ask. */
  scratchLimit?: number;
  /** Deferred calls tracked at once; absent takes a default. */
  deferredLimit?: number;
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

  /** Spawns host threads for RAM-resident models. Calls `uform_executor_for_ram`. */
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
   *  Waits for every call submitted to this executor so far, then idles in
   *  `mood`. Calls `uform_executor_drain`.
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
 *  without blocking, `wait` blocks and answers, and `await` runs that wait on the libuv pool; both
 *  answer the same cached result, or throw the same failure, however often they are asked. On the
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

  /** Waits off the calling thread, which is what `await` reaches for. Calls `uform_outcome_wait` on
   *  the libuv pool. */
  then<Fulfilled = T, Rejected = never>(
    onFulfilled?: ((value: T) => Fulfilled | PromiseLike<Fulfilled>) | null,
    onRejected?: ((reason: unknown) => Rejected | PromiseLike<Rejected>) | null,
  ): Promise<Fulfilled | Rejected> {
    this.#settling ??= this.#outcome
      ? Promise.resolve().then(() => this.#unwrap())
      : this.#call.settled().then(() => {
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

/** How a receiver hands back what its verbs answer: as is, or as a {@link Deferred} of it. */
export type Answer<Receiving, Answered> = Receiving extends "deferred" ? Deferred<Answered> : Answered;

/** The completion a receiver submits with, and the way it wraps the answer. */
export function submitAs<Receiving extends "blocking" | "deferred", Raw, Answered>(
  receiving: Receiving,
  call: (completion: 0 | 1) => unknown,
  answer: (raw: Raw) => Answered,
): Answer<Receiving, Answered> {
  if (receiving === "deferred")
    return new Deferred(call(1) as CompiledDeferred, answer as (raw: never) => Answered) as Answer<Receiving, Answered>;
  return answer(call(0) as Raw) as Answer<Receiving, Answered>;
}
