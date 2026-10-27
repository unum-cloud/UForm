/**
 *  @file javascript/train.ts
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief Training: the mesh a run's processes join, and the trainer whose steps join masked reconstruction and the
 *      pairwise contrastive loss into one optimizer step. GPU-only, so the WebAssembly backend reports
 *      `unsupported`.
 */
import type { CompiledMesh, CompiledMeshDescription, CompiledStepped, CompiledTrainer, Completion } from "./library.js";
import { Rows, type Tape, bytesTape, samplesTape } from "./encode.js";
import {
  Answer,
  Arena,
  Executor,
  LateInteractionSide,
  Modality,
  MultimemTier,
  Negatives,
  Objective,
  PairLossForm,
  Scalar,
  Status,
  backend,
  fail,
  modalityCode,
  negativesCode,
  objectiveCode,
  scalarCode,
  submitAs,
} from "./types.js";

/** Where this process sits among the processes training one model, and how they meet. Mirrors `uform_mesh_options_t`. */
export interface MeshOptions {
  /** This process's index among `hostDomainCount`. */
  hostDomainIndex?: number;
  /** Processes in the run, one per host; absent derives one, a run with no peers. */
  hostDomainCount?: number;
  /** GPUs one NVLink multicast domain spans; absent derives the host. */
  gpusPerMultimemDomain?: number;
  /** The path of the directory every host sees, crossing as its UTF-8 bytes; required with peers. */
  directory?: string;
  /** The run's unique name, naming one directory inside `directory`; required with peers. */
  job?: string;
  /** Seconds any wait for a peer lasts before the call fails; absent derives 120. */
  timeoutSeconds?: number;
}

/** What a mesh reports about itself. Mirrors `uform_mesh_description_t`. */
export interface MeshDescription extends Omit<CompiledMeshDescription, "multimemTier"> {
  /** How its NVLink domains reduce. */
  multimemTier: MultimemTier;
}

/** The token the bootstrap holds, which is what a direct `new Mesh` cannot produce. */
const meshSeal = Symbol("Mesh");

/** One process's place among the processes of a run, made once and borrowed by every trainer. */
export class Mesh implements Disposable {
  /** The backend handle a trainer borrows. */
  readonly compiled: CompiledMesh;

  /** Reached through `Mesh.bootstrap`. */
  private constructor(seal: symbol, compiled: CompiledMesh) {
    if (seal !== meshSeal) throw fail(Status.InvalidArgument, "A Mesh is made by Mesh.bootstrap.");
    this.compiled = compiled;
  }

  /**
   *  Joins this process to the run's mesh, once every process has joined. Calls `uform_mesh_bootstrap`.
   *  @param executor - A VRAM executor whose devices are this process's members of the mesh.
   *  @param options - Where this process sits and how the processes meet.
   */
  static bootstrap(executor: Executor, options: MeshOptions = {}): Mesh {
    return new Mesh(meshSeal, backend().CompiledMesh.bootstrap(executor.compiled, options));
  }

  /** Reports the mesh's shape and this process's place in it. Calls `uform_mesh_describe`. */
  describe(): MeshDescription {
    const described = this.compiled.describe();
    return { ...described, multimemTier: described.multimemTier as MultimemTier };
  }

  /** Leaves the mesh; every trainer borrowing it must be freed first. Calls `uform_mesh_free`. */
  free(): void {
    this.compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/** What one device's step carries at most. Mirrors `uform_batch_shape_t`; every absent field reads zero. */
export interface BatchShape {
  maxRowsPerDevice?: number;
  maxSegmentsPerDevice?: number;
  maxColumnsPerRow?: number;
  maxBagsPerDevice?: number;
  /** Legal bytes one item offers to choose from at most; absent makes no choices. */
  maxChoicesPerItem?: number;
  image?: { patchSide?: number; channels?: number };
  video?: { patchSide?: number; tubeFrames?: number; frameStride?: number; channels?: number };
  audio?: { chunkSamples?: number; sampleRate?: number };
}

/** How the masked objective corrupts a batch and weighs its heads. Mirrors `uform_masking_t`; absent fields derive. */
export interface Masking {
  tokenMaskProbability?: number;
  tokenMeanSpan?: number;
  replaceFraction?: number;
  randomFraction?: number;
  imageMaskRatio?: number;
  videoMaskRatio?: number;
  audioMaskRatio?: number;
  textWeight?: number;
  imageWeight?: number;
  videoWeight?: number;
  audioWeight?: number;
}

/** How the contrastive loss scores the pooled embeddings. Mirrors `uform_contrastive_t`; absent fields derive. */
export interface Contrastive {
  matryoshkaDimensions?: number[];
  temperatureGroups?: number;
  inverseTemperature?: number;
  bias?: number;
  negatives?: Negatives;
  ambientRepulsionWeight?: number;
  maxEdgesPerDevice?: number;
  /** The pair loss over each anchor's row; absent keeps sigmoid. */
  pairLoss?: PairLossForm;
}

/** How the late-interaction loss scores token vectors. Mirrors `uform_late_interaction_t`; absent fields derive. */
export interface LateInteraction {
  /** The loss weight; absent trains no token head. */
  weight?: number;
  /** The token vectors' width, a multiple of 8; absent derives 128. */
  width?: number;
  inverseTemperature?: number;
  bias?: number;
  maxQuerySpan?: number;
  maxQueryTokensPerDevice?: number;
  maxDocumentsPerDevice?: number;
  maxDocumentTokensPerDevice?: number;
  /** The pair loss over each query's row of hard MaxSim scores; absent keeps ranked. */
  pairLoss?: PairLossForm;
}

/** The calibrated choices over the byte decoder. Mirrors `uform_choices_t`; absent fields derive. */
export interface Choices {
  /** The loss weight; absent trains no choices. */
  weight?: number;
  /** The Brier term's weight beside the soft cross-entropy; absent derives 0.5. */
  brierWeight?: number;
}

/** The learning-rate schedule. Mirrors `uform_schedule_t`. */
export interface Schedule {
  learningRate?: number;
  minLearningRate?: number;
  warmupSteps?: number;
  /** Steps the run takes, which the cosine spans; required. */
  steps: number;
}

/** A training run's geometry and objectives. Mirrors `uform_trainer_init_vram_t`. */
export interface TrainerConfig {
  /** The checkpoint to continue from; absent starts from `seed`. */
  resume?: string;
  /** Transformer blocks stacked. */
  depth: number;
  /** Embedding width, a multiple of 128. */
  hidden: number;
  batch: BatchShape;
  objective: Objective;
  masking?: Masking;
  contrastive?: Contrastive;
  lateInteraction?: LateInteraction;
  /** Read when its weight is positive, which needs the batch's `maxChoicesPerItem`. */
  choices?: Choices;
  schedule: Schedule;
  /** The base of every keyed draw. */
  seed?: number | bigint;
}

/** One edge set over a step's items, as compressed sparse rows. Mirrors `uform_edges_view_t`. */
export interface Edges {
  /** The step's `count + 1` offsets into `ids`. */
  offsets: Uint32Array | number[];
  /** Item indices each item links to, ascending within each run. */
  ids: Uint32Array | number[];
  /** The weight of each edge; absent weighs every edge one. */
  weights?: Float32Array | number[];
}

/** This process's share of one step. Mirrors the inputs of `uform_trainer_step_t`. */
export interface TrainerBatch {
  /** The modality of each item. */
  modalities: readonly Modality[];
  /** Each item's bytes, UTF-8 text or an encoded file, or a tape of them. */
  contents: readonly (string | Uint8Array)[] | Tape<Uint8Array>;
  /** The bag of each item; items sharing a bag land on one device. */
  bags?: Uint32Array | number[];
  /** The temperature group of each item. */
  temperatureGroups?: Uint32Array | number[];
  /** The late-interaction side of each item; absent makes every item `LateInteractionSide.None`. */
  sides?: readonly LateInteractionSide[];
  /**
   *  Each item's legal choices, one byte each, as `textChoose` takes them; an item offering none makes no choice. Read
   *  when the run trains choices; the call appends the mask byte to each choosing text.
   */
  choices?: readonly (string | Uint8Array)[] | Tape<Uint8Array>;
  /** The target probability of each choice, an entry per item as long as its choices and summing to one, as
   *  `textChoose` answers its `Probabilities`; required with `choices`. */
  choiceTargets?: readonly Float32Array[] | Tape<Float32Array>;
  /** The contrastive targets. */
  relations?: { attract?: Edges; repel?: Edges; ignore?: Edges };
}

/** Where a step lands the embeddings it scored, when a caller wants them. */
export interface TrainerStepConfig {
  /** The memory the embeddings land in; absent skips the copy. */
  arena?: Arena;
  /** The scalar they land as; absent lands `f32`. */
  outputsScalar?: Scalar;
}

/** What one step reports, and the embeddings it scored when an arena was given. */
export interface Stepped {
  stats: CompiledStepped["stats"];
  embeddings?: Rows;
}

function uint32Of(values: Uint32Array | number[] | undefined): Uint32Array | undefined {
  return values === undefined || values instanceof Uint32Array ? values : Uint32Array.from(values);
}

function float32Of(values: Float32Array | number[] | undefined): Float32Array | undefined {
  return values === undefined || values instanceof Float32Array ? values : Float32Array.from(values);
}

/** The verbs every receiver of a trainer shares. */
abstract class TrainerVerbs<Receiving extends "blocking" | "deferred"> {
  protected constructor(
    protected readonly compiled: CompiledTrainer,
    readonly executor: Executor,
    protected readonly receiving: Receiving,
  ) {}

  /**
   *  Applies one optimizer step over this process's items and every peer's. Calls `uform_trainer_step`.
   *  @param batch - This process's items, their modalities, bags and relations.
   *  @param config - Where the scored embeddings land.
   */
  step(batch: TrainerBatch, config: TrainerStepConfig = {}): Answer<Receiving, Stepped> {
    const contents = bytesTape(batch.contents);
    const modalities = Int32Array.from(batch.modalities, modalityCode);
    const { attract, repel, ignore } = batch.relations ?? {};
    if ((batch.choices === undefined) !== (batch.choiceTargets === undefined))
      throw fail(Status.InvalidArgument, "Pass both 'choices' and 'choiceTargets', or neither.");
    const choices = batch.choices && bytesTape(batch.choices);
    const choiceTargets = batch.choiceTargets && samplesTape(batch.choiceTargets);
    const arena = config.arena?.compiled;
    const scalar = arena ? scalarCode(config.outputsScalar ?? Scalar.F32) : undefined;
    return submitAs(
      this.receiving,
      (completion: Completion) =>
        this.compiled.step(
          this.executor.compiled,
          completion,
          modalities,
          contents.data,
          contents.offsets,
          uint32Of(batch.bags),
          uint32Of(batch.temperatureGroups),
          batch.sides && Int32Array.from(batch.sides),
          choices?.data,
          choices?.offsets,
          choiceTargets?.data,
          choiceTargets?.offsets,
          uint32Of(attract?.offsets),
          uint32Of(attract?.ids),
          float32Of(attract?.weights),
          uint32Of(repel?.offsets),
          uint32Of(repel?.ids),
          float32Of(repel?.weights),
          uint32Of(ignore?.offsets),
          uint32Of(ignore?.ids),
          arena,
          scalar,
        ),
      (stepped: CompiledStepped) => ({
        stats: stepped.stats,
        embeddings: stepped.embeddings && new Rows(stepped.embeddings),
      }),
    );
  }

  /**
   *  Writes the weights and the step as a safetensors checkpoint; only process zero writes. Calls `uform_trainer_save`.
   *  @param path - The destination, written whole and renamed into place.
   */
  save(path: string): Answer<Receiving, void> {
    return submitAs(
      this.receiving,
      (completion: Completion) => this.compiled.save(this.executor.compiled, completion, path),
      () => undefined,
    );
  }
}

/** The trainer's verbs, submitted to its executor and answered as {@link Deferred} calls: `trainer.deferred()`. */
export class TrainerDeferredOn extends TrainerVerbs<"deferred"> {
  /** Made by `trainer.deferred()`, never directly. */
  constructor(compiled: CompiledTrainer, executor: Executor) {
    super(compiled, executor, "deferred");
  }
}

/** The token the creation holds, which is what a direct `new Trainer` cannot produce. */
const trainerSeal = Symbol("Trainer");

/**
 *  A training run on a VRAM executor's devices, every verb blocking on that executor; `deferred()` submits them there
 *  instead.
 */
export class Trainer extends TrainerVerbs<"blocking"> implements Disposable {
  /** Reached through `Trainer.initVram`. */
  private constructor(seal: symbol, compiled: CompiledTrainer, executor: Executor) {
    if (seal !== trainerSeal) throw fail(Status.InvalidArgument, "A Trainer is made by Trainer.initVram.");
    super(compiled, executor, "blocking");
  }

  /**
   *  Builds one replica per device and starts or resumes the run. Calls `uform_trainer_init_vram`.
   *  @param executor - A VRAM executor, every device of which holds one replica.
   *  @param config - The geometry, objectives and schedule.
   *  @param mesh - The mesh to borrow until the trainer is freed; absent runs this process alone.
   */
  static initVram(executor: Executor, config: TrainerConfig, mesh?: Mesh): Trainer {
    const coded = {
      ...config,
      objective: objectiveCode(config.objective),
      contrastive: config.contrastive && {
        ...config.contrastive,
        negatives: config.contrastive.negatives && negativesCode(config.contrastive.negatives),
      },
    };
    return new Trainer(trainerSeal, backend().CompiledTrainer.initVram(executor.compiled, mesh?.compiled, coded), executor);
  }

  /** The same verbs, each submitted to the trainer's executor and answered as a {@link Deferred}. Binding-only. */
  deferred(): TrainerDeferredOn {
    return new TrainerDeferredOn(this.compiled, this.executor);
  }

  /** Frees the trainer and lets go of its mesh. Calls `uform_trainer_free`. */
  free(): void {
    this.compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}
