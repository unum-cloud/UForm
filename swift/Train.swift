//
//  Train.swift
//  UForm
//
//  Training steps, mirroring `c/uform/train.h`: the mesh joining a run's processes, the trainer owning its
//  replicas and optimizer, and the closed sets that configure them. Training is GPU-only, so every constructor
//  here takes a VRAM executor.
//

import UFormC

// MARK: - Closed Sets

/// What the bytes of one training item hold, carrying the C enumerator.
public struct Modality: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_modality_t

    /// UTF-8 text, a byte per token.
    public static let text = Modality(raw: uform_modality_text_k)
    /// A JPEG, PNG or WebP file.
    public static let image = Modality(raw: uform_modality_image_k)
    /// An MP4 or WebM file.
    public static let video = Modality(raw: uform_modality_video_k)
    /// A WAV, FLAC, MP3 or Opus file.
    public static let audio = Modality(raw: uform_modality_audio_k)

    /// The modality as the library spells it. Calls `uform_modality_name`.
    public var description: String { String(cString: uform_modality_name(raw)) }

    /// Reads a spelling back, and nil for one no modality carries. Calls `uform_modality_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_modality_from_name, uform_modality_name) else { return nil }
        self.raw = raw
    }

    init(raw: uform_modality_t) { self.raw = raw }
}

/// Which losses a step relaxes, carrying the C enumerator.
public struct Objective: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_objective_t

    /// Masked reconstruction alone, as the Base pretrains.
    public static let masked = Objective(raw: uform_objective_masked_k)
    /// The pairwise contrastive loss alone, as the Embedder trains.
    public static let contrastive = Objective(raw: uform_objective_contrastive_k)
    /// Both, joined at the final-norm states.
    public static let joint = Objective(raw: uform_objective_joint_k)

    /// The objective as the library spells it. Calls `uform_objective_name`.
    public var description: String { String(cString: uform_objective_name(raw)) }

    /// Reads a spelling back, and nil for one no objective carries. Calls `uform_objective_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_objective_from_name, uform_objective_name) else { return nil }
        self.raw = raw
    }

    init(raw: uform_objective_t) { self.raw = raw }
}

/// Which pairs the contrastive loss pushes apart, carrying the C enumerator.
public struct Negatives: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_negatives_t

    /// Repel edges, and every pair no edge names, at the ambient repulsion weight.
    public static let ambient = Negatives(raw: uform_negatives_ambient_k)
    /// Repel edges alone.
    public static let named = Negatives(raw: uform_negatives_named_k)

    /// The policy as the library spells it. Calls `uform_negatives_name`.
    public var description: String { String(cString: uform_negatives_name(raw)) }

    /// Reads a spelling back, and nil for one no policy carries. Calls `uform_negatives_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, uform_negatives_from_name, uform_negatives_name) else { return nil }
        self.raw = raw
    }

    init(raw: uform_negatives_t) { self.raw = raw }
}

/// How a mesh's NVLink domains reduce, carrying the C enumerator.
public struct MultimemTier: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: uform_multimem_tier_t

    /// The NVLink switch reduces through multicast memory.
    public static let switchReduced = MultimemTier(raw: uform_multimem_tier_switch_reduced_k)
    /// Every member reads its peers in a fixed order, where multicast is absent.
    public static let peerRead = MultimemTier(raw: uform_multimem_tier_peer_read_k)

    /// The tier as the library spells it. Calls `uform_multimem_tier_name`.
    public var description: String { String(cString: uform_multimem_tier_name(raw)) }

    init(raw: uform_multimem_tier_t) { self.raw = raw }
}

/// How a contrastive term turns a row of scores into forces, carrying the C enumerator.
public struct PairLossForm: Hashable, Sendable {
    /// The C enumerator.
    let raw: uform_pair_loss_form_t

    /// The term's own form: sigmoid for pooled embeddings, ranked for token vectors.
    public static let derived = PairLossForm(raw: uform_pair_loss_form_derived_k)
    /// The sigmoid cross-entropy of every cell, its attract and repel halves normalized apart.
    public static let balanced = PairLossForm(raw: uform_pair_loss_form_balanced_k)
    /// The sigmoid cross-entropy of every positive's score over every negative's, where the bias cancels.
    public static let ranked = PairLossForm(raw: uform_pair_loss_form_ranked_k)
    /// SigLIP's sigmoid cross-entropy of every cell, each repel weighed like an attract.
    public static let sigmoid = PairLossForm(raw: uform_pair_loss_form_sigmoid_k)
}

/// Which side of a late-interaction score an item takes, carrying the C enumerator.
public struct LateInteractionSide: Hashable, Sendable {
    /// The C enumerator.
    let raw: uform_late_interaction_side_t

    /// The item takes no part in late interaction.
    public static let none = LateInteractionSide(raw: uform_late_interaction_side_none_k)
    /// The item's tokens score the documents it attracts and repels.
    public static let query = LateInteractionSide(raw: uform_late_interaction_side_query_k)
    /// The item's tokens are scored by the queries naming it.
    public static let document = LateInteractionSide(raw: uform_late_interaction_side_document_k)
    /// The item plays both sides.
    public static let queryAndDocument = LateInteractionSide(raw: uform_late_interaction_side_query_and_document_k)
}

// MARK: - Mesh

/// Where this process sits among the processes training one model, and how they meet. Mirrors
/// `uform_mesh_options_t`, every zero deriving as the C field documents.
public struct MeshOptions: Sendable {
    /// This process's index among `hostDomainCount`.
    public var hostDomainIndex: Int
    /// Processes in the run, one per host; zero derives one, a run with no peers.
    public var hostDomainCount: Int
    /// GPUs one NVLink multicast domain spans; zero derives the host.
    public var gpusPerMultimemDomain: Int
    /// The directory every host sees, where processes exchange their facts; required with peers.
    public var directory: String
    /// The run's unique name inside `directory`; required with peers.
    public var job: String
    /// Seconds any wait for a peer lasts before the call fails; zero derives 120.
    public var timeoutSeconds: Double

    public init(
        hostDomainIndex: Int = 0,
        hostDomainCount: Int = 0,
        gpusPerMultimemDomain: Int = 0,
        directory: String = "",
        job: String = "",
        timeoutSeconds: Double = 0
    ) {
        self.hostDomainIndex = hostDomainIndex
        self.hostDomainCount = hostDomainCount
        self.gpusPerMultimemDomain = gpusPerMultimemDomain
        self.directory = directory
        self.job = job
        self.timeoutSeconds = timeoutSeconds
    }
}

/// What a mesh reports about itself. Mirrors `uform_mesh_description_t`.
public struct MeshDescription: Sendable {
    /// GPUs one process drives.
    public var gpusPerHostDomain: Int
    /// GPUs one NVLink domain spans.
    public var gpusPerMultimemDomain: Int
    /// GPUs in the whole run.
    public var gpusPerColocatedDomain: Int
    /// This process's index among the run's processes.
    public var hostDomainIndex: Int
    /// The run-wide index of this process's first GPU, where its share of a sharded dataset begins.
    public var firstGpuIndex: Int
    /// How its NVLink domains reduce.
    public var multimemTier: MultimemTier

    init(_ raw: uform_mesh_description_t) {
        gpusPerHostDomain = raw.gpus_per_host_domain
        gpusPerMultimemDomain = raw.gpus_per_multimem_domain
        gpusPerColocatedDomain = raw.gpus_per_colocated_domain
        hostDomainIndex = raw.host_domain_index
        firstGpuIndex = raw.first_gpu_index
        multimemTier = MultimemTier(raw: raw.multimem_tier)
    }
}

/// One process's place among the processes of a run, made once and borrowed by every trainer.
public final class Mesh: @unchecked Sendable {
    /// The native handle, released by `deinit`.
    let raw: uform_mesh_t
    /// The VRAM executor the mesh was made over, held so the mesh leaves first.
    let executor: Executor

    private init(raw: uform_mesh_t, executor: Executor) {
        self.raw = raw
        self.executor = executor
    }

    /// Calls `uform_mesh_free`.
    deinit { uform_mesh_free(raw) }

    /// Joins this process to the run's mesh over the devices of `executor`; every process of the run calls it
    /// with the same count, multicast extent, directory and job. Calls `uform_mesh_bootstrap`.
    public static func bootstrap(executor: Executor, options: MeshOptions = MeshOptions()) throws(UFormError) -> Mesh {
        let landing = Landing()
        let mesh = landing.slots(uform_mesh_t?.none)
        var arguments = uform_mesh_bootstrap_t(
            mesh: mesh,
            executor: executor.raw,
            outcome: landing.outcome,
            completion: landing.completion,
            options: uform_mesh_options_t(
                host_domain_index: options.hostDomainIndex,
                host_domain_count: options.hostDomainCount,
                gpus_per_multimem_domain: options.gpusPerMultimemDomain,
                directory: landing.lend(options.directory),
                job: landing.lend(options.job),
                timeout_seconds: options.timeoutSeconds
            )
        )
        uform_mesh_bootstrap(&arguments)
        return try landing.read { () throws(UFormError) in Mesh(raw: try created(mesh), executor: executor) }
    }

    /// Reports the mesh's shape and this process's place in it. Calls `uform_mesh_describe`.
    public func describe() throws(UFormError) -> MeshDescription {
        let landing = Landing()
        let description = landing.slots(uform_mesh_description_t())
        var arguments = uform_mesh_describe_t(mesh: raw, outcome: landing.outcome, description: description)
        uform_mesh_describe(&arguments)
        return try landing.read { MeshDescription(description.pointee) }
    }
}

// MARK: - Trainer Configuration

/// The image slice of a batch envelope; a zero `patchSide` leaves images out of the run.
public struct ImageShape: Sendable {
    /// The square patch side images tile into, in pixels.
    public var patchSide: Int
    /// Channels every pixel carries; zero derives four.
    public var channels: Int

    public init(patchSide: Int = 0, channels: Int = 0) {
        self.patchSide = patchSide
        self.channels = channels
    }
}

/// The video slice of a batch envelope; a zero `patchSide` leaves video out of the run.
public struct VideoShape: Sendable {
    /// The square patch side one tube tiles into, in pixels.
    public var patchSide: Int
    /// Frames one tube folds ahead of its patch; zero derives two.
    public var tubeFrames: Int
    /// Source frames between sampled ones; zero derives one.
    public var frameStride: Int
    /// Channels every pixel carries; zero derives three.
    public var channels: Int

    public init(patchSide: Int = 0, tubeFrames: Int = 0, frameStride: Int = 0, channels: Int = 0) {
        self.patchSide = patchSide
        self.tubeFrames = tubeFrames
        self.frameStride = frameStride
        self.channels = channels
    }
}

/// The audio slice of a batch envelope; a zero `chunkSamples` leaves audio out of the run.
public struct AudioShape: Sendable {
    /// Raw samples per frame, the fan-in of the learned frontend.
    public var chunkSamples: Int
    /// The mono rate every recording is resampled to; zero derives 16000.
    public var sampleRate: Int

    public init(chunkSamples: Int = 0, sampleRate: Int = 0) {
        self.chunkSamples = chunkSamples
        self.sampleRate = sampleRate
    }
}

/// What one device's step carries at most, whatever the modality mix. Mirrors `uform_batch_shape_t`.
public struct BatchShape: Sendable {
    /// Rows one device's step packs.
    public var maxRowsPerDevice: Int
    /// Items one device's step carries across all its rows.
    public var maxSegmentsPerDevice: Int
    /// Positions one row spans, the sequence length.
    public var maxColumnsPerRow: Int
    /// Bags one device's step carries; zero keeps every item its own routing unit.
    public var maxBagsPerDevice: Int
    /// Legal bytes one item offers to choose from at most; zero makes no choices.
    public var maxChoicesPerItem: Int
    /// The image geometry.
    public var image: ImageShape
    /// The video geometry.
    public var video: VideoShape
    /// The audio geometry.
    public var audio: AudioShape

    public init(
        maxRowsPerDevice: Int = 0,
        maxSegmentsPerDevice: Int = 0,
        maxColumnsPerRow: Int = 0,
        maxBagsPerDevice: Int = 0,
        maxChoicesPerItem: Int = 0,
        image: ImageShape = ImageShape(),
        video: VideoShape = VideoShape(),
        audio: AudioShape = AudioShape()
    ) {
        self.maxRowsPerDevice = maxRowsPerDevice
        self.maxSegmentsPerDevice = maxSegmentsPerDevice
        self.maxColumnsPerRow = maxColumnsPerRow
        self.maxBagsPerDevice = maxBagsPerDevice
        self.maxChoicesPerItem = maxChoicesPerItem
        self.image = image
        self.video = video
        self.audio = audio
    }

    var raw: uform_batch_shape_t {
        uform_batch_shape_t(
            max_rows_per_device: maxRowsPerDevice,
            max_segments_per_device: maxSegmentsPerDevice,
            max_columns_per_row: maxColumnsPerRow,
            max_bags_per_device: maxBagsPerDevice,
            max_choices_per_item: maxChoicesPerItem,
            image: uform_image_shape_t(patch_side: image.patchSide, channels: image.channels),
            video: uform_video_shape_t(
                patch_side: video.patchSide,
                tube_frames: video.tubeFrames,
                frame_stride: video.frameStride,
                channels: video.channels
            ),
            audio: uform_audio_shape_t(chunk_samples: audio.chunkSamples, sample_rate: audio.sampleRate)
        )
    }
}

/// How the masked objective corrupts a batch and weighs its heads. Mirrors `uform_masking_t`, every zero deriving
/// the default the C field names.
public struct Masking: Sendable {
    public var tokenMaskProbability: Float = 0
    public var tokenMeanSpan: Float = 0
    public var replaceFraction: Float = 0
    public var randomFraction: Float = 0
    public var imageMaskRatio: Float = 0
    public var videoMaskRatio: Float = 0
    public var audioMaskRatio: Float = 0
    public var textWeight: Float = 0
    public var imageWeight: Float = 0
    public var videoWeight: Float = 0
    public var audioWeight: Float = 0

    public init() {}

    var raw: uform_masking_t {
        uform_masking_t(
            token_mask_probability: tokenMaskProbability,
            token_mean_span: tokenMeanSpan,
            replace_fraction: replaceFraction,
            random_fraction: randomFraction,
            image_mask_ratio: imageMaskRatio,
            video_mask_ratio: videoMaskRatio,
            audio_mask_ratio: audioMaskRatio,
            text_weight: textWeight,
            image_weight: imageWeight,
            video_weight: videoWeight,
            audio_weight: audioWeight
        )
    }
}

/// How the contrastive loss scores the pooled embeddings. Mirrors `uform_contrastive_t`, every zero deriving the
/// default the C field names.
public struct Contrastive: Sendable {
    /// Matryoshka prefix widths, strictly ascending, each a multiple of 8, at most 8; empty scores the full width.
    public var matryoshkaDimensions: [Int] = []
    public var temperatureGroups: Int = 0
    public var inverseTemperature: Float = 0
    public var bias: Float = 0
    public var negatives: Negatives = .ambient
    public var ambientRepulsionWeight: Float = 0
    public var maxEdgesPerDevice: Int = 0
    public var pairLoss: PairLossForm = .derived

    public init() {}
}

/// How the late-interaction loss scores the per-token vectors of query and document items. Mirrors
/// `uform_late_interaction_t`, every zero deriving the default the C field names; a zero `weight` trains no token
/// head.
public struct LateInteraction: Sendable {
    public var weight: Float = 0
    public var width: Int = 0
    public var inverseTemperature: Float = 0
    public var bias: Float = 0
    public var maxQuerySpan: Int = 0
    public var maxQueryTokensPerDevice: Int = 0
    public var maxDocumentsPerDevice: Int = 0
    public var maxDocumentTokensPerDevice: Int = 0
    public var pairLoss: PairLossForm = .derived

    public init() {}

    var raw: uform_late_interaction_t {
        uform_late_interaction_t(
            weight: weight,
            width: width,
            inverse_temperature: inverseTemperature,
            bias: bias,
            max_query_span: maxQuerySpan,
            max_query_tokens_per_device: maxQueryTokensPerDevice,
            max_documents_per_device: maxDocumentsPerDevice,
            max_document_tokens_per_device: maxDocumentTokensPerDevice,
            pair_loss: pairLoss.raw
        )
    }
}

/// Calibrated choices over the byte decoder: each choice item scores the soft cross-entropy of its legal bytes'
/// softmax, plus `brierWeight` times its Brier score. Mirrors `uform_choices_t`.
public struct Choices: Sendable {
    /// The loss weight; zero trains no choices.
    public var weight: Float
    /// The Brier term's weight beside the soft cross-entropy; zero derives 0.5.
    public var brierWeight: Float

    public init(weight: Float = 0, brierWeight: Float = 0) {
        self.weight = weight
        self.brierWeight = brierWeight
    }
}

/// The learning-rate schedule: a linear warmup to the peak, then a cosine decay to the floor. Mirrors
/// `uform_schedule_t`.
public struct Schedule: Sendable {
    /// The peak learning rate; zero derives 5e-3.
    public var learningRate: Float
    /// The floor the cosine settles at; zero derives 1e-5.
    public var minLearningRate: Float
    /// Steps from zero to the peak; zero derives a tenth of `steps`, at most 400.
    public var warmupSteps: Int
    /// Steps the run takes across resumes, which the cosine spans; zero is refused.
    public var steps: Int

    public init(learningRate: Float = 0, minLearningRate: Float = 0, warmupSteps: Int = 0, steps: Int = 0) {
        self.learningRate = learningRate
        self.minLearningRate = minLearningRate
        self.warmupSteps = warmupSteps
        self.steps = steps
    }
}

// MARK: - Steps

/// One edge set over a step's items, as compressed sparse rows: item `i` links to `ids[offsets[i] ..<
/// offsets[i + 1]]`. Mirrors `uform_edges_view_t`.
public struct Edges: Sendable {
    /// The step's count + 1 non-decreasing offsets into `ids`.
    public var offsets: [Int]
    /// Indices into the step's items, ascending within each run.
    public var ids: [UInt32]
    /// The weight of each edge, paired with `ids`; nil weighs every edge one.
    public var weights: [Float]?

    public init(offsets: [Int], ids: [UInt32], weights: [Float]? = nil) {
        self.offsets = offsets
        self.ids = ids
        self.weights = weights
    }
}

/// The contrastive targets of one step; a pair `ignore` names is skipped, one `attract` names pulls together, one
/// `repel` names pushes apart, and every other pair follows the run's negatives. Mirrors `uform_relations_view_t`.
public struct Relations: Sendable {
    public var attract: Edges?
    public var repel: Edges?
    public var ignore: Edges?

    public init(attract: Edges? = nil, repel: Edges? = nil, ignore: Edges? = nil) {
        self.attract = attract
        self.repel = repel
        self.ignore = ignore
    }
}

/// Each item's legal choices and their targets: item `i` offers the distinct bytes `bytes[offsets[i] ..<
/// offsets[i + 1]]`, an item offering none makes no choice, and the targets pair with `bytes`, summing to one per
/// item. Mirrors `uform_choices_view_t`.
public struct ChoicesView: Sendable {
    /// The step's count + 1 non-decreasing offsets into `bytes`.
    public var offsets: [Int]
    /// Every item's legal bytes, end to end.
    public var bytes: [UInt8]
    /// The target probability of each byte.
    public var targets: [Float]

    public init(offsets: [Int], bytes: [UInt8], targets: [Float]) {
        self.offsets = offsets
        self.bytes = bytes
        self.targets = targets
    }
}

/// What a step reports, every loss its world mean. Mirrors `uform_trainer_step_stats_t`.
public struct TrainerStepStats: Sendable {
    /// The optimizer step the call applied, counted from one across resumes.
    public var step: Int
    /// The learning rate it applied.
    public var learningRate: Float
    /// The total loss the step minimized.
    public var loss: Float
    public var contrastiveLoss: Float
    public var maskedTextLoss: Float
    public var maskedImageLoss: Float
    public var maskedVideoLoss: Float
    public var maskedAudioLoss: Float
    public var lateInteractionLoss: Float
    public var choicesLoss: Float
    /// Items cut at a row's width.
    public var truncated: Int

    init(_ raw: uform_trainer_step_stats_t) {
        step = raw.step
        learningRate = raw.learning_rate
        loss = raw.loss
        contrastiveLoss = raw.contrastive_loss
        maskedTextLoss = raw.masked_text_loss
        maskedImageLoss = raw.masked_image_loss
        maskedVideoLoss = raw.masked_video_loss
        maskedAudioLoss = raw.masked_audio_loss
        lateInteractionLoss = raw.late_interaction_loss
        choicesLoss = raw.choices_loss
        truncated = raw.truncated
    }
}

/// What one step wrote: its report, and the pooled embeddings it scored when given an arena.
public struct TrainerStep {
    public let stats: TrainerStepStats
    /// One row per item, as the contrastive loss scored it before the update, or nil without an arena.
    public let embeddings: Rows?
}

// MARK: - Trainer

/// A training run: the tower replicas, gradients, optimizer moments, heads and temperatures on the devices of
/// its executor, every verb blocking on that executor.
public final class Trainer: @unchecked Sendable {
    /// The native handle, released by `deinit`.
    let raw: uform_trainer_t
    /// The VRAM executor the trainer was made on, which every verb names.
    let executor: Executor
    /// The mesh it borrows until freed.
    let mesh: Mesh?

    private init(raw: uform_trainer_t, executor: Executor, mesh: Mesh?) {
        self.raw = raw
        self.executor = executor
        self.mesh = mesh
    }

    /// Calls `uform_trainer_free`.
    deinit { uform_trainer_free(raw) }

    /// Builds one replica per device of `executor` and starts the run from `seed`, or continues it from the
    /// `resume` checkpoint; every process of the run calls it with the same geometry, objective and seed. Calls
    /// `uform_trainer_init_vram`.
    public static func initVram(
        executor: Executor,
        depth: Int,
        hidden: Int,
        batch: BatchShape,
        objective: Objective = .joint,
        masking: Masking = Masking(),
        contrastive: Contrastive = Contrastive(),
        lateInteraction: LateInteraction = LateInteraction(),
        choices: Choices = Choices(),
        schedule: Schedule,
        mesh: Mesh? = nil,
        resume: String = "",
        seed: UInt64 = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) throws(UFormError) -> Trainer {
        let landing = Landing()
        let trainer = landing.slots(uform_trainer_t?.none)
        var arguments = uform_trainer_init_vram_t()
        arguments.trainer = trainer
        arguments.executor = executor.raw
        arguments.outcome = landing.outcome
        arguments.completion = landing.completion
        arguments.resume = landing.lend(resume)
        arguments.depth = depth
        arguments.hidden = hidden
        arguments.batch = batch.raw
        arguments.objective = objective.raw
        arguments.masking = masking.raw
        arguments.contrastive = uform_contrastive_t(
            matryoshka_dimensions: landing.lend(contrastive.matryoshkaDimensions),
            matryoshka_dimensions_count: contrastive.matryoshkaDimensions.count,
            temperature_groups: contrastive.temperatureGroups,
            inverse_temperature: contrastive.inverseTemperature,
            bias: contrastive.bias,
            negatives: contrastive.negatives.raw,
            ambient_repulsion_weight: contrastive.ambientRepulsionWeight,
            max_edges_per_device: contrastive.maxEdgesPerDevice,
            pair_loss: contrastive.pairLoss.raw
        )
        arguments.late_interaction = lateInteraction.raw
        arguments.choices = uform_choices_t(weight: choices.weight, brier_weight: choices.brierWeight)
        arguments.schedule = uform_schedule_t(
            learning_rate: schedule.learningRate,
            min_learning_rate: schedule.minLearningRate,
            warmup_steps: schedule.warmupSteps,
            steps: schedule.steps
        )
        arguments.mesh = mesh?.raw
        arguments.seed = seed
        arguments.progress = landing.lend(progress, steps: progressSteps)
        uform_trainer_init_vram(&arguments)
        return try landing.read { () throws(UFormError) in
            Trainer(raw: try created(trainer), executor: executor, mesh: mesh)
        }
    }

    /// Applies one optimizer step over this process's items and every peer's; every process of the run calls it
    /// once per step. With an `arena`, the step also lands the pooled embeddings it scored there, as
    /// `outputsScalar`. `sides` places each item in the late-interaction loss, nil leaving every item out, and
    /// `choices` names the legal bytes and targets the choices term reads. Calls `uform_trainer_step`.
    public func step(
        modalities: [Modality],
        contents: [[UInt8]],
        bags: [UInt32]? = nil,
        temperatureGroups: [UInt32]? = nil,
        sides: [LateInteractionSide]? = nil,
        relations: Relations = Relations(),
        choices: ChoicesView? = nil,
        arena: Arena? = nil,
        outputsScalar: Scalar = .f32
    ) throws(UFormError) -> TrainerStep {
        let count = contents.count
        // The C side reads one entry per item, and one offset more, behind pointers that carry no count.
        let edges = [relations.attract, relations.repel, relations.ignore].compactMap { $0 }
        guard modalities.count == count, bags.map({ $0.count == count }) ?? true,
            temperatureGroups.map({ $0.count == count }) ?? true, sides.map({ $0.count == count }) ?? true,
            edges.allSatisfy({ $0.offsets.count == count + 1 && ($0.weights?.count ?? $0.ids.count) == $0.ids.count }),
            choices.map({ $0.offsets.count == count + 1 && $0.targets.count == $0.bytes.count }) ?? true
        else {
            throw UFormError(status: .shapeMismatch, message: "every per-item list must name one entry per item")
        }
        let landing = Landing()
        let stats = landing.slots(uform_trainer_step_stats_t())
        let embeddings = arena.map { _ in landing.slots(uform_rows_view_t()) }
        let write = arena.map { landing.lend($0) }
        func lent(_ edges: Edges?) -> uform_edges_view_t {
            guard let edges else { return uform_edges_view_t() }
            return uform_edges_view_t(
                offsets: landing.lend(edges.offsets),
                ids: landing.lend(edges.ids),
                weights: edges.weights.flatMap { landing.lend($0) }
            )
        }
        var arguments = uform_trainer_step_t()
        arguments.trainer = raw
        arguments.executor = executor.raw
        arguments.outcome = landing.outcome
        arguments.completion = landing.completion
        arguments.count = count
        arguments.modalities = landing.lend(modalities.map(\.raw))
        arguments.contents = landing.tape(contents, .u8)
        arguments.bags = bags.flatMap { landing.lend($0) }
        arguments.temperature_groups = temperatureGroups.flatMap { landing.lend($0) }
        arguments.sides = sides.flatMap { landing.lend($0.map(\.raw)) }
        if let choices {
            arguments.choices = uform_choices_view_t(
                offsets: landing.lend(choices.offsets),
                bytes: landing.lend(choices.bytes),
                targets: landing.lend(choices.targets)
            )
        }
        arguments.relations = uform_relations_view_t(
            attract: lent(relations.attract),
            repel: lent(relations.repel),
            ignore: lent(relations.ignore)
        )
        arguments.embeddings = embeddings
        arguments.arena = arena?.raw
        arguments.outputs_scalar = outputsScalar.raw
        arguments.stats = stats
        uform_trainer_step(&arguments)
        return try landing.read {
            TrainerStep(
                stats: TrainerStepStats(stats.pointee),
                embeddings: embeddings.map { Rows(write: write!, view: $0.pointee, count: count) }
            )
        }
    }

    /// Writes the weights and the step to a safetensors checkpoint, which `Model.loadVram` and its siblings serve
    /// and `initVram` resumes; only the process of index zero writes. Calls `uform_trainer_save`.
    public func save(path: String, progress: Progress? = nil, progressSteps: Int = 0) throws(UFormError) {
        let landing = Landing()
        var arguments = uform_trainer_save_t(
            trainer: raw,
            executor: executor.raw,
            outcome: landing.outcome,
            completion: landing.completion,
            path: landing.lend(path),
            progress: landing.lend(progress, steps: progressSteps)
        )
        uform_trainer_save(&arguments)
        try landing.read {}
    }
}
