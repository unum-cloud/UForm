//
//  Encode.swift
//  UForm
//
//  The forward passes, mirroring `c/uform/encode.h`: text, images, video and audio in, pooled embeddings and
//  per-token states out, one verb per input form, each on the model and on both of its receivers.
//

import UFormC

/// The surfaces an encode call keeps from its one forward, each deciding whether the binding passes the C
/// output pointer of the same name; `.embeddings` alone by default.
public struct Outputs: OptionSet, Sendable {
    public let rawValue: UInt8
    public init(rawValue: UInt8) { self.rawValue = rawValue }

    /// One pooled embedding per item, read past the attention pool and the projection head.
    public static let embeddings = Outputs(rawValue: 1 << 0)
    /// The final-norm states, one row per token each item kept after truncation.
    public static let states = Outputs(rawValue: 1 << 1)
}

/// What an encode call reports beside its outputs. Mirrors every `uform_*_encode_stats_t`.
public struct EncodeStats: Equatable, Sendable {
    /// Items cut at the model's context.
    public var truncated: Int
    /// Tokens the call encoded, after truncation: bytes of text, patches, tubes or frames.
    public var tokens: Int
}

/// The knobs every encode verb takes beside its inputs, outputs and arena.
public struct EncodeConfig {
    /// The scalar every output lands as.
    public var outputsScalar: Scalar
    /// Called with items encoded.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(outputsScalar: Scalar = .f32, progress: Progress? = nil, progressSteps: Int = 0) {
        self.outputsScalar = outputsScalar
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// What one encode call wrote into its arena, read in place until the arena's next call.
public struct Encoded {
    /// One row per item, or nil when `outputs` left the embeddings out.
    public let embeddings: Rows?
    /// One run of rows per item, one row per token kept, or nil when `outputs` left the states out.
    public let states: RowsTape?
    /// What the call counted on the way.
    public let stats: EncodeStats
}

// MARK: - Model

extension Model {
    /// Encodes UTF-8 texts, a byte per token. Calls `uform_text_encode`.
    public func textEncode(
        _ texts: [String],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try here.textEncode(texts, outputs: outputs, arena: arena, config: config)
    }

    /// Decodes and encodes JPEG, PNG or WebP files. Calls `uform_image_encode`.
    public func imageEncode(
        _ images: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try here.imageEncode(images, outputs: outputs, arena: arena, config: config)
    }

    /// Encodes decoded images of interleaved 8-bit RGB, item `i` holding `heights[i]` rows of `widths[i]` pixels.
    /// Calls `uform_image_encode_pixels`.
    public func imageEncodePixels(
        _ pixels: [[UInt8]],
        heights: [UInt32],
        widths: [UInt32],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try here.imageEncodePixels(
            pixels,
            heights: heights,
            widths: widths,
            outputs: outputs,
            arena: arena,
            config: config
        )
    }

    /// Decodes and encodes MP4 or WebM clips. Calls `uform_video_encode`.
    public func videoEncode(
        _ videos: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try here.videoEncode(videos, outputs: outputs, arena: arena, config: config)
    }

    /// Decodes and encodes WAV, FLAC, MP3 or Opus recordings. Calls `uform_audio_encode`.
    public func audioEncode(
        _ recordings: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try here.audioEncode(recordings, outputs: outputs, arena: arena, config: config)
    }

    /// Encodes decoded mono recordings sharing one `sampleRate`, resampled to the model's where it differs.
    /// Calls `uform_audio_encode_samples`.
    public func audioEncodeSamples(
        _ samples: [[Float]],
        sampleRate: UInt32,
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try here.audioEncodeSamples(samples, sampleRate: sampleRate, outputs: outputs, arena: arena, config: config)
    }
}

// MARK: - Receivers

extension ModelOn {
    /// Calls `uform_text_encode`.
    public func textEncode(
        _ texts: [String],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try run { (call) throws(UFormError) in try call.textEncode(texts, outputs, arena, config) }
    }

    /// Calls `uform_image_encode`.
    public func imageEncode(
        _ images: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try run { (call) throws(UFormError) in try call.imageEncode(images, outputs, arena, config) }
    }

    /// Calls `uform_image_encode_pixels`.
    public func imageEncodePixels(
        _ pixels: [[UInt8]],
        heights: [UInt32],
        widths: [UInt32],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try run { (call) throws(UFormError) in
            try call.imageEncodePixels(pixels, heights, widths, outputs, arena, config)
        }
    }

    /// Calls `uform_video_encode`.
    public func videoEncode(
        _ videos: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try run { (call) throws(UFormError) in try call.videoEncode(videos, outputs, arena, config) }
    }

    /// Calls `uform_audio_encode`.
    public func audioEncode(
        _ recordings: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try run { (call) throws(UFormError) in try call.audioEncode(recordings, outputs, arena, config) }
    }

    /// Calls `uform_audio_encode_samples`.
    public func audioEncodeSamples(
        _ samples: [[Float]],
        sampleRate: UInt32,
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Encoded {
        try run { (call) throws(UFormError) in
            try call.audioEncodeSamples(samples, sampleRate, outputs, arena, config)
        }
    }
}

extension ModelDeferredOn {
    /// Calls `uform_text_encode`.
    public func textEncode(
        _ texts: [String],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Deferred<Encoded> {
        try submit { (call) throws(UFormError) in try call.textEncode(texts, outputs, arena, config) }
    }

    /// Calls `uform_image_encode`.
    public func imageEncode(
        _ images: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Deferred<Encoded> {
        try submit { (call) throws(UFormError) in try call.imageEncode(images, outputs, arena, config) }
    }

    /// Calls `uform_image_encode_pixels`.
    public func imageEncodePixels(
        _ pixels: [[UInt8]],
        heights: [UInt32],
        widths: [UInt32],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Deferred<Encoded> {
        try submit { (call) throws(UFormError) in
            try call.imageEncodePixels(pixels, heights, widths, outputs, arena, config)
        }
    }

    /// Calls `uform_video_encode`.
    public func videoEncode(
        _ videos: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Deferred<Encoded> {
        try submit { (call) throws(UFormError) in try call.videoEncode(videos, outputs, arena, config) }
    }

    /// Calls `uform_audio_encode`.
    public func audioEncode(
        _ recordings: [[UInt8]],
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Deferred<Encoded> {
        try submit { (call) throws(UFormError) in try call.audioEncode(recordings, outputs, arena, config) }
    }

    /// Calls `uform_audio_encode_samples`.
    public func audioEncodeSamples(
        _ samples: [[Float]],
        sampleRate: UInt32,
        outputs: Outputs = .embeddings,
        arena: Arena,
        config: EncodeConfig = EncodeConfig()
    ) throws(UFormError) -> Deferred<Encoded> {
        try submit { (call) throws(UFormError) in
            try call.audioEncodeSamples(samples, sampleRate, outputs, arena, config)
        }
    }
}

// MARK: - Calls

/// The fields every `uform_*_encode_stats_t` shares.
protocol EncodeStatsFields {
    init()
    var truncated: Int { get }
    var tokens: Int { get }
}

extension uform_text_encode_stats_t: EncodeStatsFields {}
extension uform_image_encode_stats_t: EncodeStatsFields {}
extension uform_video_encode_stats_t: EncodeStatsFields {}
extension uform_audio_encode_stats_t: EncodeStatsFields {}

/// The fields every encode argument structure shares, so one function lowers what the six verbs have in common.
protocol EncodeArguments {
    associatedtype Stats: EncodeStatsFields
    init()
    var model: uform_model_t! { get set }
    var executor: uform_executor_t! { get set }
    var outcome: UnsafeMutablePointer<uform_outcome_t>! { get set }
    var completion: uform_completion_t { get set }
    var count: Int { get set }
    var embeddings: UnsafeMutablePointer<uform_rows_view_t>! { get set }
    var states: UnsafeMutablePointer<uform_rows_tape_view_t>! { get set }
    var arena: uform_arena_t! { get set }
    // swift-format-ignore: AlwaysUseLowerCamelCase
    var outputs_scalar: uform_scalar_t { get set }
    var progress: uform_progress_t { get set }
    var stats: UnsafeMutablePointer<Stats>! { get set }
}

extension uform_text_encode_t: EncodeArguments {}
extension uform_image_encode_t: EncodeArguments {}
extension uform_image_encode_pixels_t: EncodeArguments {}
extension uform_video_encode_t: EncodeArguments {}
extension uform_audio_encode_t: EncodeArguments {}
extension uform_audio_encode_samples_t: EncodeArguments {}

extension ModelCall {
    /// Fills what every encode verb shares into `arguments`, already holding the verb's own inputs, runs `verb`
    /// over it, and reads the surfaces `outputs` named.
    private func encode<Arguments: EncodeArguments>(
        _ arguments: Arguments,
        count: Int,
        outputs: Outputs,
        arena: Arena,
        config: EncodeConfig,
        _ verb: (UnsafeMutablePointer<Arguments>) -> Void
    ) throws(UFormError) -> Submission<Encoded> {
        var arguments = arguments
        let embeddings = outputs.contains(.embeddings) ? landing.slots(uform_rows_view_t()) : nil
        let states = outputs.contains(.states) ? landing.slots(uform_rows_tape_view_t()) : nil
        let stats = landing.slots(Arguments.Stats())
        let write = landing.lend(arena)
        arguments.model = model.raw
        arguments.executor = executor?.raw
        arguments.outcome = landing.outcome
        arguments.completion = landing.completion
        arguments.count = count
        arguments.embeddings = embeddings
        arguments.states = states
        arguments.arena = arena.raw
        arguments.outputs_scalar = config.outputsScalar.raw
        arguments.progress = landing.lend(config.progress, steps: config.progressSteps)
        arguments.stats = stats
        verb(&arguments)
        return try landing.submitted {
            Encoded(
                embeddings: embeddings.map { Rows(write: write, view: $0.pointee, count: count) },
                states: states.map { RowsTape(write: write, view: $0.pointee, count: count) },
                stats: EncodeStats(truncated: stats.pointee.truncated, tokens: stats.pointee.tokens)
            )
        }
    }

    func textEncode(
        _ texts: [String],
        _ outputs: Outputs,
        _ arena: Arena,
        _ config: EncodeConfig
    ) throws(UFormError) -> Submission<Encoded> {
        var arguments = uform_text_encode_t()
        arguments.texts = landing.tape(texts.map(\.utf8), .u8)
        return try encode(arguments, count: texts.count, outputs: outputs, arena: arena, config: config) {
            uform_text_encode($0)
        }
    }

    func imageEncode(
        _ images: [[UInt8]],
        _ outputs: Outputs,
        _ arena: Arena,
        _ config: EncodeConfig
    ) throws(UFormError) -> Submission<Encoded> {
        var arguments = uform_image_encode_t()
        arguments.images = landing.tape(images, .u8)
        return try encode(arguments, count: images.count, outputs: outputs, arena: arena, config: config) {
            uform_image_encode($0)
        }
    }

    func imageEncodePixels(
        _ pixels: [[UInt8]],
        _ heights: [UInt32],
        _ widths: [UInt32],
        _ outputs: Outputs,
        _ arena: Arena,
        _ config: EncodeConfig
    ) throws(UFormError) -> Submission<Encoded> {
        // The C side reads one height and one width per item behind pointers that carry no count.
        guard heights.count == pixels.count, widths.count == pixels.count else {
            throw UFormError(status: .shapeMismatch, message: "the heights and widths must name one per item")
        }
        var arguments = uform_image_encode_pixels_t()
        arguments.pixels = landing.tape(pixels, .u8)
        arguments.heights = landing.lend(heights)
        arguments.widths = landing.lend(widths)
        return try encode(arguments, count: pixels.count, outputs: outputs, arena: arena, config: config) {
            uform_image_encode_pixels($0)
        }
    }

    func videoEncode(
        _ videos: [[UInt8]],
        _ outputs: Outputs,
        _ arena: Arena,
        _ config: EncodeConfig
    ) throws(UFormError) -> Submission<Encoded> {
        var arguments = uform_video_encode_t()
        arguments.videos = landing.tape(videos, .u8)
        return try encode(arguments, count: videos.count, outputs: outputs, arena: arena, config: config) {
            uform_video_encode($0)
        }
    }

    func audioEncode(
        _ recordings: [[UInt8]],
        _ outputs: Outputs,
        _ arena: Arena,
        _ config: EncodeConfig
    ) throws(UFormError) -> Submission<Encoded> {
        var arguments = uform_audio_encode_t()
        arguments.recordings = landing.tape(recordings, .u8)
        return try encode(arguments, count: recordings.count, outputs: outputs, arena: arena, config: config) {
            uform_audio_encode($0)
        }
    }

    func audioEncodeSamples(
        _ samples: [[Float]],
        _ sampleRate: UInt32,
        _ outputs: Outputs,
        _ arena: Arena,
        _ config: EncodeConfig
    ) throws(UFormError) -> Submission<Encoded> {
        var arguments = uform_audio_encode_samples_t()
        arguments.samples = landing.tape(samples, .f32)
        arguments.sample_rate = sampleRate
        return try encode(arguments, count: samples.count, outputs: outputs, arena: arena, config: config) {
            uform_audio_encode_samples($0)
        }
    }
}
