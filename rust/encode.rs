//! The forward passes: text, images, video and audio in, pooled embeddings, per-token states and
//! token vectors out, one verb per input form, and calibrated choices over text states.
//!
//! Every verb runs the same tower and keeps whichever surfaces its [`Outputs`] name: one pooled
//! embedding per item, one final-norm state or one token-head vector per token left after
//! truncation, or any mix from one forward. All land in the arena the verb borrows, written as the
//! config's `outputs_scalar`.
//!
//! Models are tokenizer-free: text arrives as UTF-8 bytes and every byte is a token. Separate items
//! are gathered into one tape before the call, while a `(data, offsets)` pair reaches the library
//! as it is.
//!
//! File: rust/encode.rs
//! Author: Ash Vardanian
//! Date: October 18, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{BitOr, BitOrAssign};
use core::ptr;

use crate::model::{uform_model_t, Answer, ModelVerbs};
use crate::types::{
    fail, lower_progress, uform_arena_t, uform_completion_t, uform_executor_t, uform_outcome_t, uform_progress_t,
    uform_rows_tape_view_t, uform_rows_view_t, uform_scalar_t, uform_tape_view_t, Accepts, Arena, Callbacks, Calling,
    Completion, Deferring, Exported, Gathered, Handles, ProgressCallback, Result, Rows, RowsTape, Scalar, Status, Tape,
    TapeView,
};

// region: Raw Declarations

pub type uform_text_encode_stats_t = TextEncodeStats;
pub type uform_image_encode_stats_t = ImageEncodeStats;
pub type uform_video_encode_stats_t = VideoEncodeStats;
pub type uform_audio_encode_stats_t = AudioEncodeStats;

#[repr(C)]
pub struct uform_text_encode_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub texts: uform_tape_view_t,
    pub embeddings: *mut uform_rows_view_t,
    pub states: *mut uform_rows_tape_view_t,
    pub token_vectors: *mut uform_rows_tape_view_t,
    pub arena: uform_arena_t,
    pub outputs_scalar: uform_scalar_t,
    pub progress: uform_progress_t,
    pub stats: *mut uform_text_encode_stats_t,
}

#[repr(C)]
pub struct uform_image_encode_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub images: uform_tape_view_t,
    pub embeddings: *mut uform_rows_view_t,
    pub states: *mut uform_rows_tape_view_t,
    pub token_vectors: *mut uform_rows_tape_view_t,
    pub arena: uform_arena_t,
    pub outputs_scalar: uform_scalar_t,
    pub progress: uform_progress_t,
    pub stats: *mut uform_image_encode_stats_t,
}

#[repr(C)]
pub struct uform_image_encode_pixels_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub pixels: uform_tape_view_t,
    pub heights: *const u32,
    pub widths: *const u32,
    pub embeddings: *mut uform_rows_view_t,
    pub states: *mut uform_rows_tape_view_t,
    pub token_vectors: *mut uform_rows_tape_view_t,
    pub arena: uform_arena_t,
    pub outputs_scalar: uform_scalar_t,
    pub progress: uform_progress_t,
    pub stats: *mut uform_image_encode_stats_t,
}

#[repr(C)]
pub struct uform_video_encode_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub videos: uform_tape_view_t,
    pub embeddings: *mut uform_rows_view_t,
    pub states: *mut uform_rows_tape_view_t,
    pub token_vectors: *mut uform_rows_tape_view_t,
    pub arena: uform_arena_t,
    pub outputs_scalar: uform_scalar_t,
    pub progress: uform_progress_t,
    pub stats: *mut uform_video_encode_stats_t,
}

#[repr(C)]
pub struct uform_audio_encode_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub recordings: uform_tape_view_t,
    pub embeddings: *mut uform_rows_view_t,
    pub states: *mut uform_rows_tape_view_t,
    pub token_vectors: *mut uform_rows_tape_view_t,
    pub arena: uform_arena_t,
    pub outputs_scalar: uform_scalar_t,
    pub progress: uform_progress_t,
    pub stats: *mut uform_audio_encode_stats_t,
}

#[repr(C)]
pub struct uform_audio_encode_samples_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub samples: uform_tape_view_t,
    pub sample_rate: u32,
    pub embeddings: *mut uform_rows_view_t,
    pub states: *mut uform_rows_tape_view_t,
    pub token_vectors: *mut uform_rows_tape_view_t,
    pub arena: uform_arena_t,
    pub outputs_scalar: uform_scalar_t,
    pub progress: uform_progress_t,
    pub stats: *mut uform_audio_encode_stats_t,
}

#[repr(C)]
pub struct uform_text_choose_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub texts: uform_tape_view_t,
    pub choices: uform_tape_view_t,
    pub temperature_groups: *const u32,
    pub probabilities: *mut uform_tape_view_t,
    pub arena: uform_arena_t,
    pub progress: uform_progress_t,
    pub stats: *mut uform_text_encode_stats_t,
}

extern "C" {
    pub fn uform_text_encode(arguments: *mut uform_text_encode_t);
    pub fn uform_text_choose(arguments: *mut uform_text_choose_t);
    pub fn uform_image_encode(arguments: *mut uform_image_encode_t);
    pub fn uform_image_encode_pixels(arguments: *mut uform_image_encode_pixels_t);
    pub fn uform_video_encode(arguments: *mut uform_video_encode_t);
    pub fn uform_audio_encode(arguments: *mut uform_audio_encode_t);
    pub fn uform_audio_encode_samples(arguments: *mut uform_audio_encode_samples_t);
}

// endregion: Raw Declarations

// region: Outputs

/// The surfaces an encode verb keeps from its one forward, named after the C fields they fill;
/// [`Outputs::default`] keeps the embeddings alone.
///
/// Rust-only; decides which of the C `embeddings`, `states` and `token_vectors` pointers
/// a verb passes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Outputs(u32);

impl Outputs {
    /// One pooled embedding per item, past the attention pool and the projection head.
    pub const EMBEDDINGS: Outputs = Outputs(1);
    /// One final-norm state per token left after truncation.
    pub const STATES: Outputs = Outputs(2);
    /// One unit vector of the token head per token left after truncation, the surface
    /// late-interaction rerankers score; a model without a token head fails the call with
    /// [`Status::Unsupported`].
    pub const TOKEN_VECTORS: Outputs = Outputs(4);

    /// Whether every surface `other` names is kept.
    pub const fn contains(self, other: Outputs) -> bool { self.0 & other.0 == other.0 }
}

impl Default for Outputs {
    fn default() -> Self { Outputs::EMBEDDINGS }
}

impl BitOr for Outputs {
    type Output = Outputs;
    fn bitor(self, other: Outputs) -> Outputs { Outputs(self.0 | other.0) }
}

impl BitOrAssign for Outputs {
    fn bitor_assign(&mut self, other: Outputs) { self.0 |= other.0 }
}

impl fmt::Debug for Outputs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = [
            (Outputs::EMBEDDINGS, "EMBEDDINGS"),
            (Outputs::STATES, "STATES"),
            (Outputs::TOKEN_VECTORS, "TOKEN_VECTORS"),
        ];
        let mut kept = names
            .iter()
            .filter(|(flag, _)| self.contains(*flag))
            .map(|(_, name)| *name);
        formatter.write_str(kept.next().unwrap_or("NONE"))?;
        kept.try_for_each(|name| write!(formatter, " | {name}"))
    }
}

/// What an encode verb wrote into the arena it borrows: the surfaces its [`Outputs`] named.
///
/// Rust-only.
pub struct Encoded<'arena> {
    /// One pooled embedding of the model's dimensions per item, without normalization.
    pub embeddings: Option<Rows<'arena>>,
    /// One final-norm state per token left after truncation, item by item.
    pub states: Option<RowsTape<'arena>>,
    /// One unit vector of the model's token width per token left after truncation, the prefix
    /// included, item by item.
    pub token_vectors: Option<RowsTape<'arena>>,
}

/// The output views a verb lends the library, and which of them it asked for.
struct Surfaces {
    outputs: Outputs,
    embeddings: Exported<uform_rows_view_t>,
    states: Exported<uform_rows_tape_view_t>,
    token_vectors: Exported<uform_rows_tape_view_t>,
    count: usize,
}

/// The C `embeddings`, `states` and `token_vectors` pointers of one call, null where skipped.
struct LentSurfaces {
    embeddings: *mut uform_rows_view_t,
    states: *mut uform_rows_tape_view_t,
    token_vectors: *mut uform_rows_tape_view_t,
}

impl Surfaces {
    fn of(outputs: Outputs) -> Self {
        Surfaces {
            outputs,
            embeddings: Exported::default(),
            states: Exported::default(),
            token_vectors: Exported::default(),
            count: 0,
        }
    }

    /// The output pointers for a call over `count` items.
    fn lend(&mut self, count: usize) -> LentSurfaces {
        self.count = count;
        let outputs = self.outputs;
        fn kept<View>(kept: bool, view: &mut View) -> *mut View {
            match kept {
                true => ptr::from_mut(view),
                false => ptr::null_mut(),
            }
        }
        LentSurfaces {
            embeddings: kept(outputs.contains(Outputs::EMBEDDINGS), &mut self.embeddings.0),
            states: kept(outputs.contains(Outputs::STATES), &mut self.states.0),
            token_vectors: kept(outputs.contains(Outputs::TOKEN_VECTORS), &mut self.token_vectors.0),
        }
    }

    fn finish(self, arena: &Arena) -> Encoded<'_> {
        Encoded {
            embeddings: self.outputs.contains(Outputs::EMBEDDINGS).then_some(Rows {
                arena,
                view: self.embeddings,
                count: self.count,
            }),
            states: self.outputs.contains(Outputs::STATES).then_some(RowsTape {
                arena,
                view: self.states,
                count: self.count,
            }),
            token_vectors: self.outputs.contains(Outputs::TOKEN_VECTORS).then_some(RowsTape {
                arena,
                view: self.token_vectors,
                count: self.count,
            }),
        }
    }
}

// endregion: Outputs

// region: Configs

/// What a text call reports beside its outputs, overwritten by the call.
///
/// Mirrors `uform_text_encode_stats_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct TextEncodeStats {
    /// Items cut at the model's context.
    pub truncated: usize,
    /// Tokens the call encoded, after truncation.
    pub tokens: usize,
}

/// What an image call reports beside its outputs, overwritten by the call.
///
/// Mirrors `uform_image_encode_stats_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct ImageEncodeStats {
    /// Items whose patch grid was cut at the model's context.
    pub truncated: usize,
    /// Patches the call encoded, after truncation.
    pub tokens: usize,
}

/// What a video call reports beside its outputs, overwritten by the call.
///
/// Mirrors `uform_video_encode_stats_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct VideoEncodeStats {
    /// Items cut at the model's context in tubes.
    pub truncated: usize,
    /// Tubes the call encoded, after truncation.
    pub tokens: usize,
}

/// What an audio call reports beside its outputs, overwritten by the call.
///
/// Mirrors `uform_audio_encode_stats_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct AudioEncodeStats {
    /// Items cut at the model's context in frames.
    pub truncated: usize,
    /// Frames the call encoded, after truncation.
    pub tokens: usize,
}

/// The scalar and the knobs of [`EncodeVerbs::text_encode`]; a bare [`Scalar`] is the whole config.
///
/// Mirrors `uform_text_encode_t`.
pub struct TextEncodeConfig<'config> {
    /// The scalar type every output lands as.
    pub outputs_scalar: Scalar,
    /// Reports the items encoded and stops wherever it says to; a stopped call reports
    /// [`Status::Cancelled`] and leaves no outputs.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the truncation report lands, overwritten by the call.
    pub stats: Option<&'config mut TextEncodeStats>,
}

impl From<Scalar> for TextEncodeConfig<'_> {
    fn from(outputs_scalar: Scalar) -> Self {
        TextEncodeConfig {
            outputs_scalar,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The scalar and the knobs of [`EncodeVerbs::image_encode`]; a bare [`Scalar`] is
/// the whole config.
///
/// Mirrors `uform_image_encode_t`.
pub struct ImageEncodeConfig<'config> {
    /// The scalar type every output lands as.
    pub outputs_scalar: Scalar,
    /// Reports the items encoded and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the truncation report lands, overwritten by the call.
    pub stats: Option<&'config mut ImageEncodeStats>,
}

impl From<Scalar> for ImageEncodeConfig<'_> {
    fn from(outputs_scalar: Scalar) -> Self {
        ImageEncodeConfig {
            outputs_scalar,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The scalar and the knobs of [`EncodeVerbs::image_encode_pixels`]; a bare [`Scalar`] is
/// the whole config.
///
/// Mirrors `uform_image_encode_pixels_t`.
pub struct ImageEncodePixelsConfig<'config> {
    /// The scalar type every output lands as.
    pub outputs_scalar: Scalar,
    /// Reports the items encoded and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the truncation report lands, overwritten by the call.
    pub stats: Option<&'config mut ImageEncodeStats>,
}

impl From<Scalar> for ImageEncodePixelsConfig<'_> {
    fn from(outputs_scalar: Scalar) -> Self {
        ImageEncodePixelsConfig {
            outputs_scalar,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The scalar and the knobs of [`EncodeVerbs::video_encode`]; a bare [`Scalar`] is
/// the whole config.
///
/// Mirrors `uform_video_encode_t`.
pub struct VideoEncodeConfig<'config> {
    /// The scalar type every output lands as.
    pub outputs_scalar: Scalar,
    /// Reports the items encoded and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the truncation report lands, overwritten by the call.
    pub stats: Option<&'config mut VideoEncodeStats>,
}

impl From<Scalar> for VideoEncodeConfig<'_> {
    fn from(outputs_scalar: Scalar) -> Self {
        VideoEncodeConfig {
            outputs_scalar,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The scalar and the knobs of [`EncodeVerbs::audio_encode`]; a bare [`Scalar`] is
/// the whole config.
///
/// Mirrors `uform_audio_encode_t`.
pub struct AudioEncodeConfig<'config> {
    /// The scalar type every output lands as.
    pub outputs_scalar: Scalar,
    /// Reports the items encoded and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the truncation report lands, overwritten by the call.
    pub stats: Option<&'config mut AudioEncodeStats>,
}

impl From<Scalar> for AudioEncodeConfig<'_> {
    fn from(outputs_scalar: Scalar) -> Self {
        AudioEncodeConfig {
            outputs_scalar,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The scalar and the knobs of [`EncodeVerbs::audio_encode_samples`]; a bare [`Scalar`] is
/// the whole config.
///
/// Mirrors `uform_audio_encode_samples_t`.
pub struct AudioEncodeSamplesConfig<'config> {
    /// The scalar type every output lands as.
    pub outputs_scalar: Scalar,
    /// Reports the items encoded and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the truncation report lands, overwritten by the call.
    pub stats: Option<&'config mut AudioEncodeStats>,
}

impl From<Scalar> for AudioEncodeSamplesConfig<'_> {
    fn from(outputs_scalar: Scalar) -> Self {
        AudioEncodeSamplesConfig {
            outputs_scalar,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The question types and the knobs of [`EncodeVerbs::text_choose`]; [`TextChooseConfig::default`]
/// puts every item in group zero.
///
/// Mirrors `uform_text_choose_t`.
#[derive(Default)]
pub struct TextChooseConfig<'config> {
    /// The question type of each item, below the model's `choice_temperature_groups`; empty puts
    /// every item in group zero.
    pub temperature_groups: &'config [u32],
    /// Reports the items scored and stops wherever it says to; a stopped call reports
    /// [`Status::Cancelled`] and leaves no outputs.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the truncation report lands, overwritten by the call.
    pub stats: Option<&'config mut TextEncodeStats>,
}

impl Accepts<TextEncodeConfig<'static>> for Deferring {}
impl Accepts<TextChooseConfig<'static>> for Deferring {}
impl Accepts<ImageEncodeConfig<'static>> for Deferring {}
impl Accepts<ImageEncodePixelsConfig<'static>> for Deferring {}
impl Accepts<VideoEncodeConfig<'static>> for Deferring {}
impl Accepts<AudioEncodeConfig<'static>> for Deferring {}
impl Accepts<AudioEncodeSamplesConfig<'static>> for Deferring {}

// endregion: Configs

// region: Verbs

/// Every encode verb, written once for every receiver: `&Model` blocks on the calling thread, and
/// the [`On`](crate::On) that [`Model::on`](crate::Model::on) or
/// [`Model::deferred`](crate::Model::deferred) makes runs on an executor.
///
/// Every verb takes its inputs, the [`Outputs`] to keep, a config every bare [`Scalar`]
/// converts into, and last the [`Arena`] its [`Encoded`] result borrows. A blocking receiver
/// answers what the call produced; a deferring one answers a [`Deferred`](crate::Deferred) and
/// takes owned inputs.
///
/// Rust-only; implemented for every receiver of [`ModelVerbs`].
pub trait EncodeVerbs<'model>: ModelVerbs<'model> {
    /// Encodes UTF-8 texts through one forward, a token per byte.
    ///
    /// Calls `uform_text_encode`.
    fn text_encode<'config, 'arena, Texts: TapeView<u8>>(
        self,
        texts: Texts,
        outputs: Outputs,
        config: impl Into<TextEncodeConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'model, Self, TextEncode<'config, 'arena, Texts>>
    where
        Self::Completion: Accepts<Texts> + Accepts<TextEncodeConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(TextEncode {
            handles: self.handles(),
            texts,
            gathered: Gathered::default(),
            surfaces: Surfaces::of(outputs),
            config: config.into(),
            arena,
        })
    }

    /// Scores each item's legal choices through one forward: `texts` are UTF-8 states, each ending
    /// where its choice goes, and `choices` holds each item's legal bytes, one choice per byte.
    /// Answers one probability per choice in `choices` order, a softmax over the item's choices
    /// alone at its question type's calibrated inverse temperature.
    ///
    /// A state longer than the model's context loses its head, never its end, and a model without
    /// the byte decoder fails the call with [`Status::Unsupported`].
    ///
    /// Calls `uform_text_choose`.
    fn text_choose<'config, 'arena, Texts: TapeView<u8>, Choices: TapeView<u8>>(
        self,
        texts: Texts,
        choices: Choices,
        config: impl Into<TextChooseConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'model, Self, TextChoose<'config, 'arena, Texts, Choices>>
    where
        Self::Completion: Accepts<Texts> + Accepts<Choices> + Accepts<TextChooseConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(TextChoose {
            handles: self.handles(),
            texts,
            choices,
            gathered_texts: Gathered::default(),
            gathered_choices: Gathered::default(),
            probabilities: Exported::default(),
            count: 0,
            config: config.into(),
            arena,
        })
    }

    /// Decodes and encodes JPEG, PNG or WebP files; an undecodable file fails the call with
    /// [`Status::Malformed`].
    ///
    /// Calls `uform_image_encode`.
    fn image_encode<'config, 'arena, Images: TapeView<u8>>(
        self,
        images: Images,
        outputs: Outputs,
        config: impl Into<ImageEncodeConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'model, Self, ImageEncode<'config, 'arena, Images>>
    where
        Self::Completion: Accepts<Images> + Accepts<ImageEncodeConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(ImageEncode {
            handles: self.handles(),
            images,
            gathered: Gathered::default(),
            surfaces: Surfaces::of(outputs),
            config: config.into(),
            arena,
        })
    }

    /// Encodes decoded images of interleaved 8-bit RGB, item `i` holding `heights[i]` rows of
    /// `widths[i]` pixels; an item of another length fails with [`Status::ShapeMismatch`].
    ///
    /// Calls `uform_image_encode_pixels`.
    fn image_encode_pixels<'config, 'arena, Pixels: TapeView<u8>, Heights: AsRef<[u32]>, Widths: AsRef<[u32]>>(
        self,
        pixels: Pixels,
        heights: Heights,
        widths: Widths,
        outputs: Outputs,
        config: impl Into<ImageEncodePixelsConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'model, Self, ImageEncodePixels<'config, 'arena, Pixels, Heights, Widths>>
    where
        Self::Completion:
            Accepts<Pixels> + Accepts<Heights> + Accepts<Widths> + Accepts<ImageEncodePixelsConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(ImageEncodePixels {
            handles: self.handles(),
            pixels,
            heights,
            widths,
            gathered: Gathered::default(),
            surfaces: Surfaces::of(outputs),
            config: config.into(),
            arena,
        })
    }

    /// Decodes and encodes MP4 or WebM clips.
    ///
    /// Calls `uform_video_encode`.
    fn video_encode<'config, 'arena, Videos: TapeView<u8>>(
        self,
        videos: Videos,
        outputs: Outputs,
        config: impl Into<VideoEncodeConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'model, Self, VideoEncode<'config, 'arena, Videos>>
    where
        Self::Completion: Accepts<Videos> + Accepts<VideoEncodeConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(VideoEncode {
            handles: self.handles(),
            videos,
            gathered: Gathered::default(),
            surfaces: Surfaces::of(outputs),
            config: config.into(),
            arena,
        })
    }

    /// Decodes and encodes WAV, FLAC, MP3 or Opus recordings, resampled to mono.
    ///
    /// Calls `uform_audio_encode`.
    fn audio_encode<'config, 'arena, Recordings: TapeView<u8>>(
        self,
        recordings: Recordings,
        outputs: Outputs,
        config: impl Into<AudioEncodeConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'model, Self, AudioEncode<'config, 'arena, Recordings>>
    where
        Self::Completion: Accepts<Recordings> + Accepts<AudioEncodeConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(AudioEncode {
            handles: self.handles(),
            recordings,
            gathered: Gathered::default(),
            surfaces: Surfaces::of(outputs),
            config: config.into(),
            arena,
        })
    }

    /// Encodes decoded mono recordings sharing one `sample_rate`, resampled to the model's rate
    /// where it differs.
    ///
    /// Calls `uform_audio_encode_samples`.
    fn audio_encode_samples<'config, 'arena, Samples: TapeView<f32>>(
        self,
        samples: Samples,
        sample_rate: u32,
        outputs: Outputs,
        config: impl Into<AudioEncodeSamplesConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'model, Self, AudioEncodeSamples<'config, 'arena, Samples>>
    where
        Self::Completion: Accepts<Samples> + Accepts<AudioEncodeSamplesConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(AudioEncodeSamples {
            handles: self.handles(),
            samples,
            sample_rate,
            gathered: Gathered::default(),
            surfaces: Surfaces::of(outputs),
            config: config.into(),
            arena,
        })
    }
}

impl<'model, Receiver: ModelVerbs<'model>> EncodeVerbs<'model> for Receiver {}

// endregion: Verbs

// region: Calls

/// A text forward; see [`EncodeVerbs::text_encode`].
///
/// Calls `uform_text_encode`.
pub struct TextEncode<'config, 'arena, Texts> {
    handles: Handles,
    texts: Texts,
    gathered: Gathered<u8>,
    surfaces: Surfaces,
    config: TextEncodeConfig<'config>,
    arena: &'arena mut Arena,
}

impl<'arena, Texts: TapeView<u8>> Calling for TextEncode<'_, 'arena, Texts> {
    type Arguments = uform_text_encode_t;
    type Output = Encoded<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.texts.tape(&mut self.gathered)?;
        let lent = self.surfaces.lend(batch.count);
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_text_encode_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            texts: batch.tape,
            embeddings: lent.embeddings,
            states: lent.states,
            token_vectors: lent.token_vectors,
            arena: self.arena.handle,
            outputs_scalar: self.config.outputs_scalar as uform_scalar_t,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_text_encode(arguments) }

    fn finish(self) -> Result<Encoded<'arena>> { Ok(self.surfaces.finish(self.arena)) }
}

/// A text forward scoring each item's legal choices; see [`EncodeVerbs::text_choose`].
///
/// Calls `uform_text_choose`.
pub struct TextChoose<'config, 'arena, Texts, Choices> {
    handles: Handles,
    texts: Texts,
    choices: Choices,
    gathered_texts: Gathered<u8>,
    gathered_choices: Gathered<u8>,
    probabilities: Exported<uform_tape_view_t>,
    count: usize,
    config: TextChooseConfig<'config>,
    arena: &'arena mut Arena,
}

impl<'arena, Texts: TapeView<u8>, Choices: TapeView<u8>> Calling for TextChoose<'_, 'arena, Texts, Choices> {
    type Arguments = uform_text_choose_t;
    type Output = Tape<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let texts = self.texts.tape(&mut self.gathered_texts)?;
        let choices = self.choices.tape(&mut self.gathered_choices)?;
        let groups = self.config.temperature_groups;
        if choices.count != texts.count || (!groups.is_empty() && groups.len() != texts.count) {
            return Err(fail(
                Status::ShapeMismatch,
                "every item needs its choices and, if any, one group",
            ));
        }
        self.count = texts.count;
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_text_choose_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count: texts.count,
            texts: texts.tape,
            choices: choices.tape,
            temperature_groups: match groups.is_empty() {
                true => ptr::null(),
                false => groups.as_ptr(),
            },
            probabilities: ptr::from_mut(&mut self.probabilities.0),
            arena: self.arena.handle,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_text_choose(arguments) }

    fn finish(self) -> Result<Tape<'arena>> {
        Ok(Tape {
            arena: self.arena,
            view: self.probabilities,
            count: self.count,
        })
    }
}

/// An image forward over encoded files; see [`EncodeVerbs::image_encode`].
///
/// Calls `uform_image_encode`.
pub struct ImageEncode<'config, 'arena, Images> {
    handles: Handles,
    images: Images,
    gathered: Gathered<u8>,
    surfaces: Surfaces,
    config: ImageEncodeConfig<'config>,
    arena: &'arena mut Arena,
}

impl<'arena, Images: TapeView<u8>> Calling for ImageEncode<'_, 'arena, Images> {
    type Arguments = uform_image_encode_t;
    type Output = Encoded<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.images.tape(&mut self.gathered)?;
        let lent = self.surfaces.lend(batch.count);
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_image_encode_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            images: batch.tape,
            embeddings: lent.embeddings,
            states: lent.states,
            token_vectors: lent.token_vectors,
            arena: self.arena.handle,
            outputs_scalar: self.config.outputs_scalar as uform_scalar_t,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_image_encode(arguments) }

    fn finish(self) -> Result<Encoded<'arena>> { Ok(self.surfaces.finish(self.arena)) }
}

/// An image forward over decoded pixels; see [`EncodeVerbs::image_encode_pixels`].
///
/// Calls `uform_image_encode_pixels`.
pub struct ImageEncodePixels<'config, 'arena, Pixels, Heights, Widths> {
    handles: Handles,
    pixels: Pixels,
    heights: Heights,
    widths: Widths,
    gathered: Gathered<u8>,
    surfaces: Surfaces,
    config: ImageEncodePixelsConfig<'config>,
    arena: &'arena mut Arena,
}

impl<'arena, Pixels: TapeView<u8>, Heights: AsRef<[u32]>, Widths: AsRef<[u32]>> Calling
    for ImageEncodePixels<'_, 'arena, Pixels, Heights, Widths>
{
    type Arguments = uform_image_encode_pixels_t;
    type Output = Encoded<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.pixels.tape(&mut self.gathered)?;
        let (heights, widths) = (self.heights.as_ref(), self.widths.as_ref());
        if heights.len() != batch.count || widths.len() != batch.count {
            return Err(fail(Status::ShapeMismatch, "every item needs one height and one width"));
        }
        let lent = self.surfaces.lend(batch.count);
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_image_encode_pixels_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            pixels: batch.tape,
            heights: heights.as_ptr(),
            widths: widths.as_ptr(),
            embeddings: lent.embeddings,
            states: lent.states,
            token_vectors: lent.token_vectors,
            arena: self.arena.handle,
            outputs_scalar: self.config.outputs_scalar as uform_scalar_t,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_image_encode_pixels(arguments) }

    fn finish(self) -> Result<Encoded<'arena>> { Ok(self.surfaces.finish(self.arena)) }
}

/// A video forward over encoded clips; see [`EncodeVerbs::video_encode`].
///
/// Calls `uform_video_encode`.
pub struct VideoEncode<'config, 'arena, Videos> {
    handles: Handles,
    videos: Videos,
    gathered: Gathered<u8>,
    surfaces: Surfaces,
    config: VideoEncodeConfig<'config>,
    arena: &'arena mut Arena,
}

impl<'arena, Videos: TapeView<u8>> Calling for VideoEncode<'_, 'arena, Videos> {
    type Arguments = uform_video_encode_t;
    type Output = Encoded<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.videos.tape(&mut self.gathered)?;
        let lent = self.surfaces.lend(batch.count);
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_video_encode_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            videos: batch.tape,
            embeddings: lent.embeddings,
            states: lent.states,
            token_vectors: lent.token_vectors,
            arena: self.arena.handle,
            outputs_scalar: self.config.outputs_scalar as uform_scalar_t,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_video_encode(arguments) }

    fn finish(self) -> Result<Encoded<'arena>> { Ok(self.surfaces.finish(self.arena)) }
}

/// An audio forward over encoded recordings; see [`EncodeVerbs::audio_encode`].
///
/// Calls `uform_audio_encode`.
pub struct AudioEncode<'config, 'arena, Recordings> {
    handles: Handles,
    recordings: Recordings,
    gathered: Gathered<u8>,
    surfaces: Surfaces,
    config: AudioEncodeConfig<'config>,
    arena: &'arena mut Arena,
}

impl<'arena, Recordings: TapeView<u8>> Calling for AudioEncode<'_, 'arena, Recordings> {
    type Arguments = uform_audio_encode_t;
    type Output = Encoded<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.recordings.tape(&mut self.gathered)?;
        let lent = self.surfaces.lend(batch.count);
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_audio_encode_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            recordings: batch.tape,
            embeddings: lent.embeddings,
            states: lent.states,
            token_vectors: lent.token_vectors,
            arena: self.arena.handle,
            outputs_scalar: self.config.outputs_scalar as uform_scalar_t,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_audio_encode(arguments) }

    fn finish(self) -> Result<Encoded<'arena>> { Ok(self.surfaces.finish(self.arena)) }
}

/// An audio forward over decoded samples; see [`EncodeVerbs::audio_encode_samples`].
///
/// Calls `uform_audio_encode_samples`.
pub struct AudioEncodeSamples<'config, 'arena, Samples> {
    handles: Handles,
    samples: Samples,
    sample_rate: u32,
    gathered: Gathered<f32>,
    surfaces: Surfaces,
    config: AudioEncodeSamplesConfig<'config>,
    arena: &'arena mut Arena,
}

impl<'arena, Samples: TapeView<f32>> Calling for AudioEncodeSamples<'_, 'arena, Samples> {
    type Arguments = uform_audio_encode_samples_t;
    type Output = Encoded<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.samples.tape(&mut self.gathered)?;
        let lent = self.surfaces.lend(batch.count);
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_audio_encode_samples_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            samples: batch.tape,
            sample_rate: self.sample_rate,
            embeddings: lent.embeddings,
            states: lent.states,
            token_vectors: lent.token_vectors,
            arena: self.arena.handle,
            outputs_scalar: self.config.outputs_scalar as uform_scalar_t,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_audio_encode_samples(arguments) }

    fn finish(self) -> Result<Encoded<'arena>> { Ok(self.surfaces.finish(self.arena)) }
}

// endregion: Calls

// region: Tests

#[cfg(test)]
mod tests {
    use core::future::{Future, IntoFuture};
    use core::ops::ControlFlow;
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};

    use super::*;
    use crate::model::Model;
    use crate::types::{Error, Executor, Mood};

    fn model() -> Result<Model> { Model::load_ram_buffer(&crate::model::tests::checkpoint()) }

    /// CRC-32 of `bytes`, as PNG chunks carry it.
    fn crc32(bytes: &[u8]) -> u32 {
        !bytes.iter().fold(!0u32, |crc, &byte| {
            (0..8).fold(crc ^ u32::from(byte), |crc, _| {
                (crc >> 1) ^ (0xEDB8_8320 & 0u32.wrapping_sub(crc & 1))
            })
        })
    }

    /// A valid 8 x 8 RGB PNG of a gradient, its pixels in one stored deflate block.
    fn png() -> Vec<u8> {
        let raw: Vec<u8> = (0..8u32)
            .flat_map(|row| core::iter::once(0).chain((0..24u32).map(move |byte| (row * 32 + byte * 8) as u8)))
            .collect();
        let (low, high) = raw.iter().fold((1u32, 0u32), |(low, high), &byte| {
            let low = (low + u32::from(byte)) % 65521;
            (low, (high + low) % 65521)
        });
        let length = raw.len() as u16;
        let stream = [
            &[0x78, 0x01, 0x01][..],
            &length.to_le_bytes(),
            &(!length).to_le_bytes(),
            &raw,
            &((high << 16) | low).to_be_bytes(),
        ]
        .concat();
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        for (kind, data) in [
            (b"IHDR", &[0, 0, 0, 8, 0, 0, 0, 8, 8, 2, 0, 0, 0][..]),
            (b"IDAT", &stream),
            (b"IEND", &[]),
        ] {
            let body = [&kind[..], data].concat();
            png.extend((data.len() as u32).to_be_bytes());
            png.extend(&body);
            png.extend(crc32(&body).to_be_bytes());
        }
        png
    }

    /// A valid 16 kHz mono 16-bit WAV of a sawtooth.
    fn wav() -> Vec<u8> {
        let samples: Vec<u8> = (0..1600i32)
            .flat_map(|sample| (((sample % 64) * 512 - 16384) as i16).to_le_bytes())
            .collect();
        let format = [1u16, 1].map(u16::to_le_bytes).concat();
        let rates = [16_000u32, 32_000].map(u32::to_le_bytes).concat();
        let framing = [2u16, 16].map(u16::to_le_bytes).concat();
        [
            &b"RIFF"[..],
            &(36 + samples.len() as u32).to_le_bytes(),
            b"WAVEfmt ",
            &16u32.to_le_bytes(),
            &format,
            &rates,
            &framing,
            b"data",
            &(samples.len() as u32).to_le_bytes(),
            &samples,
        ]
        .concat()
    }

    /// One executor per test, since an executor orders the calls submitted to it.
    fn executor() -> Result<Executor> { Executor::for_ram(&[], None, 1, 0) }

    /// The rows an embeddings surface holds, copied out of the arena.
    fn embeddings_of(encoded: &Encoded<'_>) -> Result<Vec<f32>> {
        let embeddings = encoded
            .embeddings
            .as_ref()
            .ok_or(fail(Status::Internal, "no embeddings were kept"))?;
        Ok(embeddings.values::<f32>()?.to_vec())
    }

    #[test]
    fn outputs_name_the_surfaces_they_keep() {
        let both = Outputs::EMBEDDINGS | Outputs::STATES;
        assert!(both.contains(Outputs::STATES) && both.contains(Outputs::EMBEDDINGS));
        assert!(!Outputs::STATES.contains(Outputs::EMBEDDINGS));
        assert_eq!(Outputs::default(), Outputs::EMBEDDINGS);
        assert_eq!(format!("{both:?}"), "EMBEDDINGS | STATES");
        assert_eq!(format!("{:?}", Outputs::TOKEN_VECTORS), "TOKEN_VECTORS");
    }

    /// One embedding per item, one state per byte, and the same rows from every receiver.
    #[test]
    fn texts_land_one_embedding_per_item_and_one_state_per_byte() -> Result<()> {
        let model = model()?;
        let dimensions = model.describe()?.dimensions;
        let texts = ["a caption", "", "another, longer caption"];
        let mut arena = Arena::new()?;
        let mut stats = TextEncodeStats::default();
        let config = TextEncodeConfig {
            stats: Some(&mut stats),
            ..TextEncodeConfig::from(Scalar::F32)
        };
        let encoded = model.text_encode(&texts, Outputs::EMBEDDINGS | Outputs::STATES, config, &mut arena)?;

        let embeddings = encoded
            .embeddings
            .as_ref()
            .ok_or(fail(Status::Internal, "no embeddings"))?;
        assert_eq!((embeddings.count(), embeddings.columns()), (texts.len(), dimensions));
        assert_eq!(embeddings.scalar(), Some(Scalar::F32));
        assert_eq!(embeddings.values::<f32>()?.len(), texts.len() * dimensions);
        assert_eq!(embeddings.as_bytes()?.len(), texts.len() * embeddings.stride());
        assert_eq!(embeddings.values::<f64>().unwrap_err().status, Status::InvalidArgument);

        let states = encoded.states.as_ref().ok_or(fail(Status::Internal, "no states"))?;
        assert_eq!(states.offsets()?, [0, 9, 9, 32]);
        assert_eq!(states.rows()?.count(), 32);
        assert_eq!(states.item(1)?.count(), 0);
        assert_eq!(
            states.item(2)?.values::<f32>()?,
            &states.rows()?.values::<f32>()?[9 * dimensions..]
        );
        assert_eq!(
            states.item(3).err().map(|error| error.status),
            Some(Status::InvalidArgument)
        );
        let local = embeddings_of(&encoded)?;
        assert_eq!((stats.truncated, stats.tokens), (0, 32));

        let executor = executor()?;
        let blocking = model
            .on(&executor)
            .text_encode(texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
        assert_eq!(embeddings_of(&blocking)?, local);
        let texts: Vec<String> = texts.iter().map(|text| text.to_string()).collect();
        let deferred = model
            .deferred(&executor)
            .text_encode(texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
        assert_eq!(embeddings_of(&deferred.wait()?)?, local);
        Ok(())
    }

    /// Token vectors share the states' offsets and have unit length, or a model without a token
    /// head refuses them.
    #[test]
    fn token_vectors_follow_the_states_or_are_refused() -> Result<()> {
        let model = model()?;
        let token_width = model.describe()?.token_width;
        let mut arena = Arena::new()?;
        let encoded = model.text_encode(
            &["query: uform", "a"],
            Outputs::STATES | Outputs::TOKEN_VECTORS,
            Scalar::F32,
            &mut arena,
        );
        if token_width == 0 {
            assert_eq!(encoded.err().map(|error| error.status), Some(Status::Unsupported));
            return Ok(());
        }
        let encoded = encoded?;
        let states = encoded.states.ok_or(fail(Status::Internal, "no states"))?;
        let vectors = encoded
            .token_vectors
            .ok_or(fail(Status::Internal, "no token vectors"))?;
        assert_eq!(vectors.offsets()?, states.offsets()?);
        let rows = vectors.rows()?;
        assert_eq!(rows.columns(), token_width);
        for vector in rows.values::<f32>()?.chunks(token_width) {
            let squares: f32 = vector.iter().map(|value| value * value).sum();
            assert!((0.98..1.02).contains(&squares), "every token vector has unit length");
        }
        Ok(())
    }

    /// Each item's probabilities pair with its choices and sum to one, or a model without the byte
    /// decoder refuses them.
    #[test]
    fn choices_sum_to_one_or_are_refused() -> Result<()> {
        let model = model()?;
        let groups = model.describe()?.choice_temperature_groups;
        let mut arena = Arena::new()?;
        let texts = ["keep this candidate?", "expand which neighbour?"];
        let choices = ["yn", "ABCD"];
        let probabilities = model.text_choose(&texts, &choices, TextChooseConfig::default(), &mut arena);
        if groups == 0 {
            assert_eq!(probabilities.err().map(|error| error.status), Some(Status::Unsupported));
            return Ok(());
        }
        let probabilities = probabilities?;
        assert_eq!(probabilities.offsets()?, [0, 2, 6]);
        for item in 0..texts.len() {
            let sum: f32 = probabilities.item::<f32>(item)?.iter().sum();
            assert!((0.999..1.001).contains(&sum), "each item's probabilities sum to one");
        }
        let past = TextChooseConfig {
            temperature_groups: &[0, groups as u32],
            ..TextChooseConfig::default()
        };
        let refused = model.text_choose(&texts, &choices, past, &mut arena);
        assert_eq!(refused.err().map(|error| error.status), Some(Status::InvalidArgument));
        Ok(())
    }

    #[test]
    fn states_alone_skip_the_embeddings() -> Result<()> {
        let model = model()?;
        let mut arena = Arena::new()?;
        let encoded = model.text_encode(&["uform"], Outputs::STATES, Scalar::BF16, &mut arena)?;
        assert!(encoded.embeddings.is_none());
        let states = encoded.states.ok_or(fail(Status::Internal, "no states"))?;
        assert_eq!(states.rows()?.scalar(), Some(Scalar::BF16));
        assert_eq!(states.offsets()?, [0, 5]);
        Ok(())
    }

    /// A borrowed tape reaches the library as it is, and lands what the same items gathered land.
    #[test]
    fn borrowed_tapes_match_gathered_items() -> Result<()> {
        let model = model()?;
        let (mut gathered, mut borrowed) = (Arena::new()?, Arena::new()?);
        let image = png();
        let items = vec![image.clone(), image.clone()];
        let joined = items.concat();
        let tape = (joined.as_slice(), [0usize, image.len(), 2 * image.len()]);
        let from_items = model.image_encode(&items, Outputs::EMBEDDINGS, Scalar::F32, &mut gathered)?;
        let from_tape = model.image_encode(tape, Outputs::EMBEDDINGS, Scalar::F32, &mut borrowed)?;
        assert_eq!(embeddings_of(&from_items)?, embeddings_of(&from_tape)?);

        let past_the_end = (b"first".as_slice(), [0usize, 6]);
        let refusal = model.video_encode(past_the_end, Outputs::EMBEDDINGS, Scalar::F32, &mut borrowed);
        assert_eq!(refusal.err().map(|error| error.status), Some(Status::ShapeMismatch));
        Ok(())
    }

    #[test]
    fn texts_past_the_context_are_truncated() -> Result<()> {
        let model = model()?;
        let context = model.describe()?.context;
        let mut stats = TextEncodeStats::default();
        let config = TextEncodeConfig {
            stats: Some(&mut stats),
            ..TextEncodeConfig::from(Scalar::F32)
        };
        let long = "u".repeat(context + 7);
        let mut arena = Arena::new()?;
        let encoded = model.text_encode(&[long.as_str(), "short"], Outputs::STATES, config, &mut arena)?;
        let offsets = encoded
            .states
            .ok_or(fail(Status::Internal, "no states"))?
            .offsets()?
            .to_vec();
        assert_eq!(offsets, [0, context, context + 5]);
        assert_eq!((stats.truncated, stats.tokens), (1, context + 5));
        Ok(())
    }

    #[test]
    fn pixels_need_one_height_and_width_per_item() -> Result<()> {
        let model = model()?;
        let mut arena = Arena::new()?;
        let image = vec![0u8; 16 * 8 * 3];
        let encoded = model.image_encode_pixels(
            [image.as_slice()],
            [16u32],
            [8u32],
            Outputs::EMBEDDINGS,
            Scalar::F16,
            &mut arena,
        )?;
        assert_eq!(encoded.embeddings.map(|rows| rows.count()), Some(1));

        let unpaired = model.image_encode_pixels(
            [&image],
            [16u32, 16],
            [8u32],
            Outputs::EMBEDDINGS,
            Scalar::F16,
            &mut arena,
        );
        assert_eq!(unpaired.err().map(|error| error.status), Some(Status::ShapeMismatch));
        let misshapen =
            model.image_encode_pixels([&image], [16u32], [9u32], Outputs::EMBEDDINGS, Scalar::F16, &mut arena);
        assert_eq!(misshapen.err().map(|error| error.status), Some(Status::ShapeMismatch));
        Ok(())
    }

    #[test]
    fn samples_and_recordings_encode_through_their_own_verbs() -> Result<()> {
        let model = model()?;
        let mut arena = Arena::new()?;
        let samples = vec![vec![0.25f32; 1000], vec![-0.5f32; 10]];
        let mut stats = AudioEncodeStats::default();
        let config = AudioEncodeSamplesConfig {
            stats: Some(&mut stats),
            ..AudioEncodeSamplesConfig::from(Scalar::F32)
        };
        let encoded = model.audio_encode_samples(&samples, 16_000, Outputs::EMBEDDINGS, config, &mut arena)?;
        assert_eq!(encoded.embeddings.map(|rows| rows.count()), Some(2));
        assert!(stats.tokens >= 2, "every recording spans a frame");

        let recording = wav();
        let recordings = [recording.as_slice()];
        let encoded = model.audio_encode(recordings, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
        assert_eq!(encoded.embeddings.map(|rows| rows.count()), Some(1));
        Ok(())
    }

    #[test]
    fn a_stopping_handler_cancels_the_call_and_a_panicking_one_resurfaces() -> Result<()> {
        let model = model()?;
        let mut arena = Arena::new()?;
        let mut stop = |_, _| ControlFlow::Break(());
        let config = TextEncodeConfig {
            progress: Some(&mut stop),
            ..TextEncodeConfig::from(Scalar::F32)
        };
        let stopped = model.text_encode(&["a", "b"], Outputs::EMBEDDINGS, config, &mut arena);
        assert_eq!(stopped.err().map(|error| error.status), Some(Status::Cancelled));

        let raised = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
            let mut explode = |_, _| -> ControlFlow<()> { panic!("the handler failed") };
            let config = TextEncodeConfig {
                progress: Some(&mut explode),
                ..TextEncodeConfig::from(Scalar::F32)
            };
            let _ = model.text_encode(&["a"], Outputs::EMBEDDINGS, config, &mut arena);
        }));
        assert!(raised.is_err(), "the handler's panic resurfaces once the call is final");
        Ok(())
    }

    /// An executor of the stand-in holds deferred calls until a drain, so a call is observably not
    /// final until then, and awaiting it stays pending.
    #[test]
    fn an_awaited_call_is_pending_until_final() -> Result<()> {
        let model = model()?;
        let executor = executor()?;
        let mut arena = Arena::new_in(&executor)?;
        let call =
            model
                .deferred(&executor)
                .text_encode(vec!["uform"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
        assert_eq!(call.status(), Err(Status::Deferred));
        let mut awaited = pin!(call.into_future());
        let mut context = Context::from_waker(Waker::noop());
        assert!(awaited.as_mut().poll(&mut context).is_pending());
        executor.drain(Mood::Parked)?;
        let Poll::Ready(encoded) = awaited.as_mut().poll(&mut context) else {
            panic!("a drained call must be final");
        };
        let blocking =
            embeddings_of(&model.text_encode(&["uform"], Outputs::EMBEDDINGS, Scalar::F32, &mut Arena::new()?)?)?;
        assert_eq!(embeddings_of(&encoded?)?, blocking);
        assert!(matches!(awaited.as_mut().poll(&mut context), Poll::Ready(Err(_))));
        Ok(())
    }

    /// Dropping an unfinished call waits for it, and a leaked one leaves its arena busy rather than
    /// unsafe: a call naming it is refused as contended.
    #[test]
    fn unfinished_calls_drop_by_waiting_and_leaked_ones_keep_their_arena() -> Result<()> {
        let model = model()?;
        let executor = executor()?;
        let mut arena = Arena::new_in(&executor)?;
        let call = model
            .deferred(&executor)
            .text_encode(vec!["a"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
        drop(call);
        assert!(model
            .text_encode(&["b"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena)
            .is_ok());

        let leaked = model
            .deferred(&executor)
            .text_encode(vec!["c"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
        core::mem::forget(leaked);
        let busy = model.text_encode(&["d"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena);
        assert!(matches!(
            busy,
            Err(Error {
                status: Status::Contended,
                ..
            })
        ));
        executor.drain(Mood::Parked)?;
        Ok(())
    }

    #[test]
    fn a_full_deferred_table_answers_contended() -> Result<()> {
        let model = model()?;
        let executor = Executor::for_ram(&[], None, 1, 1)?;
        let (mut first_arena, mut second_arena) = (Arena::new_in(&executor)?, Arena::new_in(&executor)?);
        let receiver = model.deferred(&executor);
        let first = receiver.text_encode(vec!["a"], Outputs::EMBEDDINGS, Scalar::F32, &mut first_arena)?;
        let second = receiver.text_encode(vec!["b"], Outputs::EMBEDDINGS, Scalar::F32, &mut second_arena)?;
        assert_eq!(second.status(), Err(Status::Contended));
        assert!(first.wait().is_ok());
        Ok(())
    }
}

// endregion: Tests
