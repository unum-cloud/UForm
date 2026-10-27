//! Training steps: multimodal batches in, masked reconstruction, pairwise contrastive,
//! late-interaction and calibrated-choice losses joined into one optimizer step, checkpoints out.
//!
//! A [`Mesh`] places this process among the processes of a run, made once by [`Mesh::bootstrap`]
//! and borrowed by every [`Trainer`]. A trainer owns the tower replicas, gradients, optimizer
//! moments, heads and temperatures on its executor's devices; training is GPU-only, so its one
//! constructor is [`Trainer::init_vram`]. Its verbs come from [`TrainVerbs`], which run on the
//! executor the trainer was made on: `trainer.on(&executor)` blocks, and `trainer.deferred(&executor)`
//! answers a [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
//!
//! File: rust/train.rs
//! Author: Ash Vardanian
//! Date: October 18, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_void};
use core::fmt;
use core::marker::PhantomData;
use core::ptr;
use core::str::FromStr;
use std::path::Path;

use crate::types::{
    check, fail, lower_progress, path_view, read_spelling, spelling, uform_arena_t, uform_bytes_view_t,
    uform_completion_blocking_k, uform_completion_t, uform_executor_t, uform_gpu_class_t, uform_outcome_t,
    uform_progress_t, uform_rows_view_t, uform_scalar_t, uform_tape_view_t, Accepts, Arena, Callbacks, Calling,
    Completion, Deferring, Error, Executor, Exported, GPUClass, Gathered, Handles, On, ProgressCallback, Result, Rows,
    Scalar, Status, TapeView,
};

// region: Raw Declarations

pub type uform_modality_t = c_int;
pub type uform_objective_t = c_int;
pub type uform_negatives_t = c_int;
pub type uform_multimem_tier_t = c_int;
pub type uform_pair_loss_form_t = c_int;
pub type uform_late_interaction_side_t = c_int;

pub type uform_batch_shape_t = BatchShape;
pub type uform_masking_t = Masking;
pub type uform_late_interaction_t = LateInteraction;
pub type uform_choices_t = Choices;
pub type uform_schedule_t = Schedule;
pub type uform_trainer_step_stats_t = TrainerStepStats;

#[repr(C)]
pub struct uform_contrastive_t {
    pub matryoshka_dimensions: *const usize,
    pub matryoshka_dimensions_count: usize,
    pub temperature_groups: usize,
    pub inverse_temperature: f32,
    pub bias: f32,
    pub negatives: uform_negatives_t,
    pub ambient_repulsion_weight: f32,
    pub max_edges_per_device: usize,
    pub pair_loss: uform_pair_loss_form_t,
}

#[repr(C)]
pub struct uform_mesh_options_t {
    pub host_domain_index: usize,
    pub host_domain_count: usize,
    pub gpus_per_multimem_domain: usize,
    pub directory: uform_bytes_view_t,
    pub job: uform_bytes_view_t,
    pub timeout_seconds: f64,
}

pub type uform_mesh_t = *mut c_void;

#[repr(C)]
pub struct uform_mesh_bootstrap_t {
    pub mesh: *mut uform_mesh_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub options: uform_mesh_options_t,
}

#[repr(C)]
#[derive(Default)]
pub struct uform_mesh_description_t {
    pub gpus_per_host_domain: usize,
    pub gpus_per_multimem_domain: usize,
    pub gpus_per_colocated_domain: usize,
    pub host_domain_index: usize,
    pub first_gpu_index: usize,
    pub multimem_tier: uform_multimem_tier_t,
    pub gpu_class: uform_gpu_class_t,
}

#[repr(C)]
pub struct uform_mesh_describe_t {
    pub mesh: uform_mesh_t,
    pub outcome: *mut uform_outcome_t,
    pub description: *mut uform_mesh_description_t,
}

pub type uform_trainer_t = *mut c_void;

#[repr(C)]
pub struct uform_trainer_init_vram_t {
    pub trainer: *mut uform_trainer_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub resume: uform_bytes_view_t,
    pub depth: usize,
    pub hidden: usize,
    pub batch: uform_batch_shape_t,
    pub objective: uform_objective_t,
    pub masking: uform_masking_t,
    pub contrastive: uform_contrastive_t,
    pub late_interaction: uform_late_interaction_t,
    pub choices: uform_choices_t,
    pub schedule: uform_schedule_t,
    pub mesh: uform_mesh_t,
    pub seed: u64,
    pub progress: uform_progress_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_edges_view_t {
    pub offsets: *const usize,
    pub ids: *const u32,
    pub weights: *const f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uform_relations_view_t {
    pub attract: uform_edges_view_t,
    pub repel: uform_edges_view_t,
    pub ignore: uform_edges_view_t,
}

#[repr(C)]
pub struct uform_trainer_step_t {
    pub trainer: uform_trainer_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub count: usize,
    pub modalities: *const uform_modality_t,
    pub contents: uform_tape_view_t,
    pub bags: *const u32,
    pub temperature_groups: *const u32,
    pub sides: *const uform_late_interaction_side_t,
    pub choices: uform_tape_view_t,
    pub choice_targets: uform_tape_view_t,
    pub relations: uform_relations_view_t,
    pub embeddings: *mut uform_rows_view_t,
    pub arena: uform_arena_t,
    pub outputs_scalar: uform_scalar_t,
    pub stats: *mut uform_trainer_step_stats_t,
}

#[repr(C)]
pub struct uform_trainer_save_t {
    pub trainer: uform_trainer_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub path: uform_bytes_view_t,
    pub progress: uform_progress_t,
}

extern "C" {
    pub fn uform_modality_name(modality: uform_modality_t) -> *const c_char;
    pub fn uform_modality_from_name(name: uform_bytes_view_t) -> uform_modality_t;
    pub fn uform_objective_name(objective: uform_objective_t) -> *const c_char;
    pub fn uform_objective_from_name(name: uform_bytes_view_t) -> uform_objective_t;
    pub fn uform_negatives_name(negatives: uform_negatives_t) -> *const c_char;
    pub fn uform_negatives_from_name(name: uform_bytes_view_t) -> uform_negatives_t;
    pub fn uform_multimem_tier_name(tier: uform_multimem_tier_t) -> *const c_char;
    pub fn uform_multimem_tier_from_name(name: uform_bytes_view_t) -> uform_multimem_tier_t;

    pub fn uform_mesh_bootstrap(arguments: *mut uform_mesh_bootstrap_t);
    pub fn uform_mesh_describe(arguments: *mut uform_mesh_describe_t);
    pub fn uform_mesh_free(mesh: uform_mesh_t);

    pub fn uform_trainer_init_vram(arguments: *mut uform_trainer_init_vram_t);
    pub fn uform_trainer_step(arguments: *mut uform_trainer_step_t);
    pub fn uform_trainer_save(arguments: *mut uform_trainer_save_t);
    pub fn uform_trainer_free(trainer: uform_trainer_t);
}

// endregion: Raw Declarations

// region: Enums

/// What the bytes of one training item hold.
///
/// Mirrors `uform_modality_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Modality {
    /// UTF-8 text, a byte per token.
    Text = 0,
    /// A JPEG, PNG or WebP file.
    Image = 1,
    /// An MP4 or WebM file.
    Video = 2,
    /// A WAV, FLAC, MP3 or Opus file.
    Audio = 3,
}

impl Modality {
    fn from_raw(value: uform_modality_t) -> Option<Self> {
        match value {
            0 => Some(Modality::Text),
            1 => Some(Modality::Image),
            2 => Some(Modality::Video),
            3 => Some(Modality::Audio),
            _ => None,
        }
    }
}

impl fmt::Display for Modality {
    /// Calls `uform_modality_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_modality_name, *self as c_int))
    }
}

impl FromStr for Modality {
    type Err = Error;

    /// Calls `uform_modality_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, uform_modality_from_name, uform_modality_name, Modality::from_raw)
    }
}

/// Which losses a step relaxes, one enumerator per combination the engine trains.
///
/// Mirrors `uform_objective_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Objective {
    /// Masked reconstruction alone, as the Base pretrains.
    #[default]
    Masked = 0,
    /// The pairwise contrastive loss alone, as the Embedder trains.
    Contrastive = 1,
    /// Both, joined at the final-norm states.
    Joint = 2,
}

impl Objective {
    fn from_raw(value: uform_objective_t) -> Option<Self> {
        match value {
            0 => Some(Objective::Masked),
            1 => Some(Objective::Contrastive),
            2 => Some(Objective::Joint),
            _ => None,
        }
    }
}

impl fmt::Display for Objective {
    /// Calls `uform_objective_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_objective_name, *self as c_int))
    }
}

impl FromStr for Objective {
    type Err = Error;

    /// Calls `uform_objective_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(
            text,
            uform_objective_from_name,
            uform_objective_name,
            Objective::from_raw,
        )
    }
}

/// Which pairs the contrastive loss pushes apart.
///
/// Mirrors `uform_negatives_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Negatives {
    /// Repel edges, and every pair no edge names, at the ambient repulsion weight.
    #[default]
    Ambient = 0,
    /// Repel edges alone.
    Named = 1,
}

impl Negatives {
    fn from_raw(value: uform_negatives_t) -> Option<Self> {
        match value {
            0 => Some(Negatives::Ambient),
            1 => Some(Negatives::Named),
            _ => None,
        }
    }
}

impl fmt::Display for Negatives {
    /// Calls `uform_negatives_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_negatives_name, *self as c_int))
    }
}

impl FromStr for Negatives {
    type Err = Error;

    /// Calls `uform_negatives_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(
            text,
            uform_negatives_from_name,
            uform_negatives_name,
            Negatives::from_raw,
        )
    }
}

/// How a mesh's NVLink domains reduce.
///
/// Mirrors `uform_multimem_tier_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MultimemTier {
    /// The NVLink switch reduces through multicast memory.
    SwitchReduced = 0,
    /// Every member reads its peers in a fixed order, where multicast is absent; bit-identical results.
    PeerRead = 1,
}

impl MultimemTier {
    fn from_raw(value: uform_multimem_tier_t) -> Option<Self> {
        match value {
            0 => Some(MultimemTier::SwitchReduced),
            1 => Some(MultimemTier::PeerRead),
            _ => None,
        }
    }
}

impl fmt::Display for MultimemTier {
    /// Calls `uform_multimem_tier_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(uform_multimem_tier_name, *self as c_int))
    }
}

impl FromStr for MultimemTier {
    type Err = Error;

    /// Calls `uform_multimem_tier_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(
            text,
            uform_multimem_tier_from_name,
            uform_multimem_tier_name,
            MultimemTier::from_raw,
        )
    }
}

/// How a contrastive term turns a row of scores into forces.
///
/// Mirrors `uform_pair_loss_form_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum PairLossForm {
    /// The term's own form: sigmoid for pooled vertices, ranked for token vectors.
    #[default]
    Derived = 0,
    /// The sigmoid cross-entropy of every cell, its attract and repel halves normalized apart.
    Balanced = 1,
    /// The sigmoid cross-entropy of every positive's score over every negative's, where the bias
    /// cancels.
    Ranked = 2,
    /// SigLIP's sigmoid cross-entropy of every cell, each repel weighed like an attract.
    Sigmoid = 3,
}

/// Which side of a late-interaction score an item takes: a query's tokens score a document's, and a
/// symmetric pair plays both.
///
/// Mirrors `uform_late_interaction_side_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum LateInteractionSide {
    /// The item takes no part in late interaction.
    #[default]
    None = 0,
    /// The item's tokens score the documents it attracts and repels.
    Query = 1,
    /// The item's tokens are scored by the queries naming it.
    Document = 2,
    /// The item plays both sides.
    QueryAndDocument = 3,
}

// endregion: Enums

// region: Configs

/// The image slice of a batch envelope; a zero `patch_side` leaves images out of the run.
///
/// Mirrors `uform_image_shape_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct ImageShape {
    /// The square patch side images tile into, in pixels.
    pub patch_side: usize,
    /// Channels every pixel carries; zero derives four, RGBA end to end.
    pub channels: usize,
}

/// The video slice of a batch envelope; a zero `patch_side` leaves video out of the run.
///
/// Mirrors `uform_video_shape_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct VideoShape {
    /// The square patch side one tube tiles into, in pixels.
    pub patch_side: usize,
    /// Frames one tube folds ahead of its patch; zero derives two.
    pub tube_frames: usize,
    /// Source frames between sampled ones; zero derives one, every frame.
    pub frame_stride: usize,
    /// Channels every pixel carries; zero derives three.
    pub channels: usize,
}

/// The audio slice of a batch envelope; a zero `chunk_samples` leaves audio out of the run.
///
/// Mirrors `uform_audio_shape_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct AudioShape {
    /// Raw samples per frame, the fan-in of the learned frontend.
    pub chunk_samples: usize,
    /// The mono rate every recording is resampled to; zero derives 16000.
    pub sample_rate: usize,
}

/// What one device's step carries at most, whatever the modality mix: capacity fixed at creation.
/// Text always takes part; a media modality takes part when its slice is live.
///
/// Mirrors `uform_batch_shape_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct BatchShape {
    /// Rows one device's step packs.
    pub max_rows_per_device: usize,
    /// Items one device's step carries across all its rows, which is also its pooling axis.
    pub max_segments_per_device: usize,
    /// Positions one row spans, the sequence length.
    pub max_columns_per_row: usize,
    /// Bags one device's step carries; zero keeps every item its own routing unit.
    pub max_bags_per_device: usize,
    /// Legal bytes one item offers to choose from at most; zero makes no choices.
    pub max_choices_per_item: usize,
    /// The image geometry.
    pub image: ImageShape,
    /// The video geometry.
    pub video: VideoShape,
    /// The audio geometry.
    pub audio: AudioShape,
}

/// How the masked objective corrupts a batch and weighs its heads; every zero derives the default
/// the C header names.
///
/// Mirrors `uform_masking_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct Masking {
    /// Fraction of text positions covered by spans; 0.15.
    pub token_mask_probability: f32,
    /// Mean geometric span length in bytes; 16.
    pub token_mean_span: f32,
    /// Of covered positions, the share shown as MASK; 0.8.
    pub replace_fraction: f32,
    /// Of covered positions, the share replaced by a random byte, the rest kept clean; 0.1.
    pub random_fraction: f32,
    /// Fraction of image patches masked; 0.75.
    pub image_mask_ratio: f32,
    /// Fraction of video tubes masked, tubed across time; 0.75.
    pub video_mask_ratio: f32,
    /// Fraction of audio frames masked; 0.75.
    pub audio_mask_ratio: f32,
    /// The weight of the text head; 1.
    pub text_weight: f32,
    /// The weight of the image head; 1.
    pub image_weight: f32,
    /// The weight of the video head; 1.
    pub video_weight: f32,
    /// The weight of the audio head; 1.
    pub audio_weight: f32,
}

/// How the contrastive loss scores the pooled embeddings; every zero derives the default the C
/// header names.
///
/// Mirrors `uform_contrastive_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Contrastive<'config> {
    /// Matryoshka prefix widths, strictly ascending, each a multiple of 8, at most 8 of them, every
    /// one scored; empty scores the full width alone. Copied, not borrowed.
    pub matryoshka_dimensions: &'config [usize],
    /// Learnable inverse temperature and bias pairs, one per group a step's items select; one.
    pub temperature_groups: usize,
    /// The starting inverse temperature of every group; 10 under the sigmoid form, SigLIP's,
    /// else 1.
    pub inverse_temperature: f32,
    /// The starting logit bias of every group; −10 under the sigmoid form, SigLIP's, else zero.
    pub bias: f32,
    /// Which pairs repel.
    pub negatives: Negatives,
    /// The weight of every pair no edge names, under [`Negatives::Ambient`]; 1.
    pub ambient_repulsion_weight: f32,
    /// Edges one device's step names per edge set at most; `max_segments_per_device`.
    pub max_edges_per_device: usize,
    /// The pair loss over each anchor's row; sigmoid.
    pub pair_loss: PairLossForm,
}

impl Contrastive<'_> {
    fn lower(&self) -> uform_contrastive_t {
        uform_contrastive_t {
            matryoshka_dimensions: match self.matryoshka_dimensions.is_empty() {
                true => ptr::null(),
                false => self.matryoshka_dimensions.as_ptr(),
            },
            matryoshka_dimensions_count: self.matryoshka_dimensions.len(),
            temperature_groups: self.temperature_groups,
            inverse_temperature: self.inverse_temperature,
            bias: self.bias,
            negatives: self.negatives as uform_negatives_t,
            ambient_repulsion_weight: self.ambient_repulsion_weight,
            max_edges_per_device: self.max_edges_per_device,
            pair_loss: self.pair_loss as uform_pair_loss_form_t,
        }
    }
}

/// How the late-interaction loss scores the per-token vectors of query and document items; every
/// zero derives the default the C header names.
///
/// Mirrors `uform_late_interaction_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct LateInteraction {
    /// The loss weight; zero trains no token head.
    pub weight: f32,
    /// The token vectors' width, a multiple of 8; 128.
    pub width: usize,
    /// The starting inverse temperature of every contrastive group; 1.
    pub inverse_temperature: f32,
    /// The starting logit bias of every contrastive group; zero is zero.
    pub bias: f32,
    /// Tokens one query item contributes at most, its first past the prefix; 64.
    pub max_query_span: usize,
    /// Query token rows one device's step holds at most; `max_query_span` of every item.
    pub max_query_tokens_per_device: usize,
    /// Document items one device's step holds at most; `max_segments_per_device`.
    pub max_documents_per_device: usize,
    /// Document token rows one device's step holds at most; the batch's tokens per device.
    pub max_document_tokens_per_device: usize,
    /// The pair loss over each query's row of hard MaxSim scores; ranked.
    pub pair_loss: PairLossForm,
}

/// SearchJev's calibrated choices over the byte decoder: the masked head's byte logits read at a
/// mask byte appended to a state, restricted to the item's legal bytes. Each choice item scores
/// `−Σ t·log(p) + brier_weight·Σ (p − t)²`, with `p` that restricted softmax and `t` its targets.
///
/// Mirrors `uform_choices_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct Choices {
    /// The loss weight; zero trains no choices.
    pub weight: f32,
    /// The Brier term's weight beside the soft cross-entropy; 0.5.
    pub brier_weight: f32,
}

/// The learning-rate schedule: a linear warmup to the peak, then a cosine decay to the floor.
///
/// Mirrors `uform_schedule_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct Schedule {
    /// The peak learning rate; zero derives 5e-4.
    pub learning_rate: f32,
    /// The floor the cosine settles at; zero derives 1e-5.
    pub min_learning_rate: f32,
    /// Steps from zero to the peak; zero derives a tenth of `steps`, at most 400.
    pub warmup_steps: usize,
    /// Steps the run takes across resumes, which the cosine spans; zero is refused.
    pub steps: usize,
}

/// Where this process sits among the processes training one model, and how they meet.
///
/// Mirrors `uform_mesh_options_t`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshOptions<'options> {
    /// This process's index among `host_domain_count`.
    pub host_domain_index: usize,
    /// Processes in the run, one per host; zero derives one, a run with no peers.
    pub host_domain_count: usize,
    /// GPUs one NVLink multicast domain spans, a fabric property no local probe sees; zero derives the host.
    pub gpus_per_multimem_domain: usize,
    /// The directory every host sees, where processes exchange their facts; required with peers.
    pub directory: &'options Path,
    /// The run's unique name inside `directory`; required with peers.
    pub job: &'options str,
    /// Seconds any wait for a peer lasts before the call fails; zero derives 120.
    pub timeout_seconds: f64,
}

impl Default for MeshOptions<'_> {
    fn default() -> Self {
        MeshOptions {
            host_domain_index: 0,
            host_domain_count: 0,
            gpus_per_multimem_domain: 0,
            directory: Path::new(""),
            job: "",
            timeout_seconds: 0.0,
        }
    }
}

/// What a mesh reports about itself.
///
/// Mirrors `uform_mesh_description_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct MeshDescription {
    /// GPUs one process drives.
    pub gpus_per_host_domain: usize,
    /// GPUs one NVLink domain spans.
    pub gpus_per_multimem_domain: usize,
    /// GPUs in the whole run.
    pub gpus_per_colocated_domain: usize,
    /// This process's index among the run's processes.
    pub host_domain_index: usize,
    /// The run-wide index of this process's first GPU, where its share of a sharded dataset begins.
    pub first_gpu_index: usize,
    /// How its NVLink domains reduce.
    pub multimem_tier: MultimemTier,
    /// The class every GPU of the run dispatches as: the executor's pin, or the devices' own.
    pub gpu_class: Option<GPUClass>,
}

/// The geometry, objective and starting point of a run; see [`Trainer::init_vram`].
///
/// Mirrors `uform_trainer_init_vram_t`.
#[derive(Default)]
pub struct TrainerInitVramConfig<'config, 'mesh> {
    /// The checkpoint to continue from, at its saved step; `None` starts from `seed`.
    pub resume: Option<&'config Path>,
    /// Transformer blocks stacked.
    pub depth: usize,
    /// Embedding width, a multiple of 128.
    pub hidden: usize,
    /// What one device's step carries at most.
    pub batch: BatchShape,
    /// Which losses a step relaxes.
    pub objective: Objective,
    /// The masked objective's corruption and weights, read unless the objective is contrastive alone.
    pub masking: Masking,
    /// The contrastive loss's scoring, read unless the objective is masked alone.
    pub contrastive: Contrastive<'config>,
    /// The late-interaction loss beside the pooled one, read when its weight is positive.
    pub late_interaction: LateInteraction,
    /// The calibrated choices beside every other term, read when its weight is positive, which
    /// needs the batch's `max_choices_per_item`.
    pub choices: Choices,
    /// The learning-rate schedule.
    pub schedule: Schedule,
    /// The mesh made over the same executor, borrowed until the trainer drops; `None` runs this
    /// process alone.
    pub mesh: Option<&'mesh Mesh>,
    /// The base of every keyed draw: initialization, corruption and the shuffle.
    pub seed: u64,
    /// Reports the tensors placed on a resume and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

/// One edge set over a step's items, as compressed sparse rows; empty offsets name no edges.
///
/// Mirrors `uform_edges_view_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Edges<'config> {
    /// The step's count plus one non-decreasing offsets into `ids`.
    pub offsets: &'config [usize],
    /// Item `i` links to items `ids[offsets[i]..offsets[i + 1]]`, indices into the step's items,
    /// ascending within each run.
    pub ids: &'config [u32],
    /// The weight of each edge, paired with `ids`; empty weighs every edge one.
    pub weights: &'config [f32],
}

impl Edges<'_> {
    /// Lends the edge set to a step over `count` items, refusing one the library would read past.
    fn lower(&self, count: usize) -> Result<uform_edges_view_t> {
        if self.offsets.is_empty() {
            return match self.ids.is_empty() && self.weights.is_empty() {
                true => Ok(uform_edges_view_t {
                    offsets: ptr::null(),
                    ids: ptr::null(),
                    weights: ptr::null(),
                }),
                false => Err(fail(Status::InvalidArgument, "edges without offsets name no items")),
            };
        }
        if self.offsets.len() != count + 1 {
            return Err(fail(
                Status::ShapeMismatch,
                "an edge set holds one more offset than the step's items",
            ));
        }
        if self.offsets.windows(2).any(|pair| pair[1] < pair[0]) || self.offsets[count] > self.ids.len() {
            return Err(fail(Status::Malformed, "the edge offsets decrease or run past the ids"));
        }
        if self.ids.iter().any(|&id| id as usize >= count) {
            return Err(fail(Status::InvalidArgument, "an edge names an item outside the step"));
        }
        if !self.weights.is_empty() && self.weights.len() != self.ids.len() {
            return Err(fail(Status::ShapeMismatch, "pass no weights or one per edge"));
        }
        Ok(uform_edges_view_t {
            offsets: self.offsets.as_ptr(),
            ids: self.ids.as_ptr(),
            weights: match self.weights.is_empty() {
                true => ptr::null(),
                false => self.weights.as_ptr(),
            },
        })
    }
}

/// The contrastive targets of one step, resolved per pair in this order: a pair `ignore` names is
/// skipped, one `attract` names pulls together, one `repel` names pushes apart, and every other
/// pair follows the run's [`Negatives`].
///
/// Mirrors `uform_relations_view_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Relations<'config> {
    /// Pairs pulled together.
    pub attract: Edges<'config>,
    /// Pairs pushed apart.
    pub repel: Edges<'config>,
    /// Pairs skipped, like known false negatives; their weights are not read.
    pub ignore: Edges<'config>,
}

/// What a step reports, every loss its world mean, overwritten by the call.
///
/// Mirrors `uform_trainer_step_stats_t`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct TrainerStepStats {
    /// The optimizer step the call applied, counted from one across resumes.
    pub step: usize,
    /// The learning rate it applied.
    pub learning_rate: f32,
    /// The total loss the step minimized.
    pub loss: f32,
    /// The contrastive term; zero when the objective is masked alone.
    pub contrastive_loss: f32,
    /// The text reconstruction term, before its weight; zero when absent.
    pub masked_text_loss: f32,
    /// The image reconstruction term, before its weight; zero when absent.
    pub masked_image_loss: f32,
    /// The video reconstruction term, before its weight; zero when absent.
    pub masked_video_loss: f32,
    /// The audio reconstruction term, before its weight; zero when absent.
    pub masked_audio_loss: f32,
    /// The late-interaction term, weighted; zero when absent.
    pub late_interaction_loss: f32,
    /// The choices term over the items that made a choice, before its weight; zero when absent.
    pub choices_loss: f32,
    /// Items cut at a row's width.
    pub truncated: usize,
    /// Segments dropped whole because the batch had no room for them, since the last step that
    /// read its stats.
    pub dropped: usize,
}

/// Lends a step over `count` items its choices and their targets, both `(data, offsets)` tapes or
/// neither, refusing ones the library would read past or pair unevenly.
fn lower_choices(
    choices: (&[u8], &[usize]),
    targets: (&[f32], &[usize]),
    count: usize,
) -> Result<(uform_tape_view_t, uform_tape_view_t)> {
    if choices.1.is_empty() && targets.1.is_empty() {
        return match choices.0.is_empty() && targets.0.is_empty() {
            true => Ok((uform_tape_view_t::default(), uform_tape_view_t::default())),
            false => Err(fail(Status::InvalidArgument, "choices without offsets name no items")),
        };
    }
    let lent = choices.tape(&mut Gathered::default())?;
    let aimed = targets.tape(&mut Gathered::default())?;
    if lent.count != count || aimed.count != count {
        return Err(fail(
            Status::ShapeMismatch,
            "choices and their targets hold one more offset than the step's items",
        ));
    }
    let mut entries = choices.1.windows(2).zip(targets.1.windows(2));
    if entries.any(|(offered, aimed)| offered[1] - offered[0] != aimed[1] - aimed[0]) {
        return Err(fail(Status::ShapeMismatch, "every choice needs one target"));
    }
    Ok((lent.tape, aimed.tape))
}

/// The routing, targets and reports of [`TrainVerbs::step`].
///
/// Mirrors `uform_trainer_step_t`.
pub struct TrainerStepConfig<'config, 'arena> {
    /// The bag of each item, items sharing a bag landing on one device; empty keeps every item its
    /// own bag.
    pub bags: &'config [u32],
    /// The temperature group of each item, each below the run's `temperature_groups`; empty puts
    /// every item in group zero.
    pub temperature_groups: &'config [u32],
    /// The late-interaction side of each item; empty makes every item
    /// [`LateInteractionSide::None`].
    pub sides: &'config [LateInteractionSide],
    /// Each item's legal choices, one byte each, as a `(data, offsets)` tape like the one
    /// [`text_choose`](crate::EncodeVerbs::text_choose) takes; empty offsets make no choices. Read when
    /// the run trains them: a text item offering any is the state its answer follows, and the call
    /// appends the mask byte, cutting a state longer than a row at its head.
    pub choices: (&'config [u8], &'config [usize]),
    /// The target probability of each choice, a `(data, offsets)` tape whose entries are as long as
    /// those of `choices` and sum to one, as `text_choose` answers its probabilities.
    pub choice_targets: (&'config [f32], &'config [usize]),
    /// The contrastive targets, read unless the objective is masked alone.
    pub relations: Relations<'config>,
    /// The arena the pooled embeddings land in, as the contrastive loss scored them before the
    /// update; `None` skips the copy, and the step answers no rows.
    pub arena: Option<&'arena mut Arena>,
    /// The scalar type the embeddings land as, read only with an arena.
    pub outputs_scalar: Scalar,
    /// Where the losses land, overwritten by the call; `None` skips their readback too.
    pub stats: Option<&'config mut TrainerStepStats>,
}

impl Default for TrainerStepConfig<'_, '_> {
    fn default() -> Self {
        TrainerStepConfig {
            bags: &[],
            temperature_groups: &[],
            sides: &[],
            choices: (&[], &[]),
            choice_targets: (&[], &[]),
            relations: Relations::default(),
            arena: None,
            outputs_scalar: Scalar::F32,
            stats: None,
        }
    }
}

impl<'config> From<Relations<'config>> for TrainerStepConfig<'config, '_> {
    fn from(relations: Relations<'config>) -> Self {
        TrainerStepConfig {
            relations,
            ..TrainerStepConfig::default()
        }
    }
}

/// The path a checkpoint is written to and the handler counting its tensors; see [`TrainVerbs::save`].
///
/// Mirrors `uform_trainer_save_t`.
pub struct TrainerSaveConfig<'config, Location> {
    /// Destination of the safetensors checkpoint, written whole and renamed into place.
    pub path: Location,
    /// Reports the tensors written and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<Location: AsRef<Path>> From<Location> for TrainerSaveConfig<'_, Location> {
    fn from(path: Location) -> Self {
        TrainerSaveConfig {
            path,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl Accepts<TrainerStepConfig<'static, '_>> for Deferring {}
impl<Location> Accepts<TrainerSaveConfig<'static, Location>> for Deferring where Deferring: Accepts<Location> {}

// endregion: Configs

// region: Mesh

/// One process's place among the processes of a run: the topology every process agreed on at
/// bootstrap, and the InfiniBand domain joining its NVLink domains, borrowed by every trainer.
///
/// Mirrors `uform_mesh_t`; dropping it calls `uform_mesh_free`, after every trainer borrowing it.
#[derive(Debug)]
pub struct Mesh {
    handle: uform_mesh_t,
}

// Safety: a mesh is process-wide, made once and only read by the trainers that borrow it.
unsafe impl Send for Mesh {}
unsafe impl Sync for Mesh {}

impl Drop for Mesh {
    fn drop(&mut self) { unsafe { uform_mesh_free(self.handle) } }
}

impl Mesh {
    /// Joins this process to the run's mesh through a one-time rendezvous in `options.directory`;
    /// every process of the run calls it with the same count, multicast extent, directory and job,
    /// over a VRAM executor whose devices are this process's members.
    ///
    /// Calls `uform_mesh_bootstrap`.
    pub fn bootstrap(executor: &Executor, options: &MeshOptions) -> Result<Self> {
        let mut mesh = Mesh {
            handle: ptr::null_mut(),
        };
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_mesh_bootstrap(&mut uform_mesh_bootstrap_t {
                mesh: &mut mesh.handle,
                executor: executor.handle,
                outcome: &mut outcome,
                completion: uform_completion_blocking_k,
                options: uform_mesh_options_t {
                    host_domain_index: options.host_domain_index,
                    host_domain_count: options.host_domain_count,
                    gpus_per_multimem_domain: options.gpus_per_multimem_domain,
                    directory: path_view(options.directory)?,
                    job: options.job.as_bytes().into(),
                    timeout_seconds: options.timeout_seconds,
                },
            })
        };
        check(outcome).map(|()| mesh)
    }

    /// Reports the mesh's shape and this process's place in it.
    ///
    /// Calls `uform_mesh_describe`.
    pub fn describe(&self) -> Result<MeshDescription> {
        let mut description = uform_mesh_description_t::default();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_mesh_describe(&mut uform_mesh_describe_t {
                mesh: self.handle,
                outcome: &mut outcome,
                description: &mut description,
            })
        };
        check(outcome)?;
        Ok(MeshDescription {
            gpus_per_host_domain: description.gpus_per_host_domain,
            gpus_per_multimem_domain: description.gpus_per_multimem_domain,
            gpus_per_colocated_domain: description.gpus_per_colocated_domain,
            host_domain_index: description.host_domain_index,
            first_gpu_index: description.first_gpu_index,
            multimem_tier: MultimemTier::from_raw(description.multimem_tier).ok_or(fail(
                Status::Internal,
                "the library reported a tier this binding cannot name",
            ))?,
            gpu_class: GPUClass::from_raw(description.gpu_class),
        })
    }
}

// endregion: Mesh

// region: Trainer

/// A training run: the tower replicas, gradients, optimizer moments, heads, temperatures and the
/// collectives sized by them, on its executor's devices, whose verbs come from [`TrainVerbs`].
///
/// ```no_run
/// use uform::*;
///
/// let executor = unsafe { Executor::for_vram(&[0], None, &[], 0, 0)? };
/// let trainer = Trainer::init_vram(&executor, TrainerInitVramConfig {
///     depth: 12,
///     hidden: 512,
///     batch: BatchShape { max_rows_per_device: 64, max_segments_per_device: 256, max_columns_per_row: 1024,
///                         ..BatchShape::default() },
///     schedule: Schedule { steps: 100_000, ..Schedule::default() },
///     seed: 42,
///     ..TrainerInitVramConfig::default()
/// })?;
/// let mut stats = TrainerStepStats::default();
/// let config = TrainerStepConfig { stats: Some(&mut stats), ..TrainerStepConfig::default() };
/// trainer.on(&executor).step([Modality::Text], ["a caption"], config)?;
/// trainer.on(&executor).save("checkpoint.safetensors")?;
/// # Ok::<(), uform::Error>(())
/// ```
///
/// Mirrors `uform_trainer_t`; dropping it calls `uform_trainer_free`. `'mesh` is the borrow of the
/// [`Mesh`] it was made over.
#[derive(Debug)]
pub struct Trainer<'mesh> {
    handle: uform_trainer_t,
    mesh: PhantomData<&'mesh Mesh>,
}

// Safety: a trainer goes with the executor it was made on, which may move between threads.
unsafe impl Send for Trainer<'_> {}

impl Drop for Trainer<'_> {
    fn drop(&mut self) { unsafe { uform_trainer_free(self.handle) } }
}

impl<'mesh> Trainer<'mesh> {
    /// Builds one replica per device of `executor`, the NVLink domain its consumers reserve over the
    /// mesh, and starts or resumes the run; every process of the run calls it with the same
    /// geometry, objective and seed.
    ///
    /// Calls `uform_trainer_init_vram`.
    pub fn init_vram<'config>(
        executor: &Executor,
        config: impl Into<TrainerInitVramConfig<'config, 'mesh>>,
    ) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut trainer = Trainer {
            handle: ptr::null_mut(),
            mesh: PhantomData,
        };
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_trainer_init_vram(&mut uform_trainer_init_vram_t {
                trainer: &mut trainer.handle,
                executor: executor.handle,
                outcome: &mut outcome,
                completion: uform_completion_blocking_k,
                resume: config.resume.map_or(Ok(uform_bytes_view_t::default()), path_view)?,
                depth: config.depth,
                hidden: config.hidden,
                batch: config.batch,
                objective: config.objective as uform_objective_t,
                masking: config.masking,
                contrastive: config.contrastive.lower(),
                late_interaction: config.late_interaction,
                choices: config.choices,
                schedule: config.schedule,
                mesh: config.mesh.map_or(ptr::null_mut(), |mesh| mesh.handle),
                seed: config.seed,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| trainer)
    }

    /// The receiver whose verbs block on `executor`, the one the trainer was made on.
    ///
    /// Rust-only.
    pub fn on<'trainer>(&'trainer self, executor: &'trainer Executor) -> On<'trainer, Trainer<'mesh>> {
        On {
            handle: self,
            executor,
            completion: PhantomData,
        }
    }

    /// The receiver whose verbs submit their calls to `executor`, answering a
    /// [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
    ///
    /// Rust-only.
    pub fn deferred<'trainer>(&'trainer self, executor: &'trainer Executor) -> On<'trainer, Trainer<'mesh>, Deferring> {
        On {
            handle: self,
            executor,
            completion: PhantomData,
        }
    }
}

/// What a verb answers through `Receiver`: what the call produced, or a [`Deferred`](crate::Deferred).
type Answer<'trainer, Receiver, Verb> =
    <<Receiver as TrainVerbs<'trainer>>::Completion as Completion>::Answer<'trainer, Verb>;

/// Every trainer verb, written once for both receivers of [`Trainer::on`] and [`Trainer::deferred`],
/// which run on the executor the trainer was made on.
///
/// Rust-only.
pub trait TrainVerbs<'trainer>: Copy {
    /// [`Blocking`](crate::Blocking) or [`Deferring`].
    type Completion: Completion;

    /// The trainer this receiver reaches and the executor its calls run on.
    fn handles(&self) -> Handles;

    /// Applies one optimizer step over this process's items and every peer's: forward, losses,
    /// backward, the world gradient reduction, and the update. Every process of the run calls it
    /// once per step, with one modality per item and each item's bytes in `contents`.
    ///
    /// Answers the pooled embeddings when the config names an arena.
    ///
    /// Calls `uform_trainer_step`.
    fn step<'config, 'arena, Modalities: AsRef<[Modality]>, Contents: TapeView<u8>>(
        self,
        modalities: Modalities,
        contents: Contents,
        config: impl Into<TrainerStepConfig<'config, 'arena>>,
    ) -> Answer<'trainer, Self, TrainerStep<'config, 'arena, Modalities, Contents>>
    where
        Self::Completion: Accepts<Modalities> + Accepts<Contents> + Accepts<TrainerStepConfig<'config, 'arena>>,
    {
        <Self::Completion as Completion>::answer(TrainerStep {
            handles: self.handles(),
            modalities,
            contents,
            gathered: Gathered::default(),
            config: config.into(),
            embeddings: Exported::default(),
            count: 0,
        })
    }

    /// Writes the weights and the step, which the model loads serve and [`Trainer::init_vram`]
    /// resumes; only the process of index zero writes, the rest return at once.
    ///
    /// Calls `uform_trainer_save`.
    fn save<'config, Location: AsRef<Path>>(
        self,
        config: impl Into<TrainerSaveConfig<'config, Location>>,
    ) -> Answer<'trainer, Self, TrainerSave<'config, Location>>
    where
        Self::Completion: Accepts<TrainerSaveConfig<'config, Location>>,
    {
        <Self::Completion as Completion>::answer(TrainerSave {
            handles: self.handles(),
            config: config.into(),
        })
    }
}

impl<'trainer, Completion: crate::types::Completion> TrainVerbs<'trainer> for On<'trainer, Trainer<'_>, Completion> {
    type Completion = Completion;

    fn handles(&self) -> Handles {
        Handles {
            handle: self.handle.handle,
            executor: self.executor.handle,
        }
    }
}

// endregion: Trainer

// region: Calls

/// One optimizer step; see [`TrainVerbs::step`].
///
/// Calls `uform_trainer_step`.
pub struct TrainerStep<'config, 'arena, Modalities, Contents> {
    handles: Handles,
    modalities: Modalities,
    contents: Contents,
    gathered: Gathered<u8>,
    config: TrainerStepConfig<'config, 'arena>,
    embeddings: Exported<uform_rows_view_t>,
    count: usize,
}

impl<'arena, Modalities: AsRef<[Modality]>, Contents: TapeView<u8>> Calling
    for TrainerStep<'_, 'arena, Modalities, Contents>
{
    type Arguments = uform_trainer_step_t;
    type Output = Option<Rows<'arena>>;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        _: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.contents.tape(&mut self.gathered)?;
        let count = batch.count;
        let modalities = self.modalities.as_ref();
        let config = &mut self.config;
        if modalities.len() != count {
            return Err(fail(Status::ShapeMismatch, "every item needs one modality"));
        }
        if [config.bags.len(), config.temperature_groups.len(), config.sides.len()]
            .iter()
            .any(|&length| length != 0 && length != count)
        {
            return Err(fail(Status::ShapeMismatch, "pass no bags, groups or sides, or one per item"));
        }
        let (choices, choice_targets) = lower_choices(config.choices, config.choice_targets, count)?;
        let relations = uform_relations_view_t {
            attract: config.relations.attract.lower(count)?,
            repel: config.relations.repel.lower(count)?,
            ignore: config.relations.ignore.lower(count)?,
        };
        let optional = |list: &[u32]| match list.is_empty() {
            true => ptr::null(),
            false => list.as_ptr(),
        };
        self.count = count;
        Ok(uform_trainer_step_t {
            trainer: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            count,
            // `Modality` is `repr(i32)`, the platform `int` the C enum lowers to.
            modalities: modalities.as_ptr().cast(),
            contents: batch.tape,
            bags: optional(config.bags),
            temperature_groups: optional(config.temperature_groups),
            // `LateInteractionSide` is `repr(i32)`, the platform `int` the C enum lowers to.
            sides: match config.sides.is_empty() {
                true => ptr::null(),
                false => config.sides.as_ptr().cast(),
            },
            choices,
            choice_targets,
            relations,
            embeddings: match config.arena {
                Some(_) => ptr::from_mut(&mut self.embeddings.0),
                None => ptr::null_mut(),
            },
            arena: config.arena.as_deref().map_or(ptr::null_mut(), |arena| arena.handle),
            outputs_scalar: config.outputs_scalar as uform_scalar_t,
            stats: config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_trainer_step(arguments) }

    fn finish(self) -> Result<Option<Rows<'arena>>> {
        let (embeddings, count) = (self.embeddings, self.count);
        Ok(self.config.arena.map(|arena| Rows {
            arena,
            view: embeddings,
            count,
        }))
    }
}

/// A checkpoint write; see [`TrainVerbs::save`].
///
/// Calls `uform_trainer_save`.
pub struct TrainerSave<'config, Location> {
    handles: Handles,
    config: TrainerSaveConfig<'config, Location>,
}

impl<Location: AsRef<Path>> Calling for TrainerSave<'_, Location> {
    type Arguments = uform_trainer_save_t;
    type Output = ();

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let path = path_view(self.config.path.as_ref())?;
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(uform_trainer_save_t {
            trainer: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            path,
            progress,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_trainer_save(arguments) }

    fn finish(self) -> Result<()> { Ok(()) }
}

// endregion: Calls

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// A trainer whose handle is null, which every verb refuses past the binding's own checks.
    fn null_trainer() -> Trainer<'static> {
        Trainer {
            handle: ptr::null_mut(),
            mesh: PhantomData,
        }
    }

    fn status_of<Value>(result: Result<Value>) -> Option<Status> { result.err().map(|error| error.status) }

    #[test]
    fn training_enums_round_trip_through_the_library() {
        for modality in [Modality::Text, Modality::Image, Modality::Video, Modality::Audio] {
            assert_eq!(modality.to_string().parse(), Ok(modality));
        }
        for objective in [Objective::Masked, Objective::Contrastive, Objective::Joint] {
            assert_eq!(objective.to_string().parse(), Ok(objective));
        }
        for negatives in [Negatives::Ambient, Negatives::Named] {
            assert_eq!(negatives.to_string().parse(), Ok(negatives));
        }
        assert_eq!(Objective::Joint.to_string(), "joint");
        for tier in [MultimemTier::SwitchReduced, MultimemTier::PeerRead] {
            assert_eq!(tier.to_string().parse(), Ok(tier));
        }
        assert_eq!(MultimemTier::PeerRead.to_string(), "peer-read");
        assert!("ring".parse::<MultimemTier>().is_err());
        assert!(
            "speech".parse::<Modality>().is_err(),
            "the default a stranger reads as is refused"
        );
        assert!("both".parse::<Objective>().is_err());
        assert!("all".parse::<Negatives>().is_err());
    }

    #[test]
    fn training_structs_lay_out_as_the_header_declares() {
        use core::mem::{offset_of, size_of};

        let word = size_of::<usize>();
        assert_eq!(offset_of!(BatchShape, max_choices_per_item), 4 * word);
        assert_eq!(offset_of!(uform_contrastive_t, pair_loss), size_of::<uform_contrastive_t>() - word);
        assert_eq!(offset_of!(LateInteraction, pair_loss), 7 * word);
        assert_eq!(size_of::<Choices>(), 8);
        assert_eq!(offset_of!(uform_trainer_step_t, choices), offset_of!(uform_trainer_step_t, sides) + word);
        assert_eq!(
            offset_of!(uform_trainer_step_t, choice_targets),
            offset_of!(uform_trainer_step_t, choices) + 3 * word
        );
        assert_eq!(
            offset_of!(uform_trainer_step_t, relations),
            offset_of!(uform_trainer_step_t, choice_targets) + 3 * word
        );
        let late_interaction_loss = offset_of!(TrainerStepStats, late_interaction_loss);
        assert_eq!(late_interaction_loss, offset_of!(TrainerStepStats, masked_audio_loss) + 4);
        assert_eq!(offset_of!(TrainerStepStats, choices_loss), late_interaction_loss + 4);
        assert_eq!(offset_of!(TrainerStepStats, dropped), offset_of!(TrainerStepStats, truncated) + word);
        assert_eq!(offset_of!(uform_mesh_description_t, gpu_class), 5 * word + 4);
    }

    /// Training is GPU-only, so a build without GPUs refuses both constructors as the device's fault.
    #[test]
    fn a_build_without_gpus_refuses_to_train() -> Result<()> {
        let executor = Executor::for_ram(&[], None, 1, 0)?;
        let options = MeshOptions {
            directory: Path::new("/tmp"),
            job: "uform-rust",
            ..MeshOptions::default()
        };
        assert_eq!(status_of(Mesh::bootstrap(&executor, &options)), Some(Status::Device));
        #[cfg(unix)]
        {
            use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

            let options = MeshOptions {
                directory: Path::new(OsStr::from_bytes(b"/tmp/\xff")),
                ..options
            };
            let refused = status_of(Mesh::bootstrap(&executor, &options));
            assert_eq!(refused, Some(Status::Device), "a path crosses as the OS's bytes");
        }
        let config = TrainerInitVramConfig {
            depth: 2,
            hidden: 128,
            schedule: Schedule {
                steps: 10,
                ..Schedule::default()
            },
            contrastive: Contrastive {
                matryoshka_dimensions: &[64, 128],
                ..Contrastive::default()
            },
            ..TrainerInitVramConfig::default()
        };
        assert_eq!(status_of(Trainer::init_vram(&executor, config)), Some(Status::Device));
        Ok(())
    }

    /// The library reads every list a step names for its count of items, so a list of another
    /// length, or an edge naming an item outside the step, is refused before the call.
    #[test]
    fn steps_check_their_lists_before_the_call() -> Result<()> {
        let trainer = null_trainer();
        let executor = Executor::for_ram(&[], None, 1, 0)?;
        let receiver = trainer.on(&executor);
        let modalities = [Modality::Text, Modality::Image];
        let contents = ["a caption".as_bytes(), b"\xff\xd8\xff".as_slice()];

        let unpaired = receiver.step([Modality::Text], contents, TrainerStepConfig::default());
        assert_eq!(status_of(unpaired), Some(Status::ShapeMismatch));
        let bags = TrainerStepConfig {
            bags: &[0],
            ..TrainerStepConfig::default()
        };
        assert_eq!(
            status_of(receiver.step(modalities, contents, bags)),
            Some(Status::ShapeMismatch)
        );
        let sides = TrainerStepConfig {
            sides: &[LateInteractionSide::Query],
            ..TrainerStepConfig::default()
        };
        assert_eq!(
            status_of(receiver.step(modalities, contents, sides)),
            Some(Status::ShapeMismatch)
        );
        let untargeted = TrainerStepConfig {
            choices: (b"yn", &[0, 2, 2]),
            choice_targets: (&[1.0], &[0, 1, 1]),
            ..TrainerStepConfig::default()
        };
        assert_eq!(
            status_of(receiver.step(modalities, contents, untargeted)),
            Some(Status::ShapeMismatch)
        );
        let unpaired_targets = TrainerStepConfig {
            choices: (b"yn", &[0, 2, 2]),
            ..TrainerStepConfig::default()
        };
        assert_eq!(
            status_of(receiver.step(modalities, contents, unpaired_targets)),
            Some(Status::InvalidArgument)
        );

        fn edges(offsets: &'static [usize], ids: &'static [u32]) -> Relations<'static> {
            Relations {
                attract: Edges {
                    offsets,
                    ids,
                    weights: &[],
                },
                ..Relations::default()
            }
        }
        let refusals = [
            (edges(&[0, 1], &[1]), Status::ShapeMismatch),
            (edges(&[0, 1, 3], &[1, 0]), Status::Malformed),
            (edges(&[0, 1, 2], &[1, 2]), Status::InvalidArgument),
            (edges(&[], &[1]), Status::InvalidArgument),
        ];
        for (relations, status) in refusals {
            assert_eq!(status_of(receiver.step(modalities, contents, relations)), Some(status));
        }

        // A well-formed step reaches the library, which refuses the null trainer or, first, the RAM executor.
        let refused = [Some(Status::InvalidArgument), Some(Status::Device)];
        let mut arena = Arena::new_in(&executor)?;
        let config = TrainerStepConfig {
            relations: edges(&[0, 1, 2], &[1, 0]),
            arena: Some(&mut arena),
            ..TrainerStepConfig::default()
        };
        assert!(refused.contains(&status_of(receiver.step(modalities, contents, config))));
        let deferred =
            trainer
                .deferred(&executor)
                .step(vec![Modality::Text], vec!["a"], TrainerStepConfig::default())?;
        assert!(refused.contains(&status_of(deferred.wait())));
        assert!(refused.contains(&status_of(receiver.save("/tmp/uform-rust.safetensors"))));
        Ok(())
    }
}

// endregion: Tests
