//! UForm - Distributed training and inference pipeline for multi-modal perception.
//!
//! The crate links the precompiled library, so building it compiles no tower.
//!
//! ```no_run
//! use uform::*;
//!
//! let executor = Executor::for_ram(&[], None, 0)?;
//! let mut arena = Arena::new_in(&executor)?;
//! let model = Model::load_ram("model.safetensors")?;
//! let texts = ["a photo of a cat", "a dog on the beach"];
//! let encoded = model.on(&executor).text_encode(&texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
//! if let Some(embeddings) = encoded.embeddings {
//!     let values: &[f32] = embeddings.values()?;
//!     println!("{} embeddings of {} dimensions", embeddings.count(), embeddings.columns());
//! }
//! # Ok::<(), uform::Error>(())
//! ```
//!
//! File: rust/lib.rs
//! Author: Ash Vardanian
//! Date: October 18, 2026

#![warn(missing_docs)]

mod encode;
mod model;
mod train;
mod types;

// The C mirror items are public within their modules, so only these names leave the crate.
pub use encode::{
    AudioEncode, AudioEncodeConfig, AudioEncodeSamples, AudioEncodeSamplesConfig, AudioEncodeStats, EncodeVerbs,
    Encoded, ImageEncode, ImageEncodeConfig, ImageEncodePixels, ImageEncodePixelsConfig, ImageEncodeStats, Outputs,
    TextChoose, TextChooseConfig, TextEncode, TextEncodeConfig, TextEncodeStats, VideoEncode, VideoEncodeConfig,
    VideoEncodeStats,
};
pub use model::{
    Model, ModelDescribe, ModelDescription, ModelLoadRamBufferConfig, ModelLoadRamConfig, ModelLoadVramBufferConfig,
    ModelLoadVramConfig, ModelVerbs,
};
pub use train::{
    AudioShape, BatchShape, Choices, Contrastive, Edges, ImageShape, LateInteraction, LateInteractionSide, Masking,
    Mesh, MeshDescription, MeshOptions, Modality, MultimemTier, Negatives, Objective, PairLossForm, Relations,
    Schedule, TrainVerbs, Trainer, TrainerInitVramConfig, TrainerSave, TrainerSaveConfig, TrainerStep,
    TrainerStepConfig, TrainerStepStats, VideoShape,
};
pub use types::{
    capabilities_available, capabilities_compiled, gpu_devices, version, Arena, ArenaDescription, Blocking, CPUClass,
    Deferred, Deferring, Element, Error, Executor, GPUClass, GPUDevice, GPUVendor, Mood, On, ProgressCallback,
    Residency, Result, Rows, RowsTape, Scalar, Status, Tape, Wait,
};
