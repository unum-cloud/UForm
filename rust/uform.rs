//! UForm v4: a PyTorch-free CPU inference library for multimodal embeddings, built on NumKong's
//! packed AMX / SVE kernels — no interpreter, no raw FFI, no hot-path allocations beyond the
//! attention KV blobs, pending NumKong's `try_repack`.
//!
//! The crate mirrors the model family the training stack produces:
//!
//! - [`checkpoint`] — the SafeTensors reader, handing out zero-copy typed payload views.
//! - [`base`] — [`BaseModel`], the raw encoder tower pretraining produces: the per-modality
//!   entry, the transformer block stack, and the final RMSNorm, yielding per-token
//!   [`HiddenStates`]. Geometry and hyperparameters derive from the checkpoint itself.
//! - [`embedder`] — [`EmbedderModel`], the contrastive product built on top of a Base: the
//!   attention pool and the projection head, yielding one [`Embeddings`] vector per document.
//!
//! Everything is generic over a [`Precision`]: [`Bf16`] runs the training dtype end to end,
//! packed zero-copy off the checkpoint mmap; block-scaled E4M3 and I8 markers arrive later
//! without touching the layers. The `forward` interfaces are batch-only — documents ride as
//! independent ragged segments of one packed pass — and take an optional `fork_union` pool that
//! fans the GEMM and attention kernels across cores. Loading and parsing return [`Result`];
//! forward passes never do — their shapes are load-time invariants, and the output containers
//! grow themselves.

pub mod error;

pub mod checkpoint;

pub mod base;
pub mod embedder;

pub use base::{BaseModel, Bf16, HiddenStates, Precision};
pub use checkpoint::{SafeTensors, TensorPayload};
pub use embedder::{Embeddings, EmbedderModel, EmbedderScratch};
pub use error::{Error, Result};
