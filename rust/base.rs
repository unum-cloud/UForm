//! The Base model: the raw encoder tower pretraining produces — the per-modality entry, the
//! pre-norm transformer block stack, and the final RMSNorm — yielding per-token
//! [`HiddenStates`]. The op order and numerics mirror `cuda_core/encoder_tower.hpp` exactly:
//!
//! ```text
//!   entry — per-modality gather or projection
//!     ↓
//!   × depth:
//!     RMSNorm → fused Q·K·V → unit QK-norm → M-RoPE → ragged SDPA → output → ⊕ residual
//!     RMSNorm → fused gate|up → SiLU(gate)·up → down → ⊕ residual
//!     ↓
//!   RMSNorm → HiddenStates
//! ```
//!
//! Every dimension and hyperparameter derives from the checkpoint: the geometry from tensor
//! shapes, `head_width`, `rotary_base`, and `rms_epsilon` from its metadata — with the training
//! stack's defaults when a checkpoint predates those keys. RoPE pairs channels NeoX split-half,
//! channel k against k + head_width⁄2; text coordinates restart per document, so the spatial
//! pairs sit at identity and the rotation rows are prefix gathers of one precomputed table.
//! Weights are row-major `[out, in]`, so every product is `y = x·Wᵀ` — NumKong's packed `dots`
//! contract. Norms, rotations, and attention operands run in the [`Precision`]'s operand dtype,
//! matching the CUDA tower's recipe. Media entries are not ported yet — text only.
//!
//! ```no_run
//! use uform::{BaseModel, Bf16, HiddenStates};
//! use uform::base::Scratch;
//!
//! # fn main() -> uform::Result<()> {
//! let model = BaseModel::<Bf16>::open("model.safetensors".as_ref(), 8192)?;
//! let mut scratch = Scratch::new();
//! let mut states = HiddenStates::new();
//! model.forward(&mut scratch, &mut states, &["first document", "the second"], None);
//! let first_document_states = states.segment(0);
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use fork_union::ThreadPool;
use numkong::{
    bf16, configure_thread, Attention, AttentionKeyValueCache, CastDtype, CastOps, DotsPackedOps,
    DotsPackedParallelOps, EachSwiglu, Global, Matrix, MatrixView, PackedMatrix, ReduceRmsNorm,
    StorageElement, TrigRope,
};

use crate::checkpoint::{SafeTensors, TensorPayload};
use crate::error::{Error, Result};

/// The training stack's values for checkpoints that predate self-describing metadata.
const DEFAULT_HEAD_WIDTH: usize = 128;
const DEFAULT_ROTARY_BASE: f64 = 10000.0;
const DEFAULT_RMS_EPSILON: f32 = 1.0e-6;

/// Ragged segment offsets in NumKong's attention ABI — the one place the crate speaks `u32`.
pub(crate) type TokenOffset = u32;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for numkong::bf16 {}
    impl Sealed for numkong::e4m3 {}
}

/// The scalar a [`Precision`] computes in: packed-GEMM and attention operand, plus the fused
/// normalization / rotation kernels that run on the staged activations. Sealed — the set of
/// operand scalars is fixed by the NumKong kernels, not extensible downstream.
pub trait Operand:
    sealed::Sealed
    + Attention
    + CastDtype
    + ReduceRmsNorm
    + TrigRope
    + StorageElement
    + Copy
    + Send
    + Sync
{
}
impl Operand for bf16 {}
impl Operand for numkong::e4m3 {}

/// How a tower stores and computes its projection weights — the Rust spelling of the CUDA
/// twins' `_bf16_t` / `_e4m3_t` suffix, as a zero-sized marker in type position:
/// `BaseModel<Bf16>`. Block-scaled E4M3 and I8 markers arrive later without touching the
/// layers — the attention and GEMM kernels already implement both scalars.
pub trait Precision: Sized + Send + Sync + 'static {
    type Operand: Operand;
    /// The packed weight storage — `PackedMatrix` today, block-scaled tensors later.
    type Packed: Send + Sync;
    /// Pack one `[out, in]` checkpoint matrix — once, at load.
    ///
    /// # Errors
    /// [`Error::MissingTensor`] when the tensor is absent; [`Error::UnexpectedDtype`] when it is
    /// stored in a dtype this precision cannot pack; [`Error::Kernel`] on pack or allocation
    /// failure.
    fn import(weights: &SafeTensors, name: &str) -> Result<Self::Packed>;
    fn rows(packed: &Self::Packed) -> usize;
    fn cols(packed: &Self::Packed) -> usize;
    /// `out = staged·Wᵀ`, fanned over the pool when one is provided.
    fn project_into(packed: &Self::Packed, staged: &Matrix<Self::Operand>, out: &mut Matrix<f32>,
                    parallelism: Option<&mut ThreadPool>);
}

/// The training precision: BF16 operands, F32 accumulation, weights packed zero-copy off the
/// checkpoint's own BF16 bits.
pub struct Bf16;

impl Precision for Bf16 {
    type Operand = bf16;
    type Packed = PackedMatrix<bf16>;

    fn import(weights: &SafeTensors, name: &str) -> Result<Self::Packed> {
        let payload = weights.view(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
        match payload {
            TensorPayload::Bf16(view) => PackedMatrix::try_pack(&view).map_err(Error::kernel),
            TensorPayload::F32(view) => {
                let mut staged =
                    Matrix::<bf16>::try_zeros(view.shape()).map_err(Error::kernel)?;
                view.try_cast_into(&mut staged).map_err(Error::kernel)?;
                PackedMatrix::try_pack(&staged).map_err(Error::kernel)
            }
            TensorPayload::I8(_) => Err(Error::UnexpectedDtype {
                name: name.into(),
                found: "I8",
                expected: "BF16 or F32",
            }),
        }
    }

    fn rows(packed: &Self::Packed) -> usize {
        packed.dims().0
    }

    fn cols(packed: &Self::Packed) -> usize {
        packed.dims().1
    }

    fn project_into(packed: &Self::Packed, staged: &Matrix<bf16>, out: &mut Matrix<f32>,
                    parallelism: Option<&mut ThreadPool>) {
        match parallelism {
            Some(pool) => staged.try_dots_packed_parallel_into(packed, out, pool)
                .expect("invariant: GEMM shapes fixed at load"),
            None => staged.try_dots_packed_into(packed, out)
                .expect("invariant: GEMM shapes fixed at load"),
        }
    }
}

/// The tower's load-time dimensions, grouped so scratch sizing takes one parameter instead of
/// a parade of constants.
pub(crate) struct Geometry {
    pub(crate) hidden: usize,
    pub(crate) head_width: usize,
    pub(crate) feed_forward_width: usize,
}

/// A `[1, 1]` placeholder for scratch fields before the first `ensure` grows them.
pub(crate) fn empty<Scalar: StorageElement + Default + Clone>() -> Matrix<Scalar> {
    Matrix::<Scalar>::try_zeros(&[1, 1]).expect("invariant: a 1×1 allocation cannot fail")
}

/// A zeroed matrix at the requested shape — the scratch growth primitive.
pub(crate) fn grow<Scalar: StorageElement + Default + Clone>(shape: &[usize; 2]) -> Matrix<Scalar> {
    Matrix::<Scalar>::try_zeros(shape).expect("scratch allocation")
}

/// Text-degenerate rotation rows, fully precomputed: position `p`'s row holds the
/// temporal pairs at `angle = p·base^(−2·pair⁄head_width)` and identity — cos 1, sin 0 — for the
/// spatial pairs a text coordinate leaves untouched. Positions restart per
/// document, so a batch's rotation rows are per-segment prefix gathers of these tables.
struct RotaryTables {
    /// [max_positions, pairs]
    cosines: Matrix<f32>,
    /// [max_positions, pairs]
    sines: Matrix<f32>,
}

impl RotaryTables {
    fn build(max_positions: usize, head_width: usize, base: f64) -> Result<Self> {
        let pairs = head_width / 2;
        let temporal_pairs = head_width / 4;
        let mut cosines =
            Matrix::<f32>::try_full(&[max_positions, pairs], 1.0).map_err(Error::kernel)?;
        let mut sines =
            Matrix::<f32>::try_zeros(&[max_positions, pairs]).map_err(Error::kernel)?;
        let (cos_cells, sin_cells) = (cosines.as_mut_slice(), sines.as_mut_slice());
        for position in 0..max_positions {
            for pair in 0..temporal_pairs {
                let angle = position as f64 * base.powf(-2.0 * pair as f64 / head_width as f64);
                cos_cells[position * pairs + pair] = angle.cos() as f32;
                sin_cells[position * pairs + pair] = angle.sin() as f32;
            }
        }
        Ok(Self { cosines, sines })
    }

    /// Gather each segment's `[length, pairs]` prefix rows into the scratch grids.
    fn gather_into(&self, offsets: &[TokenOffset], cosine_rows: &mut Matrix<f32>,
                   sine_rows: &mut Matrix<f32>) {
        let pairs = self.cosines.shape()[1];
        let tokens = offsets[offsets.len() - 1] as usize;
        cosine_rows.try_resize(&[tokens, pairs]).expect("invariant: sized by ensure");
        sine_rows.try_resize(&[tokens, pairs]).expect("invariant: sized by ensure");
        for segment in offsets.windows(2) {
            let (start, stop) = (segment[0] as usize, segment[1] as usize);
            let cells = (stop - start) * pairs;
            cosine_rows.as_mut_slice()[start * pairs..start * pairs + cells]
                .copy_from_slice(&self.cosines.as_slice()[..cells]);
            sine_rows.as_mut_slice()[start * pairs..start * pairs + cells]
                .copy_from_slice(&self.sines.as_slice()[..cells]);
        }
    }
}

/// One `y = x·Wᵀ` projection: stages the f32 source into the precision's operand dtype, then
/// runs the packed GEMM into the caller's accumulator matrix.
pub(crate) struct Linear<P: Precision> {
    pub(crate) packed: P::Packed,
}

impl<P: Precision> Linear<P> {
    pub(crate) fn load(weights: &SafeTensors, name: &str) -> Result<Self> {
        Ok(Self { packed: P::import(weights, name)? })
    }

    pub(crate) fn project_into(&self, source: &Matrix<f32>, stage: &mut Matrix<P::Operand>,
                               out: &mut Matrix<f32>, parallelism: Option<&mut ThreadPool>) {
        let tokens = source.shape()[0];
        debug_assert_eq!(source.shape()[1], P::cols(&self.packed));
        stage.try_resize(&[tokens, P::cols(&self.packed)]).expect("invariant: sized by ensure");
        source.try_cast_into(stage).expect("invariant: matching shapes");
        out.try_resize(&[tokens, P::rows(&self.packed)]).expect("invariant: sized by ensure");
        P::project_into(&self.packed, stage, out, parallelism);
    }
}

/// The byte-level token table; the entry is a per-byte row gather — the one indexed copy no
/// kernel expresses.
struct Embedding {
    /// [vocabulary, hidden]
    table: Matrix<f32>,
}

impl Embedding {
    /// Gather each document's byte rows; `offsets` is the single truncation authority — exactly
    /// `offsets[d+1] − offsets[d]` leading bytes of document `d` are consumed.
    fn gather_into<Document: AsRef<[u8]>>(&self, documents: &[Document], offsets: &[TokenOffset],
                                          destination: &mut Matrix<f32>) {
        let hidden = self.table.shape()[1];
        let tokens = offsets[offsets.len() - 1] as usize;
        destination.try_resize(&[tokens, hidden]).expect("invariant: sized by ensure");
        let table = self.table.as_slice();
        let cells = destination.as_mut_slice();
        for (document, segment) in documents.iter().zip(offsets.windows(2)) {
            let (start, stop) = (segment[0] as usize, segment[1] as usize);
            for (position, byte) in document.as_ref()[..stop - start].iter().enumerate() {
                let row = *byte as usize * hidden;
                let cell = (start + position) * hidden;
                cells[cell..cell + hidden].copy_from_slice(&table[row..row + hidden]);
            }
        }
    }
}

/// Learnable RMSNorm: `y = x·rsqrt(mean(x²) + ε)·γ`.
pub(crate) struct RmsNorm {
    /// [1, hidden]
    gamma: Matrix<f32>,
    epsilon: f32,
}

impl RmsNorm {
    pub(crate) fn load(weights: &SafeTensors, name: &str, epsilon: f32) -> Result<Self> {
        let gamma = weights.view(name).and_then(|payload| payload.try_decode::<f32>())
            .ok_or_else(|| Error::MissingTensor(name.into()))?;
        Ok(Self { gamma, epsilon })
    }

    pub(crate) fn forward(&self, x: &Matrix<f32>, y: &mut Matrix<f32>) {
        y.try_resize(&[x.shape()[0], x.shape()[1]]).expect("invariant: sized by ensure");
        f32::rmsnorm_into(x, Some(self.gamma.as_slice()), y, 1, self.epsilon, 1.0)
            .expect("invariant: matching shapes");
    }
}

/// The attention sub-block's staging buffers: the operand stage for its two GEMMs, the fused
/// QKV accumulator, its pre-norm and rotated operand twins — the same a → b dataflow as the
/// CUDA tower — and the SDPA output.
struct AttentionScratch<O: Operand> {
    /// [tokens, hidden] — GEMM operand staging.
    stage: Matrix<O>,
    /// [tokens, 3·hidden]
    query_key_value: Matrix<f32>,
    /// [tokens, 3·hidden] — the Q|K sections staged before the QK-norm.
    pre_norm: Matrix<O>,
    /// [tokens, 3·hidden] — normalized and rotated Q|K plus the staged V, what SDPA consumes.
    rotated: Matrix<O>,
    /// [tokens, hidden]
    context: Matrix<f32>,
    /// The packed KV blob, allocated once and refreshed in place every forward.
    key_value: AttentionKeyValueCache<O>,
}

/// The attention sub-block: fused QKV projection, unit QK-norm and NeoX rotation on the staged
/// operands, the fused ragged SDPA over the batch's segments, and the output projection. Owns
/// the two metadata-borne constants its kernels need; everything else derives from its weights.
struct SelfAttention<P: Precision> {
    query_key_value: Linear<P>,
    output: Linear<P>,
    head_width: usize,
    epsilon: f32,
}

impl<P: Precision> SelfAttention<P> {
    fn forward(&self, normalized: &Matrix<f32>, scratch: &mut AttentionScratch<P::Operand>,
               offsets: &[TokenOffset], cosines: &Matrix<f32>, sines: &Matrix<f32>,
               out: &mut Matrix<f32>, mut parallelism: Option<&mut ThreadPool>) {
        let tokens = normalized.shape()[0];
        let hidden = P::cols(&self.query_key_value.packed);
        let heads = hidden / self.head_width;
        self.query_key_value.project_into(normalized, &mut scratch.stage,
                                          &mut scratch.query_key_value,
                                          parallelism.as_deref_mut());

        // Q and K land in `pre_norm`; V goes straight to its final place in `rotated`; the
        // QK-norm writes pre_norm → rotated and the rotation runs in place — every section a
        // strided column view of the fused buffers.
        scratch.pre_norm.try_resize(&[tokens, 3 * hidden]).expect("invariant: sized by ensure");
        scratch.rotated.try_resize(&[tokens, 3 * hidden]).expect("invariant: sized by ensure");
        scratch.query_key_value.try_slice((.., 0..2 * hidden)).expect("invariant: fused columns")
            .try_cast_into(&mut scratch.pre_norm.slice_mut((.., 0..2 * hidden))
                .expect("invariant: fused columns"))
            .expect("invariant: matching shapes");
        scratch.query_key_value.try_slice((.., 2 * hidden..3 * hidden))
            .expect("invariant: fused columns")
            .try_cast_into(&mut scratch.rotated.slice_mut((.., 2 * hidden..3 * hidden))
                .expect("invariant: fused columns"))
            .expect("invariant: matching shapes");
        for role in [0usize, hidden] {
            let pre_norm = scratch.pre_norm.try_slice((.., role..role + hidden))
                .expect("invariant: fused columns");
            let mut rotated = scratch.rotated.slice_mut((.., role..role + hidden))
                .expect("invariant: fused columns");
            P::Operand::rmsnorm_into(&pre_norm, None, &mut rotated, heads, self.epsilon, 1.0)
                .expect("invariant: matching shapes");
            P::Operand::rope_into(&mut rotated, cosines.as_slice(), sines.as_slice(), heads,
                                  self.head_width / 2, 1.0)
                .expect("invariant: rotation rows gathered per segment");
        }

        let key = scratch.rotated.try_slice((.., hidden..2 * hidden))
            .expect("invariant: fused columns");
        let value = scratch.rotated.try_slice((.., 2 * hidden..3 * hidden))
            .expect("invariant: fused columns");
        let query = scratch.rotated.try_slice((.., 0..hidden)).expect("invariant: fused columns");
        scratch.context.try_resize(&[tokens, hidden]).expect("invariant: sized by ensure");
        scratch.key_value.try_pack_into(&key, &value, self.head_width, offsets, None)
            .expect("invariant: offsets cover the batch");
        match parallelism.as_deref_mut() {
            Some(pool) => scratch.key_value
                .try_attention_parallel_into(&query, offsets, None, &mut scratch.context, pool)
                .expect("invariant: packed KV matches the query grid"),
            None => scratch.key_value
                .try_attention_into(&query, offsets, None, &mut scratch.context)
                .expect("invariant: packed KV matches the query grid"),
        }

        self.output.project_into(&scratch.context, &mut scratch.stage, out, parallelism);
    }
}

/// The SwiGLU feed-forward's staging buffers: the operand stage for its two GEMMs, the fused
/// gate|up accumulator, and the activation.
struct FeedForwardScratch<O: Operand> {
    /// [tokens, max(hidden, feed_forward_width)] — GEMM operand staging.
    stage: Matrix<O>,
    /// [tokens, 2·feed_forward_width]
    gate_up: Matrix<f32>,
    /// [tokens, feed_forward_width]
    activated: Matrix<f32>,
}

/// The SwiGLU feed-forward: fused gate|up projection, `silu(gate)·up` over its column halves,
/// and the down projection. Its width derives from its own weights.
struct FeedForward<P: Precision> {
    gate_up: Linear<P>,
    down: Linear<P>,
}

impl<P: Precision> FeedForward<P> {
    fn forward(&self, normalized: &Matrix<f32>, scratch: &mut FeedForwardScratch<P::Operand>,
               out: &mut Matrix<f32>, mut parallelism: Option<&mut ThreadPool>) {
        let tokens = normalized.shape()[0];
        let feed_forward_width = P::rows(&self.gate_up.packed) / 2;
        self.gate_up.project_into(normalized, &mut scratch.stage, &mut scratch.gate_up,
                                  parallelism.as_deref_mut());
        scratch.activated.try_resize(&[tokens, feed_forward_width])
            .expect("invariant: sized by ensure");
        let gate = scratch.gate_up.try_slice((.., 0..feed_forward_width))
            .expect("invariant: fused columns");
        let up = scratch.gate_up.try_slice((.., feed_forward_width..2 * feed_forward_width))
            .expect("invariant: fused columns");
        f32::swiglu_into(&gate, Some(&up), &mut scratch.activated, 1.0)
            .expect("invariant: matching shapes");
        self.down.project_into(&scratch.activated, &mut scratch.stage, out, parallelism);
    }
}

/// One pre-norm transformer block; `forward` folds both residuals into `scratch.residual` in
/// place.
struct TransformerBlock<P: Precision> {
    pre: RmsNorm,
    attention: SelfAttention<P>,
    post: RmsNorm,
    feed_forward: FeedForward<P>,
}

impl<P: Precision> TransformerBlock<P> {
    fn load(weights: &SafeTensors, index: usize, head_width: usize, epsilon: f32) -> Result<Self> {
        let name = |part: &str| format!("model.blocks.{index}.{part}");
        Ok(Self {
            pre: RmsNorm::load(weights, &name("pre_norm"), epsilon)?,
            attention: SelfAttention {
                query_key_value: Linear::load(weights, &name("qkv"))?,
                output: Linear::load(weights, &name("out"))?,
                head_width,
                epsilon,
            },
            post: RmsNorm::load(weights, &name("post_norm"), epsilon)?,
            feed_forward: FeedForward {
                gate_up: Linear::load(weights, &name("gate_up"))?,
                down: Linear::load(weights, &name("down"))?,
            },
        })
    }

    fn forward(&self, scratch: &mut Scratch<P>, offsets: &[TokenOffset],
               mut parallelism: Option<&mut ThreadPool>) {
        self.pre.forward(&scratch.residual, &mut scratch.normalized);
        self.attention.forward(&scratch.normalized, &mut scratch.attention, offsets,
                               &scratch.rotation_cosines, &scratch.rotation_sines,
                               &mut scratch.projected, parallelism.as_deref_mut());
        scratch.residual.span().add_inplace(&scratch.projected.view())
            .expect("invariant: matching shapes");
        self.post.forward(&scratch.residual, &mut scratch.normalized);
        self.feed_forward.forward(&scratch.normalized, &mut scratch.feed_forward,
                                  &mut scratch.projected, parallelism);
        scratch.residual.span().add_inplace(&scratch.projected.view())
            .expect("invariant: matching shapes");
    }
}

/// Every buffer one Base forward touches, owned per encode stream, grown once to the largest
/// observed token count and resized in place per call.
pub struct Scratch<P: Precision> {
    /// [tokens, hidden] — the residual stream.
    residual: Matrix<f32>,
    /// [tokens, hidden]
    normalized: Matrix<f32>,
    /// [tokens, hidden] — output / down projection target.
    projected: Matrix<f32>,
    /// [tokens, pairs] — per-segment prefix gathers of the rotary tables.
    rotation_cosines: Matrix<f32>,
    /// [tokens, pairs]
    rotation_sines: Matrix<f32>,
    attention: AttentionScratch<P::Operand>,
    feed_forward: FeedForwardScratch<P::Operand>,
}

impl<P: Precision> Scratch<P> {
    pub fn new() -> Self {
        Self {
            residual: empty(),
            normalized: empty(),
            projected: empty(),
            rotation_cosines: empty(),
            rotation_sines: empty(),
            attention: AttentionScratch {
                stage: empty(),
                query_key_value: empty(),
                pre_norm: empty(),
                rotated: empty(),
                context: empty(),
                key_value: AttentionKeyValueCache::empty_in(Global),
            },
            feed_forward: FeedForwardScratch {
                stage: empty(),
                gate_up: empty(),
                activated: empty(),
            },
        }
    }

    fn ensure(&mut self, tokens: usize, geometry: &Geometry) {
        let Geometry { hidden, head_width, feed_forward_width } = *geometry;
        if self.residual.capacity() >= tokens * hidden {
            return;
        }
        self.residual = grow(&[tokens, hidden]);
        self.normalized = grow(&[tokens, hidden]);
        self.projected = grow(&[tokens, hidden]);
        self.rotation_cosines = grow(&[tokens, head_width / 2]);
        self.rotation_sines = grow(&[tokens, head_width / 2]);
        self.attention.stage = grow(&[tokens, hidden]);
        self.attention.query_key_value = grow(&[tokens, 3 * hidden]);
        self.attention.pre_norm = grow(&[tokens, 3 * hidden]);
        self.attention.rotated = grow(&[tokens, 3 * hidden]);
        self.attention.context = grow(&[tokens, hidden]);
        self.feed_forward.stage = grow(&[tokens, hidden.max(feed_forward_width)]);
        self.feed_forward.gate_up = grow(&[tokens, 2 * feed_forward_width]);
        self.feed_forward.activated = grow(&[tokens, feed_forward_width]);
    }
}

impl<P: Precision> Default for Scratch<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// The Base model's product: the final-norm per-token states of one packed batch, plus the
/// segment offsets that slice them per document. Grows itself as batches demand — construct
/// with [`HiddenStates::new`], or pre-reserve with [`HiddenStates::with_capacity`]. "Token"
/// here is modality-agnostic: text bytes, image patches, video tubes, and audio frames all
/// become tokens at the entry.
pub struct HiddenStates {
    /// [tokens, hidden]
    states: Matrix<f32>,
    /// [segments + 1]
    offsets: Vec<TokenOffset>,
}

impl HiddenStates {
    pub fn new() -> Self {
        Self { states: empty(), offsets: Vec::new() }
    }

    /// Pre-reserve for a known batch footprint — purely an optimization hint.
    pub fn with_capacity(max_tokens: usize, hidden: usize) -> Self {
        Self { states: grow(&[max_tokens, hidden]), offsets: Vec::new() }
    }

    /// Storage capacity in scalars — the growth predicate for reusing one container.
    pub fn capacity(&self) -> usize {
        self.states.capacity()
    }

    /// All segments' states, packed `[tokens, hidden]`.
    pub fn states(&self) -> MatrixView<'_, f32> {
        self.states.view()
    }

    /// The cumulative segment offsets, `segments + 1` long — the authoritative per-document
    /// slicing of [`Self::states`].
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn segments(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// One document's states, `[length, hidden]`.
    ///
    /// # Panics
    /// Panics when `index >= self.segments()`.
    pub fn segment(&self, index: usize) -> MatrixView<'_, f32> {
        let (start, stop) = (self.offsets[index] as usize, self.offsets[index + 1] as usize);
        self.states.try_slice((start..stop, ..)).expect("invariant: offsets bound the states")
    }
}

impl Default for HiddenStates {
    fn default() -> Self {
        Self::new()
    }
}

/// The raw encoder tower: entry, block stack, final norm. Weights are `Sync`; the caller brings
/// a [`Scratch`] and a reusable [`HiddenStates`].
pub struct BaseModel<P: Precision> {
    geometry: Geometry,
    rms_epsilon: f32,
    max_positions: usize,
    embedding: Embedding,
    blocks: Vec<TransformerBlock<P>>,
    final_norm: RmsNorm,
    rotary: RotaryTables,
}

impl<P: Precision> BaseModel<P> {
    /// Load from an opened checkpoint, deriving the geometry from tensor shapes and the
    /// hyperparameters from its metadata. Documents longer than `max_positions` bytes are
    /// truncated to it on every forward.
    ///
    /// # Errors
    /// [`Error::MissingTensor`] / [`Error::UnexpectedDtype`] / [`Error::Kernel`] when the
    /// checkpoint lacks a required tensor, stores it in a dtype the precision cannot pack, or a
    /// pack allocation fails.
    pub fn load(weights: &SafeTensors, max_positions: usize) -> Result<Self> {
        configure_thread();
        let [_, hidden] = weights.shape("model.embedding")
            .ok_or_else(|| Error::MissingTensor("model.embedding".into()))?;
        let depth = (0..).take_while(|i| weights.contains(&format!("model.blocks.{i}.qkv"))).count();
        let [gate_up_rows, _] = weights.shape("model.blocks.0.gate_up")
            .ok_or_else(|| Error::MissingTensor("model.blocks.0.gate_up".into()))?;
        let head_width = weights.metadata("head_width").unwrap_or(DEFAULT_HEAD_WIDTH);
        let rotary_base = weights.metadata("rotary_base").unwrap_or(DEFAULT_ROTARY_BASE);
        let rms_epsilon = weights.metadata("rms_epsilon").unwrap_or(DEFAULT_RMS_EPSILON);
        let table = weights.view("model.embedding")
            .and_then(|payload| payload.try_decode::<f32>())
            .ok_or_else(|| Error::MissingTensor("model.embedding".into()))?;
        Ok(Self {
            geometry: Geometry { hidden, head_width, feed_forward_width: gate_up_rows / 2 },
            rms_epsilon,
            max_positions,
            embedding: Embedding { table },
            blocks: (0..depth)
                .map(|i| TransformerBlock::load(weights, i, head_width, rms_epsilon))
                .collect::<Result<_>>()?,
            final_norm: RmsNorm::load(weights, "model.final_norm", rms_epsilon)?,
            rotary: RotaryTables::build(max_positions, head_width, rotary_base)?,
        })
    }

    /// Convenience: open the checkpoint file and load.
    ///
    /// # Errors
    /// Everything [`Self::load`] returns, plus [`Error::Io`] / [`Error::MalformedHeader`] from
    /// the file itself.
    pub fn open(path: &Path, max_positions: usize) -> Result<Self> {
        Self::load(&SafeTensors::open(path)?, max_positions)
    }

    pub fn hidden(&self) -> usize {
        self.geometry.hidden
    }

    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    pub fn head_width(&self) -> usize {
        self.geometry.head_width
    }

    pub fn heads(&self) -> usize {
        self.geometry.hidden / self.geometry.head_width
    }

    pub fn feed_forward_width(&self) -> usize {
        self.geometry.feed_forward_width
    }

    pub fn rms_epsilon(&self) -> f32 {
        self.rms_epsilon
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    pub(crate) fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The batch's token footprint under the truncation contract: `Σ min(len, max_positions)`.
    pub fn token_count<Document: AsRef<[u8]>>(&self, documents: &[Document]) -> usize {
        documents.iter().map(|document| document.as_ref().len().min(self.max_positions)).sum()
    }

    /// One packed forward over a batch of documents, each an independent ragged attention
    /// segment — every GEMM carries all their rows, amortizing each streamed weight tile across
    /// the batch. Documents longer than `max_positions` bytes are truncated to it. Batch enough
    /// rows to keep the AMX tiles busy: hundreds of thousands per call, e.g. 128 documents of
    /// 4096 bytes. The final RMSNorm writes straight into `states`, which grows as needed.
    pub fn forward<Document: AsRef<[u8]>>(&self, scratch: &mut Scratch<P>,
                                          states: &mut HiddenStates, documents: &[Document],
                                          mut parallelism: Option<&mut ThreadPool>) {
        let mut tokens = 0usize;
        states.offsets.clear();
        states.offsets.push(0);
        for document in documents {
            tokens += document.as_ref().len().min(self.max_positions);
            states.offsets.push(tokens as TokenOffset);
        }
        scratch.ensure(tokens, &self.geometry);
        if states.capacity() < tokens * self.geometry.hidden {
            states.states = grow(&[tokens, self.geometry.hidden]);
        }

        self.embedding.gather_into(documents, &states.offsets, &mut scratch.residual);
        self.rotary.gather_into(&states.offsets, &mut scratch.rotation_cosines,
                                &mut scratch.rotation_sines);
        for block in &self.blocks {
            block.forward(scratch, &states.offsets, parallelism.as_deref_mut());
        }
        states.states.try_resize(&[tokens, self.geometry.hidden])
            .expect("invariant: grown above");
        self.final_norm.forward(&scratch.residual, &mut states.states);
    }
}
