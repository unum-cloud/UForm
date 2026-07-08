//! The Embedder model: the contrastive embedding product built on top of a [`BaseModel`] — the
//! learned single-query attention pool over the Base's hidden states, then the
//! `RMSNorm → intermediate → SiLU → output` projection head — yielding one embedding vector per
//! document. The pool and head tiers stay BF16 in every precision, feeding the cosine geometry
//! directly, matching the CUDA twins' split in `cuda_core/encoder_tower.hpp`.

use std::path::Path;

use fork_union::ThreadPool;
use numkong::{bf16, AttentionKeyValueCache, EachSwiglu, Global, Matrix, MatrixView};

use crate::base::{empty, grow, BaseModel, Bf16, HiddenStates, Linear, Precision, RmsNorm,
                  Scratch, TokenOffset};
use crate::checkpoint::SafeTensors;
use crate::error::{Error, Result};

/// The learned single-query attention pool: one query cross-attends each segment's hidden
/// states — K = V, no projections — yielding one pooled row per segment. The query is staged to
/// BF16 once at load; the pool owns the head width its KV packing needs.
struct AttentionPool {
    /// [1, hidden]
    query: Matrix<bf16>,
    head_width: usize,
}

impl AttentionPool {
    fn forward(&self, states: &HiddenStates, scratch: &mut PoolScratch,
               parallelism: Option<&mut ThreadPool>) {
        let (tokens, hidden) = (states.states().shape()[0], states.states().shape()[1]);
        let segments = states.segments();
        scratch.keys.try_resize(&[tokens, hidden]).expect("invariant: sized by ensure");
        states.states().try_cast_into(&mut scratch.keys).expect("invariant: matching shapes");
        // One broadcast query row per segment, and the 0..=segments identity offsets.
        scratch.queries.try_resize(&[segments, hidden]).expect("invariant: sized by ensure");
        let query_row = self.query.as_slice();
        for segment in 0..segments {
            scratch.queries.as_mut_slice()[segment * hidden..(segment + 1) * hidden]
                .copy_from_slice(query_row);
        }
        scratch.query_offsets.clear();
        scratch.query_offsets.extend(0..=segments as TokenOffset);
        scratch.pooled.try_resize(&[segments, hidden]).expect("invariant: sized by ensure");
        scratch.key_value.try_pack_into(&scratch.keys.view(), &scratch.keys.view(),
                                        self.head_width, states.offsets(), None)
            .expect("invariant: offsets cover the batch");
        match parallelism {
            Some(pool) => scratch.key_value
                .try_attention_parallel_into(&scratch.queries, &scratch.query_offsets, None,
                                             &mut scratch.pooled, pool)
                .expect("invariant: packed KV matches the query grid"),
            None => scratch.key_value
                .try_attention_into(&scratch.queries, &scratch.query_offsets, None,
                                    &mut scratch.pooled)
                .expect("invariant: packed KV matches the query grid"),
        }
    }
}

/// The projection head `RMSNorm → intermediate → SiLU → output`, operating on the pool's
/// `[segments, hidden]` rows.
struct ProjectionHead {
    norm: RmsNorm,
    intermediate: Linear<Bf16>,
    output: Linear<Bf16>,
}

impl ProjectionHead {
    fn forward(&self, pooled: &Matrix<f32>, scratch: &mut HeadScratch, out: &mut Matrix<f32>,
               mut parallelism: Option<&mut ThreadPool>) {
        self.norm.forward(pooled, &mut scratch.gated);
        self.intermediate.project_into(&scratch.gated, &mut scratch.stage,
                                       &mut scratch.projected, parallelism.as_deref_mut());
        f32::swiglu_into::<_, Matrix<f32>, _, 2, 2, 2>(&scratch.projected, None,
                                                       &mut scratch.gated, 1.0)
            .expect("invariant: matching shapes");
        self.output.project_into(&scratch.gated, &mut scratch.stage, out, parallelism);
    }
}

/// The attention pool's buffers: BF16-staged keys, the broadcast queries, and the pooled rows —
/// the pool's product and the head's input.
struct PoolScratch {
    /// [tokens, hidden]
    keys: Matrix<bf16>,
    /// [segments, hidden]
    queries: Matrix<bf16>,
    query_offsets: Vec<TokenOffset>,
    /// [segments, hidden]
    pooled: Matrix<f32>,
    /// The packed KV blob, allocated once and refreshed in place every forward.
    key_value: AttentionKeyValueCache<bf16>,
}

/// The projection head's buffers: the `[segments, hidden]` ping/pong pair and its GEMM operand
/// staging.
struct HeadScratch {
    /// [segments, hidden]
    gated: Matrix<f32>,
    /// [segments, hidden]
    projected: Matrix<f32>,
    /// [segments, hidden]
    stage: Matrix<bf16>,
}

/// Every buffer one Embedder forward touches: the Base tower's scratch, its [`HiddenStates`]
/// product, and the pool + head tails.
pub struct EmbedderScratch<P: Precision> {
    base: Scratch<P>,
    pub(crate) states: HiddenStates,
    pool: PoolScratch,
    head: HeadScratch,
}

impl<P: Precision> EmbedderScratch<P> {
    pub fn new() -> Self {
        Self {
            base: Scratch::new(),
            states: HiddenStates::new(),
            pool: PoolScratch {
                keys: empty(),
                queries: empty(),
                query_offsets: Vec::new(),
                pooled: empty(),
                key_value: AttentionKeyValueCache::empty_in(Global),
            },
            head: HeadScratch { gated: empty(), projected: empty(), stage: empty() },
        }
    }

    /// The Base's per-token product of the last forward — both tiers of one pass are readable.
    pub fn states(&self) -> &HiddenStates {
        &self.states
    }

    fn ensure(&mut self, tokens: usize, segments: usize, hidden: usize) {
        if self.pool.keys.capacity() >= tokens * hidden
            && self.pool.queries.capacity() >= segments * hidden
        {
            return;
        }
        self.pool.keys = grow(&[tokens, hidden]);
        self.pool.queries = grow(&[segments, hidden]);
        self.pool.pooled = grow(&[segments, hidden]);
        self.head.gated = grow(&[segments, hidden]);
        self.head.projected = grow(&[segments, hidden]);
        self.head.stage = grow(&[segments, hidden]);
    }
}

impl<P: Precision> Default for EmbedderScratch<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// The Embedder's product: one `[segments, hidden]` embedding matrix per forward. Grows itself
/// as batches demand — construct with [`Embeddings::new`], or pre-reserve with
/// [`Embeddings::with_capacity`].
pub struct Embeddings {
    /// [segments, hidden]
    vectors: Matrix<f32>,
}

impl Embeddings {
    pub fn new() -> Self {
        Self { vectors: empty() }
    }

    /// Pre-reserve for a known batch footprint — purely an optimization hint.
    pub fn with_capacity(max_segments: usize, hidden: usize) -> Self {
        Self { vectors: grow(&[max_segments, hidden]) }
    }

    /// One embedding row per document of the last forward.
    pub fn vectors(&self) -> MatrixView<'_, f32> {
        self.vectors.view()
    }

    /// The document count of the last forward.
    pub fn segments(&self) -> usize {
        self.vectors.shape()[0]
    }
}

impl Default for Embeddings {
    fn default() -> Self {
        Self::new()
    }
}

/// The embedding product: a [`BaseModel`] plus the attention pool and projection head its
/// contrastive training added. Weights are `Sync`; the caller brings an [`EmbedderScratch`] and
/// a reusable [`Embeddings`].
pub struct EmbedderModel<P: Precision> {
    base: BaseModel<P>,
    pool: AttentionPool,
    head: ProjectionHead,
}

impl<P: Precision> EmbedderModel<P> {
    /// Load from an opened checkpoint. Documents longer than `max_positions` bytes are
    /// truncated to it on every forward.
    ///
    /// # Errors
    /// [`Error::NotAnEmbedder`] when the pool + head tiers are absent — a Base-only checkpoint
    /// loads as a [`BaseModel`] instead — plus everything [`BaseModel::load`] returns.
    pub fn load(weights: &SafeTensors, max_positions: usize) -> Result<Self> {
        let query = weights.view("model.pool_query")
            .and_then(|payload| payload.try_decode::<bf16>())
            .ok_or_else(|| Error::NotAnEmbedder("model.pool_query".into()))?;
        if !weights.contains("model.head.fc1") {
            return Err(Error::NotAnEmbedder("model.head.fc1".into()));
        }
        let base = BaseModel::load(weights, max_positions)?;
        let head = ProjectionHead {
            norm: RmsNorm::load(weights, "model.head.norm", base.rms_epsilon())?,
            intermediate: Linear::load(weights, "model.head.fc1")?,
            output: Linear::load(weights, "model.head.fc2")?,
        };
        let pool = AttentionPool { query, head_width: base.head_width() };
        Ok(Self { base, pool, head })
    }

    /// Convenience: open the checkpoint file and load.
    ///
    /// # Errors
    /// Everything [`Self::load`] returns, plus [`Error::Io`] / [`Error::MalformedHeader`] from
    /// the file itself.
    pub fn open(path: &Path, max_positions: usize) -> Result<Self> {
        Self::load(&SafeTensors::open(path)?, max_positions)
    }

    /// The Base tower this embedder is built on.
    pub fn base(&self) -> &BaseModel<P> {
        &self.base
    }

    /// One packed forward over a batch of documents: the Base tower into `scratch.states()`,
    /// the attention pool over each segment, and the head into `embeddings` — one vector per
    /// document, growing `embeddings` as needed. Documents longer than `max_positions` bytes
    /// are truncated to it. Batch enough rows to keep the AMX tiles busy.
    pub fn forward<Document: AsRef<[u8]>>(&self, scratch: &mut EmbedderScratch<P>,
                                          embeddings: &mut Embeddings, documents: &[Document],
                                          mut parallelism: Option<&mut ThreadPool>) {
        let segments = documents.len();
        let hidden = self.base.hidden();
        let tokens = self.base.token_count(documents);
        scratch.ensure(tokens, segments, hidden);
        self.base.forward(&mut scratch.base, &mut scratch.states, documents,
                          parallelism.as_deref_mut());
        self.pool.forward(&scratch.states, &mut scratch.pool, parallelism.as_deref_mut());
        if embeddings.vectors.capacity() < segments * hidden {
            embeddings.vectors = grow(&[segments, hidden]);
        }
        embeddings.vectors.try_resize(&[segments, hidden]).expect("invariant: grown above");
        self.head.forward(&scratch.pool.pooled, &mut scratch.head, &mut embeddings.vectors,
                          parallelism);
    }
}
