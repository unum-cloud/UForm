<h1 align="center">UForm</h1>
<h3 align="center">
Pocket-Sized Multimodal AI<br/>
For Content Understanding and Generation<br/>
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/jsMURnSFM2"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/discord.svg" alt="Discord"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://www.linkedin.com/company/unum-cloud/"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/linkedin.svg" alt="LinkedIn"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://twitter.com/unum_cloud"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/twitter.svg" alt="Twitter"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://unum.cloud/post"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/blog.svg" alt="Blog"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://github.com/unum-cloud/UForm"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/github.svg" alt="GitHub"></a>
</p>

<p align="center">
Multimodal Perception from 64 to 1024 Dimensions
<br/>
Text • Images • Videos • Audio • Code • Graphs
<br/>
CUDA Training • CPU, NVIDIA and AMD Inference • WebAssembly
<br/>
<a href="https://github.com/unum-cloud/UForm/tree/main/c/uform">C</a>
 • 
<a href="https://github.com/unum-cloud/UForm/blob/main/python/README.md">Python</a>
 • 
<a href="https://github.com/unum-cloud/UForm/blob/main/rust/README.md">Rust</a>
 • 
<a href="https://github.com/unum-cloud/UForm/blob/main/javascript/README.md">JavaScript</a>
 • 
<a href="https://github.com/unum-cloud/UForm/blob/main/swift/README.md">Swift</a>
</p>

---

UForm 4 is a family of neural networks trained from scratch to learn __shared multi-modal representations__ across text, images, audio, video, and code on a GPU, a single DGX node, or an entire cluster, designed bottom-up from scratch, to fully leverage modern hardware.
A few convictions shape the whole stack:

- __Communication buys batch size buys quality__ — custom lower-level cross-GPU collectives unlock far larger global batches, and with contrastive objectives more in-batch negatives directly sharpen the learned space.
- __The data path is half the model__ — for IO-heavy modalities like images and video, the parallel CPU/GPU decode-and-load scaffolding is as worth optimizing as the forward and backward passes.
- __Lower precision wins twice__ — narrow number formats drove most of the last decade's FLOP/s while moving fewer bytes: more tensor-core throughput _and_ less traffic across the memory wall, from GPU SRAM/HBM to CPU L1/L2.
  UForm trains in mixed BFloat16/Float32 and serves scale-free E4M3 Float8 from one `quantization` argument at load — quantization built in, not bolted on — for pocket-sized weights and 2.5× faster Float8 matmuls on Blackwell.
- __A shared uniform encoder across modalities__ — attention-pooling the output of a single same-width encoder into one pooled embedding is the most direct route to emergent cross-modal transfer.
- __Non-autoregressive generation__ — emitting the entire output in one forward pass instead of token-by-token pushes single-GPU throughput up by one or two orders of magnitude for tasks that can be handled with such models — from ≈100 tokens/s to up to ≈10'000.

This repository is the full training and inference stack, tuned for NVIDIA Hopper and Blackwell GPUs: custom CUDA kernels, a C++ trainer, and GPU-accelerated loaders & decoders for IO-heavy modalities.
Neither training nor inference carries a __PyTorch or JAX dependency__.
There is not even NCCL or NVSHMEM, no dynamic memory allocations on the hot paths, locks, mutexes, or even need for multiple threads.

## Architecture & Goals

Every UForm 4 model is a __uniform byte encoder__: bytes in, same-width transformer blocks, pooled embedding out.
The encoder stage consists of N transformer blocks, all at one fixed width — there is no sequence narrowing between them.
Each block contains RMSNorm, QKV attention, and a SwiGLU MLP, with rotary position information supplied by a multi-axis M-RoPE so the same block serves one-dimensional text, two-dimensional image, and three-dimensional video coordinates.
An __attention-pooled embedding__ is read from the encoder output for retrieval and cross-modal matching; the __reranker__ and __MLM__ heads read the per-byte encoder output directly.
The subsections below trace it end to end.

- [Coding-Rate & Lipschitz Regularization](#coding-rate--lipschitz-regularization) shows how the pooled embedding is shaped into a high-rank, well-spread code;
- [Encoders Strike Back](#encoders-strike-back) argues why one forward pass beats autoregression on bounded tasks;
- [Tokenization-Free](#tokenization-free) covers the byte-level inputs that need no tokenizer; and
- [Matryoshka-like Multi-scale Embedders](#matryoshka-like-multi-scale-embedders) with [ColBERT-style Late-interaction Rerankers](#colbert-style-late-interaction-rerankers) are the two heads that read the encoder output.
- Every tier is trained as one weight-shared family; the scales, recipes, and warm-start lineage live under [Pretrained Models & Recipes](#pretrained-models--recipes).

### Coding-Rate & Lipschitz Regularization

The uniform encoder processes all bytes at full width, then an attention-pool distills the per-byte output into the pooled embedding where retrieval and cross-modal matching happen.
The pooled embedding is the deployment surface — every retrieval query, image-text match, and reranker score runs through one fixed-length vector — so its _information density_ and _dimensional decorrelation_ together set retrieval quality.
Neither follows from masked-modelling cross-entropy alone: under pure reconstruction pressure the pool collapses rank, concentrating its energy in a handful of directions.
Two opt-in auxiliary regularizers shape that pooled code's spectrum from both ends.
The trainer's configuration declares both as `tcr_lambda` and `lipschitz_lambda`, zero by default, and their kernels are not yet wired into the step.

__Total Coding Rate__ (TCR), from the [MCR²](https://arxiv.org/abs/2006.08558) work, raises the floor — the coding rate of an L2-normalized batch of pooled rows $Z$:

$$
R(Z) = \tfrac{1}{2} \log\det\!\left(I_D + \tfrac{D}{N \varepsilon^2}\, Z^\top Z\right)
$$

Maximizing $R(Z)$ sums the log-eigenvalues of the Gram, so the gradient pushes energy _into_ every under-used direction — a soft, scale-controlled effective-rank maximization.
It costs one hyperparameter $\varepsilon$ and a single forward pass, and is modality-agnostic: it sees only a $[B, T_\text{pool}, D]$ tensor, so the same kernel serves text, image, video, and audio.
Aggregating the rate over the full batch rather than local slices sharpens cross-lingual retrieval markedly, at essentially no cost to the masked-modelling objective.

A __Lipschitz__ spectral-norm penalty lowers the ceiling — it bounds the largest singular value of the same batch, estimated by a few steps of power iteration on the Gram:

$$
L_\text{Lip} = \lambda\,\sigma_\max^2(Z) = \lambda\,\lambda_\max\!\left(Z^\top Z\right), \qquad \lambda \approx 0.001
$$

Where TCR lifts the whole eigenspectrum, the Lipschitz term pulls down its top, so no single high-gain axis dominates and the code stays smooth and well-conditioned.
Unlike global TCR it is per-rank-local — the spectral norm of a rank-local Gram is itself rank-local — so it needs no cross-GPU collective.
Together the two roughly quadruple the pooled embedding's active subspace at the Tiny tier, and the gain carries through to downstream retrieval.

### Encoders Strike Back

In the age of autoregressive generative models, it is easy to forget that a large class of "generative" tasks are not open-ended generation at all.
They are structured transductions of a bounded input — at most a 4096-element window — that can be produced in a single forward pass.
The one-pass byte encoder does exactly that, at encoder throughput rather than token-by-token decode speed.
On tasks where the output is bounded and largely determined by the input, that one-pass transduction stays competitive in accuracy while running one to two orders of magnitude faster.

Robotics is the sharpest case for it.
A control loop cannot wait on token-by-token decoding, and an autoregressive policy's latency grows with the length of the action it emits.
A non-autoregressive head produces an entire action chunk in one forward pass, at fixed latency budget — fast enough to close the loop, small enough to run on the robot itself.

The same property pays off anywhere the output is bounded and mostly determined by its input:

- __Data Anonymization__ — redacting secrets and PII from in-flight messages before they are indexed, logged, shared, or used for training; the output is the input with sensitive spans rewritten.
- __Logs Streams Normalization__ — folding unstructured packets into a shared `{time}{source}{priority}{message}` schema, supporting template induction, alert deduplication, and correlation; a schema-fill, not free generation.
- __Code Documentation Extraction__ – bringing complex multi-language docstrings to one clean Web-friendly documentation form.

Similar story with media modalities:

- __De-identification__ – redacting faces, license plates, and on-screen PII from images and video before they are shared, indexed, or used for training; the output is the input with sensitive regions masked — the visual analog of a rewritten span.
- __Semantic Segmentation__ – labeling every patch with the object or region it belongs to, folding a frame into a dense class map; a per-pixel schema-fill, not free generation.

All of this rides on one shared backbone and the same GPU-accelerated scaffolding that already powers the scalar-output __rerankers__ and __embedders__ — and everything in between, such as a page-level __classifier__ for memory forensics that labels 4096-byte process dumps by role (code vs data, stack vs heap, kernel vs user).
One model family, one shared encoder, one set of kernels — from a single similarity score to a full structured rewrite.

### Tokenization-Free

All models are __tokenization-free__:

- No Byte-Pair Encoding (BPE), SentencePiece, or WordPiece for text, just bytes!
- No VQ tokenization (VQ-VAE, MAGVIT) for images and videos, just patch and tube projections!
- No neural audio codec (EnCodec, SoundStream, residual-VQ) for audio, just raw waveform frames — the spectral frontend is learned inside the encoder!

There is no tokenizer drift across languages or model versions, no out-of-vocabulary tokens, and no vocabulary-update migration story — a checkpoint is bytes-in and bytes-out at deployment time, and the learned byte embedding and uniform encoder cluster bytes into effective subwords with no fixed pooling rule.

This sits in a small tokenization-free lineage, but at an unusual intersection.
The byte-level thread runs from [CANINE](https://arxiv.org/abs/2103.06874) and [ByT5](https://arxiv.org/abs/2105.13626) to [MEGABYTE](https://arxiv.org/abs/2305.07185) and Meta's [Byte Latent Transformer](https://arxiv.org/abs/2412.09871), which matches subword models at 8B-parameter scale by patching bytes dynamically.
Where those patch by a fixed or entropy-driven rule, UForm's learned byte embedding and uniform encoder _are_ the tokenizer — attention clusters bytes into effective subwords with no fixed pooling rule, the same instinct as [Charformer](https://arxiv.org/abs/2106.12672)'s gradient-based subword blocks.
And because the pooled embedding is a small modality-agnostic latent — the design [Perceiver IO](https://arxiv.org/abs/2107.14795) validated on raw bytes, pixels, and audio — a single backbone ingests every modality with no per-modality codec.

### Matryoshka-like Multi-scale Embedders

Dense vector embeddings are the workhorse of modern retrieval.
Since __[Matryoshka](https://arxiv.org/abs/2205.13147)__ Representation Learning paper it has been standard practice to train _truncatable_ embeddings, where the leading dimensions already form a usable vector and each later block only sharpens it.
That enables adaptive, high-throughput search: the coarse phases of an index traversal compare only the first 64 dimensions to separate unrelated content, then widen to 128, 256, 512, and the full 1024 dimensions as a query closes in on a relevant cluster — spending precision only where it changes the ranking.
Every tier trains the nested prefixes of {64, 128, 256, 512, 1024} that fit its width plus the full width, so {64, 128, 256, 512} for Tiny and {64, …, 1024, 2048} for Large, but under a __[SigLIP](https://arxiv.org/abs/2303.15343) sigmoid loss__ rather than the Matryoshka InfoNCE of the original work.

There is little architectural novelty here — the difficulty is entirely in the implementation.
A contrastive objective lives at the scale of a single batch.
Write $s_{pj}$ for the cosine similarity between anchor $p$ and candidate $j$ in a batch of $N$ pairs, $t$ for the temperature and $b$ for a bias.
Classic CLIP-style InfoNCE normalizes each anchor over the whole batch with a softmax,

$$
L_\text{InfoNCE} = -\log \frac{\exp(t \cdot s_{pp})}{\sum_{j} \exp(t \cdot s_{pj})},
$$

so every other column $j \neq p$ competes as a negative in one shared denominator, and when the similarities coincide the loss pins to its uniform attractor $\log N$ — the collapse the model falls into on a small batch.
SigLIP instead scores each cell of the $[N, N]$ matrix as an independent sigmoid classification,

$$
L_\text{SigLIP} = -\log \sigma(t \cdot s_{pp} + b) - \sum_{j \neq p} \log\!\left(1 - \sigma(t \cdot s_{pj} + b)\right),
$$

with no cross-column normalization, so an unlabeled positive sitting among the negatives costs one bounded $-\log\!\left(1 - \sigma\right)$ term instead of stealing mass from the true pair in a shared denominator.
The more rows that fit, the more negatives each pair is scored against and the sharper the geometry — batch size _is_ the objective's resolution.

So UForm computes one SigLIP loss across every GPU in the NVLink domain — all-gathering the per-rank embedding towers into a single global similarity band and reduce-scattering the gradients back, with no NCCL, no host staging, and not even CPU-side scheduling, fully relying on `multimem` PTX instructions.
Each rank contributes up to 1'024 anchor–positive pairs, so a 72-GPU NVL72 domain forms a global contrastive batch of up to 73'728 Small 1024-d BFloat16 embeddings spread across its ≈13.5 TB of pooled HBM3e — _roughly 2.25× the 32'768-sample batch OpenAI used to train the original CLIP_.
As the batch grows into the tens of thousands, false negatives become statistically unavoidable, and SigLIP's per-cell independence is what keeps the signal sharp instead of corrupted at that scale.

Two loader-side properties keep the step cheap.
The text towers pack many short documents into a single 4096-byte row and pool one embedding per packed segment, so a large contrastive batch of short inputs never balloons the step's sequence budget — the GPU stays saturated whether the corpus is sentences or full passages.
And each entry can carry _several positives and explicit hard negatives_, which the sharded loss gathers as extra band columns, converging faster than a pipeline restricted to one positive and in-batch negatives alone.

### ColBERT-style Late-interaction Rerankers

A reranker shares almost everything with the embedder above — still a sigmoid cross-entropy objective, still one loss sharded across the whole NVLink domain — but it scores documents differently.
Instead of pooling a passage into a single attention-pooled embedding, it reads the _full encoder output_ — one embedding per byte, refined through a per-byte projection head — and scores a query against a candidate with a [ColBERT](https://arxiv.org/abs/2004.12832)-style late-interaction MaxSim.
Keeping per-byte geometry preserves far more of a passage's meaning than collapsing it into one dense embedding, which is exactly the granularity a reranker is asked for in the second stage of a retrieval pipeline.

Writing $q_t$ for a query byte past its prefix and $d_s$ for a candidate byte, both unit vectors from the projection head, the document score is the hard MaxSim: each query byte keeps its single best candidate byte, averaged over the query,

$$
S(q, d) = \frac{1}{|q|} \sum_{t} \max_{s} \langle q_t, d_s \rangle,
$$

the very score a MaxSim index computes at serving, with each query byte's gradient flowing through its best match alone.
These $S(q, d)$ scores fill the same $[N, N]$ band the embedder builds, and by default each query's row is scored by the ranked sigmoid cross-entropy of every positive's score over every negative's, where the bias cancels, so a query's own score scale never shifts its threshold; the balanced and SigLIP forms stay selectable.
The reranker thus inherits the large-batch and false-negative-robustness story wholesale; only the per-cell score grows from one cosine into a full late interaction, and the all-gathered tower grows from one vector per document to one per byte.
Because the encoder weights are shared with the byte-MLM head, the reranker warm-starts from the MLM base with an encoder that already reconstructs masked bytes, then fine-tunes for per-byte contrastive scoring, with explicit hard negatives gated in per example.

The cost is storage.
A 4'000-byte passage becomes 4'000 per-byte vectors — at the Tiny tier's narrow 128-d reranker-head width that is ≈1 MB raw in BFloat16, half that in Float8, far too much to keep uncompressed at corpus scale.
A production late-interaction index therefore quantizes the per-byte vectors into cluster centroids plus residual codes, the way PLAID compresses ColBERT — a serving-layer decision the checkpoint deliberately leaves open rather than baking into the model.

## Training & Data Pipelines

UForm not only publicly lists all of the data mixtures and code used for training, which is rare by itself, but also provides a PyTorch- and JAX-free training codebase, GPU-accelerated end-to-end, with orders of magnitude less complexity and overhead than traditional frameworks.
The pipeline follows the convictions up top:

- [Mixed Precision by Design](#) — BF16 training, E4M3 inference, one master copy per weight.
- [GPU-Accelerated Data-loaders & IO](#) — bytes to batch on the GPU, decoding there wherever the codec allows.
- [NVLink-based Cross-GPU Collaboration](#) — cross-rank collectives inside the captured graph, no NCCL.
- [Optimizing via Adafactor](#) — magnitude-aware updates at byte-level scale.

### GPU-Accelerated Data-loaders & IO

The defining property of the UForm training stack is that __nothing should leave the GPU between fetch and forward__.
Parquet corpora are read into device memory by cuDF over GPUDirect Storage as Apache Arrow-compatible columns, then byte-tokenized, gathered and packed on-device by the C++ loaders, so text takes no host-device copy in the loop.
The binary-column path sustains 2–3 GB/s of media bytes at \~5 ms/batch, comfortably ahead of what a training step consumes, so the prefetcher rarely stalls.
This easily supersedes single-process PyTorch `DataLoader`s by \~25× even for text processing, \~25× even for images, even without augmentations, and by a criminal number for videos.

Each modality goes from raw bytes to a masked batch on-device, and every item — a caption, an image, a clip — becomes a variable-width __segment__ packed onto one shared `[rows, columns]` grid and tagged by modality, so a single uniform encoder ingests them all:

- __Tokens__ — byte-tokenized, bucket-stratified, packed, and span-masked in CUDA; paired sentences average a few hundred bytes, so packing many into one 4 KB row keeps the batch dense without wasting context on padding.
- __Images__ — JPEG decoded on the GPU by nvJPEG, PNG and WebP on host threads, into __4-channel RGBA__ (alpha is 255 where the codec carries none), then cropped and resized by one NPP batch, __2D patchified__ and MAE-masked.
- __Video__ — decoded by NVDEC, NV12 → planar RGB, then __3D tube-extracted__ and tube-masked into a `[3, T, H, W]` tensor.
- __Audio__ — decoded from WAV, FLAC, MP3 or Opus by a pool of libavcodec workers, then chunked into __raw waveform frames__ on the GPU — the mel / STFT frontend is a learnable layer of the encoder, not a fixed loader transform — and masked like the rest.
- __Code__ – ...

First-party NVIDIA libraries do the decode and IO:

- __[nvJPEG](https://developer.nvidia.com/nvjpeg)__ — batched on-GPU JPEG decode, on the hardware decode engine where the GPU has one.
- __[NPP](https://developer.nvidia.com/npp)__ — batched crop and resize of decoded images into the batch's patch grid.
- __[NVDEC](https://developer.nvidia.com/video-codec-sdk)__ — on-GPU video decode, NV12 → planar RGB.
- __[cuFile](https://developer.nvidia.com/gpudirect-storage)__ — streams compressed bytes from disk straight to device memory on GDS-capable mounts, with a buffered fallback.
- __[cuDF](https://github.com/rapidsai/cudf)__ — GPU-side Parquet reads through its KvikIO layer, on explicit per-loader streams.

And the host-side C++ libraries that carry bytes up to the device line:

- __[NumKong](https://github.com/ashvardanian/NumKong)__ — the `nk::` tensor and span types every host- and device-side buffer is shaped against.
- __[ForkUnion](https://github.com/ashvardanian/ForkUnion)__ — a lock-free CPU thread pool that fans out host-side decode and file IO without `std::thread`.
- __[StringZilla](https://github.com/ashvardanian/StringZilla)__ — SIMD byte-string ops and the Arrow strings tape that holds raw records.
- __[SmashTable](https://github.com/ashvardanian/SmashTable)__ — the vectors and hash tables every host-side table grows in.
- __[yyjson](https://github.com/ibireme/yyjson)__ — the exception-free JSON parser behind JSONL dataset records and SafeTensors headers.

### NVLink-based Cross-GPU Collaboration

Every training step is a single captured CUDA Graph, replayed each iteration and driven from one host thread across every GPU in one process.
No variable-size dynamic allocations on the hot path, no shared pointers, no `std::async` or `std::thread` to control GPUs: the calling thread owns every `cudaGraph` and `cudaStream_t`, host-side callbacks are avoided, and everything else is shifted into fused on-GPU operations.

Cross-rank collectives live _inside_ that captured graph, written directly in `multimem.*` PTX.
An all-reduce is two instructions: `multimem.ld_reduce` reads a vector and sums it across every peer in the NVLink fabric in a single op, and `multimem.st` writes the result back to every peer in a single op — the switch itself performs the reduction, so there is no ring, no tree, and no communicator object.
Ranks synchronize with an in-kernel arrive-and-wait barrier — a `multimem.red.add` on a multicast-shared counter carried on a monotonic epoch, so a graph replay never races a stale reset — and the vectorized paths cover BFloat16 (`v4.bf16x2`) and Float32 (`v4.f32`).

The whole forward, backward, gradient all-reduce, and optimizer step is therefore one graph the host launches once and replays — no phase decomposition, no host barrier, no CPU-side scheduling between collectives.
We call this path the __Multicast World__; it needs Compute Capability 9.0+ with NVLink multicast (Hopper or Blackwell) and is validated against single-rank references across world sizes 2 and 4.
Where multicast is absent, every member reads its peers in a fixed order instead, with bit-identical results.
Across NVLink domains, device-initiated InfiniBand writes on a recursive-doubling schedule all-gather the contrastive embeddings and all-reduce the gradients from inside the same captured graph.
The [FAQ](#why-hand-roll-ptx-instead-of-nccl-and-nvshmem) covers why this is hand-rolled rather than delegated to NCCL or NVSHMEM.

### Mixed Precision by Design

__Training is BF16.__
Weights and activations are BFloat16 throughout — the forward and backward GEMMs (FP32-accumulated on the tensor cores), RMSNorm, attention, SwiGLU, embedding lookups, and the loss kernels.
The optimizer is Adafactor with a first moment: a BF16 master weight, a Float32 first moment of the clipped update with Adam's bias correction, and a Float32 factored second moment — one per-row and one per-col vector per matrix rather than a full variance.
Most norm scales are the exception, keeping a Float32 master with a BF16 mirror: initialized at magnitude 1.0, their updates fall below the BF16 ULP of \~7.8e-3 and would round to zero in pure BF16.

Float32 is not rare — it lives wherever values accumulate or reduce.
Gradients accumulate in Float32 before being cast to BF16 for the optimizer step, and the Adafactor moments, the attention outputs and log-sum-exps, the RMSNorm inverse-variances, the TCR and Lipschitz Gram matrices, and the sharded-loss partial-gradient slabs are all Float32.
The rule is simple: store and compute in BF16, accumulate and normalize in Float32.

Activation recomputation is always on: each block keeps only its Float32 input, and its backward replays the forward from that input to refill one set of stashes every block shares.
Activation memory is one block's intermediates plus one residual stream per block, bought with one extra forward of every block but the top one.

__Inference is BF16 or E4M3 (FP8), fixed per model at load time.__
The compute mode is the one `quantization` scalar a load names — `bf16` as the checkpoint stores it, or `e4m3` quantized at load — not a per-layer toggle; an E4M3 model runs its transformer blocks in Float8, while the entries, the attention pool and the projection head stay BFloat16.
Each load packs its weights for one CPU or GPU class, the hardware's own or one the caller pins, so a single binary carries tuned variants from Haswell to Diamond Rapids on x86, from Graviton2 to Vera on Arm, and from Ampere to Blackwell and CDNA3 to CDNA5 on GPUs.
[NumKong](https://github.com/ashvardanian/NumKong) runs every matrix product and every attention, forward and backward — BF16 and E4M3 alike, against weights packed once per step — and [CUB](https://github.com/NVIDIA/cccl) supplies the device-wide scans and reductions inside the custom kernels.

### Optimizing via Adafactor (the Sole Optimizer)

UForm trains exclusively with a custom CUDA [Adafactor](https://arxiv.org/abs/1804.04235) implementation in [`optimizer.hpp`](include/uform/core/optimizer.hpp).
Neither [Lion](https://arxiv.org/abs/2302.06675) nor [AdamW](https://arxiv.org/abs/1711.05101) ships in the stack; both appear below only as ablation baselines.
Lion is hyperparameter-economical — two betas, no second-moment state — but its sign-only update $\mathrm{sign}(\mathrm{EMA}(g))$ discards gradient _magnitude_ and steps every parameter by the same amount.
On a unit-RMSNorm encoder that is fatal: the variance-destroying low-rank direction costs the optimizer no more than an informative one, so the pooled embedding slides into rank collapse.
A controlled ablation reproduced exactly that — Lion drove the pooled embedding to effective rank ≈1 and pooled-embedding cosine ≈0.99 across two seeds and stalled at the unigram floor, while AdamW on identical data and init grew the rank and converged; a from-identical-init lockstep ruled out a kernel bug.

AdamW recovers magnitude with a per-parameter variance estimate, but carries two moments per parameter and adds $\beta_2$ and $\varepsilon$ to tune per modality and per schedule.
Adafactor is the middle ground: it keeps a _factored_ second moment — per-row $[N]$ plus per-col $[D]$ statistics, $O(N + D)$ instead of the full $O(N \cdot D)$ variance — so it is magnitude-aware like AdamW at about half the optimizer state, under a single learning rate that covers every tensor from the byte embedding to the 1-D norms.
In our optimizer × pooled-embedding-regularizer sweep it matched AdamW's validation loss and produced the lowest activation off-diagonal covariance of any variant — the decorrelation cosine retrieval depends on — trailing AdamW only slightly on bitext retrieval.

The byte-level precedent is established: [ByT5](https://arxiv.org/abs/2105.13626) showed Adafactor trains stably at these sequence lengths and [PaLM](https://arxiv.org/abs/2204.02311) that it scales.
The usual memory win is minor here — the 258-entry byte vocabulary makes the embedding table tiny — so the load-bearing reasons are the magnitude Lion throws away and the decorrelation AdamW only half-delivers.

## Pretrained Models & Recipes

All models share weights, not just structure, along a two-axis lineage:

- __Growing scale__ — a larger tier warm-starts from the one below it through a function-preserving Net2Net step that doubles exactly one of width or depth, from Tiny through Small, Medium and Large to XL.
  Deepening inserts identity blocks with zeroed out and down projections; widening duplicates the residual stream as `[x, x]` with the input-side columns halved.
  `uform_grow <in> <out> --deepen` or `--widen` runs either offline and refuses to write unless the grown model reproduces the source's embeddings to a cosine of 0.9999.
- __Specializing capability__ — a reranker is initialized from the masked-pretraining base, and an embedder from the reranker.

The same recipe is reused across every scale and task; only the head and the objective change.
Checkpoints are saved in the Hugging Face [SafeTensors](https://github.com/safetensors/safetensors) format, and each checkpoint directory carries a `README.md` recording the commit, hardware, config tier, modality, recipe, datasets, warm-start source, and training command.

### Uniform Byte Models

Five tiers form a ladder in which each step doubles exactly one of width or depth.
A run names its tier with `--tier` and its data with `--mixture`, and the mixture picks the row width:

```bash
uform_base --tier small --mixture scripts/mixtures/text_contrastive.json --objective contrastive
```

| Tier   | hidden | depth | params  | peak learning rate | pooled-embedding dim               | ColBERT dim |
| ------ | ------ | ----- | ------- | ------------------ | ---------------------------------- | ----------- |
| tiny   | 512    | 12    | \~51 M  | 5e-4               | 64 / 128 / 256 / 512               | 128         |
| small  | 1024   | 12    | \~204 M | 2e-4               | 64 / 128 / 256 / 512 / 1024        | 128         |
| medium | 1024   | 24    | \~405 M | 2e-4               | 64 / 128 / 256 / 512 / 1024        | 128         |
| large  | 2048   | 24    | \~1.6 B | 1e-4               | 64 / 128 / 256 / 512 / 1024 / 2048 | 128         |
| xl     | 2048   | 48    | \~3.2 B | 1e-4               | 64 / 128 / 256 / 512 / 1024 / 2048 | 128         |

Every tier takes 4096-byte rows in text-only runs and 1024-token rows in any run with media, where an image or a 10.24 s audio clip fills half a row and a video clip a full one.
At 16-pixel patches a square photo fills 506 of a half row's 512 cells as a 22×23 grid, 352 px tall and 368 px wide.

Parameters count the blocks, the byte embedding and the pooled-embedding head, without the frontends and reconstruction heads a mixture attaches.
`--learning-rate` overrides the tier's peak learning rate.

A text-only mixture uses the byte-gather frontend only; a mixture with media attaches the frontend of each modality it names (vision patch projection, video tube projection, audio waveform-frame projection) to the same uniform encoder backbone.
Every tier ships in three or more capability variants, warm-started in sequence — a Net2Net step grows each tier from the one below it, and each scale trains Base → Reranker → Embedder on top of its own Base:

- __Base__ — span-masked byte modelling on the per-byte encoder output; the pretraining root.
- __Reranker__ — ColBERT-style per-byte hard MaxSim on the per-byte encoder output, under the ranked sigmoid loss.
- __Embedder__ — attention-pooled embedding under Matryoshka SigLIP.

The Base `i18n` mixture spans a superset of the 21 languages UForm v3 shipped — English, German, Spanish, French, Italian, Russian, Japanese, Korean, Turkish, Chinese, Polish, Arabic, Armenian, Hebrew, Hindi, Indonesian, Persian, Portuguese, Thai, Ukrainian, Vietnamese — drawn from FineWeb-2's 42-language crawl and FineWiki's 29, balanced roughly:

- __\~69% web text__ — multilingual FineWeb-2, English FineWeb, multilingual FineWiki.
- __\~18% scientific__ — peS2o full text.
- __\~9% community Q\&A__ — StackExchange, Hacker News.
- __\~4% code__ — CodeSearchNet, with StarCoder / CodeContests / ComPile-IR expanding the share.

Reranker and Embedder variants then fine-tune on retrieval and QA pairs — MS MARCO, AllNLI, Mr. TyDi, Natural Questions, HotpotQA, and more.

### One Omni Model

All modalities share one uniform encoder via per-modality input frontends:

- __Tokens__ — a byte-gather frontend embeds raw UTF-8 bytes directly, with no tokenizer step.
- __Images__ — a vision patch-projection frontend converts 16×16 pixel patches into encoder-width vectors using 2D M-RoPE coordinates.
- __Video__ — a tube-projection frontend lifts 16×16×2 spatio-temporal tubes into the same width using 3D M-RoPE, with absolute time on the temporal axis so clips sampled at any frame rate share one coordinate frame.
- __Audio__ — a waveform-frame frontend projects raw sample chunks into encoder-width vectors, learning the spectral transform end to end rather than baking a fixed mel filterbank.

Each frontend is a thin adapter (patch / tube / waveform-frame projection or byte embedding lookup); the uniform encoder backbone is identical across all modalities — the same N blocks at the same width, the same M-RoPE, the same attention-pooled embedding.
This is how a run with media achieves cross-modal retrieval: image↔text, video↔text, and audio↔text pairs are scored using the same pooled-embedding space, so emergent cross-modal transfer (image↔video, audio↔text at zero-shot) follows from the shared encoder geometry.

Cross-modal retrieval and the structured-output tasks — de-identification, segmentation, log normalization, document extraction — all run over the same backbone and the same GPU-accelerated scaffolding, at encoder throughput.

### Evaluation and Calibration

`uform_eval <checkpoint>` scores a checkpoint through the C interface as Recall@1, @5 and @10 in both directions between captions and media.
`--coco` takes COCO 2017 val, `--msrvtt` the MSR-VTT 1K-A test, `--clotho` the Clotho evaluation set, and `--coco-sm` the COCO-SM captions in 21 languages.
It encodes on host threads, or in the VRAM of the GPU `--gpu` names.

A run with `--choices-weight` also trains calibrated choices through the byte decoder: a softmax over each item's legal bytes alone, scored by a soft cross-entropy and a Brier term `--brier-weight` weighs.
`uform_eval <checkpoint> --calibrate <choices.jsonl> --output <calibrated.safetensors>` fits one inverse temperature per question type on held-out choices.
It prints each group's negative log-likelihood and the top-choice calibration error before and after, and writes the checkpoint again with them as `choices.inverse_temperatures`, which `uform_text_choose` reads.

## Repository Layout

```bash
uform/
├── c/                the C ABI in uform/*.h, shared by every binding, and its test
├── include/uform/    the templates and models of every component, one folder per namespace
│   ├── core/         device memory, batches, streams, mesh, transports, model cells, checkpoints
│   ├── encoder/      the encoder towers and kits, their geometry and entries
│   ├── media/        image, video and audio decoding on the host
│   ├── loaders/      storage sessions, shard readers, and GPU image, video and audio decode
│   └── train/        the composition of a training run
├── src/              the instantiations, the dispatch and the per-class units behind the C ABI
│   ├── *.cpp         the verbs of each header: types, model, encode and train
│   ├── encoder/      one unit per CPU and GPU class, and each vendor's kernels
│   └── ...           the compiled bodies of the other include/uform/ folders
├── src_stub/         a stand-in answering the same verbs over the C++ standard library and ForkUnion
├── tools/            the trainer, evaluation and growth binaries
├── bench/            the encoder and training-kernel benchmarks
├── test/             conformance suites, one file per cell, and the binding suites in python/
├── verification/     Promela models of the mesh protocols, checked with Spin
├── scripts/
│   ├── data/         dataset acquisition, repacking and mining
│   ├── mixtures/     the dataset mixtures a run trains on
│   └── slurm/        cluster runners for training, tests and acquisition
├── python/           the CPython extension over the C ABI
├── rust/             the Rust crate over the C ABI
├── javascript/       the Node-API addon and the WebAssembly module over the C ABI
├── swift/            the Swift package over the C ABI
└── docs/             the documentation site
```

Every binding is a thin layer over the C ABI in `c/uform/`: one module per header, one method per verb, and the same receivers, executors and arenas in each language.
Each links either the engine or the stand-in of `src_stub/`, which answers every verb with deterministic rows in seconds of build time, so the binding suites run without a checkpoint.
Conventions, hooks and the commit format are described in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## FAQ

### Why TCR and Lipschitz, but not VICReg or DimNorm?

Both VICReg and DimNorm attack the _symptom_ — rank collapse — rather than the pooled embedding's information density, and both were superseded.
DimNorm normalized per-dimension statistics every forward pass to hold the pooled code off the rank-1 attractor; TCR's rank-maximization plus a fresh-encoder warm-start cover that role at the loss, so there is no separate normalization pass.
An earlier [VICReg](https://arxiv.org/abs/2105.04906) loss was dropped for a sharper reason: its variance and covariance terms act on mean-centered statistics, so they are structurally blind to the shared-mean anisotropy the multi-prefix contrastive objective induces — in ablation, adding VICReg left raw cosine at 0.9999 and retrieval at chance.
TCR and the Lipschitz bound instead shape the _raw_ second moment $Z^\top Z$ directly, and the contrastive escape term covers anti-collapse more cheaply than a dedicated variance loss.
Only VICReg's covariance kernels survive — repurposed as the effective-rank metric.

### Why attention pooling instead of mean pooling?

Mean pooling is the uniform-attention special case of an attention pool, so attention pooling is strictly more expressive at no extra asymptotic cost.
The pool query is a single learned encoder-width vector, broadcast across the batch as $Q$ with sequence length one; the per-byte encoder outputs serve as $K$ and $V$.
NumKong's ragged attention makes the expressive route as cheap as the mean, so there is one pool path and no toggle.

### Why no cuDNN, FlashAttention, xformers, or FA3?

[NumKong](https://github.com/ashvardanian/NumKong) already runs a [FlashAttention](https://arxiv.org/abs/2205.14135)-style ragged attention, forward and backward, on every GPU generation and on the CPU, with each segment and head summed in a fixed order.
One kernel library behind the products and the attention keeps the external surface small across driver and architecture bumps.
ThunderKittens, FA3, CUTLASS and TransformerEngine stay out of the stack for the same reason.

### Why hand-roll PTX instead of NCCL and NVSHMEM?

The whole step — forward, backward, all-reduce, optimizer — is one CUDA Graph captured once and replayed from a single host thread, so collectives must run _inside_ the capture.
NCCL is only partially capture-safe: a captured `ncclAllReduce` instantiates fine but silently no-ops on replay —

```cpp
cudaStreamBeginCapture(stream);
forward_backward(...);
ncclAllReduce(grad, grad, n, ncclBfloat16, ncclSum, comm, stream);
cudaGraphLaunch(exec, stream);   // replays — but grad is never reduced
```

— and the standard workaround, splitting the step so the reduce runs on the host between sub-graphs, puts a host sync back into every step, which gave intermittent NaNs and stalled the multi-collective sharded-loss path.
NVSHMEM replays correctly but bootstraps one PE per process (`nvshmem_n_pes()` returns 1 from our single-process trainer), so adopting it means re-architecting under MPI or `torchrun`.
`multimem.*` is the NVLink-fabric primitive both libraries dispatch internally anyway — capture-safe and single-process — so UForm calls it directly; the design is in [NVLink-based Cross-GPU Collaboration](#nvlink-based-cross-gpu-collaboration).
