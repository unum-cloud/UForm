/**
 *  @brief The encoder tower: everything between a @ref multimodal_batch and the pooled contrastive
 *      vertices — the per-modality entry (token embedding or image / video / audio patch projection),
 *      the transformer block stack, the final norm, the attention pool, and the projection head. Two
 *      function objects: @ref encoder_tower_bf16_t (training, forward + backward) and its quantized
 *      forward-only twin @ref encoder_tower_e4m3_t , constructed only through
 *      @ref encoder_tower_bf16_t::quantize_weights .
 *  @file cuda_core/encoder_tower.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  The op sequence, one packed batch end to end:
 *
 *      batch → entry (per modality) → [transformer_block ×depth] → RMSNorm → attention pool per segment
 *            → head RMSNorm → fc1 → SiLU → fc2 → pooled vertices [segments,hidden]
 *
 *  @section tower_modes The two objectives share one forward
 *
 *  The tower serves both training objectives, selected by which spans the caller populates — never a
 *  mode flag. A @b contrastive step passes a @b [segments,hidden] @c vertices span: the full sequence
 *  runs and the loss scores the pooled vertices. A @b masked step passes an empty @c vertices span:
 *  the tower stops after the final norm and exposes the per-token states through @ref normalized , which
 *  the masked head reads. @ref backward mirrors this: @c vertices_grad drives the contrastive path,
 *  @c hidden_grad the masked path, and when both are present the masked force is added onto the pooled
 *  force at the final-norm states so a co-trained step carries @c dL_pooled + dL_tokens exactly.
 *
 *  @section tower_entry Per-segment entry over the packed grid
 *
 *  Rows mix modalities and pack several items, so the entry walks the live segments instead of slicing
 *  modality bands: @ref fill_coords writes every cell's @b (t,h,w) in one pass, then the packed
 *  @b [rows·columns,hidden] state is zeroed and built modality by modality. Text segments gather the
 *  embedding table into their cells; each media modality memsets its patch scratch, a segment-driven
 *  kernel normalises only that modality's cells in place (a cell's content lives at the same flat grid
 *  position as its state, so no gather offset is needed — image / video pixels by per-modality mean / std,
 *  audio raw waveform frames as-is), and one @ref cublaslt_plan_t projection @b accumulates ( @c β=1 ) into the
 *  state — its zero non-modality rows leave every other cell untouched, so the four writes compose with no
 *  scatter and no atomics. The step processes the full @c max_tokens capacity; PAD and inter-segment gap
 *  cells stay zero.
 *
 *  @section tower_determinism Determinism, including the embedding scatter
 *
 *  Every reduction is fixed-order; the one classically non-deterministic op — the embedding-table
 *  gradient scatter over duplicate tokens — is made deterministic by sorting the step's (token, position)
 *  pairs with a radix sort at a pinned workspace and accumulating each run serially. No atomics anywhere.
 *
 *  @section tower_precision The quantized twin's precision split
 *
 *  @ref encoder_tower_e4m3_t runs the block stack in E4M3 (the dominant cost) while the tiny entries,
 *  attention pool, and head stay BFloat16 — the standard mixed-precision inference split, chosen, not
 *  postponed: those tiers are a fraction of a percent of the FLOPs and the head feeds the cosine geometry
 *  directly.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cmath>   // `sqrtf`, `expf`

#include <array>  // `std::array` — the fixed-size pinned-plan tables
#include <memory> // `std::unique_ptr` — the blocks' shared scratch

#include <cuda.h> // `cuMemsetD8Async`, `cuMemcpyDtoDAsync`

#include <numkong/attention.h> // `nk_attention_pack_bf16_best`, `nk_attention_bidirectional_packed_bf16_best`
#include <numkong/cast.h>      // `nk_cast_best` — the pooled Float32 rows into BFloat16
#include <numkong/tensor.hpp>  // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "cuda_core/types.hpp"             // `expected`, `expected_t`, `status_t`, `differentiable_weights_t`
#include "cuda_core/multimodal_batch.hpp"  // `multimodal_batch_on_gpu_t`, `multimodal_batch_shape_t`, `modality_t`
#include "cuda_core/transformer_block.cuh" // the block stack, `rope_angles_t`, shared `rmsnorm_*`

namespace unum::uform::core {

/**
 *  @brief The baked geometry of one tower — a different config is a different object. Media modalities
 *      are opt-in: a modality whose patch geometry is zero has no entry, projection, or plan.
 */
struct encoder_tower_config_t {
    /** @brief Token-embedding rows; 0 ⇒ the tower carries no token entry. */
    u32_t vocabulary = 0;
    /** @brief Transformer blocks stacked. */
    u32_t depth = 1;
    /** @brief The block geometry every layer shares (fed from the batch shape at @ref make ). */
    transformer_block_config_t block {};

    /** @brief Image patch extents in pixels; either 0 ⇒ no image entry. */
    u32_t image_patch_height = 0, image_patch_width = 0;
    /** @brief The channel count every image pixel carries — RGBA end to end, the innermost projection rank. */
    u32_t image_channels = 4;
    /** @brief Video tube extents (temporal × spatial); any 0 ⇒ no video entry. */
    u32_t video_tube_frames = 0, video_patch_height = 0, video_patch_width = 0;
    /** @brief The channel count every video pixel carries — independent of image, the innermost content rank. */
    u32_t video_channels = 3;
    /** @brief Raw samples per audio frame; 0 ⇒ no audio entry. Audio is 1-D — a frame is a chunk of raw
     *      waveform, not a patch — so this is the learnable-frontend projection fan-in directly. The mel /
     *      spectral transform is that projection (and the blocks above it), learned, not a fixed loader step. */
    u32_t audio_chunk_samples = 0;

    /** @brief Per-channel image pixel mean, ImageNet RGB plus a neutral alpha — the entry normalises
     *      @c (u8/255 − mean)·(1/std) over @ref image_channels channels. */
    f32_t image_pixel_mean[4] = {0.485f, 0.456f, 0.406f, 0.0f};
    /** @brief Per-channel image pixel standard deviation, ImageNet RGB plus a unit alpha. */
    f32_t image_pixel_std[4] = {0.229f, 0.224f, 0.225f, 1.0f};
    /** @brief Per-channel video pixel mean, ImageNet RGB over @ref video_channels channels. */
    f32_t video_pixel_mean[3] = {0.485f, 0.456f, 0.406f};
    /** @brief Per-channel video pixel standard deviation, ImageNet RGB. */
    f32_t video_pixel_std[3] = {0.229f, 0.224f, 0.225f};

    /** @brief The rotary axis split; the tower owns its @ref rope_angles_t from it. */
    rope_shape_t rope {};
    /** @brief Longest position the angle tables cover. */
    u32_t rope_max_positions = 0;
    /** @brief The rotary frequency base. */
    f32_t rope_base = 10000.0f;
};

#if defined(__CUDACC__)

/**
 *  @brief The attention pool's forward, shared by both towers: one learned query attends each segment's
 *      live states as both keys and values, through NumKong's ragged attention.
 */
struct tower_pool_t {

    /** @brief Allocate the pool off @p arena for @p block 's capacities: the query offsets
     *      @c {0,1,…,segments} that give every segment the one query, the K=V pack sized for every segment at
     *      @c max_segment_tokens , and the Float32 pooled rows. Uploads the offsets — outside any capture. */
    [[nodiscard]] static expected<tower_pool_t> make(gpu_arena_t &arena,
                                                     transformer_block_config_t const &block) noexcept;

    /**
     *  @brief Pool every segment of @p states — pure enqueue: pack K=V, attend, cast to BFloat16.
     *  @param[in] query @b [hidden] the learned pool query.
     *  @param[in] states @b [tokens,hidden] the final-norm states.
     *  @param[in] segment_offsets @b [segments+1] flat segment starts.
     *  @param[in] segment_lengths @b [segments] live token counts, zero for PAD, which pools to zeros.
     *  @param[out] pooled @b [segments,hidden] the pooled rows.
     *  @param[inout] stream The stream every launch enqueues on.
     */
    [[nodiscard]] expected_t forward(bf16r1_view_t query, bf16r2_view_t states, u32r1_view_t segment_offsets,
                                     u32r1_view_t segment_lengths, bf16r2_span_t pooled, gpu_stream_t &stream) noexcept;

  private:
    /** @brief The NumKong capabilities of the arena's device, picking every launch. */
    nk_capability_t capabilities_ = 0;
    /** @brief @b [segments+1] the query offsets @c {0,1,…,segments} . */
    u32r1_tensor_arena_t query_offsets_ {};
    /** @brief The K=V pack, sized by NumKong's pack-size query. */
    b8r1_tensor_arena_t packed_ {};
    /** @brief @b [segments,hidden] the Float32 pooled rows before their BFloat16 cast. */
    f32r2_tensor_arena_t pooled_ {};
};

/**
 *  @brief The training tower: entry + block stack + final norm + attention pool + head, every weight,
 *      gradient, stash, and plan owned — @ref forward stashes, @ref backward consumes, both pure enqueues.
 */
struct encoder_tower_bf16_t {

    /** @brief Allocate and plan the whole tower off @p arena — token / media entries, @c depth blocks,
     *      the attention pool, and the head — validating each present modality's patch count against
     *      @p shape 's @c max_columns_per_row . */
    [[nodiscard]] static expected<encoder_tower_bf16_t> make(gpu_arena_t &arena, cublaslt_session_t &blas,
                                                             cudnn_session_t &cudnn, encoder_tower_config_t config,
                                                             multimodal_batch_shape_t shape) noexcept;

    /** @brief One tower forward over @p batch — pure enqueue; every stash the backward reads is written
     *      along the way, including the per-token rotary @b (t,h,w) coordinates @ref coords_ derives fresh
     *      from @p batch 's own geometry. Empty @p vertices ⇒ masked path (stops after the final norm;
     *      @ref normalized exposed). @b [segments,hidden] @p vertices ⇒ contrastive path (attention pool
     *      → head). */
    [[nodiscard]] expected_t forward(multimodal_batch_on_gpu_t const &batch, bf16r2_span_t vertices,
                                     gpu_stream_t &stream) noexcept;

    /** @brief Tower backward from the loss's force to every weight-gradient buffer (and the embedding
     *      table's, deterministically). Populated @p vertices_grad drives the contrastive path; populated
     *      @p hidden_grad the masked path; both present ⇒ the masked force is added onto the pooled force
     *      at the final-norm states. Reuses @ref coords_ from the paired forward — the caller must not run
     *      another forward in between. */
    [[nodiscard]] expected_t backward(bf16r2_view_t vertices_grad, bf16r2_view_t hidden_grad,
                                      multimodal_batch_on_gpu_t const &batch, gpu_stream_t &stream) noexcept;

    /** @brief Stage each present media modality's raw pixels into the @b clean per-channel-normalized patch
     *      vectors a masked-autoencoder step reconstructs against — the same @c (byte/255 − mean)·inverse_std
     *      normalization the entry projects from, written straight into the caller's @b [live_tokens,patch_scalars]
     *      @p image_patches / @p video_patches over the packed grid (non-modality and gap rows left zero), with no
     *      projection and without running @ref forward . A modality is staged only when its projection is
     *      configured, its @p batch tensor is present, and its output span is non-empty; audio is not staged here —
     *      its reconstruction target is the raw @c batch.audio_samples frames the caller reshapes directly. */
    [[nodiscard]] expected_t patchify_media_targets(multimodal_batch_on_gpu_t const &batch, bf16r2_span_t image_patches,
                                                    bf16r2_span_t video_patches, gpu_stream_t &stream) noexcept;

    /** @brief Fill @p weights with every optimizer-facing weight in the canonical reduction order:
     *      entry → media projections → per-layer block weights → final norm → pool query → head.
     *      Each entry carries the checkpoint name every SafeTensors parser keys by — @c embedding , the
     *      per-modality @c image_projection / @c video_projection / @c audio_projection (present iff
     *      configured), the per-layer @c blocks.{i}.qkv / @c blocks.{i}.out / @c blocks.{i}.gate_up /
     *      @c blocks.{i}.down / @c blocks.{i}.pre_norm / @c blocks.{i}.post_norm , then @c final_norm ,
     *      @c pool_query , @c head.norm , @c head.fc1 , and @c head.fc2 — the composition root adds only
     *      its @c model. prefix. */
    [[nodiscard]] expected_t fill(array<differentiable_weights_t> &weights) noexcept;

    /** @brief Pack every block's weights for NumKong's products — pure enqueue, capture-safe. Call once per
     *      optimizer step, after the weights change and before the first @ref forward ; every forward and
     *      backward of the step reuses the packs. */
    [[nodiscard]] expected_t pack_weights(gpu_stream_t &stream) noexcept;

    /** @brief Quantize this tower's block stack into the E4M3 inference twin (entries / pool / head stay
     *      BFloat16) — the twin's only constructor. */
    [[nodiscard]] expected<struct encoder_tower_e4m3_t> quantize_weights(gpu_arena_t &arena,
                                                                         gpu_stream_t &stream) noexcept;

    /** @brief The per-token final-norm states — the masked head's input on the masked path. */
    [[nodiscard]] bf16r2_view_t normalized() const noexcept { return normalized_.view(); }

    /** @brief The baked geometry. */
    [[nodiscard]] encoder_tower_config_t const &config() const noexcept { return config_; }
    /** @brief The block at @p layer_index . */
    [[nodiscard]] transformer_block_bf16_t const &block(std::size_t layer_index) const noexcept {
        return blocks_[layer_index];
    }

  private:
    friend struct encoder_tower_e4m3_t;

    /** @brief The pinned @b cuBLASLt plan roles, in the order @ref make pins them: the three head
     *      products, then two per present media modality (projection forward, projection weight grad)
     *      from @ref plan_media_base_k , each modality's base recorded into @c *_plan_base_ . The token
     *      entry is a kernel, not a plan. */
    enum cublaslt_plan_role_t : std::size_t {
        plan_head_forward_k = 0,
        plan_head_data_grad_k = 1,
        plan_head_weight_grad_k = 2,
        plan_media_base_k = 3,
    };
    /** @brief The @c plans_ capacity: the head roles plus two per media modality. */
    static constexpr std::size_t plan_capacity_k = plan_media_base_k + 2u * 3u;
    /** @brief Sentinel for a modality with no pinned projection plan. */
    static constexpr std::size_t plan_absent_k = ~std::size_t {0};

    /** @brief Make one pinned cuBLASLt plan (forcing a BFloat16 @c C ), record it at @p role (post-incremented
     *      into @c plans_ ), and grow @p workspace_bytes to its scratch demand — the plan-pinning step @ref make
     *      repeats for the head and every media projection. */
    [[nodiscard]] expected_t pin_plan(cublaslt_plan_config_t plan_config, std::size_t &role,
                                      std::size_t &workspace_bytes) noexcept;
    /** @brief Pin one present media modality's two plans (projection forward, projection weight grad) from
     *      @p role , recording its base into @p plan_base ; the row extent is pinned at @p max_tokens (the worst
     *      case) purely to size the heuristic + workspace, the live tile count coming from the operands. */
    [[nodiscard]] expected_t pin_media_plans(std::size_t &plan_base, u32_t fan_in, u32_t max_tokens, u32_t hidden,
                                             std::size_t &role, std::size_t &workspace_bytes) noexcept;

    /** @brief The baked geometry and the borrowed cuBLASLt session — never owned here. */
    encoder_tower_config_t config_ {};
    cublaslt_session_t *blas_ = nullptr;
    /** @brief The NumKong capabilities of the arena's device, picking the residual stream's casts and norm. */
    nk_capability_t capabilities_ = 0;

    /** @brief The transients every block shares; heap-held so the blocks' borrows survive the tower's moves. */
    std::unique_ptr<transformer_scratch_t> scratch_ {};
    /** @brief The @c depth transformer blocks, owned in order. */
    array<transformer_block_bf16_t> blocks_ {};

    /** @brief The @b [vocabulary,hidden] token embedding table and its gradient (present iff
     *      @c config.vocabulary != 0 ). */
    bf16r2_tensor_arena_t entry_weight_ {}, entry_weight_grad_ {};
    /** @brief The three media patch-projection weights and their gradients (present per configured
     *      modality): image @b [hidden, ph·pw·image_channels] , video @b [hidden, tf·ph·pw·video_channels] ,
     *      audio @b [hidden, chunk_samples] (a learnable linear frontend over one raw waveform frame) . */
    bf16r2_tensor_arena_t image_projection_ {}, image_projection_grad_ {};
    bf16r2_tensor_arena_t video_projection_ {}, video_projection_grad_ {};
    bf16r2_tensor_arena_t audio_projection_ {}, audio_projection_grad_ {};
    /** @brief The device per-channel image and video pixel mean and inverse-std, uploaded once at make
     *      ( @b [image_channels] and @b [video_channels] ); audio frames are raw waveform with no per-channel
     *      statistics, so audio uploads no mean/inverse-std. */
    f32r1_tensor_arena_t image_pixel_mean_ {}, image_pixel_inverse_std_ {};
    f32r1_tensor_arena_t video_pixel_mean_ {}, video_pixel_inverse_std_ {};
    /** @brief Recorded plan-role bases per present media modality (or @c plan_absent_k ). */
    std::size_t image_plan_base_ = plan_absent_k, video_plan_base_ = plan_absent_k, audio_plan_base_ = plan_absent_k;

    /** @brief The final norm scale and its gradient. */
    bf16r1_tensor_arena_t final_norm_scale_ {}, final_norm_scale_grad_ {};
    /** @brief The final norm scale widened to the Float32 @c γ NumKong's rmscast reads, each forward. */
    f32r1_tensor_arena_t widened_final_norm_scale_ {};

    /** @brief The learned attention-pool query @b [heads,head_dim]=[hidden] and its gradient. */
    bf16r1_tensor_arena_t pool_query_ {}, pool_query_grad_ {};
    /** @brief The attention pool's forward. The backward is a house kernel pair that recomputes the softmax
     *      and folds the exact Jacobian itself. */
    tower_pool_t pool_ {};

    /** @brief The head's learnable-scale RMSNorm @c γ and its gradient (@b [hidden] vectors). */
    bf16r1_tensor_arena_t head_norm_scale_ {}, head_norm_scale_grad_ {};
    /** @brief The head's two @b [hidden,hidden] projections and their gradients. */
    bf16r2_tensor_arena_t head_fc1_ {}, head_fc1_grad_ {}, head_fc2_ {}, head_fc2_grad_ {};

    /** @brief RoPE angle tables (model state). */
    rope_angles_t angles_ {};
    /** @brief This step's owned per-token @b [tokens,3] rotary @b (t,h,w) coordinates — derived fresh
     *      every @ref forward from @c batch 's own geometry, resized to the live extent, and reused by
     *      the paired @ref backward (the caller must not run another forward in between). */
    i16r2_tensor_arena_t coords_ {};

    /** @brief Forward stashes the backward reads, all @b [rows,hidden] : the entry output, the per-token
     *      final-norm states, and the pooled / head activations. */
    bf16r2_tensor_arena_t entry_output_ {}, normalized_ {}, pooled_ {}, head_normalized_ {}, head_hidden_ {},
        head_activated_ {};
    /** @brief Float32 @c 1/rms of the final ( @b [tokens] ) and head ( @b [segments] ) norms: the head norm's
     *      forward stashes its own, and the final norm's backward recomputes its own. */
    f32r1_tensor_arena_t final_inverse_rms_ {}, head_inverse_rms_ {};
    /** @brief The block stack's @b [tokens,hidden] Float32 residual-stream ping-pong, the widened entry state
     *      entering at @c activations_b_ , and the borrowed last-layer view. */
    f32r2_tensor_arena_t activations_a_ {}, activations_b_ {};
    f32r2_view_t stack_output_ {};

    /** @brief The @b [tokens,hidden] BFloat16 final-norm states' gradient, later the narrowed entry gradient. */
    bf16r2_tensor_arena_t normalized_grad_ {};
    /** @brief The @b [tokens,hidden] Float32 gradient ping-pong: the widened final-norm gradient, then the block
     *      stack's flowing gradient. */
    f32r2_tensor_arena_t token_grad_a_ {}, token_grad_b_ {};
    /** @brief The @b [segments,hidden] head gradient ping-pong. */
    bf16r2_tensor_arena_t segment_grad_a_ {}, segment_grad_b_ {};

    /** @brief The deterministic embedding-scatter scratch: (key,value) double buffers + radix workspace. */
    tokens_tensor_arena_t sort_keys_ {};
    u32r1_tensor_arena_t sort_values_ {};
    b8r1_tensor_arena_t sort_workspace_ {};
    /** @brief The @b [rows,patch_pixels] patch-cut scratch the media entries fill before their projection. */
    bf16r2_tensor_arena_t patch_scratch_ {};
    /** @brief The step's token ids / media presence, borrowed for the entry backward. */
    tokens_view_t live_token_ids_ {};

    /** @brief The shared cuBLASLt scratch, sized to the max any pinned plan needs. */
    b8r1_tensor_arena_t workspace_ {};
    /** @brief The pinned products, indexed by the @c plan_*_k role constants. */
    std::array<cublaslt_plan_t, plan_capacity_k> plans_ {};
};

/**
 *  @brief The quantized forward-only inference twin: an E4M3 block stack with BFloat16 entries, attention
 *      pool, and head. Constructed only through @ref encoder_tower_bf16_t::quantize_weights .
 */
struct encoder_tower_e4m3_t {

    /** @brief One quantized tower forward over @p batch — same dual-mode contract as the training twin,
     *      forward only, deriving its own per-token rotary @b (t,h,w) coordinates into @ref coords_ . */
    [[nodiscard]] expected_t forward(multimodal_batch_on_gpu_t const &batch, bf16r2_span_t vertices,
                                     gpu_stream_t &stream) noexcept;

    /** @brief The per-token final-norm states — the masked head's input on the masked path. */
    [[nodiscard]] bf16r2_view_t normalized() const noexcept { return normalized_.view(); }

    /** @brief The baked geometry. */
    [[nodiscard]] encoder_tower_config_t const &config() const noexcept { return config_; }

  private:
    friend struct encoder_tower_bf16_t;

    /** @brief The pinned @b cuBLASLt plan roles: the fused head product, then one projection per present
     *      media modality from @ref plan_media_base_k . The token entry is a kernel, not a plan. */
    enum cublaslt_plan_role_t : std::size_t {
        plan_head_forward_k = 0,
        plan_media_base_k = 1,
    };
    /** @brief The @c plans_ capacity: the head role plus one per media modality. */
    static constexpr std::size_t plan_capacity_k = plan_media_base_k + 3u;
    /** @brief Sentinel for a modality with no pinned projection plan. */
    static constexpr std::size_t plan_absent_k = ~std::size_t {0};

    /** @brief Make one pinned cuBLASLt plan (forcing a BFloat16 @c C ), record it at @p role (post-incremented
     *      into @c plans_ ), and grow @p workspace_bytes to its scratch demand — the twin's mirror of the
     *      training tower's plan-pinning step, repeated by @ref encoder_tower_bf16_t::quantize_weights . */
    [[nodiscard]] expected_t pin_plan(cublaslt_plan_config_t plan_config, std::size_t &role,
                                      std::size_t &workspace_bytes) noexcept;
    /** @brief Pin one present media modality's single forward projection plan at @p role , recording its
     *      base into @p plan_base — forward-only, so no weight-grad sibling; the row extent is pinned at
     *      @p max_tokens (the worst case) purely to size the heuristic + workspace. */
    [[nodiscard]] expected_t pin_media_plan(std::size_t &plan_base, u32_t fan_in, u32_t max_tokens, u32_t hidden,
                                            std::size_t &role, std::size_t &workspace_bytes) noexcept;

    /** @brief The baked geometry and the borrowed cuBLASLt session — never owned here. */
    encoder_tower_config_t config_ {};
    cublaslt_session_t *blas_ = nullptr;

    /** @brief The @c depth E4M3 transformer blocks, owned in order — the quantized-precision tier. */
    array<transformer_block_e4m3_t> blocks_ {};

    /** @brief The @b [vocabulary,hidden] BFloat16 token embedding table (forward only, no gradient). */
    bf16r2_tensor_arena_t entry_weight_ {};
    /** @brief The three media patch-projection weights (present per configured modality), BFloat16 —
     *      image @b [hidden, ph·pw·image_channels] , video @b [hidden, tf·ph·pw·video_channels] ,
     *      audio @b [hidden, chunk_samples] (a learnable linear frontend over one raw waveform frame) . */
    bf16r2_tensor_arena_t image_projection_ {}, video_projection_ {}, audio_projection_ {};
    /** @brief The device per-channel image and video pixel mean and inverse-std, uploaded once at make. */
    f32r1_tensor_arena_t image_pixel_mean_ {}, image_pixel_inverse_std_ {};
    f32r1_tensor_arena_t video_pixel_mean_ {}, video_pixel_inverse_std_ {};
    /** @brief Recorded plan-role bases per present media modality (or @c plan_absent_k ). */
    std::size_t image_plan_base_ = plan_absent_k, video_plan_base_ = plan_absent_k, audio_plan_base_ = plan_absent_k;

    /** @brief The final norm scale — the norm tier stays BFloat16. */
    bf16r1_tensor_arena_t final_norm_scale_ {};
    /** @brief The learned attention-pool query @b [heads,head_dim]=[hidden] , BFloat16. */
    bf16r1_tensor_arena_t pool_query_ {};
    /** @brief The attention pool's forward. */
    tower_pool_t pool_ {};
    /** @brief The head's learnable-scale RMSNorm @c γ ( @b [hidden] vector), BFloat16. */
    bf16r1_tensor_arena_t head_norm_scale_ {};
    /** @brief The head's two @b [hidden,hidden] projections, BFloat16. */
    bf16r2_tensor_arena_t head_fc1_ {}, head_fc2_ {};

    /** @brief RoPE angle tables (model state). */
    rope_angles_t angles_ {};
    /** @brief This step's owned per-token @b [tokens,3] rotary @b (t,h,w) coordinates — derived fresh
     *      every @ref forward from @c batch 's own geometry, resized to the live extent. */
    i16r2_tensor_arena_t coords_ {};

    /** @brief Forward activations, all @b [rows,hidden] : the entry output, the per-token final-norm
     *      states, and the pooled / head activations — forward only, nothing stashed for a backward. */
    bf16r2_tensor_arena_t entry_output_ {}, normalized_ {}, pooled_ {}, head_normalized_ {}, head_hidden_ {},
        head_activated_ {};
    /** @brief Float32 @c 1/rms for the final ( @b [tokens] ) and head ( @b [segments] ) norms. */
    f32r1_tensor_arena_t final_inverse_rms_ {}, head_inverse_rms_ {};
    /** @brief The block stack's @b [tokens,hidden] activation ping-pong. */
    bf16r2_tensor_arena_t activations_a_ {}, activations_b_ {};
    /** @brief The @b [rows,patch_pixels] patch-cut scratch the media entries fill before their projection. */
    bf16r2_tensor_arena_t patch_scratch_ {};
    /** @brief The shared cuBLASLt scratch, sized to the max any pinned plan needs. */
    b8r1_tensor_arena_t workspace_ {};
    /** @brief The pinned products, indexed by the @c plan_*_k role constants. */
    std::array<cublaslt_plan_t, plan_capacity_k> plans_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::core
