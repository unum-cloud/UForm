/**
 *  @file include/uform/train/multimodal_masking.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The multimodal masked-modeling objective: deterministic corruption of a clean batch into a
 *      distorted one, and the reconstruction heads that score the tower's states against the clean
 *      target.
 *
 *  The heads are byte cross-entropy for tokens, normalized-pixel MSE for image / video patches, and
 *  normalized-sample MSE for 1-D audio frames.
 *
 *  @section masking_two_instances Two instances of one batch
 *
 *  The objective runs on two instances of the same representation: the clean one is the reconstruction
 *  target, the distorted one is what the tower sees. Corruption is a pure function of @c (seed,data_epoch,
 *  position) through the data epoch shuffle's mixer — identical on every host_domain, bit-reproducible, no
 *  cuRAND and no state. Tokens are corrupted in vocabulary space (MASK / random byte / kept); for image /
 *  video patches and 1-D audio frames the driver zeros the masked raw media rows in place before the tower
 *  re-patchifies.
 *
 *  @section masking_gradient The additive join with the tower
 *
 *  Every head reads the tower's final-norm token states and writes @c d(normalized) into a caller-owned
 *  buffer the tower's backward ADDS before its final-norm backward — so @c L = L_pooled + Σ λ·L_masked
 *  by construction, and each objective stays ignorant of the others. The tower's entry projection and
 *  RoPE are never touched here: the only boundary is @c normalized() in, @c normalized_grad out.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "uform/utilities/vram_nvidia_device.hpp"   // `gpu_arena_t`, `gpu_arena_alloc`
#include "uform/encoder/multimodal_batch.hpp"       // `modality_t`, `byte_vocabulary_k`, `byte_pad_k`, `byte_mask_k`
#include "uform/utilities/vram_nvidia_schedule.cuh" // `gpu_stream_t`, the NumKong products, the view aliases
#include "uform/utilities/types.hpp" // `expected`, `expected_t`, `status_t`, the scalar aliases, `st::basic_vector`

namespace unum::uform::train {
using namespace unum::uform::utilities;
using encoder::byte_vocabulary_k;
using encoder::modality_t;
using encoder::modality_view_t;
using encoder::multimodal_batch_view_t;

/** The deterministic corruption policy — a host value type, baked per run. */
struct masking_recipe_t {

    /** The run's masking seed — independent of the shuffle seed by convention. */
    u64_t seed = 0;

    /** Target fraction of token positions covered by spans. */
    f32_t token_mask_probability = 0.15f;

    /** Mean geometric span length; a span opens at rate @c token_mask_probability/token_mean_span . */
    f32_t token_mean_span = 16.0f;

    /** Of covered token positions: shown as MASK / replaced by a random byte / kept clean. */
    f32_t replace_fraction = 0.80f, random_fraction = 0.10f;
};

/** The domain-separation salts that keep the independent draws at one corruption position from
 *  correlating: each purpose re-keys the hash so the span-open coin, the span-continue coin, the
 *  token-action coin, the random byte, the patch-mask coin, the audio-frame-mask coin, and the
 *  anchor of the span an otherwise unmasked row gets are statistically distinct streams. The
 *  @ref masked_frames_t frame path keys @c frame_mask_k rather than reusing the 2-D @c patch_mask_k
 *  salt. */
enum class masking_draw_purpose_t : u32_t {
    span_open_k = 0,
    span_continue_k = 1,
    token_action_k = 2,
    random_byte_k = 3,
    patch_mask_k = 4,
    frame_mask_k = 5,
    span_anchor_k = 6,
};

/** One counter-based draw in @b [0,1) , keyed through the one @ref splitmix64_hash_t — pure,
 *  stateless, device-callable, bit-reproducible on every rank. @p purpose is a
 *  @ref masking_draw_purpose_t salt cast to @c u32_t . */
[[nodiscard]] constexpr f32_t masking_draw(u64_t seed, data_epoch_t data_epoch, u32_t sample_index, u32_t position,
                                           u32_t purpose) noexcept {
    u64_t const mixed = splitmix64_hash_t {}(seed, data_epoch.ordinal, purpose, sample_index, position);
    return static_cast<f32_t>(mixed >> 40) * 0x1.0p-24f; // The top 24 bits as a uniform [0,1).
}

/** The baked geometry and weight of one byte masked-modeling head — a different config is a
 *  different object. */
struct masked_tokens_config_t {

    /** Decoder rows; @ref byte_vocabulary_k for the byte objective. */
    u32_t vocabulary = byte_vocabulary_k;

    /** Embedding width — must match the tower's. */
    u32_t hidden = 0;

    /** Most token positions one call carries — the @c normalized / @c normalized_grad extent. */
    u32_t max_tokens = 0;

    /** λ — the objective's weight inside the composite loss, folded into the gradient scale. */
    f32_t weight = 1.0f;

    /** Most segments one call carries, the answer rows @ref masked_tokens_t::choose gathers; zero
     *  allocates none and makes no choice. */
    u32_t max_segments = 0;
};

/** SearchJev's calibrated choices over the byte decoder; a zero weight trains none. */
struct choices_config_t {

    /** λ of the choices term in the composite loss. */
    f32_t weight = 0.0f;

    /** λ_B, the Brier term's weight beside the soft cross-entropy. */
    f32_t brier_weight = 0.5f;
};

/**
 *  @brief The baked geometry and weight of one masked-patch (MAE) head — a different config is a
 *      different object. Image and video share this shape; they differ only in @c patch_pixels , the mask
 *      ratio, and whether the mask is tubed across time.
 *  @note Audio no longer rides this 2-D patch head: it is raw 1-D frames on the sibling @ref masked_frames_t ,
 *      which masks whole Float32 frames and scores their raw samples with the same normalized MSE rather
 *      than this patch objective.
 */
struct masked_patches_config_t {

    /** Embedding width — must match the tower's. */
    u32_t hidden = 0;

    /** Scalars per patch vector: image @c 3·p² , video @c tube_time·3·th·tw . */
    u32_t patch_pixels = 0;

    /** Most patches one call carries — the @c normalized / patch-table extent. */
    u32_t max_patches = 0;

    /** Fraction of patches masked and scored — 0.75 image, 0.90 video. */
    f32_t mask_ratio = 0.75f;

    /** One for per-patch random masking; the video temporal-group count for tube masking, where a
     *  masked spatial cell stays masked across all its time groups. @c max_patches must divide evenly. */
    u32_t temporal_groups = 1;

    /** The modality whose cells the head masks and scores, @c image_k or @c video_k . */
    modality_t modality = modality_t::image_k;

    /** λ — the objective's weight inside the composite loss, folded into the gradient scale. */
    f32_t weight = 1.0f;
};

/** The baked geometry and weight of one masked-frame (1-D raw-audio MAE) head — a different config is
 *  a different object. Audio arrives as raw waveform frames, not 2-D patches, so this head masks whole
 *  Float32 frames per-frame i.i.d. and reconstructs their raw samples; there is no tube concept. */
struct masked_frames_config_t {

    /** Embedding width — must match the tower's. */
    u32_t hidden = 0;

    /** Float32 samples per waveform frame — the innermost @c audio_samples extent and the recon
     *  head's fan-out @c hidden → @c chunk_samples . */
    u32_t chunk_samples = 0;

    /** Most frames one call carries — the @c normalized / frame-table extent. */
    u32_t max_frames = 0;

    /** Fraction of frames masked and scored, each frame drawn independently. */
    f32_t mask_ratio = 0.75f;

    /** λ — the objective's weight inside the composite loss, folded into the gradient scale. */
    f32_t weight = 1.0f;
};

#if defined(__CUDACC__)

/**
 *  @brief The byte masked-modeling objective: a deterministic span-masking corruptor and an untied byte
 *      decoder with fused cross-entropy. @ref corrupt turns a clean token stream into the distorted one
 *      the tower reads plus the per-position loss weights.
 *
 *  @ref operator() scores the tower's final-norm states against the clean bytes and writes
 *  @c d(normalized) for the additive join.
 *
 *  The decoder is deliberately @b untied from the tower's embedding table: at ~258×hidden it costs
 *  kilobytes, gets its own clean @ref differentiable_weights_t , and the loss is @b dense — every position
 *  is scored, gated by its loss weight — so there is no host-side compaction, no staging ring, and the
 *  whole step is a pure capturable enqueue.
 */
struct masked_tokens_t {

    /** Allocate the decoder, its NumKong packs, and every scratch — once, at setup — calling the
     *  kernels of @p numerics . */
    [[nodiscard]] static expected<masked_tokens_t> make(gpu_arena_t &arena, masked_tokens_config_t config,
                                                        gpu_numerics_t const &numerics) noexcept;

    /** Pack the decoder for NumKong's products — pure enqueue, capture-safe. Call once per optimizer
     *  step, after the weights change and before the first @ref operator() . */
    [[nodiscard]] expected_t pack_weights(gpu_stream_t &stream) noexcept;

    /**
     *  @brief Corrupt @p clean into @p distorted and emit the per-position @p loss_weights — pure enqueue.
     *
     *  Only @c text_k segments are corrupted; a segment whose @p segment_modality is any media
     *  kind, or whose @p segment_choices row opens with a byte other than @c byte_pad_k , is
     *  skipped outright, leaving its tokens clean with weight zero (mixed rows form at mix time, so
     *  this routes on the tag rather than trusting media positions to hold PAD). A text segment's
     *  tokens are walked left-to-right, where geometric spans continue at @c 1-1/token_mean_span
     *  and open at @c token_mask_probability/token_mean_span ; a covered position is shown MASK / a
     *  random byte / its clean self per the recipe fractions, and carries a loss weight of one. PAD
     *  is never eligible; uncovered positions copy clean with weight zero. A segment with bytes the
     *  walk leaves uncovered gets one span from a keyed anchor, so every text row is scored.
     *
     *  @param[in] clean The clean token stream @b [tokens] in flat @c segment_offsets space.
     *  @param[out] distorted The tower's input @b [tokens] — clean elsewhere, corrupted at covered spans.
     *  @param[out] loss_weights @b [tokens] : one where the loss scores this position, zero elsewhere.
     *  @param[in] segment_offsets @b [segments+1] flat start positions; @p segment_lengths @b [segments]
     *      live token counts (a segment never crosses its row).
     *  @param[in] segment_modality @b [segments] per-segment routing tag; only @c text_k segments corrupt.
     *  @param[in] segment_choices @b [segments,choices] each segment's legal bytes, empty when the
     *      batch carries none; a choice segment stays clean.
     */
    [[nodiscard]] expected_t corrupt(tokens_view_t clean, tokens_span_t distorted, u32r1_span_t loss_weights,
                                     u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                     modality_view_t segment_modality, nk::tensor_view<token_id_t, 2> segment_choices,
                                     masking_recipe_t recipe, data_epoch_t data_epoch, gpu_stream_t &stream) noexcept;

    /**
     *  @brief Forward + backward over every token, gated by its loss weight — pure enqueue on @p stream .
     *
     *  @param[in] normalized The tower's final-norm token states @b [tokens,hidden] .
     *  @param[in] clean_targets The clean bytes @b [tokens] the cross-entropy scores against.
     *  @param[in] loss_weights @b [tokens] from @ref corrupt — a zero weight skips the position entirely.
     *  @param[inout] loss_summands @b [2] Float32 the call adds @c {λ·Σ CE, live count} onto, so
     *      the pair runs over every step since the caller zeroed it — the global mean is
     *      @c Σ_ranks summand / Σ_ranks count ; the gradient uses this step's rank-local mean.
     *  @param[out] normalized_grad @b [tokens,hidden] @c d(normalized) — the tower's backward ADDS it.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t normalized, tokens_view_t clean_targets,
                                        u32r1_view_t loss_weights, f32r1_span_t loss_summands,
                                        bf16r2_span_t normalized_grad, gpu_stream_t &stream) noexcept;

    /**
     *  @brief Soft-target cross-entropy plus the Brier term over each choice segment's legal bytes,
     *      read at its last row of @p normalized , adding into @p normalized_grad ; reuses
     *      @c logits_ and @c logit_grad_ — pure enqueue on @p stream .
     *
     *  Per choice segment, @c p is the softmax of the decoder's logits restricted to its bytes and
     *  @c t its targets, and the summand is @c −Σt·log(p)+λ_B·Σ(p−t)² , scaled by @c λ over the
     *  live choice segments. Every other vocabulary column and every other row gets no gradient.
     *  The decoder's weight gradient is added onto, so the masked @ref operator() runs first when
     *  both train.
     *
     *  @param[in] normalized The tower's final-norm token states @b [tokens,hidden] .
     *  @param[in] batch The step's segment geometry, choices and targets.
     *  @param[in] config The weights @c λ and @c λ_B .
     *  @param[inout] loss_summands @b [2] Float32 the call adds @c {λ·Σ summand, live count} onto.
     *  @param[inout] normalized_grad @b [tokens,hidden] @c d(normalized) , added onto at the answer
     *      rows.
     */
    [[nodiscard]] expected_t choose(bf16r2_view_t normalized, multimodal_batch_view_t const &batch,
                                    choices_config_t config, f32r1_span_t loss_summands, bf16r2_span_t normalized_grad,
                                    gpu_stream_t &stream) noexcept;

    /** Append the decoder weight to @p weights under the checkpoint name @p name — the owning
     *  replica spells the instance name, since one head class may serve several modalities. */
    [[nodiscard]] expected_t fill(sz::string_view_t name, st::basic_vector<differentiable_weights_t> &weights) noexcept;

    /** The decoder weights @b [vocabulary·hidden] — the checkpoint enumerator's handle. */
    [[nodiscard]] nk::tensor_span<bf16_t, 1> decoder_weights() noexcept { return decoder_weight_.span().flatten<1>(); }

    /** The decoder's gradient @b [vocabulary·hidden] , which a step choosing without reconstructing
     *  zeroes before @ref choose adds onto it. */
    [[nodiscard]] nk::tensor_span<bf16_t, 1> decoder_gradients() noexcept {
        return decoder_weight_grad_.span().flatten<1>();
    }

  private:
    masked_tokens_config_t config_ {};

    /** The NumKong kernels of the arena's device class: every product and cast. */
    gpu_numerics_t numerics_ {};

    /** The untied decoder @b [vocabulary,hidden] and its gradient buffer. */
    bf16r2_tensor_arena_t decoder_weight_ {}, decoder_weight_grad_ {};

    /** The decoder packed as the logits' B operand and, transposed, as @c d(normalized) 's. */
    b8r1_tensor_arena_t packed_decoder_ {}, packed_decoder_transposed_ {};

    /** @b [max_tokens,vocabulary] Float32 logits — the cross-entropy's stable surface. */
    f32r2_tensor_arena_t logits_ {};

    /** @b [max_tokens,vocabulary] BFloat16 @c d(logits) — the backward products' operand, its rows padded
     *  to 8 for NumKong's 16-byte stride rule, since the byte vocabulary is 258 wide. */
    bf16r2_tensor_arena_t logit_grad_ {};

    /** @b [max_tokens] per-position cross-entropy summands, the first @c max_segments of which also
     *  hold the choice summands. */
    f32r1_tensor_arena_t position_loss_ {};

    /** @b [2] @c {λ/max(count,1), count} — device-computed, so the count never visits the host. */
    f32r1_tensor_arena_t gradient_scale_ {};

    /** The weight gradient's transposes, the latter also staging the decoder's for @ref pack_weights ,
     *  the pack of @c normalizedᵀ , and the Float32 products before their casts. */
    bf16r1_tensor_arena_t transposed_grad_ {}, transposed_input_ {};
    b8r1_tensor_arena_t packed_input_ {};
    f32r1_tensor_arena_t products_ {};

    /** @b [max_segments,hidden] each choice segment's answer row, contiguous for one product; empty
     *  without choices. */
    bf16r2_tensor_arena_t gathered_ {};
};

/**
 *  @brief The masked-image / video objective: a deterministic patch-mask sampler and a raw-media corruptor,
 *      plus a linear reconstruction head with a per-patch normalized-MSE loss.
 *
 *  @ref sample_mask draws the patch mask and @ref corrupt_raw zeros the masked raw media rows in place
 *  before the tower re-patchifies; @ref operator() reconstructs the clean patches from the tower's states
 *  and writes @c d(normalized) for the additive join.
 *
 *  The head is a plain linear projection @c hidden → @c patch_pixels : the tower already RMSNorms the states it
 *  hands back through @c normalized() , so a second pre-norm here would only re-normalize. A masked patch's
 *  raw row is zeroed — no @c d(patches) is needed from the tower, so the objective stays fully orthogonal to
 *  the entry projection.
 *
 *  The per-patch MSE normalizes the target by its own per-patch mean and standard deviation and
 *  regresses the raw prediction onto it, as He et al.'s MAE does. Predicting the patch mean scores
 *  @c E[targ_norm^2]=1 , a closed-form chance baseline of @c 1.0 the multi-objective composer
 *  divides by, onto the same @c [0,1] scale as the MLM ( @c ln256 ) and contrastive ( @c ln2 )
 *  losses. Only cells of the configured modality are masked.
 */
struct masked_patches_t {

    /** Allocate the recon head, its NumKong packs, and every scratch — once, at setup — calling the
     *  kernels of @p numerics . */
    [[nodiscard]] static expected<masked_patches_t> make(gpu_arena_t &arena, masked_patches_config_t config,
                                                         gpu_numerics_t const &numerics) noexcept;

    /** Pack the recon head for NumKong's products — pure enqueue, capture-safe. Call once per optimizer
     *  step, after the weights change and before the first @ref operator() . */
    [[nodiscard]] expected_t pack_weights(gpu_stream_t &stream) noexcept;

    /**
     *  @brief Sample the patch mask into @c mask_ — pure enqueue, nothing else.
     *
     *  A cell of a segment whose @p segment_modality is the configured one is masked with
     *  probability @c mask_ratio , tubed across time groups when @c temporal_groups > 1 so a masked
     *  spatial cell stays masked across all its time groups; every other cell stays visible and
     *  unscored.
     */
    [[nodiscard]] expected_t sample_mask(masking_recipe_t recipe, data_epoch_t data_epoch, u32r1_view_t segment_offsets,
                                         u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                         gpu_stream_t &stream) noexcept;

    /**
     *  @brief Corrupt the raw batch media in place — pure enqueue. Zeros every raw patch row whose @c mask_
     *      entry is one and leaves the visible rows untouched, so the tower re-patchifies the corrupted
     *      pixels. Call after @ref sample_mask , on the live batch media the tower reads.
     *
     *  @param[inout] raw_patches The batch's image / video patches @b [patches,patch_scalars] of raw
     *      @ref pixel_t — @c batch.images / @c batch.videos flattened to one row per patch.
     */
    [[nodiscard]] expected_t corrupt_raw(nk::tensor_span<pixel_t, 2> raw_patches, gpu_stream_t &stream) noexcept;

    /**
     *  @brief Reconstruct and score the masked patches — pure enqueue on @p stream .
     *
     *  @param[in] normalized The tower's final-norm patch states @b [patches,hidden] .
     *  @param[in] clean_patches The reconstruction target @b [patches,patch_pixels] .
     *  @param[inout] loss_summands @b [2] Float32 the call adds @c {λ·Σ per-patch normalized-MSE,
     *      masked count} onto.
     *  @param[out] normalized_grad @b [patches,hidden] @c d(normalized) — the tower's backward ADDS
     *      it.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t normalized, bf16r2_view_t clean_patches,
                                        f32r1_span_t loss_summands, bf16r2_span_t normalized_grad,
                                        gpu_stream_t &stream) noexcept;

    /** Append the recon weight to @p weights under the checkpoint name @p name — the owning
     *  replica spells the instance name, since one head class may serve several modalities. */
    [[nodiscard]] expected_t fill(sz::string_view_t name, st::basic_vector<differentiable_weights_t> &weights) noexcept;

    /** The recon weights @b [patch_pixels·hidden] — the checkpoint enumerator's handle. */
    [[nodiscard]] nk::tensor_span<bf16_t, 1> recon_weights() noexcept { return recon_weight_.span().flatten<1>(); }

  private:
    masked_patches_config_t config_ {};

    /** The NumKong kernels of the arena's device class: every product and cast. */
    gpu_numerics_t numerics_ {};

    /** The linear reconstruction head @b [patch_pixels,hidden] and its gradient buffer. */
    bf16r2_tensor_arena_t recon_weight_ {}, recon_weight_grad_ {};

    /** @b [max_patches,patch_pixels] reconstructed patches and their loss gradient. */
    bf16r2_tensor_arena_t predicted_ {}, predicted_grad_ {};

    /** @b [max_patches] the sampled mask (one masked, zero visible) the loss gates on. */
    u32r1_tensor_arena_t mask_ {};

    /** @b [max_patches] per-patch normalized-MSE summands. */
    f32r1_tensor_arena_t position_loss_ {};

    /** @b [2] @c {λ·2/(patch_pixels·max(count,1)), count} — device-computed. */
    f32r1_tensor_arena_t gradient_scale_ {};

    /** The recon head packed as the prediction's B operand and, transposed, as @c d(normalized) 's. */
    b8r1_tensor_arena_t packed_recon_ {}, packed_recon_transposed_ {};

    /** The weight gradient's transposes, the latter also staging the head's for @ref pack_weights , the
     *  pack of @c normalizedᵀ , the Float32 products before their casts, and the widened @c d(normalized)
     *  this head adds onto. */
    bf16r1_tensor_arena_t transposed_grad_ {}, transposed_input_ {};
    b8r1_tensor_arena_t packed_input_ {};
    f32r1_tensor_arena_t products_ {}, widened_grad_ {};
};

/**
 *  @brief The masked-audio objective: a deterministic per-frame mask sampler and a raw-media corruptor,
 *      plus a linear reconstruction head with a per-frame normalized-MSE loss over the raw Float32 waveform.
 *
 *  @ref sample_mask draws the frame mask and @ref corrupt_raw zeros the masked raw waveform rows in place
 *  before the tower re-projects; @ref operator() reconstructs the clean frames from the tower's states and
 *  writes @c d(normalized) for the additive join.
 *
 *  Audio is delivered raw as @c audio_samples[rows,columns,chunk_samples] — a run of 1-D waveform frames,
 *  not 2-D patches — and the tower's audio entry linearly projects each @c chunk_samples -wide frame to
 *  @c hidden . This head is the mirror on the way out: a plain linear projection @c hidden → @c chunk_samples
 *  whose target is the raw frame. The states arrive already RMSNormed through @c normalized() , so no
 *  pre-norm is repeated here, and a masked frame's raw row is zeroed, so the objective stays orthogonal to
 *  the entry projection.
 *
 *  The per-frame MSE normalizes the raw waveform target by its own per-frame mean and standard
 *  deviation and regresses the raw prediction onto it, identical to @ref masked_patches_t , with
 *  the same chance baseline of @c 1.0 . Only cells of audio segments are masked.
 */
struct masked_frames_t {

    /** Allocate the recon head, its NumKong packs, and every scratch — once, at setup — calling the
     *  kernels of @p numerics . */
    [[nodiscard]] static expected<masked_frames_t> make(gpu_arena_t &arena, masked_frames_config_t config,
                                                        gpu_numerics_t const &numerics) noexcept;

    /** Pack the recon head for NumKong's products — pure enqueue, capture-safe. Call once per optimizer
     *  step, after the weights change and before the first @ref operator() . */
    [[nodiscard]] expected_t pack_weights(gpu_stream_t &stream) noexcept;

    /**
     *  @brief Sample the frame mask into @c mask_ — pure enqueue, nothing else.
     *
     *  A frame of an audio segment is masked with probability @c mask_ratio , drawn independently
     *  per frame; every other cell stays visible and unscored.
     */
    [[nodiscard]] expected_t sample_mask(masking_recipe_t recipe, data_epoch_t data_epoch, u32r1_view_t segment_offsets,
                                         u32r1_view_t segment_lengths, modality_view_t segment_modality,
                                         gpu_stream_t &stream) noexcept;

    /**
     *  @brief Corrupt the raw batch media in place — pure enqueue. Zeros every raw waveform frame whose
     *      @c mask_ entry is one and leaves the visible frames untouched, so the tower re-projects the
     *      corrupted samples. Call after @ref sample_mask , on the live batch audio the tower reads.
     *
     *  @param[inout] raw_frames The batch's audio frames @b [frames,chunk_samples] Float32 — @c audio_samples
     *      flattened to one row per frame.
     */
    [[nodiscard]] expected_t corrupt_raw(f32r2_span_t raw_frames, gpu_stream_t &stream) noexcept;

    /**
     *  @brief Reconstruct and score the masked frames — pure enqueue on @p stream .
     *
     *  @param[in] normalized The tower's final-norm frame states @b [frames,hidden] .
     *  @param[in] clean_frames The reconstruction target @b [frames,chunk_samples] Float32.
     *  @param[inout] loss_summands @b [2] Float32 the call adds @c {λ·Σ per-frame normalized-MSE,
     *      masked count} onto.
     *  @param[out] normalized_grad @b [frames,hidden] @c d(normalized) — the tower's backward ADDS
     *      it.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t normalized, f32r2_view_t clean_frames, f32r1_span_t loss_summands,
                                        bf16r2_span_t normalized_grad, gpu_stream_t &stream) noexcept;

    /** Append the recon weight to @p weights under the checkpoint name @p name — the owning
     *  replica spells the instance name, since one head class may serve several modalities. */
    [[nodiscard]] expected_t fill(sz::string_view_t name, st::basic_vector<differentiable_weights_t> &weights) noexcept;

    /** The recon weights @b [chunk_samples·hidden] — the checkpoint enumerator's handle. */
    [[nodiscard]] nk::tensor_span<bf16_t, 1> recon_weights() noexcept { return recon_weight_.span().flatten<1>(); }

  private:
    masked_frames_config_t config_ {};

    /** The NumKong kernels of the arena's device class: every product and cast. */
    gpu_numerics_t numerics_ {};

    /** The linear reconstruction head @b [chunk_samples,hidden] and its gradient buffer. */
    bf16r2_tensor_arena_t recon_weight_ {}, recon_weight_grad_ {};

    /** @b [max_frames,chunk_samples] reconstructed frames and their loss gradient. */
    bf16r2_tensor_arena_t predicted_ {}, predicted_grad_ {};

    /** @b [max_frames] the sampled mask (one masked, zero visible) the loss gates on. */
    u32r1_tensor_arena_t mask_ {};

    /** @b [max_frames] per-frame normalized-MSE summands. */
    f32r1_tensor_arena_t position_loss_ {};

    /** @b [2] @c {λ·2/(chunk_samples·max(count,1)), count} — device-computed. */
    f32r1_tensor_arena_t gradient_scale_ {};

    /** The recon head packed as the prediction's B operand and, transposed, as @c d(normalized) 's. */
    b8r1_tensor_arena_t packed_recon_ {}, packed_recon_transposed_ {};

    /** The weight gradient's transposes, the latter also staging the head's for @ref pack_weights , the
     *  pack of @c normalizedᵀ , the Float32 products before their casts, and the widened @c d(normalized)
     *  this head adds onto. */
    bf16r1_tensor_arena_t transposed_grad_ {}, transposed_input_ {};
    b8r1_tensor_arena_t packed_input_ {};
    f32r1_tensor_arena_t products_ {}, widened_grad_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::train
