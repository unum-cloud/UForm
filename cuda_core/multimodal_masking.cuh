/**
 *  @brief The multimodal masked-modeling objective: deterministic corruption of a clean batch into a
 *      distorted one, and the reconstruction heads that score the tower's states against the clean
 *      target — byte cross-entropy for tokens, normalized-pixel MSE for image / video / audio patches.
 *  @file cuda_core/multimodal_masking.cuh
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  @section masking_two_instances Two instances of one batch
 *
 *  The objective runs on two instances of the same representation: the clean one is the reconstruction
 *  target, the distorted one is what the tower sees. Corruption is a pure function of @c (seed,epoch,
 *  position) through the epoch shuffle's mixer — identical on every host_domain, bit-reproducible, no
 *  cuRAND and no state. Tokens are corrupted in vocabulary space (MASK / random byte / kept); media
 *  patches are zeroed at masked rows before the tower's entry projection.
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

#include <array> // `std::array` — the pinned matmul plans

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "cuda_core/types.cuh"          // `expected`, `expected_t`, `status_t`, the scalar aliases
#include "cuda_core/array.cuh"          // `array` — the parameter-table append
#include "cuda_core/gpu_allocators.cuh" // `gpu_arena_t`, `gpu_arena_alloc`
#include "cuda_core/schedule.cuh"       // `gpu_stream_t`, `cublaslt_session_t`, `cublaslt_plan_t`, the view aliases

namespace unum::uform::core {

/** @brief The byte vocabulary: 256 UTF-8 byte values plus the PAD and MASK specials. */
inline constexpr u32_t byte_vocabulary_k = 258u;
/** @brief The PAD byte id — token 256; ineligible for masking, its loss weight is always zero. */
inline constexpr token_id_t byte_pad_k = 256u;
/** @brief The MASK byte id — token 257, what the replace fraction of masked positions shows the tower. */
inline constexpr token_id_t byte_mask_k = 257u;

/** @brief The deterministic corruption policy — a host value type, baked per run. */
struct masking_recipe_t {
    /** @brief The run's masking seed — independent of the shuffle seed by convention. */
    u64_t seed = 0;
    /** @brief Target fraction of token positions covered by spans. */
    f32_t token_mask_probability = 0.15f;
    /** @brief Mean geometric span length; a span opens at rate @c token_mask_probability/token_mean_span . */
    f32_t token_mean_span = 16.0f;
    /** @brief Of covered token positions: shown as MASK / replaced by a random byte / kept clean. */
    f32_t replace_fraction = 0.80f, random_fraction = 0.10f;
    /** @brief Per-patch mask fractions for the media MAE heads. */
    f32_t image_mask_ratio = 0.75f, video_mask_ratio = 0.90f, audio_mask_ratio = 0.75f;
    /** @brief The video tube extent — a masked spatial cell is masked across all @c video_tube_time frames. */
    u32_t video_tube_time = 2, video_tube_height = 16, video_tube_width = 16;
};

/** @brief One counter-based draw in @b [0,1) , keyed through the one @ref splitmix64_hash_t — pure,
 *      stateless, device-callable, bit-reproducible on every rank. */
[[nodiscard]] constexpr f32_t masking_draw(u64_t seed, u32_t epoch, u32_t sample_index, u32_t position,
                                           u32_t purpose) noexcept {
    u64_t const mixed = splitmix64_hash_t {}(seed, epoch, purpose, sample_index, position);
    return static_cast<f32_t>(mixed >> 40) * 0x1.0p-24f; // The top 24 bits as a uniform [0,1).
}

/** @brief The baked geometry and weight of one byte masked-modeling head — a different config is a
 *      different object. */
struct masked_tokens_config_t {
    /** @brief Decoder rows; @ref byte_vocabulary_k for the byte objective. */
    u32_t vocabulary = byte_vocabulary_k;
    /** @brief Embedding width — must match the tower's. */
    u32_t hidden = 0;
    /** @brief Most token positions one call carries — the @c normalized / @c normalized_grad extent. */
    u32_t max_tokens = 0;
    /** @brief λ — the objective's weight inside the composite loss, folded into the gradient scale. */
    f32_t weight = 1.0f;
};

#if defined(__CUDACC__)

/**
 *  @brief The byte masked-modeling objective: a deterministic span-masking corruptor and an untied byte
 *      decoder with fused cross-entropy. @ref corrupt turns a clean token stream into the distorted one
 *      the tower reads plus the per-position loss weights; @ref operator() scores the tower's final-norm
 *      states against the clean bytes and writes @c d(normalized) for the additive join.
 *
 *  The decoder is deliberately @b untied from the tower's embedding table: at ~258×hidden it costs
 *  kilobytes, gets its own clean @ref differentiable_weights_t , and the loss is @b dense — every position
 *  is scored, gated by its loss weight — so there is no host-side compaction, no staging ring, and the
 *  whole step is a pure capturable enqueue.
 */
struct masked_tokens_t {

    /** @brief Allocate the decoder, every scratch, and pin the three matmul plans — once, at setup.
     *  @param[in] blas The device's one tensor-core session, main-owned and borrowed. */
    [[nodiscard]] static expected<masked_tokens_t> make(gpu_arena_t &arena, cublaslt_session_t &blas,
                                                        masked_tokens_config_t config) noexcept;

    /**
     *  @brief Corrupt @p clean into @p distorted and emit the per-position @p loss_weights — pure enqueue.
     *      Each segment's tokens are walked left-to-right: geometric spans open at
     *      @c token_mask_probability/token_mean_span and continue at @c 1-1/token_mean_span ; a covered
     *      position is shown MASK / a random byte / its clean self per the recipe fractions, and carries a
     *      loss weight of one. PAD is never eligible; uncovered positions copy clean with weight zero.
     *
     *  @param[in] clean The clean token stream @b [tokens] in flat @c segment_offsets space.
     *  @param[out] distorted The tower's input @b [tokens] — clean elsewhere, corrupted at covered spans.
     *  @param[out] loss_weights @b [tokens] : one where the loss scores this position, zero elsewhere.
     *  @param[in] segment_offsets @b [segments+1] flat start positions; @p segment_lengths @b [segments]
     *      live token counts (a segment never crosses its row).
     */
    [[nodiscard]] expected_t corrupt(tokens_view_t clean, tokens_span_t distorted, u32r1_span_t loss_weights,
                                     u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                     masking_recipe_t recipe, epoch_t epoch, gpu_stream_t &stream) noexcept;

    /**
     *  @brief Forward + backward over every token, gated by its loss weight — pure enqueue on @p stream .
     *
     *  @param[in] normalized The tower's final-norm token states @b [tokens,hidden] .
     *  @param[in] clean_targets The clean bytes @b [tokens] the cross-entropy scores against.
     *  @param[in] loss_weights @b [tokens] from @ref corrupt — a zero weight skips the position entirely.
     *  @param[out] loss_summands @b [2] Float32: @c {λ·Σ CE, live count} — the global mean is
     *      @c Σ_ranks summand / Σ_ranks count ; the gradient uses the rank-local mean.
     *  @param[out] normalized_grad @b [tokens,hidden] @c d(normalized) — the tower's backward ADDS it.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t normalized, tokens_view_t clean_targets,
                                        u32r1_view_t loss_weights, f32r1_span_t loss_summands,
                                        bf16r2_span_t normalized_grad, gpu_stream_t &stream) noexcept;

    /** @brief Append the decoder weight's optimizer slot. */
    [[nodiscard]] expected_t parameters(array<differentiable_weights_t> &slots) noexcept;
    /** @brief The decoder weights @b [vocabulary·hidden] — the checkpoint enumerator's handle. */
    [[nodiscard]] nk::tensor_span<bf16_t, 1> decoder_weights() noexcept { return decoder_weight_.span().flatten<1>(); }

  private:
    masked_tokens_config_t config_ {};
    /** @brief The device's one main-owned tensor-core session — borrowed, never created here. */
    cublaslt_session_t *session_ = nullptr;
    /** @brief The untied decoder @b [vocabulary,hidden] and its gradient buffer. */
    bf16r2_tensor_arena_t decoder_weight_ {}, decoder_weight_grad_ {};
    /** @brief @b [max_tokens,vocabulary] Float32 logits — the cross-entropy's stable surface. */
    f32r2_tensor_arena_t logits_ {};
    /** @brief @b [max_tokens,vocabulary] BFloat16 @c d(logits) — the backward matmuls' operand. */
    bf16r2_tensor_arena_t logit_grad_ {};
    /** @brief @b [max_tokens] per-position cross-entropy summands. */
    f32r1_tensor_arena_t position_loss_ {};
    /** @brief @b [2] @c {λ/max(count,1), count} — device-computed, so the count never visits the host. */
    f32r1_tensor_arena_t gradient_scale_ {};
    /** @brief The shared tensor-core workspace of the three pinned plans. */
    b8r1_tensor_arena_t workspace_ {};
    /** @brief Pinned plans: logits, @c d(normalized) , @c d(decoder) . */
    std::array<cublaslt_plan_t, 3> plans_ {};
};

#endif // defined(__CUDACC__)

/** @brief The baked geometry and weight of one masked-patch (MAE) head — a different config is a
 *      different object. Image, video, and audio share this shape; they differ only in @c patch_pixels ,
 *      the mask ratio, and whether the mask is tubed across time. */
struct masked_patches_config_t {
    /** @brief Embedding width — must match the tower's. */
    u32_t hidden = 0;
    /** @brief Scalars per patch vector: image @c 3·p² , video @c tube_time·3·th·tw , audio @c mel_h·mel_w . */
    u32_t patch_pixels = 0;
    /** @brief Most patches one call carries — the @c normalized / patch-table extent. */
    u32_t max_patches = 0;
    /** @brief Fraction of patches masked and scored — 0.75 image / audio, 0.90 video. */
    f32_t mask_ratio = 0.75f;
    /** @brief One for per-patch random masking; the video temporal-group count for tube masking, where a
     *      masked spatial cell stays masked across all its time groups. @c max_patches must divide evenly. */
    u32_t temporal_groups = 1;
    /** @brief λ — the objective's weight inside the composite loss, folded into the gradient scale. */
    f32_t weight = 1.0f;
};

#if defined(__CUDACC__)

/**
 *  @brief The masked-image / video / audio objective: a deterministic patch-mask sampler with a
 *      fixed-zero substitution, and a linear reconstruction head with a per-patch normalized-MSE loss.
 *      @ref sample_and_corrupt zeros the masked patch vectors before the tower's entry projection;
 *      @ref operator() reconstructs the clean patches from the tower's states and writes @c d(normalized)
 *      for the additive join.
 *
 *  The head is a plain linear projection @c hidden→patch_pixels : the tower already RMSNorms the states it
 *  hands back through @c normalized() , so a second pre-norm here would only re-normalize. The mask token
 *  is a fixed zero (UFormTrain zero-inits it too) — no @c d(patches) is needed from the tower, so the
 *  objective stays fully orthogonal to the entry projection.
 */
struct masked_patches_t {

    /** @brief Allocate the recon head, every scratch, and pin the three matmul plans — once, at setup. */
    [[nodiscard]] static expected<masked_patches_t> make(gpu_arena_t &arena, cublaslt_session_t &blas,
                                                         masked_patches_config_t config) noexcept;

    /**
     *  @brief Sample the patch mask and write @p distorted_patches — pure enqueue. A patch is masked with
     *      probability @c mask_ratio (tubed across time groups when @c temporal_groups > 1 ); a masked
     *      patch's vector is zeroed, an unmasked one copies clean.
     *
     *  @param[in] clean_patches The frontend's normalized patch vectors @b [patches,patch_pixels] .
     *  @param[out] distorted_patches The tower's input @b [patches,patch_pixels] — clean or zeroed.
     */
    [[nodiscard]] expected_t sample_and_corrupt(bf16r2_view_t clean_patches, bf16r2_span_t distorted_patches,
                                                masking_recipe_t recipe, epoch_t epoch, gpu_stream_t &stream) noexcept;

    /**
     *  @brief Reconstruct and score the masked patches — pure enqueue on @p stream .
     *
     *  @param[in] normalized The tower's final-norm patch states @b [patches,hidden] .
     *  @param[in] clean_patches The reconstruction target @b [patches,patch_pixels] .
     *  @param[out] loss_summands @b [2] Float32: @c {λ·Σ per-patch normalized-MSE, masked count} .
     *  @param[out] normalized_grad @b [patches,hidden] @c d(normalized) — the tower's backward ADDS it.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t normalized, bf16r2_view_t clean_patches,
                                        f32r1_span_t loss_summands, bf16r2_span_t normalized_grad,
                                        gpu_stream_t &stream) noexcept;

    /** @brief Append the recon weight's optimizer slot. */
    [[nodiscard]] expected_t parameters(array<differentiable_weights_t> &slots) noexcept;
    /** @brief The recon weights @b [patch_pixels·hidden] — the checkpoint enumerator's handle. */
    [[nodiscard]] nk::tensor_span<bf16_t, 1> recon_weights() noexcept { return recon_weight_.span().flatten<1>(); }

  private:
    masked_patches_config_t config_ {};
    /** @brief The device's one main-owned tensor-core session — borrowed, never created here. */
    cublaslt_session_t *session_ = nullptr;
    /** @brief The linear reconstruction head @b [patch_pixels,hidden] and its gradient buffer. */
    bf16r2_tensor_arena_t recon_weight_ {}, recon_weight_grad_ {};
    /** @brief @b [max_patches,patch_pixels] reconstructed patches and their loss gradient. */
    bf16r2_tensor_arena_t predicted_ {}, predicted_grad_ {};
    /** @brief @b [max_patches] the sampled mask (one masked, zero visible) the loss gates on. */
    u32r1_tensor_arena_t mask_ {};
    /** @brief @b [max_patches] per-patch normalized-MSE summands. */
    f32r1_tensor_arena_t position_loss_ {};
    /** @brief @b [2] @c {λ·2/(patch_pixels·max(count,1)), count} — device-computed. */
    f32r1_tensor_arena_t gradient_scale_ {};
    /** @brief The shared tensor-core workspace of the three pinned plans. */
    b8r1_tensor_arena_t workspace_ {};
    /** @brief Pinned plans: predicted, @c d(normalized) , @c d(recon) . */
    std::array<cublaslt_plan_t, 3> plans_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::core
