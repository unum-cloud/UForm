/**
 *  @file src/core/transformer_block.cuh
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The pre-norm transformer block: @ref transformer_block_bf16_t trains forward and
 *      backward, and @ref transformer_block_e4m3_t is its forward-only inference twin, made
 *      only by @ref transformer_block_bf16_t::quantize_weights .
 *
 *      x → RMSNorm → fused QKV → unit QK-norm → M-RoPE → attention → out → +x
 *        → RMSNorm → fused gate/up → SiLU(gate)·up → down → +residual
 *
 *  Every product and every attention is NumKong's: BFloat16 operands into Float32 on the
 *  training block, E4M3 on the twin. @ref transformer_block_bf16_t::pack_weights packs each
 *  weight once per optimizer step, as @c W for the forward and as @c Wᵀ for the data
 *  gradient. The backwards of the norms, the QK-norm, SwiGLU and the rotation are
 *  deterministic house kernels.
 *
 *  Tokens arrive packed in one @b [tokens,hidden] buffer with @b [segments+1] start offsets
 *  and @b [segments] lengths. The QKV product stays @b [tokens,3·hidden] , with Q, K and V as
 *  strided views into it. NumKong reads the starts as the query and the key offsets alike: a
 *  gap between segments rides along as query rows no loss reads, and its key rows get no
 *  gradient. Capacities are baked at construction, so every call is a capture-safe enqueue.
 *
 *  The residual stream stays Float32 between blocks, and each product's consumer narrows it:
 *  a norm through NumKong's @c rmscast , anything else through a plain cast. The attention
 *  keeps its Float32 output and log-sum-exp for the backward. Inference scales each weight by
 *  @c max_abs/240 into its E4M3 shadow, and each activation token by a scale of its own.
 *
 *  Rotary angles are model state in @ref rope_angles_t , and the per-token @b (t,h,w)
 *  coordinates come from the tower. Pairing is NeoX split-half, pair @c k coupling channels
 *  @b (k,k+64) with the temporal pairs first, so a text token @b (position,0,0) leaves the
 *  spatial pairs untouched.
 */
#pragma once

#include <cstddef> // `std::size_t`, `std::byte`
#include <cmath>   // `std::sqrt`, `std::pow`, and the device `rsqrtf`

#include <array> // `std::array` — the four weight packs
#include <span>  // `std::span` — host staging views

#include <cuda.h>      // `cuMemsetD8Async`, `cuMemcpyDtoDAsync`
#include <cuda_bf16.h> // `__nv_bfloat16`
#include <cuda_fp8.h>  // `__nv_fp8_e4m3`

#include <numkong/attention.h> // `nk_attention_pack_*_best`, `nk_attention_packed_*_best`, the BFloat16 gradients
#include <numkong/cast.h>      // `nk_cast_best` — Float32 norm scales and the narrowed products
#include <numkong/dots.h>      // `nk_dots_pack_bf16_best`, `nk_dots_packed_bf16_best` — every product
#include <numkong/each.h>      // `nk_each_rmscast_bf16_best`, `nk_each_swiglu_f32_best`, `nk_each_sum_f32_best`
#include <numkong/tensor.hpp>  // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "core/types.hpp"         // `expected`, `expected_t`, `status_t`, the integer aliases
#include "core/schedule.cuh"      // `gpu_stream_t`, `launch_kernel`, the NumKong products
#include "core/tower_kernels.cuh" // the rotary and E4M3 scaling bodies, `e4m3_range_k`

namespace unum::uform::core {

/** The one head width the blocks support; heads = hidden / 128. */
inline constexpr u32_t transformer_head_dim_k = 128u;

/** The softmax scale @c 1/√head_dim every attention of the blocks and the pool applies. */
[[nodiscard]] inline f32_t transformer_attention_scale() noexcept {
    return 1.0f / std::sqrt(f32_t {transformer_head_dim_k});
}

/** The baked geometry and numerics of one block — a different config is a different object. */
struct transformer_block_config_t {

    /** Embedding width; must be a multiple of @ref transformer_head_dim_k . */
    u32_t hidden = 0;

    /** Feed-forward expansion; the SwiGLU inner width is @c ffn_expansion·hidden . */
    u32_t ffn_expansion = 4u;

    /** Most packed tokens one call carries. */
    u32_t max_tokens = 0;

    /** Most packed segments (documents) one call carries. */
    u32_t max_segments = 0;

    /** Longest single segment. */
    u32_t max_segment_tokens = 0;

    /** RMSNorm variance epsilon. */
    f32_t rms_eps = 1.0e-6f;

    /** Drop probability at each residual join; 0 disables the pass. Inverted (kept activations
     *  scale by 1/(1−p)), so inference is identity and the E4M3 twin never launches it. */
    f32_t residual_dropout = 0.0f;
};

/** How @c head_dim divides across the three rotary axes, in channels (pairs × 2); the three
 *  sum to the head width. */
struct rope_shape_t {
    u32_t temporal = 64u;
    u32_t height = 32u;
    u32_t width = 32u;
};

/** Borrowed rotary state: the three per-axis precomputed angle grids — model property, shared
 *  by every block, threaded per call. Per-token coordinates travel separately as the tower's
 *  derived scratch, and the shape is never stored beside the angles: each axis's channel width is
 *  twice its grid's pair extent, so @ref shape derives it. */
struct rope_angles_view_t {

    /** @b [max_t,temporal/2] Float32 temporal angles. */
    f32r2_view_t temporal {};

    /** @b [max_h,height/2] Float32 height angles. */
    f32r2_view_t height {};

    /** @b [max_w,width/2] Float32 width angles. */
    f32r2_view_t width {};

    /** How @c head_dim divides across the axes — pure arithmetic over the grids' extents. */
    [[nodiscard]] constexpr rope_shape_t shape() const noexcept {
        rope_shape_t divided;
        divided.temporal = 2u * static_cast<u32_t>(temporal.extent(1));
        divided.height = 2u * static_cast<u32_t>(height.extent(1));
        divided.width = 2u * static_cast<u32_t>(width.extent(1));
        return divided;
    }
};

/** The owned rotary state: the three per-axis angle grids, precomputed on the host once and
 *  uploaded — model property, shared by every tower call through @ref view . */
struct rope_angles_t {

    /** Precompute @c angle[position,pair] = position·base^(−2·pair/denominator) per axis and
     *  upload — once, at setup, outside any capture. The temporal denominator is the whole head
     *  width (the shape's sum), so text-degenerate coordinates match 1-D RoPE bit for bit. */
    [[nodiscard]] static expected<rope_angles_t> make(gpu_arena_t &arena, u32_t max_positions, rope_shape_t shape,
                                                      f32_t base) noexcept;

    /** The kernel-facing bundle every block borrows. */
    [[nodiscard]] rope_angles_view_t view() const noexcept {
        rope_angles_view_t bundle;
        bundle.temporal = f32r2_view_t {angles_.data() + axis_offsets_[0], max_positions_, shape_.temporal / 2u};
        bundle.height = f32r2_view_t {angles_.data() + axis_offsets_[1], max_positions_, shape_.height / 2u};
        bundle.width = f32r2_view_t {angles_.data() + axis_offsets_[2], max_positions_, shape_.width / 2u};
        return bundle;
    }

  private:
    rope_shape_t shape_ {};
    u32_t max_positions_ = 0;
    std::size_t axis_offsets_[3] = {};
    f32r1_tensor_arena_t angles_ {};
};

#if defined(__CUDACC__)

/** The conversions the training kernels hand the bodies of @ref tower_kernels.cuh . */
struct tower_scalars_t {
    __host__ __device__ f32_t widen(bf16_t value) const noexcept { return __bfloat162float(value); }
    __host__ __device__ f32_t widen(f32_t value) const noexcept { return value; }
    __host__ __device__ bf16_t narrow(f32_t value) const noexcept { return __float2bfloat16_rn(value); }
    __host__ __device__ e4m3_t narrow_e4m3(f32_t value) const noexcept { return __nv_fp8_e4m3 {value}; }

    /** A template, so the host pass never instantiates the device-only @c __sincosf . */
    template <typename angle_type_>
    __host__ __device__ void sincos(angle_type_ angle, f32_t *sine, f32_t *cosine) const noexcept {
        __sincosf(angle, sine, cosine);
    }
};

/**
 *  @brief Learnable RMSNorm forward over strided BFloat16 rows: @c y = x·rsqrt(mean(x²)+eps)·γ , saving the
 *      Float32 @c 1/rms per row for the backward.
 *
 *  The @c rmsnorm_* and @c rmscast_* kernels carry @b external linkage so both this block and the encoder tower
 *  launch the same symbols through @ref launch_kernel : one definition each, no wrapper, no extra file. RMSNorm
 *  is a shared primitive, so the names take no @c transformer_ prefix; the block's unit QK-norm is a different
 *  operator (no learnable scale, per-head-vector) and stays file-local.
 *
 *  __grid_size__ @c [rows,1,1] ; one CTA per row; threads stripe the row's channels; no dynamic shared
 *  memory.
 */
__global__ void rmsnorm_forward_kernel(bf16r2_view_t input, bf16r1_view_t scale, bf16r2_span_t output,
                                       f32r1_span_t inverse_rms, f32_t epsilon);

/**
 *  @brief Learnable RMSNorm backward for the row gradient:
 *      @c dx = (dy·γ − x̂·mean(dy·γ·x̂))·(1/rms) with @c x̂ = x·(1/rms) , written to
 *      @p input_grad , and the band's scale-gradient partial @c Σ_band dy·x̂ into row
 *      @c blockIdx.x of @p scale_grad_partials .
 *
 *  __grid_size__ @c [⌈rows/block⌉,1,1] ; each CTA owns a band of @c block rows, one warp per row,
 *  then one thread per channel folds the band's rows in order; no dynamic shared memory.
 */
__global__ void rmsnorm_backward_kernel(bf16r2_view_t input, bf16r1_view_t scale, f32r1_view_t inverse_rms,
                                        bf16r2_view_t output_grad, bf16r2_span_t input_grad,
                                        f32r2_span_t scale_grad_partials);

/**
 *  @brief The scale gradient of either norm, @c dγ[c] = Σ_band partials[band,c] , folded in a fixed
 *      order over the @b [bands,width] partials the row backward wrote.
 *
 *  __grid_size__ @c [⌈width/32⌉,1,1] ; warp @c w sums every eighth band from @c w , then the CTA
 *  adds its warps in order; no dynamic shared memory.
 */
__global__ void rmsnorm_scale_grad_kernel(f32r2_view_t scale_grad_partials, bf16r1_span_t scale_grad);

/**
 *  @brief The row-gradient backward of NumKong's rmscast over the Float32 residual stream:
 *      @c dx = residual + (dy·γ − x̂·mean(dy·γ·x̂))·(1/rms) , recomputing @c 1/rms from @p input ,
 *      and the band's scale-gradient partial into row @c blockIdx.x of @p scale_grad_partials for
 *      @ref rmsnorm_scale_grad_kernel ; an empty @p residual_grad adds nothing.
 *
 *  __grid_size__ @c [⌈rows/block⌉,1,1] ; each CTA owns a band of @c block rows, one warp per row,
 *  then one thread per channel folds the band's rows in order; no dynamic shared memory.
 */
__global__ void rmscast_backward_kernel(f32r2_view_t input, bf16r1_view_t scale, f32r2_view_t output_grad,
                                        f32r2_view_t residual_grad, f32r2_span_t input_grad,
                                        f32r2_span_t scale_grad_partials, f32_t epsilon);

/**
 *  @brief Inverted dropout, in place and keyed — declared here with @b external linkage (like the
 *      @c rmsnorm_* family) so the block, the tower, and any future consumer launch the one shared symbol.
 *
 *  A single op serves both directions: forward runs it on the activations, backward on the same site's
 *  branch gradient, and because the keep decision is re-derived from @p salt and the device @p data_epoch it
 *  needs no stored mask — the two launches agree by construction. Keep with prob @c 1−p (a keyed draw
 *  @c ≥ p ), scale kept values by @c 1/(1−p) , zero the rest; @p data_epoch is read from device memory so a
 *  captured step re-keys per replay. Launch only when @c drop_probability > 0 .
 *
 *  __grid_size__ grid-stride over the flattened extent; no dynamic shared memory.
 */
__global__ void dropout_kernel(bf16r1_span_t values, u32r1_view_t data_epoch, u32_t salt, f32_t drop_probability);

/** The transients every block of a stack shares: blocks run one after another on one stream and no
 *  transient outlives the block that wrote it, so a stack's footprint is one scratch plus the per-block
 *  stashes, weights and packs. Owned by the stack, borrowed by each @ref transformer_block_bf16_t . */
struct transformer_scratch_t {

    /** Allocate every transient off @p arena for @p config 's capacities, wide enough for the
     *  weight gradients of every entry projecting one of @p entry_inputs scalars per token, of the
     *  tower head over the segments, and of a token head of @p token_width outputs when nonzero;
     *  the NumKong pack and workspace sizes come from the capabilities of the arena's device. */
    [[nodiscard]] static expected<transformer_scratch_t> make(gpu_arena_t &arena,
                                                              transformer_block_config_t const &config,
                                                              std::span<u32_t const> entry_inputs,
                                                              u32_t token_width) noexcept;

    /** Float32 @b [tokens·ffn] the SwiGLU output before its cast, then each data gradient before the
     *  house kernel or cast consuming it. */
    f32r1_tensor_arena_t products {};

    /** Float32 @b [tokens,hidden] the residual-stream gradient between the two norm backwards. */
    f32r2_tensor_arena_t residual_grad {};

    /** BFloat16 @b [tokens·max(3·hidden,2·ffn)] each product's output gradient narrowed into the A
     *  operand of its data and weight gradients. */
    bf16r1_tensor_arena_t narrowed_grad {};

    /** BFloat16 @b [max(3·hidden,2·ffn),tokens] @c dYᵀ with token rows padded to 8, which keeps
     *  NumKong's 16-byte stride rule for any token count. */
    bf16r1_tensor_arena_t transposed_grad {};

    /** BFloat16 @b [max(hidden,ffn),tokens] @c Xᵀ with token rows padded to 8. */
    bf16r1_tensor_arena_t transposed_input {};

    /** BFloat16 @b [max(3·hidden,2·ffn)·hidden] the one weight @c pack_weights transposes at a time. */
    bf16r1_tensor_arena_t transposed_weight {};

    /** @b [tokens,3·hidden] the backward's Q|K|V gradient, narrowed from @ref qkv_grad . */
    bf16r2_tensor_arena_t packed_qkv {};

    /** Float32 @b [tokens,3·hidden] the attention's dQ|dK|dV, zeroed before each backward, and
     *  outside the attention backward the @b [⌈tokens/block⌉,hidden] partials of each norm's
     *  scale gradient. */
    f32r2_tensor_arena_t qkv_grad {};

    /** The keys and values of the block in flight in NumKong's attention layout, packed by its
     *  forward and again by its backward. */
    b8r1_tensor_arena_t packed_keys_values {};

    /** Float32 @b [max(3·hidden,2·ffn)·hidden] each weight gradient before its cast. */
    f32r1_tensor_arena_t weight_grad_products {};

    /** The NumKong pack of @c Xᵀ , sized for the largest. */
    b8r1_tensor_arena_t packed_input {};

    /** The ordered depth slices of each weight gradient's product, sized for the most any of them
     *  takes. */
    b8r1_tensor_arena_t weight_gradient_workspace {};
};

/** The training transformer block: BFloat16 products over a Float32 residual stream, forward and
 *  backward, with every weight, gradient and stash baked at @ref make . */
struct transformer_block_bf16_t {

    /** Allocate the weights and their gradients, every stash the backward reads, and the weight
     *  packs; construction validates the config, and every call after it is a pure enqueue. */
    [[nodiscard]] static expected<transformer_block_bf16_t> make(gpu_arena_t &arena, transformer_scratch_t &scratch,
                                                                 transformer_block_config_t config) noexcept;

    /** Pack the four weights as the forward's B operands and their transposes as the data gradients' —
     *  pure enqueue, capture-safe. Call once per optimizer step, after the weights change and before the
     *  first @ref forward ; every micro-batch and call of the step reuses the packs. */
    [[nodiscard]] expected_t pack_weights(gpu_stream_t &stream) noexcept;

    /**
     *  @brief One block forward — pure enqueue, capture-safe; stashes everything the backward reads.
     *
     *  @param[in] input Packed @b [tokens,hidden] Float32 residual stream; @c tokens<=max_tokens .
     *  @param[out] output Same-shaped Float32 destination; may not alias @p input .
     *  @param[in] segment_offsets @b [segments+1] flat segment START positions (gapped at row jumps).
     *  @param[in] segment_lengths @b [segments] live token count per segment; PAD segments carry 0.
     *  @param[in] angles Borrowed rotary angle grids.
     *  @param[in] coords @b [tokens,3] the tower-derived per-token @b (t,h,w) coordinates.
     */
    [[nodiscard]] expected_t forward(f32r2_view_t input, f32r2_span_t output, u32r1_view_t segment_offsets,
                                     u32r1_view_t segment_lengths, rope_angles_view_t angles, i16r2_view_t coords,
                                     gpu_stream_t &stream) noexcept;

    /** One block backward, consuming the stashes @ref forward wrote — pure enqueue, capture-safe.
     *  Weight gradients overwrite this block's buffers; @p input_grad receives the full upstream Float32
     *  token gradient and may not alias @p output_grad . */
    [[nodiscard]] expected_t backward(f32r2_view_t output_grad, f32r2_span_t input_grad, u32r1_view_t segment_offsets,
                                      u32r1_view_t segment_lengths, rope_angles_view_t angles, i16r2_view_t coords,
                                      gpu_stream_t &stream) noexcept;

    /** The baked geometry. */
    [[nodiscard]] transformer_block_config_t const &config() const noexcept { return config_; }

    /** The four matmul weights, flattened, in declaration order — the mutable optimizer / IO
     *  boundary (initialization, checkpoint loads, updates). */
    [[nodiscard]] bf16r1_span_t qkv_weight() noexcept { return qkv_weight_.span().flatten<1>(); }
    [[nodiscard]] bf16r1_span_t out_weight() noexcept { return out_weight_.span().flatten<1>(); }
    [[nodiscard]] bf16r1_span_t gate_up_weight() noexcept { return gate_up_weight_.span().flatten<1>(); }
    [[nodiscard]] bf16r1_span_t down_weight() noexcept { return down_weight_.span().flatten<1>(); }

    /** The two RMSNorm scales — same boundary. */
    [[nodiscard]] bf16r1_span_t pre_norm_scale() noexcept { return pre_norm_scale_.span(); }
    [[nodiscard]] bf16r1_span_t post_norm_scale() noexcept { return post_norm_scale_.span(); }

    /** The per-block gradient buffers the optimizer reduces — written by @ref backward with
     *  @c beta=0 , mirroring the weight accessors one-for-one. */
    [[nodiscard]] bf16r1_view_t qkv_weight_grad() const noexcept { return qkv_weight_grad_.view().flatten<1>(); }
    [[nodiscard]] bf16r1_view_t out_weight_grad() const noexcept { return out_weight_grad_.view().flatten<1>(); }
    [[nodiscard]] bf16r1_view_t gate_up_weight_grad() const noexcept {
        return gate_up_weight_grad_.view().flatten<1>();
    }
    [[nodiscard]] bf16r1_view_t down_weight_grad() const noexcept { return down_weight_grad_.view().flatten<1>(); }
    [[nodiscard]] bf16r1_view_t pre_norm_scale_grad() const noexcept { return pre_norm_scale_grad_.view(); }
    [[nodiscard]] bf16r1_view_t post_norm_scale_grad() const noexcept { return post_norm_scale_grad_.view(); }

    /** Quantize this block's weights into the E4M3 inference twin — the twin's only constructor. */
    [[nodiscard]] expected<struct transformer_block_e4m3_t> quantize_weights(gpu_arena_t &arena,
                                                                             gpu_stream_t &stream) noexcept;

  private:
    friend struct transformer_block_e4m3_t;

    /** Pack this block's stashed keys and values into the scratch's @c packed_keys_values . */
    [[nodiscard]] expected_t pack_keys_values_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                               gpu_stream_t &stream) noexcept;

    /** The baked geometry. */
    transformer_block_config_t config_ {};

    /** The NumKong capabilities of the arena's device, picking every product and element-wise verb. */
    nk_capability_t capabilities_ = 0;

    /** The shared transients — borrowed, owned by the stack. */
    transformer_scratch_t *scratch_ = nullptr;

    /** Product roles, indexing @ref packed_weights_ and @ref packed_weights_transposed_ . */
    static constexpr std::size_t product_qkv_k = 0, product_out_k = 1, product_gate_up_k = 2, product_down_k = 3;

    /** The four learnable weight buffers: fused QKV, out-projection, fused gate/up, down —
     *  @b [out,in] row-major each. */
    bf16r2_tensor_arena_t qkv_weight_ {}, out_weight_ {}, gate_up_weight_ {}, down_weight_ {};

    /** The two RMSNorm scale vectors @c γ , pre-attention and post-attention. */
    bf16r1_tensor_arena_t pre_norm_scale_ {}, post_norm_scale_ {};

    /** @b [2·hidden] both scales widened to the Float32 @c γ NumKong's rmscast reads, each forward. */
    f32r1_tensor_arena_t widened_norm_scales_ {};

    /** The weight gradients, one per weight buffer, overwritten each backward. */
    bf16r2_tensor_arena_t qkv_weight_grad_ {}, out_weight_grad_ {}, gate_up_weight_grad_ {}, down_weight_grad_ {};

    /** The two norm-scale gradients. */
    bf16r1_tensor_arena_t pre_norm_scale_grad_ {}, post_norm_scale_grad_ {};

    /** Float32 forward stashes the backward reads: the residual-stream input, the QKV product holding the
     *  pre-norm Q and K, the attention residual, and the gate/up product, all @b [tokens,·] and resized to
     *  the step's live token count each forward. */
    f32r2_tensor_arena_t input_stash_ {}, qkv_products_ {}, attn_residual_ {}, gate_up_products_ {};

    /** BFloat16 forward stashes, every product's A operand: the pre-attention norm output, the
     *  attention output, the post-attention norm output, and SiLU·up. */
    bf16r2_tensor_arena_t normalized_ {}, attn_output_ {}, mlp_normalized_ {}, silu_up_ {};

    /** @b [tokens,3·hidden] the normalized and rotated Q|K|V the attention reads, and its backward
     *  again. */
    bf16r2_tensor_arena_t qkv_ {};

    /** Float32 @b [tokens,hidden] the attention output before its cast, read by its backward. */
    f32r2_tensor_arena_t attention_products_ {};

    /** @b [tokens·heads] the attention's log-sum-exp, the backward's weights recomputed from it. */
    f32r1_tensor_arena_t softmax_log_sum_exp_ {};

    /** Each weight @b [out,in] and its transpose @b [in,out] in NumKong's packed layout, refreshed by
     *  @ref pack_weights : the forward's and the data gradients' B operands. */
    std::array<b8r1_tensor_arena_t, 4> packed_weights_ {}, packed_weights_transposed_ {};
};

/** The quantized forward-only inference twin: E4M3 weights packed once for NumKong's products,
 *  per-token activation scales, and NumKong's E4M3 attention — @b no gradients, @b no stashes,
 *  roughly a quarter of the training block's memory. Constructed only through
 *  @ref transformer_block_bf16_t::quantize_weights , so an instance is always fully quantized. */
struct transformer_block_e4m3_t {

    /** One quantized block forward — pure enqueue over the packed multi-segment layout. Norms and
     *  rotation run BFloat16; matmuls run E4M3 with @b per-token activation scales (folded back
     *  after each product); attention runs unscaled E4M3 over the ragged segments. */
    [[nodiscard]] expected_t forward(bf16r2_view_t input, bf16r2_span_t output, u32r1_view_t segment_offsets,
                                     u32r1_view_t segment_lengths, rope_angles_view_t angles, i16r2_view_t coords,
                                     gpu_stream_t &stream) noexcept;

    /** The baked geometry. */
    [[nodiscard]] transformer_block_config_t const &config() const noexcept { return config_; }

  private:
    friend struct transformer_block_bf16_t;

    /** Product roles, indexing @ref packed_weights_ and @ref weight_descales_ alike. */
    static constexpr std::size_t product_qkv_k = 0, product_out_k = 1, product_gate_up_k = 2, product_down_k = 3;

    /**
     *  @brief One E4M3 projection: @c activations·Wᵀ into the Float32 @ref products_ , then one pass folding
     *      @p token_scales and the weight descale back and adding @p residual into the BFloat16 @p values .
     *  @param[in] product The role, one of the @c product_*_k constants.
     *  @param[in] activations @b [tokens,in] the per-token-quantized operand.
     *  @param[in] token_scales @b [tokens] per-token scales, or empty when the operand carries none.
     *  @param[in] residual @b [tokens,out] added after the scales, or empty.
     *  @param[out] values @b [tokens,out] the BFloat16 result.
     *  @param[inout] stream The stream every launch enqueues on.
     */
    [[nodiscard]] expected_t project_(std::size_t product, e4m3r2_view_t activations, f32r1_view_t token_scales,
                                      bf16r2_view_t residual, bf16r2_span_t values, gpu_stream_t &stream) noexcept;

    /** The baked geometry. */
    transformer_block_config_t config_ {};

    /** The NumKong capabilities of the device the twin was quantized on, picking every product. */
    nk_capability_t capabilities_ = 0;

    /** The four E4M3 shadow weights: fused QKV, out-projection, fused gate/up, down —
     *  @b [out,in] row-major each, the source of @ref packed_weights_ . */
    e4m3r2_tensor_arena_t qkv_weight_e4m3_ {}, out_weight_e4m3_ {}, gate_up_weight_e4m3_ {}, down_weight_e4m3_ {};

    /** The four shadow weights in NumKong's packed layout, sized by its pack-size query. */
    std::array<b8r1_tensor_arena_t, 4> packed_weights_ {};

    /** @b [4] Float32 weight descales, folded after each product: qkv, out, gate_up, down. */
    f32r1_tensor_arena_t weight_descales_ {};

    /** @b [tokens,max(3·hidden,2·ffn)] the Float32 output of the projection or attention in flight. */
    f32r2_tensor_arena_t products_ {};

    /** The two RMSNorm scale vectors @c γ , widened from the training block's. */
    f32r1_tensor_arena_t pre_norm_scale_ {}, post_norm_scale_ {};

    /** The BFloat16 activation ping-pong, sized once — no stashes and no gradients on this twin;
     *  all @b [tokens,·] . */
    bf16r2_tensor_arena_t normalized_ {}, qkv_ {}, attn_residual_ {}, gate_up_ {}, silu_up_ {};

    /** The matmul operands ride per-token scales, @b [tokens,·] like their BFloat16 sources. */
    e4m3r2_tensor_arena_t normalized_e4m3_ {}, attn_output_e4m3_ {}, silu_up_e4m3_ {};

    /** @b [tokens,3·hidden] the packed Q|K|V, cast unscaled. */
    e4m3r2_tensor_arena_t qkv_e4m3_ {};

    /** The K and V slices of @ref qkv_e4m3_ in NumKong's attention layout, sized for
     *  @c max_tokens over @c max_segments . */
    b8r1_tensor_arena_t packed_keys_values_ {};

    /** The per-token activation scales the matmul fold reads. */
    f32r1_tensor_arena_t token_scales_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::core
