/**
 *  @brief The pre-norm transformer block as two function objects: @ref transformer_block_bf16_t with full
 *      forward + backward for training, and @ref transformer_block_e4m3_t , its quantized forward-only
 *      inference twin — constructed @b only through @ref transformer_block_bf16_t::quantize_weights .
 *  @file cuda_core/transformer_block.cuh
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  One block computes, in order (all norms are RMSNorm, activation is SwiGLU, @c head_dim is fixed at
 *  @ref transformer_head_dim_k ):
 *
 *      x → RMSNorm(γ_pre) → fused QKV matmul → unit QK-norm → M-RoPE → SDPA → out matmul → +x
 *        → RMSNorm(γ_post) → fused gate/up matmul → SiLU(gate)·up → down matmul → +residual
 *
 *  @section transformer_backends Backends — the fast primitive for every op
 *
 *      - Every matmul is a pinned @ref cublaslt_plan_t tensor-core product — BFloat16 operands with
 *        Float32 accumulation on the training block, E4M3 operands with pinned native descale pointers
 *        on the inference block. The QKV and gate/up projections are @b fused single products.
 *      - Attention is cuDNN's fused SDPA through the frontend graph DSL — the flash-attention-class
 *        kernel, built once per block at construction with the ragged (packed-segments) layout, the
 *        BFloat16 backward consuming the stashed @c softmax_log_sum_exp_ .
 *      - Everything else — RMSNorm forward / backward, the non-learnable unit QK-norm, M-RoPE rotation
 *        and its inverse, SiLU·up and its backward, residual adds, and the E4M3 quantisation — is a
 *        deterministic house kernel launched through @ref launch_kernel .
 *
 *  @section transformer_shapes Shapes, packing, capture
 *
 *  Tokens are @b packed: the block sees one @b [tokens,hidden] buffer and a @c cu_seqlens prefix sum
 *  @b [segments+1] (plus per-segment lengths @b [segments] ) describing document boundaries; SDPA runs
 *  the ragged THD path with block-diagonal masking, so no @b O(seq²) bias ever materialises. The fused
 *  QKV output stays packed @b [tokens,3·hidden] — Q / K / V are strided views into it (token stride
 *  @c 3·hidden ), consumed in place by the norm / rotation kernels and by SDPA's strided tensors, so no
 *  reshuffle kernel exists. All capacities ( @c max_tokens , @c max_segments , @c max_segment_tokens )
 *  are baked at construction: plans, graphs, and every owned tensor are capture-stable, a call is a pure
 *  enqueue, and a different shape is a different object.
 *
 *  @section transformer_precision Precision recipe
 *
 *  Training is BFloat16 end-to-end with Float32 accumulation in every matmul and every reduction; the
 *  RMSNorm statistics ( @c 1/rms ) and the SDPA @c softmax_log_sum_exp_ stash in Float32. Inference quantisation
 *  is per-tensor @b static E4M3: each weight is scaled by @c max_abs/240 into its shadow, the Float32
 *  descale scalar rides the matmul descriptor's native scale pointer (no post-matmul descale kernel),
 *  and activations quantise scale-free — the norms bound their range. Weight gradients are written with
 *  @c beta=0 into per-block BFloat16 buffers; cross-microbatch accumulation is the optimizer's concern.
 *
 *  @section transformer_rope Rotary position embedding
 *
 *  The precomputed angle grids are model state owned by @ref rope_angles_t and shared by every block;
 *  per-token
 *  @b (t,h,w) coordinates are the tower's derived scratch, recomputed each forward from the batch
 *  geometry and borrowed per call as a plain view. The channel-pair convention and the per-axis
 *  shape follow the storage invariant of the angle grids (temporal pairs first, then height, then
 *  width); a text token @b (position,0,0) leaves the spatial pairs at identity.
 *
 *  @section transformer_ragged The ragged boundary
 *
 *  Segments arrive as the batch's flat @c u32 START positions (gapped where a segment would cross a
 *  row) plus stored lengths. cuDNN's variable-sequence SDPA takes gapped element offsets directly —
 *  batch stride 0, the ragged prefix supplies each segment's base — but hard-requires Int32, so ONE
 *  kernel ( @ref transformer_ragged_offsets_kernel_ ) converts starts and lengths into block-owned
 *  Int32 staging; nothing else in the stack ever sees Int32.
 */
#pragma once

#include <array>   // `std::array` — the fixed-size pinned-plan tables
#include <cmath>   // `sqrtf`, `rsqrtf`, `fabsf`, `expf`, `std::pow`
#include <cstddef> // `std::size_t`, `std::byte`
#include <span>    // `std::span` — host staging views

#include <cuda.h>      // `cuMemsetD8Async`, `cuMemcpyDtoDAsync`
#include <cuda_bf16.h> // `__nv_bfloat16`
#include <cuda_fp8.h>  // `__nv_fp8_e4m3`

#include <unordered_map>    // the frontend-dispatch fallback map (SDPA backward only), required by cuDNN
#include <cudnn_frontend.h> // the SDPA graph builder; execution is raw backend wherever provably exact

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "types.cuh"    // `expected`, `expected_t`, `status_t`, the integer aliases
#include "schedule.cuh" // `gpu_stream_t`, `launch_kernel`, `cublaslt_*`, `cudnn_session_t`

namespace unum::uform::core {

using i16r2_view_t = nk::tensor_view<i16_t, 2>;
using i32r1_view_t = nk::tensor_view<i32_t, 1>;
using i32r1_span_t = nk::tensor_span<i32_t, 1>;
using bf16r2_view_t = nk::tensor_view<__nv_bfloat16, 2>;
using bf16r2_span_t = nk::tensor_span<__nv_bfloat16, 2>;
using bf16r1_span_t = nk::tensor_span<__nv_bfloat16, 1>;
using bf16r1_view_t = nk::tensor_view<__nv_bfloat16, 1>;
using e4m3r1_tensor_arena_t = nk::tensor<e4m3_t, gpu_arena_alloc<e4m3_t>, 1>;
using bf16r1_tensor_arena_t = nk::tensor<bf16_t, gpu_arena_alloc<bf16_t>, 1>;
using i32r1_tensor_arena_t = nk::tensor<i32_t, gpu_arena_alloc<i32_t>, 1>;

/** @brief The one head width every SDPA engine tier supports here; heads = hidden / 128. */
inline constexpr u32_t transformer_head_dim_k = 128u;
/** @brief E4M3's largest normal magnitude — the quantisation range max_abs maps onto. */
inline constexpr f32_t e4m3_range_k = 240.0f;
/** @brief The descale floor keeping a zero tensor representable — @c max_abs/240 clamps to this. */
inline constexpr f32_t e4m3_descale_floor_k = 1.0e-12f;

/**
 *  @brief The baked geometry and numerics of one block — a different config is a different object.
 */
struct transformer_block_config_t {
    /** @brief Embedding width; must be a multiple of @ref transformer_head_dim_k . */
    u32_t hidden = 0;
    /** @brief Feed-forward expansion; the SwiGLU inner width is @c ffn_expansion·hidden . */
    u32_t ffn_expansion = 4u;
    /** @brief Most packed tokens one call carries. */
    u32_t max_tokens = 0;
    /** @brief Most packed segments (documents) one call carries — SDPA's logical batch. */
    u32_t max_segments = 0;
    /** @brief Longest single segment — SDPA's logical sequence extent. */
    u32_t max_segment_tokens = 0;
    /** @brief RMSNorm variance epsilon. */
    f32_t rms_eps = 1.0e-6f;
};

/** @brief How @c head_dim divides across the three rotary axes, in channels (pairs × 2); the three
 *      sum to the head width. */
struct rope_shape_t {
    u32_t temporal = 64u;
    u32_t height = 32u;
    u32_t width = 32u;
};

/** @brief Borrowed rotary state: the three per-axis precomputed angle grids — model property, shared
 *      by every block, threaded per call. Per-token coordinates travel separately as the tower's
 *      derived scratch, and the shape is never stored beside the angles: each axis's channel width is
 *      twice its grid's pair extent, so @ref shape derives it. */
struct rope_angles_view_t {
    /** @brief @b [max_t,temporal/2] Float32 temporal angles. */
    f32r2_view_t temporal {};
    /** @brief @b [max_h,height/2] Float32 height angles. */
    f32r2_view_t height {};
    /** @brief @b [max_w,width/2] Float32 width angles. */
    f32r2_view_t width {};

    /** @brief How @c head_dim divides across the axes — pure arithmetic over the grids' extents. */
    [[nodiscard]] constexpr rope_shape_t shape() const noexcept {
        rope_shape_t divided;
        divided.temporal = 2u * static_cast<u32_t>(temporal.extent(1));
        divided.height = 2u * static_cast<u32_t>(height.extent(1));
        divided.width = 2u * static_cast<u32_t>(width.extent(1));
        return divided;
    }
};

/**
 *  @brief The owned rotary state: the three per-axis angle grids, precomputed on the host once and
 *      uploaded — model property, shared by every tower call through @ref view .
 */
struct rope_angles_t {

    /** @brief Precompute @c angle[position,pair] = position·base^(−2·pair/denominator) per axis and
     *      upload — once, at setup, outside any capture. The temporal denominator is the whole head
     *      width (the shape's sum), so text-degenerate coordinates match 1-D RoPE bit for bit. */
    [[nodiscard]] static expected<rope_angles_t> make(gpu_arena_t &arena, u32_t max_positions, rope_shape_t shape,
                                                      f32_t base) noexcept;

    /** @brief The kernel-facing bundle every block borrows. */
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

namespace fe = cudnn_frontend;

// Variant-pack UIDs mirrored across the forward and backward SDPA graphs.
inline constexpr i64_t sdpa_uid_query_k = 1, sdpa_uid_key_k = 2, sdpa_uid_value_k = 3, sdpa_uid_output_k = 4;
inline constexpr i64_t sdpa_uid_stats_k = 5, sdpa_uid_seq_len_q_k = 8, sdpa_uid_seq_len_kv_k = 9;
inline constexpr i64_t sdpa_uid_attn_scale_k = 10, sdpa_uid_output_grad_k = 101, sdpa_uid_query_grad_k = 102;
inline constexpr i64_t sdpa_uid_key_grad_k = 103, sdpa_uid_value_grad_k = 104;
inline constexpr i64_t sdpa_uid_ragged_base_k = 20; // 20..27: Q,K,V,O,dO,dQ,dK,dV ragged prefix offsets.
inline constexpr i64_t sdpa_uid_descale_q_k = 11, sdpa_uid_descale_k_k = 12, sdpa_uid_descale_v_k = 13;
inline constexpr i64_t sdpa_uid_descale_s_k = 14, sdpa_uid_scale_s_k = 15, sdpa_uid_scale_o_k = 16;
inline constexpr i64_t sdpa_uid_amax_s_k = 17, sdpa_uid_amax_o_k = 18;

/** @brief Most tensors any of the SDPA variant packs binds — the fixed capacity of the binding table. */
inline constexpr u32_t sdpa_binding_capacity_k = 20u;

/**
 *  @brief The cudnn-frontend graph with the two protected boundaries exposed that raw execution needs:
 *      the pinned backend execution plan, and the uid set its variant pack must bind. The frontend remains
 *      the graph @b builder (construction-time only); execution goes through the backend C API with
 *      plain arrays — no maps, no vectors, no shared pointers on the dispatch path.
 */
struct sdpa_graph_t final : fe::graph::Graph {

    /** @brief The pinned backend execution plan — the first argument of @c cudnnBackendExecute . */
    [[nodiscard]] cudnnBackendDescriptor_t raw_plan() const noexcept {
        return plans.execution_plans[plans.candidate]->get_raw_desc();
    }

    /** @brief The pinned plan's own workspace appetite (the backend's, with no frontend prefix). */
    [[nodiscard]] i64_t plan_workspace_bytes() const noexcept {
        return plans.execution_plans[plans.candidate]->getWorkspaceSize();
    }

    /**
     *  @brief Prove the raw execution path is exact for this graph — at construction, never at dispatch:
     *      the fixed uid table must cover the variant-pack uid set precisely (a pass-by-value scalar or
     *      an internal workspace tensor would appear here as an unexpected uid), and the frontend must
     *      claim no workspace prefix of its own (its auxiliary-kernel boundary).
     */
    [[nodiscard]] expected_t verify_raw_bindings(i64_t const *uids, u32_t count) noexcept;
};

/**
 *  @brief One built SDPA plan: the graph (held by value) plus the fixed uid / pointer tables its raw
 *      variant pack binds. Owned pointers are pinned once at construction; per-call pointers are plain
 *      indexed stores — the dispatch path allocates nothing and touches no std container.
 */
struct sdpa_plan_t {
    sdpa_plan_t() noexcept = default;
    // The frontend graph's move is not marked noexcept (vendor containers), but it never throws in
    // practice; pinning noexcept here keeps every owner relocatable in `array<...>` — a throw would
    // terminate, which is this codebase's contract anyway.
    sdpa_plan_t(sdpa_plan_t &&) noexcept = default;
    sdpa_plan_t &operator=(sdpa_plan_t &&) noexcept = default;
    sdpa_plan_t(sdpa_plan_t const &) = delete;
    sdpa_plan_t &operator=(sdpa_plan_t const &) = delete;

    sdpa_graph_t graph {};
    std::array<i64_t, sdpa_binding_capacity_k> uids {};
    std::array<void *, sdpa_binding_capacity_k> pointers {};
    u32_t binding_count = 0;
    i64_t workspace_bytes = 0;
    /** @brief Frontend-dispatch fallback for graphs whose variant pack carries frontend-internal
     *      workspace tensors the raw path provably cannot bind (SDPA backward's dQ-accumulator family) —
     *      populated once at construction, value-updated per call, so dispatch still allocates nothing. */
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> fallback_bindings {}; // vendor-boundary
    bool raw = true;

    /** @brief Append one binding at construction; returns its pointer-table slot for per-call stores. */
    u32_t bind(i64_t uid, void *pointer = nullptr) noexcept {
        uids[binding_count] = uid;
        pointers[binding_count] = pointer;
        return binding_count++;
    }

    /**
     *  @brief Prove the raw path exact, or arm the frontend fallback: when @ref
     *      sdpa_graph_t::verify_raw_bindings rejects the graph (internal workspace tensors), the map is
     *      pre-populated so per-call dispatch stays allocation-free, and @ref workspace_bytes widens to
     *      the frontend's total (its prefix plus the plan's).
     */
    [[nodiscard]] expected_t finalize_dispatch() noexcept {
        expected_t exact = graph.verify_raw_bindings(uids.data(), binding_count);
        raw = static_cast<bool>(exact);
        if (raw) {
            workspace_bytes = graph.plan_workspace_bytes();
            return {};
        }
        fallback_bindings.reserve(binding_count);
        for (u32_t slot = 0; slot != binding_count; ++slot) fallback_bindings.emplace(uids[slot], pointers[slot]);
        return graph.get_workspace_size(workspace_bytes).is_good()
                   ? expected_t {}
                   : expected_t::fail(status_t::cuda_error_k, "SDPA workspace query failed");
    }

    /** @brief Execute the pinned plan: one variant-pack descriptor from the plain arrays and one
     *      @c cudnnBackendExecute on the raw path; the frontend's own dispatch on the fallback path. */
    [[nodiscard]] expected_t execute(cudnnHandle_t handle, void *workspace) noexcept;
};

/**
 *  @brief The training transformer block: BFloat16 forward + backward, every tensor and plan owned —
 *      weights, gradients, stashes, matmul plans, and SDPA graphs are all baked at @ref make .
 */
struct transformer_block_bf16_t {

    /** @brief Allocate and plan everything: weights and their gradients, every stash the backward reads,
     *      twelve pinned matmul plans, the forward and backward SDPA graphs, and the shared workspace —
     *      construction is the validation and the planning; calls are pure enqueues. */
    [[nodiscard]] static expected<transformer_block_bf16_t> make(gpu_arena_t &arena, cublaslt_session_t const &blas,
                                                                 cudnn_session_t const &cudnn,
                                                                 transformer_block_config_t config) noexcept;

    /**
     *  @brief One block forward — pure enqueue, capture-safe; stashes everything the backward reads.
     *
     *  @param[in] input Packed @b [tokens,hidden] BFloat16 buffer; @c tokens<=max_tokens .
     *  @param[out] output Same-shaped destination; may not alias @p input .
     *  @param[in] segment_offsets @b [segments+1] flat segment START positions (gapped at row jumps).
     *  @param[in] segment_lengths @b [segments] live token count per segment; PAD segments carry 0.
     *  @param[in] angles Borrowed rotary angle grids.
     *  @param[in] coords @b [tokens,3] the tower-derived per-token @b (t,h,w) coordinates.
     */
    [[nodiscard]] expected_t forward(bf16r2_view_t input, bf16r2_span_t output, u32r1_view_t segment_offsets,
                                     u32r1_view_t segment_lengths, rope_angles_view_t angles, i16r2_view_t coords,
                                     gpu_stream_t const &stream) noexcept;

    /**
     *  @brief One block backward, consuming the stashes @ref forward wrote — pure enqueue, capture-safe.
     *      Weight gradients overwrite this block's buffers ( @c beta=0 ); @p input_grad receives the full
     *      upstream token gradient.
     */
    [[nodiscard]] expected_t backward(bf16r2_view_t output_grad, bf16r2_span_t input_grad, u32r1_view_t segment_offsets,
                                      u32r1_view_t segment_lengths, rope_angles_view_t angles, i16r2_view_t coords,
                                      gpu_stream_t const &stream) noexcept;

    /** @brief The baked geometry. */
    [[nodiscard]] transformer_block_config_t const &config() const noexcept { return config_; }
    /** @brief The four matmul weights, flattened, in declaration order — the mutable optimizer / IO
     *      boundary (initialization, checkpoint loads, updates). */
    [[nodiscard]] bf16r1_span_t qkv_weight() noexcept { return {qkv_weight_.data(), qkv_weight_.extent(0)}; }
    [[nodiscard]] bf16r1_span_t out_weight() noexcept { return {out_weight_.data(), out_weight_.extent(0)}; }
    [[nodiscard]] bf16r1_span_t gate_up_weight() noexcept {
        return {gate_up_weight_.data(), gate_up_weight_.extent(0)};
    }
    [[nodiscard]] bf16r1_span_t down_weight() noexcept { return {down_weight_.data(), down_weight_.extent(0)}; }
    /** @brief The two RMSNorm scales — same boundary. */
    [[nodiscard]] bf16r1_span_t pre_norm_scale() noexcept {
        return {pre_norm_scale_.data(), pre_norm_scale_.extent(0)};
    }
    [[nodiscard]] bf16r1_span_t post_norm_scale() noexcept {
        return {post_norm_scale_.data(), post_norm_scale_.extent(0)};
    }
    /** @brief The per-block gradient buffers the optimizer reduces — written by @ref backward with
     *      @c beta=0 , mirroring the weight accessors one-for-one. */
    [[nodiscard]] bf16r1_view_t qkv_weight_grad() const noexcept { return qkv_weight_grad_.view(); }
    [[nodiscard]] bf16r1_view_t out_weight_grad() const noexcept { return out_weight_grad_.view(); }
    [[nodiscard]] bf16r1_view_t gate_up_weight_grad() const noexcept { return gate_up_weight_grad_.view(); }
    [[nodiscard]] bf16r1_view_t down_weight_grad() const noexcept { return down_weight_grad_.view(); }
    [[nodiscard]] bf16r1_view_t pre_norm_scale_grad() const noexcept { return pre_norm_scale_grad_.view(); }
    [[nodiscard]] bf16r1_view_t post_norm_scale_grad() const noexcept { return post_norm_scale_grad_.view(); }

    /** @brief Quantize this block's weights into the E4M3 inference twin — the twin's only constructor. */
    [[nodiscard]] expected<struct transformer_block_e4m3_t> quantize_weights(gpu_arena_t &arena,
                                                                             gpu_stream_t const &stream) noexcept;

  private:
    friend struct transformer_block_e4m3_t;

    /** @brief Plan slots, in the order @ref plan_matmuls_ builds them. */
    static constexpr std::size_t plan_qkv_forward_k = 0, plan_out_forward_k = 1, plan_gate_up_forward_k = 2,
                                 plan_down_forward_k = 3, plan_qkv_data_grad_k = 4, plan_out_data_grad_k = 5,
                                 plan_gate_up_data_grad_k = 6, plan_down_data_grad_k = 7, plan_qkv_weight_grad_k = 8,
                                 plan_out_weight_grad_k = 9, plan_gate_up_weight_grad_k = 10,
                                 plan_down_weight_grad_k = 11;

    /** @brief Pin the twelve BFloat16 matmul plans: four forward products, four data gradients
     *      ( @c dX=dY·W ), four weight gradients ( @c dW=dYᵀ·X ), and allocate the shared workspace. */
    [[nodiscard]] expected_t plan_matmuls_(gpu_arena_t &arena) noexcept;

    /** @brief Build the ragged THD SDPA graphs — forward with stats, backward — over the packed QKV
     *      strides, and allocate their shared workspace. Built once, outside capture. */
    [[nodiscard]] expected_t build_sdpa_(gpu_arena_t &arena, gpu_arena_alloc<b8_t> b8_alloc) noexcept;

    /** @brief Fill the block-owned Int32 staging from the batch's u32 starts and lengths — the one
     *      vendor-boundary conversion; see @ref transformer_ragged_offsets_kernel_ . */
    [[nodiscard]] expected_t stage_ragged_offsets_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                   gpu_stream_t const &stream) noexcept;

    /** @brief Store the per-call pointers into the fixed binding table and raw-execute the forward
     *      plan — no map, no allocation, one backend call. Ragged slots 0..3 are Q, K, V (packed-QKV
     *      strides) and O (hidden strides). */
    [[nodiscard]] expected_t run_sdpa_forward_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                               gpu_stream_t const &stream) noexcept;

    /** @brief The backward sibling of @ref run_sdpa_forward_ — the Q / K / V gradients land in the
     *      packed @c qkv_grad_ block. Ragged slots 0..7 are Q, K, V, O, dO, dQ, dK, dV. */
    [[nodiscard]] expected_t run_sdpa_backward_(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                                gpu_stream_t const &stream) noexcept;

    /** @brief The baked geometry and the two borrowed vendor sessions — never owned here. */
    transformer_block_config_t config_ {};
    cublaslt_session_t const *blas_ = nullptr;
    cudnn_session_t const *cudnn_ = nullptr;

    /** @brief The four learnable weight buffers: fused QKV, out-projection, fused gate/up, down. */
    bf16r1_tensor_arena_t qkv_weight_ {}, out_weight_ {}, gate_up_weight_ {}, down_weight_ {};
    /** @brief The two RMSNorm scale vectors @c γ , pre-attention and post-attention. */
    bf16r1_tensor_arena_t pre_norm_scale_ {}, post_norm_scale_ {};
    /** @brief The weight gradients, one per weight buffer, written @c beta=0 each backward. */
    bf16r1_tensor_arena_t qkv_weight_grad_ {}, out_weight_grad_ {}, gate_up_weight_grad_ {}, down_weight_grad_ {};
    /** @brief The two norm-scale gradients. */
    bf16r1_tensor_arena_t pre_norm_scale_grad_ {}, post_norm_scale_grad_ {};
    /** @brief Forward stashes the backward reads: the raw input, the pre-attention norm output, the
     *      packed QKV, and the pre-QK-norm query / key slices. */
    bf16r1_tensor_arena_t input_stash_ {}, normalized_ {}, qkv_ {}, query_prenorm_ {}, key_prenorm_ {};
    /** @brief Forward stashes: the SDPA output, the attention residual, the post-norm MLP input, and
     *      the packed gate/up and SiLU·up activations. */
    bf16r1_tensor_arena_t attn_output_ {}, attn_residual_ {}, mlp_normalized_ {}, gate_up_ {}, silu_up_ {};
    /** @brief Float32 @c 1/rms stashes: the two RMSNorms (per token) and the unit QK-norms (per head
     *      vector), all recomputed-free in the backward. */
    f32r1_tensor_arena_t pre_inverse_rms_ {}, post_inverse_rms_ {}, query_inverse_rms_ {}, key_inverse_rms_ {};
    /** @brief The SDPA log-sum-exp the fused forward emits and the fused backward consumes. */
    f32r1_tensor_arena_t softmax_log_sum_exp_ {};
    /** @brief @b [1] device @c 1/sqrt(head_dim) — the SDPA scale as a bound tensor, not a baked scalar. */
    f32r1_tensor_arena_t attn_scale_ {};
    /** @brief @b [max_segments+1] element-offset tables, one per SDPA operand stride family, plus the
     *      @b [max_segments] Int32 lengths — the block-owned staging behind the vendor boundary. */
    i32r1_tensor_arena_t ragged_qkv_offsets_ {}, ragged_hidden_offsets_ {}, ragged_seq_lens_ {};
    /** @brief Backward transients: the packed QKV gradient, the two hidden-gradient ping-pong buffers,
     *      and the gate/up and SiLU·up gradients. */
    bf16r1_tensor_arena_t qkv_grad_ {}, hidden_grad_a_ {}, hidden_grad_b_ {}, gate_up_grad_ {}, silu_up_grad_ {};
    /** @brief The shared cuBLASLt and cuDNN scratch, sized to the max any pinned plan / graph needs. */
    b8r1_tensor_arena_t blas_workspace_ {}, sdpa_workspace_ {};
    /** @brief The twelve pinned products, indexed by the @c plan_*_k slot constants. */
    std::array<cublaslt_plan_t, 12> plans_ {};
    /** @brief The two SDPA plans built once at construction — forward (stats-generating) and backward. */
    sdpa_plan_t forward_sdpa_ {}, backward_sdpa_ {};
    /** @brief Pointer-table slots of the per-call bindings: the seq-length pair and the first ragged
     *      offset of each graph (the rest follow contiguously). */
    u32_t forward_seq_slot_ = 0, forward_ragged_slot_ = 0, backward_seq_slot_ = 0, backward_ragged_slot_ = 0;
};

/**
 *  @brief The quantized forward-only inference twin: E4M3 shadow weights with pinned native descale
 *      pointers, scale-free E4M3 activations, and the FP8 SDPA graph — @b no gradients, @b no stashes,
 *      roughly a quarter of the training block's memory. Constructed only through
 *      @ref transformer_block_bf16_t::quantize_weights , so an instance is always fully quantized.
 */
struct transformer_block_e4m3_t {

    /**
     *  @brief One quantized block forward — pure enqueue over the packed multi-segment layout. Norms and
     *      rotation run BFloat16; matmuls run E4M3 with @b per-token activation scales (folded back
     *      after each product); SDPA runs E4M3 dense per segment with one dynamic per-tensor scale —
     *      the engines' own granularity.
     */
    [[nodiscard]] expected_t forward(bf16r2_view_t input, bf16r2_span_t output, u32r1_view_t segment_offsets,
                                     u32r1_view_t segment_lengths, rope_angles_view_t angles, i16r2_view_t coords,
                                     gpu_stream_t const &stream) noexcept;

    /** @brief The baked geometry. */
    [[nodiscard]] transformer_block_config_t const &config() const noexcept { return config_; }

  private:
    friend struct transformer_block_bf16_t;

    /** @brief Plan slots, in the order the FP8 plans are built. */
    static constexpr std::size_t plan_qkv_k = 0, plan_out_k = 1, plan_gate_up_k = 2, plan_down_k = 3;

    /** @brief Bind and execute the FP8 SDPA graph over the dense packed operands; the Q / K / V descale
     *      is the shared dynamic scalar the pack pass just wrote. Lengths pass through the twin's own
     *      Int32 staging — the vendor boundary. */
    [[nodiscard]] expected_t run_sdpa_(u32r1_view_t segment_lengths, gpu_stream_t const &stream) noexcept;

    /** @brief The baked geometry and the two borrowed vendor sessions — never owned here. */
    transformer_block_config_t config_ {};
    cublaslt_session_t const *blas_ = nullptr;
    cudnn_session_t const *cudnn_ = nullptr;

    /** @brief The four E4M3 shadow weights: fused QKV, out-projection, fused gate/up, down. */
    e4m3r1_tensor_arena_t qkv_weight_e4m3_ {}, out_weight_e4m3_ {}, gate_up_weight_e4m3_ {}, down_weight_e4m3_ {};
    /** @brief @b [4] Float32 weight descales, pinned into the matmul descriptors: qkv, out, gate_up, down. */
    f32r1_tensor_arena_t weight_descales_ {};
    /** @brief The two RMSNorm scale vectors @c γ , copied verbatim from the training block. */
    bf16r1_tensor_arena_t pre_norm_scale_ {}, post_norm_scale_ {};
    /** @brief The BFloat16 activation ping-pong, sized once — no stashes and no gradients on this twin. */
    bf16r1_tensor_arena_t normalized_ {}, qkv_ {}, attn_residual_ {}, gate_up_ {}, silu_up_ {};
    /** @brief Throwaway pre-QK-norm stash the unit-norm kernel needs but no backward reads. */
    bf16r1_tensor_arena_t prenorm_scratch_ {};
    /** @brief The matmul operands ride per-token scales; the SDPA operands are dense zero-padded
     *      @b [segments,seq_max,·] blocks (Q|K|V, then the attention output) under one dynamic
     *      per-tensor scale — each format at its engine's own granularity. */
    e4m3r1_tensor_arena_t normalized_e4m3_ {}, qkv_e4m3_ {}, attn_dense_e4m3_ {}, attn_output_e4m3_ {},
        silu_up_e4m3_ {};
    /** @brief The per-token activation scales the matmul fold reads, and the dynamic Q/K/V descale. */
    f32r1_tensor_arena_t token_scales_ {}, qkv_descale_ {};
    /** @brief @b [1] the packed maximum-absolute bits the dynamic descale is derived from. */
    u32r1_tensor_arena_t max_bits_ {};
    /** @brief Float32 @c 1/rms scratch the norms write and discard — per token for @c inverse_rms_ ,
     *      per QK-norm head vector for @c inverse_rms_heads_ ; no backward reads either. */
    f32r1_tensor_arena_t inverse_rms_ {}, inverse_rms_heads_ {};
    /** @brief @b [1] the unit scale bound to the FP8 graph's scale slots, and the mandatory amax output. */
    f32r1_tensor_arena_t ones_ {}, absolute_max_scratch_ {};
    /** @brief @b [max_segments] Int32 padding-mask lengths — the twin's slice of the vendor boundary. */
    i32r1_tensor_arena_t ragged_seq_lens_ {};
    /** @brief The shared cuBLASLt and cuDNN scratch, sized to the max any pinned plan / graph needs. */
    b8r1_tensor_arena_t blas_workspace_ {}, sdpa_workspace_ {};
    /** @brief Pointer-table slot of the per-call seq-length pair. */
    u32_t seq_slot_ = 0;
    /** @brief The four pinned FP8 products, indexed by the @c plan_*_k slot constants. */
    std::array<cublaslt_plan_t, 4> plans_ {};
    /** @brief The FP8 SDPA plan built once at construction. */
    sdpa_plan_t sdpa_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::core
