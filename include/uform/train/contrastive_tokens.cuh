/**
 *  @file include/uform/train/contrastive_tokens.cuh
 *  @author Ash Vardanian
 *  @date October 21, 2026
 *  @brief Late-interaction contrastive loss over per-token vectors: hard MaxSim scores between
 *      query and document segments, relaxed by the same pair-loss pass the pooled segments loss
 *      runs.
 *
 *  The score of query segment @c q against document segment @c d is the mean over the query's
 *  tokens of the largest cosine against the document's tokens, @c S(q,d)=(1/|q|)·Σ_t max_s
 *  ⟨q_t,d_s⟩ , over the unit vectors the tower's token head emits, prefixes excluded. The logit is
 *  @c β·S+b with the query's temperature group, and @ref contrastive_pair_loss_t turns a whole
 *  query row into @c d(logit) over the relation graph, balanced or ranked, exactly as it does for
 *  the pooled cosines.
 *
 *  @section tokens_bands Banded products
 *
 *  The token grid of one query tile against the documents is never materialised. The documents'
 *  compact token rows split into bands of @c band_tokens rows; per band one NumKong BFloat16
 *  product @c P=Q·D_bandᵀ lands in a @b [query_tokens,band_tokens] Float32 block, and an epilogue
 *  folds its rows into the per-document running maximum of every query token; a document never
 *  crosses a band. The backward recomputes each band's product and credits the document's lowest
 *  row that attains the stored maximum, so no argmax is stored and ties resolve to the lowest row;
 *  the credited @b [query_tokens,band_tokens] block then drives two more products, @c dQ+=G·D_band
 *  and @c dD_band=Gᵀ·Q . Every fold runs in a fixed order, so runs and replays agree bitwise.
 *
 *  @section tokens_distributed Distributed contract
 *
 *  Queries move and documents stay: every GPU scores every GPU's query tile, its own included,
 *  against its own documents, reading the tile in place over NVLink. Each document GPU then returns
 *  the raw scores of every query GPU's rows against its documents, so the query's home GPU holds
 *  whole rows over the world's documents and runs the one whole-row pair-loss pass unchanged, for
 *  any relation graph. The score gradient goes back the same way, and each document GPU's
 *  query-tile forces reduce onto the home GPU. With @c G GPUs, @c S segments, @c T query tokens and
 *  width @c W per GPU, one step moves per GPU and each way:
 *
 *      - query tiles, read in place: @c (G-1)·T·W·2 bytes;
 *      - raw scores out, score gradients back: @c 2·(G-1)·S²·4 bytes;
 *      - query-tile forces: @c (G-1)·T·W·4 bytes, through the domain reduce;
 *      - the live-query count, temperature gradients and loss: @c O(G) scalars.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` — per-rank object lists

#include <numkong/tensor.hpp> // `nk::tensor_view`, `nk::tensor_span`, `nk::tensor`

#include "uform/train/contrastive_segments.cuh"       // `contrastive_pair_loss_t`, `contrastive_pair_loss_form_t`
#include "uform/utilities/infiniband_collectives.cuh" // `colocated_domain_t`, the cross-domain gathers and reduces
#include "uform/utilities/multimem_collectives.cuh"   // `multimem_domain_t` — the main-owned level-1 surface
#include "uform/encoder/multimodal_batch.hpp"         // `multimodal_batch_view_t`
#include "uform/utilities/vram_nvidia_schedule.cuh"   // `gpu_stream_t`, `launch_kernel`, the NumKong products
#include "uform/utilities/types.hpp"                  // `expected_t`, `status_t`, `late_interaction_side_t`

namespace unum::uform::train {
using namespace unum::uform::utilities;
using encoder::multimodal_batch_view_t;

#if defined(__CUDACC__)

/** The baked geometry of one tokens loss; a different config is a different object. */
struct contrastive_tokens_config_t {

    /** Segments per GPU — the rows of the relation graph and both axes of one score block. */
    u32_t segments_per_gpu = 0;

    /** Rows of one GPU's token head output, the tower's @c max_tokens . */
    u32_t head_tokens_per_gpu = 0;

    /** Query token rows one GPU's tile holds, a multiple of 8. */
    u32_t query_tokens_per_gpu = 0;

    /** Document segments one GPU holds at most. */
    u32_t documents_per_gpu = 0;

    /** Document token rows one GPU holds, rounded up to whole bands; a document never crosses a
     *  band, so the rows a band's tail leaves free count against it. */
    u32_t document_tokens_per_gpu = 0;

    /** Document token rows per product band, a multiple of 32 and at least the longest segment. */
    u32_t band_tokens = 0;

    /** Tokens one query segment contributes at most, its first past the prefix; the rest still
     *  serve the segment as a document. */
    u32_t query_span_tokens = ~0u;

    /** The token vectors' width, a multiple of 8. */
    u32_t width = 128;

    /** Query tiles scored per step — the world's GPU count. */
    u32_t gpu_subgraphs_per_step = 1;

    /** Learnable β and bias entries the recipe selects by group. */
    u32_t temperature_groups = 1;

    /** The pair loss relaxing each query row. */
    contrastive_pair_loss_form_t pair_loss = contrastive_pair_loss_form_t::ranked_k;

    /** The loss weight λ scaling the loss, its forces and its temperature gradients. */
    f32_t weight = 1.0f;
};

/** One GPU's query tile: its own, or a peer's read in place. */
struct gpu_query_subgraph_t {

    /** @b [query_tokens_per_gpu,width] unit query token vectors, zero past the live rows. */
    bf16r2_view_t query_vectors {};

    /** @b [query_tokens_per_gpu] each row's segment on its home GPU, @c ~0u past the live rows. */
    u32r1_view_t query_segments {};

    /** @b [segments_per_gpu+1] each home segment's first row in @ref query_vectors . */
    u32r1_view_t query_offsets {};
};

/** One GPU's share of the tokens loss: its query tile, its documents and the packs, scores and
 *  forces over them, all baked at @ref make — a call is a pure enqueue. */
struct contrastive_tokens_t {

    /** Validate @p config and allocate every tile, pack and scratch tensor off @p arena ,
     *  calling the kernels of @p numerics . */
    [[nodiscard]] static expected<contrastive_tokens_t> make(gpu_arena_t &arena, contrastive_tokens_config_t config,
                                                             gpu_numerics_t const &numerics) noexcept;

    /**
     *  @brief Single-GPU forward and backward for one step.
     *
     *  @param[in] token_vectors @b [head_tokens_per_gpu,width] the token head's unit vectors.
     *  @param[in] token_offsets @b [segments_per_gpu+1] each segment's first row in
     *      @p token_vectors .
     *  @param[in] batch The step's segment sides.
     *  @param[in] relations The staged relation graph over the segments.
     *  @param[in] recipe β and bias, per temperature group of the query rows.
     *  @param[out] loss_out @b [1] the mean over the live queries. Overwritten.
     *  @param[out] token_vectors_grad @b [head_tokens_per_gpu,width] the gradient of
     *      @p token_vectors .
     *  @param[inout] stream The stream every launch enqueues on.
     *  @param[out] log_inverse_temperature_grad Optional @b [temperature_groups] summands.
     *  @param[out] bias_grad Optional @b [temperature_groups] summands.
     */
    [[nodiscard]] expected_t operator()(bf16r2_view_t token_vectors, u32r1_view_t token_offsets,
                                        multimodal_batch_view_t const &batch, contrastive_relations_view_t relations,
                                        contrastive_recipe_t recipe, f32r1_span_t loss_out,
                                        bf16r2_span_t token_vectors_grad, gpu_stream_t &stream,
                                        f32r1_span_t log_inverse_temperature_grad = {},
                                        f32r1_span_t bias_grad = {}) noexcept;

    /** Lay out this GPU's query tile and documents from the token head's rows, and pack the
     *  documents. */
    [[nodiscard]] expected_t stage(bf16r2_view_t token_vectors, u32r1_view_t token_offsets,
                                   multimodal_batch_view_t const &batch, gpu_stream_t &stream) noexcept;

    /** This GPU's own query tile, as every GPU scores it. */
    [[nodiscard]] gpu_query_subgraph_t query_subgraph() const noexcept {
        return {query_tile_.view(), query_segments_.view(), query_offsets_.view()};
    }

    /** Score query tile @p subgraph , numbered @p subgraph_index , against this GPU's documents
     *  into @p raw_scores @b [segments_per_gpu,segments_per_gpu] , query segments by document
     *  segments. */
    [[nodiscard]] expected_t score(std::size_t subgraph_index, gpu_query_subgraph_t const &subgraph,
                                   f32r2_span_t raw_scores, gpu_stream_t &stream) noexcept;

    /** Differentiate @ref score for @p subgraph under @p score_grad , the gradient of its raw
     *  scores: the query-tile force into @p query_grad , overwritten, and the document force added
     *  onto this GPU's own. */
    [[nodiscard]] expected_t force(std::size_t subgraph_index, gpu_query_subgraph_t const &subgraph,
                                   f32r2_view_t score_grad, f32r2_span_t query_grad, gpu_stream_t &stream) noexcept;

    /** Turn the @b [1,subgraphs,segments,segments] raw score rows of this GPU's queries into the
     *  score gradient in place, through the pair loss over @p relations whose PAD columns are
     *  replaced by the token spans. @p world_document_lengths and @p live_queries carry the world's
     *  document spans and query count. */
    [[nodiscard]] expected_t relax(f32r4_span_t scores, contrastive_relations_view_t relations,
                                   contrastive_recipe_t recipe, u32r1_view_t world_document_lengths,
                                   f32r1_view_t live_queries, f32r1_span_t loss_out, gpu_stream_t &stream,
                                   f32r1_span_t log_inverse_temperature_grad, f32r1_span_t bias_grad) noexcept;

    /** Fold this GPU's summed query-tile force and document force into @p token_vectors_grad . */
    [[nodiscard]] expected_t finish(f32r2_view_t query_grad, u32r1_view_t token_offsets,
                                    multimodal_batch_view_t const &batch, bf16r2_span_t token_vectors_grad,
                                    gpu_stream_t &stream) noexcept;

    /** @b [segments_per_gpu] this GPU's document token spans by segment, zero for the rest. */
    [[nodiscard]] u32r1_view_t document_lengths() const noexcept { return document_lengths_.view(); }

    /** @b [1] this GPU's live query count, a Float32 summand of the world divisor. */
    [[nodiscard]] f32r1_view_t live_queries() const noexcept { return live_queries_.view(); }

    /** @b [1] nonzero when the last step's query rows, document rows or documents outgrew their
     *  capacities and the excess went unscored — a fault the caller's telemetry reports. */
    [[nodiscard]] u32r1_view_t overflow() const noexcept { return overflow_.view(); }

  private:

    /** The baked geometry. */
    contrastive_tokens_config_t config_ {};

    /** The NumKong kernels of the arena's device class, every product's. */
    gpu_numerics_t numerics_ {};

    /** The whole-row pair loss over this GPU's query rows. */
    contrastive_pair_loss_t pair_loss_ {};

    /** @b [query_tokens,width] this GPU's query tile, and each row's segment, @c ~0u past the live
     *  rows. */
    bf16r2_tensor_arena_t query_tile_ {};
    u32r1_tensor_arena_t query_segments_ {};

    /** @b [segments+1] each segment's first query row, and @b [segments] its query span. */
    u32r1_tensor_arena_t query_offsets_ {}, query_lengths_ {};

    /** @b [document_tokens,width] this GPU's documents, in document order, and each row's
     *  document. */
    bf16r2_tensor_arena_t document_tile_ {};
    u32r1_tensor_arena_t document_of_row_ {};

    /** @b [documents+1] each document's first row, the rows used last; @b [documents] its segment,
     *  @c ~0u past the live documents. */
    u32r1_tensor_arena_t document_offsets_ {}, document_segments_ {};

    /** @b [segments] each segment's document span and document, @c ~0u when it is none. */
    u32r1_tensor_arena_t document_lengths_ {}, document_of_segment_ {};

    /** @b [1] the live query count, and the mark of a step whose tiles outgrew their capacities. */
    f32r1_tensor_arena_t live_queries_ {};
    u32r1_tensor_arena_t overflow_ {};

    /** @b [width,query_tokens] and @b [width,document_tokens] the tiles transposed, rows padded to
     *  8. */
    bf16r2_tensor_arena_t query_tile_transposed_ {}, document_tile_transposed_ {};

    /** Every band's document pack, forward and transposed, band after band, and the query pack. */
    b8r1_tensor_arena_t packed_documents_ {}, packed_documents_transposed_ {}, packed_queries_transposed_ {};

    /** One band's forward and transposed pack bytes, as NumKong sized them at @ref make . */
    std::size_t band_pack_bytes_ = 0, band_pack_transposed_bytes_ = 0;

    /** @b [subgraphs,query_tokens,documents] each query token's running maximum over each
     *  document. */
    f32r3_tensor_arena_t maxima_ {};

    /** @b [query_tokens,band_tokens] one band's product and its credited gradient. */
    f32r2_tensor_arena_t band_products_ {};
    bf16r2_tensor_arena_t band_grad_ {};

    /** @b [band_tokens,query_tokens] the credited gradient transposed, rows padded to 8. */
    bf16r2_tensor_arena_t band_grad_transposed_ {};

    /** @b [query_tokens,width] and @b [band_tokens,width] one band's force products. */
    f32r2_tensor_arena_t query_grad_products_ {}, document_grad_products_ {};

    /** @b [document_tokens,width] this GPU's document force, summed over the subgraphs. */
    f32r2_tensor_arena_t document_grad_ {};

    /** @b [query_tokens,width] the single-GPU query force; @b [segments·segments] the single-GPU
     *  scores. */
    f32r2_tensor_arena_t query_grad_ {};
    f32r1_tensor_arena_t scores_ {};

    /** Band @p band_index 's pack inside @p packs , band after band of @p band_bytes each. */
    [[nodiscard]] static b8r1_span_t packed_band_(b8r1_tensor_arena_t &packs, std::size_t band_bytes,
                                                  std::size_t band_index) noexcept {
        nk::tensor_span<b8_t, 2> const bands = packs.span().reshape<2>({packs.extent(0) / band_bytes, band_bytes});
        return bands[band_index, nk::slice];
    }

    /** The band count over the document rows. */
    [[nodiscard]] std::size_t bands_() const noexcept { return config_.document_tokens_per_gpu / config_.band_tokens; }
};

/** One GPU's inputs and output of a distributed tokens step. */
struct gpu_tokens_inputs_t {

    /** @b [head_tokens_per_gpu,width] the token head's unit vectors, and @b [segments+1] their
     *  segment offsets. */
    bf16r2_view_t token_vectors {};
    u32r1_view_t token_offsets {};

    /** The step's segment sides. */
    multimodal_batch_view_t batch {};

    /** The GPU's staged relation graph over its segments and the world's. */
    contrastive_relations_view_t relations {};

    /** β and bias of the GPU's query rows. */
    contrastive_recipe_t recipe {};

    /** @b [head_tokens_per_gpu,width] the gradient of @ref token_vectors , overwritten. */
    bf16r2_span_t token_vectors_grad {};
};

/**
 *  @brief The distributed sibling of @ref contrastive_tokens_t : one object owning every local
 *      GPU's share of a world of multimem domains.
 *
 *  It holds a reserved window of the main-owned @ref multimem_domain_t and the cross-domain members
 *  wired through the main-owned @ref colocated_domain_t .
 *
 *  Inside a domain, query tiles, scores and score gradients are read in place from every member's
 *  arena and window, so every arena must be visible to every local device and one process holds the
 *  whole domain. Across domains, each member pairs with the same member of every other domain: it
 *  all-gathers its own query tile, its outbound scores and its inbound score gradients, so a GPU
 *  reads another domain's block from the gathered copy of the local member that owns that member
 *  index. The query forces reduce in two levels, the domain's switch reduce then a cross-domain
 *  all-reduce, beside the scalars.
 *
 *  A step runs five phases, each closed by the domain barrier and enqueued for every GPU before the
 *  next: stage and gather the tiles; score every world tile and gather the outbound scores; relax
 *  the home rows and gather the score gradients; force every world tile; reduce the query forces
 *  and scalars and finish. Every process enqueues its gathers and reduces in the same order.
 */
struct distributed_contrastive_tokens_t {

    /** How many Float32 payload scalars this consumer reserves from the shared domain, for main's
     *  sizing. */
    [[nodiscard]] static constexpr std::size_t payload_scalars_needed(contrastive_tokens_config_t config) noexcept {
        std::size_t const subgraphs = config.gpu_subgraphs_per_step, segments = config.segments_per_gpu;
        return 2u * subgraphs * segments * segments +
               subgraphs * std::size_t {config.query_tokens_per_gpu} * config.width + world_scalar_count_(config);
    }

    /**
     *  @brief Build every local GPU's share, reserve the window, validate the arenas' visibility
     *      and wire every member's cross-domain gathers and reduces.
     *  @param[inout] domain The main-owned level-1 surface, one window reserved from it.
     *  @param[inout] colocated The main-owned level-2 surface; identities in a one-domain world.
     *  @param[in] arenas @b [gpus] per-GPU arenas, every one visible to every local device.
     *  @param[in] streams @b [gpus] per-GPU streams scoping each per-device setup.
     *  @param[in] config The per-GPU geometry; @c gpu_subgraphs_per_step must equal the world's
     *      GPUs.
     *  @param[in] numerics The NumKong kernels every GPU's products call.
     */
    [[nodiscard]] static expected<distributed_contrastive_tokens_t> make(
        multimem_domain_t &domain, colocated_domain_t &colocated, std::span<gpu_arena_t> arenas,
        std::span<gpu_stream_t> streams, contrastive_tokens_config_t config, gpu_numerics_t const &numerics,
        f64_t timeout_seconds = 120.0) noexcept;

    /** One step over every local GPU, enqueued phase by phase across @p inputs and @p streams . */
    [[nodiscard]] expected_t operator()(std::span<gpu_tokens_inputs_t const> inputs,
                                        std::span<gpu_stream_t> streams) noexcept;

    /** Rank @p rank 's @b [2·temperature_groups] world-summed temperature gradients, log-β then
     *  bias. */
    [[nodiscard]] f32r1_view_t temperature_grads(std::size_t rank) const noexcept {
        return world_scalars_[rank][nk::range(0, 2u * std::size_t {config_.temperature_groups}), nk::slice];
    }

    /** Rank @p rank 's @b [1] world loss of the last call. */
    [[nodiscard]] f32r1_view_t world_loss(std::size_t rank) const noexcept {
        std::size_t const gradients = 2u * std::size_t {config_.temperature_groups};
        return world_scalars_[rank][nk::range(gradients, gradients + 1u), nk::slice];
    }

  private:

    /** The scalars reduced across the world: the temperature gradients, the loss and the live-query
     *  count. */
    [[nodiscard]] static constexpr std::size_t world_scalar_count_(contrastive_tokens_config_t config) noexcept {
        return 2u * std::size_t {config.temperature_groups} + 2u;
    }

    /** The scalars of one GPU's gathered tile description: its query rows' segments, its query
     *  offsets and its document spans. */
    [[nodiscard]] std::size_t description_scalars_() const noexcept {
        return std::size_t {config_.query_tokens_per_gpu} + 2u * config_.segments_per_gpu + 1u;
    }

    /** Member @p member 's window: @b [subgraphs,segments,segments] raw scores by query GPU, the
     *  same shape of score gradients by document GPU, @b [subgraphs,query_tokens,width] query
     *  forces, then the scalars. */
    [[nodiscard]] f32r1_span_t window_payload_(std::size_t member) noexcept {
        f32r1_span_t const payload = domain_->payload(member);
        return payload[nk::range(window_.offset, window_.offset + window_.scalars), nk::slice];
    }
    [[nodiscard]] f32r3_span_t window_scores_(std::size_t member, std::size_t block) noexcept;
    [[nodiscard]] f32r3_span_t window_query_forces_(std::size_t member) noexcept;
    [[nodiscard]] f32r1_span_t window_tail_(std::size_t member) noexcept;

    /** How many multimem domains the world spans, from the retained topology. */
    [[nodiscard]] u32_t world_domains_() const noexcept { return colocated_->multimem_domain_count(); }

    /** The query tile of world GPU @p world_gpu : read in place inside this domain, from the
     *  gathered copies across domains. */
    [[nodiscard]] gpu_query_subgraph_t world_subgraph_(std::size_t world_gpu) noexcept;

    /** @b [segments] world GPU @p world_gpu 's document spans, in place or gathered. */
    [[nodiscard]] u32r1_view_t world_document_lengths_of_(std::size_t world_gpu) noexcept;

    /** World GPU @p world_gpu 's block @p block of its window scores,
     *  @b [subgraphs,segments,segments] , in place or gathered. */
    [[nodiscard]] f32r3_view_t world_scores_(std::size_t world_gpu, std::size_t block) noexcept;

    /** The step's phases for GPU @p rank , each enqueued before the barrier that closes it. */
    [[nodiscard]] expected_t stage_(std::size_t rank, gpu_tokens_inputs_t const &inputs, gpu_stream_t &stream) noexcept;
    [[nodiscard]] expected_t score_(std::size_t rank, gpu_tokens_inputs_t const &inputs, gpu_stream_t &stream) noexcept;
    [[nodiscard]] expected_t relax_(std::size_t rank, gpu_tokens_inputs_t const &inputs, gpu_stream_t &stream) noexcept;
    [[nodiscard]] expected_t force_(std::size_t rank, gpu_tokens_inputs_t const &inputs, gpu_stream_t &stream) noexcept;
    [[nodiscard]] expected_t reduce_(std::size_t rank, gpu_tokens_inputs_t const &inputs,
                                     gpu_stream_t &stream) noexcept;

    /** The baked geometry. */
    contrastive_tokens_config_t config_ {};

    /** One share per local GPU. */
    per_gpu_array<contrastive_tokens_t> losses_ {};

    /** The main-owned level-1 surface, borrowed. */
    multimem_domain_t *domain_ = nullptr;

    /** The main-owned level-2 surface, borrowed; its topology counts the domains. */
    colocated_domain_t *colocated_ = nullptr;

    /** This process's own multimem domain coordinate, the world GPU index's major part. */
    u32_t home_domain_index_ = 0;

    /** This consumer's window of the domain payload. */
    multimem_window_t window_ {};

    /** Per rank: the home score rows @b [subgraphs·segments·segments] , the world document spans
     *  @b [subgraphs·segments] , and the reduced query force @b [query_tokens,width] . */
    per_gpu_array<f32r1_tensor_arena_t> home_scores_ {};
    per_gpu_array<u32r1_tensor_arena_t> world_document_lengths_ {};
    per_gpu_array<f32r2_tensor_arena_t> query_grads_ {};

    /** Per rank: @b [2·temperature_groups+2] the world-summed tail. */
    per_gpu_array<f32r1_tensor_arena_t> world_scalars_ {};

    /** Per rank, wired only across several domains: the gathers of its query tile, its tile
     *  description, its outbound scores and its inbound score gradients. */
    per_gpu_array<infiniband_all_gather<bf16_t>> tile_gathers_ {};
    per_gpu_array<infiniband_all_gather<u32_t>> description_gathers_ {};
    per_gpu_array<infiniband_all_gather<f32_t>> score_gathers_ {}, gradient_gathers_ {};

    /** Per rank, wired only across several domains: the query forces with the scalars, and the
     *  live-query count. */
    per_gpu_array<infiniband_all_reduce<f32_t>> force_reduces_ {}, count_reduces_ {};
};

#endif // defined(__CUDACC__)

} // namespace unum::uform::train
