/**
 *  @file include/uform/core/multimodal_batch.cuh
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief The NVIDIA half of the multimodal batch: its device and pinned residencies, the
 *      coordinate fill, and the mixer that shuffles batches across GPUs.
 *
 *  The batch contract itself — shapes, the allocator-parameterized @ref multimodal_batch , and its views — lives
 *  in @c multimodal_batch.hpp and compiles without CUDA. See its file comment for the layout and the flow.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` — the mixer borrows the composition root's per-GPU tables as views

#include "uform/core/gpu_allocators.hpp"         // `gpu_arena_alloc_t`, `pinned_alloc_t`
#include "uform/core/infiniband_collectives.cuh" // `colocated_domain_t` — the cross-host_domain lane
#include "uform/core/multimem_collectives.cuh"   // `multimem_domain_t` — the barrier the exchange brackets
#include "uform/core/multimodal_batch.hpp"       // the shape, the batch template, the views, `modality_t`
#include "uform/core/schedule.cuh"               // `gpu_stream_t` — the mixer's enqueue surface

namespace unum::uform::core {

/** The device-resident batch the towers read and the mixer routes. */
using multimodal_batch_on_gpu_t = multimodal_batch<gpu_arena_alloc_t>;

/** The pinned-host batch a host packer writes before one @c copy_from to the device twin. */
using multimodal_batch_on_cpu_t = multimodal_batch<pinned_alloc_t>;

/**
 *  @brief Write every live segment's coordinate @c (t,h,w) into @p coords @b [tokens,3] , one CTA per
 *      segment, dispatching on @p segment_modality and reading the matching per-modality shape tensor.
 *
 *  It is one segment walk over every modality: a segment reads its @c (offset,length) , then threads stripe
 *  @c [0,length) writing @c coords[offset+k] as @c text_k / @c audio_k → (k,0,0) (both 1-D sequences — audio
 *  frames walk the same position axis as tokens, and read no shape descriptor); @c image_k →
 *  (0,k/patch_cols,k%patch_cols) with @c patch_cols=image_shapes[s,1] ; @c video_k →
 *  (k/gpf·tube_frames·frame_stride,(k%gpf)/pc,(k%gpf)%pc) with @c (tg,pr,pc)=video_shapes[s] and @c gpf=pr·pc ,
 *  so the temporal coordinate is the absolute source frame a tube starts at. PAD segments ( @c length==0 ) write
 *  nothing, so the caller must pre-zero @p coords .
 *
 *  @param[in] tube_frames The video tube depth — the batch's fixed value.
 *  @param[in] frame_stride The source frames between two sampled ones — the batch's fixed value, 0 read as 1.
 */
[[nodiscard]] expected_t fill_coords(u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                     modality_span_t segment_modality, image_shape_view_t image_shapes,
                                     video_shape_view_t video_shapes, u32_t tube_frames, u32_t frame_stride,
                                     i16r2_span_t coords, gpu_stream_t &stream) noexcept;

/** Mark every segment of @p batch as making no choice, @c byte_pad_k across its choice rows and
 *  zero targets, for a source that names none; a batch without choice columns stays as it is. */
[[nodiscard]] expected_t pad_choices(multimodal_batch_on_gpu_t &batch, gpu_stream_t &stream) noexcept;

/**
 *  @brief Shuffles the loaded batches between the GPUs of one multimem_domain — and, once wired,
 *      across the colocated_domain via InfiniBand — so every step trains on a fresh cross-GPU mix.
 *      Borrows the composition root's single @c multimem_domain_t and @c colocated_domain_t .
 *
 *  A mix reads every member's @b loaded batch and writes each member's @b mixed batch — two distinct
 *  per-GPU instances, so no barrier separates reads from overwrites within one mix. Routing is a
 *  keyed bijection every member derives identically in-kernel (an implementation detail of the
 *  translation unit), so it needs no exchange; the multimem_domain's barrier brackets the exchange,
 *  and the ordering contract's cross-consumer sequence starts here. Runs eagerly, outside any
 *  captured graph, in the same phase as the loader fills.
 */
struct multimodal_batch_mixer_t {

    /** How many Float32 payload scalars @ref make will @c reserve from the multimem_domain over @p mesh — zero
     *  when one process holds the whole world and reads every loaded batch in place; otherwise one packed batch
     *  per multimem_domain in every member's window. */
    [[nodiscard]] static std::size_t payload_scalars_needed(multimodal_batch_shape_t const &shape,
                                                            mesh_shape_t const &mesh) noexcept;

    /**
     *  @brief Validate the world, bake the routing key, and — when no process holds the whole world — reserve
     *      the window every member packs its batch into, plus one @c infiniband_all_gather buffer per local
     *      member when the world spans several multimem_domains.
     *
     *  A process holding the whole world reads its loaded batches in place, peer addresses travelling as
     *  kernel arguments bounded by @ref max_gpus_per_host_k . Otherwise world GPU @c (domain,member) is read
     *  from that member's window, block @c domain , resolved arithmetically, so the world is unbounded. Every
     *  arena must be visible to every local member (validated, never granted here).
     *
     *  @param[in] arenas One per member this process drives, @c [multimem.member_offset(),+arenas.size()) .
     *  @param[in] streams Each local member's stream — the all-gather wiring seeds its round scalar there.
     *  @param[in] timeout_seconds The coordinate-exchange deadline handed to each all-gather wiring.
     */
    [[nodiscard]] static expected<multimodal_batch_mixer_t> make(multimem_domain_t &multimem,
                                                                 colocated_domain_t &colocated,
                                                                 std::span<gpu_arena_t> arenas,
                                                                 multimodal_batch_shape_t shape, u64_t seed,
                                                                 std::span<gpu_stream_t> streams,
                                                                 f64_t timeout_seconds = 120.0) noexcept;

    /**
     *  @brief Local member @p local 's slice of one mix — pure enqueue onto @p stream : gather its
     *      incoming segments from every world GPU's loaded batch into its own @p mixed batch.
     *
     *  Every member barriers (its own self-advancing round) before its peer reads, and members are
     *  enqueued in order @c 0..members-1 within a step. Packed windows and gather payloads are reused by the
     *  next mix, which the step's later domain barriers order after this mix's reads.
     *
     *  @param[in] local This GPU's index among the process's members — domain member
     *      @c multimem.member_offset()+local , world GPU @c first_gpu_index_in_colocated_domain_+that .
     *  @param[in] data_epoch The keyed-draw counter the routing bijection folds with the run's seed.
     *  @param[in] loaded Every local member's freshly loaded batch, in local order.
     *  @param[out] mixed This member's destination batch — distinct storage from every @p loaded .
     *  @param[in] stream This member's stream; the caller holds no context around the call.
     */
    [[nodiscard]] expected_t operator()(std::size_t local, data_epoch_t data_epoch,
                                        std::span<multimodal_batch_on_gpu_t const> loaded,
                                        multimodal_batch_on_gpu_t &mixed, gpu_stream_t &stream) noexcept;

    /** The eager convenience: every member's slice in order over the per-GPU streams. */
    [[nodiscard]] expected_t operator()(data_epoch_t data_epoch, std::span<multimodal_batch_on_gpu_t const> loaded,
                                        std::span<multimodal_batch_on_gpu_t> mixed,
                                        std::span<gpu_stream_t> streams) noexcept;

    /** Local member @p local 's @b [world_gpus,max_segments_per_gpu] mixed lengths of every world
     *  GPU's last mix, zero for a PAD segment; flattened, the global vertex indexes them as the
     *  contrastive relations read them. */
    [[nodiscard]] u32r2_view_t world_lengths(std::size_t local) const noexcept { return world_lengths_[local].view(); }

    /** Local member @p local 's @b [world_gpus,max_segments_per_gpu] world sources of every world
     *  GPU's last mix, each @c source_gpu·max_segments_per_gpu+source_segment , @c ~0u for PAD. */
    [[nodiscard]] u32r2_view_t world_sources(std::size_t local) const noexcept { return world_sources_[local].view(); }

    /** Local member @p local 's @b [world_gpus,max_segments_per_gpu] keys of every world GPU's last
     *  mix, read from each segment's source and zero for PAD; flattened, the global vertex indexes
     *  them as the contrastive relations read them. */
    [[nodiscard]] u64r2_view_t world_keys(std::size_t local) const noexcept { return world_keys_[local].view(); }

  private:

    /** The main-owned multimem_domain — borrowed; its barrier brackets every mix. */
    multimem_domain_t *multimem_ = nullptr;

    /** The run's routing seed; combined with the per-call data epoch inside the routing bijection. */
    u64_t seed_ = 0;

    /** The baked batch capacities every route and repack cursor is bounded by. */
    multimodal_batch_shape_t shape_ {};

    /** Every member's packed batch per multimem_domain, domain-major — empty when this process holds the whole
     *  world and reads the loaded batches in place. */
    multimem_window_t window_ {};

    /** One cross-domain full-batch all-gather per local member, pairing same-index members of every domain —
     *  empty on a single-multimem_domain world. */
    per_gpu_array<infiniband_all_gather<b8_t>> gather_buffers_ {};

    /** Per-member @b [world_gpus,max_segments_per_gpu] routing tables of every world GPU's mix,
     *  which each member derives alone from the same loaded batches. The geometry pass records the
     *  member's own slice of the sources and the gather pass reads it back, since variable-degree
     *  bags make a destination's source un-derivable per segment. On the member's own arena; never
     *  exchanged, never part of a batch. */
    per_gpu_array<u32r2_tensor_arena_t> world_lengths_ {}, world_sources_ {};

    /** Per-member @b [world_gpus,max_segments_per_gpu] keys of every world GPU's mix, read through
     *  the sources once the world pass completes them. */
    per_gpu_array<u64r2_tensor_arena_t> world_keys_ {};

    /** How many multimem_domains the world spans; 1 is the single-NVLink-domain fast path. */
    u32_t domains_ = 1;

    /** This process's multimem_domain index — the block its members pack their own batch into. */
    u32_t multimem_domain_index_ = 0;

    /** This multimem_domain's first member's world coordinate (the mesh's inverse mapping) — member
     *  @c m 's routing destination is @c first+m , so the domains claim disjoint destination ranges in
     *  the @c members·domains world. */
    u32_t first_gpu_index_in_colocated_domain_ = 0;
};

} // namespace unum::uform::core
