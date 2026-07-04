/**
 *  @brief Memory-owning multimodal training batches, one per GPU. The allocator template argument
 *      decides residency: a @c pinned_alloc_t instance is what a host packer writes, a
 *      @c gpu_arena_alloc_t instance is what the towers read, and one @c copy_from between the
 *      two is the whole transfer. Distribution across the multimem_domain and the colocated_domain
 *      is @ref multimodal_batch_mixer_t 's job; batches never know they are distributed.
 *  @file cuda_core/multimodal_batch.cuh
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *
 *  @section multimodal_batch_rows Rectangular rows, grouped by modality
 *
 *  A batch is a @b [rows,columns] rectangle of fixed-width rows packing segments back to back,
 *  grouped by @ref modality_t in the enum's order — token rows first, then image, video, audio — so
 *  a row's modality is arithmetic over the shape's baked item counts and no tag column exists.
 *  Media rows carry their @b raw signals — pixels, waveforms — and every learnable transform (patch
 *  cut, projection, embedding lookup) happens inside the differentiated step with fresh weights,
 *  never at packing time.
 *
 *  @section multimodal_batch_pipeline How batches flow through a training step
 *
 *  Each GPU loads a few random dataset shards and samples a few batches of various modality content.
 *  In Omni models the GPU itself chooses how many examples of each modality to take, so that their
 *  total fits @c max_segments_per_gpu — media modalities are typically long enough that one segment
 *  covers its whole row. The loaded multimodal batches are then shuffled between the GPUs of the
 *  multimem_domain and across the colocation via InfiniBand — the mixer's job.
 *
 *  Under the @b masked objective, once inputs are exchanged, each GPU itself decides how to corrupt
 *  every modality's contents — beyond simple masking, any distortion is allowed — and reconstructs at
 *  the output: training runs on two instances of the same batch, the clean one as the reconstruction
 *  target, the distorted one as the input. Under the @b contrastive objective the input stays
 *  uncorrupted and the relations graph rides separately (see @c contrastive_relations.cuh ) — with
 *  extra attention in the mixer so pairs and anchors land on the same GPU.
 *
 *  @section multimodal_batch_capacity Capacity once, PAD through data
 *
 *  Every tensor is capacity-allocated at @ref multimodal_batch::make and keeps constant extents. PAD
 *  segments repeat the previous offset (live segments are never empty by the packer's contract) —
 *  existence is always read from the data, and a captured step replays against freshly copied
 *  contents at stable addresses.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <memory>  // `std::allocator_traits` — the batch's per-type rebinding

#include <numkong/tensor.hpp> // `nk::tensor` — the owning storage

#include "types.cuh"                  // the scalar aliases, `token_id_t`, `pixel_t`
#include "array.cuh"                  // `per_gpu_array` — the composition root's per-GPU tables
#include "schedule.cuh"               // `gpu_stream_t` — the mixer's enqueue surface
#include "gpu_allocators.cuh"         // `gpu_arena_alloc_t`, `pinned_alloc_t`
#include "multimem_collectives.cuh"   // `multimem_domain_t` — the barrier the exchange brackets
#include "infiniband_collectives.cuh" // `colocated_domain_t` — the cross-host_domain lane

namespace unum::uform::core {

/** @brief One row's input kind — which raw tensor feeds it, which RoPE pattern it walks, and, under
 *      masking, which reconstruction head scores it. Also the rows' grouping order inside a batch.
 *      Unscoped over @c u32_t . */
enum modality_t : u32_t {
    tokens_k = 0,
    image_k = 1,
    video_k = 2,
    audio_k = 3,
};

/** @brief The pixel side of the square image that exactly fills one @p columns -token row: with
 *      @p patch_size -pixel patches, @p columns patches tile a @c sqrt(columns)×sqrt(columns) grid, one
 *      encoder token each. Returns @c 0 when @p columns is not a perfect square — a row budget that
 *      cannot tile a square image is a build-time error the caller rejects. Media geometry flows one
 *      way: the tier fixes @c max_columns_per_row and the image resolution is sized so one image
 *      occupies a whole row, never a baked constant. */
[[nodiscard]] constexpr u32_t image_side(u32_t columns, u32_t patch_size) noexcept {
    u32_t side = 0;
    while ((side + 1u) * (side + 1u) <= columns) ++side;
    return side * side == columns ? side * patch_size : 0u;
}

/** @brief Scalars in one image patch vector — @c channels·patch_size² , ordered (channel,row,column). */
[[nodiscard]] constexpr u32_t patch_scalars(u32_t patch_size, u32_t channels) noexcept {
    return channels * patch_size * patch_size;
}

/** @brief Scalars in one video tube vector — @c frames·channels·side² , ordered
 *      (local_time,channel,row,column). */
[[nodiscard]] constexpr u32_t tube_scalars(u32_t frames, u32_t side, u32_t channels) noexcept {
    return frames * channels * side * side;
}

/** @brief Scalars in one mel patch vector — @c side² , a single-channel spectrogram tile. */
[[nodiscard]] constexpr u32_t mel_patch_scalars(u32_t side) noexcept { return side * side; }

/** @brief The baked capacities of one GPU's batch — a different shape is a different batch object.
 *      A batch is a @b [max_rows_per_gpu,max_columns_per_row] rectangle: every row is an independent
 *      attention sequence of fixed width, and every segment (a document, an image, a clip) occupies
 *      a contiguous column span of exactly one row, several sharing a row back to back. A single row
 *      budget covers every modality — the composition is the loader's per-step choice — so each
 *      media tensor is sized for @c max_rows_per_gpu items of its raw extents; a modality with zero
 *      extents stays unallocated and out of the composition. */
struct multimodal_batch_shape_t {
    /** @brief How many rows one GPU's step carries, whatever the modality mix. */
    u32_t max_rows_per_gpu = 0;
    /** @brief How many segments one GPU's step carries, across all its rows — also the pooling axis
     *      the contrastive anchors live on. */
    u32_t max_segments_per_gpu = 0;
    /** @brief How many positions one row spans — what most of the literature calls @c seq_len . */
    u32_t max_columns_per_row = 0;

    /** @brief One image's raw extents, as decoded; zeros leave the pixel tensor unallocated. */
    u32_t image_height = 0, image_width = 0, image_channels = 3;
    /** @brief One clip's frame count; frames share the image extents. Zero leaves videos unallocated. */
    u32_t video_frames = 0;
    /** @brief One clip's mono sample count (the loader downmixes at decode); zero leaves audio
     *      unallocated. */
    u32_t clip_samples = 0;
};

/**
 *  @brief The rectangular geometry and the raw inputs of one GPU's batch: one tensor per modality,
 *      each with its natural extents and dtype, sized by a single @ref multimodal_batch_shape_t — a
 *      modality with a zero cap stays unallocated.
 *
 *  @tparam alloc_type_ A byte-typed allocator handle ( @c gpu_arena_alloc_t ,
 *      @c pinned_alloc_t , @c gpu_alloc_t ) — rebound per element type inside.
 */
template <typename alloc_type_>
struct multimodal_batch {

    /** @brief @p alloc_type_ rebound to @p value_type_ — every tensor's allocator spelling. */
    template <typename value_type_>
    using rebound = typename std::allocator_traits<alloc_type_>::template rebind_alloc<value_type_>;

    /** @brief @b [max_segments_per_gpu+1] segment START positions in the flat @c row·columns+column
     *      space (the last entry is the pack cursor's end). A position's row is its offset divided by
     *      the row width, and a segment never crosses its row — which leaves gaps before row jumps,
     *      so lengths are never adjacent differences; they live in @ref segment_lengths . */
    nk::tensor<u32_t, rebound<u32_t>, 1> segment_offsets {};
    /** @brief @b [max_segments_per_gpu] live token counts — stored, not derived: the row-jump gaps
     *      between consecutive segments make adjacent offset differences overestimates. PAD segments
     *      carry 0. */
    nk::tensor<u32_t, rebound<u32_t>, 1> segment_lengths {};
    /** @brief @b [rows,columns] the discrete inputs of token rows. */
    nk::tensor<token_id_t, rebound<token_id_t>, 2> tokens {};
    /** @brief @b [images,height,width,channels] raw pixels, one item per image row. */
    nk::tensor<pixel_t, rebound<pixel_t>, 4> images {};
    /** @brief @b [videos,frames,height,width,channels] raw pixels, one item per video row. */
    nk::tensor<pixel_t, rebound<pixel_t>, 5> videos {};
    /** @brief @b [clips,samples] raw mono waveforms, one item per audio row. */
    nk::tensor<f32_t, rebound<f32_t>, 2> audios {};

    std::size_t rows() const noexcept {
        return tokens.extent(0) + images.extent(0) + videos.extent(0) + audios.extent(0);
    }

    /** @brief Allocate every tensor the shape names — once, at setup. Modalities with zero caps
     *      allocate nothing; the geometry and token tensors always exist. */
    [[nodiscard]] static expected<multimodal_batch> make(alloc_type_ allocator,
                                                         multimodal_batch_shape_t shape) noexcept {
        using result_t = expected<multimodal_batch>;
        if (shape.max_rows_per_gpu == 0 || shape.max_segments_per_gpu == 0 || shape.max_columns_per_row == 0)
            return result_t::fail(status_t::invalid_input_k,
                                  "a batch needs all three capacities (rows={} segments={} columns={})",
                                  shape.max_rows_per_gpu, shape.max_segments_per_gpu, shape.max_columns_per_row);
        multimodal_batch batch;
        batch.segment_offsets = nk::tensor<u32_t, rebound<u32_t>, 1>::try_empty({shape.max_segments_per_gpu + 1u},
                                                                                rebound<u32_t> {allocator});
        batch.segment_lengths = nk::tensor<u32_t, rebound<u32_t>, 1>::try_empty({shape.max_segments_per_gpu},
                                                                                rebound<u32_t> {allocator});
        batch.tokens = nk::tensor<token_id_t, rebound<token_id_t>, 2>::try_empty(
            {shape.max_rows_per_gpu, shape.max_columns_per_row}, rebound<token_id_t> {allocator});
        if (!batch.segment_offsets || !batch.segment_lengths || !batch.tokens)
            return result_t::fail(status_t::bad_alloc_k, "batch allocation failed (rows={} segments={} columns={})",
                                  shape.max_rows_per_gpu, shape.max_segments_per_gpu, shape.max_columns_per_row);
        if (shape.image_height != 0 && shape.image_width != 0) {
            batch.images = nk::tensor<pixel_t, rebound<pixel_t>, 4>::try_empty(
                {shape.max_rows_per_gpu, shape.image_height, shape.image_width, shape.image_channels},
                rebound<pixel_t> {allocator});
            if (!batch.images)
                return result_t::fail(status_t::bad_alloc_k, "image allocation failed ({} of [{},{},{}])",
                                      shape.max_rows_per_gpu, shape.image_height, shape.image_width,
                                      shape.image_channels);
        }
        if (shape.video_frames != 0) {
            batch.videos = nk::tensor<pixel_t, rebound<pixel_t>, 5>::try_empty(
                {shape.max_rows_per_gpu, shape.video_frames, shape.image_height, shape.image_width,
                 shape.image_channels},
                rebound<pixel_t> {allocator});
            if (!batch.videos)
                return result_t::fail(status_t::bad_alloc_k, "video allocation failed ({} of {} frames)",
                                      shape.max_rows_per_gpu, shape.video_frames);
        }
        if (shape.clip_samples != 0) {
            batch.audios = nk::tensor<f32_t, rebound<f32_t>, 2>::try_empty({shape.max_rows_per_gpu, shape.clip_samples},
                                                                           rebound<f32_t> {allocator});
            if (!batch.audios)
                return result_t::fail(status_t::bad_alloc_k, "audio allocation failed ({} of {} samples)",
                                      shape.max_rows_per_gpu, shape.clip_samples);
        }
        return result_t {std::move(batch)};
    }
};

/** @brief The device-resident batch the towers read and the mixer routes. */
using multimodal_batch_on_gpu_t = multimodal_batch<gpu_arena_alloc_t>;
/** @brief The pinned-host batch a host packer writes before one @c copy_from to the device twin. */
using multimodal_batch_on_cpu_t = multimodal_batch<pinned_alloc_t>;

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

    /** @brief Validate the world and bake the routing key — once, at setup. The mixer owns no memory:
     *      peer addresses travel as kernel arguments (bounded by @ref max_gpus_per_host_k ) and routes
     *      are re-derived in-register, so there is no scratch to allocate. Every arena must be visible
     *      to every member of the multimem_domain (validated, never granted here). Worlds spanning
     *      several multimem_domains fail loudly until the InfiniBand lane is wired. */
    [[nodiscard]] static expected<multimodal_batch_mixer_t> make(multimem_domain_t &multimem,
                                                                 colocated_domain_t &colocated,
                                                                 per_gpu_array<gpu_arena_t> &arenas,
                                                                 multimodal_batch_shape_t shape, u64_t seed) noexcept;

    /**
     *  @brief Member @p member_index 's slice of one mix — pure enqueue onto @p stream : gather its
     *      incoming segments from all members' @p loaded batches into its own @p mixed batch. Member
     *      0's slice carries the multimem_domain's epoch advance; every member barriers before its
     *      peer reads, and members are enqueued in order @c 0..members-1 within a step.
     *
     *  @param[in] member_index This GPU's index inside the borrowed @b multimem_domain —
     *      @c [0,multimem_domain_t::member_count()) , the same axis @c member() and @c barrier()
     *      take. Not a host_domain index and not a colocated_domain coordinate: the mixer's world is
     *      one multimem_domain until the InfiniBand lane is wired.
     *  @param[in] epoch The keyed-draw counter the routing bijection folds with the run's seed.
     *  @param[in] loaded Every member's freshly loaded batch, in member order.
     *  @param[out] mixed This member's destination batch — distinct storage from every @p loaded .
     *  @param[in] stream This member's stream; the caller holds no context around the call.
     */
    [[nodiscard]] expected_t operator()(std::size_t member_index, epoch_t epoch,
                                        per_gpu_array<multimodal_batch_on_gpu_t> const &loaded,
                                        multimodal_batch_on_gpu_t &mixed, gpu_stream_t const &stream) noexcept;

    /** @brief The eager convenience: every member's slice in order over the per-GPU streams. */
    [[nodiscard]] expected_t operator()(epoch_t epoch, per_gpu_array<multimodal_batch_on_gpu_t> const &loaded,
                                        per_gpu_array<multimodal_batch_on_gpu_t> &mixed,
                                        per_gpu_array<gpu_stream_t> const &streams) noexcept;

  private:
    /** @brief The main-owned multimem_domain — borrowed; its barrier brackets every mix. */
    multimem_domain_t *multimem_ = nullptr;
    /** @brief The run's routing seed; combined with the per-call epoch inside the routing bijection. */
    u64_t seed_ = 0;
    /** @brief The baked batch capacities every route and repack cursor is bounded by. */
    multimodal_batch_shape_t shape_ {};
};

} // namespace unum::uform::core
