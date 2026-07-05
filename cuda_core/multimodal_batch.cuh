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
 *
 *  @section multimodal_batch_adapters Content is loader-tiled and patch-arranged; the batch only derives position
 *
 *  A token carries two things: its @b content (a patch vector, or a scalar id) and its @b position (where
 *  it sits in the encoder's shared coordinate space). The content is stored ready — the data-loader tiles
 *  each modality into the uniform patch grid ( @ref images , @ref videos ) and the batch keeps only the
 *  small per-item @ref image_shapes / @ref video_shapes descriptor. @ref image_coords , @ref video_coords ,
 *  and @ref token_coords are the batch's whole contribution to position: each reads the descriptor (or the
 *  segment geometry) and writes the shared @c (temporal,height,width) coordinate, so one uniform encoder
 *  serves every modality, rotating against the coordinate blind to which modality produced it.
 *
 *      Modality    Content stored as     Coordinate @c (t,h,w)
 *      --------    ------------------    -------------------------------------------------
 *      text        a scalar token id     @c (pos,0,0) — 1-D, the temporal axis is sequence
 *      image       a patch vector        @c (0,row,col) — 2-D spatial
 *      video       a tube vector         @c (frame,row,col) — 3-D
 *      audio       a mel patch           @c (frame,mel_band,0) — time on the temporal axis
 *
 *  @section multimodal_batch_grid Geometry is a factorization of the row budget, not a baked shape
 *
 *  The token count per media item is fixed — it equals @c max_columns_per_row (one encoder token per
 *  patch), so a media item fills a whole row. But the @b grid that count factors into is free: 4096
 *  patches can tile a 64×64 image, a 32×128 one, or a 128×32 one, and the encoder cannot tell — it is
 *  still 4096 tokens at one width. The only thing that changes is the coordinate the descriptor drives.
 *  So @b aspect ratio is a positional-encoding-shape choice (the item's @ref image_shapes picks the
 *  factorization), and @b video spatial-vs-temporal resolution is likewise a runtime grid choice: the
 *  same clip fed at @c (T=16,16×16) and @c (T=4,32×32) both fill the row, differing only in coordinates.
 *  Absolute time on the temporal axis (a tube at real frame @c T carries coord @c T regardless of
 *  sampling density) is what makes those multi-resolution views commensurable — one video file can drive
 *  several inputs at different tradeoffs, sharing one coordinate frame.
 *
 *  A corollary the masking cell reuses: tube masking is just a @b key the mask draws on @c (h,w) and
 *  broadcasts over @c t — derived from the coordinates, never a separate temporal-group parameter.
 *
 *  The batch owns no learnable weights and cuts no pixels — tiling is the loader's, projection and M-RoPE
 *  rotation are the tower's. The coordinate adapters are pure @ref launch_kernel enqueues over the
 *  descriptor, carrying no content and no vendor plan.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <memory> // `std::allocator_traits` — the batch's per-type rebinding

#include <numkong/tensor.hpp> // `nk::tensor` — the owning storage

#include "cuda_core/types.cuh"                  // the scalar aliases, `token_id_t`, `pixel_t`
#include "cuda_core/array.cuh"                  // `per_gpu_array` — the composition root's per-GPU tables
#include "cuda_core/gpu_allocators.cuh"         // `gpu_arena_alloc_t`, `pinned_alloc_t`
#include "cuda_core/schedule.cuh"               // `gpu_stream_t` — the mixer's enqueue surface
#include "cuda_core/multimem_collectives.cuh"   // `multimem_domain_t` — the barrier the exchange brackets
#include "cuda_core/infiniband_collectives.cuh" // `colocated_domain_t` — the cross-host_domain lane

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

/** @brief Scalars in one image patch — @c patch_size²·channels , the projection fan-in the tower peels
 *      @c (patch_side,patch_side,channels) into; channels are the innermost rank. */
[[nodiscard]] constexpr u32_t patch_scalars(u32_t patch_size, u32_t channels) noexcept {
    return patch_size * patch_size * channels;
}

/** @brief Scalars in one video tube — @c tube_frames·patch_scalars(side,channels) , the projection fan-in
 *      the tower peels @c (tube_frames,patch_side,patch_side,channels) into. */
[[nodiscard]] constexpr u32_t tube_scalars(u32_t tube_frames, u32_t side, u32_t channels) noexcept {
    return tube_frames * patch_scalars(side, channels);
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

    /** @brief The RGB channel count every image / video pixel carries — the innermost content rank. */
    u32_t image_channels = 3;
    /** @brief The square patch side the image modality tiles into, in pixels. Zero leaves the image
     *      tensors unallocated; nonzero sizes @ref multimodal_batch::images to
     *      @c [rows,columns,patch_size,patch_size,channels] — every image is @c max_columns_per_row square
     *      patches whatever its aspect. */
    u32_t image_patch_size = 0;
    /** @brief The temporal depth of one video tube in frames and its spatial patch side in pixels. Either
     *      being zero leaves the video tensors unallocated; nonzero sizes @ref multimodal_batch::videos to
     *      @c [rows,columns,tube_frames,patch_size,patch_size,channels] . */
    u32_t video_tube_frames = 0, video_patch_size = 0;
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
    /** @brief @b [rows,columns] discrete token ids — text is the degenerate modality, a scalar per grid
     *      position whose coordinate is @c (position,0,0) . Every media modality lifts this same
     *      @c [rows,columns] grid one rank higher, replacing the scalar with a patch vector; one encoder
     *      then serves them all because they share the grid and differ only in what a cell holds and where
     *      its coordinate places it. */
    nk::tensor<token_id_t, rebound<token_id_t>, 2> tokens {};
    /** @brief @b [rows,columns,patch_side,patch_side,channels] patch-arranged pixels — every image is
     *      exactly @c max_columns_per_row square patches whatever its source aspect or resolution, so this
     *      content tensor keeps one fixed shape the whole batch through. A data-loader only tiles pixels
     *      into this grid and records @ref image_shapes ; it never learns the encoder's coordinate
     *      convention or rotary geometry — loader and model share only the @c (content,shapes) contract, and
     *      M-RoPE stays the tower's business. @c channels is the innermost rank, so the entry peels
     *      @c (patch_side,patch_side,channels) into one projection fan-in with a @c reshape<2> . */
    nk::tensor<pixel_t, rebound<pixel_t>, 5> images {};
    /** @brief @b [rows,2] each image's @c (patch_rows,patch_cols) factorization of @c max_columns_per_row —
     *      the @b only place an image's aspect ratio survives. A 64×64, a 32×128, and a 128×32 image are
     *      the same @c columns -patch @ref images tensor and differ solely in this descriptor, which the
     *      tower turns into M-RoPE @c (t,h,w) coordinates. Aspect is a positional-encoding fact, not a
     *      storage one, so the model eats arbitrarily shaped inputs with no change to the loader, the tensor
     *      rank, or the batch geometry. */
    nk::tensor<u16_t, rebound<u16_t>, 2> image_shapes {};
    /** @brief @b [rows,columns,tube_frames,patch_side,patch_side,channels] patch-arranged video tubes —
     *      images with a temporal depth folded into the content axis, still exactly @c columns tubes per
     *      row and carrying the same @c (content,shapes) contract as @ref images . */
    nk::tensor<pixel_t, rebound<pixel_t>, 6> videos {};
    /** @brief @b [rows,3] each clip's @c (temporal_groups,patch_rows,patch_cols) factorization of
     *      @c columns . One clip fed at different spatial-versus-temporal resolutions is one @ref videos
     *      tensor differing only in this descriptor and the absolute-frame temporal coordinate the tower
     *      derives from it, so multi-resolution views of a single file stay commensurable. */
    nk::tensor<u16_t, rebound<u16_t>, 2> video_shapes {};
    /** @brief @b [clips,samples] raw mono waveforms, one item per audio row — the audio modality's
     *      mel-patch storage lands with the Phase-6 loader; this raw-waveform staging is interim. */
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
        if (shape.image_patch_size != 0) {
            batch.images = nk::tensor<pixel_t, rebound<pixel_t>, 5>::try_empty(
                {shape.max_rows_per_gpu, shape.max_columns_per_row, shape.image_patch_size, shape.image_patch_size,
                 shape.image_channels},
                rebound<pixel_t> {allocator});
            batch.image_shapes = nk::tensor<u16_t, rebound<u16_t>, 2>::try_empty({shape.max_rows_per_gpu, 2u},
                                                                                 rebound<u16_t> {allocator});
            if (!batch.images || !batch.image_shapes)
                return result_t::fail(status_t::bad_alloc_k, "image allocation failed ({} rows of {} patches)",
                                      shape.max_rows_per_gpu, shape.max_columns_per_row);
        }
        if (shape.video_tube_frames != 0 && shape.video_patch_size != 0) {
            batch.videos = nk::tensor<pixel_t, rebound<pixel_t>, 6>::try_empty(
                {shape.max_rows_per_gpu, shape.max_columns_per_row, shape.video_tube_frames, shape.video_patch_size,
                 shape.video_patch_size, shape.image_channels},
                rebound<pixel_t> {allocator});
            batch.video_shapes = nk::tensor<u16_t, rebound<u16_t>, 2>::try_empty({shape.max_rows_per_gpu, 3u},
                                                                                 rebound<u16_t> {allocator});
            if (!batch.videos || !batch.video_shapes)
                return result_t::fail(status_t::bad_alloc_k, "video allocation failed ({} rows of {} tubes)",
                                      shape.max_rows_per_gpu, shape.max_columns_per_row);
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

/** @brief Rank-5 pixel view of an image group, shaped @b [rows,columns,patch_side,patch_side,channels] —
 *      the patch-arranged content the tower entry peels ( @c reshape<2> ) and projects. */
using pixel_patches_view_t = nk::tensor_view<pixel_t, 5>;
/** @brief Rank-6 pixel view of a video group, shaped
 *      @b [rows,columns,tube_frames,patch_side,patch_side,channels] . */
using pixel_tubes_view_t = nk::tensor_view<pixel_t, 6>;
/** @brief Per-image @c (patch_rows,patch_cols) shapes, shaped @b [rows,2] — the aspect factorization the
 *      tower's coordinate derivation reads. */
using image_shape_view_t = nk::tensor_view<u16_t, 2>;
/** @brief Per-clip @c (temporal_groups,patch_rows,patch_cols) shapes, shaped @b [rows,3] . */
using video_shape_view_t = nk::tensor_view<u16_t, 2>;

/** @brief Write each image patch's coordinate @c (0,patch_row,patch_col) into @p coords @b [tokens,3] ,
 *      one image per @p image_shapes row — a pure enqueue and the only thing the batch derives for images,
 *      the content itself being stored patch-arranged (loader-tiled) in @ref multimodal_batch::images .
 *      Each image's @c (patch_rows,patch_cols) grid comes from @p image_shapes , so any aspect rides the
 *      coordinates with no crop or distortion. PAD rows write nothing, so the caller pre-zeroes @p coords . */
[[nodiscard]] expected_t image_coords(image_shape_view_t image_shapes, i16r2_span_t coords,
                                      gpu_stream_t &stream) noexcept;

/** @brief Write each video tube's coordinate @c (time_group·tube_frames,row_group,col_group) into
 *      @p coords @b [tokens,3] , one clip per @p video_shapes row — a pure enqueue. The temporal axis
 *      carries the tube's absolute start frame, so a clip sampled at any temporal resolution shares one
 *      coordinate frame; @p video_shapes gives each clip's @c (temporal_groups,patch_rows,patch_cols) .
 *      PAD rows write nothing. */
[[nodiscard]] expected_t video_coords(video_shape_view_t video_shapes, i16r2_span_t coords, u32_t tube_frames,
                                      gpu_stream_t &stream) noexcept;

/** @brief Write each text token's coordinate @c (position,0,0) into @p coords @b [tokens,3] , one
 *      segment's positions restarting at 0 — pure enqueue. @p segment_offsets / @p segment_lengths
 *      are the batch's packed geometry; PAD segments ( @c length==0 ) write nothing, so the caller
 *      must pre-zero @p coords and uncovered positions stay @c (0,0,0) . */
[[nodiscard]] expected_t token_coords(u32r1_view_t segment_offsets, u32r1_view_t segment_lengths, i16r2_span_t coords,
                                      gpu_stream_t &stream) noexcept;

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
                                        multimodal_batch_on_gpu_t &mixed, gpu_stream_t &stream) noexcept;

    /** @brief The eager convenience: every member's slice in order over the per-GPU streams. */
    [[nodiscard]] expected_t operator()(epoch_t epoch, per_gpu_array<multimodal_batch_on_gpu_t> const &loaded,
                                        per_gpu_array<multimodal_batch_on_gpu_t> &mixed,
                                        per_gpu_array<gpu_stream_t> &streams) noexcept;

  private:
    /** @brief The main-owned multimem_domain — borrowed; its barrier brackets every mix. */
    multimem_domain_t *multimem_ = nullptr;
    /** @brief The run's routing seed; combined with the per-call epoch inside the routing bijection. */
    u64_t seed_ = 0;
    /** @brief The baked batch capacities every route and repack cursor is bounded by. */
    multimodal_batch_shape_t shape_ {};
};

} // namespace unum::uform::core
