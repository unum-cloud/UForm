/**
 *  @file include/uform/encoder/multimodal_batch.hpp
 *  @author Ash Vardanian
 *  @date July 3, 2026
 *  @brief Memory-owning multimodal training batches, one per GPU.
 *
 *  The allocator template argument decides residency: a @c pinned_alloc_t instance is what a host
 *  packer writes, a @c gpu_arena_alloc_t instance is what the towers read, and one @c copy_from
 *  between the two is the whole transfer. Distribution across the multimem_domain and the
 *  colocated_domain is @ref unum::uform::encoder::multimodal_batch_mixer_t 's job; batches never
 *  know they are distributed. This header is the contract every build shares and compiles without
 *  CUDA; the NVIDIA residencies, the coordinate fill and the mixer live in
 *  @c multimodal_batch.cuh .
 *
 *  @section multimodal_batch_rows Rectangular rows, segments of any modality
 *
 *  A batch is a @b [rows,columns] rectangle of fixed-width rows packing segments back to back; each
 *  segment carries its own @ref unum::uform::encoder::multimodal_batch::segment_modality tag rather
 *  than a row band implying it, so a row can hold segments of any modality — thumbnails, a caption,
 *  and an audio clip together. A single loader fills one modality (it owns the whole segment
 *  table), and the @b mixer interleaves modalities across GPUs, so that free mixing is a property
 *  of the @b post-mix batch, not of one loader's fill. Media segments carry their @b raw signals —
 *  pixels, raw waveform frames — and every learnable transform (patch cut, spectral frontend,
 *  projection, embedding lookup) happens inside the differentiated step with fresh weights, never
 *  at packing time.
 *
 *  @section multimodal_batch_pipeline How batches flow through a training step
 *
 *  Each GPU loads a few random dataset shards and samples a few batches of various modality
 *  content. In Omni models the GPU itself chooses how many examples of each modality to take, so
 *  that their total fits @c max_segments_per_gpu — media modalities are typically long enough that
 *  one segment covers its whole row. The loaded multimodal batches are then shuffled between the
 *  GPUs of the multimem_domain and across the colocation via InfiniBand — the mixer's job.
 *
 *  Under the @b masked objective, once inputs are exchanged, each GPU itself decides how to corrupt
 *  every modality's contents — beyond simple masking, any distortion is allowed — and reconstructs
 *  at the output: training runs on two instances of the same batch, the clean one as the
 *  reconstruction target, the distorted one as the input. Under the @b contrastive objective the
 *  input stays uncorrupted and the relations graph rides separately (see
 *  @c contrastive_relations.cuh ) — with extra attention in the mixer so pairs and anchors land on
 *  the same GPU.
 *
 *  @section multimodal_batch_capacity Capacity once, PAD through data
 *
 *  Every tensor is capacity-allocated at @ref unum::uform::encoder::multimodal_batch::make and
 *  keeps constant extents. PAD segments repeat the previous offset (live segments are never empty
 *  by the packer's contract) — existence is always read from the data, and a captured step replays
 *  against freshly copied contents at stable addresses.
 *
 *  @section multimodal_batch_adapters Content is loader-tiled and patch-arranged; the batch only derives position
 *
 *  A token carries two things: its @b content (a patch vector, a raw sample chunk, or a scalar id)
 *  and its @b position (where it sits in the encoder's shared coordinate space). The content is
 *  stored ready — the loader tiles the 2-D / 3-D modalities into the patch grid (
 *  @ref unum::uform::encoder::multimodal_batch::images ,
 *  @ref unum::uform::encoder::multimodal_batch::videos ) and lays the 1-D ones as plain frame runs
 *  ( @ref unum::uform::encoder::multimodal_batch::audio_samples ,
 *  @ref unum::uform::encoder::multimodal_batch::tokens ); the batch keeps only the small
 *  per-segment @ref unum::uform::encoder::multimodal_batch::image_shapes /
 *  @ref unum::uform::encoder::multimodal_batch::video_shapes descriptor (audio and text need none).
 *  @ref unum::uform::encoder::fill_coords is the batch's whole contribution to position: one CTA
 *  per segment dispatches on @ref unum::uform::encoder::multimodal_batch::segment_modality , reads
 *  the matching descriptor (or the segment geometry) and writes the shared
 *  @c (temporal,height,width) coordinate, so one uniform encoder serves every modality, rotating
 *  against the coordinate blind to which modality produced it.
 *
 *      Modality    Content stored as     Coordinate @c (t,h,w)
 *      text        a scalar token id     @c (pos,0,0) — 1-D, the temporal axis is sequence
 *      image       a patch vector        @c (0,row,col) — 2-D spatial
 *      video       a tube vector         @c (frame,row,col) — 3-D
 *      audio       a raw waveform chunk  @c (frame,0,0) — 1-D, identical to text: a run of frames
 *
 *  @section multimodal_batch_grid Geometry is derived from content, not a baked shape
 *
 *  A media segment's column footprint is @b derived from its decoded content — an image's native
 *  aspect factors into @ref unum::uform::encoder::multimodal_batch::image_shapes , a clip's into
 *  @ref unum::uform::encoder::multimodal_batch::video_shapes — clamped to a row, so segments are
 *  variable-width and several share a row. The encoder sees only tokens at one width; the
 *  factorization changes only the coordinate the descriptor drives, so @b aspect ratio and @b video
 *  spatial-vs-temporal resolution are positional-encoding choices, not storage ones. One clip
 *  sampled two ways (a dense and a sparse @c frame_stride ) is two segments sharing one coordinate
 *  frame: absolute time on the temporal axis (a tube at real frame @c T carries coord @c T
 *  regardless of sampling density) is what makes those representations commensurable.
 *
 *  A corollary the masking cell reuses: tube masking is just a @b key the mask draws
 *  on @c (h,w) and broadcasts over @c t — derived from the coordinates, never a
 *  separate temporal-group parameter.
 *
 *  The batch owns no learnable weights and cuts no pixels — tiling is the loader's, projection and
 *  M-RoPE rotation are the tower's. The coordinate adapters are pure @c launch_kernel enqueues over
 *  the descriptor, carrying no content and no vendor plan.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <memory> // `std::allocator_traits` — the batch's per-type rebinding

#include <numkong/tensor.hpp> // `nk::tensor` — the owning storage

#include "uform/utilities/types.hpp" // the scalar aliases, `token_id_t`, `pixel_t`

namespace unum::uform::encoder {
using namespace unum::uform::utilities;

/** The byte vocabulary: 256 UTF-8 byte values plus the PAD and MASK specials. */
inline constexpr u32_t byte_vocabulary_k = 258u;

/** The PAD byte ID — token 256; ineligible for masking, its loss weight is always zero. */
inline constexpr token_id_t byte_pad_k = 256u;

/** The MASK byte ID — token 257, what the replace fraction of masked positions shows the tower, and
 *  the position a choice is read at. */
inline constexpr token_id_t byte_mask_k = 257u;

/** One segment's input kind — which raw tensor feeds it, which RoPE pattern it walks, and, under
 *  masking, which reconstruction head scores it. Carried per segment (any row may mix modalities),
 *  so the coordinate adapter and the tower route on it. Scoped over @c u8_t — the whole batch is
 *  typed, never a raw tag byte. */
enum class modality_t : u8_t {
    text_k = 0,
    image_k = 1,
    video_k = 2,
    audio_k = 3,
};

/** Rank-1 read view over a segment's @ref modality_t tags — @b [max_segments_per_gpu] . */
using modality_view_t = nk::tensor_view<modality_t, 1>;

/** Rank-1 mutable span over a segment's @ref modality_t tags — @b [max_segments_per_gpu] . */
using modality_span_t = nk::tensor_span<modality_t, 1>;

/** One segment's part in its bag: the anchor and its positives attract each other, and a negative
 *  repels the anchor while it ignores the positives and the other negatives. */
enum class segment_role_t : u8_t {
    anchor_k = 0,
    positive_k = 1,
    negative_k = 2,
};

/** Which side of a late-interaction score a segment takes: a query's tokens score a document's,
 *  and a symmetric pair plays both. */
enum class late_interaction_side_t : u8_t {
    none_k = 0,
    query_k = 1,
    document_k = 2,
    query_and_document_k = 3,
};

/** Scalars in one image patch — @c patch_size² · channels , the projection fan-in the tower peels
 *  @c (patch_side,patch_side,channels) into; channels are the innermost rank. */
[[nodiscard]] constexpr u32_t patch_scalars(u32_t patch_size, u32_t channels) noexcept {
    return patch_size * patch_size * channels;
}

/** Scalars in one video tube — @c tube_frames · patch_scalars(side,channels) , the projection
 *  fan-in the tower peels @c (tube_frames,patch_side,patch_side,channels) into. */
[[nodiscard]] constexpr u32_t tube_scalars(u32_t tube_frames, u32_t side, u32_t channels) noexcept {
    return tube_frames * patch_scalars(side, channels);
}

/** The image modality's slice of a batch shape — everything the image path needs and
 *  nothing of any other modality. A zero @ref patch_side leaves the image content tensor
 *  unallocated and the modality out of the composition; the image loader is handed this
 *  alone, never the sibling modalities. */
struct image_batch_shape_t {

    /** The square patch side the image tiles into, in pixels; nonzero sizes
     *  @ref multimodal_batch::images to @c [rows,columns,patch_side,patch_side,channels] . Zero
     *  disables the modality. */
    u32_t patch_side = 0;

    /** The channel count every image pixel carries — RGBA end to end, the
     *  innermost content rank. */
    u32_t channels = 4;
};

/**
 *  @brief The video modality's slice of a batch shape — the tube geometry a clip is decoded
 *      and patchified into.
 *
 *  A clip is decoded @b once at its native resolution and tiled exactly like an image with a
 *  temporal axis folded ahead of the spatial patch; @ref frame_stride selects which source frames a
 *  representation samples, and the footprint is derived from the decoded resolution and clamped to
 *  a row (an oversized clip downscales to fit), never a fixed budget. Independent of the image
 *  modality (its own patch side and channels). A zero @ref patch_side disables the modality.
 */
struct video_batch_shape_t {

    /** The square patch side one video tube tiles into, in pixels; nonzero sizes
     *  @ref multimodal_batch::videos to
     *  @c [rows,columns,tube_frames,patch_side,patch_side,channels] . Zero
     *  disables the modality. */
    u32_t patch_side = 0;

    /** The channel count every video pixel carries — independent of image, the
     *  innermost content rank. */
    u32_t channels = 3;

    /** The temporal depth of one video tube in frames — the content axis folded ahead of
     *  the spatial patch. */
    u32_t tube_frames = 0;

    /** Sample every Nth source frame — the temporal density; a sparser stride spans more real
     *  time per tube. */
    u32_t frame_stride = 1;
};

/**
 *  @brief The audio modality's slice of a batch shape — and nothing of any other modality.
 *
 *  A waveform is intrinsically @b 1-D, so audio is not patched (patching only serializes the extra
 *  axes of 2-D images and 3-D video onto the column axis). Each column is one @b frame — a chunk of
 *  raw samples — and a segment is a run of frames, structurally identical to a text run; its
 *  coordinate is @c (frame,0,0) , the same position-only axis text walks. The mel / spectral
 *  transform is @b not computed here: it is a learnable first layer of the encoder, so the loader
 *  only resamples and chunks the waveform. Two params, mirroring the image modality; a zero
 *  @ref chunk_samples disables audio.
 */
struct audio_batch_shape_t {

    /** The raw samples per column-frame — the innermost extent of
     *  @ref multimodal_batch::audio_samples ( @c [rows,columns,chunk_samples] ) and the tower's
     *  learnable-frontend projection fan-in. Zero disables the modality. */
    u32_t chunk_samples = 0;

    /** The mono rate every clip is resampled to before chunking — the model's fixed input rate. */
    u32_t sample_rate = 16000;
};

/**
 *  @brief The baked capacities of one GPU's batch — a different shape is a different batch object.
 *
 *  A batch is a @b [max_rows_per_gpu,max_columns_per_row] rectangle: every row is an independent
 *  attention sequence of fixed width, and every segment (a document, an image, a clip) occupies a
 *  contiguous column span of exactly one row, several sharing a row back to back. A single row
 *  budget covers every modality — the composition is the loader's per-step choice — so each media
 *  tensor is sized for @c max_rows_per_gpu items of its raw extents; a modality with zero extents
 *  stays unallocated and out of the composition. The per-modality geometry is @b nested so each
 *  loader receives only its own slice ( @ref image_batch_shape_t / @ref video_batch_shape_t /
 *  @ref audio_batch_shape_t ), never a sibling's details — and each nested slice is the single
 *  source of its modality's processing parameters.
 */
struct multimodal_batch_shape_t {

    /** How many rows one GPU's step carries, whatever the modality mix. */
    u32_t max_rows_per_gpu = 0;

    /** How many segments one GPU's step carries, across all its rows — also the pooling axis
     *  the contrastive anchors live on. */
    u32_t max_segments_per_gpu = 0;

    /** How many positions one row spans — what most of the literature calls @c seq_len . */
    u32_t max_columns_per_row = 0;

    /** How many contrastive bags one GPU's step carries — the mixer's routing unit, so a bag's
     *  segments land together on one member. The @b count is latched capacity while each bag's
     *  segment degree is content (fixed capacity, variable degree). Zero keeps every segment its
     *  own routing unit — the masked pipeline's degenerate case and the default. */
    u32_t max_bags_per_gpu = 0;

    /** Most segments one bag holds: an anchor, its positives and its negatives, as the loader's
     *  roles allow. Zero means no loader groups segments into role bags. */
    u32_t max_bag_segments = 0;

    /** Most legal bytes one choice segment offers, the width of
     *  @ref multimodal_batch::segment_choices ; zero allocates no choice columns. */
    u32_t max_choices_per_segment = 0;

    /** The image modality's geometry (patch side, channels) — handed to the image loader alone. */
    image_batch_shape_t image {};

    /** The video modality's geometry (tube frames, patch side, channels, frame stride) — to
     *  the video loader. */
    video_batch_shape_t video {};

    /** The audio modality's geometry (chunk samples, sample rate) — handed to the
     *  audio loader alone. */
    audio_batch_shape_t audio {};

    /** Whether @p modality participates in this composition — text always, a media modality iff its
     *  nested slice is live (the "zero disables" rule above, spelled once). */
    [[nodiscard]] constexpr bool contains(modality_t modality) const noexcept {
        switch (modality) {
        case modality_t::text_k: return true;
        case modality_t::image_k: return image.patch_side != 0u;
        case modality_t::video_k: return video.patch_side != 0u && video.tube_frames != 0u;
        case modality_t::audio_k: return audio.chunk_samples != 0u;
        }
        return false;
    }

    /** Whether any media modality participates, which sets a run's row width. */
    [[nodiscard]] constexpr bool media_live() const noexcept {
        return contains(modality_t::image_k) || contains(modality_t::video_k) || contains(modality_t::audio_k);
    }
};

/** The cells one media item fills: half a row for an image or audio clip, so two share a row at
 *  columns 0 and columns/2, and the whole row for a video clip; zero for freely packed text. */
[[nodiscard]] constexpr u32_t media_cells(modality_t modality, u32_t columns) noexcept {
    if (modality == modality_t::video_k) return columns;
    if (modality == modality_t::image_k || modality == modality_t::audio_k) return columns / 2u;
    return 0u;
}

/** @p content viewed as @b [rows,scalars] , one row per grid cell or per segment, so a run of
 *  cells is a range of whole rows; an absent modality's empty view stays empty. */
[[nodiscard]] constexpr auto as_rows(auto &&content, std::size_t rows) noexcept {
    return content.template reshape<2>({rows, content.numel() / rows});
}

/** A read-only borrow of one @ref multimodal_batch , whatever its residency — what a tower
 *  reads. Each field views the batch tensor of the same name; an absent modality's content
 *  view is empty. */
struct multimodal_batch_view_t {
    nk::tensor_view<u32_t, 1> segment_offsets {};
    nk::tensor_view<u32_t, 1> segment_lengths {};
    nk::tensor_view<modality_t, 1> segment_modality {};
    nk::tensor_view<segment_role_t, 1> segment_role {};
    nk::tensor_view<u32_t, 1> segment_prefix_length {};
    nk::tensor_view<late_interaction_side_t, 1> segment_side {};
    nk::tensor_view<token_id_t, 2> tokens {};
    nk::tensor_view<pixel_t, 5> images {};
    nk::tensor_view<u16_t, 2> image_shapes {};
    nk::tensor_view<pixel_t, 6> videos {};
    nk::tensor_view<u16_t, 2> video_shapes {};
    nk::tensor_view<f32_t, 3> audio_samples {};
    nk::tensor_view<token_id_t, 2> segment_choices {};
    nk::tensor_view<f32_t, 2> segment_choice_targets {};
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

    /** @p alloc_type_ rebound to @p value_type_ — every tensor's allocator spelling. */
    template <typename value_type_>
    using rebound = typename std::allocator_traits<alloc_type_>::template rebind_alloc<value_type_>;

    /** @b [max_segments_per_gpu+1] segment START positions in the flat @c row · columns+column
     *  space (the last entry is the pack cursor's end). A position's row is its offset divided by
     *  the row width, and a segment never crosses its row — which leaves gaps before row jumps,
     *  so lengths are never adjacent differences; they live in @ref segment_lengths . */
    nk::tensor<u32_t, rebound<u32_t>, 1> segment_offsets {};

    /** @b [max_segments_per_gpu] live token counts — stored, not derived: the row-jump gaps
     *  between consecutive segments make adjacent offset differences overestimates. PAD
     *  segments carry 0. */
    nk::tensor<u32_t, rebound<u32_t>, 1> segment_lengths {};

    /** @b [max_segments_per_gpu] each live segment's @ref modality_t — a row may mix modalities, so
     *  the tag is per segment, not per row. The coordinate adapter dispatches on it, the mixer
     *  copies it, and the tower routes each segment to its modality's projection. PAD segments
     *  carry @c text_k . */
    nk::tensor<modality_t, rebound<modality_t>, 1> segment_modality {};

    /** @b [max_segments_per_gpu] each live segment's bag — the loader packs a bag's segments
     *  contiguously and tags them with its local bag index in @c [0,max_bags_per_gpu) , and the
     *  mixer routes whole bags, renumbering to the destination-local index. Meaningful only when
     *  the shape's @ref multimodal_batch_shape_t::max_bags_per_gpu is nonzero; a PAD segment's bag
     *  value is inert — the mixer's nonzero-length filter excludes it from every bag before the
     *  tag is consulted. */
    nk::tensor<u32_t, rebound<u32_t>, 1> segment_bag {};

    /** @b [max_segments_per_gpu] each segment's @ref segment_role_t in its bag; PAD segments carry
     *  @c anchor_k , inert. */
    nk::tensor<segment_role_t, rebound<segment_role_t>, 1> segment_role {};

    /** @b [max_segments_per_gpu] leading tokens of each segment no score reads, such as a
     *  "query: " prefix. */
    nk::tensor<u32_t, rebound<u32_t>, 1> segment_prefix_length {};

    /** @b [max_segments_per_gpu] each segment's sample identity, such as the media key; equal
     *  nonzero keys are one item, and zero names none. */
    nk::tensor<u64_t, rebound<u64_t>, 1> segment_key {};

    /** @b [max_segments_per_gpu] each segment's @ref late_interaction_side_t ; PAD and @c none_k
     *  segments are skipped. */
    nk::tensor<late_interaction_side_t, rebound<late_interaction_side_t>, 1> segment_side {};

    /** @b [rows,columns] discrete token ids — text is the degenerate modality, a scalar per grid
     *  position whose coordinate is @c (position,0,0) . Every media modality lifts this same
     *  @c [rows,columns] grid one rank higher, replacing the scalar with a patch vector; one
     *  encoder then serves them all because they share the grid and differ only in what a cell
     *  holds and where its coordinate places it. */
    nk::tensor<token_id_t, rebound<token_id_t>, 2> tokens {};

    /** @b [rows,columns,patch_side,patch_side,channels] patch-arranged pixels — each image is a
     *  sub-row segment of the patches its resolution earns, several sharing a row, so this content
     *  tensor keeps one fixed shape the whole batch through. A data-loader only tiles pixels into
     *  this grid and records @ref unum::uform::encoder::multimodal_batch::image_shapes ; it never
     *  learns the encoder's coordinate convention or rotary geometry — loader and model share only
     *  the @c (content,shapes) contract, and M-RoPE stays the tower's business. @c channels is the
     *  innermost rank, so the entry peels @c (patch_side,patch_side,channels) into one projection
     *  fan-in with a @c reshape<2> . */
    nk::tensor<pixel_t, rebound<pixel_t>, 5> images {};

    /** @b [max_segments_per_gpu,2] each image segment's @c (patch_rows,patch_cols) — the native
     *  aspect factored into a row-clamped footprint, the @b only place an image's aspect ratio
     *  survives, segment-indexed since several images may share a row. An entry is meaningful iff
     *  @ref unum::uform::encoder::multimodal_batch::segment_modality is @c image_k . The
     *  coordinate adapter turns it into M-RoPE @c (t,h,w) ; aspect is a positional-encoding fact,
     *  not a storage one. */
    nk::tensor<u16_t, rebound<u16_t>, 2> image_shapes {};

    /** @b [rows,columns,video_tube_frames,video_patch_side,video_patch_side,video_channels]
     *  patch-arranged video tubes — images with a temporal depth folded ahead of the spatial patch,
     *  each segment scattered into its column sub-range and carrying the same @c (content,shapes)
     *  contract as @ref unum::uform::encoder::multimodal_batch::images . */
    nk::tensor<pixel_t, rebound<pixel_t>, 6> videos {};

    /** @b [max_segments_per_gpu,3] each video segment's @c (temporal_groups,patch_rows,patch_cols)
     *  — its native resolution factored into a row-clamped footprint, segment-indexed. An entry is
     *  meaningful iff @ref unum::uform::encoder::multimodal_batch::segment_modality is @c video_k .
     *  One clip sampled at different frame strides is several segments differing only in this
     *  descriptor and the absolute-frame temporal coordinate. */
    nk::tensor<u16_t, rebound<u16_t>, 2> video_shapes {};

    /** @b [rows,columns,chunk_samples] the raw mono waveform, one chunk of @c chunk_samples samples
     *  per column-frame. Audio is 1-D and unpatched — a segment is a run of frames like a text run
     *  — so the loader only resamples and chunks the waveform; the mel / spectral frontend is a
     *  learnable first layer of the encoder. No per-segment shape descriptor: a frame's coordinate
     *  is @c (frame,0,0) , derived from the segment geometry alone. */
    nk::tensor<f32_t, rebound<f32_t>, 3> audio_samples {};

    /** @b [max_segments_per_gpu,max_choices_per_segment] each choice segment's legal bytes, padded
     *  with @c byte_pad_k ; a segment whose first entry is the pad makes no choice. Its last token
     *  carries @c byte_mask_k . */
    nk::tensor<token_id_t, rebound<token_id_t>, 2> segment_choices {};

    /** @b [max_segments_per_gpu,max_choices_per_segment] each choice's target probability,
     *  summing to one per segment; zero under the pads. */
    nk::tensor<f32_t, rebound<f32_t>, 2> segment_choice_targets {};

    /** @b [1] segments dropped whole because the batch had no room for them. Its owner zeroes it
     *  before a fill and reads it after, and every placement walk adds its drops; the mixer adds a
     *  loaded batch's count and its own drops to the mixed batch's. */
    nk::tensor<u32_t, rebound<u32_t>, 1> dropped {};

    /** The segment capacity of this batch — @c max_segments_per_gpu , PAD-inclusive, the pooling
     *  and routing axis every modality shares. Modalities do not own disjoint row bands, so a
     *  segment count (not a per-modality row count) is the only axis with a batch-wide meaning. */
    std::size_t live_segments() const noexcept { return segment_lengths.extent(0); }

    /** Every tensor, borrowed read-only. */
    [[nodiscard]] multimodal_batch_view_t view() const noexcept {
        return {segment_offsets.view(),
                segment_lengths.view(),
                segment_modality.view(),
                segment_role.view(),
                segment_prefix_length.view(),
                segment_side.view(),
                tokens.view(),
                images.view(),
                image_shapes.view(),
                videos.view(),
                video_shapes.view(),
                audio_samples.view(),
                segment_choices.view(),
                segment_choice_targets.view()};
    }

    /**
     *  @brief Allocate every tensor the shape names — once, at setup.
     *
     *  The geometry, token, per-segment
     *  @ref unum::uform::encoder::multimodal_batch::segment_modality , and the
     *  @ref unum::uform::encoder::multimodal_batch::image_shapes /
     *  @ref unum::uform::encoder::multimodal_batch::video_shapes descriptors always exist (they are
     *  tiny and segment-indexed; audio needs none — its frames carry a position-only coordinate); a
     *  modality's @b content tensor is gated on its patch side (audio: @c chunk_samples ) and stays
     *  unallocated when zero.
     */
    [[nodiscard]] static expected<multimodal_batch> make(alloc_type_ allocator,
                                                         multimodal_batch_shape_t shape) noexcept {
        using result_t = expected<multimodal_batch>;
        if (shape.max_rows_per_gpu == 0 || shape.max_segments_per_gpu == 0 || shape.max_columns_per_row == 0)
            return result_t::fail(status_t::invalid_argument_k,
                                  "a batch needs all three capacities (rows={} segments={} columns={})",
                                  shape.max_rows_per_gpu, shape.max_segments_per_gpu, shape.max_columns_per_row);
        if (shape.max_bags_per_gpu > shape.max_segments_per_gpu)
            return result_t::fail(status_t::invalid_argument_k,
                                  "bag budget exceeds segment budget (bags={} segments={}) — each bag needs a segment",
                                  shape.max_bags_per_gpu, shape.max_segments_per_gpu);
        multimodal_batch batch;
        batch.segment_offsets = nk::tensor<u32_t, rebound<u32_t>, 1>::uninitialized({shape.max_segments_per_gpu + 1u},
                                                                                    rebound<u32_t> {allocator})
                                    .value;
        batch.segment_lengths = nk::tensor<u32_t, rebound<u32_t>, 1>::uninitialized({shape.max_segments_per_gpu},
                                                                                    rebound<u32_t> {allocator})
                                    .value;
        batch.segment_modality = nk::tensor<modality_t, rebound<modality_t>, 1>::uninitialized(
                                     {shape.max_segments_per_gpu}, rebound<modality_t> {allocator})
                                     .value;
        batch.segment_bag = nk::tensor<u32_t, rebound<u32_t>, 1>::uninitialized({shape.max_segments_per_gpu},
                                                                                rebound<u32_t> {allocator})
                                .value;
        batch.segment_role = nk::tensor<segment_role_t, rebound<segment_role_t>, 1>::uninitialized(
                                 {shape.max_segments_per_gpu}, rebound<segment_role_t> {allocator})
                                 .value;
        batch.segment_prefix_length = nk::tensor<u32_t, rebound<u32_t>, 1>::uninitialized({shape.max_segments_per_gpu},
                                                                                          rebound<u32_t> {allocator})
                                          .value;
        batch.segment_key = nk::tensor<u64_t, rebound<u64_t>, 1>::uninitialized({shape.max_segments_per_gpu},
                                                                                rebound<u64_t> {allocator})
                                .value;
        batch.segment_side = nk::tensor<late_interaction_side_t, rebound<late_interaction_side_t>, 1>::uninitialized(
                                 {shape.max_segments_per_gpu}, rebound<late_interaction_side_t> {allocator})
                                 .value;
        batch.tokens = nk::tensor<token_id_t, rebound<token_id_t>, 2>::uninitialized(
                           {shape.max_rows_per_gpu, shape.max_columns_per_row}, rebound<token_id_t> {allocator})
                           .value;
        batch.image_shapes = nk::tensor<u16_t, rebound<u16_t>, 2>::uninitialized({shape.max_segments_per_gpu, 2u},
                                                                                 rebound<u16_t> {allocator})
                                 .value;
        batch.video_shapes = nk::tensor<u16_t, rebound<u16_t>, 2>::uninitialized({shape.max_segments_per_gpu, 3u},
                                                                                 rebound<u16_t> {allocator})
                                 .value;
        batch.dropped = nk::tensor<u32_t, rebound<u32_t>, 1>::uninitialized({1u}, rebound<u32_t> {allocator}).value;
        if (!batch.segment_offsets || !batch.segment_lengths || !batch.segment_modality || !batch.segment_bag ||
            !batch.segment_role || !batch.segment_prefix_length || !batch.segment_key || !batch.segment_side ||
            !batch.tokens || !batch.image_shapes || !batch.video_shapes || !batch.dropped)
            return result_t::fail(status_t::bad_alloc_k, "batch allocation failed (rows={} segments={} columns={})",
                                  shape.max_rows_per_gpu, shape.max_segments_per_gpu, shape.max_columns_per_row);
        if (shape.image.patch_side != 0) {
            batch.images = nk::tensor<pixel_t, rebound<pixel_t>, 5>::uninitialized(
                               {shape.max_rows_per_gpu, shape.max_columns_per_row, shape.image.patch_side,
                                shape.image.patch_side, shape.image.channels},
                               rebound<pixel_t> {allocator})
                               .value;
            if (!batch.images)
                return result_t::fail(status_t::bad_alloc_k, "image allocation failed ({} rows of {} patches)",
                                      shape.max_rows_per_gpu, shape.max_columns_per_row);
        }
        if (shape.video.patch_side != 0) {
            batch.videos = nk::tensor<pixel_t, rebound<pixel_t>, 6>::uninitialized(
                               {shape.max_rows_per_gpu, shape.max_columns_per_row, shape.video.tube_frames,
                                shape.video.patch_side, shape.video.patch_side, shape.video.channels},
                               rebound<pixel_t> {allocator})
                               .value;
            if (!batch.videos)
                return result_t::fail(status_t::bad_alloc_k, "video allocation failed ({} rows of {} tubes)",
                                      shape.max_rows_per_gpu, shape.max_columns_per_row);
        }
        if (shape.audio.chunk_samples != 0) {
            batch.audio_samples = nk::tensor<f32_t, rebound<f32_t>, 3>::uninitialized(
                                      {shape.max_rows_per_gpu, shape.max_columns_per_row, shape.audio.chunk_samples},
                                      rebound<f32_t> {allocator})
                                      .value;
            if (!batch.audio_samples)
                return result_t::fail(status_t::bad_alloc_k, "audio allocation failed ({} rows of {} frames)",
                                      shape.max_rows_per_gpu, shape.max_columns_per_row);
        }
        if (shape.max_choices_per_segment != 0) {
            batch.segment_choices = nk::tensor<token_id_t, rebound<token_id_t>, 2>::uninitialized(
                                        {shape.max_segments_per_gpu, shape.max_choices_per_segment},
                                        rebound<token_id_t> {allocator})
                                        .value;
            batch.segment_choice_targets = nk::tensor<f32_t, rebound<f32_t>, 2>::uninitialized(
                                               {shape.max_segments_per_gpu, shape.max_choices_per_segment},
                                               rebound<f32_t> {allocator})
                                               .value;
            if (!batch.segment_choices || !batch.segment_choice_targets)
                return result_t::fail(status_t::bad_alloc_k, "choice allocation failed ({} segments of {} choices)",
                                      shape.max_segments_per_gpu, shape.max_choices_per_segment);
        }
        return result_t {std::move(batch)};
    }
};

/** Rank-5 pixel view of an image group, shaped @b [rows,columns,patch_side,patch_side,channels] —
 *  the patch-arranged content the tower entry peels ( @c reshape<2> ) and projects. */
using pixel_patches_view_t = nk::tensor_view<pixel_t, 5>;

/** Rank-6 pixel view of a video group, shaped
 *  @b [rows,columns,tube_frames,patch_side,patch_side,channels] . */
using pixel_tubes_view_t = nk::tensor_view<pixel_t, 6>;

/** Rank-3 read view of the raw audio frames, shaped @b [rows,columns,chunk_samples] . */
using audio_samples_view_t = nk::tensor_view<f32_t, 3>;

/** Mutable span twin of @ref audio_samples_view_t — @b [rows,columns,chunk_samples] ,
 *  the loader writes. */
using audio_samples_span_t = nk::tensor_span<f32_t, 3>;

/** Rank-1 read view over the per-segment @ref modality_t tags — @b [max_segments_per_gpu] . */
using segment_modality_view_t = nk::tensor_view<modality_t, 1>;

/** Rank-1 mutable span over the per-segment @ref modality_t tags — @b [max_segments_per_gpu] . */
using segment_modality_span_t = nk::tensor_span<modality_t, 1>;

/** Per-segment @c (patch_rows,patch_cols) image shapes, shaped @b [max_segments_per_gpu,2] — the
 *  aspect factorization the coordinate adapter reads. */
using image_shape_view_t = nk::tensor_view<u16_t, 2>;

/** Mutable span twin of @ref image_shape_view_t — @b [max_segments_per_gpu,2] ,
 *  the loader writes. */
using image_shape_span_t = nk::tensor_span<u16_t, 2>;

/** Per-segment @c (temporal_groups,patch_rows,patch_cols) video shapes, @b [max_segments_per_gpu,3]
 *  — a rank-2 view over the three-wide descriptor, matching the @c [seg,3] extents. */
using video_shape_view_t = nk::tensor_view<u16_t, 2>;

/** Mutable span twin of @ref video_shape_view_t — @b [max_segments_per_gpu,3] . */
using video_shape_span_t = nk::tensor_span<u16_t, 2>;

} // namespace unum::uform::encoder
