/**
 *  @brief Image and video planning: how many patches an item earns, how they factor, and what gets cropped.
 *  @file src/media/planning.hpp
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  The NVIDIA loaders, the host decoders and inference all plan through these functions, so an item tiles into
 *  the same grid on every residency. They are pure integer arithmetic over native extents and the batch
 *  geometry, with no decoding and no allocation.
 */
#pragma once

#include "core/types.hpp"            // the scalar aliases
#include "core/multimodal_batch.hpp" // `video_batch_shape_t`

namespace unum::uform::media {

namespace uc = unum::uform::core;
namespace nk = ashvardanian::numkong;
namespace st = ashvardanian::smashtable;

using uc::f64_t;
using uc::u16_t;
using uc::u32_t;
using uc::u64_t;
using uc::video_batch_shape_t;

/** A picture's pixel dimensions — a size travels as one value, not two loose scalars. */
struct image_extent_t {

    /** The pixel width. */
    u32_t width = 0;

    /** The pixel height. */
    u32_t height = 0;
};

/** The @c (rows,columns) factorization of the patch budget one image tiles into — its column sub-budget in a
 *  batch row and the shape its resized pixels take, @c (columns·patch_side,rows·patch_side) . */
struct patch_grid_t {

    /** The patch rows the image tiles into. */
    u16_t rows;

    /** The patch columns the image tiles into. */
    u16_t columns;
};

/** A source rectangle in pixels, the region a resize reads. */
struct crop_t {

    /** The left edge. */
    u32_t x = 0;

    /** The top edge. */
    u32_t y = 0;

    /** The rectangle's width. */
    u32_t width = 0;

    /** The rectangle's height. */
    u32_t height = 0;
};

/** A video segment's 3-D patch factorization; its product is the segment's column footprint. */
struct video_grid_t {

    /** The tube time-windows the sampled frames pack into. */
    u16_t temporal_groups = 1;

    /** The spatial patch rows every frame is scaled to. */
    u16_t patch_rows = 1;

    /** The spatial patch columns every frame is scaled to. */
    u16_t patch_columns = 1;
};

/**
 *  @brief One image's column budget — how many patches it earns from its native resolution, so a bigger image
 *      packs more patches and several small images share a row.
 *
 *  The native patch count, the @c patch_side² tiles covering the source ceiled per axis, is rounded @b up to a
 *  multiple of the floor budget @c columns/segments for a divisor-friendly @ref image_patch_grid , then clamped to
 *  @c [columns/segments,columns] . Monotonic in resolution, and bounded so at most @p segments images fill a row.
 *
 *  @param[in] native The image's full encoded resolution.
 *  @param[in] patch_side The square patch side in pixels.
 *  @param[in] columns The batch row width in patches.
 *  @param[in] segments The most images one batch carries.
 */
[[nodiscard]] u32_t image_patch_budget(image_extent_t native, u32_t patch_side, u32_t columns, u32_t segments) noexcept;

/** The factor pair of @p budget whose @c columns/rows aspect is nearest @p native 's, so every image is exactly
 *  @p budget patches and its rectangle rides the coordinates without a crop toward a square. */
[[nodiscard]] patch_grid_t image_patch_grid(u32_t budget, image_extent_t native) noexcept;

/** The centered crop of @p source matching @p target 's aspect — the largest rectangle of that aspect that fits,
 *  so the resize that follows never stretches. */
[[nodiscard]] crop_t image_center_crop(image_extent_t source, image_extent_t target) noexcept;

/** The largest @c k≤3 whose @c 1/2ᵏ JPEG DCT scaling still covers @p target — decode as small as possible
 *  without dropping below the resize target, so the resize only downsamples. Numerically equal to nvJPEG's
 *  @c nvjpegScaleFactor_t and to libjpeg's @c scale_denom=2ᵏ . */
[[nodiscard]] u32_t image_decode_scale(image_extent_t native, image_extent_t target) noexcept;

/** The extent a @c 1/2^scale DCT-scaled decode lands on, a ceiling division per axis — the rounding nvJPEG and
 *  libjpeg share. */
[[nodiscard]] image_extent_t image_decoded_extent(image_extent_t native, u32_t scale) noexcept;

/**
 *  @brief A video segment's @c (temporal_groups,patch_rows,patch_columns) from the clip's native
 *      @c (frames,width,height) — the image budget's analogue with a temporal axis folded ahead of the patch.
 *
 *  The rows and columns tile the native frame at the shape's @c patch_side ; the temporal groups pack the
 *  @c frame_stride -sampled frames into @c tube_frames -deep tubes. A footprint over @p columns is scaled toward
 *  its own shape by a cube root, then trimmed on its largest axis until it fits one row, so an arbitrarily large
 *  clip downscales instead of overflowing.
 *
 *  @param[in] frames The clip's native frame count, 0 when unknown.
 *  @param[in] native The clip's native frame extent.
 *  @param[in] shape The video modality's geometry: patch side, tube depth and frame stride.
 *  @param[in] columns The batch row width in patches.
 */
[[nodiscard]] video_grid_t video_tube_grid(u32_t frames, image_extent_t native, video_batch_shape_t const &shape,
                                           u32_t columns) noexcept;

} // namespace unum::uform::media
