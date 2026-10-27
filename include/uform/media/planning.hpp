/**
 *  @file include/uform/media/planning.hpp
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *  @brief Image and video planning: how many patches an item earns, how they factor, and what gets
 *      cropped.
 *
 *  The NVIDIA loaders, the host decoders and inference all plan through these functions, so an item tiles into
 *  the same grid on every residency. They are pure integer arithmetic over native extents and the batch
 *  geometry, with no decoding and no allocation.
 */
#pragma once

#include "uform/encoder/multimodal_batch.hpp" // `video_batch_shape_t`
#include "uform/utilities/types.hpp"          // the scalar aliases

namespace unum::uform::media {

namespace ue = unum::uform::encoder;
namespace uu = unum::uform::utilities;
namespace nk = ashvardanian::numkong;
namespace st = ashvardanian::smashtable;

using uu::f64_t;
using uu::u16_t;
using uu::u32_t;
using uu::u64_t;
using ue::video_batch_shape_t;

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

/** The largest patch grid within one @p columns row that keeps @p native 's aspect, so every image
 *  fills its row whatever its resolution: a small image is upscaled and a large one shrunk to the
 *  same patch count. */
[[nodiscard]] patch_grid_t image_patch_grid(u32_t columns, image_extent_t native) noexcept;

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
 *      @c (frames,width,height) — the image grid with a temporal axis folded ahead of the patch.
 *
 *  The rows and columns tile the native frame at the shape's @c patch_side ; the temporal groups
 *  pack the @c frame_stride -sampled frames into @c tube_frames -deep tubes. A footprint under
 *  @p columns grows its frame until the clip fills the row; one over it is scaled toward its own
 *  shape by a cube root, then trimmed on its largest axis until it fits one row.
 *
 *  @param[in] frames The clip's native frame count, 0 when unknown.
 *  @param[in] native The clip's native frame extent.
 *  @param[in] shape The video modality's geometry: patch side, tube depth and frame stride.
 *  @param[in] columns The batch row width in patches.
 */
[[nodiscard]] video_grid_t video_tube_grid(u32_t frames, image_extent_t native, video_batch_shape_t const &shape,
                                           u32_t columns) noexcept;

} // namespace unum::uform::media
