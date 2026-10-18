/**
 *  @brief Host image decoding: triage a compressed blob, decode it to RGBA, and Lanczos-resize its center crop into
 *      the batch's patch layout.
 *  @file src/media/image.hpp
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  JPEG decodes through libjpeg-turbo's TurboJPEG, PNG through libpng and WebP through libwebp, all to interleaved RGBA, JPEG
 *  with a constant @c 255 alpha. The RAM and AMD executors run the whole @ref decode_image ; the NVIDIA loader takes
 *  only @ref probe_image and @ref decode_image_rgba for the blobs nvJPEG cannot, and resizes on the device with NPP.
 *
 *  Both resizes are Lanczos-3 over the same @ref image_center_crop to the same extent. The host filter keeps the
 *  6-tap support at every scale, as NPP's @c NPPI_INTER_LANCZOS does, and replicates the crop's edge pixels; the two
 *  agree to rounding where NPP's border and fixed-point details match, which its documentation does not pin.
 */
#pragma once

#include <span> // `std::span`

#include "core/types.hpp"     // `expected`, `status_t`, the scalar aliases
#include "media/planning.hpp" // `image_extent_t`, `patch_grid_t`, `crop_t`

namespace unum::uform::media {

using uc::b8_t;
using uc::expected;
using uc::expected_t;
using uc::pixel_t;
using uc::status_t;
using uc::u8_t;

/** The channels every decoded pixel carries — interleaved RGBA. */
inline constexpr u32_t image_channels_k = 4u;

/** A writable @b [height,width,4] RGBA picture, the destination of @ref decode_image_rgba . */
using pixel_image_span_t = nk::tensor_span<pixel_t, 3>;

/** A read-only @b [height,width,4] RGBA picture, the source of @ref resize_image . */
using pixel_image_view_t = nk::tensor_view<pixel_t, 3>;

/** A patch-arranged pixel grid @b [rows,columns,patch_side,patch_side,channels] . For one image it is the
 *  @c rows·columns cells of its segment, which sit contiguously in a batch row in exactly this order. */
using pixel_patches_span_t = nk::tensor_span<pixel_t, 5>;

/** The compressed formats a blob can be, decided by its magic bytes. */
enum image_format_t : u8_t { format_jpeg_k = 0, format_png_k = 1, format_webp_k = 2, format_unknown_k = 3 };

/** One compressed blob's triage — its format and native pixel extent. */
struct image_probe_t {

    /** The compressed format decided by the blob's magic bytes. */
    image_format_t format = format_unknown_k;

    /** The native pixel extent parsed from the blob's header. */
    image_extent_t extent {};
};

/** Everything the host path decides about one image before decoding it. */
struct image_plan_t {

    /** The compressed format decided by the blob's magic bytes. */
    image_format_t format = format_unknown_k;

    /** The image's full encoded resolution. */
    image_extent_t native {};

    /** The extent the decode lands on: @c native , or shrunk by @ref scale for a JPEG. */
    image_extent_t decoded {};

    /** The patch factorization the resize targets. */
    patch_grid_t grid {};

    /** The JPEG DCT scale @c k of a @c 1/2ᵏ decode, 0 for PNG and WebP. */
    u32_t scale = 0;
};

/**
 *  @brief Sniff a blob's format from its magic bytes and parse its native extent from its header.
 *
 *  A prefix suffices: the JPEG start-of-frame, the PNG IHDR or the WebP VP8 header, all within the first few KB
 *  unless metadata pushes a JPEG frame header further.
 *
 *  @retval status_t::invalid_input_k Unrecognized magic, a truncated header or a zero extent.
 */
[[nodiscard]] expected<image_probe_t> probe_image(std::span<b8_t const> encoded) noexcept;

/**
 *  @brief Plan one image the way the NVIDIA loader plans it: probe, budget, factor, and the DCT scale toward the
 *      target a hardware JPEG decode would take.
 *
 *  @param[in] encoded The compressed blob, or at least its header prefix.
 *  @param[in] patch_side The square patch side in pixels.
 *  @param[in] columns The batch row width in patches.
 *  @param[in] segments The most images one batch carries.
 */
[[nodiscard]] expected<image_plan_t> plan_image(std::span<b8_t const> encoded, u32_t patch_side, u32_t columns,
                                                u32_t segments) noexcept;

/**
 *  @brief Decode one blob to interleaved RGBA at @p out 's extent.
 *
 *  PNG and WebP must decode at their native extent. A JPEG lands on @p out exactly through the @c 1/2ᵏ DCT scale
 *  whose output matches it, so a plan's @c decoded extent fits.
 *
 *  @param[in] encoded The whole compressed blob.
 *  @param[in] format The blob's format, from @ref probe_image .
 *  @param[out] out The @b [height,width,4] destination.
 *  @retval status_t::invalid_input_k A corrupt bitstream or an extent no scale reaches.
 */
[[nodiscard]] expected_t decode_image_rgba(std::span<b8_t const> encoded, image_format_t format,
                                           pixel_image_span_t out) noexcept;

/**
 *  @brief Crop @p source to the aspect of @p out 's patch grid, Lanczos-3 resize it to @c (columns·P,rows·P) , and
 *      write the result patch by patch, as the NVIDIA loader's NPP resize and tile kernel do.
 *
 *  @param[in] source The decoded @b [height,width,4] picture.
 *  @param[out] out The image's @b [rows,columns,P,P,4] cells.
 *  @retval status_t::shape_unsupported_k Channel counts other than 4, or an empty grid or source.
 *  @retval status_t::bad_alloc_k The intermediate row buffer could not be allocated.
 */
[[nodiscard]] expected_t resize_image(pixel_image_view_t source, pixel_patches_span_t out) noexcept;

/**
 *  @brief Decode, crop, resize and tile one image into its cells — the whole host path.
 *
 *  @param[in] encoded The whole compressed blob.
 *  @param[in] plan The plan from @ref plan_image .
 *  @param[out] out The image's @b [plan.grid.rows,plan.grid.columns,P,P,4] cells.
 */
[[nodiscard]] expected_t decode_image(std::span<b8_t const> encoded, image_plan_t const &plan,
                                      pixel_patches_span_t out) noexcept;

} // namespace unum::uform::media
