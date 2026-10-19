/**
 *  @brief Host image decoding — see the header for the contract.
 *  @file src/media/image.cpp
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include <cstring> // `std::memcpy` — the unaligned load behind `load_big_endian_`

#include <bit> // `std::byteswap`, `std::endian`

#include <png.h>         // vendor-boundary: PNG decode
#include <turbojpeg.h>   // vendor-boundary: JPEG decode with DCT scaling, reporting corrupt streams
#include <webp/decode.h> // vendor-boundary: WebP decode

#include "core/array.hpp" // `array` — the decode scratch
#include "media/image.hpp"

namespace um = unum::uform::media;
using namespace um;
using uc::array;

/** Load a big-endian unsigned integer from @p bytes — one unaligned load, then a byte swap on a little-endian
 *  host. */
template <typename integer_type_>
[[nodiscard]] static integer_type_ load_big_endian_(b8_t const *bytes) noexcept {
    integer_type_ value;
    std::memcpy(&value, bytes, sizeof(value));
    if constexpr (std::endian::native == std::endian::little) value = std::byteswap(value);
    return value;
}

/** The @c unsigned @c char pointer TurboJPEG, libpng and libwebp read compressed @p bytes through. */
[[nodiscard]] static unsigned char const *codec_bytes_(b8_t const *bytes) noexcept {
    return reinterpret_cast<unsigned char const *>(bytes);
}

/** Sniff a blob's compressed format from its leading bytes — JPEG @c FFD8FF , PNG @c 89504E47 ,
 *  WebP @c "RIFF"…"WEBP" ; anything else is unknown. */
[[nodiscard]] static image_format_t sniff_format_(std::span<b8_t const> encoded) noexcept {
    auto const byte = [encoded](std::size_t index) noexcept { return std::to_integer<u8_t>(encoded[index]); };
    std::size_t const size = encoded.size();
    if (size >= 3 && byte(0) == 0xFF && byte(1) == 0xD8 && byte(2) == 0xFF) return format_jpeg_k;
    if (size >= 8 && byte(0) == 0x89 && byte(1) == 0x50 && byte(2) == 0x4E && byte(3) == 0x47) return format_png_k;
    if (size >= 12 && byte(0) == 'R' && byte(1) == 'I' && byte(2) == 'F' && byte(3) == 'F' && byte(8) == 'W' &&
        byte(9) == 'E' && byte(10) == 'B' && byte(11) == 'P')
        return format_webp_k;
    return format_unknown_k;
}

/** A JPEG's marker walk over @p window , the blob's bytes from @p window_offset on, for a
 *  start-of-frame (SOFn): after the 2-byte segment length come precision, then height and width as
 *  big-endian UInt16. Only the frame headers carry them ( @c C0..CF except DHT @c C4 , JPG @c C8 ,
 *  DAC @c CC ). The walk starts past the SOI marker at the blob's head and at the window's first
 *  marker otherwise, and a walk leaving the window returns the offset it resumes at. */
[[nodiscard]] static expected<image_probe_t> jpeg_probe_(std::span<b8_t const> window,
                                                         std::size_t window_offset) noexcept {
    b8_t const *const bytes = window.data();
    std::size_t const size = window.size();
    std::size_t offset = window_offset == 0 ? 2 : 0;
    while (offset + 9 < size) {
        if (std::to_integer<u8_t>(bytes[offset]) != 0xFF) {
            ++offset;
            continue;
        }
        u8_t const marker = std::to_integer<u8_t>(bytes[offset + 1]);
        bool const is_frame = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (is_frame) {
            u32_t const height = load_big_endian_<u16_t>(bytes + offset + 5);
            u32_t const width = load_big_endian_<u16_t>(bytes + offset + 7);
            if (width == 0 || height == 0)
                return expected<image_probe_t>::fail(status_t::invalid_input_k, "JPEG start-of-frame has no extent");
            return image_probe_t {format_jpeg_k, image_extent_t {width, height}, 0};
        }
        offset += 2 + load_big_endian_<u16_t>(bytes + offset + 2); // skip this marker segment
    }
    return image_probe_t {format_jpeg_k, image_extent_t {}, window_offset + offset};
}

/** A PNG's pixel extent from its IHDR — width and height are big-endian UInt32 at byte offsets 16 and 20 of the
 *  signature+IHDR prologue. */
[[nodiscard]] static expected<image_extent_t> png_dimensions_(std::span<b8_t const> encoded) noexcept {
    b8_t const *const bytes = encoded.data();
    if (encoded.size() < 24)
        return expected<image_extent_t>::fail(status_t::invalid_input_k, "PNG header prefix too short");
    u32_t const width = load_big_endian_<u32_t>(bytes + 16), height = load_big_endian_<u32_t>(bytes + 20);
    if (width == 0 || height == 0)
        return expected<image_extent_t>::fail(status_t::invalid_input_k, "PNG IHDR has zero extent");
    return image_extent_t {width, height};
}

/** A WebP's pixel extent — the container's canvas size via libwebp's header parser. */
[[nodiscard]] static expected<image_extent_t> webp_dimensions_(std::span<b8_t const> encoded) noexcept {
    int width = 0, height = 0;
    if (WebPGetInfo(codec_bytes_(encoded.data()), encoded.size(), &width, &height) == 0)
        return expected<image_extent_t>::fail(status_t::invalid_input_k, "malformed WebP header");
    return image_extent_t {static_cast<u32_t>(width), static_cast<u32_t>(height)};
}

expected<image_probe_t> um::probe_image(std::span<b8_t const> encoded, std::size_t window_offset) noexcept {
    if (window_offset != 0) return jpeg_probe_(encoded, window_offset);
    image_format_t const format = sniff_format_(encoded);
    if (format == format_unknown_k)
        return expected<image_probe_t>::fail(status_t::invalid_input_k, "unrecognized image format");
    if (format == format_jpeg_k) return jpeg_probe_(encoded, 0);
    expected<image_extent_t> extent = format == format_png_k ? png_dimensions_(encoded) : webp_dimensions_(encoded);
    if (!extent) return expected<image_probe_t>::fail(std::move(extent));
    return image_probe_t {format, extent.value_, 0};
}

expected<image_plan_t> um::plan_image(std::span<b8_t const> encoded, u32_t patch_side, u32_t columns,
                                      u32_t segments) noexcept {
    expected<image_probe_t> probed = probe_image(encoded);
    if (!probed) return expected<image_plan_t>::fail(std::move(probed));
    if (probed->extent.width == 0)
        return expected<image_plan_t>::fail(status_t::invalid_input_k, "no JPEG start-of-frame in the blob");
    image_plan_t plan;
    plan.format = probed->format;
    plan.native = probed->extent;
    plan.grid = image_patch_grid(image_patch_budget(plan.native, patch_side, columns, segments), plan.native);
    image_extent_t const target {u32_t(plan.grid.columns) * patch_side, u32_t(plan.grid.rows) * patch_side};
    plan.scale = plan.format == format_jpeg_k ? image_decode_scale(plan.native, target) : 0u;
    plan.decoded = image_decoded_extent(plan.native, plan.scale);
    return plan;
}

/** Decode a JPEG through @p decoder into @p out with the @c 1/2ᵏ DCT scale landing exactly on its extent. TurboJPEG
 *  reports a corrupt or truncated stream where libjpeg's default error handler exits the process; @c TJPF_RGBA
 *  emits a constant 255 alpha, so a JPEG matches the RGBA grid. */
[[nodiscard]] static expected_t decode_jpeg_(tjhandle decoder, unsigned char const *bytes, std::size_t size,
                                             pixel_image_span_t out) noexcept {
    if (tj3DecompressHeader(decoder, bytes, size) != 0)
        return expected_t::fail(status_t::invalid_input_k, "malformed JPEG header: {}", tj3GetErrorStr(decoder));
    int const width = static_cast<int>(out.extent(1)), height = static_cast<int>(out.extent(0));
    int const native_width = tj3Get(decoder, TJPARAM_JPEGWIDTH), native_height = tj3Get(decoder, TJPARAM_JPEGHEIGHT);
    tjscalingfactor scale {1, 1};
    for (int denominator = 1; denominator <= 8; denominator *= 2) {
        scale = {1, denominator};
        if (TJSCALED(native_width, scale) == width && TJSCALED(native_height, scale) == height) break;
    }
    if (TJSCALED(native_width, scale) != width || TJSCALED(native_height, scale) != height)
        return expected_t::fail(status_t::invalid_input_k, "no DCT scale takes a {}x{} JPEG to {}x{}", native_width,
                                native_height, width, height);
    if (tj3SetScalingFactor(decoder, scale) != 0 ||
        tj3Decompress8(decoder, bytes, size, out.data(), width * static_cast<int>(image_channels_k), TJPF_RGBA) != 0)
        return expected_t::fail(status_t::invalid_input_k, "corrupt JPEG: {}", tj3GetErrorStr(decoder));
    return {};
}

expected_t um::decode_image_rgba(std::span<b8_t const> encoded, image_format_t format,
                                 pixel_image_span_t out) noexcept {
    unsigned char const *const bytes = codec_bytes_(encoded.data());
    std::size_t const size = encoded.size();
    u32_t const height = static_cast<u32_t>(out.extent(0)), width = static_cast<u32_t>(out.extent(1));
    pixel_t *const pixels = out.data();
    std::size_t const stride = static_cast<std::size_t>(width) * image_channels_k;
    if (out.extent(2) != image_channels_k)
        return expected_t::fail(status_t::shape_unsupported_k, "RGBA decode needs 4 channels, got {}", out.extent(2));
    if (format == format_webp_k)
        return WebPDecodeRGBAInto(bytes, size, pixels, stride * height, static_cast<int>(stride)) != nullptr
                   ? expected_t {}
                   : expected_t::fail(status_t::invalid_input_k, "corrupt WebP bitstream or extent mismatch");
    if (format == format_png_k) {
        png_image image {};
        image.version = PNG_IMAGE_VERSION;
        if (png_image_begin_read_from_memory(&image, bytes, size) == 0)
            return expected_t::fail(status_t::invalid_input_k, "malformed PNG header");
        bool const fits = image.width == width && image.height == height;
        image.format = PNG_FORMAT_RGBA;
        bool const ok = fits &&
                        png_image_finish_read(&image, nullptr, pixels, static_cast<png_int_32>(stride), nullptr) != 0;
        png_image_free(&image);
        return ok ? expected_t {} : expected_t::fail(status_t::invalid_input_k, "corrupt PNG or extent mismatch");
    }
    if (format != format_jpeg_k) return expected_t::fail(status_t::invalid_input_k, "unrecognized image format");
    tjhandle const decoder = tj3Init(TJINIT_DECOMPRESS);
    if (!decoder) return expected_t::fail(status_t::bad_alloc_k, "JPEG decoder allocation failed");
    expected_t decoded = decode_jpeg_(decoder, bytes, size, out);
    tj3Destroy(decoder);
    return decoded;
}

expected_t um::decode_image(std::span<b8_t const> encoded, image_plan_t const &plan,
                            pixel_patches_span_t out) noexcept {
    if (out.extent(0) != plan.grid.rows || out.extent(1) != plan.grid.columns)
        return expected_t::fail(status_t::shape_unsupported_k, "cells [{},{}] do not match the planned grid [{},{}]",
                                out.extent(0), out.extent(1), plan.grid.rows, plan.grid.columns);
    std::size_t const bytes = static_cast<std::size_t>(plan.decoded.width) * plan.decoded.height * image_channels_k;
    array<pixel_t> decoded;
    if (decoded.resize_uninitialized(bytes) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "decoded pixel allocation failed ({} bytes)", bytes);
    pixel_image_span_t const pixels {decoded.data(), {plan.decoded.height, plan.decoded.width, image_channels_k}};
    if (expected_t decoded_rgba = decode_image_rgba(encoded, plan.format, pixels); !decoded_rgba) return decoded_rgba;
    return resize_image(
        pixel_image_view_t {decoded.data(), {plan.decoded.height, plan.decoded.width, image_channels_k}}, out);
}
