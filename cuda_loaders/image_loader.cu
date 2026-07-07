/**
 *  @brief The Parquet image loader's bodies and kernels — see the header for the contract. Its shape
 *      mirrors @c tokens_loader.cu : a keyed file pick, one pooled reservoir, a keyed row draw, then a
 *      per-batch decode. The decode is the shared @ref decode_images : JPEG rows through the @ref
 *      nvjpeg_session_t straight off the reservoir tape, PNG/WebP rows on the CPU pool, then one NPP batch
 *      crops each to its factor-pair aspect and resizes it into the batch's shared patch grid.
 *  @file cuda_loaders/image_loader.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 */
#include <cstdio>  // `FILE` — jpeglib.h's `jpeg_stdio_*` declarations need it in scope
#include <cstring> // `std::memcpy` — the unaligned load behind `load_big_endian_`

#include <bit>       // `std::byteswap`, `std::endian`
#include <algorithm> // `std::upper_bound`, `std::find`, `std::min`, `std::max`

#include <nvjpeg.h>                   // vendor-boundary: device-native batched JPEG decode
#include <nppi_geometry_transforms.h> // vendor-boundary: batched crop+resize
#include <jpeglib.h>                  // vendor-boundary: CPU JPEG fallback (progressive / rejected)
#include <png.h>                      // vendor-boundary: CPU PNG decode
#include <webp/decode.h>              // vendor-boundary: CPU WebP decode

#include <fork_union.hpp> // vendor-boundary: the host-CPU decode worker pool

#include "cuda_loaders/image_loader.cuh"

namespace uc = unum::uform::core;
namespace ul = unum::uform::loaders;
using namespace uc;
using namespace ul;

namespace fu = ashvardanian::fork_union;

/** @brief Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *      decorrelated under one @c (seed,step,rank) — the same salts the tokens loader uses. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/** @brief Header bytes gathered per drawn blob — enough for the magic sniff and the dimension parse (JPEG
 *      start-of-frame, PNG IHDR, WebP VP8 header) without lifting the whole compressed blob. One device
 *      gather kernel packs these prefixes contiguously so the whole triage reads back in a single D2H. */
inline constexpr std::size_t header_probe_bytes_k = 1u << 12;
/** @brief The channels every decoded pixel carries — interleaved RGBA. JPEG has no alpha (the hardware
 *      nvJPEG path decodes RGB, then @ref rgb_to_rgba_kernel_ pads a constant @c 255 ; libjpeg-turbo's CPU
 *      fallback emits @c JCS_EXT_RGBA directly), while PNG / WebP carry a real alpha through @ref cpu_decode_ . */
inline constexpr u32_t image_channels_k = 4u;

/** @brief The compressed formats a drawn blob can be, decided by its magic bytes. */
enum image_format_t : u8_t { format_jpeg_k = 0, format_png_k = 1, format_webp_k = 2, format_unknown_k = 3 };

/** @brief The @c (rows,cols) factorization of the patch budget one image tiles into — the per-image column
 *      sub-budget the segment cursor lays down and the resize/tile scatter read. */
struct patch_grid_t {
    /** @brief The patch rows the image tiles into. */
    u16_t rows;
    /** @brief The patch columns the image tiles into. */
    u16_t cols;
};

/**
 *  @brief Write the batch's segment geometry under the token-cursor model: each drawn image is @b one segment
 *      occupying its own column sub-budget ( @c patch_rows·patch_cols , taken from its host-computed
 *      @p grids factorization), and several images pack into one row. Thread 0 walks the segments serially — a
 *      strict scan — jumping the cursor past a row boundary a segment would cross (never crossing a row),
 *      dropping loudly (zero length) any segment that would overflow the @c rows·columns grid, and tagging
 *      each placed live segment @c image_k (a dropped one PADs to @c text_k , length 0). It also lands each live
 *      segment's @c (patch_rows,patch_cols) into
 *      @p image_shapes on the device — the descriptors are arena (device) memory, so the loader never writes
 *      them from the host. Segments past @p live_images PAD: zero length, repeated offset, @c text_k .
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void image_segments_kernel_(patch_grid_t const *grids, u32_t live_images, u32_t columns_per_row,
                                              u32_t capacity, image_shape_span_t image_shapes,
                                              u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                              modality_span_t segment_modality) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    u32_t const total_segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const live = live_images < total_segments ? live_images : total_segments;
    u32_t cursor = 0;
    for (u32_t segment = 0; segment != total_segments; ++segment) {
        bool const is_live_image = segment < live;
        u32_t columns = 0;
        if (is_live_image) {
            image_shapes[segment, 0] = grids[segment].rows;
            image_shapes[segment, 1] = grids[segment].cols;
            columns = static_cast<u32_t>(grids[segment].rows) * static_cast<u32_t>(grids[segment].cols);
        }
        // A segment never crosses its row: jump the cursor to the next row when it would.
        u32_t const room_in_row = columns_per_row - cursor % columns_per_row;
        if (columns > room_in_row) cursor += room_in_row;
        if (cursor + columns > capacity) {
            printf("uform: loaded image batch overflows at segment %u — segment dropped\n", segment);
            columns = 0;
        }
        segment_offsets[segment] = cursor;
        segment_lengths[segment] = columns;
        // A live image that overflowed and dropped (length 0) reads as PAD, not a zero-length image_k segment.
        segment_modality[segment] = is_live_image && columns != 0u ? modality_t::image_k : modality_t::text_k;
        cursor += columns;
    }
    segment_offsets[total_segments] = cursor;
}

/** @brief One image's header-gather descriptor: where its compressed bytes start in the device tape and how
 *      many of them (capped at the probe size) to lift. */
struct gather_entry_t {
    /** @brief Where the image's compressed bytes start in the device tape. */
    u8_t const *source;
    /** @brief How many of them (capped at the probe size) to lift. */
    u32_t length;
};

/**
 *  @brief Gather each drawn blob's header prefix from the scattered device tape into a contiguous staging
 *      buffer — one block per image, threads stripe the copy — so triage reads them back in one D2H instead
 *      of one small transfer per image.
 *
 *  __grid_size__ @c [min(images,cap),1,1] , grid-stride over images; no dynamic shared memory.
 */
static __global__ void gather_headers_kernel_(gather_entry_t const *entries, u8_t *staging, u32_t stride,
                                              u32_t images) {
    for (u32_t image = blockIdx.x; image < images; image += gridDim.x) {
        gather_entry_t const entry = entries[image];
        u8_t *const destination = staging + static_cast<std::size_t>(image) * stride;
        for (u32_t byte = threadIdx.x; byte < entry.length; byte += blockDim.x) destination[byte] = entry.source[byte];
    }
}

/** @brief A decoded image's pixel dimensions — a size travels as one value, not two loose scalars. */
struct image_extent_t {
    /** @brief The image's pixel width. */
    u32_t width = 0;
    /** @brief The image's pixel height. */
    u32_t height = 0;
};

/** @brief One compressed blob's triage — its format and native pixel extent. */
struct image_probe_t {
    /** @brief The compressed format decided by the blob's magic bytes. */
    image_format_t format = format_unknown_k;
    /** @brief The native pixel extent parsed from the blob's header. */
    image_extent_t extent {};
};

/** @brief A writable @b [height,width,channels] view of one decoded image — the CPU decode's destination,
 *      sliced out of a per-fill pixel heap. */
using pixel_image_span_t = nk::tensor_span<pixel_t, 3>;

/** @brief Load a big-endian unsigned integer from @p bytes — one unaligned load, then @ref std::byteswap on a
 *      little-endian host. Host-only (the parsers run on the CPU after the header gather); a device sniff
 *      would need a byte-shift twin. */
template <typename integer_type_>
[[nodiscard]] static integer_type_ load_big_endian_(u8_t const *bytes) noexcept {
    integer_type_ value;
    std::memcpy(&value, bytes, sizeof(value));
    if constexpr (std::endian::native == std::endian::little) value = std::byteswap(value);
    return value;
}

/** @brief Sniff a blob's compressed format from its leading bytes — JPEG @c FFD8FF , PNG @c 89504E47 ,
 *      WebP @c "RIFF"…"WEBP" ; anything else is unknown and fails the fill loudly. */
[[nodiscard]] static image_format_t sniff_format_(sz::string_view encoded) noexcept {
    u8_t const *const bytes = reinterpret_cast<u8_t const *>(encoded.data());
    std::size_t const size = encoded.size();
    if (size >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF) return format_jpeg_k;
    if (size >= 8 && bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47) return format_png_k;
    if (size >= 12 && bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F' && bytes[8] == 'W' &&
        bytes[9] == 'E' && bytes[10] == 'B' && bytes[11] == 'P')
        return format_webp_k;
    return format_unknown_k;
}

/** @brief A JPEG's pixel extent, scanned from its markers for a start-of-frame (SOFn): after the 2-byte
 *      segment length come precision, then height and width as big-endian UInt16. Only the frame headers
 *      (@c C0..CF except DHT @c C4 , JPG @c C8 , DAC @c CC ) carry them. */
[[nodiscard]] static expected<image_extent_t> jpeg_dimensions_(sz::string_view encoded) noexcept {
    u8_t const *const bytes = reinterpret_cast<u8_t const *>(encoded.data());
    std::size_t const size = encoded.size();
    for (std::size_t offset = 2; offset + 9 < size;) { // start past the SOI marker (FFD8)
        if (bytes[offset] != 0xFF) {
            ++offset;
            continue;
        }
        u8_t const marker = bytes[offset + 1];
        bool const is_frame = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (is_frame) {
            u32_t const height = load_big_endian_<u16_t>(bytes + offset + 5);
            u32_t const width = load_big_endian_<u16_t>(bytes + offset + 7);
            if (width == 0 || height == 0) break;
            return image_extent_t {width, height};
        }
        offset += 2 + load_big_endian_<u16_t>(bytes + offset + 2); // skip this marker segment
    }
    return expected<image_extent_t>::fail(status_t::invalid_input_k, "no JPEG start-of-frame in the header prefix");
}

/** @brief A PNG's pixel extent from its IHDR — width and height are big-endian UInt32 at byte offsets 16 and
 *      20 of the signature+IHDR prologue. */
[[nodiscard]] static expected<image_extent_t> png_dimensions_(sz::string_view encoded) noexcept {
    u8_t const *const bytes = reinterpret_cast<u8_t const *>(encoded.data());
    if (encoded.size() < 24)
        return expected<image_extent_t>::fail(status_t::invalid_input_k, "PNG header prefix too short");
    u32_t const width = load_big_endian_<u32_t>(bytes + 16), height = load_big_endian_<u32_t>(bytes + 20);
    if (width == 0 || height == 0)
        return expected<image_extent_t>::fail(status_t::invalid_input_k, "PNG IHDR has zero extent");
    return image_extent_t {width, height};
}

/** @brief A WebP's pixel extent — the container's canvas size via libwebp's header parser (a device-side
 *      parse is a follow-up once the whole sniff moves onto the GPU). */
[[nodiscard]] static expected<image_extent_t> webp_dimensions_(sz::string_view encoded) noexcept {
    int width = 0, height = 0;
    if (WebPGetInfo(reinterpret_cast<u8_t const *>(encoded.data()), encoded.size(), &width, &height) == 0)
        return expected<image_extent_t>::fail(status_t::invalid_input_k, "malformed WebP header");
    return image_extent_t {static_cast<u32_t>(width), static_cast<u32_t>(height)};
}

/** @brief Triage one compressed blob — its format (magic bytes) and native extent (format-specific header
 *      parse). The one call the loader's triage and, later, the GPU sniff kernel share. */
[[nodiscard]] static expected<image_probe_t> probe_image_(sz::string_view encoded) noexcept {
    image_format_t const format = sniff_format_(encoded);
    if (format == format_unknown_k)
        return expected<image_probe_t>::fail(status_t::invalid_input_k, "unrecognized image format");
    expected<image_extent_t> extent = format == format_jpeg_k  ? jpeg_dimensions_(encoded)
                                      : format == format_png_k ? png_dimensions_(encoded)
                                                               : webp_dimensions_(encoded);
    if (!extent) return expected<image_probe_t>::fail(std::move(extent));
    return image_probe_t {format, extent.value_};
}

/** @brief The factor pair of @p num_patches whose @c cols/rows aspect is nearest @p extent 's, clamped to the
 *      @c [1×num_patches,num_patches×1] the budget allows — every image is exactly @p num_patches patches, so
 *      its rectangle rides the coordinates without a crop toward a square. */
[[nodiscard]] static patch_grid_t factor_pair_(image_extent_t extent, u32_t num_patches) noexcept {
    f64_t const target = static_cast<f64_t>(extent.width) / static_cast<f64_t>(std::max<u32_t>(extent.height, 1u));
    u32_t best_rows = 1, best_cols = num_patches;
    f64_t best_error = 1.0e300;
    for (u32_t rows = 1; rows <= num_patches; ++rows) {
        if (num_patches % rows != 0) continue;
        u32_t const cols = num_patches / rows;
        f64_t const aspect = static_cast<f64_t>(cols) / static_cast<f64_t>(rows);
        f64_t const error = aspect > target ? aspect / target : target / aspect;
        if (error < best_error) best_error = error, best_rows = rows, best_cols = cols;
    }
    return patch_grid_t {static_cast<u16_t>(best_rows), static_cast<u16_t>(best_cols)};
}

/** @brief One image's column budget — how many grid cells (patches) it earns from its native resolution, so a
 *      bigger image packs more patches and several small images share a row. The native patch count (the
 *      @c patch_size²-tiles that cover the source, ceiled per axis) is rounded @b up to a multiple of the
 *      floor budget for a divisor-friendly @ref factor_pair_ , then clamped to
 *      @c [max_columns/max_segments,max_columns] . Monotonic in resolution and bounded so at most
 *      @c max_segments images fill one row; the whole policy lives here and nowhere else. */
[[nodiscard]] static u32_t column_budget_(image_extent_t native, u32_t patch_size, u32_t max_columns,
                                          u32_t max_segments) noexcept {
    u32_t const floor_budget = std::max<u32_t>(1u, max_columns / std::max<u32_t>(max_segments, 1u));
    u32_t const patch_rows = (native.height + patch_size - 1u) / patch_size;
    u32_t const patch_cols = (native.width + patch_size - 1u) / patch_size;
    u32_t const native_patches = std::max<u32_t>(1u, patch_rows * patch_cols);
    u32_t const rounded = ((native_patches + floor_budget - 1u) / floor_budget) * floor_budget;
    return std::min<u32_t>(std::max<u32_t>(rounded, floor_budget), max_columns);
}

/** @brief The largest 1/2ᵏ (k≤3) nvJPEG scale whose decoded extent still covers @p target — decode as small
 *      as possible without dropping below the resize target, so the crop-resize only downsamples. */
[[nodiscard]] static nvjpegScaleFactor_t pick_scale_(image_extent_t native, image_extent_t target) noexcept {
    u32_t level = 0;
    while (level < 3 && (native.width >> (level + 1)) >= target.width &&
           (native.height >> (level + 1)) >= target.height)
        ++level;
    return static_cast<nvjpegScaleFactor_t>(level);
}

/** @brief nvJPEG's decoded extent for a 1/2ᵏ scale — a ceiling division per axis matching the library's
 *      rounding. */
[[nodiscard]] static image_extent_t scaled_extent_(image_extent_t extent, nvjpegScaleFactor_t scale) noexcept {
    u32_t const factor = 1u << static_cast<u32_t>(scale);
    return image_extent_t {(extent.width + factor - 1u) / factor, (extent.height + factor - 1u) / factor};
}

/** @brief The centered source crop of @p source matching @p target 's aspect — the largest rectangle of that
 *      aspect that fits, so the resize that follows never stretches. */
[[nodiscard]] static NppiRect center_crop_(image_extent_t source, image_extent_t target) noexcept {
    NppiRect rect {};
    // Compare source aspect to target aspect via cross-multiplication (no division, no rounding drift).
    if (static_cast<u64_t>(source.width) * target.height > static_cast<u64_t>(target.width) * source.height) {
        rect.height = static_cast<int>(source.height);
        rect.width = static_cast<int>(static_cast<u64_t>(source.height) * target.width / target.height);
    }
    else {
        rect.width = static_cast<int>(source.width);
        rect.height = static_cast<int>(static_cast<u64_t>(source.width) * target.height / target.width);
    }
    rect.x = (static_cast<int>(source.width) - rect.width) / 2;
    rect.y = (static_cast<int>(source.height) - rect.height) / 2;
    return rect;
}

/** @brief Decode one CPU-routed blob (PNG / WebP / progressive-or-rejected JPEG) into @p destination as
 *      interleaved RGBA at its @b [height,width,channels(4)] extent — PNG / WebP carry a real alpha, a JPEG
 *      (no alpha) fills a constant @c 255 via libjpeg-turbo's @c JCS_EXT_RGBA . When a rejected JPEG's
 *      destination is smaller than its native size (a hardware plan that scaled it), libjpeg's 1/2ᵏ DCT scaling
 *      lands the decode exactly on @p destination , so it still fits its @c decoded -sized scratch region. Runs
 *      in a @c fork_union worker, so it hands back a bare @ref status_t ( @c invalid_input_k on a corrupt
 *      bitstream) — no diagnostic to carry there. */
[[nodiscard]] static status_t cpu_decode_(image_format_t format, sz::string_view encoded,
                                          pixel_image_span_t destination) noexcept {
    u8_t const *const bytes = reinterpret_cast<u8_t const *>(encoded.data());
    std::size_t const size = encoded.size();
    u32_t const height = static_cast<u32_t>(destination.extent(0)), width = static_cast<u32_t>(destination.extent(1));
    u8_t *const pixels = reinterpret_cast<u8_t *>(destination.data());
    std::size_t const stride = static_cast<std::size_t>(width) * image_channels_k;
    if (format == format_webp_k)
        return WebPDecodeRGBAInto(bytes, size, pixels, stride * height, static_cast<int>(stride)) != nullptr
                   ? status_t::success_k
                   : status_t::invalid_input_k;
    if (format == format_png_k) {
        png_image image {};
        image.version = PNG_IMAGE_VERSION;
        if (png_image_begin_read_from_memory(&image, bytes, size) == 0) return status_t::invalid_input_k;
        image.format = PNG_FORMAT_RGBA;
        bool const ok = png_image_finish_read(&image, nullptr, pixels, static_cast<png_int_32>(stride), nullptr) != 0;
        png_image_free(&image);
        return ok ? status_t::success_k : status_t::invalid_input_k;
    }
    // JPEG the GPU batch could not take (progressive, 4:1:0 / 4:1:1 subsampling) — libjpeg-turbo on the CPU.
    // JCS_EXT_RGBA emits a 4-channel pixel with a constant 255 alpha, so a rejected JPEG matches the RGBA grid.
    jpeg_decompress_struct info {};
    jpeg_error_mgr error {};
    info.err = jpeg_std_error(&error);
    jpeg_create_decompress(&info);
    jpeg_mem_src(&info, bytes, size);
    if (jpeg_read_header(&info, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&info);
        return status_t::invalid_input_k;
    }
    info.out_color_space = JCS_EXT_RGBA;
    // The hardware plan may have scaled this JPEG down; pick the 1/2ᵏ DCT scaling that lands exactly on the
    // destination extent so a rejected-batch fallback still fits its @c decoded -sized scratch region.
    for (unsigned denom = 1; denom <= 8; denom *= 2) {
        info.scale_num = 1;
        info.scale_denom = denom;
        jpeg_calc_output_dimensions(&info);
        if (info.output_width == width && info.output_height == height) break;
    }
    jpeg_start_decompress(&info);
    bool ok = info.output_width == width && info.output_height == height &&
              info.output_components == static_cast<int>(image_channels_k);
    while (ok && info.output_scanline < info.output_height) {
        u8_t *scanline = pixels + static_cast<std::size_t>(info.output_scanline) * stride;
        ok = jpeg_read_scanlines(&info, &scanline, 1) == 1;
    }
    if (ok) jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    return ok ? status_t::success_k : status_t::invalid_input_k;
}

/**
 *  @brief Where one plan's pixels are decoded — the single knob that routes the JPEG-device vs CPU-fallback
 *      split. The invariant lives here and nowhere else: a plan is @c jpeg_device_k iff its format is JPEG @b and
 *      the device batch was accepted; every other plan (PNG / WebP from the start, or a JPEG whose batch nvJPEG
 *      rejected) is @c cpu_fallback_k . @ref triage_images_ sets the format-driven default, and a rejected batch
 *      flips its every JPEG entry to @c cpu_fallback_k in @ref decode_jpeg_plans_ — at the @c decoded scale
 *      already chosen, so the CPU decode fills the very same @c decoded -sized scratch region. @ref
 *      decode_cpu_plans_ then decodes exactly the @c cpu_fallback_k plans. A named route, never a bool.
 */
enum class route_t : u8_t {
    /** @brief Decoded device-native through nvJPEG's batched engine (an accepted JPEG batch). */
    jpeg_device_k = 0,
    /** @brief Decoded on the CPU pool — PNG / WebP, or a JPEG the device batch rejected. */
    cpu_fallback_k = 1,
};

/** @brief One drawn image's decode plan: its format and native size, the size it actually decodes to in the
 *      scratch (scaled for a hardware JPEG, native otherwise), that image's byte offset, the target patch
 *      factorization, the nvJPEG scale, and which road it decodes on. */
struct image_plan_t {
    /** @brief The compressed format decided by the blob's magic bytes. */
    image_format_t format = format_unknown_k;
    /** @brief The image's full encoded resolution, read from its header. */
    image_extent_t native {};
    /** @brief What actually lands in the scratch and sizes its RGBA region: @c native , or shrunk by the decode
     *      scale for a JPEG (nvJPEG's hardware scale, or libjpeg's scaled fallback) — the NPP resize's input. */
    image_extent_t decoded {};
    /** @brief This image's byte offset into the @c decoded -sized RGBA scratch. */
    std::size_t scratch_offset = 0;
    /** @brief The target patch factorization the resize/tile scatter honors. */
    patch_grid_t grid {};
    /** @brief The decode scale — NONE unless a JPEG shrinks toward its target (nvJPEG hardware or libjpeg
     *      fallback); a @c decoded -shrinking scale the CPU fallback honors so its output still fits the region. */
    nvjpegScaleFactor_t scale = NVJPEG_SCALE_NONE;
    /** @brief Which road the image decodes on — its format-driven default, flipped to @c cpu_fallback_k for a
     *      JPEG whose device batch was rejected (see @ref route_t for the invariant). */
    route_t route = route_t::cpu_fallback_k;
};

nvjpeg_session_t &nvjpeg_session_t::operator=(nvjpeg_session_t &&other) noexcept {
    if (this != &other) {
        this->~nvjpeg_session_t();
        handle_ = std::exchange(other.handle_, nullptr);
        jpeg_state_ = std::exchange(other.jpeg_state_, nullptr);
        backend_ = other.backend_;
    }
    return *this;
}

nvjpeg_session_t::~nvjpeg_session_t() noexcept {
    if (jpeg_state_ != nullptr) nvjpegJpegStateDestroy(jpeg_state_);
    if (handle_ != nullptr) nvjpegDestroy(handle_);
    handle_ = nullptr;
    jpeg_state_ = nullptr;
}

expected<nvjpeg_session_t> nvjpeg_session_t::make() noexcept {
    using result_t = expected<nvjpeg_session_t>;
    nvjpeg_session_t session;
    nvjpegHandle_t &handle = session.handle_;
    // The hardware backend takes device-resident bitstreams, honors the decode-time scale factor, and is the
    // fast path; fall back to the GPU-hybrid device backend (also device input, no scale) where it is absent.
    if (nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE_DEVICE, nullptr, nullptr, 0, &handle) == NVJPEG_STATUS_SUCCESS)
        session.backend_ = nvjpeg_backend_t::hardware_k;
    else if (nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID_DEVICE, nullptr, nullptr, 0, &handle) != NVJPEG_STATUS_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "nvJPEG device backend unavailable");
    if (nvjpegJpegStateCreate(handle, &session.jpeg_state_) != NVJPEG_STATUS_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "nvJPEG state creation failed");
    return result_t {std::move(session)};
}

expected_t nvjpeg_session_t::decode_jpeg_batch(std::span<jpeg_entry_t const> entries, gpu_stream_t &stream) noexcept {
    std::size_t const count = entries.size();
    if (count == 0) return {};
    // vendor-boundary: the driver stream cast to nvJPEG's runtime-API `cudaStream_t` handle.
    cudaStream_t const cuda_stream = reinterpret_cast<cudaStream_t>(stream.get());
    // The owning entry table fills nvJPEG's parallel arrays here, at the vendor boundary only.
    array<unsigned char const *> data;
    array<std::size_t> lengths;
    array<nvjpegImage_t> destinations;
    if (data.try_resize_uninitialized(count) != status_t::success_k ||
        lengths.try_resize_uninitialized(count) != status_t::success_k ||
        destinations.try_resize_uninitialized(count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "nvJPEG parallel-array allocation failed ({} images)", count);
    for (std::size_t index = 0; index != count; ++index) {
        data[index] = entries[index].bitstream;
        lengths[index] = entries[index].length;
        destinations[index] = entries[index].destination;
    }
    if (nvjpegDecodeBatchedInitialize(handle_, jpeg_state_, static_cast<int>(count), 1, NVJPEG_OUTPUT_RGBI) !=
        NVJPEG_STATUS_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "nvJPEG batch init failed ({} images)", count);
    // The hybrid backend ignores per-image params — a plain batched decode at full scale.
    if (backend_ != nvjpeg_backend_t::hardware_k)
        return nvjpegDecodeBatched(handle_, jpeg_state_, data.data(), lengths.data(), destinations.data(),
                                   cuda_stream) == NVJPEG_STATUS_SUCCESS
                   ? expected_t {}
                   : expected_t::fail(status_t::cuda_error_k, "nvJPEG batched decode failed");
    // Hardware path: one decode-params object per image carries its scale factor.
    array<nvjpegDecodeParams_t> params;
    if (params.try_resize(count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "nvJPEG decode-params allocation failed");
    std::size_t created = 0;
    expected_t outcome {};
    for (created = 0; created != count; ++created) {
        if (nvjpegDecodeParamsCreate(handle_, &params[created]) != NVJPEG_STATUS_SUCCESS) {
            outcome = expected_t::fail(status_t::cuda_error_k, "nvJPEG decode-params create failed");
            break;
        }
        nvjpegDecodeParamsSetOutputFormat(params[created], NVJPEG_OUTPUT_RGBI);
        nvjpegDecodeParamsSetScaleFactor(params[created], entries[created].scale);
    }
    if (outcome && nvjpegDecodeBatchedEx(handle_, jpeg_state_, data.data(), lengths.data(), destinations.data(),
                                         params.data(), cuda_stream) != NVJPEG_STATUS_SUCCESS)
        outcome = expected_t::fail(status_t::cuda_error_k, "nvJPEG batched decode (scaled) failed");
    for (std::size_t index = 0; index != created; ++index) nvjpegDecodeParamsDestroy(params[index]);
    return outcome;
}

/** @brief Fill an @c NppStreamContext from the driver — NPP needs the device's launch limits, which come
 *      from @c cuDeviceGetAttribute (no CUDA Runtime), and the driver stream cast to @c cudaStream_t . */
[[nodiscard]] static expected<NppStreamContext> npp_context_(gpu_stream_t const &stream) noexcept {
    using result_t = expected<NppStreamContext>;
    NppStreamContext context {};
    // vendor-boundary: the driver stream cast to NPP's runtime-API `cudaStream_t` handle.
    context.hStream = reinterpret_cast<cudaStream_t>(stream.get());
    CUdevice device = 0;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "NPP device query failed");
    context.nCudaDeviceId = device;
    // Accumulate the first failing driver result and push it up typed — never collapse it to a bare bool.
    CUresult result = cuStreamGetFlags(stream.get(), &context.nStreamFlags);
    auto attribute = [&](CUdevice_attribute which, int &into) {
        if (result == CUDA_SUCCESS) result = cuDeviceGetAttribute(&into, which, device);
    };
    attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, context.nMultiProcessorCount);
    attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, context.nMaxThreadsPerMultiProcessor);
    attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, context.nMaxThreadsPerBlock);
    int shared = 0;
    attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, shared);
    context.nSharedMemPerBlock = static_cast<std::size_t>(shared);
    attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, context.nCudaDevAttrComputeCapabilityMajor);
    attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, context.nCudaDevAttrComputeCapabilityMinor);
    if (result != CUDA_SUCCESS) return result_t::fail(result, "NPP device attribute query failed");
    return result_t {context};
}

/** @brief Triage every drawn blob on the host: sniff its format and native size off a header prefix, derive
 *      its column budget (from native resolution, clamped so many images share a row) and its factor pair
 *      (left in @c plan.grid , landed device-side later by @ref image_segments_kernel_ ) and — for a hardware
 *      JPEG — its decode scale, and lay out the @c decoded -sized RGBA scratch regions (a scaled hardware JPEG
 *      needs only its shrunk pixels; a CPU-fallback plan's @c decoded equals its native size). @p tape slices
 *      the reservoir (device bytes, host offsets) via @c tape[row] . Returns the scratch byte total. One
 *      header-prefix hop, one sync. */
[[nodiscard]] static expected<std::size_t> triage_images_(bool hardware, column_view_t tape,
                                                          std::span<u64_t const> drawn_rows, u32_t patch_size,
                                                          u32_t max_columns, u32_t max_segments,
                                                          std::span<image_plan_t> plans,
                                                          gpu_stream_t &stream) noexcept {
    using result_t = expected<std::size_t>;
    std::size_t const images = drawn_rows.size();
    // Build the per-image gather descriptors on the host — a device source pointer and a capped header length
    // straight from each tape slice — then lift every header in one kernel + one D2H, not one copy per image.
    array<gather_entry_t> host_entries;
    array<gather_entry_t, gpu_alloc<gather_entry_t>> device_entries {gpu_alloc<gather_entry_t> {stream.get()}};
    array<u8_t, gpu_alloc<u8_t>> staging {gpu_alloc<u8_t> {stream.get()}};
    array<u8_t, pinned_alloc<u8_t>> probe;
    if (host_entries.try_resize(images) != status_t::success_k ||
        device_entries.try_resize_uninitialized(images) != status_t::success_k ||
        staging.try_resize_uninitialized(images * header_probe_bytes_k) != status_t::success_k ||
        probe.try_resize_uninitialized(images * header_probe_bytes_k) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "header gather staging allocation failed");
    for (std::size_t index = 0; index != images; ++index) {
        auto const blob = tape[drawn_rows[index]];
        host_entries[index] = gather_entry_t {reinterpret_cast<u8_t const *>(blob.data()),
                                              static_cast<u32_t>(std::min(blob.size(), header_probe_bytes_k))};
    }
    if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_entries.data()), host_entries.data(),
                          images * sizeof(gather_entry_t), stream.get()) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "header gather descriptor upload failed");
    if (expected_t launched = launch_kernel(
            reinterpret_cast<void const *>(&gather_headers_kernel_), dim3 {grid_size_for(images, 1u), 1u, 1u},
            dim3 {kernel_block_size_default_k, 1u, 1u}, stream, device_entries.data(), staging.data(),
            static_cast<u32_t>(header_probe_bytes_k), static_cast<u32_t>(images));
        !launched)
        return result_t::fail(std::move(launched));
    if (cuMemcpyDtoHAsync(probe.data(), reinterpret_cast<CUdeviceptr>(staging.data()), images * header_probe_bytes_k,
                          stream.get()) != CUDA_SUCCESS ||
        cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "header gather readback failed");

    std::size_t scratch_bytes = 0;
    for (std::size_t index = 0; index != images; ++index) {
        sz::string_view const header {reinterpret_cast<char const *>(probe.data() + index * header_probe_bytes_k),
                                      host_entries[index].length};
        auto probed = probe_image_(header);
        if (!probed) return result_t::fail(std::move(probed));
        image_plan_t &plan = plans[index];
        plan.format = probed.value_.format;
        plan.native = probed.value_.extent;
        // The image's column budget is native-resolution driven, then factored toward its aspect. The grid
        // lands in @c image_shapes on the device (via the segment kernel), never host-written to arena memory.
        u32_t const budget = column_budget_(plan.native, patch_size, max_columns, max_segments);
        plan.grid = factor_pair_(plan.native, budget);
        // The RGBA scratch region is sized by @c decoded , not @c native : a hardware JPEG shrinks toward its
        // target at decode (down to ~1/64 the pixels), so a native region would over-allocate up to 64x. A CPU
        // fallback plan (PNG / WebP, or a rejected JPEG) keeps @c decoded == @c native , so it still fits.
        image_extent_t const target {u32_t(plan.grid.cols) * patch_size, u32_t(plan.grid.rows) * patch_size};
        bool const jpeg = plan.format == format_jpeg_k;
        plan.route = jpeg ? route_t::jpeg_device_k : route_t::cpu_fallback_k;
        plan.scale = jpeg && hardware ? pick_scale_(plan.native, target) : NVJPEG_SCALE_NONE;
        plan.decoded = scaled_extent_(plan.native, plan.scale);
        plan.scratch_offset = scratch_bytes;
        scratch_bytes += static_cast<std::size_t>(plan.decoded.width) * plan.decoded.height * image_channels_k;
    }
    return result_t {scratch_bytes};
}

/** @brief One JPEG entry's RGB→RGBA expansion: the packed @b [pixels,3] nvJPEG output and its @b [pixels,4]
 *      destination in the RGBA scratch, both contiguous (packed per row), so pixel @c p maps @c 3p→4p . */
struct rgba_expand_entry_t {
    /** @brief The packed @b [pixels,3] nvJPEG RGB output to widen. */
    u8_t const *rgb;
    /** @brief The packed @b [pixels,4] RGBA scratch destination. */
    u8_t *rgba;
    /** @brief The pixel count shared by both, so pixel @c p maps @c 3p→4p . */
    u32_t pixels;
};

/**
 *  @brief Expand each decoded JPEG's packed RGB into its RGBA scratch region, padding a constant @c 255 alpha —
 *      nvJPEG has no alpha output, so the device path decodes RGB and this kernel widens it in place of the
 *      library. PNG / WebP keep their real alpha and never pass through here.
 *
 *  __grid_size__ @c [entries,1,1] , one block per entry, grid-stride over both axes; no dynamic shared memory.
 */
static __global__ void rgb_to_rgba_kernel_(rgba_expand_entry_t const *entries, u32_t count) {
    for (u32_t index = blockIdx.x; index < count; index += gridDim.x) {
        rgba_expand_entry_t const entry = entries[index];
        for (u32_t pixel = threadIdx.x; pixel < entry.pixels; pixel += blockDim.x) {
            entry.rgba[pixel * 4u + 0u] = entry.rgb[pixel * 3u + 0u];
            entry.rgba[pixel * 4u + 1u] = entry.rgb[pixel * 3u + 1u];
            entry.rgba[pixel * 4u + 2u] = entry.rgb[pixel * 3u + 2u];
            entry.rgba[pixel * 4u + 3u] = 255u;
        }
    }
}

/** @brief Decode the plan's JPEG rows through @p jpeg into a packed RGB scratch, then widen each into its RGBA
 *      scratch region ( @ref rgb_to_rgba_kernel_ , alpha @c=255 ) — nvJPEG emits no alpha. A rejected batch is
 *      not fatal: every JPEG entry flips to @c route_t::cpu_fallback_k keeping its scaled @c decoded extent, so
 *      the CPU fallback shrinks to the same @c decoded -sized scratch region it would have filled. */
[[nodiscard]] static expected_t decode_jpeg_plans_(nvjpeg_session_t &jpeg, std::span<image_plan_t> plans, u8_t *scratch,
                                                   column_view_t tape, std::span<u64_t const> drawn_rows,
                                                   gpu_stream_t &stream) noexcept {
    constexpr u32_t rgb_channels_k = 3u; // nvJPEG's interleaved-RGB output, widened to RGBA afterward
    array<jpeg_entry_t> entries;
    std::size_t rgb_bytes = 0;
    for (std::size_t index = 0; index != plans.size(); ++index) {
        image_plan_t const &plan = plans[index];
        if (plan.format != format_jpeg_k) continue;
        auto const blob = tape[drawn_rows[index]];
        std::size_t const pixels = static_cast<std::size_t>(plan.decoded.width) * plan.decoded.height;
        if (entries.try_push_back(jpeg_entry_t {reinterpret_cast<unsigned char const *>(blob.data()), blob.size(),
                                                index, rgb_bytes, plan.scale, nvjpegImage_t {}}) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "JPEG batch table growth failed");
        rgb_bytes += pixels * rgb_channels_k;
    }
    if (entries.empty()) return {};

    // The nvJPEG batch lands packed RGB into this temporary; the widen kernel then fills each RGBA scratch region.
    array<u8_t, gpu_alloc<u8_t>> rgb_scratch {gpu_alloc<u8_t> {stream.get()}};
    if (rgb_scratch.try_resize_uninitialized(rgb_bytes) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "JPEG RGB scratch allocation failed ({} bytes)", rgb_bytes);
    u8_t *const rgb_base = rgb_scratch.data();
    // Wire each entry's RGB destination into the fresh scratch; the owning entry table then feeds nvJPEG's
    // parallel arrays straight from itself, so the split lives only inside @ref decode_jpeg_batch .
    for (std::size_t index = 0; index != entries.size(); ++index) {
        jpeg_entry_t &entry = entries[index];
        image_plan_t const &plan = plans[entry.plan_index];
        entry.destination.channel[0] = rgb_base + entry.rgb_offset;
        entry.destination.pitch[0] = static_cast<std::size_t>(plan.decoded.width) * rgb_channels_k;
    }
    if (expected_t decoded = jpeg.decode_jpeg_batch({entries.data(), entries.size()}, stream); !decoded) {
        // Route every JPEG to the CPU pool but keep its scaled @c decoded extent and @c scale , so the CPU
        // decode shrinks to fit the @c decoded -sized scratch region triage laid out for it.
        for (std::size_t index = 0; index != entries.size(); ++index)
            plans[entries[index].plan_index].route = route_t::cpu_fallback_k;
        return {};
    }

    // Widen the packed RGB into the RGBA scratch (alpha 255) — one descriptor per decoded entry, uploaded once.
    array<rgba_expand_entry_t> host_expand;
    array<rgba_expand_entry_t, gpu_alloc<rgba_expand_entry_t>> device_expand {
        gpu_alloc<rgba_expand_entry_t> {stream.get()}};
    if (host_expand.try_resize(entries.size()) != status_t::success_k ||
        device_expand.try_resize_uninitialized(entries.size()) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "RGBA expansion descriptor allocation failed");
    for (std::size_t index = 0; index != entries.size(); ++index) {
        jpeg_entry_t const &entry = entries[index];
        image_plan_t const &plan = plans[entry.plan_index];
        host_expand[index] = rgba_expand_entry_t {rgb_base + entry.rgb_offset, scratch + plan.scratch_offset,
                                                  static_cast<u32_t>(plan.decoded.width) * plan.decoded.height};
    }
    if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_expand.data()), host_expand.data(),
                          entries.size() * sizeof(rgba_expand_entry_t), stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "RGBA expansion descriptor upload failed");
    if (expected_t widened = launch_kernel(reinterpret_cast<void const *>(&rgb_to_rgba_kernel_),
                                           dim3 {grid_size_for(entries.size(), 1u), 1u, 1u},
                                           dim3 {kernel_block_size_default_k, 1u, 1u}, stream, device_expand.data(),
                                           static_cast<u32_t>(entries.size()));
        !widened)
        return widened;
    // The RGB scratch and the descriptor table are freed stream-ordered after the widen kernel — hold the sync
    // for the caller's resize, which reads only the RGBA scratch.
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "JPEG RGBA expansion synchronization failed");
    return {};
}

/** @brief Decode the plan's @c cpu_fallback_k rows (PNG, WebP, and any rejected JPEG) on @p pool : one pinned blob
 *      hop in, an RGBA decode per worker at the plan's @c decoded extent (native for PNG / WebP, the scaled
 *      size for a rejected hardware JPEG), then one bulk host-to-device copy into each scratch region. */
[[nodiscard]] static expected_t decode_cpu_plans_(cpu_decode_pool_t &pool, std::span<image_plan_t> plans, u8_t *scratch,
                                                  column_view_t tape, std::span<u64_t const> drawn_rows,
                                                  gpu_stream_t &stream) noexcept {
    struct cpu_task_t {
        /** @brief The plan this task decodes. */
        std::size_t plan_index;
        /** @brief The task's byte offset into the pinned blob tape. */
        std::size_t blob_offset;
        /** @brief The compressed blob's byte count. */
        std::size_t blob_length;
        /** @brief The task's byte offset into the pinned decoded-pixel tape. */
        std::size_t pixel_offset;
        /** @brief The decoded RGBA pixel byte count. */
        std::size_t pixel_length;
    };
    array<cpu_task_t> tasks;
    std::size_t blob_bytes = 0, pixel_bytes = 0;
    for (std::size_t index = 0; index != plans.size(); ++index) {
        image_plan_t const &plan = plans[index];
        if (plan.route != route_t::cpu_fallback_k) continue;
        std::size_t const length = tape[drawn_rows[index]].size();
        std::size_t const pixels = static_cast<std::size_t>(plan.decoded.width) * plan.decoded.height *
                                   image_channels_k;
        if (tasks.try_push_back({index, blob_bytes, length, pixel_bytes, pixels}) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "CPU task list growth failed");
        blob_bytes += length;
        pixel_bytes += pixels;
    }
    if (tasks.empty()) return {};

    array<u8_t, pinned_alloc<u8_t>> blobs, decoded;
    array<u8_t> results;
    if (blobs.try_resize_uninitialized(blob_bytes) != status_t::success_k ||
        decoded.try_resize_uninitialized(pixel_bytes) != status_t::success_k ||
        results.try_resize(tasks.size()) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "CPU staging allocation failed");
    for (std::size_t index = 0; index != tasks.size(); ++index) {
        cpu_task_t const &task = tasks[index];
        auto const blob = tape[drawn_rows[task.plan_index]];
        if (cuMemcpyDtoHAsync(blobs.data() + task.blob_offset, reinterpret_cast<CUdeviceptr>(blob.data()),
                              task.blob_length, stream.get()) != CUDA_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "CPU blob copy failed at image {}", task.plan_index);
    }
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "CPU blob synchronization failed");

    cpu_task_t const *const task_base = tasks.data();
    image_plan_t const *const plan_base = plans.data();
    u8_t *const blob_base = blobs.data();
    u8_t *const pixel_base = decoded.data();
    u8_t *const result_base = results.data();
    pool.for_n(tasks.size(), [=](fu::prong_t prong) noexcept {
        cpu_task_t const &task = task_base[prong.task];
        image_plan_t const &plan = plan_base[task.plan_index];
        sz::string_view const encoded {reinterpret_cast<char const *>(blob_base + task.blob_offset), task.blob_length};
        pixel_image_span_t const destination {reinterpret_cast<pixel_t *>(pixel_base + task.pixel_offset),
                                              {plan.decoded.height, plan.decoded.width, image_channels_k}};
        result_base[prong.task] = cpu_decode_(plan.format, encoded, destination) == status_t::success_k ? 1u : 0u;
    });
    for (std::size_t index = 0; index != tasks.size(); ++index) {
        cpu_task_t const &task = tasks[index];
        if (!results[index])
            return expected_t::fail(status_t::invalid_input_k, "CPU decode failed at image {}", task.plan_index);
        if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(scratch + plans[task.plan_index].scratch_offset),
                              pixel_base + task.pixel_offset, task.pixel_length, stream.get()) != CUDA_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "CPU pixel upload failed at image {}", task.plan_index);
    }
    // Hold for the uploads: the pinned decode tape ( @c blobs / @c decoded ) frees synchronously on return,
    // so it must outlive its in-flight @c cuMemcpyHtoDAsync into the device scratch (a use-after-free otherwise).
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "CPU pixel upload synchronization failed");
    return {};
}

/** @brief Scatter one image-segment's resized contiguous @b [gr·P,gc·P,C] frame into the batch's shared patch
 *      grid @b [rows,columns,P,P,C] — patch @c (pr,pc) lands at the flat cell @c offset+pr·gc+pc , i.e.
 *      @c images[row,column_begin+local,:,:,:] where @c row=offset/columns and @c column_begin=offset%columns .
 *      Reads the segment's placement from @p segment_offsets / @p segment_lengths and its resized source base
 *      from @p resized_offsets , so several images share a row and each rides its own column sub-range.
 *
 *  __grid_size__ @c [images,1,1] , one block per segment (a PAD / overflow-dropped segment, length 0, exits
 *      immediately); threads stride the segment's @c budget·P²·C elements. No dynamic shared memory. */
static __global__ void tile_patches_kernel_(pixel_patches_span_t images, pixel_t const *resized,
                                            u32_t const *resized_offsets, image_shape_view_t image_shapes,
                                            u32r1_view_t segment_offsets, u32r1_view_t segment_lengths) {
    u32_t const seg = blockIdx.x;
    u32_t const budget = segment_lengths[seg];
    if (budget == 0u) return;
    u32_t const columns_per_row = static_cast<u32_t>(images.extent(1));
    u32_t const patch_size = static_cast<u32_t>(images.extent(2));
    u32_t const channels = static_cast<u32_t>(images.extent(4));
    u32_t const offset = segment_offsets[seg];
    u32_t const row = offset / columns_per_row;
    u32_t const column_begin = offset - row * columns_per_row;
    u32_t const cols = image_shapes[seg, 1];
    std::size_t const per_image = static_cast<std::size_t>(budget) * patch_size * patch_size * channels;
    std::size_t const source_base = static_cast<std::size_t>(resized_offsets[seg]);
    for (std::size_t element = threadIdx.x; element < per_image; element += blockDim.x) {
        std::size_t rest = element;
        u32_t const channel = static_cast<u32_t>(rest % channels);
        rest /= channels;
        u32_t const patch_x = static_cast<u32_t>(rest % patch_size);
        rest /= patch_size;
        u32_t const patch_y = static_cast<u32_t>(rest % patch_size);
        rest /= patch_size;
        u32_t const local = static_cast<u32_t>(rest); // this segment's patch index, 0..budget
        u32_t const patch_row = local / cols;
        u32_t const patch_col = local - patch_row * cols;
        std::size_t const source_row = static_cast<std::size_t>(patch_row) * patch_size + patch_y;
        std::size_t const source_col = static_cast<std::size_t>(patch_col) * patch_size + patch_x;
        std::size_t const source_width = static_cast<std::size_t>(cols) * patch_size;
        std::size_t const source = source_base + (source_row * source_width + source_col) * channels + channel;
        images[row, column_begin + local, patch_y, patch_x, channel] = resized[source];
    }
}

/** @brief Crop each decoded RGBA scratch image to its factor-pair aspect, Lanczos-resize it into a contiguous
 *      scratch tape (one @c nppiResize per image — NPP's batched variant takes neither single-image batches
 *      nor Lanczos), then scatter each into its column sub-range of the batch's shared patch
 *      grid — the scratch and device-resident offsets are held alive until the whole stream-ordered chain
 *      retires. Budgets vary per image now, so each resized frame's byte size (and its base in the tape) is a
 *      running prefix, not a fixed stride. @p segment_offsets / @p segment_lengths carry the placement the
 *      segment kernel laid down. */
[[nodiscard]] static expected_t resize_plans_(std::span<image_plan_t const> plans, u8_t *scratch,
                                              pixel_patches_span_t pixels, image_shape_view_t image_shapes,
                                              u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                              gpu_stream_t &stream) noexcept {
    std::size_t const images = plans.size();
    u32_t const patch_size = static_cast<u32_t>(pixels.extent(2)); // the square patch side, straight off the grid
    auto context = npp_context_(stream);
    if (!context) return expected_t::fail(std::move(context));

    array<u32_t> host_resized_offsets; // each image's element base in the packed resized tape
    if (host_resized_offsets.try_resize(images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "resize offset allocation failed");
    // Lay the variable-budget resized frames end to end; a prefix scan gives each one's base in the tape.
    std::size_t resized_elements = 0;
    for (std::size_t index = 0; index != images; ++index) {
        image_plan_t const &plan = plans[index];
        std::size_t const budget = static_cast<std::size_t>(plan.grid.rows) * plan.grid.cols;
        host_resized_offsets[index] = static_cast<u32_t>(resized_elements);
        resized_elements += budget * patch_size * patch_size * image_channels_k;
    }

    // The batched resize lands each image contiguously in a scratch tape; a tile kernel then scatters it into
    // the patch grid, since a resized image's row-major pixels are not the batch's [columns,P,P,C] order.
    array<u8_t, gpu_alloc<u8_t>> resized {gpu_alloc<u8_t> {stream.get()}};
    if (resized.try_resize_uninitialized(resized_elements) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "resized pixel scratch allocation failed ({} bytes)",
                                resized_elements);
    pixel_t *const resized_base = resized.data();

    // NPP's batched resize rejects single-image batches (its @c nBatchSize contract demands more than one)
    // and rejects Lanczos outright ( @c NPP_INTERPOLATION_ERROR ) — per-image @c nppiResize keeps the
    // Lanczos quality the training data wants, needs no device-resident descriptor tables, and the calls
    // serialize on the one stream exactly like a batched launch would.
    for (std::size_t index = 0; index != images; ++index) {
        image_plan_t const &plan = plans[index];
        image_extent_t const source = plan.decoded;
        image_extent_t const target {u32_t(plan.grid.cols) * patch_size, u32_t(plan.grid.rows) * patch_size};
        NppiSize const source_size {static_cast<int>(source.width), static_cast<int>(source.height)};
        NppiSize const target_size {static_cast<int>(target.width), static_cast<int>(target.height)};
        NppStatus const resized_one = nppiResize_8u_C4R_Ctx(
            reinterpret_cast<Npp8u const *>(scratch + plan.scratch_offset),
            static_cast<int>(source.width * image_channels_k), source_size, center_crop_(source, target),
            reinterpret_cast<Npp8u *>(resized_base + host_resized_offsets[index]),
            static_cast<int>(target.width * image_channels_k), target_size,
            NppiRect {0, 0, target_size.width, target_size.height}, NPPI_INTER_LANCZOS, context.value_);
        if (resized_one != NPP_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "image resize failed (status={} image={})",
                                    static_cast<int>(resized_one), index);
    }
    // The per-image resized bases must be device-resident for the tile kernel — owning. It reads each
    // segment's column count straight from the device-side @c image_shapes , no second copy here.
    array<u32_t, gpu_alloc<u32_t>> device_resized_offsets {gpu_alloc<u32_t> {stream.get()}};
    if (device_resized_offsets.try_resize_uninitialized(images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "resize offset device allocation failed");
    if (CUresult uploaded = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_resized_offsets.data()),
                                              host_resized_offsets.data(), images * sizeof(u32_t), stream.get());
        uploaded != CUDA_SUCCESS)
        return expected_t::fail(uploaded, "resize offset upload failed");

    // Scatter every resized image into its column sub-range (one CTA per segment), then retire the whole
    // stream-ordered chain before the owning scratch / descriptor arrays fall out of scope.
    if (expected_t tiled = launch_kernel(reinterpret_cast<void const *>(&tile_patches_kernel_),
                                         dim3 {static_cast<u32_t>(images), 1u, 1u},
                                         dim3 {kernel_block_size_default_k, 1u, 1u}, stream, pixels, resized_base,
                                         device_resized_offsets.data(), image_shapes, segment_offsets, segment_lengths);
        !tiled)
        return tiled;
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "resize/tile synchronization failed");
    return {};
}

expected_t ul::decode_images(nvjpeg_session_t &jpeg, cpu_decode_pool_t &pool, column_view_t tape,
                             std::span<u64_t const> drawn_rows, std::span<image_batch_shape_t const> representations,
                             pixel_patches_span_t pixels, image_shape_span_t shapes, u32r1_span_t segment_offsets,
                             u32r1_span_t segment_lengths, modality_span_t segment_modality,
                             gpu_stream_t &stream) noexcept {
    std::size_t const images = drawn_rows.size();
    if (images == 0) return {};
    // The batch grid names the geometry: columns is the row width, extent(2) the square patch side, extent(4)
    // the RGBA channel count; the segment axis is the descriptors' leading extent.
    u32_t const columns_per_row = static_cast<u32_t>(pixels.extent(1));
    u32_t const patch_size = static_cast<u32_t>(pixels.extent(2));
    u32_t const channels = static_cast<u32_t>(pixels.extent(4));
    u32_t const max_segments = static_cast<u32_t>(segment_lengths.extent(0));
    std::size_t const capacity = static_cast<std::size_t>(pixels.extent(0)) * columns_per_row;
    if (images > max_segments || shapes.extent(0) < max_segments || segment_offsets.extent(0) < max_segments + 1u ||
        segment_modality.extent(0) < max_segments)
        return expected_t::fail(status_t::shape_unsupported_k, "batch segments cannot hold {} images", images);
    if (patch_size == 0 || columns_per_row == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "patch_size and columns must be nonzero");
    // The shape is the single source of geometry; the grid is the validated destination. Image carries no per-
    // representation variation yet (footprint is native-derived, patch side and channels are shared), so it consumes
    // one representation, matching the uniform decode contract while multi-representation awaits a varying field.
    if (representations.empty())
        return expected_t::fail(status_t::invalid_input_k, "decode_images requires a non-empty representations span");
    if (representations[0].patch_side != patch_size || representations[0].channels != channels)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "representation (patch_side={} channels={}) must match the image grid ({},{})",
                                representations[0].patch_side, representations[0].channels, patch_size, channels);

    array<image_plan_t> plans;
    if (plans.try_resize(images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "image plan allocation failed");
    std::span<image_plan_t> const plan_span {plans.data(), images};
    auto scratch_bytes = triage_images_(jpeg.backend() == nvjpeg_backend_t::hardware_k, tape, drawn_rows, patch_size,
                                        columns_per_row, max_segments, plan_span, stream);
    if (!scratch_bytes) return expected_t::fail(std::move(scratch_bytes));

    // Land the host-computed factor pairs device-side and walk the token cursor: it writes image_shapes ,
    // segment_offsets / lengths , and segment_modality , so the resize/tile then read placement off the device.
    array<patch_grid_t> host_grids;
    array<patch_grid_t, gpu_alloc<patch_grid_t>> device_grids {gpu_alloc<patch_grid_t> {stream.get()}};
    if (host_grids.try_resize(images) != status_t::success_k ||
        device_grids.try_resize_uninitialized(images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "segment grid staging allocation failed");
    for (std::size_t index = 0; index != images; ++index) host_grids[index] = plans[index].grid;
    if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_grids.data()), host_grids.data(),
                          images * sizeof(patch_grid_t), stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "segment grid upload failed");
    if (expected_t segmented = launch_kernel(reinterpret_cast<void const *>(&image_segments_kernel_), dim3 {1u, 1u, 1u},
                                             dim3 {1u, 1u, 1u}, stream, device_grids.data(), static_cast<u32_t>(images),
                                             columns_per_row, static_cast<u32_t>(capacity), shapes, segment_offsets,
                                             segment_lengths, segment_modality);
        !segmented)
        return segmented;

    // One decoded-sized RGBA device scratch holds every image's decoded pixels, laid out by drawn order (a
    // hardware JPEG's region is sized to its scaled output, not native) — owning, so it frees itself after the
    // resize's sync.
    array<u8_t, gpu_alloc<u8_t>> scratch {gpu_alloc<u8_t> {stream.get()}};
    if (scratch.try_resize_uninitialized(scratch_bytes.value_) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "decoded pixel scratch allocation failed ({} bytes)",
                                scratch_bytes.value_);
    u8_t *const scratch_base = scratch.data();

    if (expected_t r = decode_jpeg_plans_(jpeg, plan_span, scratch_base, tape, drawn_rows, stream); !r) return r;
    if (expected_t r = decode_cpu_plans_(pool, plan_span, scratch_base, tape, drawn_rows, stream); !r) return r;
    return resize_plans_(plan_span, scratch_base, pixels,
                         image_shape_view_t {shapes.data(), {shapes.extent(0), shapes.extent(1)}},
                         u32r1_view_t {segment_offsets.data(), segment_offsets.extent(0)},
                         u32r1_view_t {segment_lengths.data(), segment_lengths.extent(0)}, stream);
}

expected<image_loader_parquet_t> image_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                              sz::string_view path_or_glob, u64_t seed,
                                                              cpu_decode_pool_t &pool,
                                                              image_batch_shape_t image) noexcept {
    using result_t = expected<image_loader_parquet_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths — POSIX glob(3).
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    image_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.pool_ = &pool;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;
    loader.image_ = image;

    // Footer only — host-side, no GPU: a running prefix of each file's row count, the weighted file draw's
    // search table; its back holds the dataset's total rows.
    if (loader.row_prefix_.try_resize(loader.absolute_paths_.size() + 1) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "shard prefix allocation failed ({} shards)",
                              loader.absolute_paths_.size());
    loader.row_prefix_[0] = 0;
    for (std::size_t index = 0; index != loader.absolute_paths_.size(); ++index) {
        auto metadata = parquet_file_t::table_metadata(session, loader.absolute_paths_[index]);
        if (!metadata) return result_t::fail(std::move(metadata));
        if (metadata->rows == 0)
            return result_t::fail(status_t::invalid_input_k, "shard {} is empty", loader.absolute_paths_[index]);
        loader.row_prefix_[index + 1] = loader.row_prefix_[index] + metadata->rows;
    }
    return result_t {std::move(loader)};
}

expected_t image_loader_parquet_t::fill(gpu_stream_t &gpu_stream, nvjpeg_session_t &decoder,
                                        image_loader_parquet_config_t config, u32_t gpu_rank, u32_t training_step,
                                        std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    // The draw count is the segment capacity now — many small images pack into a row, so a fill draws up to
    // max_segments_per_gpu items and the cursor keeps as many as the rows·columns grid holds.
    std::size_t const images_per_batch = batches[0].segment_lengths.extent(0);
    if (images_per_batch == 0 || batches[0].images.extent(0) == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "the batch carries no image segments");
    std::size_t const total_images = unroll * images_per_batch;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_rows = row_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};

    // Open distinct files, drawn with probability proportional to their row count — a keyed row position over
    // the whole dataset binary-searches the prefix to its owning file — until the pool covers the fill's
    // images; never fewer than two, unless the whole dataset is one file.
    array<sz::string_view> picked_paths;
    array<std::size_t> picked_shards;
    if (picked_paths.try_resize(shard_count) != status_t::success_k ||
        picked_shards.try_resize(shard_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
    std::size_t picked_count = 0, pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < total_images || picked_count < floor_files) && picked_count < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        std::span<std::size_t const> const chosen {picked_shards.data(), picked_count};
        if (std::find(chosen.begin(), chosen.end(), file) != chosen.end()) continue;
        picked_shards[picked_count] = file;
        picked_paths[picked_count] = absolute_paths_[file];
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
        ++picked_count;
    }

    // Decode the picked files' blob column into one device reservoir; the true pool is the decoded rows.
    auto reservoir_read = parquet_file_t::make(*storage_, picked_paths.view().subspan(0, picked_count),
                                               {&config.column, 1}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_images)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files cover fewer than the {} images this fill needs",
                                pooled_rows, picked_count, total_images);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");

    // The reservoir's blob offsets to the host once — every per-row byte bound is then host arithmetic.
    column_view_t const column = reservoir.column(0);
    array<u64_t> reservoir_offsets;
    if (reservoir_offsets.try_resize_uninitialized(column.offsets_.size()) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "reservoir offsets landing failed ({} rows)", pooled_rows);
    if (cuMemcpyDtoHAsync(reservoir_offsets.data(), reinterpret_cast<CUdeviceptr>(column.offsets_.data()),
                          column.offsets_.size() * sizeof(u64_t), gpu_stream.get()) != CUDA_SUCCESS ||
        cuStreamSynchronize(gpu_stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "reservoir offsets readback failed");

    // Draw the whole fill's rows on the host — a keyed bijection over the pool, so each batch's slice is a
    // distinct scatter of rows across the pooled files.
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    array<u64_t> record_indices;
    if (record_indices.try_resize_uninitialized(total_images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_images);
    for (std::size_t index = 0; index != total_images; ++index)
        record_indices[index] = epoch_permutation(row_key, data_epoch_t {training_step}, index, pooled_rows);

    // The blob column re-viewed with its offsets on host: device bytes, host-indexable via @c tape[row] .
    sz::span<u64_t const> const host_offsets {reservoir_offsets.data(), reservoir_offsets.size()};
    column_view_t const tape {column.buffer_, host_offsets};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (images_per_batch > loaded.image_shapes.extent(0) || images_per_batch > loaded.segment_lengths.extent(0))
            return expected_t::fail(status_t::shape_unsupported_k, "batch {} cannot hold {} image segments", batch,
                                    images_per_batch);
        // The batch's [rows,columns,P,P,C] image tensor names the whole grid directly: extent(2) is the square
        // patch side and extent(4) the RGBA channels — no arithmetic from a flattened run.
        std::span<u64_t const> const drawn {record_indices.data() + batch * images_per_batch, images_per_batch};

        pixel_patches_span_t const pixels {loaded.images.data(),
                                           {loaded.images.extent(0), loaded.images.extent(1), loaded.images.extent(2),
                                            loaded.images.extent(3), loaded.images.extent(4)}};
        image_shape_span_t const shapes {loaded.image_shapes.data(),
                                         {loaded.image_shapes.extent(0), loaded.image_shapes.extent(1)}};
        u32r1_span_t const segment_offsets {loaded.segment_offsets.data(), loaded.segment_offsets.extent(0)};
        u32r1_span_t const segment_lengths {loaded.segment_lengths.data(), loaded.segment_lengths.extent(0)};
        modality_span_t const segment_modality {loaded.segment_modality.data(), loaded.segment_modality.extent(0)};
        std::span<image_batch_shape_t const> const representations {&image_, 1}; // one representation today
        if (expected_t decoded = decode_images(decoder, *pool_, tape, drawn, representations, pixels, shapes,
                                               segment_offsets, segment_lengths, segment_modality, gpu_stream);
            !decoded)
            return decoded;
    }
    return {};
}
