/**
 *  @brief The Parquet image loader's bodies and kernels — see the header for the contract. Its shape
 *      mirrors @c tokens_loader.cu : a keyed file pick, one pooled reservoir, a keyed row draw, then a
 *      per-batch decode. The decode is the shared @ref decode_images : JPEG rows through the @ref
 *      nvjpeg_session_t straight off the reservoir tape, PNG/WebP rows on the CPU pool, then one NPP batch
 *      crops each to its factor-pair aspect and resizes it into the batch's uniform @c images run.
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
#include <fork_union.hpp>             // vendor-boundary: the host-CPU decode worker pool

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
/** @brief The channels every decoded pixel carries — interleaved RGB. */
inline constexpr u32_t image_channels_k = 3u;

/** @brief The compressed formats a drawn blob can be, decided by its magic bytes. */
enum image_format_t : u8_t { format_jpeg_k = 0, format_png_k = 1, format_webp_k = 2, format_unknown_k = 3 };

/**
 *  @brief Write the batch's segment geometry: one segment per image row, each spanning its whole row of the
 *      flat @c row·columns space — a media item covers its row by the batch's contract. Thread 0 walks the
 *      slots serially; slots past @c live_images PAD (zero length, repeated offset).
 *
 *  __grid_size__ @c [1,1,1] , one thread live; no dynamic shared memory.
 */
static __global__ void image_segments_kernel_(u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                              u32_t live_images, u32_t columns_per_row) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    u32_t const total_segments = static_cast<u32_t>(segment_lengths.extent(0));
    u32_t const live = live_images < total_segments ? live_images : total_segments;
    for (u32_t slot = 0; slot != total_segments; ++slot) {
        segment_offsets[slot] = (slot < live ? slot : live) * columns_per_row;
        segment_lengths[slot] = slot < live ? columns_per_row : 0u;
    }
    segment_offsets[total_segments] = live * columns_per_row;
}

/** @brief One image's header-gather descriptor: where its compressed bytes start in the device tape and how
 *      many of them (capped at the probe size) to lift. */
struct gather_slot_t {
    u8_t const *source;
    u32_t length;
};

/**
 *  @brief Gather each drawn blob's header prefix from the scattered device tape into a contiguous staging
 *      buffer — one block per image, threads stripe the copy — so triage reads them back in one D2H instead
 *      of one small transfer per image.
 *
 *  __grid_size__ @c [min(images,cap),1,1] , grid-stride over images; no dynamic shared memory.
 */
static __global__ void gather_headers_kernel_(gather_slot_t const *slots, u8_t *staging, u32_t stride, u32_t images) {
    for (u32_t image = blockIdx.x; image < images; image += gridDim.x) {
        gather_slot_t const slot = slots[image];
        u8_t *const destination = staging + static_cast<std::size_t>(image) * stride;
        for (u32_t byte = threadIdx.x; byte < slot.length; byte += blockDim.x) destination[byte] = slot.source[byte];
    }
}

/** @brief A decoded image's pixel dimensions — a size travels as one value, not two loose scalars. */
struct image_extent_t {
    u32_t width = 0, height = 0;
};

/** @brief The @c (rows,cols) factorization of the patch budget one image tiles into. */
struct patch_grid_t {
    u16_t rows = 0, cols = 0;
};

/** @brief One compressed blob's triage — its format and native pixel extent. */
struct image_probe_t {
    image_format_t format = format_unknown_k;
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
 *      interleaved RGB at its @b [height,width,channels] extent. Returns false on a corrupt bitstream. */
[[nodiscard]] static bool cpu_decode_(image_format_t format, sz::string_view encoded,
                                      pixel_image_span_t destination) noexcept {
    u8_t const *const bytes = reinterpret_cast<u8_t const *>(encoded.data());
    std::size_t const size = encoded.size();
    u32_t const height = static_cast<u32_t>(destination.extent(0)), width = static_cast<u32_t>(destination.extent(1));
    u8_t *const pixels = reinterpret_cast<u8_t *>(destination.data());
    std::size_t const stride = static_cast<std::size_t>(width) * image_channels_k;
    if (format == format_webp_k)
        return WebPDecodeRGBInto(bytes, size, pixels, stride * height, static_cast<int>(stride)) != nullptr;
    if (format == format_png_k) {
        png_image image {};
        image.version = PNG_IMAGE_VERSION;
        if (png_image_begin_read_from_memory(&image, bytes, size) == 0) return false;
        image.format = PNG_FORMAT_RGB;
        bool const ok = png_image_finish_read(&image, nullptr, pixels, static_cast<png_int_32>(stride), nullptr) != 0;
        png_image_free(&image);
        return ok;
    }
    // JPEG the GPU batch could not take (progressive, 4:1:0 / 4:1:1 subsampling) — libjpeg-turbo on the CPU.
    jpeg_decompress_struct info {};
    jpeg_error_mgr error {};
    info.err = jpeg_std_error(&error);
    jpeg_create_decompress(&info);
    jpeg_mem_src(&info, bytes, size);
    if (jpeg_read_header(&info, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&info);
        return false;
    }
    info.out_color_space = JCS_RGB;
    jpeg_start_decompress(&info);
    bool ok = info.output_width == width && info.output_height == height && info.output_components == 3;
    while (ok && info.output_scanline < info.output_height) {
        u8_t *scanline = pixels + static_cast<std::size_t>(info.output_scanline) * stride;
        ok = jpeg_read_scanlines(&info, &scanline, 1) == 1;
    }
    if (ok) jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    return ok;
}

/** @brief One drawn image's decode plan: its format and native size, the size it actually decodes to in the
 *      scratch (scaled for a hardware JPEG, native otherwise), that slot's byte offset, the target patch
 *      factorization, the nvJPEG scale, and whether it routes to the CPU pool. */
struct image_plan_t {
    image_format_t format = format_unknown_k;
    image_extent_t native {};  // the image's full encoded resolution, read from its header
    image_extent_t decoded {}; // what actually lands in the scratch: @c native , or shrunk by the nvJPEG
                               // decode-time scale for a hardware JPEG — the NPP resize's input
    std::size_t scratch_offset = 0;
    patch_grid_t grid {}; // the target patch factorization
    nvjpegScaleFactor_t scale = NVJPEG_SCALE_NONE;
    bool on_cpu = false;
};

nvjpeg_session_t &nvjpeg_session_t::operator=(nvjpeg_session_t &&other) noexcept {
    if (this != &other) {
        this->~nvjpeg_session_t();
        handle_ = std::exchange(other.handle_, nullptr);
        jpeg_state_ = std::exchange(other.jpeg_state_, nullptr);
        hardware_ = other.hardware_;
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
        session.hardware_ = true;
    else if (nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID_DEVICE, nullptr, nullptr, 0, &handle) != NVJPEG_STATUS_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "nvJPEG device backend unavailable");
    if (nvjpegJpegStateCreate(handle, &session.jpeg_state_) != NVJPEG_STATUS_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "nvJPEG state creation failed");
    return result_t {std::move(session)};
}

expected_t nvjpeg_session_t::decode_jpeg_batch(std::span<unsigned char const *const> data,
                                               std::span<std::size_t const> lengths,
                                               std::span<nvjpegImage_t> destinations,
                                               std::span<nvjpegScaleFactor_t const> scales,
                                               gpu_stream_t &stream) noexcept {
    std::size_t const count = data.size();
    if (count == 0) return {};
    cudaStream_t const cuda_stream = reinterpret_cast<cudaStream_t>(stream.get());
    if (nvjpegDecodeBatchedInitialize(handle_, jpeg_state_, static_cast<int>(count), 1, NVJPEG_OUTPUT_RGBI) !=
        NVJPEG_STATUS_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "nvJPEG batch init failed ({} images)", count);
    // The hybrid backend ignores per-image params — a plain batched decode at full scale.
    if (!hardware_)
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
        nvjpegDecodeParamsSetScaleFactor(params[created], scales[created]);
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

/** @brief Triage every drawn blob on the host: sniff its format and native size off a header prefix, choose
 *      its factor pair (written to @p shapes ) and — for a hardware JPEG — its decode scale, and lay out the
 *      native-sized scratch slots. @p tape slices the reservoir (device bytes, host offsets) via @c tape[row] .
 *      Returns the scratch byte total. One header-prefix hop, one sync. */
[[nodiscard]] static expected<std::size_t> triage_images_(bool hardware, column_view_t tape,
                                                          std::span<u64_t const> drawn_rows, u32_t patch_size,
                                                          u32_t num_patches, std::span<image_plan_t> plans,
                                                          image_shape_span_t shapes, gpu_stream_t &stream) noexcept {
    using result_t = expected<std::size_t>;
    std::size_t const images = drawn_rows.size();
    // Build the per-image gather descriptors on the host — a device source pointer and a capped header length
    // straight from each tape slice — then lift every header in one kernel + one D2H, not one copy per image.
    array<gather_slot_t> host_slots;
    array<gather_slot_t, gpu_alloc<gather_slot_t>> device_slots {gpu_alloc<gather_slot_t> {stream.get()}};
    array<u8_t, gpu_alloc<u8_t>> staging {gpu_alloc<u8_t> {stream.get()}};
    array<u8_t, pinned_alloc<u8_t>> probe;
    if (host_slots.try_resize(images) != status_t::success_k ||
        device_slots.try_resize_uninitialized(images) != status_t::success_k ||
        staging.try_resize_uninitialized(images * header_probe_bytes_k) != status_t::success_k ||
        probe.try_resize_uninitialized(images * header_probe_bytes_k) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "header gather staging allocation failed");
    for (std::size_t index = 0; index != images; ++index) {
        auto const blob = tape[drawn_rows[index]];
        host_slots[index] = gather_slot_t {reinterpret_cast<u8_t const *>(blob.data()),
                                           static_cast<u32_t>(std::min(blob.size(), header_probe_bytes_k))};
    }
    if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_slots.data()), host_slots.data(),
                          images * sizeof(gather_slot_t), stream.get()) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "header gather descriptor upload failed");
    if (expected_t launched = launch_kernel(
            reinterpret_cast<void const *>(&gather_headers_kernel_), dim3 {grid_size_for(images, 1u), 1u, 1u},
            dim3 {kernel_block_size_default_k, 1u, 1u}, stream, device_slots.data(), staging.data(),
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
                                      host_slots[index].length};
        auto probed = probe_image_(header);
        if (!probed) return result_t::fail(std::move(probed));
        image_plan_t &plan = plans[index];
        plan.format = probed.value_.format;
        plan.native = probed.value_.extent;
        plan.grid = factor_pair_(plan.native, num_patches);
        shapes[index, 0] = plan.grid.rows;
        shapes[index, 1] = plan.grid.cols;
        // Native-sized scratch slot — the fallback safety net — but a hardware JPEG decodes into it scaled.
        image_extent_t const target {u32_t(plan.grid.cols) * patch_size, u32_t(plan.grid.rows) * patch_size};
        bool const jpeg = plan.format == format_jpeg_k;
        plan.on_cpu = !jpeg;
        plan.scale = jpeg && hardware ? pick_scale_(plan.native, target) : NVJPEG_SCALE_NONE;
        plan.decoded = scaled_extent_(plan.native, plan.scale);
        plan.scratch_offset = scratch_bytes;
        scratch_bytes += static_cast<std::size_t>(plan.native.width) * plan.native.height * image_channels_k;
    }
    return result_t {scratch_bytes};
}

/** @brief Decode the plan's JPEG rows through @p jpeg into their scratch slots. A rejected batch is not fatal:
 *      every JPEG row flips to @c on_cpu at native resolution — which its native-sized slot already fits. */
[[nodiscard]] static expected_t decode_jpeg_plans_(nvjpeg_session_t &jpeg, std::span<image_plan_t> plans, u8_t *scratch,
                                                   column_view_t tape, std::span<u64_t const> drawn_rows,
                                                   gpu_stream_t &stream) noexcept {
    array<unsigned char const *> data;
    array<std::size_t> lengths;
    array<nvjpegImage_t> destinations;
    array<nvjpegScaleFactor_t> scales;
    array<std::size_t> plan_of_slot; // maps a batch slot back to its plan for fallback marking
    for (std::size_t index = 0; index != plans.size(); ++index) {
        image_plan_t const &plan = plans[index];
        if (plan.format != format_jpeg_k) continue;
        auto const blob = tape[drawn_rows[index]];
        nvjpegImage_t destination {};
        destination.channel[0] = scratch + plan.scratch_offset;
        destination.pitch[0] = static_cast<std::size_t>(plan.decoded.width) * image_channels_k;
        if (data.try_push_back(reinterpret_cast<unsigned char const *>(blob.data())) != status_t::success_k ||
            lengths.try_push_back(blob.size()) != status_t::success_k ||
            destinations.try_push_back(destination) != status_t::success_k ||
            scales.try_push_back(plan.scale) != status_t::success_k ||
            plan_of_slot.try_push_back(index) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "JPEG batch table growth failed");
    }
    if (data.empty()) return {};
    if (expected_t decoded = jpeg.decode_jpeg_batch({data.data(), data.size()}, {lengths.data(), lengths.size()},
                                                    {destinations.data(), destinations.size()},
                                                    {scales.data(), scales.size()}, stream);
        !decoded)
        for (std::size_t slot = 0; slot != plan_of_slot.size(); ++slot) {
            image_plan_t &plan = plans[plan_of_slot[slot]];
            plan.on_cpu = true;
            plan.scale = NVJPEG_SCALE_NONE;
            plan.decoded = plan.native;
        }
    return {};
}

/** @brief Decode the plan's @c on_cpu rows (PNG, WebP, and any rejected JPEG) on @p pool : one pinned blob
 *      hop in, a native RGB decode per worker, then one bulk host-to-device copy into each scratch slot. */
[[nodiscard]] static expected_t decode_cpu_plans_(cpu_decode_pool_t &pool, std::span<image_plan_t> plans, u8_t *scratch,
                                                  column_view_t tape, std::span<u64_t const> drawn_rows,
                                                  gpu_stream_t &stream) noexcept {
    struct cpu_task_t {
        std::size_t plan_index, blob_offset, blob_length, pixel_offset, pixel_length;
    };
    array<cpu_task_t> tasks;
    std::size_t blob_bytes = 0, pixel_bytes = 0;
    for (std::size_t index = 0; index != plans.size(); ++index) {
        image_plan_t const &plan = plans[index];
        if (!plan.on_cpu) continue;
        std::size_t const length = tape[drawn_rows[index]].size();
        std::size_t const pixels = static_cast<std::size_t>(plan.native.width) * plan.native.height * image_channels_k;
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
    for (std::size_t slot = 0; slot != tasks.size(); ++slot) {
        cpu_task_t const &task = tasks[slot];
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
                                              {plan.native.height, plan.native.width, image_channels_k}};
        result_base[prong.task] = cpu_decode_(plan.format, encoded, destination) ? 1u : 0u;
    });
    for (std::size_t slot = 0; slot != tasks.size(); ++slot) {
        cpu_task_t const &task = tasks[slot];
        if (!results[slot])
            return expected_t::fail(status_t::invalid_input_k, "CPU decode failed at image {}", task.plan_index);
        if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(scratch + plans[task.plan_index].scratch_offset),
                              pixel_base + task.pixel_offset, task.pixel_length, stream.get()) != CUDA_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "CPU pixel upload failed at image {}", task.plan_index);
    }
    return {};
}

/** @brief Crop each decoded scratch image to its factor-pair aspect and resize it into the batch's run — one
 *      NPP batched call over device-resident descriptors, holding the scratch alive until it retires. */
[[nodiscard]] static expected_t resize_plans_(std::span<image_plan_t const> plans, u8_t *scratch,
                                              pixel_run_span_t pixels, u32_t patch_size,
                                              gpu_stream_t &stream) noexcept {
    std::size_t const images = plans.size();
    auto context = npp_context_(stream);
    if (!context) return expected_t::fail(std::move(context));
    array<NppiImageDescriptor> host_src, host_dst;
    array<NppiResizeBatchROI_Advanced> host_roi;
    if (host_src.try_resize(images) != status_t::success_k || host_dst.try_resize(images) != status_t::success_k ||
        host_roi.try_resize(images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "resize descriptor allocation failed");
    int max_width = 0, max_height = 0;
    std::size_t const run = pixels.extent(1);
    pixel_t *const run_base = pixels.data();
    for (std::size_t index = 0; index != images; ++index) {
        image_plan_t const &plan = plans[index];
        image_extent_t const source = plan.decoded;
        image_extent_t const target {u32_t(plan.grid.cols) * patch_size, u32_t(plan.grid.rows) * patch_size};
        NppiSize const source_size {static_cast<int>(source.width), static_cast<int>(source.height)};
        NppiSize const target_size {static_cast<int>(target.width), static_cast<int>(target.height)};
        host_src[index] = NppiImageDescriptor {scratch + plan.scratch_offset,
                                               static_cast<int>(source.width * image_channels_k), source_size};
        host_dst[index] = NppiImageDescriptor {run_base + index * run,
                                               static_cast<int>(target.width * image_channels_k), target_size};
        host_roi[index] = NppiResizeBatchROI_Advanced {center_crop_(source, target),
                                                       NppiRect {0, 0, target_size.width, target_size.height}};
        max_width = std::max(max_width, source_size.width);
        max_height = std::max(max_height, source_size.height);
    }
    // The three descriptor arrays must be device-resident for the batched call — owning `gpu_alloc` buffers.
    array<NppiImageDescriptor, gpu_alloc<NppiImageDescriptor>> device_src {
        gpu_alloc<NppiImageDescriptor> {stream.get()}};
    array<NppiImageDescriptor, gpu_alloc<NppiImageDescriptor>> device_dst {
        gpu_alloc<NppiImageDescriptor> {stream.get()}};
    array<NppiResizeBatchROI_Advanced, gpu_alloc<NppiResizeBatchROI_Advanced>> device_roi {
        gpu_alloc<NppiResizeBatchROI_Advanced> {stream.get()}};
    if (device_src.try_resize_uninitialized(images) != status_t::success_k ||
        device_dst.try_resize_uninitialized(images) != status_t::success_k ||
        device_roi.try_resize_uninitialized(images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "resize descriptor device allocation failed");
    bool const uploaded = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_src.data()), host_src.data(),
                                            images * sizeof(NppiImageDescriptor), stream.get()) == CUDA_SUCCESS &&
                          cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_dst.data()), host_dst.data(),
                                            images * sizeof(NppiImageDescriptor), stream.get()) == CUDA_SUCCESS &&
                          cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device_roi.data()), host_roi.data(),
                                            images * sizeof(NppiResizeBatchROI_Advanced), stream.get()) == CUDA_SUCCESS;
    if (!uploaded) return expected_t::fail(status_t::cuda_error_k, "resize descriptor upload failed");
    NppStatus const resized = nppiResizeBatch_8u_C3R_Advanced_Ctx(
        max_width, max_height, device_src.data(), device_dst.data(), device_roi.data(),
        static_cast<unsigned int>(images), NPPI_INTER_LANCZOS, context.value_);
    if (resized != NPP_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "batched resize failed (status={})", static_cast<int>(resized));
    // The device descriptors and the scratch are read by the stream-ordered resize; the sync retires it
    // before their owning arrays fall out of scope.
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "resize synchronization failed");
    return {};
}

expected_t ul::decode_images(nvjpeg_session_t &jpeg, cpu_decode_pool_t &pool, column_view_t tape,
                             std::span<u64_t const> drawn_rows, pixel_run_span_t pixels, image_shape_span_t shapes,
                             u32_t patch_size, u32_t num_patches, gpu_stream_t &stream) noexcept {
    std::size_t const images = drawn_rows.size();
    if (images == 0) return {};
    if (pixels.extent(0) < images || shapes.extent(0) < images)
        return expected_t::fail(status_t::shape_unsupported_k, "pixels/shapes cannot hold {} images", images);
    if (patch_size == 0 || num_patches == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "patch_size and num_patches must be nonzero");

    array<image_plan_t> plans;
    if (plans.try_resize(images) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "image plan allocation failed");
    std::span<image_plan_t> const plan_span {plans.data(), images};
    auto scratch_bytes = triage_images_(jpeg.hardware_decode(), tape, drawn_rows, patch_size, num_patches, plan_span,
                                        shapes, stream);
    if (!scratch_bytes) return expected_t::fail(std::move(scratch_bytes));

    // One native-sized device scratch holds every image's decoded RGB, laid out by drawn order — owning, so
    // it frees itself after the resize's sync.
    array<u8_t, gpu_alloc<u8_t>> scratch {gpu_alloc<u8_t> {stream.get()}};
    if (scratch.try_resize_uninitialized(scratch_bytes.value_) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "native pixel scratch allocation failed ({} bytes)",
                                scratch_bytes.value_);
    u8_t *const scratch_base = scratch.data();

    if (expected_t r = decode_jpeg_plans_(jpeg, plan_span, scratch_base, tape, drawn_rows, stream); !r) return r;
    if (expected_t r = decode_cpu_plans_(pool, plan_span, scratch_base, tape, drawn_rows, stream); !r) return r;
    return resize_plans_(plan_span, scratch_base, pixels, patch_size, stream);
}

expected<image_loader_parquet_t> image_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                              sz::string_view path_or_glob, u64_t seed,
                                                              cpu_decode_pool_t &pool) noexcept {
    using result_t = expected<image_loader_parquet_t>;

    // Expand the glob (or a plain path) into the dataset's sorted absolute shard paths — POSIX glob(3).
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    image_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.pool_ = &pool;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;

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
    std::size_t const images_per_batch = batches[0].images.extent(0);
    if (images_per_batch == 0) return expected_t::fail(status_t::shape_unsupported_k, "the batch carries no images");
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
    for (std::size_t slot = 0; slot != total_images; ++slot)
        record_indices[slot] = epoch_permutation(row_key, epoch_t {training_step}, slot, pooled_rows);

    // The blob column re-viewed with its offsets on host: device bytes, host-indexable via @c tape[row] .
    sz::span<u64_t const> const host_offsets {reservoir_offsets.data(), reservoir_offsets.size()};
    column_view_t const tape {column.buffer_, host_offsets};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (images_per_batch > loaded.images.extent(0) || images_per_batch > loaded.image_shapes.extent(0) ||
            images_per_batch > loaded.segment_lengths.extent(0))
            return expected_t::fail(status_t::shape_unsupported_k, "batch {} cannot hold {} images", batch,
                                    images_per_batch);
        u32_t const columns_per_row = static_cast<u32_t>(loaded.tokens.extent(1));
        u32_t const patch_scalars = static_cast<u32_t>(loaded.images.extent(1) / columns_per_row);
        u32_t const spatial = patch_scalars / image_channels_k; // patch_size²
        u32_t patch_size = 1;
        while (patch_size * patch_size < spatial) ++patch_size;
        std::span<u64_t const> const drawn {record_indices.data() + batch * images_per_batch, images_per_batch};

        pixel_run_span_t const pixels {loaded.images.data(), {images_per_batch, loaded.images.extent(1)}};
        image_shape_span_t const shapes {loaded.image_shapes.data(), {images_per_batch, std::size_t {2}}};
        if (expected_t decoded = decode_images(decoder, *pool_, tape, drawn, pixels, shapes, patch_size,
                                               columns_per_row, gpu_stream);
            !decoded)
            return decoded;

        u32r1_span_t const segment_offsets {loaded.segment_offsets.data(), loaded.segment_offsets.extent(0)};
        u32r1_span_t const segment_lengths {loaded.segment_lengths.data(), loaded.segment_lengths.extent(0)};
        if (expected_t segmented = launch_kernel(
                reinterpret_cast<void const *>(&image_segments_kernel_), dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u},
                gpu_stream, segment_offsets, segment_lengths, static_cast<u32_t>(images_per_batch), columns_per_row);
            !segmented)
            return segmented;
    }
    return {};
}
