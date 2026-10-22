/**
 *  @brief The Parquet image loader's bodies and kernels — see the header for the contract.
 *  @file src/loaders/image_loader.cu
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  Its shape mirrors @c tokens_loader.cu : a keyed file pick, one pooled reservoir, a keyed row draw, then a
 *  per-batch decode. The decode is the shared @ref decode_images : JPEG rows through the
 *  @ref nvjpeg_session_t straight off the reservoir tape, PNG/WebP rows on the CPU pool, then one NPP batch
 *  crops each to its factor-pair aspect and resizes it into the batch's shared patch grid.
 */
#include <unistd.h> // `::write`, `STDERR_FILENO` — the triage-drop diagnostic

#include <algorithm> // `std::upper_bound`, `std::find`, `std::min`, `std::max`

#include <nvjpeg.h>                   // vendor-boundary: device-native batched JPEG decode
#include <nppi_geometry_transforms.h> // vendor-boundary: batched crop+resize

#include <forkunion.hpp> // vendor-boundary: the host-CPU decode worker pool

#include "media/planning.hpp" // the budget, factor pair and center crop
#include "media/image.hpp"    // the triage probe and the CPU-fallback decode
#include "loaders/image_loader.cuh"

namespace uc = unum::uform::core;
namespace ul = unum::uform::loaders;
namespace um = unum::uform::media;
using namespace uc;
using namespace ul;
using um::image_channels_k;
using um::image_extent_t;
using um::image_format_t;
using um::patch_grid_t;
using enum um::image_format_t;

namespace fu = ashvardanian::forkunion;

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *  decorrelated under one @c (seed,step,rank) — the same salts the tokens loader uses. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/** Header bytes gathered per drawn blob — enough for the magic sniff and the dimension parse (JPEG
 *  start-of-frame, PNG IHDR, WebP VP8 header) without lifting the whole compressed blob. One device
 *  gather kernel packs these prefixes contiguously so the whole triage reads back in a single D2H. */
inline constexpr std::size_t header_probe_bytes_k = 1u << 12;

/**
 *  @brief Write the batch's segment geometry under the token-cursor model: each drawn image is @b one segment
 *      occupying its own column sub-budget ( @c patch_rows·patch_cols , taken from its host-computed
 *      @p grids factorization), and several images pack into one row.
 *
 *  Thread 0 walks the segments serially — a strict scan — jumping the cursor past a row boundary a segment
 *  would cross (never crossing a row), dropping loudly (zero length) any segment that would overflow the
 *  @c rows·columns grid, and tagging each placed live segment @c image_k (a dropped one PADs to @c text_k ,
 *  length 0). It also lands each live segment's @c (patch_rows,patch_cols) into @p image_shapes on the
 *  device — the descriptors are arena (device) memory, so the loader never writes them from the host.
 *  Segments past @p live_images PAD: zero length, repeated offset, @c text_k .
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
            image_shapes[segment, 1] = grids[segment].columns;
            columns = static_cast<u32_t>(grids[segment].rows) * static_cast<u32_t>(grids[segment].columns);
        }
        // Every image owns half a row from its start.
        u32_t const half_row = media_cells(modality_t::image_k, columns_per_row);
        cursor = (cursor + half_row - 1u) / half_row * half_row;
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

/** One image's header-gather descriptor: where its compressed bytes start in the device tape and how
 *  many of them (capped at the probe size) to lift. */
struct gather_entry_t {

    /** Where the image's compressed bytes start in the device tape. */
    b8_t const *source;

    /** How many of them (capped at the probe size) to lift. */
    u32_t length;
};

/**
 *  @brief Gather each drawn blob's header prefix from the scattered device tape into a contiguous staging
 *      buffer — one block per image, threads stripe the copy — so triage reads them back in one D2H instead
 *      of one small transfer per image.
 *
 *  __grid_size__ @c [min(images,cap),1,1] , grid-stride over images; no dynamic shared memory.
 */
static __global__ void gather_headers_kernel_(gather_entry_t const *entries, b8_t *staging, u32_t stride,
                                              u32_t images) {
    for (u32_t image = blockIdx.x; image < images; image += gridDim.x) {
        gather_entry_t const entry = entries[image];
        b8_t *const destination = staging + static_cast<std::size_t>(image) * stride;
        for (u32_t byte = threadIdx.x; byte < entry.length; byte += blockDim.x) destination[byte] = entry.source[byte];
    }
}

/** The @c unsigned @c char pointer nvJPEG reads compressed @p bytes through. */
[[nodiscard]] static unsigned char const *nvjpeg_bytes_(b8_t const *bytes) noexcept {
    return reinterpret_cast<unsigned char const *>(bytes);
}

/**
 *  @brief Where one plan's pixels are decoded — the single knob that routes the JPEG-device vs CPU-fallback
 *      split.
 *
 *  The invariant lives here and nowhere else: a plan is @c jpeg_device_k iff its format is JPEG
 *  @b and the device batch was accepted; every other plan, PNG or WebP from the start or a JPEG
 *  whose batch nvJPEG rejected, is @c cpu_fallback_k . @ref triage_images_ sets the format-driven
 *  default, and a rejected batch flips its every JPEG entry to @c cpu_fallback_k in
 *  @ref decode_jpeg_plans_ , so the CPU decode fills the very same native-sized scratch region.
 *  @ref decode_cpu_plans_ then decodes exactly the @c cpu_fallback_k plans. A named route, never a
 *  bool.
 */
enum class route_t : u8_t {

    /** Decoded device-native through nvJPEG's batched engine (an accepted JPEG batch). */
    jpeg_device_k = 0,

    /** Decoded on the CPU pool — PNG / WebP, or a JPEG the device batch rejected. */
    cpu_fallback_k = 1,

    /** Unprobeable at triage (unrecognized magic, a header prefix swallowed by oversized
     *  metadata, a truncated blob) — decodes nothing and PADs its segment through the zero grid,
     *  the same containment the overflow drop has. */
    dropped_k = 2,
};

/** One drawn image's decode plan: its format and native size, that image's byte offset into the
 *  scratch, the target patch factorization, and which road it decodes on. */
struct image_plan_t {

    /** The compressed format decided by the blob's magic bytes. */
    image_format_t format = format_unknown_k;

    /** The image's full encoded resolution, read from its header, which it decodes at and the NPP
     *  resize takes as input. */
    image_extent_t native {};

    /** This image's byte offset into the native-sized RGBA scratch. */
    std::size_t scratch_offset = 0;

    /** The target patch factorization the resize/tile scatter honors. */
    patch_grid_t grid {};

    /** Which road the image decodes on — its format-driven default, flipped to @c cpu_fallback_k for a
     *  JPEG whose device batch was rejected (see @ref route_t for the invariant). */
    route_t route = route_t::cpu_fallback_k;
};

nvjpeg_session_t &nvjpeg_session_t::operator=(nvjpeg_session_t &&other) noexcept {
    if (this != &other) {
        this->~nvjpeg_session_t();
        handle_ = std::exchange(other.handle_, nullptr);
        jpeg_state_ = std::exchange(other.jpeg_state_, nullptr);
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
    // The hardware backend takes device-resident bitstreams and is the fast path; fall back to the
    // GPU-hybrid device backend, also device input, where it is absent.
    if (nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE_DEVICE, nullptr, nullptr, 0, &handle) != NVJPEG_STATUS_SUCCESS &&
        nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID_DEVICE, nullptr, nullptr, 0, &handle) != NVJPEG_STATUS_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "nvJPEG device backend unavailable");
    if (nvjpegJpegStateCreate(handle, &session.jpeg_state_) != NVJPEG_STATUS_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "nvJPEG state creation failed");
    return result_t {std::move(session)};
}

expected_t nvjpeg_session_t::decode_jpeg_batch(std::span<jpeg_entry_t const> entries, gpu_stream_t &stream) noexcept {
    std::size_t const count = entries.size();
    if (count == 0) return {};
    // The owning entry table fills nvJPEG's parallel arrays here, at the vendor boundary only.
    st::basic_vector<unsigned char const *> data;
    st::basic_vector<std::size_t> lengths;
    st::basic_vector<nvjpegImage_t> destinations;
    if (st::failed(data.resize(st::assume_uninitialized, count)) ||
        st::failed(lengths.resize(st::assume_uninitialized, count)) ||
        st::failed(destinations.resize(st::assume_uninitialized, count)))
        return expected_t::fail(status_t::bad_alloc_k, "nvJPEG parallel-array allocation failed ({} images)", count);
    for (std::size_t index = 0; index != count; ++index) {
        data[index] = nvjpeg_bytes_(entries[index].bitstream);
        lengths[index] = entries[index].length;
        destinations[index] = entries[index].destination;
    }
    if (nvjpegDecodeBatchedInitialize(handle_, jpeg_state_, static_cast<int>(count), 1, NVJPEG_OUTPUT_RGBI) !=
        NVJPEG_STATUS_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "nvJPEG batch init failed ({} images)", count);
    // The hardware backend writes the destinations from streams not ordered after `stream`, so the
    // stream-ordered destinations, and the work that last held their memory, must retire first.
    if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
        return cuda_failure(result, "nvJPEG destination drain failed ({} images)", count);
    if (nvjpegDecodeBatched(handle_, jpeg_state_, data.data(), lengths.data(), destinations.data(), stream.get()) !=
        NVJPEG_STATUS_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "nvJPEG batched decode failed");
    return {};
}

/** Fill an @c NppStreamContext from the driver — NPP needs the device's launch limits, which come
 *  from @c cuDeviceGetAttribute (no CUDA Runtime), and the driver stream, which is already a @c cudaStream_t . */
[[nodiscard]] static expected<NppStreamContext> npp_context_(gpu_stream_t const &stream) noexcept {
    using result_t = expected<NppStreamContext>;
    NppStreamContext context {};
    context.hStream = stream.get();
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
    if (result != CUDA_SUCCESS) return cuda_failure<NppStreamContext>(result, "NPP device attribute query failed");
    return result_t {context};
}

/** Triage every drawn blob on the host: sniff its format and native size off a header prefix,
 *  derive its column budget (from native resolution, clamped so many images share a row) and its
 *  factor pair (left in @c plan.grid , landed device-side later by @ref image_segments_kernel_ ),
 *  and lay out the native-sized RGBA scratch regions. @p tape slices the reservoir (device bytes,
 *  host offsets) via @c tape[row] . Returns the scratch byte total. One header hop and one sync per
 *  round, a round per JPEG metadata segment past the first window. */
[[nodiscard]] static expected<std::size_t> triage_images_(column_view_t tape, u64r1_view_t drawn_rows, u32_t patch_size,
                                                          u32_t max_columns, std::span<image_plan_t> plans,
                                                          gpu_stream_t &stream) noexcept {
    using result_t = expected<std::size_t>;
    std::size_t const images = drawn_rows.size();
    // Each round lifts one header window per pending image, a device source pointer and a capped
    // length straight from its tape slice, in one kernel and one D2H. A JPEG whose start-of-frame
    // lies past its window stays pending at the marker its walk reached, a round per EXIF or ICC
    // segment.
    st::basic_vector<gather_entry_t> host_entries;
    st::basic_vector<gather_entry_t, gpu_alloc<gather_entry_t>> device_entries {
        gpu_alloc<gather_entry_t> {stream.get()}};
    st::basic_vector<b8_t, gpu_alloc<b8_t>> staging {gpu_alloc<b8_t> {stream.get()}};
    st::basic_vector<b8_t, pinned_alloc<b8_t>> probe;
    st::basic_vector<um::image_probe_t> probes;
    st::basic_vector<std::size_t> pending;
    if (st::failed(host_entries.resize(images)) ||
        st::failed(device_entries.resize(st::assume_uninitialized, images)) ||
        st::failed(staging.resize(st::assume_uninitialized, images * header_probe_bytes_k)) ||
        st::failed(probe.resize(st::assume_uninitialized, images * header_probe_bytes_k)) ||
        st::failed(probes.resize(images)) || st::failed(pending.resize(images)))
        return result_t::fail(status_t::bad_alloc_k, "header gather staging allocation failed");
    for (std::size_t index = 0; index != images; ++index) pending[index] = index;
    // Web-scale shards carry the occasional unprobeable blob — one bad row PADs its segment loudly
    // instead of failing the whole fill, the same containment the overflow drop has.
    auto const warn_dropped = [&](std::size_t index, sz::string_view_t reason) noexcept {
        sz::string_t warning;
        if (format_into(warning, "uform: image row {} dropped at triage: {}\n", drawn_rows[index], reason)) {
            sz::string_view_t const text = warning.view();
            ssize_t const written = ::write(STDERR_FILENO, text.data(), text.size());
            (void)written;
        }
    };
    for (std::size_t count = images; count != 0;) {
        for (std::size_t entry = 0; entry != count; ++entry) {
            std::size_t const index = pending[entry], window_offset = probes[index].resume_offset;
            auto const blob = tape[drawn_rows[index]];
            host_entries[entry] = gather_entry_t {
                blob.data() + window_offset,
                static_cast<u32_t>(std::min(blob.size() - window_offset, header_probe_bytes_k))};
        }
        if (expected_t uploaded = copy_host_to_device<gather_entry_t>(
                std::span<gather_entry_t const> {host_entries}.first(count), std::span<gather_entry_t> {device_entries},
                stream);
            !uploaded)
            return result_t::fail(std::move(uploaded));
        if (expected_t launched = launch_kernel(&gather_headers_kernel_, dim3 {grid_size_for(count, 1u), 1u, 1u},
                                                dim3 {kernel_block_size_default_k, 1u, 1u}, stream,
                                                device_entries.data(), staging.data(),
                                                static_cast<u32_t>(header_probe_bytes_k), static_cast<u32_t>(count));
            !launched)
            return result_t::fail(std::move(launched));
        if (expected_t read = copy_device_to_host<b8_t>(
                std::span<b8_t const> {staging}.first(count * header_probe_bytes_k), std::span<b8_t> {probe}, stream);
            !read)
            return result_t::fail(std::move(read));
        if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
            return result_t::fail(status_t::cuda_error_k, "header gather readback failed");
        std::size_t still_pending = 0;
        for (std::size_t entry = 0; entry != count; ++entry) {
            std::size_t const index = pending[entry], window_offset = probes[index].resume_offset;
            std::span<b8_t const> const window = std::span<b8_t const> {probe}.subspan(entry * header_probe_bytes_k,
                                                                                       host_entries[entry].length);
            expected<um::image_probe_t> probed = um::probe_image(window, window_offset);
            if (!probed) {
                warn_dropped(index, probed.description_.view());
                probes[index] = um::image_probe_t {};
                continue;
            }
            probes[index] = probed.value_;
            if (probed->extent.width != 0) continue;
            // A walk that cannot advance past its window, or runs off the blob, finds no frame.
            if (probed->resume_offset <= window_offset || probed->resume_offset >= tape[drawn_rows[index]].size()) {
                warn_dropped(index, "no JPEG start-of-frame in the blob");
                probes[index] = um::image_probe_t {};
                continue;
            }
            pending[still_pending++] = index;
        }
        count = still_pending;
    }

    std::size_t scratch_bytes = 0;
    for (std::size_t index = 0; index != images; ++index) {
        image_plan_t &plan = plans[index];
        if (probes[index].extent.width == 0) {
            plan = image_plan_t {};
            plan.route = route_t::dropped_k;
            plan.scratch_offset = scratch_bytes;
            continue;
        }
        plan.format = probes[index].format;
        plan.native = probes[index].extent;
        // Every image fills its half row at its own aspect. The grid lands in `image_shapes` on the
        // device (via the segment kernel), never host-written to arena memory.
        plan.grid = um::image_patch_grid(media_cells(modality_t::image_k, max_columns), plan.native);
        // Every JPEG decodes at native size and the resize alone shrinks it: a batched hardware
        // decode at a DCT scale writes past its scaled extent, overwriting the images after it.
        bool const jpeg = plan.format == format_jpeg_k;
        plan.route = jpeg ? route_t::jpeg_device_k : route_t::cpu_fallback_k;
        plan.scratch_offset = scratch_bytes;
        scratch_bytes += static_cast<std::size_t>(plan.native.width) * plan.native.height * image_channels_k;
    }
    return result_t {scratch_bytes};
}

/** One JPEG entry's RGB → RGBA expansion: the nvJPEG output and its RGBA scratch destination. */
struct rgba_expand_entry_t {

    /** The @b [pixels,3] nvJPEG RGB output to widen. */
    nk::tensor_view<u8_t, 2> rgb;

    /** The @b [pixels,4] RGBA scratch destination. */
    nk::tensor_span<u8_t, 2> rgba;
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
        for (std::size_t pixel = threadIdx.x; pixel < entry.rgb.extent(0); pixel += blockDim.x) {
            for (u32_t channel = 0; channel != 3u; ++channel) entry.rgba[pixel, channel] = entry.rgb[pixel, channel];
            entry.rgba[pixel, 3] = 255u;
        }
    }
}

/** Decode the plan's JPEG rows through @p jpeg into a packed RGB scratch, then widen each into its
 *  RGBA scratch region ( @ref rgb_to_rgba_kernel_ , alpha @c=255 ) — nvJPEG emits no alpha. A
 *  rejected batch is not fatal: every JPEG entry flips to @c route_t::cpu_fallback_k , and the CPU
 *  fallback fills the same native-sized scratch region it would have filled. */
[[nodiscard]] static expected_t decode_jpeg_plans_(nvjpeg_session_t &jpeg, std::span<image_plan_t> plans, u8_t *scratch,
                                                   column_view_t tape, u64r1_view_t drawn_rows,
                                                   gpu_stream_t &stream) noexcept {
    constexpr u32_t rgb_channels_k = 3u; // nvJPEG's interleaved-RGB output, widened to RGBA afterward
    st::basic_vector<jpeg_entry_t> entries;
    std::size_t rgb_bytes = 0;
    for (std::size_t index = 0; index != plans.size(); ++index) {
        image_plan_t const &plan = plans[index];
        if (plan.format != format_jpeg_k) continue;
        auto const blob = tape[drawn_rows[index]];
        std::size_t const pixels = static_cast<std::size_t>(plan.native.width) * plan.native.height;
        if (st::failed(entries.push_back(jpeg_entry_t {blob.data(), blob.size(), index, rgb_bytes, nvjpegImage_t {}})))
            return expected_t::fail(status_t::bad_alloc_k, "JPEG batch table growth failed");
        rgb_bytes += pixels * rgb_channels_k;
    }
    if (entries.empty()) return {};

    // The nvJPEG batch lands packed RGB into this temporary; the widen kernel then fills each RGBA scratch region.
    st::basic_vector<u8_t, gpu_alloc<u8_t>> rgb_scratch {gpu_alloc<u8_t> {stream.get()}};
    if (st::failed(rgb_scratch.resize(st::assume_uninitialized, rgb_bytes)))
        return expected_t::fail(status_t::bad_alloc_k, "JPEG RGB scratch allocation failed ({} bytes)", rgb_bytes);
    u8_t *const rgb_base = rgb_scratch.data();
    // Wire each entry's RGB destination into the fresh scratch; the owning entry table then feeds nvJPEG's
    // parallel arrays straight from itself, so the split lives only inside `decode_jpeg_batch`.
    for (std::size_t index = 0; index != entries.size(); ++index) {
        jpeg_entry_t &entry = entries[index];
        image_plan_t const &plan = plans[entry.plan_index];
        entry.destination.channel[0] = rgb_base + entry.rgb_offset;
        entry.destination.pitch[0] = static_cast<std::size_t>(plan.native.width) * rgb_channels_k;
    }
    if (expected_t decoded = jpeg.decode_jpeg_batch(entries, stream); !decoded) {
        // Route every JPEG to the CPU pool, which fills the region triage laid out for it.
        for (std::size_t index = 0; index != entries.size(); ++index)
            plans[entries[index].plan_index].route = route_t::cpu_fallback_k;
        return {};
    }

    // Widen the packed RGB into the RGBA scratch (alpha 255) — one descriptor per decoded entry, uploaded once.
    st::basic_vector<rgba_expand_entry_t> host_expand;
    st::basic_vector<rgba_expand_entry_t, gpu_alloc<rgba_expand_entry_t>> device_expand {
        gpu_alloc<rgba_expand_entry_t> {stream.get()}};
    if (st::failed(host_expand.resize(entries.size())) ||
        st::failed(device_expand.resize(st::assume_uninitialized, entries.size())))
        return expected_t::fail(status_t::bad_alloc_k, "RGBA expansion descriptor allocation failed");
    for (std::size_t index = 0; index != entries.size(); ++index) {
        jpeg_entry_t const &entry = entries[index];
        image_plan_t const &plan = plans[entry.plan_index];
        std::size_t const pixels = static_cast<std::size_t>(plan.native.width) * plan.native.height;
        host_expand[index] = rgba_expand_entry_t {{rgb_base + entry.rgb_offset, pixels, 3},
                                                  {scratch + plan.scratch_offset, pixels, 4}};
    }
    if (expected_t uploaded = copy_host_to_device<rgba_expand_entry_t>(
            std::span<rgba_expand_entry_t const> {host_expand}, std::span<rgba_expand_entry_t> {device_expand}, stream);
        !uploaded)
        return uploaded;
    if (expected_t widened = launch_kernel(&rgb_to_rgba_kernel_, dim3 {grid_size_for(entries.size(), 1u), 1u, 1u},
                                           dim3 {kernel_block_size_default_k, 1u, 1u}, stream, device_expand.data(),
                                           static_cast<u32_t>(entries.size()));
        !widened)
        return widened;
    return {};
}

/** Decode the plan's @c cpu_fallback_k rows (PNG, WebP, and any rejected JPEG) on @p pool : one pinned blob
 *  hop in, an RGBA decode per worker at the plan's @c native extent, then one bulk host-to-device
 *  copy into each scratch region. */
[[nodiscard]] static expected_t decode_cpu_plans_(cpu_decode_pool_t &pool, std::span<image_plan_t> plans, u8_t *scratch,
                                                  column_view_t tape, u64r1_view_t drawn_rows,
                                                  gpu_stream_t &stream) noexcept {
    struct cpu_task_t {

        /** The plan this task decodes. */
        std::size_t plan_index;

        /** The task's byte offset into the pinned blob tape. */
        std::size_t blob_offset;

        /** The compressed blob's byte count. */
        std::size_t blob_length;

        /** The task's byte offset into the pinned decoded-pixel tape. */
        std::size_t pixel_offset;

        /** The decoded RGBA pixel byte count. */
        std::size_t pixel_length;
    };
    st::basic_vector<cpu_task_t> tasks;
    std::size_t blob_bytes = 0, pixel_bytes = 0;
    for (std::size_t index = 0; index != plans.size(); ++index) {
        image_plan_t const &plan = plans[index];
        if (plan.route != route_t::cpu_fallback_k) continue;
        std::size_t const length = tape[drawn_rows[index]].size();
        std::size_t const pixels = static_cast<std::size_t>(plan.native.width) * plan.native.height * image_channels_k;
        if (st::failed(tasks.push_back({index, blob_bytes, length, pixel_bytes, pixels})))
            return expected_t::fail(status_t::bad_alloc_k, "CPU task list growth failed");
        blob_bytes += length;
        pixel_bytes += pixels;
    }
    if (tasks.empty()) return {};

    st::basic_vector<b8_t, pinned_alloc<b8_t>> blobs;
    st::basic_vector<pixel_t, pinned_alloc<pixel_t>> decoded;
    st::basic_vector<u8_t> results;
    if (st::failed(blobs.resize(st::assume_uninitialized, blob_bytes)) ||
        st::failed(decoded.resize(st::assume_uninitialized, pixel_bytes)) || st::failed(results.resize(tasks.size())))
        return expected_t::fail(status_t::bad_alloc_k, "CPU staging allocation failed");
    for (std::size_t index = 0; index != tasks.size(); ++index) {
        cpu_task_t const &task = tasks[index];
        auto const blob = tape[drawn_rows[task.plan_index]];
        if (expected_t copied = copy_device_to_host<b8_t>(
                blob, std::span<b8_t> {blobs}.subspan(task.blob_offset, task.blob_length), stream);
            !copied)
            return copied;
    }
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "CPU blob synchronization failed");

    cpu_task_t const *const task_base = tasks.data();
    image_plan_t const *const plan_base = plans.data();
    b8_t const *const blob_base = blobs.data();
    pixel_t *const pixel_base = decoded.data();
    u8_t *const result_base = results.data();
    pool.for_n(tasks.size(), [=](std::size_t index, fu::thread_in_domain_t) noexcept {
        cpu_task_t const &task = task_base[index];
        image_plan_t const &plan = plan_base[task.plan_index];
        std::span<b8_t const> const encoded {blob_base + task.blob_offset, task.blob_length};
        um::pixel_image_span_t const destination {pixel_base + task.pixel_offset,
                                                  {plan.native.height, plan.native.width, image_channels_k}};
        result_base[index] = um::decode_image_rgba(encoded, plan.format, destination) ? 1u : 0u;
    });
    for (std::size_t index = 0; index != tasks.size(); ++index) {
        cpu_task_t const &task = tasks[index];
        if (!results[index])
            return expected_t::fail(status_t::invalid_input_k, "CPU decode failed at image {}", task.plan_index);
        if (expected_t uploaded = copy_host_to_device(
                nk::tensor_view<pixel_t, 1> {pixel_base + task.pixel_offset, task.pixel_length},
                nk::tensor_span<pixel_t, 1> {scratch + plans[task.plan_index].scratch_offset, task.pixel_length},
                stream);
            !uploaded)
            return uploaded;
    }
    // Hold for the uploads: the pinned decode tape (`blobs` / `decoded`) frees synchronously on return,
    // so it must outlive its in-flight `cuMemcpyHtoDAsync` into the device scratch (a use-after-free otherwise).
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "CPU pixel upload synchronization failed");
    return {};
}

/**
 *  @brief Scatter one image-segment's resized contiguous @b [gr·P,gc·P,C] frame into the batch's shared patch
 *      grid @b [rows,columns,P,P,C].
 *
 *  Patch @c (pr,pc) lands at the flat cell @c offset+pr·gc+pc , i.e.
 *  @c images[row,column_begin+local,:,:,:] where @c row=offset/columns and @c column_begin=offset%columns .
 *  Reads the segment's placement from @p segment_offsets / @p segment_lengths and its resized source base
 *  from @p resized_offsets , so several images share a row and each rides its own column sub-range.
 *
 *  __grid_size__ @c [images,1,1] , one block per segment (a PAD / overflow-dropped segment, length 0, exits
 *  immediately); threads stride the segment's @c budget·P²·C elements. No dynamic shared memory.
 */
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

/** Crop each decoded RGBA scratch image to its factor-pair aspect, Lanczos-resize it into a contiguous
 *  scratch tape (one @c nppiResize per image — NPP's batched variant takes neither single-image batches
 *  nor Lanczos), then scatter each into its column sub-range of the batch's shared patch grid. The
 *  scratch and device-resident offsets free stream-ordered behind it, so nothing waits here.
 *  Budgets vary per image now, so each resized frame's byte size (and its base in the tape) is a
 *  running prefix, not a fixed stride. @p segment_offsets / @p segment_lengths carry the placement the
 *  segment kernel laid down. */
[[nodiscard]] static expected_t resize_plans_(std::span<image_plan_t const> plans, u8_t *scratch,
                                              pixel_patches_span_t pixels, image_shape_view_t image_shapes,
                                              u32r1_view_t segment_offsets, u32r1_view_t segment_lengths,
                                              gpu_stream_t &stream) noexcept {
    std::size_t const images = plans.size();
    u32_t const patch_size = static_cast<u32_t>(pixels.extent(2)); // the square patch side, straight off the grid
    auto context = npp_context_(stream);
    if (!context) return expected_t::fail(std::move(context));

    st::basic_vector<u32_t> host_resized_offsets; // each image's element base in the packed resized tape
    if (st::failed(host_resized_offsets.resize(images)))
        return expected_t::fail(status_t::bad_alloc_k, "resize offset allocation failed");
    // Lay the variable-budget resized frames end to end; a prefix scan gives each one's base in the tape.
    std::size_t resized_elements = 0;
    for (std::size_t index = 0; index != images; ++index) {
        image_plan_t const &plan = plans[index];
        std::size_t const budget = static_cast<std::size_t>(plan.grid.rows) * plan.grid.columns;
        host_resized_offsets[index] = static_cast<u32_t>(resized_elements);
        resized_elements += budget * patch_size * patch_size * image_channels_k;
    }

    // The batched resize lands each image contiguously in a scratch tape; a tile kernel then scatters it into
    // the patch grid, since a resized image's row-major pixels are not the batch's [columns,P,P,C] order.
    st::basic_vector<u8_t, gpu_alloc<u8_t>> resized {gpu_alloc<u8_t> {stream.get()}};
    if (st::failed(resized.resize(st::assume_uninitialized, resized_elements)))
        return expected_t::fail(status_t::bad_alloc_k, "resized pixel scratch allocation failed ({} bytes)",
                                resized_elements);
    pixel_t *const resized_base = resized.data();

    // NPP's batched resize rejects single-image batches (its `nBatchSize` contract demands more than one)
    // and rejects Lanczos outright (`NPP_INTERPOLATION_ERROR`) — per-image `nppiResize` keeps the
    // Lanczos quality the training data wants, needs no device-resident descriptor tables, and the calls
    // serialize on the one stream exactly like a batched launch would.
    for (std::size_t index = 0; index != images; ++index) {
        image_plan_t const &plan = plans[index];
        if (plan.route == route_t::dropped_k) continue; // a triage-dropped blob PADs — nothing to resize
        image_extent_t const source = plan.native;
        image_extent_t const target {u32_t(plan.grid.columns) * patch_size, u32_t(plan.grid.rows) * patch_size};
        NppiSize const source_size {static_cast<int>(source.width), static_cast<int>(source.height)};
        NppiSize const target_size {static_cast<int>(target.width), static_cast<int>(target.height)};
        um::crop_t const crop = um::image_center_crop(source, target);
        NppiRect const source_rect {static_cast<int>(crop.x), static_cast<int>(crop.y), static_cast<int>(crop.width),
                                    static_cast<int>(crop.height)};
        NppStatus const resized_one = nppiResize_8u_C4R_Ctx(
            scratch + plan.scratch_offset, static_cast<int>(source.width * image_channels_k), source_size, source_rect,
            resized_base + host_resized_offsets[index], static_cast<int>(target.width * image_channels_k), target_size,
            NppiRect {0, 0, target_size.width, target_size.height}, NPPI_INTER_LANCZOS, context.value_);
        if (resized_one != NPP_SUCCESS)
            return expected_t::fail(status_t::cuda_error_k, "image resize failed (status={} image={})",
                                    static_cast<int>(resized_one), index);
    }
    // The per-image resized bases must be device-resident for the tile kernel — owning. It reads each
    // segment's column count straight from the device-side `image_shapes`, no second copy here.
    st::basic_vector<u32_t, gpu_alloc<u32_t>> device_resized_offsets {gpu_alloc<u32_t> {stream.get()}};
    if (st::failed(device_resized_offsets.resize(st::assume_uninitialized, images)))
        return expected_t::fail(status_t::bad_alloc_k, "resize offset device allocation failed");
    if (expected_t uploaded = copy_host_to_device<u32_t>(std::span<u32_t const> {host_resized_offsets},
                                                         std::span<u32_t> {device_resized_offsets}, stream);
        !uploaded)
        return uploaded;

    // Scatter every resized image into its column sub-range (one CTA per segment).
    if (expected_t tiled = launch_kernel(&tile_patches_kernel_, dim3 {static_cast<u32_t>(images), 1u, 1u},
                                         dim3 {kernel_block_size_default_k, 1u, 1u}, stream, pixels, resized_base,
                                         device_resized_offsets.data(), image_shapes, segment_offsets, segment_lengths);
        !tiled)
        return tiled;
    return {};
}

expected_t ul::decode_images(nvjpeg_session_t &jpeg, cpu_decode_pool_t &pool, column_view_t tape,
                             u64r1_view_t drawn_rows, std::span<image_batch_shape_t const> representations,
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

    st::basic_vector<image_plan_t> plans;
    if (st::failed(plans.resize(images)))
        return expected_t::fail(status_t::bad_alloc_k, "image plan allocation failed");
    std::span<image_plan_t> const plan_span {plans.data(), images};
    auto scratch_bytes = triage_images_(tape, drawn_rows, patch_size, columns_per_row, plan_span, stream);
    if (!scratch_bytes) return expected_t::fail(std::move(scratch_bytes));

    // Land the host-computed factor pairs device-side and walk the token cursor: it writes image_shapes ,
    // segment_offsets / lengths , and segment_modality , so the resize/tile then read placement off the device.
    st::basic_vector<patch_grid_t> host_grids;
    st::basic_vector<patch_grid_t, gpu_alloc<patch_grid_t>> device_grids {gpu_alloc<patch_grid_t> {stream.get()}};
    if (st::failed(host_grids.resize(images)) || st::failed(device_grids.resize(st::assume_uninitialized, images)))
        return expected_t::fail(status_t::bad_alloc_k, "segment grid staging allocation failed");
    for (std::size_t index = 0; index != images; ++index) host_grids[index] = plans[index].grid;
    if (expected_t uploaded = copy_host_to_device<patch_grid_t>(std::span<patch_grid_t const> {host_grids},
                                                                std::span<patch_grid_t> {device_grids}, stream);
        !uploaded)
        return uploaded;
    if (expected_t segmented = launch_kernel(&image_segments_kernel_, dim3 {1u, 1u, 1u}, dim3 {1u, 1u, 1u}, stream,
                                             device_grids.data(), static_cast<u32_t>(images), columns_per_row,
                                             static_cast<u32_t>(capacity), shapes, segment_offsets, segment_lengths,
                                             segment_modality);
        !segmented)
        return segmented;

    // One native-sized RGBA device scratch holds every image's decoded pixels, laid out by drawn
    // order — owning, so it frees itself after the resize's sync.
    st::basic_vector<u8_t, gpu_alloc<u8_t>> scratch {gpu_alloc<u8_t> {stream.get()}};
    if (st::failed(scratch.resize(st::assume_uninitialized, scratch_bytes.value_)))
        return expected_t::fail(status_t::bad_alloc_k, "decoded pixel scratch allocation failed ({} bytes)",
                                scratch_bytes.value_);
    u8_t *const scratch_base = scratch.data();

    if (expected_t r = decode_jpeg_plans_(jpeg, plan_span, scratch_base, tape, drawn_rows, stream); !r) return r;
    if (expected_t r = decode_cpu_plans_(pool, plan_span, scratch_base, tape, drawn_rows, stream); !r) return r;
    return resize_plans_(plan_span, scratch_base, pixels, shapes, segment_offsets, segment_lengths, stream);
}

expected<image_loader_parquet_t> image_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                              sz::string_view_t path_or_glob, u64_t seed,
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
    if (st::failed(loader.row_prefix_.resize(loader.absolute_paths_.size() + 1)))
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
    st::basic_vector<sz::string_view_t> picked_paths;
    st::basic_vector<std::size_t> picked_shards;
    std::size_t pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < total_images || picked_shards.size() < floor_files) &&
           picked_shards.size() < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        if (std::find(picked_shards.begin(), picked_shards.end(), file) != picked_shards.end()) continue;
        if (st::failed(picked_shards.push_back(std::size_t {file})) ||
            st::failed(picked_paths.push_back(absolute_paths_[file].view())))
            return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
    }

    // Decode the picked files' blob column into one device reservoir; the true pool is the decoded rows.
    auto reservoir_read = parquet_file_t::make(*storage_, picked_paths, {&config.column, 1}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_images)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files cover fewer than the {} images this fill needs",
                                pooled_rows, picked_shards.size(), total_images);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");

    // The reservoir's blob offsets to the host once — every per-row byte bound is then host arithmetic.
    column_view_t const column = reservoir.column(0);
    std::size_t const offsets_count = column.size() + 1u;
    st::basic_vector<u64_t> reservoir_offsets;
    if (st::failed(reservoir_offsets.resize(st::assume_uninitialized, offsets_count)))
        return expected_t::fail(status_t::bad_alloc_k, "reservoir offsets landing failed ({} rows)", pooled_rows);
    if (expected_t landed = copy_device_to_host(column.offsets, std::span<u64_t> {reservoir_offsets}, gpu_stream);
        !landed)
        return landed;
    if (cuStreamSynchronize(gpu_stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "reservoir offsets readback failed");

    // Draw the whole fill's rows on the host — a keyed bijection over the pool, so each batch
    // draws a distinct scatter of rows across the pooled files.
    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    st::basic_vector<u64_t> record_indices;
    if (st::failed(record_indices.resize(st::assume_uninitialized, total_images)))
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_images);
    for (std::size_t index = 0; index != total_images; ++index)
        record_indices[index] = epoch_permutation(row_key, data_epoch_t {training_step}, index, pooled_rows);

    // The blob column re-viewed with its offsets on host: device bytes, host-indexable via `tape[row]`.
    column_view_t const tape {column.payload, std::span<u64_t const> {reservoir_offsets}};
    u64r2_view_t const batch_records {record_indices.data(), unroll, images_per_batch};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (images_per_batch > loaded.image_shapes.extent(0) || images_per_batch > loaded.segment_lengths.extent(0))
            return expected_t::fail(status_t::shape_unsupported_k, "batch {} cannot hold {} image segments", batch,
                                    images_per_batch);
        // The batch's [rows,columns,P,P,C] image tensor names the whole grid directly: extent(2) is the square
        // patch side and extent(4) the RGBA channels — no arithmetic from a flattened run.
        std::span<image_batch_shape_t const> const representations {&image_, 1}; // one representation today
        if (expected_t decoded = decode_images(decoder, *pool_, tape, batch_records[batch, nk::slice], representations,
                                               loaded.images.span(), loaded.image_shapes.span(),
                                               loaded.segment_offsets.span(), loaded.segment_lengths.span(),
                                               loaded.segment_modality.span(), gpu_stream);
            !decoded)
            return decoded;
    }
    return {};
}
