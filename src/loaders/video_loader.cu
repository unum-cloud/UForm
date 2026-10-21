/**
 *  @brief The video loader's bodies: the multi-engine NVDEC decoder plus the draw → stage → demux → decode →
 *      tile batch path — see the header for the contract.
 *  @file src/loaders/video_loader.cu
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *
 *  Two firewalls live here. The NVDEC path ( libnvcuvid via the FFmpeg @c nv-codec-headers dynamic loader ) and
 *  the host demux ( libavformat ) are C APIs that never throw across this boundary: every call is checked and
 *  folded into @ref expected_t . Container parsing is the one host-bound, parallel stage, so it runs over the
 *  borrowed CPU pool — the pool's @b only job here. The NVDEC decode is @b not a thread-per-engine fan-out: a
 *  @b single host thread drives an @b interleaved small-chunk feed across the @c engine_count engines, so decode
 *  work stays submitted to every physical engine at once while the silicon runs it asynchronously — the GPU is the
 *  master, the pool a slave that only demuxes. The engines own @b no streams: the whole GPU tail (the
 *  NV12 → interleaved-RGB conversion and the tube tiling) rides the caller's one stream.
 */
#include <unistd.h> // `::write`, `STDERR_FILENO`

#include <algorithm> // `std::min`, `std::max`, `std::find`, `std::upper_bound`
#include <memory>    // `std::unique_ptr` — the session's pimpl (`nvdec_state_t`)

#include <cuda.h> // vendor-boundary: the Driver API types (`CUstream`, `CUresult`) the NVDEC headers build on

#include <ffnvcodec/dynlink_nvcuvid.h> // vendor-boundary: NVDEC parser / decoder structs + typedefs
#include <ffnvcodec/dynlink_loader.h>  // vendor-boundary: `CuvidFunctions`, `cuvid_load_functions`

#include <forkunion.hpp> // vendor-boundary: `fu::flat_pool_t` — the host demux / decode-drive pool

#include "media/planning.hpp" // `video_tube_grid` — each segment's 3-D factorization
#include "media/video.hpp"    // `demux_video`, `demuxed_t`, `video_codec_t`
#include "loaders/video_loader.cuh"

namespace uc = unum::uform::core;
namespace ul = unum::uform::loaders;
namespace um = unum::uform::media;
using namespace uc;
using namespace ul;
using um::demuxed_t;
using um::video_codec_count_k;
using um::video_codec_t;
using um::video_grid_t;

namespace fu = ashvardanian::forkunion;

/** Purpose salts folded through @ref splitmix64_hash_t so the file draw and the row draw stay
 *  decorrelated under one @c (seed,step,rank) — the same salts the tokens / image / audio loaders use. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

/** One codec's NVDEC parser + decoder within one engine — not a cache: the @b parser is built once and
 *  persists across clips, while the @b decoder is destroyed and recreated from each clip's probed sequence
 *  format on its sequence header (per-job width/height), so a decoder never survives a job. */
struct codec_entry_t {

    /** The persistent NVDEC parser for this codec — built lazily on first use, reused across every clip. */
    CUvideoparser parser = nullptr;

    /** The NVDEC decoder — rebuilt from each clip's sequence header to that job's grid target, never cached. */
    CUvideodecoder decoder = nullptr;
};

/** Per-clip mutable state the three NVDEC parser callbacks reach through the engine's @c current pointer —
 *  the destination frames, the frame budget, and the grid-scaled geometry. */
struct video_decode_context_t {

    /** Base of this decode group's frames in the shared scratch — the NV12 → RGB destination. */
    pixel_t *frames_base = nullptr;

    /** The caller's stream every display-callback conversion enqueues on (the engines own none). */
    gpu_stream_t *stream = nullptr;

    /** The frame budget this job keeps — the display callback stops once this many are decoded. */
    std::size_t frames_requested = 0;

    /** Frames converted and written so far — the running index into @c frames_base . */
    std::size_t frames_decoded = 0;

    /** Displayed frames seen so far — drives the @c temporal_stride subsampling independent of keeps. */
    std::size_t frames_displayed = 0;

    /** Keep every @c temporal_stride -th displayed frame — the representation's explicit temporal density. */
    std::size_t temporal_stride = 1;

    /** The clip's grid target height in pixels: @c patch_rows·P . */
    std::size_t height = 0;

    /** The clip's grid target width in pixels: @c patch_cols·P . */
    std::size_t width = 0;

    /** The resolved codec — selects this engine's @ref codec_entry_t for the parser / decoder. */
    video_codec_t codec = video_codec_t::h264_k;

    /** The first fault the callbacks hit — kept so the drive can mark-and-continue past a bad job. */
    status_t status = status_t::success_k;
};

/** One decode engine, self-sufficient for the NVDEC callbacks: its own per-codec parser / decoder entry
 *  table, plus copies of the shared @c cuvid handle, context lock, and decode-surface ceiling (aliases, not owned
 *  — @ref nvdec_state_t frees the underlying resources). The callbacks reach everything through the engine, so no
 *  back-pointer to the session is needed. An engine owns no stream: the whole GPU tail rides the caller's one
 *  stream, and NVDEC engine parallelism comes from the distinct decoders, not from streams. */
struct nvdec_engine_t {

    /** Shared cuvid function table owned by @ref nvdec_state_t — a copy so the callbacks reach it. */
    CuvidFunctions *cuvid = nullptr;

    /** Shared context lock, aliased — serializes the context-critical cuvid calls across engines. */
    CUvideoctxlock context_lock = nullptr;

    /** Decode-surface ceiling on both sides, capping every decoder this engine builds. */
    unsigned long max_side = 0;

    /** The per-codec parser / decoder entries, indexed by @ref video_codec_t — built lazily per codec. */
    codec_entry_t entries[video_codec_count_k] {};

    /** The job the callbacks are serving right now — set before each feed, cleared when the job retires. */
    video_decode_context_t *current = nullptr;
};

/** The session's owned NVDEC state and sole resource owner: the cuvid loader, the CUDA context, the shared
 *  context lock, and the decode @ref nvdec_engine_t engines. Defined at global scope (the header forward-declares
 *  it there) so this translation unit needs no namespace block. */
struct nvdec_state_t {

    /** The dynamically loaded libnvcuvid function table — the one owner, freed in the destructor. */
    CuvidFunctions *cuvid = nullptr;

    /** The CUDA context the session was made against — captured from the current context at @c make . */
    CUcontext context = nullptr;

    /** The shared context lock serializing the context-critical cuvid calls across every engine. */
    CUvideoctxlock context_lock = nullptr;

    /** The owned decode engines, each with its own per-codec parser / decoder entries. */
    array<nvdec_engine_t> engines {};

    ~nvdec_state_t() noexcept {
        if (cuvid == nullptr) return;
        for (std::size_t index = 0; index != engines.size(); ++index)
            for (codec_entry_t &entry : engines[index].entries) {
                if (entry.parser != nullptr) cuvid->cuvidDestroyVideoParser(entry.parser);
                if (entry.decoder != nullptr) cuvid->cuvidDestroyDecoder(entry.decoder);
            }
        if (context_lock != nullptr) cuvid->cuvidCtxLockDestroy(context_lock);
        CuvidFunctions *functions = cuvid;
        cuvid_free_functions(&functions);
    }
};

/**
 *  @brief NV12 → interleaved-RGB UInt8 conversion of one decoded frame into its @b [height,width,3] destination
 *      slice, BT.601 limited-range coefficients.
 *
 *  __grid_size__ grid-stride over the frame's @c height·width pixels; no dynamic shared memory. NVDEC surfaces are
 *  pitch-strided ( @p pitch >= width ), and the shared-pitch chroma plane sits @c pitch·height bytes past luma; the
 *  destination is tightly packed at @c width·3 .
 */
static __global__ void nv12_to_interleaved_rgb_kernel_(pixel_t const *nv12_luma, pixel_t const *nv12_chroma,
                                                       pixel_t *rgb_out, std::size_t height, std::size_t width,
                                                       std::size_t pitch) {
    std::size_t const total_pixels = height * width;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t pixel_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pixel_index < total_pixels; pixel_index += stride) {
        std::size_t const pixel_row = pixel_index / width;
        std::size_t const pixel_column = pixel_index - pixel_row * width;
        std::size_t const luma_index = pixel_row * pitch + pixel_column;
        std::size_t const chroma_offset = (pixel_row >> 1u) * pitch + (pixel_column >> 1u) * 2u;
        float const luma = 1.164383f * (static_cast<float>(nv12_luma[luma_index]) - 16.0f);
        float const chroma_blue = static_cast<float>(nv12_chroma[chroma_offset + 0u]) - 128.0f;
        float const chroma_red = static_cast<float>(nv12_chroma[chroma_offset + 1u]) - 128.0f;
        auto clamp_to_uint8 = [](float value) -> pixel_t {
            return static_cast<pixel_t>(__float2int_rn(fminf(fmaxf(value, 0.0f), 255.0f)));
        };
        pixel_t *destination = rgb_out + pixel_index * 3u;
        destination[0] = clamp_to_uint8(luma + 1.792741f * chroma_red);
        destination[1] = clamp_to_uint8(luma - 0.213249f * chroma_blue - 0.532909f * chroma_red);
        destination[2] = clamp_to_uint8(luma + 2.112402f * chroma_blue);
    }
}

/** Per-job source bookkeeping for the sub-row tube scatter — one entry per (clip,representation) segment. Its
 *  frames live in the clip's shared decode-group buffer at @c frames_offset (all of the clip's frames at this
 *  segment's @c (pr·P,pc·P) spatial target, temporal_stride 1); the segment subsamples that buffer temporally by
 *  @c frame_stride and lands its tubes at @c (dest_row,dest_column_begin) in the batch grid. A dropped/short
 *  segment carries @c length==0 and contributes nothing. */
struct tile_job_t {

    /** Element offset of this job's decode-group buffer in the shared frame scratch — the group holds every
     *  decoded frame at this segment's @c (pr·P,pc·P) spatial target, shared across the clip's representations. */
    u64_t frames_offset;

    /** The segment's column span = @c temporal_groups·patch_rows·patch_cols ; 0 means dropped/short. */
    u32_t length;

    /** The grid row this segment lands in: @c segment_offset / columns_per_row . */
    u32_t dest_row;

    /** The first grid column of this segment's span: @c segment_offset % columns_per_row . */
    u32_t dest_column_begin;

    /** Temporal subsample of the shared decode-group buffer — the representation's frame density (≥1): the
     *  tile draws compact frame @c k from group frame @c k·frame_stride , so reps of one clip share one decode. */
    u32_t frame_stride;

    /** Spatial patch rows of this segment's factorization — the tile kernel's inner geometry. */
    u16_t patch_rows;

    /** Spatial patch columns of this segment's factorization — the tile kernel's inner geometry. */
    u16_t patch_cols;
};

/** Least @p index in @c [0,count] with @c prefix[index] > @p value — an in-register upper_bound so each
 *  output element finds its job in @c O(log jobs) . Zero-length jobs leave equal consecutive prefixes and so are
 *  never selected as an owner. */
[[nodiscard]] static __device__ u32_t upper_bound_prefix_(u64_t const *prefix, u32_t count, u64_t value) noexcept {
    u32_t low = 0, high = count;
    while (low < high) {
        u32_t const mid = low + ((high - low) >> 1u);
        if (prefix[mid] <= value) low = mid + 1u;
        else high = mid;
    }
    return low;
}

/**
 *  @brief Scatter every segment's tubes, drawn from its clip's shared decode-group buffer @b [frames,pr·P,pc·P,C],
 *      into the batch tube grid @b [rows,columns_per_row,tube_frames,P,P,C] at the segment's
 *      @c [dest_column_begin,dest_column_begin+length) cells of row @c dest_row .
 *
 *  A segment's column is @c (tg·pr+patch_row)·pc+patch_col , and its tube frame @c t is drawn from the group buffer
 *  at frame @c (tg·tube_frames+t)·frame_stride , the representation's temporal subsample. NVDEC already scaled
 *  each frame to the @c (pc·P,pr·P) target, so this is a pure gather; a short clip's missing frames and every
 *  unwritten grid cell stay PAD-zero (the grid is memset ahead of this on the same stream).
 *
 *  __grid_size__ grid-stride over the union of every job's output elements; each element locates its job by an
 *  upper_bound on the contiguous @p element_prefix @b [job_count+1] . No dynamic shared memory.
 */
static __global__ void tile_tubes_kernel_(pixel_t *tiled, pixel_t const *frames, tile_job_t const *jobs,
                                          u64_t const *element_prefix, u32_t job_count, u32_t columns_per_row,
                                          u32_t tube_frames, u32_t patch_size, u32_t channels) {
    u64_t const total_elements = element_prefix[job_count];
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (u64_t element = static_cast<u64_t>(blockIdx.x) * blockDim.x + threadIdx.x; element < total_elements;
         element += stride) {
        // upper_bound returns the first prefix strictly greater than `element`; the owning job is the one before it.
        u32_t const upper = upper_bound_prefix_(element_prefix, job_count, element);
        u32_t const job = upper == 0u ? 0u : upper - 1u;
        tile_job_t const info = jobs[job];
        if (info.length == 0u) continue;
        u64_t local = element - element_prefix[job];
        u32_t const channel = static_cast<u32_t>(local % channels);
        local /= channels;
        u32_t const patch_x = static_cast<u32_t>(local % patch_size);
        local /= patch_size;
        u32_t const patch_y = static_cast<u32_t>(local % patch_size);
        local /= patch_size;
        u32_t const tube_frame = static_cast<u32_t>(local % tube_frames);
        local /= tube_frames;
        u32_t const column = static_cast<u32_t>(local); // in [0,length)
        u32_t const patch_rows = info.patch_rows;
        u32_t const patch_cols = info.patch_cols;
        u32_t const spatial = patch_rows * patch_cols;
        u32_t const temporal_group = column / spatial;
        u32_t const spatial_index = column - temporal_group * spatial;
        u32_t const patch_row = spatial_index / patch_cols;
        u32_t const patch_col = spatial_index - patch_row * patch_cols;
        // The shared decode-group buffer holds every decoded frame (temporal_stride 1); this representation
        // subsamples it by `frame_stride`, so compact frame `tg·tube_frames+t` sits at group frame `·frame_stride`.
        std::size_t const compact_frame = static_cast<std::size_t>(temporal_group) * tube_frames + tube_frame;
        std::size_t const source_frame = compact_frame * info.frame_stride;
        std::size_t const frame_width = static_cast<std::size_t>(patch_cols) * patch_size;
        std::size_t const frame_pixels = static_cast<std::size_t>(patch_rows) * patch_size * frame_width;
        std::size_t const source_y = static_cast<std::size_t>(patch_row) * patch_size + patch_y;
        std::size_t const source_x = static_cast<std::size_t>(patch_col) * patch_size + patch_x;
        std::size_t const source = info.frames_offset +
                                   (source_frame * frame_pixels + source_y * frame_width + source_x) * channels +
                                   channel;
        std::size_t const grid_column = static_cast<std::size_t>(info.dest_column_begin) + column;
        std::size_t const cell = static_cast<std::size_t>(info.dest_row) * columns_per_row + grid_column;
        std::size_t const destination = ((cell * tube_frames + tube_frame) * patch_size + patch_y) * patch_size *
                                            channels +
                                        static_cast<std::size_t>(patch_x) * channels + channel;
        tiled[destination] = frames[source];
    }
}

/** Map a resolved codec onto the NVDEC codec enum. */
[[nodiscard]] static cudaVideoCodec to_nvdec_codec(video_codec_t codec) noexcept {
    switch (codec) {
    case video_codec_t::h264_k: return cudaVideoCodec_H264;
    case video_codec_t::h265_k: return cudaVideoCodec_HEVC;
    case video_codec_t::vp9_k: return cudaVideoCodec_VP9;
    case video_codec_t::av1_k: return cudaVideoCodec_AV1;
    case video_codec_t::mpeg2_k: return cudaVideoCodec_MPEG2;
    case video_codec_t::mpeg4_k: return cudaVideoCodec_MPEG4;
    case video_codec_t::vc1_k: return cudaVideoCodec_VC1;
    }
    return cudaVideoCodec_H264;
}

/** Baseline decode-surface count — covers the reordering depth of the mainstream profiles when a stream's
 *  own @c min_num_decode_surfaces is smaller. */
inline constexpr int default_decode_surface_count_k = 20;

/** Ceiling on a decoder's decode-surface pool — the depth every engine's parser is built to index, so the
 *  sequence callback can honor a stream's @c min_num_decode_surfaces without the parser re-clamping it. Covers
 *  the deepest DPB the mainstream HEVC / AV1 profiles this loader targets ever request. */
inline constexpr int nvdec_max_decode_surfaces_k = 32;

/** The one decoder-create recipe: 8-bit NV12 4:2:0, scaling each coded frame to the clip's grid target. */
[[nodiscard]] static CUVIDDECODECREATEINFO make_decoder_create_info(
    cudaVideoCodec codec, cudaVideoChromaFormat chroma, CUvideoctxlock context_lock, unsigned long coded_width,
    unsigned long coded_height, unsigned long target_width, unsigned long target_height, unsigned long max_width,
    unsigned long max_height) noexcept {
    CUVIDDECODECREATEINFO create_info {};
    create_info.CodecType = codec;
    create_info.ChromaFormat = chroma;
    create_info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    create_info.bitDepthMinus8 = 0;
    create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    create_info.ulNumOutputSurfaces = 2u;
    create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    create_info.ulNumDecodeSurfaces = default_decode_surface_count_k;
    create_info.vidLock = context_lock;
    create_info.ulWidth = coded_width;
    create_info.ulHeight = coded_height;
    create_info.ulMaxWidth = max_width;
    create_info.ulMaxHeight = max_height;
    create_info.ulTargetWidth = target_width;
    create_info.ulTargetHeight = target_height;
    return create_info;
}

/** NVDEC sequence callback: reject non-8-bit / non-4:2:0 streams, else (re)create this engine's codec-entry
 *  decoder scaling to the clip's grid target. */
extern "C" int CUDAAPI on_video_sequence(void *user_data, CUVIDEOFORMAT *format) noexcept {
    auto *engine = static_cast<nvdec_engine_t *>(user_data);
    if (engine == nullptr || engine->current == nullptr || format == nullptr) return 0;
    video_decode_context_t *context = engine->current;
    if (format->bit_depth_luma_minus8 != 0 || format->bit_depth_chroma_minus8 != 0 ||
        format->chroma_format != cudaVideoChromaFormat_420) {
        context->status = status_t::shape_unsupported_k;
        return 0;
    }
    codec_entry_t &entry = engine->entries[static_cast<std::size_t>(context->codec)];
    if (entry.decoder != nullptr) {
        engine->cuvid->cuvidDestroyDecoder(entry.decoder);
        entry.decoder = nullptr;
    }
    unsigned long decoder_max_width = std::max<unsigned long>(engine->max_side, format->coded_width);
    unsigned long decoder_max_height = std::max<unsigned long>(engine->max_side, format->coded_height);
    // The engine's bound follows the grid's width, which can pass what NVDEC decodes, and a decoder
    // asked for more than its codec's maximum fails to create.
    CUVIDDECODECAPS caps {};
    caps.eCodecType = to_nvdec_codec(context->codec);
    caps.eChromaFormat = format->chroma_format;
    caps.nBitDepthMinus8 = format->bit_depth_luma_minus8;
    if (engine->cuvid->cuvidGetDecoderCaps(&caps) != CUDA_SUCCESS || !caps.bIsSupported ||
        format->coded_width > caps.nMaxWidth || format->coded_height > caps.nMaxHeight) {
        context->status = status_t::shape_unsupported_k;
        return 0;
    }
    decoder_max_width = std::min<unsigned long>(decoder_max_width, caps.nMaxWidth);
    decoder_max_height = std::min<unsigned long>(decoder_max_height, caps.nMaxHeight);
    // Honor the stream's minimum DPB depth — high-reference HEVC / AV1 needs more surfaces than the baseline, and a
    // decoder built below `min_num_decode_surfaces` faults. Cap at the ceiling the engine's parser was built to index
    // so decoder and parser never disagree.
    int decode_surfaces = std::max<int>(default_decode_surface_count_k,
                                        static_cast<int>(format->min_num_decode_surfaces));
    if (decode_surfaces > nvdec_max_decode_surfaces_k) decode_surfaces = nvdec_max_decode_surfaces_k;
    CUVIDDECODECREATEINFO create_info = make_decoder_create_info(
        to_nvdec_codec(context->codec), format->chroma_format, engine->context_lock, format->coded_width,
        format->coded_height, static_cast<unsigned long>(context->width), static_cast<unsigned long>(context->height),
        decoder_max_width, decoder_max_height);
    create_info.ulNumDecodeSurfaces = static_cast<unsigned long>(decode_surfaces);
    if (engine->cuvid->cuvidCreateDecoder(&entry.decoder, &create_info) != CUDA_SUCCESS) {
        entry.decoder = nullptr;
        context->status = status_t::cuda_error_k;
        return 0;
    }
    return decode_surfaces;
}

/** NVDEC decode callback: schedule the picture unless the budget is already met. */
extern "C" int CUDAAPI on_picture_decode(void *user_data, CUVIDPICPARAMS *params) noexcept {
    auto *engine = static_cast<nvdec_engine_t *>(user_data);
    if (engine == nullptr || engine->current == nullptr) return 0;
    codec_entry_t &entry = engine->entries[static_cast<std::size_t>(engine->current->codec)];
    if (entry.decoder == nullptr) return 0;
    if (engine->cuvid->cuvidDecodePicture(entry.decoder, params) != CUDA_SUCCESS) {
        engine->current->status = status_t::cuda_error_k;
        return 0;
    }
    return 1;
}

/** NVDEC display callback: map the NV12 surface, launch the interleaved-RGB conversion of the sub-sampled
 *  frame onto the engine's stream, unmap. */
extern "C" int CUDAAPI on_picture_display(void *user_data, CUVIDPARSERDISPINFO *display_info) noexcept {
    auto *engine = static_cast<nvdec_engine_t *>(user_data);
    if (engine == nullptr || engine->current == nullptr) return 0;
    video_decode_context_t *context = engine->current;
    codec_entry_t &entry = engine->entries[static_cast<std::size_t>(context->codec)];
    if (entry.decoder == nullptr) return 0;
    if (context->frames_decoded >= context->frames_requested) return 1;
    std::size_t const this_display = context->frames_displayed++;
    if (context->temporal_stride > 1u && (this_display % context->temporal_stride) != 0u) return 1;

    CUVIDPROCPARAMS proc_params {};
    proc_params.progressive_frame = display_info->progressive_frame;
    proc_params.second_field = display_info->repeat_first_field + 1;
    proc_params.top_field_first = display_info->top_field_first;
    proc_params.unpaired_field = display_info->repeat_first_field < 0;
    proc_params.output_stream = context->stream->get();

    unsigned long long nv12_pointer = 0;
    unsigned int nv12_pitch = 0;
    if (engine->cuvid->cuvidMapVideoFrame(entry.decoder, display_info->picture_index, &nv12_pointer, &nv12_pitch,
                                          &proc_params) != CUDA_SUCCESS) {
        context->status = status_t::cuda_error_k;
        return 0;
    }
    pixel_t const *nv12_luma = device_pointer<pixel_t const>(nv12_pointer);
    pixel_t const *nv12_chroma = nv12_luma + static_cast<std::size_t>(nv12_pitch) * context->height;
    std::size_t const per_frame_pixels = context->height * context->width;
    pixel_t *frame_destination = context->frames_base + context->frames_decoded * per_frame_pixels * 3u;
    expected_t launched = launch_kernel(
        &nv12_to_interleaved_rgb_kernel_, dim3 {grid_size_for(per_frame_pixels), 1u, 1u},
        dim3 {kernel_block_size_default_k, 1u, 1u}, *context->stream, nv12_luma, nv12_chroma, frame_destination,
        context->height, context->width, static_cast<std::size_t>(nv12_pitch));
    // The RGB read and the mapped surface both live on `context->stream`; unmap only returns cuvid's host-side
    // surface handle, whose device reuse is re-enqueued behind this kernel on the SAME stream. Serial per-engine
    // stream order is therefore the fence against surface recycling — do NOT move this launch to another stream.
    engine->cuvid->cuvidUnmapVideoFrame(entry.decoder, nv12_pointer);
    if (!launched) {
        context->status = launched.status_;
        return 0;
    }
    ++context->frames_decoded;
    return 1;
}

/** The @c unsigned @c char payload pointer NVDEC's parser reads @p bytes through. */
[[nodiscard]] static unsigned char const *cuvid_payload_(b8_t const *bytes) noexcept {
    return reinterpret_cast<unsigned char const *>(bytes);
}

/** Feed an AV1 low-overhead OBU stream to the parser one temporal unit at a time — submitting the whole
 *  clip at once faults inside libnvcuvid after the first frame. */
[[nodiscard]] static CUresult feed_av1_temporal_units(CuvidFunctions *cuvid, CUvideoparser parser,
                                                      std::span<b8_t const> stream) noexcept {
    auto feed_unit = [&](std::size_t begin, std::size_t end, bool last) -> CUresult {
        if (end <= begin) return CUDA_SUCCESS;
        CUVIDSOURCEDATAPACKET packet {};
        packet.payload = cuvid_payload_(stream.data() + begin);
        packet.payload_size = static_cast<unsigned long>(end - begin);
        packet.flags = last ? CUVID_PKT_ENDOFSTREAM : 0u;
        return cuvid->cuvidParseVideoData(parser, &packet);
    };
    std::size_t temporal_unit_start = 0;
    std::size_t offset = 0;
    while (offset < stream.size()) {
        u8_t const obu_header = std::to_integer<u8_t>(stream[offset]);
        u8_t const obu_type = (obu_header >> 3u) & 0x0Fu;
        bool const has_extension = ((obu_header >> 2u) & 0x01u) != 0u;
        bool const has_size_field = ((obu_header >> 1u) & 0x01u) != 0u;
        std::size_t cursor = offset + 1u + (has_extension ? 1u : 0u);
        if (!has_size_field || cursor > stream.size()) return CUDA_ERROR_INVALID_VALUE;
        std::size_t obu_size = 0;
        std::size_t shift = 0;
        bool size_complete = false;
        while (cursor < stream.size() && shift < 56u) {
            u8_t const leb_byte = std::to_integer<u8_t>(stream[cursor++]);
            obu_size |= static_cast<std::size_t>(leb_byte & 0x7Fu) << shift;
            if ((leb_byte & 0x80u) == 0u) {
                size_complete = true;
                break;
            }
            shift += 7u;
        }
        std::size_t const obu_end = cursor + obu_size;
        if (!size_complete || obu_end > stream.size()) return CUDA_ERROR_INVALID_VALUE;
        if (obu_type == 2u && offset != temporal_unit_start) {
            if (CUresult status = feed_unit(temporal_unit_start, offset, false); status != CUDA_SUCCESS) return status;
            temporal_unit_start = offset;
        }
        offset = obu_end;
    }
    return feed_unit(temporal_unit_start, stream.size(), true);
}

#pragma region Host Container Demux

/** Build the parser for @p codec 's entry in @p engine on first use — @c pUserData is the engine so the
 *  callbacks reach its entries and in-flight context. */
[[nodiscard]] static expected_t ensure_parser(nvdec_engine_t &engine, video_codec_t codec) noexcept {
    codec_entry_t &entry = engine.entries[static_cast<std::size_t>(codec)];
    if (entry.parser != nullptr) return {};
    CUVIDPARSERPARAMS parser_params {};
    parser_params.CodecType = to_nvdec_codec(codec);
    parser_params.ulMaxNumDecodeSurfaces = nvdec_max_decode_surfaces_k;
    parser_params.ulMaxDisplayDelay = 0;
    parser_params.pUserData = &engine;
    parser_params.pfnSequenceCallback = &on_video_sequence;
    parser_params.pfnDecodePicture = &on_picture_decode;
    parser_params.pfnDisplayPicture = &on_picture_display;
    if (engine.cuvid->cuvidCreateVideoParser(&entry.parser, &parser_params) != CUDA_SUCCESS) {
        entry.parser = nullptr;
        return expected_t::fail(status_t::cuda_error_k, "NVDEC parser creation failed (codec {})",
                                static_cast<int>(codec));
    }
    return {};
}

/** The feed granularity of the interleaved decode drive — small enough that no single parse call monopolizes
 *  the one host thread, so decode work stays submitted across every engine at once. Measured on a 7-engine
 *  H100: a whole-clip-per-call feed uses one engine and runs slower than a single decoder, while 64 KB chunks
 *  reach ~6x — near the engine count. */
inline constexpr std::size_t nvdec_feed_chunk_bytes_k = 64u * 1024u;

nvdec_session_t::nvdec_session_t() noexcept = default;
nvdec_session_t::nvdec_session_t(std::unique_ptr<nvdec_state_t> state) noexcept : state_(std::move(state)) {}
nvdec_session_t::nvdec_session_t(nvdec_session_t &&) noexcept = default;
nvdec_session_t &nvdec_session_t::operator=(nvdec_session_t &&) noexcept = default;
nvdec_session_t::~nvdec_session_t() noexcept = default;

expected<nvdec_session_t> nvdec_session_t::make(nvdec_config_t config) noexcept {
    using result_t = expected<nvdec_session_t>;
    if (config.max_side == 0) return result_t::fail(status_t::invalid_input_k, "need max_side>0");
    // There is no runtime query for a GPU's NVDEC engine count (NVML exposes only decoder utilization), so the
    // caller — which knows the deployment GPU — states how many concurrent decode engines to run. Each engine holds a
    // real decoder with a surface pool, so this is not free to over-provision; it is not defaulted.
    if (config.engine_count == 0)
        return result_t::fail(status_t::invalid_input_k,
                              "need engine_count>0 — the concurrent decode-engine count; set it to the target GPU's " //
                              "NVDEC engine count (no runtime query for it exists)");
    u32_t const engine_count = config.engine_count;

    auto state = std::unique_ptr<nvdec_state_t> {new (std::nothrow) nvdec_state_t {}};
    if (state == nullptr) return result_t::fail(status_t::bad_alloc_k, "NVDEC session state allocation failed");

    if (cuCtxGetCurrent(&state->context) != CUDA_SUCCESS || state->context == nullptr)
        return result_t::fail(status_t::invalid_input_k, "no current CUDA context for the NVDEC session");
    if (cuvid_load_functions(&state->cuvid, nullptr) != 0 || state->cuvid == nullptr)
        return result_t::fail(status_t::cuda_error_k, "libnvcuvid dynamic load failed");
    if (state->cuvid->cuvidCtxLockCreate(&state->context_lock, state->context) != CUDA_SUCCESS)
        return result_t::fail(status_t::cuda_error_k, "NVDEC context lock creation failed");
    if (state->engines.resize(engine_count) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "NVDEC engine table allocation failed ({} engines)", engine_count);
    // Alias the shared handles into each engine so the parser callbacks reach them without a back-pointer; the engines
    // own only their per-codec decoders (freed by ~nvdec_state_t). No streams — decode rides the caller's stream.
    for (std::size_t engine_index = 0; engine_index != engine_count; ++engine_index) {
        nvdec_engine_t &engine = state->engines[engine_index];
        engine.cuvid = state->cuvid;
        engine.context_lock = state->context_lock;
        engine.max_side = static_cast<unsigned long>(config.max_side);
    }
    return result_t {nvdec_session_t {std::move(state)}};
}

/** The batch grid's decode geometry, read once from the @b [rows,columns,tube_frames,P,P,C] videos span. */
struct video_grid_geometry_t {

    /** Grid rows — the number of tube rows the segments pack across. */
    u32_t rows = 0;

    /** Grid columns per row — a row's cell budget the token cursor never lets a segment cross. */
    u32_t columns = 0;

    /** Tube depth in frames — the temporal extent of one grid cell. */
    u32_t tube_frames = 0;

    /** Patch edge P in pixels — a cell's spatial P×P footprint. */
    u32_t patch_size = 0;

    /** Channel count C — fixed at 3, the NV12 → RGB conversion's output. */
    u32_t channels = 0;

    /** Elements per grid cell — @c tube_frames·P·P·C , the compact-frame and grid stride. */
    std::size_t per_cell = 0;
};

/** Read and validate the batch grid geometry from @p videos — the NV12 → RGB path emits exactly 3 channels and
 *  the grid needs a nonzero tube depth, both hard failures. */
[[nodiscard]] static expected<video_grid_geometry_t> decode_geometry_(pixel_tubes_span_t videos) noexcept {
    using result_t = expected<video_grid_geometry_t>;
    u32_t const channels = static_cast<u32_t>(videos.extent(5));
    if (channels != 3u)
        return result_t::fail(status_t::shape_unsupported_k,
                              "the NV12 → RGB conversion emits 3 channels, but videos wants {}", channels);
    u32_t const tube_frames = static_cast<u32_t>(videos.extent(2));
    if (tube_frames == 0)
        return result_t::fail(status_t::shape_unsupported_k, "the video grid needs a nonzero tube depth");
    u32_t const patch_size = static_cast<u32_t>(videos.extent(3));
    video_grid_geometry_t geometry;
    geometry.rows = static_cast<u32_t>(videos.extent(0));
    geometry.columns = static_cast<u32_t>(videos.extent(1));
    geometry.tube_frames = tube_frames;
    geometry.patch_size = patch_size;
    geometry.channels = channels;
    geometry.per_cell = static_cast<std::size_t>(tube_frames) * patch_size * patch_size * channels;
    return result_t {std::move(geometry)};
}

/** One decode unit: a clip decoded once for a distinct spatial target @c (patch_rows,patch_cols) , shared by
 *  every representation of that clip whose factorization lands on that target (they differ only in
 *  @c frame_stride , a temporal subsample the tile applies). The group owns a contiguous frame buffer holding
 *  @c frames_to_decode frames at @c (patch_rows·P,patch_cols·P,C) , temporal_stride 1; the drive decodes the clip's
 *  elementary stream into it once, and each of the clip's tile jobs at this target reads back through it. */
struct decode_group_t {

    /** Index of the clip whose one elementary stream this group decodes — into the demux result table. */
    u32_t clip;

    /** Element offset of this group's frame buffer in the shared frame scratch — the tile jobs' @c frames_offset
     *  and the frame-scratch accumulator's running total. */
    u64_t frames_offset;

    /** Frames to decode / buffer for this group — @c max over its representations of
     *  @c temporal_groups·tube_frames·frame_stride , the largest frame index any sharing tile job reads (plus one),
     *  so the shared buffer never under-covers a strided subsample. A short stream leaves the tail PAD-zero. */
    u32_t frames_to_decode;

    /** Spatial patch rows of this group's target — NVDEC scales each frame to @c patch_rows·P tall. */
    u16_t patch_rows;

    /** Spatial patch columns of this group's target — NVDEC scales each frame to @c patch_cols·P wide. */
    u16_t patch_cols;
};

/** The host repack scan's whole output: the per-segment grid geometry (offsets/lengths/modality), the per-job
 *  tile placement and spatial factors, the output-element prefix, and the per-clip decode groups — everything the
 *  device uploads, the decode drive, and the tile stage read. Owns its scratch so one scan fills it and the later
 *  stages borrow spans into it. */
struct cursor_plan_t {

    /** Segment start columns @b [max_segments+1] — element @c max_segments is the packed total. */
    array<u32_t> offsets {};

    /** Per-segment column span @b [max_segments] ; 0 for a PAD, dropped, or overflowing segment. */
    array<u32_t> lengths {};

    /** Per-segment modality @b [max_segments] — @c video_k for a live segment, else @c text_k PAD. */
    array<modality_t> modality {};

    /** Per-job (tg,pr,pc) factor triple @b [job_count·3] — the device video-shape descriptor. */
    array<u16_t> shapes {};

    /** Per-job tile placement + spatial factors @b [job_count] — the tile kernel's source table; each job's
     *  @c frames_offset points at its clip's decode group and its @c frame_stride subsamples that group. */
    array<tile_job_t> jobs {};

    /** Output-element prefix @b [job_count+1] — the tile kernel's grid-stride owner search; independent of the
     *  shared frame buffers, whose sizing lives on the decode groups. */
    array<u64_t> prefix {};

    /** The distinct-spatial decode groups @b [group_count] — one clip decode per @c (clip,pr,pc) . */
    array<decode_group_t> groups {};

    /** Per-group NVDEC callback context @b [group_count] — the drive publishes and the callbacks fill it. */
    array<video_decode_context_t> contexts {};

    /** Total shared-frame-buffer bytes across every decode group — the frame-scratch size. */
    u64_t frame_prefix = 0;
};

/**
 *  @brief Stage 2 — the serial token-cursor repack scan.
 *
 *  Walk every segment: derive each live (clip,representation) segment's footprint from the clip's decoded native
 *  resolution, clamped to a row ( @ref um::video_tube_grid ), jump the cursor past a row boundary it would cross,
 *  drop-with-diagnostic a segment once the grid is full, and record the segment geometry plus the per-job tile
 *  placement. Each live job binds to its clip's @c (clip,pr,pc) decode group ( @ref decode_group_t ) — one NVDEC
 *  decode per distinct spatial target, shared across the clip's representations, which differ only in
 *  @c frame_stride and subsample the shared frames in the tile. A second pass lays the groups out in the shared
 *  frame scratch and a third resolves each job's @c frames_offset to its group's buffer. Segments past the drawn
 *  ones PAD. Returns the fully populated @ref cursor_plan_t .
 */
[[nodiscard]] static expected<cursor_plan_t> build_token_cursor_(video_grid_geometry_t const &geometry,
                                                                 std::span<video_batch_shape_t const> representations,
                                                                 std::span<demuxed_t const> demuxed,
                                                                 std::size_t max_segments) noexcept {
    using result_t = expected<cursor_plan_t>;
    u32_t const reps_count = static_cast<u32_t>(representations.size());
    std::size_t const job_count = demuxed.size() * reps_count;
    cursor_plan_t plan;
    array<u32_t> job_group; // per-job index into `plan.groups`, meaningful only where the job's length is nonzero
    if (plan.offsets.resize(max_segments + 1u) != status_t::success_k ||
        plan.lengths.resize(max_segments) != status_t::success_k ||
        plan.modality.resize(max_segments) != status_t::success_k ||
        plan.shapes.resize(job_count * 3u) != status_t::success_k ||
        plan.jobs.resize(job_count) != status_t::success_k ||
        plan.prefix.resize(job_count + 1u) != status_t::success_k || job_group.resize(job_count) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "video geometry scratch allocation failed ({} segments)",
                              max_segments);
    nk::tensor_span<u16_t, 2> const shapes {plan.shapes.data(), job_count, 3};

    u32_t const columns = geometry.columns;
    u32_t const capacity = geometry.rows * geometry.columns;
    u32_t const patch_size = geometry.patch_size;
    u32_t const tube_frames = geometry.tube_frames;
    std::size_t const per_cell = geometry.per_cell;
    u32_t cursor = 0;
    u64_t output_elements = 0; // the tile's output-element prefix — independent of the shared frame buffers
    // Pass 1: place each segment on the grid, record its tile job, and bind a live job to its clip's decode group.
    for (std::size_t segment = 0; segment != max_segments; ++segment) {
        u32_t length = 0;
        modality_t modality = modality_t::text_k;
        if (segment < job_count) {
            // A segment/job is (clip, rep): clip = segment / reps_count, rep = segment % reps_count.
            std::size_t const clip = segment / reps_count;
            video_batch_shape_t const &rep = representations[segment % reps_count];
            demuxed_t const &decoded = demuxed[clip];
            u32_t const stride = rep.frame_stride == 0u ? 1u : rep.frame_stride;
            video_grid_t const grid = um::video_tube_grid(decoded.frames, decoded.extent, rep, columns);
            length = static_cast<u32_t>(grid.temporal_groups) * grid.patch_rows * grid.patch_columns;
            modality = modality_t::video_k;
            // A segment never crosses its row: jump the cursor to the next row when it would. `video_tube_grid`
            // already clamped `length` to one row, so a drop here means the grid ran out of rows, not that one
            // clip was too big.
            u32_t const room_in_row = columns - cursor % columns;
            if (length > room_in_row) cursor += room_in_row;
            if (cursor + length > capacity) {
                sz::string_t line;
                if (format_into(line, "uform: loaded video batch full at segment {} — dropped\n", segment)) {
                    ssize_t const written = ::write(STDERR_FILENO, line.data(), line.size());
                    (void)written;
                }
                length = 0;
            }
            shapes[segment, 0] = grid.temporal_groups;
            shapes[segment, 1] = grid.patch_rows;
            shapes[segment, 2] = grid.patch_columns;
            tile_job_t job;
            job.frames_offset = 0; // resolved in pass 3, once the group buffers have addresses
            job.length = length;
            job.dest_row = cursor / columns;
            job.dest_column_begin = cursor % columns;
            job.frame_stride = stride;
            job.patch_rows = grid.patch_rows;
            job.patch_cols = grid.patch_columns;
            plan.jobs[segment] = job;
            plan.prefix[segment] = output_elements;
            output_elements += static_cast<u64_t>(length) * per_cell;
            // Bind a live job to its clip's (pr,pc) decode group — reuse a matching one or append. The group's frame
            // budget is the largest strided reach across its jobs: (tg·tube_frames)·frame_stride frames, one past the
            // highest frame any sharing tile job reads, so a strided subsample never over-runs the shared buffer.
            u32_t group_index = 0;
            if (length != 0u) {
                u32_t const reach = static_cast<u32_t>(grid.temporal_groups) * tube_frames * stride;
                // Segments run clip by clip, so only the table's tail can hold this clip's groups.
                std::size_t found = plan.groups.size();
                u32_t const clip_index = static_cast<u32_t>(clip);
                for (std::size_t g = plan.groups.size(); g != 0 && plan.groups[g - 1].clip == clip_index; --g)
                    if (plan.groups[g - 1].patch_rows == grid.patch_rows &&
                        plan.groups[g - 1].patch_cols == grid.patch_columns) {
                        found = g - 1;
                        break;
                    }
                if (found == plan.groups.size()) {
                    decode_group_t new_group;
                    new_group.clip = clip_index;
                    new_group.frames_offset = 0; // assigned in pass 2
                    new_group.frames_to_decode = reach;
                    new_group.patch_rows = grid.patch_rows;
                    new_group.patch_cols = grid.patch_columns;
                    if (plan.groups.push_back(new_group) != status_t::success_k)
                        return result_t::fail(status_t::bad_alloc_k, "video decode-group table allocation failed");
                }
                else {
                    decode_group_t &existing = plan.groups[found];
                    existing.frames_to_decode = std::max(existing.frames_to_decode, reach);
                }
                group_index = static_cast<u32_t>(found);
            }
            job_group[segment] = group_index;
        }
        plan.offsets[segment] = cursor;
        plan.lengths[segment] = length;
        plan.modality[segment] = modality;
        cursor += length;
    }
    plan.offsets[max_segments] = cursor;
    plan.prefix[job_count] = output_elements;

    // Pass 2: lay the decode groups out contiguously in the shared frame scratch, each sized to hold its frame budget
    // at its (pr·P,pc·P,C) spatial target; the running total is the whole scratch size.
    u64_t frame_prefix = 0;
    for (std::size_t g = 0; g != plan.groups.size(); ++g) {
        decode_group_t &group = plan.groups[g];
        group.frames_offset = frame_prefix;
        std::size_t const per_frame = static_cast<std::size_t>(group.patch_rows) * patch_size *
                                      static_cast<std::size_t>(group.patch_cols) * patch_size * geometry.channels;
        frame_prefix += static_cast<u64_t>(group.frames_to_decode) * per_frame;
    }
    plan.frame_prefix = frame_prefix;

    // Pass 3: the group buffers now have addresses — point each live job at its group, and size the per-group context
    // table the drive publishes and the callbacks fill.
    for (std::size_t job = 0; job != job_count; ++job)
        if (plan.jobs[job].length != 0u) plan.jobs[job].frames_offset = plan.groups[job_group[job]].frames_offset;
    if (plan.contexts.resize(plan.groups.size()) != status_t::success_k)
        return result_t::fail(status_t::bad_alloc_k, "video decode context table allocation failed ({} groups)",
                              plan.groups.size());
    return result_t {std::move(plan)};
}

/** Stage 2's device side — upload the per-segment geometry (offsets/lengths/modality) and the per-job
 *  (tg,pr,pc) descriptor to the batch's device spans, then zero the whole tube grid so PAD cells and every
 *  failed/short clip stay PAD-zero. The tile and NV12 → RGB kernels write on the same stream, so this memset is
 *  their fence — no event. */
[[nodiscard]] static expected_t upload_geometry_(cursor_plan_t const &plan, pixel_tubes_span_t videos,
                                                 video_shape_span_t video_shapes, u32r1_span_t segment_offsets,
                                                 u32r1_span_t segment_lengths, modality_span_t segment_modality,
                                                 std::size_t job_count, std::size_t max_segments,
                                                 gpu_stream_t &stream) noexcept {
    expected_t uploaded = copy_host_to_device(plan.offsets.view().first(max_segments + 1u), segment_offsets, stream);
    if (uploaded) uploaded = copy_host_to_device(plan.lengths.view().first(max_segments), segment_lengths, stream);
    if (uploaded) uploaded = copy_host_to_device(plan.modality.view().first(max_segments), segment_modality, stream);
    nk::tensor_span<u16_t, 2> const job_shapes = video_shapes[nk::range(0, job_count), nk::slice];
    if (uploaded)
        uploaded = copy_host_to_device(plan.shapes.view().first(job_shapes.numel()), job_shapes.flatten<1>(), stream);
    if (!uploaded) return uploaded;
    return zero_async(videos.flatten<1>(), stream);
}

/** The interleaved decode drive's shared inputs — the engine table, the per-clip demux results, and the
 *  per-group decode targets the drive publishes onto an engine before feeding it. A borrow bundle: every span of
 *  scratch the caller keeps alive across the whole drive. The decode unit is a @ref decode_group_t — one clip
 *  decoded once per distinct spatial target, its frames shared by that clip's representations. */
struct decode_jobs_t {

    /** The owned NVDEC state — its engines' persistent parsers and per-clip decoders the drive feeds. */
    nvdec_state_t *state = nullptr;

    /** Per-clip demux results @b [clip_count] — the codec and Annex-B bytes, empty for a failed clip. */
    std::span<demuxed_t const> demuxed {};

    /** The decode groups @b [group_count] — each a clip decoded once to a distinct @c (pr,pc) target. */
    std::span<decode_group_t const> groups {};

    /** Per-group NVDEC callback context @b [group_count] — published on an engine, filled by the callbacks. */
    std::span<video_decode_context_t> contexts {};

    /** Base of the shared frame scratch — group @c g lands at @c frames_base+groups[g].frames_offset . */
    pixel_t *frames_base = nullptr;

    /** Patch edge P — a group's decode target is @c patch_rows·P by @c patch_cols·P . */
    u32_t patch_size = 0;

    /** The caller's stream every NV12 → RGB conversion enqueues on. */
    gpu_stream_t *stream = nullptr;
};

/** The outcome of opening one group on an engine: a chunk-fed clip is @c opened (its Annex-B tail follows), an
 *  AV1 clip is @c av1_done (its OBU temporal units were fed inline, so the engine advances), and a spent stripe
 *  is @c drained . Replaces the old empty-span return that conflated @c av1_done with @c drained . */
enum class feed_state_ : u8_t { opened, av1_done, drained };

/** @ref open_clip_ 's return — the feed outcome plus, when @c opened , the Annex-B tail to chunk-feed. */
struct clip_feed_t {

    /** Which of the three feed outcomes @ref open_clip_ reached. */
    feed_state_ state = feed_state_::drained;

    /** The opened group's Annex-B tail to chunk-feed; nonempty only when @c state==opened . */
    std::span<b8_t const> pending {};
};

/** One engine's live position in the interleaved feed: the decode group it is on and that group's
 *  still-unsubmitted Annex-B tail. The tail empties once the group's final chunk is fed, cueing the engine to its
 *  next group. */
struct engine_cursor_t {

    /** The group index this engine is on — its @c {e,e+N,…} stripe of the decode groups. */
    std::size_t group_index = 0;

    /** The current group's Annex-B bytes still to submit; empty once the engine is drained. */
    std::span<b8_t const> pending {};
};

/** Advance engine @p e to its next feedable group at/after @p group (skipping demux-failed, empty, and
 *  parser-build-failed groups), publish that group's decode context on the engine, and report the feed outcome. A
 *  non-AV1 clip returns @c opened with its Annex-B tail for the caller to chunk-feed; an AV1 clip is fed inline
 *  here (its OBU temporal units must not be split) and returns @c av1_done ; a spent stripe returns @c drained .
 *  @p group is left pointing at the group that was opened or fed inline. */
[[nodiscard]] static clip_feed_t open_clip_(decode_jobs_t &drive, std::size_t e, std::size_t &group) noexcept {
    nvdec_state_t *const state = drive.state;
    std::size_t const engine_count = state->engines.size();
    std::size_t const group_count = drive.groups.size();
    nvdec_engine_t &engine = state->engines[e];
    for (; group < group_count; group += engine_count) {
        // A group decodes one clip's elementary stream once, keeping every frame (temporal_stride 1) at its (pr,pc)
        // spatial target; the clip's representations subsample the shared frames per their frame_stride in the tile.
        decode_group_t const &unit = drive.groups[group];
        demuxed_t const &decoded = drive.demuxed[unit.clip];
        if (decoded.annex_b.empty() || unit.frames_to_decode == 0u) continue;
        if (!ensure_parser(engine, decoded.codec)) {
            drive.contexts[group].status = status_t::cuda_error_k;
            continue;
        }
        video_decode_context_t &context = drive.contexts[group];
        context.frames_base = drive.frames_base + unit.frames_offset;
        context.frames_requested = unit.frames_to_decode;
        context.temporal_stride = 1u; // decode-once keeps every displayed frame; reps subsample in the tile
        context.height = static_cast<std::size_t>(unit.patch_rows) * drive.patch_size;
        context.width = static_cast<std::size_t>(unit.patch_cols) * drive.patch_size;
        context.codec = decoded.codec;
        context.stream = drive.stream;
        engine.current = &context;
        std::span<b8_t const> const bytes = decoded.annex_b.view();
        if (decoded.codec != video_codec_t::av1_k) return clip_feed_t {feed_state_::opened, bytes}; // chunk-fed
        CUvideoparser const parser = engine.entries[static_cast<std::size_t>(decoded.codec)].parser;
        if (feed_av1_temporal_units(engine.cuvid, parser, bytes) != CUDA_SUCCESS &&
            context.status == status_t::success_k)
            context.status = status_t::cuda_error_k;
        engine.current = nullptr; // retire the AV1 group — the caller advances and re-opens this engine
        return clip_feed_t {feed_state_::av1_done, {}};
    }
    return clip_feed_t {feed_state_::drained, {}};
}

/** Drive engine @p e forward until it opens a chunk-fed group or drains, retrying past every AV1 group it
 *  decodes inline. Returns the Annex-B tail to feed — empty exactly once the engine's @c {e,e+N,…} stripe is
 *  spent. */
[[nodiscard]] static std::span<b8_t const> advance_engine_(decode_jobs_t &drive, std::size_t e,
                                                           std::size_t &group) noexcept {
    for (;;) {
        clip_feed_t const feed = open_clip_(drive, e, group);
        if (feed.state == feed_state_::opened) return feed.pending;
        if (feed.state == feed_state_::drained) return {};
        group += drive.state->engines.size(); // av1_done: this AV1 group is decoded — advance and retry the engine
    }
}

/**
 *  @brief Stage 3 — decode from this single host thread, chunk-interleaved across the engines: the GPU is the
 *      master, not a thread pool.
 *
 *  Engine @c e owns groups @c {e,e+N,…} (a group is one clip decoded once for a distinct spatial target) and works
 *  them in order; each round feeds one @ref nvdec_feed_chunk_bytes_k chunk of every live engine's current group so
 *  decode work stays submitted across every physical NVDEC unit at once (a whole-clip-per-call feed uses only one).
 *  The decode is hardware-async and every NV12 → RGB conversion enqueues on the caller's one stream, so there are
 *  no per-engine streams, events, or workers.
 */
[[nodiscard]] static expected_t drive_decode_(decode_jobs_t &drive, gpu_stream_t &stream) noexcept {
    nvdec_state_t *const state = drive.state;
    std::size_t const engine_count = state->engines.size();
    array<engine_cursor_t> cursors;
    if (cursors.resize(engine_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "decode cursor allocation failed ({} engines)", engine_count);

    // Prime each engine on its stripe's first feedable group, then count the live engines once. `live` only ever
    // decrements from here — when an engine feeds its final chunk and its next advance drains its stripe.
    u32_t live = 0;
    for (std::size_t e = 0; e != engine_count; ++e) {
        cursors[e].group_index = e;
        cursors[e].pending = advance_engine_(drive, e, cursors[e].group_index);
        live += !cursors[e].pending.empty();
    }
    while (live != 0)
        for (std::size_t e = 0; e != engine_count; ++e) {
            engine_cursor_t &cursor = cursors[e];
            if (cursor.pending.empty()) continue; // this engine drained its queue
            nvdec_engine_t &engine = state->engines[e];
            std::size_t const take = std::min(nvdec_feed_chunk_bytes_k, cursor.pending.size());
            CUVIDSOURCEDATAPACKET packet {};
            packet.payload = cuvid_payload_(cursor.pending.data());
            packet.payload_size = static_cast<unsigned long>(take);
            packet.flags = take == cursor.pending.size() ? CUVID_PKT_ENDOFSTREAM : 0u; // EOS on the final chunk
            CUvideoparser const parser = engine.entries[static_cast<std::size_t>(engine.current->codec)].parser;
            engine.cuvid->cuvidParseVideoData(parser, &packet);
            cursor.pending = cursor.pending.subspan(take);
            // A clip with every requested frame ends here; the rest would decode to be dropped.
            if (!cursor.pending.empty() && engine.current->frames_decoded >= engine.current->frames_requested) {
                CUVIDSOURCEDATAPACKET end {};
                end.flags = CUVID_PKT_ENDOFSTREAM;
                engine.cuvid->cuvidParseVideoData(parser, &end);
                cursor.pending = {};
            }
            if (cursor.pending.empty()) { // fed EOS — retire this group, advance to the engine's next (or drain)
                engine.current = nullptr;
                cursor.group_index += engine_count;
                cursor.pending = advance_engine_(drive, e, cursor.group_index);
                live -= cursor.pending.empty();
            }
        }
    return {};
}

/** A wholly black batch is a systematic failure (unsupported codec, driver fault), not graceful degradation,
 *  so fail loud when every decodable group produced no frames; a lone bad clip stays PAD-zero and the drive
 *  mark-and-continues past it. */
[[nodiscard]] static expected_t audit_black_batch_(std::span<demuxed_t const> demuxed,
                                                   std::span<decode_group_t const> groups,
                                                   std::span<video_decode_context_t const> contexts) noexcept {
    std::size_t const group_count = groups.size();
    std::size_t attempted = 0, produced = 0, hard_failures = 0;
    for (std::size_t group = 0; group != group_count; ++group)
        if (!demuxed[groups[group].clip].annex_b.empty() && groups[group].frames_to_decode != 0u) {
            ++attempted;
            if (contexts[group].frames_decoded != 0) ++produced;
            if (contexts[group].status != status_t::success_k) ++hard_failures;
        }
    if (attempted != 0 && produced == 0)
        return expected_t::fail(status_t::cuda_error_k,
                                "every one of {} decodable clips produced no frames ({} hard decode errors)", attempted,
                                hard_failures);
    return {};
}

/**
 *  @brief Stage 4 — scatter every job's frames into the grid at its segment placement, then synchronize.
 *
 *  Upload the per-job tile table plus the output-element prefix, launch one grid-strided tile kernel over the union
 *  of the jobs' output elements (each subsampling its clip's shared decode-group buffer by @c frame_stride ), and
 *  block until the grid is complete.
 */
[[nodiscard]] static expected_t tile_and_sync_(video_grid_geometry_t const &geometry, pixel_tubes_span_t videos,
                                               pixel_t *frames_base, std::span<tile_job_t const> jobs,
                                               std::span<u64_t const> prefix, gpu_stream_t &stream) noexcept {
    std::size_t const job_count = jobs.size();
    u64_t const output_elements = prefix[job_count]; // the tile's total output-element count, the grid's extent
    array<tile_job_t, gpu_alloc<tile_job_t>> device_jobs {gpu_alloc<tile_job_t> {stream.get()}};
    array<u64_t, gpu_alloc<u64_t>> device_prefix {gpu_alloc<u64_t> {stream.get()}};
    if (device_jobs.resize_uninitialized(job_count) != status_t::success_k ||
        device_prefix.resize_uninitialized(job_count + 1u) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "tile job device allocation failed");
    expected_t uploaded = copy_host_to_device<tile_job_t>(jobs, device_jobs.span(), stream);
    if (uploaded) uploaded = copy_host_to_device<u64_t>(prefix, device_prefix.span(), stream);
    if (!uploaded) return uploaded;
    if (output_elements != 0)
        if (expected_t tiled = launch_kernel(&tile_tubes_kernel_, dim3 {grid_size_for(output_elements), 1u, 1u},
                                             dim3 {kernel_block_size_default_k, 1u, 1u}, stream, videos.data(),
                                             frames_base, device_jobs.data(), device_prefix.data(),
                                             static_cast<u32_t>(job_count), geometry.columns, geometry.tube_frames,
                                             geometry.patch_size, geometry.channels);
            !tiled)
            return tiled;
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "tube tile synchronization failed");
    return {};
}

expected_t nvdec_session_t::decode_into(cpu_decode_pool_t &pool, column_view_t clips,
                                        std::span<video_batch_shape_t const> representations, pixel_tubes_span_t videos,
                                        video_shape_span_t video_shapes, u32r1_span_t segment_offsets,
                                        u32r1_span_t segment_lengths, modality_span_t segment_modality,
                                        gpu_stream_t &stream) noexcept {
    if (state_ == nullptr) return expected_t::fail(status_t::invalid_input_k, "moved-from NVDEC session");
    std::size_t const clip_count = clips.size();
    if (clip_count == 0) return {};

    // The batch's [rows,columns,tube_frames,P,P,C] tensor names the grid. A clip no longer fills a whole row: each
    // representation emits one segment whose footprint is derived from the clip's decoded native (frames,width,height).
    auto geometry_read = decode_geometry_(videos);
    if (!geometry_read) return expected_t::fail(std::move(geometry_read));
    video_grid_geometry_t const geometry = std::move(geometry_read.value_);

    // The loader's `fill` always hands a non-empty representations span (defaulting to one at frame_stride 1). An empty
    // span here is therefore a contract breach — fail loudly rather than silently produce zero jobs.
    if (representations.empty())
        return expected_t::fail(status_t::invalid_input_k,
                                "decode_into requires a non-empty representations span from fill");
    // The shape is the single source of geometry; every representation must match the grid's patch geometry (they
    // vary only in frame_stride, the temporal sampling this decode reads off each representation).
    for (video_batch_shape_t const &representation : representations)
        if (representation.patch_side != geometry.patch_size || representation.tube_frames != geometry.tube_frames ||
            representation.channels != geometry.channels)
            return expected_t::fail(status_t::shape_unsupported_k,
                                    "representation ({},{},{}) must match the video grid ({},{},{})",
                                    representation.patch_side, representation.tube_frames, representation.channels,
                                    geometry.patch_size, geometry.tube_frames, geometry.channels);
    u32_t const reps_count = static_cast<u32_t>(representations.size());
    std::size_t const job_count = clip_count * reps_count;
    std::size_t const max_segments = segment_lengths.extent(0);
    if (job_count > max_segments || segment_offsets.extent(0) < max_segments + 1u ||
        segment_modality.extent(0) < max_segments || video_shapes.extent(0) < job_count)
        return expected_t::fail(status_t::shape_unsupported_k, "batch cannot hold {} segments (cap {})", job_count,
                                max_segments);

    // Stage 1: parallel host demux over the borrowed pool — the one host-bound, embarrassingly parallel step. A clip
    // that fails to demux keeps an empty stream, contributes nothing, and stays PAD-zero.
    array<demuxed_t> demuxed;
    if (demuxed.resize(clip_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "demux result table allocation failed ({} clips)", clip_count);
    demuxed_t *const demuxed_data = demuxed.data();
    column_view_t const clips_tape = clips;
    {
        auto join = pool.for_n(clip_count,
                               [demuxed_data, clips_tape](std::size_t clip_index, fu::thread_in_domain_t) noexcept {
                                   expected<demuxed_t> clip = um::demux_video(clips_tape[clip_index]);
                                   if (clip) demuxed_data[clip_index] = std::move(clip.value_);
                               });
        (void)join; // The join's destructor blocks until every worker finishes.
    }
    std::span<demuxed_t const> const demuxed_view = demuxed.view().first(clip_count);

    // Stage 2: the token cursor — the serial repack scan writing the segment geometry plus the per-job tile
    // placement and decode parameters.
    auto cursor_built = build_token_cursor_(geometry, representations, demuxed_view, max_segments);
    if (!cursor_built) return expected_t::fail(std::move(cursor_built));
    cursor_plan_t plan = std::move(cursor_built.value_);

    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");

    if (expected_t staged = upload_geometry_(plan, videos, video_shapes, segment_offsets, segment_lengths,
                                             segment_modality, job_count, max_segments, stream);
        !staged)
        return staged;

    // Shared frame scratch: one contiguous [frames_to_decode,pr·P,pc·P,C] block per decode group, at
    // `plan.groups[g].frames_offset`. Zeroed so a short clip's missing frames stay PAD-zero once tiled.
    array<u8_t, gpu_alloc<u8_t>> frames {gpu_alloc<u8_t> {stream.get()}};
    std::size_t const frames_bytes = static_cast<std::size_t>(plan.frame_prefix);
    if (frames.resize_uninitialized(frames_bytes == 0 ? 1u : frames_bytes) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "frame scratch allocation failed ({} bytes)", frames_bytes);
    if (expected_t cleared = zero_async<u8_t, 1>(frames.span(), stream); !cleared) return cleared;

    // Stage 3: decode from this single host thread, chunk-interleaved across the engines — one decode per group.
    std::size_t const group_count = plan.groups.size();
    decode_jobs_t drive;
    drive.state = state_.get();
    drive.demuxed = demuxed_view;
    drive.groups = plan.groups.view();
    drive.contexts = plan.contexts.span().first(group_count);
    drive.frames_base = frames.data();
    drive.patch_size = geometry.patch_size;
    drive.stream = &stream;
    if (expected_t decoded = drive_decode_(drive, stream); !decoded) return decoded;

    if (expected_t audited = audit_black_batch_(demuxed_view, drive.groups, drive.contexts); !audited) return audited;

    // Stage 4: scatter every job's frames into the grid at its segment placement, then synchronize.
    return tile_and_sync_(geometry, videos, frames.data(), plan.jobs.view().first(job_count),
                          plan.prefix.view().first(job_count + 1u), stream);
}

#pragma endregion Host Container Demux

#pragma region Decode Orchestrator and Dataset Loader

expected_t ul::decode_videos(nvdec_session_t &nvdec, cpu_decode_pool_t &pool, column_view_t tape,
                             u64r1_view_t drawn_rows, std::span<video_batch_shape_t const> representations,
                             pixel_tubes_span_t videos, video_shape_span_t video_shapes, u32r1_span_t segment_offsets,
                             u32r1_span_t segment_lengths, modality_span_t segment_modality,
                             gpu_stream_t &stream) noexcept {
    std::size_t const clips = drawn_rows.size();
    if (clips == 0) return {};

    // Hop each drawn clip's container bytes device → pinned into a terminator-free tape (buffer + offsets, so
    // `tape[i]` is exactly the clip's bytes) — the demuxer's host-side source, one tape type end to end rather
    // than a caller-built table of spans. The demuxer reads by size, so no clip needs a delimiter.
    array<b8_t, pinned_alloc<b8_t>> encoded;
    array<u64_t> offsets;
    std::size_t total_bytes = 0;
    for (std::size_t index = 0; index != clips; ++index) total_bytes += tape[drawn_rows[index]].size();
    if (encoded.resize_uninitialized(total_bytes) != status_t::success_k ||
        offsets.resize(clips + 1u) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "clip staging allocation failed ({} bytes)", total_bytes);
    std::size_t cursor = 0;
    offsets[0] = 0;
    for (std::size_t index = 0; index != clips; ++index) {
        auto const blob = tape[drawn_rows[index]];
        if (expected_t hopped = copy_device_to_host(b8r1_view_t {blob}, encoded.span().subspan(cursor, blob.size()),
                                                    stream);
            !hopped)
            return hopped;
        cursor += blob.size();
        offsets[index + 1] = static_cast<u64_t>(cursor);
    }
    if (cuStreamSynchronize(stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "clip byte staging synchronization failed");

    column_view_t const pinned {encoded.view(), offsets.view()};
    return nvdec.decode_into(pool, pinned, representations, videos, video_shapes, segment_offsets, segment_lengths,
                             segment_modality, stream);
}

expected<video_loader_parquet_t> video_loader_parquet_t::make(gpu_direct_storage_session_t &session,
                                                              sz::string_view_t path_or_glob, u64_t seed,
                                                              cpu_decode_pool_t &pool,
                                                              video_batch_shape_t video) noexcept {
    using result_t = expected<video_loader_parquet_t>;
    auto paths = expand_glob(path_or_glob);
    if (!paths) return result_t::fail(std::move(paths));

    video_loader_parquet_t loader;
    loader.storage_ = &session;
    loader.pool_ = &pool;
    loader.absolute_paths_ = std::move(paths.value_);
    loader.seed_ = seed;
    loader.video_ = video;

    if (loader.row_prefix_.resize(loader.absolute_paths_.size() + 1) != status_t::success_k)
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

expected_t video_loader_parquet_t::fill(gpu_stream_t &gpu_stream, nvdec_session_t &decoder,
                                        video_loader_parquet_config_t config, u32_t gpu_rank, u32_t training_step,
                                        std::span<multimodal_batch_on_gpu_t> batches) noexcept {
    if (batches.empty()) return {};
    std::size_t const unroll = batches.size();
    std::size_t const rows = batches[0].videos.extent(0);
    std::size_t const columns_per_row = batches[0].videos.extent(1);
    std::size_t const max_segments = batches[0].segment_lengths.extent(0);
    if (rows == 0 || columns_per_row == 0)
        return expected_t::fail(status_t::shape_unsupported_k, "batch carries no video");

    // The loader's single flat `video_batch_shape_t` is the one representation (the shape is the single source of a
    // step's geometry) — a size-1 span today, cross-product-ready for a future multi-representation config. Each drawn
    // clip emits one segment per representation, sized from its decoded native resolution; conservatively budget each
    // representation-segment at a whole row so the token cursor never overflows (a small clip leaves its row slack —
    // pack density is a production-tuning knob, not a correctness one).
    std::span<video_batch_shape_t const> const representations {&video_, 1};
    std::size_t const per_clip_estimate = representations.size() * columns_per_row; // worst case: each rep fills a row
    std::size_t const clips_per_batch = std::min(max_segments / representations.size(),
                                                 (rows * columns_per_row) / per_clip_estimate);
    if (clips_per_batch == 0)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "the batch's {} segments / {} cells cannot hold one clip's {} representations",
                                max_segments, rows * columns_per_row, representations.size());
    std::size_t const total_clips = unroll * clips_per_batch;

    std::size_t const shard_count = absolute_paths_.size();
    std::size_t const floor_files = std::min<std::size_t>(2, shard_count);
    std::size_t const total_rows = row_prefix_[shard_count];
    constexpr splitmix64_hash_t hash {};

    array<sz::string_view_t> picked_paths;
    array<std::size_t> picked_shards;
    std::size_t pooled_estimate = 0, attempt = 0;
    while ((pooled_estimate < total_clips || picked_shards.size() < floor_files) &&
           picked_shards.size() < shard_count) {
        std::size_t const position = hash(seed_, training_step, gpu_rank, file_draw_salt_k, attempt) % total_rows;
        ++attempt;
        std::size_t const file = static_cast<std::size_t>(
            std::upper_bound(row_prefix_.begin(), row_prefix_.end(), position) - row_prefix_.begin() - 1);
        if (std::find(picked_shards.begin(), picked_shards.end(), file) != picked_shards.end()) continue;
        if (picked_shards.push_back(file) != status_t::success_k ||
            picked_paths.push_back(absolute_paths_[file].view()) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "file pick allocation failed");
        pooled_estimate += row_prefix_[file + 1] - row_prefix_[file];
    }

    auto reservoir_read = parquet_file_t::make(*storage_, picked_paths.view(), {&config.clip_column, 1}, gpu_stream);
    if (!reservoir_read) return expected_t::fail(std::move(reservoir_read));
    parquet_file_t const reservoir = std::move(reservoir_read.value_);
    std::size_t const pooled_rows = reservoir.rows();
    if (pooled_rows < total_clips)
        return expected_t::fail(status_t::shape_unsupported_k,
                                "{} pooled rows across {} files cover fewer than the {} clips this fill needs",
                                pooled_rows, picked_shards.size(), total_clips);

    switch_to_gpu_t gpu_scope {gpu_stream};
    if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed for the loader stream");

    column_view_t const column = reservoir.column(0);
    std::size_t const offsets_count = column.size() + 1u;
    array<u64_t> reservoir_offsets;
    if (reservoir_offsets.resize_uninitialized(offsets_count) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "reservoir offsets landing failed ({} rows)", pooled_rows);
    if (expected_t landed = copy_device_to_host(column.offsets, reservoir_offsets.span(), gpu_stream); !landed)
        return landed;
    if (cuStreamSynchronize(gpu_stream.get()) != CUDA_SUCCESS)
        return expected_t::fail(status_t::cuda_error_k, "reservoir offsets readback failed");

    u64_t const row_key = hash(seed_, gpu_rank, row_draw_salt_k);
    array<u64_t> record_indices;
    if (record_indices.resize_uninitialized(total_clips) != status_t::success_k)
        return expected_t::fail(status_t::bad_alloc_k, "row index allocation failed ({} draws)", total_clips);
    for (std::size_t index = 0; index != total_clips; ++index)
        record_indices[index] = epoch_permutation(row_key, data_epoch_t {training_step}, index, pooled_rows);

    column_view_t const tape {column.payload, reservoir_offsets.view()};
    u64r2_view_t const batch_records {record_indices.data(), unroll, clips_per_batch};
    for (std::size_t batch = 0; batch != unroll; ++batch) {
        multimodal_batch_on_gpu_t &loaded = batches[batch];
        if (loaded.videos.extent(0) != rows || loaded.videos.extent(1) != columns_per_row ||
            loaded.segment_lengths.extent(0) != max_segments || loaded.video_shapes.extent(0) != max_segments)
            return expected_t::fail(status_t::shape_unsupported_k, "batch {} does not match the lead batch geometry",
                                    batch);
        // The whole [rows,columns,…] grid is the scatter target now — segments land across its rows, not one per row.
        if (expected_t decoded = decode_videos(decoder, *pool_, tape, batch_records[batch, nk::slice], representations,
                                               loaded.videos.span(), loaded.video_shapes.span(),
                                               loaded.segment_offsets.span(), loaded.segment_lengths.span(),
                                               loaded.segment_modality.span(), gpu_stream);
            !decoded)
            return decoded;
    }
    return {};
}

#pragma endregion Decode Orchestrator and Dataset Loader
