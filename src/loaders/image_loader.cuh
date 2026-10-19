/**
 *  @brief The Parquet image loader: tied to a @b dataset, not a GPU — the same shape as
 *      @ref tokens_loader_parquet_t .
 *  @file src/loaders/image_loader.cuh
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  @ref make reads only each shard's Parquet footer once, and every @ref fill takes the caller's stream, rank,
 *  and step, so one loader serves every GPU on every step. Each @ref fill opens @b as many randomly chosen
 *  shards as it takes to pool the images it needs — at least two unless the dataset is one file — decodes
 *  their blob column into one device reservoir over GDS, draws rows by a keyed bijection over the whole pool,
 *  and decodes the drawn compressed blobs into the batch's shared patch grid with @ref nvjpeg_session_t .
 *
 *  @section image_decode_split JPEG on the GPU, the rest on the CPU pool
 *
 *  A drawn blob is routed by its magic bytes. @b JPEG — the overwhelming majority of web corpora — decodes
 *  @b device-native: nvJPEG's hardware backend takes the compressed bitstream straight from the GDS-resident
 *  reservoir tape (device pointers into its payload, no host hop) and writes interleaved RGB to device.
 *  @b PNG / @b WebP have no device decoder, so those rows take the audio loader's road — the compressed blob
 *  hops to a pinned tape, a @c forkunion pool decodes it on the CPU (libjpeg-turbo / libpng / libwebp) into
 *  a pinned pixel tape, and one bulk copy lands it on device. Both roads meet at the resize.
 *
 *  @section image_aspect Sub-row segments, per-image budget and aspect on the side
 *
 *  The loader bakes @b no resolution. Each drawn image is @b one segment whose column budget is derived from
 *  its native resolution (a bigger image earns more patches, monotonically, clamped to
 *  @c [max_columns/max_segments,max_columns] so many small images pack into a row), then factored toward the
 *  native aspect into @c (patch_rows,patch_cols) — its decoded pixels resize to
 *  @c (patch_rows·patch_side,patch_cols·patch_side) . A serial cursor packs the segments row-safe (never
 *  crossing a row); each scatters its patches into a column sub-range of the shared @c [rows,columns,P,P,4]
 *  grid, its @c (patch_rows,patch_cols) landing in @c image_shapes and its placement in
 *  @c segment_offsets/lengths — the aspect the M-RoPE coordinates read downstream, never a crop or stretch
 *  toward a square here. Patchify and projection happen later, in the differentiated step, never in the loader.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span`

#include <nvjpeg.h> // vendor-boundary: `nvjpegHandle_t` / `nvjpegJpegState_t` — the session's device state

#include <forkunion.hpp> // vendor-boundary: `fu::flat_pool_t` — the borrowed CPU-decode pool

#include "core/types.hpp"             // `expected`, `status_t`, `data_epoch_t`, `epoch_permutation`
#include "core/array.hpp"             // `array` — the per-fill index scratch
#include "core/gpu_allocators.hpp"    // `pinned_alloc` — the non-JPEG fallback's host tapes
#include "core/schedule.cuh"          // `gpu_stream_t`, `switch_to_gpu_t`, `launch_kernel`
#include "core/multimodal_batch.cuh"  // `multimodal_batch_on_gpu_t` — the decode target
#include "media/image.hpp"            // `pixel_patches_span_t`
#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "loaders/dataset_shards.hpp" // `parquet_file_t`, `expand_glob`, `column_view_t`

namespace unum::uform::loaders {

/** The writable patch-arranged pixel grid of one batch, shaped @b [rows,columns,patch_side,patch_side,4] — the
 *  decode's destination. Each drawn image scatters its patches into a column sub-range of one row, several
 *  images sharing a row. */
using media::pixel_patches_span_t;

/** The host-side worker pool the CPU fallback fans PNG / WebP / progressive-JPEG decodes across —
 *  one per run, owned at the composition root (the @b only sanctioned host-thread pool), borrowed here. */
using cpu_decode_pool_t = ashvardanian::forkunion::flat_pool_t;

/** Which nvJPEG device backend @ref nvjpeg_session_t::make landed. Only the hardware engine honors the
 *  decode-time scale factor, so @ref decode_images gates scaling on it — a named tier, never a bool. */
enum class nvjpeg_backend_t : u8_t {

    /** The hardware decode engine — device bitstream input, honors the decode-time scale factor. */
    hardware_k = 0,

    /** The GPU-hybrid fallback — device bitstream input, no per-image scale factor. */
    gpu_hybrid_k = 1,
};

/** One JPEG batch entry: the compressed bitstream and length, the plan it maps back to (for fallback
 *  marking and RGBA expansion), its byte offset into the packed-RGB scratch, its decode scale, and the
 *  interleaved-RGB destination wired once that scratch is allocated — the session's decode input item, whose
 *  parallel arrays @ref nvjpeg_session_t::decode_jpeg_batch fills straight from an owning table of these. */
struct jpeg_entry_t {

    /** Device bitstream pointer into the reservoir tape. */
    b8_t const *bitstream;

    /** The bitstream's byte count. */
    std::size_t length;

    /** The plan this entry decodes — maps back for fallback marking / RGBA expansion. */
    std::size_t plan_index;

    /** The entry's byte offset into the packed-RGB scratch. */
    std::size_t rgb_offset;

    /** The decode scale factor for this entry. */
    nvjpegScaleFactor_t scale;

    /** The interleaved-RGB destination, wired once the RGB scratch is allocated. */
    nvjpegImage_t destination;
};

/**
 *  @brief The nvJPEG session: the device-native batched JPEG decoder, one per GPU at the composition root,
 *      threaded by reference into every @ref image_loader_parquet_t::fill on that GPU.
 *
 *  Its whole state is the nvJPEG handle and batched-decode state — both light C handles, so there is no
 *  pimpl; move-only RAII. The CPU-fallback pool and the per-fill scratch are not the session's — they are
 *  borrowed / per-call.
 */
struct nvjpeg_session_t {

    nvjpeg_session_t() noexcept = default;
    nvjpeg_session_t(nvjpeg_session_t const &) = delete;
    nvjpeg_session_t &operator=(nvjpeg_session_t const &) = delete;
    nvjpeg_session_t(nvjpeg_session_t &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), jpeg_state_(std::exchange(other.jpeg_state_, nullptr)),
          backend_(other.backend_) {}
    nvjpeg_session_t &operator=(nvjpeg_session_t &&other) noexcept;
    ~nvjpeg_session_t() noexcept;

    /** Create the nvJPEG handle on a device-bitstream backend and its batched-decode state — once, at
     *  setup. The whole thing is the two handles; every fill's staging is a per-call local. */
    [[nodiscard]] static expected<nvjpeg_session_t> make() noexcept;

    /** Which device backend the handle drives — only @ref nvjpeg_backend_t::hardware_k honors the
     *  decode-time scale factor, so @ref decode_images gates scaling on it. */
    [[nodiscard]] nvjpeg_backend_t backend() const noexcept { return backend_; }

    /**
     *  @brief Decode one assembled JPEG batch — the session's nvJPEG-only slice of @ref decode_images .
     *
     *  Fails as a whole (the batch is atomic), leaving the caller to route a rejected batch to the CPU.
     *
     *  @param[in] entries The batch's decode entries — each a bitstream, length, RGB destination, and scale;
     *      the method fills nvJPEG's parallel arrays straight from them at the vendor boundary.
     *  @param[inout] stream The stream the decode enqueues on.
     */
    [[nodiscard]] expected_t decode_jpeg_batch(std::span<jpeg_entry_t const> entries, gpu_stream_t &stream) noexcept;

  private:

    /** The nvJPEG library handle, created on a device-bitstream backend. */
    nvjpegHandle_t handle_ = nullptr;

    /** The batched-decode state, re-initialized per decode to that fill's JPEG count. */
    nvjpegJpegState_t jpeg_state_ = nullptr;

    /** Which device backend @ref make landed — defaults to the GPU-hybrid fallback. */
    nvjpeg_backend_t backend_ = nvjpeg_backend_t::gpu_hybrid_k;
};

/**
 *  @brief Decode a batch's drawn images into its shared patch grid as sub-row segments — the media decode
 *      every image loader shares (Parquet today, tarball next).
 *
 *  Each drawn blob becomes @b one segment whose column budget is derived from its native resolution (bigger
 *  images get more patches, clamped so many pack into a row); a serial cursor packs the segments row-safe,
 *  never crossing a row, and each image's @c patch_rows·patch_cols patches scatter into its column sub-range.
 *  Routes each blob by its magic bytes: JPEG through @p jpeg (device-native, scaled toward the target at
 *  decode), PNG / WebP / rejected JPEG across @p pool on the CPU, then one NPP batch crops each to its
 *  factor-pair aspect and resizes it into the grid. Pure enqueue on @p stream but for the triage and
 *  CPU-fallback host hops.
 *
 *  @param[inout] jpeg This GPU's nvJPEG session, borrowed for the JPEG rows.
 *  @param[inout] pool The run's CPU-decode worker pool, borrowed for the non-JPEG / rejected-JPEG rows.
 *  @param[in] tape The pooled blob column with its offsets landed on host — device bytes, host-indexable, so
 *      @c tape[row] is a device slice computed on the host.
 *  @param[in] drawn_rows The tape rows to decode; each becomes one live image segment, in draw order.
 *  @param[in] representations The flat @ref image_batch_shape_t representations ( @c patch_side , @c channels ); the
 *      shape is the single source of geometry, validated against the grid. Image carries no per-representation
 *      variation yet, so it consumes one (must be non-empty).
 *  @param[out] pixels The patch-arranged @b [rows,columns,P,P,C] grid to fill; its extents give the row width
 *      (columns), the square patch side (extent 2), and the RGBA channel count (extent 4).
 *  @param[out] shapes Each segment's @c (patch_rows,patch_cols) factorization of its column budget, shaped
 *      @b [max_segments,2] ; segment @c s is meaningful iff @p segment_modality is @c image_k .
 *  @param[out] segment_offsets @b [max_segments+1] flat start positions (the cursor's terminal at the back).
 *  @param[out] segment_lengths @b [max_segments] each segment's column budget (0 for PAD or overflow-dropped).
 *  @param[out] segment_modality @b [max_segments] @c image_k for a live image, @c text_k for a PAD segment.
 *  @param[inout] stream The stream the staging, decode, resize, and copies run on.
 */
[[nodiscard]] expected_t decode_images(nvjpeg_session_t &jpeg, cpu_decode_pool_t &pool, column_view_t tape,
                                       u64r1_view_t drawn_rows, std::span<image_batch_shape_t const> representations,
                                       pixel_patches_span_t pixels, image_shape_span_t shapes,
                                       u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                       modality_span_t segment_modality, gpu_stream_t &stream) noexcept;

/** The Parquet image loader's baked knobs — a different config is a different loader. */
struct image_loader_parquet_config_t {

    /** The Parquet column holding the encoded image blobs ( @c list<uint8> ). */
    sz::string_view_t column = sz::string_view_t {"image", 5};
};

/**
 *  @brief The loader object: dataset-scoped, rank-agnostic, GPU-accelerated — the image sibling of
 *      @ref tokens_loader_parquet_t .
 *
 *  @warning Holds one dataset's shard footers; keep one per input dataset. @ref fill allocates its scratch
 *      on demand and is safe to call concurrently for different GPU ranks on different streams.
 */
struct image_loader_parquet_t {

    /**
     *  @brief Failure-safe exception-free constructor that indexes all the shards' footers (host, once).
     *  @param[inout] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's Parquet paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed.
     *  @param[inout] pool The run's CPU-decode worker pool, owned at the composition root and borrowed here
     *      for the PNG / WebP / progressive-JPEG fallback; the loader never spawns threads of its own.
     *  @param[in] image The image modality's own geometry slice ( @c patch_side , @c channels ) — the single source
     *      of a step's image geometry; it never sees the sibling modalities' shapes.
     */
    [[nodiscard]] static expected<image_loader_parquet_t> make(gpu_direct_storage_session_t &session,
                                                               sz::string_view_t path_or_glob, u64_t seed,
                                                               cpu_decode_pool_t &pool,
                                                               image_batch_shape_t image) noexcept;

    /**
     *  @brief Fill the next few @p batches ' @c images grid on one GPU rank, each image one sub-row segment.
     *  @sa multimodal_batch_mixer_t for the subsequent cross-GPU mixing.
     *
     *  Opens ≥2 randomly chosen shards (keyed by @c (seed,step,rank) ), pools their blob column into one
     *  reservoir, draws up to @c max_segments_per_gpu rows across it, and decodes each batch's drawn blobs
     *  into its patch grid — each resized to its own factor pair of a native-resolution-derived column budget
     *  and packed row-safe into a column sub-range — writing @c image_shapes , @c segment_offsets/lengths ,
     *  and @c segment_modality @c=image_k per live image.
     *
     *  @param[inout] gpu_stream The GPU-specific stream the staging, decode, resize, and segmenting run on.
     *  @param[inout] decoder This GPU's nvJPEG session, borrowed for the decode.
     *  @param[in] config Which Parquet column holds the image blobs.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches Output batches whose @c images grid, @c image_shapes , @c segment_offsets/lengths ,
     *      and @c segment_modality get populated.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, nvjpeg_session_t &decoder,
                                  image_loader_parquet_config_t config, u32_t gpu_rank, u32_t training_step,
                                  std::span<multimodal_batch_on_gpu_t> batches) noexcept;

  private:

    /** The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;

    /** The run's CPU-decode worker pool — borrowed from the composition root, handed to the decode's
     *  PNG / WebP / progressive-JPEG fallback; never owned, never spawned here. */
    cpu_decode_pool_t *pool_ = nullptr;

    /** The dataset's Parquet shard paths, sorted by the glob — owned. */
    array<sz::string_t> absolute_paths_ {};

    /** The run seed keying the random file and row draws. */
    u64_t seed_ = 0;

    /** @b [shards+1] cumulative footer row counts — the weighted file draw's search table (an
     *  @c upper_bound maps a keyed row position to its owning file) and, at its back, the dataset's
     *  total rows. Read once per dataset; shared across every GPU the loader serves. */
    array<std::size_t> row_prefix_ {};

    /** The image modality's geometry ( @c patch_side , @c channels ) every @ref fill hands the decode as its
     *  one representation — fixed at @ref make . */
    image_batch_shape_t image_ {};
};

} // namespace unum::uform::loaders
