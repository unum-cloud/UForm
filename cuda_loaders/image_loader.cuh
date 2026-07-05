/**
 *  @brief The Parquet image loader: tied to a @b dataset, not a GPU — the same shape as
 *      @ref tokens_loader_parquet_t . @ref make reads only each shard's Parquet footer once, and every
 *      @ref fill takes the caller's stream, rank, and step, so one loader serves every GPU on every step.
 *      Each @ref fill opens @b as many randomly chosen shards as it takes to pool the images it needs — at
 *      least two unless the dataset is one file — decodes their blob column into one device reservoir over
 *      GDS, draws rows by a keyed bijection over the whole pool, and decodes the drawn compressed blobs
 *      into the batches' uniform @c images run with @ref nvjpeg_session_t .
 *  @file cuda_loaders/image_loader.cuh
 *  @author Ash Vardanian
 *  @date July 4, 2026
 *
 *  @section image_decode_split JPEG on the GPU, the rest on the CPU pool
 *
 *  A drawn blob is routed by its magic bytes. @b JPEG — the overwhelming majority of web corpora — decodes
 *  @b device-native: nvJPEG's hardware backend takes the compressed bitstream straight from the GDS-resident
 *  reservoir tape (device pointers into @c buffer_ , no host hop) and writes interleaved RGB to device.
 *  @b PNG / @b WebP have no device decoder, so those rows take the audio loader's road — the compressed blob
 *  hops to a pinned tape, a @c fork_union pool decodes it on the CPU (libjpeg-turbo / libpng / libwebp) into
 *  a pinned pixel tape, and one bulk copy lands it on device. Both roads meet at the resize.
 *
 *  @section image_aspect One uniform run, per-image aspect on the side
 *
 *  The loader bakes @b no resolution. Every image becomes exactly @c max_columns_per_row square patches, so
 *  its decoded pixels resize to its @b own @c (patch_rows·patch_size,patch_cols·patch_size) — the factor
 *  pair of @c max_columns_per_row nearest the native aspect, clamped to the @c [1×N,N×1] the batch's own
 *  patch budget @c N allows. Because the patch count is fixed, the byte count is one across the batch: pixels
 *  land in the batch's uniform @c images run and each row's @c (patch_rows,patch_cols) in @c image_shapes ,
 *  the aspect @ref patchify_images ' M-RoPE coordinates read downstream — never a crop or stretch toward a
 *  square here. Patchify and projection happen later, in the differentiated step, never in the loader.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <span>    // `std::span`

#include <nvjpeg.h> // vendor-boundary: `nvjpegHandle_t` / `nvjpegJpegState_t` — the session's device state

#include <fork_union.hpp> // vendor-boundary: `fu::basic_pool_t` — the borrowed CPU-decode pool

#include "cuda_core/types.cuh"             // `expected`, `status_t`, the scalar aliases, `epoch_t`, `epoch_permutation`
#include "cuda_core/array.cuh"             // `array` — the per-fill index scratch
#include "cuda_core/gpu_allocators.cuh"    // `pinned_alloc` — the non-JPEG fallback's host tapes
#include "cuda_core/schedule.cuh"          // `gpu_stream_t`, `switch_to_gpu_t`, `launch_kernel`
#include "cuda_core/multimodal_batch.cuh"  // `multimodal_batch_on_gpu_t` — the decode target
#include "cuda_loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "cuda_loaders/dataset_shards.hpp" // `parquet_file_t`, `expand_glob`, `column_view_t`

namespace unum::uform::loaders {

/** @brief The writable uniform pixel run of one batch's image rows, shaped @b [images,columns·patch_scalars]
 *      — the decode's destination, one image's pixels @c (row,column,channel) at its own resolution. */
using pixel_run_span_t = nk::tensor_span<pixel_t, 2>;
/** @brief The writable per-image @c (patch_rows,patch_cols) shapes of one batch, shaped @b [images,2] . */
using image_shape_span_t = nk::tensor_span<u16_t, 2>;
/** @brief The host-side worker pool the CPU fallback fans PNG / WebP / progressive-JPEG decodes across —
 *      one per run, owned at the composition root (the @b only sanctioned host-thread pool), borrowed here. */
using cpu_decode_pool_t = ashvardanian::fork_union::basic_pool_t;

/**
 *  @brief The nvJPEG session: the device-native batched JPEG decoder, one per GPU at the composition root,
 *      threaded by reference into every @ref image_loader_parquet_t::fill on that GPU. Its whole state is the
 *      nvJPEG handle and batched-decode state — both light C handles, so there is no pimpl; move-only RAII.
 *      The CPU-fallback pool and the per-fill scratch are not the session's — they are borrowed / per-call.
 */
struct nvjpeg_session_t {

    nvjpeg_session_t() noexcept = default;
    nvjpeg_session_t(nvjpeg_session_t const &) = delete;
    nvjpeg_session_t &operator=(nvjpeg_session_t const &) = delete;
    nvjpeg_session_t(nvjpeg_session_t &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), jpeg_state_(std::exchange(other.jpeg_state_, nullptr)),
          hardware_(other.hardware_) {}
    nvjpeg_session_t &operator=(nvjpeg_session_t &&other) noexcept;
    ~nvjpeg_session_t() noexcept;

    /** @brief Create the nvJPEG handle on a device-bitstream backend and its batched-decode state — once, at
     *      setup. The whole thing is the two handles; every fill's staging is a per-call local. */
    [[nodiscard]] static expected<nvjpeg_session_t> make() noexcept;

    /** @brief Whether the handle drives the hardware decode engine (vs the GPU-hybrid fallback) — only the
     *      hardware backend honors the decode-time scale factor, so @ref decode_images gates scaling on it. */
    [[nodiscard]] bool hardware_decode() const noexcept { return hardware_; }

    /**
     *  @brief Decode one assembled JPEG batch — the session's nvJPEG-only slice of @ref decode_images .
     *      Fails as a whole (the batch is atomic), leaving the caller to route a rejected batch to the CPU.
     *
     *  @param[in] data Per-image device bitstream pointers into the reservoir tape.
     *  @param[in] lengths Per-image bitstream byte counts, matching @p data .
     *  @param[out] destinations Per-image RGB targets (interleaved, one plane), sized to each image's scaled output.
     *  @param[in] scales Per-image decode scale factor — honored only under @ref hardware_decode , else must be NONE.
     *  @param[inout] stream The stream the decode enqueues on.
     */
    [[nodiscard]] expected_t decode_jpeg_batch(std::span<unsigned char const *const> data,
                                               std::span<std::size_t const> lengths,
                                               std::span<nvjpegImage_t> destinations,
                                               std::span<nvjpegScaleFactor_t const> scales,
                                               gpu_stream_t &stream) noexcept;

  private:
    /** @brief The nvJPEG library handle, created on a device-bitstream backend. */
    nvjpegHandle_t handle_ = nullptr;
    /** @brief The batched-decode state, re-initialized per decode to that fill's JPEG count. */
    nvjpegJpegState_t jpeg_state_ = nullptr;
    /** @brief Whether @ref make landed the hardware backend (vs the GPU-hybrid fallback). */
    bool hardware_ = false;
};

/**
 *  @brief Decode a batch's drawn image rows into its uniform pixel run and per-image shapes — the media
 *      decode every image loader shares (Parquet today, tarball next). Routes each drawn blob by its magic
 *      bytes: JPEG through @p jpeg (device-native, scaled toward the target at decode), PNG / WebP / rejected
 *      JPEG across @p pool on the CPU, then one NPP batch crops each to its factor-pair aspect and resizes it
 *      into the run. Pure enqueue on @p stream but for the triage and CPU-fallback host hops.
 *
 *  @param[inout] jpeg This GPU's nvJPEG session, borrowed for the JPEG rows.
 *  @param[inout] pool The run's CPU-decode worker pool, borrowed for the non-JPEG / rejected-JPEG rows.
 *  @param[in] tape The pooled blob column with its offsets landed on host — device bytes, host-indexable, so
 *      @c tape[row] is a device slice computed on the host.
 *  @param[in] drawn_rows The tape rows to decode, one per output image row.
 *  @param[out] pixels The uniform per-image pixel run to fill; its leading extent must cover @p drawn_rows .
 *  @param[out] shapes Each image's @c (patch_rows,patch_cols) factorization of @p num_patches ; matches @p pixels .
 *  @param[in] patch_size The square patch side each image's grid tiles into.
 *  @param[in] num_patches The fixed patch budget per image; its factor pair nearest the aspect sets the crop.
 *  @param[inout] stream The stream the staging, decode, resize, and copies run on.
 */
[[nodiscard]] expected_t decode_images(nvjpeg_session_t &jpeg, cpu_decode_pool_t &pool, column_view_t tape,
                                       std::span<u64_t const> drawn_rows, pixel_run_span_t pixels,
                                       image_shape_span_t shapes, u32_t patch_size, u32_t num_patches,
                                       gpu_stream_t &stream) noexcept;

/** @brief The Parquet image loader's baked knobs — a different config is a different loader. */
struct image_loader_parquet_config_t {
    /** @brief The Parquet column holding the encoded image blobs ( @c list<uint8> ). */
    sz::string_view column = sz::string_view {"image", 5};
};

/**
 *  @brief The loader object: dataset-scoped, rank-agnostic, GPU-accelerated — the image sibling of
 *      @ref tokens_loader_parquet_t .
 *
 *  @warning Holds one dataset's shard footers; keep one per input dataset. @ref fill allocates its scratch
 *      on demand and is safe to call concurrently for different GPU ranks on different streams.
 */
struct image_loader_parquet_t {

    /** @brief Failure-safe exception-free constructor that indexes all the shards' footers (host, once).
     *  @param[inout] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's Parquet paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed.
     *  @param[inout] pool The run's CPU-decode worker pool, owned at the composition root and borrowed here
     *      for the PNG / WebP / progressive-JPEG fallback; the loader never spawns threads of its own. */
    [[nodiscard]] static expected<image_loader_parquet_t> make(gpu_direct_storage_session_t &session,
                                                               sz::string_view path_or_glob, u64_t seed,
                                                               cpu_decode_pool_t &pool) noexcept;

    /**
     *  @brief Fill the next few @p batches ' @c images run on one GPU rank, one decoded image per row.
     *  @sa multimodal_batch_mixer_t for the subsequent cross-GPU mixing.
     *
     *  Opens ≥2 randomly chosen shards (keyed by @c (seed,step,rank) ), pools their blob column into one
     *  reservoir, draws a scatter of rows across it, and decodes each batch's drawn blobs into its image
     *  rows — each resized to its own factor pair of the batch's patch budget — writing @c image_shapes and
     *  one full-row segment per image.
     *
     *  @param[inout] gpu_stream The GPU-specific stream the staging, decode, resize, and segmenting run on.
     *  @param[inout] decoder This GPU's nvJPEG session, borrowed for the decode.
     *  @param[in] config Which Parquet column holds the image blobs.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches Output batches whose @c images run, @c image_shapes , and segment geometry get
     *      populated.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, nvjpeg_session_t &decoder,
                                  image_loader_parquet_config_t config, u32_t gpu_rank, u32_t training_step,
                                  std::span<multimodal_batch_on_gpu_t> batches) noexcept;

  private:
    /** @brief The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;
    /** @brief The run's CPU-decode worker pool — borrowed from the composition root, handed to the decode's
     *      PNG / WebP / progressive-JPEG fallback; never owned, never spawned here. */
    cpu_decode_pool_t *pool_ = nullptr;
    /** @brief The dataset's Parquet shard paths, sorted by the glob — owned. */
    array<sz::string> absolute_paths_ {};
    /** @brief The run seed keying the random file and row draws. */
    u64_t seed_ = 0;
    /** @brief @b [shards+1] cumulative footer row counts — the weighted file draw's search table (an
     *      @c upper_bound maps a keyed row position to its owning file) and, at its back, the dataset's
     *      total rows. Read once per dataset; shared across every GPU the loader serves. */
    array<std::size_t> row_prefix_ {};
};

} // namespace unum::uform::loaders
