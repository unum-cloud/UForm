/**
 *  @file src/loaders/video_loader.cuh
 *  @author Ash Vardanian
 *  @date July 5, 2026
 *  @brief The Parquet video loader: tied to a @b dataset, not a GPU — the same shape as
 *      @ref tokens_loader_parquet_t , @ref image_loader_parquet_t and
 *      @ref audio_loader_parquet_t .
 *
 *  @ref make reads each shard's Parquet footer once; every @ref fill takes the caller's stream, rank, and step,
 *  pools randomly chosen shards' clip column into one device reservoir over GDS, draws rows by a keyed bijection,
 *  and decodes the drawn clips into the batches' @c videos tube grid with @ref decode_videos .
 *
 *  @section video_host_hop The one host hop, and why
 *
 *  A container ( @c mp4 / @c webm ) can only be parsed on the host — libavformat walks its box / EBML structure
 *  and rewrites the elementary stream to Annex-B — so the drawn clips' compressed bytes make one device-to-pinned
 *  hop before demux. Demux is the one place STYLE sanctions a worker pool ( @ref decode_videos borrows the run's
 *  @ref cpu_decode_pool_t ): container parsing is host-bound and embarrassingly parallel across clips. Everything
 *  after demux stays on device.
 *
 *  @section video_multi_engine Saturating the physical decode units — GPU-driven
 *
 *  A GPU carries several physical NVDEC engines. @ref nvdec_session_t holds @c engine_count independent decode
 *  @b engines — each with its own per-codec NVDEC parser plus a decoder rebuilt from every clip's sequence header —
 *  and @ref nvdec_session_t::decode_into feeds the drawn clips across them in small @b interleaved chunks from a
 *  @b single host thread, so decode work stays submitted to every engine at once. The GPU is the master: decode is
 *  hardware-asynchronous, so the parallelism is in the silicon, not in CPU threads — measured at ~6x on a 7-engine
 *  H100, beating a thread-per-engine pool. (A whole-clip-per-call feed, by contrast, uses only one engine.) The
 *  engines own no streams — the whole GPU tail rides the caller's one stream. The CPU pool stays a @b slave: it does
 *  only the host-bound demux, never the GPU.
 *
 *  @section video_jobs Jobs, the clips×representations cross-product
 *
 *  A @b job is one @c (clip,representation) pair; a fill's jobs are the full @b cross-product of the drawn clips and
 *  the configured representations (each a flat @ref video_batch_shape_t ). Each job emits exactly @b one segment
 *  packed onto the @c [rows,columns] grid — the @b segment / column axis, not a tensor rank — whose footprint is
 *  derived from the clip's decoded native resolution ( @ref derive_triple_ ), never a fixed budget. The common
 *  single-representation (Omni) config is one decode per clip. Representations of one clip differ only in
 *  @c frame_stride and (almost always) share their spatial target, so the drive decodes each clip @b once per
 *  distinct spatial target — keeping every frame in a shared buffer — and each representation
 *  temporally subsamples those frames by its own @c frame_stride when the tile kernel gathers its tubes. A
 *  multi-representation config therefore pays one decode per distinct @c (patch_rows,patch_cols) , not one per job.
 *
 *  @section video_shape The patch-arranged contract
 *
 *  @ref multimodal_batch_on_gpu_t::videos is @b [rows,columns,tube_frames,patch_side,patch_side,channels] UInt8.
 *  Each drawn clip emits one @b segment per representation, its @c (temporal_groups,patch_rows,patch_cols) derived
 *  from the clip's decoded native @c (frames,width,height) — @c patch_rows and @c patch_cols tile the native frame at
 *  @c patch_side and @c temporal_groups packs the @c frame_stride -sampled frames into @c tube_frames -deep tubes,
 *  exactly like an image with a temporal axis — then clamped to one row so an oversized clip downscales to fit
 *  rather than dropping. NVDEC's post-processor scales each sampled frame straight to the
 *  segment's @c (patch_cols·P,patch_rows·P) grid target, @c tile_tubes_kernel_ scatters the tubes into the segment's
 *  @c [column_begin,column_begin+len) cells of its row, and the loader records each segment's factorization into
 *  @c video_shapes . No learnable transform (projection, M-RoPE) is applied here — that lives in the differentiated
 *  step.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <memory> // `std::unique_ptr` — the pimpl owning the NVDEC engine state
#include <span>   // `std::span`

#include <forkunion.hpp> // vendor-boundary: `fu::flat_pool_t` — the borrowed CPU demux / decode-drive pool

#include "core/types.hpp"             // `expected`, `status_t`, `data_epoch_t`, `epoch_permutation`
#include "core/gpu_allocators.hpp"    // `pinned_alloc`
#include "core/schedule.cuh"          // `gpu_stream_t`, `switch_to_gpu_t`, `launch_kernel`
#include "core/multimodal_batch.cuh"  // `multimodal_batch_on_gpu_t`
#include "media/video.hpp"            // `pixel_tubes_span_t`
#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "loaders/dataset_shards.hpp" // `parquet_file_t`, `expand_glob`, `column_view_t`

/** The session's owned NVDEC engine state — forward-declared at global scope and defined in
 *  @c video_loader.cu (which reaches its namespace with using-directives, never a namespace block), so this
 *  pimpl-carrying header never pulls in libnvcuvid or libav. */
struct nvdec_state_t;

namespace unum::uform::loaders {

/** The host-side worker pool the media decoders fan their host-CPU work across — one per run, owned at the
 *  composition root (the @b only sanctioned host-thread pool), borrowed here. */
using cpu_decode_pool_t = ashvardanian::forkunion::flat_pool_t;

/** The writable @b [rows,columns,tube_frames,patch_side,patch_side,channels] tube grid of one batch's video
 *  rows — the decode's destination. Each drawn clip emits one segment @b per representation, and every segment
 *  scatters its @c temporal_groups·patch_rows·patch_columns tubes into a column sub-range of one row. */
using media::pixel_tubes_span_t;

/** The baked knobs of the @ref nvdec_session_t — the engine budget and each decoder's surface ceiling. */
struct nvdec_config_t {

    /** Concurrent decode engines to stand up — @b required (0 is rejected). Each engine owns a per-codec
     *  parser / decoder entry table with its own surface pool, so this is not free to over-provision. Set it to
     *  the target GPU's physical NVDEC engine count; there is no runtime query for that count (NVML exposes only
     *  decoder utilization), so the caller — which knows the deployment GPU — must state it. */
    u32_t engine_count = 0;

    /** Upper bound on the frame side any engine's NVDEC decoder scales to — the decode-surface budget. */
    u32_t max_side = 0;
};

/** The multi-engine NVDEC RAII context: @c engine_count decode engines (each a per-codec parser / decoder
 *  entry table) plus the shared context lock — made once at the composition root with a current context on the
 *  target device, threaded by reference. The engines own no streams; decode rides the caller's stream. Move-only;
 *  all vendor state behind a pimpl. It @b borrows the run's CPU pool per call (for demux) rather than owning one. */
struct nvdec_session_t {

  private:
    nvdec_session_t() noexcept;

  public:
    nvdec_session_t(nvdec_session_t const &) = delete;
    nvdec_session_t &operator=(nvdec_session_t const &) = delete;
    nvdec_session_t(nvdec_session_t &&) noexcept;
    nvdec_session_t &operator=(nvdec_session_t &&) noexcept;
    ~nvdec_session_t() noexcept;

    /** Stand up the engines' per-codec NVDEC parser / decoder entries and the shared context lock — once, at
     *  setup, with a current context on the target device. No streams: decode rides the caller's stream. */
    [[nodiscard]] static expected<nvdec_session_t> make(nvdec_config_t config) noexcept;

    /**
     *  @brief Decode the drawn @p clips into the @p videos tube grid, one segment per representation of every clip.
     *
     *  Demux @p clips (parallel, over @p pool — the pool's only job here), pack one segment @b per
     *  representation of every clip into the token-cursor geometry, derive each segment's
     *  @c (temporal_groups,patch_rows,patch_cols) from the clip's decoded native resolution, NVDEC-decode + scale
     *  to that grid across the engines round-robin from a single host thread, and scatter the frames into @p videos
     *  at each segment's @c [column_begin,column_begin+len) placement; write each segment's factorization into
     *  @p video_shapes . All GPU work rides @p stream .
     *
     *  @param[inout] pool The run's CPU pool, borrowed for the host-bound demux only — it never drives the decode.
     *  @param[in] clips The drawn clips' container bytes as a host-pinned tape; @c [0,clips.size()) clips.
     *  @param[in] representations The flat @ref video_batch_shape_t representations; each clip emits one segment per
     *      representation. Must be non-empty (the loader defaults it to one at @c frame_stride 1).
     *  @param[out] videos @b [rows,columns,tube_frames,P,P,C] destination grid; segments scatter across its rows.
     *  @param[out] video_shapes @b [max_segments,3] per-segment factorization.
     *  @param[out] segment_offsets @b [max_segments+1] , @param[out] segment_lengths / @param[out] segment_modality
     *      @b [max_segments] — the token cursor this fill writes: @c clips.size()·representations.size() live video
     *      segments packed row-safe, the rest PAD.
     *  @param[inout] stream The one stream every decode, conversion, and tile enqueues on.
     */
    [[nodiscard]] expected_t decode_into(cpu_decode_pool_t &pool, column_view_t clips,
                                         std::span<video_batch_shape_t const> representations,
                                         pixel_tubes_span_t videos, video_shape_span_t video_shapes,
                                         u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                         modality_span_t segment_modality, gpu_stream_t &stream) noexcept;

  private:
    explicit nvdec_session_t(std::unique_ptr<nvdec_state_t> state) noexcept;
    std::unique_ptr<nvdec_state_t> state_;
};

/**
 *  @brief Decode a batch's drawn clip rows into its @c videos tube grid — the video decode every video loader
 *      shares (Parquet today, tarball next). Hops each drawn clip's container bytes off the device @p tape to a
 *      pinned tape, then hands them to @p nvdec 's multi-engine @ref nvdec_session_t::decode_into .
 *
 *  @param[inout] nvdec The multi-engine NVDEC context doing the demux + decode + tile.
 *  @param[inout] pool The run's CPU pool, forwarded for the parallel demux and engine drive.
 *  @param[in] tape The pooled clip column with its offsets on host — device bytes, host-indexable via @c tape[row] .
 *  @param[in] drawn_rows The tape rows to decode, one clip each; each clip emits one segment per representation.
 *  @param[in] representations The flat @ref video_batch_shape_t representations; must be non-empty.
 *  @param[out] videos The @b [rows,columns,tube_frames,P,P,C] tube grid to fill; segments scatter across its rows.
 *  @param[out] video_shapes The @b [max_segments,3] per-segment factorization to fill.
 *  @param[out] segment_offsets / @param[out] segment_lengths / @param[out] segment_modality The batch's token
 *      cursor this fill writes for its video segments.
 *  @param[inout] stream The stream the staging and tiling enqueue on.
 */
[[nodiscard]] expected_t decode_videos(nvdec_session_t &nvdec, cpu_decode_pool_t &pool, column_view_t tape,
                                       u64r1_view_t drawn_rows, std::span<video_batch_shape_t const> representations,
                                       pixel_tubes_span_t videos, video_shape_span_t video_shapes,
                                       u32r1_span_t segment_offsets, u32r1_span_t segment_lengths,
                                       modality_span_t segment_modality, gpu_stream_t &stream) noexcept;

/** The Parquet video loader's baked knob — just which column; the video geometry travels on the loader's
 *  @c video_batch_shape_t (fixed at @ref video_loader_parquet_t::make ), not here, so the shape stays the single
 *  source of a step's geometry. */
struct video_loader_parquet_config_t {

    /** The Parquet column holding the encoded clip blobs ( @c list<uint8> , mp4 / webm bytes). */
    sz::string_view_t clip_column = sz::string_view_t {"video", 5};
};

/** The loader object: dataset-scoped, rank-agnostic, GPU-accelerated — the video sibling of
 *  @ref image_loader_parquet_t . Holds one dataset's shard footers; keep one per input dataset. @ref fill
 *  allocates its scratch on demand and is safe to call concurrently for different GPU ranks on different streams
 *  (each @ref fill borrows its own engines through the @p decoder it is handed). */
struct video_loader_parquet_t {

    /**
     *  @brief Failure-safe exception-free constructor that indexes all the shards' footers (host, once).
     *
     *  @param[inout] session GPU Direct Storage session for IO.
     *  @param[in] path_or_glob Dataset's Parquet paths or a GLOB pattern for all the files.
     *  @param[in] seed The training-run specific order-affecting seed.
     *  @param[inout] pool The run's CPU-decode worker pool, owned at the composition root and borrowed here.
     *  @param[in] video The video modality's own geometry slice ( @c tube_frames , @c patch_side , @c frame_stride )
     *      — the single source of a step's video geometry; it never sees the sibling modalities' shapes.
     */
    [[nodiscard]] static expected<video_loader_parquet_t> make(gpu_direct_storage_session_t &session,
                                                               sz::string_view_t path_or_glob, u64_t seed,
                                                               cpu_decode_pool_t &pool,
                                                               video_batch_shape_t video) noexcept;

    /**
     *  @brief Fill the next few @p batches ' @c videos tube grid on one GPU rank, packing each drawn clip's segment
     *      (one per representation of the loader's @c video_batch_shape_t ) into the shared @c [rows,columns] cursor.
     *  @sa multimodal_batch_mixer_t for the subsequent cross-GPU mixing.
     *
     *  @param[inout] gpu_stream The GPU-specific stream the staging, decode, and segmenting run on.
     *  @param[inout] decoder The multi-engine NVDEC context to decode through.
     *  @param[in] config Which Parquet column holds the clips.
     *  @param[in] gpu_rank Ensures different GPUs draw different rows from the same dataset.
     *  @param[in] training_step Ensures subsequent steps sample different data.
     *  @param[out] batches Output batches whose @c videos grid, @c video_shapes , and segment geometry get populated.
     */
    [[nodiscard]] expected_t fill(gpu_stream_t &gpu_stream, nvdec_session_t &decoder,
                                  video_loader_parquet_config_t config, u32_t gpu_rank, u32_t training_step,
                                  std::span<multimodal_batch_on_gpu_t> batches) noexcept;

  private:

    /** The configured KvikIO / GDS storage session — borrowed, the reader's precondition. */
    gpu_direct_storage_session_t *storage_ = nullptr;

    /** The run's CPU-decode worker pool — borrowed from the composition root, forwarded to the decode. */
    cpu_decode_pool_t *pool_ = nullptr;

    /** The dataset's Parquet shard paths, sorted by the glob — owned. */
    st::basic_vector<sz::string_t> absolute_paths_ {};

    /** The run seed keying the random file and row draws. */
    u64_t seed_ = 0;

    /** @b [shards+1] cumulative footer row counts — the weighted file draw's search table and, at its back,
     *  the dataset's total rows. Read once per dataset; shared across every GPU the loader serves. */
    st::basic_vector<std::size_t> row_prefix_ {};

    /** The video modality's geometry ( @c tube_frames , @c patch_side , @c frame_stride ) every @ref fill
     *  sources its one representation from — fixed at @ref make . */
    video_batch_shape_t video_ {};
};

} // namespace unum::uform::loaders
