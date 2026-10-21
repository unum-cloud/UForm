/**
 *  @brief Conformance suite for the media data-loaders and the batch's coordinate adapter — one binary,
 *      four tiers.
 *  @file test/loaders.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  Tier A is host-only: the keyed draw and the weighted file pick. Tier B is a host oracle
 *  of the multi-segment-per-row packing cursor. Tier C is on-device with no external media: a fabricated
 *  mixed row driven through @ref fill_coords and checked against a per-modality coordinate oracle. Tier D
 *  is on-device and env-driven: audio / image / video decoded into a real @ref multimodal_batch_on_gpu_t
 *  and read back against the segment contract, plus decode fault-isolation and a real mixed packet.
 *
 *  Every decode tier takes its media through an environment variable ( @c UFORM_TEST_AUDIO ,
 *  @c UFORM_TEST_IMAGE , @c UFORM_TEST_VIDEO ), each a comma-separated path list, so CI can point the same
 *  suite at arbitrarily-shaped real media: the decode assertions are @b shape-agnostic — the modality tag,
 *  the segment geometry, the factorization products, and finite content — never a baked resolution. With no
 *  fixture set, a decode tier simply skips; there is no embedded stand-in media in the suite.
 */
#include <cstddef> // `std::size_t`
#include <cstdlib> // `std::getenv`
#include <cstdio>  // `std::fopen` — reading an optional media fixture
#include <cmath>   // `std::isfinite`

#include <sys/stat.h> // `mkdir` — the glob fixture
#include <unistd.h>   // `symlink`, `unlink`, `rmdir`

#include <utility>   // `std::move`
#include <algorithm> // `std::min`, `std::max`
#include <span>      // `std::span`

#include <cuda.h> // `cuInit`, `cuStreamSynchronize`

#include "core/types.hpp"             // `splitmix64_hash_t`, `epoch_permutation`, `data_epoch_t`, `to_string`
#include "core/array.hpp"             // `stack_array` — the host-oracle and staging scratch
#include "core/gpu_allocators.hpp"    // `gpu_arena_t`, `gpu_arena_alloc_t`, `pinned_alloc`, the tensor aliases
#include "core/schedule.cuh"          // `gpu_stream_t`, `copy_host_to_device`, `copy_device_to_host`
#include "core/mesh.hpp"              // `mesh_topology_t`, `host_domain_index_from_environment`
#include "core/multimodal_batch.cuh"  // `multimodal_batch_on_gpu_t`, the shape, `fill_coords`, `modality_t`
#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "loaders/dataset_shards.hpp" // `column_view_t`
#include "loaders/tokens_loader.cuh"  // `tokens_loader_parquet_t`, `drawn_shards`
#include "loaders/audio_loader.cuh"   // `decode_audios`, `cpu_decode_pool_t`
#include "loaders/image_loader.cuh"   // `nvjpeg_session_t`, `decode_images`
#include "loaders/video_loader.cuh"   // `nvdec_session_t`, `decode_videos`
#include "harness.hpp"                // `test_suite_t`

using namespace unum::uform::core;
using namespace unum::uform::loaders;
using namespace unum::uform::test;
namespace um = unum::uform::media;

/** The loaders' file-local keyed-draw salts, pinned here as a frozen copy (like Tier A's finalize
 *  golden): a changed loader salt diverges these host oracles, catching a silent reshuffle of the draw. */
inline constexpr u64_t file_draw_salt_k = 0x9E3779B97F4A7C15ull;

/** The frozen row-draw salt — the loaders' per-row keyed-permutation constant, mirrored for Tier A. */
inline constexpr u64_t row_draw_salt_k = 0xD1B54A32D192ED03ull;

using i16r2_tensor_pinned_t = nk::tensor<i16_t, pinned_alloc<i16_t>, 2>;
using u16r2_tensor_pinned_t = nk::tensor<u16_t, pinned_alloc<u16_t>, 2>;
using f32r2_tensor_pinned_t = nk::tensor<f32_t, pinned_alloc<f32_t>, 2>;
using modr1_tensor_pinned_t = nk::tensor<modality_t, pinned_alloc<modality_t>, 1>;
using pixelr2_tensor_pinned_t = nk::tensor<pixel_t, pinned_alloc<pixel_t>, 2>;
using pixelr1_tensor_pinned_t = nk::tensor<pixel_t, pinned_alloc<pixel_t>, 1>;

// Fixtures: a fabricated device tape and the environment-variable media reader.

/**
 *  @brief An owning fabricated device tape: the concatenated blob bytes staged onto the device arena plus
 *      the host offsets, in the Apache Arrow packed layout @ref column_view_t reads.
 *
 *  Each blob is followed by one @c 0x00 guard byte held @b inside its packed-view element (the packed view
 *  excludes no terminator), a trailing null the media decoders ignore and a cheap margin for a bytestream
 *  parser that reads one past the end. Move-only; the arena reclaims the device bytes with its lifetime.
 */
struct fabricated_tape_t {

    /** The concatenated blob bytes on the device arena — every blob followed by its guard byte. */
    b8r1_tensor_arena_t device_bytes {};

    /** @b [blobs+1] byte offsets on the host — the Arrow offsets @ref view hands to @ref column_view_t . */
    st::basic_vector<u64_t> offsets {};

    /** The Arrow packed view a loader decodes: the device buffer with 64-bit host offsets, so
     *  @c tape[row] is a device slice computed on the host. */
    [[nodiscard]] column_view_t view() const noexcept {
        return column_view_t {device_bytes.view(), std::span<u64_t const> {offsets}};
    }

    /**
     *  @brief Pack @p blobs into a fresh device tape on @p arena , staging the host accumulator through
     *      @ref copy_host_to_device and synchronizing so the pageable accumulator is safe to drop.
     *  @param[inout] arena The device arena the concatenated bytes are allocated from.
     *  @param[inout] stream The stream the staging copy enqueues on; synchronized before returning.
     *  @param[in] blobs The blobs to pack back to back, one Arrow element each.
     */
    [[nodiscard]] static expected<fabricated_tape_t> make(gpu_arena_t &arena, gpu_stream_t &stream,
                                                          std::span<std::span<b8_t const> const> blobs) noexcept {
        using result_t = expected<fabricated_tape_t>;
        fabricated_tape_t tape;
        if (st::failed(tape.offsets.push_back(0)))
            return result_t::fail(status_t::bad_alloc_k, "tape offsets seed failed");
        st::basic_vector<b8_t> host;
        for (std::span<b8_t const> const blob : blobs) {
            for (b8_t const byte : blob)
                if (st::failed(host.push_back(b8_t {byte})))
                    return result_t::fail(status_t::bad_alloc_k, "tape byte accumulate failed");
            if (st::failed(host.push_back(b8_t {0}))) // one guard byte per blob
                return result_t::fail(status_t::bad_alloc_k, "tape guard append failed");
            if (st::failed(tape.offsets.push_back(static_cast<u64_t>(host.size()))))
                return result_t::fail(status_t::bad_alloc_k, "tape offset append failed");
        }
        std::size_t const total = std::max<std::size_t>(host.size(), 1u);
        // Device memory: `uninitialized` + explicit memset — `zeros` writes through the host pointer.
        tape.device_bytes = b8r1_tensor_arena_t::uninitialized({total}, gpu_arena_alloc<b8_t> {arena}).value;
        if (!tape.device_bytes) return result_t::fail(status_t::bad_alloc_k, "tape device buffer alloc failed");
        if (CUresult reset = cuMemsetD8(device_address(tape.device_bytes.data()), 0, total); reset != CUDA_SUCCESS)
            return cuda_failure<fabricated_tape_t>(reset, "tape device buffer reset failed");
        if (expected_t staged = copy_host_to_device(std::span<b8_t const> {host}, tape.device_bytes.span(), stream);
            !staged)
            return result_t::fail(std::move(staged));
        if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
            return cuda_failure<fabricated_tape_t>(synced, "tape staging sync failed");
        return result_t {std::move(tape)};
    }
};

/**
 *  @brief Read a file's bytes into an @c st::basic_vector , or an empty one if @p path is empty or
 *      unreadable. Takes a @c sz::string_view_t subspan and null-terminates it, as an
 *      @c sz::string_t , only at the C @c fopen boundary.
 *  @param[in] path The filesystem path to read.
 */
[[nodiscard]] static st::basic_vector<b8_t> read_fixture(sz::string_view_t path) noexcept {
    st::basic_vector<b8_t> bytes;
    if (path.empty()) return bytes;
    sz::string_t owned;
    if (sz::failed(owned.assign(path))) return bytes;
    std::FILE *file = std::fopen(owned.c_str(), "rb");
    if (file == nullptr) return bytes;
    std::fseek(file, 0, SEEK_END);
    long const size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size > 0) {
        if (st::failed(bytes.resize(static_cast<std::size_t>(size)))) {
            std::fclose(file);
            return bytes;
        }
        std::size_t const read = std::fread(bytes.data(), 1u, bytes.size(), file);
        (void)bytes.resize(read);
    }
    std::fclose(file);
    return bytes;
}

/**
 *  @brief Hand each comma-separated path in environment variable @p name to @p on_path — CI points a modality
 *      at several real files so one green run proves the loader handles every input shape. Splits with
 *      StringZilla ( @c sz::string_view_t::split on the @c ',' delimiter ), no per-path allocation for the parse.
 *  @param[in] name The environment variable holding the comma-separated path list.
 *  @param[in] on_path The callback handed each non-empty path subspan.
 */
template <typename path_fn_>
static void for_each_env_input(char const *name, path_fn_ &&on_path) noexcept {
    char const *value = std::getenv(name);
    if (value == nullptr) return;
    for (sz::string_view_t path : sz::string_view_t {value}.split(","))
        if (!path.empty()) on_path(path);
}

/**
 *  @brief The bytes of the first comma-separated path in environment variable @p name — empty when the
 *      variable is unset — for the cases that need a single fixture rather than a fan-out.
 *  @param[in] name The environment variable holding the comma-separated path list.
 */
[[nodiscard]] static st::basic_vector<b8_t> first_env_fixture(char const *name) noexcept {
    char const *value = std::getenv(name);
    if (value == nullptr) return {};
    for (sz::string_view_t path : sz::string_view_t {value}.split(","))
        if (!path.empty()) return read_fixture(path);
    return {};
}

// Tier A — host-only primitives.

/** The keyed draw is stable (a splitmix64 golden fold) and a genuine permutation that reshuffles by rank. */
static void draw_primitives_are_stable(test_suite_t &suite) {
    constexpr splitmix64_hash_t hash {};
    u64_t const seed = 0xC0FFEEull;

    auto finalize = [](u64_t word) {
        word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ull;
        word = (word ^ (word >> 27)) * 0x94D049BB133111EBull;
        return word ^ (word >> 31);
    };
    u64_t const folded = hash(seed, file_draw_salt_k);
    suite.check(folded == finalize(seed + 0x9E3779B97F4A7C15ull + file_draw_salt_k),
                "two-arg hash must be the splitmix64 golden-ratio fold (got {})", folded);

    std::size_t const draw_count = 100;
    st::basic_vector<u8_t> seen;
    if (st::failed(seen.resize(draw_count))) return suite.check(false, "seen scratch alloc failed");
    bool bijection = true;
    for (std::size_t index = 0; index != draw_count; ++index) {
        std::size_t const row = epoch_permutation(hash(seed, 3u, row_draw_salt_k), data_epoch_t {7u}, index,
                                                  draw_count);
        bijection = bijection && row < draw_count && seen[row] == 0;
        if (row < draw_count) seen[row] = 1;
    }
    suite.check(bijection, "row draw must be a bijection over [0,{})", draw_count);

    bool differs = false;
    for (std::size_t index = 0; index != draw_count; ++index)
        differs = differs ||
                  epoch_permutation(hash(seed, 3u, row_draw_salt_k), data_epoch_t {7u}, index, draw_count) !=
                      epoch_permutation(hash(seed, 4u, row_draw_salt_k), data_epoch_t {7u}, index, draw_count);
    suite.check(differs, "a different rank must reshuffle the draw");
}

/** A JPEG whose APP1 metadata pushes its start-of-frame past a 4 KiB window probes to a zero extent
 *  and the next marker's offset, the window there resumes the walk to the frame, and the whole blob
 *  probes in one window. */
static void jpeg_probe_resumes_past_metadata(test_suite_t &suite) {
    constexpr std::size_t app1_length_k = 6000, frame_offset_k = 4 + app1_length_k, window_k = 4096;
    st::basic_vector<b8_t> blob;
    if (st::failed(blob.resize(frame_offset_k + 19))) return suite.check(false, "blob alloc failed");
    u8_t const head[6] = {0xFF, 0xD8, 0xFF, 0xE1, app1_length_k >> 8, app1_length_k & 0xFF};
    u8_t const frame[9] = {0xFF, 0xC0, 0x00, 0x11, 8, 480 >> 8, 480 & 0xFF, 640 >> 8, 640 & 0xFF};
    for (std::size_t index = 0; index != 6; ++index) blob[index] = b8_t {head[index]};
    for (std::size_t index = 0; index != 9; ++index) blob[frame_offset_k + index] = b8_t {frame[index]};

    std::span<b8_t const> const bytes = blob;
    auto const prefix = um::probe_image(bytes.first(window_k));
    suite.check(prefix && prefix->extent.width == 0 && prefix->resume_offset == frame_offset_k,
                "a prefix short of the frame resumes at offset {} (got {})", frame_offset_k,
                prefix ? prefix->resume_offset : 0);
    auto const resumed = um::probe_image(bytes.subspan(frame_offset_k), frame_offset_k);
    suite.check(resumed && resumed->extent.width == 640 && resumed->extent.height == 480,
                "the resumed window finds the 640x480 frame");
    auto const whole = um::probe_image(bytes);
    suite.check(whole && whole->extent.width == 640, "the whole blob probes in one window");
}

/** `**` spans zero and two directory levels, a symlinked directory under `*` matches, and a
 *  trailing `**` is refused. */
static void glob_spans_depths_and_symlinks(test_suite_t &suite) {
    char root[] = "/tmp/uform-glob-XXXXXX";
    if (::mkdtemp(root) == nullptr) return suite.check(false, "temporary directory creation failed");
    auto const path = [&](char const *suffix) {
        sz::string_t joined;
        if (sz::failed(joined.assign(sz::string_view_t {root})) ||
            sz::failed(joined.append(sz::string_view_t {suffix})))
            joined.clear();
        return joined;
    };
    bool const made = ::mkdir(path("/a").c_str(), 0700) == 0 && ::mkdir(path("/a/b").c_str(), 0700) == 0 &&
                      ::mkdir(path("/real").c_str(), 0700) == 0 &&
                      ::symlink(path("/real").c_str(), path("/link").c_str()) == 0;
    for (char const *file : {"/top.parquet", "/a/b/deep.parquet", "/real/shard.parquet"})
        if (std::FILE *handle = std::fopen(path(file).c_str(), "w")) std::fclose(handle);
    if (!made) return suite.check(false, "glob fixture creation failed");

    auto const recursive = expand_glob(path("/**/*.parquet").view());
    std::size_t const recursive_count = recursive ? recursive->size() : 0;
    suite.check(recursive_count == 3, "** matches the top, the two-level and the real directory's shards (got {})",
                recursive_count);
    auto const linked = expand_glob(path("/l*/shard.parquet").view());
    suite.check(linked && linked->size() == 1, "a symlinked directory under * matches");
    suite.check(!expand_glob(path("/**").view()), "a trailing ** is refused");
    for (char const *file : {"/top.parquet", "/a/b/deep.parquet", "/real/shard.parquet", "/link"})
        ::unlink(path(file).c_str());
    for (char const *directory : {"/a/b", "/a", "/real", ""}) ::rmdir(path(directory).c_str());
}

/** The weighted file pick is row-proportional, distinct, and floored at two shards. */
static void file_pick_is_weighted_and_distinct(test_suite_t &suite) {
    constexpr splitmix64_hash_t hash {};
    u64_t const seed = 0x515Eull;
    stack_array<std::size_t, 5> rows; // file 2 owns ~85% of the rows
    for (std::size_t count : {std::size_t {10}, std::size_t {5}, std::size_t {200}, std::size_t {20}, std::size_t {5}})
        (void)rows.push_back(count);
    st::basic_vector<std::size_t> prefix;
    if (st::failed(prefix.push_back(0))) return suite.check(false, "prefix scratch alloc failed");
    for (std::size_t r = 0; r != rows.size(); ++r)
        if (st::failed(prefix.push_back(prefix[r] + rows[r]))) return suite.check(false, "prefix scratch alloc failed");
    std::size_t const total_rows = prefix[rows.size()];

    st::basic_vector<std::size_t> hits;
    if (st::failed(hits.resize(rows.size()))) return suite.check(false, "hits scratch alloc failed");
    std::size_t const steps = 4000;
    bool floored = true, distinct = true;
    for (std::size_t step = 0; step != steps; ++step) {
        stack_array<std::size_t, 5> picked;
        std::size_t attempt = 0;
        while (picked.size() < 2 && picked.size() < rows.size()) {
            std::size_t const position = hash(seed, step, 0u, file_draw_salt_k, attempt) % total_rows;
            ++attempt;
            std::size_t file = 0;
            while (file + 1 < prefix.size() && prefix[file + 1] <= position) ++file; // upper_bound - 1
            bool already = false;
            for (std::size_t p = 0; p != picked.size(); ++p) already = already || picked[p] == file;
            if (already) continue;
            (void)picked.push_back(file);
        }
        floored = floored && picked.size() >= 2;
        distinct = distinct && (picked.size() < 2 || picked[0] != picked[1]);
        for (std::size_t p = 0; p != picked.size(); ++p) ++hits[picked[p]];
    }
    suite.check(floored, "every pick must draw at least two shards");
    suite.check(distinct, "the drawn shards must be distinct");
    suite.check(hits[2] > hits[1] * 5, "the pick must be weighted by row count (file2={} file1={})", hits[2], hits[1]);
}

// Tier B — the multi-segment-per-row packing cursor, host oracle of the shared token-cursor scan every media
// loader now obeys: each `[offset,offset+len)` stays in one row, pre-jump gaps appear, PAD repeats the cursor.

/** The packing cursor keeps every segment in one row, packs several per row, jumps before a crossing,
 *  and repeats the cursor with zero length on PAD — the host oracle every media loader's scan mirrors. */
static void segment_geometry_packs_multiple_per_row(test_suite_t &suite) {
    u32_t const columns = 8u, rows = 3u, capacity = rows * columns, total_segments = 8u;
    stack_array<u64_t, 5> item_bytes; // truncated to the row width, packed back to back
    for (u64_t count : {u64_t {3}, u64_t {4}, u64_t {2}, u64_t {5}, u64_t {6}}) (void)item_bytes.push_back(count);
    u32_t const live_segments = static_cast<u32_t>(item_bytes.size());

    st::basic_vector<u32_t> offsets, lengths;
    if (st::failed(offsets.resize(total_segments + 1u)) || st::failed(lengths.resize(total_segments)))
        return suite.check(false, "segment scratch alloc failed");
    u32_t cursor = 0, jumps = 0;
    for (std::size_t index = 0; index != total_segments; ++index) {
        u32_t columns_here = 0;
        if (index < live_segments) columns_here = static_cast<u32_t>(std::min<u64_t>(item_bytes[index], columns));
        u32_t const room_in_row = columns - cursor % columns; // a segment never crosses its row
        if (columns_here > room_in_row) {
            cursor += room_in_row;
            ++jumps;
        }
        if (cursor + columns_here > capacity) columns_here = 0; // overflow-drop (unreached here)
        offsets[index] = cursor;
        lengths[index] = columns_here;
        cursor += columns_here;
    }
    offsets[total_segments] = cursor;

    bool row_safe = true, pad_ok = true;
    u32_t shared_rows = 0;
    for (std::size_t index = 0; index != total_segments; ++index) {
        if (lengths[index] != 0u) {
            u32_t const first = offsets[index] / columns, last = (offsets[index] + lengths[index] - 1u) / columns;
            row_safe = row_safe && first == last;
            if (index > 0 && lengths[index - 1] != 0u && offsets[index] / columns == offsets[index - 1] / columns)
                ++shared_rows;
        }
        else pad_ok = pad_ok && offsets[index] == cursor;
    }
    suite.check(row_safe, "every live segment's [offset,offset+len) must stay within one row of {} columns", columns);
    suite.check(shared_rows >= 1u, "several items must pack back to back into one row (shared adjacencies={})",
                shared_rows);
    suite.check(jumps >= 1u, "an item that would cross a row must jump the cursor first (jumps={})", jumps);
    suite.check(offsets[total_segments] <= capacity, "the pack cursor must never exceed the {}-cell grid", capacity);
    suite.check(pad_ok, "PAD segments must repeat the live cursor with zero length");
    suite.check(offsets[total_segments] == cursor, "the terminal offset must close the last live segment");
}

// Tier C / D — on-device decode into a real batch. Every helper allocates the actual
// `multimodal_batch_on_gpu_t::make` batch (exercising the real allocation + zero-gating), decodes a blob or
// fabricates a segment table, and asserts the segment contract shape-agnostically, so an external `UFORM_TEST_*`
// fixture of any resolution passes.

/**
 *  @brief The batch's positional crux: fabricate a row mixing text + image + video + audio segments (with
 *      their shape descriptors), stage them onto a real batch, run @ref fill_coords , and check every cell's
 *      @c (t,h,w) against a host oracle.
 *
 *  Text / audio walk the 1-D position axis, image is 2-D spatial, video is 3-D absolute-frame.
 *  Needs no external media.
 *
 *  @param[inout] suite The running tally.
 *  @param[inout] arena The device arena the batch and coordinate scratch are allocated from.
 *  @param[inout] stream The stream every stage enqueues on.
 */
static void fill_coords_routes_every_modality(test_suite_t &suite, gpu_arena_t &arena, gpu_stream_t &stream) {
    u32_t const columns = 32u, max_segments = 8u, tube_frames = 2u, frame_stride = 3u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 1u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.video.tube_frames = tube_frames; // image/video patch sides stay 0 — fill_coords reads only descriptors
    shape.video.frame_stride = frame_stride;
    auto made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
    if (!made) return suite.check(false, "coords batch make: {}", to_string(made.status_));
    multimodal_batch_on_gpu_t batch = std::move(made.value_);

    // Four packed segments in one row: text(3), image(6 = 2×3), video(8 = tg2·pr2·pc2), audio(4).
    u32r1_tensor_pinned_t offsets = u32r1_tensor_pinned_t::zeros({max_segments + 1u}, pinned_alloc<u32_t> {}).value;
    u32r1_tensor_pinned_t lengths = u32r1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<u32_t> {}).value;
    modr1_tensor_pinned_t modality = modr1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<modality_t> {}).value;
    u16r2_tensor_pinned_t image_shapes = u16r2_tensor_pinned_t::zeros({max_segments, 2u}, pinned_alloc<u16_t> {}).value;
    u16r2_tensor_pinned_t video_shapes = u16r2_tensor_pinned_t::zeros({max_segments, 3u}, pinned_alloc<u16_t> {}).value;
    if (!offsets || !lengths || !modality || !image_shapes || !video_shapes)
        return suite.check(false, "coords staging alloc failed");
    u32_t const live_offsets[] = {0u, 3u, 9u, 17u, 21u, 21u, 21u, 21u, 21u};
    u32_t const live_lengths[] = {3u, 6u, 8u, 4u, 0u, 0u, 0u, 0u};
    modality_t const live_modality[] = {modality_t::text_k,  modality_t::image_k, modality_t::video_k,
                                        modality_t::audio_k, modality_t::text_k,  modality_t::text_k,
                                        modality_t::text_k,  modality_t::text_k};
    for (std::size_t s = 0; s != max_segments + 1u; ++s) offsets[s] = live_offsets[s];
    for (std::size_t s = 0; s != max_segments; ++s) {
        lengths[s] = live_lengths[s];
        modality[s] = live_modality[s];
    }
    image_shapes[1, 0] = 2; // segment 1: patch_rows=2, patch_cols=3
    image_shapes[1, 1] = 3;
    video_shapes[2, 0] = 2; // segment 2: tg=2, pr=2, pc=2
    video_shapes[2, 1] = 2;
    video_shapes[2, 2] = 2;

    if (!copy_host_to_device(offsets.view(), batch.segment_offsets.span(), stream) ||
        !copy_host_to_device(lengths.view(), batch.segment_lengths.span(), stream) ||
        !copy_host_to_device(modality.view(), batch.segment_modality.span(), stream) ||
        !copy_host_to_device(image_shapes.view().flatten<1>(), batch.image_shapes.span().flatten<1>(), stream) ||
        !copy_host_to_device(video_shapes.view().flatten<1>(), batch.video_shapes.span().flatten<1>(), stream))
        return suite.check(false, "coords staging copy failed");

    std::size_t const tokens = columns; // one row
    // Device memory: `uninitialized` + explicit memset — `zeros` writes through the host pointer.
    i16r2_tensor_arena_t coords =
        i16r2_tensor_arena_t::uninitialized({tokens, 3u}, gpu_arena_alloc<i16_t> {arena}).value;
    if (!coords) return suite.check(false, "coords scratch alloc failed");
    if (cuMemsetD8(device_address(coords.data()), 0, tokens * 3u * sizeof(i16_t)) != CUDA_SUCCESS)
        return suite.check(false, "coords scratch reset failed");
    expected_t filled = fill_coords(batch.segment_offsets.span(), batch.segment_lengths.span(),
                                    batch.segment_modality.span(), batch.image_shapes.view(), batch.video_shapes.view(),
                                    tube_frames, frame_stride, coords.span(), stream);
    suite.check(static_cast<bool>(filled), "fill_coords should succeed: {}", to_string(filled.status_));
    if (!filled) return;

    i16r2_tensor_pinned_t host = i16r2_tensor_pinned_t::zeros({tokens, 3u}, pinned_alloc<i16_t> {}).value;
    if (!host) return suite.check(false, "coords readback alloc failed");
    if (!copy_device_to_host(coords.view().flatten<1>(), host.span().flatten<1>(), stream))
        return suite.check(false, "coords readback copy failed");
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return suite.check(false, "coords readback sync failed");

    bool ok = true;
    for (std::size_t s = 0; s != max_segments; ++s)
        for (std::size_t k = 0; k != lengths[s]; ++k) {
            i16_t want_t = 0, want_h = 0, want_w = 0;
            switch (modality[s]) {
            case modality_t::text_k:
            case modality_t::audio_k: want_t = static_cast<i16_t>(k); break; // 1-D position, like a token
            case modality_t::image_k: {
                u32_t const pc = image_shapes[s, 1];
                want_h = static_cast<i16_t>(k / pc);
                want_w = static_cast<i16_t>(k % pc);
                break;
            }
            case modality_t::video_k: {
                u32_t const pc = video_shapes[s, 2], gpf = static_cast<u32_t>(video_shapes[s, 1]) * pc;
                want_t = static_cast<i16_t>((k / gpf) * tube_frames * frame_stride); // the tube's first source frame
                want_h = static_cast<i16_t>((k % gpf) / pc);
                want_w = static_cast<i16_t>((k % gpf) % pc);
                break;
            }
            }
            std::size_t const token = offsets[s] + k;
            ok = ok && host[token, 0] == want_t && host[token, 1] == want_h && host[token, 2] == want_w;
        }
    suite.check(ok, "every cell's (t,h,w) must match the per-modality coordinate oracle");
}

/** `pack_tokens` writes the prefix, then the text, and counts both in the segment's length. */
static void pack_tokens_starts_with_prefix(test_suite_t &suite, gpu_arena_t &arena, gpu_stream_t &stream) {
    char const text[] = "abcde", prefix_text[] = "q:";
    u64_t const host_offsets[] = {0, 3, 5}, host_elements[] = {1, 0};
    using bytes_t = nk::tensor<b8_t, gpu_arena_alloc<b8_t>, 1>;
    using u64s_t = nk::tensor<u64_t, gpu_arena_alloc<u64_t>, 1>;
    using u32s_t = nk::tensor<u32_t, gpu_arena_alloc<u32_t>, 1>;
    auto payload = bytes_t::uninitialized({5u}, gpu_arena_alloc<b8_t> {arena}).value;
    auto prefix = bytes_t::uninitialized({2u}, gpu_arena_alloc<b8_t> {arena}).value;
    auto offsets = u64s_t::uninitialized({3u}, gpu_arena_alloc<u64_t> {arena}).value;
    auto elements = u64s_t::uninitialized({2u}, gpu_arena_alloc<u64_t> {arena}).value;
    auto tokens = nk::tensor<token_id_t, gpu_arena_alloc<token_id_t>, 2>::uninitialized(
                      {1u, 16u}, gpu_arena_alloc<token_id_t> {arena})
                      .value;
    auto segment_offsets = u32s_t::uninitialized({3u}, gpu_arena_alloc<u32_t> {arena}).value;
    auto segment_lengths = u32s_t::uninitialized({2u}, gpu_arena_alloc<u32_t> {arena}).value;
    auto segment_modality = nk::tensor<modality_t, gpu_arena_alloc<modality_t>, 1>::uninitialized(
                                {2u}, gpu_arena_alloc<modality_t> {arena})
                                .value;
    if (!payload || !prefix || !offsets || !elements || !tokens || !segment_offsets || !segment_lengths ||
        !segment_modality)
        return suite.check(false, "pack scratch alloc failed");
    bool const staged = copy_host_to_device<b8_t>(std::as_bytes(std::span {text, 5}), payload.span(), stream) &&
                        copy_host_to_device<b8_t>(std::as_bytes(std::span {prefix_text, 2}), prefix.span(), stream) &&
                        copy_host_to_device<u64_t>(std::span {host_offsets}, offsets.span(), stream) &&
                        copy_host_to_device<u64_t>(std::span {host_elements}, elements.span(), stream);
    if (!staged) return suite.check(false, "pack staging failed");
    expected_t packed = pack_tokens(column_view_t {payload.view(), offsets.view()}, elements.view(), prefix.view(),
                                    tokens.span(), segment_offsets.span(), segment_lengths.span(),
                                    segment_modality.span(), stream);
    if (!packed) return suite.check(false, "pack_tokens failed: {}", packed.description_);

    token_id_t host_tokens[16] = {};
    u32_t host_lengths[2] = {};
    bool const landed = copy_device_to_host(tokens.view().flatten<1>(), std::span {host_tokens}, stream) &&
                        copy_device_to_host(segment_lengths.view(), std::span {host_lengths}, stream) &&
                        cuStreamSynchronize(stream.get()) == CUDA_SUCCESS;
    if (!landed) return suite.check(false, "pack readback failed");
    char const expected[] = "q:deq:abc";
    bool matches = host_lengths[0] == 4u && host_lengths[1] == 5u;
    for (std::size_t index = 0; index != 9; ++index)
        matches = matches && host_tokens[index] == static_cast<token_id_t>(static_cast<u8_t>(expected[index]));
    suite.check(matches, "two segments read \"q:de\" and \"q:abc\" (lengths {} and {})", host_lengths[0],
                host_lengths[1]);
}

/**
 *  @brief Decode @p clip into a one-row audio batch and assert the raw-frame contract: one @c audio_k segment
 *      filling the row, finite samples, and a PAD @c text_k remainder.
 *  @param[inout] suite The running tally.
 *  @param[in] label The case label the diagnostics carry.
 *  @param[in] clip The encoded clip bytes; an empty clip skips the case.
 *  @param[inout] arena The device arena.
 *  @param[inout] stream The stream every stage enqueues on.
 *  @param[inout] pool The CPU-decode worker pool.
 */
static void run_audio_decode(test_suite_t &suite, sz::string_view_t label, st::basic_vector<b8_t> const &clip,
                             gpu_arena_t &arena, gpu_stream_t &stream, cpu_decode_pool_t &pool) {
    if (clip.empty()) return suite.skip("the audio fixture is empty or unreadable");
    u32_t const columns = 24u, chunk = 400u, max_segments = 4u, sample_rate = 16000u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 1u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.audio.chunk_samples = chunk;
    shape.audio.sample_rate = sample_rate;
    auto made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
    if (!made) return suite.check(false, "[{}] audio batch make: {}", label, to_string(made.status_));
    multimodal_batch_on_gpu_t batch = std::move(made.value_);

    std::span<b8_t const> blob = clip;
    auto tape_made = fabricated_tape_t::make(arena, stream, {&blob, 1});
    if (!tape_made) return suite.check(false, "[{}] audio tape: {}", label, to_string(tape_made.status_));
    fabricated_tape_t tape = std::move(tape_made.value_);
    stack_array<u64_t, 1> drawn;
    (void)drawn.push_back(0);
    audio_batch_shape_t const representation {chunk, sample_rate};
    std::span<audio_batch_shape_t const> const representations {&representation, 1};
    expected_t decoded = decode_audios(pool, tape.view(), drawn.view(), representations, batch.audio_samples.span(),
                                       batch.segment_offsets.span(), batch.segment_lengths.span(),
                                       batch.segment_modality.span(), stream);
    suite.check(static_cast<bool>(decoded), "[{}] decode_audios should succeed: {}", label, to_string(decoded.status_));
    if (!decoded) return;

    modr1_tensor_pinned_t modality = modr1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<modality_t> {}).value;
    u32r1_tensor_pinned_t lengths = u32r1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<u32_t> {}).value;
    u32r1_tensor_pinned_t offsets = u32r1_tensor_pinned_t::zeros({max_segments + 1u}, pinned_alloc<u32_t> {}).value;
    f32r1_tensor_pinned_t samples =
        f32r1_tensor_pinned_t::zeros({batch.audio_samples.numel()}, pinned_alloc<f32_t> {}).value;
    if (!modality || !lengths || !offsets || !samples) return suite.check(false, "[{}] audio readback alloc", label);
    if (!copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) ||
        !copy_device_to_host(batch.segment_lengths.view(), lengths.span(), stream) ||
        !copy_device_to_host(batch.segment_offsets.view(), offsets.span(), stream) ||
        !copy_device_to_host(batch.audio_samples.view().flatten<1>(), samples.span(), stream))
        return suite.check(false, "[{}] audio readback copy failed", label);
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return suite.check(false, "[{}] audio readback sync failed", label);

    suite.check(modality[0] == modality_t::audio_k, "[{}] first segment must be audio_k (got {})", label,
                static_cast<int>(modality[0]));
    suite.check(lengths[0] == columns, "[{}] one clip fills the row: length {} must equal columns {}", label,
                lengths[0], columns);
    suite.check(offsets[0] == 0u, "[{}] first segment must start at the row origin (got {})", label, offsets[0]);
    bool finite = true;
    std::size_t const live = std::size_t {lengths[0]} * chunk;
    for (std::size_t i = 0; i != live; ++i) finite = finite && std::isfinite(samples[i]);
    suite.check(finite, "[{}] every audio sample must be finite", label);
    bool pad = true;
    for (std::size_t s = 1; s != max_segments; ++s) pad = pad && lengths[s] == 0u && modality[s] == modality_t::text_k;
    suite.check(pad, "[{}] audio PAD segments must carry zero length and text_k", label);
    suite.check(offsets[max_segments] == lengths[0], "[{}] terminal offset must close the last segment", label);
}

/**
 *  @brief A real env audio clip and an in-test garbage blob decode into one two-row batch: mark-and-continue
 *      keeps the batch coherent — the garbage row fault-isolates to silence while both segments stay @c audio_k
 *      and the decode as a whole still succeeds.
 *  @param[inout] suite The running tally.
 *  @param[inout] arena The device arena.
 *  @param[inout] stream The stream every stage enqueues on.
 *  @param[inout] pool The CPU-decode worker pool.
 *  @param[in] clip The real env audio clip; an empty clip skips the case.
 */
static void audio_fault_isolation(test_suite_t &suite, gpu_arena_t &arena, gpu_stream_t &stream,
                                  cpu_decode_pool_t &pool, st::basic_vector<b8_t> const &clip) {
    if (clip.empty()) return suite.skip("needs a UFORM_TEST_AUDIO fixture");
    u32_t const columns = 16u, chunk = 400u, max_segments = 4u, sample_rate = 16000u;
    st::basic_vector<b8_t> garbage; // not a decodable container → silence, but not the whole batch
    if (st::failed(garbage.resize(64u))) return suite.check(false, "garbage scratch alloc failed");
    for (std::size_t i = 0; i != garbage.size(); ++i) garbage[i] = b8_t {0x7F};

    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 2u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.audio.chunk_samples = chunk;
    shape.audio.sample_rate = sample_rate;
    auto made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
    if (!made) return suite.check(false, "isolation batch make: {}", to_string(made.status_));
    multimodal_batch_on_gpu_t batch = std::move(made.value_);

    stack_array<std::span<b8_t const>, 2> blobs;
    (void)blobs.push_back(std::span<b8_t const> {clip});
    (void)blobs.push_back(std::span<b8_t const> {garbage});
    auto tape_made = fabricated_tape_t::make(arena, stream, blobs.view());
    if (!tape_made) return suite.check(false, "isolation tape: {}", to_string(tape_made.status_));
    fabricated_tape_t tape = std::move(tape_made.value_);
    stack_array<u64_t, 2> drawn;
    (void)drawn.push_back(0);
    (void)drawn.push_back(1);
    audio_batch_shape_t const representation {chunk, sample_rate};
    std::span<audio_batch_shape_t const> const representations {&representation, 1};
    expected_t decoded = decode_audios(pool, tape.view(), drawn.view(), representations, batch.audio_samples.span(),
                                       batch.segment_offsets.span(), batch.segment_lengths.span(),
                                       batch.segment_modality.span(), stream);
    suite.check(static_cast<bool>(decoded), "one corrupt clip among good must not fail the batch: {}",
                to_string(decoded.status_));
    if (!decoded) return;

    modr1_tensor_pinned_t modality = modr1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<modality_t> {}).value;
    f32r2_tensor_pinned_t samples =
        f32r2_tensor_pinned_t::zeros({2u, std::size_t {columns} * chunk}, pinned_alloc<f32_t> {}).value;
    if (!modality || !samples) return suite.check(false, "isolation readback alloc failed");
    if (!copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) ||
        !copy_device_to_host(batch.audio_samples.view().flatten<1>(), samples.span().flatten<1>(), stream))
        return suite.check(false, "isolation readback copy failed");
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return suite.check(false, "isolation readback sync failed");

    suite.check(modality[0] == modality_t::audio_k && modality[1] == modality_t::audio_k,
                "both drawn clips must be audio_k segments");
    bool garbage_silent = true;
    for (std::size_t i = 0; i != std::size_t {columns} * chunk; ++i)
        garbage_silent = garbage_silent && samples[1, i] == 0.0f; // row 1 = the corrupt clip
    suite.check(garbage_silent, "a corrupt clip must fault-isolate to silence, not corrupt its neighbour");
}

/**
 *  @brief Decode @p blob into a one-row RGBA image batch and assert: one @c image_k segment, its
 *      @c (patch_rows,patch_cols) factoring its length, sub-row placement, and a valid alpha byte.
 *  @param[inout] suite The running tally.
 *  @param[in] label The case label the diagnostics carry.
 *  @param[in] blob The encoded image bytes; an empty blob skips the case.
 *  @param[inout] arena The device arena.
 *  @param[inout] stream The stream every stage enqueues on.
 *  @param[inout] pool The CPU-decode worker pool.
 *  @param[inout] jpeg This GPU's nvJPEG session.
 */
static void run_image_decode(test_suite_t &suite, sz::string_view_t label, st::basic_vector<b8_t> const &blob,
                             gpu_arena_t &arena, gpu_stream_t &stream, cpu_decode_pool_t &pool,
                             nvjpeg_session_t &jpeg) {
    if (blob.empty()) return suite.skip("the image fixture is empty or unreadable");
    u32_t const columns = 64u, patch = 16u, channels = 4u, max_segments = 4u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 1u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.image.patch_side = patch;
    shape.image.channels = channels;
    auto made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
    if (!made) return suite.check(false, "[{}] image batch make: {}", label, to_string(made.status_));
    multimodal_batch_on_gpu_t batch = std::move(made.value_);

    std::span<b8_t const> one = blob;
    auto tape_made = fabricated_tape_t::make(arena, stream, {&one, 1});
    if (!tape_made) return suite.check(false, "[{}] image tape: {}", label, to_string(tape_made.status_));
    fabricated_tape_t tape = std::move(tape_made.value_);
    stack_array<u64_t, 1> drawn;
    (void)drawn.push_back(0);
    image_batch_shape_t const representation {patch, channels};
    std::span<image_batch_shape_t const> const representations {&representation, 1};
    expected_t decoded = decode_images(jpeg, pool, tape.view(), drawn.view(), representations, batch.images.span(),
                                       batch.image_shapes.span(), batch.segment_offsets.span(),
                                       batch.segment_lengths.span(), batch.segment_modality.span(), stream);
    suite.check(static_cast<bool>(decoded), "[{}] decode_images should succeed: {}", label, decoded.description_);
    if (!decoded) return;

    modr1_tensor_pinned_t modality = modr1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<modality_t> {}).value;
    u32r1_tensor_pinned_t lengths = u32r1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<u32_t> {}).value;
    u32r1_tensor_pinned_t offsets = u32r1_tensor_pinned_t::zeros({max_segments + 1u}, pinned_alloc<u32_t> {}).value;
    u16r2_tensor_pinned_t shapes = u16r2_tensor_pinned_t::zeros({max_segments, 2u}, pinned_alloc<u16_t> {}).value;
    pixelr2_tensor_pinned_t pixels = pixelr2_tensor_pinned_t::zeros(
                                         {batch.images.numel() / channels, std::size_t {channels}},
                                         pinned_alloc<pixel_t> {})
                                         .value;
    if (!modality || !lengths || !offsets || !shapes || !pixels)
        return suite.check(false, "[{}] image readback alloc", label);
    if (!copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) ||
        !copy_device_to_host(batch.segment_lengths.view(), lengths.span(), stream) ||
        !copy_device_to_host(batch.segment_offsets.view(), offsets.span(), stream) ||
        !copy_device_to_host(batch.image_shapes.view().flatten<1>(), shapes.span().flatten<1>(), stream) ||
        !copy_device_to_host(batch.images.view().flatten<1>(), pixels.span().flatten<1>(), stream))
        return suite.check(false, "[{}] image readback copy failed", label);
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return suite.check(false, "[{}] image readback sync failed", label);

    u32_t const patch_rows = shapes[0, 0], patch_cols = shapes[0, 1];
    suite.check(modality[0] == modality_t::image_k, "[{}] first segment must be image_k (got {})", label,
                static_cast<int>(modality[0]));
    suite.check(lengths[0] > 0u && lengths[0] <= columns, "[{}] one image is a sub-row segment (length {} of {})",
                label, lengths[0], columns);
    suite.check(patch_rows * patch_cols == lengths[0],
                "[{}] image_shapes (rows,cols)=({},{}) must multiply to the segment length {}", label, patch_rows,
                patch_cols, lengths[0]);
    suite.check(offsets[0] == 0u, "[{}] first segment must start at the row origin (got {})", label, offsets[0]);
    u32_t max_alpha = 0u;
    std::size_t const live_pixels = std::size_t {lengths[0]} * patch * patch; // cells × pixels-per-patch
    for (std::size_t p = 0; p != live_pixels; ++p) max_alpha = std::max<u32_t>(max_alpha, pixels[p, 3]); // alpha lane
    suite.check(max_alpha <= 255u, "[{}] alpha channel must be a valid byte", label); // shape-agnostic
}

/** Decode every @p blobs image in one batch, then each alone with the same shape, and require each
 *  image's patches to match bit for bit, so no image's decode writes into another's region. */
static void image_batch_matches_single(test_suite_t &suite, std::span<st::basic_vector<b8_t> const> blobs,
                                       gpu_arena_t &arena, gpu_stream_t &stream, cpu_decode_pool_t &pool,
                                       nvjpeg_session_t &jpeg) {
    if (blobs.size() < 2) return suite.skip("needs at least two UFORM_TEST_IMAGE fixtures");
    u32_t const columns = 256u, patch = 16u, channels = 4u, images = static_cast<u32_t>(blobs.size());
    std::size_t const cell_pixels = std::size_t {patch} * patch * channels;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = images;
    shape.max_segments_per_gpu = images;
    shape.max_columns_per_row = columns;
    shape.image.patch_side = patch;
    shape.image.channels = channels;
    st::basic_vector<std::span<b8_t const>> views;
    for (st::basic_vector<b8_t> const &blob : blobs)
        if (st::failed(views.push_back(std::span<b8_t const> {blob}))) return suite.check(false, "blob view append");
    auto tape_made = fabricated_tape_t::make(arena, stream, views);
    if (!tape_made) return suite.check(false, "image tape: {}", to_string(tape_made.status_));
    fabricated_tape_t tape = std::move(tape_made.value_);
    image_batch_shape_t const representation {patch, channels};

    // One decode of `drawn` into a fresh batch, landing its offsets, lengths and cell pixels.
    struct landed_t {
        u32r1_tensor_pinned_t offsets, lengths;
        pixelr1_tensor_pinned_t pixels;
    };
    auto decode = [&](std::span<u64_t const> drawn, landed_t &landed) -> bool {
        auto made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
        if (!made) return false;
        multimodal_batch_on_gpu_t batch = std::move(made.value_);
        if (!decode_images(jpeg, pool, tape.view(), drawn, {&representation, 1}, batch.images.span(),
                           batch.image_shapes.span(), batch.segment_offsets.span(), batch.segment_lengths.span(),
                           batch.segment_modality.span(), stream))
            return false;
        landed.offsets = u32r1_tensor_pinned_t::zeros({images + 1u}, pinned_alloc<u32_t> {}).value;
        landed.lengths = u32r1_tensor_pinned_t::zeros({images}, pinned_alloc<u32_t> {}).value;
        landed.pixels = pixelr1_tensor_pinned_t::zeros({batch.images.numel()}, pinned_alloc<pixel_t> {}).value;
        return landed.offsets && landed.lengths && landed.pixels &&
               copy_device_to_host(batch.segment_offsets.view(), landed.offsets.span(), stream) &&
               copy_device_to_host(batch.segment_lengths.view(), landed.lengths.span(), stream) &&
               copy_device_to_host(batch.images.view().flatten<1>(), landed.pixels.span(), stream) &&
               cuStreamSynchronize(stream.get()) == CUDA_SUCCESS;
    };
    st::basic_vector<u64_t> drawn;
    for (std::size_t row = 0; row != images; ++row)
        if (st::failed(drawn.push_back(u64_t {row}))) return suite.check(false, "drawn row append");
    landed_t together;
    if (!decode(drawn, together)) return suite.check(false, "the batched decode should succeed");
    st::basic_vector<landed_t> singles;
    for (std::size_t image = 0; image != images; ++image) {
        landed_t alone;
        if (!decode(std::span<u64_t const> {drawn}.subspan(image, 1), alone) ||
            st::failed(singles.push_back(std::move(alone))))
            return suite.check(false, "image {} decoded alone should succeed", image);
    }
    // The mean absolute pixel difference between batched `image` and single decode `single`.
    auto distance = [&](std::size_t image, std::size_t single) noexcept -> double {
        landed_t const &alone = singles[single];
        std::size_t const values = std::size_t {std::min(together.lengths[image], alone.lengths[0])} * cell_pixels;
        double total = 0;
        for (std::size_t value = 0; value != values; ++value)
            total += std::abs(double(together.pixels[std::size_t {together.offsets[image]} * cell_pixels + value]) -
                              double(alone.pixels[std::size_t {alone.offsets[0]} * cell_pixels + value]));
        return values ? total / double(values) : 0.0;
    };
    for (std::size_t image = 0; image != images; ++image) {
        double const own = distance(image, image);
        std::size_t nearest = image;
        for (std::size_t single = 0; single != images; ++single)
            if (distance(image, single) < distance(image, nearest)) nearest = single;
        suite.check(own == 0.0 && together.lengths[image] == singles[image].lengths[0],
                    "image {} of {} must decode the same in a batch as alone: mean |diff| {:.3f}, " //
                    "nearest single {} at {:.3f}, cells {} vs {}",
                    image, images, own, nearest, distance(image, nearest), together.lengths[image],
                    singles[image].lengths[0]);
    }
}

/**
 *  @brief Decode @p clip into one video batch, one representation per entry of @p frame_strides , and assert each
 *      emits a @c video_k segment whose derived @c (temporal_groups,patch_rows,patch_cols) is self-consistent and
 *      fits the grid.
 *
 *  Distinct strides yield distinct temporal factorizations. The footprint is derived from the
 *  fixture's native resolution and clamped to a row, so a fixture of any size downscales to fit.
 *
 *  @param[inout] suite The running tally.
 *  @param[in] label The case label the diagnostics carry.
 *  @param[in] clip The encoded clip bytes; an empty clip skips the case.
 *  @param[in] frame_strides One temporal stride per representation.
 *  @param[inout] arena The device arena.
 *  @param[inout] stream The stream every stage enqueues on.
 *  @param[inout] pool The CPU demux pool.
 *  @param[inout] nvdec The multi-engine NVDEC session.
 */
static void run_video_decode(test_suite_t &suite, sz::string_view_t label, st::basic_vector<b8_t> const &clip,
                             std::span<u32_t const> frame_strides, gpu_arena_t &arena, gpu_stream_t &stream,
                             cpu_decode_pool_t &pool, nvdec_session_t &nvdec) {
    if (clip.empty()) return suite.skip("the video fixture is empty or unreadable");
    u32_t const columns = 4096u, patch = 32u, channels = 3u, tube_frames = 2u, max_segments = 8u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 2u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.video.patch_side = patch;
    shape.video.channels = channels;
    shape.video.tube_frames = tube_frames;
    auto made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, shape);
    if (!made) return suite.check(false, "[{}] video batch make: {}", label, to_string(made.status_));
    multimodal_batch_on_gpu_t batch = std::move(made.value_);

    stack_array<video_batch_shape_t, 4> representations; // one flat shape per stride, sharing the grid's patch geometry
    for (u32_t stride : frame_strides)
        (void)representations.push_back(video_batch_shape_t {patch, channels, tube_frames, stride});

    std::span<b8_t const> one = clip;
    auto tape_made = fabricated_tape_t::make(arena, stream, {&one, 1});
    if (!tape_made) return suite.check(false, "[{}] video tape: {}", label, to_string(tape_made.status_));
    fabricated_tape_t tape = std::move(tape_made.value_);
    stack_array<u64_t, 1> drawn;
    (void)drawn.push_back(0);
    expected_t decoded = decode_videos(nvdec, pool, tape.view(), drawn.view(), representations.view(),
                                       batch.videos.span(), batch.video_shapes.span(), batch.segment_offsets.span(),
                                       batch.segment_lengths.span(), batch.segment_modality.span(), stream);
    suite.check(static_cast<bool>(decoded), "[{}] decode_videos should succeed: {}", label, to_string(decoded.status_));
    if (!decoded) return;

    modr1_tensor_pinned_t modality = modr1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<modality_t> {}).value;
    u32r1_tensor_pinned_t lengths = u32r1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<u32_t> {}).value;
    u16r2_tensor_pinned_t shapes = u16r2_tensor_pinned_t::zeros({max_segments, 3u}, pinned_alloc<u16_t> {}).value;
    if (!modality || !lengths || !shapes) return suite.check(false, "[{}] video readback alloc", label);
    if (!copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) ||
        !copy_device_to_host(batch.segment_lengths.view(), lengths.span(), stream) ||
        !copy_device_to_host(batch.video_shapes.view().flatten<1>(), shapes.span().flatten<1>(), stream))
        return suite.check(false, "[{}] video readback copy failed", label);
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return suite.check(false, "[{}] video readback sync failed", label);

    bool all_video = true, all_consistent = true, all_fit = true, any_live = false;
    for (std::size_t v = 0; v != representations.size(); ++v) {
        all_video = all_video && modality[v] == modality_t::video_k; // dropped or live, a drawn segment is video_k
        u32_t const product = static_cast<u32_t>(shapes[v, 0]) * shapes[v, 1] * shapes[v, 2];
        if (lengths[v] != 0u) {
            any_live = true;
            all_consistent = all_consistent && product == lengths[v];
            all_fit = all_fit && lengths[v] <= columns;
        }
    }
    suite.check(all_video, "[{}] every representation must emit a video_k segment", label);
    suite.check(all_consistent, "[{}] each live segment's (tg,pr,pc) must multiply to its length", label);
    suite.check(all_fit, "[{}] each live segment must fit one row ({} columns)", label, columns);
    suite.check(any_live, "[{}] at least one representation must produce a live segment", label);
    if (representations.size() == 2 && lengths[0] != 0u && lengths[1] != 0u)
        suite.check(shapes[0, 0] != shapes[1, 0] || shapes[0, 1] != shapes[1, 1] || shapes[0, 2] != shapes[1, 2],
                    "[{}] distinct frame strides must yield distinct factorizations", label);
}

/**
 *  @brief A real multimodal packet: decode a real audio clip and image, then pack both real segments — their
 *      true decoded lengths and the image's real @c (patch_rows,patch_cols) — into one mixed row and check
 *      @ref fill_coords routes each modality's coordinates.
 *
 *  A single loader owns the whole segment table at fill, so a mixed row is assembled here the way the
 *  cross-rank mixer produces it, over genuinely decoded geometry.
 *
 *  @param[inout] suite The running tally.
 *  @param[inout] arena The device arena.
 *  @param[inout] stream The stream every stage enqueues on.
 *  @param[inout] pool The CPU-decode worker pool.
 *  @param[inout] jpeg This GPU's nvJPEG session.
 *  @param[in] audio_clip The real env audio clip; empty skips the case.
 *  @param[in] image_blob The real env image blob; empty skips the case.
 */
static void multimodal_packet_routes(test_suite_t &suite, gpu_arena_t &arena, gpu_stream_t &stream,
                                     cpu_decode_pool_t &pool, nvjpeg_session_t &jpeg,
                                     st::basic_vector<b8_t> const &audio_clip,
                                     st::basic_vector<b8_t> const &image_blob) {
    if (audio_clip.empty() || image_blob.empty())
        return suite.skip("needs both UFORM_TEST_AUDIO and UFORM_TEST_IMAGE fixtures");

    // Decode the clip into an audio-only batch → its real segment length.
    u32_t const chunk = 400u, audio_columns = 12u;
    multimodal_batch_shape_t audio_shape {};
    audio_shape.max_rows_per_gpu = 1u;
    audio_shape.max_segments_per_gpu = 2u;
    audio_shape.max_columns_per_row = audio_columns;
    audio_shape.audio.chunk_samples = chunk;
    audio_shape.audio.sample_rate = 16000u;
    auto audio_made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, audio_shape);
    if (!audio_made) return suite.check(false, "multimodal: audio batch make: {}", to_string(audio_made.status_));
    multimodal_batch_on_gpu_t audio_batch = std::move(audio_made.value_);
    std::span<b8_t const> audio_one = audio_clip;
    auto audio_tape_made = fabricated_tape_t::make(arena, stream, {&audio_one, 1});
    if (!audio_tape_made) return suite.check(false, "multimodal: audio tape: {}", to_string(audio_tape_made.status_));
    fabricated_tape_t audio_tape = std::move(audio_tape_made.value_);
    stack_array<u64_t, 1> audio_drawn;
    (void)audio_drawn.push_back(0);
    audio_batch_shape_t const audio_representation {chunk, 16000u};
    std::span<audio_batch_shape_t const> const audio_reps {&audio_representation, 1};
    if (!decode_audios(pool, audio_tape.view(), audio_drawn.view(), audio_reps, audio_batch.audio_samples.span(),
                       audio_batch.segment_offsets.span(), audio_batch.segment_lengths.span(),
                       audio_batch.segment_modality.span(), stream))
        return suite.check(false, "multimodal: audio decode failed");

    // Decode the blob into an image-only batch → its real segment length and (patch_rows,patch_cols).
    u32_t const image_columns = 32u, patch = 16u;
    multimodal_batch_shape_t image_shape {};
    image_shape.max_rows_per_gpu = 1u;
    image_shape.max_segments_per_gpu = 2u;
    image_shape.max_columns_per_row = image_columns;
    image_shape.image.patch_side = patch;
    image_shape.image.channels = 4u;
    auto image_made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, image_shape);
    if (!image_made) return suite.check(false, "multimodal: image batch make: {}", to_string(image_made.status_));
    multimodal_batch_on_gpu_t image_batch = std::move(image_made.value_);
    std::span<b8_t const> image_one = image_blob;
    auto image_tape_made = fabricated_tape_t::make(arena, stream, {&image_one, 1});
    if (!image_tape_made) return suite.check(false, "multimodal: image tape: {}", to_string(image_tape_made.status_));
    fabricated_tape_t image_tape = std::move(image_tape_made.value_);
    stack_array<u64_t, 1> image_drawn;
    (void)image_drawn.push_back(0);
    image_batch_shape_t const image_representation {patch, 4u};
    std::span<image_batch_shape_t const> const image_reps {&image_representation, 1};
    if (!decode_images(jpeg, pool, image_tape.view(), image_drawn.view(), image_reps, image_batch.images.span(),
                       image_batch.image_shapes.span(), image_batch.segment_offsets.span(),
                       image_batch.segment_lengths.span(), image_batch.segment_modality.span(), stream))
        return suite.check(false, "multimodal: image decode failed");

    // Read both decodes' real geometry back before assembling the mixed row.
    u32r1_tensor_pinned_t audio_len = u32r1_tensor_pinned_t::zeros({2u}, pinned_alloc<u32_t> {}).value;
    u32r1_tensor_pinned_t image_len = u32r1_tensor_pinned_t::zeros({2u}, pinned_alloc<u32_t> {}).value;
    u16r2_tensor_pinned_t image_grid = u16r2_tensor_pinned_t::zeros({2u, 2u}, pinned_alloc<u16_t> {}).value;
    if (!audio_len || !image_len || !image_grid) return suite.check(false, "multimodal: geometry readback alloc");
    if (!copy_device_to_host(audio_batch.segment_lengths.view(), audio_len.span(), stream) ||
        !copy_device_to_host(image_batch.segment_lengths.view(), image_len.span(), stream) ||
        !copy_device_to_host(image_batch.image_shapes.view().flatten<1>(), image_grid.span().flatten<1>(), stream))
        return suite.check(false, "multimodal: geometry readback copy failed");
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return suite.check(false, "multimodal: geometry readback sync failed");
    u32_t const audio_length = audio_len[0], image_length = image_len[0];
    u16_t const grid_rows = image_grid[0, 0], grid_cols = image_grid[0, 1];

    // Pack both real segments into one mixed row and route their coordinates.
    u32_t const columns = 64u, max_segments = 4u, tube_frames = 2u;
    if (audio_length + image_length > columns)
        return suite.check(false, "multimodal: the two real segments must fit one row ({}+{} <= {})", audio_length,
                           image_length, columns);
    multimodal_batch_shape_t mixed_shape {};
    mixed_shape.max_rows_per_gpu = 1u;
    mixed_shape.max_segments_per_gpu = max_segments;
    mixed_shape.max_columns_per_row = columns;
    mixed_shape.video.tube_frames = tube_frames; // present so image/video shape descriptors exist
    auto mixed_made = multimodal_batch_on_gpu_t::make(gpu_arena_alloc_t {arena}, mixed_shape);
    if (!mixed_made) return suite.check(false, "multimodal: mixed batch make: {}", to_string(mixed_made.status_));
    multimodal_batch_on_gpu_t mixed = std::move(mixed_made.value_);

    u32r1_tensor_pinned_t offsets = u32r1_tensor_pinned_t::zeros({max_segments + 1u}, pinned_alloc<u32_t> {}).value;
    u32r1_tensor_pinned_t lengths = u32r1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<u32_t> {}).value;
    modr1_tensor_pinned_t modality = modr1_tensor_pinned_t::zeros({max_segments}, pinned_alloc<modality_t> {}).value;
    u16r2_tensor_pinned_t shapes = u16r2_tensor_pinned_t::zeros({max_segments, 2u}, pinned_alloc<u16_t> {}).value;
    u16r2_tensor_pinned_t video_shapes = u16r2_tensor_pinned_t::zeros({max_segments, 3u}, pinned_alloc<u16_t> {}).value;
    if (!offsets || !lengths || !modality || !shapes || !video_shapes)
        return suite.check(false, "multimodal: mixed staging alloc failed");
    for (std::size_t s = 0; s != max_segments + 1u; ++s) offsets[s] = audio_length + image_length;
    offsets[0] = 0u;
    offsets[1] = audio_length;
    lengths[0] = audio_length;
    lengths[1] = image_length;
    modality[0] = modality_t::audio_k;
    modality[1] = modality_t::image_k;
    shapes[1, 0] = grid_rows;
    shapes[1, 1] = grid_cols;
    if (!copy_host_to_device(offsets.view(), mixed.segment_offsets.span(), stream) ||
        !copy_host_to_device(lengths.view(), mixed.segment_lengths.span(), stream) ||
        !copy_host_to_device(modality.view(), mixed.segment_modality.span(), stream) ||
        !copy_host_to_device(shapes.view().flatten<1>(), mixed.image_shapes.span().flatten<1>(), stream) ||
        !copy_host_to_device(video_shapes.view().flatten<1>(), mixed.video_shapes.span().flatten<1>(), stream))
        return suite.check(false, "multimodal: mixed staging copy failed");

    std::size_t const tokens = columns;
    // Device memory: `uninitialized` + explicit memset — `zeros` writes through the host pointer.
    i16r2_tensor_arena_t coords =
        i16r2_tensor_arena_t::uninitialized({tokens, 3u}, gpu_arena_alloc<i16_t> {arena}).value;
    if (!coords) return suite.check(false, "multimodal: coords alloc failed");
    if (cuMemsetD8(device_address(coords.data()), 0, tokens * 3u * sizeof(i16_t)) != CUDA_SUCCESS)
        return suite.check(false, "multimodal: coords reset failed");
    expected_t filled = fill_coords(mixed.segment_offsets.span(), mixed.segment_lengths.span(),
                                    mixed.segment_modality.span(), mixed.image_shapes.view(), mixed.video_shapes.view(),
                                    tube_frames, mixed_shape.video.frame_stride, coords.span(), stream);
    suite.check(static_cast<bool>(filled), "multimodal: fill_coords over the mixed packet: {}",
                to_string(filled.status_));
    if (!filled) return;

    i16r2_tensor_pinned_t host = i16r2_tensor_pinned_t::zeros({tokens, 3u}, pinned_alloc<i16_t> {}).value;
    if (!host) return suite.check(false, "multimodal: coords readback alloc failed");
    if (!copy_device_to_host(coords.view().flatten<1>(), host.span().flatten<1>(), stream))
        return suite.check(false, "multimodal: coords readback copy failed");
    if (CUresult synced = cuStreamSynchronize(stream.get()); synced != CUDA_SUCCESS)
        return suite.check(false, "multimodal: coords readback sync failed");

    bool audio_ok = true, image_ok = true;
    for (std::size_t k = 0; k != audio_length; ++k) { // audio walks the 1-D position axis
        std::size_t const token = k;
        audio_ok = audio_ok && host[token, 0] == static_cast<i16_t>(k) && host[token, 1] == 0 && host[token, 2] == 0;
    }
    u32_t const pc = grid_cols;
    for (std::size_t k = 0; k != image_length; ++k) { // image is 2-D spatial over its real grid
        std::size_t const token = std::size_t {audio_length} + k;
        image_ok = image_ok && host[token, 0] == 0 && host[token, 1] == static_cast<i16_t>(k / pc) &&
                   host[token, 2] == static_cast<i16_t>(k % pc);
    }
    suite.check(audio_ok, "multimodal: the packed audio segment's real cells must route to (k,0,0)");
    suite.check(image_ok, "multimodal: the packed image segment's real cells must route to (0,row,col)");
}

/**
 *  @brief The distributed sampling tier: every rank bootstraps the mesh, draws its shard set from a shared
 *      dataset (the keyed weighted pick), publishes it to the rendezvous, and asserts its set differs from
 *      every peer's.
 *
 *  This is the cross-node "different nodes sample different files" contract the single-process oracle can
 *  only assert as arithmetic. Gated on @c UFORM_HOST_DOMAIN_INDEX ; a no-op without a @c UFORM_TEST_DATASET
 *  glob, so a laptop run skips it and a 4-8-16 launch exercises it (see @c loaders-4-8-16.sbatch ).
 */
static void cross_rank_sampling(test_suite_t &suite) {
    char const *const dataset = std::getenv("UFORM_TEST_DATASET");
    if (dataset == nullptr) return suite.skip("needs a shared UFORM_TEST_DATASET");

    u32_t const index = host_domain_index_from_environment();
    u32_t count = 0;
    for (char const *const variable : {"UFORM_HOST_DOMAIN_COUNT", "SLURM_NTASKS"})
        if (char const *const value = std::getenv(variable)) {
            count = static_cast<u32_t>(std::atoi(value));
            break;
        }
    if (count == 0) return suite.check(false, "world size should be set ($UFORM_HOST_DOMAIN_COUNT / $SLURM_NTASKS)");

    auto bootstrapped = mesh_topology_t::bootstrap(index, count);
    if (!bootstrapped) return suite.check(false, "the cluster should bootstrap: {}", bootstrapped.description_);
    mesh_topology_t topology = std::move(bootstrapped.value_);

    auto session_made = gpu_direct_storage_session_t::make();
    if (!session_made) return suite.check(false, "the GDS session should make: {}", session_made.description_);
    gpu_direct_storage_session_t session = std::move(session_made.value_);

    auto loader_made = tokens_loader_parquet_t::make(session, sz::string_view_t {dataset}, 0xC0FFEEull);
    if (!loader_made) return suite.check(false, "the tokens loader should make: {}", loader_made.description_);
    tokens_loader_parquet_t loader = std::move(loader_made.value_);

    // This rank's shard set for step 0 — 64 segments is a generous pool target that draws several shards.
    auto drawn = loader.drawn_shards(index, 0u, 64u);
    if (!drawn) return suite.check(false, "drawn_shards should succeed: {}", drawn.description_);
    sz::string_t mine;
    for (sz::string_view_t shard : drawn.value_) (void)format_into(mine, "{}\n", shard);

    if (expected_t published = topology.publish_text("shards", mine.view()); !published)
        return suite.check(false, "publishing the shard set should succeed: {}", published.description_);

    bool distinct = true;
    for (std::size_t peer = 0; peer != count; ++peer) {
        if (peer == index) continue;
        sz::string_t other;
        if (expected_t awaited = topology.await_text(static_cast<u32_t>(peer), "shards", other, 60.0); !awaited)
            distinct = false;
        else if (other.view() == mine.view()) distinct = false; // a peer drew the byte-identical shard set
    }
    suite.check(distinct, "rank {} must draw a shard set distinct from every peer", index);
    suite.check(static_cast<bool>(topology.barrier("done", 60.0)), "the closing barrier should complete");
}

int main(int, char **argv) {
    test_suite_t suite = test_suite_t::make(argv[0]);

    suite.run("draw_primitives_are_stable", draw_primitives_are_stable);
    suite.run("jpeg_probe_resumes_past_metadata", jpeg_probe_resumes_past_metadata);
    suite.run("glob_spans_depths_and_symlinks", glob_spans_depths_and_symlinks);
    suite.run("file_pick_is_weighted_and_distinct", file_pick_is_weighted_and_distinct);
    suite.run("segment_geometry_packs_multiple_per_row", segment_geometry_packs_multiple_per_row);

    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
        suite.skip("fill_coords_routes_every_modality", no_cuda_device_k);
        suite.skip("multimodal_packet_routes", no_cuda_device_k);
        suite.skip("audio_fault_isolation", no_cuda_device_k);
        return suite.finish();
    }

    if (std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr) { // the distributed sampling tier owns the process
        suite.run("cross_rank_sampling", cross_rank_sampling);
        return suite.finish();
    }

    auto stream_made = gpu_stream_t::make(0);
    if (!stream_made) return suite.check(false, "stream make: {}", stream_made.description_), suite.finish();
    gpu_stream_t stream = std::move(stream_made.value_);
    // Every decode tier below allocates pinned staging, and `cuMemAllocHost` needs a current context —
    // `gpu_stream_t::make` deliberately pushes and pops, so the suite holds the scope itself.
    switch_to_gpu_t gpu_scope {stream};
    if (!gpu_scope.ok()) return suite.check(false, "context entry should succeed"), suite.finish();
    auto arena_made = gpu_arena_t::make(0, 2ull << 30);
    if (!arena_made) return suite.check(false, "gpu arena make: {}", to_string(arena_made.status_)), suite.finish();
    gpu_arena_t arena = std::move(arena_made.value_);
    cpu_decode_pool_t pool;
    if (failed(pool.spawn(4u))) return suite.check(false, "the CPU decode pool should spawn"), suite.finish();

    // Tier C — the coordinate oracle, no external media.
    suite.run("fill_coords_routes_every_modality", fill_coords_routes_every_modality, arena, stream);
    suite.run("pack_tokens_starts_with_prefix", pack_tokens_starts_with_prefix, arena, stream);

    // Tier D — every UFORM_TEST_AUDIO clip through the audio decode contract.
    for_each_env_input("UFORM_TEST_AUDIO", [&](sz::string_view_t path) {
        suite.run("audio_decode", run_audio_decode, "env-audio", read_fixture(path), arena, stream, pool);
    });

    // Image — every UFORM_TEST_IMAGE blob, plus the real audio+image mixed packet.
    auto jpeg_made = nvjpeg_session_t::make();
    suite.check(static_cast<bool>(jpeg_made), "nvjpeg_session_t::make should succeed: {}",
                to_string(jpeg_made.status_));
    if (jpeg_made) {
        nvjpeg_session_t jpeg = std::move(jpeg_made.value_);
        st::basic_vector<st::basic_vector<b8_t>> image_blobs;
        for_each_env_input("UFORM_TEST_IMAGE", [&](sz::string_view_t path) {
            suite.run("image_decode", run_image_decode, "env-image", read_fixture(path), arena, stream, pool, jpeg);
            (void)image_blobs.push_back(read_fixture(path));
        });
        suite.run("image_batch_matches_single", image_batch_matches_single, image_blobs, arena, stream, pool, jpeg);
        suite.run("multimodal_packet_routes", multimodal_packet_routes, arena, stream, pool, jpeg,
                  first_env_fixture("UFORM_TEST_AUDIO"), first_env_fixture("UFORM_TEST_IMAGE"));
    }

    // Video — external only (a codec-bearing clip can't be embedded cheaply); each file at one representation, then
    // two representations at distinct frame strides (a dense and a sparse temporal sampling).
    auto nvdec_made = nvdec_session_t::make(nvdec_config_t {2u, 512u});
    if (nvdec_made) {
        nvdec_session_t nvdec = std::move(nvdec_made.value_);
        for_each_env_input("UFORM_TEST_VIDEO", [&](sz::string_view_t path) {
            st::basic_vector<b8_t> const clip = read_fixture(path);
            stack_array<u32_t, 1> one_stride;
            (void)one_stride.push_back(1u);
            suite.run("video_decode/one_view", run_video_decode, "video-1view", clip, one_stride.view(), arena, stream,
                      pool, nvdec);
            stack_array<u32_t, 2> two_strides;
            (void)two_strides.push_back(1u);
            (void)two_strides.push_back(4u);
            suite.run("video_decode/two_views", run_video_decode, "video-2view", clip, two_strides.view(), arena,
                      stream, pool, nvdec);
        });
    }

    // Decode fault-isolation over the first real audio fixture.
    suite.run("audio_fault_isolation", audio_fault_isolation, arena, stream, pool,
              first_env_fixture("UFORM_TEST_AUDIO"));

    return suite.finish();
}
