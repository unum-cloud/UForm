/**
 *  @file test/loaders.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Conformance suite for the media data-loaders and the batch's coordinate adapter — one
 *      binary, four tiers.
 *
 *  Tier A is host-only: the keyed draw and the weighted file pick. Tier B is a host oracle of the
 *  multi-segment-per-row packing cursor. Tier C is on-device with no external media: a fabricated
 *  mixed row driven through @ref fill_coords and checked against a per-modality coordinate oracle.
 *  Tier D is on-device and env-driven: audio / image / video decoded into a real
 *  @ref multimodal_batch_on_gpu_t and read back against the segment contract, plus decode
 *  fault-isolation and a real mixed packet.
 *
 *  Every decode tier takes its media through an environment variable ( @c UFORM_TEST_AUDIO ,
 *  @c UFORM_TEST_IMAGE , @c UFORM_TEST_VIDEO ), each a comma-separated path list, so CI can point
 *  the same suite at arbitrarily-shaped real media: the decode assertions are @b shape-agnostic —
 *  the modality tag, the segment geometry, the factorization products, and finite content — never a
 *  baked resolution. With no fixture set, a decode tier simply skips; there is no embedded stand-in
 *  media in the suite. The paired fill reads the image entries of every @c UFORM_TEST_MIXTURE
 *  mixture the same way.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdlib> // `std::getenv`
#include <cstdio>  // `std::fopen` — reading an optional media fixture
#include <cstring> // `std::memcmp` — a fill against its replay
#include <cmath>   // `std::isfinite`

#include <sys/stat.h> // `mkdir` — the glob fixture
#include <unistd.h>   // `symlink`

#include <utility> // `std::move`
#include <span>    // `std::span`

#include <cuda.h> // `cuInit`, `cuStreamSynchronize`

#include "uform/utilities/array.hpp"              // `stack_array` — the host-oracle and staging scratch
#include "uform/utilities/vram_nvidia_memory.hpp" // `gpu_arena_t`, `gpu_arena_alloc_t`, `pinned_alloc`, the tensor aliases
#include "uform/utilities/mesh.hpp"               // `mesh_topology_t`, `host_domain_index_from_environment`
#include "uform/encoder/multimodal_batch.cuh"     // `multimodal_batch_on_gpu_t`, the shape, `fill_coords`, `modality_t`
#include "uform/utilities/vram_nvidia_schedule.cuh" // `executor_stream_t`, `copy_host_to_device`, `copy_device_to_host`
#include "uform/utilities/types.hpp"        // `splitmix64_hash_t`, `epoch_permutation`, `data_epoch_t`, `to_string`
#include "uform/loaders/audio_loader.cuh"   // `decode_audios`
#include "uform/loaders/dataset_shards.hpp" // `column_view_t`, `row_group_draw_t`
#include "uform/loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "uform/loaders/image_loader.cuh"   // `nvjpeg_session_t`, `decode_images`
#include "uform/loaders/paired_loader.cuh"  // `paired_loader_parquet_t`
#include "uform/loaders/tokens_loader.cuh"  // `tokens_loader_parquet_t`, `drawn_shards`
#include "uform/loaders/video_loader.cuh"   // `nvdec_session_t`, `decode_videos`
#include "uform/media/planning.hpp"         // `image_patch_grid`, `video_tube_grid`
#include "uform/train/datasets.hpp"         // `mixture_sampler_t`, `mixture_entry_t`

#include "harness.hpp" // `run_test_case`, `expect_true`, `scratch_directory_t`

namespace ue = unum::uform::encoder;
using ue::audio_batch_shape_t;
using ue::image_batch_shape_t;
using ue::modality_t;
using ue::multimodal_batch_on_gpu_t;
using ue::multimodal_batch_shape_t;
using ue::video_batch_shape_t;
namespace util = unum::uform::utilities;
using namespace unum::uform::loaders;
using namespace unum::uform::test;
namespace um = unum::uform::media;
namespace ut = unum::uform::train;

/*  The loaders' keyed-draw salts, frozen like Tier A's finalize golden: a changed salt reshuffles
 *  every draw, so it fails this build instead of silently diverging the host oracles. */
static_assert(file_draw_salt_k == 0x9E3779B97F4A7C15ull && row_draw_salt_k == 0xD1B54A32D192ED03ull,
              "a loader draw salt changed, which reshuffles every seeded draw");

using i16r2_tensor_pinned_t = nk::tensor<util::i16_t, util::pinned_alloc<util::i16_t>, 2>;
using u16r2_tensor_pinned_t = nk::tensor<util::u16_t, util::pinned_alloc<util::u16_t>, 2>;
using f32r2_tensor_pinned_t = nk::tensor<util::f32_t, util::pinned_alloc<util::f32_t>, 2>;
using modr1_tensor_pinned_t = nk::tensor<modality_t, util::pinned_alloc<modality_t>, 1>;
using pixelr2_tensor_pinned_t = nk::tensor<util::pixel_t, util::pinned_alloc<util::pixel_t>, 2>;
using pixelr1_tensor_pinned_t = nk::tensor<util::pixel_t, util::pinned_alloc<util::pixel_t>, 1>;

// Fixtures: a fabricated device tape and the environment-variable media reader.

/**
 *  @brief An owning fabricated device tape: the concatenated blob bytes staged onto the device
 *      arena plus the host offsets, in the Apache Arrow packed layout @ref column_view_t reads.
 *
 *  Each blob is followed by one @c 0x00 guard byte held @b inside its packed-view element (the
 *  packed view excludes no terminator), a trailing null the media decoders ignore and a cheap
 *  margin for a bytestream parser that reads one past the end. Move-only; the arena reclaims the
 *  device bytes with its lifetime.
 */
struct fabricated_tape_t {

    /** The concatenated blob bytes on the device arena — every blob followed by its guard byte. */
    util::b8r1_tensor_arena_t device_bytes {};

    /** @b [blobs+1] byte offsets on the host — the Arrow offsets @ref view hands to
     *  @ref column_view_t . */
    st::basic_vector<util::u64_t> offsets {};

    /** The Arrow packed view a loader decodes: the device buffer with 64-bit host offsets, so
     *  @c tape[row] is a device slice computed on the host. */
    [[nodiscard]] column_view_t view() const noexcept {
        return column_view_t {device_bytes.view(), std::span<util::u64_t const> {offsets}};
    }

    /**
     *  @brief Pack @p blobs into a fresh device tape on @p arena , staging the host accumulator
     *      through @ref copy_host_to_device and synchronizing so the pageable accumulator is
     *      safe to drop.
     *  @param[in,out] arena The device arena the concatenated bytes are allocated from.
     *  @param[in,out] stream The stream the staging copy enqueues on;
     *      synchronized before returning.
     *  @param[in] blobs The blobs to pack back to back, one Arrow element each.
     */
    [[nodiscard]] static util::expected<fabricated_tape_t> make(util::gpu_arena_t& arena,
                                                                util::executor_stream_t& stream,
                                                                std::span<util::bytes_view_t const> blobs) noexcept {
        using result_t = util::expected<fabricated_tape_t>;
        fabricated_tape_t tape;
        if (st::failed(tape.offsets.push_back(0)))
            return result_t::fail(util::status_code_t::out_of_memory_k, "tape offsets seed failed");
        st::basic_vector<util::b8_t> host;
        for (util::bytes_view_t const blob : blobs) {
            for (util::b8_t const byte : blob)
                if (st::failed(host.push_back(util::b8_t {byte})))
                    return result_t::fail(util::status_code_t::out_of_memory_k, "tape byte accumulate failed");
            // One guard byte per blob.
            if (st::failed(host.push_back(util::b8_t {0})))
                return result_t::fail(util::status_code_t::out_of_memory_k, "tape guard append failed");
            if (st::failed(tape.offsets.push_back(static_cast<util::u64_t>(host.size()))))
                return result_t::fail(util::status_code_t::out_of_memory_k, "tape offset append failed");
        }
        std::size_t const total = fu::max_of<std::size_t>(host.size(), 1u);
        // Device memory: `uninitialized` + explicit memset — `zeros` writes through
        // the host pointer.
        tape.device_bytes = make_uninitialized<util::b8r1_tensor_arena_t>({total},
                                                                          util::gpu_arena_alloc<util::b8_t> {arena});
        if (!tape.device_bytes)
            return result_t::fail(util::status_code_t::out_of_memory_k, "tape device buffer alloc failed");
        if (CUresult reset = cuMemsetD8Async(util::device_address(tape.device_bytes.data()), 0, total,
                                             util::native_stream(stream));
            reset != CUDA_SUCCESS)
            return util::cuda_failure<fabricated_tape_t>(reset, "tape device buffer reset failed");
        if (util::expected_t staged = util::copy_host_to_device(util::bytes_view_t {host}, tape.device_bytes.span(),
                                                                stream);
            !staged)
            return result_t::fail(std::move(staged));
        if (CUresult synced = cuStreamSynchronize(util::native_stream(stream)); synced != CUDA_SUCCESS)
            return util::cuda_failure<fabricated_tape_t>(synced, "tape staging sync failed");
        return result_t {std::move(tape)};
    }
};

/**
 *  @brief Read a file's bytes into an @c st::basic_vector , or an empty one if @p path is empty or
 *      unreadable. Takes a @c sz::string_view_t subspan and null-terminates it, as an
 *      @c sz::string_t , only at the C @c fopen boundary.
 *  @param[in] path The filesystem path to read.
 */
[[nodiscard]] static st::basic_vector<util::b8_t> read_fixture(sz::string_view_t path) noexcept {
    st::basic_vector<util::b8_t> bytes;
    if (path.empty()) return bytes;
    sz::string_t owned;
    if (sz::failed(owned.assign(path))) return bytes;
    std::FILE* file = std::fopen(owned.c_str(), "rb");
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
        [[maybe_unused]] st::status_t const shrunk = bytes.resize(read);
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
static void for_each_env_input(char const* name, path_fn_&& on_path) noexcept {
    char const* value = std::getenv(name);
    if (value == nullptr) return;
    for (sz::string_view_t path : sz::string_view_t {value}.split(","))
        if (!path.empty()) on_path(path);
}

/**
 *  @brief The bytes of the first comma-separated path in environment variable @p name — empty when
 *      the variable is unset — for the cases that need a single fixture rather than a fan-out.
 *  @param[in] name The environment variable holding the comma-separated path list.
 */
[[nodiscard]] static st::basic_vector<util::b8_t> first_env_fixture(char const* name) noexcept {
    char const* value = std::getenv(name);
    if (value == nullptr) return {};
    for (sz::string_view_t path : sz::string_view_t {value}.split(","))
        if (!path.empty()) return read_fixture(path);
    return {};
}

// Tier A — host-only primitives.

/** The keyed draw is stable (a splitmix64 golden fold) and a genuine permutation that
 *  reshuffles by rank. */
static void test_draw_primitives_are_stable() {
    constexpr util::splitmix64_hash_t hash {};
    util::u64_t const seed = 0xC0FFEEull;

    auto finalize = [](util::u64_t word) {
        word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ull;
        word = (word ^ (word >> 27)) * 0x94D049BB133111EBull;
        return word ^ (word >> 31);
    };
    util::u64_t const folded = hash(seed, file_draw_salt_k);
    expect_true(folded == finalize(seed + 0x9E3779B97F4A7C15ull + file_draw_salt_k),
                "two-arg hash must be the splitmix64 golden-ratio fold (got {})", folded);

    std::size_t const draw_count = 100;
    st::basic_vector<util::u8_t> seen;
    if (!expect_true(!st::failed(seen.resize(draw_count)), "seen scratch alloc failed")) return;
    bool bijection = true;
    for (std::size_t index = 0; index != draw_count; ++index) {
        std::size_t const row = util::epoch_permutation(hash(seed, 3u, row_draw_salt_k), util::data_epoch_t {7u}, index,
                                                        draw_count);
        bijection = bijection && row < draw_count && seen[row] == 0;
        if (row < draw_count) seen[row] = 1;
    }
    expect_true(bijection, "row draw must be a bijection over [0,{})", draw_count);

    bool differs = false;
    for (std::size_t index = 0; index != draw_count; ++index)
        differs = differs || util::epoch_permutation(hash(seed, 3u, row_draw_salt_k), util::data_epoch_t {7u}, index,
                                                     draw_count) !=
                                 util::epoch_permutation(hash(seed, 4u, row_draw_salt_k), util::data_epoch_t {7u},
                                                         index, draw_count);
    expect_true(differs, "a different rank must reshuffle the draw");
}

/** A JPEG whose APP1 metadata pushes its start-of-frame past a 4 KiB window probes to a zero extent
 *  and the next marker's offset, the window there resumes the walk to the frame, and the whole blob
 *  probes in one window. */
static void test_jpeg_probe_resumes_past_metadata() {
    constexpr std::size_t app1_length_k = 6000, frame_offset_k = 4 + app1_length_k, window_k = 4096;
    st::basic_vector<util::b8_t> blob;
    if (!expect_true(!st::failed(blob.resize(frame_offset_k + 19)), "blob alloc failed")) return;
    util::u8_t const head[6] = {0xFF, 0xD8, 0xFF, 0xE1, app1_length_k >> 8, app1_length_k & 0xFF};
    util::u8_t const frame[9] = {0xFF, 0xC0, 0x00, 0x11, 8, 480 >> 8, 480 & 0xFF, 640 >> 8, 640 & 0xFF};
    for (std::size_t index = 0; index != 6; ++index) blob[index] = util::b8_t {head[index]};
    for (std::size_t index = 0; index != 9; ++index) blob[frame_offset_k + index] = util::b8_t {frame[index]};

    util::bytes_view_t const bytes = blob;
    auto [prefix, probe_prefix_status] = um::probe_image(bytes.first(window_k));
    expect_true(probe_prefix_status && prefix.extent.width == 0 && prefix.resume_offset == frame_offset_k,
                "a prefix short of the frame resumes at offset {} (got {})", frame_offset_k,
                probe_prefix_status ? prefix.resume_offset : 0);
    auto [resumption, probe_resumption_status] = um::probe_image(bytes.subspan(frame_offset_k), frame_offset_k);
    expect_true(probe_resumption_status && resumption.extent.width == 640 && resumption.extent.height == 480,
                "the resumed window finds the 640x480 frame");
    auto [whole, probe_whole_status] = um::probe_image(bytes);
    expect_true(probe_whole_status && whole.extent.width == 640, "the whole blob probes in one window");
}

/** `**` spans zero and two directory levels, a symlinked directory under `*` matches, and a
 *  trailing `**` is refused. */
static void test_glob_spans_depths_and_symlinks() {
    scratch_directory_t const scratch;
    if (!expect_true(!scratch.path.empty(), "temporary directory creation failed")) return;
    char const* const root = scratch.path.c_str();
    auto const path = [&](char const* suffix) {
        sz::string_t joined;
        if (sz::failed(joined.assign(sz::string_view_t {root})) ||
            sz::failed(joined.append(sz::string_view_t {suffix})))
            joined.clear();
        return joined;
    };
    bool const made = ::mkdir(path("/a").c_str(), 0700) == 0 && ::mkdir(path("/a/b").c_str(), 0700) == 0 &&
                      ::mkdir(path("/real").c_str(), 0700) == 0 &&
                      ::symlink(path("/real").c_str(), path("/link").c_str()) == 0;
    for (char const* file : {"/top.parquet", "/a/b/deep.parquet", "/real/shard.parquet"})
        if (std::FILE* handle = std::fopen(path(file).c_str(), "w")) std::fclose(handle);
    if (!expect_true(made, "glob fixture creation failed")) return;

    auto [recursive_matches, expand_recursive_matches_status] = expand_glob(path("/**/*.parquet").view());
    std::size_t const recursive_count = expand_recursive_matches_status ? recursive_matches.size() : 0;
    expect_true(recursive_count == 3, "** matches the top, the two-level and the real directory's shards (got {})",
                recursive_count);
    auto [symlink_matches, expand_symlink_matches_status] = expand_glob(path("/l*/shard.parquet").view());
    expect_true(expand_symlink_matches_status && symlink_matches.size() == 1, "a symlinked directory under * matches");
    expect_true(!expand_glob(path("/**").view()), "a trailing ** is refused");
}

/** An entry capped at one pass retires once the world's fills drew exactly its rows, its weight
 *  moves to its modality's live entry rather than across modalities, and a run resumed at a later
 *  epoch replays the keyed draws to the same live entries. */
static void test_mixture_sampler_retires_capped_entries() {
    auto const entry_of = [](char const* name, util::f64_t weight, std::size_t epochs, modality_t modality) {
        ut::mixture_entry_t entry;
        [[maybe_unused]] sz::status_t const named = entry.name.assign(sz::string_view_t {name});
        [[maybe_unused]] sz::status_t const anchored = entry.anchor.assign(sz::string_view_t {"caption"});
        entry.weight = weight, entry.epochs = epochs, entry.modality = modality;
        return entry;
    };
    st::basic_vector<ut::mixture_entry_t> entries;
    if (!expect_true(!st::failed(entries.push_back(entry_of("capped", 1.0, 1, modality_t::image_k))) &&
                         !st::failed(entries.push_back(entry_of("image", 1.0, 0, modality_t::image_k))) &&
                         !st::failed(entries.push_back(entry_of("text", 2.0, 0, modality_t::text_k))),
                     "entry table alloc failed"))
        return;
    // Two GPUs refilling on alternate epochs, each fill drawing two batches of two pairs.
    constexpr std::size_t world_gpus = 2, batches_per_fill = 2, segments = 4, epochs = 2000;
    constexpr std::size_t fill_pairs = batches_per_fill * (segments / 2);
    constexpr util::u64_t seed = 0xE90C5ull;
    std::size_t const rows[3] = {12, 1000, 1000};
    auto const make = [&]() noexcept -> expected<ut::mixture_sampler_t> {
        return ut::mixture_sampler_t::make(entries, rows, world_gpus, batches_per_fill, segments);
    };
    auto const refills = [](std::size_t epoch, std::size_t gpu) {
        return epoch == 0 || (epoch + gpu * batches_per_fill / world_gpus) % batches_per_fill == 0;
    };

    auto [sampler, make_sampler_status] = make();
    if (!expect_true(make_sampler_status, "the sampler should make: {}", make_sampler_status.description_)) return;
    st::basic_vector<std::size_t> drawn; // per world fill, left zero where a GPU does not refill
    if (!expect_true(!st::failed(drawn.resize(epochs * world_gpus)), "draw table alloc failed")) return;
    nk::tensor_span<std::size_t, 2> const draws {drawn.data(), epochs, world_gpus};
    std::size_t capped_fills = 0, retired_epoch = 0;
    for (std::size_t epoch = 0; epoch != epochs; ++epoch)
        for (std::size_t gpu = 0; gpu != world_gpus; ++gpu)
            if (refills(epoch, gpu)) {
                std::size_t const entry = sampler.draw(seed, util::data_epoch_t {static_cast<util::u32_t>(epoch)},
                                                       static_cast<util::u32_t>(gpu), nullptr);
                draws[epoch, gpu] = entry;
                if (entry == 0) ++capped_fills, retired_epoch = epoch;
            }
    expect_true(capped_fills * fill_pairs == rows[0], "the capped entry retires after exactly its {} rows, drew {}",
                rows[0], capped_fills * fill_pairs);

    std::size_t image_fills = 0, later_fills = 0;
    for (std::size_t epoch = retired_epoch + 1; epoch != epochs; ++epoch)
        for (std::size_t gpu = 0; gpu != world_gpus; ++gpu)
            if (refills(epoch, gpu)) ++later_fills, image_fills += draws[epoch, gpu] == 1;
    expect_true(image_fills * 10 >= later_fills * 4 && image_fills * 10 <= later_fills * 6,
                "the image entry takes the image modality's half of the draws, not a third, got {} of {}", image_fills,
                later_fills);

    for (std::size_t const resumed_at : {retired_epoch, retired_epoch + 1, epochs / 2}) {
        auto [sampler_resumed, make_sampler_resumed_status] = make();
        if (!expect_true(make_sampler_resumed_status, "the sampler should make: {}",
                         make_sampler_resumed_status.description_))
            return;
        bool agrees = true;
        for (std::size_t epoch = resumed_at; epoch != epochs; ++epoch)
            for (std::size_t gpu = 0; gpu != world_gpus; ++gpu)
                if (refills(epoch, gpu))
                    agrees &= sampler_resumed.draw(seed, util::data_epoch_t {static_cast<util::u32_t>(epoch)},
                                                   static_cast<util::u32_t>(gpu), nullptr) == draws[epoch, gpu];
        expect_true(agrees, "a run resumed at epoch {} draws as the fresh one did", resumed_at);
    }
}

/** The row-group draw is row-proportional, distinct, floored at two shards, and reads only the
 *  groups it drew, in the ascending order cuDF reads them. */
static void test_file_pick_is_weighted_and_distinct() {
    util::u64_t const seed = 0x515Eull;
    constexpr std::size_t groups_per_shard = 4;
    util::stack_array<std::size_t, 5> rows; // shard 2 owns ~85% of the rows
    for (std::size_t count :
         {std::size_t {10}, std::size_t {5}, std::size_t {200}, std::size_t {20}, std::size_t {5}}) {
        [[maybe_unused]] util::status_t const pushed = rows.push_back(count);
    }
    sz::string_view_t const names[5] = {"a", "b", "c", "d", "e"};
    row_group_draw_t draw;
    for (std::size_t shard = 0; shard != rows.size(); ++shard) {
        parquet_length_metadata_t footer;
        footer.rows = rows[shard] * groups_per_shard;
        if (!expect_true(!st::failed(footer.group_complete_rows.resize(groups_per_shard, std::size_t {rows[shard]})) &&
                             !st::failed(footer.group_rows.resize(groups_per_shard, std::size_t {rows[shard]})),
                         "footer scratch alloc failed"))
            return;
        if (util::expected_t appended = draw.append_shard(names[shard], footer);
            !expect_true(appended, "a shard should append: {}", appended.description_))
            return;
    }
    if (util::expected_t reserved = draw.reserve(rows.size() * groups_per_shard);
        !expect_true(reserved, "the pick storage should reserve: {}", reserved.description_))
        return;

    st::basic_vector<std::size_t> hits;
    if (!expect_true(!st::failed(hits.resize(rows.size())), "hits scratch alloc failed")) return;
    std::size_t const steps = 4000;
    bool floored = true, ordered = true, partial = true;
    for (std::size_t step = 0; step != steps; ++step) {
        if (util::expected_t drawn = draw.draw(seed, step, 0u, file_draw_salt_k, 1u, 2u);
            !expect_true(drawn, "a draw should succeed: {}", drawn.description_))
            return;
        sz::strings_view_t const paths = draw.picked_paths();
        util::tape_view<util::u32_t const> const groups = draw.picked_groups();
        floored = floored && paths.size() >= 2 && groups.size() == paths.size();
        partial = partial && groups.elements.size() < groups_per_shard * paths.size();
        for (std::size_t pick = 0; pick != paths.size(); ++pick) {
            ordered = ordered && (pick == 0 || paths[pick - 1] < paths[pick]);
            std::span<util::u32_t const> const local = groups[pick];
            for (std::size_t index = 1; index < local.size(); ++index)
                ordered = ordered && local[index - 1] < local[index];
            ++hits[static_cast<std::size_t>(paths[pick].front() - 'a')];
        }
    }
    expect_true(floored, "every draw must pick at least two shards");
    expect_true(ordered, "the picked shards and each shard's groups must be distinct and ascending");
    expect_true(partial, "a draw must read only the groups it drew, not whole shards");
    expect_true(hits[2] > hits[1] * 5, "the draw must be weighted by row count (shard2={} shard1={})", hits[2],
                hits[1]);
}

// Tier B — the multi-segment-per-row packing cursor, host oracle of the token loader's scan: each
// `[offset,offset+len)` stays in one row, pre-jump gaps appear, PAD repeats the cursor. Media never
// share a row.

/** The packing cursor keeps every segment in one row, packs several per row, jumps before a
 *  crossing, and repeats the cursor with zero length on PAD — the host oracle the token
 *  loader's scan mirrors. */
static void test_segment_geometry_packs_multiple_per_row() {
    util::u32_t const columns = 8u, rows = 3u, capacity = rows * columns, total_segments = 8u;
    util::stack_array<util::u64_t, 5> item_bytes; // truncated to the row width, packed back to back
    for (util::u64_t count : {util::u64_t {3}, util::u64_t {4}, util::u64_t {2}, util::u64_t {5}, util::u64_t {6}}) {
        [[maybe_unused]] util::status_t const pushed = item_bytes.push_back(count);
    }
    util::u32_t const live_segments = static_cast<util::u32_t>(item_bytes.size());

    st::basic_vector<util::u32_t> offsets, lengths;
    if (!expect_true(!st::failed(offsets.resize(total_segments + 1u)) && !st::failed(lengths.resize(total_segments)),
                     "segment scratch alloc failed"))
        return;
    util::u32_t cursor = 0, jumps = 0;
    for (std::size_t index = 0; index != total_segments; ++index) {
        util::u32_t columns_here = 0;
        if (index < live_segments)
            columns_here = static_cast<util::u32_t>(fu::min_of<util::u64_t>(item_bytes[index], columns));
        util::u32_t const room_in_row = columns - cursor % columns; // a segment never crosses its row
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
    util::u32_t shared_rows = 0;
    for (std::size_t index = 0; index != total_segments; ++index) {
        if (lengths[index] != 0u) {
            util::u32_t const first = offsets[index] / columns, last = (offsets[index] + lengths[index] - 1u) / columns;
            row_safe = row_safe && first == last;
            if (index > 0 && lengths[index - 1] != 0u && offsets[index] / columns == offsets[index - 1] / columns)
                ++shared_rows;
        }
        else pad_ok = pad_ok && offsets[index] == cursor;
    }
    expect_true(row_safe, "every live segment's [offset,offset+len) must stay within one row of {} columns", columns);
    expect_true(shared_rows >= 1u, "several items must pack back to back into one row (shared adjacencies={})",
                shared_rows);
    expect_true(jumps >= 1u, "an item that would cross a row must jump the cursor first (jumps={})", jumps);
    expect_true(offsets[total_segments] <= capacity, "the pack cursor must never exceed the {}-cell grid", capacity);
    expect_true(pad_ok, "PAD segments must repeat the live cursor with zero length");
    expect_true(offsets[total_segments] == cursor, "the terminal offset must close the last live segment");
}

/** Every image fills half a row and every clip a whole one, whatever their resolution: a 224² photo
 *  upscales to the near-square half-row grid, a 4:3 one keeps its aspect, and a short
 *  low-resolution clip grows its frame. */
static void test_media_fill_their_rows() {
    util::u32_t const columns = 1024u, half_row = media_cells(modality_t::image_k, columns);
    expect_true(half_row == 512u && media_cells(modality_t::audio_k, columns) == 512u &&
                    media_cells(modality_t::video_k, columns) == columns &&
                    media_cells(modality_t::text_k, columns) == 0u,
                "images and audio take half a row, video a whole one, text packs freely");
    um::patch_grid_t const square = um::image_patch_grid(half_row, um::image_extent_t {224u, 224u});
    expect_true(square.rows == 22u && square.columns == 23u, "a 224² image fills a 22×23 grid, got {}×{}", square.rows,
                square.columns);
    um::patch_grid_t const landscape = um::image_patch_grid(half_row, um::image_extent_t {640u, 480u});
    util::u32_t const landscape_cells = util::u32_t {landscape.rows} * landscape.columns;
    expect_true(
        landscape.columns > landscape.rows && landscape_cells <= half_row && landscape_cells * 50u >= half_row * 47u,
        "a 4:3 image keeps its aspect and fills ≥94% of its half row, got {}×{}", landscape.rows, landscape.columns);
    video_batch_shape_t video_shape {};
    video_shape.patch_side = 16u, video_shape.tube_frames = 2u, video_shape.frame_stride = 1u;
    um::video_grid_t const clip = um::video_tube_grid(32u, um::image_extent_t {128u, 128u}, video_shape, columns);
    util::u32_t const clip_cells = util::u32_t {clip.temporal_groups} * clip.patch_rows * clip.patch_columns;
    expect_true(clip.temporal_groups == 16u && clip_cells <= columns && clip_cells * 50u >= columns * 45u,
                "a 32-frame 128² clip keeps its 16 tubes and fills ≥90% of the row, got {}×{}×{}", clip.temporal_groups,
                clip.patch_rows, clip.patch_columns);
}

// Tier C / D — on-device decode into a real batch. Every helper allocates the actual
// `multimodal_batch_on_gpu_t::make` batch (exercising the real allocation + zero-gating), decodes a
// blob or fabricates a segment table, and asserts the segment contract shape-agnostically, so an
// external `UFORM_TEST_*` fixture of any resolution passes.

/**
 *  @brief The batch's positional crux: fabricate a row mixing text + image + video + audio segments
 *      (with their shape descriptors), stage them onto a real batch, run @ref fill_coords , and
 *      check every cell's @c (t,h,w) against a host oracle.
 *
 *  Text / audio walk the 1-D position axis, image is 2-D spatial, video is 3-D absolute-frame.
 *  Needs no external media.
 *
 *  @param[in,out] arena The device arena the batch and coordinate scratch are allocated from.
 *  @param[in,out] stream The stream every stage enqueues on.
 */
static void test_fill_coords_routes_every_modality(util::gpu_arena_t& arena, util::executor_stream_t& stream) {
    util::u32_t const columns = 32u, max_segments = 8u, tube_frames = 2u, frame_stride = 3u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 1u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.video.tube_frames = tube_frames; // image/video patch sides stay 0 — fill_coords reads only descriptors
    shape.video.frame_stride = frame_stride;
    auto [batch, status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
    if (!expect_true(status, "coords batch make: {}", util::to_string(status.status_))) return;

    // Four packed segments in one row: text(3), image(6 = 2×3), video(8 = tg2·pr2·pc2), audio(4).
    util::u32r1_tensor_pinned_t offsets = make_zeros<util::u32r1_tensor_pinned_t>({max_segments + 1u});
    util::u32r1_tensor_pinned_t lengths = make_zeros<util::u32r1_tensor_pinned_t>({max_segments});
    modr1_tensor_pinned_t modality = make_zeros<modr1_tensor_pinned_t>({max_segments});
    u16r2_tensor_pinned_t image_shapes = make_zeros<u16r2_tensor_pinned_t>({max_segments, 2u});
    u16r2_tensor_pinned_t video_shapes = make_zeros<u16r2_tensor_pinned_t>({max_segments, 3u});
    if (!expect_true(offsets && lengths && modality && image_shapes && video_shapes, "coords staging alloc failed"))
        return;
    util::u32_t const live_offsets[] = {0u, 3u, 9u, 17u, 21u, 21u, 21u, 21u, 21u};
    util::u32_t const live_lengths[] = {3u, 6u, 8u, 4u, 0u, 0u, 0u, 0u};
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

    if (!expect_true(util::copy_host_to_device(offsets.view(), batch.segment_offsets.span(), stream) &&
                         util::copy_host_to_device(lengths.view(), batch.segment_lengths.span(), stream) &&
                         util::copy_host_to_device(modality.view(), batch.segment_modality.span(), stream) &&
                         util::copy_host_to_device(image_shapes.view().flatten<1>(),
                                                   batch.image_shapes.span().flatten<1>(), stream) &&
                         util::copy_host_to_device(video_shapes.view().flatten<1>(),
                                                   batch.video_shapes.span().flatten<1>(), stream),
                     "coords staging copy failed"))
        return;

    std::size_t const tokens = columns; // one row
    // Device memory: `uninitialized` + explicit memset — `zeros` writes through the host pointer.
    util::i16r2_tensor_arena_t coords = make_uninitialized<util::i16r2_tensor_arena_t>(
        {tokens, 3u}, util::gpu_arena_alloc<util::i16_t> {arena});
    if (!expect_true(coords, "coords scratch alloc failed")) return;
    if (!expect_true(
            cuMemsetD8(util::device_address(coords.data()), 0, tokens * 3u * sizeof(util::i16_t)) == CUDA_SUCCESS,
            "coords scratch reset failed"))
        return;
    util::expected_t filled = fill_coords(batch.segment_offsets.span(), batch.segment_lengths.span(),
                                          batch.segment_modality.span(), batch.image_shapes.view(),
                                          batch.video_shapes.view(), tube_frames, frame_stride, coords.span(), stream);
    expect_true(static_cast<bool>(filled), "fill_coords should succeed: {}", util::to_string(filled.status_));
    if (!filled) return;

    i16r2_tensor_pinned_t host = make_zeros<i16r2_tensor_pinned_t>({tokens, 3u});
    if (!expect_true(host, "coords readback alloc failed")) return;
    if (!expect_true(util::copy_device_to_host(coords.view().flatten<1>(), host.span().flatten<1>(), stream),
                     "coords readback copy failed"))
        return;
    if (CUresult synced = cuStreamSynchronize(util::native_stream(stream));
        !expect_true(synced == CUDA_SUCCESS, "coords readback sync failed"))
        return;

    bool ok = true;
    for (std::size_t s = 0; s != max_segments; ++s)
        for (std::size_t k = 0; k != lengths[s]; ++k) {
            util::i16_t want_t = 0, want_h = 0, want_w = 0;
            switch (modality[s]) {
            case modality_t::text_k:
            case modality_t::audio_k: want_t = static_cast<util::i16_t>(k); break; // 1-D position, like a token
            case modality_t::image_k: {
                util::u32_t const pc = image_shapes[s, 1];
                want_h = static_cast<util::i16_t>(k / pc);
                want_w = static_cast<util::i16_t>(k % pc);
                break;
            }
            case modality_t::video_k: {
                util::u32_t const pc = video_shapes[s, 2], gpf = static_cast<util::u32_t>(video_shapes[s, 1]) * pc;
                want_t = static_cast<util::i16_t>((k / gpf) * tube_frames *
                                                  frame_stride); // the tube's first source frame
                want_h = static_cast<util::i16_t>((k % gpf) / pc);
                want_w = static_cast<util::i16_t>((k % gpf) % pc);
                break;
            }
            }
            std::size_t const token = offsets[s] + k;
            ok = ok && host[token, 0] == want_t && host[token, 1] == want_h && host[token, 2] == want_w;
        }
    expect_true(ok, "every cell's (t,h,w) must match the per-modality coordinate oracle");
}

/** `pack_tokens` writes the prefix, then the text, and counts both in the segment's length. */
static void test_pack_tokens_starts_with_prefix(util::gpu_arena_t& arena, util::executor_stream_t& stream) {
    char const text[] = "abcde", prefix_text[] = "q:";
    util::u64_t const host_offsets[] = {0, 3, 5}, host_elements[] = {1, 0};
    using bytes_t = nk::tensor<util::b8_t, util::gpu_arena_alloc<util::b8_t>, 1>;
    using u64s_t = nk::tensor<util::u64_t, util::gpu_arena_alloc<util::u64_t>, 1>;
    using u32s_t = nk::tensor<util::u32_t, util::gpu_arena_alloc<util::u32_t>, 1>;
    using tokens_t = nk::tensor<util::token_id_t, util::gpu_arena_alloc<util::token_id_t>, 2>;
    using modalities_t = nk::tensor<modality_t, util::gpu_arena_alloc<modality_t>, 1>;
    bytes_t payload = make_uninitialized<bytes_t>({5u}, util::gpu_arena_alloc<util::b8_t> {arena});
    bytes_t prefix = make_uninitialized<bytes_t>({2u}, util::gpu_arena_alloc<util::b8_t> {arena});
    u64s_t offsets = make_uninitialized<u64s_t>({3u}, util::gpu_arena_alloc<util::u64_t> {arena});
    u64s_t elements = make_uninitialized<u64s_t>({2u}, util::gpu_arena_alloc<util::u64_t> {arena});
    tokens_t tokens = make_uninitialized<tokens_t>({1u, 16u}, util::gpu_arena_alloc<util::token_id_t> {arena});
    u32s_t segment_offsets = make_uninitialized<u32s_t>({3u}, util::gpu_arena_alloc<util::u32_t> {arena});
    u32s_t segment_lengths = make_uninitialized<u32s_t>({2u}, util::gpu_arena_alloc<util::u32_t> {arena});
    modalities_t segment_modality = make_uninitialized<modalities_t>({2u}, util::gpu_arena_alloc<modality_t> {arena});
    u32s_t dropped = make_uninitialized<u32s_t>({1u}, util::gpu_arena_alloc<util::u32_t> {arena});
    if (!expect_true(payload && prefix && offsets && elements && tokens && segment_offsets && segment_lengths &&
                         segment_modality && dropped,
                     "pack scratch alloc failed"))
        return;
    bool const staged =
        util::copy_host_to_device<util::b8_t>(std::as_bytes(std::span {text, 5}), payload.span(), stream) &&
        util::copy_host_to_device<util::b8_t>(std::as_bytes(std::span {prefix_text, 2}), prefix.span(), stream) &&
        util::copy_host_to_device<util::u64_t>(std::span {host_offsets}, offsets.span(), stream) &&
        util::copy_host_to_device<util::u64_t>(std::span {host_elements}, elements.span(), stream) &&
        util::zero_async(dropped.span(), stream);
    if (!expect_true(staged, "pack staging failed")) return;
    util::expected_t packed = pack_tokens(column_view_t {payload.view(), offsets.view()}, elements.view(),
                                          prefix.view(), tokens.span(), segment_offsets.span(), segment_lengths.span(),
                                          segment_modality.span(), dropped.span(), stream);
    if (!expect_true(packed, "pack_tokens failed: {}", packed.description_)) return;

    util::token_id_t host_tokens[16] = {};
    util::u32_t host_lengths[2] = {}, host_dropped = 1;
    bool const landed = util::copy_device_to_host(tokens.view().flatten<1>(), std::span {host_tokens}, stream) &&
                        util::copy_device_to_host(segment_lengths.view(), std::span {host_lengths}, stream) &&
                        util::copy_device_to_host(dropped.view(), std::span {&host_dropped, 1}, stream) &&
                        cuStreamSynchronize(util::native_stream(stream)) == CUDA_SUCCESS;
    if (!expect_true(landed, "pack readback failed")) return;
    char const expected[] = "q:deq:abc";
    bool matches = host_lengths[0] == 4u && host_lengths[1] == 5u && host_dropped == 0u;
    for (std::size_t index = 0; index != 9; ++index)
        matches = matches &&
                  host_tokens[index] == static_cast<util::token_id_t>(static_cast<util::u8_t>(expected[index]));
    expect_true(matches, "two segments read \"q:de\" and \"q:abc\" (lengths {} and {}, {} dropped)", host_lengths[0],
                host_lengths[1], host_dropped);
}

/**
 *  @brief Decode @p clip into a one-row audio batch and assert the raw-frame contract: one
 *      @c audio_k segment filling the row, finite samples, and a PAD @c text_k remainder.
 *  @param[in] label The case label the diagnostics print.
 *  @param[in] clip The encoded clip bytes; an empty clip skips the case.
 *  @param[in,out] arena The device arena.
 *  @param[in,out] stream The stream every stage enqueues on.
 *  @param[in,out] pool The CPU-decode worker pool.
 */
static case_result_t test_audio_decode(sz::string_view_t label, st::basic_vector<util::b8_t> const& clip,
                                       util::gpu_arena_t& arena, util::executor_stream_t& stream, fu::pool_t& pool) {
    if (clip.empty()) return case_result_t::skipped("needs a readable UFORM_TEST_AUDIO fixture");
    util::u32_t const columns = 24u, chunk = 400u, max_segments = 4u, sample_rate = 16000u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 1u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.audio.chunk_samples = chunk;
    shape.audio.sample_rate = sample_rate;
    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
    if (!expect_true(make_batch_status, "[{}] audio batch make: {}", label, util::to_string(make_batch_status.status_)))
        return {};

    util::bytes_view_t blob = clip;
    auto [tape, make_tape_status] = fabricated_tape_t::make(arena, stream, {&blob, 1});
    if (!expect_true(make_tape_status, "[{}] audio tape: {}", label, util::to_string(make_tape_status.status_)))
        return {};
    util::stack_array<util::u64_t, 1> drawn;
    [[maybe_unused]] util::status_t const pushed = drawn.push_back(0);
    audio_batch_shape_t const representation {chunk, sample_rate};
    std::span<audio_batch_shape_t const> const representations {&representation, 1};
    util::expected_t decoded = decode_audios(
        pool, tape.view(), drawn.view(), representations, batch.audio_samples.span(), batch.segment_offsets.span(),
        batch.segment_lengths.span(), batch.segment_modality.span(), batch.dropped.span(), stream);
    expect_true(static_cast<bool>(decoded), "[{}] decode_audios should succeed: {}", label,
                util::to_string(decoded.status_));
    if (!decoded) return {};

    modr1_tensor_pinned_t modality = make_zeros<modr1_tensor_pinned_t>({max_segments});
    util::u32r1_tensor_pinned_t lengths = make_zeros<util::u32r1_tensor_pinned_t>({max_segments});
    util::u32r1_tensor_pinned_t offsets = make_zeros<util::u32r1_tensor_pinned_t>({max_segments + 1u});
    util::f32r1_tensor_pinned_t samples = make_zeros<util::f32r1_tensor_pinned_t>({batch.audio_samples.numel()});
    if (!expect_true(modality && lengths && offsets && samples, "[{}] audio readback alloc", label)) return {};
    if (!expect_true(util::copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) &&
                         util::copy_device_to_host(batch.segment_lengths.view(), lengths.span(), stream) &&
                         util::copy_device_to_host(batch.segment_offsets.view(), offsets.span(), stream) &&
                         util::copy_device_to_host(batch.audio_samples.view().flatten<1>(), samples.span(), stream),
                     "[{}] audio readback copy failed", label))
        return {};
    if (CUresult synced = cuStreamSynchronize(util::native_stream(stream));
        !expect_true(synced == CUDA_SUCCESS, "[{}] audio readback sync failed", label))
        return {};

    expect_true(modality[0] == modality_t::audio_k, "[{}] first segment must be audio_k (got {})", label,
                static_cast<int>(modality[0]));
    util::u32_t const half_row = media_cells(modality_t::audio_k, columns);
    expect_true(lengths[0] == half_row, "[{}] one clip fills half the row: length {} must equal {}", label, lengths[0],
                half_row);
    expect_true(offsets[0] == 0u, "[{}] first segment must start at the row origin (got {})", label, offsets[0]);
    bool finite = true;
    std::size_t const live = std::size_t {lengths[0]} * chunk;
    for (std::size_t i = 0; i != live; ++i) finite = finite && std::isfinite(samples[i]);
    expect_true(finite, "[{}] every audio sample must be finite", label);
    bool pad = true;
    for (std::size_t s = 1; s != max_segments; ++s) pad = pad && lengths[s] == 0u && modality[s] == modality_t::text_k;
    expect_true(pad, "[{}] audio PAD segments must have zero length and text_k", label);
    expect_true(offsets[max_segments] == lengths[0], "[{}] terminal offset must close the last segment", label);
    return {};
}

/**
 *  @brief A real env audio clip and an in-test garbage blob decode into one two-row batch: mark-and-continue
 *      keeps the batch coherent — the garbage row fault-isolates to silence while both segments stay @c audio_k
 *      and the decode as a whole still succeeds.
 *  @param[in,out] arena The device arena.
 *  @param[in,out] stream The stream every stage enqueues on.
 *  @param[in,out] pool The CPU-decode worker pool.
 *  @param[in] clip The real env audio clip; an empty clip skips the case.
 */
static case_result_t test_audio_fault_isolation(util::gpu_arena_t& arena, util::executor_stream_t& stream,
                                                fu::pool_t& pool, st::basic_vector<util::b8_t> const& clip) {
    if (clip.empty()) return case_result_t::skipped("needs a UFORM_TEST_AUDIO fixture");
    util::u32_t const columns = 16u, chunk = 400u, max_segments = 4u, sample_rate = 16000u;
    st::basic_vector<util::b8_t> garbage; // not a decodable container → silence, but not the whole batch
    if (!expect_true(!st::failed(garbage.resize(64u)), "garbage scratch alloc failed")) return {};
    for (std::size_t i = 0; i != garbage.size(); ++i) garbage[i] = util::b8_t {0x7F};

    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 2u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.audio.chunk_samples = chunk;
    shape.audio.sample_rate = sample_rate;
    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
    if (!expect_true(make_batch_status, "isolation batch make: {}", util::to_string(make_batch_status.status_)))
        return {};

    util::stack_array<util::bytes_view_t, 2> blobs;
    [[maybe_unused]] util::status_t const pushed_clip = blobs.push_back(util::bytes_view_t {clip});
    [[maybe_unused]] util::status_t const pushed_garbage = blobs.push_back(util::bytes_view_t {garbage});
    auto [tape, make_tape_status] = fabricated_tape_t::make(arena, stream, blobs.view());
    if (!expect_true(make_tape_status, "isolation tape: {}", util::to_string(make_tape_status.status_))) return {};
    util::stack_array<util::u64_t, 2> drawn;
    [[maybe_unused]] util::status_t const pushed_first = drawn.push_back(0);
    [[maybe_unused]] util::status_t const pushed_second = drawn.push_back(1);
    audio_batch_shape_t const representation {chunk, sample_rate};
    std::span<audio_batch_shape_t const> const representations {&representation, 1};
    util::expected_t decoded = decode_audios(
        pool, tape.view(), drawn.view(), representations, batch.audio_samples.span(), batch.segment_offsets.span(),
        batch.segment_lengths.span(), batch.segment_modality.span(), batch.dropped.span(), stream);
    expect_true(static_cast<bool>(decoded), "one corrupt clip among good must not fail the batch: {}",
                util::to_string(decoded.status_));
    if (!decoded) return {};

    modr1_tensor_pinned_t modality = make_zeros<modr1_tensor_pinned_t>({max_segments});
    f32r2_tensor_pinned_t samples = make_zeros<f32r2_tensor_pinned_t>({2u, std::size_t {columns} * chunk});
    if (!expect_true(modality && samples, "isolation readback alloc failed")) return {};
    if (!expect_true(
            util::copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) &&
                util::copy_device_to_host(batch.audio_samples.view().flatten<1>(), samples.span().flatten<1>(), stream),
            "isolation readback copy failed"))
        return {};
    if (CUresult synced = cuStreamSynchronize(util::native_stream(stream));
        !expect_true(synced == CUDA_SUCCESS, "isolation readback sync failed"))
        return {};

    expect_true(modality[0] == modality_t::audio_k && modality[1] == modality_t::audio_k,
                "both drawn clips must be audio_k segments");
    bool garbage_silent = true;
    for (std::size_t i = 0; i != std::size_t {columns} * chunk; ++i)
        garbage_silent = garbage_silent && samples[1, i] == 0.0f; // row 1 = the corrupt clip
    expect_true(garbage_silent, "a corrupt clip must fault-isolate to silence, not corrupt its neighbour");
    return {};
}

/**
 *  @brief Decode @p blob into a one-row RGBA image batch and assert: one @c image_k segment, its
 *      @c (patch_rows,patch_cols) factoring its length, sub-row placement, and a valid alpha byte.
 *  @param[in] label The case label the diagnostics print.
 *  @param[in] blob The encoded image bytes; an empty blob skips the case.
 *  @param[in,out] arena The device arena.
 *  @param[in,out] stream The stream every stage enqueues on.
 *  @param[in,out] pool The CPU-decode worker pool.
 *  @param[in,out] jpeg This GPU's nvJPEG session.
 */
static case_result_t test_image_decode(sz::string_view_t label, st::basic_vector<util::b8_t> const& blob,
                                       util::gpu_arena_t& arena, util::executor_stream_t& stream, fu::pool_t& pool,
                                       nvjpeg_session_t& jpeg) {
    if (blob.empty()) return case_result_t::skipped("needs a readable UFORM_TEST_IMAGE fixture");
    util::u32_t const columns = 64u, patch = 16u, channels = 4u, max_segments = 4u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 1u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.image.patch_side = patch;
    shape.image.channels = channels;
    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
    if (!expect_true(make_batch_status, "[{}] image batch make: {}", label, util::to_string(make_batch_status.status_)))
        return {};

    util::bytes_view_t one = blob;
    auto [tape, make_tape_status] = fabricated_tape_t::make(arena, stream, {&one, 1});
    if (!expect_true(make_tape_status, "[{}] image tape: {}", label, util::to_string(make_tape_status.status_)))
        return {};
    util::stack_array<util::u64_t, 1> drawn;
    [[maybe_unused]] util::status_t const pushed = drawn.push_back(0);
    image_batch_shape_t const representation {patch, channels};
    std::span<image_batch_shape_t const> const representations {&representation, 1};
    util::expected_t decoded = decode_images(jpeg, pool, tape.view(), drawn.view(), representations,
                                             batch.images.span(), batch.image_shapes.span(),
                                             batch.segment_offsets.span(), batch.segment_lengths.span(),
                                             batch.segment_modality.span(), batch.dropped.span(), stream);
    expect_true(static_cast<bool>(decoded), "[{}] decode_images should succeed: {}", label, decoded.description_);
    if (!decoded) return {};

    modr1_tensor_pinned_t modality = make_zeros<modr1_tensor_pinned_t>({max_segments});
    util::u32r1_tensor_pinned_t lengths = make_zeros<util::u32r1_tensor_pinned_t>({max_segments});
    util::u32r1_tensor_pinned_t offsets = make_zeros<util::u32r1_tensor_pinned_t>({max_segments + 1u});
    u16r2_tensor_pinned_t shapes = make_zeros<u16r2_tensor_pinned_t>({max_segments, 2u});
    pixelr2_tensor_pinned_t pixels = make_zeros<pixelr2_tensor_pinned_t>(
        {batch.images.numel() / channels, std::size_t {channels}});
    if (!expect_true(modality && lengths && offsets && shapes && pixels, "[{}] image readback alloc", label)) return {};
    if (!expect_true(
            util::copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) &&
                util::copy_device_to_host(batch.segment_lengths.view(), lengths.span(), stream) &&
                util::copy_device_to_host(batch.segment_offsets.view(), offsets.span(), stream) &&
                util::copy_device_to_host(batch.image_shapes.view().flatten<1>(), shapes.span().flatten<1>(), stream) &&
                util::copy_device_to_host(batch.images.view().flatten<1>(), pixels.span().flatten<1>(), stream),
            "[{}] image readback copy failed", label))
        return {};
    if (CUresult synced = cuStreamSynchronize(util::native_stream(stream));
        !expect_true(synced == CUDA_SUCCESS, "[{}] image readback sync failed", label))
        return {};

    util::u32_t const patch_rows = shapes[0, 0], patch_cols = shapes[0, 1];
    expect_true(modality[0] == modality_t::image_k, "[{}] first segment must be image_k (got {})", label,
                static_cast<int>(modality[0]));
    expect_true(lengths[0] > 0u && lengths[0] <= columns, "[{}] one image is a sub-row segment (length {} of {})",
                label, lengths[0], columns);
    expect_true(patch_rows * patch_cols == lengths[0],
                "[{}] image_shapes (rows,cols)=({},{}) must multiply to the segment length {}", label, patch_rows,
                patch_cols, lengths[0]);
    expect_true(offsets[0] == 0u, "[{}] first segment must start at the row origin (got {})", label, offsets[0]);
    util::u32_t max_alpha = 0u;
    std::size_t const live_pixels = std::size_t {lengths[0]} * patch * patch; // cells × pixels-per-patch
    for (std::size_t p = 0; p != live_pixels; ++p)
        max_alpha = fu::max_of<util::u32_t>(max_alpha, pixels[p, 3]);                 // alpha lane
    expect_true(max_alpha <= 255u, "[{}] alpha channel must be a valid byte", label); // shape-agnostic
    return {};
}

/** Decode every @p blobs image in one batch, then each alone with the same shape, and require each
 *  image's patches to match bit for bit, so no image's decode writes into another's region. */
static case_result_t test_image_batch_matches_single(std::span<st::basic_vector<util::b8_t> const> blobs,
                                                     util::gpu_arena_t& arena, util::executor_stream_t& stream,
                                                     fu::pool_t& pool, nvjpeg_session_t& jpeg) {
    if (blobs.size() < 2) return case_result_t::skipped("needs at least two UFORM_TEST_IMAGE fixtures");
    util::u32_t const columns = 256u, patch = 16u, channels = 4u, images = static_cast<util::u32_t>(blobs.size());
    std::size_t const cell_pixels = std::size_t {patch} * patch * channels;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = images;
    shape.max_segments_per_gpu = images;
    shape.max_columns_per_row = columns;
    shape.image.patch_side = patch;
    shape.image.channels = channels;
    st::basic_vector<util::bytes_view_t> views;
    for (st::basic_vector<util::b8_t> const& blob : blobs)
        if (!expect_true(!st::failed(views.push_back(util::bytes_view_t {blob})), "blob view append")) return {};
    auto [tape, make_tape_status] = fabricated_tape_t::make(arena, stream, views);
    if (!expect_true(make_tape_status, "image tape: {}", util::to_string(make_tape_status.status_))) return {};
    image_batch_shape_t const representation {patch, channels};

    // One decode of `drawn` into a fresh batch, landing its offsets, lengths and cell pixels.
    struct landed_t {
        util::u32r1_tensor_pinned_t offsets, lengths;
        pixelr1_tensor_pinned_t pixels;
    };
    auto decode = [&](std::span<util::u64_t const> drawn, landed_t& landed) -> bool {
        auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
        if (!make_batch_status) return false;
        if (!decode_images(jpeg, pool, tape.view(), drawn, {&representation, 1}, batch.images.span(),
                           batch.image_shapes.span(), batch.segment_offsets.span(), batch.segment_lengths.span(),
                           batch.segment_modality.span(), batch.dropped.span(), stream))
            return false;
        landed.offsets = make_zeros<util::u32r1_tensor_pinned_t>({images + 1u});
        landed.lengths = make_zeros<util::u32r1_tensor_pinned_t>({images});
        landed.pixels = make_zeros<pixelr1_tensor_pinned_t>({batch.images.numel()});
        return landed.offsets && landed.lengths && landed.pixels &&
               util::copy_device_to_host(batch.segment_offsets.view(), landed.offsets.span(), stream) &&
               util::copy_device_to_host(batch.segment_lengths.view(), landed.lengths.span(), stream) &&
               util::copy_device_to_host(batch.images.view().flatten<1>(), landed.pixels.span(), stream) &&
               cuStreamSynchronize(util::native_stream(stream)) == CUDA_SUCCESS;
    };
    st::basic_vector<util::u64_t> drawn;
    for (std::size_t row = 0; row != images; ++row)
        if (!expect_true(!st::failed(drawn.push_back(util::u64_t {row})), "drawn row append")) return {};
    landed_t together;
    if (!expect_true(decode(drawn, together), "the batched decode should succeed")) return {};
    st::basic_vector<landed_t> singles;
    for (std::size_t image = 0; image != images; ++image) {
        landed_t alone;
        if (!expect_true(decode(std::span<util::u64_t const> {drawn}.subspan(image, 1), alone) &&
                             !st::failed(singles.push_back(std::move(alone))),
                         "image {} decoded alone should succeed", image))
            return {};
    }
    // The mean absolute pixel difference between batched `image` and single decode `single`.
    auto distance = [&](std::size_t image, std::size_t single) noexcept -> double {
        landed_t const& alone = singles[single];
        std::size_t const values = std::size_t {fu::min_of(together.lengths[image], alone.lengths[0])} * cell_pixels;
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
        expect_true(own == 0.0 && together.lengths[image] == singles[image].lengths[0],
                    "image {} of {} must decode the same in a batch as alone: mean |diff| {:.3f}, " //
                    "nearest single {} at {:.3f}, cells {} vs {}",
                    image, images, own, nearest, distance(image, nearest), together.lengths[image],
                    singles[image].lengths[0]);
    }
    return {};
}

/**
 *  @brief Decode @p clip into one video batch, one representation per entry of @p frame_strides ,
 *      and assert each emits a @c video_k segment whose derived
 *      @c (temporal_groups,patch_rows,patch_cols) is self-consistent and fits the grid.
 *
 *  Distinct strides yield distinct temporal factorizations. The footprint is derived from the
 *  fixture's native resolution and clamped to a row, so a fixture of any size downscales to fit.
 *
 *  @param[in] label The case label the diagnostics print.
 *  @param[in] clip The encoded clip bytes; an empty clip skips the case.
 *  @param[in] frame_strides One temporal stride per representation.
 *  @param[in,out] arena The device arena.
 *  @param[in,out] stream The stream every stage enqueues on.
 *  @param[in,out] pool The CPU demux pool.
 *  @param[in,out] nvdec The multi-engine NVDEC session.
 */
static case_result_t test_video_decode(sz::string_view_t label, st::basic_vector<util::b8_t> const& clip,
                                       std::span<util::u32_t const> frame_strides, util::gpu_arena_t& arena,
                                       util::executor_stream_t& stream, fu::pool_t& pool, nvdec_session_t& nvdec) {
    if (clip.empty()) return case_result_t::skipped("needs a readable UFORM_TEST_VIDEO fixture");
    util::u32_t const columns = 4096u, patch = 32u, channels = 3u, tube_frames = 2u, max_segments = 8u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = 2u;
    shape.max_segments_per_gpu = max_segments;
    shape.max_columns_per_row = columns;
    shape.video.patch_side = patch;
    shape.video.channels = channels;
    shape.video.tube_frames = tube_frames;
    auto [batch, make_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena}, shape);
    if (!expect_true(make_batch_status, "[{}] video batch make: {}", label, util::to_string(make_batch_status.status_)))
        return {};

    util::stack_array<video_batch_shape_t, 4>
        representations; // one flat shape per stride, sharing the grid's patch geometry
    for (util::u32_t stride : frame_strides) {
        [[maybe_unused]] util::status_t const pushed = representations.push_back(
            video_batch_shape_t {patch, channels, tube_frames, stride});
    }

    util::bytes_view_t one = clip;
    auto [tape, make_tape_status] = fabricated_tape_t::make(arena, stream, {&one, 1});
    if (!expect_true(make_tape_status, "[{}] video tape: {}", label, util::to_string(make_tape_status.status_)))
        return {};
    util::stack_array<util::u64_t, 1> drawn;
    [[maybe_unused]] util::status_t const pushed = drawn.push_back(0);
    util::expected_t decoded = decode_videos(
        nvdec, pool, tape.view(), drawn.view(), representations.view(), batch.videos.span(), batch.video_shapes.span(),
        batch.segment_offsets.span(), batch.segment_lengths.span(), batch.segment_modality.span(), stream);
    expect_true(static_cast<bool>(decoded), "[{}] decode_videos should succeed: {}", label,
                util::to_string(decoded.status_));
    if (!decoded) return {};

    modr1_tensor_pinned_t modality = make_zeros<modr1_tensor_pinned_t>({max_segments});
    util::u32r1_tensor_pinned_t lengths = make_zeros<util::u32r1_tensor_pinned_t>({max_segments});
    u16r2_tensor_pinned_t shapes = make_zeros<u16r2_tensor_pinned_t>({max_segments, 3u});
    if (!expect_true(modality && lengths && shapes, "[{}] video readback alloc", label)) return {};
    if (!expect_true(
            util::copy_device_to_host(batch.segment_modality.view(), modality.span(), stream) &&
                util::copy_device_to_host(batch.segment_lengths.view(), lengths.span(), stream) &&
                util::copy_device_to_host(batch.video_shapes.view().flatten<1>(), shapes.span().flatten<1>(), stream),
            "[{}] video readback copy failed", label))
        return {};
    if (CUresult synced = cuStreamSynchronize(util::native_stream(stream));
        !expect_true(synced == CUDA_SUCCESS, "[{}] video readback sync failed", label))
        return {};

    bool all_video = true, all_consistent = true, all_fit = true, any_live = false;
    for (std::size_t v = 0; v != representations.size(); ++v) {
        all_video = all_video && modality[v] == modality_t::video_k; // dropped or live, a drawn segment is video_k
        util::u32_t const product = static_cast<util::u32_t>(shapes[v, 0]) * shapes[v, 1] * shapes[v, 2];
        if (lengths[v] != 0u) {
            any_live = true;
            all_consistent = all_consistent && product == lengths[v];
            all_fit = all_fit && lengths[v] <= columns;
        }
    }
    expect_true(all_video, "[{}] every representation must emit a video_k segment", label);
    expect_true(all_consistent, "[{}] each live segment's (tg,pr,pc) must multiply to its length", label);
    expect_true(all_fit, "[{}] each live segment must fit one row ({} columns)", label, columns);
    expect_true(any_live, "[{}] at least one representation must produce a live segment", label);
    if (representations.size() == 2 && lengths[0] != 0u && lengths[1] != 0u)
        expect_true(shapes[0, 0] != shapes[1, 0] || shapes[0, 1] != shapes[1, 1] || shapes[0, 2] != shapes[1, 2],
                    "[{}] distinct frame strides must yield distinct factorizations", label);
    return {};
}

/**
 *  @brief A real multimodal packet: decode a real audio clip and image, then pack both real
 *      segments — their true decoded lengths and the image's real @c (patch_rows,patch_cols) — into
 *      one mixed row and check @ref fill_coords routes each modality's coordinates.
 *
 *  A single loader owns the whole segment table at fill, so a mixed row is assembled here the way
 *  the cross-rank mixer produces it, over genuinely decoded geometry.
 *
 *  @param[in,out] arena The device arena.
 *  @param[in,out] stream The stream every stage enqueues on.
 *  @param[in,out] pool The CPU-decode worker pool.
 *  @param[in,out] jpeg This GPU's nvJPEG session.
 *  @param[in] audio_clip The real env audio clip; empty skips the case.
 *  @param[in] image_blob The real env image blob; empty skips the case.
 */
static case_result_t test_multimodal_packet_routes(util::gpu_arena_t& arena, util::executor_stream_t& stream,
                                                   fu::pool_t& pool, nvjpeg_session_t& jpeg,
                                                   st::basic_vector<util::b8_t> const& audio_clip,
                                                   st::basic_vector<util::b8_t> const& image_blob) {
    if (audio_clip.empty() || image_blob.empty())
        return case_result_t::skipped("needs both UFORM_TEST_AUDIO and UFORM_TEST_IMAGE fixtures");

    // Decode the clip into an audio-only batch → its real segment length.
    util::u32_t const chunk = 400u, audio_columns = 12u;
    multimodal_batch_shape_t audio_shape {};
    audio_shape.max_rows_per_gpu = 1u;
    audio_shape.max_segments_per_gpu = 2u;
    audio_shape.max_columns_per_row = audio_columns;
    audio_shape.audio.chunk_samples = chunk;
    audio_shape.audio.sample_rate = 16000u;
    auto [audio_batch, make_audio_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena},
                                                                                  audio_shape);
    if (!expect_true(make_audio_batch_status, "multimodal: audio batch make: {}",
                     util::to_string(make_audio_batch_status.status_)))
        return {};
    util::bytes_view_t audio_one = audio_clip;
    auto [audio_tape, make_audio_tape_status] = fabricated_tape_t::make(arena, stream, {&audio_one, 1});
    if (!expect_true(make_audio_tape_status, "multimodal: audio tape: {}",
                     util::to_string(make_audio_tape_status.status_)))
        return {};
    util::stack_array<util::u64_t, 1> audio_drawn;
    [[maybe_unused]] util::status_t const pushed_audio = audio_drawn.push_back(0);
    audio_batch_shape_t const audio_representation {chunk, 16000u};
    std::span<audio_batch_shape_t const> const audio_reps {&audio_representation, 1};
    if (!expect_true(
            decode_audios(pool, audio_tape.view(), audio_drawn.view(), audio_reps, audio_batch.audio_samples.span(),
                          audio_batch.segment_offsets.span(), audio_batch.segment_lengths.span(),
                          audio_batch.segment_modality.span(), audio_batch.dropped.span(), stream),
            "multimodal: audio decode failed"))
        return {};

    // Decode the blob into an image-only batch → its real segment
    // length and (patch_rows,patch_cols).
    util::u32_t const image_columns = 32u, patch = 16u;
    multimodal_batch_shape_t image_shape {};
    image_shape.max_rows_per_gpu = 1u;
    image_shape.max_segments_per_gpu = 2u;
    image_shape.max_columns_per_row = image_columns;
    image_shape.image.patch_side = patch;
    image_shape.image.channels = 4u;
    auto [image_batch, make_image_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena},
                                                                                  image_shape);
    if (!expect_true(make_image_batch_status, "multimodal: image batch make: {}",
                     util::to_string(make_image_batch_status.status_)))
        return {};
    util::bytes_view_t image_one = image_blob;
    auto [image_tape, make_image_tape_status] = fabricated_tape_t::make(arena, stream, {&image_one, 1});
    if (!expect_true(make_image_tape_status, "multimodal: image tape: {}",
                     util::to_string(make_image_tape_status.status_)))
        return {};
    util::stack_array<util::u64_t, 1> image_drawn;
    [[maybe_unused]] util::status_t const pushed_image = image_drawn.push_back(0);
    image_batch_shape_t const image_representation {patch, 4u};
    std::span<image_batch_shape_t const> const image_reps {&image_representation, 1};
    if (!expect_true(decode_images(jpeg, pool, image_tape.view(), image_drawn.view(), image_reps,
                                   image_batch.images.span(), image_batch.image_shapes.span(),
                                   image_batch.segment_offsets.span(), image_batch.segment_lengths.span(),
                                   image_batch.segment_modality.span(), image_batch.dropped.span(), stream),
                     "multimodal: image decode failed"))
        return {};

    // Read both decodes' real geometry back before assembling the mixed row.
    util::u32r1_tensor_pinned_t audio_len = make_zeros<util::u32r1_tensor_pinned_t>({2u});
    util::u32r1_tensor_pinned_t image_len = make_zeros<util::u32r1_tensor_pinned_t>({2u});
    u16r2_tensor_pinned_t image_grid = make_zeros<u16r2_tensor_pinned_t>({2u, 2u});
    if (!expect_true(audio_len && image_len && image_grid, "multimodal: geometry readback alloc")) return {};
    if (!expect_true(util::copy_device_to_host(audio_batch.segment_lengths.view(), audio_len.span(), stream) &&
                         util::copy_device_to_host(image_batch.segment_lengths.view(), image_len.span(), stream) &&
                         util::copy_device_to_host(image_batch.image_shapes.view().flatten<1>(),
                                                   image_grid.span().flatten<1>(), stream),
                     "multimodal: geometry readback copy failed"))
        return {};
    if (CUresult synced = cuStreamSynchronize(util::native_stream(stream));
        !expect_true(synced == CUDA_SUCCESS, "multimodal: geometry readback sync failed"))
        return {};
    util::u32_t const audio_length = audio_len[0], image_length = image_len[0];
    util::u16_t const grid_rows = image_grid[0, 0], grid_cols = image_grid[0, 1];

    // Pack both real segments into one mixed row and route their coordinates.
    util::u32_t const columns = 64u, max_segments = 4u, tube_frames = 2u;
    if (!expect_true(audio_length + image_length <= columns,
                     "multimodal: the two real segments must fit one row ({}+{} <= {})", audio_length, image_length,
                     columns))
        return {};
    multimodal_batch_shape_t mixed_shape {};
    mixed_shape.max_rows_per_gpu = 1u;
    mixed_shape.max_segments_per_gpu = max_segments;
    mixed_shape.max_columns_per_row = columns;
    mixed_shape.video.tube_frames = tube_frames; // present so image/video shape descriptors exist
    auto [mixed_batch, make_mixed_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena},
                                                                                  mixed_shape);
    if (!expect_true(make_mixed_batch_status, "multimodal: mixed batch make: {}",
                     util::to_string(make_mixed_batch_status.status_)))
        return {};

    util::u32r1_tensor_pinned_t offsets = make_zeros<util::u32r1_tensor_pinned_t>({max_segments + 1u});
    util::u32r1_tensor_pinned_t lengths = make_zeros<util::u32r1_tensor_pinned_t>({max_segments});
    modr1_tensor_pinned_t modality = make_zeros<modr1_tensor_pinned_t>({max_segments});
    u16r2_tensor_pinned_t shapes = make_zeros<u16r2_tensor_pinned_t>({max_segments, 2u});
    u16r2_tensor_pinned_t video_shapes = make_zeros<u16r2_tensor_pinned_t>({max_segments, 3u});
    if (!expect_true(offsets && lengths && modality && shapes && video_shapes,
                     "multimodal: mixed staging alloc failed"))
        return {};
    for (std::size_t s = 0; s != max_segments + 1u; ++s) offsets[s] = audio_length + image_length;
    offsets[0] = 0u;
    offsets[1] = audio_length;
    lengths[0] = audio_length;
    lengths[1] = image_length;
    modality[0] = modality_t::audio_k;
    modality[1] = modality_t::image_k;
    shapes[1, 0] = grid_rows;
    shapes[1, 1] = grid_cols;
    if (!expect_true(util::copy_host_to_device(offsets.view(), mixed_batch.segment_offsets.span(), stream) &&
                         util::copy_host_to_device(lengths.view(), mixed_batch.segment_lengths.span(), stream) &&
                         util::copy_host_to_device(modality.view(), mixed_batch.segment_modality.span(), stream) &&
                         util::copy_host_to_device(shapes.view().flatten<1>(),
                                                   mixed_batch.image_shapes.span().flatten<1>(), stream) &&
                         util::copy_host_to_device(video_shapes.view().flatten<1>(),
                                                   mixed_batch.video_shapes.span().flatten<1>(), stream),
                     "multimodal: mixed staging copy failed"))
        return {};

    std::size_t const tokens = columns;
    // Device memory: `uninitialized` + explicit memset — `zeros` writes through the host pointer.
    util::i16r2_tensor_arena_t coords = make_uninitialized<util::i16r2_tensor_arena_t>(
        {tokens, 3u}, util::gpu_arena_alloc<util::i16_t> {arena});
    if (!expect_true(coords, "multimodal: coords alloc failed")) return {};
    if (!expect_true(
            cuMemsetD8(util::device_address(coords.data()), 0, tokens * 3u * sizeof(util::i16_t)) == CUDA_SUCCESS,
            "multimodal: coords reset failed"))
        return {};
    util::expected_t filled = fill_coords(mixed_batch.segment_offsets.span(), mixed_batch.segment_lengths.span(),
                                          mixed_batch.segment_modality.span(), mixed_batch.image_shapes.view(),
                                          mixed_batch.video_shapes.view(), tube_frames, mixed_shape.video.frame_stride,
                                          coords.span(), stream);
    expect_true(static_cast<bool>(filled), "multimodal: fill_coords over the mixed packet: {}",
                util::to_string(filled.status_));
    if (!filled) return {};

    i16r2_tensor_pinned_t host = make_zeros<i16r2_tensor_pinned_t>({tokens, 3u});
    if (!expect_true(host, "multimodal: coords readback alloc failed")) return {};
    if (!expect_true(util::copy_device_to_host(coords.view().flatten<1>(), host.span().flatten<1>(), stream),
                     "multimodal: coords readback copy failed"))
        return {};
    if (CUresult synced = cuStreamSynchronize(util::native_stream(stream));
        !expect_true(synced == CUDA_SUCCESS, "multimodal: coords readback sync failed"))
        return {};

    bool audio_ok = true, image_ok = true;
    for (std::size_t k = 0; k != audio_length; ++k) { // audio walks the 1-D position axis
        std::size_t const token = k;
        audio_ok = audio_ok && host[token, 0] == static_cast<util::i16_t>(k) && host[token, 1] == 0 &&
                   host[token, 2] == 0;
    }
    util::u32_t const pc = grid_cols;
    for (std::size_t k = 0; k != image_length; ++k) { // image is 2-D spatial over its real grid
        std::size_t const token = std::size_t {audio_length} + k;
        image_ok = image_ok && host[token, 0] == 0 && host[token, 1] == static_cast<util::i16_t>(k / pc) &&
                   host[token, 2] == static_cast<util::i16_t>(k % pc);
    }
    expect_true(audio_ok, "multimodal: the packed audio segment's real cells must route to (k,0,0)");
    expect_true(image_ok, "multimodal: the packed image segment's real cells must route to (0,row,col)");
    return {};
}

/** Fill two steps back to back on one stream with no drain between, then replay the first step
 *  and require it bitwise equal: a fill frees its reservoir stream-ordered, so the next fill's
 *  read must not land in memory the first fill's queued decode still reads. */
static void test_paired_fill_replays_after_back_to_back_fills(ut::mixture_entry_t const& entry,
                                                              util::gpu_arena_t& arena, util::executor_stream_t& stream,
                                                              fu::pool_t& pool, nvjpeg_session_t& jpeg) {
    sz::string_view_t const name = entry.name.view();
    auto [session, make_session_status] = gpu_direct_storage_session_t::make(1);
    if (!expect_true(make_session_status, "the storage session should make: {}", make_session_status.description_))
        return;
    util::u32_t const columns = 256u, patch = 16u, channels = 4u, rows = 8u, segments = 16u;
    multimodal_batch_shape_t shape {};
    shape.max_rows_per_gpu = rows;
    shape.max_segments_per_gpu = segments;
    shape.max_bags_per_gpu = segments / 2u;
    shape.max_columns_per_row = columns;
    shape.image.patch_side = patch;
    shape.image.channels = channels;
    paired_loader_parquet_config_t config;
    config.anchor = entry.anchor.view(), config.positive = entry.positive.view();
    auto [loader, make_loader_status] = paired_loader_parquet_t::make(session, entry.glob.view(), 42u, pool, config,
                                                                      modality_t::image_k, shape);
    if (!expect_true(make_loader_status, "[{}] the paired loader should make: {}", name,
                     make_loader_status.description_))
        return;

    st::basic_vector<multimodal_batch_on_gpu_t> first, second;
    for (std::size_t batch = 0; batch != 2; ++batch) {
        auto [first_batch, make_first_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena},
                                                                                      shape);
        auto [second_batch, make_second_batch_status] = multimodal_batch_on_gpu_t::make(util::gpu_arena_alloc_t {arena},
                                                                                        shape);
        if (!expect_true(make_first_batch_status && make_second_batch_status &&
                             !st::failed(first.push_back(std::move(first_batch))) &&
                             !st::failed(second.push_back(std::move(second_batch))),
                         "[{}] the batches should make", name))
            return;
    }
    using tokensr1_tensor_pinned_t = nk::tensor<util::token_id_t, util::pinned_alloc<util::token_id_t>, 1>;
    std::size_t const pixels = first[0].images.numel(), tokens = first[0].tokens.numel();
    pixelr1_tensor_pinned_t filled_pixels = make_zeros<pixelr1_tensor_pinned_t>({2 * pixels});
    pixelr1_tensor_pinned_t replayed_pixels = make_zeros<pixelr1_tensor_pinned_t>({2 * pixels});
    tokensr1_tensor_pinned_t filled_tokens = make_zeros<tokensr1_tensor_pinned_t>({2 * tokens});
    tokensr1_tensor_pinned_t replayed_tokens = make_zeros<tokensr1_tensor_pinned_t>({2 * tokens});
    if (!expect_true(filled_pixels && replayed_pixels && filled_tokens && replayed_tokens,
                     "[{}] the readback buffers should allocate", name))
        return;
    auto const read_back = [&](pixelr1_tensor_pinned_t& into_pixels, tokensr1_tensor_pinned_t& into_tokens) noexcept {
        for (std::size_t batch = 0; batch != first.size(); ++batch)
            if (!util::copy_device_to_host(
                    first[batch].images.view().flatten<1>(),
                    nk::tensor_span<util::pixel_t, 1> {into_pixels.data() + batch * pixels, pixels}, stream) ||
                !util::copy_device_to_host(
                    first[batch].tokens.view().flatten<1>(),
                    nk::tensor_span<util::token_id_t, 1> {into_tokens.data() + batch * tokens, tokens}, stream))
                return false;
        return cuStreamSynchronize(util::native_stream(stream)) == CUDA_SUCCESS;
    };

    util::expected_t filled = loader.fill(stream, jpeg, 0u, 0u, first);
    if (filled) filled = loader.fill(stream, jpeg, 0u, 1u, second);
    if (!expect_true(filled, "[{}] the back-to-back fills should succeed: {}", name, filled.description_)) return;
    if (!expect_true(read_back(filled_pixels, filled_tokens), "[{}] the first readback", name)) return;
    if (util::expected_t replayed = loader.fill(stream, jpeg, 0u, 0u, first);
        !expect_true(replayed, "[{}] the replay should succeed: {}", name, replayed.description_))
        return;
    if (!expect_true(read_back(replayed_pixels, replayed_tokens), "[{}] the replay readback", name)) return;
    expect_true(std::memcmp(filled_pixels.data(), replayed_pixels.data(), 2 * pixels * sizeof(util::pixel_t)) == 0,
                "[{}] the first fill's pixels should match its replay", name);
    expect_true(std::memcmp(filled_tokens.data(), replayed_tokens.data(), 2 * tokens * sizeof(util::token_id_t)) == 0,
                "[{}] the first fill's tokens should match its replay", name);
}

/**
 *  @brief The distributed sampling tier: every rank bootstraps the mesh, draws its shard set from a
 *      shared dataset (the keyed weighted pick), publishes it to the rendezvous, and asserts its
 *      set differs from every peer's.
 *
 *  This is the cross-node "different nodes sample different files" contract the single-process
 *  oracle can only assert as arithmetic. Gated on @c UFORM_HOST_DOMAIN_INDEX ; a no-op without a
 *  @c UFORM_TEST_DATASET glob, so a laptop run skips it and a 4-8-16 launch exercises it (see
 *  @c loaders-4-8-16.sbatch ).
 */
static case_result_t test_cross_rank_sampling() {
    char const* const dataset = std::getenv("UFORM_TEST_DATASET");
    if (dataset == nullptr) return case_result_t::skipped("needs a shared UFORM_TEST_DATASET");

    util::u32_t const index = util::host_domain_index_from_environment();
    util::u32_t count = 0;
    for (char const* const variable : {"UFORM_HOST_DOMAIN_COUNT", "SLURM_NTASKS"})
        if (char const* const value = std::getenv(variable)) {
            count = static_cast<util::u32_t>(std::atoi(value));
            break;
        }
    if (!expect_true(count != 0, "world size should be set ($UFORM_HOST_DOMAIN_COUNT / $SLURM_NTASKS)")) return {};

    auto [topology, bootstrap_status] = util::mesh_topology_t::bootstrap(index, count, &pause_between_polls);
    if (!expect_true(bootstrap_status, "the cluster should bootstrap: {}", bootstrap_status.description_)) return {};

    auto [session, make_session_status] = gpu_direct_storage_session_t::make(1);
    if (!expect_true(make_session_status, "the GDS session should make: {}", make_session_status.description_))
        return {};

    auto [loader, make_loader_status] = tokens_loader_parquet_t::make(session, sz::string_view_t {dataset},
                                                                      0xC0FFEEull);
    if (!expect_true(make_loader_status, "the tokens loader should make: {}", make_loader_status.description_))
        return {};

    // This rank's shard set for step 0 — 64 segments is a generous pool target that
    // draws several shards.
    auto [shards, draw_status] = loader.drawn_shards(index, 0u, 64u);
    if (!expect_true(draw_status, "drawn_shards should succeed: {}", draw_status.description_)) return {};
    sz::string_t mine;
    for (sz::string_view_t shard : shards)
        if (!expect_true(util::format_into(mine, "{}\n", shard), "the shard list should format")) return {};

    if (util::expected_t published = topology.publish_text("shards", mine.view());
        !expect_true(published, "publishing the shard set should succeed: {}", published.description_))
        return {};

    bool distinct = true;
    for (std::size_t peer = 0; peer != count; ++peer) {
        if (peer == index) continue;
        sz::string_t other;
        if (util::expected_t awaited = topology.await_text(static_cast<util::u32_t>(peer), "shards", other,
                                                           wait_limit_k.count());
            !awaited)
            distinct = false;
        else if (other.view() == mine.view()) distinct = false; // a peer drew the byte-identical shard set
    }
    expect_true(distinct, "rank {} must draw a shard set distinct from every peer", index);
    expect_true(static_cast<bool>(topology.barrier("done", wait_limit_k.count())),
                "the closing barrier should complete");
    return {};
}

int main(int argc, char** argv) {
    start_test_run(argc, argv);

    run_test_case("test_draw_primitives_are_stable", test_draw_primitives_are_stable);
    run_test_case("test_jpeg_probe_resumes_past_metadata", test_jpeg_probe_resumes_past_metadata);
    run_test_case("test_glob_spans_depths_and_symlinks", test_glob_spans_depths_and_symlinks);
    run_test_case("test_mixture_sampler_retires_capped_entries", test_mixture_sampler_retires_capped_entries);
    run_test_case("test_file_pick_is_weighted_and_distinct", test_file_pick_is_weighted_and_distinct);
    run_test_case("test_segment_geometry_packs_multiple_per_row", test_segment_geometry_packs_multiple_per_row);
    run_test_case("test_media_fill_their_rows", test_media_fill_their_rows);

    int device_count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
        for (char const* label : {"test_fill_coords_routes_every_modality", "test_pack_tokens_starts_with_prefix",
                                  "test_multimodal_packet_routes", "test_audio_fault_isolation"})
            run_test_case(label, [] { return case_result_t::skipped(no_gpu_device_k); });
        return finish_test_run();
    }

    if (std::getenv("UFORM_HOST_DOMAIN_INDEX") != nullptr) { // the distributed sampling tier owns the process
        run_test_case("test_cross_rank_sampling", test_cross_rank_sampling);
        return finish_test_run();
    }

    int const stream_devices[] = {static_cast<int>(0)};
    auto [stream_executor, make_stream_status] = spawn_test_executor(stream_devices);
    if (!expect_true(make_stream_status, "stream make: {}", make_stream_status.description_)) return finish_test_run();
    auto& stream = stream_executor.stream(0);
    // Every decode tier below allocates pinned staging, and `cuMemAllocHost` needs a
    // current context — the executor binds a device only while it spawns, so the suite
    // holds the scope itself.
    [[maybe_unused]] util::status_t const gpu_scope = (stream).bind();
    if (!expect_true(gpu_scope.succeeded(), "context entry should succeed")) return finish_test_run();
    auto [arena, make_arena_status] = util::gpu_arena_t::make(0, 2ull << 30);
    if (!expect_true(make_arena_status, "gpu arena make: {}", util::to_string(make_arena_status.status_)))
        return finish_test_run();
    fu::topology_t topology;
    fu::pool_t pool;
    if (!expect_true(!fu::failed(topology.harvest()) &&
                         util::spawn_pool(pool, topology, 4, fu::caller_inclusive_k, fu_capabilities_all_k),
                     "the CPU decode pool should spawn"))
        return finish_test_run();

    // Tier C — the coordinate oracle, no external media.
    run_test_case("test_fill_coords_routes_every_modality", test_fill_coords_routes_every_modality, arena, stream);
    run_test_case("test_pack_tokens_starts_with_prefix", test_pack_tokens_starts_with_prefix, arena, stream);

    // Tier D — every UFORM_TEST_AUDIO clip through the audio decode contract.
    for_each_env_input("UFORM_TEST_AUDIO", [&](sz::string_view_t path) {
        run_test_case("test_audio_decode", test_audio_decode, "env-audio", read_fixture(path), arena, stream, pool);
    });

    // Image — every UFORM_TEST_IMAGE blob, plus the real audio+image mixed packet.
    auto [jpeg, make_jpeg_status] = nvjpeg_session_t::make();
    if (expect_true(make_jpeg_status, "nvjpeg_session_t::make should succeed: {}",
                    util::to_string(make_jpeg_status.status_))) {
        st::basic_vector<st::basic_vector<util::b8_t>> image_blobs;
        for_each_env_input("UFORM_TEST_IMAGE", [&](sz::string_view_t path) {
            run_test_case("test_image_decode", test_image_decode, "env-image", read_fixture(path), arena, stream, pool,
                          jpeg);
            expect_true(st::succeeded(image_blobs.push_back(read_fixture(path))), "the image fixture should append");
        });
        run_test_case("test_image_batch_matches_single", test_image_batch_matches_single, image_blobs, arena, stream,
                      pool, jpeg);
        run_test_case("test_multimodal_packet_routes", test_multimodal_packet_routes, arena, stream, pool, jpeg,
                      first_env_fixture("UFORM_TEST_AUDIO"), first_env_fixture("UFORM_TEST_IMAGE"));
        for_each_env_input("UFORM_TEST_MIXTURE", [&](sz::string_view_t path) {
            auto [mixture, read_status] = ut::mixture_t::read(path);
            if (!expect_true(read_status, "{} should read: {}", path, read_status.description_)) return;
            for (ut::mixture_entry_t const& entry : mixture.datasets)
                if (entry.modality == modality_t::image_k)
                    run_test_case("test_paired_fill_replays_after_back_to_back_fills",
                                  test_paired_fill_replays_after_back_to_back_fills, entry, arena, stream, pool, jpeg);
        });
    }

    // Video — external only (a codec-bearing clip can't be embedded cheaply); each file at
    // one representation, then two representations at distinct frame strides (a dense and a
    // sparse temporal sampling).
    auto [nvdec, make_nvdec_status] = nvdec_session_t::make(nvdec_config_t {2u, 512u});
    if (make_nvdec_status) {
        for_each_env_input("UFORM_TEST_VIDEO", [&](sz::string_view_t path) {
            st::basic_vector<util::b8_t> const clip = read_fixture(path);
            util::stack_array<util::u32_t, 1> one_stride;
            [[maybe_unused]] util::status_t const pushed = one_stride.push_back(1u);
            run_test_case(R"(test_video_decode("video-1view"))", test_video_decode, "video-1view", clip,
                          one_stride.view(), arena, stream, pool, nvdec);
            util::stack_array<util::u32_t, 2> two_strides;
            [[maybe_unused]] util::status_t const pushed_unit = two_strides.push_back(1u);
            [[maybe_unused]] util::status_t const pushed_strided = two_strides.push_back(4u);
            run_test_case(R"(test_video_decode("video-2view"))", test_video_decode, "video-2view", clip,
                          two_strides.view(), arena, stream, pool, nvdec);
        });
    }

    // Decode fault-isolation over the first real audio fixture.
    run_test_case("test_audio_fault_isolation", test_audio_fault_isolation, arena, stream, pool,
                  first_env_fixture("UFORM_TEST_AUDIO"));

    return finish_test_run();
}
