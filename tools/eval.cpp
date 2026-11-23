/**
 *  @file tools/eval.cpp
 *  @author Ash Vardanian
 *  @date October 21, 2026
 *  @brief The retrieval evaluation binary: text-to-media Recall@K of a checkpoint through
 *      the C interface.
 *
 *  It loads a checkpoint `uform_base --checkpoint` wrote onto the host, encodes each split's
 *  captions and media in batches into pooled embeddings, L2-normalizes them, and prints Recall@1,
 *  @5 and @10 in both directions:
 *
 *      - @c --coco : COCO 2017 val, 5000 images and their ~25K captions from
 *        @c annotations/captions_val2017.json . It is not the Karpathy 5K test split published COCO
 *        numbers use, and its lines say so.
 *      - @c --msrvtt : the MSR-VTT 1K-A test, @c videos/ clips with one caption each from
 *        @c msrvtt_test_1k.json .
 *      - @c --clotho : the Clotho evaluation set, @c audio/ recordings with five captions each from
 *        @c manifest.json .
 *      - @c --coco-sm : COCO-SM, 4993 Karpathy test images from @c --coco-sm-images with their
 *        captions machine-translated into 21 languages by three translators, one
 *        @c meta/<translator>.json each. The images encode once for all 63 splits, and every
 *        translator ends with its mean over languages.
 *
 *  A caption's rank counts the media scoring strictly above its own; a medium's rank is the best
 *  over its captions. Scores are dot products of the Float32 embeddings, which NumKong's packed
 *  kernels widen into Float64, scored block by block on host threads. A split whose modality the
 *  checkpoint was not trained on prints a skip line. With @c --gpu the model loads into that
 *  device's VRAM and encodes there instead of on host threads.
 *
 *  With @c --calibrate it fits each question type's inverse temperature of the choice head on the
 *  held-out choices that JSON-lines file lists, prints each group's negative log-likelihood and the
 *  top-choice calibration error before and after, and writes the checkpoint again at @c --output
 *  with them as @c choices.inverse_temperatures .
 *
 *  Every encoding prints, at most once a second, the items it has encoded of its split. The first
 *  SIGINT or SIGTERM stops it between batches: it prints the phase it stopped in, writes no
 *  calibrated checkpoint, and exits with 130.
 */
#include <cmath>   // `std::sqrt`
#include <csignal> // `std::signal`, `SIGINT`, `SIGTERM`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fopen`, `std::fread`
#include <cstring> // `std::memcpy`

#include <array>            // `std::array` - the flag table
#include <initializer_list> // `std::initializer_list`
#include <limits>           // `std::numeric_limits` — the peak a softmax starts below
#include <span>             // `std::span` — the fixed-size hit and recall tables
#include <string_view>      // `std::string_view`

#include <cuda.h>       // `cuMemcpyDtoH`
#include <fmt/format.h> // `fmt::println`, `fmt::format_to_n`
#include <yyjson.h>     // `yyjson_read_file`

#include <forkunion/atomics.hpp>           // `fu::standard_atomic_ref` - the scoring threads
#include <numkong/capabilities.h>          // `nk_capabilities_enabled_cpu`
#include <numkong/dots.h>                  // `nk_dots_pack_f32_best`, `nk_dots_packed_f32_best`
#include <smashtable/basic_hash_table.hpp> // `hash_map` — COCO image IDs to their files
#include <smashtable/basic_vector.hpp>     // `basic_vector` — every growable table here

#include "uform/encode.h"
#include "uform/model.h"
#include "uform/types.h"

#include "uform/utilities/safetensors_format.hpp" // `checkpoint_to_load_t`, `checkpoint_to_save_t` — the calibrated copy
#include "uform/utilities/types.hpp"              // `divide_round_up`
#include "uform/utilities/ram_device.hpp"         // `executor_t`, `pool_view_t`, `topology_t`

#include "arguments.hpp" // `flag_t`, `parse_or_exit`, `stop_requested`, `verb_phase_t`

namespace fu = ashvardanian::forkunion;
namespace st = ashvardanian::smashtable;
namespace util = unum::uform::utilities;
using util::caller_exclusivity_t;
using util::executor_t;
using util::expected;
using util::pool_view_t;
using util::topology_t;
using namespace unum::uform::tools;

/** Append @p count trivially copyable @p items onto @p values , or report the refusal. */
template <typename value_type_>
[[nodiscard]] static bool append_(st::basic_vector<value_type_>& values, std::span<value_type_ const> items) noexcept {
    std::size_t const size = values.size();
    if (st::failed(values.resize(st::assume_uninitialized, size + items.size()))) return false;
    if (!items.empty()) std::memcpy(values.data() + size, items.data(), items.size_bytes());
    return true;
}

/** Strings end to end with one more offset than strings, as a @c uform_tape_view_t
 *  lays them out. */
struct strings_t {
    st::basic_vector<char> bytes {};
    st::basic_vector<std::size_t> offsets {};

    /** Append one string made of @p parts in order. */
    [[nodiscard]] bool append(std::initializer_list<std::string_view> parts) noexcept {
        if (offsets.empty() && st::failed(offsets.push_back(std::size_t {0}))) return false;
        for (std::string_view part : parts)
            if (!append_<char>(bytes, std::span<char const> {part})) return false;
        return !st::failed(offsets.push_back(bytes.size()));
    }
    [[nodiscard]] std::size_t count() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
    [[nodiscard]] std::string_view operator[](std::size_t index) const noexcept {
        return {bytes.data() + offsets[index], offsets[index + 1] - offsets[index]};
    }
};

/** The encode verb a split's media take. */
enum class media_t { image_k, video_k, audio_k };

/** One split: its media files, its captions, and the medium each caption describes. */
struct split_t {
    char const* name = "";
    char const* media_name = "";
    media_t media = media_t::image_k;
    strings_t files {};
    strings_t captions {};
    st::basic_vector<std::uint32_t> owners {};
};

/** The model, the executor it runs on, the arena every call rewinds, the bytes every caption starts
 *  with, and on a GPU the primary context its outputs copy out through. */
struct encoder_t {
    uform_executor_t executor = nullptr;
    uform_arena_t arena = nullptr;
    uform_model_t model = nullptr;
    CUcontext context = nullptr;
    std::size_t dimensions = 0, batch = 256, threads = 0;
    std::string_view query_prefix {};
};

/** How one split's encoding ended. */
enum class encoded_t { ok_k, skipped_k, failed_k, stopped_k };

/** What a split's media count in, as its progress lines print them. */
[[nodiscard]] static std::string_view media_unit_(media_t media) noexcept {
    switch (media) {
    case media_t::image_k: return "images";
    case media_t::video_k: return "videos";
    case media_t::audio_k: return "recordings";
    }
    return "items";
}

/** Read @p path whole onto the end of @p bytes . */
[[nodiscard]] static bool read_file_(std::string_view path, st::basic_vector<std::uint8_t>& bytes) noexcept {
    char terminated[4096];
    if (path.size() >= sizeof(terminated)) return false;
    std::memcpy(terminated, path.data(), path.size()), terminated[path.size()] = '\0';
    std::FILE* file = std::fopen(terminated, "rb");
    if (!file) return fmt::println(stderr, "uform_eval: cannot open {}", path), false;
    std::fseek(file, 0, SEEK_END);
    long const length = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    std::size_t const size = bytes.size();
    bool read = length >= 0 && !st::failed(bytes.resize(st::assume_uninitialized, size + std::size_t(length))) &&
                std::fread(bytes.data() + size, 1, std::size_t(length), file) == std::size_t(length);
    std::fclose(file);
    if (!read) (void)bytes.resize(st::assume_uninitialized, size); // a shrink never fails
    return read;
}

/** Parse the JSON at @p directory / @p name , or print why not. */
[[nodiscard]] static yyjson_doc* read_json_(std::string_view directory, std::string_view name) noexcept {
    strings_t path;
    if (!path.append({directory, "/", name, std::string_view {"\0", 1}})) return nullptr;
    yyjson_doc* document = yyjson_read_file(path.bytes.data(), 0, nullptr, nullptr);
    if (!document) fmt::println(stderr, "uform_eval: cannot parse {}/{}", directory, name);
    return document;
}

[[nodiscard]] static bool load_coco_(std::string_view directory, split_t& split) noexcept {
    split.name = "coco val2017 (not Karpathy)", split.media_name = "image", split.media = media_t::image_k;
    yyjson_doc* document = read_json_(directory, "annotations/captions_val2017.json");
    if (!document) return false;
    yyjson_val* root = yyjson_doc_get_root(document);
    yyjson_val *images = yyjson_obj_get(root, "images"), *annotations = yyjson_obj_get(root, "annotations");
    st::hash_map<std::uint64_t, std::uint32_t> files_by_id;
    bool loaded = images && annotations;
    std::size_t index, count;
    yyjson_val* entry;
    yyjson_arr_foreach(images, index, count, entry) {
        char const* file = yyjson_get_str(yyjson_obj_get(entry, "file_name"));
        st::mapping<std::uint64_t, std::uint32_t> const id {yyjson_get_uint(yyjson_obj_get(entry, "id")),
                                                            std::uint32_t(split.files.count())};
        loaded = loaded && file && !st::failed(files_by_id.upsert(id)) &&
                 split.files.append({directory, "/val2017/", file});
    }
    yyjson_arr_foreach(annotations, index, count, entry) {
        std::uint64_t const image = yyjson_get_uint(yyjson_obj_get(entry, "image_id"));
        char const* caption = yyjson_get_str(yyjson_obj_get(entry, "caption"));
        auto const owner = files_by_id.find(image);
        if (owner == files_by_id.end() || !caption) continue;
        loaded = loaded && split.captions.append({caption}) &&
                 !st::failed(split.owners.push_back(std::uint32_t(owner.value())));
    }
    yyjson_doc_free(document);
    return loaded;
}

[[nodiscard]] static bool load_msrvtt_(std::string_view directory, split_t& split) noexcept {
    split.name = "msrvtt 1K-A", split.media_name = "video", split.media = media_t::video_k;
    yyjson_doc* document = read_json_(directory, "msrvtt_test_1k.json");
    if (!document) return false;
    bool loaded = true;
    std::size_t index, count;
    yyjson_val* entry;
    yyjson_arr_foreach(yyjson_doc_get_root(document), index, count, entry) {
        char const* file = yyjson_get_str(yyjson_obj_get(entry, "video"));
        char const* caption = yyjson_get_str(yyjson_obj_get(entry, "caption"));
        loaded = loaded && file && caption && split.files.append({directory, "/videos/", file}) &&
                 split.captions.append({caption}) &&
                 !st::failed(split.owners.push_back(std::uint32_t(split.files.count() - 1)));
    }
    yyjson_doc_free(document);
    return loaded && split.files.count();
}

[[nodiscard]] static bool load_clotho_(std::string_view directory, split_t& split) noexcept {
    split.name = "clotho v2 evaluation", split.media_name = "audio", split.media = media_t::audio_k;
    yyjson_doc* document = read_json_(directory, "manifest.json");
    if (!document) return false;
    bool loaded = true;
    std::size_t index, count;
    yyjson_val* entry;
    yyjson_arr_foreach(yyjson_obj_get(yyjson_doc_get_root(document), "items"), index, count, entry) {
        char const* file = yyjson_get_str(yyjson_obj_get(entry, "file"));
        loaded = loaded && file && split.files.append({directory, "/", file});
        std::size_t caption_index, caption_count;
        yyjson_val* caption;
        yyjson_arr_foreach(yyjson_obj_get(entry, "captions"), caption_index, caption_count, caption) {
            loaded = loaded && split.captions.append({yyjson_get_str(caption)}) &&
                     !st::failed(split.owners.push_back(std::uint32_t(split.files.count() - 1)));
        }
    }
    yyjson_doc_free(document);
    return loaded && split.files.count();
}

/** Load one COCO-SM language's @p entries into @p split : its images under @p images the first
 *  time, and every later time the check that they name the same files in the same order, so the
 *  image embeddings carry over. */
[[nodiscard]] static bool load_coco_sm_(std::string_view images, yyjson_val* entries, split_t& split) noexcept {
    bool const first = !split.files.count();
    bool loaded = yyjson_arr_size(entries) && (first || yyjson_arr_size(entries) == split.files.count());
    split.captions = {}, split.owners = {};
    std::size_t index, count;
    yyjson_val* entry;
    yyjson_arr_foreach(entries, index, count, entry) {
        char const* file = yyjson_get_str(yyjson_obj_get(entry, "filename"));
        loaded = loaded && file &&
                 (first ? split.files.append({images, "/", file})
                        : split.files[index].substr(images.size() + 1) == std::string_view {file});
        std::size_t caption_index, caption_count;
        yyjson_val* caption;
        yyjson_arr_foreach(yyjson_obj_get(entry, "captions"), caption_index, caption_count, caption) {
            loaded = loaded && split.captions.append({yyjson_get_str(caption)}) &&
                     !st::failed(split.owners.push_back(std::uint32_t(index)));
        }
    }
    return loaded;
}

/** Append the @p count F32 rows @p embeddings holds onto @p rows , each scaled to unit length,
 *  copying them off the device first when @p encoder runs on one. */
[[nodiscard]] static bool keep_rows_(encoder_t const& encoder, uform_rows_view_t const& embeddings, std::size_t count,
                                     st::basic_vector<float>& rows) noexcept {
    std::size_t const columns = embeddings.columns, size = rows.size();
    if (st::failed(rows.resize(st::assume_uninitialized, size + count * columns))) return false;
    float* const landed = rows.data() + size;
    void const* data = embeddings.data;
    std::size_t stride = embeddings.stride;
    if (encoder.context) {
        // Arena rows are packed, so the device's come over in one copy and scale in place.
        if (cuMemcpyDtoH(landed, reinterpret_cast<CUdeviceptr>(data), count * columns * sizeof(float)) != CUDA_SUCCESS)
            return false;
        data = landed, stride = columns * sizeof(float);
    }
    for (std::size_t item = 0; item != count; ++item) {
        float const* row = reinterpret_cast<float const*>(static_cast<std::uint8_t const*>(data) + item * stride);
        float squares = 0;
        for (std::size_t column = 0; column != columns; ++column) squares += row[column] * row[column];
        float const scale = squares > 0 ? 1.0f / std::sqrt(squares) : 0.0f;
        float* kept = landed + item * columns;
        for (std::size_t column = 0; column != columns; ++column) kept[column] = row[column] * scale;
    }
    return true;
}

/** Map the outcome of one batch onto how its split's encoding goes on. */
[[nodiscard]] static encoded_t judge_(uform_outcome_t const& outcome, split_t const& split) noexcept {
    if (outcome.status == uform_status_success_k) return encoded_t::ok_k;
    if (outcome.status == uform_status_cancelled_k) return encoded_t::stopped_k;
    if (outcome.status == uform_status_unsupported_k)
        return fmt::println("skip {}: {}", split.name, outcome.message ? outcome.message : "unsupported"),
               encoded_t::skipped_k;
    fmt::println(stderr, "uform_eval: {} failed: {}", split.name, outcome.message ? outcome.message : "no message");
    return encoded_t::failed_k;
}

/** Append the embeddings of every caption of @p split onto @p rows , batch by batch, reporting into
 *  @p phase and stopping between batches once a signal asks. */
[[nodiscard]] static encoded_t embed_captions_(encoder_t& encoder, split_t const& split, verb_phase_t& phase,
                                               st::basic_vector<float>& rows) noexcept {
    strings_t framed;
    for (std::size_t caption = 0; caption != split.captions.count(); ++caption)
        if (!framed.append({encoder.query_prefix, split.captions[caption]})) return encoded_t::failed_k;
    phase.unit = "captions", phase.total = framed.count();
    for (std::size_t first = 0; first < framed.count(); first += encoder.batch) {
        if (stop_requested) return encoded_t::stopped_k;
        std::size_t const count = fu::min_of(encoder.batch, framed.count() - first);
        phase.first = first;
        uform_outcome_t outcome {};
        uform_rows_view_t embeddings {};
        uform_text_encode_t arguments {.model = encoder.model,
                                       .executor = encoder.executor,
                                       .outcome = &outcome,
                                       .count = count,
                                       .texts = {framed.bytes.data(), framed.offsets.data() + first, uform_scalar_u8_k},
                                       .embeddings = &embeddings,
                                       .arena = encoder.arena,
                                       .outputs_scalar = uform_scalar_f32_k,
                                       .progress = verb_progress_of(phase)};
        uform_text_encode(&arguments);
        if (encoded_t const judged = judge_(outcome, split); judged != encoded_t::ok_k) return judged;
        if (!keep_rows_(encoder, embeddings, count, rows)) return encoded_t::failed_k;
    }
    return encoded_t::ok_k;
}

/** Append the embeddings of every medium of @p split onto @p rows , batch by batch, reporting into
 *  @p phase and stopping between batches once a signal asks. */
[[nodiscard]] static encoded_t embed_media_(encoder_t& encoder, split_t const& split, verb_phase_t& phase,
                                            st::basic_vector<float>& rows) noexcept {
    phase.unit = media_unit_(split.media), phase.total = split.files.count();
    for (std::size_t first = 0; first < split.files.count(); first += encoder.batch) {
        if (stop_requested) return encoded_t::stopped_k;
        std::size_t const count = fu::min_of(encoder.batch, split.files.count() - first);
        phase.first = first;
        st::basic_vector<std::uint8_t> bytes;
        st::basic_vector<std::size_t> offsets;
        if (st::failed(offsets.push_back(std::size_t {0}))) return encoded_t::failed_k;
        for (std::size_t item = first; item != first + count; ++item)
            if (!read_file_(split.files[item], bytes) || st::failed(offsets.push_back(bytes.size())))
                return encoded_t::failed_k;
        uform_tape_view_t const tape {bytes.data(), offsets.data(), uform_scalar_u8_k};
        uform_outcome_t outcome {};
        uform_rows_view_t embeddings {};
        switch (split.media) {
        case media_t::image_k: {
            uform_image_encode_t arguments {.model = encoder.model,
                                            .executor = encoder.executor,
                                            .outcome = &outcome,
                                            .count = count,
                                            .images = tape,
                                            .embeddings = &embeddings,
                                            .arena = encoder.arena,
                                            .outputs_scalar = uform_scalar_f32_k,
                                            .progress = verb_progress_of(phase)};
            uform_image_encode(&arguments);
        } break;
        case media_t::video_k: {
            uform_video_encode_t arguments {.model = encoder.model,
                                            .executor = encoder.executor,
                                            .outcome = &outcome,
                                            .count = count,
                                            .videos = tape,
                                            .embeddings = &embeddings,
                                            .arena = encoder.arena,
                                            .outputs_scalar = uform_scalar_f32_k,
                                            .progress = verb_progress_of(phase)};
            uform_video_encode(&arguments);
        } break;
        case media_t::audio_k: {
            uform_audio_encode_t arguments {.model = encoder.model,
                                            .executor = encoder.executor,
                                            .outcome = &outcome,
                                            .count = count,
                                            .recordings = tape,
                                            .embeddings = &embeddings,
                                            .arena = encoder.arena,
                                            .outputs_scalar = uform_scalar_f32_k,
                                            .progress = verb_progress_of(phase)};
            uform_audio_encode(&arguments);
        } break;
        }
        if (encoded_t const judged = judge_(outcome, split); judged != encoded_t::ok_k) return judged;
        if (!keep_rows_(encoder, embeddings, count, rows)) return encoded_t::failed_k;
    }
    return encoded_t::ok_k;
}

/** Print Recall@1, @5 and @10 of @p recalls_at , text-to-media then media-to-text, a line each. */
static void report_(split_t const& split, std::span<double const[3], 2> recalls_at) noexcept {
    std::size_t const counts[2] = {split.captions.count(), split.files.count()};
    char directions[2][32];
    *fmt::format_to_n(directions[0], sizeof(directions[0]) - 1, "text → {}", split.media_name).out = '\0';
    *fmt::format_to_n(directions[1], sizeof(directions[1]) - 1, "{} → text", split.media_name).out = '\0';
    for (std::size_t direction = 0; direction != 2; ++direction)
        fmt::println("{}  {:<14}  R@1 {:.4f}  R@5 {:.4f}  R@10 {:.4f}  {} queries over {} items", split.name,
                     directions[direction], recalls_at[direction][0], recalls_at[direction][1],
                     recalls_at[direction][2], counts[direction], counts[1 - direction]);
}

/** The ranks Recall@K reports. */
static constexpr std::size_t recall_depths_k[3] = {1, 5, 10};

/** Rows one score block spans, so a thread reuses each packed panel across a few dozen queries. */
static constexpr std::size_t score_block_rows_k = 64;

/** Score every @p queries row against every @p items row on @p pool , and add into @p hits how many
 *  queries rank within each of @ref recall_depths_k . A query's targets are @p targets from
 *  @p offsets [query] to @p offsets [query + 1], and its rank counts the items scoring strictly
 *  above the best of them. */
[[nodiscard]] static bool count_hits_(pool_view_t pool, nk_capability_t capabilities,
                                      st::basic_vector<float> const& queries, st::basic_vector<float> const& items,
                                      std::size_t dimensions, std::size_t const* offsets, std::uint32_t const* targets,
                                      std::span<std::size_t, 3> hits) noexcept {
    std::size_t const queries_count = queries.size() / dimensions, items_count = items.size() / dimensions;
    nk_size_t packed_bytes = 0;
    st::basic_vector<std::uint8_t> packed;
    st::basic_vector<nk_f64_t> blocks;
    if (nk_dots_pack_size_f32_best(items_count, dimensions, capabilities, &packed_bytes) != nk_success_k ||
        st::failed(packed.resize(st::assume_uninitialized, packed_bytes)) ||
        st::failed(blocks.resize(st::assume_uninitialized, pool.threads_count() * score_block_rows_k * items_count)) ||
        nk_dots_pack_f32_best(items.data(), items_count, dimensions, dimensions * sizeof(float), packed.data(), 0,
                              items_count, capabilities, nullptr) != nk_success_k)
        return false;
    bool scored = true;
    std::size_t const blocks_count = util::divide_round_up(queries_count, score_block_rows_k);
    util::status_t const dispatched = pool.for_n_dynamic(
        blocks_count, [&](std::size_t block, std::size_t thread) noexcept {
            std::size_t const first = block * score_block_rows_k;
            std::size_t const count = fu::min_of(score_block_rows_k, queries_count - first);
            nk_f64_t* const scores = blocks.data() + thread * score_block_rows_k * items_count;
            if (nk_dots_packed_f32_best(queries.data() + first * dimensions, packed.data(), scores, count, items_count,
                                        dimensions, dimensions * sizeof(float), items_count * sizeof(nk_f64_t), nullptr,
                                        0, capabilities, nullptr) != nk_success_k)
                return fu::standard_atomic_ref<bool> {scored}.store(false, std::memory_order_relaxed);
            std::size_t block_hits[3] = {};
            for (std::size_t query = first; query != first + count; ++query) {
                nk_f64_t const* row = scores + (query - first) * items_count;
                nk_f64_t best = -2;
                for (std::size_t position = offsets[query]; position != offsets[query + 1]; ++position)
                    best = fu::max_of(best, row[targets[position]]);
                std::size_t rank = 0;
                for (std::size_t item = 0; item != items_count; ++item) rank += row[item] > best;
                for (std::size_t k = 0; k != 3; ++k) block_hits[k] += rank < recall_depths_k[k];
            }
            for (std::size_t k = 0; k != 3; ++k)
                fu::standard_atomic_ref<std::size_t> {hits[k]}.fetch_add(block_hits[k], std::memory_order_relaxed);
        });
    return dispatched == util::status_code_t::success_k && scored; // `for_n_dynamic` joined every thread
}

/** Score every caption against every medium on @p encoder 's thread count, report both directions,
 *  and keep their @p recalls_at . */
[[nodiscard]] static bool score_(encoder_t const& encoder, split_t const& split,
                                 st::basic_vector<float> const& captions, st::basic_vector<float> const& media,
                                 std::span<double[3], 2> recalls_at) noexcept {
    std::size_t const queries = split.captions.count(), items = split.files.count();
    // Every medium's captions, gathered once by counting, so a medium ranks over its own alone.
    st::basic_vector<std::size_t> caption_offsets, owner_offsets;
    st::basic_vector<std::uint32_t> captions_by_item;
    if (st::failed(caption_offsets.resize(items + 1)) ||
        st::failed(owner_offsets.resize(st::assume_uninitialized, queries + 1)) ||
        st::failed(captions_by_item.resize(st::assume_uninitialized, queries)))
        return false;
    for (std::size_t query = 0; query != queries; ++query) ++caption_offsets[split.owners[query] + 1];
    for (std::size_t item = 0; item != items; ++item) caption_offsets[item + 1] += caption_offsets[item];
    for (std::size_t query = 0; query != queries; ++query)
        captions_by_item[caption_offsets[split.owners[query]]++] = std::uint32_t(query);
    // The scatter left each offset at its medium's end, so shift them back to the starts.
    for (std::size_t item = items; item != 0; --item) caption_offsets[item] = caption_offsets[item - 1];
    caption_offsets[0] = 0;
    // A caption's one target is its own medium.
    for (std::size_t query = 0; query <= queries; ++query) owner_offsets[query] = query;
    nk_capability_t capabilities = 0;
    if (nk_capabilities_enabled_cpu(&capabilities) != nk_success_k) return false;
    auto [topology, make_topology_status] = topology_t::make();
    if (!make_topology_status) return false;
    auto [pool, make_pool_status] = executor_t::make(topology,
                                                     encoder.threads ? encoder.threads : topology.logical_cores_count(),
                                                     caller_exclusivity_t::caller_inclusive_k);
    if (!make_pool_status) return false;
    std::size_t text_hits[3] = {}, media_hits[3] = {};
    if (!count_hits_(pool.view(), capabilities, captions, media, encoder.dimensions, owner_offsets.data(),
                     split.owners.data(), text_hits) ||
        !count_hits_(pool.view(), capabilities, media, captions, encoder.dimensions, caption_offsets.data(),
                     captions_by_item.data(), media_hits))
        return false;
    for (std::size_t k = 0; k != 3; ++k) {
        recalls_at[0][k] = double(text_hits[k]) / double(queries);
        recalls_at[1][k] = double(media_hits[k]) / double(items);
    }
    report_(split, recalls_at);
    return true;
}

/** Encode and score @p split , reporting into @p phase ; a stop scores nothing. */
[[nodiscard]] static bool evaluate_(encoder_t& encoder, split_t const& split, verb_phase_t& phase) noexcept {
    st::basic_vector<float> captions, media;
    encoded_t encoded = embed_media_(encoder, split, phase, media);
    if (encoded == encoded_t::ok_k) encoded = embed_captions_(encoder, split, phase, captions);
    if (encoded == encoded_t::skipped_k) return true;
    double recalls_at[2][3];
    return encoded == encoded_t::ok_k && score_(encoder, split, captions, media, recalls_at);
}

/** Evaluate COCO-SM: encode its images once, score every translator's every language against them,
 *  and report each translator's mean over its languages, reporting into @p phase and stopping
 *  between languages once a signal asks. */
[[nodiscard]] static bool evaluate_coco_sm_(encoder_t& encoder, std::string_view directory, std::string_view images,
                                            verb_phase_t& phase) noexcept {
    split_t split;
    split.media_name = "image", split.media = media_t::image_k;
    st::basic_vector<float> media;
    char name[64];
    for (char const* translator : {"google", "bing", "nllb"}) {
        *fmt::format_to_n(name, sizeof(name) - 1, "meta/{}.json", translator).out = '\0';
        yyjson_doc* document = read_json_(directory, name);
        if (!document) return false;
        double means_at[2][3] = {};
        std::size_t index, count;
        yyjson_val *language, *entries;
        yyjson_obj_foreach(yyjson_doc_get_root(document), index, count, language, entries) {
            *fmt::format_to_n(name, sizeof(name) - 1, "coco-sm {} {}", translator, yyjson_get_str(language)).out = '\0';
            split.name = name;
            if (stop_requested || !load_coco_sm_(images, entries, split)) return yyjson_doc_free(document), false;
            st::basic_vector<float> captions;
            double recalls_at[2][3];
            encoded_t encoded = media.empty() ? embed_media_(encoder, split, phase, media) : encoded_t::ok_k;
            if (encoded == encoded_t::ok_k) encoded = embed_captions_(encoder, split, phase, captions);
            if (encoded != encoded_t::ok_k || !score_(encoder, split, captions, media, recalls_at))
                return yyjson_doc_free(document), encoded == encoded_t::skipped_k;
            for (std::size_t direction = 0; direction != 2; ++direction)
                for (std::size_t k = 0; k != 3; ++k) means_at[direction][k] += recalls_at[direction][k] / double(count);
        }
        yyjson_doc_free(document);
        *fmt::format_to_n(name, sizeof(name) - 1, "coco-sm {} mean of {}", translator, count).out = '\0';
        split.name = name;
        report_(split, means_at);
    }
    return true;
}

/** Held-out choices, one per line of the @c --calibrate file: each state, its choice bytes, their
 *  target probabilities end to end, and its question type. */
struct choices_t {
    strings_t states {}, choices {};
    st::basic_vector<float> targets {};
    st::basic_vector<std::uint32_t> groups {};
};

/** Read the JSON lines at @p path , each `{"state": text, "choices": bytes, "targets": [p...],
 *  "group": n}` with one target per choice byte and @c group optional, into @p loaded . */
[[nodiscard]] static bool load_choices_(std::string_view path, choices_t& loaded) noexcept {
    st::basic_vector<std::uint8_t> bytes;
    if (!read_file_(path, bytes)) return false;
    std::string_view text {reinterpret_cast<char const*>(bytes.data()), bytes.size()};
    for (std::size_t line_number = 1; !text.empty(); ++line_number) {
        std::size_t const end = fu::min_of(text.find('\n'), text.size());
        std::string_view const line = text.substr(0, end);
        text.remove_prefix(fu::min_of(end + 1, text.size()));
        if (line.find_first_not_of(" \t\r") == std::string_view::npos) continue;
        yyjson_doc* document = yyjson_read(line.data(), line.size(), 0);
        yyjson_val* root = yyjson_doc_get_root(document);
        yyjson_val *state = yyjson_obj_get(root, "state"), *offered = yyjson_obj_get(root, "choices");
        yyjson_val *targets = yyjson_obj_get(root, "targets"), *group = yyjson_obj_get(root, "group");
        bool parsed = yyjson_is_str(state) && yyjson_is_str(offered) && yyjson_is_arr(targets) &&
                      yyjson_arr_size(targets) == yyjson_get_len(offered) && (!group || yyjson_is_uint(group)) &&
                      loaded.states.append({{yyjson_get_str(state), yyjson_get_len(state)}}) &&
                      loaded.choices.append({{yyjson_get_str(offered), yyjson_get_len(offered)}}) &&
                      !st::failed(loaded.groups.push_back(group ? std::uint32_t(yyjson_get_uint(group)) : 0u));
        for (std::size_t index = 0; parsed && index != yyjson_arr_size(targets); ++index)
            parsed = !st::failed(loaded.targets.push_back(float(yyjson_get_num(yyjson_arr_get(targets, index)))));
        yyjson_doc_free(document);
        if (!parsed) return fmt::println(stderr, "uform_eval: {} line {} is no choice", path, line_number), false;
    }
    return true;
}

/** The mean negative log-likelihood of @p targets under the softmax of @p logits at inverse
 *  temperature @p beta , over the items @p group holds, or every item for a group past them. */
[[nodiscard]] static double choice_loss_(choices_t const& held_out, std::span<double const> logits, double beta,
                                         std::uint32_t group) noexcept {
    double total = 0;
    std::size_t items = 0;
    for (std::size_t item = 0; item != held_out.states.count(); ++item) {
        if (held_out.groups[item] != group) continue;
        std::size_t const begin = held_out.choices.offsets[item], end = held_out.choices.offsets[item + 1];
        double peak = -std::numeric_limits<double>::infinity(), sum = 0, expected = 0;
        for (std::size_t choice = begin; choice != end; ++choice) peak = fu::max_of(peak, beta * logits[choice]);
        for (std::size_t choice = begin; choice != end; ++choice) sum += std::exp(beta * logits[choice] - peak);
        for (std::size_t choice = begin; choice != end; ++choice)
            expected += held_out.targets[choice] * (beta * logits[choice] - peak - std::log(sum));
        total -= expected, ++items;
    }
    return items ? total / double(items) : 0;
}

/** The share of items whose highest logit is the target's most likely choice, the same
 *  under every temperature. */
[[nodiscard]] static double top_choice_accuracy_(choices_t const& held_out, std::span<double const> logits) noexcept {
    std::size_t const items = held_out.states.count();
    std::size_t correct = 0;
    for (std::size_t item = 0; item != items; ++item) {
        std::size_t const begin = held_out.choices.offsets[item], end = held_out.choices.offsets[item + 1];
        std::size_t top = begin, wanted = begin;
        for (std::size_t choice = begin; choice != end; ++choice) {
            if (logits[choice] > logits[top]) top = choice;
            if (held_out.targets[choice] > held_out.targets[wanted]) wanted = choice;
        }
        correct += begin != end && top == wanted;
    }
    return items ? double(correct) / double(items) : 0;
}

/** The top-choice expected calibration error over 15 equal confidence bins, a choice correct when
 *  it is the target's most likely, under @p beta per group. */
[[nodiscard]] static double calibration_error_(choices_t const& held_out, std::span<double const> logits,
                                               std::span<double const> betas) noexcept {
    double confidence_sums[15] = {}, correct_sums[15] = {};
    std::size_t const items = held_out.states.count();
    for (std::size_t item = 0; item != items; ++item) {
        std::size_t const begin = held_out.choices.offsets[item], end = held_out.choices.offsets[item + 1];
        if (begin == end) continue;
        double const beta = betas[held_out.groups[item]];
        std::size_t top = begin, wanted = begin;
        double sum = 0;
        for (std::size_t choice = begin; choice != end; ++choice) {
            if (logits[choice] > logits[top]) top = choice;
            if (held_out.targets[choice] > held_out.targets[wanted]) wanted = choice;
        }
        for (std::size_t choice = begin; choice != end; ++choice)
            sum += std::exp(beta * (logits[choice] - logits[top]));
        double const confidence = 1 / sum;
        std::size_t const bin = fu::min_of<std::size_t>(std::size_t(confidence * 15), 14);
        confidence_sums[bin] += confidence, correct_sums[bin] += top == wanted;
    }
    double error = 0;
    for (std::size_t bin = 0; bin != 15; ++bin) error += std::fabs(confidence_sums[bin] - correct_sums[bin]);
    return items ? error / double(items) : 0;
}

/**
 *  @brief Fit each question type's inverse temperature on the held-out choices at @p choices_path
 *      and write @p checkpoint_path again at @p output_path with them.
 *
 *  The choices score once through @c uform_text_choose in group zero, whose stored temperature
 *  turns the probabilities back into logits up to each item's constant. Each group's temperature
 *  then minimizes its mean negative log-likelihood, convex in it, by a golden-section search over
 *  its logarithm in Float64.
 */
[[nodiscard]] static bool calibrate_(encoder_t& encoder, std::string_view checkpoint_path,
                                     std::string_view choices_path, std::string_view output_path,
                                     verb_phase_t& phase) noexcept {
    choices_t held_out;
    if (!load_choices_(choices_path, held_out)) return false;
    std::size_t const items = held_out.states.count(), offered = held_out.targets.size();
    std::uint32_t groups = 1;
    for (std::uint32_t const group : held_out.groups) groups = fu::max_of(groups, group + 1);
    auto [source, status] = util::checkpoint_to_load_t::make(checkpoint_path);
    if (!status) return fmt::println(stderr, "uform_eval: {}", std::string_view {status.description_}), false;
    util::f32r2_view_t const stored = source.float32("choices.inverse_temperatures");
    double const scoring_beta = stored.empty() ? 1.0 : double(stored[0, 0]);

    st::basic_vector<double> logits, betas;
    st::basic_vector<float> probabilities;
    if (st::failed(logits.resize(st::assume_uninitialized, offered)) ||
        st::failed(probabilities.resize(st::assume_uninitialized, offered)) ||
        st::failed(betas.resize(st::assume_uninitialized, groups)))
        return fmt::println(stderr, "uform_eval: calibration staging failed"), false;
    phase.total = items;
    for (std::size_t first = 0; first < items; first += encoder.batch) {
        if (stop_requested) return false;
        std::size_t const count = fu::min_of(encoder.batch, items - first);
        phase.first = first;
        uform_outcome_t outcome {};
        uform_tape_view_t chosen {};
        uform_text_choose_t arguments {
            .model = encoder.model,
            .executor = encoder.executor,
            .outcome = &outcome,
            .count = count,
            .texts = {held_out.states.bytes.data(), held_out.states.offsets.data() + first, uform_scalar_u8_k},
            .choices = {held_out.choices.bytes.data(), held_out.choices.offsets.data() + first, uform_scalar_u8_k},
            .probabilities = &chosen,
            .arena = encoder.arena,
            .progress = verb_progress_of(phase)};
        uform_text_choose(&arguments);
        if (outcome.status == uform_status_cancelled_k) return false;
        if (outcome.status != uform_status_success_k)
            return fmt::println(stderr, "uform_eval: choosing failed: {}", outcome.message), false;
        std::size_t const begin = held_out.choices.offsets[first],
                          scored = held_out.choices.offsets[first + count] - begin;
        float* const landed = probabilities.data() + begin;
        if (!encoder.context) std::memcpy(landed, chosen.data, scored * sizeof(float));
        else if (cuMemcpyDtoH(landed, reinterpret_cast<CUdeviceptr>(chosen.data), scored * sizeof(float)) !=
                 CUDA_SUCCESS)
            return fmt::println(stderr, "uform_eval: copying the probabilities failed"), false;
    }
    for (std::size_t choice = 0; choice != offered; ++choice)
        logits[choice] = std::log(fu::max_of(double(probabilities[choice]), 1e-30)) / scoring_beta;

    for (std::uint32_t group = 0; group != groups; ++group) {
        double low = std::log(1e-2), high = std::log(1e2);
        double const golden = (std::sqrt(5.0) - 1) / 2;
        for (std::size_t step = 0; step != 100; ++step) {
            double const left = high - golden * (high - low), right = low + golden * (high - low);
            if (choice_loss_(held_out, logits, std::exp(left), group) <
                choice_loss_(held_out, logits, std::exp(right), group))
                high = right;
            else low = left;
        }
        betas[group] = std::exp((low + high) / 2);
    }
    st::basic_vector<double> before;
    if (st::failed(before.resize(st::assume_uninitialized, groups)))
        return fmt::println(stderr, "uform_eval: calibration staging failed"), false;
    for (std::uint32_t group = 0; group != groups; ++group) {
        before[group] = group < stored.extent(1) ? double(stored[0, group]) : scoring_beta;
        fmt::println("calibrate group {}: inverse temperature {:.4f} -> {:.4f}, NLL {:.4f} -> {:.4f}", group,
                     before[group], betas[group], choice_loss_(held_out, logits, before[group], group),
                     choice_loss_(held_out, logits, betas[group], group));
    }
    fmt::println("calibrate: {} choices over {} groups, top-choice accuracy {:.4f}, ECE {:.4f} -> {:.4f}", items,
                 groups, top_choice_accuracy_(held_out, logits), calibration_error_(held_out, logits, before),
                 calibration_error_(held_out, logits, betas));

    st::basic_vector<float> fitted;
    if (st::failed(fitted.resize(st::assume_uninitialized, groups)))
        return fmt::println(stderr, "uform_eval: calibration staging failed"), false;
    for (std::uint32_t group = 0; group != groups; ++group) fitted[group] = float(betas[group]);
    util::checkpoint_to_save_t calibrated; // Float32 entries go first, packed ahead of the BFloat16 weights
    util::expected_t written = calibrated.insert("choices.inverse_temperatures",
                                                 util::f32r2_view_t {fitted.data(), 1, groups});
    for (util::checkpoint_entry_t const& entry : source.entries())
        if (written && entry.dtype == util::checkpoint_dtype_t::f32_k &&
            entry.name.view() != "choices.inverse_temperatures")
            written = calibrated.insert(entry.name.view(), source.float32(entry.name.view()));
    util::checkpoint_to_load_t const& weights = source;
    for (util::checkpoint_entry_t const& entry : weights.entries())
        if (written && entry.dtype == util::checkpoint_dtype_t::bf16_k)
            written = calibrated.insert(entry.name.view(), weights[entry.name.view()]);
    for (util::checkpoint_metadata_t const& pair : weights.metadata())
        if (written) written = calibrated.metadata(pair.key.view(), pair.value.view());
    // A stop arriving while the temperatures fit still writes nothing.
    if (stop_requested) return false;
    if (written) written = calibrated.save(output_path);
    if (!written) return fmt::println(stderr, "uform_eval: {}", std::string_view {written.description_}), false;
    fmt::println("calibrate: wrote {}", output_path);
    return true;
}

int main(int argc, char** argv) {
    // Progress lines reach a redirected log as they are printed.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    encoder_t encoder;
    std::string_view coco, msrvtt, clotho, coco_sm, coco_sm_images, query_prefix, calibrate, output;
    // An ordinal past every `int32_t` never parses, so the largest count marks the flag absent.
    std::size_t constexpr host_k = (std::numeric_limits<std::size_t>::max)();
    count_t batch {256}, threads {0, 0}, gpu {host_k, 0, (std::numeric_limits<std::int32_t>::max)()};
    std::array<flag_t, 11> const flags {{
        {"--coco", "directory", "COCO 2017 val, its annotations/ and val2017/", &coco},
        {"--msrvtt", "directory", "the MSR-VTT 1K-A test, its msrvtt_test_1k.json and videos/", &msrvtt},
        {"--clotho", "directory", "the Clotho evaluation set, its manifest.json and audio/", &clotho},
        {"--coco-sm", "directory", "COCO-SM, its meta/<translator>.json files", &coco_sm},
        {"--coco-sm-images", "directory", "the Karpathy test images COCO-SM captions", &coco_sm_images},
        {"--gpu", "ordinal", "encode in the VRAM of this GPU instead of on host threads", &gpu},
        {"--batch", "items", "items one encoding call takes", &batch},
        {"--threads", "count", "host threads encoding and scoring, zero for every core", &threads},
        {"--query-prefix", "text", "the bytes every caption starts with", &query_prefix},
        {"--calibrate", "choices.jsonl", "fit the choice head's temperatures on these held-out choices", &calibrate},
        {"--output", "calibrated.safetensors", "where --calibrate writes the checkpoint again", &output},
    }};
    std::array<std::string_view, 1> checkpoints;
    std::size_t const given = parse_or_exit(argc, argv, "uform_eval <checkpoint> [flags]", flags, checkpoints);
    if (given != 1) return fmt::println(stderr, "uform_eval: pass the checkpoint, see --help"), 1;
    if (coco_sm.empty() != coco_sm_images.empty())
        return fmt::println(stderr, "uform_eval: --coco-sm and --coco-sm-images come together"), 1;
    if (calibrate.empty() != output.empty())
        return fmt::println(stderr, "uform_eval: --calibrate and --output come together"), 1;
    encoder.batch = batch.value, encoder.threads = threads.value, encoder.query_prefix = query_prefix;
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);

    uform_outcome_t outcome {};
    bool const on_gpu = gpu.value != host_k;
    std::int32_t const device = static_cast<std::int32_t>(gpu.value);
    CUdevice handle = 0;
    if (on_gpu) {
        uform_executor_for_vram_t executor_arguments {.executor = &encoder.executor,
                                                      .outcome = &outcome,
                                                      .vram = {&device, 1}};
        uform_executor_for_vram(&executor_arguments);
        // Every call pushes and pops its own context, so copying outputs out needs the
        // device's made current.
        if (outcome.status == uform_status_success_k &&
            (cuDeviceGet(&handle, device) != CUDA_SUCCESS ||
             cuDevicePrimaryCtxRetain(&encoder.context, handle) != CUDA_SUCCESS ||
             cuCtxSetCurrent(encoder.context) != CUDA_SUCCESS))
            return fmt::println(stderr, "uform_eval: cannot make GPU {} current", device), 1;
    }
    else {
        uform_executor_for_ram_t executor_arguments {.executor = &encoder.executor,
                                                     .outcome = &outcome,
                                                     .threads = encoder.threads};
        uform_executor_for_ram(&executor_arguments);
    }
    if (outcome.status != uform_status_success_k) return fmt::println(stderr, "uform_eval: {}", outcome.message), 1;
    uform_arena_init_t arena_arguments {.arena = &encoder.arena, .executor = encoder.executor, .outcome = &outcome};
    uform_arena_init(&arena_arguments);
    if (outcome.status != uform_status_success_k) return fmt::println(stderr, "uform_eval: {}", outcome.message), 1;
    std::string_view const checkpoint = checkpoints[0];
    verb_phase_t phase {.name = "load", .unit = "tensors"};
    if (on_gpu) {
        uform_model_load_vram_t load_arguments {.model = &encoder.model,
                                                .executor = encoder.executor,
                                                .outcome = &outcome,
                                                .path = bytes_view(checkpoint),
                                                .vram = {&device, 1},
                                                .quantization = uform_scalar_bf16_k,
                                                .progress = verb_progress_of(phase)};
        uform_model_load_vram(&load_arguments);
    }
    else {
        uform_model_load_ram_t load_arguments {.model = &encoder.model,
                                               .executor = encoder.executor,
                                               .outcome = &outcome,
                                               .path = bytes_view(checkpoint),
                                               .quantization = uform_scalar_bf16_k,
                                               .progress = verb_progress_of(phase)};
        uform_model_load_ram(&load_arguments);
    }
    if (outcome.status != uform_status_success_k) {
        uform_arena_free(encoder.arena);
        uform_executor_free(encoder.executor);
        if (encoder.context) cuDevicePrimaryCtxRelease(handle);
        if (stop_requested) return verb_stopped(phase), exit_stopped_k;
        return fmt::println(stderr, "uform_eval: {}", outcome.message), 1;
    }
    {
        uform_model_description_t description {};
        uform_model_describe_t describe_arguments {.model = encoder.model,
                                                   .executor = encoder.executor,
                                                   .outcome = &outcome,
                                                   .description = &description};
        uform_model_describe(&describe_arguments);
        encoder.dimensions = description.dimensions;
        fmt::println("uform_eval: {} ({} dimensions, context {}, {})", checkpoint, description.dimensions,
                     description.context,
                     description.cpu_class != uform_cpu_class_unknown_k ? uform_cpu_class_name(description.cpu_class)
                                                                        : uform_gpu_class_name(description.gpu_class));
    }

    bool evaluated = true;
    // Each phase is named after its flag, and the last one started is the one a stop names.
    struct named_t {
        std::string_view flag, directory;
        bool (*load)(std::string_view, split_t&) noexcept;
    } const splits[] = {{"coco", coco, load_coco_}, {"msrvtt", msrvtt, load_msrvtt_}, {"clotho", clotho, load_clotho_}};
    for (named_t const& named : splits) {
        if (named.directory.empty() || stop_requested) continue;
        phase = {.name = named.flag};
        split_t split;
        if (!named.load(named.directory, split)) {
            evaluated = false;
            continue;
        }
        evaluated = evaluate_(encoder, split, phase) && evaluated;
    }
    if (!coco_sm.empty() && !stop_requested) {
        phase = {.name = "coco-sm"};
        evaluated = evaluate_coco_sm_(encoder, coco_sm, coco_sm_images, phase) && evaluated;
    }
    if (!calibrate.empty() && !stop_requested) {
        phase = {.name = "calibrate", .unit = "choices"};
        evaluated = calibrate_(encoder, checkpoint, calibrate, output, phase) && evaluated;
    }
    uform_model_free(encoder.model);
    uform_arena_free(encoder.arena);
    uform_executor_free(encoder.executor);
    if (encoder.context) cuDevicePrimaryCtxRelease(handle);
    if (stop_requested) return verb_stopped(phase), exit_stopped_k;
    return evaluated ? 0 : 1;
}
