/**
 *  @file tools/eval.cpp
 *  @author Ash Vardanian
 *  @date October 21, 2026
 *  @brief The retrieval evaluation binary: text-to-media Recall@K of a checkpoint through the C
 *      interface.
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
 *
 *  A caption's rank counts the media scoring strictly above its own; a medium's rank is the best
 *  over its captions. Scores are Float32 dot products. A split whose modality the checkpoint was
 *  not trained on prints a skip line.
 */
#include <cmath>   // `std::sqrt`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fopen`, `std::fread`
#include <cstdlib> // `std::strtoull`
#include <cstring> // `std::memcpy`, `std::strcmp`

#include <algorithm>        // `std::min`, `std::max`
#include <initializer_list> // `std::initializer_list`
#include <memory>           // `std::unique_ptr`
#include <new>              // `std::nothrow`
#include <string_view>      // `std::string_view`

#include <fmt/format.h> // `fmt::println`
#include <yyjson.h>     // `yyjson_read_file`

#include "uform/encode.h"
#include "uform/model.h"
#include "uform/types.h"

/** A growable run of trivially copyable values that fails softly instead of throwing. */
template <typename value_type_>
struct buffer {
    std::unique_ptr<value_type_[]> values {};
    std::size_t size = 0, capacity = 0;

    [[nodiscard]] bool reserve(std::size_t wanted) noexcept {
        if (wanted <= capacity) return true;
        std::size_t grown = capacity ? capacity : 64;
        while (grown < wanted) grown *= 2;
        std::unique_ptr<value_type_[]> moved {new (std::nothrow) value_type_[grown]};
        if (!moved) return false;
        if (size) std::memcpy(moved.get(), values.get(), size * sizeof(value_type_));
        values = std::move(moved), capacity = grown;
        return true;
    }
    [[nodiscard]] bool append(value_type_ const *items, std::size_t count) noexcept {
        if (!reserve(size + count)) return false;
        if (count) std::memcpy(values.get() + size, items, count * sizeof(value_type_));
        size += count;
        return true;
    }
    [[nodiscard]] bool push(value_type_ value) noexcept { return append(&value, 1); }
    [[nodiscard]] value_type_ *data() const noexcept { return values.get(); }
    [[nodiscard]] value_type_ &operator[](std::size_t index) const noexcept { return values[index]; }
};

/** Strings end to end with one more offset than strings, as a @c uform_tape_view_t lays them
 *  out. */
struct strings_t {
    buffer<char> bytes {};
    buffer<std::size_t> offsets {};

    /** Append one string made of @p parts in order. */
    [[nodiscard]] bool append(std::initializer_list<std::string_view> parts) noexcept {
        if (!offsets.size && !offsets.push(0)) return false;
        for (std::string_view part : parts)
            if (!bytes.append(part.data(), part.size())) return false;
        return offsets.push(bytes.size);
    }
    [[nodiscard]] std::size_t count() const noexcept { return offsets.size ? offsets.size - 1 : 0; }
    [[nodiscard]] std::string_view operator[](std::size_t index) const noexcept {
        return {bytes.data() + offsets[index], offsets[index + 1] - offsets[index]};
    }
};

/** The encode verb a split's media take. */
enum class media_t { image_k, video_k, audio_k };

/** One split: its media files, its captions, and the medium each caption describes. */
struct split_t {
    char const *name = "";
    char const *media_name = "";
    media_t media = media_t::image_k;
    strings_t files {};
    strings_t captions {};
    buffer<std::uint32_t> owners {};
};

/** The model, the host executor it runs on, the arena every call rewinds, and the bytes every
 *  caption starts with. */
struct encoder_t {
    uform_executor_t executor = nullptr;
    uform_arena_t arena = nullptr;
    uform_model_t model = nullptr;
    std::size_t dimensions = 0, batch = 256, threads = 0;
    std::string_view query_prefix {};
};

/** How one split's encoding ended. */
enum class encoded_t { ok_k, skipped_k, failed_k };

/** Read @p path whole onto the end of @p bytes . */
[[nodiscard]] static bool read_file_(std::string_view path, buffer<std::uint8_t> &bytes) noexcept {
    char terminated[4096];
    if (path.size() >= sizeof(terminated)) return false;
    std::memcpy(terminated, path.data(), path.size()), terminated[path.size()] = '\0';
    std::FILE *file = std::fopen(terminated, "rb");
    if (!file) return fmt::println(stderr, "uform_eval: cannot open {}", path), false;
    std::fseek(file, 0, SEEK_END);
    long const length = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    bool const read = length >= 0 && bytes.reserve(bytes.size + std::size_t(length)) &&
                      std::fread(bytes.data() + bytes.size, 1, std::size_t(length), file) == std::size_t(length);
    std::fclose(file);
    if (read) bytes.size += std::size_t(length);
    return read;
}

/** Parse the JSON at @p directory / @p name , or print why not. */
[[nodiscard]] static yyjson_doc *read_json_(std::string_view directory, std::string_view name) noexcept {
    strings_t path;
    if (!path.append({directory, "/", name, std::string_view {"\0", 1}})) return nullptr;
    yyjson_doc *document = yyjson_read_file(path.bytes.data(), 0, nullptr, nullptr);
    if (!document) fmt::println(stderr, "uform_eval: cannot parse {}/{}", directory, name);
    return document;
}

[[nodiscard]] static bool load_coco_(std::string_view directory, split_t &split) noexcept {
    split.name = "coco val2017 (not Karpathy)", split.media_name = "image", split.media = media_t::image_k;
    yyjson_doc *document = read_json_(directory, "annotations/captions_val2017.json");
    if (!document) return false;
    yyjson_val *root = yyjson_doc_get_root(document);
    yyjson_val *images = yyjson_obj_get(root, "images"), *annotations = yyjson_obj_get(root, "annotations");
    buffer<std::uint64_t> ids;
    bool loaded = images && annotations;
    std::size_t index, count;
    yyjson_val *entry;
    yyjson_arr_foreach(images, index, count, entry) {
        char const *file = yyjson_get_str(yyjson_obj_get(entry, "file_name"));
        loaded = loaded && file && ids.push(yyjson_get_uint(yyjson_obj_get(entry, "id"))) &&
                 split.files.append({directory, "/val2017/", file});
    }
    yyjson_arr_foreach(annotations, index, count, entry) {
        std::uint64_t const image = yyjson_get_uint(yyjson_obj_get(entry, "image_id"));
        char const *caption = yyjson_get_str(yyjson_obj_get(entry, "caption"));
        std::size_t owner = 0;
        while (owner != ids.size && ids[owner] != image) ++owner;
        if (owner == ids.size || !caption) continue;
        loaded = loaded && split.captions.append({caption}) && split.owners.push(std::uint32_t(owner));
    }
    yyjson_doc_free(document);
    return loaded;
}

[[nodiscard]] static bool load_msrvtt_(std::string_view directory, split_t &split) noexcept {
    split.name = "msrvtt 1K-A", split.media_name = "video", split.media = media_t::video_k;
    yyjson_doc *document = read_json_(directory, "msrvtt_test_1k.json");
    if (!document) return false;
    bool loaded = true;
    std::size_t index, count;
    yyjson_val *entry;
    yyjson_arr_foreach(yyjson_doc_get_root(document), index, count, entry) {
        char const *file = yyjson_get_str(yyjson_obj_get(entry, "video"));
        char const *caption = yyjson_get_str(yyjson_obj_get(entry, "caption"));
        loaded = loaded && file && caption && split.files.append({directory, "/videos/", file}) &&
                 split.captions.append({caption}) && split.owners.push(std::uint32_t(split.files.count() - 1));
    }
    yyjson_doc_free(document);
    return loaded && split.files.count();
}

[[nodiscard]] static bool load_clotho_(std::string_view directory, split_t &split) noexcept {
    split.name = "clotho v2 evaluation", split.media_name = "audio", split.media = media_t::audio_k;
    yyjson_doc *document = read_json_(directory, "manifest.json");
    if (!document) return false;
    bool loaded = true;
    std::size_t index, count;
    yyjson_val *entry;
    yyjson_arr_foreach(yyjson_obj_get(yyjson_doc_get_root(document), "items"), index, count, entry) {
        char const *file = yyjson_get_str(yyjson_obj_get(entry, "file"));
        loaded = loaded && file && split.files.append({directory, "/", file});
        std::size_t caption_index, caption_count;
        yyjson_val *caption;
        yyjson_arr_foreach(yyjson_obj_get(entry, "captions"), caption_index, caption_count, caption) {
            loaded = loaded && split.captions.append({yyjson_get_str(caption)}) &&
                     split.owners.push(std::uint32_t(split.files.count() - 1));
        }
    }
    yyjson_doc_free(document);
    return loaded && split.files.count();
}

/** Append the @p count F32 rows @p embeddings holds onto @p rows , each scaled to unit length. */
[[nodiscard]] static bool keep_rows_(uform_rows_view_t const &embeddings, std::size_t count,
                                     buffer<float> &rows) noexcept {
    std::size_t const columns = embeddings.columns;
    if (!rows.reserve(rows.size + count * columns)) return false;
    for (std::size_t item = 0; item != count; ++item) {
        float const *row = reinterpret_cast<float const *>(static_cast<std::uint8_t const *>(embeddings.data) +
                                                           item * embeddings.stride);
        float squares = 0;
        for (std::size_t column = 0; column != columns; ++column) squares += row[column] * row[column];
        float const scale = squares > 0 ? 1.0f / std::sqrt(squares) : 0.0f;
        float *kept = rows.data() + rows.size;
        for (std::size_t column = 0; column != columns; ++column) kept[column] = row[column] * scale;
        rows.size += columns;
    }
    return true;
}

/** Map the outcome of one batch onto how its split's encoding goes on. */
[[nodiscard]] static encoded_t judge_(uform_outcome_t const &outcome, split_t const &split) noexcept {
    if (outcome.status == uform_status_success_k) return encoded_t::ok_k;
    if (outcome.status == uform_status_unsupported_k)
        return fmt::println("skip {}: {}", split.name, outcome.message ? outcome.message : "unsupported"),
               encoded_t::skipped_k;
    fmt::println(stderr, "uform_eval: {} failed: {}", split.name, outcome.message ? outcome.message : "no message");
    return encoded_t::failed_k;
}

[[nodiscard]] static encoded_t embed_captions_(encoder_t &encoder, split_t const &split, buffer<float> &rows) noexcept {
    strings_t framed;
    for (std::size_t caption = 0; caption != split.captions.count(); ++caption)
        if (!framed.append({encoder.query_prefix, split.captions[caption]})) return encoded_t::failed_k;
    for (std::size_t first = 0; first < framed.count(); first += encoder.batch) {
        std::size_t const count = std::min(encoder.batch, framed.count() - first);
        uform_outcome_t outcome {};
        uform_rows_view_t embeddings {};
        uform_text_encode_t arguments {.model = encoder.model,
                                       .executor = encoder.executor,
                                       .outcome = &outcome,
                                       .count = count,
                                       .texts = {framed.bytes.data(), framed.offsets.data() + first, uform_scalar_u8_k},
                                       .embeddings = &embeddings,
                                       .arena = encoder.arena,
                                       .outputs_scalar = uform_scalar_f32_k};
        uform_text_encode(&arguments);
        if (encoded_t const judged = judge_(outcome, split); judged != encoded_t::ok_k) return judged;
        if (!keep_rows_(embeddings, count, rows)) return encoded_t::failed_k;
    }
    return encoded_t::ok_k;
}

[[nodiscard]] static encoded_t embed_media_(encoder_t &encoder, split_t const &split, buffer<float> &rows) noexcept {
    for (std::size_t first = 0; first < split.files.count(); first += encoder.batch) {
        std::size_t const count = std::min(encoder.batch, split.files.count() - first);
        buffer<std::uint8_t> bytes;
        buffer<std::size_t> offsets;
        if (!offsets.push(0)) return encoded_t::failed_k;
        for (std::size_t item = first; item != first + count; ++item)
            if (!read_file_(split.files[item], bytes) || !offsets.push(bytes.size)) return encoded_t::failed_k;
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
                                            .outputs_scalar = uform_scalar_f32_k};
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
                                            .outputs_scalar = uform_scalar_f32_k};
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
                                            .outputs_scalar = uform_scalar_f32_k};
            uform_audio_encode(&arguments);
        } break;
        }
        if (encoded_t const judged = judge_(outcome, split); judged != encoded_t::ok_k) return judged;
        if (!keep_rows_(embeddings, count, rows)) return encoded_t::failed_k;
    }
    return encoded_t::ok_k;
}

/** Print Recall@1, @5 and @10 of the three @p hits_at over @p queries , one line naming the
 *  direction. */
static void report_(split_t const &split, char const *direction, std::size_t const *hits_at, std::size_t queries,
                    std::size_t items) noexcept {
    fmt::println("{}  {:<14}  R@1 {:.4f}  R@5 {:.4f}  R@10 {:.4f}  {} queries over {} items", split.name, direction,
                 double(hits_at[0]) / double(queries), double(hits_at[1]) / double(queries),
                 double(hits_at[2]) / double(queries), queries, items);
}

/** Score every caption against every medium in Float32 and report both directions. */
[[nodiscard]] static bool score_(split_t const &split, buffer<float> const &captions, buffer<float> const &media,
                                 std::size_t dimensions) noexcept {
    std::size_t const queries = split.captions.count(), items = split.files.count();
    buffer<float> scores;
    if (!scores.reserve(queries * items)) return false;
    for (std::size_t query = 0; query != queries; ++query) {
        float const *left = captions.data() + query * dimensions;
        float *row = scores.data() + query * items;
        for (std::size_t item = 0; item != items; ++item) {
            float const *right = media.data() + item * dimensions;
            float dot = 0;
            for (std::size_t column = 0; column != dimensions; ++column) dot += left[column] * right[column];
            row[item] = dot;
        }
    }
    std::size_t constexpr ks[3] = {1, 5, 10};
    std::size_t text_hits[3] = {}, media_hits[3] = {};
    for (std::size_t query = 0; query != queries; ++query) {
        float const *row = scores.data() + query * items;
        float const own = row[split.owners[query]];
        std::size_t rank = 0;
        for (std::size_t item = 0; item != items; ++item) rank += row[item] > own;
        for (std::size_t k = 0; k != 3; ++k) text_hits[k] += rank < ks[k];
    }
    for (std::size_t item = 0; item != items; ++item) {
        float const *column = scores.data() + item;
        float best = -2.0f;
        for (std::size_t query = 0; query != queries; ++query)
            if (split.owners[query] == item) best = std::max(best, column[query * items]);
        std::size_t rank = 0;
        for (std::size_t query = 0; query != queries; ++query) rank += column[query * items] > best;
        for (std::size_t k = 0; k != 3; ++k) media_hits[k] += rank < ks[k];
    }
    char direction[32];
    std::snprintf(direction, sizeof(direction), "text → %s", split.media_name);
    report_(split, direction, text_hits, queries, items);
    std::snprintf(direction, sizeof(direction), "%s → text", split.media_name);
    report_(split, direction, media_hits, items, queries);
    return true;
}

[[nodiscard]] static bool evaluate_(encoder_t &encoder, split_t const &split) noexcept {
    buffer<float> captions, media;
    encoded_t encoded = embed_media_(encoder, split, media);
    if (encoded == encoded_t::ok_k) encoded = embed_captions_(encoder, split, captions);
    if (encoded == encoded_t::skipped_k) return true;
    return encoded == encoded_t::ok_k && score_(split, captions, media, encoder.dimensions);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fmt::println(stderr, "Usage: uform_eval <checkpoint> [--coco <dir>] [--msrvtt <dir>] [--clotho <dir>] " //
                             "[--batch <items>] [--threads <count>] [--query-prefix <text>]");
        return 1;
    }
    encoder_t encoder;
    char const *coco = nullptr, *msrvtt = nullptr, *clotho = nullptr;
    for (int index = 2; index + 1 < argc; index += 2) {
        char const *flag = argv[index], *value = argv[index + 1];
        if (!std::strcmp(flag, "--coco")) coco = value;
        else if (!std::strcmp(flag, "--msrvtt")) msrvtt = value;
        else if (!std::strcmp(flag, "--clotho")) clotho = value;
        else if (!std::strcmp(flag, "--batch")) encoder.batch = std::strtoull(value, nullptr, 10);
        else if (!std::strcmp(flag, "--threads")) encoder.threads = std::strtoull(value, nullptr, 10);
        else if (!std::strcmp(flag, "--query-prefix")) encoder.query_prefix = value;
        else return fmt::println(stderr, "uform_eval: unknown flag {}", flag), 1;
    }
    if (!encoder.batch) return fmt::println(stderr, "uform_eval: --batch takes a positive count"), 1;

    uform_outcome_t outcome {};
    uform_executor_for_ram_t executor_arguments {.executor = &encoder.executor,
                                                 .outcome = &outcome,
                                                 .threads = encoder.threads};
    uform_executor_for_ram(&executor_arguments);
    if (outcome.status != uform_status_success_k) return fmt::println(stderr, "uform_eval: {}", outcome.message), 1;
    uform_arena_init_t arena_arguments {.arena = &encoder.arena, .executor = encoder.executor, .outcome = &outcome};
    uform_arena_init(&arena_arguments);
    if (outcome.status != uform_status_success_k) return fmt::println(stderr, "uform_eval: {}", outcome.message), 1;
    std::string_view const checkpoint = argv[1];
    uform_model_load_ram_t load_arguments {
        .model = &encoder.model,
        .executor = encoder.executor,
        .outcome = &outcome,
        .path = {reinterpret_cast<std::uint8_t const *>(checkpoint.data()), checkpoint.size()},
        .quantization = uform_scalar_bf16_k};
    uform_model_load_ram(&load_arguments);
    if (outcome.status != uform_status_success_k) return fmt::println(stderr, "uform_eval: {}", outcome.message), 1;
    uform_model_description_t description {};
    uform_model_describe_t describe_arguments {.model = encoder.model,
                                               .executor = encoder.executor,
                                               .outcome = &outcome,
                                               .description = &description};
    uform_model_describe(&describe_arguments);
    encoder.dimensions = description.dimensions;
    fmt::println("uform_eval: {} ({} dimensions, context {}, {})", checkpoint, description.dimensions,
                 description.context, description.capabilities ? description.capabilities : "?");

    bool evaluated = true;
    struct named_t {
        char const *directory;
        bool (*load)(std::string_view, split_t &) noexcept;
    } const splits[] = {{coco, load_coco_}, {msrvtt, load_msrvtt_}, {clotho, load_clotho_}};
    for (named_t const &named : splits) {
        if (!named.directory) continue;
        split_t split;
        if (!named.load(named.directory, split)) {
            evaluated = false;
            continue;
        }
        evaluated = evaluate_(encoder, split) && evaluated;
    }
    uform_model_free(encoder.model);
    uform_arena_free(encoder.arena);
    uform_executor_free(encoder.executor);
    return evaluated ? 0 : 1;
}
