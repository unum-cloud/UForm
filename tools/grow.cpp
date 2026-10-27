/**
 *  @file tools/grow.cpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The checkpoint growth binary: one step up the tier ladder, doubling a checkpoint's depth
 *      or its width while the grown tower computes the same function.
 *
 *  It rewrites the weights a `uform_base --checkpoint` file holds, records the grown geometry at
 *  step 0, and checks the result through the C interface:
 *
 *      - @c --deepen : every block is followed by its copy with zero @c out and @c down , whose
 *        attention and MLP add exact zeros onto the Float32 residual stream, so the copy passes it
 *        through.
 *      - @c --widen : the residual stream becomes @b [x,x] and the heads @b h0…hn,h0…hn . Every row
 *        writing a widened vector repeats, the Q, K and V thirds of @c qkv and the gate and up
 *        halves of @c gate_up each on their own, and every column reading a widened vector repeats
 *        at half its value. RMSNorm keeps its mean square over a repeated vector, while the
 *        QK-norm, the rotary pairs and both attentions run per 128-channel head, so every state and
 *        embedding opens with the source's own.
 *
 *  The optimizer moments stay behind, so a @c --resume restarts them from zero and warms up afresh;
 *  the contrastive temperatures and biases carry over. A tensor with no rule here stops the tool
 *  before it writes anything.
 *
 *  Halving a BFloat16 weight is exact, so a widened product differs from the source's only in the
 *  order its Float32 sum runs, and the embedding comes out as @b [v,v] , whose prefix is @c v
 *  itself. The check loads both files into the VRAM of GPU 0, encodes three dozen captions and,
 *  with @c --images , the first 16 files of that directory, and requires every grown embedding's
 *  prefix as wide as the source's to meet it at a cosine of 0.9999. On a miss it prints the worst
 *  item, deletes the output, and exits with 1.
 */
#include <cmath>   // `std::sqrt`
#include <cstdio>  // `std::fopen`, `std::fread`, `std::remove`
#include <cstring> // `std::memset`, `std::memcpy`, `std::strcmp`

#include <dirent.h> // `opendir`, `readdir` — the image directory

#include <initializer_list> // `std::initializer_list`
#include <string_view>      // `std::string_view`

#include <cuda.h>       // `cuMemcpyDtoH`
#include <fmt/format.h> // `fmt::println`, `fmt::format_to_n`

#include <numkong/capabilities.h>      // `nk_cpu_capabilities_enabled`
#include <numkong/each.h>              // `nk_each_scale_bf16_best` — the halved columns
#include <smashtable/basic_vector.hpp> // `basic_vector` — every growable table here

#include "uform/encode.h"
#include "uform/model.h"
#include "uform/types.h"

#include "uform/utilities/safetensors_format.hpp"
#include "uform/encoder/geometry.hpp"

namespace nk = ashvardanian::numkong;
namespace st = ashvardanian::smashtable;
namespace sz = ashvardanian::stringzilla;
namespace ue = unum::uform::encoder;

using namespace unum::uform::utilities;

/** The cosine every grown embedding's prefix must reach against the source's. */
static constexpr float cosine_floor_k = 0.9999f;

/** Files the check encodes from the @c --images directory. */
static constexpr std::size_t images_count_k = 16;

/** The step up the ladder: twice the blocks, or twice the width. */
enum class step_t { deepen_k, widen_k };

/** What widening does to a matrix's columns: they read an input of unchanged width, they hold a
 *  widened vector and repeat, or they read a widened vector and repeat at half their value. */
enum class columns_t { kept_k, repeated_k, halved_k };

/** What the copy deepening inserts after a block holds of one of its weights. */
enum class copy_t { kept_k, zeroed_k };

/** One weight's growth: its name past @c model. or @c model.blocks.{i}. , how many equal bands of
 *  rows each repeat in place when widening, zero keeping the rows, its columns, and its copy. */
struct growth_t {
    char const *name;
    std::size_t row_bands;
    columns_t columns;
    copy_t copy;
};

/** Every weight outside the blocks, each optional. */
static constexpr growth_t tower_growths_k[] = {
    {"embedding", 0, columns_t::repeated_k, copy_t::kept_k},
    {"image_projection", 1, columns_t::kept_k, copy_t::kept_k},
    {"video_projection", 1, columns_t::kept_k, copy_t::kept_k},
    {"audio_projection", 1, columns_t::kept_k, copy_t::kept_k},
    {"final_norm", 0, columns_t::repeated_k, copy_t::kept_k},
    {"pool_query", 0, columns_t::repeated_k, copy_t::kept_k},
    {"head.norm", 0, columns_t::repeated_k, copy_t::kept_k},
    {"head.fc1", 1, columns_t::halved_k, copy_t::kept_k},
    {"head.fc2", 1, columns_t::halved_k, copy_t::kept_k},
    {"token_head.norm", 0, columns_t::repeated_k, copy_t::kept_k},
    {"token_head.projection", 0, columns_t::halved_k, copy_t::kept_k},
    {"masked_tokens", 0, columns_t::halved_k, copy_t::kept_k},
    {"masked_image", 0, columns_t::halved_k, copy_t::kept_k},
    {"masked_video", 0, columns_t::halved_k, copy_t::kept_k},
    {"masked_audio", 0, columns_t::halved_k, copy_t::kept_k},
};

/** Every weight of a block, each required. */
static constexpr growth_t block_growths_k[] = {
    {"qkv", 3, columns_t::halved_k, copy_t::kept_k},        {"out", 1, columns_t::halved_k, copy_t::zeroed_k},
    {"gate_up", 2, columns_t::halved_k, copy_t::kept_k},    {"down", 1, columns_t::halved_k, copy_t::zeroed_k},
    {"pre_norm", 0, columns_t::repeated_k, copy_t::kept_k}, {"post_norm", 0, columns_t::repeated_k, copy_t::kept_k},
};

/** The geometry keys that pass through unchanged; @c columns is absent from older files. */
static constexpr char const *kept_metadata_k[] = {
    "ffn_expansion",      "head_width",       "rms_epsilon",      "rope_temporal",    "rope_height",
    "rope_width",         "rope_base",        "context",          "columns",          "image_patch_side",
    "image_channels",     "image_pixel_mean", "image_pixel_std",  "video_patch_side", "video_tube_frames",
    "video_frame_stride", "video_channels",   "video_pixel_mean", "video_pixel_std",  "audio_chunk_samples",
    "audio_sample_rate",
};

/** Insert @p source as @p name , widened by @p row_bands and @p columns through @p staging ; zero
 *  bands and kept columns copy it as it is. */
[[nodiscard]] static expected_t insert_grown_(checkpoint_to_save_t &grown, sz::string_view_t name,
                                              nk::tensor_view<nk_bf16_t, 2> source, std::size_t row_bands,
                                              columns_t columns, st::basic_vector<nk_bf16_t> &staging) noexcept {
    std::size_t const rows = source.extent(0), cols = source.extent(1);
    std::size_t const band = row_bands ? rows / row_bands : rows;
    std::size_t const grown_rows = row_bands ? 2 * rows : rows;
    std::size_t const grown_cols = columns == columns_t::kept_k ? cols : 2 * cols;
    if (st::failed(staging.resize(st::assume_uninitialized, grown_rows * grown_cols)))
        return expected_t::fail(status_t::bad_alloc_k, "{} staging failed", name);
    nk::tensor_span<nk_bf16_t, 2> const target {staging.data(), grown_rows, grown_cols};
    nk_capability_t capabilities = 0;
    if (columns == columns_t::halved_k && nk_cpu_capabilities_enabled(&capabilities) != nk_success_k)
        return expected_t::fail(status_t::unsupported_k, "{} found no CPU capabilities", name);
    nk_f32_t const half = 0.5f, zero = 0.0f;
    for (std::size_t row = 0; row != grown_rows; ++row) {
        std::size_t const source_row = row_bands ? row / (2 * band) * band + row % band : row;
        nk_bf16_t const *const from = &source[source_row, 0];
        for (std::size_t first = 0; first != grown_cols; first += cols) {
            nk_bf16_t *const to = &target[row, first];
            if (columns != columns_t::halved_k) std::memcpy(to, from, cols * sizeof(nk_bf16_t));
            else if (nk_each_scale_bf16_best(from, cols, &half, &zero, to, capabilities, nullptr) != nk_success_k)
                return expected_t::fail(status_t::unsupported_k, "{} halving failed", name);
        }
    }
    return grown.insert(name, nk::tensor_view<nk_bf16_t, 2> {staging.data(), grown_rows, grown_cols});
}

/** Insert the zero matrix shaped like @p source as @p name through @p staging . */
[[nodiscard]] static expected_t insert_zeros_(checkpoint_to_save_t &grown, sz::string_view_t name,
                                              nk::tensor_view<nk_bf16_t, 2> source,
                                              st::basic_vector<nk_bf16_t> &staging) noexcept {
    std::size_t const rows = source.extent(0), cols = source.extent(1);
    if (st::failed(staging.resize(st::assume_uninitialized, rows * cols)))
        return expected_t::fail(status_t::bad_alloc_k, "{} staging failed", name);
    std::memset(staging.data(), 0, rows * cols * sizeof(nk_bf16_t)); // BFloat16 +0 is all zero bits
    return grown.insert(name, nk::tensor_view<nk_bf16_t, 2> {staging.data(), rows, cols});
}

/** Fill @p grown with the weights and geometry of @p source taken one @p step up, refusing a source
 *  tensor this neither grows, carries nor leaves behind. */
[[nodiscard]] static expected_t grow_(checkpoint_to_load_t const &source, ue::geometry_t const &geometry, step_t step,
                                      checkpoint_to_save_t &grown) noexcept {
    std::size_t accounted = 0;
    // The Float32 state goes first, as the payload packs it ahead of the BFloat16 weights.
    for (char const *carried : {"contrastive.inverse_temperatures", "contrastive.biases",
                                "late_interaction.inverse_temperatures", "late_interaction.biases"}) {
        f32r2_view_t const values = source.float32(carried);
        if (values.empty()) continue;
        ++accounted;
        if (expected_t inserted = grown.insert(carried, values); !inserted) return inserted;
    }
    for (char const *moments : {"contrastive.temperature_moments", "late_interaction.temperature_moments"})
        accounted += !source.float32(moments).empty();

    sz::string_t suffix, name;
    // The source weight `model.{suffix}`, counted with the optimizer state it leaves behind.
    auto find = [&]() noexcept {
        name.clear();
        nk::tensor_view<nk_bf16_t, 2> found {};
        if (format_into(name, "model.{}", suffix.view())) found = source[name.view()];
        if (found.empty()) return found;
        ++accounted;
        for (char const *part : {"row_moments", "column_moments", "master", "first_moment"}) {
            sz::string_t moment;
            if (format_into(moment, "optimizer.{}.{}", suffix.view(), part))
                accounted += !source.float32(moment.view()).empty();
        }
        return found;
    };
    st::basic_vector<nk_bf16_t> staging;
    bool const widened = step == step_t::widen_k;
    for (growth_t const &growth : tower_growths_k) {
        if (sz::failed(suffix.assign(growth.name)))
            return expected_t::fail(status_t::bad_alloc_k, "name staging failed");
        nk::tensor_view<nk_bf16_t, 2> const weights = find();
        if (weights.empty()) continue;
        if (expected_t inserted = insert_grown_(grown, name.view(), weights, widened ? growth.row_bands : 0,
                                                widened ? growth.columns : columns_t::kept_k, staging);
            !inserted)
            return inserted;
    }
    sz::string_t grown_name;
    for (std::size_t block = 0; block != geometry.depth; ++block)
        for (growth_t const &growth : block_growths_k) {
            suffix.clear();
            if (!format_into(suffix, "blocks.{}.{}", block, growth.name))
                return expected_t::fail(status_t::bad_alloc_k, "name staging failed");
            nk::tensor_view<nk_bf16_t, 2> const weights = find();
            if (weights.empty())
                return expected_t::fail(status_t::invalid_argument_k, "the source misses model.{}", suffix.view());
            if (widened) {
                if (expected_t inserted = insert_grown_(grown, name.view(), weights, growth.row_bands, growth.columns,
                                                        staging);
                    !inserted)
                    return inserted;
                continue;
            }
            for (std::size_t instance = 0; instance != 2; ++instance) {
                grown_name.clear();
                if (!format_into(grown_name, "model.blocks.{}.{}", 2 * block + instance, growth.name))
                    return expected_t::fail(status_t::bad_alloc_k, "name staging failed");
                expected_t inserted = instance != 0 && growth.copy == copy_t::zeroed_k
                                          ? insert_zeros_(grown, grown_name.view(), weights, staging)
                                          : insert_grown_(grown, grown_name.view(), weights, 0, columns_t::kept_k,
                                                          staging);
                if (!inserted) return inserted;
            }
        }
    if (accounted != source.size())
        return expected_t::fail(status_t::invalid_argument_k, "{} of the source's {} tensors have no growth rule",
                                source.size() - accounted, source.size());

    for (char const *key : kept_metadata_k)
        if (sz::string_view_t const value = source.metadata(key); !value.empty())
            if (expected_t recorded = grown.metadata(key, value); !recorded) return recorded;
    sz::string_t depth, hidden;
    if (!format_into(depth, "{}", geometry.depth * (widened ? 1u : 2u)) ||
        !format_into(hidden, "{}", geometry.hidden * (widened ? 2u : 1u)))
        return expected_t::fail(status_t::bad_alloc_k, "metadata staging failed");
    expected_t recorded = grown.metadata("step", "0");
    if (recorded) recorded = grown.metadata("depth", depth.view());
    if (recorded) recorded = grown.metadata("hidden", hidden.view());
    return recorded;
}

/** Strings end to end with one more offset than strings, as a @c uform_tape_view_t lays them
 *  out. */
struct strings_t {
    st::basic_vector<char> bytes {};
    st::basic_vector<std::size_t> offsets {};

    /** Append one string made of @p parts in order. */
    [[nodiscard]] bool append(std::initializer_list<std::string_view> parts) noexcept {
        if (offsets.empty() && st::failed(offsets.push_back(std::size_t {0}))) return false;
        for (std::string_view part : parts) {
            std::size_t const size = bytes.size();
            if (st::failed(bytes.resize(st::assume_uninitialized, size + part.size()))) return false;
            if (part.size()) std::memcpy(bytes.data() + size, part.data(), part.size());
        }
        return !st::failed(offsets.push_back(bytes.size()));
    }

    /** Append the whole file at @p path as one string. */
    [[nodiscard]] bool append_file(char const *path) noexcept {
        if (offsets.empty() && st::failed(offsets.push_back(std::size_t {0}))) return false;
        std::FILE *file = std::fopen(path, "rb");
        if (!file) return false;
        std::fseek(file, 0, SEEK_END);
        long const length = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        std::size_t const size = bytes.size();
        bool const read = length > 0 &&
                          !st::failed(bytes.resize(st::assume_uninitialized, size + std::size_t(length))) &&
                          std::fread(bytes.data() + size, 1, std::size_t(length), file) == std::size_t(length);
        std::fclose(file);
        return read && !st::failed(offsets.push_back(bytes.size()));
    }
    [[nodiscard]] std::size_t count() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
    [[nodiscard]] std::string_view operator[](std::size_t index) const noexcept {
        return {bytes.data() + offsets[index], offsets[index + 1] - offsets[index]};
    }
    [[nodiscard]] uform_tape_view_t tape() const noexcept { return {bytes.data(), offsets.data(), uform_scalar_u8_k}; }
};

/** The captions every check encodes, short enough for any context, in a dozen scripts. */
static constexpr std::string_view captions_k[] = {
    "a dog running across a grassy field",
    "two people riding bicycles down a city street",
    "a plate of pasta with tomato sauce and basil",
    "an airplane taking off at sunset",
    "a cat sleeping on a windowsill",
    "a red double-decker bus in London",
    "children playing soccer on a beach",
    "a bowl of fresh fruit on a wooden table",
    "a man surfing a large wave",
    "snow-covered mountains under a clear blue sky",
    "a train crossing a bridge over a river",
    "a woman reading a book in a park",
    "a laptop and a cup of coffee on a desk",
    "a herd of elephants walking through the savanna",
    "fireworks over a harbor at night",
    "a close-up of a bee on a sunflower",
    "un perro corriendo por la playa",
    "une tour Eiffel illuminée la nuit",
    "ein rotes Auto parkt vor einem Haus",
    "un gatto che dorme sul divano",
    "um barco de pesca no porto",
    "een fietser op een dijk in de regen",
    "кошка сидит на подоконнике",
    "собака играет с мячом в парке",
    "一只猫在沙发上睡觉",
    "繁忙的城市街道上的出租车",
    "桜の木の下でピクニックをする家族",
    "雪の中を走る電車",
    "바닷가에서 노는 아이들",
    "رجل يركب دراجة في الشارع",
    "एक लड़का पतंग उड़ा रहा है",
    "γάτα που κοιμάται στον ήλιο",
    "חתול יושב על גדר",
    "köpek parkta top oynuyor",
    "một bát phở nóng trên bàn",
    "ช้างกำลังเดินในป่า",
};

/** The GPU the check encodes on: its executor, the arena every call rewinds, and the primary
 *  context outputs copy out through. */
struct session_t {
    uform_executor_t executor = nullptr;
    uform_arena_t arena = nullptr;
    CUcontext context = nullptr;
    CUdevice device = 0;
};

/** Append the @p count Float32 rows @p embeddings holds in VRAM onto @p rows . */
[[nodiscard]] static bool keep_rows_(uform_rows_view_t const &embeddings, std::size_t count,
                                     st::basic_vector<float> &rows) noexcept {
    std::size_t const columns = embeddings.columns, size = rows.size();
    if (st::failed(rows.resize(st::assume_uninitialized, size + count * columns))) return false;
    CUdeviceptr const data = reinterpret_cast<CUdeviceptr>(embeddings.data);
    for (std::size_t item = 0; item != count; ++item)
        if (cuMemcpyDtoH(rows.data() + size + item * columns, data + item * embeddings.stride,
                         columns * sizeof(float)) != CUDA_SUCCESS)
            return false;
    return true;
}

/** Load @p checkpoint into VRAM and append the embeddings of @p captions , then of @p images , onto
 *  @p rows , @p dimensions wide; or print why not. */
[[nodiscard]] static bool encode_(session_t const &session, std::string_view checkpoint, strings_t const &captions,
                                  strings_t const &images, std::size_t &dimensions,
                                  st::basic_vector<float> &rows) noexcept {
    uform_outcome_t outcome {};
    uform_model_t model = nullptr;
    std::int32_t const device = 0;
    uform_model_load_vram_t load_arguments {
        .model = &model,
        .executor = session.executor,
        .outcome = &outcome,
        .path = {reinterpret_cast<std::uint8_t const *>(checkpoint.data()), checkpoint.size()},
        .vram = {&device, 1},
        .quantization = uform_scalar_bf16_k};
    uform_model_load_vram(&load_arguments);
    uform_model_description_t description {};
    if (outcome.status == uform_status_success_k) {
        uform_model_describe_t describe_arguments {.model = model,
                                                   .executor = session.executor,
                                                   .outcome = &outcome,
                                                   .description = &description};
        uform_model_describe(&describe_arguments);
        dimensions = description.dimensions;
    }
    uform_rows_view_t embeddings {};
    if (outcome.status == uform_status_success_k) {
        uform_text_encode_t arguments {.model = model,
                                       .executor = session.executor,
                                       .outcome = &outcome,
                                       .count = captions.count(),
                                       .texts = captions.tape(),
                                       .embeddings = &embeddings,
                                       .arena = session.arena,
                                       .outputs_scalar = uform_scalar_f32_k};
        uform_text_encode(&arguments);
    }
    bool kept = outcome.status == uform_status_success_k && keep_rows_(embeddings, captions.count(), rows);
    if (kept && images.count()) {
        uform_image_encode_t arguments {.model = model,
                                        .executor = session.executor,
                                        .outcome = &outcome,
                                        .count = images.count(),
                                        .images = images.tape(),
                                        .embeddings = &embeddings,
                                        .arena = session.arena,
                                        .outputs_scalar = uform_scalar_f32_k};
        uform_image_encode(&arguments);
        kept = outcome.status == uform_status_success_k && keep_rows_(embeddings, images.count(), rows);
    }
    if (!kept)
        fmt::println(stderr, "uform_grow: {}: {}", checkpoint,
                     outcome.status == uform_status_success_k ? "copying embeddings off the GPU failed"
                     : outcome.message                        ? outcome.message
                                                              : "no message");
    uform_model_free(model);
    return kept;
}

/** The item a check found at the lowest cosine, and that cosine. */
struct worst_t {
    std::size_t item = 0;
    float cosine = 1.0f;
};

/** The item among @p first to @p end whose @p grown embedding's first @p dimensions meet the
 *  @p source one at the lowest cosine. */
[[nodiscard]] static worst_t worst_(st::basic_vector<float> const &source, st::basic_vector<float> const &grown,
                                    std::size_t dimensions, std::size_t grown_dimensions, std::size_t first,
                                    std::size_t end) noexcept {
    worst_t worst {first, 1.0f};
    for (std::size_t item = first; item != end; ++item) {
        float const *left = source.data() + item * dimensions;
        float const *right = grown.data() + item * grown_dimensions;
        float dot = 0, left_squares = 0, right_squares = 0;
        for (std::size_t column = 0; column != dimensions; ++column) {
            dot += left[column] * right[column];
            left_squares += left[column] * left[column];
            right_squares += right[column] * right[column];
        }
        float const cosine = dot / std::sqrt(left_squares * right_squares);
        if (!(cosine >= worst.cosine)) worst = {item, cosine}; // a NaN is the worst of all
    }
    return worst;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fmt::println(stderr, "Usage: uform_grow <in.safetensors> <out.safetensors> --deepen | --widen " //
                             "[--images <dir>]");
        return 1;
    }
    char const *source_path = argv[1], *grown_path = argv[2];
    step_t step = step_t::deepen_k;
    std::size_t steps = 0;
    char const *images_directory = nullptr;
    for (int index = 3; index != argc; ++index) {
        char const *flag = argv[index];
        if (!std::strcmp(flag, "--deepen")) step = step_t::deepen_k, ++steps;
        else if (!std::strcmp(flag, "--widen")) step = step_t::widen_k, ++steps;
        else if (!std::strcmp(flag, "--images") && index + 1 != argc) images_directory = argv[++index];
        else return fmt::println(stderr, "uform_grow: unknown flag {}", flag), 1;
    }
    if (steps != 1) return fmt::println(stderr, "uform_grow: pass one of --deepen and --widen"), 1;

    std::size_t hidden = 0, depth = 0;
    {
        expected<checkpoint_to_load_t> source = checkpoint_to_load_t::make(source_path);
        expected<ue::geometry_t> geometry = source ? ue::geometry_t::from(*source)
                                                   : expected<ue::geometry_t>::fail(std::move(source));
        checkpoint_to_save_t grown;
        expected_t written = geometry ? grow_(*source, *geometry, step, grown) : expected_t::fail(std::move(geometry));
        if (written) written = grown.save(grown_path);
        if (!written) return fmt::println(stderr, "uform_grow: {}", std::string_view {written.description_}), 1;
        hidden = geometry->hidden, depth = geometry->depth;
    } // the source unmaps and the grown payload frees before the check loads both again

    strings_t captions, images, image_names;
    for (std::string_view caption : captions_k)
        if (!captions.append({caption})) return fmt::println(stderr, "uform_grow: caption staging failed"), 1;
    if (images_directory) {
        DIR *directory = opendir(images_directory);
        if (!directory) return fmt::println(stderr, "uform_grow: cannot open {}", images_directory), 1;
        char path[4096];
        for (dirent *entry = readdir(directory); entry && images.count() != images_count_k;
             entry = readdir(directory)) {
            if (entry->d_name[0] == '.') continue;
            *fmt::format_to_n(path, sizeof(path) - 1, "{}/{}", images_directory, entry->d_name).out = '\0';
            if (!images.append_file(path) || !image_names.append({entry->d_name}))
                return closedir(directory), fmt::println(stderr, "uform_grow: cannot read {}", path), 1;
        }
        closedir(directory);
    }

    uform_outcome_t outcome {};
    session_t session;
    std::int32_t const device = 0;
    uform_executor_for_vram_t executor_arguments {.executor = &session.executor,
                                                  .outcome = &outcome,
                                                  .vram = {&device, 1}};
    uform_executor_for_vram(&executor_arguments);
    if (outcome.status == uform_status_success_k) {
        uform_arena_init_t arena_arguments {.arena = &session.arena, .executor = session.executor, .outcome = &outcome};
        uform_arena_init(&arena_arguments);
    }
    if (outcome.status != uform_status_success_k)
        return fmt::println(stderr, "uform_grow: {}", outcome.message ? outcome.message : "no GPU"), 1;
    // Every call pushes and pops its own context, so copies need the device's context current.
    if (cuDeviceGet(&session.device, device) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&session.context, session.device) != CUDA_SUCCESS ||
        cuCtxSetCurrent(session.context) != CUDA_SUCCESS)
        return fmt::println(stderr, "uform_grow: cannot make GPU {} current", device), 1;

    st::basic_vector<float> source_rows, grown_rows;
    std::size_t source_dimensions = 0, grown_dimensions = 0;
    bool checked = encode_(session, source_path, captions, images, source_dimensions, source_rows) &&
                   encode_(session, grown_path, captions, images, grown_dimensions, grown_rows) &&
                   source_dimensions == hidden && grown_dimensions >= hidden;
    if (checked) {
        std::size_t const texts = captions.count(), items = texts + images.count();
        worst_t const worst_text = worst_(source_rows, grown_rows, hidden, grown_dimensions, 0, texts);
        worst_t const worst_image = worst_(source_rows, grown_rows, hidden, grown_dimensions, texts, items);
        fmt::println("uform_grow: {} blocks of {} → {} blocks of {}", depth, hidden,
                     step == step_t::deepen_k ? 2 * depth : depth, grown_dimensions);
        fmt::println("uform_grow: worst text cosine {:.7f} over {} captions, at \"{}\"", worst_text.cosine, texts,
                     captions[worst_text.item]);
        if (images.count())
            fmt::println("uform_grow: worst image cosine {:.7f} over {} images, at {}", worst_image.cosine,
                         images.count(), image_names[worst_image.item - texts]);
        checked = worst_text.cosine >= cosine_floor_k && worst_image.cosine >= cosine_floor_k;
    }
    uform_arena_free(session.arena);
    uform_executor_free(session.executor);
    cuDevicePrimaryCtxRelease(session.device);
    if (checked) return fmt::println("uform_grow: wrote {}", grown_path), 0;
    std::remove(grown_path);
    fmt::println(stderr, "uform_grow: the check failed below a cosine of {}, so {} is deleted", cosine_floor_k,
                 grown_path);
    return 1;
}
