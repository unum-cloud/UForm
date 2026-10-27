/**
 *  @file src/train/datasets.cu
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The mixture reader and the dataset source declared in @c uform/train/datasets.hpp .
 */
#include <cmath> // `std::ceil` — the reservoir's per-row term at the mixture's density

#include <algorithm> // `std::upper_bound`, `std::min`, `std::max`, `std::any_of`
#include <utility>   // `std::swap` — a step's batches into the mixer's table and back

#include <cuda.h> // `cuStreamSynchronize`

#include <yyjson.h> // the mixture JSON — exceptions-free C

#include "uform/train/datasets.hpp"

namespace fu = ashvardanian::forkunion;
namespace ue = unum::uform::encoder;
using ue::audio_batch_shape_t;
using ue::byte_pad_k;
using ue::image_batch_shape_t;
using ue::late_interaction_side_t;
using ue::modality_t;
using ue::multimodal_batch_mixer_t;
using ue::multimodal_batch_on_gpu_t;
using ue::multimodal_batch_shape_t;
using ue::pad_choices;
using ue::segment_role_t;
using ue::video_batch_shape_t;
using namespace unum::uform::utilities;
using namespace unum::uform::loaders;
using namespace unum::uform::train;

/** The media column names each modality goes by: the Hugging Face feature names and the WebDataset
 *  extensions. */
[[nodiscard]] static modality_t modality_of_(sz::string_view_t column) noexcept {
    for (char const *name : {"image", "jpg", "jpeg", "png", "webp"})
        if (column == name) return modality_t::image_k;
    for (char const *name : {"video", "mp4", "webm"})
        if (column == name) return modality_t::video_k;
    for (char const *name : {"audio", "wav", "flac", "mp3"})
        if (column == name) return modality_t::audio_k;
    return modality_t::text_k;
}

/** Copy the string @p value into @p into , or fail naming @p key . */
[[nodiscard]] static expected_t read_string_(yyjson_val *value, char const *key, sz::string_t &into) noexcept {
    if (!yyjson_is_str(value)) return expected_t::fail(status_t::invalid_input_k, "\"{}\" is not a string", key);
    if (sz::failed(into.assign(sz::string_view_t {yyjson_get_str(value), yyjson_get_len(value)})))
        return expected_t::fail(status_t::bad_alloc_k, "\"{}\" staging failed", key);
    return {};
}

/** Read a @c query key into @p sides : the named role queries; the bag's others are documents. */
[[nodiscard]] static expected_t read_query_sides_(yyjson_val *value, role_sides_t &sides) noexcept {
    if (!yyjson_is_str(value)) return expected_t::fail(status_t::invalid_input_k, "\"query\" is not a string");
    sz::string_view_t const query {yyjson_get_str(value), yyjson_get_len(value)};
    using side_t = late_interaction_side_t;
    if (query == "anchor") sides = {side_t::query_k, side_t::document_k, side_t::document_k};
    else if (query == "positives") sides = {side_t::document_k, side_t::query_k, side_t::document_k};
    else if (query == "both") sides = {side_t::query_and_document_k, side_t::query_and_document_k, side_t::document_k};
    else return expected_t::fail(status_t::invalid_input_k, "\"query\" is {}, not anchor, positives or both", query);
    return {};
}

/** Read a @c choices object into @p entry 's state, choices, targets and group columns. */
[[nodiscard]] static expected_t read_choices_(yyjson_val *object, mixture_entry_t &entry) noexcept {
    if (!yyjson_is_obj(object)) return expected_t::fail(status_t::invalid_input_k, "\"choices\" is not an object");
    std::size_t index = 0, count = 0;
    yyjson_val *key = nullptr, *value = nullptr;
    yyjson_obj_foreach(object, index, count, key, value) {
        sz::string_view_t const name {yyjson_get_str(key), yyjson_get_len(key)};
        expected_t read;
        if (name == "state") read = read_string_(value, "state", entry.state);
        else if (name == "choices") read = read_string_(value, "choices", entry.choices);
        else if (name == "targets") read = read_string_(value, "targets", entry.targets);
        else if (name == "group") read = read_string_(value, "group", entry.group);
        else return expected_t::fail(status_t::invalid_input_k, "unknown choices key \"{}\"", name);
        if (!read) return read;
    }
    if (entry.state.empty() || entry.choices.empty() || entry.targets.empty())
        return expected_t::fail(status_t::invalid_input_k, "\"choices\" needs \"state\", \"choices\" and \"targets\"");
    return {};
}

/** Parse one entry of the @c datasets array into @p entry . */
[[nodiscard]] static expected_t read_entry_(yyjson_val *object, mixture_entry_t &entry) noexcept {
    if (!yyjson_is_obj(object)) return expected_t::fail(status_t::invalid_input_k, "a dataset is not an object");
    std::size_t index = 0, count = 0;
    yyjson_val *key = nullptr, *value = nullptr;
    yyjson_obj_foreach(object, index, count, key, value) {
        sz::string_view_t const name {yyjson_get_str(key), yyjson_get_len(key)};
        expected_t read;
        if (name == "name") read = read_string_(value, "name", entry.name);
        else if (name == "glob") read = read_string_(value, "glob", entry.glob);
        else if (name == "text") read = read_string_(value, "text", entry.text);
        else if (name == "anchor") read = read_string_(value, "anchor", entry.anchor);
        else if (name == "positive") read = read_string_(value, "positive", entry.positive);
        else if (name == "media") read = read_string_(value, "media", entry.media);
        else if (name == "negative") read = read_string_(value, "negative", entry.negative);
        else if (name == "anchor_prefix") read = read_string_(value, "anchor_prefix", entry.anchor_prefix);
        else if (name == "positive_prefix") read = read_string_(value, "positive_prefix", entry.positive_prefix);
        else if (name == "query") read = read_query_sides_(value, entry.sides);
        else if (name == "choices") read = read_choices_(value, entry);
        else if (name == "weight") {
            if (!yyjson_is_num(value)) return expected_t::fail(status_t::invalid_input_k, "\"weight\" is not a number");
            entry.weight = yyjson_get_num(value);
        }
        else if (name == "epochs") {
            if (!yyjson_is_uint(value) || yyjson_get_uint(value) == 0)
                return expected_t::fail(status_t::invalid_input_k, "\"epochs\" is not a positive integer");
            entry.epochs = static_cast<std::size_t>(yyjson_get_uint(value));
        }
        // Read by the acquisition and mining scripts, not by the loaders.
        else if (name != "positives_per_caption" && name != "min_epochs")
            return expected_t::fail(status_t::invalid_input_k, "unknown key \"{}\"", name);
        if (!read) return read;
    }
    if (entry.name.empty() || entry.glob.empty() || !(entry.weight > 0.0))
        return expected_t::fail(status_t::invalid_input_k, "a dataset needs a name, a glob and a positive weight");
    if (!entry.text.empty() + !entry.anchor.empty() + !entry.state.empty() != 1)
        return expected_t::fail(status_t::invalid_input_k, "{} needs one of \"text\", \"anchor\" or \"choices\"",
                                entry.name.view());
    if (!entry.anchor.empty() && entry.positive.empty())
        return expected_t::fail(status_t::invalid_input_k, "{} pairs its anchor with no \"positive\"",
                                entry.name.view());
    if (entry.media.empty()) return {};
    if (entry.media != entry.positive)
        return expected_t::fail(status_t::invalid_input_k, "{} names media {} other than its positive {}",
                                entry.name.view(), entry.media.view(), entry.positive.view());
    entry.modality = modality_of_(entry.media.view());
    if (entry.modality == modality_t::text_k)
        return expected_t::fail(status_t::invalid_input_k, "{} names media {}, which tells no modality",
                                entry.name.view(), entry.media.view());
    return {};
}

expected<mixture_t> mixture_t::read(sz::string_view_t path) noexcept {
    using result_t = expected<mixture_t>;
    if (path.contains('\0')) return result_t::fail(status_t::invalid_input_k, "the mixture path holds a NUL byte");
    sz::string_t terminated;
    if (sz::failed(terminated.assign(path))) return result_t::fail(status_t::bad_alloc_k, "path staging failed");
    yyjson_read_err error {};
    yyjson_doc *const document = yyjson_read_file(terminated.c_str(), 0, nullptr, &error);
    if (document == nullptr)
        return result_t::fail(
            error.code == YYJSON_READ_ERROR_FILE_OPEN ? status_t::io_error_k : status_t::invalid_input_k,
            "{}: {} at byte {}", path, error.msg, error.pos);
    mixture_t mixture;
    expected_t parsed = [&]() noexcept -> expected_t {
        yyjson_val *const root = yyjson_doc_get_root(document);
        yyjson_val *const name = yyjson_obj_get(root, "mixture");
        yyjson_val *const datasets = yyjson_obj_get(root, "datasets");
        if (!yyjson_is_arr(datasets) || yyjson_arr_size(datasets) == 0)
            return expected_t::fail(status_t::invalid_input_k, "no \"datasets\" array");
        if (expected_t read = read_string_(name, "mixture", mixture.name); !read) return read;
        if (yyjson_obj_size(root) != 2)
            return expected_t::fail(status_t::invalid_input_k, "keys beside \"mixture\" and \"datasets\"");
        std::size_t index = 0, count = 0;
        yyjson_val *object = nullptr;
        yyjson_arr_foreach(datasets, index, count, object) {
            mixture_entry_t entry;
            if (expected_t read = read_entry_(object, entry); !read) return read;
            if (st::failed(mixture.datasets.push_back(std::move(entry))))
                return expected_t::fail(status_t::bad_alloc_k, "dataset table allocation failed");
        }
        u32_t pairs = 0;
        for (mixture_entry_t const &entry : mixture.datasets) pairs += !entry.anchor.empty();
        if (pairs != 0 && pairs != mixture.datasets.size())
            return expected_t::fail(status_t::invalid_input_k, "pairs share the mixture with plain text or choices");
        return {};
    }();
    yyjson_doc_free(document);
    if (!parsed) return result_t::fail(status_t::invalid_input_k, "{}: {}", path, parsed.description_.view());
    return result_t {std::move(mixture)};
}

bool mixture_t::names(modality_t modality) const noexcept {
    for (mixture_entry_t const &entry : datasets)
        if (!entry.media.empty() && entry.modality == modality) return true;
    return false;
}

bool mixture_t::chooses() const noexcept {
    for (mixture_entry_t const &entry : datasets)
        if (!entry.state.empty()) return true;
    return false;
}

multimodal_batch_shape_t mixture_t::shape(multimodal_batch_shape_t shape) const noexcept {
    // The paired loader's bag holds an anchor and a positive, and a negative when the entry names
    // one.
    std::span<mixture_entry_t const> const entries = datasets;
    if (!entries.empty() && !entries[0].anchor.empty())
        shape.max_bag_segments = std::any_of(entries.begin(), entries.end(),
                                             [](mixture_entry_t const &entry) { return !entry.negative.empty(); })
                                     ? 3u
                                     : 2u;
    if (names(modality_t::image_k)) shape.image = image_batch_shape_t {16u};
    if (names(modality_t::video_k)) shape.video = video_batch_shape_t {16u, 3u, 2u, 1u};
    if (names(modality_t::audio_k)) shape.audio = audio_batch_shape_t {320u, 16000u};
    // A choice offers distinct bytes, so a segment holds at most every byte value once.
    if (chooses()) shape.max_choices_per_segment = byte_pad_k;
    return shape;
}

expected<mixture_sampler_t> mixture_sampler_t::make(std::span<mixture_entry_t const> entries,
                                                    std::span<std::size_t const> rows, std::size_t world_gpus,
                                                    std::size_t batches_per_fill, std::size_t segments) noexcept {
    using result_t = expected<mixture_sampler_t>;
    mixture_sampler_t sampler;
    sampler.entries_ = entries;
    sampler.world_gpus_ = world_gpus, sampler.batches_per_fill_ = batches_per_fill;
    if (st::failed(sampler.weight_prefix_.push_back(0.0)))
        return result_t::fail(status_t::bad_alloc_k, "weight table allocation failed");
    for (std::size_t entry = 0; entry != entries.size(); ++entry) {
        mixture_entry_t const &dataset = entries[entry];
        if (dataset.epochs != 0 && rows[entry] == 0)
            return result_t::fail(status_t::invalid_input_k, "{} caps its epochs, but only Parquet footers count rows",
                                  dataset.name.view());
        // A fill draws as many pairs as its segments hold, or documents for plain text, whether or
        // not its batches admit them all.
        std::size_t const roles = dataset.anchor.empty() ? 1u : dataset.negative.empty() ? 2u : 3u;
        if (st::failed(sampler.rows_.push_back(std::size_t {rows[entry]})) ||
            st::failed(sampler.fill_pairs_.push_back(batches_per_fill * (segments / roles))) ||
            st::failed(sampler.drawn_pairs_.push_back(std::size_t {0})) ||
            st::failed(sampler.weight_prefix_.push_back(sampler.weight_prefix_[entry] + dataset.weight)))
            return result_t::fail(status_t::bad_alloc_k, "sampler table allocation failed");
    }
    return result_t {std::move(sampler)};
}

std::size_t mixture_sampler_t::draw(u64_t seed, data_epoch_t data_epoch, u32_t gpu_index,
                                    void (*report)(sz::string_view_t) noexcept) noexcept {
    auto const capped_live = [&]() noexcept {
        for (std::size_t entry = 0; entry != entries_.size(); ++entry)
            if (entries_[entry].epochs != 0 && live_(entry)) return true;
        return false;
    };
    // A retirement changes every later draw, so the replay walks the epochs in order, and stops
    // once no capped entry is left to retire.
    for (; replayed_epochs_ < data_epoch.ordinal && capped_live(); ++replayed_epochs_) {
        bool retired = false;
        for (std::size_t gpu = 0; gpu != world_gpus_; ++gpu) {
            std::size_t const phase = gpu * batches_per_fill_ / world_gpus_;
            if (replayed_epochs_ != 0 && (replayed_epochs_ + phase) % batches_per_fill_ != 0) continue;
            std::size_t const entry = search_(seed, replayed_epochs_, static_cast<u32_t>(gpu));
            bool const was_live = live_(entry);
            drawn_pairs_[entry] += fill_pairs_[entry];
            if (!was_live || live_(entry)) continue;
            retired = true;
            sz::string_t line;
            if (report && format_into(line, "uform: {} retired at step {} after {} pairs ({} passes over {} rows)",
                                      entries_[entry].name.view(), replayed_epochs_, drawn_pairs_[entry],
                                      entries_[entry].epochs, rows_[entry]))
                report(line.view());
        }
        if (retired) reweigh_(replayed_epochs_, report);
    }
    return search_(seed, data_epoch.ordinal, gpu_index);
}

bool mixture_sampler_t::live_(std::size_t entry) const noexcept {
    std::size_t const epochs = entries_[entry].epochs;
    return epochs == 0 || drawn_pairs_[entry] < epochs * rows_[entry];
}

std::size_t mixture_sampler_t::search_(u64_t seed, u32_t epoch, u32_t gpu_index) const noexcept {
    constexpr splitmix64_hash_t hash {};
    f64_t const draw = static_cast<f64_t>(hash(seed, epoch, gpu_index) >> 11) * 0x1.0p-53 *
                       weight_prefix_[weight_prefix_.size() - 1];
    return std::min<std::size_t>(
        std::upper_bound(weight_prefix_.begin(), weight_prefix_.end(), draw) - weight_prefix_.begin() - 1,
        entries_.size() - 1);
}

void mixture_sampler_t::reweigh_(u32_t epoch, void (*report)(sz::string_view_t) noexcept) noexcept {
    std::size_t const entries = entries_.size();
    // A modality with no live entry adds nothing; the draw's scale spreads its share over the rest.
    for (std::size_t entry = 0; entry != entries; ++entry) {
        f64_t modality_weight = 0.0, live_weight = 0.0;
        for (std::size_t peer = 0; peer != entries; ++peer)
            if (entries_[peer].modality == entries_[entry].modality)
                modality_weight += entries_[peer].weight, live_weight += live_(peer) ? entries_[peer].weight : 0.0;
        f64_t const weight = live_(entry) ? entries_[entry].weight * modality_weight / live_weight : 0.0;
        weight_prefix_[entry + 1] = weight_prefix_[entry] + weight;
    }
    if (weight_prefix_[entries] > 0.0) return;
    for (std::size_t entry = 0; entry != entries; ++entry)
        weight_prefix_[entry + 1] = weight_prefix_[entry] + entries_[entry].weight;
    sz::string_t line;
    if (report && format_into(line, "uform: every entry retired at step {}, drawing by the mixture's weights", epoch))
        report(line.view());
}

expected_t dataset_source_t::open(dataset_source_t &placed, trainer_config_t &config,
                                  gpu_direct_storage_session_t &session, mixture_t mixture,
                                  std::size_t batches_per_fill) noexcept {
    multimodal_batch_shape_t const &shape = config.batch;
    placed.mixture_ = std::move(mixture);
    std::span<mixture_entry_t const> const entries = placed.mixture_.datasets;
    bool const pairs = !entries.empty() && !entries[0].anchor.empty();

    if (pairs && fu::failed(placed.pool_.spawn(fu::allowed_cores_count())))
        return expected_t::fail(status_t::bad_alloc_k, "the decode pool did not spawn");

    auto adopt = [&](auto loader) noexcept -> expected_t {
        if (!loader) return expected_t::fail(std::move(loader));
        if (st::failed(placed.loaders_.push_back(std::move(loader.value_))))
            return expected_t::fail(status_t::bad_alloc_k, "loader table allocation failed");
        return {};
    };
    for (mixture_entry_t const &entry : entries) {
        sz::string_view_t const glob = entry.glob.view();
        expected_t made;
        if (!entry.state.empty() && glob.ends_with(".parquet"))
            made = adopt(choices_loader_parquet_t::make(
                session, glob, config.seed, {entry.state.view(), entry.choices.view(), entry.targets.view()}));
        else if (!entry.state.empty())
            made = expected_t::fail(status_t::invalid_input_k, "{} holds no Parquet shards of choices", glob);
        else if (!pairs && glob.ends_with(".parquet"))
            made = adopt(tokens_loader_parquet_t::make(session, glob, config.seed));
        else if (!pairs && glob.ends_with(".jsonl"))
            made = adopt(tokens_loader_jsonl_t::make(session, glob, config.seed));
        else if (!pairs && (glob.ends_with(".csv") || glob.ends_with(".txt")))
            made = adopt(tokens_loader_csv_t::make(session, glob, config.seed));
        else if (!pairs)
            made = expected_t::fail(status_t::invalid_input_k, "{} holds no Parquet, JSONL, CSV or text shards", glob);
        else if (entry.modality == modality_t::video_k && config.nvdec_engines == 0)
            made = expected_t::fail(status_t::invalid_input_k,
                                    "{} holds video: --nvdec-engines names the GPU's NVDEC engine count",
                                    entry.name.view());
        else
            made = adopt(
                paired_loader_parquet_t::make(session, glob, config.seed, placed.pool_,
                                              {entry.anchor.view(), entry.positive.view(), entry.negative.view(),
                                               entry.anchor_prefix.view(), entry.positive_prefix.view(), entry.sides},
                                              entry.modality, shape));
        if (!made) return expected_t::fail(made.status_, "{}: {}", entry.name.view(), made.description_.view());
    }
    reservoir_bound_t reservoir;
    f64_t row_bytes = 0.0;
    for (std::size_t index = 0; index != placed.loaders_.size(); ++index) {
        loader_t const &loader = placed.loaders_[index];
        std::visit([&](auto const &typed) noexcept { reservoir.charge(typed.reservoir()); }, loader);
        // An entry's widest row group bounds its items' bytes, a byte of text is a cell, and the
        // densest entry sizes the segment budget: a text entry packs a row with documents, a text
        // pair with its roles' texts, and a media pair as many to a row as `media_cells` places.
        std::size_t const item_cells = std::visit(
            [](auto const &typed) noexcept { return typed.reservoir().row_bytes / reservoir_bound_t::copies_k; },
            loader);
        f64_t const columns = static_cast<f64_t>(shape.max_columns_per_row);
        f64_t density = 1.0, roles = 1.0;
        if (auto const *paired = std::get_if<paired_loader_parquet_t>(&loader)) {
            roles = entries[index].negative.empty() ? 2.0 : 3.0;
            density = roles * paired->pairs_per_row(shape.max_columns_per_row);
        }
        else if ((std::holds_alternative<tokens_loader_parquet_t>(loader) ||
                  std::holds_alternative<choices_loader_parquet_t>(loader)) &&
                 item_cells != 0)
            density = columns / static_cast<f64_t>(item_cells);
        config.segments_per_row = std::max(config.segments_per_row, density);
        // A fill reads one entry, so a batch row costs that entry's own source rows, a pair or a
        // document each.
        f64_t const entry_row_bytes = static_cast<f64_t>(
            std::visit([](auto const &typed) noexcept { return typed.reservoir().row_bytes; }, loader));
        row_bytes = std::max(row_bytes, entry_row_bytes * density / roles);
    }
    // One fill pools the rows of every batch it packs, so the per-row term scales with them.
    config.reservoir_fixed_bytes = reservoir.fixed_bytes;
    config.reservoir_row_bytes = static_cast<std::size_t>(std::ceil(row_bytes * static_cast<f64_t>(batches_per_fill)));
    return {};
}

expected_t dataset_source_t::make(dataset_source_t &placed, trainer_t &trainer, std::size_t batches_per_fill) noexcept {
    trainer_config_t const &config = trainer.config;
    multimodal_batch_shape_t shape = config.batch;
    std::size_t const gpus = trainer.replicas.size(), segments = shape.max_segments_per_gpu;
    std::span<mixture_entry_t const> const entries = placed.mixture_.datasets;
    // A pair spends two segments and a triplet three, each in one bag, so pairs bound the bags.
    if (!entries.empty() && !entries[0].anchor.empty())
        shape.max_bags_per_gpu = static_cast<u32_t>(shape.max_segments_per_gpu / 2u);

    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {trainer.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        st::basic_vector<multimodal_batch_on_gpu_t> filled;
        for (std::size_t batch_index = 0; batch_index != batches_per_fill; ++batch_index) {
            expected<multimodal_batch_on_gpu_t> batch = multimodal_batch_on_gpu_t::make(
                gpu_arena_alloc_t {trainer.arenas[member]}, shape);
            if (!batch) return expected_t::fail(std::move(batch));
            if (st::failed(filled.push_back(std::move(batch.value_))))
                return expected_t::fail(status_t::bad_alloc_k, "batch table allocation failed");
        }
        if (placed.filled_.push_back(std::move(filled)) != status_t::success_k ||
            placed.loaded_.push_back(multimodal_batch_on_gpu_t {}) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "batch table full");
        if (placed.mixture_.names(modality_t::image_k)) {
            expected<nvjpeg_session_t> jpeg = nvjpeg_session_t::make();
            if (!jpeg) return expected_t::fail(std::move(jpeg));
            if (placed.jpeg_.push_back(std::move(jpeg.value_)) != status_t::success_k)
                return expected_t::fail(status_t::bad_alloc_k, "nvJPEG session table full");
        }
        if (placed.mixture_.names(modality_t::video_k)) {
            expected<nvdec_session_t> nvdec = nvdec_session_t::make(
                nvdec_config_t {config.nvdec_engines, shape.video.patch_side * shape.max_columns_per_row});
            if (!nvdec) return expected_t::fail(std::move(nvdec));
            if (placed.nvdec_.push_back(std::move(nvdec.value_)) != status_t::success_k)
                return expected_t::fail(status_t::bad_alloc_k, "NVDEC session table full");
        }
    }
    expected<multimodal_batch_mixer_t> mixer = multimodal_batch_mixer_t::make(
        trainer.domain, trainer.mesh->colocated, trainer.arenas, shape, config.seed, trainer.streams,
        trainer.mesh->timeout_seconds);
    if (!mixer) return expected_t::fail(std::move(mixer));
    placed.mixer_ = std::move(mixer.value_);
    std::size_t const world_gpus = trainer.mesh->topology.shape.gpus_per_colocated_domain;
    for (std::size_t member = 0; member != gpus; ++member) {
        std::size_t const world_gpu = trainer.mesh->topology.first_gpu_index() + member;
        if (placed.phases_.push_back(world_gpu * batches_per_fill / world_gpus) != status_t::success_k)
            return expected_t::fail(status_t::bad_alloc_k, "fill phase table full");
    }
    st::basic_vector<std::size_t> rows;
    for (loader_t const &loader : placed.loaders_) {
        std::size_t counted = 0;
        if (auto const *paired = std::get_if<paired_loader_parquet_t>(&loader)) counted = paired->rows();
        else if (auto const *parquet = std::get_if<tokens_loader_parquet_t>(&loader)) counted = parquet->rows();
        else if (auto const *choices = std::get_if<choices_loader_parquet_t>(&loader)) counted = choices->rows();
        if (st::failed(rows.push_back(std::size_t {counted})))
            return expected_t::fail(status_t::bad_alloc_k, "row count table allocation failed");
    }
    expected<mixture_sampler_t> sampler = mixture_sampler_t::make(entries, rows, world_gpus, batches_per_fill,
                                                                  segments);
    if (!sampler) return expected_t::fail(std::move(sampler));
    placed.sampler_ = std::move(sampler.value_);

    if (!config.contrasts()) return {};
    u32_t const edges = config.contrastive.max_edges_per_gpu
                            ? config.contrastive.max_edges_per_gpu
                            : relation_edges_capacity(static_cast<u32_t>(segments), shape.max_bag_segments);
    placed.bags_ = u32r2_tensor_pinned_t::uninitialized({gpus, segments}, pinned_alloc<u32_t> {}).value;
    placed.lengths_ = u32r2_tensor_pinned_t::uninitialized({gpus, segments}, pinned_alloc<u32_t> {}).value;
    placed.roles_ = nk::tensor<segment_role_t, pinned_alloc<segment_role_t>, 2>::uninitialized(
                        {gpus, segments}, pinned_alloc<segment_role_t> {})
                        .value;
    if (!placed.bags_ || !placed.lengths_ || !placed.roles_)
        return expected_t::fail(status_t::bad_alloc_k, "relation scratch allocation failed");
    expected<contrastive_edges_pinned_t> edge_sets = contrastive_edges_pinned_t::make(static_cast<u32_t>(segments),
                                                                                      edges);
    if (!edge_sets) return expected_t::fail(std::move(edge_sets));
    placed.edges_ = std::move(edge_sets.value_);
    return {};
}

expected_t dataset_source_t::fill(trainer_t &trainer, data_epoch_t data_epoch, step_report_t &) noexcept {
    std::size_t const gpus = trainer.replicas.size();
    std::size_t const batches_per_fill = filled_[0].size();
    // Each GPU takes its batches in its own phase of the period and refills where that wraps, so a
    // step pays for a share of the reads and the world's datasets rotate a few at a time.
    auto const batch_of = [&](std::size_t member) noexcept {
        return (served_steps_ + phases_[member]) % batches_per_fill;
    };
    for (std::size_t member = 0; member != gpus; ++member)
        if (batch_of(member) == 0 || served_steps_ == 0)
            if (expected_t filled = fill_member_(trainer, member, data_epoch); !filled) return filled;
    // The mixer reads one batch per GPU side by side, so this step's batches swap in and back out.
    for (std::size_t member = 0; member != gpus; ++member)
        std::swap(loaded_[member], filled_[member][batch_of(member)]);
    expected_t mixed;
    for (std::size_t member = 0; member != gpus && mixed; ++member)
        mixed = mixer_(member, data_epoch, loaded_.view(), trainer.replicas[member].batch, trainer.streams[member]);
    for (std::size_t member = 0; member != gpus; ++member)
        std::swap(loaded_[member], filled_[member][batch_of(member)]);
    ++served_steps_;
    if (!mixed) return mixed;
    if (!trainer.config.contrasts()) return {};
    return relate_(trainer);
}

expected_t dataset_source_t::fill_member_(trainer_t &trainer, std::size_t member, data_epoch_t data_epoch) noexcept {
    trainer_config_t const &config = trainer.config;
    u32_t const gpu_index = trainer.mesh->topology.first_gpu_index() + static_cast<u32_t>(member);
    std::size_t const entry = sampler_.draw(config.seed, data_epoch, gpu_index,
                                            trainer.mesh->topology.is_lead() ? &report_line : nullptr);
    mixture_entry_t const &dataset = mixture_.datasets[entry];
    std::span<multimodal_batch_on_gpu_t> const batches = filled_[member];
    gpu_stream_t &stream = trainer.streams[member];
    u32_t const step = data_epoch.ordinal;
    expected_t filled;
    for (multimodal_batch_on_gpu_t &batch : batches)
        if (filled) filled = zero_async(batch.dropped.span(), stream);
    if (!filled) return filled;
    if (auto *parquet = std::get_if<tokens_loader_parquet_t>(&loaders_[entry]))
        filled = parquet->fill(stream, {dataset.text.view()}, gpu_index, step, batches);
    else if (auto *jsonl = std::get_if<tokens_loader_jsonl_t>(&loaders_[entry]))
        filled = jsonl->fill(stream, {dataset.text.view()}, gpu_index, step, batches);
    else if (auto *csv = std::get_if<tokens_loader_csv_t>(&loaders_[entry])) {
        // A `.txt` shard is one document per line: a headerless table of one column, as no line
        // holds a tab.
        tokens_base_loader_csv_config_t text {dataset.text.view()};
        if (dataset.glob.view().ends_with(".txt")) text.delimiter = '\t', text.header = csv_header_t::positional_k;
        filled = csv->fill(stream, text, gpu_index, step, batches);
    }
    else if (auto *choices = std::get_if<choices_loader_parquet_t>(&loaders_[entry]))
        filled = choices->fill(stream, gpu_index, step, batches);
    else {
        paired_loader_parquet_t &paired = std::get<paired_loader_parquet_t>(loaders_[entry]);
        if (paired.modality() == modality_t::image_k)
            filled = paired.fill(stream, jpeg_[member], gpu_index, step, batches);
        else if (paired.modality() == modality_t::video_k)
            filled = paired.fill(stream, nvdec_[member], gpu_index, step, batches);
        else filled = paired.fill(stream, gpu_index, step, batches);
    }
    if (!filled)
        return expected_t::fail(filled.status_, "{} on GPU {}: {}", dataset.name.view(), gpu_index,
                                filled.description_.view());
    // Every other loader names no choice, and a batch reused from a choices fill keeps its rows.
    if (dataset.state.empty())
        for (multimodal_batch_on_gpu_t &batch : batches)
            if (expected_t padded = pad_choices(batch, stream); !padded) return padded;
    return {};
}

expected_t dataset_source_t::relate_(trainer_t &trainer) noexcept {
    trainer_config_t const &config = trainer.config;
    std::size_t const gpus = trainer.replicas.size();
    u32_t const segments = config.batch.max_segments_per_gpu;
    std::size_t const world_segments = std::size_t {segments} * trainer.mesh->topology.shape.gpus_per_colocated_domain;
    u32_t const first = trainer.mesh->topology.first_gpu_index();
    bool const pairs = !mixture_.datasets[0].anchor.empty();
    if (!pairs) {
        for (std::size_t member = 0; member != gpus; ++member) {
            switch_to_gpu_t gpu_scope {trainer.streams[member]};
            if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
            base_replica_t &replica = trainer.replicas[member];
            if (expected_t related = replica.relations.upload_self_edges(
                    segments, replica.batch.segment_lengths.view(), mixer_.world_lengths(member).flatten<1>(),
                    std::size_t {first + member} * segments, config.contrastive.ambient_repulsion_weight,
                    trainer.streams[member]);
                !related)
                return related;
            trainer.contrastive.rebind_relations(member, replica.relations.view());
        }
        return {};
    }

    for (std::size_t member = 0; member != gpus; ++member) {
        switch_to_gpu_t gpu_scope {trainer.streams[member]};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        multimodal_batch_on_gpu_t const &mixed = trainer.replicas[member].batch;
        expected_t copied = copy_device_to_host(mixed.segment_bag.view(), bags_[member, nk::slice],
                                                trainer.streams[member]);
        if (copied)
            copied = copy_device_to_host(mixed.segment_lengths.view(), lengths_[member, nk::slice],
                                         trainer.streams[member]);
        if (copied)
            copied = copy_device_to_host(mixed.segment_role.view(), roles_[member, nk::slice], trainer.streams[member]);
        if (!copied) return copied;
    }
    if (expected_t drained = trainer.synchronize(); !drained) return drained;

    for (std::size_t member = 0; member != gpus; ++member) {
        u32_t const vertex_base = (first + static_cast<u32_t>(member)) * segments;
        if (expected_t related = relate_roles(bags_[member, nk::slice], roles_[member, nk::slice],
                                              lengths_[member, nk::slice], f32r1_view_t {}, vertex_base, edges_);
            !related)
            return related;

        base_replica_t &replica = trainer.replicas[member];
        gpu_stream_t &stream = trainer.streams[member];
        switch_to_gpu_t gpu_scope {stream};
        if (!gpu_scope.ok()) return expected_t::fail(status_t::cuda_error_k, "context entry failed");
        expected_t staged = replica.relations.upload(
            edges_.attract(), edges_.repel(), edges_.ignore(), replica.batch.segment_lengths.view(),
            mixer_.world_lengths(member).flatten<1>(), replica.batch.segment_key.view(),
            mixer_.world_keys(member).flatten<1>(), world_segments, vertex_base,
            config.contrastive.ambient_repulsion_weight, stream);
        if (staged) // the host columns are reused for the next GPU, so their copies land first
            if (CUresult result = cuStreamSynchronize(stream.get()); result != CUDA_SUCCESS)
                staged = cuda_failure(result, "relation staging drain failed (member={})", member);
        if (!staged) return staged;
        trainer.contrastive.rebind_relations(member, replica.relations.view());
    }
    return {};
}
