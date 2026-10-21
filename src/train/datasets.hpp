/**
 *  @file src/train/datasets.hpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief Real data for a run: @ref mixture_t , the weighted datasets a mixture JSON in
 *      @c scripts/mixtures names, and @ref dataset_source_t , the batch source that loads them
 *      through the loaders and the mixer.
 *
 *  Linked into the @c uform_base binary alone, since the C library links neither the loaders nor
 *  cuDF.
 *
 *  @section datasets_mixture A mixture is all text or all pairs
 *
 *  A text entry names its document column in @c text ; its shards are Parquet, JSONL, CSV with a
 *  header row, or @c .txt files of one document per line, whose only column is @c 0 . A paired
 *  entry names its anchor text column in @c anchor and its positive in @c positive : another text,
 *  or media also named in @c media , the column's name telling the modality. A paired entry may
 *  name hard negatives in @c negative , and text prefixes in @c anchor_prefix and
 *  @c positive_prefix . Each GPU draws one entry by weight per fill and packs its next
 *  @c batches_per_fill batches from it, refilling on its own phase of that period, so the mixer's
 *  world-wide shuffle sees datasets rotate a few GPUs at a time. Pairs route as bags, so one
 *  mixture never mixes them with plain text, whose segments route alone.
 */
#pragma once

#include <variant> // `std::variant` — one loader per entry, of the entry's kind

#include "loaders/direct_storage.hpp" // `gpu_direct_storage_session_t`
#include "loaders/tokens_loader.cuh"  // the Parquet, JSONL and CSV tokens loaders
#include "loaders/paired_loader.cuh"  // `paired_loader_parquet_t`, `nvjpeg_session_t`, `nvdec_session_t`
#include "train/trainer.hpp"          // `batch_source_t`, `trainer_t`

namespace unum::uform::train {

/** One dataset of a mixture, as its JSON entry names it. */
struct mixture_entry_t {

    /** The dataset's name. */
    sz::string_t name {};

    /** The shard paths or glob; a relative one resolves against the working directory, the
     *  repository root. */
    sz::string_t glob {};

    /** The share of the draws this entry takes, relative to the other entries. */
    f64_t weight = 0.0;

    /** A text entry's document column; empty for a pair. */
    sz::string_t text {};

    /** A pair's caption column, its positive column and the media column among them. */
    sz::string_t anchor {}, positive {}, media {};

    /** A pair's hard-negative column. */
    sz::string_t negative {};

    /** The bytes every anchor starts with, and every positive and negative text. */
    sz::string_t anchor_prefix {}, positive_prefix {};

    /** The late-interaction side of each role, from the @c query key: @c "anchor" , @c "positives"
     *  or @c "both" queries and the bag's other roles are documents; none without the key. */
    loaders::role_sides_t sides {};

    /** The modality of @ref media , told by its name; @c text_k for a text entry or a text pair. */
    modality_t modality = modality_t::text_k;
};

/** A weighted list of datasets, one mixture JSON in @c scripts/mixtures . */
struct mixture_t {

    /** The mixture's name. */
    sz::string_t name {};

    /** The datasets in file order. */
    array<mixture_entry_t> datasets {};

    /**
     *  @brief Parse the mixture JSON at @p path .
     *  @retval status_t::io_error_k The file does not open.
     *  @retval status_t::invalid_input_k Malformed JSON, an unknown key, a missing field, or a
     *      media column whose name tells no modality.
     */
    [[nodiscard]] static expected<mixture_t> read(sz::string_view_t path) noexcept;

    /** Whether any entry's media are of @p modality . */
    [[nodiscard]] bool names(modality_t modality) const noexcept;

    /** @p shape with the slice of every media modality the entries name lit at the mixture
     *  geometry: RGBA images in 16-pixel patches, RGB video in 2-frame tubes of 16 pixels, and
     *  256-sample audio frames at 16 kHz. */
    [[nodiscard]] multimodal_batch_shape_t shape(multimodal_batch_shape_t shape) const noexcept;
};

/**
 *  @brief The datasets of a mixture, loaded per GPU and routed across the mesh by the mixer, with
 *      the relations of every pair under a contrastive objective.
 *
 *  Owns the decode pool, one loader per entry, the nvJPEG and NVDEC sessions of every GPU its media
 *  need, the loaded batches, the mixer and the host scratch the relations are built in; borrows the
 *  process's storage session from the composition root. Placed and never moved, since the loaders
 *  borrow the pool.
 */
struct dataset_source_t final : batch_source_t {

    /** Make one loader per entry of @p mixture over @p storage , before the trainer, and charge
     *  @p config with the device bytes their reservoirs can take outside the arena. */
    [[nodiscard]] static expected_t open(dataset_source_t &placed, trainer_config_t &config,
                                         loaders::gpu_direct_storage_session_t &storage, mixture_t mixture) noexcept;

    /** Make the loaded batches, the decoder sessions and the mixer of an @ref open source from
     *  @p trainer 's arenas. */
    [[nodiscard]] static expected_t make(dataset_source_t &placed, trainer_t &trainer) noexcept;

    /** Take every GPU's batch of this step, refilling a GPU whose period starts here; the mixer
     *  then routes them into the replicas. */
    [[nodiscard]] expected_t fill(trainer_t &trainer, data_epoch_t data_epoch, step_report_t &report) noexcept override;

  private:

    /** GPU @p member draws one entry by weight, keyed by the run seed, @p data_epoch and its world
     *  coordinate, and packs its @c batches_per_fill batches from one read of it. */
    [[nodiscard]] expected_t fill_member_(trainer_t &trainer, std::size_t member, data_epoch_t data_epoch) noexcept;

    /** Rebuild every replica's relations over the world's vertices, PAD read off the mixer's world
     *  lengths: the segments of a bag attract each other, and text alone keeps self-edges. */
    [[nodiscard]] expected_t relate_(trainer_t &trainer) noexcept;

    using loader_t = std::variant<loaders::tokens_loader_parquet_t, loaders::tokens_loader_jsonl_t,
                                  loaders::tokens_loader_csv_t, loaders::paired_loader_parquet_t>;

    /** The entries the loaders and their columns come from. */
    mixture_t mixture_ {};

    /** The host workers the image fallback and the audio and video demux fan across. */
    loaders::cpu_decode_pool_t pool_ {};

    /** One loader per entry, in entry order. */
    array<loader_t> loaders_ {};

    /** @b [entries+1] cumulative weights, the entry draw's search table. */
    array<f64_t> weight_prefix_ {};

    /** Every GPU's decoder sessions, present when an entry's media need them. */
    per_gpu_array<loaders::nvjpeg_session_t> jpeg_ {};
    per_gpu_array<loaders::nvdec_session_t> nvdec_ {};

    /** Every GPU's @c batches_per_fill filled batches, which one fill packs from one shard read. */
    per_gpu_array<array<multimodal_batch_on_gpu_t>> filled_ {};

    /** Every GPU's batch of this step, swapped in from @ref filled_ for the mixer and out after. */
    per_gpu_array<multimodal_batch_on_gpu_t> loaded_ {};

    /** Every GPU's phase in the fill period, its world coordinate's share of it, so the GPUs refill
     *  on different steps. */
    per_gpu_array<std::size_t> phases_ {};

    /** The steps this source served since @ref make , counting the period even for a held-out
     *  source called at sparse epochs; the first refills every GPU. */
    std::size_t served_steps_ = 0;
    multimodal_batch_mixer_t mixer_ {};

    /** @b [gpus,segments] every mixed segment's bag, length and role across this process's GPUs,
     *  landed for the relations. */
    u32r2_tensor_pinned_t bags_ {}, lengths_ {};
    nk::tensor<segment_role_t, pinned_alloc<segment_role_t>, 2> roles_ {};

    /** One GPU's attract, repel and ignore sets. */
    contrastive_edges_pinned_t edges_ {};
};

} // namespace unum::uform::train
