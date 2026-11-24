/**
 *  @file cpp/uform/train.hpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief The training verbs of @c uform/train.h: a mesh joining the processes of a run, a trainer
 *      owning one replica per device, and the receivers stepping and saving it.
 *
 *  Training is GPU-only, so a mesh and a trainer are made over a VRAM executor. Every
 *  configuration crosses as the C struct it mirrors, and a field left out is zero, derived.
 *
 *  @code{.cpp}
 *  auto [mesh, mesh_status] = uf::mesh_t::bootstrap({.executor = executor.handle(), .options = options});
 *  auto [trainer, trainer_status] = uf::trainer_t::init_vram({.executor = executor.handle(), .depth = 12,
 *      .hidden = 512, .batch = batch, .objective = uform_objective_joint_k, .schedule = {.steps = 100000},
 *      .mesh = mesh.handle(), .seed = 42});
 *  auto [embeddings, step_status] = trainer.on(executor).step(modalities, contents, {.relations = relations});
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <span> // `std::span`

#include "uform/train.h"   // `uform_mesh_*` and `uform_trainer_*` verbs and their argument structs
#include "uform/types.hpp" // `deferrable`, `answered_`, `opened_`, `owner`, `executor_t`, `arena_t`

namespace unum::uform {

/** Owns one process's place among the processes of a run; every trainer made over it must be
 *  freed first. */
class mesh_t {
    owner<uform_mesh_t, uform_mesh_free> owner_;

  public:
    uform_mesh_t handle() const noexcept { return owner_.handle(); }
    uform_mesh_t* handle_field() noexcept { return owner_.handle_field(); }

    /** Joins this process to the run's mesh, blocking until every process has joined. */
    static expected<mesh_t> bootstrap(uform_mesh_bootstrap_t arguments) noexcept {
        arguments.completion = uform_completion_blocking_k;
        return opened_<mesh_t>(uform_mesh_bootstrap, arguments, &uform_mesh_bootstrap_t::mesh);
    }

    /** Reports the mesh's shape and this process's place in it. */
    expected<uform_mesh_description_t> describe() const noexcept {
        return answered_<expected<uform_mesh_description_t>>(
            [&](uform_outcome_t* outcome, outputs<uform_mesh_description_t>* written) noexcept {
                uform_mesh_describe_t arguments {.mesh = owner_.handle(),
                                                 .outcome = outcome,
                                                 .description = &written->value};
                uform_mesh_describe(&arguments);
            });
    }
};

/** The inputs and knobs of a step beside its modalities and contents; every empty span and tape
 *  is left out of the call. */
struct trainer_step_config_t {

    /** The bag of each item, whose items land on one device; empty keeps every item its own bag. */
    std::span<std::uint32_t const> bags {};

    /** The temperature group of each item; empty puts every item in group zero. */
    std::span<std::uint32_t const> temperature_groups {};

    /** The late-interaction side of each item; empty takes no item into late interaction. */
    std::span<uform_late_interaction_side_t const> sides {};

    /** The legal choices of each item, one byte each, as @c text_choose takes them. */
    tape_view<char> choices {};

    /** The target probability of each choice, in @c choices order, as @c text_choose answers. */
    tape_view<float> choice_targets {};

    /** The attract, repel and ignore edges over the step's items. */
    uform_relations_view_t relations {};

    /** Where the pooled embeddings the loss scored land; @c nullptr skips the copy. */
    arena_t* arena = nullptr;

    /** The scalar type the embeddings land as. */
    uform_scalar_t outputs_scalar = uform_scalar_unknown_k;

    /** The report the step overwrites; @c nullptr skips it and the readback of the losses. */
    uform_trainer_step_stats_t* stats = nullptr;
};

/** A trainer and its verbs, completed as @p receiver_type_ says: the owning handle and @c on block,
 *  while @c deferred answers deferred calls, whose every input stays the caller's until final. */
template <typename receiver_type_>
class trainer {
    static constexpr uform_completion_t completion_k = receiver_type_::completion;
    template <typename value_type_ = void>
    using deferrable = uform::deferrable<value_type_, completion_k>;
    receiver_type_ receiver_;

    template <typename>
    friend class trainer;
    explicit trainer(receiver_type_ receiver) noexcept : receiver_(std::move(receiver)) {}

  public:
    trainer() noexcept
        requires owns_handle<receiver_type_>
    = default;

    uform_trainer_t handle() const noexcept { return receiver_.handle(); }

    /** The output field a lifecycle verb writes the new handle into. */
    uform_trainer_t* handle_field() noexcept
        requires owns_handle<receiver_type_>
    {
        return receiver_.handle_field();
    }

    /** The same handle, whose verbs block on @p executor. */
    trainer<caller_owned<uform_trainer_t, uform_completion_blocking_k>> on(executor_t const& executor) const noexcept {
        return trainer<caller_owned<uform_trainer_t, uform_completion_blocking_k>>(
            {receiver_.handle(), executor.handle()});
    }

    /** The same handle, whose verbs submit to @p executor, each answering a @c deferred call. */
    trainer<caller_owned<uform_trainer_t, uform_completion_deferred_k>> deferred(
        executor_t const& executor) const noexcept {
        return trainer<caller_owned<uform_trainer_t, uform_completion_deferred_k>>(
            {receiver_.handle(), executor.handle()});
    }

    /** Applies one optimizer step over this process's items, tagged by @p modalities, and every
     *  peer's, answering the pooled embeddings the loss scored when @p config names an arena. */
    deferrable<written_rows_t> step(std::span<uform_modality_t const> modalities, tape_view<std::uint8_t> contents,
                                    trainer_step_config_t const& config = {}) const noexcept {
        if (status_t const refusal = contents.check(); refusal.failed())
            return answered_<deferrable<written_rows_t>>(refusal);
        std::size_t const count = contents.count();
        // The call has one count, so only this side sees the per-item inputs disagree with it.
        if (modalities.size() != count)
            return answered_<deferrable<written_rows_t>>(
                {status_code_t::shape_mismatch_k, "Every item needs one modality"});
        if ((!config.bags.empty() && config.bags.size() != count) ||
            (!config.temperature_groups.empty() && config.temperature_groups.size() != count) ||
            (!config.sides.empty() && config.sides.size() != count))
            return answered_<deferrable<written_rows_t>>(
                {status_code_t::shape_mismatch_k, "Pass no bags, groups or sides, or one per item"});
        if (!config.choices.offsets.empty()) {
            if (status_t const refusal = config.choices.check(); refusal.failed())
                return answered_<deferrable<written_rows_t>>(refusal);
            if (status_t const refusal = config.choice_targets.check(); refusal.failed())
                return answered_<deferrable<written_rows_t>>(refusal);
            if (config.choices.count() != count || config.choice_targets.count() != count)
                return answered_<deferrable<written_rows_t>>(
                    {status_code_t::shape_mismatch_k, "Every item needs its choices and their targets"});
        }
        return answered_<deferrable<written_rows_t>>(
            [&](uform_outcome_t* outcome, outputs<written_rows_t>* written) noexcept {
                uform_trainer_step_t arguments {
                    .trainer = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = count,
                    .modalities = modalities.data(),
                    .contents = contents.view(),
                    .bags = config.bags.data(),
                    .temperature_groups = config.temperature_groups.data(),
                    .sides = config.sides.data(),
                    .choices = config.choices.view(),
                    .choice_targets = config.choice_targets.view(),
                    .relations = config.relations,
                    .outputs_scalar = config.outputs_scalar,
                    .stats = config.stats,
                };
                if (config.arena) {
                    arguments.embeddings = &written->value.view;
                    arguments.arena = config.arena->handle();
                    written->value.written = count;
                }
                uform_trainer_step(&arguments);
            });
    }

    /** Writes the weights and the step to a safetensors checkpoint at @p path, which only the
     *  process of index zero writes. */
    deferrable<> save(uform_bytes_view_t path, progress_t progress = {}) const noexcept {
        return answered_<deferrable<>>([&](uform_outcome_t* outcome, outputs<void>*) noexcept {
            uform_trainer_save_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k, path,
                                            progress.raw()};
            uform_trainer_save(&arguments);
        });
    }

    /** Builds one replica per device of the executor @p arguments names and starts or resumes the
     *  run, blocking until every process of the run has. */
    static expected<trainer> init_vram(uform_trainer_init_vram_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        arguments.completion = uform_completion_blocking_k;
        return opened_<trainer>(uform_trainer_init_vram, arguments, &uform_trainer_init_vram_t::trainer);
    }
};

/** Owns one training run on a VRAM executor's devices; destruction waits for its deferred calls. */
using trainer_t = trainer<owning<uform_trainer_t, uform_trainer_free>>;

/** A trainer the caller owns and its executor, completing its verbs as @p completion_ says. */
template <uform_completion_t completion_>
using trainer_on = trainer<caller_owned<uform_trainer_t, completion_>>;

} // namespace unum::uform
