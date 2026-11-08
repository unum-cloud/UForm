/**
 *  @file src/train_absent.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The verbs of @c train.h in a build without CUDA: training runs on NVIDIA alone, so no
 *      mesh or trainer is ever created.
 *
 *  Every verb is submitted and deferred like any call. A RAM executor is refused with
 *  @c uform_status_device_k , as the NVIDIA build refuses it, and any other with
 *  @c uform_status_unsupported_k ; the frees stay callable.
 */
#include "lib.hpp"

using namespace unum::uform;

/** Finish a training verb this build cannot run, leaving every handle it would create null. */
template <auto verb_, typename arguments_type_>
static void refuse(arguments_type_ *arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    uform_arena_t arena = nullptr;
    if constexpr (requires { arguments->arena; }) arena = arguments->arena;
    call_outcome_t outcome(arguments->outcome, arena);
    if constexpr (requires { *arguments->mesh = nullptr; }) *arguments->mesh = nullptr;
    if constexpr (requires { *arguments->trainer = nullptr; }) *arguments->trainer = nullptr;
    if (!arguments->executor || !std::holds_alternative<vram_executor_t>(arguments->executor->executor))
        return propagate_error(uform_status_device_k, outcome, "The verb needs a VRAM executor"), void();
    propagate_error(uform_status_unsupported_k, outcome, "This build trains on no GPU");
}

extern "C" {

UFORM_EXPORT void uform_mesh_bootstrap(uform_mesh_bootstrap_t *arguments) { refuse<uform_mesh_bootstrap>(arguments); }

UFORM_EXPORT void uform_mesh_describe(uform_mesh_describe_t *arguments) {
    call_outcome_t outcome(arguments->outcome);
    propagate_error(uform_status_unsupported_k, outcome, "This build trains on no GPU");
}

UFORM_EXPORT void uform_mesh_free(uform_mesh_t mesh) { finish_calls_using(mesh); }

UFORM_EXPORT void uform_trainer_init_vram(uform_trainer_init_vram_t *arguments) {
    refuse<uform_trainer_init_vram>(arguments);
}

UFORM_EXPORT void uform_trainer_step(uform_trainer_step_t *arguments) { refuse<uform_trainer_step>(arguments); }

UFORM_EXPORT void uform_trainer_save(uform_trainer_save_t *arguments) { refuse<uform_trainer_save>(arguments); }

UFORM_EXPORT void uform_trainer_free(uform_trainer_t trainer) { finish_calls_using(trainer); }

} // extern "C"
