/**
 *  @file src_stub/train.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief A stand-in for the @c uform_mesh_* and @c uform_trainer_* C ABI, where training needs a
 *      GPU this library never carries.
 *
 *  The constructors refuse with @c uform_status_device_k, as the engine does on a machine without
 *  NVIDIA devices, so no mesh or trainer handle ever exists here; the verbs taking one refuse a
 *  null handle, and the frees stay callable. The names answer as the engine's do.
 */
#include "lib.hpp"

using namespace unum::uform;

extern "C" {

char const *uform_modality_name(uform_modality_t modality) {
    switch (modality) {
    case uform_modality_text_k: return "text";
    case uform_modality_image_k: return "image";
    case uform_modality_video_k: return "video";
    case uform_modality_audio_k: return "audio";
    default: return "unrecognized";
    }
}

uform_status_t uform_modality_from_name(uform_bytes_view_t name, uform_modality_t *modality) {
    return read_name<uform_modality_t>(
        name, {uform_modality_text_k, uform_modality_image_k, uform_modality_video_k, uform_modality_audio_k},
        uform_modality_name, modality);
}

char const *uform_objective_name(uform_objective_t objective) {
    switch (objective) {
    case uform_objective_masked_k: return "masked";
    case uform_objective_contrastive_k: return "contrastive";
    case uform_objective_joint_k: return "joint";
    default: return "unrecognized";
    }
}

uform_status_t uform_objective_from_name(uform_bytes_view_t name, uform_objective_t *objective) {
    return read_name<uform_objective_t>(
        name, {uform_objective_masked_k, uform_objective_contrastive_k, uform_objective_joint_k}, uform_objective_name,
        objective);
}

char const *uform_negatives_name(uform_negatives_t negatives) {
    switch (negatives) {
    case uform_negatives_ambient_k: return "ambient";
    case uform_negatives_named_k: return "named";
    default: return "unrecognized";
    }
}

uform_status_t uform_negatives_from_name(uform_bytes_view_t name, uform_negatives_t *negatives) {
    return read_name<uform_negatives_t>(name, {uform_negatives_ambient_k, uform_negatives_named_k},
                                        uform_negatives_name, negatives);
}

char const *uform_multimem_tier_name(uform_multimem_tier_t tier) {
    switch (tier) {
    case uform_multimem_tier_switch_reduced_k: return "switch-reduced";
    case uform_multimem_tier_peer_read_k: return "peer-read";
    default: return "unrecognized";
    }
}

uform_status_t uform_multimem_tier_from_name(uform_bytes_view_t name, uform_multimem_tier_t *tier) {
    return read_name<uform_multimem_tier_t>(
        name, {uform_multimem_tier_switch_reduced_k, uform_multimem_tier_peer_read_k}, uform_multimem_tier_name, tier);
}

void uform_mesh_bootstrap(uform_mesh_bootstrap_t *arguments) {
    if (through_executor<uform_mesh_bootstrap>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->mesh) *arguments->mesh = nullptr;
    propagate_error(uform_status_device_k, outcome, "This build carries no GPU backends");
}

void uform_mesh_describe(uform_mesh_describe_t *arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->description) *arguments->description = {};
    reject_if(!arguments->mesh || !arguments->description, uform_status_invalid_argument_k,
              "The mesh and the description must not be null", outcome);
}

void uform_mesh_free(uform_mesh_t mesh) {
    wait_for_calls_using(mesh);
    delete mesh;
}

void uform_trainer_init_vram(uform_trainer_init_vram_t *arguments) {
    if (through_executor<uform_trainer_init_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->trainer) *arguments->trainer = nullptr;
    propagate_error(uform_status_device_k, outcome, "This build carries no GPU backends");
}

void uform_trainer_step(uform_trainer_step_t *arguments) {
    if (through_executor<uform_trainer_step>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    if (arguments->embeddings) *arguments->embeddings = {};
    if (arguments->stats) *arguments->stats = {};
    if (reject_if(!arguments->trainer, uform_status_invalid_argument_k, "The trainer must not be null", outcome))
        return;
    propagate_error(uform_status_device_k, outcome, "This build carries no GPU backends");
}

void uform_trainer_save(uform_trainer_save_t *arguments) {
    if (through_executor<uform_trainer_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->trainer, uform_status_invalid_argument_k, "The trainer must not be null", outcome))
        return;
    propagate_error(uform_status_device_k, outcome, "This build carries no GPU backends");
}

void uform_trainer_free(uform_trainer_t trainer) {
    wait_for_calls_using(trainer);
    delete trainer;
}

} // extern "C"
