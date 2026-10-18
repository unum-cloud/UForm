/**
 *  @brief The Base training binary — a thin @c main over the training library in @c src/train/trainer.hpp .
 *  @file tools/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  It parses the command line into a @ref trainer_config_t , maps the launcher's environment onto the mesh options,
 *  and hands the whole run to @ref trainer_t::run , which owns the mesh, the trainer, the steady loop over synthetic
 *  batches, and the checkpoint.
 *
 *  The same library backs @c test/base.cu 's conformance test and the C verbs, so the driver lives in one place.
 *
 *  The modality mixture is the batch envelope itself: @c --mixture @c tokens (the default) trains the byte
 *  token head alone — span-masked byte pretraining, the "Uniform Byte Models" root the Reranker and Embedder
 *  warm-start from — and @c --mixture @c omni adds the raw-media masked-autoencoder heads (image /
 *  video patches, 1-D audio frames) over one heterogeneous batch. @c --eager drops the model-region capture.
 */
#include <cstdlib> // `::setenv`, `std::getenv` — the launcher-contract boundary
#include <cstring> // `std::strlen`

#include "train/trainer.hpp"

using namespace unum::uform::train;

int main(int argc, char **argv) {
    // Lazy module loading deadlocks against the InfiniBand doorbell kernels' flag spins (see the mixer
    // and base conformance suites); eager loading before `cuInit` front-loads every module.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    trainer_config_t config = trainer_config_t::parse(argc, argv);
    // The library never reads the environment, so the launcher contract maps onto the config and the mesh here:
    // UFORM_DEBUG_ASSERTS arms the gradient-boundary signal probes, the rest place this process in the run.
    if (std::getenv("UFORM_DEBUG_ASSERTS") != nullptr) config.telemetry_channels |= telemetry_signals_k;
    mesh_options_t options;
    options.host_domain_index = host_domain_index_from_environment();
    options.host_domain_count = host_domain_count_from_environment();
    if (char const *multimem = std::getenv("UFORM_GPUS_PER_MULTIMEM_DOMAIN"); multimem != nullptr)
        options.gpus_per_multimem_domain = static_cast<u32_t>(std::atoi(multimem));
    if (char const *directory = std::getenv("UFORM_EXCHANGE_DIR"); directory != nullptr)
        options.directory = sz::string_view_t {directory, std::strlen(directory)};
    if (char const *job = std::getenv("UFORM_JOB_ID"); job != nullptr)
        options.job = sz::string_view_t {job, std::strlen(job)};
    return trainer_t::run(config, options);
}
