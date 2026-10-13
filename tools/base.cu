/**
 *  @brief The Base training binary — a thin @c main over the training library in @c src/train/base.cuh . It
 *      parses the command line into a @ref base_config_t and hands the whole run to @ref base_trainer_t::run ,
 *      which owns the composition root, the two-region capture, the steady loop, and the checkpoint.
 *  @file tools/base.cu
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  The same library backs @c test/base.cu 's conformance test, so the driver lives in one place and neither
 *  consumer copies the other.
 *
 *  The modality mixture is the batch envelope itself: @c --mixture @c tokens (the default) trains the byte
 *  token head alone — span-masked byte pretraining, the "Uniform Byte Models" root the Reranker and Embedder
 *  warm-start from — and @c --mixture @c omni adds the raw-media masked-autoencoder heads (image /
 *  video patches, 1-D audio frames) over one heterogeneous batch. @c --eager drops the model-region capture.
 */
#include <cstdlib> // `::setenv`, `std::getenv` — the launcher-contract boundary

#include "train/base.cuh"

using namespace unum::uform::train;

int main(int argc, char **argv) {
    // Lazy module loading deadlocks against the InfiniBand doorbell kernels' flag spins (see the mixer
    // and base conformance suites); eager loading before `cuInit` front-loads every module.
    ::setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    base_config_t config = base_config_t::parse(argc, argv);
    // The ledger's one environment boundary: the library never reads env, so the launcher contract maps
    // onto the config here — UFORM_DEBUG_ASSERTS arms the gradient-boundary signal probes.
    if (std::getenv("UFORM_DEBUG_ASSERTS") != nullptr) config.telemetry_channels |= telemetry_signals_k;
    return base_trainer_t::run(config);
}
