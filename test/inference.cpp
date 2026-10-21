/**
 *  @brief Parity of the host tower against the Rust crate on the seeded tiny checkpoint.
 *  @file test/inference.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *
 *  Loads the checkpoint `test/tiny_checkpoint.cpp` writes, encodes three texts on the CPU in BFloat16 and E4M3, and
 *  compares the BFloat16 states and embeddings against raw Float32 dumps of the Rust crate's outputs for the same
 *  texts. The Rust crate deviates from the training recipe in three known places, so agreement is expected at the
 *  BFloat16 floor, not bitwise: its QK-norm reads BFloat16 rather than the Float32 product, its pooled rows stay
 *  Float32, and its head's SiLU runs in Float32. It also keeps its states in Float32 where the tower rounds them.
 *
 *  With a CUDA device present, both precisions also encode on it and compare against the CPU, expecting the
 *  BFloat16 floor again: the products accumulate in another order and the rotary angles take the fast sine.
 *
 *  Usage: `uform_test_inference <tiny.safetensors> [<rust_states.f32> <rust_embeddings.f32>]`.
 */
#include <cmath>   // `std::fabs`, `std::sqrt`
#include <cstdio>  // `std::fopen`, `std::fread`
#include <cstring> // `std::strlen`, `std::memset`

#include <vector> // `std::vector`

#include <forkunion.hpp>
#include <numkong/capabilities.h>
#include <numkong/dots.h> // `nk_dots_pack_size_*_best` — which precisions a device runs
#include <numkong/memory.hpp>

#include "core/checkpoint.hpp"
#include "core/multimodal_batch.hpp"
#include "inference/geometry.hpp"
#include "inference/tower.hpp"
#include "harness.hpp" // `test_suite_t`

using namespace unum::uform::inference;
using namespace unum::uform::test;
namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;

static char const *const texts_k[] = {"Hello, world!", "UForm encodes bytes.", "a"};
static constexpr std::size_t items_k = 3;

/** The largest absolute difference relative to the reference's largest magnitude, and the worst row cosine. */
struct agreement_t {
    double relative = 0, cosine = 1;
};

static agreement_t compare(f32r2_view_t tower, f32r2_view_t reference) {
    agreement_t agreement;
    double largest_difference = 0, largest_reference = 0;
    for (std::size_t row = 0; row != tower.extent(0); ++row) {
        double dot = 0, tower_norm = 0, reference_norm = 0;
        for (std::size_t column = 0; column != tower.extent(1); ++column) {
            double const t = tower[row, column], r = reference[row, column];
            largest_difference = std::fmax(largest_difference, std::fabs(t - r));
            largest_reference = std::fmax(largest_reference, std::fabs(r));
            dot += t * r, tower_norm += t * t, reference_norm += r * r;
        }
        agreement.cosine = std::fmin(agreement.cosine, dot / std::sqrt(tower_norm * reference_norm));
    }
    agreement.relative = largest_difference / largest_reference;
    return agreement;
}

static std::vector<float> read_floats(char const *path) {
    std::vector<float> values;
    std::FILE *file = std::fopen(path, "rb");
    if (!file) return values;
    float value;
    while (std::fread(&value, sizeof(value), 1, file) == 1) values.push_back(value);
    std::fclose(file);
    return values;
}

/** Encodes the texts in @p precision into @p states and @p embeddings , both Float32: on @p pool 's threads for a
 *  CPU mask, or on @p stream for a GPU one, with the batch and the outputs in memory both sides address. */
static bool encode(test_suite_t &suite, uc::checkpoint_to_load_t const &checkpoint, geometry_t const &geometry,
                   precision_t precision, nk_capability_t capabilities, nk_stream_t stream, fu::flat_pool_t &pool,
                   std::vector<float> &states, std::vector<float> &embeddings) {
    placement_t placement {capabilities, stream, {}};
    uc::expected<tower_t> tower = tower_t::make(checkpoint, geometry, precision, placement);
    if (!tower) return suite.check(false, "tower: {}", tower.description_), false;

    bool const on_device = (capabilities & nk_cap_gpus_k) != 0;
    nk_allocator_t shared {}, scratch_policy {};
    nk_status_t const initialized = on_device ? nk_allocator_init_unified_best(&shared, capabilities)
                                              : nk_allocator_init_heap(&shared, 64);
    if (initialized != nk_success_k || (on_device ? nk_allocator_init_device_best(&scratch_policy, capabilities)
                                                  : nk_allocator_init_heap(&scratch_policy, 64)) != nk_success_k)
        return suite.check(false, "allocators: {}", nk_status_name(initialized)), false;
    nk::allocator<uc::b8_t> const allocator {shared, stream};

    using batch_t = uc::multimodal_batch<nk::allocator<uc::b8_t>>;
    uc::multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = items_k, shape.max_segments_per_gpu = items_k,
    shape.max_columns_per_row = geometry.context;
    uc::expected<batch_t> batch = batch_t::make(allocator, shape);
    if (!batch) return suite.check(false, "batch: {}", batch.description_), false;
    auto offsets = batch->segment_offsets.span();
    auto lengths = batch->segment_lengths.span();
    auto modalities = batch->segment_modality.span();
    auto grid = batch->tokens.span();
    std::size_t tokens = 0;
    for (std::size_t item = 0; item != items_k; ++item) {
        std::size_t const length = std::strlen(texts_k[item]);
        offsets[item] = static_cast<uc::u32_t>(item * geometry.context);
        lengths[item] = static_cast<uc::u32_t>(length);
        modalities[item] = uc::modality_t::text_k;
        for (std::size_t cell = 0; cell != length; ++cell)
            grid[item, cell] = static_cast<uc::token_id_t>(static_cast<unsigned char>(texts_k[item][cell]));
        tokens += length;
    }
    offsets[items_k] = static_cast<uc::u32_t>(items_k * geometry.context);

    uc::expected<std::size_t> bytes = tower->scratch_bytes(tokens, items_k, capabilities);
    if (!bytes) return suite.check(false, "scratch: {}", bytes.description_), false;
    st::basic_vector<uc::b8_t, nk::allocator<uc::b8_t>> scratch {nk::allocator<uc::b8_t> {scratch_policy, stream}};
    st::basic_vector<float, nk::allocator<float>> shared_states {nk::allocator<float> {allocator}},
        shared_embeddings {nk::allocator<float> {allocator}};
    if (st::failed(scratch.resize(st::assume_uninitialized, *bytes)) ||
        st::failed(shared_states.resize(st::assume_uninitialized, tokens * geometry.hidden)) ||
        st::failed(shared_embeddings.resize(st::assume_uninitialized, items_k * geometry.hidden)))
        return suite.check(false, "scratch and outputs allocation failed"), false;
    placement.scratch = scratch;
    outputs_t const outputs {shared_embeddings.data(), shared_states.data(), nk_f32_k};
    uc::expected_t encoded = on_device ? tower->encode(batch->view(), outputs, placement)
                                       : tower->encode(batch->view(), outputs, placement, pool);
    if (!encoded) return suite.check(false, "encode: {}", encoded.description_), false;
    nk_status_t const finished = nk_stream_synchronize_best(capabilities, stream);
    if (finished != nk_success_k) return suite.check(false, "encode: {}", nk_status_name(finished)), false;
    states.assign(shared_states.data(), shared_states.data() + shared_states.size());
    embeddings.assign(shared_embeddings.data(), shared_embeddings.data() + shared_embeddings.size());
    return true;
}

/** Encodes in @p precision on the first CUDA device, when there is one, and compares with the
 *  CPU's @p states and @p embeddings within @p tolerance ; a precision NumKong has no device
 *  kernels for skips. */
static void device_parity(test_suite_t &suite, uc::checkpoint_to_load_t const &checkpoint, geometry_t const &geometry,
                          fu::flat_pool_t &pool, precision_t precision, std::vector<float> const &states,
                          std::vector<float> const &embeddings, double tolerance) {
    nk_size_t devices = 0;
    nk_capability_t capabilities = 0;
    if (nk_cuda_count_devices(&devices) != nk_success_k || devices == 0 ||
        nk_cuda_capabilities_enabled(0, &capabilities) != nk_success_k || capabilities == 0)
        return suite.skip(no_cuda_device_k);
    std::size_t const hidden = geometry.hidden, tokens = states.size() / hidden;
    nk_size_t bytes = 0;
    nk_status_t const supported = precision == precision_t::bf16_k
                                      ? nk_dots_pack_size_bf16_best(hidden, hidden, capabilities, &bytes)
                                      : nk_dots_pack_size_e4m3_best(hidden, hidden, capabilities, &bytes);
    if (supported != nk_success_k) return suite.skip(nk_status_name(supported));
    nk_stream_t stream = nullptr;
    if (nk_stream_init_cuda(0, &stream) != nk_success_k) return suite.check(false, "device stream init");
    std::vector<float> device_states, device_embeddings;
    if (encode(suite, checkpoint, geometry, precision, capabilities, stream, pool, device_states, device_embeddings)) {
        agreement_t const states_agreement = compare({device_states.data(), tokens, hidden},
                                                     {states.data(), tokens, hidden});
        agreement_t const embeddings_agreement = compare({device_embeddings.data(), items_k, hidden},
                                                         {embeddings.data(), items_k, hidden});
        suite.check(states_agreement.relative < tolerance && embeddings_agreement.relative < tolerance,
                    "device matches the CPU: states max relative {:.4e} worst cosine {:.6f}, embeddings max " //
                    "relative {:.4e} worst cosine {:.6f}",
                    states_agreement.relative, states_agreement.cosine, embeddings_agreement.relative,
                    embeddings_agreement.cosine);
    }
    suite.check(nk_stream_free_cuda(stream) == nk_success_k, "device stream free");
}

/** Compares the CPU's BFloat16 @p states and @p embeddings with the Rust crate's dumps at
 *  @p states_path and @p embeddings_path , which skips without them. */
static void rust_parity(test_suite_t &suite, char const *states_path, char const *embeddings_path,
                        std::vector<float> const &states, std::vector<float> const &embeddings, std::size_t hidden) {
    if (states_path == nullptr || embeddings_path == nullptr) return suite.skip("needs the Rust crate's dumps");
    std::vector<float> const rust_states = read_floats(states_path), rust_embeddings = read_floats(embeddings_path);
    if (rust_states.size() != states.size() || rust_embeddings.size() != embeddings.size())
        return suite.check(false, "rust dumps hold {} and {} floats, expected {} and {}", rust_states.size(),
                           rust_embeddings.size(), states.size(), embeddings.size());
    std::size_t const state_rows = states.size() / hidden;
    agreement_t const states_agreement = compare({states.data(), state_rows, hidden},
                                                 {rust_states.data(), state_rows, hidden});
    agreement_t const embeddings_agreement = compare({embeddings.data(), items_k, hidden},
                                                     {rust_embeddings.data(), items_k, hidden});
    suite.check(states_agreement.relative < 5e-2 && embeddings_agreement.relative < 5e-2,
                "bf16 matches the Rust crate: states max relative {:.4e} worst cosine {:.6f}, embeddings max " //
                "relative {:.4e} worst cosine {:.6f}",
                states_agreement.relative, states_agreement.cosine, embeddings_agreement.relative,
                embeddings_agreement.cosine);
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 4)
        return fmt::println(stderr, "Usage: {} <tiny.safetensors> [<states.f32> <embeddings.f32>]", argv[0]), 1;
    test_suite_t suite = test_suite_t::make(argv[0]);
    uc::expected<uc::checkpoint_to_load_t> checkpoint = uc::checkpoint_to_load_t::make({argv[1], std::strlen(argv[1])});
    if (!checkpoint) return suite.check(false, "checkpoint: {}", checkpoint.description_), suite.finish();
    uc::expected<geometry_t> geometry = geometry_t::from(*checkpoint);
    if (!geometry) return suite.check(false, "geometry: {}", geometry.description_), suite.finish();

    nk_capability_t capabilities = 0;
    if (nk_cpu_capabilities_enabled(&capabilities) != nk_success_k)
        return suite.check(false, "no CPU capabilities"), suite.finish();
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(4))) return suite.check(false, "pool spawn failed"), suite.finish();

    std::vector<float> states, embeddings, quantized_states, quantized_embeddings;
    if (!encode(suite, *checkpoint, *geometry, precision_t::bf16_k, capabilities, nullptr, pool, states, embeddings) ||
        !encode(suite, *checkpoint, *geometry, precision_t::e4m3_k, capabilities, nullptr, pool, quantized_states,
                quantized_embeddings))
        return suite.finish();
    suite.run("device_parity/bf16", device_parity, *checkpoint, *geometry, pool, precision_t::bf16_k, states,
              embeddings, 5e-2);
    suite.run("device_parity/e4m3", device_parity, *checkpoint, *geometry, pool, precision_t::e4m3_k, quantized_states,
              quantized_embeddings, 1e-1);
    suite.run("rust_parity", rust_parity, argc == 4 ? argv[2] : nullptr, argc == 4 ? argv[3] : nullptr, states,
              embeddings, geometry->hidden);
    return suite.finish();
}
