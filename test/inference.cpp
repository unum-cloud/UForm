/**
 *  @file test/inference.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Parity of the encoder's class variants with each other, with a device, and with the Rust
 *      crate, on the seeded tiny checkpoint.
 *
 *  Loads the checkpoint `test/tiny_checkpoint.cpp` writes and encodes three texts in BFloat16 and
 *  E4M3 through the serial floor's variant, the reference every other run compares with:
 *
 *  - Every CPU class this build compiled and this machine runs encodes through its own variant and
 *    matches the floor at the BFloat16 floor, its kernels summing in another order. A class whose
 *    kits match its parent's has no variant and skips.
 *  - Every device @c UFORM_DEVICES selects encodes through the newest class its vendor's ladder
 *    admits and matches the floor again: the products accumulate in another order and the rotary
 *    angles take the fast sine.
 *  - With the Rust crate's dumps, the floor's BFloat16 states and embeddings match them. The crate
 *    deviates from the training recipe in three known places, so agreement is expected at the
 *    BFloat16 floor, not bitwise: its QK-norm reads BFloat16 rather than the Float32 product, its
 *    pooled rows stay Float32, and its head's SiLU runs in Float32.
 *
 *  Usage: `uform_test_inference <tiny.safetensors> [<rust_states.f32> <rust_embeddings.f32>]`.
 */
#include <cmath>   // `std::fabs`, `std::sqrt`
#include <cstdio>  // `std::fopen`, `std::fread`
#include <cstring> // `std::strlen`, `std::memset`

#include <vector> // `std::vector`

#include <forkunion.hpp>
#include <numkong/capabilities.h>
#include <numkong/memory.hpp>

#include "uform/core/checkpoint.hpp"
#include "uform/core/multimodal_batch.hpp"
#include "uform/encoder/geometry.hpp"

#include "encoder_punned.hpp" // `encoder_verbs`, `encoder_class_tables`, `pick_gpu_variant`
#include "harness.hpp"        // `test_suite_t`

using namespace unum::uform;
using namespace unum::uform::encoder;
using namespace unum::uform::test;

static char const *const texts_k[] = {"Hello, world!", "UForm encodes bytes.", "a"};
static constexpr std::size_t items_k = 3;

/** The largest absolute difference relative to the reference's largest magnitude, and the worst row cosine. */
struct agreement_t {
    double relative = 0, cosine = 1;
};

/** What one variant encoded from the texts, every surface in Float32; a surface a run skips is
 *  empty. */
struct encoded_t {
    std::vector<float> states, embeddings, token_vectors, masked_logits;
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

/** Encodes the texts through @p verbs in @p precision into @p encoded : on @p executor , a pool
 *  for a host variant or a stream for a device one, with the batch and the outputs in memory both
 *  sides address, placed under @p capabilities . */
template <encoder_home_t home_>
static bool encode(test_suite_t &suite, encoder_verbs<home_> const &verbs, uc::checkpoint_to_load_t const &checkpoint,
                   geometry_t const &geometry, precision_t precision, nk_capability_t capabilities,
                   typename encoder_home_executor<home_>::type executor, nk_stream_t stream, encoded_t &encoded) {
    uc::expected<void *> made = verbs.create(checkpoint, geometry, precision, stream);
    if (!made) return suite.check(false, "tower: {}", made.description_), false;
    void *const tower = *made;

    bool const on_device = (capabilities & nk_cap_gpus_k) != 0;
    nk_allocator_t shared {}, scratch_policy {};
    nk_status_t const initialized = on_device ? nk_allocator_init_unified_best(&shared, capabilities)
                                              : nk_allocator_init_heap(&shared, 64);
    if (initialized != nk_success_k || (on_device ? nk_allocator_init_device_best(&scratch_policy, capabilities)
                                                  : nk_allocator_init_heap(&scratch_policy, 64)) != nk_success_k)
        return verbs.destroy(tower), suite.check(false, "allocators: {}", nk_status_name(initialized)), false;
    nk::allocator<uc::b8_t> const allocator {shared, stream};

    using batch_t = uc::multimodal_batch<nk::allocator<uc::b8_t>>;
    uc::multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = items_k, shape.max_segments_per_gpu = items_k,
    shape.max_columns_per_row = geometry.context;
    uc::expected<batch_t> batch = batch_t::make(allocator, shape);
    if (!batch) return verbs.destroy(tower), suite.check(false, "batch: {}", batch.description_), false;
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

    uc::expected<std::size_t> bytes = verbs.scratch_bytes(tower, tokens, items_k);
    if (!bytes) return verbs.destroy(tower), suite.check(false, "scratch: {}", bytes.description_), false;
    st::basic_vector<uc::b8_t, nk::allocator<uc::b8_t>> scratch {nk::allocator<uc::b8_t> {scratch_policy, stream}};
    using shared_floats_t = st::basic_vector<float, nk::allocator<float>>;
    shared_floats_t states {nk::allocator<float> {allocator}}, embeddings {nk::allocator<float> {allocator}},
        token_vectors {nk::allocator<float> {allocator}}, masked_logits {nk::allocator<float> {allocator}};
    if (st::failed(scratch.resize(st::assume_uninitialized, *bytes)) ||
        st::failed(states.resize(st::assume_uninitialized, tokens * geometry.hidden)) ||
        st::failed(embeddings.resize(st::assume_uninitialized, items_k * geometry.hidden)) ||
        st::failed(token_vectors.resize(st::assume_uninitialized, tokens * geometry.token_width)) ||
        st::failed(masked_logits.resize(st::assume_uninitialized, items_k * geometry.masked_vocabulary)))
        return verbs.destroy(tower), suite.check(false, "scratch and outputs allocation failed"), false;
    outputs_t const outputs {embeddings.data(), states.data(), token_vectors.empty() ? nullptr : token_vectors.data(),
                             masked_logits.empty() ? nullptr : masked_logits.data(), nk_f32_k};
    uc::expected_t ran = verbs.encode(tower, batch->view(), outputs, scratch, executor);
    nk_status_t const finished = on_device ? nk_stream_synchronize_best(capabilities, stream) : nk_success_k;
    verbs.destroy(tower);
    if (!ran) return suite.check(false, "encode: {}", ran.description_), false;
    if (finished != nk_success_k) return suite.check(false, "encode: {}", nk_status_name(finished)), false;
    encoded.states.assign(states.begin(), states.end());
    encoded.embeddings.assign(embeddings.begin(), embeddings.end());
    encoded.token_vectors.assign(token_vectors.begin(), token_vectors.end());
    encoded.masked_logits.assign(masked_logits.begin(), masked_logits.end());
    return true;
}

/** Checks @p encoded against @p reference within @p tolerance , naming @p who in the message. */
static void check_agreement(test_suite_t &suite, char const *who, encoded_t const &encoded, encoded_t const &reference,
                            std::size_t hidden, double tolerance) {
    std::size_t const rows = reference.states.size() / hidden;
    auto const check = [&](char const *surface, std::vector<float> const &tower, std::vector<float> const &floor,
                           std::size_t surface_rows) {
        if (tower.empty() || floor.empty()) return;
        std::size_t const width = floor.size() / surface_rows;
        agreement_t const agreement = compare({tower.data(), surface_rows, width}, {floor.data(), surface_rows, width});
        suite.check(agreement.relative < tolerance,
                    "{} {} match the serial floor: max relative {:.4e} worst cosine {:.6f}", who, surface,
                    agreement.relative, agreement.cosine);
    };
    check("states", encoded.states, reference.states, rows);
    check("embeddings", encoded.embeddings, reference.embeddings, items_k);
    check("token vectors", encoded.token_vectors, reference.token_vectors, rows);
    check("masked logits", encoded.masked_logits, reference.masked_logits, items_k);
}

/** Encodes in @p precision through the variant of @p class_ , when this machine runs it and it
 *  advances past its parent, and compares with the serial floor's @p reference . */
template <cpu_class_t class_>
static void class_parity(test_suite_t &suite, uc::checkpoint_to_load_t const &checkpoint, geometry_t const &geometry,
                         fu::flat_pool_t &pool, nk_capability_t available, precision_t precision,
                         encoded_t const &reference, double tolerance) {
    if (!numkong_admits(available, class_)) return suite.skip("this machine does not run the class");
    encoder_verbs<encoder_home_t::ram_k> const *const verbs =
        encoder_class_tables<encoder_home_t::ram_k, class_>::find();
    if (!verbs) return suite.skip("the class picks the kits its parent does");
    encoded_t encoded;
    if (!encode(suite, *verbs, checkpoint, geometry, precision, numkong_capabilities_of(class_), pool, nullptr,
                encoded))
        return;
    check_agreement(suite, uform_cpu_class_name(static_cast<uform_cpu_class_t>(class_)), encoded, reference,
                    geometry.hidden, tolerance);
}

/** Encodes in @p precision on @p stream through @p verbs , the variant a device's vendor ladder
 *  picked, and compares with the serial floor's @p reference . */
template <encoder_home_t home_>
static void check_device_variant(test_suite_t &suite, encoder_verbs<home_> const *verbs,
                                 uc::checkpoint_to_load_t const &checkpoint, geometry_t const &geometry,
                                 precision_t precision, nk_capability_t capabilities, nk_stream_t stream,
                                 encoded_t const &reference, double tolerance) {
    if (!verbs) return suite.skip("this build compiles no variant for the device's vendor");
    encoded_t encoded;
    if (encode(suite, *verbs, checkpoint, geometry, precision, capabilities, stream, stream, encoded))
        check_agreement(suite, uform_gpu_class_name(static_cast<uform_gpu_class_t>(verbs->gpu_class)), encoded,
                        reference, geometry.hidden, tolerance);
}

/** Encodes in @p precision on @p device through the newest class its vendor's ladder admits, and
 *  compares with the serial floor's @p reference . */
static void device_parity(test_suite_t &suite, uc::checkpoint_to_load_t const &checkpoint, geometry_t const &geometry,
                          uform_gpu_device_t device, precision_t precision, encoded_t const &reference,
                          double tolerance) {
    bool const amd = device.vendor == uform_gpu_vendor_amd_k;
    nk::expected<nk::device_t> const opened = nk::device_t::make(
        amd ? nk::device_kind_t::rocm_k : nk::device_kind_t::cuda_k, device.vendor_ordinal);
    if (!opened) return suite.check(false, "device: {}", nk::status_name(opened.status));
    nk::expected<nk_capability_t> const capabilities = opened.value.capabilities_enabled();
    if (!capabilities || !capabilities.value) return suite.skip("this build holds no kernels for the device");
    nk::expected<nk::stream_t> const stream = nk::stream_t::make(opened.value);
    if (!stream) return suite.check(false, "device stream: {}", nk::status_name(stream.status));
    if (amd)
        check_device_variant(suite,
                             pick_gpu_variant<encoder_home_t::vram_amd_k>(amd_ladder_t {}, uform_gpu_class_unknown_k,
                                                                          capabilities.value),
                             checkpoint, geometry, precision, capabilities.value, stream.value.get(), reference,
                             tolerance);
    else
        check_device_variant(suite,
                             pick_gpu_variant<encoder_home_t::vram_nvidia_k>(
                                 nvidia_ladder_t {}, uform_gpu_class_unknown_k, capabilities.value),
                             checkpoint, geometry, precision, capabilities.value, stream.value.get(), reference,
                             tolerance);
}

/** Compares the serial floor's BFloat16 @p reference with the Rust crate's dumps at @p states_path
 *  and @p embeddings_path , which skips without them. */
static void rust_parity(test_suite_t &suite, char const *states_path, char const *embeddings_path,
                        encoded_t const &reference, std::size_t hidden) {
    if (states_path == nullptr || embeddings_path == nullptr) return suite.skip("needs the Rust crate's dumps");
    encoded_t rust {read_floats(states_path), read_floats(embeddings_path)};
    if (rust.states.size() != reference.states.size() || rust.embeddings.size() != reference.embeddings.size())
        return suite.check(false, "rust dumps hold {} and {} floats, expected {} and {}", rust.states.size(),
                           rust.embeddings.size(), reference.states.size(), reference.embeddings.size());
    check_agreement(suite, "the Rust crate", rust, reference, hidden, 5e-2);
}

/** Checks every token vector of the serial floor's @p reference has unit length at the BFloat16
 *  floor. */
static void token_vectors_unit_length(test_suite_t &suite, encoded_t const &reference, std::size_t width) {
    if (reference.token_vectors.empty()) return suite.skip("the checkpoint carries no token head");
    f32r2_view_t const vectors {reference.token_vectors.data(), reference.token_vectors.size() / width, width};
    double worst = 0;
    for (std::size_t row = 0; row != vectors.extent(0); ++row) {
        double squares = 0;
        for (std::size_t column = 0; column != width; ++column)
            squares += double(vectors[row, column]) * vectors[row, column];
        worst = std::fmax(worst, std::fabs(std::sqrt(squares) - 1));
    }
    suite.check(worst < 1e-2, "token vectors have unit length: worst deviation {:.4e}", worst);
}

/** Runs @ref class_parity for every class of the ladder this build compiled, in both precisions. */
template <cpu_class_t... classes_>
static void run_class_parity(test_suite_t &suite, cpu_ladder<classes_...>, uc::checkpoint_to_load_t const &checkpoint,
                             geometry_t const &geometry, fu::flat_pool_t &pool, nk_capability_t available,
                             encoded_t const &bf16_reference, encoded_t const &e4m3_reference) {
    auto rung = [&]<cpu_class_t class_>() {
        if constexpr (cpu_class_compiled(class_) && class_ != cpu_class_t::serial_k) {
            char const *const name = uform_cpu_class_name(static_cast<uform_cpu_class_t>(class_));
            suite.run(fmt::format("class_parity/{}/bf16", name), class_parity<class_>, checkpoint, geometry, pool,
                      available, precision_t::bf16_k, bf16_reference, 5e-2);
            suite.run(fmt::format("class_parity/{}/e4m3", name), class_parity<class_>, checkpoint, geometry, pool,
                      available, precision_t::e4m3_k, e4m3_reference, 1e-1);
        }
    };
    (rung.template operator()<classes_>(), ...);
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 4)
        return fmt::println(stderr, "Usage: {} <tiny.safetensors> [<states.f32> <embeddings.f32>]", argv[0]), 1;
    test_suite_t suite = test_suite_t::make(argv[0]);
    uc::expected<uc::checkpoint_to_load_t> checkpoint = uc::checkpoint_to_load_t::make({argv[1], std::strlen(argv[1])});
    if (!checkpoint) return suite.check(false, "checkpoint: {}", checkpoint.description_), suite.finish();
    uc::expected<geometry_t> geometry = geometry_t::from(*checkpoint);
    if (!geometry) return suite.check(false, "geometry: {}", geometry.description_), suite.finish();

    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k)
        return suite.check(false, "no CPU capabilities"), suite.finish();
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(4))) return suite.check(false, "pool spawn failed"), suite.finish();

    encoder_verbs<encoder_home_t::ram_k> const &serial =
        *encoder_class_tables<encoder_home_t::ram_k, cpu_class_t::serial_k>::find();
    encoded_t bf16_reference, e4m3_reference;
    if (!encode(suite, serial, *checkpoint, *geometry, precision_t::bf16_k, nk_cap_serial_k, pool, nullptr,
                bf16_reference) ||
        !encode(suite, serial, *checkpoint, *geometry, precision_t::e4m3_k, nk_cap_serial_k, pool, nullptr,
                e4m3_reference))
        return suite.finish();
    suite.run("token_vectors/unit_length", token_vectors_unit_length, bf16_reference, geometry->token_width);
    run_class_parity(suite, cpu_ladder_t {}, *checkpoint, *geometry, pool, available, bf16_reference, e4m3_reference);
    per_gpu_array<uform_gpu_device_t> const listed = listed_gpu_devices();
    for (i32_t const device : suite.settings.devices) {
        suite.run(fmt::format("device_parity/{}/bf16", device), device_parity, *checkpoint, *geometry, listed[device],
                  precision_t::bf16_k, bf16_reference, 5e-2);
        suite.run(fmt::format("device_parity/{}/e4m3", device), device_parity, *checkpoint, *geometry, listed[device],
                  precision_t::e4m3_k, e4m3_reference, 1e-1);
    }
    if (suite.settings.devices.size() == 0) suite.skip("device_parity", "no GPU device listed");
    suite.run("rust_parity", rust_parity, argc == 4 ? argv[2] : nullptr, argc == 4 ? argv[3] : nullptr, bf16_reference,
              geometry->hidden);
    return suite.finish();
}
