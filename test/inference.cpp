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
 *  - Every CPU class this build compiled encodes through the C interface, loaded under a
 *    @c cpu_class_ceiling naming it, and matches the floor at the BFloat16 floor, its kernels
 *    summing in another order. A ceiling the load resolves to a lower class, one this machine does
 *    not run or whose kits match its parent's, skips.
 *  - Every device @c UFORM_DEVICES selects encodes through the newest class its vendor's ladder
 *    admits and matches the floor again at the same bound, its products accumulating in another
 *    order.
 *  - With the Rust crate's dumps, the floor's BFloat16 states and embeddings match them. The crate
 *    deviates from the training recipe in three known places, so agreement is expected at the
 *    BFloat16 floor, not bitwise: its QK-norm reads BFloat16 rather than the Float32 product, its
 *    pooled rows stay Float32, and its head's SiLU runs in Float32.
 *
 *  The checkpoint also round-trips: every tensor and metadata pair is saved twice beside it, the
 *  second save replacing the first, and must read back byte for byte with no staging file left.
 *
 *  Usage: `uform_test_inference <tiny.safetensors> [<rust_states.f32> <rust_embeddings.f32>]`.
 */
#include <cmath>   // `std::fabs`, `std::sqrt`
#include <cstdio>  // `std::fopen`, `std::fread`
#include <cstring> // `std::strlen`, `std::memset`

#include <algorithm>  // `std::equal`
#include <filesystem> // `std::filesystem::exists`, `std::filesystem::remove` — the round trip's files
#include <string>     // `std::string` — the round trip's path
#include <vector>     // `std::vector`

#include <numkong/capabilities.h>
#include <numkong/memory.hpp>

#include "uform/utilities/safetensors_format.hpp"
#include "uform/encoder/multimodal_batch.hpp"
#include "uform/encoder/geometry.hpp"

#include "uform/uform.h" // `uform_model_load_ram`, `uform_text_encode` - the class cells' path

#include "encoder_punned.hpp" // `encoder_verbs`, `encoder_class_tables`, `pick_gpu_variant`
#include "harness.hpp"        // `test_suite_t`

using namespace unum::uform;
using namespace unum::uform::encoder;
using namespace unum::uform::test;

static char const* const texts_k[] = {"Hello, world!", "UForm encodes bytes.", "a"};
static constexpr std::size_t items_k = 3;

/** The largest absolute difference relative to the reference's largest magnitude, and the
 *  worst row cosine. */
struct agreement_t {
    double relative = 0, cosine = 1;
};

/** What one variant encoded from the texts, every surface in Float32; a surface a run
 *  skips is empty. */
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

static std::vector<float> read_floats(char const* path) {
    std::vector<float> values;
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return values;
    float value;
    while (std::fread(&value, sizeof(value), 1, file) == 1) values.push_back(value);
    std::fclose(file);
    return values;
}

/** Encodes the texts through @p verbs in @p precision into @p encoded : on @p executor , a pool
 *  for a host variant or a stream for a device one, with the batch and the outputs in memory both
 *  sides address, placed under @p capabilities . */
template <vendored_residency_t residency_>
static bool encode(test_suite_t& suite, encoder_verbs<residency_> const& verbs, checkpoint_to_load_t const& checkpoint,
                   geometry_t const& geometry, precision_t precision, nk_capability_t capabilities,
                   typename vendored_residency_executor<residency_>::type executor, nk_stream_t stream,
                   encoded_t& encoded) {
    expected<void*> made = verbs.create(checkpoint, geometry, precision, stream);
    if (!made && made.status_ == status_t::unsupported_k && precision == precision_t::e4m3_k)
        return suite.skip("the class has no E4M3 products"), false;
    if (!made) return suite.check(false, "tower: {}", made.description_), false;
    void* const tower = *made;

    bool const on_device = (capabilities & nk_cap_gpus_k) != 0;
    nk_allocator_t shared {}, scratch_policy {};
    nk_status_t const initialized = on_device ? nk_allocator_init_unified_best(&shared, capabilities)
                                              : nk_allocator_init_heap(&shared, 64);
    if (initialized != nk_success_k || (on_device ? nk_allocator_init_device_best(&scratch_policy, capabilities)
                                                  : nk_allocator_init_heap(&scratch_policy, 64)) != nk_success_k)
        return verbs.destroy(tower), suite.check(false, "allocators: {}", nk_status_name(initialized)), false;
    nk::allocator<b8_t> const allocator {shared, stream};

    using batch_t = encoder::multimodal_batch<nk::allocator<b8_t>>;
    encoder::multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = items_k, shape.max_segments_per_gpu = items_k,
    shape.max_columns_per_row = geometry.context;
    expected<batch_t> batch = batch_t::make(allocator, shape);
    if (!batch) return verbs.destroy(tower), suite.check(false, "batch: {}", batch.description_), false;
    auto offsets = batch->segment_offsets.span();
    auto lengths = batch->segment_lengths.span();
    auto modalities = batch->segment_modality.span();
    auto grid = batch->tokens.span();
    std::size_t tokens = 0;
    for (std::size_t item = 0; item != items_k; ++item) {
        std::size_t const length = std::strlen(texts_k[item]);
        offsets[item] = static_cast<u32_t>(item * geometry.context);
        lengths[item] = static_cast<u32_t>(length);
        modalities[item] = encoder::modality_t::text_k;
        for (std::size_t cell = 0; cell != length; ++cell)
            grid[item, cell] = static_cast<token_id_t>(static_cast<unsigned char>(texts_k[item][cell]));
        tokens += length;
    }
    offsets[items_k] = static_cast<u32_t>(items_k * geometry.context);

    expected<std::size_t> bytes = verbs.scratch_bytes(tower, tokens, items_k, executor);
    if (!bytes) return verbs.destroy(tower), suite.check(false, "scratch: {}", bytes.description_), false;
    st::basic_vector<b8_t, nk::allocator<b8_t>> scratch {nk::allocator<b8_t> {scratch_policy, stream}};
    using shared_floats_t = st::basic_vector<float, nk::allocator<float>>;
    shared_floats_t states {nk::allocator<float> {allocator}}, embeddings {nk::allocator<float> {allocator}},
        token_vectors {nk::allocator<float> {allocator}}, masked_logits {nk::allocator<float> {allocator}};
    if (st::failed(scratch.resize(st::assume_uninitialized, *bytes)) ||
        st::failed(states.resize(st::assume_uninitialized, tokens * geometry.hidden)) ||
        st::failed(embeddings.resize(st::assume_uninitialized, items_k * geometry.hidden)) ||
        st::failed(token_vectors.resize(st::assume_uninitialized, tokens * geometry.token_width)) ||
        st::failed(masked_logits.resize(st::assume_uninitialized, items_k * geometry.masked_vocabulary)))
        return verbs.destroy(tower), suite.check(false, "scratch and outputs allocation failed"), false;
    outputs_t const outputs {{f32r2_span_t {embeddings.data(), items_k, geometry.hidden}, nk_f32_k},
                             {f32r2_span_t {states.data(), tokens, geometry.hidden}, nk_f32_k},
                             {f32r2_span_t {token_vectors.data(), tokens, geometry.token_width}, nk_f32_k},
                             {masked_logits.data(), items_k, geometry.masked_vocabulary}};
    expected_t ran = verbs.encode(tower, batch->view(), outputs, scratch, executor);
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
static void check_agreement(test_suite_t& suite, char const* who, encoded_t const& encoded, encoded_t const& reference,
                            std::size_t hidden, double tolerance) {
    std::size_t const rows = reference.states.size() / hidden;
    auto const check = [&](char const* surface, std::vector<float> const& tower, std::vector<float> const& floor,
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

/** Copies @p rows rows of @p view , written as Float32, into @p values . */
static void copy_rows(uform_rows_view_t const& view, std::size_t rows, std::vector<float>& values) {
    values.resize(rows * view.columns);
    for (std::size_t row = 0; row != rows; ++row)
        std::memcpy(values.data() + row * view.columns, static_cast<char const*>(view.data) + row * view.stride,
                    view.columns * sizeof(float));
}

/** Loads the checkpoint at @p path through the C interface under the ceiling @p class_ , in
 *  @p precision , and when the load picks that class, encodes the texts through
 *  @c uform_text_encode and compares with the serial floor's @p reference . A ceiling the load
 *  resolves lower, as one the machine cannot run or whose kits match its parent's, would only
 *  repeat that lower class. */
static void class_parity(test_suite_t& suite, char const* path, uform_cpu_class_t class_, precision_t precision,
                         encoded_t const& reference, double tolerance) {
    uform_outcome_t outcome {};
    uform_model_t model = nullptr;
    uform_model_load_ram_t load {
        .model = &model,
        .outcome = &outcome,
        .path = {reinterpret_cast<std::uint8_t const*>(path), std::strlen(path)},
        .ram = {.cpu_class_ceiling = class_},
        .quantization = precision == precision_t::e4m3_k ? uform_scalar_e4m3_k : uform_scalar_bf16_k};
    uform_model_load_ram(&load);
    if (outcome.status == uform_status_unsupported_k && precision == precision_t::e4m3_k)
        return suite.skip("the class has no E4M3 products");
    if (outcome.status != uform_status_success_k) return suite.check(false, "load: {}", outcome.message);
    uform_model_description_t description {};
    uform_model_describe_t describe {.model = model, .outcome = &outcome, .description = &description};
    uform_model_describe(&describe);
    if (description.cpu_class != class_) {
        uform_model_free(model);
        return suite.skip("the class does not run here, so its ceiling resolves to a lower class");
    }

    uform_arena_t arena = nullptr;
    uform_arena_init_t init {.arena = &arena, .outcome = &outcome};
    uform_arena_init(&init);
    if (outcome.status != uform_status_success_k)
        return uform_model_free(model), suite.check(false, "arena: {}", outcome.message);
    std::string bytes;
    std::size_t offsets[items_k + 1] {};
    for (std::size_t item = 0; item != items_k; ++item) bytes += texts_k[item], offsets[item + 1] = bytes.size();
    uform_rows_view_t embeddings {};
    uform_rows_tape_view_t states {}, token_vectors {};
    uform_text_encode_t encode {.model = model,
                                .outcome = &outcome,
                                .count = items_k,
                                .texts = {bytes.data(), offsets, uform_scalar_u8_k},
                                .embeddings = &embeddings,
                                .states = &states,
                                .token_vectors = description.token_width ? &token_vectors : nullptr,
                                .arena = arena,
                                .outputs_scalar = uform_scalar_f32_k};
    uform_text_encode(&encode);
    encoded_t encoded;
    if (outcome.status == uform_status_success_k) {
        copy_rows(embeddings, items_k, encoded.embeddings);
        copy_rows(states.rows, states.offsets[items_k], encoded.states);
        if (description.token_width)
            copy_rows(token_vectors.rows, token_vectors.offsets[items_k], encoded.token_vectors);
    }
    uform_arena_free(arena);
    uform_model_free(model);
    if (outcome.status != uform_status_success_k) return suite.check(false, "encode: {}", outcome.message);
    check_agreement(suite, uform_cpu_class_name(class_), encoded, reference, states.rows.columns, tolerance);
}

/** Encodes in @p precision on @p stream through @p verbs , the variant a device's vendor ladder
 *  picked, and compares with the serial floor's @p reference . */
template <vendored_residency_t residency_>
static void check_device_variant(test_suite_t& suite, encoder_verbs<residency_> const* verbs,
                                 checkpoint_to_load_t const& checkpoint, geometry_t const& geometry,
                                 precision_t precision, nk_capability_t capabilities, nk_stream_t stream,
                                 encoded_t const& reference, double tolerance) {
    if (!verbs) return suite.skip("this build compiles no variant for the device's vendor");
    encoded_t encoded;
    if (encode(suite, *verbs, checkpoint, geometry, precision, capabilities, stream, stream, encoded))
        check_agreement(suite, uform_gpu_class_name(static_cast<uform_gpu_class_t>(verbs->gpu_class)), encoded,
                        reference, geometry.hidden, tolerance);
}

/** Encodes in @p precision on @p device through the newest class its vendor's ladder admits, and
 *  compares with the serial floor's @p reference . */
static void device_parity(test_suite_t& suite, checkpoint_to_load_t const& checkpoint, geometry_t const& geometry,
                          uform_gpu_device_t device, precision_t precision, encoded_t const& reference,
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
                             pick_gpu_variant<vendored_residency_t::vram_amd_k>(
                                 amd_ladder_t {}, uform_gpu_class_unknown_k, capabilities.value),
                             checkpoint, geometry, precision, capabilities.value, stream.value.get(), reference,
                             tolerance);
    else
        check_device_variant(suite,
                             pick_gpu_variant<vendored_residency_t::vram_nvidia_k>(
                                 nvidia_ladder_t {}, uform_gpu_class_unknown_k, capabilities.value),
                             checkpoint, geometry, precision, capabilities.value, stream.value.get(), reference,
                             tolerance);
}

/** Compares the serial floor's BFloat16 @p reference with the Rust crate's dumps at @p states_path
 *  and @p embeddings_path , which skips without them. */
static void rust_parity(test_suite_t& suite, char const* states_path, char const* embeddings_path,
                        encoded_t const& reference, std::size_t hidden) {
    if (states_path == nullptr || embeddings_path == nullptr) return suite.skip("needs the Rust crate's dumps");
    encoded_t rust {read_floats(states_path), read_floats(embeddings_path)};
    if (rust.states.size() != reference.states.size() || rust.embeddings.size() != reference.embeddings.size())
        return suite.check(false, "rust dumps hold {} and {} floats, expected {} and {}", rust.states.size(),
                           rust.embeddings.size(), reference.states.size(), reference.embeddings.size());
    check_agreement(suite, "the Rust crate", rust, reference, hidden, 5e-2);
}

/** Checks every token vector of the serial floor's @p reference has unit length at
 *  the BFloat16 floor. */
static void token_vectors_unit_length(test_suite_t& suite, encoded_t const& reference, std::size_t width) {
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

/** Saves every tensor and metadata pair of @p checkpoint to @p path twice, the second save
 *  replacing the first, and reads them back byte for byte with no @c .tmp sibling left behind.
 *  Float32 entries go first, as the payload packs them ahead of the BFloat16 ones. */
static void checkpoint_round_trip(test_suite_t& suite, checkpoint_to_load_t const& checkpoint,
                                  std::string const& path) {
    checkpoint_to_save_t saved;
    for (checkpoint_metadata_t const& pair : checkpoint.metadata())
        suite.check(static_cast<bool>(saved.metadata(pair.key.view(), pair.value.view())), "metadata {}",
                    pair.key.view());
    for (checkpoint_dtype_t const dtype : {checkpoint_dtype_t::f32_k, checkpoint_dtype_t::bf16_k})
        for (checkpoint_entry_t const& entry : checkpoint.entries()) {
            if (entry.dtype != dtype) continue;
            sz::string_view_t const name = entry.name.view();
            expected_t inserted = dtype == checkpoint_dtype_t::f32_k ? saved.insert(name, checkpoint.float32(name))
                                                                     : saved.insert(name, checkpoint[name]);
            suite.check(static_cast<bool>(inserted), "insert {}: {}", name, inserted.description_);
        }
    for (std::size_t round = 0; round != 2; ++round) {
        expected_t written = saved.save(sz::string_view_t {path});
        suite.check(static_cast<bool>(written), "save {}: {}", round, written.description_);
    }
    suite.check(!std::filesystem::exists(path + ".tmp"), "{}.tmp outlived the save", path);

    expected<checkpoint_to_load_t> loaded = checkpoint_to_load_t::make(sz::string_view_t {path});
    suite.check(static_cast<bool>(loaded), "load: {}", loaded.description_);
    if (loaded) {
        suite.check_equal(loaded->size(), checkpoint.size(), "tensors");
        suite.check_equal(loaded->metadata().size(), checkpoint.metadata().size(), "metadata pairs");
        for (checkpoint_entry_t const& entry : checkpoint.entries()) {
            sz::string_view_t const name = entry.name.view();
            bytes_view_t const original = entry.dtype == checkpoint_dtype_t::f32_k
                                              ? nk::as_bytes(checkpoint.float32(name)).values().value
                                              : nk::as_bytes(checkpoint[name]).values().value;
            bytes_view_t const reread = entry.dtype == checkpoint_dtype_t::f32_k
                                            ? nk::as_bytes(loaded->float32(name)).values().value
                                            : nk::as_bytes((*loaded)[name]).values().value;
            suite.check(std::equal(original.begin(), original.end(), reread.begin(), reread.end()), "{} differs", name);
        }
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

/** Runs @ref class_parity under a ceiling naming each class of the ladder this build compiled, in
 *  both precisions. */
template <cpu_class_t... classes_>
static void run_class_parity(test_suite_t& suite, cpu_ladder<classes_...>, char const* path,
                             encoded_t const& bf16_reference, encoded_t const& e4m3_reference) {
    auto rung = [&]<cpu_class_t class_>() {
        if constexpr (cpu_class_compiled(class_) && class_ != cpu_ladder_floor_k) {
            uform_cpu_class_t const ceiling = static_cast<uform_cpu_class_t>(class_);
            char const* const name = uform_cpu_class_name(ceiling);
            suite.run(fmt::format("class_parity/{}/bf16", name), class_parity, path, ceiling, precision_t::bf16_k,
                      bf16_reference, 5e-2);
            suite.run(fmt::format("class_parity/{}/e4m3", name), class_parity, path, ceiling, precision_t::e4m3_k,
                      e4m3_reference, 1e-1);
        }
    };
    (rung.template operator()<classes_>(), ...);
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 4)
        return fmt::println(stderr, "Usage: {} <tiny.safetensors> [<states.f32> <embeddings.f32>]", argv[0]), 1;
    test_suite_t suite = test_suite_t::make(argv[0]);
    expected<checkpoint_to_load_t> checkpoint = checkpoint_to_load_t::make({argv[1], std::strlen(argv[1])});
    if (!checkpoint) return suite.check(false, "checkpoint: {}", checkpoint.description_), suite.finish();
    suite.run("checkpoint/round_trip", checkpoint_round_trip, *checkpoint, std::string {argv[1]} + ".round_trip");
    expected<geometry_t> geometry = geometry_t::from(*checkpoint);
    if (!geometry) return suite.check(false, "geometry: {}", geometry.description_), suite.finish();

    expected<executor_t> pool = executor_t::make(4, caller_exclusivity_t::caller_inclusive_k);
    if (!pool) return suite.check(false, "pool: {}", pool.description_), suite.finish();

    encoder_verbs<vendored_residency_t::ram_k> const& reference =
        *encoder_class_tables<vendored_residency_t::ram_k, cpu_ladder_floor_k>::find();
    nk_capability_t const reference_capabilities = numkong_capabilities_of(cpu_ladder_floor_k);
    encoded_t bf16_reference, e4m3_reference;
    if (!encode(suite, reference, *checkpoint, *geometry, precision_t::bf16_k, reference_capabilities, pool->view(),
                nullptr, bf16_reference) ||
        !encode(suite, reference, *checkpoint, *geometry, precision_t::e4m3_k, reference_capabilities, pool->view(),
                nullptr, e4m3_reference))
        return suite.finish();
    suite.run("token_vectors/unit_length", token_vectors_unit_length, bf16_reference, geometry->token_width);
    run_class_parity(suite, cpu_ladder_t {}, argv[1], bf16_reference, e4m3_reference);
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
