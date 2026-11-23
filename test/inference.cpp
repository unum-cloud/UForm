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
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::fabs`, `std::sqrt`
#include <cstdio>  // `std::fopen`, `std::fread`
#include <cstring> // `std::strlen`, `std::memset`

#include <algorithm>  // `std::equal`
#include <filesystem> // `std::filesystem::exists`, `std::filesystem::remove` — the round trip's files
#include <span>       // `std::span` — the Rust dumps' destination
#include <string>     // `std::string` — the round trip's path
#include <vector>     // `std::vector`

#include <numkong/capabilities.h>
#include <numkong/memory.hpp>

#include "uform/utilities/safetensors_format.hpp"
#include "uform/encoder/multimodal_batch.hpp"
#include "uform/encoder/geometry.hpp"

#include "uform/uform.h" // `uform_model_load_ram`, `uform_text_encode` - the class cells' path

#include "encoder_punned.hpp" // `encoder_verbs`, `encoder_class_tables`, `pick_gpu_variant`, `visit_vram_residency`
#include "harness.hpp"        // `run_test_case`, `expect_true`

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

/** Reads the Float32 values at @p path into @p values , as many as fit, answering how many the
 *  file holds, none when it is missing. */
static std::size_t read_floats(char const* path, std::span<float> values) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return 0;
    std::size_t count = 0;
    for (float value; std::fread(&value, sizeof(value), 1, file) == 1; ++count)
        if (count < values.size()) values[count] = value;
    std::fclose(file);
    return count;
}

/** Encodes the texts through @p verbs in @p precision into @p encoded : on @p executor , a pool
 *  for a host variant or a stream for a device one, the batch and the outputs in memory that both
 *  the host and the device address. Skips a class without E4M3 products, and leaves @p encoded
 *  empty unless it encoded. */
template <vendored_residency_t residency_>
static case_result_t encode(encoder_verbs<residency_> const& verbs, checkpoint_to_load_t const& checkpoint,
                            geometry_t const& geometry, precision_t precision,
                            typename vendored_residency_executor<residency_>::type executor, nk_stream_t stream,
                            encoded_t& encoded) {
    auto [tower, create_status] = verbs.create(checkpoint, geometry, precision, stream);
    if (!create_status && create_status.status_ == status_code_t::unsupported_k && precision == precision_t::e4m3_k)
        return case_result_t::skipped("needs uform_scalar_e4m3_k products in the class");
    if (!expect_true(create_status, "tower: {}", create_status.description_)) return {};

    // A device variant takes its vendor's pinned pages for what both sides address, as its kernels
    // read the batch's segments in place and the host reads the outputs there
    nk_allocator_t shared {}, scratch_policy {};
    if constexpr (residency_ == vendored_residency_t::ram_k) shared = scratch_policy = host_pages_policy();
    else if constexpr (gpu_vendor_compiled(vendor_of(residency_))) {
        shared = entries_kit_of<residency_>::policy(stream, uform_residency_ram_k);
        scratch_policy = entries_kit_of<residency_>::policy(stream, uform_residency_vram_k);
    }
    if (!expect_true(shared.allocate && scratch_policy.allocate, "allocators: none for the device"))
        return verbs.destroy(tower), case_result_t {};
    nk::allocator<b8_t> const allocator {shared, stream};

    using batch_t = encoder::multimodal_batch<nk::allocator<b8_t>>;
    encoder::multimodal_batch_shape_t shape;
    shape.max_rows_per_gpu = items_k, shape.max_segments_per_gpu = items_k,
    shape.max_columns_per_row = geometry.context;
    auto [batch, make_status] = batch_t::make(allocator, shape);
    if (!expect_true(make_status, "batch: {}", make_status.description_)) return verbs.destroy(tower), case_result_t {};
    auto offsets = batch.segment_offsets.span();
    auto lengths = batch.segment_lengths.span();
    auto modalities = batch.segment_modality.span();
    auto grid = batch.tokens.span();
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

    auto [bytes, measure_status] = verbs.scratch_bytes(tower, tokens, items_k, executor);
    if (!expect_true(measure_status, "scratch: {}", measure_status.description_))
        return verbs.destroy(tower), case_result_t {};
    st::basic_vector<b8_t, nk::allocator<b8_t>> scratch {nk::allocator<b8_t> {scratch_policy, stream}};
    using shared_floats_t = st::basic_vector<float, nk::allocator<float>>;
    shared_floats_t states {nk::allocator<float> {allocator}}, embeddings {nk::allocator<float> {allocator}},
        token_vectors {nk::allocator<float> {allocator}}, masked_logits {nk::allocator<float> {allocator}};
    bool const allocated = !st::failed(scratch.resize(st::assume_uninitialized, bytes)) &&
                           !st::failed(states.resize(st::assume_uninitialized, tokens * geometry.hidden)) &&
                           !st::failed(embeddings.resize(st::assume_uninitialized, items_k * geometry.hidden)) &&
                           !st::failed(token_vectors.resize(st::assume_uninitialized, tokens * geometry.token_width)) &&
                           !st::failed(
                               masked_logits.resize(st::assume_uninitialized, items_k * geometry.masked_vocabulary));
    if (!expect_true(allocated, "scratch and outputs allocation failed")) return verbs.destroy(tower), case_result_t {};
    outputs_t const outputs {{f32r2_span_t {embeddings.data(), items_k, geometry.hidden}, nk_f32_k},
                             {f32r2_span_t {states.data(), tokens, geometry.hidden}, nk_f32_k},
                             {f32r2_span_t {token_vectors.data(), tokens, geometry.token_width}, nk_f32_k},
                             {masked_logits.data(), items_k, geometry.masked_vocabulary}};
    expected_t ran = verbs.encode(tower, batch.view(), outputs, scratch, executor);
    nk_status_t finished = nk_success_k;
    if constexpr (residency_ != vendored_residency_t::ram_k && gpu_vendor_compiled(vendor_of(residency_)))
        finished = entries_kit_of<residency_>::synchronize(stream);
    verbs.destroy(tower);
    if (!expect_true(ran, "encode: {}", ran.description_)) return {};
    if (!expect_true(finished == nk_success_k, "encode: {}", nk_status_name(finished))) return {};
    encoded.states.assign(states.begin(), states.end());
    encoded.embeddings.assign(embeddings.begin(), embeddings.end());
    encoded.token_vectors.assign(token_vectors.begin(), token_vectors.end());
    encoded.masked_logits.assign(masked_logits.begin(), masked_logits.end());
    return {};
}

/** Checks @p encoded against @p reference within @p tolerance , naming @p who in the message. */
static void check_agreement(char const* who, encoded_t const& encoded, encoded_t const& reference, std::size_t hidden,
                            double tolerance) {
    std::size_t const rows = reference.states.size() / hidden;
    auto const check = [&](char const* surface, std::vector<float> const& tower, std::vector<float> const& floor,
                           std::size_t surface_rows) {
        if (tower.empty() || floor.empty()) return;
        std::size_t const width = floor.size() / surface_rows;
        agreement_t const agreement = compare({tower.data(), surface_rows, width}, {floor.data(), surface_rows, width});
        expect_true(agreement.relative < tolerance,
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

/** The exact spelling of each CPU class constant, for labels and messages. */
struct cpu_class_spelling_t {
    uform_cpu_class_t value;
    char const* spelling;
};
static constexpr cpu_class_spelling_t cpu_class_spellings_k[] = {
    {uform_cpu_class_serial_k, "uform_cpu_class_serial_k"},
    {uform_cpu_class_monoversioned_k, "uform_cpu_class_monoversioned_k"},
    {uform_cpu_class_intel_haswell_k, "uform_cpu_class_intel_haswell_k"},
    {uform_cpu_class_intel_sierra_k, "uform_cpu_class_intel_sierra_k"},
    {uform_cpu_class_intel_skylake_k, "uform_cpu_class_intel_skylake_k"},
    {uform_cpu_class_intel_icelake_k, "uform_cpu_class_intel_icelake_k"},
    {uform_cpu_class_amd_genoa_k, "uform_cpu_class_amd_genoa_k"},
    {uform_cpu_class_intel_sapphire_k, "uform_cpu_class_intel_sapphire_k"},
    {uform_cpu_class_intel_granite_k, "uform_cpu_class_intel_granite_k"},
    {uform_cpu_class_intel_diamond_k, "uform_cpu_class_intel_diamond_k"},
    {uform_cpu_class_aws_g2_k, "uform_cpu_class_aws_g2_k"},
    {uform_cpu_class_apple_m2_k, "uform_cpu_class_apple_m2_k"},
    {uform_cpu_class_apple_m4_k, "uform_cpu_class_apple_m4_k"},
    {uform_cpu_class_aws_g5_k, "uform_cpu_class_aws_g5_k"},
    {uform_cpu_class_nvidia_vera_k, "uform_cpu_class_nvidia_vera_k"},
    {uform_cpu_class_alibaba_c930_k, "uform_cpu_class_alibaba_c930_k"},
    {uform_cpu_class_ibm_power10_k, "uform_cpu_class_ibm_power10_k"},
    {uform_cpu_class_wasm32_v128_k, "uform_cpu_class_wasm32_v128_k"},
    {uform_cpu_class_wasm64_extended_k, "uform_cpu_class_wasm64_extended_k"},
};

static char const* cpu_class_spelling(uform_cpu_class_t value) {
    for (cpu_class_spelling_t const& entry : cpu_class_spellings_k)
        if (entry.value == value) return entry.spelling;
    return "uform_cpu_class_unknown_k";
}

/** Loads the checkpoint at @p path through the C interface under the ceiling @p class_ , in
 *  @p precision , and when the load picks that class, encodes the texts through
 *  @c uform_text_encode and compares with the serial floor's @p reference . A ceiling the load
 *  resolves lower, as one the machine cannot run or whose kits match its parent's, would only
 *  repeat that lower class. */
static case_result_t test_class_parity(char const* path, uform_cpu_class_t class_, precision_t precision,
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
        return case_result_t::skipped("needs uform_scalar_e4m3_k products in the class");
    if (!expect_true(outcome.status == uform_status_success_k, "load: {}", outcome.message)) return {};
    uform_model_description_t description {};
    uform_model_describe_t describe {.model = model, .outcome = &outcome, .description = &description};
    uform_model_describe(&describe);
    if (description.cpu_class != class_) {
        uform_model_free(model);
        return case_result_t::skipped("needs a machine that runs the class");
    }

    uform_arena_t arena = nullptr;
    uform_arena_init_t init {.arena = &arena, .outcome = &outcome};
    uform_arena_init(&init);
    if (!expect_true(outcome.status == uform_status_success_k, "arena: {}", outcome.message))
        return uform_model_free(model), case_result_t {};
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
    if (!expect_true(outcome.status == uform_status_success_k, "encode: {}", outcome.message)) return {};
    check_agreement(cpu_class_spelling(class_), encoded, reference, states.rows.columns, tolerance);
    return {};
}

/** Encodes in @p precision on @p stream through @p verbs , the variant a device's vendor ladder
 *  picked, and compares with the serial floor's @p reference . */
template <vendored_residency_t residency_>
static case_result_t device_variant_parity(encoder_verbs<residency_> const* verbs,
                                           checkpoint_to_load_t const& checkpoint, geometry_t const& geometry,
                                           precision_t precision, nk_stream_t stream, encoded_t const& reference,
                                           double tolerance) {
    if (!verbs) return case_result_t::skipped("needs a variant for the device's vendor");
    encoded_t encoded;
    case_result_t const encoded_or_skipped = encode(*verbs, checkpoint, geometry, precision, stream, stream, encoded);
    check_agreement(uform_gpu_class_name(static_cast<uform_gpu_class_t>(verbs->gpu_class)), encoded, reference,
                    geometry.hidden, tolerance);
    return encoded_or_skipped;
}

/** Encodes in @p precision on @p device through the newest class its vendor's ladder admits, and
 *  compares with the serial floor's @p reference . */
static case_result_t test_device_parity(checkpoint_to_load_t const& checkpoint, geometry_t const& geometry,
                                        uform_gpu_device_t device, precision_t precision, encoded_t const& reference,
                                        double tolerance) {
    nk::device_kind_t kind = nk::device_kind_t::cuda_k;
    if (device.vendor == uform_gpu_vendor_amd_k) kind = nk::device_kind_t::rocm_k;
    if (device.vendor == uform_gpu_vendor_apple_k) kind = nk::device_kind_t::metal_k;
    auto const [opened, open_status] = nk::device_t::make(kind, device.vendor_ordinal);
    if (!expect_true(nk::succeeded(open_status), "device: {}", nk::status_name(open_status))) return {};
    auto const [capabilities, query_status] = opened.capabilities_enabled();
    if (nk::failed(query_status) || !capabilities) return case_result_t::skipped("needs kernels for the device");
    auto const [stream, create_status] = nk::stream_t::make(opened);
    if (!expect_true(nk::succeeded(create_status), "device stream: {}", nk::status_name(create_status))) return {};
    case_result_t result {};
    visit_vram_residency(capabilities, [&]<vendored_residency_t residency_, typename ladder_type_>() {
        result = device_variant_parity(
            pick_gpu_variant<residency_>(ladder_type_ {}, uform_gpu_class_unknown_k, capabilities), checkpoint,
            geometry, precision, stream.get(), reference, tolerance);
        return true;
    });
    return result;
}

/** Compares the serial floor's BFloat16 @p reference with the Rust crate's dumps at @p states_path
 *  and @p embeddings_path , which skips without them. */
static case_result_t test_rust_parity(char const* states_path, char const* embeddings_path, encoded_t const& reference,
                                      std::size_t hidden) {
    if (states_path == nullptr || embeddings_path == nullptr)
        return case_result_t::skipped("needs the states and embeddings dumps in argv");
    encoded_t rust;
    rust.states.resize(reference.states.size()), rust.embeddings.resize(reference.embeddings.size());
    std::size_t const states_count = read_floats(states_path, rust.states);
    std::size_t const embeddings_count = read_floats(embeddings_path, rust.embeddings);
    if (!expect_true(states_count == reference.states.size() && embeddings_count == reference.embeddings.size(),
                     "rust dumps hold {} and {} floats, expected {} and {}", states_count, embeddings_count,
                     reference.states.size(), reference.embeddings.size()))
        return {};
    check_agreement("the Rust crate", rust, reference, hidden, 5e-2);
    return {};
}

/** Checks every token vector of the serial floor's @p reference has unit length at
 *  the BFloat16 floor. */
static case_result_t test_token_vectors_unit_length(encoded_t const& reference, std::size_t width) {
    if (reference.token_vectors.empty()) return case_result_t::skipped("needs a checkpoint with a token head");
    f32r2_view_t const vectors {reference.token_vectors.data(), reference.token_vectors.size() / width, width};
    double worst = 0;
    for (std::size_t row = 0; row != vectors.extent(0); ++row) {
        double squares = 0;
        for (std::size_t column = 0; column != width; ++column)
            squares += double(vectors[row, column]) * vectors[row, column];
        worst = std::fmax(worst, std::fabs(std::sqrt(squares) - 1));
    }
    expect_true(worst < 1e-2, "token vectors have unit length: worst deviation {:.4e}", worst);
    return {};
}

/** Saves every tensor and metadata pair of @p checkpoint to @p path twice, the second save
 *  replacing the first, and reads them back byte for byte with no @c .tmp sibling left behind.
 *  Float32 entries go first, as the payload packs them ahead of the BFloat16 ones. */
static void test_checkpoint_round_trip(checkpoint_to_load_t const& checkpoint, std::string const& path) {
    checkpoint_to_save_t saved;
    for (checkpoint_metadata_t const& pair : checkpoint.metadata())
        expect_true(saved.metadata(pair.key.view(), pair.value.view()), "metadata {}", pair.key.view());
    for (checkpoint_dtype_t const dtype : {checkpoint_dtype_t::f32_k, checkpoint_dtype_t::bf16_k})
        for (checkpoint_entry_t const& entry : checkpoint.entries()) {
            if (entry.dtype != dtype) continue;
            sz::string_view_t const name = entry.name.view();
            expected_t inserted = dtype == checkpoint_dtype_t::f32_k ? saved.insert(name, checkpoint.float32(name))
                                                                     : saved.insert(name, checkpoint[name]);
            expect_true(inserted, "insert {}: {}", name, inserted.description_);
        }
    for (std::size_t round = 0; round != 2; ++round) {
        expected_t written = saved.save(sz::string_view_t {path});
        expect_true(written, "save {}: {}", round, written.description_);
    }
    expect_true(!std::filesystem::exists(path + ".tmp"), "{}.tmp outlived the save", path);

    auto [checkpoint_loaded, status] = checkpoint_to_load_t::make(sz::string_view_t {path});
    if (expect_true(status, "load: {}", status.description_)) {
        expect_equal(checkpoint_loaded.size(), checkpoint.size(), "tensors");
        expect_equal(checkpoint_loaded.metadata().size(), checkpoint.metadata().size(), "metadata pairs");
        for (checkpoint_entry_t const& entry : checkpoint.entries()) {
            sz::string_view_t const name = entry.name.view();
            auto const [original, read_status] = entry.dtype == checkpoint_dtype_t::f32_k
                                                     ? nk::as_bytes(checkpoint.float32(name)).values()
                                                     : nk::as_bytes(checkpoint[name]).values();
            auto const [reread, reread_status] = entry.dtype == checkpoint_dtype_t::f32_k
                                                     ? nk::as_bytes(checkpoint_loaded.float32(name)).values()
                                                     : nk::as_bytes(checkpoint_loaded[name]).values();
            expect_true(nk::succeeded(read_status) && nk::succeeded(reread_status) &&
                            std::equal(original.begin(), original.end(), reread.begin(), reread.end()),
                        "{} differs", name);
        }
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

/** Runs @ref test_class_parity under a ceiling naming each class of the ladder this build compiled,
 *  in both precisions. */
template <cpu_class_t... classes_>
static void run_class_parity(cpu_ladder<classes_...>, char const* path, encoded_t const& bf16_reference,
                             encoded_t const& e4m3_reference) {
    auto rung = [&]<cpu_class_t class_>() {
        if constexpr (cpu_class_compiled(class_) && class_ != cpu_ladder_floor_k) {
            uform_cpu_class_t const ceiling = static_cast<uform_cpu_class_t>(class_);
            std::string const label = fmt::format("test_class_parity({}, ", cpu_class_spelling(ceiling));
            run_test_case((label + "precision_t::bf16_k)").c_str(), test_class_parity, path, ceiling,
                          precision_t::bf16_k, bf16_reference, 5e-2);
            run_test_case((label + "precision_t::e4m3_k)").c_str(), test_class_parity, path, ceiling,
                          precision_t::e4m3_k, e4m3_reference, 1e-1);
        }
    };
    (rung.template operator()<classes_>(), ...);
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 4)
        return fmt::println(stderr, "Usage: {} <tiny.safetensors> [<states.f32> <embeddings.f32>]", argv[0]), 1;
    start_test_run(argc, argv);
    auto [checkpoint, make_checkpoint_status] = checkpoint_to_load_t::make({argv[1], std::strlen(argv[1])});
    if (!expect_true(make_checkpoint_status, "checkpoint: {}", make_checkpoint_status.description_))
        return finish_test_run();
    run_test_case("test_checkpoint_round_trip", test_checkpoint_round_trip, checkpoint,
                  std::string {argv[1]} + ".round_trip");
    auto [geometry, read_status] = geometry_t::from(checkpoint);
    if (!expect_true(read_status, "geometry: {}", read_status.description_)) return finish_test_run();

    auto [pool, make_pool_status] = executor_t::make(4, caller_exclusivity_t::caller_inclusive_k);
    if (!expect_true(make_pool_status, "pool: {}", make_pool_status.description_)) return finish_test_run();

    encoder_verbs<vendored_residency_t::ram_k> const& reference =
        *encoder_class_tables<vendored_residency_t::ram_k, cpu_ladder_floor_k>::find();
    encoded_t bf16_reference, e4m3_reference;
    encode(reference, checkpoint, geometry, precision_t::bf16_k, pool.view(), nullptr, bf16_reference);
    encode(reference, checkpoint, geometry, precision_t::e4m3_k, pool.view(), nullptr, e4m3_reference);
    if (!expect_true(!bf16_reference.states.empty() && !e4m3_reference.states.empty(),
                     "the serial floor encodes in both precisions"))
        return finish_test_run();
    run_test_case("test_token_vectors_unit_length", test_token_vectors_unit_length, bf16_reference,
                  geometry.token_width);
    run_class_parity(cpu_ladder_t {}, argv[1], bf16_reference, e4m3_reference);
    per_gpu_array<uform_gpu_device_t> const listed = listed_gpu_devices();
    for (i32_t const device : test_run.settings.devices) {
        run_test_case(fmt::format("test_device_parity({}, precision_t::bf16_k)", device).c_str(), test_device_parity,
                      checkpoint, geometry, listed[device], precision_t::bf16_k, bf16_reference, 5e-2);
        run_test_case(fmt::format("test_device_parity({}, precision_t::e4m3_k)", device).c_str(), test_device_parity,
                      checkpoint, geometry, listed[device], precision_t::e4m3_k, e4m3_reference, 1e-1);
    }
    if (test_run.settings.devices.size() == 0)
        run_test_case("test_device_parity", [] { return case_result_t::skipped(no_gpu_device_k); });
    run_test_case("test_rust_parity", test_rust_parity, argc == 4 ? argv[2] : nullptr, argc == 4 ? argv[3] : nullptr,
                  bf16_reference, geometry.hidden);
    return finish_test_run();
}
