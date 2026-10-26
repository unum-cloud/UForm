/**
 *  @file test/tiny_checkpoint.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief Writes a seeded, tiny but structurally complete UForm checkpoint, so the C suite and the bindings
 *      load a real tower on any machine without a training run.
 *
 *  The checkpoint is safetensors in BFloat16: every tensor the tower reads, under the names
 *  training saves, and the full geometry in the safetensors metadata with every value a string. The
 *  tower is 2 blocks of hidden width 256, SwiGLU width 512, 8-pixel patches, 2-frame tubes and
 *  256-sample audio frames, with a 32-wide token head and the byte decoder, so token vectors and
 *  choices run too. Weights are xorshift draws scaled to about 0.02 and every norm scale is one, so
 *  encodings are finite and reproducible.
 *
 *  Usage: `tiny_checkpoint <output.safetensors>`.
 */
#include <cstdint> // `std::uint16_t`, `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::fopen`, `std::fwrite`, `std::fclose`, `std::fputs`

#include <bit>    // `std::bit_cast`
#include <string> // `std::string`, `std::to_string`
#include <vector> // `std::vector`

#include <fmt/format.h> // `fmt::println`

/** The tower's geometry, every value spelled as the metadata spells it. */
static constexpr std::size_t hidden_k = 256, depth_k = 2, ffn_expansion_k = 2, patch_side_k = 8, tube_frames_k = 2;
static constexpr std::size_t image_channels_k = 4, video_channels_k = 3, chunk_samples_k = 256, token_width_k = 32;

/** The byte vocabulary of the text entry: 256 byte values, then the mask and padding tokens. */
static constexpr std::size_t vocabulary_k = 258;

/** One tensor to write: its name, its two extents, and how it starts, a norm scale at one. */
struct tensor_plan_t {
    std::string name;
    std::size_t rows;
    std::size_t columns;
    enum class fill_t { random_k, ones_k } fill;
};

/** Narrows to the high half of an IEEE single, rounding half to even. */
static std::uint16_t to_bf16(float value) {
    std::uint32_t const bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

/** Every tensor the tower needs, in the order they are written. */
static std::vector<tensor_plan_t> plan_tensors() {
    using fill_t = tensor_plan_t::fill_t;
    std::size_t const ffn = ffn_expansion_k * hidden_k;
    std::vector<tensor_plan_t> plans;
    plans.push_back({"model.embedding", vocabulary_k, hidden_k, fill_t::random_k});
    for (std::size_t block = 0; block != depth_k; ++block) {
        std::string const prefix = "model.blocks." + std::to_string(block) + ".";
        plans.push_back({prefix + "qkv", 3 * hidden_k, hidden_k, fill_t::random_k});
        plans.push_back({prefix + "out", hidden_k, hidden_k, fill_t::random_k});
        plans.push_back({prefix + "gate_up", 2 * ffn, hidden_k, fill_t::random_k});
        plans.push_back({prefix + "down", hidden_k, ffn, fill_t::random_k});
        plans.push_back({prefix + "pre_norm", 1, hidden_k, fill_t::ones_k});
        plans.push_back({prefix + "post_norm", 1, hidden_k, fill_t::ones_k});
    }
    plans.push_back({"model.final_norm", 1, hidden_k, fill_t::ones_k});
    plans.push_back({"model.pool_query", 1, hidden_k, fill_t::random_k});
    plans.push_back({"model.head.norm", 1, hidden_k, fill_t::ones_k});
    plans.push_back({"model.head.fc1", hidden_k, hidden_k, fill_t::random_k});
    plans.push_back({"model.head.fc2", hidden_k, hidden_k, fill_t::random_k});
    plans.push_back({"model.token_head.norm", 1, hidden_k, fill_t::ones_k});
    plans.push_back({"model.token_head.projection", token_width_k, hidden_k, fill_t::random_k});
    plans.push_back({"model.masked_tokens", vocabulary_k, hidden_k, fill_t::random_k});
    std::size_t const patch_pixels = patch_side_k * patch_side_k;
    plans.push_back({"model.image_projection", hidden_k, patch_pixels * image_channels_k, fill_t::random_k});
    plans.push_back(
        {"model.video_projection", hidden_k, tube_frames_k * patch_pixels * video_channels_k, fill_t::random_k});
    plans.push_back({"model.audio_projection", hidden_k, chunk_samples_k, fill_t::random_k});
    return plans;
}

/** The geometry keys training writes at save, in the order the reader documents them. */
static std::string metadata_json() {
    std::string const entries[][2] = {
        {"step", "0"},
        {"depth", std::to_string(depth_k)},
        {"hidden", std::to_string(hidden_k)},
        {"ffn_expansion", std::to_string(ffn_expansion_k)},
        {"head_width", "128"},
        {"rms_epsilon", "1e-06"},
        {"rope_temporal", "64"},
        {"rope_height", "32"},
        {"rope_width", "32"},
        {"rope_base", "10000"},
        {"context", "512"},
        {"image_patch_side", std::to_string(patch_side_k)},
        {"image_channels", std::to_string(image_channels_k)},
        {"image_pixel_mean", "0.485,0.456,0.406,0"},
        {"image_pixel_std", "0.229,0.224,0.225,1"},
        {"video_patch_side", std::to_string(patch_side_k)},
        {"video_tube_frames", std::to_string(tube_frames_k)},
        {"video_frame_stride", "1"},
        {"video_channels", std::to_string(video_channels_k)},
        {"video_pixel_mean", "0.485,0.456,0.406"},
        {"video_pixel_std", "0.229,0.224,0.225"},
        {"audio_chunk_samples", std::to_string(chunk_samples_k)},
        {"audio_sample_rate", "16000"},
    };
    std::string json = "\"__metadata__\":{";
    for (auto const &entry : entries) {
        if (json.back() != '{') json += ',';
        json += "\"" + entry[0] + "\":\"" + entry[1] + "\"";
    }
    return json + '}';
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fputs("Usage: tiny_checkpoint <output.safetensors>\n", stderr);
        return 1;
    }
    std::vector<tensor_plan_t> const plans = plan_tensors();

    std::string header = "{" + metadata_json();
    std::uint64_t offset = 0;
    for (tensor_plan_t const &plan : plans) {
        std::uint64_t const end = offset + plan.rows * plan.columns * sizeof(std::uint16_t);
        header += ",\"" + plan.name + "\":{\"dtype\":\"BF16\",\"shape\":[" + std::to_string(plan.rows) + "," +
                  std::to_string(plan.columns) + "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(end) + "]}";
        offset = end;
    }
    header += '}';
    // Readers align the payload to 8 bytes, and the format pads the header with spaces to get there.
    while (header.size() % 8) header += ' ';

    std::vector<std::uint16_t> payload;
    payload.reserve(offset / sizeof(std::uint16_t));
    std::uint64_t state = 42;
    for (tensor_plan_t const &plan : plans)
        for (std::size_t scalar = 0; scalar != plan.rows * plan.columns; ++scalar) {
            if (plan.fill == tensor_plan_t::fill_t::ones_k) {
                payload.push_back(to_bf16(1.0f));
                continue;
            }
            state ^= state << 13, state ^= state >> 7, state ^= state << 17;
            float const unit = static_cast<float>(state >> 40) * (1.0f / 16777216.0f);
            payload.push_back(to_bf16((unit * 2.0f - 1.0f) * 0.02f));
        }

    std::FILE *const file = std::fopen(argv[1], "wb");
    if (!file) {
        std::fputs("tiny_checkpoint: the output could not be opened for writing\n", stderr);
        return 1;
    }
    std::uint8_t length_bytes[8];
    for (std::size_t byte = 0; byte != 8; ++byte)
        length_bytes[byte] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(header.size()) >> (byte * 8));
    bool const written = std::fwrite(length_bytes, 1, 8, file) == 8 &&
                         std::fwrite(header.data(), 1, header.size(), file) == header.size() &&
                         std::fwrite(payload.data(), sizeof(std::uint16_t), payload.size(), file) == payload.size();
    if (std::fclose(file) != 0 || !written) {
        std::fputs("tiny_checkpoint: the checkpoint could not be written\n", stderr);
        return 1;
    }
    fmt::println(stderr, "tiny_checkpoint: {} tensors, {} payload bytes -> {}", plans.size(), offset, argv[1]);
    return 0;
}
