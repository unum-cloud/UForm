/**
 *  @file src_stub/model.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief A stand-in for the @c uform_model_* C ABI: any readable file or buffer loads as a model of 64
 *      dimensions and a context of 512 tokens.
 *
 *  Nothing is parsed, so a binding can exercise loading, describing and freeing with any bytes it has at
 *  hand. The checkpoint is read in full and closed, as the engine reads it, and its length is what the
 *  model reports holding. Only the RAM verbs answer; the VRAM ones refuse as a build without GPUs does.
 */
#include <cstdio> // `std::fopen`, `std::fread`, `std::fclose`

#include <string> // `std::string`, the path with the terminator `fopen` needs

#include "lib.hpp"

using namespace unum::uform;

/** Refuses a quantization the engine does not run its blocks in. */
static bool reject_quantization(uform_scalar_t quantization, uform_outcome_t *outcome) {
    return reject_if(quantization != uform_scalar_bf16_k && quantization != uform_scalar_e4m3_k,
                     uform_status_unsupported_k, "Blocks run in bf16 or e4m3 only", outcome);
}

/** Reads the whole file at @p path, answering its length, or zero once @p outcome records why not. */
static std::size_t read_whole_file(uform_bytes_view_t path, uform_outcome_t *outcome) {
    std::string const terminated(as_text(path));
    std::FILE *const file = terminated.empty() ? nullptr : std::fopen(terminated.c_str(), "rb");
    if (reject_if(!file, uform_status_io_k, "The checkpoint could not be opened for reading", outcome)) return 0;
    std::byte chunk[4096];
    std::size_t length = 0;
    for (std::size_t read = sizeof(chunk); read == sizeof(chunk); length += read)
        read = std::fread(chunk, 1, sizeof(chunk), file);
    int const failed = std::ferror(file);
    std::fclose(file);
    if (reject_if(failed, uform_status_io_k, "The checkpoint could not be read", outcome)) return 0;
    reject_if(!length, uform_status_malformed_k, "The checkpoint is empty", outcome);
    return length;
}

/** Publishes a model of @p memory_usage bytes into @p output, once the handler, told of its one tensor, lets
 *  the load finish. */
static void publish_model(uform_model_t *output, uform_scalar_t quantization, std::size_t memory_usage,
                          uform_progress_t progress, uform_outcome_t *outcome) {
    if (reject_if(!progress_ticks_t {progress, 1}.carry_on(1), uform_status_cancelled_k,
                  "The caller's handler stopped the load", outcome))
        return;
    uform_model_handle_t *const model = new (std::nothrow) uform_model_handle_t {quantization, memory_usage};
    if (reject_if(!model, uform_status_out_of_memory_k, "The model could not be allocated", outcome)) return;
    *output = model;
}

/** Refuses a pin to any class but serial, the only one the stand-in runs as. */
static bool reject_pin(uform_cpu_class_t cpu_class, uform_outcome_t *outcome) {
    return reject_if(cpu_class != uform_cpu_class_unknown_k && cpu_class != uform_cpu_class_serial_k,
                     uform_status_unsupported_k, "The stand-in runs as the serial class alone", outcome);
}

extern "C" {

void uform_model_load_ram(uform_model_load_ram_t *arguments) {
    if (through_executor<uform_model_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->model, uform_status_invalid_argument_k, "The model output must not be null", outcome))
        return;
    *arguments->model = nullptr;
    if (reject_quantization(arguments->quantization, outcome) || reject_pin(arguments->ram.cpu_class, outcome)) return;
    if (reject_if(!arguments->path.data || !arguments->path.length, uform_status_invalid_argument_k,
                  "The path must not be empty", outcome))
        return;
    std::size_t const length = read_whole_file(arguments->path, outcome);
    if (!length) return;
    publish_model(arguments->model, arguments->quantization, length, arguments->progress, outcome);
}

void uform_model_load_ram_buffer(uform_model_load_ram_buffer_t *arguments) {
    if (through_executor<uform_model_load_ram_buffer>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->model, uform_status_invalid_argument_k, "The model output must not be null", outcome))
        return;
    *arguments->model = nullptr;
    if (reject_quantization(arguments->quantization, outcome) || reject_pin(arguments->ram.cpu_class, outcome)) return;
    if (reject_if(!arguments->buffer.data || !arguments->buffer.length, uform_status_invalid_argument_k,
                  "The checkpoint must not be empty", outcome))
        return;
    publish_model(arguments->model, arguments->quantization, arguments->buffer.length, arguments->progress, outcome);
}

void uform_model_load_vram(uform_model_load_vram_t *arguments) {
    if (through_executor<uform_model_load_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->model) *arguments->model = nullptr;
    propagate_error(uform_status_device_k, outcome, "This build carries no GPU backends");
}

void uform_model_load_vram_buffer(uform_model_load_vram_buffer_t *arguments) {
    if (through_executor<uform_model_load_vram_buffer>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->model) *arguments->model = nullptr;
    propagate_error(uform_status_device_k, outcome, "This build carries no GPU backends");
}

void uform_model_describe(uform_model_describe_t *arguments) {
    if (through_executor<uform_model_describe>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->model || !arguments->description, uform_status_invalid_argument_k,
                  "The model and the description must not be null", outcome))
        return;
    uform_model_handle_t const &model = *arguments->model;
    *arguments->description = {dimensions_k,
                               context_k,
                               model.quantization,
                               model.memory_usage,
                               "serial",
                               uform_cpu_class_serial_k,
                               uform_gpu_class_unknown_k};
}

void uform_model_free(uform_model_t model) {
    wait_for_calls_using(model);
    delete model;
}

} // extern "C"
