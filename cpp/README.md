# UForm for C++

The C++ SDK is a header-only layer over the stable C ABI in `c/uform/`, so it compiles no tower and links the same library C does.
It needs a C++20 compiler and CMake 3.28, and throws nothing: every call answers `auto [value, status]`, and every handle is an RAII owner.
Every verb is a batch, and every output whose length the call decides lands in an arena you reuse.

- `model_t` encodes text, images, video and audio into pooled embeddings, per-token states and token vectors, and scores calibrated choices over text.
- `trainer_t` trains one on GPUs, joined across hosts by a `mesh_t`.
- `executor_t` owns the threads or streams every call runs on, apart from any model.

## Installation

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D UFORM_BUILD_SHARED=ON && cmake --build build_release && cmake --install build_release
```

The install carries the C and C++ headers, both libraries, and a CMake package whose `uform::cpp` target names the SDK headers, the C headers of `uform::abi` beneath them, and the archive `uform::static`:

```cmake
find_package(uform CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE uform::cpp)
```

Inside a superbuild, `add_subdirectory` of the repository gives the same `uform::cpp`.
`uform::cpp` reaches only `c/uform/*.h` and the standard library, so the compiler refuses any SDK include of the engine.
`-D UFORM_BUILD_STUB=ON -D UFORM_BUILD_TEST=ON` builds a stand-in that answers every verb over the standard library alone, and the suite against it:

```sh
cmake --preset stub_library -D UFORM_BUILD_TEST=ON && cmake --build --preset stub_library --target uform_test_cpp_stub
ctest --test-dir build_stub_library --tests-regex uform_test_cpp_stub
```

## Quickstart

Encode two captions and a photo into one space, then pick the caption closest to the photo:

```cpp
#include <cmath>    // `std::sqrt`
#include <cstdio>   // `std::printf`
#include <fstream>  // `std::ifstream`
#include <iterator> // `std::istreambuf_iterator`
#include <vector>   // `std::vector`

#include <uform/uform.hpp>

namespace uf = unum::uform;

int main() {
    auto [executor, executor_status] = uf::executor_t::for_ram(); // the threads `on(executor)` names
    auto [texts_arena, texts_arena_status] = uf::arena_t::make(); // host pages for the calling thread
    auto [images_arena, images_arena_status] = uf::arena_t::make({.executor = executor.handle()}); // a call rewinds its arena
    auto [model, model_status] = uf::model_t::load_ram( // BFloat16 blocks, as the checkpoint stores them
        {.path = uf::bytes_view("model.safetensors"), .quantization = uform_scalar_bf16_k});
    if (model_status.failed()) return 1;

    std::string_view const captions[] = {"a cat on a mat", "a sunrise over the bay"};
    std::ifstream file("cat.png", std::ios::binary);
    std::vector<std::uint8_t> const photo {std::istreambuf_iterator<char>(file), {}};
    std::size_t const photo_offsets[] = {0, photo.size()};
    std::size_t offsets[3];
    char elements[64];
    auto [captions_tape, gather_status] = uf::gathered<char>(captions, offsets, elements); // one copy, end to end
    auto [texts, texts_status] = model.text_encode(captions_tape, uf::encode_outputs_t::embeddings_k, // the calling thread
                                                   uform_scalar_f32_k, texts_arena);
    auto [images, images_status] = model.on(executor).image_encode({photo, photo_offsets}, uf::encode_outputs_t::embeddings_k,
                                                                   uform_scalar_f32_k, images_arena); // blocks on the executor
    if (texts_status.failed() || images_status.failed()) return 1;

    std::span<float const> const image = images.embeddings.row<float>(0); // zero-copy from the arena; embeddings come unnormalized
    auto cosine = [&](std::span<float const> text) {
        float dot = 0, text_norm = 0, image_norm = 0;
        for (std::size_t dimension = 0; dimension != text.size(); ++dimension) {
            dot += text[dimension] * image[dimension];
            text_norm += text[dimension] * text[dimension];
            image_norm += image[dimension] * image[dimension];
        }
        return dot / std::sqrt(text_norm * image_norm);
    };
    std::size_t const best = cosine(texts.embeddings.row<float>(1)) > cosine(texts.embeddings.row<float>(0));
    std::printf("'%.*s' describes the photo best\n", int(captions[best].size()), captions[best].data());
    return 0;
}
```

## Receivers & Deferred Calls

The receiver decides where a verb runs and how it completes, while the verb and its arguments stay the same:

- `model.text_encode(..)` runs on the calling thread.
- `model.on(executor).text_encode(..)` blocks on the executor, failing `status_code_t::contended_k` while another call holds it.
- `model.deferred(executor).text_encode(..)` submits the call and answers a `deferred`.

`model<receiver>` writes every verb once for all three: `model_t` owns the model and loads it, while `on` and `deferred` answer a `model_on<completion>` that refers to it and the executor, and `trainer<receiver>` does the same for a trainer.
A blocking verb answers a `status_t` when it outputs nothing else and an `expected<T>` otherwise, so nothing about asynchrony reaches a caller who never asks for it.
Each class spells that choice once, as its `deferrable<T>`, which names a `deferred<T>` under a deferred receiver.

```cpp
uf::deferred<uf::encoded_t> call =
    model.deferred(executor).text_encode(texts, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k, arena); // running now
other_work();
auto [encoded, status] = call.wait(); // blocks until final; destruction waits too
```

Any coroutine type awaits a deferred call, with any loop offering `post(void (*)(void*) noexcept, void*)`, so one model encodes on two executors at once:

```cpp
app::task encode_both(uf::model_t const& model, uf::executor_t const& east_executor, uf::executor_t const& west_executor,
                      uf::tape_view<char> texts, app::loop& loop) {
    auto [east_arena, east_arena_status] = uf::arena_t::make({.executor = east_executor.handle()});
    auto [west_arena, west_arena_status] = uf::arena_t::make({.executor = west_executor.handle()});
    uf::deferred<uf::encoded_t> east_call = model.deferred(east_executor)
                                                .text_encode(texts, uf::encode_outputs_t::embeddings_k,
                                                             uform_scalar_f32_k, east_arena);
    uf::deferred<uf::encoded_t> west_call = model.deferred(west_executor)
                                                .text_encode(texts, uf::encode_outputs_t::embeddings_k,
                                                             uform_scalar_f32_k, west_arena);
    auto [east_encoded, east_status] = co_await east_call.wait(loop); // both calls were already running
    auto [west_encoded, west_status] = co_await west_call.wait(loop);
    if (east_status.failed() || west_status.failed()) co_return;
    merge(east_encoded, west_encoded);
}
```

A `deferred` never moves, since the library writes into it, so it lives on the caller's stack or in the awaiting coroutine's frame.
`status` answers `status_code_t::deferred_k` until the call is final, then its final status, and `wait()` blocks until then and answers what a blocking call would have.
`co_await call.wait(loop)` suspends the coroutine and re-polls the status through `loop`, any type with `post(void (*)(void*) noexcept, void*)`, so the awaiter posts two words and allocates nothing, as Rust's waker re-wakes itself.
The library has no wake-up, so a scheduler runs other posted work between polls, and its own coroutine type owns the frame.
Destroying an unfinished `deferred` waits for the call.

The library reads a deferred call's inputs after the verb returns, so the tapes, spans, handlers and reports a deferred verb names must outlive the call, which C++ leaves to the caller.
A deferred call allocates nothing: its outcome and outputs live inside the `deferred` itself.
A refusal the SDK catches before the call crosses answers a `deferred` that is final from the start.
An executor without threads of its own may run a deferred call at submission, final when the verb returns.

## Executors

An executor owns host threads or device streams, the activation scratch, and the pinned buffers staging passes through, grown to a high-water mark.
One executor serves any number of models of its residency.

```cpp
auto [threads, threads_status] = uf::executor_t::for_ram({.threads = 8});         // eight threads, where zero takes 7/8 of the cores less one
std::int32_t const domains[] = {1};
auto [numa, numa_status] = uf::executor_t::for_ram({.ram = {.memory_domains = domains, .memory_domains_count = 1}});
auto [serial, serial_status] = uf::executor_t::for_ram({.ram = {.cpu_class_ceiling = uform_cpu_class_serial_k}});
std::int32_t const devices[] = {0, 1};
auto [streams, streams_status] = uf::executor_t::for_vram({.vram = {.devices = devices, .devices_count = 2}});
uf::executor_t local;                                                   // owns nothing, runs on the calling thread

uf::status_t const drained = threads.drain(uform_mood_spinning_k);     // waits until no call holds it, then keeps spinning
```

Every factory takes the C argument struct by designator, so a field left out is zero, which the library derives.
Naming one memory domain spawns the threads near it, while none or several interleave them across the machine.
`for_vram` opens one stream per device, or adopts the caller's `streams`, one per named device, and the executor destroys them; naming no devices drives every listed device of the first vendor with a known class.
`uf::gpu_devices(devices)` fills a span of `uform_gpu_device_t` and answers how many GPUs the process can drive, and a device's position there is the ordinal `for_vram` and `load_vram` name it by, while `uf::version_name()` spells the linked library's version.
An executor runs one call at a time, so two threads that genuinely overlap take one apiece.
Destroying an executor waits until no call holds it.

## Arenas

Every output whose length only the call knows, the embeddings, the states and their offsets, lands in an `arena_t` you pass, and the verb answers views over it without copying.
`arena_t::make({.executor = executor.handle()})` places it where that executor computes, decided once, and `arena_t::make()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, pass that call another arena; an arena serves one call at a time, so each deferred call needs its own.
An arena from a VRAM executor holds device memory, which `describe()` reports as `uform_residency_vram_k` and host code must not read, while `view.data` still hands the rows to another library, such as a USearch VRAM index.

```cpp
auto [arena, arena_status] = uf::arena_t::make({.executor = executor.handle()});
for (uf::tape_view<char> const batch : batches) {
    auto [encoded, status] = model.on(executor).text_encode(batch, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k, arena);
    consume(encoded.embeddings); // rewinds next iteration
}
```

Destroying an arena waits for the calls writing into it, and a call naming an arena that serves another call finishes with `status_code_t::contended_k`.
An arena never outlives the executor that made it, so declare the executor before the arena and scope exit destroys the arena first, and an arena from `arena_t::make()` lives until it is destroyed.

## Encoding

One verb per input form, each spelled as in C, and each taking its inputs, the `encode_outputs_t` to keep, a config, and last the arena:

- `text_encode(texts, ..)` takes UTF-8 texts as a `tape_view<char>`, a byte per token, since models are tokenizer-free.
- `image_encode(files, ..)` takes JPEG, PNG or WebP files as a `tape_view<std::uint8_t>`.
- `image_encode_pixels(pixels, heights, widths, ..)` takes interleaved 8-bit RGB, `heights[i]` rows of `widths[i]` pixels per item.
- `video_encode(files, ..)` takes MP4 or WebM clips.
- `audio_encode(files, ..)` takes WAV, FLAC, MP3 or Opus recordings.
- `audio_encode_samples(samples, sample_rate, ..)` takes mono `float` samples as a `tape_view<float>`.

A `tape_view` is the elements and `count + 1` offsets, which the library reads without a copy once the SDK checks the offsets stay inside the elements.
Separate items gather into memory the caller owns with one copy, through `gathered<char>(items, offsets, elements)`, which answers the view every verb takes, or `status_code_t::invalid_argument_k` when either span is too short:

```cpp
std::string_view const joined = "first documentthe second";
std::size_t const offsets[] = {0, 14, 24};
auto [encoded, status] = model.text_encode({joined, offsets}, uf::encode_outputs_t::states_k, uform_scalar_bf16_k, arena);
std::span<std::size_t const> const items = encoded.states.offsets(); // item `i` spans rows `items[i]` up to `items[i + 1]`
uf::written_rows_t const second = encoded.states.item(1);              // the rows of one item
```

`encode_outputs_t::embeddings_k`, `states_k` and `token_vectors_k` join with `|`, and a surface left out answers an empty view and costs nothing past the final norm.
Token vectors are the token head's unit rows, one per state, which late-interaction rerankers score; a model without the head answers `status_code_t::unsupported_k`.
A bare `uform_scalar_t` converts into the verb's config, the scalar every output lands as; the config's other fields reach the knobs:

```cpp
uform_text_encode_stats_t stats {};
auto handler = [](std::size_t completed, std::size_t total) noexcept {
    return completed * 2 < total ? uform_continue_k : uform_cancel_k;
};
uf::text_encode_config_t const config {uform_scalar_f32_k, {handler}, &stats};
auto [encoded, status] = model.text_encode(texts, uf::encode_outputs_t::embeddings_k, config, arena);
```

`written_rows_t` holds the C `view`, its `stride`, `columns` and `scalar`, beside the rows `written`, and `row<float>(i)` reads one row, empty when the rows hold another scalar type.
`written_rows_tape_t` adds the `offsets()` and an `item(i)` per input.
A handler answering `uform_cancel_k` stops the call, which fails with `status_code_t::cancelled_k` and publishes nothing.

## Choices

`text_choose(texts, choices, ..)` scores each item's legal choices, one byte each, through one forward, and answers a `tape_view<float>` of one probability per choice that sums to one per item:

```cpp
std::uint32_t const groups[] = {0, 1};
auto [probabilities, status] = model.text_choose(states, choices, {.temperature_groups = groups}, arena);
std::span<float const> const expand = probabilities.entry(1); // the probabilities of 'A' through 'D'
```

Each group is a question type below the model's `choice_temperature_groups`, and a model without the byte decoder answers `status_code_t::unsupported_k`.

## Models

`model_t::load_ram` and `model_t::load_vram` copy a safetensors checkpoint into host memory or onto GPUs, and `load_ram_buffer` and `load_vram_buffer` take the checkpoint's bytes instead.
Each takes the C argument struct, which reaches the executor, the placement and its class ceiling, the quantization and a progress handler, and always loads blocking:

```cpp
std::int32_t const device = 0;
auto [model, model_status] = uf::model_t::load_vram({
    .executor = executor.handle(),
    .path = uf::bytes_view("model.safetensors"),
    .vram = {.devices = &device, .devices_count = 1},
    .quantization = uform_scalar_e4m3_k,
});
auto [description, describe_status] = model.on(executor).describe(); // geometry, placement, classes and heads
```

A description reports the `cpu_class` or `gpu_class` the model dispatched as, and the heads the checkpoint carries: a nonzero `token_width` for token vectors, and a nonzero `choice_temperature_groups` for choices.
Blocks run in `uform_scalar_bf16_k`, as the checkpoint stores them, or `uform_scalar_e4m3_k` quantized at load, and a zero quantization is refused.

## Training

Training is GPU-only.
The configuration arrives as the C structs themselves, so a field left out is zero, which the library derives:

```cpp
auto [executor, executor_status] = uf::executor_t::for_vram({.vram = {.devices = devices, .devices_count = 2}});
auto [mesh, mesh_status] = uf::mesh_t::bootstrap({
    .executor = executor.handle(),
    .options = {.host_domain_index = rank, .host_domain_count = nodes,
                .directory = uf::bytes_view("/shared/exchange"), .job = uf::bytes_view("run-17")},
});
auto [trainer, trainer_status] = uf::trainer_t::init_vram({
    .executor = executor.handle(),
    .depth = 12,
    .hidden = 512,
    .batch = {.max_rows_per_device = 64, .max_segments_per_device = 256, .max_columns_per_row = 1024,
              .max_bags_per_device = 128, .image = {.patch_side = 16}},
    .objective = uform_objective_joint_k,
    .schedule = {.steps = 100000},
    .mesh = mesh.handle(),
    .seed = 42,
});

std::size_t const attract_offsets[] = {0, 1, 2, 3, 4};
std::uint32_t const attract_ids[] = {1, 0, 3, 2}, bags[] = {0, 0, 1, 1};
uform_trainer_step_stats_t stats {};
uf::trainer_step_config_t const config {
    .bags = bags, .relations = {.attract = {attract_offsets, attract_ids, nullptr}}, .stats = &stats};
auto [embeddings, step_status] = trainer.on(executor).step(modalities, contents, config);
uf::status_t const saved = trainer.on(executor).save(uf::bytes_view("checkpoint.safetensors"));
```

`trainer_step_config_t` gathers a step's per-item inputs: `bags`, `temperature_groups`, `sides`, the `choices` and `choice_targets` tapes `text_choose` takes and answers, the `relations` edge sets, and an optional `arena` for the pooled embeddings the loss scored.
The SDK checks every per-item span and tape against the step's items before the call, and leaves the edge sets to the library.
`step` answers the pooled embeddings when its config names an arena, and an empty `written_rows_t` otherwise.
A trainer refers to its mesh, so the mesh outlives every trainer made over it, and `mesh.describe()` answers a `uform_mesh_description_t`.

## Errors

Every fallible call answers a `status_t` alone, or an `expected<T>`, a plain pair unpacked as `auto [value, status]`, whose value is default-constructed on failure.
A `status_t` carries a `status_code_t` and an optional static `note`, so nothing allocates on the failure path.
Recovery branches on the code, while the note names the broken invariant, step or resource for logs, falling back to the status name.
A failure the SDK catches before the call crosses the boundary reports the status the library would have used.

```cpp
auto [encoded, status] = model.on(executor).text_encode(texts, uf::encode_outputs_t::embeddings_k, uform_scalar_f32_k, arena);
if (status == uf::status_code_t::contended_k) retry();
else if (status.failed()) fmt::println(stderr, "{}", status.note);
else fmt::println("{} items", encoded.embeddings.written);

```

`status_code_t` carries the C enumerators' values, and `status_code_t::deferred_k` reaches only `deferred::status`.
The C name tables, such as `uform_status_name` and `uform_scalar_from_name`, are called directly, since every SDK header includes its C header.
A handler is a `noexcept` callable, so no fault unwinds through the library.

## Design

- __A thin layer over the C ABI.__
  One header per C header, `types.hpp`, `model.hpp`, `encode.hpp` and `train.hpp`, behind `uform.hpp`; every name is its C name without `uform_`, and every factory takes its C argument struct by designator.
- __The receiver chooses, the verb stays.__
  The model runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `uform_outcome_status`, `wait()` blocks through `uform_outcome_wait`, and `co_await wait(loop)` re-polls through the caller's scheduler, since the library has no wake-up.
- __No life of its own.__
  The SDK starts no threads and keeps no queues, counts or caches; destroying a `deferred` waits for the call first.
- __Arenas belong to the caller and are reused, never shared.__
  Every output verb takes an arena and answers views into it, valid until the arena's next call.
- __Nothing costs anything until it is used.__
  A blocking call keeps its outcome and outputs on the stack, a deferred one inside the `deferred` it answers, and awaiting posts two words to the caller's loop, so the SDK allocates nothing on any path.
- __Suites run against the stub library__, and against the real one with the tiny checkpoint.

| Path | C++ | Rust |
|---|---|---|
| Blocking call | 0 | 0 |
| Refusal before the call | 0 | 1 error |
| Deferred submit and wait | 0: the `deferred` lives with the caller | 0: the `Deferred` is pinned where it is polled |
| Each callback | 0 | 0 |
