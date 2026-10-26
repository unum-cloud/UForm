# UForm for Rust

The crate links the precompiled library and calls the stable C ABI in `c/uform/` through `extern "C"` declarations, so no tower is compiled in your build.
It needs Rust 1.100 or newer, CMake 3.27 and a C++23 compiler, Clang by default.
Every verb is a batch, and every output whose length the call decides lands in an arena you reuse.

- `Model` encodes text, images, video and audio into pooled embeddings, per-token states and token vectors, and scores calibrated choices over text.
- `Trainer` trains one on GPUs, joined across hosts by a `Mesh`.
- `Executor` owns the threads or streams every call runs on, apart from any model.

## Installation

```sh
cargo add uform
```

CMake builds the library from the crate's sources into Cargo's output directory and links it as a static archive, along with the system image, video and audio codecs.
Three features swap what is linked, and `build.rs` refuses the stub beside either GPU feature:

- `cuda` adds the NVIDIA residency, the trainer and the mesh, and needs a CUDA toolkit at build time.
- `rocm` adds the AMD residency, and needs ROCm at build time.
- `stub` links a stand-in that answers the same verbs over the C++ standard library alone, with deterministic rows, seconds to build and no codecs, for testing a caller and never for shipping.

CMake builds the NumKong, StringZilla, ForkUnion and SmashTable trees of the crates Cargo resolved, rather than fetching their pinned revisions again.
A WASI build reads the SDK from `WASI_SDK_PATH` and needs `-C target-feature=+simd128`.
The suites run against the stub:

```sh
CC=clang-22 CXX=clang++-22 cargo test --features stub
```

## Quickstart

```rust
use uform::*;

let executor = Executor::for_ram(&[], None, 0, 0)?;           // the threads every receiver names
let mut arena = Arena::new_in(&executor)?;                    // the memory outputs land in
let model = Model::load_ram("model.safetensors")?;            // BFloat16 blocks, as the checkpoint stores them

let texts = ["a cat on a mat", "a dog in the fog"];
let encoded = model.on(&executor).text_encode(&texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
if let Some(embeddings) = encoded.embeddings {
    let values: &[f32] = embeddings.values()?;                // count × columns, zero-copy from the arena
    assert_eq!(values.len(), texts.len() * model.describe()?.dimensions);
}
```

## Receivers & Deferred Calls

The receiver decides where a verb runs and how it completes, while the verb and its arguments stay the same:

- `model.text_encode(..)` runs on the calling thread, through `&Model`.
- `model.on(&executor).text_encode(..)` blocks on the executor, after the calls submitted there before it.
- `model.deferred(&executor).text_encode(..)` submits the call and answers a `Deferred`.

`ModelVerbs` and `EncodeVerbs` write every verb once for all three, and `TrainVerbs` does so for a trainer's two receivers.
A blocking verb returns its `Result` directly, so nothing about asynchrony reaches a caller who never asks for it.
An `On` borrows the model and the executor, so neither it nor a `Deferred` it submits can outlive them.

```rust
let texts = vec![String::from("a cat on a mat"), String::from("a dog in the fog")];
let call = model.deferred(&executor).text_encode(texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
if call.status() == Err(Status::Deferred) {
    // other work
}
let encoded = call.await?;                                    // or: call.wait()?
```

A `Deferred` owns the inputs the library reads until the call is final, and borrows the model and the arena, as a blocking call does.
`status` answers `Ok(())` once the call succeeded, `Err(Status::Deferred)` while it is not final, and its failure otherwise.
Inside an `async` block, `.await` polls the call's status until it is final, so other tasks run meanwhile, while `wait` blocks the thread without spinning.
The crate starts no thread, so waiting off-thread goes through a scoped thread, as `std::thread::scope(|scope| scope.spawn(move || call.wait()).join())` does.
Dropping an unfinished `Deferred` waits for the call, and a panic a progress handler raised resurfaces from `wait`, `.await`, or a drop outside unwinding.

The library reads a deferred call's inputs after the verb returns, so a deferred receiver takes owned inputs: items in a `Vec` or an array of owned or `'static` values, a tape as `(Vec<_>, Vec<usize>)`, a `PathBuf`, `String` or `'static` path, and `'static` configs and handlers.
Lending `&Vec<String>` fails to compile with "a deferred receiver needs owned inputs", and a config borrowing a local fails as "does not live long enough".
A deferred call allocates one box holding its outcome, argument struct and inputs until it is final.
An executor without threads of its own may hold a deferred call until its `wait`, its next `drain` or blocking call, or the drop of what it names, so `drain` it before an `.await` there.
A submission finding the executor's deferred table full finishes at once with `Status::Contended`.

## Executors

An executor owns host threads or device streams, the activation scratch, and the pinned buffers staging passes through, grown to a high-water mark.
One executor serves any number of models of its residency.

```rust
let threads = Executor::for_ram(&[], None, 8, 0)?;                      // eight threads, where zero takes 7/8 of the cores
let domains = Executor::for_ram(&[0, 1], None, 0, 0)?;                  // threads interleaved across two NUMA nodes
let serial = Executor::for_ram(&[], Some(CPUClass::Serial), 0, 0)?;     // dispatches as the portable floor
let streams = unsafe { Executor::for_vram(&[0, 1], None, &[], 0, 0)? }; // one stream per device, scratch grown as asked
let shared = unsafe { Executor::on_pool(&[], None, pool, 0)? };         // borrows a ForkUnion pool's threads
let local = Executor::default();                                        // owns nothing, runs on the calling thread

threads.drain(Mood::Spinning)?;                                         // waits for every call, then keeps spinning
```

The second argument pins a `CPUClass` or a `GPUClass` to dispatch as, and `None` derives the hardware's own.
`gpu_devices(&mut devices)` fills a slice of `GPUDevice` and answers how many GPUs the process can drive, and a device's position there is the ordinal `for_vram` and `load_vram` name it by.
`for_vram` and `on_pool` are `unsafe`, since the streams or the pool they borrow must outlive the executor; `on_pool` mirrors the C verb and answers `Status::Unsupported` until borrowing is designed.
Each constructor ends with a `deferred_limit` sizing its table of deferred calls, where zero takes the default.
Calls on one executor run in submission order, so two threads that genuinely overlap take one apiece; an executor is `Send` but not `Sync`, while models stay freely shared.
Dropping an executor drains it, unless a GPU arena it made still holds it, which then does so on its own drop.

## Arenas

Every output whose length only the call knows, the embeddings, the states and their offsets, lands in an `Arena` you pass, and the verb answers views over it without copying.
`Arena::new_in(&executor)` places it where that executor computes, decided once, and `Arena::new()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
Every call, blocking or deferred, borrows the arena as `&mut Arena` and its result borrows it in turn, so the borrow checker refuses a stale view at compile time.
To keep a result past the next call, pass that call another arena; an arena serves one call at a time, so each deferred call needs its own.
An arena from a VRAM executor holds device memory, which host accessors refuse with `Status::Device`, while `as_ptr` still hands the rows to another library, such as a USearch VRAM index.

```rust
let mut arena = Arena::new_in(&executor)?;
for batch in &batches {
    let encoded = model.on(&executor).text_encode(batch, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
    consume(encoded.embeddings.map(|rows| rows.values::<f32>()).transpose()?); // rewinds next iteration
}
```

The library never frees an arena under a call: dropping one waits for the calls writing into it, and a call naming an arena that serves another call finishes with `Status::Contended`, so even a leaked `Deferred` leaves it safe.

## Encoding

One verb per input form, each spelled as in C, and each taking its inputs, the `Outputs` to keep, a config, and last the arena:

- `text_encode(texts, ..)` takes UTF-8 texts, a byte per token, since models are tokenizer-free.
- `image_encode(files, ..)` takes JPEG, PNG or WebP files.
- `image_encode_pixels(pixels, heights, widths, ..)` takes interleaved 8-bit RGB, `heights[i]` rows of `widths[i]` pixels per item.
- `video_encode(files, ..)` takes MP4 or WebM clips.
- `audio_encode(files, ..)` takes WAV, FLAC, MP3 or Opus recordings.
- `audio_encode_samples(samples, sample_rate, ..)` takes mono `f32` samples.

A batch is a slice, array or `Vec` of separate items, `&str`, `String`, `&[u8]` or `Vec<u8>`, or `&[f32]` and `Vec<f32>` for samples, gathered into one tape with one copy.
A `(data, offsets)` pair, the data and `count + 1` offsets, is lent to the library as a tape without a copy, once the binding checks the offsets stay inside the data:

```rust
let tape = (b"first documentthe second".as_slice(), [0usize, 14, 24]);
let encoded = model.text_encode(tape, Outputs::STATES, Scalar::BF16, &mut arena)?;
let states = encoded.states.expect("STATES were asked for");
let offsets = states.offsets()?;                              // item `i` spans rows `offsets[i]` up to `offsets[i + 1]`
let second = states.item(1)?;                                 // the rows of one item, as `Rows`
```

`Outputs::EMBEDDINGS` is the default, `|` joins it with `Outputs::STATES` and `Outputs::TOKEN_VECTORS`, and a surface left out answers `None` and costs nothing past the final norm.
Token vectors are the token head's unit rows, one per state, which late-interaction rerankers score; a model without the head answers `Status::Unsupported`.
Every bare `Scalar` converts into the verb's config, the scalar every output lands as; the config's other fields reach the knobs:

```rust
let mut stats = TextEncodeStats::default();
let mut handler = |completed, total| if completed * 2 < total { ControlFlow::Continue(()) } else { ControlFlow::Break(()) };
let config = TextEncodeConfig { stats: Some(&mut stats), progress: Some(&mut handler), ..TextEncodeConfig::from(Scalar::F32) };
let encoded = model.text_encode(&texts, Outputs::EMBEDDINGS, config, &mut arena);
```

`Rows` reports its `count`, `columns`, `stride` and `scalar`, and reads as bytes through `as_bytes` or as values through `values::<f32>()`, refused when the rows hold another scalar type.
`RowsTape` adds the `offsets` and an `item` per input.
A handler answering `ControlFlow::Break` stops the call, which fails with `Status::Cancelled` and publishes nothing.

`text_choose(texts, choices, ..)` scores each item's legal choices, one byte each, through one forward, and answers a `Tape` of one `f32` probability per choice that sums to one per item:

```rust
let config = TextChooseConfig { temperature_groups: &[0, 1], ..TextChooseConfig::default() };
let probabilities = model.text_choose(&["keep this candidate?", "expand which neighbour?"], &["yn", "ABCD"], config, &mut arena)?;
let expand: &[f32] = probabilities.item(1)?;                  // the probabilities of 'A' through 'D'
```

Each group is a question type below the model's `choice_temperature_groups`, and a model without the byte decoder answers `Status::Unsupported`.

## Models

`Model::load_ram(path)` and `Model::load_vram(path)` copy a safetensors checkpoint into host memory or onto GPUs, and `load_ram_buffer` and `load_vram_buffer` take the checkpoint's bytes instead.
A bare path or buffer loads on the calling thread with `Scalar::BF16` blocks, and its config reaches the executor, the placement, `Scalar::E4M3` quantization and a progress handler:

```rust
let model = Model::load_vram(ModelLoadVramConfig {
    executor: Some(&executor),
    devices: &[0],
    quantization: Scalar::E4M3,
    ..ModelLoadVramConfig::from("model.safetensors")
})?;
let description = model.describe()?;                          // geometry, placement, classes and heads
```

A description reports the `cpu_class` or `gpu_class` the model dispatched as, and the heads the checkpoint carries: a nonzero `token_width` for token vectors, and a nonzero `choice_temperature_groups` for choices.
`Model` carries no type parameter: quantization is the runtime `Scalar` passed at load, as in C.

## Training

Training is GPU-only.
The configuration arrives as structs mirroring the C ones field by field, so a field left at its default is zero, which the library derives:

```rust
let executor = unsafe { Executor::for_vram(&[0, 1], None, &[], 0, 0)? };
let mesh = Mesh::bootstrap(&executor, &MeshOptions {
    host_domain_index: rank,
    host_domain_count: nodes,
    directory: Path::new("/shared/exchange"),
    job: "run-17",
    ..MeshOptions::default()
})?;
let trainer = Trainer::init_vram(&executor, TrainerInitVramConfig {
    mesh: Some(&mesh),
    depth: 12,
    hidden: 512,
    objective: Objective::Joint,
    batch: BatchShape { max_rows_per_device: 64, max_segments_per_device: 256, max_columns_per_row: 1024,
                        max_bags_per_device: 128, image: ImageShape { patch_side: 16, channels: 0 },
                        ..BatchShape::default() },
    schedule: Schedule { steps: 100_000, ..Schedule::default() },
    seed: 42,
    ..TrainerInitVramConfig::default()
})?;

let mut stats = TrainerStepStats::default();
let relations = Relations { attract: Edges { offsets: &[0, 1, 2, 3, 4], ids: &[1, 0, 3, 2], weights: &[] },
                            ..Relations::default() };
let config = TrainerStepConfig { bags: &[0, 0, 1, 1], relations, stats: Some(&mut stats), ..TrainerStepConfig::default() };
trainer.on(&executor).step(&modalities, &contents, config)?;
trainer.on(&executor).save("checkpoint.safetensors")?;
```

`BatchShape` nests an `ImageShape`, a `VideoShape` and an `AudioShape`, while `Masking`, `Contrastive`, `LateInteraction`, `Choices` and `Schedule` configure the objectives, and `PairLossForm` picks each contrastive term's pair loss.
`Relations` holds three `Edges` sets as compressed sparse rows, which the binding checks against the step's items before the call.
A step's `sides` give each item a `LateInteractionSide`, and its `ChoicesView` gives each choice item its legal bytes and their target probabilities, checked the same way.
`step` answers the pooled embeddings the loss scored when its config names an arena, and `None` otherwise.
A trainer borrows its mesh, so the mesh outlives every trainer made over it, and `mesh.describe()` answers a `MeshDescription` whose `multimem_tier` is a `MultimemTier` and whose `gpu_class` is the `GPUClass` every GPU of the run dispatches as.

## Errors

Every fallible call returns `uform::Result`, whose `Error` carries a `Status` and a static `message`, so nothing allocates on the failure path.
A failure the binding catches before the call crosses the boundary reports the status the library would have used.

```rust
match model.on(&executor).text_encode(&texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena) {
    Ok(encoded) => println!("{} items", encoded.embeddings.map_or(0, |rows| rows.count())),
    Err(Error { status: Status::Contended, .. }) => println!("the arena serves another call"),
    Err(error) => println!("{}: {}", error.status, error.message),
}
```

`Status` carries the C enumerators' values and leaves success to `Result`, so an `Error` never carries `Status::Deferred`, which only `Deferred::status` reports.
`Status`, `Scalar`, `Mood`, `CPUClass`, `GPUClass`, `Modality`, `Objective` and `Negatives` print through `Display` and parse through `FromStr` the names the C tables own, such as `"shape-mismatch"` or `"intel-haswell"`, and `MultimemTier` prints its own, which C never reads back.
A panic inside a progress handler is caught at the C frame, since unwinding out of one is an abort, and resurfaces once the call is final.

## Design

- __A thin layer over the C ABI.__
  One private module per header, `types`, `model`, `encode` and `train`, re-exported flat; every public name is its C name without `uform_` and `_t` in CamelCase, a verb's argument struct becomes its `*Config`, and every wrapper's documentation names the C symbol it calls.
  Enums mirror the C enums through the C name functions, never strings.
- __The receiver chooses, the verb stays.__
  `&Model` runs a verb on the calling thread, `on(&executor)` blocks on an executor, and `deferred(&executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `uform_outcome_status`, `wait` blocks through `uform_outcome_wait`, and a `Deferred` is `IntoFuture`, so `.await` polls the status word, since the library has no wake-up.
- __No life of its own.__
  The binding starts no threads and keeps no queues or caches; releasing a `Deferred` rides `Drop`, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output verb takes an arena and answers views borrowing it, so the borrow checker retires stale views at compile time.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table entry and allocate no box, a surface `Outputs` leaves out is never computed, and an arena allocates only when a call writes into it.
- __A deferred receiver refuses borrowed inputs at compile time.__
  The `Accepts` bound admits only owned inputs, so a borrow the library would read after the verb returns never compiles.
- __Suites run against the stub library__, and against the real one with the tiny checkpoint.
