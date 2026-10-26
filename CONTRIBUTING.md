# Contributing to UForm

UForm 4 is a training and inference engine behind one C ABI, with bindings for Python, Rust, JavaScript and Swift, and Python data tooling around it.

## Building

The prerequisites are listed at the top of `CMakeLists.txt`.
Every preset in `CMakePresets.json` builds into `build_<preset>`, with Clang compiling the host code and hosting `nvcc`:

```sh
cmake --preset cpu_library -D UFORM_BUILD_TEST=ON   # the C library for the host, and its suites
cmake --build --preset cpu_library
ctest --test-dir build_cpu_library
```

- `cpu_library` builds the engine for the host alone, and `stub_library` the stand-in in `src_stub/`.
- `nvidia_amd_library` adds both GPU residencies, while `debug`, `check`, `relwithdebinfo` and `release` also build the trainer and its suites.
- `wasm32_emscripten_release` and `wasm32_wasi_release` build for WebAssembly, with Node or Wasmtime running the tests.
- `-D UFORM_BUILD_BENCH=ON` adds the benchmarks [`bench/README.md`](bench/README.md) describes.

Configuring CMake also arms the hooks in `.githooks/`; to arm them without a build, run `git config core.hooksPath .githooks`.

## Layout and Namespaces

| Path                     | Contents                                                      | Namespace              |
| :----------------------- | :------------------------------------------------------------ | :--------------------- |
| `c/uform/*.h`            | the C ABI, `uform.h` over the four headers beside it          | none                   |
| `include/uform/core/`    | types, batches, device memory, cells, the mesh and transports | `unum::uform::core`    |
| `include/uform/encoder/` | the encoder towers and kits, their geometry and entries       | `unum::uform::encoder` |
| `include/uform/media/`   | image, video and audio decoding on the host                   | `unum::uform::media`   |
| `include/uform/loaders/` | storage sessions, shard readers, GPU media decode             | `unum::uform::loaders` |
| `include/uform/train/`   | the composition of a training run, `mesh_t` and `trainer_t`   | `unum::uform::train`   |
| `src/*.cpp`              | the verbs of each header, over the cells below                | `unum::uform`          |
| `src/encoder/`           | one unit per CPU and GPU class, and each vendor's kernels     | `unum::uform::encoder` |
| `src/media/`             | the compiled bodies of `include/uform/media/`                 | `unum::uform::media`   |
| `src/core/`              | the compiled bodies of `include/uform/core/`                  | `unum::uform::core`    |
| `src/loaders/`           | the compiled bodies of `include/uform/loaders/`               | `unum::uform::loaders` |
| `src/train/`             | the compiled bodies of `include/uform/train/`                 | `unum::uform::train`   |
| `src_stub/`              | the stand-in, every verb over the C++ library and ForkUnion   | `unum::uform`          |
| `tools/`                 | the `uform_base`, `uform_eval` and `uform_grow` binaries      | none                   |
| `bench/`                 | the encoder and training-kernel benchmarks and their harness  | `unum::uform::bench`   |
| `test/`                  | one conformance suite per cell, the harness, Python suites    | `unum::uform::test`    |
| `verification/`          | Promela models of the mesh protocols and their Spin runner    | none                   |
| `scripts/data/`          | acquisition, repack and mining scripts, and their tests       | none                   |
| `scripts/mixtures/`      | the dataset mixture JSONs a run trains on                     | none                   |
| `scripts/slurm/`         | the `.sbatch` runners and the cluster notes                   | none                   |
| `python/`                | the CPython extension over the C ABI                          | none                   |
| `rust/`                  | the Rust crate over the C ABI                                 | none                   |
| `javascript/`            | the Node-API addon and the WebAssembly module over the C ABI  | none                   |
| `swift/`                 | the Swift package over the C ABI                              | none                   |
| `docs/`                  | the Sphinx site                                               | none                   |

Namespaces mirror directories and nothing else: a directory adds a namespace component, and no namespace groups symbols by theme.
`c/`, `include/` and `src/` are include roots that add no component, so `src/core/` and `include/uform/core/` both open `unum::uform::core`.
Templates and model implementations live in `include/uform/<dir>/` and are always spelled `"uform/<dir>/..."`, while `src/` holds their instantiations, the dispatch and the per-class units, with only its flat `lib.hpp`, `encode.hpp` and encoder dispatch headers included as `"lib.hpp"`.
Dependencies run one way:

- `encoder` and `media` build on the host libraries alone and never include an NVIDIA header; each vendor's kernels sit in their own `vram_kernels_<vendor>` unit.
- The mesh, the multimem collectives and InfiniBand live in `core` beside the losses that use them, and of the verbs only `src/train.cpp` reaches them.
- `loaders` and `train` build on `core`, `tools` and `bench` build on `train`, `core` or the C ABI, and `test` may include any of them.
- `c/uform/` includes only itself and C headers, and every binding includes only `c/uform/`.

The pre-commit hook refuses a namespace no directory backs and a nameless one.

## Code Rules

- __Atomics.__
  Shared words are plain values touched through the class's ForkUnion atomic reference, and waits go through the class's waiter by address.
  `std::atomic`, `std::atomic_ref`, `std::atomic_flag` and `notify_*` appear nowhere; device code uses `cuda::atomic_ref`.
- __Numerics.__
  Engines call kit verbs; kits call NumKong's per-capability kernels by name.
  `_serial` kernels appear only in the serial class's kit rows and in test oracles, and `_best` or a capability mask never appears under `include/`.
  An operation NumKong lacks is a kit operation in UForm, composed from NumKong components or written custom.
- __Ranges.__
  Past the C boundary a range is a `std::span`, a NumKong `tensor_view` or `tensor_span`, or a `tape_view`; a pointer with a count is a C ABI or vendor shape only.

## Data Scripts

Every script in `scripts/data/` carries a PEP 723 header, so it runs on its own as `uv run scripts/data/<name>.py`.
Datasets land under `UFORM_DATA_ROOT`, or under the untracked `data/` at the repository root when it is unset.
The tests run the way CI runs them, from the repository root:

```sh
pip install --group test
python -X faulthandler -m pytest -x scripts/data
```

`UFORM_FILTER` keeps the tests whose node id matches it, as a regex or else as a substring.
`UFORM_SEED` pins the seed, `42` by default or `random`, and pytest prints both in its header.

## Bindings

Each binding links either the engine or the stand-in, and its suite loads the tiny checkpoint `UFORM_TEST_CHECKPOINT` names, or else stand-in bytes only the stub loads, reading every shape from the model's description.
The CI runs them the same way:

```sh
pip install --group test . && UFORM_TEST_CHECKPOINT=build_cpu_library/tiny.safetensors pytest -x test/python
cargo test --features stub
npm install --ignore-scripts && npm run build-native-stub && npm run build-js && npm test
cmake --preset stub_library && cmake --build --preset stub_library --target uform_static_stub
swift test -Xlinker -Lbuild_stub_library/stub
```

`npm run build-wasm` adds the WebAssembly module, whose tests then run in the same `npm test`.
Swift formatting follows the `.swift-format` file at the root, applied with `swift-format . -i -r`.

## Commits

`.githooks/commit-msg` checks every message:

- The title reads `<Verb>: <Summary>`, where the verb is one of `Add`, `Break`, `Build`, `Chore`, `Docs`, `Fix`, `Improve` or `Make`.
- The summary opens with a capital letter, unless it opens with a code identifier, which keeps its own case.
- The title runs to at most 80 characters, and a body line to at most 100 columns, unless it holds a URL or is indented code.
- The message carries no AI-assistant attribution, since authorship is the human who committed.

Git's merge, `fixup!` and `squash!` titles and TinySemVer's `Release:` titles are exempt from the title rules.
