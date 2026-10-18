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

Configuring CMake also arms the hooks in `.githooks/`; to arm them without a build, run `git config core.hooksPath .githooks`.

## Layout and Namespaces

| Path             | Contents                                                               | Namespace                |
| :--------------- | :--------------------------------------------------------------------- | :----------------------- |
| `c/uform/*.h`    | the C ABI, with `uform.h` over the model, encode and train headers     | none                     |
| `src/*.cpp`      | the verbs of each header, over the cells below                         | `unum::uform`            |
| `src/inference/` | the tower on the host, and its NVIDIA and AMD kernels                  | `unum::uform::inference` |
| `src/media/`     | image, video and audio decoding on the host                            | `unum::uform::media`     |
| `src/core/`      | device memory, batches, graphs, mesh, transports, cells, checkpoints   | `unum::uform::core`      |
| `src/loaders/`   | storage sessions, shard readers, and GPU image, video and audio decode | `unum::uform::loaders`   |
| `src/train/`     | the composition of a training run, `mesh_t` and `trainer_t`            | `unum::uform::train`     |
| `src_stub/`      | the stand-in, answering every verb over the C++ standard library alone | `unum::uform`            |
| `tools/`         | the `uform_base` trainer binary and its command line                   | `unum::uform::tools`     |
| `test/`          | one conformance suite per cell, their harness, and the Python suites   | `unum::uform::test`      |
| `scripts/data/`  | acquisition, repack and mining scripts, the mixture JSONs, the tests   | none                     |
| `scripts/slurm/` | the `.sbatch` runners and the cluster notes                            | none                     |
| `python/`        | the CPython extension over the C ABI                                   | none                     |
| `rust/`          | the Rust crate over the C ABI                                          | none                     |
| `javascript/`    | the Node-API addon and the WebAssembly module over the C ABI           | none                     |
| `swift/`         | the Swift package over the C ABI                                       | none                     |
| `docs/`          | the Sphinx site                                                        | none                     |

Namespaces mirror directories and nothing else: a directory adds a namespace component, and no namespace groups symbols by theme.
`c/` and `src/` are include roots that add no component, so `src/core/` opens `unum::uform::core` and headers include each other as `"core/types.hpp"`.
Dependencies run one way:

- `inference` and `media` build on the host libraries alone and never include an NVIDIA header; each vendor's kernels sit in their own `vram_<vendor>_kernels` unit.
- Only `src/train/` and `src/train.cpp` touch the mesh, the multimem collectives and InfiniBand.
- `loaders` and `train` build on `core`, `tools` builds on `train`, and `test` may include any of them.
- `c/uform/` includes only itself and C headers, and every binding includes only `c/uform/`.

The pre-commit hook refuses a namespace no directory backs and a nameless one.

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
