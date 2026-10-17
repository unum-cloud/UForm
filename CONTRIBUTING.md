# Contributing to UForm

UForm 4 is a CUDA training and inference stack with Python data tooling around it.

## CUDA

The prerequisites are listed at the top of `CMakeLists.txt`.
Every preset in `CMakePresets.json` builds into `build_<preset>`:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Configuring CMake also arms the hooks in `.githooks/`; to arm them without a build, run `git config core.hooksPath .githooks`.

## Layout and Namespaces

| Path             | Contents                                                             | Namespace              |
| :--------------- | :------------------------------------------------------------------- | :--------------------- |
| `c/uform/*.h`    | the C ABI, with `uform.h` over the model, encode and train headers   | none                   |
| `src/core/`      | device memory, batches, graphs, mesh, transports, cells, checkpoints | `unum::uform::core`    |
| `src/loaders/`   | storage sessions, shard readers, and image, video and audio decode   | `unum::uform::loaders` |
| `src/train/`     | the composition root of a training run, `base_trainer_t::run`        | `unum::uform::train`   |
| `tools/`         | the `uform_base` trainer binary and its command line                 | `unum::uform::tools`   |
| `test/`          | one conformance suite per cell, and their harness                    | `unum::uform::test`    |
| `scripts/data/`  | acquisition, repack and mining scripts, the mixture JSONs, the tests | none                   |
| `scripts/slurm/` | the `.sbatch` runners and the cluster notes                          | none                   |
| `python/`        | the Python package, moving onto the C ABI                            | none                   |
| `javascript/`    | the JavaScript package, moving onto the C ABI                        | none                   |
| `swift/`         | the Swift package, moving onto the C ABI                             | none                   |
| `rust/`          | the Rust crate, moving onto the C ABI                                | none                   |
| `docs/`          | the Sphinx site                                                      | none                   |

Namespaces mirror directories and nothing else: a directory adds a namespace component, and no namespace groups symbols by theme.
`c/` and `src/` are include roots that add no component, so `src/core/` opens `unum::uform::core` and headers include each other as `"core/types.hpp"`.
Dependencies run one way: `loaders` and `train` build on `core`, `tools` builds on `train`, and `test` may include any of them.
`c/uform/` includes only itself and C headers.
The pre-commit hook refuses a namespace no directory backs and a nameless one.

## Data Scripts

Every script in `scripts/data/` carries a PEP 723 header, so it runs on its own as `uv run scripts/data/<name>.py`.
Datasets land under `UFORM_DATA_ROOT`, or under the untracked `data/` at the repository root when it is unset.
The tests run the way CI runs them, from the repository root:

```sh
pip install --group test
python -X faulthandler -m pytest -x
```

`UFORM_FILTER` keeps the tests whose node id matches it, as a regex or else as a substring.
`UFORM_SEED` pins the seed, `42` by default or `random`, and pytest prints both in its header.

## Swift

The Swift package still holds UForm 3 code, and the binding moves onto the C ABI in `c/uform/`.
To build and test it, use the following commands:

```bash
swift build
swift test
```

Swift formatting is enforced with `swift-format` default utility from Apple.
To install and run it on all the files in the project, use the following command:

```bash
brew install swift-format
swift-format . -i -r
```

The style is controlled by the `.swift-format` JSON file in the root of the repository.
As there is no standard for Swift formatting, even Apple's own `swift-format` tool and Xcode differ in their formatting rules, and available settings.

## JavaScript

The JavaScript package still holds UForm 3 code, and the binding moves onto the C ABI in `c/uform/`.
Before submitting any changes, please make sure that the tests pass.

```sh
npm install
npm test
```

## Commits

`.githooks/commit-msg` checks every message:

- The title reads `<Verb>: <Summary>`, where the verb is one of `Add`, `Break`, `Build`, `Chore`, `Docs`, `Fix`, `Improve` or `Make`.
- The summary opens with a capital letter, unless it opens with a code identifier, which keeps its own case.
- The title runs to at most 80 characters, and a body line to at most 100 columns, unless it holds a URL or is indented code.
- The message carries no AI-assistant attribution, since authorship is the human who committed.

Git's merge, `fixup!` and `squash!` titles and TinySemVer's `Release:` titles are exempt from the title rules.
