# Contributing to UForm

UForm 4 is a training and inference engine behind one C ABI, with bindings for Python, Rust, JavaScript and Swift, and Python data tooling around it.

## Before You Start

Pull the NumKong, StringZilla, ForkUnion and SmashTable submodules before building anything, and put each on the branch `.gitmodules` names, so it tracks that branch instead of sitting on a detached commit:

```sh
git submodule update --init --recursive
git submodule foreach --quiet 'git switch --quiet "$(git config -f "$toplevel/.gitmodules" "submodule.$name.branch")"'
```

Later, bring every submodule to the tip of its branch before bumping the commits this repository records:

```sh
git submodule foreach --quiet 'git pull --ff-only --quiet'
```

`CMakeLists.txt` builds each submodule in place when it is checked out, and otherwise fetches the commit `uform_sibling` pins for it.
A commit that moves a submodule moves its pin too, and the pre-commit hook refuses one where the two disagree.
Machine specifics such as a compiler path go into a gitignored `CMakeUserPresets.json`, whose presets inherit the ones in `CMakePresets.json`.

## Building

The prerequisites are listed at the top of `CMakeLists.txt`.
Every preset in `CMakePresets.json` builds into `build_<preset>`, with Clang compiling the host code and hosting `nvcc`:

```sh
cmake --preset library -D UFORM_BUILD_TEST=ON   # the C library for the host, and its suites
cmake --build --preset library
ctest --test-dir build_library
```

- `library` builds the engine for the host alone, and `stub_library` the stand-in in `src_stub/`.
- `cuda_rocm_library` adds both GPU residencies and the trainer, `linux_cuda_rocm` both residencies alone, the tree whose `readelf` check guards the vendor boundary, while `debug`, `check`, `relwithdebinfo` and `release` also build the trainer and its suites, `check` with warnings as errors.
- `wasm32_emscripten` and `wasm32_wasi` build for WebAssembly, with Node or Wasmtime running the tests.
- `-D UFORM_BUILD_BENCH=ON` adds the benchmarks [`bench/README.md`](bench/README.md) describes.

Configuring CMake also arms the hooks in `.githooks/`; to arm them without a build, run `git config core.hooksPath .githooks`.

## Layout and Namespaces

| Path                       | Contents                                                     | Namespace                |
| :------------------------- | :----------------------------------------------------------- | :----------------------- |
| `c/uform/*.h`              | the C ABI, `uform.h` over the four headers beside it         | none                     |
| `include/uform/utilities/` | vocabulary, devices, schedules, the mesh and collectives     | `unum::uform::utilities` |
| `include/uform/encoder/`   | towers, kits, geometry, entries, batches and tower kernels   | `unum::uform::encoder`   |
| `include/uform/media/`     | image, video and audio decoding on the host                  | `unum::uform::media`     |
| `include/uform/loaders/`   | storage sessions, shard readers, GPU media decode            | `unum::uform::loaders`   |
| `include/uform/train/`     | the training engine: cells, losses, optimizer and trainer    | `unum::uform::train`     |
| `src/*.cpp`                | the verbs of each header, over the cells below               | `unum::uform`            |
| `src/lib.hpp`              | the classes, ladders, handles and outcomes every verb shares | `unum::uform`            |
| `src/encoder_*.hpp`        | the verb tables of a variant and the towers each class picks | `unum::uform`            |
| `src/encoder/`             | one unit per CPU and GPU class, and each vendor's kernels    | `unum::uform::encoder`   |
| `src/media/`               | the compiled bodies of `include/uform/media/`                | `unum::uform::media`     |
| `src/utilities/`           | the compiled bodies of `include/uform/utilities/`            | `unum::uform::utilities` |
| `src/loaders/`             | the compiled bodies of `include/uform/loaders/`              | `unum::uform::loaders`   |
| `src/train/`               | the compiled bodies of `include/uform/train/`                | `unum::uform::train`     |
| `src_stub/`                | the stand-in, every verb over the C++ library and ForkUnion  | `unum::uform`            |
| `tools/`                   | the `uform_base`, `uform_eval` and `uform_grow` binaries     | none                     |
| `bench/`                   | the encoder and training-kernel benchmarks and their harness | `unum::uform::bench`     |
| `test/`                    | one conformance suite per cell, the harness, Python suites   | `unum::uform::test`      |
| `verification/`            | Promela models of the mesh protocols and their Spin runner   | none                     |
| `scripts/data/`            | acquisition, repack and mining scripts, and their tests      | none                     |
| `scripts/mixtures/`        | the dataset mixture JSONs a run trains on                    | none                     |
| `scripts/slurm/`           | the `.sbatch` runners and the cluster notes                  | none                     |
| `python/`                  | the CPython extension over the C ABI                         | none                     |
| `rust/`                    | the Rust crate over the C ABI                                | none                     |
| `javascript/`              | the Node-API addon and the WebAssembly module over the C ABI | none                     |
| `swift/`                   | the Swift package over the C ABI                             | none                     |
| `docs/`                    | the Sphinx site                                              | none                     |

Namespaces mirror directories and nothing else: a directory adds a namespace component, and no namespace groups symbols by theme.
`c/`, `include/` and `src/` are include roots that add no component, so `src/train/` and `include/uform/train/` both open `unum::uform::train`.
Templates and model implementations live in `include/uform/<dir>/` and are always spelled `"uform/<dir>/..."`, while `src/` holds their instantiations, the dispatch and the per-class units, with only its flat `lib.hpp`, `encode.hpp` and encoder dispatch headers included as `"lib.hpp"`.
Dependencies run one way:

- `encoder` and `media` build on the host libraries alone and never include an NVIDIA header; each vendor's kernels sit in their own `vram_kernels_<vendor>` unit.
- The mesh, the multimem collectives and InfiniBand live in `utilities` beneath the losses that use them, and of the verbs only `src/train.cpp` reaches them.
- `loaders` and `train` build on `utilities`, `tools` and `bench` build on `train`, `utilities` or the C ABI, and `test` may include any of them.
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
  A ragged list of numbers, like the row groups of each picked shard, is a `tape_view`, never a span of vectors.
  A list of strings an interface reads, like shard paths or column names, is a StringZilla `sz::strings_view_t`, which borrows an Arrow tape and a span of `sz::string_view_t` alike, so the caller keeps whichever it holds.
  Owners behind those views stay plain: an `st::basic_vector` of `sz::string_t` or of `sz::string_view_t` is fine as a member or a local.
  Structs read inside kernels keep pointer and count members, marked `device-boundary`, where a view would change the register budget.

### Naming

- `_t` marks concrete types and the aliases that materialize a template, and `_k` marks constants and enumerators.
- Class and variable templates carry neither suffix: `ram_tower`, `encoder_class_tables`, `device_capabilities_of`.
- Template parameters end in one underscore, as `typename tower_type_` and `encoder_home_t home_` do, and so do private and protected data members, so neither shadows the public names around it.
- Identifiers are whole words, like `index`, `count`, `threads` and `segments`.
- One word names one concept: memory is allocated or reserved, a contiguous region is a buffer, a chunk or a plane, and the hook refuses `carve`, `slab`, `seam` and `slot`.
- A `bool` parameter of the C ABI or the library becomes a two-value `enum class`, as `caller_exclusivity_t` names `caller_inclusive_k` and `caller_exclusive_k`.
- A file one residency owns opens with `ram_` or `vram_`, and a vendor's own file names the vendor next, as `vram_nvidia_device.hpp` and `vram_entries_amd.cpp` do.
- A file every GPU vendor compiles names no vendor, and its symbols say `vram_`, never `cuda_`, `hip_` or `metal_`.
- Host headers are `.hpp`, CUDA headers `.cuh`, device units `.cu` and `.hip`, and only C sources use `.h`.
- Prose writes the abbreviation as ID and IDs, never `id` or `ids`.

### Templates and Macros

Every template head and argument list is spelled out where it is used, and constraints are named C++20 concepts, like `task_callback` in `include/uform/utilities/ram_device.hpp`.
A requires-clause parenthesizes only what the grammar needs, and never opens with a negation, since the constrained overload already wins wherever its concept holds.
Macros are object-like: include guards, configuration flags such as `UFORM_TARGET_<CLASS>`, and the `UFORM_EXPORT` marker; the hook refuses a function-like one.
Preprocessor conditionals nest two levels deep at most beside the include guard, no `#elif` chain sits inside another, and the `#endif` closing a pair names its condition, as `#endif // NUMKONG_TARGET_HASWELL` does.

### Errors and Memory

Nothing in the library throws, and every function under `include/uform/` and `src/` is `noexcept`, since an exception escaping into the C ABI would terminate the caller.
A fallible host function returns `expected<value_type_>`, or `expected_t` when it has no value, both from `include/uform/utilities/types.hpp`, and fails through `expected::fail(status, format, arguments...)`.
`status_t` takes every value from `uform_status_t`, so a failure crosses the C ABI as a cast.
Errors are checked with an explicit `if` and `return`, and a two-way structured binding reads `[thing, status]`, never a participle.
A helper that lets a vendor's exceptions reach its caller, as cuDF's readers do, carries a `vendor-boundary` marker on its signature, as does any other line a vendor API forces past these rules.

The library under `include/uform/` and `src/` holds none of these standard types:

| Instead of                                | Use                                                   |
| :---------------------------------------- | :---------------------------------------------------- |
| `std::string`                             | StringZilla `sz::string_t`                            |
| `std::vector`, `std::deque`, maps or sets | SmashTable `st::basic_vector`, sorted for lookups     |
| `std::optional`                           | `expected<value_type_>`                               |
| `std::pair`, `std::tuple`                 | a named struct, a NumKong view or a StringZilla slice |
| `std::function`                           | a template callback parameter                         |
| `std::thread`, `std::mutex`, `std::async` | `pool_view_t`, with ForkUnion locks and waiters       |
| `std::shared_ptr`                         | one owner                                             |
| `std::ranges`                             | the plain algorithms                                  |
| `std::filesystem`, iostreams              | the C library and memory mappings                     |

The suites, tools and benchmarks may also hold a `std::string`, a `std::vector` or a standard map, and reach files through `std::filesystem`.
No library type derives from another: shared behavior is a member the type holds.
A member that locks or writes is not `const`, and no `const_cast` turns a read-only view writable.
Memory a call computes through belongs to its executor, its outputs land in the arena it names, and both keep their high-water mark, so a steady loop allocates once.
Tensors index by rank, `tensor[row, column]`, and slice through the same subscript, `tensor[nk::range(0, rows), nk::slice]`, never through the call form or a flat `row * columns + column`.

### Threads and Placement

Host threads and the machine's topology come from ForkUnion's C ABI in `forkunion.h`, wrapped once by `include/uform/utilities/ram_device.hpp`, and nothing else calls a `fu_*` function.
There `topology_t` harvests the machine through `fu_topology_new`, `executor_t` spawns a pool through `fu_pool_new` and `fu_pool_spawn` and deletes it in its destructor, and `lent_pool_t` wraps a pool another library lends through `uform_executor_on_pool`, dispatching on it but never deleting it.
Engines, loaders, tools and tests dispatch through the `pool_view_t` either one hands out, `for_n` for tasks of similar cost and `for_n_dynamic` for uneven ones, and a dispatch finding the pool busy answers `status_t::contended_k` rather than waiting.
The hook refuses a ForkUnion pool or topology class under `include/uform/`, and a `std::thread` in a CUDA source, where one host thread drives every device through its streams and events.

Shared words follow the Atomics rule above, and `src/lib.hpp` alone picks the primitives for each CPU class.
`waiter_for_cpu_class` pairs a class with its waiter: PAUSE on the AVX2 and AVX-512 cores, TPAUSE on the WAITPKG Xeons, YIELD on the older Arm cores, WFET on the Armv9.2 ones, WRS on RISC-V and `memory.atomic.wait` on WebAssembly.
`atomic_ref_for_cpu_class` pairs it with its atomic reference: RCpc or LSE on the Armv8.6 and newer cores, CMPCCXADD on the newest Xeons, Zacas on RISC-V and a notifying reference on WebAssembly.
Each specialization sits under the `FORKUNION_TARGET_<BIT>` verdict of the probe.
A class whose bit is off falls back to the standard yield and reference, and `forkunion_capabilities_of` reads the bits a class needs off the two types, so a ladder admits it only where the runtime confirms them.
Engines receive these as template parameters and never spell a class, an instruction or a sleep of their own, while the C layer, compiled once at the baseline, waits through the `host_verbs_t` of its executor's class.

### Documentation and Comments

- Every file opens with `@file`, `@author`, `@date` and `@brief`, in that order and contiguous; `@file` names the path from the repository root.
- `@date` reads like `October 25, 2026` and marks when the code first appeared, so it is never rewritten.
- Docblock text sits two spaces past its asterisk, and a tag's continuation hangs four columns past its `@`.
- Declarations are documented with `/** */`, never with a bare `//` line above them, and every docblock has an empty line above it.
- A block with no tag is prose that opens on the `/**` line and closes on its last; a block with a tag gets `/**` and `*/` lines of its own.
- A `@brief` separates a summary from what follows, so a block with nothing after the summary carries no tag, and a summary runs three lines at most.
- Every `@param` states its direction: `@param[in]`, `@param[out]` or `@param[in,out]`.
- A closing `@par Statuses:` block lists only the statuses a verb's contract adds, while resource failures, cancellation, deferral and defects are covered once by the categories in the `uform_status_t` docblock.
- Docblocks mark code with `@c`, `@p`, `@b` and `@ref`, never Markdown emphasis, and a `//` comment names code in backticks, never through a Doxygen command.
- Comment prose stops at 100 columns while code runs to 120, and a paragraph is refilled when an edit leaves room on a line.
- Comments state what the code cannot, never narrate history, and never justify a design by another project's.
- Prose arrows are → with a space on each side, never `->`.
- A table inside a comment is columns aligned with spaces, never pipes.
- Sections are `#pragma region Name` and `#pragma endregion Name`, labelled with letters, digits and spaces and spanning 30 lines or more, never banners of dashes or equals signs.

### Includes and Printing

Includes form groups separated by blank lines: C library headers such as `<cstdint>`, platform headers such as `<unistd.h>`, C++ standard headers, external libraries such as `<yyjson.h>`, sibling projects such as `<numkong/tensor.hpp>`, then UForm's own headers in quotes.
Those come last in three groups of their own: the C ABI as `"uform/types.h"`, the families as `"uform/<dir>/..."`, and the private headers of `src/` such as `"lib.hpp"`.
Each include line names in a trailing comment what the file takes from it.
Sibling projects are reached through the aliases `fu`, `nk`, `sz` and `st`, and the families through `uu`, `ue`, `um`, `ul` and `ut`, as [`include/uform/README.md`](include/uform/README.md) lists.
CUDA sources call the Driver API alone, never include `<cuda_runtime.h>`, and launch kernels through `launch_kernel` or `cuLaunchKernel`, never through triple chevrons.
The library never prints, and suites, tools and benchmarks print whole lines through `fmt::println`, never through `printf`, `fprintf` or iostreams, while the C tests keep their stdio.
Return types are declared before the name, never after an arrow.
clang-format 22 with the checked-in `.clang-format` judges every changed line, and no file turns a formatter off in-band.

## Tests and Benchmarks

Suites are plain `int main` binaries, one `uform_test_<name>` per cell, built under `UFORM_BUILD_TEST` and sharing `test/harness.hpp` rather than a framework.
`test_suite_t::make` reads the settings and prints the version, the capabilities, the seed, the filter and the devices, `run` calls a case for every name the filter selects, and `finish` answers the exit code:

```cpp
static void divide_round_up_bounds(test_suite_t &suite, std::size_t denominator) {
    for (std::size_t numerator = 0; numerator != 100; ++numerator) {
        std::size_t const quotient = divide_round_up(numerator, denominator);
        suite.check(quotient * denominator >= numerator, "{} / {} rounded to {}", numerator, denominator, quotient);
    }
    suite.check_equal(divide_round_up(0, denominator), std::size_t {0}, "zero over {}", denominator);
}

int main(int, char **argv) {
    test_suite_t suite = test_suite_t::make(argv[0]);
    suite.run("divide_round_up/bounds", divide_round_up_bounds, std::size_t {7});
    return suite.finish();
}
```

A passing check is silent, a failing one prints its file, line and operands to stderr, and `run` prints one result line per case and a rerun line under a failure.
A case body prints nothing to stdout, since the harness owns the output.
Oracles compute in Float64 through NumKong's `nk::cast` and `nk::dot` rather than loops widening one BFloat16 at a time, and never through `assert`, which compiles out under `NDEBUG`.
Every suite reads three variables once:

- `UFORM_SEED`: the seed every draw derives from, `42` by default or `random` for a fresh one.
- `UFORM_FILTER`: the cases to run by name, as an ECMAScript regex or else as a substring; a filter matching no case fails the suite.
- `UFORM_DEVICES`: the comma-separated UForm device ordinals the GPU cases run on, the first device of every vendor when unset.

CTest labels sort the suites by what they need: `host` runs anywhere, `cuda` wants a GPU, and `cluster` marks `uform_test_env`, the preflight every multi-node `sbatch` runs first.
`uform_tiny_checkpoint` writes `tiny.safetensors` into the build tree once per run, the fixture `uform_test_inference` and the C tests `uform_test_c` and `uform_test_vram` load.

```sh
ctest --test-dir build_library -L host
UFORM_SEED=random ctest --test-dir build_cuda -L cuda --output-on-failure
```

[`test/README.md`](test/README.md) describes the tiers from one process to many nodes.
The benchmarks in `bench/` share `bench/harness.hpp`: `uform_bench_encoder` builds on every tree, `uform_bench_core` needs CUDA, and [`bench/README.md`](bench/README.md) lists their flags.

## Flags and Constants

Every CMake option carries the `UFORM_` prefix, and every `UFORM_BUILD_*` switch is off by default, so a build names only what it wants:

- `UFORM_BUILD_SHARED`: `uform_static` and `uform_shared` in the default build and the install; without it they build only for the targets that link them.
- `UFORM_BUILD_STUB`: `uform_static_stub` and `uform_shared_stub` from `src_stub/`, which the bindings then link instead.
- `UFORM_BUILD_CUDA`: the NVIDIA residency.
- `UFORM_BUILD_ROCM`: the AMD residency.
- `UFORM_BUILD_TRAIN`: the trainer, the mesh and the loaders over CUDA, which the `cuda` preset and the machine profiles set; only they link CUDA libraries into the C library.
- `UFORM_BUILD_TEST`: the suites in `test/` and the C tests.
- `UFORM_BUILD_TOOLS`: the trainer in `tools/`, with `UFORM_BUILD_TRAIN`.
- `UFORM_BUILD_BENCH`: the benchmarks in `bench/`.
- `UFORM_BUILD_PYTHON` and `UFORM_BUILD_NODE`: the CPython extension and the Node addon, which scikit-build-core and cmake-js set.
- `UFORM_WERROR`: warnings as errors, which the `check` preset sets.
- `UFORM_INSTALL`: with `UFORM_BUILD_SHARED`, the C headers, `uform_static`, `uform_shared` and a CMake package under `lib/cmake/uform`, which `find_package(uform)` reads as `uform::static`, `uform::shared` and `uform::header`, on in a top-level build.
- `UFORM_WITH_INFINIBAND`: `AUTO` probes for InfiniBand and enables it when present, `ON` requires it and `OFF` leaves it out.
- `UFORM_WITH_CODECS`: the host image, video and audio codecs the verbs over files decode with, on by default and off on WebAssembly, Android and Windows presets; CUDA needs it.
- `UFORM_TARGET_ARCH`: the CPU UForm's units and its libraries' are tuned for, `native` or a name like `graniterapids`, and on WebAssembly the module's `serial`, `v128` or `v128relaxed`; empty by default, seeded from the environment.
- `UFORM_TARGET_CLASSES`: the CPU classes the encoder compiles beside `serial`, every class when empty, and under `native`, which needs `UFORM_TARGET_ARCH=native`, the newest of each lineage this host runs.
- `UFORM_CUDA_ARCHITECTURES`: the GPU codes every CUDA unit compiles for, `75-real;80-real;89-real;90a-real;100f-real;120f-real;80-virtual` by default, and `native` for the family code of each visible GPU.
- The `xeon4_hopper`, `xeon6_blackwell`, `grace_hopper` and `vera_rubin` presets build the C library for one machine, pinning its CPU, its one CPU class and its one GPU code.
- `UFORM_SITE_PACKAGES` and `UFORM_CUDF_ROOT`: the directory the libcudf wheel discovery searches, or the prefix holding the libraries themselves.

Binaries take long kebab-case flags with no short forms, as `uform_bench_encoder --checkpoint model.safetensors --cpu-class intel-haswell` does, and spell classes, scalars and statuses as the C name tables do.
A namespace-scope `_k` constant names an invariant or the default a zero field derives, as `deferred_limit_default_k` and `default_progress_steps_k` in `src/lib.hpp` do, while a size a caller may want otherwise is a field of an argument structure.

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
pip install --group test . && UFORM_TEST_CHECKPOINT=build_library/tiny.safetensors pytest -x test/python
cargo test --features stub
npm install --ignore-scripts && npm run build-native-stub && npm run build-js && npm test
cmake --preset swift -D UFORM_BUILD_STUB=ON && cmake --build --preset swift
UFORM_SWIFT_ARTIFACT=build_swift/UFormC.artifactbundle swift test
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

## Implementing a New Engine

An engine here is a CPU or GPU class: a rung of a ladder that runs the encoder tower on kits picked for its instructions.
[`src/README.md`](src/README.md) walks the dispatch end to end, and a new class touches the places every existing one does:

1. Kits: where NumKong compiles kernels no existing kit calls, add a `<capability>_products_kit_t`, `_attention_kit_t` or `_elementwise_kit_t` to `include/uform/encoder/ram_kits.hpp`, or to `vram_kits.hpp` for a GPU, under `#if NUMKONG_TARGET_<CAPABILITY>` beside a twin whose `compiled_k` is false, and list it in its family's ladder in priority order.
2. The C enum: append the class to `uform_cpu_class_t` or `uform_gpu_class_t` in `c/uform/types.h`, spell it in the name tables of `src/types.cpp`, and move the bounds in `src/lib.hpp` and `c/test.c` that name the last enumerator.
3. The class: in `src/lib.hpp`, add its `cpu_class_t` or `gpu_class_t` enumerator, its row of `numkong_capabilities_of` as its parent's mask plus a delta, its `parent_class_of`, and its place newest first in `cpu_ladder_t`, `nvidia_ladder_t` or `amd_ladder_t`.
4. Its primitives: a CPU class adds its cases to `forkunion_capabilities_of` and `cpu_class_compiled`, and specializes `waiter_for_cpu_class` and `atomic_ref_for_cpu_class` when it pairs a ForkUnion primitive its parent lacks; a GPU class adds the architecture numbers its driver reports to `gpu_class_of`.
5. The build: a CPU class restates its delta in `src/CMakeLists.txt` as `uform_cpu_class(<class> <CLASS> OVER <parent> <capability> <MACRO>... FORKUNION <capability>...)`, which reaches `src/lib.hpp` as `UFORM_TARGET_<CLASS>`, and a GPU class is one `uform_nvidia_class(<class>)` or `uform_amd_class(<class>)` line.
6. The class unit: `src/encoder/ram_variants_<class>.cpp` instantiates `encoder_class_tables` for `encoder_home_t::ram_k` and `host_class_tables` for the class, while `src/encoder/vram_variants_<vendor>_<class>.cpp` instantiates `encoder_class_tables` for its vendor's home.

A class whose kits match its parent's answers no table, so the ladder falls through to the parent, and a CPU row whose delta NumKong did not compile builds no unit at all.
A new GPU vendor also adds an `encoder_home_t`, an entries kit declared in `vram_kits.hpp` and defined in `src/encoder/vram_entries_<vendor>.cpp` beside the device-only bundle of its `vram_kernels_<vendor>` unit, and its arms of `vendor_of`, `entries_kit_of` and `visit_vram_home`, so the C verbs never name it.
`uform_test_inference` encodes through every compiled CPU class against the serial floor in both precisions, and through the newest class each device `UFORM_DEVICES` selects admits, so a new class is checked as soon as its unit compiles.

## Implementing a New Language Binding

A binding wraps the C ABI described in [`c/README.md`](c/README.md), includes only `c/uform/`, and never instantiates a C++ template.
It mirrors the headers as one module each, `types`, `model`, `encode` and `train`, with one method per exported verb spelled as in C.

A binding has no life of its own: the C executor owns every thread and stream, so a binding starts no threads and keeps no queues or caches.
A receiver chooses where a verb runs while the verb stays the same: the model runs it on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it and answers a deferred call.
A deferred call polls through `uform_outcome_status` and waits through `uform_outcome_wait`, and releasing one rides the language's own lifetime construct, which waits first.
Every output verb takes an arena made through `uform_arena_init` and answers views borrowing it, which the next call on that arena invalidates.
Enums cross through the C `_name` and `_from_name` functions rather than tables of the binding's own.

Each binding reaches the C ABI through a CMake build and compiles nothing of `src/` itself:

- `python/`: `pyproject.toml` builds the CPython extension through scikit-build-core, which sets `UFORM_BUILD_PYTHON`.
- `rust/`: `build.rs` configures CMake with the switches its enabled features name, and links what that build produced.
- `javascript/`: cmake-js builds the Node-API addon under `UFORM_BUILD_NODE`, and `npm run build-wasm` the WebAssembly module.
- `swift/`: SwiftPM compiles no C, and links the `uform_static` archive the `swift` preset packs as an XCFramework or an artifact bundle.

The Design sections of [`python/README.md`](python/README.md), [`rust/README.md`](rust/README.md), [`javascript/README.md`](javascript/README.md) and [`swift/README.md`](swift/README.md) spell these rules out in each language.
