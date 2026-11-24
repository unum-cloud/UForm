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

`cmake/uform_siblings.cmake` builds each submodule in place when it is checked out, and otherwise fetches the commit `uform_sibling` pins for it.
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
- `cuda_rocm_library` adds both GPU residencies and the trainer, `linux_cuda_rocm` both residencies alone, the tree whose `readelf` check guards the vendor boundary, `darwin_metal` the Apple residency on Apple silicon, while `debug`, `check`, `relwithdebinfo` and `release` also build the trainer and its suites, `check` with warnings as errors.
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
| `docs/`                    | the Doxygen configuration of the C reference                 | none                     |

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
  A list of strings an interface reads, like shard paths or column names, is a StringZilla `sz::strings_view_t`, which views an Arrow tape and a span of `sz::string_view_t` alike, so the caller keeps whichever it holds.
  Owners behind those views stay plain: an `st::basic_vector` of `sz::string_t` or of `sz::string_view_t` is fine as a member or a local.
  Structs read inside kernels keep pointer and count members, marked `device-boundary`, where a view would change the register budget.

### Naming

- `_t` marks concrete types and the aliases that materialize a template, and `_k` marks constants and enumerators.
- Class, alias and variable templates take neither suffix: `ram_tower`, `encoder_class_tables`, `vram_tower_of`.
- Template parameters end in one underscore, as `typename tower_type_` and `vendored_residency_t residency_` do, and so do private and protected data members, so neither shadows the public names around it.
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

Every all-caps name starts with the full project name, `UFORM_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                    | Form                           | Example                 |
| :------------------------ | :----------------------------- | :---------------------- |
| CPU or GPU class          | `UFORM_TARGET_<CLASS>`         | `UFORM_TARGET_SERIAL`   |
| Optional facility         | `UFORM_WITH_[<OS>_]<FACILITY>` | `UFORM_WITH_INFINIBAND` |
| Architecture fact         | `UFORM_ARCH_<ARCH>_`           | `UFORM_ARCH_CUDA_`      |
| Operating-system identity | `UFORM_OS_<OS>_`               | `UFORM_OS_LINUX_`       |

Every name in these families is always defined, as 0 or 1, and tested with `#if`.
`defined(...)` appears only in a `!defined` default and in a guard that an `#error` follows.
A dependency facility, like `UFORM_WITH_INFINIBAND`, defaults to 0, and the target that links the dependency sets it.
Only the Platform Identity region of `c/uform/types.h` reads operating-system predefines like `__linux__`.
Compiler predefines like `__clang__`, `__GNUC__` and `_MSC_VER` are read raw where they matter.

### Errors and Memory

Nothing in the library throws, and every function under `include/uform/` and `src/` is `noexcept`, since an exception escaping into the C ABI would terminate the caller.
A fallible host function returns `expected<value_type_>`, or `expected_t` when it has no value, both from `include/uform/utilities/types.hpp`, and fails through `expected::fail(status, format, arguments...)`.
`status_t` takes every value from `uform_status_t`, so a failure crosses the C ABI as a cast.
Errors are checked with an explicit `if` and `return`, and a two-way structured binding reads `[thing, status]`, never a participle.
A helper that lets a vendor's exceptions reach its caller, as cuDF's readers do, has a `vendor-boundary` marker on its signature, as does any other line a vendor API forces past these rules.

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

Host threads and the machine's topology come from ForkUnion's C ABI in `forkunion/forkunion.h`, wrapped once by `include/uform/utilities/ram_device.hpp`, and nothing else calls a `fu_*` function.
There `topology_t` harvests the machine through `fu_topology_new`, and `ram_executor_t` spawns a pool through `fu_pool_new` and `fu_pool_spawn` and deletes it in its destructor.
Engines, loaders, tools and tests dispatch through the `pool_view_t` it hands out, `for_n` for tasks of similar cost and `for_n_dynamic` for uneven ones, and a dispatch finding the pool busy answers `status_t::contended_k` rather than waiting.
The hook refuses a ForkUnion pool or topology class under `include/uform/`, and a `std::thread` in a CUDA source, where one host thread drives every device through its streams and events.
The binary's root spawns one pool and passes its `pool_view_t` to every loader and data source, which never spawn one.
KvikIO is set up at one site, sized from the cores the root's pools left idle.
An executor spawns its threads near the memory domain the caller names, and the library never names cores of its own.
A GPU stream is a plain value the caller creates and destroys, and every per-device step makes its device current first.

Shared words follow the Atomics rule above, and `src/lib.hpp` alone picks the primitives for each CPU class.
`waiter_for_cpu_class` pairs a class with its waiter: PAUSE on the AVX2 and AVX-512 cores, TPAUSE on the WAITPKG Xeons, YIELD on the older Arm cores, WFE on the Armv9.2 ones, WFET only where a monoversioned preset vouches for it, WRS on RISC-V, `pause_short` on Power and `memory.atomic.wait` on WebAssembly.
`atomic_ref_for_cpu_class` pairs it with its atomic reference: RCpc or LSE on the Armv8.6 and newer cores, CMPCCXADD on the newest Xeons, Zacas on RISC-V, the ISA 3.0 store atomics on Power and a notifying reference on WebAssembly.
Each specialization sits under the `FORKUNION_TARGET_<BIT>` verdict of the probe.
A class whose bit is off falls back to the standard yield and reference, and `forkunion_capabilities_of` reads the bits a class needs off the two types, so a ladder admits it only where the runtime confirms them.
Engines receive these as template parameters and never spell a class, an instruction or a sleep of their own, while the C layer, compiled once at the baseline, waits through the `host_verbs_t` of its executor's class.

### Documentation and Comments

- Every file opens with `@file`, `@author`, `@date` and `@brief`, in that order and contiguous; `@file` names the path from the repository root.
- `@date` reads like `October 25, 2026` and marks when the code first appeared, so it is never rewritten.
- Docblock text sits two spaces past its asterisk, and a tag's continuation hangs four columns past its `@`.
- Declarations are documented with `/** */`, never with a bare `//` line above them, and every docblock has an empty line above it.
- A block with no tag is prose that opens on the `/**` line and closes on its last; a block with a tag gets `/**` and `*/` lines of its own.
- A `@brief` separates a summary from what follows, so a block with nothing after the summary has no tag, and a summary runs three lines at most.
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

### Query Views

A batched kernel takes its queries as one flat transparent view over a static extent, like a `std::array` of views sized by the batch's extent, never as a span of pointers it chases one query at a time.
A query type that owns packed storage exposes a view of it, and the kernel reads that view.
`.githooks/pre-commit` refuses a `std::span` of pointers to const under `include/`, except a span of C strings, which names things rather than batching queries.

### Refusals

A call the library cannot run answers one status, picked by what is wrong, in this order: `invalid_argument`, `read_only`, `config_mismatch`, `unsupported`, `contended`, `overflow`.
A `NULL` handle the call operates on answers `uninitialized` before any of them.
A malformed argument or list, one that is empty, `NULL` or names a handle twice, answers `invalid_argument`.
A write to a handle opened read-only answers `read_only`.
Handles of another kind, or of a geometry the call cannot join, answer `config_mismatch`.
A well-formed case nothing implements yet, like UForm's training in a build without CUDA, answers `unsupported`.
An option the library cannot honour is refused with `unsupported`, never ignored.
A handle another call holds answers `contended` rather than waiting, and a size past its counter answers `overflow`.
A refusal moves nothing: every output and handle reads as it did before the call.
Each header lists a verb's refusals under `@par Statuses` in this order and in these words, and the stub library answers every one of them.

### Statuses

Statuses 0 to 19 keep the same numbers and meanings in every Unum library: success 0, deferred 1, out_of_memory 2, shape_mismatch 3, unsupported_layout 4, unknown_kind 5, read_only 6, io 7, device 8, malformed 9, internal 10, overflow 11, uninitialized 12, config_mismatch 13, contended 14, cancelled 15, unsupported 16, invalid_argument 17, not_found 18, already_exists 19.
A library leaves out the ones it cannot answer but never renumbers the rest, and numbers its own statuses from 20.
Every enum the C interface names, statuses included, has a `uform_*_name` answering "unrecognized" for a value it does not know and a `uform_*_from_name(view, &out)` answering `invalid_argument` for a name it does not know, leaving `out` as it was.
CPU and GPU class values are ABI: each class keeps the number USearch gives the same hardware, written explicitly, with unused numbers reserved rather than reused.
`uform_version_name` answers the full version, pre-release suffix included, in the engine and in the stub alike.

### Compile-Time Choices

Code picks between alternatives at compile time through three gates, each for one kind of choice.

`#if` picks a toolchain or a platform, and nothing else.
A feature the compiler or the operating system lacks is a `#if`; a choice between two algorithms is not.

```cpp
#if defined(_WIN32) // allowed: the platform decides
using file_handle_t = void*;
#else
using file_handle_t = int;
#endif
```

A `requires` clause picks an overload when a whole function differs.
Every constrained overload has an unconstrained fallback beside it, so a type the constraint rejects still finds a function.

```cpp
template <typename index_type_>
status_t run_plan(index_type_& index) noexcept requires is_nvme_index<index_type_> { /* NVMe rounds */ }

template <typename index_type_>
status_t run_plan(index_type_& index) noexcept { /* every other residency */ }
```

`if constexpr` branches on a named concept or a `constexpr bool`, one level deep.
An `else if constexpr` chain is still one level.

```cpp
template <typename query_type_>
concept hinted_query = requires(query_type_& query) { query.find_hinted(0u); };

if constexpr (hinted_query<query_type_>) seeds = query.find_hinted(shard); // allowed: a named concept
else if constexpr (serial_class_k) seeds = {};                            // allowed: one chain, one level
```

The hook refuses each of these, in the directories its `gated_source_directories` names:

```cpp
if constexpr (requires { query.find_hinted(0u); }) { /* ... */ }   // refused: name the probe as a concept

if constexpr (is_ram_k) {
    if constexpr (is_serial_k) { /* ... */ }                        // refused: a second choice inside the first
}

template <typename value_type_, typename = std::enable_if_t<std::is_integral_v<value_type_>>>
value_type_ clamp(value_type_ value) noexcept;                      // refused: write `requires` on the overload

struct sorted_tag_t {};
void insert(entry_t entry, sorted_tag_t) noexcept;                  // refused: a tag that only picks an overload
```

Control flow nests at most three deep: `if`, `for`, `while`, `switch`, `do`, `try` and `catch` each count, and so does `if constexpr`.
An `else` or `catch` body sits at its `if`'s or `try`'s depth, so an `else if` chain stays flat.
A body without braces counts like a braced one, while a lambda or a plain block adds nothing.
Both sides of a `#if` count from the depth the `#if` began at.

```cpp
for (shard_t& shard : shards)                       // 1
    for (node_t node : shard.nodes)                 // 2
        if (node.live)                              // 3
            while (node.next != end) step(node);    // refused: 4 deep
```

To flatten, return early from failures, extract the inner loop into a function named for what it does, or split an overload per case.

The checks are `check_constexpr_conditions_named`, `check_constexpr_branches_flat`, `check_enable_if_and_tags_absent` and `check_constexpr_nesting`.
They judge only the lines a commit adds, so existing code flattens as it is touched.

### Split String Literals

A string literal split across lines by hand ends every fragment but the last with a trailing `//`.
Without the pin, clang-format joins the fragments or restacks them into a staircase.
This holds in C, C++, CUDA and HIP sources alike.

```cpp
char const help[] =                                 //
    "Usage: probe [options]\n"                      // allowed: every fragment but the last is pinned
    "Runs one probe.\n";

char const help[] = "Usage: probe [options]\n"
                    "Runs one probe.\n";            // refused: the first fragment is unpinned
```

The check is `check_split_string_literal_pinned`.
A comment that explains the line pins it just as well, and a bare `//` on a member line is not taken for a member's documentation.

### Vocabulary

| Concept                                                          | Word                                  | Refused                                       |
| :--------------------------------------------------------------- | :------------------------------------ | :-------------------------------------------- |
| a kind of work in a plan, ended by a barrier                     | phase                                 | stage, step, pass                             |
| the same work again, counted                                     | repetition, named for what repeats: round, depth, nesting | phase, pass       |
| an independent unit of one phase, run by any thread in any order | task                                 | step, chunk                                   |
| sequential work                                                  | step                                  | task                                          |
| a per-capability kernel bundle                                   | kit                                   | trait                                         |
| a compiled alternative per class, dtype or residency             | variant                               | arm, cell                                     |
| RAM, VRAM or NVMe                                                | residency                             | home, memory side                             |
| an exported operation and its table entry                        | verb; trampoline                      | …                                             |
| a fixed-shape collection a verb fills                            | outputs                               | landing, lander                               |
| a member of a plan, cursor or walk                               | what it holds and when it holds it    | bits, current, state, data, flags, tmp, scratch |

## Test Suites

### Principles

The C++ suites are plain `int main` binaries, one `uform_test_<name>` per cell, built under `UFORM_BUILD_TEST` and sharing `test/harness.hpp` rather than a framework.
The C suites, `c/test.c` and `test/vram.c`, reach the library through `c/uform/*.h` alone, and each binding's suite through that binding.
Oracles compute in Float64 through NumKong's `nk::cast` and `nk::dot` rather than loops widening one BFloat16 at a time, and never through `assert`, so every test unit opens with `#undef NDEBUG // ! A test's oracle must stay live in every build`.
Benchmarks never include `test/`, each tree keeps its own harness, and neither uses a test or benchmark framework.

The harness owns stdout, and a case body prints nothing there.
Settings are read once, before the first case, and a value that does not parse prints one line naming it and exits with 1.
Skips are honest: a case that cannot run says why, a filter that selects no case fails the run, and a test missing a dependency names it.

The harness holds one mutable global, `test_run`, the run in progress.
Checks fire on worker threads and inside helpers as well as in a case's own body, so the run counts failed checks in one place and `run_test_case` diffs that count around each case, rather than threading a tally through every call.
After a crash, `p test_run` in a core dump names the case in progress and the last failed check, and the signal handlers print the case too.

### Running

Every preset that sets `UFORM_BUILD_TEST` builds the suites, and CTest labels sort them by what they need: `host` runs anywhere, `gpu` wants a GPU, and `cluster` marks `uform_test_env`, the preflight every multi-node `sbatch` runs first.
`uform_tiny_checkpoint` writes `tiny.safetensors` into the build tree once per run, the fixture `uform_test_inference` and the C tests `uform_test_c` and `uform_test_vram` load.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --test-dir build_debug -L host
UFORM_SEED=random ctest --test-dir build_cuda -L gpu --output-on-failure
```

Each binary prints the library's version, the machine and its settings, then one verdict line per case and a summary:

```text
- Seed: 42, scale: 1, filter: none
- test_slices_match_world(1, 1, 1) ... ok (0.12 s)
  check failed at test/relations.cpp:131
    expected 4, got 5
- test_slices_match_world(2, 8, 16) ... FAILED: 1 failed check(s)
  rerun: build_debug/test/uform_test_relations --seed 42 --filter '^test_slices_match_world\(2, 8, 16\)$'
- test_slices_match_world(4, 8, 16) ... ok (0.31 s)
2 passed, 1 failed, 0 skipped, seed 42
```

A failed check prints where it failed and the operands, and the case goes on, so one run lists every mismatch it reaches.
The run exits with 1 when a case failed, a check failed outside any case, or nothing ran, and with 3 when `UFORM_NO_SKIPS` is set and a case skipped.
`UFORM_TRAP_ON_FAILURE=1` raises `SIGTRAP` at the first failed check instead, which stops a debugger there, or dumps a core without one:

```sh
UFORM_TRAP_ON_FAILURE=1 gdb -ex run --args build_debug/test/uform_test_relations --filter '^test_slices_match_world\(2, 8, 16\)$'
```

Each binding links either the engine or the stand-in, and its suite loads the tiny checkpoint `UFORM_TEST_CHECKPOINT` names, or else stand-in bytes only the stub loads, reading every shape from the model's description.
The CI runs them the same way:

```sh
pip install --group test . && UFORM_TEST_CHECKPOINT=build_library/tiny.safetensors pytest -x test/python
pip install --group test && pytest -x scripts/data
cargo test --features stub
pixi run test
npm install --ignore-scripts && npm run build-native-stub && npm run build-js && npm test
cmake --preset swift -D UFORM_BUILD_STUB=ON && cmake --build --preset swift
UFORM_SWIFT_ARTIFACT=build_swift/UFormC.artifactbundle swift test
```

`test/python` and `scripts/data` run as separate pytest sessions, as each has a `conftest.py` and a `base.py` of its own.
`npm run build-wasm` adds the WebAssembly module, whose tests then run in the same `npm test`.
Swift formatting follows the `.swift-format` file at the root, applied with `swift-format . -i -r`.

### Environment

The suites and benchmarks read these variables, and a C++ test or bench binary also takes each as a flag, which overrides it:

| Variable                | Flag                | Default            | Meaning                                                                                  |
| :---------------------- | :------------------ | :----------------- | :--------------------------------------------------------------------------------------- |
| `UFORM_SEED`            | `--seed`            | `42`               | Seed every draw derives from, an unsigned 64-bit integer or `random`, which a run prints |
| `UFORM_FILTER`          | `--filter`          | unset              | C++ case labels or pytest node IDs to run, as an ECMAScript regex or else a substring    |
| `UFORM_SCALE`           | `--scale`           | `1`                | Multiplier of every baseline iteration count and sweep, like `0.1` or `10`               |
| `UFORM_TRAP_ON_FAILURE` | `--trap-on-failure` | unset              | C++ only: raises `SIGTRAP` at the first failed check, to stop there under a debugger     |
| `UFORM_NO_SKIPS`        | `--no-skips`        | unset              | C++ only: a run in which a case skipped exits with 3                                     |
| `UFORM_DEVICES`         | `--devices`         | one of each vendor | C++ only: comma-separated UForm device ordinals the GPU cases run on                     |
| `UFORM_TEST_CHECKPOINT` | none                | unset              | The tiny checkpoint `c/test.c` and every binding's suite load when no argument names one |
| `UFORM_WARMUP`          | `--warmup`          | `1s`               | Benchmarks only: the untimed run per benchmark, like `200ms` or `1s`                     |
| `UFORM_TIME_LIMIT`      | `--time-limit`      | `10s`              | Benchmarks only: the timed run per benchmark, like `200ms` or `10s`                      |

The C suites read `UFORM_SEED` and `UFORM_FILTER`, matching the filter as a substring, as C has no portable regex.
The Python suites read `UFORM_SEED`, `UFORM_FILTER` and `UFORM_SCALE`, alongside `pytest -k`, and the Rust suite reads `UFORM_SEED`.

`UFORM_SEED` keys every case's own stream by the case's name, as `stream_key(seed, name, index)`, identical in C++, C, Python, Rust and Mojo.
A case's inputs therefore never depend on which cases ran before it, so `UFORM_SEED` together with `UFORM_FILTER` is a faithful reproduction recipe.

### Writing a Test

A C++ case is a function `main` registers through `run_test_case`, under a label `<area>.<case>[.<variant>…]` in lowercase snake case, where the area is the file's stem and the case is the function's name without `test_`:

```cpp
static void test_divide_round_up_bounds(std::size_t denominator) {
    for (std::size_t numerator = 0; numerator != 100; ++numerator) {
        std::size_t const quotient = divide_round_up(numerator, denominator);
        expect_true(quotient * denominator >= numerator, "{} / {} rounded to {}", numerator, denominator, quotient);
    }
    expect_equal(divide_round_up(0, denominator), std::size_t {0});
}

int main(int argc, char **argv) {
    print(probe_machine());
    start_test_run(argc, argv);
    run_test_case("tools.divide_round_up_bounds", test_divide_round_up_bounds, std::size_t {7});
    return finish_test_run();
}
```

Each check counts a failed check, lets the case go on, and returns whether it held; a message is always the trailing `format_at_t` and its operands:

- `expect_true(held)`, for any condition;
- `expect_equal(actual, wanted)`, which prints both operands when they differ;
- `expect_near(actual, wanted, tolerance)`, which holds when `|actual - wanted| <= tolerance * max(1, |actual|, |wanted|)`.

Where the step after a failed check would index or dereference past what it checked, the case stops on the check's answer, as in `if (!expect_equal(rows.size(), wanted.size())) return;`.
A case that cannot run returns `case_result_t::skipped(reason)`, with the reason in one of four forms: `needs <resource>`, `built without <SWITCH>`, `<facility> is refused on this machine` or `no <device>`, as in `no_gpu_device_k`.
A case draws from `split_mix_generator_t {case_stream_key()}`, sizes its loops with `scaled_iterations(baseline)` and `sweep_stride(coverage)`, and waits with `spin_until`, bounded by `wait_limit_k`.
A C case runs between `start_test_case(label)` and `finish_test_case(skip_reason)`, and checks with `expect_(condition)`, or `expect_at` where a message names the subject.

A Python test is a `test_<behaviour sentence>` function with a one-line docstring, and its module imports `base` and `uform`, never another test module.
It draws from the `seed` or `rng` fixture, never from a literal seed, and checks what the interpreter reports as unraisable through the `unraisable` fixture.
A test the build cannot run takes a named mark from `base.py`, as `@requires_cpu_only_build`, and one that needs an optional dependency says so where it is decided, as `@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")`.
`MAX_CASES_PER_TEST` in `base.py` caps how many cases one test function expands to, and raising it is a decision for review.

A Rust test is named as a behaviour sentence, returns `()` and calls `.unwrap()`, so a failure names its own line.
One that cannot run on this machine returns `testing::skipped(reason)`, which prints the reason, and the helpers several modules' tests share, like `testing::checkpoint` and `testing::executor`, live in `rust/testing.rs`.
A Mojo test is registered in `main`, as Mojo has no test discovery, and one that cannot run calls `_skipped(reason)` and returns.

One behaviour keeps one name in every language: the Rust test's name is the stem, and C, C++, Python and Mojo name theirs `test_<stem>`, as `a_held_executor_refuses_other_calls` and `test_a_held_executor_refuses_other_calls`.

### Benchmarks

The benchmarks in `bench/` share `bench/harness.hpp`, which reads the same settings through the same readers as the suites: `uform_bench_encoder` builds on every tree, and `uform_bench_core` needs CUDA.
Their rows follow the suites' label grammar, and [`bench/README.md`](bench/README.md) lists their flags.

### Devices

`UFORM_DEVICES=0,2` picks the devices the GPU cases run on, by the ordinals `uform_gpu_devices` lists, whatever their vendor.
Unset, the cases visit the first listed device of every vendor, and a named device past the listed count ends the run with 1.
On a machine without a GPU, every GPU case reports `skipped: no GPU device`.

### Private Surface Waivers

The C++ suites test the templates in `include/uform/`, as those are the cells, and these reach into `src/` for what neither surface exports:

- `test/inference.cpp` includes `src/encoder_punned.hpp` for `encoder_verbs`, `encoder_class_tables`, `pick_gpu_variant` and `visit_vram_residency`, the ladder a load walks to pick a variant.
- `test/base.cu`, `test/contrastive.cu` and `test/gradients.cu` include `src/train/numerics.hpp` for `gpu_numerics_of`, the kernel table a mesh's agreed capabilities map to.

## Flags and Constants

Every CMake option takes the `UFORM_` prefix, and every `UFORM_BUILD_*` switch is off and every `UFORM_*_ARCH` empty by default, so a build names only what it wants:

- `UFORM_BUILD_SHARED`: `uform_static` and `uform_shared` in the default build and the install; without it they build only for the targets that link them.
- `UFORM_BUILD_STUB`: `uform_static_stub` and `uform_shared_stub` from `src_stub/`, which the bindings then link instead.
- `UFORM_CUDA_ARCH`: the NVIDIA residency, for `all` the release set, `native` the family code of each visible GPU, or one code like `100f`; only the GPU classes NumKong compiles for those codes exist.
- `UFORM_ROCM_ARCH`: the AMD residency, for `all` of `gfx942`, `gfx950`, `gfx1250` and `gfx1251`, `native`, or one target.
- `UFORM_METAL_ARCH`: the Apple residency, for `all` of `apple7`, `apple9` and `apple10`, or one family, never `native`, as each GPU compiles the kernels at first use; off Apple platforms its units compile as a check and nothing links.
- `UFORM_BUILD_TRAIN`: the trainer, the mesh and the loaders over CUDA, which needs `UFORM_CUDA_ARCH` and which the `cuda` preset and the machine profiles set; only they link CUDA libraries into the C library.
- `UFORM_BUILD_TEST`: the suites in `test/` and the C tests.
- `UFORM_BUILD_TOOLS`: the trainer in `tools/`, with `UFORM_BUILD_TRAIN`.
- `UFORM_BUILD_BENCH`: the benchmarks in `bench/`.
- `UFORM_BUILD_PYTHON` and `UFORM_BUILD_NODE`: the CPython extension and the Node addon, which scikit-build-core and cmake-js set.
- `UFORM_WERROR`: warnings as errors, which the `check` preset sets.
- `UFORM_INSTALL`: with `UFORM_BUILD_SHARED`, the C headers, `uform_static`, `uform_shared`, a `uform.pc` and a CMake package under `lib/cmake/uform`, which `find_package(uform)` reads as `uform::static`, `uform::shared` and `uform::header`, on in a top-level build. Its `sdk` component holds the shared library, the C headers, both package files and the license, and `cpack --config build_library/CPackConfig.cmake` packs it alone, stripped, into `uform-<version>-<platform>-sdk.tar.gz`; the `library` and `release` presets remap the source directory out of every binary.
- `UFORM_WITH_INFINIBAND`: `AUTO` probes for InfiniBand and enables it when present, `ON` requires it and `OFF` leaves it out.
- `UFORM_WITH_CODECS`: the host image, video and audio codecs the verbs over files decode with, on by default and off on WebAssembly, Android and Windows presets; CUDA needs it.
- `UFORM_MARCH`: the `-march` floor of UForm's units and its libraries', an ISA level, `native` or a CPU name like `graniterapids`, and on WebAssembly the module's `serial`, `v128` or `v128relaxed`; empty by default, seeded from the environment.
- `UFORM_MULTIVERSIONED`: on by default, every CPU class that exists, `serial` included, picked at run time; off, one `monoversioned` CPU class over every capability the libraries enabled, which is what the floor implies and what a preset vouches for with `-D <LIBRARY>_TARGET_<CAPABILITY>=1`, and only the newest GPU class of each vendor, so a ceiling below it is refused; configure stops when no class runs the one GPU code.
- The `xeon4_hopper`, `xeon6_blackwell`, `grace_hopper` and `vera_rubin` presets build the C library for one machine, naming its CPU and its one GPU code.

GPU code follows one policy per build kind.
The release set compiles UForm's own CUDA units for `80`, `89`, `90a`, `100f` and `120f` cubins with a `compute_80` PTX tail, the encoder bundle's generic SIMT kernels as `compute_75` PTX alone, and the `multimem` bundle as one plain `90` cubin with its PTX, as it needs nothing past Hopper.
A native or named code compiles every unit and bundle as that code's SASS alone.
AMD compiles one code object per listed target either way.

- `UFORM_SITE_PACKAGES` and `UFORM_CUDF_ROOT`: the directory the libcudf wheel discovery searches, or the prefix holding the libraries themselves.

Binaries take long kebab-case flags with no short forms, as `uform_bench_encoder --checkpoint model.safetensors --cpu-class-ceiling intel-haswell` does, and spell classes, scalars and statuses as the C name tables do.
A namespace-scope `_k` constant names an invariant or the default a zero field derives, as `idle_cores_denominator_k` and `default_progress_steps_k` in `src/lib.hpp` do, while a size a caller may want otherwise is a field of an argument structure.

## Data Scripts

Every script in `scripts/data/` has a PEP 723 header, so it runs on its own as `uv run scripts/data/<name>.py`.
Datasets land under `UFORM_DATA_ROOT`, or under the untracked `data/` at the repository root when it is unset.
Their tests run as [Test Suites](#test-suites) describes, in a pytest session of their own.

## Bindings

Every binding is a thin layer over the C interface, and these rules hold in each of them.
Typed bindings are those whose compiler checks callback signatures, and dynamic bindings are the rest; in UForm the typed ones are Rust, Swift and Mojo, and the dynamic ones are Python and JavaScript.

The C library decides what is valid.
A binding forwards its status, turns each status into its language's error in one place, re-checks nothing the library refuses, and keeps no copy of state the library already holds.
Name tables come from the library's `uform_*_name` and `uform_*_from_name` pairs, never from a switch written in the binding.

One mechanism reports every outcome: `uform_outcome_status` reads it without waiting, and `uform_outcome_wait` parks until it is final.
Nothing in a binding takes a lock, sends a notification or adds synchronization of its own.
The object that submitted a deferred call is the only one that polls its outcome, so no other thread ever waits on it, and whatever a second caller asks, the library answers.
A blocking `wait` calls `uform_outcome_wait`.
An asynchronous bridge never parks a thread: it reads `uform_outcome_status` and, while the call is deferred, yields to its runtime.
A runtime with a timer pauses between reads with a backoff growing from about 1 µs to 1 ms, like Python's `asyncio.sleep`, Swift's `Task.sleep`, JavaScript's `setTimeout` and C#'s `Task.Delay`.
A Rust future yields only: `poll` reads the status and, while it is deferred, calls `wake_by_ref` and returns `Poll::Pending`, with no sleep and no timer thread.

A callback never raises into the library.
Its fault never unwinds through C frames and never waits in a latch for the call to end: typed bindings give their callback types no way to throw, like a `noexcept` C++ callback type, while dynamic bindings report the fault through their language's unraisable path, like Python's `sys.unraisablehook` and JavaScript's uncaught-exception handler, and the callback answers cancel.
A Rust callback that panics aborts the process, as Rust does for any unwind reaching an `extern "C"` frame; nothing catches the unwind.
A binding that hands the library no callback of its own has nothing to report.

A view the library hands back lives until the next call on its arena.
The library describes where the memory sits, and the caller checks it; the binding neither copies the view to be safe nor checks its residency on the caller's behalf.

Every array a verb reads shares the call's one count, so mismatched shapes cannot be expressed.
Where a binding's own types hold two lengths, like a slice of keys beside a matrix of values, it checks once that they agree before the call crosses into the library, and nowhere else.

Each verb has one body, which fills the C struct and calls the library with no layers in between.
What it returns is read from the struct's output slot, a trampoline receives the handler itself as its state with no hub around it, and whether a call blocks or defers follows from the type of the receiver it is called on, never from a flag.
Verbs are never shared through a base class; a trait whose default methods hold the one body per verb is fine, like Rust's and Mojo's, since each receiver supplies only its handle.

A typed binding takes the scalar as a type parameter of each call, like USearch's `search::<f32>`, rather than letting a handle's type fix one.
A deferred Rust call owns what the library reads until the call is final, since the borrow checker cannot see the library holding it.
A C++ wrapper over the C interface is a full typed binding: it has a deferred receiver, `noexcept` callback types and per-call scalar templates.

A dynamic binding publishes every field of every report the library fills.
A type the binding hands to C by cast, like a slice of parts read as the C array, is a transparent wrapper over the C struct, like Rust's `#[repr(transparent)]` newtype, so the cast is sound by construction and nothing asserts its size or alignment.
Tests never restate a struct's layout that the compiler already pins: no `offset_of!`, `size_of`, `MemoryLayout` or `Marshal.SizeOf` asserts against the C structs; `offsetof` tables that declare members are declarations, not tests.
Tests of how names print and parse, and of an engine run again through the binding, stay.

`.githooks/pre-commit` refuses locks in binding sources, throwing callback types, caught unwinds and kept exceptions.

## Commits

`.githooks/commit-msg` checks every message:

- The title reads `<Verb>: <Summary>`, where the verb is one of `Add`, `Break`, `Chore`, `Docs`, `Fix`, `Improve` or `Make`.
- The summary opens with a capital letter, unless it opens with a code identifier, which keeps its own case.
- The title runs to at most 80 characters, and a body line to at most 100 columns, unless it holds a URL or is indented code.
- The message has no AI-assistant attribution, since authorship is the human who committed.

Git's merge, `fixup!` and `squash!` titles are exempt from the title rules.
A `Release: vX.Y.Z` title is exempt too, but its commit must stamp every version site, and no other commit may change `VERSION`.
`.githooks/pre-commit` formats and judges the staged lines, and `.githooks/selftest` runs both hooks against their fixtures.
On a pull request, `.github/scripts/check-commits.sh` replays `commit-msg` over every commit and runs `pre-commit` over the whole range.

## Releases

`.github/scripts/mint-release.sh X.Y.Z` cuts a release from a clean checkout of the branch it ships.
It builds the Swift binaries for that version, stamps every version site and the binaries' checksums, and commits `Release: vX.Y.Z`.
`.github/scripts/check-release.sh` then checks that commit, even in a clone where `core.hooksPath` is unset, before the script tags it and pushes both.
Then it publishes the GitHub release with the binaries attached, marking `X.Y.Z-rc.N` candidates as prereleases.
Publishing starts every `release-<binding>.yml` at once, and each runs `check-release.sh` before it builds anything, so each registry trusts only its own binding's file.

## Classes and Picks

Class values are shared by USearch, UStore, UCall and UForm, so one value names the same hardware in all four.

| Value | CPU class         | What it adds                                                   |
| ----: | :---------------- | :------------------------------------------------------------- |
|     0 | `unknown`         | No ceiling: the newest class this machine runs                 |
|     1 | `serial`          | The portable floor, no SIMD                                    |
|     2 | `monoversioned`   | The one class of a monoversioned build, as the libraries built |
|     3 | `intel-haswell`   | AVX2 and FMA                                                   |
|     4 | `intel-sierra`    | AVX-VNNI over Haswell, without AVX-512                         |
|     5 | `intel-skylake`   | AVX-512 over Haswell                                           |
|     6 | `intel-icelake`   | AVX-512 VNNI and VPOPCNT                                       |
|     7 | `amd-genoa`       | AVX-512 bf16 dot products                                      |
|     8 | `intel-sapphire`  | AVX-512 FP16 and the AMX-BF16 and AMX-INT8 tiles               |
|     9 | `intel-granite`   | The AMX-FP16 tiles                                             |
|    10 | `intel-diamond`   | AVX10.2 FP8 conversions                                        |
|    12 | `aws-g2`          | The NEON floor                                                 |
|    13 | `apple-m2`        | Armv8.6 dot products and bf16 and f16 multiplies               |
|    14 | `apple-m4`        | The SME outer-product tiles                                    |
|    15 | `aws-g5`          | `WFE` waits over `apple-m2`                                    |
|    16 | `nvidia-vera`     | FP8 dot products over `apple-m2`                               |
|    17 | `alibaba-c930`    | RVA23 vectors                                                  |
|    18 | `ibm-power10`     | VSX vectors and the MMA accumulators                           |
|    19 | `wasm32-v128`     | The fixed-width `simd128` tier                                 |
|    20 | `wasm64-extended` | Relaxed SIMD, shared memory and 64-bit addressing              |

| Value | GPU class             | What it adds                                    |
| ----: | :-------------------- | :---------------------------------------------- |
|     0 | `unknown`             | No class: the vendor's generic kernels, or none |
|     1 | `nvidia-ampere`       | The NVIDIA floor kits                           |
|     2 | `nvidia-ada`          | fp8 conversions                                 |
|     3 | `nvidia-hopper`       | Hopper's kits over Ada's                        |
|     4 | `nvidia-blackwell`    | Blackwell's kits over Ada's                     |
|     5 | `nvidia-blackwellrtx` | Consumer Blackwell's kits over Ada's            |
|     6 | `nvidia-rubin`        | Rides `nvidia-blackwell`, compiling nothing     |
|     7 | `amd-cdna3`           | The CDNA3 kits                                  |
|     8 | `amd-cdna4`           | The CDNA4 kits over CDNA3's                     |
|     9 | `amd-cdna5`           | The CDNA5 kits over CDNA4's                     |
|    10 | `apple-metal7`        | The generic Metal kits                          |
|    11 | `apple-metal9`        | The Apple9 kits over the Metal ones             |
|    12 | `apple-metal10`       | The Apple10 kits over the Metal ones            |

A CPU class is picked by `pick_cpu_ceiling` in `src/lib.hpp`.
It reads the masks NumKong and ForkUnion detect on this CPU, never NumKong's enabled mask, as UForm compiles its kits into its own units.
It grants the AMX tiles the CPU offers and drops their bits where the kernel refuses them, once per process.
A ceiling no class takes is an invalid argument, and any other intersects with those masks, so a ceiling above the machine runs as the machine.
The newest compiled class whose masks fit the intersection then runs, and only an intersection no compiled class fits, like a ceiling below a monoversioned build's class, is unsupported.

A GPU class is picked per device by `pick_gpu_class` in `src/lib.hpp`, newest first down its vendor's ladder.
A class serves a device when it is built, sits at or under the ceiling, and NumKong's loader runs all its kits there, as `nk_capabilities_compiled_cuda`, `nk_capabilities_compiled_rocm` and `nk_capabilities_compiled_metal` answer.
A device no class serves is still listed, as `unknown`, and refuses work as unsupported.
A VRAM executor runs every named device as the best class all of them load at or under its ceiling, as USearch places an index, and devices sharing no such class are unsupported.

## Implementing a New Engine

An engine here is a CPU or GPU class: a rung of a ladder that runs the encoder tower on kits picked for its instructions.
[`src/README.md`](src/README.md) walks the dispatch end to end, and a new class touches the places every existing one does:

1. Kits: where NumKong compiles kernels no existing kit calls, add a `<capability>_products_kit_t`, `_attention_kit_t` or `_elementwise_kit_t` to `include/uform/encoder/ram_kits.hpp`, or to `vram_kits.hpp` for a GPU, under `#if NUMKONG_TARGET_<CAPABILITY>` beside a twin whose `compiled_k` is false, and list it in its family's ladder in priority order.
2. The C enum: give the class the next free value of the registry USearch, UStore, UCall and UForm share, after the highest any of them uses, in `uform_cpu_class_t` or `uform_gpu_class_t` in `c/uform/types.h`, under its family's comment, spell it in the name tables of `src/types.cpp` and `src_stub/types.cpp`, and list it in `cpu_classes_k` or `gpu_classes_k` in `src/lib.hpp`.
3. The class: in `src/lib.hpp`, add its `cpu_class_t` or `gpu_class_t` enumerator, its row of `numkong_capabilities_of` as its parent's mask plus a delta, its `parent_class_of`, and its place newest first in `cpu_ladder_t`, `nvidia_ladder_t`, `amd_ladder_t` or `apple_ladder_t`.
4. Its primitives: a CPU class adds its cases to `forkunion_capabilities_of` and `cpu_class_compiled`, and specializes `waiter_for_cpu_class` and `atomic_ref_for_cpu_class` when it pairs a ForkUnion primitive its parent lacks; a GPU class adds its case to `gpu_vendor_of`.
5. The build: a CPU class restates its delta in `src/CMakeLists.txt` as `uform_cpu_class(<class> <CLASS> OVER <parent> <capability> <MACRO>... FORKUNION <capability>...)`, which reaches `src/lib.hpp` as `UFORM_TARGET_<CLASS>`, and a GPU class is one `uform_gpu_class(<vendor> <class> <CLASS> <capability>)` line, oldest first.
6. The class unit: `src/encoder/ram_variants_<class>.cpp` instantiates `encoder_class_tables` for `vendored_residency_t::ram_k` and `host_class_tables` for the class, while `src/encoder/vram_variants_<vendor>_<class>.cpp` instantiates `encoder_class_tables` for its vendor's residency.

A class whose kits match its parent's answers no table, so the ladder falls through to the parent, and a CPU row whose delta NumKong did not compile builds no unit at all.
A new GPU vendor also adds an `vendored_residency_t`, an entries kit declared in `vram_kits.hpp` and defined in `src/encoder/vram_entries_<vendor>.cpp` beside the device-only bundle of its `vram_kernels_<vendor>` unit, and its arms of `vendor_of`, `entries_kit_of` and `visit_vram_residency`, so the C verbs never name it.
`uform_test_inference` encodes through every compiled CPU class, loaded through the C interface under a `cpu_class_ceiling` naming it, against the serial floor in both precisions, and through the newest class each device `UFORM_DEVICES` selects admits, so a new class is checked as soon as its unit compiles.

## Implementing a New Language Binding

A binding wraps the C ABI described in [`c/README.md`](c/README.md), includes only `c/uform/`, and never instantiates a C++ template.
It mirrors the headers as one module each, `types`, `model`, `encode` and `train`, with one method per exported verb spelled as in C.

A binding has no life of its own: the C executor owns every thread and stream, so a binding starts no threads and keeps no queues or caches.
A receiver chooses where a verb runs while the verb stays the same: the model runs it on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it and answers a deferred call.
A deferred call polls through `uform_outcome_status` and waits through `uform_outcome_wait`, and releasing one rides the language's own lifetime construct, which waits first.
Every output verb takes an arena made through `uform_arena_init` and answers views into it, which the next call on that arena invalidates.
Enums cross through the C `_name` and `_from_name` functions rather than tables of the binding's own.

Each binding reaches the C ABI through a CMake build and compiles nothing of `src/` itself:

- `python/`: `pyproject.toml` builds the CPython extension through scikit-build-core, which sets `UFORM_BUILD_PYTHON`.
- `rust/`: `build.rs` configures CMake with the switches its enabled features name, and links what that build produced.
- `javascript/`: cmake-js builds the Node-API addon under `UFORM_BUILD_NODE`, and `npm run build-wasm` the WebAssembly module.
- `swift/`: SwiftPM compiles no C, and links the `uform_static` archive the `swift` preset packs as an XCFramework or an artifact bundle.

The Design sections of [`python/README.md`](python/README.md), [`rust/README.md`](rust/README.md), [`javascript/README.md`](javascript/README.md) and [`swift/README.md`](swift/README.md) spell these rules out in each language.
