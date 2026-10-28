# Sources and Dispatch

`src/` implements the C ABI of `c/uform/*.h` over the templates of `include/uform/`, and none of it is a public header.
`uform_host` compiles the verbs once at the baseline instruction set, every class unit compiles with its own class's flags, and `c/CMakeLists.txt` assembles both into `uform_static` and `uform_shared`.

## Folders

- `lib.hpp`: what every verb shares, the CPU and GPU classes with their ladders, each CPU class's waiter and atomic reference, the outcome and progress crossings, the executor, arena and model handles, and the deferred calls; it closes by aliasing `unum::uform` as `uf`.
- `types.cpp`: the verbs of `types.h`, outcomes, executors, arenas, the name tables, the probes, and the registry of deferred calls.
- `model.cpp`: the verbs of `model.h`, picking a variant at load and packing one host tower or one tower per GPU.
- `encode.hpp` and `encode.cpp`: the verbs of `encode.h` over decoded inputs, and the body every one runs once its packer is chosen.
- `encode_media.cpp`: the verbs over encoded files, decoded on the host by `media/`, replaced by `encode_media_absent.cpp` in builds without the codecs.
- `train.cpp`: the verbs of `train.h` over the mesh and the trainer, replaced by `train_absent.cpp` in builds without CUDA.
- `encoder_punned.hpp` and `encoder_specializations.hpp`: the variant tables described below.
- `encoder/`: one unit per class, `geometry.cpp`, each GPU vendor's entries in `vram_entries_<vendor>.cpp`, and `vram_kernels_nvidia.cu` and `vram_kernels_amd.hip`, compiled device-only into one bundle per vendor that the entries launch by name.
- `media/`, `utilities/`, `loaders/` and `train/`: the compiled bodies of the families of the same names, with `loaders/` described in [its own guide](loaders/README.md).

## Every Verb

Every exported verb takes one argument structure and opens the same way:

```cpp
UFORM_EXPORT void uform_text_encode(uform_text_encode_t *arguments) {
    if (submitted<uform_text_encode>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    uform_arena_t const arena = claim_arena(arguments->arena, arguments->executor, {inputs}, outcome);
    if (!arena) return;
    // ...
}
```

- `submitted<verb>` copies a deferred call by value into its executor's table, holding its arena, while a blocking call first runs every call the executor holds.
- `call_outcome_t` marks the outcome pending, collects every step's status privately, and on unwinding releases the arena before the one final store, so a path that forgot to report finishes `uform_status_internal_k` rather than hanging its waiters.
- `claim_arena` refuses a null arena, one of another residency or executor, one another call holds, and inputs inside it, then rewinds it for the call's outputs.

A verb runs the variant its model picked at load through that variant's table, and never asks which residency or vendor the model holds.

## Classes and Ladders

A CPU class is its parent plus a delta.
`numkong_capabilities_of` lowers a class onto NumKong's cumulative mask, `parent_class_of` names the rung it degrades to, and `forkunion_capabilities_of` names the ForkUnion bits its waiter and atomic reference need.
`cpu_ladder_t` lists this architecture's classes newest first, ending on `serial`, or in a monoversioned build its one `monoversioned` class, whose masks are the ones the libraries compiled:

- x86: `intel_diamond` → `intel_granite` → `intel_sapphire` → `amd_genoa` → `intel_icelake` → `intel_skylake` → `intel_sierra` → `intel_haswell`.
- Arm: `nvidia_vera` → `apple_m4` → `apple_m2` → `aws_g2`, with `aws_g5` declared but compiled by no build.
- RISC-V: `alibaba_c930`, and WebAssembly: `wasm64_extended` → `wasm32_v128`.

`nvidia_ladder_t` reads `nvidia_blackwellrtx` → `nvidia_blackwell` → `nvidia_hopper` → `nvidia_ada` → `nvidia_ampere`, and `amd_ladder_t` reads `amd_cdna5` → `amd_cdna4` → `amd_cdna3`.
Each ends on its vendor floor, `gpu_class_t::unknown_k`, which runs NumKong's generic kits on any device of the vendor without a class of its own.

- `pick_cpu_ceiling` is the one caller of NumKong's and ForkUnion's CPU detection: unpinned it answers the masks this machine runs, pinned it answers the class's masks, refusing a class the machine does not run and a ceiling no compiled class fits.
- `uform_executor_for_ram` and `uform_executor_for_vram` keep their ceiling and the `host_verbs_t` of the newest class under it, which the C layer waits through.
- `uform_model_load_ram` walks `pick_cpu_variant` to the newest rung with `cpu_class_compiled` and `cpu_rung_fits` under the load's ceiling, which `pick_cpu_ceiling` made sure exists.
- A monoversioned build compiles no `serial` unit and no named class, so a pin below its masks is refused rather than run on them.
- A rung fits when its two masks are subsets of the ceiling's, so pinning behaves as running on a machine of that class.
- A device's GPU class comes from its driver: `gpu_class_of` maps NVIDIA's compute capability and AMD's gfx numbers, which the entries kits' `describe` records.
- `uform_model_load_vram` lets `visit_vram_home` pick the vendor's home and ladder from the devices' capabilities, and `pick_gpu_variant` walks it under the pinned `gpu_class`.
- `cpu_class_compiled`, `gpu_class_compiled` and `gpu_vendor_compiled` read the `UFORM_TARGET_<CLASS>`, `UFORM_ARCH_CUDA_` and `UFORM_ARCH_ROCM_` definitions `src/CMakeLists.txt` passes as 1 for every unit it compiled and 0 otherwise, so a ladder never asks for a table the build lacks.

## Waiters and Atomics

`waiter_for_cpu_class` and `atomic_ref_for_cpu_class` in `lib.hpp` are the one place a CPU class meets its synchronization primitives.
Each specialization is gated by the `FORKUNION_TARGET_<BIT>` verdict of ForkUnion's probe, so the class units and the dispatch unit name the same rungs, and a class whose bit is off takes the standard yield and the standard atomic reference.
`ram_tower_of` hands the class's atomic reference to the host tower, and `host_class_tables<class>::wait` reads a word through that reference between calls of the class's waiter, which is how an executor's callers wait.
The C layer itself is compiled at the baseline, so the outcome words it stores go through `fu::standard_atomic_ref`, which every class's waiter reads.

## Variant Tables

- `encoder_punned.hpp` is what the dispatch unit and the class units agree on, and includes no tower: `encoder_verbs<home>`, the `create`, `destroy`, `scratch_bytes` and `encode` pointers one variant compiles over its tower passed as `void*`, `encoder_class_tables<home, class>`, whose `find` answers a class's table or null, and the ladder walks.
- `encoder_specializations.hpp` is the one place variants are enumerated, and only class units include it.
  `ram_tower_of` picks each kit from its family's ladder under the class's NumKong mask, and `vram_tower_of` does the same per GPU generation beside the vendor's entries kit.
  `ram_bodies` and `vram_bodies` hold the static members each table entry points at, and `ram_variant_advances` and `vram_variant_advances` drop a class whose tower matches its parent's, so its `find` answers null and the ladder falls through.
- A class unit is two lines: `encoder/ram_variants_apple_m4.cpp` instantiates `encoder_class_tables<encoder_home_t::ram_k, cpu_class_t::apple_m4_k>` and `host_class_tables<cpu_class_t::apple_m4_k>`, and `encoder/vram_variants_nvidia_hopper.cpp` instantiates the table of `encoder_home_t::vram_nvidia_k`.
- `uform_model_handle_t` keeps the picked table as `encoder_home_tables_t`, one alternative per home, and its towers as `void*`, which only that table's entries cast back.

Every body is a static member of a template over its tower type, so no body compiled under one class's flags merges into a caller compiled for another.
The variant libraries link after `uform_host`, so the baseline's copy of any shared inline function comes first.

## Compilation Units

- `uform_host`: the verbs, `geometry.cpp`, the checkpoint reader, and the media planning and resize, joined by the host codecs outside WebAssembly, all compiled at the baseline.
- `uform_encoder_ram_variants`: the CPU class units, each compiled with the flags its `uform_cpu_class` row computes, its parent's plus those of every delta capability NumKong compiled, and present only when its parent is and some delta compiled.
- `uform_encoder_vram_nvidia_variants` and `uform_encoder_vram_amd_variants`: the GPU class units, the vendor floor `vram_variants_<vendor>.cpp` and the entries, which compile no device code of their own, since NumKong's libraries carry every generation's kernels.
- `uform_encoder_nvidia` and `uform_encoder_amd`: the device-only bundles of UForm's own kernels, linked into a read-only section through an assembler stub and loaded by name, so nothing registers them with a GPU runtime.
- `uform::train`, `uform::loaders` and `uform::datasets`: the training engine, the mesh and the loaders, built with CUDA alone, with `uform_nvidia_train` serving the verbs of `train.h`.
