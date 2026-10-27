# UForm Templates

`include/uform/` holds the header-only templates and model implementations the C ABI is built from, for advanced users composing their own towers, loaders or training loops.
The versioned interface stays in `c/uform/*.h`, and `src/` keeps the verbs over it, the dispatch and the units compiled per class.
Only the C ABI is versioned, so these headers change with the code that instantiates them, and an install copies only `c/uform/`.

## Families

- `utilities/`, in `unum::uform::utilities`:
  - `types.hpp`: statuses, `expected`, the scalar and tensor aliases, byte spans and tapes, and the keyed hasher.
  - `ram_device.hpp`: `topology_t`, `executor_t`, `lent_pool_t` and `pool_view_t`, the one owner of host threads, over ForkUnion's C ABI.
  - `array.hpp`: fixed-capacity arrays with inline storage, for tables of one entry per GPU.
  - `safetensors_format.hpp`: reading and writing safetensors checkpoints through NumKong views.
  - `vram_nvidia_device.hpp` and `vram_nvidia_schedule.cuh`: single-GPU allocators, owning streams, and graphs captured once and replayed.
  - `mesh.hpp`, `multimem_collectives.cuh` and `infiniband_collectives.cuh`: the multi-level GPU mesh, and the collectives inside one multimem domain and across them.
- `encoder/`, in `unum::uform::encoder`:
  - `geometry.hpp`: the shape and numerics of one tower, read back from a checkpoint.
  - `tower.hpp`: what every tower shares, the packed weights, the activations a call reserves and the forward recipe.
  - `ram_tower.hpp` and `vram_tower.hpp`: the tower in host memory over a thread pool, and in one GPU's memory over a stream.
  - `ram_kits.hpp` and `vram_kits.hpp`: the kits a tower runs on, NumKong's products, attention and element-wise kernels per capability, and a ladder per family picking the highest one a mask runs.
  - `entries.hpp` and `tower_kernels.cuh`: the steps NumKong has no kernel for, the packed batch layout, the rotary grids and the E4M3 scaling.
  - `multimodal_batch.hpp` and `multimodal_batch.cuh`: training batches one per GPU, and the mixer shuffling them across GPUs.
- `media/`, in `unum::uform::media`:
  - `planning.hpp`: how many patches an image or a clip earns, how they factor, and what gets cropped.
  - `image.hpp`, `video.hpp` and `audio.hpp`: host decoding of images, clips and recordings into a batch's layout.
- `loaders/`, in `unum::uform::loaders`:
  - `direct_storage.hpp` and `dataset_shards.hpp`: one process-wide storage setup, and the GPU-native shard readers over it.
  - `tokens_loader.cuh`, `image_loader.cuh`, `video_loader.cuh`, `audio_loader.cuh`, `paired_loader.cuh` and `choices_loader.cuh`: the Parquet loaders of each modality and pairing.
- `train/`, in `unum::uform::train`:
  - `encoder_tower.hpp` and `transformer_block.cuh`: the trainable tower and its pre-norm blocks, whose E4M3 twin runs forward only.
  - `contrastive_segments.cuh`, `contrastive_tokens.cuh` and `contrastive_relations.cuh`: the pooled and late-interaction contrastive losses, and the relation graph they relax.
  - `multimodal_masking.hpp`: the masked-modeling objective and its reconstruction heads.
  - `optimizer.hpp`: Adafactor with factored second moments.
  - `base.cuh`, `datasets.hpp`, `introspect.cuh` and `trainer.hpp`: one run's configuration and replicas, its data mixtures, its telemetry, and the trainer stepping it.

## Dependencies

Families include one way, `utilities` ← `encoder` ← `media` ← `loaders` ← `train`, so a header reaches only the families to its left.
`utilities/` includes nothing from the other four, and the hook refuses a header there that does.
The mesh and its collectives include only system headers, the host types, the device allocators, the schedule and each other, so they lift out of UForm whole.
A `ram_` header includes no `vram_` one, a vendor's header includes only its own vendor's and the shared ones, and a header every vendor compiles names none.
Nothing here names a class, holds a capability mask other than a kit's own `capability_k`, owns a ForkUnion pool, or sleeps on its own schedule: a template takes the kits, atomic references and pool views that `src/` picks for it.

## Namespaces and Aliases

Every folder opens `unum::uform::<dir>`, and its headers are spelled `"uform/<dir>/..."` everywhere, `src/` included.
A file reaching another family names it through one alias, declared in the namespace or translation unit that uses it:

- `uu` for `unum::uform::utilities`.
- `ue` for `unum::uform::encoder`.
- `um` for `unum::uform::media`.
- `ul` for `unum::uform::loaders`.
- `ut` for `unum::uform::train`.

Sibling projects follow the same pattern through `fu`, `nk`, `sz` and `st`.
The full `unum::uform::` spelling appears only on alias and `using` lines, on the lines opening and closing a namespace, and in specializations of foreign templates such as `std::formatter`.
A `.cpp` or `.cu` unit never opens a namespace block of its own, reaching the one it defines through `using namespace` or an alias.
