# Slurm Runners

This directory holds the Slurm runners: the training launchers `base-4-8-16.sbatch` and `base-2-8-16.sbatch`, and the `test_*.sbatch` conformance runners, one per `uform_test_<name>` binary they launch.
The point of this note is two things: how UForm's __gang-scheduled__ training jobs are laid out on a Slurm cluster, and how a small __2×HGX-H100 (Hopper)__ cluster is used to emulate the multi-tier NVLink-multicast topology of the future __GB200 Blackwell (NVL72)__ and __Rubin__ racks.
It is grounded in a live 2×HGX-H100 run (driver 580, CUDA 13); the Hopper findings below are measured, not assumed.

The __data-acquisition__ job `acquire.sbatch` shares the directory but none of this: documented in `scripts/data/README.md`, it is embarrassingly parallel, GPU-agnostic, and shape-_agnostic_ — the opposite of everything here.

## Mesh Shape

The world is a shape `{host_domain, multimem_domain, colocated_domain}` with 1 ≤ `host_domain` ≤ `multimem_domain` ≤ `colocated_domain` (see `include/uform/utilities/mesh.hpp`).
`host_domain` is the GPUs __one process__ drives; `multimem_domain` is the GPUs reachable by `multimem.*` over one NVLink/NVSwitch island (HGX = 8, NVL72 = 72); `colocated_domain` is the whole job.
When `host_domain < multimem_domain`, __several processes compose one multicast domain__ — the interesting cross-process case (NVL72 is `{4, 72, 144}`: 18 processes per rack-wide multicast domain).

## Rack Emulation

- __`{8, 8, 16}` — works today.__
  One process per node driving all 8 GPUs; `host_domain == multimem_domain`, so no cross-process multicast.
  This is the straightforward 2-node launch.
- __`{4, 8, 16}` — the emulation target__ (`base-4-8-16.sbatch`).
  Two processes per node, each 4 GPUs; the node's 8-GPU island is one `multimem_domain` __composed of two processes__ (cross-process multicast within the node), and the two nodes form the `colocated_domain` over RDMA.
  This is the small-scale stand-in for the racks: it exercises __both tiers__ a GB200-NVL72 / Rubin deployment will use — intra-island cross-process multicast, then inter-island RDMA — so the mesh bootstrap is validated on Hopper before the Blackwell/Rubin hardware exists.

## Hopper Findings

- __UForm's `multimem_domain_t`, single-process, 8-GPU: PASS, `tier=switch_reduced`__ — real hardware multicast (not a peer-read fallback), distributed reduce matches the Float64 oracle.
- __Cross-process 2-GPU multicast via `POSIX_FILE_DESCRIPTOR` + a Unix-socket rendezvous: PASS__ (a reduce across a process boundary).
  This is the handle type the `{4, 8, 16}` shape needs.
- __Cross-process via `CU_MEM_HANDLE_TYPE_FABRIC`: FAILS ("operation not permitted")__ even with `nvidia-imex` + `nvidia-fabricmanager` active — a standalone HGX box is a __standalone NVSwitch island__ (`GPU Fabric GUID = N/A`, `CliqueId = 0`), with no provisioned NVLink-_network_ fabric.
- __Topology:__ GPUs 0-3 on NUMA0 (cores 0-63, NIC4-7), GPUs 4-7 on NUMA1 (cores 64-127, NIC0-3).

__Takeaway:__ on standalone HGX, `FABRIC` handles are unavailable (they need a GB200-NVL72-class NVLink-network fabric); the same-node cross-process handle type is `POSIX_FILE_DESCRIPTOR`, and it works.
`gpu_symmetric_arena_t` in `include/uform/utilities/multimem_collectives.cuh` shares a cross-process domain's slices as `FABRIC` handles where every local device exports them, as on NVL72/144 racks once `nvidia-imex` puts every node in one IMEX domain.
On a fabric-less box it falls back to `POSIX_FILE_DESCRIPTOR`, whose Unix-socket handoff is not implemented yet, so such a domain refuses to export rather than run.

## Gang Scheduling

- __Gang scheduling.__
  One allocation, a __single `srun`__ launches all tasks together.
  The mesh's filesystem-rendezvous barrier assumes co-start; independently-scheduled tasks would hang.
  (Slurm gives this for free; on vanilla Kubernetes it needs Volcano / Kueue / the Training-Operator.)
- __Task layout.__
  `--ntasks-per-node` = host_domains per node, `--gpus-per-task` = host_domain size; GRES cgroup enforcement then pins each task's `CUDA_VISIBLE_DEVICES` to exactly its GPUs.
- __Host-domain index = `$SLURM_PROCID`, not `$SLURM_NODEID`.__
  With 2 host_domains per node, `SLURM_NODEID` is `0,0,1,1` — wrong; the launcher exports `UFORM_HOST_DOMAIN_INDEX=$SLURM_PROCID` (`0,1,2,3`), which `mesh.hpp` reads first.
- __NUMA / NIC affinity__ via `--cpu-bind=cores --distribution=block:block` — task 0 → cores 0-63 (NUMA0, near GPUs 0-3 and NIC4-7), task 1 → cores 64-127.
- __`--exclusive`__ on the one-process-per-node runners — own the node, since tightly-coupled jobs never share GPUs.
  The two-process `{4,8,16}` runners drop it, as the GPU-masking gotcha below explains.
- __`--no-kill`__ on `base-2-8-16.sbatch` and the two-process runners — keep survivors up so the mesh can re-bootstrap a spare host_domain into a dead peer's index.
- __Env-var contract__ — `UFORM_GPUS_PER_MULTIMEM_DOMAIN`, `UFORM_EXCHANGE_DIR` (a shared-filesystem rendezvous path), `UFORM_JOB_ID=$SLURM_JOB_ID`, exported to every task.

## Runner Contents

The split is deliberate: a runner is __shape-only__ so it commits once and runs on any site; everything site-specific is passed at submit time or documented here.

__In the `.sbatch` file__ (portable, committed):

- the mesh __shape__ as `#SBATCH` geometry — `--nodes`, `--ntasks`, `--ntasks-per-node`, `--gpus-per-task` — and the filename encodes it (`test_mesh-4-8-16.sbatch` is `{host,multimem,colocated} = {4,8,16}`).
- the `srun` launch and the env wiring the binary reads: `UFORM_HOST_DOMAIN_INDEX=$SLURM_PROCID`, `UFORM_HOST_DOMAIN_COUNT=$SLURM_NTASKS`, `UFORM_GPUS_PER_MULTIMEM_DOMAIN`, `UFORM_JOB_ID`, `UFORM_EXCHANGE_DIR`.
- a preflight that fails loudly if the binary is missing.
- __no__ `--partition`, `--account`, or site paths — those are submit-time flags.

__Here in the README__ (site-specific, not committed into the runner):

- how to build and where to stage the binary so every node sees it.
- the shared-filesystem path for `UFORM_EXCHANGE_DIR`.
- the submit command with your partition/account.
- the cluster-standup notes and their gotchas.

## Distributed Runs

The binary must sit on a filesystem every node sees, and `UFORM_EXCHANGE_DIR` must be genuinely shared.
`uform_test_mesh` links the training libraries, so every node needs the shared libraries the build found, and `uform_test_env` names any that is missing.

```
cmake --build build_cuda --target uform_test_mesh
cp build_cuda/uform_test_mesh /shared/uform/uform_test_mesh
sbatch --chdir=/shared/uform --partition=<p> --account=<a> \
       --export=ALL,UFORM_EXCHANGE_DIR=/shared/uform/exchange,UFORM_TEST_BINARY=/shared/uform/uform_test_mesh \
       scripts/slurm/test_mesh-4-8-16.sbatch
```

Every rank prints its cases and a `total N passed, N skipped, 0 failed` line into `mesh-4-8-16-<jobid>.out`; the job passes iff every rank passes.

## B300 Bring-Up

Two DGX B300 nodes form the `{8, 8, 16}` world: one process per node drives its 8 GPUs over NVLink, and the two nodes reduce over InfiniBand.
Build once on a node, in a checkout on the shared filesystem, so every rung runs from the same tree and the mixtures' relative `data/` globs resolve:

```sh
cmake --preset cuda -D UFORM_CUDA_ARCH=100f -D UFORM_WITH_INFINIBAND=ON
cmake --build build_cuda --parallel
export UFORM_EXCHANGE_DIR=/shared/uform/exchange
mkdir -p "${UFORM_EXCHANGE_DIR}"
```

`UFORM_WITH_INFINIBAND=ON` fails the configure where `libibverbs` or `libmlx5` is missing, instead of building a single-node trainer.
Each rung below must pass before the next one runs, and every command runs from the checkout's root with the site's `--partition` and `--account`:

1. The preflight on both nodes, which names any missing library, GPUDirect registration or unshared exchange directory:
   `srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --exclusive build_cuda/uform_test_env`
2. The mesh bootstrap across both nodes, every rank assembling a byte-identical world table:
   `sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=8 --export=ALL,UFORM_TEST_BINARY=build_cuda/uform_test_mesh scripts/slurm/test_mesh-4-8-16.sbatch`
3. The batch mixer on one node, then across both:
   `sbatch --nodes=1 --export=ALL,UFORM_TEST_BINARY=build_cuda/uform_test_mixer scripts/slurm/test_mixer-8-16.sbatch`, then the same with `--nodes=2`.
4. The cross-domain contrastive loss, one process per node against the Float64 global reference:
   `srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --exclusive --export=ALL,UFORM_JOB_ID=contrastive-$(date +%s) bash -c 'UFORM_HOST_DOMAIN_INDEX=$SLURM_PROCID exec build_cuda/uform_test_contrastive'`
5. The Base conformance suite on one node, then across both:
   `sbatch --nodes=1 --export=ALL,UFORM_TEST_BINARY=build_cuda/uform_test_base scripts/slurm/test_base-4-8-16.sbatch`, then the same with `--nodes=2`.
6. Training on the tiny mixtures, one node of 8 GPUs first, then both nodes:
   `srun --nodes=1 --ntasks=1 --gpus-per-node=8 --exclusive build_cuda/uform_base --tier tiny --mixture scripts/mixtures/tiny_text_masked.json --objective masked --steps 50`, then
   `sbatch --export=ALL,UFORM_TIER=tiny,UFORM_MIXTURE=scripts/mixtures/tiny_text_masked.json,UFORM_OBJECTIVE=masked scripts/slurm/base-2-8-16.sbatch` for the masked objective, and
   `sbatch --export=ALL,UFORM_TIER=tiny,UFORM_MIXTURE=scripts/mixtures/tiny_image_contrastive.json,UFORM_OBJECTIVE=joint scripts/slurm/base-2-8-16.sbatch` for the contrastive pairs.

The contrastive line of the training log is the whole world's loss, both nodes' summands added, so the two processes print the same value.

## GPU Masking

A `{4,8,16}` runner puts __two__ host_domains (processes) on each node, each owning 4 of the 8 GPUs.
For that split to be real, Slurm must mask GPUs per task — `TaskPlugin=task/cgroup` + `cgroup.conf` `ConstrainDevices=yes`, and __no__ `--exclusive` (it hands every task all 8 GPUs, so each process sees `CUDA_VISIBLE_DEVICES=0..7` and the bootstrap silently builds a fictitious 8-GPU-per-process world).
Correct masking shows each task `CUDA_VISIBLE_DEVICES=0,1,2,3` (its own four, remapped) and the merged manifest reports `devices 4` per host_domain.

## Environment Variables

Every UForm binary is configured through two channels with distinct owners.
Command-line flags carry _operator intent_ — the knobs one human sets on the one binary they run (`uform_base --tier tiny --mixture omni --eager`).
Environment variables carry the _launcher contract_ — the per-process facts a scheduler stamps on every rank of a distributed job, where `srun` broadcasts one identical command line and only the environment can differ per node.
The C++ suites take no argv: their whole configuration is this contract plus the fixture paths below.
The C tests take the tiny checkpoint's path as their one argument.

### Mesh Contract

The mains read the host_domain index and count at the launcher boundary and pass them into `mesh_topology_t::bootstrap` as explicit arguments, so a misconfigured launcher cannot change the math behind an already-constructed world.
Bootstrap falls back to the environment only for the multicast extent, the exchange directory and the job ID, when its arguments leave them unset.

- `UFORM_HOST_DOMAIN_INDEX`: this process's node index in the world, via `host_domain_index_from_environment`.
  Defaults to `SLURM_NODEID`, then `OMPI_COMM_WORLD_RANK`, then 0.
- `UFORM_HOST_DOMAIN_COUNT`: the world's node count, via `host_domain_count_from_environment`; 1 skips the InfiniBand tier entirely.
  Defaults to `SLURM_JOB_NUM_NODES`, then `OMPI_COMM_WORLD_SIZE`, then 1.
- `UFORM_GPUS_PER_MULTIMEM_DOMAIN`: GPUs one NVLink multicast domain spans, 72 on a GB200 NVL72 rack; several processes may share one.
  Defaults to this process's visible GPUs.
- `UFORM_JOB_ID`: the rendezvous namespace, minted once per job so two jobs on one filesystem never collide.
  Defaults to `SLURM_JOB_ID`.
- `UFORM_EXCHANGE_DIR`: the shared-filesystem directory the multi-node bootstrap and collectives rendezvous through.
  Has no default and is required when nodes > 1.

### Driver Controls

- `CUDA_MODULE_LOADING=EAGER`, set by `uform_base` and the base and mixer suites before `cuInit`: front-loads all kernel modules.
  Under the default lazy loading, an InfiniBand doorbell kernel's flag spin blocks its context while the peer's next launch blocks on module loading behind it, and the two ranks deadlock.
  Set in-process so the binary is correct by construction, not by launch discipline.
- `CUDA_VISIBLE_DEVICES`, set by the launcher or operator: the standard device mask.
  The two-domain single-box repro splits one node's GPUs into two pseudo-nodes with it.

- `--telemetry signals`, passed by the operator: arms `uform_base`'s gradient-boundary signal probes.

### Test Fixtures

- `UFORM_TEST_IMAGE`, `UFORM_TEST_VIDEO`, `UFORM_TEST_AUDIO`, `UFORM_TEST_DATASET`: media and Parquet fixture paths the `uform_test_loaders` conformance suite reads.
- `UFORM_TEST_CHECKPOINT`: the tiny checkpoint `uform_test_c` reads when no argument names one.
- `UFORM_SEED`, `UFORM_FILTER` or `UFORM_TEST_FILTER`, and `UFORM_DEVICES`: the harness's seed, case filter and device list, documented in `test/harness.hpp`.
- `UFORM_TEST_BINARY`: the staged test executable the `scripts/slurm/test_*.sbatch` runners launch, on a shared filesystem for multi-node jobs.
- `UFORM_ENV_BINARY`: the preflight probe those runners launch first, defaulting to `uform_test_env` beside `UFORM_TEST_BINARY`.
  Its failure aborts the job before the suite runs.

### New Variables

New launcher-contract variables get a `*_from_environment` reader beside the two in `include/uform/utilities/mesh.hpp`, an entry here, and explicit pass-through into whatever they configure — no new component below the mains should call `getenv` itself.
