# Slurm cluster configuration, and emulating Blackwell / Rubin on dual Hopper nodes

_Draft — distilled from the operational notes gathered on the live cluster; to be polished._

This directory holds the training launch templates.
The point of this note is two things: how UForm's __gang-scheduled__ training jobs are laid out on a Slurm cluster, and how a small __2×HGX-H100 (Hopper)__ cluster is used to emulate the multi-tier NVLink-multicast topology of the future __GB200 Blackwell (NVL72)__ and __Rubin__ racks.
It is grounded in a live 2×HGX-H100 run (driver 580, CUDA 13); the "verified" tags below are measured, not assumed.

It is deliberately separate from the __data-acquisition__ job (`python_data/acquire.sbatch`, documented in `data/README.md`), which is embarrassingly parallel, GPU-agnostic, and shape-_agnostic_ — the opposite of everything here.

## The mesh, in one paragraph

The world is a shape `{host_domain, multimem_domain, colocated_domain}` with `1 <= host_domain <= multimem_domain <= colocated_domain` (see `cuda_core/mesh.cuh`).
`host_domain` is the GPUs __one process__ drives; `multimem_domain` is the GPUs reachable by `multimem.*` over one NVLink/NVSwitch island (HGX = 8, NVL72 = 72); `colocated_domain` is the whole job.
When `host_domain < multimem_domain`, __several processes compose one multicast domain__ — the interesting cross-process case (NVL72 is `{4, 72, 144}`: 18 processes per rack-wide multicast domain).

## The two shapes, and why dual Hopper emulates the racks

- __`{8, 8, 16}` — works today.__
One process per node driving all 8 GPUs; `host_domain == multimem_domain`, so no cross-process multicast.
This is the straightforward 2-node launch.
- __`{4, 8, 16}` — the emulation target__ (`base-4-8-16.sbatch`).
Two processes per node, each 4 GPUs; the node's 8-GPU island is one `multimem_domain` __composed of two processes__ (cross-process multicast within the node), and the two nodes form the `colocated_domain` over RDMA.
This is the small-scale stand-in for the racks: it exercises __both tiers__ a GB200-NVL72 / Rubin deployment will use — intra-island cross-process multicast, then inter-island RDMA — so the mesh bootstrap is validated on Hopper before the Blackwell/Rubin hardware exists.

## Verified operational findings (2×HGX-H100)

- __UForm's `multimem_domain_t`, single-process, 8-GPU: PASS, `tier=switch_reduced`__ — real hardware multicast (not a peer-read fallback), distributed reduce matches the Float64 oracle.
- __Cross-process 2-GPU multicast via `POSIX_FILE_DESCRIPTOR` + a Unix-socket rendezvous: PASS__ (a reduce across a process boundary).
This is the handle type the `{4, 8, 16}` shape needs.
- __Cross-process via `CU_MEM_HANDLE_TYPE_FABRIC`: FAILS ("operation not permitted")__ even with `nvidia-imex` + `nvidia-fabricmanager` active — a standalone HGX box is a __standalone NVSwitch island__ (`GPU Fabric GUID = N/A`, `CliqueId = 0`), with no provisioned NVLink-_network_ fabric.
- __Topology:__ GPUs 0-3 on NUMA0 (cores 0-63, NIC4-7), GPUs 4-7 on NUMA1 (cores 64-127, NIC0-3).

__Takeaway:__ on standalone HGX, `FABRIC` handles are unavailable (they need a GB200-NVL72-class NVLink-network fabric); the same-node cross-process handle type is `POSIX_FILE_DESCRIPTOR`, and it works.
UForm's primitives (`cuda_core/multimem_collectives.cuh`) allocate with `POSIX_FILE_DESCRIPTOR`, the portable choice for same-node emulation; on real NVL72/144 racks the cross-process path becomes `FABRIC` and drops in once `nvidia-imex` puts every node in one IMEX domain — the mesh bootstrap is launcher-agnostic.

## Slurm best practices for the gang-scheduled training jobs

- __Gang scheduling.__
One allocation, a __single `srun`__ launches all tasks together.
The mesh's filesystem-rendezvous barrier assumes co-start; independently-scheduled tasks would hang.
(Slurm gives this for free; on vanilla Kubernetes it needs Volcano / Kueue / the Training-Operator.)
- __Task layout.__
`--ntasks-per-node` = host_domains per node, `--gpus-per-task` = host_domain size; GRES cgroup enforcement then pins each task's `CUDA_VISIBLE_DEVICES` to exactly its GPUs.
- __Host-domain index = `$SLURM_PROCID`, not `$SLURM_NODEID`.__
With 2 host_domains per node, `SLURM_NODEID` is `0,0,1,1` — wrong; the launcher exports `UFORM_HOST_DOMAIN_INDEX=$SLURM_PROCID` (`0,1,2,3`), which `mesh.cuh` reads first.
- __NUMA / NIC affinity__ via `--cpu-bind=cores --distribution=block:block` — task 0 → cores 0-63 (NUMA0, near GPUs 0-3 and NIC4-7), task 1 → cores 64-127.
- __`--exclusive` + `--no-kill`__ — own the node (tightly-coupled jobs never share GPUs); keep survivors up so the mesh can re-bootstrap a spare host_domain into a dead peer's index.
- __Env-var contract__ — `UFORM_GPUS_PER_MULTIMEM_DOMAIN`, `UFORM_EXCHANGE_DIR` (a shared-filesystem rendezvous path), `UFORM_JOB_ID=$SLURM_JOB_ID`, exported to every task.
