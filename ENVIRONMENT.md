# Environment Variables

Every UForm binary is configured through two channels with distinct owners.
Command-line flags carry _operator intent_ — the knobs one human sets on the one binary they run (`uform_base --mixture omni --eager`).
Environment variables carry the _launcher contract_ — the per-process facts a scheduler stamps on every rank of a distributed job, where `srun` broadcasts one identical command line and only the environment can differ per node.
Tests take no argv at all: their whole configuration is this contract plus the fixture paths below.

## The mesh contract

Read by the mains at the launcher boundary and passed into `mesh_topology_t::bootstrap` as explicit arguments — bootstrap itself never reads the environment, so a misconfigured launcher cannot change the math behind an already-constructed world.

| Variable | Default | Meaning |
| --- | --- | --- |
| `UFORM_HOST_DOMAIN_INDEX` | `SLURM_NODEID`, then `OMPI_COMM_WORLD_RANK`, then 0 | This process's node index in the world, via `host_domain_index_from_environment`. |
| `UFORM_HOST_DOMAIN_COUNT` | `SLURM_JOB_NUM_NODES`, then `OMPI_COMM_WORLD_SIZE`, then 1 | The world's node count, via `host_domain_count_from_environment`; 1 skips the InfiniBand tier entirely. |
| `UFORM_GPUS_PER_MULTIMEM_DOMAIN` | all visible GPUs | GPUs this process owns on its node — one process per node drives them all over NVLink. |
| `UFORM_JOB_ID` | `SLURM_JOB_ID` | The rendezvous namespace, minted once per job so two jobs on one filesystem never collide. |
| `UFORM_EXCHANGE_DIR` | — (required when nodes > 1) | Shared-filesystem directory the multi-node bootstrap and collectives rendezvous through. |

## Driver controls

| Variable | Set by | Meaning |
| --- | --- | --- |
| `CUDA_MODULE_LOADING=EAGER` | every main, before `cuInit` | Front-loads all kernel modules: under the default lazy loading, an InfiniBand doorbell kernel's flag spin blocks its context while the peer's next launch blocks on module loading behind it, and the two ranks deadlock. Set in-process so the binary is correct by construction, not by launch discipline. |
| `CUDA_VISIBLE_DEVICES` | the launcher / operator | The standard device mask; the two-domain single-box repro splits one node's GPUs into two pseudo-nodes with it. |

## Test fixtures and launch plumbing

| Variable | Consumer | Meaning |
| --- | --- | --- |
| `UFORM_TEST_IMAGE` / `UFORM_TEST_VIDEO` / `UFORM_TEST_AUDIO` / `UFORM_TEST_DATASET` | `cuda_test_loaders` | Media and Parquet fixture paths the loader conformance suite reads. |
| `UFORM_TEST_BINARY` | the `*.sbatch` scripts | The staged test executable to run (on a shared filesystem for multi-node jobs). |
| `UFORM_ENV_BINARY` | the `*.sbatch` scripts | The preflight probe; defaults to `cuda_test_env` beside `UFORM_TEST_BINARY`, and its failure aborts the job before the suite runs. |

## Adding a variable

New launcher-contract variables get a `*_from_environment` reader beside the two in `cuda_core/mesh.hpp`, a row here, and explicit pass-through into whatever they configure — no component below the mains should call `getenv` itself.
