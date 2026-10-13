# Environment Variables

Every UForm binary is configured through two channels with distinct owners.
Command-line flags carry _operator intent_ — the knobs one human sets on the one binary they run (`uform_base --mixture omni --eager`).
Environment variables carry the _launcher contract_ — the per-process facts a scheduler stamps on every rank of a distributed job, where `srun` broadcasts one identical command line and only the environment can differ per node.
Tests take no argv at all: their whole configuration is this contract plus the fixture paths below.

## The mesh contract

Read by the mains at the launcher boundary and passed into `mesh_topology_t::bootstrap` as explicit arguments — bootstrap itself never reads the environment, so a misconfigured launcher cannot change the math behind an already-constructed world.

- `UFORM_HOST_DOMAIN_INDEX`: this process's node index in the world, via `host_domain_index_from_environment`.
  Defaults to `SLURM_NODEID`, then `OMPI_COMM_WORLD_RANK`, then 0.
- `UFORM_HOST_DOMAIN_COUNT`: the world's node count, via `host_domain_count_from_environment`; 1 skips the InfiniBand tier entirely.
  Defaults to `SLURM_JOB_NUM_NODES`, then `OMPI_COMM_WORLD_SIZE`, then 1.
- `UFORM_GPUS_PER_MULTIMEM_DOMAIN`: GPUs this process owns on its node — one process per node drives them all over NVLink.
  Defaults to all visible GPUs.
- `UFORM_JOB_ID`: the rendezvous namespace, minted once per job so two jobs on one filesystem never collide.
  Defaults to `SLURM_JOB_ID`.
- `UFORM_EXCHANGE_DIR`: the shared-filesystem directory the multi-node bootstrap and collectives rendezvous through.
  Has no default and is required when nodes > 1.

## Driver controls

- `CUDA_MODULE_LOADING=EAGER`, set by every main before `cuInit`: front-loads all kernel modules.
  Under the default lazy loading, an InfiniBand doorbell kernel's flag spin blocks its context while the peer's next launch blocks on module loading behind it, and the two ranks deadlock.
  Set in-process so the binary is correct by construction, not by launch discipline.
- `CUDA_VISIBLE_DEVICES`, set by the launcher or operator: the standard device mask.
  The two-domain single-box repro splits one node's GPUs into two pseudo-nodes with it.

## Test fixtures and launch plumbing

- `UFORM_TEST_IMAGE`, `UFORM_TEST_VIDEO`, `UFORM_TEST_AUDIO`, `UFORM_TEST_DATASET`: media and Parquet fixture paths the `uform_test_loaders` conformance suite reads.
- `UFORM_TEST_BINARY`: the staged test executable the `scripts/slurm/test_*.sbatch` runners launch, on a shared filesystem for multi-node jobs.
- `UFORM_ENV_BINARY`: the preflight probe those runners launch first, defaulting to `uform_test_env` beside `UFORM_TEST_BINARY`.
  Its failure aborts the job before the suite runs.

## Adding a variable

New launcher-contract variables get a `*_from_environment` reader beside the two in `src/core/mesh.hpp`, an entry here, and explicit pass-through into whatever they configure — no component below the mains should call `getenv` itself.
