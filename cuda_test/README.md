# `cuda_test/` — conformance suite

Plain `int main` binaries that check the primitives cell against Float64 oracles, byte-for-byte determinism, and — for the distributed cells — a real multi-node rendezvous.
Each file targets one library cell and shares one primitive: `harness.hpp`'s `test_suite_t`, threaded by reference into every case.
A green run prints one summary line; the only per-check output is a failure, which names its `file:line` and the actual-versus-expected operands (it reuses `expected::fail`'s `format_at_t` + `format_into` machinery).

## Tiers

Every distributed test has the same three tiers, gated by what the hardware offers:

- __single process__ — host logic and single-GPU conformance; always runs.
- __one process, many GPUs__ — a whole box; runs when the GPUs are visible.
- __many nodes__ — the `*.sbatch` runners below; the genuine happy path.

## What lives in a `.sbatch` vs here

The split is deliberate: a runner is __shape-only__ so it commits once and runs on any site; everything site-specific is passed at submit time or documented here.

__In the `.sbatch` file__ (portable, committed):

- the mesh __shape__ as `#SBATCH` geometry — `--nodes`, `--ntasks`, `--ntasks-per-node`, `--gpus-per-task` — and the filename encodes it (`mesh-4-8-16.sbatch` is `{host,multimem,colocated} = {4,8,16}`).
- the `srun` launch and the env wiring the binary reads: `UFORM_HOST_DOMAIN_INDEX=$SLURM_PROCID`, `UFORM_HOST_DOMAIN_COUNT=$SLURM_NTASKS`, `UFORM_GPUS_PER_MULTIMEM_DOMAIN`, `UFORM_JOB_ID`, `UFORM_EXCHANGE_DIR`.
- a preflight that fails loudly if the binary is missing.
- __no__ `--partition`, `--account`, or site paths — those are submit-time flags.

__Here in the README__ (site-specific, not committed into the runner):

- how to build and where to stage the binary so every node sees it.
- the shared-filesystem path for `UFORM_EXCHANGE_DIR`.
- the submit command with your partition/account.
- the cluster-standup notes and their gotchas.

## Running a distributed runner

The binary must sit on a filesystem every node sees, and `UFORM_EXCHANGE_DIR` must be genuinely shared.
`cuda_test_mesh` links only `libcuda.so.1`, so nothing else needs staging.

```
cmake --build build_release --target cuda_test_mesh
cp build_release/cuda_test_mesh /shared/uform/cuda_test_mesh
sbatch --chdir=/shared/uform --partition=<p> --account=<a> \
       --export=ALL,UFORM_EXCHANGE_DIR=/shared/uform/exchange,UFORM_TEST_BINARY=/shared/uform/cuda_test_mesh \
       cuda_test/mesh-4-8-16.sbatch
```

Every rank prints `PASS: mesh conformance (N/N checks passed)` into `mesh-4-8-16-<jobid>.out`; the job passes iff every rank passes.

## The per-task GPU-masking gotcha

A `{4,8,16}` runner puts __two__ host_domains (processes) on each node, each owning 4 of the 8 GPUs.
For that split to be real, Slurm must mask GPUs per task — `TaskPlugin=task/cgroup` + `cgroup.conf` `ConstrainDevices=yes`, and __no__ `--exclusive` (it hands every task all 8 GPUs, so each process sees `CUDA_VISIBLE_DEVICES=0..7` and the bootstrap silently builds a fictitious 8-GPU-per-process world).
Correct masking shows each task `CUDA_VISIBLE_DEVICES=0,1,2,3` (its own four, remapped) and the merged manifest reports `devices 4` per host_domain.
