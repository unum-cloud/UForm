#!/usr/bin/env bash
# Cross-rank loader-sampling conformance at shape {host_domain, multimem_domain, colocated_domain} = {4,8,16}
#
# The distributed tier of test/loaders.cu: every rank (host_domain) bootstraps the mesh, draws its shard
# set from the shared dataset with the keyed weighted pick (seed,step,rank), publishes it to the filesystem
# rendezvous, and asserts its set differs from every peer's — the "different nodes sample different files"
# contract the single-process oracle can only assert as arithmetic. Same {4,8,16} shape as test_mesh-4-8-16.sbatch
# (a node runs TWO host_domains that share one 8-GPU multimem_domain; the two nodes form the colocated_domain
# over 4 host_domains); the rendezvous is filesystem-only (no InfiniBand). The job passes iff every rank passes.
#
# Site config is not baked in — pass the cluster-specific bits and the dataset glob at submit time:
#   sbatch --partition=<p> --account=<a> \
#          --export=ALL,UFORM_EXCHANGE_DIR=/mnt/shared/uform_rendezvous,UFORM_TEST_DATASET='/mnt/shared/tokens/*.parquet' \
#          scripts/slurm/test_loaders-4-8-16.sbatch
#SBATCH --job-name=loaders-4-8-16
#SBATCH --nodes=2                     # 2 NVLink islands = 2 multimem_domains
#SBATCH --ntasks=4                    # 4 host_domains total (colocated 16 / host 4)
#SBATCH --ntasks-per-node=2           # 2 host_domains/node -> the 8-GPU multimem_domain spans 2 PROCESSES
#SBATCH --gpus-per-task=4             # host_domain = 4 GPUs (mesh dim 0); cgroup masks each task to its 4
#SBATCH --cpus-per-task=64            # one NUMA node's cores per task (128 cores / 2 tasks)
#SBATCH --no-kill                     # keep survivors up if a rank dies (the mesh re-bootstraps a spare)
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# The sampling tier needs a GPU (the GDS session opens the dataset) but no InfiniBand — the rendezvous is
# filesystem-only, so a lean build suffices.
BINARY="${UFORM_TEST_BINARY:-build_release/uform_test_loaders}"
[[ -x "$BINARY" ]] || {
    echo "FATAL: $BINARY not built — cmake --build build_release --target uform_test_loaders"
    exit 1
}
: "${UFORM_TEST_DATASET:?set UFORM_TEST_DATASET to a shared-FS Parquet glob every rank samples}"

# Shared-filesystem rendezvous, visible from EVERY node; unique per job.
EXCHANGE_DIR="${UFORM_EXCHANGE_DIR:?set UFORM_EXCHANGE_DIR to a shared-FS path}/${SLURM_JOB_ID}"
mkdir -p "$EXCHANGE_DIR"

# ONE gang-scheduled srun: all 4 tasks co-start (the rendezvous barrier requires it). The host_domain index
# is the GLOBAL task rank SLURM_PROCID (0..3), NOT SLURM_NODEID (0,0,1,1 with 2 tasks/node); setting it
# activates the distributed sampling tier in test/loaders.cu.
srun --kill-on-bad-exit=1 --cpu-bind=cores --distribution=block:block \
    --export=ALL,UFORM_GPUS_PER_MULTIMEM_DOMAIN=8,UFORM_EXCHANGE_DIR="$EXCHANGE_DIR",UFORM_JOB_ID="$SLURM_JOB_ID",UFORM_HOST_DOMAIN_COUNT="$SLURM_NTASKS",UFORM_TEST_DATASET="$UFORM_TEST_DATASET" \
    bash -c '
        export UFORM_HOST_DOMAIN_INDEX="$SLURM_PROCID"   # 0..3 global task rank; activates the sampling tier
        echo "[host_domain $UFORM_HOST_DOMAIN_INDEX on $(hostname)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        exec "'"$BINARY"'"
    '
