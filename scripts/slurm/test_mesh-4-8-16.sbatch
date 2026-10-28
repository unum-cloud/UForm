#!/usr/bin/env bash
# Mesh rendezvous conformance at shape {host_domain, multimem_domain, colocated_domain} = {4,8,16}
#
# The cluster tier of test/mesh.cpp, at the shape where the host_domain (4 GPUs, one process) is
# strictly smaller than the multimem_domain (8 GPUs, a whole NVLink island): a node runs TWO host_domains
# that share one multimem_domain, and the two nodes form the colocated_domain over 4 host_domains. This is
# the shape that exercises the distinctions {8,8,16} collapses away — per-task GPU masking, the
# first_gpu_index() local<->global mapping, and multimem-extent agreement across processes. The rendezvous
# is filesystem-only (no InfiniBand); the job passes iff every rank passes.
#
# Site config is not baked in (keeps this file shape-only) — pass the cluster-specific bits at submit time:
#   sbatch --partition=<p> --account=<a> \
#          --export=ALL,UFORM_EXCHANGE_DIR=/mnt/shared/uform_rendezvous \
#          scripts/slurm/test_mesh-4-8-16.sbatch
#SBATCH --job-name=mesh-4-8-16
#SBATCH --nodes=2                     # 2 NVLink islands = 2 multimem_domains
#SBATCH --ntasks=4                    # 4 host_domains total (colocated 16 / host 4)
#SBATCH --ntasks-per-node=2           # 2 host_domains/node -> the 8-GPU multimem_domain spans 2 PROCESSES
#SBATCH --gpus-per-task=4             # host_domain = 4 GPUs (mesh dim 0); cgroup masks each task to its 4
#SBATCH --cpus-per-task=64            # one NUMA node's cores per task (128 cores / 2 tasks)
#SBATCH --no-kill                     # keep survivors up if a rank dies (the mesh re-bootstraps a spare)
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# The mesh test needs no InfiniBand — the rendezvous is filesystem-only, so a lean build suffices.
BINARY="${UFORM_TEST_BINARY:-build_cuda/uform_test_mesh}"
[[ -x "$BINARY" ]] || {
    echo "FATAL: $BINARY not built — cmake --build build_cuda --target uform_test_mesh"
    exit 1
}

# Shared-filesystem rendezvous, visible from EVERY node; unique per job.
EXCHANGE_DIR="${UFORM_EXCHANGE_DIR:?set UFORM_EXCHANGE_DIR to a shared-FS path}/${SLURM_JOB_ID}"
mkdir -p "$EXCHANGE_DIR"

# ONE gang-scheduled srun: all 4 tasks co-start (the rendezvous barrier requires it). The host_domain
# index is the GLOBAL task rank SLURM_PROCID (0..3), NOT SLURM_NODEID (0,0,1,1 with 2 tasks/node). The
# multimem extent of 8 GPUs spans a node's two processes, so it is asserted explicitly, not probed.
srun --kill-on-bad-exit=1 --cpu-bind=cores --distribution=block:block \
    --export=ALL,UFORM_GPUS_PER_MULTIMEM_DOMAIN=8,UFORM_EXCHANGE_DIR="$EXCHANGE_DIR",UFORM_JOB_ID="$SLURM_JOB_ID",UFORM_HOST_DOMAIN_COUNT="$SLURM_NTASKS" \
    bash -c '
        export UFORM_HOST_DOMAIN_INDEX="$SLURM_PROCID"   # 0..3 global task rank; activates the cluster tier
        echo "[host_domain $UFORM_HOST_DOMAIN_INDEX on $(hostname)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        exec "'"$BINARY"'"
    '
