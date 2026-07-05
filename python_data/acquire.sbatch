#!/usr/bin/env bash
#
# acquire.sbatch — acquire the whole training-data corpus in parallel across the entire Slurm cluster.
#
# Unlike the gang-scheduled mesh / training jobs (see cuda_train/README.md), this is embarrassingly
# parallel and shape-AGNOSTIC: it grabs every idle node, spawns one worker per GPU on each (acquisition is
# CPU/IO-bound — the GPUs go unused, their count is just a convenient, hardware-matched fan-out), and
# splits each node's cores evenly among its workers. The global worker rank is $SLURM_PROCID and the world
# is $SLURM_NTASKS; sharding.resolve_rank_world() reads exactly those, so every worker pulls
# shards[rank::world] — a disjoint stride, no leader, lock, or queue. Completion is per-shard atomic
# rename, so --requeue and plain re-submission are safe: a worker skips shards already committed and
# fetches only what is missing from its own partition. Bump the node range to add workers; nothing
# coordinates.
#
# Submit from the repository root, optionally capping volume cluster-wide with FRACTION:
#
#   sbatch python_data/acquire.sbatch
#   sbatch --export=ALL,FRACTION=0.35 python_data/acquire.sbatch
#
# Single-machine is the identical code path with no Slurm env:
#
#   uv run python_data/download_<name>.py
#
#SBATCH --job-name=uform-acquire
#SBATCH --nodes=1-9999        # take every available node; the scheduler hands over what is idle
#SBATCH --exclusive           # own each node so all its cores are ours to split
#SBATCH --requeue             # per-shard atomic rename makes preempt / re-submit safe
#SBATCH --output=uform-acquire-%A.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
: "${UFORM_DATA_ROOT:?export UFORM_DATA_ROOT=<shared filesystem path visible to every node>}"

# The two work-lists, computed once on the batch node and exported to every worker. Shardable downloaders
# rank-partition their shards across all workers; whole-dataset downloaders cannot sub-shard, so they are
# handed out one-per-worker, round-robin. Derived from the download_*.py scripts, never hardcoded here.
SHARDABLE="$(uv run python_data/list_registry.py --acquire-shardable | tr '\n' ' ')"
WHOLE="$(uv run python_data/list_registry.py --acquire-whole | tr '\n' ' ')"
export SHARDABLE WHOLE FRACTION="${FRACTION:-}"

# One worker per GPU on each node; that node's cores split evenly among them (cgroup-pinned via cpu-bind).
# Homogeneous nodes assumed; falls back to 8 GPUs when the submit host has no GPU to probe.
gpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "${gpus}" -gt 0 ] 2>/dev/null || gpus=8
cores="${SLURM_CPUS_ON_NODE:-$(nproc)}"

srun --ntasks-per-node="${gpus}" --cpus-per-task=$(( cores / gpus )) --cpu-bind=cores bash -c '

    set -euo pipefail
    cap=(); [ -n "${FRACTION}" ] && cap=(--fraction "${FRACTION}")

    # shardable: every worker runs each; acquire() takes shards[$SLURM_PROCID::$SLURM_NTASKS]
    for ds in ${SHARDABLE}; do
        uv run "python_data/download_${ds}.py" "${cap[@]}"
    done

    # whole: round-robin, one worker per dataset
    i=0
    for ds in ${WHOLE}; do
        [ $(( i++ % SLURM_NTASKS )) -eq "${SLURM_PROCID}" ] && uv run "python_data/download_${ds}.py"
    done

'
