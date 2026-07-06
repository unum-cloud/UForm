#!/usr/bin/env bash
#
# acquire.sbatch — stage the whole training-data corpus in parallel across the Slurm cluster.
#
# A thin launcher over python_data/acquire_corpus.py. It grabs every idle node, spawns one worker per GPU
# (acquisition is CPU/IO-bound — the GPUs go unused, their count is just a convenient, hardware-matched
# fan-out), splits each node's cores among its workers, and runs the driver once per worker. Each worker
# reads its global rank ($SLURM_PROCID) and world ($SLURM_NTASKS) through sharding.resolve_rank_world, so
# shardable datasets stride shards[rank::world] and whole datasets are assigned one-per-worker — no leader,
# lock, or queue. Completion is per-shard atomic rename, so --requeue and plain re-submission are safe: a
# worker skips shards already committed and fetches only what is missing. Bump the node range to add
# workers; nothing coordinates. The disk budget flows from acquire_manifest (--budget-gb / $UFORM_DATA_BUDGET_GB
# / free-disk), so every dataset's cap is the multimodal-first allocation, not a flat fraction.
#
# Submit from the repository root; optionally pin the budget or narrow the modality:
#
#   sbatch python_data/acquire.sbatch
#   sbatch --export=ALL,BUDGET_GB=6000 python_data/acquire.sbatch
#   sbatch --export=ALL,MODALITY=video python_data/acquire.sbatch
#
# Single-machine is the identical code path with no Slurm env:
#
#   uv run python_data/acquire_corpus.py --modality all
#
#SBATCH --job-name=uform-acquire
#SBATCH --nodes=1-9999        # take every available node; the scheduler hands over what is idle
#SBATCH --exclusive           # own each node so all its cores are ours to split
#SBATCH --requeue             # per-shard atomic rename makes preempt / re-submit safe
#SBATCH --output=uform-acquire-%A.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
: "${UFORM_DATA_ROOT:?export UFORM_DATA_ROOT=<shared filesystem path visible to every node>}"

# Modality / budget knobs (via --export), and the yt-dlp cookies for the YouTube media scrapers
# (howto100m / vatex / activitynet / auto_acd) — without it those scrapers bot-wall (see acquire_clips.py).
# All three are exported so every srun worker inherits them.
export MODALITY="${MODALITY:-all}" BUDGET_GB="${BUDGET_GB:-}" UFORM_YTDLP_COOKIES="${UFORM_YTDLP_COOKIES:-}"

# One worker per GPU on each node; that node's cores split evenly among them (cgroup-pinned via cpu-bind).
# Homogeneous nodes assumed; falls back to 8 GPUs when the submit host has no GPU to probe.
gpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "${gpus}" -gt 0 ] 2>/dev/null || gpus=8
cores="${SLURM_CPUS_ON_NODE:-$(nproc)}"

# One driver per worker; it reads $SLURM_PROCID / $SLURM_NTASKS and stages exactly this worker's share.
srun --ntasks-per-node="${gpus}" --cpus-per-task=$(( cores / gpus )) --cpu-bind=cores bash -c '

    set -euo pipefail
    budget=(); [ -n "${BUDGET_GB}" ] && budget=(--budget-gb "${BUDGET_GB}")
    uv run python_data/acquire_corpus.py --modality "${MODALITY}" "${budget[@]}"

'
