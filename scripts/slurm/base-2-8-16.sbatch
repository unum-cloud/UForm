#!/usr/bin/env bash
#
# base-2-8-16.sbatch — train Base (tools/base.cu) on two DGX B300 nodes: one process per node driving its
# 8 GPUs over NVLink, the two nodes joined over InfiniBand, so the mesh is {8,8,16}:
#
#   sbatch --partition=<p> --account=<a> \
#          --export=ALL,UFORM_BASE_BIN=/shared/uform/uform_base,UFORM_EXCHANGE_DIR=/shared/uform/exchange,UFORM_TIER=tiny,UFORM_MIXTURE=scripts/mixtures/text_contrastive.json \
#          scripts/slurm/base-2-8-16.sbatch
#
# The run reads its shape from the tier, its data from a mixture JSON and its losses from the objective; all
# ride the environment so one committed runner serves every rung of the ladder in scripts/slurm/README.md:
#
#   UFORM_TIER             — tiny, small, medium, large or xl; required
#   UFORM_MIXTURE          — a scripts/mixtures/*.json mixture; required
#   UFORM_OBJECTIVE        — masked, contrastive or joint; contrastive by default
#   UFORM_STEPS            — optimizer steps, 200 by default
#   UFORM_CHECKPOINT_EVERY — steps between checkpoints beside the final one, 100 by default
#   UFORM_ROWS             — rows per GPU, left to the measuring pass when unset
#   UFORM_RESUME           — a checkpoint on the shared filesystem to resume from
#   UFORM_NVDEC_ENGINES    — the NVDEC engines per GPU, which a video mixture needs
#
# Unset rows are left to the measuring pass, which takes the most every GPU of both nodes holds.
#
#SBATCH --job-name=uform-base-2-8-16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --no-kill
#SBATCH --time=02:00:00
#SBATCH --output=uform-base-2-8-16-%j.out

set -euo pipefail

base_binary="${UFORM_BASE_BIN:-build_release/uform_base}"
if [[ ! -x "${base_binary}" ]]; then
    echo "uform-base-2-8-16: no executable at ${base_binary}; stage it on the shared filesystem" >&2
    exit 1
fi

export UFORM_HOST_DOMAIN_COUNT="${SLURM_JOB_NUM_NODES:-2}"
export UFORM_GPUS_PER_MULTIMEM_DOMAIN="${SLURM_GPUS_ON_NODE:-8}"
export UFORM_JOB_ID="${UFORM_JOB_ID:-${SLURM_JOB_ID:-$(date +%s)}}"
# Both nodes rendezvous through files, so the directory must be shared; a node-local /tmp never meets.
if [[ -z "${UFORM_EXCHANGE_DIR:-}" && -z "${SCRATCH:-}" ]]; then
    echo "uform-base-2-8-16: set UFORM_EXCHANGE_DIR or SCRATCH to a directory both nodes share" >&2
    exit 1
fi
export UFORM_EXCHANGE_DIR="${UFORM_EXCHANGE_DIR:-${SCRATCH}/uform-exchange}"
mkdir -p "${UFORM_EXCHANGE_DIR}"

optional=()
if [[ -n "${UFORM_ROWS:-}" ]]; then optional+=(--rows "${UFORM_ROWS}"); fi
if [[ -n "${UFORM_RESUME:-}" ]]; then optional+=(--resume "${UFORM_RESUME}"); fi
if [[ -n "${UFORM_NVDEC_ENGINES:-}" ]]; then optional+=(--nvdec-engines "${UFORM_NVDEC_ENGINES}"); fi

tier="${UFORM_TIER:?set UFORM_TIER to tiny|small|medium|large|xl}"
mixture="${UFORM_MIXTURE:?set UFORM_MIXTURE to a scripts/mixtures/*.json}"
objective="${UFORM_OBJECTIVE:-contrastive}"
echo "uform-base-2-8-16: ${UFORM_HOST_DOMAIN_COUNT} nodes x ${UFORM_GPUS_PER_MULTIMEM_DOMAIN} GPUs, ${tier} ${objective} on ${mixture}"

# The preflight names every environment gap in one line per node before the run spends a minute on it.
env_binary="${UFORM_ENV_BINARY:-$(dirname "${base_binary}")/uform_test_env}"
if [[ -x "${env_binary}" ]]; then
    srun --ntasks="${UFORM_HOST_DOMAIN_COUNT}" --ntasks-per-node=1 --gpu-bind=none "${env_binary}"
fi

# One task per node owns all 8 of its GPUs, so it must see them all; UFORM_HOST_DOMAIN_INDEX is the task's rank.
srun --ntasks="${UFORM_HOST_DOMAIN_COUNT}" --ntasks-per-node=1 --gpu-bind=none \
    bash -c 'UFORM_HOST_DOMAIN_INDEX="${SLURM_PROCID}" exec stdbuf -oL "$@"' uform-base \
    "${base_binary}" \
    --tier "${tier}" \
    --mixture "${mixture}" \
    --objective "${objective}" \
    --steps "${UFORM_STEPS:-200}" \
    --checkpoint "${UFORM_CHECKPOINT:-base-2-8-16-${objective}-${UFORM_JOB_ID}.safetensors}" \
    --checkpoint-every "${UFORM_CHECKPOINT_EVERY:-100}" \
    --report-every "${UFORM_REPORT_EVERY:-10}" \
    "${optional[@]}"
