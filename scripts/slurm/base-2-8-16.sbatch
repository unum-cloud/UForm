#!/usr/bin/env bash
#
# base-2-8-16.sbatch — train Base (tools/base.cu) on two DGX B300 nodes: one process per node driving its
# 8 GPUs over NVLink, the two nodes joined over InfiniBand, so the mesh is {8,8,16}:
#
#   sbatch --partition=<p> --account=<a> \
#          --export=ALL,UFORM_BASE_BIN=/shared/uform/uform_base,UFORM_EXCHANGE_DIR=/shared/uform/exchange \
#          scripts/slurm/base-2-8-16.sbatch
#
# The run reads its data from a mixture JSON and its losses from the objective; both ride the environment so
# one committed runner serves every rung of the ladder in scripts/slurm/README.md:
#
#   UFORM_MIXTURE          — a scripts/data/*.json mixture, tiny_text.json by default
#   UFORM_OBJECTIVE        — masked, contrastive or joint; masked by default
#   UFORM_STEPS            — optimizer steps, 200 by default
#   UFORM_CHECKPOINT_EVERY — steps between checkpoints beside the final one, 100 by default
#
# The rows per GPU are left to the measuring pass, which takes the most every GPU of both nodes holds.
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
export UFORM_EXCHANGE_DIR="${UFORM_EXCHANGE_DIR:-${SCRATCH:-/tmp}/uform-exchange}"
mkdir -p "${UFORM_EXCHANGE_DIR}"

mixture="${UFORM_MIXTURE:-scripts/data/tiny_text.json}"
objective="${UFORM_OBJECTIVE:-masked}"
echo "uform-base-2-8-16: ${UFORM_HOST_DOMAIN_COUNT} nodes x ${UFORM_GPUS_PER_MULTIMEM_DOMAIN} GPUs, ${objective} on ${mixture}"

# The preflight names every environment gap in one line per node before the run spends a minute on it.
env_binary="${UFORM_ENV_BINARY:-$(dirname "${base_binary}")/uform_test_env}"
if [[ -x "${env_binary}" ]]; then
    srun --ntasks="${UFORM_HOST_DOMAIN_COUNT}" --ntasks-per-node=1 --gpu-bind=none "${env_binary}"
fi

# One task per node owns all 8 of its GPUs, so it must see them all; UFORM_HOST_DOMAIN_INDEX is the task's rank.
srun --ntasks="${UFORM_HOST_DOMAIN_COUNT}" --ntasks-per-node=1 --gpu-bind=none \
    bash -c 'UFORM_HOST_DOMAIN_INDEX="${SLURM_PROCID}" exec stdbuf -oL "$@"' uform-base \
    "${base_binary}" \
    --mixture "${mixture}" \
    --objective "${objective}" \
    --steps "${UFORM_STEPS:-200}" \
    --checkpoint "${UFORM_CHECKPOINT:-base-2-8-16-${objective}-${UFORM_JOB_ID}.safetensors}" \
    --checkpoint-every "${UFORM_CHECKPOINT_EVERY:-100}" \
    --report-every "${UFORM_REPORT_EVERY:-10}"
