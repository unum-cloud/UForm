#!/usr/bin/env bash
#
# base-4-8-16.sbatch — launch distributed span-masked Base pretraining (cuda_train/base.cu) at 4, 8, or
# 16 GPUs, unchanged binary from a single NVLink node to a two-node InfiniBand world.
#
# The model is data-parallel: ONE process per host_domain (node), and that process drives every GPU on
# its node over NVLink (the multimem_domain). Nodes reduce over InfiniBand (the colocated_domain), which
# only wakes up when host_domain_count > 1. So the three tiers are just allocation shapes:
#
#   4 GPUs : sbatch --nodes=1 --gpus-per-node=4  cuda_train/base-4-8-16.sbatch
#   8 GPUs : sbatch --nodes=1 --gpus-per-node=8  cuda_train/base-4-8-16.sbatch
#   16 GPUs: sbatch --nodes=2 --gpus-per-node=8  cuda_train/base-4-8-16.sbatch
#
# The body reads the allocation Slurm actually granted (SLURM_JOB_NUM_NODES, SLURM_GPUS_ON_NODE) and
# wires the mesh env contract base.cu / mesh_topology_t::bootstrap read:
#
#   UFORM_HOST_DOMAIN_COUNT        — nodes; 1 skips the InfiniBand rendezvous entirely
#   UFORM_GPUS_PER_MULTIMEM_DOMAIN — GPUs this process owns on its node
#   UFORM_JOB_ID / UFORM_EXCHANGE_DIR — the multi-node rendezvous namespace + shared directory
#
# host_domain_index is resolved per process from SLURM_NODEID (host_domain_index_from_environment).
#
#SBATCH --job-name=uform-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=uform-base-%j.out

set -euo pipefail

host_domain_count="${SLURM_JOB_NUM_NODES:-1}"
gpus_per_multimem_domain="${SLURM_GPUS_ON_NODE:-8}"

export UFORM_HOST_DOMAIN_COUNT="${host_domain_count}"
export UFORM_GPUS_PER_MULTIMEM_DOMAIN="${gpus_per_multimem_domain}"

# Multi-node worlds rendezvous through a shared-filesystem directory keyed by a job id minted ONCE here;
# single-node runs touch neither (bootstrap short-circuits at host_domain_count == 1).
export UFORM_JOB_ID="${UFORM_JOB_ID:-${SLURM_JOB_ID:-$(date +%s)}}"
if [[ "${host_domain_count}" -gt 1 ]]; then
    export UFORM_EXCHANGE_DIR="${UFORM_EXCHANGE_DIR:-${SCRATCH:-/tmp}/uform-exchange/${UFORM_JOB_ID}}"
    mkdir -p "${UFORM_EXCHANGE_DIR}"
fi

total_gpus=$(( host_domain_count * gpus_per_multimem_domain ))
echo "uform-base: ${total_gpus} GPU(s) = ${host_domain_count} node(s) x ${gpus_per_multimem_domain} GPU(s)"

# One task per node; that process owns all of its node's GPUs, so it must SEE them all (no per-task GPU
# mask). Slurm sets SLURM_NODEID per task, which base.cu reads as its host_domain_index.
srun --ntasks="${host_domain_count}" --ntasks-per-node=1 --gpu-bind=none \
    "${UFORM_BASE_BIN:-./build/uform_base}" \
    --hidden "${UFORM_HIDDEN:-512}" \
    --depth "${UFORM_DEPTH:-12}" \
    --rows "${UFORM_ROWS:-8}" \
    --columns "${UFORM_COLUMNS:-2048}" \
    --steps "${UFORM_STEPS:-1000}" \
    --checkpoint "${UFORM_CHECKPOINT:-base-${host_domain_count}x${gpus_per_multimem_domain}.safetensors}"
