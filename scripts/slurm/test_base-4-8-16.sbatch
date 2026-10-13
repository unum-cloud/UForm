#!/usr/bin/env bash
#
# test_base-4-8-16.sbatch — run the Base training conformance test (test/base.cu) at 4, 8, or 16 GPUs,
# unchanged binary from a single NVLink node to a two-node InfiniBand world:
#
#   4 GPUs : sbatch --nodes=1 --gpus-per-node=4  scripts/slurm/test_base-4-8-16.sbatch
#   8 GPUs : sbatch --nodes=1 --gpus-per-node=8  scripts/slurm/test_base-4-8-16.sbatch
#   16 GPUs: sbatch --nodes=2 --gpus-per-node=8  scripts/slurm/test_base-4-8-16.sbatch
#
# One process per host_domain (node) drives all of its node's GPUs over NVLink (the multimem_domain);
# two nodes reduce over InfiniBand (the colocated_domain), which only wakes up when the node count is
# above one. The test's main reads the same mesh env contract the training binary does —
#
#   UFORM_HOST_DOMAIN_COUNT        — nodes; 1 skips the InfiniBand rendezvous entirely
#   UFORM_GPUS_PER_MULTIMEM_DOMAIN — GPUs this process owns on its node
#   UFORM_JOB_ID / UFORM_EXCHANGE_DIR — the multi-node rendezvous namespace + shared directory
#
# so the two-node tier exercises the InfiniBand reduce under every case: the replay conformance for both
# mixtures, the media-subset x row-budget sweep, the capture checks, and the gradient propagation. The
# job passes iff every host_domain's suite passes.
#
#SBATCH --job-name=uform-test-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=uform-test-base-%j.out

set -euo pipefail

host_domain_count="${SLURM_JOB_NUM_NODES:-1}"
gpus_per_multimem_domain="${SLURM_GPUS_ON_NODE:-8}"

export UFORM_HOST_DOMAIN_COUNT="${host_domain_count}"
export UFORM_GPUS_PER_MULTIMEM_DOMAIN="${gpus_per_multimem_domain}"

# Multi-node worlds rendezvous through a shared-filesystem directory keyed by a job id minted once here;
# single-node runs touch neither (bootstrap short-circuits at host_domain_count == 1).
export UFORM_JOB_ID="${UFORM_JOB_ID:-${SLURM_JOB_ID:-$(date +%s)}}"
if [[ "${host_domain_count}" -gt 1 ]]; then
    export UFORM_EXCHANGE_DIR="${UFORM_EXCHANGE_DIR:-${SCRATCH:-/tmp}/uform-exchange/${UFORM_JOB_ID}}"
    mkdir -p "${UFORM_EXCHANGE_DIR}"
fi

total_gpus=$(( host_domain_count * gpus_per_multimem_domain ))
echo "uform-test-base: ${total_gpus} GPU(s) = ${host_domain_count} node(s) x ${gpus_per_multimem_domain} GPU(s)"

# One task per node; that process owns all of its node's GPUs, so it must see them all (no per-task GPU
# mask). Slurm sets SLURM_NODEID per task, which the test reads as its host_domain_index.
#
# The preflight probe runs first: uform_test_env names each environment gap (library resolution, GPUDirect
# registration, the rendezvous directory) in one line per node; under `set -e` a failing probe aborts the
# job before the suite spends a minute discovering the gap mid-run.
env_binary="${UFORM_ENV_BINARY:-$(dirname "${UFORM_TEST_BINARY:-build_release/uform_test_base}")/uform_test_env}"
if [[ -x "${env_binary}" ]]; then
    srun --ntasks="${host_domain_count}" --ntasks-per-node=1 --gpu-bind=none "${env_binary}"
fi

# Line-buffered so a hung run shows how far it got instead of a silent block-buffered void; the optional
# UFORM_TEST_ONLY=tokens|omni|sweep fast lane rides through --export like the rest of the contract.
srun --ntasks="${host_domain_count}" --ntasks-per-node=1 --gpu-bind=none \
    stdbuf -oL "${UFORM_TEST_BINARY:-build_release/uform_test_base}"
