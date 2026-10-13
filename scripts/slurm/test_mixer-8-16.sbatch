#!/usr/bin/env bash
#
# test_mixer-8-16.sbatch — run the batch-mixer conformance test (test/mixer.cu) at 8 or 16 GPUs,
# unchanged binary from a single NVLink node to a two-node InfiniBand world:
#
#   8 GPUs : sbatch --nodes=1 --gpus-per-node=8 scripts/slurm/test_mixer-8-16.sbatch
#   16 GPUs: sbatch --nodes=2 --gpus-per-node=8 scripts/slurm/test_mixer-8-16.sbatch
#
# One process per host_domain (node) drives all of its node's GPUs over NVLink; two nodes add the
# InfiniBand full-batch exchange (infiniband_all_to_all) so every mixed segment can source from any of
# the 16 global GPUs. The test's main reads the same mesh env contract the base suite does:
#
#   UFORM_HOST_DOMAIN_COUNT        — nodes; 1 skips the InfiniBand exchange entirely
#   UFORM_GPUS_PER_MULTIMEM_DOMAIN — GPUs this process owns on its node
#   UFORM_JOB_ID / UFORM_EXCHANGE_DIR — the multi-node rendezvous namespace + shared directory
#
# The job passes iff every host_domain's suite passes.
#
#SBATCH --job-name=uform-test-mixer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=uform-test-mixer-%j.out

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
echo "uform-test-mixer: ${total_gpus} GPU(s) = ${host_domain_count} node(s) x ${gpus_per_multimem_domain} GPU(s)"

# The preflight probe runs first: uform_test_env names each environment gap (library resolution, GPUDirect
# registration, the rendezvous directory) in one line per node; under `set -e` a failing probe aborts the
# job before the suite spends a minute discovering the gap mid-run.
env_binary="${UFORM_ENV_BINARY:-$(dirname "${UFORM_TEST_BINARY:-build_release/uform_test_mixer}")/uform_test_env}"
if [[ -x "${env_binary}" ]]; then
    srun --ntasks="${host_domain_count}" --ntasks-per-node=1 --gpu-bind=none "${env_binary}"
fi

# One task per node; that process owns all of its node's GPUs, so it must see them all (no per-task GPU
# mask). Slurm sets SLURM_NODEID per task, which the test reads as its host_domain_index.
srun --ntasks="${host_domain_count}" --ntasks-per-node=1 --gpu-bind=none \
    "${UFORM_TEST_BINARY:-build_release/uform_test_mixer}"
