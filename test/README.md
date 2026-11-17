# `test/` — conformance suite

Plain `int main` binaries that check the primitives cell against Float64 oracles, byte-for-byte determinism, and — for the distributed cells — a real multi-node rendezvous.
Each file builds one `uform_test_<name>` binary and targets one library cell, sharing `harness.hpp`, whose one mutable global, `test_run`, is the run in progress.
How to run, filter and write them, and the variables they read, are in [CONTRIBUTING.md](../CONTRIBUTING.md#test-suites).

## Tiers

Every distributed test has the same three tiers, gated by what the hardware offers:

- __single process__ — host logic and single-GPU conformance; always runs.
- __one process, many GPUs__ — a whole box; runs when the GPUs are visible.
- __many nodes__ — the `scripts/slurm/test_*.sbatch` runners, documented in `scripts/slurm/README.md`; the genuine happy path.
