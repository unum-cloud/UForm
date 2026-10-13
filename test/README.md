# `test/` — conformance suite

Plain `int main` binaries that check the primitives cell against Float64 oracles, byte-for-byte determinism, and — for the distributed cells — a real multi-node rendezvous.
Each file builds one `uform_test_<name>` binary, targets one library cell, and shares one primitive: `harness.hpp`'s `test_suite_t`, threaded by reference into every case.
A green run prints one summary line; the only per-check output is a failure, which names its `file:line` and the actual-versus-expected operands (it reuses `expected::fail`'s `format_at_t` + `format_into` machinery).

## Tiers

Every distributed test has the same three tiers, gated by what the hardware offers:

- __single process__ — host logic and single-GPU conformance; always runs.
- __one process, many GPUs__ — a whole box; runs when the GPUs are visible.
- __many nodes__ — the `scripts/slurm/test_*.sbatch` runners, documented in `scripts/slurm/README.md`; the genuine happy path.
