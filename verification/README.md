# Verification

Model checking for the mesh protocols: the shared-filesystem rendezvous, the multimem domain's counting barrier, the windows its consumers stage and read, and the InfiniBand recursive doubling.
[Spin](https://spinroot.com) checks each protocol as a Promela model.
These protocols are GPU flags, RDMA writes and files rather than C++ atomics, so no GenMC client reads the code itself.

- `weak_memory.pml` is the C++ memory model as views, for Promela, under the models chosen by `-Dmemory=`: `sc`, one copy of every word, and `rc11`, the default, RC11's release-acquire-relaxed fragment.
- `check.sh` is the runner.

## Writing a Model

A model is one mechanism of the code under test, a flat `<name>.pml` that defines its shape, includes the memory model when its words are atomics, and starts its roles from one `init { atomic { run <role>(<index>); … } }`.
There is no conditional compilation: a knob is a choice the code makes, defaulted under `#ifndef` and read as the guard of a plain `if` with an `else` branch.

Every expected verdict is a `@verify` line in the docblock of the file it checks:

```text
@verify <pass|fail|stuck> <memory>[,<memory>...] [<knob>=<value> ...][: <finding>]
```

`pass` is a search that completed with no error, `fail` a violated assertion, and `stuck` an invalid end state; a search cut short or a knob defined twice is `broken`.
Every model has a `pass` line, every knob appears on some line, and adding a model, a knob or a finding never edits `check.sh`.
Run every line with `verification/check.sh`, or one model with `verification/check.sh window_reuse.pml`.

## What the Models Found

Each finding is the `@verify` line that replays it, where the full counterexample is written out.

- Every wiring phase keys its payloads and barriers with a generation the topology hands out, or the second multimem domain of a run reads the first one's slice handles and passes its stale release: [`mesh_rendezvous.pml`](mesh_rendezvous.pml).
- The distributed contrastive loss crosses a barrier after its reduce reads its peers' windows, or a loss run twice with no optimizer step between restages a window a peer still reads; the mixer has no such barrier and relies on the optimizer's: [`window_reuse.pml`](window_reuse.pml).
- Every member crosses every consumer's barrier, in one order, since the barrier counts arrivals and carries no label, and the arrival is a release reduction for the spin's acquire: [`multimem_barrier.pml`](multimem_barrier.pml).
- The InfiniBand all-reduce lands into the landing of its step's parity, while the all-gather lands into its payload with no parity and needs a later exchange before its next step, which the trainer's force and gradient all-reduces give: [`gdaki_doubling.pml`](gdaki_doubling.pml).
