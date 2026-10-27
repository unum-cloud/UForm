# Verification

Model checking for the mesh protocols, each a Promela model checked with [Spin](https://spinroot.com).
These protocols are GPU flags, RDMA writes and files rather than C++ atomics, so no GenMC client reads the code itself.

- `mesh_rendezvous.pml`: two processes wiring one phase twice over the shared exchange directory.
- `multimem_barrier.pml`: two members and two consumers sharing the one counting barrier of a multimem domain.
- `window_reuse.pml`: two members staging a multimem window, reading each other's after a domain barrier, and restaging it in the next step.
- `gdaki_doubling.pml`: two multimem domains running one recursive-doubling round over InfiniBand, step after step.
- `gateway_ordering.pml`: two hosts of a gateway and a peer each, running the trainer's step three times.
- `probe_teardown.pml`: two hosts wiring a queue pair per probe generation and destroying it before the next generation wires its own.

## Running

`verification/check.sh` runs every `@verify` line through ForkUnion's runner, from the `forkunion` submodule or from the checkout `FORKUNION_SOURCE` names.
Run one model with `verification/check.sh window_reuse.pml`.
[ForkUnion's verification README](../forkunion/verification/README.md) documents the runner, the `@verify` grammar and the memory budgets.

## Model Findings

- Every wiring phase keys its payloads and barriers with a generation the topology hands out, or the second multimem domain of a run reads the first one's slice handles and passes its stale release: [`mesh_rendezvous.pml`](mesh_rendezvous.pml).
- The distributed contrastive loss crosses a barrier after its reduce reads its peers' windows, or a loss run twice with no optimizer step between restages a window a peer still reads; the mixer has no such barrier and relies on the optimizer's: [`window_reuse.pml`](window_reuse.pml).
- Every member crosses every consumer's barrier, in one order, since the barrier counts arrivals and carries no label, and the arrival is a release reduction for the spin's acquire: [`multimem_barrier.pml`](multimem_barrier.pml).
- The InfiniBand all-reduce lands into the landing of its step's parity, while the all-gather lands into its payload with no parity and needs a later exchange before its next step, which the trainer's force and gradient all-reduces give: [`gdaki_doubling.pml`](gdaki_doubling.pml).
- Every peer reads the gateway's world mean after the optimizer's second barrier, and the mixer's barrier already orders the gateway's restage after those reads, which the first barrier alone must do in a step with no mixer and no contrasts: [`gateway_ordering.pml`](gateway_ordering.pml).
- A host destroys a probe's queue pair only after a host barrier, since its NIC may acknowledge the partner's last write after the host saw it land, and the barrier suffices because every post kernel drains its own write's completion: [`probe_teardown.pml`](probe_teardown.pml).
