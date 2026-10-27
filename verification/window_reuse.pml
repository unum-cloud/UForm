/**
 *  @file verification/window_reuse.pml
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Two members staging a multimem window, reading each other's after a domain barrier, and
 *      restaging it in the next step.
 *
 *  `multimem_domain_t` from `include/uform/utilities/multimem_collectives.cuh` gives every consumer a window in
 *  each member's replica, and the consumers of `src/train/contrastive_segments.cu`,
 *  `src/encoder/multimodal_batch.cu` and `src/train/optimizer.cu` share one shape: stage into the own
 *  window, cross the domain barrier, read the peers' windows. The next step stages into the same
 *  window, so some barrier must separate a peer's reads from the restage. The barrier is the
 *  counting one of `multimem_barrier_peer_kernel_`: every arrival adds one to every member's flag,
 *  and a member leaves once its own flag holds every arrival of the round.
 *
 *  The distributed contrastive loss crosses a second barrier after its reduce reads, so it may run
 *  twice with nothing between. The mixer reads its windows with no trailing barrier and relies on
 *  the optimizer's barriers later in the trainer's step, which the last line pins.
 *
 *  @verify pass sc
 *  @verify fail sc trailing_barrier=false: the contrastive loss crosses a barrier after its reduce
 *      reads its peers' windows; without it, a loss run twice with no optimizer step between
 *      restages a window a peer still reads
 *  @verify pass sc trailing_barrier=false optimizer_between=true: the optimizer's barriers order
 *      the mixer's window reads before the next step restages, so the trainer needs no trailing
 *      barrier in the mixer
 */

// The knobs: the consumer's own trailing barrier, and another consumer's barrier between steps.
#ifndef trailing_barrier
#define trailing_barrier true
#endif
#ifndef optimizer_between
#define optimizer_between false
#endif

#define members 2
#define steps 2

byte window[members]; // the step each member's window was last staged for, plus one
byte flag[members];   // each member's arrival counter

// one counting barrier: an arrival on every member's flag, then a wait on the own one
inline cross_barrier(member, target) {
    atomic {
        flag[0] = flag[0] + 1;
        flag[1] = flag[1] + 1
    };
    target = target + members;
    (flag[member] >= target)
}

proctype consumer(byte member) {
    byte step = 0, target = 0, observed = 0;
    do
    :: step == steps -> break
    :: else ->
        window[member] = step + 1;
        cross_barrier(member, target);
        observed = window[1 - member];
        assert(observed == step + 1);
        if
        :: trailing_barrier -> cross_barrier(member, target)
        :: else
        fi;
        if
        :: optimizer_between -> cross_barrier(member, target)
        :: else
        fi;
        step++
    od
}

init {
    atomic { run consumer(0); run consumer(1) }
}
