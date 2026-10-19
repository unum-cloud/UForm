/**
 *  @file verification/gdaki_doubling.pml
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Two multimem domains running one recursive-doubling round over InfiniBand, step after
 *      step, with no other exchange between the steps.
 *
 *  `infiniband_reach_t` from `src/core/infiniband_collectives.cu` posts its block into the
 *  partner's heap, then the partner's arrival flag, on one queue pair, so the payload lands before
 *  the flag. The partner waits for the flag of this step and folds or copies out what landed. The
 *  all-reduce lands into one of two landings picked by the step's parity, so a partner one step
 *  ahead writes the other landing; the all-gather lands straight into the payload it gathers, with
 *  no parity, so its caller must not let a partner post the next step before this one is copied
 *  out. The trainer meets that through its later exchanges: the contrastive force all-reduce and
 *  the gateway's gradient all-reduce, each ordered after the copy on its own stream.
 *
 *  @verify pass sc
 *  @verify fail sc parity=false: the all-reduce lands into the landing of its step's parity; one
 *      landing lets a partner a step ahead overwrite a block before it is folded
 *  @verify fail sc gather=true: the all-gather lands into its payload with no parity, so two
 *      all-gathers back to back let the faster domain overwrite a block before it is copied out
 *  @verify pass sc gather=true consumer_sync=true: an exchange after the copy-out, like the
 *      trainer's force and gradient all-reduces, keeps a partner from posting the next all-gather
 *      early
 */

// The knobs: the all-reduce's parity landings, the all-gather's single payload, and a later exchange.
#ifndef parity
#define parity true
#endif
#ifndef gather
#define gather false
#endif
#ifndef consumer_sync
#define consumer_sync false
#endif

#define steps 3
#define landing_of(step) ((parity && !gather) -> ((step) % 2) : 0)

byte landing[4];  // domain·2+landing, the step whose block landed there, plus one
byte arrival[4];  // domain·2+parity, the newest step whose flag arrived, plus one
byte exchange[2]; // the later exchange's arrival per domain

proctype domain(byte index) {
    byte step = 0, folded = 0, partner = 1 - index;
    do
    :: step == steps -> break
    :: else ->
        atomic {
            landing[partner * 2 + landing_of(step)] = step + 1;
            arrival[partner * 2 + step % 2] = step + 1
        };
        (arrival[index * 2 + step % 2] >= step + 1);
        folded = landing[index * 2 + landing_of(step)];
        assert(folded == step + 1);
        if
        :: consumer_sync ->
            exchange[partner] = step + 1;
            (exchange[index] >= step + 1)
        :: else
        fi;
        step++
    od
}

init {
    atomic { run domain(0); run domain(1) }
}
