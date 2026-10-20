/**
 *  @file verification/gateway_ordering.pml
 *  @author Ash Vardanian
 *  @date October 20, 2026
 *  @brief Two hosts of a gateway and a peer each, running the trainer's step three times: the
 *      mixer's cross-host all-gather, the contrastive force all-reduce, and the gateway's mean.
 *
 *  `trainer_t::step` from `src/train/trainer.cu:555` runs the data source's mixer, then the
 *  contrastive loss, then the optimizer, every member on its own stream. The mixer of
 *  `src/core/multimodal_batch.cu:684` all-gathers each member's block with the same member of the
 *  other host over InfiniBand, straight into the payload with no parity, copies the payload out
 *  and crosses the domain barrier at `:692`. With contrasts, `distributed_contrastive_segments_t`
 *  of `src/core/contrastive_segments.cu` crosses its barriers around each member's own force
 *  all-reduce at `:996`. `distributed_adafactor_t::step` of `src/core/optimizer.cu` stages every
 *  peer's gradient into its window before the barrier at `:449`, while the gateway stages after it
 *  at `:450`, since its window still holds the last world mean. `world_gradient_step_` at `:494`
 *  has the gateway reduce the peers' windows, all-reduce with the other gateway at `:508`, and copy
 *  the mean into its window at `:511`; every peer reads it after the barrier at `:513`.
 *
 *  The barrier is the counting one that `multimem_barrier.pml` checks: every member leaves a round
 *  with every member's view at its arrival. An InfiniBand exchange is the payload, then a release
 *  flag on the same queue pair, and an acquire spin on the own flag, as `gdaki_doubling.pml` has
 *  it; one flag per receiver stands for the parity pair, since the counters are monotonic.
 *
 *  @verify pass sc,rc11 contrasts=false
 *  @verify pass sc,rc11 contrasts=true
 *  @verify fail sc,rc11 drop_b2=true: every peer reads the world mean after the barrier the
 *      gateway crosses once it copied the mean in; without it a peer reads its window too early
 *  @verify pass sc,rc11 stage_before_b1=true: the mixer's barrier already orders the gateway's
 *      restage after every peer read the last mean, so staging before the first barrier is safe in
 *      a step that mixes
 *  @verify fail sc,rc11 stage_before_b1=true mixer=false: a synthetic source skips the mixer, and
 *      with no contrasts only the first barrier keeps the gateway from restaging over the mean a
 *      peer has yet to read
 *  @verify pass sc,rc11 stage_before_b1=true mixer=false contrasts=true: the contrastive loss's
 *      barriers order the restage as the mixer's does
 */
#define thread_count 4
#define location_count 18
#define steps 3
#define history_depth (2 * steps + 1)
#include "weak_memory.pml"

// The knobs: the step's mixer and contrasts, an early gateway stage, and a dropped second barrier.
#ifndef mixer
#define mixer true
#endif
#ifndef contrasts
#define contrasts false
#endif
#ifndef stage_before_b1
#define stage_before_b1 false
#endif
#ifndef drop_b2
#define drop_b2 false
#endif

// Thread t is member t % 2 of host t / 2, member 0 the gateway, and pairs with member t ^ 2.
#define payload(t) (t)
#define gather_flag(t) (4 + (t))
#define force_flag(t) (8 + (t))
#define window(t) (12 + (t))
#define gradient_flag(t) (16 + (t) / 2)
#define gradient_of(step) (2 * (step) + 1)
#define mean_of(step) (2 * (step) + 2)

byte barrier_arrivals[2]; // per host, the arrivals of the barrier's current round
byte barrier_rounds[2];   // per host, the rounds every member arrived at

// one round of a host's counting barrier: the last arrival hands every member every view
inline cross_barrier(t, round) {
    atomic {
        barrier_arrivals[t / 2] = barrier_arrivals[t / 2] + 1;
        if
        :: barrier_arrivals[t / 2] == 2 ->
            merge_view((t / 2) * 2, (t / 2) * 2 + 1);
            merge_view((t / 2) * 2 + 1, (t / 2) * 2);
            barrier_arrivals[t / 2] = 0;
            barrier_rounds[t / 2] = barrier_rounds[t / 2] + 1
        :: else
        fi
    };
    round = round + 1;
    (barrier_rounds[t / 2] >= round)
}

// one exchange with the partner: the release flag after the payload, then the spin on the own flag
inline exchange(t, partner_flag, own_flag, step, observed) {
    store(t, partner_flag, order_release, step + 1);
    do
    :: load(t, own_flag, order_acquire, observed);
        if
        :: observed >= step + 1 -> break
        :: else -> (newest_value(own_flag) >= step + 1)
        fi
    od
}

proctype member(byte t) {
    byte step = 0, round = 0, observed = 0;
    do
    :: step == steps -> break
    :: else ->
        if
        :: mixer ->
            store(t, payload(t ^ 2), order_relaxed, step + 1);
            exchange(t, gather_flag(t ^ 2), gather_flag(t), step, observed);
            load(t, payload(t), order_relaxed, observed);
            assert(observed == step + 1);
            cross_barrier(t, round)
        :: else
        fi;
        if
        :: contrasts ->
            cross_barrier(t, round);
            exchange(t, force_flag(t ^ 2), force_flag(t), step, observed);
            cross_barrier(t, round)
        :: else
        fi;
        if
        :: t % 2 == 1 || stage_before_b1 -> store(t, window(t), order_relaxed, gradient_of(step))
        :: else
        fi;
        cross_barrier(t, round);
        if
        :: t % 2 == 0 ->
            if
            :: !stage_before_b1 -> store(t, window(t), order_relaxed, gradient_of(step))
            :: else
            fi;
            load(t, window(t + 1), order_relaxed, observed);
            assert(observed == gradient_of(step));
            exchange(t, gradient_flag(t ^ 2), gradient_flag(t), step, observed);
            store(t, window(t), order_relaxed, mean_of(step))
        :: else
        fi;
        if
        :: drop_b2
        :: else -> cross_barrier(t, round)
        fi;
        if
        :: t % 2 == 1 ->
            load(t, window(t - 1), order_relaxed, observed);
            assert(observed == mean_of(step))
        :: else
        fi;
        step++
    od
}

init {
    atomic { run member(0); run member(1); run member(2); run member(3) }
}
