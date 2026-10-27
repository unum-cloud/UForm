/**
 *  @file verification/multimem_barrier.pml
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Two members and two consumers sharing the one counting barrier of a multimem domain.
 *
 *  `multimem_barrier_peer_kernel_` from `src/utilities/multimem_collectives.cu` adds one to every
 *  member's arrival flag with a release reduction, spins on its own flag with acquire loads until
 *  it holds one more round of arrivals, and self-advances its round. The flag is anonymous: the
 *  mixer, the contrastive loss and the optimizer all cross it, so a barrier is labelled only by its
 *  position in the step, and every member must cross the same barriers in the same order. Each
 *  consumer here stages its word, crosses the barrier and reads its peer's word: first the mixer's,
 *  then the contrastive loss's.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc drop_barrier=true: every member crosses every consumer's barrier; when one
 *      member skips the loss's barrier, the other's loss barrier pairs with nothing new, and the
 *      member that skipped reads its peer's loss word before the peer staged it
 *  @verify fail rc11 arrival_order=order_relaxed: the arrival is a release reduction and the spin
 *      an acquire load, so a peer's staged word is visible once its arrival is; a relaxed arrival
 *      lets the reader see the arrival and an unstaged word
 */
#define thread_count 2
#define location_count 6
#define history_depth 5
#include "weak_memory.pml"

// The knobs: the arrival's order, and a member skipping the second consumer's barrier.
#ifndef arrival_order
#define arrival_order order_release
#endif
#ifndef drop_barrier
#define drop_barrier false
#endif

// The words: each consumer's staged word per member, then each member's arrival flag.
#define mixer_word(member) (member)
#define loss_word(member) (2 + (member))
#define arrivals(member) (4 + (member))

// one barrier crossing of member @p t , whose round @p target advances by the member count
inline cross_barrier(t, target, observed, counted) {
    read_modify_write(t, arrivals(0), arrival_order, counted, counted + 1);
    read_modify_write(t, arrivals(1), arrival_order, counted, counted + 1);
    target = target + 2;
    do
    :: load(t, arrivals(t), order_acquire, observed);
        if
        :: observed >= target -> break
        :: else -> (newest_value(arrivals(t)) >= target)
        fi
    od
}

proctype member(byte t) {
    byte target = 0, observed = 0, counted = 0, peer_word = 0;
    store(t, mixer_word(t), order_relaxed, 1);
    cross_barrier(t, target, observed, counted);
    load(t, mixer_word(1 - t), order_relaxed, peer_word);
    assert(peer_word == 1);
    store(t, loss_word(t), order_relaxed, 1);
    if
    :: drop_barrier && t == 1
    :: else -> cross_barrier(t, target, observed, counted)
    fi;
    load(t, loss_word(1 - t), order_relaxed, peer_word);
    assert(peer_word == 1)
}

init {
    atomic { run member(0); run member(1) }
}
