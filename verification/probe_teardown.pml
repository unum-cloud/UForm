/**
 *  @file verification/probe_teardown.pml
 *  @author Ash Vardanian
 *  @date October 20, 2026
 *  @brief Two hosts wiring a queue pair per probe generation, writing one flag across, and
 *      destroying the queue pair before the next generation wires its own.
 *
 *  `probe_footprint_` from `src/train/trainer.cu:157` builds a probe trainer, takes one step, waits
 *  for its own streams in `trainer_t::synchronize` at `:691`, and destroys the probe on return,
 *  three times over. Each generation's queue pair is wired by `infiniband_reach_t::make_` of
 *  `src/utilities/infiniband_collectives.cu`, which publishes its coordinates and awaits the partner's
 *  armed one at `:511`, so a first write always finds the partner's queue pair. Every post kernel
 *  writes the payload, then a signaled flag, and drains that one completion with
 *  `infiniband_drain_one` of `include/uform/utilities/infiniband_collectives.cuh:979`; on a Reliable Connection
 *  the completion is the partner's acknowledgement, so the writer's synchronize returns only once
 *  its own writes are acknowledged. The partner's NIC sends that acknowledgement on its own time
 *  after the flag landed, and a responder that destroyed its queue pair first leaves the writer to
 *  retry into a queue pair that no longer exists.
 *
 *  The host barrier is `mesh_topology_t::barrier` of `include/uform/utilities/mesh.hpp:553`, keyed per generation.
 *  `probe_footprint_` wires the mixer's all-gathers after its synchronize, and that publish and
 *  await of every partner's coordinates acts as the same barrier before the probe is destroyed.
 *
 *  @verify pass sc
 *  @verify fail sc teardown_barrier=false: a host that saw its partner's flag and its own
 *      acknowledgement may destroy its queue pair before its NIC acknowledged the partner's write,
 *      and the partner's retry reaches the destroyed queue pair
 *  @verify fail sc drain=false: the barrier suffices only because every post kernel drains its
 *      completion; without the drain a host reaches the barrier with its write unacknowledged, and
 *      the acknowledgement reaches the queue pair its host destroyed after the barrier
 */

// The knobs: a host barrier between synchronize and teardown, and the post kernel's drain.
#ifndef teardown_barrier
#define teardown_barrier true
#endif
#ifndef drain
#define drain true
#endif

#define generations 2
#define pair_of(host, generation) ((host) * generations + (generation))
#define created 1
#define destroyed 2

byte queue_pair[4]; // host·generations+generation: zero until wired, then created, then destroyed
bool armed[4];      // the published coordinates and armed marker of each queue pair
bool posted[4];     // per writer's queue pair, its payload and flag write on the wire
bool landed[4];     // per writer's queue pair, its flag in the partner's memory
bool completed[4];  // per writer's queue pair, the acknowledgement reaped as its completion
bool arrived[4];    // per host and generation, the teardown barrier's arrival

// the fabric carrying one queue pair's write to its partner and the acknowledgement back
proctype link(byte writer) {
    byte responder = pair_of(1 - writer / generations, writer % generations);
    (posted[writer]);
    assert(queue_pair[responder] == created);
    landed[writer] = true;
    // the acknowledgement leaves the responder's queue pair, or the writer's retry reaches it
    assert(queue_pair[responder] == created);
    assert(queue_pair[writer] == created);
    completed[writer] = true
}

proctype host(byte index) {
    byte generation = 0, own = 0, partner = 0;
    do
    :: generation == generations -> break
    :: else ->
        own = pair_of(index, generation);
        partner = pair_of(1 - index, generation);
        queue_pair[own] = created;
        armed[own] = true;
        (armed[partner]);
        posted[own] = true;
        if
        :: drain -> (completed[own])
        :: else
        fi;
        (landed[partner]);
        if
        :: teardown_barrier ->
            arrived[own] = true;
            (arrived[partner])
        :: else
        fi;
        queue_pair[own] = destroyed;
        generation++
    od
}

init {
    atomic { run host(0); run host(1); run link(0); run link(1); run link(2); run link(3) }
}
