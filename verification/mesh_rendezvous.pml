/**
 *  @file verification/mesh_rendezvous.pml
 *  @author Ash Vardanian
 *  @date October 19, 2026
 *  @brief Two processes wiring one phase twice over the shared exchange directory, as the probe
 *      trainers and the real one each make a multimem domain.
 *
 *  `mesh_topology_t` from `include/uform/utilities/mesh.hpp` publishes a payload by an atomic
 *  rename, awaits a peer's by polling until its file exists, and crosses a barrier by arrive and
 *  release files. Files are never removed within a job, so a phase that runs twice reads whatever
 *  the first run left unless its keys differ. Every wiring phase claims a generation from the
 *  topology and keys its payloads and barriers with it, so each round reads its own files.
 *
 *  - Each process awaits the peer's payload and checks it names this round.
 *  - After the barrier, the lead must have reached this round too.
 *
 *  @verify pass sc
 *  @verify fail sc generation_keys=false: with fixed keys the second round reads the first round's
 *      payload and passes the first round's release
 */

// The knob: the phase keys its files with its generation; overridden by a `@verify` line.
#ifndef generation_keys
#define generation_keys true
#endif

#define rounds 2
#define key(round) (generation_keys -> (round) : 0)

byte payload[2 * rounds]; // host · rounds+key, the payload a host published under that key
byte arrive[rounds];      // the follower's arrival marker under a barrier key
byte release[rounds];     // the lead's release marker under a barrier key
bool lead_reached[rounds];

proctype host(byte index) {
    byte round = 0, seen = 0;
    do
    :: round == rounds -> break
    :: else ->
        payload[index * rounds + key(round)] = round + 1;
        // await: poll until the peer's file exists, then read it whole
        (payload[(1 - index) * rounds + key(round)] != 0);
        seen = payload[(1 - index) * rounds + key(round)];
        assert(seen == round + 1);
        if
        :: index == 0 ->
            lead_reached[round] = true;
            (arrive[key(round)] != 0);
            release[key(round)] = round + 1
        :: else ->
            arrive[key(round)] = round + 1;
            (release[key(round)] != 0);
            assert(lead_reached[round])
        fi;
        round++
    od
}

init {
    atomic { run host(0); run host(1) }
}
