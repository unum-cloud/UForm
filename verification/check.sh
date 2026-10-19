#!/usr/bin/env bash
# Runs every `@verify` line of every Promela scenario and compares each verdict with the expected one:
# `pass` means the search completed with no error, `fail` that an assertion was violated, `stuck` that
# a process waits forever. Needs `spin` and a C compiler.

set -u
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
# Every verdict runs in its own directory in the background, `VERIFY_JOBS` at a time - four by
# default, since each pan holds a hash table of its own; `finish` prints them in the order they
# were queued, so the output reads like a serial run.
jobs_limit=${VERIFY_JOBS:-4}
queued=0

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$jobs_limit" ]; do sleep 0.1; done; }

# section <title>: a heading, printed in its turn
section() {
    mkdir -p "$build/$queued"
    echo "$1" >"$build/$queued/result"
    queued=$((queued + 1))
}

# verify <model.pml> <expected> <spin defines...>
verify() {
    local model=$1 expected=$2 dir=$build/$queued
    shift 2
    mkdir -p "$dir"
    queued=$((queued + 1))
    throttle
    (
        local verdict
        if ! (cd "$dir" && spin -a "$@" "$OLDPWD/$model" >pan.out 2>&1 &&
            cc -O2 -DSAFETY -DCOLLAPSE -DVECTORSZ=4096 -w -o pan pan.c >>pan.out 2>&1); then verdict=broken
        else
            (cd "$dir" && ./pan -m1000000 -w26 >>pan.out 2>&1)
            # A cut-short search, a knob defined twice, or a write or index past a scenario's shape proves nothing
            if grep -qE "macro redefined|max search depth too small|invalid array index|violated.*history_full" "$dir/pan.out"; then verdict=broken
            elif grep -q "errors: 0" "$dir/pan.out"; then verdict=pass
            elif grep -q "pan:1: assertion violated" "$dir/pan.out"; then verdict=fail
            elif grep -q "pan:1: invalid end state" "$dir/pan.out"; then verdict=stuck
            else verdict=broken; fi
        fi
        report "$verdict" "$expected" "$model $*" "$dir/pan.out" >"$dir/result"
    ) &
}

report() { # report <verdict> <expected> <label> <log>
    if [ "$1" = "$2" ]; then echo "  ok    $3"
    else echo "  WRONG $3 : expected $2, got $1"; sed -n 1,12p "$4"; touch "$(dirname "$4")/wrong"; fi
}

# finish: waits for every verdict, prints them in order, and exits with the count of wrong ones
finish() {
    wait
    local index failed=0
    for ((index = 0; index < queued; index++)); do
        cat "$build/$index/result"
        [ -e "$build/$index/wrong" ] && failed=$((failed + 1))
    done
    exit $failed
}

# verify_all [file...]: every `@verify` line of the given scenarios, or of every one here and one
# directory down, then the verdicts in order. A line names the expected verdict and the memory
# models, then knob overrides. Every knob the files declare must be set on some line, and every knob
# a line sets must be declared.
verify_all() {
    local files=("$@") file expected memories rest memory declared used unmatched
    [ ${#files[@]} -gt 0 ] || files=(*.pml */*.pml)
    declared=$(sed -n 's/^#ifndef \([a-z0-9_]*\)$/\1/p' "${files[@]}" */protocol.pml *.pml 2>/dev/null | grep -vx memory | sort -u)
    used=$(sed -n 's/^ \*  @verify [a-z]* [a-z0-9,]* \([^:]*\).*/\1/p' "${files[@]}" 2>/dev/null | tr ' ' '\n' | sed -n 's/=.*//p' | sort -u)
    [ $# -gt 0 ] && declared=$(comm -12 <(echo "$declared") <(echo "$used"))
    unmatched=$(comm -3 <(echo "$declared") <(echo "$used") | xargs)
    mkdir -p "$build/$queued"
    if [ -z "$unmatched" ]; then echo "  ok    every knob is declared and set" >"$build/$queued/result"
    else
        echo "  WRONG knobs declared or set alone: $unmatched" >"$build/$queued/result"
        touch "$build/$queued/wrong"
    fi
    queued=$((queued + 1))
    for file in "${files[@]}"; do
        grep -qs '^ \*  @verify' "$file" || continue
        section "$file: $(sed -n 's/^ \*  @brief //p' "$file" | head -1)"
        while read -r expected memories rest; do
            for memory in ${memories//,/ }; do
                verify "$file" "$expected" -Dmemory="$memory" $(for knob in $rest; do echo "-D$knob"; done)
            done
        done < <(sed -n 's/^ \*  @verify \([^:]*\).*/\1/p' "$file")
    done
    finish
}

# Sourced, it only defines the functions above; run directly, it checks the directory it lives in.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
cd "$(dirname "$0")"
verify_all "$@"
