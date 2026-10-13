# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Cluster-safety primitives for offline data acquisition.

The acquisition-time analog of the (deleted) train-time `_seeding` / `_chunked_sampler`. Parallelism
for acquisition comes from the Slurm scheduler, never from Python threads or multiprocessing (banned
by the stylebook; the repackers already fan out with GNU parallel). One array task is one
`(rank, world)`; every task runs `partition` over the same sorted shard list and pulls a disjoint
stride. Idempotency is by atomic rename: the final shard appearing on the shared filesystem is the
commit record, so there is no per-dataset sentinel that concurrent tasks would race on.

Rank/world resolution order (first that fully resolves wins):
    1. Slurm array   SLURM_ARRAY_TASK_ID  / SLURM_ARRAY_TASK_COUNT   (the acquisition array job)
    2. Slurm srun    SLURM_PROCID         / SLURM_NTASKS             (an interactive `srun -n K`)
    3. explicit CLI  --rank / --world                                (local N-way testing)
    4. single host   (0, 1)                                          (pull everything)

This module never imports or reimplements Slurm; it only reads the env vars Slurm exports.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence
from pathlib import Path


def resolve_rank_world(cli_rank: int | None = None, cli_world: int | None = None) -> tuple[int, int]:
    """Resolve `(rank, world)` from the Slurm array env, then srun env, then the CLI, then `(0, 1)`.

    @param[in] cli_rank `--rank` if the caller passed one, else None.
    @param[in] cli_world `--world` if the caller passed one, else None.
    @return `(rank, world)` with `0 <= rank < world` and `world >= 1`.
    """

    def env_pair(rank_key: str, world_key: str) -> tuple[int, int] | None:
        raw_rank, raw_world = os.environ.get(rank_key), os.environ.get(world_key)
        if raw_rank is None or raw_world is None:
            return None
        rank, world = int(raw_rank), int(raw_world)
        return (rank, world) if world >= 1 and 0 <= rank < world else None

    explicit = (cli_rank, cli_world) if cli_rank is not None and cli_world is not None else None
    return (
        env_pair("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_COUNT")
        or env_pair("SLURM_PROCID", "SLURM_NTASKS")
        or explicit
        or (0, 1)
    )


def partition[element_type](sorted_items: Sequence[element_type], rank: int, world: int) -> list[element_type]:
    """Return this rank's disjoint stride of an already-sorted, already-capped item list.

    Strided (`items[rank::world]`), not a contiguous block, so that when the shards are sorted by
    size every task draws an even mix of small and large ones — no straggler stuck on all the big
    shards. `world <= 1` returns everything, so a single-machine run pulls the whole capped set.

    @param[in] sorted_items The shards to divide; must be sorted identically on every task.
    @param[in] rank This task's index in `[0, world)`.
    @param[in] world Total number of tasks.
    @return `sorted_items[rank::world]`, or the whole list when `world <= 1`.
    """
    if world <= 1:
        return list(sorted_items)
    return list(sorted_items[rank::world])


def already_done(final_path: str | os.PathLike) -> bool:
    """Return whether a shard's final artifact exists — the atomic-rename commit record.

    Replaces the per-dataset `.downloaded` sentinel, which is wrong the moment several array tasks
    each write part of a dataset. Because every writer commits through `atomic_write`, the presence
    of the final file means a prior run wrote it completely; a resumed task skips it and re-fetches
    only the still-missing shards in its own partition.
    """
    return Path(final_path).exists()


@contextlib.contextmanager
def atomic_write(final_path: str | os.PathLike, rank: int = 0) -> Iterator[Path]:
    """Yield a rank-tagged temporary sibling; commit it with `os.replace` on clean exit.

    Generalizes repack_one.py's `temporary_path = parquet_path + ".tmp"; ...; os.replace(...)` idiom,
    adding a `.rank<r>` infix so two tasks that harmlessly target the same final path never collide
    on the temporary. The rename is atomic within a filesystem, so a reader or a resumed task never
    observes a footer-less parquet, and the final file's existence is the commit.

    @param[in] final_path The destination the caller wants to appear atomically.
    @param[in] rank This task's rank, mixed into the temporary name to avoid cross-task collisions.
    @return The temporary `Path` to write to; renamed onto `final_path` when the block exits cleanly.
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f"{final.name}.rank{rank}.tmp")
    try:
        yield temporary
        os.replace(temporary, final)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
