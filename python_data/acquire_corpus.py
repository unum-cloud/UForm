# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Drive the balanced multi-modal corpus staging on `$UFORM_DATA_ROOT` from `acquire_manifest.py`.

Reads the declarative `STAGING_ROWS` manifest and, for each selected dataset THIS worker owns, runs its
downloader (and per-shard repack, where one applies) at the budget allocator's derived cap, logging the
target GB per modality. The objective is a large, balanced, real corpus across text / image / video / audio.

One driver for both single-host and Slurm — the only difference is `(rank, world)`:
  - RANK/WORLD: resolved via `sharding.resolve_rank_world` (Slurm array / srun env, else `--rank`/`--world`,
    else `(0, 1)`). Shardable rows (their downloader strides `[rank::world]` internally) run on every worker,
    each pulling its disjoint stride; whole rows (single-file repackers, multi-repo, `text_pairs`) cannot
    sub-shard, so each is assigned to one worker by `row_index % world == rank`.
  - RESUMABLE: no dataset is coarse-skipped; every owned row is invoked and resume is the downloader's own
    per-shard atomic rename (`already_done`) or output sentinel — so a giant pull that crashed midway
    finishes on re-submit. The on-disk shard count is printed for visibility.
  - MODALITY-SCOPED: `--modality text|image|video|audio|all` stages one modality at a time.
  - DRY-RUN: `--dry-run` prints the exact commands (with injected caps) and the balance table, downloading
    nothing — the safe default to review before launch.

Pure stdlib (no cudf / registry import) so it runs under a lightweight `uv run` environment.

Usage:
    uv run python_data/acquire_corpus.py --dry-run                 # plan only, no download
    uv run python_data/acquire_corpus.py --modality audio          # stage audio for real (single host)
    uv run python_data/acquire_corpus.py --rank 0 --world 2        # this worker's share of a local 2-way split
    srun ... uv run python_data/acquire_corpus.py --modality all   # cluster-wide (rank/world from Slurm)
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import acquire_manifest
from acquire_manifest import (
    MODALITIES,
    MODALITY_WEIGHTS,
    StagingRow,
    allocate_targets,
    derive_cap,
    resolve_budget_gigabytes,
    rows_for_modality,
)
from sharding import resolve_rank_world

#: Repository root (parent of `python_data/`); registry globs are anchored here.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

def staged_shard_count(registry_glob: str) -> int:
    """Count shards already on disk for a registry glob (recursive, anchored to PROJECT_ROOT) — for display."""
    absolute = registry_glob if Path(registry_glob).is_absolute() else str(PROJECT_ROOT / registry_glob)
    return len(glob.glob(absolute, recursive=True))


def run_command(command: list[str], dry_run: bool) -> int:
    """Print, and unless `dry_run`, execute one argv from the project root. Returns its exit code."""
    print(f"    $ {' '.join(command)}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)  # noqa: S603 — manifest argv.
    return completed.returncode


def run_shell(command: str, dry_run: bool) -> int:
    """Print, and unless `dry_run`, execute one shell command (used for repack pipelines)."""
    print(f"    $ {command}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), shell=True, check=False)  # noqa: S602 — manifest cmd.
    return completed.returncode


def stage_row(row: StagingRow, dry_run: bool, target_gigabytes: float, rank: int, world: int) -> tuple[str, float]:
    """Plan or stage one dataset row for this worker.

    Args:
        row: The manifest row to stage.
        dry_run: When True, only print the commands that would run.
        target_gigabytes: The budget allocator's target on-disk GB for this row.
        rank, world: This worker's Slurm split; a shardable row forwards them so its downloader strides.

    Returns:
        `(status, gigabytes)` where status is one of "planned" / "staged" / "blocked" / "failed", and
        gigabytes is the row's target GB (0 when blocked with no command). Resume is the downloader's own
        per-shard `already_done` / output sentinel, so a row is always invoked, never coarse-skipped.
    """
    cap = derive_cap(row, target_gigabytes)
    cap_text = f" cap={cap[0]} {cap[1]}" if cap else ""
    kind = "shardable" if row.shardable else "whole"
    header = (
        f"  [{row.dataset}] modality={row.modality} {kind} target={target_gigabytes:.0f} GB{cap_text} "
        f"(full~{row.estimated_full_gigabytes:.0f} GB, mechanism={row.fraction_mechanism}, "
        f"on-disk shards={staged_shard_count(row.registry_glob)})"
    )
    print(header, flush=True)
    if row.flag:
        print(f"    FLAG: {row.flag}", flush=True)

    command = row.resolved_download_command(target_gigabytes)
    if command is None:
        print("    -> BLOCKED: no download command (see FLAG above) — staged 0 GB.", flush=True)
        return "blocked", 0.0
    if row.shardable and world > 1:
        command = [*command, "--rank", str(rank), "--world", str(world)]

    download_code = run_command(command, dry_run)
    if download_code != 0 and not dry_run:
        print(f"    -> FAILED: downloader exited {download_code}.", flush=True)
        return "failed", 0.0

    # The GNU-parallel repack pipelines walk the whole tree and are not rank-partitioned, so only rank 0
    # runs them; repack_one is idempotent (skips an existing parquet), so a re-submit sweeps up any tars
    # other workers committed after rank 0's pass.
    if row.repack_command is not None and rank == 0:
        repack_code = run_shell(row.repack_command, dry_run)
        if repack_code != 0 and not dry_run:
            print(f"    -> FAILED: repack exited {repack_code}.", flush=True)
            return "failed", target_gigabytes

    return ("planned" if dry_run else "staged"), target_gigabytes


def print_balance_table(rows: list[StagingRow], budget_gigabytes: float, targets: dict[str, float]) -> None:
    """Print the per-modality allocated-GB vs budget-share table plus the grand total for the given rows."""
    print(f"\nBalance table (budget-allocated GB; total budget {budget_gigabytes:.0f} GB):", flush=True)
    print(f"  {'modality':<8} {'alloc GB':>10} {'share GB':>10} {'datasets':>9}  blockers", flush=True)
    grand_total = 0.0
    for modality in MODALITIES:
        modality_rows = [row for row in rows if row.modality == modality]
        if not modality_rows:
            continue
        allocated = sum(targets.get(row.dataset, 0.0) for row in modality_rows)
        share = budget_gigabytes * MODALITY_WEIGHTS.get(modality, 0.0)
        grand_total += allocated
        blockers = sum(1 for row in modality_rows if row.requires_new_knob or row.fraction_mechanism == "no-downloader")
        print(
            f"  {modality:<8} {allocated:>10.0f} {share:>10.0f} {len(modality_rows):>9} {blockers:>3} blocker(s)",
            flush=True,
        )
    print(f"  {'-' * 50}", flush=True)
    print(f"  {'TOTAL':<8} {grand_total:>10.0f} {budget_gigabytes:>10.0f}", flush=True)


def print_blocker_summary(rows: list[StagingRow]) -> None:
    """List the rows that block clean staging (missing fraction knob or missing downloader)."""
    blockers = [row for row in rows if row.requires_new_knob or row.fraction_mechanism == "no-downloader"]
    if not blockers:
        print("\n[blockers] none — every selected dataset has a working fraction-capable downloader.", flush=True)
        return
    print("\nBlockers (need a knob / new script before launch):", flush=True)
    for row in blockers:
        kind = "NO DOWNLOADER" if row.fraction_mechanism == "no-downloader" else "NEEDS FRACTION KNOB"
        print(f"  [{row.dataset}] {kind}: {row.flag}", flush=True)


def main() -> int:
    """Parse the CLI, stage (or plan) the selected modality, and print the balance table.

    Returns:
        Process exit code: `0` when every selected dataset was staged, skipped, or (in dry-run)
        planned; `1` when any real download/repack failed.
    """
    parser = argparse.ArgumentParser(description="Stage the balanced multi-modal corpus on $UFORM_DATA_ROOT.")
    parser.add_argument(
        "--modality",
        default="all",
        choices=["all", *MODALITIES],
        help="Which modality to stage; 'all' stages every modality.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact download/repack commands and allocated GB without downloading anything.",
    )
    parser.add_argument(
        "--budget-gb",
        type=float,
        default=None,
        help="Total on-disk budget in GB; default auto-detects free disk * SAFETY_MARGIN (or $UFORM_DATA_BUDGET_GB).",
    )
    parser.add_argument("--rank", type=int, default=None, help="this worker's index; usually Slurm-derived, explicit for local N-way runs.")
    parser.add_argument("--world", type=int, default=None, help="total workers; usually Slurm-derived, explicit for local N-way runs.")
    args = parser.parse_args()

    rank, world = resolve_rank_world(args.rank, args.world)
    budget_gigabytes = resolve_budget_gigabytes(args.budget_gb)
    targets = allocate_targets(budget_gigabytes)
    rows = rows_for_modality(args.modality)
    print(
        f"[stage] modality={args.modality}  rank={rank}/{world}  datasets={len(rows)}  dry_run={args.dry_run}  "
        f"budget={budget_gigabytes:.0f} GB  data_root={acquire_manifest.DATA_ROOT}",
        flush=True,
    )

    statuses: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not row.shardable and world > 1 and index % world != rank:
            continue  # a whole-dataset row runs on exactly one worker
        status, _gigabytes = stage_row(
            row, args.dry_run, targets.get(row.dataset, row.estimated_download_gigabytes), rank, world
        )
        statuses[status] = statuses.get(status, 0) + 1

    if rank == 0:
        print_balance_table(rows, budget_gigabytes, targets)
        print_blocker_summary(rows)
    print(f"\n[stage] rank {rank}/{world} status counts: {statuses}", flush=True)

    return 1 if statuses.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
