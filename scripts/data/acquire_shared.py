"""Shared acquisition helpers: the volume-cap and artifact-naming rules `acquire` reuses across sources.

The cap and naming rules must be identical for every source kind, so they live here once. Naming is the
load-bearing one: an artifact is named by its source shard's stem, never by an enumeration index, so
that the same shard maps to the same output on every rank and across resumes — which is what makes the
atomic-rename `already_done` skip correct under a rank partition.

File: scripts/data/acquire_shared.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import math
import os
from collections.abc import Sequence
from pathlib import Path


def data_root() -> Path:
    """The acquisition output root: `$UFORM_DATA_ROOT`, else the repo's `data/` directory.

    The single definition of the output root every standalone downloader shares. `acquire` and
    `acquire_clips` resolve the same path; a dataset appends its own `<name>` subdirectory.
    """
    return Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))


def cap_shards(shards: Sequence[str], fraction: float | None, max_shards: int | None) -> list[str]:
    """Apply the leading-slice volume cap: `fraction` and `max_shards` compose, the tighter one wins.

    Extracted verbatim from the 52 downloaders' `_shard_allow_patterns` math. Runs before the rank
    partition, so `world` tasks split the same capped set rather than each capping independently.
    """
    keep = len(shards)
    if fraction is not None:
        keep = min(keep, max(1, math.ceil(len(shards) * fraction)))
    if max_shards is not None:
        keep = min(keep, max_shards)
    return list(shards[:keep])


def artifact_path(out_dir: Path, out_name: str, source_shard: str) -> Path:
    """The canonical output path for one source shard: `<out_dir>/<out_name>-<flattened-source>.parquet`.

    Named by the source's full relative path (its directories joined by `-`, suffix dropped) rather than a
    bare basename, so nested shards that share a basename — e.g. `partial-{train,test,val}/0000.parquet` —
    never collide, and never by a positional index so rank 0 and rank 1 don't both write `...-00000`. A
    resumed task recognizes its already-committed shards by this stable, source-derived name.
    """
    relative = Path(source_shard)
    flattened = "-".join([*relative.parent.parts, relative.stem]) if relative.parent.parts else relative.stem
    return out_dir / f"{out_name}-{flattened}.parquet"
