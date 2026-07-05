# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Idempotent downloader for VGGSound (Loie/VGGSound mirror, ~338 GB).

Modality: audio + video.  200k YouTube clips x 10s with audio tracks + visual frames +
class labels.  Cleanly licensed (CC BY 4.0) and the canonical bridge corpus for
audio-visual cross-modal training - significantly cleaner than AudioSet at a fraction of
the size.  Pairs naturally with the upcoming audio_codec_pool_t for log-mel features.

Shard cap.
The full corpus is ~338 GB. `--max-shards N` (or `--fraction F`) cap how many of the upstream
archive shards (`.zip` / `.tar` / `.tar.gz` / `.parquet`) are pulled BEFORE any transfer starts,
mirroring `download_cc12m.py`. With NEITHER flag the whole repo is pulled (backwards-compatible
default).

CLI usage:
    uv run python_data/download_vggsound.py                       # whole repo (~338 GB)
    uv run python_data/download_vggsound.py --max-shards 4        # first 4 archive shards
    uv run python_data/download_vggsound.py --fraction 0.30       # leading 30% of the shards
    uv run python_data/download_vggsound.py --out-dir $UFORM_DATA_ROOT/vggsound
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

_PRIMARY_REPO_ID = "Loie/VGGSound"
_DEFAULT_OUT_DIR = Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "vggsound"
#: Archive shard suffixes; everything else (README, csv labels) is always kept.
_SHARD_SUFFIXES = (".zip", ".tar", ".tar.gz", ".parquet")


def _shard_allow_patterns(repo_id: str, max_shards: int | None, fraction: float | None) -> list[str] | None:
    """Return the `allow_patterns` capping the pull to the leading archive shards, or `None` for all.

    Probes the repo's file list, partitions it into sorted archive shards and non-shard metadata
    (always kept), then keeps the first `keep_count` shards where `keep_count` is the tighter of
    `--fraction` and `--max-shards`. Returns `None` when neither cap is set.

    Args:
        repo_id: HuggingFace dataset repo id.
        max_shards: Hard cap on the number of archive shards, or `None` for no hard cap.
        fraction: Share of the shards to keep in (0, 1], or `None` for no fractional cap.

    Returns:
        The `allow_patterns` list for `snapshot_download`, or `None` to pull the whole repo.
    """
    if max_shards is None and fraction is None:
        return None
    all_files = list(list_repo_files(repo_id, repo_type="dataset"))
    shards = sorted(name for name in all_files if name.endswith(_SHARD_SUFFIXES))
    metadata = [name for name in all_files if not name.endswith(_SHARD_SUFFIXES)]
    keep_count = len(shards)
    if fraction is not None:
        keep_count = min(keep_count, max(1, math.ceil(len(shards) * fraction)))
    if max_shards is not None:
        keep_count = min(keep_count, max_shards)
    kept = shards[:keep_count]
    print(
        f"download_vggsound: limiting to {len(kept)} of {len(shards)} archive shards + {len(metadata)} metadata files"
    )
    return kept + metadata


def download(
    out_dir: Path,
    repo_id: str | None = None,
    force: bool = False,
    max_shards: int | None = None,
    fraction: float | None = None,
) -> int:
    """Download VGGSound (or a leading shard slice) to out_dir. Skip if populated unless force=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_vggsound: already at {out_dir}; skip (pass --force)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    allow_patterns = _shard_allow_patterns(chosen_repo, max_shards, fraction)
    try:
        snapshot_download(
            repo_id=chosen_repo, repo_type="dataset", local_dir=str(out_dir), allow_patterns=allow_patterns
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_vggsound: {chosen_repo} failed ({exc})", file=sys.stderr)
        return 1
    sentinel.touch()
    print(f"download_vggsound: VGGSound ready at {out_dir} (repo={chosen_repo})")
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader for VGGSound (~338 GB); --max-shards / --fraction cap the pull."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR, help=f"Destination directory (default: {_DEFAULT_OUT_DIR})."
    )
    parser.add_argument(
        "--repo-id", type=str, default=None, help=f"Override HuggingFace dataset id (default: {_PRIMARY_REPO_ID})."
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Cap the pull to the first N archive shards (default: whole repo).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Cap the pull to the leading fraction (0, 1] of the archive shards (default: whole repo).",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if the sentinel file exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.repo_id, args.force, args.max_shards, args.fraction))
