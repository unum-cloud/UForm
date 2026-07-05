# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Idempotent downloader for a HowTo100M narration subset (~16k rows, text-only, a few MB).

Modality: text (instructional narration). HowTo100M is an instructional video + ASR caption corpus
(~100M clips at full scale, ~12 TB). The clip-bearing mirrors left HuggingFace (the original
`OmniMMI/HowTo100M-100k` was removed; the official `HuggingFaceM4/howto100m` is now a streaming
loading script that fetches raw YouTube URLs, not snapshot-pullable shards), so this downloader now
targets `diyarhamedi/HowTo100M-subtitles-small`: a single parquet of ASR narration with columns
`video_id` (YouTube id), `category`, `subcategory`, `rank`, `task_id`, and `text` (the narration).
There are NO clip bytes here — only the narration text keyed by `video_id`, suitable for the
text-side of the video-text contrastive / reranker smokes (pair the `text` with externally fetched
clips by `video_id` if vision is needed).

Shard cap.
This subset is a single small parquet shard, so `--max-shards N` / `--fraction F` are effectively
no-ops here, but they are kept (and still cap the leading `.parquet` / `.tar` / `.tar.gz` / `.zip`
shards BEFORE transfer, mirroring `download_cc12m.py`) so a future clip-bearing mirror can be slotted
in via `--repo-id` without changing the interface. With NEITHER flag the whole subset is pulled.

CLI usage:
    uv run python_data/download_howto100m.py                  # whole subset (a few MB)
    uv run python_data/download_howto100m.py --max-shards 16  # first 16 archive shards
    uv run python_data/download_howto100m.py --fraction 0.25  # leading 25% of the shards
    uv run python_data/download_howto100m.py --out-dir $UFORM_DATA_ROOT/howto100m
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

_PRIMARY_REPO_ID = "diyarhamedi/HowTo100M-subtitles-small"
_DEFAULT_OUT_DIR = (
    Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "howto100m"
)
#: Archive shard suffixes; everything else (README, metadata json/csv) is always kept.
_SHARD_SUFFIXES = (".parquet", ".tar", ".tar.gz", ".zip")


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
        f"download_howto100m: limiting to {len(kept)} of {len(shards)} archive shards "
        f"+ {len(metadata)} metadata files"
    )
    return kept + metadata


def _snapshot_one(repo_id: str, out_dir: Path, allow_patterns: list[str] | None) -> bool:
    """Try a single HuggingFace snapshot pull, optionally restricted to `allow_patterns`.

    Args:
        repo_id: The HuggingFace dataset repository id.
        out_dir: Local destination directory for the snapshot.
        allow_patterns: Shard/metadata allow-list, or `None` for the whole repo.

    Returns:
        `True` on a successful pull, `False` if the pull raised.
    """
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=allow_patterns,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_howto100m: {repo_id} failed ({exc})")
        return False
    return True


def download(
    out_dir: Path,
    repo_id: str | None = None,
    force: bool = False,
    max_shards: int | None = None,
    fraction: float | None = None,
) -> int:
    """Download the HowTo100M sample (or a leading shard slice) to out_dir. Skip if populated unless force=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_howto100m: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    allow_patterns = _shard_allow_patterns(chosen_repo, max_shards, fraction)
    if _snapshot_one(chosen_repo, out_dir, allow_patterns):
        sentinel.touch()
        print(f"download_howto100m: HowTo100M sample ready at {out_dir} (repo={chosen_repo})")
        return 0
    print("download_howto100m: HuggingFace mirror failed", file=sys.stderr)
    return 1


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader for a HowTo100M narration subset; --max-shards / --fraction cap the pull.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Destination directory (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"Override HuggingFace dataset id (default: {_PRIMARY_REPO_ID}).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Cap the pull to the first N archive shards (default: whole sample).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Cap the pull to the leading fraction (0, 1] of the archive shards (default: whole sample).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the sentinel file exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.repo_id, args.force, args.max_shards, args.fraction))
