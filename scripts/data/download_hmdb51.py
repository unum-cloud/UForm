# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Idempotent downloader for HMDB-51 (51 actions, 6.8K clips, ~2 GB).

Modality: video. HMDB-51 is the smallest competitive video classification benchmark, used as the
clip-level VideoMAE smoke target without paying the 300+ GB Kinetics download cost. The HuggingFace
snapshot is tried first, with a fallback to the Brown CS mirror RAR archive.

The default mirror is `jxie/hmdb51` (the original `serpapi/HMDB51` was removed). It ships parquet
shards under `data/` (train-*-of-8, test-*-of-4) whose single `video: struct<bytes, path>` column
carries the per-clip MP4 bytes plus a `path` encoding the action class — already the inline
video-bytes shape the registry video schema expects.

CLI usage:
    uv run scripts/data/download_hmdb51.py
    uv run scripts/data/download_hmdb51.py --out-dir /custom/path
    uv run scripts/data/download_hmdb51.py --repo-id jxie/hmdb51
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from download_smokes import smoke_cache_root
from huggingface_hub import snapshot_download

_PRIMARY_REPO_ID = "jxie/hmdb51"
_DIRECT_URL = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"


def _try_huggingface(repo_id: str, out_dir: Path) -> bool:
    """Try a HuggingFace snapshot pull.

    Args:
        repo_id: The HuggingFace dataset repository id.
        out_dir: Local destination directory for the snapshot.

    Returns:
        `True` on success, `False` if the pull raised (the caller falls back to the direct mirror).
    """
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_hmdb51: HuggingFace {repo_id} failed ({exc}); trying direct mirror")
        return False
    return True


def _try_direct(out_dir: Path) -> bool:
    """Fetch the official Brown CS mirror RAR archive. Extraction is left to the caller."""
    target_archive = out_dir / "hmdb51_org.rar"
    if target_archive.exists():
        print(f"download_hmdb51: archive already present at {target_archive}")
        return True
    try:
        print(f"download_hmdb51: fetching {_DIRECT_URL}")
        with (
            urllib.request.urlopen(_DIRECT_URL) as response,
            open(target_archive, "wb") as handle,
        ):
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as exc:  # noqa: BLE001
        print(f"download_hmdb51: direct mirror failed ({exc})")
        if target_archive.exists():
            target_archive.unlink()
        return False
    print(
        "download_hmdb51: archive downloaded; extract with `unar hmdb51_org.rar`"
        " (rar archives are not unpacked here to avoid a hard dep on `unrar`)"
    )
    return True


def download(out_dir: Path, repo_id: str | None = None, force: bool = False) -> int:
    """Download HMDB-51 to out_dir. Skip if already populated unless force=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_hmdb51: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    if _try_huggingface(chosen_repo, out_dir) or _try_direct(out_dir):
        sentinel.touch()
        print(f"download_hmdb51: HMDB-51 ready at {out_dir}")
        return 0
    print("download_hmdb51: every download path failed", file=sys.stderr)
    return 1


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader for HMDB-51 (~2 GB).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=smoke_cache_root() / "hmdb-51",
        help="Destination directory (default: $UFORM_SMOKE_CACHE/hmdb-51 or ~/.cache/uform/smokes/hmdb-51).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"Override HuggingFace dataset id (default: {_PRIMARY_REPO_ID}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the sentinel file exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.repo_id, args.force))
