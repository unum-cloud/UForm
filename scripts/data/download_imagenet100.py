# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Idempotent downloader for ImageNet-100 (130K images, ~13 GB).

Modality: image. ImageNet-100 is a balanced 100-class slice of ImageNet-1K used as the
linear-probe sanity benchmark for the image contrastive smoke. One epoch on H100 finishes in well
under a minute, so it is the smallest useful target above MNIST-class toys.

CLI usage:
    uv run scripts/data/download_imagenet100.py
    uv run scripts/data/download_imagenet100.py --out-dir /custom/path
    uv run scripts/data/download_imagenet100.py --repo-id ambityga/imagenet100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from download_smokes import smoke_cache_root
from huggingface_hub import snapshot_download

_PRIMARY_REPO_ID = "clane9/imagenet-100"
_FALLBACK_REPO_ID = "ambityga/imagenet100"


def _snapshot_one(repo_id: str, out_dir: Path) -> bool:
    """Try a single HuggingFace snapshot pull.

    Args:
        repo_id: The HuggingFace dataset repository id.
        out_dir: Local destination directory for the snapshot.

    Returns:
        `True` on a successful pull, `False` if the pull raised.
    """
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_imagenet100: {repo_id} failed ({exc})")
        return False
    return True


def download(out_dir: Path, repo_id: str | None = None, force: bool = False) -> int:
    """Download ImageNet-100 to out_dir. Skip if already populated unless force=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_imagenet100: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    candidates = [repo_id] if repo_id else [_PRIMARY_REPO_ID, _FALLBACK_REPO_ID]
    for candidate in candidates:
        if _snapshot_one(candidate, out_dir):
            sentinel.touch()
            print(f"download_imagenet100: ImageNet-100 ready at {out_dir} (repo={candidate})")
            return 0
    print("download_imagenet100: every HuggingFace mirror failed", file=sys.stderr)
    return 1


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader for ImageNet-100 (~13 GB).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=smoke_cache_root() / "imagenet-100",
        help="Destination directory (default: $UFORM_SMOKE_CACHE/imagenet-100 or ~/.cache/uform/smokes/imagenet-100).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"Override HuggingFace dataset id (default tries {_PRIMARY_REPO_ID} then {_FALLBACK_REPO_ID}).",
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
