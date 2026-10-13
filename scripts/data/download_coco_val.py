# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Idempotent downloader for the COCO 2017 val split (5K images, ~1 GB).

Modality: image. The val split is the standard image-text retrieval smoke harness; Recall@1 / @5 /
@10 numbers are directly comparable with OpenCLIP regression tracking. The HuggingFace snapshot is
tried first, with a fallback to the cocodataset.org mirror when the hub is unavailable.

CLI usage:
    uv run scripts/data/download_coco_val.py
    uv run scripts/data/download_coco_val.py --out-dir /custom/path
    uv run scripts/data/download_coco_val.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from download_smokes import smoke_cache_root
from huggingface_hub import snapshot_download

_HF_REPO_ID = "HuggingFaceM4/COCO"
_DIRECT_URLS = (
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
)


def _try_huggingface(out_dir: Path) -> bool:
    """Attempt the HuggingFace snapshot fetch.

    Args:
        out_dir: Local destination directory for the snapshot.

    Returns:
        `True` if the snapshot landed, `False` if the fetch raised (the caller falls back to the
        direct mirror).
    """
    try:
        snapshot_download(
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=["*val2017*", "*annotations*"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_coco_val: HuggingFace path failed ({exc}); falling back to direct URLs")
        return False
    return True


def _download_zip(url: str, target_zip: Path) -> None:
    """Stream the zip URL into target_zip and extract into its parent."""
    print(f"download_coco_val: fetching {url}")
    with urllib.request.urlopen(url) as response, open(target_zip, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    print(f"download_coco_val: extracting {target_zip.name}")
    with zipfile.ZipFile(target_zip) as archive:
        archive.extractall(target_zip.parent)
    target_zip.unlink()


def _try_direct(out_dir: Path) -> bool:
    """Fetch the COCO val zips from the cocodataset.org mirror."""
    try:
        for url in _DIRECT_URLS:
            zip_name = url.rsplit("/", 1)[-1]
            target_zip = out_dir / zip_name
            if not target_zip.exists():
                _download_zip(url, target_zip)
    except Exception as exc:  # noqa: BLE001
        print(f"download_coco_val: direct mirror failed ({exc})")
        return False
    return True


def download(out_dir: Path, force: bool = False) -> int:
    """Download COCO 2017 val into out_dir. Skip if already populated unless force=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_coco_val: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    if _try_huggingface(out_dir) or _try_direct(out_dir):
        sentinel.touch()
        print(f"download_coco_val: COCO 2017 val ready at {out_dir}")
        return 0
    print("download_coco_val: every download path failed", file=sys.stderr)
    return 1


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader for COCO 2017 val (~1 GB).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=smoke_cache_root() / "coco-val",
        help="Destination directory (default: $UFORM_SMOKE_CACHE/coco-val or ~/.cache/uform/smokes/coco-val).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the sentinel file exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.force))
