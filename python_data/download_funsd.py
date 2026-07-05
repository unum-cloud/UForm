# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Idempotent downloader for FUNSD (nielsr/funsd + nielsr/funsd-layoutlmv3, <100 MB).

Modality: image+text.  Form Understanding Noisy Scanned Documents - 199 fully annotated
scanned forms with OCR text + spatial layout.  Tiny but extremely high quality for the
document-understanding axis; complements caption-style image-text corpora with structured-
text-in-images.  Use for eval / probe more than train.

This corpus is tiny (<100 MB), so there is no `--max-shards` / `--fraction` volume cap: the
whole thing is always pulled.

CLI usage:
    uv run python_data/download_funsd.py
    uv run python_data/download_funsd.py --out-dir $UFORM_DATA_ROOT/funsd
    uv run python_data/download_funsd.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

_REPOS = ("nielsr/funsd", "nielsr/funsd-layoutlmv3")
_DEFAULT_OUT_DIR = Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "funsd"


def download(out_dir: Path, force: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_funsd: already at {out_dir}; skip (pass --force)")
        return 0
    ok = True
    for repo_id in _REPOS:
        sub_dir = out_dir / repo_id.split("/")[-1]
        sub_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(sub_dir))
        except Exception as exc:
            print(f"download_funsd: {repo_id} failed ({exc})", file=sys.stderr)
            ok = False
    if not ok:
        return 1
    sentinel.touch()
    print(f"download_funsd: FUNSD + FUNSD-layoutlmv3 ready at {out_dir}")
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Idempotent downloader for FUNSD (nielsr mirrors).")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.force))
