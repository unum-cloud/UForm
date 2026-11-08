"""Top-level smoke dataset downloader.

Wraps every per-modality `scripts/data/download_<dataset>.py` into one entry point with a `--only`
filter. Each subscript is itself idempotent, so re-running this top-level driver costs at most one
HEAD per shard. Also exposes `smoke_cache_root`, the per-host cache-root helper the per-modality
downloaders import.

CLI usage:
    uv run scripts/data/download_smokes.py
    uv run scripts/data/download_smokes.py --only coco_val,hmdb51
    uv run scripts/data/download_smokes.py --force

File: scripts/data/download_smokes.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///

import argparse
import importlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

DownloadFn = Callable[[Path, bool], int]


def smoke_cache_root() -> Path:
    """Per-host cache root for smoke datasets.

    Defaults to ``~/.cache/uform/smokes`` (XDG-style), overridable via
    ``$UFORM_SMOKE_CACHE``. Smoke data is ephemeral by design and must never
    share a tree with training corpora under ``data/``.
    """
    override = os.environ.get("UFORM_SMOKE_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "uform" / "smokes"


def _default_out_dir(tag: str) -> Path:
    """Return the canonical smoke-cache target directory for a smoke tag."""
    subdirs = {
        "coco_val": "coco-val",
        "imagenet100": "imagenet-100",
        "hmdb51": "hmdb-51",
        "msrvtt": "msr-vtt",
        "droid": "droid-1k",
    }
    return smoke_cache_root() / subdirs[tag]


def _all_downloaders() -> dict[str, DownloadFn]:
    """Return the registered downloader functions keyed by short dataset tag."""
    registry: dict[str, DownloadFn] = {}
    for tag in ("coco_val", "imagenet100", "hmdb51", "msrvtt", "droid"):
        module = importlib.import_module(f"download_{tag}")
        registry[tag] = module.download
    return registry


def run_all(only: list[str] | None = None, force: bool = False) -> int:
    """Run every downloader (or only the named ones) and return 0 if all succeed."""
    downloaders = _all_downloaders()
    targets = list(downloaders) if only is None else [t for t in only if t in downloaders]
    if only is not None:
        unknown = [t for t in only if t not in downloaders]
        if unknown:
            print(
                f"download_smokes: unknown dataset tag(s): {unknown}; valid={list(downloaders)}",
                file=sys.stderr,
            )
            return 1

    results: dict[str, tuple[bool, float]] = {}
    overall_started = time.time()
    for tag in targets:
        print("=" * 78)
        print(f"== {tag} download")
        print("=" * 78)
        started = time.time()
        out_dir = _default_out_dir(tag)
        try:
            return_code = downloaders[tag](out_dir, force)
        except Exception as exc:  # noqa: BLE001
            print(f"download_smokes: {tag} crashed: {exc}")
            return_code = 1
        elapsed = time.time() - started
        ok = return_code == 0
        results[tag] = (ok, elapsed)
        print(f"== {tag} {'PASSED' if ok else 'FAILED'} in {elapsed:.1f}s")

    print("=" * 78)
    print("download_smokes summary:")
    all_ok = True
    for tag, (ok, elapsed) in results.items():
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {tag:16s} {elapsed:7.1f}s")
    print(f"total elapsed: {time.time() - overall_started:.1f}s")
    return 0 if all_ok else 1


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every smoke dataset downloader.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of dataset tags (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the sentinel files exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    only_list = None
    if args.only:
        only_list = [s.strip() for s in args.only.split(",") if s.strip()]
    sys.exit(run_all(only=only_list, force=args.force))
