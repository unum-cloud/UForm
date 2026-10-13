# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Idempotent downloader for M4-Instruct-Data (lmms-lab/M4-Instruct-Data, ~234 GB).

Modality: image+text instruction tuning.  ~3M multimodal instruction-response pairs covering
VQA / reasoning / referring / captioning across many vision tasks.  We have zero multimodal
instruction-tuning data; this fills that gap and pairs with the `ImageTextCaptionStep`
(image-text caption) trainer for downstream alignment.  The HuggingFace mirror ships the
instruction annotations as JSON plus the supporting image bytes in archive shards.

Shard cap.
The full corpus is ~234 GB, dominated by the image archive shards (`.tar.gz` / `.tar` / `.zip`).
`--max-shards N` (or `--fraction F`) cap how many of those upstream archive shards are pulled BEFORE
any transfer starts, mirroring `download_cc12m.py`; the JSON annotations and other metadata are always
kept.  With NEITHER flag the whole repo is pulled (backwards-compatible default).

CLI usage:
    uv run scripts/data/download_m4_instruct.py                       # whole repo (~234 GB)
    uv run scripts/data/download_m4_instruct.py --max-shards 8        # first 8 archive shards
    uv run scripts/data/download_m4_instruct.py --fraction 0.1        # leading 10% of the shards
    uv run scripts/data/download_m4_instruct.py --out-dir $UFORM_DATA_ROOT/m4-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acquire_shared import cap_shards, data_root
from huggingface_hub import list_repo_files, snapshot_download

_PRIMARY_REPO_ID = "lmms-lab/M4-Instruct-Data"
_DEFAULT_OUT_DIR = data_root() / "m4-instruct"
#: Image-archive shard suffixes; everything else (annotations json, README) is always kept.
_SHARD_SUFFIXES = (".tar.gz", ".tar", ".zip")


def _shard_allow_patterns(repo_id: str, max_shards: int | None, fraction: float | None) -> list[str] | None:
    """Return the `allow_patterns` capping the pull to the leading shards, or `None` for the whole repo.

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
    kept = cap_shards(shards, fraction, max_shards)
    print(
        f"download_m4_instruct: limiting to {len(kept)} of {len(shards)} archive shards "
        f"+ {len(metadata)} metadata files"
    )
    return kept + metadata


def download(
    out_dir: Path,
    repo_id: str | None = None,
    force: bool = False,
    max_shards: int | None = None,
    fraction: float | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_m4_instruct: already at {out_dir}; skip (pass --force)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    allow_patterns = _shard_allow_patterns(chosen_repo, max_shards, fraction)
    try:
        snapshot_download(
            repo_id=chosen_repo,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        print(f"download_m4_instruct: {chosen_repo} failed ({exc})", file=sys.stderr)
        return 1
    sentinel.touch()
    print(f"download_m4_instruct: M4-Instruct ready at {out_dir} (repo={chosen_repo})")
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader for M4-Instruct-Data (~234 GB); --max-shards / --fraction cap the pull.",
    )
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--repo-id", type=str, default=None)
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.repo_id, args.force, args.max_shards, args.fraction))
