# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Stage REAL COCO captions into the `coco_captions` registry schema, replacing the VQA-as-caption data.

Modality: image+text. The `coco_captions` registry entry expects each row to pair an `image`
struct<bytes,path> with `sentences_raw`, a `list<string>` of the FIVE human-written captions for that
image (anchor_field=sentences_raw, positive_field=image). Do NOT source captions from
`lmms-lab/COCO-Caption2017`: its captioning-EVAL `answer` column is empty on the test split, so every
`sentences_raw` decodes to `['None']` and carries no caption signal at all.

This downloader stages the genuine COCO captions instead. The `2014_captions` config of
`HuggingFaceM4/COCO` (the Karpathy-split COCO captioning corpus, same image pool and the same five
human captions per image as COCO 2017) is auto-converted to parquet on the `refs/convert/parquet`
branch ALREADY in the exact registry schema:

    image:         struct<bytes: binary, path: string>
    sentences_raw: list<string>   (the five real human captions, grouped per image)

so the repack is a pure two-column projection — no caption is invented. ~30 train + ~2 validation
shards, ~4000 images each (~120K train + ~5K val images).

Shard cap and replacement.
`--max-shards N` (or `--fraction F`) cap how many of the upstream caption shards are pulled BEFORE any
transfer, mirroring `download_audioset.py`. Because this REPLACES the broken VQA corpus, `--replace`
(default on) deletes any pre-existing `*.parquet` in the destination `data/` dir before writing the
real shards, so the registry glob `data/coco_captions/data/*.parquet` picks up only the real shards.

CLI usage:
    uv run scripts/data/download_coco_captions.py                    # whole caption corpus (~18 GB)
    uv run scripts/data/download_coco_captions.py --max-shards 1     # one shard (~4000 images) probe
    uv run scripts/data/download_coco_captions.py --out-dir $UFORM_DATA_ROOT/coco_captions
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from acquire_shared import cap_shards, data_root
from huggingface_hub import list_repo_files, snapshot_download

_PRIMARY_REPO_ID = "HuggingFaceM4/COCO"
#: The grouped (five-captions-per-image) config, auto-converted to parquet on the convert branch.
_PARQUET_REVISION = "refs/convert/parquet"
_CONFIG_PREFIX = "2014_captions/"
#: Only the train + validation splits carry captions; the test split is held out / unlabelled.
_CAPTION_SPLIT_MARKERS = ("coco-train", "coco-validation")
_DEFAULT_OUT_DIR = data_root() / "coco_captions"
_REGISTRY_IMAGE_FIELD = "image"
_REGISTRY_CAPTION_FIELD = "sentences_raw"


def _caption_shards(repo_id: str) -> list[str]:
    """List the convert-branch caption parquet shards (train + validation), sorted deterministically."""
    all_files = list_repo_files(repo_id, repo_type="dataset", revision=_PARQUET_REVISION)
    return sorted(
        name
        for name in all_files
        if name.startswith(_CONFIG_PREFIX)
        and name.endswith(".parquet")
        and any(marker in name for marker in _CAPTION_SPLIT_MARKERS)
    )


def _shard_allow_patterns(repo_id: str, max_shards: int | None, fraction: float | None) -> list[str]:
    """Return the explicit caption-shard `allow_patterns`, capped by the tighter of fraction / max-shards.

    Unlike the whole-repo downloaders, COCO's convert branch carries several unrelated configs (the
    exploded `2014/` split, the test split), so the allow-list is always the explicit caption shards —
    never `None` — and the cap simply trims that list.

    Args:
        repo_id: HuggingFace dataset repo id.
        max_shards: Hard cap on the number of caption shards, or `None` for no hard cap.
        fraction: Share of the caption shards to keep in (0, 1], or `None` for no fractional cap.

    Returns:
        The `allow_patterns` list of caption parquet paths for `snapshot_download`.
    """
    shards = _caption_shards(repo_id)
    kept = cap_shards(shards, fraction, max_shards)
    print(f"download_coco_captions: pulling {len(kept)} of {len(shards)} caption shards (train + validation)")
    return kept


def _repack_to_registry_schema(source_parquet: Path, destination_parquet: Path) -> int:
    """Project one source shard to the `[image, sentences_raw]` registry schema; return rows written.

    The source already carries both columns, so this is a pure two-column select written atomically via
    a temporary sibling file then `replace`.
    """
    table = pq.read_table(source_parquet, columns=[_REGISTRY_IMAGE_FIELD, _REGISTRY_CAPTION_FIELD])
    temporary = destination_parquet.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(destination_parquet)
    return table.num_rows


def download(
    out_dir: Path,
    repo_id: str | None = None,
    force: bool = False,
    max_shards: int | None = None,
    fraction: float | None = None,
    replace: bool = True,
) -> int:
    """Snapshot the real COCO caption shards and repack them into the registry schema under out_dir/data.

    Args:
        out_dir: Destination root; shards land under `out_dir/data/` (registry glob points there).
        repo_id: Override HuggingFace dataset id.
        force: Re-run even if the sentinel exists.
        max_shards: Cap on upstream caption shards.
        fraction: Fractional cap on upstream caption shards.
        replace: Delete any pre-existing `*.parquet` in the destination `data/` dir first (replacing the
            stale VQA-as-caption corpus). On by default.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded_real_captions"
    if sentinel.exists() and not force:
        print(f"download_coco_captions: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    raw_dir = out_dir / ".hf_raw"
    allow_patterns = _shard_allow_patterns(chosen_repo, max_shards, fraction)
    if not allow_patterns:
        print("download_coco_captions: no caption shards found on the convert branch", file=sys.stderr)
        return 1
    try:
        snapshot_download(
            repo_id=chosen_repo,
            repo_type="dataset",
            revision=_PARQUET_REVISION,
            local_dir=str(raw_dir),
            allow_patterns=allow_patterns,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_coco_captions: {chosen_repo} failed ({exc})", file=sys.stderr)
        return 1

    source_shards = sorted(raw_dir.rglob("*.parquet"))
    if not source_shards:
        print("download_coco_captions: no parquet shards landed", file=sys.stderr)
        return 1
    if replace:
        for stale in data_dir.glob("*.parquet"):
            stale.unlink()
        print(f"download_coco_captions: replaced stale shards in {data_dir} (removed prior *.parquet)")
    total_rows = 0
    for index, source in enumerate(source_shards):
        destination = data_dir / f"coco-captions-{index:05d}.parquet"
        total_rows += _repack_to_registry_schema(source, destination)
    shutil.rmtree(raw_dir, ignore_errors=True)
    sentinel.touch()
    print(
        f"download_coco_captions: real COCO captions ready at {data_dir} "
        f"({len(source_shards)} shards, {total_rows} images, schema image + sentences_raw)"
    )
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage REAL COCO captions into the coco_captions registry schema (replaces VQA data).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR, help=f"Destination root (default: {_DEFAULT_OUT_DIR})."
    )
    parser.add_argument(
        "--repo-id", type=str, default=None, help=f"Override HuggingFace dataset id (default: {_PRIMARY_REPO_ID})."
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Cap the pull to the first N of ~32 caption shards (default: whole caption corpus).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Cap the pull to the leading fraction (0, 1] of the caption shards (default: whole corpus).",
    )
    parser.add_argument(
        "--keep-existing",
        dest="replace",
        action="store_false",
        help="Keep any pre-existing parquet in the destination data/ dir (default: replace it).",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if the sentinel file exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.repo_id, args.force, args.max_shards, args.fraction, args.replace))
