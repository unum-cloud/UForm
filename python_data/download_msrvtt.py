# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Idempotent downloader for MSR-VTT (10K clips + 200K captions, ~10 GB) — pulls AND repacks.

Modality: video. MSR-VTT exercises the video-text contrastive smoke with paired captions. The
license blocks redistribution, so the dataset is fetched on demand at smoke time and never staged
in the shared corpus tree.

The snapshot ships `msrvtt_train_9k.json` (per-video caption arrays + metadata) and
`MSRVTT_Videos.zip` (10K mp4 files) at the repo root. This single file does both halves: it snapshots
the raw mirror, then materialises the registry-facing parquet
(`<out-dir>/train_9k.parquet`) — a binary `mp4` column plus a `captions: list<string>` column and
every JSON sidecar field (`video_id` / `source` / `category` / `url` / `start_time` / `end_time` /
`id`). The repack reads a whole zip, so the download orchestrates it via GNU `parallel`, self-invoking
this file's hidden `--repack-shard <json>` worker mode — one file is both orchestrator and worker, no
separate repack script.

CLI usage:
    uv run python_data/download_msrvtt.py
    uv run python_data/download_msrvtt.py --out-dir /custom/path
    uv run python_data/download_msrvtt.py --repo-id friedrichor/MSR-VTT
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from download_smokes import smoke_cache_root
from huggingface_hub import snapshot_download

_PRIMARY_REPO_ID = "friedrichor/MSR-VTT"
#: The train JSON sidecar and the videos zip the snapshot ships at its root.
_TRAIN_JSON = "msrvtt_train_9k.json"
_VIDEOS_ZIP = "MSRVTT_Videos.zip"
#: The registry-facing parquet shard name (registered glob: `msrvtt-captions/train_9k.parquet`).
_OUTPUT_PARQUET = "train_9k.parquet"

#: JSON sidecar key -> parquet column name (whitespace normalised for parquet friendliness).
JSON_KEY_RENAME = {
    "start time": "start_time",
    "end time": "end_time",
    "caption": "captions",  # the list-of-strings — flag plurality in the column name
}


def _canonical_column_name(raw_key: str) -> str:
    """Translate one raw JSON-sidecar key to the parquet column name."""
    return JSON_KEY_RENAME.get(raw_key, raw_key)


def repack(json_path: Path, zip_path: Path, out_path: Path, *, force: bool = False, head: int | None = None) -> None:
    """Repack MSR-VTT-captions into one parquet shard, preserving all annotation fields.

    Atomic write (temporary sibling file + `os.replace`) so a crashed run leaves the source intact and
    never strands a footer-less, unreadable parquet a later run would skip.

    Args:
        json_path: Path to the MSR-VTT JSON sidecar (e.g. `msrvtt_train_9k.json`).
            Each entry has `video` (filename inside the zip), `caption` (list of caption
            strings), and metadata fields `video_id` / `source` / `category` / `url` /
            `start time` / `end time` / `id`.
        zip_path: Path to `MSRVTT_Videos.zip` shipped by the upstream HuggingFace mirror.
            Videos live under `video/<videoN>.mp4`.
        out_path: Destination parquet shard.
        force: When `True`, overwrite `out_path` even if it already exists.
        head: When set, cap the number of rows written to the output parquet to this many;
            `None` (the default) writes every row.
    """
    if out_path.exists() and not force:
        print(f"repack_msrvtt: {out_path} already exists; skipping (pass --force to overwrite).")
        return

    with open(json_path) as handle:
        entries = json.load(handle)
    print(f"repack_msrvtt: {len(entries)} entries in {json_path.name}")

    # Collect per-column lists by integer index. We cast the binary payload explicitly so the parquet
    # footer carries `binary` not `large_binary`.
    columns: dict[str, list] = {
        "captions": [],
        "mp4": [],
        "video_id": [],
        "source": [],
        "category": [],
        "url": [],
        "start_time": [],
        "end_time": [],
        "id": [],
    }
    missing: list[str] = []

    with zipfile.ZipFile(zip_path, mode="r") as videos_zip:
        # `<video_name> -> <full member name>` lookup; one pass over namelist.
        zip_members = {name.rsplit("/", 1)[-1]: name for name in videos_zip.namelist() if name.endswith(".mp4")}
        for entry in entries:
            video_name = entry["video"]
            captions = entry.get("caption") or []
            if not captions:
                continue
            member_name = zip_members.get(video_name)
            if member_name is None:
                missing.append(video_name)
                continue
            with videos_zip.open(member_name) as video_handle:
                mp4_bytes = video_handle.read()
            columns["captions"].append(list(captions))
            columns["mp4"].append(mp4_bytes)
            columns["video_id"].append(entry.get("video_id"))
            columns["source"].append(entry.get("source"))
            columns["category"].append(entry.get("category"))
            columns["url"].append(entry.get("url"))
            columns["start_time"].append(entry.get("start time"))
            columns["end_time"].append(entry.get("end time"))
            columns["id"].append(entry.get("id"))
            if head is not None and len(columns["captions"]) >= head:
                break

    if missing:
        print(f"repack_msrvtt: {len(missing)} videos referenced in JSON not in zip; first 3: {missing[:3]}")

    # Build the table with explicit types for the columns pyarrow can't infer well.
    table = pa.table(
        {
            "captions": pa.array(columns["captions"], type=pa.list_(pa.string())),
            "mp4": pa.array(columns["mp4"], type=pa.binary()),
            "video_id": pa.array(columns["video_id"], type=pa.string()),
            "source": pa.array(columns["source"], type=pa.string()),
            "category": pa.array(columns["category"], type=pa.int64()),
            "url": pa.array(columns["url"], type=pa.string()),
            "start_time": pa.array(columns["start_time"], type=pa.float64()),
            "end_time": pa.array(columns["end_time"], type=pa.float64()),
            "id": pa.array(columns["id"], type=pa.int64()),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp"
    pq.write_table(table, tmp_path, compression="zstd", row_group_size=1024)
    os.replace(tmp_path, out_path)
    size_mb = out_path.stat().st_size / (1 << 20)
    print(
        f"repack_msrvtt: wrote {out_path} — {len(columns['captions'])} rows, "
        f"{size_mb:.1f} MB, schema={table.schema.names}."
    )


def _repack_one(json_path: str, *, force: bool, head: int | None) -> int:
    """Worker mode: repack one MSR-VTT JSON sidecar to the registry parquet, returning 0/1.

    Derives the videos zip and the output parquet as siblings of the JSON (the snapshot ships both
    at its root), so the orchestrator only needs to pass the JSON path.
    """
    json_file = Path(json_path)
    if not json_file.is_file():
        print(f"download_msrvtt: JSON sidecar not found: {json_file}", file=sys.stderr)
        return 1
    zip_path = json_file.parent / _VIDEOS_ZIP
    if not zip_path.is_file():
        print(f"download_msrvtt: videos zip not found beside JSON: {zip_path}", file=sys.stderr)
        return 1
    out_path = json_file.parent / _OUTPUT_PARQUET
    try:
        repack(json_file, zip_path, out_path, force=force, head=head)
    except Exception as error:  # noqa: BLE001 — surface a clean failure rather than a raw traceback
        print(f"[FAIL] {json_file}: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
        return 1
    return 0


def _orchestrate_repack(out_dir: Path) -> int:
    """Repack the train sidecar via GNU parallel, self-invoking the `--repack-shard` worker mode."""
    script = Path(__file__).resolve()
    command = f"find {out_dir} -name '{_TRAIN_JSON}' | parallel -j1 uv run {script} --repack-shard {{}}"
    return subprocess.run(command, shell=True, check=False).returncode  # noqa: S602 — fixed self-invoke argv.


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
        print(f"download_msrvtt: {repo_id} failed ({exc})")
        return False
    return True


def download(out_dir: Path, repo_id: str | None = None, force: bool = False) -> int:
    """Download MSR-VTT to out_dir and repack the train sidecar. Skip if populated unless force=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_msrvtt: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    if not _snapshot_one(chosen_repo, out_dir):
        print("download_msrvtt: HuggingFace mirror failed", file=sys.stderr)
        return 1
    code = _orchestrate_repack(out_dir)
    sentinel.touch()
    print(f"download_msrvtt: MSR-VTT ready at {out_dir} (repo={chosen_repo})")
    # The registered `msrvtt_captions` entry points at the repacked `train_9k.parquet` (built above),
    return code


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader + repacker for MSR-VTT (~10 GB).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=smoke_cache_root() / "msr-vtt",
        help="Destination directory (default: $UFORM_SMOKE_CACHE/msr-vtt or ~/.cache/uform/smokes/msr-vtt).",
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
        help="Redownload and re-repack even if the sentinel file exists.",
    )
    parser.add_argument(
        "--repack-shard",
        type=str,
        default=None,
        help="Worker mode: repack one MSR-VTT JSON sidecar to parquet (used by parallel).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Worker mode: cap the number of rows written to the output parquet (default: every row).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    if args.repack_shard is not None:
        sys.exit(_repack_one(args.repack_shard, force=args.force, head=args.head))
    sys.exit(download(args.out_dir, args.repo_id, args.force))
