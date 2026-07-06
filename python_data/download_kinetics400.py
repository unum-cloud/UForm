# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
#     "xxhash",
# ]
# ///
"""Idempotent downloader for Kinetics-400 (kiyoonkim/kinetics-400-targz mirror) — pulls AND repacks.

Modality: video+text (label).  Full Kinetics-400 action-recognition corpus as `.tar.gz` shards
with per-clip MP4s; the class label per clip serves as the caption.  ~467 GB, ~240k clips across
~242 `part_*.tar.gz` shards.  The only complete non-gated K400 mirror on HF.

This single file does both halves. Default mode snapshots the (optionally shard-capped) tar.gz shards
plus the per-split annotations CSVs, then repacks each train shard into a sibling
`part_*.parquet` (mp4 binary + the action label joined in from `annotations/train.csv`) and deletes
the verified tar. The per-shard repack is heavy (gzip-decompress + per-clip CSV join of a multi-GB
shard), so the download orchestrates it across processes with GNU `parallel`, self-invoking this
file's hidden `--repack-shard <path>` worker mode — one file is both orchestrator and per-shard
worker, no separate repack script. Only the train split is repacked (the registry reads train).

Shard cap.
The full corpus is ~467 GB. `--max-shards N` (or `--fraction F`) cap how many of the upstream
`.tar.gz` shards are pulled BEFORE any transfer starts, mirroring `download_audioset.py`. With
NEITHER flag the whole repo is pulled (backwards-compatible default).

CLI usage:
    uv run python_data/download_kinetics400.py                  # whole repo (~467 GB)
    uv run python_data/download_kinetics400.py --max-shards 73  # first 73 tar.gz shards (~140 GB)
    uv run python_data/download_kinetics400.py --fraction 0.30  # leading 30% of the shards
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from engine_base import cap_shards, data_root
from huggingface_hub import list_repo_files, snapshot_download

_PRIMARY_REPO_ID = "kiyoonkim/kinetics-400-targz"
_DEFAULT_OUT_DIR = data_root() / "kinetics400"
#: tar.gz shard suffix; everything else (README, label csv) is always kept.
_SHARD_SUFFIXES = (".tar.gz",)
#: Parquet row-group size — small enough for streaming sub-shard reads in the loader.
ROW_GROUP_SIZE = 1024
#: Repack fan-out width when orchestrating the per-shard worker via GNU parallel. Each shard is
#: multi-GB, so this stays modest to bound peak memory and disk churn.
_REPACK_JOBS = 4


def _shard_allow_patterns(repo_id: str, max_shards: int | None, fraction: float | None) -> list[str] | None:
    """Return the `allow_patterns` capping the pull to the leading tar.gz shards, or `None` for all.

    Probes the repo's file list, partitions it into sorted `.tar.gz` shards and non-shard metadata
    (always kept), then keeps the first `keep_count` shards where `keep_count` is the tighter of
    `--fraction` and `--max-shards`. Returns `None` when neither cap is set.

    Args:
        repo_id: HuggingFace dataset repo id.
        max_shards: Hard cap on the number of tar.gz shards, or `None` for no hard cap.
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
        f"download_kinetics400: limiting to {len(kept)} of {len(shards)} tar.gz shards + {len(metadata)} metadata files"
    )
    return kept + metadata


def parquet_sibling_path(tar_path: str) -> str:
    """Return `<shard>.parquet` for a `<shard>.tar.gz` input."""
    if tar_path.endswith(".tar.gz"):
        return tar_path[:-7] + ".parquet"
    if tar_path.endswith(".tgz"):
        return tar_path[:-4] + ".parquet"
    raise ValueError(f"not a tar.gz shard: {tar_path!r}")


def load_annotations_index(csv_path: str) -> dict:
    """Load the per-split CSV into a `{youtube_id_timestart_timeend: row_dict}` index.

    Args:
        csv_path: Path to `annotations/{train,val,test}.csv`.

    Returns:
        Mapping from the join key (the bare filename stem) to the full annotation row.
    """
    index: dict[str, dict] = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            youtube_id = row["youtube_id"]
            time_start = int(row["time_start"])
            time_end = int(row["time_end"])
            stem = f"{youtube_id}_{time_start:06d}_{time_end:06d}"
            index[stem] = row
    return index


def repack_shard(
    tar_path: str,
    annotations_index: dict,
    *,
    keep_tar: bool,
    overwrite: bool,
) -> None:
    """Repack one Kinetics-400 tar.gz shard to a Parquet sibling, joining in CSV labels.

    Emits one parquet row per clip with the canonical schema (mp4 binary + flat annotation columns):
    `mp4` / `youtube_id` / `label` / `time_start` / `time_end` / `split` / `is_cc` / `key`. The clip
    filename inside the tar is `{youtube_id}_{time_start:06d}_{time_end:06d}.mp4`, joining to exactly
    one CSV row. Atomic in-place replace (temporary sibling file + `os.replace`); a round-trip verify
    re-reads the parquet and hashes every payload before the source tar is unlinked.

    Args:
        tar_path: Path to the `part_NNN.tar.gz`.
        annotations_index: Pre-loaded CSV index from `load_annotations_index`.
        keep_tar: When `True`, skip the source tar deletion after verify.
        overwrite: When `False`, skip a shard whose `.parquet` sibling already exists.
    """
    parquet_path = parquet_sibling_path(tar_path)
    if os.path.exists(parquet_path) and not overwrite:
        print(f"[skip] {tar_path}: {parquet_path} already exists", flush=True)
        return

    # Per-column lists (one row per matched clip).
    mp4_bytes: list[bytes] = []
    youtube_id: list[str] = []
    label: list[str] = []
    time_start: list[int] = []
    time_end: list[int] = []
    split: list[str] = []
    is_cc: list[int] = []
    key: list[str] = []
    payload_hashes: list[bytes] = []  # for round-trip verify

    dropped_no_anno = 0
    dropped_not_mp4 = 0
    with tarfile.open(tar_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if not base.endswith(".mp4"):
                dropped_not_mp4 += 1
                continue
            stem = base[: -len(".mp4")]
            row = annotations_index.get(stem)
            if row is None:
                dropped_no_anno += 1
                continue
            file_handle = archive.extractfile(member)
            if file_handle is None:
                continue
            payload = file_handle.read()
            mp4_bytes.append(payload)
            youtube_id.append(row["youtube_id"])
            label.append(row["label"])
            time_start.append(int(row["time_start"]))
            time_end.append(int(row["time_end"]))
            split.append(row["split"])
            is_cc.append(int(row["is_cc"]))
            key.append(stem)
            payload_hashes.append(xxhash.xxh64(payload).digest())

    if not mp4_bytes:
        print(f"[empty] {tar_path}: no joinable clips (dropped_no_anno={dropped_no_anno})", flush=True)
        return

    table = pa.table(
        {
            "mp4": pa.array(mp4_bytes, type=pa.binary()),
            "youtube_id": pa.array(youtube_id, type=pa.string()),
            "label": pa.array(label, type=pa.string()),
            "time_start": pa.array(time_start, type=pa.int32()),
            "time_end": pa.array(time_end, type=pa.int32()),
            "split": pa.array(split, type=pa.string()),
            "is_cc": pa.array(is_cc, type=pa.int8()),
            "key": pa.array(key, type=pa.string()),
        }
    )

    tmp_path = parquet_path + ".tmp"
    pq.write_table(table, tmp_path, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    os.replace(tmp_path, parquet_path)

    # Verify round-trip: re-read the parquet, hash every mp4 payload, compare to the hashes recorded
    # during the tar pass. Cheap (one streaming read of the binary column). If this passes the tar
    # payload made it through losslessly.
    parquet_table = pq.read_table(parquet_path, columns=["mp4", "key"])
    parquet_payloads = parquet_table.column("mp4").to_pylist()
    parquet_keys = parquet_table.column("key").to_pylist()
    if len(parquet_payloads) != len(mp4_bytes):
        raise ValueError(
            f"verify: row count mismatch in {parquet_path} — in-mem {len(mp4_bytes)} vs on-disk {len(parquet_payloads)}"
        )
    for row_position, (clip_key, payload) in enumerate(zip(parquet_keys, parquet_payloads, strict=False)):
        if clip_key != key[row_position]:
            raise ValueError(f"verify: key reorder at row {row_position} ({clip_key!r} vs {key[row_position]!r})")
        if xxhash.xxh64(payload).digest() != payload_hashes[row_position]:
            raise ValueError(f"verify: mp4 payload mismatch at row {row_position} (key={clip_key!r})")

    tar_bytes = os.path.getsize(tar_path)
    parquet_bytes = os.path.getsize(parquet_path)
    if not keep_tar:
        os.unlink(tar_path)
    print(
        f"[done] {tar_path}  {tar_bytes / 1e6:.0f} MB -> {parquet_bytes / 1e6:.0f} MB  "
        f"({len(mp4_bytes)} clips, dropped {dropped_no_anno} not-in-CSV + {dropped_not_mp4} non-mp4)",
        flush=True,
    )


def _repack_one(tar_path: str, annotations_csv: str, *, keep_tar: bool, overwrite: bool) -> int:
    """Worker mode: repack a single tar.gz shard, returning 0 on success and 1 on any failure.

    One bad shard must not poison the parallel run, so every error is caught, the partial temporary
    parquet is removed, and a `[FAIL]` line is printed.
    """
    if not os.path.isfile(annotations_csv):
        print(f"download_kinetics400: annotations CSV not found: {annotations_csv}", file=sys.stderr)
        return 1
    annotations_index = load_annotations_index(annotations_csv)
    try:
        repack_shard(tar_path, annotations_index, keep_tar=keep_tar, overwrite=overwrite)
    except Exception as error:  # noqa: BLE001 — one bad shard must not poison the parallel run
        try:
            stale = parquet_sibling_path(tar_path) + ".tmp"
            if os.path.exists(stale):
                os.remove(stale)
        except ValueError:
            pass
        print(f"[FAIL] {tar_path}: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
        return 1
    return 0


def _orchestrate_repack(out_dir: Path) -> int:
    """Fan the per-shard train repack across processes via GNU parallel, self-invoking `--repack-shard`."""
    script = Path(__file__).resolve()
    annotations_csv = out_dir / "annotations" / "train.csv"
    command = (
        f"find {out_dir}/train -name 'part_*.tar.gz' | sort | "
        f"parallel -j{_REPACK_JOBS} uv run {script} --repack-shard {{}} --annotations {annotations_csv}"
    )
    return subprocess.run(command, shell=True, check=False).returncode  # noqa: S602 — fixed self-invoke argv.


def download(
    out_dir: Path,
    repo_id: str | None = None,
    force: bool = False,
    max_shards: int | None = None,
    fraction: float | None = None,
) -> int:
    """Download Kinetics-400 (or a leading shard slice) to out_dir and repack the train split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_kinetics400: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    allow_patterns = _shard_allow_patterns(chosen_repo, max_shards, fraction)
    try:
        snapshot_download(
            repo_id=chosen_repo, repo_type="dataset", local_dir=str(out_dir), allow_patterns=allow_patterns
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_kinetics400: {chosen_repo} failed ({exc})", file=sys.stderr)
        return 1
    code = _orchestrate_repack(out_dir)
    parquet_count = len(list((out_dir / "train").glob("part_*.parquet"))) if (out_dir / "train").is_dir() else 0
    sentinel.touch()
    print(f"download_kinetics400: Kinetics-400 ready at {out_dir} ({parquet_count} train parquet shard(s))")
    return code


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader + repacker for Kinetics-400 (~467 GB); --max-shards / --fraction cap the pull."
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
        help="Cap the pull to the first N of ~242 tar.gz shards (default: whole repo).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Cap the pull to the leading fraction (0, 1] of the tar.gz shards (default: whole repo).",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if the sentinel file exists.")
    parser.add_argument(
        "--repack-shard",
        type=str,
        default=None,
        help="Worker mode: repack one tar.gz shard to parquet (used by parallel).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Worker mode: path to the matching split's annotations CSV (annotations/train.csv etc.).",
    )
    parser.add_argument(
        "--keep-tar", action="store_true", help="Worker mode: skip the tar deletion after verify (validation runs)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Worker mode: re-repack a shard whose `.parquet` sibling exists."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    if args.repack_shard is not None:
        if args.annotations is None:
            print("download_kinetics400: --repack-shard requires --annotations", file=sys.stderr)
            sys.exit(1)
        sys.exit(_repack_one(args.repack_shard, args.annotations, keep_tar=args.keep_tar, overwrite=args.overwrite))
    sys.exit(download(args.out_dir, args.repo_id, args.force, args.max_shards, args.fraction))
