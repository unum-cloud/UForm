# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
#     "xxhash",
# ]
# ///
"""Pull DiDeMo (dense video moment descriptions) and repack it to the registry video schema.

DiDeMo is ~10K Flickr clips, each annotated with a handful of ORDERED moment descriptions ("camera
zooms in on baby", "the slobber drips ...", "the baby drops her head"). The `friedrichor/DiDeMo` HF
mirror ships the MP4 clips as tar archives (`DiDeMo_Videos_mp4_test.tar`, ~5 GB, and an 8-part
byte-split `DiDeMo_Videos_mp4_train.tar.part-00N`, ~43 GB assembled) plus caption JSON. The
`wo_concat/didemo_<split>_wo_concat.json` files carry the per-moment captions as a LIST already
(`{"video": "train/<flickr>.mp4", "caption": ["...", "...", "..."], "source": "DiDeMo"}`), which is
exactly the multi-positive anchor the video loader reads as `captions: list<string>`.

This single file is BOTH the orchestrator and the per-shard repack worker:

  * Orchestrator (default): snapshots the caption JSON + the selected split tar archives, assembles
    the byte-split train parts into one tar, symlinks `data/didemo` at the host data root, then fans
    out one repack worker per split tar via GNU `parallel` (each worker self-invokes this file with
    `--repack-shard`).
  * Worker (`--repack-shard <tar> --captions <wo_concat.json>`): joins every `.mp4` member of the tar
    to its caption list by basename and writes `<root>/didemo/data/didemo-<split>.parquet`:

        mp4:       binary        # the clip bytes
        captions:  list<string>  # the ordered moment descriptions (multi-positive anchor)
        video:     string        # the mirror's relative path (e.g. train/<flickr>.mp4)
        key:       string        # the bare clip basename, for cross-dataset dedup

Atomic write (temporary sibling file + `os.replace`) verified by re-hashing every mp4 payload; the
source tar is deleted on success unless `--keep-raw` is passed. Mirrors the tar->parquet+join pattern
of `repack_kinetics400_to_parquet.py` and the symlink-into-`data/` guard of `download_poj104.py`.

Volume cap: the cap unit is the split (sorted `["test", "train"]`); with neither `--fraction` nor
`--max-shards` both splits are pulled, either flag keeps the leading slice (so `--max-shards 1` pulls
the smaller, self-contained test split — handy for smoke runs).

Usage:
    uv run python_data/download_didemo.py                       # both splits (~48 GB), then repack
    uv run python_data/download_didemo.py --max-shards 1        # test split only
    uv run python_data/download_didemo.py --data-root $UFORM_DATA_ROOT --no-repack
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from huggingface_hub import snapshot_download

#: Default host data root — overridable via `UFORM_DATA_ROOT`; the symlink target `data/didemo`.
DEFAULT_DATA_ROOT = os.environ.get(
    "UFORM_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)
#: HF mirror that ships the DiDeMo MP4 tars plus the per-moment caption JSON.
DIDEMO_REPO = "friedrichor/DiDeMo"
#: Per-split caption JSON whose `caption` field is already a LIST of ordered moment descriptions.
CAPTIONS_JSON = {
    "test": "wo_concat/didemo_test_wo_concat.json",
    "train": "wo_concat/didemo_train_wo_concat.json",
}
#: Per-split video tar(s): the test split is one complete archive, the train split is byte-split parts.
SPLIT_TAR_PATTERNS = {
    "test": ["DiDeMo_Videos_mp4_test.tar"],
    "train": ["DiDeMo_Videos_mp4_train.tar.part-*"],
}
#: Assembled tar filename per split (the byte-split train parts are concatenated into this).
ASSEMBLED_TAR = {
    "test": "DiDeMo_Videos_mp4_test.tar",
    "train": "DiDeMo_Videos_mp4_train.tar",
}
#: Parquet row-group size — small enough for streaming sub-shard reads in the loader.
ROW_GROUP_SIZE = 256


def _selected_splits(max_shards: int | None, fraction: float | None) -> list[str]:
    """Return the leading slice of sorted split names kept under the volume cap.

    Args:
        max_shards: Hard cap on the number of splits, or `None` for no hard cap.
        fraction: Share of the splits to keep in (0, 1], or `None` for no fractional cap.

    Returns:
        The kept split names (a prefix of `sorted(SPLIT_TAR_PATTERNS)`).
    """
    splits = sorted(SPLIT_TAR_PATTERNS)
    keep_count = len(splits)
    if fraction is not None:
        keep_count = min(keep_count, max(1, math.ceil(len(splits) * fraction)))
    if max_shards is not None:
        keep_count = min(keep_count, max_shards)
    kept = splits[:keep_count]
    print(f"[didemo] staging {len(kept)} of {len(splits)} split(s): {kept}", flush=True)
    return kept


def _assemble_train_tar(destination: str) -> None:
    """Concatenate the byte-split `DiDeMo_Videos_mp4_train.tar.part-*` into one tar, deleting parts.

    The mirror splits the ~43 GB train tar into 5 GB byte parts; a plain ordered concatenation
    reproduces the original archive. Writes to a temporary sibling then `os.replace` to stay atomic,
    and only runs when every expected part is present.

    Args:
        destination: The `<root>/didemo` directory holding the downloaded parts.
    """
    parts = sorted(
        os.path.join(destination, name)
        for name in os.listdir(destination)
        if name.startswith("DiDeMo_Videos_mp4_train.tar.part-")
    )
    if not parts:
        return
    assembled = os.path.join(destination, ASSEMBLED_TAR["train"])
    pending = assembled + ".pending"
    print(f"[didemo] assembling {len(parts)} train part(s) -> {assembled}", flush=True)
    with open(pending, "wb") as sink:
        for part in parts:
            with open(part, "rb") as source:
                shutil.copyfileobj(source, sink, length=16 * 1024 * 1024)
    os.replace(pending, assembled)
    for part in parts:
        os.unlink(part)


def _symlink_into_data(destination: str) -> None:
    """Point `<repo>/data/didemo` at the host data-root copy (the `download_poj104.py` guard)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    symlink_path = os.path.join(project_root, "data", "didemo")
    if os.path.realpath(symlink_path) == os.path.realpath(destination):
        print(f"[didemo] data already under {symlink_path}; no symlink needed", flush=True)
        return
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        if os.path.islink(symlink_path) and os.readlink(symlink_path) == destination:
            print(f"[didemo] symlink {symlink_path} -> {destination} already correct", flush=True)
            return
        os.unlink(symlink_path)
    os.makedirs(os.path.dirname(symlink_path), exist_ok=True)
    os.symlink(destination, symlink_path)
    print(f"[didemo] symlinked {symlink_path} -> {destination}", flush=True)


def _load_caption_index(captions_json: str) -> dict[str, list[str]]:
    """Load `wo_concat/didemo_<split>_wo_concat.json` into a `{clip_basename: [captions]}` index."""
    with open(captions_json, encoding="utf-8") as handle:
        records = json.load(handle)
    index: dict[str, list[str]] = {}
    for record in records:
        moments = record["caption"]
        if isinstance(moments, str):
            moments = [moments]
        index[os.path.basename(record["video"])] = [text for text in moments if text]
    return index


def repack_shard(tar_path: str, captions_json: str, *, keep_raw: bool, overwrite: bool) -> int:
    """Repack one DiDeMo split tar to the registry video schema. Returns 0 on success, 1 on failure.

    Args:
        tar_path: The assembled `DiDeMo_Videos_mp4_<split>.tar`.
        captions_json: The matching `wo_concat/didemo_<split>_wo_concat.json`.
        keep_raw: When `True`, keep the source tar after a verified repack.
        overwrite: When `False`, skip a split whose output parquet already exists.
    """
    split = "train" if "train" in os.path.basename(tar_path) else "test"
    out_dir = os.path.join(os.path.dirname(tar_path), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"didemo-{split}.parquet")
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {out_path} exists (pass --overwrite to rebuild)", flush=True)
        return 0

    caption_index = _load_caption_index(captions_json)
    mp4_bytes: list[bytes] = []
    captions: list[list[str]] = []
    video_paths: list[str] = []
    keys: list[str] = []
    payload_hashes: list[bytes] = []
    dropped_no_caption = 0
    with tarfile.open(tar_path, mode="r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".mp4"):
                continue
            base = os.path.basename(member.name)
            moments = caption_index.get(base)
            if moments is None:
                dropped_no_caption += 1
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            payload = handle.read()
            mp4_bytes.append(payload)
            captions.append(moments)
            video_paths.append(member.name)
            keys.append(base[: -len(".mp4")])
            payload_hashes.append(xxhash.xxh64(payload).digest())

    if not mp4_bytes:
        print(f"[empty] {tar_path}: no clips joined to a caption list", file=sys.stderr)
        return 1

    table = pa.table(
        {
            "mp4": pa.array(mp4_bytes, type=pa.binary()),
            "captions": pa.array(captions, type=pa.list_(pa.string())),
            "video": pa.array(video_paths, type=pa.string()),
            "key": pa.array(keys, type=pa.string()),
        }
    )
    pending = out_path + ".pending"
    pq.write_table(table, pending, compression="zstd", row_group_size=ROW_GROUP_SIZE)

    verify = pq.read_table(pending, columns=["mp4", "key"])
    verify_payloads = verify.column("mp4").to_pylist()
    verify_keys = verify.column("key").to_pylist()
    if len(verify_payloads) != len(mp4_bytes):
        os.unlink(pending)
        print(f"[FAIL] {tar_path}: verify row-count mismatch", file=sys.stderr)
        return 1
    for position, (key, payload) in enumerate(zip(verify_keys, verify_payloads, strict=True)):
        if key != keys[position] or xxhash.xxh64(payload).digest() != payload_hashes[position]:
            os.unlink(pending)
            print(f"[FAIL] {tar_path}: verify payload mismatch at row {position}", file=sys.stderr)
            return 1
    os.replace(pending, out_path)
    if not keep_raw:
        os.unlink(tar_path)
    print(
        f"[done] {os.path.basename(tar_path)} -> {os.path.basename(out_path)}  "
        f"({len(mp4_bytes)} clips, dropped {dropped_no_caption} without captions)",
        flush=True,
    )
    return 0


def download(
    data_root: str, max_shards: int | None, fraction: float | None, *, repack: bool, keep_raw: bool, overwrite: bool
) -> int:
    """Snapshot the selected DiDeMo splits, assemble the train tar, symlink, then fan out the repack."""
    destination = os.path.join(data_root, "didemo")
    os.makedirs(destination, exist_ok=True)
    splits = _selected_splits(max_shards, fraction)

    allow_patterns: list[str] = []
    for split in splits:
        allow_patterns.append(CAPTIONS_JSON[split])
        allow_patterns.extend(SPLIT_TAR_PATTERNS[split])
    print(f"[didemo] target: {destination}", flush=True)
    snapshot_download(
        repo_id=DIDEMO_REPO,
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    if "train" in splits:
        _assemble_train_tar(destination)
    _symlink_into_data(destination)

    if not repack:
        print("[didemo] --no-repack set; raw tars left in place", flush=True)
        return 0

    worker_jobs: list[str] = []
    for split in splits:
        tar_path = os.path.join(destination, ASSEMBLED_TAR[split])
        captions_json = os.path.join(destination, CAPTIONS_JSON[split])
        if os.path.exists(tar_path):
            worker_jobs.append(f"{tar_path}\t{captions_json}")
    if not worker_jobs:
        print("[didemo] no tars to repack", flush=True)
        return 0

    this_file = os.path.abspath(__file__)
    keep_flag = " --keep-raw" if keep_raw else ""
    over_flag = " --overwrite" if overwrite else ""
    command = (
        f"parallel --colsep '\\t' -j2 uv run {this_file} --repack-shard {{1}} --captions {{2}}{keep_flag}{over_flag}"
    )
    print(f"[didemo] fanning out {len(worker_jobs)} repack worker(s) via parallel", flush=True)
    return subprocess.run(command, input="\n".join(worker_jobs), text=True, shell=True, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Host data root for `data/didemo` (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--max-shards", type=int, default=None, help="Cap the pull to the first N splits (default: all)."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Cap the pull to the leading fraction (0, 1] of splits (default: all).",
    )
    parser.add_argument("--no-repack", action="store_true", help="Download and assemble only; skip the repack fan-out.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep the source tar(s) after a verified repack.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if an output parquet already exists.")
    # Hidden per-shard worker mode: this file self-invokes it via `parallel` after the download.
    parser.add_argument("--repack-shard", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--captions", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.repack_shard is not None:
        if args.captions is None:
            parser.error("--repack-shard requires --captions")
        return repack_shard(args.repack_shard, args.captions, keep_raw=args.keep_raw, overwrite=args.overwrite)

    return download(
        args.data_root,
        args.max_shards,
        args.fraction,
        repack=not args.no_repack,
        keep_raw=args.keep_raw,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
