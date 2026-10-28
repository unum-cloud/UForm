# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Pull MeViS (referring motion-expression video segmentation) and repack it to the video schema.

MeViS pairs ~2K short clips with ~33K natural-language motion expressions ("Horse running around",
"Horse sitting down") — each expression refers to one moving object in the clip and originally carries
a dense per-frame mask. We DROP the masks (a mask-head loader is out of scope) and keep only the clean
language <-> video signal: every clip's expressions become a `captions: list<string>` multi-positive
anchor against the clip's pixels. The expressions describe sub-clip object motion rather than a single
global caption, so this is a noisier retrieval signal than a captioning set — wire it as supplementary
video-text, not as a precision benchmark.

The `xing0047/mevis` HF mirror ships, per split, `JPEGImages.tar` (the decoded frames, ~16 GB train /
~2 GB valid) and `meta_expressions.json` (the expressions + frame ordering). MeViS distributes frames,
not encoded clips, so the worker re-encodes each clip's ordered JPEG frames into an MP4 with `ffmpeg`
(a system binary, no pip dependency) to land the registry `mp4: binary` column.

This single file is BOTH the orchestrator and the per-shard repack worker:

  * Orchestrator (default): snapshots the selected splits, symlinks `data/mevis` at the host data
    root, then fans out one repack worker per split via GNU `parallel` (each self-invokes this file
    with `--repack-shard`).
  * Worker (`--repack-shard <JPEGImages.tar> --expressions <meta.json> --split <name>`): groups the
    tar's frames by clip, encodes each clip to MP4, joins its expression list, and writes
    `<root>/mevis/data/mevis-<split>.parquet`:

        mp4:         binary        # the clip re-encoded from its ordered JPEG frames
        captions:    list<string>  # the clip's referring motion expressions (multi-positive anchor)
        video:       string        # the MeViS clip id
        num_frames:  int32         # frame count that went into the encode

Atomic write (temporary sibling file + `os.replace`); the source tar is deleted on success unless
`--keep-raw` is passed. Mirrors the tar->parquet pattern of `repack_kinetics400_to_parquet.py` and the
symlink-into-`data/` guard of `download_poj104.py`.

Volume cap: the cap unit is the split (sorted `["train", "valid", "valid_u"]`); with neither
`--fraction` nor `--max-shards` all splits are pulled, either flag keeps the leading slice.

Usage:
    uv run scripts/data/download_mevis.py                       # all splits, then repack
    uv run scripts/data/download_mevis.py --max-shards 1        # train split only
    uv run scripts/data/download_mevis.py --data-root $UFORM_DATA_ROOT --no-repack
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from acquire_shared import cap_shards, data_root
from huggingface_hub import snapshot_download

#: Default host data root — overridable via `UFORM_DATA_ROOT`; the symlink target `data/mevis`.
DEFAULT_DATA_ROOT = str(data_root())
#: HF mirror that ships per-split JPEG frame tars plus the expression/frame-order JSON.
MEVIS_REPO = "xing0047/mevis"
#: Splits the mirror exposes with BOTH frames and expressions (we skip the mask-only `mask_dict.json`).
SPLITS = ("train", "valid", "valid_u")
#: Parquet row-group size — small enough for streaming sub-shard reads in the loader.
ROW_GROUP_SIZE = 64
#: Default playback rate for the re-encode; MeViS frames are uniformly sampled, so any steady rate works.
DEFAULT_FPS = 5


def _selected_splits(max_shards: int | None, fraction: float | None) -> list[str]:
    """Return the leading slice of sorted split names kept under the volume cap."""
    splits = sorted(SPLITS)
    kept = cap_shards(splits, fraction, max_shards)
    print(f"[mevis] staging {len(kept)} of {len(splits)} split(s): {kept}", flush=True)
    return kept


def _symlink_into_data(destination: str) -> None:
    """Point `<repo>/data/mevis` at the host data-root copy (the `download_poj104.py` guard)."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    symlink_path = os.path.join(project_root, "data", "mevis")
    if os.path.realpath(symlink_path) == os.path.realpath(destination):
        print(f"[mevis] data already under {symlink_path}; no symlink needed", flush=True)
        return
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        if os.path.islink(symlink_path) and os.readlink(symlink_path) == destination:
            print(f"[mevis] symlink {symlink_path} -> {destination} already correct", flush=True)
            return
        os.unlink(symlink_path)
    os.makedirs(os.path.dirname(symlink_path), exist_ok=True)
    os.symlink(destination, symlink_path)
    print(f"[mevis] symlinked {symlink_path} -> {destination}", flush=True)


def _load_expression_index(expressions_json: str) -> dict[str, list[str]]:
    """Load `meta_expressions.json` into a `{clip_id: [expressions]}` index (mask fields ignored)."""
    with open(expressions_json, encoding="utf-8") as handle:
        videos = json.load(handle)["videos"]
    index: dict[str, list[str]] = {}
    for clip_id, record in videos.items():
        expressions = record["expressions"]
        ordered = [expressions[number]["exp"] for number in sorted(expressions, key=int)]
        index[clip_id] = [text for text in ordered if text]
    return index


def _encode_frames_to_mp4(frames: list[bytes], fps: int) -> bytes:
    """Encode a clip's ordered JPEG frame bytes into MP4 bytes via `ffmpeg` (image2pipe -> libx264).

    Some MeViS clips carry odd frame dimensions (e.g. clip `db801a188332` is 1281x720). libx264 with
    `yuv420p` 4:2:0 chroma subsampling can only sample even width/height, so an odd dimension aborts the
    encoder with `width not divisible by 2` and ffmpeg exits 187 — the staging failure this set hit. We
    round each axis down to the nearest even size with `scale=trunc(iw/2)*2:trunc(ih/2)*2`, dropping at
    most one boundary row/column (imperceptible for retrieval) so the encode is robust to odd sources;
    already-even frames pass through unchanged.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as sink:
        out_path = sink.name
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        subprocess.run(command, input=b"".join(frames), check=True, capture_output=True)
        with open(out_path, "rb") as handle:
            return handle.read()
    finally:
        os.unlink(out_path)


def _clip_id_and_stem(member_name: str) -> tuple[str, str] | None:
    """Split a `.../JPEGImages/<clip>/<stem>.jpg` member into `(clip, stem)`, or `None` if not a frame."""
    if not member_name.lower().endswith((".jpg", ".jpeg")):
        return None
    parts = member_name.split("/")
    if len(parts) < 2:
        return None
    return parts[-2], os.path.splitext(parts[-1])[0]


def repack_shard(tar_path: str, expressions_json: str, split: str, *, fps: int, keep_raw: bool, overwrite: bool) -> int:
    """Repack one MeViS split (frame tar + expressions) to the video schema. Returns 0 on success.

    Frames stream out of the tar grouped by clip (MeViS stores each clip's frames contiguously); on a
    clip boundary the buffered frames are encoded to MP4, joined to the clip's expression list, and
    appended to the open parquet writer so peak memory stays at one clip's frames.
    """
    out_dir = os.path.join(os.path.dirname(tar_path), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mevis-{split}.parquet")
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {out_path} exists (pass --overwrite to rebuild)", flush=True)
        return 0

    expression_index = _load_expression_index(expressions_json)
    schema = pa.schema(
        [("mp4", pa.binary()), ("captions", pa.list_(pa.string())), ("video", pa.string()), ("num_frames", pa.int32())]
    )
    pending = out_path + ".pending"
    written = 0
    dropped_no_expression = 0

    def flush_clip(writer: pq.ParquetWriter, clip_id: str, frames: list[tuple[str, bytes]]) -> int:
        moments = expression_index.get(clip_id)
        if not moments:
            return 0
        ordered = [payload for _, payload in sorted(frames, key=lambda item: item[0])]
        mp4 = _encode_frames_to_mp4(ordered, fps)
        batch = pa.record_batch(
            {
                "mp4": pa.array([mp4], type=pa.binary()),
                "captions": pa.array([moments], type=pa.list_(pa.string())),
                "video": pa.array([clip_id], type=pa.string()),
                "num_frames": pa.array([len(ordered)], type=pa.int32()),
            },
            schema=schema,
        )
        writer.write_batch(batch)
        return 1

    writer = pq.ParquetWriter(pending, schema, compression="zstd")
    try:
        current_clip: str | None = None
        current_frames: list[tuple[str, bytes]] = []
        with tarfile.open(tar_path, mode="r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parsed = _clip_id_and_stem(member.name)
                if parsed is None:
                    continue
                clip_id, stem = parsed
                if clip_id != current_clip and current_frames:
                    emitted = flush_clip(writer, current_clip, current_frames)
                    written += emitted
                    dropped_no_expression += 1 - emitted
                    current_frames = []
                current_clip = clip_id
                handle = archive.extractfile(member)
                if handle is not None:
                    current_frames.append((stem, handle.read()))
            if current_frames:
                emitted = flush_clip(writer, current_clip, current_frames)
                written += emitted
                dropped_no_expression += 1 - emitted
    finally:
        writer.close()

    if written == 0:
        os.unlink(pending)
        print(f"[empty] {tar_path}: no clips joined to an expression list", file=sys.stderr)
        return 1
    if pq.ParquetFile(pending).metadata.num_rows != written:
        os.unlink(pending)
        print(f"[FAIL] {tar_path}: verify row-count mismatch", file=sys.stderr)
        return 1
    os.replace(pending, out_path)
    if not keep_raw:
        os.unlink(tar_path)
    print(
        f"[done] {os.path.basename(tar_path)} -> {os.path.basename(out_path)}  "
        f"({written} clips encoded, dropped {dropped_no_expression} without expressions)",
        flush=True,
    )
    return 0


def download(
    data_root: str,
    max_shards: int | None,
    fraction: float | None,
    *,
    fps: int,
    repack: bool,
    keep_raw: bool,
    overwrite: bool,
) -> int:
    """Snapshot the selected MeViS splits, symlink, then fan out the per-split repack workers."""
    destination = os.path.join(data_root, "mevis")
    os.makedirs(destination, exist_ok=True)
    splits = _selected_splits(max_shards, fraction)

    allow_patterns: list[str] = []
    for split in splits:
        allow_patterns.append(f"{split}/JPEGImages.tar")
        allow_patterns.append(f"{split}/meta_expressions.json")
    print(f"[mevis] target: {destination}", flush=True)
    snapshot_download(
        repo_id=MEVIS_REPO, repo_type="dataset", local_dir=destination, allow_patterns=allow_patterns, max_workers=8
    )
    _symlink_into_data(destination)

    if not repack:
        print("[mevis] --no-repack set; raw tars left in place", flush=True)
        return 0

    worker_jobs: list[str] = []
    for split in splits:
        tar_path = os.path.join(destination, split, "JPEGImages.tar")
        expressions_json = os.path.join(destination, split, "meta_expressions.json")
        if os.path.exists(tar_path) and os.path.exists(expressions_json):
            worker_jobs.append(f"{tar_path}\t{expressions_json}\t{split}")
    if not worker_jobs:
        print("[mevis] no split tars to repack", flush=True)
        return 0

    this_file = os.path.abspath(__file__)
    keep_flag = " --keep-raw" if keep_raw else ""
    over_flag = " --overwrite" if overwrite else ""
    command = (
        "parallel --colsep '\\t' -j3 "
        f"uv run {this_file} --repack-shard {{1}} --expressions {{2}} --split {{3}} --fps {fps}{keep_flag}{over_flag}"
    )
    print(f"[mevis] fanning out {len(worker_jobs)} repack worker(s) via parallel", flush=True)
    return subprocess.run(command, input="\n".join(worker_jobs), text=True, shell=True, check=False).returncode


def main() -> int:
    """Download MeViS and repack it, or repack the single shard named by the hidden worker flags."""
    parser =argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Host data root for `data/mevis` (default: {DEFAULT_DATA_ROOT}).",
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
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS, help=f"Re-encode playback rate (default: {DEFAULT_FPS})."
    )
    parser.add_argument("--no-repack", action="store_true", help="Download only; skip the repack fan-out.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep the source frame tar(s) after a verified repack.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if an output parquet already exists.")
    # Hidden per-shard worker mode: this file self-invokes it via `parallel` after the download.
    parser.add_argument("--repack-shard", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expressions", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--split", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.repack_shard is not None:
        if args.expressions is None or args.split is None:
            parser.error("--repack-shard requires --expressions and --split")
        return repack_shard(
            args.repack_shard,
            args.expressions,
            args.split,
            fps=args.fps,
            keep_raw=args.keep_raw,
            overwrite=args.overwrite,
        )

    return download(
        args.data_root,
        args.max_shards,
        args.fraction,
        fps=args.fps,
        repack=not args.no_repack,
        keep_raw=args.keep_raw,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
