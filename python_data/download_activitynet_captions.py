# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=21.0.0",
#     "yt-dlp",
# ]
# ///
"""Build the ActivityNet Captions video corpus by fetching each video from YouTube.

Modality: video + text.  ActivityNet Captions (`HuggingFaceM4/ActivitiyNet_Captions` — note the typo
in the repo id) provides only dense per-video caption metadata: each of the ~20K videos carries on
average 3.65 temporally localized sentences.  The HuggingFace loader script itself pulls the metadata
from `captions.zip` on the original author's host (https://cs.stanford.edu/people/ranjaykrishna/densevid/),
so this script does the same: it downloads `captions.zip`, reads `train.json` (and `val_1.json`), whose
keys are `v_<youtube_id>` with `timestamps` (`[start, end]` pairs) and `sentences`.  The videos
themselves must be fetched.  This one script caps the video count with `--fraction`, shards the work,
and self-invokes `--repack-shard` once per shard under GNU `parallel`.  Each shard
worker fetches the whole video with `python_data/fetch_media_clips.py` (an H.264 `bestvideo+bestaudio`
mux) and writes a registry-schema parquet:

    captions:        list<string>   (all of the video's localized sentences)
    mp4:             binary         (the full video bytes)
    video_id:        string         (the source watch id)
    captions_starts: list<double>   (per-sentence segment start seconds)
    captions_ends:   list<double>   (per-sentence segment end seconds)

Per-shard yields are merged into `<out-dir>/fetch_manifest.json`; dead / private / geo-blocked ids are
dropped (never silently truncated), so expect a real-world video yield around 60-80%.

This corpus is research-only: fetching is subject to YouTube's Terms of Service and the per-video
rights, and the videos are not redistributable.

CLI usage:
    uv run python_data/download_activitynet_captions.py                                 # all videos, parallel
    uv run python_data/download_activitynet_captions.py --fraction 0.01 --num-shards 8  # ~1% slice
    uv run python_data/download_activitynet_captions.py --out-dir $UFORM_DATA_ROOT/activitynet-captions --jobs 16
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fetch_media_clips import (
    FetchManifest,
    MediaClip,
    MediaKind,
    fetch_shard,
    plan_shards,
    run_shard_workers,
    summarise_shard_manifests,
)

#: The dense-captioning metadata archive the HuggingFace loader script also pulls.
_CAPTIONS_ZIP_URL = "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"
#: Caption files inside the zip to ingest (train + the first validation split).
_CAPTION_MEMBERS = ("train.json", "val_1.json")
_OUTPUT_PREFIX = "activitynet"
_DEFAULT_OUT_DIR = (
    Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "activitynet-captions"
)
_CAPTIONS_TYPE = pa.list_(pa.string())
_SECONDS_TYPE = pa.list_(pa.float64())
_THIS_SCRIPT = Path(__file__).resolve()


def _load_caption_metadata(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """Download `captions.zip` (cached) and merge the requested caption-member JSON objects."""
    cache_zip = args.out_dir / ".captions" / "captions.zip"
    if not cache_zip.exists():
        cache_zip.parent.mkdir(parents=True, exist_ok=True)
        print(f"download_activitynet_captions: fetching {_CAPTIONS_ZIP_URL}")
        urllib.request.urlretrieve(_CAPTIONS_ZIP_URL, cache_zip)  # noqa: S310 — fixed trusted https Stanford host
    merged: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(io.BytesIO(cache_zip.read_bytes())) as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        for member in _CAPTION_MEMBERS:
            if member in names:
                merged.update(json.loads(archive.read(names[member])))
    return merged


def _build_records(args: argparse.Namespace) -> list[dict[str, object]]:
    """Load the caption metadata, cap to the leading `--fraction`, and build per-video clip records."""
    metadata = _load_caption_metadata(args)
    keys = list(metadata)
    keep_count = max(1, math.ceil(len(keys) * args.fraction)) if keys else 0
    print(f"download_activitynet_captions: keeping {keep_count} of {len(keys)} videos (fraction={args.fraction})")
    records: list[dict[str, object]] = []
    for key in keys[:keep_count]:
        info = metadata[key]
        # Keys are `v_<youtube_id>`; the leading `v_` is the ActivityNet prefix, not part of the id.
        watch_id = key[2:] if key.startswith("v_") else key
        timestamps = info.get("timestamps", [])
        record = MediaClip(source_id=watch_id, kind=MediaKind.VIDEO).to_json()
        record["captions"] = [str(sentence) for sentence in info.get("sentences", [])]
        record["captions_starts"] = [float(pair[0]) for pair in timestamps]
        record["captions_ends"] = [float(pair[1]) for pair in timestamps]
        records.append(record)
    return records


def repack_shard(
    input_jsonl: Path, output_parquet: Path, manifest_path: Path, *, retries: int, timeout_seconds: float
) -> int:
    """Fetch one shard's videos and write the dense-caption registry parquet."""
    manifest = FetchManifest()
    captions: list[list[str]] = []
    mp4: list[bytes] = []
    video_id: list[str] = []
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    for record, result in fetch_shard(input_jsonl, manifest=manifest, retries=retries, timeout_seconds=timeout_seconds):
        if not result.succeeded or result.payload is None:
            continue
        captions.append([str(caption) for caption in record["captions"]])  # type: ignore[union-attr]
        mp4.append(result.payload)
        video_id.append(result.clip.source_id)
        starts.append([float(value) for value in record["captions_starts"]])  # type: ignore[union-attr]
        ends.append([float(value) for value in record["captions_ends"]])  # type: ignore[union-attr]
    manifest.write(manifest_path)
    if mp4:
        table = pa.table(
            {
                "captions": pa.array(captions, type=_CAPTIONS_TYPE),
                "mp4": pa.array(mp4, type=pa.binary()),
                "video_id": pa.array(video_id, type=pa.string()),
                "captions_starts": pa.array(starts, type=_SECONDS_TYPE),
                "captions_ends": pa.array(ends, type=_SECONDS_TYPE),
            }
        )
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        partial = output_parquet.with_suffix(".parquet.partial")
        pq.write_table(table, partial, compression="zstd", row_group_size=128)
        partial.replace(output_parquet)
    print(
        f"download_activitynet_captions: shard {input_jsonl.name} -> {len(mp4)} videos "
        f"({len(manifest.succeeded)}/{len(manifest.attempted)} yield)"
    )
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    """Pull caption metadata, shard the work, fan out the fetch under GNU parallel, and report the yield."""
    output_dir = args.out_dir / "data"
    sentinel = args.out_dir / ".built"
    if sentinel.exists() and not args.force:
        print(f"download_activitynet_captions: already built at {args.out_dir}; skip (pass --force to rebuild)")
        return 0
    records = _build_records(args)
    staging_dir = args.out_dir / ".staging"
    if args.force and staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    plans = plan_shards(
        records,
        num_shards=args.num_shards,
        staging_dir=staging_dir,
        output_dir=output_dir,
        output_prefix=_OUTPUT_PREFIX,
    )
    run_shard_workers(_THIS_SCRIPT, plans, jobs=args.jobs)
    summary = summarise_shard_manifests(plans)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fetch_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    sentinel.touch()
    print(
        f"download_activitynet_captions: built {summary['succeeded_count']} videos of "
        f"{summary['attempted_count']} attempted ({float(summary['yield_fraction']) * 100:.1f}% yield) under {output_dir}"
    )
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch + repack ActivityNet Captions dense video captions into registry parquet."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR, help=f"Destination root (default: {_DEFAULT_OUT_DIR})."
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="Leading share (0, 1] of the videos to fetch.")
    parser.add_argument("--num-shards", type=int, default=128, help="Number of shard processes to split the work into.")
    parser.add_argument("--jobs", type=int, default=8, help="Concurrent shard processes under GNU parallel.")
    parser.add_argument("--retries", type=int, default=3, help="Per-clip transient retry budget.")
    parser.add_argument(
        "--timeout-seconds", type=float, default=360.0, help="Wall-clock ceiling per yt-dlp invocation."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if the sentinel file exists.")
    parser.add_argument("--repack-shard", type=Path, default=None, help="Internal: fetch+repack a single shard JSONL.")
    parser.add_argument("--shard-output", type=Path, default=None, help="Internal: output parquet for --repack-shard.")
    parser.add_argument("--shard-manifest", type=Path, default=None, help="Internal: manifest path for --repack-shard.")
    return parser.parse_args()


def main() -> int:
    args = _parse_cli_args()
    if args.repack_shard is not None:
        if args.shard_output is None or args.shard_manifest is None:
            print(
                "download_activitynet_captions: --repack-shard needs --shard-output and --shard-manifest",
                file=sys.stderr,
            )
            return 2
        return repack_shard(
            args.repack_shard,
            args.shard_output,
            args.shard_manifest,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
