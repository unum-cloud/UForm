# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=21.0.0",
#     "yt-dlp",
# ]
# ///
"""Build the VATEX multilingual video-caption corpus by fetching each clip from YouTube.

Modality: video + text.  VATEX ships caption JSON only — `vatex_training_v1.0.json` and
`vatex_validation_v1.0.json` from the official site (https://eric-xw.github.io/vatex-website/) — where
each entry carries a `videoID` of the form `<youtube_id>_<start:06d>_<end:06d>` plus `enCap` (10 English
captions) and `chCap` (10 Chinese captions).  The clips themselves must be fetched.  This one script
pulls the caption JSON, caps the clip count with `--fraction`, shards the work, and self-invokes
`--repack-shard` once per shard under GNU `parallel`.  Each shard worker fetches the
clip with `python_data/fetch_media_clips.py` (an H.264 `bestvideo+bestaudio` mux trimmed to the
clip's 10-second window) and writes a registry-schema parquet:

    captions: list<string>   (the 20 captions per clip: 10 English + 10 Chinese, multilingual)
    mp4:      binary         (the trimmed clip bytes)
    video_id: string         (the source watch id, for cross-dataset dedup)

Per-shard yields are merged into `<out-dir>/fetch_manifest.json`; dead / private / geo-blocked ids are
dropped (never silently truncated), so expect a real-world video yield around 60-80%.

This corpus is research-only: fetching is subject to YouTube's Terms of Service and the per-video
rights, and the clips are not redistributable.

CLI usage:
    uv run python_data/download_vatex.py                                  # all caption clips, parallel fetch
    uv run python_data/download_vatex.py --fraction 0.01 --num-shards 8   # ~1% slice
    uv run python_data/download_vatex.py --out-dir $UFORM_DATA_ROOT/vatex --jobs 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import urllib.request
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

#: Official VATEX caption JSON (validation first — it is small and good for a quick smoke).
_CAPTION_JSON_URLS = (
    "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json",
    "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json",
)
_OUTPUT_PREFIX = "vatex"
_DEFAULT_OUT_DIR = Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "vatex"
_CAPTIONS_TYPE = pa.list_(pa.string())
_THIS_SCRIPT = Path(__file__).resolve()


def _parse_video_id(video_id: str) -> tuple[str, float, float]:
    """Split a VATEX `videoID` (`<watch_id>_<start:06d>_<end:06d>`) into watch id and window seconds.

    The watch id may itself contain underscores, so the two trailing `_`-separated fields (start, end)
    are split off from the right.

    Args:
        video_id: A VATEX `videoID` value.

    Returns:
        A `(watch_id, start_seconds, end_seconds)` triple.
    """
    watch_id, start_text, end_text = video_id.rsplit("_", 2)
    return watch_id, float(int(start_text)), float(int(end_text))


def _download_json(url: str, destination: Path) -> list[dict[str, object]]:
    """Download a VATEX caption JSON to `destination` (cached) and return its parsed entries."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"download_vatex: fetching {url}")
        urllib.request.urlretrieve(url, destination)  # noqa: S310 — fixed trusted https VATEX host
    return json.loads(destination.read_text())


def _build_records(args: argparse.Namespace) -> list[dict[str, object]]:
    """Pull every caption JSON, cap to the leading `--fraction`, and build clip records."""
    cache_dir = args.out_dir / ".captions"
    entries: list[dict[str, object]] = []
    for url in _CAPTION_JSON_URLS:
        entries += _download_json(url, cache_dir / Path(url).name)
    keep_count = max(1, math.ceil(len(entries) * args.fraction)) if entries else 0
    print(f"download_vatex: keeping {keep_count} of {len(entries)} caption clips (fraction={args.fraction})")
    records: list[dict[str, object]] = []
    for entry in entries[:keep_count]:
        watch_id, start, end = _parse_video_id(str(entry["videoID"]))
        captions = list(entry.get("enCap", [])) + list(entry.get("chCap", []))
        record = MediaClip(source_id=watch_id, kind=MediaKind.VIDEO, start_seconds=start, end_seconds=end).to_json()
        record["captions"] = captions
        records.append(record)
    return records


def repack_shard(
    input_jsonl: Path, output_parquet: Path, manifest_path: Path, *, retries: int, timeout_seconds: float
) -> int:
    """Fetch one shard's clips and write the `[captions, mp4, video_id]` registry parquet."""
    manifest = FetchManifest()
    captions: list[list[str]] = []
    mp4: list[bytes] = []
    video_id: list[str] = []
    for record, result in fetch_shard(input_jsonl, manifest=manifest, retries=retries, timeout_seconds=timeout_seconds):
        if not result.succeeded or result.payload is None:
            continue
        captions.append([str(caption) for caption in record["captions"]])  # type: ignore[union-attr]
        mp4.append(result.payload)
        video_id.append(result.clip.source_id)
    manifest.write(manifest_path)
    if mp4:
        table = pa.table(
            {
                "captions": pa.array(captions, type=_CAPTIONS_TYPE),
                "mp4": pa.array(mp4, type=pa.binary()),
                "video_id": pa.array(video_id, type=pa.string()),
            }
        )
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        partial = output_parquet.with_suffix(".parquet.partial")
        pq.write_table(table, partial, compression="zstd", row_group_size=256)
        partial.replace(output_parquet)
    print(
        f"download_vatex: shard {input_jsonl.name} -> {len(mp4)} clips "
        f"({len(manifest.succeeded)}/{len(manifest.attempted)} yield)"
    )
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    """Pull caption JSON, shard the work, fan out the fetch under GNU parallel, and report the yield."""
    output_dir = args.out_dir / "data"
    sentinel = args.out_dir / ".built"
    if sentinel.exists() and not args.force:
        print(f"download_vatex: already built at {args.out_dir}; skip (pass --force to rebuild)")
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
        f"download_vatex: built {summary['succeeded_count']} clips of {summary['attempted_count']} attempted "
        f"({float(summary['yield_fraction']) * 100:.1f}% yield) under {output_dir}"
    )
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch + repack VATEX bilingual video captions into registry parquet.")
    parser.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR, help=f"Destination root (default: {_DEFAULT_OUT_DIR})."
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="Leading share (0, 1] of the caption clips to fetch."
    )
    parser.add_argument("--num-shards", type=int, default=128, help="Number of shard processes to split the work into.")
    parser.add_argument("--jobs", type=int, default=8, help="Concurrent shard processes under GNU parallel.")
    parser.add_argument("--retries", type=int, default=3, help="Per-clip transient retry budget.")
    parser.add_argument(
        "--timeout-seconds", type=float, default=240.0, help="Wall-clock ceiling per yt-dlp invocation."
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
            print("download_vatex: --repack-shard needs --shard-output and --shard-manifest", file=sys.stderr)
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
