# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
#     "yt-dlp",
# ]
# ///
"""Build the Auto-ACD audio-caption corpus by fetching each clip's audio from YouTube.

Modality: audio + text.  Auto-ACD (`Loie/Auto-ACD`) ships a single CSV of `(youtube_id, caption)`
rows — 1,920,945 of them — and nothing else; the audio has to be fetched at build time.  This one
script does the whole job: it pulls the CSV with `huggingface_hub`, caps the row count with
`--fraction` (default ~0.05, because the full 1.9M-clip fetch is impractical and research-only), shards
the work, and self-invokes `--repack-shard` once per shard under GNU `parallel`.  Each
shard worker fetches the audio with `python_data/fetch_media_clips.py` (bestaudio extracted to mp3) and
writes a registry-schema parquet:

    caption: string
    audio:   struct<bytes: binary, path: string>   (the mp3 payload + a `<id>.mp3` path tag)

Yields are audited per shard and merged into `<out-dir>/fetch_manifest.json`; dead / private /
geo-blocked ids are dropped (never silently truncated), so expect a real-world audio yield around
60-80%.  Auto-ACD is derived from AudioSet/VGGSound, so many `youtube_id` values carry an AudioSet-style
`_<6-digit-start-second>` suffix (and an occasional leading `Y`); `_parse_clip_id` best-effort recovers
the watch id and the 10-second window, and any id that does not resolve simply drops.

This corpus is research-only: fetching is subject to YouTube's Terms of Service and the per-video
rights, and the audio is not redistributable.

CLI usage:
    uv run python_data/download_auto_acd.py                                  # ~5% of 1.9M, parallel fetch
    uv run python_data/download_auto_acd.py --fraction 0.001 --num-shards 4  # tiny slice
    uv run python_data/download_auto_acd.py --out-dir $UFORM_DATA_ROOT/auto-acd --jobs 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
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
from huggingface_hub import hf_hub_download

_REPO_ID = "Loie/Auto-ACD"
_CSV_FILENAME = "train.csv"
_OUTPUT_PREFIX = "auto-acd"
_DEFAULT_OUT_DIR = Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "auto-acd"
#: Auto-ACD inherits AudioSet's fixed 10-second clip window for ids that carry a start-second suffix.
_CLIP_WINDOW_SECONDS = 10.0
#: AudioSet-style segment suffix: a real watch id followed by a zero-padded 6-digit start second.
_SEGMENT_SUFFIX_RE = re.compile(r"^(?P<watch_id>.+)_(?P<start>\d{6})$")
#: The registry audio column: an mp3 payload wrapped in the same `struct<bytes,path>` as AudioCaps.
_AUDIO_STRUCT_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
_THIS_SCRIPT = Path(__file__).resolve()


def _parse_clip_id(raw_id: str) -> tuple[str, float | None, float | None]:
    """Recover the YouTube watch id and optional `[start, end]` window from an Auto-ACD id.

    Auto-ACD ids come in two shapes: a bare watch id (fetch the whole audio) or an AudioSet-style
    `<watch_id>_<start:06d>` segment id (fetch a 10-second window).  A leading `Y` — AudioSet's file
    prefix — is stripped when doing so yields the canonical 11-character watch id.  This is best-effort;
    an id that resolves to a dead clip simply drops during fetch.

    Args:
        raw_id: The raw `youtube_id` cell from the CSV.

    Returns:
        A `(watch_id, start_seconds, end_seconds)` triple; the seconds are `None` for whole-clip ids.
    """
    match = _SEGMENT_SUFFIX_RE.match(raw_id)
    if match:
        watch_id = match.group("watch_id")
        start = float(int(match.group("start")))
        window = (start, start + _CLIP_WINDOW_SECONDS)
    else:
        watch_id = raw_id
        window = (None, None)
    if len(watch_id) == 12 and watch_id.startswith("Y"):
        watch_id = watch_id[1:]
    return watch_id, window[0], window[1]


def _read_caption_rows(csv_path: Path, fraction: float) -> list[tuple[str, str]]:
    """Read `(youtube_id, caption)` rows from the CSV, capped to the leading `fraction` of rows."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [(row["youtube_id"], row["caption"]) for row in reader if row.get("youtube_id") and row.get("caption")]
    keep_count = max(1, math.ceil(len(rows) * fraction)) if rows else 0
    print(f"download_auto_acd: keeping {keep_count} of {len(rows)} caption rows (fraction={fraction})")
    return rows[:keep_count]


def _build_records(rows: list[tuple[str, str]]) -> list[dict[str, object]]:
    """Turn `(youtube_id, caption)` rows into JSON clip records for `plan_shards`."""
    records: list[dict[str, object]] = []
    for raw_id, caption in rows:
        watch_id, start, end = _parse_clip_id(raw_id)
        record = MediaClip(source_id=watch_id, kind=MediaKind.AUDIO, start_seconds=start, end_seconds=end).to_json()
        record["caption"] = caption
        records.append(record)
    return records


def repack_shard(
    input_jsonl: Path, output_parquet: Path, manifest_path: Path, *, retries: int, timeout_seconds: float
) -> int:
    """Fetch one shard's audio clips and write the `[caption, audio]` registry parquet.

    Args:
        input_jsonl: Shard file of clip records (from `plan_shards`).
        output_parquet: Destination parquet for this shard.
        manifest_path: Destination for this shard's yield manifest.
        retries: Per-clip transient retry budget.
        timeout_seconds: Per-invocation wall-clock ceiling.

    Returns:
        Process exit code (always 0; a fully-dropped shard is a valid — if empty — outcome).
    """
    manifest = FetchManifest()
    captions: list[str] = []
    audio: list[dict[str, object]] = []
    for record, result in fetch_shard(input_jsonl, manifest=manifest, retries=retries, timeout_seconds=timeout_seconds):
        if not result.succeeded or result.payload is None:
            continue
        captions.append(str(record["caption"]))
        audio.append({"bytes": result.payload, "path": f"{result.clip.source_id}.{result.extension}"})
    manifest.write(manifest_path)
    if captions:
        table = pa.table(
            {"caption": pa.array(captions, type=pa.string()), "audio": pa.array(audio, type=_AUDIO_STRUCT_TYPE)}
        )
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        partial = output_parquet.with_suffix(".parquet.partial")
        pq.write_table(table, partial, compression="zstd", row_group_size=512)
        partial.replace(output_parquet)
    print(
        f"download_auto_acd: shard {input_jsonl.name} -> {len(captions)} clips "
        f"({len(manifest.succeeded)}/{len(manifest.attempted)} yield)"
    )
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    """Pull the CSV, shard the work, fan out the fetch under GNU parallel, and report the merged yield."""
    output_dir = args.out_dir / "data"
    sentinel = args.out_dir / ".built"
    if sentinel.exists() and not args.force:
        print(f"download_auto_acd: already built at {args.out_dir}; skip (pass --force to rebuild)")
        return 0
    csv_path = Path(hf_hub_download(repo_id=_REPO_ID, filename=_CSV_FILENAME, repo_type="dataset"))
    records = _build_records(_read_caption_rows(csv_path, args.fraction))
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
        f"download_auto_acd: built {summary['succeeded_count']} clips of {summary['attempted_count']} attempted "
        f"({float(summary['yield_fraction']) * 100:.1f}% yield) under {output_dir}"
    )
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch + repack Auto-ACD audio captions into registry parquet.")
    parser.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR, help=f"Destination root (default: {_DEFAULT_OUT_DIR})."
    )
    parser.add_argument(
        "--fraction", type=float, default=0.05, help="Leading share (0, 1] of the 1.9M caption rows to fetch."
    )
    parser.add_argument("--num-shards", type=int, default=256, help="Number of shard processes to split the work into.")
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
            print("download_auto_acd: --repack-shard needs --shard-output and --shard-manifest", file=sys.stderr)
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
