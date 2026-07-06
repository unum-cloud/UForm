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
rows — 1,920,945 of them — and nothing else; the audio has to be fetched at build time.  This thin
builder keeps ONLY its bespoke index read + id parse (`_clip_records`); the shared
`fetch_media_clips.acquire_clips` orchestrator does the rest — a Slurm `(rank, world)` split so each
worker fetches a disjoint `clips[rank::world]` stride of the corpus, intra-host GNU-`parallel` fan-out,
per-shard yt-dlp fetch (bestaudio extracted to mp3), and the uniform registry parquet:

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
    uv run python_data/download_auto_acd.py --fraction 0.001 --fanout 4      # tiny slice
    uv run python_data/download_auto_acd.py --rank 0 --world 16              # this worker's disjoint stride
    uv run python_data/download_auto_acd.py --print-plan --world 2 --rank 0  # dry-run: show the clip stride
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from fetch_media_clips import MediaClip, MediaKind, acquire_clips
from huggingface_hub import hf_hub_download

_REPO_ID = "Loie/Auto-ACD"
_CSV_FILENAME = "train.csv"
#: Auto-ACD inherits AudioSet's fixed 10-second clip window for ids that carry a start-second suffix.
_CLIP_WINDOW_SECONDS = 10.0
#: AudioSet-style segment suffix: a real watch id followed by a zero-padded 6-digit start second.
_SEGMENT_SUFFIX_RE = re.compile(r"^(?P<watch_id>.+)_(?P<start>\d{6})$")


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


def _clip_records(out_dir: Path, index_cap: int | None) -> list[dict[str, object]]:
    """Pull the Auto-ACD CSV and build one `{...MediaClip, caption}` record per `(youtube_id, caption)` row.

    Streams the CSV row by row and stops once `index_cap` records are built, so a capped run never
    materialises all 1.9M rows before the fetch cap trims them; `index_cap=None` reads the whole CSV.
    """
    csv_path = Path(hf_hub_download(repo_id=_REPO_ID, filename=_CSV_FILENAME, repo_type="dataset"))
    records: list[dict[str, object]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if index_cap is not None and len(records) >= index_cap:
                break
            raw_id, caption = row.get("youtube_id"), row.get("caption")
            if not raw_id or not caption:
                continue
            watch_id, start, end = _parse_clip_id(raw_id)
            record = MediaClip(source_id=watch_id, kind=MediaKind.AUDIO, start_seconds=start, end_seconds=end).to_json()
            record["caption"] = caption
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(
        acquire_clips(
            "auto_acd",
            clip_records=_clip_records,
            media_kind=MediaKind.AUDIO,
            out_name="auto-acd",
            media_column="audio",
            script=Path(__file__).resolve(),
            default_fraction=0.05,
            default_fanout=256,
            row_group_size=512,
        )
    )
