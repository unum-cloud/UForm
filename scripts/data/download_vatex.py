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
captions) and `chCap` (10 Chinese captions).  The clips themselves must be fetched.  This thin builder
keeps only its bespoke index read + id parse (`_clip_records`); the shared
`acquire_clips.acquire_clips` orchestrator does the rest — a Slurm `(rank, world)` split so each
worker fetches a disjoint `clips[rank::world]` stride, intra-host GNU-`parallel` fan-out, per-shard
yt-dlp fetch (an H.264 `bestvideo+bestaudio` mux trimmed to the clip's window), and the uniform registry
parquet:

    caption: string                                (the clip's 20 captions — 10 English + 10 Chinese — joined)
    mp4:     struct<bytes: binary, path: string>   (the trimmed clip bytes + a `<id>.mp4` path tag)

Per-shard yields are merged into `<out-dir>/fetch_manifest.json`; dead / private / geo-blocked ids are
dropped (never silently truncated), so expect a real-world video yield around 60-80%.

This corpus is research-only: fetching is subject to YouTube's Terms of Service and the per-video
rights, and the clips are not redistributable.

CLI usage:
    uv run scripts/data/download_vatex.py                                  # all caption clips, parallel fetch
    uv run scripts/data/download_vatex.py --fraction 0.01 --fanout 8       # ~1% slice
    uv run scripts/data/download_vatex.py --rank 0 --world 16              # this worker's disjoint stride
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from acquire_clips import MediaClip, MediaKind, acquire_clips

#: Official VATEX caption JSON (validation first — it is small and good for a quick smoke).
_CAPTION_JSON_URLS = (
    "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json",
    "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json",
)


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


def _clip_records(out_dir: Path, index_cap: int | None) -> list[dict[str, object]]:
    """Pull every caption JSON and build one `{...MediaClip, caption}` record per captioned clip.

    VATEX is naturally small (~41k clips), so `index_cap` is a harmless upper bound honoured for
    uniformity with the other clip datasets; `index_cap=None` reads every entry.
    """
    cache_dir = out_dir / ".captions"
    records: list[dict[str, object]] = []
    for url in _CAPTION_JSON_URLS:
        for entry in _download_json(url, cache_dir / Path(url).name):
            if index_cap is not None and len(records) >= index_cap:
                return records
            watch_id, start, end = _parse_video_id(str(entry["videoID"]))
            captions = list(entry.get("enCap", [])) + list(entry.get("chCap", []))
            record = MediaClip(
                source_id=watch_id, kind=MediaKind.VIDEO, start_seconds=start, end_seconds=end
            ).to_json()
            record["caption"] = "\n".join(str(caption) for caption in captions)
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(
        acquire_clips(
            "vatex",
            clip_records=_clip_records,
            media_kind=MediaKind.VIDEO,
            out_name="vatex",
            media_column="mp4",
            script=Path(__file__).resolve(),
            default_fanout=128,
        )
    )
