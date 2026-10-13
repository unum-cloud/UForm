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
so this builder does the same: it downloads `captions.zip`, reads `train.json` (and `val_1.json`), whose
keys are `v_<youtube_id>` with `timestamps` (`[start, end]` pairs) and `sentences`.  The videos
themselves must be fetched.  This thin builder keeps only its bespoke index read (`_clip_records`); the
shared `acquire_clips.acquire_clips` orchestrator does the rest — a Slurm `(rank, world)` split so
each worker fetches a disjoint `clips[rank::world]` stride, intra-host GNU-`parallel` fan-out, per-shard
yt-dlp fetch (an H.264 `bestvideo+bestaudio` mux of the whole video), and the uniform registry parquet:

    caption: string                                (the video's localized sentences, joined)
    mp4:     struct<bytes: binary, path: string>   (the full video bytes + a `<id>.mp4` path tag)

Per-shard yields are merged into `<out-dir>/fetch_manifest.json`; dead / private / geo-blocked ids are
dropped (never silently truncated), so expect a real-world video yield around 60-80%.

This corpus is research-only: fetching is subject to YouTube's Terms of Service and the per-video
rights, and the videos are not redistributable.

CLI usage:
    uv run scripts/data/download_activitynet_captions.py                                 # all videos, parallel
    uv run scripts/data/download_activitynet_captions.py --fraction 0.01 --fanout 8      # ~1% slice
    uv run scripts/data/download_activitynet_captions.py --rank 0 --world 16             # this worker's stride
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

from acquire_clips import MediaClip, MediaKind, acquire_clips

#: The dense-captioning metadata archive the HuggingFace loader script also pulls.
_CAPTIONS_ZIP_URL = "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"
#: Caption files inside the zip to ingest (train + the first validation split).
_CAPTION_MEMBERS = ("train.json", "val_1.json")


def _load_caption_metadata(out_dir: Path) -> dict[str, dict[str, object]]:
    """Download `captions.zip` (cached) and merge the requested caption-member JSON objects."""
    cache_zip = out_dir / ".captions" / "captions.zip"
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


def _clip_records(out_dir: Path, index_cap: int | None) -> list[dict[str, object]]:
    """Load the caption metadata and build one `{...MediaClip, caption}` record per video.

    ActivityNet Captions is naturally small (~20k videos), so `index_cap` is a harmless upper bound
    honoured for uniformity with the other clip datasets; `index_cap=None` reads every video.
    """
    metadata = _load_caption_metadata(out_dir)
    records: list[dict[str, object]] = []
    for key, info in metadata.items():
        if index_cap is not None and len(records) >= index_cap:
            break
        # Keys are `v_<youtube_id>`; the leading `v_` is the ActivityNet prefix, not part of the id.
        watch_id = key[2:] if key.startswith("v_") else key
        record = MediaClip(source_id=watch_id, kind=MediaKind.VIDEO).to_json()
        record["caption"] = "\n".join(str(sentence) for sentence in info.get("sentences", []))
        records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(
        acquire_clips(
            "activitynet_captions",
            clip_records=_clip_records,
            media_kind=MediaKind.VIDEO,
            out_name="activitynet-captions",
            media_column="mp4",
            script=Path(__file__).resolve(),
            default_fanout=128,
        )
    )
