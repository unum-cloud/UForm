# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ijson",
#     "pyarrow>=21.0.0",
#     "yt-dlp",
# ]
# ///
"""Build a bounded slice of the HowTo100M instructional video corpus by fetching clips from YouTube.

Modality: video + text.  HowTo100M is ~136M instructional video clips — each a `(video_id, [start, end]
window, ASR narration)` — mined from ~1.2M YouTube videos.  The clip BYTES live only on YouTube (no
snapshot mirror redistributes them), so a builder needs a clip-level caption index carrying the per-clip
`(video_id, start, end, text)` and then fetches each window itself.

Caption source.
This builder reads `Jazzcharles/HowTo100M_llama3_refined_caption` — a HuggingFace mirror of HowTo100M's
ASR narration re-aligned to per-clip windows and cleaned by Llama-3.  It ships ONE JSON object,
`htm_llama3_refined.json` (~9 GB), mapping a running integer index to a per-clip record:

    {"<i>": {"vid": <youtube_id>, "start_second": <float>, "end_second": <float>,
             "text": <raw ASR narration>, "refined_text": <Llama-3-cleaned instruction>, ...}, ...}

We prefer `refined_text` (falling back to `text`) as the caption.  This mirror was chosen over the
official `caption.json` (a 5 GB monolithic dict inside `HowTo100M.zip` on the INRIA/Willow host) because
it is directly HuggingFace-hosted, per-clip already, and — crucially — streamable: the records appear in
file order, so a leading slice can be read WITHOUT downloading or parsing the whole 9 GB.

Bounded index (the 100M-clip trap).
HowTo100M is far too large to materialise; `_clip_records` therefore STREAMS the JSON with `ijson` and
stops after `_INDEX_CAP` distinct `(vid, start, end)` windows (default 200k, overridable via
`UFORM_HOWTO100M_INDEX_CAP`), reading only the leading bytes of the remote file.  Records that share a
window are folded into one clip whose captions are joined, so no window is fetched twice.  The tiny built
index (a few hundred-k small dicts) is cached under `<out-dir>/.captions/` keyed by the cap so re-runs and
`--print-plan` are instant.

The rest is the shared `fetch_media_clips.acquire_clips` orchestrator: a Slurm `(rank, world)` split so each
worker fetches a disjoint `clips[rank::world]` stride, intra-host GNU-`parallel` fan-out, per-shard yt-dlp
fetch (an H.264 `bestvideo+bestaudio` mux trimmed to the clip's window), and the uniform registry parquet:

    caption: string                                (the clip's refined narration, joined per window)
    mp4:     struct<bytes: binary, path: string>   (the trimmed clip bytes + a `<id>.mp4` path tag)

Sizing.
A bare run fetches the whole capped index (~200k clips); `--fraction` / `--max-clips` cap the fetch BELOW
that ceiling (they cannot raise it — raise `UFORM_HOWTO100M_INDEX_CAP` to index more).  Per-shard yields
merge into `<out-dir>/fetch_manifest.json`; dead / private / geo-blocked ids are dropped (never silently
truncated), so expect a real-world video yield around 60-80%.

This corpus is research-only: fetching is subject to YouTube's Terms of Service and the per-video rights,
and the clips are not redistributable.

CLI usage:
    uv run python_data/download_howto100m.py                                # ~200k-clip slice, parallel fetch
    uv run python_data/download_howto100m.py --print-plan --max-clips 8     # dry-run the clip index, no fetch
    uv run python_data/download_howto100m.py --fraction 0.05 --num-shards 8 # ~5% of the capped index
    uv run python_data/download_howto100m.py --rank 0 --world 16            # this worker's disjoint stride
    UFORM_HOWTO100M_INDEX_CAP=1000000 uv run python_data/download_howto100m.py  # index a larger slice
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import ijson
from fetch_media_clips import MediaClip, MediaKind, acquire_clips

#: HuggingFace `resolve` URL of the single per-clip refined-caption JSON (streamed, never fully downloaded).
_REFINED_CAPTION_URL = (
    "https://huggingface.co/datasets/Jazzcharles/HowTo100M_llama3_refined_caption/"
    "resolve/main/htm_llama3_refined.json"
)
#: Default ceiling on distinct `(vid, start, end)` windows the streamed index materialises — HowTo100M has
#: ~136M clips, so this hard bound (overridable via `UFORM_HOWTO100M_INDEX_CAP`) keeps the list small and a
#: bare run's fetch conservative. `--fraction` / `--max-clips` cap the fetch BELOW this, never above it.
_DEFAULT_INDEX_CAP = 200_000


def _index_cap() -> int:
    """Return the distinct-window index ceiling from `UFORM_HOWTO100M_INDEX_CAP`, else `_DEFAULT_INDEX_CAP`."""
    raw = os.environ.get("UFORM_HOWTO100M_INDEX_CAP")
    return max(1, int(raw)) if raw else _DEFAULT_INDEX_CAP


def _stream_leading_records(cap: int) -> list[dict[str, object]]:
    """Stream the leading portion of the refined-caption JSON into up to `cap` windowed clip records.

    Opens the remote 9 GB JSON as a byte stream and walks its top-level `{index: clip}` entries in file
    order with `ijson`, folding records that share a `(vid, start, end)` window into a single clip whose
    captions are joined. Stops as soon as `cap` DISTINCT windows have been seen, so only the leading bytes
    of the file are ever transferred and the in-memory footprint stays bounded by `cap`.

    Args:
        cap: Maximum number of distinct `(vid, start, end)` windows to materialise.

    Returns:
        One `{...MediaClip.to_json(), "caption": str}` record per distinct window, in first-seen order.
    """
    grouped: dict[tuple[str, float, float], list[str]] = {}
    request = urllib.request.Request(_REFINED_CAPTION_URL, headers={"User-Agent": "uform-howto100m"})
    print(f"download_howto100m: streaming up to {cap} clip windows from {_REFINED_CAPTION_URL}")
    with urllib.request.urlopen(request) as stream:  # noqa: S310 — fixed trusted https HuggingFace host
        for _index, entry in ijson.kvitems(stream, ""):
            window = (str(entry["vid"]), float(entry["start_second"]), float(entry["end_second"]))
            caption = str(entry.get("refined_text") or entry.get("text") or "")
            captions = grouped.get(window)
            if captions is None:
                if len(grouped) >= cap:
                    break
                grouped[window] = [caption] if caption else []
            elif caption:
                captions.append(caption)
    records: list[dict[str, object]] = []
    for (vid, start, end), captions in grouped.items():
        record = MediaClip(source_id=vid, kind=MediaKind.VIDEO, start_seconds=start, end_seconds=end).to_json()
        record["caption"] = "\n".join(dict.fromkeys(captions))
        records.append(record)
    return records


def _clip_records(out_dir: Path) -> list[dict[str, object]]:
    """Build (and cache) the bounded HowTo100M clip index — up to `_index_cap()` `{...MediaClip, caption}`."""
    cap = _index_cap()
    cache_path = out_dir / ".captions" / f"htm_refined_index_{cap}.jsonl"
    if cache_path.exists():
        records = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
        print(f"download_howto100m: reusing cached index of {len(records)} clips at {cache_path}")
        return records
    records = _stream_leading_records(cap)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records))
    print(f"download_howto100m: indexed {len(records)} clip windows (cached at {cache_path})")
    return records


if __name__ == "__main__":
    raise SystemExit(
        acquire_clips(
            "howto100m",
            clip_records=_clip_records,
            media_kind=MediaKind.VIDEO,
            out_name="howto100m",
            media_column="mp4",
            script=Path(__file__).resolve(),
            default_num_shards=128,
        )
    )
