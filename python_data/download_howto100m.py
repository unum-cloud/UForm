# /// script
# requires-python = ">=3.12"
# dependencies = [
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
This builder reads `bpiyush/howto100m_captions_with_verb_nouns` — a HuggingFace PARQUET mirror of
HowTo100M's per-clip ASR narration.  It ships six `data/train-000NN-of-00006.parquet` shards whose rows
are the per-clip windows, one caption each:

    video_id: string   start: int64   end: int64   text: string   (+ nouns / verbs / stem_verbs, unused)

We take `text` (the clip's ASR narration) as the caption.  Parquet was chosen over the earlier
`Jazzcharles/HowTo100M_llama3_refined_caption` JSON mirror (a single ~9 GB pandas dump that emits bare
`NaN`s and raw control bytes inside strings, which no streaming JSON parser accepts): the parquet columns
are typed and null-safe, and pyarrow reads them row group by row group over HTTP range requests — so a
leading slice is read WITHOUT downloading or parsing a whole shard, at bounded memory.

Bounded index (the 100M-clip trap).
HowTo100M is far too large to materialise; `_clip_records` therefore STREAMS the parquet shards in order
with `pyarrow.parquet.ParquetFile.iter_batches` and stops after `_INDEX_CAP` distinct `(vid, start, end)`
windows (default 200k, set by `--index-cap` / `$UFORM_INDEX_CAP`), pulling only the footer and the
leading row groups of the remote files.  Records that share a window are folded into one clip whose
captions are joined, so no window is fetched twice.  The tiny built index (a few hundred-k small dicts) is
cached under `<out-dir>/.captions/` keyed by the cap so re-runs and `--print-plan` are instant.

The rest is the shared `fetch_media_clips.acquire_clips` orchestrator: a Slurm `(rank, world)` split so each
worker fetches a disjoint `clips[rank::world]` stride, intra-host GNU-`parallel` fan-out, per-shard yt-dlp
fetch (an H.264 `bestvideo+bestaudio` mux trimmed to the clip's window), and the uniform registry parquet:

    caption: string                                (the clip's ASR narration, joined per window)
    mp4:     struct<bytes: binary, path: string>   (the trimmed clip bytes + a `<id>.mp4` path tag)

Sizing.
A bare run fetches the whole capped index (~200k clips); `--fraction` / `--max-clips` cap the fetch BELOW
that ceiling (they cannot raise it — raise `--index-cap` to index more).  Per-shard yields
merge into `<out-dir>/fetch_manifest.json`; dead / private / geo-blocked ids are dropped (never silently
truncated), so expect a real-world video yield around 60-80%.

This corpus is research-only: fetching is subject to YouTube's Terms of Service and the per-video rights,
and the clips are not redistributable.

CLI usage:
    uv run python_data/download_howto100m.py                                # ~200k-clip slice, parallel fetch
    uv run python_data/download_howto100m.py --print-plan --max-clips 8     # dry-run the clip index, no fetch
    uv run python_data/download_howto100m.py --fraction 0.05 --fanout 8     # ~5% of the capped index
    uv run python_data/download_howto100m.py --rank 0 --world 16            # this worker's disjoint stride
    uv run python_data/download_howto100m.py --index-cap 1000000                # index a larger slice
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from fetch_media_clips import MediaClip, MediaKind, acquire_clips

#: HuggingFace `resolve` URLs of the per-clip caption parquet shards (read by row group, never fully fetched).
_CAPTION_PARQUET_URLS: tuple[str, ...] = tuple(
    "https://huggingface.co/datasets/bpiyush/howto100m_captions_with_verb_nouns/"
    f"resolve/main/data/train-{shard:05d}-of-00006.parquet"
    for shard in range(6)
)
#: Row-group batch size for the streaming scan (rows held in memory at once while grouping windows).
_ROW_BATCH: int = 20_000
#: Default ceiling on distinct `(vid, start, end)` windows the streamed index materialises — HowTo100M has
#: ~136M clips, so this hard bound (the `--index-cap` default; raise it via `--index-cap` / `$UFORM_INDEX_CAP`)
#: keeps the list small and a bare run's fetch conservative. `--fraction` / `--max-clips` cap the fetch BELOW
#: this, never above it.
_DEFAULT_INDEX_CAP = 200_000


class _HttpRangeReader:
    """A minimal seekable, HTTP-range-backed byte reader so `pyarrow.parquet` can read a remote parquet
    file's footer and only the leading row groups it consumes — bounded transfer, never the whole shard.

    Parquet's metadata lives in the file footer, so a reader must seek: pure forward streaming cannot work.
    This wrapper serves pyarrow's many small, mostly-sequential reads from a `_BLOCK`-sized read-ahead cache
    (one HTTP `Range` GET refills it), coalescing what would otherwise be hundreds of round trips into a
    handful. It exposes just the surface pyarrow touches: `read` / `seek` / `tell` / `seekable` / `closed`.
    """

    _BLOCK: int = 1 << 22  # 4 MiB read-ahead; a footer read larger than this fetches exactly what it asks

    def __init__(self, url: str) -> None:
        self._url = url
        self._position = 0
        self._cache = b""
        self._cache_start = 0
        self.closed = False
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "uform-howto100m"})
        with urllib.request.urlopen(request) as response:  # noqa: S310 — fixed trusted https HuggingFace host
            self._length = int(response.headers["Content-Length"])

    def seek(self, offset: int, whence: int = 0) -> int:
        """Reposition the read cursor (`whence`: 0 absolute, 1 relative, 2 from end) and return it."""
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        else:
            self._position = self._length + offset
        return self._position

    def tell(self) -> int:
        """Return the current absolute read position."""
        return self._position

    def seekable(self) -> bool:
        """Report random access, so pyarrow reads the footer then only the row groups it consumes."""
        return True

    def readable(self) -> bool:
        """Report the stream is readable."""
        return True

    def _fetch(self, start: int, end: int) -> bytes:
        """Fetch the half-open byte range `[start, end)` with a single HTTP `Range` GET."""
        headers = {"User-Agent": "uform-howto100m", "Range": f"bytes={start}-{end - 1}"}
        request = urllib.request.Request(self._url, headers=headers)
        with urllib.request.urlopen(request) as response:  # noqa: S310 — fixed trusted https HuggingFace host
            return response.read()

    def read(self, size: int = -1) -> bytes:
        """Return up to `size` bytes from the cursor (all remaining when `size` < 0), refilling the cache."""
        end = self._length if size is None or size < 0 else min(self._length, self._position + size)
        if end <= self._position:
            return b""
        if not (self._cache_start <= self._position and end <= self._cache_start + len(self._cache)):
            fetch_end = min(self._length, self._position + max(end - self._position, self._BLOCK))
            self._cache = self._fetch(self._position, fetch_end)
            self._cache_start = self._position
        offset = self._position - self._cache_start
        chunk = self._cache[offset : offset + (end - self._position)]
        self._position = end
        return chunk

    def close(self) -> None:
        """Mark the reader closed (there is no persistent socket to release)."""
        self.closed = True


def _stream_leading_records(cap: int) -> list[dict[str, object]]:
    """Stream the leading row groups of the caption parquet shards into up to `cap` windowed clip records.

    Opens each remote parquet shard in order through `_HttpRangeReader` and walks its rows with
    `ParquetFile.iter_batches` (only the `video_id` / `start` / `end` / `text` columns), folding rows that
    share a `(vid, start, end)` window into a single clip whose captions are joined. Stops as soon as `cap`
    DISTINCT windows have been seen, so only the footer and leading row groups of the shards are ever
    transferred and the in-memory footprint stays bounded by `cap`.

    Args:
        cap: Maximum number of distinct `(vid, start, end)` windows to materialise.

    Returns:
        One `{...MediaClip.to_json(), "caption": str}` record per distinct window, in first-seen order.
    """
    grouped: dict[tuple[str, float, float], list[str]] = {}
    for url in _CAPTION_PARQUET_URLS:
        if len(grouped) >= cap:
            break
        print(f"download_howto100m: streaming up to {cap} clip windows (have {len(grouped)}) from {url}")
        parquet = pq.ParquetFile(_HttpRangeReader(url))
        for batch in parquet.iter_batches(batch_size=_ROW_BATCH, columns=["video_id", "start", "end", "text"]):
            columns = batch.to_pydict()
            for vid, start, end, text in zip(
                columns["video_id"], columns["start"], columns["end"], columns["text"], strict=True
            ):
                window = (str(vid), float(start), float(end))
                caption = str(text or "")
                captions = grouped.get(window)
                if captions is None:
                    if len(grouped) >= cap:
                        break
                    grouped[window] = [caption] if caption else []
                elif caption:
                    captions.append(caption)
            if len(grouped) >= cap:
                break
    records: list[dict[str, object]] = []
    for (vid, start, end), captions in grouped.items():
        record = MediaClip(source_id=vid, kind=MediaKind.VIDEO, start_seconds=start, end_seconds=end).to_json()
        record["caption"] = "\n".join(dict.fromkeys(captions))
        records.append(record)
    return records


def _clip_records(out_dir: Path, index_cap: int | None) -> list[dict[str, object]]:
    """Build (and cache) the bounded HowTo100M clip index — up to `index_cap` `{...MediaClip, caption}`."""
    cap = index_cap if index_cap is not None else _DEFAULT_INDEX_CAP
    cache_path = out_dir / ".captions" / f"howto100m_index_{cap}.jsonl"
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
            default_index_cap=_DEFAULT_INDEX_CAP,
            default_fanout=128,
        )
    )
