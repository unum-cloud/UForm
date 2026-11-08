# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "yt-dlp",
# ]
# ///
r"""Host-side YouTube media-clip fetcher shared by the fetch-dependent dataset builders.

Modality: audio + video. Several research corpora (Auto-ACD, VATEX, ActivityNet Captions) ship only
captions plus a YouTube id and an optional `[start, end]` window — the media itself has to be
fetched at repack time. This module is the single host-side helper those builders call: given an
iterable of `MediaClip` records it downloads each clip with `yt-dlp` (audio: `bestaudio` extracted
to mp3; video: an H.264 `bestvideo+bestaudio` mux trimmed to the requested window, written as mp4),
retries transient failures with exponential backoff, skips permanently dead / private / geo-blocked
ids, and accounts for every attempted id in a `FetchManifest` so a shard's yield is auditable and is
never silently truncated.

Concurrency: this module never spawns an in-process `Pool` / `ThreadPool`; it fetches one clip at a
time. Shard-level parallelism is the caller's job — each dataset builder runs one process per shard
under GNU `parallel`, and every such process drives this helper serially.

`yt-dlp` is declared in the PEP 723 header above so `uv run scripts/data/acquire_clips.py ...`
provisions it (and the binary is invoked as `python -m yt_dlp`, so the module is never imported into
the caller's interpreter).  `ffmpeg` must be on `PATH` for the mp3 extraction and the keyframe-accurate
mp4 trim.

CLI usage:
    # Smoke-test a handful of public ids (some failures are expected and reported, not fatal):
    uv run scripts/data/acquire_clips.py --kind video --ids dQw4w9WgXcQ aqz-KE-bpKQ \
        --out-dir /tmp/clips --manifest /tmp/clips/manifest.json
    # Fetch a 10-second audio window from one id:
    uv run scripts/data/acquire_clips.py --kind audio --ids YS6CoSybkYU --start 30 --end 40 \
        --out-dir /tmp/audio
    # Shard mode used by the dataset builders under GNU parallel (one JSON clip spec per line):
    uv run scripts/data/acquire_clips.py --clips-jsonl shard_007.jsonl --out-dir staging/shard_007
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from sharding import already_done, partition, resolve_rank_world

#: Default retry budget for a single clip before it is dropped as failed.
DEFAULT_RETRIES: int = 3
#: Default base backoff (seconds); attempt `n` sleeps `backoff * 2**n` before retrying.
DEFAULT_BACKOFF_SECONDS: float = 2.0
#: Default wall-clock ceiling for one `yt-dlp` invocation before it is killed and retried.
DEFAULT_TIMEOUT_SECONDS: float = 240.0

#: Substrings (matched case-insensitively in yt-dlp output) that mark a PERMANENT failure: the
#: clip is gone / private / geo-blocked / rights-stripped, so retrying cannot help and the id
#: is dropped at once.
PERMANENT_ERROR_MARKERS: tuple[str, ...] = (
    "video unavailable",
    "this video is unavailable",
    "private video",
    "this video is private",
    "video is no longer available",
    "no longer available",
    "has been removed",
    "removed by the uploader",
    "account associated with this video has been terminated",
    "account has been terminated",
    "not available in your country",
    "who has blocked it in your country",
    "blocked it in your country on copyright",
    "copyright grounds",
    "members-only",
    "join this channel",
    "is not available",
    "requested format is not available",
)

#: Cap the fetched video rendition to this height (pixels).
#: The encoder trains at a small patch grid, so there is no
#: benefit to pulling 1080p+; 480p slashes bandwidth, disk,
#: and per-clip memory.
_MAX_VIDEO_HEIGHT: int = 480

#: Env var naming a Netscape `cookies.txt` (or a DIRECTORY of them) that authenticates
#: yt-dlp against YouTube's bot-wall. An off-repo secret; when unset the media scrapers
#: will almost certainly bot-wall.
_COOKIES_ENV: str = "UFORM_YTDLP_COOKIES"


@functools.cache
def _cookies_args() -> tuple[str, ...]:
    """`("--cookies", <file>)` for yt-dlp from `$UFORM_YTDLP_COOKIES`, or `()` when unset (computed once).

    The value is a Netscape `cookies.txt` path, or a DIRECTORY of them — a worker then picks one by its
    Slurm rank (`cookies[rank % n]`) to spread the fetch across accounts and soften per-account limits.
    A set-but-missing path (or empty directory) is an explicit error: a typo must not silently bot-wall.
    """
    raw = os.environ.get(_COOKIES_ENV)
    if not raw:
        return ()
    path = Path(raw)
    if path.is_dir():
        files = sorted(entry for entry in path.iterdir() if entry.is_file())
        if not files:
            raise FileNotFoundError(f"{_COOKIES_ENV}={raw} is an empty directory; expected cookies.txt file(s).")
        rank, _world = resolve_rank_world()
        path = files[rank % len(files)]
    if not path.is_file():
        raise FileNotFoundError(f"{_COOKIES_ENV}={raw} does not exist; export a Netscape cookies.txt path.")
    return ("--cookies", str(path))


def _hit_bot_wall(output: str) -> bool:
    """Whether yt-dlp output shows the YouTube bot / sign-in wall (a missing/expired cookie, not a dead id)."""
    lowered = output.lower()
    return "not a bot" in lowered or "sign in to confirm" in lowered or "use --cookies" in lowered


def _warn_if_no_cookies() -> None:
    """Warn once (to stderr) when no cookies are configured — the YouTube fetch will likely bot-wall."""
    if not os.environ.get(_COOKIES_ENV):
        print(
            f"acquire_clips: WARNING — {_COOKIES_ENV} is unset; YouTube fetch will likely hit the bot-wall. "
            "Export a Netscape cookies.txt path (or a directory of them) to authenticate.",
            file=sys.stderr,
        )


class MediaKind(StrEnum):
    """Closed set of clip kinds this fetcher knows how to materialise."""

    AUDIO = "audio"
    VIDEO = "video"

    @property
    def file_extension(self) -> str:
        """Return the container extension the fetched payload is written with."""
        return "mp3" if self is MediaKind.AUDIO else "mp4"


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaClip:
    """One unit of work: a YouTube id, its kind, and an optional `[start, end]` second window.

    Attributes:
        source_id: YouTube video id (the `watch?v=` token), e.g. `"dQw4w9WgXcQ"`.
        kind: Whether to fetch the audio track (mp3) or the video clip (mp4).
        start_seconds: Window start in seconds, or `None` to fetch from the beginning.
        end_seconds: Window end in seconds, or `None` to fetch to the end.
    """

    source_id: str
    kind: MediaKind
    start_seconds: float | None = None
    end_seconds: float | None = None

    @property
    def has_window(self) -> bool:
        """Return whether a finite `[start, end]` trim window is set."""
        return self.start_seconds is not None and self.end_seconds is not None

    def to_json(self) -> dict[str, object]:
        """Serialise to the one-line JSON record consumed by `--clips-jsonl` shard mode."""
        return {
            "source_id": self.source_id,
            "kind": str(self.kind),
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }

    @classmethod
    def from_json(cls, record: dict[str, object]) -> MediaClip:
        """Rebuild a `MediaClip` from a `--clips-jsonl` record (inverse of `to_json`)."""
        start = record.get("start_seconds")
        end = record.get("end_seconds")
        return cls(
            source_id=str(record["source_id"]),
            kind=MediaKind(str(record["kind"])),
            start_seconds=None if start is None else float(start),  # type: ignore[arg-type]
            end_seconds=None if end is None else float(end),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClipFetch:
    """Outcome of fetching one clip.

    Attributes:
        clip: The clip that was attempted.
        payload: The fetched bytes, or `None` when the clip was dropped.
        extension: The container extension of `payload` (`"mp3"` / `"mp4"`); set even when dropped.
        dropped_reason: Empty when the fetch succeeded, otherwise a short human-readable cause.
    """

    clip: MediaClip
    payload: bytes | None
    extension: str
    dropped_reason: str

    @property
    def succeeded(self) -> bool:
        """Return whether bytes were obtained for this clip."""
        return self.payload is not None


@dataclass(slots=True)
class FetchManifest:
    """Running tally of a shard's fetch so its yield is auditable (no silent truncation).

    Every attempted id lands in exactly one of `succeeded` / `dropped`; `record` enforces that, and
    `write` serialises the counts plus the per-id drop reasons.
    """

    attempted: list[str] = field(default_factory=list)
    succeeded: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)

    def record(self, result: ClipFetch) -> None:
        """Fold one `ClipFetch` into the tally."""
        self.attempted.append(result.clip.source_id)
        if result.succeeded:
            self.succeeded.append(result.clip.source_id)
        else:
            self.dropped.append({"source_id": result.clip.source_id, "reason": result.dropped_reason})

    @property
    def yield_fraction(self) -> float:
        """Return the succeeded / attempted ratio (0.0 when nothing was attempted)."""
        return len(self.succeeded) / len(self.attempted) if self.attempted else 0.0

    def to_json(self) -> dict[str, object]:
        """Serialise the manifest to a JSON-ready dict."""
        return {
            "attempted_count": len(self.attempted),
            "succeeded_count": len(self.succeeded),
            "dropped_count": len(self.dropped),
            "yield_fraction": round(self.yield_fraction, 4),
            "dropped": self.dropped,
        }

    def write(self, path: Path) -> None:
        """Write the manifest as pretty JSON, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False))


def _youtube_watch_url(source_id: str) -> str:
    """Return the canonical watch URL for a YouTube id."""
    return f"https://www.youtube.com/watch?v={source_id}"


def _safe_stem(source_id: str) -> str:
    """Return a filesystem-safe output stem derived from a YouTube id."""
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in source_id)


def _build_ytdlp_command(clip: MediaClip, output_template: str) -> list[str]:
    """Build the `python -m yt_dlp` argv for one clip (audio mp3 or H.264 mp4, optionally trimmed).

    Args:
        clip: The clip to fetch.
        output_template: yt-dlp `-o` template, e.g. `"<dir>/<stem>.%(ext)s"`.

    Returns:
        The full argv list (invoked with `subprocess.run`, never through a shell).
    """
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "--no-warnings",
        "--quiet",
        "--ignore-config",
        *_cookies_args(),
        "--sleep-requests",
        "1",
        "--socket-timeout",
        "30",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        "-o",
        output_template,
    ]
    if clip.has_window:
        # `*start-end` selects a single time range; force-keyframes makes the cut frame-accurate.
        command += ["--download-sections", f"*{clip.start_seconds}-{clip.end_seconds}", "--force-keyframes-at-cuts"]
    if clip.kind is MediaKind.AUDIO:
        command += ["--format", "bestaudio/best", "--extract-audio", "--audio-format", "mp3", "--audio-quality", "5"]
    else:
        # Prefer an H.264 (avc1) video stream muxed with the best audio into an mp4 container, CAPPED at
        # `_MAX_VIDEO_HEIGHT`: the encoder trains at a small patch grid, so a 480p fetch saves bandwidth,
        # disk, and memory versus pulling 1080p+. The final uncapped `/best` is a last resort so a clip
        # whose only renditions exceed the cap still downloads rather than being dropped.
        cap = f"[height<={_MAX_VIDEO_HEIGHT}]"
        video_format = (
            f"bestvideo[vcodec^=avc1]{cap}+bestaudio[ext=m4a]/best[vcodec^=avc1]{cap}/best{cap}[ext=mp4]/best{cap}/best"
        )
        command += ["--format", video_format, "--merge-output-format", "mp4"]
    command.append(_youtube_watch_url(clip.source_id))
    return command


def _classify_failure(output: str) -> str:
    """Return a permanent drop reason if the output matches a known dead-clip marker, else `""`."""
    lowered = output.lower()
    for marker in PERMANENT_ERROR_MARKERS:
        if marker in lowered:
            return f"permanent: {marker}"
    return ""


def _read_payload(work_dir: Path, stem: str, extension: str) -> bytes | None:
    """Return the bytes of the produced file (preferring the expected extension), or `None`."""
    candidates = sorted(work_dir.glob(f"{stem}.*"))
    if not candidates:
        return None
    preferred = [path for path in candidates if path.suffix == f".{extension}"]
    chosen = preferred[0] if preferred else candidates[0]
    return chosen.read_bytes()


def fetch_one_clip(
    clip: MediaClip,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ClipFetch:
    """Fetch a single clip, retrying transient failures with exponential backoff.

    A permanent failure (dead / private / geo-blocked id) is dropped immediately without retrying.
    Any other failure — network hiccup, timeout, transient HTTP error — is retried up to `retries`
    times with a `backoff_seconds * 2**attempt` sleep between attempts, then dropped.

    Args:
        clip: The clip to fetch.
        retries: Maximum transient retries before giving up.
        backoff_seconds: Base backoff; attempt `n` sleeps `backoff_seconds * 2**n`.
        timeout_seconds: Wall-clock ceiling for each `yt-dlp` invocation.

    Returns:
        A `ClipFetch` whose `payload` is the bytes on success or `None` (with `dropped_reason`) on drop.
    """
    extension = clip.kind.file_extension
    stem = _safe_stem(clip.source_id)
    last_reason = "exhausted retries"
    for attempt in range(retries + 1):
        with tempfile.TemporaryDirectory(prefix="fetch_clip_") as work_name:
            work_dir = Path(work_name)
            command = _build_ytdlp_command(clip, str(work_dir / f"{stem}.%(ext)s"))
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=timeout_seconds, check=False
                )
            except subprocess.TimeoutExpired:
                last_reason = f"transient: timeout after {timeout_seconds:.0f}s"
                completed = None
            if completed is not None:
                if completed.returncode == 0:
                    payload = _read_payload(work_dir, stem, extension)
                    if payload:
                        return ClipFetch(clip=clip, payload=payload, extension=extension, dropped_reason="")
                    last_reason = "transient: yt-dlp succeeded but produced no output file"
                else:
                    output = completed.stdout + "\n" + completed.stderr
                    if _hit_bot_wall(output):
                        raise RuntimeError(
                            f"yt-dlp hit YouTube's bot / sign-in wall — set ${_COOKIES_ENV} to a valid Netscape "
                            "cookies.txt (a stale/expired cookie looks the same). Aborting rather than silently "
                            "dropping every clip."
                        )
                    permanent = _classify_failure(output)
                    if permanent:
                        return ClipFetch(clip=clip, payload=None, extension=extension, dropped_reason=permanent)
                    tail = (completed.stderr.strip() or completed.stdout.strip() or "unknown error")[-200:]
                    last_reason = f"transient: {tail}"
        if attempt < retries:
            time.sleep(backoff_seconds * (2**attempt))
    return ClipFetch(clip=clip, payload=None, extension=extension, dropped_reason=last_reason)


def fetch_clips(
    clips: Iterable[MediaClip],
    *,
    manifest: FetchManifest | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[ClipFetch]:
    """Fetch clips one at a time, yielding each `ClipFetch` and recording it in `manifest`.

    This is a generator so a caller can stream rows into a parquet writer without buffering an entire
    shard of media bytes in memory.  It never parallelises internally — the caller
    parallelises across shards with GNU `parallel`.

    Args:
        clips: The clips to fetch, in order.
        manifest: A `FetchManifest` to fold each outcome into; one is created internally if omitted.
        retries: Per-clip transient retry budget.
        backoff_seconds: Base backoff between retries.
        timeout_seconds: Per-invocation wall-clock ceiling.

    Yields:
        A `ClipFetch` for every clip, succeeded or dropped (so nothing is silently truncated).
    """
    tally = manifest if manifest is not None else FetchManifest()
    for clip in clips:
        result = fetch_one_clip(clip, retries=retries, backoff_seconds=backoff_seconds, timeout_seconds=timeout_seconds)
        tally.record(result)
        yield result


@dataclass(frozen=True, slots=True, kw_only=True)
class ShardPlan:
    """Filesystem plan for one shard: its clip records, its output parquet, and its yield manifest.

    Attributes:
        input_jsonl: Shard file of one JSON clip record per line (the dataset builder reads it back).
        output_parquet: Where the shard's worker process writes its registry-schema parquet.
        manifest_json: Where the shard's worker writes its per-shard `FetchManifest`.
    """

    input_jsonl: Path
    output_parquet: Path
    manifest_json: Path


def plan_shards(
    records: Iterable[dict[str, object]],
    *,
    num_shards: int,
    staging_dir: Path,
    output_dir: Path,
    output_prefix: str,
) -> list[ShardPlan]:
    """Round-robin clip records into `num_shards` JSONL shard files and return their `ShardPlan`s.

    Each record is a JSON-serialisable dict carrying at least the `MediaClip` fields (see
    `MediaClip.to_json`) plus whatever caption payload the dataset builder needs to assemble a row.
    Round-robin (rather than contiguous) assignment keeps every shard a representative sample, so a
    partial run still spans the corpus.

    Args:
        records: The per-clip work records to distribute.
        num_shards: Requested shard count; clamped to at most the number of records.
        staging_dir: Directory the shard JSONL + manifest files live under.
        output_dir: Directory the per-shard output parquet files live under.
        output_prefix: Filename stem shared by the shard inputs and outputs.

    Returns:
        One `ShardPlan` per non-empty shard.
    """
    materialised = list(records)
    if not materialised:
        return []
    shard_count = max(1, min(num_shards, len(materialised)))
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[list[dict[str, object]]] = [[] for _ in range(shard_count)]
    for position, record in enumerate(materialised):
        buckets[position % shard_count].append(record)
    plans: list[ShardPlan] = []
    for shard_number, bucket in enumerate(buckets):
        if not bucket:
            continue
        input_jsonl = staging_dir / f"{output_prefix}-shard-{shard_number:05d}.jsonl"
        input_jsonl.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in bucket))
        plans.append(
            ShardPlan(
                input_jsonl=input_jsonl,
                output_parquet=output_dir / f"{output_prefix}-{shard_number:05d}.parquet",
                manifest_json=staging_dir / f"{output_prefix}-shard-{shard_number:05d}.manifest.json",
            )
        )
    return plans


def run_shard_workers(builder_script: Path, plans: list[ShardPlan], *, jobs: int) -> None:
    """Fetch + repack every shard by self-invoking `builder_script --repack-shard`, one process per shard.

    Uses GNU `parallel` (one process per shard) when it is on `PATH` and `jobs != 1`;
    otherwise falls back to a sequential subprocess loop so the builder still runs without `parallel`.
    A failing shard never aborts the run (`--halt never`); its drops surface in the merged manifest.

    Args:
        builder_script: Absolute path of the dataset builder to re-invoke in shard mode.
        plans: The shard plans produced by `plan_shards`.
        jobs: Degree of parallelism (number of concurrent shard processes).
    """
    if not plans:
        return
    control_path = plans[0].input_jsonl.parent / f"{builder_script.stem}_shard_control.tsv"
    control_path.write_text(
        "".join(f"{plan.input_jsonl}\t{plan.output_parquet}\t{plan.manifest_json}\n" for plan in plans)
    )
    worker = ["uv", "run", str(builder_script), "--repack-shard"]
    if jobs != 1 and shutil.which("parallel") is not None:
        subprocess.run(
            [
                "parallel",
                "--colsep",
                "\t",
                "-j",
                str(jobs),
                "--halt",
                "never",
                *worker,
                "{1}",
                "--shard-output",
                "{2}",
                "--shard-manifest",
                "{3}",
                "::::",
                str(control_path),
            ],
            check=False,
        )
        return
    for plan in plans:
        subprocess.run(
            [
                *worker,
                str(plan.input_jsonl),
                "--shard-output",
                str(plan.output_parquet),
                "--shard-manifest",
                str(plan.manifest_json),
            ],
            check=False,
        )


def summarise_shard_manifests(plans: list[ShardPlan]) -> dict[str, object]:
    """Aggregate the per-shard manifests into one yield summary (counts + concatenated drop reasons)."""
    attempted = 0
    succeeded = 0
    dropped: list[dict[str, str]] = []
    for plan in plans:
        if plan.manifest_json.exists():
            data = json.loads(plan.manifest_json.read_text())
            attempted += int(data.get("attempted_count", 0))
            succeeded += int(data.get("succeeded_count", 0))
            dropped += data.get("dropped", [])
    return {
        "attempted_count": attempted,
        "succeeded_count": succeeded,
        "dropped_count": len(dropped),
        "yield_fraction": round(succeeded / attempted, 4) if attempted else 0.0,
        "dropped": dropped,
    }


def fetch_shard(
    input_jsonl: Path,
    *,
    manifest: FetchManifest,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[tuple[dict[str, object], ClipFetch]]:
    """Yield `(record, ClipFetch)` for each clip record in a shard JSONL, folding outcomes into `manifest`.

    The dataset builder consumes this in its `--repack-shard` worker: it reads the caption payload
    from `record` and the media bytes from `result.payload`, skipping (but never silently dropping —
    `manifest` records them) the clips that failed to fetch.

    Args:
        input_jsonl: Shard file written by `plan_shards`.
        manifest: The shard's `FetchManifest`, updated per clip.
        retries: Per-clip transient retry budget.
        backoff_seconds: Base backoff between retries.
        timeout_seconds: Per-invocation wall-clock ceiling.

    Yields:
        Each clip's original record paired with its `ClipFetch` outcome.
    """
    for line in input_jsonl.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        result = fetch_one_clip(
            MediaClip.from_json(record),
            retries=retries,
            backoff_seconds=backoff_seconds,
            timeout_seconds=timeout_seconds,
        )
        manifest.record(result)
        yield record, result


#: One clip's work record consumed by the shared orchestrator: the `MediaClip.to_json()` fields plus
#: a single `caption` string. The dataset's `clip_records(out_dir, index_cap)` builds these —
#: enumerating at most `index_cap` records (None = the source's natural size) so a streaming reader
#: can stop early — and the shared repack writes them as a uniform `{caption: string,
#: <media_column>: struct<bytes, path>}` parquet.
ClipRecords = Callable[[Path, int | None], list[dict[str, object]]]


def _clips_data_root() -> Path:
    """The acquisition output root: `$UFORM_DATA_ROOT`, else the repo's `data/` directory (mirrors acquire.py)."""
    return Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))


def _record_start_key(record: dict[str, object]) -> float:
    """Sort key component: the record's start second, or `-1.0` for a whole-clip record with no window."""
    start = record.get("start_seconds")
    return float(start) if start is not None else -1.0  # type: ignore[arg-type]


def _cap_records(
    records: list[dict[str, object]], fraction: float | None, max_clips: int | None
) -> list[dict[str, object]]:
    """Return the leading slice of `records` under the tighter of `--fraction` and `--max-clips` (or all)."""
    keep = len(records)
    if fraction is not None and records:
        keep = min(keep, max(1, math.ceil(len(records) * fraction)))
    if max_clips is not None:
        keep = min(keep, max_clips)
    return records[:keep]


def resolve_index_cap(cli_index_cap: int | None, default_index_cap: int | None) -> int | None:
    """Resolve the index materialisation cap: `--index-cap` > `$UFORM_INDEX_CAP` > the dataset default.

    Distinct from the `--fraction` / `--max-clips` FETCH cap: this bounds how many records a dataset's
    `clip_records` callback enumerates at all, so a streaming reader can stop early instead of building the
    whole index and trimming it afterwards. `None` means the source's natural size — only safe for bounded
    corpora. The fetch cap layers below it: `--fraction` / `--max-clips` lower the fetch, never raise it
    above the materialised index.
    """
    if cli_index_cap is not None:
        return max(1, cli_index_cap)
    raw = os.environ.get("UFORM_INDEX_CAP")
    if raw:
        return max(1, int(raw))
    return default_index_cap


def _repack_clip_shard(
    input_jsonl: Path,
    output_parquet: Path,
    manifest_path: Path,
    *,
    media_column: str,
    row_group_size: int,
    retries: int,
    timeout_seconds: float,
) -> int:
    """Shared `--repack-shard` worker: fetch one shard's clips and write the uniform registry parquet.

    Writes `{caption: string, <media_column>: struct<bytes, path>}` (mp3 for AUDIO, mp4 for VIDEO) through
    a `.partial` sibling + atomic replace — the final parquet appearing IS the shard's commit record — plus
    the per-shard `FetchManifest`. Dropped ids are recorded, never silently truncated.
    """
    import pyarrow as pa  # noqa: PLC0415 — lazy: the smoke CLI needs only yt-dlp; builders declare pyarrow
    import pyarrow.parquet as pq  # noqa: PLC0415 — lazy: paired with pyarrow above

    media_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    manifest = FetchManifest()
    captions: list[str] = []
    media: list[dict[str, object]] = []
    for record, result in fetch_shard(input_jsonl, manifest=manifest, retries=retries, timeout_seconds=timeout_seconds):
        if not result.succeeded or result.payload is None:
            continue
        captions.append(str(record.get("caption", "")))
        media.append({"bytes": result.payload, "path": f"{result.clip.source_id}.{result.extension}"})
    manifest.write(manifest_path)
    if captions:
        table = pa.table(
            {"caption": pa.array(captions, type=pa.string()), media_column: pa.array(media, type=media_struct)}
        )
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        partial = output_parquet.with_suffix(".parquet.partial")
        pq.write_table(table, partial, compression="zstd", row_group_size=row_group_size)
        partial.replace(output_parquet)
    print(
        f"acquire_clips: shard {input_jsonl.name} -> {len(captions)} clips "
        f"({len(manifest.succeeded)}/{len(manifest.attempted)} yield)"
    )
    return 0


def _print_clip_plan(name: str, rank: int, world: int, total: int, capped: int, mine: list[dict[str, object]]) -> None:
    """Dry-run hook: report this rank's disjoint clip stride (count + `(id, start)` keys) without fetching."""
    keys = [(str(record["source_id"]), _record_start_key(record)) for record in mine]
    print(f"acquire_clips[{name}]: rank {rank}/{world} — {total} indexed, {capped} after cap, {len(mine)} this rank")
    print(f"acquire_clips[{name}]: selected keys = {keys}")


def acquire_clips(
    name: str,
    *,
    clip_records: ClipRecords,
    media_kind: MediaKind,
    out_name: str,
    media_column: str,
    script: Path,
    default_fraction: float = 1.0,
    default_index_cap: int | None = None,
    default_fanout: int = 128,
    default_jobs: int = 8,
    row_group_size: int = 256,
) -> int:
    """Shared Slurm-sharded YouTube-clip acquirer the thin per-dataset builders delegate to.

    One dataset's whole job: parse the uniform CLI, resolve `(rank, world)` from Slurm (`sharding`), build
    the full clip list via the dataset's `clip_records`, sort it into a deterministic order, cap it by
    `--fraction` / `--max-clips`, take THIS worker's disjoint stride `clips[rank::world]`, sub-shard that
    stride into `--fanout` JSONLs, and fan the yt-dlp fetch out intra-host with GNU `parallel` by
    self-invoking `script --repack-shard` (the shared `_repack_clip_shard` worker). Output identity ties to
    `(rank, world)` — a shard named `<out_name>-r{rank}w{world}-{i:05d}.parquet` — and its atomic rename is
    the commit: a re-run of the same world skips finished shards (`already_done`) unless `--force`. There is
    no per-dataset `.built` sentinel (wrong under multi-rank).

    @param[in] name The catalog / mixture dataset name (used only in log lines).
    @param[in] clip_records `out_dir -> list[{source_id, kind, start_seconds, end_seconds, caption}]` index reader.
    @param[in] media_kind Whether the clips are audio (mp3) or video (mp4); the records also carry it per clip.
    @param[in] out_name The `data/<out_name>` output subdirectory and shard-name prefix.
    @param[in] media_column The struct column the payload lands in (e.g. `audio` / `mp4`).
    @param[in] script The calling builder's `__file__` (self-invoked in `--repack-shard` mode by GNU parallel).
    @param[in] default_fraction The `--fraction` default when the caller omits it (some corpora are huge).
    @param[in] default_index_cap The `--index-cap` default: how many windows the index materialises when
        neither `--index-cap` nor `$UFORM_INDEX_CAP` is set (None = the source's natural size, bounded corpora only).
    @param[in] default_fanout The `--fanout` default (intra-host sub-shard fan-out width per worker).
    @param[in] default_jobs The `--jobs` default (concurrent shard processes under GNU parallel).
    @param[in] row_group_size The parquet row-group size for the written shards.
    @return Process exit code (`0` on success).
    """
    parser = argparse.ArgumentParser(description=f"Fetch + repack {name} YouTube clips into rank-sharded parquet.")
    parser.add_argument("--fraction", type=float, default=default_fraction, help="leading-slice clip cap in (0, 1].")
    parser.add_argument("--max-clips", type=int, default=None, help="hard cap on the number of clips (leading slice).")
    parser.add_argument(
        "--index-cap",
        type=int,
        default=None,
        help="materialisation bound: how many clip windows the index enumerates (before the fetch cap).",
    )
    parser.add_argument("--rank", type=int, default=None, help="usually Slurm-derived; explicit for local N-way runs.")
    parser.add_argument("--world", type=int, default=None, help="usually Slurm-derived; explicit for local N-way runs.")
    parser.add_argument("--jobs", type=int, default=default_jobs, help="concurrent shard processes under GNU parallel.")
    parser.add_argument(
        "--fanout",
        type=int,
        default=default_fanout,
        help="JSONL sub-shards fanned out per worker (fetch parallelism, not a volume cap).",
    )
    parser.add_argument("--force", action="store_true", help="re-fetch shards whose parquet already exists.")
    parser.add_argument("--print-plan", action="store_true", help="print this rank's clip stride and exit (no fetch).")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="per-clip transient retry budget.")
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="wall-clock ceiling per yt-dlp call."
    )
    parser.add_argument("--repack-shard", type=Path, default=None, help="internal: fetch+repack a single shard JSONL.")
    parser.add_argument("--shard-output", type=Path, default=None, help="internal: output parquet for --repack-shard.")
    parser.add_argument("--shard-manifest", type=Path, default=None, help="internal: manifest path for --repack-shard.")
    args = parser.parse_args()

    if args.repack_shard is not None:
        if args.shard_output is None or args.shard_manifest is None:
            print(f"acquire_clips[{name}]: --repack-shard needs --shard-output and --shard-manifest", file=sys.stderr)
            return 2
        return _repack_clip_shard(
            args.repack_shard,
            args.shard_output,
            args.shard_manifest,
            media_column=media_column,
            row_group_size=row_group_size,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )

    rank, world = resolve_rank_world(args.rank, args.world)
    out_dir = _clips_data_root() / out_name
    index_cap = resolve_index_cap(args.index_cap, default_index_cap)
    records = clip_records(out_dir, index_cap)
    records.sort(key=lambda record: (str(record["source_id"]), _record_start_key(record)))
    capped = _cap_records(records, args.fraction, args.max_clips)
    mine = partition(capped, rank, world)
    if args.print_plan:
        _print_clip_plan(name, rank, world, len(records), len(capped), mine)
        return 0

    _warn_if_no_cookies()
    output_dir = out_dir / "data"
    staging_dir = out_dir / ".staging" / f"r{rank}w{world}"
    if args.force and staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    plans = plan_shards(
        mine,
        num_shards=args.fanout,
        staging_dir=staging_dir,
        output_dir=output_dir,
        output_prefix=f"{out_name}-r{rank}w{world}",
    )
    pending = [plan for plan in plans if args.force or not already_done(plan.output_parquet)]
    run_shard_workers(script, pending, jobs=args.jobs)
    summary = summarise_shard_manifests(plans)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fetch_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"acquire_clips[{name}]: rank {rank}/{world} built {summary['succeeded_count']} {media_kind} clips of "
        f"{summary['attempted_count']} attempted ({float(summary['yield_fraction']) * 100:.1f}% yield) under {output_dir}"
    )
    return 0


def _read_clips_jsonl(path: Path) -> list[MediaClip]:
    """Load a `--clips-jsonl` shard file into a list of `MediaClip` (one JSON record per line)."""
    clips: list[MediaClip] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            clips.append(MediaClip.from_json(json.loads(stripped)))
    return clips


def _clips_from_args(args: argparse.Namespace) -> list[MediaClip]:
    """Build the clip work-list from the parsed CLI arguments (`--clips-jsonl` or `--ids`)."""
    if args.clips_jsonl is not None:
        return _read_clips_jsonl(args.clips_jsonl)
    kind = MediaKind(args.kind)
    return [
        MediaClip(source_id=source_id, kind=kind, start_seconds=args.start, end_seconds=args.end)
        for source_id in args.ids
    ]


def _run_cli(args: argparse.Namespace) -> int:
    """Fetch every requested clip, write the payloads and the yield manifest, and report the yield."""
    clips = _clips_from_args(args)
    if not clips:
        print("acquire_clips: no clips requested (pass --ids or --clips-jsonl)", file=sys.stderr)
        return 2
    _warn_if_no_cookies()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = FetchManifest()
    for result in fetch_clips(
        clips,
        manifest=manifest,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        timeout_seconds=args.timeout_seconds,
    ):
        if result.succeeded and result.payload is not None:
            destination = out_dir / f"{_safe_stem(result.clip.source_id)}.{result.extension}"
            destination.write_bytes(result.payload)
            print(f"acquire_clips: ok   {result.clip.source_id} -> {destination.name}")
        else:
            print(f"acquire_clips: drop {result.clip.source_id} ({result.dropped_reason})")
    manifest_path = args.manifest if args.manifest is not None else out_dir / "fetch_manifest.json"
    manifest.write(manifest_path)
    print(
        f"acquire_clips: yield {len(manifest.succeeded)}/{len(manifest.attempted)} "
        f"({manifest.yield_fraction * 100:.1f}%); manifest at {manifest_path}"
    )
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube audio/video clips with yt-dlp; report an auditable per-shard yield."
    )
    parser.add_argument("--ids", nargs="*", default=[], help="YouTube ids to fetch (smoke / ad-hoc mode).")
    parser.add_argument(
        "--clips-jsonl",
        type=Path,
        default=None,
        help="Shard file of one JSON clip spec per line (overrides --ids; used under GNU parallel).",
    )
    parser.add_argument(
        "--kind",
        choices=[str(kind) for kind in MediaKind],
        default=str(MediaKind.VIDEO),
        help="Clip kind for --ids mode (default: video).",
    )
    parser.add_argument("--start", type=float, default=None, help="Trim-window start in seconds for --ids mode.")
    parser.add_argument("--end", type=float, default=None, help="Trim-window end in seconds for --ids mode.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("fetched_clips"), help="Directory the payloads + manifest are written to."
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="Manifest path (default: <out-dir>/fetch_manifest.json)."
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Per-clip transient retry budget.")
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=DEFAULT_BACKOFF_SECONDS,
        help="Base exponential backoff between retries.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Wall-clock ceiling per yt-dlp invocation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(_run_cli(_parse_cli_args()))
