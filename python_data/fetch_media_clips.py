# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "yt-dlp",
# ]
# ///
"""Host-side YouTube media-clip fetcher shared by the fetch-dependent dataset builders.

Modality: audio + video.  Several research corpora (Auto-ACD, VATEX, ActivityNet Captions) ship
only captions plus a YouTube id and an optional `[start, end]` window — the media itself has to be
fetched at repack time.  This module is the single host-side helper those builders call: given an
iterable of `MediaClip` records it downloads each clip with `yt-dlp` (audio: `bestaudio` extracted to
mp3; video: an H.264 `bestvideo+bestaudio` mux trimmed to the requested window, written as mp4),
RETRIES transient failures with exponential backoff, SKIPS permanently dead / private / geo-blocked
ids, and accounts for every attempted id in a `FetchManifest` so a shard's yield is auditable and is
never silently truncated.

Concurrency: this module never spawns an in-process `Pool` / `ThreadPool`; it fetches
one clip at a time.  Shard-level parallelism is the caller's job — each dataset builder runs one
process per shard under GNU `parallel`, and every such process drives this helper serially.

`yt-dlp` is declared in the PEP 723 header above so `uv run python_data/fetch_media_clips.py ...`
provisions it (and the binary is invoked as `python -m yt_dlp`, so the module is never imported into
the caller's interpreter).  `ffmpeg` must be on `PATH` for the mp3 extraction and the keyframe-accurate
mp4 trim.

CLI usage:
    # Smoke-test a handful of public ids (some failures are expected and reported, not fatal):
    uv run python_data/fetch_media_clips.py --kind video --ids dQw4w9WgXcQ aqz-KE-bpKQ \\
        --out-dir /tmp/clips --manifest /tmp/clips/manifest.json
    # Fetch a 10-second audio window from one id:
    uv run python_data/fetch_media_clips.py --kind audio --ids YS6CoSybkYU --start 30 --end 40 \\
        --out-dir /tmp/audio
    # Shard mode used by the dataset builders under GNU parallel (one JSON clip spec per line):
    uv run python_data/fetch_media_clips.py --clips-jsonl shard_007.jsonl --out-dir staging/shard_007
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

#: Default retry budget for a single clip before it is dropped as failed.
DEFAULT_RETRIES: int = 3
#: Default base backoff (seconds); attempt `n` sleeps `backoff * 2**n` before retrying.
DEFAULT_BACKOFF_SECONDS: float = 2.0
#: Default wall-clock ceiling for one `yt-dlp` invocation before it is killed and retried.
DEFAULT_TIMEOUT_SECONDS: float = 240.0

#: Substrings (matched case-insensitively in yt-dlp output) that mark a PERMANENT failure: the clip is
#: gone / private / geo-blocked / rights-stripped, so retrying cannot help and the id is dropped at once.
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

#: Cap the fetched video rendition to this height (pixels). The encoder trains at a small patch grid, so
#: there is no benefit to pulling 1080p+; 480p slashes bandwidth, disk, and per-clip memory.
_MAX_VIDEO_HEIGHT: int = 480


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
                    permanent = _classify_failure(completed.stdout + "\n" + completed.stderr)
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
    shard of media bytes in memory.  It NEVER parallelises internally — the caller
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
        print("fetch_media_clips: no clips requested (pass --ids or --clips-jsonl)", file=sys.stderr)
        return 2
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
            print(f"fetch_media_clips: ok   {result.clip.source_id} -> {destination.name}")
        else:
            print(f"fetch_media_clips: drop {result.clip.source_id} ({result.dropped_reason})")
    manifest_path = args.manifest if args.manifest is not None else out_dir / "fetch_manifest.json"
    manifest.write(manifest_path)
    print(
        f"fetch_media_clips: yield {len(manifest.succeeded)}/{len(manifest.attempted)} "
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
