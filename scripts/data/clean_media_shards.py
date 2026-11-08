"""Offline ingestion-cleanse: decode-verify every media row and rewrite corrupt-free Parquet.

Modality: omni (images today; audio + video registered). A single truncated PNG in a training
shard W=8-deadlocked a live omni run — once one rank's image decoder threw mid-step the remaining
ranks blocked forever on the next collective. This tool removes that failure mode at the source:
a one-time, off-line pass that fully decodes every media payload, drops the rows that fail to
decode, and writes cleaned shards to a *parallel* output tree (the originals are never touched).
Training then reads the `*_clean` tree and never sees an undecodable payload.

TWO decoders, picked with `--decoder` (default `gpu` for images):

  - `gpu` (DEFAULT, images) — probes with the EXACT on-device decoder training uses
    (`cuda_binding._ext` `ByteTape.from_cudf_binary` + `image_codec_pool_t.decode_image_tape_resized`,
    same path as `omni_loader._decode_images_robust`). This is the only decoder that catches what
    training catches: PIL is NOT a superset of the GPU decoder's reject-set — a real omni run
    deadlocked on a COCO PNG the GPU decoder rejects (`libpng error: PNG input buffer is incomplete`
    → C++ `codec_rejected`) while a CPU/PIL probe found that same shard 100% decodable. The probe
    runs in ONE process / one CUDA context and parallelises via GPU batch SIMD (a whole-tape decode
    over a batch; on a `RuntimeError` it falls back to per-row sub-tapes to identify the offenders),
    NOT via CPU processes. A row that raises `codec_rejected` / `shape_mismatch` is DROPPED.

    GPU-CONTENTION GUARD: this mode runs a kernel on a training GPU. Run it only in an idle-GPU
    window — do NOT run it blindly while the two W=8 training runs own all eight GPUs. `--device`
    is required in `gpu` mode (no implicit pick) and the batch is kept modest; check `nvidia-smi`
    before launch and confirm the chosen device is idle.

  - `cpu` — decodes with Pillow on the host (no CUDA / cuDF / CuPy). Parallelised with a
    context-managed `concurrent.futures.ThreadPoolExecutor`: PIL releases the GIL during decode, so
    threads parallelise the pixel pipeline without pickling Arrow chunks across processes (the old
    `multiprocessing.Pool` paid that pickling + per-fork re-import cost). Useful as a fast,
    GPU-free smoke / triage pass, but it does NOT reproduce the GPU decoder's reject-set — prefer
    `gpu` for the authoritative cleanse.

The unit of work is a (shard, row-range) chunk. Large shards (imagenet-1k-128x128 ~98k rows/shard)
split into row chunks so no single unit is a long pole; small shards (coco ~3k rows) are one chunk.

Idempotent + resumable: a shard whose cleaned output already exists is skipped, so a re-run
continues where an interrupted one stopped. `--dry-run` probes and reports counts without writing.
Originals are never touched: cleaned shards land in a parallel `*_clean` tree (or `--out-dir`) via
atomic temp-then-rename, with a per-modality drop manifest JSON.

    # GPU probe (authoritative) of a few coco shards on an IDLE device, write nothing:
    uv run scripts/data/clean_media_shards.py --modality images --decoder gpu --device 0 \
        --datasets 'data/coco_captions/data/train-0000[0-3]*.parquet' --dry-run

    # Full GPU cleanse of every registered image dataset into the parallel *_clean trees:
    uv run scripts/data/clean_media_shards.py --modality images --decoder gpu --device 0

    # CPU/PIL triage pass (GPU-free), threaded:
    uv run scripts/data/clean_media_shards.py --modality images --decoder cpu --workers 64 --dry-run

File: scripts/data/clean_media_shards.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "pillow",
#     "pyarrow>=21.0.0",
# ]
# ///

import argparse
import concurrent.futures
import dataclasses
import glob
import io
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence

import cuda_binding._ext as ext
import cudf
import pyarrow
import pyarrow.parquet
from PIL import Image

LOGGER = logging.getLogger("clean_media_shards")

#: Rows per work chunk. Large shards (imagenet ~98k rows) split into this many chunks so a single
#: worker never becomes a long pole; small shards (coco ~3k rows) collapse to one chunk. Tuned so
#: a chunk is a few seconds of decode — small enough to balance, large enough to amortize startup.
DEFAULT_CHUNK_ROWS = 4096

#: ThreadPoolExecutor worker ceiling for the CPU/PIL path. The box has 128 physical cores but two
#: W=8 training runs need host headroom for their data loaders + dispatch threads, so the cleanse
#: caps itself to leave cores free. PIL releases the GIL during decode so threads do parallelise.
DEFAULT_WORKERS = min(64, os.cpu_count() or 1)

#: GPU-decoder batch size: rows handed to one whole-tape `decode_image_tape_resized` call. Kept
#: MODEST on purpose — this kernel runs on a training GPU, so a small batch bounds the transient
#: device-memory + SM footprint. A rejected row in a batch triggers a per-row re-decode of just
#: that batch, so a small batch also bounds the slow-path cost of a corrupt row.
DEFAULT_GPU_BATCH = 256

#: Decoded square side for the GPU probe.
#: The probe only asks "does this decode", so the exact resize target is immaterial;
#: a small side keeps the decode-and-resize workspace tiny.
#: The native source resolution (what actually rejects) is bounded by `DEFAULT_GPU_MAX_SOURCE_SIDE`.
DEFAULT_GPU_PROBE_SIDE = 64

#: Native (pre-resize) source-side upper bound the GPU decoder pool's workspace arena is sized for.
#: `decode_image_tape_resized` stages each image at FULL decoded resolution before resizing, so this
#: must bound the source side (COCO ~640) or the arena undersizes and decode reports
#: `shape_mismatch` on a VALID image — a false drop. Matches the omni loader's `max_source_side`
#: default (2048).
DEFAULT_GPU_MAX_SOURCE_SIDE = 2048

#: Per-image compressed-byte upper bound the GPU decoder pool pins. Sized generously for full-res
#: source JPEG/PNG payloads; a payload above this would be a genuine reject, not a false drop.
DEFAULT_GPU_MAX_COMPRESSED_BYTES = 8 * 1024 * 1024


def _decode_image(payload: bytes) -> None:
    """Fully decode an image payload, raising on truncation or corruption.

    `PIL.Image.verify()` only checks the container header and is documented to *not* catch a
    truncated image body; only forcing the pixel pipeline with `.load()` surfaces a truncated
    PNG/JPEG — which is exactly the failure that deadlocked training. We open a fresh image for
    `.load()` because `.verify()` leaves the file object in an unusable state.

    Args:
        payload: Raw encoded image bytes (PNG/JPEG/...).

    Raises:
        Exception: Any decode failure (the caller treats every exception as "drop this row").
    """
    with Image.open(io.BytesIO(payload)) as image:
        image.load()


def _decode_audio(payload: bytes) -> None:
    """Decode-probe an audio payload (registered, not yet active).

    The production audio path decodes through the C++ `ext.AudioDecoderPool` (libavcodec on
    device), which this CPU-only tool must not invoke. No CPU audio decoder (soundfile / PyAV /
    librosa / torchaudio) is installed in this environment, and the `data/audiocaps` and
    `data/audioset` trees hold no Parquet yet, so there is nothing to probe. This stub raises so
    the modality cannot be silently run as a no-op that "passes" everything; install a CPU decoder
    and replace the body before enabling audio cleansing.

    Args:
        payload: Raw encoded audio bytes (WAV for AudioCaps, OGG-Opus for AudioSet).

    Raises:
        NotImplementedError: Always — see the modality TODO above.
    """
    raise NotImplementedError(
        "audio decode-probe needs a CPU codec (soundfile/PyAV/librosa); none installed and no audio "
        "Parquet present yet — see _decode_audio docstring"
    )


@dataclasses.dataclass(frozen=True)
class Modality:
    """A registered media modality: how to find its bytes column and how to decode-probe it.

    Attributes:
        name: CLI modality token (`images`, `audio`, `video`).
        default_globs: Default shard globs when `--datasets` is omitted.
        column_candidates: Ordered candidate top-level column names holding the media. The first
            present column is used; a `struct<bytes, path>` (HuggingFace media layout) is unwrapped
            to its `bytes` child, a bare `binary` column is used directly.
        decoder: CPU decode-probe; raises on undecodable payload.
        available: Whether the modality can actually run (False marks a registered-but-TODO entry).
        unavailable_reason: Human-readable reason when `available` is False.
    """

    name: str
    default_globs: tuple[str, ...]
    column_candidates: tuple[str, ...]
    decoder: Callable[[bytes], None]
    available: bool = True
    unavailable_reason: str = ""


#: The modality registry.
#: Adding a modality is a single entry here;
#: video is a registered no-op until msrvtt (or another source) lands Parquet, at which point
#: it gets an `available=True` decoder.
#: The output tree for any dataset is its directory with a `_clean` suffix on the dataset root
#: (see `_clean_output_path`), so cleansing never overwrites originals.
MODALITY_REGISTRY: dict[str, Modality] = {
    "images": Modality(
        name="images",
        default_globs=(
            "data/coco_captions/data/*.parquet",
            "data/imagenet-1k-128x128/**/*.parquet",
        ),
        column_candidates=("image", "jpg", "png", "img"),
        decoder=_decode_image,
    ),
    "audio": Modality(
        name="audio",
        default_globs=(
            "data/audiocaps/**/*.parquet",
            "data/audioset/**/*.parquet",
        ),
        column_candidates=("audio", "flac", "wav", "ogg"),
        decoder=_decode_audio,
        available=False,
        unavailable_reason=(
            "no CPU audio codec installed (soundfile/PyAV/librosa/torchaudio) and no audio Parquet "
            "downloaded yet; prod audio decode is the C++ on-device AudioDecoderPool which this "
            "CPU-only tool must not use"
        ),
    ),
    "video": Modality(
        name="video",
        default_globs=(),
        column_candidates=("video", "mp4", "webm"),
        decoder=_decode_image,  # placeholder; never reached while available=False
        available=False,
        unavailable_reason="msrvtt has no Parquet yet — registered no-op until video data lands",
    ),
}


def _resolve_media_column(schema: pyarrow.Schema, candidates: Sequence[str]) -> str:
    """Pick the media-bytes column from a shard schema by candidate order.

    Args:
        schema: The shard's Arrow schema.
        candidates: Ordered preferred column names.

    Returns:
        The chosen top-level column name.

    Raises:
        KeyError: No candidate column is present in the schema.
    """
    names = set(schema.names)
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"no media column among {tuple(candidates)} in schema columns {schema.names}")


def _payload_of(cell: object) -> bytes:
    """Extract raw encoded bytes from one media cell.

    Handles both the HuggingFace `struct<bytes, path>` layout (a dict with a `bytes` key after
    pyarrow `to_pylist`) and a bare `binary` column (already bytes).

    Args:
        cell: One decoded Arrow cell (dict for struct columns, bytes for binary columns).

    Returns:
        The raw encoded payload bytes.

    Raises:
        TypeError: The cell is neither a `{bytes: ...}` struct nor raw bytes.
    """
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, dict) and "bytes" in cell:
        return bytes(cell["bytes"])
    raise TypeError(f"unrecognized media cell type {type(cell).__name__}")


@dataclasses.dataclass(frozen=True)
class ChunkSpec:
    """One unit of work: a contiguous row range of one shard."""

    shard_path: str
    column: str
    decoder: Callable[[bytes], None]
    row_offset: int
    row_count: int


@dataclasses.dataclass
class ChunkResult:
    """Outcome of probing one chunk."""

    shard_path: str
    row_offset: int
    keep_mask: list[bool]
    drops: list[tuple[int, str]]  # (absolute_row_index, reason)


def _probe_chunk_cpu(spec: ChunkSpec) -> ChunkResult:
    """CPU/PIL decode-probe of every row in a chunk; return which rows to keep and why others drop.

    Runs in a `ThreadPoolExecutor` worker. PIL releases the GIL while it decodes pixels, so many
    of these run concurrently without process forking or Arrow-chunk pickling. Reads only its own
    row slice of the shard (pyarrow column projection keeps the read cheap).

    Args:
        spec: The chunk to probe.

    Returns:
        A `ChunkResult` with a per-row keep mask (chunk-local order) and absolute-indexed drops.
    """
    table = pyarrow.parquet.read_table(spec.shard_path, columns=[spec.column])
    column = table.column(spec.column).slice(spec.row_offset, spec.row_count).to_pylist()

    keep_mask: list[bool] = []
    drops: list[tuple[int, str]] = []
    for local_index, cell in enumerate(column):
        absolute_index = spec.row_offset + local_index
        try:
            payload = _payload_of(cell)
            spec.decoder(payload)
            keep_mask.append(True)
        except Exception as error:  # noqa: BLE001 — any decode failure means drop the row
            keep_mask.append(False)
            drops.append((absolute_index, f"{type(error).__name__}: {error}"))
    return ChunkResult(spec.shard_path, spec.row_offset, keep_mask, drops)


def _probe_shard_gpu(
    shard_path: str,
    column: str,
    total_rows: int,
    decoder_pool: object,
    byte_tape_factory: Callable[..., object],
    device: int,
    probe_side: int,
    gpu_batch: int,
) -> ChunkResult:
    """GPU decode-probe a whole shard with the SAME on-device decoder training uses.

    Mirrors `omni_loader._decode_images_robust`: load the shard's media column into a cuDF BINARY
    column, then for each `gpu_batch`-row window build an `ext.ByteTape.from_cudf_binary` over the
    slice's device pointers and decode the whole tape in one call. If that raises a `RuntimeError`
    (the C++ decoder's `codec_rejected` / `shape_mismatch` — exactly what desync-deadlocked the live
    omni run), fall back to per-row single-row sub-tapes to identify WHICH rows reject; those rows
    are dropped, the rest kept. The cuDF offsets are absolute and the C++ decode rebases the body
    from `offsets[0]`, so advancing the offsets pointer by `i` int32 entries with `count=1` hands the
    decoder exactly row `i` (same per-row sub-tape trick as the omni loader).

    Single CUDA context, single process: the parallelism is the GPU batch SIMD inside one
    `decode_image_tape_resized`, not host threads/processes.

    Args:
        shard_path: Source shard path.
        column: Resolved media column name (a HF `struct<bytes,path>` is unwrapped to its `bytes`).
        total_rows: Shard row count (the returned keep mask covers all of them, in order).
        decoder_pool: A constructed `ext.ImageDecoderPool`.
        byte_tape_factory: `ext.ByteTape.from_cudf_binary` (injected by `_build_gpu_decoder`).
        device: CUDA device ordinal the pool was built on (tapes are built on the same device).
        probe_side: Decoded square side passed to `decode_image_tape_resized` (resize target only).
        gpu_batch: Rows per whole-tape decode call.

    Returns:
        A `ChunkResult` over the whole shard (`row_offset == 0`), keep mask in row order.
    """
    keep_mask: list[bool] = [True] * total_rows
    drops: list[tuple[int, str]] = []

    frame = cudf.read_parquet(shard_path, columns=[column])
    media_series = frame[column]
    # HF media layout is a struct<bytes,path>; the bytes child is the encoded payload column.
    if str(media_series.dtype).startswith("struct"):
        media_series = media_series.struct.field("bytes")

    for batch_start in range(0, total_rows, gpu_batch):
        batch_count = min(gpu_batch, total_rows - batch_start)
        # Slice the cuDF column to this window; pointers are slice-rebased (mirrors the image loader's
        # `_cudf_binary_device_pointers`): cuDF slices are views that record `col.offset` but leave the
        # parent offsets child un-advanced, so the offsets pointer must step by `col.offset` int32s.
        sliced = media_series.iloc[batch_start : batch_start + batch_count]._column
        plc_column = sliced.to_pylibcudf()
        offsets_base = int(plc_column.child(0).data().__cuda_array_interface__["data"][0])
        data_ptr = int(sliced.data.ptr)
        offsets_ptr = offsets_base + int(sliced.offset) * 4

        tape = byte_tape_factory(data_ptr, offsets_ptr, batch_count, device=device)
        try:
            decoder_pool.decode_image_tape_resized(tape, probe_side, probe_side)
            continue  # whole batch decoded — every row in this window is kept
        except RuntimeError:
            pass

        # At least one row in this window rejects: re-decode each row in isolation to find which.
        # data_ptr stays the parent base (offsets are absolute); only the offsets pointer advances.
        for local_index in range(batch_count):
            row_tape = byte_tape_factory(data_ptr, offsets_ptr + local_index * 4, 1, device=device)
            try:
                decoder_pool.decode_image_tape_resized(row_tape, probe_side, probe_side)
            except RuntimeError as error:  # codec_rejected / shape_mismatch → drop this row
                absolute_index = batch_start + local_index
                keep_mask[absolute_index] = False
                drops.append((absolute_index, f"{type(error).__name__}: {error}"))

    return ChunkResult(shard_path, 0, keep_mask, drops)


def _least_busy_gpu() -> int | None:
    """Return the ordinal of the least-busy visible GPU via `nvidia-smi`, or None if unavailable.

    "Least busy" = lowest used-memory among GPUs reporting 0% utilization; if none are idle, the
    lowest used-memory GPU overall. This is only a *suggestion* surfaced to the operator when
    `--device` is omitted in gpu mode — gpu mode REQUIRES an explicit `--device` (the caller errors
    out and prints this hint) so the cleanse never silently lands on a GPU a training run owns.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    rows: list[tuple[int, int, int]] = []
    for line in out.strip().splitlines():
        index_s, mem_s, util_s = (field.strip() for field in line.split(","))
        rows.append((int(index_s), int(mem_s), int(util_s)))
    if not rows:
        return None
    idle = [r for r in rows if r[2] == 0]
    pick_from = idle if idle else rows
    return min(pick_from, key=lambda r: r[1])[0]


def _build_gpu_decoder(
    device: int, gpu_batch: int, max_source_side: int, max_compressed_bytes: int
) -> tuple[object, Callable[..., object]]:
    """Construct the on-device image decoder pool training uses, plus the ByteTape factory.

    The pool is built on `device` and sized so a VALID full-resolution image does not falsely reject
    — `max_source_side` bounds the native (pre-resize) side because `decode_image_tape_resized`
    stages each image at full resolution before resizing (undersizing
    the arena surfaces as a spurious `shape_mismatch`); see the omni loader's pool construction.

    Args:
        device: CUDA device ordinal to build the pool on.
        gpu_batch: Upper-bound batch the pool is sized for (== the whole-tape decode window).
        max_source_side: Native source-resolution upper bound for the workspace arena.
        max_compressed_bytes: Per-image compressed-byte upper bound the pool pins.

    Returns:
        `(decoder_pool, byte_tape_factory)` — the `ext.ImageDecoderPool` and
        `ext.ByteTape.from_cudf_binary`.
    """
    pool = ext.ImageDecoderPool(
        int(gpu_batch),
        int(max_compressed_bytes),
        int(max_source_side),
        int(max_source_side),
        device=int(device),
    )
    return pool, ext.ByteTape.from_cudf_binary


def _chunk_specs_for_shard(shard_path: str, modality: Modality, chunk_rows: int) -> tuple[list[ChunkSpec], int, str]:
    """Build the chunk work units for one shard.

    Args:
        shard_path: Path to the source shard.
        modality: The modality being cleansed.
        chunk_rows: Maximum rows per chunk.

    Returns:
        `(chunk_specs, total_rows, column)` — the per-chunk work units, the shard row count, and
        the resolved media column name.
    """
    metadata = pyarrow.parquet.ParquetFile(shard_path)
    total_rows = metadata.metadata.num_rows
    column = _resolve_media_column(metadata.schema_arrow, modality.column_candidates)
    specs = [
        ChunkSpec(shard_path, column, modality.decoder, offset, min(chunk_rows, total_rows - offset))
        for offset in range(0, total_rows, chunk_rows)
    ]
    return specs, total_rows, column


def _clean_output_path(shard_path: str, out_dir: str | None) -> str:
    """Map a source shard path to its cleaned output path in a parallel tree.

    With no `--out-dir`, the first path component is the `data/` root and the second is the dataset
    directory, which gets a `_clean` suffix (`data/coco_captions/...` -> `data/coco_captions_clean/...`)
    so originals are never overwritten. With `--out-dir`, the shard's path tail below `data/` (or its
    basename) is rehomed under that directory.

    Args:
        shard_path: Source shard path.
        out_dir: Optional explicit output root.

    Returns:
        Absolute-or-relative cleaned output path (its parent is created by the writer).
    """
    norm = os.path.normpath(shard_path)
    parts = norm.split(os.sep)
    if out_dir is not None:
        tail = os.sep.join(parts[parts.index("data") + 1 :]) if "data" in parts else os.path.basename(norm)
        return os.path.join(out_dir, tail)
    if "data" in parts:
        data_index = parts.index("data")
        dataset_index = data_index + 1
        if dataset_index < len(parts):
            parts[dataset_index] = parts[dataset_index] + "_clean"
            return os.sep.join(parts)
    root, ext = os.path.splitext(norm)
    return f"{root}_clean{ext}"


def _drop_manifest_path(out_dir: str | None, modality_name: str) -> str:
    """Path of the per-modality drop manifest JSON."""
    base = out_dir if out_dir is not None else "data"
    return os.path.join(base, f"clean_media_drops_{modality_name}.json")


@dataclasses.dataclass
class ShardOutcome:
    """Per-shard cleanse result for the summary + manifest."""

    shard_path: str
    output_path: str
    total_rows: int
    kept_rows: int
    dropped_rows: int
    drops: list[tuple[int, str]]
    skipped_existing: bool = False


def _write_clean_shard(shard_path: str, column: str, keep_mask: list[bool], output_path: str) -> None:
    """Write the kept rows of a shard to its cleaned output via atomic temp-then-rename.

    Reads the full shard (all columns — the clean copy keeps every field), filters by the keep mask,
    and writes to a `.tmp` sibling that is renamed into place only after a successful close, so an
    interrupted run never leaves a footer-less Parquet that the resume logic would skip.

    Args:
        shard_path: Source shard path (read for all columns).
        column: Unused beyond documentation symmetry; kept for signature clarity.
        keep_mask: Per-row keep flags in shard order.
        output_path: Destination cleaned shard path.
    """
    table = pyarrow.parquet.read_table(shard_path)
    mask = pyarrow.array(keep_mask, type=pyarrow.bool_())
    cleaned = table.filter(mask)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    pyarrow.parquet.write_table(cleaned, tmp_path, compression="zstd")
    os.replace(tmp_path, output_path)


def iter_shards(globs: Sequence[str]) -> Iterator[str]:
    """Yield deduplicated, sorted shard paths matching the given globs (recursive `**` honored)."""
    seen: set[str] = set()
    for pattern in globs:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                yield path


@dataclasses.dataclass(frozen=True)
class GpuProbeConfig:
    """GPU-decoder probe parameters (only consulted when `--decoder gpu`)."""

    device: int
    gpu_batch: int = DEFAULT_GPU_BATCH
    probe_side: int = DEFAULT_GPU_PROBE_SIDE
    max_source_side: int = DEFAULT_GPU_MAX_SOURCE_SIDE
    max_compressed_bytes: int = DEFAULT_GPU_MAX_COMPRESSED_BYTES


def _probe_cpu(
    pending_specs: Sequence[ChunkSpec], workers: int, modality_name: str, shard_count: int
) -> Iterator[ChunkResult]:
    """Decode-probe pending chunks on the host with a context-managed `ThreadPoolExecutor`.

    Submits one task per (shard, row-range) chunk, yields results as they complete with per-chunk
    exception propagation (a worker exception re-raises here, not silently swallowed). PIL releases
    the GIL during `.load()`, so the threads parallelise the decode without process forking.
    """
    effective_workers = max(1, min(workers, len(pending_specs)))
    LOGGER.info(
        "modality %r [cpu]: %d chunk(s) across %d shard(s), %d thread(s)",
        modality_name,
        len(pending_specs),
        shard_count,
        effective_workers,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_probe_chunk_cpu, spec): spec for spec in pending_specs}
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                yield future.result()
            except Exception:  # noqa: BLE001 — surface a chunk's read/probe failure with its identity
                LOGGER.error("cpu probe failed on %s rows [%d,+%d)", spec.shard_path, spec.row_offset, spec.row_count)
                raise


def _probe_gpu(
    shard_meta: dict[str, tuple[int, str, str]], modality_name: str, gpu: GpuProbeConfig
) -> Iterator[ChunkResult]:
    """Decode-probe every shard on `gpu.device` with the SAME on-device decoder training uses.

    Builds ONE decoder pool + CUDA context for the whole modality and probes shards sequentially
    (one whole-shard `ChunkResult`); the parallelism is the GPU batch SIMD inside each decode call,
    not host concurrency. Yields one `ChunkResult` per shard in `shard_meta`.
    """
    decoder_pool, byte_tape_factory = _build_gpu_decoder(
        gpu.device, gpu.gpu_batch, gpu.max_source_side, gpu.max_compressed_bytes
    )
    LOGGER.info(
        "modality %r [gpu]: %d shard(s) on device %d (batch=%d, probe_side=%d)",
        modality_name,
        len(shard_meta),
        gpu.device,
        gpu.gpu_batch,
        gpu.probe_side,
    )
    for shard, (total_rows, column, _output_path) in shard_meta.items():
        yield _probe_shard_gpu(
            shard,
            column,
            total_rows,
            decoder_pool,
            byte_tape_factory,
            gpu.device,
            gpu.probe_side,
            gpu.gpu_batch,
        )


def cleanse_modality(
    modality: Modality,
    globs: Sequence[str],
    out_dir: str | None,
    workers: int,
    chunk_rows: int,
    dry_run: bool,
    decoder_backend: str = "gpu",
    gpu: GpuProbeConfig | None = None,
) -> list[ShardOutcome]:
    """Decode-verify and rewrite every shard of one modality.

    Plans the pending shards (skipping any whose cleaned output already exists), probes them with the
    chosen backend — `cpu` (PIL, `ThreadPoolExecutor`) or `gpu` (the on-device decoder training uses,
    one CUDA context) — reassembles per-shard keep masks, writes cleaned shards (unless `--dry-run`),
    and writes the drop manifest. Idempotent + resumable via the skip-existing check.

    Args:
        modality: The registered modality to cleanse.
        globs: Shard globs.
        out_dir: Optional explicit output root.
        workers: ThreadPoolExecutor size (cpu backend only).
        chunk_rows: Maximum rows per chunk (cpu backend only; the gpu backend probes whole shards).
        dry_run: When True, probe and report but write nothing.
        decoder_backend: `"gpu"` (default for images) or `"cpu"`.
        gpu: GPU probe config; required when `decoder_backend == "gpu"`.

    Returns:
        Per-shard outcomes.
    """
    if not modality.available:
        LOGGER.warning("modality %r unavailable: %s", modality.name, modality.unavailable_reason)
        return []
    if decoder_backend == "gpu" and gpu is None:
        raise ValueError("gpu decoder backend requires a GpuProbeConfig")

    shards = list(iter_shards(globs))
    if not shards:
        LOGGER.warning("modality %r: no shards matched %s", modality.name, tuple(globs))
        return []

    # Plan: collect chunk specs for the shards we still need to process, remembering skips.
    pending_specs: list[ChunkSpec] = []
    shard_meta: dict[str, tuple[int, str, str]] = {}  # shard -> (total_rows, column, output_path)
    outcomes_by_shard: dict[str, ShardOutcome] = {}
    for shard in shards:
        output_path = _clean_output_path(shard, out_dir)
        if not dry_run and os.path.exists(output_path):
            LOGGER.info("skip (clean output exists): %s", shard)
            outcomes_by_shard[shard] = ShardOutcome(shard, output_path, 0, 0, 0, [], skipped_existing=True)
            continue
        specs, total_rows, column = _chunk_specs_for_shard(shard, modality, chunk_rows)
        shard_meta[shard] = (total_rows, column, output_path)
        pending_specs.extend(specs)

    # Probe every pending unit with the chosen backend, collecting per-shard keep masks + drops.
    keep_by_shard: dict[str, dict[int, list[bool]]] = {s: {} for s in shard_meta}
    drops_by_shard: dict[str, list[tuple[int, str]]] = {s: [] for s in shard_meta}
    if shard_meta:
        if decoder_backend == "cpu":
            results: Iterator[ChunkResult] = _probe_cpu(pending_specs, workers, modality.name, len(shard_meta))
        else:
            assert gpu is not None  # guarded above
            results = _probe_gpu(shard_meta, modality.name, gpu)
        for result in results:
            keep_by_shard[result.shard_path][result.row_offset] = result.keep_mask
            drops_by_shard[result.shard_path].extend(result.drops)

    # Reassemble per-shard keep masks in row order, write cleaned shards, build outcomes.
    for shard, (total_rows, column, output_path) in shard_meta.items():
        offsets = sorted(keep_by_shard[shard])
        keep_mask: list[bool] = []
        for offset in offsets:
            keep_mask.extend(keep_by_shard[shard][offset])
        drops = sorted(drops_by_shard[shard])
        kept = sum(keep_mask)
        dropped = total_rows - kept
        if not dry_run:
            _write_clean_shard(shard, column, keep_mask, output_path)
        outcomes_by_shard[shard] = ShardOutcome(shard, output_path, total_rows, kept, dropped, drops)

    outcomes = [outcomes_by_shard[s] for s in shards]

    if not dry_run:
        manifest_path = _drop_manifest_path(out_dir, modality.name)
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        manifest = [
            {"shard": o.shard_path, "row_index": row_index, "reason": reason}
            for o in outcomes
            for (row_index, reason) in o.drops
        ]
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        LOGGER.info("modality %r: wrote drop manifest %s (%d entries)", modality.name, manifest_path, len(manifest))

    return outcomes


def _print_summary(modality_name: str, outcomes: Sequence[ShardOutcome], dry_run: bool) -> None:
    """Print a per-modality summary table (CLI `__main__` boundary — prints are allowed here)."""
    total_rows = sum(o.total_rows for o in outcomes)
    total_dropped = sum(o.dropped_rows for o in outcomes)
    skipped = sum(1 for o in outcomes if o.skipped_existing)
    pct = (100.0 * total_dropped / total_rows) if total_rows else 0.0
    mode = "DRY-RUN" if dry_run else "WROTE"
    print(f"\nmodality {modality_name} [{mode}]")
    print(f"{'shard':<70} {'rows':>10} {'dropped':>9} {'%drop':>7}")
    for o in outcomes:
        if o.skipped_existing:
            print(f"{os.path.basename(o.shard_path):<70} {'(skipped: clean output exists)':>27}")
            continue
        opct = (100.0 * o.dropped_rows / o.total_rows) if o.total_rows else 0.0
        print(f"{os.path.basename(o.shard_path):<70} {o.total_rows:>10} {o.dropped_rows:>9} {opct:>6.3f}%")
    print(f"{'TOTAL':<70} {total_rows:>10} {total_dropped:>9} {pct:>6.3f}%")
    if skipped:
        print(f"({skipped} shard(s) skipped as already-clean)")
    if not dry_run and outcomes and not all(o.skipped_existing for o in outcomes):
        sample_out = next((o.output_path for o in outcomes if not o.skipped_existing), "")
        print(f"output tree: {os.path.dirname(sample_out)}")


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode-verify cleanse of media Parquet shards (gpu = the on-device decoder "
        "training uses; cpu = host PIL). Python-only, no C++/cmake."
    )
    parser.add_argument(
        "--modality",
        choices=(*MODALITY_REGISTRY.keys(), "all"),
        default="all",
        help="Which modality to cleanse (default: all available).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Shard glob(s); defaults to the modality's registered globs. '**' is recursive.",
    )
    parser.add_argument(
        "--decoder",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Decode backend. 'gpu' (DEFAULT) = the on-device decoder training uses (authoritative, "
        "catches what PIL misses; requires --device + an idle GPU). 'cpu' = host PIL (GPU-free triage; "
        "NOT a superset of the GPU reject-set).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device ordinal for --decoder gpu (REQUIRED in gpu mode). Pick an IDLE GPU — do not "
        "run blindly while the W=8 training runs own all eight GPUs.",
    )
    parser.add_argument("--out-dir", default=None, help="Explicit output root (default: parallel '*_clean' tree).")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="ThreadPoolExecutor size for --decoder cpu."
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS, help="Rows per cpu work chunk.")
    parser.add_argument(
        "--gpu-batch", type=int, default=DEFAULT_GPU_BATCH, help="Rows per whole-tape GPU decode (kept modest)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Probe and report counts; write nothing.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the CLI, cleanse the selected modalities shard by shard, and print a summary per modality."""
    args = _parse_cli_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    gpu: GpuProbeConfig | None = None
    if args.decoder == "gpu":
        if args.device is None:
            hint = _least_busy_gpu()
            hint_text = f" (least-busy visible GPU right now looks like device {hint})" if hint is not None else ""
            print(
                "--decoder gpu requires an explicit --device N: this runs a decode kernel on a training "
                f"GPU, so pick an IDLE device and confirm with `nvidia-smi`{hint_text}. Refusing to auto-pick "
                "so the cleanse never lands on a GPU a W=8 run owns.",
                file=sys.stderr,
            )
            return 2
        gpu = GpuProbeConfig(device=int(args.device), gpu_batch=int(args.gpu_batch))

    selected = list(MODALITY_REGISTRY.values()) if args.modality == "all" else [MODALITY_REGISTRY[args.modality]]
    any_ran = False
    for modality in selected:
        globs = tuple(args.datasets) if args.datasets is not None else modality.default_globs
        if not modality.available:
            if args.modality != "all":
                print(f"modality {modality.name}: UNAVAILABLE (TODO) — {modality.unavailable_reason}")
            continue
        outcomes = cleanse_modality(
            modality,
            globs,
            args.out_dir,
            args.workers,
            args.chunk_rows,
            args.dry_run,
            decoder_backend=args.decoder,
            gpu=gpu,
        )
        if outcomes:
            any_ran = True
            _print_summary(modality.name, outcomes, args.dry_run)
    if not any_ran:
        print("nothing cleansed (no available modality matched shards).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
