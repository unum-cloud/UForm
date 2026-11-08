"""Mine K hard negatives per (anchor, positive) pair via a pretrained CLIP encoder.

Modality: image+text (coco_captions) — also runnable for video+text by encoding the
first frame of each clip (msrvtt_captions).

Per-shard mining: for each registered corpus shard, decodes every positive media on
host, encodes it with `openai/clip-vit-base-patch32` (fp16 on B200), builds a brute-force
cosine kNN within that shard, and adds two columns __in place__:

    negative_indices: list<int32>   — K shard-local row indices of the visually-most-
                                       similar non-self media per row
    negative_keys:    list<string>  — matching keys (cocoid / video_id) for dedup audits

Schema choice rationale:

cuDF's parquet reader hangs the GPU driver on `list<binary>` reads, and flat `binary` columns are
read as `string` columns internally — the nested-binary path is not built out. `list<int32>` is
fully supported (the same type `_sample_list_element` and the Mr. TyDi extractor read). Storing K=8
row indices per row plus K matching keys costs ~96 bytes per row — vs ~1.2 MB per row for inline
`list<binary>`. Disk overhead for COCO's 118K rows is ~12 MB total; for MSR-VTT's 9K rows ~900 KB.
No GPU hang.

Constraint: mining is __per-shard__ (negatives drawn only from the same shard's ~3K candidates for
COCO, ~9K for MSR-VTT single-file). Per-shard pools are still well-sized for hard-negative quality.
The trainer hot path resolves the indices to bytes via
`frame[image_field].struct.field('bytes').take(picked_indices)` on the __full__ shard frame at batch
yield time (see `ImageTextParquetLoader._iter_cudf`).

This is an __offline pre-processing pass__, not a training hot path — PIL is used
for decode here because the no-CPU-decode rule governs only the training prefetcher
path, not offline builds.

Each shard is rewritten __atomically in place__ — temp file, then `os.replace` to overwrite the
source. No `.with-mined-negatives.parquet` siblings: one form per dataset.

CLI:
    uv run scripts/data/mine_hard_negatives.py --corpus coco_captions --k 8
    uv run scripts/data/mine_hard_negatives.py --corpus msrvtt_captions --k 8

Pass `--limit-shards N` for a small validation run before the full mining sweep.
Pass `--overwrite` to re-mine shards that already carry `negative_indices`.

File: scripts/data/mine_hard_negatives.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "av",
#     "pillow>=10.0",
#     "pyarrow>=21.0.0",
#     "torch",
#     "tqdm",
#     "transformers>=4.40",
# ]
# ///

import argparse
import glob
import io
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mixtures
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

#: HuggingFace model id; ViT-B/32 is the standard mining encoder (~88M params, 512-d).
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

#: GPU device. B200 / single-device mining (corpora are small enough).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Per-shard encode batch size for the CLIP image encoder.
ENCODE_BATCH_SIZE = 128

#: Thread-pool width for PIL decode. PIL releases the GIL during decode, so threads scale; matches
#: the 160-core box without oversubscribing the GPU encode.
DECODE_THREAD_POOL_SIZE = 32

#: Corpus -> key-column mapping.
#: Only the key column is mining-specific (stable per-row identifier for negative dedup audits)
#: and lives here; the shard glob and media column are the mixture JSONs' single-source-of-truth
#: (`mixtures.dataset_glob(name)` / `.dataset_media_column(name)`), no longer inlined.
CORPUS_KEY_COLUMN: dict[str, str] = {
    "coco_captions": "cocoid",
    "msrvtt_captions": "video_id",
}


def _resolve_corpus_shards(corpus_name: str) -> tuple[list[Path], str, str]:
    """Resolve a corpus name to its parquet shards + media + key column names.

    Glob and media column come from the mixture JSONs; the mining-specific key column from
    `CORPUS_KEY_COLUMN`.
    """
    if corpus_name not in CORPUS_KEY_COLUMN:
        raise ValueError(
            f"corpus {corpus_name!r}: not in CORPUS_KEY_COLUMN; add an entry. Available: {sorted(CORPUS_KEY_COLUMN)}"
        )
    shard_glob = mixtures.dataset_glob(corpus_name)
    media_column = mixtures.dataset_media_column(corpus_name)
    key_column = CORPUS_KEY_COLUMN[corpus_name]
    shard_paths = sorted(Path(p) for p in glob.glob(shard_glob, recursive=True))
    if not shard_paths:
        raise FileNotFoundError(f"corpus {corpus_name!r}: no shards under {shard_glob}")
    return shard_paths, media_column, key_column


def _extract_image_bytes(media_value) -> bytes | None:
    """Pull image bytes out of a row's media value (struct{bytes, path} or raw bytes)."""
    if isinstance(media_value, dict):
        return media_value.get("bytes")
    if isinstance(media_value, (bytes, bytearray, memoryview)):
        return bytes(media_value)
    return None


def _decode_image(raw_bytes: bytes | None) -> Image.Image | None:
    """Decode JPEG/PNG bytes to a PIL RGB image; return `None` on failure or empty input."""
    if not raw_bytes:
        return None
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception:
        return None


def _decode_video_first_frame(raw_bytes: bytes | None) -> Image.Image | None:
    """Extract the first frame of an MP4 as a PIL RGB image (msrvtt path), via pyav."""
    if not raw_bytes:
        return None
    try:
        import av  # noqa: PLC0415 — lazy import; only needed on video mining path
    except ImportError:
        print(
            "mine_hard_negatives: video-corpus mining needs `pyav` (`pip install av`); "
            "image-corpus mining works without it.",
            file=sys.stderr,
        )
        raise
    try:
        with av.open(io.BytesIO(raw_bytes)) as container:
            for frame in container.decode(video=0):
                image = frame.to_image()
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image
    except Exception:
        return None
    return None


def encode_shard(
    table: pa.Table,
    media_column: str,
    decode_fn,
    model: CLIPModel,
    processor: CLIPProcessor,
    decode_pool: ThreadPoolExecutor,
) -> tuple[torch.Tensor, list[int]]:
    """Encode every decodable media in one shard's table.

    Args:
        table: pyarrow Table for one shard.
        media_column: Column holding image bytes (struct{bytes,path}) or raw mp4 binary.
        decode_fn: `_decode_image` or `_decode_video_first_frame`.
        model: Loaded CLIP model on `DEVICE`.
        processor: Matching CLIPProcessor.
        decode_pool: ThreadPoolExecutor for parallel host-side decode.

    Returns:
        `(embeddings, valid_row_indices)`:
        - `embeddings`: `(M, 512)` cosine-normalised fp16 tensor on `DEVICE` where M is
          the number of rows whose media decoded successfully.
        - `valid_row_indices`: list of length M; original shard row indices for the
          rows whose media decoded.  Rows with decode failures are dropped here; the
          caller must restrict the negative-index column accordingly.
    """
    raw_media = [_extract_image_bytes(v) for v in table.column(media_column).to_pylist()]
    decoded = list(decode_pool.map(decode_fn, raw_media))
    valid_row_indices = [i for i, img in enumerate(decoded) if img is not None]
    if not valid_row_indices:
        return torch.empty((0, model.config.projection_dim), dtype=torch.float16, device=DEVICE), []

    valid_images = [decoded[i] for i in valid_row_indices]
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(valid_images), ENCODE_BATCH_SIZE):
        batch = valid_images[start : start + ENCODE_BATCH_SIZE]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            feat = model.visual_projection(pooled)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        embeddings.append(feat)
    return torch.cat(embeddings, dim=0), valid_row_indices


def find_topk_within_shard(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """Per-row top-K cosine kNN within a single shard, excluding self.

    Args:
        embeddings: `(M, D)` cosine-normalised fp16 tensor on `DEVICE`.
        k: Number of negatives to return per row.

    Returns:
        `(M, k)` int32 tensor on `DEVICE`, each row holds K column indices into the
        shard (the K hardest non-self negatives).
    """
    row_count = embeddings.shape[0]
    similarity = embeddings.float() @ embeddings.float().T  # (M, M) fp32
    diag = torch.arange(row_count, device=DEVICE)
    similarity[diag, diag] = float("-inf")  # exclude self
    return torch.topk(similarity, k, dim=1).indices.to(torch.int32)


def mine_one_shard(
    shard_path: Path,
    media_column: str,
    key_column: str,
    is_video: bool,
    k: int,
    model: CLIPModel,
    processor: CLIPProcessor,
    decode_pool: ThreadPoolExecutor,
    overwrite: bool,
) -> None:
    """Mine K hard negatives per row of one shard and rewrite the parquet in place.

    Reads the shard, encodes every decodable media, builds within-shard kNN, then
    rewrites the parquet atomically with two new columns: `negative_indices` (list<int32>,
    shard-local row indices) and `negative_keys` (list<string>, matched keys).  Rows
    whose media failed to decode get empty lists in both columns so row alignment is
    preserved (downstream extractors drop empty negatives via `_sample_list_element`'s
    safe-len fallback or via an explicit length filter).

    Args:
        shard_path: Parquet shard to mine + rewrite in place.
        media_column: Column with the positive media payload.
        key_column: Stable per-row identifier column (cocoid / video_id).
        is_video: When `True`, decode the first frame of each MP4 instead of as image.
        k: Number of hard negatives per row.
        model: Pre-loaded CLIP encoder.
        processor: Pre-loaded CLIP processor.
        decode_pool: Pre-loaded media decode pool.
        overwrite: When `False`, skip shards that already have a `negative_indices` column.
    """
    decode_fn = _decode_video_first_frame if is_video else _decode_image
    table = pq.read_table(shard_path)
    if "negative_indices" in table.column_names and not overwrite:
        return  # idempotent
    # With --overwrite, drop any pre-existing mined columns BEFORE re-appending —
    # otherwise `append_column` happily duplicates the names and the resulting
    # parquet has `[..., negative_indices, negative_keys, negative_indices, negative_keys]`,
    # which pyarrow's dataset factory rejects (`ArrowInvalid: Can't unify schema with
    # duplicate field names`) on subsequent reads.
    for stale in ("negative_indices", "negative_keys"):
        if stale in table.column_names:
            table = table.drop([stale])

    embeddings, valid_row_indices = encode_shard(
        table,
        media_column,
        decode_fn,
        model,
        processor,
        decode_pool,
    )
    n_rows = table.num_rows

    # Build per-row negative_indices / negative_keys lists with empty lists for the
    # rows whose media failed to decode.  Indices are stored as shard-local int32.
    negative_indices_per_row: list[list[int]] = [[] for _ in range(n_rows)]
    negative_keys_per_row: list[list[str]] = [[] for _ in range(n_rows)]
    if embeddings.shape[0] >= k + 1:
        topk = find_topk_within_shard(embeddings, k).cpu().numpy()  # (M, k), shard-local int32
        # Map M-indexed topk rows back to original shard row indices.
        valid_to_shard = valid_row_indices  # list[int], length M
        all_keys = table.column(key_column).to_pylist()
        for m_row, shard_row in enumerate(valid_to_shard):
            # topk[m_row] holds M-space indices; translate each back to shard space.
            translated = [int(valid_to_shard[int(idx)]) for idx in topk[m_row]]
            negative_indices_per_row[shard_row] = translated
            negative_keys_per_row[shard_row] = [str(all_keys[i]) for i in translated]

    new_table = table.append_column(
        "negative_indices",
        pa.array(negative_indices_per_row, type=pa.list_(pa.int32())),
    ).append_column(
        "negative_keys",
        pa.array(negative_keys_per_row, type=pa.list_(pa.string())),
    )

    # Atomic in-place replace: tmp then os.replace.
    tmp_path = str(shard_path) + ".tmp"
    pq.write_table(new_table, tmp_path, compression="zstd", row_group_size=1024)
    os.replace(tmp_path, shard_path)


def main() -> int:
    """Mine hard negatives for every shard of the selected corpus."""
    parser =argparse.ArgumentParser(description="Mine K hard negatives per row via CLIP within-shard kNN.")
    parser.add_argument(
        "--corpus", required=True, choices=("coco_captions", "msrvtt_captions"), help="Registered corpus name to mine."
    )
    parser.add_argument("--k", type=int, default=8, help="Number of hard negatives per row.")
    parser.add_argument(
        "--limit-shards", type=int, default=None, help="Only process the leading N shards (validation runs)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-mine shards that already carry a `negative_indices` column."
    )
    args = parser.parse_args()

    shard_paths, media_column, key_column = _resolve_corpus_shards(args.corpus)
    if args.limit_shards is not None:
        shard_paths = shard_paths[: args.limit_shards]
    is_video = args.corpus == "msrvtt_captions"
    print(
        f"mine_hard_negatives: {args.corpus} — {len(shard_paths)} shards, k={args.k}, "
        f"video={is_video}, schema=list<int32>+list<string>",
        flush=True,
    )

    # Install SIGALRM watchdog so a per-shard cuDF/Torch hang doesn't wedge the GPU
    # indefinitely.
    def _shard_timeout(signum, frame):
        raise TimeoutError("per-shard mining exceeded watchdog timeout")

    signal.signal(signal.SIGALRM, _shard_timeout)

    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    with ThreadPoolExecutor(max_workers=DECODE_THREAD_POOL_SIZE) as decode_pool:
        for shard_path in tqdm(shard_paths, desc="mining shards"):
            signal.alarm(600)  # 10 min/shard watchdog
            try:
                mine_one_shard(
                    shard_path,
                    media_column,
                    key_column,
                    is_video,
                    args.k,
                    model,
                    processor,
                    decode_pool,
                    args.overwrite,
                )
            except TimeoutError:
                print(f"\n  TIMEOUT on {shard_path} — skipping (re-run with --overwrite)", file=sys.stderr)
            finally:
                signal.alarm(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
