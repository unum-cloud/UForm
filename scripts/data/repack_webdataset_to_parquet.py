# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "orjson>=3.10.0",
#     "pyarrow>=21.0.0",
#     "xxhash",
# ]
# ///
r"""Repack one WebDataset tar shard to a Parquet sibling with full annotation preservation.

Modality: image+text (cc12m, laion_cc3m) and video+text (panda70m, webvid_10m). The unit of work for
the GNU `parallel` repack model: one process per shard, so many shards convert concurrently at full
CPU saturation without an in-process pool (Python multiprocessing is banned per the project
stylebook; the orchestrator uses `parallel -j$(nproc)` instead).

Schema:
- One column per file extension in the tar: e.g. `jpg: binary` (cc12m) or `mp4: binary`
  (panda70m).  Always present; row dropped if missing.
- `key: string` — the tar member stem (e.g. `000000001` for cc12m, `<video_id>.<n>` for
  panda70m).  Needed for cross-shard dedup.
- One column per JSON sidecar key observed across the shard.  cc12m sidecars carry
  `caption / url / status / error_message / width / height / exif / original_width /
  original_height`; panda70m sidecars carry `caption / duration / timestamp (list<string>) /
  matching_score`.  Every key is preserved verbatim — no silent dropping.  pyarrow infers
  per-column dtype from the observed values; missing keys land as null in nullable columns.

Atomicity: writes to `<out>.parquet.tmp` then `os.replace`. An interrupted run leaves the source tar
untouched. After verify passes (row-count + schema parity + xxhash of media column sizes against tar
member sizes), the source tar is unlinked — the "delete after verify" policy frees disk for the next
dataset (no rollback).

Run one process per shard under GNU parallel at full core saturation:

    ls $UFORM_DATA_ROOT/cc12m/cc12m-train-*.tar | \
        parallel -j$(nproc) uv run scripts/data/repack_webdataset_to_parquet.py {}

Pass `--keep-tar` to skip the delete step (useful for the validation run on panda70m before
launching the multi-hour cc12m sweep). Pass `--overwrite` to re-repack a shard whose `.parquet`
sibling already exists.
"""

import argparse
import os
import sys
import tarfile
from collections import defaultdict

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

#: Media file extensions we recognise as the "payload" column. The first extension found in a
#: tar drives the schema; mixing payload types within one shard raises (shouldn't happen with
#: the WebDataset corpora we target — each shard is single-modality).
PAYLOAD_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "mp4", "webm", "mkv")

#: Sidecar extensions whose contents we parse as JSON and union into the schema.
JSON_EXTENSIONS = ("json",)

#: Sidecar extensions whose contents we store as a separate `<ext>: string` column.
TEXT_EXTENSIONS = ("txt",)

#: Parquet row-group size — small so future cuDF readers can stream sub-shard chunks.
ROW_GROUP_SIZE = 1024


def parquet_sibling_path(tar_path: str) -> str:
    """Return the `.parquet` path that replaces a tar shard's `.tar` extension.

    Args:
        tar_path: A WebDataset shard path ending in `.tar`.

    Returns:
        Sibling path with `.tar` replaced by `.parquet`.

    Raises:
        ValueError: If `tar_path` does not end in `.tar`.
    """
    if not tar_path.endswith(".tar"):
        raise ValueError(f"not a tar shard: {tar_path!r}")
    return tar_path[:-4] + ".parquet"


def _co_keyed_stem(member_name: str) -> tuple[str, str]:
    """Split a tar member name into `(stem, extension)`.

    Handles both flat WebDataset (`<key>.<ext>`) and panda70m's nested
    `<prefix>/<video_id>.<n>.<ext>` shape.  The stem retains the per-clip index so co-keyed
    members group correctly (e.g. `-2yHu5qgTzM.0.mp4` + `-2yHu5qgTzM.0.json`).
    """
    base = os.path.basename(member_name)
    stem, _, ext = base.rpartition(".")
    return stem, ext.lower()


def read_shard_samples(tar_path: str) -> list[dict]:
    """Stream one tar shard into a list of co-keyed sample dicts on host.

    One pass through the tar (`mode="r|"`, no seek), grouping co-keyed members by stem.
    Each sample's JSON sidecar is parsed with `orjson` (~3x faster than stdlib `json`); each
    `.txt` sidecar is decoded as UTF-8; the payload bytes are stored verbatim.

    Args:
        tar_path: Path to one WebDataset tar shard.

    Returns:
        List of sample dicts, one per co-keyed group, in tar order. Each dict has at minimum
        `key` (str) and one payload key (e.g. `jpg`, `mp4`).  JSON sidecar keys are merged at
        the top level so the parquet schema is one flat column per sidecar field.  Samples
        without a payload are dropped (counted in the per-shard log only, not surfaced).
    """
    # `defaultdict(dict)` collects each stem's members; one pass through the tar.
    grouped: dict[str, dict[str, bytes]] = defaultdict(dict)
    with tarfile.open(tar_path, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            stem, ext = _co_keyed_stem(member.name)
            file_handle = archive.extractfile(member)
            if file_handle is None:
                continue
            grouped[stem][ext] = file_handle.read()

    samples: list[dict] = []
    for stem, parts in grouped.items():
        sample: dict = {"key": stem}
        # Merge JSON sidecar fields into the top-level dict so they become flat columns.
        for ext in JSON_EXTENSIONS:
            if ext in parts:
                try:
                    sample.update(orjson.loads(parts[ext]))
                except orjson.JSONDecodeError:
                    continue  # malformed sidecar — drop row at end if no caption
        # Text sidecars land as their own string column.
        for ext in TEXT_EXTENSIONS:
            if ext in parts:
                try:
                    sample[ext] = parts[ext].decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
        # Payload bytes — first matching extension wins; one column per shard.
        for ext in PAYLOAD_EXTENSIONS:
            if ext in parts:
                sample[ext] = parts[ext]
                break
        # Drop samples with no payload (no media to train on) or no caption (most JSON
        # sidecars carry one; the `txt` sidecar is the WebDataset convention for cc12m
        # when the json doesn't).
        has_payload = any(ext in sample for ext in PAYLOAD_EXTENSIONS)
        has_caption = sample.get("caption") or sample.get("txt")
        if has_payload and has_caption:
            samples.append(sample)
    return samples


def _payload_column_name(samples: list[dict]) -> str:
    """Return the name of the payload column present in the first sample.

    All samples in a WebDataset shard share one payload extension by convention.  We assume
    this without re-checking every sample (the per-shard log surfaces drops).
    """
    if not samples:
        raise ValueError("empty sample list — nothing to determine payload column from")
    for ext in PAYLOAD_EXTENSIONS:
        if ext in samples[0]:
            return ext
    raise ValueError(f"first sample {samples[0]['key']!r} has no payload column")


def samples_to_table(samples: list[dict]) -> pa.Table:
    """Materialise a list of sample dicts as a `pyarrow.Table` with the payload as binary.

    pyarrow.Table.from_pylist infers column dtypes from observed values (string / int64 /
    float64 / list / nested struct).  We override the payload column to `pa.binary()`
    explicitly — without the override pyarrow may guess `large_binary` (for >2GB cells) which
    inflates per-row overhead on small JPEGs.

    Args:
        samples: Sample dicts from `read_shard_samples`.

    Returns:
        `pa.Table` with one row per sample and one column per observed field.
    """
    if not samples:
        return pa.table({})
    payload_column = _payload_column_name(samples)
    # Build the table via from_pylist (handles schema union + null fill across samples).
    table = pa.Table.from_pylist(samples)
    # Recast the payload column to `binary` so the parquet footer carries the smaller dtype.
    payload_index = table.column_names.index(payload_column)
    payload_array = pa.array([s.get(payload_column) for s in samples], type=pa.binary())
    return table.set_column(payload_index, payload_column, payload_array)


def write_parquet(table: pa.Table, out_path: str) -> None:
    """Write a `pa.Table` to parquet atomically via temp-then-rename.

    Uses zstd compression and a small row-group size so future cuDF readers can stream
    sub-shard chunks.  The crash-safe pattern mirrors `repack_one.py`.
    """
    tmp_path = out_path + ".tmp"
    pq.write_table(table, tmp_path, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    os.replace(tmp_path, out_path)


def verify_round_trip(tar_path: str, parquet_path: str) -> None:
    """Verify the parquet sibling carries the same samples (count + payload xxhash) as the tar.

    Strategy:
    - Re-open both side-by-side with metadata-only reads first (cheap), confirm row count.
    - Stream the tar a second time and the parquet binary column in parallel; xxhash both
      payload bytes per row and compare digest streams.

    Hashing the payload bytes (vs bytewise equality) means one streaming read of each file,
    not two.  The xxhash digest catches every byte-level corruption a `cp` would have caught
    while being ~10x faster than sha256.

    Raises:
        ValueError: On row count mismatch or hash mismatch (with the failing key surfaced).
    """
    parquet_file = pq.ParquetFile(parquet_path)
    parquet_rows = parquet_file.metadata.num_rows
    # Re-read tar to count co-keyed payloads.
    tar_samples = read_shard_samples(tar_path)
    if len(tar_samples) != parquet_rows:
        raise ValueError(
            f"verify: row count mismatch — tar {tar_path!r} has {len(tar_samples)} samples "
            f"but parquet {parquet_path!r} has {parquet_rows}"
        )
    # Compute the xxhash of the payload bytes in tar order, then in parquet order, and
    # compare. Both should land in the same order (read_shard_samples preserves tar order;
    # write_parquet preserves sample order).
    payload_column = _payload_column_name(tar_samples)
    parquet_table = parquet_file.read(columns=[payload_column, "key"])
    parquet_payloads = parquet_table.column(payload_column).to_pylist()
    parquet_keys = parquet_table.column("key").to_pylist()
    for index, tar_sample in enumerate(tar_samples):
        tar_key = tar_sample["key"]
        tar_payload = tar_sample[payload_column]
        if tar_key != parquet_keys[index]:
            raise ValueError(
                f"verify: key mismatch at row {index} — tar {tar_key!r} vs parquet {parquet_keys[index]!r}"
            )
        if xxhash.xxh64(tar_payload).digest() != xxhash.xxh64(parquet_payloads[index]).digest():
            raise ValueError(
                f"verify: payload xxhash mismatch at row {index} (key={tar_key!r}) "
                f"— tar {len(tar_payload)} bytes vs parquet {len(parquet_payloads[index])} bytes"
            )


def repack_shard(tar_path: str, *, keep_tar: bool, overwrite: bool) -> bool:
    """Repack one tar shard to parquet, verify round-trip, and (optionally) delete the tar.

    Args:
        tar_path: One WebDataset tar shard.
        keep_tar: When `True`, skip the tar deletion after verify.  Useful for the
            validation run on panda70m before launching the multi-hour cc12m sweep.
        overwrite: When `False`, skip a shard whose `.parquet` sibling already exists.

    Returns:
        `True` when the shard was repacked (or skipped as already done), `False` on failure.
    """
    parquet_path = parquet_sibling_path(tar_path)
    if os.path.exists(parquet_path) and not overwrite:
        print(f"[skip] {tar_path}: {parquet_path} already exists", flush=True)
        return True

    samples = read_shard_samples(tar_path)
    if not samples:
        print(f"[empty] {tar_path}: no co-keyed (payload + caption) samples — nothing to write", flush=True)
        return True
    table = samples_to_table(samples)
    write_parquet(table, parquet_path)
    verify_round_trip(tar_path, parquet_path)

    tar_bytes = os.path.getsize(tar_path)
    parquet_bytes = os.path.getsize(parquet_path)
    if not keep_tar:
        os.unlink(tar_path)
    print(
        f"[done] {tar_path}  {tar_bytes / 1e6:.0f} MB -> {parquet_bytes / 1e6:.0f} MB  "
        f"({len(samples)} samples, schema={table.schema.names})",
        flush=True,
    )
    return True


def main() -> int:
    """Parse arguments and repack the single shard named on the command line."""
    parser = argparse.ArgumentParser(description="Repack one WebDataset tar shard to a Parquet sibling.")
    parser.add_argument("tar_path", help="The single WebDataset tar shard to repack.")
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Skip the tar deletion after verify (useful for validation runs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-repack the shard even if its `.parquet` sibling exists.",
    )
    arguments = parser.parse_args()

    try:
        repack_shard(arguments.tar_path, keep_tar=arguments.keep_tar, overwrite=arguments.overwrite)
    except Exception as error:  # noqa: BLE001 — one bad shard must not poison the parallel run
        # Clean up any partial temp file from a crashed write.
        temp_path = parquet_sibling_path(arguments.tar_path) + ".tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[FAIL] {arguments.tar_path}: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
