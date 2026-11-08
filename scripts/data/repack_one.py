"""Repack exactly one JSONL / JSONL.gz / JSONL.zst shard to a Parquet sibling.

Modality: text. The unit of work for the GNU `parallel` repack model: one process per shard,
so many shards convert concurrently without an in-process `Pool` (Python multiprocessing is
banned per the stylebook). The shard path is the single positional argument; the Parquet sibling
is written next to it via an atomic temporary-then-rename so an interrupted run never leaves a
footer-less, cuDF-unreadable Parquet that a later run would skip as "already done".

ZSTD shards (e.g. peS2o) can be large enough that a single-frame cuDF load OOMs the GPU, and a
CUDA out-of-memory corrupts the device context for the whole process; cuDF's GPU ZSTD reader also
rejects frames written without a content-size header. So ZSTD shards decode and repack entirely on
the CPU in bounded-memory line chunks. The `.jsonl` / `.gz` shards take the cuDF GPU path.

Run one process per shard under GNU `parallel`:

    ls data/<src>/*.jsonl.zst | parallel -j32 uv run scripts/data/repack_one.py {}

Pass `--compression snappy|none` to change the Parquet codec (default `zstd`); pass `--overwrite`
to re-repack a shard whose `.parquet` sibling already exists.

File: scripts/data/repack_one.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cudf-cu12",
#     "pandas",
#     "pyarrow>=21.0.0",
#     "zstandard",
# ]
# ///

import argparse
import io
import os
import sys

import cudf
import pandas
import pyarrow
import pyarrow.parquet
import zstandard

#: Source extensions cuDF reads slowly (row-based JSON parsing + non-nvCOMP decompression).
#: Order matters: multi-part extensions (`.jsonl.zst`) must precede the bare `.zst` so
#: `parquet_sibling_path` strips the longest match first. Bare `.zst` / `.gz` are JSON-lines
#: shards named without the `.jsonl` / `.json` infix (e.g. peS2o `train-*.zst`).
JSONL_EXTENSIONS = (".jsonl.gz", ".json.gz", ".jsonl.zst", ".json.zst", ".jsonl", ".zst", ".gz")

#: Map a JSONL extension to the cuDF `read_json` compression argument.
COMPRESSION_OF_EXTENSION = {
    ".jsonl": None,
    ".jsonl.gz": "gzip",
    ".json.gz": "gzip",
    ".gz": "gzip",
    ".jsonl.zst": "zstd",
    ".json.zst": "zstd",
    ".zst": "zstd",
}


def parquet_sibling_path(shard_path: str) -> str:
    """Return the `.parquet` path that replaces a shard's JSONL extension.

    Args:
        shard_path: A JSONL-family shard path ending in one of `JSONL_EXTENSIONS`.

    Returns:
        The sibling path with the JSONL extension replaced by `.parquet`.

    Raises:
        ValueError: If `shard_path` does not end in a recognized JSONL extension.
    """
    for extension in JSONL_EXTENSIONS:
        if shard_path.endswith(extension):
            return shard_path[: -len(extension)] + ".parquet"
    raise ValueError(f"not a recognized JSONL shard: {shard_path!r}")


def repack_zstd_via_cpu_chunks(
    shard_path: str, parquet_path: str, parquet_codec: str | None, chunk_lines: int = 50_000
) -> None:
    """Repack one ZSTD JSON-lines shard to Parquet entirely on the CPU, in bounded-memory chunks.

    Stream-decompresses with the Python `zstandard` library (no content-size header needed), reads
    the JSON-lines in `chunk_lines`-row batches with pandas, and writes each batch as one Parquet
    row group via pyarrow. Never loads the whole (multi-gigabyte) shard onto the GPU or into RAM.

    Args:
        shard_path: The ZSTD-compressed JSON-lines shard to read.
        parquet_path: The output Parquet path (typically a temporary that is renamed afterward).
        parquet_codec: The pyarrow Parquet compression codec, or `None` for uncompressed.
        chunk_lines: Number of JSON lines per pandas read batch and Parquet row group.
    """
    decompressor = zstandard.ZstdDecompressor()
    parquet_writer: pyarrow.parquet.ParquetWriter | None = None
    writer_schema: pyarrow.Schema | None = None
    try:
        with open(shard_path, "rb") as compressed, decompressor.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for chunk_frame in pandas.read_json(text_stream, lines=True, chunksize=chunk_lines):
                table = pyarrow.Table.from_pandas(chunk_frame, preserve_index=False)
                if parquet_writer is None:
                    writer_schema = table.schema
                    parquet_writer = pyarrow.parquet.ParquetWriter(
                        parquet_path, writer_schema, compression=(parquet_codec or "none")
                    )
                else:
                    table = table.cast(writer_schema)
                parquet_writer.write_table(table)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()


def repack_shard(shard_path: str, compression: str, overwrite: bool) -> bool:
    """Convert one JSONL shard to a Parquet sibling, atomically (temporary then rename).

    ZSTD shards take the CPU-chunked, GPU-free path; `.jsonl` / `.gz` shards take the cuDF GPU path.
    The atomic temporary-then-rename guarantees a concurrent reader never observes a partially
    written Parquet, and an interrupted run leaves the source untouched.

    Args:
        shard_path: The JSONL-family shard to convert.
        compression: Parquet codec name: `"zstd"`, `"snappy"`, or `"none"`.
        overwrite: When `False`, skip a shard whose `.parquet` sibling already exists.

    Returns:
        `True` when the shard was repacked, `False` when it was skipped as already done.
    """
    parquet_path = parquet_sibling_path(shard_path)
    if os.path.exists(parquet_path) and not overwrite:
        print(f"[skip] {shard_path}: {parquet_path} already exists", flush=True)
        return False

    extension = next(ext for ext in JSONL_EXTENSIONS if shard_path.endswith(ext))
    read_compression = COMPRESSION_OF_EXTENSION[extension]
    parquet_codec = None if compression == "none" else compression
    temporary_path = parquet_path + ".tmp"

    if read_compression == "zstd":
        repack_zstd_via_cpu_chunks(shard_path, temporary_path, parquet_codec)
    else:
        frame = cudf.read_json(shard_path, lines=True, compression=read_compression)
        frame.to_parquet(temporary_path, compression=parquet_codec)
    os.replace(temporary_path, parquet_path)

    source_bytes = os.path.getsize(shard_path)
    parquet_bytes = os.path.getsize(parquet_path)
    print(f"[done] {shard_path}  {source_bytes / 2**20:.0f} MB -> {parquet_bytes / 2**20:.0f} MB", flush=True)
    return True


def main() -> int:
    """Parse arguments and repack the single shard named on the command line.

    Returns:
        Process exit code: `0` on success or skip, `1` when the repack raised.
    """
    parser = argparse.ArgumentParser(description="Repack one JSONL shard to a Parquet sibling.")
    parser.add_argument("shard_path", help="The single JSONL-family shard to repack.")
    parser.add_argument(
        "--compression",
        choices=("zstd", "snappy", "none"),
        default="zstd",
        help="Parquet compression: zstd (smallest + ~20x read), snappy (~30x read), none.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-repack the shard even if its .parquet sibling exists."
    )
    arguments = parser.parse_args()

    try:
        repack_shard(arguments.shard_path, arguments.compression, arguments.overwrite)
    except Exception as error:  # noqa: BLE001 — one bad shard must not poison the parallel run
        temporary_path = parquet_sibling_path(arguments.shard_path) + ".tmp"
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        print(f"[FAIL] {arguments.shard_path}: {type(error).__name__}: {str(error)[:160]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
