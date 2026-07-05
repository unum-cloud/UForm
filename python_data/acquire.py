# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cudf-cu12",
#     "huggingface_hub",
#     "orjson",
#     "pandas",
#     "pyarrow>=21.0.0",
#     "xxhash",
#     "yt-dlp",
#     "zstandard",
# ]
# ///
"""Shared acquisition helper the thin per-dataset downloaders call — one function, one dataset to parquet.

Folds the per-source-kind fetch engines (`hf_snapshot` / `webdataset` / `text_jsonl` / the `url_index`
metadata pass-through / the `url_crawl` image fetch over those URLs) into a single `acquire(...)`. A
downloader is now a one-liner:

    from acquire import acquire
    raise SystemExit(acquire("coco_captions", repo_id="HuggingFaceM4/COCO",
                             revision="refs/convert/parquet", source_kind="hf_snapshot",
                             out_name="coco_captions"))

The per-dataset variance that used to live on a catalog row is now explicit parameters (`repo_id`,
`revision`, `shard_suffix`, `source_kind`, `out_name`) plus an optional `transform` — a
`pyarrow.Table | cudf.DataFrame -> same` reshape that replaces the old declarative `column_map`.

The flow is the same for every dataset and single-threaded (parallelism is one `uv run` per Slurm array
task, resolved through `sharding`): parse the uniform CLI, enumerate the sorted upstream shards, cap by
fraction/max (composing, tighter wins), take this rank's disjoint stride, and per assigned shard pull it,
apply `transform`, and commit one canonical parquet through the atomic rename — skipping shards already
committed. The one addition over the old engines is the column ASSERT: after `transform`, every column the
mixture JSONs declare for this dataset (`mixtures.dataset_columns`) must be present in the produced shard,
so a broken reshape fails at acquisition instead of silently mistraining. Enumeration, the tar-mux, the
zstd/cuDF repack, and the round-trip verify are reused verbatim from the repack cores.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import mixtures
import pyarrow as pa
import pyarrow.compute as compute
import pyarrow.parquet as parquet
from engine_base import artifact_path, cap_shards
from huggingface_hub import list_repo_files, snapshot_download
from sharding import already_done, atomic_write, partition, resolve_rank_world

# The repack cores (`repack_one` pulls cuDF at import; `repack_webdataset` pulls orjson/xxhash) are imported
# lazily inside the source-kind branch that needs them, so a passthrough hf_snapshot downloader that declares
# only huggingface_hub + pyarrow can `import acquire` without those heavier deps in its uv env.

#: Source kinds whose shards enumerate as sorted repo files filtered by suffix (a HuggingFace dataset repo).
#: `url_index` normalizes its URL-index parquet exactly like a snapshot (a metadata pass-through); `url_crawl`
#: enumerates the same index parquet but then fetches the images those URLs point at (see `_crawl_url_index`);
#: `youtube_fetch` still needs the per-dataset index-column names and is unimplemented.
_repo_source_kinds = ("hf_snapshot", "webdataset", "text_jsonl", "url_index", "url_crawl")


def _data_root() -> Path:
    """The acquisition output root: `$UFORM_DATA_ROOT`, else the repo's `data/` directory."""
    return Path(os.environ.get("UFORM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))


def _read_compression(shard_name: str) -> str | None:
    """The cuDF `read_json` compression for a JSON-lines shard, by longest matching JSONL extension."""
    from repack_one import COMPRESSION_OF_EXTENSION, JSONL_EXTENSIONS  # lazy: pulls cuDF

    for extension in sorted(JSONL_EXTENSIONS, key=len, reverse=True):
        if shard_name.endswith(extension):
            return COMPRESSION_OF_EXTENSION[extension]
    return None


def _enumerate_shards(source_kind: str, repo_id: str | None, revision: str, shard_suffix: tuple[str, ...]) -> list[str]:
    """Sorted upstream shard ids for the dataset — the stable order the rank partition slices.

    Every repo-backed kind enumerates identically (`list_repo_files` filtered by `shard_suffix`); metadata
    files (README, `.gitattributes`) fall out because they do not end in a shard suffix.
    """
    if source_kind == "youtube_fetch":
        raise NotImplementedError(
            "youtube_fetch reuses fetch_media_clips over each index shard's (id, window, caption) rows; it "
            "needs the per-dataset index-column names and is wired in a follow-up — until then the youtube "
            "download_*.py scripts remain the acquisition path."
        )
    if source_kind not in _repo_source_kinds:
        raise ValueError(f"acquire {source_kind!r}: unknown source_kind; expected one of {_repo_source_kinds}")
    if repo_id is None:
        raise ValueError(f"acquire: source_kind {source_kind!r} needs a repo_id to enumerate shards")
    files = list(list_repo_files(repo_id, repo_type="dataset", revision=revision))
    return sorted(name for name in files if name.endswith(shard_suffix))


#: img2dataset writes the fetched image bytes under `jpg` and the passed-through caption under `caption`, the
#: same pair the WebDataset image sets (cc12m / laion_cc3m) commit — so a crawled shard is schema-identical to
#: a muxed one and drops straight into the `image_caption` mixture (`anchor=caption`, `positive=media=jpg`).
_CRAWL_IMAGE_COLUMN = "jpg"
_CRAWL_CAPTION_COLUMN = "caption"


def _crawl_url_index(index_shard: Path, crawl_dir: Path, *, url_column: str, caption_column: str) -> pa.Table:
    """Fetch the images one URL-index parquet shard points at and return the canonical `{jpg, caption}` table.

    The crawl itself is delegated to `img2dataset` — the LAION/DataComp downloader that owns the concurrency,
    retries, timeouts, content-type checks, and resize/re-encode we must not reimplement with the stylebook-
    banned Python threads. It is invoked as a subprocess (like the repackers' GNU-parallel fan-out), fed this
    shard's `url_column` + `caption_column`, and told to emit parquet into a rank-private scratch directory.

    Its parquet output carries a row per input URL — `jpg` holds the re-encoded bytes on success and null on a
    dead link (`status != "success"`) — plus the `caption` it passed through. Web crawls yield partially, so we
    keep only the rows that actually downloaded (`jpg` non-null) and project to the two canonical columns; an
    all-dead shard collapses to a valid empty parquet, which the atomic-rename skip still records as done.

    @param[in] index_shard The already-snapshotted URL-index parquet (the img2dataset `--url_list`).
    @param[in] crawl_dir A fresh rank-private directory for img2dataset's output shards (created by the tool).
    @param[in] url_column The index column holding the image URL to fetch.
    @param[in] caption_column The index column holding the caption to carry onto each fetched image.
    @return A `pyarrow.Table` of the successfully fetched rows with columns `jpg` (image bytes) and `caption`.
    """
    img2dataset = shutil.which("img2dataset") or str(Path(sys.executable).with_name("img2dataset"))
    command = [
        img2dataset,
        "--url_list", str(index_shard),
        "--input_format", "parquet",
        "--url_col", url_column,
        "--caption_col", caption_column,
        "--output_folder", str(crawl_dir),
        "--output_format", "parquet",
        "--processes_count", "1",  # concurrency is img2dataset's threads, not ours; one worker per shard
        "--thread_count", "64",
        "--image_size", "256",
        "--resize_mode", "keep_ratio",
        "--encode_format", "jpg",
        "--encode_quality", "90",
        "--retries", "2",
        "--timeout", "20",
        "--number_sample_per_shard", "100000",
        "--disallowed_header_directives", "[]",  # honor no robots opt-out header; crawl every reachable URL
    ]
    subprocess.run(command, check=True)  # non-zero only on bad args; partial download yield is a clean exit

    written = sorted(crawl_dir.glob("*.parquet"))  # one or more output shards; `*_stats.json` is ignored
    columns = [_CRAWL_IMAGE_COLUMN, _CRAWL_CAPTION_COLUMN]
    if not written:
        empty = {_CRAWL_IMAGE_COLUMN: pa.array([], pa.binary()), _CRAWL_CAPTION_COLUMN: pa.array([], pa.string())}
        return pa.table(empty)
    table = pa.concat_tables(parquet.read_table(path, columns=columns) for path in written)
    table = table.filter(compute.is_valid(table.column(_CRAWL_IMAGE_COLUMN)))  # drop the dead-link rows
    return pa.table({name: table.column(name) for name in columns})


def _fetch_one_shard(
    source: str,
    *,
    repo_id: str,
    revision: str,
    source_kind: str,
    out_name: str,
    out_dir: Path,
    final: Path,
    rank: int,
    url_column: str,
    caption_column: str,
    transform: Callable | None,
) -> set[str]:
    """Pull one source shard, apply `transform`, commit one canonical parquet, and return its column names.

    `hf_snapshot` / `url_index` read the shard's parquet directly; `webdataset` muxes the tar and verifies
    the xxhash round trip; `text_jsonl` takes repack_one's dual path (streaming CPU zstd vs the cuDF GPU
    read); `url_crawl` snapshots the URL-index parquet like `url_index` but then fetches the images those
    URLs point at (`_crawl_url_index`). The returned column set feeds the post-transform column assert.
    """
    scratch = out_dir / f".{source_kind}_raw" / f"rank{rank}"  # rank-private so disjoint pulls never collide
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(scratch),
        allow_patterns=[source],  # pull only this shard
    )
    shard = scratch / source
    try:
        if source_kind == "url_crawl":
            table = _crawl_url_index(shard, scratch / "crawl", url_column=url_column, caption_column=caption_column)
            table = transform(table) if transform is not None else table
            with atomic_write(final, rank=rank) as temporary:
                parquet.write_table(table, temporary, compression="zstd")
            return set(table.column_names)

        if source_kind == "webdataset":
            from repack_webdataset_to_parquet import (  # lazy: pulls orjson/xxhash
                ROW_GROUP_SIZE,
                read_shard_samples,
                samples_to_table,
                verify_round_trip,
            )

            table = samples_to_table(read_shard_samples(str(shard)))
            table = transform(table) if transform is not None else table
            with atomic_write(final, rank=rank) as temporary:
                parquet.write_table(table, temporary, compression="zstd", row_group_size=ROW_GROUP_SIZE)
            verify_round_trip(str(shard), str(final))  # xxhash media-size parity before we drop the tar
            return set(table.column_names)

        if source_kind == "text_jsonl":
            from repack_one import repack_zstd_via_cpu_chunks  # lazy: pulls cuDF

            compression = _read_compression(source)
            if compression == "zstd":
                # zstd shards decode on the CPU in bounded-memory chunks (the GPU reader OOMs on large
                # frames and rejects headerless ones); this path is schema-preserving, so no transform.
                with atomic_write(final, rank=rank) as temporary:
                    repack_zstd_via_cpu_chunks(str(shard), str(temporary), "zstd")
                return set(parquet.read_schema(final).names)
            import cudf  # noqa: PLC0415 — GPGPU dep; imported lazily so non-cuDF source kinds stay light

            frame = cudf.read_json(str(shard), lines=True, compression=compression)  # cuDF GPU path
            frame = transform(frame) if transform is not None else frame
            columns = {str(column) for column in frame.columns}
            with atomic_write(final, rank=rank) as temporary:
                frame.to_parquet(str(temporary), compression="zstd")
            return columns

        # hf_snapshot / url_index: the shard is already canonical parquet (the URL index is the artifact).
        table = parquet.read_table(shard)
        table = transform(table) if transform is not None else table
        with atomic_write(final, rank=rank) as temporary:
            parquet.write_table(table, temporary, compression="zstd")
        return set(table.column_names)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _assert_columns(name: str, produced_columns: set[str]) -> None:
    """Every column the mixture JSONs declare for `name` must be present in the produced shard.

    Vacuous — and skipped — when the dataset is in no mixture (`dataset_columns` empty, e.g. an eval probe).
    """
    declared = mixtures.dataset_columns(name)
    missing = declared - produced_columns
    if missing:
        raise AssertionError(
            f"acquire {name!r}: mixture-declared columns missing from the produced shard — "
            f"missing={sorted(missing)}; declared={sorted(declared)}; produced={sorted(produced_columns)}. "
            "Fix the transform or the mixture JSON so the column contract holds."
        )


def acquire(
    name: str,
    *,
    repo_id: str | None,
    revision: str = "main",
    shard_suffix: tuple[str, ...] = (".parquet",),
    source_kind: str,
    out_name: str,
    url_column: str = "url",
    caption_column: str = "caption",
    transform: Callable[[pa.Table], pa.Table] | None = None,
) -> int:
    """Acquire and normalize one dataset to canonical parquet, one artifact per source shard.

    @param[in] name The catalog / mixture dataset name; drives the column assert.
    @param[in] repo_id The upstream HuggingFace dataset repo id.
    @param[in] revision The repo revision / branch (e.g. `refs/convert/parquet`).
    @param[in] shard_suffix Shard-file suffixes to enumerate (a tuple `str.endswith` accepts).
    @param[in] source_kind One of hf_snapshot / webdataset / text_jsonl / url_index / url_crawl / youtube_fetch.
    @param[in] out_name The `data/<out_name>` output subdirectory and artifact prefix.
    @param[in] url_column For `url_crawl`: the index column holding the image URL to fetch (default `url`).
    @param[in] caption_column For `url_crawl`: the index column holding the caption to carry (default `caption`).
    @param[in] transform Optional per-dataset reshape of each shard's `pyarrow.Table | cudf.DataFrame`.
    @return Process exit code (`0` on success).
    """
    parser = argparse.ArgumentParser(description=f"Acquire and normalize {name} to canonical parquet.")
    parser.add_argument("--fraction", type=float, default=None, help="leading-slice volume cap in (0, 1]")
    parser.add_argument("--max-shards", type=int, default=None, help="hard cap on the number of shards")
    parser.add_argument("--rank", type=int, default=None, help="usually env-derived; explicit for local N-way runs")
    parser.add_argument("--world", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-fetch shards whose artifact already exists")
    args = parser.parse_args()

    rank, world = resolve_rank_world(args.rank, args.world)
    out_dir = _data_root() / out_name

    shards = _enumerate_shards(source_kind, repo_id, revision, shard_suffix)  # sorted
    capped = cap_shards(shards, args.fraction, args.max_shards)
    mine = partition(capped, rank, world)  # this rank's disjoint stride of the capped set

    written = 0
    for source in mine:
        final = artifact_path(out_dir, out_name, source)
        if already_done(final) and not args.force:
            continue
        produced_columns = _fetch_one_shard(
            source,
            repo_id=repo_id,  # non-None: enumeration would have raised otherwise
            revision=revision,
            source_kind=source_kind,
            out_name=out_name,
            out_dir=out_dir,
            final=final,
            rank=rank,
            url_column=url_column,
            caption_column=caption_column,
            transform=transform,
        )
        _assert_columns(name, produced_columns)
        written += 1

    print(f"acquire {name}: rank {rank}/{world} wrote {written} shards -> {out_dir}")
    return 0
