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

Folds the four per-source-kind fetch engines (`hf_snapshot` / `webdataset` / `text_jsonl` / the
`url_index` seam of media-fetch) into a single `acquire(...)`. A downloader is now a one-liner:

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
from collections.abc import Callable
from pathlib import Path

import mixtures
import pyarrow as pa
import pyarrow.parquet as parquet
from engine_base import artifact_path, cap_shards
from huggingface_hub import list_repo_files, snapshot_download
from sharding import already_done, atomic_write, partition, resolve_rank_world

# The repack cores (`repack_one` pulls cuDF at import; `repack_webdataset` pulls orjson/xxhash) are imported
# lazily inside the source-kind branch that needs them, so a passthrough hf_snapshot downloader that declares
# only huggingface_hub + pyarrow can `import acquire` without those heavier deps in its uv env.

#: Source kinds whose shards enumerate as sorted repo files filtered by suffix (a HuggingFace dataset repo).
#: `url_index` normalizes its URL-index parquet exactly like a snapshot (the media crawl over the URLs is a
#: deferred next step); `youtube_fetch` still needs the per-dataset index-column names and is unimplemented.
_repo_source_kinds = ("hf_snapshot", "webdataset", "text_jsonl", "url_index")


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
    transform: Callable | None,
) -> set[str]:
    """Pull one source shard, apply `transform`, commit one canonical parquet, and return its column names.

    `hf_snapshot` / `url_index` read the shard's parquet directly; `webdataset` muxes the tar and verifies
    the xxhash round trip; `text_jsonl` takes repack_one's dual path (streaming CPU zstd vs the cuDF GPU
    read). The returned column set feeds the post-transform column assert.
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
    transform: Callable[[pa.Table], pa.Table] | None = None,
) -> int:
    """Acquire and normalize one dataset to canonical parquet, one artifact per source shard.

    @param[in] name The catalog / mixture dataset name; drives the column assert.
    @param[in] repo_id The upstream HuggingFace dataset repo id.
    @param[in] revision The repo revision / branch (e.g. `refs/convert/parquet`).
    @param[in] shard_suffix Shard-file suffixes to enumerate (a tuple `str.endswith` accepts).
    @param[in] source_kind One of hf_snapshot / webdataset / text_jsonl / url_index / youtube_fetch.
    @param[in] out_name The `data/<out_name>` output subdirectory and artifact prefix.
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
            transform=transform,
        )
        _assert_columns(name, produced_columns)
        written += 1

    print(f"acquire {name}: rank {rank}/{world} wrote {written} shards -> {out_dir}")
    return 0
