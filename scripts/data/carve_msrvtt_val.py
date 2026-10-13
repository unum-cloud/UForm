# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=21.0.0",
# ]
# ///
r"""Carve the standard MSR-VTT 1K test split out of the train_9k parquet.

MSR-VTT ships a canonical held-out test set of 1,000 video clips defined in
`msrvtt_test_1k.json` (video IDs video7010..video9999, the last 3K of the 10K
corpus, consistent with the JSFusion / HGR / Frozen-in-Time evaluation
protocol).  The `train_9k.parquet` repacked shard covers the 9K training
split; the 1K test set is a *different* (non-overlapping) partition.

This script:
1. Reads `msrvtt_test_1k.json` to get the canonical 1K video ID list.
2. Reads the train_9k parquet (dry-run safe with ``--limit``).
3. Filters to the rows whose ``video_id`` appears in the 1K list.
4. Writes the result to ``data/msrvtt-captions/test_1k.parquet`` (default).

NOTE: ``train_9k.parquet`` covers video0..video6999 + (parts of 7K split).
The canonical 1K test clips (video7010..video9999) are NOT in that shard —
they were repacked separately and the standard protocol evaluates on video IDs
listed in ``msrvtt_test_1k.json``.  To repack the test videos, call
``repack_msrvtt_to_parquet`` with ``--json msrvtt_test_1k.json``:

    uv run scripts/data/repack_msrvtt_to_parquet.py \
        --json  data/msrvtt-captions/msrvtt_test_1k.json \
        --zip   data/msrvtt-captions/MSRVTT_Videos.zip \
        --out   data/msrvtt-captions/test_1k.parquet

This carve script is the complementary tool: if ``test_1k.parquet`` is already
packed it subsets it by the JSON ID list (useful for verifying completeness or
producing a smaller held-out slice for quick smoke eval).

CLI usage:
    # Dry-run / schema check (no GPU, no write):
    uv run scripts/data/carve_msrvtt_val.py --dry_run
    # Carve with row limit (for testing the script itself):
    uv run scripts/data/carve_msrvtt_val.py --limit 50 --out /tmp/msrvtt_micro.parquet
    # Full carve from the already-repacked test parquet:
    uv run scripts/data/carve_msrvtt_val.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Constants: all paths relative to the project root (CWD when invoked via uv).
_DEFAULT_TEST_JSON = Path("data/msrvtt-captions/msrvtt_test_1k.json")
_DEFAULT_SOURCE_PARQUET = Path("data/msrvtt-captions/train_9k.parquet")
_DEFAULT_OUT_PARQUET = Path("data/msrvtt-captions/test_1k.parquet")


def _load_test_video_ids(json_path: Path) -> set[str]:
    """Load the canonical 1K test video IDs from the MSR-VTT JSON sidecar.

    Args:
        json_path: Path to ``msrvtt_test_1k.json``.

    Returns:
        Set of ``video_id`` strings (e.g. ``{"video7020", ...}``).
    """
    with open(json_path) as handle:
        entries = json.load(handle)
    ids = {entry["video_id"] for entry in entries if "video_id" in entry}
    print(f"carve_msrvtt_val: loaded {len(ids)} test video IDs from {json_path.name}")
    return ids


def carve(
    source_parquet: Path,
    test_json: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Carve the MSR-VTT 1K test split from a repacked parquet shard.

    The function reads the source parquet (optionally capped to ``limit`` rows),
    filters to rows whose ``video_id`` is in the canonical test JSON, and writes
    the result.

    Args:
        source_parquet: Path to the repacked parquet (e.g. ``train_9k.parquet``
            or a separately repacked ``test_1k.parquet``).
        test_json: Path to ``msrvtt_test_1k.json`` defining the 1K video ID list.
        out_path: Destination parquet path.
        limit: If set, read at most this many rows from ``source_parquet`` before
            filtering.  Useful for schema validation without scanning the full 1.9 GB
            shard.
        dry_run: When ``True``, print schema and match count but do NOT write.
        force: Overwrite ``out_path`` if it already exists.

    Returns:
        0 on success, 1 on error.
    """
    try:
        import pyarrow as pa  # noqa: PLC0415 — try/except ImportError guard: pyarrow is a standalone-script dep (not a project-wide dep); absence prints a user-friendly error
        import pyarrow.parquet as pq  # noqa: PLC0415 — try/except ImportError guard: grouped with pyarrow above
    except ImportError:
        print("carve_msrvtt_val: pyarrow is required — run `uv sync` first.", file=sys.stderr)
        return 1

    if not source_parquet.exists():
        print(f"carve_msrvtt_val: source parquet not found: {source_parquet}", file=sys.stderr)
        print(
            "  Hint: repack first with:\n"
            "    uv run scripts/data/repack_msrvtt_to_parquet.py \\\n"
            "        --json  data/msrvtt-captions/msrvtt_test_1k.json \\\n"
            "        --zip   data/msrvtt-captions/MSRVTT_Videos.zip \\\n"
            "        --out   data/msrvtt-captions/test_1k.parquet",
            file=sys.stderr,
        )
        return 1

    if not test_json.exists():
        print(f"carve_msrvtt_val: test JSON not found: {test_json}", file=sys.stderr)
        return 1

    if out_path.exists() and not force and not dry_run:
        print(f"carve_msrvtt_val: {out_path} already exists; skip (pass --force to overwrite).")
        return 0

    test_ids = _load_test_video_ids(test_json)

    # Read — cap to `limit` rows when given.
    pf = pq.ParquetFile(source_parquet)
    print(f"carve_msrvtt_val: source schema: {pf.schema_arrow}")
    print(f"carve_msrvtt_val: source rows:   {pf.metadata.num_rows}")

    # Stream row-group batches and collect matching rows.
    matched_batches: list[pa.RecordBatch] = []
    total_read = 0
    for batch in pf.iter_batches(batch_size=256):
        if limit is not None and total_read >= limit:
            break
        if limit is not None:
            batch = batch.slice(0, min(256, limit - total_read))
        total_read += len(batch)
        vid_ids = batch.column("video_id")
        mask = pa.array(
            [str(v.as_py()) in test_ids for v in vid_ids],
            type=pa.bool_(),
        )
        filtered = batch.filter(mask)
        if len(filtered):
            matched_batches.append(filtered)

    total_matched = sum(len(b) for b in matched_batches)
    print(f"carve_msrvtt_val: read {total_read} rows, matched {total_matched} test IDs")

    if dry_run:
        print("carve_msrvtt_val: --dry_run set; not writing output.")
        return 0

    if not matched_batches:
        print(
            "carve_msrvtt_val: 0 rows matched — the source parquet likely covers a different ID range.\n"
            "  The train_9k.parquet covers video0..video6999; the 1K test clips "
            "(video7010..video9999) must be repacked from MSRVTT_Videos.zip using "
            "repack_msrvtt_to_parquet with --json msrvtt_test_1k.json.",
            file=sys.stderr,
        )
        return 1

    table = pa.Table.from_batches(matched_batches)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp"
    pq.write_table(table, tmp_path, compression="zstd", row_group_size=512)
    os.replace(tmp_path, str(out_path))
    size_mb = out_path.stat().st_size / (1 << 20)
    print(f"carve_msrvtt_val: wrote {out_path} — {len(table)} rows, {size_mb:.1f} MB")
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carve MSR-VTT 1K test split from a repacked parquet shard.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE_PARQUET,
        help=f"Source parquet (default: {_DEFAULT_SOURCE_PARQUET}).",
    )
    parser.add_argument(
        "--test_json",
        type=Path,
        default=_DEFAULT_TEST_JSON,
        help=f"Canonical 1K test JSON (default: {_DEFAULT_TEST_JSON}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT_PARQUET,
        help=f"Output parquet (default: {_DEFAULT_OUT_PARQUET}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap source rows read (for smoke / schema testing, e.g. --limit 100).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print schema and match count without writing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(carve(args.source, args.test_json, args.out, limit=args.limit, dry_run=args.dry_run, force=args.force))
