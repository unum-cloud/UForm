"""Download CC-12M: 12M Common-Crawl (image, caption) WebDataset tar shards -> canonical parquet.

File: scripts/data/download_cc12m.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "orjson",
#     "pyarrow>=21.0.0",
#     "xxhash",
# ]
# ///

import pyarrow as pa
from acquire import acquire


def _transform(table: pa.Table) -> pa.Table:
    """Select the canonical `caption` + `jpg` payload, retaining `key` for the round-trip verify.

    The muxer flattens the JSON sidecar into a `caption` column; a handful of WebDataset shards carry
    the caption in a `.txt` sidecar instead, surfaced as `txt` — fall back to it so no row loses its
    anchor. `jpg` (the payload) and `key` keep their names because `verify_round_trip` reads them back.
    """
    caption = "caption" if "caption" in table.column_names else "txt"
    return pa.table({"caption": table.column(caption), "jpg": table.column("jpg"), "key": table.column("key")})


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "cc12m",
            repo_id="pixparse/cc12m-wds",
            shard_suffix=(".tar",),
            source_kind="webdataset",
            out_name="cc12m",
            transform=_transform,
        )
    )
