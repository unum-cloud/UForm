"""Download LAION CC-3M: ~3M Common-Crawl (image, caption) WebDataset tar shards -> canonical parquet.

File: scripts/data/download_laion_cc3m.py
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

    Same schema as CC-12M: the caption rides the JSON sidecar (`caption`), falling back to the `.txt`
    sidecar (`txt`); `jpg` and `key` stay named so `verify_round_trip` can read them from the parquet.
    """
    caption = "caption" if "caption" in table.column_names else "txt"
    return pa.table({"caption": table.column(caption), "jpg": table.column("jpg"), "key": table.column("key")})


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "laion_cc3m",
            repo_id="pixparse/cc3m-wds",
            shard_suffix=(".tar",),
            source_kind="webdataset",
            out_name="laion-cc3m",
            transform=_transform,
        )
    )
