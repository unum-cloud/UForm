"""Download Panda-70M testing split: ~70k short (video, caption) WebDataset tar shards -> parquet.

File: scripts/data/download_panda70m_wds.py
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
    """Select the canonical `caption` + `mp4` payload, retaining `key` for the round-trip verify.

    The muxer flattens the JSON sidecar (`caption` / `duration` / `timestamp` / `matching_score`); we
    keep only the `caption` anchor and the `mp4` payload, plus `key` because `verify_round_trip` reads
    the payload column and `key` back from the produced parquet.
    """
    return table.select(["caption", "mp4", "key"])


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "panda70m_testing",
            repo_id="Qinghao/panda70m_wds",
            shard_suffix=(".tar",),
            source_kind="webdataset",
            out_name="panda70m-wds",
            transform=_transform,
        )
    )
