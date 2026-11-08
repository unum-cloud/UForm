"""Download StarCoder data: BigCode 86-language code corpus (MLM `content`).

File: scripts/data/download_starcoderdata.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
from acquire import acquire

if __name__ == "__main__":
    # Per-language `<lang>/train-*.parquet` shards already carry a `content` column
    # — pure passthrough.
    raise SystemExit(
        acquire(
            "starcoderdata",
            repo_id="bigcode/starcoderdata",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="starcoderdata",
        )
    )
