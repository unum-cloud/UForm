"""Download AudioCaps: 50k (caption, audio) clips already carrying `caption` + `audio` parquet columns.

File: scripts/data/download_audiocaps.py
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
    raise SystemExit(
        acquire(
            "audiocaps",
            repo_id="OpenSound/audiocaps",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="audiocaps",
        )
    )
