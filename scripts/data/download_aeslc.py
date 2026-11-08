"""Download AESLC: Annotated Enron Subject Line Corpus — `(email_body, subject_line)` pairs.

File: scripts/data/download_aeslc.py
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
    # Already flat `(email_body, subject_line)` parquet (`data/train-*.parquet`); no
    # repack — passthrough.
    raise SystemExit(
        acquire(
            "aeslc",
            repo_id="aeslc",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="aeslc",
        )
    )
