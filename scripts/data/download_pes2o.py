"""Download peS2o: AI2 cleaned full-text scientific corpus (MLM `text`).

File: scripts/data/download_pes2o.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cudf-cu12",
#     "huggingface_hub",
#     "pandas",
#     "pyarrow>=21.0.0",
#     "zstandard",
# ]
# ///
from acquire import acquire

if __name__ == "__main__":
    # The v3 train split ships as zstd JSON-lines shards (`data/v3/train-*.zst`), one document
    # per line with a `text` field; the text_jsonl zstd path is schema-preserving, so no
    # transform is needed.
    raise SystemExit(
        acquire(
            "pes2o",
            repo_id="allenai/peS2o",
            revision="main",
            shard_suffix=(".zst",),
            source_kind="text_jsonl",
            out_name="peS2o",
        )
    )
