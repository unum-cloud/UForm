"""Download GitHub Typo Corpus: human typo edits (`text` anchor / `correction` positive).

File: scripts/data/download_github_typo.py
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
    # The upstream S3 bucket is decommissioned; the flat parquet mirror already carries
    # `(text, correction)` columns (the extracted typo edits) — pure passthrough.
    raise SystemExit(
        acquire(
            "github_typo",
            repo_id="chirunder/github_typo_corrections",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="github-typo-corpus",
        )
    )
