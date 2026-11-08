"""Download S2ORC: scientific abstract<->citation pairs (`abstract` / `citation`) for text_contrastive.

File: scripts/data/download_s2orc.py
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
    # `abstract-citation-pair/train-*.parquet` already carries `abstract` + `citation` — pure
    # passthrough, and that config sorts first among the repo's parquet. Cap with `--max-shards 185`
    # (or `--fraction`) to stay inside it: the later `title-abstract-pair` / `title-citation-pair`
    # configs lack the `citation` column, so acquire's column-assert (correctly) rejects them past
    # the abstract shards.
    raise SystemExit(
        acquire(
            "s2orc",
            repo_id="sentence-transformers/s2orc",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="s2orc",
        )
    )
