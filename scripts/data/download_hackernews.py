# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download Hacker News: item corpus whose comment/story `text` feeds the text_base MLM blend."""
from acquire import acquire

if __name__ == "__main__":
    # The `data/train-*.parquet` shards already carry a `text` column (the story / comment body) — pure
    # passthrough; the other item fields (title, url, score, ...) ride along unused.
    raise SystemExit(
        acquire(
            "hackernews_text",
            repo_id="OpenPipe/hacker-news",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="hackernews",
        )
    )
