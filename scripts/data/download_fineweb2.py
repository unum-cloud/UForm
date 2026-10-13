# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download FineWeb-2: multilingual (non-English) web MLM corpus (`text`)."""
from acquire import acquire

if __name__ == "__main__":
    # Per-language `data/<code>/train/*.parquet` shards already carry a `text` column — pure passthrough.
    raise SystemExit(
        acquire(
            "fineweb_2",
            repo_id="HuggingFaceFW/fineweb-2",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="fineweb-2",
        )
    )
