# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download FineWeb: English web MLM corpus (`text`) — the 100BT-sample slice used by text_masked."""
from acquire import acquire

if __name__ == "__main__":
    # Every FineWeb parquet (`sample/100BT/*.parquet` and the `data/CC-MAIN-*` dumps) already carries a
    # `text` column — pure passthrough. The full repo is enormous, so bound the pull with `--fraction` /
    # `--max-shards`; the leading capped slice is the reproducible stand-in for the 100BT sample.
    raise SystemExit(
        acquire(
            "fineweb_en",
            repo_id="HuggingFaceFW/fineweb",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="fineweb-100bt",
        )
    )
