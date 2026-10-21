# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download FineWiki: 325-language Wikipedia MLM corpus (`text`) used by text_masked."""
from acquire import acquire

if __name__ == "__main__":
    # Per-language `data/<code>wiki/*.parquet` shards already carry a `text` column — pure passthrough.
    raise SystemExit(
        acquire(
            "finewiki_i18n",
            repo_id="HuggingFaceFW/finewiki",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="finewiki-100bt",
        )
    )
