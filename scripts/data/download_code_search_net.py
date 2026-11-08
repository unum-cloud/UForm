"""Download CodeSearchNet: function bodies (`func_code_string`) for the code share of the text_masked mixture.

File: scripts/data/download_code_search_net.py
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
    # Every config (`all/`, `python/`, `java/`, ...) ships `func_code_string` — pure passthrough.
    # The text_masked mixture reads only the `all/train-*.parquet` split; the per-language
    # configs ride along.
    raise SystemExit(
        acquire(
            "code_search_net_text",
            repo_id="code-search-net/code_search_net",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="code_search_net",
        )
    )
