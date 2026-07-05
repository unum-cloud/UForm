# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download CodeSearchNet: function bodies (`func_code_string`) for the text_base code-MLM blend."""
from acquire import acquire

if __name__ == "__main__":
    # Every config (`all/`, `python/`, `java/`, ...) ships `func_code_string` — pure passthrough. The
    # text_base mixture reads only the `all/train-*.parquet` split; the per-language configs ride along.
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
