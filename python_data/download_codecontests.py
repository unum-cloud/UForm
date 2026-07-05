# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download DeepMind CodeContests: competitive-programming problems + solutions."""
from acquire import acquire

if __name__ == "__main__":
    # Already parquet on the hub main branch (`*.parquet`); no repack — pure passthrough.
    raise SystemExit(
        acquire(
            "code_contests",
            repo_id="deepmind/code_contests",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="code-contests",
        )
    )
