# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download XLCoST: cross-lingual NL<->code pairs (`text` anchor / `code` positive)."""
from acquire import acquire

if __name__ == "__main__":
    # The loader-script main branch is rejected, so pull the auto-converted parquet
    # (`<lang>-program-level/train/*.parquet`, cols `text`/`code`) — both columns already present.
    raise SystemExit(
        acquire(
            "xlcost",
            repo_id="codeparrot/xlcost-text-to-code",
            revision="refs/convert/parquet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="xlcost",
        )
    )
