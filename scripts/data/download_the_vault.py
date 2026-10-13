# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download The Vault: multilingual docstring<->function pairs (`docstring` / `code`)."""
from acquire import acquire

if __name__ == "__main__":
    # The convert branch is corrupted, so pull the REAL parquet on main
    # (`data/train/<tier>/<lang>-*.parquet`) — `docstring`/`code` are already present, pure passthrough.
    raise SystemExit(
        acquire(
            "the_vault",
            repo_id="Fsoft-AIC/the-vault-function",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="the-vault",
        )
    )
