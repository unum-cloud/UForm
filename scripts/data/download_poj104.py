"""Download POJ-104: CodeXGLUE C++ clone-detection pairs (`label` anchor / `code` positive).

File: scripts/data/download_poj104.py
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
    # The loader-script main branch is rejected by modern `datasets`, so pull the
    # auto-converted parquet (`default/train/*.parquet`, cols `id`/`code`/`label`) —
    # `label`/`code` are already present.
    raise SystemExit(
        acquire(
            "poj_104",
            repo_id="google/code_x_glue_cc_clone_detection_poj104",
            revision="refs/convert/parquet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="poj-104",
        )
    )
