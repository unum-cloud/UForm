"""Download ComPile: textual LLVM IR (`content`) — use --fraction / --max-shards (full corpus ~2.7 TB).

File: scripts/data/download_compile_ir.py
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
    # Already parquet on the hub (`*.parquet`); the IR text lives in `content` with license
    # provenance columns alongside — pure passthrough. The full corpus is ~2.7 TB, so cap the pull.
    raise SystemExit(
        acquire(
            "compile_ir",
            repo_id="llvm-ml/ComPile",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="compile-ir",
        )
    )
