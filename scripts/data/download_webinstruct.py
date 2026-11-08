"""Download WebInstructFull: 11M mined instruction-response pairs; native (question, answer) parquet.

The upstream `TIGER-Lab/WebInstructFull` ships flat `data/train-*.parquet` already carrying the
`question` / `answer` columns the `webinstruct` mixture entry declares, so this is a passthrough
`hf_snapshot` — no transform. (The parquet source means the acquire kind is `hf_snapshot`, not the
`text_jsonl` a jsonl corpus would use.)

File: scripts/data/download_webinstruct.py
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
    raise SystemExit(
        acquire(
            "webinstruct",
            repo_id="TIGER-Lab/WebInstructFull",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="webinstruct",
        )
    )
