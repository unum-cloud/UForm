"""Download AudioSet: 2M labelled (audio, human_labels) clips as parquet -> canonical parquet.

The sorted shard order is `bal_train` then `eval` then `unbal_train`, so a `--max-shards` /
`--fraction` cap keeps the curated ~48 GB `bal_train` + `eval` slice first; an uncapped run pulls the
full ~2.4 TB corpus (the label -> comma-joined-caption reshape is a train-time extractor, not applied
here — acquisition preserves `human_labels` + `audio` verbatim).

File: scripts/data/download_audioset.py
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
            "audioset",
            repo_id="agkphysics/AudioSet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="audioset",
        )
    )
