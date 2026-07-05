# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download WebVid-10M: ~10M short (video, text) clips shipped inline as parquet -> canonical parquet."""

from acquire import acquire

if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "webvid_10m",
            repo_id="jxie/webvid_10m",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="webvid-10m",
        )
    )
