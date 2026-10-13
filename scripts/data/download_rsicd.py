# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download RSICD: ~11K remote-sensing images, five captions each — already in the image schema."""
from acquire import acquire

if __name__ == "__main__":
    # The `arampacha/rsicd` split shards already ship `image` struct + `captions` list, so the source is
    # canonical and needs no transform; all three splits (train / valid / test) are pulled by default.
    raise SystemExit(
        acquire(
            "rsicd",
            repo_id="arampacha/rsicd",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="rsicd",
        )
    )
