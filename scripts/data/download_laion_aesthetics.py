# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download LAION-aesthetics: the high-aesthetic slice of reLAION-2B as a URL-index parquet, passthrough.

The `laion/relaion2B-multi-aesthetic` repo ships metadata + image URLs only (no image bytes); the
media crawl over those URLs is a deferred next step, so `url_index` normalizes the URL parquet exactly
like a snapshot.
"""

from acquire import acquire

if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "laion_aesthetics",
            repo_id="laion/relaion2B-multi-aesthetic",
            shard_suffix=(".parquet",),
            source_kind="url_index",
            out_name="laion-aesthetics",
        )
    )
