# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download DataComp-small: 12.8M CLIP-filtered (image URL, caption) metadata parquet shards.

The mirror ships the URL-metadata parquet only (`url` + `text` + filtering scores); the image crawl
over those URLs is a deferred follow-up, so acquisition passes the URL index through unchanged.
"""

from acquire import acquire

if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "datacomp_small",
            repo_id="mlfoundations/datacomp_small",
            shard_suffix=(".parquet",),
            source_kind="url_index",
            out_name="datacomp-small",
        )
    )
