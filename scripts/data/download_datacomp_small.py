# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "img2dataset",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download DataComp-small: crawl the 12.8M CLIP-filtered (image URL, caption) index into image parquet.

The `mlfoundations/datacomp_small` mirror ships only the URL metadata (`url` + `text` caption + CLIP
scores, no image bytes). `url_crawl` enumerates those index shards, then fetches the images each shard's
`url` points at with `img2dataset` and commits the canonical `{jpg, caption}` parquet — schema-identical
to the CC-12M / LAION-CC3M image sets, so it slots straight into the `image_contrastive` mixture once given a
training weight. Crawls yield partially (dead links drop out); the per-output-shard atomic rename still
records each finished shard, and the Slurm array (`acquire.sbatch --acquire-shardable`) fans the 26 index
shards across workers.
"""

from acquire import acquire

if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "datacomp_small",
            repo_id="mlfoundations/datacomp_small",
            shard_suffix=(".parquet",),
            source_kind="url_crawl",
            out_name="datacomp-small",
            url_column="url",
            caption_column="text",
        )
    )
