"""Download Flickr30k: ~31K images, five human captions each, into the coco_captions image schema.

File: scripts/data/download_flickr30k.py
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
import pyarrow as pa
from acquire import acquire


def _transform(table: pa.Table) -> pa.Table:
    """Pass the `image` struct through and rename the source `caption` list to `sentences_raw`."""
    return table.select(["image", "caption"]).rename_columns(["image", "sentences_raw"])


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "flickr30k",
            repo_id="nlphuji/flickr30k",
            revision="refs/convert/parquet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="flickr30k",
            transform=_transform,
        )
    )
