"""Download Fashion-IQ: composed-retrieval triplets collapsed to (target image, modifier captions).

Lossy simplification: the reference ("candidate") image is dropped and the relative modifier `caption`
is used as the target's `captions`, so the pair is weaker supervision than a true caption. The
`royokong/fashioniq_val` mirror is validation-only (~6K triplets).

File: scripts/data/download_fashion_iq.py
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
    """Keep the `target` image (renamed `image`) and the modifier `caption` list (renamed `captions`)."""
    return table.select(["target", "caption"]).rename_columns(["image", "captions"])


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "fashion_iq",
            repo_id="royokong/fashioniq_val",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="fashion_iq",
            transform=_transform,
        )
    )
