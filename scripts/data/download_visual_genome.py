"""Download Visual Genome: ~108K images with dense region descriptions, into the image registry schema.

File: scripts/data/download_visual_genome.py
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

#: Cap on region phrases packed per image (Visual Genome carries ~50) to bound the
#: `captions` list width.
_MAX_CAPTIONS_PER_IMAGE = 16


def _transform(table: pa.Table) -> pa.Table:
    """Flatten each image's `region_descriptions` into a deduplicated, width-capped `captions` list."""
    per_image = table.column("region_descriptions").to_pylist()
    rows: list[list[str]] = []
    for regions in per_image:
        unique: list[str] = []
        for phrase in regions or []:
            if phrase and phrase not in unique:
                unique.append(phrase)
            if len(unique) >= _MAX_CAPTIONS_PER_IMAGE:
                break
        rows.append(unique)
    captions = pa.array(rows, type=pa.list_(pa.string()))
    out = pa.table({"image": table.column("image"), "captions": captions})
    keep = pa.array([len(row) > 0 for row in rows], type=pa.bool_())
    return out.filter(keep)


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "visual_genome",
            repo_id="nirmalendu01/visual_genome",
            revision="refs/convert/parquet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="visual_genome",
            transform=_transform,
        )
    )
