# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download DOCCI: ~15K images, one long dense description each, into the image registry schema."""
import pyarrow as pa
from acquire import acquire


def _transform(table: pa.Table) -> pa.Table:
    """Pack the dense `detailed_caption` as a one-element `captions` list beside `image`; drop empties."""
    descriptions = table.column("detailed_caption").to_pylist()
    captions = pa.array([[text] if text else [] for text in descriptions], type=pa.list_(pa.string()))
    out = pa.table({"image": table.column("image"), "captions": captions})
    keep = pa.array([len(row) > 0 for row in captions.to_pylist()], type=pa.bool_())
    return out.filter(keep)


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "docci",
            repo_id="ljnlonoljpiljm/docci",
            revision="refs/convert/parquet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="docci",
            transform=_transform,
        )
    )
