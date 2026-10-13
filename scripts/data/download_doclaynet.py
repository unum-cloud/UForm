# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download DocLayNet: document page images with layout-class-name weak captions, into the image schema."""
import pyarrow as pa
from acquire import acquire

#: The eleven DocLayNet COCO category ids (1-indexed) mapped to their canonical layout class names.
_CATEGORY_NAMES = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


def _transform(table: pa.Table) -> pa.Table:
    """Map each page's `category_id` list to deduplicated class names as `captions`; drop unlabelled pages."""
    per_page = table.column("category_id").to_pylist()
    rows: list[list[str]] = []
    for category_ids in per_page:
        present = {int(category_id) for category_id in (category_ids or []) if category_id in _CATEGORY_NAMES}
        rows.append([_CATEGORY_NAMES[category_id] for category_id in sorted(present)])
    captions = pa.array(rows, type=pa.list_(pa.string()))
    out = pa.table({"image": table.column("image"), "captions": captions})
    keep = pa.array([len(row) > 0 for row in rows], type=pa.bool_())
    return out.filter(keep)


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "doclaynet",
            repo_id="ds4sd/DocLayNet-v1.1",
            revision="refs/convert/parquet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="doclaynet",
            transform=_transform,
        )
    )
