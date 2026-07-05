# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download XM3600: 3,600 images captioned by humans in 36 languages, into the image registry schema.

Each source shard is one language file (`data/<lang>-00000-of-00001.parquet`) carrying that language's
`captions` for all 3,600 images. Acquisition is per-shard, so this emits one canonical shard PER
LANGUAGE (rebuilding the null-`path` source struct as `struct<bytes, path=image_id>`); the original
downloader's cross-language group-by `image_id` into a single multilingual `captions` list is not
expressible in the per-shard `acquire` model and is dropped.
"""
import pyarrow as pa
from acquire import acquire

_IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _transform(table: pa.Table) -> pa.Table:
    """Rebuild the image struct with `path=image_id` and keep this language's non-empty captions."""
    image_ids = table.column("image_id").to_pylist()
    image_structs = table.column("image").to_pylist()
    caption_lists = table.column("captions").to_pylist()
    images = pa.array(
        [{"bytes": struct["bytes"], "path": image_id} for struct, image_id in zip(image_structs, image_ids, strict=True)],
        type=_IMAGE_STRUCT,
    )
    captions = pa.array(
        [[caption for caption in (caption_list or []) if caption] for caption_list in caption_lists],
        type=pa.list_(pa.string()),
    )
    out = pa.table({"image": images, "captions": captions})
    keep = pa.array([len(row) > 0 for row in captions.to_pylist()], type=pa.bool_())
    return out.filter(keep)


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "xm3600",
            repo_id="floschne/xm3600",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="xm3600",
            transform=_transform,
        )
    )
