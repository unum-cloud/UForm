# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download Clotho v2: 6,974 audio clips with FIVE captions each -> `captions` list + `audio` parquet."""

import pyarrow as pa
from acquire import acquire

#: The five per-clip caption columns in the gijs/clotho mirror.
_CAPTION_COLUMNS = tuple(f"caption_{index}" for index in range(1, 6))


def _transform(table: pa.Table) -> pa.Table:
    """Pack the five `caption_i` columns into one `captions` list<string>, keeping the `audio` struct.

    Empty captions are dropped per row so each clip's `captions` holds only its non-blank positives —
    the multi-positive anchor the audio loader reads against the `audio` payload.
    """
    columns = [table.column(name).to_pylist() for name in _CAPTION_COLUMNS]
    per_row = [[caption for caption in row if caption] for row in zip(*columns)]
    return pa.table({"audio": table.column("audio"), "captions": pa.array(per_row, type=pa.list_(pa.string()))})


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "clotho",
            repo_id="gijs/clotho",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="clotho",
            transform=_transform,
        )
    )
