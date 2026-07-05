# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download Civil Comments: real toxic/benign comments into a flat safety schema (`text`)."""
import pyarrow as pa
import pyarrow.compute as pc
from acquire import acquire

#: A continuous `toxicity` annotator fraction at or above this is labelled toxic (the standard 0.5 cut).
_TOXIC_THRESHOLD = 0.5


def _transform(table: pa.Table) -> pa.Table:
    """Repack a Civil Comments shard into `(text, label, toxicity_score, target_group, source)`.

    Derives the binary `label` by thresholding the continuous `toxicity` score; Civil Comments carries
    no per-row target group, so a constant placeholder is written (matching the ToxiGen safety schema).
    """
    score = table.column("toxicity").cast(pa.float64())
    label = pc.if_else(pc.greater_equal(score, _TOXIC_THRESHOLD), pa.scalar(1, pa.int8()), pa.scalar(0, pa.int8()))
    return pa.table(
        {
            "text": table.column("text").cast(pa.string()),
            "label": label,
            "toxicity_score": score,
            "target_group": pa.array(["unspecified"] * table.num_rows, pa.string()),
            "source": pa.array(["civil_comments"] * table.num_rows, pa.string()),
        }
    )


if __name__ == "__main__":
    # The main branch already ships native parquet (`data/<split>-*.parquet`), cols `text` + seven
    # continuous toxicity scores — CC0-1.0, so no convert-branch round-trip is needed.
    raise SystemExit(
        acquire(
            "civil_comments",
            repo_id="google/civil_comments",
            revision="main",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="civil_comments",
            transform=_transform,
        )
    )
