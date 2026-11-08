"""Download ToxiGen: machine-generated toxic/benign statements into a flat safety schema (`text`).

File: scripts/data/download_toxigen.py
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
import pyarrow.compute as pc
from acquire import acquire

#: A human `toxicity_human` Likert mean (1-5) at or above this is labelled toxic.
_TOXIC_THRESHOLD = 3.0


def _transform(table: pa.Table) -> pa.Table:
    """Unify either ToxiGen config into `(text, label, toxicity_score, target_group, source)`.

    The convert branch ships two configs with different schemas: the machine-generated `train`
    (`generation` / `group` / `prompt_label` / `roberta_prediction`) and the human `annotated`
    (`text` / `target_group` / `toxicity_human`). The config is recognized by its columns.
    """
    if "generation" in table.column_names:  # machine-generated `train` config
        text = table.column("generation")
        score = table.column("roberta_prediction").cast(pa.float64())
        label = table.column("prompt_label").cast(pa.int8())
        target_group = table.column("group")
        source = "train"
    else:  # human-annotated config — derive the binary label from the Likert mean
        text = table.column("text")
        score = table.column("toxicity_human").cast(pa.float64())
        label = pc.if_else(pc.greater_equal(score, _TOXIC_THRESHOLD), pa.scalar(1, pa.int8()), pa.scalar(0, pa.int8()))
        target_group = table.column("target_group")
        source = "annotated"
    return pa.table(
        {
            "text": text.cast(pa.string()),
            "label": label,
            "toxicity_score": score,
            "target_group": target_group.cast(pa.string()),
            "source": pa.array([source] * table.num_rows, pa.string()),
        }
    )


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "toxigen",
            repo_id="toxigen/toxigen-data",
            revision="refs/convert/parquet",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="toxigen",
            transform=_transform,
        )
    )
