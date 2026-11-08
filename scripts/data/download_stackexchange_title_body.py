"""Download StackExchange title/body posts, the shared dir behind `stackexchange_title_body` + `stackexchange_text`.

Each per-site `<site>.jsonl.gz` line is `{"texts": [title, body], "tags": [...]}`. The `_transform` splits
`texts` into the positional string columns `"0"` (title) / `"1"` (body) that the contrastive
`stackexchange_title_body` mixture reads, and mirrors the body into a named `body` column that the
`stackexchange_text` MLM mixture reads — so one acquisition satisfies both mixtures' column contracts.

File: scripts/data/download_stackexchange_title_body.py
Author: Ash Vardanian
Date: July 5, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cudf-cu12",
#     "huggingface_hub",
#     "pandas",
#     "pyarrow>=21.0.0",
#     "zstandard",
# ]
# ///

from acquire import acquire


def _transform(frame):
    """Split the `texts` list into `"0"` (title) / `"1"` (body), plus a `body` alias for the MLM mixture."""
    import cudf  # noqa: PLC0415 — GPGPU dep imported lazily, matching acquire's text_jsonl path

    titles = frame["texts"].list.get(0)
    bodies = frame["texts"].list.get(1)
    return cudf.DataFrame({"0": titles, "1": bodies, "body": bodies})


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "stackexchange_title_body",
            repo_id="flax-sentence-embeddings/stackexchange_title_body_jsonl",
            revision="main",
            shard_suffix=(".jsonl.gz",),
            source_kind="text_jsonl",
            out_name="stackexchange-title-body",
            transform=_transform,
        )
    )
