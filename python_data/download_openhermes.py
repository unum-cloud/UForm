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
"""Download OpenHermes-2.5: ~1M curated instruction-response pairs, flattened to (prompt, response)."""

from acquire import acquire


def _flatten_conversation(conversations) -> tuple[str, str] | None:
    """First human turn -> prompt, first gpt turn after it -> response; None if either is absent."""
    prompt: str | None = None
    response: str | None = None
    for turn in conversations:
        role = turn.get("from")
        value = turn.get("value")
        if value is None:
            continue
        if prompt is None and role == "human":
            prompt = value
        elif prompt is not None and response is None and role == "gpt":
            response = value
            break
    if prompt is None or response is None:
        return None
    return prompt, response


def _transform(frame):
    """Flatten the OpenHermes `conversations` list-of-turns column to flat (prompt, response) rows."""
    import cudf  # noqa: PLC0415 — GPGPU dep imported lazily, matching acquire's text_jsonl path

    prompts: list[str] = []
    responses: list[str] = []
    sources: list[str | None] = []
    categories: list[str | None] = []
    for row in frame.to_pandas().to_dict("records"):
        pair = _flatten_conversation(row.get("conversations") or [])
        if pair is None:
            continue
        prompt, response = pair
        prompts.append(prompt)
        responses.append(response)
        sources.append(row.get("source"))
        categories.append(row.get("category"))
    return cudf.DataFrame({"prompt": prompts, "response": responses, "source": sources, "category": categories})


if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "openhermes",
            repo_id="teknium/OpenHermes-2.5",
            shard_suffix=(".json",),
            source_kind="text_jsonl",
            out_name="openhermes",
            transform=_transform,
        )
    )
