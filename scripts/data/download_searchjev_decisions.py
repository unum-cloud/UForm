"""Download SearchDecision-Bench and convert its questions into choice rows.

Every `choice` question and every `noul` proposition becomes one row: the state, the question and
its options lettered `A`, `B`, ... as text, the option letters as the choice bytes, and one
little-endian Float32 target probability per letter. A `noul` is a two-option choice between true
and false. `score` questions, masked questions and questions with more options than letters are
left out.

The train split lands as Parquet shards under `train/` for the choices loader; the val and test
splits land as JSON lines under `heldout/` for `uform_eval --calibrate`, each line carrying its
question type as `group`: 0 for a choice, 1 for a proposition.

Usage:
    uv run scripts/data/download_searchjev_decisions.py [--data-root data]

File: scripts/data/download_searchjev_decisions.py
Author: Ash Vardanian
Date: October 27, 2026
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///

import argparse
import json
import struct
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
from huggingface_hub import snapshot_download

#: The release and the revision every conversion reads.
repo_id = "SearchJev/search_jev_unified_v2_release_20260928"
revision = "ce677ef8db3dbddce4608d8182bfeb93b62edf81"

#: The option letters, one choice byte each; a question with more options is left out.
option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

#: The calibration group of each kept question type.
group_of_type = {"choice": 0, "noul": 1}

#: Rows per Parquet shard and per row group, the unit the choices loader draws.
rows_per_shard = 65_536
rows_per_group = 4_096


def _state_text(state: object) -> str:
    """The state as text: a string as is, an object as `key: value` lines, a list one item per line."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{key}: {_value_text(value)}" for key, value in state.items())
    if isinstance(state, list):
        return "\n".join(_value_text(value) for value in state)
    return _value_text(state)


def _value_text(value: object) -> str:
    """A string as is, anything else as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _options(question: dict, answer: dict) -> tuple[list[tuple[str, str]], list[float]] | None:
    """The question's `(name, description)` options and their target probabilities, or `None` to skip it."""
    if question["type"] == "noul":
        criteria = question.get("criteria") or {}
        probability = float(answer["noul"])
        options = [("true", criteria.get("true", "")), ("false", criteria.get("false", ""))]
        return options, [probability, 1.0 - probability]
    criteria = question.get("criteria") or {}
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= len(option_letters):
        return None
    options = [(str(name), _value_text(description)) for name, description in criteria.items()]
    probabilities = answer.get("probabilities") or {answer.get("choice"): 1.0}
    targets = [max(float(probabilities.get(name, 0.0)), 0.0) for name, _ in options]
    total = sum(targets)
    if total <= 0.0:
        return None
    return options, [target / total for target in targets]


def _rows(data_path: Path, index_path: Path) -> Iterator[tuple[str, bytes, list[float], int]]:
    """Yield `(state, choices, targets, group)` for every kept question of one split."""
    with data_path.open(encoding="utf-8") as data_file, index_path.open(encoding="utf-8") as index_file:
        for line, index_line in zip(data_file, index_file, strict=True):
            row, index = json.loads(line), json.loads(index_line)
            masked = index.get("masked_questions") or {}
            state = _state_text(row["state"])
            for question_id, question in row["questions"].items():
                group = group_of_type.get(question["type"])
                answer = row["answers"].get(question_id)
                if group is None or answer is None or question_id in masked:
                    continue
                kept = _options(question, answer)
                if kept is None:
                    continue
                options, targets = kept
                letters = option_letters[: len(options)]
                lines = [state, "", question.get("instructions", "")]
                for letter, (name, description) in zip(letters, options, strict=True):
                    lines.append(f"{letter}) {name}: {description}" if description else f"{letter}) {name}")
                lines.append("Answer: ")
                yield "\n".join(lines), letters.encode("ascii"), targets, group


def _write_shards(rows: Iterator[tuple[str, bytes, list[float], int]], directory: Path) -> int:
    """Write @p rows as Parquet shards of `state`, `choices`, `targets` and `group` into @p directory."""
    directory.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [("state", pa.string()), ("choices", pa.binary()), ("targets", pa.binary()), ("group", pa.uint32())]
    )
    states, offered, probabilities, groups = [], [], [], []
    shards = written = 0

    def flush() -> None:
        nonlocal shards
        table = pa.table([states, offered, probabilities, groups], schema=schema)
        path = directory / f"searchjev_train_{shards:04d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        parquet.write_table(table, temporary, compression="zstd", row_group_size=rows_per_group)
        temporary.rename(path)
        shards += 1
        for column in (states, offered, probabilities, groups):
            column.clear()

    for state, choices, targets, group in rows:
        states.append(state)
        offered.append(choices)
        probabilities.append(struct.pack(f"<{len(targets)}f", *targets))
        groups.append(group)
        written += 1
        if len(states) == rows_per_shard:
            flush()
    if states:
        flush()
    return written


def _write_heldout(rows: Iterator[tuple[str, bytes, list[float], int]], path: Path) -> int:
    """Write @p rows as the `{"state", "choices", "targets", "group"}` JSON lines `uform_eval` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for state, choices, targets, group in rows:
            line = {"state": state, "choices": choices.decode("ascii"), "targets": targets, "group": group}
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
    temporary.rename(path)
    return written


def main() -> int:
    """Fetch the pinned release, then convert its train, val and test splits."""
    parser = argparse.ArgumentParser(description="Download SearchDecision-Bench and convert it to choice rows.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    arguments = parser.parse_args()
    root = arguments.data_root / "searchjev-decisions"
    raw = root / "raw"
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=raw,
        allow_patterns=["*.jsonl", "*.json", "*.md"],
    )
    trained = _write_shards(_rows(raw / "train.jsonl", raw / "train_index.jsonl"), root / "train")
    print(f"searchjev-decisions: {trained} training questions")
    for split in ("val", "test"):
        rows = _rows(raw / f"{split}.jsonl", raw / f"{split}_index.jsonl")
        kept = _write_heldout(rows, root / "heldout" / f"{split}.jsonl")
        print(f"searchjev-decisions: {kept} held-out {split} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
