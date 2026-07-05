# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Thin, cuDF-free reader over the per-mixture training JSONs (one flat `data/<mixture>.json`).

A mixture JSON is the language-neutral recipe both Python acquisition and C++ training read: a weighted
list of datasets, each carrying its shard `glob` and the parquet column names it exposes (`text` for the
MLM blends; `anchor` / `positive` / `negative` / `media` for the paired blends). It describes the data
only — objective, sequence length, and other trainer knobs live in the training config, not here. A
mixture file is told apart from a plain catalog `<name>.json` by the presence of a top-level `"mixture"`
key; everything without it is skipped.

Two views sit on top of the parsed JSONs. The mixture view (`mixtures`, `resolve_mixture`) is what the
loaders read: `resolve_mixture` yields the exact `(resolved_glob, weight, text_field)` tuple the deleted
`TextMaskLoader` and the C++ loader consume — that shape is a hard contract, do not widen it. The dataset
view (`dataset_spec`, `dataset_glob`, `dataset_columns`, `dataset_media_column`) aggregates a dataset
across every mixture it appears in, asserting the declared column fields agree, and supplies the column
contract the `acquire` assert and `staging_manifest` / `mine_hard_negatives` read. Stdlib `json` + `glob`
only, so importing it stays cheap enough that those tools read it instead of re-inlining glob strings.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

#: The repo root a relative `glob` anchors under, and the tracked `data/` directory holding the mixture JSONs.
project_root = Path(__file__).resolve().parent.parent
mixtures_dir = project_root / "data"

#: The dataset-entry keys whose VALUES are actual parquet column names; their union is a dataset's column
#: contract. `text` is the MLM document column; the other four are the contrastive relation columns.
column_fields = ("text", "anchor", "positive", "negative", "media")

#: Per-mixture (not per-dataset) key, excluded from the cross-mixture agreement check in `dataset_spec`.
_per_mixture_dataset_keys = ("weight",)


def _resolve_glob(glob: str) -> str:
    """Absolute path-glob, anchoring a repo-relative mixture `glob` under the project root."""
    return glob if Path(glob).is_absolute() else str(project_root / glob)


@cache
def mixtures() -> dict[str, dict]:
    """The `mixture_name -> parsed-mixture` mapping over every `data/*.json` with a `"mixture"` key."""
    parsed: dict[str, dict] = {}
    for path in sorted(mixtures_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "mixture" in data:  # a plain catalog row has `name`, not `mixture`
            parsed[data["mixture"]] = data
    return parsed


def resolve_mixture(name: str) -> list[tuple[str, float, str | None]]:
    """Map one mixture's datasets to `[(resolved_glob, weight, text_field)]`, the mask/loader tuple shape.

    `text_field` is the dataset's MLM `text` column, or its `positive` column when the mixture is
    contrastive; `None` only if a dataset declares neither. This exact three-tuple is a hard contract with
    `data_unigram_floor.py` and the C++ loader — do not add or reorder fields.
    """
    if name not in mixtures():
        raise KeyError(f"unknown mixture {name!r}; known: {sorted(mixtures())}")
    resolved: list[tuple[str, float, str | None]] = []
    for dataset in mixtures()[name]["datasets"]:
        text_field = dataset.get("text") or dataset.get("positive")
        resolved.append((_resolve_glob(dataset["glob"]), float(dataset["weight"]), text_field))
    return resolved


@cache
def _dataset_names() -> frozenset[str]:
    """Every dataset name declared by any mixture (used to make `dataset_columns` tolerant of non-members)."""
    return frozenset(dataset["name"] for mixture in mixtures().values() for dataset in mixture["datasets"])


@cache
def dataset_spec(name: str) -> dict:
    """The one dataset's spec, aggregated across every mixture it appears in.

    A dataset may appear in several mixtures (with different `weight`s); every other field must agree, so
    this unions the per-mixture entries and raises on any conflicting non-`weight` value. The returned dict
    carries `name`, `glob`, and whichever column fields (`text` / `anchor` / `positive` / `negative` /
    `media`) and prefixes the entries declared — `weight` is dropped, being per-mixture.
    """
    entries = [dataset for mixture in mixtures().values() for dataset in mixture["datasets"] if dataset["name"] == name]
    if not entries:
        raise KeyError(f"unknown dataset {name!r}; declared by no mixture. known: {sorted(_dataset_names())}")
    merged: dict = {}
    for entry in entries:
        for key, value in entry.items():
            if key in _per_mixture_dataset_keys:
                continue
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"dataset {name!r}: conflicting {key!r} across mixtures ({merged[key]!r} vs {value!r}); "
                    "a dataset's non-weight fields must agree in every mixture that references it"
                )
            merged[key] = value
    return merged


def dataset_glob(name: str) -> str:
    """The resolved (absolute) shard glob for a dataset, aggregated across its mixtures."""
    return _resolve_glob(dataset_spec(name)["glob"])


def dataset_columns(name: str) -> set[str]:
    """The set of parquet column names a dataset's mixtures declare — its column contract.

    The union of the `text` / `anchor` / `positive` / `negative` / `media` VALUES. A dataset referenced by
    no mixture (e.g. an eval-only probe) returns the empty set rather than raising, so the `acquire` column
    assert over it is vacuous.
    """
    if name not in _dataset_names():
        return set()
    spec = dataset_spec(name)
    return {spec[field] for field in column_fields if spec.get(field)}


def dataset_media_column(name: str) -> str | None:
    """The dataset's `media` column, or `None` when it declares none (a text-only or index dataset)."""
    return dataset_spec(name).get("media")


@cache
def datasets() -> dict[str, dict]:
    """The flat `dataset_name -> aggregated-spec` view unioned across every mixture.

    The dataset-centric counterpart to `mixtures()`: one entry per distinct dataset name, its value the
    cross-mixture-agreed `dataset_spec` (its `glob`, column fields, and prefixes, `weight` dropped). Tools
    that want a single corpus registry rather than the per-mixture blends read this in place of the deleted
    catalog's `catalog()` map.
    """
    return {name: dataset_spec(name) for name in sorted(_dataset_names())}
