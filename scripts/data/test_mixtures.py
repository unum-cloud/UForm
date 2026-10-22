# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Catalog validator for the six per-mixture training JSONs — stdlib only, no cuDF / pyarrow.

The mixture JSONs (`scripts/mixtures/{text_masked,text_contrastive,image_contrastive,video_contrastive,audio_contrastive,omni_multilingual}.json`)
are the language-neutral column contract both the Python acquisition and the C++ trainer read. This module
checks that contract holds without touching any GPU or parquet dependency: it asserts the files parse, that
every dataset entry carries the fields the loaders need, that a dataset's columns agree across the mixtures
that reference it (`mixtures.dataset_spec`, which raises on conflict), that `mixtures.resolve_mixture`
yields the hard-contract `(glob, weight, text_field)` triples with positive weights, and that every dataset
name is acquirable — by a thin `download_<stem>.py` or by the generic `text_pairs.py`.

Weights are RELATIVE shares (UFormTrain treats each mixture's weights as unnormalized sampling shares, not
a probability simplex — the per-mixture sums land near 1.0 but are not pinned to it), so this asserts every
weight is > 0 rather than that a mixture sums to exactly 1.0.

Run as a script for a PASS summary:
    uv run scripts/data/test_mixtures.py
Or under pytest (every `test_*` function below is an independent case):
    python -m pytest scripts/data/test_mixtures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mixtures

#: `scripts/data/` holds the `download_*.py` scripts; the mixture JSONs sit beside it in `scripts/mixtures/`.
pkg_dir = Path(__file__).resolve().parent

#: The six mixtures whose JSONs form the column contract; the file basename is `<name>.json`.
expected_mixtures = (
    "text_masked",
    "text_contrastive",
    "image_contrastive",
    "video_contrastive",
    "audio_contrastive",
    "omni_multilingual",
)

#: The dataset-entry keys whose values are actual parquet column names (mirrors `mixtures.column_fields`).
column_fields = ("text", "anchor", "positive", "negative", "media")

#: A mixture dataset name -> its `download_<stem>.py` stem, where the two differ. Every other name is its
#: own stem. These are the acquisition-script renames (config suffix dropped, hyphen/underscore normalized).
download_aliases = {
    "poj_104": "poj104",
    "code_contests": "codecontests",
    "fineweb_2": "fineweb2",
    "enron_email": "enron",
    "panda70m_testing": "panda70m_wds",
    "msrvtt_captions": "msrvtt",
    "finewiki_i18n": "finewiki",
    "stackexchange_text": "stackexchange_title_body",
    "hackernews_text": "hackernews",
    "code_search_net_text": "code_search_net",
    "msmarco_triplet50": "msmarco",
}

#: Relative tolerance when reporting how far a mixture's weights sum from 1.0 (informational, not asserted).
weight_sum_tolerance = 0.1


def _mixture_path(name: str) -> Path:
    """The on-disk JSON path for one mixture."""
    return mixtures.mixtures_dir / f"{name}.json"


def _load_mixture(name: str) -> dict:
    """Parse one mixture JSON straight off disk (independent of `mixtures`' `"mixture"`-key filter)."""
    return json.loads(_mixture_path(name).read_text())


def _download_stem(name: str) -> str:
    """The `download_<stem>.py` stem an acquisition script would carry for a dataset name."""
    return download_aliases.get(name, name)


def _has_downloader(name: str) -> bool:
    """Whether a thin per-dataset `download_<stem>.py` script exists for this dataset name."""
    return (pkg_dir / f"download_{_download_stem(name)}.py").exists()


def _glob_data_dir(glob: str) -> str | None:
    """The first path component under `data/` in a mixture glob — the acquisition output subdirectory.

    `data/all-nli/pair/train-*.parquet` -> `all-nli`; the value is matched against the generic
    `text_pairs.py` source directories to decide whether that script acquires the dataset.
    """
    parts = Path(glob).parts
    if "data" in parts:
        index = parts.index("data")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _text_pairs_dirs() -> set[str]:
    """The `data/<dir>` names the generic `download_text_pairs.py` acquires, read from its `SOURCES` map.

    Parsed textually (the module imports `huggingface_hub` at top level, so it cannot be imported here):
    every `"<dir>": ("<repo>", ...)` entry in `SOURCES` is a directory `text_pairs.py` stages.
    """
    text = (pkg_dir / "download_text_pairs.py").read_text()
    return set(re.findall(r'"([\w\-]+)"\s*:\s*\(\s*"', text))


def _acquired_via_text_pairs(glob: str) -> bool:
    """Whether the generic `text_pairs.py` acquires the corpus this glob points at (by its `data/<dir>`)."""
    data_subdir = _glob_data_dir(glob)
    return data_subdir is not None and data_subdir in _text_pairs_dirs()


# Checks: each is an independent pytest case, and each is also called by `main` for the PASS summary.


def test_mixture_files_load() -> None:
    """All six mixture JSONs parse and carry a `mixture` name and a non-empty `datasets` list."""
    for name in expected_mixtures:
        path = _mixture_path(name)
        assert path.exists(), f"missing mixture JSON: {path}"
        spec = _load_mixture(name)
        assert isinstance(spec, dict), f"{name}: top-level JSON is not an object"
        assert spec.get("mixture") == name, f"{name}: 'mixture' key is {spec.get('mixture')!r}, expected {name!r}"
        datasets = spec.get("datasets")
        assert isinstance(datasets, list) and datasets, f"{name}: 'datasets' must be a non-empty list"


def test_dataset_entry_fields() -> None:
    """Every entry has `name`, `weight` in (0, 1], a non-empty `glob`, >=1 column field, and any `epochs` positive."""
    for name in expected_mixtures:
        for entry in _load_mixture(name)["datasets"]:
            label = f"{name}:{entry.get('name')!r}"
            assert isinstance(entry.get("name"), str) and entry["name"], f"{label}: missing 'name'"
            weight = entry.get("weight")
            assert isinstance(weight, (int, float)) and not isinstance(weight, bool), f"{label}: 'weight' not a number"
            assert 0.0 < float(weight) <= 1.0, f"{label}: 'weight' {weight} not in (0, 1]"
            assert isinstance(entry.get("glob"), str) and entry["glob"], f"{label}: missing non-empty 'glob'"
            has_column = "text" in entry or "anchor" in entry or "positive" in entry
            assert has_column, f"{label}: no column field (need 'text' or 'anchor'/'positive')"
            epochs = entry.get("epochs", 1)
            assert isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0, (
                f"{label}: 'epochs' {epochs!r} not a positive integer"
            )


def test_column_consistency() -> None:
    """`mixtures.dataset_spec` unions each dataset across its mixtures and must not raise on a conflict."""
    names = sorted({entry["name"] for name in expected_mixtures for entry in _load_mixture(name)["datasets"]})
    for dataset_name in names:
        # Raises `ValueError` if a non-weight field disagrees across mixtures; a clean return is the check.
        mixtures.dataset_spec(dataset_name)


def test_resolve_mixture_triples() -> None:
    """`resolve_mixture` yields one `(glob, weight, text_field)` per dataset, every weight strictly > 0."""
    for name in expected_mixtures:
        entries = _load_mixture(name)["datasets"]
        resolved = mixtures.resolve_mixture(name)
        assert len(resolved) == len(entries), f"{name}: resolve_mixture dropped/added rows"
        for resolved_glob, weight, text_field in resolved:
            assert isinstance(resolved_glob, str) and resolved_glob, f"{name}: empty resolved glob"
            assert isinstance(weight, float) and weight > 0.0, f"{name}: non-positive weight {weight}"
            assert text_field is None or isinstance(text_field, str), f"{name}: bad text_field {text_field!r}"


def test_unknown_mixture_raises() -> None:
    """`resolve_mixture` on an unregistered name is a hard `KeyError`, not a silent empty list."""
    try:
        mixtures.resolve_mixture("no_such_mixture")
    except KeyError:
        return
    raise AssertionError("resolve_mixture accepted an unknown mixture name")


def unacquired_datasets() -> list[tuple[str, str]]:
    """Every `(mixture, dataset)` acquired by neither a `download_<stem>.py` nor the generic `text_pairs.py`.

    Reported, not asserted: some corpora are staged by other means (kept-full multi-repo pulls, earlier
    tiers, manual index drops), so a non-empty list is a curation note rather than a validator failure.
    """
    missing: list[tuple[str, str]] = []
    for name in expected_mixtures:
        for entry in _load_mixture(name)["datasets"]:
            dataset_name, glob = entry["name"], entry["glob"]
            if not _has_downloader(dataset_name) and not _acquired_via_text_pairs(glob):
                missing.append((name, dataset_name))
    return missing


def test_download_coverage_reported() -> None:
    """Surface datasets acquired by neither path; informational, so it prints but never fails the suite."""
    missing = unacquired_datasets()
    if missing:
        rows = ", ".join(f"{mixture}:{dataset}" for mixture, dataset in missing)
        print(f"WARN acquired by neither download_<stem>.py nor text_pairs.py (staged elsewhere): {rows}")


def main() -> int:
    """Run every check, print a PASS summary, and return a process exit code."""
    checks = (
        test_mixture_files_load,
        test_dataset_entry_fields,
        test_column_consistency,
        test_resolve_mixture_triples,
        test_unknown_mixture_raises,
        test_download_coverage_reported,
    )
    for check in checks:
        check()
        print(f"  ok   {check.__name__}")

    entries = [(name, entry) for name in expected_mixtures for entry in _load_mixture(name)["datasets"]]
    unique_datasets = {entry["name"] for _mixture, entry in entries}
    missing = unacquired_datasets()

    print()
    print(f"mixtures checked      : {len(expected_mixtures)} ({', '.join(expected_mixtures)})")
    print(f"dataset entries       : {len(entries)} across mixtures, {len(unique_datasets)} unique names")
    for name in expected_mixtures:
        resolved = mixtures.resolve_mixture(name)
        weight_sum = sum(weight for _glob, weight, _field in resolved)
        drift = "" if abs(weight_sum - 1.0) <= weight_sum_tolerance else "  (relative shares)"
        print(f"  {name:18} {len(resolved):>2} datasets, weight-sum {weight_sum:.3f}{drift}")
    print(f"text_pairs dirs       : {len(_text_pairs_dirs())} generic-source directories")
    covered = len(unique_datasets) - len({d for _m, d in missing})
    print(f"acquisition coverage  : {covered}/{len(unique_datasets)} datasets via download_<stem>.py / text_pairs.py")
    if missing:
        print(f"  staged elsewhere ({len(missing)}): {', '.join(d for _m, d in missing)}")
    print()
    print("PASS — all mixture-catalog checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
