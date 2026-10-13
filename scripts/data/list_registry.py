# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print every training mixture and source corpus registered in this repo.

The discoverability entry point — when in doubt, run it before launching any trainer. Mixtures are
the canonical `(corpus_name, weight)` blends defined in the `scripts/data/*.json` mixture files and
keyed by `<MODALITY>_<NATURE>` slug (`mixtures.mixtures()`); they resolve against the
single `DATASETS` map, the flat cross-mixture dataset view (`mixtures.datasets()`).

Usage:
    uv run scripts/data/list_registry.py             # all sections
    uv run scripts/data/list_registry.py--trainers   # trainer-to-data convention
    uv run scripts/data/list_registry.py--mixtures   # per-trainer corpus mixtures
    uv run scripts/data/list_registry.py--sources    # the DATASETS corpus registry
    uv run scripts/data/list_registry.py--json       # machine-readable
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from mixtures import datasets, mixtures

#: Full name->spec corpus map, the flat cross-mixture dataset view replacing the old `registry.DATASETS`.
DATASETS: dict[str, dict] = datasets()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The `scripts/data/` directory holding the `download_*.py` acquisition scripts.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# The text trainers collapsed into one objective-weighted unified trainer (`python -m python_train`).
# Its MLM + contrastive-pair mixtures are the single-source-of-truth `scripts/data/*.json` mixtures.
#: Canonical mixture registry — keyed by the `<MODALITY>_<NATURE>` slug, each a `(corpus, weight)` list.
MIXTURES: dict[str, list[tuple[str, float]]] = {
    name: [(dataset["name"], float(dataset["weight"])) for dataset in mixture["datasets"]]
    for name, mixture in mixtures().items()
}


#: One row per text-modality trainer, describing how it consumes data.
#: Keys: cli (entry point), task, data_arg (what to pass), mixture (the module
#: constant the trainer defaults to), example (a concrete known-good arg), notes.
_TRAINERS: list[dict] = [
    {
        "cli": "python -m python_train --task text_base",
        "task": "MLM (masked language model) — text_base recipe of the unified trainer",
        "data_arg": "driven by the text_base mixture (scripts/data/text_base.json)",
        "mixture": "text_base (scripts/data/text_base.json)",
        "example": "--config text-tiny --total-steps 50000",
        "notes": "Single-text i18n mixture; weighted glob of parquet / jsonl shards.",
    },
    {
        "cli": "python -m python_train --task text_contrastive",
        "task": "Matryoshka SigLIP embedder — text_contrastive recipe of the unified trainer",
        "data_arg": "driven by the text_contrastive mixture (scripts/data/text_contrastive.json)",
        "mixture": "text_contrastive (scripts/data/text_contrastive.json)",
        "example": "--config text-tiny --warm-start-from <mlm.safetensors>",
        "notes": "Paired (anchor, positive) data; in-batch negatives.",
    },
]


def _resolve_glob(default_path: str) -> str:
    """Anchor a registry-relative path to the repo root for `_dataset_size`."""
    if os.path.isabs(default_path):
        return default_path
    return os.path.join(PROJECT_ROOT, default_path)


def _dataset_size(default_path: str) -> str:
    """Return a human-readable size summary (file count, total MB) for a glob.

    Returns `"missing"` when no shard matches — useful for catching stale
    `DATASETS` entries whose data was never downloaded. Uses an inexpensive
    `os.scandir` walk; skips large `**` globs by bailing after 100 matches.
    """
    full = _resolve_glob(default_path)
    matches = glob.glob(full, recursive=True)
    if not matches:
        return "missing"
    total_bytes = 0
    counted = 0
    for path in matches:
        try:
            total_bytes += os.path.getsize(path)
        except OSError:
            continue
        counted += 1
        if counted >= 100:
            break
    total_mb = total_bytes / (1024.0 * 1024.0)
    if counted >= 100:
        return f"{len(matches):>5} shards (>{total_mb:>7.1f} MB sampled)"
    return f"{counted:>5} shards ({total_mb:>7.1f} MB)"


def _print_trainers() -> None:
    print("\nText trainers\n")
    print(f"  {'CLI module':35} {'task':28} {'data arg'}")
    print(f"  {'-' * 35} {'-' * 28} {'-' * 35}")
    for entry in _TRAINERS:
        print(f"  {entry['cli']:35} {entry['task']:28} {entry['data_arg']}")
    print()
    for entry in _TRAINERS:
        print(f"  __{entry['cli']}__")
        print(f"    mixture:  {entry['mixture']}")
        print(f"    example:  python -m {entry['cli']} {entry['example']} ...")
        print(f"    notes:    {entry['notes']}")
        print()


def _print_mixtures() -> None:
    print("\nPer-trainer mixtures (each weighted list resolves against DATASETS)\n")
    for name, entries in MIXTURES.items():
        weight_sum = sum(weight for _corpus, weight in entries)
        print(f"  __{name}__  (sum={weight_sum:.2f})")
        for corpus_name, weight in entries:
            entry = DATASETS.get(corpus_name)
            size = _dataset_size(entry["glob"]) if entry is not None else "UNKNOWN corpus"
            print(f"    {weight:>5.2f}  {corpus_name:30} {size}")
        print()


def _print_sources() -> None:
    print("\nSource corpus registry (DATASETS)\n")
    print(f"  {'name':30} {'anchor field':30} {'positive field':25} {'media':8} disk")
    print(f"  {'-' * 30} {'-' * 30} {'-' * 25} {'-' * 8} {'-' * 35}")
    for name, entry in DATASETS.items():
        anchor = entry.get("anchor") or entry.get("text") or "-"
        positive = entry.get("positive") or "-"
        media = entry.get("media") or "-"
        size = _dataset_size(entry["glob"])
        print(f"  {name:30} {anchor:30} {positive:25} {media:8} {size}")
    print()


def _emit_json() -> None:
    """Machine-readable dump for tooling / future doc-generators."""
    payload = {
        "trainers": _TRAINERS,
        "mixtures": {
            name: [{"corpus": corpus, "weight": w} for corpus, w in entries] for name, entries in MIXTURES.items()
        },
        "sources": {
            name: {
                "default_path": entry["glob"],
                "anchor_field": entry.get("anchor"),
                "positive_field": entry.get("positive"),
                "text_field": entry.get("text"),
                "media_field": entry.get("media"),
                "negative_field": entry.get("negative"),
                "anchor_prefix": entry.get("anchor_prefix"),
                "positive_prefix": entry.get("positive_prefix"),
            }
            for name, entry in DATASETS.items()
        },
    }
    print(json.dumps(payload, indent=2))


def _acquire_plan() -> tuple[list[str], list[str]]:
    """Split the per-dataset downloaders into `(shardable, whole)` stems for `scripts/slurm/acquire.sbatch`.

    A shardable downloader reads a Slurm `(rank, world)` split, so it rank-partitions its work across the
    acquisition workers — the HuggingFace-shard path (`from acquire import acquire`), the shared YouTube-clip
    path (`acquire_clips`, striding `clips[rank::world]`), or a direct `resolve_rank_world` (e.g. the msmarco
    derived-join). A whole downloader (TSV-CSV / multi-repo / the HF-mirror media sets / the generic
    `text_pairs`) cannot sub-shard and runs one-per-worker. `download_smokes.py` is excluded — it stages
    eval probes, not a training corpus.
    """
    #: Any of these tokens in a downloader means it consumes a Slurm `(rank, world)` split.
    rank_world_markers = ("from acquire import acquire", "acquire_clips", "resolve_rank_world")
    shardable: list[str] = []
    whole: list[str] = []
    for path in sorted(glob.glob(os.path.join(PKG_DIR, "download_*.py"))):
        stem = os.path.basename(path)[len("download_") : -len(".py")]
        if stem == "smokes":
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        bucket = shardable if any(marker in text for marker in rank_world_markers) else whole
        bucket.append(stem)
    return shardable, whole


def main() -> None:
    """Print the selected registry sections, or the shardable and whole downloader stems."""
    parser =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainers", action="store_true", help="Print trainer-to-data convention.")
    parser.add_argument("--mixtures", action="store_true", help="Print per-trainer corpus mixtures.")
    parser.add_argument("--sources", action="store_true", help="Print the DATASETS corpus registry.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON dump.")
    parser.add_argument(
        "--acquire-shardable", action="store_true", help="Print shardable downloader stems, one per line."
    )
    parser.add_argument(
        "--acquire-whole", action="store_true", help="Print whole-dataset downloader stems, one per line."
    )
    args = parser.parse_args()

    if args.acquire_shardable or args.acquire_whole:
        shardable, whole = _acquire_plan()
        print("\n".join(shardable if args.acquire_shardable else whole))
        return

    if args.json:
        _emit_json()
        return

    any_section = args.trainers or args.mixtures or args.sources
    if not any_section or args.trainers:
        _print_trainers()
    if not any_section or args.mixtures:
        _print_mixtures()
    if not any_section or args.sources:
        _print_sources()


if __name__ == "__main__":
    main()
