# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""Download the Tier-4 contrastive-pairs corpora into the host data root and symlink into `data/`.

Each corpus listed in `data/README.md` under "Text ↔ Text — Contrastive Pairs" that is not already on
disk is pulled here via `huggingface_hub.snapshot_download`. Every target is ungated, so
the download runs anonymously (`token=False`) — this sidesteps the stale-token version mismatch
between the `hf` CLI and the installed `huggingface_hub` Python library.

Layout matches the existing symlink farm: each corpus lands at `<DATA_ROOT>/<name>` and a
`data/<name>` symlink points at it, mirroring how `finewiki-100bt` / `msmarco` / etc. are wired.

Idempotent: a `.downloaded` sentinel inside each target short-circuits re-runs unless `--force`.

Usage:
    uv run python_data/download_text_pairs.py                 # all Tier-4 sources
    uv run python_data/download_text_pairs.py --only all-nli,quora-duplicates
    uv run python_data/download_text_pairs.py --tier light    # the four small CC pulls only
"""

from __future__ import annotations

import argparse
import os
import sys

from acquire_shared import cap_shards, data_root
from huggingface_hub import list_repo_files, snapshot_download

#: Default host data root — overridable via `UFORM_DATA_ROOT`; the symlink target every existing
#: `data/<name>` points at.
DEFAULT_DATA_ROOT = str(data_root())
#: Suffixes that mark a downloadable data shard (everything else — README, metadata — is always kept).
SHARD_SUFFIXES = (".parquet", ".json", ".jsonl", ".gz")

#: Tier-4 download set. Each entry: local dir name → (hf_repo_id, tier).
#: `tier` values: `light` (< 1 GB CC pulls), `multilingual` (retrieval), `heavy` (STS-gold).
SOURCES: dict[str, tuple[str, str]] = {
    "all-nli": ("sentence-transformers/all-nli", "light"),
    "quora-duplicates": ("sentence-transformers/quora-duplicates", "light"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "light"),
    "gooaq": ("sentence-transformers/gooaq", "light"),
    "mr-tydi": ("castorini/mr-tydi", "multilingual"),
    "mmarco": ("unicamp-dl/mmarco", "multilingual"),
    "paq": ("sentence-transformers/paq", "heavy"),
    "wikianswers-duplicates": ("sentence-transformers/wikianswers-duplicates", "heavy"),
    # Cross-lingual translation pairs — the biggest gap in the contrastive corpus. Both
    # sentence-transformers parallel-sentences mirrors ship flat `(english, non_english)` parquet
    # across 40+ language-pair subsets (true cross-lingual positives, hundreds of languages).
    "parallel-sent-wikimatrix": ("sentence-transformers/parallel-sentences-wikimatrix", "multilingual"),
    "parallel-sent-tatoeba": ("sentence-transformers/parallel-sentences-tatoeba", "multilingual"),
    # PAWS-X — 7-language adversarial word-order paraphrase ID (`sentence1`/`sentence2`/`label`).
    # High lexical overlap / different meaning = the hardest fuzzy positives+negatives; the
    # registry extractor keeps only `label==1` (true paraphrase) rows.
    "paws-x": ("google-research-datasets/paws-x", "multilingual"),
    # Natural Questions — Google search queries paired to their Wikipedia answer passage. One `pair/`
    # config, flat `(query, answer)` parquet — exactly the text_contrastive columns, so a plain snapshot
    # reproduces `data/natural-questions/pair/train-*.parquet` with no transform. Ungated.
    "natural-questions": ("sentence-transformers/natural-questions", "multilingual"),
    # PubMedQA — niche MEDICAL retrieval, the biggest distinctive-domain gap (no biomedical signal in
    # the prior mixture). Flat `(question, long_answer)` parquet under `pqa_artificial/` (211K synthetic)
    # + `pqa_labeled/` (1K expert) subdirs; the registry extractor pairs question -> long_answer. The
    # `pqa_unlabeled/` split has no long_answer so the glob omits it. Ungated, ~300 MB.
    "pubmedqa": ("qiaojin/PubMedQA", "niche"),
    # Reddit title->body — CONVERSATIONAL / informal-register robustness (slang, casual phrasing), a
    # register the QA/web mixture lacks. Ships 12 yearly `reddit_title_text_<year>.jsonl.gz` (39 GB
    # total); we keep only the 2010-2012 files via the registry glob so the conversational bucket stays
    # modest without a 39 GB pull driving the mix. Ungated; `{title, body, subreddit}` JSONL.
    "reddit-title-body": ("sentence-transformers/reddit-title-body", "conversational"),
}


def _fraction_allow_patterns(repo: str, fraction: float) -> list[str] | None:
    """Select the leading `fraction` of a dataset repo's data shards as `allow_patterns`.

    Lists the repo's files, partitions them into data shards (those ending in a known shard
    suffix) and everything else (README, metadata, which are always kept), then returns the
    non-shard files plus the first `max(1, ceil(shard_count * fraction))` data shards. Returns
    `None` when `fraction` is at or above `1.0` so the caller downloads the whole repo unfiltered.

    Args:
        repo: HuggingFace dataset repo id.
        fraction: Fraction of the repo's data shards to keep, in (0, 1].

    Returns:
        The `allow_patterns` list to pass to `snapshot_download`, or `None` to download everything.
    """
    if fraction >= 1.0:
        return None
    repo_files = list_repo_files(repo, repo_type="dataset")
    shards = sorted(name for name in repo_files if name.endswith(SHARD_SUFFIXES))
    non_shards = [name for name in repo_files if not name.endswith(SHARD_SUFFIXES)]
    return non_shards + cap_shards(shards, fraction, None)


def _link_into_data(root: str, name: str) -> None:
    """Create or refresh the `data/<name>` → `<root>/<name>` symlink, idempotently.

    Ensures both the target directory and the repo `data/` dir exist, skips when the symlink
    already points at the right target, and replaces a stale or broken symlink in place.
    """
    target = os.path.join(root, name)
    os.makedirs(target, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    link = os.path.join("data", name)
    if os.path.islink(link):
        if os.readlink(link) == target:
            return
        os.unlink(link)  # stale or broken symlink — replace it
    elif os.path.exists(link):
        return  # a real file/dir already occupies the path; leave it untouched
    os.symlink(target, link)
    print(f"  symlinked data/{name} -> {target}", flush=True)


def main() -> int:
    """Pull the selected Tier-4 corpora, write per-corpus sentinels, and symlink into `data/`.

    Returns:
        Process exit code: `0` when every selected source completed, `1` on a selection miss or any
        per-source failure.
    """
    parser = argparse.ArgumentParser(description="Download Tier-4 contrastive-pairs corpora.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--only", default="", help="Comma-separated subset of local dir names.")
    parser.add_argument(
        "--tier",
        default="",
        choices=["", "light", "multilingual", "heavy", "niche", "conversational"],
        help="Restrict to one tier; empty = all.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if sentinel present.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction (0, 1] of each repo's data shards to download; 1.0 downloads everything.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Row cap honoured by the repack scripts; accepted here for a uniform CLI but with no "
        "effect on full-file snapshot downloads.",
    )
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = {
        name: repo
        for name, (repo, tier) in SOURCES.items()
        if (not only or name in only) and (not args.tier or tier == args.tier)
    }
    if not selected:
        print(f"[download] nothing selected (only={only or '*'} tier={args.tier or '*'})", flush=True)
        return 1

    print(f"[download] data_root={args.data_root}  selected={list(selected)}", flush=True)
    failures: list[str] = []
    for name, repo in selected.items():
        target = os.path.join(args.data_root, name)
        sentinel = os.path.join(target, ".downloaded")
        if os.path.exists(sentinel) and not args.force:
            print(f"[skip] {name}: sentinel present", flush=True)
            _link_into_data(args.data_root, name)
            continue
        print(f"[pull] {name} <- {repo}", flush=True)
        try:
            allow_patterns = _fraction_allow_patterns(repo, args.fraction)
            snapshot_download(repo, repo_type="dataset", local_dir=target, token=False, allow_patterns=allow_patterns)
            with open(sentinel, "w") as handle:
                handle.write(repo + "\n")
            _link_into_data(args.data_root, name)
            print(f"[done] {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 — report and continue to next source
            print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            failures.append(name)

    if failures:
        print(f"[download] FAILED: {failures}", flush=True)
        return 1
    print("[download] all selected sources complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
