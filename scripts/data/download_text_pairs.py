# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download the Tier-4 contrastive-pairs corpora into the host data root and symlink into `data/`.

Each corpus listed in `scripts/data/README.md` under "Text ↔ Text — Contrastive Pairs" that is not already on
disk is pulled here via `huggingface_hub.snapshot_download`. Every target is ungated, so
the download runs anonymously (`token=False`) — this sidesteps the stale-token version mismatch
between the `hf` CLI and the installed `huggingface_hub` Python library.

Layout matches the existing symlink farm: each corpus lands at `<DATA_ROOT>/<name>` and a
`data/<name>` symlink points at it, mirroring how `finewiki-100bt` / `msmarco` / etc. are wired.

Three snapshots ship no Parquet the mixture loader can read, so each is repacked after its pull into the
`(query, positive, negative)` or `(title, body)` string columns its mixture entry names:

  * `mr-tydi` — each language's `train.jsonl.gz` becomes `mrtydi-v1.1-<language>/train.parquet`, one row per
    query with the text of its first positive and first negative passage; `combined` repeats the others and
    is skipped.
  * `reddit-title-body` — each `reddit_title_text_<year>.jsonl.gz` becomes `parquet/reddit_<year>.parquet`.
  * `mmarco` — the first `triples.train.ids.small.tsv` triple of every query is resolved through each Google
    translation's queries and collection into `pairs/<language>/train.parquet`.

A query without a positive or a negative is dropped, as the paired loader skips a row with an empty role.

Idempotent: a `.downloaded` sentinel inside each target short-circuits re-runs unless `--force`, and an
existing repacked file is kept unless `--force`.

Usage:
    uv run scripts/data/download_text_pairs.py                 # all Tier-4 sources
    uv run scripts/data/download_text_pairs.py --only all-nli,quora-duplicates
    uv run scripts/data/download_text_pairs.py --tier light    # the four small CC pulls only
    uv run scripts/data/download_text_pairs.py --repack-only --only mmarco --data-root /data/Data
"""

import argparse
import gzip
import json
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as compute
import pyarrow.csv as arrow_csv
import pyarrow.json as arrow_json
import pyarrow.parquet as parquet
from huggingface_hub import list_repo_files, snapshot_download

SHARD_SUFFIXES = (".parquet", ".json", ".jsonl", ".gz")
"""Suffixes that mark a downloadable data shard (everything else — README, metadata — is always kept)."""
_ROW_GROUP_SIZE: int = 50_000
"""Rows per Parquet row group of a repacked file, matching the other text repackers."""
_JSON_BLOCK_BYTES: int = 32 << 20
"""Bytes of JSON one streamed Reddit batch parses; far above the longest single post."""
_TRIPLET_SCHEMA = pa.schema([("query", pa.string()), ("positive", pa.string()), ("negative", pa.string())])
"""The `(query, positive, negative)` columns of the Mr. TyDi and mMARCO mixture entries."""
_REDDIT_SCHEMA = pa.schema([("title", pa.string()), ("body", pa.string())])
"""The `(title, body)` columns of the `reddit_title_body` mixture entry."""

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
    # total), each repacked to one `parquet/reddit_<year>.parquet`; the mixture weight, not the pull,
    # keeps the conversational bucket modest. Ungated; `{title, body, subreddit}` JSONL.
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
    from acquire_shared import cap_shards  # noqa: PLC0415 — lazy, so a stdin-piped repack needs no sibling module

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


def _write_parquet(path: Path, schema: pa.Schema, tables: Iterable[pa.Table | pa.RecordBatch]) -> None:
    """Stream `tables` into `path` through a `.tmp` rename, in row groups of at most `_ROW_GROUP_SIZE` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    rows = 0
    with parquet.ParquetWriter(pending, schema, compression="zstd") as writer:
        for table in tables:
            writer.write(table, row_group_size=_ROW_GROUP_SIZE)
            rows += table.num_rows
    os.replace(pending, path)
    print(f"  {path}: {rows} rows, {path.stat().st_size / 1e9:.2f} GB", flush=True)


def _read_tsv(path: Path, column_types: dict[str, pa.DataType]) -> pa.Table:
    """Read a headerless, unquoted TSV into the named typed columns, skipping rows with a stray tab."""
    return arrow_csv.read_csv(
        path,
        read_options=arrow_csv.ReadOptions(column_names=list(column_types)),
        parse_options=arrow_csv.ParseOptions(delimiter="\t", quote_char=False, invalid_row_handler=lambda row: "skip"),
        convert_options=arrow_csv.ConvertOptions(column_types=column_types),
    )


def _resolve_texts(table: pa.Table, ids: pa.ChunkedArray) -> pa.ChunkedArray:
    """The whitespace-trimmed `text` of each of `ids` in an `(id, text)` table, null where the table lacks it."""
    positions = compute.index_in(ids, value_set=table["id"].combine_chunks())
    return compute.utf8_trim_whitespace(table["text"].take(positions))


def _repack_mr_tydi(target: Path, force: bool) -> None:
    """Flatten each language's `train.jsonl.gz` into `train.parquet`: the query, first positive and first negative."""
    for source in sorted(target.glob("mrtydi-v1.1-*/train.jsonl.gz")):
        out = source.with_name("train.parquet")
        if source.parent.name == "mrtydi-v1.1-combined" or (out.exists() and not force):
            continue
        columns: dict[str, list[str]] = {"query": [], "positive": [], "negative": []}
        with gzip.open(source, "rt", encoding="utf-8") as lines:
            for line in lines:
                record = json.loads(line)
                if not record["positive_passages"] or not record["negative_passages"]:
                    continue
                columns["query"].append(record["query"])
                columns["positive"].append(record["positive_passages"][0]["text"])
                columns["negative"].append(record["negative_passages"][0]["text"])
        _write_parquet(out, _TRIPLET_SCHEMA, [pa.table(columns, schema=_TRIPLET_SCHEMA)])


def _repack_reddit_year(source: Path, out: Path) -> None:
    """Stream one yearly `reddit_title_text_<year>.jsonl.gz` into `out`, title and body."""
    batches = arrow_json.open_json(
        pa.input_stream(str(source), compression="gzip"),
        read_options=arrow_json.ReadOptions(block_size=_JSON_BLOCK_BYTES),
        parse_options=arrow_json.ParseOptions(explicit_schema=_REDDIT_SCHEMA, unexpected_field_behavior="ignore"),
    )
    _write_parquet(out, _REDDIT_SCHEMA, batches)


def _repack_reddit_title_body(target: Path, force: bool) -> None:
    """Repack every year into `parquet/reddit_<year>.parquet`, one process per year since each gzip decodes serially."""
    sources, outs = [], []
    for source in sorted(target.glob("reddit_title_text_*.jsonl.gz")):
        year = source.name.removeprefix("reddit_title_text_").removesuffix(".jsonl.gz")
        out = target / "parquet" / f"reddit_{year}.parquet"
        if force or not out.exists():
            sources.append(source)
            outs.append(out)
    # Forked workers, since a script piped over stdin cannot be re-imported by spawned ones.
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as pool:
        list(pool.map(_repack_reddit_year, sources, outs))


def _repack_mmarco(target: Path, force: bool) -> None:
    """Resolve each query's first training triple into `pairs/<language>/train.parquet` for every Google translation.

    The id triples are shared by every language; each language's queries and ~8.8M-passage collection are read
    once as Arrow tables and joined by `index_in`, so no per-passage Python object is ever built.
    """
    data = target / "data"
    triples = _read_tsv(
        data / "triples.train.ids.small.tsv", {"qid": pa.int64(), "positive": pa.int64(), "negative": pa.int64()}
    )
    first = triples.group_by("qid", use_threads=False).aggregate([("positive", "first"), ("negative", "first")])
    del triples
    text_columns = {"id": pa.int64(), "text": pa.large_string()}
    for collection in sorted((data / "google" / "collections").glob("*_collection.tsv")):
        language = collection.name.removesuffix("_collection.tsv")
        out = target / "pairs" / language / "train.parquet"
        if out.exists() and not force:
            continue
        queries = _read_tsv(data / "google" / "queries" / "train" / f"{language}_queries.train.tsv", text_columns)
        passages = _read_tsv(collection, text_columns)
        pairs = pa.table(
            {
                "query": _resolve_texts(queries, first["qid"]),
                "positive": _resolve_texts(passages, first["positive_first"]),
                "negative": _resolve_texts(passages, first["negative_first"]),
            }
        )
        resolved = (
            (compute.field("query") != "") & (compute.field("positive") != "") & (compute.field("negative") != "")
        )
        _write_parquet(out, _TRIPLET_SCHEMA, [pairs.filter(resolved).cast(_TRIPLET_SCHEMA)])


_REPACKS: dict[str, Callable[[Path, bool], None]] = {
    "mr-tydi": _repack_mr_tydi,
    "reddit-title-body": _repack_reddit_title_body,
    "mmarco": _repack_mmarco,
}
"""The repack each corpus needs after its snapshot, keyed like `SOURCES`."""


def main() -> int:
    """Pull the selected Tier-4 corpora, write per-corpus sentinels, symlink into `data/`, and repack.

    Returns:
        Process exit code: `0` when every selected source completed, `1` on a selection miss or any
        per-source failure.
    """
    parser = argparse.ArgumentParser(description="Download Tier-4 contrastive-pairs corpora.")
    parser.add_argument("--data-root", default=None, help="Host data root; default `$UFORM_DATA_ROOT`, else `data/`.")
    parser.add_argument("--only", default="", help="Comma-separated subset of local dir names.")
    parser.add_argument(
        "--tier",
        default="",
        choices=["", "light", "multilingual", "heavy", "niche", "conversational"],
        help="Restrict to one tier; empty = all.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-repack even if already done.")
    parser.add_argument(
        "--repack-only", action="store_true", help="Repack already-downloaded snapshots without pulling or linking."
    )
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
    if args.data_root is None:
        from acquire_shared import data_root  # noqa: PLC0415 — lazy, so a stdin-piped repack needs no sibling module

        args.data_root = str(data_root())

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
        try:
            if args.repack_only:
                print(f"[skip] {name}: repack only", flush=True)
            elif os.path.exists(sentinel) and not args.force:
                print(f"[skip] {name}: sentinel present", flush=True)
                _link_into_data(args.data_root, name)
            else:
                print(f"[pull] {name} <- {repo}", flush=True)
                allow_patterns = _fraction_allow_patterns(repo, args.fraction)
                snapshot_download(
                    repo, repo_type="dataset", local_dir=target, token=False, allow_patterns=allow_patterns
                )
                with open(sentinel, "w") as handle:
                    handle.write(repo + "\n")
                _link_into_data(args.data_root, name)
            if name in _REPACKS:
                print(f"[repack] {name}", flush=True)
                _REPACKS[name](Path(target), args.force)
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
