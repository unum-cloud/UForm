"""Build `msmarco_triplet50`: MS MARCO `(query, positive, 50 hard negatives)` triplets, all text.

`msmarco_triplet50` is NOT a single HuggingFace pull — it is a DERIVED artifact joined from three
configs of `sentence-transformers/msmarco`, all keyed by opaque string ids:

  * `labeled-list` — one row per query: `query_id`, a rank-ordered `doc_ids` list of retrieved
    candidate passages (best-first), and a parallel `labels` list (`1` = relevant positive,
    `0` = a hard negative, i.e. a top-retrieved passage that is NOT the gold answer). This IS the
    offline hard-negative mining output; no BM25/dense retrieval is re-run here.
  * `corpus` — the passage id -> passage text table (`passage_id`, `passage`).
  * `queries` — the query id -> query text table (`query_id`, `query`).

The sibling `data/msmarco/triplet/*.parquet` (config `triplets`) is id-only (`query_id` /
`positive_id` / `negative_id`) and carries a SINGLE negative, so it cannot back the 50-negative
entry; `labeled-list` is the same repo's 50-way source. For each `labeled-list` row this selects the
first `label == 1` doc as the positive and the leading `--max-negatives` (50) `label == 0` docs, in
list order (best-first preserved), then resolves every id to text through the corpus / queries tables.

The produced schema matches the `msmarco_triplet50` mixture column contract
(`mixtures.dataset_columns` -> `query` / `positive` / `negative`): `query` (string), `positive`
(string), and `negative` a `list<string>` of up to 50 best-first hard negatives — the same
`list<string>` packing the repo uses elsewhere (COCO / MeViS `captions`), so the paired loader's
`take_list_prefix(K)` surfaces the K hardest negatives of one anchor as a K-degree repel CSR. One
output parquet per source `labeled-list` shard lands at `data/msmarco/triplet-50/<shard>.parquet`.

Parallelism follows the acquisition convention (`sharding`): one Slurm array task is one
`(rank, world)`; each rank pulls a disjoint stride of the capped `labeled-list` shard list and
commits through the atomic rename, so `already_done` skips a resumed shard. The corpus / queries
resolution tables are streamed once and filtered to only the passage / query ids the assigned shards
reference, so memory stays bounded by the selected shards rather than the full 8.8M-passage corpus.

Usage:
    uv run scripts/data/download_msmarco.py                    # all labeled-list shards
    uv run scripts/data/download_msmarco.py --max-shards 1     # one shard (a tiny smoke build)
    uv run scripts/data/download_msmarco.py --fraction 0.25    # leading quarter of the shards

File: scripts/data/download_msmarco.py
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

import argparse
from pathlib import Path

import mixtures
import pyarrow as pa
import pyarrow.compute as compute
import pyarrow.parquet as parquet
from acquire_shared import cap_shards, data_root
from huggingface_hub import hf_hub_download, list_repo_files
from sharding import already_done, atomic_write, partition, resolve_rank_world

#: The single upstream repo every config is joined from.
_REPO_ID = "sentence-transformers/msmarco"
#: The mixture / catalog name whose column contract the produced shards must satisfy.
_DATASET_NAME = "msmarco_triplet50"
#: Output subdirectory under the data root — the `msmarco_triplet50`
#: glob is `data/msmarco/triplet-50/*.parquet`.
_OUT_SUBDIR = ("msmarco", "triplet-50")


def _labeled_list_shards() -> list[str]:
    """The sorted `labeled-list/*.parquet` shard ids — the stable order the rank partition slices."""
    files = list_repo_files(_REPO_ID, repo_type="dataset")
    return sorted(name for name in files if name.startswith("labeled-list/") and name.endswith(".parquet"))


def _config_files(prefix: str) -> list[str]:
    """The sorted parquet shard ids of a repo config (`corpus/` or `queries/`) — the resolution tables."""
    files = list_repo_files(_REPO_ID, repo_type="dataset")
    return sorted(name for name in files if name.startswith(prefix) and name.endswith(".parquet"))


def _select_triplet(doc_ids: list[str], labels: list[int], max_negatives: int) -> tuple[str | None, list[str]]:
    """Split one `labeled-list` row into `(positive_id, negative_ids)` — first relevant doc, then top-K hard negatives.

    `doc_ids` is best-first (retrieval rank order) and `labels` is parallel (`1` relevant, `0` hard negative).
    The positive is the first `label == 1` doc; the negatives are the leading `max_negatives` `label == 0` docs
    with their best-first order preserved. Returns `(None, [])` for a row with no positive (dropped upstream).
    """
    positive_id: str | None = None
    negative_ids: list[str] = []
    for doc_id, label in zip(doc_ids, labels):
        if label == 1:
            if positive_id is None:
                positive_id = doc_id
        elif len(negative_ids) < max_negatives:
            negative_ids.append(doc_id)
    return positive_id, negative_ids


def _plan_shard(shard_path: Path, max_negatives: int) -> tuple[list[tuple[str, str, list[str]]], set[str], set[str]]:
    """Read one `labeled-list` shard into `(query_id, positive_id, negative_ids)` rows plus the ids they reference.

    First pass of the two-pass join: keeps the per-row id triplets and accumulates the passage / query id sets the
    corpus / queries tables must resolve, so the second pass filters those tables to only what this shard needs.
    """
    table = parquet.read_table(shard_path, columns=["query_id", "doc_ids", "labels"])
    query_ids = table.column("query_id").to_pylist()
    doc_id_lists = table.column("doc_ids").to_pylist()
    label_lists = table.column("labels").to_pylist()

    rows: list[tuple[str, str, list[str]]] = []
    needed_passages: set[str] = set()
    needed_queries: set[str] = set()
    for query_id, doc_ids, labels in zip(query_ids, doc_id_lists, label_lists):
        positive_id, negative_ids = _select_triplet(doc_ids, labels, max_negatives)
        if positive_id is None or not negative_ids:
            continue  # a query with no gold positive or no mined negative carries no triplet supervision
        rows.append((query_id, positive_id, negative_ids))
        needed_queries.add(query_id)
        needed_passages.add(positive_id)
        needed_passages.update(negative_ids)
    return rows, needed_passages, needed_queries


def _resolve_texts(config_files: list[str], id_column: str, text_column: str, needed_ids: set[str]) -> dict[str, str]:
    """Stream a resolution config's shards and build an `id -> text` map restricted to `needed_ids`.

    Filters each shard to the referenced ids before materializing (`compute.is_in`), so the map holds only the
    passages / queries the assigned `labeled-list` shards cite rather than the whole 8.8M-passage corpus.
    """
    if not needed_ids:
        return {}
    id_array = pa.array(sorted(needed_ids), type=pa.string())
    resolved: dict[str, str] = {}
    for config_file in config_files:
        shard = Path(hf_hub_download(_REPO_ID, config_file, repo_type="dataset"))
        table = parquet.read_table(shard, columns=[id_column, text_column])
        keep = table.filter(compute.is_in(table.column(id_column), value_set=id_array))
        for identifier, text in zip(keep.column(id_column).to_pylist(), keep.column(text_column).to_pylist()):
            resolved[identifier] = text
        if len(resolved) == len(needed_ids):
            break  # every referenced id resolved; no need to scan the remaining shards
    return resolved


def _materialize(
    rows: list[tuple[str, str, list[str]]], passages: dict[str, str], queries: dict[str, str]
) -> pa.Table:
    """Resolve the id triplets to text and build the canonical `(query, positive, negative)` parquet table.

    Drops a row whose query or positive id failed to resolve, and drops any individual negative id missing from
    the corpus; `negative` is a `list<string>` of the surviving best-first hard negatives (rows left with none
    after resolution are dropped, keeping the multi-negative contract non-vacuous).
    """
    query_texts: list[str] = []
    positive_texts: list[str] = []
    negative_texts: list[list[str]] = []
    for query_id, positive_id, negative_ids in rows:
        query = queries.get(query_id)
        positive = passages.get(positive_id)
        if query is None or positive is None:
            continue
        negatives = [passages[negative_id] for negative_id in negative_ids if negative_id in passages]
        if not negatives:
            continue
        query_texts.append(query)
        positive_texts.append(positive)
        negative_texts.append(negatives)
    return pa.table(
        {
            "query": pa.array(query_texts, type=pa.string()),
            "positive": pa.array(positive_texts, type=pa.string()),
            "negative": pa.array(negative_texts, type=pa.list_(pa.string())),
        }
    )


def _assert_columns(produced_columns: set[str]) -> None:
    """Every column the mixture JSONs declare for `msmarco_triplet50` must be present in the produced shard."""
    declared = mixtures.dataset_columns(_DATASET_NAME)
    missing = declared - produced_columns
    if missing:
        raise AssertionError(
            f"download_msmarco {_DATASET_NAME!r}: mixture-declared columns missing from the produced shard — "
            f"missing={sorted(missing)}; declared={sorted(declared)}; produced={sorted(produced_columns)}. "
            "Fix the transform or the mixture JSON so the column contract holds."
        )


def main() -> int:
    """Join `labeled-list` against `corpus` / `queries`, select 50 hard negatives, and commit triplet-50 shards."""
    parser = argparse.ArgumentParser(description="Build MS MARCO 50-hard-negative text triplets (msmarco_triplet50).")
    parser.add_argument("--fraction", type=float, default=None, help="leading-slice volume cap in (0, 1]")
    parser.add_argument("--max-shards", type=int, default=None, help="hard cap on the number of labeled-list shards")
    parser.add_argument("--max-negatives", type=int, default=50, help="hard negatives kept per query (best-first)")
    parser.add_argument("--rank", type=int, default=None, help="usually env-derived; explicit for local N-way runs")
    parser.add_argument("--world", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-fetch shards whose artifact already exists")
    args = parser.parse_args()

    rank, world = resolve_rank_world(args.rank, args.world)
    out_dir = data_root().joinpath(*_OUT_SUBDIR)

    shards = _labeled_list_shards()
    capped = cap_shards(shards, args.fraction, args.max_shards)
    mine = partition(capped, rank, world)

    corpus_files = _config_files("corpus/")
    queries_files = _config_files("queries/")

    written = 0
    for source in mine:
        final = out_dir / Path(source).name  # e.g. labeled-list/train-00000-of-00015.parquet -> train-00000-...parquet
        if already_done(final) and not args.force:
            continue

        shard = Path(hf_hub_download(_REPO_ID, source, repo_type="dataset"))
        rows, needed_passages, needed_queries = _plan_shard(shard, args.max_negatives)
        passages = _resolve_texts(corpus_files, "passage_id", "passage", needed_passages)
        queries = _resolve_texts(queries_files, "query_id", "query", needed_queries)
        table = _materialize(rows, passages, queries)

        _assert_columns(set(table.column_names))
        with atomic_write(final, rank=rank) as temporary:
            parquet.write_table(table, temporary, compression="zstd")
        print(f"  {Path(source).name}: {table.num_rows} triplets (from {len(rows)} candidate rows)", flush=True)
        written += 1

    print(f"download_msmarco {_DATASET_NAME}: rank {rank}/{world} wrote {written} shards -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
