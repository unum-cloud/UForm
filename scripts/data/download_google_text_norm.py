# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download Google Text Normalization (Sproat & Jaitly) English data AND repack it to Parquet.

Modality: text (denoise / normalization pairs).  Written <-> spoken token pairs across the
semiotic classes PLAIN / PUNCT / DATE / CARDINAL / ORDINAL / DECIMAL / MONEY / MEASURE / TIME /
TELEPHONE / ELECTRONIC / FRACTION / ADDRESS / LETTERS / VERBATIM / DIGIT — i.e. exactly the
date / digit / currency / measure / address formatting variants the on-device email / messages /
notes fuzzy-search use case needs as multi-positive surface-robustness supervision.

The canonical Google Cloud Storage bucket the original GitHub README points at
(`storage.googleapis.com/text-normalization/en_with_types.tgz`) is now access-gated, so this
driver pulls the community parquet/CSV mirror `pavanBuduguppa/asr_inverse_text_normalization`,
whose `en_train.csv` is the same Sproat token table with the full
`(sentence_id, token_id, class, before, after)` schema.  `before` is the WRITTEN form
(`"2006"`, `"$3.00"`) and `after` is the SPOKEN form (`"two thousand six"`, `"three dollars"`).

This single file does both halves: it pulls the ~325 MB CSV, symlinks the corpus dir, then projects
each token row to `(written=before, spoken=after, semiotic_class=class)` and writes a single
GPU-readable Parquet sibling.  The repack is a light CSV-to-columns projection, so it runs inline (no
separate script).  The CSV is kept (not deleted) so a re-run can rebuild the parquet without re-fetch.

VOLUME CONTROL: the source is a SINGLE `en_train.csv`, so there is no shard cap on the download and
deliberately no fake `--fraction` flag; `--max-records N` caps the repacked token pairs (0 = the whole
table).  Identity rows are dropped ON DEVICE by the registry extractor, not here.

Layout: the source CSV lands under `<DATA_ROOT>/google-text-norm/en_train.csv` with a
`data/google-text-norm` symlink.

    uv run scripts/data/download_google_text_norm.py
    uv run scripts/data/download_google_text_norm.py --overwrite
    UFORM_DATA_ROOT=$UFORM_DATA_ROOT uv run scripts/data/download_google_text_norm.py   # env-var root
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import pyarrow
import pyarrow.parquet
from acquire_shared import data_root
from huggingface_hub import hf_hub_download

#: Default host data root — overridable via `UFORM_DATA_ROOT`; the symlink target
#: `data/google-text-norm` points at.
DEFAULT_DATA_ROOT = str(data_root())
#: Community mirror carrying the Sproat English token table with the full `class` / `after` columns.
SOURCE_REPO = "pavanBuduguppa/asr_inverse_text_normalization"

#: The train CSV is the only split that ships the `class` + `after` (spoken) columns the pairs need;
#: the test CSV is `before`-only.
SOURCE_FILE = "en_train.csv"

#: Rows per Parquet row group — bounds writer memory on the ~10M-token table.
ROW_GROUP_SIZE: int = 200_000

#: The semiotic-class sentence-boundary marker rows the Sproat format inserts; not a real token pair.
EOS_MARKER: str = "<eos>"


def download_google_text_norm(data_root: str) -> str:
    """Pull the English text-normalization CSV into the host data root and symlink into `data/`.

    Args:
        data_root: Host data root the symlink farm points at (e.g. `$UFORM_DATA_ROOT`).

    Returns:
        The corpus destination directory holding `en_train.csv` (and the repacked parquet).
    """
    destination = os.path.join(data_root, "google-text-norm")
    os.makedirs(destination, exist_ok=True)

    print(f"[google-text-norm] downloading {SOURCE_FILE} from {SOURCE_REPO}", flush=True)
    cached = hf_hub_download(repo_id=SOURCE_REPO, filename=SOURCE_FILE, repo_type="dataset")
    target = os.path.join(destination, SOURCE_FILE)
    if os.path.abspath(cached) != os.path.abspath(target):
        # `hf_hub_download` returns a path inside the HF cache; mirror it under the data root so the
        # repack and the symlink farm see a stable, non-cache-managed location.
        shutil.copyfile(cached, target)
    print(f"[google-text-norm] CSV at {target} ({os.path.getsize(target) / 1e6:.1f} MB)", flush=True)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    symlink_path = os.path.join(project_root, "data", "google-text-norm")
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        if os.path.islink(symlink_path) and os.readlink(symlink_path) == destination:
            print(f"[google-text-norm] symlink {symlink_path} -> {destination} already correct", flush=True)
            return destination
        os.unlink(symlink_path)
    os.symlink(destination, symlink_path)
    print(f"[google-text-norm] symlinked {symlink_path} -> {destination}", flush=True)
    return destination


def repack(csv_path: Path, overwrite: bool, max_records: int) -> int:
    """Project the text-normalization CSV to a `(written, spoken, semiotic_class)` Parquet, atomically.

    Args:
        csv_path: Path to `en_train.csv`.
        overwrite: Re-repack even if the `.parquet` sibling already exists.
        max_records: Stop after this many token pairs (`0` = the whole table); a small cap stages a
            quick parse-gate sample without materialising the full ~10M-row projection.

    Returns:
        Process exit code (0 on success / skip, 1 on a missing source file).
    """
    if not csv_path.is_file():
        print(f"repack_google_text_norm: source not found: {csv_path}", file=sys.stderr)
        return 1
    out_path = csv_path.with_suffix(".parquet")
    if out_path.exists() and not overwrite:
        print(f"repack_google_text_norm: {out_path.name} exists; pass --overwrite to rebuild")
        return 0

    schema = pyarrow.schema(
        [
            ("written", pyarrow.string()),
            ("spoken", pyarrow.string()),
            ("semiotic_class", pyarrow.string()),
        ]
    )
    tmp_path = out_path.with_suffix(".parquet.tmp")
    written_col: list[str] = []
    spoken_col: list[str] = []
    class_col: list[str] = []
    kept = 0
    writer = pyarrow.parquet.ParquetWriter(str(tmp_path), schema, compression="zstd")
    try:

        def flush() -> None:
            if not written_col:
                return
            writer.write_table(
                pyarrow.table(
                    {"written": written_col, "spoken": spoken_col, "semiotic_class": class_col},
                    schema=schema,
                )
            )
            written_col.clear()
            spoken_col.clear()
            class_col.clear()

        with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                semiotic_class = (row.get("class") or "").strip()
                before = row.get("before")
                after = row.get("after")
                if before is None or after is None:
                    continue
                # Sproat inserts `<eos>` boundary rows between sentences; they are not token pairs.
                if before == EOS_MARKER or after == EOS_MARKER:
                    continue
                # `<self>` is the format's "spoken form == written form" shorthand; expand it so the
                # downstream identity-filter sees a real string pair rather than the sentinel.
                if after == "<self>":
                    after = before
                written_col.append(before)
                spoken_col.append(after)
                class_col.append(semiotic_class)
                kept += 1
                if len(written_col) >= ROW_GROUP_SIZE:
                    flush()
                if max_records and kept >= max_records:
                    break
        flush()
    finally:
        writer.close()
    os.replace(tmp_path, out_path)
    print(f"repack_google_text_norm: wrote {kept} (written, spoken, class) token rows -> {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Host data root that `data/google-text-norm` symlinks to (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-repack even if the .parquet sibling exists.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Stop after this many token pairs (0 = whole table); a small cap stages a parse-gate sample.",
    )
    args = parser.parse_args()
    destination = download_google_text_norm(args.data_root)
    code = repack(Path(destination) / SOURCE_FILE, args.overwrite, args.max_records)
    if code != 0:
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
