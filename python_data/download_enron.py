# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download the Enron Email corpus (CMU FERC release), symlink into `data/`, AND repack to Parquet.

Modality: text (email pairs).  The only large real email corpus in the public domain: ~500K
messages across ~150 users' maildirs, released by FERC and hosted by CMU.  Thread structure yields
`subject -> body` and `thread -> reply` pairs, plus same-thread messages as in-batch positives and
cross-thread as hard negatives — the closest real signal to the on-device personal-email search
use case.

The corpus ships as a single ~443 MB gzipped tar (`enron_mail_20150507.tar.gz`) of a `maildir/`
tree (one file per message, RFC-822 headers + body).  This single file does both halves: it fetches
the tarball, symlinks the corpus dir, then streams the tarball member-by-member (no full extraction
to disk), parses each message with the stdlib `email` module, and emits `(subject, body)` rows: the
subject line is a terse "query" over the message body.  The repack is a light streaming projection,
so it runs inline (no separate script).  The tarball is kept (not deleted) so a wider `--max-messages`
pass can be re-run without re-fetching.

DISK / DEFERRAL: the tarball is ~443 MB compressed / ~1.4 GB extracted (~500K messages).  The repack
defaults to a bounded `--max-messages 50000` sample (enough for the parse-gate and a representative
pair set); pass `--max-messages 0` to repack the whole corpus.

VOLUME CONTROL: the source is a SINGLE ~443 MB tarball, so there is no shard cap to apply on the
download and deliberately no fake `--fraction` flag.  Corpus volume is controlled by the
`--max-messages` knob (`0` repacks all ~500K messages).

Layout: the tarball lands under `<DATA_ROOT>/enron/` with a `data/enron` symlink.

    uv run python_data/download_enron.py
    uv run python_data/download_enron.py --max-messages 0 --overwrite
    UFORM_DATA_ROOT=$UFORM_DATA_ROOT uv run python_data/download_enron.py      # env-var root
"""

from __future__ import annotations

import argparse
import email
import email.policy
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import pyarrow
import pyarrow.parquet

#: Default host data root — overridable via `UFORM_DATA_ROOT`; the symlink target `data/enron`
#: points at.
DEFAULT_DATA_ROOT = os.environ.get(
    "UFORM_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)
#: CMU's FERC Enron release tarball.
SOURCE_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

#: Local tarball filename.
TARBALL_NAME = "enron_mail_20150507.tar.gz"

#: Rows per Parquet row group — bounds writer memory.
ROW_GROUP_SIZE: int = 50_000

#: Prefixes that mark forwarded / quoted header lines; their bodies are echoed thread history, not
#: the author's own prose, so the cleaner drops everything from the first such marker onward.
_QUOTE_MARKERS: tuple[str, ...] = (
    "-" * 5 + "Original Message" + "-" * 5,
    "-" * 5 + " Forwarded by",
    "________________________________",
)


def download_enron(data_root: str) -> str:
    """Fetch the Enron tarball into the host data root and symlink into `data/`.

    Args:
        data_root: Host data root the symlink farm points at (e.g. `$UFORM_DATA_ROOT`).

    Returns:
        The corpus destination directory holding the tarball (and the repacked parquet).
    """
    destination = os.path.join(data_root, "enron")
    os.makedirs(destination, exist_ok=True)
    tarball_path = os.path.join(destination, TARBALL_NAME)

    if os.path.exists(tarball_path) and os.path.getsize(tarball_path) > 400_000_000:
        print(f"[enron] tarball already at {tarball_path}; skip download", flush=True)
    else:
        print(f"[enron] downloading {SOURCE_URL} (~443 MB)", flush=True)
        urllib.request.urlretrieve(SOURCE_URL, tarball_path)
        print(f"[enron] tarball at {tarball_path} ({os.path.getsize(tarball_path) / 1e6:.1f} MB)", flush=True)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    symlink_path = os.path.join(project_root, "data", "enron")
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        if os.path.islink(symlink_path) and os.readlink(symlink_path) == destination:
            print(f"[enron] symlink {symlink_path} -> {destination} already correct", flush=True)
            return destination
        os.unlink(symlink_path)
    os.symlink(destination, symlink_path)
    print(f"[enron] symlinked {symlink_path} -> {destination}", flush=True)
    return destination


def clean_body(raw: str) -> str:
    """Strip quoted-reply lines and forwarded-header blocks from an email body.

    Args:
        raw: The decoded plain-text payload of one message.

    Returns:
        The author's own prose: everything before the first forwarded / original-message marker,
        with leading `>`-quoted lines removed and surrounding whitespace collapsed.
    """
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if any(marker in line for marker in _QUOTE_MARKERS):
            break
        if stripped.startswith(">"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def extract_pair(message_bytes: bytes) -> tuple[str, str] | None:
    """Parse one RFC-822 message into a `(subject, body)` pair.

    Args:
        message_bytes: Raw bytes of one maildir message file.

    Returns:
        A `(subject, body)` pair, or `None` when the subject or cleaned body is empty.
    """
    parsed = email.message_from_bytes(message_bytes, policy=email.policy.default)
    subject = (parsed.get("subject") or "").strip()
    if not subject:
        return None
    try:
        payload = parsed.get_body(preferencelist=("plain",))
        raw = payload.get_content() if payload is not None else parsed.get_payload(decode=False)
    except (LookupError, ValueError, TypeError):
        return None
    if not isinstance(raw, str):
        return None
    body = clean_body(raw)
    if not body:
        return None
    return subject, body


def repack(tarball_path: Path, out_dir: Path, max_messages: int, overwrite: bool) -> int:
    """Stream the Enron tarball into a `(subject, body)` Parquet, atomically.

    Args:
        tarball_path: Path to `enron_mail_20150507.tar.gz`.
        out_dir: Directory the `enron-subject-body.parquet` shard is written under.
        max_messages: Stop after this many trainable pairs (`0` = the whole corpus).
        overwrite: Re-repack even if the target Parquet already exists.

    Returns:
        Process exit code (0 on success / skip, 1 on a missing source file).
    """
    if not tarball_path.is_file():
        print(f"repack_enron: tarball not found: {tarball_path}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "enron-subject-body.parquet"
    if out_path.exists() and not overwrite:
        print(f"repack_enron: {out_path.name} exists; pass --overwrite to rebuild")
        return 0

    schema = pyarrow.schema([("subject", pyarrow.string()), ("body", pyarrow.string())])
    tmp_path = out_path.with_suffix(".parquet.tmp")
    subjects: list[str] = []
    bodies: list[str] = []
    kept = 0
    writer = pyarrow.parquet.ParquetWriter(str(tmp_path), schema, compression="zstd")
    try:

        def flush() -> None:
            if not subjects:
                return
            writer.write_table(pyarrow.table({"subject": subjects, "body": bodies}, schema=schema))
            subjects.clear()
            bodies.clear()

        with tarfile.open(tarball_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                pair = extract_pair(handle.read())
                if pair is None:
                    continue
                subject, body = pair
                subjects.append(subject)
                bodies.append(body)
                kept += 1
                if len(subjects) >= ROW_GROUP_SIZE:
                    flush()
                if max_messages and kept >= max_messages:
                    break
        flush()
    finally:
        writer.close()
    os.replace(tmp_path, out_path)
    print(f"repack_enron: wrote {kept} (subject, body) pairs -> {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Host data root that `data/enron` symlinks to (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=50_000,
        help="Stop after this many pairs (0 = whole corpus); a bounded cap stages the parse-gate sample.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-repack even if the target parquet exists.")
    args = parser.parse_args()
    destination = download_enron(args.data_root)
    code = repack(Path(destination) / TARBALL_NAME, Path(destination), args.max_messages, args.overwrite)
    if code != 0:
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
