# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cudf-cu12",
#     "numpy",
# ]
# ///
"""Compute the unigram entropy of masked bytes in the actual MLM training dataset.

Modality: text. Diagnostic answering whether the 3.27 plateau is the same as the unigram entropy
of the bytes the model is asked to predict: if yes, the model has fully learned unigram and is
bottlenecked by something else (capacity, recipe, architecture); if no, there is still slack to
descend through unigram. The method reads the resolved parquet globs of the MLM text mixture
directly (cuDF `read_parquet` on each corpus's `text_field`), rebuilds the trainer's byte sequences
and span mask with the same arguments, tallies the byte frequency of targets at masked positions,
and computes `-sum(p · log p)` — the cross-entropy a unigram model achieves at those positions.
Span masking is content-independent, so the masked-target byte distribution equals the corpus byte
distribution: this reproduces the loader's floor without the deleted `TextMaskLoader`.

Usage:
    uv run scripts/data/data_unigram_floor.py --batches 20 --batch_size 64 --max_seq 1024
"""

from __future__ import annotations

import argparse
import glob as globbing
import math
import sys
from collections.abc import Iterator

import numpy as np
from mixtures import resolve_mixture


def _document_stream(resolved_glob: str, text_field: str) -> Iterator[str]:
    """Lazily yield the `text_field` documents of one corpus's parquet shards, read via cuDF.

    Shards are opened one at a time so a large mixture never materialises fully on the GPU; only the
    columns the floor needs (`text_field`) are read.
    """
    import cudf  # noqa: PLC0415 — GPGPU dep; imported lazily so `--help` and syntax checks stay light

    for shard_path in sorted(globbing.glob(resolved_glob, recursive=True)):
        frame = cudf.read_parquet(shard_path, columns=[text_field])
        for text in frame[text_field].to_arrow().to_pylist():
            if text:
                yield str(text)


def _next_document(streams: list[Iterator[str]], exhausted: list[bool], preferred: int) -> str | None:
    """Pull the next document, preferring corpus `preferred` and falling back over exhausted streams."""
    order = [preferred, *(i for i in range(len(streams)) if i != preferred)]
    for i in order:
        if exhausted[i]:
            continue
        try:
            return next(streams[i])
        except StopIteration:
            exhausted[i] = True
    return None


def _span_mask(length: int, mask_prob: float, mask_span_len: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean loss mask over `length` bytes: ~`mask_prob` active, in Poisson-length spans (mean `mask_span_len`).

    Matches the trainer's span-masking shape; because which bytes are masked is content-independent, the
    masked-target byte distribution — hence the unigram floor — is unchanged by the exact span layout.
    """
    mask = np.zeros(length, dtype=bool)
    target = int(round(length * mask_prob))
    if target <= 0:
        return mask
    guard = 0
    guard_limit = 10 * length + 10
    while int(mask.sum()) < target and guard < guard_limit:
        guard += 1
        start = int(rng.integers(0, length))
        span = max(1, int(rng.poisson(mask_span_len)))
        mask[start : start + span] = True
    return mask


def main() -> int:
    """Tally masked-target byte frequencies over N batches and print the unigram entropy floor.

    Returns:
        Process exit code: `0` on success, `1` when no masked tokens were observed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-span-len", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # `resolve_mixture("text_base")` yields `(resolved_glob, weight, text_field)` — the same tuple the
    # deleted `TextMaskLoader` consumed; we now read those globs ourselves as the byte source.
    mixture = resolve_mixture("text_base")
    rng = np.random.default_rng(args.seed)
    weights = np.array([weight for _glob, weight, _field in mixture], dtype=np.float64)
    weights = weights / weights.sum()
    streams = [_document_stream(resolved_glob, text_field) for resolved_glob, _weight, text_field in mixture]
    exhausted = [False] * len(streams)

    vocab_size = 258  # 256 bytes + PAD + MASK
    counts = np.zeros(vocab_size, dtype=np.int64)
    total_targets = 0
    total_positions = 0
    total_mask_positions = 0

    for _b in range(args.batches):
        for _s in range(args.batch_size):
            if all(exhausted):
                break
            preferred = int(rng.choice(len(streams), p=weights))
            text = _next_document(streams, exhausted, preferred)
            if text is None:
                continue
            sequence = np.frombuffer(text.encode("utf-8", "ignore")[: args.max_seq], dtype=np.uint8).astype(np.int64)
            if sequence.size == 0:
                continue
            mask_active = _span_mask(sequence.size, args.mask_prob, args.mask_span_len, rng)
            targets_at_mask = sequence[mask_active]
            # filter out any garbage tokens >= vocab (shouldn't happen but be safe)
            valid = targets_at_mask < vocab_size
            targets_at_mask = targets_at_mask[valid]
            np.add.at(counts, targets_at_mask, 1)
            total_targets += int(targets_at_mask.size)
            total_positions += int(sequence.size)
            total_mask_positions += int(mask_active.sum())

    print(f"\nDataset unigram floor: text_base, {args.batches} batches x {args.batch_size} x {args.max_seq}")
    print(f"  total positions seen         : {total_positions:,}")
    print(f"  total mask-active positions  : {total_mask_positions:,}  ({total_mask_positions / total_positions:.1%})")
    print(f"  total target tokens counted  : {total_targets:,}")
    if total_targets == 0:
        print("ERROR: no masked tokens in any batch?")
        return 1

    p = counts / counts.sum()
    nonzero = p > 0
    H = float(-(p[nonzero] * np.log(p[nonzero])).sum())
    H_bits = H / math.log(2.0)
    H_uniform = math.log(vocab_size)
    H_uniform_bits = H_uniform / math.log(2.0)

    print()
    print(f"  Unigram entropy (nats)       : {H:.4f}")
    print(f"  Unigram entropy (BPB)        : {H_bits:.4f}")
    print(f"  Random (log {vocab_size})           : {H_uniform:.4f} nats / {H_uniform_bits:.4f} BPB")
    print("  Our current plateau          : ~3.27 nats / ~4.72 BPB")
    print(f"  Gap (current - unigram)      : {3.27 - H:+.4f} nats")
    print()
    print("  Top-10 most common targets (byte_id, count, freq):")
    top_idx = np.argsort(counts)[::-1][:10]
    for rank, idx in enumerate(top_idx):
        repr_byte = chr(idx) if 32 <= idx < 127 and idx not in (256, 257, 258) else f"<id{idx}>"
        print(f"    #{rank + 1:<2}  id={idx:3d}  count={counts[idx]:>10,d}  freq={p[idx]:.4f}  '{repr_byte}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
