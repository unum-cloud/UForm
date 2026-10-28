# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Unit checks for the acquisition-layer sizing primitives: cap_shards, index-cap, budget allocator.

Plain-script test (mirrors `test_mixtures.py`): `uv run scripts/data/test_staging.py` or `python3 ...`;
exits non-zero on the first failed assertion. Pure stdlib — imports `acquire_shared` / `acquire_clips` /
`acquire_manifest`, none of which pull a heavy dependency at import time.
"""

import math
import os

import acquire_clips
import acquire_manifest
import acquire_shared


def test_cap_shards_leading_slice() -> None:
    """`cap_shards` keeps a leading slice under the tighter of fraction / max_shards."""
    shards = [f"s{i:02d}" for i in range(10)]
    assert acquire_shared.cap_shards(shards, None, None) == shards
    assert acquire_shared.cap_shards(shards, 0.5, None) == shards[:5]
    assert acquire_shared.cap_shards(shards, None, 3) == shards[:3]
    assert acquire_shared.cap_shards(shards, 0.5, 3) == shards[:3]  # max_shards tighter
    assert acquire_shared.cap_shards(shards, 0.3, 8) == shards[:3]  # fraction tighter (ceil)
    assert acquire_shared.cap_shards(shards, 0.01, None) == shards[:1]  # at least one
    assert acquire_shared.cap_shards([], 0.5, None) == []


def test_cap_shards_matches_old_inline() -> None:
    """`cap_shards` is byte-identical to the formula the 6 migrated downloaders used to inline."""

    def old(items: list[str], fraction: float | None, max_shards: int | None) -> list[str]:
        keep = len(items)
        if fraction is not None:
            keep = min(keep, max(1, math.ceil(len(items) * fraction)))
        if max_shards is not None:
            keep = min(keep, max_shards)
        return list(items[:keep])

    items = [f"x{i}" for i in range(17)]
    for fraction in (None, 0.1, 0.25, 0.5, 0.999, 1.0):
        for max_shards in (None, 1, 4, 100):
            assert acquire_shared.cap_shards(items, fraction, max_shards) == old(items, fraction, max_shards)


def test_resolve_index_cap_precedence() -> None:
    """`--index-cap` > `$UFORM_INDEX_CAP` > the dataset default; None stays None."""
    os.environ.pop("UFORM_INDEX_CAP", None)
    assert acquire_clips.resolve_index_cap(None, None) is None
    assert acquire_clips.resolve_index_cap(None, 200_000) == 200_000
    assert acquire_clips.resolve_index_cap(5, 200_000) == 5  # cli wins over default
    os.environ["UFORM_INDEX_CAP"] = "1234"
    try:
        assert acquire_clips.resolve_index_cap(None, 200_000) == 1234  # env beats default
        assert acquire_clips.resolve_index_cap(9, 200_000) == 9  # cli beats env
    finally:
        os.environ.pop("UFORM_INDEX_CAP", None)


def test_resolve_budget_precedence() -> None:
    """`--budget-gb` > `$UFORM_DATA_BUDGET_GB` (auto free-disk default not exercised here)."""
    os.environ.pop("UFORM_DATA_BUDGET_GB", None)
    assert acquire_manifest.resolve_budget_gigabytes(6000.0) == 6000.0
    os.environ["UFORM_DATA_BUDGET_GB"] = "4321"
    try:
        assert acquire_manifest.resolve_budget_gigabytes(None) == 4321.0
        assert acquire_manifest.resolve_budget_gigabytes(7000.0) == 7000.0  # cli beats env
    finally:
        os.environ.pop("UFORM_DATA_BUDGET_GB", None)


def test_allocate_within_budget_and_full() -> None:
    """No row exceeds its full size; no modality exceeds its share (beyond fixed-full overflow)."""
    budget = 6000.0
    targets = acquire_manifest.allocate_targets(budget)
    for row in acquire_manifest.STAGING_ROWS:
        assert targets[row.dataset] <= row.estimated_full_gigabytes + 1e-6, row.dataset
    for modality, weight in acquire_manifest.MODALITY_WEIGHTS.items():
        rows = [r for r in acquire_manifest.STAGING_ROWS if r.modality == modality]
        alloc = sum(targets[r.dataset] for r in rows)
        fixed_full = sum(r.estimated_full_gigabytes for r in rows if r.weight <= 0.0)
        assert alloc <= max(budget * weight, fixed_full) + 1e-6, modality
    assert abs(sum(acquire_manifest.MODALITY_WEIGHTS.values()) - 1.0) < 1e-9


def test_allocate_scales_with_budget() -> None:
    """A bigger budget never shrinks any row's target (monotone)."""
    small = acquire_manifest.allocate_targets(3000.0)
    large = acquire_manifest.allocate_targets(9000.0)
    for row in acquire_manifest.STAGING_ROWS:
        assert large[row.dataset] >= small[row.dataset] - 1e-6, row.dataset


def test_derive_cap() -> None:
    """GB targets convert to the right `(cap_flag, value)`, clamped at full; always-full rows give None."""
    cc12m = next(r for r in acquire_manifest.STAGING_ROWS if r.dataset == "cc12m")
    flag, value = acquire_manifest.derive_cap(cc12m, cc12m.estimated_full_gigabytes / 2)
    assert flag == "--max-shards" and int(value) == round(cc12m.full_units * 0.5)
    _flag, value = acquire_manifest.derive_cap(cc12m, cc12m.estimated_full_gigabytes * 5)
    assert int(value) == cc12m.full_units  # target >= full clamps to all units

    pes2o = next(r for r in acquire_manifest.STAGING_ROWS if r.dataset == "pes2o")
    flag, value = acquire_manifest.derive_cap(pes2o, pes2o.estimated_full_gigabytes * 0.25)
    assert flag == "--fraction" and abs(float(value) - 0.25) < 1e-3

    coco = next(r for r in acquire_manifest.STAGING_ROWS if r.dataset == "coco_captions")
    assert acquire_manifest.derive_cap(coco, 10.0) is None  # cap_flag None -> always-full


def test_resolved_download_command_injection() -> None:
    """The allocator's derived cap is injected into each row's argv; fixed rows stay unchanged."""
    targets = acquire_manifest.allocate_targets(6000.0)

    cc12m = next(r for r in acquire_manifest.STAGING_ROWS if r.dataset == "cc12m")
    cmd = cc12m.resolved_download_command(targets["cc12m"])
    assert "--max-shards" in cmd
    expected = str(round(min(1.0, targets["cc12m"] / cc12m.estimated_full_gigabytes) * cc12m.full_units))
    assert cmd[cmd.index("--max-shards") + 1] == expected

    audioset = next(r for r in acquire_manifest.STAGING_ROWS if r.dataset == "audioset")
    acmd = audioset.resolved_download_command(targets["audioset"])
    assert "--fraction" in acmd and 0.0 < float(acmd[acmd.index("--fraction") + 1]) <= 1.0

    coco = next(r for r in acquire_manifest.STAGING_ROWS if r.dataset == "coco_captions")
    assert coco.resolved_download_command(targets["coco_captions"]) == list(coco.download_command)


def test_no_row_uses_unsupported_acquire_flags() -> None:
    """No manifest command passes a flag the reconciled CLIs reject (the pre-existing breakage stays fixed)."""
    banned = {"--splits", "--top10", "--tier", "--num-shards"}
    for row in acquire_manifest.STAGING_ROWS:
        if row.download_command is None:
            continue
        assert banned.isdisjoint(row.download_command), (row.dataset, row.download_command)


def test_shardable_classification() -> None:
    """`shardable` marks the rank/world-aware downloaders (acquire / acquire_clips), not the whole ones."""
    by_name = {row.dataset: row for row in acquire_manifest.STAGING_ROWS}
    assert by_name["cc12m"].shardable  # acquire (webdataset)
    assert by_name["audioset"].shardable  # acquire (hf_snapshot)
    assert by_name["howto100m"].shardable  # acquire_clips
    assert not by_name["coco_captions"].shardable  # archetype B, own argparse
    assert not by_name["didemo"].shardable  # archetype B
    assert not by_name["paq"].shardable  # download_text_pairs (multi-repo, whole)


def main() -> int:
    """Run every sizing check, printing one line per test; return non-zero on the first failure."""
    tests = [
        test_cap_shards_leading_slice,
        test_cap_shards_matches_old_inline,
        test_resolve_index_cap_precedence,
        test_resolve_budget_precedence,
        test_allocate_within_budget_and_full,
        test_allocate_scales_with_budget,
        test_derive_cap,
        test_resolved_download_command_injection,
        test_no_row_uses_unsupported_acquire_flags,
        test_shardable_classification,
    ]
    for test in tests:
        test()
        print(f"  ok   {test.__name__}")
    print("\nPASS — all staging / sizing checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
