# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Balanced multi-modal staging manifest for the ~3 TB real corpus on `$UFORM_DATA_ROOT`.

This module is the single declarative source the staging driver
(`python_data.stage_balanced_corpus`) reads. It carries one `StagingRow` per dataset the owner
wants staged, grouped into four modalities — text / image / video / audio — each targeted at
`MODALITY_TARGET_GIGABYTES` (~750 GB) so the grand total lands near 3 TB. Each row's shard glob is
sourced from the stdlib-only JSON `python_data.mixtures` reader (no cudf / cupy), so the driver still
runs under a lightweight `uv run` environment while dropping the previously duplicated glob strings.

Sizing model.
`estimated_full_gigabytes` is the size the FULL upstream dataset would occupy on `$UFORM_DATA_ROOT` in the
registry's on-disk schema (decoded media bytes inside parquet for the media corpora). Figures are
approximate, taken from each dataset's HuggingFace card / the downloader docstring, with the
assumption noted inline. `fraction` is the share of that full dataset this manifest stages, chosen so
each modality sums to ~750 GB: it stays in the owner's 0.35-0.40 band where a modality has room, and
is capped DOWN for the giant text/web/audio/video sources that would otherwise dwarf their modality.
`estimated_download_gigabytes = fraction * estimated_full_gigabytes`.

Fraction mechanism.
`fraction_mechanism` records HOW the fraction is realised on the wire for each downloader:

  - "fraction"      — downloader has a true `--fraction` flag (shards * fraction).
  - "num-shards"    — downloader caps by `--num-shards N` (compile_ir, post_ocr).
  - "max-shards"    — downloader caps by `--max-shards N` (audioset).
  - "shards"        — fineweb2 per-language depth `--shards N` (+ `--fraction` over languages).
  - "language-subset" — starcoderdata `--top10` / `--only` language gate.
  - "pre-staged"    — already fully on `$UFORM_DATA_ROOT`; no download script exists (and none needed).
  - "snapshot-full" — downloader pulls the WHOLE repo: NO fraction knob today. `flag` says so; to
                      hit the per-modality budget without over-pulling, a `--max-shards`-style knob
                      must be added (see `requires_new_knob`).
  - "no-downloader" — NO download script exists for this registry dataset yet (gap; `flag` says so).

Every row whose `requires_new_knob` is True or whose `fraction_mechanism` is "no-downloader" is a
BLOCKER for reaching the 750 GB target cleanly and is surfaced by the driver's balance table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mixtures

#: Repo root a relative `glob_override` is anchored under (mirrors `mixtures._resolve_glob`).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Per-modality on-disk budget; the four modalities each target this, summing to ~3 TB.
MODALITY_TARGET_GIGABYTES: float = 750.0

#: Host data root every registry `data/<name>` glob is anchored under.
DATA_ROOT: str = os.environ.get("UFORM_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))

#: The four balanced modalities.
MODALITIES: tuple[str, ...] = ("text", "image", "video", "audio")


@dataclass(frozen=True, slots=True)
class StagingRow:
    """One dataset's staging plan: what to pull, how much, and with which command.

    Attributes:
        dataset: Dataset name (matches a `python_data.mixtures` dataset entry); its glob is read from there.
        modality: One of `MODALITIES`.
        fraction: Share of the full dataset to stage, in (0, 1].
        estimated_full_gigabytes: Approx size of the FULL dataset on disk (see module docstring).
        estimated_download_gigabytes: `fraction * estimated_full_gigabytes`, the planned on-disk add.
        fraction_mechanism: How the fraction is realised (see module docstring).
        download_command: Argv the driver runs to stage the dataset, or `None` when no script exists.
        repack_command: Shell command to repack raw artifacts into the registry schema, or `None`.
        requires_new_knob: True when the downloader lacks a fraction/shard knob needed for the budget.
        flag: Human-readable blocker / caveat surfaced in the balance table, or "" when clean.
        glob_override: Repo-relative shard glob for rows no mixture references (synthetic rollups or
            eval/smoke corpora); when set it supplies `registry_glob` in place of the mixture lookup.
    """

    dataset: str
    modality: str
    fraction: float
    estimated_full_gigabytes: float
    estimated_download_gigabytes: float
    fraction_mechanism: str
    download_command: list[str] | None
    repack_command: str | None = None
    requires_new_knob: bool = False
    flag: str = ""
    glob_override: str | None = None

    @property
    def registry_glob(self) -> str:
        """Absolute shard glob the driver checks presence against, sourced from the mixture JSONs.

        Falls back to `glob_override` (anchored under the repo root) for rows no mixture references —
        the synthetic present-corpus rollup and the eval/smoke datasets not carried by any mixture.
        """
        if self.glob_override is not None:
            return self.glob_override if Path(self.glob_override).is_absolute() else str(PROJECT_ROOT / self.glob_override)
        return mixtures.dataset_glob(self.dataset)


def _uv(*args: str) -> list[str]:
    """Build a `uv run python_data/<script>` argv from its trailing arguments."""
    return ["uv", "run", *args]


#: The balanced manifest. Sizes are approximate (HuggingFace cards / downloader docstrings); the
#: per-modality sums are tuned to ~750 GB each (printed by the driver's balance table).
STAGING_ROWS: list[StagingRow] = [
    # Text modality, ~750 GB target.
    # Text is the dominant source and is already ~700 GB staged on $UFORM_DATA_ROOT, so its fractions are
    # capped FAR below 0.35 for the web/code giants (fineweb-2, starcoderdata, compile_ir) — staging
    # 35% of compile_ir alone (~950 GB) would blow the whole budget. The web/wiki/QA corpora already
    # present are kept (the driver skips them as present); the only NEW text pulls are a code slice.
    StagingRow(
        dataset="fineweb_2",
        modality="text",
        fraction=0.40,
        estimated_full_gigabytes=840.0,  # curated 42-language superset at full per-language depth.
        estimated_download_gigabytes=336.0,  # 42 langs x 4 shards already staged (~40% depth design).
        fraction_mechanism="shards",
        download_command=_uv("python_data/download_fineweb2.py", "--data-root", DATA_ROOT, "--shards", "4"),
    ),
    StagingRow(
        dataset="finewiki_i18n",
        modality="text",
        fraction=1.00,
        estimated_full_gigabytes=185.0,  # 325-language FineWiki, fully staged.
        estimated_download_gigabytes=185.0,
        fraction_mechanism="pre-staged",
        download_command=None,
        flag="pre-staged: no download script (FineWiki was provisioned out of band); already on disk.",
    ),
    StagingRow(
        dataset="fineweb_en",
        modality="text",
        fraction=1.00,
        estimated_full_gigabytes=21.0,  # FineWeb 100BT English sample.
        estimated_download_gigabytes=21.0,
        fraction_mechanism="pre-staged",
        download_command=None,
        flag="pre-staged: no download script; already on disk.",
    ),
    StagingRow(
        dataset="pes2o",
        modality="text",
        fraction=0.40,
        estimated_full_gigabytes=80.0,  # peS2o v3 train, full (136 zstd shards).
        estimated_download_gigabytes=32.0,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_pes2o.py", "--out-dir", f"{DATA_ROOT}/peS2o", "--fraction", "0.40"),
        flag="download_pes2o.py pulls + repacks (self-parallel) 54 of 136 v3-train zst shards (~32 GB) to parquet.",
    ),
    StagingRow(
        dataset="s2orc_abstract",
        glob_override="data/s2orc/abstract-citation-pair/train-*.parquet",  # no s2orc_abstract entry in any mixture
        modality="text",
        fraction=1.00,
        estimated_full_gigabytes=31.0,
        estimated_download_gigabytes=31.0,
        fraction_mechanism="pre-staged",
        download_command=None,
        flag="pre-staged: no download script; already on disk.",
    ),
    StagingRow(
        dataset="paq",
        modality="text",
        fraction=0.40,
        estimated_full_gigabytes=70.0,
        estimated_download_gigabytes=28.0,
        fraction_mechanism="fraction",
        download_command=_uv(
            "python_data/download_text_pairs.py", "--only", "paq", "--fraction", "0.40", "--data-root", DATA_ROOT
        ),
    ),
    StagingRow(
        dataset="wikianswers",
        modality="text",
        fraction=0.40,
        estimated_full_gigabytes=80.0,
        estimated_download_gigabytes=32.0,
        fraction_mechanism="fraction",
        download_command=_uv(
            "python_data/download_text_pairs.py",
            "--only",
            "wikianswers-duplicates",
            "--fraction",
            "0.40",
            "--data-root",
            DATA_ROOT,
        ),
    ),
    StagingRow(
        dataset="starcoderdata",
        modality="text",
        fraction=0.05,
        estimated_full_gigabytes=783.0,  # full 86-language BigCode slice.
        estimated_download_gigabytes=40.0,  # --top10 high-volume languages only.
        fraction_mechanism="language-subset",
        download_command=_uv("python_data/download_starcoderdata.py", "--top10", "--data-root", DATA_ROOT),
    ),
    StagingRow(
        dataset="compile_ir",
        glob_override="data/compile-ir/data/**/*.parquet",  # no compile_ir entry in any mixture
        modality="text",
        fraction=0.012,
        estimated_full_gigabytes=2700.0,  # full ComPile LLVM-IR corpus.
        estimated_download_gigabytes=32.0,  # first 50 of ~2700 shards.
        fraction_mechanism="num-shards",
        download_command=_uv("python_data/download_compile_ir.py", "--num-shards", "50", "--data-root", DATA_ROOT),
    ),
    StagingRow(
        dataset="poj_104",
        modality="text",
        fraction=1.00,
        estimated_full_gigabytes=0.1,  # 104 C++ problems x ~500 solutions, ~50 MB convert-parquet.
        estimated_download_gigabytes=0.1,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_poj104.py", "--data-root", DATA_ROOT),
        flag="code equivalence classes: each problem's ~500 solutions are mutual positives (download_poj104.py).",
    ),
    StagingRow(
        dataset="the_vault",
        modality="text",
        fraction=0.20,
        estimated_full_gigabytes=40.0,  # full tier across 10 languages; the small tier is ~2 GB.
        estimated_download_gigabytes=2.0,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_the_vault.py", "--tier", "small", "--data-root", DATA_ROOT),
        flag="multilingual docstring<->code, 10 languages; small tier staged (--tier medium/full for more).",
    ),
    StagingRow(
        dataset="xlcost",
        modality="text",
        fraction=1.00,
        estimated_full_gigabytes=0.5,  # 7-language program-level convert-parquet, ~0.5 GB.
        estimated_download_gigabytes=0.5,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_xlcost.py", "--data-root", DATA_ROOT),
        flag="cross-lingual NL<->code across 7 languages (download_xlcost.py).",
    ),
    StagingRow(
        dataset="civil_comments",
        modality="text",
        fraction=0.40,
        estimated_full_gigabytes=0.6,  # ~1.9M comments, text-only parquet.
        estimated_download_gigabytes=0.24,
        fraction_mechanism="fraction",
        download_command=_uv(
            "python_data/download_civil_comments.py", "--out-dir", f"{DATA_ROOT}/civil_comments", "--fraction", "0.40"
        ),
        flag="defensive safety cluster (CC0, commercial-OK); flat text+label, label==1/0 = unsafe/safe split.",
    ),
    StagingRow(
        dataset="toxigen",
        modality="text",
        fraction=1.00,
        estimated_full_gigabytes=0.05,  # ~260K rows text-only.
        estimated_download_gigabytes=0.05,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_toxigen.py", "--out-dir", f"{DATA_ROOT}/toxigen"),
        flag="defensive safety cluster; ToxiGen is RESEARCH-USE (no SPDX) — vet before commercial; "
        "implicit toxic/benign + target_group; label==1/0 = unsafe/safe split.",
    ),
    StagingRow(
        dataset="text_pairs_present_rollup",
        modality="text",
        # Rollup of the small contrastive-pair corpora already staged: msmarco, mmarco, natural_questions,
        # eli5, arxiv, code_search_net, hackernews, commitpackft, stackexchange_*, all-nli, quora,
        # gooaq, hotpotqa, mr-tydi. Each is < ~15 GB and already on $UFORM_DATA_ROOT; counted as one row.
        glob_override="data/msmarco/triplet/*.parquet",  # synthetic rollup; not a catalog row
        fraction=1.00,
        estimated_full_gigabytes=13.0,
        estimated_download_gigabytes=13.0,
        fraction_mechanism="pre-staged",
        download_command=None,
        flag="rollup of small pair corpora already staged (msmarco/eli5/arxiv/all-nli/quora/...).",
    ),
    # Image modality, ~750 GB target.
    StagingRow(
        dataset="cc12m",
        modality="image",
        fraction=0.35,
        estimated_full_gigabytes=1500.0,  # 12M image+caption WebDataset, ~2176 tar shards.
        estimated_download_gigabytes=525.0,
        fraction_mechanism="max-shards",
        download_command=_uv("python_data/download_cc12m.py", "--out-dir", f"{DATA_ROOT}/cc12m", "--max-shards", "762"),
        repack_command=(
            f"find {DATA_ROOT}/cc12m -name '*.tar' | head -n 762 | "
            "parallel -j8 uv run python_data/repack_webdataset_to_parquet.py {}"
        ),
        flag="has --max-shards knob; 762 of ~2176 tars (~525 GB) carries the image modality; repack is per-tar.",
    ),
    StagingRow(
        dataset="laion_cc3m",
        modality="image",
        fraction=0.35,
        estimated_full_gigabytes=500.0,  # 3M image+caption WebDataset, ~575 tar shards.
        estimated_download_gigabytes=175.0,
        fraction_mechanism="max-shards",
        download_command=_uv(
            "python_data/download_laion_cc3m.py", "--out-dir", f"{DATA_ROOT}/laion-cc3m", "--max-shards", "200"
        ),
        repack_command=(
            f"find {DATA_ROOT}/laion-cc3m -name '*.tar' | head -n 200 | "
            "parallel -j8 uv run python_data/repack_webdataset_to_parquet.py {}"
        ),
        flag="has --max-shards knob; 200 of ~575 tars (~175 GB) carries the image modality; repack is per-tar.",
    ),
    StagingRow(
        dataset="coco_captions",
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=18.0,  # COCO captions (~120K train + ~5K val images x 5 captions).
        estimated_download_gigabytes=18.0,
        fraction_mechanism="max-shards",
        download_command=_uv("python_data/download_coco_captions.py", "--out-dir", f"{DATA_ROOT}/coco_captions"),
        requires_new_knob=False,
        flag="download_coco_captions.py stages the REAL 5-caption-per-image corpus (HuggingFaceM4/COCO "
        "2014_captions: image struct + sentences_raw) and --replace purges the prior VQA-as-caption "
        "shards; has a --max-shards knob.",
    ),
    StagingRow(
        dataset="flickr30k",
        glob_override="data/flickr30k/flickr30k-*.parquet",  # no flickr30k entry in any mixture
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=9.0,  # nlphuji/flickr30k, 31K images x 5 captions.
        estimated_download_gigabytes=9.0,
        fraction_mechanism="max-shards",
        download_command=_uv("python_data/download_flickr30k.py", "--out-dir", f"{DATA_ROOT}/flickr30k"),
        requires_new_knob=False,
        flag="download_flickr30k.py snapshots nlphuji/flickr30k (convert-branch parquet) and repacks to "
        "the coco_captions schema (image struct + sentences_raw); has a --max-shards knob.",
    ),
    StagingRow(
        dataset="docci",
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=7.5,  # 14.6K images x 1 dense human description, bytes inline.
        estimated_download_gigabytes=7.5,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_docci.py", "--out-dir", f"{DATA_ROOT}/docci"),
        flag="dense human descriptions; download_docci.py pulls + repacks (image + 1-element captions).",
    ),
    StagingRow(
        dataset="visual_genome",
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=15.5,  # ~108K images x dense region phrases, bytes inline.
        estimated_download_gigabytes=15.5,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_visual_genome.py", "--out-dir", f"{DATA_ROOT}/visual_genome"),
        flag="dense region descriptions (<=16 per image, multi-positive); download_visual_genome.py pulls + repacks.",
    ),
    StagingRow(
        dataset="doclaynet",
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=9.3,  # ~28K document pages with image bytes inline.
        estimated_download_gigabytes=9.3,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_doclaynet.py", "--out-dir", f"{DATA_ROOT}/doclaynet"),
        flag="document pages; WEAK caption = layout class names (bbox head out of scope); pulls + repacks.",
    ),
    StagingRow(
        dataset="xm3600",
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=7.0,  # 36-language pull, ~3,600 unique images after group-by.
        estimated_download_gigabytes=7.0,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_xm3600.py", "--out-dir", f"{DATA_ROOT}/xm3600"),
        flag="36-language HUMAN captions, cross-lingual multi-positive; tiny high-quality set (min_epochs=1).",
    ),
    StagingRow(
        dataset="rsicd",
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=1.0,  # ~11K remote-sensing images x 5 captions.
        estimated_download_gigabytes=1.0,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_rsicd.py", "--out-dir", f"{DATA_ROOT}/rsicd"),
        flag="remote-sensing / aerial, 5 captions each (multi-positive); download_rsicd.py pulls + repacks.",
    ),
    StagingRow(
        dataset="fashion_iq",
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=2.0,  # royokong/fashioniq_val, ~6K triplets with image bytes.
        estimated_download_gigabytes=2.0,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_fashion_iq.py", "--out-dir", f"{DATA_ROOT}/fashion_iq"),
        flag="SIMPLIFIED triplet->pair (modifier caption, target image, reference dropped); val-only ~6K.",
    ),
    StagingRow(
        dataset="droid",
        glob_override="data/droid_100/droid_*.parquet",  # no droid entry in any mixture; flat repacked parquet only
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=5.0,  # droid_100 1K-trajectory subset.
        estimated_download_gigabytes=5.0,
        fraction_mechanism="snapshot-full",
        download_command=_uv("python_data/download_droid.py", "--out-dir", f"{DATA_ROOT}/droid_100"),
        flag="download_droid.py pulls + self-repacks the LeRobot camera MP4 into a flat "
        "(image, instruction, action) parquet inline; no separate repack step.",
    ),
    StagingRow(
        dataset="funsd",
        glob_override="data/funsd/funsd/data/*.parquet",  # no funsd entry in any mixture
        modality="image",
        fraction=1.00,
        estimated_full_gigabytes=0.5,  # 149 train + 50 test scanned forms.
        estimated_download_gigabytes=0.5,
        fraction_mechanism="snapshot-full",
        download_command=_uv("python_data/download_funsd.py", "--out-dir", f"{DATA_ROOT}/funsd"),
        flag="layout dataset; image struct loads but words/bboxes need a layout-aware loader to train.",
    ),
    # Video modality, ~750 GB target.
    StagingRow(
        dataset="webvid_10m",
        glob_override="data/webvid-10m/data/train-*.parquet",  # no webvid_10m entry in any mixture
        modality="video",
        fraction=0.24,
        estimated_full_gigabytes=385.0,  # ~452 parquet shards; on-disk 108 shards measured at ~92 GB.
        estimated_download_gigabytes=92.0,
        fraction_mechanism="max-shards",
        download_command=_uv(
            "python_data/download_webvid_10m.py", "--out-dir", f"{DATA_ROOT}/webvid-10m", "--max-shards", "108"
        ),
        flag="has --max-shards knob; 108 of ~452 parquet shards (~92 GB measured) carries the video modality. "
        "Already parquet — no repack.",
    ),
    StagingRow(
        dataset="kinetics400",
        glob_override="data/kinetics400/train/*.parquet",  # no kinetics400 entry in any mixture
        modality="video",
        fraction=0.30,
        estimated_full_gigabytes=467.0,  # 242 train tar.gz shards, ~240K clips.
        estimated_download_gigabytes=140.0,
        fraction_mechanism="max-shards",
        download_command=_uv(
            "python_data/download_kinetics400.py", "--out-dir", f"{DATA_ROOT}/kinetics400", "--max-shards", "73"
        ),
        flag="has --max-shards knob; 73 of ~242 tar.gz shards (~140 GB) carries the video modality; "
        "download_kinetics400.py pulls + self-repacks (--repack-shard fan-out under GNU parallel), "
        "train split only, joined to annotations/train.csv.",
    ),
    StagingRow(
        dataset="panda70m_testing",
        glob_override="data/panda70m-wds/**/*.parquet",  # no panda70m_testing entry in any mixture; repacked tars
        modality="video",
        fraction=1.00,
        estimated_full_gigabytes=12.0,  # testing split only (~70K clips).
        estimated_download_gigabytes=12.0,
        fraction_mechanism="snapshot-full",
        download_command=_uv("python_data/download_panda70m_wds.py", "--out-dir", f"{DATA_ROOT}/panda70m-wds"),
        repack_command=(
            f"find {DATA_ROOT}/panda70m-wds -name '*.tar' | "
            "parallel -j8 uv run python_data/repack_webdataset_to_parquet.py {}"
        ),
    ),
    StagingRow(
        dataset="msrvtt_captions",
        modality="video",
        fraction=1.00,
        estimated_full_gigabytes=2.0,  # 9K clips repacked to one parquet.
        estimated_download_gigabytes=2.0,
        fraction_mechanism="snapshot-full",
        download_command=_uv("python_data/download_msrvtt.py", "--out-dir", f"{DATA_ROOT}/msrvtt-captions"),
        flag="download_msrvtt.py pulls + self-repacks the train sidecar into train_9k.parquet inline "
        "(--repack-shard worker mode); no separate repack step.",
    ),
    StagingRow(
        dataset="didemo",
        glob_override="data/didemo/data/didemo-train.parquet",  # no didemo entry in any mixture; train split only
        modality="video",
        fraction=1.00,
        estimated_full_gigabytes=48.0,  # test tar ~5 GB + 8-part train tar ~43 GB assembled.
        estimated_download_gigabytes=48.0,
        fraction_mechanism="max-shards",
        download_command=_uv("python_data/download_didemo.py", "--data-root", DATA_ROOT),
        flag="~4 ordered moment captions/clip (multi-positive); MP4 inline, no fetch; download_didemo.py "
        "pulls + self-repacks (--repack-shard fan-out). Registry reads the train split only.",
    ),
    StagingRow(
        dataset="mevis",
        glob_override="data/mevis/train/data/*.parquet",  # no mevis entry in any mixture
        modality="video",
        fraction=1.00,
        estimated_full_gigabytes=17.0,  # train JPEGImages.tar ~16 GB; re-encoded MP4 parquet is smaller.
        estimated_download_gigabytes=17.0,
        fraction_mechanism="max-shards",
        download_command=_uv("python_data/download_mevis.py", "--data-root", DATA_ROOT, "--max-shards", "1"),
        flag="frames+expressions ship cleanly; masks DROPPED (mask-head out of scope); needs ffmpeg at "
        "repack; download_mevis.py self-repacks. Expressions are object-motion-referring (noisier).",
    ),
    StagingRow(
        dataset="vatex",
        glob_override="data/vatex/data/*.parquet",  # no vatex entry in any mixture; fetched-clip parquet shards
        modality="video",
        fraction=1.00,
        estimated_full_gigabytes=50.0,  # ~29K clips x 10s H.264 mp4 (~1.7 MB/clip) fetched from YouTube.
        estimated_download_gigabytes=35.0,  # x ~0.7 real-world fetch yield.
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_vatex.py", "--out-dir", f"{DATA_ROOT}/vatex", "--fraction", "1.0"),
        flag="research-only (YouTube ToS): download_vatex.py pulls bilingual caption JSON (10 EN + 10 ZH/clip), "
        "then fetches an H.264 mp4 per 10s window via fetch_media_clips.py under GNU parallel; ~60-80% yield.",
    ),
    StagingRow(
        dataset="activitynet_captions",
        glob_override="data/activitynet-captions/data/*.parquet",  # no activitynet_captions entry in any mixture
        modality="video",
        fraction=0.20,
        estimated_full_gigabytes=450.0,  # ~15K FULL-length videos (~30 MB/video) fetched from YouTube.
        estimated_download_gigabytes=63.0,  # 0.20 x 450 GB x ~0.7 yield.
        fraction_mechanism="fraction",
        download_command=_uv(
            "python_data/download_activitynet_captions.py",
            "--out-dir",
            f"{DATA_ROOT}/activitynet-captions",
            "--fraction",
            "0.20",
        ),
        flag="research-only (YouTube ToS): download_activitynet_captions.py pulls densevid caption metadata, "
        "then fetches the full H.264 mp4 per video via fetch_media_clips.py under GNU parallel; ~60-80% yield.",
    ),
    # Audio modality, ~750 GB target.
    StagingRow(
        dataset="audioset",
        glob_override="data/audioset/data/**/*.parquet",  # no audioset entry in any mixture; bal_train/eval splits
        modality="audio",
        fraction=0.02,
        estimated_full_gigabytes=2400.0,  # full AudioSet (943 parquet shards: bal_train/eval/unbal_train).
        estimated_download_gigabytes=48.0,
        fraction_mechanism="splits",
        download_command=_uv(
            "python_data/download_audioset.py", "--out-dir", f"{DATA_ROOT}/audioset", "--splits", "bal_train,eval"
        ),
        flag="has --splits/--fraction knobs; bal_train+eval only (73 shards, ~48 GB) — audio intentionally small "
        "(unbal_train's 2.2 TB dropped).",
    ),
    StagingRow(
        dataset="audiocaps",
        modality="audio",
        fraction=1.00,
        estimated_full_gigabytes=41.0,  # full AudioCaps, already staged.
        estimated_download_gigabytes=41.0,
        fraction_mechanism="snapshot-full",
        download_command=_uv("python_data/download_audiocaps.py", "--out-dir", f"{DATA_ROOT}/audiocaps"),
    ),
    StagingRow(
        dataset="clotho",
        modality="audio",
        glob_override="data/clotho/**/clotho-train-*.parquet",  # no clotho.json in the catalog yet
        fraction=1.00,
        estimated_full_gigabytes=6.0,  # 6,974 clips, WAV inline; ~5 GB train.
        estimated_download_gigabytes=6.0,
        fraction_mechanism="fraction",
        download_command=_uv("python_data/download_clotho.py", "--out-dir", f"{DATA_ROOT}/clotho"),
        flag="5 human captions/clip (multi-positive); WAV audio inline, no fetch; download_clotho.py pulls + repacks.",
    ),
    StagingRow(
        dataset="auto_acd",
        glob_override="data/auto-acd/data/*.parquet",  # no auto_acd entry in any mixture; fetched-clip parquet shards
        modality="audio",
        fraction=0.05,
        estimated_full_gigabytes=230.0,  # 1.9M x ~10s mp3 (~120 KB/clip) fetched from YouTube.
        estimated_download_gigabytes=8.0,  # 0.05 x 230 GB x ~0.7 real-world fetch yield.
        fraction_mechanism="fraction",
        download_command=_uv(
            "python_data/download_auto_acd.py", "--out-dir", f"{DATA_ROOT}/auto-acd", "--fraction", "0.05"
        ),
        flag="research-only (YouTube ToS): download_auto_acd.py fetches bestaudio->mp3 per clip via "
        "fetch_media_clips.py under GNU parallel; ~60-80% audio yield audited in fetch_manifest.json.",
    ),
]


def rows_for_modality(modality: str) -> list[StagingRow]:
    """Return the manifest rows for one modality, or all rows when `modality == 'all'`."""
    if modality == "all":
        return list(STAGING_ROWS)
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}; expected one of {('all', *MODALITIES)}.")
    return [row for row in STAGING_ROWS if row.modality == modality]


def modality_download_gigabytes() -> dict[str, float]:
    """Sum the planned download GB per modality across the manifest."""
    totals = dict.fromkeys(MODALITIES, 0.0)
    for row in STAGING_ROWS:
        totals[row.modality] += row.estimated_download_gigabytes
    return totals
