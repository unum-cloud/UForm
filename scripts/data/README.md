# Datasets

UForm trains from public corpora across text, images, video, audio, 3D, and robotics.
Every corpus is referenced by name through the mixture JSONs — five per-mixture recipes (`scripts/mixtures/{text_masked,text_contrastive,image_contrastive,video_contrastive,audio_contrastive}.json`) that are the language-neutral column contract read by both the Python acquisition and the C++ trainer, parsed by `scripts/data/mixtures.py`; run `uv run scripts/data/list_registry.py` for the live name-to-glob-to-mixture mapping.
This page is the modality-organized catalog of what those names point to, including the corpora staged for upcoming tiers.

A checkbox marks whether the repository already has parsing code for the dataset: ✅ a mixture entry with a working downloader and a modality loader, 🔜 planned but not parseable yet — a wishlist row awaiting a downloader or loader, or a corpus not yet added.
The format reads `downloaded → stored`; every corpus is converted on download into the same GPU-friendly parquet that cuDF reads zero-copy, so the stored form is always `parquet`.
Where a section is a list rather than a table, each entry reads `corpus: role; format; coverage`.

## Staging the Balanced Corpus

The tables above list what _exists_; this section is how a balanced slice of it lands on disk.
The host data root comes from the `UFORM_DATA_ROOT` env var, falling back to the repository's own `data/` tree.
The declarative plan lives in `scripts/data/acquire_manifest.py` — one `StagingRow` per dataset, grouped into four modalities (text / image / video / audio).
Sizing is disk-aware: `resolve_budget_gigabytes` derives one total budget (`--budget-gb` > `$UFORM_DATA_BUDGET_GB` > free disk × `SAFETY_MARGIN`), `MODALITY_WEIGHTS` splits it across modalities (cross-modal video / image lead, text stays substantial, audio is thinnest), and `allocate_targets` water-fills each modality's share across its growable rows.
`scripts/data/acquire_corpus.py` reads that manifest and, per row, runs that row's `download_command` — a `uv run scripts/data/download_<name>.py` invocation that pulls and repacks each assigned shard to parquet in one pass.
`--dry-run` prints the exact commands and a per-modality balance table without downloading anything.

```sh
uv run scripts/data/acquire_corpus.py --dry-run                  # plan + balance table (auto-detect budget)
uv run scripts/data/acquire_corpus.py --dry-run --budget-gb 6000 # pin a 6 TB budget, reproducible plan
uv run scripts/data/acquire_corpus.py --modality all             # stage everything for real (resumable)
```

__Budgets and the cap vocabulary.__
No source is pulled whole; the allocator's per-dataset GB target converts to one uniform cap flag.
`--fraction F` (a leading `units × f` slice) and `--max-shards N` compose with the tighter one winning (`acquire_shared.cap_shards`); `--max-clips N` is the clip-native spelling for the YouTube-fetch datasets; and `--index-cap N` (`$UFORM_INDEX_CAP`) bounds how many windows an unbounded clip index even enumerates — distinct from the fetch cap, so `--fraction` / `--max-clips` lower the fetch below it, never above.
`--fanout N` on the clip fetchers is a parallelism knob, not a cap; the record repackers (C4-200M, Enron, Google Text-Norm) cap by `--max-records N`.
A downloader caps _before_ transfer: it enumerates the sorted upstream shard list and keeps only the leading slice, so a fraction never downloads-then-deletes.

__Lifecycle: acquire → repack → clean.__
Some corpora arrive in the mixture's parquet schema and are usable immediately (WebVid, AudioSet, COCO); others arrive as raw shards and need a one-time repack to parquet, which the shared `acquire()` helper does inline per shard, dispatching on the downloader's `source_kind`.
WebDataset tars are muxed to parquet by the `webdataset` source kind (CC-12M, LAION-CC3M, Panda-70M), Kinetics-400 by its media-fetch downloader (joined to `annotations/train.csv`), zstd / TSV text shards by the `text_jsonl` source kind (C4-200M, Google Text Normalization), and HuggingFace parquet snapshots — peS2o among them — by the `hf_snapshot` source kind with a column projection.
Each path writes into a rank-private scratch, commits through an atomic rename, and drops the raw shard after a verified parquet write, so a repack is disk-neutral-to-positive and resumable.
`clean_media_shards.py` is an optional later pass that decode-verifies media and drops corrupt rows into a parallel `*_clean` tree.

__Acquisition CLI.__
Each dataset has its own thin script: `uv run scripts/data/download_<name>.py [--fraction F | --max-shards N]` calls the shared `acquire()` helper with that dataset's source constants and an optional `_transform`, which dispatches on `source_kind` and writes canonical parquet under `UFORM_DATA_ROOT/<out_name>` (falling back to the repository's `data/`); a handful of corpora that don't fit the single-repo-per-shard model (YouTube fetch, TSV/CSV, multi-repo, the generic `text_pairs.py`) keep a full bespoke downloader instead.
It is single-threaded and rank-aware.
`scripts/slurm/acquire.sbatch` is a thin launcher over `scripts/data/acquire_corpus.py`: it grabs every idle node, spawns one worker per GPU (acquisition is CPU/IO — the GPUs go unused, the count is just a hardware-matched fan-out), splits each node's cores evenly among its workers, and runs the driver once per worker.
Each worker reads its rank/world (`$SLURM_PROCID` / `$SLURM_NTASKS` via `scripts/data/sharding.py`) and stages exactly its share: _shardable_ datasets (the `acquire` / `acquire_clips` downloaders, including the YouTube fetchers) run on every worker and stride `shards[rank::world]` internally, while _whole-dataset_ ones that can't sub-shard (single-file TSV/CSV repackers, multi-repo, the generic `text_pairs.py`) are assigned one-per-worker by `row_index % world`.
The disk budget flows from `acquire_manifest` (`--budget-gb` / `$UFORM_DATA_BUDGET_GB` / free-disk), so each dataset's cap is the multimodal-first allocation, not a flat fraction.
Completion is the per-shard atomic rename, so the run is idempotent and requeue-safe: re-submit to fill gaps, raise the node range to add workers, nothing coordinates.
A single-machine `uv run scripts/data/acquire_corpus.py --modality all` is the identical code path with no Slurm env (`world = 1`, pulls everything).

__Disk runbook.__
The data root is typically a large scratch volume; keep headroom.
Over-pulls are trimmed by deleting the unwanted split or shards (`snapshot_download` only adds, never removes) — e.g. AudioSet's `unbal_train` (2.2 TB) is dropped so audio stays curated.
Audio is intentionally the thinnest modality (`MODALITY_WEIGHTS` gives it the smallest share); growing it into AudioSet's `unbal_train` (2.2 TB) is available via a larger `--fraction`, but deliberately down-weighted.

## Text

The text stack spans masked-language pretraining (`MLM`), contrastive retrieval `pairs`, and hard-negative `triplets`, across 40+ natural languages and several programming languages.

### Web & Reference

| Dataset                       | Role       | Format  | Coverage                             |
| ----------------------------- | ---------- | ------- | ------------------------------------ |
| ✅ [FineWeb][fineweb]         | MLM        | parquet | English web, 100-billion-token slice |
| ✅ [FineWeb-2][fineweb2]      | MLM        | parquet | Multilingual web, 42 languages       |
| ✅ [FineWiki][finewiki]       | MLM, pairs | parquet | Wikipedia, 325 languages ¹           |
| ✅ [peS2o][pes2o]             | MLM        | parquet | AI2 scientific full text             |
| ✅ [S2ORC][s2orc]             | MLM, pairs | parquet | Abstracts; abstract → citation pairs |
| ✅ [arXiv][arxiv]             | pairs      | parquet | Article → abstract                   |
| 🔜 [FineWeb-Edu][fineweb-edu] | MLM        | parquet | Quality-filtered web ²               |
| 🔜 [RedPajama-V2][redpajama]  | MLM        | parquet | Alternative crawl ²                  |

> ¹ FineWiki globs every `<lang>wiki/` subdirectory — 325 languages including 3-letter codes — with `finewiki_en` the English-only slice and `finewiki_i18n` the non-English remainder used by the MLM recipe.
> ² FineWeb-Edu (quality-filtered) and RedPajama-V2 (different crawl and filtering) are wishlist diversifiers that add bias variety without re-downloading the same FineWeb pages.

[fineweb]: https://huggingface.co/datasets/HuggingFaceFW/fineweb
[fineweb2]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
[finewiki]: https://huggingface.co/datasets/wikimedia/wikipedia
[pes2o]: https://huggingface.co/datasets/allenai/peS2o
[s2orc]: https://github.com/allenai/s2orc
[arxiv]: https://huggingface.co/datasets/ccdv/arxiv-summarization
[fineweb-edu]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
[redpajama]: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2

### Retrieval & QA Pairs

- ✅ [MS MARCO][msmarco]: pairs, triplets; parquet; query → passage; 1 or 50 BM25 hard negatives ¹
- 🔜 [mMARCO][mmarco]: pairs; parquet; multilingual MS MARCO ²
- ✅ [Mr. TyDi][mrtydi]: triplets; parquet; 11-language retrieval, multiple positives per query
- ✅ [Natural Questions][nq]: pairs; parquet; query → answer
- ✅ [HotpotQA][hotpotqa]: pairs; parquet; multi-hop question → passage
- ✅ [ELI5][eli5]: pairs; parquet; question → long-form answer
- ✅ [GooAQ][gooaq]: pairs; parquet; question → answer
- ✅ [PAQ][paq]: pairs; parquet; 64M synthetic QA pairs
- ✅ [AllNLI][allnli]: triplets; parquet; NLI anchor / positive / negative
- ✅ [Quora][quora]: triplets; parquet; duplicate-question pairs
- ✅ [WikiAnswers][wikianswers]: pairs; parquet; 27M paraphrase pairs
- ✅ [StackExchange][stackexchange]: MLM, pairs; parquet; title → answer; duplicate pairs ³
- ✅ [HackerNews][hackernews]: MLM, pairs; parquet; title → body
- ✅ [WebInstruct][webinstruct]: pairs; parquet; 481K instruction Q → A ⁴
- ✅ [OpenHermes 2.5][openhermes]: pairs; JSON → parquet; 1M conversational instructions ⁴
- 🔜 [Infinity-Instruct][infinity]: pairs; parquet; 7M+ instructions ⁵

> ¹ Named `msmarco` (pairs), `msmarco_triplet` (one BM25 hard negative), and `msmarco_triplet50` (fifty negatives reshaped on device via `cudf.melt`).
> ² Multilingual MS MARCO, staged but still awaiting a mixture entry and downloader.
> ³ Title → top-answer pairs feed contrastive training while the long-form body feeds MLM via `stackexchange_text`, with duplicate-question pairs as a separate config.
> ⁴ WebInstruct routes through the standard pair loader; OpenHermes is flattened to a flat (prompt, response) parquet by the `text_jsonl` source kind and then read through the same loader.
> ⁵ The largest open instruction corpus, on the wishlist and not yet staged.

[msmarco]: https://microsoft.github.io/msmarco/
[mmarco]: https://huggingface.co/datasets/unicamp-dl/mmarco
[mrtydi]: https://huggingface.co/datasets/castorini/mr-tydi
[nq]: https://ai.google.com/research/NaturalQuestions
[hotpotqa]: https://hotpotqa.github.io/
[eli5]: https://huggingface.co/datasets/sentence-transformers/eli5
[gooaq]: https://huggingface.co/datasets/sentence-transformers/gooaq
[paq]: https://github.com/facebookresearch/PAQ
[allnli]: https://huggingface.co/datasets/sentence-transformers/all-nli
[quora]: https://huggingface.co/datasets/sentence-transformers/quora-duplicates
[wikianswers]: https://huggingface.co/datasets/sentence-transformers/wikianswers-duplicates
[stackexchange]: https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_body_jsonl
[hackernews]: https://huggingface.co/datasets/OpenPipe/hacker-news
[webinstruct]: https://huggingface.co/datasets/TIGER-Lab/WebInstructSub
[openhermes]: https://huggingface.co/datasets/teknium/OpenHermes-2.5
[infinity]: https://huggingface.co/datasets/BAAI/Infinity-Instruct

### Personal Search (Email / Messages / Notes)

The on-device local fuzzy-search use case — a short, typo-laden query against a user's own email / messages / notes — has no single public corpus, so the mixture pairs the few public email + chat sets with synthetic surface robustness from the augmentation family (date / number / casing / spelling variants; see `### Denoising & Normalization`).
Each entry slots into the mixture JSONs' `(anchor / positive / negative)` field contract; hard negatives are BM25- or ANN-mined at pack time where the source ships none.
The augmentation family supplies the _multi-positive_ side by construction: each query variant (reformatted date, regrouped digits, mixed case, injected typo) is an additional positive of the same target, which is also what satisfies the multi-positive/multi-negative CSR guard.

- ✅ [Enron Email][enron]: pairs; tar → parquet; subject → body, the only large real email corpus ¹
- ✅ [AESLC][aeslc]: pairs; parquet; email body → subject line (Annotated Enron Subject Line) ²
- 🔜 [Ubuntu Dialogue][ubuntu]: pairs; parquet; multi-turn chat context → next turn, informal register
- 🔜 [Taskmaster][taskmaster]: pairs; JSON → parquet; task-oriented dialog turn → response across 6 domains
- 🔜 [MultiWOZ][multiwoz]: pairs; JSON → parquet; multi-domain goal-oriented dialog turn → response
- 🔜 [WikiHow][wikihow]: pairs; parquet; step title → step body, a public proxy for note-like short docs ³

> ¹ Public (FERC release); named `enron_email` and acquired via `uv run scripts/data/download_enron.py`, whose `webdataset` source kind muxes the tarball into `(subject, body)` pairs (quoted / forwarded thread history stripped).
> Thread → reply pairs and cross-thread hard-negative mining remain future work.
> ² Named `aeslc` and acquired via `uv run scripts/data/download_aeslc.py` (the `hf_snapshot` source kind); ~14K body → subject pairs where the subject is a terse, abbreviation-heavy "query" over the body, matching the short-query regime.
> ³ Notes are private by nature, so no public corpus exists; WikiHow's title → body and the synthetic augmentation variants stand in until any opt-in on-device personalization is added.

[enron]: https://www.cs.cmu.edu/~enron/
[aeslc]: https://huggingface.co/datasets/aeslc
[ubuntu]: https://huggingface.co/datasets/sentence-transformers/ubuntu-dialogue
[taskmaster]: https://github.com/google-research-datasets/Taskmaster
[multiwoz]: https://huggingface.co/datasets/multi_woz_v22
[wikihow]: https://huggingface.co/datasets/wikihow

### Code

- ✅ [CodeSearchNet][codesearchnet]: MLM, pairs; parquet; docstring → function, 6 languages
- ✅ [POJ-104][poj104]: pairs; parquet; 104 C++ problems × ~500 clone solutions; equivalence-class positives
- ✅ [The Vault][thevault]: pairs; parquet; docstring → function, 10 languages, 11 docstring styles
- ✅ [XLCoST][xlcost]: pairs; parquet; NL → code aligned across 7 languages
- ✅ [CommitPackFT][commitpackft]: pairs; parquet; commit message → diff, 277 languages
- ✅ [CodeContests][codecontests]: triplets; parquet; problem → solution; wrong solutions as hard negatives ¹
- ✅ [ComPile-IR][compile-ir]: MLM; parquet; LLVM IR modules ²
- ✅ [The Stack v2][thestack]: MLM, pairs; parquet; \~1T tokens, 86 languages ³
- ✅ [CommitPackFT-apply][commitpackft]: denoise pair; parquet; `[old + subject] → new` fast-apply ⁴
- 🔜 [the-Stack lint/format][thestack]: denoise pair; parquet (+ generated); deterministic ruff/black/gofmt before → after ³
- 🔜 [CodeNet][codenet]: pairs; tar → parquet; 14M submissions, 50+ languages ⁵
- 🔜 [AnghaBench / ExeBench][exebench]: pairs; parquet; C → LLVM IR pairs ⁶

> ¹ Acquired via `uv run scripts/data/download_codecontests.py` (the `hf_snapshot` source kind), with a competition-skewed language mix (C++ / Python / Java) that buys NL ↔ code alignment and equivalence supervision rather than language breadth.
> ² Production LLVM IR from Rust / Julia / Swift / C-C++ (\~1.3T tokens), staged as a capped shard count via `uv run scripts/data/download_compile_ir.py --max-shards N` (the `hf_snapshot` source kind) so the mixture weight, not the byte count, controls representation.
> ³ Named `starcoderdata` (single-text code-MLM, full files beyond CommitPackFT's diffs) and acquired via `uv run scripts/data/download_starcoderdata.py` (the `hf_snapshot` source kind, 86 languages); the same corpus also feeds the deterministic lint/format before → after pairs, which are still 🔜.
> ⁴ A sibling view of `commitpackft` — `old_contents` + the commit `subject` → `new_contents` — for non-AR copy/edit fast-apply, expressed as the separate `commitpackft_apply` mixture entry reusing `TextPairLoader`.
> ⁵ Wishlist competitive submissions carrying accept/reject, CPU-time, and memory labels with the broadest language coverage of the code sets, though the HF mirrors leave their parquet schema undocumented.
> ⁶ Wishlist whose source ↔ IR pairs need a clang `-emit-llvm` pass, with ExeBench also shipping I/O examples for execution-grounded objectives.

[codesearchnet]: https://github.com/github/CodeSearchNet
[poj104]: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104
[thevault]: https://huggingface.co/datasets/Fsoft-AIC/the-vault-function
[xlcost]: https://huggingface.co/datasets/codeparrot/xlcost-text-to-code
[commitpackft]: https://huggingface.co/datasets/bigcode/commitpackft
[codecontests]: https://huggingface.co/datasets/deepmind/code_contests
[compile-ir]: https://huggingface.co/datasets/llvm-ml/ComPile
[thestack]: https://huggingface.co/datasets/bigcode/the-stack-v2
[codenet]: https://github.com/IBM/Project_CodeNet
[exebench]: https://huggingface.co/datasets/jordiae/exebench

### Denoising & Normalization

Paired `(corrupted → canonical)` transductions that pressure the uniform byte encoder toward surface-invariant semantics (the denoising-autoencoder regularizer in `NORMALIZATION_TASKS.md`).
These now parse through the mixture JSONs' `(anchor, positive)` contract on the standard `TextPairLoader` path, supplying the multi-positive surface-robustness side of the personal-search mixture; in-place corruption-MLM over the web corpora above remains a complementary cheap vehicle.

- ✅ [C4\_200M][c4200m]: denoise pair; zip → parquet; 185M synthetic GEC pairs; permissive training bulk ¹
- ✅ [Google Text Normalization][googletn]: denoise pair; CSV → parquet; written ↔ spoken hard classes; best non-AR fit ²
- 🔜 [LogHub / LogHub-2.0][loghub]: denoise, extract; logs → parquet; `{time}{source}{priority}{message}` schema-fill
- ✅ [PleIAs Post-OCR][pleias]: denoise pair; parquet; \~1B words permissive OCR → clean
- ✅ [GitHub Typo Corpus][githubtypo]: denoise pair; parquet; real code-comment-register typos ³

> ¹ Named `c4_200m` and acquired via `uv run scripts/data/download_c4_200m.py --max-shards N`, whose `text_jsonl` source kind repacks one TSV shard at a time (default one ~18M-pair shard); C4-derived with Apache-licensed corruption code — the permissive workhorse for the semantic-regularizer A/B.
> The gold GEC sets (cLang-8, JFLEG, W\&I+LOCNESS, CoNLL-2014) stay as evals because they are non-commercial or gated.
> ² Named `google_text_norm` and acquired via `uv run scripts/data/download_google_text_norm.py`, whose `text_jsonl` source kind repacks into `(written, spoken, semiotic_class)` columns; the downloader drops the PLAIN / PUNCT identity rows on device so only the class-conditional rewrites (date / cardinal / ordinal / decimal / money / measure / time / telephone / electronic / fraction / address / letters) reach the loss — the sweet spot for the one-pass non-AR head.
> Ships via the `pavanBuduguppa/asr_inverse_text_normalization` mirror of the Sproat data (the original GCS bucket is now gated).
> ³ Named `github_typo` and acquired via `uv run scripts/data/download_github_typo.py` (the `hf_snapshot` source kind) from the `chirunder/github_typo_corrections` parquet mirror (the upstream S3 bucket is decommissioned); ~353K real edits with the best register match to the code mix, but a murky GitHub-ToS license — stage research-only, complementing the free synthetic keyboard/OCR-confusion generators.

[c4200m]: https://huggingface.co/datasets/liweili/c4_200m
[googletn]: https://github.com/rwsproat/text-normalization-data
[loghub]: https://github.com/logpai/loghub
[pleias]: https://huggingface.co/datasets/PleIAs/Post-OCR-Correction
[githubtypo]: https://github.com/mhagiwara/github-typo-corpus

### Safety & Robustness

A defensive content-moderation / safe-search track, mirroring the de-identification track: the model learns a robust toxic/benign boundary, never a generation target.
CSAM is entirely out of scope.

- ✅ [ToxiGen][toxigen]: safety; parquet; ~260K implicit toxic/benign statements over 13 groups; text+label ¹
- ✅ [Civil Comments][civilcom]: safety; parquet; ~1.9M real comments with continuous toxicity scores; text+label, CC0 ²
- 🔜 [Hateful Memes][hmemes]: safety; —; multimodal hate; Meta/DrivenData-gated, non-commercial — not HF-stageable ³
- 🔜 [XD-Violence][xdv]: safety; —; weakly-labeled violence; HF mirror ships features, not raw media ⁴
- 🔜 [UCF-Crime][ucfc]: safety; —; 13 real-crime surveillance classes; zip video, license tag needs vetting ⁵

> ¹ `toxigen`, acquired via `uv run scripts/data/download_toxigen.py` (the `hf_snapshot` source kind; configs annotated+train → flat text/label/toxicity_score/target_group/source).
> RESEARCH-USE: no SPDX license on HF — vet before commercial.
> `label==1/0` is the unsafe/safe contrastive split.
> ² `civil_comments`, acquired via `uv run scripts/data/download_civil_comments.py` (the `hf_snapshot` source kind; same flat schema; label from toxicity ≥ 0.5).
> CC0-1.0, commercial-OK.
> ³ Original gated via DrivenData under a Meta research-only / non-commercial agreement; HF mirrors are unlicensed re-uploads — not staged.
> ⁴ `jherng/xd-violence` (MIT tag) exposes C3D/I3D/Swin `.npy` features + copyrighted movie clips; doesn't fit the raw-media parquet loaders.
> ⁵ `jinmang2/ucf_crime` (cc0 tag) ships anomaly/normal video zips; raw-video → parquet repack + license-provenance vetting required before staging.

[toxigen]: https://huggingface.co/datasets/toxigen/toxigen-data
[civilcom]: https://huggingface.co/datasets/google/civil_comments
[hmemes]: https://hatefulmemeschallenge.com/
[xdv]: https://huggingface.co/datasets/jherng/xd-violence
[ucfc]: https://huggingface.co/datasets/jinmang2/ucf_crime

## Images

### Captioning & Contrastive

- ✅ [COCO Captions][coco]: image-text; parquet; human captions; 5 per image, 8 hard negatives ¹
- ✅ [Conceptual 12M][cc12m]: image-text; WebDataset → parquet; 12M web alt-text pairs ²
- ✅ [LAION-CC3M][cc3m]: image-text; WebDataset → parquet; \~3M web image-text pairs ²
- 🔜 [DataComp-small][datacomp]: image-text; WebDataset → parquet; 12.8M CLIP-scored pairs ³
- 🔜 [BLIP3-KALE][kale]: image-text; parquet (URLs); 171M synthetic captions ³
- 🔜 [LAION-Aesthetics][laionaes]: image-text; parquet (URLs); aesthetic-filtered web ³
- 🔜 [ImageNet][imagenet]: image; JPEG; pre-resized; linear-probe evaluation only
- ✅ [DOCCI][docci]: image-text; parquet; \~15K images, one long dense human description each ⁴
- ✅ [XM3600][xm3600]: image-text; parquet; 3,600 geo-diverse images, HUMAN captions in 36 languages ⁴
- ✅ [RSICD][rsicd]: image-text; parquet; \~11K remote-sensing / aerial images, 5 captions each ⁴
- ✅ [DocLayNet][doclaynet]: image-text; parquet; document pages; weak captions = layout class names ⁴

> ¹ Hard negatives are mined offline by `mine_hard_negatives.py` (eight per row) and the first non-empty caption is the positive.
> ² Repacked from WebDataset tars to parquet by the `webdataset` source kind and read zero-copy through `ImageTextParquetLoader`.
> ³ URL-only metadata (CLIP scores but no decoded bytes), so each needs an `img2dataset`-style fetch into parquet before a loader can read it.
> ⁴ Single-file pull + repack (image struct + a `captions` list<string>), all ship decoded bytes (no URL fetch): DOCCI's dense human description, XM3600's 36-language human captions (cross-lingual multi-positive), RSICD's 5 captions, and DocLayNet's layout class names (a WEAK signal — its bbox/mask head is out of scope).

[coco]: https://cocodataset.org
[cc12m]: https://github.com/google-research-datasets/conceptual-12m
[cc3m]: https://github.com/google-research-datasets/conceptual-captions
[datacomp]: https://huggingface.co/datasets/mlfoundations/datacomp_small
[kale]: https://huggingface.co/datasets/Salesforce/blip3-kale
[laionaes]: https://laion.ai/blog/laion-aesthetics/
[imagenet]: https://www.image-net.org/
[docci]: https://huggingface.co/datasets/ljnlonoljpiljm/docci
[xm3600]: https://huggingface.co/datasets/floschne/xm3600
[rsicd]: https://huggingface.co/datasets/arampacha/rsicd
[doclaynet]: https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1

### Grounding

- ✅ [Visual Genome][visualgenome]: image-text; parquet; \~108K images, dense region descriptions (≤16 per image, multi-positive) ²
- ✅ [Flickr30k][flickr30k]: image-text; parquet; caption retrieval; Entities variant is phrase → bbox, gated ¹
- ✅ [Fashion-IQ][fashioniq]: image-text; parquet; composed retrieval; SIMPLIFIED to (modifier caption, target image) ³
- 🔜 [FineHARD][finehard]: region-text; URL/JSON → parquet; 12M images, 40M described boxes, 10M hard negatives; Apache-2.0 tag, source-image rights separate; images fetched with img2dataset
- 🔜 [STRAP][strap]: region-text; annotation parquet; 2M images, 10.35M objects; no image bytes; CC BY 4.0 tag conflicts with research-only / no-redistribution access terms

> ¹ The plain `nlphuji/flickr30k` captions mirror is parquet; the Flickr30k-Entities grounding variant (276K phrase → bbox links) is a separate, gated artifact.
> ² Acquired via `uv run scripts/data/download_visual_genome.py` (the `hf_snapshot` source kind, nirmalendu01/visual_genome mirror, image bytes inline), flattening up to 16 region descriptions per image into a `captions` list<string>; the bbox/relation graph stays out of scope.
> ³ Fashion-IQ is natively a (reference image, modifier text, target image) triplet; the `royokong/fashioniq_val` mirror ships both images' bytes, but it is staged here as a lossy (modifier, target) pair (reference dropped, val-only ~6K).
> Proper composed retrieval needs a bespoke triplet loader.

[fashioniq]: https://huggingface.co/datasets/royokong/fashioniq_val

[visualgenome]: https://homes.cs.washington.edu/~ranjay/visualgenome/
[flickr30k]: https://huggingface.co/datasets/nlphuji/flickr30k
[finehard]: https://huggingface.co/datasets/qihoo360/FineHARD
[strap]: https://huggingface.co/datasets/vrg-prague/STRAP

### VLM Instructions

| Dataset                      | Role       | Format         | Coverage                                 |
| ---------------------------- | ---------- | -------------- | ---------------------------------------- |
| 🔜 [LLaVA (Mix-665K)][llava] | image-text | JSON → parquet | single-image instruction SFT, Apache-2.0 |
| 🔜 [ShareGPT4V][sharegpt4v]  | image-text | JSON → parquet | dense GPT-4V captions, CC BY 4.0         |
| 🔜 [M4-Instruct][m4]         | image-text | ZIP → parquet  | multi-image instruction tuning ¹         |

> ¹ Per-task ZIP archives with varied schemas; needs a repack to parquet plus the `ImageInstructLoader` (image decode + `TextPacker` masking) before the VLM trainer can consume real data instead of its synthetic smoke batch.

[llava]: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
[sharegpt4v]: https://huggingface.co/datasets/Lin-Chen/ShareGPT4V
[m4]: https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data

### Documents

| Dataset                         | Role     | Format  | Coverage                             |
| ------------------------------- | -------- | ------- | ------------------------------------ |
| 🔜 [FUNSD][funsd]               | document | parquet | Scanned forms; words + boxes + NER ¹ |
| 🔜 [FUNSD-LayoutLMv3][funsd-v3] | document | parquet | LayoutLMv3 repackaging of FUNSD ¹    |

> ¹ Form-understanding corpora (149 train + 50 test) whose image struct loads today but whose word / bbox / NER layout needs a layout-aware loader; can double as an OCR-region → normalization probe.

[funsd]: https://guillaumejaume.github.io/FUNSD/
[funsd-v3]: https://huggingface.co/datasets/nielsr/funsd-layoutlmv3

### Privacy & De-identification

- 🔜 [WIDER FACE][widerface]: redact; imgs+boxes → parquet; face detection, 32K images / 393K faces; heavy scale, pose, occlusion variation
- 🔜 [CrowdHuman][crowdhuman]: redact; imgs+boxes → parquet; dense persons in crowds; head, visible, and full-body boxes
- 🔜 [CCPD][ccpd]: redact; imgs+quads → parquet; \~250K license plates with 4-corner polygons and plate strings (China)
- 🔜 [RodoSol-ALPR][rodosol]: redact; imgs+quads → parquet; Brazilian car + motorcycle plates; geographic coverage beyond CCPD
- 🔜 _synthetic composites_: redact; rendered → parquet; faces / plates / PII composited onto scenes → unlimited exact masks, no labeling

> The training target is a per-pixel __redaction mask__ keyed by type (face / plate / person / PII), composited (blur / replace) _externally_ so the encoder never regenerates pixels.
> The street corpora below (Cityscapes, Mapillary) and the on-disk DROID dashcam clips supply in-the-wild faces + plates; synthetic compositing is the scalable, annotation-free source.

[widerface]: http://shuoyang1213.me/WIDERFACE/
[crowdhuman]: https://www.crowdhuman.org/
[ccpd]: https://github.com/detectRecog/CCPD
[rodosol]: https://github.com/raysonlaroca/rodosol-alpr-dataset

### Dense Labeling & Segmentation

- 🔜 [ADE20K][ade20k]: seg; imgs+masks → parquet; scene parsing, 150 classes; the semantic-segmentation standard
- 🔜 [COCO-Stuff][cocostuff]: seg; imgs+masks → parquet; things + stuff, 171 classes, \~120K images
- 🔜 [Cityscapes][cityscapes]: seg, redact; imgs+masks → parquet; urban street, 19 classes; dual-use — in-frame faces + plates feed de-id too
- 🔜 [Mapillary Vistas][mapillary]: seg, redact; imgs+masks → parquet; global street-level, 66+ classes; geographic + de-id overlap
- 🔜 [SA-1B][sa1b]: masks; imgs+masks → parquet; 11M images / 1.1B class-agnostic masks; warm-starts the dense mask head

> The target is a per-patch __class map__ aligned to the encoder output grid.
> SA-1B is class-agnostic (mask-head pretraining, not semantics); the street corpora are shared with de-identification, so one download feeds both dense heads.

[ade20k]: https://groups.csail.mit.edu/vision/datasets/ADE20K/
[cocostuff]: https://github.com/nightrome/cocostuff
[cityscapes]: https://www.cityscapes-dataset.com/
[mapillary]: https://www.mapillary.com/dataset/vistas
[sa1b]: https://ai.meta.com/datasets/segment-anything/

## Video

- ✅ [MSR-VTT][msrvtt]: video-text; parquet; 9K clips, \~20 captions each, 8 hard negatives ¹
- ✅ [WebVid-10M][webvid]: video-text; WebDataset → parquet; 10M caption → clip pairs ²
- ✅ [Panda-70M][panda70m]: video-text; WebDataset → parquet; captioned clips, testing split staged ²
- ✅ [Kinetics-400][kinetics]: video, action; tar.gz → parquet; 400 action classes as captions ²
- ✅ [DiDeMo][didemo]: video-text; tar → parquet; \~10K Flickr clips, \~4 ordered moment captions each ⁴
- ✅ [MeViS][mevis]: video-text; frames → parquet; \~2K clips, \~33K referring motion expressions ⁵
- ✅ [VATEX][vatex]: video-text; YouTube fetch → parquet; \~29K clips, 20 captions each (10 EN + 10 ZH) ⁶
- ✅ [ActivityNet Captions][activitynet]: video-text; YouTube fetch → parquet; \~15K videos, \~3.65 dense captions each ⁶
- 🔜 [Something-Something v2][somethingv2]: video, action; webm → parquet; temporal action, 220K clips
- ✅ [HowTo100M][howto100m]: video-text; YouTube fetch → parquet; \~136M narrated clips, per-clip ASR captions ³
- 🔜 [HowToGround1M][howtoground1m] / [iGround][iground]: grounded video-text; JSONL → parquet; automatic / manual box supervision; media acquired separately; CC BY-NC-SA 4.0, non-commercial
- 🔜 [ViCaS][vicas]: grounded video-text; JSON/masks → parquet; captions and tracked masks; official release excludes videos; repository CC0, dataset card CC variant and source-video terms need clarification
- 🔜 [VideoRefer-700K][videorefer]: region-video-text; JSON/video archives → parquet; Panda media included, other source-media coverage unverified; Apache-2.0 tag conflicts with Panda-70M non-commercial research terms
- 🔜 [BOSTVG][bostvg]: multi-object video grounding; video ZIP/JSON → parquet; 10,018 videos with tube annotations; research-only per paper, no explicit release license

> ¹ Fetched and repacked on demand via `uv run scripts/data/download_msrvtt.py` (its media-fetch downloader) because the license blocks redistribution, with an in-batch shuffle synthesizing rerank negatives.
> ² Repacked to parquet by the `webdataset` source kind (Panda-70M's tars), the `hf_snapshot` source kind (WebVid's snapshot parquet), or its media-fetch downloader (Kinetics-400's CSV-joined action label); WebVid is Shutterstock-licensed stock video.
> ³ Acquired via `uv run scripts/data/download_howto100m.py`: streams a bounded per-clip ASR-caption parquet index (`bpiyush/howto100m_captions_with_verb_nouns`, capped by `--index-cap` / `$UFORM_INDEX_CAP` windows), then fetches an H.264 mp4 per window via `acquire_clips` (rank/world split); bulk YouTube fetch needs `yt-dlp` cookies.
> HMDB-51 (`uv run scripts/data/download_hmdb51.py`) is kept only as a small VideoMAE smoke, not a training corpus.
> ⁴ The per-moment descriptions are packed into a `captions` list<string> by the `webdataset` source kind, which muxes the MP4 tars to parquet inline; no external fetch.
> ⁵ The `webdataset` source kind re-encodes each clip's JPEG frames to MP4 (ffmpeg) and packs the referring motion expressions into a `captions` list<string>; the dense masks are dropped (mask-head loader out of scope), and expressions refer to sub-clip object motion, so treat MeViS as supplementary video-text rather than a precision benchmark.
> ⁶ Fetched on demand from YouTube by its media-fetch downloader — `uv run scripts/data/download_vatex.py` / `uv run scripts/data/download_activitynet_captions.py` — because the media is not redistributable.
> Real-world yield is ~60-80% (dead / private / geo-blocked / bot-gated ids are dropped and audited in each build's `fetch_manifest.json`, never silently truncated); research-only under YouTube's Terms of Service.

[msrvtt]: https://huggingface.co/datasets/friedrichor/MSR-VTT
[didemo]: https://github.com/LisaAnne/LocalizingMoments
[mevis]: https://github.com/henghuiding/MeViS
[vatex]: https://eric-xw.github.io/vatex-website/
[activitynet]: https://cs.stanford.edu/people/ranjaykrishna/densevid/
[webvid]: https://github.com/m-bain/webvid
[panda70m]: https://snap-research.github.io/Panda-70M/
[kinetics]: https://github.com/cvdfoundation/kinetics-dataset
[somethingv2]: https://developer.qualcomm.com/software/ai-datasets/something-something
[howto100m]: https://www.di.ens.fr/willow/research/howto100m/
[howtoground1m]: https://huggingface.co/datasets/ekazakos/HowToGround1M
[iground]: https://huggingface.co/datasets/ekazakos/iGround
[vicas]: https://github.com/Ali2500/ViCaS
[videorefer]: https://huggingface.co/datasets/DAMO-NLP-SG/VideoRefer-700K
[bostvg]: https://huggingface.co/datasets/Jelly3000/OmniSTVG

## Audio

UForm carries no custom audio kernels, and no GPU audio codec exists across the NVIDIA, AMD, and Intel SDKs.
Audio is decoded host-side to PCM via libavformat, then an on-GPU `cuFFT` short-time Fourier transform (Hann window) and a mel-filterbank `cuBLAS` GEMM produce log-mel frames in the same token-stream shape as the byte encoder.
AudioCaps and AudioSet are read through `AudioParquetLoader`; VGGSound is on the roadmap.
Each entry reads `corpus: modalities; format; scale; license`.

- ✅ [AudioCaps][audiocaps]: audio + text; parquet; 50K captioned clips ¹; research
- ✅ [Clotho v2][clotho]: audio + text; WAV → parquet; 6,974 clips × 5 human captions ³; research
- ✅ [Auto-ACD][autoacd]: audio + text; YouTube fetch → parquet; 1.9M captioned clips ⁴; research
- ✅ [AudioSet][audioset]: audio + labels; parquet; 2M clips ²; CC BY 4.0
- 🔜 [VGGSound][vggsound]: video + audio + label; video → parquet; clean audio-visual; CC BY 4.0

> ¹ WAV/PCM payloads decoded by `AudioParquetLoader` → `AudioDecoderPool.decode_clip_tape_to_logmel`.
> ² Parquet payloads with event labels joined into the contrastive text.
> Acquired via `uv run scripts/data/download_audioset.py` and capped with `--fraction` / `--max-shards` (the `hf_snapshot` source kind).
> The default stages only the curated `bal_train` + `eval` splits (~48 GB), with `unbal_train` (2.2 TB) opt-in.
> ³ The five human captions per clip are packed into a `captions` list<string> by the `webdataset` source kind; WAV audio bytes ship inline, so no external fetch is needed.
> ⁴ Fetched on demand from YouTube via `uv run scripts/data/download_auto_acd.py` (its media-fetch downloader, bestaudio → mp3 per clip) because the audio is not redistributable; ~60-80% yield, audited in `fetch_manifest.json`; research-only under YouTube ToS.

[audiocaps]: https://audiocaps.github.io/
[clotho]: https://zenodo.org/records/4783391
[autoacd]: https://huggingface.co/datasets/Loie/Auto-ACD
[audioset]: https://research.google.com/audioset/
[vggsound]: https://www.robots.ox.ac.uk/~vgg/data/vggsound/

## 3D Meshes

A future modality with nothing staged yet, so the corpus and its loader will land together.
The two text-richest options, each reading `corpus: format; text per object; license`:

- 🔜 [Objaverse-XL][objaverse] + [Cap3D][cap3d]: mesh → parquet; \~10M meshes, \~660K with \~10-word captions ¹; mixed, Cap3D subset CC0 / CC BY
- 🔜 [ABO][abo]: mesh → parquet; \~398K products with full e-commerce text ²; CC BY-NC 4.0

> ¹ Sketchfab / Thingiverse / Smithsonian descriptions of varied quality, plus the uniform Cap3D LLM-caption layer.
> ² Title, multi-paragraph description, feature list, dimensions, brand, and category.

[objaverse]: https://objaverse.allenai.org/
[cap3d]: https://huggingface.co/datasets/tiange/Cap3D
[abo]: https://amazon-berkeley-objects.s3.amazonaws.com/index.html

## Robotics / VLA

Vision-language-action trajectories for the VLA trainer (`image_text_act.py`, today a synthetic smoke).
Neither is staged; both need an RLDS → parquet repacker plus a `VLAParquetLoader`.

| Dataset                       | Role              | Format           | Coverage                                  |
| ----------------------------- | ----------------- | ---------------- | ----------------------------------------- |
| 🔜 [DROID][droid]             | image+text+action | RLDS → parquet ¹ | 1K-trajectory subset                      |
| 🔜 [Open-X-Embodiment][openx] | image+text+action | RLDS → parquet   | umbrella robotics corpus DROID belongs to |

> ¹ The 1K subset is acquired via `uv run scripts/data/download_droid.py` (the `hf_snapshot` source kind); episodes ship as TFDS/RLDS TFRecord, so the repacker needs a `tensorflow`/`tfds` step to flatten trajectories into `[image, instruction, action]` rows — heavier than the existing tar/JSONL repackers.

[droid]: https://droid-dataset.github.io/
[openx]: https://robotics-transformer-x.github.io/

## Multimodal Scenes

Video paired with audio, dialogue, narration, or sensor streams, for scene-level understanding and reconstruction — a future tier whose candidate set is below.
VGGSound and AudioSet (see Audio) also serve here in their audio-visual role.

Each entry reads `corpus: modalities; format; scale; license`.

- 🔜 [HD-VILA-100M][hdvila]: video + audio + ASR; video → parquet; 100M clips; MIT
- 🔜 [Ego4D][ego4d]: video + audio + speech + gaze + IMU + narration; video → parquet; 3,670 hours; research
- 🔜 [LSMDC][lsmdc]: video + audio + audio-description + dialogue; video → parquet; 200 movies, 128K clips; research
- 🔜 [Condensed Movies][condensed]: video + dialogue + plot summaries; video → parquet; 33K clips, 3.6K movies; trailers public-domain

[hdvila]: https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m
[ego4d]: https://ego4d-data.org/
[lsmdc]: https://sites.google.com/site/describingmovies/
[condensed]: https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/

## Evaluation Probes

Pulled on demand for downstream metrics; never staged into the training mixtures.

### Image-Text

| Probe                      | Modality   | Use                                  |
| -------------------------- | ---------- | ------------------------------------ |
| [MMMU][mmmu]               | image-text | Multi-discipline reasoning           |
| [MMVet][mmvet]             | image-text | VLM capability probe                 |
| [Winoground][winoground]   | image-text | Compositional reasoning              |
| [SugarCrepe][sugarcrepe]   | image-text | Negation / swap / replace robustness |
| [RefCOCO / + / g][refcoco] | image-text | Referring-expression grounding       |
| [TextVQA][textvqa]         | image-text | OCR-aware question answering         |
| [OK-VQA][okvqa]            | image-text | Outside-knowledge question answering |
| COCO 2017 val              | image-text | Image-text retrieval recall smoke    |

[mmmu]: https://huggingface.co/datasets/MMMU/MMMU
[mmvet]: https://github.com/yuweihao/MM-Vet
[winoground]: https://huggingface.co/datasets/facebook/winoground
[sugarcrepe]: https://github.com/RAIVNLab/sugar-crepe
[refcoco]: https://github.com/lichengunc/refer
[textvqa]: https://textvqa.org/
[okvqa]: https://okvqa.allenai.org/

### Text Normalization

| Probe                           | Modality | Use                                             |
| ------------------------------- | -------- | ----------------------------------------------- |
| [CoNLL-2014][conll]             | text     | GEC M²/ERRANT F0.5 gold (gated NUCLE agreement) |
| [JFLEG][jfleg]                  | text     | GEC fluency GLEU (CC BY-NC-SA; eval only)       |
| [W\&I+LOCNESS][wilocness]       | text     | GEC ERRANT P/R/F0.5 (LOCNESS non-commercial)    |
| [ICDAR-2019 Post-OCR][icdar]    | text     | CER/WER reduction vs raw OCR                    |
| [LogHub-2.0 labeled][loghub]    | text     | log template / field-extraction accuracy        |
| [Birkbeck / Holbrook][birkbeck] | text     | small real-misspelling probe                    |
| noised-query MTEB-mini          | text     | the semantic-regularizer probe ¹                |

> ¹ The decisive cell for the denoising A/B: clean vs keyboard-noised query retrieval, reading the treatment − control gap (it should widen exactly on noised queries), not the denoise loss.

[conll]: https://www.comp.nus.edu.sg/~nlp/conll14st.html
[jfleg]: https://huggingface.co/datasets/jhu-clsp/jfleg
[wilocness]: https://www.cl.cam.ac.uk/research/nl/bea2019st/
[icdar]: https://sites.google.com/view/icdar2019-postcorrectionocr
[birkbeck]: https://www.dcs.bbk.ac.uk/~roger/corpora.html
