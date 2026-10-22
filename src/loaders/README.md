# UForm Hot-Path Data Loaders

UForm achieves massively faster training compared to other AI toolkits due to extreme attention to data-flow organization.
Data copies are avoided.
OS context switches are avoided.
GPU Direct Storage is leveraged to the fullest.
And it doesn't stop there.

There are a few important constraints that shape everything:

- __Reproducible__ shuffling and batch composition must be part of training.
  Caching and pre-shuffling are an antipatterns that must be avoided in modern software.
- All data-loaders should leverage GPUs as much as possible, be it for loading data, decoding images or videos, or shuffling.
  Sometimes the bare action of moving the data from CPU to GPU costs more than the actual task, so keeping everything in one place is important.
- Dataset shards may contain boring identical data, so each data loader must load 2+ shards from disk and sample batch contents from a mixture.
  Never from the same file.
  Example: all documents in one Parquet file have the same language or identical length.
- Datasets must contain vastly non-uniform data, so using the "arena" model-oriented allocators or hoisting allocated buffers is heavily discouraged.
  Data-structures for each individual file are re-allocated dynamically on the hot path and reclaimed once the batch has been populated.
  Example: one dataset shart contains 20 Gigabytes of data, while others contain 100 Megabytes - hoisting 20 GB of owner reserved buffers per loader is a bad idea.
- To amortize the cost of opening 2+ shards per call and the cost of the dynamic memory allocations, models must pre-populate multiple batches at once.
  Those are later peeled off for serveral steps, and the process repeats.

It's also important to remember, that `src/loaders/` isn't designed in vacuum, it must be compatible with the `src/core/`.
Most importantly, remember that:

- Our models are compatible with different text lengths and dynamically pack short and long strings into segments of individual input rows.
- Our models are compatible with different media aspect ratios by using positional encoding, so we don't have to crop or distort our files towards squares.
- Our distribtued training loop shuffles rows in composed batches across the "multimem domain" via NVLink and "colocated domain" via Infiniband, so the diversity is guaranteed through scaling horizontally and not vertically.

## How a Batch Is Assembled

Every step, each GPU trains on a batch that went through the same five stages, and every rule below names the code that enforces it.

```
read     each GPU draws one dataset and reads its row groups from ≥2 shards
  │
decode   nvJPEG · NVDEC · audio demux, on that GPU
  │
pack     batches_per_fill batches of rows × columns, cut into segments and bags
  │
mix      every step, bags shuffle across all GPUs over multimem and InfiniBand
  │
train    relations → encoder → loss
```

```
loaded   GPU 0 ▣▣▣▣   GPU 1 ▤▤▤▤   GPU 2 ▥▥▥▥   GPU 3 ▦▦▦▦     one dataset per GPU
mixed    GPU 0 ▣▤▥▦   GPU 1 ▦▣▤▥   GPU 2 ▥▦▣▤   GPU 3 ▤▥▦▣     every dataset on every GPU
```

- __One dataset per GPU per fill.__
  A fill packs all its batches from one entry, so a read is paid once per period, and mixing across datasets is the mixer's job, never the fill's.
  `dataset_source_t::fill` in `src/train/datasets.cu` enforces it.
- __Staggered refills.__
  Each GPU refills on its own phase of the period, so a step pays for a share of the reads and the world's datasets rotate a few at a time.
  `dataset_source_t::phases_` holds the phases.
- __Diversity from the world.__
  The mixer routes every bag to any GPU of the colocated domain, so a post-mix batch holds segments from as many datasets as there are GPUs.
  `multimodal_batch_mixer_t` in `src/core/multimodal_batch.cu` routes them.
- __Capacity is latched, content varies.__
  Rows, segments and bags are fixed capacities of the batch, while how many items fill them is content.
  Whatever cannot fit drops whole and loudly, never split: a bag that cannot land on its destination drops in the mixer.
- __Segments share rows.__
  A segment never crosses a row, and several share one row back to back, so short texts and small images pack many per row.
- __Anchors trail.__
  The loader and the mixer both lay every other role out first and the anchors after them, so short captions pack together instead of each stranding a row beside full-row media.
  `route_destination_` in `src/core/multimodal_batch.cu` lands each bag as one run per band.
- __The post-mix batch sets the budgets.__
  The model trains on what the mixer delivers, so every capacity must hold at the destination, not only where a fill packed it.
- __Keyed and reproducible.__
  Every draw and the bijection are keyed by the seed, the step and the world coordinate, so the same seed replays the same run.
  A resume refills every GPU at its first step, so it replays its own continuation, not the uninterrupted run.
- __One host thread.__
  The calling thread drives every GPU, so overlap comes from queuing GPU work first and reading while it runs, never from loader threads.
- __Freed per fill.__
  A fill's reservoir and decode scratch are allocated for that fill and freed when it ends, while tables a fill reuses across its batches are allocated once per fill.
