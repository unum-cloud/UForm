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
