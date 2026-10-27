# UForm Benchmarks

Two binaries time UForm on the machine they run on.
Each prints the version, the capabilities and its settings, then one row per benchmark with the median call in milliseconds, the rate, and the timed calls.

## Building Benchmarks

The benchmarks build with `-D UFORM_BUILD_BENCH=ON`, beside the C library:

```sh
cmake --preset release -D UFORM_BUILD_BENCH=ON
cmake --build build_release --target uform_bench_encoder uform_bench_core
```

`uform_bench_encoder` builds on every tree, the CPU-only `library` preset included, while `uform_bench_core` needs `-D UFORM_BUILD_CUDA=ON`.

## Encoder Throughput

`uform_bench_encoder` loads one checkpoint through the C interface and times blocking calls over synthetic texts that fill the model's context:

- `encode/batch_1` and `encode/batch_64`: text encoding to pooled embeddings, in tokens per second.
- `choose/batch_1` and `choose/batch_64`: scoring four choices per item, in items per second, for checkpoints carrying the byte decoder.

```sh
build_release/uform_bench_encoder --checkpoint model.safetensors
build_release/uform_bench_encoder --checkpoint model.safetensors --cpu-class intel-haswell --precision e4m3
build_release/uform_bench_encoder --checkpoint model.safetensors --residency vram --gpu-class nvidia-hopper
```

- `--checkpoint`: the `.safetensors` file to load, required.
- `--residency`: `ram` encodes on host threads and `vram` on the first GPU, `ram` by default.
- `--cpu-class`: the CPU class to dispatch as, like `serial` or `intel-sapphire`, the machine's own by default.
- `--gpu-class`: the GPU class to dispatch as, like `nvidia-ampere`, the device's own by default.
- `--precision`: the weights' quantization, `bf16` or `e4m3`, `bf16` by default.

## Training Kernels

`uform_bench_core` times the core cell's training kernels on the first GPU at depth 6 and hidden width 384, in windows of back-to-back launches between two CUDA events:

- `tower/forward_backward`: a text-only tower forward to the pooled vertices and its backward.
- `contrastive/loss`: the rank-local contrastive loss and its forces over self-edges.
- `contrastive/tokens`: the late-interaction loss and its forces, 256 queries of 64 tokens against 256 documents of 1024 tokens at width 128.

## Shared Settings

Both binaries read the same environment variables:

- `UFORM_FILTER`: the benchmarks to run by name, as an ECMAScript regex or a substring.
- `UFORM_SEED`: the seed of the synthetic inputs, an unsigned integer or `random`, 42 by default.
- `UFORM_WARMUP`: the untimed run per benchmark, like `200ms` or `1s`, `1s` by default.
- `UFORM_TIME_LIMIT`: the timed run per benchmark, like `200ms` or `10s`, `10s` by default.

```sh
UFORM_FILTER=choose UFORM_TIME_LIMIT=2s build_release/uform_bench_encoder --checkpoint model.safetensors
```
