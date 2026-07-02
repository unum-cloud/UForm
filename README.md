<h1 align="center">UForm</h1>
<h3 align="center">
Pocket-Sized Multimodal AI<br/>
For Content Understanding and Generation<br/>
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/jsMURnSFM2"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/discord.svg" alt="Discord"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://www.linkedin.com/company/unum-cloud/"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/linkedin.svg" alt="LinkedIn"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://twitter.com/unum_cloud"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/twitter.svg" alt="Twitter"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://unum.cloud/post"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/blog.svg" alt="Blog"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://github.com/unum-cloud/uform"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/github.svg" alt="GitHub"></a>
</p>

<p align="center">
Multimodal Perception from 64 to 1024 Dimensions
<br/>
Text • Images • Videos • Audio • Code • Graphs
<br/>
CUDA Training • CUDA Inference • ONNX • CoreML
<br/>
<a href="https://github.com/unum-cloud/uform/blob/main/python/README.md">Python</a>
 • 
<a href="https://github.com/unum-cloud/uform/blob/main/javascript/README.md">JavaScript</a>
 • 
<a href="https://github.com/unum-cloud/uform/blob/main/swift/README.md">Swift</a>
</p>

---

UForm 4 is a family of neural networks trained from scratch to learn __shared multi-modal representations__ across text, images, audio, video, and code on a GPU, a single DGX node, or an entire cluster, designed bottom-up from scratch, to fully leverage modern hardware.
A few convictions shape the whole stack:

- __Communication buys batch size buys quality__ — custom lower-level cross-GPU collectives unlock far larger global batches, and with contrastive objectives more in-batch negatives directly sharpen the learned space.
- __The data path is half the model__ — for IO-heavy modalities like images and video, the parallel CPU/GPU decode-and-load scaffolding is as worth optimizing as the forward and backward passes.
- __Lower precision wins twice__ — narrow number formats drove most of the last decade's FLOP/s while moving fewer bytes: more tensor-core throughput _and_ less traffic across the memory wall, from GPU SRAM/HBM to CPU L1/L2.
  UForm trains in mixed BFloat16/Float32 and serves scale-free E4M3 Float8 from one `mode` flag — quantization built in, not bolted on — for pocket-sized weights and 2.5× faster Float8 matmuls on Blackwell.
- __A shared uniform encoder across modalities__ — attention-pooling the output of a single same-width encoder into one pooled embedding is the most direct route to emergent cross-modal transfer.
- __Non-autoregressive generation__ — emitting the entire output in one forward pass instead of token-by-token pushes single-GPU throughput up by one or two orders of magnitude for tasks that can be ahndled with such models — from ≈100 tokens/s to up to ≈10'000.

This repository is the full training and inference stack, tuned for NVIDIA Hopper and Blackwell GPUs: custom CUDA kernels, Python trainers, and GPU-accelerated loaders & decoders for IO-heavy modalities.
The trained model carries __no PyTorch or JAX dependency__ — those appear only as offline parity oracles that validate the CUDA kernels.
There is not even NCCL or NVSHMEM, no dynamic memory allcoations on the hot paths, locks, mutexes, or even need for multiple threads.

