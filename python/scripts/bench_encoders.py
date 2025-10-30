#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UForm Multimodal Embedding Model Benchmark Script

Benchmarks image and text encoding throughput across different models, devices, and backends.

Installation:
    uv pip install -e ".[torch,onnx-gpu,dev]"

Usage Examples:
    # Quick GPU benchmark (PyTorch only, recommended for B200/H100)
    uv run python python/scripts/bench_encoders.py --gpu --torch --batch-size 256

    # Compare ONNX vs PyTorch performance on GPU
    uv run python python/scripts/bench_encoders.py --gpu --batch-size 512

    # CPU-only benchmarks with small batch size
    uv run python python/scripts/bench_encoders.py --cpu --batch-size 10

    # Verbose output showing all metrics (latency, efficiency, etc.)
    uv run python python/scripts/bench_encoders.py --gpu --torch -v

    # Stress test GPU memory limits (B200 has 192GB)
    uv run python python/scripts/bench_encoders.py --gpu --torch --batch-size 8192

    # Run only on small model (filter out others)
    uv run python python/scripts/bench_encoders.py --gpu --torch --batch-size 2048 --filter "base|large|multilingual"

Output Metrics:
    Default:
        - Model, Device, Backend, Images/s, Texts/s, Precision, VRAM(GB), GPU Compute%, Power(W)

    Verbose (-v/--verbose):
        - All above plus: Params(M), Size(MB), Img Latency (ms), Text Latency (ms),
          Avg Tokens, VRAM%, Imgs/s/W, Imgs/s/GB, CPU%

Flags:
    --cpu              Enable CPU benchmarks (default: nothing runs unless explicitly enabled)
    --gpu              Enable GPU benchmarks (default: nothing runs unless explicitly enabled)
    --torch            Enable PyTorch backend (default: nothing runs unless explicitly enabled)
    --onnx             Enable ONNX backend (default: nothing runs unless explicitly enabled)
    --preprocessing    Benchmark preprocessing performance (default: preprocess once and replicate)
    --batch-size N     Batch size (default: 50). Use 1 for latency, 256+ for GPU throughput
    --filter REGEX     Filter out models matching regex pattern (e.g., "base|large")
    -v, --verbose      Show all metrics including latencies and efficiency ratios

Notes:
    - A warm-up is performed before benchmarking to load models and fill caches
    - GPU% refers to GPU compute utilization (not memory utilization)
    - VRAM% is the percentage of total GPU memory used
    - For accurate GPU benchmarks, use batch sizes 256+ to saturate the GPU
"""

from functools import partial
from time import perf_counter
from dataclasses import dataclass
from typing import List, Tuple, Literal, Callable, Generator
import re
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image
import pandas as pd

from uform import get_model, Modality, ExecutionProviderError

# Auto-configure TensorRT library path for ONNX Runtime
# TensorRT libraries are installed via tensorrt-cu12 but need to be in LD_LIBRARY_PATH
try:
    import tensorrt_libs
    tensorrt_lib_path = Path(tensorrt_libs.__file__).parent
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if str(tensorrt_lib_path) not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{tensorrt_lib_path}:{current_ld_path}"
except ImportError:
    pass  # TensorRT not installed, ONNX will use CUDA provider only

# Define global constants for the hardware availability
torch_available = False
try:
    import torch

    torch_available = True
except ImportError:
    pass
onnx_available = False
try:
    import onnx

    onnx_available = True
except ImportError:
    pass
cuda_available = False
try:
    if torch_available:
        cuda_available = torch.cuda.is_available()
    elif onnx_available:
        import onnxruntime

        cuda_available = onnxruntime.get_device() == "GPU"
except ImportError:
    pass

# Hardware monitoring libraries
psutil_available = False
try:
    import psutil

    psutil_available = True
except ImportError:
    pass

pynvml_available = False
try:
    import pynvml

    pynvml.nvmlInit()
    pynvml_available = True
except:
    pass


# Helper functions for monitoring and analysis
def get_model_info(model, backend_name: str, model_path: str = None):
    """Extract model parameter count, size, and precision."""
    info = {"params_millions": 0, "size_mb": 0, "precision": "unknown"}

    if backend_name == "torch" and hasattr(model, "parameters"):
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        info["params_millions"] = round(total_params / 1e6, 1)

        # Calculate size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        info["size_mb"] = round((param_size + buffer_size) / (1024**2), 1)

        # Detect precision
        if hasattr(model, "parameters"):
            first_param = next(model.parameters(), None)
            if first_param is not None:
                dtype_str = str(first_param.dtype).replace("torch.", "")
                info["precision"] = dtype_str

    elif backend_name == "onnx" and model_path:
        # Get ONNX file size
        import os

        if os.path.exists(model_path):
            info["size_mb"] = round(os.path.getsize(model_path) / (1024**2), 1)

    return info


def get_gpu_metrics(device_index: int = 0):
    """Get GPU utilization, memory, and power metrics."""
    metrics = {
        "gpu_util_percent": 0,
        "gpu_mem_used_mb": 0,
        "gpu_mem_total_mb": 0,
        "gpu_mem_util_percent": 0,
        "gpu_power_watts": 0,
    }

    if pynvml_available:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            metrics["gpu_util_percent"] = util.gpu
            metrics["gpu_mem_used_mb"] = round(mem_info.used / (1024**2), 0)
            metrics["gpu_mem_total_mb"] = round(mem_info.total / (1024**2), 0)
            metrics["gpu_mem_util_percent"] = round((mem_info.used / mem_info.total) * 100, 1)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                metrics["gpu_power_watts"] = round(power, 1)
            except:
                pass
        except:
            pass
    elif torch_available and cuda_available:
        # Fallback to torch.cuda for memory only
        try:
            metrics["gpu_mem_used_mb"] = round(torch.cuda.memory_allocated(device_index) / (1024**2), 0)
            metrics["gpu_mem_total_mb"] = round(
                torch.cuda.get_device_properties(device_index).total_memory / (1024**2), 0
            )
            if metrics["gpu_mem_total_mb"] > 0:
                metrics["gpu_mem_util_percent"] = round(
                    (metrics["gpu_mem_used_mb"] / metrics["gpu_mem_total_mb"]) * 100, 1
                )
        except:
            pass

    return metrics


def get_cpu_metrics():
    """Get CPU utilization and system memory usage."""
    metrics = {"cpu_util_percent": 0, "sys_mem_used_gb": 0, "sys_mem_total_gb": 0}

    if psutil_available:
        try:
            metrics["cpu_util_percent"] = round(psutil.cpu_percent(interval=0.1), 1)
            mem = psutil.virtual_memory()
            metrics["sys_mem_used_gb"] = round(mem.used / (1024**3), 1)
            metrics["sys_mem_total_gb"] = round(mem.total / (1024**3), 1)
        except:
            pass

    return metrics


def get_token_stats(processor, captions):
    """Analyze token counts in text data."""
    stats = {"avg_tokens": 0, "min_tokens": 0, "max_tokens": 0}

    try:
        tokens_data = processor(captions)

        if isinstance(tokens_data, dict) and "input_ids" in tokens_data:
            input_ids = tokens_data["input_ids"]
            attention_mask = tokens_data.get("attention_mask")

            if attention_mask is not None:
                # Count non-padding tokens using attention mask
                if hasattr(attention_mask, "numpy"):
                    attention_mask = attention_mask.numpy()
                elif hasattr(attention_mask, "cpu"):
                    attention_mask = attention_mask.cpu().numpy()

                token_counts = attention_mask.sum(axis=1)
                stats["avg_tokens"] = round(float(token_counts.mean()), 1)
                stats["min_tokens"] = int(token_counts.min())
                stats["max_tokens"] = int(token_counts.max())
    except:
        pass

    return stats


@dataclass
class BenchmarkResult:
    model_name: str
    device_name: Literal["cpu", "cuda"] = "cpu"
    backend_name: Literal["torch", "onnx"] = "torch"
    duration_image_preprocessing: float = 0
    duration_image_embedding: float = 0
    duration_text_preprocessing: float = 0
    duration_text_embedding: float = 0
    # Model info
    params_millions: float = 0
    size_mb: float = 0
    precision: str = "unknown"
    # Hardware metrics
    gpu_util_percent: float = 0
    gpu_mem_used_mb: float = 0
    gpu_mem_util_percent: float = 0
    gpu_power_watts: float = 0
    cpu_util_percent: float = 0
    # Token statistics
    avg_tokens_per_text: float = 0
    # Batch size
    batch_size: int = 0

    @property
    def images_per_sec(self) -> float:
        """Throughput: images processed per second."""
        return round(1 / self.duration_image_embedding, 2) if self.duration_image_embedding else 0

    @property
    def texts_per_sec(self) -> float:
        """Throughput: texts processed per second."""
        return round(1 / self.duration_text_embedding, 2) if self.duration_text_embedding else 0

    @property
    def img_latency_ms(self) -> float:
        """Latency: milliseconds per image."""
        return round(self.duration_image_embedding * 1000, 2) if self.duration_image_embedding else 0

    @property
    def text_latency_ms(self) -> float:
        """Latency: milliseconds per text."""
        return round(self.duration_text_embedding * 1000, 2) if self.duration_text_embedding else 0

    @property
    def gpu_mem_used_gb(self) -> float:
        """GPU memory in gigabytes."""
        return round(self.gpu_mem_used_mb / 1024, 1)

    @property
    def imgs_per_sec_per_watt(self) -> float:
        """Power efficiency: images per second per watt."""
        return round(self.images_per_sec / self.gpu_power_watts, 2) if self.gpu_power_watts else 0

    @property
    def imgs_per_sec_per_gb(self) -> float:
        """Memory efficiency: images per second per GB of VRAM."""
        gb = self.gpu_mem_used_gb
        return round(self.images_per_sec / gb, 2) if gb else 0

    def to_dict_with_computed(self) -> dict:
        """Convert to dict including computed properties."""
        data = self.__dict__.copy()
        # Add computed properties
        data.update({
            'images_per_sec': self.images_per_sec,
            'texts_per_sec': self.texts_per_sec,
            'img_latency_ms': self.img_latency_ms,
            'text_latency_ms': self.text_latency_ms,
            'gpu_mem_used_gb': self.gpu_mem_used_gb,
            'imgs_per_sec_per_watt': self.imgs_per_sec_per_watt,
            'imgs_per_sec_per_gb': self.imgs_per_sec_per_gb,
        })
        return data


# Column metadata: (display_name, verbose_only)
COLUMN_METADATA = {
    "model_name": ("Model", False),
    "device_name": ("Device", False),
    "backend_name": ("Backend", False),
    "batch_size": ("Batch", True),
    "params_millions": ("Params(M)", True),
    "size_mb": ("Size(MB)", True),
    "precision": ("Precision", False),
    "images_per_sec": ("Images/s", False),
    "texts_per_sec": ("Texts/s", False),
    "img_latency_ms": ("Img Latency (ms)", True),
    "text_latency_ms": ("Text Latency (ms)", True),
    "avg_tokens_per_text": ("Avg Tokens", True),
    "gpu_util_percent": ("GPU Compute%", False),
    "gpu_mem_used_gb": ("VRAM(GB)", False),
    "gpu_mem_util_percent": ("VRAM%", True),
    "gpu_power_watts": ("Power(W)", False),
    "imgs_per_sec_per_watt": ("Imgs/s/W", True),
    "imgs_per_sec_per_gb": ("Imgs/s/GB", True),
    "cpu_util_percent": ("CPU%", True),
}


def get_display_columns(verbose: bool) -> list:
    """Get columns to display based on verbose mode."""
    return [
        col for col, (_, verbose_only) in COLUMN_METADATA.items()
        if not verbose_only or verbose
    ]


def get_column_names() -> dict:
    """Get display names for all columns."""
    return {col: display_name for col, (display_name, _) in COLUMN_METADATA.items()}


def print_result_row(result: BenchmarkResult, verbose: bool = False):
    """Print a single benchmark result as a formatted row."""
    # Format: Model (truncated) | Device | Backend | Images/s | Texts/s | Precision | VRAM(GB) | GPU% | Power(W)
    model_short = result.model_name.split('/')[-1][:40]  # Truncate model name
    row = (
        f"✓ {model_short:40} | {result.device_name:4} | {result.backend_name:5} | "
        f"{result.images_per_sec:8.1f} | {result.texts_per_sec:8.1f} | "
        f"{result.precision:9} | {result.gpu_mem_used_gb:7.1f} | "
        f"{result.gpu_util_percent:6.0f} | {result.gpu_power_watts:7.1f}"
    )
    if verbose:
        row += f" | Lat: {result.img_latency_ms:.2f}ms"
    print(row)


def benchmark_preprocessing_adaptive(processor_func, data, batch_size: int) -> float:
    """Benchmark preprocessing with adaptive iteration count based on batch size.

    For small batches: run more iterations for stable timing
    For large batches: run fewer iterations to save time
    """
    if batch_size <= 10:
        max_iterations = 100
        min_duration = 10.0
    elif batch_size <= 256:
        max_iterations = 20
        min_duration = 5.0
    else:  # Large batches: just do 5-10 runs
        max_iterations = 10
        min_duration = 2.0

    total_duration = 0
    total_iterations = 0

    while total_duration < min_duration and total_iterations < max_iterations:
        seconds, _ = duration(lambda: processor_func(data))
        total_duration += seconds
        total_iterations += len(data)

    return total_duration / total_iterations if total_iterations else 0


def should_run(spec: BenchmarkResult, args, filter_regex: str = None) -> bool:
    """Check if benchmark spec matches user-selected filters.

    Args:
        spec: Benchmark specification
        args: Argparse arguments with --cpu, --gpu, --torch, --onnx flags
        filter_regex: Optional regex pattern to filter out models/backends/devices

    Returns:
        True if benchmark should run, False otherwise
    """
    # Check regex filter first
    if filter_regex:
        pattern = re.compile(filter_regex)
        if any(pattern.search(s) for s in [spec.model_name, spec.backend_name, spec.device_name]):
            return False

    # If no device/backend flags specified, run NOTHING (user must be explicit)
    if not (args.cpu or args.gpu or args.torch or args.onnx):
        return False

    # Check device selection
    device_ok = False
    if args.cpu and spec.device_name == "cpu":
        device_ok = True
    if args.gpu and spec.device_name == "cuda":
        device_ok = True

    # Check backend selection
    backend_ok = False
    if args.torch and spec.backend_name == "torch":
        backend_ok = True
    if args.onnx and spec.backend_name == "onnx":
        backend_ok = True

    # If no device flags set, allow any device. Same for backend
    if not (args.cpu or args.gpu):
        device_ok = True
    if not (args.torch or args.onnx):
        backend_ok = True

    return device_ok and backend_ok


def duration(callable, synchronize=False):
    """Profile the duration of a callable and return the duration and the result."""
    if synchronize and torch_available and cuda_available:
        torch.cuda.synchronize()  # Wait for CUDA operations to complete
    start = perf_counter()
    result = callable()
    if synchronize and torch_available and cuda_available:
        torch.cuda.synchronize()  # Ensure all CUDA kernels have finished
    stop = perf_counter()
    return stop - start, result


def get_captioned_images() -> List[Tuple[Image.Image, str]]:
    """Get a list of pre-downloaded and decoded images and their captions."""
    image_urls = [
        "https://images.unsplash.com/photo-1697665666330-7acf230fa830?q=80&w=2787&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1695653422543-7da6d6744364?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDF8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1703244551371-ecffad9cc3b6?q=80&w=2859&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_photo-1702910931866-2642eee270b1?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_photo-1700583712241-893aded49e69?q=80&w=2942&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    ]
    images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
    captions = [
        "lonely house in a beautiful valley. house is made of white wood and black bricks. its surrounded by a green field",
        "grab last-mile delivery driver on a scooter grabbing a delivery in Jakarta",
        "monochrome picture of new york in the late 2th century on a sunny day, showing a few canonical brick buildings and the citizens bank",
        "asian girl sleeping in a bed. top down view",
        "a few food containers, with past, corn, olives, and sliced red & green peppers, with a man pouring sous on top of it",
    ]
    return list(zip(images, captions))


def yield_benchmarks(batch_size: int, args=None) -> Generator[Tuple[BenchmarkResult, Callable], None, None]:
    """Yields callable benchmarks for all supported backends of the given model."""

    # Pull the content and artificially grow the batch size
    images, captions = zip(*get_captioned_images())

    if len(images) < batch_size:
        import math

        multiplier = int(math.ceil(batch_size / len(images)))
        images *= multiplier
        captions *= multiplier
    images = images[:batch_size]
    captions = captions[:batch_size]

    def run(model_name: str, device: str, backend_name: str):
        result = BenchmarkResult(
            model_name=model_name,
            backend_name=backend_name,
            device_name=device,
            duration_image_preprocessing=0,
            duration_image_embedding=0,
            duration_text_preprocessing=0,
            duration_text_embedding=0,
            batch_size=batch_size,
        )

        sync = backend_name == "torch"
        processors, models = get_model(
            model_name,
            device=device,
            modalities=[Modality.IMAGE_ENCODER, Modality.TEXT_ENCODER],
            backend=backend_name,
        )

        model_text = models[Modality.TEXT_ENCODER]
        model_image = models[Modality.IMAGE_ENCODER]
        processor_text = processors[Modality.TEXT_ENCODER]
        processor_image = processors[Modality.IMAGE_ENCODER]

        # Collect model info
        model_info = get_model_info(model_image, backend_name)
        result.params_millions = model_info["params_millions"]
        result.size_mb = model_info["size_mb"]
        result.precision = model_info["precision"]

        # Collect token statistics
        token_stats = get_token_stats(processor_text, captions)
        result.avg_tokens_per_text = token_stats["avg_tokens"]

        # Image preprocessing
        if args and args.preprocessing:
            # Benchmark preprocessing with adaptive iteration count
            result.duration_image_preprocessing = benchmark_preprocessing_adaptive(
                processor_image, images, batch_size
            )
            images_data = processor_image(images)
        else:
            # Skip preprocessing benchmark - preprocess once and replicate
            result.duration_image_preprocessing = 0
            single_image_data = processor_image([images[0]])

            # Replicate preprocessed data to fill batch
            if isinstance(single_image_data, dict):
                # Check if dict contains torch tensors or numpy arrays
                first_value = next(iter(single_image_data.values()))
                if backend_name == "torch":
                    # PyTorch tensors
                    images_data = {k: v.repeat(batch_size, *([1] * (v.dim() - 1)))
                                  for k, v in single_image_data.items()}
                else:
                    # ONNX backend: Numpy arrays
                    images_data = {k: np.repeat(v, batch_size, axis=0)
                                  for k, v in single_image_data.items()}
            else:
                # ONNX image processor returns numpy array directly
                images_data = np.repeat(single_image_data, batch_size, axis=0)

        # Image embedding
        total_duration = 0
        total_iterations = 0
        while total_duration < 10 and total_iterations < 100:
            seconds, _ = duration(lambda: model_image.encode(images_data), synchronize=sync)
            total_duration += seconds
            total_iterations += len(images)
        duration_per_iteration = total_duration / total_iterations
        result.duration_image_embedding = duration_per_iteration

        # Text preprocessing
        if args and args.preprocessing:
            # Benchmark preprocessing with adaptive iteration count
            result.duration_text_preprocessing = benchmark_preprocessing_adaptive(
                processor_text, captions, batch_size
            )
            texts_data = processor_text(captions)
        else:
            # Skip preprocessing benchmark - preprocess once and replicate
            result.duration_text_preprocessing = 0
            single_text_data = processor_text([captions[0]])

            # Replicate preprocessed data to fill batch
            if isinstance(single_text_data, dict):
                if backend_name == "torch":
                    # PyTorch tensors
                    texts_data = {k: v.repeat(batch_size, *([1] * (v.dim() - 1)))
                                 for k, v in single_text_data.items()}
                else:
                    # ONNX backend: Numpy arrays
                    texts_data = {k: np.repeat(v, batch_size, axis=0)
                                 for k, v in single_text_data.items()}
            else:
                # Single array (shouldn't happen for text, but handle it)
                texts_data = np.repeat(single_text_data, batch_size, axis=0)

        # Text embedding
        total_duration = 0
        total_iterations = 0
        while total_duration < 10 and total_iterations < 100:
            seconds, _ = duration(lambda: model_text.encode(texts_data), synchronize=sync)
            total_duration += seconds
            total_iterations += len(captions)
        duration_per_iteration = total_duration / total_iterations
        result.duration_text_embedding = duration_per_iteration

        # Collect hardware metrics after benchmarking
        if device == "cuda":
            gpu_metrics = get_gpu_metrics(device_index=0)
            result.gpu_util_percent = gpu_metrics["gpu_util_percent"]
            result.gpu_mem_used_mb = gpu_metrics["gpu_mem_used_mb"]
            result.gpu_mem_util_percent = gpu_metrics["gpu_mem_util_percent"]
            result.gpu_power_watts = gpu_metrics["gpu_power_watts"]

        cpu_metrics = get_cpu_metrics()
        result.cpu_util_percent = cpu_metrics["cpu_util_percent"]

        return result

    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")
    backends = []
    if torch_available:
        backends.append("torch")
    if onnx_available:
        backends.append("onnx")

    for device in devices:
        for backend_name in backends:
            for model_name in [
                "unum-cloud/uform3-image-text-english-small",
                "unum-cloud/uform3-image-text-english-base",
                "unum-cloud/uform3-image-text-english-large",
                "unum-cloud/uform3-image-text-multilingual-base",
            ]:
                yield BenchmarkResult(
                    model_name=model_name,
                    device_name=device,
                    backend_name=backend_name,
                ), partial(run, model_name, device, backend_name)


def main(args=None, batch_size: int = 10, verbose: bool = False):
    # Print condensed benchmark configuration
    print("=" * 80)
    print(f"UForm Benchmark - Batch Size: {batch_size}")
    print("=" * 80)

    # Hardware info
    if torch_available and cuda_available:
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if len(gpus) == 1:
            gpu_str = gpus[0]
        elif len(gpus) <= 3:
            gpu_str = ", ".join(gpus)
        else:
            gpu_str = f"{len(gpus)}x {gpus[0]}"
        print(f"Hardware: {gpu_str} (CUDA {torch.version.cuda})")
    else:
        print(f"Hardware: CPU only")

    # Software versions
    versions = []
    if torch_available:
        versions.append(f"PyTorch {torch.__version__}")
    if onnx_available:
        import onnxruntime as ort
        versions.append(f"ONNX {ort.__version__}")
    if versions:
        print(f"Backends: {', '.join(versions)}")

    # Only show warnings for missing monitoring tools
    missing = []
    if not pynvml_available and cuda_available:
        missing.append("nvidia-ml-py (limited GPU metrics)")
    if not psutil_available:
        missing.append("psutil (no CPU metrics)")
    if missing:
        print(f"⚠️  Missing: {', '.join(missing)}")

    print("=" * 80)
    print()

    # Print progressive results header
    print("Live Results:")
    header_row = (
        f"  {'Model':40} | {'Dev':4} | {'Back':5} | "
        f"{'Imgs/s':>8} | {'Txts/s':>8} | "
        f"{'Precision':>9} | {'VRAM(GB)':>7} | {'GPU%':>6} | {'Power(W)':>7}"
    )
    print(header_row)
    print("  " + "-" * (len(header_row) - 2))

    results = []
    filter_regex = args.filter if args else None
    for specs, func in yield_benchmarks(batch_size=batch_size, args=args):
        # Check if this benchmark should run based on CLI flags and filter
        if args and not should_run(specs, args, filter_regex):
            continue

        try:
            print(f"  Running {specs.model_name}/{specs.device_name}/{specs.backend_name}...", end=" ", flush=True)
            result = func()
            results.append(result)
            print()  # New line after completion
            print_result_row(result, verbose=verbose)
        except ExecutionProviderError as e:
            print(f"\n  ✗ Skipped - missing backend")
            if verbose:
                print(f"     {e}")

    # Check if any benchmarks were run
    if not results:
        print()
        print("=" * 80)
        print("No benchmarks were run!")
        print("=" * 80)
        print("Tip: You must explicitly enable device and/or backend flags:")
        print("  --cpu      Enable CPU benchmarks")
        print("  --gpu      Enable GPU/CUDA benchmarks")
        print("  --torch    Enable PyTorch backend")
        print("  --onnx     Enable ONNX backend")
        print()
        print("Examples:")
        print("  python bench_encoders.py --gpu --torch")
        print("  python bench_encoders.py --cpu --onnx")
        print("  python bench_encoders.py --gpu --batch-size 512")
        print()
        return

    # Convert results to DataFrame with computed properties
    results = sorted(results, key=lambda x: x.model_name)
    results_dicts = [r.to_dict_with_computed() for r in results]
    df = pd.DataFrame(results_dicts)

    # Get display columns and names from metadata
    display_columns = get_display_columns(verbose)
    column_names = get_column_names()

    df_display = df[display_columns].rename(columns=column_names)

    # Print final summary table
    print()
    print("=" * 80)
    print("Final Summary Table")
    print("=" * 80)
    print(df_display.to_markdown(index=False, floatfmt=".1f"))
    print()
    print("=" * 80)
    print(f"Statistics")
    print("=" * 80)
    print(f"Batch Size: {batch_size}")
    print(f"Models Tested: {len(df['model_name'].unique())}")
    print(f"Configurations: {len(df)} (device x backend combinations)")
    print()

    # Performance warnings (only for CUDA runs)
    if pynvml_available:
        low_util_runs = df[(df["device_name"] == "cuda") & (df["gpu_util_percent"] < 50)]
        if not low_util_runs.empty:
            print("⚠️  WARNING: Low GPU Utilization Detected on CUDA runs:")
            for _, row in low_util_runs.iterrows():
                print(
                    f"  - {row['model_name']} ({row['backend_name']}): {row['gpu_util_percent']:.0f}% GPU compute utilization"
                )
            print("  Tip: Increase batch size or check for CPU bottlenecks in preprocessing")
            print()

    print("=" * 80)
    print("Tips:")
    print("  - For latency measurement: use --batch-size 1")
    print("  - For GPU throughput: use --batch-size 256-2048")
    print("  - For memory stress testing: use --batch-size 4096-8192")
    print("  - For GPU-only benchmarks: use --gpu --torch")
    print("  - For CPU-only benchmarks: use --cpu --batch-size 10")
    print("  - For backend comparison: use --gpu (compares torch vs onnx)")
    if not pynvml_available or not psutil_available:
        print()
        print("  📊 Install monitoring tools for full metrics:")
        if not pynvml_available:
            print("     pip install pynvml  (for detailed GPU metrics)")
        if not psutil_available:
            print("     pip install psutil  (for CPU monitoring)")
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Device and backend selectors
    parser.add_argument("--cpu", action="store_true", help="Enable CPU benchmarks")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU/CUDA benchmarks")
    parser.add_argument("--torch", action="store_true", help="Enable PyTorch backend")
    parser.add_argument("--onnx", action="store_true", help="Enable ONNX backend")

    # Model/backend/device filter
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter out models, backends, or devices matching this regex pattern (e.g., 'base|large')",
    )

    # Batch size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for the benchmark. Batch size 1 measures latency. Larger batch sizes (50-256) show GPU throughput. Default: 50 (recommended for GPU benchmarking).",
    )

    # Preprocessing benchmark flag
    parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Benchmark preprocessing performance (by default, preprocessing is done once and replicated)",
    )

    # Verbose output
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all metrics including latencies, efficiency ratios, and detailed stats",
    )

    args = parser.parse_args()

    main(args=args, batch_size=args.batch_size, verbose=args.verbose)
