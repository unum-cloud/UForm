from os.path import join, exists
from typing import Dict, Optional, Tuple, Literal, Union, Callable

from huggingface_hub import snapshot_download, utils

from uform.shared import ExecutionProviderError, Modality


def _normalize_modalities(modalities: Tuple[str, Modality]) -> Tuple[Modality]:
    if modalities is None:
        return (Modality.TEXT_ENCODER, Modality.IMAGE_ENCODER, Modality.TEXT_DECODER, Modality.VIDEO_ENCODER)

    return tuple(x if isinstance(x, Modality) else Modality(x) for x in modalities)


def get_checkpoint(
    model_name: str,
    modalities: Tuple[str, Modality],
    token: Optional[str] = None,
    format: Literal[".pt", ".onnx"] = ".pt",
) -> Tuple[str, Dict[Modality, str], Optional[str]]:
    """Downloads a model checkpoint from the Hugging Face Hub.

    :param model_name: The name of the model to download, like `unum-cloud/uform3-image-text-english-small`
    :param token: The Hugging Face API token, if required
    :param modalities: The modalities to download, like `("text_encoder", "image_encoder")`
    :param format: The format of the model checkpoint, either `.pt` or `.onnx`
    :return: A tuple of the config path, dictionary of paths to different modalities, and tokenizer path
    """

    modalities = _normalize_modalities(modalities)

    # It is not recommended to use `.pth` extension when checkpointing models
    # because it collides with Python path (`.pth`) configuration files.
    merged_model_names = [x + format for x in ["torch_weight", "weight", "model"]]
    separate_modality_names = [(x.value if isinstance(x, Modality) else x) + format for x in modalities]
    config_names = ["torch_config.json", "config.json"]
    tokenizer_names = ["tokenizer.json"]

    old_progress_behavior = utils.are_progress_bars_disabled()
    utils.disable_progress_bars()

    # The download stats depend on the number of times the `config.json` is pulled
    # https://huggingface.co/docs/hub/models-download-stats
    model_path = snapshot_download(
        repo_id=model_name,
        token=token,
        allow_patterns=merged_model_names + separate_modality_names + config_names + tokenizer_names,
    )

    if old_progress_behavior:
        utils.enable_progress_bars()

    # Find the first name in `config_names` that is present
    config_path = None
    for config_name in config_names:
        if exists(join(model_path, config_name)):
            config_path = join(model_path, config_name)
            break

    # Same for the tokenizer
    tokenizer_path = None
    for tokenizer_name in tokenizer_names:
        if exists(join(model_path, tokenizer_name)):
            tokenizer_path = join(model_path, tokenizer_name)
            break

    # Ideally, we want to separately fetch all the models.
    # If those aren't available, aggregate separate modalities and merge them.
    modality_paths = None
    for file_name in merged_model_names:
        if exists(join(model_path, file_name)):
            modality_paths = join(model_path, file_name)
            break

    if modality_paths is None:
        modality_paths = {}
        for separate_modality_name in separate_modality_names:
            if exists(join(model_path, separate_modality_name)):
                modality_name, _, _ = separate_modality_name.partition(".")
                modality_paths[Modality(modality_name)] = join(model_path, separate_modality_name)

    return config_path, modality_paths, tokenizer_path


def get_model_torch(
    model_name: str,
    *,
    token: Optional[str] = None,
    device: Literal["cpu", "cuda"] = "cpu",
    modalities: Optional[Tuple[Union[str, Modality]]] = None,
    dtype: Optional[str] = "bfloat16",
) -> Tuple[Dict[Modality, Callable], Dict]:
    """
    Fetches and constructs a PyTorch model with its processors based on provided modalities.

    :param model_name: The identifier of the model on the Hugging Face Hub.
    :param token: Optional API token for authenticated access to the model.
    :param device: The device to load the model onto ('cpu' or 'cuda').
    :param modalities: A tuple specifying the types of model components to fetch (e.g., text encoder).
    :param dtype: Model precision/dtype. Options: "float32", "float16", "bfloat16" (default).
                  bfloat16 provides ~2x speedup with minimal accuracy loss on modern hardware.
                  Works on CPUs with AVX-512 BF16 support and all modern CUDA GPUs.
                  Set to None to keep original float32 precision.
    :return: A tuple containing dictionaries for processors and models keyed by their respective modalities.
    """
    import torch
    from uform.torch_encoders import TextEncoder, ImageEncoder
    from uform.torch_processors import TextProcessor, ImageProcessor

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: None,
    }

    if dtype not in dtype_map:
        raise ValueError(f"Invalid dtype: {dtype}. Must be one of {list(dtype_map.keys())}")

    torch_dtype = dtype_map[dtype]

    modalities = _normalize_modalities(modalities)
    config_path, modality_paths, tokenizer_path = get_checkpoint(model_name, modalities, token=token, format=".pt")

    result_processors = {}
    result_models = {}

    if Modality.TEXT_ENCODER in modalities:
        processor = TextProcessor(config_path, tokenizer_path)
        encoder = TextEncoder.from_pretrained(config_path, modality_paths.get(Modality.TEXT_ENCODER))
        encoder = encoder.eval().to(device)
        if torch_dtype is not None:
            encoder = encoder.to(dtype=torch_dtype)
        result_processors[Modality.TEXT_ENCODER] = processor
        result_models[Modality.TEXT_ENCODER] = encoder

    if Modality.IMAGE_ENCODER in modalities:
        processor = ImageProcessor(config_path)
        encoder = ImageEncoder.from_pretrained(config_path, modality_paths.get(Modality.IMAGE_ENCODER))
        encoder = encoder.eval().to(device)
        if torch_dtype is not None:
            encoder = encoder.to(dtype=torch_dtype)
        result_processors[Modality.IMAGE_ENCODER] = processor
        result_models[Modality.IMAGE_ENCODER] = encoder

    return result_processors, result_models


def get_model_onnx(
    model_name: str,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    token: Optional[str] = None,
    modalities: Optional[Tuple[str]] = None,
):
    """
    Fetches and constructs an ONNX model with its processors based on provided modalities.

    :param model_name: The identifier of the model on the Hugging Face Hub.
    :param device: The device on which the model will operate ('cpu' or 'cuda').
    :param token: Optional API token for authenticated access to the model.
    :param modalities: A tuple specifying the types of model components to fetch (e.g., text encoder).
    :return: A tuple containing dictionaries for processors and models keyed by their respective modalities.
    """
    from uform.onnx_encoders import TextEncoder, ImageEncoder
    from uform.numpy_processors import TextProcessor, ImageProcessor

    modalities = _normalize_modalities(modalities)
    config_path, modality_paths, tokenizer_path = get_checkpoint(model_name, modalities, token=token, format=".onnx")

    result_processors = {}
    result_models = {}

    if Modality.TEXT_ENCODER in modalities:
        processor = TextProcessor(config_path, tokenizer_path)
        encoder = TextEncoder(modality_paths.get(Modality.TEXT_ENCODER), device=device)
        result_processors[Modality.TEXT_ENCODER] = processor
        result_models[Modality.TEXT_ENCODER] = encoder

    if Modality.IMAGE_ENCODER in modalities:
        processor = ImageProcessor(config_path)
        encoder = ImageEncoder(modality_paths.get(Modality.IMAGE_ENCODER), device=device)
        result_processors[Modality.IMAGE_ENCODER] = processor
        result_models[Modality.IMAGE_ENCODER] = encoder

    return result_processors, result_models


def get_model(
    model_name: str,
    *,
    device: Literal["cpu", "cuda"] = "cpu",  # change this if you have a GPU
    backend: Literal["onnx", "torch"] = "onnx",  # lighter = better
    modalities: Optional[Tuple[str, Modality]] = None,  # all by default
    token: Optional[str] = None,  # optional HuggingFace Hub token for private models
    dtype: Optional[str] = "bfloat16",  # precision for torch backend (float32, float16, bfloat16)
) -> Tuple[Dict[Modality, Callable], Dict]:
    """
    Fetches a model and its processors from the Hugging Face Hub, using either the ONNX or Torch backend.

    :param model_name: The identifier of the model on the Hugging Face Hub.
    :param device: The device to load the model onto ('cpu' or 'cuda').
    :param backend: The backend framework to use ('onnx' or 'torch').
    :param modalities: A tuple specifying the types of model components to fetch.
    :param dtype: PyTorch model precision. Options: "float32", "float16", "bfloat16" (default).
                  Only applies to torch backend. bfloat16 provides ~2x speedup with minimal accuracy loss.
                  Works on CPUs with AVX-512 BF16 support and all modern CUDA GPUs.
    :param token: Optional API token for authenticated access to the model.
    :return: A tuple containing dictionaries for processors and models keyed by their respective modalities.
    """
    if backend == "onnx":
        return get_model_onnx(model_name, device=device, token=token, modalities=modalities)
    elif backend == "torch":
        return get_model_torch(model_name, device=device, token=token, modalities=modalities, dtype=dtype)
    else:
        raise ValueError(f"Unknown backend: {backend}")
