from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Union, Mapping, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL.Image import Image

from uform.shared import read_config


def _is_on_gpu(model: nn.Module) -> bool:
    try:
        return next(model.parameters()).device.type == "cuda"
    except StopIteration:
        return False


@dataclass(eq=False)
class Attention(nn.Module):
    dim: int
    num_heads: int
    dropout_prob: float = 0

    def __post_init__(self):
        super().__init__()

        self.use_sdp = int(torch.__version__[0]) > 1

        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)

        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        query = self.reshape(self.query(x))
        key = self.reshape(self.key(x if context is None else context))
        value = self.reshape(self.value(x if context is None else context))

        if self.use_sdp:
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p=self.dropout_prob if self.training else 0,
                is_causal=is_causal,
            )
        else:
            attn = query @ key.transpose(-2, -1) * self.scale
            if attn_mask is not None:
                attn += attn_mask

            attn = attn.softmax(dim=-1)
            x = attn @ value

        return self.out(x.transpose(2, 1).flatten(2))

    def reshape(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(2, 1)


@dataclass(eq=False)
class MLP(nn.Module):
    dim: int
    dim_expand_factor: int = 4

    def __post_init__(self):
        super().__init__()

        self.hidden_layer = nn.Linear(self.dim, self.dim * self.dim_expand_factor)
        self.output_layer = nn.Linear(self.dim * self.dim_expand_factor, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.hidden_layer(x))
        return self.output_layer(x)


@dataclass(eq=False)
class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5
    inplace: bool = False

    def __post_init__(self):
        super().__init__()
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


@dataclass(eq=False)
class TextEncoderBlock(nn.Module):
    dim: int
    num_heads: int
    dropout_prob: float
    cross_attention: bool = False

    def __post_init__(self):
        super().__init__()

        self.norm_attn = nn.LayerNorm(self.dim, eps=1e-12)
        self.attention = Attention(self.dim, self.num_heads, self.dropout_prob)

        if self.cross_attention:
            self.norm_crossattn = nn.LayerNorm(self.dim, eps=1e-12)
            self.crossattn = Attention(self.dim, self.num_heads, self.dropout_prob)

        self.norm_mlp = nn.LayerNorm(self.dim, eps=1e-12)
        self.mlp = MLP(self.dim)

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.norm_attn(x + self.dropout(self.attention(x, attn_mask)))

        if self.cross_attention and context is not None:
            x = self.norm_crossattn(
                x + self.dropout(self.crossattn(x, context=context)),
            )

        return self.norm_mlp(x + self.dropout(self.mlp(x)))


@dataclass(eq=False)
class ImageEncoderBlock(nn.Module):
    dim: int
    num_heads: int

    def __post_init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(self.dim, eps=1e-6)
        self.attn = Attention(self.dim, self.num_heads)
        self.ls1 = LayerScale(self.dim)

        self.norm2 = nn.LayerNorm(self.dim, eps=1e-6)
        self.mlp = MLP(self.dim)
        self.ls2 = LayerScale(self.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


@dataclass(eq=False)
class TextEncoder(nn.Module):
    model_type: str
    dim: int
    context_dim: int
    vocab_size: int
    padding_idx: int
    num_layers: int
    num_heads: int
    embedding_dim: int
    multimodal_layers_ids: tuple
    head_one_neuron: bool
    pooling: str = "cls"
    max_position_embeddings: int = 77
    dropout_prob: float = 0

    def __post_init__(self):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            self.vocab_size,
            self.dim,
            padding_idx=self.padding_idx,
        )
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.dim)

        if self.model_type == "bert":
            self.register_buffer(
                "position_ids",
                torch.arange(self.max_position_embeddings).unsqueeze(0),
                persistent=False,
            )

        self.layer_norm = nn.LayerNorm(self.dim, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.blocks = nn.ModuleList(
            [
                TextEncoderBlock(
                    self.dim,
                    self.num_heads,
                    self.dropout_prob,
                    layer_id in self.multimodal_layers_ids,
                )
                for layer_id in range(self.num_layers)
            ],
        )

        self.embedding_projection = nn.Linear(self.dim, self.embedding_dim, bias=False)
        self.matching_head = nn.Linear(self.dim, 1 if self.head_one_neuron else 2)

        if self.context_dim != self.dim:
            self.context_projection = nn.Linear(self.context_dim, self.dim, bias=False)
        else:
            self.context_projection = nn.Identity()
        self.return_features = False

    def forward_features(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.embed_text(x)
        attn_mask = self.get_attention_mask(attn_mask, x.dtype)

        for block in self.blocks:
            if not block.cross_attention:
                x = block(x, attn_mask)

        return x

    def forward_embedding(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        return self.embedding_projection(self.pool_features(x, attn_mask))

    def pool_features(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        if self.pooling == "cls":
            return x[:, 0]

        attn_mask = attn_mask.unsqueeze(2).type_as(x)
        return (x * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)

    def get_attention_mask(self, attn_mask: Tensor, dtype: torch.dtype) -> Tensor:
        attn_mask = attn_mask.to(dtype)
        attn_mask = (1.0 - attn_mask) * torch.finfo(dtype).min
        return attn_mask.unsqueeze(1).expand(-1, attn_mask.shape[1], -1).unsqueeze(1)

    def get_position_ids(self, x: Tensor) -> Tensor:
        if self.model_type == "roberta":
            mask = x.ne(self.padding_idx).int()
            return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx

        return self.position_ids[:, : x.shape[1]]

    def embed_text(self, x: Tensor) -> Tensor:
        positional_embedding = self.position_embeddings(self.get_position_ids(x))
        x = self.word_embeddings(x) + positional_embedding
        return self.dropout(self.layer_norm(x))

    def forward(
        self,
        x: Union[Tensor, dict],
        attention_mask: Optional[Tensor] = None,
        return_features: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        if isinstance(x, dict):
            assert attention_mask is None, "If `x` is a dictionary, then `attention_mask` should be None"
            attention_mask = x["attention_mask"]
            x = x["input_ids"]
        elif attention_mask is None:
            # If no attention mask is provided - create one with all ones
            attention_mask = torch.ones_like(x)

        # If the model is on the GPU and the input matrices are not, shift them there
        if _is_on_gpu(self) and not x.is_cuda:
            x = x.cuda()
            attention_mask = attention_mask.cuda()

        features = self.forward_features(x, attention_mask)
        embeddings = self.forward_embedding(features, attention_mask)

        return_features = return_features if return_features is not None else self.return_features
        if return_features:
            return features, embeddings
        return embeddings

    def encode(
        self,
        x: Union[Tensor, dict],
        attention_mask: Optional[Tensor] = None,
        return_features: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        result = self.forward(x, attention_mask, return_features)
        if isinstance(result, tuple):
            return result[0].detach(), result[1].detach()
        else:
            return result.detach()

    @staticmethod
    def from_pretrained(config: Union[PathLike, str, object], model: Union[PathLike, str]) -> TextEncoder:
        """Load the image encoder from the given configuration and model path.

        :param config: the configuration dictionary or path to the JSON configuration file
        :param model: the model state dictionary or path to the `.pt` model file
        """
        config = read_config(config)
        if "text_encoder" in config:
            config = config["text_encoder"]

        # We must strip all the non-member attributes before initializing the classes.
        text_fields = TextEncoder.__dataclass_fields__
        config = {k: v for k, v in config.items() if k in text_fields}
        encoder = TextEncoder(**config)

        # Load from disk
        if isinstance(model, (PathLike, str)):
            state = torch.load(model)
        else:
            state = model
        if "text_encoder" in state:
            state = state["text_encoder"]
        encoder.load_state_dict(state)
        return encoder


@dataclass(eq=False)
class ImageEncoder(nn.Module):
    dim: int
    patch_size: int
    image_size: int
    num_layers: int
    num_heads: int
    embedding_dim: int
    pooling: str
    num_reg_tokens: int = 0

    def __post_init__(self):
        super().__init__()

        seq_len = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, self.dim, self.patch_size, self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        if self.num_reg_tokens > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, self.num_reg_tokens, self.dim))

        self.blocks = nn.Sequential(
            *[ImageEncoderBlock(self.dim, self.num_heads) for _ in range(self.num_layers)],
        )

        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.embedding_projection = nn.Linear(self.dim, self.embedding_dim, bias=False)
        self.return_features = False

    def forward_features(self, x: Union[Tensor, dict]) -> Tensor:
        x = self.patch_embed(x).flatten(start_dim=2).transpose(2, 1)
        x = x + self.pos_embed
        special_tokens = [self.cls_token.expand(x.shape[0], -1, -1)]

        if self.num_reg_tokens > 0:
            special_tokens.append(self.reg_token.expand(x.shape[0], -1, -1))

        x = torch.cat(special_tokens + [x], dim=1)
        x = self.blocks(x)
        return self.norm(x)

    def forward_embedding(self, x: Tensor) -> Tensor:
        if self.pooling == "cls":
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.embedding_projection(x)

    def forward(self, x: Union[Tensor, dict], return_features: Optional[bool] = None) -> Tensor:
        if isinstance(x, dict):
            x = x["images"]

        # If the model is on the GPU and the input matrices are not, shift them there
        if _is_on_gpu(self) and not x.is_cuda:
            x = x.cuda()

        # Convert input to match model dtype (handles bfloat16/float16 conversion)
        if hasattr(self, 'patch_embed') and hasattr(self.patch_embed, 'weight'):
            model_dtype = self.patch_embed.weight.dtype
            if x.dtype != model_dtype:
                x = x.to(dtype=model_dtype)

        features = self.forward_features(x)
        embeddings = self.forward_embedding(features)
        return_features = return_features if return_features is not None else self.return_features
        if return_features:
            return features, embeddings
        return embeddings

    def encode(self, x: Union[Tensor, dict], return_features: Optional[bool] = None) -> Tensor:
        result = self.forward(x, return_features)
        if isinstance(result, tuple):
            return result[0].detach(), result[1].detach()
        else:
            return result.detach()

    @staticmethod
    def from_pretrained(
        config: Union[PathLike, str, object],
        model: Union[PathLike, str, Mapping[str, Any]],
    ) -> ImageEncoder:
        """Load the image encoder from the given configuration and model path.

        :param config: the configuration dictionary or path to the JSON configuration file
        :param model: the model state dictionary or path to the `.pt` model file
        """
        config = read_config(config)
        if "image_encoder" in config:
            config = config["image_encoder"]

        # We must strip all the non-member attributes before initializing the classes.
        image_fields = ImageEncoder.__dataclass_fields__
        config = {k: v for k, v in config.items() if k in image_fields}
        encoder = ImageEncoder(**config)

        # Load from disk
        if isinstance(model, (PathLike, str)):
            state = torch.load(model)
        else:
            state = model
        if "image_encoder" in state:
            state = state["image_encoder"]
        encoder.load_state_dict(state)
        return encoder
