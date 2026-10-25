"""Minimal InputAdapter and OutputHead implementations (the full set lands in M4)."""

from __future__ import annotations

import math

from torch import Tensor, nn


class TokenAdapter(nn.Module):
    """Integer token ids ``(...,)`` -> ``(..., d_model)``."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.embed(x)


class VectorAdapter(nn.Module):
    """Continuous vectors ``(..., d_in)`` -> ``(..., d_model)``."""

    def __init__(self, d_in: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_in, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class TensorAdapter(nn.Module):
    """Feature tensors ``(..., *feature_shape)`` -> ``(..., d_model)`` by flattening."""

    def __init__(self, feature_shape: tuple[int, ...], d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.feature_shape = tuple(feature_shape)
        self.proj = nn.Linear(math.prod(self.feature_shape), d_model)

    def forward(self, x: Tensor) -> Tensor:
        n = len(self.feature_shape)
        if tuple(x.shape[-n:]) != self.feature_shape:
            raise ValueError(f"expected trailing shape {self.feature_shape}, got {tuple(x.shape)}")
        return self.proj(x.flatten(-n))


class RegressionHead(nn.Module):
    """``(..., d_model)`` -> ``(..., d_out)`` real-valued predictions."""

    def __init__(self, d_model: int, d_out: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)


class ClassificationHead(nn.Module):
    """``(..., d_model)`` -> ``(..., n_classes)`` logits."""

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)
