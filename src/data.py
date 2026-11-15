"""
Synthetic intent datasets.

Each sample is a sequence drawn from a known generating function.
The model sees `context_len` steps and must extrapolate `extrapolate_steps` ahead.
Generating functions are the "intents" the model must identify and project forward.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable


def _linear(n: int, rng: np.random.RandomState) -> np.ndarray:
    a = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    b = rng.uniform(-2.0, 2.0)
    return (a * np.arange(n) / n + b).astype(np.float32)


def _quadratic(n: int, rng: np.random.RandomState) -> np.ndarray:
    a = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    b = rng.uniform(-1.0, 1.0)
    t = np.arange(n) / n
    vals = a * t ** 2 + b * t
    return vals.astype(np.float32)


def _exponential(n: int, rng: np.random.RandomState) -> np.ndarray:
    r = rng.uniform(1.5, 4.0)
    a = rng.uniform(0.1, 0.5)
    vals = a * r ** (np.arange(n) / n * 4)
    vals = (vals - vals.mean()) / (vals.std() + 1e-8)
    return vals.astype(np.float32)


def _sinusoidal(n: int, rng: np.random.RandomState) -> np.ndarray:
    freq = rng.uniform(1.0, 6.0)
    phase = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(0.5, 2.0)
    return (amp * np.sin(freq * np.arange(n) / n * 2 * np.pi + phase)).astype(np.float32)


def _fibonacci(n: int, rng: np.random.RandomState) -> np.ndarray:
    a, b = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)
    vals = [a, b]
    for _ in range(n - 2):
        vals.append(vals[-1] + vals[-2])
    vals = np.array(vals[:n], dtype=np.float32)
    return (vals - vals.mean()) / (vals.std() + 1e-8)


def _power(n: int, rng: np.random.RandomState) -> np.ndarray:
    exp = rng.uniform(1.5, 3.5) * rng.choice([-1, 1])
    t = np.arange(n) / n + 0.1
    vals = np.sign(exp) * np.abs(t) ** abs(exp)
    return (vals.astype(np.float32) - vals.mean()) / (vals.std() + 1e-8)


GENERATORS: dict[str, Callable] = {
    "linear": _linear,
    "quadratic": _quadratic,
    "exponential": _exponential,
    "sinusoidal": _sinusoidal,
    "fibonacci": _fibonacci,
    "power": _power,
}

INTENT_NAMES = list(GENERATORS.keys())


class IntentDataset(Dataset):
    """
    Each sample:
      context  (context_len, 1)         — visible input
      next_step (context_len, 1)         — shifted-by-1 next-step targets
      future   (extrapolate_steps, 1)   — far-future targets for intent loss
      label    int                       — generating function index
    """

    def __init__(
        self,
        n_samples: int,
        context_len: int,
        extrapolate_steps: int,
        noise_std: float = 0.02,
        seed: int = 42,
    ):
        self.context_len = context_len
        self.extrapolate_steps = extrapolate_steps
        total = context_len + extrapolate_steps + 1  # +1 for next-step shift

        rng = np.random.RandomState(seed)
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []

        names = INTENT_NAMES
        for i in range(n_samples):
            intent = names[i % len(names)]
            local_rng = np.random.RandomState(rng.randint(0, 2 ** 31))
            seq = GENERATORS[intent](total, local_rng)
            seq += (local_rng.randn(total) * noise_std).astype(np.float32)
            self.sequences.append(seq)
            self.labels.append(names.index(intent))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        context = torch.from_numpy(seq[: self.context_len]).unsqueeze(-1)
        next_step = torch.from_numpy(seq[1 : self.context_len + 1]).unsqueeze(-1)
        future = torch.from_numpy(
            seq[self.context_len : self.context_len + self.extrapolate_steps]
        ).unsqueeze(-1)
        return context, next_step, future, self.labels[idx]
