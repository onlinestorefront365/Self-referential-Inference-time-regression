"""The public contracts: SequenceLayer, InputAdapter, OutputHead.

A SequenceLayer only ever sees ``(batch, d_model)`` vectors. Turning tokens,
vectors or feature tensors into those vectors is an InputAdapter's job, and
turning them back into predictions is an OutputHead's, so no backend has to
know about input types.

Layers are pure: ``step`` and ``scan`` return a new state and never modify the
one passed in. Mutation belongs to the Session wrapper (M4).
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch
from torch import Tensor, nn

from srwm import _compat
from srwm.core.state import State

S = TypeVar("S", bound=State)


@runtime_checkable
class SequenceLayer(Protocol[S]):
    """A recurrent layer with explicit state.

    Shapes: ``x`` is ``(B, d_model)``, ``xs`` is ``(B, T, d_model)``, ``mask`` is
    ``(B,)`` bool. ``scan(xs, s)`` must equal ``step`` applied ``T`` times.
    """

    d_model: int

    def init_state(self, batch: int) -> S: ...

    def step(self, x: Tensor, state: S) -> tuple[Tensor, S]: ...

    def scan(self, xs: Tensor, state: S) -> tuple[Tensor, S]: ...

    def reset(self, state: S, mask: Tensor) -> S: ...


@runtime_checkable
class InputAdapter(Protocol):
    """Maps raw inputs of any type to ``(..., d_model)`` vectors."""

    d_model: int

    def __call__(self, x: Tensor) -> Tensor: ...


@runtime_checkable
class OutputHead(Protocol):
    """Maps ``(..., d_model)`` vectors to predictions."""

    d_model: int

    def __call__(self, h: Tensor) -> Tensor: ...


class SequenceLayerBase(nn.Module, Generic[S]):
    """Optional base class: subclasses implement ``init_state`` and ``_step``.

    It supplies ``step`` (which advances the step counter), a ``scan`` that loops
    over ``step`` and a ``reset`` built on ``torch.where``. None of them branch on
    tensor values, so all three compile with ``fullgraph=True``.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def init_state(self, batch: int) -> S:
        raise NotImplementedError

    def _step(self, x: Tensor, state: S) -> tuple[Tensor, S]:
        """Compute one step. Must not touch ``state.step``; ``step`` advances it."""
        raise NotImplementedError

    def step(self, x: Tensor, state: S) -> tuple[Tensor, S]:
        y, new = self._step(x, state)
        return y, new.replace(step=state.step + 1)

    def scan(self, xs: Tensor, state: S) -> tuple[Tensor, S]:
        # A Python loop: torch.compile unrolls it, so compile scan for a fixed T
        # (one chunk) and call it repeatedly for longer sequences.
        ys = []
        for t in range(xs.shape[1]):
            y, state = self.step(xs[:, t], state)
            ys.append(y)
        return torch.stack(ys, dim=1), state

    def reset(self, state: S, mask: Tensor) -> S:
        fresh = self.init_state(state.batch_size)

        def pick(new: Tensor, old: Tensor) -> Tensor:
            m = mask.reshape(mask.shape + (1,) * (old.dim() - 1))
            return torch.where(m, new.to(old.dtype), old)

        out: S = _compat.tree_map(pick, fresh, state)
        return out
