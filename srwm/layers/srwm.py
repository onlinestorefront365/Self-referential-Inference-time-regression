"""Self-Referential Weight Matrix (Irie, Schlag, Csordás & Schmidhuber, 2022).

One matrix per head produces the output, the key, the query and its own
learning rates, then rewrites itself with the delta rule. There are no separate
slow weights: after training, only the initial matrix ``W0`` is fixed, and every
later matrix is written by the model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from srwm.core.protocols import SequenceLayerBase
from srwm.core.state import State, state_dataclass

N_LR = 4  # one learning rate each for the y, k, q and β sub-blocks


@state_dataclass
class SRWMState(State):
    W: Tensor  # (B, H, 3 * d_head + 4, d_head) self-modified weight matrix


class SRWMLayer(SequenceLayerBase[SRWMState]):
    """Multi-head SRWM followed by an output projection.

    Per head, with x the head's slice of the input and φ = softmax::

        [y, k, q, β] = W x
        v = W φ(q);  v̄ = W φ(k)
        W ← W + σ(β) ⊙ (v − v̄) ⊗ φ(k)

    ``σ(β)`` holds 4 learning rates, one per sub-block of rows (y, k, q, β).
    The output ``y`` is read from the matrix before the update.

    ``self_modify=False`` freezes ``W`` at ``W0``: the ablation that shows the
    self-update matters.
    """

    def __init__(self, d_model: int, n_heads: int = 1, *, self_modify: bool = True) -> None:
        super().__init__(d_model)
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.self_modify = self_modify
        rows = 3 * self.d_head + N_LR
        self.W0 = nn.Parameter(torch.randn(n_heads, rows, self.d_head) / math.sqrt(self.d_head))
        self.out = nn.Linear(d_model, d_model)

    @property
    def split_sizes(self) -> list[int]:
        return [self.d_head, self.d_head, self.d_head, N_LR]

    def init_state(self, batch: int) -> SRWMState:
        return SRWMState(
            step=torch.zeros(batch, dtype=torch.int64, device=self.W0.device),
            W=self.W0.unsqueeze(0).repeat(batch, 1, 1, 1),
        )

    def _step(self, x: Tensor, state: SRWMState) -> tuple[Tensor, SRWMState]:
        b, h, d = x.shape[0], self.n_heads, self.d_head
        W = state.W
        y, k, q, beta = torch.einsum("bhrd,bhd->bhr", W, x.reshape(b, h, d)).split(
            self.split_sizes, dim=-1
        )
        if self.self_modify:
            phi_k, phi_q = k.softmax(-1), q.softmax(-1)
            v = torch.einsum("bhrd,bhd->bhr", W, phi_q)
            v_bar = torch.einsum("bhrd,bhd->bhr", W, phi_k)
            lr = torch.sigmoid(beta)
            lr_rows = torch.cat(
                [lr[..., i : i + 1].expand(b, h, n) for i, n in enumerate(self.split_sizes)],
                dim=-1,
            )
            W = W + (lr_rows * (v - v_bar)).unsqueeze(-1) * phi_k.unsqueeze(-2)
        return self.out(y.reshape(b, h * d)), state.replace(W=W)
