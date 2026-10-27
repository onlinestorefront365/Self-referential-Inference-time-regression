"""DeltaNet (Schlag et al., 2021): a slow network programs a separate fast-weight matrix.

This is the non-self-referential baseline for SRWM: the projections that make
keys, values, queries and learning rates are fixed after training, and only the
fast-weight matrix changes at inference time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from srwm.core.protocols import SequenceLayerBase
from srwm.core.state import State, state_dataclass


@state_dataclass
class DeltaNetState(State):
    W: Tensor  # (B, H, d_head, d_head) fast weights, value x key


class DeltaNetLayer(SequenceLayerBase[DeltaNetState]):
    """Multi-head delta-rule fast-weight layer.

    Per head::

        q, k, v, b = slow(x)
        v̄ = W φ(k);  W ← W + σ(b) (v − v̄) ⊗ φ(k);  y = W φ(q)

    with φ = softmax. The initial fast weights are a learned parameter (zeros at
    init), so the state has the same requires_grad at every step.
    """

    def __init__(self, d_model: int, n_heads: int = 1) -> None:
        super().__init__(d_model)
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.slow = nn.Linear(d_model, n_heads * (3 * self.d_head + 1))
        self.W0 = nn.Parameter(torch.zeros(n_heads, self.d_head, self.d_head))
        self.out = nn.Linear(d_model, d_model)

    def init_state(self, batch: int) -> DeltaNetState:
        return DeltaNetState(
            step=torch.zeros(batch, dtype=torch.int64, device=self.W0.device),
            W=self.W0.unsqueeze(0).repeat(batch, 1, 1, 1),
        )

    def _step(self, x: Tensor, state: DeltaNetState) -> tuple[Tensor, DeltaNetState]:
        b, h, d = x.shape[0], self.n_heads, self.d_head
        q, k, v, beta = self.slow(x).reshape(b, h, 3 * d + 1).split([d, d, d, 1], dim=-1)
        phi_q, phi_k = q.softmax(-1), k.softmax(-1)
        W = state.W
        v_bar = torch.einsum("bhvk,bhk->bhv", W, phi_k)
        W = W + (torch.sigmoid(beta) * (v - v_bar)).unsqueeze(-1) * phi_k.unsqueeze(-2)
        y = torch.einsum("bhvk,bhk->bhv", W, phi_q)
        return self.out(y.reshape(b, h * d)), state.replace(W=W)
