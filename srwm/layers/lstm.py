"""LSTM behind the SequenceLayer contract: the simple reference implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from srwm.core.protocols import SequenceLayerBase
from srwm.core.state import State, state_dataclass


@state_dataclass
class LSTMState(State):
    h: Tensor  # (B, d_model)
    c: Tensor  # (B, d_model)


class LSTMLayer(SequenceLayerBase[LSTMState]):
    """Single LSTM cell with a learned initial state; output is the hidden state."""

    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.cell = nn.LSTMCell(d_model, d_model)
        self.h0 = nn.Parameter(torch.zeros(d_model))
        self.c0 = nn.Parameter(torch.zeros(d_model))

    def init_state(self, batch: int) -> LSTMState:
        return LSTMState(
            step=torch.zeros(batch, dtype=torch.int64, device=self.h0.device),
            # repeat, not expand: a stride-0 initial state would make torch.compile
            # recompile once the state becomes contiguous after the first step.
            h=self.h0.repeat(batch, 1),
            c=self.c0.repeat(batch, 1),
        )

    def _step(self, x: Tensor, state: LSTMState) -> tuple[Tensor, LSTMState]:
        h, c = self.cell(x, (state.h, state.c))
        return h, state.replace(h=h, c=c)
