"""The suite is only worth something if it fails broken layers."""

import pytest
import torch
from torch import Tensor

from srwm.layers.deltanet import DeltaNetLayer, DeltaNetState
from srwm.layers.lstm import LSTMLayer, LSTMState
from srwm.testing import CHECKS, LayerSpec
from srwm.testing.conformance import ConformanceError


class InPlaceLSTM(LSTMLayer):
    """Mutates the incoming state instead of returning a new one."""

    def _step(self, x: Tensor, state: LSTMState) -> tuple[Tensor, LSTMState]:
        y, new = super()._step(x, state)
        state.h.copy_(new.h.detach())
        return y, new


class ExpandedInitLSTM(LSTMLayer):
    """Returns a stride-0 initial state, which recompiles after the first step."""

    def init_state(self, batch: int) -> LSTMState:
        s = super().init_state(batch)
        return s.replace(h=self.h0.expand(batch, -1), c=self.c0.expand(batch, -1))


class CrossTalkLSTM(LSTMLayer):
    """Leaks information between streams through a batch mean."""

    def _step(self, x: Tensor, state: LSTMState) -> tuple[Tensor, LSTMState]:
        return super()._step(x + x.mean(0, keepdim=True), state)


class BranchingLSTM(LSTMLayer):
    """Branches on a tensor value, which breaks fullgraph compilation."""

    def _step(self, x: Tensor, state: LSTMState) -> tuple[Tensor, LSTMState]:
        if x.sum() > 0:
            x = x * 2
        return super()._step(x, state)


class ConstantInitDeltaNet(DeltaNetLayer):
    """Starts from a constant zero memory: no grad at step 0, grad from step 1 on."""

    def init_state(self, batch: int) -> DeltaNetState:
        s = super().init_state(batch)
        return s.replace(W=torch.zeros_like(s.W).detach())


def test_grad_check_rejects_constant_initial_memory() -> None:
    spec = LayerSpec(factory=lambda: ConstantInitDeltaNet(d_model=4, n_heads=2), d_model=4)
    with pytest.raises(ConformanceError, match="grad enabled|no gradient"):
        CHECKS["compile_step_grad"](spec)
    # The same layer is fine for inference: the no-grad compile check passes.
    CHECKS["compile_step"](spec)


@pytest.mark.parametrize(
    ("layer_cls", "check"),
    [
        (InPlaceLSTM, "purity"),
        (ExpandedInitLSTM, "compile_step"),
        (CrossTalkLSTM, "batch_independence"),
        (BranchingLSTM, "compile_step"),
    ],
)
def test_suite_rejects(layer_cls: type[LSTMLayer], check: str) -> None:
    spec = LayerSpec(factory=lambda: layer_cls(d_model=4), d_model=4)
    with torch.no_grad(), pytest.raises(Exception) as info:  # noqa: B017
        CHECKS[check](spec)
    assert (
        isinstance(info.value, ConformanceError)
        or "graph break" in str(info.value).lower()
        or "Unsupported" in type(info.value).__name__
    )
