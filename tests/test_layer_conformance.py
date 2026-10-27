from collections.abc import Callable

import pytest
from torch import nn

from srwm.layers import DeltaNetLayer, LSTMLayer, SRWMLayer
from srwm.testing import CHECKS, LayerSpec, run_conformance

FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "lstm": lambda: LSTMLayer(d_model=4),
    "deltanet": lambda: DeltaNetLayer(d_model=4, n_heads=2),
    "srwm": lambda: SRWMLayer(d_model=4, n_heads=2),
    "srwm_frozen": lambda: SRWMLayer(d_model=4, n_heads=2, self_modify=False),
}
SPECS = {name: LayerSpec(factory=f, d_model=4) for name, f in FACTORIES.items()}


@pytest.mark.parametrize("check", list(CHECKS))
@pytest.mark.parametrize("layer", list(SPECS))
def test_conformance(layer: str, check: str) -> None:
    CHECKS[check](SPECS[layer])


def test_run_conformance_reports_all_checks() -> None:
    skip = ("compiled_matches_eager",)
    results = run_conformance(SPECS["lstm"], skip=skip)
    assert [r.name for r in results] == [c for c in CHECKS if c not in skip]
    assert all(r.passed for r in results), [r.error for r in results if not r.passed]
