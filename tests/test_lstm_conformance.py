import pytest

from srwm.layers import LSTMLayer
from srwm.testing import CHECKS, LayerSpec, run_conformance

SPEC = LayerSpec(factory=lambda: LSTMLayer(d_model=4), d_model=4)


@pytest.mark.parametrize("check", list(CHECKS))
def test_lstm_conformance(check: str) -> None:
    CHECKS[check](SPEC)


def test_run_conformance_reports_all_checks() -> None:
    results = run_conformance(SPEC, skip=("compiled_matches_eager",))
    assert [r.name for r in results] == [c for c in CHECKS if c != "compiled_matches_eager"]
    assert all(r.passed for r in results), [r.error for r in results if not r.passed]
