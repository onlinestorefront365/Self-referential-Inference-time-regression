import dataclasses

import pytest
import torch
from torch import Tensor

from srwm import State, _compat, state_dataclass
from srwm.layers import LSTMState


def _state() -> LSTMState:
    return LSTMState(step=torch.zeros(2, dtype=torch.int64), h=torch.ones(2, 3), c=torch.ones(2, 3))


def test_non_tensor_field_rejected() -> None:
    with pytest.raises(TypeError, match="must be Tensor"):

        @state_dataclass
        class Bad(State):
            counter: int


def test_schema_version_as_field_rejected() -> None:
    with pytest.raises(TypeError, match="without an annotation"):

        @state_dataclass
        class Bad(State):
            SCHEMA_VERSION: int = 2  # type: ignore[misc]


def test_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _state().h = torch.zeros(2, 3)  # type: ignore[misc]


def test_pytree_round_trip() -> None:
    s = _state()
    leaves, spec = _compat.tree_flatten(s)
    assert len(leaves) == 3
    assert isinstance(_compat.tree_unflatten(leaves, spec), LSTMState)


def test_state_dict_rejects_wrong_version_and_type() -> None:
    d = _state().state_dict()
    assert LSTMState.from_state_dict(d).h.shape == (2, 3)
    with pytest.raises(ValueError, match="schema version"):
        LSTMState.from_state_dict({**d, "schema_version": 99})
    with pytest.raises(ValueError, match="not"):
        LSTMState.from_state_dict({**d, "type": "other.State"})
    with pytest.raises(ValueError, match="fields"):
        LSTMState.from_state_dict({**d, "tensors": {"step": d["tensors"]["step"]}})


def test_annotation_checks_accept_tensor() -> None:
    @state_dataclass
    class Good(State):
        w: Tensor

    assert [f.name for f in dataclasses.fields(Good)] == ["step", "w"]


def test_schema_version_override() -> None:
    @state_dataclass
    class V2(State):
        SCHEMA_VERSION = 2
        w: Tensor

    assert V2.SCHEMA_VERSION == 2
    assert [f.name for f in dataclasses.fields(V2)] == ["step", "w"]
