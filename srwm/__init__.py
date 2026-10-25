"""Stateful sequence layers that learn at inference time, behind one contract."""

from srwm.core import (
    InputAdapter,
    OutputHead,
    SequenceLayer,
    SequenceLayerBase,
    State,
    state_dataclass,
)

__version__ = "0.0.1"

__all__ = [
    "InputAdapter",
    "OutputHead",
    "SequenceLayer",
    "SequenceLayerBase",
    "State",
    "state_dataclass",
]
