"""Recurrent state: frozen, tensor-only, pytree-registered dataclasses.

Every field must be a tensor. A Python ``int`` or ``bool`` in the state becomes a
guard in torch.compile, so a step counter stored as ``int`` would recompile on
every step. Metadata that never changes per step (the schema version) lives on
the class, not the instance.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, ClassVar, TypeVar

import torch
from torch import Tensor

from srwm import _compat

S = TypeVar("S", bound="State")


@dataclasses.dataclass(frozen=True)
class State:
    """Base class for the state a :class:`SequenceLayer` carries between steps.

    ``step`` holds the number of steps each stream has taken, shape ``(batch,)``,
    int64. It is per stream so that resetting one stream restarts its count.

    Subclasses add tensor fields and must be decorated with :func:`state_dataclass`.
    Bump ``SCHEMA_VERSION`` whenever the fields or their meaning change.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    step: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.step.shape[0])

    def replace(self: S, **changes: Tensor) -> S:
        return dataclasses.replace(self, **changes)

    def tensors(self) -> dict[str, Tensor]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def state_dict(self) -> dict[str, Any]:
        """A ``torch.save``-able dict that loads with ``weights_only=True``."""
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "schema_version": type(self).SCHEMA_VERSION,
            "tensors": self.tensors(),
        }

    @classmethod
    def from_state_dict(cls: type[S], data: dict[str, Any]) -> S:
        expected = f"{cls.__module__}.{cls.__qualname__}"
        if data.get("type") != expected:
            raise ValueError(f"state dict is for {data.get('type')!r}, not {expected!r}")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"{expected} schema version {version} cannot be loaded by version "
                f"{cls.SCHEMA_VERSION}"
            )
        tensors = data["tensors"]
        names = {f.name for f in dataclasses.fields(cls)}
        if set(tensors) != names:
            raise ValueError(f"state dict fields {sorted(tensors)} != {sorted(names)}")
        return cls(**tensors)


def state_dataclass(cls: type[S]) -> type[S]:
    """Make ``cls`` a frozen dataclass, check it is tensor-only, register it as a pytree."""
    if not issubclass(cls, State):
        raise TypeError(f"{cls.__qualname__} must subclass State")
    if "SCHEMA_VERSION" in inspect.get_annotations(cls):
        raise TypeError(
            f"{cls.__qualname__}: set SCHEMA_VERSION = N without an annotation; "
            "an annotation would make it a dataclass field"
        )
    out = dataclasses.dataclass(frozen=True)(cls)
    for f in dataclasses.fields(out):
        if f.type not in (Tensor, "Tensor", torch.Tensor, "torch.Tensor"):
            raise TypeError(
                f"{cls.__qualname__}.{f.name} is annotated {f.type!r}; state fields must be Tensor"
            )
    _compat.register_pytree_dataclass(out)
    return out


_compat.register_pytree_dataclass(State)
