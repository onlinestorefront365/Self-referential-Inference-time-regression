"""The only module allowed to touch semi-private PyTorch APIs.

``torch.utils._pytree`` and ``torch._dynamo`` carry no stability guarantee. Every
use of them in the library goes through the wrappers below, so a PyTorch upgrade
that moves them needs a fix here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as _pytree

MIN_TORCH = (2, 6)


def _torch_version() -> tuple[int, int]:
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    return int(major), int(minor)


if _torch_version() < MIN_TORCH:
    raise ImportError(
        f"srwm requires torch>={MIN_TORCH[0]}.{MIN_TORCH[1]}, found {torch.__version__}"
    )


# --- pytrees -----------------------------------------------------------------


def register_pytree_dataclass(cls: type[Any]) -> None:
    """Register a dataclass so torch pytree utilities flatten it into its fields."""
    _pytree.register_dataclass(cls, serialized_type_name=f"{cls.__module__}.{cls.__qualname__}")


def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    return _pytree.tree_flatten(tree)


def tree_unflatten(leaves: list[Any], spec: Any) -> Any:
    return _pytree.tree_unflatten(leaves, spec)


def tree_map(fn: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    return _pytree.tree_map(fn, tree, *rest)


# --- dynamo --------------------------------------------------------------------


class CompileCounter:
    """A torch.compile backend that counts compiled frames and runs them eagerly.

    ``frame_count`` is 1 after the first compile and goes up by one per recompile.
    """

    def __init__(self) -> None:
        from torch._dynamo.testing import CompileCounter as _Counter

        self._counter = _Counter()

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Any:
        return self._counter(gm, example_inputs)

    @property
    def frame_count(self) -> int:
        return int(self._counter.frame_count)


def dynamo_reset() -> None:
    """Drop all cached compiled frames so a count starts from zero."""
    torch._dynamo.reset()
