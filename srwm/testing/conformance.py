"""Conformance suite every SequenceLayer must pass, including third-party ones.

Use it from pytest::

    spec = LayerSpec(factory=lambda: MyLayer(d_model=4), d_model=4)

    @pytest.mark.parametrize("check", CHECKS, ids=list(CHECKS))
    def test_conformance(check):
        CHECKS[check](spec)

or call :func:`run_conformance` for a pass/fail report.
"""

from __future__ import annotations

import dataclasses
import io
import traceback
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from srwm import _compat
from srwm.core.protocols import SequenceLayer
from srwm.core.state import State


@dataclasses.dataclass
class LayerSpec:
    """How to build the layer under test.

    ``factory`` must return a fresh ``nn.Module`` implementing SequenceLayer,
    with float32 parameters. The suite seeds the RNG before each call and casts
    with ``.to(dtype)`` itself. Keep ``d_model`` small: gradcheck runs on it.
    """

    factory: Callable[[], nn.Module]
    d_model: int
    batch: int = 3
    seq_len: int = 6
    seed: int = 0


class ConformanceError(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConformanceError(msg)


def _build(spec: LayerSpec, dtype: torch.dtype = torch.float32) -> Any:
    torch.manual_seed(spec.seed)
    layer = spec.factory().to(dtype)
    _check(isinstance(layer, SequenceLayer), f"{type(layer).__name__} is not a SequenceLayer")
    _check(layer.d_model == spec.d_model, "layer.d_model != spec.d_model")
    return layer


def _inputs(spec: LayerSpec, dtype: torch.dtype, seq_len: int | None = None) -> Tensor:
    g = torch.Generator().manual_seed(spec.seed + 1)
    t = spec.seq_len if seq_len is None else seq_len
    return torch.randn(spec.batch, t, spec.d_model, generator=g, dtype=dtype)


def _states_equal(a: State, b: State, atol: float = 0.0) -> bool:
    la, sa = _compat.tree_flatten(a)
    lb, sb = _compat.tree_flatten(b)
    if sa != sb:
        return False
    if atol == 0.0:
        return all(torch.equal(x, y) for x, y in zip(la, lb, strict=True))
    return all(torch.allclose(x, y, atol=atol, rtol=0) for x, y in zip(la, lb, strict=True))


# --- checks ----------------------------------------------------------------------


def check_state_is_tensor_pytree(spec: LayerSpec) -> None:
    layer = _build(spec)
    state = layer.init_state(spec.batch)
    _check(isinstance(state, State), "init_state must return a State")
    leaves, treespec = _compat.tree_flatten(state)
    _check(len(leaves) == len(dataclasses.fields(state)), "state must flatten into its fields")
    _check(all(isinstance(x, Tensor) for x in leaves), "every state field must be a Tensor")
    _check(
        _states_equal(_compat.tree_unflatten(leaves, treespec), state),
        "pytree round-trip must reproduce the state",
    )
    _check(state.step.shape == (spec.batch,), "state.step must have shape (batch,)")
    _check(state.step.dtype == torch.int64, "state.step must be int64")
    _check(bool((state.step == 0).all()), "init_state must start step at 0")


def check_shapes_and_dtypes(spec: LayerSpec) -> None:
    for dtype in (torch.float32, torch.float64):
        layer = _build(spec, dtype)
        xs = _inputs(spec, dtype)
        s0 = layer.init_state(spec.batch)
        y, s1 = layer.step(xs[:, 0], s0)
        _check(y.shape == (spec.batch, spec.d_model), f"step output shape {tuple(y.shape)}")
        _check(y.dtype == dtype, f"step output dtype {y.dtype}, expected {dtype}")
        _check(y.device == xs.device, "output must be on the input's device")
        _, spec0 = _compat.tree_flatten(s0)
        _, spec1 = _compat.tree_flatten(s1)
        _check(spec0 == spec1, "step must not change the state's structure")
        for name, a in s0.tensors().items():
            b = s1.tensors()[name]
            _check(a.shape == b.shape, f"state.{name} changed shape across a step")
            _check(a.dtype == b.dtype, f"state.{name} changed dtype across a step")
            _check(b.device == xs.device, f"state.{name} is not on the input's device")
        _check(bool((s1.step == 1).all()), "step must advance state.step by 1")
        ys, sT = layer.scan(xs, s0)
        _check(ys.shape == xs.shape, f"scan output shape {tuple(ys.shape)}")
        _check(bool((sT.step == spec.seq_len).all()), "scan must advance state.step by T")


def check_scan_step_equivalence(spec: LayerSpec) -> None:
    layer = _build(spec, torch.float64)
    xs = _inputs(spec, torch.float64)
    s = layer.init_state(spec.batch)
    ys_scan, s_scan = layer.scan(xs, s)
    ys = []
    for t in range(xs.shape[1]):
        y, s = layer.step(xs[:, t], s)
        ys.append(y)
    _check(
        torch.allclose(torch.stack(ys, 1), ys_scan, atol=1e-6, rtol=0),
        "scan outputs differ from a step loop",
    )
    _check(_states_equal(s, s_scan, atol=1e-6), "scan final state differs from a step loop")


def check_reset(spec: LayerSpec) -> None:
    _check(spec.batch >= 2, "reset check needs batch >= 2")
    layer = _build(spec)
    _, s = layer.scan(_inputs(spec, torch.float32), layer.init_state(spec.batch))
    mask = torch.zeros(spec.batch, dtype=torch.bool)
    mask[::2] = True
    r = layer.reset(s, mask)
    fresh = layer.init_state(spec.batch)
    for name, after in r.tensors().items():
        before, init = s.tensors()[name], fresh.tensors()[name]
        _check(torch.equal(after[mask], init[mask]), f"reset streams: {name} != init_state")
        _check(torch.equal(after[~mask], before[~mask]), f"kept streams: {name} changed")


def check_batch_independence(spec: LayerSpec) -> None:
    layer = _build(spec, torch.float64)
    xs = _inputs(spec, torch.float64)
    s = layer.init_state(spec.batch)
    ys, _ = layer.scan(xs, s)
    xs2 = xs.clone()
    xs2[0] += 1.0
    ys2, _ = layer.scan(xs2, s)
    _check(not torch.allclose(ys[0], ys2[0]), "changing stream 0 must change its output")
    _check(
        torch.allclose(ys[1:], ys2[1:], atol=1e-12, rtol=0),
        "changing stream 0 changed another stream's output",
    )


def check_purity(spec: LayerSpec) -> None:
    layer = _build(spec)
    xs = _inputs(spec, torch.float32)
    s = layer.init_state(spec.batch)
    _, s = layer.step(xs[:, 0], s)
    snapshot = _compat.tree_map(lambda t: t.detach().clone(), s)
    params = {k: v.detach().clone() for k, v in layer.state_dict().items()}
    layer.step(xs[:, 1], s)
    layer.scan(xs, s)
    layer.reset(s, torch.ones(spec.batch, dtype=torch.bool))
    _check(_states_equal(s, snapshot), "step/scan/reset mutated the input state")
    for k, v in layer.state_dict().items():
        _check(torch.equal(v, params[k]), f"a forward pass modified layer buffer/param {k}")


def check_serialization(spec: LayerSpec) -> None:
    layer = _build(spec)
    xs = _inputs(spec, torch.float32)
    k = spec.seq_len // 2
    with torch.no_grad():
        _, mid = layer.scan(xs[:, :k], layer.init_state(spec.batch))
        ys_full, end_full = layer.scan(xs[:, k:], mid)
        buf = io.BytesIO()
        torch.save(mid.state_dict(), buf)
        buf.seek(0)
        loaded = type(mid).from_state_dict(torch.load(buf, weights_only=True))
        _check(_states_equal(loaded, mid), "state_dict round-trip changed the state")
        ys_resumed, end_resumed = layer.scan(xs[:, k:], loaded)
    _check(torch.equal(ys_full, ys_resumed), "a resumed stream differs from an uninterrupted one")
    _check(_states_equal(end_full, end_resumed), "resumed final state differs")
    bad = mid.state_dict()
    bad["schema_version"] = type(mid).SCHEMA_VERSION + 1
    try:
        type(mid).from_state_dict(bad)
    except ValueError:
        pass
    else:
        raise ConformanceError("from_state_dict must reject a different schema version")


def check_determinism(spec: LayerSpec) -> None:
    outs = []
    for _ in range(2):
        layer = _build(spec)
        ys, s = layer.scan(_inputs(spec, torch.float32), layer.init_state(spec.batch))
        outs.append((ys, s))
    _check(torch.equal(outs[0][0], outs[1][0]), "same seed gave different outputs")
    _check(_states_equal(outs[0][1], outs[1][1]), "same seed gave different states")


class _ScanFromInit(nn.Module):
    """Wraps a layer so torch.func.functional_call can swap its parameters."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, xs: Tensor) -> Tensor:
        ys, _ = self.layer.scan(xs, self.layer.init_state(xs.shape[0]))
        return ys


def check_gradients(spec: LayerSpec) -> None:
    layer = _build(spec, torch.float64)
    wrapper = _ScanFromInit(layer)
    named = [(n, p) for n, p in wrapper.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    params = tuple(p.detach().clone().requires_grad_() for _, p in named)
    xs = _inputs(spec, torch.float64, seq_len=3)[:2].clone().requires_grad_()

    def f(xs: Tensor, *ps: Tensor) -> Tensor:
        out: Tensor = torch.func.functional_call(wrapper, dict(zip(names, ps, strict=True)), (xs,))
        return out

    _check(
        torch.autograd.gradcheck(f, (xs, *params), atol=1e-6, rtol=1e-4),
        "gradcheck failed",
    )


def check_compile_step(spec: LayerSpec, n_steps: int = 100) -> None:
    layer = _build(spec)
    xs = _inputs(spec, torch.float32, seq_len=n_steps)
    _compat.dynamo_reset()
    counter = _compat.CompileCounter()
    step = torch.compile(layer.step, backend=counter, fullgraph=True)
    with torch.no_grad():
        s = layer.init_state(spec.batch)
        for t in range(n_steps):
            _, s = step(xs[:, t], s)
    _check(
        counter.frame_count == 1,
        f"step compiled {counter.frame_count} times, expected 1. A common cause is init_state "
        "returning tensors whose strides differ from later states (e.g. an un-cloned expand).",
    )
    _check(bool((s.step == n_steps).all()), "compiled step did not advance state.step")


def check_compile_step_grad(spec: LayerSpec, n_steps: int = 10) -> None:
    """Training path: step compiled with grad on, then backprop through time."""
    layer = _build(spec)
    xs = _inputs(spec, torch.float32, seq_len=n_steps)
    params = dict(layer.named_parameters())

    def run(step_fn: Callable[[Tensor, Any], tuple[Tensor, Any]]) -> dict[str, Tensor | None]:
        layer.zero_grad(set_to_none=True)
        s = layer.init_state(spec.batch)
        loss = xs.new_zeros(())
        for t in range(n_steps):
            y, s = step_fn(xs[:, t], s)
            loss = loss + y.square().mean()
        loss.backward()
        return {n: None if p.grad is None else p.grad.clone() for n, p in params.items()}

    eager = run(layer.step)
    _compat.dynamo_reset()
    counter = _compat.CompileCounter()
    compiled = run(torch.compile(layer.step, backend=counter, fullgraph=True))
    _check(
        counter.frame_count == 1,
        f"step compiled {counter.frame_count} times with grad enabled, expected 1. A common "
        "cause is an init_state that does not require grad while later states do "
        "(e.g. a constant initial memory instead of one derived from a parameter).",
    )
    for name in params:
        g_eager, g_comp = eager[name], compiled[name]
        _check(g_eager is not None, f"parameter {name} received no gradient")
        _check(g_comp is not None, f"parameter {name} received no gradient when compiled")
        assert g_eager is not None and g_comp is not None
        _check(
            torch.allclose(g_eager, g_comp, atol=1e-5, rtol=1e-4),
            f"compiled gradient for {name} differs from eager",
        )


def check_compile_scan(spec: LayerSpec, n_chunks: int = 3) -> None:
    layer = _build(spec)
    xs = _inputs(spec, torch.float32, seq_len=spec.seq_len * n_chunks)
    _compat.dynamo_reset()
    counter = _compat.CompileCounter()
    scan = torch.compile(layer.scan, backend=counter, fullgraph=True)
    with torch.no_grad():
        s = layer.init_state(spec.batch)
        for i in range(n_chunks):
            _, s = scan(xs[:, i * spec.seq_len : (i + 1) * spec.seq_len], s)
    _check(
        counter.frame_count <= 2,
        f"scan compiled {counter.frame_count} times over {n_chunks} chunks; at most 1 recompile",
    )


def check_compiled_matches_eager(spec: LayerSpec) -> None:
    layer = _build(spec)
    xs = _inputs(spec, torch.float32)
    _compat.dynamo_reset()
    scan = torch.compile(layer.scan, backend="inductor", fullgraph=True)
    with torch.no_grad():
        s0 = layer.init_state(spec.batch)
        ys_eager, s_eager = layer.scan(xs, s0)
        ys_comp, s_comp = scan(xs, s0)
    _check(torch.allclose(ys_eager, ys_comp, atol=1e-5, rtol=0), "compiled outputs != eager")
    _check(_states_equal(s_eager, s_comp, atol=1e-5), "compiled final state != eager")


CHECKS: dict[str, Callable[[LayerSpec], None]] = {
    "state_is_tensor_pytree": check_state_is_tensor_pytree,
    "shapes_and_dtypes": check_shapes_and_dtypes,
    "scan_step_equivalence": check_scan_step_equivalence,
    "reset": check_reset,
    "batch_independence": check_batch_independence,
    "purity": check_purity,
    "serialization": check_serialization,
    "determinism": check_determinism,
    "gradients": check_gradients,
    "compile_step": check_compile_step,
    "compile_step_grad": check_compile_step_grad,
    "compile_scan": check_compile_scan,
    "compiled_matches_eager": check_compiled_matches_eager,
}


@dataclasses.dataclass
class CheckResult:
    name: str
    passed: bool
    error: str | None = None


def run_conformance(spec: LayerSpec, *, skip: tuple[str, ...] = ()) -> list[CheckResult]:
    """Run every check not in ``skip``; never raises on a failing check."""
    results = []
    for name, check in CHECKS.items():
        if name in skip:
            continue
        try:
            check(spec)
        except Exception:
            results.append(CheckResult(name, False, traceback.format_exc()))
        else:
            results.append(CheckResult(name, True))
    return results
