# Self-referential recurrent neural network

`srwm` is a PyTorch library of **stateful sequence layers that learn at inference
time**. A layer of this kind updates its own internal memory (for SRWM, its own
weight matrix) while it reads a sequence, so it can fit a new function from
examples in context without any gradient steps.

The headline model is the **Self-Referential Weight Matrix (SRWM)**
(Irie, Schlag, Csordás & Schmidhuber, 2022). Every layer — SRWM, DeltaNet, LSTM
and any third-party layer — sits behind one contract, `SequenceLayer`. Callers
program against the contract, so the model behind it can change without breaking
them.

> **Status: M1 complete.** The contract, the conformance suite and an LSTM
> reference layer are done and green in CI. SRWM and DeltaNet are M2.
> See [`DEVELOPMENT.log`](DEVELOPMENT.log) for the full history.

---

## Contents

1. [Why this exists](#why-this-exists)
2. [Install](#install)
3. [Quickstart](#quickstart)
4. [Architecture](#architecture)
5. [The `SequenceLayer` contract](#the-sequencelayer-contract)
6. [State rules](#state-rules)
7. [Writing a layer](#writing-a-layer)
8. [Input adapters and output heads](#input-adapters-and-output-heads)
9. [Conformance suite](#conformance-suite)
10. [torch.compile guidance](#torchcompile-guidance)
11. [PyTorch internals policy](#pytorch-internals-policy)
12. [Project layout](#project-layout)
13. [Development workflow](#development-workflow)
14. [Branches](#branches)
15. [Roadmap](#roadmap)
16. [Design decisions](#design-decisions)
17. [Known limitations](#known-limitations)
18. [References](#references)

---

## Why this exists

Ordinary networks are fixed once trained. A layer that **learns at inference
time** keeps a memory that it rewrites as it reads, so each new sequence can
describe a new task.

SRWM's update is one step of the delta rule:

```
[y_t, k_t, q_t, β_t] = W_{t-1} φ(x_t)
v_t  = W_{t-1} φ(q_t)            # what the model wants stored
v̄_t = W_{t-1} φ(k_t)            # what is currently stored under k_t
W_t  = W_{t-1} + σ(β_t) (v_t − v̄_t) ⊗ φ(k_t)
```

That is one SGD step on `‖W φ(k) − v‖²`, so in effect the layer runs online
least-squares regression on its own weights. Through meta-training it learns
what to regress on and how fast. This is the "inference-time regression" in
the repository name.

The library's audience is ML researchers and systems engineers building stateful
agents, adaptive memory buffers and in-context learning systems. Data starts
as synthetic task distributions (for meta-training) and scales to real event
streams, time series and token sequences.

## Install

Requires Python ≥ 3.10 and PyTorch ≥ 2.6.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
```

## Quickstart

```python
import torch
from srwm.layers import LSTMLayer

layer = LSTMLayer(d_model=16)
state = layer.init_state(batch=4)

# streaming: one token at a time
x = torch.randn(4, 16)
y, state = layer.step(x, state)

# whole sequence
xs = torch.randn(4, 32, 16)
ys, state = layer.scan(xs, state)

# restart streams 0 and 2, keep 1 and 3
state = layer.reset(state, torch.tensor([True, False, True, False]))

# persist and resume
torch.save(state.state_dict(), "state.pt")
state = type(state).from_state_dict(torch.load("state.pt", weights_only=True))

# compiled streaming step: one compile, no recompiles
step = torch.compile(layer.step, fullgraph=True)
```

## Architecture

The library is built in four layers. Each one depends only on the ones below it.

| Layer | Contents | Status |
|---|---|---|
| 4. Backends | PyTorch (reference); JAX later, behind the same protocols | PyTorch done |
| 3. Training plumbing | `TaskSampler`, `meta_train`, versioned checkpoints | M3 |
| 2. Inference-time regressor | `Regressor`, `Session` (`observe`, `predict`, save/load, `fork`) | M4 |
| 1. Core contract | `SequenceLayer`, `State`, `InputAdapter`, `OutputHead`, conformance suite | **M1 done** |

**Layers are pure functions of `(input, state)`.** All mutation lives in the
`Session` wrapper (M4), which holds the current state and replaces it on every
call. This keeps the core compilable and makes a JAX port mechanical.

## The `SequenceLayer` contract

```python
class SequenceLayer(Protocol[S]):
    d_model: int

    def init_state(self, batch: int) -> S: ...
    def step(self, x, state: S) -> tuple[Tensor, S]: ...  # (B, d) -> (B, d)
    def scan(self, xs, state: S) -> tuple[Tensor, S]: ...  # (B, T, d) -> (B, T, d)
    def reset(self, state: S, mask) -> S: ...  # mask: (B,) bool
```

| Method | Contract |
|---|---|
| `init_state(batch)` | A fresh state for `batch` independent streams, on the layer's device and in its dtype. Its tensor strides must match the states `step` produces later. |
| `step(x, state)` | One timestep. Returns the output and a **new** state; must not mutate `state`. Advances `state.step` by 1. |
| `scan(xs, state)` | `T` timesteps. Must equal `step` applied `T` times (within 1e-6 in float64). |
| `reset(state, mask)` | Streams where `mask` is true go back to `init_state`; the others are left bit-identical. No control flow that depends on data. |

`SequenceLayer` is `runtime_checkable`, so `isinstance(layer, SequenceLayer)`
works.

## State rules

State is a `State` subclass decorated with `@state_dataclass`:

```python
@state_dataclass
class LSTMState(State):
    h: Tensor  # (B, d_model)
    c: Tensor  # (B, d_model)
```

- **Frozen dataclass.** To change a field, use `state.replace(h=...)`; assignment raises.
- **Only tensor fields.** `@state_dataclass` rejects any other annotation at
  class-definition time. A Python `int` or `bool` in the state becomes a
  torch.compile guard, and a counter stored that way recompiles on every step.
- **`step` is always present.** It is an int64 tensor of shape `(B,)` that counts
  steps per stream, so resetting one stream restarts only its count.
- **`SCHEMA_VERSION` is a class constant.** Override it as `SCHEMA_VERSION = 2`
  with no annotation; an annotation would make it a field, and is rejected.
- **Registered as a pytree**, so `tree_map`, `tree_flatten` and torch.compile
  all see through it.
- **Serialisation:** `state.state_dict()` returns
  `{"type", "schema_version", "tensors"}`, which loads with
  `torch.load(..., weights_only=True)`. `State.from_state_dict()` rejects a
  mismatched type, schema version or field set.

## Writing a layer

Subclass `SequenceLayerBase` and implement two methods:

```python
from srwm import SequenceLayerBase, State, state_dataclass


@state_dataclass
class MyState(State):
    h: Tensor


class MyLayer(SequenceLayerBase[MyState]):
    def init_state(self, batch): ...  # use .repeat(), not .expand()
    def _step(self, x, state): ...  # step, scan and reset come from the base class
```

`SequenceLayerBase` supplies:
- `step`, which calls `_step` and advances `state.step`;
- `scan`, a loop over `step`;
- `reset`, which builds a fresh `init_state` and combines it with the old state through `torch.where`.

Then run the conformance suite on it (next section). A layer is done when it passes.

## Input adapters and output heads

A `SequenceLayer` only ever sees `(B, d_model)` vectors. Raw inputs are
converted outside the layer, so no backend needs to know about input types:

| Class | Input → output |
|---|---|
| `TokenAdapter(vocab, d)` | int ids `(...,)` → `(..., d)` |
| `VectorAdapter(d_in, d)` | `(..., d_in)` → `(..., d)` |
| `TensorAdapter(shape, d)` | `(..., *shape)` → `(..., d)` (flattened) |
| `RegressionHead(d, d_out)` | `(..., d)` → `(..., d_out)` |
| `ClassificationHead(d, n)` | `(..., d)` → `(..., n)` logits |

These live in `srwm.adapters`. The full set arrives in M4.

## Conformance suite

`srwm.testing` is public API, so third-party layers can certify themselves:

```python
import pytest
from srwm.testing import CHECKS, LayerSpec

SPEC = LayerSpec(factory=lambda: MyLayer(d_model=4), d_model=4)


@pytest.mark.parametrize("check", list(CHECKS))
def test_conformance(check):
    CHECKS[check](SPEC)
```

Alternatively, `run_conformance(SPEC)` returns a list of `CheckResult`s and never raises.

| Check | What it asserts |
|---|---|
| `state_is_tensor_pytree` | Only tensor fields; pytree round-trip; `step` is `(B,)` int64 starting at 0 |
| `shapes_and_dtypes` | float32 and float64; output and state shapes, dtypes and device stay stable across steps |
| `scan_step_equivalence` | `scan` equals a `step` loop within 1e-6 (float64) |
| `reset` | Masked streams equal `init_state`; the others are bit-identical |
| `batch_independence` | Changing stream 0 changes only stream 0 |
| `purity` | `step`, `scan` and `reset` don't mutate their input state or the layer's parameters |
| `serialization` | Round-trips through `torch.load(weights_only=True)`; a resumed stream equals an uninterrupted one; a wrong schema version is rejected |
| `determinism` | The same seed gives bit-identical results |
| `gradients` | `gradcheck` on the inputs and every parameter (float64) |
| `compile_step` | `fullgraph=True`, exactly one compile across 100 streaming steps |
| `compile_scan` | `fullgraph=True` for a fixed chunk length; at most one recompile across 3 chunks |
| `compiled_matches_eager` | Inductor output equals eager within 1e-5 |

`tests/test_conformance_catches_bugs.py` checks that the suite **fails**
deliberately broken layers:
- one that mutates its input state in place,
- one with a stride-0 initial state,
- one that leaks information between streams,
- one that branches on a tensor value.

## torch.compile guidance

- **Streaming:** compile `step` with `fullgraph=True`. Its guards are stable,
  so it compiles once. For per-token latency on GPU, use
  `mode="reduce-overhead"` (CUDA graphs); compiling alone does not remove
  per-step dispatch overhead.
- **Long sequences:** don't compile `scan` over a variable `T`. Dynamo unrolls
  the Python loop, so compile time grows with `T` and every new length
  recompiles. Instead, compile `scan` for a fixed chunk length (e.g. 16) and
  call it in a loop. At most the final partial chunk triggers a recompile.
- **Initial state layout:** build initial state with `.repeat()` or
  `.clone()`, not `.expand()`. A stride-0 tensor at step 0 that turns
  contiguous at step 1 forces a recompile. The conformance suite catches this
  and names the cause.

## PyTorch internals policy

`torch.utils._pytree` and `torch._dynamo` are semi-private and can change
between releases. They are imported **only** in `srwm/_compat.py`. That module
also fails at import time with a clear message on torch < 2.6. A PyTorch upgrade
that moves those APIs needs a fix in that one file.

## Project layout

```
srwm/
  __init__.py            public API re-exports
  _compat.py             the only importer of torch._dynamo / torch.utils._pytree
  adapters.py            InputAdapter / OutputHead implementations
  core/
    state.py             State, @state_dataclass
    protocols.py         SequenceLayer, InputAdapter, OutputHead, SequenceLayerBase
  layers/
    lstm.py              LSTMLayer, LSTMState
  testing/
    conformance.py       LayerSpec, CHECKS, run_conformance
tests/
  test_lstm_conformance.py
  test_conformance_catches_bugs.py
  test_state.py
.github/workflows/ci.yml
DEVELOPMENT.log          chronological development log
```

## Development workflow

Run the full gate before every push:

```bash
ruff check . && ruff format --check . && mypy && pytest -q -p no:logging
```

`ruff format` also formats Python code blocks inside Markdown, so run it after
editing this README. Missing that broke CI run #1.

**CI** (`.github/workflows/ci.yml`) runs on every push and pull request:
- Ubuntu with Python 3.10 and 3.12, PyTorch CPU wheels;
- lint, format check, `mypy --strict` on `srwm/core`, pytest.

## Branches

| Branch | Role |
|---|---|
| `master` | Stable baseline. Holds only finished, CI-green milestones (currently M1). |
| `tmp/m2` | Temporary working branch for M2 (SRWM and DeltaNet). Merges into `master` once green. |
| `main` | The repository's original default branch (initial commit only). |
| `ccr-d347cbb9-47pcb6` | The session branch where M1 was developed. |

## Roadmap

| # | Deliverable | Acceptance criterion |
|---|---|---|
| M1 ✅ | Contract, conformance suite, LSTM | LSTM passes the suite in CI |
| M2 | SRWM and DeltaNet layers; compile check with gradients on | Both pass the suite; delta-rule and gradcheck tests pass |
| M3 | `TaskSampler` (linear and 2-layer MLP), `meta_train`, checkpoints | SRWM error within 2× of OLS at n ≥ 2d |
| M4 | `Regressor` / `Session` API (`observe`, `predict`, save/load, `fork`); full set of adapters | Streaming example runs; a saved-and-resumed session gives identical outputs |
| M5 | Packaging, docs, v0.1.0 | `pip install` works; CI green |
| M6 | JAX backend | Parity test against PyTorch passes |

## Design decisions

| Decision | Why |
|---|---|
| A contract first, SRWM second | SRWM is a 2022 research model. Tying the public API to its internals would break every caller on each model change. |
| Python library before a network service | Building a service before the core is stable creates operational debt. `Session` can be wrapped in FastAPI or gRPC later. |
| Pure layers, mutable `Session` | Pure layers compile with `fullgraph=True` and port to JAX mechanically; mutation is confined to one small wrapper. |
| `state.step` is `(B,)`, not 0-d | Per-stream reset needs a per-stream count. It is still a tensor, so it causes no recompiles. |
| Only tensors in state | Python scalars become compile guards and cause recompiles. |
| Compile `scan` per chunk, not over the whole sequence | A loop over variable `T` unrolls, recompiles per length and has unbounded compile time. |
| PyTorch only until M6 | Get the reference right first. A functional core costs PyTorch nothing and keeps the JAX port cheap. |
| Linear and MLP task samplers | Linear tasks verify convergence to OLS/ridge. MLP tasks show that non-linear in-context learning works, which justifies the library over plain linear attention. |

## Known limitations

- **CPU only in CI.** The CUDA-graph streaming speedup is untested.
- **The compile checks run without gradients.** A check with gradients on is planned
  for M2, before M3 meta-training.
- **`main` and `master` both exist.** Pick one as the default branch in
  the GitHub settings.

## References

- Irie, Schlag, Csordás, Schmidhuber. *A Modern Self-Referential Weight Matrix
  That Learns to Modify Itself.* ICML 2022.
- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight
  Programmers.* ICML 2021 (DeltaNet).
- Schmidhuber. *A self-referential weight matrix.* ICANN 1993.
