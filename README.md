# Self-referential recurrent neural network

A PyTorch library of stateful sequence layers that learn at inference time.
Every layer (SRWM, DeltaNet, LSTM) sits behind one contract, so callers program
against the contract and the model behind it can change without breaking them.

**Status: M1.** The contract, the conformance suite and an LSTM reference layer
are done. SRWM and DeltaNet arrive in M2.

## The contract

```python
class SequenceLayer(Protocol[S]):
    d_model: int

    def init_state(self, batch: int) -> S: ...
    def step(self, x, state: S) -> tuple[Tensor, S]: ...  # (B, d) -> (B, d)
    def scan(self, xs, state: S) -> tuple[Tensor, S]: ...  # (B, T, d) -> (B, T, d)
    def reset(self, state: S, mask) -> S: ...  # mask: (B,) bool
```

- **Layers are pure.** `step` and `scan` return a new state and never modify the
  one passed in. Mutable sessions arrive in M4, as a wrapper around this contract.
- **State is a frozen dataclass holding only tensors**, registered as a pytree.
  `state.step` is an int64 tensor of shape `(B,)`. The schema version is a class
  constant. Python scalars in the state would make `torch.compile` recompile.
- **Layers only see `(B, d_model)` vectors.** Tokens, vectors and feature tensors
  are converted by an `InputAdapter`, and predictions are produced by an
  `OutputHead` (`srwm.adapters`).

Writing a layer:

```python
from srwm import SequenceLayerBase, State, state_dataclass


@state_dataclass
class MyState(State):
    h: Tensor


class MyLayer(SequenceLayerBase[MyState]):
    def init_state(self, batch): ...
    def _step(self, x, state): ...  # step, scan and reset come from the base class
```

## Conformance suite

`srwm.testing` is public, so third-party layers can certify themselves:

```python
from srwm.testing import CHECKS, LayerSpec

SPEC = LayerSpec(factory=lambda: MyLayer(d_model=4), d_model=4)


@pytest.mark.parametrize("check", list(CHECKS))
def test_conformance(check):
    CHECKS[check](SPEC)
```

| Check | What it asserts |
|---|---|
| `state_is_tensor_pytree` | Only tensor fields; pytree round-trip; `step` is `(B,)` int64 starting at 0 |
| `shapes_and_dtypes` | float32 and float64; output and state shapes, dtypes and device stay stable |
| `scan_step_equivalence` | `scan` equals a `step` loop within 1e-6 (float64) |
| `reset` | Masked streams equal `init_state`; the others are bit-identical |
| `batch_independence` | Changing stream 0 changes only stream 0 |
| `purity` | `step`, `scan` and `reset` don't mutate their input state or the layer's parameters |
| `serialization` | `state_dict` round-trips through `torch.load(weights_only=True)`; a resumed stream equals an uninterrupted one; a wrong schema version is rejected |
| `determinism` | The same seed gives bit-identical results |
| `gradients` | `gradcheck` on inputs and every parameter (float64) |
| `compile_step` | `fullgraph=True`, exactly 1 compile across 100 streaming steps |
| `compile_scan` | `fullgraph=True` for a fixed chunk length, at most 1 recompile across 3 chunks |
| `compiled_matches_eager` | Inductor output equals eager within 1e-5 |

`tests/test_conformance_catches_bugs.py` checks that the suite fails layers with
known bugs: in-place state mutation, a stride-0 initial state, information
leaking between streams, and branching on tensor values.

## PyTorch internals

`torch.utils._pytree` and `torch._dynamo` are semi-private. They are only used
in `srwm/_compat.py`, so a PyTorch upgrade that moves them needs a fix in that
one file. Minimum supported version: torch 2.6.

## Development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest -q -p no:logging
```

CI runs only on CPU. The streaming speedup from CUDA graphs
(`mode="reduce-overhead"`) is not yet tested on a GPU.

## Roadmap

| # | Deliverable |
|---|---|
| M1 ✅ | Contract, conformance suite, LSTM |
| M2 | SRWM and DeltaNet layers |
| M3 | `TaskSampler` (linear and 2-layer MLP), `meta_train`, checkpoints |
| M4 | `Regressor` and `Session` API (observe, predict, save/load, fork); full set of adapters |
| M5 | Packaging, docs, v0.1.0 |
| M6 | JAX backend |
