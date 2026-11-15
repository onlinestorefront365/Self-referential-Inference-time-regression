# Self-Referential Inference-Time Regressor

A sequence model that updates its own weights at inference time, enabling intent
extrapolation far beyond the visible context window.

## Core idea

Standard transformers hold context in a KV cache that grows linearly with sequence
length. This model instead compresses context into a **fast weight matrix**
`W ∈ ℝ^(d_key × d_model)` that is updated at every inference step via a
Hebbian outer-product rule:

```
W_t = decay · W_{t-1} + lr · k_t ⊗ v_t
```

The fast weight state has fixed size regardless of sequence length. With nonlinear
retrieval, the effective discriminability of stored patterns grows exponentially with
`d_key` (VC dimension argument), giving exponential effective context capacity.

A separate **intent head** reads only from the flattened fast weight state — not
from the raw context — and predicts far-future values. Because it has no direct
access to context tokens, it must learn to extract the generating-process
parameters (slope, frequency, growth rate, …) from the compressed weight
representation. This is the "intent extrapolation" property.

## Architecture

```
Input x_t ──► input_proj
                │
            SRBlock × n_blocks:
              ├── FastWeightMemory
              │     write: k_t ⊗ v_t  → ΔW
              │     read:  q_t @ W_t  → retrieved
              │     update: W_t = decay·W_{t-1} + lr·ΔW
              └── Feed-forward (LayerNorm + GELU)
                │
            pred_head  ──► x̂_{t+1}  (next-step prediction)
            intent_head ─► x̂_{T+1..T+K}  (far-future intent, reads from W only)
```

## Usage

```bash
pip install torch numpy matplotlib

# Full training (10k samples, 50 epochs)
python demo.py

# Quick smoke test (2k samples, 5 epochs)
python demo.py --quick

# Visualize with untrained model
python demo.py --no-train
```

Produces:
- `intent_extrapolation.png` — model extrapolation vs ground truth for each intent type
- `fast_weight_dynamics.png` — fast weight accumulation and autoregressive continuation
- `training_history.png` — train/val loss curves

## Training objective

| Component | Loss | Weight |
|-----------|------|--------|
| Next-step prediction | MSE on x̂_{t+1} vs x_{t+1} | 1.0 |
| Intent extrapolation | MSE on final intent vs x_{T+1..T+K} | 0.5 |
| Coherence | Variance of late-context intent predictions | 0.1 |

The coherence term encourages the intent prediction to stabilize once the model
has seen sufficient context, rather than changing arbitrarily with each new token.

## Intent types

The dataset contains sequences from six generating functions:
linear, quadratic, exponential, sinusoidal, Fibonacci, power.

## References

- Ba et al. (2016) — "Using Fast Weights to Attend to the Recent Past"
- Schmidhuber (1992) — "Learning to Control Fast-Weight Memories"
- Sun et al. (2024) — "Learning to (Learn at Test Time)"
