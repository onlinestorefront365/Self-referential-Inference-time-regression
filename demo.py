"""
Praxis AI Demo — Self-Referential Inference-Time Regressor with Modal Logic

Usage:
  python demo.py              # train if no checkpoint, then visualize
  python demo.py --quick      # 5-epoch smoke test
  python demo.py --no-train   # visualize with untrained model
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SRConfig
from src.data import GENERATORS
from src.infer import load_model, run_inference, describe_decisions
from src.modal import ModalOp, MODAL_DYNAMICS
from src.model import PraxisRegressor


def _make_sequence(intent: str, total_len: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    seq = GENERATORS[intent](total_len, rng)
    noise = (rng.randn(total_len) * 0.02).astype(np.float32)
    return seq + noise


def plot_intent_extrapolation(model: PraxisRegressor, cfg: SRConfig, out: str):
    intents = list(GENERATORS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Praxis AI — Intent Extrapolation via Modal Fast Weight Updates", fontsize=12)

    for ax, intent in zip(axes.flatten(), intents):
        total = cfg.context_len + cfg.extrapolate_steps + 1
        seq = _make_sequence(intent, total)
        ctx_np = seq[: cfg.context_len]
        future_np = seq[cfg.context_len: cfg.context_len + cfg.extrapolate_steps]

        context = torch.tensor(ctx_np).unsqueeze(0).unsqueeze(-1)
        result = run_inference(model, context)

        t_ctx = np.arange(cfg.context_len)
        t_fut = np.arange(cfg.context_len, cfg.context_len + cfg.extrapolate_steps)

        ax.plot(t_ctx, ctx_np, "steelblue", lw=2, label="Context")
        ax.plot(t_fut, future_np, "seagreen", lw=1.5, ls="--", label="True future")
        ax.plot(t_fut, result["final_intent"][0, :, 0].numpy(),
                "tomato", lw=2, label="Intent extrapolation")
        ax.axvline(cfg.context_len, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_title(intent, fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_modal_dynamics(model: PraxisRegressor, cfg: SRConfig, out: str):
    """
    Show per-operator fast weight accumulation over context steps.
    □ necessity builds slowly and persists.
    ◇ possibility saturates faster but decays more.
    P permission and A ability behave distinctly.
    """
    seq = _make_sequence("sinusoidal", cfg.context_len + cfg.extrapolate_steps + 10)
    ctx_np = seq[: cfg.context_len]
    context = torch.tensor(ctx_np).unsqueeze(0).unsqueeze(-1)
    result = run_inference(model, context, n_autoregressive=cfg.extrapolate_steps)

    fw_norms = result["fw_norms"]   # (T, N_MODAL)
    T = fw_norms.shape[0]
    t = np.arange(T)

    op_colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Praxis AI — Modal Fast Weight Dynamics", fontsize=12)

    ax = axes[0]
    for op in ModalOp:
        label = MODAL_DYNAMICS[op]["label"]
        ax.plot(t, fw_norms[:, op], color=op_colors[op], lw=2, label=label)
    ax.set_title("Per-operator fast weight norm vs context step")
    ax.set_xlabel("Step")
    ax.set_ylabel("||W_op||₂")
    ax.legend()
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    t_ctx = np.arange(cfg.context_len)
    t_ar = np.arange(cfg.context_len, cfg.context_len + cfg.extrapolate_steps)
    t_true = np.arange(len(seq))

    ax2.plot(t_true, seq, "seagreen", lw=1, ls="--", alpha=0.4, label="True")
    ax2.plot(t_ctx, ctx_np, "steelblue", lw=2, label="Context")
    if result["autoregressive"] is not None:
        ax2.plot(t_ar, result["autoregressive"][0, :, 0].numpy(),
                 "tomato", lw=2, label="Autoregressive (from modal state)")
    ax2.axvline(cfg.context_len, color="gray", ls=":", lw=1, alpha=0.6)
    ax2.set_title("Autoregressive continuation from modal fast weight state")
    ax2.legend()
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_praxis_decisions(model: PraxisRegressor, cfg: SRConfig):
    """Print the step-by-step praxis decisions for a sample sequence."""
    for intent in ["sinusoidal", "exponential"]:
        seq = _make_sequence(intent, cfg.context_len + 1)
        ctx_np = seq[: cfg.context_len]
        context = torch.tensor(ctx_np).unsqueeze(0).unsqueeze(-1)
        result = run_inference(model, context)

        print(f"\nPraxis decisions for '{intent}' sequence:")
        print(describe_decisions(result["decisions"], idx=0))


def plot_training_history(history: list[dict], out: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([h["epoch"] for h in history], [h["train"] for h in history], label="Train", lw=2)
    ax.plot([h["epoch"] for h in history], [h["val"] for h in history], label="Val", lw=2)
    ax.set_title("Training History — Praxis AI")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5-epoch quick train")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    cfg = SRConfig()
    ckpt_path = Path("checkpoints/best.pt")
    history = None

    if args.no_train:
        model = PraxisRegressor(cfg)
        print("Using untrained model.")
    elif ckpt_path.exists():
        model, cfg = load_model(str(ckpt_path))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        from src.train import train as do_train
        cfg.n_epochs = 5 if args.quick else 50
        n_samples = 2_000 if args.quick else 10_000
        print(f"Training: {cfg.n_epochs} epochs, {n_samples} samples...")
        model, history = do_train(cfg, n_samples=n_samples)

    model.eval()

    print("\n--- Plots ---")
    plot_intent_extrapolation(model, cfg, "intent_extrapolation.png")
    plot_modal_dynamics(model, cfg, "modal_dynamics.png")
    if history:
        plot_training_history(history, "training_history.png")

    print("\n--- Praxis Decisions ---")
    print_praxis_decisions(model, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
