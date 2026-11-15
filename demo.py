"""
Demo: Self-Referential Inference-Time Regressor

Usage:
  python demo.py              # train if no checkpoint, then visualize
  python demo.py --quick      # 5-epoch quick train (smoke test)
  python demo.py --no-train   # visualize with untrained model
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SRConfig
from src.data import GENERATORS, IntentDataset
from src.infer import load_model, run_inference
from src.model import SelfReferentialRegressor


def _make_sequence(intent: str, total_len: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    seq = GENERATORS[intent](total_len, rng)
    noise = (rng.randn(total_len) * 0.02).astype(np.float32)
    return seq + noise


def plot_intent_extrapolation(model: SelfReferentialRegressor, cfg: SRConfig, out: str):
    intents = list(GENERATORS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Intent Extrapolation via Inference-Time Fast Weight Updates", fontsize=13, y=1.01
    )

    for ax, intent in zip(axes.flatten(), intents):
        total = cfg.context_len + cfg.extrapolate_steps
        seq = _make_sequence(intent, total + 1)

        ctx_np = seq[: cfg.context_len]
        future_np = seq[cfg.context_len : cfg.context_len + cfg.extrapolate_steps]

        context = torch.tensor(ctx_np).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        result = run_inference(model, context)

        t_ctx = np.arange(cfg.context_len)
        t_fut = np.arange(cfg.context_len, cfg.context_len + cfg.extrapolate_steps)

        ax.plot(t_ctx, ctx_np, "steelblue", lw=2, label="Context")
        ax.plot(t_fut, future_np, "seagreen", lw=1.5, ls="--", label="True future")
        ax.plot(
            t_fut,
            result["final_intent"][0, :, 0].numpy(),
            "tomato",
            lw=2,
            label="Intent extrapolation",
        )
        ax.axvline(cfg.context_len, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_title(intent, fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_fast_weight_dynamics(model: SelfReferentialRegressor, cfg: SRConfig, out: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fast weight accumulation vs context length
    seq = _make_sequence("sinusoidal", cfg.context_len * 3 + cfg.extrapolate_steps + 10)
    for ctx_len, color in zip([8, 16, 32, cfg.context_len], ["#aad4f5", "#5ba3d0", "#1f6ea8", "#0a2d4e"]):
        ctx = torch.tensor(seq[:ctx_len]).unsqueeze(0).unsqueeze(-1)
        result = run_inference(model, ctx)
        ax1.plot(result["fw_norms"], color=color, lw=1.5, label=f"ctx={ctx_len}")

    ax1.set_title("Fast Weight Norm vs Context Step\n(context compression accumulation)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("||W_fast||₂")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # Right: autoregressive continuation from fast weight state
    ctx_np = seq[: cfg.context_len]
    context = torch.tensor(ctx_np).unsqueeze(0).unsqueeze(-1)
    result = run_inference(model, context, n_autoregressive=cfg.extrapolate_steps)

    t_all = np.arange(len(seq))
    t_ctx = np.arange(cfg.context_len)
    t_ar = np.arange(cfg.context_len, cfg.context_len + cfg.extrapolate_steps)

    ax2.plot(t_all[: cfg.context_len + cfg.extrapolate_steps + 10],
             seq[: cfg.context_len + cfg.extrapolate_steps + 10],
             "seagreen", lw=1, ls="--", alpha=0.5, label="True sequence")
    ax2.plot(t_ctx, ctx_np, "steelblue", lw=2, label="Context")
    if result["autoregressive"] is not None:
        ax2.plot(t_ar, result["autoregressive"][0, :, 0].numpy(),
                 "tomato", lw=2, label="Autoregressive continuation")

    ax2.axvline(cfg.context_len, color="gray", ls=":", lw=1, alpha=0.6)
    ax2.set_title("Autoregressive Generation from Fast Weight State")
    ax2.set_xlabel("Step")
    ax2.legend()
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_training_history(history: list[dict], out: str):
    epochs = [h["epoch"] for h in history]
    trains = [h["train"] for h in history]
    vals = [h["val"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, trains, label="Train", lw=2)
    ax.plot(epochs, vals, label="Val", lw=2)
    ax.set_title("Training History")
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
    parser.add_argument("--no-train", action="store_true", help="skip training, use random weights")
    args = parser.parse_args()

    cfg = SRConfig()
    ckpt_path = Path("checkpoints/best.pt")
    history = None

    if args.no_train:
        model = SelfReferentialRegressor(cfg)
        print("Using untrained model (--no-train).")
    elif ckpt_path.exists():
        model, cfg = load_model(str(ckpt_path))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        from src.train import train as do_train

        if args.quick:
            cfg.n_epochs = 5
            n_samples = 2000
        else:
            n_samples = 10_000

        print(f"Training for {cfg.n_epochs} epochs on {n_samples} samples...")
        model, history = do_train(cfg, n_samples=n_samples)

    model.eval()

    print("\n--- Generating plots ---")
    plot_intent_extrapolation(model, cfg, "intent_extrapolation.png")
    plot_fast_weight_dynamics(model, cfg, "fast_weight_dynamics.png")
    if history:
        plot_training_history(history, "training_history.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
