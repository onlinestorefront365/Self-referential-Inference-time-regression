import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from .config import SRConfig
from .data import IntentDataset
from .model import PraxisRegressor
from .praxis import PraxisDecision, ActionVerb, N_VERBS


def _loss(
    preds: torch.Tensor,
    intents: torch.Tensor,
    decisions: list[PraxisDecision],
    targets_next: torch.Tensor,
    targets_future: torch.Tensor,
    cfg: SRConfig,
) -> tuple[torch.Tensor, dict]:
    # Next-step regression
    pred_loss = nn.functional.mse_loss(preds, targets_next)

    # Intent extrapolation (from final step, after full context)
    intent_loss = nn.functional.mse_loss(intents[:, -1], targets_future)

    # Coherence: late-context intent variance
    T = intents.shape[1]
    late = intents[:, T // 2:, :, :]
    coherence = late.var(dim=1).mean()

    # Praxis confidence regularization: encourage high confidence at the final step
    final_conf = decisions[-1].confidence              # (B,)
    conf_loss = (1.0 - final_conf).mean()

    # Praxis intent consistency: when model commits (COMMIT verb logit high),
    # future predictions should be more certain (lower variance)
    commit_logit = torch.stack(
        [d.verb_logits[:, ActionVerb.COMMIT] for d in decisions], dim=1
    )  # (B, T)
    # Soft penalty: high commit signal + high intent variance = bad
    intent_var = intents.var(dim=-2).mean(dim=-1)      # (B,)  variance over extrapolate dim
    commit_coherence = (torch.sigmoid(commit_logit[:, -1]) * intent_var).mean()

    total = (
        cfg.pred_weight * pred_loss
        + cfg.intent_weight * intent_loss
        + cfg.coherence_weight * coherence
        + 0.05 * conf_loss
        + 0.05 * commit_coherence
    )
    return total, {
        "pred": pred_loss.item(),
        "intent": intent_loss.item(),
        "coherence": coherence.item(),
        "conf": conf_loss.item(),
    }


def train(cfg: SRConfig, n_samples: int = 10_000, save_dir: str = "checkpoints") -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = IntentDataset(n_samples, cfg.context_len, cfg.extrapolate_steps, seed=42)
    n_val = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=2, pin_memory=True)

    model = PraxisRegressor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.n_epochs * len(train_dl)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr * 0.01)

    Path(save_dir).mkdir(exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(cfg.n_epochs):
        model.train()
        t0 = time.time()
        train_metrics: list[dict] = []

        for context, next_step, future, _ in train_dl:
            context = context.to(device)
            next_step = next_step.to(device)
            future = future.to(device)

            preds, intents, decisions, _ = model(context)
            loss, metrics = _loss(preds, intents, decisions, next_step, future, cfg)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_metrics.append(metrics)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for context, next_step, future, _ in val_dl:
                context = context.to(device)
                next_step = next_step.to(device)
                future = future.to(device)
                preds, intents, decisions, _ = model(context)
                loss, _ = _loss(preds, intents, decisions, next_step, future, cfg)
                val_losses.append(loss.item())

        avg_train = np.mean([m["pred"] + m["intent"] for m in train_metrics])
        avg_val = np.mean(val_losses)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1:3d}/{cfg.n_epochs} | "
            f"train {avg_train:.4f} | val {avg_val:.4f} | "
            f"pred {np.mean([m['pred'] for m in train_metrics]):.4f} | "
            f"intent {np.mean([m['intent'] for m in train_metrics]):.4f} | "
            f"{elapsed:.1f}s"
        )

        entry = {"epoch": epoch + 1, "train": float(avg_train), "val": float(avg_val)}
        history.append(entry)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({"cfg": cfg, "model_state": model.state_dict()}, f"{save_dir}/best.pt")

    with open(f"{save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val:.4f}  →  {save_dir}/best.pt")
    return model, history


if __name__ == "__main__":
    cfg = SRConfig()
    train(cfg)
