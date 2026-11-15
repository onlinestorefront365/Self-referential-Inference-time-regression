"""
Inference utilities.

run_inference: steps through context, optionally continues autoregressively.
load_model: loads a checkpoint.
"""

import numpy as np
import torch

from .config import SRConfig
from .model import SelfReferentialRegressor


def run_inference(
    model: SelfReferentialRegressor,
    context: torch.Tensor,
    n_autoregressive: int = 0,
) -> dict:
    """
    context: (B, T, 1)
    n_autoregressive: steps to continue autoregressively after context

    Returns dict:
      next_step_preds  (B, T, 1)
      final_intent     (B, extrapolate_steps, 1)  — intent after full context
      autoregressive   (B, n_autoregressive, 1) or None
      fw_norms         (T,)  L2 norm of fast weight state per step
    """
    model.eval()
    device = next(model.parameters()).device
    context = context.to(device)
    B, T, _ = context.shape

    fast_states = model.init_fast_states(B, device)
    all_preds: list[torch.Tensor] = []
    all_intents: list[torch.Tensor] = []
    fw_norms: list[float] = []

    with torch.no_grad():
        for t in range(T):
            pred, intent, fast_states = model.step(context[:, t, :], fast_states, t)
            all_preds.append(pred)
            all_intents.append(intent)
            fw_norms.append(sum(W.norm().item() for W in fast_states))

        preds = torch.stack(all_preds, dim=1)       # (B, T, 1)
        final_intent = all_intents[-1]               # (B, extrapolate_steps, 1)

        ar_preds: list[torch.Tensor] = []
        if n_autoregressive > 0:
            last = preds[:, -1, :]                   # (B, 1) — use last pred as next input
            for t in range(n_autoregressive):
                pred, _, fast_states = model.step(last, fast_states, T + t)
                ar_preds.append(pred)
                last = pred

    return {
        "next_step_preds": preds.cpu(),
        "final_intent": final_intent.cpu(),
        "autoregressive": torch.stack(ar_preds, dim=1).cpu() if ar_preds else None,
        "fw_norms": np.array(fw_norms),
    }


def load_model(path: str) -> tuple[SelfReferentialRegressor, SRConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg: SRConfig = ckpt["cfg"]
    model = SelfReferentialRegressor(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg
