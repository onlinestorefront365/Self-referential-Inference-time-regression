"""Inference utilities for Praxis AI."""

import numpy as np
import torch

from .config import SRConfig
from .model import PraxisRegressor
from .modal import ModalState
from .praxis import PraxisDecision, ActionVerb


def run_inference(
    model: PraxisRegressor,
    context: torch.Tensor,
    n_autoregressive: int = 0,
) -> dict:
    """
    context: (B, T, 1)
    Returns dict:
      next_step_preds  (B, T, 1)
      final_intent     (B, extrapolate_steps, 1)
      decisions        list[T PraxisDecision]
      autoregressive   (B, n_autoregressive, 1) or None
      fw_norms         (T, N_MODAL)  per-operator fast weight norms over time
    """
    model.eval()
    device = next(model.parameters()).device
    context = context.to(device)
    B, T, _ = context.shape

    modal_states = model.init_modal_states(B, device)
    all_preds: list[torch.Tensor] = []
    all_intents: list[torch.Tensor] = []
    all_decisions: list[PraxisDecision] = []
    fw_norms: list[list[float]] = []

    with torch.no_grad():
        for t in range(T):
            pred, intent, decision, modal_states = model.step(
                context[:, t, :], modal_states, t
            )
            all_preds.append(pred)
            all_intents.append(intent)
            all_decisions.append(decision)

            # Track per-operator norms across all blocks
            step_norms = []
            from .modal import N_MODAL
            for op_i in range(N_MODAL):
                norm = sum(ms.W[op_i].norm().item() for ms in modal_states)
                step_norms.append(norm)
            fw_norms.append(step_norms)

        preds = torch.stack(all_preds, dim=1)
        final_intent = all_intents[-1]

        ar_preds: list[torch.Tensor] = []
        if n_autoregressive > 0:
            last = preds[:, -1, :]
            for t in range(n_autoregressive):
                pred, _, _, modal_states = model.step(last, modal_states, T + t)
                ar_preds.append(pred)
                last = pred

    return {
        "next_step_preds": preds.cpu(),
        "final_intent": final_intent.cpu(),
        "decisions": all_decisions,
        "autoregressive": torch.stack(ar_preds, dim=1).cpu() if ar_preds else None,
        "fw_norms": np.array(fw_norms),  # (T, N_MODAL)
    }


def load_model(path: str) -> tuple[PraxisRegressor, SRConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg: SRConfig = ckpt["cfg"]
    model = PraxisRegressor(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def describe_decisions(decisions: list[PraxisDecision], idx: int = 0) -> str:
    """Pretty-print the praxis decision sequence for sample idx."""
    lines = []
    for t, d in enumerate(decisions):
        lines.append(f"  t={t:2d}: {d.describe(idx)}")
    return "\n".join(lines)
