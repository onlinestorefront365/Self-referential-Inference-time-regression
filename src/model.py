"""
Praxis AI — Self-Referential Inference-Time Regressor with Modal Logic.

At each inference step the model:
  1. Projects input to hidden space
  2. Runs N modal fast-weight channels (□ necessity, ◇ possibility, P permission, A ability)
     — each channel has distinct write rate and decay, encoding different modal dynamics
  3. Reads from all channels via associative query (Kripke accessibility)
  4. Updates the modal state (Hebbian outer-product write, operator-gated)
  5. Predicts next value  (regression head)
  6. Extrapolates intent  (intent head reads from compressed modal state)
  7. Produces a PraxisDecision  (verb, justification, confidence, execute/defer)

The fast weight matrix for each operator IS a compressed Kripke frame:
  keys   ~ world descriptors (accessible-world index)
  values ~ propositions that hold in those worlds
  query  ~ current world → retrieves what holds in accessible worlds

Necessity:   propositions that hold in ALL retrieved worlds (high cosine similarity)
Possibility: propositions that hold in ANY retrieved world
Permission:  deontic overlay: permits vs. forbids in accessible worlds
Ability:     agent capability envelope (stable, slow-changing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SRConfig
from .modal import ModalStateTracker, ModalState
from .praxis import PraxisHead, PraxisDecision


class PraxisBlock(nn.Module):
    def __init__(self, d_model: int, d_key: int, dropout: float):
        super().__init__()
        self.modal_tracker = ModalStateTracker(d_model, d_key)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, state: ModalState
    ) -> tuple[torch.Tensor, ModalState]:
        retrieved, new_state = self.modal_tracker(self.norm1(x), state)
        x = x + retrieved
        x = x + self.ff(self.norm2(x))
        return x, new_state


class PraxisRegressor(nn.Module):
    """
    Full Praxis AI model.

    Inputs:  sequence x (B, T, input_dim)
    Outputs:
      preds      (B, T, output_dim)         — next-step regression
      intents    (B, T, extrapolate, 1)     — far-future intent extrapolation
      decisions  list of T PraxisDecision   — modal action decisions per step
      modal_states  list of T list[ModalState] — per-step modal world states
    """

    def __init__(self, cfg: SRConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, cfg.context_len + cfg.extrapolate_steps + 64, cfg.d_model)
        )

        self.blocks = nn.ModuleList([
            PraxisBlock(cfg.d_model, cfg.d_key, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])

        self.norm = nn.LayerNorm(cfg.d_model)
        self.pred_head = nn.Linear(cfg.d_model, cfg.output_dim)

        # Intent head: reads from flattened modal state (all blocks, all operators)
        from .modal import N_MODAL
        fw_dim = cfg.n_blocks * N_MODAL * cfg.d_key * cfg.d_model
        self.intent_head = nn.Sequential(
            nn.Linear(fw_dim, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.extrapolate_steps * cfg.output_dim),
        )

        self.praxis_head = PraxisHead(cfg.d_model, cfg.d_key, cfg.n_blocks)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for block in self.blocks:
            for writer in block.modal_tracker.writers:
                nn.init.orthogonal_(writer.k.weight)
            for reader in block.modal_tracker.readers:
                nn.init.orthogonal_(reader.q.weight)

    def init_modal_states(self, batch: int, device) -> list[ModalState]:
        return [b.modal_tracker.init_state(batch, device) for b in self.blocks]

    def step(
        self,
        x_t: torch.Tensor,
        modal_states: list[ModalState],
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, PraxisDecision, list[ModalState]]:
        B = x_t.shape[0]
        h = self.input_proj(x_t)
        if t < self.pos_emb.shape[1]:
            h = h + self.pos_emb[:, t, :]

        new_states: list[ModalState] = []
        for block, state in zip(self.blocks, modal_states):
            h, new_state = block(h, state)
            new_states.append(new_state)

        h = self.norm(h)
        pred = self.pred_head(h)

        # Intent from full modal state
        modal_flat = torch.cat([ms.flat() for ms in new_states], dim=-1)
        intent = self.intent_head(modal_flat).view(B, self.cfg.extrapolate_steps, self.cfg.output_dim)

        # Praxis decision
        decision = self.praxis_head(h, new_states)

        return pred, intent, decision, new_states

    def forward(
        self,
        x: torch.Tensor,
        modal_states: list[ModalState] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[PraxisDecision], list[ModalState]]:
        """
        x: (B, T, input_dim)
        Returns:
          preds     (B, T, output_dim)
          intents   (B, T, extrapolate_steps, output_dim)
          decisions list[T PraxisDecision]
          modal_states  final per-block modal state
        """
        B, T, _ = x.shape
        if modal_states is None:
            modal_states = self.init_modal_states(B, x.device)

        all_preds, all_intents, all_decisions = [], [], []

        for t in range(T):
            pred, intent, decision, modal_states = self.step(x[:, t, :], modal_states, t)
            all_preds.append(pred)
            all_intents.append(intent)
            all_decisions.append(decision)

        preds = torch.stack(all_preds, dim=1)
        intents = torch.stack(all_intents, dim=1)
        return preds, intents, all_decisions, modal_states
