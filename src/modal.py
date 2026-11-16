"""
Modal logic substrate for Praxis AI.

Implements four modal operators tracked as separate fast-weight channels:

  □  Necessity   — P holds in ALL accessible worlds from current state
  ◇  Possibility — P holds in SOME accessible world from current state
  P  Permission  — deontic: P is permitted/obligated (Kripke deontic frame)
  A  Ability     — alethic: agent has capacity to bring about P

Each operator has distinct dynamics:
  - Necessity decays slowly and is hard to revoke (once □P is established it persists)
  - Possibility decays faster; possibilities close as context constrains the world
  - Permission is set/cleared discretely by explicit grant/revoke signals
  - Ability is the most stable; capability envelope changes rarely

Kripke connection:
  A modal fast-weight matrix W_m stores associations between world-descriptors
  (keys) and proposition-values. The accessibility relation R(w, w') is encoded
  implicitly: high inner product q_w · k_{w'} means w' is accessible from w.
  Necessity: P holds in w iff ∀w' s.t. R(w,w'), P(w') — read as: retrievals
             with high cosine similarity all predict P.
  Possibility: P holds in w iff ∃w' s.t. R(w,w') ∧ P(w') — any retrieval
              direction that activates P.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalOp(IntEnum):
    NECESSITY = 0    # □  must / will / always / inevitably
    POSSIBILITY = 1  # ◇  can / could / might / perhaps
    PERMISSION = 2   # P  may / allowed / permitted / authorized
    ABILITY = 3      # A  able to / capable of / has the power to

    @classmethod
    def from_marker(cls, word: str) -> "ModalOp | None":
        """Identify modal operator from a surface marker word."""
        MARKER_MAP = {
            "must": cls.NECESSITY, "will": cls.NECESSITY,
            "always": cls.NECESSITY, "inevitably": cls.NECESSITY,
            "necessarily": cls.NECESSITY, "need": cls.NECESSITY,
            "can": cls.POSSIBILITY, "could": cls.POSSIBILITY,
            "might": cls.POSSIBILITY, "perhaps": cls.POSSIBILITY,
            "possibly": cls.POSSIBILITY, "maybe": cls.POSSIBILITY,
            "may": cls.PERMISSION, "allowed": cls.PERMISSION,
            "permitted": cls.PERMISSION, "authorized": cls.PERMISSION,
            "forbidden": cls.PERMISSION, "prohibited": cls.PERMISSION,
            "able": cls.ABILITY, "capable": cls.ABILITY,
            "power": cls.ABILITY, "capacity": cls.ABILITY,
        }
        return MARKER_MAP.get(word.lower().rstrip(".,;:!?"))


# Per-operator dynamics: (write_lr, decay, label)
MODAL_DYNAMICS = {
    ModalOp.NECESSITY:   dict(lr=0.15, decay=0.98, label="□ necessity"),
    ModalOp.POSSIBILITY: dict(lr=0.10, decay=0.90, label="◇ possibility"),
    ModalOp.PERMISSION:  dict(lr=0.20, decay=0.97, label="P permission"),
    ModalOp.ABILITY:     dict(lr=0.05, decay=0.99, label="A ability"),
}

N_MODAL = len(ModalOp)


@dataclass
class ModalState:
    """
    Per-step modal world state: one fast-weight matrix per operator.
    W[op]: (B, d_key, d_model) — the Kripke accessibility + valuation matrix
           for operator `op`.
    """
    W: list[torch.Tensor]  # len == N_MODAL

    @classmethod
    def zeros(cls, batch: int, d_key: int, d_model: int, device) -> "ModalState":
        return cls(W=[
            torch.zeros(batch, d_key, d_model, device=device)
            for _ in range(N_MODAL)
        ])

    def flat(self) -> torch.Tensor:
        """Concatenate all operator matrices → (B, N_MODAL * d_key * d_model)"""
        B = self.W[0].shape[0]
        return torch.cat([w.reshape(B, -1) for w in self.W], dim=-1)

    def necessity_dominance(self) -> torch.Tensor:
        """
        Returns (B, d_model): the proposition most strongly necessitated.
        Uses the □ channel's dominant singular direction as a proxy for □P.
        """
        W_nec = self.W[ModalOp.NECESSITY]      # (B, d_key, d_model)
        # Sum over key dimension — aggregates all accessible worlds' evidence
        return W_nec.sum(dim=1)                 # (B, d_model)

    def possibility_space(self) -> torch.Tensor:
        """(B, d_model): spread of possibility (high values = many open worlds)."""
        return self.W[ModalOp.POSSIBILITY].abs().sum(dim=1)

    def permission_mask(self) -> torch.Tensor:
        """(B, d_model): net permission signal (positive = permitted, negative = forbidden)."""
        return self.W[ModalOp.PERMISSION].sum(dim=1)

    def ability_envelope(self) -> torch.Tensor:
        """(B, d_model): what the agent is currently capable of."""
        return self.W[ModalOp.ABILITY].sum(dim=1)


class ModalFastWrite(nn.Module):
    """
    Writes a single Hebbian outer-product update to the modal fast weight
    matrix for a given operator, respecting its dynamics (lr, decay).
    """

    def __init__(self, d_model: int, d_key: int, op: ModalOp):
        super().__init__()
        self.op = op
        dyn = MODAL_DYNAMICS[op]
        self.lr = dyn["lr"]
        self.decay = dyn["decay"]

        self.k = nn.Linear(d_model, d_key, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.k.weight)

    def forward(
        self, h: torch.Tensor, W: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """
        h:    (B, d_model) — current hidden state
        W:    (B, d_key, d_model) — current fast weight matrix for this op
        gate: (B,) — scalar in [0,1] controlling write strength for this op
        Returns updated W.
        """
        k = F.normalize(self.k(h), dim=-1)                      # (B, d_key)
        v = self.v(h)                                             # (B, d_model)
        delta = torch.bmm(k.unsqueeze(-1), v.unsqueeze(-2))      # (B, d_key, d_model)
        g = gate.view(-1, 1, 1)
        return self.decay * W + self.lr * g * delta


class ModalFastRead(nn.Module):
    """Reads from a modal fast weight matrix via associative query."""

    def __init__(self, d_model: int, d_key: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_key, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.scale = d_key ** -0.5

    def forward(self, h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        q = self.q(h) * self.scale                              # (B, d_key)
        retrieved = torch.bmm(q.unsqueeze(1), W).squeeze(1)    # (B, d_model)
        return self.out(F.gelu(retrieved))


class ModalGating(nn.Module):
    """
    Given the hidden state, produces N_MODAL gate scalars in [0,1].
    Each gate controls how strongly the current token writes to each modal channel.

    A token bearing "must" should write heavily to □; "may" to P; etc.
    The gating is learned end-to-end but initialised so each operator responds
    to a different linear direction of the input.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, N_MODAL)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → gates: (B, N_MODAL) in [0, 1]"""
        return torch.sigmoid(self.gate_proj(h))


class ModalStateTracker(nn.Module):
    """
    One per SRBlock. Maintains N_MODAL separate fast weight matrices,
    each with its own dynamics, gated by the current hidden state.

    At each step:
      1. Compute modal gates from hidden state
      2. Read from all N_MODAL matrices → fused retrieval
      3. Write to all N_MODAL matrices with gated Hebbian updates
      4. Return fused retrieval + updated ModalState
    """

    def __init__(self, d_model: int, d_key: int):
        super().__init__()
        self.gating = ModalGating(d_model)
        self.writers = nn.ModuleList([
            ModalFastWrite(d_model, d_key, ModalOp(i)) for i in range(N_MODAL)
        ])
        self.readers = nn.ModuleList([
            ModalFastRead(d_model, d_key) for _ in range(N_MODAL)
        ])
        # Fuse N_MODAL retrievals back into d_model
        self.fusion = nn.Linear(N_MODAL * d_model, d_model)
        self.d_key = d_key
        self.d_model = d_model

    def forward(
        self, h: torch.Tensor, state: ModalState
    ) -> tuple[torch.Tensor, ModalState]:
        """
        h:     (B, d_model)
        state: ModalState
        Returns: (fused_retrieval (B, d_model), new_state)
        """
        gates = self.gating(h)                   # (B, N_MODAL)

        retrievals: list[torch.Tensor] = []
        new_Ws: list[torch.Tensor] = []

        for i in range(N_MODAL):
            retrieved = self.readers[i](h, state.W[i])
            retrievals.append(retrieved)
            new_W = self.writers[i](h, state.W[i], gates[:, i])
            new_Ws.append(new_W)

        fused = self.fusion(torch.cat(retrievals, dim=-1))   # (B, d_model)
        return fused, ModalState(W=new_Ws)

    def init_state(self, batch: int, device) -> ModalState:
        return ModalState.zeros(batch, self.d_key, self.d_model, device)
