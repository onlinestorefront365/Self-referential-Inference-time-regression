"""
Praxis Bridge — translates modal reasoning into executable actions.

"Praxis" (from Greek πρᾶξις): the translation of theory into action.

The bridge reads the four modal state vectors and decides:
  1. WHICH action to execute (action head)
  2. WHICH modality grounds it (justification head)
  3. HOW strongly it is warranted (confidence head)
  4. WHETHER to execute now or defer (execute vs. defer decision)

The execute/defer logic mirrors modal reasoning:
  - □P (necessity) → EXECUTE immediately — no alternative world avoids it
  - ◇P (possibility) → PROPOSE — may execute if no constraints block
  - P permitted → EXECUTE if permitted and □ or ◇ supports it
  - ¬P forbidden → BLOCK — permission channel overrides
  - A able → prerequisite check; if ¬A, DEFER until ability is acquired

ActionVerb categories follow speech act theory + planning semantics:
  ASSERT    — declare a fact (changes knowledge state)
  QUERY     — request information (opens possibility space)
  EXECUTE   — perform a physical/computational action (closes a possibility)
  DEFER     — delay action (keeps possibility open)
  BLOCK     — prevent an action (closes a possibility, enforces ¬P)
  COMMIT    — establish necessity for future (□P)
  RETRACT   — revoke a prior necessity or permission
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modal import ModalState, ModalOp, N_MODAL


class ActionVerb(IntEnum):
    ASSERT = 0
    QUERY = 1
    EXECUTE = 2
    DEFER = 3
    BLOCK = 4
    COMMIT = 5
    RETRACT = 6

    @property
    def label(self) -> str:
        return self.name.lower()


N_VERBS = len(ActionVerb)

# Which modal operators primarily license each verb
VERB_MODAL_AFFINITY: dict[ActionVerb, list[ModalOp]] = {
    ActionVerb.ASSERT:   [ModalOp.NECESSITY],
    ActionVerb.QUERY:    [ModalOp.POSSIBILITY],
    ActionVerb.EXECUTE:  [ModalOp.NECESSITY, ModalOp.PERMISSION, ModalOp.ABILITY],
    ActionVerb.DEFER:    [ModalOp.POSSIBILITY],
    ActionVerb.BLOCK:    [ModalOp.PERMISSION],
    ActionVerb.COMMIT:   [ModalOp.NECESSITY],
    ActionVerb.RETRACT:  [ModalOp.PERMISSION, ModalOp.NECESSITY],
}


@dataclass
class PraxisDecision:
    """The output of the praxis bridge for a single step."""
    verb: torch.Tensor          # (B,) int — ActionVerb index
    verb_logits: torch.Tensor   # (B, N_VERBS) — raw scores
    justification: torch.Tensor # (B, N_MODAL) — which modality grounds this action
    confidence: torch.Tensor    # (B,) in [0, 1]
    content: torch.Tensor       # (B, d_model) — action content vector (what to act on)
    should_execute: torch.Tensor # (B,) bool — execute now vs. defer

    def describe(self, idx: int = 0) -> str:
        verb = ActionVerb(self.verb[idx].item())
        just_idx = self.justification[idx].argmax().item()
        just_label = ModalOp(just_idx).name
        conf = self.confidence[idx].item()
        exe = self.should_execute[idx].item()
        return (
            f"{verb.label.upper()} "
            f"[grounded by {just_label}, conf={conf:.2f}, "
            f"{'EXECUTE' if exe else 'DEFER'}]"
        )


class PraxisHead(nn.Module):
    """
    Reads the full modal state → produces a PraxisDecision.

    Architecture:
      modal_state.flat() → modal_encoder → modal_repr (B, d_modal)
      hidden_state       → content (B, d_model)
      [modal_repr, content] → verb_head, justification_head, confidence_head

    Execute/defer logic:
      execute = (necessity_signal > threshold) OR
                (possibility_signal > threshold AND permission_signal > 0 AND ability_signal > 0)
      Gradients flow through the soft versions of these comparisons.
    """

    def __init__(self, d_model: int, d_key: int, n_blocks: int, execute_threshold: float = 0.5):
        super().__init__()
        self.execute_threshold = execute_threshold

        modal_flat_dim = n_blocks * N_MODAL * d_key * d_model
        d_modal = d_model * 2

        # Encode the full modal state
        self.modal_encoder = nn.Sequential(
            nn.Linear(modal_flat_dim, d_modal * 2),
            nn.GELU(),
            nn.Linear(d_modal * 2, d_modal),
        )

        d_joint = d_modal + d_model

        # Action verb head
        self.verb_head = nn.Sequential(
            nn.Linear(d_joint, d_model),
            nn.GELU(),
            nn.Linear(d_model, N_VERBS),
        )

        # Justification head: which modality grounds the verb
        self.justification_head = nn.Linear(d_joint, N_MODAL)

        # Confidence head: how strongly does the modal evidence support execution
        self.confidence_head = nn.Sequential(
            nn.Linear(d_joint, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Per-operator readout scalars for execute/defer logic
        # Maps modal signals → (necessity_score, possibility_score, permission_score, ability_score)
        self.modal_scalars = nn.Linear(d_modal, N_MODAL)

    def forward(
        self,
        h: torch.Tensor,         # (B, d_model) — final hidden state at this step
        modal_states: list[ModalState],  # one per block
    ) -> PraxisDecision:
        # Flatten all modal states from all blocks
        all_flats = torch.cat([ms.flat() for ms in modal_states], dim=-1)  # (B, n_blocks*N_MODAL*d_key*d_model)
        modal_repr = self.modal_encoder(all_flats)                           # (B, d_modal)

        joint = torch.cat([modal_repr, h], dim=-1)                          # (B, d_joint)

        verb_logits = self.verb_head(joint)                                  # (B, N_VERBS)
        verb = verb_logits.argmax(dim=-1)                                    # (B,)

        justification = F.softmax(self.justification_head(joint), dim=-1)   # (B, N_MODAL)
        confidence = self.confidence_head(joint).squeeze(-1)                 # (B,)

        # Execute/defer logic via soft modal signals
        modal_sigs = torch.sigmoid(self.modal_scalars(modal_repr))           # (B, N_MODAL)
        nec  = modal_sigs[:, ModalOp.NECESSITY]
        poss = modal_sigs[:, ModalOp.POSSIBILITY]
        perm = modal_sigs[:, ModalOp.PERMISSION]
        abil = modal_sigs[:, ModalOp.ABILITY]

        # □P alone licenses execution; ◇P requires P ∧ A
        execute_score = nec + poss * perm * abil
        should_execute = execute_score > self.execute_threshold

        return PraxisDecision(
            verb=verb,
            verb_logits=verb_logits,
            justification=justification,
            confidence=confidence,
            content=h,
            should_execute=should_execute,
        )
