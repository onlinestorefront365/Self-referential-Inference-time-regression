"""
Self-Referential Inference-Time Regressor

Core idea: at each inference step the model writes a Hebbian outer-product update
to its own fast weight matrix, then reads from it associatively. The fast weight
state W ∈ ℝ^(d_key × d_model) compresses the entire context history into a fixed
O(d²) representation regardless of sequence length, enabling intent extrapolation
far beyond the visible context window.

Write:  W_t = decay · W_{t-1} + lr · k_t ⊗ v_t
Read:   retrieved_t = q_t @ W_t
Intent: intent_t = head(flatten(W_t))  →  predicts steps T+1 … T+extrapolate_steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SRConfig


class FastWeightMemory(nn.Module):
    """
    Associative memory implemented as a fast weight matrix.

    Write path: normalized key ⊗ value → rank-1 additive update to W
    Read path:  scaled query @ W → retrieval via inner product

    After T steps: W = Σ_t decay^(T-t) · k_t ⊗ v_t
    This is a linear associative memory; nonlinear retrieval (via downstream MLP)
    makes the effective discriminability grow exponentially with d_key.
    """

    def __init__(self, d_model: int, d_key: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_key, bias=False)
        self.k = nn.Linear(d_model, d_key, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.scale = d_key ** -0.5
        self.d_key = d_key
        self.d_model = d_model

        nn.init.orthogonal_(self.k.weight)
        nn.init.orthogonal_(self.q.weight)

    def read(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_model)
        W: (B, d_key, d_model)
        returns: (B, d_model)
        """
        q = self.q(x) * self.scale                           # (B, d_key)
        retrieved = torch.bmm(q.unsqueeze(1), W).squeeze(1)  # (B, d_model)
        return self.out(F.gelu(retrieved))

    def write(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_model)
        returns delta_W: (B, d_key, d_model)
        """
        k = F.normalize(self.k(x), dim=-1)  # (B, d_key) — unit sphere for stability
        v = self.v(x)                         # (B, d_model)
        # outer product: k^T ⊗ v
        return torch.bmm(k.unsqueeze(-1), v.unsqueeze(-2))  # (B, d_key, d_model)

    def forward(
        self, x: torch.Tensor, W: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        retrieved = self.read(x, W)
        delta_W = self.write(x)
        return retrieved, delta_W

    def init_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.d_key, self.d_model, device=device)


class SRBlock(nn.Module):
    def __init__(self, d_model: int, d_key: int, dropout: float):
        super().__init__()
        self.fast_mem = FastWeightMemory(d_model, d_key)
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
        self, x: torch.Tensor, W: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        retrieved, delta_W = self.fast_mem(self.norm1(x), W)
        x = x + retrieved
        x = x + self.ff(self.norm2(x))
        return x, delta_W


class SelfReferentialRegressor(nn.Module):
    """
    Self-referential sequence model with inference-time weight updates.

    Processing loop per step t:
      1. Project input → hidden
      2. For each block: read from fast W, compute delta, update W
      3. Predict x_{t+1}  (next-step head)
      4. Extrapolate intent  (far-future head reads from flattened fast W state)

    The intent head has NO direct access to context values — it must read
    the generating-process parameters out of the compressed fast weight state.
    """

    def __init__(self, cfg: SRConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, cfg.context_len + cfg.extrapolate_steps + 64, cfg.d_model)
        )

        self.blocks = nn.ModuleList(
            [SRBlock(cfg.d_model, cfg.d_key, cfg.dropout) for _ in range(cfg.n_blocks)]
        )

        self.norm = nn.LayerNorm(cfg.d_model)
        self.pred_head = nn.Linear(cfg.d_model, cfg.output_dim)

        fw_dim = cfg.n_blocks * cfg.d_key * cfg.d_model
        self.intent_head = nn.Sequential(
            nn.Linear(fw_dim, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.extrapolate_steps * cfg.output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-apply orthogonal init for fast weight projections (overrides above)
        for block in self.blocks:
            nn.init.orthogonal_(block.fast_mem.k.weight)
            nn.init.orthogonal_(block.fast_mem.q.weight)

    def init_fast_states(self, batch: int, device) -> list[torch.Tensor]:
        return [b.fast_mem.init_state(batch, device) for b in self.blocks]

    def step(
        self,
        x_t: torch.Tensor,
        fast_states: list[torch.Tensor],
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Single step with fast weight read/write.
        x_t:        (B, input_dim)
        fast_states: list of (B, d_key, d_model) per block
        returns: pred (B, output_dim), intent (B, extrapolate_steps, output_dim), new states
        """
        B = x_t.shape[0]
        h = self.input_proj(x_t)
        if t < self.pos_emb.shape[1]:
            h = h + self.pos_emb[:, t, :]

        new_states: list[torch.Tensor] = []
        fw_flat: list[torch.Tensor] = []

        for block, W in zip(self.blocks, fast_states):
            h, delta_W = block(h, W)
            new_W = self.cfg.fast_decay * W + self.cfg.fast_lr * delta_W
            new_states.append(new_W)
            fw_flat.append(new_W.reshape(B, -1))

        h = self.norm(h)
        pred = self.pred_head(h)                              # (B, output_dim)

        fw_state = torch.cat(fw_flat, dim=-1)                 # (B, n_blocks*d_key*d_model)
        intent = self.intent_head(fw_state).view(             # (B, extrapolate_steps, output_dim)
            B, self.cfg.extrapolate_steps, self.cfg.output_dim
        )

        return pred, intent, new_states

    def forward(
        self,
        x: torch.Tensor,
        fast_states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        x: (B, T, input_dim)
        Returns:
          preds:   (B, T, output_dim)
          intents: (B, T, extrapolate_steps, output_dim)
          fast_states: updated list
        """
        B, T, _ = x.shape
        if fast_states is None:
            fast_states = self.init_fast_states(B, x.device)

        all_preds, all_intents = [], []
        for t in range(T):
            pred, intent, fast_states = self.step(x[:, t, :], fast_states, t)
            all_preds.append(pred)
            all_intents.append(intent)

        preds = torch.stack(all_preds, dim=1)    # (B, T, output_dim)
        intents = torch.stack(all_intents, dim=1) # (B, T, extrapolate_steps, output_dim)
        return preds, intents, fast_states
