from dataclasses import dataclass


@dataclass
class SRConfig:
    # Architecture
    d_model: int = 64
    d_key: int = 32           # fast weight key/query dimension
    n_blocks: int = 2
    dropout: float = 0.1

    # Fast weight dynamics — per-operator rates set in modal.py MODAL_DYNAMICS

    # Task
    input_dim: int = 1
    output_dim: int = 1
    context_len: int = 32
    extrapolate_steps: int = 64

    # Training
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50

    # Loss weights
    pred_weight: float = 1.0
    intent_weight: float = 0.5
    coherence_weight: float = 0.1
