"""NEER - Neural Embedding based Estimation and Reconstruction

Package: src/training
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS
"""

from __future__ import annotations

from src.training.losses import (
    DEFAULT_DEPTH_DIM,
    DEFAULT_EPS,
    DEFAULT_SPATIAL_DIMS,
    NEERLossConfig,
    anomaly_loss,
    gradient_loss,
    masked_mse_loss,
    vertical_smoothness_loss,
)

from src.training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointManager,
    CompatibilityReport,
    LoadedCheckpoint,
    check_compatibility,
    config_fingerprint,
    git_metadata,
    load_checkpoint,
    save_checkpoint,
)

from src.training.pretrain import (
    DEFAULT_PRETRAIN_CHECKPOINT_PATH,
    DEFAULT_PRETRAIN_HISTORY_PATH,
    load_pretrained_encoder,
    load_pretrained_into_neer_model,
    load_training_history,
    run_pretrain_epoch,
    save_pretrained_encoder_checkpoint,
    save_training_history,
    train_pretrain,
)

__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointManager",
    "CompatibilityReport",
    "LoadedCheckpoint",
    "check_compatibility",
    "config_fingerprint",
    "git_metadata",
    "load_checkpoint",
    "save_checkpoint",
    "DEFAULT_DEPTH_DIM",
    "DEFAULT_EPS",
    "DEFAULT_SPATIAL_DIMS",
    "NEERLossConfig",
    "anomaly_loss",
    "gradient_loss",
    "masked_mse_loss",
    "vertical_smoothness_loss",
    "DEFAULT_PRETRAIN_CHECKPOINT_PATH",
    "DEFAULT_PRETRAIN_HISTORY_PATH",
    "save_pretrained_encoder_checkpoint",
    "load_pretrained_encoder",
    "load_pretrained_into_neer_model",
    "save_training_history",
    "load_training_history",
    "run_pretrain_epoch",
    "train_pretrain",
]

# `NEERLoss` itself is only defined when torch is installed (same lazy
# pattern as `src.models`); re-export it too when available so
# `from src.training import NEERLoss` works without reaching into
# `src.training.losses` directly.
try:  # pragma: no cover - environment dependent
    from src.training.losses import NEERLoss

    __all__.append("NEERLoss")
except ImportError:  # pragma: no cover - environment dependent
    pass