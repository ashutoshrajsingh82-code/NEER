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

__all__ = [
    "DEFAULT_DEPTH_DIM",
    "DEFAULT_EPS",
    "DEFAULT_SPATIAL_DIMS",
    "NEERLossConfig",
    "anomaly_loss",
    "gradient_loss",
    "masked_mse_loss",
    "vertical_smoothness_loss",
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