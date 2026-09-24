"""NEER - Neural Embedding based Estimation and Reconstruction

Package: src/models/baselines
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

Phase 24 baseline models — every one of them implements
`src.evaluation.interface.BaselineModel` (`fit(train: PooledSplit)`,
`predict(split: PooledSplit) -> (n_samples, n_depth)`), so they are
fit and scored through the exact same
`src.evaluation.interface.evaluate_baseline(s)` path, on the exact
same `TensorBundle`-derived `PooledSplit`s, as each other and as NEER
itself:

* `ClimatologyBaseline` (`climatology.py`) — always available. Predicts
  the training-period per-depth mean for a sample's calendar month.
* `RidgeBaseline` (`ridge.py`) — needs `scikit-learn`. One regularized
  linear model per depth level.
* `LightGBMBaseline` (`lightgbm_baseline.py`) — optional, needs
  `lightgbm`. One gradient-boosted tree ensemble per depth level.
  `is_lightgbm_available()` lets a caller check before using it,
  matching Phase 24's "Optional LightGBM".

Both `RidgeBaseline` and `LightGBMBaseline` raise
`src.data.loaders.errors.MissingDependencyError` (not at import time —
only when actually constructed/fit) if their dependency is not
installed, the same lazy-import convention `src/data/dataset.py` and
`src/data/loaders` already use for `torch`/`xarray`/`netCDF4`.
"""

from src.models.baselines.climatology import ClimatologyBaseline
from src.models.baselines.lightgbm_baseline import LightGBMBaseline, is_lightgbm_available
from src.models.baselines.ridge import RidgeBaseline

#: Every baseline Phase 24 defines, in the order `scripts/evaluate_baselines.py`
#: runs them by default (LightGBM is skipped there automatically when
#: `not is_lightgbm_available()`).
BASELINE_REGISTRY = {
    "climatology": ClimatologyBaseline,
    "ridge": RidgeBaseline,
    "lightgbm": LightGBMBaseline,
}

__all__ = [
    "ClimatologyBaseline",
    "RidgeBaseline",
    "LightGBMBaseline",
    "is_lightgbm_available",
    "BASELINE_REGISTRY",
]