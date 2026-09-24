"""
Phase 24 — Ridge regression baseline.

A linear model from the pooled input channels (Phase 08's eleven-channel
`NEER_CHANNEL_ORDER`, spatially averaged per sample by
`src.evaluation.interface.pool_tensor_bundle`) to the pooled
temperature-anomaly profile — one independently regularized linear
model per depth level, since the vertical temperature gradient means
different depths have very different scales (the same reasoning
`src.data.preprocessing.normalization.Normalizer` applies per depth
level, Phase 07).

`scikit-learn` is an optional dependency, same lazy-import treatment as
`torch` in `src/data/dataset.py` and `xarray`/`netCDF4` in
`src/data/loaders`: importing this module never requires it to be
installed, and only constructing `RidgeBaseline` (or calling `fit`)
raises `MissingDependencyError` naming the install command.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.evaluation.interface import BaselineModel, PooledSplit

DEFAULT_ALPHA = 1.0


def _import_sklearn():
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            package="scikit-learn",
            purpose="the Ridge regression baseline (src/models/baselines/ridge.py)",
            install_hint="pip install scikit-learn",
        ) from exc
    return Ridge, StandardScaler


class RidgeBaseline(BaselineModel):
    """One `sklearn.linear_model.Ridge` per depth level, on standardized
    (training-fit-only) input features.

    A depth level's model is fit only on the rows where that depth's
    `profile_mask` is `True` in the training split — rows with no valid
    training target at that depth are excluded from that depth's fit
    rather than being given a fabricated `0.0` target to regress onto
    (mirroring how `pool_profile` in `scripts/train.py` keeps
    gap-filled depths out of `NEERLoss` via exactly the same kind of
    mask). If a depth level has fewer than two valid training rows, its
    "model" predicts that depth's training mean (or `0.0` if there are
    none at all) — Ridge cannot be fit meaningfully from fewer points
    than that, and this keeps `predict` total rather than raising deep
    into a rarely-hit edge case.
    """

    name = "ridge"

    def __init__(self, *, alpha: float = DEFAULT_ALPHA, random_state: int = 0) -> None:
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self._fitted = False
        self._scaler = None
        self._models: List[Any] = []
        self._depth_fallback: Optional[np.ndarray] = None
        self._n_depth: int = 0

    def config(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "random_state": self.random_state}

    def fit(self, train: PooledSplit) -> "RidgeBaseline":
        Ridge, StandardScaler = _import_sklearn()

        scaler = StandardScaler()
        X = scaler.fit_transform(train.features)

        n_depth = train.n_depth
        models: List[Any] = [None] * n_depth
        fallback = np.zeros(n_depth, dtype=np.float64)

        for d in range(n_depth):
            valid = train.profile_mask[:, d]
            y = train.profile[valid, d]
            fallback[d] = y.mean() if y.size else 0.0
            if valid.sum() < 2:
                continue  # not enough training points; falls back to the mean
            model = Ridge(alpha=self.alpha, random_state=self.random_state)
            model.fit(X[valid], y)
            models[d] = model

        self._scaler = scaler
        self._models = models
        self._depth_fallback = fallback
        self._n_depth = n_depth
        self._fitted = True
        return self

    def predict(self, split: PooledSplit) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RidgeBaseline.predict called before fit()")
        if split.n_depth != self._n_depth:
            raise ValueError(
                f"fitted on {self._n_depth} depth levels, got a split with {split.n_depth}"
            )

        X = self._scaler.transform(split.features)
        prediction = np.empty((split.n_samples, self._n_depth), dtype=np.float32)
        for d, model in enumerate(self._models):
            if model is None:
                prediction[:, d] = self._depth_fallback[d]
            else:
                prediction[:, d] = model.predict(X)
        return prediction