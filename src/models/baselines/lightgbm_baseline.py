"""
Phase 24 — optional LightGBM baseline.

Same pooled-features -> pooled-profile task as `RidgeBaseline`
(`src/models/baselines/ridge.py`), but with one gradient-boosted
regression tree ensemble per depth level instead of a linear model —
captures nonlinear channel interactions Ridge cannot, at the cost of
needing more data to avoid overfitting the small pooled feature vector.

`lightgbm` is optional, same lazy-import treatment as `torch`
(`src/data/dataset.py`) and `scikit-learn` (`ridge.py` in this
package): importing this module never requires it installed, and only
constructing `LightGBMBaseline` (or calling `fit`) raises
`MissingDependencyError` naming the install command. Phase 24 lists
LightGBM as optional for exactly this reason — the climatology and
Ridge baselines, and the common evaluation interface, work fully
without it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.evaluation.interface import BaselineModel, PooledSplit

DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = -1  # LightGBM convention: no limit
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_NUM_LEAVES = 15
#: Below this many valid training rows for a depth level, a boosted-tree
#: model is too likely to overfit the pooled feature vector to be
#: meaningful; that depth falls back to its training mean instead.
MIN_TRAINING_ROWS = 10


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            package="lightgbm",
            purpose="the LightGBM baseline (src/models/baselines/lightgbm_baseline.py)",
            install_hint="pip install lightgbm",
        ) from exc
    return lgb


class LightGBMBaseline(BaselineModel):
    """One `lightgbm.LGBMRegressor` per depth level, on the same pooled
    input features `RidgeBaseline` uses (no standardization needed —
    tree splits are scale-invariant).

    Follows the same per-depth masking and small-data fallback as
    `RidgeBaseline.fit`: a depth is only fit on rows where its
    `profile_mask` is `True`, and depths with fewer than
    `MIN_TRAINING_ROWS` valid training rows predict their training mean
    instead of a model that would just memorize a handful of points.
    """

    name = "lightgbm"

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        num_leaves: int = DEFAULT_NUM_LEAVES,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.random_state = int(random_state)
        self._fitted = False
        self._models: List[Any] = []
        self._depth_fallback: Optional[np.ndarray] = None
        self._n_depth: int = 0

    def config(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
        }

    def fit(self, train: PooledSplit) -> "LightGBMBaseline":
        lgb = _import_lightgbm()

        n_depth = train.n_depth
        models: List[Any] = [None] * n_depth
        fallback = np.zeros(n_depth, dtype=np.float64)

        for d in range(n_depth):
            valid = train.profile_mask[:, d]
            y = train.profile[valid, d]
            fallback[d] = y.mean() if y.size else 0.0
            if valid.sum() < MIN_TRAINING_ROWS:
                continue
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbosity=-1,
            )
            model.fit(train.features[valid], y)
            models[d] = model

        self._models = models
        self._depth_fallback = fallback
        self._n_depth = n_depth
        self._fitted = True
        return self

    def predict(self, split: PooledSplit) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LightGBMBaseline.predict called before fit()")
        if split.n_depth != self._n_depth:
            raise ValueError(
                f"fitted on {self._n_depth} depth levels, got a split with {split.n_depth}"
            )

        prediction = np.empty((split.n_samples, self._n_depth), dtype=np.float32)
        for d, model in enumerate(self._models):
            if model is None:
                prediction[:, d] = self._depth_fallback[d]
            else:
                prediction[:, d] = model.predict(split.features)
        return prediction


def is_lightgbm_available() -> bool:
    """Whether `lightgbm` is importable, without raising if it is not —
    lets callers (e.g. `scripts/evaluate_baselines.py`) skip it
    gracefully instead of crashing an otherwise-complete run."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True