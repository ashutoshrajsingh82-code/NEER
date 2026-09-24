"""
Phase 24 — climatology baseline.

The simplest defensible predictor for a seasonal ocean field: for a
given depth level and calendar month, predict the *mean of the training
data* for that depth and month, regardless of anything else about the
input. Any model NEER trains has to beat this to be worth its added
complexity — it is the same "predict the seasonal mean" idea
`src.data.preprocessing.climatology.MonthlyClimatology` computes at the
spatial-field level (Phase 11), applied here to the pooled profile
quantity `src.evaluation.interface.PooledSplit` reduces every sample to,
so it is directly comparable to Ridge, LightGBM and NEER's own pooled
predictions on the same split.

Fit on training data only
--------------------------
`fit` only ever reads `train: PooledSplit` — the training split
`pool_tensor_bundle` carved from the `TensorBundle`'s own chronological
`split_masks`. Nothing about val or test ever reaches the per-month
means this baseline predicts from, so scoring it on val/test is a fair
test of generalization, not a lookup into data it was fitted on.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.evaluation.interface import BaselineModel, PooledSplit


class ClimatologyBaseline(BaselineModel):
    """Predicts, for each depth level, the training-period mean profile
    value for that sample's calendar month.

    Falls back to the training-period per-depth mean across *all*
    months for any depth/month combination with zero valid training
    observations (e.g. a very short training window that does not cover
    every calendar month), so `predict` never returns NaN.
    """

    name = "climatology"

    def __init__(self) -> None:
        self._fitted = False
        self._monthly_mean: np.ndarray | None = None   # (12, n_depth)
        self._monthly_has_data: np.ndarray | None = None  # (12, n_depth) bool
        self._overall_mean: np.ndarray | None = None   # (n_depth,) fallback
        self._n_depth: int = 0

    def config(self) -> Dict[str, Any]:
        return {}

    def fit(self, train: PooledSplit) -> "ClimatologyBaseline":
        n_depth = train.n_depth
        monthly_mean = np.zeros((12, n_depth), dtype=np.float64)
        monthly_has_data = np.zeros((12, n_depth), dtype=bool)

        for month in range(1, 13):
            rows = train.months == month
            if not rows.any():
                continue
            values = train.profile[rows]        # (n_rows, n_depth)
            mask = train.profile_mask[rows]      # (n_rows, n_depth)
            for d in range(n_depth):
                valid = mask[:, d]
                if valid.any():
                    monthly_mean[month - 1, d] = values[valid, d].mean()
                    monthly_has_data[month - 1, d] = True

        overall_mean = np.zeros(n_depth, dtype=np.float64)
        for d in range(n_depth):
            valid = train.profile_mask[:, d]
            overall_mean[d] = train.profile[valid, d].mean() if valid.any() else 0.0

        self._monthly_mean = monthly_mean
        self._monthly_has_data = monthly_has_data
        self._overall_mean = overall_mean
        self._n_depth = n_depth
        self._fitted = True
        return self

    def predict(self, split: PooledSplit) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ClimatologyBaseline.predict called before fit()")
        if split.n_depth != self._n_depth:
            raise ValueError(
                f"fitted on {self._n_depth} depth levels, got a split with {split.n_depth}"
            )

        prediction = np.empty((split.n_samples, self._n_depth), dtype=np.float32)
        for i, month in enumerate(split.months):
            row_has_data = self._monthly_has_data[month - 1]
            row = np.where(row_has_data, self._monthly_mean[month - 1], self._overall_mean)
            prediction[i] = row
        return prediction