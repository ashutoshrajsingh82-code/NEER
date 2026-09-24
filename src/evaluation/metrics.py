"""
Phase 24/25 — common regression metrics for depth-profile predictions.

Every model this project scores — the climatology baseline, the Ridge
baseline, the optional LightGBM baseline, and (via the same
`(n_samples, n_depth)` profile shape `scripts/train.py` already pools
`NEERModel` predictions into) NEER itself — ends up producing a
`(n_samples, n_depth)` array of predicted temperature-anomaly profiles.
This module is the one place that turns a `(prediction, target, mask)`
triple of that shape into metrics, so "how good is this model" is
computed identically however the prediction was produced.

Masking
-------
`mask` is `True`/nonzero exactly where a `(sample, depth)` entry is a
real target (see `pool_profile` in `scripts/train.py` and
`PooledSplit` in `src/evaluation/interface.py` for how it is built).
Every function here treats masked-out entries as if they did not
exist — they are never averaged in, never counted, and never turned
into a fabricated zero-error. A depth level with zero valid entries
anywhere in the input reports `None` rather than `nan` silently
propagating into a report.

Phase 25 adds `pearson` (the Pearson correlation coefficient, distinct
from `r2` — they agree only when the prediction is an unbiased linear
rescaling of the target) as a first-class metric alongside RMSE/MAE/
bias, and a `per_variable` breakdown next to `per_depth`, for tensor
bundles whose targets stack more than one physical variable's depth
levels (`TensorBundle.target_names`; see `src/evaluation/interface.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """Flatten to 1D and keep only entries `mask` marks valid."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.asarray(mask).reshape(-1).astype(bool)
    return y_true[mask], y_pred[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Root-mean-squared error over valid entries only, or `None` if none."""
    t, p = _masked(y_true, y_pred, mask)
    if t.size == 0:
        return None
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Mean absolute error over valid entries only, or `None` if none."""
    t, p = _masked(y_true, y_pred, mask)
    if t.size == 0:
        return None
    return float(np.mean(np.abs(t - p)))


def bias(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Mean signed error (`prediction - target`); positive = over-prediction."""
    t, p = _masked(y_true, y_pred, mask)
    if t.size == 0:
        return None
    return float(np.mean(p - t))


def r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Coefficient of determination. `None` if fewer than 2 valid points or
    the target has zero variance (R^2 is undefined, not 1.0 or 0.0, there)."""
    t, p = _masked(y_true, y_pred, mask)
    if t.size < 2:
        return None
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot == 0:
        return None
    ss_res = np.sum((t - p) ** 2)
    return float(1.0 - ss_res / ss_tot)


def pearson(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Pearson product-moment correlation coefficient between `y_true` and
    `y_pred` over valid entries only.

    Unlike `r2`, this is invariant to a constant additive bias or a
    positive linear rescaling of the prediction — it measures whether
    the prediction *tracks* the target's ups and downs, not whether it
    matches its magnitude. A model can have a strong `pearson` and a
    poor `r2` (e.g. a systematically biased or mis-scaled but otherwise
    well-correlated prediction), which is exactly why Phase 25 reports
    both rather than one in place of the other.

    `None` if fewer than 2 valid points, or if either series has zero
    variance over those points (the correlation is undefined there, not
    0.0 or 1.0).
    """
    t, p = _masked(y_true, y_pred, mask)
    if t.size < 2:
        return None
    t_std = t.std()
    p_std = p.std()
    if t_std == 0 or p_std == 0:
        return None
    covariance = np.mean((t - t.mean()) * (p - p.mean()))
    return float(covariance / (t_std * p_std))


def n_valid(mask: np.ndarray) -> int:
    return int(np.asarray(mask).astype(bool).sum())


@dataclass
class MetricSet:
    """RMSE/MAE/bias/Pearson-r/R^2 plus the sample count they were computed
    from."""

    rmse: Optional[float]
    mae: Optional[float]
    bias: Optional[float]
    pearson: Optional[float]
    r2: Optional[float]
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "bias": self.bias,
            "pearson": self.pearson,
            "r2": self.r2,
            "n": self.n,
        }


def compute_metric_set(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> MetricSet:
    return MetricSet(
        rmse=rmse(y_true, y_pred, mask),
        mae=mae(y_true, y_pred, mask),
        bias=bias(y_true, y_pred, mask),
        pearson=pearson(y_true, y_pred, mask),
        r2=r2(y_true, y_pred, mask),
        n=n_valid(mask),
    )


@dataclass
class ProfileMetrics:
    """Metrics for a `(n_samples, n_depth)` profile prediction: one
    `MetricSet` pooled over every valid entry (`overall`), one per depth
    level (`per_depth`, so a model that is good on average but bad at
    500m does not hide behind the overall number), and — when the
    profile's depth columns are labeled with which physical variable
    they belong to — one per variable (`per_variable`), pooling every
    depth level that variable contributes so a model's temperature
    skill does not hide a poor salinity skill (or vice versa) behind
    one overall number. `per_variable` is empty when the profile is
    single-variable or no variable labels were given — see
    `compute_profile_metrics`'s `variable_names`.
    """

    overall: MetricSet
    per_depth: Dict[str, MetricSet] = field(default_factory=dict)
    per_variable: Dict[str, MetricSet] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "overall": self.overall.to_dict(),
            "per_depth": {name: metrics.to_dict() for name, metrics in self.per_depth.items()},
        }
        if self.per_variable:
            result["per_variable"] = {
                name: metrics.to_dict() for name, metrics in self.per_variable.items()
            }
        return result


def compute_profile_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    *,
    depth_names: Optional[Sequence[str]] = None,
    variable_names: Optional[Sequence[str]] = None,
) -> ProfileMetrics:
    """`y_true`/`y_pred`/`mask` are all `(n_samples, n_depth)`.

    `depth_names` labels the per-depth breakdown (e.g. `["0m", "50m",
    ...]`); defaults to the column index as a string when not given.

    `variable_names`, when given, must have one entry per depth column
    naming which target variable that column belongs to (e.g. a bundle
    stacking `subsurface_temp`'s 15 depths then `salinity`'s 15 depths
    would pass `["subsurface_temp"] * 15 + ["salinity"] * 15`). Columns
    sharing a name are pooled into one `MetricSet` in `per_variable`,
    in first-seen order. Omitted (or all-identical, i.e. genuinely one
    variable) `variable_names` leaves `per_variable` empty — a
    single-variable profile has nothing a variable breakdown would add
    beyond `per_depth`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask).astype(bool)
    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape} mask={mask.shape}"
        )
    if y_true.ndim != 2:
        raise ValueError(f"expected (n_samples, n_depth); got shape {y_true.shape}")

    n_depth = y_true.shape[1]
    names = list(depth_names) if depth_names is not None else [str(i) for i in range(n_depth)]
    if len(names) != n_depth:
        raise ValueError(f"depth_names has {len(names)} entries but there are {n_depth} depths")

    overall = compute_metric_set(y_true, y_pred, mask)
    per_depth = {
        names[d]: compute_metric_set(y_true[:, d], y_pred[:, d], mask[:, d])
        for d in range(n_depth)
    }

    per_variable: Dict[str, MetricSet] = {}
    if variable_names is not None:
        var_names = list(variable_names)
        if len(var_names) != n_depth:
            raise ValueError(
                f"variable_names has {len(var_names)} entries but there are {n_depth} depths"
            )
        distinct = list(dict.fromkeys(var_names))
        if len(distinct) > 1:
            for var in distinct:
                columns = [d for d, name in enumerate(var_names) if name == var]
                per_variable[var] = compute_metric_set(
                    y_true[:, columns], y_pred[:, columns], mask[:, columns]
                )

    return ProfileMetrics(overall=overall, per_depth=per_depth, per_variable=per_variable)