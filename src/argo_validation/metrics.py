"""
Phase 26 — ARGO-vs-NEER comparison metrics and depth coverage.

Inputs are matched arrays, all `(n_profiles, n_depth)`:

    obs    the ARGO profile interpolated onto NEER's depths (NaN where the
           vertical rules left a depth empty)
    pred   NEER's prediction for that profile's month (and cell, if the
           model is gridded), degC
    valid  True where both exist

The metric definitions are the project's own — `src.evaluation.metrics`
(`rmse`, `mae`, `bias` = prediction - observation, `pearson`) — reused so
"RMSE" means the same thing here as in the reanalysis-target evaluation.
Only those pure functions are shared: no split logic, report object or
data structure from `src.evaluation` is used, so the two evaluations stay
independent.

Correlation caveat
-------------------
Pooled over all depths, Pearson r mostly measures whether the model knows
the ocean is warm at the surface and cold at 1000 m — which even a
climatology gets right, so it is near 1 almost regardless of skill. The
per-depth correlations are the informative ones and are reported for that
reason; the pooled figure is labelled accordingly in the report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.metrics import bias as _bias
from src.evaluation.metrics import mae as _mae
from src.evaluation.metrics import pearson as _pearson
from src.evaluation.metrics import rmse as _rmse

#: Below this many pairs a metric is flagged as low-sample in the report.
LOW_SAMPLE_PAIRS = 10


def _metric_block(obs: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
    n = int(np.asarray(valid, dtype=bool).sum())
    return {
        "rmse": _rmse(obs, pred, valid),
        "mae": _mae(obs, pred, valid),
        "bias": _bias(obs, pred, valid),
        "correlation": _pearson(obs, pred, valid),
        "n_pairs": n,
        "low_sample": n < LOW_SAMPLE_PAIRS,
    }


def depth_labels(depths_m: Sequence[float]) -> List[str]:
    return [f"{int(round(d))}m" for d in depths_m]


def compute_argo_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    depths_m: Sequence[float],
    splits: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Overall, per-depth and per-NEER-split metrics.

    `bias` is prediction minus observation (positive = NEER warmer than
    ARGO). `splits` (one entry per profile: ``train``/``val``/``test`` or
    ``""``) adds a `by_split` block — see the pipeline's independence note.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(obs) & np.isfinite(pred)
    labels = depth_labels(depths_m)

    result: Dict[str, Any] = {
        "convention": "bias = NEER - ARGO (degC); positive = NEER warmer than ARGO",
        "overall": _metric_block(obs, pred, valid),
        "per_depth": {labels[d]: _metric_block(obs[:, d], pred[:, d], valid[:, d]) for d in range(len(labels))},
    }
    result["overall"]["note"] = (
        "pooled across depths: correlation here is dominated by the vertical temperature "
        "gradient and is inflated; use per_depth correlation"
    )

    if splits is not None:
        s = np.asarray(splits, dtype=object)
        by_split: Dict[str, Any] = {}
        for name in sorted({str(x) for x in s}):
            m = np.asarray([str(x) == name for x in s])
            by_split[name or "unsplit"] = {
                "n_profiles": int(m.sum()),
                **_metric_block(obs[m], pred[m], valid[m]),
            }
        result["by_split"] = by_split
    return result


def compute_depth_coverage(valid: np.ndarray, depths_m: Sequence[float]) -> Dict[str, Any]:
    """How much of NEER's vertical grid the matched ARGO profiles cover."""
    valid = np.asarray(valid, dtype=bool)
    n_p, n_d = valid.shape if valid.ndim == 2 else (0, len(depths_m))
    labels = depth_labels(depths_m)
    if n_p == 0:
        return {"n_profiles": 0, "per_depth": {l: {"n_profiles": 0, "fraction": None} for l in labels}}

    depths = np.asarray(depths_m, dtype=float)
    deepest = np.array([depths[valid[i]].max() if valid[i].any() else np.nan for i in range(n_p)])
    per_profile_levels = valid.sum(axis=1)
    return {
        "n_profiles": int(n_p),
        "n_model_depths": int(n_d),
        "mean_levels_per_profile": float(per_profile_levels.mean()),
        "mean_fraction_of_model_depths": float(per_profile_levels.mean() / n_d),
        "profiles_covering_all_depths": int((per_profile_levels == n_d).sum()),
        "deepest_valid_depth_m": {
            "min": float(np.nanmin(deepest)),
            "median": float(np.nanmedian(deepest)),
            "max": float(np.nanmax(deepest)),
        },
        "per_depth": {
            labels[d]: {
                "n_profiles": int(valid[:, d].sum()),
                "fraction": float(valid[:, d].mean()),
            }
            for d in range(n_d)
        },
    }