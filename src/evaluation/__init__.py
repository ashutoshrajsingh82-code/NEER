"""NEER - Neural Embedding based Estimation and Reconstruction

Package: src/evaluation
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

Phase 24 adds the common evaluation interface every model - the
baselines in `src/models/baselines` and, via the same pooled-profile
quantity `scripts/train.py` already reduces `NEERModel`'s predictions
to, NEER itself - is scored through:

* `src.evaluation.metrics` - RMSE / MAE / bias / R^2, overall and per
  depth level, computed the same way for every model.
* `src.evaluation.interface` - `PooledSplit` / `pool_tensor_bundle`
  (reads the *same* `TensorBundle` `.npz` file and split masks every
  model must be compared on), the `BaselineModel` interface every
  baseline implements, and `evaluate_baseline(s)` / `EvaluationReport`
  that turns a fitted model plus pooled data into a metrics report.
"""

from src.evaluation.interface import (
    SPLIT_NAMES,
    BaselineModel,
    EvaluationReport,
    PooledSplit,
    evaluate_baseline,
    evaluate_baselines,
    pool_tensor_bundle,
)
from src.evaluation.metrics import (
    MetricSet,
    ProfileMetrics,
    compute_metric_set,
    compute_profile_metrics,
    mae,
    n_valid,
    r2,
    rmse,
)
from src.evaluation.metrics import bias as bias_metric

__all__ = [
    "SPLIT_NAMES",
    "BaselineModel",
    "EvaluationReport",
    "PooledSplit",
    "evaluate_baseline",
    "evaluate_baselines",
    "pool_tensor_bundle",
    "MetricSet",
    "ProfileMetrics",
    "compute_metric_set",
    "compute_profile_metrics",
    "rmse",
    "mae",
    "bias_metric",
    "r2",
    "n_valid",
]