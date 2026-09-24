"""NEER - Neural Embedding based Estimation and Reconstruction

Package: src/evaluation
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

Phase 24 adds the common evaluation interface every model - the
baselines in `src/models/baselines` and, via the same pooled-profile
quantity `scripts/train.py` already reduces `NEERModel`'s predictions
to, NEER itself - is scored through:

* `src.evaluation.metrics` - RMSE / MAE / bias / Pearson-r / R^2,
  overall, per depth level, and (where more than one target variable
  is present) per variable, computed the same way for every model.
* `src.evaluation.interface` - `PooledSplit` / `pool_tensor_bundle`
  (reads the *same* `TensorBundle` `.npz` file and split masks every
  model must be compared on), the `BaselineModel` interface every
  baseline implements, and `evaluate_baseline(s)` / `EvaluationReport`
  that turns a fitted model plus pooled data into a metrics report.

Phase 25 builds the evaluation engine on top of this: JSON reports
(`scripts/run_evaluation.py`) and RMSE/MAE/bias/Pearson-r-vs-depth
plots (`src.evaluation.plots`), both of which clearly label results
computed from `TensorBundle.attrs["is_synthetic"]` data as synthetic.
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
    pearson,
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
    "pearson",
    "r2",
    "n_valid",
]