"""
NEER - Neural Embedding based Estimation and Reconstruction

Package: src/argo_validation
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

Phase 26 — independent ARGO validation. Compares NEER's predicted
temperature profiles with ARGO float profiles:

    ARGO profiles -> QC -> time matching -> spatial matching
      -> vertical interpolation -> NEER prediction -> comparison -> metrics

Deliberately separate from `src/evaluation` (reanalysis / test-target
evaluation): different data, different matching, different reports under
``reports/argo_validation/``. Only the four scalar metric functions are
shared. Demo/synthetic ARGO data is always labelled and never reported
as validation — see `pipeline.py`.
"""

from src.argo_validation.demo import (
    DEFAULT_DEMO_FILENAME,
    DISCLAIMER as DEMO_DISCLAIMER,
    demo_stand_in_predictions,
    generate_demo_argo,
)
from src.argo_validation.interpolation import VerticalConfig, interpolate_to_depths
from src.argo_validation.io import load_argo, load_argo_csv, load_argo_netcdf
from src.argo_validation.matching import (
    NeerGrid,
    SpaceMatchConfig,
    TimeMatchConfig,
    match_space,
    match_time,
)
from src.argo_validation.metrics import compute_argo_metrics, compute_depth_coverage
from src.argo_validation.pipeline import (
    ArgoValidationConfig,
    ArgoValidationResult,
    run_argo_validation,
    write_outputs,
)
from src.argo_validation.predictions import (
    NeerPredictions,
    load_prediction_table,
    predictions_from_checkpoint,
)
from src.argo_validation.profiles import (
    ArgoProfile,
    ArgoProfileSet,
    clean_profile,
    pressure_to_depth,
)

__all__ = [
    "ArgoProfile", "ArgoProfileSet", "ArgoValidationConfig", "ArgoValidationResult",
    "DEFAULT_DEMO_FILENAME", "DEMO_DISCLAIMER", "NeerGrid", "NeerPredictions",
    "SpaceMatchConfig", "TimeMatchConfig", "VerticalConfig", "clean_profile",
    "compute_argo_metrics", "compute_depth_coverage", "demo_stand_in_predictions",
    "generate_demo_argo", "interpolate_to_depths", "load_argo", "load_argo_csv",
    "load_argo_netcdf", "load_prediction_table", "match_space", "match_time",
    "predictions_from_checkpoint", "pressure_to_depth", "run_argo_validation", "write_outputs",
]