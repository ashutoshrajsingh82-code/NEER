"""
Phase 29B-1 — the evaluation layer behind `GET /metrics` and
`GET /evaluation/argo`.

    FastAPI route -> Pydantic validation -> (this module) -> existing
    evaluation/model/data modules -> response schema

Nothing here reimplements a metric or a validation step; every number
comes from a function `src/evaluation` or `src/argo_validation` already
had before this phase:

* `metrics()` scores the already-loaded `NEERRepository.service`/`bundle`
  the same way `scripts/run_evaluation.py::evaluate_neer_checkpoint`
  scores a checkpoint — `src.evaluation.interface.pool_tensor_bundle` for
  the data, `src.evaluation.metrics.compute_profile_metrics` for the
  numbers — but reuses the model the repository already loaded once at
  startup instead of loading the checkpoint from disk a second time.
* `argo_evaluation()` is a thin wrapper around the Phase 26 pipeline
  (`src.argo_validation.run_argo_validation`), reusing the same
  checkpoint -> predictions conversion (`predictions_from_checkpoint`)
  and demo/labelling rules (`generate_demo_argo`, `demo_stand_in_predictions`)
  `scripts/run_argo_validation.py` uses.

Never-fabricate rule (Requirement 4/7)
----------------------------------------
Neither function ever invents a metric, a prediction, or an ARGO profile.
Missing data/model/checkpoint/metadata always raises the matching
`NeerApiError` (503/404) instead. The one form of synthetic data this
module will produce — a `DEMO_SYNTHETIC` ARGO pipeline-check — only runs
when the caller explicitly passes `demo=True`; it is never a silent
fallback, and the response it returns is labelled `DEMO_SYNTHETIC` /
`observational_validation: false` throughout, exactly as
`src.argo_validation.pipeline` already labels it for the CLI script.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.app.errors import (
    DataUnavailableError,
    InferenceFailedError,
    InvalidParameterError,
    ModelUnavailableError,
)
from backend.app.services.repository import NEERRepository
from src.argo_validation import (
    ArgoValidationConfig,
    NeerGrid,
    demo_stand_in_predictions,
    generate_demo_argo,
    load_argo,
    predictions_from_checkpoint,
    run_argo_validation,
)
from src.data.loaders.errors import LoaderError, MissingDependencyError
from src.data.preprocessing._utils import json_safe
from src.evaluation.interface import SPLIT_NAMES, pool_tensor_bundle
from src.evaluation.metrics import compute_profile_metrics

# --------------------------------------------------------------------------
# GET /metrics
# --------------------------------------------------------------------------


def metrics(repository: NEERRepository, *, split: str) -> Dict[str, Any]:
    """Real RMSE/MAE/bias/Pearson-r/R^2 for `split`, computed by running the
    already-loaded model over the already-loaded tensor bundle.

    Raises `InvalidParameterError` for an unknown split, `DataUnavailableError`
    if no dataset (or no targets, or no such split) is loaded,
    `ModelUnavailableError` if no checkpoint is loaded, and
    `InferenceFailedError` if the forward pass itself fails or the
    checkpoint's output shape does not match the loaded dataset's depths.
    """
    if split not in SPLIT_NAMES:
        raise InvalidParameterError(f"split must be one of {list(SPLIT_NAMES)}, got {split!r}")

    bundle = repository.require_bundle()
    service = repository.require_service()

    if bundle.targets is None or bundle.target_mask is None:
        raise DataUnavailableError(
            "the loaded dataset has no target values; cannot compute evaluation metrics "
            "(it was preprocessed without a target variable)"
        )

    try:
        pooled = pool_tensor_bundle(bundle, splits=(split,), source_tensors_path=repository.tensor_path)
    except KeyError as exc:
        raise DataUnavailableError(f"the loaded dataset has no '{split}' split: {exc}") from exc
    except ValueError as exc:
        raise DataUnavailableError(f"could not pool the tensor bundle for evaluation: {exc}") from exc

    pooled_split = pooled[split]
    if pooled_split.n_samples == 0:
        raise DataUnavailableError(f"split '{split}' has no samples in the loaded dataset")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - service already requires torch to exist
        raise ModelUnavailableError("torch is not installed; install it to compute metrics") from exc

    mask = np.asarray(bundle.split_masks[split], dtype=bool)
    inputs = torch.from_numpy(np.asarray(bundle.inputs[mask], dtype=np.float32)).to(service.device)
    try:
        with torch.no_grad():
            prediction = service.model(inputs).detach().to("cpu").numpy()
    except (RuntimeError, ValueError) as exc:
        raise InferenceFailedError(f"metrics computation failed: {exc}") from exc

    if prediction.shape != pooled_split.profile.shape:
        raise InferenceFailedError(
            f"model produced predictions of shape {prediction.shape}, expected "
            f"{pooled_split.profile.shape}; the loaded checkpoint does not match the loaded dataset"
        )

    profile_metrics = compute_profile_metrics(
        pooled_split.profile,
        prediction,
        pooled_split.profile_mask,
        depth_names=pooled_split.depth_names,
        variable_names=pooled_split.variable_names or None,
    )

    is_synthetic = bool(bundle.is_synthetic)
    return {
        "phase": 29,
        "split": split,
        "n_samples": pooled_split.n_samples,
        "is_synthetic": is_synthetic,
        "data_mode": str(bundle.attrs.get("data_mode", "UNKNOWN")),
        "disclaimer": (
            "Computed on synthetic/demo data — NOT real observations. For pipeline "
            "demonstration only."
            if is_synthetic
            else None
        ),
        "source_tensors_path": str(repository.tensor_path),
        "checkpoint": None if repository.checkpoint_path is None else str(repository.checkpoint_path),
        "metrics": profile_metrics.to_dict(),
    }


# --------------------------------------------------------------------------
# GET /evaluation/argo
# --------------------------------------------------------------------------


def _find_argo_source(raw_dir: Path) -> Optional[Path]:
    """Where real ARGO data conventionally lives under `paths.data_raw`
    (mirrors `repository._find_checkpoint`'s "try the obvious names" shape):
    `argo_profiles.csv`, or an `argo/` directory of `.nc` files (or, failing
    that, the first `.csv` inside it)."""
    if not raw_dir.exists():
        return None
    candidate = raw_dir / "argo_profiles.csv"
    if candidate.exists():
        return candidate
    argo_dir = raw_dir / "argo"
    if argo_dir.is_dir():
        if any(argo_dir.glob("*.nc")):
            return argo_dir
        csvs = sorted(argo_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
    return None


def _real_predictions(repository: NEERRepository, bundle) -> "tuple[Optional[Any], Optional[str]]":
    """`predictions_from_checkpoint`, reusing the repository's already-resolved
    checkpoint/metadata paths. Returns `(predictions, None)` on success or
    `(None, reason)` — never raises, so the caller can decide whether a
    missing/broken checkpoint is fatal (real run) or falls back to the
    labelled demo stand-in (`demo=True` run)."""
    if repository.service is None:
        return None, repository.model_error()
    if repository.checkpoint_path is None:
        return None, "no NEER checkpoint is loaded"
    if not repository.metadata_path.exists():
        return None, (
            f"no preprocessing metadata at {repository.metadata_path} "
            "(needed to convert the checkpoint's output back to degC)"
        )
    try:
        predictions = predictions_from_checkpoint(
            repository.checkpoint_path, bundle, repository.metadata_path
        )
    except (FileNotFoundError, ValueError, MissingDependencyError) as exc:
        return None, str(exc)
    return predictions, None


def argo_evaluation(repository: NEERRepository, *, demo: bool) -> Dict[str, Any]:
    """Run the Phase 26 ARGO validation pipeline and return its report.

    With `demo=False` (the default): uses the real, already-loaded
    checkpoint to predict, and real ARGO data found under
    `paths.data_raw` (Requirement 5). Either being unavailable is a
    `ModelUnavailableError`/`DataUnavailableError` — never a fabricated
    result (Requirement 7).

    With `demo=True`: an explicit opt-in to a clearly labelled
    `DEMO_SYNTHETIC` pipeline check (never the default), matching
    `scripts/run_argo_validation.py --demo`. Real predictions are still
    used when a checkpoint is available; only the ARGO side becomes
    demo data.
    """
    bundle = repository.require_bundle()
    try:
        grid = NeerGrid.from_bundle(bundle)
    except ValueError as exc:
        raise DataUnavailableError(f"the loaded dataset cannot be used for ARGO validation: {exc}") from exc

    predictions, prediction_error = _real_predictions(repository, bundle)
    if predictions is None:
        if not demo:
            raise ModelUnavailableError(
                f"cannot build NEER predictions for ARGO validation ({prediction_error}); "
                "pass demo=true to run a labelled pipeline-check instead of real validation"
            )
        predictions = demo_stand_in_predictions(grid)

    if demo:
        with tempfile.TemporaryDirectory() as tmp_dir:
            demo_path = Path(tmp_dir) / "demo_argo_profiles_SYNTHETIC.csv"
            generate_demo_argo(demo_path, grid)
            profiles = load_argo(demo_path)
    else:
        source = _find_argo_source(repository.data_raw_path)
        if source is None:
            raise DataUnavailableError(
                f"no ARGO profile data found under {repository.data_raw_path} "
                "(expected argo_profiles.csv or an argo/ directory of .nc files); "
                "pass demo=true to run a labelled pipeline-check instead of real validation"
            )
        try:
            profiles = load_argo(source)
        except (FileNotFoundError, LoaderError) as exc:
            raise DataUnavailableError(f"could not load ARGO data from {source}: {exc}") from exc

    try:
        result = run_argo_validation(profiles, grid, predictions, ArgoValidationConfig())
    except ValueError as exc:
        raise InferenceFailedError(f"ARGO validation failed: {exc}") from exc

    # `result.report` may hold NaN/inf and NumPy scalars; `json_safe` is the
    # same NumPy/NaN -> JSON conversion `NEERRepository` already uses for
    # `/model/info` and `/embedding` (`src/data/preprocessing/_utils.py`).
    return json_safe(result.report)