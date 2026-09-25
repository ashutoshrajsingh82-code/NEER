"""
Phase 29B-1 — evaluation service layer: `GET /metrics` and `GET /evaluation/argo`.

Glue only, same idiom as `backend/app/services/repository.py`: every
number in here is produced by an existing NEER module —
`src.evaluation` (Phase 24/25, reanalysis/test-target metrics) or
`src.argo_validation` (Phase 26, independent ARGO validation) — never
recomputed or approximated here. This module's job is to turn
`NEERRepository`'s already-loaded bundle/model into the *inputs* those
modules expect, call them, and turn a failure mode (no targets, no
checkpoint, no ARGO data, a scoring/pipeline exception) into the right
`NeerApiError` subclass. Routes (`backend/app/routers/metrics.py`,
`backend/app/routers/evaluation.py`) call exactly one function each
here and reshape the returned dict into the declared response model —
no NEER logic lives in the routes (Requirement 10/11/12).

`/metrics` — Phase 24/25 reanalysis/test-target evaluation
-------------------------------------------------------------
Runs the repository's already-loaded checkpoint (`NEERRepository.require_service`)
over one split's real input grid, the same way `scripts/run_evaluation.py::
evaluate_neer_checkpoint` scores NEER — reusing the loaded model instead of
reloading the checkpoint file a second time — and scores the
`(n_samples, n_depth)` output against that split's real, held-out
`TensorBundle.targets` via `src.evaluation.interface.pool_tensor_bundle`
and `src.evaluation.metrics.compute_profile_metrics`. A dataset with no
targets, no checkpoint, or an empty split is a real `503`/`400` — never a
fabricated number.

`/evaluation/argo` — Phase 26 independent ARGO validation
--------------------------------------------------------------
Runs the exact pipeline `scripts/run_argo_validation.py` runs
(`src.argo_validation.run_argo_validation`), sourcing NEER predictions
from the repository's checkpoint (`predictions_from_checkpoint`) and ARGO
profiles from whatever is found under `NEERRepository.data_raw_path`
(`_discover_argo_source` below — the API has no `--argo`/`--neer-checkpoint`
flags to pass by hand, so this is the auto-discovery a live service needs
in their place). `demo=true` is an explicit, never-default opt-in to a
clearly labelled `DEMO_SYNTHETIC` pipeline check (real predictions/ARGO
data are still preferred and used when they are actually available; the
labelled synthetic stand-ins from `src.argo_validation.demo` only fill in
what demo mode couldn't otherwise get) — same "demo data is never
validation" rule `src/argo_validation/pipeline.py` documents. A real
(non-demo) run with no ARGO source anywhere under `data_raw_path` is a
meaningful `503`, never a demo substitution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.app.errors import (
    ArgoDataUnavailableError,
    DataUnavailableError,
    EvaluationFailedError,
    InvalidParameterError,
    ModelUnavailableError,
)
from backend.app.services.repository import NEERRepository
from src.argo_validation import (
    DEFAULT_DEMO_FILENAME,
    ArgoValidationConfig,
    NeerGrid,
    NeerPredictions,
    demo_stand_in_predictions,
    generate_demo_argo,
    load_argo,
    predictions_from_checkpoint,
    run_argo_validation,
)
from src.argo_validation.pipeline import _sanitize as _sanitize_report
from src.argo_validation.profiles import ArgoProfileSet
from src.data.loaders.errors import LoaderError, MissingDependencyError
from src.evaluation.interface import SPLIT_NAMES, pool_tensor_bundle
from src.evaluation.metrics import compute_profile_metrics
from src.inference.service import InferenceService

#: Conventional filenames this service looks for under `NEERRepository.data_raw_path`
#: (matching the paths `README.md`'s `run_argo_validation.py` examples use:
#: `data/raw/argo_profiles.csv`, `data/raw/argo_nc/`). `_discover_argo_source`
#: falls back to any `.csv`/`.nc` actually found there so a differently-named
#: real file is still picked up, without the API needing a `--argo`-style
#: parameter the way the CLI script has one.
_ARGO_CSV_NAME = "argo_profiles.csv"
_ARGO_NC_DIRNAME = "argo_nc"


# --------------------------------------------------------------------------
# /metrics
# --------------------------------------------------------------------------


def _predict_pooled_profiles(
    service: InferenceService, inputs: np.ndarray, *, batch_size: int = 8
) -> np.ndarray:
    """Run `service`'s already-loaded `NEERModel` over `inputs`
    (`(n_samples, n_channels, n_lat, n_lon)`), batched, and return its
    `(n_samples, n_depth)` pooled-profile output.

    Same forward-pass loop `scripts/run_evaluation.py::evaluate_neer_checkpoint`
    and `src.argo_validation.predictions.predictions_from_checkpoint` use,
    against the model `NEERRepository` already loaded — no re-parsing the
    checkpoint file a second time.
    """
    import torch

    n = int(inputs.shape[0])
    if n == 0:
        return np.empty((0, len(service.depths)), dtype=float)

    model = service.model
    outputs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = np.asarray(inputs[start : start + batch_size], dtype=np.float32)
            x = torch.from_numpy(batch).to(service.device)
            outputs.append(model(x).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(float)


def metrics(repository: NEERRepository, *, split: str) -> Dict[str, Any]:
    """Real NEER evaluation metrics for one data split (Requirement 3/4/6).

    Raises `InvalidParameterError` for a `split` outside
    `src.evaluation.interface.SPLIT_NAMES`; `DataUnavailableError` for a
    dataset with no targets, no such split, or an empty split;
    `ModelUnavailableError` for no loaded checkpoint; `EvaluationFailedError`
    if scoring itself raises or the model's output shape doesn't match the
    target it's being compared to.
    """
    if split not in SPLIT_NAMES:
        raise InvalidParameterError(
            f"split must be one of {list(SPLIT_NAMES)}, got {split!r}"
        )

    bundle = repository.require_bundle()
    if bundle.targets is None or bundle.target_mask is None:
        raise DataUnavailableError(
            "the loaded dataset has no targets; cannot compute evaluation metrics "
            "(run 'python scripts/preprocess_data.py' against a source with a target variable)"
        )
    if split not in bundle.split_masks:
        raise DataUnavailableError(
            f"the loaded dataset has no '{split}' split "
            f"(available: {sorted(bundle.split_masks) or 'none'})"
        )

    try:
        pooled = pool_tensor_bundle(
            bundle, splits=(split,), source_tensors_path=repository.tensor_path
        )[split]
    except (KeyError, ValueError) as exc:
        raise DataUnavailableError(str(exc)) from exc

    if pooled.n_samples == 0:
        raise DataUnavailableError(f"the '{split}' split has no samples to score")

    service = repository.require_service()
    mask = np.asarray(bundle.split_masks[split], dtype=bool)
    try:
        prediction = _predict_pooled_profiles(service, bundle.inputs[mask])
    except (ValueError, RuntimeError) as exc:
        raise EvaluationFailedError(f"metrics computation failed: {exc}") from exc

    if prediction.shape != pooled.profile.shape:
        raise EvaluationFailedError(
            f"model output shape {prediction.shape} does not match the "
            f"'{split}' split's target shape {pooled.profile.shape}"
        )

    profile_metrics = compute_profile_metrics(
        pooled.profile,
        prediction,
        pooled.profile_mask,
        depth_names=pooled.depth_names,
        variable_names=pooled.variable_names or None,
    )

    return {
        "phase": 25,
        "split": split,
        "n_samples": pooled.n_samples,
        "is_synthetic": bool(bundle.is_synthetic),
        "data_mode": str(bundle.attrs.get("data_mode", "UNKNOWN")),
        "disclaimer": bundle.attrs.get("disclaimer"),
        "source_tensors_path": str(repository.tensor_path),
        "checkpoint": str(repository.checkpoint_path) if repository.checkpoint_path else None,
        "metrics": profile_metrics.to_dict(),
    }


# --------------------------------------------------------------------------
# /evaluation/argo
# --------------------------------------------------------------------------


def _discover_argo_source(data_raw_path: Path) -> Optional[Path]:
    """Find a real ARGO source under `data_raw_path`, or `None`.

    Tries the conventional names `README.md`'s `run_argo_validation.py`
    examples use first (`argo_profiles.csv`, an `argo_nc/` directory of
    GDAC files), then falls back to any `.csv`/`.nc` file actually present
    directly under `data_raw_path` — this service has no `--argo` flag a
    caller can point at a specific file, unlike the CLI script.
    """
    data_raw_path = Path(data_raw_path)
    if not data_raw_path.is_dir():
        return None

    conventional_csv = data_raw_path / _ARGO_CSV_NAME
    if conventional_csv.is_file():
        return conventional_csv

    conventional_nc_dir = data_raw_path / _ARGO_NC_DIRNAME
    if conventional_nc_dir.is_dir() and any(conventional_nc_dir.glob("*.nc")):
        return conventional_nc_dir

    csvs = sorted(data_raw_path.glob("*.csv"))
    if csvs:
        return csvs[0]

    if any(data_raw_path.glob("*.nc")):
        return data_raw_path

    return None


def _generate_demo_argo_profiles(grid: NeerGrid) -> ArgoProfileSet:
    """A freshly-generated, clearly-labelled `DEMO_SYNTHETIC` profile set
    covering `grid`'s domain/time range (`src.argo_validation.demo`),
    written to a private temp directory (never the real `data/` tree —
    concurrent requests each get their own) and loaded back through the
    same `load_argo` a real file goes through."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="neer_argo_demo_"))
    demo_path = generate_demo_argo(tmp_dir / DEFAULT_DEMO_FILENAME, grid)
    return load_argo(demo_path)


def _real_predictions(repository: NEERRepository, bundle) -> NeerPredictions:
    """NEER predictions from the repository's loaded checkpoint, for a
    real (non-demo) run. Requires both a checkpoint and preprocessing
    metadata (to convert the model's z-scored output back to degC)."""
    checkpoint_path = repository.checkpoint_path
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise ModelUnavailableError(repository.model_error())

    metadata_path = repository.metadata_path
    if not Path(metadata_path).exists():
        raise DataUnavailableError(
            f"no preprocessing metadata at {metadata_path}; required to convert NEER's "
            "output back to degC for ARGO validation "
            "(run 'python scripts/preprocess_data.py', which writes it alongside the tensors)"
        )

    try:
        return predictions_from_checkpoint(checkpoint_path, bundle, metadata_path)
    except MissingDependencyError as exc:
        raise ModelUnavailableError(str(exc)) from exc
    except (ValueError, RuntimeError, OSError) as exc:
        raise EvaluationFailedError(f"could not obtain NEER predictions: {exc}") from exc


def _real_argo_profiles(repository: NEERRepository) -> ArgoProfileSet:
    """Real ARGO profiles from `NEERRepository.data_raw_path`, for a real
    (non-demo) run. No source found or found-but-unreadable is a real
    `503 argo_data_unavailable` — never a demo substitution."""
    source = _discover_argo_source(repository.data_raw_path)
    if source is None:
        raise ArgoDataUnavailableError(
            f"no ARGO data found under {repository.data_raw_path} "
            "(expected e.g. 'argo_profiles.csv', a long-format CSV, or a directory of GDAC "
            "*_prof.nc files); pass demo=true for a labelled pipeline check instead, or add "
            "real ARGO data there"
        )
    try:
        return load_argo(source)
    except (LoaderError, OSError) as exc:
        raise ArgoDataUnavailableError(f"could not load ARGO data from {source}: {exc}") from exc


def _demo_predictions(repository: NEERRepository, bundle, grid: NeerGrid) -> NeerPredictions:
    """Predictions for a `demo=true` run: a real checkpoint prediction is
    tried first when both a checkpoint and metadata are actually present
    (real numbers preferred over a stand-in whenever they're available),
    falling back to the labelled `demo_stand_in_NOT_NEER` only when that
    isn't possible — never raises."""
    checkpoint_path = repository.checkpoint_path
    metadata_path = repository.metadata_path
    if (
        checkpoint_path is not None
        and Path(checkpoint_path).exists()
        and Path(metadata_path).exists()
    ):
        try:
            return predictions_from_checkpoint(checkpoint_path, bundle, metadata_path)
        except (MissingDependencyError, ValueError, RuntimeError, OSError):
            pass
    return demo_stand_in_predictions(grid)


def _demo_argo_profiles(repository: NEERRepository, grid: NeerGrid) -> ArgoProfileSet:
    """ARGO profiles for a `demo=true` run: a real source under
    `data_raw_path` is tried first, falling back to a freshly-generated,
    clearly-labelled `DEMO_SYNTHETIC` set only when none is found/readable
    — never raises."""
    source = _discover_argo_source(repository.data_raw_path)
    if source is not None:
        try:
            return load_argo(source)
        except (LoaderError, OSError):
            pass
    return _generate_demo_argo_profiles(grid)


def argo_evaluation(repository: NEERRepository, *, demo: bool) -> Dict[str, Any]:
    """Run the Phase 26 ARGO validation pipeline and return its report
    (Requirement 5/6/7/8). `demo=False` (the default) is a real
    validation attempt: a missing checkpoint/metadata or ARGO source
    raises the appropriate `NeerApiError` rather than substituting demo
    data. `demo=True` is an explicit opt-in to a labelled
    `DEMO_SYNTHETIC` pipeline check — see this module's docstring.
    """
    bundle = repository.require_bundle()
    try:
        grid = NeerGrid.from_bundle(bundle)
    except ValueError as exc:
        raise DataUnavailableError(str(exc)) from exc

    if demo:
        predictions = _demo_predictions(repository, bundle, grid)
        profiles = _demo_argo_profiles(repository, grid)
    else:
        predictions = _real_predictions(repository, bundle)
        profiles = _real_argo_profiles(repository)

    try:
        result = run_argo_validation(profiles, grid, predictions, ArgoValidationConfig())
    except ValueError as exc:
        raise EvaluationFailedError(f"ARGO validation pipeline failed: {exc}") from exc

    return _sanitize_report(result.report)