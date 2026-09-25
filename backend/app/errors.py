"""
Phase 29A — API-level error types.

Every failure mode the API needs to report as a *meaningful* HTTP error
(missing model, missing data, an out-of-domain coordinate, a date that
does not exist in the dataset, an inference failure, ...) is one of
these, so routes never have to construct `HTTPException` ad hoc and the
error body shape is consistent everywhere.

These wrap failures from the underlying NEER modules (`LoaderError`,
`MissingDependencyError`, `StepConfigurationError`, ...) rather than
duplicating their logic — the repository layer (`backend/app/services`)
catches those and re-raises the appropriate `NeerApiError` subclass.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class NeerApiError(Exception):
    """Base class for every error the API raises deliberately.

    `status_code` is the HTTP status the FastAPI exception handler
    (registered in `backend/app/main.py`) responds with. `error_code` is
    a short, stable, machine-readable identifier a client can branch on
    without parsing `detail`'s prose.
    """

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, detail: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self.detail = detail
        self.extra = extra or {}
        super().__init__(detail)

    def body(self) -> Dict[str, Any]:
        return {"error": self.error_code, "detail": self.detail, **self.extra}


class InvalidParameterError(NeerApiError):
    """A request parameter is present and well-typed but semantically invalid.

    (Missing parameters and outright type errors — a non-numeric `lat`,
    a missing required query arg — are handled by FastAPI/Pydantic's own
    422 validation before a route body ever runs. This is for
    range/logic problems a `Query(...)` bound can't express, such as a
    coordinate outside the NEER domain or `lat_min >= lat_max`.)
    """

    status_code = 400
    error_code = "invalid_parameter"


class DateNotFoundError(NeerApiError):
    """The requested date is not one of the dataset's available timesteps."""

    status_code = 404
    error_code = "date_not_found"


class GridUnavailableError(NeerApiError):
    """The requested grid region/cells cannot be served."""

    status_code = 404
    error_code = "grid_unavailable"


class ModelUnavailableError(NeerApiError):
    """No usable model checkpoint is loaded."""

    status_code = 503
    error_code = "model_unavailable"


class DataUnavailableError(NeerApiError):
    """No usable input dataset/tensor bundle is loaded."""

    status_code = 503
    error_code = "data_unavailable"


class InferenceFailedError(NeerApiError):
    """The model/service raised while actually running inference."""

    status_code = 500
    error_code = "inference_failed"


class EvaluationFailedError(NeerApiError):
    """Evaluation/scoring itself raised (not a missing input) — e.g. the
    loaded checkpoint's output shape doesn't match the target it is being
    scored against, or the ARGO validation pipeline raised while running.

    Phase 29B-1 (`/metrics`, `/evaluation/argo`): distinct from
    `InferenceFailedError`, which is a single reconstruction call failing,
    not a multi-sample evaluation/validation run.
    """

    status_code = 500
    error_code = "evaluation_failed"


class ArgoDataUnavailableError(NeerApiError):
    """No usable ARGO source data (CSV/NetCDF) was found for `/evaluation/argo`.

    Phase 29B-1, Requirement 7/8: a real (non-demo) ARGO validation run
    with nothing under the configured `data/raw` is a meaningful `503`,
    never a fabricated or demo-substituted result.
    """

    status_code = 503
    error_code = "argo_data_unavailable"


class ExplainabilityFailedError(NeerApiError):
    """The explainability computation itself raised (not a missing model
    or dataset) — e.g. the gradient computation failed for an
    otherwise-loaded model/input.

    Phase 29B-2, `/explainability`: distinct from `ModelUnavailableError`/
    `DataUnavailableError`, which mean nothing was there to explain in
    the first place.
    """

    status_code = 500
    error_code = "explainability_failed"


class DataQualityFailedError(NeerApiError):
    """Data-quality computation itself raised for an otherwise-loaded
    dataset (Phase 29B-2, `/data/quality`) — never a fabricated statistic
    in its place.
    """

    status_code = 500
    error_code = "data_quality_failed"


class NetCDFUnavailableError(NeerApiError):
    """NetCDF output cannot be produced in this environment — `xarray`
    and/or a NetCDF engine (`netCDF4`/`h5netcdf`) is not installed.

    Phase 29B-2, `/reconstruct/netcdf`: an environment/dependency gap,
    same "service degraded" meaning as `ModelUnavailableError`/
    `DataUnavailableError`, not a computation failure.
    """

    status_code = 503
    error_code = "netcdf_unavailable"


class NetCDFGenerationFailedError(NeerApiError):
    """NetCDF assembly/writing itself raised for an otherwise-available
    pipeline (Phase 29B-2, `/reconstruct/netcdf`). Never a fabricated or
    empty file in its place.
    """

    status_code = 500
    error_code = "netcdf_generation_failed"