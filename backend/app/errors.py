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