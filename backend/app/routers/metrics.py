"""GET /metrics — real NEER evaluation metrics for a data split (Phase 29B-1).

Thin by design (Requirement 10): gather the validated query params -> one
call into `backend.app.services.evaluation.metrics` (which defers to
`NEERRepository` and the existing `src.evaluation` module for everything
scientific) -> reshape the returned dict into the declared response
model. No metric computation, model forward pass, or NEER-domain logic
lives here — see `backend/app/services/evaluation.py`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import (
    ErrorResponse,
    MetricsQueryParams,
    MetricsResponse,
    metrics_query_params,
)
from backend.app.services import evaluation as evaluation_service
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["evaluation"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid split name."},
    500: {"model": ErrorResponse, "description": "Metrics computation failed."},
    503: {"model": ErrorResponse, "description": "Model, dataset, or targets not available."},
}


@router.get("/metrics", response_model=MetricsResponse, responses=_ERROR_RESPONSES)
def get_metrics(
    params: MetricsQueryParams = Depends(metrics_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> MetricsResponse:
    """Real evaluation metrics (RMSE/MAE/bias/Pearson-r/R^2, overall +
    per-depth [+ per-variable]) for one data split.

    Computed by running the repository's loaded checkpoint over that
    split's real input grid and scoring its output against the split's
    real, held-out targets via `src.evaluation.metrics.compute_profile_metrics`
    — never a fabricated, random, or placeholder value (Requirement 3/4).
    An unrecognized `split` is a `400`; a dataset with no targets/that
    split, or no loaded model, is a `503`; a scoring failure is a `500`.
    """
    try:
        result = evaluation_service.metrics(repository, split=params.split)
    except NeerApiError:
        raise
    return MetricsResponse(**result)