"""GET /evaluation/argo — the Phase 26 ARGO validation pipeline, served
through the API (Phase 29B-1).

Thin by design (Requirement 10): gather the validated query params -> one
call into `backend.app.services.evaluation.argo_evaluation` (which defers
to `NEERRepository` and the existing `src.argo_validation` package for
everything scientific) -> return its report through the declared response
model. No matching, interpolation, metric, or NEER-domain logic lives
here — see `backend/app/services/evaluation.py` and `src/argo_validation/`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import (
    ArgoEvaluationQueryParams,
    ArgoEvaluationResponse,
    ErrorResponse,
    argo_evaluation_query_params,
)
from backend.app.services import evaluation as evaluation_service
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["evaluation"])

_ERROR_RESPONSES = {
    500: {"model": ErrorResponse, "description": "The ARGO validation pipeline failed."},
    503: {
        "model": ErrorResponse,
        "description": "Model, dataset, preprocessing metadata, or ARGO data not available.",
    },
}


@router.get(
    "/evaluation/argo", response_model=ArgoEvaluationResponse, responses=_ERROR_RESPONSES
)
def get_argo_evaluation(
    params: ArgoEvaluationQueryParams = Depends(argo_evaluation_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> ArgoEvaluationResponse:
    """Run `src.argo_validation.run_argo_validation` against the
    repository's loaded checkpoint and whatever real ARGO source data is
    found under the configured `data/raw`, and return its report.

    `demo` defaults to `false`: a real run with no checkpoint,
    preprocessing metadata, or ARGO source raises a meaningful `503`
    (Requirement 7/8) rather than fabricating a result. `demo=true` is an
    explicit opt-in to a clearly labelled `DEMO_SYNTHETIC` pipeline check
    (`validation_type`/`observational_validation`/`banner` in the response
    say so) — real predictions/ARGO data are still preferred and used
    when actually available; it is never the default and never a silent
    substitution.
    """
    try:
        report = evaluation_service.argo_evaluation(repository, demo=params.demo)
    except NeerApiError:
        raise
    return ArgoEvaluationResponse(**report)