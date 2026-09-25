"""GET /explainability — real gradient-based attribution (Phase 29B-2).

Thin by design: gather validated query params -> one call into
`NEERRepository.explain` (which runs `src.explainability.gradients.explain_point`
on the actual loaded model and the actual input for `date`) -> reshape
the returned dict into the declared response model. No backward pass or
NEER-domain logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import (
    ErrorResponse,
    ExplainabilityQueryParams,
    ExplainabilityResponse,
    explainability_query_params,
)
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["explainability"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid parameter (e.g. negative depth)."},
    404: {"model": ErrorResponse, "description": "Date not available."},
    500: {"model": ErrorResponse, "description": "Explainability computation failed."},
    503: {"model": ErrorResponse, "description": "Model or dataset not loaded."},
}


@router.get(
    "/explainability", response_model=ExplainabilityResponse, responses=_ERROR_RESPONSES
)
def explainability(
    params: ExplainabilityQueryParams = Depends(explainability_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> ExplainabilityResponse:
    """Gradient-x-input attribution for one date's real input sample.

    `depth` omitted explains the sum of every model depth level's
    predicted anomaly; a specific `depth` is snapped to the nearest
    model depth, same convention as `/reconstruct`. Every number comes
    from a real backward pass — never a fabricated or placeholder score.
    """
    try:
        result = repository.explain(date=str(params.date), depth=params.depth)
    except NeerApiError:
        raise
    return ExplainabilityResponse(**result)
