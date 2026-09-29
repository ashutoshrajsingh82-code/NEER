"""GET /explainability — actual Integrated Gradients (Phase 44).

Thin by design: gather validated query params -> one call into
NEERRepository.explain runs Integrated Gradients on the actual normalized
model input and returns point prediction context with signed attributions.
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
    """Integrated Gradients for the exact requested date/location/depth."""
    try:
        result = repository.explain(
            date=str(params.date), lat=params.lat, lon=params.lon, depth=params.depth
        )
    except NeerApiError:
        raise
    return ExplainabilityResponse(**result)
