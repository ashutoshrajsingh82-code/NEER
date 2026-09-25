"""GET /data/quality — real loaded-bundle quality report (Phase 29B-2).

Thin by design: one call into `NEERRepository.data_quality` (which reuses
`TensorBundle.summary` / `spatial_coverage` / `channel_quality` /
`target_quality`) -> the declared response model. No statistic is
invented here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import DataQualityResponse, ErrorResponse
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["data-quality"])

_ERROR_RESPONSES = {
    500: {"model": ErrorResponse, "description": "Data-quality computation failed."},
    503: {"model": ErrorResponse, "description": "Dataset not loaded."},
}


@router.get("/data/quality", response_model=DataQualityResponse, responses=_ERROR_RESPONSES)
def data_quality(
    repository: NEERRepository = Depends(get_repository),
) -> DataQualityResponse:
    """Quality report for the tensor bundle this process actually loaded.

    Coverage, per-channel observed ranges, and target completeness come
    from the bundle's own masks and arrays. No dataset is a `503`; a
    failure computing the report for an otherwise-loaded bundle is a `500`.
    """
    try:
        return DataQualityResponse(**repository.data_quality())
    except NeerApiError:
        raise
