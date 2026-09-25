"""GET /reconstruct and GET /reconstruct/grid — point and grid reconstructions
(Requirements 8/9).

Both routes are thin: gather validated query params -> one call into
`NEERRepository` (which itself defers to `InferenceService`, Phase 28)
-> reshape the resulting `PredictionResult` into the declared response
model. No forward pass, tensor assembly, or NEER-domain logic lives
here — see `backend/app/services/repository.py` and
`src/inference/service.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import (
    ErrorResponse,
    GridQueryParams,
    GridReconstructionResponse,
    PointQueryParams,
    PointReconstructionResponse,
    grid_query_params,
    point_query_params,
)
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["reconstruct"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid parameter (e.g. out-of-domain coordinate)."},
    404: {"model": ErrorResponse, "description": "Date or grid region not available."},
    500: {"model": ErrorResponse, "description": "Inference failed."},
    503: {"model": ErrorResponse, "description": "Model or dataset not loaded."},
}


def _point_response(result_dict: Dict[str, Any]) -> PointReconstructionResponse:
    return PointReconstructionResponse(
        mode=result_dict["mode"],
        lat=result_dict["lat"],
        lon=result_dict["lon"],
        date=result_dict["date"],
        depth=result_dict["depth"],
        temperature=result_dict["temperature"],
        anomaly=result_dict["anomaly"],
        climatology=result_dict["climatology"],
        embedding_dim=len(result_dict["embedding"]),
        data_mode=result_dict["data_mode"],
        latency_ms=result_dict["latency_ms"],
        cache_hit=result_dict["cache_hit"],
        notes=result_dict["notes"],
    )


@router.get("/reconstruct", response_model=PointReconstructionResponse, responses=_ERROR_RESPONSES)
def reconstruct_point(
    params: PointQueryParams = Depends(point_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> PointReconstructionResponse:
    """Reconstructed temperature at one lat/lon/date/depth.

    `lat`/`lon` type and `[-90, 90]`/`[-180, 360]` range are validated
    by FastAPI/Pydantic before this body runs (a `422`); an in-range but
    out-of-NEER-domain coordinate, a `date` absent from the loaded
    dataset, or a missing model/dataset raise the appropriate
    `NeerApiError` from `NEERRepository`, handled centrally in
    `backend/app/main.py`.
    """
    try:
        result = repository.reconstruct_point(
            lat=params.lat, lon=params.lon, date=str(params.date), depth=params.depth
        )
    except NeerApiError:
        raise
    return _point_response(result.to_dict())


def _grid_response(
    result_dict: Dict[str, Any], *, requested_depth: Optional[float]
) -> GridReconstructionResponse:
    depth: Optional[float] = None
    depths: Optional[Any] = None
    if requested_depth is not None:
        depth = result_dict["depth"]
    else:
        depths = result_dict["depth"]
    return GridReconstructionResponse(
        mode=result_dict["mode"],
        date=result_dict["date"],
        depth=depth,
        depths=depths,
        lat=result_dict["lat"],
        lon=result_dict["lon"],
        temperature=result_dict["temperature"],
        climatology=result_dict["climatology"],
        data_mode=result_dict["data_mode"],
        latency_ms=result_dict["latency_ms"],
        cache_hit=result_dict["cache_hit"],
        notes=result_dict["notes"],
    )


@router.get(
    "/reconstruct/grid", response_model=GridReconstructionResponse, responses=_ERROR_RESPONSES
)
def reconstruct_grid(
    params: GridQueryParams = Depends(grid_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> GridReconstructionResponse:
    """Reconstructed temperature over a lat/lon region, one date.

    `depth` omitted returns every model depth level
    (`(n_lat, n_lon, num_depths)`); a specific `depth` returns one
    `(n_lat, n_lon)` slice, snapped to the nearest model depth. Raises
    `400` for `lat_min >= lat_max`/`lon_min >= lon_max` or a region
    exceeding the configured cell limit, `404` when no grid cell falls
    inside the requested region or the date is unavailable, and
    `503`/`500` for a missing model or an inference failure.
    """
    try:
        result = repository.reconstruct_grid(
            lat_min=params.lat_min,
            lat_max=params.lat_max,
            lon_min=params.lon_min,
            lon_max=params.lon_max,
            date=str(params.date),
            depth=params.depth,
        )
    except NeerApiError:
        raise
    return _grid_response(result.to_dict(), requested_depth=params.depth)