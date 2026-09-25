"""GET /profile — a full reconstructed vertical temperature profile (Requirement 10)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import (
    ErrorResponse,
    ProfileQueryParams,
    ProfileReconstructionResponse,
    profile_query_params,
)
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["profile"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid parameter (e.g. out-of-domain coordinate)."},
    404: {"model": ErrorResponse, "description": "Date not available."},
    500: {"model": ErrorResponse, "description": "Inference failed."},
    503: {"model": ErrorResponse, "description": "Model or dataset not loaded."},
}


@router.get("/profile", response_model=ProfileReconstructionResponse, responses=_ERROR_RESPONSES)
def reconstruct_profile(
    params: ProfileQueryParams = Depends(profile_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> ProfileReconstructionResponse:
    """The full depth-temperature profile at one lat/lon/date.

    Every model depth level is returned (`self.depths` on the loaded
    `InferenceService`/`NEERModel`) — this never interpolates or
    fabricates intermediate depths.
    """
    try:
        result = repository.reconstruct_profile(lat=params.lat, lon=params.lon, date=str(params.date))
    except NeerApiError:
        raise
    d = result.to_dict()
    return ProfileReconstructionResponse(
        mode=d["mode"],
        lat=d["lat"],
        lon=d["lon"],
        date=d["date"],
        depths=d["depth"],
        temperature=d["temperature"],
        anomaly=d["anomaly"],
        climatology=d["climatology"],
        embedding_dim=len(d["embedding"]),
        data_mode=d["data_mode"],
        latency_ms=d["latency_ms"],
        cache_hit=d["cache_hit"],
        notes=d["notes"],
    )