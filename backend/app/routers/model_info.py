"""GET /model/info — real architecture/config/checkpoint info (Requirement 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import ErrorResponse, ModelInfoResponse
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["model"])


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse, "description": "No model checkpoint is loaded."}},
)
def model_info(repository: NEERRepository = Depends(get_repository)) -> ModelInfoResponse:
    """Architecture, configuration and checkpoint metadata for the loaded model.

    Raises `503` (via `ModelUnavailableError`, handled in
    `backend/app/main.py`) when no checkpoint is loaded — this never
    fabricates a description of a model that isn't actually in memory.
    """
    try:
        return ModelInfoResponse(**repository.model_info())
    except NeerApiError:
        raise