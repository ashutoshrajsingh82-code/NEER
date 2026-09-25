"""GET /health — actual API/data/model status (Phase 29A, Requirement 5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.schemas import HealthResponse
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(repository: NEERRepository = Depends(get_repository)) -> HealthResponse:
    """Liveness + readiness in one call.

    Always returns `200` (the process is up and answering), but the body
    reports each component's *real* state — never a hard-coded "ok".
    `status` is `"degraded"` whenever the data or model component isn't
    actually loaded.
    """
    return HealthResponse(**repository.health())