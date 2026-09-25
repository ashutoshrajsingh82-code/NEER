"""GET /dates — dates that actually exist in the loaded tensor bundle (Requirement 7)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import DatesResponse, ErrorResponse
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["dates"])


@router.get(
    "/dates",
    response_model=DatesResponse,
    responses={503: {"model": ErrorResponse, "description": "No dataset is loaded."}},
)
def list_dates(repository: NEERRepository = Depends(get_repository)) -> DatesResponse:
    """Every date available in the loaded NEER tensor bundle.

    Raises `503` (`DataUnavailableError`) when no tensor bundle is
    loaded — the dataset actually on disk is the only source of truth
    here, never a fabricated or hard-coded date range.
    """
    try:
        return DatesResponse(**repository.list_dates())
    except NeerApiError:
        raise