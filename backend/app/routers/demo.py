"""Live demonstration context, derived only from the loaded demo backend."""

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import ErrorResponse
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["demo"])


@router.get("/demo/context", responses={503: {"model": ErrorResponse, "description": "Demo dataset or model unavailable."}})
def demo_context(repository: NEERRepository = Depends(get_repository)) -> dict:
    """Return actual backend-supported demo selections or an explicit 503."""
    try:
        return repository.demo_context()
    except NeerApiError:
        raise
