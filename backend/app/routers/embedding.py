"""GET /embedding — the actual pooled embedding the model produced (Requirement 11)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import (
    EmbeddingQueryParams,
    EmbeddingResponse,
    ErrorResponse,
    embedding_query_params,
)
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["embedding"])

_ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Date not available."},
    500: {"model": ErrorResponse, "description": "Inference failed."},
    503: {"model": ErrorResponse, "description": "Model or dataset not loaded."},
}


@router.get("/embedding", response_model=EmbeddingResponse, responses=_ERROR_RESPONSES)
def embedding(
    params: EmbeddingQueryParams = Depends(embedding_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> EmbeddingResponse:
    """The pooled encoder embedding for one date's input sample.

    This is the same `(embed_dim,)` vector every `predict_*` call for
    this date reads off its cached forward pass (see
    `src/inference/service.py`) — never a randomly generated or
    otherwise fabricated vector.
    """
    try:
        return EmbeddingResponse(**repository.embedding(date=str(params.date)))
    except NeerApiError:
        raise