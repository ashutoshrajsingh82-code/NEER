"""
Phase 29A — dependency injection for `NEERRepository`.

The repository is built once, at application startup (see
`backend/app/main.py`'s lifespan handler), and stored on `app.state` —
loading a checkpoint and a tensor bundle on every request would be both
slow and wrong (Phase 28's `InferenceService` is itself designed to be
constructed once; see `src/inference/__init__.py`).

Tests never touch `app.state` directly: they build their own
`NEERRepository` against fixture paths and override this dependency with
`app.dependency_overrides[get_repository] = lambda: fixture_repository`,
the standard FastAPI testing pattern.
"""

from __future__ import annotations

from fastapi import Request

from backend.app.services.repository import NEERRepository


def get_repository(request: Request) -> NEERRepository:
    """Return the `NEERRepository` attached to this app instance."""
    return request.app.state.repository