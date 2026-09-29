"""CORS policy tests for the public FastAPI application."""

from fastapi.testclient import TestClient

from backend.app.main import CORS_ORIGINS, app


def test_cors_configuration_uses_explicit_origins():
    assert CORS_ORIGINS
    assert "*" not in CORS_ORIGINS
    middleware = next(item for item in app.user_middleware if item.cls.__name__ == "CORSMiddleware")
    assert middleware.kwargs["allow_origins"] == CORS_ORIGINS


def test_cors_preflight_allows_local_frontend_and_rejects_unlisted_origin():
    allowed_origin = CORS_ORIGINS[0]
    with TestClient(app) as client:
        allowed = client.options(
            "/health",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        rejected = client.options(
            "/health",
            headers={
                "Origin": "https://unlisted.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == allowed_origin
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers
