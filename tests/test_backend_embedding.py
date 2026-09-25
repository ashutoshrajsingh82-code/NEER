"""Tests for `GET /embedding` (Requirement 11)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from tests._backend_fixtures import (  # noqa: E402
    DATES,
    client,
    client_no_data,
    client_no_model,
    repository,
    repository_no_data,
    repository_no_model,
)

VALID_PARAMS = {"date": DATES[1]}


def test_embedding_valid_request_status_code(client):
    assert client.get("/embedding", params=VALID_PARAMS).status_code == 200


def test_embedding_response_schema_and_shape(client):
    data = client.get("/embedding", params=VALID_PARAMS).json()
    assert set(data.keys()) == {"date", "embedding", "dim", "data_mode", "cache_hit", "latency_ms"}
    assert data["date"] == VALID_PARAMS["date"]
    assert data["dim"] == 256
    assert len(data["embedding"]) == 256
    assert all(isinstance(v, float) for v in data["embedding"])


def test_embedding_identical_requests_hit_the_cache(client):
    first = client.get("/embedding", params=VALID_PARAMS).json()
    second = client.get("/embedding", params=VALID_PARAMS).json()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["embedding"] == pytest.approx(second["embedding"])


def test_embedding_different_dates_give_different_embeddings(client):
    first = client.get("/embedding", params={"date": DATES[0]}).json()
    second = client.get("/embedding", params={"date": DATES[1]}).json()
    assert first["embedding"] != pytest.approx(second["embedding"])


# -- missing/invalid parameters -----------------------------------------


def test_embedding_missing_required_parameter_is_422(client):
    assert client.get("/embedding", params={}).status_code == 422


def test_embedding_malformed_date_is_422(client):
    assert client.get("/embedding", params={"date": "not-a-date"}).status_code == 422


def test_embedding_unavailable_date_is_404(client):
    response = client.get("/embedding", params={"date": "2099-01-01"})
    assert response.status_code == 404
    assert response.json()["error"] == "date_not_found"


# -- missing model / missing data -----------------------------------------


def test_embedding_without_model_is_503(client_no_model):
    response = client_no_model.get("/embedding", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_embedding_without_data_is_503(client_no_data):
    response = client_no_data.get("/embedding", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"