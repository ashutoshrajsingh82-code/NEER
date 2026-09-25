"""Tests for `GET /explainability` (Phase 29B-2)."""

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


def test_explainability_valid_request_status_code(client):
    assert client.get("/explainability", params=VALID_PARAMS).status_code == 200


def test_explainability_response_schema(client):
    data = client.get("/explainability", params=VALID_PARAMS).json()
    assert data["date"] == VALID_PARAMS["date"]
    assert data["aggregated_over_depths"] is True
    assert data["depth"] is None
    assert data["method"] == "gradient_x_input"
    assert isinstance(data["channels"], list) and data["channels"]
    assert "name" in data["channels"][0]
    assert "importance" in data["channels"][0]
    importances = [c["importance"] for c in data["channels"]]
    assert pytest.approx(sum(importances), abs=1e-5) == 1.0
    assert len(data["spatial_saliency"]) == data["spatial_saliency_shape"][0]


def test_explainability_specific_depth_is_not_aggregated(client):
    data = client.get("/explainability", params={**VALID_PARAMS, "depth": 100.0}).json()
    assert data["aggregated_over_depths"] is False
    assert data["depth"] == 100.0
    assert data["depth_index"] is not None


def test_explainability_missing_date_is_422(client):
    assert client.get("/explainability", params={}).status_code == 422


def test_explainability_negative_depth_is_422(client):
    assert client.get("/explainability", params={**VALID_PARAMS, "depth": -5.0}).status_code == 422


def test_explainability_unavailable_date_is_404(client):
    response = client.get("/explainability", params={"date": "2099-01-01"})
    assert response.status_code == 404
    assert response.json()["error"] == "date_not_found"


def test_explainability_without_model_is_503(client_no_model):
    response = client_no_model.get("/explainability", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_explainability_without_data_is_503(client_no_data):
    response = client_no_data.get("/explainability", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "data_unavailable"
