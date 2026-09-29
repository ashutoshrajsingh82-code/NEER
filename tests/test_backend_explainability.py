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

VALID_PARAMS = {"date": DATES[1], "lat": 15.0, "lon": 55.0, "depth": 100.0}


def test_explainability_valid_request_status_code(client):
    assert client.get("/explainability", params=VALID_PARAMS).status_code == 200


def test_explainability_response_schema(client):
    data = client.get("/explainability", params=VALID_PARAMS).json()
    assert data["date"] == VALID_PARAMS["date"]
    assert data["lat"] == VALID_PARAMS["lat"]
    assert data["lon"] == VALID_PARAMS["lon"]
    assert data["depth"] == VALID_PARAMS["depth"]
    assert data["method"] == "Integrated Gradients"
    assert data["baseline"]
    assert data["feature_order"] == ["sst", "sss", "sla", "u_current", "v_current", "u_wind", "v_wind"]
    assert [item["name"] for item in data["features"]] == data["feature_order"]
    assert len(data["features"]) == 7
    assert all(isinstance(item["attribution"], (int, float)) for item in data["features"])
    assert data["attribution_sum"] == pytest.approx(
        data["output_delta_from_baseline"] + data["completeness_error"], abs=1e-5
    )
    assert isinstance(data["temperature"], float)
    assert data["climatology"] is None
    assert data["anomaly"] is None


def test_explainability_depth_and_location_are_echoed(client):
    data = client.get("/explainability", params={**VALID_PARAMS, "lat": 12.3, "lon": 54.6}).json()
    assert data["depth"] == 100.0
    assert data["depth_index"] == 7
    assert data["lat"] == 12.3
    assert data["lon"] == 54.6


def test_explainability_missing_date_is_422(client):
    assert client.get("/explainability", params={}).status_code == 422


def test_explainability_negative_depth_is_422(client):
    assert client.get("/explainability", params={**VALID_PARAMS, "depth": -5.0}).status_code == 422


def test_explainability_requires_location_and_depth(client):
    assert client.get("/explainability", params={"date": DATES[1]}).status_code == 422


def test_explainability_rejects_unsupported_depth(client):
    response = client.get("/explainability", params={**VALID_PARAMS, "depth": 95.0})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_parameter"


def test_explainability_unavailable_date_is_404(client):
    response = client.get("/explainability", params={**VALID_PARAMS, "date": "2099-01-01"})
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
