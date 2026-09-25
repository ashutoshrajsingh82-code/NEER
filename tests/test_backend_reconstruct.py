"""Tests for `GET /reconstruct` (Requirement 8)."""

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

VALID_PARAMS = {"lat": 12.0, "lon": 55.0, "date": DATES[1], "depth": 100.0}


def test_reconstruct_valid_request_status_code(client):
    response = client.get("/reconstruct", params=VALID_PARAMS)
    assert response.status_code == 200


def test_reconstruct_response_schema(client):
    data = client.get("/reconstruct", params=VALID_PARAMS).json()
    assert set(data.keys()) == {
        "mode", "lat", "lon", "date", "depth", "temperature", "anomaly",
        "climatology", "embedding_dim", "data_mode", "latency_ms",
        "cache_hit", "notes",
    }
    assert data["mode"] == "point"
    assert isinstance(data["temperature"], float)
    assert data["lat"] == pytest.approx(VALID_PARAMS["lat"])
    assert data["lon"] == pytest.approx(VALID_PARAMS["lon"])
    assert data["date"] == VALID_PARAMS["date"]
    assert data["embedding_dim"] == 256


def test_reconstruct_snaps_to_nearest_model_depth(client):
    # 100 is one of the demo config's 15 depth levels exactly.
    data = client.get("/reconstruct", params=VALID_PARAMS).json()
    assert data["depth"] == 100.0


def test_reconstruct_without_climatology_notes_the_fallback(client):
    # No climatology fixture is ever written -> temperature == anomaly,
    # and the service is honest about it via `notes`.
    data = client.get("/reconstruct", params=VALID_PARAMS).json()
    assert data["climatology"] is None
    assert data["temperature"] == pytest.approx(data["anomaly"])
    assert any("climatology" in note for note in data["notes"])


def test_reconstruct_identical_requests_hit_the_cache(client):
    first = client.get("/reconstruct", params=VALID_PARAMS).json()
    second = client.get("/reconstruct", params=VALID_PARAMS).json()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


# -- missing required parameters / invalid types -> 422 --------------------


def test_reconstruct_missing_required_parameter_is_422(client):
    params = dict(VALID_PARAMS)
    del params["lat"]
    assert client.get("/reconstruct", params=params).status_code == 422


def test_reconstruct_non_numeric_lat_is_422(client):
    params = dict(VALID_PARAMS, lat="not-a-number")
    assert client.get("/reconstruct", params=params).status_code == 422


def test_reconstruct_malformed_date_is_422(client):
    params = dict(VALID_PARAMS, date="not-a-date")
    assert client.get("/reconstruct", params=params).status_code == 422


def test_reconstruct_negative_depth_is_422(client):
    params = dict(VALID_PARAMS, depth=-5.0)
    assert client.get("/reconstruct", params=params).status_code == 422


# -- invalid latitude/longitude ---------------------------------------------


def test_reconstruct_lat_outside_earth_range_is_422(client):
    params = dict(VALID_PARAMS, lat=120.0)
    assert client.get("/reconstruct", params=params).status_code == 422


def test_reconstruct_lat_outside_neer_domain_is_400(client):
    # 60 is a valid latitude on Earth but outside the NEER domain (5-30).
    params = dict(VALID_PARAMS, lat=60.0)
    response = client.get("/reconstruct", params=params)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_parameter"


def test_reconstruct_lon_outside_neer_domain_is_400(client):
    params = dict(VALID_PARAMS, lon=200.0)
    response = client.get("/reconstruct", params=params)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_parameter"


# -- unavailable dates -------------------------------------------------------


def test_reconstruct_unavailable_date_is_404(client):
    params = dict(VALID_PARAMS, date="2099-01-01")
    response = client.get("/reconstruct", params=params)
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "date_not_found"
    assert "nearest_available_dates" in body


# -- missing model / missing data -------------------------------------------


def test_reconstruct_without_model_is_503(client_no_model):
    response = client_no_model.get("/reconstruct", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_reconstruct_without_data_is_503(client_no_data):
    # `repository_no_data` has neither a checkpoint nor a tensor bundle;
    # `NEERRepository.reconstruct_point` checks the model before it
    # reads the input tensor for the date, so this is reported as
    # "model_unavailable" (still an honest 503, never a fabricated
    # reconstruction).
    response = client_no_data.get("/reconstruct", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"