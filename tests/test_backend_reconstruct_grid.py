"""Tests for `GET /reconstruct/grid` (Requirement 9)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from tests._backend_fixtures import (  # noqa: E402
    DATES,
    client,
    client_low_grid_limit,
    client_no_data,
    client_no_model,
    repository,
    repository_no_data,
    repository_no_model,
)

# Well inside the fixture's actual lat=[10,20]/lon=[50,60] coverage.
VALID_PARAMS = {"lat_min": 11.0, "lat_max": 19.0, "lon_min": 51.0, "lon_max": 59.0, "date": DATES[1]}


def test_reconstruct_grid_valid_request_status_code(client):
    response = client.get("/reconstruct/grid", params=VALID_PARAMS)
    assert response.status_code == 200


def test_reconstruct_grid_all_depths_response_schema(client):
    data = client.get("/reconstruct/grid", params=VALID_PARAMS).json()
    assert data["mode"] == "grid"
    assert data["depth"] is None
    assert data["depths"] is not None and len(data["depths"]) == 15
    assert isinstance(data["lat"], list) and isinstance(data["lon"], list)
    # (n_lat, n_lon, num_depths) when depth is omitted.
    assert len(data["temperature"]) == len(data["lat"])
    assert len(data["temperature"][0]) == len(data["lon"])
    assert len(data["temperature"][0][0]) == 15


def test_reconstruct_grid_single_depth_response_shape(client):
    params = dict(VALID_PARAMS, depth=100.0)
    data = client.get("/reconstruct/grid", params=params).json()
    assert data["depth"] == 100.0
    assert data["depths"] is None
    # (n_lat, n_lon) when a single depth is given.
    assert len(data["temperature"]) == len(data["lat"])
    assert isinstance(data["temperature"][0], list)
    assert isinstance(data["temperature"][0][0], (int, float))


def test_reconstruct_grid_returned_lat_lon_are_within_the_requested_region(client):
    data = client.get("/reconstruct/grid", params=VALID_PARAMS).json()
    assert all(VALID_PARAMS["lat_min"] <= lat <= VALID_PARAMS["lat_max"] for lat in data["lat"])
    assert all(VALID_PARAMS["lon_min"] <= lon <= VALID_PARAMS["lon_max"] for lon in data["lon"])


# -- missing/invalid parameters ----------------------------------------------


def test_reconstruct_grid_missing_required_parameter_is_422(client):
    params = dict(VALID_PARAMS)
    del params["lat_min"]
    assert client.get("/reconstruct/grid", params=params).status_code == 422


def test_reconstruct_grid_non_numeric_bound_is_422(client):
    params = dict(VALID_PARAMS, lat_min="north")
    assert client.get("/reconstruct/grid", params=params).status_code == 422


def test_reconstruct_grid_lat_min_gte_lat_max_is_400(client):
    params = dict(VALID_PARAMS, lat_min=15.0, lat_max=15.0)
    response = client.get("/reconstruct/grid", params=params)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_parameter"


def test_reconstruct_grid_lon_min_gte_lon_max_is_400(client):
    params = dict(VALID_PARAMS, lon_min=55.0, lon_max=52.0)
    response = client.get("/reconstruct/grid", params=params)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_parameter"


def test_reconstruct_grid_out_of_domain_bound_is_400(client):
    # 200 is a valid longitude on Earth (<=360) but outside the NEER
    # domain (45-105).
    params = dict(VALID_PARAMS, lon_max=200.0)
    response = client.get("/reconstruct/grid", params=params)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_parameter"


def test_reconstruct_grid_negative_depth_is_422(client):
    params = dict(VALID_PARAMS, depth=-1.0)
    assert client.get("/reconstruct/grid", params=params).status_code == 422


def test_reconstruct_grid_too_many_cells_is_400(client_low_grid_limit):
    response = client_low_grid_limit.get("/reconstruct/grid", params=VALID_PARAMS)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_parameter"


# -- unavailable grid cells / dates ------------------------------------------


def test_reconstruct_grid_region_outside_loaded_cells_is_404(client):
    # Inside the NEER domain (5-30/45-105) but outside the fixture's
    # actual lat/lon coverage (10-20/50-60) -> no cells to serve.
    params = dict(VALID_PARAMS, lat_min=25.0, lat_max=29.0, lon_min=95.0, lon_max=100.0)
    response = client.get("/reconstruct/grid", params=params)
    assert response.status_code == 404
    assert response.json()["error"] == "grid_unavailable"


def test_reconstruct_grid_unavailable_date_is_404(client):
    params = dict(VALID_PARAMS, date="2099-01-01")
    response = client.get("/reconstruct/grid", params=params)
    assert response.status_code == 404
    assert response.json()["error"] == "date_not_found"


# -- missing model / missing data --------------------------------------------


def test_reconstruct_grid_without_model_is_503(client_no_model):
    response = client_no_model.get("/reconstruct/grid", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_reconstruct_grid_without_data_is_404_or_503(client_no_data):
    # `_grid_slice` (called before the model check) raises
    # `DataUnavailableError` itself when there is no bundle to slice.
    response = client_no_data.get("/reconstruct/grid", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "data_unavailable"