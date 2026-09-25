"""Tests for `GET /reconstruct/netcdf` (Phase 29B-2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")
pytest.importorskip("xarray", reason="xarray is required to write NetCDF")

from src.data.loaders._xarray import available_netcdf_engines  # noqa: E402

if not available_netcdf_engines():
    pytest.skip("no NetCDF engine installed (netCDF4 / h5netcdf / scipy)", allow_module_level=True)

from tests._backend_fixtures import (  # noqa: E402
    DATES,
    client,
    client_no_data,
    client_no_model,
    repository,
    repository_no_data,
    repository_no_model,
)

VALID_PARAMS = {
    "lat_min": 11.0,
    "lat_max": 19.0,
    "lon_min": 51.0,
    "lon_max": 59.0,
    "date": DATES[1],
}


def test_reconstruct_netcdf_valid_request_status_code(client):
    response = client.get("/reconstruct/netcdf", params=VALID_PARAMS)
    assert response.status_code == 200
    assert "netcdf" in response.headers.get("content-type", "").lower() or response.content[:4] in (
        b"CDF\x01",
        b"CDF\x02",
        b"\x89HDF",
    )
    assert len(response.content) > 0
    disposition = response.headers.get("content-disposition", "")
    assert "neer_reconstruct_" in disposition
    assert str(VALID_PARAMS["date"]) in disposition


def test_reconstruct_netcdf_single_depth_still_a_file(client):
    response = client.get("/reconstruct/netcdf", params={**VALID_PARAMS, "depth": 100.0})
    assert response.status_code == 200
    assert len(response.content) > 0
    assert "100m" in response.headers.get("content-disposition", "")


def test_reconstruct_netcdf_roundtrip(client, tmp_path):
    from src.data.loaders import load_netcdf

    response = client.get("/reconstruct/netcdf", params=VALID_PARAMS)
    assert response.status_code == 200

    out_file = tmp_path / "roundtrip.nc"
    out_file.write_bytes(response.content)

    dataset = load_netcdf(out_file, check_domain=False)
    assert "temperature" in dataset.variables
    temp_var = dataset.variables["temperature"]
    assert temp_var.units == "degC"
    assert temp_var.dims == ("time", "depth", "lat", "lon")
    assert "time" in dataset.coords
    assert "depth" in dataset.coords
    assert "lat" in dataset.coords
    assert "lon" in dataset.coords
    assert dataset.attrs.get("source") == "NEER /reconstruct/netcdf"
    assert dataset.attrs.get("date") == str(VALID_PARAMS["date"])
    assert "cache_hit" in dataset.attrs


def test_reconstruct_netcdf_single_depth_roundtrip(client, tmp_path):
    from src.data.loaders import load_netcdf

    response = client.get("/reconstruct/netcdf", params={**VALID_PARAMS, "depth": 100.0})
    assert response.status_code == 200

    out_file = tmp_path / "roundtrip_100m.nc"
    out_file.write_bytes(response.content)

    dataset = load_netcdf(out_file, check_domain=False)
    assert "temperature" in dataset.variables
    temp_var = dataset.variables["temperature"]
    assert temp_var.units == "degC"
    assert temp_var.dims == ("time", "lat", "lon")
    assert "time" in dataset.coords
    assert "lat" in dataset.coords
    assert "lon" in dataset.coords
    assert temp_var.attrs.get("depth_m") == 100.0


def test_reconstruct_netcdf_invalid_bounds_is_400(client):
    params = dict(VALID_PARAMS, lat_min=15.0, lat_max=15.0)
    response = client.get("/reconstruct/netcdf", params=params)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_parameter"


def test_reconstruct_netcdf_unavailable_date_is_404(client):
    params = dict(VALID_PARAMS, date="2099-01-01")
    response = client.get("/reconstruct/netcdf", params=params)
    assert response.status_code == 404
    assert response.json()["error"] == "date_not_found"


def test_reconstruct_netcdf_without_model_is_503(client_no_model):
    response = client_no_model.get("/reconstruct/netcdf", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_reconstruct_netcdf_without_data_is_503(client_no_data):
    response = client_no_data.get("/reconstruct/netcdf", params=VALID_PARAMS)
    assert response.status_code == 503
    assert response.json()["error"] == "data_unavailable"
