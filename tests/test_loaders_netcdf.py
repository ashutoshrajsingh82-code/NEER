"""
Tests for `load_netcdf` (src/data/loaders/netcdf_loader.py).

xarray and a NetCDF engine are optional dependencies, so the whole module
skips when they are absent — except for the one test that asserts the
absence is reported usefully. Where they are installed, the tests write a
NetCDF from the demo dataset and read it back, including the awkward
cases real products ship with: non-canonical dimension order, aliased
coordinate names, `_FillValue` sentinels, and Kelvin temperatures.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import DataValidationError, load_netcdf
from src.data.loaders._xarray import available_netcdf_engines

xr = pytest.importorskip("xarray", reason="xarray is an optional dependency")

pytestmark = pytest.mark.skipif(
    not available_netcdf_engines(),
    reason="no NetCDF engine installed (netCDF4 / h5netcdf / scipy)",
)


@pytest.fixture
def sample_netcdf_path(demo_sample, tmp_path) -> Path:
    """The demo sample written as a conventional gridded NetCDF file."""
    path = tmp_path / "sample.nc"
    dataset = xr.Dataset(
        data_vars={
            "sst": (("time", "lat", "lon"), demo_sample.sst, {"units": "degC"}),
            "sla": (("time", "lat", "lon"), demo_sample.sla, {"units": "m"}),
        },
        coords={
            "time": demo_sample.time,
            "lat": demo_sample.lat,
            "lon": demo_sample.lon,
        },
        attrs={"data_mode": "DEMO_SYNTHETIC", "title": "NEER loader test sample"},
    )
    dataset.to_netcdf(path)
    return path


# --------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------


def test_round_trip_preserves_values(sample_netcdf_path, demo_sample):
    dataset = load_netcdf(sample_netcdf_path)
    assert np.allclose(dataset["sst"].values, demo_sample.sst, equal_nan=True)
    assert np.allclose(dataset["sla"].values, demo_sample.sla, equal_nan=True)


def test_round_trip_preserves_coordinates(sample_netcdf_path, demo_sample):
    dataset = load_netcdf(sample_netcdf_path)
    assert np.allclose(dataset.lat, demo_sample.lat)
    assert np.allclose(dataset.lon, demo_sample.lon)
    assert np.array_equal(dataset.time, demo_sample.time)


def test_round_trip_preserves_gaps(sample_netcdf_path, demo_sample):
    dataset = load_netcdf(sample_netcdf_path)
    assert np.array_equal(
        np.isnan(dataset["sst"].values), np.isnan(demo_sample.sst)
    )


def test_dims_and_units_are_canonical(sample_netcdf_path):
    dataset = load_netcdf(sample_netcdf_path)
    assert dataset["sst"].dims == ("time", "lat", "lon")
    assert dataset["sst"].units == "degC"
    assert dataset.source_format == "netcdf"


def test_global_attributes_are_preserved(sample_netcdf_path):
    dataset = load_netcdf(sample_netcdf_path)
    assert dataset.attrs["title"] == "NEER loader test sample"
    assert dataset.is_synthetic is True


def test_validation_report_is_attached(sample_netcdf_path):
    dataset = load_netcdf(sample_netcdf_path)
    assert dataset.validation is not None and dataset.validation.ok


# --------------------------------------------------------------------------
# Naming and layout quirks of real products
# --------------------------------------------------------------------------


def test_aliased_coordinate_names_are_canonicalized(demo_sample, tmp_path):
    path = tmp_path / "aliased.nc"
    xr.Dataset(
        data_vars={
            "analysed_sst": (("time", "latitude", "longitude"), demo_sample.sst, {"units": "degC"})
        },
        coords={
            "time": demo_sample.time,
            "latitude": demo_sample.lat,
            "longitude": demo_sample.lon,
        },
    ).to_netcdf(path)

    dataset = load_netcdf(path)
    assert set(dataset.coord_names) == {"time", "lat", "lon"}
    assert "sst" in dataset
    assert dataset["sst"].attrs["source_name"] == "analysed_sst"


def test_non_canonical_dimension_order_is_transposed(demo_sample, tmp_path):
    path = tmp_path / "transposed.nc"
    # (lon, lat, time) — the axis order some products actually ship.
    flipped = np.transpose(demo_sample.sst, (2, 1, 0))
    xr.Dataset(
        data_vars={"sst": (("lon", "lat", "time"), flipped, {"units": "degC"})},
        coords={"time": demo_sample.time, "lat": demo_sample.lat, "lon": demo_sample.lon},
    ).to_netcdf(path)

    dataset = load_netcdf(path)
    assert dataset["sst"].dims == ("time", "lat", "lon")
    assert np.allclose(dataset["sst"].values, demo_sample.sst, equal_nan=True)


def test_kelvin_is_converted_to_celsius(demo_sample, tmp_path):
    path = tmp_path / "kelvin.nc"
    xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), demo_sample.sst + 273.15, {"units": "K"})},
        coords={"time": demo_sample.time, "lat": demo_sample.lat, "lon": demo_sample.lon},
    ).to_netcdf(path)

    dataset = load_netcdf(path)
    assert dataset["sst"].units == "degC"
    assert dataset["sst"].attrs["source_units"] == "K"
    assert np.allclose(dataset["sst"].values, demo_sample.sst, equal_nan=True, atol=1e-3)


def test_fill_values_are_replaced_with_nan(demo_sample, tmp_path):
    path = tmp_path / "filled.nc"
    values = np.where(np.isnan(demo_sample.sst), -999.0, demo_sample.sst)
    xr.Dataset(
        data_vars={
            "sst": (
                ("time", "lat", "lon"),
                values,
                {"units": "degC", "missing_value": -999.0},
            )
        },
        coords={"time": demo_sample.time, "lat": demo_sample.lat, "lon": demo_sample.lon},
    ).to_netcdf(path)

    dataset = load_netcdf(path)
    assert np.array_equal(np.isnan(dataset["sst"].values), np.isnan(demo_sample.sst))
    # The fill-value attribute is consumed, not passed on to be applied twice.
    assert "missing_value" not in dataset["sst"].attrs


def test_rename_mapping_is_applied(demo_sample, tmp_path):
    path = tmp_path / "odd_names.nc"
    xr.Dataset(
        data_vars={"temp_surface": (("time", "lat", "lon"), demo_sample.sst, {"units": "degC"})},
        coords={"time": demo_sample.time, "lat": demo_sample.lat, "lon": demo_sample.lon},
    ).to_netcdf(path)

    dataset = load_netcdf(path, rename={"temp_surface": "sst"})
    assert "sst" in dataset


# --------------------------------------------------------------------------
# Selection, errors, writing
# --------------------------------------------------------------------------


def test_variable_selection(sample_netcdf_path):
    dataset = load_netcdf(sample_netcdf_path, variables=["sla"])
    assert dataset.variable_names == ["sla"]


def test_unknown_variable_selection_raises(sample_netcdf_path):
    with pytest.raises(KeyError, match="not found"):
        load_netcdf(sample_netcdf_path, variables=["nope"])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_netcdf(tmp_path / "absent.nc")


def test_strict_mode_rejects_warnings(demo_sample, tmp_path):
    path = tmp_path / "unregistered.nc"
    xr.Dataset(
        data_vars={"mystery_field": (("time", "lat", "lon"), demo_sample.sst, {"units": "degC"})},
        coords={"time": demo_sample.time, "lat": demo_sample.lat, "lon": demo_sample.lon},
    ).to_netcdf(path)

    assert load_netcdf(path).validation.warnings
    with pytest.raises(DataValidationError):
        load_netcdf(path, strict=True)


def test_save_and_reload_is_lossless(demo_dataset, tmp_path):
    from src.data.loaders import save_netcdf

    original = demo_dataset.subset(["sst", "land_mask"])
    path = save_netcdf(original, tmp_path / "written.nc")
    reloaded = load_netcdf(path)

    assert np.allclose(
        reloaded["sst"].values, original["sst"].values, equal_nan=True
    )
    assert np.array_equal(reloaded["land_mask"].values, original["land_mask"].values)
    assert reloaded.data_mode == original.data_mode


def test_to_xarray_round_trips_through_the_representation(demo_dataset):
    subset = demo_dataset.subset(["sla"])
    converted = subset.to_xarray()
    assert set(converted.dims) == {"time", "lat", "lon"}
    assert converted["sla"].attrs["units"] == "m"
    assert np.allclose(
        converted["sla"].values, subset["sla"].values, equal_nan=True
    )
