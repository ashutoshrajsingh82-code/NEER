"""
Tests for `load_demo_dataset` (src/data/loaders/demo_loader.py).

These exercise the loading layer against the real artifact it is meant to
serve during development: the synthetic dataset in `data/demo/`. They
check the standardization contract (canonical coords, canonical units,
NaN for gaps), the validation pass, and the provenance guarantees that
keep synthetic data from being mistaken for observations.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import (
    DEMO_DATA_MODE,
    OceanDataset,
    SchemaError,
    load_demo_dataset,
    load_npz,
)
from src.utils.config import load_config

CONFIG = load_config("demo")
EXPECTED_N_LAT = 101
EXPECTED_N_LON = 241
EXPECTED_SURFACE_VARS = {"sst", "sss", "sla", "u_current", "v_current", "u_wind", "v_wind"}


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_returns_ocean_dataset(demo_dataset):
    assert isinstance(demo_dataset, OceanDataset)
    assert demo_dataset.source_format.startswith("npz")


def test_contains_every_variable_from_metadata(demo_dataset, demo_metadata):
    assert set(demo_dataset.variable_names) == set(demo_metadata["variables"])


def test_surface_variables_are_time_lat_lon(demo_dataset):
    for name in EXPECTED_SURFACE_VARS:
        assert demo_dataset[name].dims == ("time", "lat", "lon")


def test_subsurface_variable_includes_depth(demo_dataset):
    assert demo_dataset["subsurface_temp"].dims == ("time", "depth", "lat", "lon")


def test_land_mask_is_static_and_boolean(demo_dataset):
    land_mask = demo_dataset["land_mask"]
    assert land_mask.dims == ("lat", "lon")
    assert land_mask.dtype == bool
    assert land_mask.is_mask
    # Both land and ocean must be present, or the mask is meaningless.
    assert 0 < land_mask.values.mean() < 1


def test_variable_shapes_match_coordinates(demo_dataset, demo_metadata):
    n_time = demo_metadata["n_time_steps"]
    n_depth = len(demo_metadata["depths_m"])
    assert demo_dataset["sst"].shape == (n_time, EXPECTED_N_LAT, EXPECTED_N_LON)
    assert demo_dataset["subsurface_temp"].shape == (
        n_time,
        n_depth,
        EXPECTED_N_LAT,
        EXPECTED_N_LON,
    )


def test_sizes_report_every_dimension(demo_dataset, demo_metadata):
    assert demo_dataset.sizes == {
        "time": demo_metadata["n_time_steps"],
        "depth": len(demo_metadata["depths_m"]),
        "lat": EXPECTED_N_LAT,
        "lon": EXPECTED_N_LON,
    }
    assert demo_dataset.grid_shape == (EXPECTED_N_LAT, EXPECTED_N_LON)


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------


def test_coordinates_use_canonical_names(demo_dataset):
    # The .npz stores 'latitude'/'longitude'; the loader must rename them.
    assert set(demo_dataset.coord_names) == {"time", "depth", "lat", "lon"}
    assert "latitude" not in demo_dataset.coords
    assert "longitude" not in demo_dataset.coords


def test_latitude_spans_the_configured_domain(demo_dataset):
    lat = demo_dataset.lat
    assert lat[0] == pytest.approx(CONFIG.domain.lat_min)
    assert lat[-1] == pytest.approx(CONFIG.domain.lat_max)
    assert np.all(np.diff(lat) > 0)
    assert np.allclose(np.diff(lat), CONFIG.domain.resolution)


def test_longitude_spans_the_configured_domain(demo_dataset):
    lon = demo_dataset.lon
    assert lon[0] == pytest.approx(CONFIG.domain.lon_min)
    assert lon[-1] == pytest.approx(CONFIG.domain.lon_max)
    assert np.all(np.diff(lon) > 0)
    assert np.allclose(np.diff(lon), CONFIG.domain.resolution)


def test_depths_match_the_configured_levels(demo_dataset):
    assert list(demo_dataset.depth) == [float(d) for d in CONFIG.depths]


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


def test_timestamps_are_decoded_to_datetime64(demo_dataset):
    # The .npz stores dates as strings; the loader must decode them.
    assert np.issubdtype(demo_dataset.time.dtype, np.datetime64)
    assert not np.isnat(demo_dataset.time).any()


def test_timestamps_are_strictly_increasing(demo_dataset):
    assert np.all(np.diff(demo_dataset.time) > np.timedelta64(0, "ns"))


def test_timestamp_count_matches_metadata(demo_dataset, demo_metadata):
    assert demo_dataset.n_time == demo_metadata["n_time_steps"]


def test_time_range_matches_metadata(demo_dataset, demo_metadata):
    first, last = demo_metadata["time_range"]
    assert np.datetime_as_string(demo_dataset.time[0], unit="D") == first
    assert np.datetime_as_string(demo_dataset.time[-1], unit="D") == last


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_units_match_metadata(demo_dataset, demo_metadata):
    for name, entry in demo_metadata["variables"].items():
        assert demo_dataset[name].units == entry["units"], name


@pytest.mark.parametrize(
    "name,expected_units",
    [("sst", "degC"), ("sss", "psu"), ("sla", "m"), ("u_current", "m/s")],
)
def test_units_are_canonical(demo_dataset, name, expected_units):
    assert demo_dataset[name].units == expected_units


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------


def test_gaps_are_encoded_as_nan(demo_dataset):
    sst = demo_dataset["sst"]
    assert np.issubdtype(sst.dtype, np.floating)
    assert 0 < sst.missing_fraction < 1
    assert sst.n_missing == int(np.isnan(sst.values).sum())


def test_land_points_are_missing_in_surface_fields(demo_dataset):
    land = ~demo_dataset["land_mask"].values
    assert land.any()
    assert np.isnan(demo_dataset["sst"].values[0][land]).all()


def test_mask_variable_has_no_missing_values(demo_dataset):
    assert demo_dataset["land_mask"].missing_fraction == 0.0


def test_no_infinite_values(demo_dataset):
    for name in demo_dataset.variable_names:
        values = demo_dataset[name].values
        if np.issubdtype(values.dtype, np.floating):
            assert not np.isinf(values).any(), name


def test_value_range_ignores_missing_values(demo_dataset):
    low, high = demo_dataset["sst"].value_range
    assert np.isfinite(low) and np.isfinite(high)
    assert -5 < low < high < 45  # plausible SST band


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_demo_dataset_passes_validation(demo_dataset):
    report = demo_dataset.validation
    assert report is not None
    assert report.ok, f"demo dataset failed validation:\n{report}"
    assert report.errors == []


def test_demo_dataset_validates_strictly(demo_metadata):
    # The demo data is generated to be clean, so it should survive the
    # strict pass (warnings treated as failures) too. This is what makes
    # it a usable baseline for testing every other loader.
    dataset = load_demo_dataset(strict=True)
    assert dataset.validation.warnings == []


def test_validation_can_be_skipped():
    dataset = load_demo_dataset(validate=False, variables=["sst"])
    assert dataset.validation is None


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_data_mode_marks_the_dataset_synthetic(demo_dataset):
    assert demo_dataset.data_mode == DEMO_DATA_MODE
    assert demo_dataset.is_synthetic is True


def test_disclaimer_is_carried_through(demo_dataset):
    disclaimer = demo_dataset.attrs.get("disclaimer", "")
    assert "SYNTHETIC" in disclaimer
    assert "NOT satellite" in disclaimer


def test_generator_provenance_is_preserved(demo_dataset, demo_metadata):
    assert demo_dataset.attrs["generator"] == demo_metadata["generator"]
    assert demo_dataset.attrs["seed"] == demo_metadata["seed"]
    assert demo_dataset.attrs["problem_id"] == "SIH26066"


def test_rejects_data_without_the_demo_marker(tmp_path):
    path = tmp_path / "not_demo.npz"
    np.savez(
        path,
        latitude=np.array([10.0, 10.25, 10.5]),
        longitude=np.array([70.0, 70.25]),
        sst=np.full((3, 2), 28.0),
    )
    with pytest.raises(SchemaError, match="DEMO_SYNTHETIC"):
        load_demo_dataset(path, use_sidecar_metadata=False)


def test_marker_check_can_be_overridden(tmp_path):
    path = tmp_path / "not_demo.npz"
    np.savez(
        path,
        latitude=np.array([10.0, 10.25, 10.5]),
        longitude=np.array([70.0, 70.25]),
        sst=np.full((3, 2), 28.0),
    )
    dataset = load_demo_dataset(
        path, use_sidecar_metadata=False, require_demo_marker=False
    )
    assert dataset.is_synthetic is False


# --------------------------------------------------------------------------
# Selection & interop
# --------------------------------------------------------------------------


def test_variable_selection_loads_only_what_was_asked_for():
    dataset = load_demo_dataset(variables=["sst", "sla"])
    assert dataset.variable_names == ["sla", "sst"]


def test_selection_drops_unused_coordinates():
    dataset = load_demo_dataset(variables=["sst"])
    # sst has no depth axis, so no depth coordinate should be attached...
    assert "depth" not in dataset.subset(["sst"]).coords
    # ...and selecting a subsurface variable keeps it.
    subsurface = load_demo_dataset(variables=["subsurface_temp"])
    assert "depth" in subsurface.coords


def test_unknown_variable_selection_raises():
    with pytest.raises(KeyError, match="not found"):
        load_demo_dataset(variables=["no_such_variable"])


def test_missing_file_explains_how_to_generate_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="create_demo_data.py"):
        load_demo_dataset(tmp_path / "absent.npz")


def test_summary_is_json_serializable(demo_dataset):
    payload = json.dumps(demo_dataset.summary())
    restored = json.loads(payload)
    assert restored["data_mode"] == DEMO_DATA_MODE
    assert restored["is_synthetic"] is True
    assert {v["name"] for v in restored["variables"]} == set(demo_dataset.variable_names)


def test_to_dict_exposes_variables_and_coordinates(demo_dataset):
    arrays = demo_dataset.to_dict()
    assert "sst" in arrays and "lat" in arrays and "time" in arrays
    assert arrays["sst"].shape == demo_dataset["sst"].shape


def test_describe_mentions_synthetic_provenance(demo_dataset):
    text = demo_dataset.describe()
    assert DEMO_DATA_MODE in text
    assert "sst" in text


def test_generic_npz_loader_reads_the_demo_file_too(demo_dataset):
    # load_demo_dataset is a thin, safety-checked wrapper around load_npz;
    # the generic path must produce the same arrays.
    from src.data.loaders.demo_loader import DEFAULT_DEMO_DATASET

    dataset = load_npz(DEFAULT_DEMO_DATASET, variables=["sst"])
    assert np.allclose(
        dataset["sst"].values, demo_dataset["sst"].values, equal_nan=True
    )
