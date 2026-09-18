"""
Tests for `load_csv` (src/data/loaders/csv_loader.py).

The core case is a round-trip: a block of the demo dataset is written out
as a long-format observation CSV and read back, and must come back with
identical values, coordinates, and gaps. The rest cover the messiness
real observation CSVs bring — alias column names, sentinel values,
duplicate rows, units that need converting, positions that do not sit
exactly on the project grid.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import DataValidationError, SchemaError, load_csv

BASIC_CSV = """time,lat,lon,sst
2020-01-15,12.00,72.00,28.5
2020-01-15,12.25,72.00,28.7
2020-02-15,12.00,72.00,29.1
2020-02-15,12.25,72.00,29.3
"""


def write_csv(path: Path, text: str, name: str = "data.csv") -> Path:
    target = path / name
    target.write_text(text, encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# Round-trip against the demo dataset
# --------------------------------------------------------------------------


def test_round_trip_preserves_values(sample_csv_path, demo_sample):
    dataset = load_csv(sample_csv_path, units={"sea_surface_temperature": "degC"})
    assert np.allclose(dataset["sst"].values, demo_sample.sst, equal_nan=True)


def test_round_trip_preserves_coordinates(sample_csv_path, demo_sample):
    dataset = load_csv(sample_csv_path)
    assert np.allclose(dataset.lat, demo_sample.lat)
    assert np.allclose(dataset.lon, demo_sample.lon)
    assert np.array_equal(dataset.time, demo_sample.time)


def test_round_trip_preserves_gaps(sample_csv_path, demo_sample):
    dataset = load_csv(sample_csv_path)
    expected_missing = np.isnan(demo_sample.sst)
    assert np.array_equal(np.isnan(dataset["sst"].values), expected_missing)


def test_round_trip_produces_canonical_dims(sample_csv_path):
    dataset = load_csv(sample_csv_path)
    assert dataset["sst"].dims == ("time", "lat", "lon")
    assert dataset.coord_names == ["time", "lat", "lon"]


def test_round_trip_passes_validation(sample_csv_path):
    dataset = load_csv(sample_csv_path, units={"sea_surface_temperature": "degC"})
    assert dataset.validation.ok


def test_source_metadata_is_recorded(sample_csv_path, demo_sample):
    dataset = load_csv(sample_csv_path)
    assert dataset.source_format == "csv"
    assert dataset.source == str(sample_csv_path)
    assert dataset.attrs["n_rows_read"] == int(np.prod(demo_sample.shape))


# --------------------------------------------------------------------------
# Column resolution
# --------------------------------------------------------------------------


def test_coordinate_columns_are_detected_by_alias(tmp_path):
    csv = BASIC_CSV.replace("time,lat,lon,sst", "DATE,Latitude,LONGITUDE,sst")
    dataset = load_csv(write_csv(tmp_path, csv))
    assert dataset.coord_names == ["time", "lat", "lon"]
    assert dataset["sst"].shape == (2, 2, 1)


def test_variable_names_are_canonicalized(tmp_path):
    csv = BASIC_CSV.replace(",sst", ",sea_surface_temperature")
    dataset = load_csv(write_csv(tmp_path, csv))
    assert "sst" in dataset
    assert dataset["sst"].attrs["source_column"] == "sea_surface_temperature"


def test_explicit_column_names_are_respected(tmp_path):
    csv = "when,y_pos,x_pos,sst\n2020-01-15,12.0,72.0,28.5\n2020-02-15,12.0,72.0,29.0\n"
    dataset = load_csv(
        write_csv(tmp_path, csv),
        time_column="when",
        lat_column="y_pos",
        lon_column="x_pos",
    )
    assert dataset.sizes == {"time": 2, "lat": 1, "lon": 1}


def test_explicit_column_must_exist(tmp_path):
    with pytest.raises(SchemaError, match="not in the CSV"):
        load_csv(write_csv(tmp_path, BASIC_CSV), lat_column="nope")


def test_missing_position_columns_raise(tmp_path):
    csv = "time,sst\n2020-01-15,28.5\n"
    with pytest.raises(SchemaError, match="must have a lat column"):
        load_csv(write_csv(tmp_path, csv))


def test_csv_without_value_columns_raises(tmp_path):
    csv = "time,lat,lon\n2020-01-15,12.0,72.0\n"
    with pytest.raises(SchemaError, match="no value columns"):
        load_csv(write_csv(tmp_path, csv))


def test_empty_csv_raises(tmp_path):
    with pytest.raises(SchemaError, match="no data rows"):
        load_csv(write_csv(tmp_path, "time,lat,lon,sst\n"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "absent.csv")


def test_variable_selection(tmp_path):
    csv = BASIC_CSV.replace(",sst\n", ",sst,sla\n").replace(",28.5", ",28.5,0.1").replace(
        ",28.7", ",28.7,0.2"
    ).replace(",29.1", ",29.1,0.3").replace(",29.3", ",29.3,0.4")
    dataset = load_csv(write_csv(tmp_path, csv), variables=["sla"])
    assert dataset.variable_names == ["sla"]


def test_unknown_variable_selection_raises(tmp_path):
    with pytest.raises(KeyError, match="not found"):
        load_csv(write_csv(tmp_path, BASIC_CSV), variables=["missing_column"])


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------


def test_depth_column_creates_a_depth_dimension(tmp_path):
    csv = (
        "time,lat,lon,depth,subsurface_temp\n"
        "2020-01-15,12.0,72.0,0,28.5\n"
        "2020-01-15,12.0,72.0,50,26.0\n"
        "2020-01-15,12.0,72.0,100,22.0\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv))
    assert dataset["subsurface_temp"].dims == ("time", "depth", "lat", "lon")
    assert list(dataset.depth) == [0.0, 50.0, 100.0]


def test_pressure_column_is_treated_as_depth(tmp_path):
    csv = (
        "time,lat,lon,pressure,subsurface_temp\n"
        "2020-01-15,12.0,72.0,0,28.5\n"
        "2020-01-15,12.0,72.0,50,26.0\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv))
    assert "depth" in dataset.coords


def test_csv_without_time_column_yields_no_time_dimension(tmp_path):
    csv = "lat,lon,sla\n12.0,72.0,0.1\n12.25,72.0,0.2\n"
    dataset = load_csv(write_csv(tmp_path, csv))
    assert "time" not in dataset.coords
    assert dataset["sla"].dims == ("lat", "lon")


def test_unobserved_cells_become_nan(tmp_path):
    # A ragged sampling: (12.25, 72.25) is never observed.
    csv = (
        "time,lat,lon,sla\n"
        "2020-01-15,12.00,72.00,0.1\n"
        "2020-01-15,12.00,72.25,0.2\n"
        "2020-01-15,12.25,72.00,0.3\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv))
    values = dataset["sla"].values
    assert values.shape == (1, 2, 2)
    assert np.isnan(values[0, 1, 1])
    assert dataset["sla"].missing_fraction == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Values: units, sentinels, duplicates
# --------------------------------------------------------------------------


def test_units_are_converted_to_canonical(tmp_path):
    csv = "time,lat,lon,sst\n2020-01-15,12.0,72.0,301.15\n2020-02-15,12.0,72.0,300.15\n"
    dataset = load_csv(write_csv(tmp_path, csv), units={"sst": "K"})
    assert dataset["sst"].units == "degC"
    assert dataset["sst"].values[0, 0, 0] == pytest.approx(28.0)


def test_original_units_are_recorded_after_conversion(tmp_path):
    csv = "time,lat,lon,sst\n2020-01-15,12.0,72.0,301.15\n2020-02-15,12.0,72.0,300.15\n"
    dataset = load_csv(write_csv(tmp_path, csv), units={"sst": "K"})
    assert dataset["sst"].attrs["source_units"] == "K"


def test_units_row_is_read_from_the_file(tmp_path):
    csv = (
        "time,lat,lon,sst\n"
        ",,,K\n"
        "2020-01-15,12.0,72.0,301.15\n"
        "2020-02-15,12.0,72.0,300.15\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv), units_row=True)
    assert dataset["sst"].units == "degC"
    assert dataset["sst"].values[0, 0, 0] == pytest.approx(28.0)


def test_explicit_units_override_the_units_row(tmp_path):
    csv = (
        "time,lat,lon,sst\n"
        ",,,K\n"
        "2020-01-15,12.0,72.0,28.0\n"
        "2020-02-15,12.0,72.0,29.0\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv), units_row=True, units={"sst": "degC"})
    assert dataset["sst"].values[0, 0, 0] == pytest.approx(28.0)


@pytest.mark.parametrize("sentinel", ["-999", "-9999", "NA", "n/a", ""])
def test_sentinel_values_become_nan(tmp_path, sentinel):
    csv = (
        "time,lat,lon,sst\n"
        f"2020-01-15,12.0,72.0,{sentinel}\n"
        "2020-02-15,12.0,72.0,29.0\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv, name=f"s{abs(hash(sentinel))}.csv"))
    assert np.isnan(dataset["sst"].values[0, 0, 0])
    assert dataset["sst"].values[1, 0, 0] == pytest.approx(29.0)


def test_duplicate_rows_are_averaged_by_default(tmp_path):
    csv = (
        "time,lat,lon,sst\n"
        "2020-01-15,12.0,72.0,28.0\n"
        "2020-01-15,12.0,72.0,30.0\n"
        "2020-02-15,12.0,72.0,29.0\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv))
    assert dataset["sst"].values[0, 0, 0] == pytest.approx(29.0)
    assert dataset.attrs["n_duplicate_cells_aggregated"] == 1


def test_duplicate_aggregation_strategy_is_configurable(tmp_path):
    csv = (
        "time,lat,lon,sst\n"
        "2020-01-15,12.0,72.0,28.0\n"
        "2020-01-15,12.0,72.0,30.0\n"
        "2020-02-15,12.0,72.0,29.0\n"
    )
    path = write_csv(tmp_path, csv)
    assert load_csv(path, aggregate="first")["sst"].values[0, 0, 0] == pytest.approx(28.0)
    assert load_csv(path, aggregate="max")["sst"].values[0, 0, 0] == pytest.approx(30.0)


def test_duplicate_rows_can_be_made_an_error(tmp_path):
    csv = (
        "time,lat,lon,sst\n"
        "2020-01-15,12.0,72.0,28.0\n"
        "2020-01-15,12.0,72.0,30.0\n"
        "2020-02-15,12.0,72.0,29.0\n"
    )
    with pytest.raises(SchemaError, match="another row already fills"):
        load_csv(write_csv(tmp_path, csv), aggregate="error")


def test_invalid_aggregate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="aggregate must be one of"):
        load_csv(write_csv(tmp_path, BASIC_CSV), aggregate="nonsense")


def test_rows_with_missing_coordinates_are_dropped(tmp_path):
    csv = (
        "time,lat,lon,sst\n"
        "2020-01-15,12.0,72.0,28.0\n"
        "2020-01-15,,72.0,30.0\n"
        "2020-02-15,12.0,72.0,29.0\n"
    )
    dataset = load_csv(write_csv(tmp_path, csv))
    assert dataset.attrs["n_rows_dropped_missing_coords"] == 1
    assert dataset.attrs["n_rows_used"] == 2


# --------------------------------------------------------------------------
# Grid modes
# --------------------------------------------------------------------------


def test_domain_grid_mode_builds_the_full_project_grid(tmp_path):
    csv = "time,lat,lon,sst\n2020-01-15,12.0,72.0,28.5\n2020-02-15,12.0,72.0,29.0\n"
    dataset = load_csv(write_csv(tmp_path, csv), grid="domain")
    assert dataset.grid_shape == (101, 241)
    assert dataset.attrs["grid_mode"] == "domain"
    # Only the two observed cells hold data; the rest of the grid is missing.
    assert dataset["sst"].n_missing == dataset["sst"].values.size - 2


def test_domain_grid_mode_snaps_nearby_observations(tmp_path):
    # 12.03 is within half a cell of the 12.00 grid line at 0.25° resolution.
    csv = "time,lat,lon,sst\n2020-01-15,12.03,72.02,28.5\n2020-02-15,12.0,72.0,29.0\n"
    dataset = load_csv(write_csv(tmp_path, csv), grid="domain")
    lat_index = int(np.argmin(np.abs(dataset.lat - 12.0)))
    lon_index = int(np.argmin(np.abs(dataset.lon - 72.0)))
    assert dataset["sst"].values[0, lat_index, lon_index] == pytest.approx(28.5)


def test_observations_outside_the_grid_are_dropped(tmp_path):
    # 40N is outside the NEER domain (5-30N) entirely.
    csv = "time,lat,lon,sst\n2020-01-15,12.0,72.0,28.5\n2020-01-15,40.0,72.0,20.0\n"
    dataset = load_csv(write_csv(tmp_path, csv), grid="domain")
    assert dataset.attrs["n_rows_off_grid"] == 1
    assert dataset["sst"].n_missing == dataset["sst"].values.size - 1


def test_invalid_grid_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="grid must be"):
        load_csv(write_csv(tmp_path, BASIC_CSV), grid="sideways")


# --------------------------------------------------------------------------
# Validation behaviour
# --------------------------------------------------------------------------


def test_validation_failure_raises(tmp_path):
    # Every value is missing, so the variable is unusable.
    csv = "time,lat,lon,sst\n2020-01-15,12.0,72.0,\n2020-02-15,12.0,72.0,\n"
    with pytest.raises(DataValidationError, match="entirely missing"):
        load_csv(write_csv(tmp_path, csv))


def test_validation_can_be_skipped(tmp_path):
    csv = "time,lat,lon,sst\n2020-01-15,12.0,72.0,\n2020-02-15,12.0,72.0,\n"
    dataset = load_csv(write_csv(tmp_path, csv), validate=False)
    assert dataset.validation is None
    assert np.isnan(dataset["sst"].values).all()


def test_validation_error_carries_the_full_report(tmp_path):
    csv = "time,lat,lon,sst\n2020-01-15,12.0,72.0,\n2020-02-15,12.0,72.0,\n"
    with pytest.raises(DataValidationError) as excinfo:
        load_csv(write_csv(tmp_path, csv))
    assert not excinfo.value.report.ok
    assert excinfo.value.report.errors[0].check == "missing_values"


def test_strict_mode_rejects_warnings(tmp_path):
    # An unregistered variable name is a warning, not an error.
    csv = "time,lat,lon,mystery\n2020-01-15,12.0,72.0,1.0\n2020-02-15,12.0,72.0,2.0\n"
    path = write_csv(tmp_path, csv)
    assert load_csv(path).validation.warnings  # tolerated by default
    with pytest.raises(DataValidationError):
        load_csv(path, strict=True)


def test_out_of_domain_coordinates_warn_but_load(tmp_path):
    csv = "time,lat,lon,sst\n2020-01-15,-40.0,72.0,18.0\n2020-02-15,-40.0,72.0,19.0\n"
    dataset = load_csv(write_csv(tmp_path, csv), units={"sst": "degC"})
    checks = {issue.check for issue in dataset.validation.warnings}
    assert "coordinates" in checks


def test_domain_check_can_be_disabled(tmp_path):
    csv = "time,lat,lon,sst\n2020-01-15,-40.0,72.0,18.0\n2020-02-15,-40.0,72.0,19.0\n"
    dataset = load_csv(
        write_csv(tmp_path, csv), units={"sst": "degC"}, check_domain=False
    )
    assert dataset.validation.warnings == []
