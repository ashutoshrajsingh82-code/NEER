"""
Tests for `src/data/loaders/validation.py`.

Each check is exercised against a deliberately broken dataset built from
the demo data's shape, so the failures tested here are the ones that
actually occur in ocean data: a flipped latitude axis, duplicated
timestamps, a field left in Kelvin, a variable that is entirely land.

The error/warning split is part of the contract and is tested explicitly:
errors block a load, warnings do not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import (
    DataValidationError,
    OceanDataset,
    Severity,
    ValidationReport,
    build_variable,
    validate_dataset,
)
from src.data.loaders.validation import (
    ALL_CHECKS,
    check_coordinates,
    check_dimensions,
    check_missing_values,
    check_timestamps,
    check_units,
    check_variables,
)

LAT = np.array([10.0, 10.25, 10.5])
LON = np.array([70.0, 70.25])
TIME = np.array(["2020-01-15", "2020-02-15"], dtype="datetime64[ns]")


def make_dataset(*, variables=None, coords=None, **kwargs) -> OceanDataset:
    """A valid baseline dataset, overridable piece by piece."""
    if coords is None:
        coords = {"time": TIME, "lat": LAT, "lon": LON}
    if variables is None:
        values = np.full((TIME.size, LAT.size, LON.size), 28.0)
        variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
        variables = {"sst": variable}
    return OceanDataset(variables=variables, coords=coords, **kwargs)


def issues_for(report: ValidationReport, check: str, severity: Severity):
    return [i for i in report.issues if i.check == check and i.severity is severity]


def run(check_fn, dataset, **kwargs) -> ValidationReport:
    report = ValidationReport()
    check_fn(dataset, report, **kwargs)
    return report


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_valid_dataset_produces_no_errors():
    report = validate_dataset(make_dataset(), check_domain=False)
    assert report.ok
    assert report.errors == []


def test_report_is_json_serializable():
    payload = validate_dataset(make_dataset(), check_domain=False).to_dict()
    assert payload["ok"] is True
    assert payload["counts"]["errors"] == 0


def test_unknown_check_is_rejected():
    with pytest.raises(ValueError, match="unknown check"):
        validate_dataset(make_dataset(), checks=["not_a_check"])


def test_checks_can_be_selected():
    report = validate_dataset(make_dataset(), checks=["units"], check_domain=False)
    assert set(report.checks_run()) <= {"units"}


def test_all_checks_constant_matches_the_default_run():
    assert set(ALL_CHECKS) == {
        "coordinates",
        "timestamps",
        "dimensions",
        "variables",
        "units",
        "missing_values",
    }


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------


def test_descending_latitude_is_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": LAT[::-1], "lon": LON},
    )
    report = run(check_coordinates, dataset)
    assert any("descending" in i.message for i in report.errors)


def test_non_monotonic_coordinate_is_an_error():
    dataset = make_dataset(coords={"time": TIME, "lat": np.array([10.0, 11.0, 10.5]), "lon": LON})
    report = run(check_coordinates, dataset)
    assert any("not monotonic" in i.message for i in report.errors)


def test_duplicate_coordinate_values_are_an_error():
    dataset = make_dataset(coords={"time": TIME, "lat": np.array([10.0, 10.0, 10.5]), "lon": LON})
    report = run(check_coordinates, dataset)
    assert any("duplicate" in i.message for i in report.errors)


def test_nan_in_coordinate_is_an_error():
    dataset = make_dataset(coords={"time": TIME, "lat": np.array([10.0, np.nan, 10.5]), "lon": LON})
    report = run(check_coordinates, dataset)
    assert any("NaN" in i.message for i in report.errors)


def test_missing_position_coordinate_is_an_error():
    variable = build_variable("sla", np.zeros((3,)), ("lat",), units="m")
    dataset = OceanDataset(variables={"sla": variable}, coords={"lat": LAT})
    report = run(check_coordinates, dataset)
    assert any(i.target == "lon" for i in report.errors)


def test_latitude_outside_physical_limits_is_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([88.0, 91.0, 95.0]), "lon": LON},
    )
    report = run(check_coordinates, dataset)
    assert any("latitudes must lie within" in i.message for i in report.errors)


def test_negative_depth_is_an_error():
    values = np.zeros((2, 3, 3, 2))
    variable = build_variable(
        "subsurface_temp", values, ("time", "depth", "lat", "lon"), units="degC"
    )
    dataset = OceanDataset(
        variables={"subsurface_temp": variable},
        coords={"time": TIME, "depth": np.array([-10.0, 0.0, 50.0]), "lat": LAT, "lon": LON},
    )
    report = run(check_coordinates, dataset)
    assert any("non-negative" in i.message for i in report.errors)


def test_uneven_spacing_is_a_warning_not_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([10.0, 10.25, 11.5]), "lon": LON},
    )
    report = run(check_coordinates, dataset)
    assert report.errors == []
    assert any("evenly spaced" in i.message for i in report.warnings)


def test_coordinates_outside_the_domain_are_a_warning():
    from src.utils.config import load_config

    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([-40.0, -39.75, -39.5]), "lon": LON},
    )
    report = run(check_coordinates, dataset, domain=load_config("base").domain)
    assert report.errors == []
    assert any(i.target == "lat" for i in report.warnings)


def test_domain_check_is_skipped_when_no_domain_is_given():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([-40.0, -39.75, -39.5]), "lon": LON},
    )
    report = run(check_coordinates, dataset, domain=None)
    assert report.warnings == []


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


def test_undecoded_time_is_an_error():
    dataset = make_dataset(coords={"time": np.array([0.0, 1.0]), "lat": LAT, "lon": LON})
    report = run(check_timestamps, dataset)
    assert any("datetime64" in i.message for i in report.errors)


def test_unparseable_timestamps_are_an_error():
    time = np.array(["2020-01-15", "NaT"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    report = run(check_timestamps, dataset)
    assert any("NaT" in i.message for i in report.errors)


def test_duplicate_timestamps_are_an_error():
    time = np.array(["2020-01-15", "2020-01-15"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    report = run(check_timestamps, dataset)
    assert any("duplicate" in i.message for i in report.errors)


def test_unordered_timestamps_are_an_error():
    time = np.array(["2020-03-15", "2020-01-15"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    report = run(check_timestamps, dataset)
    assert any("increasing" in i.message for i in report.errors)


def test_irregular_cadence_is_a_warning():
    time = np.array(
        ["2020-01-01", "2020-01-02", "2020-06-01", "2020-06-02"], dtype="datetime64[ns]"
    )
    values = np.zeros((4, LAT.size, LON.size))
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = OceanDataset(
        variables={"sst": variable}, coords={"time": time, "lat": LAT, "lon": LON}
    )
    report = run(check_timestamps, dataset)
    assert report.errors == []
    assert any("irregular" in i.message for i in report.warnings)


def test_absent_time_coordinate_is_informational():
    variable = build_variable("land_mask", np.ones((3, 2), dtype=bool), ("lat", "lon"))
    dataset = OceanDataset(
        variables={"land_mask": variable}, coords={"lat": LAT, "lon": LON}
    )
    report = run(check_timestamps, dataset)
    assert report.errors == []
    assert report.infos


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------


def test_dimension_length_mismatch_is_an_error():
    values = np.zeros((TIME.size, LAT.size + 1, LON.size))
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_dimensions, dataset)
    assert any("has length" in i.message for i in report.errors)


def test_dimension_without_a_coordinate_is_an_error():
    values = np.zeros((TIME.size, LAT.size, LON.size, 2))
    variable = build_variable(
        "sst", values, ("time", "lat", "lon", "ensemble"), units="degC"
    )
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_dimensions, dataset)
    assert any("no matching coordinate" in i.message for i in report.errors)


def test_non_canonical_dimension_order_is_an_error():
    from src.data.loaders import Variable

    values = np.zeros((LAT.size, LON.size, TIME.size))
    # Constructed directly to bypass build_variable's automatic transpose.
    variable = Variable(name="sst", values=values, dims=("lat", "lon", "time"), units="degC")
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_dimensions, dataset)
    assert any("canonical order" in i.message for i in report.errors)


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------


def test_dataset_without_variables_is_an_error():
    dataset = OceanDataset(variables={}, coords={"lat": LAT, "lon": LON})
    report = run(check_variables, dataset)
    assert any("no variables" in i.message for i in report.errors)


def test_unregistered_variable_is_a_warning():
    variable = build_variable(
        "mystery", np.zeros((TIME.size, LAT.size, LON.size)), ("time", "lat", "lon")
    )
    dataset = make_dataset(variables={"mystery": variable})
    report = run(check_variables, dataset)
    assert report.errors == []
    assert any(i.target == "mystery" for i in report.warnings)


def test_implausible_values_are_a_warning():
    # SST left in Kelvin: numerically fine, physically absurd as Celsius.
    values = np.full((TIME.size, LAT.size, LON.size), 301.15)
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_variables, dataset)
    assert report.errors == []
    assert any("plausible range" in i.message for i in report.warnings)


def test_unexpected_dimension_is_a_warning():
    values = np.zeros((TIME.size, 3, LAT.size, LON.size))
    variable = build_variable("sst", values, ("time", "depth", "lat", "lon"), units="degC")
    dataset = OceanDataset(
        variables={"sst": variable},
        coords={"time": TIME, "depth": np.array([0.0, 10.0, 20.0]), "lat": LAT, "lon": LON},
    )
    report = run(check_variables, dataset)
    assert any("unexpected dimension" in i.message for i in report.warnings)


def test_missing_depth_dimension_is_a_warning():
    values = np.zeros((TIME.size, LAT.size, LON.size))
    variable = build_variable("subsurface_temp", values, ("time", "lat", "lon"), units="degC")
    dataset = make_dataset(variables={"subsurface_temp": variable})
    report = run(check_variables, dataset)
    assert any("no depth dimension" in i.message for i in report.warnings)


def test_missing_time_dimension_is_informational():
    variable = build_variable("sst", np.full((LAT.size, LON.size), 28.0), ("lat", "lon"), units="degC")
    dataset = OceanDataset(variables={"sst": variable}, coords={"lat": LAT, "lon": LON})
    report = run(check_variables, dataset)
    assert report.errors == [] and report.warnings == []
    assert any("no time dimension" in i.message for i in report.infos)


def test_non_numeric_variable_is_an_error():
    from src.data.loaders import Variable

    values = np.array([["a", "b"], ["c", "d"], ["e", "f"]])
    variable = Variable(name="sst", values=values, dims=("lat", "lon"))
    dataset = OceanDataset(variables={"sst": variable}, coords={"lat": LAT, "lon": LON})
    report = run(check_variables, dataset)
    assert any("not numeric" in i.message for i in report.errors)


def test_non_boolean_mask_is_a_warning():
    variable = build_variable("land_mask", np.ones((LAT.size, LON.size)), ("lat", "lon"))
    dataset = OceanDataset(variables={"land_mask": variable}, coords={"lat": LAT, "lon": LON})
    report = run(check_variables, dataset)
    assert any("expected boolean" in i.message for i in report.warnings)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_missing_units_is_a_warning():
    variable = build_variable(
        "sst", np.full((TIME.size, LAT.size, LON.size), 28.0), ("time", "lat", "lon")
    )
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_units, dataset)
    assert any("no units declared" in i.message for i in report.warnings)


def test_unrecognized_unit_is_a_warning():
    variable = build_variable(
        "sst", np.full((TIME.size, LAT.size, LON.size), 28.0), ("time", "lat", "lon"), units="furlongs"
    )
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_units, dataset)
    assert any("unrecognized unit" in i.message for i in report.warnings)


def test_wrong_unit_for_a_known_variable_is_an_error():
    variable = build_variable(
        "sst", np.full((TIME.size, LAT.size, LON.size), 28.0), ("time", "lat", "lon"), units="psu"
    )
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_units, dataset)
    assert any("does not match the canonical unit" in i.message for i in report.errors)


def test_equivalent_unit_spellings_are_accepted():
    variable = build_variable(
        "sst", np.full((TIME.size, LAT.size, LON.size), 28.0), ("time", "lat", "lon"), units="deg_C"
    )
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_units, dataset)
    assert report.errors == [] and report.warnings == []


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------


def test_fully_missing_variable_is_an_error():
    values = np.full((TIME.size, LAT.size, LON.size), np.nan)
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_missing_values, dataset)
    assert any("entirely missing" in i.message for i in report.errors)


def test_mostly_missing_variable_is_a_warning():
    # One valid point in a 2x30x2 field: 99.2% missing, past the default
    # 95% threshold but short of the fully-missing error.
    lat = np.arange(30) * 0.25 + 10.0
    values = np.full((TIME.size, lat.size, LON.size), np.nan)
    values[0, 0, 0] = 28.0
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = OceanDataset(
        variables={"sst": variable}, coords={"time": TIME, "lat": lat, "lon": LON}
    )
    report = run(check_missing_values, dataset)
    assert report.errors == []
    assert any("missing" in i.message for i in report.warnings)


def test_partial_gaps_are_only_informational():
    values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    values[0, 0, 0] = np.nan
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_missing_values, dataset)
    assert report.errors == [] and report.warnings == []
    assert report.infos


def test_infinite_values_are_an_error():
    values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    values[0, 0, 0] = np.inf
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = make_dataset(variables={"sst": variable})
    report = run(check_missing_values, dataset)
    assert any("infinite" in i.message for i in report.errors)


def test_missing_warn_threshold_is_configurable():
    values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    values[0] = np.nan
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = make_dataset(variables={"sst": variable})
    assert run(check_missing_values, dataset, warn_fraction=0.9).warnings == []
    assert run(check_missing_values, dataset, warn_fraction=0.1).warnings


# --------------------------------------------------------------------------
# Report behaviour
# --------------------------------------------------------------------------


def test_raise_for_status_is_silent_when_valid():
    validate_dataset(make_dataset(), check_domain=False).raise_for_status()


def test_raise_for_status_raises_on_errors():
    values = np.full((TIME.size, LAT.size, LON.size), np.nan)
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    report = validate_dataset(make_dataset(variables={"sst": variable}), check_domain=False)
    with pytest.raises(DataValidationError):
        report.raise_for_status()


def test_strict_mode_raises_on_warnings_only():
    variable = build_variable(
        "mystery", np.zeros((TIME.size, LAT.size, LON.size)), ("time", "lat", "lon"), units="m"
    )
    report = validate_dataset(make_dataset(variables={"mystery": variable}), check_domain=False)
    assert report.ok
    report.raise_for_status()  # lenient: fine
    with pytest.raises(DataValidationError):
        report.raise_for_status(strict=True)


def test_error_carries_the_report():
    values = np.full((TIME.size, LAT.size, LON.size), np.nan)
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    report = validate_dataset(make_dataset(variables={"sst": variable}), check_domain=False)
    with pytest.raises(DataValidationError) as excinfo:
        report.raise_for_status()
    assert excinfo.value.report is report


def test_summary_line_counts_every_severity():
    report = ValidationReport()
    report.error("units", "boom")
    report.warn("units", "hmm")
    report.info("units", "fyi")
    assert "1 error(s)" in report.summary_line()
    assert "1 warning(s)" in report.summary_line()
    assert len(report) == 3
