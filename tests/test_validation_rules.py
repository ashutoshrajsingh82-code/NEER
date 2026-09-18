"""
Tests for each validation rule (src/data/validation/rules.py).

Every rule gets the same treatment: a clean dataset must come back VALID,
and a dataset broken in exactly one way must come back with the expected
status, on the expected check, against the expected target. Keeping the
damage to one cause per test is what makes a failure here diagnostic.

The error/warning split is asserted explicitly and repeatedly, because it
is the contract Phase 06 is really asking for: a physically impossible
coordinate must block the pipeline, while data that is merely outside the
configured domain must not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import OceanDataset, build_variable
from src.data.validation import (
    ALL_RULES,
    SURFACE_VARIABLES,
    ValidationStatus,
    depth_coordinates,
    duplicate_coordinates,
    duplicate_timestamps,
    latitude_range,
    longitude_range,
    nan_values,
    resolution,
    time_coordinates,
    units,
    validate,
    variable_existence,
)
from src.utils.config import load_config

CONFIG = load_config("base")
DOMAIN = CONFIG.domain

# A small, entirely valid dataset inside the NEER domain at 0.25°.
LAT = np.array([10.0, 10.25, 10.5])
LON = np.array([70.0, 70.25])
TIME = np.array(["2020-01-15", "2020-02-15"], dtype="datetime64[ns]")
DEPTH = np.array([float(d) for d in CONFIG.depths])


def make_dataset(*, variables=None, coords=None, **kwargs) -> OceanDataset:
    if coords is None:
        coords = {"time": TIME, "lat": LAT, "lon": LON}
    if variables is None:
        values = np.full((TIME.size, LAT.size, LON.size), 28.0)
        variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
        variables = {"sst": variable}
    return OceanDataset(variables=variables, coords=coords, **kwargs)


def sst_variable(values=None, *, unit="degC"):
    if values is None:
        values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    return build_variable("sst", values, ("time", "lat", "lon"), units=unit)


# --------------------------------------------------------------------------
# Latitude range
# --------------------------------------------------------------------------


def test_valid_latitude_passes():
    assert latitude_range(make_dataset(), domain=DOMAIN).status is ValidationStatus.VALID


def test_impossible_latitude_is_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([88.0, 91.0, 95.0]), "lon": LON}
    )
    result = latitude_range(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.ERROR
    assert any("physical limits" in f.message for f in result.errors)


def test_latitude_outside_the_configured_domain_is_only_a_warning():
    # 45N is real latitude, just not in NEER's domain: the data needs
    # subsetting, it is not broken.
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([45.0, 45.25, 45.5]), "lon": LON}
    )
    result = latitude_range(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []
    assert any("configured NEER domain" in f.message for f in result.warnings)


def test_descending_latitude_is_an_error():
    dataset = make_dataset(coords={"time": TIME, "lat": LAT[::-1], "lon": LON})
    result = latitude_range(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.ERROR
    assert any("descending" in f.message for f in result.errors)


def test_nan_in_latitude_is_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([10.0, np.nan, 10.5]), "lon": LON}
    )
    assert latitude_range(dataset, domain=DOMAIN).status is ValidationStatus.ERROR


def test_absent_latitude_is_an_error():
    dataset = OceanDataset(
        variables={}, coords={"time": TIME, "lon": LON}
    )
    result = latitude_range(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.ERROR
    assert any("no 'lat' coordinate" in f.message for f in result.errors)


def test_domain_comparison_is_optional():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([45.0, 45.25, 45.5]), "lon": LON}
    )
    assert latitude_range(dataset, domain=None).status is ValidationStatus.VALID


# --------------------------------------------------------------------------
# Longitude range
# --------------------------------------------------------------------------


def test_valid_longitude_passes():
    assert longitude_range(make_dataset(), domain=DOMAIN).status is ValidationStatus.VALID


def test_impossible_longitude_is_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": LAT, "lon": np.array([400.0, 420.0])}
    )
    result = longitude_range(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.ERROR


def test_longitude_outside_the_domain_is_a_warning():
    dataset = make_dataset(
        coords={"time": TIME, "lat": LAT, "lon": np.array([-120.0, -119.75])}
    )
    result = longitude_range(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.WARNING


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_configured_resolution_passes():
    assert resolution(make_dataset(), domain=DOMAIN).status is ValidationStatus.VALID


def test_uneven_spacing_is_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([10.0, 10.25, 11.5]), "lon": LON}
    )
    result = resolution(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.ERROR
    assert any("uneven" in f.message for f in result.errors)


def test_different_but_uniform_resolution_is_a_warning():
    # A regular 1° grid: usable data that needs regridding, not bad data.
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([10.0, 11.0, 12.0]), "lon": LON}
    )
    result = resolution(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []
    assert any("configuration expects" in f.message for f in result.warnings)


def test_observed_resolution_is_reported():
    result = resolution(make_dataset(), domain=DOMAIN)
    assert result.details["lat_resolution"] == pytest.approx(0.25)
    assert result.details["configured_resolution"] == pytest.approx(DOMAIN.resolution)


def test_single_point_axis_warns_rather_than_failing():
    dataset = make_dataset(
        variables={"sst": build_variable(
            "sst", np.full((TIME.size, 1, LON.size), 28.0), ("time", "lat", "lon"), units="degC"
        )},
        coords={"time": TIME, "lat": np.array([10.0]), "lon": LON},
    )
    result = resolution(dataset, domain=DOMAIN)
    assert result.status is ValidationStatus.WARNING


# --------------------------------------------------------------------------
# Depth coordinates
# --------------------------------------------------------------------------


def subsurface_dataset(depth=DEPTH):
    values = np.full((TIME.size, np.asarray(depth).size, LAT.size, LON.size), 15.0)
    variable = build_variable(
        "subsurface_temp", values, ("time", "depth", "lat", "lon"), units="degC"
    )
    return OceanDataset(
        variables={"subsurface_temp": variable},
        coords={"time": TIME, "depth": np.asarray(depth), "lat": LAT, "lon": LON},
    )


def test_configured_depths_pass():
    result = depth_coordinates(subsurface_dataset(), depths=CONFIG.depths)
    assert result.status is ValidationStatus.VALID


def test_surface_only_dataset_skips_the_depth_check():
    result = depth_coordinates(make_dataset(), depths=CONFIG.depths)
    assert result.skipped
    assert result.status is ValidationStatus.VALID


def test_negative_depth_is_an_error():
    result = depth_coordinates(
        subsurface_dataset(np.array([-10.0, 0.0, 50.0])), depths=None
    )
    assert result.status is ValidationStatus.ERROR
    assert any("non-negative" in f.message for f in result.errors)


def test_descending_depth_is_an_error():
    result = depth_coordinates(subsurface_dataset(DEPTH[::-1]), depths=None)
    assert result.status is ValidationStatus.ERROR
    assert any("descending" in f.message for f in result.errors)


def test_implausibly_deep_axis_is_an_error():
    result = depth_coordinates(
        subsurface_dataset(np.array([0.0, 5000.0, 50000.0])), depths=None
    )
    assert result.status is ValidationStatus.ERROR
    assert any("deepest point" in f.message for f in result.errors)


def test_depths_differing_from_configuration_are_a_warning():
    result = depth_coordinates(
        subsurface_dataset(np.array([0.0, 10.0, 25.0])), depths=CONFIG.depths
    )
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []


def test_depth_comparison_is_optional():
    result = depth_coordinates(subsurface_dataset(np.array([0.0, 10.0, 25.0])), depths=None)
    assert result.status is ValidationStatus.VALID


# --------------------------------------------------------------------------
# Time coordinates
# --------------------------------------------------------------------------


def test_valid_time_axis_passes():
    assert time_coordinates(make_dataset()).status is ValidationStatus.VALID


def test_static_dataset_skips_the_time_check():
    mask = build_variable("land_mask", np.ones((LAT.size, LON.size), dtype=bool), ("lat", "lon"))
    dataset = OceanDataset(variables={"land_mask": mask}, coords={"lat": LAT, "lon": LON})
    assert time_coordinates(dataset).skipped


def test_undecoded_time_is_an_error():
    dataset = make_dataset(coords={"time": np.array([0.0, 1.0]), "lat": LAT, "lon": LON})
    result = time_coordinates(dataset)
    assert result.status is ValidationStatus.ERROR
    assert any("datetime64" in f.message for f in result.errors)


def test_unparseable_timestamps_are_an_error():
    time = np.array(["2020-01-15", "NaT"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    assert time_coordinates(dataset).status is ValidationStatus.ERROR


def test_unsorted_time_is_an_error():
    time = np.array(["2020-03-15", "2020-01-15"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    result = time_coordinates(dataset)
    assert result.status is ValidationStatus.ERROR
    assert any("ascending" in f.message for f in result.errors)


def test_implausible_epoch_is_an_error():
    # A classic wrong-epoch decode: 'days since' read against year 0.
    time = np.array(["1601-01-01", "1601-02-01"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    result = time_coordinates(dataset)
    assert result.status is ValidationStatus.ERROR
    assert any("plausible range" in f.message for f in result.errors)


def test_irregular_cadence_is_a_warning():
    time = np.array(
        ["2020-01-01", "2020-01-02", "2020-06-01", "2020-06-02"], dtype="datetime64[ns]"
    )
    values = np.full((4, LAT.size, LON.size), 28.0)
    dataset = OceanDataset(
        variables={"sst": build_variable("sst", values, ("time", "lat", "lon"), units="degC")},
        coords={"time": time, "lat": LAT, "lon": LON},
    )
    result = time_coordinates(dataset)
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []


# --------------------------------------------------------------------------
# Variable existence
# --------------------------------------------------------------------------


def test_required_variables_present_passes():
    assert variable_existence(make_dataset(), required=["sst"]).status is ValidationStatus.VALID


def test_missing_required_variable_is_an_error():
    result = variable_existence(make_dataset(), required=SURFACE_VARIABLES)
    assert result.status is ValidationStatus.ERROR
    assert any("required variable" in f.message for f in result.errors)
    assert set(result.details["missing"]) == set(SURFACE_VARIABLES) - {"sst"}


def test_dataset_with_no_variables_is_an_error():
    dataset = OceanDataset(variables={}, coords={"lat": LAT, "lon": LON})
    assert variable_existence(dataset).status is ValidationStatus.ERROR


def test_unregistered_variable_is_a_warning():
    variable = build_variable(
        "mystery", np.zeros((TIME.size, LAT.size, LON.size)), ("time", "lat", "lon"), units="m"
    )
    dataset = make_dataset(
        variables={"sst": sst_variable(), "mystery": variable}
    )
    result = variable_existence(dataset)
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []


def test_dataset_with_no_recognized_variables_at_all_is_an_error():
    variable = build_variable(
        "mystery", np.zeros((TIME.size, LAT.size, LON.size)), ("time", "lat", "lon"), units="m"
    )
    result = variable_existence(make_dataset(variables={"mystery": variable}))
    assert result.status is ValidationStatus.ERROR


def test_no_required_list_means_no_requirement():
    assert variable_existence(make_dataset()).status is ValidationStatus.VALID


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_canonical_units_pass():
    assert units(make_dataset()).status is ValidationStatus.VALID


def test_equivalent_unit_spelling_passes():
    assert units(make_dataset(variables={"sst": sst_variable(unit="deg_C")})).status is (
        ValidationStatus.VALID
    )


def test_wrong_unit_for_the_variable_is_an_error():
    result = units(make_dataset(variables={"sst": sst_variable(unit="psu")}))
    assert result.status is ValidationStatus.ERROR
    assert any("contradicts" in f.message for f in result.errors)


def test_undeclared_units_are_a_warning():
    result = units(make_dataset(variables={"sst": sst_variable(unit=None)}))
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []


def test_unrecognized_unit_is_a_warning():
    result = units(make_dataset(variables={"sst": sst_variable(unit="furlongs")}))
    assert result.status is ValidationStatus.WARNING


# --------------------------------------------------------------------------
# NaN values
# --------------------------------------------------------------------------


def test_complete_data_passes():
    assert nan_values(make_dataset()).status is ValidationStatus.VALID


def test_partial_gaps_are_not_reported():
    values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    values[0, 0, 0] = np.nan
    result = nan_values(make_dataset(variables={"sst": sst_variable(values)}))
    assert result.status is ValidationStatus.VALID
    assert result.details["sst"] > 0


def test_fully_missing_variable_is_an_error():
    values = np.full((TIME.size, LAT.size, LON.size), np.nan)
    result = nan_values(make_dataset(variables={"sst": sst_variable(values)}))
    assert result.status is ValidationStatus.ERROR
    assert any("entirely missing" in f.message for f in result.errors)


def test_mostly_missing_variable_is_a_warning():
    lat = np.arange(30) * 0.25 + 10.0
    values = np.full((TIME.size, lat.size, LON.size), np.nan)
    values[0, 0, 0] = 28.0
    variable = build_variable("sst", values, ("time", "lat", "lon"), units="degC")
    dataset = OceanDataset(
        variables={"sst": variable}, coords={"time": TIME, "lat": lat, "lon": LON}
    )
    result = nan_values(dataset)
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []


def test_infinite_values_are_an_error():
    values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    values[0, 0, 0] = np.inf
    result = nan_values(make_dataset(variables={"sst": sst_variable(values)}))
    assert result.status is ValidationStatus.ERROR
    assert any("infinite" in f.message for f in result.errors)


def test_implausible_values_are_a_warning():
    # SST left in Kelvin but labelled degC.
    values = np.full((TIME.size, LAT.size, LON.size), 301.15)
    result = nan_values(make_dataset(variables={"sst": sst_variable(values)}))
    assert result.status is ValidationStatus.WARNING
    assert any("plausible range" in f.message for f in result.warnings)


def test_nan_warn_threshold_is_configurable():
    values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    values[0] = np.nan
    dataset = make_dataset(variables={"sst": sst_variable(values)})
    assert nan_values(dataset, warn_fraction=0.9).status is ValidationStatus.VALID
    assert nan_values(dataset, warn_fraction=0.1).status is ValidationStatus.WARNING


# --------------------------------------------------------------------------
# Duplicate timestamps
# --------------------------------------------------------------------------


def test_unique_timestamps_pass():
    assert duplicate_timestamps(make_dataset()).status is ValidationStatus.VALID


def test_duplicate_timestamps_are_an_error():
    time = np.array(["2020-01-15", "2020-01-15"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    result = duplicate_timestamps(dataset)
    assert result.status is ValidationStatus.ERROR
    assert result.details["n_duplicated"] == 1
    assert "2020-01-15" in result.errors[0].message


def test_duplicate_timestamps_are_never_only_a_warning():
    # Two ocean states claiming one moment cannot be silently tolerated.
    time = np.array(["2020-01-15", "2020-01-15"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    assert duplicate_timestamps(dataset).warnings == []


def test_static_dataset_skips_the_duplicate_timestamp_check():
    mask = build_variable("land_mask", np.ones((LAT.size, LON.size), dtype=bool), ("lat", "lon"))
    dataset = OceanDataset(variables={"land_mask": mask}, coords={"lat": LAT, "lon": LON})
    assert duplicate_timestamps(dataset).skipped


# --------------------------------------------------------------------------
# Duplicate coordinates
# --------------------------------------------------------------------------


def test_unique_coordinates_pass():
    assert duplicate_coordinates(make_dataset()).status is ValidationStatus.VALID


def test_duplicate_latitudes_are_an_error():
    dataset = make_dataset(
        coords={"time": TIME, "lat": np.array([10.0, 10.0, 10.5]), "lon": LON}
    )
    result = duplicate_coordinates(dataset)
    assert result.status is ValidationStatus.ERROR
    assert result.errors[0].target == "lat"


def test_duplicate_longitudes_are_an_error():
    dataset = make_dataset(coords={"time": TIME, "lat": LAT, "lon": np.array([70.0, 70.0])})
    result = duplicate_coordinates(dataset)
    assert result.status is ValidationStatus.ERROR
    assert result.errors[0].target == "lon"


def test_duplicate_depths_are_an_error():
    result = duplicate_coordinates(subsurface_dataset(np.array([0.0, 50.0, 50.0])))
    assert result.status is ValidationStatus.ERROR
    assert result.errors[0].target == "depth"


def test_time_duplicates_are_not_reported_by_the_coordinate_rule():
    # They belong to duplicate_timestamps, so a report can tell the two apart.
    time = np.array(["2020-01-15", "2020-01-15"], dtype="datetime64[ns]")
    dataset = make_dataset(coords={"time": time, "lat": LAT, "lon": LON})
    assert duplicate_coordinates(dataset).status is ValidationStatus.VALID


# --------------------------------------------------------------------------
# The suite as a whole
# --------------------------------------------------------------------------


def test_all_ten_checks_run_by_default():
    report = validate(make_dataset())
    assert len(report) == 10
    assert [c.name for c in report.checks] == list(ALL_RULES)


def test_every_rule_is_exercised_by_this_module():
    # Guards against a rule being added to RULES without a test here.
    assert set(ALL_RULES) == {
        "latitude_range",
        "longitude_range",
        "resolution",
        "depth_coordinates",
        "time_coordinates",
        "variable_existence",
        "units",
        "nan_values",
        "duplicate_timestamps",
        "duplicate_coordinates",
    }


def test_checks_can_be_selected():
    report = validate(make_dataset(), checks=["units", "nan_values"])
    assert [c.name for c in report.checks] == ["units", "nan_values"]


def test_unknown_check_is_rejected():
    with pytest.raises(ValueError, match="unknown check"):
        validate(make_dataset(), checks=["not_a_rule"])
