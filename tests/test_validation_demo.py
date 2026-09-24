from conftest import corrupt
"""
Validation tests against the demo dataset â€” clean and deliberately corrupted.

Phase 06 asks for both halves, and the pairing is what makes the layer
trustworthy:

* the real demo dataset must validate cleanly (no errors, no warnings), so
  the rules are not producing noise on known-good data, and
* the same dataset, damaged in one specific way, must be caught on the
  right check with the right severity â€” so the rules are not merely
  passing everything.

`conftest.corrupt` builds each damaged variant from copies, never touching
the shared session dataset, and the no-mutation guarantee is asserted
directly at the end of this module.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



from src.data.loaders import load_demo_dataset
from src.data.validation import (
    DataValidator,
    SURFACE_VARIABLES,
    ValidationError,
    ValidationReport,
    ValidationStatus,
    is_valid,
    validate,
    validate_or_raise,
)

# --------------------------------------------------------------------------
# The clean demo dataset
# --------------------------------------------------------------------------


def test_demo_dataset_is_fully_valid(demo_dataset):
    report = validate(demo_dataset)
    assert report.status is ValidationStatus.VALID, report.to_text()
    assert report.is_valid and report.is_usable


def test_demo_dataset_produces_no_warnings(demo_dataset):
    # Known-good data must not generate noise, or real findings get ignored.
    report = validate(demo_dataset)
    assert report.warnings == []


def test_every_check_runs_on_the_demo_dataset(demo_dataset):
    report = validate(demo_dataset)
    assert len(report) == 10
    assert all(c.status is ValidationStatus.VALID for c in report.checks)


def test_demo_dataset_satisfies_the_surface_variable_contract(demo_dataset):
    report = validate(demo_dataset, required_variables=SURFACE_VARIABLES)
    assert report.check("variable_existence").status is ValidationStatus.VALID


def test_demo_dataset_matches_the_configured_grid(demo_dataset):
    details = validate(demo_dataset).check("resolution").details
    assert details["lat_resolution"] == pytest.approx(0.25)
    assert details["lon_resolution"] == pytest.approx(0.25)


def test_demo_dataset_matches_the_configured_depths(demo_dataset):
    from src.utils.config import load_config

    details = validate(demo_dataset).check("depth_coordinates").details
    assert details["n_levels"] == len(load_config("base").depths)


def test_demo_gappiness_does_not_trigger_warnings(demo_dataset):
    # The demo's ~76% subsurface sparsity is realistic, not a defect.
    result = validate(demo_dataset).check("nan_values")
    assert result.status is ValidationStatus.VALID
    assert result.details["subsurface_temp"] > 0.5


def test_validate_or_raise_passes_on_clean_demo_data(demo_dataset):
    report = validate_or_raise(demo_dataset, strict=True)
    assert report.is_valid


def test_is_valid_helper(demo_dataset):
    assert is_valid(demo_dataset) is True


# --------------------------------------------------------------------------
# Corrupted demo data: each case breaks exactly one thing
# --------------------------------------------------------------------------


def test_corrupted_latitude_range_is_caught(demo_dataset):
    broken = corrupt(demo_dataset, coords={"lat": demo_dataset.lat + 75.0})
    report = validate(broken)
    assert report.status is ValidationStatus.ERROR
    assert report.check("latitude_range").status is ValidationStatus.ERROR


def test_corrupted_longitude_range_is_caught(demo_dataset):
    broken = corrupt(demo_dataset, coords={"lon": demo_dataset.lon * 10.0})
    assert validate(broken).check("longitude_range").status is ValidationStatus.ERROR


def test_flipped_latitude_axis_is_caught(demo_dataset):
    broken = corrupt(demo_dataset, coords={"lat": demo_dataset.lat[::-1]})
    result = validate(broken).check("latitude_range")
    assert result.status is ValidationStatus.ERROR
    assert any("descending" in f.message for f in result.errors)


def test_corrupted_resolution_is_caught(demo_dataset):
    jittered = demo_dataset.lat.copy()
    jittered[5] += 0.1  # one displaced grid line makes the axis uneven
    report = validate(corrupt(demo_dataset, coords={"lat": jittered}))
    assert report.check("resolution").status is ValidationStatus.ERROR


def test_coarsened_grid_warns_rather_than_erroring(demo_dataset):
    coarse = demo_dataset.lat[::4]  # a regular 1.0Â° axis
    values = demo_dataset["sst"].values[:, ::4, :]
    broken = corrupt(demo_dataset, coords={"lat": coarse}, values={"sst": values},
                     drop=tuple(n for n in demo_dataset.variable_names if n != "sst"))
    result = validate(broken).check("resolution")
    assert result.status is ValidationStatus.WARNING
    assert result.errors == []


def test_corrupted_depth_coordinates_are_caught(demo_dataset):
    broken = corrupt(demo_dataset, coords={"depth": -demo_dataset.depth})
    result = validate(broken).check("depth_coordinates")
    assert result.status is ValidationStatus.ERROR
    assert any("non-negative" in f.message for f in result.errors)


def test_shuffled_depth_axis_is_caught(demo_dataset):
    broken = corrupt(demo_dataset, coords={"depth": demo_dataset.depth[::-1]})
    assert validate(broken).check("depth_coordinates").status is ValidationStatus.ERROR


def test_corrupted_time_coordinates_are_caught(demo_dataset):
    broken = corrupt(
        demo_dataset,
        coords={"time": np.arange(demo_dataset.n_time, dtype=np.float64)},
    )
    result = validate(broken).check("time_coordinates")
    assert result.status is ValidationStatus.ERROR
    assert any("datetime64" in f.message for f in result.errors)


def test_unsorted_time_axis_is_caught(demo_dataset):
    broken = corrupt(demo_dataset, coords={"time": demo_dataset.time[::-1]})
    assert validate(broken).check("time_coordinates").status is ValidationStatus.ERROR


def test_missing_required_variable_is_caught(demo_dataset):
    broken = corrupt(demo_dataset, drop=("sss",))
    report = validate(broken, required_variables=SURFACE_VARIABLES)
    result = report.check("variable_existence")
    assert result.status is ValidationStatus.ERROR
    assert result.details["missing"] == ["sss"]


def test_corrupted_units_are_caught(demo_dataset):
    broken = corrupt(demo_dataset, units={"sst": "psu"})
    result = validate(broken).check("units")
    assert result.status is ValidationStatus.ERROR
    assert any("contradicts" in f.message for f in result.errors)


def test_stripped_units_are_a_warning(demo_dataset):
    broken = corrupt(demo_dataset, units={"sst": None})
    report = validate(broken)
    assert report.check("units").status is ValidationStatus.WARNING
    assert report.is_usable  # a warning must not block the pipeline


def test_fully_missing_variable_is_caught(demo_dataset):
    blanked = np.full_like(demo_dataset["sst"].values, np.nan)
    report = validate(corrupt(demo_dataset, values={"sst": blanked}))
    result = report.check("nan_values")
    assert result.status is ValidationStatus.ERROR
    assert any(f.target == "sst" for f in result.errors)


def test_kelvin_values_labelled_celsius_are_caught(demo_dataset):
    # The corruption the unit check cannot see: the label is right, the
    # numbers are not. Caught by the physical-range check instead.
    shifted = demo_dataset["sst"].values + 273.15
    result = validate(corrupt(demo_dataset, values={"sst": shifted})).check("nan_values")
    assert result.status is ValidationStatus.WARNING
    assert any("plausible range" in f.message for f in result.warnings)


def test_infinities_are_caught(demo_dataset):
    values = demo_dataset["sla"].values.copy()
    values[0, 0, 0] = np.inf
    report = validate(corrupt(demo_dataset, values={"sla": values}))
    assert report.check("nan_values").status is ValidationStatus.ERROR


def test_duplicate_timestamps_are_caught(demo_dataset):
    time = demo_dataset.time.copy()
    time[1] = time[0]
    report = validate(corrupt(demo_dataset, coords={"time": time}))
    result = report.check("duplicate_timestamps")
    assert result.status is ValidationStatus.ERROR
    assert result.details["n_duplicated"] == 1


def test_duplicate_coordinates_are_caught(demo_dataset):
    lat = demo_dataset.lat.copy()
    lat[1] = lat[0]
    result = validate(corrupt(demo_dataset, coords={"lat": lat})).check("duplicate_coordinates")
    assert result.status is ValidationStatus.ERROR
    assert result.errors[0].target == "lat"


def test_several_corruptions_are_all_reported(demo_dataset):
    # One report must surface every problem, not stop at the first.
    time = demo_dataset.time.copy()
    time[1] = time[0]
    broken = corrupt(
        demo_dataset,
        coords={"time": time, "lat": demo_dataset.lat + 75.0},
        units={"sst": "psu"},
    )
    report = validate(broken)
    failed = {c.name for c in report.checks_by_status(ValidationStatus.ERROR)}
    assert {"latitude_range", "units", "duplicate_timestamps"} <= failed


# --------------------------------------------------------------------------
# Failure behaviour
# --------------------------------------------------------------------------


def test_validate_never_raises_on_bad_data(demo_dataset):
    # Reporting and enforcing are separate decisions.
    broken = corrupt(demo_dataset, coords={"lat": demo_dataset.lat + 75.0})
    report = validate(broken)
    assert isinstance(report, ValidationReport)
    assert report.has_errors


def test_validate_or_raise_raises_on_errors(demo_dataset):
    broken = corrupt(demo_dataset, coords={"lat": demo_dataset.lat + 75.0})
    with pytest.raises(ValidationError) as excinfo:
        validate_or_raise(broken)
    assert excinfo.value.report.has_errors


def test_strict_mode_raises_on_warnings(demo_dataset):
    broken = corrupt(demo_dataset, units={"sst": None})
    validate_or_raise(broken)  # tolerated by default
    with pytest.raises(ValidationError):
        validate_or_raise(broken, strict=True)


def test_error_message_lists_every_failure(demo_dataset):
    broken = corrupt(demo_dataset, coords={"lat": demo_dataset.lat + 75.0}, units={"sst": "psu"})
    with pytest.raises(ValidationError) as excinfo:
        validate_or_raise(broken)
    message = str(excinfo.value)
    assert "lat" in message and "sst" in message


# --------------------------------------------------------------------------
# No silent repair
# --------------------------------------------------------------------------


def test_validation_never_modifies_the_dataset(demo_dataset):
    """The central Phase 06 guarantee: report, never repair."""
    before_lat = demo_dataset.lat.copy()
    before_time = demo_dataset.time.copy()
    before_sst = demo_dataset["sst"].values.copy()
    before_units = demo_dataset["sst"].units

    broken = corrupt(
        demo_dataset,
        coords={"lat": demo_dataset.lat[::-1]},
        units={"sst": "psu"},
    )
    validate(broken)
    validate(demo_dataset)

    assert np.array_equal(demo_dataset.lat, before_lat)
    assert np.array_equal(demo_dataset.time, before_time)
    assert np.array_equal(demo_dataset["sst"].values, before_sst, equal_nan=True)
    assert demo_dataset["sst"].units == before_units


def test_corrupted_dataset_stays_corrupted_after_validation(demo_dataset):
    flipped = demo_dataset.lat[::-1]
    broken = corrupt(demo_dataset, coords={"lat": flipped})
    validate(broken)
    assert np.array_equal(broken.lat, flipped)  # not quietly re-sorted


def test_serious_problems_are_reported_with_remedies_not_fixes(demo_dataset):
    broken = corrupt(demo_dataset, coords={"lat": demo_dataset.lat[::-1]})
    report = validate(broken)
    remedies = report.remedies()
    assert remedies and any("sort" in r for r in remedies)
    # The suggestion exists; the data is untouched.
    assert np.array_equal(broken.lat, demo_dataset.lat[::-1])


# --------------------------------------------------------------------------
# Report artifact
# --------------------------------------------------------------------------


def test_report_is_json_serializable(demo_dataset):
    payload = json.loads(validate(demo_dataset).to_json())
    assert payload["status"] == "valid"
    assert len(payload["checks"]) == 10


def test_report_records_what_was_validated(demo_dataset):
    report = validate(demo_dataset)
    assert report.dataset_source == demo_dataset.source
    assert report.dataset_summary["is_synthetic"] is True
    assert "sst" in report.dataset_summary["variables"]


def test_report_text_lists_every_check(demo_dataset):
    text = validate(demo_dataset).to_text()
    for title in ("Latitude range", "Grid resolution", "Duplicate timestamps"):
        assert title in text


def test_report_markdown_includes_findings_and_remedies(demo_dataset):
    broken = corrupt(demo_dataset, units={"sst": "psu"})
    markdown = validate(broken).to_markdown()
    assert "# NEER data validation report" in markdown
    assert "ERROR" in markdown
    assert "Suggested fixes" in markdown
    assert "never modifies data" in markdown


def test_report_saves_in_the_format_the_extension_implies(demo_dataset, tmp_path):
    report = validate(demo_dataset)
    assert json.loads(report.save(tmp_path / "r.json").read_text())["status"] == "valid"
    assert report.save(tmp_path / "r.md").read_text().startswith("# NEER")
    assert "NEER data validation report" in report.save(tmp_path / "r.txt").read_text()


def test_unknown_check_lookup_lists_what_ran(demo_dataset):
    with pytest.raises(KeyError, match="ran:"):
        validate(demo_dataset).check("no_such_check")


def test_summary_line_reflects_the_worst_status(demo_dataset):
    broken = corrupt(demo_dataset, coords={"lat": demo_dataset.lat + 75.0})
    assert validate(broken).summary_line().startswith("ERROR")
    assert validate(demo_dataset).summary_line().startswith("VALID")


# --------------------------------------------------------------------------
# DataValidator
# --------------------------------------------------------------------------


def test_validator_applies_its_defaults(demo_dataset):
    validator = DataValidator(required_variables=SURFACE_VARIABLES)
    broken = corrupt(demo_dataset, drop=("sla",))
    assert validator.validate(demo_dataset).is_valid
    assert validator.validate(broken).has_errors


def test_validator_overrides_beat_its_defaults(demo_dataset):
    validator = DataValidator(required_variables=SURFACE_VARIABLES)
    broken = corrupt(demo_dataset, drop=("sla",))
    assert validator.validate(broken, required_variables=["sst"]).is_valid


def test_validator_rejects_unknown_options():
    with pytest.raises(TypeError, match="unknown validator option"):
        DataValidator(nonsense=True)


def test_validator_can_raise(demo_dataset):
    validator = DataValidator(required_variables=SURFACE_VARIABLES)
    with pytest.raises(ValidationError):
        validator.validate_or_raise(corrupt(demo_dataset, drop=("sla",)))


# --------------------------------------------------------------------------
# Interaction with the loaders' own structural validation
# --------------------------------------------------------------------------


def test_scientific_layer_agrees_with_the_loader_gate_on_clean_data():
    dataset = load_demo_dataset()  # loader validation already passed here
    assert dataset.validation.ok
    assert validate(dataset).is_usable


def test_scientific_layer_catches_what_the_loader_gate_permits(demo_dataset):
    # A 1Â° grid loads fine structurally but is not the configured grid;
    # only the scientific layer has an opinion about that.
    coarse_lat = demo_dataset.lat[::4]
    values = demo_dataset["sst"].values[:, ::4, :]
    broken = corrupt(
        demo_dataset,
        coords={"lat": coarse_lat},
        values={"sst": values},
        drop=tuple(n for n in demo_dataset.variable_names if n != "sst"),
    )
    from src.data.loaders.validation import validate_dataset as structural_validate

    assert structural_validate(broken, check_domain=False).ok
    assert validate(broken).check("resolution").status is ValidationStatus.WARNING

