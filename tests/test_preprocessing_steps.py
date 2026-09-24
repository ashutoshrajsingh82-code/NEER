from conftest import tiny_dataset
"""Tests for the individual NEER preprocessing steps
(src/data/preprocessing): the step contract, coordinate normalization,
temporal alignment, spatial alignment, missing-value handling, masking,
feature construction and normalization.

Each step is tested on its own, on small synthetic datasets where the
right answer can be written down by hand. Phase 07.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders.representation import OceanDataset, Variable  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    CoordinateNormalizer,
    FeatureBuilder,
    Masker,
    MissingValueHandler,
    NotFittedError,
    Normalizer,
    SpatialAligner,
    StepConfigurationError,
    TemporalAligner,
    interp_axis,
    snap_times,
    to_convention,
)
from src.data.preprocessing.missing import (  # noqa: E402
    OBSERVED_SUFFIX,
    interpolate_along_time,
)



# --------------------------------------------------------------------------
# The step contract
# --------------------------------------------------------------------------


def test_stateless_steps_are_fitted_by_definition():
    assert CoordinateNormalizer().fitted
    assert Masker().fitted
    assert FeatureBuilder().fitted


def test_learned_steps_are_not_fitted_until_fit_is_called():
    assert not MissingValueHandler().fitted
    assert not Normalizer().fitted


def test_learned_step_refuses_to_transform_before_fitting():
    with pytest.raises(NotFittedError):
        Normalizer().transform(tiny_dataset())


def test_steps_never_mutate_their_input():
    dataset = tiny_dataset()
    before = dataset["sst"].values.copy()
    CoordinateNormalizer().transform(dataset)
    Masker().transform(dataset)
    FeatureBuilder().transform(dataset)
    assert np.array_equal(dataset["sst"].values, before)


def test_to_dict_is_json_safe_and_describes_the_step():
    import json

    step = CoordinateNormalizer(lon_convention="-180-180")
    step.transform(tiny_dataset())
    payload = step.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["step"] == "coordinate_normalization"
    assert payload["learns_from_data"] is False
    assert payload["config"]["lon_convention"] == "-180-180"


# --------------------------------------------------------------------------
# Stage 1 â€” coordinate normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "convention,value,expected",
    [
        ("0-360", -10.0, 350.0),
        ("0-360", 45.0, 45.0),
        ("-180-180", 350.0, -10.0),
        ("-180-180", 45.0, 45.0),
        ("keep", -10.0, -10.0),
    ],
)
def test_longitude_conventions(convention, value, expected):
    assert to_convention(np.array([value]), convention)[0] == pytest.approx(expected)


def test_unknown_longitude_convention_is_rejected():
    with pytest.raises(StepConfigurationError):
        CoordinateNormalizer(lon_convention="0-180")


def test_descending_latitude_is_sorted_and_data_follows():
    dataset = tiny_dataset(with_mask=False)
    flipped = replace(
        dataset,
        variables={
            "sst": replace(dataset["sst"], values=dataset["sst"].values[:, ::-1, :]),
            "subsurface_temp": replace(
                dataset["subsurface_temp"],
                values=dataset["subsurface_temp"].values[:, :, ::-1, :],
            ),
        },
        coords={**dataset.coords, "lat": dataset.coords["lat"][::-1]},
    )
    result = CoordinateNormalizer().transform(flipped)

    assert np.all(np.diff(result.coords["lat"]) > 0)
    # Reordering the axis must reorder the data with it, not just the coords.
    assert np.allclose(result["sst"].values, dataset["sst"].values)


def test_duplicate_coordinates_are_dropped_first_wins():
    dataset = tiny_dataset(n_lat=4, with_mask=False)
    coords = dict(dataset.coords)
    coords["lat"] = np.array([5.0, 5.0, 5.5, 5.75])
    duplicated = replace(dataset, coords=coords)
    result = CoordinateNormalizer().transform(duplicated)
    assert result.coords["lat"].size == 3


def test_near_identical_coordinates_are_rounded_together():
    dataset = tiny_dataset(n_lat=2, with_mask=False)
    coords = dict(dataset.coords)
    coords["lat"] = np.array([5.000000000001, 5.25])
    result = CoordinateNormalizer(decimals=6).transform(replace(dataset, coords=coords))
    assert result.coords["lat"][0] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Stage 2 â€” temporal alignment
# --------------------------------------------------------------------------


def test_snap_times_moves_stamps_to_period_start():
    times = np.array(["2021-01-16", "2021-01-02"], dtype="datetime64[ns]")
    snapped = snap_times(times, "monthly")
    assert len(set(snapped.tolist())) == 1


def test_colliding_timesteps_are_aggregated_nan_aware():
    """Two half-covered weeks in one month must average to the mean of what exists."""
    time = np.array(["2021-01-05", "2021-01-20"], dtype="datetime64[ns]")
    values = np.array([[[1.0, np.nan]], [[3.0, 5.0]]])  # (time, lat, lon)
    dataset = OceanDataset(
        variables={"sst": Variable("sst", values, ("time", "lat", "lon"), "degC")},
        coords={"time": time, "lat": np.array([5.0]), "lon": np.array([45.0, 45.25])},
    )
    result = TemporalAligner(frequency="monthly").transform(dataset)

    assert result.n_time == 1
    assert result["sst"].values[0, 0, 0] == pytest.approx(2.0)  # mean(1, 3)
    # A cell missing in one of the two weeks keeps the week it has.
    assert result["sst"].values[0, 0, 1] == pytest.approx(5.0)


def test_gap_months_are_inserted_as_explicit_nan_steps():
    time = np.array(["2021-01-15", "2021-04-15"], dtype="datetime64[ns]")
    values = np.ones((2, 1, 1))
    dataset = OceanDataset(
        variables={"sst": Variable("sst", values, ("time", "lat", "lon"))},
        coords={"time": time, "lat": np.array([5.0]), "lon": np.array([45.0])},
    )
    result = TemporalAligner(frequency="monthly", fill_gaps=True).transform(dataset)

    assert result.n_time == 4  # Jan, Feb, Mar, Apr
    assert np.isnan(result["sst"].values[1:3]).all()


def test_unknown_frequency_is_rejected():
    with pytest.raises(StepConfigurationError):
        TemporalAligner(frequency="fortnightly")


# --------------------------------------------------------------------------
# Stage 3 â€” spatial alignment
# --------------------------------------------------------------------------


def test_interp_axis_linear_midpoint():
    values = np.array([[0.0, 10.0]])
    result = interp_axis(values, 1, np.array([0.0, 1.0]), np.array([0.5]))
    assert result[0, 0] == pytest.approx(5.0)


def test_interpolation_never_invents_a_value_from_a_missing_neighbour():
    values = np.array([[0.0, np.nan]])
    result = interp_axis(values, 1, np.array([0.0, 1.0]), np.array([0.5]))
    assert np.isnan(result[0, 0])


def test_exact_hit_does_not_inherit_a_neighbours_nan():
    values = np.array([[7.0, np.nan]])
    result = interp_axis(values, 1, np.array([0.0, 1.0]), np.array([0.0]))
    assert result[0, 0] == pytest.approx(7.0)


def test_targets_outside_the_source_domain_are_nan():
    values = np.array([[1.0, 2.0]])
    result = interp_axis(values, 1, np.array([0.0, 1.0]), np.array([5.0]))
    assert np.isnan(result[0, 0])


def test_masks_are_resampled_by_nearest_and_stay_boolean():
    values = np.array([[True, False]])
    result = interp_axis(values, 1, np.array([0.0, 1.0]), np.array([0.1]), method="nearest")
    assert result.dtype == bool and result[0, 0]


def test_matching_grid_is_left_untouched():
    """The demo data is already on the target grid; regridding it would be a no-op cost."""
    from src.data.grid import create_ocean_grid

    grid = create_ocean_grid()
    dataset = tiny_dataset(n_lat=len(grid.latitudes), n_lon=len(grid.longitudes), n_time=1)
    dataset = replace(
        dataset, coords={**dataset.coords, "lat": grid.latitudes, "lon": grid.longitudes}
    )
    step = SpatialAligner(grid=grid)
    result = step.transform(dataset)
    assert step.report()["regridded"] is False
    assert result is dataset


# --------------------------------------------------------------------------
# Stage 4 â€” missing-value handling
# --------------------------------------------------------------------------


def test_temporal_interpolation_bridges_short_gaps_only():
    series = np.array([1.0, np.nan, 3.0]).reshape(3, 1, 1)
    filled, count = interpolate_along_time(series, 0, max_gap=2)
    assert count == 1 and filled[1, 0, 0] == pytest.approx(2.0)

    long_gap = np.array([1.0, np.nan, np.nan, np.nan, 5.0]).reshape(5, 1, 1)
    _, count = interpolate_along_time(long_gap, 0, max_gap=2)
    assert count == 0


def test_temporal_interpolation_never_extrapolates():
    series = np.array([np.nan, 2.0, 3.0, np.nan]).reshape(4, 1, 1)
    filled, _ = interpolate_along_time(series, 0, max_gap=2)
    assert np.isnan(filled[0, 0, 0]) and np.isnan(filled[3, 0, 0])


def test_missing_handler_fills_gaps_and_records_what_was_observed():
    dataset = tiny_dataset()
    values = dataset["sst"].values.copy()
    values[2, 1, 1] = np.nan
    dataset = replace(
        dataset, variables={**dataset.variables, "sst": replace(dataset["sst"], values=values)}
    )

    step = MissingValueHandler()
    result = step.fit_transform(dataset)

    assert np.isfinite(result["sst"].values[2, 1, 1])
    observed = result[f"sst{OBSERVED_SUFFIX}"]
    assert observed.values.dtype == bool
    assert not observed.values[2, 1, 1]  # filled, so not observed
    assert observed.values[0, 1, 1]


def test_land_cells_are_not_filled():
    """Land is NaN everywhere; the fill cascade must not reach it."""
    dataset = tiny_dataset()
    values = dataset["sst"].values.copy()
    values[:, 0, 0] = np.nan  # the land cell from tiny_dataset
    dataset = replace(
        dataset, variables={**dataset.variables, "sst": replace(dataset["sst"], values=values)}
    )
    result = MissingValueHandler(ocean_only=True).fit_transform(dataset)
    assert np.isnan(result["sst"].values[:, 0, 0]).all()


def test_missing_handler_must_be_fitted_first():
    with pytest.raises(NotFittedError):
        MissingValueHandler().transform(tiny_dataset())


# --------------------------------------------------------------------------
# Stage 5 â€” masking
# --------------------------------------------------------------------------


def test_masker_publishes_an_ocean_mask():
    from src.data.preprocessing import OCEAN_MASK

    result = Masker().transform(tiny_dataset())
    assert OCEAN_MASK in result
    assert result[OCEAN_MASK].values.dtype == bool


def test_masker_blanks_land_cells():
    result = Masker().transform(tiny_dataset())
    assert np.isnan(result["sst"].values[:, 0, 0]).all()
    assert np.isfinite(result["sst"].values[:, 1, 1]).all()


def test_masker_without_a_mask_variable_is_a_no_op():
    dataset = tiny_dataset(with_mask=False)
    step = Masker()
    result = step.transform(dataset)
    assert "skipped" in step.report()
    assert np.allclose(result["sst"].values, dataset["sst"].values)


# --------------------------------------------------------------------------
# Stage 6 â€” feature construction
# --------------------------------------------------------------------------


def test_cyclic_time_encoding_is_on_the_unit_circle():
    dataset = tiny_dataset()
    result = FeatureBuilder(features=["cyclic_time"]).transform(dataset)
    total = result["time_sin"].values ** 2 + result["time_cos"].values ** 2
    assert np.allclose(total, 1.0, atol=1e-6)


def test_derived_speed_is_the_vector_magnitude():
    dataset = tiny_dataset(with_mask=False)
    shape = dataset["sst"].shape
    extra = {
        "u_current": Variable("u_current", np.full(shape, 3.0), ("time", "lat", "lon"), "m/s"),
        "v_current": Variable("v_current", np.full(shape, 4.0), ("time", "lat", "lon"), "m/s"),
    }
    dataset = replace(dataset, variables={**dataset.variables, **extra})
    result = FeatureBuilder(features=["current_speed"]).transform(dataset)
    assert np.allclose(result["current_speed"].values, 5.0)


def test_features_are_skipped_when_their_inputs_are_absent():
    step = FeatureBuilder(features=["current_speed"])
    result = step.transform(tiny_dataset())
    assert "current_speed" not in result
    assert "current_speed" in step.report()["skipped"]


def test_unknown_feature_is_rejected():
    with pytest.raises(StepConfigurationError):
        FeatureBuilder(features=["sea_surface_vibes"])


def test_target_variables_are_never_used_as_features():
    """Building a feature from the reconstruction target would leak the answer."""
    from src.data.preprocessing.features import TARGET_VARIABLES

    result = FeatureBuilder().transform(tiny_dataset())
    for name, variable in result.variables.items():
        derived = variable.attrs.get("derived_from")
        if derived:
            sources = derived if isinstance(derived, (list, tuple)) else [derived]
            assert not set(sources) & set(TARGET_VARIABLES)


# --------------------------------------------------------------------------
# Stage 7 â€” normalization
# --------------------------------------------------------------------------


def test_zscore_centres_and_scales():
    dataset = tiny_dataset(with_mask=False)
    step = Normalizer(method="zscore", per_depth_level=False)
    result = step.fit_transform(dataset)
    values = result["sst"].values
    assert np.nanmean(values) == pytest.approx(0.0, abs=1e-6)
    assert np.nanstd(values) == pytest.approx(1.0, abs=1e-6)


def test_per_depth_normalization_scales_each_level_independently():
    dataset = tiny_dataset(with_mask=False)
    values = dataset["subsurface_temp"].values.copy()
    values[:, 0] = values[:, 0] * 100.0  # a level with a far larger spread
    dataset = replace(
        dataset,
        variables={
            **dataset.variables,
            "subsurface_temp": replace(dataset["subsurface_temp"], values=values),
        },
    )
    result = Normalizer(per_depth_level=True).fit_transform(dataset)
    per_level = np.nanstd(result["subsurface_temp"].values, axis=(0, 2, 3))
    assert np.allclose(per_level, 1.0, atol=1e-5)


def test_inverse_transform_round_trips_to_physical_units():
    dataset = tiny_dataset(with_mask=False)
    step = Normalizer()
    restored = step.inverse_transform(step.fit_transform(dataset))
    assert np.allclose(restored["sst"].values, dataset["sst"].values, atol=1e-4)


def test_constant_field_does_not_divide_by_zero():
    dataset = tiny_dataset(with_mask=False)
    constant = np.full(dataset["sst"].shape, 7.0)
    dataset = replace(
        dataset,
        variables={**dataset.variables, "sst": replace(dataset["sst"], values=constant)},
    )
    result = Normalizer().fit_transform(dataset)
    assert np.all(np.isfinite(result["sst"].values))


def test_bounded_encodings_are_excluded_from_normalization():
    """time_sin/cos and lat/lon_norm already carry meaning; rescaling them loses it."""
    dataset = FeatureBuilder().transform(tiny_dataset())
    before = dataset["time_sin"].values.copy()
    result = Normalizer().fit_transform(dataset)
    assert np.allclose(result["time_sin"].values, before)


def test_masks_are_never_normalized():
    dataset = tiny_dataset()
    result = Normalizer().fit_transform(dataset)
    assert result["land_mask"].values.dtype == bool

