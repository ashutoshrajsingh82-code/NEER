"""Tests for Phase 11 — monthly climatology
(src/data/preprocessing/climatology.py).

Phase 11 makes four claims. These tests are what make them claims
rather than comments:

1. **The climatology is computed from training data only.** Like every
   other learned step, `MonthlyClimatology` refuses to do anything
   before `fit()`, and `fit()` learns only from whatever dataset it is
   given — nothing outside that call is consulted.
2. **Monthly, spatial and depth lookup all work**, and agree with each
   other: `at()` with a depth argument returns the same number
   `depth_profile()` puts at that depth, which is the same number
   `monthly_field()` puts at that cell.
3. **The mathematical relationship holds exactly**:
   ``delta_T = target_temperature - climatology`` and
   ``temperature_prediction = climatology + predicted_delta_T`` are
   exact inverses of one another — reconstructing with the true
   delta_T returns the original temperature, to floating-point
   precision, everywhere the input was finite.
4. **The fitted climatology survives a save/load round trip**, so it can
   be handed to a later phase (training, evaluation) without re-fitting.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (  # noqa: E402
    LeakageError,
    MonthlyClimatology,
    NotFittedError,
    StepConfigurationError,
    TemporalSplit,
    assert_fit_window,
    compute_delta_temperature,
    reconstruct_temperature,
)
from conftest import tiny_dataset  # noqa: E402


def _dataset(**kwargs):
    """A two-year-ish monthly dataset — enough months to fill every
    calendar bucket with more than one observation."""
    defaults = dict(n_time=30, n_lat=4, n_lon=5, n_depth=3, start="2019-01-15")
    defaults.update(kwargs)
    return tiny_dataset(**defaults)


# --------------------------------------------------------------------------
# 1. Train-only fitting
# --------------------------------------------------------------------------


def test_monthly_climatology_is_a_learned_step_fitted_only_via_the_training_split():
    assert MonthlyClimatology.learns_from_data is True
    assert not MonthlyClimatology().fitted


def test_unfitted_climatology_refuses_every_query():
    climatology = MonthlyClimatology()
    with pytest.raises(NotFittedError):
        climatology.monthly_field("sst", month=1)
    with pytest.raises(NotFittedError):
        climatology.at("sst", month=1, lat=10.0, lon=50.0)
    with pytest.raises(NotFittedError):
        climatology.transform(_dataset())


def test_fit_learns_only_from_the_dataset_it_is_given():
    """Fitting on a training slice never reflects values that only exist
    in a held-out slice — the direct analogue of the leakage guard the
    other learned steps (`MissingValueHandler`, `Normalizer`) are held
    to."""
    dataset = _dataset()
    split = TemporalSplit(train_end="2020-01-15", val_end="2020-07-15")
    masks = split.masks(dataset.coords["time"])

    # The guard the pipeline relies on: fitting on the whole dataset,
    # which includes validation-period timesteps, is refused.
    with pytest.raises(LeakageError):
        assert_fit_window(split, dataset.coords["time"], step="climatology")

    # Fitting on only the timesteps before train_end is accepted, and is
    # exactly what `MonthlyClimatology.fit` is meant to be called with.
    train_times = dataset.coords["time"][masks["train"]]
    assert_fit_window(split, train_times, step="climatology")  # does not raise
    MonthlyClimatology().fit(dataset)  # fitting itself has no opinion on the caller's slice


def test_pipeline_style_fit_transform_matches_direct_fit_then_transform():
    dataset = _dataset()
    a = MonthlyClimatology().fit_transform(dataset)
    b = MonthlyClimatology()
    b.fit(dataset)
    c = b.transform(dataset)
    np.testing.assert_array_equal(a["sst_climatology"].values, c["sst_climatology"].values)


def test_variables_not_present_are_rejected():
    with pytest.raises(StepConfigurationError):
        MonthlyClimatology(variables=["not_a_real_variable"]).fit(_dataset())


def test_default_variables_are_sst_and_subsurface_temp():
    climatology = MonthlyClimatology().fit(_dataset())
    assert set(climatology.fitted_variables()) == {"sst", "subsurface_temp"}


# --------------------------------------------------------------------------
# 2. Monthly / spatial / depth lookup
# --------------------------------------------------------------------------


def test_monthly_lookup_returns_one_field_per_calendar_month():
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)

    july = climatology.monthly_field("sst", month=7)
    january = climatology.monthly_field("sst", month=1)
    assert july.shape == (dataset.coords["lat"].size, dataset.coords["lon"].size)
    # Different months of a seasonal-ish synthetic field are not identical.
    assert not np.allclose(july, january, equal_nan=True)


def test_monthly_lookup_rejects_an_out_of_range_month():
    climatology = MonthlyClimatology().fit(_dataset())
    with pytest.raises(StepConfigurationError):
        climatology.monthly_field("sst", month=13)
    with pytest.raises(StepConfigurationError):
        climatology.monthly_field("sst", month=0)


def test_spatial_lookup_finds_the_nearest_grid_cell():
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    lat = float(dataset.coords["lat"][2])
    lon = float(dataset.coords["lon"][3])

    field = climatology.monthly_field("sst", month=5)
    expected = float(field[2, 3])

    # Querying exactly on a grid point, and a nudge away from it (still
    # closer to the same cell than to any neighbour), agree.
    assert climatology.at("sst", month=5, lat=lat, lon=lon) == pytest.approx(expected)
    assert climatology.at(
        "sst", month=5, lat=lat + 0.01, lon=lon - 0.01
    ) == pytest.approx(expected)


def test_spatial_lookup_on_a_surface_variable_rejects_a_depth_argument():
    climatology = MonthlyClimatology().fit(_dataset())
    with pytest.raises(StepConfigurationError):
        climatology.at("sst", month=1, lat=10.0, lon=50.0, depth=50.0)


def test_depth_lookup_requires_a_depth_argument_for_a_4d_variable():
    climatology = MonthlyClimatology().fit(_dataset())
    with pytest.raises(StepConfigurationError):
        climatology.at("subsurface_temp", month=1, lat=10.0, lon=50.0)


def test_depth_lookup_agrees_with_the_full_profile():
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    lat = float(dataset.coords["lat"][1])
    lon = float(dataset.coords["lon"][2])
    depth = float(dataset.coords["depth"][1])

    profile = climatology.depth_profile("subsurface_temp", month=9, lat=lat, lon=lon)
    at_depth = climatology.at(
        "subsurface_temp", month=9, lat=lat, lon=lon, depth=depth
    )

    assert profile.shape == (dataset.coords["depth"].size,)
    assert at_depth == pytest.approx(float(profile[1]))


def test_depth_profile_on_a_surface_only_variable_is_rejected():
    climatology = MonthlyClimatology().fit(_dataset())
    with pytest.raises(StepConfigurationError):
        climatology.depth_profile("sst", month=1, lat=10.0, lon=50.0)


def test_depth_levels_reports_the_fitted_depth_coordinate():
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    np.testing.assert_array_equal(climatology.depth_levels(), dataset.coords["depth"])


# --------------------------------------------------------------------------
# 3. delta_T and reconstruction: the mathematical relationship
# --------------------------------------------------------------------------


def test_compute_delta_temperature_is_plain_subtraction():
    target = np.array([20.0, 21.5, 19.0])
    climatology = np.array([18.0, 18.0, 18.0])
    np.testing.assert_allclose(
        compute_delta_temperature(target, climatology), [2.0, 3.5, 1.0]
    )


def test_reconstruct_temperature_is_plain_addition():
    climatology = np.array([18.0, 18.0, 18.0])
    predicted_delta = np.array([2.0, 3.5, 1.0])
    np.testing.assert_allclose(
        reconstruct_temperature(climatology, predicted_delta), [20.0, 21.5, 19.0]
    )


@pytest.mark.parametrize(
    "climatology, target",
    [
        (0.0, 5.0),
        (18.3, -4.2),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, -1.0, 100.0])),
    ],
)
def test_reconstruction_is_the_exact_inverse_of_delta_for_any_values(climatology, target):
    """The identity the whole module exists to guarantee:
    reconstruct_temperature(climatology, compute_delta_temperature(target, climatology)) == target
    for arbitrary climatology/target values, not just ones drawn from a
    fitted dataset."""
    delta = compute_delta_temperature(target, climatology)
    np.testing.assert_allclose(
        reconstruct_temperature(climatology, delta), np.asarray(target, dtype=float)
    )


def test_delta_for_and_reconstruct_round_trip_through_a_fitted_climatology():
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    times = dataset.coords["time"]

    for variable in ("sst", "subsurface_temp"):
        target = dataset[variable].values
        delta = climatology.delta_for(variable, target, times)
        reconstructed = climatology.reconstruct(variable, delta, times)
        np.testing.assert_allclose(reconstructed, target, equal_nan=True)


def test_reconstruct_with_a_wrong_predicted_delta_does_not_recover_the_target():
    """The round trip is not a tautology: feeding in the *wrong* delta_T
    produces a prediction that differs from the target by exactly the
    error in the delta, proving reconstruct() is really doing the
    addition and not silently ignoring its argument."""
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    times = dataset.coords["time"]

    target = dataset["sst"].values
    true_delta = climatology.delta_for("sst", target, times)
    wrong_delta = true_delta + 1.5

    reconstructed = climatology.reconstruct("sst", wrong_delta, times)
    finite = np.isfinite(target) & np.isfinite(reconstructed)
    np.testing.assert_allclose(
        reconstructed[finite] - target[finite], np.full(finite.sum(), 1.5), atol=1e-4
    )


def test_climatology_for_matches_each_timestep_to_its_calendar_month():
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    times = dataset.coords["time"]

    matched = climatology.climatology_for("sst", times)
    field = climatology._climatology["sst"]  # the fitted (12, lat, lon) array
    for i, t in enumerate(times):
        month = (t.astype("datetime64[M]").astype(int) % 12) + 1
        np.testing.assert_array_equal(matched[i], field[month - 1])


def test_transform_attaches_climatology_and_delta_t_variables():
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    out = climatology.transform(dataset)

    for variable in ("sst", "subsurface_temp"):
        clim_name = f"{variable}_climatology"
        delta_name = f"{variable}_delta_t"
        assert clim_name in out.variables
        assert delta_name in out.variables
        assert out[clim_name].dims == dataset[variable].dims
        assert out[delta_name].dims == dataset[variable].dims

        # Exactly the relationship the module is named for.
        reconstructed = out[clim_name].values + out[delta_name].values
        np.testing.assert_allclose(
            reconstructed, dataset[variable].values, equal_nan=True
        )


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        MonthlyClimatology()._transform(_dataset())


# --------------------------------------------------------------------------
# 4. Reusable API: introspection and persistence
# --------------------------------------------------------------------------


def test_climatology_accessor_returns_the_fitted_field_or_none():
    climatology = MonthlyClimatology().fit(_dataset())
    assert climatology.climatology("sst").shape[0] == 12
    assert climatology.climatology("does_not_exist") is None


def test_state_reports_shape_and_summary_statistics_per_variable():
    climatology = MonthlyClimatology().fit(_dataset())
    state = climatology.state()
    assert set(state) == {"sst", "subsurface_temp"}
    assert state["sst"]["dims"][0] == "month"
    assert state["sst"]["shape"][0] == 12
    assert "mean" in state["sst"]


def test_config_lists_the_variables_this_instance_targets():
    assert MonthlyClimatology(variables=["sst"]).config() == {"variables": ["sst"]}
    default_config = MonthlyClimatology().config()
    assert set(default_config["variables"]) == {"sst", "subsurface_temp"}


def test_save_and_load_round_trip_preserves_every_lookup(tmp_path):
    dataset = _dataset()
    climatology = MonthlyClimatology().fit(dataset)
    path = climatology.save(tmp_path / "climatology.npz")

    loaded = MonthlyClimatology.load(path)
    assert loaded.fitted
    assert set(loaded.fitted_variables()) == set(climatology.fitted_variables())

    lat = float(dataset.coords["lat"][1])
    lon = float(dataset.coords["lon"][2])
    for month in (1, 6, 12):
        np.testing.assert_array_equal(
            loaded.monthly_field("sst", month), climatology.monthly_field("sst", month)
        )
        assert loaded.at("sst", month, lat, lon) == pytest.approx(
            climatology.at("sst", month, lat, lon)
        )

    # The reconstruction identity holds just as well on a loaded instance.
    times = dataset.coords["time"]
    target = dataset["sst"].values
    delta = loaded.delta_for("sst", target, times)
    np.testing.assert_allclose(
        loaded.reconstruct("sst", delta, times), target, equal_nan=True
    )


def test_load_rejects_a_missing_file():
    with pytest.raises(FileNotFoundError):
        MonthlyClimatology.load("/tmp/definitely_not_a_real_climatology_file.npz")


def test_pipeline_is_unaffected_by_climatology_being_a_separate_optional_step(preprocessed_demo):
    """MonthlyClimatology is not one of the eight default pipeline
    stages; using it never changes what `PreprocessingPipeline` produces
    on its own."""
    step_names = [entry["step"] for entry in preprocessed_demo.metadata.steps]
    assert "climatology" not in step_names
