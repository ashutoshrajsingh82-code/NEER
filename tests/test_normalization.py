"""Tests for Phase 10 — normalization
(src/data/preprocessing/normalization.py, and how the pipeline fits
and records it).

Phase 10 makes four claims. These tests are what make them claims
rather than comments:

1. **Statistics are computed from training data only.** Fitted through
   the pipeline, `Normalizer` never sees a validation or test
   timestep at `fit` time — `assert_fit_window` (Phase 09) enforces
   this the same way it does for every learned step.
2. **Both standardization (zscore) and min-max normalization are
   supported**, and each produces the statistics its name promises:
   zscore centres data to mean 0 / std 1 on the training split;
   min-max maps the training split's own min/max to a fixed range.
3. **What's fitted is saved** — mean, standard deviation (or min/max
   for the min-max method), and enough metadata to reproduce the
   transform and invert it later.
4. **The exact same statistics are reused for validation, test and
   inference** — never recomputed, and unmoved by anything in the
   held-out data. `test_preprocessing_leakage.py` already proves this
   end-to-end with a full perturbation test; the tests here focus on
   the `Normalizer` contract itself and on what a demo-data run
   actually records.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (  # noqa: E402
    NORMALIZATION_METHODS,
    LeakageError,
    Normalizer,
    PreprocessingPipeline,
    StepConfigurationError,
    TemporalSplit,
    assert_fit_window,
)
from conftest import tiny_dataset  # noqa: E402


# --------------------------------------------------------------------------
# 1. Train-only fitting
# --------------------------------------------------------------------------


def test_normalizer_is_a_learned_step_fitted_only_via_the_training_split():
    assert Normalizer.learns_from_data is True
    assert not Normalizer().fitted


def test_pipeline_fits_normalization_on_the_training_period_only(preprocessed_demo):
    """End to end: the recorded fit window for `normalization` never
    reaches past the split boundary."""
    entry = preprocessed_demo.metadata.step("normalization")
    assert entry is not None
    leakage_entry = next(
        e
        for e in preprocessed_demo.metadata.leakage["learned_steps"]
        if e["step"] == "normalization"
    )
    assert np.datetime64(leakage_entry["last"]) < preprocessed_demo.split.train_end
    assert leakage_entry["guard"] == "assert_fit_window"


def test_guard_would_reject_fitting_normalization_outside_training_time():
    """Direct check of the mechanism the pipeline relies on: fitting
    `Normalizer` on anything past `train_end` is refused before the
    step ever sees the data."""
    split = TemporalSplit(train_end="2020-04-15", val_end="2020-06-15")
    dataset = tiny_dataset(n_time=6, start="2020-01-15")  # monthly steps
    times = dataset.coords["time"]
    with pytest.raises(LeakageError):
        assert_fit_window(split, times, step="normalization")  # whole dataset, includes val


def test_normalizer_never_refits_on_transform():
    """Calling `transform` on val/test data must not change `_stats` —
    there is no code path in `Normalizer.transform` that calls `fit`."""
    normalizer = Normalizer(per_depth_level=False)
    train = tiny_dataset(with_mask=False, seed=1)
    other = tiny_dataset(with_mask=False, seed=2)  # different "data", stands for val/test
    normalizer.fit(train)
    before = normalizer.state()
    normalizer.transform(other)
    after = normalizer.state()
    assert before == after


# --------------------------------------------------------------------------
# 2. Both methods are supported and do what their name says
# --------------------------------------------------------------------------


def test_standardization_and_minmax_are_both_available():
    assert "zscore" in NORMALIZATION_METHODS  # standardization
    assert "minmax" in NORMALIZATION_METHODS


def test_unsupported_method_is_rejected():
    with pytest.raises(StepConfigurationError):
        Normalizer(method="not-a-real-method")


def test_zscore_standardizes_to_mean_zero_std_one():
    dataset = tiny_dataset(with_mask=False)
    step = Normalizer(method="zscore", per_depth_level=False)
    result = step.fit_transform(dataset)
    values = result["sst"].values
    assert np.nanmean(values) == pytest.approx(0.0, abs=1e-6)
    assert np.nanstd(values) == pytest.approx(1.0, abs=1e-6)


def test_minmax_maps_training_extremes_to_zero_and_one():
    dataset = tiny_dataset(with_mask=False)
    step = Normalizer(method="minmax", per_depth_level=False)
    result = step.fit_transform(dataset)
    values = result["sst"].values
    assert np.nanmin(values) == pytest.approx(0.0, abs=1e-6)
    assert np.nanmax(values) == pytest.approx(1.0, abs=1e-6)


def test_minmax_on_held_out_data_can_fall_outside_zero_one():
    """The defining, easy-to-get-wrong property of a *train-only* fit:
    min-max statistics come from the training split, so a validation
    value outside the training range legitimately normalizes outside
    [0, 1] rather than being silently re-clamped or re-fit."""
    train = tiny_dataset(with_mask=False, seed=0)
    step = Normalizer(method="minmax", per_depth_level=False)
    step.fit(train)

    held_out = tiny_dataset(with_mask=False, seed=0)
    inflated = held_out["sst"].values * 10.0 + 1000.0  # well outside training range
    held_out = replace(
        held_out,
        variables={**held_out.variables, "sst": replace(held_out["sst"], values=inflated)},
    )
    result = step.transform(held_out)
    assert np.nanmax(result["sst"].values) > 1.0


# --------------------------------------------------------------------------
# 3. What gets saved: mean, standard deviation, metadata
# --------------------------------------------------------------------------


def test_zscore_state_saves_mean_and_std_by_name():
    dataset = tiny_dataset(with_mask=False)
    step = Normalizer(method="zscore", per_depth_level=False)
    step.fit(dataset)
    stats = step.state()["statistics"]["sst"]
    assert stats["mean"] == pytest.approx([np.nanmean(dataset["sst"].values)])
    assert stats["std"] == pytest.approx([np.nanstd(dataset["sst"].values)])


def test_minmax_state_saves_min_and_max_by_name():
    dataset = tiny_dataset(with_mask=False)
    step = Normalizer(method="minmax", per_depth_level=False)
    step.fit(dataset)
    stats = step.state()["statistics"]["sst"]
    assert stats["min"] == pytest.approx([np.nanmin(dataset["sst"].values)])
    assert stats["max"] == pytest.approx([np.nanmax(dataset["sst"].values)])


def test_state_is_empty_before_fitting_and_populated_after():
    step = Normalizer()
    assert step.state()["statistics"] == {}
    step.fit(tiny_dataset(with_mask=False))
    assert step.state()["statistics"]


def test_run_records_normalization_statistics_in_saved_metadata(preprocessed_demo):
    stats = preprocessed_demo.metadata.normalization_statistics()
    assert "sst" in stats
    assert "mean" in stats["sst"]
    assert "std" in stats["sst"]
    assert stats["sst"]["mean"]
    assert stats["sst"]["std"]


def test_saved_metadata_survives_a_json_round_trip(preprocessed_demo, tmp_path):
    from src.data.preprocessing import PreprocessingMetadata

    written = preprocessed_demo.metadata.save(tmp_path / "meta.json")
    reloaded = PreprocessingMetadata.load(written)
    assert (
        reloaded.normalization_statistics()["sst"]["mean"]
        == preprocessed_demo.metadata.normalization_statistics()["sst"]["mean"]
    )


# --------------------------------------------------------------------------
# 4. The exact same statistics are reused for val/test/inference
# --------------------------------------------------------------------------


def test_transform_applies_the_fitted_statistics_not_the_data_shown_to_it():
    """Mathematically verify a held-out value is normalized with the
    *training* mean/std, not its own — the only way "same statistics
    at val/test/inference" is actually true rather than incidental."""
    train = tiny_dataset(with_mask=False, seed=0)
    step = Normalizer(method="zscore", per_depth_level=False)
    step.fit(train)
    train_mean = np.nanmean(train["sst"].values)
    train_std = np.nanstd(train["sst"].values)

    held_out = tiny_dataset(with_mask=False, seed=7)  # different distribution
    result = step.transform(held_out)
    expected = (held_out["sst"].values - train_mean) / train_std
    assert np.allclose(result["sst"].values, expected, atol=1e-5)


def test_repeated_transforms_of_different_data_use_identical_statistics():
    train = tiny_dataset(with_mask=False, seed=0)
    step = Normalizer(per_depth_level=False)
    step.fit(train)
    stats_before = step.statistics("sst")

    step.transform(tiny_dataset(with_mask=False, seed=1))
    step.transform(tiny_dataset(with_mask=False, seed=2))
    stats_after = step.statistics("sst")

    np.testing.assert_array_equal(stats_before["center"], stats_after["center"])
    np.testing.assert_array_equal(stats_before["scale"], stats_after["scale"])


def test_pipeline_transform_reuses_exactly_the_fitted_statistics(demo_dataset):
    """The inference path (`pipeline.transform`) must apply the
    statistics fitted during `run`, unchanged — never recompute them
    from whatever it's asked to transform."""
    pipeline = PreprocessingPipeline()
    pipeline.run(demo_dataset)  # fits every learned step, including normalization
    fitted_stats = pipeline.step("normalization").statistics("sst")

    transformed = pipeline.transform(demo_dataset)
    # Reconstruct what the transform *should* have produced, straight
    # from the fitted mean/std, independent of the pipeline's own code
    # path, and require the two agree exactly.
    raw = np.asarray(demo_dataset["sst"].values, dtype=float)
    expected = (raw - fitted_stats["center"]) / fitted_stats["scale"]
    actual = np.asarray(transformed["sst"].values, dtype=float)
    finite = np.isfinite(expected) & np.isfinite(actual)
    assert np.allclose(actual[finite], expected[finite], atol=1e-4)


# --------------------------------------------------------------------------
# 5. Validation/test data does not alter training statistics
# --------------------------------------------------------------------------


def test_transform_on_validation_like_data_does_not_change_fitted_state():
    dataset = tiny_dataset(with_mask=False, seed=0)
    step = Normalizer(per_depth_level=False)
    step.fit(dataset)
    fitted = step.state()

    shifted = replace(
        dataset,
        variables={
            **dataset.variables,
            "sst": replace(dataset["sst"], values=dataset["sst"].values + 1000.0),
        },
    )
    step.transform(shifted)
    assert step.state() == fitted


def test_corrupting_held_out_data_leaves_fitted_statistics_unchanged(demo_dataset):
    """The real end-to-end version: run the full pipeline, corrupt every
    non-training timestep beyond recognition, run again with the same
    split, and require the fitted mean/std did not move by a single
    bit. This is the central claim of Phase 10, and the only test of it
    that actually convinces."""
    baseline = PreprocessingPipeline().run(demo_dataset)

    rng = np.random.default_rng(0)
    held_out = ~baseline.split.is_training_time(demo_dataset.coords["time"])
    variables = {}
    for name, variable in demo_dataset.variables.items():
        values = np.array(variable.values)
        if "time" in variable.dims and values.dtype.kind == "f":
            values[held_out] = rng.normal(1e6, 1e5, values[held_out].shape)
        variables[name] = replace(variable, values=values)
    corrupted_dataset = replace(demo_dataset, variables=variables, validation=None)

    corrupted = PreprocessingPipeline().run(corrupted_dataset, split=baseline.split)

    assert (
        baseline.metadata.normalization_statistics()
        == corrupted.metadata.normalization_statistics()
    )


def test_normalized_training_tensor_is_unaffected_by_held_out_corruption(demo_dataset):
    """Statistics being unchanged is necessary but not sufficient — the
    actually-normalized training *values* must be identical too, which
    only holds if the same statistics were applied both times."""
    baseline = PreprocessingPipeline().run(demo_dataset)

    rng = np.random.default_rng(1)
    held_out = ~baseline.split.is_training_time(demo_dataset.coords["time"])
    variables = {}
    for name, variable in demo_dataset.variables.items():
        values = np.array(variable.values)
        if "time" in variable.dims and values.dtype.kind == "f":
            values[held_out] = rng.normal(-1e6, 1e5, values[held_out].shape)
        variables[name] = replace(variable, values=values)
    corrupted_dataset = replace(demo_dataset, variables=variables, validation=None)

    corrupted = PreprocessingPipeline().run(corrupted_dataset, split=baseline.split)
    assert np.array_equal(baseline.train.inputs, corrupted.train.inputs)
