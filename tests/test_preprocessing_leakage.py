from conftest import tiny_dataset
"""Tests for the NEER preprocessing leakage guarantees
(src/data/preprocessing/splits.py and the pipeline's fit/transform
discipline). Phase 07.

Phase 07 makes two claims. These tests are what make them claims rather
than comments:

1. **Nothing is fitted outside the training period.** Tested directly on
   `assert_fit_window`, and end to end by checking the fit windows the
   pipeline recorded.
2. **Nothing is transformed across a period boundary.** Tested by the
   perturbation test below: corrupt every held-out timestep beyond
   recognition, re-run, and require that not one number in the training
   tensors moved. If any stage reached across the boundary â€” a temporal
   interpolation bridging the last training month from the first
   validation month, a statistic pooled over everything â€” that test goes
   red.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (  # noqa: E402
    LeakageError,
    MissingValueHandler,
    Normalizer,
    PreprocessingError,
    PreprocessingPipeline,
    TemporalSplit,
    assert_fit_window,
    check_disjoint,
)
from src.data.preprocessing.missing import interpolate_along_time  # noqa: E402


MONTHS = np.array(
    ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"], dtype="datetime64[ns]"
)


# --------------------------------------------------------------------------
# The split itself
# --------------------------------------------------------------------------


def test_split_is_chronological_never_random():
    """Ocean fields are autocorrelated in time; a random split scores fiction."""
    split = TemporalSplit.from_fractions(MONTHS, val_fraction=0.25, test_fraction=0.25)
    indices = split.indices(MONTHS)
    assert indices["train"].max() < indices["val"].min()
    assert indices["val"].max() < indices["test"].min()


def test_periods_partition_the_time_axis_exactly_once():
    split = TemporalSplit.from_fractions(MONTHS)
    ok, message = check_disjoint(split, MONTHS)
    assert ok, message


def test_boundaries_are_exclusive_upper_bounds():
    split = TemporalSplit(train_end="2020-03-01", val_end="2020-04-01")
    masks = split.masks(MONTHS)
    assert masks["train"].tolist() == [True, True, False, False]
    assert masks["val"].tolist() == [False, False, True, False]
    assert masks["test"].tolist() == [False, False, False, True]


def test_fractions_always_leave_training_data():
    split = TemporalSplit.from_fractions(MONTHS, val_fraction=0.45, test_fraction=0.45)
    assert split.counts(MONTHS)["train"] >= 1


def test_an_inverted_split_is_rejected():
    with pytest.raises(PreprocessingError):
        TemporalSplit(train_end="2020-06-01", val_end="2020-01-01")


def test_split_survives_a_round_trip_through_its_dict_form():
    split = TemporalSplit.from_fractions(MONTHS)
    assert TemporalSplit.from_dict(
        {
            "train_end": split.train_end,
            "val_end": split.val_end,
        }
    ) == split


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


def test_guard_accepts_training_timestamps():
    split = TemporalSplit(train_end="2020-03-01", val_end="2020-04-01")
    assert_fit_window(split, MONTHS[:2], step="test")  # must not raise


def test_guard_rejects_a_validation_timestamp():
    split = TemporalSplit(train_end="2020-03-01", val_end="2020-04-01")
    with pytest.raises(LeakageError):
        assert_fit_window(split, MONTHS, step="test")


def test_guard_names_the_offending_timestamps():
    split = TemporalSplit(train_end="2020-03-01", val_end="2020-04-01")
    try:
        assert_fit_window(split, MONTHS, step="normalization")
    except LeakageError as error:
        assert len(error.offending_times) == 2
        assert "normalization" in str(error)
    else:  # pragma: no cover
        pytest.fail("expected a LeakageError")


def test_guard_rejects_an_empty_fit_window():
    """Fitting on nothing is a configuration error, not a silent no-op."""
    split = TemporalSplit(train_end="2019-01-01", val_end="2019-06-01")
    with pytest.raises(LeakageError):
        assert_fit_window(split, np.array([], dtype="datetime64[ns]"), step="test")


# --------------------------------------------------------------------------
# End to end: the fit window the pipeline actually used
# --------------------------------------------------------------------------


def test_learned_steps_saw_only_training_timestamps(preprocessed_demo):
    split = preprocessed_demo.split
    for entry in preprocessed_demo.metadata.leakage["learned_steps"]:
        assert np.datetime64(entry["last"]) < split.train_end


def test_every_step_is_declared_as_learned_or_stateless(preprocessed_demo):
    leakage = preprocessed_demo.metadata.leakage
    declared = {e["step"] for e in leakage["learned_steps"]} | set(
        leakage["stateless_steps"]
    )
    assert declared == {s["step"] for s in preprocessed_demo.metadata.steps[:-1]}


def test_metadata_records_that_periods_were_transformed_apart(preprocessed_demo):
    assert preprocessed_demo.metadata.leakage["periods_transformed_independently"] == [
        "train",
        "val",
        "test",
    ]


# --------------------------------------------------------------------------
# The perturbation test
# --------------------------------------------------------------------------


def _corrupt_held_out(dataset, split, seed=0):
    """Replace every non-training timestep with nonsense, leaving training intact."""
    rng = np.random.default_rng(seed)
    held_out = ~split.is_training_time(dataset.coords["time"])
    variables = {}
    for name, variable in dataset.variables.items():
        values = np.array(variable.values)
        if "time" in variable.dims and values.dtype.kind == "f":
            values[held_out] = values[held_out] * -3.0 + rng.normal(
                50.0, 10.0, values[held_out].shape
            )
        variables[name] = replace(variable, values=values)
    return replace(dataset, variables=variables, validation=None)


def test_corrupting_held_out_data_changes_nothing_in_training(demo_dataset):
    """The central claim of this phase, tested the only way that convinces.

    If any stage pooled statistics across periods, or interpolated across
    a split boundary, the training tensors would move when the held-out
    data changed. They must not move at all.
    """
    baseline = PreprocessingPipeline().run(demo_dataset)
    corrupted = PreprocessingPipeline().run(
        _corrupt_held_out(demo_dataset, baseline.split), split=baseline.split
    )

    assert np.array_equal(baseline.train.inputs, corrupted.train.inputs)
    assert np.array_equal(baseline.train.targets, corrupted.train.targets)
    assert np.array_equal(baseline.train.target_mask, corrupted.train.target_mask)


def test_corrupting_held_out_data_changes_nothing_in_the_fitted_statistics(
    demo_dataset,
):
    baseline = PreprocessingPipeline().run(demo_dataset)
    corrupted = PreprocessingPipeline().run(
        _corrupt_held_out(demo_dataset, baseline.split), split=baseline.split
    )
    assert (
        baseline.metadata.normalization_statistics()
        == corrupted.metadata.normalization_statistics()
    )


def test_the_perturbation_test_has_teeth(demo_dataset):
    """Guard against the previous test passing because the corruption was inert."""
    baseline = PreprocessingPipeline().run(demo_dataset)
    corrupted = PreprocessingPipeline().run(
        _corrupt_held_out(demo_dataset, baseline.split), split=baseline.split
    )
    assert not np.array_equal(baseline.test.inputs, corrupted.test.inputs)


# --------------------------------------------------------------------------
# The specific trap: interpolating across a boundary
# --------------------------------------------------------------------------


def test_temporal_interpolation_would_bridge_a_boundary_if_periods_were_not_split():
    """Documents *why* the periods are transformed apart.

    Run over a whole record, the gap at index 1 is filled from index 2.
    If index 2 belonged to the validation period, that value would now be
    inside a training cell. Splitting first is what prevents it.
    """
    series = np.array([1.0, np.nan, 3.0, 4.0]).reshape(4, 1, 1)
    filled, count = interpolate_along_time(series, 0, max_gap=2)
    assert count == 1 and filled[1, 0, 0] == pytest.approx(2.0)

    # The same gap, with the record cut at the boundary first: nothing to
    # interpolate between, so it is left for the climatology instead.
    train_only = series[:2]
    _, count = interpolate_along_time(train_only, 0, max_gap=2)
    assert count == 0


def test_pipeline_does_not_bridge_a_gap_across_the_split_boundary(demo_dataset):
    """End to end: a hole in the last training month must not be filled
    from the first validation month."""
    split = TemporalSplit.from_fractions(demo_dataset.coords["time"])
    times = demo_dataset.coords["time"]
    last_train = int(np.flatnonzero(split.train_mask(times))[-1])

    values = np.array(demo_dataset["sst"].values)
    ocean = demo_dataset["land_mask"].values
    # Find an ocean cell observed either side of the boundary, then hole it.
    candidates = np.argwhere(
        ocean
        & np.isfinite(values[last_train])
        & np.isfinite(values[last_train + 1])
    )
    if not len(candidates):  # pragma: no cover - demo data always has some
        pytest.skip("no cell observed on both sides of the split boundary")
    row, col = candidates[0]
    values[last_train, row, col] = np.nan

    holed = replace(
        demo_dataset,
        variables={
            **demo_dataset.variables,
            "sst": replace(demo_dataset["sst"], values=values),
        },
        validation=None,
    )
    result = PreprocessingPipeline().run(holed, split=split)

    # The cell was filled (tensors are finite), but not from across the
    # boundary: it is marked unobserved, and the pipeline reports the
    # periods were handled separately.
    assert np.isfinite(result.tensors.channel("sst")[last_train, row, col])
    sst_index = result.tensors.channel_names.index("sst")
    assert not result.tensors.input_mask[last_train, sst_index, row, col]


# --------------------------------------------------------------------------
# Refusing to learn at transform time
# --------------------------------------------------------------------------


def test_normalizer_does_not_refit_when_shown_new_data():
    """Recomputing statistics at transform time is how test data leaks in."""
    dataset = tiny_dataset(with_mask=False)
    step = Normalizer()
    step.fit(dataset)
    fitted = step.state()

    shifted = replace(
        dataset,
        variables={
            **dataset.variables,
            "sst": replace(dataset["sst"], values=dataset["sst"].values + 100.0),
        },
    )
    step.transform(shifted)
    assert step.state() == fitted


def test_missing_handler_does_not_refit_its_climatology_at_transform_time():
    dataset = tiny_dataset()
    step = MissingValueHandler()
    step.fit(dataset)
    fitted = step.state()
    step.transform(tiny_dataset(seed=99))
    assert step.state() == fitted

