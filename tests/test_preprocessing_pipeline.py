"""Tests for the NEER preprocessing pipeline as a whole
(src/data/preprocessing/pipeline.py, tensors.py, metadata.py): ordering,
the split, tensor shapes and masks, metadata persistence, and the
inference path. Phase 07.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (  # noqa: E402
    METADATA_FILENAME,
    TENSOR_FILENAME,
    CoordinateNormalizer,
    Normalizer,
    PreprocessingError,
    PreprocessingMetadata,
    PreprocessingPipeline,
    TemporalSplit,
    TensorAssembler,
    TensorBundle,
    default_steps,
)
from src.data.preprocessing.pipeline import concat_along_time  # noqa: E402
from conftest import tiny_dataset  # noqa: E402


# --------------------------------------------------------------------------
# Construction and ordering
# --------------------------------------------------------------------------


def test_default_pipeline_runs_the_seven_stages_in_order():
    assert [step.name for step in default_steps()] == [
        "coordinate_normalization",
        "temporal_alignment",
        "spatial_alignment",
        "missing_values",
        "masking",
        "feature_construction",
        "normalization",
    ]


def test_default_steps_accepts_overrides_without_restating_the_rest():
    steps = default_steps(normalization=Normalizer(method="robust"))
    assert len(steps) == 7
    assert steps[-1].method == "robust"


def test_overriding_an_unknown_step_is_rejected():
    with pytest.raises(PreprocessingError):
        default_steps(dimensionality_reduction=CoordinateNormalizer())


def test_from_config_builds_the_pipeline_declared_in_configs():
    pipeline = PreprocessingPipeline.from_config()
    assert [s.name for s in pipeline.steps] == [s.name for s in default_steps()]
    assert pipeline.step("normalization").method == "zscore"


def test_pipeline_is_not_fitted_until_it_runs():
    assert not PreprocessingPipeline().fitted


def test_learned_steps_are_exactly_the_two_that_hold_statistics():
    names = [s.name for s in PreprocessingPipeline().learned_steps]
    assert names == ["missing_values", "normalization"]


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def test_run_produces_tensors_for_every_period(preprocessed_demo):
    result = preprocessed_demo
    assert result.train.n_time + result.val.n_time + result.test.n_time == (
        result.tensors.n_time
    )
    assert result.train.n_time > result.test.n_time  # most of the record is training


def test_input_tensor_has_the_expected_rank_and_named_channels(preprocessed_demo):
    tensors = preprocessed_demo.tensors
    assert tensors.inputs.ndim == 4  # (time, channel, lat, lon)
    assert len(tensors.channel_names) == tensors.n_channels
    assert len(set(tensors.channel_names)) == tensors.n_channels  # no duplicates
    assert tensors.inputs.dtype == np.float32


def test_target_tensor_matches_the_depth_axis(preprocessed_demo):
    tensors = preprocessed_demo.tensors
    assert tensors.targets.shape[0] == tensors.n_time
    assert tensors.targets.shape[1] == tensors.depth.size
    assert tensors.targets.shape[2:] == tensors.grid_shape


def test_tensors_are_finite_everywhere(preprocessed_demo):
    """A model cannot consume NaN; masks carry the missingness instead."""
    assert np.all(np.isfinite(preprocessed_demo.tensors.inputs))
    assert np.all(np.isfinite(preprocessed_demo.tensors.targets))


def test_target_mask_is_sparser_than_the_grid(preprocessed_demo):
    """Subsurface coverage is sparse; a mask that is all True would mean it was lost."""
    mask = preprocessed_demo.tensors.target_mask
    assert mask.dtype == bool
    assert 0.0 < mask.mean() < 1.0


def test_land_is_never_marked_valid(preprocessed_demo):
    tensors = preprocessed_demo.tensors
    land = ~tensors.ocean_mask
    assert not tensors.input_mask[:, :, land].any()


def test_channel_lookup_returns_that_channels_field(preprocessed_demo):
    tensors = preprocessed_demo.tensors
    sst = tensors.channel("sst")
    assert sst.shape == (tensors.n_time, *tensors.grid_shape)
    with pytest.raises(KeyError):
        tensors.channel("not_a_channel")


def test_split_masks_travel_with_the_tensors(preprocessed_demo):
    masks = preprocessed_demo.tensors.split_masks
    assert set(masks) == {"train", "val", "test"}
    stacked = np.vstack([masks[p] for p in ("train", "val", "test")])
    assert np.all(stacked.sum(axis=0) == 1)  # each timestep in exactly one period


def test_periods_are_chronological_and_do_not_overlap(preprocessed_demo):
    train, val, test = (
        preprocessed_demo.train,
        preprocessed_demo.val,
        preprocessed_demo.test,
    )
    assert train.time.max() < val.time.min()
    assert val.time.max() < test.time.min()


def test_time_axis_is_ascending_after_periods_are_rejoined(preprocessed_demo):
    time = preprocessed_demo.tensors.time
    assert np.all(np.diff(time) > np.timedelta64(0, "ns"))


def test_synthetic_provenance_survives_the_whole_pipeline(preprocessed_demo):
    """Demo tensors must never be mistakable for real observations."""
    assert preprocessed_demo.tensors.is_synthetic
    assert preprocessed_demo.metadata.is_synthetic
    assert any("SYNTHETIC" in note for note in preprocessed_demo.metadata.notes)


def test_run_does_not_mutate_the_input_dataset(demo_dataset):
    before = demo_dataset["sst"].values.copy()
    PreprocessingPipeline().run(demo_dataset)
    assert np.array_equal(demo_dataset["sst"].values, before, equal_nan=True)


def test_an_explicit_split_is_honoured(demo_dataset):
    split = TemporalSplit(train_end="2021-01-01", val_end="2021-06-01")
    result = PreprocessingPipeline().run(demo_dataset, split=split)
    assert result.train.time.max() < np.datetime64("2021-01-01")
    assert result.val.time.min() >= np.datetime64("2021-01-01")


# --------------------------------------------------------------------------
# Rejoining periods
# --------------------------------------------------------------------------


def test_concat_along_time_restores_the_original_dataset():
    dataset = tiny_dataset(n_time=6)
    from src.data.preprocessing._utils import select_times

    mask = np.zeros(6, dtype=bool)
    mask[:4] = True
    rejoined = concat_along_time(
        [select_times(dataset, mask), select_times(dataset, ~mask)]
    )
    assert np.array_equal(rejoined.coords["time"], dataset.coords["time"])
    assert np.allclose(rejoined["sst"].values, dataset["sst"].values)


def test_concat_rejects_periods_that_disagree_on_a_static_field():
    dataset = tiny_dataset(n_time=2)
    other = replace(
        dataset,
        variables={
            **dataset.variables,
            "land_mask": replace(
                dataset["land_mask"], values=~dataset["land_mask"].values
            ),
        },
    )
    with pytest.raises(PreprocessingError):
        concat_along_time([dataset, other])


# --------------------------------------------------------------------------
# The inference path
# --------------------------------------------------------------------------


def test_transform_reuses_fitted_parameters_and_learns_nothing(demo_dataset):
    pipeline = PreprocessingPipeline()
    pipeline.run(demo_dataset)
    before = pipeline.step("normalization").state()

    pipeline.transform(demo_dataset)
    assert pipeline.step("normalization").state() == before


def test_transform_before_run_raises_rather_than_refitting():
    from src.data.preprocessing import NotFittedError

    with pytest.raises(NotFittedError):
        PreprocessingPipeline().transform(tiny_dataset())


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def test_metadata_records_every_step_in_order(preprocessed_demo):
    recorded = [entry["step"] for entry in preprocessed_demo.metadata.steps]
    assert recorded[:7] == [s.name for s in default_steps()]
    assert recorded[-1] == "tensor_assembly"


def test_metadata_records_the_fit_window_for_learned_steps(preprocessed_demo):
    leakage = preprocessed_demo.metadata.leakage
    fitted = {entry["step"] for entry in leakage["learned_steps"]}
    assert fitted == {"missing_values", "normalization"}
    for entry in leakage["learned_steps"]:
        assert entry["guard"] == "assert_fit_window"


def test_metadata_keeps_normalization_statistics_for_inverting_predictions(
    preprocessed_demo,
):
    stats = preprocessed_demo.metadata.normalization_statistics()
    assert "subsurface_temp" in stats
    entry = stats["subsurface_temp"]
    assert len(entry["center"]) == len(entry["scale"]) > 1  # per depth level


def test_metadata_is_json_serializable(preprocessed_demo):
    json.loads(preprocessed_demo.metadata.to_json())


def test_metadata_round_trips_through_disk(preprocessed_demo, tmp_path):
    path = preprocessed_demo.metadata.save(tmp_path / METADATA_FILENAME)
    reloaded = PreprocessingMetadata.load(path)
    assert reloaded.to_dict() == preprocessed_demo.metadata.to_dict()


def test_saving_writes_tensors_metadata_and_a_readable_report(
    preprocessed_demo, tmp_path
):
    written = preprocessed_demo.save(tmp_path)
    assert written["tensors"].name == TENSOR_FILENAME
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0
    assert "SYNTHETIC" in written["report"].read_text(encoding="utf-8")


def test_tensor_bundle_round_trips_through_disk(preprocessed_demo, tmp_path):
    path = preprocessed_demo.tensors.save(tmp_path / "tensors.npz")
    reloaded = TensorBundle.load(path)
    assert reloaded.channel_names == preprocessed_demo.tensors.channel_names
    assert np.array_equal(reloaded.inputs, preprocessed_demo.tensors.inputs)
    assert np.array_equal(reloaded.target_mask, preprocessed_demo.tensors.target_mask)
    assert set(reloaded.split_masks) == {"train", "val", "test"}


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


def test_a_dataset_without_a_time_axis_is_rejected():
    dataset = tiny_dataset()
    stripped = replace(
        dataset,
        variables={"land_mask": dataset["land_mask"]},
        coords={"lat": dataset.coords["lat"], "lon": dataset.coords["lon"]},
    )
    with pytest.raises(PreprocessingError):
        PreprocessingPipeline().run(stripped)


def test_assembler_rejects_a_dataset_with_no_usable_channels():
    dataset = tiny_dataset(with_mask=False).subset(["subsurface_temp"])
    with pytest.raises(PreprocessingError):
        TensorAssembler().assemble(dataset)
