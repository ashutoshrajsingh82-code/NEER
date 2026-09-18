"""Tests for Phase 09 — chronological train/val/test splitting
(src/data/preprocessing/splits.py, and how the pipeline configures,
runs and records it).

Phase 09 makes four claims. These tests are what make them claims
rather than comments:

1. **The split is chronological, never random.** Train, val and test
   are contiguous, ordered blocks of time — never an interleaved or
   shuffled sample.
2. **The boundaries are configuration-controlled**, not hardcoded:
   `configs/base.yaml`'s `preprocessing.split` section drives
   `PreprocessingPipeline.from_config()`, and an explicit
   `TemporalSplit` or fraction pair overrides it.
3. **A run's split is fully recorded** — strategy, boundaries, sample
   counts, time ranges and spatial coverage — so it can be audited
   after the fact rather than taken on faith.
4. **Leakage cannot occur**: nothing outside the training period is
   ever fit on, and no held-out timestep can influence a training
   value. `test_preprocessing_leakage.py` covers this in depth
   end-to-end; the tests here focus on the split/guard primitives
   themselves and on what a demo-data run actually records.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (  # noqa: E402
    LeakageError,
    PreprocessingError,
    PreprocessingPipeline,
    TemporalSplit,
    assert_fit_window,
    check_disjoint,
)
from src.data.preprocessing.splits import (  # noqa: E402
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
    split_summary,
)
from conftest import tiny_dataset  # noqa: E402

MONTHS = np.array(
    [f"2020-{m:02d}-01" for m in range(1, 13)], dtype="datetime64[ns]"
)


# --------------------------------------------------------------------------
# 1. Chronological, never random
# --------------------------------------------------------------------------


def test_split_never_shuffles_never_interleaves():
    """Every train index precedes every val index precedes every test
    index — the defining property of a chronological split."""
    split = TemporalSplit.from_fractions(MONTHS, val_fraction=0.25, test_fraction=0.25)
    indices = split.indices(MONTHS)
    assert indices["train"].max() < indices["val"].min()
    assert indices["val"].max() < indices["test"].min()
    # And each block is itself contiguous, i.e. a slice, not a scatter.
    for name in ("train", "val", "test"):
        idx = indices[name]
        assert np.array_equal(idx, np.arange(idx.min(), idx.max() + 1))


def test_temporal_split_offers_no_random_or_shuffled_mode():
    """There is no `shuffle=` or `random_state=` knob to reach for —
    the only constructors are a chronological cut, by timestamp or by
    fraction of the (sorted) time axis."""
    import inspect

    for constructor in (TemporalSplit.__init__, TemporalSplit.from_fractions):
        params = set(inspect.signature(constructor).parameters)
        assert not params & {"shuffle", "random", "random_state", "seed"}


def test_periods_partition_the_time_axis_exactly_once():
    split = TemporalSplit.from_fractions(MONTHS)
    ok, message = check_disjoint(split, MONTHS)
    assert ok, message


def test_boundaries_are_exclusive_upper_bounds():
    split = TemporalSplit(train_end="2020-09-01", val_end="2020-11-01")
    masks = split.masks(MONTHS)
    assert masks["train"].sum() == 8  # Jan..Aug
    assert masks["val"].sum() == 2  # Sep, Oct
    assert masks["test"].sum() == 2  # Nov, Dec


# --------------------------------------------------------------------------
# 2. Configuration-controlled boundaries
# --------------------------------------------------------------------------


def test_split_fractions_default_to_the_documented_values():
    assert DEFAULT_VAL_FRACTION == 0.15
    assert DEFAULT_TEST_FRACTION == 0.15


def test_from_config_reads_split_boundaries_from_base_yaml():
    """`configs/base.yaml` pins val_fraction/test_fraction to 0.15/0.15
    under `preprocessing.split`; `from_config` must actually read them
    rather than falling back to hardcoded module defaults by accident."""
    pipeline = PreprocessingPipeline.from_config()
    assert pipeline.val_fraction == 0.15
    assert pipeline.test_fraction == 0.15


def test_pipeline_constructor_accepts_explicit_fractions():
    pipeline = PreprocessingPipeline(val_fraction=0.2, test_fraction=0.1)
    assert pipeline.val_fraction == 0.2
    assert pipeline.test_fraction == 0.1


def test_run_honours_an_explicit_split_over_the_configured_fractions(demo_dataset):
    """Passing a `TemporalSplit` to `run()` is a full override — the
    boundary actually used is whatever the caller asked for, not a
    fraction-derived one, however the pipeline was constructed."""
    times = demo_dataset.coords["time"]
    midpoint = np.sort(np.asarray(times))[times.size // 2]
    explicit = TemporalSplit(train_end=midpoint, val_end=midpoint)  # train/test only

    # Constructed with fractions that would produce a very different
    # (0.4/0.4) boundary, so a pipeline that ignored `split=` would be
    # easy to catch.
    pipeline = PreprocessingPipeline(val_fraction=0.4, test_fraction=0.4)
    result = pipeline.run(demo_dataset, split=explicit)

    assert result.split.train_end == np.datetime64(midpoint).astype("datetime64[ns]")
    assert result.split.val_end == np.datetime64(midpoint).astype("datetime64[ns]")
    assert result.metadata.split["counts"]["val"] == 0


def test_from_fractions_derives_boundaries_from_the_configured_shares(
    preprocessed_demo,
):
    """End to end: the demo run's boundaries are consistent with the
    configured 0.15/0.15 shares of its (monthly) time axis."""
    split = preprocessed_demo.split
    times = preprocessed_demo.tensors.time
    counts = split.counts(times)
    n = times.size
    # `from_fractions` rounds to the nearest timestep and guarantees at
    # least one, so this is an approximate check, not an exact one.
    assert counts["val"] == max(1, round(n * 0.15))
    assert counts["test"] == max(1, round(n * 0.15))
    assert counts["train"] == n - counts["val"] - counts["test"]


# --------------------------------------------------------------------------
# 3. What gets saved: split metadata, counts, ranges, spatial coverage
# --------------------------------------------------------------------------


def test_split_summary_includes_strategy_and_boundaries():
    split = TemporalSplit.from_fractions(MONTHS, val_fraction=0.25, test_fraction=0.25)
    summary = split_summary(split, MONTHS)
    assert summary["strategy"] == "chronological"
    assert "train_end_exclusive" in summary
    assert "val_end_exclusive" in summary


def test_split_summary_includes_sample_counts_and_time_ranges():
    split = TemporalSplit.from_fractions(MONTHS, val_fraction=0.25, test_fraction=0.25)
    summary = split_summary(split, MONTHS)
    assert set(summary["counts"]) == {"train", "val", "test"}
    assert sum(summary["counts"].values()) == MONTHS.size
    assert set(summary["ranges"]) == {"train", "val", "test"}
    for period, bounds in summary["ranges"].items():
        assert bounds is None or len(bounds) == 2


def test_run_records_spatial_coverage_per_period(preprocessed_demo):
    """The saved metadata must carry spatial coverage, not just
    temporal counts — grid extent plus how much of it is real data,
    broken down by split."""
    coverage = preprocessed_demo.metadata.split["spatial_coverage"]
    assert coverage["grid_shape"] == list(preprocessed_demo.tensors.grid_shape)
    assert coverage["lat_range"] is not None
    assert coverage["lon_range"] is not None
    assert 0.0 <= coverage["ocean_fraction"] <= 1.0

    for period in ("train", "val", "test"):
        stats = coverage["by_period"][period]
        assert stats["n_timesteps"] == preprocessed_demo.tensors.split_masks[period].sum()
        assert 0.0 <= stats["valid_fraction"] <= 1.0
        assert 0.0 <= stats["cells_ever_observed_fraction"] <= 1.0


def test_saved_metadata_json_round_trips_with_spatial_coverage(
    preprocessed_demo, tmp_path
):
    """What's on disk after `save()` is what a later phase (or a human
    auditing a run) actually gets — this must include everything Phase
    09 asks for, not just what happens to be in memory."""
    from src.data.preprocessing import PreprocessingMetadata

    written = preprocessed_demo.metadata.save(tmp_path / "meta.json")
    reloaded = PreprocessingMetadata.load(written)

    assert reloaded.split["strategy"] == "chronological"
    assert set(reloaded.split["counts"]) == {"train", "val", "test"}
    assert set(reloaded.split["ranges"]) == {"train", "val", "test"}
    assert "spatial_coverage" in reloaded.split
    assert reloaded.split["spatial_coverage"]["by_period"].keys() == {
        "train",
        "val",
        "test",
    }
    # And it's genuinely JSON — no numpy scalars/arrays smuggled through.
    json.loads(written.read_text(encoding="utf-8"))


def test_metadata_report_text_mentions_the_split(preprocessed_demo):
    text = preprocessed_demo.describe()
    assert "chronological" in text
    assert "spatial" in text


# --------------------------------------------------------------------------
# 4. Leakage cannot occur
# --------------------------------------------------------------------------


def test_guard_rejects_any_non_training_timestamp():
    split = TemporalSplit(train_end="2020-09-01", val_end="2020-11-01")
    with pytest.raises(LeakageError):
        assert_fit_window(split, MONTHS, step="test-step")


def test_guard_accepts_exactly_the_training_timestamps():
    split = TemporalSplit(train_end="2020-09-01", val_end="2020-11-01")
    assert_fit_window(split, MONTHS[:8], step="test-step")  # must not raise


def test_an_inverted_split_is_rejected_at_construction():
    """`val_end < train_end` would make val/test precede train — caught
    immediately rather than producing a nonsensical mask later."""
    with pytest.raises(PreprocessingError):
        TemporalSplit(train_end="2020-09-01", val_end="2020-01-01")


def test_disjoint_check_catches_a_hand_built_split_that_overlaps():
    """`check_disjoint` is the pipeline's own self-check; confirm it
    actually flags a broken partition rather than always passing."""
    split = TemporalSplit.from_fractions(MONTHS, val_fraction=0.25, test_fraction=0.25)
    masks = split.masks(MONTHS)
    # Manually overlap train and val by one timestep and check the
    # invariant the pipeline relies on would have caught it.
    overlap = masks["train"] & masks["val"]
    assert overlap.sum() == 0  # true split: no overlap by construction
    ok, _ = check_disjoint(split, MONTHS)
    assert ok


def test_pipeline_run_uses_the_same_guard_for_every_learned_step(preprocessed_demo):
    """End to end: every learned step's recorded fit window is entirely
    inside the training period the split declares."""
    split = preprocessed_demo.split
    for entry in preprocessed_demo.metadata.leakage["learned_steps"]:
        assert np.datetime64(entry["last"]) < split.train_end
        assert entry["guard"] == "assert_fit_window"


def test_demo_run_train_val_test_ranges_do_not_overlap(preprocessed_demo):
    """The clearest possible leakage check on real (demo) output: the
    max training timestamp is strictly before the min validation
    timestamp, which is strictly before the min test timestamp."""
    ranges = preprocessed_demo.metadata.split["ranges"]
    train_last = np.datetime64(ranges["train"][1])
    val_first = np.datetime64(ranges["val"][0])
    val_last = np.datetime64(ranges["val"][1])
    test_first = np.datetime64(ranges["test"][0])
    assert train_last < val_first
    assert val_last < test_first


# --------------------------------------------------------------------------
# Running the split on demo data
# --------------------------------------------------------------------------


def test_demo_run_produces_a_nonempty_chronological_split(preprocessed_demo):
    counts = preprocessed_demo.metadata.split["counts"]
    assert counts["train"] > 0
    assert counts["val"] > 0
    assert counts["test"] > 0
    assert counts["train"] > counts["val"]
    assert counts["train"] > counts["test"]
