"""Tests for Phase 24 — baseline models and the common evaluation interface
(src/evaluation/{metrics,interface}.py, src/models/baselines/*).

Phase 24 makes four claims. These tests are what make them claims
rather than comments:

1. **Every baseline shares one evaluation interface.** `ClimatologyBaseline`,
   `RidgeBaseline` and `LightGBMBaseline` all implement `BaselineModel`
   (`fit`/`predict`/`config`) and are scored through the same
   `evaluate_baseline`, which produces the same `EvaluationReport` shape
   for each of them.
2. **Every baseline uses exactly the same train/val/test data.**
   `pool_tensor_bundle` builds every split from one `TensorBundle`'s own
   `split_masks`, and `evaluate_baseline`/`evaluate_baselines` refuse a
   `PooledSplit` dict whose splits trace back to different source files.
3. **Fitting never touches val or test.** A baseline fit on training
   data with one set of val/test targets scores differently than the
   same baseline "fit" by peeking at val — proving `fit` is actually
   train-only, not a no-op that would make every split's score
   identical regardless of what it was "trained" on.
4. **Metrics are computed correctly and consistently**: RMSE/MAE/bias/R^2
   match hand-computed values on masked data, and masked-out entries
   never influence a score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.evaluation.interface import (  # noqa: E402
    PooledSplit,
    SPLIT_NAMES,
    evaluate_baseline,
    evaluate_baselines,
    pool_tensor_bundle,
)
from src.evaluation.metrics import (  # noqa: E402
    bias,
    compute_profile_metrics,
    mae,
    r2,
    rmse,
)
from src.models.baselines import ClimatologyBaseline, RidgeBaseline  # noqa: E402
from src.models.baselines.lightgbm_baseline import is_lightgbm_available  # noqa: E402


# --------------------------------------------------------------------------
# A small, fast, hand-built TensorBundle — no demo-data dependency.
# --------------------------------------------------------------------------


def _tiny_bundle(seed: int = 0, n_time: int = 36, n_lat: int = 3, n_lon: int = 3, n_depth: int = 2):
    rng = np.random.default_rng(seed)
    n_channels = 4
    channel_names = [f"chan_{i}" for i in range(n_channels)]

    time = np.array(
        [np.datetime64("2018-01", "M") + np.timedelta64(i, "M") for i in range(n_time)]
    ).astype("datetime64[ns]")
    lat = np.linspace(5.0, 6.0, n_lat)
    lon = np.linspace(45.0, 46.0, n_lon)
    depth = np.array([0.0, 50.0][:n_depth])

    inputs = rng.normal(0, 1, (n_time, n_channels, n_lat, n_lon)).astype(np.float32)
    input_mask = np.ones_like(inputs, dtype=bool)

    # A seasonal signal (so climatology has something real to learn)
    # plus noise driven partly by the inputs (so Ridge has something to fit).
    months = (time.astype("datetime64[M]").astype(int) % 12) + 1
    seasonal = 15.0 + 5.0 * np.sin(2 * np.pi * months / 12.0)
    targets = np.empty((n_time, n_depth, n_lat, n_lon), dtype=np.float32)
    for d in range(n_depth):
        base = seasonal - 2.0 * d
        signal = base[:, None, None] + 0.5 * inputs[:, 0][:, :, :]
        targets[:, d] = signal + rng.normal(0, 0.1, (n_time, n_lat, n_lon))
    target_mask = np.ones_like(targets, dtype=bool)

    n_val = max(2, n_time // 6)
    n_test = max(2, n_time // 6)
    n_train = n_time - n_val - n_test
    split_masks = {
        "train": np.zeros(n_time, dtype=bool),
        "val": np.zeros(n_time, dtype=bool),
        "test": np.zeros(n_time, dtype=bool),
    }
    split_masks["train"][:n_train] = True
    split_masks["val"][n_train : n_train + n_val] = True
    split_masks["test"][n_train + n_val :] = True

    return TensorBundle(
        inputs=inputs,
        input_mask=input_mask,
        channel_names=channel_names,
        time=time,
        lat=lat,
        lon=lon,
        targets=targets,
        target_mask=target_mask,
        target_names=["subsurface_temp"],
        depth=depth,
        ocean_mask=np.ones((n_lat, n_lon), dtype=bool),
        split_masks=split_masks,
        attrs={"data_mode": "SYNTHETIC_TEST"},
    )


@pytest.fixture
def tiny_bundle():
    return _tiny_bundle()


# --------------------------------------------------------------------------
# 1 & 2. Common interface + shared data
# --------------------------------------------------------------------------


def test_pool_tensor_bundle_uses_the_bundles_own_split_masks(tiny_bundle):
    pooled = pool_tensor_bundle(tiny_bundle)
    assert set(pooled) == set(SPLIT_NAMES)
    for name in SPLIT_NAMES:
        assert pooled[name].n_samples == int(tiny_bundle.split_masks[name].sum())
        assert pooled[name].n_depth == tiny_bundle.targets.shape[1]
        assert pooled[name].channel_names == list(tiny_bundle.channel_names)


def test_pool_tensor_bundle_rejects_an_unknown_split_name(tiny_bundle):
    with pytest.raises(KeyError):
        pool_tensor_bundle(tiny_bundle, splits=["train", "not_a_split"])


def test_every_baseline_produces_the_same_shaped_report(tiny_bundle):
    pooled = pool_tensor_bundle(tiny_bundle, source_tensors_path="tensors.npz")
    models = [ClimatologyBaseline(), RidgeBaseline()]
    reports = evaluate_baselines(models, pooled)

    assert len(reports) == 2
    for report in reports:
        assert set(report.metrics) == set(SPLIT_NAMES)
        assert report.source_tensors_path == "tensors.npz"
        for split_name in SPLIT_NAMES:
            assert report.split_sample_counts[split_name] == pooled[split_name].n_samples
            profile_metrics = report.metrics[split_name]
            assert set(profile_metrics.per_depth) == set(pooled[split_name].depth_names)
            assert profile_metrics.overall.n > 0


def test_evaluate_baseline_refuses_pooled_splits_from_different_sources(tiny_bundle):
    pooled_a = pool_tensor_bundle(tiny_bundle, source_tensors_path="a.npz")
    pooled_b = pool_tensor_bundle(tiny_bundle, source_tensors_path="b.npz")
    mixed = {"train": pooled_a["train"], "val": pooled_b["val"], "test": pooled_a["test"]}

    with pytest.raises(ValueError, match="different tensor bundle"):
        evaluate_baseline(ClimatologyBaseline(), mixed)


def test_predict_shape_mismatch_is_caught(tiny_bundle):
    pooled = pool_tensor_bundle(tiny_bundle)

    class BrokenBaseline(ClimatologyBaseline):
        name = "broken"

        def predict(self, split):
            return super().predict(split)[:, :1]  # wrong depth count

    with pytest.raises(ValueError, match="expected"):
        evaluate_baseline(BrokenBaseline(), pooled)


# --------------------------------------------------------------------------
# 3. Fitting is train-only
# --------------------------------------------------------------------------


def test_climatology_fit_ignores_val_and_test_targets(tiny_bundle):
    pooled = pool_tensor_bundle(tiny_bundle)
    baseline = ClimatologyBaseline().fit(pooled["train"])
    prediction_before = baseline.predict(pooled["val"])

    # Corrupt only the training split's targets and refit; if fit() were
    # accidentally reading from val/test this would leave the prediction
    # on val unchanged, which it must not.
    corrupted_train = PooledSplit(
        features=pooled["train"].features,
        profile=pooled["train"].profile + 100.0,
        profile_mask=pooled["train"].profile_mask,
        months=pooled["train"].months,
        time=pooled["train"].time,
        channel_names=pooled["train"].channel_names,
        depth_names=pooled["train"].depth_names,
        split="train",
    )
    baseline_corrupted = ClimatologyBaseline().fit(corrupted_train)
    prediction_after = baseline_corrupted.predict(pooled["val"])

    assert not np.allclose(prediction_before, prediction_after)
    # And the corruption shows up as roughly a +100 shift, confirming fit()
    # really is reading train.profile and nothing else.
    assert np.allclose(prediction_after - prediction_before, 100.0, atol=1e-3)


def test_ridge_baseline_fits_and_predicts_the_right_shape(tiny_bundle):
    pooled = pool_tensor_bundle(tiny_bundle)
    baseline = RidgeBaseline(alpha=1.0).fit(pooled["train"])
    prediction = baseline.predict(pooled["test"])
    assert prediction.shape == pooled["test"].profile.shape
    assert np.isfinite(prediction).all()


def test_climatology_recovers_the_seasonal_signal_reasonably_well(tiny_bundle):
    """A sanity check that the baseline is actually learning something:
    on data with a real seasonal cycle, the climatology's test-set RMSE
    should be small relative to the cycle's own amplitude."""
    pooled = pool_tensor_bundle(tiny_bundle)
    baseline = ClimatologyBaseline().fit(pooled["train"])
    report = evaluate_baseline(baseline, pooled, fit=False)
    test_rmse = report.metrics["test"].overall.rmse
    assert test_rmse is not None
    assert test_rmse < 5.0  # well inside the +-5degC seasonal amplitude


@pytest.mark.skipif(not is_lightgbm_available(), reason="lightgbm not installed")
def test_lightgbm_baseline_fits_and_predicts_the_right_shape(tiny_bundle):
    from src.models.baselines import LightGBMBaseline

    pooled = pool_tensor_bundle(tiny_bundle)
    baseline = LightGBMBaseline(n_estimators=10).fit(pooled["train"])
    prediction = baseline.predict(pooled["test"])
    assert prediction.shape == pooled["test"].profile.shape
    assert np.isfinite(prediction).all()


def test_ridge_baseline_without_sklearn_raises_missing_dependency(monkeypatch, tiny_bundle):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.linear_model" or name.startswith("sklearn"):
            raise ImportError("simulated missing sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pooled = pool_tensor_bundle(tiny_bundle)

    from src.data.loaders.errors import MissingDependencyError

    with pytest.raises(MissingDependencyError):
        RidgeBaseline().fit(pooled["train"])


# --------------------------------------------------------------------------
# 4. Metrics correctness
# --------------------------------------------------------------------------


def test_rmse_mae_bias_match_hand_computed_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.0, 2.0, 5.0])
    mask = np.array([True, True, True, True])

    errors = y_pred - y_true  # [0.5, 0.0, -1.0, 1.0]
    assert rmse(y_true, y_pred, mask) == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert mae(y_true, y_pred, mask) == pytest.approx(np.mean(np.abs(errors)))
    assert bias(y_true, y_pred, mask) == pytest.approx(np.mean(errors))


def test_masked_out_entries_never_affect_metrics():
    y_true = np.array([1.0, 2.0, 100.0])
    y_pred = np.array([1.0, 2.0, -999.0])
    mask = np.array([True, True, False])

    assert rmse(y_true, y_pred, mask) == pytest.approx(0.0)
    assert mae(y_true, y_pred, mask) == pytest.approx(0.0)
    assert bias(y_true, y_pred, mask) == pytest.approx(0.0)


def test_r2_is_none_for_zero_variance_target():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    mask = np.array([True, True, True])
    assert r2(y_true, y_pred, mask) is None


def test_metrics_are_none_when_nothing_is_valid():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    mask = np.array([False, False])
    assert rmse(y_true, y_pred, mask) is None
    assert mae(y_true, y_pred, mask) is None
    assert bias(y_true, y_pred, mask) is None
    assert r2(y_true, y_pred, mask) is None


def test_compute_profile_metrics_breaks_down_by_depth():
    y_true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    y_pred = np.array([[1.0, 10.0], [2.0, 25.0], [3.0, 30.0]])
    mask = np.ones_like(y_true, dtype=bool)

    result = compute_profile_metrics(y_true, y_pred, mask, depth_names=["0m", "50m"])
    assert result.per_depth["0m"].rmse == pytest.approx(0.0)
    assert result.per_depth["50m"].rmse == pytest.approx(np.sqrt((5.0**2) / 3))
    assert result.overall.n == 6


def test_compute_profile_metrics_rejects_a_shape_mismatch():
    y_true = np.zeros((3, 2))
    y_pred = np.zeros((3, 3))
    mask = np.ones((3, 2), dtype=bool)
    with pytest.raises(ValueError):
        compute_profile_metrics(y_true, y_pred, mask)