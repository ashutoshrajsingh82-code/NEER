"""Tests for Phase 25 — the evaluation engine
(`src.evaluation.metrics.pearson`/`per_variable`, `src.evaluation.plots`).

Phase 25 makes three claims on top of Phase 24's shared evaluation
interface. These tests are what make them claims rather than comments:

1. **Pearson correlation is a real, independent metric.** It matches a
   hand/numpy-computed correlation, is `None` exactly when it is
   mathematically undefined (fewer than 2 points, or zero variance in
   either series), and — the whole reason Phase 25 reports it alongside
   `r2` rather than instead of it — is invariant to a bias shift or a
   positive rescaling that *does* move `r2`.
2. **Per-variable metrics pool the right columns and nothing else.**
   `compute_profile_metrics(..., variable_names=...)` groups depth
   columns by variable correctly, stays empty for a single-variable
   profile (nothing a variable breakdown would add over `per_depth`
   there), and `pool_tensor_bundle` derives `variable_names` from
   `TensorBundle.target_names` the same way `TensorAssembler` lays
   multi-variable targets out (variable 0's depths, then variable 1's).
3. **The plotting module produces real, non-empty depth plots** for
   every one of the four Phase 25 metrics, without needing a display,
   and refuses an unknown metric name outright rather than drawing
   something meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.evaluation.interface import (  # noqa: E402
    SPLIT_NAMES,
    evaluate_baselines,
    pool_tensor_bundle,
)
from src.evaluation.metrics import (  # noqa: E402
    compute_profile_metrics,
    pearson,
    r2,
)
from src.evaluation.plots import METRIC_LABELS, plot_all_metrics, plot_metric_vs_depth  # noqa: E402
from src.models.baselines import ClimatologyBaseline, RidgeBaseline  # noqa: E402


# --------------------------------------------------------------------------
# Shared fixture — a small, fast, hand-built TensorBundle, optionally with
# two stacked target variables (mirrors tests/test_baselines.py's
# _tiny_bundle, extended with `n_variables` for the per-variable tests).
# --------------------------------------------------------------------------


def _tiny_bundle(
    seed: int = 0,
    n_time: int = 36,
    n_lat: int = 3,
    n_lon: int = 3,
    n_depth: int = 2,
    n_variables: int = 1,
):
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

    months = (time.astype("datetime64[M]").astype(int) % 12) + 1
    seasonal = 15.0 + 5.0 * np.sin(2 * np.pi * months / 12.0)

    all_target_names = ["subsurface_temp", "salinity"][:n_variables]
    var_blocks = []
    for v in range(n_variables):
        block = np.empty((n_time, n_depth, n_lat, n_lon), dtype=np.float32)
        for d in range(n_depth):
            base = seasonal - 2.0 * d - 10.0 * v  # each variable on its own level
            signal = base[:, None, None] + 0.5 * inputs[:, 0][:, :, :]
            block[:, d] = signal + rng.normal(0, 0.1, (n_time, n_lat, n_lon))
        var_blocks.append(block)
    targets = np.concatenate(var_blocks, axis=1) if n_variables > 1 else var_blocks[0]
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
        target_names=all_target_names,
        depth=depth,
        ocean_mask=np.ones((n_lat, n_lon), dtype=bool),
        split_masks=split_masks,
        attrs={"data_mode": "SYNTHETIC_TEST", "is_synthetic": True},
    )


@pytest.fixture
def tiny_bundle():
    return _tiny_bundle()


@pytest.fixture
def tiny_bundle_two_variables():
    return _tiny_bundle(n_variables=2)


# --------------------------------------------------------------------------
# 1. Pearson correlation
# --------------------------------------------------------------------------


def test_pearson_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.2, size=50)
    mask = np.ones_like(y_true, dtype=bool)

    expected = np.corrcoef(y_true, y_pred)[0, 1]
    assert pearson(y_true, y_pred, mask) == pytest.approx(expected)


def test_pearson_is_none_when_undefined():
    # Fewer than 2 valid points.
    assert pearson(np.array([1.0]), np.array([1.0]), np.array([True])) is None
    # Zero variance in the target.
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    mask = np.array([True, True, True])
    assert pearson(y_true, y_pred, mask) is None
    # Zero variance in the prediction.
    assert pearson(y_pred, y_true, mask) is None
    # Nothing valid at all.
    mask_none = np.array([False, False, False])
    assert pearson(y_true, y_pred, mask_none) is None


def test_pearson_ignores_masked_out_entries():
    y_true = np.array([1.0, 2.0, 3.0, 999.0])
    y_pred = np.array([1.1, 2.1, 2.9, -999.0])
    mask = np.array([True, True, True, False])

    full_true, full_pred = y_true[:3], y_pred[:3]
    expected = np.corrcoef(full_true, full_pred)[0, 1]
    assert pearson(y_true, y_pred, mask) == pytest.approx(expected)


def test_pearson_differs_from_r2_under_bias_and_scale():
    """The whole point of reporting `pearson` alongside `r2`: a
    prediction that tracks the target perfectly up to a bias shift and a
    rescale has `pearson` == 1.0 but a poor `r2`."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=30)
    y_pred = 3.0 * y_true + 10.0  # perfectly correlated, badly scaled/biased
    mask = np.ones_like(y_true, dtype=bool)

    p = pearson(y_true, y_pred, mask)
    r_squared = r2(y_true, y_pred, mask)
    assert p == pytest.approx(1.0, abs=1e-9)
    assert r_squared is not None
    assert r_squared < 0.5  # badly off in magnitude despite perfect correlation


# --------------------------------------------------------------------------
# 2. Per-variable metrics
# --------------------------------------------------------------------------


def test_per_variable_pools_the_right_columns():
    # 4 depth columns: first two are "temp", last two are "salinity".
    y_true = np.array(
        [
            [1.0, 2.0, 100.0, 200.0],
            [1.0, 2.0, 100.0, 200.0],
            [3.0, 4.0, 300.0, 400.0],
        ]
    )
    y_pred = np.array(
        [
            [1.0, 2.0, 110.0, 200.0],
            [1.0, 2.0, 100.0, 210.0],
            [3.0, 4.0, 300.0, 400.0],
        ]
    )
    mask = np.ones_like(y_true, dtype=bool)
    variable_names = ["temp", "temp", "salinity", "salinity"]

    result = compute_profile_metrics(
        y_true, y_pred, mask, depth_names=["0m", "50m", "0m", "50m"], variable_names=variable_names
    )

    assert set(result.per_variable) == {"temp", "salinity"}
    # "temp" columns are predicted perfectly -> zero error.
    assert result.per_variable["temp"].rmse == pytest.approx(0.0)
    assert result.per_variable["temp"].n == 6
    # "salinity" columns have two small errors (10.0 each) among 6 entries.
    assert result.per_variable["salinity"].n == 6
    assert result.per_variable["salinity"].rmse > 0.0
    # Per-variable pooling shouldn't disturb the pre-existing per-depth /
    # overall breakdown.
    assert set(result.per_depth) == {"0m", "50m"}
    assert result.overall.n == 12


def test_per_variable_is_empty_for_a_single_variable_profile():
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    y_pred = y_true.copy()
    mask = np.ones_like(y_true, dtype=bool)

    # No variable_names given at all.
    result = compute_profile_metrics(y_true, y_pred, mask, depth_names=["0m", "50m"])
    assert result.per_variable == {}

    # variable_names given but all identical -> still nothing to add.
    result = compute_profile_metrics(
        y_true, y_pred, mask, depth_names=["0m", "50m"], variable_names=["temp", "temp"]
    )
    assert result.per_variable == {}


def test_per_variable_rejects_a_length_mismatch():
    y_true = np.zeros((3, 2))
    y_pred = np.zeros((3, 2))
    mask = np.ones((3, 2), dtype=bool)
    with pytest.raises(ValueError, match="variable_names"):
        compute_profile_metrics(y_true, y_pred, mask, variable_names=["only_one"])


def test_to_dict_omits_per_variable_key_when_empty_but_includes_it_when_present():
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    mask = np.ones_like(y_true, dtype=bool)

    single = compute_profile_metrics(y_true, y_true, mask, depth_names=["0m", "50m"])
    assert "per_variable" not in single.to_dict()

    multi = compute_profile_metrics(
        y_true, y_true, mask, depth_names=["0m", "50m"], variable_names=["temp", "salinity"]
    )
    assert "per_variable" in multi.to_dict()
    assert set(multi.to_dict()["per_variable"]) == {"temp", "salinity"}


def test_pool_tensor_bundle_derives_variable_names_from_target_names(tiny_bundle):
    pooled = pool_tensor_bundle(tiny_bundle)
    for name in SPLIT_NAMES:
        # Single-variable bundle: every depth column labeled the same.
        assert pooled[name].variable_names == ["subsurface_temp"] * pooled[name].n_depth


def test_pool_tensor_bundle_variable_names_for_a_multi_variable_bundle(
    tiny_bundle_two_variables,
):
    bundle = tiny_bundle_two_variables
    pooled = pool_tensor_bundle(bundle)
    n_depth_per_var = bundle.targets.shape[1] // 2
    expected = ["subsurface_temp"] * n_depth_per_var + ["salinity"] * n_depth_per_var
    for name in SPLIT_NAMES:
        assert pooled[name].variable_names == expected


def test_evaluate_baselines_report_per_variable_metrics_for_a_multi_variable_bundle(
    tiny_bundle_two_variables,
):
    pooled = pool_tensor_bundle(tiny_bundle_two_variables, source_tensors_path="tensors.npz")
    reports = evaluate_baselines([ClimatologyBaseline(), RidgeBaseline()], pooled)
    for report in reports:
        for split_name in SPLIT_NAMES:
            per_variable = report.metrics[split_name].per_variable
            assert set(per_variable) == {"subsurface_temp", "salinity"}
            assert per_variable["subsurface_temp"].n > 0
            assert per_variable["salinity"].n > 0


# --------------------------------------------------------------------------
# 3. Plots
# --------------------------------------------------------------------------


def _profile_metrics_for(tiny_bundle_fixture, model):
    pooled = pool_tensor_bundle(tiny_bundle_fixture)
    from src.evaluation.interface import evaluate_baseline

    report = evaluate_baseline(model, pooled)
    return pooled["test"].depth_names, report.metrics["test"]


def test_plot_all_metrics_writes_one_png_per_metric(tmp_path, tiny_bundle):
    depth_names, profile_metrics = _profile_metrics_for(tiny_bundle, ClimatologyBaseline())
    saved = plot_all_metrics(
        depth_names, {"climatology": profile_metrics}, tmp_path, split="test"
    )

    assert set(saved) == set(METRIC_LABELS)
    for metric, path in saved.items():
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.parent == tmp_path


def test_plot_all_metrics_overlays_multiple_models(tmp_path, tiny_bundle):
    depth_names, clim_metrics = _profile_metrics_for(tiny_bundle, ClimatologyBaseline())
    _, ridge_metrics = _profile_metrics_for(tiny_bundle, RidgeBaseline())

    saved = plot_all_metrics(
        depth_names,
        {"climatology": clim_metrics, "ridge": ridge_metrics},
        tmp_path,
        split="test",
    )
    for path in saved.values():
        assert path.exists() and path.stat().st_size > 0


def test_plot_metric_vs_depth_rejects_an_unknown_metric(tiny_bundle):
    depth_names, profile_metrics = _profile_metrics_for(tiny_bundle, ClimatologyBaseline())
    with pytest.raises(ValueError, match="unknown metric"):
        plot_metric_vs_depth("not_a_metric", depth_names, {"climatology": profile_metrics})


def test_plot_marks_synthetic_data_in_the_title(tmp_path, tiny_bundle):
    depth_names, profile_metrics = _profile_metrics_for(tiny_bundle, ClimatologyBaseline())
    fig = plot_metric_vs_depth(
        "rmse", depth_names, {"climatology": profile_metrics}, is_synthetic=True
    )
    assert "SYNTHETIC" in fig.axes[0].get_title()

    fig_real = plot_metric_vs_depth(
        "rmse", depth_names, {"climatology": profile_metrics}, is_synthetic=False
    )
    assert "SYNTHETIC" not in fig_real.axes[0].get_title()