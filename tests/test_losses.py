"""Tests for the Phase 19 NEER loss functions (`src/training/losses.py`).

Same two-group layout as the other test files in this suite: a handful
of config-level checks that don't need torch, then the full numeric
suite skipped via `pytest.importorskip` when torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.losses import DEFAULT_DEPTH_DIM, DEFAULT_SPATIAL_DIMS, NEERLossConfig  # noqa: E402

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_config_defaults():
    cfg = NEERLossConfig()
    assert cfg.mse_weight == 1.0
    assert cfg.smoothness_weight == 0.0
    assert cfg.gradient_weight == 0.0
    assert cfg.anomaly_weight == 0.0
    assert cfg.depth_dim == DEFAULT_DEPTH_DIM == -3
    assert cfg.spatial_dims == DEFAULT_SPATIAL_DIMS == (-2, -1)


def test_config_weights_are_configurable():
    cfg = NEERLossConfig(
        mse_weight=2.0, smoothness_weight=0.1, gradient_weight=0.05, anomaly_weight=0.2
    )
    assert (cfg.mse_weight, cfg.smoothness_weight, cfg.gradient_weight, cfg.anomaly_weight) == (
        2.0,
        0.1,
        0.05,
        0.2,
    )


def test_config_rejects_non_positive_mse_weight():
    with pytest.raises(ValueError, match="mse_weight"):
        NEERLossConfig(mse_weight=0.0)
    with pytest.raises(ValueError, match="mse_weight"):
        NEERLossConfig(mse_weight=-1.0)


@pytest.mark.parametrize(
    "field", ["smoothness_weight", "gradient_weight", "anomaly_weight"]
)
def test_config_rejects_negative_optional_weights(field):
    with pytest.raises(ValueError, match=field):
        NEERLossConfig(**{field: -0.1})


def test_config_allows_zero_optional_weights():
    # Zero disables a term's contribution to `total` but is still valid
    # (the term is still computed and returned for logging).
    cfg = NEERLossConfig(smoothness_weight=0.0, gradient_weight=0.0, anomaly_weight=0.0)
    assert cfg.smoothness_weight == cfg.gradient_weight == cfg.anomaly_weight == 0.0


def test_config_rejects_empty_spatial_dims():
    with pytest.raises(ValueError, match="spatial_dims"):
        NEERLossConfig(spatial_dims=())


def test_config_replace():
    cfg = NEERLossConfig()
    updated = cfg.replace(smoothness_weight=0.3)
    assert updated.smoothness_weight == 0.3
    assert cfg.smoothness_weight == 0.0  # original untouched (frozen dataclass)


# --------------------------------------------------------------------------
# Numeric tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.training.losses import (  # noqa: E402
    NEERLoss,
    anomaly_loss,
    gradient_loss,
    masked_mse_loss,
    vertical_smoothness_loss,
)


def _grid(batch=2, depth=4, lat=5, lon=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(batch, depth, lat, lon, generator=g)
    target = torch.randn(batch, depth, lat, lon, generator=g)
    mask = torch.rand(batch, depth, lat, lon, generator=g) > 0.3
    return pred, target, mask


def _profile(batch=3, depth=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(batch, depth, generator=g)
    target = torch.randn(batch, depth, generator=g)
    mask = torch.rand(batch, depth, generator=g) > 0.3
    return pred, target, mask


# -- masked_mse_loss --------------------------------------------------------


def test_masked_mse_matches_manual_computation_on_full_grid():
    pred, target, mask = _grid()
    got = masked_mse_loss(pred, target, mask)
    mask_f = mask.float()
    expected = ((pred - target) ** 2 * mask_f).sum() / mask_f.sum()
    assert torch.allclose(got, expected, atol=1e-6)


def test_masked_mse_unmasked_equals_plain_mse():
    pred, target, _ = _grid()
    got = masked_mse_loss(pred, target, mask=None)
    expected = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(got, expected, atol=1e-6)


def test_masked_mse_is_zero_for_perfect_prediction():
    pred, target, mask = _grid()
    assert torch.allclose(masked_mse_loss(target, target, mask), torch.tensor(0.0))


def test_masked_mse_all_invalid_mask_is_zero_not_nan():
    pred, target, _ = _grid()
    mask = torch.zeros_like(pred, dtype=torch.bool)
    loss = masked_mse_loss(pred, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_masked_mse_land_cells_do_not_affect_the_loss():
    # A huge error confined entirely to masked-out ("land") cells must
    # not move the loss at all versus the same tensors with those
    # cells zeroed out instead.
    pred, target, mask = _grid()
    corrupted_pred = pred.clone()
    corrupted_pred[~mask] += 1000.0
    assert torch.allclose(
        masked_mse_loss(pred, target, mask), masked_mse_loss(corrupted_pred, target, mask)
    )


def test_masked_mse_shape_mismatch_raises():
    pred = torch.randn(2, 3)
    target = torch.randn(2, 4)
    with pytest.raises(ValueError, match="shape"):
        masked_mse_loss(pred, target)


def test_masked_mse_gradient_only_flows_through_valid_cells():
    pred, target, mask = _grid()
    pred = pred.clone().requires_grad_(True)
    loss = masked_mse_loss(pred, target, mask)
    loss.backward()
    assert torch.all(pred.grad[~mask] == 0.0)
    assert torch.any(pred.grad[mask] != 0.0)


# -- vertical_smoothness_loss ------------------------------------------------


def test_smoothness_is_zero_for_a_constant_profile():
    pred = torch.ones(2, 6, 3, 3) * 5.0
    assert torch.allclose(vertical_smoothness_loss(pred), torch.tensor(0.0))


def test_smoothness_penalizes_jagged_profiles_more():
    smooth = torch.linspace(0, 1, 6).view(1, 6, 1, 1).expand(1, 6, 2, 2).contiguous()
    jagged = smooth.clone()
    jagged[:, 2] += 10.0  # one big spike in the middle of the profile
    assert vertical_smoothness_loss(jagged) > vertical_smoothness_loss(smooth)


def test_smoothness_single_depth_level_is_zero():
    pred = torch.randn(2, 1, 3, 3)
    assert torch.allclose(vertical_smoothness_loss(pred), torch.tensor(0.0))


def test_smoothness_masks_out_pairs_touching_land():
    pred, _, mask = _grid()
    # A depth pair only counts when both levels are valid; corrupting
    # `pred` only where at least one neighbor is invalid must not move
    # the loss.
    depth_dim = -3
    mask_upper = torch.narrow(mask, depth_dim, 1, mask.shape[depth_dim] - 1)
    mask_lower = torch.narrow(mask, depth_dim, 0, mask.shape[depth_dim] - 1)
    valid_pair = mask_upper & mask_lower

    baseline = vertical_smoothness_loss(pred, mask)
    corrupted = pred.clone()
    # Bump every level that touches at least one invalid neighbor.
    touches_invalid = ~mask
    corrupted[touches_invalid] += 500.0
    also_baseline = vertical_smoothness_loss(corrupted, mask)
    assert torch.allclose(baseline, also_baseline)
    assert valid_pair.any()  # sanity: the test setup actually has valid pairs


def test_smoothness_works_on_profile_only_tensors():
    pred, _, mask = _profile()
    loss = vertical_smoothness_loss(pred, mask, depth_dim=-1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# -- gradient_loss -----------------------------------------------------------


def test_gradient_loss_is_zero_for_identical_fields():
    pred, _, mask = _grid()
    assert torch.allclose(gradient_loss(pred, pred, mask), torch.tensor(0.0), atol=1e-6)


def test_gradient_loss_is_zero_for_a_shared_constant_offset():
    # A uniform bias changes no gradient at all.
    pred, target, mask = _grid()
    assert torch.allclose(
        gradient_loss(pred + 3.0, pred, mask), gradient_loss(target + 3.0, target, mask)
    )
    offset_pred = pred + 7.0
    assert torch.allclose(
        gradient_loss(offset_pred, pred, mask), torch.tensor(0.0), atol=1e-6
    )


def test_gradient_loss_none_spatial_dims_is_a_zero_noop():
    pred, target, mask = _profile()
    loss = gradient_loss(pred, target, mask, spatial_dims=None)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_gradient_loss_too_small_spatial_dims_is_zero():
    pred = torch.randn(2, 3, 1, 1)
    target = torch.randn(2, 3, 1, 1)
    loss = gradient_loss(pred, target, spatial_dims=(-2, -1))
    assert torch.allclose(loss, torch.tensor(0.0))


def test_gradient_loss_shape_mismatch_raises():
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 5, 5)
    with pytest.raises(ValueError, match="shape"):
        gradient_loss(pred, target)


def test_gradient_loss_penalizes_blur():
    # A blurred version of the target should score worse under
    # gradient_loss than the exact target, even though a plain MSE
    # would also register the blur — the point is that this term is
    # sensitive to edges being erased.
    torch.manual_seed(0)
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, :, 4:] = 5.0  # a sharp edge halfway across
    blurred = torch.nn.functional.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    assert gradient_loss(blurred, target) > 0.0
    assert torch.allclose(gradient_loss(target, target), torch.tensor(0.0), atol=1e-6)


# -- anomaly_loss -------------------------------------------------------------


def test_anomaly_loss_zero_for_perfect_prediction():
    pred, target, mask = _grid()
    assert torch.allclose(anomaly_loss(target, target, mask), torch.tensor(0.0), atol=1e-6)


def test_anomaly_loss_ignores_a_shared_constant_bias_without_baseline():
    # Without an explicit baseline, both pred and target are demeaned
    # by the target's own masked spatial mean, so a uniform additive
    # bias on top of a perfect prediction should vanish.
    pred, target, mask = _grid()
    biased_pred = target + 3.0
    assert torch.allclose(
        anomaly_loss(biased_pred, target, mask), torch.tensor(0.0), atol=1e-5
    )


def test_anomaly_loss_with_explicit_baseline_matches_manual_subtraction():
    pred, target, mask = _grid()
    baseline = torch.randn_like(target)
    got = anomaly_loss(pred, target, mask, baseline=baseline)
    expected = masked_mse_loss(pred - baseline, target - baseline, mask)
    assert torch.allclose(got, expected, atol=1e-6)


def test_anomaly_loss_none_spatial_dims_uses_global_baseline():
    pred, target, mask = _profile()
    loss = anomaly_loss(pred, target, mask, spatial_dims=None)
    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_anomaly_loss_shape_mismatch_raises():
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 5, 5)
    with pytest.raises(ValueError, match="shape"):
        anomaly_loss(pred, target)


# -- NEERLoss (composite) -----------------------------------------------------


def test_neer_loss_returns_all_named_components():
    criterion = NEERLoss()
    pred, target, mask = _grid()
    losses = criterion(pred, target, mask)
    assert set(losses.keys()) == {"mse", "smoothness", "gradient", "anomaly", "total"}
    for value in losses.values():
        assert value.dim() == 0
        assert torch.isfinite(value)


def test_neer_loss_total_matches_manual_weighted_sum():
    config = NEERLossConfig(
        mse_weight=1.0, smoothness_weight=0.5, gradient_weight=0.25, anomaly_weight=0.1
    )
    criterion = NEERLoss(config)
    pred, target, mask = _grid()
    losses = criterion(pred, target, mask)
    expected_total = (
        config.mse_weight * losses["mse"]
        + config.smoothness_weight * losses["smoothness"]
        + config.gradient_weight * losses["gradient"]
        + config.anomaly_weight * losses["anomaly"]
    )
    assert torch.allclose(losses["total"], expected_total, atol=1e-6)


def test_neer_loss_default_total_equals_mse_only():
    # Default weights leave every optional term at 0.0, so total should
    # equal the (weight-1.0) mse term exactly.
    criterion = NEERLoss()
    pred, target, mask = _grid()
    losses = criterion(pred, target, mask)
    assert torch.allclose(losses["total"], losses["mse"], atol=1e-6)


def test_neer_loss_optional_terms_are_computed_even_when_weight_is_zero():
    # Weight 0.0 still logs the term; it just doesn't move `total`.
    criterion = NEERLoss(NEERLossConfig(smoothness_weight=0.0))
    pred, target, mask = _grid()
    jagged_pred = pred.clone()
    jagged_pred[:, 1] += 50.0
    baseline = criterion(pred, target, mask)
    jagged = criterion(jagged_pred, target, mask)
    assert jagged["smoothness"] > baseline["smoothness"]
    # ...but `total` (mse-only here) is unaffected by the vertical jump
    # itself except through the ordinary mse term.
    assert torch.allclose(jagged["total"], jagged["mse"], atol=1e-6)


def test_neer_loss_backward_populates_gradients():
    criterion = NEERLoss(
        NEERLossConfig(smoothness_weight=0.1, gradient_weight=0.1, anomaly_weight=0.1)
    )
    pred, target, mask = _grid()
    pred = pred.clone().requires_grad_(True)
    losses = criterion(pred, target, mask)
    losses["total"].backward()
    assert pred.grad is not None
    assert torch.any(pred.grad != 0.0)


def test_neer_loss_works_on_pooled_profile_shape():
    # NEERModel.forward's (batch, num_depths) convention: no spatial
    # axis, so gradient_loss must be configured off (or degrade to
    # zero) rather than error.
    config = NEERLossConfig(depth_dim=-1, spatial_dims=None, smoothness_weight=0.2)
    criterion = NEERLoss(config)
    pred, target, mask = _profile()
    losses = criterion(pred, target, mask)
    assert torch.allclose(losses["gradient"], torch.tensor(0.0))
    assert torch.isfinite(losses["total"])


def test_neer_loss_is_an_nn_module():
    criterion = NEERLoss()
    assert isinstance(criterion, torch.nn.Module)
    # No learnable parameters of its own.
    assert list(criterion.parameters()) == []
