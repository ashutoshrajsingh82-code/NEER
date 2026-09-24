"""Tests for Phase 23 — optional heteroscedastic regression.

Covers three layers, in order:

1. Config — `uncertainty.enabled` in `src/utils/config.py` / `configs/*.yaml`,
   no torch required.
2. `DepthDecoder` / `NEERModel` — the log-variance head itself, `forward`
   staying mean-only regardless of the flag, `forward_with_uncertainty`
   raising when disabled, and `log_variance_to_sigma`. Needs torch.
3. `src.training.losses.gaussian_nll_loss` — the masked Gaussian NLL used
   to train against the extra head. Needs torch.

Same two-group layout as `tests/test_gnn.py` / `tests/test_gnn_graph.py`:
config-level checks first, then a torch-gated suite skipped as a whole via
`pytest.importorskip` when torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.depth_decoder import DepthDecoderConfig  # noqa: E402
from src.models.depth_embedding import DepthEmbeddingConfig  # noqa: E402
from src.utils.config import (  # noqa: E402
    UncertaintyConfig,
    load_config,
    validate_uncertainty,
)

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_decoder_config_uncertainty_defaults_to_false():
    cfg = DepthDecoderConfig()
    assert cfg.uncertainty_enabled is False


def test_decoder_config_uncertainty_is_settable():
    cfg = DepthDecoderConfig(uncertainty_enabled=True)
    assert cfg.uncertainty_enabled is True


@pytest.mark.parametrize("environment", ["demo", "development", "production", "base"])
def test_uncertainty_disabled_by_default_in_every_environment(environment):
    config = load_config(environment)
    assert config.uncertainty == UncertaintyConfig(enabled=False)


def test_validate_uncertainty_requires_enabled_key():
    with pytest.raises(Exception, match="uncertainty.enabled"):
        validate_uncertainty({})


def test_validate_uncertainty_rejects_non_boolean():
    with pytest.raises(Exception, match="uncertainty.enabled"):
        validate_uncertainty({"enabled": "false"})


def test_validate_uncertainty_accepts_valid_config():
    validate_uncertainty({"enabled": True})
    validate_uncertainty({"enabled": False})


# --------------------------------------------------------------------------
# Model tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models.depth_decoder import DepthDecoder, log_variance_to_sigma  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.neer_model import NEERModel  # noqa: E402
from src.models.vit import ViTConfig  # noqa: E402
from src.training.losses import gaussian_nll_loss  # noqa: E402

DEPTHS = (0.0, 10.0, 100.0, 1000.0)
EMBED = 32


def _small_decoder(uncertainty_enabled: bool) -> DepthDecoder:
    depth_cfg = DepthEmbeddingConfig(depths=DEPTHS, embed_dim=EMBED)
    cfg = DepthDecoderConfig(
        embed_dim=EMBED,
        num_heads=4,
        dropout=0.0,
        depth_config=depth_cfg,
        uncertainty_enabled=uncertainty_enabled,
    )
    return DepthDecoder(cfg)


def _small_model(uncertainty_enabled: bool) -> NEERModel:
    cnn_cfg = CNNEncoderConfig(in_channels=8, channels=(8, 16), dropout=0.0)
    vit_cfg = ViTConfig(in_channels=16, patch_size=4, embed_dim=EMBED, num_heads=4, depth=1, dropout=0.0)
    decoder_cfg = DepthDecoderConfig(
        embed_dim=EMBED,
        num_heads=4,
        dropout=0.0,
        depth_config=DepthEmbeddingConfig(depths=DEPTHS, embed_dim=EMBED),
        uncertainty_enabled=uncertainty_enabled,
    )
    return NEERModel(cnn_cfg, vit_cfg, decoder_cfg)


# --- DepthDecoder: disabled (default) --------------------------------------


def test_disabled_decoder_builds_no_log_var_head():
    decoder = _small_decoder(uncertainty_enabled=False)
    assert decoder.uncertainty_enabled is False
    assert decoder.log_var_head is None


def test_disabled_decoder_forward_with_uncertainty_raises():
    decoder = _small_decoder(uncertainty_enabled=False)
    with pytest.raises(RuntimeError, match="uncertainty_enabled=True"):
        decoder.forward_with_uncertainty(torch.randn(2, EMBED))


def test_disabled_decoder_forward_is_unaffected():
    decoder = _small_decoder(uncertainty_enabled=False).eval()
    out = decoder(torch.randn(3, EMBED))
    assert out.shape == (3, len(DEPTHS))


# --- DepthDecoder: enabled ---------------------------------------------


def test_enabled_decoder_builds_log_var_head():
    decoder = _small_decoder(uncertainty_enabled=True)
    assert decoder.uncertainty_enabled is True
    assert isinstance(decoder.log_var_head, torch.nn.Linear)
    assert decoder.log_var_head.out_features == 1


def test_enabled_decoder_forward_with_uncertainty_shapes():
    decoder = _small_decoder(uncertainty_enabled=True).eval()
    mean, log_variance = decoder.forward_with_uncertainty(torch.randn(5, EMBED))
    assert mean.shape == (5, len(DEPTHS))
    assert log_variance.shape == (5, len(DEPTHS))
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_variance).all()


def test_enabled_decoder_forward_still_returns_mean_only():
    # forward()'s contract (shape, and the mean values themselves) must
    # not change just because uncertainty is enabled.
    decoder = _small_decoder(uncertainty_enabled=True).eval()
    x = torch.randn(4, EMBED)
    with torch.no_grad():
        mean_only = decoder(x)
        mean_from_pair, _ = decoder.forward_with_uncertainty(x)
    assert mean_only.shape == (4, len(DEPTHS))
    assert torch.allclose(mean_only, mean_from_pair, atol=1e-6)


def test_enabled_decoder_mean_and_log_var_heads_are_independent_params():
    decoder = _small_decoder(uncertainty_enabled=True)
    assert decoder.head is not decoder.log_var_head
    head_params = set(map(id, decoder.head.parameters()))
    log_var_params = set(map(id, decoder.log_var_head.parameters()))
    assert head_params.isdisjoint(log_var_params)


# --- log_variance_to_sigma ---------------------------------------------


def test_log_variance_to_sigma_formula():
    log_variance = torch.tensor([0.0, 2.0, -4.0])
    sigma = log_variance_to_sigma(log_variance)
    expected = torch.exp(0.5 * log_variance)
    assert torch.allclose(sigma, expected)
    assert (sigma > 0).all()


def test_log_variance_to_sigma_zero_log_variance_is_unit_sigma():
    sigma = log_variance_to_sigma(torch.zeros(3, 4))
    assert torch.allclose(sigma, torch.ones(3, 4))


# --- NEERModel wiring ----------------------------------------------------


def test_model_uncertainty_disabled_by_default():
    model = NEERModel()
    assert model.uncertainty_enabled is False


def test_model_disabled_forward_with_uncertainty_raises():
    model = _small_model(uncertainty_enabled=False)
    x = torch.randn(2, 8, 17, 17)
    with pytest.raises(RuntimeError, match="uncertainty_enabled=True"):
        model.forward_with_uncertainty(x)
    with pytest.raises(RuntimeError, match="uncertainty_enabled=True"):
        model.predict_uncertainty(x)


def test_model_enabled_forward_with_uncertainty_shapes():
    model = _small_model(uncertainty_enabled=True).eval()
    x = torch.randn(2, 8, 17, 17)
    with torch.no_grad():
        mean, log_variance = model.forward_with_uncertainty(x)
    assert mean.shape == (2, len(DEPTHS))
    assert log_variance.shape == (2, len(DEPTHS))


def test_model_predict_uncertainty_matches_manual_sigma_conversion():
    model = _small_model(uncertainty_enabled=True).eval()
    x = torch.randn(2, 8, 17, 17)
    with torch.no_grad():
        mean_a, log_variance = model.forward_with_uncertainty(x)
        mean_b, sigma = model.predict_uncertainty(x)
    assert torch.allclose(mean_a, mean_b)
    assert torch.allclose(sigma, log_variance_to_sigma(log_variance))


def test_model_forward_unaffected_by_uncertainty_flag():
    off = _small_model(uncertainty_enabled=False).eval()
    on = _small_model(uncertainty_enabled=True).eval()
    x = torch.randn(2, 8, 17, 17)
    with torch.no_grad():
        assert off(x).shape == on(x).shape == (2, len(DEPTHS))


def test_from_config_wires_uncertainty_enabled():
    config = load_config("demo")
    assert config.uncertainty.enabled is False
    model = NEERModel.from_config(config)
    assert model.uncertainty_enabled is False


def test_from_config_respects_env_override(monkeypatch):
    monkeypatch.setenv("NEER_UNCERTAINTY_ENABLED", "true")
    config = load_config("demo")
    assert config.uncertainty.enabled is True
    model = NEERModel.from_config(config)
    assert model.uncertainty_enabled is True


# --------------------------------------------------------------------------
# gaussian_nll_loss
# --------------------------------------------------------------------------


def test_gaussian_nll_zero_when_mean_matches_target_and_log_variance_is_zero():
    pred_mean = torch.zeros(4)
    target = torch.zeros(4)
    log_variance = torch.zeros(4)
    loss = gaussian_nll_loss(pred_mean, log_variance, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_gaussian_nll_matches_hand_computed_value():
    pred_mean = torch.tensor([0.0, 1.0])
    target = torch.tensor([2.0, 1.0])
    log_variance = torch.tensor([0.0, 0.0])
    # cell 0: 0.5 * (0 + (2-0)^2 / exp(0)) = 2.0; cell 1: 0.5 * (0 + 0) = 0.0
    loss = gaussian_nll_loss(pred_mean, log_variance, target)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_gaussian_nll_masking_excludes_invalid_cells():
    pred_mean = torch.tensor([0.0, 100.0])
    target = torch.tensor([0.0, -100.0])
    log_variance = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, False])
    loss = gaussian_nll_loss(pred_mean, log_variance, target, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_gaussian_nll_all_invalid_mask_returns_zero_not_nan():
    pred_mean = torch.randn(3)
    target = torch.randn(3)
    log_variance = torch.randn(3)
    mask = torch.zeros(3, dtype=torch.bool)
    loss = gaussian_nll_loss(pred_mean, log_variance, target, mask)
    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_gaussian_nll_rejects_mismatched_log_variance_shape():
    with pytest.raises(ValueError, match="log_variance"):
        gaussian_nll_loss(torch.randn(3), torch.randn(4), torch.randn(3))


def test_gaussian_nll_rejects_mismatched_pred_target_shape():
    with pytest.raises(ValueError):
        gaussian_nll_loss(torch.randn(3), torch.randn(3), torch.randn(4))


def test_gaussian_nll_penalizes_overconfident_wrong_predictions_more():
    # Same mean error, but a smaller (more confident) predicted variance
    # for the wrong prediction must cost more, not less.
    pred_mean = torch.tensor([0.0])
    target = torch.tensor([5.0])
    confident = gaussian_nll_loss(pred_mean, torch.tensor([-2.0]), target)
    unsure = gaussian_nll_loss(pred_mean, torch.tensor([2.0]), target)
    assert confident > unsure


def test_gaussian_nll_end_to_end_with_decoder_output():
    decoder = _small_decoder(uncertainty_enabled=True).eval()
    with torch.no_grad():
        mean, log_variance = decoder.forward_with_uncertainty(torch.randn(3, EMBED))
    target = torch.randn(3, len(DEPTHS))
    loss = gaussian_nll_loss(mean, log_variance, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_gaussian_nll_is_differentiable_through_both_heads():
    decoder = _small_decoder(uncertainty_enabled=True)
    mean, log_variance = decoder.forward_with_uncertainty(torch.randn(3, EMBED))
    target = torch.randn(3, len(DEPTHS))
    loss = gaussian_nll_loss(mean, log_variance, target)
    loss.backward()
    assert decoder.head.weight.grad is not None
    assert decoder.log_var_head.weight.grad is not None