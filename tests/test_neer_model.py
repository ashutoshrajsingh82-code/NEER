"""Tests for the Phase 18 complete NEER model (`src/models/neer_model.py`).

Same two-group layout as the other model test files: a couple of
construction/validation checks that don't need torch, then the full
suite (including the required complete CPU forward-pass test) skipped
via `pytest.importorskip` when torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models.depth_decoder import DepthDecoderConfig  # noqa: E402
from src.models.depth_embedding import DEFAULT_DEPTHS, DepthEmbeddingConfig  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.vit import ViTConfig  # noqa: E402

# --------------------------------------------------------------------------
# Config-level checks — no torch required for the config objects
# themselves (constructing NEERModel does need torch; see below).
# --------------------------------------------------------------------------


def test_mismatched_decoder_embed_dim_is_a_config_level_incompatibility():
    # NEERModel itself needs torch to construct, but the configs that
    # would trigger its embed_dim mismatch error can be built without
    # it, and disagree with each other independent of torch:
    vit_cfg = ViTConfig(in_channels=64, embed_dim=64, num_heads=4)
    decoder_cfg = DepthDecoderConfig(embed_dim=32, num_heads=4)
    assert vit_cfg.embed_dim != decoder_cfg.embed_dim


# --------------------------------------------------------------------------
# Model tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.models.neer_model import NEERModel  # noqa: E402


def _small_model(**overrides) -> NEERModel:
    cnn_cfg = overrides.pop(
        "cnn_config", CNNEncoderConfig(in_channels=16, channels=(8, 16), dropout=0.0)
    )
    vit_cfg = overrides.pop(
        "vit_config", ViTConfig(in_channels=16, patch_size=4, embed_dim=32, num_heads=4, depth=1, dropout=0.0)
    )
    decoder_cfg = overrides.pop(
        "decoder_config",
        DepthDecoderConfig(
            embed_dim=32,
            num_heads=4,
            dropout=0.0,
            depth_config=DepthEmbeddingConfig(depths=(0.0, 10.0, 100.0, 1000.0), embed_dim=32),
        ),
    )
    return NEERModel(cnn_cfg, vit_cfg, decoder_cfg)


# --------------------------------------------------------------------------
# Construction / wiring
# --------------------------------------------------------------------------


def test_default_construction_wires_matching_embed_dims():
    model = NEERModel()
    assert model.in_channels == NEER_N_CHANNELS
    assert model.embed_dim == 256
    assert model.num_depths == 15
    assert model.depths == DEFAULT_DEPTHS


def test_mismatched_decoder_embed_dim_raises():
    vit_cfg = ViTConfig(in_channels=16, embed_dim=32, num_heads=4)
    decoder_cfg = DepthDecoderConfig(embed_dim=64, num_heads=4)
    with pytest.raises(ValueError, match="embed_dim"):
        NEERModel(CNNEncoderConfig(channels=(8, 16)), vit_cfg, decoder_cfg)


def test_decoder_config_omitted_is_derived_from_encoder_embed_dim():
    vit_cfg = ViTConfig(in_channels=16, embed_dim=32, num_heads=4)
    model = NEERModel(CNNEncoderConfig(channels=(8, 16)), vit_cfg)
    assert model.decoder.embed_dim == 32


# --------------------------------------------------------------------------
# forward(): shape and correctness
# --------------------------------------------------------------------------


def test_forward_output_shape():
    model = _small_model().eval()
    out = model(torch.randn(3, 16, 20, 30))
    assert out.shape == (3, 4)


def test_forward_equals_decoder_of_get_embedding():
    # forward() must be exactly decoder(get_embedding(x)) — no separate
    # or divergent computation path.
    model = _small_model().eval()
    x = torch.randn(2, 16, 12, 12)
    with torch.no_grad():
        out = model(x)
        embedding = model.get_embedding(x)
        expected = model.decoder(embedding)
    assert torch.equal(out, expected)


def test_default_config_end_to_end_on_real_neer_grid():
    model = NEERModel().eval()
    with torch.no_grad():
        out = model(torch.randn(1, NEER_N_CHANNELS, 101, 241))
    assert out.shape == (1, 15)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# get_embedding()
# --------------------------------------------------------------------------


def test_get_embedding_shape_and_matches_encoder():
    model = _small_model().eval()
    x = torch.randn(2, 16, 12, 12)
    with torch.no_grad():
        embedding = model.get_embedding(x)
        expected = model.encoder.get_embedding(x)
    assert embedding.shape == (2, 32)
    assert torch.equal(embedding, expected)


# --------------------------------------------------------------------------
# predict_profile()
# --------------------------------------------------------------------------


def test_predict_profile_adds_climatology_to_forward_output():
    model = _small_model().eval()
    x = torch.randn(3, 16, 12, 12)
    climatology = torch.tensor([10.0, 8.0, 4.0, 2.0])  # (num_depths,)
    with torch.no_grad():
        anomalies = model(x)
        reconstructed = model.predict_profile(x, climatology)
    assert reconstructed.shape == (3, 4)
    assert torch.allclose(reconstructed, anomalies + climatology.unsqueeze(0))


def test_predict_profile_accepts_per_sample_climatology():
    model = _small_model().eval()
    x = torch.randn(3, 16, 12, 12)
    climatology = torch.rand(3, 4) * 20.0  # (batch, num_depths)
    with torch.no_grad():
        anomalies = model(x)
        reconstructed = model.predict_profile(x, climatology)
    assert torch.allclose(reconstructed, anomalies + climatology)


def test_predict_profile_accepts_plain_list_climatology():
    model = _small_model().eval()
    x = torch.randn(2, 16, 12, 12)
    climatology = [10.0, 8.0, 4.0, 2.0]
    with torch.no_grad():
        reconstructed = model.predict_profile(x, climatology)
    assert reconstructed.shape == (2, 4)
    assert torch.isfinite(reconstructed).all()


def test_predict_profile_rejects_mismatched_climatology_shape():
    model = _small_model().eval()
    x = torch.randn(2, 16, 12, 12)
    with pytest.raises(ValueError, match="climatology"):
        model.predict_profile(x, torch.zeros(3))  # wrong length


def test_zero_climatology_reconstructs_to_the_raw_anomaly():
    # climatology=0 is the identity case: reconstructed temperature
    # should equal the predicted anomaly exactly.
    model = _small_model().eval()
    x = torch.randn(2, 16, 12, 12)
    with torch.no_grad():
        anomalies = model(x)
        reconstructed = model.predict_profile(x, torch.zeros(4))
    assert torch.equal(reconstructed, anomalies)


# --------------------------------------------------------------------------
# Complete CPU forward-pass test (as required by this phase)
# --------------------------------------------------------------------------


def test_complete_cpu_forward_pass():
    """End-to-end smoke test of the whole untrained pipeline on the CPU:

    surface fields -> CNN -> ViT -> 256D embedding -> depth embeddings
    -> depth decoder -> 15 anomalies -> + climatology -> reconstructed
    temperature. Nothing is trained; this only proves the wiring works
    and produces the right shapes, on CPU, for every exposed method.
    """
    device = torch.device("cpu")
    model = NEERModel().to(device).eval()

    batch = 2
    x = torch.randn(batch, NEER_N_CHANNELS, 101, 241, device=device)

    with torch.no_grad():
        embedding = model.get_embedding(x)
        anomalies = model(x)

        # A real caller would build this from
        # MonthlyClimatology.depth_profile(...) per sample; a flat
        # baseline is enough to exercise the wiring here.
        climatology = torch.full((batch, model.num_depths), 15.0, device=device)
        temperature = model.predict_profile(x, climatology)

    assert embedding.shape == (batch, 256)
    assert torch.isfinite(embedding).all()

    assert anomalies.shape == (batch, 15)
    assert torch.isfinite(anomalies).all()

    assert temperature.shape == (batch, 15)
    assert torch.isfinite(temperature).all()
    assert torch.allclose(temperature, anomalies + 15.0)

    assert model.depths == DEFAULT_DEPTHS
    assert temperature.device.type == "cpu"


# --------------------------------------------------------------------------
# Batch inference
# --------------------------------------------------------------------------


def test_batched_forward_matches_per_sample_forward():
    model = _small_model().eval()
    x = torch.randn(4, 16, 10, 14)
    with torch.no_grad():
        batched = model(x)
        single = torch.cat([model(x[i : i + 1]) for i in range(4)], dim=0)
    assert torch.allclose(batched, single, atol=1e-4)


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_forward_shape_across_batch_sizes(batch):
    model = _small_model().eval()
    out = model(torch.randn(batch, 16, 10, 14))
    assert out.shape == (batch, 4)


# --------------------------------------------------------------------------
# Gradient flow
# --------------------------------------------------------------------------


def test_gradients_flow_from_forward_to_every_parameter():
    model = _small_model()
    x = torch.randn(2, 16, 12, 12, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_gradients_flow_through_predict_profile_back_to_the_encoder():
    model = _small_model()
    x = torch.randn(2, 16, 12, 12, requires_grad=True)
    climatology = torch.rand(4)
    model.predict_profile(x, climatology).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"


def test_cpu_friendly_parameter_count_is_modest():
    n_params = sum(p.numel() for p in NEERModel().parameters())
    assert n_params < 10_000_000
