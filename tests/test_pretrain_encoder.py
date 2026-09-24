"""Tests for the Phase 21A.1 pretraining encoder (`src/models/pretrain_encoder.py`).

Same two-group layout as `tests/test_encoder.py` / `tests/test_vit.py`:
config tests that run without torch, then model tests skipped via
`pytest.importorskip` when torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.pretrain_encoder import PretrainEncoderConfig  # noqa: E402
from src.models.vit import DEFAULT_EMBED_DIM, ViTConfig  # noqa: E402

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_default_embedding_dim_matches_configuration_system_default():
    # DEFAULT_EMBED_DIM mirrors configs/base.yaml's model.embedding_dim (256).
    assert PretrainEncoderConfig().embedding_dim == DEFAULT_EMBED_DIM
    assert PretrainEncoderConfig().embedding_dim == 256


def test_embedding_dim_is_configurable():
    config = PretrainEncoderConfig(embedding_dim=128)
    assert config.embedding_dim == 128
    assert config.resolved_vit_config().embed_dim == 128


@pytest.mark.parametrize("embedding_dim", [0, -1, -256])
def test_non_positive_embedding_dim_rejected(embedding_dim):
    with pytest.raises(ValueError, match="embedding_dim"):
        PretrainEncoderConfig(embedding_dim=embedding_dim)


def test_vit_config_derived_from_cnn_output_and_embedding_dim():
    config = PretrainEncoderConfig(
        cnn_config=CNNEncoderConfig(channels=(16, 40)), embedding_dim=64
    )
    resolved = config.resolved_vit_config()
    assert resolved.in_channels == 40
    assert resolved.embed_dim == 64


def test_explicit_vit_config_is_used_as_is():
    vit_config = ViTConfig(in_channels=64, embed_dim=32, num_heads=4)
    config = PretrainEncoderConfig(
        cnn_config=CNNEncoderConfig(channels=(16, 64)),
        vit_config=vit_config,
        embedding_dim=32,
    )
    assert config.resolved_vit_config() is vit_config


def test_explicit_vit_config_mismatched_in_channels_rejected():
    vit_config = ViTConfig(in_channels=999, embed_dim=32, num_heads=4)
    with pytest.raises(ValueError, match="in_channels"):
        PretrainEncoderConfig(
            cnn_config=CNNEncoderConfig(channels=(16, 64)),
            vit_config=vit_config,
            embedding_dim=32,
        )


def test_explicit_vit_config_mismatched_embedding_dim_rejected():
    vit_config = ViTConfig(in_channels=64, embed_dim=32, num_heads=4)
    with pytest.raises(ValueError, match="embedding_dim"):
        PretrainEncoderConfig(
            cnn_config=CNNEncoderConfig(channels=(16, 64)),
            vit_config=vit_config,
            embedding_dim=256,  # disagrees with vit_config.embed_dim (32)
        )


def test_default_cnn_config_matches_cnn_encoder_default():
    assert PretrainEncoderConfig().cnn_config == CNNEncoderConfig()


# --------------------------------------------------------------------------
# Model tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.models.pretrain_encoder import PretrainEncoder  # noqa: E402


def _small_encoder(**overrides) -> PretrainEncoder:
    cnn_cfg = overrides.pop("cnn_config", CNNEncoderConfig(channels=(8, 16), dropout=0.0))
    embedding_dim = overrides.pop("embedding_dim", 32)
    vit_config = overrides.pop("vit_config", None)
    if vit_config is None:
        vit_kwargs = dict(
            in_channels=cnn_cfg.out_channels,
            patch_size=4,
            embed_dim=embedding_dim,
            num_heads=4,
            depth=2,
            dropout=0.0,
        )
        vit_kwargs.update(overrides)
        vit_config = ViTConfig(**vit_kwargs)
    config = PretrainEncoderConfig(
        cnn_config=cnn_cfg, vit_config=vit_config, embedding_dim=vit_config.embed_dim
    )
    return PretrainEncoder(config)


# --- initialization ---------------------------------------------------------


def test_default_construction():
    encoder = PretrainEncoder()
    assert encoder.in_channels == NEER_N_CHANNELS
    assert encoder.embed_dim == 256


def test_construction_with_custom_config():
    encoder = _small_encoder(embedding_dim=64)
    assert encoder.embed_dim == 64
    assert isinstance(encoder.cnn, torch.nn.Module)
    assert isinstance(encoder.vit, torch.nn.Module)


def test_configurable_embedding_dim_at_construction():
    for embedding_dim in (16, 32, 64):
        encoder = _small_encoder(embedding_dim=embedding_dim)
        assert encoder.embed_dim == embedding_dim


# --- forward pass: CNN -> ViT -> embedding -----------------------------------


def test_forward_pass_returns_latent_embedding():
    encoder = _small_encoder(embedding_dim=32).eval()
    x = torch.randn(3, NEER_N_CHANNELS, 20, 30)
    with torch.no_grad():
        embedding = encoder(x)
    assert embedding.shape == (3, 32)
    assert torch.isfinite(embedding).all()


def test_forward_equals_manual_cnn_then_vit_then_pool():
    # forward() must literally be CNN -> ViT -> mean-pool over tokens,
    # not a separate or divergent computation.
    encoder = _small_encoder(embedding_dim=32).eval()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 16)
    with torch.no_grad():
        out = encoder(x)
        expected = encoder.vit(encoder.cnn(x)).mean(dim=1)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_tensor_shape_correctness_across_batch_sizes(batch):
    encoder = _small_encoder(embedding_dim=32)
    out = encoder(torch.randn(batch, NEER_N_CHANNELS, 10, 14))
    assert out.shape == (batch, 32)


def test_default_config_end_to_end_on_real_neer_grid():
    encoder = PretrainEncoder().eval()
    with torch.no_grad():
        out = encoder(torch.randn(1, NEER_N_CHANNELS, 101, 241))
    assert out.shape == (1, 256)
    assert torch.isfinite(out).all()


def test_batched_forward_matches_per_sample_forward():
    encoder = _small_encoder(embedding_dim=32).eval()
    x = torch.randn(4, NEER_N_CHANNELS, 10, 14)
    with torch.no_grad():
        batched = encoder(x)
        single = torch.cat([encoder(x[i : i + 1]) for i in range(4)], dim=0)
    assert torch.allclose(batched, single, atol=1e-4)


# --- configurable embedding dimension (model-level) --------------------------


@pytest.mark.parametrize("embedding_dim", [16, 32, 64, 128])
def test_configurable_embedding_dim_shapes_output(embedding_dim):
    encoder = _small_encoder(embedding_dim=embedding_dim)
    out = encoder(torch.randn(2, NEER_N_CHANNELS, 12, 12))
    assert out.shape == (2, embedding_dim)


# --- invalid input shape handling --------------------------------------------


def test_rejects_non_4d_input():
    encoder = _small_encoder()
    with pytest.raises(ValueError, match="4D"):
        encoder(torch.randn(NEER_N_CHANNELS, 12, 12))


def test_rejects_wrong_channel_count():
    encoder = _small_encoder()
    with pytest.raises(ValueError):
        encoder(torch.randn(2, NEER_N_CHANNELS + 1, 12, 12))


def test_rejects_empty_batch_shape_mismatch():
    encoder = _small_encoder()
    with pytest.raises(ValueError):
        encoder(torch.randn(2, 3))  # far too few dims


# --- gradient flow / reusability ----------------------------------------------


def test_gradients_flow_to_cnn_and_vit_parameters():
    encoder = _small_encoder(embedding_dim=32)
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12, requires_grad=True)
    encoder(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_cnn_and_vit_are_reusable_standalone_modules():
    # The whole point of reusing CNNEncoder/VisionTransformer rather
    # than reimplementing them: encoder.cnn and encoder.vit are the
    # exact same classes CNNViTEncoder wraps, so their state dicts can
    # be lifted straight into another model built from those classes.
    from src.models.encoder import CNNEncoder
    from src.models.vit import VisionTransformer

    encoder = _small_encoder(embedding_dim=32)
    assert isinstance(encoder.cnn, CNNEncoder)
    assert isinstance(encoder.vit, VisionTransformer)


def test_cpu_friendly_parameter_count_is_modest():
    n_params = sum(p.numel() for p in PretrainEncoder().parameters())
    assert n_params < 5_000_000