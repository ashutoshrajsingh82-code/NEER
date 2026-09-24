"""Tests for the Phase 14 Vision Transformer (`src/models/vit.py`).

Same two-group layout as `tests/test_encoder.py`: config tests that run
without torch, then model tests skipped via `pytest.importorskip` when
torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models.encoder import DEFAULT_CHANNELS, CNNEncoderConfig  # noqa: E402
from src.models.vit import (  # noqa: E402
    VALID_PAD_MODES,
    ViTConfig,
    grid_shape_for,
)

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_default_embed_dim_is_256():
    assert ViTConfig().embed_dim == 256


def test_default_in_channels_matches_default_cnn_output():
    assert ViTConfig().in_channels == DEFAULT_CHANNELS[-1]
    assert ViTConfig().in_channels == CNNEncoderConfig().out_channels


def test_patch_size_int_and_pair_normalize():
    assert ViTConfig(patch_size=8).patch_hw == (8, 8)
    assert ViTConfig(patch_size=(4, 8)).patch_hw == (4, 8)


@pytest.mark.parametrize("patch_size", [0, -4, (0, 4), (4, -1)])
def test_non_positive_patch_size_rejected(patch_size):
    with pytest.raises(ValueError, match="patch_size"):
        ViTConfig(patch_size=patch_size)


@pytest.mark.parametrize("patch_size", [(4,), (4, 4, 4), "8", 2.5])
def test_malformed_patch_size_rejected(patch_size):
    with pytest.raises(ValueError, match="patch_size"):
        ViTConfig(patch_size=patch_size)


def test_embed_dim_must_divide_by_heads():
    with pytest.raises(ValueError, match="num_heads"):
        ViTConfig(embed_dim=256, num_heads=7)


def test_embed_dim_must_divide_by_four():
    with pytest.raises(ValueError, match="divisible by 4"):
        ViTConfig(embed_dim=30, num_heads=3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"in_channels": 0}, "in_channels"),
        ({"embed_dim": 0}, "embed_dim"),
        ({"num_heads": 0}, "num_heads"),
        ({"depth": 0}, "depth"),
        ({"mlp_ratio": 0}, "mlp_ratio"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"attention_dropout": 1.0}, "attention_dropout"),
        ({"pad_mode": "reflect"}, "pad_mode"),
    ],
)
def test_invalid_values_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ViTConfig(**kwargs)


@pytest.mark.parametrize("pad_mode", VALID_PAD_MODES)
def test_valid_pad_modes_accepted(pad_mode):
    assert ViTConfig(pad_mode=pad_mode).pad_mode == pad_mode


def test_derived_dims():
    cfg = ViTConfig(embed_dim=256, num_heads=8, mlp_ratio=2.0)
    assert cfg.head_dim == 32
    assert cfg.mlp_hidden_dim == 512


@pytest.mark.parametrize(
    "h, w, patch, expected",
    [
        (101, 241, 8, (13, 31)),  # real NEER grid, padded up
        (16, 32, 8, (2, 4)),  # exact multiple
        (16, 32, (4, 8), (4, 4)),
        (1, 1, 8, (1, 1)),
    ],
)
def test_grid_shape_for(h, w, patch, expected):
    assert grid_shape_for(h, w, patch) == expected


# --------------------------------------------------------------------------
# Model tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.models.vit import (  # noqa: E402
    CNNViTEncoder,
    VisionTransformer,
    sincos_2d_position_embedding,
)


def _small_vit(**overrides) -> VisionTransformer:
    kwargs = dict(in_channels=16, patch_size=4, embed_dim=32, num_heads=4, depth=2, dropout=0.0)
    kwargs.update(overrides)
    return VisionTransformer(ViTConfig(**kwargs))


# --- forward pass & tensor shapes ------------------------------------------


def test_forward_pass_is_finite():
    vit = _small_vit().eval()
    out = vit(torch.randn(2, 16, 16, 24))
    assert torch.isfinite(out).all()


def test_output_shape_exact_multiple():
    vit = _small_vit(patch_size=4)
    out = vit(torch.randn(2, 16, 16, 24))
    assert out.shape == (2, (16 // 4) * (24 // 4), 32)


@pytest.mark.parametrize(
    "h, w, patch, expected_tokens",
    [
        (101, 241, 8, 13 * 31),  # real NEER grid: padded, nothing cropped
        (10, 10, 4, 3 * 3),
        (5, 9, 4, 2 * 3),
        (3, 3, 8, 1),  # smaller than one patch
    ],
)
def test_output_shape_pads_non_divisible_inputs(h, w, patch, expected_tokens):
    vit = _small_vit(patch_size=patch, depth=1)
    out = vit(torch.randn(1, 16, h, w))
    assert out.shape == (1, expected_tokens, 32)
    assert vit.num_tokens(h, w) == expected_tokens


def test_rectangular_patches():
    vit = _small_vit(patch_size=(4, 8), depth=1)
    out = vit(torch.randn(2, 16, 16, 32))
    assert out.shape == (2, 4 * 4, 32)
    assert vit.grid_shape(16, 32) == (4, 4)


@pytest.mark.parametrize("pad_mode", VALID_PAD_MODES)
def test_both_pad_modes_work(pad_mode):
    vit = _small_vit(pad_mode=pad_mode, depth=1)
    out = vit(torch.randn(2, 16, 10, 13))
    assert out.shape == (2, 3 * 4, 32)
    assert torch.isfinite(out).all()


def test_default_config_shapes_on_real_neer_grid():
    vit = VisionTransformer(ViTConfig()).eval()
    assert vit.embed_dim == 256
    out = vit(torch.randn(1, DEFAULT_CHANNELS[-1], 101, 241))
    assert out.shape == (1, 403, 256)


@pytest.mark.parametrize("embed_dim, num_heads", [(32, 1), (32, 2), (64, 8), (128, 16)])
def test_configurable_embed_dim_and_heads(embed_dim, num_heads):
    vit = _small_vit(embed_dim=embed_dim, num_heads=num_heads, depth=1)
    out = vit(torch.randn(2, 16, 8, 8))
    assert out.shape == (2, 4, embed_dim)


@pytest.mark.parametrize("depth", [1, 2, 6])
def test_configurable_depth(depth):
    vit = _small_vit(depth=depth)
    assert len(vit.blocks) == depth
    assert vit(torch.randn(1, 16, 8, 8)).shape == (1, 4, 32)


def test_tokens_to_map_round_trip_shape():
    vit = _small_vit(patch_size=4, depth=1)
    tokens = vit(torch.randn(3, 16, 10, 14))
    fmap = vit.tokens_to_map(tokens, 10, 14)
    assert fmap.shape == (3, 32, 3, 4)
    # token (row i, col j) must land at fmap[:, :, i, j]
    assert torch.equal(fmap[:, :, 1, 2], tokens[:, 1 * 4 + 2, :])


def test_tokens_to_map_rejects_wrong_token_count():
    vit = _small_vit(patch_size=4, depth=1)
    with pytest.raises(ValueError, match="expected tokens"):
        vit.tokens_to_map(torch.randn(2, 5, 32), 10, 14)


# --- batch processing ------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 7])
def test_batch_sizes(batch):
    vit = _small_vit(depth=1)
    out = vit(torch.randn(batch, 16, 8, 12))
    assert out.shape == (batch, 2 * 3, 32)


def test_batch_matches_per_sample_processing_in_eval_mode():
    # Samples must not influence each other: running a batch is the same
    # as running each sample alone.
    vit = _small_vit(dropout=0.1).eval()
    x = torch.randn(4, 16, 10, 14)
    with torch.no_grad():
        batched = vit(x)
        single = torch.cat([vit(x[i : i + 1]) for i in range(4)], dim=0)
    assert torch.allclose(batched, single, atol=1e-4)


def test_batch_of_identical_samples_gives_identical_outputs():
    vit = _small_vit().eval()
    x = torch.randn(1, 16, 8, 8).repeat(3, 1, 1, 1)
    with torch.no_grad():
        out = vit(x)
    assert torch.allclose(out[0], out[1], atol=1e-5)
    assert torch.allclose(out[0], out[2], atol=1e-5)


# --- dropout ---------------------------------------------------------------


def test_dropout_disabled_in_eval_mode():
    vit = _small_vit(dropout=0.5, attention_dropout=0.5).eval()
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(vit(x), vit(x))


def test_dropout_active_in_train_mode():
    vit = _small_vit(dropout=0.5).train()
    x = torch.randn(2, 16, 8, 8)
    assert not torch.allclose(vit(x), vit(x))


def test_zero_dropout_is_deterministic_in_train_mode():
    vit = _small_vit(dropout=0.0, attention_dropout=0.0).train()
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(vit(x), vit(x))


# --- input validation ------------------------------------------------------


def test_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="input channels"):
        _small_vit()(torch.randn(2, 15, 8, 8))


def test_rejects_non_4d_input():
    with pytest.raises(ValueError, match="4D"):
        _small_vit()(torch.randn(16, 8, 8))


# --- position embedding ----------------------------------------------------


def test_position_embedding_shape_and_range():
    pos = sincos_2d_position_embedding(5, 7, 32)
    assert pos.shape == (35, 32)
    assert torch.isfinite(pos).all()
    assert pos.abs().max() <= 1.0 + 1e-6


def test_position_embedding_is_unique_per_position():
    pos = sincos_2d_position_embedding(6, 9, 64)
    assert torch.unique(pos, dim=0).shape[0] == 6 * 9


def test_position_embedding_distinguishes_rows_from_columns():
    # A (2, 3) grid: token 1 is (row 0, col 1); token 3 is (row 1, col 0).
    pos = sincos_2d_position_embedding(2, 3, 32)
    assert not torch.allclose(pos[1], pos[3])


# --- training-related sanity ----------------------------------------------


def test_gradients_flow_to_all_parameters():
    vit = _small_vit()
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    vit(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in vit.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_cpu_friendly_parameter_count_is_modest():
    n_params = sum(p.numel() for p in VisionTransformer(ViTConfig()).parameters())
    assert n_params < 5_000_000


def test_runs_on_cpu_device_explicitly():
    device = torch.device("cpu")
    vit = _small_vit().to(device)
    out = vit(torch.randn(2, 16, 8, 8, device=device))
    assert out.device.type == "cpu"


# --------------------------------------------------------------------------
# CNN -> ViT connection
# --------------------------------------------------------------------------


def _small_pipeline(**vit_overrides) -> CNNViTEncoder:
    cnn_cfg = CNNEncoderConfig(channels=(8, 16), dropout=0.0)
    vit_kwargs = dict(in_channels=16, patch_size=4, embed_dim=32, num_heads=4, depth=2, dropout=0.0)
    vit_kwargs.update(vit_overrides)
    return CNNViTEncoder(cnn_cfg, ViTConfig(**vit_kwargs))


def test_cnn_to_vit_forward_shape():
    model = _small_pipeline()
    out = model(torch.randn(2, NEER_N_CHANNELS, 20, 30))
    assert out.shape == (2, 5 * 8, 32)
    assert model.grid_shape(20, 30) == (5, 8)
    assert model.in_channels == NEER_N_CHANNELS
    assert model.embed_dim == 32


def test_cnn_to_vit_batch_processing():
    model = _small_pipeline().eval()
    x = torch.randn(3, NEER_N_CHANNELS, 12, 12)
    with torch.no_grad():
        batched = model(x)
        single = torch.cat([model(x[i : i + 1]) for i in range(3)], dim=0)
    assert batched.shape == (3, 9, 32)
    assert torch.allclose(batched, single, atol=1e-4)


def test_cnn_to_vit_tokens_to_map():
    model = _small_pipeline()
    tokens = model(torch.randn(2, NEER_N_CHANNELS, 20, 30))
    assert model.tokens_to_map(tokens, 20, 30).shape == (2, 32, 5, 8)


def test_vit_config_derived_from_cnn_when_omitted():
    model = CNNViTEncoder(CNNEncoderConfig(channels=(8, 24)))
    assert model.vit.in_channels == 24
    assert model.vit.embed_dim == 256  # default


def test_mismatched_channels_rejected():
    with pytest.raises(ValueError, match="in_channels"):
        CNNViTEncoder(CNNEncoderConfig(channels=(8, 16)), ViTConfig(in_channels=32))


def test_cnn_to_vit_gradients_reach_the_cnn():
    model = _small_pipeline()
    model(torch.randn(2, NEER_N_CHANNELS, 12, 12)).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"


def test_cnn_to_vit_real_neer_grid_end_to_end():
    # Defaults everywhere: 11 channels -> CNN (32, 64) -> ViT(256).
    model = CNNViTEncoder().eval()
    with torch.no_grad():
        out = model(torch.randn(1, NEER_N_CHANNELS, 101, 241))
    assert out.shape == (1, 403, 256)
    assert torch.isfinite(out).all()
