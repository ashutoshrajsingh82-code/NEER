"""Tests for the Phase 13 CNN spatial encoder (`src/models/encoder.py`).

Two groups, mirroring `tests/test_dataset.py`'s pattern for the same
optional-torch situation:

- Tests that run regardless of whether `torch` is installed, since the
  module (and `CNNEncoderConfig` in particular) must stay importable and
  usable without it.
- Tests that need `torch`, skipped with `pytest.importorskip` when it is
  not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_CHANNEL_ORDER, NEER_N_CHANNELS  # noqa: E402
from src.models.encoder import (  # noqa: E402
    VALID_ACTIVATIONS,
    VALID_NORMS,
    CNNEncoderConfig,
)

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_default_in_channels_matches_neer_contract():
    config = CNNEncoderConfig()
    assert config.in_channels == NEER_N_CHANNELS
    assert config.in_channels == len(NEER_CHANNEL_ORDER)


def test_default_out_channels():
    config = CNNEncoderConfig()
    assert config.out_channels == config.channels[-1]


@pytest.mark.parametrize("channels", [(), []])
def test_empty_channels_rejected(channels):
    with pytest.raises(ValueError, match="channels must be non-empty"):
        CNNEncoderConfig(channels=channels)


@pytest.mark.parametrize("channels", [(0, 32), (32, -1), (16, 0, 8)])
def test_non_positive_channel_entry_rejected(channels):
    with pytest.raises(ValueError, match="positive"):
        CNNEncoderConfig(channels=channels)


def test_non_positive_in_channels_rejected():
    with pytest.raises(ValueError, match="in_channels"):
        CNNEncoderConfig(in_channels=0)


@pytest.mark.parametrize("kernel_size", [0, -3, 2, 4])
def test_invalid_kernel_size_rejected(kernel_size):
    with pytest.raises(ValueError, match="kernel_size"):
        CNNEncoderConfig(kernel_size=kernel_size)


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
def test_valid_odd_kernel_sizes_accepted(kernel_size):
    config = CNNEncoderConfig(kernel_size=kernel_size)
    assert config.kernel_size == kernel_size


def test_invalid_norm_rejected():
    with pytest.raises(ValueError, match="norm"):
        CNNEncoderConfig(norm="layernorm")


@pytest.mark.parametrize("norm", VALID_NORMS)
def test_all_valid_norms_accepted(norm):
    config = CNNEncoderConfig(norm=norm)
    assert config.norm == norm


def test_invalid_activation_rejected():
    with pytest.raises(ValueError, match="activation"):
        CNNEncoderConfig(activation="tanh")


@pytest.mark.parametrize("activation", VALID_ACTIVATIONS)
def test_all_valid_activations_accepted(activation):
    config = CNNEncoderConfig(activation=activation)
    assert config.activation == activation


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_invalid_dropout_rejected(dropout):
    with pytest.raises(ValueError, match="dropout"):
        CNNEncoderConfig(dropout=dropout)


@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5, 0.99])
def test_valid_dropout_accepted(dropout):
    config = CNNEncoderConfig(dropout=dropout)
    assert config.dropout == dropout


def test_invalid_padding_mode_rejected():
    with pytest.raises(ValueError, match="padding_mode"):
        CNNEncoderConfig(padding_mode="wrap")


def test_non_positive_norm_groups_rejected():
    with pytest.raises(ValueError, match="norm_groups"):
        CNNEncoderConfig(norm_groups=0)


def test_importing_encoder_module_does_not_require_torch():
    # Regression guard for the lazy-import pattern (mirrors
    # src/data/dataset.py): importing the module and constructing a
    # config must never require torch to be installed, whether or not
    # it actually is in this environment.
    import src.models.encoder as encoder_module

    assert hasattr(encoder_module, "CNNEncoder")
    assert hasattr(encoder_module, "CNNEncoderConfig")


# --------------------------------------------------------------------------
# CNNEncoder tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.models.encoder import CNNEncoder, DEFAULT_CHANNELS  # noqa: E402


def _make_encoder(**overrides) -> CNNEncoder:
    config = CNNEncoderConfig(**overrides)
    return CNNEncoder(config)


def test_output_shape_default_config():
    encoder = _make_encoder()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 16)
    out = encoder(x)
    assert out.shape == (2, DEFAULT_CHANNELS[-1], 12, 16)


@pytest.mark.parametrize(
    "batch, height, width",
    [(1, 8, 8), (3, 5, 9), (4, 101, 241)],  # 101x241: the real NEER grid
)
def test_output_shape_preserves_spatial_dims(batch, height, width):
    encoder = _make_encoder(channels=(16, 32))
    x = torch.randn(batch, NEER_N_CHANNELS, height, width)
    out = encoder(x)
    assert out.shape == (batch, 32, height, width)


@pytest.mark.parametrize(
    "channels",
    [(8,), (16, 32), (32, 64, 128), (4, 4, 4, 4)],
)
def test_output_channels_matches_config_for_various_depths(channels):
    encoder = _make_encoder(channels=channels)
    x = torch.randn(2, NEER_N_CHANNELS, 10, 10)
    out = encoder(x)
    assert out.shape == (2, channels[-1], 10, 10)
    assert encoder.out_channels == channels[-1]
    assert encoder.in_channels == NEER_N_CHANNELS


def test_custom_in_channels():
    encoder = _make_encoder(in_channels=5, channels=(8,))
    x = torch.randn(2, 5, 10, 10)
    out = encoder(x)
    assert out.shape == (2, 8, 10, 10)


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_kernel_size_preserves_spatial_dims(kernel_size):
    encoder = _make_encoder(channels=(16,), kernel_size=kernel_size)
    x = torch.randn(2, NEER_N_CHANNELS, 20, 30)
    out = encoder(x)
    assert out.shape == (2, 16, 20, 30)


@pytest.mark.parametrize("norm", VALID_NORMS)
def test_all_norm_types_produce_correct_shape(norm):
    encoder = _make_encoder(channels=(16, 32), norm=norm)
    x = torch.randn(2, NEER_N_CHANNELS, 10, 12)
    out = encoder(x)
    assert out.shape == (2, 32, 10, 12)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("activation", VALID_ACTIVATIONS)
def test_all_activation_types_produce_correct_shape(activation):
    encoder = _make_encoder(channels=(16,), activation=activation)
    x = torch.randn(2, NEER_N_CHANNELS, 10, 12)
    out = encoder(x)
    assert out.shape == (2, 16, 10, 12)
    assert torch.isfinite(out).all()


def test_group_norm_falls_back_when_channels_not_divisible_by_groups():
    # 5 output channels is not divisible by the default 8 groups; the
    # block should fall back to 1 group rather than raising.
    encoder = _make_encoder(channels=(5,), norm="group", norm_groups=8)
    x = torch.randn(2, NEER_N_CHANNELS, 6, 6)
    out = encoder(x)
    assert out.shape == (2, 5, 6, 6)


def test_batch_size_one_with_group_norm_does_not_error():
    # Batch-size-1 is exactly the case GroupNorm (unlike BatchNorm) must
    # tolerate — see the "CPU-friendly" rationale in encoder.py.
    encoder = _make_encoder(channels=(16,), norm="group")
    x = torch.randn(1, NEER_N_CHANNELS, 10, 10)
    out = encoder(x)
    assert out.shape == (1, 16, 10, 10)


def test_dropout_zero_is_deterministic_in_train_mode():
    encoder = _make_encoder(channels=(16,), dropout=0.0, norm="none")
    encoder.train()
    x = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    out1 = encoder(x)
    out2 = encoder(x)
    assert torch.allclose(out1, out2)


def test_dropout_is_disabled_in_eval_mode():
    encoder = _make_encoder(channels=(16,), dropout=0.5, norm="none")
    encoder.eval()
    x = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    out1 = encoder(x)
    out2 = encoder(x)
    assert torch.allclose(out1, out2)


def test_dropout_nonzero_varies_across_calls_in_train_mode():
    encoder = _make_encoder(channels=(32,), dropout=0.5, norm="none")
    encoder.train()
    x = torch.randn(4, NEER_N_CHANNELS, 16, 16)
    out1 = encoder(x)
    out2 = encoder(x)
    assert not torch.allclose(out1, out2)


def test_rejects_wrong_number_of_input_channels():
    encoder = _make_encoder()
    x = torch.randn(2, NEER_N_CHANNELS + 1, 10, 10)
    with pytest.raises(ValueError, match="input channels"):
        encoder(x)


def test_rejects_non_4d_input():
    encoder = _make_encoder()
    x = torch.randn(NEER_N_CHANNELS, 10, 10)  # missing batch dim
    with pytest.raises(ValueError, match="4D"):
        encoder(x)


def test_gradients_flow_to_all_parameters():
    encoder = _make_encoder(channels=(16, 32))
    x = torch.randn(2, NEER_N_CHANNELS, 10, 10, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"no gradient reached parameter {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_cpu_friendly_parameter_count_is_modest():
    # Not a hard architectural limit, just a guard against the default
    # config silently growing into something that stops being CPU
    # friendly (see module docstring).
    encoder = _make_encoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    assert n_params < 200_000


def test_runs_on_cpu_device_explicitly():
    device = torch.device("cpu")
    encoder = _make_encoder(channels=(16, 32)).to(device)
    x = torch.randn(2, NEER_N_CHANNELS, 10, 10, device=device)
    out = encoder(x)
    assert out.device.type == "cpu"
    assert out.shape == (2, 32, 10, 10)


def test_repr_contains_key_config_fields():
    encoder = _make_encoder(channels=(16, 32), norm="batch", activation="gelu")
    text = repr(encoder)
    assert "CNNEncoder" in text
    assert "16" in text and "32" in text
    assert "batch" in text
    assert "gelu" in text


def test_real_neer_grid_shape_end_to_end():
    # The actual NEER domain grid from configs/base.yaml: 5N-30N,
    # 45E-105E at 0.25 degrees -> 101 x 241.
    encoder = _make_encoder(channels=(32, 64))
    x = torch.randn(1, NEER_N_CHANNELS, 101, 241)
    out = encoder(x)
    assert out.shape == (1, 64, 101, 241)
