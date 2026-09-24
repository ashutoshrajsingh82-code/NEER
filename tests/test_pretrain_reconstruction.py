"""Tests for Phase 21A.2 reconstruction decoder, complete pretraining model, and masked reconstruction loss.

Layout:
- Config tests that run without torch.
- Model tests skipped via `pytest.importorskip` when torch is not installed:
  1. Decoder initialization
  2. Complete forward pass
  3. Reconstruction output shape
  4. Masked reconstruction MSE
  5. Land/invalid-cell masking
  6. Different masking ratios
  7. Gradient/backpropagation through the model
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.pretrain_decoder import (  # noqa: E402
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_SPATIAL_SHAPE,
    ReconstructionDecoderConfig,
)
from src.models.pretrain_encoder import PretrainEncoderConfig  # noqa: E402
from src.models.pretrain_reconstruction import (  # noqa: E402
    MaskedReconstructionLossConfig,
    PretrainOutput,
    PretrainReconstructionConfig,
)
from src.models.vit import DEFAULT_EMBED_DIM, ViTConfig  # noqa: E402

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_default_decoder_config():
    cfg = ReconstructionDecoderConfig()
    assert cfg.embed_dim == DEFAULT_EMBED_DIM
    assert cfg.out_channels == NEER_N_CHANNELS
    assert cfg.decoder_channels == DEFAULT_DECODER_CHANNELS
    assert cfg.spatial_shape == DEFAULT_SPATIAL_SHAPE
    assert cfg.kernel_size == 3
    assert cfg.norm == "group"
    assert cfg.activation == "gelu"
    assert cfg.dropout == 0.0
    assert cfg.use_coords is True


def test_decoder_config_custom_dimensions():
    cfg1 = ReconstructionDecoderConfig(decoder_dimensions=(64, 32, 16))
    assert cfg1.decoder_channels == (64, 32, 16)

    cfg2 = ReconstructionDecoderConfig(decoder_dimensions=64)
    assert cfg2.decoder_channels == (64, 32)

    cfg3 = ReconstructionDecoderConfig(decoder_channels=(48, 24))
    assert cfg3.decoder_channels == (48, 24)


@pytest.mark.parametrize("embed_dim", [0, -1, -256])
def test_decoder_config_rejects_non_positive_embed_dim(embed_dim):
    with pytest.raises(ValueError, match="embed_dim"):
        ReconstructionDecoderConfig(embed_dim=embed_dim)


@pytest.mark.parametrize("out_channels", [0, -1, -11])
def test_decoder_config_rejects_non_positive_out_channels(out_channels):
    with pytest.raises(ValueError, match="out_channels"):
        ReconstructionDecoderConfig(out_channels=out_channels)


def test_decoder_config_rejects_empty_or_non_positive_channels():
    with pytest.raises(ValueError, match="decoder_channels"):
        ReconstructionDecoderConfig(decoder_channels=())
    with pytest.raises(ValueError, match="decoder_channels"):
        ReconstructionDecoderConfig(decoder_channels=(32, -1))


def test_decoder_config_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="kernel_size"):
        ReconstructionDecoderConfig(kernel_size=4)


def test_decoder_config_rejects_invalid_norm_or_activation():
    with pytest.raises(ValueError, match="norm"):
        ReconstructionDecoderConfig(norm="invalid_norm")
    with pytest.raises(ValueError, match="activation"):
        ReconstructionDecoderConfig(activation="invalid_act")


def test_decoder_config_rejects_invalid_dropout_or_spatial_shape():
    with pytest.raises(ValueError, match="dropout"):
        ReconstructionDecoderConfig(dropout=-0.1)
    with pytest.raises(ValueError, match="dropout"):
        ReconstructionDecoderConfig(dropout=1.0)
    with pytest.raises(ValueError, match="spatial_shape"):
        ReconstructionDecoderConfig(spatial_shape=(0, 241))


def test_default_pretrain_reconstruction_config():
    cfg = PretrainReconstructionConfig()
    assert cfg.embedding_dim == DEFAULT_EMBED_DIM
    assert cfg.masking_ratio == 0.5
    assert cfg.encoder_config.embedding_dim == DEFAULT_EMBED_DIM
    assert cfg.resolved_decoder_config().embed_dim == DEFAULT_EMBED_DIM
    assert cfg.resolved_decoder_config().out_channels == NEER_N_CHANNELS
    assert cfg.resolved_loss_config().masking_ratio == 0.5


def test_pretrain_reconstruction_config_configurable():
    cfg = PretrainReconstructionConfig(
        encoder_config=PretrainEncoderConfig(embedding_dim=128),
        embedding_dim=128,
        masking_ratio=0.3,
        decoder_dimensions=(64, 32),
    )
    assert cfg.embedding_dim == 128
    assert cfg.masking_ratio == 0.3
    resolved_dec = cfg.resolved_decoder_config()
    assert resolved_dec.embed_dim == 128
    assert resolved_dec.decoder_channels == (64, 32)
    assert cfg.resolved_loss_config().masking_ratio == 0.3


@pytest.mark.parametrize("ratio", [-0.1, 1.1, 2.0])
def test_pretrain_reconstruction_config_rejects_invalid_masking_ratio(ratio):
    with pytest.raises(ValueError, match="masking_ratio"):
        PretrainReconstructionConfig(masking_ratio=ratio)


def test_pretrain_reconstruction_config_rejects_mismatched_embedding_dim():
    with pytest.raises(ValueError, match="embedding_dim"):
        PretrainReconstructionConfig(
            encoder_config=PretrainEncoderConfig(embedding_dim=64),
            embedding_dim=128,
        )


def test_loss_config_defaults_and_validation():
    cfg = MaskedReconstructionLossConfig()
    assert cfg.masking_ratio == 0.5
    assert cfg.eps == 1e-8
    assert cfg.spatial_masking is True
    assert cfg.reduction == "mean"

    with pytest.raises(ValueError, match="masking_ratio"):
        MaskedReconstructionLossConfig(masking_ratio=-0.05)
    with pytest.raises(ValueError, match="eps"):
        MaskedReconstructionLossConfig(eps=0.0)
    with pytest.raises(ValueError, match="reduction"):
        MaskedReconstructionLossConfig(reduction="invalid")


# --------------------------------------------------------------------------
# Model tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.models.pretrain_decoder import ReconstructionDecoder  # noqa: E402
from src.models.pretrain_reconstruction import (  # noqa: E402
    MaskedReconstructionLoss,
    PretrainReconstructionModel,
    masked_reconstruction_loss,
    masked_reconstruction_mse,
)


def _small_pretrain_model(**overrides) -> PretrainReconstructionModel:
    cnn_cfg = overrides.pop("cnn_config", CNNEncoderConfig(channels=(8, 16), dropout=0.0))
    embedding_dim = overrides.pop("embedding_dim", 32)
    masking_ratio = overrides.pop("masking_ratio", 0.5)
    decoder_dimensions = overrides.pop("decoder_dimensions", (32, 16))

    vit_cfg = ViTConfig(
        in_channels=cnn_cfg.out_channels,
        patch_size=4,
        embed_dim=embedding_dim,
        num_heads=4,
        depth=2,
        dropout=0.0,
    )
    enc_cfg = PretrainEncoderConfig(
        cnn_config=cnn_cfg,
        vit_config=vit_cfg,
        embedding_dim=embedding_dim,
    )
    cfg = PretrainReconstructionConfig(
        encoder_config=enc_cfg,
        embedding_dim=embedding_dim,
        masking_ratio=masking_ratio,
        decoder_dimensions=decoder_dimensions,
    )
    return PretrainReconstructionModel(cfg)


# --- 1. Decoder initialization ----------------------------------------------


def test_decoder_default_initialization():
    decoder = ReconstructionDecoder()
    assert decoder.embed_dim == DEFAULT_EMBED_DIM
    assert decoder.in_channels == DEFAULT_EMBED_DIM
    assert decoder.out_channels == NEER_N_CHANNELS
    assert isinstance(decoder.in_proj, torch.nn.Linear)
    assert isinstance(decoder.out_conv, torch.nn.Conv2d)
    assert len(decoder.blocks) == len(DEFAULT_DECODER_CHANNELS)


def test_decoder_custom_config_initialization():
    cfg = ReconstructionDecoderConfig(
        embed_dim=64,
        out_channels=7,
        decoder_channels=(32, 16),
        activation="relu",
        norm="group",
        dropout=0.1,
    )
    decoder = ReconstructionDecoder(cfg)
    assert decoder.embed_dim == 64
    assert decoder.out_channels == 7
    assert decoder.in_proj.in_features == 64
    assert decoder.in_proj.out_features == 32
    assert decoder.out_conv.out_channels == 7
    assert len(decoder.blocks) == 2


@pytest.mark.parametrize("norm", ["group", "batch", "instance", "none"])
def test_decoder_norm_variants(norm):
    cfg = ReconstructionDecoderConfig(norm=norm, decoder_channels=(16, 8))
    decoder = ReconstructionDecoder(cfg)
    assert isinstance(decoder, torch.nn.Module)


@pytest.mark.parametrize("act", ["relu", "gelu", "silu", "leaky_relu"])
def test_decoder_activation_variants(act):
    cfg = ReconstructionDecoderConfig(activation=act, decoder_channels=(16, 8))
    decoder = ReconstructionDecoder(cfg)
    assert isinstance(decoder, torch.nn.Module)


# --- 2. Complete forward pass -----------------------------------------------


def test_complete_forward_pass_returns_both_embedding_and_reconstruction():
    model = _small_pretrain_model().eval()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 16)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, PretrainOutput)
    assert hasattr(out, "reconstruction")
    assert hasattr(out, "embedding")
    assert out.embedding.shape == (2, 32)
    assert out.reconstruction.shape == (2, NEER_N_CHANNELS, 12, 16)
    assert torch.isfinite(out.embedding).all()
    assert torch.isfinite(out.reconstruction).all()


def test_forward_pass_exposes_aliases_and_tuple_unpacking():
    model = _small_pretrain_model().eval()
    x = torch.randn(2, NEER_N_CHANNELS, 10, 14)
    with torch.no_grad():
        out = model(x)

    # Terminology from Phase 21A.2 prompt
    assert torch.equal(out.latent_embedding, out.embedding)
    assert torch.equal(out.reconstructed_surface_fields, out.reconstruction)

    # Named index and dict access
    assert torch.equal(out[0], out.reconstruction)
    assert torch.equal(out[1], out.embedding)
    assert torch.equal(out["reconstruction"], out.reconstruction)
    assert torch.equal(out["embedding"], out.embedding)

    # Tuple unpacking
    rec, emb = out
    assert torch.equal(rec, out.reconstruction)
    assert torch.equal(emb, out.embedding)


def test_complete_forward_pass_matches_encoder_then_decoder():
    # Architecture requirement: surface fields -> CNN -> ViT -> embedding -> decoder -> reconstructed fields
    model = _small_pretrain_model().eval()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 16)
    with torch.no_grad():
        out = model(x)
        expected_emb = model.encoder(x)
        expected_rec = model.decoder(expected_emb, spatial_shape=(12, 16))

    assert torch.equal(out.embedding, expected_emb)
    assert torch.equal(out.reconstruction, expected_rec)


def test_model_encode_and_get_embedding_methods():
    model = _small_pretrain_model().eval()
    x = torch.randn(3, NEER_N_CHANNELS, 12, 12)
    with torch.no_grad():
        emb1 = model.encode(x)
        emb2 = model.get_embedding(x)
        out = model(x)
    assert torch.equal(emb1, out.embedding)
    assert torch.equal(emb2, out.embedding)


def test_forward_pass_with_input_mask():
    model = _small_pretrain_model().eval()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    mask = torch.ones_like(x)
    mask[:, :, 0:6, :] = 0.0  # mask top half

    with torch.no_grad():
        out_unmasked = model(x)
        out_masked = model(x, mask=mask)

    assert out_masked.embedding.shape == (2, 32)
    assert out_masked.reconstruction.shape == (2, NEER_N_CHANNELS, 12, 12)
    # Masking input should produce different embedding than unmasked input
    assert not torch.allclose(out_masked.embedding, out_unmasked.embedding)


def test_default_config_end_to_end_on_real_neer_grid():
    model = PretrainReconstructionModel().eval()
    x = torch.randn(1, NEER_N_CHANNELS, 101, 241)
    with torch.no_grad():
        out = model(x)
    assert out.embedding.shape == (1, 256)
    assert out.reconstruction.shape == (1, NEER_N_CHANNELS, 101, 241)
    assert torch.isfinite(out.embedding).all()
    assert torch.isfinite(out.reconstruction).all()


# --- 3. Reconstruction output shape -----------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_reconstruction_output_shape_across_batch_sizes(batch):
    model = _small_pretrain_model()
    x = torch.randn(batch, NEER_N_CHANNELS, 12, 16)
    out = model(x)
    assert out.reconstruction.shape == (batch, NEER_N_CHANNELS, 12, 16)
    assert out.embedding.shape == (batch, 32)


@pytest.mark.parametrize(
    "spatial_shape",
    [(8, 8), (12, 16), (20, 30), (10, 14), (101, 241)],
)
def test_reconstruction_output_shape_across_spatial_dimensions(spatial_shape):
    model = _small_pretrain_model()
    h, w = spatial_shape
    x = torch.randn(2, NEER_N_CHANNELS, h, w)
    out = model(x)
    assert out.reconstruction.shape == (2, NEER_N_CHANNELS, h, w)


def test_decoder_standalone_output_shape():
    decoder = ReconstructionDecoder(
        ReconstructionDecoderConfig(embed_dim=32, out_channels=11, decoder_channels=(16, 8))
    )
    embedding = torch.randn(3, 32)
    out1 = decoder(embedding, spatial_shape=(15, 25))
    assert out1.shape == (3, 11, 15, 25)

    # Defaults to config spatial shape when not passed
    out2 = decoder(embedding)
    assert out2.shape == (3, 11, 101, 241)


def test_decoder_rejects_invalid_inputs():
    decoder = ReconstructionDecoder(ReconstructionDecoderConfig(embed_dim=32))
    # 3D embedding rejected
    with pytest.raises(ValueError, match="2-D"):
        decoder(torch.randn(2, 32, 1))
    # Wrong embedding dim rejected
    with pytest.raises(ValueError, match="embed_dim"):
        decoder(torch.randn(2, 16))
    # Non-positive spatial shape rejected
    with pytest.raises(ValueError, match="spatial_shape"):
        decoder(torch.randn(2, 32), spatial_shape=(0, 10))


# --- 4. Masked reconstruction MSE -------------------------------------------


def test_masked_reconstruction_mse_identical_tensors_zero_loss():
    t = torch.randn(2, NEER_N_CHANNELS, 10, 10)
    loss = masked_reconstruction_loss(t, t)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_masked_reconstruction_mse_numerical_correctness():
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.5, 2.0], [3.0, 6.0]]]])
    # (0.5^2 + 0.0^2 + 0.0^2 + 2.0^2) / 4 = (0.25 + 4.0) / 4 = 1.0625
    loss = masked_reconstruction_loss(pred, target, masking_ratio=1.0)
    assert torch.isclose(loss, torch.tensor(1.0625))


def test_loss_module_matches_functional_and_alias():
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    loss1 = masked_reconstruction_loss(pred, target, masking_ratio=1.0)
    loss2 = masked_reconstruction_mse(pred, target, masking_ratio=1.0)
    loss3 = MaskedReconstructionLoss(masking_ratio=1.0)(pred, target)
    assert torch.equal(loss1, loss2)
    assert torch.equal(loss1, loss3)


def test_model_compute_loss_matches_loss_fn():
    model = _small_pretrain_model()
    pred = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    target = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    loss_direct = model.loss_fn(pred, target, masking_ratio=1.0)
    loss_method = model.compute_loss(pred, target, masking_ratio=1.0)
    assert torch.equal(loss_direct, loss_method)


def test_all_invalid_cells_returns_zero_without_nan():
    pred = torch.randn(2, 3, 5, 5)
    target = torch.full((2, 3, 5, 5), float("nan"))
    loss = masked_reconstruction_loss(pred, target)
    assert torch.isclose(loss, torch.tensor(0.0))
    assert not torch.isnan(loss)


def test_reduction_sum_vs_mean():
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    # squared diffs: 1, 4, 9, 16 -> sum = 30, mean = 7.5
    loss_mean = masked_reconstruction_loss(pred, target, masking_ratio=1.0, reduction="mean")
    loss_sum = masked_reconstruction_loss(pred, target, masking_ratio=1.0, reduction="sum")
    assert torch.isclose(loss_mean, torch.tensor(7.5))
    assert torch.isclose(loss_sum, torch.tensor(30.0))


# --- 5. Land/invalid-cell masking -------------------------------------------


def test_loss_ignores_land_cells_with_land_mask():
    pred = torch.randn(2, NEER_N_CHANNELS, 10, 10)
    target = torch.randn(2, NEER_N_CHANNELS, 10, 10)

    # Mark top 5 rows as land (True = land)
    land_mask = torch.zeros(10, 10, dtype=torch.bool)
    land_mask[0:5, :] = True

    loss_clean = masked_reconstruction_loss(pred, target, land_mask=land_mask, masking_ratio=1.0)

    # Corrupt target and pred heavily on land cells
    corrupted_target = target.clone()
    corrupted_pred = pred.clone()
    corrupted_target[:, :, 0:5, :] = 999999.0
    corrupted_pred[:, :, 0:5, :] = -999999.0

    loss_corrupted = masked_reconstruction_loss(
        corrupted_pred, corrupted_target, land_mask=land_mask, masking_ratio=1.0
    )

    # Land cells MUST NOT contribute to the loss
    assert torch.isclose(loss_clean, loss_corrupted, atol=1e-5)


def test_loss_ignores_land_cells_with_ocean_mask():
    pred = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    target = torch.randn(2, NEER_N_CHANNELS, 8, 8)

    # Ocean mask: True = ocean, False = land
    ocean_mask = torch.zeros(8, 8, dtype=torch.bool)
    ocean_mask[4:, 4:] = True  # only lower-right quadrant is ocean

    loss_clean = masked_reconstruction_loss(pred, target, ocean_mask=ocean_mask, masking_ratio=1.0)

    # Corrupt land cells (lower ocean_mask is False)
    corrupted_target = target.clone()
    corrupted_target[:, :, :4, :] = 1e6

    loss_corrupted = masked_reconstruction_loss(
        pred, corrupted_target, ocean_mask=ocean_mask, masking_ratio=1.0
    )
    assert torch.isclose(loss_clean, loss_corrupted, atol=1e-5)


def test_loss_ignores_nan_and_invalid_target_cells():
    pred = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    target = torch.randn(2, NEER_N_CHANNELS, 8, 8)

    # Exclude cell (0, 0) explicitly
    valid_mask = torch.ones(8, 8, dtype=torch.bool)
    valid_mask[0, 0] = False
    loss_reference = masked_reconstruction_loss(pred, target, mask=valid_mask, masking_ratio=1.0)

    # Inject NaN and Inf at (0, 0) in target
    corrupted_target = target.clone()
    corrupted_target[:, :, 0, 0] = float("nan")

    loss_with_nan = masked_reconstruction_loss(pred, corrupted_target, masking_ratio=1.0)

    assert not torch.isnan(loss_with_nan)
    assert not torch.isinf(loss_with_nan)
    assert torch.isclose(loss_reference, loss_with_nan, atol=1e-5)


def test_land_is_ocean_flag():
    pred = torch.randn(2, 4, 6, 6)
    target = torch.randn(2, 4, 6, 6)
    # Under legacy NEER convention, land_mask has True = ocean
    ocean_as_land_mask = torch.zeros(6, 6, dtype=torch.bool)
    ocean_as_land_mask[3:, :] = True

    loss_flag = masked_reconstruction_loss(
        pred, target, land_mask=ocean_as_land_mask, land_is_ocean=True, masking_ratio=1.0
    )
    loss_ocean = masked_reconstruction_loss(
        pred, target, ocean_mask=ocean_as_land_mask, masking_ratio=1.0
    )
    assert torch.equal(loss_flag, loss_ocean)


# --- 6. Different masking ratios -------------------------------------------


@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_loss_supports_different_masking_ratios(ratio):
    pred = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    target = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    loss = masked_reconstruction_loss(pred, target, masking_ratio=ratio)
    assert torch.isfinite(loss)
    if ratio == 0.0:
        assert torch.isclose(loss, torch.tensor(0.0))
    else:
        assert loss > 0.0


def test_deterministic_masking_with_generator():
    pred = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    target = torch.randn(2, NEER_N_CHANNELS, 12, 12)

    g1 = torch.Generator().manual_seed(42)
    loss1 = masked_reconstruction_loss(pred, target, masking_ratio=0.5, generator=g1)

    g2 = torch.Generator().manual_seed(42)
    loss2 = masked_reconstruction_loss(pred, target, masking_ratio=0.5, generator=g2)

    assert torch.equal(loss1, loss2)


def test_explicit_target_mask_overrides_random_masking():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    target_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    target_mask[0, 0, 0, 0] = True  # exactly one cell

    loss = masked_reconstruction_loss(pred, target, target_mask=target_mask, masking_ratio=0.5)
    # Only cell (0, 0) evaluated: (0 - 1)^2 / 1 = 1.0
    assert torch.isclose(loss, torch.tensor(1.0))


# --- 7. Gradient / backpropagation through the model -------------------------


def test_gradients_flow_through_entire_pretraining_model():
    model = _small_pretrain_model()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12, requires_grad=True)
    target = torch.randn(2, NEER_N_CHANNELS, 12, 12)

    out = model(x)
    loss = masked_reconstruction_loss(out.reconstruction, target, masking_ratio=0.5)
    loss.backward()

    # Input gradient
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Encoder CNN gradients
    for name, param in model.encoder.cnn.named_parameters():
        assert param.grad is not None, f"no grad reached encoder CNN {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    # Encoder ViT gradients
    for name, param in model.encoder.vit.named_parameters():
        assert param.grad is not None, f"no grad reached encoder ViT {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    # Decoder in_proj gradients
    assert model.decoder.in_proj.weight.grad is not None
    assert torch.isfinite(model.decoder.in_proj.weight.grad).all()

    # Decoder block gradients
    for i, block in enumerate(model.decoder.blocks):
        for name, param in block.named_parameters():
            assert param.grad is not None, f"no grad reached decoder block {i} {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for decoder block {i} {name}"

    # Decoder out_conv gradients
    assert model.decoder.out_conv.weight.grad is not None
    assert torch.isfinite(model.decoder.out_conv.weight.grad).all()


def test_gradients_flow_from_embedding_alone():
    model = _small_pretrain_model()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12, requires_grad=True)
    out = model(x)
    out.embedding.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in model.encoder.named_parameters():
        assert param.grad is not None, f"no grad for {name}"


def test_land_cells_produce_zero_gradient_on_inputs():
    model = _small_pretrain_model().eval()
    x = torch.randn(1, NEER_N_CHANNELS, 8, 8, requires_grad=True)
    target = torch.randn(1, NEER_N_CHANNELS, 8, 8)

    # Land mask covering bottom half
    land_mask = torch.zeros(8, 8, dtype=torch.bool)
    land_mask[4:, :] = True

    out = model(x)
    loss = masked_reconstruction_loss(out.reconstruction, target, land_mask=land_mask, masking_ratio=1.0)
    loss.backward()

    # Input gradient must exist and be finite
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Loss gradient with respect to output on land cells is zero
    diff = torch.where(~land_mask, out.reconstruction - target, torch.zeros_like(target))
    assert (diff[:, :, 4:, :] == 0.0).all()
