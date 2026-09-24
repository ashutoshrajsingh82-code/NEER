"""Tests for Phase 21B self-supervised pretraining pipeline.

Covers all required areas:
1. Pretraining model initialization
2. CNN -> ViT -> embedding -> decoder forward pass
3. Output shape correctness
4. Masked reconstruction loss
5. Invalid/land-cell masking
6. Training loop execution on a tiny dataset
7. Training history generation
8. Checkpoint saving
9. Checkpoint loading (including reuse by the main NEERModel)
10. Configuration handling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.pretrain_decoder import ReconstructionDecoderConfig  # noqa: E402
from src.models.pretrain_encoder import PretrainEncoderConfig  # noqa: E402
from src.models.pretrain_reconstruction import (  # noqa: E402
    MaskedReconstructionLossConfig,
    PretrainOutput,
    PretrainReconstructionConfig,
)
from src.models.vit import DEFAULT_EMBED_DIM, ViTConfig  # noqa: E402

# --------------------------------------------------------------------------
# Config & CLI handling tests — no torch required
# --------------------------------------------------------------------------


def test_configuration_handling_demo():
    from pretrain import build_parser, parse_environment
    from src.utils.config import load_config

    assert parse_environment("configs/demo.yaml") == "demo"
    assert parse_environment("demo") == "demo"
    assert parse_environment("configs/base.yaml") == "base"

    cfg = load_config("demo")
    assert cfg.environment == "demo"
    assert cfg.demo.enabled is True
    assert cfg.demo.seed == 42


def test_cli_parser_defaults():
    from pretrain import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.config == "configs/demo.yaml"
    assert args.epochs == 5
    assert args.batch_size == 4
    assert args.masking_ratio == 0.5
    assert args.device == "auto"


def test_custom_pretrain_config_handling():
    cfg = PretrainReconstructionConfig(
        embedding_dim=128,
        masking_ratio=0.35,
        decoder_dimensions=(64, 32),
    )
    assert cfg.embedding_dim == 128
    assert cfg.masking_ratio == 0.35
    assert cfg.resolved_decoder_config().decoder_channels == (64, 32)
    assert cfg.resolved_loss_config().masking_ratio == 0.35


# --------------------------------------------------------------------------
# Model & Pipeline tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from pretrain import main as pretrain_cli_main  # noqa: E402
from src.models.neer_model import NEERModel  # noqa: E402
from src.models.pretrain_decoder import ReconstructionDecoder  # noqa: E402
from src.models.pretrain_encoder import PretrainEncoder  # noqa: E402
from src.models.pretrain_reconstruction import (  # noqa: E402
    MaskedReconstructionLoss,
    PretrainReconstructionModel,
    masked_reconstruction_loss,
)
from src.training.pretrain import (  # noqa: E402
    load_pretrained_encoder,
    load_pretrained_into_neer_model,
    load_training_history,
    run_pretrain_epoch,
    save_pretrained_encoder_checkpoint,
    save_training_history,
    train_pretrain,
)


def _tiny_model(**kwargs) -> PretrainReconstructionModel:
    cnn_cfg = CNNEncoderConfig(channels=(8, 16), dropout=0.0)
    embedding_dim = kwargs.pop("embedding_dim", 32)
    masking_ratio = kwargs.pop("masking_ratio", 0.5)
    decoder_dimensions = kwargs.pop("decoder_dimensions", (32, 16))

    vit_cfg = ViTConfig(
        in_channels=16,
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


# --- 1. Pretraining model initialization ------------------------------------


def test_pretraining_model_initialization_defaults():
    model = PretrainReconstructionModel()
    assert isinstance(model.encoder, PretrainEncoder)
    assert isinstance(model.decoder, ReconstructionDecoder)
    assert isinstance(model.loss_fn, MaskedReconstructionLoss)
    assert model.in_channels == NEER_N_CHANNELS
    assert model.out_channels == NEER_N_CHANNELS
    assert model.embed_dim == DEFAULT_EMBED_DIM
    assert model.masking_ratio == 0.5


def test_pretraining_model_initialization_custom():
    model = _tiny_model(embedding_dim=64, masking_ratio=0.25, decoder_dimensions=(48, 24))
    assert model.embed_dim == 64
    assert model.masking_ratio == 0.25
    assert model.decoder.config.decoder_channels == (48, 24)


def test_pretraining_model_parameter_count_modest():
    model = PretrainReconstructionModel()
    n_params = sum(p.numel() for p in model.parameters())
    # Lightweight CPU-friendly model (< 5M parameters)
    assert n_params < 5_000_000


# --- 2. CNN -> ViT -> embedding -> decoder forward pass ---------------------


def test_complete_forward_pass_stages():
    model = _tiny_model().eval()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 16)
    with torch.no_grad():
        out = model(x)
        cnn_feat = model.encoder.cnn(x)
        vit_tokens = model.encoder.vit(cnn_feat)
        embedding = vit_tokens.mean(dim=1)
        reconstruction = model.decoder(embedding, spatial_shape=(12, 16))

    assert torch.equal(out.embedding, embedding)
    assert torch.equal(out.reconstruction, reconstruction)


def test_forward_pass_exposes_embedding_and_reconstruction():
    model = _tiny_model().eval()
    x = torch.randn(2, NEER_N_CHANNELS, 10, 14)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, PretrainOutput)
    assert out.embedding.shape == (2, 32)
    assert out.reconstruction.shape == (2, NEER_N_CHANNELS, 10, 14)
    assert torch.equal(out.latent_embedding, out.embedding)
    assert torch.equal(out.reconstructed_surface_fields, out.reconstruction)

    # Tuple unpacking
    rec, emb = out
    assert torch.equal(rec, out.reconstruction)
    assert torch.equal(emb, out.embedding)


# --- 3. Output shape correctness --------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("spatial_shape", [(8, 8), (12, 16), (20, 24)])
def test_output_shape_correctness(batch, spatial_shape):
    model = _tiny_model().eval()
    h, w = spatial_shape
    x = torch.randn(batch, NEER_N_CHANNELS, h, w)
    out = model(x)
    assert out.reconstruction.shape == (batch, NEER_N_CHANNELS, h, w)
    assert out.embedding.shape == (batch, 32)


# --- 4. Masked reconstruction loss ------------------------------------------


def test_masked_reconstruction_loss_identical_is_zero():
    t = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    loss = masked_reconstruction_loss(t, t, masking_ratio=0.5)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_masked_reconstruction_loss_different_ratios():
    pred = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    target = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    for ratio in [0.1, 0.25, 0.5, 0.75, 1.0]:
        loss = masked_reconstruction_loss(pred, target, masking_ratio=ratio)
        assert torch.isfinite(loss)
        assert loss > 0.0


# --- 5. Invalid / land-cell masking -----------------------------------------


def test_loss_ignores_land_cells():
    pred = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    target = torch.randn(2, NEER_N_CHANNELS, 8, 8)

    land_mask = torch.zeros(8, 8, dtype=torch.bool)
    land_mask[:4, :] = True  # Top half is land

    loss_clean = masked_reconstruction_loss(pred, target, land_mask=land_mask, masking_ratio=1.0)

    # Corrupt land cells completely
    corrupted_target = target.clone()
    corrupted_target[:, :, :4, :] = 99999.0
    corrupted_pred = pred.clone()
    corrupted_pred[:, :, :4, :] = -99999.0

    loss_corrupted = masked_reconstruction_loss(
        corrupted_pred, corrupted_target, land_mask=land_mask, masking_ratio=1.0
    )
    assert torch.isclose(loss_clean, loss_corrupted, atol=1e-5)


def test_loss_ignores_nan_cells():
    pred = torch.randn(2, NEER_N_CHANNELS, 8, 8)
    target = torch.randn(2, NEER_N_CHANNELS, 8, 8)

    valid_mask = torch.ones(8, 8, dtype=torch.bool)
    valid_mask[0, 0] = False
    ref_loss = masked_reconstruction_loss(pred, target, mask=valid_mask, masking_ratio=1.0)

    corrupted_target = target.clone()
    corrupted_target[:, :, 0, 0] = float("nan")
    nan_loss = masked_reconstruction_loss(pred, corrupted_target, masking_ratio=1.0)

    assert not torch.isnan(nan_loss)
    assert torch.isclose(ref_loss, nan_loss, atol=1e-5)


# --- 6. Training loop execution on a tiny dataset ---------------------------


def test_training_loop_execution_on_tiny_dataset(tmp_path):
    model = _tiny_model()
    # Tiny dataset with 4 samples
    x = torch.randn(4, NEER_N_CHANNELS, 8, 8)
    dataset = TensorDataset(x)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    ckpt_file = tmp_path / "encoder_pretrained.pt"
    hist_file = tmp_path / "pretrain_history.json"

    trained_model, history = train_pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        lr=1e-3,
        masking_ratio=0.5,
        checkpoint_path=ckpt_file,
        history_path=hist_file,
        device="cpu",
        verbose=False,
    )

    assert len(history) == 3
    assert ckpt_file.exists()
    assert hist_file.exists()
    for rec in history:
        assert rec["train_loss"] > 0
        assert rec["val_loss"] > 0


def test_run_pretrain_epoch_train_and_eval():
    model = _tiny_model()
    loader = DataLoader(TensorDataset(torch.randn(4, NEER_N_CHANNELS, 8, 8)), batch_size=2)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train mode
    train_metrics = run_pretrain_epoch(model, loader, device, optimizer=optimizer)
    assert "loss" in train_metrics
    assert "reconstruction_mse" in train_metrics
    assert train_metrics["loss"] > 0

    # Eval mode
    eval_metrics = run_pretrain_epoch(model, loader, device, optimizer=None)
    assert "loss" in eval_metrics
    assert eval_metrics["loss"] > 0


# --- 7. Training history generation -----------------------------------------


def test_training_history_generation_and_serialization(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 0.45, "train_mse": 0.45, "val_loss": 0.48, "learning_rate": 0.001, "time_seconds": 0.1},
        {"epoch": 2, "train_loss": 0.38, "train_mse": 0.38, "val_loss": 0.41, "learning_rate": 0.001, "time_seconds": 0.1},
    ]
    out_file = tmp_path / "history.json"
    save_training_history(history, out_file)
    assert out_file.exists()

    loaded = load_training_history(out_file)
    assert len(loaded) == 2
    assert loaded[0]["epoch"] == 1
    assert loaded[1]["train_loss"] == 0.38


# --- 8. Checkpoint saving ---------------------------------------------------


def test_checkpoint_saving_structure(tmp_path):
    model = _tiny_model()
    ckpt_path = tmp_path / "checkpoints" / "encoder_pretrained.pt"
    save_pretrained_encoder_checkpoint(
        path=ckpt_path,
        model=model,
        epoch=2,
        train_loss=0.35,
        val_loss=0.40,
        history=[{"epoch": 1}, {"epoch": 2}],
    )

    assert ckpt_path.exists()
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert data["epoch"] == 2
    assert "encoder_state_dict" in data
    assert "cnn_state_dict" in data
    assert "vit_state_dict" in data
    assert "decoder_state_dict" in data
    assert "model_state_dict" in data
    assert data["embedding_dim"] == 32
    assert data["in_channels"] == NEER_N_CHANNELS


# --- 9. Checkpoint loading & reuse by NEERModel ------------------------------


def test_checkpoint_loading_into_pretrain_encoder(tmp_path):
    model = _tiny_model()
    ckpt_path = tmp_path / "encoder.pt"
    save_pretrained_encoder_checkpoint(ckpt_path, model)

    # Load into a fresh PretrainEncoder
    fresh_encoder = PretrainEncoder(model.config.encoder_config)
    load_pretrained_encoder(ckpt_path, target=fresh_encoder)

    # Test numerical equality of forward embeddings
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    with torch.no_grad():
        orig_emb = model.encoder(x)
        loaded_emb = fresh_encoder(x)
    assert torch.equal(orig_emb, loaded_emb)


def test_checkpoint_loading_into_neer_model(tmp_path):
    pretrain_model = _tiny_model()
    ckpt_path = tmp_path / "encoder.pt"
    save_pretrained_encoder_checkpoint(ckpt_path, pretrain_model)

    # Initialize a NEERModel matching the encoder dimensions
    neer_model = NEERModel(
        cnn_config=pretrain_model.config.encoder_config.cnn_config,
        vit_config=pretrain_model.config.encoder_config.resolved_vit_config(),
    )

    # Load pretrained weights into NEERModel
    load_pretrained_into_neer_model(ckpt_path, neer_model)

    # Verify CNN and ViT state dicts match exactly
    for k, v in pretrain_model.encoder.cnn.state_dict().items():
        assert torch.equal(neer_model.encoder.cnn.state_dict()[k], v)
    for k, v in pretrain_model.encoder.vit.state_dict().items():
        assert torch.equal(neer_model.encoder.vit.state_dict()[k], v)

    # Verify NEERModel.get_embedding(x) equals pretrain_model.encoder(x)
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    with torch.no_grad():
        emb_pretrain = pretrain_model.encoder(x)
        emb_neer = neer_model.get_embedding(x)
        anomalies = neer_model(x)

    assert torch.equal(emb_pretrain, emb_neer)
    assert anomalies.shape == (2, 15)


def test_neer_model_load_pretrained_encoder_method(tmp_path):
    pretrain_model = _tiny_model()
    ckpt_path = tmp_path / "encoder.pt"
    save_pretrained_encoder_checkpoint(ckpt_path, pretrain_model)

    neer_model = NEERModel(
        cnn_config=pretrain_model.config.encoder_config.cnn_config,
        vit_config=pretrain_model.config.encoder_config.resolved_vit_config(),
    )
    neer_model.load_pretrained_encoder(ckpt_path)

    x = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    with torch.no_grad():
        assert torch.equal(pretrain_model.encoder(x), neer_model.get_embedding(x))


def test_checkpoint_loading_creates_encoder_when_target_none(tmp_path):
    model = _tiny_model()
    ckpt_path = tmp_path / "encoder.pt"
    save_pretrained_encoder_checkpoint(ckpt_path, model)

    loaded_enc = load_pretrained_encoder(ckpt_path, target=None)
    assert isinstance(loaded_enc, PretrainEncoder)
    assert loaded_enc.embed_dim == 32


# --- 10. CLI execution with demo config --------------------------------------


def test_cli_execution_with_demo_config(tmp_path):
    hist_file = tmp_path / "hist.json"
    ckpt_file = tmp_path / "encoder_pretrained.pt"

    ret = pretrain_cli_main([
        "--config", "configs/demo.yaml",
        "--epochs", "1",
        "--batch-size", "4",
        "--checkpoint-path", str(ckpt_file),
        "--history-path", str(hist_file),
    ])

    assert ret == 0
    assert ckpt_file.exists()
    assert hist_file.exists()
    history = load_training_history(hist_file)
    assert len(history) == 1
    assert "train_loss" in history[0]
