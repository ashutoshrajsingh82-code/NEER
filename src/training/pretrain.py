"""
Phase 21B — Pretraining Training Loop, History Tracking & Checkpointing.

Provides the end-to-end training engine for the self-supervised reconstruction
architecture created in Phase 21A:

    surface fields -> CNN -> ViT -> embedding -> decoder -> reconstructed surface fields

Key features:
1. Training loop using masked reconstruction MSE loss.
2. Per-epoch history tracking (epoch, train loss, val loss, individual MSE components,
   learning rate, duration).
3. Reusable history serialization as JSON under the artifacts directory.
4. Pretrained encoder checkpoint saving to `artifacts/checkpoints/encoder_pretrained.pt`.
5. Checkpoint loading enabling pretrained CNN and ViT encoder weights to be reused
   directly by `PretrainEncoder`, `PretrainReconstructionModel`, or the main `NEERModel`.

torch is optional
------------------
Importing this module never requires torch; only running training or loading checkpoints
raises `MissingDependencyError` when torch is absent.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.data.loaders.errors import MissingDependencyError

try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="the NEER pretraining engine (src.training.pretrain)",
            install_hint="pip install torch",
        )


logger = logging.getLogger("neer.pretrain")

#: Default checkpoint path for the pretrained encoder
DEFAULT_PRETRAIN_CHECKPOINT_PATH = Path("artifacts/checkpoints/encoder_pretrained.pt")

#: Default history path for pretraining metrics
DEFAULT_PRETRAIN_HISTORY_PATH = Path("artifacts/pretrain_history.json")


# --------------------------------------------------------------------------
# Checkpointing: Saving & Loading
# --------------------------------------------------------------------------


def save_pretrained_encoder_checkpoint(
    path: Union[str, Path],
    model: Any,
    optimizer: Optional[Any] = None,
    epoch: int = 1,
    history: Optional[List[Dict[str, Any]]] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    best_loss: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save pretrained encoder checkpoint so weights can be reused by NEER models.

    Parameters
    ----------
    path:
        Destination file path (e.g. `artifacts/checkpoints/encoder_pretrained.pt`).
    model:
        `PretrainReconstructionModel` or `PretrainEncoder`.
    optimizer:
        Optional optimizer whose state dict is preserved.
    epoch:
        Completed epoch count.
    history:
        List of per-epoch metric dictionaries.
    train_loss:
        Final or best training loss.
    val_loss:
        Final or best validation loss.
    best_loss:
        Best loss observed.
    extra_metadata:
        Optional dictionary of additional metadata.

    Returns
    -------
    Resolved `Path` to saved checkpoint.
    """
    _require_torch()
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine encoder submodule
    if hasattr(model, "encoder"):
        encoder = model.encoder
        decoder_state = model.decoder.state_dict() if hasattr(model, "decoder") else None
        model_state = model.state_dict()
    else:
        encoder = model
        decoder_state = None
        model_state = model.state_dict()

    from dataclasses import asdict

    encoder_state = encoder.state_dict()
    cnn_state = encoder.cnn.state_dict() if hasattr(encoder, "cnn") else None
    vit_state = encoder.vit.state_dict() if hasattr(encoder, "vit") else None
    cnn_cfg_dict = asdict(encoder.cnn.config) if hasattr(encoder, "cnn") and hasattr(encoder.cnn, "config") else None
    vit_cfg_dict = asdict(encoder.vit.config) if hasattr(encoder, "vit") and hasattr(encoder.vit, "config") else None

    checkpoint_data: Dict[str, Any] = {
        "epoch": epoch,
        "encoder_state_dict": encoder_state,
        "cnn_state_dict": cnn_state,
        "vit_state_dict": vit_state,
        "decoder_state_dict": decoder_state,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_loss": best_loss,
        "history": history or [],
        "embedding_dim": getattr(encoder, "embed_dim", getattr(model, "embed_dim", None)),
        "in_channels": getattr(encoder, "in_channels", getattr(model, "in_channels", None)),
        "masking_ratio": getattr(model, "masking_ratio", None),
        "cnn_config": cnn_cfg_dict,
        "vit_config": vit_cfg_dict,
        "extra_metadata": extra_metadata or {},
    }

    torch.save(checkpoint_data, target_path)
    logger.info("Saved pretrained encoder checkpoint to %s", target_path)
    return target_path


def load_pretrained_encoder(
    checkpoint: Union[str, Path, Dict[str, Any]],
    target: Optional[Any] = None,
    strict: bool = True,
    device: Optional[Union[str, Any]] = None,
) -> Any:
    """Load pretrained CNN/ViT weights from a pretraining checkpoint.

    Parameters
    ----------
    checkpoint:
        File path to `.pt` checkpoint or pre-loaded checkpoint dictionary.
    target:
        Optional destination model instance (`PretrainEncoder`,
        `PretrainReconstructionModel`, or `NEERModel`). If None, a new
        `PretrainEncoder` is initialized and returned.
    strict:
        Whether to enforce exact key matching during `load_state_dict`.
    device:
        Device to map checkpoint tensors to (defaults to CPU).

    Returns
    -------
    The model instance with pretrained encoder weights loaded.
    """
    _require_torch()
    if isinstance(checkpoint, (str, Path)):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        data = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    elif isinstance(checkpoint, dict):
        data = checkpoint
    else:
        raise TypeError(f"checkpoint must be a path or dict, got {type(checkpoint)}")

    cnn_state = data.get("cnn_state_dict")
    vit_state = data.get("vit_state_dict")
    encoder_state = data.get("encoder_state_dict")

    # If target is None, construct a PretrainEncoder matching saved architecture
    if target is None:
        from src.models.encoder import CNNEncoderConfig
        from src.models.pretrain_encoder import PretrainEncoder, PretrainEncoderConfig
        from src.models.vit import ViTConfig

        embed_dim = data.get("embedding_dim", 256)
        cnn_cfg_data = data.get("cnn_config")
        vit_cfg_data = data.get("vit_config")
        cnn_cfg = CNNEncoderConfig(**cnn_cfg_data) if cnn_cfg_data else CNNEncoderConfig()
        vit_cfg = ViTConfig(**vit_cfg_data) if vit_cfg_data else None
        config = PretrainEncoderConfig(
            cnn_config=cnn_cfg,
            vit_config=vit_cfg,
            embedding_dim=embed_dim,
        )
        target = PretrainEncoder(config)

    # 1. Target is a NEERModel: load into target.encoder.cnn and target.encoder.vit
    from src.models.neer_model import NEERModel

    if isinstance(target, NEERModel):
        if cnn_state is not None:
            target.encoder.cnn.load_state_dict(cnn_state, strict=strict)
        if vit_state is not None:
            target.encoder.vit.load_state_dict(vit_state, strict=strict)
        logger.info("Successfully loaded pretrained CNN and ViT encoder weights into NEERModel")
        return target

    # 2. Target is a PretrainReconstructionModel
    from src.models.pretrain_reconstruction import PretrainReconstructionModel

    if isinstance(target, PretrainReconstructionModel):
        if encoder_state is not None:
            target.encoder.load_state_dict(encoder_state, strict=strict)
        elif cnn_state is not None and vit_state is not None:
            target.encoder.cnn.load_state_dict(cnn_state, strict=strict)
            target.encoder.vit.load_state_dict(vit_state, strict=strict)
        logger.info("Successfully loaded pretrained encoder weights into PretrainReconstructionModel")
        return target

    # 3. Target is a PretrainEncoder
    from src.models.pretrain_encoder import PretrainEncoder

    if isinstance(target, PretrainEncoder):
        if encoder_state is not None:
            target.load_state_dict(encoder_state, strict=strict)
        elif cnn_state is not None and vit_state is not None:
            target.cnn.load_state_dict(cnn_state, strict=strict)
            target.vit.load_state_dict(vit_state, strict=strict)
        logger.info("Successfully loaded pretrained encoder weights into PretrainEncoder")
        return target

    # 4. Fallback for any model holding `.encoder` or `.cnn`/`.vit`
    if hasattr(target, "encoder"):
        if hasattr(target.encoder, "cnn") and cnn_state is not None:
            target.encoder.cnn.load_state_dict(cnn_state, strict=strict)
        if hasattr(target.encoder, "vit") and vit_state is not None:
            target.encoder.vit.load_state_dict(vit_state, strict=strict)
        return target

    if hasattr(target, "cnn") and cnn_state is not None:
        target.cnn.load_state_dict(cnn_state, strict=strict)
    if hasattr(target, "vit") and vit_state is not None:
        target.vit.load_state_dict(vit_state, strict=strict)

    return target


def load_pretrained_into_neer_model(
    checkpoint: Union[str, Path, Dict[str, Any]],
    neer_model: Any,
    strict: bool = True,
    device: Optional[Union[str, Any]] = None,
) -> Any:
    """Convenience helper to load pretrained encoder weights directly into a `NEERModel`."""
    return load_pretrained_encoder(checkpoint, target=neer_model, strict=strict, device=device)


# --------------------------------------------------------------------------
# Training History
# --------------------------------------------------------------------------


def save_training_history(
    history: List[Dict[str, Any]],
    path: Union[str, Path] = DEFAULT_PRETRAIN_HISTORY_PATH,
) -> Path:
    """Save training history list as JSON under the artifacts directory.

    Parameters
    ----------
    history:
        List of per-epoch metric dicts.
    path:
        Output JSON file path.

    Returns
    -------
    Resolved `Path` to written JSON file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved pretraining history to %s", out_path)
    return out_path


def load_training_history(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load training history JSON file."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"History file not found at: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Single Epoch Execution
# --------------------------------------------------------------------------


def run_pretrain_epoch(
    model: Any,
    loader: Any,
    device: Any,
    optimizer: Optional[Any] = None,
    masking_ratio: Optional[float] = None,
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """Run one epoch over a DataLoader for self-supervised reconstruction.

    Parameters
    ----------
    model:
        `PretrainReconstructionModel` instance.
    loader:
        `DataLoader` yielding tensor batches or NEER dict batches.
    device:
        `torch.device` to execute on.
    optimizer:
        Optimizer if training, None if evaluating.
    masking_ratio:
        Masking ratio override (defaults to model's configured ratio).
    grad_clip_norm:
        Max gradient norm for clipping (0 or None disables).

    Returns
    -------
    Dictionary of epoch metrics: `{"loss": ..., "reconstruction_mse": ...}`.
    """
    _require_torch()
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_mse = 0.0
    n_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            # Extract inputs and optional validity mask
            if isinstance(batch, dict):
                inputs = batch["inputs"].to(device, non_blocking=True)
                mask_dict = batch.get("mask", {})
                mask = mask_dict.get("input")
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device, non_blocking=True)
                mask = batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
            else:
                inputs = batch.to(device, non_blocking=True)
                mask = None

            batch_size = inputs.shape[0]

            if training:
                optimizer.zero_grad(set_to_none=True)

            out = model(inputs)
            loss = model.compute_loss(
                pred=out.reconstruction,
                target=inputs,
                mask=mask,
                masking_ratio=masking_ratio,
            )

            if training:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            loss_val = float(loss.detach())
            total_loss += loss_val * batch_size
            total_mse += loss_val * batch_size
            n_samples += batch_size

    n_samples = max(n_samples, 1)
    return {
        "loss": total_loss / n_samples,
        "reconstruction_mse": total_mse / n_samples,
    }


# --------------------------------------------------------------------------
# Full Pretraining Loop
# --------------------------------------------------------------------------


def train_pretrain(
    model: Optional[Any] = None,
    train_loader: Optional[Any] = None,
    val_loader: Optional[Any] = None,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    masking_ratio: float = 0.5,
    grad_clip_norm: Optional[float] = 1.0,
    checkpoint_path: Optional[Union[str, Path]] = DEFAULT_PRETRAIN_CHECKPOINT_PATH,
    history_path: Optional[Union[str, Path]] = DEFAULT_PRETRAIN_HISTORY_PATH,
    device: Union[str, Any] = "auto",
    verbose: bool = True,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Execute complete self-supervised pretraining pipeline.

    Parameters
    ----------
    model:
        `PretrainReconstructionModel` (constructed with defaults if omitted).
    train_loader:
        `DataLoader` for training set.
    val_loader:
        Optional `DataLoader` for validation set.
    epochs:
        Number of pretraining epochs.
    lr:
        Learning rate for AdamW optimizer.
    weight_decay:
        Weight decay for AdamW.
    masking_ratio:
        Masking ratio for masked reconstruction MSE.
    grad_clip_norm:
        Max gradient norm for clipping.
    checkpoint_path:
        Destination file path for best/latest pretrained encoder checkpoint.
    history_path:
        Destination JSON file path for training history under artifacts.
    device:
        Device identifier (`"auto"`, `"cpu"`, `"cuda"`).
    verbose:
        Whether to log progress to stdout/logger.

    Returns
    -------
    `(model, history)`: Trained model and list of per-epoch metric dicts.
    """
    _require_torch()

    # Device resolution
    if isinstance(device, str):
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
    else:
        dev = device

    if model is None:
        from src.models.pretrain_reconstruction import PretrainReconstructionModel

        model = PretrainReconstructionModel(masking_ratio=masking_ratio)

    model = model.to(dev)

    if train_loader is None:
        raise ValueError("train_loader must be provided for pretraining")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_PRETRAIN_CHECKPOINT_PATH

    if verbose:
        logger.info("=" * 68)
        logger.info("Starting NEER Self-Supervised Pretraining (Phase 21B)")
        logger.info("Epochs: %d | Batch Size: %s | Device: %s | Masking Ratio: %.2f",
                    epochs, getattr(train_loader, "batch_size", "unknown"), dev, masking_ratio)
        logger.info("=" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = run_pretrain_epoch(
            model=model,
            loader=train_loader,
            device=dev,
            optimizer=optimizer,
            masking_ratio=masking_ratio,
            grad_clip_norm=grad_clip_norm,
        )
        duration = round(time.time() - t0, 3)

        val_metrics: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_metrics = run_pretrain_epoch(
                model=model,
                loader=val_loader,
                device=dev,
                optimizer=None,
                masking_ratio=masking_ratio,
            )

        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_mse": round(train_metrics["reconstruction_mse"], 6),
            "learning_rate": lr,
            "time_seconds": duration,
        }
        if val_metrics is not None:
            epoch_record["val_loss"] = round(val_metrics["loss"], 6)
            epoch_record["val_mse"] = round(val_metrics["reconstruction_mse"], 6)

        history.append(epoch_record)

        current_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss

        # Save latest/best pretrained encoder checkpoint
        save_pretrained_encoder_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"] if val_metrics else None,
            best_loss=best_loss,
        )

        if verbose:
            val_str = f" | val_loss: {epoch_record['val_loss']:.6f}" if val_metrics else ""
            logger.info("Epoch %3d/%d | train_loss: %.6f%s | time: %.2fs",
                        epoch, epochs, train_metrics["loss"], val_str, duration)

    # Save complete history to JSON
    if history_path:
        save_training_history(history, path=history_path)

    return model, history
