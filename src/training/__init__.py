"""NEER - Neural Embedding based Estimation and Reconstruction

Package: src/training
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS
"""

from __future__ import annotations

from src.training.losses import (
    DEFAULT_DEPTH_DIM,
    DEFAULT_EPS,
    DEFAULT_SPATIAL_DIMS,
    NEERLossConfig,
    anomaly_loss,
    gradient_loss,
    masked_mse_loss,
    vertical_smoothness_loss,
)

from src.training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointManager,
    CompatibilityReport,
    LoadedCheckpoint,
    check_compatibility,
    config_fingerprint,
    git_metadata,
    load_checkpoint,
    save_checkpoint,
)

from src.training.pretrain import (
    DEFAULT_PRETRAIN_CHECKPOINT_PATH,
    DEFAULT_PRETRAIN_HISTORY_PATH,
    load_pretrained_encoder,
    load_pretrained_into_neer_model,
    load_training_history,
    run_pretrain_epoch,
    save_pretrained_encoder_checkpoint,
    save_training_history,
    train_pretrain,
)

__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointManager",
    "CompatibilityReport",
    "LoadedCheckpoint",
    "check_compatibility",
    "config_fingerprint",
    "git_metadata",
    "load_checkpoint",
    "save_checkpoint",
    "DEFAULT_DEPTH_DIM",
    "DEFAULT_EPS",
    "DEFAULT_SPATIAL_DIMS",
    "NEERLossConfig",
    "anomaly_loss",
    "gradient_loss",
    "masked_mse_loss",
    "vertical_smoothness_loss",
    "DEFAULT_PRETRAIN_CHECKPOINT_PATH",
    "DEFAULT_PRETRAIN_HISTORY_PATH",
    "save_pretrained_encoder_checkpoint",
    "load_pretrained_encoder",
    "load_pretrained_into_neer_model",
    "save_training_history",
    "load_training_history",
    "run_pretrain_epoch",
    "train_pretrain",
]

# `NEERLoss` itself is only defined when torch is installed (same lazy
# pattern as `src.models`); re-export it too when available so
# `from src.training import NEERLoss` works without reaching into
# `src.training.losses` directly.
try:  # pragma: no cover - environment dependent
    from src.training.losses import NEERLoss

    __all__.append("NEERLoss")
except ImportError:  # pragma: no cover - environment dependent
    pass

## `scripts/train.py`
#!/usr/bin/env python3
"""
NEER — Phase 20 training engine.

Trains the complete `NEERModel` (Phase 18) against `NEERLoss` (Phase 19)
on tensors produced by `scripts/preprocess_data.py`.

Bridging the model's output shape and the dataset's target shape
--------------------------------------------------------------------
`NEERModel.forward` predicts one pooled temperature-anomaly *profile*
per sample, `(batch, num_depths)` (Phase 18) — it summarizes a whole
input grid into a single spatial embedding before decoding depths, it
does not predict per-pixel anomalies. `NEERDataset`, however, yields
the full-grid subsurface target and mask, `(depth, lat, lon)` each
(Phase 12), because that is what the tensor pipeline assembles and
what a future per-pixel model would want unchanged.

This script is what reconciles the two: for every batch, the full-grid
`targets`/`target_mask` are pooled into a `(batch, depth)` profile with
a mask-weighted spatial mean (`pool_profile` below) — exactly the
fallback baseline `anomaly_loss` already uses elsewhere in this
codebase — before either is handed to the model's prediction or to
`NEERLoss`. A depth level with no valid observations anywhere in a
sample's grid pools to a `False` profile-mask entry (never a
fabricated `0.0` target), so `masked_mse_loss` correctly excludes it
rather than training the model to output zero for gap-filled depths.

Usage
-----
    python scripts/train.py
    python scripts/train.py --environment demo --epochs 50 --batch-size 4
    python scripts/train.py --device cuda --lr 3e-4 --patience 10

Outputs
-------
    artifacts/checkpoints/neer_last.pt   the most recently completed epoch
    artifacts/checkpoints/neer_best.pt   the epoch with the lowest val loss
    reports/training_history.json        per-epoch metrics for every epoch run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import NEERDataset, make_dataloader  # noqa: E402
from src.data.loaders.errors import MissingDependencyError  # noqa: E402
from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.models.neer_model import NEERModel  # noqa: E402
from src.training.checkpoint import CheckpointManager, CheckpointCompatibilityError  # noqa: E402
from src.training.losses import NEERLoss, NEERLossConfig  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

try:
    import torch
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


logger = get_logger("neer.train")

DEFAULT_TENSORS_PATH = PROJECT_ROOT / "data" / "processed" / "neer_tensors.npz"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
POOL_EPS = 1e-8


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="the NEER training engine (scripts/train.py)",
            install_hint="pip install torch",
        )


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed every source of randomness the training loop touches.

    Covers Python's `random`, numpy (the `DataLoader` worker shuffling and
    any numpy-side augmentation), and torch's CPU *and* CUDA generators, so
    a run is reproducible whichever device it ends up training on. Also
    turns off cuDNN's nondeterministic autotuned kernels, trading a little
    speed for a training run that behaves the same way twice.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------


def resolve_device(requested: str) -> "torch.device":
    """`requested` is 'auto', 'cpu', or 'cuda'. 'auto' picks CUDA when it
    is actually available, so the same command line works unchanged on a
    CPU-only machine and a GPU box."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------
# Bridging full-grid targets to the model's pooled-profile output
# --------------------------------------------------------------------------


def pool_profile(
    values: "torch.Tensor", mask: "torch.Tensor", eps: float = POOL_EPS
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Collapse `(batch, depth, lat, lon)` to a `(batch, depth)` profile.

    Parameters
    ----------
    values:
        `(batch, depth, lat, lon)` target field.
    mask:
        Same shape, True/nonzero where a cell is valid (ocean, observed).

    Returns
    -------
    profile:
        `(batch, depth)` mask-weighted spatial mean of `values`. Depth
        slices with zero valid cells get `0.0` (never used — see
        `profile_mask`).
    profile_mask:
        `(batch, depth)` float tensor, `1.0` where at least one spatial
        cell was valid for that (sample, depth), else `0.0`. This — not
        `values` itself — is what determines whether a depth contributes
        to the loss, so a `0.0` placeholder in `profile` for an
        all-invalid depth can never be scored as if it were a real
        target.
    """
    mask_f = mask.to(dtype=values.dtype)
    valid_count = mask_f.sum(dim=(-2, -1))
    profile = (values * mask_f).sum(dim=(-2, -1)) / valid_count.clamp_min(eps)
    profile_mask = (valid_count > 0).to(dtype=values.dtype)
    return profile, profile_mask


# --------------------------------------------------------------------------
# One epoch
# --------------------------------------------------------------------------


def run_epoch(
    model: "NEERModel",
    loader: "DataLoader",
    criterion: "NEERLoss",
    device: "torch.device",
    *,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Run one pass over `loader`. Trains (backward + step) when
    `optimizer` is given, otherwise runs in `no_grad` eval mode.

    Returns the sample-weighted average of every `NEERLoss` component
    over the whole epoch, plus (when training) the mean gradient norm
    actually applied, so the caller can log both loss and optimization
    health per epoch.
    """
    training = optimizer is not None
    model.train(mode=training)

    totals: Dict[str, float] = {"mse": 0.0, "smoothness": 0.0, "gradient": 0.0, "anomaly": 0.0, "total": 0.0}
    grad_norms: List[float] = []
    n_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets_full = batch["targets"].to(device, non_blocking=True)
            target_mask_full = batch["mask"]["target"].to(device, non_blocking=True)
            batch_size = inputs.shape[0]

            target_profile, profile_mask = pool_profile(targets_full, target_mask_full)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred_profile = model(inputs)
            losses = criterion(pred_profile, target_profile, profile_mask)

            if training:
                losses["total"].backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_norm
                    )
                    grad_norms.append(float(grad_norm))
                optimizer.step()

            for key in totals:
                totals[key] += float(losses[key].detach()) * batch_size
            n_samples += batch_size

    n_samples = max(n_samples, 1)
    metrics = {key: value / n_samples for key, value in totals.items()}
    if grad_norms:
        metrics["grad_norm"] = sum(grad_norms) / len(grad_norms)
    return metrics


# --------------------------------------------------------------------------
# Checkpointing
#
# Phase 27's `CheckpointManager` (src/training/checkpoint.py) owns the
# `neer_best.pt` / `neer_last.pt` files: it bundles model + optimizer state
# with epoch, seed, the resolved config, training stats and git/project
# metadata, and validates a checkpoint's compatibility with the current
# model/config before loading it. This script only decides *when* to call
# `save_best`/`save_last`, and what to put in `training_stats`.
# --------------------------------------------------------------------------


def _training_stats(
    *, val_loss: float, best_val_loss: float, best_epoch: int, train_metrics: Dict[str, float],
    val_metrics: Dict[str, float], history: List[Dict[str, Any]], stopped_early: bool,
) -> Dict[str, Any]:
    return {
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "stopped_early": stopped_early,
        "history": history,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the NEER model (Phase 20).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--environment",
        default="demo",
        choices=["base", "demo", "development", "production"],
        help="Config environment (used for the default seed; default: demo).",
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=DEFAULT_TENSORS_PATH,
        help=f"Path to the assembled tensor bundle (default: {DEFAULT_TENSORS_PATH}).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="0 disables clipping.")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many epochs with no val-loss improvement. 0 disables early stopping.",
    )
    parser.add_argument("--min-delta", type=float, default=1e-5, help="Minimum val-loss improvement that resets patience.")
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--anomaly-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a neer_last.pt-style checkpoint to resume optimizer/model state from.",
    )
    return parser


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    _require_torch()
    args = build_parser().parse_args(argv)

    config = load_config(args.environment)
    seed = args.seed if args.seed is not None else config.demo.seed
    set_seed(seed)

    device = resolve_device(args.device)

    logger.info("=" * 72)
    logger.info("NEER training — Phase 20")
    logger.info("=" * 72)
    logger.info("environment=%s  seed=%d  device=%s", args.environment, seed, device)

    if not args.tensors.exists():
        logger.error(
            "Tensor bundle not found at %s. Generate it first with "
            "'python scripts/create_demo_data.py' then "
            "'python scripts/preprocess_data.py --environment %s'.",
            args.tensors,
            args.environment,
        )
        return 2

    bundle = TensorBundle.load(args.tensors)
    logger.info(
        "Loaded tensor bundle: %d timesteps, grid=%s, channels=%s",
        bundle.n_time,
        bundle.grid_shape,
        bundle.channel_names,
    )

    train_ds = NEERDataset(bundle, split="train")
    val_ds = NEERDataset(bundle, split="val")
    logger.info("train samples=%d  val samples=%d", len(train_ds), len(val_ds))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Built from the config so `model.use_gnn` (or NEER_MODEL_USE_GNN=true)
    # actually reaches the model. With the shipped configs this is
    # identical to a plain `NEERModel()`.
    model = NEERModel.from_config(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%d parameters)", repr(model), n_params)

    loss_config = NEERLossConfig(
        smoothness_weight=args.smoothness_weight,
        gradient_weight=args.gradient_weight,
        anomaly_weight=args.anomaly_weight,
        spatial_dims=None,  # (batch, depth) pooled profile — no spatial axis
    )
    criterion = NEERLoss(loss_config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    manager = CheckpointManager(args.checkpoint_dir, name="neer")

    if args.resume is not None:
        if not args.resume.exists():
            logger.error("--resume checkpoint not found: %s", args.resume)
            return 2
        logger.info("Resuming from %s", args.resume)
        try:
            # strict=False: an interrupted/older run may have been saved with
            # slightly different auxiliary state; a real architecture change
            # still surfaces as a shape mismatch, which is always an error
            # (see CheckpointManager / check_compatibility).
            loaded = manager.load_checkpoint(
                args.resume, model=model, optimizer=optimizer, config=config, strict=False
            )
        except CheckpointCompatibilityError as exc:
            logger.error("--resume checkpoint is not compatible with this model/config: %s", exc)
            return 2
        stats = loaded.training_stats
        history = stats.get("history", [])
        best_val_loss = stats.get("best_val_loss", float("inf"))
        best_epoch = stats.get("best_epoch", 0)
        start_epoch = loaded.epoch + 1
        logger.info(
            "Resumed at epoch %d (best_val_loss so far=%.6f)", start_epoch, best_val_loss
        )

    reports_dir = args.reports_dir
    history_path = reports_dir / "training_history.json"

    epochs_without_improvement = 0
    training_start = time.time()
    stopped_early = False
    final_epoch = start_epoch - 1

    logger.info(
        "Training config: epochs=%d batch_size=%d lr=%g weight_decay=%g "
        "grad_clip_norm=%g patience=%s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.grad_clip_norm,
        args.patience if args.patience > 0 else "disabled",
    )
    logger.info("-" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        epoch_time = time.time() - epoch_start
        val_loss = val_metrics["total"]
        improved = val_loss < best_val_loss - args.min_delta

        logger.info(
            "epoch %3d/%d | train_loss=%.6f (mse=%.6f) | val_loss=%.6f (mse=%.6f) "
            "| grad_norm=%.4f | %.1fs%s",
            epoch,
            args.epochs,
            train_metrics["total"],
            train_metrics["mse"],
            val_loss,
            val_metrics["mse"],
            train_metrics.get("grad_norm", 0.0),
            epoch_time,
            "  *** best ***" if improved else "",
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": epoch_time,
        }
        history.append(record)
        final_epoch = epoch

        # Always refresh the "last" checkpoint so a crash never loses more
        # than the epoch in progress.
        stats = _training_stats(
            val_loss=val_loss, best_val_loss=min(best_val_loss, val_loss), best_epoch=best_epoch,
            train_metrics=train_metrics, val_metrics=val_metrics, history=history, stopped_early=False,
        )
        manager.save_last(
            model=model, optimizer=optimizer, epoch=epoch, seed=seed, config=config,
            training_stats=stats, extra={"args": vars(args)},
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            stats = _training_stats(
                val_loss=val_loss, best_val_loss=best_val_loss, best_epoch=best_epoch,
                train_metrics=train_metrics, val_metrics=val_metrics, history=history, stopped_early=False,
            )
            manager.save_best(
                model=model, optimizer=optimizer, epoch=epoch, seed=seed, config=config,
                training_stats=stats, extra={"args": vars(args)},
            )
            logger.info("  -> saved new best checkpoint to %s", manager.best_path)
        else:
            epochs_without_improvement += 1

        # Training history is rewritten every epoch (not just at the end)
        # so an interrupted run still leaves a complete, readable record
        # of everything it completed.
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "environment": args.environment,
                    "seed": seed,
                    "device": str(device),
                    "model_parameters": n_params,
                    "loss_config": asdict(loss_config),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "stopped_early": stopped_early,
                    "epochs_run": final_epoch,
                    "history": history,
                },
                f,
                indent=2,
            )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logger.info(
                "Early stopping: no val_loss improvement for %d epochs (patience=%d).",
                epochs_without_improvement,
                args.patience,
            )
            stopped_early = True
            break

    total_time = time.time() - training_start

    # Final rewrite so `stopped_early`/`epochs_run` reflect the true outcome
    # even when the loop above ended via early stopping rather than the
    # last in-loop write (which is otherwise identical).
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "environment": args.environment,
                "seed": seed,
                "device": str(device),
                "model_parameters": n_params,
                "loss_config": asdict(loss_config),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "stopped_early": stopped_early,
                "epochs_run": final_epoch,
                "total_training_time_seconds": total_time,
                "history": history,
            },
            f,
            indent=2,
        )

    logger.info("-" * 72)
    logger.info(
        "Training finished: %d epoch(s) in %.1fs. Best val_loss=%.6f at epoch %d.",
        final_epoch - start_epoch + 1,
        total_time,
        best_val_loss,
        best_epoch,
    )
    logger.info("  last checkpoint : %s", manager.last_path)
    logger.info("  best checkpoint : %s", manager.best_path)
    logger.info("  training history: %s", history_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Tests for Phase 27 — the checkpoint manager (`src/training/checkpoint.py`).

Two groups, same split as the rest of this project's torch-dependent
modules (see `tests/test_neer_model.py`):

1. **No torch required.** JSON-safety, config-to-dict/fingerprint, git
   metadata (both outside and inside a real git repo), and the
   `check_compatibility` diff logic itself — it only needs objects with a
   `.state_dict()` method, not real tensors, so it's exercised here with
   a tiny fake model.
2. **Requires torch** (`pytest.importorskip`). Real save/restore
   round-trips with `torch.nn.Module`/`torch.optim`: `save_best`,
   `save_last`, `load_checkpoint` restore weights and optimizer state
   bit-for-bit; a shape-changed model is refused *before* any weights are
   touched; a legacy (pre-Phase-27) checkpoint still loads; and the save
   is atomic (a crash mid-write never corrupts the previous checkpoint).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.checkpoint import (  # noqa: E402
    CheckpointCompatibilityError,
    CheckpointManager,
    LEGACY_SCHEMA,
    SCHEMA_VERSION,
    _config_to_dict,
    _dig,
    _json_safe,
    _normalize_legacy,
    check_compatibility,
    config_fingerprint,
    git_metadata,
    load_checkpoint,
    save_checkpoint,
)

# --------------------------------------------------------------------------
# 1. no torch required
# --------------------------------------------------------------------------


def test_json_safe_handles_numpy_paths_sets_and_unknown_objects():
    import numpy as np

    class Unrepresentable:
        def __str__(self):
            return "<weird>"

    out = _json_safe(
        {"f": np.float32(1.5), "i": np.int64(3), "arr": np.array([1, 2]),
         "p": Path("a/b"), "s": {1, 2}, "weird": Unrepresentable(), "n": None, "ok": "text"}
    )
    assert out == {"f": 1.5, "i": 3, "arr": [1, 2], "p": "a/b", "s": [1, 2], "weird": "<weird>",
                   "n": None, "ok": "text"}
    json.dumps(out)  # must be actually serializable


def test_config_to_dict_accepts_neerconfig_like_dict_and_dataclass():
    from dataclasses import dataclass

    class FakeNeerConfig:
        environment = "demo"
        raw = {"model": {"embedding_dim": 256}}

    assert _config_to_dict(FakeNeerConfig())["model"]["embedding_dim"] == 256
    assert _config_to_dict(FakeNeerConfig())["environment"] == "demo"
    assert _config_to_dict({"a": 1}) == {"a": 1}
    assert _config_to_dict(None) == {}

    @dataclass
    class Plain:
        x: int = 3

    assert _config_to_dict(Plain()) == {"x": 3}
    with pytest.raises(TypeError):
        _config_to_dict(object())


def test_dig_reads_dotted_paths_and_is_none_safe():
    d = {"model": {"embedding_dim": 256}}
    assert _dig(d, "model.embedding_dim") == 256
    assert _dig(d, "model.missing") is None
    assert _dig(d, "nope.x") is None
    assert _dig({}, "a.b.c") is None


def test_config_fingerprint_is_stable_and_sensitive_to_architecture_fields():
    base = {"environment": "demo", "model": {"embedding_dim": 256, "use_gnn": False},
            "uncertainty": {"enabled": False}, "depths": [0, 5, 10]}
    assert config_fingerprint(base) == config_fingerprint(dict(base))  # deterministic
    for mutation in (
        {"model": {"embedding_dim": 128, "use_gnn": False}},
        {"model": {"embedding_dim": 256, "use_gnn": True}},
        {"uncertainty": {"enabled": True}},
        {"depths": [0, 5, 10, 20]},
        {"environment": "production"},
    ):
        other = {**base, **mutation}
        assert config_fingerprint(other) != config_fingerprint(base), mutation
    # a field this project doesn't consider architecture-relevant shouldn't move it
    unrelated = {**base, "project": {"name": "different"}}
    assert config_fingerprint(unrelated) == config_fingerprint(base)


def test_git_metadata_outside_a_repo_degrades_gracefully(tmp_path):
    info = git_metadata(tmp_path)  # tmp_path is never a git repo
    assert info["available"] is False and "reason" in info


def test_git_metadata_inside_a_repo_reports_commit_and_dirty_flag(tmp_path):
    pytest.importorskip("subprocess")
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git not installed")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    clean = git_metadata(tmp_path)
    assert clean["available"] is True and clean["dirty"] is False and len(clean["commit"]) == 40

    (tmp_path / "f.txt").write_text("y")
    dirty = git_metadata(tmp_path)
    assert dirty["available"] is True and dirty["dirty"] is True and dirty["commit"] == clean["commit"]


class _FakeParam:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n


class _FakeModel:
    """Enough of `nn.Module`'s surface for `check_compatibility`/`build_metadata`
    to work on, with no torch dependency."""

    def __init__(self, shapes):
        self._shapes = dict(shapes)

    def state_dict(self):
        return {k: _FakeParam(v) for k, v in self._shapes.items()}

    def parameters(self):
        return list(self.state_dict().values())


def _metadata_for(model, config=None):
    from src.training.checkpoint import build_metadata

    return build_metadata(kind="manual", epoch=1, model=model, config=config)


def test_check_compatibility_passes_on_matching_shapes():
    m = _FakeModel({"w": (4, 4), "b": (4,)})
    report = check_compatibility(_metadata_for(m), model=m)
    assert report.ok and not report.errors and not report.warnings


def test_check_compatibility_flags_shape_mismatch_as_an_error_regardless_of_strict():
    meta = _metadata_for(_FakeModel({"w": (4, 4), "b": (4,)}))
    changed = _FakeModel({"w": (8, 4), "b": (4,)})
    for strict in (True, False):
        report = check_compatibility(meta, model=changed, strict=strict)
        assert not report.ok
        assert report.shape_mismatches == ["w: model expects [8, 4], checkpoint has [4, 4]"]
        with pytest.raises(CheckpointCompatibilityError):
            report.raise_if_incompatible()


def test_check_compatibility_missing_and_unexpected_keys_depend_on_strict():
    meta = _metadata_for(_FakeModel({"w": (4, 4), "b": (4,)}))
    extra = _FakeModel({"w": (4, 4), "b": (4,), "extra": (2,)})
    strict_report = check_compatibility(meta, model=extra, strict=True)
    assert not strict_report.ok and strict_report.missing_keys == ["extra"]
    lenient_report = check_compatibility(meta, model=extra, strict=False)
    assert lenient_report.ok and lenient_report.warnings and lenient_report.missing_keys == ["extra"]


def test_check_compatibility_config_mismatch_is_only_a_warning():
    cfg_a = {"environment": "demo", "model": {"embedding_dim": 256, "use_gnn": False},
             "uncertainty": {"enabled": False}, "depths": [0, 5, 10]}
    cfg_b = {**cfg_a, "model": {"embedding_dim": 128, "use_gnn": False}}
    meta = _metadata_for(None, config=cfg_a)
    report = check_compatibility(meta, config=cfg_b)
    assert report.ok and any("config fingerprint" in w for w in report.warnings)


def test_check_compatibility_rejects_unknown_schema_version():
    report = check_compatibility({"schema_version": 999})
    assert not report.ok and "schema_version" in report.errors[0]


def test_check_compatibility_accepts_current_and_legacy_schema():
    assert check_compatibility({"schema_version": SCHEMA_VERSION}).ok
    legacy_report = check_compatibility({"schema_version": LEGACY_SCHEMA})
    assert legacy_report.ok and legacy_report.warnings


def test_normalize_legacy_extracts_known_fields_without_metadata():
    raw = {"epoch": 5, "model_state_dict": {}, "optimizer_state_dict": {}, "val_loss": 0.2,
           "best_val_loss": 0.1, "history": [{"epoch": 5}], "args": {"lr": 0.01}}
    meta = _normalize_legacy(raw)
    assert meta["schema_version"] == LEGACY_SCHEMA
    assert meta["epoch"] == 5
    assert meta["training_stats"] == {"val_loss": 0.2, "best_val_loss": 0.1, "history": [{"epoch": 5}]}
    assert meta["git"]["available"] is False


def test_manager_without_torch_still_creates_its_directory(tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts", name="probe")
    assert (tmp_path / "ckpts").is_dir()
    assert mgr.best_path.name == "probe_best.pt" and mgr.last_path.name == "probe_last.pt"
    assert mgr.list_checkpoints() == []


# --------------------------------------------------------------------------
# 2. requires torch — real save/restore round trips
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


class _TinyNet(nn.Module):
    def __init__(self, hidden: int = 4):
        super().__init__()
        self.linear = nn.Linear(4, hidden)

    def forward(self, x):
        return self.linear(x)


def _make_model_and_optimizer(hidden=4, lr=0.01):
    model = _TinyNet(hidden)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


def _train_one_step(model, optimizer):
    x = torch.randn(2, 4)
    loss = model(x).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


DEMO_CONFIG = {
    "environment": "demo",
    "model": {"embedding_dim": 256, "use_gnn": False},
    "uncertainty": {"enabled": False},
    "depths": [0.0, 5.0, 10.0],
}


def test_save_best_and_save_last_restore_weights_and_optimizer_state(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    _train_one_step(model, optimizer)  # give the optimizer real (non-empty) state
    original_weight = model.linear.weight.detach().clone()
    original_opt_state = optimizer.state_dict()

    mgr = CheckpointManager(tmp_path, name="neer")
    mgr.save_last(model=model, optimizer=optimizer, epoch=3, seed=42, config=DEMO_CONFIG,
                  training_stats={"val_loss": 0.5, "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}]})
    mgr.save_best(model=model, optimizer=optimizer, epoch=3, seed=42, config=DEMO_CONFIG,
                  training_stats={"val_loss": 0.5})

    assert mgr.best_path.exists() and mgr.last_path.exists()
    assert mgr.list_checkpoints() == sorted([mgr.best_path, mgr.last_path])
    sidecar = json.loads(mgr.last_path.with_suffix(".pt.json").read_text())
    assert sidecar["epoch"] == 3 and sidecar["seed"] == 42 and sidecar["kind"] == "last"
    assert sidecar["model"]["class_name"] == "_TinyNet"
    assert set(sidecar["state_dict_shapes"]) == set(model.state_dict())

    fresh_model, fresh_optimizer = _make_model_and_optimizer()
    loaded = mgr.load_checkpoint(model=fresh_model, optimizer=fresh_optimizer, config=DEMO_CONFIG)

    assert loaded.epoch == 3
    assert loaded.training_stats["val_loss"] == 0.5
    assert loaded.report.ok and not loaded.report.warnings
    assert torch.equal(fresh_model.linear.weight, original_weight)
    assert fresh_optimizer.state_dict()["param_groups"] == original_opt_state["param_groups"]


def test_load_checkpoint_defaults_to_last_when_no_path_given(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    mgr = CheckpointManager(tmp_path, name="neer")
    mgr.save_last(model=model, optimizer=optimizer, epoch=1, config=DEMO_CONFIG)
    loaded = mgr.load_checkpoint()  # no path -> last_path
    assert loaded.path == mgr.last_path


def test_incompatible_model_is_refused_before_any_weights_are_touched(tmp_path):
    model, optimizer = _make_model_and_optimizer(hidden=4)
    mgr = CheckpointManager(tmp_path, name="neer")
    mgr.save_best(model=model, optimizer=optimizer, epoch=1, config=DEMO_CONFIG)

    mismatched_model = _TinyNet(hidden=8)  # different output width -> shape mismatch
    sentinel = mismatched_model.linear.weight.detach().clone()
    with pytest.raises(CheckpointCompatibilityError) as exc_info:
        mgr.load_checkpoint(mgr.best_path, model=mismatched_model, config=DEMO_CONFIG)
    assert "shape mismatch" in str(exc_info.value)
    assert torch.equal(mismatched_model.linear.weight, sentinel)  # untouched


def test_strict_false_tolerates_extra_module_but_strict_true_does_not(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    mgr = CheckpointManager(tmp_path, name="neer")
    mgr.save_best(model=model, optimizer=optimizer, epoch=1, config=DEMO_CONFIG)

    class Wider(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.extra = nn.Linear(2, 2)

    lenient = mgr.load_checkpoint(mgr.best_path, model=Wider(), strict=False)
    assert lenient.report.ok and any("extra" in w for w in lenient.report.warnings)
    with pytest.raises(CheckpointCompatibilityError):
        mgr.load_checkpoint(mgr.best_path, model=Wider(), strict=True)


def test_validate_false_still_returns_a_report_but_does_not_raise(tmp_path):
    model, optimizer = _make_model_and_optimizer(hidden=4)
    mgr = CheckpointManager(tmp_path, name="neer")
    mgr.save_best(model=model, optimizer=optimizer, epoch=1, config=DEMO_CONFIG)
    bad = _TinyNet(hidden=8)
    loaded = mgr.load_checkpoint(mgr.best_path, model=bad, validate=False)
    assert not loaded.report.ok and loaded.report.shape_mismatches


def test_legacy_checkpoint_without_metadata_still_loads(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    legacy_path = tmp_path / "neer_last.pt"
    torch.save(
        {"epoch": 9, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
         "val_loss": 0.3, "best_val_loss": 0.2, "history": [{"epoch": 9}], "args": {"lr": 0.01}},
        legacy_path,
    )
    fresh, _ = _make_model_and_optimizer()
    loaded = load_checkpoint(legacy_path, model=fresh)
    assert loaded.epoch == 9
    assert loaded.metadata["schema_version"] == LEGACY_SCHEMA
    assert loaded.training_stats == {"val_loss": 0.3, "best_val_loss": 0.2, "history": [{"epoch": 9}]}
    assert torch.equal(fresh.linear.weight, model.linear.weight)


def test_save_checkpoint_is_atomic_a_failed_save_does_not_clobber_the_existing_file(tmp_path, monkeypatch):
    model, optimizer = _make_model_and_optimizer()
    path = tmp_path / "neer_best.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, config=DEMO_CONFIG)
    original_bytes = path.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("disk full (simulated)")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError):
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=2, config=DEMO_CONFIG)

    assert path.read_bytes() == original_bytes  # untouched by the failed write
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())  # no leftover temp file


def test_save_checkpoint_works_without_an_optimizer_and_without_a_config(tmp_path):
    model, _ = _make_model_and_optimizer()
    path = save_checkpoint(tmp_path / "solo.pt", model=model, epoch=1)
    loaded = load_checkpoint(path, model=_TinyNet())
    assert loaded.metadata["config"] == {} and loaded.metadata["config_fingerprint"] is None
    assert loaded.raw["optimizer_state_dict"] is None


def test_loading_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_loading_a_non_checkpoint_file_raises_a_clear_error(tmp_path):
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"just": "a dict"}, path)
    from src.training.checkpoint import CheckpointError

    with pytest.raises(CheckpointError, match="does not look like a NEER checkpoint"):
        load_checkpoint(path)


def test_metadata_captures_git_and_environment_info(tmp_path):
    model, _ = _make_model_and_optimizer()
    path = save_checkpoint(tmp_path / "x.pt", model=model, epoch=1, config=DEMO_CONFIG,
                            seed=7, extra={"note": "unit test"})
    loaded = load_checkpoint(path)
    meta = loaded.metadata
    assert meta["seed"] == 7 and meta["extra"] == {"note": "unit test"}
    assert "available" in meta["git"]
    assert meta["environment_info"]["torch_version"] == torch.__version__
    assert meta["config_fingerprint"] == config_fingerprint(DEMO_CONFIG)
"""

## `README.md` — section added before "## ARGO validation (Phase 26)"

"""
"""markdown
## Checkpoint management (Phase 27)

`src/training/checkpoint.py` is the one place every training script saves and
loads checkpoints. `scripts/train.py` uses it for `neer_best.pt` / `neer_last.pt`
and `--resume`.
"""
python
from src.training.checkpoint import CheckpointManager

mgr = CheckpointManager("artifacts/checkpoints", name="neer")
mgr.save_last(model=model, optimizer=optimizer, epoch=epoch, seed=seed, config=config,
              training_stats={"val_loss": val_loss, "history": history})
if improved:
    mgr.save_best(model=model, optimizer=optimizer, epoch=epoch, seed=seed,
                  config=config, training_stats={"val_loss": val_loss})

loaded = mgr.load_checkpoint(mgr.best_path, model=model, optimizer=optimizer, config=config)
print(loaded.epoch, loaded.training_stats, loaded.report.warnings)
""""

Each checkpoint bundles model + optimizer state with the epoch, seed, the
resolved config (plus a fast `config_fingerprint` of its architecture-relevant
fields), training statistics, parameter shapes, and git/project metadata
(commit, branch, dirty flag — degrades to `{"available": false}` outside a repo
or without git installed, never an error). A human-readable `.json` sidecar is
written next to every `.pt` file. Saves are atomic (write-to-temp then rename),
so a crash mid-write never corrupts the previous checkpoint.

**Compatibility is checked before any weights are touched.** Loading diffs the
checkpoint's stored parameter shapes against the live model: a shape mismatch is
always a hard error (`CheckpointCompatibilityError`, raised before
`load_state_dict` runs); missing/unexpected parameter names are errors under
`strict=True` (the default) and warnings under `strict=False`. A config
fingerprint mismatch is only a warning — the shape check is authoritative.
Checkpoints saved before this module existed still load, as a `"legacy"`
schema with best-effort compatibility checking.

"""