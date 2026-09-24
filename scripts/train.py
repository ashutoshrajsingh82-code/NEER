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
# --------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    *,
    model: "NEERModel",
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    history: List[Dict[str, Any]],
    val_loss: float,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stringify every value (Paths included) so the checkpoint contains only
    # plain built-in types — keeps it loadable with torch's safer
    # `weights_only=True` default in the future and avoids pickling
    # environment-specific objects (e.g. argparse.Namespace, pathlib.Path).
    serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
            "args": serializable_args,
        },
        path,
    )


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

    model = NEERModel().to(device)
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

    if args.resume is not None:
        if not args.resume.exists():
            logger.error("--resume checkpoint not found: %s", args.resume)
            return 2
        logger.info("Resuming from %s", args.resume)
        # weights_only=False: this checkpoint is our own (saved by
        # save_checkpoint above), not third-party input, and its "args" dict
        # plus history entries are plain Python objects torch's safer
        # weights_only loader isn't needed to protect against here.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint.get("history", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = checkpoint.get("epoch", 0) + 1
        if history:
            best_epoch = max(
                (h["epoch"] for h in history if h["val_loss"] <= best_val_loss + 1e-12),
                default=0,
            )
        logger.info(
            "Resumed at epoch %d (best_val_loss so far=%.6f)", start_epoch, best_val_loss
        )

    checkpoint_dir = args.checkpoint_dir
    reports_dir = args.reports_dir
    best_path = checkpoint_dir / "neer_best.pt"
    last_path = checkpoint_dir / "neer_last.pt"
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
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            val_loss=val_loss,
            best_val_loss=min(best_val_loss, val_loss),
            args=args,
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                args=args,
            )
            logger.info("  -> saved new best checkpoint to %s", best_path)
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
    logger.info("  last checkpoint : %s", last_path)
    logger.info("  best checkpoint : %s", best_path)
    logger.info("  training history: %s", history_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())