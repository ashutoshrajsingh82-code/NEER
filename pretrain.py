#!/usr/bin/env python3
"""
NEER — Phase 21B Self-Supervised Pretraining CLI.

Executes masked reconstruction pretraining for the NEER model backbone:
    surface fields -> CNN -> ViT -> embedding -> decoder -> reconstructed surface fields

Outputs:
    artifacts/checkpoints/encoder_pretrained.pt
    artifacts/pretrain_history.json

Usage:
    python pretrain.py --config configs/demo.yaml
    python pretrain.py --config configs/demo.yaml --epochs 10 --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import NEERDataset, make_dataloader  # noqa: E402
from src.data.loaders.errors import MissingDependencyError  # noqa: E402
from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.models.pretrain_reconstruction import PretrainReconstructionModel  # noqa: E402
from src.training.pretrain import (  # noqa: E402
    DEFAULT_PRETRAIN_CHECKPOINT_PATH,
    DEFAULT_PRETRAIN_HISTORY_PATH,
    train_pretrain,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="the NEER pretraining script (pretrain.py)",
            install_hint="pip install torch",
        )


logger = get_logger("neer.pretrain_cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-supervised pretraining for NEER surface fields reconstruction (Phase 21B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/demo.yaml",
        help="Path to YAML configuration overlay (e.g. configs/demo.yaml) or environment name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of pretraining epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--masking-ratio",
        type=float,
        default=0.5,
        help="Fraction of valid ocean cells masked as reconstruction targets (default: 0.5).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Latent embedding dimension (defaults to config model.embedding_dim or 256).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to train on (default: auto).",
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("data/processed/neer_tensors.npz"),
        help="Path to assembled tensor bundle (default: data/processed/neer_tensors.npz).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_PRETRAIN_CHECKPOINT_PATH,
        help="Destination path for pretrained encoder checkpoint (default: artifacts/checkpoints/encoder_pretrained.pt).",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_PRETRAIN_HISTORY_PATH,
        help="Destination path for training history JSON (default: artifacts/pretrain_history.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (defaults to config demo.seed).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    return parser


def parse_environment(config_arg: str) -> str:
    """Resolve environment name from config path or name."""
    p = Path(config_arg)
    stem = p.stem.lower()
    if stem in ("demo", "base", "development", "production"):
        return stem
    return "demo"


def main(argv: Optional[List[str]] = None) -> int:
    _require_torch()
    parser = build_parser()
    args = parser.parse_args(argv)

    env_name = parse_environment(args.config)
    config = load_config(env_name)

    seed = args.seed if args.seed is not None else config.demo.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    embedding_dim = args.embedding_dim or config.model.embedding_dim

    print("=" * 72)
    print("NEER — Phase 21B Self-Supervised Pretraining")
    print(f"Config: {args.config} (env: {env_name}) | Seed: {seed}")
    print(f"Embedding Dim: {embedding_dim} | Masking Ratio: {args.masking_ratio:.2f}")
    print(f"Checkpoint Target: {args.checkpoint_path}")
    print(f"History Target:    {args.history_path}")
    print("=" * 72)

    # Resolve tensors path
    tensors_path = args.tensors
    if not tensors_path.is_absolute():
        tensors_path = PROJECT_ROOT / tensors_path

    if not tensors_path.exists():
        logger.error(
            "Tensor bundle not found at %s. Please run preprocessing first.", tensors_path
        )
        return 2

    bundle = TensorBundle.load(tensors_path)
    train_ds = NEERDataset(bundle, split="train")
    val_ds = NEERDataset(bundle, split="val")

    gen = torch.Generator().manual_seed(seed)
    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=gen,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = PretrainReconstructionModel(
        embedding_dim=embedding_dim,
        masking_ratio=args.masking_ratio,
    )

    # Resolve absolute destination paths
    ckpt_path = args.checkpoint_path
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    hist_path = args.history_path
    if not hist_path.is_absolute():
        hist_path = PROJECT_ROOT / hist_path

    model, history = train_pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        masking_ratio=args.masking_ratio,
        checkpoint_path=ckpt_path,
        history_path=hist_path,
        device=args.device,
        verbose=True,
    )

    print("\n" + "=" * 72)
    print("Pretraining complete!")
    print(f"Final train loss: {history[-1]['train_loss']:.6f}")
    if "val_loss" in history[-1]:
        print(f"Final val loss:   {history[-1]['val_loss']:.6f}")
    print(f"Saved pretrained encoder checkpoint to: {ckpt_path}")
    print(f"Saved pretraining history to:            {hist_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
