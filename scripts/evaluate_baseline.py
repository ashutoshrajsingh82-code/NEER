#!/usr/bin/env python3
"""
NEER — Phase 24 baseline evaluation.

Fits and scores the baseline models (`src/models/baselines`) — monthly
climatology, Ridge regression, and optionally LightGBM — against
*exactly* the same tensor bundle, and the same chronological
train/val/test split within it, that `scripts/train.py` trains and
evaluates `NEERModel` against. There is deliberately only one way in
this script to get train/val/test data (`--tensors` -> one
`TensorBundle.load` -> one `pool_tensor_bundle` call, shared by every
model below), so it is structurally impossible for two baselines in
one run's report to have been compared on different data.

Usage
-----
    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --tensors data/processed/neer_tensors.npz
    python scripts/evaluate_baselines.py --models climatology ridge
    python scripts/evaluate_baselines.py --ridge-alpha 5.0

Output
------
    reports/baseline_evaluation.json   RMSE/MAE/bias/R^2, overall and per
                                        depth level, for every split and
                                        every model that ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.errors import MissingDependencyError  # noqa: E402
from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.evaluation.interface import (  # noqa: E402
    BaselineModel,
    SPLIT_NAMES,
    evaluate_baseline,
    pool_tensor_bundle,
)
from src.models.baselines import (  # noqa: E402
    ClimatologyBaseline,
    LightGBMBaseline,
    RidgeBaseline,
    is_lightgbm_available,
)
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("neer.evaluate_baselines")

DEFAULT_TENSORS_PATH = PROJECT_ROOT / "data" / "processed" / "neer_tensors.npz"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODELS = ["climatology", "ridge", "lightgbm"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NEER's baseline models (Phase 24).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=DEFAULT_TENSORS_PATH,
        help=(
            "Path to the assembled tensor bundle (default: "
            f"{DEFAULT_TENSORS_PATH}). The same file scripts/train.py "
            "reads — every baseline is scored on exactly this file's "
            "train/val/test split."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(DEFAULT_MODELS),
        help="Which baselines to run (default: all of them).",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge L2 strength.")
    parser.add_argument("--lgbm-n-estimators", type=int, default=100)
    parser.add_argument("--lgbm-max-depth", type=int, default=-1)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-num-leaves", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser


def build_models(args: argparse.Namespace) -> List[BaselineModel]:
    models: List[BaselineModel] = []
    for requested in args.models:
        if requested == "climatology":
            models.append(ClimatologyBaseline())
        elif requested == "ridge":
            models.append(RidgeBaseline(alpha=args.ridge_alpha, random_state=args.seed))
        elif requested == "lightgbm":
            if not is_lightgbm_available():
                logger.warning(
                    "LightGBM requested but not installed (pip install lightgbm) — skipping."
                )
                continue
            models.append(
                LightGBMBaseline(
                    n_estimators=args.lgbm_n_estimators,
                    max_depth=args.lgbm_max_depth,
                    learning_rate=args.lgbm_learning_rate,
                    num_leaves=args.lgbm_num_leaves,
                    random_state=args.seed,
                )
            )
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"unknown baseline '{requested}'")
    return models


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logger.info("=" * 72)
    logger.info("NEER baseline evaluation — Phase 24")
    logger.info("=" * 72)

    if not args.tensors.exists():
        logger.error(
            "Tensor bundle not found at %s. Generate it first with "
            "'python scripts/create_demo_data.py' then "
            "'python scripts/preprocess_data.py'.",
            args.tensors,
        )
        return 2

    bundle = TensorBundle.load(args.tensors)
    logger.info(
        "Loaded tensor bundle: %d timesteps, grid=%s, channels=%s",
        bundle.n_time,
        bundle.grid_shape,
        bundle.channel_names,
    )

    # One pool, shared by every model below — the structural guarantee
    # that every baseline in this run sees the same train/val/test data.
    pooled = pool_tensor_bundle(bundle, source_tensors_path=args.tensors)
    for name in SPLIT_NAMES:
        logger.info("  %-5s: %d samples", name, pooled[name].n_samples)

    models = build_models(args)
    if not models:
        logger.error("No baselines to run (check --models and installed dependencies).")
        return 2

    reports = {}
    for model in models:
        logger.info("-" * 72)
        logger.info("Fitting and evaluating: %s", model.name)
        try:
            report = evaluate_baseline(model, pooled)
        except MissingDependencyError as exc:
            logger.warning("Skipping %s: %s", model.name, exc)
            continue
        reports[model.name] = report.to_dict()
        for split in SPLIT_NAMES:
            overall = report.metrics[split].overall
            logger.info(
                "  %-5s | rmse=%s mae=%s bias=%s r2=%s (n=%d)",
                split,
                _fmt(overall.rmse),
                _fmt(overall.mae),
                _fmt(overall.bias),
                _fmt(overall.r2),
                overall.n,
            )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.reports_dir / "baseline_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_tensors_path": str(args.tensors),
                "split_sample_counts": {name: pooled[name].n_samples for name in SPLIT_NAMES},
                "models": reports,
            },
            f,
            indent=2,
        )

    logger.info("-" * 72)
    logger.info("Wrote baseline evaluation report to %s", output_path)
    return 0


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())