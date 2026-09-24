#!/usr/bin/env python3
"""
NEER — Phase 25 evaluation engine.

Builds on Phase 24's shared evaluation interface
(`src.evaluation.interface`/`src.evaluation.metrics`) to produce the
two artifacts a model comparison actually needs:

  * ``reports/evaluation_report.json`` — RMSE, MAE, bias and Pearson
    correlation, computed overall, per depth level, and (where the
    tensor bundle stacks more than one target variable) per variable,
    for every split and every model that ran.
  * ``reports/plots/*.png`` — RMSE, MAE, bias and Pearson correlation
    plotted against depth, one PNG per metric per split, overlaying
    every model that ran so they can be compared depth-by-depth.

Models evaluated
-----------------
Every baseline in ``src.models.baselines`` (climatology always,
Ridge/LightGBM when their dependency is installed) runs the same way
``scripts/evaluate_baseline.py`` runs them — fit on ``train``, scored
on every split, all from one ``pool_tensor_bundle`` call so every model
in a run is compared on identical data.

If ``--neer-checkpoint`` points at a checkpoint written by
``scripts/train.py`` and ``torch`` is installed, the trained
``NEERModel`` is evaluated too: it is run over each split's full input
grid, its ``(n_samples, n_depth)`` output is scored against the exact
same pooled target/mask ``pool_tensor_bundle`` built for the baselines
(so NEER and the baselines are held to one shared target, not two
approximations of it), and it appears in the same report and plots
under the name ``neer``. Without ``torch`` installed (or without a
checkpoint at that path), NEER is skipped with a clear log message —
never silently — and the report still contains whichever baselines did
run.

Synthetic-data labeling
------------------------
``reports/evaluation_report.json`` records ``"is_synthetic"`` (and the
bundle's ``"data_mode"``) read straight from the tensor bundle's own
provenance (``TensorBundle.attrs``, set by
``scripts/preprocess_data.py`` from the dataset it read). Every plot's
title and every console summary line carries the same label. Run
against the shipped demo tensor bundle (the default ``--tensors``),
this is always ``DEMO_SYNTHETIC`` — the results below are for
exercising this evaluation engine end-to-end, not a scientific
statement about ocean temperatures.

Usage
-----
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --tensors data/processed/neer_tensors.npz
    python scripts/run_evaluation.py --models climatology ridge
    python scripts/run_evaluation.py --neer-checkpoint artifacts/checkpoints/neer_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.errors import MissingDependencyError  # noqa: E402
from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.evaluation.interface import (  # noqa: E402
    BaselineModel,
    EvaluationReport,
    PooledSplit,
    SPLIT_NAMES,
    evaluate_baseline,
    pool_tensor_bundle,
)
from src.evaluation.metrics import ProfileMetrics, compute_profile_metrics  # noqa: E402
from src.evaluation.plots import plot_all_metrics  # noqa: E402
from src.models.baselines import (  # noqa: E402
    ClimatologyBaseline,
    LightGBMBaseline,
    RidgeBaseline,
    is_lightgbm_available,
)
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("neer.run_evaluation")

DEFAULT_TENSORS_PATH = PROJECT_ROOT / "data" / "processed" / "neer_tensors.npz"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_NEER_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "neer_best.pt"
DEFAULT_MODELS = ["climatology", "ridge", "lightgbm"]

SYNTHETIC_BANNER = "*" * 72 + "\n*** SYNTHETIC DEMO DATA — NOT REAL OBSERVATIONS ***\n" + "*" * 72


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NEER evaluation engine (Phase 25): JSON reports + depth plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tensors", type=Path, default=DEFAULT_TENSORS_PATH)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, choices=list(DEFAULT_MODELS)
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--lgbm-n-estimators", type=int, default=100)
    parser.add_argument("--lgbm-max-depth", type=int, default=-1)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-num-leaves", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--neer-checkpoint",
        type=Path,
        default=DEFAULT_NEER_CHECKPOINT,
        help="A scripts/train.py checkpoint to evaluate alongside the baselines "
        "(skipped with a log message if missing, or if torch isn't installed).",
    )
    parser.add_argument("--no-neer", action="store_true", help="Never attempt to evaluate NEER.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--split", default="test", choices=list(SPLIT_NAMES), help="Split the plots are drawn for."
    )
    return parser


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def build_baselines(args: argparse.Namespace) -> List[BaselineModel]:
    models: List[BaselineModel] = []
    for requested in args.models:
        if requested == "climatology":
            models.append(ClimatologyBaseline())
        elif requested == "ridge":
            models.append(RidgeBaseline(alpha=args.ridge_alpha, random_state=args.seed))
        elif requested == "lightgbm":
            if not is_lightgbm_available():
                logger.warning("LightGBM requested but not installed — skipping.")
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
    return models


# --------------------------------------------------------------------------
# Optional NEER checkpoint evaluation
# --------------------------------------------------------------------------


def evaluate_neer_checkpoint(
    checkpoint_path: Path,
    bundle: TensorBundle,
    pooled: Dict[str, PooledSplit],
) -> Optional[EvaluationReport]:
    """Score a trained `NEERModel` checkpoint on the exact same pooled
    target/mask every baseline above was scored against.

    `NEERModel.forward` takes the *full* input grid, `(batch, channel,
    lat, lon)` (unlike the baselines, which fit `PooledSplit.features`),
    so this reads `bundle.inputs` directly per split rather than going
    through `evaluate_baseline`. Its `(n_samples, n_depth)` prediction is
    then scored with the same `compute_profile_metrics` call the
    baselines use, against `pooled[split].profile`/`profile_mask` —
    so "NEER's number" and "the baselines' numbers" are never two
    different measurements of two different things.

    Returns `None` (after logging why) if `torch` is not installed or
    `checkpoint_path` does not exist — never raises, since a missing
    optional model should not stop the baselines' report from being
    written.
    """
    if not checkpoint_path.exists():
        logger.info("No NEER checkpoint at %s — skipping NEER evaluation.", checkpoint_path)
        return None
    try:
        import torch

        from src.models.neer_model import NEERModel
        from src.utils.config import load_config
    except ImportError:
        logger.warning(
            "torch is not installed in this environment — skipping NEER evaluation "
            "(the baselines above are unaffected). Install with 'pip install torch' "
            "to include a trained NEERModel checkpoint in this report."
        )
        return None

    logger.info("Loading NEER checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    environment = checkpoint.get("args", {}).get("environment", "demo")
    model = NEERModel.from_config(load_config(environment))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics: Dict[str, ProfileMetrics] = {}
    counts: Dict[str, int] = {}
    for split_name, split in pooled.items():
        mask = np.asarray(bundle.split_masks[split_name], dtype=bool)
        inputs = torch.from_numpy(np.asarray(bundle.inputs[mask], dtype=np.float32))
        with torch.no_grad():
            prediction = model(inputs).cpu().numpy()
        metrics[split_name] = compute_profile_metrics(
            split.profile,
            prediction,
            split.profile_mask,
            depth_names=split.depth_names,
            variable_names=split.variable_names or None,
        )
        counts[split_name] = split.n_samples

    return EvaluationReport(
        model_name="neer",
        model_config={"checkpoint": str(checkpoint_path), "epoch": checkpoint.get("epoch")},
        metrics=metrics,
        source_tensors_path=None,
        split_sample_counts=counts,
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logger.info("=" * 72)
    logger.info("NEER evaluation engine — Phase 25")
    logger.info("=" * 72)

    if not args.tensors.exists():
        logger.error(
            "Tensor bundle not found at %s. Generate it first with "
            "'python scripts/create_demo_data.py' then 'python scripts/preprocess_data.py'.",
            args.tensors,
        )
        return 2

    bundle = TensorBundle.load(args.tensors)
    is_synthetic = bool(bundle.attrs.get("is_synthetic")) or "SYNTHETIC" in str(
        bundle.attrs.get("data_mode", "")
    ).upper()
    data_mode = bundle.attrs.get("data_mode", "UNKNOWN")

    if is_synthetic:
        logger.warning("\n%s", SYNTHETIC_BANNER)
    logger.info(
        "Loaded tensor bundle: %d timesteps, grid=%s, channels=%s, data_mode=%s",
        bundle.n_time,
        bundle.grid_shape,
        bundle.channel_names,
        data_mode,
    )

    pooled = pool_tensor_bundle(bundle, source_tensors_path=args.tensors)
    for name in SPLIT_NAMES:
        logger.info("  %-5s: %d samples", name, pooled[name].n_samples)

    reports: Dict[str, EvaluationReport] = {}

    for model in build_baselines(args):
        logger.info("-" * 72)
        logger.info("Fitting and evaluating: %s", model.name)
        try:
            report = evaluate_baseline(model, pooled)
        except MissingDependencyError as exc:
            logger.warning("Skipping %s: %s", model.name, exc)
            continue
        reports[model.name] = report
        for split in SPLIT_NAMES:
            overall = report.metrics[split].overall
            logger.info(
                "  %-5s | rmse=%s mae=%s bias=%s pearson=%s (n=%d)",
                split,
                _fmt(overall.rmse),
                _fmt(overall.mae),
                _fmt(overall.bias),
                _fmt(overall.pearson),
                overall.n,
            )

    if not args.no_neer:
        logger.info("-" * 72)
        neer_report = evaluate_neer_checkpoint(args.neer_checkpoint, bundle, pooled)
        if neer_report is not None:
            reports["neer"] = neer_report
            for split in SPLIT_NAMES:
                overall = neer_report.metrics[split].overall
                logger.info(
                    "  %-5s | rmse=%s mae=%s bias=%s pearson=%s (n=%d)",
                    split,
                    _fmt(overall.rmse),
                    _fmt(overall.mae),
                    _fmt(overall.bias),
                    _fmt(overall.pearson),
                    overall.n,
                )

    if not reports:
        logger.error("No models produced a report (check --models / dependencies).")
        return 2

    # -- JSON report ---------------------------------------------------
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.reports_dir / "evaluation_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": 25,
                "is_synthetic": is_synthetic,
                "data_mode": data_mode,
                "disclaimer": (
                    "Computed on the NEER synthetic demo dataset — NOT real "
                    "observations. For pipeline demonstration only."
                    if is_synthetic
                    else None
                ),
                "source_tensors_path": str(args.tensors),
                "split_sample_counts": {name: pooled[name].n_samples for name in SPLIT_NAMES},
                "metrics_reported": ["rmse", "mae", "bias", "pearson", "r2"],
                "models": {name: report.to_dict() for name, report in reports.items()},
            },
            f,
            indent=2,
        )
    logger.info("-" * 72)
    logger.info("Wrote evaluation report to %s", output_path)

    # -- Plots -----------------------------------------------------------
    depth_names = pooled[args.split].depth_names
    series = {name: report.metrics[args.split] for name, report in reports.items()}
    try:
        saved = plot_all_metrics(
            depth_names,
            series,
            args.reports_dir / "plots",
            split=args.split,
            is_synthetic=is_synthetic,
        )
    except MissingDependencyError as exc:
        logger.warning("Skipping plots: %s", exc)
        saved = {}

    for metric, path in saved.items():
        logger.info("Wrote %s-vs-depth plot to %s", metric, path)

    if is_synthetic:
        logger.warning("\n%s", SYNTHETIC_BANNER)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())