#!/usr/bin/env python3
"""
NEER — Phase 26 independent ARGO validation.

    ARGO profiles -> QC -> time matching -> spatial matching
      -> vertical interpolation -> NEER prediction -> comparison -> metrics

Reports: float count, profile count, matched profiles, depth coverage,
RMSE, MAE, bias (NEER - ARGO) and correlation — overall, per depth and per
NEER split. Written under ``reports/argo_validation/``, separate from the
reanalysis/test-target evaluation (``scripts/run_evaluation.py``).

Real ARGO data
--------------
    python scripts/run_argo_validation.py --argo data/raw/argo_profiles.csv \
        --neer-checkpoint artifacts/checkpoints/neer_best.pt
    python scripts/run_argo_validation.py --argo data/raw/argo_nc/ \
        --predictions artifacts/predictions/neer_pooled_predictions.npz

``--argo`` takes a long-format CSV, a GDAC ``*_prof.nc`` file, or a
directory of them. NEER predictions come from a checkpoint (needs torch)
or a saved prediction table (`NeerPredictions.save`).

No real ARGO data? Demo mode
-----------------------------
    python scripts/run_argo_validation.py --demo

generates a clearly labelled SYNTHETIC ARGO-like file, runs the pipeline on
it, and labels every output ``DEMO_SYNTHETIC``. Without a checkpoint or
prediction table the prediction side is a labelled stand-in that is NOT
NEER. Demo runs only show that the pipeline executes; they are never
observational validation, and this script never produces demo data unless
``--demo`` is passed.

Exit codes: 0 ok · 2 missing/invalid inputs · 3 ran, but no profiles matched
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.argo_validation import (  # noqa: E402
    ArgoValidationConfig,
    NeerGrid,
    SpaceMatchConfig,
    TimeMatchConfig,
    VerticalConfig,
    demo_stand_in_predictions,
    generate_demo_argo,
    load_argo,
    load_prediction_table,
    predictions_from_checkpoint,
    run_argo_validation,
    write_outputs,
)
from src.argo_validation.pipeline import DEMO_BANNER  # noqa: E402
from src.data.loaders.errors import LoaderError  # noqa: E402
from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("neer.argo_validation")

DEFAULT_TENSORS = PROJECT_ROOT / "data" / "processed" / "neer_tensors.npz"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "neer_preprocessing_metadata.json"
DEFAULT_REPORTS = PROJECT_ROOT / "reports" / "argo_validation"
DEFAULT_DEMO_PATH = PROJECT_ROOT / "data" / "demo" / "argo" / "demo_argo_profiles_SYNTHETIC.csv"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEER ARGO validation (Phase 26).",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("ARGO data")
    src.add_argument("--argo", type=Path, help="ARGO profiles: .csv, GDAC .nc, or a directory of .nc")
    src.add_argument("--demo", action="store_true",
                     help="generate + use a clearly labelled SYNTHETIC ARGO-like file (NOT real data)")
    src.add_argument("--demo-output", type=Path, default=DEFAULT_DEMO_PATH)
    src.add_argument("--n-floats", type=int, default=24, help="demo only")
    src.add_argument("--seed", type=int, default=42, help="demo only")

    neer = p.add_argument_group("NEER side")
    neer.add_argument("--tensors", type=Path, default=DEFAULT_TENSORS, help="TensorBundle .npz (grid, time, splits)")
    neer.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                      help="preprocessing metadata JSON (normalization stats, for --neer-checkpoint)")
    neer.add_argument("--neer-checkpoint", type=Path, help="scripts/train.py checkpoint (requires torch)")
    neer.add_argument("--predictions", type=Path, help="saved NeerPredictions .npz (torch-free)")
    neer.add_argument("--save-predictions", type=Path, help="write the checkpoint's predictions here")

    m = p.add_argument_group("matching / QC / interpolation")
    m.add_argument("--accept-qc", nargs="+", default=["1"], help="ARGO QC flags accepted for values (default: 1)")
    m.add_argument("--min-levels", type=int, default=5)
    m.add_argument("--time-mode", choices=("auto", "monthly", "nearest"), default="auto")
    m.add_argument("--time-tolerance-days", type=float, default=None)
    m.add_argument("--max-cell-offset", type=float, default=0.5, help="grid cells (0.5 = must lie inside the cell)")
    m.add_argument("--gap-floor-m", type=float, default=25.0)
    m.add_argument("--gap-rel", type=float, default=0.25)
    m.add_argument("--edge-tolerance-m", type=float, default=10.0)

    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--no-plots", action="store_true")
    return p


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.argo is None and not args.demo:
        logger.error(
            "No ARGO data given. Pass --argo <file|dir> for real profiles, or --demo to run on a clearly "
            "labelled SYNTHETIC demo file (never real validation)."
        )
        return 2
    if args.argo is not None and args.demo:
        logger.error("--argo and --demo are mutually exclusive.")
        return 2
    if not args.tensors.exists():
        logger.error("Tensor bundle not found at %s (run scripts/create_demo_data.py and scripts/preprocess_data.py).",
                     args.tensors)
        return 2

    bundle = TensorBundle.load(args.tensors)
    grid = NeerGrid.from_bundle(bundle)

    # --- NEER predictions ----------------------------------------------------
    try:
        if args.neer_checkpoint is not None:
            if not args.neer_checkpoint.exists():
                logger.error("Checkpoint not found: %s", args.neer_checkpoint)
                return 2
            predictions = predictions_from_checkpoint(args.neer_checkpoint, bundle, args.metadata)
            if args.save_predictions:
                predictions.save(args.save_predictions)
                logger.info("Saved predictions to %s", args.save_predictions)
        elif args.predictions is not None:
            predictions = load_prediction_table(args.predictions)
        elif args.demo:
            predictions = demo_stand_in_predictions(grid)
            logger.warning("No checkpoint/prediction table: using the labelled DEMO STAND-IN (NOT NEER).")
        else:
            logger.error("No NEER predictions: pass --neer-checkpoint or --predictions.")
            return 2
    except (FileNotFoundError, ValueError, LoaderError) as exc:
        logger.error("Could not obtain NEER predictions: %s", exc)
        return 2

    # --- ARGO profiles ---------------------------------------------------------
    if args.demo:
        path = generate_demo_argo(args.demo_output, grid, n_floats=args.n_floats, seed=args.seed)
        logger.warning("Wrote SYNTHETIC demo ARGO file: %s", path)
        profiles = load_argo(path)
    else:
        try:
            profiles = load_argo(args.argo)
        except (FileNotFoundError, LoaderError) as exc:
            logger.error("Could not load ARGO data: %s", exc)
            return 2

    config = ArgoValidationConfig(
        accept_qc=tuple(args.accept_qc),
        min_levels=args.min_levels,
        time=TimeMatchConfig(mode=args.time_mode, tolerance_days=args.time_tolerance_days),
        space=SpaceMatchConfig(max_cell_offset=args.max_cell_offset),
        vertical=VerticalConfig(args.gap_floor_m, args.gap_rel, args.edge_tolerance_m),
    )
    result = run_argo_validation(profiles, grid, predictions, config)
    written = write_outputs(result, args.reports_dir, plots=not args.no_plots)

    # --- console summary ---------------------------------------------------------
    report = result.report
    banner = "*" * 76 + f"\n*** {DEMO_BANNER}\n" + "*" * 76
    if result.is_demo:
        logger.warning("\n%s", banner)
    c = report["counts"]
    logger.info("floats=%d profiles=%d | after QC=%d | matched (time+space)=%d | matched with depth overlap=%d "
                "(%d floats)", c["float_count"], c["profile_count"], c["profiles_after_qc"],
                c["profiles_matched_time_and_space"], c["matched_profiles"], c["matched_float_count"])
    logger.info("rejected: qc=%s matching=%s", c["profiles_rejected_qc"], c["profiles_rejected_matching"])
    cov = report["depth_coverage"]
    if cov.get("n_profiles"):
        logger.info("depth coverage: %.0f%% of model depths per profile on average; deepest valid depth median %g m",
                    100 * cov["mean_fraction_of_model_depths"], cov["deepest_valid_depth_m"]["median"])
    metrics = report.get("metrics") or report.get("pipeline_check_metrics")
    label = "PIPELINE-CHECK (demo)" if result.is_demo else "ARGO validation"
    if metrics:
        o = metrics["overall"]
        logger.info("%s | rmse=%s mae=%s bias=%s corr(pooled)=%s (n=%d pairs)", label, _fmt(o["rmse"]),
                    _fmt(o["mae"]), _fmt(o["bias"]), _fmt(o["correlation"]), o["n_pairs"])
        for name, b in metrics.get("by_split", {}).items():
            logger.info("  %-7s | rmse=%s mae=%s bias=%s (profiles=%d)", name, _fmt(b["rmse"]), _fmt(b["mae"]),
                        _fmt(b["bias"]), b["n_profiles"])
    for w in report["warnings"]:
        logger.warning("WARNING: %s", w)
    for key, path in written.items():
        logger.info("wrote %s: %s", key, path)
    if result.is_demo:
        logger.warning("\n%s", banner)

    return 0 if result.matched else 3


if __name__ == "__main__":
    raise SystemExit(main())