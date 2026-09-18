#!/usr/bin/env python3
"""
NEER — run the preprocessing pipeline and write model-ready tensors.

Usage
-----
    # The synthetic demo dataset, with settings from configs/
    python scripts/preprocess_data.py

    # A real file, into a chosen directory
    python scripts/preprocess_data.py data/raw/argo_grid.nc -o data/processed/argo

    # Override the split, or the normalization method
    python scripts/preprocess_data.py --test-fraction 0.2 --normalization robust

    # See what would happen without writing anything
    python scripts/preprocess_data.py --dry-run

Outputs (into `-o`, default `data/processed/`)
---------------------------------------------
    neer_tensors.npz                     inputs, targets, masks, coords, split
    neer_preprocessing_metadata.json     the full record of the run
    neer_preprocessing_report.txt        the same, readable

Exit codes
----------
    0  preprocessing completed
    1  preprocessing failed (leakage guard tripped, bad config, ...)
    2  the input dataset could not be loaded

The pipeline never modifies its input. Synthetic data keeps its
`DEMO_SYNTHETIC` marker all the way into the tensor file and the
metadata, so demo output cannot quietly be mistaken for real results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import (  # noqa: E402
    LoaderError,
    load_csv,
    load_demo_dataset,
    load_netcdf,
    load_npz,
)
from src.data.preprocessing import (  # noqa: E402
    Normalizer,
    PreprocessingError,
    PreprocessingPipeline,
    TemporalSplit,
)
from src.utils.config import load_config  # noqa: E402

LOADERS = {
    ".nc": load_netcdf,
    ".nc4": load_netcdf,
    ".netcdf": load_netcdf,
    ".csv": load_csv,
    ".npz": load_npz,
}

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NEER preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset to preprocess. Defaults to the synthetic demo dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for tensors and metadata (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-e",
        "--environment",
        default=None,
        help="Config environment to take pipeline settings from.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Share of the time axis held out for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Share of the time axis held out for testing.",
    )
    parser.add_argument(
        "--train-end",
        default=None,
        help="Explicit split boundary, e.g. 2021-01-01 (exclusive upper bound of train).",
    )
    parser.add_argument(
        "--val-end",
        default=None,
        help="Explicit end of the validation period, e.g. 2021-07-01.",
    )
    parser.add_argument(
        "--normalization",
        choices=("zscore", "minmax", "robust"),
        default=None,
        help="Override the normalization method.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Write the tensor archive uncompressed (faster, much larger).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline and report, but write nothing to disk.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only print the output paths."
    )
    return parser


def load_dataset(path_argument: str | None):
    """Load the demo dataset, or a file, choosing the loader by extension."""
    if path_argument is None:
        return load_demo_dataset(), "demo dataset"

    path = Path(path_argument)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise LoaderError(
            f"no loader for '{path.suffix}'; supported: {sorted(LOADERS)}"
        )
    return loader(path), str(path)


def build_pipeline(args: argparse.Namespace) -> PreprocessingPipeline:
    """Assemble the pipeline from config, then apply command-line overrides."""
    config = load_config(args.environment)
    pipeline = PreprocessingPipeline.from_config(config, environment=args.environment)

    if args.val_fraction is not None:
        pipeline.val_fraction = args.val_fraction
    if args.test_fraction is not None:
        pipeline.test_fraction = args.test_fraction
    if args.normalization is not None:
        normalizer = pipeline.step("normalization")
        pipeline.steps[pipeline.steps.index(normalizer)] = Normalizer(
            method=args.normalization,
            per_depth_level=normalizer.per_depth_level,
            exclude=normalizer.exclude,
        )
    if args.train_end or args.val_end:
        if not (args.train_end and args.val_end):
            raise PreprocessingError(
                "--train-end and --val-end must be given together"
            )
        pipeline.split = TemporalSplit(train_end=args.train_end, val_end=args.val_end)
    return pipeline


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a))

    try:
        dataset, description = load_dataset(args.dataset)
    except (LoaderError, FileNotFoundError, OSError) as error:
        print(f"Could not load the dataset: {error}", file=sys.stderr)
        return 2

    say(f"Loaded {description}")
    say(f"  {dataset.describe()}")
    if dataset.is_synthetic:
        say("\n  ⚠ SYNTHETIC DATA — the tensors produced here are not scientific output.")

    try:
        pipeline = build_pipeline(args)
        say(f"\n{pipeline.describe()}\n")
        result = pipeline.run(dataset)
    except PreprocessingError as error:
        print(f"\nPreprocessing failed: {error}", file=sys.stderr)
        return 1

    say(result.metadata.to_text())

    for check in result.checks:
        say(f"  ✓ {check}")

    if args.dry_run:
        say("\nDry run — nothing written.")
        return 0

    written = result.save(args.output, compress=not args.no_compress)
    say("")
    for label, path in written.items():
        print(f"{label:>9}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
