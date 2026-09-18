#!/usr/bin/env python3
"""
NEER — validate a dataset and write a validation report.

Usage
-----
    # Validate the synthetic demo dataset
    python scripts/validate_data.py

    # Validate a real file, enforcing the surface-variable contract
    python scripts/validate_data.py data/raw/argo_grid.nc --require-surface

    # Write the report somewhere, in the format the extension implies
    python scripts/validate_data.py --report reports/demo_validation.md

    # Fail the command on warnings too (for CI)
    python scripts/validate_data.py --strict

Exit codes
----------
    0  valid (or warnings only, unless --strict)
    1  errors found (or warnings with --strict)
    2  the file could not be loaded at all

This script never modifies the data it validates. Serious problems are
reported with suggested fixes for a human to apply; nothing is repaired
automatically.
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
from src.data.validation import SURFACE_VARIABLES, ValidationStatus, validate  # noqa: E402

LOADERS = {
    ".nc": load_netcdf,
    ".nc4": load_netcdf,
    ".cdf": load_netcdf,
    ".netcdf": load_netcdf,
    ".csv": load_csv,
    ".tsv": load_csv,
    ".npz": load_npz,
}


def load_any(path: Path):
    """Load a dataset by file extension, without validating on the way in.

    Load-time validation is switched off deliberately: this script's job
    is to *report* on the data, and a dataset that the loader's structural
    gate would reject is exactly the case where a full report is most
    useful.
    """
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise LoaderError(
            f"no loader for '{path.suffix}' files; supported: {sorted(LOADERS)}"
        )
    return loader(path, validate=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a NEER dataset and write a validation report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="dataset to validate (.nc, .csv, .npz). Omit to use the demo dataset.",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="write the report here; .json/.md/.txt select the format",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="format printed to stdout (default: text)",
    )
    parser.add_argument(
        "--environment",
        help="configuration environment to validate against (default: active config)",
    )
    parser.add_argument(
        "--require-surface",
        action="store_true",
        help=f"require the surface variables {list(SURFACE_VARIABLES)} to be present",
    )
    parser.add_argument(
        "--require",
        metavar="VAR",
        nargs="+",
        help="require these specific variables to be present",
    )
    parser.add_argument(
        "--no-domain-check",
        action="store_true",
        help="skip comparison against the configured lat/lon domain",
    )
    parser.add_argument(
        "--no-depth-check",
        action="store_true",
        help="skip comparison against the configured depth levels",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print only the summary line"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.path:
            dataset = load_any(Path(args.path))
        else:
            dataset = load_demo_dataset(validate=False)
    except (LoaderError, FileNotFoundError, OSError) as exc:
        print(f"Could not load the dataset: {exc}", file=sys.stderr)
        return 2

    required = list(args.require) if args.require else None
    if args.require_surface:
        required = list(SURFACE_VARIABLES) + (required or [])

    report = validate(
        dataset,
        environment=args.environment,
        required_variables=required,
        check_domain=not args.no_domain_check,
        check_depths=not args.no_depth_check,
    )

    if args.quiet:
        print(report.summary_line())
    elif args.format == "json":
        print(report.to_json())
    elif args.format == "markdown":
        print(report.to_markdown())
    else:
        print(report.to_text())

    if args.report:
        written = report.save(args.report)
        print(f"\nReport written to {written}", file=sys.stderr)

    if dataset.is_synthetic and not args.quiet:
        print(
            "\nNote: this dataset is SYNTHETIC demo data, not observations.",
            file=sys.stderr,
        )

    failed = report.has_errors or (args.strict and report.status is ValidationStatus.WARNING)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
