"""
NEER — Neural Embedding based Estimation and Reconstruction

Package: src/data/validation
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

The scientific data-validation layer. Audits a loaded `OceanDataset`
against the project's configuration and against physical reality, and
produces a three-level report — **valid / warning / error**.

    from src.data.loaders import load_demo_dataset
    from src.data.validation import validate

    report = validate(load_demo_dataset())
    report.status          # ValidationStatus.VALID
    report.is_usable       # True when there are no errors
    print(report.to_text())
    report.save("reports/validation.md")

Ten rules run by default (`rules.ALL_RULES`):

    latitude_range        depth_coordinates     nan_values
    longitude_range       time_coordinates      duplicate_timestamps
    resolution            variable_existence    duplicate_coordinates
                          units

**This layer never repairs data.** Findings may carry a suggested
`remedy` in prose, for a human to act on, but nothing here modifies the
dataset it was given — a serious problem is reported, never silently
worked around. See the "No repairs" note in `report.py`.

For the load-time structural gate that decides whether a file can become
an `OceanDataset` at all, see `src/data/loaders/validation.py`; the
distinction is explained at the top of `validator.py`.
"""

from src.data.validation.report import (
    CheckResult,
    Finding,
    ValidationError,
    ValidationReport,
    ValidationStatus,
    merge_statuses,
)
from src.data.validation.rules import (
    ALL_RULES,
    DEFAULT_NAN_WARN_FRACTION,
    LAT_LIMITS,
    LON_LIMITS,
    MAX_DEPTH_M,
    RULES,
    SURFACE_VARIABLES,
    depth_coordinates,
    duplicate_coordinates,
    duplicate_timestamps,
    latitude_range,
    longitude_range,
    nan_values,
    resolution,
    time_coordinates,
    units,
    variable_existence,
)
from src.data.validation.validator import DataValidator, is_valid, validate, validate_or_raise

__all__ = [
    # Entry points
    "validate",
    "validate_or_raise",
    "is_valid",
    "DataValidator",
    # Report
    "ValidationReport",
    "ValidationStatus",
    "ValidationError",
    "CheckResult",
    "Finding",
    "merge_statuses",
    # Rules
    "ALL_RULES",
    "RULES",
    "latitude_range",
    "longitude_range",
    "resolution",
    "depth_coordinates",
    "time_coordinates",
    "variable_existence",
    "units",
    "nan_values",
    "duplicate_timestamps",
    "duplicate_coordinates",
    # Constants
    "SURFACE_VARIABLES",
    "DEFAULT_NAN_WARN_FRACTION",
    "LAT_LIMITS",
    "LON_LIMITS",
    "MAX_DEPTH_M",
]
