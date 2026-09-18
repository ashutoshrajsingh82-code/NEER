"""
The validator: runs the rule suite and assembles the report.

    from src.data.loaders import load_demo_dataset
    from src.data.validation import validate

    report = validate(load_demo_dataset())
    print(report.status)        # ValidationStatus.VALID
    print(report.to_text())
    report.save("reports/demo_validation.md")

Relationship to the loaders' own validation
-------------------------------------------
`src/data/loaders/validation.py` (Phase 05) is the *structural gate* that
runs inside every loader: it decides whether a file can become an
`OceanDataset` at all, and raises on the way in. It is deliberately cheap
and its vocabulary is about structure — dims, coordinate arrays, dtypes.

This package is the *scientific audit* of a dataset that already loaded.
It asks different questions (is the grid resolution the one we configured?
are these the depth levels we expect? are the required variables here?),
produces a durable report artifact with a three-level status, and suggests
remedies without applying any. Both layers share the same primitives —
`loaders.schema`, `loaders.units`, `loaders.missing` — so the two never
disagree about what a canonical unit or a missing value is.

Nothing here modifies the dataset it is given.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from src.data.validation.report import (
    CheckResult,
    ValidationError,
    ValidationReport,
    ValidationStatus,
)
from src.data.validation.rules import ALL_RULES, DEFAULT_NAN_WARN_FRACTION, RULES


def _dataset_summary(dataset) -> Dict[str, Any]:
    """A compact description of what was validated, embedded in the report."""
    summary: Dict[str, Any] = {
        "source": dataset.source,
        "source_format": dataset.source_format,
        "variables": list(dataset.variable_names),
        "sizes": dataset.sizes,
    }
    if dataset.data_mode:
        summary["data_mode"] = dataset.data_mode
        summary["is_synthetic"] = dataset.is_synthetic
    return summary


def validate(
    dataset,
    *,
    config: Any = None,
    environment: Optional[str] = None,
    domain: Any = None,
    depths: Optional[Sequence[float]] = None,
    required_variables: Optional[Sequence[str]] = None,
    checks: Optional[Sequence[str]] = None,
    check_domain: bool = True,
    check_depths: bool = True,
    nan_warn_fraction: float = DEFAULT_NAN_WARN_FRACTION,
) -> ValidationReport:
    """Validate `dataset` scientifically and return a `ValidationReport`.

    Parameters
    ----------
    dataset:
        An `OceanDataset` from any loader. Never modified.
    config / environment:
        The `NeerConfig` to judge the dataset against, or the name of an
        environment to load (`"base"`, `"demo"`, ...). Defaults to the
        active configuration.
    domain / depths:
        Override just the domain bounds or the expected depth levels,
        instead of taking them from the configuration.
    required_variables:
        Variables that must be present; absence is an ERROR. No default —
        see `rules.variable_existence`. `rules.SURFACE_VARIABLES` is a
        ready-made preset.
    checks:
        Subset of `rules.ALL_RULES` to run. Defaults to all ten.
    check_domain / check_depths:
        Set False to skip comparison against the configured domain bounds
        or depth levels (useful for a global product awaiting subsetting).
    nan_warn_fraction:
        Missing fraction above which a variable earns a warning.

    Returns
    -------
    ValidationReport
        Always returned — this function does not raise on invalid data.
        Use `report.raise_for_status()` or `validate_or_raise` for that.

    Raises
    ------
    ValueError
        `checks` names a rule that does not exist.
    """
    selected = tuple(checks) if checks is not None else ALL_RULES
    unknown = [name for name in selected if name not in RULES]
    if unknown:
        raise ValueError(f"unknown check(s): {unknown}; valid checks are {list(ALL_RULES)}")

    if (domain is None and check_domain) or (depths is None and check_depths):
        if config is None:
            from src.utils.config import load_config

            config = load_config(environment)
        if domain is None:
            domain = config.domain
        if depths is None:
            depths = config.depths

    effective_domain = domain if check_domain else None
    effective_depths = depths if check_depths else None

    # Each rule takes only the arguments it needs; keeping this mapping
    # explicit means a new rule cannot silently receive the wrong context.
    arguments: Dict[str, Dict[str, Any]] = {
        "latitude_range": {"domain": effective_domain},
        "longitude_range": {"domain": effective_domain},
        "resolution": {"domain": effective_domain},
        "depth_coordinates": {"depths": effective_depths},
        "time_coordinates": {},
        "variable_existence": {"required": required_variables},
        "units": {},
        "nan_values": {"warn_fraction": nan_warn_fraction},
        "duplicate_timestamps": {},
        "duplicate_coordinates": {},
    }

    report = ValidationReport(
        dataset_source=dataset.source,
        dataset_summary=_dataset_summary(dataset),
    )
    for name in selected:
        result = RULES[name](dataset, **arguments[name])
        report.add(result)
    return report


def validate_or_raise(dataset, *, strict: bool = False, **kwargs: Any) -> ValidationReport:
    """Validate and raise `ValidationError` if the dataset is not usable.

    With `strict=True`, warnings are fatal too. On success the report is
    returned, so a caller can still inspect what was merely warned about.
    """
    report = validate(dataset, **kwargs)
    report.raise_for_status(strict=strict)
    return report


def is_valid(dataset, **kwargs: Any) -> bool:
    """True when `dataset` validates with no errors (warnings tolerated)."""
    return validate(dataset, **kwargs).is_usable


class DataValidator:
    """A validator with fixed settings, for validating many datasets alike.

    Useful in an ingest loop or a test suite where the same contract —
    the same required variables, the same domain, the same thresholds —
    applies to every file.

        validator = DataValidator(required_variables=SURFACE_VARIABLES)
        for path in paths:
            report = validator.validate(load_netcdf(path))
    """

    def __init__(self, **defaults: Any):
        unknown = set(defaults) - {
            "config",
            "environment",
            "domain",
            "depths",
            "required_variables",
            "checks",
            "check_domain",
            "check_depths",
            "nan_warn_fraction",
        }
        if unknown:
            raise TypeError(f"unknown validator option(s): {sorted(unknown)}")
        self.defaults = defaults

    def validate(self, dataset, **overrides: Any) -> ValidationReport:
        return validate(dataset, **{**self.defaults, **overrides})

    def validate_or_raise(
        self, dataset, *, strict: bool = False, **overrides: Any
    ) -> ValidationReport:
        return validate_or_raise(
            dataset, strict=strict, **{**self.defaults, **overrides}
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<DataValidator {self.defaults}>"


__all__ = [
    "validate",
    "validate_or_raise",
    "is_valid",
    "DataValidator",
    "ValidationReport",
    "ValidationStatus",
    "ValidationError",
    "CheckResult",
]
