"""
Validation for loaded NEER datasets.

Every loader runs the checks in this module before handing a dataset
back, so a dataset that reaches modeling code has already been examined
for the failure modes that silently ruin ocean ML pipelines: a flipped
latitude axis, a duplicated timestamp, a variable in Kelvin when the rest
of the pipeline assumes Celsius, a field that is 100% land/NaN.

Errors vs warnings
------------------
The split is deliberate and load-bearing:

* **ERROR** — the dataset is structurally unusable as NEER data.
  Duplicate timestamps, a dimension whose length contradicts its
  coordinate, a fully-missing variable, latitudes outside [-90, 90].
  Loaders raise on these by default (`validate=True`).
* **WARNING** — the dataset is usable but suspicious, and the right
  response depends on the caller. Coordinates outside the configured
  domain, an unknown variable name, an undeclared unit, values outside a
  plausibility band, an irregular time cadence. Loaders record these and
  carry on unless the caller asks for `strict=True`.

Nothing here mutates the dataset. Validation reports; loaders decide.

Scope
-----
This module is the *structural gate* that runs inside every loader: it
decides whether a source can become a usable `OceanDataset` at all, and
loaders raise on its errors during the load. For the fuller *scientific*
audit of a dataset that has already loaded — grid resolution against the
configured one, depth levels against the configured set, required-variable
contracts, and a saveable valid/warning/error report — see
`src/data/validation/`. Both layers share the same primitives
(`schema`, `units`, `missing`), so they cannot disagree about what a
canonical unit or a missing value is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.loaders.errors import DataValidationError
from src.data.loaders.missing import missing_fraction
from src.data.loaders.schema import CANONICAL_DIM_ORDER, lookup_spec
from src.data.loaders.units import normalize_unit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.data.loaders.representation import OceanDataset

#: Physical limits. Outside these the coordinate is simply wrong, not merely
#: outside the project's area of interest.
LAT_LIMITS = (-90.0, 90.0)
LON_LIMITS = (-360.0, 360.0)
MAX_DEPTH_M = 11_000.0  # deeper than the Challenger Deep

#: A variable missing more than this fraction is flagged. The demo dataset's
#: subsurface temperature is deliberately ~75-85% missing (land + sparse
#: ARGO-like sampling), which is realistic, so the threshold sits above that
#: and exists to catch the "everything is NaN" case that follows a bad read.
DEFAULT_MISSING_WARN_FRACTION = 0.95

#: Relative tolerance for "evenly spaced" coordinate checks.
SPACING_RTOL = 1e-3


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    """One finding from one check."""

    severity: Severity
    check: str
    message: str
    target: Optional[str] = None  # variable or coordinate the issue concerns

    def __str__(self) -> str:
        where = f" [{self.target}]" if self.target else ""
        return f"{self.severity.value.upper():<7} {self.check}{where}: {self.message}"


@dataclass
class ValidationReport:
    """The collected findings for one dataset."""

    issues: List[ValidationIssue] = field(default_factory=list)
    dataset_source: Optional[str] = None

    # -- building ----------------------------------------------------------

    def add(
        self, severity: Severity, check: str, message: str, target: Optional[str] = None
    ) -> None:
        self.issues.append(ValidationIssue(severity, check, message, target))

    def error(self, check: str, message: str, target: Optional[str] = None) -> None:
        self.add(Severity.ERROR, check, message, target)

    def warn(self, check: str, message: str, target: Optional[str] = None) -> None:
        self.add(Severity.WARNING, check, message, target)

    def info(self, check: str, message: str, target: Optional[str] = None) -> None:
        self.add(Severity.INFO, check, message, target)

    def extend(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues.extend(issues)

    # -- querying ----------------------------------------------------------

    def _of(self, severity: Severity) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity is severity]

    @property
    def errors(self) -> List[ValidationIssue]:
        return self._of(Severity.ERROR)

    @property
    def warnings(self) -> List[ValidationIssue]:
        return self._of(Severity.WARNING)

    @property
    def infos(self) -> List[ValidationIssue]:
        return self._of(Severity.INFO)

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings are allowed)."""
        return not self.errors

    def checks_run(self) -> List[str]:
        return sorted({i.check for i in self.issues})

    def summary_line(self) -> str:
        return (
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s), "
            f"{len(self.infos)} info"
        )

    def raise_for_status(self, *, strict: bool = False) -> None:
        """Raise `DataValidationError` if validation failed.

        With `strict=True`, warnings are treated as failures too.
        """
        failures = list(self.errors)
        if strict:
            failures += self.warnings
        if not failures:
            return
        detail = "\n".join(f"  - {issue}" for issue in failures)
        source = f" for {self.dataset_source}" if self.dataset_source else ""
        raise DataValidationError(
            self,
            f"Dataset validation failed{source} ({self.summary_line()}):\n{detail}",
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form, for logs and API responses."""
        return {
            "source": self.dataset_source,
            "ok": self.ok,
            "counts": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.infos),
            },
            "issues": [
                {
                    "severity": i.severity.value,
                    "check": i.check,
                    "target": i.target,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }

    def __str__(self) -> str:
        if not self.issues:
            return "Validation passed with no issues."
        return "\n".join([self.summary_line()] + [f"  {i}" for i in self.issues])

    def __len__(self) -> int:
        return len(self.issues)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_coordinates(dataset: "OceanDataset", report: ValidationReport, *, domain=None) -> None:
    """Validate coordinate arrays: presence, dtype, monotonicity, spacing, range.

    `domain` (a `DomainConfig`) enables the "inside the NEER domain" check;
    pass None to skip it.
    """
    check = "coordinates"

    if not dataset.coords:
        report.error(check, "dataset has no coordinates")
        return

    for name in ("lat", "lon"):
        if name not in dataset.coords:
            report.error(check, f"required coordinate '{name}' is missing", target=name)

    for name, values in dataset.coords.items():
        if name == "time":
            continue  # handled by check_timestamps

        array = np.asarray(values)
        if array.size == 0:
            report.error(check, "coordinate is empty", target=name)
            continue
        if not np.issubdtype(array.dtype, np.number):
            report.error(
                check, f"coordinate must be numeric, got dtype {array.dtype}", target=name
            )
            continue
        if np.isnan(array).any():
            report.error(check, "coordinate contains NaN", target=name)
            continue
        if array.size != np.unique(array).size:
            report.error(check, "coordinate contains duplicate values", target=name)

        if array.size > 1:
            diffs = np.diff(array)
            if not np.all(diffs > 0):
                if np.all(diffs < 0):
                    report.error(
                        check,
                        "coordinate is descending; NEER expects ascending coordinates "
                        "(flip the axis before loading)",
                        target=name,
                    )
                else:
                    report.error(check, "coordinate is not monotonic", target=name)
            elif name in ("lat", "lon"):
                spacing = float(np.median(diffs))
                if spacing > 0 and not np.allclose(diffs, spacing, rtol=SPACING_RTOL):
                    report.warn(
                        check,
                        f"coordinate is not evenly spaced (median step {spacing:g}°); "
                        "regridding may be required",
                        target=name,
                    )

        if name == "lat":
            if array.min() < LAT_LIMITS[0] or array.max() > LAT_LIMITS[1]:
                report.error(
                    check,
                    f"latitudes must lie within {LAT_LIMITS}, got "
                    f"[{array.min():g}, {array.max():g}]",
                    target=name,
                )
        elif name == "lon":
            if array.min() < LON_LIMITS[0] or array.max() > LON_LIMITS[1]:
                report.error(
                    check,
                    f"longitudes must lie within {LON_LIMITS}, got "
                    f"[{array.min():g}, {array.max():g}]",
                    target=name,
                )
        elif name == "depth":
            if array.min() < 0:
                report.error(
                    check,
                    f"depths must be non-negative (positive down), got min {array.min():g}",
                    target=name,
                )
            if array.max() > MAX_DEPTH_M:
                report.warn(
                    check,
                    f"maximum depth {array.max():g} m exceeds {MAX_DEPTH_M:g} m; "
                    "are these metres?",
                    target=name,
                )

    if domain is not None:
        _check_within_domain(dataset, report, domain, check=check)


def _check_within_domain(
    dataset: "OceanDataset", report: ValidationReport, domain, *, check: str
) -> None:
    """Warn when coordinates fall outside the configured NEER domain.

    A warning rather than an error: a source file covering a wider area is
    perfectly legitimate input, it just needs subsetting before use.
    """
    lat = dataset.coords.get("lat")
    lon = dataset.coords.get("lon")

    if lat is not None and lat.size:
        if lat.min() < domain.lat_min or lat.max() > domain.lat_max:
            report.warn(
                check,
                f"latitudes [{lat.min():g}, {lat.max():g}] extend outside the NEER domain "
                f"[{domain.lat_min:g}, {domain.lat_max:g}]",
                target="lat",
            )
    if lon is not None and lon.size:
        if lon.min() < domain.lon_min or lon.max() > domain.lon_max:
            report.warn(
                check,
                f"longitudes [{lon.min():g}, {lon.max():g}] extend outside the NEER domain "
                f"[{domain.lon_min:g}, {domain.lon_max:g}]",
                target="lon",
            )


def check_timestamps(dataset: "OceanDataset", report: ValidationReport) -> None:
    """Validate the time coordinate: dtype, NaT, ordering, duplicates, cadence."""
    check = "timestamps"
    time = dataset.coords.get("time")

    if time is None:
        report.info(check, "dataset has no time coordinate (single snapshot or static field)")
        return

    array = np.asarray(time)
    if array.size == 0:
        report.error(check, "time coordinate is empty", target="time")
        return

    if not np.issubdtype(array.dtype, np.datetime64):
        report.error(
            check,
            f"time must be decoded to datetime64, got dtype {array.dtype}; "
            "the loader could not parse the source time values",
            target="time",
        )
        return

    if np.isnat(array).any():
        n_bad = int(np.isnat(array).sum())
        report.error(check, f"time contains {n_bad} unparseable timestamp(s) (NaT)", target="time")
        return

    if array.size != np.unique(array).size:
        n_dupes = int(array.size - np.unique(array).size)
        report.error(check, f"time contains {n_dupes} duplicate timestamp(s)", target="time")

    diffs = np.diff(array)
    if array.size > 1 and not np.all(diffs > np.timedelta64(0, "ns")):
        report.error(check, "timestamps are not in strictly increasing order", target="time")
        return

    if array.size > 2:
        steps = diffs.astype("timedelta64[s]").astype(np.int64)
        median = float(np.median(steps))
        if median > 0 and not np.allclose(steps, median, rtol=0.2):
            report.warn(
                check,
                "time steps are irregular "
                f"(median {median / 86400:.2f} days, range "
                f"{steps.min() / 86400:.2f}-{steps.max() / 86400:.2f} days)",
                target="time",
            )


def check_dimensions(dataset: "OceanDataset", report: ValidationReport) -> None:
    """Validate that variable dims are declared, sized correctly, and ordered."""
    check = "dimensions"
    sizes = dataset.sizes

    for name in dataset.variable_names:
        variable = dataset[name]

        if not variable.dims:
            report.error(check, "variable has no declared dimensions", target=name)
            continue

        for dim, length in variable.sizes.items():
            if dim not in sizes:
                report.error(
                    check,
                    f"dimension '{dim}' has no matching coordinate in the dataset",
                    target=name,
                )
            elif sizes[dim] != length:
                report.error(
                    check,
                    f"dimension '{dim}' has length {length} but coordinate '{dim}' "
                    f"has length {sizes[dim]}",
                    target=name,
                )

        known = [d for d in variable.dims if d in CANONICAL_DIM_ORDER]
        expected_order = [d for d in CANONICAL_DIM_ORDER if d in known]
        if known != expected_order:
            report.error(
                check,
                f"dimensions {variable.dims} are not in canonical order "
                f"{tuple(expected_order)}",
                target=name,
            )

        if len(set(variable.dims)) != len(variable.dims):
            report.error(check, f"duplicate dimension names: {variable.dims}", target=name)


def check_variables(dataset: "OceanDataset", report: ValidationReport) -> None:
    """Validate variables: presence, dtype, known names, expected dims, value ranges."""
    check = "variables"

    if not dataset.variables:
        report.error(check, "dataset contains no variables")
        return

    for name in dataset.variable_names:
        variable = dataset[name]
        spec = lookup_spec(name)

        if variable.values.size == 0:
            report.error(check, "variable array is empty", target=name)
            continue

        if not (
            np.issubdtype(variable.dtype, np.number) or variable.dtype == bool
        ):
            report.error(
                check, f"variable dtype {variable.dtype} is not numeric or boolean", target=name
            )
            continue

        if spec is None:
            report.warn(
                check,
                "variable is not in the NEER variable registry "
                "(src/data/loaders/schema.py); it will be carried through unchecked",
                target=name,
            )
            continue

        # Expected dims: a missing time axis is common (a single snapshot)
        # and not an error, but an unexpected *extra* axis is worth flagging.
        expected = set(spec.expected_dims)
        actual = set(variable.dims)
        unexpected = actual - expected
        if unexpected:
            report.warn(
                check,
                f"has unexpected dimension(s) {sorted(unexpected)}; "
                f"expected {spec.expected_dims}",
                target=name,
            )
        elif "time" in expected - actual:
            report.info(
                check,
                f"has no time dimension; expected {spec.expected_dims} "
                "(a single snapshot is fine)",
                target=name,
            )
        if "depth" in expected - actual:
            report.warn(
                check,
                f"has no depth dimension; expected {spec.expected_dims}",
                target=name,
            )

        if spec.is_mask and variable.dtype != bool:
            report.warn(
                check,
                f"mask variable has dtype {variable.dtype}; expected boolean",
                target=name,
            )

        if spec.valid_range is not None:
            value_range = variable.value_range
            if value_range is None:
                continue  # fully missing: reported by check_missing_values
            low, high = spec.valid_range
            observed_low, observed_high = value_range
            if observed_low < low or observed_high > high:
                report.warn(
                    check,
                    f"values [{observed_low:g}, {observed_high:g}] {spec.units} fall outside "
                    f"the plausible range [{low:g}, {high:g}] for {spec.description.lower()}; "
                    "check the source units and any scale/offset factors",
                    target=name,
                )


def check_units(dataset: "OceanDataset", report: ValidationReport) -> None:
    """Validate that units are declared, recognized, and canonical."""
    check = "units"

    for name in dataset.variable_names:
        variable = dataset[name]
        spec = lookup_spec(name)

        if variable.units is None:
            report.warn(check, "no units declared for this variable", target=name)
            continue

        normalized = normalize_unit(variable.units)
        if normalized is None:
            report.warn(
                check,
                f"unrecognized unit {variable.units!r}; add it to "
                "src/data/loaders/units.py if it is valid",
                target=name,
            )
            continue

        if spec is None:
            continue

        expected = normalize_unit(spec.units) or spec.units
        if normalized != expected:
            report.error(
                check,
                f"unit {normalized!r} does not match the canonical unit {expected!r} "
                f"for '{spec.name}' and could not be converted",
                target=name,
            )


def check_missing_values(
    dataset: "OceanDataset",
    report: ValidationReport,
    *,
    warn_fraction: float = DEFAULT_MISSING_WARN_FRACTION,
) -> None:
    """Report per-variable missing fractions; error on fully-missing variables."""
    check = "missing_values"

    for name in dataset.variable_names:
        variable = dataset[name]
        if variable.values.size == 0:
            continue

        fraction = missing_fraction(variable.values)
        if fraction >= 1.0:
            report.error(
                check,
                "variable is entirely missing (100% NaN) — check the fill value, "
                "the variable name, and the subsetting applied at read time",
                target=name,
            )
        elif fraction > warn_fraction:
            report.warn(
                check,
                f"{fraction:.1%} of values are missing",
                target=name,
            )
        elif fraction > 0:
            report.info(check, f"{fraction:.1%} of values are missing", target=name)

        # Infinities are not a valid encoding of anything in NEER; they
        # usually indicate a division that went wrong upstream.
        if np.issubdtype(variable.dtype, np.floating):
            n_inf = int(np.isinf(variable.values).sum())
            if n_inf:
                report.error(
                    check,
                    f"variable contains {n_inf} infinite value(s); missing data must be NaN",
                    target=name,
                )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

#: All checks, in the order they run.
ALL_CHECKS = ("coordinates", "timestamps", "dimensions", "variables", "units", "missing_values")


def validate_dataset(
    dataset: "OceanDataset",
    *,
    domain=None,
    check_domain: bool = True,
    checks: Optional[Sequence[str]] = None,
    missing_warn_fraction: float = DEFAULT_MISSING_WARN_FRACTION,
) -> ValidationReport:
    """Run the validation suite over `dataset` and return a report.

    Parameters
    ----------
    domain:
        The `DomainConfig` to check coordinates against. Defaults to the
        project domain from `configs/*.yaml` when `check_domain` is True.
    check_domain:
        Set False to skip the "inside the NEER domain" comparison entirely
        (useful when loading a global product before subsetting).
    checks:
        Subset of `ALL_CHECKS` to run. Defaults to all of them.
    missing_warn_fraction:
        Missing-fraction above which a variable is flagged with a warning.

    This function never raises on data problems — inspect
    `report.ok` / `report.errors`, or call `report.raise_for_status()`.
    """
    selected = tuple(checks) if checks is not None else ALL_CHECKS
    unknown = [c for c in selected if c not in ALL_CHECKS]
    if unknown:
        raise ValueError(f"unknown check(s): {unknown}; valid checks are {list(ALL_CHECKS)}")

    if domain is None and check_domain:
        from src.utils.config import load_config

        domain = load_config().domain

    report = ValidationReport(dataset_source=dataset.source)

    if "coordinates" in selected:
        check_coordinates(dataset, report, domain=domain if check_domain else None)
    if "timestamps" in selected:
        check_timestamps(dataset, report)
    if "dimensions" in selected:
        check_dimensions(dataset, report)
    if "variables" in selected:
        check_variables(dataset, report)
    if "units" in selected:
        check_units(dataset, report)
    if "missing_values" in selected:
        check_missing_values(dataset, report, warn_fraction=missing_warn_fraction)

    return report
