"""
The scientific validation rules.

One function per item in the Phase 06 checklist. Each takes an
`OceanDataset` plus the configuration to judge it against, and returns a
`CheckResult`. None of them modifies the dataset — see the "No repairs"
note in `report.py`.

    latitude_range        depth_coordinates     nan_values
    longitude_range       time_coordinates      duplicate_timestamps
    resolution            variable_existence    duplicate_coordinates
                          units

Physical limits vs configured domain
------------------------------------
These are deliberately two different severities. A latitude of 130° is
*impossible* — that is an ERROR whatever the project is doing. A latitude
of 45°N is perfectly real but lies outside NEER's configured Arabian
Sea / Bay of Bengal domain (5-30°N), so it is a WARNING: the file is
valid data that needs subsetting, not broken data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.loaders.missing import finite_range, missing_fraction
from src.data.loaders.schema import CANONICAL_COORDS, VARIABLE_SPECS, lookup_spec
from src.data.loaders.units import normalize_unit
from src.data.validation.report import CheckResult, Finding, ValidationStatus

ERROR = ValidationStatus.ERROR
WARNING = ValidationStatus.WARNING

# Physical limits. Outside these, a coordinate is wrong, not merely unusual.
LAT_LIMITS: Tuple[float, float] = (-90.0, 90.0)
LON_LIMITS: Tuple[float, float] = (-360.0, 360.0)
MAX_DEPTH_M = 11_000.0  # deeper than the Challenger Deep

#: Relative tolerance for "is this axis evenly spaced" and
#: "does this spacing match the configured resolution".
SPACING_RTOL = 1e-3

#: A variable missing more than this fraction is flagged. Above the demo
#: dataset's realistic ~76% subsurface gappiness, so it catches the
#: "everything came back NaN" case rather than normal sparse sampling.
DEFAULT_NAN_WARN_FRACTION = 0.95

#: Timestamps outside this era are almost certainly a decoding accident
#: (a wrong epoch, seconds read as days) rather than real data.
PLAUSIBLE_TIME_RANGE = (np.datetime64("1900-01-01"), np.datetime64("2100-01-01"))

#: Convenience preset for `variable_existence(required=...)`: the surface
#: fields NEER's reconstruction work is built around. The configuration
#: files do not declare a required-variable list, so this is a preset to
#: pass explicitly, never an implicit default.
SURFACE_VARIABLES: Tuple[str, ...] = ("sst", "sss", "sla", "u_current", "v_current")


def _result(
    name: str, title: str, findings: List[Finding], details: Optional[Dict[str, Any]] = None
) -> CheckResult:
    return CheckResult(name=name, title=title, findings=findings, details=details or {})


def _skipped(name: str, title: str, reason: str) -> CheckResult:
    return CheckResult(name=name, title=title, skipped_reason=reason)


# --------------------------------------------------------------------------
# 1 & 2. Latitude / longitude range
# --------------------------------------------------------------------------


def _axis_range_check(
    dataset,
    *,
    axis: str,
    name: str,
    title: str,
    limits: Tuple[float, float],
    domain_bounds: Optional[Tuple[float, float]],
) -> CheckResult:
    values = dataset.coords.get(axis)
    if values is None:
        return _result(
            name,
            title,
            [
                Finding(
                    ERROR,
                    f"dataset has no '{axis}' coordinate",
                    target=axis,
                    remedy=f"confirm the source file has a {axis} axis and that the loader "
                    f"recognized its name (see schema.COORD_ALIASES)",
                )
            ],
        )

    values = np.asarray(values, dtype=np.float64)
    findings: List[Finding] = []
    details: Dict[str, Any] = {"n": int(values.size)}

    if values.size == 0:
        findings.append(Finding(ERROR, f"'{axis}' coordinate is empty", target=axis))
        return _result(name, title, findings, details)

    if np.isnan(values).any():
        findings.append(
            Finding(
                ERROR,
                f"{int(np.isnan(values).sum())} of {values.size} values are NaN",
                target=axis,
                remedy=f"drop or reconstruct the rows with a missing {axis}",
            )
        )
        values = values[~np.isnan(values)]
        if values.size == 0:
            return _result(name, title, findings, details)

    observed = (float(values.min()), float(values.max()))
    details["observed_range"] = list(observed)
    details["physical_limits"] = list(limits)

    if observed[0] < limits[0] or observed[1] > limits[1]:
        findings.append(
            Finding(
                ERROR,
                f"range [{observed[0]:g}, {observed[1]:g}] falls outside the physical "
                f"limits [{limits[0]:g}, {limits[1]:g}]",
                target=axis,
                remedy=f"the {axis} axis is not in degrees, or is offset — check the "
                "source units and any scale/offset attributes",
            )
        )

    if domain_bounds is not None:
        details["configured_domain"] = list(domain_bounds)
        if observed[0] < domain_bounds[0] or observed[1] > domain_bounds[1]:
            findings.append(
                Finding(
                    WARNING,
                    f"range [{observed[0]:g}, {observed[1]:g}] extends outside the "
                    f"configured NEER domain [{domain_bounds[0]:g}, {domain_bounds[1]:g}]",
                    target=axis,
                    remedy=f"subset the dataset to the configured {axis} range before use, "
                    "or widen domain in configs/base.yaml",
                )
            )

    if values.size > 1:
        diffs = np.diff(values)
        if np.any(diffs <= 0):
            direction = "descending" if np.all(diffs < 0) else "non-monotonic"
            findings.append(
                Finding(
                    ERROR,
                    f"'{axis}' axis is {direction}; NEER requires strictly ascending "
                    "coordinates",
                    target=axis,
                    remedy=f"sort the dataset along {axis} (and the data arrays with it) "
                    "before loading",
                )
            )

    return _result(name, title, findings, details)


def latitude_range(dataset, *, domain=None) -> CheckResult:
    """Latitude within ±90°, ascending, and inside the configured domain."""
    bounds = (float(domain.lat_min), float(domain.lat_max)) if domain else None
    return _axis_range_check(
        dataset,
        axis="lat",
        name="latitude_range",
        title="Latitude range",
        limits=LAT_LIMITS,
        domain_bounds=bounds,
    )


def longitude_range(dataset, *, domain=None) -> CheckResult:
    """Longitude within ±360°, ascending, and inside the configured domain."""
    bounds = (float(domain.lon_min), float(domain.lon_max)) if domain else None
    return _axis_range_check(
        dataset,
        axis="lon",
        name="longitude_range",
        title="Longitude range",
        limits=LON_LIMITS,
        domain_bounds=bounds,
    )


# --------------------------------------------------------------------------
# 3. Resolution
# --------------------------------------------------------------------------


def resolution(dataset, *, domain=None, rtol: float = SPACING_RTOL) -> CheckResult:
    """Grid spacing: even along each axis, and equal to the configured resolution.

    Uneven spacing is an ERROR — a model that assumes a regular grid will
    produce quietly wrong results on an irregular one. A uniform spacing
    that simply differs from `domain.resolution` is a WARNING: the data is
    fine, it just needs regridding.
    """
    findings: List[Finding] = []
    details: Dict[str, Any] = {}
    expected = float(domain.resolution) if domain else None
    if expected is not None:
        details["configured_resolution"] = expected

    present = [axis for axis in ("lat", "lon") if axis in dataset.coords]
    if not present:
        return _skipped("resolution", "Grid resolution", "no lat/lon coordinates to measure")

    for axis in present:
        values = np.asarray(dataset.coords[axis], dtype=np.float64)
        if values.size < 2:
            findings.append(
                Finding(
                    WARNING,
                    f"'{axis}' has a single point; resolution cannot be determined",
                    target=axis,
                )
            )
            continue

        diffs = np.diff(values)
        if np.any(diffs <= 0):
            # Ordering is reported by the range checks; spacing is meaningless here.
            continue

        observed = float(np.median(diffs))
        details[f"{axis}_resolution"] = observed
        details[f"{axis}_spacing_range"] = [float(diffs.min()), float(diffs.max())]

        if not np.allclose(diffs, observed, rtol=rtol):
            findings.append(
                Finding(
                    ERROR,
                    f"'{axis}' spacing is uneven: steps range from {diffs.min():g}° to "
                    f"{diffs.max():g}° (median {observed:g}°)",
                    target=axis,
                    remedy="regrid onto a regular axis; NEER's spatial modeling assumes "
                    "uniform grid spacing",
                )
            )
            continue

        if expected is not None and not np.isclose(observed, expected, rtol=1e-3):
            findings.append(
                Finding(
                    WARNING,
                    f"'{axis}' resolution is {observed:g}° but the configuration expects "
                    f"{expected:g}°",
                    target=axis,
                    remedy=f"regrid to {expected:g}°, or set domain.resolution to "
                    f"{observed:g}° in configs/base.yaml",
                )
            )

    return _result("resolution", "Grid resolution", findings, details)


# --------------------------------------------------------------------------
# 4. Depth coordinates
# --------------------------------------------------------------------------


def depth_coordinates(dataset, *, depths: Optional[Sequence[float]] = None) -> CheckResult:
    """Depths non-negative, ascending, unique, plausible, and as configured.

    Skipped when the dataset has no depth axis — a surface-only dataset is
    entirely legitimate, not a failure.
    """
    values = dataset.coords.get("depth")
    if values is None:
        return _skipped(
            "depth_coordinates",
            "Depth coordinates",
            "dataset has no depth axis (surface-only data)",
        )

    values = np.asarray(values, dtype=np.float64)
    findings: List[Finding] = []
    details: Dict[str, Any] = {"n_levels": int(values.size)}

    if values.size == 0:
        return _result(
            "depth_coordinates",
            "Depth coordinates",
            [Finding(ERROR, "depth coordinate is empty", target="depth")],
            details,
        )

    details["observed_range"] = [float(values.min()), float(values.max())]

    if np.isnan(values).any():
        findings.append(Finding(ERROR, "depth coordinate contains NaN", target="depth"))

    finite = values[~np.isnan(values)]
    if finite.size and finite.min() < 0:
        findings.append(
            Finding(
                ERROR,
                f"depths must be non-negative (positive down), found {finite.min():g}",
                target="depth",
                remedy="negate the axis if the source uses negative-down height, and "
                "re-sort the data with it",
            )
        )

    if finite.size > 1 and np.any(np.diff(finite) <= 0):
        direction = "descending" if np.all(np.diff(finite) < 0) else "non-monotonic"
        findings.append(
            Finding(
                ERROR,
                f"depth axis is {direction}; NEER requires shallow-to-deep ordering",
                target="depth",
                remedy="sort the depth axis ascending, reordering the data arrays with it",
            )
        )

    if finite.size and finite.max() > MAX_DEPTH_M:
        findings.append(
            Finding(
                ERROR,
                f"maximum depth {finite.max():g} exceeds the deepest point in any ocean "
                f"({MAX_DEPTH_M:g} m)",
                target="depth",
                remedy="check the depth units — centimetres or pressure in Pa read as "
                "metres produce values like this",
            )
        )

    if depths is not None:
        configured = np.asarray([float(d) for d in depths], dtype=np.float64)
        details["configured_levels"] = configured.tolist()
        if finite.size != configured.size or not np.allclose(finite, configured):
            extra = sorted(set(finite.tolist()) - set(configured.tolist()))
            absent = sorted(set(configured.tolist()) - set(finite.tolist()))
            parts = []
            if absent:
                parts.append(f"missing {absent}")
            if extra:
                parts.append(f"unexpected {extra}")
            findings.append(
                Finding(
                    WARNING,
                    "depth levels differ from the configured set"
                    + (f" ({'; '.join(parts)})" if parts else ""),
                    target="depth",
                    remedy="interpolate onto the configured depth levels, or update depths "
                    "in configs/base.yaml to match the source",
                )
            )

    return _result("depth_coordinates", "Depth coordinates", findings, details)


# --------------------------------------------------------------------------
# 5. Time coordinates
# --------------------------------------------------------------------------


def time_coordinates(dataset) -> CheckResult:
    """Time decoded, free of NaT, ascending, plausibly dated, regularly spaced.

    Duplicate timestamps are the separate `duplicate_timestamps` rule, so
    that a report distinguishes "unsorted" from "repeated".
    """
    values = dataset.coords.get("time")
    if values is None:
        return _skipped(
            "time_coordinates",
            "Time coordinates",
            "dataset has no time axis (static field or single snapshot)",
        )

    values = np.asarray(values)
    findings: List[Finding] = []
    details: Dict[str, Any] = {"n_steps": int(values.size)}

    if values.size == 0:
        return _result(
            "time_coordinates",
            "Time coordinates",
            [Finding(ERROR, "time coordinate is empty", target="time")],
            details,
        )

    if not np.issubdtype(values.dtype, np.datetime64):
        return _result(
            "time_coordinates",
            "Time coordinates",
            [
                Finding(
                    ERROR,
                    f"time is not decoded to datetime64 (dtype {values.dtype})",
                    target="time",
                    remedy="decode the source's CF time units (e.g. 'days since ...') "
                    "before validating; load_netcdf does this automatically",
                )
            ],
            details,
        )

    if np.isnat(values).any():
        findings.append(
            Finding(
                ERROR,
                f"{int(np.isnat(values).sum())} timestamp(s) could not be parsed (NaT)",
                target="time",
                remedy="fix or drop the unparseable timestamps at the source",
            )
        )
        values = values[~np.isnat(values)]
        if values.size == 0:
            return _result("time_coordinates", "Time coordinates", findings, details)

    details["observed_range"] = [str(values.min()), str(values.max())]

    low, high = PLAUSIBLE_TIME_RANGE
    if values.min() < low or values.max() > high:
        findings.append(
            Finding(
                ERROR,
                f"timestamps span {values.min()} to {values.max()}, outside the plausible "
                f"range {low} to {high}",
                target="time",
                remedy="the time axis was almost certainly decoded against the wrong epoch "
                "or unit — check its 'units' attribute",
            )
        )

    if values.size > 1:
        diffs = np.diff(values)
        if np.any(diffs < np.timedelta64(0, "ns")):
            findings.append(
                Finding(
                    ERROR,
                    "timestamps are not in ascending order",
                    target="time",
                    remedy="sort the dataset along time, reordering the data arrays with it",
                )
            )
        elif values.size > 2:
            steps = diffs.astype("timedelta64[s]").astype(np.int64)
            median = float(np.median(steps))
            details["median_step_days"] = round(median / 86400.0, 4)
            if median > 0 and not np.allclose(steps, median, rtol=0.2):
                findings.append(
                    Finding(
                        WARNING,
                        f"time steps are irregular: {steps.min() / 86400:.2f}-"
                        f"{steps.max() / 86400:.2f} days (median {median / 86400:.2f})",
                        target="time",
                        remedy="resample onto a regular cadence, or confirm the gaps are "
                        "expected for this product",
                    )
                )

    return _result("time_coordinates", "Time coordinates", findings, details)


# --------------------------------------------------------------------------
# 6. Variable existence
# --------------------------------------------------------------------------


def variable_existence(dataset, *, required: Optional[Sequence[str]] = None) -> CheckResult:
    """Required variables present; unknown variables flagged.

    `required` has no default: the configuration files do not declare a
    required-variable list, and inventing one here would fail datasets
    that are legitimately partial. Pass `rules.SURFACE_VARIABLES` (or your
    own list) to enforce a contract. With `required=None` the rule still
    errors on a dataset holding *no* recognized NEER variable at all,
    which almost always means the wrong file or the wrong variable names.
    """
    findings: List[Finding] = []
    present = list(dataset.variable_names)
    known = [name for name in present if lookup_spec(name) is not None]
    details: Dict[str, Any] = {
        "present": present,
        "recognized": known,
        "required": list(required) if required else [],
    }

    if not present:
        return _result(
            "variable_existence",
            "Variable existence",
            [
                Finding(
                    ERROR,
                    "dataset contains no variables",
                    remedy="check that the loader selected the right variables from the source",
                )
            ],
            details,
        )

    if required:
        absent = [name for name in required if name not in dataset]
        details["missing"] = absent
        if absent:
            findings.append(
                Finding(
                    ERROR,
                    f"required variable(s) absent: {absent}",
                    remedy="load them from the source, or drop them from the required list "
                    "if this dataset is not expected to carry them",
                )
            )
    elif not known:
        findings.append(
            Finding(
                ERROR,
                f"none of the variables {present} is a recognized NEER variable",
                remedy="rename them to NEER names (see schema.VARIABLE_ALIASES) or pass "
                "rename= to the loader; otherwise nothing downstream can interpret them",
            )
        )

    unknown = [name for name in present if lookup_spec(name) is None]
    if unknown:
        findings.append(
            Finding(
                WARNING,
                f"variable(s) not in the NEER registry: {unknown}",
                remedy="add them to schema.VARIABLE_SPECS so their units and physical "
                "ranges can be validated, or leave them as unchecked passengers",
            )
        )

    return _result("variable_existence", "Variable existence", findings, details)


# --------------------------------------------------------------------------
# 7. Units
# --------------------------------------------------------------------------


def units(dataset) -> CheckResult:
    """Units declared, recognized, and correct for the variable they label."""
    findings: List[Finding] = []
    details: Dict[str, Any] = {}

    for name in dataset.variable_names:
        variable = dataset[name]
        declared = variable.units
        details[name] = declared
        spec = lookup_spec(name)

        if declared is None:
            findings.append(
                Finding(
                    WARNING,
                    "no units declared",
                    target=name,
                    remedy="set the variable's units at the source, or pass units= to the "
                    "loader; unit-blind data cannot be checked or safely combined",
                )
            )
            continue

        canonical = normalize_unit(declared)
        if canonical is None:
            findings.append(
                Finding(
                    WARNING,
                    f"unrecognized unit {declared!r}",
                    target=name,
                    remedy="add the spelling to src/data/loaders/units.py if it is valid",
                )
            )
            continue

        if spec is None:
            continue

        expected = normalize_unit(spec.units) or spec.units
        if canonical != expected:
            findings.append(
                Finding(
                    ERROR,
                    f"unit {canonical!r} contradicts the expected unit {expected!r} for "
                    f"{spec.description.lower()}",
                    target=name,
                    remedy=f"convert the values to {expected!r} (the loader does this "
                    "automatically for known conversions such as K→degC), or correct the "
                    "variable name if it was mislabelled",
                )
            )

    return _result("units", "Units", findings, details)


# --------------------------------------------------------------------------
# 8. NaN values
# --------------------------------------------------------------------------


def nan_values(dataset, *, warn_fraction: float = DEFAULT_NAN_WARN_FRACTION) -> CheckResult:
    """Per-variable missing-data fractions, plus infinities and value ranges.

    Gaps are normal in ocean data, so a partially-missing field is not
    reported at all. A *fully* missing variable is an ERROR: it carries no
    information and usually means a bad read or a wrong fill value.
    Infinities are always an ERROR — NEER encodes missing data as NaN, so
    an Inf is a computation that went wrong upstream, not a gap.
    """
    findings: List[Finding] = []
    details: Dict[str, Any] = {}

    for name in dataset.variable_names:
        variable = dataset[name]
        values = variable.values
        if values.size == 0:
            findings.append(Finding(ERROR, "variable array is empty", target=name))
            continue

        fraction = missing_fraction(values)
        details[name] = round(fraction, 6)

        if fraction >= 1.0:
            findings.append(
                Finding(
                    ERROR,
                    "variable is entirely missing (100% NaN)",
                    target=name,
                    remedy="check the fill value, the variable name, and any subsetting "
                    "applied at read time",
                )
            )
        elif fraction > warn_fraction:
            findings.append(
                Finding(
                    WARNING,
                    f"{fraction:.1%} of values are missing",
                    target=name,
                    remedy="confirm this sparsity is expected for the product before using "
                    "the variable for training",
                )
            )

        if np.issubdtype(values.dtype, np.floating):
            n_inf = int(np.isinf(values).sum())
            if n_inf:
                findings.append(
                    Finding(
                        ERROR,
                        f"{n_inf} infinite value(s) present; missing data must be NaN",
                        target=name,
                        remedy="find the upstream computation producing Inf and replace "
                        "those entries with NaN",
                    )
                )

        spec = lookup_spec(name)
        if spec is not None and spec.valid_range is not None:
            observed = finite_range(values)
            if observed is not None:
                low, high = spec.valid_range
                if observed[0] < low or observed[1] > high:
                    findings.append(
                        Finding(
                            WARNING,
                            f"values [{observed[0]:g}, {observed[1]:g}] {spec.units} fall "
                            f"outside the plausible range [{low:g}, {high:g}]",
                            target=name,
                            remedy="check the source units and any scale_factor/add_offset; "
                            "a field left in Kelvin looks exactly like this",
                        )
                    )

    return _result("nan_values", "Missing (NaN) values", findings, details)


# --------------------------------------------------------------------------
# 9. Duplicate timestamps
# --------------------------------------------------------------------------


def duplicate_timestamps(dataset) -> CheckResult:
    """No timestamp may appear twice.

    Always an ERROR. A duplicated time step means two different states of
    the ocean claim the same moment, and silently keeping one of them
    would make every downstream time series wrong in a way that is nearly
    impossible to trace back.
    """
    values = dataset.coords.get("time")
    if values is None:
        return _skipped(
            "duplicate_timestamps", "Duplicate timestamps", "dataset has no time axis"
        )

    values = np.asarray(values)
    if values.size == 0:
        return _skipped(
            "duplicate_timestamps", "Duplicate timestamps", "time axis is empty"
        )

    unique, counts = np.unique(values, return_counts=True)
    repeated = unique[counts > 1]
    details = {
        "n_steps": int(values.size),
        "n_unique": int(unique.size),
        "n_duplicated": int(repeated.size),
    }

    findings: List[Finding] = []
    if repeated.size:
        shown = ", ".join(str(t) for t in repeated[:5])
        if repeated.size > 5:
            shown += f", ... (+{repeated.size - 5} more)"
        findings.append(
            Finding(
                ERROR,
                f"{repeated.size} timestamp(s) appear more than once: {shown}",
                target="time",
                remedy="de-duplicate at the source, deciding explicitly whether to keep "
                "the first, the last, or the mean of each duplicated step",
            )
        )

    return _result("duplicate_timestamps", "Duplicate timestamps", findings, details)


# --------------------------------------------------------------------------
# 10. Duplicate coordinates
# --------------------------------------------------------------------------


def duplicate_coordinates(dataset) -> CheckResult:
    """No spatial or depth coordinate may repeat a value.

    Always an ERROR, for the same reason as duplicate timestamps: two grid
    cells claiming the same position make any spatial operation ambiguous.
    """
    findings: List[Finding] = []
    details: Dict[str, Any] = {}

    axes = [axis for axis in CANONICAL_COORDS if axis != "time" and axis in dataset.coords]
    if not axes:
        return _skipped(
            "duplicate_coordinates",
            "Duplicate coordinates",
            "dataset has no spatial or depth coordinates",
        )

    for axis in axes:
        values = np.asarray(dataset.coords[axis], dtype=np.float64)
        finite = values[~np.isnan(values)]
        unique, counts = np.unique(finite, return_counts=True)
        repeated = unique[counts > 1]
        details[axis] = {"n": int(values.size), "n_unique": int(unique.size)}

        if repeated.size:
            shown = ", ".join(f"{v:g}" for v in repeated[:5])
            if repeated.size > 5:
                shown += f", ... (+{repeated.size - 5} more)"
            findings.append(
                Finding(
                    ERROR,
                    f"{repeated.size} repeated value(s) on the '{axis}' axis: {shown}",
                    target=axis,
                    remedy=f"de-duplicate the {axis} axis at the source; two cells cannot "
                    "share a coordinate",
                )
            )

    return _result("duplicate_coordinates", "Duplicate coordinates", findings, details)


#: Every rule, in report order, with the keyword arguments each accepts.
#: `validator.py` uses this to run the suite and to validate `checks=`.
RULES: Dict[str, Any] = {
    "latitude_range": latitude_range,
    "longitude_range": longitude_range,
    "resolution": resolution,
    "depth_coordinates": depth_coordinates,
    "time_coordinates": time_coordinates,
    "variable_existence": variable_existence,
    "units": units,
    "nan_values": nan_values,
    "duplicate_timestamps": duplicate_timestamps,
    "duplicate_coordinates": duplicate_coordinates,
}

ALL_RULES: Tuple[str, ...] = tuple(RULES)
