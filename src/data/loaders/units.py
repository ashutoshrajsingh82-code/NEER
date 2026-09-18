"""
Unit normalization and conversion for the NEER data loading layer.

Ocean products spell the same unit a dozen ways (`degC`, `deg_C`,
`degrees_Celsius`, `Celsius`) and occasionally use a different one
entirely (Kelvin for SST, cm for sea level, cm/s for currents). Loaders
run every variable through here so that downstream code can assume the
canonical unit from `schema.VARIABLE_SPECS` without re-checking.

Two separate operations live here, and the distinction matters:

* `normalize_unit` only rewrites the unit *string* (`"deg_C"` -> `"degC"`).
  Values are untouched.
* `convert_to_canonical` rewrites the *values* when the source unit is a
  known, convertible alternative (Kelvin -> degC). It never guesses: an
  unknown unit is left alone and reported, not silently assumed correct.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

#: Canonical spellings, keyed by a normalized lookup form (lowercase with
#: spaces/underscores/dots/hyphens removed — see `_normalize_key`).
_UNIT_ALIASES: Dict[str, str] = {
    # temperature
    "degc": "degC",
    "c": "degC",
    "celsius": "degC",
    "degreec": "degC",
    "degreesc": "degC",
    "degreecelsius": "degC",
    "degreescelsius": "degC",
    "k": "K",
    "kelvin": "K",
    "degk": "K",
    "degreekelvin": "K",
    "degf": "degF",
    "fahrenheit": "degF",
    "degreefahrenheit": "degF",
    # salinity
    "psu": "psu",
    "pss": "psu",
    "pss78": "psu",
    "practicalsalinity": "psu",
    "practicalsalinityunit": "psu",
    "practicalsalinityunits": "psu",
    "1e3": "psu",  # CF convention writes practical salinity as 1e-3
    "gkg": "psu",
    "g/kg": "psu",
    # length / depth / sea level
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "dbar": "dbar",
    "decibar": "dbar",
    "decibars": "dbar",
    # speed
    "m/s": "m/s",
    "ms1": "m/s",
    "ms-1": "m/s",
    "meterpersecond": "m/s",
    "meterspersecond": "m/s",
    "metrepersecond": "m/s",
    "metrespersecond": "m/s",
    "cm/s": "cm/s",
    "cms1": "cm/s",
    "cms-1": "cm/s",
    "centimeterpersecond": "cm/s",
    "centimeterspersecond": "cm/s",
    "knot": "knots",
    "knots": "knots",
    "kt": "knots",
    # dimensionless / masks
    "bool": "bool",
    "boolean": "bool",
    "flag": "bool",
    "mask": "bool",
    "1": "1",
    "none": "1",
    "dimensionless": "1",
    "unitless": "1",
    "": "1",
}


def _normalize_key(unit: str) -> str:
    """Reduce a unit string to a lookup key: lowercase, no spaces/underscores/dots."""
    key = str(unit).strip().lower()
    key = key.replace("**", "")
    key = re.sub(r"[\s_.·*]", "", key)
    return key


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """Return the canonical spelling of `unit`, or None if it is unrecognized.

    `None` and the empty string both mean "no unit declared" and return
    None, so callers can distinguish "not declared" (None in, None out)
    from "declared but unknown" (string in, None out) by checking the
    input themselves.
    """
    if unit is None:
        return None
    key = _normalize_key(unit)
    if key == "":
        return None
    return _UNIT_ALIASES.get(key)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

#: (from_unit, to_unit) -> elementwise conversion. Only conversions that are
#: exact or standard live here; anything requiring an assumption about the
#: data (dbar -> m, which depends on latitude and density) is handled
#: separately and reported.
_CONVERSIONS: Dict[Tuple[str, str], Callable[[np.ndarray], np.ndarray]] = {
    ("K", "degC"): lambda v: v - 273.15,
    ("degC", "K"): lambda v: v + 273.15,
    ("degF", "degC"): lambda v: (v - 32.0) * 5.0 / 9.0,
    ("cm", "m"): lambda v: v / 100.0,
    ("mm", "m"): lambda v: v / 1000.0,
    ("km", "m"): lambda v: v * 1000.0,
    ("m", "cm"): lambda v: v * 100.0,
    ("cm/s", "m/s"): lambda v: v / 100.0,
    ("knots", "m/s"): lambda v: v * 0.514444,
    ("m/s", "cm/s"): lambda v: v * 100.0,
    # Approximate: 1 dbar of pressure corresponds to ~1 m of depth in
    # seawater (exact conversion depends on latitude and density). Used
    # for depth coordinates given as pressure; flagged by the caller.
    ("dbar", "m"): lambda v: v * 1.0,
}

#: Conversions that are approximations rather than exact identities.
APPROXIMATE_CONVERSIONS = frozenset({("dbar", "m")})


def can_convert(from_unit: Optional[str], to_unit: Optional[str]) -> bool:
    """True if values in `from_unit` can be converted to `to_unit`."""
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    if source is None or target is None:
        return False
    return source == target or (source, target) in _CONVERSIONS


def convert(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert `values` from one unit to another.

    Raises ValueError if no conversion is defined — callers that prefer to
    carry on with the original unit should check `can_convert` first.
    """
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    if source is None:
        raise ValueError(f"unrecognized source unit: {from_unit!r}")
    if target is None:
        raise ValueError(f"unrecognized target unit: {to_unit!r}")
    if source == target:
        return values
    try:
        conversion = _CONVERSIONS[(source, target)]
    except KeyError:
        raise ValueError(f"no known conversion from {source!r} to {target!r}") from None

    array = np.asarray(values)
    converted = conversion(array.astype(np.float64))
    # Preserve float32 storage (the project's on-disk precision) rather
    # than silently doubling memory on every conversion.
    if array.dtype == np.float32:
        return converted.astype(np.float32)
    return converted


def convert_to_canonical(
    values: np.ndarray, source_unit: Optional[str], canonical_unit: Optional[str]
) -> Tuple[np.ndarray, Optional[str], bool]:
    """Best-effort conversion of `values` into `canonical_unit`.

    Returns `(values, resulting_unit, converted)`:

    * If no canonical unit is expected, or the source unit is missing or
      unrecognized, the values are returned untouched with the normalized
      (or original) source unit and `converted=False`.
    * If the source already is the canonical unit, the values are returned
      untouched with `converted=False`.
    * Otherwise the values are converted and `converted=True`.

    Unconvertible-but-known mismatches (psu declared where m is expected)
    are left alone here; `validation.check_units` is what reports them.
    """
    if canonical_unit is None:
        return values, normalize_unit(source_unit) or source_unit, False

    normalized_source = normalize_unit(source_unit)
    normalized_target = normalize_unit(canonical_unit) or canonical_unit

    if normalized_source is None:
        return values, source_unit, False
    if normalized_source == normalized_target:
        return values, normalized_target, False
    if not can_convert(normalized_source, normalized_target):
        return values, normalized_source, False

    return convert(values, normalized_source, normalized_target), normalized_target, True
