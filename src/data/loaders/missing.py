"""
Missing-value handling for the NEER data loading layer.

NEER's internal convention is simple and absolute: **missing data is
`NaN` in a floating-point array.** Nothing downstream should ever have to
remember that a particular product encodes gaps as `-999`, `-9999`,
`1e20`, or `9.96921e36`, or that an integer field carries a `_FillValue`
attribute.

This module converts every one of those encodings into NaN at load time,
and provides the mask/statistics helpers the rest of the layer uses.

Why sentinels are matched with a tolerance
------------------------------------------
Sentinels routinely survive a float32 round-trip with a tiny error (the
classic `9.96921e+36` NetCDF default, or `-9999.0` stored as float32 and
read back as `-9998.999...`). An exact `==` comparison misses those, so
matching is done with `np.isclose`. The tolerance is relative, so it
cannot swallow legitimate physical values that merely sit near zero.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

#: Sentinel values used by common ocean/atmosphere products to mean "no data".
#: Deliberately conservative: every entry here is far outside the physical
#: range of any variable in `schema.VARIABLE_SPECS`, so replacing them
#: cannot destroy real measurements.
DEFAULT_SENTINEL_VALUES: Tuple[float, ...] = (
    -999.0,
    -999.9,
    -9999.0,
    -99999.0,
    -32767.0,
    -32768.0,
    9.96921e36,
    1e20,
    -1e20,
    1e30,
    -1e30,
)

#: Attribute names that carry a per-variable fill value in NetCDF/HDF sources.
FILL_VALUE_ATTRS: Tuple[str, ...] = ("_FillValue", "missing_value", "fill_value")

#: Strings that pandas should treat as missing on top of its own defaults.
CSV_NA_STRINGS: Tuple[str, ...] = (
    "NA",
    "N/A",
    "n/a",
    "NaN",
    "nan",
    "NULL",
    "null",
    "None",
    "none",
    "-",
    "--",
    "?",
    "missing",
    "MISSING",
    "-999",
    "-999.0",
    "-9999",
    "-9999.0",
    "99999",
    "-99999",
)


def _as_float_array(values: np.ndarray) -> np.ndarray:
    """Return a float array that can hold NaN, copying only when needed."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        return array.copy()
    # Integers (and anything else numeric) cannot represent NaN; widen to
    # float32 for small ints so a fill-value replacement is possible.
    if array.dtype.itemsize <= 2:
        return array.astype(np.float32)
    return array.astype(np.float64)


def replace_fill_values(
    values: np.ndarray,
    *,
    attrs: Optional[dict] = None,
    extra_values: Optional[Iterable[float]] = None,
    use_default_sentinels: bool = True,
    rtol: float = 1e-5,
) -> Tuple[np.ndarray, int]:
    """Replace fill/sentinel values in `values` with NaN.

    Parameters
    ----------
    values:
        The raw array as read from the source.
    attrs:
        The variable's source attributes; any of `FILL_VALUE_ATTRS` found
        there is treated as a fill value for this variable specifically.
    extra_values:
        Additional caller-supplied sentinels.
    use_default_sentinels:
        Whether to also match `DEFAULT_SENTINEL_VALUES`. Turn this off for
        sources known to use a value in that list legitimately.

    Returns
    -------
    (array, n_replaced)
        A float array with NaN in place of every fill value, and how many
        elements were replaced. Boolean and datetime arrays are returned
        untouched with a count of 0 — NaN is meaningless for those.
    """
    array = np.asarray(values)
    if array.dtype == bool or np.issubdtype(array.dtype, np.datetime64):
        return array, 0
    if not np.issubdtype(array.dtype, np.number):
        return array, 0

    sentinels: list = []
    for key in FILL_VALUE_ATTRS:
        if attrs and key in attrs:
            candidate = attrs[key]
            candidate = np.asarray(candidate).ravel()
            sentinels.extend(float(c) for c in candidate if np.isfinite(float(c)))
    if extra_values:
        sentinels.extend(float(v) for v in extra_values)
    if use_default_sentinels:
        sentinels.extend(DEFAULT_SENTINEL_VALUES)

    if not sentinels:
        return array, 0

    result = _as_float_array(array)
    mask = np.zeros(result.shape, dtype=bool)
    for sentinel in dict.fromkeys(sentinels):  # de-duplicate, keep order
        mask |= np.isclose(result, sentinel, rtol=rtol, atol=0.0, equal_nan=False)

    n_replaced = int(mask.sum())
    if n_replaced:
        result[mask] = np.nan
        return result, n_replaced

    # Nothing to replace: hand back the original array (or its widened
    # copy, which is already detached from the caller's memory).
    return (result if result.dtype != array.dtype else array), 0


def missing_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask, True where `values` is missing.

    Missing means NaN for floats and NaT for datetimes. Booleans and
    integers have no missing representation, so the mask is all-False.
    """
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        return np.isnan(array)
    if np.issubdtype(array.dtype, np.datetime64):
        return np.isnat(array)
    if array.dtype == object:
        return np.array([v is None for v in array.ravel()]).reshape(array.shape)
    return np.zeros(array.shape, dtype=bool)


def missing_fraction(values: np.ndarray) -> float:
    """Fraction of `values` that is missing, in [0, 1]. Empty arrays give 0.0."""
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(missing_mask(array).mean())


def finite_range(values: np.ndarray) -> Optional[Tuple[float, float]]:
    """(min, max) over the non-missing entries, or None if everything is missing."""
    array = np.asarray(values)
    if array.size == 0:
        return None
    if array.dtype == bool:
        return (float(array.min()), float(array.max()))
    if not np.issubdtype(array.dtype, np.number):
        return None
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return (float(finite.min()), float(finite.max()))
