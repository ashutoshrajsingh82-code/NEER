"""
The shared standardization pipeline every NEER loader runs.

`load_netcdf`, `load_csv`, `load_npz` and `load_demo_dataset` differ only
in how they get bytes off disk. Once the raw arrays exist, all four take
exactly the same path through this module:

    raw array
      -> canonical variable name      (schema.canonical_variable_name)
      -> canonical dim names & order  (schema.transpose_to_canonical)
      -> fill values replaced by NaN  (missing.replace_fill_values)
      -> units converted to canonical (units.convert_to_canonical)
      -> Variable
    ... then, for the dataset as a whole:
      -> time coordinate as datetime64[ns]
      -> validation                   (validation.validate_dataset)

Keeping this in one place is what makes the loaders interchangeable: a
CSV of ARGO-style point observations and a gridded NetCDF product end up
as the same kind of object, with the same guarantees, or they fail for
the same reasons.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.loaders.missing import replace_fill_values
from src.data.loaders.representation import OceanDataset, Variable
from src.data.loaders.schema import (
    canonical_dims,
    canonical_variable_name,
    lookup_spec,
    transpose_to_canonical,
)
from src.data.loaders.units import convert_to_canonical, normalize_unit
from src.data.loaders.validation import DEFAULT_MISSING_WARN_FRACTION, validate_dataset


def coerce_time_values(values: Any) -> np.ndarray:
    """Coerce assorted time representations into `datetime64[ns]`.

    Handles what the loaders actually run into: already-decoded
    `datetime64`, ISO date strings (the demo dataset stores
    `"2020-01-15"`), Python `datetime` objects, and pandas timestamps.
    Timezone-aware input is normalized to UTC and stored naive, so that
    every timestamp in NEER is comparable without carrying tzinfo around.

    Values that cannot be parsed become `NaT` rather than raising here —
    `validation.check_timestamps` reports them with full context.
    """
    array = np.asarray(values)

    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ns]")

    # Numeric times (e.g. "days since 1950-01-01") cannot be decoded without
    # their units attribute; hand them back untouched so validation can
    # report the problem instead of inventing an epoch.
    if np.issubdtype(array.dtype, np.number):
        return array

    index = pd.to_datetime(pd.Index(array.ravel()), errors="coerce", utc=True)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.to_numpy(dtype="datetime64[ns]").reshape(array.shape)


def standardize_variable(
    name: str,
    values: np.ndarray,
    dims: Sequence[str],
    *,
    units: Optional[str] = None,
    attrs: Optional[Mapping[str, Any]] = None,
    standardize_names: bool = True,
    replace_sentinels: bool = True,
    normalize_units: bool = True,
    extra_fill_values: Optional[Sequence[float]] = None,
) -> Variable:
    """Turn one raw array into a canonical `Variable`.

    Every transformation that changes the data records what it did in the
    variable's `attrs`, so a loaded dataset can always explain itself:
    `source_name`, `source_units`, `n_fill_values_replaced`.
    """
    attrs = dict(attrs or {})
    source_name = str(name)
    canonical_name = canonical_variable_name(name) if standardize_names else source_name
    if canonical_name != source_name:
        attrs["source_name"] = source_name

    dims = canonical_dims(dims)
    array = np.asarray(values)
    array, dims = transpose_to_canonical(array, dims)

    if replace_sentinels:
        array, n_replaced = replace_fill_values(
            array, attrs=attrs, extra_values=extra_fill_values
        )
        if n_replaced:
            attrs["n_fill_values_replaced"] = n_replaced
    # The fill values have been applied; keeping the attributes would
    # invite a second, wrong application downstream.
    for key in ("_FillValue", "missing_value", "fill_value"):
        attrs.pop(key, None)

    resulting_units = normalize_unit(units) or units
    if normalize_units:
        spec = lookup_spec(canonical_name)
        canonical_unit = spec.units if spec else None
        array, resulting_units, converted = convert_to_canonical(array, units, canonical_unit)
        if converted:
            attrs["source_units"] = units

    return Variable(
        name=canonical_name,
        values=array,
        dims=dims,
        units=resulting_units,
        attrs=attrs,
    )


def standardize_coords(coords: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Canonicalize coordinate arrays (time decoded, everything else as-is)."""
    from src.data.loaders.schema import canonical_coord_name

    out: Dict[str, np.ndarray] = {}
    for name, values in coords.items():
        canonical = canonical_coord_name(name)
        array = np.asarray(values)
        if canonical == "time":
            array = coerce_time_values(array)
        out[canonical] = array
    return out


def finalize_dataset(
    dataset: OceanDataset,
    *,
    validate: bool = True,
    strict: bool = False,
    domain=None,
    check_domain: bool = True,
    missing_warn_fraction: float = DEFAULT_MISSING_WARN_FRACTION,
) -> OceanDataset:
    """Validate a freshly-built dataset and attach the report.

    With `validate=False` the dataset is returned untouched and carries no
    report — useful for inspecting a file that is known to be broken.
    """
    if not validate:
        return dataset

    report = validate_dataset(
        dataset,
        domain=domain,
        check_domain=check_domain,
        missing_warn_fraction=missing_warn_fraction,
    )
    report.raise_for_status(strict=strict)
    return dataset.with_validation(report)


def select_variables(
    available: Sequence[str], requested: Optional[Sequence[str]], *, source: str
) -> list:
    """Resolve a caller's `variables=` selection against what the source holds.

    Accepts either NEER names or the source's own names, so
    `variables=["sst"]` works whether the file calls it `sst` or
    `analysed_sst`. Raises KeyError listing what is actually available.
    """
    if requested is None:
        return list(available)

    by_canonical: Dict[str, str] = {}
    for name in available:
        by_canonical.setdefault(canonical_variable_name(name), name)
        by_canonical.setdefault(str(name), str(name))

    resolved = []
    missing = []
    for name in requested:
        key = str(name)
        if key in by_canonical:
            resolved.append(by_canonical[key])
        elif canonical_variable_name(key) in by_canonical:
            resolved.append(by_canonical[canonical_variable_name(key)])
        else:
            missing.append(key)

    if missing:
        raise KeyError(
            f"variable(s) {missing} not found in {source}; available: {sorted(available)}"
        )
    # De-duplicate while preserving the caller's order.
    return list(dict.fromkeys(resolved))
