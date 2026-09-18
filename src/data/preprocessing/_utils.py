"""
Small array/dataset helpers shared by the preprocessing steps.

Nothing here is part of the public API. These exist so that every step
rebuilds an `OceanDataset` the same way — `OceanDataset` and `Variable`
are frozen dataclasses, so a step never edits data in place; it builds a
new object with `dataclasses.replace` and hands it on. That immutability
is what makes the pipeline safe to re-run and easy to test: a step can be
applied to the training slice and to the full dataset without the first
call contaminating the second.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from src.data.loaders.representation import OceanDataset, Variable

#: Variables that are masks/metadata rather than geophysical fields, and
#: are therefore never interpolated, filled, or normalized.
MASK_SUFFIX = "_observed"


# --------------------------------------------------------------------------
# NaN-aware reductions (without the RuntimeWarning noise)
# --------------------------------------------------------------------------


def nan_mean(values: np.ndarray, axis=None) -> np.ndarray:
    """`np.nanmean` that returns NaN for all-NaN slices, quietly.

    An all-NaN slice is normal here (a land cell, a depth level with no
    coverage in the training period), not an anomaly worth warning about
    on every call.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def nan_std(values: np.ndarray, axis=None) -> np.ndarray:
    """`np.nanstd` that returns NaN for all-NaN slices, quietly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(values, axis=axis)


def finite_count(values: np.ndarray) -> int:
    """Number of finite (non-missing) entries in a float array."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        return int(array.size)
    return int(np.isfinite(array).sum())


# --------------------------------------------------------------------------
# Variable / dataset rebuilding
# --------------------------------------------------------------------------


def is_mask_variable(name: str, variable: Variable) -> bool:
    """True for boolean masks and the `<var>_observed` companions."""
    return variable.is_mask or name.endswith(MASK_SUFFIX) or variable.dtype == bool


def data_variables(
    dataset: OceanDataset, *, include_masks: bool = False
) -> Dict[str, Variable]:
    """The geophysical variables of `dataset` (masks excluded by default)."""
    return {
        name: variable
        for name, variable in dataset.variables.items()
        if include_masks or not is_mask_variable(name, variable)
    }


def with_values(variable: Variable, values: np.ndarray, **changes: Any) -> Variable:
    """A copy of `variable` carrying new `values` (and optionally new dims/attrs)."""
    return replace(variable, values=np.asarray(values), **changes)


def with_variables(
    dataset: OceanDataset,
    variables: Mapping[str, Variable],
    *,
    coords: Optional[Mapping[str, np.ndarray]] = None,
    attrs: Optional[Mapping[str, Any]] = None,
) -> OceanDataset:
    """A copy of `dataset` with these variables/coords/attrs.

    The validation report is dropped deliberately: it described the
    dataset as it was loaded, and preprocessing has since changed it.
    Re-validate if you need a current one.
    """
    return replace(
        dataset,
        variables=dict(variables),
        coords=dict(coords) if coords is not None else dict(dataset.coords),
        attrs={**dataset.attrs, **(attrs or {})},
        validation=None,
    )


def axis_of(variable: Variable, dim: str) -> Optional[int]:
    """Index of `dim` in `variable.dims`, or None if it has no such axis."""
    return variable.dims.index(dim) if dim in variable.dims else None


def take_along_dim(variable: Variable, dim: str, indices: np.ndarray) -> Variable:
    """Select `indices` along `dim`; returns `variable` unchanged if it has no `dim`."""
    axis = axis_of(variable, dim)
    if axis is None:
        return variable
    return with_values(variable, np.take(variable.values, np.asarray(indices), axis=axis))


def select_along_dim(
    dataset: OceanDataset, dim: str, indices: np.ndarray
) -> OceanDataset:
    """Subset/reorder every variable and the coordinate along `dim`.

    Used for chronological splitting (`dim="time"`) and for sorting the
    latitude/longitude axes into ascending order.
    """
    indices = np.asarray(indices)
    variables = {
        name: take_along_dim(variable, dim, indices)
        for name, variable in dataset.variables.items()
    }
    coords = dict(dataset.coords)
    if dim in coords:
        coords[dim] = coords[dim][indices]
    return with_variables(dataset, variables, coords=coords)


def select_times(dataset: OceanDataset, mask: np.ndarray) -> OceanDataset:
    """The dataset restricted to the timesteps where `mask` is True."""
    mask = np.asarray(mask, dtype=bool)
    time = dataset.coords.get("time")
    if time is None:
        raise ValueError("dataset has no 'time' coordinate to select on")
    if mask.shape != time.shape:
        raise ValueError(
            f"time mask has shape {mask.shape} but the time axis has shape {time.shape}"
        )
    return select_along_dim(dataset, "time", np.flatnonzero(mask))


def broadcast_spatial(
    values: np.ndarray, target_dims: Sequence[str], source_dims: Sequence[str]
) -> np.ndarray:
    """Broadcast an array from `source_dims` up to `target_dims`.

    `source_dims` must be a subset of `target_dims` in the same relative
    order — which canonical dim ordering guarantees. Used to apply a
    `(lat, lon)` land mask to a `(time, depth, lat, lon)` field.
    """
    source_dims = tuple(source_dims)
    target_dims = tuple(target_dims)
    missing = [d for d in source_dims if d not in target_dims]
    if missing:
        raise ValueError(f"cannot broadcast dims {missing} into {target_dims}")
    shape = [
        values.shape[source_dims.index(dim)] if dim in source_dims else 1
        for dim in target_dims
    ]
    return np.asarray(values).reshape(shape)


def ocean_mask_of(dataset: OceanDataset, mask_variable: str = "land_mask") -> Optional[np.ndarray]:
    """The `(lat, lon)` ocean mask (True = ocean), or None if absent.

    NEER's `land_mask` is True over ocean — the name comes from the
    source products, the semantics come from `schema.VARIABLE_SPECS`.
    """
    variable = dataset.variables.get(mask_variable)
    if variable is None:
        return None
    return np.asarray(variable.values, dtype=bool)


def month_of(times: np.ndarray) -> np.ndarray:
    """Calendar month (1-12) for each entry of a datetime64 array."""
    times = np.asarray(times)
    return times.astype("datetime64[M]").astype(int) % 12 + 1


def day_of_year(times: np.ndarray) -> np.ndarray:
    """Day of year (1-366) for each entry of a datetime64 array."""
    times = np.asarray(times).astype("datetime64[D]")
    year_start = times.astype("datetime64[Y]").astype("datetime64[D]")
    return (times - year_start).astype(int) + 1


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars/arrays/datetimes into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def summarize_array(values: np.ndarray, *, decimals: int = 6) -> Dict[str, Any]:
    """Compact, JSON-safe statistics for an array (for metadata, not data)."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": int(array.size), "n_finite": 0, "min": None, "max": None, "mean": None}
    return {
        "n": int(array.size),
        "n_finite": int(finite.size),
        "min": round(float(finite.min()), decimals),
        "max": round(float(finite.max()), decimals),
        "mean": round(float(finite.mean()), decimals),
    }


def unique_preserving_order(names: Iterable[str]) -> list:
    """De-duplicate while keeping first-seen order."""
    return list(dict.fromkeys(names))
