"""
CSV loading for NEER (`load_csv`).

CSV is how point/profile observations usually reach a project like this:
one row per observation, carrying its own time and position, with one
column per measured variable — the shape an ARGO profile export, a
mooring record, or a hand-built QC table arrives in.

    time,lat,lon,depth,temperature,salinity
    2020-01-15,12.5,72.0,0,28.4,35.9
    2020-01-15,12.5,72.0,50,26.1,35.7

Reading that is the easy part. The real work, and the reason this loader
is more than a `pd.read_csv` wrapper, is turning irregular rows into the
same gridded representation the NetCDF path produces:

* coordinate axes are derived from the values actually present (or snapped
  onto the project grid with `grid="domain"`),
* every (time, depth, lat, lon) cell with no row becomes NaN — an
  unobserved cell and a failed measurement are both simply missing,
* duplicate rows for one cell are resolved explicitly (`aggregate=`)
  rather than silently taking whichever pandas saw last.

Wide/gridded CSVs (a matrix of longitudes across the header) are not
supported: they cannot express depth or time without one file per slice,
and NetCDF is the right format for that case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.loaders._common import (
    coerce_time_values,
    finalize_dataset,
    standardize_variable,
)
from src.data.loaders.errors import SchemaError
from src.data.loaders.missing import CSV_NA_STRINGS
from src.data.loaders.representation import OceanDataset
from src.data.loaders.schema import CANONICAL_DIM_ORDER, canonical_coord_name
from src.data.loaders.validation import DEFAULT_MISSING_WARN_FRACTION

PathLike = Union[str, Path]

CSV_SUFFIXES = (".csv", ".txt", ".tsv", ".dat")

#: How duplicate rows for the same (time, depth, lat, lon) cell are resolved.
AGGREGATIONS = ("mean", "median", "first", "last", "min", "max", "error")

#: Tolerance (in degrees) when snapping observation positions onto the
#: project grid with `grid="domain"`. Half a grid cell at 0.25° resolution.
DEFAULT_SNAP_TOLERANCE = 0.125


def _find_column(columns: Sequence[str], target: str) -> Optional[str]:
    """Find the column whose name canonicalizes to `target` (e.g. 'lat')."""
    for column in columns:
        if canonical_coord_name(column) == target:
            return column
    return None


def _resolve_coord_columns(
    columns: Sequence[str],
    *,
    time_column: Optional[str],
    lat_column: Optional[str],
    lon_column: Optional[str],
    depth_column: Optional[str],
) -> Dict[str, str]:
    """Map canonical coordinate names onto actual CSV column names."""
    explicit = {
        "time": time_column,
        "lat": lat_column,
        "lon": lon_column,
        "depth": depth_column,
    }

    resolved: Dict[str, str] = {}
    for canonical, given in explicit.items():
        if given is not None:
            if given not in columns:
                raise SchemaError(
                    f"column {given!r} (given as the {canonical} column) is not in the CSV; "
                    f"columns are {list(columns)}"
                )
            resolved[canonical] = given
            continue
        found = _find_column(columns, canonical)
        if found is not None:
            resolved[canonical] = found

    for required in ("lat", "lon"):
        if required not in resolved:
            raise SchemaError(
                f"CSV must have a {required} column (any of the accepted aliases, or pass "
                f"{required}_column=...); columns are {list(columns)}"
            )
    return resolved


def _axis_values(series: pd.Series, canonical: str) -> np.ndarray:
    """The sorted, unique axis values for one coordinate column."""
    values = series.dropna().unique()
    if canonical == "time":
        values = coerce_time_values(values)
        return np.sort(np.asarray(values))
    return np.sort(np.asarray(values, dtype=np.float64))


def _snap_to_grid(values: np.ndarray, axis: np.ndarray, tolerance: float) -> np.ndarray:
    """Snap observation coordinates onto `axis`, returning axis indices.

    Values further than `tolerance` from every axis point get index -1 and
    are dropped by the caller (with a count reported in the dataset attrs).
    """
    if axis.size == 0:
        return np.full(values.shape, -1, dtype=np.int64)
    positions = np.searchsorted(axis, values)
    positions = np.clip(positions, 1, axis.size - 1) if axis.size > 1 else np.zeros_like(positions)
    left = axis[np.maximum(positions - 1, 0)]
    right = axis[np.minimum(positions, axis.size - 1)]
    choose_right = np.abs(values - right) < np.abs(values - left)
    indices = np.where(choose_right, np.minimum(positions, axis.size - 1), np.maximum(positions - 1, 0))
    nearest = axis[indices]
    too_far = np.abs(values - nearest) > tolerance
    indices = np.where(too_far, -1, indices)
    return indices.astype(np.int64)


def _project_grid_axes() -> Tuple[np.ndarray, np.ndarray]:
    """The project's lat/lon axes from `configs/base.yaml` (via src.data.grid)."""
    from src.data.grid import create_latitude_grid, create_longitude_grid

    return create_latitude_grid(), create_longitude_grid()


def load_csv(
    path: PathLike,
    *,
    variables: Optional[Sequence[str]] = None,
    time_column: Optional[str] = None,
    lat_column: Optional[str] = None,
    lon_column: Optional[str] = None,
    depth_column: Optional[str] = None,
    units: Optional[Mapping[str, str]] = None,
    units_row: bool = False,
    grid: str = "observed",
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE,
    aggregate: str = "mean",
    rename: Optional[Mapping[str, str]] = None,
    standardize_names: bool = True,
    normalize_units: bool = True,
    replace_sentinels: bool = True,
    attrs: Optional[Mapping[str, Any]] = None,
    validate: bool = True,
    strict: bool = False,
    check_domain: bool = True,
    domain: Any = None,
    missing_warn_fraction: float = DEFAULT_MISSING_WARN_FRACTION,
    read_csv_kwargs: Optional[Mapping[str, Any]] = None,
) -> OceanDataset:
    """Load a long-format (one row per observation) CSV into an `OceanDataset`.

    Parameters
    ----------
    path:
        Path to the CSV/TSV file. Tab-separated files (`.tsv`) are detected
        by extension.
    variables:
        Which value columns to load. Defaults to every non-coordinate column.
    time_column / lat_column / lon_column / depth_column:
        Explicit column names. Omit to auto-detect via
        `schema.COORD_ALIASES` (`latitude`, `LAT`, `y`, ... all work).
        lat and lon are required; time and depth are optional, and a file
        without them yields a dataset without that dimension.
    units:
        `{column: unit}` for the value columns. CSVs rarely carry units, so
        supplying them here is what lets the loader convert Kelvin to
        Celsius and lets validation check for unit mismatches.
    units_row:
        Set True when the first data row holds units rather than data.
    grid:
        `"observed"` (default) builds axes from the coordinate values
        present in the file. `"domain"` snaps observations onto the
        project's 0.25° grid from `configs/base.yaml`, producing a full
        grid with NaN wherever nothing was observed — which is what
        gridded-input models need.
    snap_tolerance:
        Maximum distance (degrees) an observation may be from a grid point
        when `grid="domain"`. Further-away rows are dropped and counted in
        `attrs["n_rows_off_grid"]`.
    aggregate:
        How to resolve several rows landing in the same cell: one of
        `mean`, `median`, `first`, `last`, `min`, `max`, or `error` to
        raise instead.

    Returns
    -------
    OceanDataset

    Raises
    ------
    FileNotFoundError, SchemaError, DataValidationError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {list(AGGREGATIONS)}, got {aggregate!r}")
    if grid not in ("observed", "domain"):
        raise ValueError(f"grid must be 'observed' or 'domain', got {grid!r}")

    read_kwargs: Dict[str, Any] = dict(read_csv_kwargs or {})
    read_kwargs.setdefault("na_values", list(CSV_NA_STRINGS))
    read_kwargs.setdefault("keep_default_na", True)
    if path.suffix.lower() == ".tsv":
        read_kwargs.setdefault("sep", "\t")

    column_units: Dict[str, str] = dict(units or {})

    if units_row:
        # Read the units line separately, then skip it so pandas can infer
        # numeric dtypes from the real data instead of seeing strings.
        header = pd.read_csv(path, nrows=1, **read_kwargs)
        column_units = {
            **{c: str(header.iloc[0][c]) for c in header.columns if pd.notna(header.iloc[0][c])},
            **column_units,  # explicit units= wins over the file's row
        }
        read_kwargs["skiprows"] = [1]

    frame = pd.read_csv(path, **read_kwargs)

    if frame.empty:
        raise SchemaError(f"{path} contains no data rows")

    if rename:
        frame = frame.rename(columns=dict(rename))
        column_units = {dict(rename).get(k, k): v for k, v in column_units.items()}

    columns = [str(c) for c in frame.columns]
    frame.columns = columns

    coord_columns = _resolve_coord_columns(
        columns,
        time_column=time_column,
        lat_column=lat_column,
        lon_column=lon_column,
        depth_column=depth_column,
    )
    value_columns = [c for c in columns if c not in set(coord_columns.values())]
    if variables is not None:
        requested = [str(v) for v in variables]
        unknown = [v for v in requested if v not in value_columns]
        if unknown:
            raise KeyError(
                f"value column(s) {unknown} not found in {path}; available: {value_columns}"
            )
        value_columns = requested
    if not value_columns:
        raise SchemaError(
            f"{path} has no value columns — every column was interpreted as a coordinate "
            f"({sorted(coord_columns.values())})"
        )

    dataset_attrs: Dict[str, Any] = dict(attrs or {})
    n_rows_read = len(frame)

    # -- coordinates -------------------------------------------------------

    if "time" in coord_columns:
        frame[coord_columns["time"]] = coerce_time_values(frame[coord_columns["time"]].to_numpy())

    before = len(frame)
    frame = frame.dropna(subset=list(coord_columns.values()))
    n_dropped_coords = before - len(frame)
    if n_dropped_coords:
        dataset_attrs["n_rows_dropped_missing_coords"] = int(n_dropped_coords)
    if frame.empty:
        raise SchemaError(f"{path} has no rows with complete coordinates")

    axes: Dict[str, np.ndarray] = {}
    indices: Dict[str, np.ndarray] = {}

    if grid == "domain":
        lat_axis, lon_axis = _project_grid_axes()
        axes["lat"], axes["lon"] = lat_axis, lon_axis
    for canonical, column in coord_columns.items():
        if canonical in axes:
            continue
        axes[canonical] = _axis_values(frame[column], canonical)

    n_off_grid = 0
    for canonical, column in coord_columns.items():
        axis = axes[canonical]
        raw_values = frame[column].to_numpy()
        if canonical == "time":
            index_map = {value: i for i, value in enumerate(axis)}
            positions = np.array([index_map.get(v, -1) for v in raw_values], dtype=np.int64)
        elif grid == "domain" and canonical in ("lat", "lon"):
            positions = _snap_to_grid(
                raw_values.astype(np.float64), axis, snap_tolerance
            )
        else:
            index_map = {float(value): i for i, value in enumerate(axis)}
            positions = np.array(
                [index_map.get(float(v), -1) for v in raw_values], dtype=np.int64
            )
        indices[canonical] = positions
        n_off_grid = max(n_off_grid, int((positions < 0).sum()))

    keep = np.ones(len(frame), dtype=bool)
    for positions in indices.values():
        keep &= positions >= 0
    if not keep.all():
        dropped = int((~keep).sum())
        dataset_attrs["n_rows_off_grid"] = dropped
        frame = frame.loc[keep]
        indices = {k: v[keep] for k, v in indices.items()}
    if frame.empty:
        raise SchemaError(
            f"{path}: no rows fell within the target grid "
            f"(snap tolerance {snap_tolerance}°). Check the coordinate columns and `grid=`."
        )

    dims = tuple(d for d in CANONICAL_DIM_ORDER if d in axes)
    shape = tuple(int(axes[d].size) for d in dims)

    # Flat cell index per row, so duplicates can be grouped cheaply.
    flat_index = np.zeros(len(frame), dtype=np.int64)
    for dim in dims:
        flat_index = flat_index * int(axes[dim].size) + indices[dim]

    n_cells = int(np.prod(shape)) if shape else 1
    n_duplicates = int(len(flat_index) - np.unique(flat_index).size)
    if n_duplicates:
        if aggregate == "error":
            raise SchemaError(
                f"{path}: {n_duplicates} row(s) map to a (time, depth, lat, lon) cell that "
                "another row already fills. Pass aggregate='mean' (or 'first', 'last', ...) "
                "to resolve them."
            )
        dataset_attrs["n_duplicate_cells_aggregated"] = n_duplicates
        dataset_attrs["duplicate_aggregation"] = aggregate

    # -- values ------------------------------------------------------------

    built = {}
    for column in value_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        grouped = pd.Series(series.to_numpy(), index=flat_index)
        if n_duplicates:
            reduced = getattr(grouped.groupby(level=0), aggregate)()
        else:
            reduced = grouped

        flat = np.full(n_cells, np.nan, dtype=np.float64)
        positions = reduced.index.to_numpy(dtype=np.int64)
        flat[positions] = reduced.to_numpy(dtype=np.float64)

        variable = standardize_variable(
            column,
            flat.reshape(shape),
            dims,
            units=column_units.get(column),
            attrs={"source_column": column},
            standardize_names=standardize_names,
            replace_sentinels=replace_sentinels,
            normalize_units=normalize_units,
        )
        built[variable.name] = variable

    dataset_attrs.setdefault("n_rows_read", int(n_rows_read))
    dataset_attrs.setdefault("n_rows_used", int(len(frame)))
    dataset_attrs.setdefault("grid_mode", grid)

    dataset = OceanDataset(
        variables=built,
        coords={d: axes[d] for d in dims},
        attrs=dataset_attrs,
        source=str(path),
        source_format="csv",
    )

    return finalize_dataset(
        dataset,
        validate=validate,
        strict=strict,
        domain=domain,
        check_domain=check_domain,
        missing_warn_fraction=missing_warn_fraction,
    )
