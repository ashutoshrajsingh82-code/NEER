"""
The standardized internal representation returned by every NEER loader.

`load_netcdf`, `load_csv`, `load_npz` and `load_demo_dataset` all return
an `OceanDataset`, regardless of what the file on disk looked like. That
gives the rest of the project (preprocessing, modeling, evaluation, the
API) exactly one shape of object to program against.

Guarantees an `OceanDataset` provides
-------------------------------------
1. Coordinates are named `time`, `depth`, `lat`, `lon` — never `latitude`,
   `nav_lat`, `lev`, or `t` (see `schema.COORD_ALIASES`).
2. Coordinates are 1D NumPy arrays; `time` is `datetime64[ns]`.
3. Variable axes are in canonical order: `(time, depth, lat, lon)`,
   restricted to the dims that variable actually has.
4. Missing data is `NaN` — never `-999`, `1e20`, or a `_FillValue`
   attribute the caller has to remember (see `missing.py`).
5. Units are canonical where recognized (`degC`, `psu`, `m`, `m/s`), with
   the original unit preserved in `Variable.attrs["source_units"]` when a
   conversion was applied.
6. Provenance is preserved: `attrs` carries `data_mode` (notably
   `DEMO_SYNTHETIC`) and the disclaimer for synthetic data, so a demo
   dataset can never be mistaken for real observations downstream.

Why not just return `xarray.Dataset`?
-------------------------------------
xarray is used for NetCDF I/O and is the right tool there, but making it
the internal type would make it a hard dependency of every module that
merely *touches* data, including the API layer and the CSV/NumPy paths.
`OceanDataset` is a thin, dependency-light container that converts to and
from xarray on demand (`to_xarray` / `from_xarray`), so xarray stays an
optional dependency needed only where NetCDF is actually involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.data.loaders.missing import finite_range, missing_fraction, missing_mask
from src.data.loaders.schema import (
    CANONICAL_COORDS,
    VariableSpec,
    lookup_spec,
    transpose_to_canonical,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.data.loaders.validation import ValidationReport


@dataclass(frozen=True)
class Variable:
    """One data variable: its array, its dims, its units, its provenance."""

    name: str
    values: np.ndarray
    dims: Tuple[str, ...]
    units: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dims", tuple(str(d) for d in self.dims))
        if values.ndim != len(self.dims):
            raise ValueError(
                f"variable '{self.name}': array has {values.ndim} dimension(s) "
                f"but dims={self.dims} declares {len(self.dims)}"
            )

    # -- shape -------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def sizes(self) -> Dict[str, int]:
        """dim name -> length."""
        return dict(zip(self.dims, self.values.shape))

    # -- semantics ---------------------------------------------------------

    @property
    def spec(self) -> Optional[VariableSpec]:
        """The known-variable spec for this name, if NEER has one."""
        return lookup_spec(self.name)

    @property
    def is_mask(self) -> bool:
        return self.values.dtype == bool or bool(self.spec and self.spec.is_mask)

    # -- missing data ------------------------------------------------------

    @property
    def missing_mask(self) -> np.ndarray:
        return missing_mask(self.values)

    @property
    def missing_fraction(self) -> float:
        return missing_fraction(self.values)

    @property
    def n_missing(self) -> int:
        return int(self.missing_mask.sum())

    @property
    def value_range(self) -> Optional[Tuple[float, float]]:
        """(min, max) ignoring missing values, or None if entirely missing."""
        return finite_range(self.values)

    # -- transforms --------------------------------------------------------

    def transposed_to_canonical(self) -> "Variable":
        """Return a copy with axes in `CANONICAL_DIM_ORDER`."""
        values, dims = transpose_to_canonical(self.values, self.dims)
        if dims == self.dims:
            return self
        return replace(self, values=values, dims=dims)

    def summary(self) -> Dict[str, Any]:
        """JSON-serializable description of this variable (no data values)."""
        value_range = self.value_range
        return {
            "name": self.name,
            "dims": list(self.dims),
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "units": self.units,
            "missing_fraction": round(self.missing_fraction, 6),
            "n_missing": self.n_missing,
            "min": None if value_range is None else value_range[0],
            "max": None if value_range is None else value_range[1],
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        units = f" [{self.units}]" if self.units else ""
        return f"<Variable {self.name}{units} dims={self.dims} shape={self.shape}>"


@dataclass(frozen=True)
class OceanDataset:
    """A standardized, loader-agnostic ocean dataset.

    Construct one through a loader (`load_netcdf`, `load_csv`,
    `load_demo_dataset`, `load_npz`) rather than directly, unless you are
    writing a new loader or a test.
    """

    variables: Dict[str, Variable]
    coords: Dict[str, np.ndarray]
    attrs: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    source_format: Optional[str] = None
    validation: Optional["ValidationReport"] = None

    def __post_init__(self) -> None:
        variables = dict(self.variables)
        for key, variable in variables.items():
            if not isinstance(variable, Variable):
                raise TypeError(f"variables['{key}'] must be a Variable, got {type(variable)!r}")
            if variable.name != key:
                raise ValueError(
                    f"variable key '{key}' does not match Variable.name '{variable.name}'"
                )
        object.__setattr__(self, "variables", variables)

        coords = {}
        for name, values in self.coords.items():
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(
                    f"coordinate '{name}' must be 1-dimensional, got shape {array.shape}"
                )
            coords[str(name)] = array
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "attrs", dict(self.attrs))

    # -- mapping-style access ---------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self.variables

    def __getitem__(self, name: str) -> Variable:
        try:
            return self.variables[name]
        except KeyError:
            raise KeyError(
                f"no variable '{name}' in dataset; available: {sorted(self.variables)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    def get(self, name: str, default: Any = None) -> Any:
        return self.variables.get(name, default)

    def values_of(self, name: str) -> np.ndarray:
        """The raw NumPy array for `name` (shorthand for `ds[name].values`)."""
        return self[name].values

    # -- structure ---------------------------------------------------------

    @property
    def variable_names(self) -> List[str]:
        return sorted(self.variables)

    @property
    def coord_names(self) -> List[str]:
        """Coordinate names, canonical ones first in canonical order."""
        known = [c for c in CANONICAL_COORDS if c in self.coords]
        extra = sorted(c for c in self.coords if c not in CANONICAL_COORDS)
        return known + extra

    @property
    def sizes(self) -> Dict[str, int]:
        """dim name -> length, taken from the coordinate arrays."""
        return {name: int(self.coords[name].size) for name in self.coord_names}

    @property
    def time(self) -> Optional[np.ndarray]:
        return self.coords.get("time")

    @property
    def lat(self) -> Optional[np.ndarray]:
        return self.coords.get("lat")

    @property
    def lon(self) -> Optional[np.ndarray]:
        return self.coords.get("lon")

    @property
    def depth(self) -> Optional[np.ndarray]:
        return self.coords.get("depth")

    @property
    def n_time(self) -> int:
        time = self.coords.get("time")
        return 0 if time is None else int(time.size)

    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """(n_lat, n_lon), or None if the dataset is not on a lat/lon grid."""
        if "lat" not in self.coords or "lon" not in self.coords:
            return None
        return (int(self.coords["lat"].size), int(self.coords["lon"].size))

    # -- provenance --------------------------------------------------------

    @property
    def data_mode(self) -> Optional[str]:
        """Provenance marker, e.g. `DEMO_SYNTHETIC` for the demo dataset."""
        mode = self.attrs.get("data_mode")
        return None if mode is None else str(mode)

    @property
    def is_synthetic(self) -> bool:
        """True when this data is synthetic and must not be treated as observations."""
        mode = (self.data_mode or "").upper()
        return "SYNTHETIC" in mode or "DEMO" in mode

    # -- derived views -----------------------------------------------------

    def subset(self, names: Sequence[str]) -> "OceanDataset":
        """A new dataset with only `names`, dropping now-unused coordinates."""
        missing = [n for n in names if n not in self.variables]
        if missing:
            raise KeyError(
                f"variable(s) not in dataset: {missing}; available: {sorted(self.variables)}"
            )
        variables = {n: self.variables[n] for n in names}
        used_dims = {d for v in variables.values() for d in v.dims}
        coords = {k: v for k, v in self.coords.items() if k in used_dims}
        return replace(self, variables=variables, coords=coords, validation=None)

    def with_validation(self, report: "ValidationReport") -> "OceanDataset":
        """A copy carrying `report` (loaders use this after validating)."""
        return replace(self, validation=report)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Plain `{name: array}` for variables *and* coordinates.

        Convenient for `np.savez`, quick numeric work, and interop with
        code that predates this representation.
        """
        out: Dict[str, np.ndarray] = {name: v.values for name, v in self.variables.items()}
        out.update(self.coords)
        return out

    # -- reporting ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """JSON-serializable description of the whole dataset (no data values)."""
        time = self.coords.get("time")
        time_range = None
        if time is not None and time.size and np.issubdtype(time.dtype, np.datetime64):
            valid = time[~np.isnat(time)]
            if valid.size:
                time_range = [str(np.min(valid)), str(np.max(valid))]
        return {
            "source": self.source,
            "source_format": self.source_format,
            "data_mode": self.data_mode,
            "is_synthetic": self.is_synthetic,
            "sizes": self.sizes,
            "time_range": time_range,
            "variables": [self.variables[n].summary() for n in self.variable_names],
            "attrs": {k: v for k, v in self.attrs.items() if k != "disclaimer"},
        }

    def describe(self) -> str:
        """A short human-readable summary, for logs and notebooks."""
        lines = [
            f"OceanDataset ({self.source_format or 'unknown format'})",
            f"  source: {self.source or '-'}",
        ]
        if self.data_mode:
            marker = "  ⚠ SYNTHETIC" if self.is_synthetic else "  data_mode"
            lines.append(f"{marker}: {self.data_mode}")
        sizes = ", ".join(f"{k}={v}" for k, v in self.sizes.items())
        lines.append(f"  dims: {sizes or '-'}")
        lines.append("  variables:")
        for name in self.variable_names:
            variable = self.variables[name]
            units = variable.units or "-"
            lines.append(
                f"    {name:<20} {str(variable.dims):<32} "
                f"units={units:<6} missing={variable.missing_fraction:6.1%}"
            )
        if self.validation is not None:
            lines.append(f"  validation: {self.validation.summary_line()}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<OceanDataset vars={self.variable_names} sizes={self.sizes} "
            f"format={self.source_format!r}>"
        )

    # -- xarray interop ----------------------------------------------------

    def to_xarray(self):
        """Convert to an `xarray.Dataset` (requires xarray).

        Coordinates, dims, units and attrs are carried over, so the result
        can be written straight to NetCDF.
        """
        from src.data.loaders._xarray import require_xarray

        xr = require_xarray("converting an OceanDataset to xarray")

        data_vars = {}
        for name, variable in self.variables.items():
            attrs = dict(variable.attrs)
            if variable.units:
                attrs["units"] = variable.units
            data_vars[name] = (variable.dims, variable.values, attrs)

        coords = {name: self.coords[name] for name in self.coord_names}
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(self.attrs))

    @classmethod
    def from_xarray(
        cls,
        dataset,
        *,
        source: Optional[str] = None,
        source_format: str = "netcdf",
    ) -> "OceanDataset":
        """Build an `OceanDataset` from an `xarray.Dataset`.

        Renames coordinates/dims to canonical names and transposes each
        variable into canonical axis order. Value-level cleaning (fill
        values, unit conversion) is the loader's job, not this method's.
        """
        from src.data.loaders.schema import canonical_coord_name

        coords: Dict[str, np.ndarray] = {}
        for name, coord in dataset.coords.items():
            array = np.asarray(coord.values)
            if array.ndim != 1:
                # 2D curvilinear coordinates are out of scope for this phase.
                continue
            coords[canonical_coord_name(name)] = array

        variables: Dict[str, Variable] = {}
        for name, array in dataset.data_vars.items():
            dims = tuple(canonical_coord_name(d) for d in array.dims)
            values, dims = transpose_to_canonical(np.asarray(array.values), dims)
            attrs = {k: v for k, v in array.attrs.items()}
            units = attrs.pop("units", None)
            variables[str(name)] = Variable(
                name=str(name), values=values, dims=dims, units=units, attrs=attrs
            )

        return cls(
            variables=variables,
            coords=coords,
            attrs=dict(dataset.attrs),
            source=source,
            source_format=source_format,
        )


def build_variable(
    name: str,
    values: np.ndarray,
    dims: Sequence[str],
    *,
    units: Optional[str] = None,
    attrs: Optional[Mapping[str, Any]] = None,
) -> Variable:
    """Create a `Variable` with canonical dim names and canonical axis order."""
    from src.data.loaders.schema import canonical_dims

    dims = canonical_dims(dims)
    values, dims = transpose_to_canonical(np.asarray(values), dims)
    return Variable(
        name=name, values=values, dims=dims, units=units, attrs=dict(attrs or {})
    )
