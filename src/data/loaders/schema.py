"""
The canonical NEER data schema: dimension order, coordinate names, and
the registry of known ocean variables.

Every loader in `src/data/loaders` maps whatever naming convention the
source file happens to use onto the vocabulary defined here, so the rest
of the codebase only ever sees one spelling of "latitude" and one
dimension ordering. Source files in the wild disagree constantly
(`lat`/`latitude`/`nav_lat`/`y`, `lev`/`depth`/`z`, `(lon, lat, time)`
vs `(time, lat, lon)`), and resolving that here keeps the disagreement
from leaking into preprocessing, modeling, and the API.

Nothing in this module reads files or touches data values — it is pure
vocabulary plus small, pure helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Dimensions & coordinates
# --------------------------------------------------------------------------

#: Canonical dimension order. Every variable is transposed into this order
#: (keeping only the dims it actually has), so a 3D surface field is always
#: (time, lat, lon) and a 4D field is always (time, depth, lat, lon).
CANONICAL_DIM_ORDER: Tuple[str, ...] = ("time", "depth", "lat", "lon")

#: Coordinate names that must be 1D axes when present.
CANONICAL_COORDS: Tuple[str, ...] = CANONICAL_DIM_ORDER

#: Coordinate name aliases seen across satellite/reanalysis/ARGO products,
#: mapped onto the canonical names. Keys are compared case-insensitively.
COORD_ALIASES: Dict[str, str] = {
    # time
    "time": "time",
    "t": "time",
    "times": "time",
    "date": "time",
    "dates": "time",
    "datetime": "time",
    "timestamp": "time",
    "valid_time": "time",
    "time_counter": "time",
    "juld": "time",  # ARGO
    # latitude
    "lat": "lat",
    "lats": "lat",
    "latitude": "lat",
    "latitudes": "lat",
    "y": "lat",
    "nav_lat": "lat",
    "ylat": "lat",
    # longitude
    "lon": "lon",
    "lons": "lon",
    "long": "lon",
    "longitude": "lon",
    "longitudes": "lon",
    "x": "lon",
    "nav_lon": "lon",
    "xlon": "lon",
    # depth
    "depth": "depth",
    "depths": "depth",
    "depth_m": "depth",
    "z": "depth",
    "lev": "depth",
    "level": "depth",
    "levels": "depth",
    "zlev": "depth",
    "pres": "depth",
    "pressure": "depth",
}


def canonical_coord_name(name: Any) -> str:
    """Map a source coordinate/dimension name onto its canonical spelling.

    Unrecognized names are returned unchanged (as `str`) rather than
    rejected — a dataset may legitimately carry extra dimensions we have
    no opinion about.
    """
    key = str(name).strip().lower()
    return COORD_ALIASES.get(key, str(name))


def canonical_dims(dims: Iterable[Any]) -> Tuple[str, ...]:
    """Map every entry of `dims` onto its canonical spelling."""
    return tuple(canonical_coord_name(d) for d in dims)


def _dim_sort_key(index_and_dim: Tuple[int, str]) -> Tuple[int, int]:
    index, dim = index_and_dim
    if dim in CANONICAL_DIM_ORDER:
        return (CANONICAL_DIM_ORDER.index(dim), index)
    # Unknown dims keep their relative order and sort after the known ones.
    return (len(CANONICAL_DIM_ORDER), index)


def transpose_to_canonical(
    values: np.ndarray, dims: Sequence[str]
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Reorder `values`' axes into `CANONICAL_DIM_ORDER`.

    Dimensions not in the canonical list keep their relative order and are
    placed last. Returns the (possibly unchanged) array and its new dims.
    Raises ValueError if `values.ndim` does not match `len(dims)`.
    """
    dims = tuple(dims)
    if values.ndim != len(dims):
        raise ValueError(
            f"array has {values.ndim} dimension(s) but {len(dims)} dim name(s) were given: {dims}"
        )

    order = [i for i, _ in sorted(enumerate(dims), key=_dim_sort_key)]
    if order == list(range(len(dims))):
        return values, dims
    return np.transpose(values, axes=order), tuple(dims[i] for i in order)


def infer_dims_from_shape(
    shape: Sequence[int], coord_sizes: Dict[str, int], *, variable: str = "<array>"
) -> Tuple[str, ...]:
    """Guess a variable's dims by matching its shape against coordinate lengths.

    Used for sources that carry raw arrays with no dimension metadata
    (plain `.npy`/`.npz`). Only unambiguous matches are accepted: if two
    coordinates have the same length, or an axis matches none of them,
    the caller must supply dims explicitly.
    """
    by_size: Dict[int, list] = {}
    for name, size in coord_sizes.items():
        by_size.setdefault(size, []).append(name)

    dims = []
    used = set()
    for axis, size in enumerate(shape):
        candidates = [c for c in by_size.get(size, []) if c not in used]
        if len(candidates) != 1:
            raise ValueError(
                f"cannot infer dimension names for '{variable}' (shape {tuple(shape)}): "
                f"axis {axis} of length {size} matches {len(candidates)} available "
                f"coordinate(s). Pass dims explicitly."
            )
        dims.append(candidates[0])
        used.add(candidates[0])

    # Reported in the order the axes actually appear; the caller transposes
    # into canonical order afterwards.
    return tuple(dims)


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VariableSpec:
    """What NEER expects of a known ocean variable.

    `valid_range` is a plausibility band used for *warnings* only — real
    data with a genuinely extreme value should not be rejected outright,
    but a salinity field reading 3000 almost certainly means the units
    were misread, and that is worth flagging loudly.
    """

    name: str
    units: str
    description: str
    expected_dims: Tuple[str, ...]
    valid_range: Optional[Tuple[float, float]] = None
    is_mask: bool = False


VARIABLE_SPECS: Dict[str, VariableSpec] = {
    "sst": VariableSpec(
        name="sst",
        units="degC",
        description="Sea surface temperature",
        expected_dims=("time", "lat", "lon"),
        valid_range=(-5.0, 45.0),
    ),
    "sss": VariableSpec(
        name="sss",
        units="psu",
        description="Sea surface salinity",
        expected_dims=("time", "lat", "lon"),
        valid_range=(0.0, 45.0),
    ),
    "sla": VariableSpec(
        name="sla",
        units="m",
        description="Sea level anomaly",
        expected_dims=("time", "lat", "lon"),
        valid_range=(-3.0, 3.0),
    ),
    "u_current": VariableSpec(
        name="u_current",
        units="m/s",
        description="Zonal surface current",
        expected_dims=("time", "lat", "lon"),
        valid_range=(-5.0, 5.0),
    ),
    "v_current": VariableSpec(
        name="v_current",
        units="m/s",
        description="Meridional surface current",
        expected_dims=("time", "lat", "lon"),
        valid_range=(-5.0, 5.0),
    ),
    "u_wind": VariableSpec(
        name="u_wind",
        units="m/s",
        description="Zonal wind",
        expected_dims=("time", "lat", "lon"),
        valid_range=(-80.0, 80.0),
    ),
    "v_wind": VariableSpec(
        name="v_wind",
        units="m/s",
        description="Meridional wind",
        expected_dims=("time", "lat", "lon"),
        valid_range=(-80.0, 80.0),
    ),
    "subsurface_temp": VariableSpec(
        name="subsurface_temp",
        units="degC",
        description="Subsurface temperature",
        expected_dims=("time", "depth", "lat", "lon"),
        valid_range=(-5.0, 45.0),
    ),
    "subsurface_salinity": VariableSpec(
        name="subsurface_salinity",
        units="psu",
        description="Subsurface salinity",
        expected_dims=("time", "depth", "lat", "lon"),
        valid_range=(0.0, 45.0),
    ),
    "land_mask": VariableSpec(
        name="land_mask",
        units="bool",
        description="True = ocean, False = land",
        expected_dims=("lat", "lon"),
        is_mask=True,
    ),
}

#: Variable-name aliases used by common products, mapped onto NEER names.
#: Only unambiguous mappings belong here; anything context-dependent (a
#: bare "temperature", say) should be renamed explicitly by the caller via
#: the loaders' `rename=` argument.
VARIABLE_ALIASES: Dict[str, str] = {
    "sst": "sst",
    "analysed_sst": "sst",
    "sea_surface_temperature": "sst",
    "sea_water_surface_temperature": "sst",
    "sss": "sss",
    "sea_surface_salinity": "sss",
    "sla": "sla",
    "sea_level_anomaly": "sla",
    "ssha": "sla",
    "u_current": "u_current",
    "ugos": "u_current",
    "uo": "u_current",
    "eastward_sea_water_velocity": "u_current",
    "v_current": "v_current",
    "vgos": "v_current",
    "vo": "v_current",
    "northward_sea_water_velocity": "v_current",
    "u_wind": "u_wind",
    "uwnd": "u_wind",
    "u10": "u_wind",
    "eastward_wind": "u_wind",
    "v_wind": "v_wind",
    "vwnd": "v_wind",
    "v10": "v_wind",
    "northward_wind": "v_wind",
    "subsurface_temp": "subsurface_temp",
    "subsurface_temperature": "subsurface_temp",
    "sea_water_temperature": "subsurface_temp",
    "land_mask": "land_mask",
    "ocean_mask": "land_mask",
    "mask": "land_mask",
}


def canonical_variable_name(name: Any) -> str:
    """Map a source variable name onto its NEER name, if known."""
    key = str(name).strip().lower()
    return VARIABLE_ALIASES.get(key, str(name))


def lookup_spec(name: Any) -> Optional[VariableSpec]:
    """Return the `VariableSpec` for `name`, or None if it is not a known variable."""
    return VARIABLE_SPECS.get(canonical_variable_name(name))
