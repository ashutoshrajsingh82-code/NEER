"""
Phase 26 — matching ARGO profiles to NEER's time axis and grid.

Two independent stages, each returning either a match or a *named*
rejection reason (so the report can say why profiles were lost, not just
how many):

**Time matching** (`match_time`). NEER's tensors are monthly (the
preprocessing default: timestamps snapped to the start of the month), so
by default a profile matches the timestep in the *same calendar month*.
For a finer-cadence axis it matches the nearest timestep within a
tolerance (default: half the median spacing). Never "closest month" —
a January prediction is not a February observation.

**Spatial matching** (`match_space`). A profile matches the grid cell it
falls in: nearest cell centre, accepted only if the profile lies within
`max_cell_offset` cells (default 0.5 = inside the cell) of it, inside the
domain, on a cell the ocean mask calls ocean. Longitudes are compared
circularly and put in the grid's own convention (NEER uses 0–360; ARGO
uses -180–180).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

REJECT_NO_TIME_MATCH = "no_time_match"
REJECT_OUTSIDE_DOMAIN = "outside_domain"
REJECT_LAND_CELL = "on_land_cell"

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class NeerGrid:
    """The parts of a `TensorBundle` the ARGO pipeline needs."""

    time: np.ndarray            # (n_time,) datetime64, ascending
    lat: np.ndarray             # (n_lat,) ascending
    lon: np.ndarray             # (n_lon,) ascending
    depth: np.ndarray           # (n_depth,) metres, ascending
    ocean_mask: Optional[np.ndarray] = None   # (n_lat, n_lon) bool
    split_of_time: Optional[np.ndarray] = None  # (n_time,) str: train/val/test/""
    attrs: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_bundle(cls, bundle) -> "NeerGrid":
        if bundle.depth is None:
            raise ValueError("tensor bundle has no depth axis; cannot validate against profiles")
        split = np.full(bundle.time.shape, "", dtype=object)
        for name, mask in bundle.split_masks.items():
            split[np.asarray(mask, dtype=bool)] = name
        return cls(
            time=np.asarray(bundle.time),
            lat=np.asarray(bundle.lat, dtype=float),
            lon=np.asarray(bundle.lon, dtype=float),
            depth=np.asarray(bundle.depth, dtype=float),
            ocean_mask=None if bundle.ocean_mask is None else np.asarray(bundle.ocean_mask, dtype=bool),
            split_of_time=split if bundle.split_masks else None,
            attrs=dict(bundle.attrs),
        )


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeMatchConfig:
    mode: str = "auto"                  # auto | monthly | nearest
    tolerance_days: Optional[float] = None  # nearest mode; default half the median spacing


def _resolve_mode(grid_time: np.ndarray, cfg: TimeMatchConfig) -> Tuple[str, float]:
    if grid_time.size < 2:
        spacing = 30.0
    else:
        spacing = float(np.median(np.diff(grid_time).astype("timedelta64[s]").astype(float)) / 86400.0)
    mode = cfg.mode
    if mode == "auto":
        mode = "monthly" if spacing >= 27.0 else "nearest"
    if mode not in ("monthly", "nearest"):
        raise ValueError(f"unknown time-match mode {cfg.mode!r}")
    tol = cfg.tolerance_days if cfg.tolerance_days is not None else spacing / 2.0
    return mode, tol


def match_time(
    profile_time: np.datetime64, grid_time: np.ndarray, cfg: TimeMatchConfig = TimeMatchConfig()
) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """`(time_index, offset_days, None)` on a match, `(None, None, reason)` otherwise.

    `offset_days` is profile time minus the timestep's own stamp (for a
    monthly axis stamped at month start this is just "days into the month").
    """
    mode, tol = _resolve_mode(grid_time, cfg)
    if mode == "monthly":
        months = grid_time.astype("datetime64[M]")
        hits = np.nonzero(months == profile_time.astype("datetime64[M]"))[0]
        if hits.size == 0:
            return None, None, REJECT_NO_TIME_MATCH
        i = int(hits[0])
    else:
        deltas = np.abs((grid_time - profile_time).astype("timedelta64[s]").astype(float)) / 86400.0
        i = int(np.argmin(deltas))
        if deltas[i] > tol:
            return None, None, REJECT_NO_TIME_MATCH
    offset = float((profile_time - grid_time[i]).astype("timedelta64[s]").astype(float) / 86400.0)
    return i, offset, None


# --------------------------------------------------------------------------
# Space
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceMatchConfig:
    max_cell_offset: float = 0.5  # in grid cells, per axis


def _spacing(axis: np.ndarray) -> float:
    return float(np.median(np.diff(axis))) if axis.size > 1 else 1.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dlmb = np.deg2rad(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def match_space(
    lat: float,
    lon: float,
    grid: NeerGrid,
    cfg: SpaceMatchConfig = SpaceMatchConfig(),
) -> Tuple[Optional[Tuple[int, int]], Optional[float], Optional[str]]:
    """`((lat_index, lon_index), distance_km, None)` or `(None, None, reason)`."""
    dlat, dlon = abs(_spacing(grid.lat)), abs(_spacing(grid.lon))
    uses_0_360 = float(np.nanmax(grid.lon)) > 180.0
    lon_g = lon % 360.0 if uses_0_360 else ((lon + 180.0) % 360.0) - 180.0

    i = int(np.argmin(np.abs(grid.lat - lat)))
    circ = ((grid.lon - lon_g + 180.0) % 360.0) - 180.0
    j = int(np.argmin(np.abs(circ)))

    tol = cfg.max_cell_offset
    if abs(grid.lat[i] - lat) > tol * dlat + 1e-9 or abs(circ[j]) > tol * dlon + 1e-9:
        return None, None, REJECT_OUTSIDE_DOMAIN
    if grid.ocean_mask is not None and not bool(grid.ocean_mask[i, j]):
        return None, None, REJECT_LAND_CELL
    return (i, j), haversine_km(lat, lon_g, float(grid.lat[i]), float(grid.lon[j])), None