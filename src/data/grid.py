"""
Ocean grid construction for NEER.

Builds the regular lat/lon grid over the project's target domain
(5°N-30°N, 45°E-105°E at 0.25° resolution — see configs/base.yaml ->
`domain`, the single source of truth for these bounds). This module is
purely geographic: it builds coordinate arrays and grid metadata. No
model, embedding, or reconstruction logic lives here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.config import DomainConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "grid_metadata.json"

# Number of decimal places coordinates are rounded to, to eliminate
# floating-point drift (e.g. 45.00000000000003) while keeping 0.25°
# precision intact.
_COORD_DECIMALS = 6


def _default_domain() -> DomainConfig:
    """Load the domain config for the default (development) environment."""
    return load_config().domain


def _grid_point_count(min_value: float, max_value: float, resolution: float) -> int:
    """Number of inclusive grid points spanning [min_value, max_value] at `resolution`."""
    span_steps = (max_value - min_value) / resolution
    n_points = round(span_steps) + 1
    if n_points < 1:
        raise ValueError("Domain bounds/resolution produce zero or negative grid points")
    return n_points


def create_latitude_grid(domain: Optional[DomainConfig] = None) -> np.ndarray:
    """Build the 1D latitude coordinate array.

    Ascending order, from `domain.lat_min` to `domain.lat_max` inclusive,
    spaced by `domain.resolution` degrees.
    """
    domain = domain or _default_domain()
    n_points = _grid_point_count(domain.lat_min, domain.lat_max, domain.resolution)
    lats = domain.lat_min + np.arange(n_points) * domain.resolution
    return np.round(lats, _COORD_DECIMALS)


def create_longitude_grid(domain: Optional[DomainConfig] = None) -> np.ndarray:
    """Build the 1D longitude coordinate array.

    Ascending order, from `domain.lon_min` to `domain.lon_max` inclusive,
    spaced by `domain.resolution` degrees.
    """
    domain = domain or _default_domain()
    n_points = _grid_point_count(domain.lon_min, domain.lon_max, domain.resolution)
    lons = domain.lon_min + np.arange(n_points) * domain.resolution
    return np.round(lons, _COORD_DECIMALS)


@dataclass(frozen=True)
class OceanGrid:
    """A regular lat/lon grid over the NEER ocean domain."""

    latitudes: np.ndarray        # 1D, shape (n_lat,), ascending
    longitudes: np.ndarray       # 1D, shape (n_lon,), ascending
    lat_grid: np.ndarray         # 2D meshgrid, shape (n_lat, n_lon)
    lon_grid: np.ndarray         # 2D meshgrid, shape (n_lat, n_lon)
    resolution: float
    bounds: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_lat, n_lon) — the shape of any field defined on this grid."""
        return self.lat_grid.shape

    def metadata(self) -> dict:
        """JSON-serializable metadata describing this grid (no coordinate arrays)."""
        lat_min, lat_max, lon_min, lon_max = self.bounds
        n_lat, n_lon = self.shape
        return {
            "resolution_deg": self.resolution,
            "bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "shape": {"n_lat": int(n_lat), "n_lon": int(n_lon)},
            "n_points": int(n_lat * n_lon),
        }


def create_ocean_grid(domain: Optional[DomainConfig] = None) -> OceanGrid:
    """Build the full 2D ocean grid (coordinate arrays + meshgrids) for `domain`.

    Defaults to the domain from the default-environment config
    (configs/base.yaml merged with configs/development.yaml) when no
    domain is given.
    """
    domain = domain or _default_domain()
    lats = create_latitude_grid(domain)
    lons = create_longitude_grid(domain)

    # np.meshgrid(lons, lats) with default 'xy' indexing returns arrays of
    # shape (len(lats), len(lons)) — i.e. rows vary over latitude, columns
    # over longitude, matching the conventional (n_lat, n_lon) field shape.
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    return OceanGrid(
        latitudes=lats,
        longitudes=lons,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        resolution=domain.resolution,
        bounds=(domain.lat_min, domain.lat_max, domain.lon_min, domain.lon_max),
    )


def save_grid_metadata(grid: OceanGrid, path: Optional[Path] = None) -> Path:
    """Write the grid's metadata (bounds, resolution, shape) to a JSON file.

    Only metadata is persisted here — not the coordinate arrays themselves,
    which are cheap to regenerate deterministically from `configs/base.yaml`.
    Defaults to `data/processed/grid_metadata.json`.
    """
    path = Path(path) if path is not None else DEFAULT_METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = grid.metadata()
    metadata["project"] = "NEER"
    metadata["problem_id"] = "SIH26066"
    metadata["organization"] = "MoES / INCOIS"
    metadata["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return path
