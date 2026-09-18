"""
Stage 3 — spatial alignment.

Every source NEER ingests lives on its own grid: satellite SST at 0.05°,
altimetry at 0.25°, scatterometer winds at 0.125°, reanalysis on a
stretched grid entirely. The model needs them stacked as channels of one
array, which means one grid — the project's target grid from
`configs/base.yaml -> domain`, built by `src/data/grid.py` (5°N-30°N,
45°E-105°E at 0.25°).

This step resamples each variable's `lat` and `lon` axes onto that target
grid, one axis at a time (separable 1D interpolation: bilinear on a
regular grid is exactly the composition of two linear interpolations, and
doing it axis-by-axis keeps the implementation readable and NaN-aware).

Three details worth stating, because each one is a bug if got wrong:

* **Missing stays missing.** Interpolation never invents a value from a
  missing neighbour: if either bracketing cell is NaN, the output is NaN.
  Filling gaps is the *next* stage's job, and it does it with fitted
  statistics that respect the train/test split — quietly filling them
  here with a spatial average would bypass that entirely.
* **Masks use nearest, never linear.** A land/ocean mask averaged to 0.5
  is meaningless. Boolean variables are always resampled by nearest
  neighbour and stay boolean.
* **Outside the source domain is NaN, not extrapolation.** A target cell
  further than `tolerance` from any source data has no data, and says so.

When the source grid already equals the target grid — the common case
once a product has been ingested once, and the case for the demo dataset
— the step detects it and returns the dataset untouched, so re-running
the pipeline costs nothing.

Like the two stages before it, this step learns nothing from the data.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np

from src.data.grid import OceanGrid, create_ocean_grid
from src.data.loaders.representation import OceanDataset, Variable
from src.data.preprocessing._utils import (
    axis_of,
    is_mask_variable,
    with_values,
    with_variables,
)
from src.data.preprocessing.base import StatelessStep
from src.data.preprocessing.errors import StepConfigurationError

#: Supported resampling methods.
METHODS: Tuple[str, ...] = ("linear", "nearest")

#: How far a target coordinate may sit from the nearest source coordinate
#: (in degrees) before it is treated as outside the source domain.
DEFAULT_TOLERANCE = 1e-6


def interp_axis(
    values: np.ndarray,
    axis: int,
    source: np.ndarray,
    target: np.ndarray,
    *,
    method: str = "linear",
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Resample `values` along `axis` from `source` coordinates onto `target`.

    `source` must be ascending (the coordinate normalization stage
    guarantees it). Missing values propagate: a target cell bracketed by
    a NaN is NaN. Target coordinates outside the source range by more
    than `tolerance` are NaN.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.size == 0:
        raise StepConfigurationError("cannot interpolate from an empty source axis")

    moved = np.moveaxis(np.asarray(values), axis, -1)
    outside = (target < source[0] - tolerance) | (target > source[-1] + tolerance)

    if method == "nearest" or source.size == 1:
        right = np.searchsorted(source, target)
        left = np.clip(right - 1, 0, source.size - 1)
        right = np.clip(right, 0, source.size - 1)
        pick_right = np.abs(source[right] - target) < np.abs(source[left] - target)
        index = np.where(pick_right, right, left)
        result = moved[..., index].astype(moved.dtype)
        if result.dtype == bool:
            result = result.copy()
            result[..., outside] = False
        else:
            result = result.astype(float)
            result[..., outside] = np.nan
        return np.moveaxis(result, -1, axis)

    if method != "linear":
        raise StepConfigurationError(
            f"unknown method {method!r}; expected one of {list(METHODS)}"
        )

    moved = moved.astype(float)
    upper = np.clip(np.searchsorted(source, target), 1, source.size - 1)
    lower = upper - 1
    span = source[upper] - source[lower]
    weight = np.where(span > 0, (target - source[lower]) / np.where(span > 0, span, 1.0), 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    low_values = moved[..., lower]
    high_values = moved[..., upper]
    result = low_values * (1.0 - weight) + high_values * weight

    # An exact hit must not inherit a NaN from the neighbour it does not use.
    exact_low = weight <= 0.0
    exact_high = weight >= 1.0
    if np.any(exact_low):
        result[..., exact_low] = low_values[..., exact_low]
    if np.any(exact_high):
        result[..., exact_high] = high_values[..., exact_high]

    result[..., outside] = np.nan
    return np.moveaxis(result, -1, axis)


class SpatialAligner(StatelessStep):
    """Resample every variable onto the project's target lat/lon grid."""

    name: ClassVar[str] = "spatial_alignment"
    title: ClassVar[str] = "Spatial alignment"

    def __init__(
        self,
        *,
        grid: Optional[OceanGrid] = None,
        method: str = "linear",
        mask_method: str = "nearest",
        tolerance: float = DEFAULT_TOLERANCE,
        environment: Optional[str] = None,
    ) -> None:
        super().__init__()
        if method not in METHODS or mask_method not in METHODS:
            raise StepConfigurationError(
                f"method/mask_method must be one of {list(METHODS)}"
            )
        self.method = method
        self.mask_method = mask_method
        self.tolerance = float(tolerance)
        self.environment = environment
        self._grid = grid

    @property
    def grid(self) -> OceanGrid:
        """The target grid; built from the configured domain on first use."""
        if self._grid is None:
            from src.utils.config import load_config

            domain = load_config(self.environment).domain if self.environment else None
            self._grid = create_ocean_grid(domain)
        return self._grid

    def config(self) -> Dict[str, Any]:
        grid = self.grid
        return {
            "method": self.method,
            "mask_method": self.mask_method,
            "tolerance": self.tolerance,
            "target_grid": grid.metadata(),
        }

    # -- transform ---------------------------------------------------------

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        grid = self.grid
        target = {"lat": grid.latitudes, "lon": grid.longitudes}

        source_lat = dataset.coords.get("lat")
        source_lon = dataset.coords.get("lon")
        if source_lat is None or source_lon is None:
            self._report = {"skipped": "dataset is not on a lat/lon grid"}
            return dataset

        already_aligned = self._matches(source_lat, target["lat"]) and self._matches(
            source_lon, target["lon"]
        )
        if already_aligned:
            self._report = {
                "regridded": False,
                "reason": "source grid already matches the target grid",
                "grid_shape": list(grid.shape),
            }
            return dataset

        variables = {}
        for name, variable in dataset.variables.items():
            variables[name] = self._resample(variable, dataset.coords, target)

        coords = dict(dataset.coords)
        coords["lat"] = target["lat"]
        coords["lon"] = target["lon"]

        self._report = {
            "regridded": True,
            "method": self.method,
            "from_shape": [int(source_lat.size), int(source_lon.size)],
            "to_shape": list(grid.shape),
            "lat_range": [float(target["lat"].min()), float(target["lat"].max())],
            "lon_range": [float(target["lon"].min()), float(target["lon"].max())],
        }
        return with_variables(dataset, variables, coords=coords)

    # -- internals ---------------------------------------------------------

    def _matches(self, source: np.ndarray, target: np.ndarray) -> bool:
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        return source.shape == target.shape and bool(
            np.allclose(source, target, atol=max(self.tolerance, 1e-9))
        )

    def _resample(self, variable: Variable, coords, target) -> Variable:
        values = variable.values
        method = self.mask_method if is_mask_variable(variable.name, variable) else self.method
        changed = False
        for dim in ("lat", "lon"):
            axis = axis_of(variable, dim)
            if axis is None:
                continue
            source = np.asarray(coords[dim], dtype=float)
            if self._matches(source, target[dim]):
                continue
            values = interp_axis(
                values,
                axis,
                source,
                target[dim],
                method=method,
                tolerance=self.tolerance,
            )
            changed = True
        if not changed:
            return variable
        if is_mask_variable(variable.name, variable) and variable.dtype == bool:
            values = np.asarray(values, dtype=bool)
        return with_values(variable, values)
