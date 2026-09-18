"""
Stage 1 — coordinate normalization.

The loaders already guarantee canonical coordinate *names* (`time`,
`depth`, `lat`, `lon`) and canonical axis *order*. What they do not
guarantee is that the coordinate *values* agree between two files: one
product numbers longitudes 0-360, the next uses -180-180; one stores
latitudes descending (north first, the image convention), the next
ascending; one carries 44.99999999999999 where another carries 45.0.

Left alone, those differences turn into silent errors downstream — a
regridding step that produces an all-NaN field because the target
longitudes never match the source ones, or a field that is upside down
relative to the mask applied to it. This step puts every dataset on one
convention before anything else touches it:

* longitudes in a single, configurable convention (`0-360` by default,
  which keeps the NEER domain, 45°E-105°E, contiguous and positive);
* latitude, longitude and depth axes strictly ascending, with the data
  arrays reordered to match — not just the coordinate arrays, which
  would silently flip the data;
* coordinate values rounded to a fixed number of decimals, so that grid
  comparisons can use equality within a tolerance rather than hope;
* duplicate coordinate values dropped (first occurrence wins), since a
  repeated latitude makes every later alignment ambiguous.

The step learns nothing, so it is safe to apply to the full dataset
before the train/test split is even decided.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np

from src.data.loaders.representation import OceanDataset
from src.data.preprocessing._utils import select_along_dim, with_variables
from src.data.preprocessing.base import StatelessStep
from src.data.preprocessing.errors import StepConfigurationError

#: Supported longitude conventions.
LON_CONVENTIONS: Tuple[str, ...] = ("0-360", "-180-180", "keep")

#: Axes that must be ascending for interpolation and masking to be well defined.
ASCENDING_AXES: Tuple[str, ...] = ("lat", "lon", "depth", "time")

#: Decimal places coordinates are rounded to. Matches `src/data/grid.py`,
#: so a normalized axis compares equal to the generated target grid.
DEFAULT_DECIMALS = 6


def to_convention(lon: np.ndarray, convention: str) -> np.ndarray:
    """Re-express longitudes in `convention` (`0-360`, `-180-180`, or `keep`)."""
    lon = np.asarray(lon, dtype=float)
    if convention == "keep":
        return lon.copy()
    if convention == "0-360":
        return np.mod(lon, 360.0)
    if convention == "-180-180":
        return ((lon + 180.0) % 360.0) - 180.0
    raise StepConfigurationError(
        f"unknown longitude convention {convention!r}; expected one of {list(LON_CONVENTIONS)}"
    )


class CoordinateNormalizer(StatelessStep):
    """Put coordinate values on one convention: units, direction, precision."""

    name: ClassVar[str] = "coordinate_normalization"
    title: ClassVar[str] = "Coordinate normalization"

    def __init__(
        self,
        *,
        lon_convention: str = "0-360",
        sort_ascending: bool = True,
        decimals: Optional[int] = DEFAULT_DECIMALS,
        drop_duplicates: bool = True,
    ) -> None:
        super().__init__()
        if lon_convention not in LON_CONVENTIONS:
            raise StepConfigurationError(
                f"lon_convention must be one of {list(LON_CONVENTIONS)}, got {lon_convention!r}"
            )
        self.lon_convention = lon_convention
        self.sort_ascending = sort_ascending
        self.decimals = decimals
        self.drop_duplicates = drop_duplicates

    def config(self) -> Dict[str, Any]:
        return {
            "lon_convention": self.lon_convention,
            "sort_ascending": self.sort_ascending,
            "decimals": self.decimals,
            "drop_duplicates": self.drop_duplicates,
        }

    # -- transform ---------------------------------------------------------

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        report: Dict[str, Any] = {"converted": {}, "reordered": [], "dropped_duplicates": {}}

        # 1. Longitude convention. Done first: it can change the sort order.
        if "lon" in dataset.coords and self.lon_convention != "keep":
            original = np.asarray(dataset.coords["lon"], dtype=float)
            converted = to_convention(original, self.lon_convention)
            if not np.array_equal(original, converted):
                report["converted"]["lon"] = {
                    "convention": self.lon_convention,
                    "from": [float(original.min()), float(original.max())],
                    "to": [float(converted.min()), float(converted.max())],
                }
            coords = dict(dataset.coords)
            coords["lon"] = converted
            dataset = with_variables(dataset, dataset.variables, coords=coords)

        # 2. Rounding, before sorting/dedup so that near-identical values
        #    (45.0 and 44.999999999) are recognized as the same point.
        if self.decimals is not None:
            coords = dict(dataset.coords)
            for name, values in coords.items():
                if np.issubdtype(np.asarray(values).dtype, np.floating):
                    coords[name] = np.round(np.asarray(values, dtype=float), self.decimals)
            dataset = with_variables(dataset, dataset.variables, coords=coords)

        # 3. Ascending order, reordering the data arrays with the axis.
        if self.sort_ascending:
            for axis in ASCENDING_AXES:
                values = dataset.coords.get(axis)
                if values is None or values.size < 2:
                    continue
                order = np.argsort(values, kind="stable")
                if not np.array_equal(order, np.arange(values.size)):
                    dataset = select_along_dim(dataset, axis, order)
                    report["reordered"].append(axis)

        # 4. Duplicate coordinate values.
        if self.drop_duplicates:
            for axis in ASCENDING_AXES:
                values = dataset.coords.get(axis)
                if values is None or values.size < 2:
                    continue
                _, first_indices = np.unique(values, return_index=True)
                keep = np.sort(first_indices)
                if keep.size != values.size:
                    report["dropped_duplicates"][axis] = int(values.size - keep.size)
                    dataset = select_along_dim(dataset, axis, keep)

        report["final_ranges"] = {
            name: [str(values.min()), str(values.max())]
            for name, values in dataset.coords.items()
            if values.size
        }
        self._report = report
        return dataset
