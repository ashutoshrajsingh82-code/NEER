"""
Stage 6 — feature construction.

The model does not have to rediscover trigonometry, and giving it a few
derived fields costs nothing and removes whole classes of things it would
otherwise have to learn from limited data.

Three families are built here, all of them pure functions of data that is
already present — which is why this stage learns nothing and cannot leak:

**Derived physical fields**
    `current_speed` and `wind_speed` from their components. A network can
    of course compute a magnitude from `u` and `v`, but the magnitude is
    the physically meaningful quantity for mixing and upwelling, and
    handing it over directly is free.

**Cyclic time encodings**
    `time_sin` / `time_cos` from day-of-year. Not the raw month number:
    December (12) and January (1) are adjacent in the ocean and eleven
    apart on the number line, and that discontinuity is a real source of
    error at the year boundary. Sine and cosine of the annual phase put
    them next to each other, as they should be.

**Static geographic fields**
    `lat_norm` and `lon_norm` (position scaled to roughly [-1, 1] over
    the domain) and `coriolis`, the Coriolis parameter *f* = 2Ω·sin(φ).
    Latitude matters to subsurface structure through *f* — it governs
    geostrophic balance and the depth of the thermocline — so providing
    it directly is more useful than providing degrees north.

Scope
-----
Only features computable from what NEER already loads. Nothing here
consults an external product, and nothing consults the target variable:
`subsurface_temp` is never an input to a constructed feature, or the
reconstruction task would be partly solved in the preprocessing.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np

from src.data.loaders.representation import OceanDataset, build_variable
from src.data.preprocessing._utils import day_of_year, unique_preserving_order, with_variables
from src.data.preprocessing.base import StatelessStep
from src.data.preprocessing.errors import StepConfigurationError

#: Earth's rotation rate (rad/s), for the Coriolis parameter.
EARTH_OMEGA = 7.2921e-5

#: Features built by default — exactly the derived channels the
#: authoritative NEER input set (`channels.NEER_CHANNEL_ORDER`) needs:
#: the cyclic time encoding and normalized position. `current_speed`,
#: `wind_speed` and `coriolis` are still available below for whoever
#: wants them (a notebook, an ablation), but the model's fixed eleven
#: channels are built from raw components, not from these derived
#: magnitudes — see `channels.py`.
DEFAULT_FEATURES: Tuple[str, ...] = (
    "cyclic_time",
    "position",
)

#: Every feature this step knows how to build, default or not.
AVAILABLE_FEATURES: Tuple[str, ...] = (
    "current_speed",
    "wind_speed",
    "cyclic_time",
    "position",
    "coriolis",
)

#: Variables that must never be used as feature inputs — these are what
#: the model is asked to reconstruct.
TARGET_VARIABLES: Tuple[str, ...] = ("subsurface_temp", "subsurface_salinity")


class FeatureBuilder(StatelessStep):
    """Add derived physical, temporal and geographic features."""

    name: ClassVar[str] = "feature_construction"
    title: ClassVar[str] = "Feature construction"

    def __init__(
        self,
        *,
        features: Sequence[str] = DEFAULT_FEATURES,
        overwrite: bool = False,
        period_days: float = 365.25,
    ) -> None:
        super().__init__()
        unknown = [f for f in features if f not in AVAILABLE_FEATURES]
        if unknown:
            raise StepConfigurationError(
                f"unknown features {unknown}; available: {list(AVAILABLE_FEATURES)}"
            )
        self.features = tuple(unique_preserving_order(features))
        self.overwrite = overwrite
        self.period_days = float(period_days)

    def config(self) -> Dict[str, Any]:
        return {
            "features": list(self.features),
            "overwrite": self.overwrite,
            "period_days": self.period_days,
        }

    # -- transform ---------------------------------------------------------

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        variables = dict(dataset.variables)
        built: list = []
        skipped: Dict[str, str] = {}

        builders = {
            "current_speed": lambda: self._magnitude(
                dataset, "u_current", "v_current", "current_speed", "m/s",
                "Surface current speed, sqrt(u^2 + v^2)",
            ),
            "wind_speed": lambda: self._magnitude(
                dataset, "u_wind", "v_wind", "wind_speed", "m/s",
                "Surface wind speed, sqrt(u^2 + v^2)",
            ),
            "cyclic_time": lambda: self._cyclic_time(dataset),
            "position": lambda: self._position(dataset),
            "coriolis": lambda: self._coriolis(dataset),
        }

        for feature in self.features:
            new_variables, reason = builders[feature]()
            if reason is not None:
                skipped[feature] = reason
                continue
            for name, variable in new_variables.items():
                if name in variables and not self.overwrite:
                    skipped[feature] = f"'{name}' already exists (overwrite=False)"
                    continue
                variables[name] = variable
                built.append(name)

        self._report = {"built": built, "skipped": skipped}
        return with_variables(dataset, variables)

    # -- individual features ----------------------------------------------

    def _magnitude(
        self,
        dataset: OceanDataset,
        u_name: str,
        v_name: str,
        out_name: str,
        units: str,
        description: str,
    ):
        if u_name not in dataset or v_name not in dataset:
            return {}, f"needs both '{u_name}' and '{v_name}'"
        u = dataset[u_name]
        v = dataset[v_name]
        if u.dims != v.dims:
            return {}, f"'{u_name}' and '{v_name}' have different dims"
        magnitude = np.hypot(
            np.asarray(u.values, dtype=float), np.asarray(v.values, dtype=float)
        ).astype(np.float32)
        return (
            {
                out_name: build_variable(
                    out_name,
                    magnitude,
                    u.dims,
                    units=units,
                    attrs={"description": description, "derived_from": [u_name, v_name]},
                )
            },
            None,
        )

    def _cyclic_time(self, dataset: OceanDataset):
        times = dataset.coords.get("time")
        if times is None or times.size == 0:
            return {}, "dataset has no time axis"
        if not np.issubdtype(np.asarray(times).dtype, np.datetime64):
            return {}, "time axis is not datetime64"
        phase = 2.0 * np.pi * (day_of_year(times) - 1) / self.period_days
        attrs = {
            "description": "Annual cycle phase encoded on the unit circle",
            "period_days": self.period_days,
        }
        return (
            {
                "time_sin": build_variable(
                    "time_sin", np.sin(phase).astype(np.float32), ("time",), units="1", attrs=attrs
                ),
                "time_cos": build_variable(
                    "time_cos", np.cos(phase).astype(np.float32), ("time",), units="1", attrs=attrs
                ),
            },
            None,
        )

    def _position(self, dataset: OceanDataset):
        lat = dataset.coords.get("lat")
        lon = dataset.coords.get("lon")
        if lat is None or lon is None:
            return {}, "dataset is not on a lat/lon grid"
        lat_grid, lon_grid = self._mesh(lat, lon)
        return (
            {
                "lat_norm": build_variable(
                    "lat_norm",
                    self._scale(lat_grid).astype(np.float32),
                    ("lat", "lon"),
                    units="1",
                    attrs={"description": "Latitude scaled to [-1, 1] across the domain"},
                ),
                "lon_norm": build_variable(
                    "lon_norm",
                    self._scale(lon_grid).astype(np.float32),
                    ("lat", "lon"),
                    units="1",
                    attrs={"description": "Longitude scaled to [-1, 1] across the domain"},
                ),
            },
            None,
        )

    def _coriolis(self, dataset: OceanDataset):
        lat = dataset.coords.get("lat")
        lon = dataset.coords.get("lon")
        if lat is None or lon is None:
            return {}, "dataset is not on a lat/lon grid"
        lat_grid, _ = self._mesh(lat, lon)
        f = 2.0 * EARTH_OMEGA * np.sin(np.deg2rad(lat_grid))
        return (
            {
                "coriolis": build_variable(
                    "coriolis",
                    f.astype(np.float32),
                    ("lat", "lon"),
                    units="1/s",
                    attrs={
                        "description": "Coriolis parameter f = 2*Omega*sin(latitude)",
                        "omega": EARTH_OMEGA,
                    },
                )
            },
            None,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _mesh(lat: np.ndarray, lon: np.ndarray):
        lon_grid, lat_grid = np.meshgrid(
            np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        )
        return lat_grid, lon_grid

    @staticmethod
    def _scale(values: np.ndarray) -> np.ndarray:
        """Scale to [-1, 1] over the array's own range; constant axes give 0."""
        low = float(np.min(values))
        high = float(np.max(values))
        if high <= low:
            return np.zeros_like(values, dtype=float)
        return 2.0 * (values - low) / (high - low) - 1.0
