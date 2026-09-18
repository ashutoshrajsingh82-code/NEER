"""
Stage 7 — normalization.

The variables NEER stacks into one tensor differ by orders of magnitude:
sea surface temperature runs to ~30, sea level anomaly to ~0.3, the
Coriolis parameter to ~7e-5. Fed raw into a network, the largest-scaled
channel dominates the gradients and the smallest contributes nothing.
Normalization puts every channel on comparable footing.

This is the second of the two places leakage could enter the pipeline,
and the more commonly botched one: computing a mean and standard
deviation over the *whole* record before splitting quietly tells the
model the test period's climate. The statistics here are fitted on the
training split only, stored, and then applied unchanged to validation and
test data — which is also what happens at inference time, when there is
no future to compute statistics over.

Per-level statistics for 4D fields
----------------------------------
For a variable with a depth axis, statistics are computed **per depth
level** by default. Sea temperature at 1000 m is a different distribution
from temperature at the surface — a single mean and standard deviation
across the whole column would leave the deep levels compressed into a
sliver of the normalized range, and reconstruction there is precisely
what NEER is for.

Inverse transform
-----------------
`inverse_transform` exists and matters: a model predicts in normalized
units, and every prediction has to come back to degrees Celsius before
anyone can read it, evaluate it, or plot it. The statistics are saved
with the preprocessing metadata so a later phase can invert predictions
without re-running this stage.

Masks (`land_mask`, `ocean_mask`, `<var>_observed`) and, by default, the
already-bounded cyclic/position features are left alone.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np

from src.data.loaders.representation import OceanDataset, Variable
from src.data.preprocessing._utils import (
    axis_of,
    data_variables,
    json_safe,
    nan_mean,
    nan_std,
    with_values,
    with_variables,
)
from src.data.preprocessing.base import LearnedStep
from src.data.preprocessing.errors import StepConfigurationError

#: Supported normalization methods.
METHODS: Tuple[str, ...] = ("zscore", "minmax", "robust")

#: Features that are already bounded and carry meaning in their own units;
#: normalizing them adds nothing and makes them harder to interpret.
DEFAULT_EXCLUDE: Tuple[str, ...] = ("time_sin", "time_cos", "lat_norm", "lon_norm")

#: Guard against dividing by a near-zero spread (a constant field).
MIN_SCALE = 1e-8


class Normalizer(LearnedStep):
    """Scale variables using statistics fitted on the training split only."""

    name: ClassVar[str] = "normalization"
    title: ClassVar[str] = "Normalization"

    def __init__(
        self,
        *,
        method: str = "zscore",
        per_depth_level: bool = True,
        variables: Optional[Sequence[str]] = None,
        exclude: Sequence[str] = DEFAULT_EXCLUDE,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise StepConfigurationError(
                f"method must be one of {list(METHODS)}, got {method!r}"
            )
        self.method = method
        self.per_depth_level = per_depth_level
        self.variables = tuple(variables) if variables else None
        self.exclude = tuple(exclude)

        #: variable -> {"center": array, "scale": array, "depth_axis": int|None}
        self._stats: Dict[str, Dict[str, Any]] = {}

    def config(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "per_depth_level": self.per_depth_level,
            "variables": list(self.variables) if self.variables else None,
            "exclude": list(self.exclude),
        }

    def state(self) -> Dict[str, Any]:
        """The fitted statistics — small enough to store in full, and worth it.

        These are what `inverse_transform` needs, so a later phase can
        put predictions back into physical units from the metadata file
        alone. `center`/`scale` are the method-agnostic pair every method
        produces (what `_transform` actually subtracts and divides by);
        `mean`/`std` (zscore) and `min`/`max` (minmax) are added under
        their familiar names too, so the saved metadata is self-
        documenting without having to know which method produced it.
        """

        def named(stat: Dict[str, Any]) -> Dict[str, Any]:
            center = np.asarray(stat["center"]).ravel().tolist()
            scale = np.asarray(stat["scale"]).ravel().tolist()
            entry = {
                "center": center,
                "scale": scale,
                "per_depth_level": stat["depth_axis"] is not None,
                "units": stat.get("units"),
            }
            if self.method == "zscore":
                entry["mean"] = center
                entry["std"] = scale
            elif self.method == "minmax":
                entry["min"] = center
                entry["max"] = [c + s for c, s in zip(center, scale)]
            return entry

        return {
            "method": self.method,
            "statistics": json_safe(
                {name: named(stat) for name, stat in self._stats.items()}
            ),
        }

    # -- selection ---------------------------------------------------------

    def _targets(self, dataset: OceanDataset) -> Dict[str, Variable]:
        candidates = data_variables(dataset)
        if self.variables is not None:
            missing = [n for n in self.variables if n not in candidates]
            if missing:
                raise StepConfigurationError(
                    f"variables not present (or are masks): {missing}"
                )
            return {name: candidates[name] for name in self.variables}
        return {n: v for n, v in candidates.items() if n not in self.exclude}

    # -- fit ---------------------------------------------------------------

    def _fit(self, dataset: OceanDataset) -> None:
        self._stats = {}
        for name, variable in self._targets(dataset).items():
            values = np.asarray(variable.values, dtype=float)
            depth_axis = axis_of(variable, "depth") if self.per_depth_level else None
            axes = (
                None
                if depth_axis is None
                else tuple(a for a in range(values.ndim) if a != depth_axis)
            )
            center, scale = self._statistics(values, axes)
            self._stats[name] = {
                "center": np.asarray(center, dtype=np.float32),
                "scale": np.asarray(scale, dtype=np.float32),
                "depth_axis": depth_axis,
                "units": variable.units,
            }

    def _statistics(self, values: np.ndarray, axes) -> Tuple[np.ndarray, np.ndarray]:
        if self.method == "zscore":
            center = nan_mean(values, axis=axes)
            scale = nan_std(values, axis=axes)
        elif self.method == "minmax":
            with np.errstate(invalid="ignore"):
                low = np.nanmin(values, axis=axes) if np.any(np.isfinite(values)) else np.nan
                high = np.nanmax(values, axis=axes) if np.any(np.isfinite(values)) else np.nan
            center = low
            scale = np.asarray(high, dtype=float) - np.asarray(low, dtype=float)
        else:  # robust: median / IQR, for fields with outliers
            with np.errstate(invalid="ignore"):
                center = np.nanmedian(values, axis=axes)
                q75 = np.nanpercentile(values, 75, axis=axes)
                q25 = np.nanpercentile(values, 25, axis=axes)
            scale = np.asarray(q75, dtype=float) - np.asarray(q25, dtype=float)

        center = np.nan_to_num(np.asarray(center, dtype=float), nan=0.0)
        scale = np.asarray(scale, dtype=float)
        scale = np.where(np.isfinite(scale) & (np.abs(scale) > MIN_SCALE), scale, 1.0)
        return center, scale

    # -- transform ---------------------------------------------------------

    def _shaped(self, stat: Dict[str, Any], key: str, variable: Variable) -> np.ndarray:
        value = np.asarray(stat[key], dtype=float)
        depth_axis = stat["depth_axis"]
        if depth_axis is None or value.ndim == 0:
            return value
        shape = [1] * variable.ndim
        shape[depth_axis] = value.size
        return value.reshape(shape)

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        self.require_fitted("no statistics have been computed")
        variables = dict(dataset.variables)
        applied = []

        for name, variable in self._targets(dataset).items():
            stat = self._stats.get(name)
            if stat is None:
                continue
            center = self._shaped(stat, "center", variable)
            scale = self._shaped(stat, "scale", variable)
            values = (np.asarray(variable.values, dtype=float) - center) / scale
            variables[name] = with_values(
                variable,
                values.astype(np.float32),
                attrs={
                    **variable.attrs,
                    "normalization": self.method,
                    "physical_units": variable.units,
                },
                units="normalized",
            )
            applied.append(name)

        self._report = {"method": self.method, "normalized": applied}
        return with_variables(dataset, variables)

    # -- inverse -----------------------------------------------------------

    def inverse_transform_array(self, name: str, values: np.ndarray, dims: Sequence[str]):
        """Put a normalized array back into physical units.

        `dims` is needed to know where the depth axis sits when the
        statistics are per-level — model outputs do not carry dims with
        them, so the caller supplies them.
        """
        stat = self._stats.get(name)
        if stat is None:
            raise StepConfigurationError(f"no fitted statistics for '{name}'")
        array = np.asarray(values, dtype=float)
        center = np.asarray(stat["center"], dtype=float)
        scale = np.asarray(stat["scale"], dtype=float)
        depth_axis = None if stat["depth_axis"] is None else list(dims).index("depth")
        if depth_axis is not None and center.ndim:
            shape = [1] * array.ndim
            shape[depth_axis] = center.size
            center = center.reshape(shape)
            scale = scale.reshape(shape)
        return array * scale + center

    def inverse_transform(self, dataset: OceanDataset) -> OceanDataset:
        """Undo normalization for every variable this step normalized."""
        self.require_fitted()
        variables = dict(dataset.variables)
        for name, stat in self._stats.items():
            variable = dataset.variables.get(name)
            if variable is None:
                continue
            center = self._shaped(stat, "center", variable)
            scale = self._shaped(stat, "scale", variable)
            values = np.asarray(variable.values, dtype=float) * scale + center
            variables[name] = with_values(
                variable, values.astype(np.float32), units=stat.get("units")
            )
        return with_variables(dataset, variables)

    # -- convenience -------------------------------------------------------

    def statistics(self, name: str) -> Optional[Dict[str, Any]]:
        """The fitted `{center, scale, ...}` for `name`."""
        return self._stats.get(name)
