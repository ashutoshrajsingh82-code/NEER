"""
Stage 4 — missing-value handling.

Ocean data is mostly holes. Infrared SST is missing wherever there was
cloud; altimetry is missing between tracks; subsurface temperature is
missing everywhere an ARGO float has not drifted, which on any given
month is most of the basin. The demo dataset reproduces that on purpose.
A model cannot consume NaN, so the gaps have to be filled — and *how*
they are filled is one of the two places in this pipeline where test data
could leak into training.

The fill cascade
----------------
Each strategy is tried in order on whatever is still missing:

1. ``temporal_interpolate`` — linear interpolation along the time axis
   between the nearest observed values on either side, but only across
   gaps of at most `max_gap` steps. A two-month gap between two good
   observations is worth interpolating; a fourteen-month one is an
   invention, so it is left for the climatology.
2. ``climatology`` — the mean for that cell in that calendar month,
   computed **from the training period only**. Monthly rather than
   overall, because the seasonal cycle dominates every field here:
   filling a January gap with the annual mean would insert a warm bias
   of several degrees.
3. ``global_mean`` — the variable's training-period mean (per depth
   level for 4D fields, where the vertical gradient is far larger than
   any horizontal one), as a last resort for cells with no training
   observation at all in that month.

Anything still missing after the cascade stays NaN and is handled by the
masking stage: an honest hole, flagged as such, beats a fabricated value.

What is not filled
------------------
Land. Filling land cells with an ocean climatology would put physically
meaningless numbers into the tensor — and, worse, numbers that look
plausible. When the dataset carries a land/ocean mask, filling is
restricted to ocean cells and land is left NaN for the masking stage.

Leakage
-------
The climatology and the global means are the fitted state of this step.
`fit()` must be called on the training split, and `transform()` refuses
to run before it has been (`NotFittedError`). The pipeline enforces the
window with `splits.assert_fit_window`, and
`tests/test_preprocessing_leakage.py` holds the line by corrupting the
test period and asserting the fitted state does not move.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from src.data.loaders.representation import OceanDataset, Variable, build_variable
from src.data.preprocessing._utils import (
    axis_of,
    broadcast_spatial,
    data_variables,
    is_mask_variable,
    month_of,
    nan_mean,
    ocean_mask_of,
    summarize_array,
    with_values,
    with_variables,
)
from src.data.preprocessing.base import LearnedStep
from src.data.preprocessing.errors import StepConfigurationError

#: Fill strategies, applied in the order given.
STRATEGIES: Tuple[str, ...] = ("temporal_interpolate", "climatology", "global_mean")

DEFAULT_STRATEGIES: Tuple[str, ...] = STRATEGIES

#: Longest run of consecutive missing timesteps that linear interpolation
#: is allowed to bridge.
DEFAULT_MAX_GAP = 2

#: Suffix of the companion mask marking which cells were actually observed.
OBSERVED_SUFFIX = "_observed"


def interpolate_along_time(
    values: np.ndarray, axis: int, *, max_gap: int = DEFAULT_MAX_GAP
) -> Tuple[np.ndarray, int]:
    """Linearly interpolate missing values along `axis`, across gaps <= `max_gap`.

    No extrapolation: a cell missing at the start or the end of the
    record keeps its NaN, because there is nothing to interpolate
    *between*. Returns the filled array and how many cells were filled.
    """
    moved = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    n_time = moved.shape[0]
    if n_time < 2 or max_gap <= 0:
        return np.asarray(values, dtype=float), 0

    valid = np.isfinite(moved)
    steps = np.arange(n_time).reshape((n_time,) + (1,) * (moved.ndim - 1))

    # Index of the nearest observed step before/after each position.
    previous = np.maximum.accumulate(np.where(valid, steps, -1), axis=0)
    following = np.flip(
        np.minimum.accumulate(np.flip(np.where(valid, steps, n_time), axis=0), axis=0), axis=0
    )

    bridgeable = (
        ~valid
        & (previous >= 0)
        & (following < n_time)
        & ((following - previous - 1) <= max_gap)
    )
    if not np.any(bridgeable):
        return np.asarray(values, dtype=float), 0

    low_index = np.clip(previous, 0, n_time - 1)
    high_index = np.clip(following, 0, n_time - 1)
    low = np.take_along_axis(moved, low_index, axis=0)
    high = np.take_along_axis(moved, high_index, axis=0)
    span = np.where(following - previous > 0, following - previous, 1)
    weight = (steps - previous) / span

    filled = np.where(bridgeable, low * (1.0 - weight) + high * weight, moved)
    return np.moveaxis(filled, 0, axis), int(bridgeable.sum())


class MissingValueHandler(LearnedStep):
    """Fill gaps with a cascade of train-fitted strategies; never over land."""

    name: ClassVar[str] = "missing_values"
    title: ClassVar[str] = "Missing-value handling"

    def __init__(
        self,
        *,
        strategies: Sequence[str] = DEFAULT_STRATEGIES,
        max_gap: int = DEFAULT_MAX_GAP,
        emit_observed_masks: bool = True,
        ocean_only: bool = True,
        mask_variable: str = "land_mask",
        variables: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        unknown = [s for s in strategies if s not in STRATEGIES]
        if unknown:
            raise StepConfigurationError(
                f"unknown fill strategies {unknown}; expected any of {list(STRATEGIES)}"
            )
        self.strategies = tuple(strategies)
        self.max_gap = int(max_gap)
        self.emit_observed_masks = emit_observed_masks
        self.ocean_only = ocean_only
        self.mask_variable = mask_variable
        self.variables = tuple(variables) if variables else None

        #: variable -> (12, *spatial) monthly climatology, training period only
        self._climatology: Dict[str, np.ndarray] = {}
        #: variable -> scalar or (n_depth,) training-period mean
        self._global_mean: Dict[str, np.ndarray] = {}
        #: variable -> dims of the climatology field (minus the leading month axis)
        self._climatology_dims: Dict[str, Tuple[str, ...]] = {}

    def config(self) -> Dict[str, Any]:
        return {
            "strategies": list(self.strategies),
            "max_gap": self.max_gap,
            "emit_observed_masks": self.emit_observed_masks,
            "ocean_only": self.ocean_only,
            "mask_variable": self.mask_variable,
            "variables": list(self.variables) if self.variables else None,
        }

    def state(self) -> Dict[str, Any]:
        """Summaries of the fitted climatologies — not the fields themselves."""
        return {
            "climatology": {
                name: {
                    "shape": list(field.shape),
                    "dims": ["month", *self._climatology_dims.get(name, ())],
                    **summarize_array(field),
                }
                for name, field in self._climatology.items()
            },
            "global_mean": {
                name: summarize_array(np.atleast_1d(value))
                for name, value in self._global_mean.items()
            },
        }

    # -- fit ---------------------------------------------------------------

    def _targets(self, dataset: OceanDataset) -> Dict[str, Variable]:
        candidates = data_variables(dataset)
        if self.variables is not None:
            missing = [n for n in self.variables if n not in candidates]
            if missing:
                raise StepConfigurationError(
                    f"variables not present (or are masks): {missing}"
                )
            return {name: candidates[name] for name in self.variables}
        return candidates

    def _fit(self, dataset: OceanDataset) -> None:
        times = dataset.coords.get("time")
        self._climatology = {}
        self._global_mean = {}
        self._climatology_dims = {}

        months = None if times is None else month_of(times)

        for name, variable in self._targets(dataset).items():
            values = np.asarray(variable.values, dtype=float)
            time_axis = axis_of(variable, "time")

            if time_axis is not None and months is not None:
                moved = np.moveaxis(values, time_axis, 0)
                climatology = np.full((12,) + moved.shape[1:], np.nan, dtype=np.float32)
                for month in range(1, 13):
                    selected = moved[months == month]
                    if selected.size:
                        climatology[month - 1] = nan_mean(selected, axis=0)
                self._climatology[name] = climatology
                self._climatology_dims[name] = tuple(
                    d for d in variable.dims if d != "time"
                )

            depth_axis = axis_of(variable, "depth")
            if depth_axis is not None:
                other = tuple(a for a in range(values.ndim) if a != depth_axis)
                self._global_mean[name] = np.asarray(
                    nan_mean(values, axis=other), dtype=np.float32
                )
            else:
                self._global_mean[name] = np.asarray(
                    nan_mean(values), dtype=np.float32
                )

    # -- transform ---------------------------------------------------------

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        self.require_fitted("no climatology has been computed")
        times = dataset.coords.get("time")
        months = None if times is None else month_of(times)
        ocean = ocean_mask_of(dataset, self.mask_variable) if self.ocean_only else None

        variables = dict(dataset.variables)
        report: Dict[str, Any] = {"variables": {}, "ocean_only": ocean is not None}

        for name, variable in self._targets(dataset).items():
            values = np.asarray(variable.values, dtype=float)
            observed = np.isfinite(values)

            fillable = ~observed
            if ocean is not None and {"lat", "lon"} <= set(variable.dims):
                ocean_broadcast = broadcast_spatial(ocean, variable.dims, ("lat", "lon"))
                fillable = fillable & ocean_broadcast

            counts = {"missing_before": int((~observed).sum()), "fillable": int(fillable.sum())}
            filled = values

            for strategy in self.strategies:
                still = ~np.isfinite(filled) & fillable
                if not np.any(still):
                    break
                if strategy == "temporal_interpolate":
                    filled, n = self._fill_temporal(filled, variable, fillable)
                elif strategy == "climatology":
                    filled, n = self._fill_climatology(filled, variable, name, months, fillable)
                else:
                    filled, n = self._fill_global(filled, variable, name, fillable)
                counts[f"filled_{strategy}"] = int(n)

            remaining = ~np.isfinite(filled)
            counts["missing_after"] = int(remaining.sum())
            counts["missing_after_fillable"] = int((remaining & fillable).sum())
            n_filled = counts["missing_before"] - counts["missing_after"]
            counts["n_filled"] = int(n_filled)
            counts["fill_fraction"] = (
                0.0 if values.size == 0 else round(n_filled / values.size, 8)
            )
            report["variables"][name] = counts

            variables[name] = with_values(variable, filled.astype(np.float32))

            if self.emit_observed_masks:
                mask_name = f"{name}{OBSERVED_SUFFIX}"
                variables[mask_name] = build_variable(
                    mask_name,
                    observed,
                    variable.dims,
                    units="bool",
                    attrs={
                        "description": (
                            f"True where '{name}' was actually observed, before any gap filling"
                        ),
                        "source_variable": name,
                    },
                )

        self._report = report
        return with_variables(dataset, variables)

    # -- individual strategies --------------------------------------------

    def _fill_temporal(
        self, values: np.ndarray, variable: Variable, fillable: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        axis = axis_of(variable, "time")
        if axis is None:
            return values, 0
        interpolated, _ = interpolate_along_time(values, axis, max_gap=self.max_gap)
        use = ~np.isfinite(values) & fillable & np.isfinite(interpolated)
        if not np.any(use):
            return values, 0
        out = np.where(use, interpolated, values)
        return out, int(use.sum())

    def _fill_climatology(
        self,
        values: np.ndarray,
        variable: Variable,
        name: str,
        months: Optional[np.ndarray],
        fillable: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        climatology = self._climatology.get(name)
        time_axis = axis_of(variable, "time")
        if climatology is None or months is None or time_axis is None:
            return values, 0

        moved = np.moveaxis(values, time_axis, 0)
        fill_moved = np.moveaxis(fillable, time_axis, 0)
        # climatology is (12, *spatial) in the same spatial order as `moved`.
        candidate = climatology[np.clip(months, 1, 12) - 1].astype(float)
        use = ~np.isfinite(moved) & fill_moved & np.isfinite(candidate)
        if not np.any(use):
            return values, 0
        out = np.where(use, candidate, moved)
        return np.moveaxis(out, 0, time_axis), int(use.sum())

    def _fill_global(
        self, values: np.ndarray, variable: Variable, name: str, fillable: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        mean = self._global_mean.get(name)
        if mean is None:
            return values, 0
        mean = np.asarray(mean, dtype=float)
        depth_axis = axis_of(variable, "depth")
        if mean.ndim == 1 and depth_axis is not None:
            shape = [1] * values.ndim
            shape[depth_axis] = mean.size
            candidate = mean.reshape(shape)
        else:
            candidate = np.full((), float(mean.reshape(-1)[0]) if mean.size else np.nan)
        candidate = np.broadcast_to(candidate, values.shape)
        use = ~np.isfinite(values) & fillable & np.isfinite(candidate)
        if not np.any(use):
            return values, 0
        return np.where(use, candidate, values), int(use.sum())

    # -- convenience -------------------------------------------------------

    def climatology(self, name: str) -> Optional[np.ndarray]:
        """The fitted `(12, *spatial)` monthly climatology for `name`."""
        return self._climatology.get(name)

    def fitted_variables(self) -> Iterable[str]:
        return tuple(self._global_mean)
