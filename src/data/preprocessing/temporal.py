"""
Stage 2 — temporal alignment.

NEER's inputs do not arrive on a common clock. Satellite SST is daily,
altimetry is daily but on a different overpass time, ARGO profiles are
whenever a float happened to surface, and reanalysis products are monthly
means stamped anywhere from the 1st to the 16th of the month. Before any
of it can be stacked into a tensor with a shared time axis, all of it has
to be snapped to one cadence.

This step does three things, in order:

1. **Snap** every timestamp to the start of its period (month, day,
   year). Two products that both describe January 2021 but stamp it
   2021-01-01 and 2021-01-16 become the same timestep.
2. **Aggregate** timestamps that collide after snapping, with a
   NaN-aware mean by default: four weekly fields inside one month become
   one monthly field, and a cell missing in two of the four weeks still
   gets the mean of the two it has.
3. **Reindex** onto a complete, regular axis when `fill_gaps=True`, so a
   month with no data at all becomes an explicitly all-NaN timestep
   rather than a silent discontinuity. That distinction matters: the
   missing-value stage can only interpolate across a gap it can see.

Aggregation uses a NaN-aware mean, never a plain mean, because in this
domain missing is the normal case (cloud cover, no float in the box) and
`np.mean` would propagate a single missing week into a missing month.

Nothing here is learned from the data — it is arithmetic on the time
axis — so the step is safe to apply before the train/test split, which is
exactly what the pipeline does: the split is decided on the *aligned*
axis, where timesteps have stable identities.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np

from src.data.loaders.representation import OceanDataset, Variable
from src.data.preprocessing._utils import (
    axis_of,
    is_mask_variable,
    nan_mean,
    with_values,
    with_variables,
)
from src.data.preprocessing.base import StatelessStep
from src.data.preprocessing.errors import StepConfigurationError

#: Supported cadences, mapped onto the NumPy datetime64 unit that snaps to them.
FREQUENCIES: Dict[str, str] = {
    "yearly": "Y",
    "monthly": "M",
    "daily": "D",
    "hourly": "h",
}

#: Supported ways of combining timestamps that collide after snapping.
AGGREGATIONS: Tuple[str, ...] = ("mean", "first", "last")


def snap_times(times: np.ndarray, frequency: str) -> np.ndarray:
    """Snap datetimes to the start of their period at `frequency`."""
    if frequency not in FREQUENCIES:
        raise StepConfigurationError(
            f"unknown frequency {frequency!r}; expected one of {sorted(FREQUENCIES)}"
        )
    unit = FREQUENCIES[frequency]
    return np.asarray(times).astype(f"datetime64[{unit}]").astype("datetime64[ns]")


def regular_axis(start: np.datetime64, end: np.datetime64, frequency: str) -> np.ndarray:
    """A complete, gap-free time axis from `start` to `end` inclusive."""
    unit = FREQUENCIES[frequency]
    first = np.datetime64(start, unit)
    last = np.datetime64(end, unit)
    step = np.timedelta64(1, unit)
    return np.arange(first, last + step, step).astype("datetime64[ns]")


class TemporalAligner(StatelessStep):
    """Put every variable on one regular time axis at a common cadence."""

    name: ClassVar[str] = "temporal_alignment"
    title: ClassVar[str] = "Temporal alignment"

    def __init__(
        self,
        *,
        frequency: str = "monthly",
        aggregation: str = "mean",
        fill_gaps: bool = True,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if frequency not in FREQUENCIES:
            raise StepConfigurationError(
                f"frequency must be one of {sorted(FREQUENCIES)}, got {frequency!r}"
            )
        if aggregation not in AGGREGATIONS:
            raise StepConfigurationError(
                f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}"
            )
        self.frequency = frequency
        self.aggregation = aggregation
        self.fill_gaps = fill_gaps
        self.start = None if start is None else np.datetime64(start, "ns")
        self.end = None if end is None else np.datetime64(end, "ns")

    def config(self) -> Dict[str, Any]:
        return {
            "frequency": self.frequency,
            "aggregation": self.aggregation,
            "fill_gaps": self.fill_gaps,
            "start": None if self.start is None else str(self.start),
            "end": None if self.end is None else str(self.end),
        }

    # -- transform ---------------------------------------------------------

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        times = dataset.coords.get("time")
        if times is None or times.size == 0:
            self._report = {"skipped": "dataset has no time axis"}
            return dataset

        snapped = snap_times(times, self.frequency)
        target = self._target_axis(snapped)

        # Which source timesteps feed each target timestep.
        groups = [np.flatnonzero(snapped == stamp) for stamp in target]

        variables = {}
        for name, variable in dataset.variables.items():
            axis = axis_of(variable, "time")
            if axis is None:
                variables[name] = variable
                continue
            variables[name] = self._aggregate(variable, axis, groups)

        coords = dict(dataset.coords)
        coords["time"] = target

        self._report = {
            "frequency": self.frequency,
            "aggregation": self.aggregation,
            "n_time_in": int(times.size),
            "n_time_out": int(target.size),
            "n_collisions": int(times.size - np.unique(snapped).size),
            "n_gap_steps_inserted": int(sum(1 for g in groups if g.size == 0)),
            "time_range": [str(target.min()), str(target.max())] if target.size else None,
            "snapped": bool(not np.array_equal(np.asarray(times), snapped)),
        }
        return with_variables(dataset, variables, coords=coords)

    # -- internals ---------------------------------------------------------

    def _target_axis(self, snapped: np.ndarray) -> np.ndarray:
        """The output time axis: complete and regular, or just the unique stamps."""
        unique = np.unique(snapped)
        start = self.start if self.start is not None else unique.min()
        end = self.end if self.end is not None else unique.max()
        if self.fill_gaps:
            return regular_axis(start, end, self.frequency)
        within = unique[(unique >= np.datetime64(start, "ns")) & (unique <= np.datetime64(end, "ns"))]
        return within

    def _aggregate(self, variable: Variable, axis: int, groups) -> Variable:
        """Combine each group of source timesteps into one output timestep."""
        values = variable.values
        is_mask = is_mask_variable(variable.name, variable)

        out_shape = list(values.shape)
        out_shape[axis] = len(groups)

        if is_mask:
            out = np.zeros(out_shape, dtype=bool)
        else:
            out = np.full(out_shape, np.nan, dtype=np.result_type(values.dtype, np.float32))

        for position, indices in enumerate(groups):
            if indices.size == 0:
                continue  # gap step: stays NaN (or False for masks)
            block = np.take(values, indices, axis=axis)
            if is_mask:
                combined = np.any(block, axis=axis)
            elif self.aggregation == "mean":
                combined = nan_mean(block.astype(float), axis=axis)
            elif self.aggregation == "first":
                combined = np.take(block, 0, axis=axis)
            else:  # "last"
                combined = np.take(block, block.shape[axis] - 1, axis=axis)
            index = [slice(None)] * out.ndim
            index[axis] = position
            out[tuple(index)] = combined

        return with_values(variable, out)
