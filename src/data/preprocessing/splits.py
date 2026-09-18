"""
Chronological train/validation/test splits, and the leakage guard.

Why chronological, not random
-----------------------------
NEER reconstructs subsurface ocean state from surface fields, and ocean
fields are strongly autocorrelated in time: the SST field one month after
a given month is nearly the same field. A random split would put
near-duplicate timesteps on both sides of the boundary, and every
reported score would be optimistic for reasons that have nothing to do
with the model. So the split is a cut along the time axis — train comes
first, then validation, then test — and it is the only split this module
offers.

Splits are defined by *timestamps*, not by indices
--------------------------------------------------
`train_end` and `val_end` are datetimes, and `masks()` derives the index
sets from whatever time axis it is handed. That matters because temporal
alignment can resample the time axis: index 14 is not a stable identity,
while "everything before 2021-07-01" is. The pipeline consequently
aligns the time axis *before* deciding the split, and the same
`TemporalSplit` object then applies unambiguously to the aligned data,
to the assembled tensors, and to anything a later phase loads from disk.

The guard
---------
`assert_fit_window` is called by the pipeline every time it fits a step
that learns from data. It re-derives, from the timestamps actually
handed to that step, whether any of them fall in the validation or test
period, and raises `LeakageError` if so. It is cheap, and it means the
no-leakage property is enforced at run time rather than asserted in a
comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from src.data.preprocessing.errors import LeakageError, PreprocessingError

#: Default share of the time axis held out, in chronological order.
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15


def _as_datetime64(value: Any) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    return np.datetime64(value, "ns")


@dataclass(frozen=True)
class TemporalSplit:
    """A chronological cut of the time axis into train / val / test.

    Both boundaries are **exclusive upper bounds**:

        train : time <  train_end
        val   : train_end <= time < val_end
        test  : time >= val_end

    An empty validation or test period is allowed (set `val_end` equal to
    `train_end`, or beyond the end of the data) — small experiments
    sometimes want a plain train/test cut — but train must never be
    empty, since there would be nothing to fit on.
    """

    train_end: np.datetime64
    val_end: np.datetime64

    def __post_init__(self) -> None:
        train_end = _as_datetime64(self.train_end)
        val_end = _as_datetime64(self.val_end)
        if val_end < train_end:
            raise PreprocessingError(
                f"val_end ({val_end}) must not be earlier than train_end ({train_end})"
            )
        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "val_end", val_end)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_fractions(
        cls,
        times: np.ndarray,
        *,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        test_fraction: float = DEFAULT_TEST_FRACTION,
    ) -> "TemporalSplit":
        """Cut the *last* `val_fraction + test_fraction` of the time axis off.

        The boundaries land on actual timestamps, so no timestep can fall
        between two periods, and each period gets at least one timestep
        whenever the axis is long enough to allow it.
        """
        times = np.sort(np.asarray(times).astype("datetime64[ns]"))
        n = times.size
        if n == 0:
            raise PreprocessingError("cannot build a split from an empty time axis")
        if not (0 <= val_fraction < 1) or not (0 <= test_fraction < 1):
            raise PreprocessingError("fractions must be in [0, 1)")
        if val_fraction + test_fraction >= 1:
            raise PreprocessingError(
                "val_fraction + test_fraction must leave some training data"
            )

        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        if test_fraction > 0:
            n_test = max(1, n_test)
        if val_fraction > 0:
            n_val = max(1, n_val)
        # Always leave at least one training timestep.
        while n_val + n_test >= n:
            if n_test > n_val:
                n_test -= 1
            else:
                n_val -= 1

        n_train = n - n_val - n_test
        train_end = times[n_train] if n_train < n else times[-1] + np.timedelta64(1, "ns")
        val_index = n_train + n_val
        val_end = times[val_index] if val_index < n else times[-1] + np.timedelta64(1, "ns")
        return cls(train_end=train_end, val_end=val_end)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalSplit":
        """Rebuild a split from its `to_dict()` form."""
        return cls(train_end=data["train_end"], val_end=data["val_end"])

    # -- application -------------------------------------------------------

    def masks(self, times: np.ndarray) -> Dict[str, np.ndarray]:
        """`{'train': mask, 'val': mask, 'test': mask}` over `times`."""
        times = np.asarray(times).astype("datetime64[ns]")
        train = times < self.train_end
        val = (times >= self.train_end) & (times < self.val_end)
        test = times >= self.val_end
        return {"train": train, "val": val, "test": test}

    def indices(self, times: np.ndarray) -> Dict[str, np.ndarray]:
        """`{'train': idx, 'val': idx, 'test': idx}` as integer index arrays."""
        return {name: np.flatnonzero(mask) for name, mask in self.masks(times).items()}

    def train_mask(self, times: np.ndarray) -> np.ndarray:
        return self.masks(times)["train"]

    def counts(self, times: np.ndarray) -> Dict[str, int]:
        """How many timesteps land in each period."""
        return {name: int(mask.sum()) for name, mask in self.masks(times).items()}

    def is_training_time(self, times: np.ndarray) -> np.ndarray:
        """Elementwise: does each timestamp belong to the training period?"""
        return np.asarray(times).astype("datetime64[ns]") < self.train_end

    # -- reporting ---------------------------------------------------------

    def to_dict(self, times: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """JSON-safe description, including per-period counts when `times` is given."""
        data: Dict[str, Any] = {
            "strategy": "chronological",
            "train_end_exclusive": str(self.train_end),
            "val_end_exclusive": str(self.val_end),
        }
        if times is not None and np.asarray(times).size:
            times = np.asarray(times).astype("datetime64[ns]")
            data["counts"] = self.counts(times)
            ranges = {}
            for period, mask in self.masks(times).items():
                selected = times[mask]
                ranges[period] = (
                    None
                    if selected.size == 0
                    else [str(selected.min()), str(selected.max())]
                )
            data["ranges"] = ranges
        return data

    def describe(self, times: Optional[np.ndarray] = None) -> str:
        if times is None:
            return f"chronological split: train < {self.train_end} <= val < {self.val_end} <= test"
        counts = self.counts(times)
        return (
            f"chronological split: train={counts['train']} "
            f"val={counts['val']} test={counts['test']} timesteps "
            f"(train < {self.train_end} <= val < {self.val_end} <= test)"
        )


# --------------------------------------------------------------------------
# The leakage guard
# --------------------------------------------------------------------------


def assert_fit_window(
    split: TemporalSplit, times: Iterable[Any], *, step: str = "pipeline"
) -> None:
    """Raise `LeakageError` unless every timestamp in `times` is training data.

    Called before fitting anything that learns from data. The check is on
    the timestamps the step was actually handed, not on what the caller
    intended to hand it, which is the only version of the check worth
    having.
    """
    times = np.asarray(list(times)).astype("datetime64[ns]")
    if times.size == 0:
        raise LeakageError(
            f"'{step}' was asked to fit on an empty time axis; there is no training data "
            "in the requested split."
        )
    offending = times[~split.is_training_time(times)]
    if offending.size:
        raise LeakageError(
            f"'{step}' was about to be fitted on {offending.size} timestep(s) outside the "
            f"training period (training ends before {split.train_end}). Fit on the "
            "training split only.",
            offending_times=offending,
        )


def split_summary(split: TemporalSplit, times: np.ndarray) -> Dict[str, Any]:
    """Split description plus the training time range, for metadata."""
    times = np.asarray(times).astype("datetime64[ns]")
    train_times = times[split.train_mask(times)]
    return {
        **split.to_dict(times),
        "fitted_on": {
            "n_timesteps": int(train_times.size),
            "first": None if train_times.size == 0 else str(train_times.min()),
            "last": None if train_times.size == 0 else str(train_times.max()),
        },
    }


def check_disjoint(split: TemporalSplit, times: np.ndarray) -> Tuple[bool, str]:
    """Sanity check that the three periods partition `times` exactly once.

    Returns `(ok, message)`. Used by the pipeline's self-check and by the
    tests; a False here would mean the mask logic itself is broken.
    """
    times = np.asarray(times).astype("datetime64[ns]")
    masks = split.masks(times)
    stacked = np.vstack([masks["train"], masks["val"], masks["test"]])
    coverage = stacked.sum(axis=0)
    if not np.all(coverage == 1):
        return False, (
            f"{int((coverage != 1).sum())} timestep(s) are not assigned to exactly one period"
        )
    if masks["train"].sum() == 0:
        return False, "the training period is empty"
    return True, "train/val/test partition the time axis exactly once"
