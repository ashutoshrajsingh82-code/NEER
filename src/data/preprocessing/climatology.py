"""
Phase 11 — monthly temperature climatology.

`MissingValueHandler` (Phase 07) already computes a monthly climatology,
but only as an implementation detail of gap-filling — it is private
state used to patch holes, not something anything else can query. This
module promotes the same idea, computed the same train-only way, into a
first-class, independently reusable object: `MonthlyClimatology`.

What this is for
-----------------
A neural model reconstructing ocean temperature does better predicting
the *anomaly* than the raw field: the seasonal cycle at a given point
(warm in June, cool in January) is large, predictable, and has nothing
to do with the signal a model is meant to learn. Splitting temperature
into a monthly-mean climatology plus a residual lets the model spend its
capacity on the residual:

    delta_T = target_temperature - climatology          (training target)
    temperature_prediction = climatology + predicted_delta_T   (inference)

Both are one-line functions (`compute_delta_temperature` /
`reconstruct_temperature`) because the mathematical relationship is
exactly that simple and exactly invertible — `tests/test_climatology.py`
holds that identity as its central claim. Everything else in this module
exists to produce the `climatology` term those two functions need:
fitted from training data only, and queryable by month, by location, and
by depth.

Train-only, like every learned step
------------------------------------
`MonthlyClimatology` is a `LearnedStep`: `fit()` must be called on the
training split, and every lookup and transform refuses to run before
that (`NotFittedError`). It is not wired into the default eight-stage
pipeline (`pipeline.default_steps`) — nothing downstream depends on it —
but it follows the same fit/transform contract as every other step so it
composes the same way, standalone::

    >>> climatology = MonthlyClimatology().fit(train_dataset)
    >>> climatology.monthly_field("sst", month=7).shape   # spatial lookup, July
    (101, 241)
    >>> climatology.at("subsurface_temp", month=7, lat=12.0, lon=80.0, depth=100)
    24.9                                                  # spatial + depth lookup
    >>> climatology.depth_profile("subsurface_temp", month=7, lat=12.0, lon=80.0)
    array([28.1, 27.4, ...])                              # depth lookup, full column

`transform()` additionally attaches `<var>_climatology` and
`<var>_delta_t` variables to a dataset, matching each timestep to its
calendar month's climatology — the same pattern `MissingValueHandler`
uses to line a `(12, *spatial)` field up against a `(time, *spatial)`
one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from src.data.loaders.representation import OceanDataset, Variable, build_variable
from src.data.preprocessing._utils import (
    axis_of,
    data_variables,
    json_safe,
    month_of,
    nan_mean,
    summarize_array,
    with_variables,
)
from src.data.preprocessing.base import LearnedStep
from src.data.preprocessing.errors import StepConfigurationError

PathLike = Union[str, Path]

#: Temperature variables this step targets when `variables=` is not given.
DEFAULT_VARIABLES: Tuple[str, ...] = ("sst", "subsurface_temp")

#: Suffixes of the variables `transform()` attaches to the dataset.
CLIMATOLOGY_SUFFIX = "_climatology"
DELTA_SUFFIX = "_delta_t"

#: File format version for `save`/`load`, bumped if the layout changes.
CLIMATOLOGY_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# The mathematical relationship — plain functions, independent of fitting
# --------------------------------------------------------------------------
#
# These two lines are the entire point of the module. Keeping them as
# free functions (rather than only methods) means the identity they form
# can be tested, and used, without an `OceanDataset`, a fitted
# `MonthlyClimatology`, or any of this module's other machinery — just
# arrays.


def compute_delta_temperature(
    target_temperature: np.ndarray, climatology: np.ndarray
) -> np.ndarray:
    """``delta_T = target_temperature - climatology``, elementwise.

    The training target for an anomaly-based model: what is left of the
    temperature once the expected seasonal value at that place and month
    has been subtracted out.
    """
    return np.asarray(target_temperature, dtype=float) - np.asarray(
        climatology, dtype=float
    )


def reconstruct_temperature(
    climatology: np.ndarray, predicted_delta_temperature: np.ndarray
) -> np.ndarray:
    """``temperature_prediction = climatology + predicted_delta_T``, elementwise.

    The inverse of `compute_delta_temperature`: puts a model's predicted
    anomaly back onto the climatological baseline to get a temperature
    prediction in physical units (degC).
    """
    return np.asarray(climatology, dtype=float) + np.asarray(
        predicted_delta_temperature, dtype=float
    )


class MonthlyClimatology(LearnedStep):
    """Monthly climatology for temperature variables, fitted on training data only.

    For each target variable, the fitted state is a ``(12, *spatial)``
    field — the training-period mean for that cell in that calendar
    month, exactly as `MissingValueHandler` computes it (monthly rather
    than an overall mean, because the seasonal cycle dominates every
    field this project handles: an annual mean would put a several-degree
    warm bias into a January reconstruction and a cold one into July).

    Three ways to query what was learned:

    * **Monthly lookup** — `monthly_field(variable, month)`: the whole
      spatial field for one calendar month.
    * **Spatial lookup** — `at(variable, month, lat, lon, depth=...)`:
      the value nearest a given point, for one calendar month.
    * **Depth lookup** — `depth_profile(variable, month, lat, lon)`: the
      full water-column profile at a point, for variables that have a
      depth axis.

    `delta_for` and `reconstruct` are the dataset-shaped counterparts of
    `compute_delta_temperature` / `reconstruct_temperature`: given a
    variable name and a `time` array, they look up the matching monthly
    climatology and apply the formula, so callers do not have to do the
    month-matching by hand.
    """

    name: ClassVar[str] = "climatology"
    title: ClassVar[str] = "Monthly climatology"

    def __init__(self, *, variables: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.variables = tuple(variables) if variables else None

        #: variable -> (12, *spatial) monthly climatology, training period only
        self._climatology: Dict[str, np.ndarray] = {}
        #: variable -> dims of the climatology field, minus the month axis
        #: (i.e. `variable.dims` with `"time"` removed)
        self._dims: Dict[str, Tuple[str, ...]] = {}
        #: variable -> units, carried through for the derived variables
        self._units: Dict[str, Optional[str]] = {}
        #: coordinate arrays as seen at fit time, for spatial/depth lookup
        self._lat: Optional[np.ndarray] = None
        self._lon: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None

    def config(self) -> Dict[str, Any]:
        return {
            "variables": list(self.variables) if self.variables else list(DEFAULT_VARIABLES)
        }

    def state(self) -> Dict[str, Any]:
        """Summaries of the fitted climatologies — not the fields themselves."""
        return {
            name: {
                "shape": list(field.shape),
                "dims": ["month", *self._dims.get(name, ())],
                **summarize_array(field),
            }
            for name, field in self._climatology.items()
        }

    # -- selection -----------------------------------------------------

    def _targets(self, dataset: OceanDataset) -> Dict[str, Variable]:
        candidates = data_variables(dataset)
        names = self.variables or tuple(v for v in DEFAULT_VARIABLES if v in candidates)
        if not names:
            raise StepConfigurationError(
                f"none of the default variables {list(DEFAULT_VARIABLES)} are present; "
                "pass variables=[...] explicitly"
            )
        missing = [n for n in names if n not in candidates]
        if missing:
            raise StepConfigurationError(
                f"variables not present (or are masks): {missing}"
            )
        return {name: candidates[name] for name in names}

    # -- fit -------------------------------------------------------------

    def _fit(self, dataset: OceanDataset) -> None:
        times = dataset.coords.get("time")
        if times is None or len(times) == 0:
            raise StepConfigurationError(
                "dataset has no 'time' coordinate to fit a monthly climatology from"
            )
        months = month_of(times)

        self._climatology = {}
        self._dims = {}
        self._units = {}
        self._lat = _copy_coord(dataset.coords.get("lat"))
        self._lon = _copy_coord(dataset.coords.get("lon"))
        self._depth = _copy_coord(dataset.coords.get("depth"))

        for name, variable in self._targets(dataset).items():
            time_axis = axis_of(variable, "time")
            if time_axis is None:
                raise StepConfigurationError(
                    f"'{name}' has no time axis; cannot fit a monthly climatology"
                )
            values = np.asarray(variable.values, dtype=float)
            moved = np.moveaxis(values, time_axis, 0)

            field = np.full((12,) + moved.shape[1:], np.nan, dtype=np.float32)
            for month in range(1, 13):
                selected = moved[months == month]
                if selected.size:
                    field[month - 1] = nan_mean(selected, axis=0)

            self._climatology[name] = field
            self._dims[name] = tuple(d for d in variable.dims if d != "time")
            self._units[name] = variable.units

    # -- monthly lookup ----------------------------------------------------

    def monthly_field(self, variable: str, month: int) -> np.ndarray:
        """The full ``(*spatial[, depth])`` climatology field for one calendar month.

        `month` is 1-12 (January = 1). This is the raw fitted field
        the other lookups index into.
        """
        field = self._field(variable)
        return field[self._month_index(month)]

    # -- spatial lookup ------------------------------------------------

    def at(
        self,
        variable: str,
        month: int,
        lat: float,
        lon: float,
        depth: Optional[float] = None,
    ) -> float:
        """Nearest-neighbour climatology at one (lat, lon[, depth]) point.

        Looks up the grid cell whose latitude and longitude are closest
        to `lat`/`lon` (and, for a variable with a depth axis, the depth
        level closest to `depth` — required in that case). Raises
        `StepConfigurationError` if `variable` has a depth axis and
        `depth` is not given, or vice versa.
        """
        field = self.monthly_field(variable, month)
        dims = self._dims[variable]
        index: list = [slice(None)] * field.ndim

        if "depth" in dims:
            if depth is None:
                raise StepConfigurationError(
                    f"'{variable}' has a depth axis; pass depth= (or use depth_profile)"
                )
            index[dims.index("depth")] = self._nearest(self._depth, depth, "depth")
        elif depth is not None:
            raise StepConfigurationError(f"'{variable}' has no depth axis; depth= is not usable")

        index[dims.index("lat")] = self._nearest(self._lat, lat, "lat")
        index[dims.index("lon")] = self._nearest(self._lon, lon, "lon")
        return float(field[tuple(index)])

    # -- depth lookup --------------------------------------------------

    def depth_profile(self, variable: str, month: int, lat: float, lon: float) -> np.ndarray:
        """The full water-column climatology at (lat, lon), for one month.

        Only for variables with a depth axis (`subsurface_temp` in the
        NEER domain); raises `StepConfigurationError` for a
        surface-only variable such as `sst`.
        """
        field = self.monthly_field(variable, month)
        dims = self._dims[variable]
        if "depth" not in dims:
            raise StepConfigurationError(
                f"'{variable}' has no depth axis; there is no profile to return"
            )
        index: list = [slice(None)] * field.ndim
        index[dims.index("lat")] = self._nearest(self._lat, lat, "lat")
        index[dims.index("lon")] = self._nearest(self._lon, lon, "lon")
        return np.asarray(field[tuple(index)], dtype=float)

    def depth_levels(self) -> Optional[np.ndarray]:
        """The depth coordinate (m) seen at fit time, or `None` if there wasn't one."""
        return None if self._depth is None else np.array(self._depth, copy=True)

    # -- delta_T and reconstruction, matched to a time axis -----------------

    def climatology_for(self, variable: str, times: np.ndarray) -> np.ndarray:
        """The ``(n_time, *spatial)`` climatology matching each entry of `times`.

        Each timestep is matched to its calendar month's fitted field —
        the same lookup `transform()` uses internally, exposed directly
        so a caller with model predictions (which do not carry an
        `OceanDataset` around) can build the climatology term itself.
        """
        field = self._field(variable)
        months = month_of(np.asarray(times))
        return field[np.clip(months, 1, 12) - 1]

    def delta_for(
        self, variable: str, target_temperature: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """``delta_T`` for a ``(n_time, *spatial)`` temperature array and matching `times`.

        Equivalent to
        ``compute_delta_temperature(target_temperature, self.climatology_for(variable, times))``.
        """
        climatology = self.climatology_for(variable, times)
        return compute_delta_temperature(target_temperature, climatology)

    def reconstruct(
        self, variable: str, predicted_delta_temperature: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Temperature prediction for a ``(n_time, *spatial)`` predicted delta_T and `times`.

        Equivalent to
        ``reconstruct_temperature(self.climatology_for(variable, times), predicted_delta_temperature)``.
        This is the inverse of `delta_for`: reconstructing with the
        *actual* delta_T returns the original temperature exactly (see
        `tests/test_climatology.py`).
        """
        climatology = self.climatology_for(variable, times)
        return reconstruct_temperature(climatology, predicted_delta_temperature)

    # -- transform: attach climatology + delta_T to a dataset ---------------

    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        self.require_fitted("no climatology has been computed")
        times = dataset.coords.get("time")
        if times is None:
            raise StepConfigurationError(
                "dataset has no 'time' coordinate to match the climatology against"
            )
        months = month_of(times)

        variables = dict(dataset.variables)
        report: Dict[str, Any] = {"variables": {}}

        for name, field in self._climatology.items():
            variable = dataset.variables.get(name)
            if variable is None:
                continue
            time_axis = axis_of(variable, "time")
            if time_axis is None:
                continue

            values = np.asarray(variable.values, dtype=float)
            matched = field[np.clip(months, 1, 12) - 1].astype(float)
            matched = np.moveaxis(matched, 0, time_axis)
            delta = compute_delta_temperature(values, matched)

            clim_name = f"{name}{CLIMATOLOGY_SUFFIX}"
            delta_name = f"{name}{DELTA_SUFFIX}"
            variables[clim_name] = build_variable(
                clim_name,
                matched.astype(np.float32),
                variable.dims,
                units=self._units.get(name),
                attrs={
                    "description": f"Monthly climatology for '{name}', training period only",
                    "source_variable": name,
                },
            )
            variables[delta_name] = build_variable(
                delta_name,
                delta.astype(np.float32),
                variable.dims,
                units=self._units.get(name),
                attrs={
                    "description": (
                        f"delta_T for '{name}': target_temperature - climatology"
                    ),
                    "source_variable": name,
                },
            )
            report["variables"][name] = {
                "climatology_variable": clim_name,
                "delta_variable": delta_name,
                **summarize_array(delta),
            }

        self._report = report
        return with_variables(dataset, variables)

    # -- internal helpers -------------------------------------------------

    def _field(self, variable: str) -> np.ndarray:
        self.require_fitted("no climatology has been computed")
        field = self._climatology.get(variable)
        if field is None:
            raise StepConfigurationError(
                f"no fitted climatology for '{variable}'; fitted variables are "
                f"{sorted(self._climatology)}"
            )
        return field

    @staticmethod
    def _month_index(month: int) -> int:
        if not 1 <= int(month) <= 12:
            raise StepConfigurationError(f"month must be 1-12, got {month!r}")
        return int(month) - 1

    @staticmethod
    def _nearest(coord_array: Optional[np.ndarray], value: float, label: str) -> int:
        if coord_array is None or coord_array.size == 0:
            raise StepConfigurationError(
                f"no '{label}' coordinate was recorded at fit time"
            )
        return int(np.argmin(np.abs(np.asarray(coord_array, dtype=float) - float(value))))

    # -- convenience --------------------------------------------------------

    def climatology(self, variable: str) -> Optional[np.ndarray]:
        """The fitted ``(12, *spatial)`` monthly climatology for `variable`, or `None`."""
        return self._climatology.get(variable)

    def fitted_variables(self) -> Tuple[str, ...]:
        return tuple(self._climatology)

    # -- persistence ---------------------------------------------------------

    def save(self, path: PathLike, *, compress: bool = True) -> Path:
        """Write the fitted climatology fields to a `.npz` file (+ JSON sidecar).

        Unlike `Normalizer`'s statistics (a few numbers per variable), a
        climatology field is potentially large, so it is stored in full
        as arrays rather than summarized — the point of saving it is to
        reuse the *fields* later without re-fitting, exactly as
        `TensorBundle` stores tensors.
        """
        self.require_fitted("nothing to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {
            f"climatology__{name}": field for name, field in self._climatology.items()
        }
        if self._lat is not None:
            arrays["lat"] = self._lat
        if self._lon is not None:
            arrays["lon"] = self._lon
        if self._depth is not None:
            arrays["depth"] = self._depth

        sidecar = {
            "version": CLIMATOLOGY_VERSION,
            "config": self.config(),
            "dims": {name: list(dims) for name, dims in self._dims.items()},
            "units": dict(self._units),
        }
        arrays["_metadata"] = np.array(json.dumps(json_safe(sidecar), indent=2))

        writer = np.savez_compressed if compress else np.savez
        writer(path, **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "MonthlyClimatology":
        """Read a climatology written by `save`, already fitted."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["_metadata"]))
            names = [
                key[len("climatology__") :]
                for key in data.files
                if key.startswith("climatology__")
            ]
            instance = cls(variables=metadata.get("config", {}).get("variables"))
            instance._climatology = {name: data[f"climatology__{name}"] for name in names}
            instance._dims = {
                name: tuple(dims) for name, dims in metadata.get("dims", {}).items()
            }
            instance._units = dict(metadata.get("units", {}))
            instance._lat = data["lat"] if "lat" in data.files else None
            instance._lon = data["lon"] if "lon" in data.files else None
            instance._depth = data["depth"] if "depth" in data.files else None
            instance._fitted = True
            instance._fit_summary = {"loaded_from": str(path)}
        return instance


def _copy_coord(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if values is None else np.array(values, copy=True)
