"""
Stage 8 — normalized tensors.

The end of the pipeline. Everything so far has produced a cleaner
`OceanDataset`; this turns that into the plain, rectangular, finite
arrays a model actually consumes, and into a file that can be handed to
the training phase without it needing to know anything about ocean data
formats.

Shapes
------
::

    inputs       (n_time, n_channels, n_lat, n_lon)   float32
    input_mask   (n_time, n_channels, n_lat, n_lon)   bool
    targets      (n_time, n_depth,    n_lat, n_lon)   float32
    target_mask  (n_time, n_depth,    n_lat, n_lon)   bool

Channels-second follows the PyTorch NCHW convention, so a batch of
timesteps drops straight into a convolutional encoder with no transpose.

Three kinds of variable become input channels: surface fields
`(time, lat, lon)`; static fields `(lat, lon)` such as `coriolis`,
broadcast across time; and time-only series `(time,)` such as the cyclic
encodings, broadcast across the grid. Broadcasting them here rather than
in the model keeps the tensor self-describing — every channel has a name
in `channel_names`, and nothing downstream has to remember to append the
day-of-year.

Masks are the contract
----------------------
`inputs` is guaranteed finite: any remaining NaN (land, an unfillable
gap) is replaced with `fill_value`, which is 0.0 by default — after
normalization, 0 is the training mean, the least informative value
available. The *only* record of what was real is the mask, which is True
exactly where the cell is ocean, finite, and (for observed variables)
actually measured rather than gap-filled.

That makes the mask load-bearing, not decorative: a loss computed
without it would train the model to reproduce fill values over the
Deccan Plateau and score it on gaps that were interpolated from its own
inputs. `target_mask` is the one that matters most — subsurface coverage
is sparse, and it is the difference between evaluating on ARGO
observations and evaluating on a climatology.

The split travels with the tensors
----------------------------------
`train_mask` / `val_mask` / `test_mask` over the time axis are stored in
the file. A later phase that loads these tensors gets the same
chronological split the statistics were fitted on, rather than
re-deriving one and quietly mismatching it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.data.loaders.representation import OceanDataset
from src.data.preprocessing.channels import NEER_CHANNEL_ORDER
from src.data.preprocessing._utils import (
    broadcast_spatial,
    data_variables,
    is_mask_variable,
    json_safe,
    ocean_mask_of,
)
from src.data.preprocessing.errors import PreprocessingError, StepConfigurationError
from src.data.preprocessing.masking import OCEAN_MASK
from src.data.preprocessing.missing import OBSERVED_SUFFIX
from src.data.preprocessing.splits import TemporalSplit

PathLike = Union[str, Path]

#: Default reconstruction targets, in order of preference.
DEFAULT_TARGETS: Tuple[str, ...] = ("subsurface_temp",)

#: Value written where data is missing. Zero is the training mean after
#: z-score normalization — the least informative value available.
DEFAULT_FILL_VALUE = 0.0


@dataclass(frozen=True)
class TensorBundle:
    """Model-ready tensors plus everything needed to interpret them."""

    inputs: np.ndarray            # (time, channel, lat, lon) float32
    input_mask: np.ndarray        # (time, channel, lat, lon) bool
    channel_names: List[str]
    time: np.ndarray              # (time,) datetime64[ns]
    lat: np.ndarray               # (lat,)
    lon: np.ndarray               # (lon,)
    targets: Optional[np.ndarray] = None       # (time, depth, lat, lon) float32
    target_mask: Optional[np.ndarray] = None   # (time, depth, lat, lon) bool
    target_names: List[str] = field(default_factory=list)
    depth: Optional[np.ndarray] = None
    ocean_mask: Optional[np.ndarray] = None    # (lat, lon) bool
    split_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)

    # -- shape -------------------------------------------------------------

    @property
    def n_time(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (int(self.inputs.shape[2]), int(self.inputs.shape[3]))

    @property
    def is_synthetic(self) -> bool:
        """True when these tensors came from synthetic demo data."""
        mode = str(self.attrs.get("data_mode", "")).upper()
        return "SYNTHETIC" in mode or "DEMO" in mode

    # -- access ------------------------------------------------------------

    def channel(self, name: str) -> np.ndarray:
        """The `(time, lat, lon)` slice for one named input channel."""
        try:
            index = self.channel_names.index(name)
        except ValueError:
            raise KeyError(
                f"no channel '{name}'; available: {self.channel_names}"
            ) from None
        return self.inputs[:, index]

    def split(self, period: str) -> "TensorBundle":
        """A new bundle restricted to `period` ('train', 'val' or 'test')."""
        if period not in self.split_masks:
            raise KeyError(
                f"no '{period}' split; available: {sorted(self.split_masks)}"
            )
        mask = np.asarray(self.split_masks[period], dtype=bool)
        from dataclasses import replace

        return replace(
            self,
            inputs=self.inputs[mask],
            input_mask=self.input_mask[mask],
            time=self.time[mask],
            targets=None if self.targets is None else self.targets[mask],
            target_mask=None if self.target_mask is None else self.target_mask[mask],
            split_masks={period: np.ones(int(mask.sum()), dtype=bool)},
            attrs={**self.attrs, "split": period},
        )

    # -- reporting ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """JSON-safe description of the tensors (shapes, coverage, no data)."""
        summary: Dict[str, Any] = {
            "inputs": {
                "shape": list(self.inputs.shape),
                "dtype": str(self.inputs.dtype),
                "layout": "(time, channel, lat, lon)",
                "channels": list(self.channel_names),
                "valid_fraction": round(float(self.input_mask.mean()), 6),
                "all_finite": bool(np.isfinite(self.inputs).all()),
            },
            "time_range": (
                None
                if self.time.size == 0
                else [str(self.time.min()), str(self.time.max())]
            ),
            "grid_shape": list(self.grid_shape),
            "split_counts": {
                name: int(np.asarray(mask).sum()) for name, mask in self.split_masks.items()
            },
        }
        if self.targets is not None:
            summary["targets"] = {
                "shape": list(self.targets.shape),
                "dtype": str(self.targets.dtype),
                "layout": "(time, depth, lat, lon)",
                "variables": list(self.target_names),
                "valid_fraction": (
                    None
                    if self.target_mask is None
                    else round(float(self.target_mask.mean()), 6)
                ),
                "all_finite": bool(np.isfinite(self.targets).all()),
            }
        if self.ocean_mask is not None:
            summary["ocean_fraction"] = round(float(self.ocean_mask.mean()), 6)
        return json_safe(summary)

    def spatial_coverage(self) -> Dict[str, Any]:
        """Grid extent plus, per split period, how much of it actually
        carries valid data.

        The grid itself — shape, lat/lon range, land/ocean layout — is one
        fixed domain shared by every period; splitting only cuts the time
        axis, so that part of "coverage" cannot differ between train, val
        and test by construction. What genuinely can differ is *temporal*:
        whether a given ocean cell happened to be observed during a
        period's particular months. This reports both, so a period that
        looks fine in timestep counts but has, say, a data-sparse region
        for its whole window is visible in the metadata rather than only
        showing up later as a hole in a prediction map.

        `valid_fraction` is the mean of `input_mask` over that period —
        the same "how much of the tensor is real, not fill" statistic
        `TensorBundle.summary()` reports for the whole bundle, just scoped
        to one split. `cells_ever_observed_fraction` is coarser and
        catches a different failure: a cell can contribute a low but
        nonzero `valid_fraction` while never once being observed in a
        short validation window, which is the case that actually breaks
        evaluation over that region.
        """
        coverage: Dict[str, Any] = {
            "grid_shape": list(self.grid_shape),
            "lat_range": (
                [float(self.lat.min()), float(self.lat.max())] if self.lat.size else None
            ),
            "lon_range": (
                [float(self.lon.min()), float(self.lon.max())] if self.lon.size else None
            ),
            "ocean_fraction": (
                None if self.ocean_mask is None else round(float(self.ocean_mask.mean()), 6)
            ),
        }
        by_period: Dict[str, Any] = {}
        ocean_cells = None if self.ocean_mask is None else int(np.asarray(self.ocean_mask).sum())
        for period, mask in self.split_masks.items():
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                by_period[period] = {
                    "n_timesteps": 0,
                    "valid_fraction": None,
                    "cells_ever_observed_fraction": None,
                }
                continue
            period_mask = self.input_mask[mask]  # (time, channel, lat, lon)
            ever_observed = period_mask.any(axis=(0, 1))  # (lat, lon)
            if self.ocean_mask is not None and ocean_cells:
                cells_fraction = round(
                    float(np.logical_and(ever_observed, self.ocean_mask).sum() / ocean_cells), 6
                )
            else:
                cells_fraction = round(float(ever_observed.mean()), 6)
            by_period[period] = {
                "n_timesteps": int(mask.sum()),
                "valid_fraction": round(float(period_mask.mean()), 6),
                "cells_ever_observed_fraction": cells_fraction,
            }
        coverage["by_period"] = by_period
        return json_safe(coverage)

    def describe(self) -> str:
        """Short human-readable summary, for logs."""
        lines = [
            f"TensorBundle  inputs={self.inputs.shape} "
            f"targets={None if self.targets is None else self.targets.shape}",
            f"  channels ({self.n_channels}): {', '.join(self.channel_names)}",
        ]
        if self.split_masks:
            counts = ", ".join(
                f"{name}={int(np.asarray(mask).sum())}"
                for name, mask in self.split_masks.items()
            )
            lines.append(f"  split: {counts}")
        lines.append(
            f"  input valid: {self.input_mask.mean():.1%}"
            + (
                ""
                if self.target_mask is None
                else f"   target valid: {self.target_mask.mean():.1%}"
            )
        )
        if self.is_synthetic:
            lines.append("  ⚠ SYNTHETIC demo data — not observations")
        return "\n".join(lines)

    # -- persistence -------------------------------------------------------

    def save(self, path: PathLike, *, compress: bool = True) -> Path:
        """Write the bundle to a `.npz` file (arrays plus a JSON sidecar blob)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {
            "inputs": self.inputs,
            "input_mask": self.input_mask,
            "time": self.time,
            "lat": self.lat,
            "lon": self.lon,
        }
        if self.targets is not None:
            arrays["targets"] = self.targets
        if self.target_mask is not None:
            arrays["target_mask"] = self.target_mask
        if self.depth is not None:
            arrays["depth"] = self.depth
        if self.ocean_mask is not None:
            arrays["ocean_mask"] = self.ocean_mask
        for name, mask in self.split_masks.items():
            arrays[f"split_{name}"] = np.asarray(mask, dtype=bool)

        arrays["_metadata"] = np.array(
            json.dumps(
                {
                    "channel_names": list(self.channel_names),
                    "target_names": list(self.target_names),
                    "attrs": json_safe(self.attrs),
                },
                indent=2,
            )
        )

        writer = np.savez_compressed if compress else np.savez
        writer(path, **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "TensorBundle":
        """Read a bundle written by `save`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["_metadata"]))
            split_masks = {
                key[len("split_") :]: data[key] for key in data.files if key.startswith("split_")
            }
            return cls(
                inputs=data["inputs"],
                input_mask=data["input_mask"],
                channel_names=list(metadata["channel_names"]),
                time=data["time"],
                lat=data["lat"],
                lon=data["lon"],
                targets=data["targets"] if "targets" in data.files else None,
                target_mask=data["target_mask"] if "target_mask" in data.files else None,
                target_names=list(metadata.get("target_names", [])),
                depth=data["depth"] if "depth" in data.files else None,
                ocean_mask=data["ocean_mask"] if "ocean_mask" in data.files else None,
                split_masks=split_masks,
                attrs=dict(metadata.get("attrs", {})),
            )


class TensorAssembler:
    """Turn a preprocessed `OceanDataset` into a `TensorBundle`.

    Not a `PreprocessingStep`: every step maps a dataset to a dataset,
    and this is the terminal stage that changes the type. It learns
    nothing — it is a deterministic repacking of data the fitted steps
    have already produced.
    """

    name: ClassVar[str] = "tensor_assembly"
    title: ClassVar[str] = "Tensor assembly"

    def __init__(
        self,
        *,
        input_variables: Optional[Sequence[str]] = NEER_CHANNEL_ORDER,
        target_variables: Sequence[str] = DEFAULT_TARGETS,
        fill_value: float = DEFAULT_FILL_VALUE,
        dtype: Any = np.float32,
        require_observed_masks: bool = True,
        drop_empty_targets: bool = True,
    ) -> None:
        self.input_variables = tuple(input_variables) if input_variables else None
        self.target_variables = tuple(target_variables)
        self.fill_value = float(fill_value)
        self.dtype = dtype
        self.require_observed_masks = require_observed_masks
        self.drop_empty_targets = drop_empty_targets
        self._report: Dict[str, Any] = {}

    def config(self) -> Dict[str, Any]:
        return {
            "input_variables": list(self.input_variables) if self.input_variables else None,
            "target_variables": list(self.target_variables),
            "fill_value": self.fill_value,
            "dtype": np.dtype(self.dtype).name,
            "require_observed_masks": self.require_observed_masks,
        }

    def report(self) -> Dict[str, Any]:
        return dict(self._report)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(
            {
                "step": self.name,
                "title": self.title,
                "class": type(self).__name__,
                "learns_from_data": False,
                "config": self.config(),
                "report": self._report,
            }
        )

    # -- assembly ----------------------------------------------------------

    def select_channels(self, dataset: OceanDataset) -> List[str]:
        """Input channel names, in a deterministic order.

        By default `input_variables` is `channels.NEER_CHANNEL_ORDER` —
        the authoritative, fixed eleven-channel input contract — so this
        returns exactly that list, and raises if the dataset is missing
        any of it. Passing an explicit `input_variables` (including
        `None`, to fall back to auto-detection) overrides that: `None`
        selects every usable variable instead — surface fields first,
        then static geographic fields, then time-only series, each
        group sorted by name, so two runs on the same data produce
        identically ordered channels. That fallback exists for ad hoc
        or exploratory tensors; anything trained against
        `NEER_CHANNEL_ORDER` should leave `input_variables` at its
        default.
        """
        if self.input_variables is not None:
            missing = [n for n in self.input_variables if n not in dataset]
            if missing:
                raise StepConfigurationError(f"input variables not in dataset: {missing}")
            return list(self.input_variables)

        surface, static, temporal = [], [], []
        for name, variable in data_variables(dataset).items():
            if name in self.target_variables or "depth" in variable.dims:
                continue
            dims = set(variable.dims)
            if dims == {"time", "lat", "lon"}:
                surface.append(name)
            elif dims == {"lat", "lon"}:
                static.append(name)
            elif dims == {"time"}:
                temporal.append(name)
        return sorted(surface) + sorted(static) + sorted(temporal)

    def assemble(
        self,
        dataset: OceanDataset,
        *,
        split: Optional[TemporalSplit] = None,
    ) -> TensorBundle:
        """Build the tensors. `split` is recorded as masks over the time axis."""
        time = dataset.coords.get("time")
        lat = dataset.coords.get("lat")
        lon = dataset.coords.get("lon")
        if time is None or lat is None or lon is None:
            raise PreprocessingError(
                "tensor assembly needs time, lat and lon coordinates; got "
                f"{sorted(dataset.coords)}"
            )

        n_time, n_lat, n_lon = int(time.size), int(lat.size), int(lon.size)
        ocean = ocean_mask_of(dataset, OCEAN_MASK)
        if ocean is None:
            ocean = ocean_mask_of(dataset, "land_mask")

        channel_names = self.select_channels(dataset)
        if not channel_names:
            raise PreprocessingError("no input channels could be selected from this dataset")

        inputs = np.empty((n_time, len(channel_names), n_lat, n_lon), dtype=self.dtype)
        input_mask = np.zeros_like(inputs, dtype=bool)
        for index, name in enumerate(channel_names):
            values, valid = self._channel(dataset, name, (n_time, n_lat, n_lon), ocean)
            inputs[:, index] = values
            input_mask[:, index] = valid

        targets = target_mask = None
        target_names: List[str] = []
        depth = dataset.coords.get("depth")
        present = [n for n in self.target_variables if n in dataset]
        if present:
            stacks, masks = [], []
            for name in present:
                variable = dataset[name]
                if "depth" not in variable.dims:
                    raise StepConfigurationError(
                        f"target '{name}' has dims {variable.dims}; a depth axis is required"
                    )
                values, valid = self._target(dataset, name, ocean)
                stacks.append(values)
                masks.append(valid)
                target_names.append(name)
            targets = np.concatenate(stacks, axis=1) if len(stacks) > 1 else stacks[0]
            target_mask = np.concatenate(masks, axis=1) if len(masks) > 1 else masks[0]
        elif not self.drop_empty_targets:
            raise StepConfigurationError(
                f"none of the target variables {list(self.target_variables)} are in the dataset"
            )

        split_masks: Dict[str, np.ndarray] = {}
        if split is not None:
            split_masks = {k: np.asarray(v, dtype=bool) for k, v in split.masks(time).items()}

        bundle = TensorBundle(
            inputs=inputs,
            input_mask=input_mask,
            channel_names=channel_names,
            time=np.asarray(time),
            lat=np.asarray(lat, dtype=float),
            lon=np.asarray(lon, dtype=float),
            targets=targets,
            target_mask=target_mask,
            target_names=target_names,
            depth=None if depth is None else np.asarray(depth, dtype=float),
            ocean_mask=ocean,
            split_masks=split_masks,
            attrs={
                "data_mode": dataset.data_mode,
                "is_synthetic": dataset.is_synthetic,
                "source": dataset.source,
                "fill_value": self.fill_value,
                **(
                    {"disclaimer": dataset.attrs["disclaimer"]}
                    if "disclaimer" in dataset.attrs
                    else {}
                ),
            },
        )
        self._report = bundle.summary()
        return bundle

    # -- internals ---------------------------------------------------------

    def _valid_mask(
        self,
        dataset: OceanDataset,
        name: str,
        values: np.ndarray,
        dims: Sequence[str],
        ocean: Optional[np.ndarray],
    ) -> np.ndarray:
        """True where a cell is finite, over ocean, and genuinely observed."""
        valid = np.isfinite(values)
        if ocean is not None and {"lat", "lon"} <= set(dims):
            valid = valid & np.broadcast_to(
                broadcast_spatial(ocean, tuple(dims), ("lat", "lon")), values.shape
            )
        observed = dataset.variables.get(f"{name}{OBSERVED_SUFFIX}")
        if observed is not None and self.require_observed_masks:
            observed_values = np.asarray(observed.values, dtype=bool)
            if observed_values.shape == values.shape:
                valid = valid & observed_values
        return valid

    def _channel(self, dataset: OceanDataset, name: str, shape, ocean):
        n_time, n_lat, n_lon = shape
        variable = dataset[name]
        dims = variable.dims
        values = np.asarray(variable.values, dtype=float)
        valid = self._valid_mask(dataset, name, values, dims, ocean)

        if dims == ("time", "lat", "lon"):
            pass
        elif dims == ("lat", "lon"):
            values = np.broadcast_to(values, (n_time, n_lat, n_lon))
            valid = np.broadcast_to(valid, (n_time, n_lat, n_lon))
        elif dims == ("time",):
            values = np.broadcast_to(values.reshape(n_time, 1, 1), (n_time, n_lat, n_lon))
            valid = np.broadcast_to(valid.reshape(n_time, 1, 1), (n_time, n_lat, n_lon))
            if ocean is not None:
                valid = valid & np.broadcast_to(ocean, (n_time, n_lat, n_lon))
        else:
            raise StepConfigurationError(
                f"variable '{name}' has dims {dims}, which cannot become an input channel"
            )

        filled = np.where(np.isfinite(values), values, self.fill_value).astype(self.dtype)
        return filled, np.asarray(valid, dtype=bool)

    def _target(self, dataset: OceanDataset, name: str, ocean):
        variable = dataset[name]
        values = np.asarray(variable.values, dtype=float)
        valid = self._valid_mask(dataset, name, values, variable.dims, ocean)
        filled = np.where(np.isfinite(values), values, self.fill_value).astype(self.dtype)
        return filled, np.asarray(valid, dtype=bool)
