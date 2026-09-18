"""
Phase 12 — PyTorch `Dataset` for NEER.

Wraps a Phase 08 `TensorBundle` (`src/data/preprocessing/tensors.py`) in a
`torch.utils.data.Dataset`, so the training phase can hand
`data/processed/neer_tensors.npz` straight to a `DataLoader` without
knowing anything about ocean data formats, and without any other phase
needing to know PyTorch exists.

This module builds **only** the dataset/loading layer. No model lives
here — see `MODEL_CARD.md` and `src/models/` for why.

Unit of iteration
------------------
One sample is one timestep's full spatial grid, matching the
channels-second `(time, channel, lat, lon)` layout `TensorAssembler`
already produces (see `tensors.py`): that layout exists precisely so a
batch of timesteps drops straight into a convolutional encoder with no
transpose. `NEERDataset[i]` therefore returns a whole `(channel, lat,
lon)` image, and `DataLoader` batching prepends a batch axis to get
`(batch, channel, lat, lon)` for free.

Input / target contract
------------------------
::

    inputs    (11, n_lat, n_lon)   float32   — NEER_CHANNEL_ORDER
    targets   (15, n_lat, n_lon)   float32   — subsurface temperature,
                                                one channel per depth level

Both are already finite: Phase 08 fills gaps with `fill_value` (0.0 by
default, the training mean after normalization). The masks are the only
record of what was real.

Return contract
----------------
`__getitem__` returns a dict:

    inputs      FloatTensor  (n_channels, n_lat, n_lon)
    targets     FloatTensor  (n_depth,    n_lat, n_lon)
    mask        {"input":  BoolTensor (n_channels, n_lat, n_lon),
                 "target": BoolTensor (n_depth,    n_lat, n_lon)}
    metadata    {"date":      str,                          e.g. "2020-01-15"
                 "latitude":  float32 array (n_lat,),
                 "longitude": float32 array (n_lon,),
                 "depth":     float32 array (n_depth,)}      metres

`mask` carries both masks rather than picking one: `target_mask` is what
a loss must be masked by (subsurface coverage is sparse — scoring
against fill values measures the filler, not the model), but
`input_mask` is still useful context (e.g. to tell the model which cells
are land versus gap-filled ocean), so both travel together rather than
one being silently dropped.

`latitude`/`longitude` are the full coordinate vectors for the sample's
grid (every sample shares the same grid, so these are constant across
the dataset) rather than a single point — the natural reading of
"latitude"/"longitude" metadata for a whole-grid sample. `depth` is the
15 depth levels (metres) the target channels correspond to, in the same
order as the channel axis, so `targets[k]` is the field at `depth[k]`.

All four dict values PyTorch's default `collate_fn` handles without a
custom collate function: nested dicts collate recursively, numpy arrays
collate into stacked tensors, and a plain `str` collates into a list of
strings rather than being stacked — see `make_dataloader` below.

torch is optional
------------------
Nothing else in NEER needs torch, so — mirroring how `src/data/loaders`
treats `xarray`/`netCDF4` as optional (`src/data/loaders/_xarray.py`) —
it is imported lazily. Importing this module without torch installed
works; constructing a `NEERDataset` without it raises
`MissingDependencyError` naming the install command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing.channels import NEER_CHANNEL_ORDER, NEER_N_CHANNELS
from src.data.preprocessing.tensors import TensorBundle

try:  # pragma: no cover - environment dependent
    import torch
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _DataLoader = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

PathLike = Union[str, Path]

#: Expected number of subsurface-temperature target depth levels. Not a
#: structural requirement of `TensorBundle` (which allows any target
#: shape), so this is enforced here, with a clear error, rather than
#: assumed silently — a wiring mistake in tensor assembly should surface
#: at dataset construction, not three phases later inside a loss.
NEER_N_DEPTHS: int = 15


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="the NEER PyTorch Dataset (src.data.dataset)",
            install_hint="pip install torch",
        )


class NEERDataset(_TorchDataset):
    """`torch.utils.data.Dataset` over a NEER `TensorBundle`.

    Parameters
    ----------
    bundle:
        A `TensorBundle` produced by `TensorAssembler.assemble` (or
        `TensorBundle.load`), carrying both inputs and targets.
    split:
        If given, restrict to one of the bundle's recorded chronological
        splits (`"train"`, `"val"`, `"test"`) via `TensorBundle.split`,
        before anything else is checked or read.
    strict_shape:
        If True (default), require exactly `NEER_N_CHANNELS` (11) input
        channels and exactly `NEER_N_DEPTHS` (15) target depth levels,
        raising `ValueError` otherwise. Set False for ad hoc or
        exploratory bundles that intentionally use a different shape.

    Examples
    --------
    >>> bundle = TensorBundle.load("data/processed/neer_tensors.npz")
    >>> train_ds = NEERDataset(bundle, split="train")
    >>> loader = make_dataloader(train_ds, batch_size=4, shuffle=True)
    >>> batch = next(iter(loader))
    >>> batch["inputs"].shape
    torch.Size([4, 11, 101, 241])
    """

    def __init__(
        self,
        bundle: TensorBundle,
        *,
        split: Optional[str] = None,
        strict_shape: bool = True,
    ) -> None:
        _require_torch()

        if split is not None:
            bundle = bundle.split(split)

        if bundle.targets is None or bundle.target_mask is None:
            raise ValueError(
                "NEERDataset requires a TensorBundle assembled with subsurface "
                "temperature targets, but bundle.targets is None. Re-run "
                "TensorAssembler with a target variable configured (see "
                "src/data/preprocessing/tensors.py:DEFAULT_TARGETS)."
            )

        if strict_shape:
            if bundle.n_channels != NEER_N_CHANNELS:
                raise ValueError(
                    f"NEERDataset expects {NEER_N_CHANNELS} input channels "
                    f"{list(NEER_CHANNEL_ORDER)}, got {bundle.n_channels}: "
                    f"{bundle.channel_names}. Pass strict_shape=False to allow "
                    "a different input contract."
                )
            n_depth = int(bundle.targets.shape[1])
            if n_depth != NEER_N_DEPTHS:
                raise ValueError(
                    f"NEERDataset expects {NEER_N_DEPTHS} target depth levels, "
                    f"got {n_depth}. Pass strict_shape=False to allow a "
                    "different target contract."
                )
            if bundle.depth is not None and len(bundle.depth) != n_depth:
                raise ValueError(
                    f"depth coordinate has {len(bundle.depth)} levels but "
                    f"targets have {n_depth} depth channels; TensorBundle is "
                    "internally inconsistent."
                )

        self.bundle = bundle

    # -- construction helpers ------------------------------------------

    @classmethod
    def from_npz(
        cls,
        path: PathLike,
        *,
        split: Optional[str] = None,
        strict_shape: bool = True,
    ) -> "NEERDataset":
        """Load a bundle written by `TensorBundle.save` and wrap it.

        `path` is typically `data/processed/neer_tensors.npz`, the output
        of `scripts/preprocess_data.py`.
        """
        return cls(TensorBundle.load(path), split=split, strict_shape=strict_shape)

    # -- descriptive properties ------------------------------------------

    @property
    def channel_names(self):
        return list(self.bundle.channel_names)

    @property
    def target_names(self):
        return list(self.bundle.target_names)

    @property
    def grid_shape(self):
        return self.bundle.grid_shape

    # -- torch Dataset protocol ------------------------------------------

    def __len__(self) -> int:
        return self.bundle.n_time

    def __getitem__(self, index: int) -> Dict[str, Any]:
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"index {index} out of range for dataset of length {n}")

        b = self.bundle

        inputs = torch.from_numpy(np.ascontiguousarray(b.inputs[index])).float()
        targets = torch.from_numpy(np.ascontiguousarray(b.targets[index])).float()
        input_mask = torch.from_numpy(np.ascontiguousarray(b.input_mask[index])).bool()
        target_mask = torch.from_numpy(np.ascontiguousarray(b.target_mask[index])).bool()

        date = str(np.datetime_as_string(np.asarray(b.time[index]).astype("datetime64[D]")))
        depth = (
            b.depth.astype(np.float32)
            if b.depth is not None
            else np.full(targets.shape[0], np.nan, dtype=np.float32)
        )

        metadata = {
            "date": date,
            "latitude": b.lat.astype(np.float32),
            "longitude": b.lon.astype(np.float32),
            "depth": depth,
        }

        return {
            "inputs": inputs,
            "targets": targets,
            "mask": {"input": input_mask, "target": target_mask},
            "metadata": metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n_depth = None if self.bundle.targets is None else self.bundle.targets.shape[1]
        return (
            f"NEERDataset(n_samples={len(self)}, n_channels={self.bundle.n_channels}, "
            f"grid_shape={self.grid_shape}, n_depth={n_depth})"
        )


def make_dataloader(
    dataset: "NEERDataset",
    *,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs: Any,
) -> "_DataLoader":
    """A `DataLoader` over `dataset` using PyTorch's default collation.

    No custom `collate_fn` is needed: `NEERDataset.__getitem__` returns
    plain tensors and JSON-safe nested dicts/arrays/strings, all of which
    PyTorch's default collate handles recursively. This wrapper exists
    for a sensible default `batch_size` and so callers building a quick
    loader don't need their own `torch` import.
    """
    _require_torch()
    return _DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
