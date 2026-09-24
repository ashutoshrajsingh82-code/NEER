"""
Phase 24 — common evaluation interface for baseline models.

Every baseline (`src/models/baselines`) and `NEERModel` itself
(`scripts/train.py`, via its own `pool_profile`) ultimately predicts one
temperature-anomaly *profile* per sample — `(n_depth,)` — from one
input sample. This module gives every model that predicts a profile
one shared way to:

1. **See exactly the same data.** `pool_tensor_bundle` reads a
   `TensorBundle` (the same `.npz` file `scripts/preprocess_data.py`
   writes and `scripts/train.py` loads) and produces a `PooledSplit`
   per split (`train`/`val`/`test`) directly from `TensorBundle.split_masks`
   — the one chronological split recorded when the tensors were
   assembled (Phase 08/09). There is no second way to carve up the
   data: every baseline and `NEERModel` reads the same `.npz` file, so
   pointing two runs at different files is the only way to compare
   apples to oranges, and `evaluate_baseline` guards against exactly
   that (see below).
2. **Be scored the same way.** `evaluate_baseline` calls
   `src.evaluation.metrics.compute_profile_metrics` on every split, so
   climatology, Ridge, LightGBM and NEER reports are directly
   comparable numbers, not differently-computed approximations of the
   same idea.

Why a *pooled* profile, not the full `(depth, lat, lon)` grid
---------------------------------------------------------------
`NEERModel.forward` predicts one pooled anomaly profile per sample,
`(batch, n_depth)` — not a per-pixel field (see `scripts/train.py`'s
module docstring). For a baseline's score to mean anything next to
NEER's, it has to predict and be scored against the *same quantity*.
`pool_tensor_bundle` therefore reduces each sample's full-grid target
to a profile the same way `scripts/train.py::pool_profile` reduces
NEER's training batches: a `target_mask`-weighted spatial mean per
depth level, with a depth that has zero valid cells in a sample
producing a `False` profile-mask entry rather than a fabricated
`0.0` target. `features` is the analogous reduction of the *input*
side — an `input_mask`-weighted spatial mean per channel — so a
feature-vector baseline (Ridge, LightGBM) has something to fit
against; `NEERModel` does not need it, since it consumes the full
grid itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.data.preprocessing._utils import month_of
from src.data.preprocessing.channels import NEER_CHANNEL_ORDER
from src.data.preprocessing.tensors import TensorBundle
from src.evaluation.metrics import ProfileMetrics, compute_profile_metrics

PathLike = Union[str, Path]

POOL_EPS = 1e-8

#: The three splits every evaluation runs over, in a fixed order.
SPLIT_NAMES: Sequence[str] = ("train", "val", "test")


def _weighted_spatial_mean(values: np.ndarray, mask: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Reduce `(..., lat, lon)` to `(...)` with a mask-weighted mean.

    Returns `(pooled_values, pooled_mask)`: `pooled_mask` is `True`
    wherever at least one spatial cell was valid, and `pooled_values`
    is only meaningful where `pooled_mask` is `True` (elsewhere it is
    `0.0`, matching `scripts/train.py::pool_profile`'s convention so a
    baseline and NEER's pooled targets are bit-for-bit the same
    reduction of the same tensors).
    """
    mask_f = mask.astype(np.float64)
    valid_count = mask_f.sum(axis=(-2, -1))
    pooled = (values.astype(np.float64) * mask_f).sum(axis=(-2, -1)) / np.clip(
        valid_count, POOL_EPS, None
    )
    pooled_mask = valid_count > 0
    return pooled, pooled_mask


@dataclass(frozen=True)
class PooledSplit:
    """One split's worth of data, pooled to the profile-level quantities
    every model in `src/models/baselines` (and, via its own reduction,
    `NEERModel`) is fit and scored against.

    Attributes
    ----------
    features:
        `(n_samples, n_channels)` — mask-weighted spatial mean of each
        input channel. Feature-vector baselines fit against this.
    profile:
        `(n_samples, n_depth)` — mask-weighted spatial mean of the
        target, per depth. This is the quantity every model predicts.
    profile_mask:
        `(n_samples, n_depth)` bool — `True` where `profile` is a real
        (not fill/placeholder) value.
    months:
        `(n_samples,)` int, 1-12 — calendar month of each sample, for
        the climatology baseline's seasonal lookup.
    time:
        `(n_samples,)` datetime64 — the sample timestamps, carried
        through for reporting/auditing.
    channel_names, depth_names:
        Column labels for `features` and `profile`/`profile_mask`.
    variable_names:
        One entry per `profile`/`depth_names` column, naming which
        `TensorBundle.target_names` variable that column came from
        (Phase 25's per-variable metrics; see
        `src.evaluation.metrics.compute_profile_metrics`). Empty when
        the bundle recorded no target names.
    split:
        Which of `SPLIT_NAMES` this is.
    source_tensors_path:
        Where the `TensorBundle` these pooled arrays were built from
        was loaded from (or `None` if built in-memory), so a report can
        record — and `evaluate_baseline` can check — exactly which
        `.npz` file every model in a comparison actually used.
    """

    features: np.ndarray
    profile: np.ndarray
    profile_mask: np.ndarray
    months: np.ndarray
    time: np.ndarray
    channel_names: List[str]
    depth_names: List[str]
    split: str
    source_tensors_path: Optional[str] = None
    variable_names: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.profile.shape[0])

    @property
    def n_depth(self) -> int:
        return int(self.profile.shape[1])


def _depth_names_for_columns(depth: Optional[np.ndarray], n_depth_total: int) -> List[str]:
    """Depth labels (e.g. `"0m"`) for every profile column, `n_depth_total`
    of them.

    For a single-variable bundle, `depth` already has one entry per
    column and is used as-is. For a multi-variable bundle,
    `TensorAssembler.assemble` concatenates each variable's own depth
    axis in turn (see `_variable_names_for_depths`), so the *same*
    `depth` levels repeat once per variable; this tiles `depth`'s labels
    to match, so column `c`'s depth label lines up with column `c`'s
    entry in `_variable_names_for_depths`'s output. Falls back to a
    plain column index when there is no `depth` axis, or when
    `n_depth_total` isn't an exact multiple of `len(depth)` (a bundle
    this module's own assembler wouldn't produce).
    """
    if depth is None:
        return [str(i) for i in range(n_depth_total)]
    base = [f"{int(d)}m" for d in np.asarray(depth)]
    if len(base) == n_depth_total:
        return base
    if len(base) > 0 and n_depth_total % len(base) == 0:
        repeats = n_depth_total // len(base)
        return base * repeats
    return [str(i) for i in range(n_depth_total)]


def _variable_names_for_depths(
    target_names: Sequence[str], n_depth_total: int
) -> List[str]:
    """Expand `TensorBundle.target_names` to one label per profile column.

    `TensorAssembler.assemble` (`src/data/preprocessing/tensors.py`)
    builds a multi-variable target by concatenating each variable's own
    `(time, depth, lat, lon)` stack along the depth axis, in
    `target_names` order — variable 0's depths first, then variable 1's,
    and so on. This mirrors that layout so column `c` here names the
    same variable `pool_tensor_bundle` pooled column `c` from. Returns
    `[]` (no labels) when there are no target names or the column count
    is not an exact multiple of the variable count — the latter should
    not happen for a bundle `TensorAssembler` produced, but a hand-built
    or foreign bundle should get an empty (rather than a wrong) label
    list.
    """
    names = list(target_names)
    if not names or n_depth_total == 0:
        return []
    if n_depth_total % len(names) != 0:
        return []
    per_variable = n_depth_total // len(names)
    return [name for name in names for _ in range(per_variable)]


def pool_tensor_bundle(
    bundle: TensorBundle,
    *,
    splits: Sequence[str] = SPLIT_NAMES,
    source_tensors_path: Optional[PathLike] = None,
) -> Dict[str, PooledSplit]:
    """Build a `PooledSplit` for every split named in `bundle.split_masks`.

    This is the single point every baseline (and any future adapter for
    NEER's own pooled predictions) should go through to read
    `TensorBundle` — never re-slicing `bundle.split_masks` by hand — so
    that "same data" is a structural guarantee, not a convention.
    """
    missing = [s for s in splits if s not in bundle.split_masks]
    if missing:
        raise KeyError(
            f"tensor bundle has no split(s) {missing}; available: {sorted(bundle.split_masks)}"
        )

    n_depth_total = bundle.targets.shape[1]
    depth_names = _depth_names_for_columns(bundle.depth, n_depth_total)
    months_all = month_of(bundle.time)
    path_str = None if source_tensors_path is None else str(source_tensors_path)
    variable_names = _variable_names_for_depths(bundle.target_names, len(depth_names))

    pooled: Dict[str, PooledSplit] = {}
    for name in splits:
        mask = np.asarray(bundle.split_masks[name], dtype=bool)

        features, _ = _weighted_spatial_mean(bundle.inputs[mask], bundle.input_mask[mask])
        if bundle.targets is None or bundle.target_mask is None:
            raise ValueError("tensor bundle has no targets; cannot build a PooledSplit")
        profile, profile_mask = _weighted_spatial_mean(
            bundle.targets[mask], bundle.target_mask[mask]
        )

        pooled[name] = PooledSplit(
            features=features.astype(np.float32),
            profile=profile.astype(np.float32),
            profile_mask=profile_mask,
            months=months_all[mask],
            time=bundle.time[mask],
            channel_names=list(bundle.channel_names),
            depth_names=depth_names,
            split=name,
            source_tensors_path=path_str,
            variable_names=variable_names,
        )
    return pooled


class BaselineModel(ABC):
    """Common interface every baseline in `src/models/baselines` implements.

    A model is fit once, on the `train` `PooledSplit` only (Phase 24's
    "do not compare models using different data splits" applies to
    fitting too — a baseline that peeked at val/test would not be a
    fair comparison), and then predicts a `(n_samples, n_depth)` profile
    for any split's features.
    """

    #: Short, human-readable name used in reports. Subclasses override.
    name: str = "baseline"

    @abstractmethod
    def fit(self, train: PooledSplit) -> "BaselineModel":
        """Fit on the training split only. Returns `self` for chaining."""

    @abstractmethod
    def predict(self, split: PooledSplit) -> np.ndarray:
        """Predict a `(n_samples, n_depth)` profile for `split`."""

    def config(self) -> Dict[str, Any]:
        """JSON-safe hyperparameters, for the evaluation report."""
        return {}


@dataclass
class EvaluationReport:
    """One model's metrics across every split it was evaluated on, plus
    enough provenance to confirm every model in a comparison used the
    same data."""

    model_name: str
    model_config: Dict[str, Any]
    metrics: Dict[str, ProfileMetrics]
    source_tensors_path: Optional[str]
    split_sample_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "config": self.model_config,
            "source_tensors_path": self.source_tensors_path,
            "split_sample_counts": self.split_sample_counts,
            "metrics": {split: m.to_dict() for split, m in self.metrics.items()},
        }


def _assert_same_source(pooled: Dict[str, PooledSplit]) -> Optional[str]:
    """Every split must trace back to the same tensors file (or all
    in-memory) — the structural version of "do not compare models using
    different data splits": a caller cannot even build a mismatched
    `PooledSplit` dict without this raising first."""
    sources = {split.source_tensors_path for split in pooled.values()}
    if len(sources) > 1:
        raise ValueError(
            "PooledSplit dict mixes splits built from different tensor bundles: "
            f"{sources}. Build every split from one pool_tensor_bundle() call."
        )
    return next(iter(sources))


def evaluate_baseline(
    model: BaselineModel,
    pooled: Dict[str, PooledSplit],
    *,
    fit: bool = True,
) -> EvaluationReport:
    """Fit `model` on `pooled["train"]` (unless `fit=False`, for a model
    fit elsewhere) and score it on every split in `pooled`, via the one
    shared metrics computation in `src.evaluation.metrics`.
    """
    source = _assert_same_source(pooled)
    if fit:
        if "train" not in pooled:
            raise KeyError("evaluate_baseline needs a 'train' split to fit on")
        model.fit(pooled["train"])

    metrics: Dict[str, ProfileMetrics] = {}
    counts: Dict[str, int] = {}
    for split_name, split in pooled.items():
        prediction = model.predict(split)
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != split.profile.shape:
            raise ValueError(
                f"{model.name}.predict returned shape {prediction.shape}, "
                f"expected {split.profile.shape} (split='{split_name}')"
            )
        metrics[split_name] = compute_profile_metrics(
            split.profile,
            prediction,
            split.profile_mask,
            depth_names=split.depth_names,
            variable_names=split.variable_names or None,
        )
        counts[split_name] = split.n_samples

    return EvaluationReport(
        model_name=model.name,
        model_config=model.config(),
        metrics=metrics,
        source_tensors_path=source,
        split_sample_counts=counts,
    )


def evaluate_baselines(
    models: Sequence[BaselineModel],
    pooled: Dict[str, PooledSplit],
) -> List[EvaluationReport]:
    """`evaluate_baseline` for several models against the *same* `pooled`
    dict — the common-interface entry point `scripts/evaluate_baselines.py`
    uses so every model in one run is guaranteed to have used identical
    train/val/test data (one `pool_tensor_bundle` call, shared by
    reference across every `evaluate_baseline` call here)."""
    return [evaluate_baseline(model, pooled) for model in models]