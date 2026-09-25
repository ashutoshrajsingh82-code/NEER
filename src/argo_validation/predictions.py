"""
Phase 26 — the NEER side of the comparison: predicted temperature profiles.

The validation pipeline never runs a model itself; it consumes a
`NeerPredictions` — temperature in **degC** on NEER's depth levels, one
profile per NEER timestep — and this module has the ways to obtain one:

* `predictions_from_checkpoint` — run a trained `NEERModel` checkpoint
  (`scripts/train.py` output) over a `TensorBundle` and put its output
  back into degC. Needs torch.
* `NeerPredictions.save` / `load_prediction_table` — a torch-free `.npz`
  interchange file, so predictions produced elsewhere (or earlier) can be
  validated without the model installed.
* (`src.argo_validation.demo.demo_stand_in_predictions` — a *labelled
  stand-in that is not NEER*, for demo runs only.)

What NEER actually predicts (read this before interpreting metrics)
--------------------------------------------------------------------
`NEERModel.forward` returns ONE pooled profile per input sample, and
`scripts/train.py` trains it against the mask-weighted *spatial mean* of
the z-scored subsurface temperature over the whole domain (see
`pool_profile`). So, per timestep, NEER's output is a **domain-mean
profile**, not a field. `predictions_from_checkpoint` undoes the per-depth
z-scoring with the statistics saved in the preprocessing metadata
(`Normalizer.state`) — a linear map, so a mean of z-scores becomes the mean
in degC exactly.

Consequence: every ARGO profile matched to the same month is compared to
the *same* predicted profile, wherever in the basin it was taken. That is a
real limitation of the current model output, not of this pipeline: the
error includes the spatial spread of ARGO profiles around the domain mean.
`NeerPredictions.resolution` records it (``domain_pooled``) and the report
repeats it. A model that later predicts a gridded field can supply
`gridded_temperature_c` and the pipeline will use the matched cell's
profile instead — no other change needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.data.loaders.errors import MissingDependencyError

PathLike = Union[str, Path]

ORIGIN_CHECKPOINT = "neer_checkpoint"
ORIGIN_TABLE = "user_prediction_table"
ORIGIN_DEMO_STAND_IN = "demo_stand_in_NOT_NEER"

RESOLUTION_POOLED = "domain_pooled"
RESOLUTION_GRIDDED = "gridded"


@dataclass(frozen=True)
class NeerPredictions:
    time: np.ndarray                      # (n_time,) datetime64
    depth_m: np.ndarray                   # (n_depth,)
    temperature_c: np.ndarray             # (n_time, n_depth) domain-pooled, degC
    origin: str = ORIGIN_TABLE
    source: Optional[str] = None
    gridded_temperature_c: Optional[np.ndarray] = None  # (n_time, n_depth, n_lat, n_lon)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        t = np.asarray(self.temperature_c)
        if t.ndim != 2 or t.shape != (len(self.time), len(self.depth_m)):
            raise ValueError(
                f"temperature_c has shape {t.shape}; expected ({len(self.time)}, {len(self.depth_m)})"
            )
        g = self.gridded_temperature_c
        if g is not None and (g.ndim != 4 or g.shape[:2] != t.shape):
            raise ValueError(f"gridded_temperature_c has shape {g.shape}; expected (time, depth, lat, lon)")

    @property
    def resolution(self) -> str:
        return RESOLUTION_GRIDDED if self.gridded_temperature_c is not None else RESOLUTION_POOLED

    @property
    def is_neer_output(self) -> bool:
        """True only for predictions this code itself produced from a NEER
        checkpoint. A user-supplied table is not verified either way."""
        return self.origin == ORIGIN_CHECKPOINT

    def at(self, time_index: int, lat_index: int, lon_index: int) -> np.ndarray:
        if self.gridded_temperature_c is not None:
            return np.asarray(self.gridded_temperature_c[time_index, :, lat_index, lon_index], dtype=float)
        return np.asarray(self.temperature_c[time_index], dtype=float)

    def describe(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "is_neer_output": self.is_neer_output,
            "source": self.source,
            "resolution": self.resolution,
            "quantity": "temperature (degC) on NEER depth levels",
            "n_time": int(len(self.time)),
            "depth_m": [float(d) for d in self.depth_m],
            "notes": list(self.notes),
        }

    # -- interchange file (pooled predictions only) ---------------------------

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"origin": self.origin, "source": self.source, "notes": self.notes}
        np.savez_compressed(
            path,
            time=self.time,
            depth_m=self.depth_m,
            temperature_c=self.temperature_c,
            _metadata=np.array(json.dumps(meta)),
        )
        return path


def load_prediction_table(path: PathLike) -> NeerPredictions:
    """Load a pooled-prediction `.npz` (`time`, `depth_m`, `temperature_c`)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        for key in ("time", "depth_m", "temperature_c"):
            if key not in data.files:
                raise ValueError(f"{path}: missing array '{key}' (found {data.files})")
        meta = json.loads(str(data["_metadata"])) if "_metadata" in data.files else {}
        return NeerPredictions(
            time=np.asarray(data["time"]),
            depth_m=np.asarray(data["depth_m"], dtype=float),
            temperature_c=np.asarray(data["temperature_c"], dtype=float),
            origin=meta.get("origin", ORIGIN_TABLE),
            source=meta.get("source") or str(path),
            notes=list(meta.get("notes", [])),
        )


# --------------------------------------------------------------------------
# From a trained checkpoint
# --------------------------------------------------------------------------


def normalizer_stats_from_metadata(metadata_path: PathLike, variable: str, n_depth: int):
    """`(center, scale)` arrays of length `n_depth` for `variable`, from a
    `neer_preprocessing_metadata.json` (the `normalization` step's `state`)."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    step = next((s for s in meta.get("steps", []) if s.get("step") == "normalization"), None)
    stats = None if step is None else step.get("state", {}).get("statistics", {}).get(variable)
    if stats is None:
        raise ValueError(
            f"{metadata_path}: no normalization statistics for '{variable}'; "
            "cannot convert NEER output back to degC"
        )
    if step["state"].get("method") != "zscore":
        raise ValueError(f"only zscore normalization is supported here, got {step['state'].get('method')!r}")
    center = np.asarray(stats["center"], dtype=float).ravel()
    scale = np.asarray(stats["scale"], dtype=float).ravel()
    if center.size == 1:
        center, scale = np.repeat(center, n_depth), np.repeat(scale, n_depth)
    if center.size != n_depth:
        raise ValueError(f"normalization stats have {center.size} levels; model has {n_depth}")
    return center, scale


def predictions_from_checkpoint(
    checkpoint_path: PathLike,
    bundle,
    metadata_path: PathLike,
    *,
    batch_size: int = 4,
) -> NeerPredictions:
    """Run a trained `NEERModel` over every timestep of `bundle` and return
    domain-pooled predictions in degC.

    Mirrors `scripts/run_evaluation.py::evaluate_neer_checkpoint` for model
    construction, so the same checkpoint loads the same way in both places.
    Requires torch; raises `MissingDependencyError` otherwise.
    """
    try:
        import torch

        from src.models.neer_model import NEERModel
        from src.utils.config import load_config
    except ImportError as exc:
        raise MissingDependencyError(
            package="torch",
            purpose="running a NEER checkpoint for ARGO validation",
            install_hint="pip install torch  (or pass --predictions with a saved prediction table)",
        ) from exc

    if len(bundle.target_names) != 1:
        raise ValueError(
            f"ARGO validation supports a single target variable; bundle has {bundle.target_names}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    environment = checkpoint.get("args", {}).get("environment", "demo")
    model = NEERModel.from_config(load_config(environment))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, bundle.n_time, batch_size):
            x = torch.from_numpy(np.asarray(bundle.inputs[start : start + batch_size], dtype=np.float32))
            outputs.append(model(x).cpu().numpy())
    pooled_z = np.concatenate(outputs, axis=0).astype(float)  # (n_time, n_depth), z-scored

    if pooled_z.shape[1] != len(bundle.depth):
        raise ValueError(f"model emits {pooled_z.shape[1]} depths; bundle has {len(bundle.depth)}")
    center, scale = normalizer_stats_from_metadata(metadata_path, bundle.target_names[0], pooled_z.shape[1])
    return NeerPredictions(
        time=np.asarray(bundle.time),
        depth_m=np.asarray(bundle.depth, dtype=float),
        temperature_c=pooled_z * scale + center,
        origin=ORIGIN_CHECKPOINT,
        source=str(checkpoint_path),
        notes=[
            "NEER outputs one domain-pooled profile per timestep (mask-weighted spatial mean); "
            "converted to degC with the train-period per-depth z-score statistics.",
        ],
    )