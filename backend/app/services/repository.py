"""
Phase 29A — `NEERRepository`: the one place the FastAPI layer touches
NEER's data/model modules.

This is glue, not model logic: it loads (once, at construction) whatever
of the following actually exists on disk —

* the preprocessed tensor bundle (`data/processed/neer_tensors.npz`,
  written by `scripts/preprocess_data.py`) — `src.data.preprocessing.tensors.TensorBundle`
* a trained checkpoint (`artifacts/checkpoints/neer_best.pt`, then
  `neer_last.pt`, then any `*.pt` — written by `scripts/train.py`) —
  loaded through `src.inference.service.InferenceService`, exactly the
  object Phase 28 built for serving
* an optional fitted climatology (`data/processed/climatology.npz`) —
  `src.data.preprocessing.climatology.MonthlyClimatology`

— and turns "give me a prediction for this date/location" into calls on
`InferenceService`. No forward pass, tensor assembly, or checkpoint
parsing logic is reimplemented here; see `src/inference/service.py`,
`src/data/preprocessing/tensors.py`, and `src/training/checkpoint.py`.

Every failure mode is reported honestly (Requirement 5/12 of Phase 29A):
if a checkpoint or dataset is missing or fails to load, that is recorded
and surfaced through `health()`/`model_info()`/etc. as an actual
unavailable status — nothing here fabricates a checkpoint, a date list,
or a prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.errors import (
    DataQualityFailedError,
    DataUnavailableError,
    DateNotFoundError,
    ExplainabilityFailedError,
    GridUnavailableError,
    InferenceFailedError,
    InvalidParameterError,
    ModelUnavailableError,
)
from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing._utils import json_safe
from src.data.preprocessing.tensors import DEFAULT_TARGETS, TensorBundle
from src.inference.service import InferenceService, PredictionResult
from src.utils.config import ConfigError, ConfigValidationError, NeerConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Checkpoint filenames tried, in preference order, inside
#: `artifacts_checkpoints` (matches `scripts/train.py`'s naming).
_CHECKPOINT_CANDIDATES: Tuple[str, ...] = ("neer_best.pt", "neer_last.pt")

DEFAULT_MAX_GRID_POINTS = 4096


def _project_path(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _find_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    for name in _CHECKPOINT_CANDIDATES:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    others = sorted(checkpoint_dir.glob("*.pt"))
    return others[-1] if others else None


class NEERRepository:
    """Loads NEER's data/model once and serves the API's lookups.

    Every constructor argument has a config-derived default, so
    `NEERRepository()` is the production path (Requirement 18: the
    `environment` config picks paths the same way every other NEER
    script does). Tests pass explicit paths to point this at a small,
    self-contained fixture instead of the real `data/`/`artifacts/`
    trees.
    """

    def __init__(
        self,
        environment: Optional[str] = None,
        *,
        tensor_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        climatology_path: Optional[Path] = None,
        data_raw_path: Optional[Path] = None,
        cache_size: int = 128,
        max_grid_points: int = DEFAULT_MAX_GRID_POINTS,
    ) -> None:
        self.max_grid_points = int(max_grid_points)
        self._errors: Dict[str, str] = {}

        self.config: Optional[NeerConfig] = None
        try:
            self.config = load_config(environment)
        except (ConfigError, ConfigValidationError) as exc:
            self._errors["config"] = str(exc)

        paths = {} if self.config is None else dict(self.config.raw.get("paths", {}))

        self.tensor_path = tensor_path or _project_path(
            paths.get("data_processed", "data/processed")
        ) / "neer_tensors.npz"
        self.metadata_path = metadata_path or _project_path(
            paths.get("data_processed", "data/processed")
        ) / "neer_preprocessing_metadata.json"
        self.climatology_path = climatology_path or _project_path(
            paths.get("data_processed", "data/processed")
        ) / "climatology.npz"
        self.checkpoint_path = checkpoint_path or _find_checkpoint(
            _project_path(paths.get("artifacts_checkpoints", "artifacts/checkpoints"))
        )
        #: Where GET /evaluation/argo looks for real ARGO data when `demo`
        #: isn't requested (Phase 29B-1) — same config-derived-default
        #: pattern as every other path here (Requirement 18).
        self.data_raw_path = data_raw_path or _project_path(paths.get("data_raw", "data/raw"))

        self.bundle: Optional[TensorBundle] = None
        self._date_index: Dict[str, int] = {}
        self._date_strings: List[str] = []
        self._load_data()

        self.climatology = None
        self._load_climatology()

        self.service: Optional[InferenceService] = None
        self._checkpoint_info: Dict[str, Any] = {}
        self._load_model()

    # -- loading -------------------------------------------------------

    def _load_data(self) -> None:
        if not self.tensor_path.exists():
            self._errors["data"] = (
                f"no tensor bundle at {self.tensor_path}; run "
                "'python scripts/preprocess_data.py' first"
            )
            return
        try:
            self.bundle = TensorBundle.load(self.tensor_path)
        except (OSError, ValueError, KeyError) as exc:
            self._errors["data"] = f"failed to load {self.tensor_path}: {exc}"
            return

        self._date_strings = [
            str(np.datetime_as_string(np.asarray(t).astype("datetime64[D]")))
            for t in self.bundle.time
        ]
        self._date_index = {date: i for i, date in enumerate(self._date_strings)}

    def _load_climatology(self) -> None:
        if not self.climatology_path.exists():
            return
        try:
            from src.data.preprocessing.climatology import MonthlyClimatology

            self.climatology = MonthlyClimatology.load(self.climatology_path)
        except (OSError, ValueError, KeyError) as exc:
            self._errors["climatology"] = f"failed to load {self.climatology_path}: {exc}"

    def _load_model(self) -> None:
        if self.checkpoint_path is None:
            self._errors["model"] = (
                "no checkpoint found under artifacts/checkpoints/ "
                "(expected neer_best.pt or neer_last.pt); run 'python scripts/train.py' first"
            )
            return
        if not self.checkpoint_path.exists():
            self._errors["model"] = f"checkpoint not found: {self.checkpoint_path}"
            return

        try:
            import torch  # noqa: F401  (presence check; MissingDependencyError otherwise)
        except ImportError:
            self._errors["model"] = (
                "torch is not installed; install it to serve model endpoints (pip install torch)"
            )
            return

        target_variable = None
        if self.bundle is not None and self.bundle.target_names:
            target_variable = self.bundle.target_names[0]
        target_variable = target_variable or DEFAULT_TARGETS[0]

        kwargs: Dict[str, Any] = {
            "climatology": self.climatology,
            "cache_size": 128,
            "data_mode": "UNKNOWN" if self.bundle is None else str(
                self.bundle.attrs.get("data_mode", "UNKNOWN")
            ),
        }
        if self.metadata_path.exists():
            kwargs["metadata_path"] = str(self.metadata_path)
            kwargs["target_variable"] = target_variable

        try:
            self.service = InferenceService.from_checkpoint(str(self.checkpoint_path), **kwargs)
        except MissingDependencyError as exc:
            self._errors["model"] = str(exc)
            return
        except Exception as exc:  # noqa: BLE001 - any load failure is a real "unavailable"
            self._errors["model"] = f"failed to load checkpoint {self.checkpoint_path}: {exc}"
            return

        self._checkpoint_info = self._read_checkpoint_metadata()

    def _read_checkpoint_metadata(self) -> Dict[str, Any]:
        """Best-effort read of the checkpoint's own recorded fields
        (epoch, val_loss, training args) for `/model/info`. A failure here
        does not tear down an already-loaded `service`."""
        try:
            import torch

            raw = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            "path": str(self.checkpoint_path),
            "epoch": raw.get("epoch"),
            "val_loss": raw.get("val_loss"),
            "best_val_loss": raw.get("best_val_loss"),
            "training_args": raw.get("args"),
        }

    # -- status ----------------------------------------------------------

    @property
    def is_data_loaded(self) -> bool:
        return self.bundle is not None

    @property
    def is_model_loaded(self) -> bool:
        return self.service is not None

    def health(self) -> Dict[str, Any]:
        """Actual component status — never fabricated (Requirement 5)."""
        data_component: Dict[str, Any] = {"status": "ok" if self.is_data_loaded else "unavailable"}
        if self.bundle is not None:
            data_component.update(
                {
                    "n_timesteps": self.bundle.n_time,
                    "grid_shape": list(self.bundle.grid_shape),
                    "data_mode": str(self.bundle.attrs.get("data_mode", "UNKNOWN")),
                    "is_synthetic": bool(self.bundle.is_synthetic),
                }
            )
        elif "data" in self._errors:
            data_component["detail"] = self._errors["data"]

        model_component: Dict[str, Any] = {"status": "ok" if self.is_model_loaded else "unavailable"}
        if self.service is not None:
            model_component.update(
                {"device": self.service.device, "checkpoint": str(self.checkpoint_path)}
            )
        elif "model" in self._errors:
            model_component["detail"] = self._errors["model"]

        climatology_component = {"status": "ok" if self.climatology is not None else "unavailable"}
        if "climatology" in self._errors:
            climatology_component["detail"] = self._errors["climatology"]

        overall = "ok" if (self.is_data_loaded and self.is_model_loaded) else "degraded"
        return {
            "status": overall,
            "components": {
                "api": {"status": "ok"},
                "data": data_component,
                "model": model_component,
                "climatology": climatology_component,
            },
        }

    def model_info(self) -> Dict[str, Any]:
        if self.service is None:
            raise ModelUnavailableError(
                self._errors.get("model", "no NEER model checkpoint is loaded")
            )
        model = self.service.model
        info: Dict[str, Any] = {
            "architecture": {
                "in_channels": int(model.in_channels),
                "embed_dim": int(model.embed_dim),
                "num_depths": int(model.num_depths),
                "depths": [float(d) for d in model.depths],
                "use_gnn": bool(model.use_gnn),
                "uncertainty_enabled": bool(model.uncertainty_enabled),
            },
            "runtime": self.service.describe(),
            "checkpoint": json_safe(self._checkpoint_info),
        }
        if self.config is not None:
            info["environment"] = self.config.environment
        return info

    # -- dates -------------------------------------------------------------

    def list_dates(self) -> Dict[str, Any]:
        if self.bundle is None:
            raise DataUnavailableError(self._errors.get("data", "no dataset is loaded"))
        dates = sorted(self._date_strings)
        return {
            "dates": dates,
            "count": len(dates),
            "min_date": dates[0] if dates else None,
            "max_date": dates[-1] if dates else None,
            "data_mode": str(self.bundle.attrs.get("data_mode", "UNKNOWN")),
            "is_synthetic": bool(self.bundle.is_synthetic),
        }

    def _input_for_date(self, date: str) -> np.ndarray:
        if self.bundle is None:
            raise DataUnavailableError(self._errors.get("data", "no dataset is loaded"))
        index = self._date_index.get(date)
        if index is None:
            nearest = sorted(self._date_strings, key=lambda d: abs(
                np.datetime64(d) - np.datetime64(date)
            ))[:5]
            raise DateNotFoundError(
                f"date '{date}' is not available in the loaded dataset",
                extra={"nearest_available_dates": nearest},
            )
        return self.bundle.inputs[index]

    # -- coordinate/region validation --------------------------------------

    def _validate_latlon(self, lat: float, lon: float) -> None:
        if self.config is None:
            return
        domain = self.config.domain
        if not (domain.lat_min <= lat <= domain.lat_max):
            raise InvalidParameterError(
                f"lat={lat} is outside the NEER domain "
                f"[{domain.lat_min}, {domain.lat_max}]"
            )
        if not (domain.lon_min <= lon <= domain.lon_max):
            raise InvalidParameterError(
                f"lon={lon} is outside the NEER domain "
                f"[{domain.lon_min}, {domain.lon_max}]"
            )

    def _grid_slice(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.bundle is None:
            raise DataUnavailableError(self._errors.get("data", "no dataset is loaded"))
        if lat_min >= lat_max:
            raise InvalidParameterError("lat_min must be less than lat_max")
        if lon_min >= lon_max:
            raise InvalidParameterError("lon_min must be less than lon_max")
        self._validate_latlon(lat_min, lon_min)
        self._validate_latlon(lat_max, lon_max)

        lat_idx = np.where((self.bundle.lat >= lat_min) & (self.bundle.lat <= lat_max))[0]
        lon_idx = np.where((self.bundle.lon >= lon_min) & (self.bundle.lon <= lon_max))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            raise GridUnavailableError(
                "no grid cells fall inside the requested region "
                f"(lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}])"
            )
        n_points = int(lat_idx.size * lon_idx.size)
        if n_points > self.max_grid_points:
            raise InvalidParameterError(
                f"requested region has {n_points} grid cells, which exceeds the "
                f"{self.max_grid_points}-cell limit; narrow lat_min/lat_max/lon_min/lon_max"
            )
        return self.bundle.lat[lat_idx], self.bundle.lon[lon_idx]

    # -- reconstruction -----------------------------------------------------

    def _require_service(self) -> InferenceService:
        if self.service is None:
            raise ModelUnavailableError(
                self._errors.get("model", "no NEER model checkpoint is loaded")
            )
        return self.service

    # -- public accessors for the evaluation layer (Phase 29B-1) -----------
    #
    # `backend/app/services/evaluation.py` builds on `bundle`/`service`
    # exactly like `reconstruct_point` etc. below do; these are the same
    # "raise a real NeerApiError, never fabricate" checks as `_require_service`,
    # just public since evaluation.py lives outside this class.

    def require_bundle(self) -> TensorBundle:
        if self.bundle is None:
            raise DataUnavailableError(self._errors.get("data", "no dataset is loaded"))
        return self.bundle

    def require_service(self) -> InferenceService:
        return self._require_service()

    def model_error(self) -> str:
        """Human-readable reason no model is loaded, for callers (e.g. the
        evaluation layer) that want to explain *why* without raising."""
        return self._errors.get("model", "no NEER model checkpoint is loaded")

    def reconstruct_point(self, *, lat: float, lon: float, date: str, depth: float) -> PredictionResult:
        self._validate_latlon(lat, lon)
        if depth < 0:
            raise InvalidParameterError("depth must be non-negative")
        service = self._require_service()
        x = self._input_for_date(date)
        try:
            return service.predict_point(x, lat=lat, lon=lon, date=date, depth=depth)
        except (ValueError, RuntimeError) as exc:
            raise InferenceFailedError(f"reconstruction failed: {exc}") from exc

    def reconstruct_profile(self, *, lat: float, lon: float, date: str) -> PredictionResult:
        self._validate_latlon(lat, lon)
        service = self._require_service()
        x = self._input_for_date(date)
        try:
            return service.predict_profile(x, lat=lat, lon=lon, date=date)
        except (ValueError, RuntimeError) as exc:
            raise InferenceFailedError(f"reconstruction failed: {exc}") from exc

    def reconstruct_grid(
        self,
        *,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        date: str,
        depth: Optional[float] = None,
    ) -> PredictionResult:
        if depth is not None and depth < 0:
            raise InvalidParameterError("depth must be non-negative")
        lats, lons = self._grid_slice(lat_min, lat_max, lon_min, lon_max)
        service = self._require_service()
        x = self._input_for_date(date)
        try:
            return service.predict_grid(x, lats=lats, lons=lons, date=date, depth=depth)
        except (ValueError, RuntimeError) as exc:
            raise InferenceFailedError(f"grid reconstruction failed: {exc}") from exc

    def embedding(self, *, date: str) -> Dict[str, Any]:
        service = self._require_service()
        x = self._input_for_date(date)
        # The pooled embedding depends only on the input sample for
        # `date` (see src/inference/__init__.py) — lat/lon just select
        # which cached forward pass's `.embedding` we read, so the
        # domain centre is as good an anchor as any real point.
        if self.config is not None:
            domain = self.config.domain
            anchor_lat = (domain.lat_min + domain.lat_max) / 2.0
            anchor_lon = (domain.lon_min + domain.lon_max) / 2.0
        else:
            anchor_lat = anchor_lon = 0.0
        try:
            result = service.predict_profile(x, lat=anchor_lat, lon=anchor_lon, date=date)
        except (ValueError, RuntimeError) as exc:
            raise InferenceFailedError(f"embedding computation failed: {exc}") from exc
        return {
            "date": date,
            "embedding": [float(v) for v in result.embedding],
            "dim": int(result.embedding.shape[0]),
            "data_mode": result.data_mode,
            "cache_hit": result.cache_hit,
            "latency_ms": result.latency_ms,
        }

    # -- explainability (Phase 29B-2) ---------------------------------------

    def explain(self, *, date: str, depth: Optional[float] = None) -> Dict[str, Any]:
        """Real gradient-based explainability for one date (Requirement 1).

        Runs an actual backward pass through the actual loaded model on
        the actual input tensor for `date` (`src.explainability.gradients.
        explain_point`) — never a fabricated/random/placeholder score.
        `depth=None` explains the sum of every model depth level's
        anomaly; a specific `depth` is snapped to the nearest model depth
        level, same convention as `reconstruct_point`.
        """
        service = self._require_service()
        x = self._input_for_date(date)
        channel_names = (
            self.bundle.channel_names if self.bundle is not None else list(DEFAULT_TARGETS)
        )

        depth_index: Optional[int] = None
        if depth is not None:
            if depth < 0:
                raise InvalidParameterError("depth must be non-negative")
            depth_index = int(np.argmin(np.abs(np.asarray(service.depths) - float(depth))))

        from src.explainability.gradients import explain_point

        try:
            result = explain_point(
                service.model,
                x,
                channel_names=channel_names,
                depths=service.depths,
                depth_index=depth_index,
            )
        except (ValueError, RuntimeError) as exc:
            raise ExplainabilityFailedError(f"explainability computation failed: {exc}") from exc

        result["date"] = str(date)
        result["data_mode"] = service.data_mode
        return result

    # -- data quality (Phase 29B-2) ------------------------------------------

    def data_quality(self) -> Dict[str, Any]:
        """Real NEER dataset quality (Requirement 2): reuses `TensorBundle`'s
        own summary/coverage/channel/target reporting — nothing here
        invents a statistic the loaded bundle doesn't already carry.

        Raises `DataUnavailableError` (503) if no dataset is loaded, and
        `DataQualityFailedError` (500) if computing the report itself
        raises for an otherwise-loaded bundle.
        """
        bundle = self.require_bundle()
        try:
            dates = self.list_dates()
            report: Dict[str, Any] = {
                "data_mode": str(bundle.attrs.get("data_mode", "UNKNOWN")),
                "is_synthetic": bool(bundle.is_synthetic),
                "disclaimer": bundle.attrs.get("disclaimer"),
                "source_tensors_path": str(self.tensor_path),
                "dates": {
                    "count": dates["count"],
                    "min_date": dates["min_date"],
                    "max_date": dates["max_date"],
                },
                "summary": bundle.summary(),
                "spatial_coverage": bundle.spatial_coverage(),
                "channels": bundle.channel_quality(),
                "targets": bundle.target_quality(),
            }
        except (ValueError, KeyError) as exc:
            raise DataQualityFailedError(f"data quality computation failed: {exc}") from exc
        return json_safe(report)