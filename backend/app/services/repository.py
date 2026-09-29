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

import json
import tempfile
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
    NetCDFGenerationFailedError,
    NetCDFUnavailableError,
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
            "filename": self.checkpoint_path.name,
            "version": raw.get("version"),
            "model_version": raw.get("model_version"),
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
                {
                    "device": self.service.device,
                    "checkpoint": self.checkpoint_path.name if self.checkpoint_path else None,
                }
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

    def demo_context(self) -> Dict[str, Any]:
        """Return a reproducible selection only when the configured demo stack is live."""
        if self.config is None or not self.config.demo.enabled:
            raise DataUnavailableError("The backend is not running with the enabled demo configuration.")
        if self.bundle is None or self.service is None or self.checkpoint_path is None:
            raise DataUnavailableError("The demo requires a loaded demo dataset, model, and checkpoint.")
        if not self.bundle.is_synthetic:
            raise DataUnavailableError("The active dataset is not marked as the NEER synthetic demonstration dataset.")
        dates = self.list_dates()["dates"]
        if not dates:
            raise DataUnavailableError("The loaded demo dataset has no available dates.")
        demo_depth = 100.0
        if demo_depth not in self.service.depths:
            raise DataUnavailableError("The loaded model does not provide the required 100 m demonstration depth.")
        if self.bundle.ocean_mask is None:
            raise DataUnavailableError("The loaded demo tensor bundle has no ocean mask for choosing a valid location.")
        rows, cols = np.where(np.asarray(self.bundle.ocean_mask, dtype=bool))
        if rows.size == 0:
            raise DataUnavailableError("The loaded demo tensor bundle contains no valid ocean grid locations.")
        center_lat = (self.config.domain.lat_min + self.config.domain.lat_max) / 2
        center_lon = (self.config.domain.lon_min + self.config.domain.lon_max) / 2
        best = int(np.argmin((self.bundle.lat[rows] - center_lat) ** 2 + (self.bundle.lon[cols] - center_lon) ** 2))
        return {
            "available": True,
            "environment": self.config.environment,
            "dataset_mode": str(self.bundle.attrs.get("data_mode", "UNKNOWN")),
            "dataset": self.bundle.attrs.get("dataset") or self.bundle.attrs.get("source") or "NEER demo tensor bundle",
            "date": dates[-1],
            "latitude": float(self.bundle.lat[rows[best]]),
            "longitude": float(self.bundle.lon[cols[best]]),
            "depth": demo_depth,
            "variable_mode": "temperature",
        }

    def _input_for_date(self, date: str) -> np.ndarray:
        if self.bundle is None:
            raise DataUnavailableError(self._errors.get("data", "no dataset is loaded"))
        index = self._date_index.get(date)
        if index is None:
            if not self._date_strings:
                raise DataUnavailableError("The loaded dataset has no dates.")
            source_dates = np.asarray(self._date_strings, dtype="datetime64[D]")
            requested = np.datetime64(date, "D")
            # The UI offers every calendar day in months represented by this
            # dataset. Between source timesteps, blend input fields by elapsed
            # time; at the first/last month edges, hold the nearest sample.
            first_month = source_dates[0].astype("datetime64[M]").astype("datetime64[D]")
            last_month = source_dates[-1].astype("datetime64[M]")
            after_last_month = (last_month + np.timedelta64(1, "M")).astype("datetime64[D]")
            if requested < first_month or requested >= after_last_month:
                nearest = sorted(self._date_strings, key=lambda d: abs(np.datetime64(d) - requested))[:5]
                raise DateNotFoundError(
                    f"date '{date}' is outside the months covered by the loaded dataset",
                    extra={"nearest_available_dates": nearest},
                )
            right = int(np.searchsorted(source_dates, requested, side="right"))
            if right == 0:
                return self.bundle.inputs[0]
            if right == len(source_dates):
                return self.bundle.inputs[-1]
            left = right - 1
            elapsed = float((requested - source_dates[left]) / (source_dates[right] - source_dates[left]))
            return (1.0 - elapsed) * self.bundle.inputs[left] + elapsed * self.bundle.inputs[right]
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

    def explain(self, *, date: str, lat: float, lon: float, depth: float) -> Dict[str, Any]:
        """Actual Integrated Gradients for one selected point prediction."""
        if depth < 0:
            raise InvalidParameterError("depth must be non-negative")
        self._validate_latlon(lat, lon)
        if self.bundle is None:
            raise DataUnavailableError(self._errors.get("data", "no dataset is loaded"))
        service = self._require_service()
        x = self._input_for_date(date)
        from src.data.preprocessing.channels import NEER_CHANNEL_ORDER, validate_channel_order

        try:
            validate_channel_order(self.bundle.channel_names)
        except ValueError as exc:
            raise ExplainabilityFailedError(f"explainability input channels are invalid: {exc}") from exc

        depth_matches = np.flatnonzero(np.isclose(np.asarray(service.depths), float(depth), rtol=0.0, atol=1e-6))
        if depth_matches.size == 0:
            raise InvalidParameterError(
                f"depth={depth} is unsupported; choose one of {list(service.depths)}"
            )
        depth_index = int(depth_matches[0])

        from src.explainability.gradients import explain_point

        try:
            output_center, output_scale = service.output_normalization(depth_index)
            result = explain_point(
                service.model,
                x,
                channel_names=NEER_CHANNEL_ORDER,
                depths=service.depths,
                depth_index=depth_index,
                output_center=output_center,
                output_scale=output_scale,
                # Keep the interactive explanation responsive; the exact
                # quadrature resolution is returned in the result payload.
                steps=16,
            )
            prediction = service.predict_point(x, lat=lat, lon=lon, date=date, depth=depth)
        except (ValueError, RuntimeError) as exc:
            raise ExplainabilityFailedError(f"explainability computation failed: {exc}") from exc

        if prediction.temperature is not None and not np.isclose(
            result["predicted_model_output"], prediction.temperature, rtol=1e-5, atol=1e-5
        ):
            raise ExplainabilityFailedError(
                "Integrated Gradients target does not match the reconstructed temperature for the same request"
            )
        climatology_coordinates = (
            self.climatology.nearest_coordinates(lat, lon)
            if self.climatology is not None
            else None
        )
        result["date"] = str(date)
        result.update(
            {
                "lat": float(lat),
                "lon": float(lon),
                "climatology_lat": None if climatology_coordinates is None else climatology_coordinates[0],
                "climatology_lon": None if climatology_coordinates is None else climatology_coordinates[1],
                "temperature": prediction.temperature,
                "climatology": prediction.climatology,
                "anomaly": prediction.anomaly,
                "data_mode": service.data_mode,
                "model_version": self._checkpoint_info.get("version") or self._checkpoint_info.get("model_version"),
                "checkpoint_epoch": self._checkpoint_info.get("epoch"),
                "target": "NEER decoder absolute subsurface temperature in degC; anomaly is derived by subtracting the available local climatology",
                "feature_order": list(NEER_CHANNEL_ORDER[:7]),
            }
        )
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
            input_total = int(bundle.input_mask.size)
            input_valid = int(np.asarray(bundle.input_mask, dtype=bool).sum())
            report["input_cells"] = {
                "n_cells": input_total,
                "n_valid": input_valid,
                "n_missing": input_total - input_valid,
                "scope": "All input channels × available dates × grid cells.",
            }
            derived_context = {"time_sin", "time_cos", "lat_norm", "lon_norm"}
            observed_indices = [
                index for index, name in enumerate(bundle.channel_names)
                if name not in derived_context
            ]
            observed_mask = np.asarray(bundle.input_mask[:, observed_indices], dtype=bool)
            observed_total = int(observed_mask.size)
            observed_valid = int(observed_mask.sum())
            report["observed_input_cells"] = {
                "n_cells": observed_total,
                "n_valid": observed_valid,
                "n_missing": observed_total - observed_valid,
                "channels": [bundle.channel_names[index] for index in observed_indices],
                "scope": "Physical input variables × available dates × grid cells; derived time/location context channels excluded.",
            }
            domain_mask = (
                np.ones(bundle.grid_shape, dtype=bool)
                if bundle.ocean_mask is None
                else np.asarray(bundle.ocean_mask, dtype=bool)
            )
            observed_grid = observed_mask.any(axis=(0, 1)) if observed_indices else np.zeros(bundle.grid_shape, dtype=bool)
            coverage_cells = np.where(domain_mask, observed_grid.astype(np.int8), -1)
            report["coverage_grid"] = {
                "lat": [float(value) for value in bundle.lat],
                "lon": [float(value) for value in bundle.lon],
                "cells": coverage_cells.tolist(),  # 1 valid, 0 missing, -1 outside known ocean mask
                "valid_cells": int(np.logical_and(domain_mask, observed_grid).sum()),
                "missing_cells": int(np.logical_and(domain_mask, ~observed_grid).sum()),
                "outside_cells": int((~domain_mask).sum()) if bundle.ocean_mask is not None else None,
                "scope": "A grid cell is valid when any input channel is observed on any available date.",
            }

            def axis_resolution(values: np.ndarray) -> Dict[str, Any]:
                axis = np.asarray(values, dtype=float)
                spacing = np.abs(np.diff(axis))
                if not spacing.size or not np.isfinite(spacing).all():
                    return {"uniform": None, "degrees": None}
                uniform = bool(np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-10))
                return {"uniform": uniform, "degrees": float(spacing[0]) if uniform else None}

            report["grid_metadata"] = {
                "lat_resolution": axis_resolution(bundle.lat),
                "lon_resolution": axis_resolution(bundle.lon),
                "lat_points": int(bundle.lat.size),
                "lon_points": int(bundle.lon.size),
                "target_depths": None if bundle.depth is None else [float(value) for value in bundle.depth],
            }
            preprocessing: Dict[str, Any] = {}
            if self.metadata_path.is_file():
                try:
                    with self.metadata_path.open("r", encoding="utf-8") as handle:
                        preprocessing = json.load(handle)
                except (OSError, ValueError, TypeError):
                    preprocessing = {}
            source_metadata = preprocessing.get("source") if isinstance(preprocessing, dict) else None
            source_metadata = source_metadata if isinstance(source_metadata, dict) else {}
            variable_metadata = source_metadata.get("variables", [])
            preprocessing_steps = preprocessing.get("steps", []) if isinstance(preprocessing, dict) else []
            preprocessing_steps = preprocessing_steps if isinstance(preprocessing_steps, list) else []
            report["provenance"] = {
                "source": source_metadata.get("source") or bundle.attrs.get("source"),
                "dataset_format": source_metadata.get("source_format"),
                "dataset_identifier": source_metadata.get("dataset_id"),
                "version": preprocessing.get("version") if isinstance(preprocessing, dict) else None,
                "preprocessing_pipeline": preprocessing.get("pipeline") if isinstance(preprocessing, dict) else None,
                "preprocessing_created_at": preprocessing.get("created_at") if isinstance(preprocessing, dict) else None,
                "temporal_frequency": next((
                    step.get("config", {}).get("frequency")
                    for step in preprocessing_steps
                    if isinstance(step, dict) and step.get("step") == "temporal_alignment"
                ), None) if isinstance(preprocessing, dict) else None,
                "variables": {
                    item["name"]: {
                        "units": item.get("units"),
                        "description": item.get("description"),
                        "dims": item.get("dims"),
                        "shape": item.get("shape"),
                        "missing_fraction": item.get("missing_fraction"),
                        "n_missing": item.get("n_missing"),
                    }
                    for item in variable_metadata
                    if isinstance(item, dict) and isinstance(item.get("name"), str)
                } if isinstance(variable_metadata, list) else {},
            }
        except (ValueError, KeyError) as exc:
            raise DataQualityFailedError(f"data quality computation failed: {exc}") from exc
        return json_safe(report)

    # -- NetCDF reconstruction (Phase 29B-2) ---------------------------------

    def reconstruct_netcdf(
        self,
        *,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        date: str,
        depth: Optional[float] = None,
    ) -> Path:
        """Write a real `reconstruct_grid()` result to a NetCDF file.

        Builds an `OceanDataset` from the actual `PredictionResult` (never
        a fabricated grid), then hands it to `save_netcdf`. A missing
        xarray/NetCDF engine is `NetCDFUnavailableError` (503); any
        failure assembling or writing the file is
        `NetCDFGenerationFailedError` (500) — never an empty or dummy
        `.nc` in its place.
        """
        result = self.reconstruct_grid(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            date=date,
            depth=depth,
        )
        handle = tempfile.NamedTemporaryFile(
            prefix="neer_reconstruct_", suffix=".nc", delete=False
        )
        handle.close()
        path = Path(handle.name)
        try:
            from src.data.loaders import save_netcdf

            dataset = _ocean_dataset_from_grid_result(result)
            model_version = self._checkpoint_info.get("model_version") or self._checkpoint_info.get("version")
            if model_version is not None:
                dataset.attrs["model_version"] = str(model_version)
            if self.checkpoint_path is not None:
                dataset.attrs["checkpoint"] = self.checkpoint_path.name
            if self.bundle is not None:
                for key in ("dataset", "dataset_name", "data_source"):
                    value = self.bundle.attrs.get(key)
                    if value is not None:
                        dataset.attrs[key] = str(value)
            save_netcdf(dataset, path)
            _validate_reconstruction_netcdf(path, dataset)
            return path
        except MissingDependencyError as exc:
            _unlink_quietly(path)
            raise NetCDFUnavailableError(str(exc)) from exc
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            _unlink_quietly(path)
            raise NetCDFGenerationFailedError(
                f"NetCDF generation failed: {exc}"
            ) from exc


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _validate_reconstruction_netcdf(path: Path, expected: Any) -> None:
    """Reopen and validate an export before its path reaches FileResponse."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("generated NetCDF file is missing or empty")
    try:
        import xarray as xr
    except ImportError as exc:
        raise MissingDependencyError("xarray is required to validate NetCDF exports") from exc
    with xr.open_dataset(path) as actual:
        if not actual.data_vars:
            raise ValueError("generated NetCDF contains no scientific variables")
        for name, variable in expected.variables.items():
            if name not in actual.data_vars:
                raise ValueError(f"generated NetCDF is missing variable {name!r}")
            exported = actual[name]
            expected_values = np.asarray(variable.values)
            values = np.asarray(exported.values)
            if tuple(exported.dims) != tuple(variable.dims) or values.shape != expected_values.shape:
                raise ValueError(f"generated NetCDF variable {name!r} has an unexpected shape")
            if not np.array_equal(values, expected_values, equal_nan=True):
                raise ValueError(f"generated NetCDF variable {name!r} failed round-trip validation")
            if variable.units and exported.attrs.get("units") != variable.units:
                raise ValueError(f"generated NetCDF variable {name!r} has missing or invalid units")
        for name, coordinates in expected.coords.items():
            if name not in actual.coords:
                raise ValueError(f"generated NetCDF is missing coordinate {name!r}")
            values = np.asarray(actual[name].values)
            if values.shape != coordinates.shape or not np.array_equal(values, coordinates, equal_nan=True):
                raise ValueError(f"generated NetCDF coordinate {name!r} failed validation")
        if not actual.attrs.get("title") or not actual.attrs.get("source"):
            raise ValueError("generated NetCDF is missing provenance metadata")


def _ocean_dataset_from_grid_result(result: PredictionResult):
    """Turn a real grid `PredictionResult` into an `OceanDataset`.

    Axes follow `OceanDataset`'s canonical order (`time`, `depth`, `lat`,
    `lon`), including a singleton depth coordinate for a depth slice. Values are
    the arrays `predict_grid` already produced — nothing is synthesized
    or resampled here.
    """
    from src.data.loaders.representation import OceanDataset, Variable

    lat = np.asarray(result.lat, dtype=np.float64)
    lon = np.asarray(result.lon, dtype=np.float64)
    temperature = np.asarray(result.temperature, dtype=np.float32)
    time = np.array([np.datetime64(result.date, "ns")])

    coords: Dict[str, np.ndarray] = {"time": time, "lat": lat, "lon": lon}
    variables: Dict[str, Variable] = {}

    if temperature.ndim == 2:
        depth_attr = float(np.asarray(result.depth).reshape(-1)[0])
        coords["depth"] = np.asarray([depth_attr], dtype=np.float64)
        temp_values = temperature[np.newaxis, np.newaxis, ...]
        temp_dims = ("time", "depth", "lat", "lon")
    elif temperature.ndim == 3:
        depth = np.asarray(result.depth, dtype=np.float64)
        coords["depth"] = depth
        # (n_lat, n_lon, n_depth) -> (time, depth, lat, lon)
        temp_values = np.transpose(temperature, (2, 0, 1))[np.newaxis, ...]
        temp_dims = ("time", "depth", "lat", "lon")
        depth_attr = None
    else:
        raise ValueError(
            "grid reconstruction temperature must be (n_lat, n_lon) or "
            f"(n_lat, n_lon, n_depth); got shape {temperature.shape}"
        )

    temp_attrs: Dict[str, Any] = {
        "long_name": "reconstructed sea water temperature",
        "source": "NEER reconstruct_grid",
    }
    if temperature.ndim == 2:
        temp_attrs["depth_m"] = depth_attr
    if not np.isfinite(temperature).any():
        temp_attrs["data_status"] = "unavailable"

    variables["temperature"] = Variable(
        name="temperature",
        values=temp_values,
        dims=temp_dims,
        units="degC",
        attrs=temp_attrs,
    )

    anomaly = np.asarray(result.anomaly, dtype=np.float32)
    if anomaly.shape != temperature.shape:
        raise ValueError("grid anomaly must match the returned temperature grid coordinates")
    anomaly_values = anomaly[np.newaxis, np.newaxis, ...] if anomaly.ndim == 2 else np.transpose(anomaly, (2, 0, 1))[np.newaxis, ...]
    variables["anomaly"] = Variable(
        name="anomaly", values=anomaly_values, dims=temp_dims, units="degC",
        attrs={
            "long_name": "model output temperature anomaly from climatology",
            "definition": "model output - climatology",
            **({"data_status": "unavailable"} if not np.isfinite(anomaly).any() else {}),
        },
    )

    if result.climatology is not None:
        climatology = np.asarray(result.climatology, dtype=np.float32)
        if climatology.shape == temperature.shape:
            if climatology.ndim == 2:
                clim_values = climatology[np.newaxis, np.newaxis, ...]
            else:
                clim_values = np.transpose(climatology, (2, 0, 1))[np.newaxis, ...]
            variables["climatology"] = Variable(
                name="climatology",
                values=clim_values,
                dims=temp_dims,
                units="degC",
                attrs={
                    "long_name": "fitted climatology used to form reconstructed temperature",
                    **({"data_status": "unavailable"} if not np.isfinite(climatology).any() else {}),
                },
            )

    notes = "; ".join(str(n) for n in result.notes) if result.notes else ""
    return OceanDataset(
        variables=variables,
        coords=coords,
        attrs={
            "title": "NEER reconstructed temperature grid",
            "source": "NEER /reconstruct/netcdf",
            "data_mode": result.data_mode,
            "date": str(result.date),
            "cache_hit": int(bool(result.cache_hit)),
            "notes": notes,
        },
        source="neer.reconstruct_grid",
        source_format="netcdf",
    )
