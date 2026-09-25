"""
Phase 28 — `InferenceService`: a reusable, serving-shaped wrapper around
`NEERModel` (see `src/inference/__init__.py` for the design rationale).

Usage
-----
    >>> service = InferenceService.from_checkpoint(
    ...     "artifacts/checkpoints/neer.pt",
    ...     climatology=MonthlyClimatology.load("data/processed/climatology.npz"),
    ... )
    >>> result = service.predict_point(x, lat=12.0, lon=80.0, date="2021-07-15", depth=100.0)
    >>> result.temperature, result.latency_ms, result.cache_hit
    (24.9, 3.2, False)

`x` is always a single surface-field sample — `(NEER_N_CHANNELS, H, W)`
(or a leading batch-of-1 dimension), in `NEER_CHANNEL_ORDER` — built by
the data-loading/preprocessing layer for one timestep. Building that
tensor from raw observations is out of this phase's scope; this service
starts from it.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing._utils import json_safe, month_of

try:  # pragma: no cover - environment dependent
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

#: Default variable name looked up in a `MonthlyClimatology` for the
#: subsurface temperature this service reconstructs. Override per
#: instance with `InferenceService(..., climatology_variable=...)` if a
#: caller fitted the climatology under a different variable name.
DEFAULT_CLIMATOLOGY_VARIABLE = "subsurface_temp"

DEMO_DATA_MODE = "DEMO_SYNTHETIC"
REAL_DATA_MODE = "REAL"


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="the NEER inference service (src.inference.service)",
            install_hint="pip install torch",
        )


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """What every `InferenceService.predict_*` call returns.

    Shapes differ by `mode`:

    * ``"point"``   — `temperature`/`anomaly`/`climatology` are floats,
      `depth` is a float, `lat`/`lon` are floats.
    * ``"profile"`` — `temperature`/`anomaly`/`climatology` are
      `(num_depths,)` arrays, `depth` is the `(num_depths,)` depth axis,
      `lat`/`lon` are floats.
    * ``"grid"``    — `lat`/`lon` are the `(n_lat,)`/`(n_lon,)` coordinate
      axes; `temperature` (and `climatology`, when a climatology was
      fitted) is `(n_lat, n_lon)` for one requested depth or
      `(n_lat, n_lon, num_depths)` for every depth; `anomaly` is the one
      domain-pooled value (or `(num_depths,)` vector) this grid was
      built from — see `src/inference/__init__.py` for why the anomaly
      does not itself vary across the grid.

    `embedding` is always the `(embed_dim,)` pooled representation the
    prediction came from — the same value regardless of `mode`, since
    all three modes read it off the one cached forward pass.
    """

    mode: str
    temperature: Union[float, np.ndarray]
    anomaly: Union[float, np.ndarray]
    climatology: Optional[Union[float, np.ndarray]]
    embedding: np.ndarray
    lat: Union[float, np.ndarray]
    lon: Union[float, np.ndarray]
    depth: Union[float, np.ndarray]
    date: str
    latency_ms: float
    data_mode: str
    cache_hit: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (NumPy arrays/scalars converted to plain Python)."""
        return {
            "mode": self.mode,
            "temperature": json_safe(self.temperature),
            "anomaly": json_safe(self.anomaly),
            "climatology": json_safe(self.climatology),
            "embedding": json_safe(self.embedding),
            "lat": json_safe(self.lat),
            "lon": json_safe(self.lon),
            "depth": json_safe(self.depth),
            "date": self.date,
            "latency_ms": round(float(self.latency_ms), 4),
            "data_mode": self.data_mode,
            "cache_hit": self.cache_hit,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------


class InferenceService:
    """Loads a `NEERModel` once and serves point/profile/grid predictions.

    Parameters
    ----------
    model:
        An already-constructed `NEERModel` (any state — this class puts
        it in `eval()` mode and moves it to the chosen device).
    climatology:
        Optional fitted `MonthlyClimatology` (Phase 11). Without one,
        every prediction's `temperature` equals its `anomaly` and a note
        is attached explaining why.
    climatology_variable:
        The variable name to query in `climatology`. Defaults to
        `DEFAULT_CLIMATOLOGY_VARIABLE`.
    device:
        `"cpu"`, `"cuda"`, or `None` to auto-detect (`"cuda"` if
        available, else `"cpu"`) — done once here, not per call.
    normalizer_center, normalizer_scale:
        Optional per-depth arrays (length `model.num_depths`). When
        given, the model's raw output is treated as z-scored and
        de-normalized (`anomaly * scale + center`) before anything else
        sees it — the same linear map `predictions_from_checkpoint`
        applies (`src/argo_validation/predictions.py`). Omit both to
        treat the model's raw output as already being a physical-units
        anomaly.
    cache_size:
        Maximum number of distinct (input tensor, date) forward passes
        kept in memory. Least-recently-used entries are evicted first.
    data_mode:
        Provenance label attached to every result (`DEMO_SYNTHETIC` /
        `REAL` / a caller-supplied string) — see `data_mode` throughout
        `src/data/loaders` and `src/argo_validation`; this service does
        not infer it, it just carries whatever the caller says the
        input came from.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        *,
        climatology: Optional[Any] = None,
        climatology_variable: str = DEFAULT_CLIMATOLOGY_VARIABLE,
        device: Optional[str] = None,
        normalizer_center: Optional[Sequence[float]] = None,
        normalizer_scale: Optional[Sequence[float]] = None,
        cache_size: int = 128,
        data_mode: str = DEMO_DATA_MODE,
    ) -> None:
        _require_torch()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.climatology = climatology
        self.climatology_variable = climatology_variable
        self.data_mode = data_mode

        if (normalizer_center is None) != (normalizer_scale is None):
            raise ValueError("normalizer_center and normalizer_scale must be given together")
        self._center = (
            None if normalizer_center is None else np.asarray(normalizer_center, dtype=float)
        )
        self._scale = (
            None if normalizer_scale is None else np.asarray(normalizer_scale, dtype=float)
        )

        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

        self.depths: Tuple[float, ...] = tuple(float(d) for d in self.model.depths)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        environment: Optional[str] = None,
        metadata_path: Optional[str] = None,
        target_variable: Optional[str] = None,
        **kwargs: Any,
    ) -> "InferenceService":
        """Load model weights (mirrors `predictions_from_checkpoint`'s pattern).

        `environment` picks the config used to build the `NEERModel`
        architecture before loading weights into it; when omitted it is
        read from the checkpoint itself (`checkpoint["args"]["environment"]`,
        falling back to `"demo"`), exactly like
        `src.argo_validation.predictions.predictions_from_checkpoint`.

        `metadata_path` + `target_variable`, when both given, load
        per-depth z-score statistics from a
        `neer_preprocessing_metadata.json` so raw model output is
        de-normalized into physical units (see `normalizer_center`/
        `normalizer_scale` on `__init__`).
        """
        _require_torch()
        from src.models.neer_model import NEERModel
        from src.utils.config import load_config

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        resolved_environment = environment or checkpoint.get("args", {}).get(
            "environment", "demo"
        )
        model = NEERModel.from_config(load_config(resolved_environment))
        model.load_state_dict(checkpoint["model_state_dict"])

        if metadata_path is not None and target_variable is not None:
            from src.argo_validation.predictions import normalizer_stats_from_metadata

            center, scale = normalizer_stats_from_metadata(
                metadata_path, target_variable, model.num_depths
            )
            kwargs.setdefault("normalizer_center", center)
            kwargs.setdefault("normalizer_scale", scale)

        return cls(model, **kwargs)

    # -- introspection ------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Snapshot of this service's state — device, cache, model shape."""
        return {
            "device": self.device,
            "cuda_available": bool(torch.cuda.is_available()),
            "embed_dim": int(self.model.embed_dim),
            "depths": list(self.depths),
            "has_climatology": self.climatology is not None,
            "de_normalizes_output": self._center is not None,
            "data_mode": self.data_mode,
            "cache": {
                "size": len(self._cache),
                "max_size": self.cache_size,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            },
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    # -- the one shared forward pass ---------------------------------------

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if _TORCH_AVAILABLE and torch.is_tensor(x):
            return x.detach().to("cpu").numpy().astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32)

    def _cache_key(self, x: np.ndarray, date: str) -> str:
        digest = hashlib.sha1(x.tobytes())
        digest.update(str(date).encode("utf-8"))
        digest.update(str(self.device).encode("utf-8"))
        return digest.hexdigest()

    def _forward(self, x: Any, date: str) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Run (or fetch from cache) the encoder+decoder for one sample.

        Returns `(anomaly, embedding, latency_ms, cache_hit)`. `anomaly`
        is `(num_depths,)`, already de-normalized when this service was
        built with normalizer stats. `latency_ms` measures the whole
        call, including the cache lookup, so a hit's latency is honestly
        near-zero rather than the original compute time.
        """
        start = time.perf_counter()

        x_np = self._to_numpy(x)
        if x_np.ndim == 3:
            x_np = x_np[None, ...]
        if x_np.ndim != 4 or x_np.shape[0] != 1:
            raise ValueError(
                f"InferenceService expects one sample per call — (C, H, W) or "
                f"(1, C, H, W) — got shape {x_np.shape}"
            )

        key = self._cache_key(x_np, date)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            anomaly, embedding = cached
            latency_ms = (time.perf_counter() - start) * 1000.0
            return anomaly.copy(), embedding.copy(), latency_ms, True

        self.cache_misses += 1
        tensor = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            embedding_t = self.model.encoder.get_embedding(tensor)
            anomaly_t = self.model.decoder(embedding_t)
        anomaly = anomaly_t.squeeze(0).to("cpu").numpy().astype(float)
        embedding = embedding_t.squeeze(0).to("cpu").numpy().astype(float)

        if self._center is not None:
            anomaly = anomaly * self._scale + self._center

        self._cache[key] = (anomaly.copy(), embedding.copy())
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        latency_ms = (time.perf_counter() - start) * 1000.0
        return anomaly, embedding, latency_ms, False

    # -- shared helpers -----------------------------------------------------

    def _nearest_depth_index(self, depth: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.depths) - float(depth))))

    @staticmethod
    def _month_of(date: Any) -> int:
        return int(month_of(np.array([date], dtype="datetime64[ns]"))[0])

    def _climatology_notes(self) -> List[str]:
        if self.climatology is None:
            return [
                "no climatology supplied to this InferenceService; "
                "'temperature' equals the raw model anomaly, not an absolute value"
            ]
        return []

    def _climatology_value(self, lat: float, lon: float, month: int, depth: float) -> Optional[float]:
        if self.climatology is None:
            return None
        return self.climatology.at(self.climatology_variable, month, lat, lon, depth=depth)

    def _climatology_profile(self, lat: float, lon: float, month: int) -> Optional[np.ndarray]:
        if self.climatology is None:
            return None
        return self.climatology.depth_profile(self.climatology_variable, month, lat, lon)

    # -- public API -----------------------------------------------------

    def predict_point(
        self,
        x: Any,
        *,
        lat: float,
        lon: float,
        date: Any,
        depth: float,
        month: Optional[int] = None,
    ) -> PredictionResult:
        """Temperature at one location, one date, one depth.

        `depth` is snapped to the nearest of the model's target depths
        (`self.depths`) — the model predicts a fixed set of depth
        levels, it does not interpolate between them.
        """
        anomaly, embedding, latency_ms, cache_hit = self._forward(x, date=str(date))
        month = month or self._month_of(date)
        depth_idx = self._nearest_depth_index(depth)
        depth_value = self.depths[depth_idx]
        anomaly_value = float(anomaly[depth_idx])
        climatology_value = self._climatology_value(lat, lon, month, depth_value)
        temperature = anomaly_value if climatology_value is None else climatology_value + anomaly_value

        return PredictionResult(
            mode="point",
            temperature=temperature,
            anomaly=anomaly_value,
            climatology=climatology_value,
            embedding=embedding,
            lat=float(lat),
            lon=float(lon),
            depth=depth_value,
            date=str(date),
            latency_ms=latency_ms,
            data_mode=self.data_mode,
            cache_hit=cache_hit,
            notes=self._climatology_notes(),
        )

    def predict_profile(
        self,
        x: Any,
        *,
        lat: float,
        lon: float,
        date: Any,
        month: Optional[int] = None,
    ) -> PredictionResult:
        """Full depth-temperature profile at one location, one date."""
        anomaly, embedding, latency_ms, cache_hit = self._forward(x, date=str(date))
        month = month or self._month_of(date)
        climatology_profile = self._climatology_profile(lat, lon, month)
        temperature = (
            anomaly.copy() if climatology_profile is None else climatology_profile + anomaly
        )

        return PredictionResult(
            mode="profile",
            temperature=temperature,
            anomaly=anomaly,
            climatology=climatology_profile,
            embedding=embedding,
            lat=float(lat),
            lon=float(lon),
            depth=np.asarray(self.depths, dtype=float),
            date=str(date),
            latency_ms=latency_ms,
            data_mode=self.data_mode,
            cache_hit=cache_hit,
            notes=self._climatology_notes(),
        )

    def predict_grid(
        self,
        x: Any,
        *,
        lats: Sequence[float],
        lons: Sequence[float],
        date: Any,
        depth: Optional[float] = None,
        month: Optional[int] = None,
    ) -> PredictionResult:
        """Temperature over a lat/lon grid, one date, one depth (or every depth).

        The forward pass runs once (`x` is one domain-pooled sample, as
        everywhere else in this service); spatial variation across the
        returned grid comes entirely from the per-cell climatology, not
        from a per-cell model output — see `src/inference/__init__.py`.
        With `depth=None`, every model depth is returned and
        `temperature`/`climatology` are `(n_lat, n_lon, num_depths)`;
        with a specific `depth`, they are `(n_lat, n_lon)`.
        """
        anomaly, embedding, latency_ms, cache_hit = self._forward(x, date=str(date))
        month = month or self._month_of(date)
        lat_arr = np.asarray(lats, dtype=float)
        lon_arr = np.asarray(lons, dtype=float)
        notes = self._climatology_notes()
        notes = notes + [
            "the anomaly is one domain-pooled value per timestep; only the "
            "climatology term varies across this grid (see src/inference/__init__.py)"
        ]

        if depth is not None:
            depth_idx = self._nearest_depth_index(depth)
            depth_value = self.depths[depth_idx]
            anomaly_scalar = float(anomaly[depth_idx])
            if self.climatology is None:
                climatology_grid = None
                temperature_grid = np.full((lat_arr.size, lon_arr.size), anomaly_scalar)
            else:
                climatology_grid = np.array(
                    [
                        [
                            self.climatology.at(self.climatology_variable, month, la, lo, depth=depth_value)
                            for lo in lon_arr
                        ]
                        for la in lat_arr
                    ]
                )
                temperature_grid = climatology_grid + anomaly_scalar

            return PredictionResult(
                mode="grid",
                temperature=temperature_grid,
                anomaly=anomaly_scalar,
                climatology=climatology_grid,
                embedding=embedding,
                lat=lat_arr,
                lon=lon_arr,
                depth=depth_value,
                date=str(date),
                latency_ms=latency_ms,
                data_mode=self.data_mode,
                cache_hit=cache_hit,
                notes=notes,
            )

        # every depth: (n_lat, n_lon, num_depths)
        if self.climatology is None:
            climatology_grid = None
            temperature_grid = np.broadcast_to(
                anomaly[None, None, :], (lat_arr.size, lon_arr.size, anomaly.size)
            ).copy()
        else:
            climatology_grid = np.array(
                [
                    [self.climatology.depth_profile(self.climatology_variable, month, la, lo) for lo in lon_arr]
                    for la in lat_arr
                ]
            )
            temperature_grid = climatology_grid + anomaly[None, None, :]

        return PredictionResult(
            mode="grid",
            temperature=temperature_grid,
            anomaly=anomaly.copy(),
            climatology=climatology_grid,
            embedding=embedding,
            lat=lat_arr,
            lon=lon_arr,
            depth=np.asarray(self.depths, dtype=float),
            date=str(date),
            latency_ms=latency_ms,
            data_mode=self.data_mode,
            cache_hit=cache_hit,
            notes=notes,
        )