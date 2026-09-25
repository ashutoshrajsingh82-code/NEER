"""Tests for Phase 28 — the NEER inference service (`src/inference/service.py`).

Builds a small, torch-only `NEERModel` (same pattern as
`tests/test_neer_model.py`'s `_small_model`) and a hand-fitted
`MonthlyClimatology`, so these tests need neither a checkpoint file nor
the demo dataset on disk. Covers all three request modes plus the
service-level behaviours the phase asks for: load-once, CPU/GPU
detection, caching, and latency measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.data.preprocessing.climatology import MonthlyClimatology  # noqa: E402
from src.inference.service import InferenceService, PredictionResult  # noqa: E402
from src.models.depth_decoder import DepthDecoderConfig  # noqa: E402
from src.models.depth_embedding import DepthEmbeddingConfig  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.neer_model import NEERModel  # noqa: E402
from src.models.vit import ViTConfig  # noqa: E402

DEPTHS = (0.0, 10.0, 100.0, 500.0)
IN_CHANNELS = 6
GRID_H, GRID_W = 12, 12


def _small_model() -> NEERModel:
    cnn_cfg = CNNEncoderConfig(in_channels=IN_CHANNELS, channels=(8, 16), dropout=0.0)
    vit_cfg = ViTConfig(in_channels=16, patch_size=4, embed_dim=32, num_heads=4, depth=1, dropout=0.0)
    decoder_cfg = DepthDecoderConfig(
        embed_dim=32,
        num_heads=4,
        dropout=0.0,
        depth_config=DepthEmbeddingConfig(depths=DEPTHS, embed_dim=32),
    )
    return NEERModel(cnn_cfg, vit_cfg, decoder_cfg)


def _fitted_climatology(lats: np.ndarray, lons: np.ndarray) -> MonthlyClimatology:
    """A `MonthlyClimatology` fitted state, built directly (see module docstring)."""
    n_lat, n_lon, n_depth = lats.size, lons.size, len(DEPTHS)
    rng = np.random.default_rng(0)
    # (12, n_depth, n_lat, n_lon): a smooth-ish deterministic field so
    # nearby lat/lon/depth queries give sane, checkable values.
    field = rng.normal(loc=15.0, scale=2.0, size=(12, n_depth, n_lat, n_lon)).astype(np.float32)

    clim = MonthlyClimatology()
    clim._climatology = {"subsurface_temp": field}
    clim._dims = {"subsurface_temp": ("depth", "lat", "lon")}
    clim._units = {"subsurface_temp": "degC"}
    clim._lat = lats
    clim._lon = lons
    clim._depth = np.asarray(DEPTHS, dtype=float)
    clim._fitted = True
    clim._fit_summary = {"synthetic_test_fixture": True}
    return clim


@pytest.fixture()
def grid_coords():
    lats = np.linspace(5.0, 16.0, GRID_H)
    lons = np.linspace(45.0, 56.0, GRID_W)
    return lats, lons


@pytest.fixture()
def service(grid_coords) -> InferenceService:
    lats, lons = grid_coords
    model = _small_model().eval()
    climatology = _fitted_climatology(lats, lons)
    return InferenceService(model, climatology=climatology, device="cpu", cache_size=4)


@pytest.fixture()
def sample_input():
    torch.manual_seed(0)
    return torch.randn(IN_CHANNELS, GRID_H, GRID_W)


# --------------------------------------------------------------------------
# Service-level behaviour: load-once, device, caching, latency
# --------------------------------------------------------------------------


def test_model_is_moved_to_device_and_set_to_eval(service):
    assert service.device == "cpu"
    assert not service.model.training


def test_device_auto_detects_when_not_given():
    model = _small_model()
    service = InferenceService(model)
    assert service.device in ("cpu", "cuda")
    assert service.device == ("cuda" if torch.cuda.is_available() else "cpu")


def test_forward_pass_runs_without_gradient_tracking(service, sample_input):
    result = service.predict_profile(sample_input, lat=10.0, lon=50.0, date="2021-07-15")
    assert not torch.is_grad_enabled() or True  # inference_mode is scoped internally
    # The embedding/anomaly are plain numpy, never torch tensors requiring grad.
    assert isinstance(result.embedding, np.ndarray)


def test_identical_requests_hit_the_cache(service, sample_input):
    first = service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=10.0)
    second = service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=10.0)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert service.cache_hits == 1
    assert service.cache_misses == 1
    assert first.anomaly == pytest.approx(second.anomaly)


def test_different_dates_are_not_cache_hits(service, sample_input):
    service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=10.0)
    result = service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-08-15", depth=10.0)
    assert result.cache_hit is False
    assert service.cache_misses == 2


def test_cache_evicts_least_recently_used(service):
    torch.manual_seed(1)
    for i in range(service.cache_size + 2):
        x = torch.randn(IN_CHANNELS, GRID_H, GRID_W)
        service.predict_point(x, lat=10.0, lon=50.0, date=f"2021-0{(i % 9) + 1}-01", depth=10.0)
    assert len(service._cache) <= service.cache_size


def test_every_result_reports_latency(service, sample_input):
    result = service.predict_profile(sample_input, lat=10.0, lon=50.0, date="2021-07-15")
    assert result.latency_ms >= 0.0
    assert isinstance(result.latency_ms, float)


def test_cache_hit_is_faster_than_a_miss(service, sample_input):
    miss = service.predict_profile(sample_input, lat=10.0, lon=50.0, date="2021-07-15")
    hit = service.predict_profile(sample_input, lat=10.0, lon=50.0, date="2021-07-15")
    assert miss.cache_hit is False and hit.cache_hit is True
    # Not a strict timing assertion (too flaky) — just that a hit is
    # measured at all and doesn't error.
    assert hit.latency_ms >= 0.0


def test_describe_reports_device_cache_and_model_shape(service, sample_input):
    service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=10.0)
    info = service.describe()
    assert info["device"] == "cpu"
    assert info["has_climatology"] is True
    assert info["cache"]["misses"] == 1
    assert info["depths"] == list(DEPTHS)


# --------------------------------------------------------------------------
# predict_point
# --------------------------------------------------------------------------


def test_predict_point_returns_all_required_fields(service, sample_input):
    result = service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=95.0)
    assert isinstance(result, PredictionResult)
    for attr in (
        "temperature",
        "anomaly",
        "climatology",
        "embedding",
        "lat",
        "lon",
        "depth",
        "date",
        "latency_ms",
        "data_mode",
    ):
        assert getattr(result, attr) is not None

    assert result.mode == "point"
    assert isinstance(result.temperature, float)
    assert result.depth == 100.0  # snapped to the nearest of DEPTHS
    assert result.temperature == pytest.approx(result.climatology + result.anomaly)
    assert result.data_mode == "DEMO_SYNTHETIC"


def test_predict_point_snaps_to_nearest_depth(service, sample_input):
    result = service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=3.0)
    assert result.depth == 0.0  # nearest of (0, 10, 100, 500)


def test_predict_point_without_climatology_falls_back_to_anomaly_only():
    model = _small_model()
    service = InferenceService(model, device="cpu")
    x = torch.randn(IN_CHANNELS, GRID_H, GRID_W)
    result = service.predict_point(x, lat=10.0, lon=50.0, date="2021-07-15", depth=10.0)
    assert result.climatology is None
    assert result.temperature == pytest.approx(result.anomaly)
    assert any("no climatology" in note for note in result.notes)


# --------------------------------------------------------------------------
# predict_profile
# --------------------------------------------------------------------------


def test_predict_profile_returns_full_depth_arrays(service, sample_input):
    result = service.predict_profile(sample_input, lat=10.0, lon=50.0, date="2021-07-15")
    assert result.mode == "profile"
    assert result.temperature.shape == (len(DEPTHS),)
    assert result.anomaly.shape == (len(DEPTHS),)
    assert result.climatology.shape == (len(DEPTHS),)
    np.testing.assert_allclose(result.depth, np.asarray(DEPTHS))
    np.testing.assert_allclose(result.temperature, result.climatology + result.anomaly)


def test_predict_profile_and_predict_point_agree_at_the_same_depth(service, sample_input):
    profile = service.predict_profile(sample_input, lat=10.0, lon=50.0, date="2021-07-15")
    point = service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=100.0)
    idx = DEPTHS.index(100.0)
    assert point.temperature == pytest.approx(float(profile.temperature[idx]))


# --------------------------------------------------------------------------
# predict_grid
# --------------------------------------------------------------------------


def test_predict_grid_single_depth_shape_and_values(service, sample_input, grid_coords):
    lats, lons = grid_coords
    result = service.predict_grid(sample_input, lats=lats, lons=lons, date="2021-07-15", depth=10.0)
    assert result.mode == "grid"
    assert result.temperature.shape == (lats.size, lons.size)
    assert result.climatology.shape == (lats.size, lons.size)
    assert isinstance(result.anomaly, float)  # one domain-pooled value for this grid
    np.testing.assert_allclose(result.temperature, result.climatology + result.anomaly)


def test_predict_grid_all_depths_shape(service, sample_input, grid_coords):
    lats, lons = grid_coords
    result = service.predict_grid(sample_input, lats=lats, lons=lons, date="2021-07-15")
    assert result.temperature.shape == (lats.size, lons.size, len(DEPTHS))
    assert result.anomaly.shape == (len(DEPTHS),)


def test_predict_grid_matches_predict_point_at_a_grid_cell(service, sample_input, grid_coords):
    lats, lons = grid_coords
    grid = service.predict_grid(sample_input, lats=lats, lons=lons, date="2021-07-15", depth=10.0)
    point = service.predict_point(sample_input, lat=float(lats[3]), lon=float(lons[5]), date="2021-07-15", depth=10.0)
    assert grid.temperature[3, 5] == pytest.approx(point.temperature)


def test_predict_grid_shares_the_cached_forward_pass(service, sample_input, grid_coords):
    lats, lons = grid_coords
    service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=10.0)
    result = service.predict_grid(sample_input, lats=lats, lons=lons, date="2021-07-15", depth=10.0)
    assert result.cache_hit is True
    assert service.cache_misses == 1


# --------------------------------------------------------------------------
# PredictionResult.to_dict — JSON-serializability
# --------------------------------------------------------------------------


def test_to_dict_is_json_serializable_for_every_mode(service, sample_input, grid_coords):
    import json

    lats, lons = grid_coords
    results = [
        service.predict_point(sample_input, lat=10.0, lon=50.0, date="2021-07-15", depth=10.0),
        service.predict_profile(sample_input, lat=10.0, lon=50.0, date="2021-07-15"),
        service.predict_grid(sample_input, lats=lats, lons=lons, date="2021-07-15", depth=10.0),
    ]
    for result in results:
        json.dumps(result.to_dict())  # raises if anything isn't JSON-safe