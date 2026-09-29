#!/usr/bin/env python3
"""Verify NEER's configured demo-to-export pipeline using production code paths.

Run from the repository root with ``python scripts/smoke_test.py``. The test
is read-only apart from transient NetCDF output in a temporary directory.
It never creates demo data, trains a model, or substitutes synthetic results.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, Callable[[], str]]] = []


def check(name: str):
    def register(function: Callable[[], str]):
        CHECKS.append((name, function))
        return function
    return register


@check("Demo data")
def check_demo_data() -> str:
    from src.data.loaders import DEFAULT_DEMO_DATASET, load_demo_dataset

    if not DEFAULT_DEMO_DATASET.is_file() or DEFAULT_DEMO_DATASET.stat().st_size == 0:
        raise RuntimeError("demo dataset is missing or empty; generate it with scripts/create_demo_data.py")
    dataset = load_demo_dataset()
    if dataset.n_time < 1 or not dataset.lat.size or not dataset.lon.size:
        raise RuntimeError("demo dataset has empty time or spatial coordinates")
    if not {"sst", "sss", "sla"}.issubset(dataset.variables):
        raise RuntimeError("demo dataset is missing expected surface variables")
    if dataset.depth is None or dataset.depth.size != 15:
        raise RuntimeError("demo dataset does not contain the configured 15 depth levels")
    for coord in (dataset.time, dataset.lat, dataset.lon, dataset.depth):
        if coord is None or coord.size == 0:
            raise RuntimeError("demo dataset contains an empty coordinate")
    return f"{dataset.n_time} dates, grid {dataset.grid_shape}, mode {dataset.data_mode}"


@check("Model checkpoint")
def check_checkpoint() -> str:
    from backend.app.services.repository import _find_checkpoint
    from src.utils.config import load_config

    config = load_config("demo")
    checkpoint_dir = Path(config.raw["paths"]["artifacts_checkpoints"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    checkpoint = _find_checkpoint(checkpoint_dir)
    if checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("configured model checkpoint is missing or empty")
    return f"{checkpoint.name}, {checkpoint.stat().st_size:,} bytes"


@check("Model loading")
def check_model_loading() -> str:
    from backend.app.services.repository import NEERRepository

    repository = NEERRepository(environment="demo")
    if repository.service is None:
        raise RuntimeError(repository.health().get("components", {}).get("model", {}).get("detail", "model did not load"))
    model = repository.service.model
    if model.training:
        raise RuntimeError("serving model is in training mode")
    if not repository.is_data_loaded:
        raise RuntimeError(repository.health().get("components", {}).get("data", {}).get("detail", "demo tensor bundle unavailable"))
    if not repository.bundle.is_synthetic:
        raise RuntimeError("configured demo tensor bundle lacks its synthetic provenance marker")
    return f"eval mode, device {repository.service.device}, embedding dimension {model.embed_dim}"


@check("Forward pass and depths")
def check_forward_and_depths() -> str:
    from backend.app.services.repository import NEERRepository

    repository = NEERRepository(environment="demo")
    if repository.service is None or repository.bundle is None:
        raise RuntimeError("model or tensor bundle unavailable")
    bundle, service = repository.bundle, repository.service
    config_depths = np.asarray(repository.config.depths, dtype=float)
    model_depths = np.asarray(service.depths, dtype=float)
    data_depths = np.asarray(bundle.depth, dtype=float)
    if not (len(config_depths) == 15 and np.array_equal(model_depths, config_depths) and np.array_equal(data_depths, config_depths)):
        raise RuntimeError(f"configured, model and dataset depths differ: {config_depths.tolist()}, {model_depths.tolist()}, {data_depths.tolist()}")
    if bundle.inputs.ndim != 4 or bundle.inputs.shape[1] != len(bundle.channel_names):
        raise RuntimeError(f"unexpected preprocessed input shape {bundle.inputs.shape}")
    if len(bundle.time) == 0:
        raise RuntimeError("demo tensor bundle contains no time steps")
    timestep = 0
    anomaly, embedding, _, _ = service._forward(bundle.inputs[timestep], date=str(bundle.time[timestep]))
    if anomaly.shape != (15,) or not np.isfinite(anomaly).all():
        raise RuntimeError(f"forward pass returned invalid depth output shape/values: {anomaly.shape}")
    if embedding.size == 0 or embedding.shape[-1] != int(service.model.embed_dim) or not np.isfinite(embedding).all():
        raise RuntimeError(f"embedding has invalid shape/values: {embedding.shape}")
    return f"15 levels {model_depths.astype(int).tolist()} m; embedding {embedding.shape[-1]}D"


@check("Inference")
def check_inference() -> str:
    from backend.app.services.repository import NEERRepository

    repository = NEERRepository(environment="demo")
    if repository.bundle is None or repository.service is None:
        raise RuntimeError("demo data or model unavailable")
    bundle = repository.bundle
    date = repository.list_dates()["dates"][-1]
    depth = 100.0
    if depth not in repository.service.depths:
        raise RuntimeError("100 m is not among the loaded model depth coordinates")
    if bundle.ocean_mask is None:
        raise RuntimeError("demo tensor bundle does not provide an ocean mask")
    # Choose the valid ocean coordinate nearest the domain center from the
    # actual loaded mask, not an assumed/rounded location.
    rows, cols = np.where(np.asarray(bundle.ocean_mask, dtype=bool))
    if not len(rows):
        raise RuntimeError("demo tensor bundle has no ocean grid coordinates")
    center_lat = (repository.config.domain.lat_min + repository.config.domain.lat_max) / 2
    center_lon = (repository.config.domain.lon_min + repository.config.domain.lon_max) / 2
    index = int(np.argmin((bundle.lat[rows] - center_lat) ** 2 + (bundle.lon[cols] - center_lon) ** 2))
    lat, lon = float(bundle.lat[rows[index]]), float(bundle.lon[cols[index]])
    result = repository.reconstruct_point(lat=lat, lon=lon, date=date, depth=depth)
    if result.date != date or not np.isclose(result.depth, depth) or not np.isfinite([result.lat, result.lon]).all():
        raise RuntimeError("reconstruction result does not match requested coordinates/date/depth")
    if result.embedding is None or np.asarray(result.embedding).size != repository.service.model.embed_dim:
        raise RuntimeError("inference response has no valid model embedding")
    if result.temperature is not None and not np.isfinite(result.temperature):
        raise RuntimeError("inference returned a non-finite temperature")
    return f"{date}, {lat:.3f} N {lon:.3f} E, {depth:g} m; temperature {result.temperature}"


@check("FastAPI endpoints")
def check_api() -> str:
    from fastapi.testclient import TestClient
    from backend.app.main import app

    old_environment = os.environ.get("NEER_ENVIRONMENT")
    os.environ["NEER_ENVIRONMENT"] = "demo"
    try:
        with TestClient(app) as client:
            health = client.get("/health")
            dates = client.get("/dates")
            model = client.get("/model/info")
            if health.status_code != 200 or health.json().get("status") != "ok":
                raise RuntimeError(f"GET /health returned HTTP {health.status_code}: {health.text[:240]}")
            if dates.status_code != 200 or not dates.json().get("dates"):
                raise RuntimeError(f"GET /dates returned HTTP {dates.status_code}: {dates.text[:240]}")
            if model.status_code != 200 or not model.json().get("architecture", {}).get("depths"):
                raise RuntimeError(f"GET /model/info returned HTTP {model.status_code}: {model.text[:240]}")
            demo = client.get("/demo/context")
            if demo.status_code != 200:
                raise RuntimeError(f"GET /demo/context returned HTTP {demo.status_code}: {demo.text[:240]}")
            context = demo.json()
            result = client.get("/reconstruct", params={"lat": context["latitude"], "lon": context["longitude"], "date": context["date"], "depth": 100})
            if result.status_code != 200:
                raise RuntimeError(f"GET /reconstruct returned HTTP {result.status_code}: {result.text[:240]}")
            payload = result.json()
            required = {"date", "lat", "lon", "depth", "temperature", "anomaly", "embedding_dim", "data_mode"}
            if not required.issubset(payload) or payload["date"] != context["date"] or payload["depth"] != 100:
                raise RuntimeError("GET /reconstruct returned an incomplete or mismatched scientific response")
            return f"/health, /dates, /model/info, /demo/context, /reconstruct returned valid responses"
    finally:
        if old_environment is None:
            os.environ.pop("NEER_ENVIRONMENT", None)
        else:
            os.environ["NEER_ENVIRONMENT"] = old_environment


@check("NetCDF export round trip")
def check_netcdf() -> str:
    import xarray as xr
    from fastapi.testclient import TestClient
    from backend.app.main import app

    old_environment = os.environ.get("NEER_ENVIRONMENT")
    os.environ["NEER_ENVIRONMENT"] = "demo"
    try:
        with TestClient(app) as client, tempfile.TemporaryDirectory(prefix="neer-smoke-") as temp_dir:
            context_response = client.get("/demo/context")
            if context_response.status_code != 200:
                raise RuntimeError(f"GET /demo/context returned HTTP {context_response.status_code}")
            context = context_response.json()
            half = 0.5
            params = {"lat_min": max(5.0, context["latitude"] - half), "lat_max": min(30.0, context["latitude"] + half), "lon_min": max(45.0, context["longitude"] - half), "lon_max": min(105.0, context["longitude"] + half), "date": context["date"], "depth": 100}
            response = client.get("/reconstruct/netcdf", params=params)
            if response.status_code != 200:
                raise RuntimeError(f"GET /reconstruct/netcdf returned HTTP {response.status_code}: {response.text[:240]}")
            output = Path(temp_dir) / "smoke-export.nc"
            output.write_bytes(response.content)
            if output.stat().st_size <= 0:
                raise RuntimeError("NetCDF export is empty")
            # Client has received and closed the response; perform an explicit
            # independent open, close, reopen, and validate the second read.
            with xr.open_dataset(output) as first:
                if not {"temperature", "anomaly"}.issubset(first.data_vars):
                    raise RuntimeError("NetCDF is missing temperature or anomaly")
                first.load()
            with xr.open_dataset(output) as reopened:
                required_coords = {"time", "depth", "lat", "lon"}
                if not required_coords.issubset(reopened.coords):
                    raise RuntimeError(f"NetCDF coordinates missing: {sorted(required_coords - set(reopened.coords))}")
                temperature = reopened["temperature"]
                if temperature.dims != ("time", "depth", "lat", "lon"):
                    raise RuntimeError(f"unexpected temperature dimensions: {temperature.dims}")
                if temperature.shape != (1, 1, reopened.sizes["lat"], reopened.sizes["lon"]):
                    raise RuntimeError(f"temperature shape does not match coordinates: {temperature.shape}")
                if not np.isfinite(np.asarray(reopened["lat"].values)).all() or not np.isfinite(np.asarray(reopened["lon"].values)).all():
                    raise RuntimeError("NetCDF coordinates contain invalid values")
                values = np.asarray(temperature.values)
                if not (np.isfinite(values) | np.isnan(values)).all() or not np.isfinite(values).any():
                    raise RuntimeError("NetCDF temperature has invalid or entirely missing data")
                if temperature.attrs.get("units") != "degC" or not reopened.attrs.get("title") or not reopened.attrs.get("source"):
                    raise RuntimeError("NetCDF scientific units or provenance metadata are missing")
                if not np.array_equal(reopened["depth"].values, np.asarray([100.0])):
                    raise RuntimeError("NetCDF depth coordinate does not match the requested 100 m slice")
            return f"{output.stat().st_size:,} bytes; reopened dimensions and provenance valid"
    finally:
        if old_environment is None:
            os.environ.pop("NEER_ENVIRONMENT", None)
        else:
            os.environ["NEER_ENVIRONMENT"] = old_environment


@check("Frontend API configuration")
def check_frontend_config() -> str:
    api_source = (ROOT / "frontend" / "lib" / "api.js").read_text(encoding="utf-8")
    example = ROOT / "frontend" / ".env.example"
    if "NEXT_PUBLIC_API_URL" not in api_source:
        raise RuntimeError("frontend/lib/api.js does not read NEXT_PUBLIC_API_URL")
    if not example.is_file() or "NEXT_PUBLIC_API_URL=" not in example.read_text(encoding="utf-8"):
        raise RuntimeError("frontend/.env.example does not document the API URL setting")
    return "NEXT_PUBLIC_API_URL is consumed by the API client and documented in .env.example"


def main() -> int:
    print("=" * 64)
    print("NEER SMOKE TEST")
    print("=" * 64)
    failures = []
    for number, (name, function) in enumerate(CHECKS, start=1):
        try:
            detail = function()
            print(f"[{number}/{len(CHECKS)}] {name:<30} PASS - {detail}")
        except Exception as exc:  # Critical check failure must stay a failure.
            failures.append((name, exc))
            print(f"[{number}/{len(CHECKS)}] {name:<30} FAIL - {exc}")
    print("=" * 64)
    if failures:
        print(f"NEER SMOKE TEST: FAIL ({len(failures)} failed)")
        return 1
    print("NEER SMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
