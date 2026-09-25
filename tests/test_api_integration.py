"""Phase 30 — API Integration Tests.

End-to-end integration tests for every FastAPI endpoint:
- Live HTTP server execution over real TCP sockets with `httpx`
- Valid request verification across all 11 endpoints + root
- Invalid latitude handling (422 out-of-range, 400 out-of-domain, 400 inverted bounds, 422 non-numeric)
- Invalid longitude handling (422 out-of-range, 400 out-of-domain, 400 inverted bounds, 422 non-numeric)
- Invalid date handling (422 malformed format, 404 date_not_found)
- Invalid depth handling (422 negative depth, 422 non-numeric depth)
- Missing data handling (503 data_unavailable) tested across all data-dependent routes
- Model unavailable handling (503 model_unavailable) tested across all model-dependent routes
- Successful reconstruction (point, profile, grid, NetCDF round-trip with physical consistency checks)
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")
uvicorn = pytest.importorskip("uvicorn", reason="uvicorn is required for live server testing")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.main import app, neer_api_error_handler
from backend.app.routers import (
    data_quality,
    dates,
    embedding,
    evaluation,
    explainability,
    health,
    metrics,
    model_info,
    profile,
    reconstruct,
    reconstruct_netcdf,
)
from backend.app.services.repository import NEERRepository
from src.data.loaders import load_netcdf
from tests._backend_fixtures import (
    build_repository,
    client,
    client_for,
    client_no_data,
    client_no_model,
    repository,
    repository_no_data,
    repository_no_model,
)

# Reference parameters valid in the demo repository
VALID_DATE = "2020-06-01"
VALID_LAT = 15.0
VALID_LON = 70.0
VALID_DEPTH = 50.0

VALID_GRID = {
    "lat_min": 12.0,
    "lat_max": 18.0,
    "lon_min": 65.0,
    "lon_max": 75.0,
    "date": VALID_DATE,
}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _UvicornServer(uvicorn.Server):
    """Custom Server avoiding signal-handler registration in background threads."""

    def install_signal_handlers(self) -> None:
        pass


def _create_app_with_repo(repo: NEERRepository) -> FastAPI:
    """Build a standalone FastAPI test app bound to a specific NEERRepository."""
    test_app = FastAPI(title="NEER API Test", version="0.30.0")
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    test_app.add_exception_handler(NeerApiError, neer_api_error_handler)
    test_app.dependency_overrides[get_repository] = lambda: repo
    test_app.include_router(health.router)
    test_app.include_router(model_info.router)
    test_app.include_router(dates.router)
    test_app.include_router(reconstruct.router)
    test_app.include_router(profile.router)
    test_app.include_router(embedding.router)
    test_app.include_router(metrics.router)
    test_app.include_router(evaluation.router)
    test_app.include_router(explainability.router)
    test_app.include_router(data_quality.router)
    test_app.include_router(reconstruct_netcdf.router)

    @test_app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"project": "NEER", "problem_id": "SIH26066", "status": "ok"}

    return test_app


def _run_server_thread(target_app: FastAPI) -> tuple[str, uvicorn.Server, threading.Thread]:
    port = _find_free_port()
    config = uvicorn.Config(target_app, host="127.0.0.1", port=port, log_level="error")
    server = _UvicornServer(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10.0
    started = False
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.05)

    if not started:
        server.should_exit = True
        raise RuntimeError("Live uvicorn server failed to start within timeout.")

    return base_url, server, thread


@pytest.fixture(scope="module")
def live_server_url() -> Iterator[str]:
    """Start real uvicorn HTTP server in a daemon thread and yield base URL."""
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = _UvicornServer(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10.0
    started = False
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.1)

    if not started:
        server.should_exit = True
        raise RuntimeError("Live uvicorn server failed to start within timeout.")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)


@pytest.fixture(scope="module")
def http_client(live_server_url: str) -> Iterator[httpx.Client]:
    """Real HTTP client sending network requests over 127.0.0.1."""
    with httpx.Client(base_url=live_server_url, timeout=15.0) as client:
        yield client


@pytest.fixture(scope="module")
def live_server_no_data_url(tmp_path_factory) -> Iterator[str]:
    """Live HTTP server running an app with model present but no data bundle."""
    tmp = tmp_path_factory.mktemp("no_data_server")
    repo = build_repository(tmp, with_data=False, with_model=True)
    target_app = _create_app_with_repo(repo)
    base_url, server, thread = _run_server_thread(target_app)
    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)


@pytest.fixture(scope="module")
def http_client_no_data(live_server_no_data_url: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=live_server_no_data_url, timeout=10.0) as c:
        yield c


@pytest.fixture(scope="module")
def live_server_no_model_url(tmp_path_factory) -> Iterator[str]:
    """Live HTTP server running an app with data present but no model checkpoint."""
    tmp = tmp_path_factory.mktemp("no_model_server")
    repo = build_repository(tmp, with_data=True, with_model=False, with_targets=True)
    target_app = _create_app_with_repo(repo)
    base_url, server, thread = _run_server_thread(target_app)
    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)


@pytest.fixture(scope="module")
def http_client_no_model(live_server_no_model_url: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=live_server_no_model_url, timeout=10.0) as c:
        yield c


@pytest.fixture()
def client_no_data_with_model(tmp_path) -> Iterator[TestClient]:
    repo = build_repository(tmp_path, with_data=False, with_model=True)
    test_client = client_for(repo)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


@pytest.fixture()
def client_no_model_with_targets(tmp_path) -> Iterator[TestClient]:
    repo = build_repository(tmp_path, with_data=True, with_model=False, with_targets=True)
    test_client = client_for(repo)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


# ============================================================================
# 1. Live HTTP Requests — Valid Request on Every Endpoint
# ============================================================================


def test_live_http_root(http_client: httpx.Client):
    resp = http_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["project"] == "NEER"
    assert data["status"] == "ok"


def test_live_http_health(http_client: httpx.Client):
    resp = http_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "components" in data
    assert "api" in data["components"]
    assert "data" in data["components"]
    assert "model" in data["components"]


def test_live_http_model_info(http_client: httpx.Client):
    resp = http_client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "architecture" in data
    assert data["architecture"]["in_channels"] == 11
    assert data["architecture"]["embed_dim"] == 256
    assert "checkpoint" in data


def test_live_http_dates(http_client: httpx.Client):
    resp = http_client.get("/dates")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0
    assert VALID_DATE in data["dates"]
    assert "min_date" in data
    assert "max_date" in data


def test_live_http_reconstruct_point(http_client: httpx.Client):
    resp = http_client.get(
        "/reconstruct",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE, "depth": VALID_DEPTH},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["lat"] == VALID_LAT
    assert data["lon"] == VALID_LON
    assert data["date"] == VALID_DATE
    assert data["depth"] == VALID_DEPTH
    assert isinstance(data["temperature"], float)
    assert isinstance(data["anomaly"], float)
    assert data["embedding_dim"] == 256


def test_live_http_profile(http_client: httpx.Client):
    resp = http_client.get(
        "/profile",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["lat"] == VALID_LAT
    assert data["lon"] == VALID_LON
    assert data["date"] == VALID_DATE
    assert isinstance(data["depths"], list) and len(data["depths"]) > 0
    assert isinstance(data["temperature"], list)
    assert len(data["temperature"]) == len(data["depths"])


def test_live_http_reconstruct_grid(http_client: httpx.Client):
    resp = http_client.get("/reconstruct/grid", params=VALID_GRID)
    assert resp.status_code == 200
    data = resp.json()
    assert data["date"] == VALID_DATE
    assert isinstance(data["temperature"], list)
    assert len(data["lat"]) > 0
    assert len(data["lon"]) > 0


def test_live_http_reconstruct_netcdf(http_client: httpx.Client, tmp_path: Path):
    resp = http_client.get("/reconstruct/netcdf", params=VALID_GRID)
    assert resp.status_code == 200
    assert "netcdf" in resp.headers.get("content-type", "").lower() or resp.content[:4] in (
        b"CDF\x01",
        b"CDF\x02",
        b"\x89HDF",
    )
    out_file = tmp_path / "live_reconstruct.nc"
    out_file.write_bytes(resp.content)
    dataset = load_netcdf(out_file, check_domain=False)
    assert "temperature" in dataset.variables
    assert dataset.variables["temperature"].units == "degC"


def test_live_http_embedding(http_client: httpx.Client):
    resp = http_client.get("/embedding", params={"date": VALID_DATE})
    assert resp.status_code == 200
    data = resp.json()
    assert data["date"] == VALID_DATE
    assert data["dim"] == 256
    assert len(data["embedding"]) == 256


def test_live_http_metrics(http_client: httpx.Client):
    resp = http_client.get("/metrics", params={"split": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["split"] == "test"
    assert "metrics" in data
    assert "overall" in data["metrics"]


def test_live_http_evaluation_argo(http_client: httpx.Client):
    resp = http_client.get("/evaluation/argo", params={"demo": "true"})
    assert resp.status_code == 200
    data = resp.json()
    assert "neer_predictions" in data
    assert "argo" in data


def test_live_http_explainability(http_client: httpx.Client):
    resp = http_client.get("/explainability", params={"date": VALID_DATE})
    assert resp.status_code == 200
    data = resp.json()
    assert data["date"] == VALID_DATE
    assert data["method"] == "gradient_x_input"
    assert len(data["channels"]) == 11
    assert pytest.approx(sum(c["importance"] for c in data["channels"]), abs=1e-5) == 1.0


def test_live_http_data_quality(http_client: httpx.Client):
    resp = http_client.get("/data/quality")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "spatial_coverage" in data
    assert "channels" in data


# ============================================================================
# 2. Invalid Latitude Handling
# ============================================================================


@pytest.mark.parametrize("route", ["/reconstruct", "/profile"])
def test_invalid_latitude_out_of_bounds_is_422(http_client: httpx.Client, route: str):
    params = {"lat": 999.0, "lon": VALID_LON, "date": VALID_DATE}
    if route == "/reconstruct":
        params["depth"] = VALID_DEPTH
    resp = http_client.get(route, params=params)
    assert resp.status_code == 422


@pytest.mark.parametrize("route", ["/reconstruct/grid", "/reconstruct/netcdf"])
def test_invalid_latitude_grid_out_of_bounds_is_422(http_client: httpx.Client, route: str):
    params = dict(VALID_GRID, lat_min=-999.0, lat_max=18.0)
    resp = http_client.get(route, params=params)
    assert resp.status_code == 422


@pytest.mark.parametrize("route", ["/reconstruct", "/profile"])
def test_invalid_latitude_out_of_domain_is_400(http_client: httpx.Client, route: str):
    params = {"lat": 85.0, "lon": VALID_LON, "date": VALID_DATE}
    if route == "/reconstruct":
        params["depth"] = VALID_DEPTH
    resp = http_client.get(route, params=params)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_parameter"


@pytest.mark.parametrize("route", ["/reconstruct/grid", "/reconstruct/netcdf"])
def test_invalid_latitude_grid_out_of_domain_is_400(http_client: httpx.Client, route: str):
    params = dict(VALID_GRID, lat_min=1.0, lat_max=3.0)
    resp = http_client.get(route, params=params)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_parameter"


@pytest.mark.parametrize("route", ["/reconstruct/grid", "/reconstruct/netcdf"])
def test_invalid_latitude_inverted_bounds_is_400(http_client: httpx.Client, route: str):
    params = dict(VALID_GRID, lat_min=20.0, lat_max=10.0)
    resp = http_client.get(route, params=params)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_parameter"


@pytest.mark.parametrize("route", ["/reconstruct", "/profile"])
def test_invalid_latitude_non_numeric_is_422(http_client: httpx.Client, route: str):
    params = {"lat": "abc", "lon": VALID_LON, "date": VALID_DATE}
    if route == "/reconstruct":
        params["depth"] = VALID_DEPTH
    resp = http_client.get(route, params=params)
    assert resp.status_code == 422


# ============================================================================
# 3. Invalid Longitude Handling
# ============================================================================


@pytest.mark.parametrize("route", ["/reconstruct", "/profile"])
def test_invalid_longitude_out_of_bounds_is_422(http_client: httpx.Client, route: str):
    params = {"lat": VALID_LAT, "lon": 999.0, "date": VALID_DATE}
    if route == "/reconstruct":
        params["depth"] = VALID_DEPTH
    resp = http_client.get(route, params=params)
    assert resp.status_code == 422


@pytest.mark.parametrize("route", ["/reconstruct/grid", "/reconstruct/netcdf"])
def test_invalid_longitude_grid_out_of_bounds_is_422(http_client: httpx.Client, route: str):
    params = dict(VALID_GRID, lon_min=-999.0, lon_max=75.0)
    resp = http_client.get(route, params=params)
    assert resp.status_code == 422


@pytest.mark.parametrize("route", ["/reconstruct", "/profile"])
def test_invalid_longitude_out_of_domain_is_400(http_client: httpx.Client, route: str):
    params = {"lat": VALID_LAT, "lon": 0.0, "date": VALID_DATE}
    if route == "/reconstruct":
        params["depth"] = VALID_DEPTH
    resp = http_client.get(route, params=params)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_parameter"


@pytest.mark.parametrize("route", ["/reconstruct/grid", "/reconstruct/netcdf"])
def test_invalid_longitude_grid_out_of_domain_is_400(http_client: httpx.Client, route: str):
    params = dict(VALID_GRID, lon_min=20.0, lon_max=30.0)
    resp = http_client.get(route, params=params)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_parameter"


@pytest.mark.parametrize("route", ["/reconstruct/grid", "/reconstruct/netcdf"])
def test_invalid_longitude_inverted_bounds_is_400(http_client: httpx.Client, route: str):
    params = dict(VALID_GRID, lon_min=80.0, lon_max=60.0)
    resp = http_client.get(route, params=params)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_parameter"


@pytest.mark.parametrize("route", ["/reconstruct", "/profile"])
def test_invalid_longitude_non_numeric_is_422(http_client: httpx.Client, route: str):
    params = {"lat": VALID_LAT, "lon": "invalid_lon", "date": VALID_DATE}
    if route == "/reconstruct":
        params["depth"] = VALID_DEPTH
    resp = http_client.get(route, params=params)
    assert resp.status_code == 422


# ============================================================================
# 4. Invalid Date Handling
# ============================================================================


@pytest.mark.parametrize(
    "route,extra_params",
    [
        ("/reconstruct", {"lat": VALID_LAT, "lon": VALID_LON, "depth": VALID_DEPTH}),
        ("/profile", {"lat": VALID_LAT, "lon": VALID_LON}),
        ("/reconstruct/grid", {"lat_min": 12.0, "lat_max": 18.0, "lon_min": 65.0, "lon_max": 75.0}),
        ("/reconstruct/netcdf", {"lat_min": 12.0, "lat_max": 18.0, "lon_min": 65.0, "lon_max": 75.0}),
        ("/embedding", {}),
        ("/explainability", {}),
    ],
)
def test_invalid_date_format_is_422(http_client: httpx.Client, route: str, extra_params: dict):
    resp = http_client.get(route, params={"date": "not-a-date", **extra_params})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "route,extra_params",
    [
        ("/reconstruct", {"lat": VALID_LAT, "lon": VALID_LON, "depth": VALID_DEPTH}),
        ("/profile", {"lat": VALID_LAT, "lon": VALID_LON}),
        ("/reconstruct/grid", {"lat_min": 12.0, "lat_max": 18.0, "lon_min": 65.0, "lon_max": 75.0}),
        ("/reconstruct/netcdf", {"lat_min": 12.0, "lat_max": 18.0, "lon_min": 65.0, "lon_max": 75.0}),
        ("/embedding", {}),
        ("/explainability", {}),
    ],
)
def test_unavailable_date_is_404(http_client: httpx.Client, route: str, extra_params: dict):
    resp = http_client.get(route, params={"date": "1900-01-01", **extra_params})
    assert resp.status_code == 404
    assert resp.json()["error"] == "date_not_found"


# ============================================================================
# 5. Invalid Depth Handling
# ============================================================================


def test_invalid_negative_depth_reconstruct_is_422(http_client: httpx.Client):
    resp = http_client.get(
        "/reconstruct",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE, "depth": -10.0},
    )
    assert resp.status_code == 422


def test_invalid_negative_depth_reconstruct_grid_is_422(http_client: httpx.Client):
    resp = http_client.get("/reconstruct/grid", params={**VALID_GRID, "depth": -10.0})
    assert resp.status_code == 422


def test_invalid_negative_depth_reconstruct_netcdf_is_422(http_client: httpx.Client):
    resp = http_client.get("/reconstruct/netcdf", params={**VALID_GRID, "depth": -10.0})
    assert resp.status_code == 422


def test_invalid_negative_depth_explainability_is_422(http_client: httpx.Client):
    resp = http_client.get("/explainability", params={"date": VALID_DATE, "depth": -5.0})
    assert resp.status_code == 422


def test_invalid_non_numeric_depth_is_422(http_client: httpx.Client):
    resp = http_client.get(
        "/reconstruct",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE, "depth": "subsurface"},
    )
    assert resp.status_code == 422


# ============================================================================
# 6. Missing Data Handling (503 data_unavailable)
# ============================================================================


def test_missing_data_dates_is_503(client_no_data):
    resp = client_no_data.get("/dates")
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_data_quality_is_503(client_no_data):
    resp = client_no_data.get("/data/quality")
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_reconstruct_grid_is_503(client_no_data):
    resp = client_no_data.get(
        "/reconstruct/grid",
        params={"lat_min": 11.0, "lat_max": 19.0, "lon_min": 51.0, "lon_max": 59.0, "date": "2021-07-15"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_reconstruct_netcdf_is_503(client_no_data):
    resp = client_no_data.get(
        "/reconstruct/netcdf",
        params={"lat_min": 11.0, "lat_max": 19.0, "lon_min": 51.0, "lon_max": 59.0, "date": "2021-07-15"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_explainability_is_503(client_no_data):
    resp = client_no_data.get("/explainability", params={"date": "2021-07-15"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_reconstruct_is_503(client_no_data_with_model):
    resp = client_no_data_with_model.get(
        "/reconstruct",
        params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15", "depth": 50.0},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_profile_is_503(client_no_data_with_model):
    resp = client_no_data_with_model.get(
        "/profile",
        params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_embedding_is_503(client_no_data_with_model):
    resp = client_no_data_with_model.get(
        "/embedding",
        params={"date": "2021-07-15"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_missing_data_metrics_is_503(client_no_data_with_model):
    resp = client_no_data_with_model.get("/metrics", params={"split": "test"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_live_http_missing_data_dates_is_503(http_client_no_data: httpx.Client):
    resp = http_client_no_data.get("/dates")
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


def test_live_http_missing_data_reconstruct_is_503(http_client_no_data: httpx.Client):
    resp = http_client_no_data.get(
        "/reconstruct",
        params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15", "depth": 50.0},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "data_unavailable"


# ============================================================================
# 7. Model Unavailable Handling (503 model_unavailable)
# ============================================================================


def test_missing_model_model_info_is_503(client_no_model):
    resp = client_no_model.get("/model/info")
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_missing_model_reconstruct_is_503(client_no_model):
    resp = client_no_model.get(
        "/reconstruct",
        params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15", "depth": 50.0},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_missing_model_profile_is_503(client_no_model):
    resp = client_no_model.get(
        "/profile",
        params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_missing_model_reconstruct_grid_is_503(client_no_model):
    resp = client_no_model.get(
        "/reconstruct/grid",
        params={"lat_min": 11.0, "lat_max": 19.0, "lon_min": 51.0, "lon_max": 59.0, "date": "2021-07-15"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_missing_model_reconstruct_netcdf_is_503(client_no_model):
    resp = client_no_model.get(
        "/reconstruct/netcdf",
        params={"lat_min": 11.0, "lat_max": 19.0, "lon_min": 51.0, "lon_max": 59.0, "date": "2021-07-15"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_missing_model_embedding_is_503(client_no_model):
    resp = client_no_model.get("/embedding", params={"date": "2021-07-15"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_missing_model_explainability_is_503(client_no_model):
    resp = client_no_model.get("/explainability", params={"date": "2021-07-15"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_missing_model_metrics_is_503(client_no_model_with_targets):
    resp = client_no_model_with_targets.get("/metrics", params={"split": "test"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"


def test_live_http_missing_model_is_503(http_client_no_model: httpx.Client):
    resp = http_client_no_model.get("/model/info")
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_unavailable"

    resp2 = http_client_no_model.get(
        "/reconstruct",
        params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15", "depth": 50.0},
    )
    assert resp2.status_code == 503
    assert resp2.json()["error"] == "model_unavailable"


# ============================================================================
# 8. Successful Reconstruction Output Verification
# ============================================================================


def test_successful_reconstruction_point_physics(http_client: httpx.Client):
    resp = http_client.get(
        "/reconstruct",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE, "depth": 100.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    temp = data["temperature"]
    assert -2.0 < temp < 40.0  # Physically plausible ocean seawater temperature range
    assert isinstance(data["anomaly"], float)
    assert data["cache_hit"] in (True, False)
    assert data["latency_ms"] > 0
    assert data["mode"] == "point"


def test_successful_reconstruction_point_cache_and_snapping(http_client: httpx.Client):
    # Request depth 48.0 — should snap to nearest model depth 50.0
    resp1 = http_client.get(
        "/reconstruct",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE, "depth": 48.0},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["depth"] == 50.0  # Snapped to standard depth

    # Repeating the exact same request should hit cache
    resp2 = http_client.get(
        "/reconstruct",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE, "depth": 48.0},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["cache_hit"] is True
    assert data2["temperature"] == data1["temperature"]


def test_successful_reconstruction_profile_depth_structure(http_client: httpx.Client):
    resp = http_client.get(
        "/profile",
        params={"lat": VALID_LAT, "lon": VALID_LON, "date": VALID_DATE},
    )
    assert resp.status_code == 200
    data = resp.json()
    depths = data["depths"]
    temps = data["temperature"]
    assert len(depths) == 15
    assert len(temps) == 15
    assert depths == sorted(depths)  # Depths strictly increase
    for t in temps:
        assert -2.0 < t < 40.0
    # Physical ocean stratification: surface is warmer than deep ocean at 1000m+
    assert temps[0] > temps[-1]


def test_successful_reconstruction_grid_dimensions(http_client: httpx.Client):
    # Test specific depth slice
    resp_slice = http_client.get("/reconstruct/grid", params={**VALID_GRID, "depth": 50.0})
    assert resp_slice.status_code == 200
    data_slice = resp_slice.json()
    assert data_slice["depth"] == 50.0
    grid_slice = data_slice["temperature"]
    n_lat = len(data_slice["lat"])
    n_lon = len(data_slice["lon"])
    assert len(grid_slice) == n_lat
    assert len(grid_slice[0]) == n_lon
    # Ensure all grid values are physically plausible
    for row in grid_slice:
        for val in row:
            assert -2.0 < val < 40.0

    # Test all depths 3D grid
    resp_all = http_client.get("/reconstruct/grid", params=VALID_GRID)
    assert resp_all.status_code == 200
    data_all = resp_all.json()
    assert data_all["depth"] is None
    assert len(data_all["depths"]) == 15
    grid_all = data_all["temperature"]
    assert len(grid_all) == n_lat
    assert len(grid_all[0]) == n_lon
    assert len(grid_all[0][0]) == 15


def test_successful_reconstruction_netcdf_file_integrity(http_client: httpx.Client, tmp_path: Path):
    resp = http_client.get("/reconstruct/netcdf", params={**VALID_GRID, "depth": 75.0})
    assert resp.status_code == 200

    target_path = tmp_path / "downloaded_75m.nc"
    target_path.write_bytes(resp.content)

    dataset = load_netcdf(target_path, check_domain=False)
    assert "temperature" in dataset.variables
    temp_var = dataset.variables["temperature"]
    assert temp_var.units == "degC"
    assert temp_var.attrs.get("depth_m") == 75.0
    assert temp_var.dims == ("time", "lat", "lon")
    assert dataset.attrs["title"] == "NEER reconstructed temperature grid"
    assert dataset.attrs["date"] == VALID_DATE
    assert dataset.attrs["source"] == "NEER /reconstruct/netcdf"
