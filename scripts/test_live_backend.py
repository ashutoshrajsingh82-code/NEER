"""Phase 30 — Live Backend Runner & Integration Test Suite.

Runs the NEER FastAPI backend on a live TCP port and executes real HTTP requests
against every endpoint, testing:
- Valid request across all 11 endpoints + root
- Invalid latitude (422 out-of-bounds, 400 out-of-domain, 400 inverted bounds)
- Invalid longitude (422 out-of-bounds, 400 out-of-domain, 400 inverted bounds)
- Invalid date (422 malformed, 404 date_not_found)
- Invalid depth (422 negative depth, 422 non-numeric)
- Missing data (503 data_unavailable)
- Model unavailable (503 model_unavailable)
- Successful reconstruction (physics, depth snapping, caching, profile, grid, NetCDF)

Usage:
    python scripts/test_live_backend.py [--port PORT]
"""

from __future__ import annotations

import argparse
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from tests._backend_fixtures import build_repository


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _SilentServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        pass


def build_test_app(repo: NEERRepository) -> FastAPI:
    test_app = FastAPI(title="NEER Live Test App", version="0.30.0")
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


class LiveTestRunner:
    def __init__(self, port: int | None = None) -> None:
        self.port = port or find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.server: uvicorn.Server | None = None
        self.thread: threading.Thread | None = None
        self.results: List[Tuple[str, str, bool, str]] = []

    def start_backend(self) -> None:
        print(f"[1/4] Starting live NEER backend on {self.base_url} ...")
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = _SilentServer(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

        deadline = time.time() + 15.0
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    print(f"      Backend is READY. Health status: {r.json().get('status')}")
                    return
            except Exception:
                time.sleep(0.1)

        raise RuntimeError(f"Backend failed to respond on {self.base_url} within 15s")

    def stop_backend(self) -> None:
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=3.0)
        print("      Backend shut down cleanly.")

    def record(self, category: str, test_name: str, passed: bool, detail: str = "") -> None:
        status_sym = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_sym} [{category}] {test_name} {detail}")
        self.results.append((category, test_name, passed, detail))

    def run_tests(self) -> bool:
        with httpx.Client(base_url=self.base_url, timeout=20.0) as client:
            print("\n[2/4] Testing Real HTTP Requests Against Primary Live Backend ...\n")

            # 1. Valid Requests across all endpoints
            self._test_valid_requests(client)

            # 2. Invalid Latitude
            self._test_invalid_latitude(client)

            # 3. Invalid Longitude
            self._test_invalid_longitude(client)

            # 4. Invalid Date
            self._test_invalid_date(client)

            # 5. Invalid Depth
            self._test_invalid_depth(client)

            # 6. Successful Reconstruction
            self._test_successful_reconstruction(client)

        print("\n[3/4] Testing Missing Data & Model Unavailable Error Paths ...\n")
        self._test_missing_data_live()
        self._test_missing_model_live()

        print("\n[4/4] Summary & Verification Results\n")
        passed_count = sum(1 for _, _, p, _ in self.results if p)
        total_count = len(self.results)
        print(f"Total Tests Run: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")

        return passed_count == total_count

    def _test_valid_requests(self, client: httpx.Client) -> None:
        valid_date = "2020-06-01"
        valid_lat = 15.0
        valid_lon = 70.0
        valid_depth = 50.0

        endpoints = [
            ("Root /", "/", {}),
            ("Health", "/health", {}),
            ("Model Info", "/model/info", {}),
            ("Dates", "/dates", {}),
            (
                "Reconstruct Point",
                "/reconstruct",
                {"lat": valid_lat, "lon": valid_lon, "date": valid_date, "depth": valid_depth},
            ),
            (
                "Profile",
                "/profile",
                {"lat": valid_lat, "lon": valid_lon, "date": valid_date},
            ),
            (
                "Reconstruct Grid",
                "/reconstruct/grid",
                {"lat_min": 12.0, "lat_max": 18.0, "lon_min": 65.0, "lon_max": 75.0, "date": valid_date},
            ),
            (
                "Reconstruct NetCDF",
                "/reconstruct/netcdf",
                {"lat_min": 12.0, "lat_max": 18.0, "lon_min": 65.0, "lon_max": 75.0, "date": valid_date},
            ),
            ("Embedding", "/embedding", {"date": valid_date}),
            ("Metrics", "/metrics", {"split": "test"}),
            ("Evaluation ARGO", "/evaluation/argo", {"demo": "true"}),
            ("Explainability", "/explainability", {"date": valid_date}),
            ("Data Quality", "/data/quality", {}),
        ]

        for name, path, params in endpoints:
            try:
                r = client.get(path, params=params)
                passed = r.status_code == 200
                detail = f"(HTTP {r.status_code})"
                if not passed:
                    detail += f" - Response: {r.text[:100]}"
                self.record("Valid Request", f"GET {path} [{name}]", passed, detail)
            except Exception as e:
                self.record("Valid Request", f"GET {path} [{name}]", False, str(e))

    def _test_invalid_latitude(self, client: httpx.Client) -> None:
        # Out-of-bounds latitude ([-90, 90]) -> 422
        r = client.get("/reconstruct", params={"lat": 999.0, "lon": 70.0, "date": "2020-06-01", "depth": 50.0})
        self.record("Invalid Latitude", "Lat 999.0 out of [-90, 90] bounds -> 422", r.status_code == 422, f"status={r.status_code}")

        # Out-of-domain latitude ([5, 30]) -> 400
        r = client.get("/reconstruct", params={"lat": 85.0, "lon": 70.0, "date": "2020-06-01", "depth": 50.0})
        self.record("Invalid Latitude", "Lat 85.0 out of NEER domain -> 400", r.status_code == 400 and r.json().get("error") == "invalid_parameter", f"status={r.status_code}")

        # Inverted grid bounds (lat_min >= lat_max) -> 400
        r = client.get("/reconstruct/grid", params={"lat_min": 20.0, "lat_max": 10.0, "lon_min": 65.0, "lon_max": 75.0, "date": "2020-06-01"})
        self.record("Invalid Latitude", "Grid inverted lat_min >= lat_max -> 400", r.status_code == 400, f"status={r.status_code}")

    def _test_invalid_longitude(self, client: httpx.Client) -> None:
        # Out-of-bounds longitude ([-180, 360]) -> 422
        r = client.get("/reconstruct", params={"lat": 15.0, "lon": 999.0, "date": "2020-06-01", "depth": 50.0})
        self.record("Invalid Longitude", "Lon 999.0 out of bounds -> 422", r.status_code == 422, f"status={r.status_code}")

        # Out-of-domain longitude ([45, 105]) -> 400
        r = client.get("/reconstruct", params={"lat": 15.0, "lon": 0.0, "date": "2020-06-01", "depth": 50.0})
        self.record("Invalid Longitude", "Lon 0.0 out of NEER domain -> 400", r.status_code == 400 and r.json().get("error") == "invalid_parameter", f"status={r.status_code}")

        # Inverted grid bounds (lon_min >= lon_max) -> 400
        r = client.get("/reconstruct/grid", params={"lat_min": 12.0, "lat_max": 18.0, "lon_min": 80.0, "lon_max": 60.0, "date": "2020-06-01"})
        self.record("Invalid Longitude", "Grid inverted lon_min >= lon_max -> 400", r.status_code == 400, f"status={r.status_code}")

    def _test_invalid_date(self, client: httpx.Client) -> None:
        # Malformed date format -> 422
        r = client.get("/reconstruct", params={"lat": 15.0, "lon": 70.0, "date": "not-a-date", "depth": 50.0})
        self.record("Invalid Date", "Malformed date format 'not-a-date' -> 422", r.status_code == 422, f"status={r.status_code}")

        # Non-existent date -> 404
        r = client.get("/reconstruct", params={"lat": 15.0, "lon": 70.0, "date": "1900-01-01", "depth": 50.0})
        self.record("Invalid Date", "Unavailable date '1900-01-01' -> 404 date_not_found", r.status_code == 404 and r.json().get("error") == "date_not_found", f"status={r.status_code}")

    def _test_invalid_depth(self, client: httpx.Client) -> None:
        # Negative depth -> 422
        r = client.get("/reconstruct", params={"lat": 15.0, "lon": 70.0, "date": "2020-06-01", "depth": -10.0})
        self.record("Invalid Depth", "Negative depth -10.0 -> 422", r.status_code == 422, f"status={r.status_code}")

        # Non-numeric depth -> 422
        r = client.get("/reconstruct", params={"lat": 15.0, "lon": 70.0, "date": "2020-06-01", "depth": "subsurface"})
        self.record("Invalid Depth", "Non-numeric depth 'subsurface' -> 422", r.status_code == 422, f"status={r.status_code}")

    def _test_successful_reconstruction(self, client: httpx.Client) -> None:
        # Point reconstruction
        r = client.get("/reconstruct", params={"lat": 15.0, "lon": 70.0, "date": "2020-06-01", "depth": 50.0})
        data = r.json()
        temp = data.get("temperature", 0.0)
        passed_pt = r.status_code == 200 and (-2.0 < temp < 40.0) and isinstance(data.get("anomaly"), float)
        self.record("Successful Reconstruction", "Point reconstruction temperature physically plausible (-2 to 40 C)", passed_pt, f"temp={temp:.2f} C")

        # Cache hit
        r2 = client.get("/reconstruct", params={"lat": 15.0, "lon": 70.0, "date": "2020-06-01", "depth": 50.0})
        self.record("Successful Reconstruction", "Point reconstruction cache hit on repeat request", r2.json().get("cache_hit") is True)

        # Profile reconstruction
        r_prof = client.get("/profile", params={"lat": 15.0, "lon": 70.0, "date": "2020-06-01"})
        p_data = r_prof.json()
        depths = p_data.get("depths", [])
        temps = p_data.get("temperature", [])
        passed_prof = (
            r_prof.status_code == 200
            and len(depths) == 15
            and len(temps) == 15
            and depths == sorted(depths)
            and temps[0] > temps[-1]
        )
        self.record("Successful Reconstruction", "Profile 15 depth levels, sorted, surface warmer than deep ocean", passed_prof, f"surface={temps[0]:.2f}C, bottom={temps[-1]:.2f}C")

        # Grid reconstruction
        grid_params = {"lat_min": 12.0, "lat_max": 18.0, "lon_min": 65.0, "lon_max": 75.0, "date": "2020-06-01", "depth": 50.0}
        r_grid = client.get("/reconstruct/grid", params=grid_params)
        g_data = r_grid.json()
        g_temp = g_data.get("temperature", [])
        passed_grid = r_grid.status_code == 200 and len(g_temp) == len(g_data.get("lat", []))
        self.record("Successful Reconstruction", "Grid 2D horizontal slice matches spatial dimensions", passed_grid, f"grid={len(g_temp)}x{len(g_temp[0]) if g_temp else 0}")

        # NetCDF export
        r_nc = client.get("/reconstruct/netcdf", params=grid_params)
        passed_nc = r_nc.status_code == 200 and (
            "netcdf" in r_nc.headers.get("content-type", "").lower() or r_nc.content[:4] in (b"CDF\x01", b"CDF\x02", b"\x89HDF")
        )
        self.record("Successful Reconstruction", "NetCDF export returns valid binary NetCDF format", passed_nc, f"bytes={len(r_nc.content)}")

    def _test_missing_data_live(self) -> None:
        port = find_free_port()
        with tempfile.TemporaryDirectory() as td:
            repo = build_repository(Path(td), with_data=False, with_model=True)
            test_app = build_test_app(repo)
            cfg = uvicorn.Config(test_app, host="127.0.0.1", port=port, log_level="error")
            server = _SilentServer(cfg)
            th = threading.Thread(target=server.run, daemon=True)
            th.start()
            time.sleep(0.3)

            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as c:
                r1 = c.get("/dates")
                passed1 = r1.status_code == 503 and r1.json().get("error") == "data_unavailable"
                self.record("Missing Data", "GET /dates without data returns 503 data_unavailable", passed1, f"status={r1.status_code}")

                r2 = c.get("/reconstruct", params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15", "depth": 50.0})
                passed2 = r2.status_code == 503 and r2.json().get("error") == "data_unavailable"
                self.record("Missing Data", "GET /reconstruct without data returns 503 data_unavailable", passed2, f"status={r2.status_code}")

                r3 = c.get("/profile", params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15"})
                passed3 = r3.status_code == 503 and r3.json().get("error") == "data_unavailable"
                self.record("Missing Data", "GET /profile without data returns 503 data_unavailable", passed3, f"status={r3.status_code}")

            server.should_exit = True
            th.join(timeout=2.0)

    def _test_missing_model_live(self) -> None:
        port = find_free_port()
        with tempfile.TemporaryDirectory() as td:
            repo = build_repository(Path(td), with_data=True, with_model=False, with_targets=True)
            test_app = build_test_app(repo)
            cfg = uvicorn.Config(test_app, host="127.0.0.1", port=port, log_level="error")
            server = _SilentServer(cfg)
            th = threading.Thread(target=server.run, daemon=True)
            th.start()
            time.sleep(0.3)

            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as c:
                r1 = c.get("/model/info")
                passed1 = r1.status_code == 503 and r1.json().get("error") == "model_unavailable"
                self.record("Model Unavailable", "GET /model/info without checkpoint returns 503 model_unavailable", passed1, f"status={r1.status_code}")

                r2 = c.get("/reconstruct", params={"lat": 15.0, "lon": 55.0, "date": "2021-07-15", "depth": 50.0})
                passed2 = r2.status_code == 503 and r2.json().get("error") == "model_unavailable"
                self.record("Model Unavailable", "GET /reconstruct without checkpoint returns 503 model_unavailable", passed2, f"status={r2.status_code}")

                r3 = c.get("/metrics", params={"split": "test"})
                passed3 = r3.status_code == 503 and r3.json().get("error") == "model_unavailable"
                self.record("Model Unavailable", "GET /metrics without checkpoint returns 503 model_unavailable", passed3, f"status={r3.status_code}")

            server.should_exit = True
            th.join(timeout=2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Backend HTTP Integration Tester")
    parser.add_argument("--port", type=int, default=None, help="Port to run backend on")
    args = parser.parse_args()

    runner = LiveTestRunner(port=args.port)
    try:
        runner.start_backend()
        all_passed = runner.run_tests()
        return 0 if all_passed else 1
    finally:
        runner.stop_backend()


if __name__ == "__main__":
    sys.exit(main())
