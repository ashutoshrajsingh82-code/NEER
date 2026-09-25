"""Tests for the NEER backend `GET /health` endpoint (Requirement 5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from tests._backend_fixtures import (  # noqa: E402
    client,
    client_no_data,
    client_no_model,
    repository,
    repository_no_data,
    repository_no_model,
)


def test_health_status_code(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_reports_ok_when_data_and_model_loaded(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["components"]["api"]["status"] == "ok"
    assert data["components"]["data"]["status"] == "ok"
    assert data["components"]["model"]["status"] == "ok"


def test_health_data_component_reports_real_shape_not_fake(client):
    data = client.get("/health").json()
    data_component = data["components"]["data"]
    # Requirement 5: this must be the fixture's *actual* grid/timestep
    # counts, never a hard-coded placeholder.
    assert data_component["n_timesteps"] == 3
    assert data_component["grid_shape"] == [16, 16]
    assert data_component["is_synthetic"] is True


def test_health_is_degraded_without_a_model(client_no_model):
    data = client_no_model.get("/health").json()
    assert data["status"] == "degraded"
    assert data["components"]["model"]["status"] == "unavailable"
    assert "detail" in data["components"]["model"]
    # Data is still fine in this fixture.
    assert data["components"]["data"]["status"] == "ok"


def test_health_is_degraded_without_data_or_model(client_no_data):
    data = client_no_data.get("/health").json()
    assert data["status"] == "degraded"
    assert data["components"]["data"]["status"] == "unavailable"
    assert data["components"]["model"]["status"] == "unavailable"
    assert "detail" in data["components"]["data"]


def test_health_always_returns_200_even_when_degraded(client_no_data):
    # Liveness: the process is up and answering even if components aren't.
    assert client_no_data.get("/health").status_code == 200


def test_health_never_hardcodes_climatology_status(client):
    # No climatology fixture is ever written -> honestly "unavailable".
    data = client.get("/health").json()
    assert data["components"]["climatology"]["status"] == "unavailable"