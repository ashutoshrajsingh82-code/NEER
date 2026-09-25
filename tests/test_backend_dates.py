"""Tests for `GET /dates` (Requirement 7)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from tests._backend_fixtures import (  # noqa: E402
    DATES,
    client,
    client_no_data,
    repository,
    repository_no_data,
)


def test_dates_status_code(client):
    assert client.get("/dates").status_code == 200


def test_dates_returns_exactly_the_dates_in_the_loaded_bundle(client):
    data = client.get("/dates").json()
    assert data["dates"] == sorted(DATES)
    assert data["count"] == len(DATES)
    assert data["min_date"] == min(DATES)
    assert data["max_date"] == max(DATES)


def test_dates_reports_synthetic_data_mode(client):
    data = client.get("/dates").json()
    assert data["data_mode"] == "DEMO_SYNTHETIC"
    assert data["is_synthetic"] is True


def test_dates_response_schema(client):
    data = client.get("/dates").json()
    assert set(data.keys()) == {"dates", "count", "min_date", "max_date", "data_mode", "is_synthetic"}
    assert isinstance(data["dates"], list)
    assert all(isinstance(d, str) for d in data["dates"])


def test_dates_without_a_dataset_is_503(client_no_data):
    response = client_no_data.get("/dates")
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "data_unavailable"