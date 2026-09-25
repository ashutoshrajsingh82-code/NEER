"""Tests for `GET /data/quality` (Phase 29B-2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from tests._backend_fixtures import (  # noqa: E402
    client,
    client_metrics,
    client_no_data,
    repository,
    repository_metrics,
    repository_no_data,
)


def test_data_quality_valid_request_status_code(client):
    assert client.get("/data/quality").status_code == 200


def test_data_quality_response_schema(client):
    data = client.get("/data/quality").json()
    for key in (
        "data_mode",
        "is_synthetic",
        "source_tensors_path",
        "dates",
        "summary",
        "spatial_coverage",
        "channels",
        "targets",
    ):
        assert key in data
    assert isinstance(data["is_synthetic"], bool)
    assert isinstance(data["channels"], dict) and data["channels"]
    assert "count" in data["dates"]
    assert data["dates"]["count"] > 0


def test_data_quality_with_targets(client_metrics):
    data = client_metrics.get("/data/quality").json()
    assert data["targets"] is not None
    assert "variables" in data["targets"]
    assert "subsurface_temp" in data["targets"]["variables"]
    assert data["targets"]["valid_fraction"] == 1.0


def test_data_quality_without_data_is_503(client_no_data):
    response = client_no_data.get("/data/quality")
    assert response.status_code == 503
    assert response.json()["error"] == "data_unavailable"
