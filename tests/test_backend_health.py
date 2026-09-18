"""Tests for the NEER backend /health endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_status_code():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body():
    response = client.get("/health")
    data = response.json()
    assert data == {
        "status": "ok",
        "project": "NEER",
        "problem_id": "SIH26066",
    }
