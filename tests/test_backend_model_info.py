"""Tests for `GET /model/info` (Requirement 6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from tests._backend_fixtures import client, client_no_model, repository, repository_no_model  # noqa: E402


def test_model_info_status_code(client):
    assert client.get("/model/info").status_code == 200


def test_model_info_architecture_matches_the_loaded_checkpoint(client):
    data = client.get("/model/info").json()
    arch = data["architecture"]
    # These are the actual `demo` config's model architecture (Requirement
    # 6/12) -> in_channels/embed_dim/num_depths are never fabricated.
    assert arch["in_channels"] == NEER_N_CHANNELS
    assert arch["embed_dim"] == 256
    assert arch["num_depths"] == 15
    assert len(arch["depths"]) == 15
    assert arch["use_gnn"] is False
    assert arch["uncertainty_enabled"] is False


def test_model_info_includes_runtime_and_checkpoint_metadata(client):
    data = client.get("/model/info").json()
    assert data["runtime"]["device"] == "cpu"
    assert data["checkpoint"]["epoch"] == 1
    assert data["checkpoint"]["val_loss"] == pytest.approx(0.5)
    assert data["environment"] == "demo"


def test_model_info_response_schema_has_only_declared_top_level_fields(client):
    data = client.get("/model/info").json()
    assert set(data.keys()) == {"architecture", "runtime", "checkpoint", "environment"}


def test_model_info_without_a_checkpoint_is_503(client_no_model):
    response = client_no_model.get("/model/info")
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "model_unavailable"
    assert "detail" in body