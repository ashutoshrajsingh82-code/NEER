"""Tests for `GET /metrics` (Phase 29B-1, Requirement 3/4/6).

`backend.app.services.evaluation.metrics` runs the fixture's real
checkpoint over one split's real inputs and scores it against that
split's real, held-out targets via `src.evaluation.metrics.compute_profile_metrics`
— nothing here is mocked, so these tests exercise the actual forward
pass + scoring path production traffic uses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from tests._backend_fixtures import (  # noqa: E402
    N_DEPTHS,
    build_repository,
    client,
    client_for,
    client_metrics,
    client_no_data,
    repository,
    repository_metrics,
    repository_no_data,
)
from backend.app.dependencies import get_repository  # noqa: E402
from backend.app.main import app  # noqa: E402


@pytest.fixture()
def client_metrics_no_model(tmp_path):
    """Real data + real targets/splits, but no checkpoint -- the honest
    `model_unavailable` case for `/metrics` (distinct from the
    `data_unavailable` a targetless dataset raises)."""
    repo = build_repository(tmp_path, with_targets=True, with_model=False)
    test_client = client_for(repo)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


# -- valid requests -----------------------------------------------------


def test_metrics_valid_request_status_code(client_metrics):
    response = client_metrics.get("/metrics", params={"split": "test"})
    assert response.status_code == 200


def test_metrics_response_schema(client_metrics):
    data = client_metrics.get("/metrics", params={"split": "test"}).json()
    assert set(data.keys()) == {
        "phase", "split", "n_samples", "is_synthetic", "data_mode",
        "disclaimer", "source_tensors_path", "checkpoint", "metrics",
    }
    assert data["phase"] == 25
    assert data["split"] == "test"
    assert isinstance(data["n_samples"], int)
    assert data["n_samples"] > 0
    assert isinstance(data["is_synthetic"], bool)
    assert isinstance(data["source_tensors_path"], str)
    assert data["checkpoint"] is not None

    metrics_block = data["metrics"]
    assert set(metrics_block.keys()) == {"overall", "per_depth", "per_variable"}
    for key in ("rmse", "mae", "bias", "pearson", "r2", "n"):
        assert key in metrics_block["overall"]
    assert isinstance(metrics_block["per_depth"], dict)
    assert len(metrics_block["per_depth"]) == N_DEPTHS


def test_metrics_default_split_is_test(client_metrics):
    # `split` is optional and defaults to "test" (MetricsQueryParams).
    default = client_metrics.get("/metrics").json()
    explicit = client_metrics.get("/metrics", params={"split": "test"}).json()
    assert default["split"] == explicit["split"] == "test"


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_metrics_every_split_name_is_scorable(client_metrics, split):
    response = client_metrics.get("/metrics", params={"split": split})
    assert response.status_code == 200
    assert response.json()["split"] == split


def test_metrics_values_are_actual_numbers_not_placeholders(client_metrics):
    # Requirement 3/4: a real forward pass against random-but-fixed fixture
    # data produces finite, non-trivially-zero error metrics -- not a
    # fabricated/placeholder 0.0 or null across the board.
    data = client_metrics.get("/metrics", params={"split": "test"}).json()
    overall = data["metrics"]["overall"]
    assert overall["n"] > 0
    assert overall["rmse"] is not None
    assert overall["rmse"] >= 0.0
    assert overall["mae"] is not None
    assert overall["mae"] >= 0.0


# -- invalid split name -> 400 -------------------------------------------


def test_metrics_invalid_split_name_is_400(client_metrics):
    response = client_metrics.get("/metrics", params={"split": "bogus"})
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_parameter"


# -- unavailable data: no targets / no model -----------------------------


def test_metrics_dataset_without_targets_is_503(client):
    # `client`'s repository has real data but no targets/splits
    # (`with_targets` defaults to False).
    response = client.get("/metrics", params={"split": "test"})
    assert response.status_code == 503
    assert response.json()["error"] == "data_unavailable"


def test_metrics_without_model_is_503(client_metrics_no_model):
    response = client_metrics_no_model.get("/metrics", params={"split": "test"})
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_metrics_without_data_is_503(client_no_data):
    response = client_no_data.get("/metrics", params={"split": "test"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "data_unavailable"