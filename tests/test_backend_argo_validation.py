"""Tests for `GET /evaluation/argo` (Phase 29B-1, Requirement 5/6/7/8).

Real predictions come from the fixture checkpoint via
`src.argo_validation.predictions_from_checkpoint`; real ARGO profiles
come from a genuine CSV under `data_raw_path`. `demo=true` exercises the
labelled `DEMO_SYNTHETIC`/`DEMO_PIPELINE_CHECK_NOT_OBSERVATIONAL` path;
`demo=false` (the default) is a real validation attempt and must fail
honestly (`503`) when a real checkpoint/metadata/ARGO source isn't
actually available -- never a silent demo substitution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from src.argo_validation.pipeline import (  # noqa: E402
    VALIDATION_TYPE_DEMO,
    VALIDATION_TYPE_REAL,
)
from tests._backend_fixtures import (  # noqa: E402
    build_repository,
    client_argo_demo,
    client_argo_no_source,
    client_argo_real,
    client_for,
    client_no_data,
    repository_argo_demo,
    repository_argo_no_source,
    repository_argo_real,
    repository_no_data,
)
from backend.app.dependencies import get_repository  # noqa: E402
from backend.app.main import app  # noqa: E402


@pytest.fixture()
def client_argo_targets_no_model(tmp_path):
    """Real data + real targets/depth (so `NeerGrid.from_bundle` succeeds),
    but no checkpoint -- the honest `model_unavailable` case, distinct
    from the `data_unavailable` a bundle with no depth axis raises."""
    repo = build_repository(tmp_path, with_targets=True, with_model=False)
    test_client = client_for(repo)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


_SCHEMA_KEYS = {
    "phase", "validation_type", "observational_validation", "banner", "argo",
    "neer_predictions", "neer_grid", "separation_note", "config", "counts",
    "depth_coverage", "metrics", "pipeline_check_metrics", "warnings", "limitations",
}


# -- demo=true: always succeeds, never the default ------------------------


def test_argo_evaluation_demo_status_code(client_argo_demo):
    response = client_argo_demo.get("/evaluation/argo", params={"demo": "true"})
    assert response.status_code == 200


def test_argo_evaluation_demo_response_schema(client_argo_demo):
    data = client_argo_demo.get("/evaluation/argo", params={"demo": "true"}).json()
    assert _SCHEMA_KEYS.issubset(data.keys())
    assert data["phase"] == 26


def test_argo_evaluation_demo_is_labelled_not_observational(client_argo_demo):
    # Requirement 7/8: demo mode must never be reported as real validation.
    data = client_argo_demo.get("/evaluation/argo", params={"demo": "true"}).json()
    assert data["validation_type"] == VALIDATION_TYPE_DEMO
    assert data["observational_validation"] is False
    assert data["banner"] is not None
    assert data["metrics"] is None
    # pipeline_check_metrics may be None if nothing matched, but the real
    # "metrics" key must never be populated in demo mode.
    assert any("DEMO" in w or "demo" in w for w in data["warnings"]) or data["banner"]


def test_argo_evaluation_demo_is_not_the_default(client_argo_demo):
    # Omitting `demo` entirely must behave like `demo=false` -- a real
    # attempt, which fails honestly for this fixture (no ARGO source, no
    # preprocessing metadata under `repository_argo_demo`).
    response = client_argo_demo.get("/evaluation/argo")
    assert response.status_code == 503


# -- demo=false (real run) with everything actually available -------------


def test_argo_evaluation_real_status_code(client_argo_real):
    response = client_argo_real.get("/evaluation/argo", params={"demo": "false"})
    assert response.status_code == 200


def test_argo_evaluation_real_response_schema(client_argo_real):
    data = client_argo_real.get("/evaluation/argo", params={"demo": "false"}).json()
    assert _SCHEMA_KEYS.issubset(data.keys())
    assert data["phase"] == 26


def test_argo_evaluation_real_is_observational_validation(client_argo_real):
    data = client_argo_real.get("/evaluation/argo", params={"demo": "false"}).json()
    assert data["validation_type"] == VALIDATION_TYPE_REAL
    assert data["observational_validation"] is True
    assert data["banner"] is None
    assert data["pipeline_check_metrics"] is None


def test_argo_evaluation_real_uses_actual_predictions(client_argo_real):
    data = client_argo_real.get("/evaluation/argo", params={"demo": "false"}).json()
    assert data["neer_predictions"]["is_neer_output"] is True
    assert data["neer_predictions"]["origin"] != "demo_stand_in"


def test_argo_evaluation_real_reports_real_argo_source(client_argo_real):
    data = client_argo_real.get("/evaluation/argo", params={"demo": "false"}).json()
    assert data["argo"]["is_demo"] is False


def test_argo_evaluation_default_demo_param_is_false(client_argo_real):
    with_default = client_argo_real.get("/evaluation/argo").json()
    explicit_false = client_argo_real.get(
        "/evaluation/argo", params={"demo": "false"}
    ).json()
    assert with_default["validation_type"] == explicit_false["validation_type"] == VALIDATION_TYPE_REAL


# -- demo=false (real run) with missing pieces -> honest 503s -------------


def test_argo_evaluation_real_without_argo_source_is_503(client_argo_no_source):
    # Real checkpoint present, but no preprocessing metadata and no ARGO
    # source under `data_raw_path` -- `_real_predictions` fails first
    # (metadata missing), which is itself an honest `503`.
    response = client_argo_no_source.get("/evaluation/argo", params={"demo": "false"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] in {"data_unavailable", "argo_data_unavailable", "model_unavailable"}


def test_argo_evaluation_without_model_is_503(client_argo_targets_no_model):
    response = client_argo_targets_no_model.get("/evaluation/argo", params={"demo": "false"})
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_argo_evaluation_without_data_is_503(client_no_data):
    # `repository_no_data` has neither a tensor bundle nor a checkpoint;
    # `require_bundle()` fails first.
    response = client_no_data.get("/evaluation/argo", params={"demo": "false"})
    assert response.status_code == 503
    assert response.json()["error"] == "data_unavailable"


# -- invalid query param type -> 422 ---------------------------------------


def test_argo_evaluation_non_boolean_demo_is_422(client_argo_real):
    response = client_argo_real.get("/evaluation/argo", params={"demo": "not-a-bool"})
    assert response.status_code == 422