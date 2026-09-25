"""
Shared fixtures for the Phase 29A backend API tests (`tests/test_backend_*.py`).

Every fixture here builds *real* NEER artifacts — a genuine `NEERModel`
(constructed the same way `InferenceService.from_checkpoint` builds one,
via `NEERModel.from_config`), a genuine `TensorBundle` written to and
read back from `.npz`, and a genuine checkpoint file in exactly the
shape `scripts/train.py` writes — just small ones (a 16x16 grid, 15
model depths, `NEER_N_CHANNELS` input channels) so the test suite runs
in seconds on a laptop CPU. Nothing here is mocked or monkeypatched:
`/reconstruct`, `/profile`, `/embedding` etc. run an actual forward
pass through an actual `NEERModel` against these fixtures, the same
code path production traffic uses.

Each test module imports the fixtures it needs by name, e.g.::

    from tests._backend_fixtures import client, empty_client

which is the standard pytest pattern for sharing fixtures defined
outside `conftest.py` (this file is deliberately *not* named
`conftest.py` so importing it — and therefore requiring `torch`/
`fastapi` — is opt-in per test module, exactly like `torch =
pytest.importorskip("torch")` elsewhere in this suite).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("fastapi", reason="fastapi is required to test the backend API")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.dependencies import get_repository  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.services.repository import NEERRepository  # noqa: E402
from src.data.preprocessing.channels import NEER_CHANNEL_ORDER, NEER_N_CHANNELS  # noqa: E402
from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.models.neer_model import NEERModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402

#: Small but real spatial grid — well inside the demo/base domain
#: (lat 5-30, lon 45-105), and a multiple of the ViT's 8-cell patch size
#: so patching isn't degenerate.
GRID_N_LAT = 16
GRID_N_LON = 16
LAT = np.linspace(10.0, 20.0, GRID_N_LAT)
LON = np.linspace(50.0, 60.0, GRID_N_LON)
DATES = ["2021-06-15", "2021-07-15", "2021-08-15"]
ENVIRONMENT = "demo"


def build_tensor_bundle(path: Path, *, seed: int = 0) -> Path:
    """Write a small, real `TensorBundle` (`NEER_N_CHANNELS` channels,
    `NEER_CHANNEL_ORDER` names, three monthly timesteps) to `path`."""
    rng = np.random.default_rng(seed)
    n_time = len(DATES)
    inputs = rng.normal(0.0, 1.0, (n_time, NEER_N_CHANNELS, GRID_N_LAT, GRID_N_LON)).astype(
        "float32"
    )
    bundle = TensorBundle(
        inputs=inputs,
        input_mask=np.ones_like(inputs, dtype=bool),
        channel_names=list(NEER_CHANNEL_ORDER),
        time=np.array(DATES, dtype="datetime64[ns]"),
        lat=LAT,
        lon=LON,
        attrs={"data_mode": "DEMO_SYNTHETIC"},
    )
    bundle.save(path)
    return path


def build_checkpoint(path: Path, *, seed: int = 0) -> Path:
    """Write a real checkpoint — a `NEERModel.from_config(load_config("demo"))`
    state dict — in the flat `{epoch, model_state_dict, args, ...}` shape
    `scripts/train.py` writes (which is what `InferenceService.from_checkpoint`
    / `NEERRepository` expect)."""
    torch.manual_seed(seed)
    config = load_config(ENVIRONMENT)
    model = NEERModel.from_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.5,
            "best_val_loss": 0.5,
            "history": [],
            "args": {"environment": ENVIRONMENT},
        },
        path,
    )
    return path


def build_repository(
    tmp_path: Path,
    *,
    with_data: bool = True,
    with_model: bool = True,
    max_grid_points: int = 4096,
) -> NEERRepository:
    """A `NEERRepository` pointed entirely at fixture paths under `tmp_path`
    — never the real `data/`/`artifacts/` trees. `with_data`/`with_model`
    let a test build a repository that is honestly missing one component,
    to exercise `503 data_unavailable` / `503 model_unavailable`."""
    tensor_path = tmp_path / "neer_tensors.npz"
    if with_data:
        build_tensor_bundle(tensor_path)

    checkpoint_path = tmp_path / "neer_best.pt"
    if with_model:
        build_checkpoint(checkpoint_path)

    return NEERRepository(
        environment=ENVIRONMENT,
        tensor_path=tensor_path,
        metadata_path=tmp_path / "neer_preprocessing_metadata.json",  # never written -> absent
        checkpoint_path=checkpoint_path,
        climatology_path=tmp_path / "climatology.npz",  # never written -> absent
        cache_size=8,
        max_grid_points=max_grid_points,
    )


def client_for(repository: NEERRepository) -> TestClient:
    """A `TestClient` against the real app, with `get_repository` overridden
    to the given fixture repository instead of the app's lifespan-built one
    (the standard FastAPI dependency-override pattern — see
    `backend/app/dependencies.py`'s docstring)."""
    app.dependency_overrides[get_repository] = lambda: repository
    return TestClient(app)


@pytest.fixture()
def repository(tmp_path) -> NEERRepository:
    """A fully-loaded repository: real data + real model."""
    return build_repository(tmp_path)


@pytest.fixture()
def client(repository) -> Iterator[TestClient]:
    """A `TestClient` backed by a fully-loaded repository."""
    test_client = client_for(repository)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


@pytest.fixture()
def repository_no_model(tmp_path) -> NEERRepository:
    """A repository with real data but no checkpoint at all."""
    return build_repository(tmp_path, with_model=False)


@pytest.fixture()
def client_no_model(repository_no_model) -> Iterator[TestClient]:
    test_client = client_for(repository_no_model)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


@pytest.fixture()
def repository_no_data(tmp_path) -> NEERRepository:
    """A repository with neither a tensor bundle nor a checkpoint."""
    return build_repository(tmp_path, with_data=False, with_model=False)


@pytest.fixture()
def client_no_data(repository_no_data) -> Iterator[TestClient]:
    test_client = client_for(repository_no_data)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


@pytest.fixture()
def client_low_grid_limit(tmp_path) -> Iterator[TestClient]:
    """A repository whose `max_grid_points` is small enough that the
    fixture's own 16x16 grid trips the region-too-large check."""
    repo = build_repository(tmp_path, max_grid_points=10)
    test_client = client_for(repo)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)