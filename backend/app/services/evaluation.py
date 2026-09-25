"""
Shared fixtures for the backend API tests (`tests/test_backend_*.py`).

Every fixture here builds *real* NEER artifacts — a genuine `NEERModel`
(constructed the same way `InferenceService.from_checkpoint` builds one,
via `NEERModel.from_config`), a genuine `TensorBundle` written to and
read back from `.npz`, and a genuine checkpoint file in exactly the
shape `scripts/train.py` writes — just small ones (a 16x16 grid, 15
model depths, `NEER_N_CHANNELS` input channels) so the test suite runs
in seconds on a laptop CPU. Nothing here is mocked or monkeypatched:
`/reconstruct`, `/profile`, `/embedding`, `/metrics`, `/evaluation/argo`
etc. run an actual forward pass through an actual `NEERModel` (and, for
`/evaluation/argo`, an actual Phase 26 pipeline run) against these
fixtures, the same code path production traffic uses.

Each test module imports the fixtures it needs by name, e.g.::

    from tests._backend_fixtures import client, empty_client

which is the standard pytest pattern for sharing fixtures defined
outside `conftest.py` (this file is deliberately *not* named
`conftest.py` so importing it — and therefore requiring `torch`/
`fastapi` — is opt-in per test module, exactly like `torch =
pytest.importorskip("torch")` elsewhere in this suite).

Phase 29B-1 addition
---------------------
`/metrics` and `/evaluation/argo` need more than the Phase 29A fixtures
did: real `targets`/`target_mask`/`depth`/`split_masks` on the tensor
bundle (so there is something to score), and, for a non-demo
`/evaluation/argo` run, real preprocessing-metadata normalization
statistics and a real ARGO CSV under `data_raw_path`. `build_repository`
grows `with_targets`/`with_metadata`/`data_raw_path` for exactly that,
kept optional so every Phase 29A fixture/test above is unaffected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional

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

#: The one target variable `configs/base.yaml`'s `tensors.target_variables`
#: names (`src.data.preprocessing.tensors.DEFAULT_TARGETS`), and the model
#: depths that same config declares — the source of truth for how many
#: depth levels a fixture checkpoint's output (and therefore any fixture
#: `targets` array) must have.
TARGET_NAME = "subsurface_temp"
DEPTHS: List[float] = [float(d) for d in load_config(ENVIRONMENT).depths]
N_DEPTHS = len(DEPTHS)


def build_tensor_bundle(path: Path, *, seed: int = 0, with_targets: bool = False) -> Path:
    """Write a small, real `TensorBundle` (`NEER_N_CHANNELS` channels,
    `NEER_CHANNEL_ORDER` names, three monthly timesteps) to `path`.

    With `with_targets=True`, also writes a real `targets`/`target_mask`
    (fully observed, `N_DEPTHS` levels matching the fixture checkpoint's
    architecture), `depth`, and a `train`/`val`/`test` split — one
    timestep each — so `/metrics` and `/evaluation/argo` have something
    real to score. Phase 29A fixtures leave these unset (`with_targets`
    defaults to `False`) since `/reconstruct` etc. don't need them and
    `/metrics`'s "dataset has no targets" `503` is itself something a
    test needs to exercise against a real, targetless bundle.
    """
    rng = np.random.default_rng(seed)
    n_time = len(DATES)
    inputs = rng.normal(0.0, 1.0, (n_time, NEER_N_CHANNELS, GRID_N_LAT, GRID_N_LON)).astype(
        "float32"
    )
    kwargs = {}
    if with_targets:
        targets = rng.normal(0.0, 1.0, (n_time, N_DEPTHS, GRID_N_LAT, GRID_N_LON)).astype(
            "float32"
        )
        kwargs["targets"] = targets
        kwargs["target_mask"] = np.ones_like(targets, dtype=bool)
        kwargs["target_names"] = [TARGET_NAME]
        kwargs["depth"] = np.array(DEPTHS, dtype=float)
        # One timestep per split — enough for `pool_tensor_bundle` to
        # produce a non-empty `PooledSplit` for every split name.
        split_masks = {name: np.zeros(n_time, dtype=bool) for name in ("train", "val", "test")}
        for i, name in enumerate(("train", "val", "test")):
            split_masks[name][i % n_time] = True
        kwargs["split_masks"] = split_masks

    bundle = TensorBundle(
        inputs=inputs,
        input_mask=np.ones_like(inputs, dtype=bool),
        channel_names=list(NEER_CHANNEL_ORDER),
        time=np.array(DATES, dtype="datetime64[ns]"),
        lat=LAT,
        lon=LON,
        attrs={"data_mode": "DEMO_SYNTHETIC"},
        **kwargs,
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


def build_metadata(path: Path, *, target_name: str = TARGET_NAME, seed: int = 0) -> Path:
    """Write a minimal real `neer_preprocessing_metadata.json` — just the
    one `normalization` step's per-depth z-score `statistics`, which is
    all `normalizer_stats_from_metadata` (used by both
    `InferenceService.from_checkpoint` and
    `src.argo_validation.predictions.predictions_from_checkpoint`) reads.
    `center`/`scale` have one real (non-fabricated-result, just
    arbitrary-but-valid) entry per fixture depth level.
    """
    rng = np.random.default_rng(seed)
    center = rng.normal(15.0, 3.0, N_DEPTHS).tolist()
    scale = np.abs(rng.normal(2.0, 0.5, N_DEPTHS)).tolist()
    metadata = {
        "steps": [
            {
                "step": "normalization",
                "state": {
                    "method": "zscore",
                    "statistics": {target_name: {"center": center, "scale": scale}},
                },
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return path


def build_argo_csv(path: Path) -> Path:
    """Write a small, real long-format ARGO CSV — the same
    `PLATFORM_NUMBER,CYCLE_NUMBER,JULD,LATITUDE,LONGITUDE,PRES,TEMP,TEMP_QC`
    shape `tests/test_argo_validation.py` uses — two floats, six QC=1
    levels each (above `ArgoValidationConfig.min_levels`'s default of
    5), inside the fixture grid's lat/lon/time range. No `DEMO` float-id
    prefix and no `.meta.json` sidecar, so `load_argo` labels it
    `ARGO_USER_SUPPLIED` (real, non-demo) per `src/argo_validation/io.py`.
    """
    rows = ["PLATFORM_NUMBER,CYCLE_NUMBER,JULD,LATITUDE,LONGITUDE,PRES,TEMP,TEMP_QC\n"]
    for float_id, lat, lon, date in (("6900001", 12.0, 55.0, "2021-06-15"), ("6900002", 15.0, 57.0, "2021-07-15")):
        for pres in (5, 10, 20, 50, 100, 200):
            temp = 28.0 - 0.02 * pres
            rows.append(f"{float_id},1,{date}T00:00:00Z,{lat},{lon},{pres},{temp},1\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(rows))
    return path


def build_repository(
    tmp_path: Path,
    *,
    with_data: bool = True,
    with_model: bool = True,
    with_targets: bool = False,
    with_metadata: bool = False,
    data_raw_path: Optional[Path] = None,
    max_grid_points: int = 4096,
) -> NEERRepository:
    """A `NEERRepository` pointed entirely at fixture paths under `tmp_path`
    — never the real `data/`/`artifacts/` trees. `with_data`/`with_model`
    let a test build a repository that is honestly missing one component,
    to exercise `503 data_unavailable` / `503 model_unavailable`.
    `with_targets` adds real targets/depth/splits (Phase 29B-1, `/metrics`
    and `/evaluation/argo`); `with_metadata` adds real normalization
    statistics so a checkpoint's output can be converted to degC (needed
    for a non-demo `/evaluation/argo` run). `data_raw_path` defaults to an
    empty directory under `tmp_path` — never the real `data/raw` — so a
    test controls exactly what ARGO source, if any, is found there.
    """
    tensor_path = tmp_path / "neer_tensors.npz"
    if with_data:
        build_tensor_bundle(tensor_path, with_targets=with_targets)

    checkpoint_path = tmp_path / "neer_best.pt"
    if with_model:
        build_checkpoint(checkpoint_path)

    metadata_path = tmp_path / "neer_preprocessing_metadata.json"
    if with_metadata:
        build_metadata(metadata_path)

    return NEERRepository(
        environment=ENVIRONMENT,
        tensor_path=tensor_path,
        metadata_path=metadata_path,  # written only when with_metadata=True -> otherwise absent
        checkpoint_path=checkpoint_path,
        climatology_path=tmp_path / "climatology.npz",  # never written -> absent
        data_raw_path=data_raw_path if data_raw_path is not None else (tmp_path / "raw_empty"),
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
    """A fully-loaded repository: real data + real model (no targets)."""
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


# --------------------------------------------------------------------------
# Phase 29B-1 — /metrics and /evaluation/argo fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def repository_metrics(tmp_path) -> NEERRepository:
    """Real data + real model + real targets/depth/splits — everything
    `/metrics` needs to compute an actual score."""
    return build_repository(tmp_path, with_targets=True)


@pytest.fixture()
def client_metrics(repository_metrics) -> Iterator[TestClient]:
    test_client = client_for(repository_metrics)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


@pytest.fixture()
def repository_argo_demo(tmp_path) -> NEERRepository:
    """Real data + real model + targets/depth (so `NeerGrid.from_bundle`
    has a depth axis), but no preprocessing metadata and no ARGO source
    under `data_raw_path` — the shape a `demo=true` pipeline-check run
    needs (real predictions are attempted and used when they succeed;
    demo mode only requires *a* prediction and *some* ARGO source, and
    supplies the labelled synthetic stand-ins for both when they're
    otherwise unavailable)."""
    return build_repository(tmp_path, with_targets=True)


@pytest.fixture()
def client_argo_demo(repository_argo_demo) -> Iterator[TestClient]:
    test_client = client_for(repository_argo_demo)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


@pytest.fixture()
def repository_argo_real(tmp_path) -> NEERRepository:
    """Real data + real model + real metadata + a real ARGO CSV under
    `data_raw_path` — everything a non-demo `/evaluation/argo` run needs
    to produce a genuine (not necessarily profile-matching) validation
    report end to end."""
    raw_dir = tmp_path / "raw"
    build_argo_csv(raw_dir / "argo_profiles.csv")
    return build_repository(tmp_path, with_targets=True, with_metadata=True, data_raw_path=raw_dir)


@pytest.fixture()
def client_argo_real(repository_argo_real) -> Iterator[TestClient]:
    test_client = client_for(repository_argo_real)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)


@pytest.fixture()
def repository_argo_no_source(tmp_path) -> NEERRepository:
    """Real data + real model + targets/depth, but no ARGO data anywhere
    under `data_raw_path` — the honest "no ARGO data available" case for
    a non-demo run."""
    return build_repository(tmp_path, with_targets=True)


@pytest.fixture()
def client_argo_no_source(repository_argo_no_source) -> Iterator[TestClient]:
    test_client = client_for(repository_argo_no_source)
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_repository, None)