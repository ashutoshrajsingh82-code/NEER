"""
Shared fixtures for the NEER test suite.

The demo dataset (`data/demo/neer_demo_ocean_dataset.npz`) is the fixture
the data-loading tests are built on, as Phase 05 requires. It is loaded
once per session — it is ~35 MB uncompressed, so reloading it per test
would dominate the runtime.

Tests that need a small, writable sample (the CSV and NetCDF round-trips)
use `demo_sample`, which carves a mostly-gap-free block out of the demo
SST field. The block is located dynamically rather than hardcoded, so the
tests survive the demo dataset being regenerated with a different seed or
a different number of time steps.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_demo_dataset  # noqa: E402
from src.data.loaders.demo_loader import (  # noqa: E402
    DEFAULT_DEMO_DATASET,
    DEFAULT_DEMO_METADATA,
)

#: Size of the sample block carved out for round-trip tests.
SAMPLE_N_TIME = 2
SAMPLE_N_LAT = 5
SAMPLE_N_LON = 4


def _require_demo_dataset() -> None:
    if not DEFAULT_DEMO_DATASET.exists():
        pytest.skip(
            f"demo dataset not found at {DEFAULT_DEMO_DATASET}; "
            "generate it with: python scripts/create_demo_data.py"
        )


@pytest.fixture(scope="session")
def demo_dataset():
    """The full synthetic demo dataset, loaded once for the whole session."""
    _require_demo_dataset()
    return load_demo_dataset()


@pytest.fixture(scope="session")
def demo_metadata() -> dict:
    """The demo dataset's metadata JSON, as written by the generator."""
    _require_demo_dataset()
    if not DEFAULT_DEMO_METADATA.exists():
        pytest.skip(f"demo metadata not found at {DEFAULT_DEMO_METADATA}")
    with open(DEFAULT_DEMO_METADATA, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_clearest_window(
    values: np.ndarray, n_lat: int, n_lon: int, stride: int = 4
) -> Tuple[int, int]:
    """Find the (lat, lon) offset of the block with the fewest missing values.

    `values` is (time, lat, lon). Scanning with a stride keeps this cheap
    on a 101x241 grid while still finding a usable block: the demo's gaps
    are spatially correlated, so clear blocks are common.
    """
    missing = np.isnan(values).sum(axis=0)
    best = None
    best_count = None
    for i in range(0, missing.shape[0] - n_lat + 1, stride):
        for j in range(0, missing.shape[1] - n_lon + 1, stride):
            count = int(missing[i : i + n_lat, j : j + n_lon].sum())
            if best_count is None or count < best_count:
                best, best_count = (i, j), count
                if count == 0:
                    return best
    if best is None:  # pragma: no cover - only if the grid is smaller than the block
        raise RuntimeError("demo grid is too small to carve a sample block from")
    return best


@dataclass(frozen=True)
class DemoSample:
    """A small, self-contained slice of the demo dataset.

    Used as the source of truth for round-trip tests: write it out as CSV
    or NetCDF, load it back through the loader under test, and assert the
    values survived unchanged.
    """

    time: np.ndarray  # (n_time,) datetime64[ns]
    lat: np.ndarray  # (n_lat,)
    lon: np.ndarray  # (n_lon,)
    sst: np.ndarray  # (n_time, n_lat, n_lon) degC
    sla: np.ndarray  # (n_time, n_lat, n_lon) m

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.sst.shape


@pytest.fixture(scope="session")
def demo_sample(demo_dataset) -> DemoSample:
    """A mostly-complete (time, lat, lon) block carved from the demo dataset."""
    sst_full = demo_dataset["sst"].values
    n_time = min(SAMPLE_N_TIME, sst_full.shape[0])
    lat_start, lon_start = _find_clearest_window(
        sst_full[:n_time], SAMPLE_N_LAT, SAMPLE_N_LON
    )

    lat_slice = slice(lat_start, lat_start + SAMPLE_N_LAT)
    lon_slice = slice(lon_start, lon_start + SAMPLE_N_LON)

    return DemoSample(
        time=demo_dataset.time[:n_time].copy(),
        lat=demo_dataset.lat[lat_slice].copy(),
        lon=demo_dataset.lon[lon_slice].copy(),
        sst=demo_dataset["sst"].values[:n_time, lat_slice, lon_slice].copy(),
        sla=demo_dataset["sla"].values[:n_time, lat_slice, lon_slice].copy(),
    )


@pytest.fixture
def sample_csv_path(demo_sample, tmp_path) -> Path:
    """The demo sample written out as a long-format observation CSV.

    Deliberately uses non-canonical column names (`date`, `latitude`,
    `LONGITUDE`, `sea_surface_temperature`) so the round-trip also
    exercises alias resolution.
    """
    import csv as csv_module

    path = tmp_path / "observations.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["date", "latitude", "LONGITUDE", "sea_surface_temperature", "sla"])
        for ti, timestamp in enumerate(demo_sample.time):
            date = str(np.datetime_as_string(timestamp, unit="D"))
            for yi, lat in enumerate(demo_sample.lat):
                for xi, lon in enumerate(demo_sample.lon):
                    sst = demo_sample.sst[ti, yi, xi]
                    sla = demo_sample.sla[ti, yi, xi]
                    writer.writerow(
                        [
                            date,
                            f"{lat:.4f}",
                            f"{lon:.4f}",
                            "" if np.isnan(sst) else f"{sst:.6f}",
                            "" if np.isnan(sla) else f"{sla:.6f}",
                        ]
                    )
    return path


# --------------------------------------------------------------------------
# Corruption helpers (Phase 06 validation tests)
# --------------------------------------------------------------------------
#
# The validation tests need demo data that is broken in one specific,
# known way, so that a failing check can be attributed to exactly one
# cause. `corrupt` builds such a dataset without touching the original:
# every array it changes is copied first, which also lets the tests assert
# that validation itself never mutates its input.


def corrupt(
    dataset,
    *,
    coords: dict | None = None,
    values: dict | None = None,
    units: dict | None = None,
    drop: tuple = (),
    attrs: dict | None = None,
):
    """Return a copy of `dataset` with specific, deliberate damage applied.

    Parameters
    ----------
    coords:
        `{coord_name: new_array}` — replace a coordinate axis.
    values:
        `{variable_name: new_array}` — replace a variable's data.
    units:
        `{variable_name: unit}` — relabel a variable's units.
    drop:
        Variable names to remove entirely.
    attrs:
        Dataset attributes to add or override.

    The original dataset is never modified.
    """
    from dataclasses import replace as _replace

    new_coords = {k: np.array(v, copy=True) for k, v in dataset.coords.items()}
    for name, array in (coords or {}).items():
        new_coords[name] = np.asarray(array)

    new_variables = {}
    for name, variable in dataset.variables.items():
        if name in drop:
            continue
        changed = {}
        if values and name in values:
            changed["values"] = np.asarray(values[name])
        if units and name in units:
            changed["units"] = units[name]
        new_variables[name] = _replace(variable, **changed) if changed else variable

    return _replace(
        dataset,
        variables=new_variables,
        coords=new_coords,
        attrs={**dataset.attrs, **(attrs or {})},
        validation=None,
    )


# --------------------------------------------------------------------------
# Preprocessing fixtures (Phase 07)
# --------------------------------------------------------------------------
#
# A full pipeline run over the demo dataset takes a couple of seconds, so
# it is done once per session and shared. Tests that need to compare two
# runs (the leakage tests) build their own pipelines instead — sharing a
# fitted one between them would defeat the point.


@pytest.fixture(scope="session")
def preprocessed_demo(demo_dataset):
    """One full `PreprocessingPipeline.run` over the demo dataset."""
    from src.data.preprocessing import PreprocessingPipeline

    return PreprocessingPipeline().run(demo_dataset)


def tiny_dataset(
    *,
    n_time: int = 6,
    n_lat: int = 4,
    n_lon: int = 5,
    n_depth: int = 3,
    start: str = "2020-01-15",
    with_mask: bool = True,
    seed: int = 0,
):
    """A small, fast `OceanDataset` for unit-testing individual steps.

    The demo dataset is 35 MB and exercises the pipeline end to end; this
    is for tests that want to assert an exact number, where a 4x5 grid is
    easier to reason about than a 101x241 one.
    """
    from src.data.loaders.representation import OceanDataset, Variable

    rng = np.random.default_rng(seed)
    time = np.array(
        [np.datetime64(start, "M") + np.timedelta64(i, "M") for i in range(n_time)],
        dtype="datetime64[ns]",
    )
    lat = np.round(5.0 + 0.25 * np.arange(n_lat), 6)
    lon = np.round(45.0 + 0.25 * np.arange(n_lon), 6)
    depth = np.array([0.0, 50.0, 200.0][:n_depth], dtype=float)

    variables = {
        "sst": Variable(
            name="sst",
            values=20.0 + rng.normal(0, 1, (n_time, n_lat, n_lon)),
            dims=("time", "lat", "lon"),
            units="degC",
        ),
        "subsurface_temp": Variable(
            name="subsurface_temp",
            values=15.0 + rng.normal(0, 1, (n_time, n_depth, n_lat, n_lon)),
            dims=("time", "depth", "lat", "lon"),
            units="degC",
        ),
    }
    if with_mask:
        ocean = np.ones((n_lat, n_lon), dtype=bool)
        ocean[0, 0] = False  # one land cell
        variables["land_mask"] = Variable(
            name="land_mask", values=ocean, dims=("lat", "lon"), units="bool"
        )

    coords = {"time": time, "depth": depth, "lat": lat, "lon": lon}
    return OceanDataset(
        variables=variables, coords=coords, attrs={}, source_format="synthetic-test"
    )
