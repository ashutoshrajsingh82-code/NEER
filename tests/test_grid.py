"""Tests for the NEER ocean grid (src/data/grid.py): bounds, spacing,
monotonicity, grid shape, and metadata persistence.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.grid import (
    OceanGrid,
    create_latitude_grid,
    create_longitude_grid,
    create_ocean_grid,
    save_grid_metadata,
)
from src.utils.config import DomainConfig, load_config

DOMAIN = load_config("base").domain  # 5-30N, 45-105E, 0.25 deg (configs/base.yaml)

EXPECTED_N_LAT = 101  # (30 - 5) / 0.25 + 1
EXPECTED_N_LON = 241  # (105 - 45) / 0.25 + 1


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


def test_latitude_grid_bounds():
    lats = create_latitude_grid(DOMAIN)
    assert lats[0] == pytest.approx(DOMAIN.lat_min)
    assert lats[-1] == pytest.approx(DOMAIN.lat_max)


def test_longitude_grid_bounds():
    lons = create_longitude_grid(DOMAIN)
    assert lons[0] == pytest.approx(DOMAIN.lon_min)
    assert lons[-1] == pytest.approx(DOMAIN.lon_max)


def test_latitude_grid_within_bounds():
    lats = create_latitude_grid(DOMAIN)
    assert np.all(lats >= DOMAIN.lat_min)
    assert np.all(lats <= DOMAIN.lat_max)


def test_longitude_grid_within_bounds():
    lons = create_longitude_grid(DOMAIN)
    assert np.all(lons >= DOMAIN.lon_min)
    assert np.all(lons <= DOMAIN.lon_max)


def test_ocean_grid_bounds_match_domain():
    grid = create_ocean_grid(DOMAIN)
    assert grid.bounds == (DOMAIN.lat_min, DOMAIN.lat_max, DOMAIN.lon_min, DOMAIN.lon_max)


# --------------------------------------------------------------------------
# Spacing
# --------------------------------------------------------------------------


def test_latitude_grid_spacing_matches_resolution():
    lats = create_latitude_grid(DOMAIN)
    diffs = np.diff(lats)
    assert np.allclose(diffs, DOMAIN.resolution)


def test_longitude_grid_spacing_matches_resolution():
    lons = create_longitude_grid(DOMAIN)
    diffs = np.diff(lons)
    assert np.allclose(diffs, DOMAIN.resolution)


def test_grid_spacing_is_uniform():
    lats = create_latitude_grid(DOMAIN)
    diffs = np.diff(lats)
    assert np.ptp(diffs) < 1e-9  # max - min spacing ~ 0 (uniform)


# --------------------------------------------------------------------------
# Monotonicity
# --------------------------------------------------------------------------


def test_latitude_grid_strictly_increasing():
    lats = create_latitude_grid(DOMAIN)
    assert np.all(np.diff(lats) > 0)


def test_longitude_grid_strictly_increasing():
    lons = create_longitude_grid(DOMAIN)
    assert np.all(np.diff(lons) > 0)


def test_latitude_grid_no_duplicates():
    lats = create_latitude_grid(DOMAIN)
    assert len(np.unique(lats)) == len(lats)


def test_longitude_grid_no_duplicates():
    lons = create_longitude_grid(DOMAIN)
    assert len(np.unique(lons)) == len(lons)


# --------------------------------------------------------------------------
# Grid shape
# --------------------------------------------------------------------------


def test_latitude_grid_length():
    lats = create_latitude_grid(DOMAIN)
    assert len(lats) == EXPECTED_N_LAT


def test_longitude_grid_length():
    lons = create_longitude_grid(DOMAIN)
    assert len(lons) == EXPECTED_N_LON


def test_ocean_grid_shape():
    grid = create_ocean_grid(DOMAIN)
    assert grid.shape == (EXPECTED_N_LAT, EXPECTED_N_LON)


def test_ocean_grid_meshgrid_shapes_match():
    grid = create_ocean_grid(DOMAIN)
    assert grid.lat_grid.shape == (EXPECTED_N_LAT, EXPECTED_N_LON)
    assert grid.lon_grid.shape == (EXPECTED_N_LAT, EXPECTED_N_LON)


def test_ocean_grid_row_is_constant_latitude():
    # every column within a row of lat_grid should hold the same latitude
    grid = create_ocean_grid(DOMAIN)
    for row_idx in (0, EXPECTED_N_LAT // 2, EXPECTED_N_LAT - 1):
        row = grid.lat_grid[row_idx, :]
        assert np.allclose(row, row[0])


def test_ocean_grid_column_is_constant_longitude():
    # every row within a column of lon_grid should hold the same longitude
    grid = create_ocean_grid(DOMAIN)
    for col_idx in (0, EXPECTED_N_LON // 2, EXPECTED_N_LON - 1):
        col = grid.lon_grid[:, col_idx]
        assert np.allclose(col, col[0])


def test_ocean_grid_corners_match_bounds():
    grid = create_ocean_grid(DOMAIN)
    assert grid.lat_grid[0, 0] == pytest.approx(DOMAIN.lat_min)
    assert grid.lat_grid[-1, 0] == pytest.approx(DOMAIN.lat_max)
    assert grid.lon_grid[0, 0] == pytest.approx(DOMAIN.lon_min)
    assert grid.lon_grid[0, -1] == pytest.approx(DOMAIN.lon_max)


# --------------------------------------------------------------------------
# Defaults (no domain argument passed)
# --------------------------------------------------------------------------


def test_create_latitude_grid_uses_default_domain_when_none_given():
    lats = create_latitude_grid()
    assert lats[0] == pytest.approx(DOMAIN.lat_min)
    assert lats[-1] == pytest.approx(DOMAIN.lat_max)


def test_create_ocean_grid_uses_default_domain_when_none_given():
    grid = create_ocean_grid()
    assert grid.shape == (EXPECTED_N_LAT, EXPECTED_N_LON)


def test_ocean_grid_works_with_custom_domain():
    custom = DomainConfig(lat_min=0, lat_max=1, lon_min=0, lon_max=1, resolution=0.5)
    grid = create_ocean_grid(custom)
    assert grid.shape == (3, 3)
    assert list(grid.latitudes) == [0.0, 0.5, 1.0]
    assert list(grid.longitudes) == [0.0, 0.5, 1.0]


# --------------------------------------------------------------------------
# Metadata persistence
# --------------------------------------------------------------------------


def test_grid_metadata_contents():
    grid = create_ocean_grid(DOMAIN)
    metadata = grid.metadata()
    assert metadata["resolution_deg"] == DOMAIN.resolution
    assert metadata["bounds"] == {
        "lat_min": DOMAIN.lat_min,
        "lat_max": DOMAIN.lat_max,
        "lon_min": DOMAIN.lon_min,
        "lon_max": DOMAIN.lon_max,
    }
    assert metadata["shape"] == {"n_lat": EXPECTED_N_LAT, "n_lon": EXPECTED_N_LON}
    assert metadata["n_points"] == EXPECTED_N_LAT * EXPECTED_N_LON


def test_save_grid_metadata_writes_valid_json(tmp_path):
    grid = create_ocean_grid(DOMAIN)
    out_path = tmp_path / "grid_metadata.json"

    returned_path = save_grid_metadata(grid, path=out_path)

    assert returned_path == out_path
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["project"] == "NEER"
    assert saved["problem_id"] == "SIH26066"
    assert saved["organization"] == "MoES / INCOIS"
    assert saved["shape"] == {"n_lat": EXPECTED_N_LAT, "n_lon": EXPECTED_N_LON}
    assert "generated_at" in saved


def test_save_grid_metadata_creates_parent_dirs(tmp_path):
    grid = create_ocean_grid(DOMAIN)
    nested_path = tmp_path / "nested" / "dir" / "grid_metadata.json"

    save_grid_metadata(grid, path=nested_path)

    assert nested_path.exists()


def test_save_grid_metadata_default_path_is_data_processed():
    grid = create_ocean_grid(DOMAIN)
    path = save_grid_metadata(grid)
    try:
        assert path.exists()
        assert path.parent.name == "processed"
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["n_points"] == EXPECTED_N_LAT * EXPECTED_N_LON
    finally:
        # keep the repo clean; this test only checks the default location works
        if path.exists():
            path.unlink()
