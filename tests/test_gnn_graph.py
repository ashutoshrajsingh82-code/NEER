"""Torch-free tests for the Phase 22 grid GNN: config, graph, and switch.

Everything here needs only numpy and the config loader, so it runs in an
environment without torch. (A module-level `pytest.importorskip("torch")`
skips the *whole* module, including tests defined above it, which is why
the model tests live in `tests/test_gnn.py` instead of below these.)

- `GNNConfig` validation and defaults.
- The grid graph: grid cells are nodes, neighbouring cells are edges —
  counts, symmetry, adjacency, directions, degrees.
- The `model.use_gnn` configuration switch (default `false`, and the
  `NEER_MODEL_USE_GNN` override).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_CHANNEL_ORDER  # noqa: E402
from src.models.gnn import (  # noqa: E402
    CURRENT_EDGE_DIM,
    DEFAULT_CURRENT_CHANNELS,
    GEOMETRY_EDGE_DIM,
    VALID_CONNECTIVITIES,
    GNNConfig,
    build_grid_edges,
    build_grid_graph,
    edge_directions,
)
from src.utils.config import load_config  # noqa: E402

# --------------------------------------------------------------------------
# GNNConfig
# --------------------------------------------------------------------------


def test_default_current_channels_come_from_the_authoritative_channel_order():
    u, v = DEFAULT_CURRENT_CHANNELS
    assert NEER_CHANNEL_ORDER[u] == "u_current"
    assert NEER_CHANNEL_ORDER[v] == "v_current"


def test_default_config_values():
    cfg = GNNConfig()
    assert cfg.in_channels == 64  # the default CNNEncoder's output width
    assert cfg.connectivity == 4
    assert cfg.use_current is True
    assert cfg.num_layers == 1


def test_edge_dim_counts_current_features_only_when_they_are_used():
    assert GNNConfig(use_current=False).edge_dim == GEOMETRY_EDGE_DIM
    assert GNNConfig(use_current=True).edge_dim == GEOMETRY_EDGE_DIM + CURRENT_EDGE_DIM


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0},
        {"hidden_dim": 0},
        {"num_layers": 0},
        {"connectivity": 6},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"current_channels": (3,)},
        {"current_channels": (3, 3)},
        {"current_channels": (-1, 4)},
        {"current_channels": (3.0, 4)},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GNNConfig(**kwargs)


def test_current_channels_list_is_normalised_to_a_tuple():
    assert GNNConfig(current_channels=[5, 6]).current_channels == (5, 6)


# --------------------------------------------------------------------------
# Graph topology: grid cells are nodes, neighbouring cells are edges
# --------------------------------------------------------------------------


def _expected_edge_count(h: int, w: int, connectivity: int) -> int:
    undirected = h * (w - 1) + (h - 1) * w
    if connectivity == 8:
        undirected += 2 * (h - 1) * (w - 1)
    return 2 * undirected  # each neighbour pair is two directed edges


@pytest.mark.parametrize("connectivity", VALID_CONNECTIVITIES)
@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (5, 1), (2, 2), (4, 7), (101, 241)])
def test_edge_count_matches_the_grid_formula(shape, connectivity):
    src, dst = build_grid_edges(*shape, connectivity)
    assert src.shape == dst.shape == (_expected_edge_count(*shape, connectivity),)


@pytest.mark.parametrize("connectivity", VALID_CONNECTIVITIES)
def test_edges_are_symmetric_unique_and_have_no_self_loops(connectivity):
    h, w = 5, 6
    src, dst = build_grid_edges(h, w, connectivity)
    assert src.min() >= 0 and dst.max() < h * w
    assert np.all(src != dst)
    forward = set(zip(src.tolist(), dst.tolist()))
    assert len(forward) == len(src), "duplicate edges"
    assert forward == {(b, a) for a, b in forward}, "every edge needs a reverse edge"


def test_four_connectivity_joins_only_edge_adjacent_cells():
    h, w = 6, 7
    src, dst = build_grid_edges(h, w, 4)
    d_row = np.abs(dst // w - src // w)
    d_col = np.abs(dst % w - src % w)
    assert np.all(d_row + d_col == 1)


def test_eight_connectivity_adds_the_diagonals_and_nothing_further():
    h, w = 6, 7
    src, dst = build_grid_edges(h, w, 8)
    d_row = np.abs(dst // w - src // w)
    d_col = np.abs(dst % w - src % w)
    assert np.all((d_row <= 1) & (d_col <= 1))
    assert np.any((d_row == 1) & (d_col == 1)), "no diagonal edges were built"


def test_a_single_cell_has_no_edges():
    src, dst = build_grid_edges(1, 1, 8)
    assert src.size == 0 and dst.size == 0


@pytest.mark.parametrize("bad", [(0, 5, 4), (5, 0, 4), (3, 3, 5)])
def test_invalid_grid_arguments_raise(bad):
    with pytest.raises(ValueError):
        build_grid_edges(*bad)


def test_edge_directions_are_unit_east_north_vectors():
    h, w = 3, 4
    src, dst = build_grid_edges(h, w, 8)
    direction = edge_directions(src, dst, w)
    assert direction.shape == (len(src), 2) and direction.dtype == np.float32
    assert np.allclose(np.hypot(direction[:, 0], direction[:, 1]), 1.0, atol=1e-6)

    lookup = {(a, b): d for a, b, d in zip(src.tolist(), dst.tolist(), direction)}
    # node = row * w + col; east is +col, north is +row.
    assert np.allclose(lookup[(0, 1)], (1.0, 0.0))  # (0,0) -> (0,1): east
    assert np.allclose(lookup[(1, 0)], (-1.0, 0.0))  # west
    assert np.allclose(lookup[(0, w)], (0.0, 1.0))  # (0,0) -> (1,0): north
    assert np.allclose(lookup[(w, 0)], (0.0, -1.0))  # south
    assert np.allclose(lookup[(0, w + 1)], (np.sqrt(0.5), np.sqrt(0.5)))  # north-east


def test_grid_graph_in_degrees_and_read_only_cache():
    graph = build_grid_graph(5, 6, 4)
    degree = graph.in_degree.reshape(5, 6)
    assert degree[2, 3] == 4  # interior
    assert degree[0, 3] == 3  # edge
    assert degree[0, 0] == 2  # corner
    assert graph.num_nodes == 30 and graph.num_edges == _expected_edge_count(5, 6, 4)
    assert build_grid_graph(5, 6, 4) is graph, "graph should be cached per (H, W, connectivity)"
    with pytest.raises(ValueError):
        graph.src[0] = 99  # shared via the cache, so must not be writable

    assert build_grid_graph(5, 6, 8).in_degree.reshape(5, 6)[2, 3] == 8


# --------------------------------------------------------------------------
# Configuration switch
# --------------------------------------------------------------------------


def test_use_gnn_is_off_by_default_in_every_environment():
    for environment in ("base", "demo", "development", "production"):
        assert load_config(environment).model.use_gnn is False


def test_use_gnn_can_be_switched_on_from_the_environment(monkeypatch):
    monkeypatch.setenv("NEER_MODEL_USE_GNN", "true")
    assert load_config("base").model.use_gnn is True
    monkeypatch.setenv("NEER_MODEL_USE_GNN", "false")
    assert load_config("base").model.use_gnn is False