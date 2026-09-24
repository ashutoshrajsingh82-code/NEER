"""Tests for the Phase 16 depth embedding (`src/models/depth_embedding.py`).

Same two-group layout as `tests/test_encoder.py` / `tests/test_vit.py`:
config tests that run without torch, then model tests skipped via
`pytest.importorskip` when torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import NEER_N_DEPTHS  # noqa: E402
from src.models.depth_embedding import (  # noqa: E402
    DEFAULT_DEPTH_EMBED_DIM,
    DEFAULT_DEPTHS,
    DepthEmbeddingConfig,
)

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_default_depths_match_the_15_neer_target_levels():
    assert DEFAULT_DEPTHS == (
        0.0,
        5.0,
        10.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        125.0,
        150.0,
        200.0,
        300.0,
        500.0,
        700.0,
        1000.0,
    )


def test_default_depths_count_matches_neer_n_depths():
    assert len(DEFAULT_DEPTHS) == NEER_N_DEPTHS == 15


def test_default_embed_dim_is_256():
    assert DepthEmbeddingConfig().embed_dim == DEFAULT_DEPTH_EMBED_DIM == 256


def test_num_depths_property():
    assert DepthEmbeddingConfig().num_depths == 15
    assert DepthEmbeddingConfig(depths=(0.0, 10.0, 100.0)).num_depths == 3


def test_depths_normalized_to_float_tuple():
    cfg = DepthEmbeddingConfig(depths=(0, 10, 100))  # ints in
    assert cfg.depths == (0.0, 10.0, 100.0)
    assert all(isinstance(d, float) for d in cfg.depths)


@pytest.mark.parametrize(
    "depths",
    [
        (10.0, 0.0, 100.0),  # not ascending
        (-5.0, 0.0, 10.0),  # negative
        (0.0, 10.0, 10.0),  # duplicate
        (),  # empty
    ],
)
def test_invalid_depths_rejected(depths):
    with pytest.raises(ValueError):
        DepthEmbeddingConfig(depths=depths)


def test_invalid_embed_dim_rejected():
    with pytest.raises(ValueError, match="embed_dim"):
        DepthEmbeddingConfig(embed_dim=0)
    with pytest.raises(ValueError, match="embed_dim"):
        DepthEmbeddingConfig(embed_dim=-4)


def test_depth_to_index_is_positional():
    cfg = DepthEmbeddingConfig(depths=(0.0, 5.0, 10.0))
    assert cfg.depth_to_index == {0.0: 0, 5.0: 1, 10.0: 2}


def test_index_for_valid_depth():
    cfg = DepthEmbeddingConfig()
    assert cfg.index_for(0.0) == 0
    assert cfg.index_for(1000) == 14  # int input, exact match
    assert cfg.index_for(100.0) == 7


def test_index_for_invalid_depth_raises():
    cfg = DepthEmbeddingConfig()
    with pytest.raises(ValueError, match="not one of the configured target depths"):
        cfg.index_for(15.0)  # not a target level (nearest-depth search is out of scope)


# --------------------------------------------------------------------------
# Model tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.models.depth_embedding import DepthEmbedding  # noqa: E402


def _small_depth_embedding(**overrides) -> DepthEmbedding:
    kwargs = dict(depths=(0.0, 10.0, 100.0, 1000.0), embed_dim=16)
    kwargs.update(overrides)
    return DepthEmbedding(DepthEmbeddingConfig(**kwargs))


# --- one shared table, not 15 networks -------------------------------------


def test_single_shared_embedding_table_not_per_depth_networks():
    # The whole model's learnable state must be exactly one
    # (num_depths, embed_dim) weight matrix - nothing per-depth-shaped
    # beyond that one table.
    depth_embed = _small_depth_embedding()
    params = list(depth_embed.parameters())
    assert len(params) == 1, "expected exactly one parameter tensor (the shared table)"
    assert params[0].shape == (4, 16)
    assert sum(p.numel() for p in depth_embed.parameters()) == 4 * 16


def test_default_config_table_shape():
    depth_embed = DepthEmbedding()
    assert depth_embed.embedding.weight.shape == (15, 256)
    assert depth_embed.num_depths == 15
    assert depth_embed.embed_dim == 256


# --- shapes ------------------------------------------------------------


def test_single_index_lookup_shape():
    depth_embed = _small_depth_embedding()
    out = depth_embed(torch.tensor(2))
    assert out.shape == (16,)


def test_batch_index_lookup_shape():
    depth_embed = _small_depth_embedding()
    out = depth_embed(torch.tensor([0, 1, 2, 3, 0]))
    assert out.shape == (5, 16)


def test_all_embeddings_shape():
    depth_embed = _small_depth_embedding()
    table = depth_embed.all_embeddings()
    assert table.shape == (4, 16)


def test_default_all_embeddings_shape_is_15_by_256():
    depth_embed = DepthEmbedding()
    table = depth_embed.all_embeddings()
    assert table.shape == (15, 256)
    assert torch.isfinite(table).all()


def test_forward_from_depths_scalar_shape():
    depth_embed = _small_depth_embedding()
    out = depth_embed.forward_from_depths(100.0)
    assert out.shape == (16,)


def test_forward_from_depths_batch_shape():
    depth_embed = _small_depth_embedding()
    out = depth_embed.forward_from_depths([0.0, 1000.0, 10.0])
    assert out.shape == (3, 16)


def test_forward_from_depths_accepts_tensor_input():
    depth_embed = _small_depth_embedding()
    out = depth_embed.forward_from_depths(torch.tensor([0.0, 100.0]))
    assert out.shape == (2, 16)


# --- correctness: index vs. physical-value lookups agree ----------------


def test_forward_from_depths_matches_index_lookup():
    depth_embed = _small_depth_embedding()
    by_value = depth_embed.forward_from_depths([0.0, 100.0, 1000.0])
    by_index = depth_embed(torch.tensor([0, 2, 3]))
    assert torch.equal(by_value, by_index)


def test_all_embeddings_matches_row_by_row_lookup():
    depth_embed = _small_depth_embedding()
    table = depth_embed.all_embeddings()
    for i, depth in enumerate(depth_embed.depths):
        row = depth_embed.forward_from_depths(depth)
        assert torch.equal(table[i], row)


def test_forward_from_depths_rejects_non_target_depth():
    depth_embed = _small_depth_embedding()
    with pytest.raises(ValueError, match="not one of the configured target depths"):
        depth_embed.forward_from_depths(50.0)  # valid NEER depth, but not in this small table


def test_out_of_range_index_rejected():
    depth_embed = _small_depth_embedding()  # 4 depths -> valid indices 0..3
    with pytest.raises(ValueError, match="out of range"):
        depth_embed(torch.tensor(4))
    with pytest.raises(ValueError, match="out of range"):
        depth_embed(torch.tensor(-1))


# --- distinctness & determinism -----------------------------------------


def test_different_depths_get_different_embeddings():
    depth_embed = DepthEmbedding()
    table = depth_embed.all_embeddings()
    for i in range(table.shape[0]):
        for j in range(i + 1, table.shape[0]):
            assert not torch.equal(table[i], table[j])


def test_lookup_is_deterministic():
    depth_embed = _small_depth_embedding().eval()
    a = depth_embed.forward_from_depths(100.0)
    b = depth_embed.forward_from_depths(100.0)
    assert torch.equal(a, b)


def test_batch_lookup_matches_repeated_single_lookup():
    depth_embed = _small_depth_embedding()
    batch = depth_embed(torch.tensor([1, 1, 2]))
    singles = torch.stack(
        [depth_embed(torch.tensor(1)), depth_embed(torch.tensor(1)), depth_embed(torch.tensor(2))]
    )
    assert torch.equal(batch, singles)


# --- gradients: rows specialize independently ----------------------------


def test_gradients_flow_only_to_looked_up_depth_rows():
    depth_embed = _small_depth_embedding()
    out = depth_embed(torch.tensor([0, 2]))  # only rows 0 and 2 looked up
    out.sum().backward()

    grad = depth_embed.embedding.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[0]) > 0
    assert torch.count_nonzero(grad[2]) > 0
    assert torch.equal(grad[1], torch.zeros_like(grad[1]))
    assert torch.equal(grad[3], torch.zeros_like(grad[3]))


def test_gradients_flow_to_all_rows_when_all_are_used():
    depth_embed = _small_depth_embedding()
    depth_embed.all_embeddings().sum().backward()
    grad = depth_embed.embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    for i in range(depth_embed.num_depths):
        assert torch.count_nonzero(grad[i]) > 0
