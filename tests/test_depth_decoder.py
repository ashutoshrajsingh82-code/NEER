"""Tests for the Phase 17 depth decoder (`src/models/depth_decoder.py`).

Same two-group layout as `tests/test_encoder.py` / `tests/test_vit.py` /
`tests/test_depth_embedding.py`: config tests that run without torch,
then model tests skipped via `pytest.importorskip` when torch is not
installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.depth_decoder import (  # noqa: E402
    DEFAULT_DECODER_DROPOUT,
    DEFAULT_DECODER_MLP_RATIO,
    DEFAULT_DECODER_NUM_HEADS,
    DepthDecoderConfig,
)
from src.models.depth_embedding import DEFAULT_DEPTHS, DepthEmbeddingConfig  # noqa: E402

# --------------------------------------------------------------------------
# Config tests — no torch required
# --------------------------------------------------------------------------


def test_defaults():
    cfg = DepthDecoderConfig()
    assert cfg.embed_dim == 256
    assert cfg.num_heads == DEFAULT_DECODER_NUM_HEADS == 8
    assert cfg.mlp_ratio == DEFAULT_DECODER_MLP_RATIO
    assert cfg.dropout == DEFAULT_DECODER_DROPOUT == 0.0


def test_default_depth_config_matches_default_depths():
    cfg = DepthDecoderConfig()
    assert cfg.depths == DEFAULT_DEPTHS
    assert cfg.num_depths == 15


def test_derived_dims():
    cfg = DepthDecoderConfig(embed_dim=256, num_heads=8, mlp_ratio=2.0)
    assert cfg.head_dim == 32
    assert cfg.mlp_hidden_dim == 512


def test_embed_dim_must_divide_by_num_heads():
    with pytest.raises(ValueError, match="num_heads"):
        DepthDecoderConfig(embed_dim=100, num_heads=7)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"embed_dim": 0}, "embed_dim"),
        ({"embed_dim": -8}, "embed_dim"),
        ({"num_heads": 0}, "num_heads"),
        ({"mlp_ratio": 0}, "mlp_ratio"),
        ({"mlp_ratio": -1.0}, "mlp_ratio"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_values_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DepthDecoderConfig(**kwargs)


def test_custom_depth_config_is_used():
    small = DepthEmbeddingConfig(depths=(0.0, 10.0, 100.0), embed_dim=32)
    cfg = DepthDecoderConfig(embed_dim=32, num_heads=4, depth_config=small)
    assert cfg.depths == (0.0, 10.0, 100.0)
    assert cfg.num_depths == 3


def test_mismatched_depth_config_embed_dim_rejected():
    small = DepthEmbeddingConfig(embed_dim=64)
    with pytest.raises(ValueError, match="embed_dim"):
        DepthDecoderConfig(embed_dim=32, depth_config=small)


# --------------------------------------------------------------------------
# Model tests — require torch
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.models.depth_decoder import DepthDecoder  # noqa: E402


def _small_decoder(**overrides) -> DepthDecoder:
    depth_cfg = overrides.pop(
        "depth_config", DepthEmbeddingConfig(depths=(0.0, 10.0, 100.0, 1000.0), embed_dim=32)
    )
    kwargs = dict(embed_dim=32, num_heads=4, dropout=0.0, depth_config=depth_cfg)
    kwargs.update(overrides)
    return DepthDecoder(DepthDecoderConfig(**kwargs))


# --- shape ---------------------------------------------------------------


def test_output_shape_matches_num_depths():
    decoder = _small_decoder().eval()
    out = decoder(torch.randn(5, 32))
    assert out.shape == (5, 4)


def test_default_config_output_shape_is_batch_by_15():
    decoder = DepthDecoder().eval()
    with torch.no_grad():
        out = decoder(torch.randn(3, 256))
    assert out.shape == (3, 15)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_output_shape_across_batch_sizes(batch):
    decoder = _small_decoder().eval()
    out = decoder(torch.randn(batch, 32))
    assert out.shape == (batch, 4)


def test_rejects_wrong_embed_dim():
    decoder = _small_decoder()
    with pytest.raises(ValueError, match="expected 32"):
        decoder(torch.randn(2, 16))


def test_rejects_non_2d_input():
    decoder = _small_decoder()
    with pytest.raises(ValueError, match="2D"):
        decoder(torch.randn(2, 3, 32))


# --- depth ordering --------------------------------------------------------


def test_decoder_depths_match_the_configured_depth_list():
    decoder = DepthDecoder()
    assert decoder.depths == DEFAULT_DEPTHS
    assert decoder.num_depths == 15


def test_custom_depths_are_reported_in_configured_order():
    depth_cfg = DepthEmbeddingConfig(depths=(0.0, 10.0, 100.0, 1000.0), embed_dim=32)
    decoder = _small_decoder(depth_config=depth_cfg)
    assert decoder.depths == (0.0, 10.0, 100.0, 1000.0)


def test_output_column_i_is_computed_from_depth_i_row_alone():
    # Ties column i of the batched forward pass to the i-th row of the
    # depth-embedding table, one at a time, through the same shared
    # attention/MLP/head pipeline. If the columns were ever reordered
    # relative to `decoder.depths`, this would catch it.
    decoder = _small_decoder().eval()
    x = torch.randn(2, 32)
    with torch.no_grad():
        anomalies = decoder(x)
        context = x.unsqueeze(1)
        table = decoder.depth_embedding.all_embeddings()  # (4, 32), row i == depths[i]
        for i in range(decoder.num_depths):
            single_query = table[i : i + 1].unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, 1, 32)
            single_out = decoder._decode(single_query, context).squeeze(-1)  # (B,)
            assert torch.allclose(anomalies[:, i], single_out, atol=1e-5)


def test_reordering_the_depth_config_reorders_the_output_columns():
    # Two decoders with the same depths in different orders must
    # produce the same set of per-depth values, just permuted to match
    # each decoder's own `depths` order.
    torch.manual_seed(0)
    ascending = DepthEmbeddingConfig(depths=(0.0, 10.0, 100.0), embed_dim=16)
    descending = DepthEmbeddingConfig(depths=(0.0, 10.0, 100.0), embed_dim=16)  # config itself requires ascending

    dec_a = DepthDecoder(DepthDecoderConfig(embed_dim=16, num_heads=4, depth_config=ascending)).eval()
    dec_b = DepthDecoder(DepthDecoderConfig(embed_dim=16, num_heads=4, depth_config=descending)).eval()
    # Force decoder B's table to hold the same rows as A but permuted,
    # to simulate "a differently-ordered depth list" without violating
    # DepthEmbeddingConfig's own ascending-order requirement.
    perm = [2, 0, 1]
    with torch.no_grad():
        dec_b.depth_embedding.embedding.weight.copy_(dec_a.depth_embedding.embedding.weight[perm])
        dec_b.attn.load_state_dict(dec_a.attn.state_dict())
        dec_b.norm1.load_state_dict(dec_a.norm1.state_dict())
        dec_b.norm2.load_state_dict(dec_a.norm2.state_dict())
        dec_b.mlp.load_state_dict(dec_a.mlp.state_dict())
        dec_b.head.load_state_dict(dec_a.head.state_dict())

        x = torch.randn(3, 16)
        out_a = dec_a(x)
        out_b = dec_b(x)
    assert torch.allclose(out_a[:, perm], out_b, atol=1e-5)


# --- batch inference -------------------------------------------------------


def test_batched_inference_matches_per_sample_inference():
    decoder = _small_decoder(dropout=0.0).eval()
    x = torch.randn(4, 32)
    with torch.no_grad():
        batched = decoder(x)
        single = torch.cat([decoder(x[i : i + 1]) for i in range(4)], dim=0)
    assert torch.allclose(batched, single, atol=1e-4)


def test_batch_of_identical_samples_gives_identical_rows():
    decoder = _small_decoder().eval()
    x = torch.randn(1, 32).repeat(3, 1)
    with torch.no_grad():
        out = decoder(x)
    assert torch.allclose(out[0], out[1], atol=1e-5)
    assert torch.allclose(out[0], out[2], atol=1e-5)


def test_deterministic_in_eval_mode():
    decoder = _small_decoder(dropout=0.5).eval()
    x = torch.randn(2, 32)
    with torch.no_grad():
        a = decoder(x)
        b = decoder(x)
    assert torch.equal(a, b)


def test_nondeterministic_in_train_mode_with_dropout():
    decoder = _small_decoder(dropout=0.5).train()
    x = torch.randn(2, 32)
    assert not torch.allclose(decoder(x), decoder(x))


# --- gradient flow ---------------------------------------------------------


def test_gradients_flow_to_all_decoder_parameters():
    decoder = _small_decoder()
    x = torch.randn(2, 32, requires_grad=True)
    decoder(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_gradients_reach_every_row_of_the_depth_embedding_table():
    # Every depth is queried on every forward pass (all rows of
    # all_embeddings() go in), so every row should get a gradient.
    decoder = _small_decoder()
    decoder(torch.randn(2, 32)).sum().backward()
    grad = decoder.depth_embedding.embedding.weight.grad
    assert grad is not None
    for i in range(decoder.num_depths):
        assert torch.count_nonzero(grad[i]) > 0


def test_cpu_friendly_parameter_count_is_modest():
    n_params = sum(p.numel() for p in DepthDecoder().parameters())
    assert n_params < 5_000_000


def test_runs_on_cpu_device_explicitly():
    device = torch.device("cpu")
    decoder = _small_decoder().to(device)
    out = decoder(torch.randn(2, 32, device=device))
    assert out.device.type == "cpu"
