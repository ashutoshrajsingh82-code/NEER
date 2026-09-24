"""Tests for the Phase 15 model embedding (`VisionTransformer.get_embedding`,
`CNNViTEncoder.get_embedding`).

Torch-only: `get_embedding` doesn't exist without a live model, so unlike
`test_vit.py` there's no config-level half to this file — every test here
is skipped via `pytest.importorskip` when torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.vit import CNNViTEncoder, ViTConfig, VisionTransformer  # noqa: E402


def _small_vit(**overrides) -> VisionTransformer:
    kwargs = dict(in_channels=16, patch_size=4, embed_dim=32, num_heads=4, depth=2, dropout=0.0)
    kwargs.update(overrides)
    return VisionTransformer(ViTConfig(**kwargs))


def _small_pipeline(**vit_overrides) -> CNNViTEncoder:
    cnn_cfg = CNNEncoderConfig(channels=(8, 16), dropout=0.0)
    vit_kwargs = dict(in_channels=16, patch_size=4, embed_dim=32, num_heads=4, depth=2, dropout=0.0)
    vit_kwargs.update(vit_overrides)
    return CNNViTEncoder(cnn_cfg, ViTConfig(**vit_kwargs))


# --------------------------------------------------------------------------
# Shape: (B, embed_dim), (B, 256) under defaults
# --------------------------------------------------------------------------


def test_vit_embedding_shape():
    vit = _small_vit().eval()
    emb = vit.get_embedding(torch.randn(3, 16, 10, 14))
    assert emb.shape == (3, 32)


def test_pipeline_embedding_shape():
    model = _small_pipeline().eval()
    emb = model.get_embedding(torch.randn(5, NEER_N_CHANNELS, 20, 30))
    assert emb.shape == (5, 32)


def test_default_config_embedding_is_batch_by_256():
    model = CNNViTEncoder().eval()
    with torch.no_grad():
        emb = model.get_embedding(torch.randn(2, NEER_N_CHANNELS, 101, 241))
    assert emb.shape == (2, 256)
    assert torch.isfinite(emb).all()


@pytest.mark.parametrize("batch", [1, 2, 7])
def test_embedding_shape_across_batch_sizes(batch):
    vit = _small_vit(depth=1).eval()
    emb = vit.get_embedding(torch.randn(batch, 16, 8, 12))
    assert emb.shape == (batch, 32)


# --------------------------------------------------------------------------
# It is THE actual representation (forward's own tokens, mean-pooled) —
# not a fake, detached, or separately-computed embedding.
# --------------------------------------------------------------------------


def test_vit_embedding_equals_mean_of_forward_tokens():
    vit = _small_vit(depth=1).eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        tokens = vit(x)
        emb = vit.get_embedding(x)
    assert torch.equal(emb, tokens.mean(dim=1))


def test_pipeline_embedding_equals_mean_of_forward_tokens():
    model = _small_pipeline().eval()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    with torch.no_grad():
        tokens = model(x)
        emb = model.get_embedding(x)
    assert torch.equal(emb, tokens.mean(dim=1))


def test_embedding_changes_when_weights_change():
    # Proves the embedding is a function of the model's actual learned
    # parameters, not a constant or an untied side computation.
    vit = _small_vit(depth=1)
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        before = vit.get_embedding(x).clone()
        for p in vit.parameters():
            p.add_(1.0)
        after = vit.get_embedding(x)
    assert not torch.allclose(before, after)


def test_gradients_flow_from_embedding_to_all_vit_parameters():
    vit = _small_vit(depth=1)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    vit.get_embedding(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in vit.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_gradients_from_embedding_reach_the_cnn_too():
    # A future depth decoder trains through this embedding; the CNN
    # backbone must be reachable by backprop through it, same as forward().
    model = _small_pipeline()
    model.get_embedding(torch.randn(2, NEER_N_CHANNELS, 12, 12)).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"


def test_embedding_uses_forward_validation():
    # get_embedding must go through forward()'s own checks, not a
    # separate/looser code path.
    vit = _small_vit()
    with pytest.raises(ValueError, match="input channels"):
        vit.get_embedding(torch.randn(2, 15, 8, 8))
    with pytest.raises(ValueError, match="4D"):
        vit.get_embedding(torch.randn(16, 8, 8))


# --------------------------------------------------------------------------
# Deterministic inference
# --------------------------------------------------------------------------


def test_vit_embedding_deterministic_in_eval_mode():
    vit = _small_vit(dropout=0.5, attention_dropout=0.5).eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        a = vit.get_embedding(x)
        b = vit.get_embedding(x)
    assert torch.equal(a, b)


def test_pipeline_embedding_deterministic_in_eval_mode():
    model = _small_pipeline(dropout=0.5).eval()
    x = torch.randn(2, NEER_N_CHANNELS, 12, 12)
    with torch.no_grad():
        a = model.get_embedding(x)
        b = model.get_embedding(x)
    assert torch.equal(a, b)


def test_embedding_deterministic_across_repeated_calls_default_config():
    model = CNNViTEncoder().eval()
    x = torch.randn(1, NEER_N_CHANNELS, 101, 241)
    with torch.no_grad():
        outs = [model.get_embedding(x) for _ in range(3)]
    for out in outs[1:]:
        assert torch.equal(outs[0], out)


def test_embedding_nondeterministic_in_train_mode_with_dropout():
    # Confirms determinism in eval mode isn't just an artifact of zero
    # dropout everywhere — train-mode dropout really does perturb it.
    vit = _small_vit(dropout=0.5).train()
    x = torch.randn(2, 16, 8, 8)
    assert not torch.allclose(vit.get_embedding(x), vit.get_embedding(x))


def test_batched_embedding_matches_per_sample_embedding_in_eval_mode():
    vit = _small_vit(dropout=0.1).eval()
    x = torch.randn(4, 16, 10, 14)
    with torch.no_grad():
        batched = vit.get_embedding(x)
        single = torch.cat([vit.get_embedding(x[i : i + 1]) for i in range(4)], dim=0)
    assert torch.allclose(batched, single, atol=1e-4)
