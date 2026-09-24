"""Model tests for the Phase 22 optional grid GNN (`src/models/gnn.py`).

The config / graph / switch tests that need no torch are in
`tests/test_gnn_graph.py`; this file needs torch and is skipped as a
whole when it is absent (`pytest.importorskip`).

The tests are organised around the two configurations this phase
requires NEER to work in:

- **GNN disabled** (`use_gnn=False`, the default): the model must be
  exactly the plain CNN -> ViT -> decoder model — no GNN parameters, no
  GNN in the forward path.
- **GNN enabled** (`use_gnn=True`): the graph refinement must really be
  in the forward path, really be trained, and — when it is given the
  ocean current — really use it in its messages.

The `test_both_configurations_*` tests run one shared suite against both.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from src.data.preprocessing.channels import NEER_N_CHANNELS  # noqa: E402
from src.models import gnn as gnn_module  # noqa: E402
from src.models.depth_decoder import DepthDecoderConfig  # noqa: E402
from src.models.depth_embedding import DEFAULT_DEPTHS, DepthEmbeddingConfig  # noqa: E402
from src.models.encoder import CNNEncoderConfig  # noqa: E402
from src.models.gnn import (  # noqa: E402
    CURRENT_EDGE_DIM,
    DEFAULT_CURRENT_CHANNELS,
    GEOMETRY_EDGE_DIM,
    GNNConfig,
    GridGNN,
    build_grid_edges,
    current_edge_features,
    edge_directions,
)
from src.models.neer_model import NEERModel  # noqa: E402
from src.models.pretrain_encoder import PretrainEncoder, PretrainEncoderConfig  # noqa: E402
from src.models.vit import CNNViTEncoder, ViTConfig  # noqa: E402
from src.training.pretrain import save_pretrained_encoder_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402

#: Input channels used by the small models below. 16 is not the NEER
#: layout, so the default current channels (3, 4) are simply "the two
#: current channels of this input".
IN_CH = 16
CNN_OUT = 16
EMBED = 32
DEPTHS = (0.0, 10.0, 100.0, 1000.0)


def _small_configs():
    cnn_cfg = CNNEncoderConfig(in_channels=IN_CH, channels=(8, CNN_OUT), dropout=0.0)
    vit_cfg = ViTConfig(
        in_channels=CNN_OUT, patch_size=4, embed_dim=EMBED, num_heads=4, depth=1, dropout=0.0
    )
    decoder_cfg = DepthDecoderConfig(
        embed_dim=EMBED,
        num_heads=4,
        dropout=0.0,
        depth_config=DepthEmbeddingConfig(depths=DEPTHS, embed_dim=EMBED),
    )
    return cnn_cfg, vit_cfg, decoder_cfg


def _small_model(use_gnn: bool, gnn_config: GNNConfig | None = None, seed: int = 0) -> NEERModel:
    cnn_cfg, vit_cfg, decoder_cfg = _small_configs()
    if use_gnn and gnn_config is None:
        gnn_config = GNNConfig(in_channels=CNN_OUT, hidden_dim=8)
    torch.manual_seed(seed)
    return NEERModel(cnn_cfg, vit_cfg, decoder_cfg, use_gnn=use_gnn, gnn_config=gnn_config)


def _gnn_keys(model: torch.nn.Module) -> list:
    return [k for k in model.state_dict() if ".gnn." in f".{k}"]


# --------------------------------------------------------------------------
# GNN DISABLED
# --------------------------------------------------------------------------


def test_disabled_is_the_default():
    model = NEERModel()
    assert model.use_gnn is False
    assert model.encoder.uses_gnn is False
    assert model.encoder.gnn is None


def test_disabled_builds_no_gnn_parameters_or_state():
    model = _small_model(use_gnn=False)
    assert _gnn_keys(model) == []
    assert not any("gnn" in name for name, _ in model.named_parameters())
    assert not any(isinstance(m, GridGNN) for m in model.modules())
    # Only the pre-Phase-22 top-level parts exist.
    assert {k.split(".")[1] for k in model.state_dict() if k.startswith("encoder.")} == {"cnn", "vit"}


def test_disabled_forward_is_exactly_the_plain_cnn_vit_decoder_path():
    # With the GNN off, forward() must equal CNN -> ViT -> decoder computed
    # by hand: nothing else is in the path.
    model = _small_model(use_gnn=False).eval()
    x = torch.randn(2, IN_CH, 12, 14)
    with torch.no_grad():
        by_hand = model.decoder(model.encoder.vit.get_embedding(model.encoder.cnn(x)))
        assert torch.equal(model(x), by_hand)
        assert torch.equal(model.get_embedding(x), model.encoder.vit.get_embedding(model.encoder.cnn(x)))


def test_disabled_default_model_keeps_its_phase_18_size():
    n_params = sum(p.numel() for p in NEERModel().parameters())
    assert n_params == 3_710_753  # unchanged by Phase 22; guards against a GNN leaking in
    assert n_params < 10_000_000


def test_gnn_config_without_use_gnn_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="use_gnn"):
        NEERModel(gnn_config=GNNConfig())


# --------------------------------------------------------------------------
# GNN ENABLED — construction
# --------------------------------------------------------------------------


def test_enabled_builds_a_gnn_between_the_cnn_and_the_vit():
    model = _small_model(use_gnn=True)
    assert model.use_gnn is True
    assert isinstance(model.encoder.gnn, GridGNN)
    assert _gnn_keys(model), "no GNN parameters in state_dict"
    assert all(k.startswith("encoder.gnn.") for k in _gnn_keys(model))


def test_enabled_adds_only_gnn_parameters_to_the_disabled_model():
    off, on = _small_model(use_gnn=False), _small_model(use_gnn=True)
    non_gnn = set(on.state_dict()) - set(_gnn_keys(on))
    assert non_gnn == set(off.state_dict())
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on > n_off
    assert n_on - n_off == sum(p.numel() for p in on.encoder.gnn.parameters())


def test_enabled_default_gnn_config_matches_the_cnn_output_width():
    model = NEERModel(use_gnn=True)
    assert model.encoder.gnn.in_channels == model.encoder.cnn.out_channels
    assert model.encoder.gnn.uses_current is True


def test_enabled_default_model_stays_cpu_friendly():
    assert sum(p.numel() for p in NEERModel(use_gnn=True).parameters()) < 10_000_000


def test_gnn_width_must_match_the_cnn_output():
    cnn_cfg, vit_cfg, decoder_cfg = _small_configs()
    with pytest.raises(ValueError, match="out_channels"):
        NEERModel(cnn_cfg, vit_cfg, decoder_cfg, use_gnn=True, gnn_config=GNNConfig(in_channels=CNN_OUT + 1))


def test_current_channels_outside_the_input_are_rejected_at_construction():
    cfg = GNNConfig(in_channels=CNN_OUT, current_channels=(IN_CH, IN_CH + 1))
    with pytest.raises(ValueError, match="current_channels"):
        _small_model(use_gnn=True, gnn_config=cfg)
    # ...but they are never read when the current is not used.
    _small_model(use_gnn=True, gnn_config=dataclasses.replace(cfg, use_current=False))


def test_standalone_encoder_gnn_is_optional_and_off_by_default():
    cnn_cfg, vit_cfg, _ = _small_configs()
    assert CNNViTEncoder(cnn_cfg, vit_cfg).uses_gnn is False
    assert CNNViTEncoder(cnn_cfg, vit_cfg, GNNConfig(in_channels=CNN_OUT)).uses_gnn is True


# --------------------------------------------------------------------------
# GNN ENABLED — the GNN is really in the forward path
# --------------------------------------------------------------------------


def test_enabled_output_differs_from_disabled_with_identical_shared_weights():
    off = _small_model(use_gnn=False).eval()
    on = _small_model(use_gnn=True).eval()
    result = on.load_state_dict(off.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert set(result.missing_keys) == set(_gnn_keys(on)), "only the GNN weights should be missing"

    x = torch.randn(2, IN_CH, 12, 14)
    with torch.no_grad():
        assert (off(x) - on(x)).abs().max() > 1e-6, "GNN has no effect on the output"


def test_enabled_forward_still_equals_decoder_of_get_embedding():
    model = _small_model(use_gnn=True).eval()
    x = torch.randn(2, IN_CH, 12, 12)
    with torch.no_grad():
        assert torch.equal(model(x), model.decoder(model.get_embedding(x)))


def test_enabled_get_embedding_is_the_mean_of_the_encoder_tokens():
    model = _small_model(use_gnn=True).eval()
    x = torch.randn(2, IN_CH, 12, 12)
    with torch.no_grad():
        assert torch.allclose(
            model.encoder.get_embedding(x), model.encoder(x).mean(dim=1), atol=1e-6
        )


def test_enabled_default_config_end_to_end_on_the_real_neer_grid():
    model = NEERModel(use_gnn=True).eval()
    x = torch.randn(1, NEER_N_CHANNELS, 101, 241)
    with torch.no_grad():
        out = model(x)
        embedding = model.get_embedding(x)
        profile = model.predict_profile(x, torch.full((15,), 15.0))
    assert out.shape == (1, 15) and torch.isfinite(out).all()
    assert embedding.shape == (1, 256) and torch.isfinite(embedding).all()
    assert torch.allclose(profile, out + 15.0)
    assert model.depths == DEFAULT_DEPTHS


# --------------------------------------------------------------------------
# GNN ENABLED — the ocean current is really used in message passing
# --------------------------------------------------------------------------


def _isolated_from_current(model: NEERModel, current_channels=DEFAULT_CURRENT_CHANNELS) -> NEERModel:
    """Zero the CNN's first-layer weights on the current channels.

    After this the CNN output cannot depend on the current channels, so
    any dependence of the model output on them has to travel through the
    GNN. This isolates the GNN's use of the current from the CNN's.
    """
    conv = model.encoder.cnn.blocks[0].conv
    with torch.no_grad():
        conv.weight[:, list(current_channels)] = 0.0
    return model


def _with_changed_current(x: torch.Tensor, current_channels=DEFAULT_CURRENT_CHANNELS) -> torch.Tensor:
    changed = x.clone()
    changed[:, list(current_channels)] = torch.randn_like(changed[:, list(current_channels)]) * 3 + 1
    return changed


def test_current_reaches_the_gnn_straight_from_the_input_channels():
    model = _small_model(use_gnn=True).eval()
    seen = {}
    model.encoder.gnn.register_forward_pre_hook(lambda _m, args: seen.update(current=args[1]))
    x = torch.randn(2, IN_CH, 9, 11)
    with torch.no_grad():
        model(x)
    u, v = DEFAULT_CURRENT_CHANNELS
    assert torch.equal(seen["current"], x[:, [u, v]])
    assert seen["current"].shape == (2, 2, 9, 11)


def test_custom_current_channels_are_the_ones_read():
    cfg = GNNConfig(in_channels=CNN_OUT, hidden_dim=8, current_channels=(7, 9))
    model = _small_model(use_gnn=True, gnn_config=cfg).eval()
    seen = {}
    model.encoder.gnn.register_forward_pre_hook(lambda _m, args: seen.update(current=args[1]))
    x = torch.randn(1, IN_CH, 8, 8)
    with torch.no_grad():
        model(x)
    assert torch.equal(seen["current"], x[:, [7, 9]])


def test_current_changes_the_output_only_through_gnn_messages_when_used():
    x = torch.randn(2, IN_CH, 10, 12)
    x_changed = _with_changed_current(x)

    # GNN enabled and using the current: changing ONLY the current
    # channels changes the output, and (CNN isolated) the only route is
    # the GNN's messages.
    used = _isolated_from_current(_small_model(use_gnn=True)).eval()
    with torch.no_grad():
        assert (used(x) - used(x_changed)).abs().max() > 1e-6, "current is not incorporated"

    # GNN enabled but NOT using the current: the same change has no effect.
    not_used = _isolated_from_current(
        _small_model(use_gnn=True, gnn_config=GNNConfig(in_channels=CNN_OUT, hidden_dim=8, use_current=False))
    ).eval()
    with torch.no_grad():
        assert torch.equal(not_used(x), not_used(x_changed))

    # GNN disabled: nothing else reads the current here either.
    disabled = _isolated_from_current(_small_model(use_gnn=False)).eval()
    with torch.no_grad():
        assert torch.equal(disabled(x), disabled(x_changed))


def test_current_gets_a_gradient_through_the_gnn_edge_weights():
    # The columns of the edge projection that consume (along, across) must
    # actually be trained; if the current were decoration they would not.
    model = _small_model(use_gnn=True)
    model(torch.randn(2, IN_CH, 8, 10)).sum().backward()
    edge_w = model.encoder.gnn.layers[0].edge_proj.weight
    assert edge_w.shape[1] == GEOMETRY_EDGE_DIM + CURRENT_EDGE_DIM
    assert edge_w.grad[:, GEOMETRY_EDGE_DIM:].abs().sum() > 0


def test_gnn_with_current_requires_the_current_field():
    gnn = GridGNN(GNNConfig(in_channels=8))
    with pytest.raises(ValueError, match="use_current=True"):
        gnn(torch.randn(1, 8, 4, 4))
    with pytest.raises(ValueError, match="shape"):
        gnn(torch.randn(1, 8, 4, 4), torch.randn(1, 2, 4, 5))


def test_gnn_without_current_refuses_one_instead_of_ignoring_it():
    gnn = GridGNN(GNNConfig(in_channels=8, use_current=False))
    features = torch.randn(1, 8, 4, 4)
    assert gnn(features).shape == features.shape
    with pytest.raises(ValueError, match="use_current=False"):
        gnn(features, torch.randn(1, 2, 4, 4))


def test_current_edge_features_on_a_hand_computed_2x2_grid():
    # 2x2 grid, nodes 0=(r0,c0) 1=(r0,c1) 2=(r1,c0) 3=(r1,c1).
    # A uniform eastward current u=1, v=0 everywhere.
    src_np, dst_np = build_grid_edges(2, 2, 4)
    direction = torch.from_numpy(edge_directions(src_np, dst_np, 2))
    src, dst = torch.from_numpy(src_np), torch.from_numpy(dst_np)
    current = torch.zeros(1, 4, 2)
    current[..., 0] = 1.0
    feats = current_edge_features(current, src, dst, direction)[0]

    for k, (s, d) in enumerate(zip(src_np.tolist(), dst_np.tolist())):
        if (s, d) == (0, 1):  # east edge: flow runs along it
            assert feats[k].tolist() == pytest.approx([1.0, 0.0])
        elif (s, d) == (1, 0):  # west edge: flow runs against it
            assert feats[k].tolist() == pytest.approx([-1.0, 0.0])
        elif (s, d) == (0, 2):  # north edge: flow is perpendicular (rot90(d) = west)
            assert feats[k].tolist() == pytest.approx([0.0, -1.0])
        elif (s, d) == (2, 0):  # south edge
            assert feats[k].tolist() == pytest.approx([0.0, 1.0])


def test_current_edge_features_preserve_magnitude_and_flip_on_reversed_edges():
    h, w = 4, 5
    src_np, dst_np = build_grid_edges(h, w, 8)
    direction = torch.from_numpy(edge_directions(src_np, dst_np, w))
    src, dst = torch.from_numpy(src_np), torch.from_numpy(dst_np)
    current = torch.randn(3, h * w, 2)
    feats = current_edge_features(current, src, dst, direction)  # (3, E, 2)

    mean_current = 0.5 * (current[:, src] + current[:, dst])
    assert torch.allclose(
        feats.pow(2).sum(-1), mean_current.pow(2).sum(-1), atol=1e-5
    ), "along/across must be a rotation of the edge-mean current"

    reverse_of = {(s, d): k for k, (s, d) in enumerate(zip(src_np.tolist(), dst_np.tolist()))}
    reverse = torch.tensor([reverse_of[(d, s)] for s, d in zip(src_np.tolist(), dst_np.tolist())])
    assert torch.allclose(feats[:, reverse], -feats, atol=1e-6)


# --------------------------------------------------------------------------
# GNN ENABLED — messages travel only along grid edges
# --------------------------------------------------------------------------


@pytest.mark.parametrize("use_current", [False, True])
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_a_perturbation_spreads_exactly_one_edge_per_layer(num_layers, connectivity, use_current):
    torch.manual_seed(1)
    cfg = GNNConfig(
        in_channels=8, hidden_dim=8, num_layers=num_layers, connectivity=connectivity, use_current=use_current
    )
    gnn = GridGNN(cfg).eval()
    h, w, cell = 9, 11, (4, 5)
    features = torch.randn(1, 8, h, w)
    current = torch.randn(1, 2, h, w) if use_current else None

    # A random (not constant) per-channel change: the per-node LayerNorm is
    # invariant to adding one constant to every channel of a node.
    perturbed = features.clone()
    perturbed[0, :, cell[0], cell[1]] += torch.randn(8)
    with torch.no_grad():
        delta = (gnn(perturbed, current) - gnn(features, current))[0].abs().amax(dim=0)  # (h, w)

    rows = torch.arange(h)[:, None].expand(h, w)
    cols = torch.arange(w)[None, :].expand(h, w)
    d_row, d_col = (rows - cell[0]).abs(), (cols - cell[1]).abs()
    graph_distance = d_row + d_col if connectivity == 4 else torch.maximum(d_row, d_col)

    assert torch.all(delta[graph_distance <= num_layers] > 1e-9), "message did not reach a neighbour"
    assert torch.all(delta[graph_distance > num_layers] == 0), "information travelled off the graph"


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (6, 1), (2, 3)])
def test_gnn_handles_degenerate_grids(shape):
    gnn = GridGNN(GNNConfig(in_channels=8, hidden_dim=4, connectivity=8))
    features = torch.randn(2, 8, *shape)
    out = gnn(features, torch.randn(2, 2, *shape))
    assert out.shape == features.shape and torch.isfinite(out).all()


def test_gnn_preserves_shape_and_is_a_residual_refinement():
    gnn = GridGNN(GNNConfig(in_channels=8, hidden_dim=4)).eval()
    features = torch.randn(2, 8, 6, 7)
    with torch.no_grad():
        out = gnn(features, torch.randn(2, 2, 6, 7))
    assert out.shape == features.shape
    assert not torch.equal(out, features)
    assert torch.isfinite(out).all()


def test_gnn_validates_its_input():
    gnn = GridGNN(GNNConfig(in_channels=8, use_current=False))
    with pytest.raises(ValueError, match="4D"):
        gnn(torch.randn(8, 4, 4))
    with pytest.raises(ValueError, match="channels"):
        gnn(torch.randn(1, 9, 4, 4))


def test_a_first_call_under_inference_mode_does_not_break_later_training():
    # The graph tensors are cached across calls; ones built inside
    # inference mode could not be saved for backward afterwards.
    gnn_module._graph_tensors.cache_clear()
    gnn = GridGNN(GNNConfig(in_channels=8, hidden_dim=4, use_current=False))
    features = torch.randn(1, 8, 5, 6, requires_grad=True)
    with torch.inference_mode():
        gnn(features.detach())
    gnn(features).sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_gnn_supports_double_precision():
    model = _small_model(use_gnn=True).double().eval()
    x = torch.randn(2, IN_CH, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        out = model(x)
    assert out.dtype == torch.float64 and torch.isfinite(out).all()


# --------------------------------------------------------------------------
# BOTH CONFIGURATIONS — the same NEER contract holds with the GNN off and on
# --------------------------------------------------------------------------


@pytest.fixture(params=[False, True], ids=["gnn_disabled", "gnn_enabled"])
def use_gnn(request) -> bool:
    return request.param


def test_both_configurations_use_gnn_flag_is_reported(use_gnn):
    assert _small_model(use_gnn).use_gnn is use_gnn


def test_both_configurations_forward_and_embedding_shapes(use_gnn):
    model = _small_model(use_gnn).eval()
    x = torch.randn(3, IN_CH, 12, 14)
    with torch.no_grad():
        out, embedding = model(x), model.get_embedding(x)
    assert out.shape == (3, len(DEPTHS)) and torch.isfinite(out).all()
    assert embedding.shape == (3, EMBED) and torch.isfinite(embedding).all()
    assert model.depths == DEPTHS


def test_both_configurations_predict_profile_adds_the_climatology(use_gnn):
    model = _small_model(use_gnn).eval()
    x = torch.randn(2, IN_CH, 12, 12)
    climatology = torch.tensor([10.0, 8.0, 4.0, 2.0])
    with torch.no_grad():
        assert torch.allclose(model.predict_profile(x, climatology), model(x) + climatology)


def test_both_configurations_are_deterministic_in_eval_mode(use_gnn):
    model = _small_model(use_gnn).eval()
    x = torch.randn(2, IN_CH, 12, 12)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_both_configurations_batch_samples_are_independent(use_gnn):
    model = _small_model(use_gnn).eval()
    x = torch.randn(4, IN_CH, 10, 14)
    with torch.no_grad():
        batched = model(x)
        single = torch.cat([model(x[i : i + 1]) for i in range(4)], dim=0)
    assert torch.allclose(batched, single, atol=1e-4)


@pytest.mark.parametrize("shape", [(9, 13), (16, 16), (5, 7)])
def test_both_configurations_accept_grids_that_do_not_divide_the_patch_size(use_gnn, shape):
    out = _small_model(use_gnn).eval()(torch.randn(2, IN_CH, *shape))
    assert out.shape == (2, len(DEPTHS))


def test_both_configurations_gradients_reach_every_parameter(use_gnn):
    model = _small_model(use_gnn)
    x = torch.randn(2, IN_CH, 12, 12, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
    if use_gnn:
        assert any("gnn" in n for n, _ in model.named_parameters())


def test_both_configurations_take_a_training_step_and_update_their_weights(use_gnn):
    model = _small_model(use_gnn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(4, IN_CH, 10, 10)
    target = torch.randn(4, len(DEPTHS))
    before = {k: v.clone() for k, v in model.state_dict().items()}

    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed, "no parameter changed"
    if use_gnn:
        assert any(".gnn." in f".{k}" for k in changed), "the GNN was not trained"


def test_both_configurations_train_mode_dropout_paths_run(use_gnn):
    cnn_cfg, vit_cfg, decoder_cfg = _small_configs()
    gnn_cfg = GNNConfig(in_channels=CNN_OUT, hidden_dim=8, dropout=0.2) if use_gnn else None
    model = NEERModel(
        dataclasses.replace(cnn_cfg, dropout=0.1),
        dataclasses.replace(vit_cfg, dropout=0.1),
        decoder_cfg,
        use_gnn=use_gnn,
        gnn_config=gnn_cfg,
    ).train()
    assert torch.isfinite(model(torch.randn(2, IN_CH, 8, 8))).all()


def test_both_configurations_work_on_the_neer_input_contract(use_gnn):
    # Default (non-small) architecture on the real 11-channel NEER layout.
    model = NEERModel(use_gnn=use_gnn).eval()
    with torch.no_grad():
        out = model(torch.randn(1, NEER_N_CHANNELS, 40, 60))
    assert out.shape == (1, 15) and torch.isfinite(out).all()


# --------------------------------------------------------------------------
# Config -> model (`NEERModel.from_config`)
# --------------------------------------------------------------------------


def _with_model(config, **changes):
    return dataclasses.replace(config, model=dataclasses.replace(config.model, **changes))


def test_from_config_with_shipped_config_is_identical_to_the_default_model():
    config = load_config("demo")
    assert config.model.use_gnn is False
    from_config, default = NEERModel.from_config(config), NEERModel()
    assert from_config.use_gnn is False
    assert {k: v.shape for k, v in from_config.state_dict().items()} == {
        k: v.shape for k, v in default.state_dict().items()
    }
    assert from_config.depths == DEFAULT_DEPTHS


def test_from_config_builds_the_gnn_when_the_config_enables_it():
    config = _with_model(load_config("demo"), use_gnn=True)
    model = NEERModel.from_config(config)
    assert model.use_gnn is True and isinstance(model.encoder.gnn, GridGNN)
    assert model(torch.randn(1, NEER_N_CHANNELS, 24, 32)).shape == (1, 15)


def test_from_config_honours_the_environment_variable(monkeypatch):
    monkeypatch.setenv("NEER_MODEL_USE_GNN", "true")
    assert NEERModel.from_config(load_config("base")).use_gnn is True
    monkeypatch.setenv("NEER_MODEL_USE_GNN", "false")
    assert NEERModel.from_config(load_config("base")).use_gnn is False


def test_from_config_reads_embedding_dim_and_depths_from_the_config():
    config = dataclasses.replace(
        _with_model(load_config("demo"), embedding_dim=128), depths=[0.0, 10.0, 100.0]
    )
    model = NEERModel.from_config(config)
    assert model.embed_dim == 128
    assert model.depths == (0.0, 10.0, 100.0)
    assert model(torch.randn(1, NEER_N_CHANNELS, 16, 16)).shape == (1, 3)


def test_from_config_accepts_gnn_hyperparameters_only_when_enabled():
    tuned = GNNConfig(hidden_dim=16, num_layers=2, connectivity=8)
    enabled = NEERModel.from_config(_with_model(load_config("demo"), use_gnn=True), gnn_config=tuned)
    assert enabled.encoder.gnn.config == tuned
    with pytest.raises(ValueError, match="use_gnn"):
        NEERModel.from_config(load_config("demo"), gnn_config=tuned)


# --------------------------------------------------------------------------
# Pretrained encoder weights still load with the GNN on or off
# --------------------------------------------------------------------------


def _pretrained_checkpoint(tmp_path):
    cnn_cfg, vit_cfg, _ = _small_configs()
    torch.manual_seed(123)
    encoder = PretrainEncoder(
        PretrainEncoderConfig(cnn_config=cnn_cfg, vit_config=vit_cfg, embedding_dim=EMBED)
    )
    path = save_pretrained_encoder_checkpoint(tmp_path / "encoder.pt", encoder)
    return path, encoder


def test_both_configurations_load_a_pretrained_encoder_strictly(use_gnn, tmp_path):
    path, pretrained = _pretrained_checkpoint(tmp_path)
    model = _small_model(use_gnn, seed=7)
    gnn_before = {k: v.clone() for k, v in model.state_dict().items() if ".gnn." in f".{k}"}

    model.load_pretrained_encoder(path, strict=True)

    for name, tensor in pretrained.cnn.state_dict().items():
        assert torch.equal(model.encoder.cnn.state_dict()[name], tensor), f"cnn.{name}"
    for name, tensor in pretrained.vit.state_dict().items():
        assert torch.equal(model.encoder.vit.state_dict()[name], tensor), f"vit.{name}"
    # The GNN has no pretrained weights: it must be left exactly as built.
    gnn_after = {k: v for k, v in model.state_dict().items() if ".gnn." in f".{k}"}
    assert gnn_after.keys() == gnn_before.keys()
    assert all(torch.equal(gnn_after[k], gnn_before[k]) for k in gnn_before)
    assert bool(gnn_before) is use_gnn