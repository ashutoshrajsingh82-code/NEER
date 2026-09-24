"""
Phase 22 — optional graph refinement over the ocean grid.

An *optional* message-passing stage that refines the CNN's per-cell
feature map before the Vision Transformer patch-tokenizes it:

    surface fields   (batch, NEER_N_CHANNELS, lat, lon)
    -> CNNEncoder                                      (Phase 13)
    -> GridGNN   (only when use_gnn is true)           (this module)
    -> VisionTransformer                               (Phase 14)

It is wired in by `CNNViTEncoder` / `NEERModel` and switched by
`model.use_gnn` in `configs/base.yaml` (default `false`). When it is off,
no GNN module is constructed at all — the model's parameters, state_dict
keys and forward computation are exactly what they were before this
phase.

The graph
---------
- **Nodes are grid cells.** One node per `(lat, lon)` cell, numbered
  row-major (`node = row * W + col`), the same order `x.flatten(2)` uses,
  so a `(B, C, H, W)` feature map becomes `(B, H*W, C)` nodes with no
  reindexing.
- **Edges join neighbouring cells.** 4-connectivity (N/S/E/W) by default,
  8-connectivity (adding diagonals) optionally. Every neighbour pair is
  two *directed* edges, one each way, so information moves both ways and
  each edge carries its own direction.
- The graph depends only on `(H, W, connectivity)`, so it is built once
  per grid size and cached (see `build_grid_graph`).

Why place it after the CNN and not after the ViT
------------------------------------------------
After the ViT the representation is a grid of 8x8 *patch* tokens, not
grid cells. The CNN output is the last place every cell still has its own
feature vector, so that is where "grid cells act as nodes" is literally
true. It is a residual refinement (`h <- h + f(h)`), so the ViT still
receives a same-shaped feature map and every downstream contract
(`grid_shape`, `tokens_to_map`, the pooled embedding) is unchanged.

Ocean current as message-passing context
----------------------------------------
With `use_current=True` (default), the model's own `u_current` /
`v_current` *input* channels (eastward / northward, NEER_CHANNEL_ORDER
positions 3 and 4) condition every message. For the directed edge
`j -> i`, with unit direction `d` (east, north) and edge-mean current
`c = (c_j + c_i) / 2`, two scalars are computed:

    along  = c . d            flow component carrying j toward i
    across = c . rot90(d)     flow component perpendicular to the edge

and fed, with the static edge direction, into the edge term of the
message: `m_ji = gelu(A h_j + B h_i + E [d, along, across])`. So the
current changes what each cell hears from each neighbour, direction by
direction — it is an input to the messages themselves, not a decoration
next to them. It is a *learned* function of these features on purpose:
currents are z-scored upstream (`Normalizer`), so "positive along" is not
guaranteed to mean "physically downstream", and a hard-coded upwind rule
would bake in a sign convention the data no longer has.

With `use_current=False` the current channels are not read at all, and
`GridGNN.forward` refuses a `current` argument rather than silently
ignoring it.

Limitations (deliberate)
------------------------
- The model input has no land mask channel, so edges join land and ocean
  cells alike; the CNN and this layer both see whatever the (filled)
  tensors contain.
- Cell spacing is treated as uniform: a 0.25 degree step east-west is
  physically shorter than north-south away from the equator, and the
  edge geometry does not correct for it.

torch is optional
------------------
Same lazy pattern as `encoder.py` / `vit.py`: `GNNConfig`,
`build_grid_edges` and `build_grid_graph` are numpy-only and import
without torch. Only instantiating a `GridGNN` raises
`MissingDependencyError` when torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing.channels import NEER_CHANNEL_ORDER
from src.models.encoder import DEFAULT_CHANNELS

try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="the NEER grid GNN (src.models.gnn)",
            install_hint="pip install torch",
        )


#: Neighbourhoods a grid graph may use: 4 = N/S/E/W, 8 = plus diagonals.
VALID_CONNECTIVITIES: Tuple[int, ...] = (4, 8)

#: Positions of `u_current` / `v_current` in the model input, taken from
#: the one authoritative channel order rather than hard-coded.
DEFAULT_CURRENT_CHANNELS: Tuple[int, int] = (
    NEER_CHANNEL_ORDER.index("u_current"),
    NEER_CHANNEL_ORDER.index("v_current"),
)

#: Static per-edge features: the unit direction `(east, north)`.
GEOMETRY_EDGE_DIM = 2

#: Current-derived per-edge features: `(along, across)`.
CURRENT_EDGE_DIM = 2


@dataclass(frozen=True)
class GNNConfig:
    """Configuration for `GridGNN`.

    Parameters
    ----------
    in_channels:
        Node feature width — the channel count of the feature map being
        refined. Defaults to the last entry of `DEFAULT_CHANNELS` (the
        default `CNNEncoder`'s output width), so `GNNConfig()` plugs onto
        `CNNEncoder()`. Must equal the CNN's `out_channels`.
    hidden_dim:
        Width of each message. Kept small (32) because messages are
        computed per *edge* (~4 per cell), which dominates the cost.
    num_layers:
        Rounds of message passing. Each round lets information travel
        one edge further, so `num_layers` is the receptive radius in
        cells under 4/8-connectivity.
    connectivity:
        One of `VALID_CONNECTIVITIES`.
    use_current:
        Condition messages on the ocean current (see the module
        docstring). If true, `current_channels` must exist in the model
        input.
    current_channels:
        Indices of `(u_current, v_current)` in the *model input* tensor
        (not in the CNN feature map). Defaults to their positions in
        `NEER_CHANNEL_ORDER`.
    dropout:
        Dropout on each layer's residual update. `0.0` disables it.
    """

    in_channels: int = DEFAULT_CHANNELS[-1]
    hidden_dim: int = 32
    num_layers: int = 1
    connectivity: int = 4
    use_current: bool = True
    current_channels: Tuple[int, int] = DEFAULT_CURRENT_CHANNELS
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")
        if self.connectivity not in VALID_CONNECTIVITIES:
            raise ValueError(
                f"connectivity must be one of {VALID_CONNECTIVITIES}, got {self.connectivity!r}"
            )
        channels = tuple(self.current_channels)
        if len(channels) != 2 or not all(isinstance(c, int) and not isinstance(c, bool) for c in channels):
            raise ValueError(
                f"current_channels must be a pair of ints (u_current, v_current), got {self.current_channels!r}"
            )
        if any(c < 0 for c in channels):
            raise ValueError(f"current_channels must be non-negative, got {channels}")
        if channels[0] == channels[1]:
            raise ValueError(f"current_channels must be two distinct channels, got {channels}")
        object.__setattr__(self, "current_channels", channels)
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

    @property
    def edge_dim(self) -> int:
        """Width of the per-edge feature vector fed to each message."""
        return GEOMETRY_EDGE_DIM + (CURRENT_EDGE_DIM if self.use_current else 0)


# --------------------------------------------------------------------------
# Graph construction (numpy only — no torch needed)
# --------------------------------------------------------------------------


def build_grid_edges(
    height: int, width: int, connectivity: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Directed edges between neighbouring cells of an `height x width` grid.

    Nodes are numbered row-major, `node = row * width + col`. Every
    neighbour pair appears twice, once per direction.

    Returns
    -------
    `(src, dst)`, two int64 arrays of length `num_edges`: edge `k` sends a
    message from node `src[k]` to node `dst[k]`. Number of edges is
    `2 * (H*(W-1) + (H-1)*W)` for 4-connectivity, plus
    `4 * (H-1)*(W-1)` for 8.
    """
    if height < 1 or width < 1:
        raise ValueError(f"grid must be at least 1x1, got {height}x{width}")
    if connectivity not in VALID_CONNECTIVITIES:
        raise ValueError(f"connectivity must be one of {VALID_CONNECTIVITIES}, got {connectivity!r}")

    idx = np.arange(height * width, dtype=np.int64).reshape(height, width)
    pairs = [
        (idx[:, :-1], idx[:, 1:]),  # east neighbour
        (idx[:-1, :], idx[1:, :]),  # north neighbour
    ]
    if connectivity == 8:
        pairs += [
            (idx[:-1, :-1], idx[1:, 1:]),  # north-east neighbour
            (idx[:-1, 1:], idx[1:, :-1]),  # north-west neighbour
        ]
    a = np.concatenate([p[0].ravel() for p in pairs])
    b = np.concatenate([p[1].ravel() for p in pairs])
    return np.concatenate([a, b]), np.concatenate([b, a])


def edge_directions(src: np.ndarray, dst: np.ndarray, width: int) -> np.ndarray:
    """Unit direction `(east, north)` of each edge `src -> dst`, `(E, 2)` float32.

    Column index grows eastward and row index grows northward on the NEER
    grid (both axes are ascending — see `src.data.grid`), so east is
    `+col` and north is `+row`.
    """
    d_east = (dst % width) - (src % width)
    d_north = (dst // width) - (src // width)
    direction = np.stack([d_east, d_north], axis=1).astype(np.float64)
    if direction.shape[0]:
        direction /= np.hypot(direction[:, 0], direction[:, 1])[:, None]
    return direction.astype(np.float32)


@dataclass(frozen=True, eq=False)
class GridGraph:
    """The cell-adjacency graph of an `height x width` grid.

    `src` / `dst` are the directed edges (see `build_grid_edges`),
    `direction` the `(E, 2)` unit `(east, north)` of each, and
    `in_degree` how many edges arrive at each node (4 in the interior of
    a 4-connected grid, fewer on the boundary).
    """

    height: int
    width: int
    connectivity: int
    src: np.ndarray
    dst: np.ndarray
    direction: np.ndarray
    in_degree: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.height * self.width

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])


@lru_cache(maxsize=16)
def build_grid_graph(height: int, width: int, connectivity: int = 4) -> GridGraph:
    """Build (and cache) the `GridGraph` for a grid size.

    The arrays are marked read-only because the result is shared between
    callers through the cache.
    """
    src, dst = build_grid_edges(height, width, connectivity)
    direction = edge_directions(src, dst, width)
    in_degree = np.bincount(dst, minlength=height * width).astype(np.int64)
    for array in (src, dst, direction, in_degree):
        array.setflags(write=False)
    return GridGraph(height, width, connectivity, src, dst, direction, in_degree)


if _TORCH_AVAILABLE:

    @lru_cache(maxsize=16)
    def _graph_tensors(
        height: int, width: int, connectivity: int, device: "torch.device"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """`(src, dst, direction, inv_degree)` tensors for a graph on `device`.

        Built with inference mode explicitly off: tensors created inside
        `torch.inference_mode()` cannot later be saved for backward, and
        this cache outlives any one call — a first call under inference
        mode must not poison later training calls on the same grid size.
        """
        graph = build_grid_graph(height, width, connectivity)
        with torch.inference_mode(False):
            src = torch.from_numpy(graph.src.copy()).to(device)
            dst = torch.from_numpy(graph.dst.copy()).to(device)
            direction = torch.from_numpy(graph.direction.copy()).to(device)
            inv_degree = torch.from_numpy(
                (1.0 / np.maximum(graph.in_degree, 1)).astype(np.float32)
            ).to(device)
        return src, dst, direction, inv_degree

    def current_edge_features(
        current_nodes: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        direction: "torch.Tensor",
    ) -> "torch.Tensor":
        """Flow projected onto each edge: `(B, E, 2)` of `(along, across)`.

        Parameters
        ----------
        current_nodes:
            `(B, N, 2)` per-cell `(u_current, v_current)` (east, north).
        src, dst:
            `(E,)` edge endpoints.
        direction:
            `(E, 2)` unit `(east, north)` of each edge `src -> dst`.

        `along` is the current's component along the edge (the flow
        carrying `src` toward `dst`) and `across` the component along the
        edge rotated 90 degrees counter-clockwise. Together they preserve
        the edge-mean current's magnitude:
        `along**2 + across**2 == |mean current|**2`.
        """
        edge_current = 0.5 * (current_nodes.index_select(1, src) + current_nodes.index_select(1, dst))
        u, v = edge_current[..., 0], edge_current[..., 1]
        d_east, d_north = direction[:, 0], direction[:, 1]
        along = u * d_east + v * d_north
        across = -u * d_north + v * d_east
        return torch.stack([along, across], dim=-1)

    class GridGNNLayer(nn.Module):
        """One round of message passing over the grid graph.

        For every directed edge `j -> i` a message
        `m_ji = gelu(A n_j + B n_i + E e_ji)` is computed, where
        `n = LayerNorm(h)` are the node features and `e_ji` the edge
        features (direction, plus current-derived features when enabled).
        Each node averages the messages it receives (mean rather than sum
        so boundary cells, which have fewer neighbours, are on the same
        scale as interior cells) and adds the projected result back:
        `h_i <- h_i + Linear(mean_j m_ji)`.

        `A n_j` and `B n_i` are computed once per *node* and gathered onto
        edges, rather than running a matmul over a concatenated
        `[n_j, n_i]` per edge — the same function at roughly a quarter of
        the FLOPs.
        """

        def __init__(self, config: GNNConfig) -> None:
            super().__init__()
            c, h = config.in_channels, config.hidden_dim
            self.norm = nn.LayerNorm(c)
            self.src_proj = nn.Linear(c, h)
            self.dst_proj = nn.Linear(c, h, bias=False)
            self.edge_proj = nn.Linear(config.edge_dim, h, bias=False)
            self.out_proj = nn.Linear(h, c)
            self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        def forward(
            self,
            nodes: "torch.Tensor",
            src: "torch.Tensor",
            dst: "torch.Tensor",
            edge_attr: "torch.Tensor",
            inv_degree: "torch.Tensor",
        ) -> "torch.Tensor":
            normed = self.norm(nodes)  # (B, N, C)
            from_src = self.src_proj(normed).index_select(1, src)  # (B, E, H)
            from_dst = self.dst_proj(normed).index_select(1, dst)  # (B, E, H)
            messages = F.gelu(from_src + from_dst + self.edge_proj(edge_attr))  # (B, E, H)

            received = messages.new_zeros(nodes.shape[0], nodes.shape[1], messages.shape[-1])
            received = received.index_add(1, dst, messages)  # (B, N, H) sum over incoming edges
            received = received * inv_degree.to(received.dtype)[None, :, None]  # -> mean
            return nodes + self.dropout(self.out_proj(received))

    class GridGNN(nn.Module):
        """Optional GNN refinement of a `(B, C, H, W)` feature map.

        Treats every grid cell as a node and every pair of neighbouring
        cells as (two directed) edges, runs `config.num_layers` rounds of
        message passing, and returns a feature map of the *same shape* —
        so it can be dropped between the CNN and the ViT without
        changing either.

        If `config.use_current` is true, `forward` requires the
        `(B, 2, H, W)` `(u_current, v_current)` field, which conditions
        the messages (see the module docstring). If false, passing
        `current` raises.

        Examples
        --------
        >>> gnn = GridGNN(GNNConfig(in_channels=64))
        >>> features = torch.randn(2, 64, 20, 30)
        >>> current = torch.randn(2, 2, 20, 30)
        >>> gnn(features, current).shape
        torch.Size([2, 64, 20, 30])
        """

        def __init__(self, config: Optional[GNNConfig] = None) -> None:
            _require_torch()
            super().__init__()
            self.config = config or GNNConfig()
            self.layers = nn.ModuleList([GridGNNLayer(self.config) for _ in range(self.config.num_layers)])

        @property
        def in_channels(self) -> int:
            return self.config.in_channels

        @property
        def uses_current(self) -> bool:
            return self.config.use_current

        def forward(
            self, features: "torch.Tensor", current: Optional["torch.Tensor"] = None
        ) -> "torch.Tensor":
            """
            Parameters
            ----------
            features:
                `(batch, config.in_channels, H, W)` per-cell features.
            current:
                `(batch, 2, H, W)` `(u_current, v_current)`; required when
                `config.use_current`, and must be omitted otherwise.

            Returns
            -------
            `(batch, config.in_channels, H, W)` refined features.
            """
            if features.dim() != 4:
                raise ValueError(
                    f"GridGNN expects a 4D (batch, channels, H, W) tensor, got shape {tuple(features.shape)}"
                )
            b, c, h, w = features.shape
            if c != self.config.in_channels:
                raise ValueError(
                    f"GridGNN configured for {self.config.in_channels} channels, "
                    f"got {c} in input of shape {tuple(features.shape)}"
                )
            if self.config.use_current:
                if current is None:
                    raise ValueError(
                        "GridGNN was built with use_current=True and needs the (batch, 2, H, W) "
                        "current field; pass `current`"
                    )
                if tuple(current.shape) != (b, 2, h, w):
                    raise ValueError(
                        f"current must have shape {(b, 2, h, w)} to match features, got {tuple(current.shape)}"
                    )
            elif current is not None:
                raise ValueError(
                    "GridGNN was built with use_current=False, so it would ignore `current`; "
                    "omit it (or build the GNN with use_current=True)"
                )

            src, dst, direction, inv_degree = _graph_tensors(h, w, self.config.connectivity, features.device)
            direction = direction.to(features.dtype)

            nodes = features.flatten(2).transpose(1, 2)  # (B, H*W, C), row-major nodes
            edge_attr = direction.unsqueeze(0).expand(b, -1, -1)  # (B, E, 2) static geometry
            if self.config.use_current:
                current_nodes = current.flatten(2).transpose(1, 2).to(features.dtype)  # (B, N, 2)
                edge_attr = torch.cat(
                    [edge_attr, current_edge_features(current_nodes, src, dst, direction)], dim=-1
                )  # (B, E, 4)

            for layer in self.layers:
                nodes = layer(nodes, src, dst, edge_attr, inv_degree)
            return nodes.transpose(1, 2).reshape(b, c, h, w)

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            c = self.config
            return (
                f"GridGNN(in_channels={c.in_channels}, hidden_dim={c.hidden_dim}, "
                f"num_layers={c.num_layers}, connectivity={c.connectivity}, use_current={c.use_current})"
            )

else:  # pragma: no cover - exercised only in a torch-less environment

    class GridGNN:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed (see `encoder.CNNEncoder`)."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()