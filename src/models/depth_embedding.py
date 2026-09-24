"""
Phase 16 — learned embeddings for the 15 NEER target depths.

NEER's subsurface targets sit at a fixed, small set of depth levels —
`DEFAULT_DEPTHS` below, matching `depths:` in `configs/base.yaml` and
`NEER_N_DEPTHS` (15) in `src.data.dataset`. A future depth decoder needs
some way to tell the model *which* depth it is predicting for a given
spatial embedding (from `CNNViTEncoder.get_embedding`, Phase 15).

The wrong way to do that would be training 15 separate small networks,
one per depth — 15x the parameters, no sharing of what the model learns
about "depth" in general, and it doesn't generalize to a depth grid that
changes between runs. `DepthEmbedding` instead does what NLP token
embeddings do for a fixed vocabulary: **one** shared lookup table,
`nn.Embedding(15, embed_dim)`, a single `(15, embed_dim)` weight matrix.
Every depth level is a row in that one table; there is exactly one set
of learned weights, not fifteen. `forward` (index lookup) and
`forward_from_depths` (physical depth value lookup, e.g. metres) both
go through that same table — nothing about a depth's row is looked up,
computed, or gated by a separate module.

Rows can still specialize during training (gradients only reach the
rows actually looked up in a given batch — see
`test_gradients_flow_only_to_looked_up_depth_rows` in
`tests/test_depth_embedding.py`), which is the point of a lookup table
over, say, a single dense encoding of the depth value: nearby depths
aren't forced to share a row, but nothing about the *architecture*
duplicates parameters or logic per depth.

A downstream decoder is expected to combine one row of this table
(shape `(embed_dim,)` or `(embed_dim,)` per depth in a batch) with the
spatial embedding from `CNNViTEncoder.get_embedding` (shape
`(batch, embed_dim)`) — by addition or concatenation — to predict a
value at that depth. `DEFAULT_DEPTH_EMBED_DIM` therefore matches
`CNNViTEncoder`'s default `embed_dim` (256, from `configs/base.yaml`'s
`model.embedding_dim`) so the two combine without a projection layer,
though this is a default, not a requirement — a decoder using
concatenation is free to pick a smaller depth-embedding width.

torch is optional
------------------
Same lazy pattern as `encoder.py` / `vit.py`: importing this module and
building a `DepthEmbeddingConfig` never needs torch; only instantiating
a `DepthEmbedding` raises `MissingDependencyError` when it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union

from src.data.loaders.errors import MissingDependencyError

try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="the NEER depth embedding (src.models.depth_embedding)",
            install_hint="pip install torch",
        )


#: The 15 NEER target depth levels, in metres, shallowest first. The
#: authoritative list is `depths:` in `configs/base.yaml`; this is the
#: model-side copy every depth-aware model component defaults to, so
#: model code never has to hard-code the list or load YAML to get it.
DEFAULT_DEPTHS: Tuple[float, ...] = (
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

#: Default embedding width — matches `CNNViTEncoder`'s default
#: `embed_dim` (`configs/base.yaml`'s `model.embedding_dim`) so a
#: depth row and a spatial embedding combine without a projection layer.
DEFAULT_DEPTH_EMBED_DIM = 256


@dataclass(frozen=True)
class DepthEmbeddingConfig:
    """Configuration for `DepthEmbedding`.

    Parameters
    ----------
    depths:
        The target depth levels this table has one row for, in metres.
        Must be non-negative (positive down), strictly ascending, and
        unique — the same shape of requirement `validate_depths`
        enforces on `configs/*.yaml`'s `depths:` list
        (`src/utils/config.py`), though this dataclass doesn't read
        that file itself.
    embed_dim:
        Width of each depth's embedding row.
    """

    depths: Tuple[float, ...] = DEFAULT_DEPTHS
    embed_dim: int = DEFAULT_DEPTH_EMBED_DIM

    def __post_init__(self) -> None:
        depths = tuple(float(d) for d in self.depths)
        object.__setattr__(self, "depths", depths)

        if len(depths) == 0:
            raise ValueError("depths must be a non-empty sequence")
        if any(d < 0 for d in depths):
            raise ValueError(f"depths must be non-negative (positive down), got {depths}")
        if list(depths) != sorted(depths):
            raise ValueError(f"depths must be strictly ascending, got {depths}")
        if len(set(depths)) != len(depths):
            raise ValueError(f"depths must not contain duplicates, got {depths}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")

    @property
    def num_depths(self) -> int:
        return len(self.depths)

    @property
    def depth_to_index(self) -> Dict[float, int]:
        """`{depth_value: row_index}` for every configured depth."""
        return {depth: i for i, depth in enumerate(self.depths)}

    def index_for(self, depth: float) -> int:
        """Row index for one of this config's exact depth values.

        Raises `ValueError` (not `KeyError`) for anything that isn't
        one of `depths` — this is a lookup over a fixed, small set of
        target levels, not a nearest-depth search or interpolation.
        """
        idx = self.depth_to_index.get(float(depth))
        if idx is None:
            raise ValueError(f"depth {depth!r} is not one of the configured target depths {self.depths}")
        return idx


if _TORCH_AVAILABLE:

    class DepthEmbedding(nn.Module):
        """Shared learned embedding table for the configured target depths.

        One `nn.Embedding(config.num_depths, config.embed_dim)` — a
        single `(num_depths, embed_dim)` weight matrix — backs every
        depth level. See the module docstring for why this is one
        table rather than a per-depth network.

        Examples
        --------
        >>> depth_embed = DepthEmbedding()
        >>> depth_embed(torch.tensor([0, 14])).shape
        torch.Size([2, 256])
        >>> depth_embed.forward_from_depths(1000.0).shape
        torch.Size([256])
        >>> depth_embed.all_embeddings().shape
        torch.Size([15, 256])
        """

        def __init__(self, config: DepthEmbeddingConfig | None = None) -> None:
            _require_torch()
            super().__init__()
            self.config = config or DepthEmbeddingConfig()
            self.embedding = nn.Embedding(self.config.num_depths, self.config.embed_dim)
            nn.init.trunc_normal_(self.embedding.weight, std=0.02)

        @property
        def depths(self) -> Tuple[float, ...]:
            return self.config.depths

        @property
        def num_depths(self) -> int:
            return self.config.num_depths

        @property
        def embed_dim(self) -> int:
            return self.config.embed_dim

        def index_for(self, depth: float) -> int:
            return self.config.index_for(depth)

        def forward(self, depth_index: "torch.Tensor") -> "torch.Tensor":
            """Look up rows by integer index into `self.depths`.

            Parameters
            ----------
            depth_index:
                Long tensor of any shape, values in
                `[0, num_depths)` — index `i` means `self.depths[i]`.

            Returns
            -------
            `(*depth_index.shape, embed_dim)`.
            """
            if not torch.is_tensor(depth_index):
                depth_index = torch.as_tensor(depth_index)
            if depth_index.dtype != torch.int64:
                depth_index = depth_index.long()
            if depth_index.numel() > 0:
                lo = int(depth_index.min())
                hi = int(depth_index.max())
                if lo < 0 or hi >= self.config.num_depths:
                    raise ValueError(
                        f"depth index out of range [0, {self.config.num_depths}), "
                        f"got values spanning [{lo}, {hi}]"
                    )
            return self.embedding(depth_index)

        def forward_from_depths(
            self, depth_values: Union[float, int, Sequence[float], "torch.Tensor"]
        ) -> "torch.Tensor":
            """Look up rows by physical depth value (metres), e.g. `100.0`.

            Parameters
            ----------
            depth_values:
                A single depth value, or a 1D sequence/tensor of them.
                Each must equal one of `self.depths` exactly (this is a
                lookup over fixed target levels, not interpolation).

            Returns
            -------
            `(embed_dim,)` for a single depth value, `(N, embed_dim)`
            for a sequence of `N`.
            """
            if torch.is_tensor(depth_values) and depth_values.dim() == 0:
                depth_values = depth_values.item()
            if isinstance(depth_values, (int, float)):
                index = torch.tensor(self.index_for(depth_values), dtype=torch.long)
                return self.forward(index)

            values = depth_values.tolist() if torch.is_tensor(depth_values) else list(depth_values)
            device = depth_values.device if torch.is_tensor(depth_values) else self.embedding.weight.device
            indices = torch.tensor([self.index_for(v) for v in values], dtype=torch.long, device=device)
            return self.forward(indices)

        def all_embeddings(self) -> "torch.Tensor":
            """Every depth's row, in `self.depths` order: `(num_depths, embed_dim)`.

            Goes through the same `forward` lookup as everything else
            (`arange(num_depths)`), rather than reading `.weight`
            directly, so it can never drift from what a per-depth
            lookup would return.
            """
            index = torch.arange(self.config.num_depths, device=self.embedding.weight.device)
            return self.forward(index)

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return f"DepthEmbedding(num_depths={self.num_depths}, embed_dim={self.embed_dim})"

else:  # pragma: no cover - exercised only in a torch-less environment

    class DepthEmbedding:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed (see `encoder.CNNEncoder`)."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()