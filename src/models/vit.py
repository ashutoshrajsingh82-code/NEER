"""
Phase 14 — lightweight Vision Transformer.

Takes the dense spatial feature map the Phase 13 `CNNEncoder` produces,
`(batch, C_cnn, lat, lon)`, cuts it into non-overlapping patches, and
runs a small transformer over the patch tokens so every patch can attend
to every other one — the long-range interaction a stack of 3x3 convs
cannot provide. Output is a token sequence `(batch, n_tokens, embed_dim)`.

Design choices (all driven by "keep it CPU-friendly")
-----------------------------------------------------
- **Small defaults.** `embed_dim=256` (matching `model.embedding_dim` in
  `configs/base.yaml`), 8 heads, 4 layers, MLP ratio 2, patch 8x8. On the
  real 101 x 241 NEER grid that is 13 x 31 = 403 tokens and ~3M
  parameters. Attention cost is quadratic in the token count, so the
  patch size is the main speed knob: patch 4 would mean ~1,600 tokens.
- **Grid sizes that don't divide the patch size.** 101 and 241 are not
  multiples of any sensible patch size, so the input is padded on the
  bottom/right up to the next multiple (`replicate` by default, matching
  the physical-boundary reasoning in `encoder.py`; `zeros` also offered).
  Nothing is cropped, so no grid cell is ever dropped.
- **Fixed 2D sin-cos position embeddings** computed from the token grid
  on the fly, instead of a learned table. No parameters, and — unlike a
  learned `(1, N, D)` table — it works for any input size, so tests and
  future crops/tiles don't need a different model.
- **Pre-norm blocks** with `F.scaled_dot_product_attention`, which uses
  fused/tiled CPU kernels in torch>=2 rather than materializing
  attention matrices in Python.
- **No CLS token.** NEER's task is estimation/reconstruction over the
  grid, so every patch token is useful downstream; a global summary can
  be pooled from the tokens if a later phase wants one.

`CNNViTEncoder` is the composed CNN -> ViT model, and the intended entry
point: `(batch, NEER_N_CHANNELS, lat, lon) -> (batch, n_tokens,
embed_dim)`. `grid_shape` / `tokens_to_map` convert between the token
sequence and a `(batch, embed_dim, gh, gw)` map for a future decoder.

Phase 15 adds `get_embedding(x) -> (batch, embed_dim)` to both
`VisionTransformer` and `CNNViTEncoder`: a mean pool, over the token
dimension, of that same model's real `forward()` output — `(batch, 256)`
under every default config. There is no separate embedding path or
CLS token to fake this with; `get_embedding` literally is
`forward(x).mean(dim=1)`, so it is exactly the representation a later
depth decoder would be built on top of, and it moves with the model's
weights.

Phase 22 lets `CNNViTEncoder` optionally refine the CNN's per-cell
feature map with a grid-graph GNN (`src/models/gnn.py`) before the ViT
sees it. It is off unless a `GNNConfig` is passed; when off, no GNN
module exists and the encoder is byte-for-byte the Phase 14/15 one.

torch is optional
------------------
Same lazy pattern as `encoder.py` / `src/data/dataset.py`: importing this
module and building a `ViTConfig` never needs torch; only instantiating a
model raises `MissingDependencyError` when it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from src.data.loaders.errors import MissingDependencyError
from src.models.encoder import DEFAULT_CHANNELS, CNNEncoderConfig
from src.models.gnn import GNNConfig

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
            purpose="the NEER Vision Transformer (src.models.vit)",
            install_hint="pip install torch",
        )


#: How the feature map is padded up to a multiple of the patch size.
VALID_PAD_MODES: Tuple[str, ...] = ("replicate", "zeros")

#: Default patch size. 8x8 turns the 101 x 241 grid into 13 x 31 = 403
#: tokens — comfortably CPU-sized for full attention.
DEFAULT_PATCH_SIZE = 8

#: Default embedding dimension (matches `model.embedding_dim` in
#: `configs/base.yaml`).
DEFAULT_EMBED_DIM = 256


@dataclass(frozen=True)
class ViTConfig:
    """Configuration for `VisionTransformer`.

    Parameters
    ----------
    in_channels:
        Channels of the feature map fed in. Defaults to the last entry of
        `DEFAULT_CHANNELS` (the default `CNNEncoder`'s output width), so
        `ViTConfig()` plugs straight onto `CNNEncoder()`.
    patch_size:
        Patch edge length in cells: an int for square patches or a
        `(patch_h, patch_w)` pair.
    embed_dim:
        Token / model width. Must be divisible by `num_heads` and by 4
        (the 2D sin-cos position embedding uses a quarter of the width
        for each of sin/cos along each axis).
    num_heads:
        Attention heads per layer.
    depth:
        Number of transformer layers.
    mlp_ratio:
        Hidden width of each block's MLP as a multiple of `embed_dim`.
        2.0 rather than the usual 4.0, to keep parameters and FLOPs down.
    dropout:
        Dropout probability on the patch embeddings, attention output
        projection, and MLP. `0.0` disables it.
    attention_dropout:
        Dropout on the attention weights themselves (0.0 by default; the
        token count is small enough that regularizing here is rarely
        needed).
    pad_mode:
        One of `VALID_PAD_MODES`; how to pad H/W up to a multiple of the
        patch size.
    """

    in_channels: int = DEFAULT_CHANNELS[-1]
    patch_size: Union[int, Tuple[int, int]] = DEFAULT_PATCH_SIZE
    embed_dim: int = DEFAULT_EMBED_DIM
    num_heads: int = 8
    depth: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attention_dropout: float = 0.0
    pad_mode: str = "replicate"

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")

        patch = self.patch_hw  # raises on malformed patch_size
        if patch[0] <= 0 or patch[1] <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 4 (2D sin-cos position embedding), got {self.embed_dim}"
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.depth <= 0:
            raise ValueError(f"depth must be at least 1, got {self.depth}")
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(f"attention_dropout must be in [0.0, 1.0), got {self.attention_dropout}")
        if self.pad_mode not in VALID_PAD_MODES:
            raise ValueError(f"pad_mode must be one of {VALID_PAD_MODES}, got {self.pad_mode!r}")

    @property
    def patch_hw(self) -> Tuple[int, int]:
        """`patch_size` normalized to a `(patch_h, patch_w)` tuple."""
        p = self.patch_size
        if isinstance(p, int):
            return (p, p)
        if isinstance(p, Sequence) and len(p) == 2 and all(isinstance(v, int) for v in p):
            return (int(p[0]), int(p[1]))
        raise ValueError(f"patch_size must be an int or a (patch_h, patch_w) pair of ints, got {p!r}")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        return max(1, int(round(self.embed_dim * self.mlp_ratio)))


def grid_shape_for(height: int, width: int, patch_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Token grid `(gh, gw)` for an `(height, width)` input, padding included."""
    ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
    return (-(-height // ph), -(-width // pw))  # ceil division


if _TORCH_AVAILABLE:

    def sincos_2d_position_embedding(
        grid_h: int,
        grid_w: int,
        embed_dim: int,
        device: "torch.device | None" = None,
        dtype: "torch.dtype" = torch.float32,
    ) -> "torch.Tensor":
        """Fixed 2D sin-cos position embedding, `(grid_h * grid_w, embed_dim)`.

        Half the channels encode the row index and half the column index
        (each as sin/cos at `embed_dim // 4` geometric frequencies).
        Rows are in row-major order, matching how patch tokens are
        flattened.
        """
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
        quarter = embed_dim // 4
        exponent = torch.arange(quarter, dtype=torch.float32, device=device) / quarter
        omega = torch.pow(torch.tensor(10000.0, device=device), -exponent)  # (quarter,)

        ys = torch.arange(grid_h, dtype=torch.float32, device=device)
        xs = torch.arange(grid_w, dtype=torch.float32, device=device)
        ang_y = ys[:, None] * omega[None, :]  # (gh, quarter)
        ang_x = xs[:, None] * omega[None, :]  # (gw, quarter)
        emb_y = torch.cat([torch.sin(ang_y), torch.cos(ang_y)], dim=1)  # (gh, 2*quarter)
        emb_x = torch.cat([torch.sin(ang_x), torch.cos(ang_x)], dim=1)  # (gw, 2*quarter)

        emb_y = emb_y[:, None, :].expand(grid_h, grid_w, -1)
        emb_x = emb_x[None, :, :].expand(grid_h, grid_w, -1)
        pos = torch.cat([emb_y, emb_x], dim=-1).reshape(grid_h * grid_w, embed_dim)
        return pos.to(dtype)

    class MultiHeadSelfAttention(nn.Module):
        """Multi-head self-attention using fused scaled-dot-product attention."""

        def __init__(self, config: ViTConfig) -> None:
            super().__init__()
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.attention_dropout = config.attention_dropout
            self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
            self.proj = nn.Linear(config.embed_dim, config.embed_dim)
            self.proj_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            b, n, d = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (b, heads, n, head_dim)
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attention_dropout if self.training else 0.0
            )
            attn = attn.transpose(1, 2).reshape(b, n, d)
            return self.proj_dropout(self.proj(attn))

    class TransformerBlock(nn.Module):
        """Pre-norm transformer block: `x + Attn(LN(x))`, then `x + MLP(LN(x))`."""

        def __init__(self, config: ViTConfig) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(config.embed_dim)
            self.attn = MultiHeadSelfAttention(config)
            self.norm2 = nn.LayerNorm(config.embed_dim)
            drop = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
            self.mlp = nn.Sequential(
                nn.Linear(config.embed_dim, config.mlp_hidden_dim),
                nn.GELU(),
                drop,
                nn.Linear(config.mlp_hidden_dim, config.embed_dim),
                nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class VisionTransformer(nn.Module):
        """Lightweight ViT over a dense `(batch, C, H, W)` feature map.

        Input
        -----
        `(batch, config.in_channels, H, W)`, any `H`, `W >= 1`.

        Output
        ------
        `(batch, n_tokens, config.embed_dim)` where `n_tokens = gh * gw`,
        `(gh, gw) = ceil(H / patch_h), ceil(W / patch_w)`, tokens in
        row-major order over the patch grid.

        Examples
        --------
        >>> vit = VisionTransformer(ViTConfig(in_channels=64))
        >>> vit(torch.randn(2, 64, 101, 241)).shape
        torch.Size([2, 403, 256])
        """

        def __init__(self, config: ViTConfig | None = None) -> None:
            _require_torch()
            super().__init__()
            self.config = config or ViTConfig()
            cfg = self.config

            self.patch_embed = nn.Conv2d(
                cfg.in_channels, cfg.embed_dim, kernel_size=cfg.patch_hw, stride=cfg.patch_hw
            )
            self.embed_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
            self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.depth)])
            self.norm = nn.LayerNorm(cfg.embed_dim)
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module: "nn.Module") -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        @property
        def in_channels(self) -> int:
            return self.config.in_channels

        @property
        def embed_dim(self) -> int:
            return self.config.embed_dim

        def grid_shape(self, height: int, width: int) -> Tuple[int, int]:
            """Token grid `(gh, gw)` this model produces for an `H x W` input."""
            return grid_shape_for(height, width, self.config.patch_hw)

        def num_tokens(self, height: int, width: int) -> int:
            gh, gw = self.grid_shape(height, width)
            return gh * gw

        def tokens_to_map(self, tokens: "torch.Tensor", height: int, width: int) -> "torch.Tensor":
            """Reshape `(B, N, D)` tokens back to a `(B, D, gh, gw)` patch-grid map."""
            gh, gw = self.grid_shape(height, width)
            if tokens.dim() != 3 or tokens.shape[1] != gh * gw:
                raise ValueError(
                    f"expected tokens of shape (batch, {gh * gw}, D) for a {height}x{width} input, "
                    f"got {tuple(tokens.shape)}"
                )
            b, _, d = tokens.shape
            return tokens.transpose(1, 2).reshape(b, d, gh, gw)

        def _pad_to_patch_multiple(self, x: "torch.Tensor") -> "torch.Tensor":
            ph, pw = self.config.patch_hw
            pad_h = (-x.shape[2]) % ph
            pad_w = (-x.shape[3]) % pw
            if pad_h == 0 and pad_w == 0:
                return x
            mode = "replicate" if self.config.pad_mode == "replicate" else "constant"
            return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.dim() != 4:
                raise ValueError(
                    f"VisionTransformer expects a 4D (batch, channels, H, W) tensor, got shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.config.in_channels:
                raise ValueError(
                    f"VisionTransformer configured for {self.config.in_channels} input channels, "
                    f"got {x.shape[1]} in input of shape {tuple(x.shape)}"
                )
            x = self._pad_to_patch_multiple(x)
            x = self.patch_embed(x)  # (B, D, gh, gw)
            gh, gw = x.shape[2], x.shape[3]
            x = x.flatten(2).transpose(1, 2)  # (B, gh*gw, D), row-major
            pos = sincos_2d_position_embedding(gh, gw, self.config.embed_dim, device=x.device, dtype=x.dtype)
            x = self.embed_dropout(x + pos.unsqueeze(0))
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

        def get_embedding(self, x: "torch.Tensor") -> "torch.Tensor":
            """Pooled `(batch, embed_dim)` embedding — the model's actual
            learned representation, not a separate or synthetic one.

            This calls `forward()` (the exact same patchify -> position
            embed -> transformer-blocks -> norm computation used
            everywhere else) and mean-pools the resulting token sequence
            over the token dimension. `VisionTransformer` has no CLS
            token by design (see the module docstring), so this mean is
            the model's own global summary of the token grid — precisely
            what a downstream depth decoder would consume as "the
            embedding" for a sample, and it moves with the model's
            weights and inherits `forward`'s eval/train dropout
            behaviour, gradient flow, and shape/channel validation.

            Parameters
            ----------
            x:
                `(batch, config.in_channels, H, W)`, any `H`, `W >= 1`.

            Returns
            -------
            `(batch, config.embed_dim)`.
            """
            tokens = self.forward(x)  # (B, N, embed_dim)
            return tokens.mean(dim=1)  # (B, embed_dim)

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            c = self.config
            return (
                f"VisionTransformer(in_channels={c.in_channels}, patch_size={c.patch_hw}, "
                f"embed_dim={c.embed_dim}, heads={c.num_heads}, depth={c.depth}, dropout={c.dropout})"
            )

    class CNNViTEncoder(nn.Module):
        """CNN spatial encoder feeding a Vision Transformer.

        `(batch, NEER_N_CHANNELS, H, W)` -> `CNNEncoder` ->
        `(batch, C_cnn, H, W)` -> `VisionTransformer` ->
        `(batch, n_tokens, embed_dim)`.

        If `vit_config` is omitted it is derived so its `in_channels`
        matches the CNN's output width; if given, a mismatch raises.

        **Optional GNN (Phase 22).** If `gnn_config` is given, a
        `GridGNN` refines the CNN's `(batch, C_cnn, H, W)` feature map
        between the CNN and the ViT: `CNN -> GridGNN -> ViT`. When it is
        `None` (the default) no GNN module is created, so the parameters,
        `state_dict` keys and forward computation are exactly those of
        the plain `CNN -> ViT` encoder. If the GNN uses the ocean current
        (`gnn_config.use_current`), the `u_current` / `v_current` channels
        are read from this encoder's *input* `x` at
        `gnn_config.current_channels`.
        """

        def __init__(
            self,
            cnn_config: CNNEncoderConfig | None = None,
            vit_config: ViTConfig | None = None,
            gnn_config: GNNConfig | None = None,
        ) -> None:
            _require_torch()
            super().__init__()
            # Imported here so a torch-less environment still reaches
            # `_require_torch` above with the helpful error first.
            from src.models.encoder import CNNEncoder
            from src.models.gnn import GridGNN

            self.cnn = CNNEncoder(cnn_config)
            cnn_out = self.cnn.out_channels
            if vit_config is None:
                vit_config = ViTConfig(in_channels=cnn_out)
            elif vit_config.in_channels != cnn_out:
                raise ValueError(
                    f"ViTConfig.in_channels ({vit_config.in_channels}) must equal the CNN encoder's "
                    f"out_channels ({cnn_out})"
                )

            self.gnn = None
            if gnn_config is not None:
                if gnn_config.in_channels != cnn_out:
                    raise ValueError(
                        f"GNNConfig.in_channels ({gnn_config.in_channels}) must equal the CNN encoder's "
                        f"out_channels ({cnn_out})"
                    )
                if gnn_config.use_current:
                    out_of_range = [c for c in gnn_config.current_channels if c >= self.cnn.in_channels]
                    if out_of_range:
                        raise ValueError(
                            f"GNNConfig.current_channels {gnn_config.current_channels} must index "
                            f"the model input's {self.cnn.in_channels} channels, but {out_of_range} "
                            f"is out of range (set use_current=False, or point current_channels at "
                            f"the u/v current channels of this input layout)"
                        )
                self.gnn = GridGNN(gnn_config)

            self.vit = VisionTransformer(vit_config)

        @property
        def in_channels(self) -> int:
            return self.cnn.in_channels

        @property
        def embed_dim(self) -> int:
            return self.vit.embed_dim

        @property
        def uses_gnn(self) -> bool:
            """Whether the optional Phase 22 GNN refinement is part of this encoder."""
            return self.gnn is not None

        def grid_shape(self, height: int, width: int) -> Tuple[int, int]:
            return self.vit.grid_shape(height, width)

        def tokens_to_map(self, tokens: "torch.Tensor", height: int, width: int) -> "torch.Tensor":
            return self.vit.tokens_to_map(tokens, height, width)

        def _spatial_features(self, x: "torch.Tensor") -> "torch.Tensor":
            """The `(batch, C_cnn, H, W)` map the ViT consumes: the CNN's
            output, refined by the GNN when one is enabled."""
            features = self.cnn(x)
            if self.gnn is None:
                return features
            current = None
            if self.gnn.uses_current:
                u_idx, v_idx = self.gnn.config.current_channels
                current = x[:, [u_idx, v_idx]]  # (B, 2, H, W) straight from the model input
            return self.gnn(features, current)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.vit(self._spatial_features(x))

        def get_embedding(self, x: "torch.Tensor") -> "torch.Tensor":
            """Pooled `(batch, embed_dim)` embedding for the full
            `CNNEncoder -> VisionTransformer` pipeline.

            This is the model's real, learned representation — the
            entry point downstream code (a depth decoder, a similarity
            search, an explainability tool) should call to get "the
            embedding" for a sample. It delegates to
            `VisionTransformer.get_embedding` on this model's own CNN
            feature map (GNN-refined when one is enabled), so it is exactly the mean of this same model's
            `forward()` token output (`get_embedding(x) ==
            forward(x).mean(dim=1)`), never a separate or synthetic
            computation, and it carries gradients back through the CNN
            and inherits `forward`'s deterministic-in-eval-mode
            behaviour.

            Parameters
            ----------
            x:
                `(batch, NEER_N_CHANNELS, H, W)`, any `H`, `W >= 1`.

            Returns
            -------
            `(batch, config.embed_dim)` — `(batch, 256)` under every
            default config (`configs/base.yaml`'s `model.embedding_dim`).
            """
            return self.vit.get_embedding(self._spatial_features(x))

else:  # pragma: no cover - exercised only in a torch-less environment

    class VisionTransformer:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed (see `encoder.CNNEncoder`)."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class CNNViTEncoder:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()