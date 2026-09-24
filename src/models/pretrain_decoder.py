"""
Phase 21A.2 — Reconstruction Decoder.

Takes the latent embedding produced by `PretrainEncoder` (Phase 21A.1),
`(batch, embed_dim)`, and reconstructs the full-resolution surface fields:
`(batch, embed_dim) -> (batch, out_channels, lat, lon)`.

Architecture & Design Choices
-----------------------------
1. Latent embedding -> spatial surface fields:
   The encoder collapses the spatial feature map into a 1D embedding
   `(batch, embed_dim)` via mean-pooling over ViT tokens. The reconstruction
   decoder takes this compact representation and maps it back to the full
   multi-channel spatial grid `(batch, out_channels, H, W)`.

2. Spatial dimension matching:
   By default, the decoder reconstructs onto the standard NEER domain grid
   `(101, 241)` (`configs/base.yaml`), but it dynamically supports any target
   spatial shape `(H, W)` supplied at runtime. The output spatial and channel
   dimensions match the input surface fields exactly.

3. Lightweight, CPU-friendly convolutional decoding:
   Linear projection (`embed_dim -> c0`) followed by spatial broadcasting,
   2D normalized coordinate conditioning (`[-1, 1]` meshgrid), and a
   configurable sequence of 3x3 convolutional blocks (`conv -> norm -> activation -> dropout`)
   ending with a final 3x3 projection to `out_channels`.

4. Fully configurable:
   Embedding dimension, output channels, decoder channel dimensions (width and depth),
   normalization type, activation, and dropout are all customizable via
   `ReconstructionDecoderConfig`.

torch is optional
------------------
Importing this module and creating a `ReconstructionDecoderConfig` never requires
torch; only instantiating a `ReconstructionDecoder` raises `MissingDependencyError`
when torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing.channels import NEER_N_CHANNELS
from src.models.encoder import VALID_ACTIVATIONS, VALID_NORMS
from src.models.vit import DEFAULT_EMBED_DIM

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
            purpose="the NEER reconstruction decoder (src.models.pretrain_decoder)",
            install_hint="pip install torch",
        )


#: Default decoder hidden channels for lightweight CPU decoding.
DEFAULT_DECODER_CHANNELS: Tuple[int, ...] = (128, 64, 32)

#: Default NEER grid shape (lat, lon) from configs/base.yaml.
DEFAULT_SPATIAL_SHAPE: Tuple[int, int] = (101, 241)


@dataclass(frozen=True)
class ReconstructionDecoderConfig:
    """Configuration for `ReconstructionDecoder`.

    Parameters
    ----------
    embed_dim:
        Dimension of the input latent embedding. Defaults to `DEFAULT_EMBED_DIM` (256).
    out_channels:
        Number of output surface channels to reconstruct. Defaults to `NEER_N_CHANNELS` (11).
    decoder_channels:
        Channel progression of decoder convolutional blocks. Defaults to `(128, 64, 32)`.
    decoder_dimensions:
        Optional convenience alias or override for `decoder_channels`. Can be a single int
        or sequence of ints.
    spatial_shape:
        Default target `(lat, lon)` spatial resolution when not provided during `forward`.
        Defaults to `(101, 241)`.
    kernel_size:
        Kernel size for convolutional decoding blocks (must be positive odd integer). Defaults to 3.
    norm:
        One of `VALID_NORMS` (`"group"`, `"batch"`, `"instance"`, `"none"`). Defaults to `"group"`.
    norm_groups:
        Number of groups for group normalization. Defaults to 8.
    activation:
        One of `VALID_ACTIVATIONS` (`"relu"`, `"leaky_relu"`, `"gelu"`, `"silu"`). Defaults to `"gelu"`.
    dropout:
        Dropout probability applied after activations in decoder blocks. Defaults to 0.0.
    use_coords:
        Whether to concatenate normalized `[-1, 1]` 2D coordinate meshgrids to provide explicit
        spatial conditioning to the decoder. Defaults to True.
    """

    embed_dim: int = DEFAULT_EMBED_DIM
    out_channels: int = NEER_N_CHANNELS
    decoder_channels: Sequence[int] = field(default_factory=lambda: DEFAULT_DECODER_CHANNELS)
    decoder_dimensions: Optional[Union[Sequence[int], int]] = None
    spatial_shape: Tuple[int, int] = DEFAULT_SPATIAL_SHAPE
    kernel_size: int = 3
    norm: str = "group"
    norm_groups: int = 8
    activation: str = "gelu"
    dropout: float = 0.0
    use_coords: bool = True

    def __post_init__(self) -> None:
        if self.decoder_dimensions is not None:
            if isinstance(self.decoder_dimensions, int):
                if self.decoder_dimensions <= 0:
                    raise ValueError(f"decoder_dimensions must be positive, got {self.decoder_dimensions}")
                channels = (self.decoder_dimensions, max(1, self.decoder_dimensions // 2))
                object.__setattr__(self, "decoder_channels", channels)
            elif hasattr(self.decoder_dimensions, "__iter__"):
                channels = tuple(int(c) for c in self.decoder_dimensions)
                object.__setattr__(self, "decoder_channels", channels)
            else:
                raise TypeError(
                    f"decoder_dimensions must be an int or sequence of ints, got {type(self.decoder_dimensions)}"
                )

        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")
        if len(self.decoder_channels) == 0:
            raise ValueError("decoder_channels must be non-empty (at least one channel dimension)")
        for c in self.decoder_channels:
            if c <= 0:
                raise ValueError(f"every entry in decoder_channels must be positive, got {list(self.decoder_channels)}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {self.kernel_size}")
        if self.norm not in VALID_NORMS:
            raise ValueError(f"norm must be one of {VALID_NORMS}, got {self.norm!r}")
        if self.norm_groups <= 0:
            raise ValueError(f"norm_groups must be positive, got {self.norm_groups}")
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {VALID_ACTIVATIONS}, got {self.activation!r}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if len(self.spatial_shape) != 2 or self.spatial_shape[0] <= 0 or self.spatial_shape[1] <= 0:
            raise ValueError(f"spatial_shape must be a tuple of 2 positive ints, got {self.spatial_shape}")


if _TORCH_AVAILABLE:

    class ReconstructionDecoder(nn.Module):
        """Reconstruction decoder: latent embedding -> reconstructed surface fields.

        Maps `(batch, embed_dim)` -> `(batch, out_channels, lat, lon)`.
        """

        def __init__(self, config: Optional[ReconstructionDecoderConfig] = None) -> None:
            _require_torch()
            super().__init__()
            self.config = config or ReconstructionDecoderConfig()
            cfg = self.config

            c0 = cfg.decoder_channels[0]
            self.in_proj = nn.Linear(cfg.embed_dim, c0)

            in_features = c0 + (2 if cfg.use_coords else 0)
            blocks: list[nn.Module] = []
            cur_c = in_features
            for next_c in cfg.decoder_channels:
                blocks.append(self._build_block(cur_c, next_c))
                cur_c = next_c
            self.blocks = nn.ModuleList(blocks)

            self.out_conv = nn.Conv2d(
                cur_c,
                cfg.out_channels,
                kernel_size=cfg.kernel_size,
                padding=cfg.kernel_size // 2,
            )

        def _build_block(self, in_c: int, out_c: int) -> nn.Sequential:
            cfg = self.config
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2)
            ]

            # Normalization
            if cfg.norm == "group":
                groups = min(cfg.norm_groups, out_c)
                while out_c % groups != 0 and groups > 1:
                    groups -= 1
                layers.append(nn.GroupNorm(groups, out_c))
            elif cfg.norm == "batch":
                layers.append(nn.BatchNorm2d(out_c))
            elif cfg.norm == "instance":
                layers.append(nn.InstanceNorm2d(out_c))
            elif cfg.norm == "none":
                pass

            # Activation
            if cfg.activation == "relu":
                layers.append(nn.ReLU(inplace=False))
            elif cfg.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1, inplace=False))
            elif cfg.activation == "gelu":
                layers.append(nn.GELU())
            elif cfg.activation == "silu":
                layers.append(nn.SiLU(inplace=False))

            # Dropout
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout2d(cfg.dropout))

            return nn.Sequential(*layers)

        @property
        def embed_dim(self) -> int:
            return self.config.embed_dim

        @property
        def in_channels(self) -> int:
            return self.config.embed_dim

        @property
        def out_channels(self) -> int:
            return self.config.out_channels

        def forward(
            self,
            embedding: "torch.Tensor",
            spatial_shape: Optional[Tuple[int, int]] = None,
        ) -> "torch.Tensor":
            """
            Parameters
            ----------
            embedding:
                `(batch, embed_dim)` latent embedding.
            spatial_shape:
                Target `(lat, lon)` spatial resolution. Defaults to `config.spatial_shape`
                `(101, 241)` if omitted.

            Returns
            -------
            `(batch, out_channels, H, W)` reconstructed surface fields.
            """
            if embedding.dim() != 2:
                raise ValueError(
                    f"embedding must be 2-D (batch, embed_dim), got {tuple(embedding.shape)}"
                )
            if embedding.shape[1] != self.config.embed_dim:
                raise ValueError(
                    f"embedding dim ({embedding.shape[1]}) must match config.embed_dim "
                    f"({self.config.embed_dim})"
                )

            h, w = spatial_shape if spatial_shape is not None else self.config.spatial_shape
            if h <= 0 or w <= 0:
                raise ValueError(f"spatial_shape dimensions must be positive, got ({h}, {w})")

            batch_size = embedding.shape[0]
            # Project embedding to first channel width and broadcast spatially
            feat = self.in_proj(embedding)  # (batch, c0)
            feat = feat.view(batch_size, -1, 1, 1).expand(-1, -1, h, w)  # (batch, c0, H, W)

            if self.config.use_coords:
                # 2D coordinate grid in [-1, 1]
                ys = torch.linspace(-1.0, 1.0, h, device=embedding.device, dtype=feat.dtype)
                xs = torch.linspace(-1.0, 1.0, w, device=embedding.device, dtype=feat.dtype)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
                coords = (
                    torch.stack([grid_y, grid_x], dim=0)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1, -1)
                )  # (batch, 2, H, W)
                feat = torch.cat([feat, coords], dim=1)

            for block in self.blocks:
                feat = block(feat)

            return self.out_conv(feat)

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"ReconstructionDecoder(embed_dim={self.embed_dim}, out_channels={self.out_channels}, "
                f"decoder_channels={self.config.decoder_channels})"
            )

else:  # pragma: no cover - exercised only in a torch-less environment

    class ReconstructionDecoder:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()
