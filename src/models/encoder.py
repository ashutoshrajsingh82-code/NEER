"""
Phase 13 — CNN spatial encoder.

The first piece of actual model architecture in NEER (see `MODEL_CARD.md`
and `src/models/__init__.py` for why nothing lived here before this
phase). It consumes the fixed input contract set in Phase 08 —
`(batch, NEER_N_CHANNELS, lat, lon)`, channels in `NEER_CHANNEL_ORDER` —
and produces a spatial feature representation: a `(batch, C_out, lat,
lon)` feature map at the *same* spatial resolution as the input.

Why a plain stack of conv blocks
---------------------------------
This encoder's only job is to turn raw per-cell physical channels into a
richer per-cell feature vector, using the immediate neighbourhood (3x3
receptive field per block) to do it. It deliberately does **not**:

- Downsample or pool. A future patch-tokenizing stage (a ViT, per
  `MODEL_CARD.md`'s TBD architecture — explicitly out of scope for this
  phase) is the natural place to turn a dense `(C, H, W)` map into
  tokens; collapsing resolution here would throw away information that
  stage might want.
- Use anything heavier than 3x3 convolutions with a modest channel
  count. The NEER grid is only 101 x 241 cells (`configs/base.yaml`),
  so there is no need for the deep/wide architectures large-image CNNs
  use, and every extra parameter here is a CPU training-time cost this
  project cannot spend — see `CNNEncoderConfig` docstring.

Each block is Conv2d -> normalization -> activation -> dropout, in that
order (the conventional pre-activation-free ordering: normalize the
convolution's output, then decide which of those normalized values pass
through). `same`-style padding (`kernel_size // 2`, stride 1) keeps H and
W fixed across every block, so `CNNEncoder.out_channels` is the only
thing that changes shape.

torch is optional
------------------
Mirrors `src/data/dataset.py` (Phase 12): importing this module never
requires torch, so `from src.models.encoder import CNNEncoderConfig`
works in a minimal environment (e.g. to read defaults or build a config
for a different phase to consume later). Only instantiating a
`CNNEncoder` raises `MissingDependencyError` if torch isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing.channels import NEER_CHANNEL_ORDER, NEER_N_CHANNELS

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
            purpose="the NEER CNN spatial encoder (src.models.encoder)",
            install_hint="pip install torch",
        )


#: Normalization layers a block may use. "group" is the default because
#: it works regardless of batch size (unlike batch norm, which degrades
#: with the small batches a CPU-only training run tends to use) while
#: still stabilizing per-channel statistics; "none" is available for
#: architectures/ablations that don't want any.
VALID_NORMS: Tuple[str, ...] = ("batch", "group", "instance", "none")

#: Activations a block may use. All are cheap, CPU-friendly elementwise
#: ops — nothing here trades accuracy for hardware NEER doesn't target.
VALID_ACTIVATIONS: Tuple[str, ...] = ("relu", "leaky_relu", "gelu", "silu")


@dataclass(frozen=True)
class CNNEncoderConfig:
    """Configuration for `CNNEncoder`.

    Parameters
    ----------
    in_channels:
        Number of input channels. Defaults to `NEER_N_CHANNELS` (11) —
        the authoritative NEER input contract from
        `src.data.preprocessing.channels` — but is left configurable so
        the encoder can be unit-tested, or reused, against a different
        channel count without editing this module.
    channels:
        Output channel count of each conv block, in order. E.g. `(32,
        64)` builds two blocks: `in_channels -> 32 -> 64`. Length sets
        the depth of the encoder. Kept short and narrow by default (see
        `DEFAULT_CHANNELS`) — this is meant to run on a laptop CPU
        against a 101x241 grid, not a GPU cluster against ImageNet.
    kernel_size:
        Convolution kernel size, applied to every block. Must be odd, so
        `same`-padding (`kernel_size // 2`) keeps H and W exactly fixed.
    norm:
        One of `VALID_NORMS`.
    norm_groups:
        Number of groups for `norm="group"`. Each block's channel count
        must be divisible by this; blocks where it isn't fall back to
        one group per channel (instance-norm-equivalent) rather than
        raising, since the whole point of a configurable channel list is
        not having to hand-tune this for every combination.
    activation:
        One of `VALID_ACTIVATIONS`.
    dropout:
        Dropout probability applied (as `Dropout2d`, i.e. whole-channel
        dropout, the appropriate form for conv feature maps) after every
        block's activation. `0.0` disables it.
    padding_mode:
        Passed straight to `nn.Conv2d`. `"zeros"` is the default;
        `"replicate"` is offered because the NEER grid has real physical
        edges (the domain boundary in `configs/base.yaml`), where
        replicating the edge value is a more defensible boundary
        assumption than padding with zeros.
    """

    in_channels: int = NEER_N_CHANNELS
    channels: Sequence[int] = field(default_factory=lambda: (32, 64))
    kernel_size: int = 3
    norm: str = "group"
    norm_groups: int = 8
    activation: str = "relu"
    dropout: float = 0.1
    padding_mode: str = "zeros"

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if len(self.channels) == 0:
            raise ValueError("channels must be non-empty (at least one conv block)")
        for c in self.channels:
            if c <= 0:
                raise ValueError(f"every entry in channels must be positive, got {list(self.channels)}")
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
        if self.padding_mode not in ("zeros", "replicate", "reflect", "circular"):
            raise ValueError(
                f"padding_mode must be one of 'zeros', 'replicate', 'reflect', "
                f"'circular', got {self.padding_mode!r}"
            )

    @property
    def out_channels(self) -> int:
        """Channel count of the encoder's output feature map."""
        return self.channels[-1]


#: A small default depth/width — two blocks, 32 then 64 channels — sized
#: to stay fast on CPU against the NEER grid while still giving the
#: encoder enough capacity to combine the 11 input channels usefully.
DEFAULT_CHANNELS: Tuple[int, ...] = (32, 64)


def _make_norm(norm: str, num_channels: int, num_groups: int) -> "nn.Module":
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm == "group":
        groups = num_groups if num_channels % num_groups == 0 else 1
        return nn.GroupNorm(groups, num_channels)
    return nn.Identity()  # norm == "none"


def _make_activation(activation: str) -> "nn.Module":
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if activation == "gelu":
        return nn.GELU()
    return nn.SiLU(inplace=True)  # activation == "silu"


if _TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """One Conv2d -> norm -> activation -> dropout block.

        `same`-padding (`kernel_size // 2`, stride 1) is fixed rather
        than configurable: preserving H and W is the property this
        whole encoder exists to guarantee (see module docstring), so
        it isn't exposed as something a config could accidentally break.
        """

        def __init__(self, config: CNNEncoderConfig, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=config.kernel_size,
                stride=1,
                padding=config.kernel_size // 2,
                padding_mode=config.padding_mode,
                bias=(config.norm == "none"),
            )
            self.norm = _make_norm(config.norm, out_ch, config.norm_groups)
            self.activation = _make_activation(config.activation)
            self.dropout = nn.Dropout2d(config.dropout) if config.dropout > 0.0 else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            return x

    class CNNEncoder(nn.Module):
        """Lightweight CPU-friendly CNN spatial encoder.

        Input
        -----
        `(batch, config.in_channels, H, W)` — by default
        `(batch, NEER_N_CHANNELS, H, W)` matching `NEER_CHANNEL_ORDER`.

        Output
        ------
        `(batch, config.out_channels, H, W)` — a spatial feature map at
        the *same* H, W as the input, one `config.out_channels`-length
        feature vector per grid cell.

        Examples
        --------
        >>> config = CNNEncoderConfig(channels=(16, 32), dropout=0.1)
        >>> encoder = CNNEncoder(config)
        >>> x = torch.randn(4, NEER_N_CHANNELS, 101, 241)
        >>> encoder(x).shape
        torch.Size([4, 32, 101, 241])
        """

        def __init__(self, config: CNNEncoderConfig | None = None) -> None:
            _require_torch()
            super().__init__()
            self.config = config or CNNEncoderConfig()

            blocks = []
            in_ch = self.config.in_channels
            for out_ch in self.config.channels:
                blocks.append(ConvBlock(self.config, in_ch, out_ch))
                in_ch = out_ch
            self.blocks = nn.ModuleList(blocks)

        @property
        def in_channels(self) -> int:
            return self.config.in_channels

        @property
        def out_channels(self) -> int:
            return self.config.out_channels

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.dim() != 4:
                raise ValueError(
                    f"CNNEncoder expects a 4D (batch, channels, H, W) tensor, got shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.config.in_channels:
                raise ValueError(
                    f"CNNEncoder configured for {self.config.in_channels} input channels "
                    f"(NEER_CHANNEL_ORDER={list(NEER_CHANNEL_ORDER)} by default), got "
                    f"{x.shape[1]} in input of shape {tuple(x.shape)}"
                )
            for block in self.blocks:
                x = block(x)
            return x

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"CNNEncoder(in_channels={self.in_channels}, "
                f"channels={list(self.config.channels)}, "
                f"norm={self.config.norm!r}, activation={self.config.activation!r}, "
                f"dropout={self.config.dropout})"
            )

else:  # pragma: no cover - exercised only in a torch-less environment

    class CNNEncoder:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed.

        Defined (rather than left as a bare `NameError`) so that
        `from src.models.encoder import CNNEncoder` still succeeds and
        the helpful `MissingDependencyError` is raised at construction
        time, matching `src.data.dataset.NEERDataset`'s behaviour.
        """

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()