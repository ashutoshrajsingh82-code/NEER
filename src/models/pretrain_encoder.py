"""
Phase 21A.1 — self-supervised pretraining encoder.

The first piece of the self-supervised pretraining track. It wires the
same two building blocks the supervised pipeline uses (Phase 13's
`CNNEncoder`, Phase 14's `VisionTransformer`) into a standalone encoder
that produces one thing and one thing only: a latent embedding.

    surface fields  (batch, NEER_N_CHANNELS, lat, lon)
    -> CNNEncoder                                        (Phase 13)
    -> VisionTransformer                                 (Phase 14)
    -> mean-pool over tokens
    -> latent embedding  (batch, embed_dim)

Why this is its own module rather than reusing `CNNViTEncoder`
--------------------------------------------------------------
`CNNViTEncoder` (`src/models/vit.py`, Phase 14/15) already composes a
`CNNEncoder` and a `VisionTransformer`, and its `get_embedding` already
mean-pools the token sequence — mechanically the same computation as
this module. `PretrainEncoder` exists alongside it, rather than instead
of it, for two reasons specific to the pretraining track:

- **A forward pass that *is* the embedding.** `CNNViTEncoder.forward`
  returns the raw token sequence `(batch, n_tokens, embed_dim)` — the
  right contract for `NEERModel`'s decoder, which cross-attends over
  depth queries against a spatial embedding pooled internally.  A
  self-supervised pretraining objective (contrastive, or a future
  reconstruction decoder — explicitly out of scope here, see below)
  wants `forward(x)` to *be* `(batch, embed_dim)` with no extra call,
  so this module's `forward` does the pooling itself rather than
  requiring every caller to remember to call `get_embedding` instead.
- **A single place to grow the pretraining pipeline.** Later phases in
  this track (a reconstruction decoder, a training loop, an
  augmentation/masking scheme) attach to `PretrainEncoder`, not to
  `CNNViTEncoder` — keeping the supervised model's encoder entry point
  (`CNNViTEncoder`, still used by `NEERModel`) untouched while this one
  evolves.

Reuse, not reimplementation
----------------------------
Every learnable weight below belongs to a `CNNEncoder` or a
`VisionTransformer` built exactly as `encoder.py` / `vit.py` already
define them — this module adds no convolution, attention, or
projection of its own. `PretrainEncoderConfig` composes their configs
(`CNNEncoderConfig`, `ViTConfig`) instead of restating any of their
fields, the same pattern `CNNViTEncoder` already uses to derive a
`ViTConfig` from a `CNNEncoderConfig`'s output width when one is not
given explicitly.

CNN output -> ViT input
------------------------
`CNNEncoder.forward` returns a dense `(batch, C_cnn, H, W)` feature
map, and `VisionTransformer.forward` accepts exactly that shape as its
own input contract (its `patch_embed` is a strided `Conv2d`, which
expects `(batch, channels, H, W)`) — so "transforming" the CNN's
output into the ViT's expected input is not a reshape this module
needs to perform; it is already the same tensor shape by construction.
What this module is responsible for is the one compatibility check
that *can* go wrong: the CNN's `out_channels` must equal the ViT's
`in_channels`, exactly the check `CNNViTEncoder.__init__` already
performs and this module mirrors.

Embedding dimension via the existing configuration system
------------------------------------------------------------
`PretrainEncoderConfig.embedding_dim` (default `DEFAULT_EMBED_DIM`,
256 — `configs/base.yaml`'s `model.embedding_dim`) is the single knob
that sets the model's latent width: when `vit_config` is omitted, it
is passed straight through as `ViTConfig(embed_dim=...)`, the same
dataclass-config mechanism every other model in this package already
uses to be configurable. No new configuration path is introduced.

Scope
-----
This phase implements the encoder only:

- No reconstruction decoder (a later 21A phase's job).
- No training loop, loss, or optimizer (also later).

Reusability for the main NEER model
--------------------------------------
`PretrainEncoder.cnn` and `PretrainEncoder.vit` are, by construction,
plain `CNNEncoder` / `VisionTransformer` instances — the exact classes
`CNNViTEncoder` (and, through it, `NEERModel`) already wrap. A later
phase can transplant a pretrained encoder's weights straight into the
supervised model without any conversion:

    >>> pretrained = PretrainEncoder(config)
    >>> # ... self-supervised training happens elsewhere ...
    >>> model = NEERModel(config.cnn_config, pretrained.vit.config)
    >>> model.encoder.cnn.load_state_dict(pretrained.cnn.state_dict())
    >>> model.encoder.vit.load_state_dict(pretrained.vit.state_dict())

torch is optional
------------------
Same lazy pattern as `encoder.py` / `vit.py`: importing this module and
building a `PretrainEncoderConfig` never requires torch; only
instantiating a `PretrainEncoder` raises `MissingDependencyError` when
torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.data.loaders.errors import MissingDependencyError
from src.models.encoder import CNNEncoderConfig
from src.models.vit import DEFAULT_EMBED_DIM, ViTConfig

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
            purpose="the NEER self-supervised pretraining encoder (src.models.pretrain_encoder)",
            install_hint="pip install torch",
        )


@dataclass(frozen=True)
class PretrainEncoderConfig:
    """Configuration for `PretrainEncoder`.

    Parameters
    ----------
    cnn_config:
        Configuration for the reused `CNNEncoder`. Defaults to
        `CNNEncoderConfig()` — the same defaults `CNNViTEncoder` uses.
    vit_config:
        Configuration for the reused `VisionTransformer`. If omitted
        (the common case), one is derived automatically: its
        `in_channels` is set to `cnn_config`'s `out_channels` and its
        `embed_dim` to this config's `embedding_dim`, exactly the
        pattern `CNNViTEncoder` uses when its own `vit_config` is
        omitted. If given explicitly, it must already agree with both
        `cnn_config.out_channels` (as `in_channels`) and
        `embedding_dim` (as `embed_dim`) — a mismatch raises rather
        than silently overriding one field of a config the caller
        built on purpose.
    embedding_dim:
        The latent embedding width — the configurable knob this phase
        asks for. Defaults to `DEFAULT_EMBED_DIM` (256), matching
        `configs/base.yaml`'s `model.embedding_dim`. Ignored (but
        still validated for internal consistency, see `vit_config`
        above) when `vit_config` is supplied explicitly, since the
        embedding width is then whatever `vit_config.embed_dim` says.
    """

    cnn_config: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    vit_config: Optional[ViTConfig] = None
    embedding_dim: int = DEFAULT_EMBED_DIM

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")

        cnn_out = self.cnn_config.out_channels
        if self.vit_config is not None:
            if self.vit_config.in_channels != cnn_out:
                raise ValueError(
                    f"vit_config.in_channels ({self.vit_config.in_channels}) must equal "
                    f"cnn_config.out_channels ({cnn_out})"
                )
            if self.vit_config.embed_dim != self.embedding_dim:
                raise ValueError(
                    f"vit_config.embed_dim ({self.vit_config.embed_dim}) must equal "
                    f"embedding_dim ({self.embedding_dim}); pass a matching embedding_dim "
                    f"(or omit vit_config and let it be derived)"
                )

    def resolved_vit_config(self) -> ViTConfig:
        """The `ViTConfig` this encoder will actually build.

        Returns `vit_config` unchanged if one was given (already
        validated against `cnn_config` / `embedding_dim` in
        `__post_init__`), otherwise derives one from `cnn_config`'s
        output width and `embedding_dim`.
        """
        if self.vit_config is not None:
            return self.vit_config
        return ViTConfig(in_channels=self.cnn_config.out_channels, embed_dim=self.embedding_dim)


if _TORCH_AVAILABLE:

    class PretrainEncoder(nn.Module):
        """Self-supervised pretraining encoder: surface fields -> latent embedding.

        Composes a `CNNEncoder` and a `VisionTransformer` (reused
        unmodified from Phases 13/14) and mean-pools the transformer's
        token sequence into a single embedding per sample — the same
        pooling `VisionTransformer.get_embedding` / `CNNViTEncoder.get_embedding`
        already define, exposed here directly as `forward`'s return
        value (see the module docstring for why).

        Examples
        --------
        >>> encoder = PretrainEncoder()
        >>> x = torch.randn(4, NEER_N_CHANNELS, 101, 241)
        >>> encoder(x).shape
        torch.Size([4, 256])
        """

        def __init__(self, config: Optional[PretrainEncoderConfig] = None) -> None:
            _require_torch()
            super().__init__()
            # Imported here so a torch-less environment still reaches
            # `_require_torch` above with the helpful error first (same
            # reasoning as `CNNViTEncoder`'s deferred import).
            from src.models.encoder import CNNEncoder
            from src.models.vit import VisionTransformer

            self.config = config or PretrainEncoderConfig()
            self.cnn = CNNEncoder(self.config.cnn_config)
            self.vit = VisionTransformer(self.config.resolved_vit_config())

        @property
        def in_channels(self) -> int:
            return self.cnn.in_channels

        @property
        def embed_dim(self) -> int:
            return self.vit.embed_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x:
                `(batch, config.cnn_config.in_channels, H, W)` surface
                fields — by default `(batch, NEER_N_CHANNELS, H, W)`
                matching `NEER_CHANNEL_ORDER`. Invalid shapes (wrong
                rank or wrong channel count) raise `ValueError` from
                `CNNEncoder.forward`, the same validation
                `CNNViTEncoder` relies on.

            Returns
            -------
            `(batch, embed_dim)` latent embedding: the CNN's spatial
            feature map, patch-tokenized and transformed by the ViT,
            mean-pooled over tokens. No reconstruction decoder or
            training-loop computation happens here — see the module
            docstring.
            """
            feature_map = self.cnn(x)  # (B, C_cnn, H, W) -- already the ViT's input contract
            tokens = self.vit(feature_map)  # (B, n_tokens, embed_dim)
            return tokens.mean(dim=1)  # (B, embed_dim)

        def load_pretrained(
            self,
            checkpoint_path_or_dict: "Union[str, Path, dict]",
            strict: bool = True,
        ) -> "PretrainEncoder":
            """Load pretrained encoder weights from a Phase 21 pretraining checkpoint."""
            from src.training.pretrain import load_pretrained_encoder

            load_pretrained_encoder(checkpoint_path_or_dict, target=self, strict=strict)
            return self

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return f"PretrainEncoder(in_channels={self.in_channels}, embed_dim={self.embed_dim})"

else:  # pragma: no cover - exercised only in a torch-less environment

    class PretrainEncoder:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed (see `encoder.CNNEncoder`)."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()