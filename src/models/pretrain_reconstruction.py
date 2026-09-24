"""
Phase 21A.2 — Complete Self-Supervised Reconstruction Model & Masked Reconstruction Loss.

Wires the Phase 21A.1 `PretrainEncoder` and the Phase 21A.2 `ReconstructionDecoder`
into a complete self-supervised pretraining model:

    surface fields  (batch, NEER_N_CHANNELS, lat, lon)
    -> CNN encoder                                       (Phase 13)
    -> ViT encoder                                       (Phase 14)
    -> mean-pool over tokens
    -> latent embedding  (batch, embed_dim)              (Phase 21A.1)
    -> reconstruction decoder                            (Phase 21A.2)
    -> reconstructed surface fields  (batch, NEER_N_CHANNELS, lat, lon)

Architecture & Contract
-----------------------
1. Reuses `PretrainEncoder` from Phase 21A.1 unmodified.
2. Reconstructs full surface fields with identical spatial and channel dimensions.
3. Exposes both `latent_embedding` and `reconstructed_surface_fields` in `forward()`.
4. Provides masked reconstruction MSE loss:
   - Evaluates squared error exclusively over valid reconstruction targets.
   - Ignores land cells (via `land_mask` or `ocean_mask`).
   - Ignores invalid/missing target cells (e.g. NaN, Inf, unobserved).
   - Supports a configurable `masking_ratio` in [0.0, 1.0].
5. Configurable parameters:
   - `masking_ratio`: fraction of valid cells to reconstruct.
   - `embedding_dim`: latent bottleneck dimension.
   - `decoder_dimensions`: hidden channel widths and depth of the reconstruction decoder.
   - `loss_config`: loss configuration options (masking ratio, eps, reduction).

torch is optional
------------------
Importing this module and building configuration dataclasses never requires torch;
only instantiating models or computing loss raises `MissingDependencyError` when
torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence, Tuple, Union

from src.data.loaders.errors import MissingDependencyError
from src.models.encoder import CNNEncoderConfig
from src.models.pretrain_decoder import (
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_SPATIAL_SHAPE,
    ReconstructionDecoder,
    ReconstructionDecoderConfig,
)
from src.models.pretrain_encoder import PretrainEncoder, PretrainEncoderConfig
from src.models.vit import DEFAULT_EMBED_DIM, ViTConfig

#: Numerical stability floor for valid-cell count divisions
DEFAULT_EPS = 1e-8

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
            purpose="the NEER self-supervised reconstruction model (src.models.pretrain_reconstruction)",
            install_hint="pip install torch",
        )


__all__ = [
    "MaskedReconstructionLoss",
    "MaskedReconstructionLossConfig",
    "PretrainOutput",
    "PretrainReconstructionConfig",
    "PretrainReconstructionModel",
    "ReconstructionDecoder",
    "ReconstructionDecoderConfig",
    "DEFAULT_DECODER_CHANNELS",
    "masked_reconstruction_loss",
    "masked_reconstruction_mse",
]


def _broadcast_to_target(mask: "torch.Tensor", target_shape: Tuple[int, ...]) -> "torch.Tensor":
    """Broadcast a boolean or numeric mask to match `target_shape`."""
    if mask.shape == target_shape:
        return mask
    # 2D (H, W) -> 4D (B, C, H, W)
    if mask.dim() == 2 and len(target_shape) == 4 and mask.shape == (target_shape[-2], target_shape[-1]):
        return mask.unsqueeze(0).unsqueeze(0).expand(target_shape)
    # 3D (B, H, W) -> 4D (B, C, H, W)
    if mask.dim() == 3 and len(target_shape) == 4 and mask.shape == (target_shape[0], target_shape[-2], target_shape[-1]):
        return mask.unsqueeze(1).expand(target_shape)
    # 3D (C, H, W) -> 4D (B, C, H, W)
    if mask.dim() == 3 and len(target_shape) == 4 and mask.shape == (target_shape[1], target_shape[-2], target_shape[-1]):
        return mask.unsqueeze(0).expand(target_shape)
    try:
        return mask.expand(target_shape)
    except Exception:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} cannot be broadcast to target shape {target_shape}"
        )


# --------------------------------------------------------------------------
# Masked Reconstruction MSE Loss
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedReconstructionLossConfig:
    """Configuration for `MaskedReconstructionLoss`.

    Parameters
    ----------
    masking_ratio:
        Fraction of valid ocean cells to evaluate as reconstruction targets.
        Float in [0.0, 1.0]. Defaults to 0.5 (50% of valid cells).
        - 1.0 scores all valid cells.
        - 0.0 scores 0 cells (returns 0.0).
        - Between 0.0 and 1.0, samples a random subset of valid cells.
    spatial_masking:
        When True, masking is applied across all channels simultaneously for each
        spatial location (h, w), forcing the model to infer missing fields from spatial
        context rather than collocated channels. Defaults to True.
    eps:
        Numerical floor on valid-cell count for division, avoiding division by zero
        when no valid cells are present. Defaults to `DEFAULT_EPS` (1e-8).
    reduction:
        Reduction mode: `"mean"` (default) or `"sum"`.
    """

    masking_ratio: float = 0.5
    spatial_masking: bool = True
    eps: float = DEFAULT_EPS
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if not (0.0 <= self.masking_ratio <= 1.0):
            raise ValueError(f"masking_ratio must be in [0.0, 1.0], got {self.masking_ratio}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {self.reduction!r}")


def masked_reconstruction_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
    land_mask: Optional["torch.Tensor"] = None,
    ocean_mask: Optional["torch.Tensor"] = None,
    masking_ratio: Optional[float] = None,
    target_mask: Optional["torch.Tensor"] = None,
    land_is_ocean: bool = False,
    spatial_masking: bool = True,
    eps: float = DEFAULT_EPS,
    reduction: str = "mean",
    generator: Optional["torch.Generator"] = None,
) -> "torch.Tensor":
    """Mean squared error over valid reconstruction targets only.

    Parameters
    ----------
    pred:
        Predicted / reconstructed surface fields, matching `target.shape`.
    target:
        Target surface fields, matching `pred.shape`.
    mask:
        Optional general validity mask where True/1 indicates valid cells.
    land_mask:
        Optional land mask. If `land_is_ocean=False` (default), True/1 marks land
        cells that MUST BE IGNORED. If `land_is_ocean=True`, True/1 marks ocean.
    ocean_mask:
        Optional ocean mask where True/1 marks ocean cells (land cells are ignored).
    masking_ratio:
        Optional override for the masking ratio. If provided, samples a random subset
        of valid cells of fraction `masking_ratio`.
    target_mask:
        Optional explicit binary mask indicating the exact reconstruction target cells.
        Overrides random sampling if provided.
    land_is_ocean:
        Whether `land_mask` uses NEER's legacy convention where True = ocean. Defaults to False.
    spatial_masking:
        Whether random masking drops entire spatial columns `(h, w)` across channels.
    eps:
        Numerical stability floor for valid-cell count.
    reduction:
        `"mean"` or `"sum"`.
    generator:
        Optional torch random generator for deterministic mask sampling.

    Returns
    -------
    A 0-D scalar tensor.
    """
    _require_torch()
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have matching shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    # 1. Base validity: finite target cells only (excludes NaNs, Infs, missing entries)
    valid = torch.isfinite(target)

    # 2. Exclude land cells via ocean_mask (True = ocean)
    if ocean_mask is not None:
        ocean_b = _broadcast_to_target(ocean_mask > 0, target.shape)
        valid = valid & ocean_b

    # 3. Exclude land cells via land_mask
    if land_mask is not None:
        if land_is_ocean:
            land_b = _broadcast_to_target(land_mask > 0, target.shape)
            valid = valid & land_b
        else:
            # Standard: True = land, so ocean is ~land
            is_land = _broadcast_to_target(land_mask > 0, target.shape)
            valid = valid & (~is_land)

    # 4. Optional general validity mask
    if mask is not None:
        mask_b = _broadcast_to_target(mask > 0, target.shape)
        valid = valid & mask_b

    # 5. Masking ratio / target mask selection
    if target_mask is not None:
        t_mask_b = _broadcast_to_target(target_mask > 0, target.shape)
        eval_mask = valid & t_mask_b
    elif masking_ratio is not None:
        if not (0.0 <= masking_ratio <= 1.0):
            raise ValueError(f"masking_ratio must be in [0.0, 1.0], got {masking_ratio}")
        if masking_ratio == 1.0:
            eval_mask = valid
        elif masking_ratio == 0.0:
            eval_mask = torch.zeros_like(valid)
        else:
            if spatial_masking and valid.dim() == 4:
                # Columnar / spatial masking across all channels
                rand_shape = (valid.shape[0], 1, valid.shape[2], valid.shape[3])
                rand = torch.rand(rand_shape, device=valid.device, dtype=torch.float32, generator=generator)
                rand = rand.expand(valid.shape)
            else:
                rand = torch.rand(valid.shape, device=valid.device, dtype=torch.float32, generator=generator)

            eval_mask = valid & (rand < masking_ratio)
            # Ensure at least one valid cell is evaluated if valid cells exist but random draw hit none
            if valid.any() and not eval_mask.any():
                rand_valid = torch.where(valid, rand, torch.tensor(2.0, device=rand.device))
                eval_mask.view(-1)[rand_valid.argmin()] = True
    else:
        eval_mask = valid

    # 6. Compute masked squared error cleanly without NaN propagation from invalid cells
    eval_f = eval_mask.to(dtype=pred.dtype)
    denom = eval_f.sum().clamp_min(eps)

    # Zero-out diff on non-evaluated cells so invalid target cells (like NaN) cannot corrupt the sum
    diff = torch.where(eval_mask, pred - target, torch.zeros_like(pred))
    squared_error = diff**2

    if reduction == "sum":
        return squared_error.sum()
    return (squared_error * eval_f).sum() / denom


#: Convenient alias matching the prompt's naming
masked_reconstruction_mse = masked_reconstruction_loss


if _TORCH_AVAILABLE:

    class MaskedReconstructionLoss(nn.Module):
        """Masked reconstruction MSE loss module."""

        def __init__(
            self,
            config: Optional[MaskedReconstructionLossConfig] = None,
            masking_ratio: Optional[float] = None,
            eps: Optional[float] = None,
            reduction: Optional[str] = None,
            spatial_masking: Optional[bool] = None,
        ) -> None:
            _require_torch()
            super().__init__()
            if config is None:
                kwargs: dict = {}
                if masking_ratio is not None:
                    kwargs["masking_ratio"] = masking_ratio
                if eps is not None:
                    kwargs["eps"] = eps
                if reduction is not None:
                    kwargs["reduction"] = reduction
                if spatial_masking is not None:
                    kwargs["spatial_masking"] = spatial_masking
                config = MaskedReconstructionLossConfig(**kwargs)
            self.config = config

        @property
        def masking_ratio(self) -> float:
            return self.config.masking_ratio

        def forward(
            self,
            pred: "torch.Tensor",
            target: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
            land_mask: Optional["torch.Tensor"] = None,
            ocean_mask: Optional["torch.Tensor"] = None,
            masking_ratio: Optional[float] = None,
            target_mask: Optional["torch.Tensor"] = None,
            land_is_ocean: bool = False,
            generator: Optional["torch.Generator"] = None,
        ) -> "torch.Tensor":
            ratio = self.config.masking_ratio if masking_ratio is None else masking_ratio
            return masked_reconstruction_loss(
                pred=pred,
                target=target,
                mask=mask,
                land_mask=land_mask,
                ocean_mask=ocean_mask,
                masking_ratio=ratio,
                target_mask=target_mask,
                land_is_ocean=land_is_ocean,
                spatial_masking=self.config.spatial_masking,
                eps=self.config.eps,
                reduction=self.config.reduction,
                generator=generator,
            )

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return f"MaskedReconstructionLoss(masking_ratio={self.masking_ratio}, eps={self.config.eps})"

else:  # pragma: no cover - exercised only in a torch-less environment

    class MaskedReconstructionLoss:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


# --------------------------------------------------------------------------
# Complete Pretraining Model Output
# --------------------------------------------------------------------------


class PretrainOutput(NamedTuple):
    """Container holding the complete pretraining model's forward pass outputs.

    Attributes
    ----------
    reconstruction:
        Reconstructed surface fields, shape `(batch, out_channels, lat, lon)`.
    embedding:
        Latent embedding, shape `(batch, embed_dim)`.
    """

    reconstruction: "torch.Tensor"
    embedding: "torch.Tensor"

    @property
    def reconstructed_surface_fields(self) -> "torch.Tensor":
        """Alias for `reconstruction` matching Phase 21A.2 terminology."""
        return self.reconstruction

    @property
    def latent_embedding(self) -> "torch.Tensor":
        """Alias for `embedding` matching Phase 21A.2 terminology."""
        return self.embedding

    def __getitem__(self, item):  # type: ignore[override]
        if isinstance(item, str):
            if item in ("reconstruction", "reconstructed", "pred", "reconstructed_surface_fields"):
                return self.reconstruction
            if item in ("embedding", "latent", "latent_embedding"):
                return self.embedding
            raise KeyError(f"Unknown PretrainOutput field: {item!r}")
        return tuple.__getitem__(self, item)


# --------------------------------------------------------------------------
# Complete Pretraining Model Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PretrainReconstructionConfig:
    """Configuration for `PretrainReconstructionModel`.

    Parameters
    ----------
    encoder_config:
        Configuration for the reused Phase 21A.1 `PretrainEncoder`.
    decoder_config:
        Configuration for the `ReconstructionDecoder`. If omitted, derived automatically
        from `encoder_config` and `embedding_dim`.
    embedding_dim:
        Latent embedding dimension. Defaults to `DEFAULT_EMBED_DIM` (256).
    masking_ratio:
        Default self-supervised masking ratio for reconstruction pretraining.
        Defaults to 0.5.
    loss_config:
        Configuration for `MaskedReconstructionLoss`. If omitted, derived using `masking_ratio`.
    decoder_dimensions:
        Optional sequence of channel dimensions (or single int) for the reconstruction decoder.
    """

    encoder_config: PretrainEncoderConfig = field(default_factory=PretrainEncoderConfig)
    decoder_config: Optional[ReconstructionDecoderConfig] = None
    embedding_dim: int = DEFAULT_EMBED_DIM
    masking_ratio: float = 0.5
    loss_config: Optional[MaskedReconstructionLossConfig] = None
    decoder_dimensions: Optional[Union[Sequence[int], int]] = None

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if not (0.0 <= self.masking_ratio <= 1.0):
            raise ValueError(f"masking_ratio must be in [0.0, 1.0], got {self.masking_ratio}")

        if self.encoder_config.embedding_dim != self.embedding_dim:
            if self.encoder_config == PretrainEncoderConfig():
                # Default encoder_config was used; align with the explicitly passed embedding_dim
                object.__setattr__(
                    self,
                    "encoder_config",
                    PretrainEncoderConfig(embedding_dim=self.embedding_dim),
                )
            else:
                raise ValueError(
                    f"encoder_config.embedding_dim ({self.encoder_config.embedding_dim}) must match "
                    f"embedding_dim ({self.embedding_dim})"
                )

        if self.decoder_config is not None:
            if self.decoder_config.embed_dim != self.embedding_dim:
                raise ValueError(
                    f"decoder_config.embed_dim ({self.decoder_config.embed_dim}) must match "
                    f"embedding_dim ({self.embedding_dim})"
                )
            expected_out = self.encoder_config.cnn_config.in_channels
            if self.decoder_config.out_channels != expected_out:
                raise ValueError(
                    f"decoder_config.out_channels ({self.decoder_config.out_channels}) must match "
                    f"encoder in_channels ({expected_out})"
                )

    def resolved_decoder_config(self) -> ReconstructionDecoderConfig:
        """The `ReconstructionDecoderConfig` built for this model."""
        if self.decoder_config is not None:
            return self.decoder_config
        kwargs: dict = {
            "embed_dim": self.embedding_dim,
            "out_channels": self.encoder_config.cnn_config.in_channels,
        }
        if self.decoder_dimensions is not None:
            kwargs["decoder_dimensions"] = self.decoder_dimensions
        return ReconstructionDecoderConfig(**kwargs)

    def resolved_loss_config(self) -> MaskedReconstructionLossConfig:
        """The `MaskedReconstructionLossConfig` built for this model."""
        if self.loss_config is not None:
            return self.loss_config
        return MaskedReconstructionLossConfig(masking_ratio=self.masking_ratio)


# --------------------------------------------------------------------------
# Complete Pretraining Model
# --------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class PretrainReconstructionModel(nn.Module):
        """Complete self-supervised reconstruction model.

        Pipeline:
            surface fields -> CNN -> ViT -> embedding -> decoder -> reconstructed surface fields.

        Exposes both:
            - `latent_embedding`
            - `reconstructed_surface_fields`
        """

        def __init__(
            self,
            config: Optional[PretrainReconstructionConfig] = None,
            encoder_config: Optional[PretrainEncoderConfig] = None,
            decoder_config: Optional[ReconstructionDecoderConfig] = None,
            embedding_dim: Optional[int] = None,
            masking_ratio: Optional[float] = None,
            decoder_dimensions: Optional[Union[Sequence[int], int]] = None,
            loss_config: Optional[MaskedReconstructionLossConfig] = None,
        ) -> None:
            _require_torch()
            super().__init__()

            if config is None:
                dim = embedding_dim or (encoder_config.embedding_dim if encoder_config else DEFAULT_EMBED_DIM)
                if encoder_config is None:
                    encoder_config = PretrainEncoderConfig(embedding_dim=dim)
                elif encoder_config.embedding_dim != dim:
                    encoder_config = PretrainEncoderConfig(
                        cnn_config=encoder_config.cnn_config,
                        vit_config=encoder_config.vit_config,
                        embedding_dim=dim,
                    )
                ratio = 0.5 if masking_ratio is None else masking_ratio
                config = PretrainReconstructionConfig(
                    encoder_config=encoder_config,
                    decoder_config=decoder_config,
                    embedding_dim=dim,
                    masking_ratio=ratio,
                    decoder_dimensions=decoder_dimensions,
                    loss_config=loss_config,
                )

            self.config = config
            self.encoder = PretrainEncoder(self.config.encoder_config)
            self.decoder = ReconstructionDecoder(self.config.resolved_decoder_config())
            self.loss_fn = MaskedReconstructionLoss(self.config.resolved_loss_config())

        @property
        def in_channels(self) -> int:
            return self.encoder.in_channels

        @property
        def out_channels(self) -> int:
            return self.decoder.out_channels

        @property
        def embed_dim(self) -> int:
            return self.encoder.embed_dim

        @property
        def masking_ratio(self) -> float:
            return self.config.masking_ratio

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            """Surface fields -> `(batch, embed_dim)` latent embedding."""
            return self.encoder(x)

        def get_embedding(self, x: "torch.Tensor") -> "torch.Tensor":
            """Expose the Phase 21A.1 latent embedding at the model level."""
            return self.encoder(x)

        def decode(
            self,
            embedding: "torch.Tensor",
            spatial_shape: Optional[Tuple[int, int]] = None,
        ) -> "torch.Tensor":
            """Latent embedding -> `(batch, out_channels, H, W)` surface fields."""
            return self.decoder(embedding, spatial_shape=spatial_shape)

        def forward(
            self,
            x: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
        ) -> PretrainOutput:
            """
            Parameters
            ----------
            x:
                `(batch, in_channels, H, W)` surface fields.
            mask:
                Optional boolean/float mask where 1 indicates unmasked cells.
                If provided, input cells can be masked (set to 0.0) prior to encoding.

            Returns
            -------
            `PretrainOutput(reconstruction=..., embedding=...)` exposing both:
            - `latent_embedding` (`output.embedding` or `output.latent_embedding`)
            - `reconstructed_surface_fields` (`output.reconstruction` or `output.reconstructed_surface_fields`)
            """
            if mask is not None:
                mask_f = mask.to(dtype=x.dtype, device=x.device)
                x_in = x * mask_f
            else:
                x_in = x

            embedding = self.encoder(x_in)
            spatial_shape = (x.shape[-2], x.shape[-1])
            reconstruction = self.decoder(embedding, spatial_shape=spatial_shape)
            return PretrainOutput(reconstruction=reconstruction, embedding=embedding)

        def compute_loss(
            self,
            pred: "torch.Tensor",
            target: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
            land_mask: Optional["torch.Tensor"] = None,
            ocean_mask: Optional["torch.Tensor"] = None,
            masking_ratio: Optional[float] = None,
            target_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """Compute masked reconstruction loss using this model's loss configuration."""
            return self.loss_fn(
                pred=pred,
                target=target,
                mask=mask,
                land_mask=land_mask,
                ocean_mask=ocean_mask,
                masking_ratio=masking_ratio,
                target_mask=target_mask,
            )

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"PretrainReconstructionModel(in_channels={self.in_channels}, "
                f"embed_dim={self.embed_dim}, out_channels={self.out_channels}, "
                f"masking_ratio={self.masking_ratio})"
            )

else:  # pragma: no cover - exercised only in a torch-less environment

    class PretrainReconstructionModel:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()
