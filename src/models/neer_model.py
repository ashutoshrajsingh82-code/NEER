"""
Phase 18 — the complete NEER model.

Wires every previous phase into one `torch.nn.Module`, exactly the
pipeline this phase describes:

    (batch, NEER_N_CHANNELS, lat, lon)   input surface fields
    -> CNNEncoder                          (Phase 13)
    -> VisionTransformer                   (Phase 14)
    -> get_embedding                       (Phase 15) -> (batch, 256)
    -> DepthEmbedding, inside DepthDecoder (Phase 16)
    -> DepthDecoder                        (Phase 17) -> (batch, 15) anomalies
    -> + climatology                       (Phase 11, src.data.preprocessing.climatology)
    -> reconstructed absolute temperature, (batch, 15)

`NEERModel` does not introduce any new learnable computation of its
own — every weight belongs to the `CNNViTEncoder` or `DepthDecoder` it
holds. It exists purely to compose them under one object with one
clear input contract, and to attach the one non-learned step the
pipeline still needs: turning a predicted anomaly back into a
physical-units temperature by adding it to a climatological baseline
(Phase 11's `MonthlyClimatology`, which is fitted separately, from
data this tensor-only model never sees — lat/lon/time metadata, not
surface-field channels).

Optional GNN (Phase 22)
-----------------------
With `use_gnn=True` (config: `model.use_gnn`, default `false`) the
encoder refines the CNN's per-cell feature map with a grid-graph GNN
(`src/models/gnn.py`) before the ViT: `CNN -> GridGNN -> ViT`. With it
off, no GNN module is built and this model is exactly the Phase 18 one.
`NEERModel.from_config(config)` is how `model.use_gnn` reaches the model.
The decoder, `get_embedding` and `predict_profile` contracts are the
same either way.

Optional uncertainty (Phase 23)
--------------------------------
With `decoder_config.uncertainty_enabled=True` (config:
`uncertainty.enabled`, default `false`) the decoder additionally
predicts a per-depth log-variance alongside the mean anomaly. `forward`
and `predict_profile` are unchanged either way — they always return
just the mean/reconstructed profile. The extra output is only reachable
through `forward_with_uncertainty` / `predict_uncertainty`, and only
when the feature is enabled; both raise otherwise.
`NEERModel.from_config(config)` is how `uncertainty.enabled` reaches
the model. See `src/models/depth_decoder.py` for the head itself and
`src/training/losses.gaussian_nll_loss` for training against it.
Enabling this predicts *a* variance, not a *calibrated* one — no
calibration testing is implemented anywhere in this codebase.

Do not train yet
------------------
This phase is architecture wiring and a forward-pass smoke test only.
There is no loss function, optimizer, or training loop here, and
nothing in this module fits anything — every learnable parameter comes
from previously-tested, freshly-initialized sub-modules. See
`tests/test_neer_model.py`'s `test_complete_cpu_forward_pass` for the
full, untrained, CPU-only pipeline check this phase asks for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from src.data.loaders.errors import MissingDependencyError
from src.models.depth_decoder import DepthDecoderConfig
from src.models.depth_embedding import DepthEmbeddingConfig
from src.models.encoder import CNNEncoderConfig
from src.models.gnn import GNNConfig
from src.models.vit import ViTConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.utils.config import NeerConfig

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
            purpose="the complete NEER model (src.models.neer_model)",
            install_hint="pip install torch",
        )


if _TORCH_AVAILABLE:

    class NEERModel(nn.Module):
        """The complete NEER model: surface fields in, a depth-temperature
        profile out.

        Composes a `CNNViTEncoder` (Phases 13-15: CNN -> ViT -> pooled
        `(batch, embed_dim)` embedding) with a `DepthDecoder` (Phases
        16-17: shared depth embeddings + shared cross-attention -> 15
        anomalies). `decoder_config`'s `embed_dim` must match the
        encoder's (derived automatically when omitted, same pattern as
        `CNNViTEncoder` deriving `ViTConfig` from the CNN's output
        width).

        `use_gnn=True` inserts the optional Phase 22 grid GNN between the
        CNN and the ViT, configured by `gnn_config` (a default
        `GNNConfig` matching the CNN's output width if omitted).
        `gnn_config` without `use_gnn=True` raises rather than being
        silently ignored. `use_gnn=False` (the default) builds no GNN at
        all. To build from `configs/*.yaml`, use `NEERModel.from_config`.

        Examples
        --------
        >>> model = NEERModel().eval()
        >>> x = torch.randn(2, NEER_N_CHANNELS, 101, 241)
        >>> model(x).shape                    # forward(): anomalies
        torch.Size([2, 15])
        >>> model.get_embedding(x).shape      # get_embedding(): Phase 15's representation
        torch.Size([2, 256])
        >>> climatology = torch.zeros(15)     # a real caller uses MonthlyClimatology here
        >>> model.predict_profile(x, climatology).shape
        torch.Size([2, 15])
        """

        def __init__(
            self,
            cnn_config: Optional[CNNEncoderConfig] = None,
            vit_config: Optional[ViTConfig] = None,
            decoder_config: Optional[DepthDecoderConfig] = None,
            use_gnn: bool = False,
            gnn_config: Optional[GNNConfig] = None,
        ) -> None:
            _require_torch()
            super().__init__()
            # Imported here so a torch-less environment still reaches
            # `_require_torch` above with the helpful error first (same
            # reasoning as `CNNViTEncoder`'s own deferred import).
            from src.models.depth_decoder import DepthDecoder
            from src.models.vit import CNNViTEncoder

            if gnn_config is not None and not use_gnn:
                raise ValueError(
                    "gnn_config was given but use_gnn is False, so it would be silently ignored; "
                    "pass use_gnn=True to enable the GNN (or drop gnn_config)"
                )
            resolved_gnn_config = None
            if use_gnn:
                resolved_gnn_config = gnn_config or GNNConfig(
                    in_channels=(cnn_config or CNNEncoderConfig()).out_channels
                )

            self.encoder = CNNViTEncoder(cnn_config, vit_config, resolved_gnn_config)
            embed_dim = self.encoder.embed_dim

            if decoder_config is None:
                decoder_config = DepthDecoderConfig(embed_dim=embed_dim)
            elif decoder_config.embed_dim != embed_dim:
                raise ValueError(
                    f"DepthDecoderConfig.embed_dim ({decoder_config.embed_dim}) must equal "
                    f"the encoder's embed_dim ({embed_dim})"
                )
            self.decoder = DepthDecoder(decoder_config)

        @classmethod
        def from_config(
            cls,
            config: "NeerConfig",
            *,
            cnn_config: Optional[CNNEncoderConfig] = None,
            vit_config: Optional[ViTConfig] = None,
            decoder_config: Optional[DepthDecoderConfig] = None,
            gnn_config: Optional[GNNConfig] = None,
        ) -> "NEERModel":
            """Build a model from a resolved `NeerConfig` (`load_config(...)`).

            Reads `model.embedding_dim`, `model.use_gnn`,
            `uncertainty.enabled` and `depths` from the config — the
            single source of truth for them — and leaves every other
            hyperparameter at its default unless overridden. With the
            shipped configs (`embedding_dim: 256`, `use_gnn: false`,
            `uncertainty.enabled: false`, the 15 standard depths) the
            result is identical to `NEERModel()`.

            Parameters
            ----------
            config:
                Resolved configuration; `config.model.use_gnn` decides
                whether the Phase 22 GNN is built, and
                `config.uncertainty.enabled` decides whether the
                Phase 23 log-variance head is built.
            cnn_config, vit_config, decoder_config:
                Optional overrides. When `vit_config` / `decoder_config`
                are omitted they are derived from `config` and the CNN.
                An explicit `decoder_config` is used as-is (including
                its own `uncertainty_enabled`), so it is not overridden
                by `config.uncertainty.enabled`.
            gnn_config:
                Optional GNN hyperparameters; only valid when
                `config.model.use_gnn` is true (otherwise `NEERModel`
                raises, as for a direct `gnn_config` without `use_gnn`).
            """
            embed_dim = config.model.embedding_dim
            cnn_config = cnn_config or CNNEncoderConfig()
            vit_config = vit_config or ViTConfig(in_channels=cnn_config.out_channels, embed_dim=embed_dim)
            decoder_config = decoder_config or DepthDecoderConfig(
                embed_dim=vit_config.embed_dim,
                depth_config=DepthEmbeddingConfig(
                    depths=tuple(config.depths), embed_dim=vit_config.embed_dim
                ),
                uncertainty_enabled=config.uncertainty.enabled,
            )
            return cls(
                cnn_config,
                vit_config,
                decoder_config,
                use_gnn=config.model.use_gnn,
                gnn_config=gnn_config,
            )

        @property
        def use_gnn(self) -> bool:
            """Whether the optional Phase 22 GNN refinement is enabled."""
            return self.encoder.uses_gnn

        @property
        def in_channels(self) -> int:
            return self.encoder.in_channels

        @property
        def embed_dim(self) -> int:
            return self.encoder.embed_dim

        @property
        def depths(self):
            """The `num_depths` target depths, in `forward`'s output-column order."""
            return self.decoder.depths

        @property
        def num_depths(self) -> int:
            return self.decoder.num_depths

        @property
        def uncertainty_enabled(self) -> bool:
            """Whether this model's decoder was built with a log-variance head (Phase 23)."""
            return self.decoder.uncertainty_enabled

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x:
                `(batch, NEER_N_CHANNELS, H, W)` surface fields.

            Returns
            -------
            `(batch, num_depths)` predicted temperature anomalies
            (delta_T), one per `self.depths`, in that exact order.
            """
            embedding = self.encoder.get_embedding(x)
            return self.decoder(embedding)

        def get_embedding(self, x: "torch.Tensor") -> "torch.Tensor":
            """The Phase 15 spatial embedding, exposed at the whole-model level.

            Parameters
            ----------
            x:
                `(batch, NEER_N_CHANNELS, H, W)` surface fields.

            Returns
            -------
            `(batch, embed_dim)` — exactly
            `self.encoder.get_embedding(x)`, i.e. the same pooled
            representation `forward` builds its anomaly prediction
            from, not a separate computation.
            """
            return self.encoder.get_embedding(x)

        def forward_with_uncertainty(self, x: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
            """Phase 23 — predicted anomalies AND their log-variance.

            Only available when this model was built with
            `uncertainty.enabled=True` (`decoder_config.uncertainty_enabled`
            for a direct `NEERModel(...)` construction). The feature is
            deliberately only exposed when explicitly enabled; `forward`
            itself always returns the mean anomalies only, unaffected by
            this flag.

            Parameters
            ----------
            x:
                `(batch, NEER_N_CHANNELS, H, W)` surface fields.

            Returns
            -------
            `(mean, log_variance)`, each `(batch, num_depths)`, one
            column per `self.depths`. Convert `log_variance` to a
            standard deviation with
            `src.models.depth_decoder.log_variance_to_sigma`, or use
            `predict_uncertainty` below to get `sigma` directly.

            Raises
            ------
            RuntimeError:
                If `uncertainty_enabled` is `False` for this model.
            """
            embedding = self.encoder.get_embedding(x)
            return self.decoder.forward_with_uncertainty(embedding)

        def predict_uncertainty(self, x: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
            """Phase 23 — predicted anomalies AND their standard deviation.

            The `sigma`-flavored counterpart to `forward_with_uncertainty`:
            same log-variance under the hood, converted with
            `sigma = exp(0.5 * log_variance)`
            (`src.models.depth_decoder.log_variance_to_sigma`) so callers
            that want an uncertainty band (e.g. `mean +/- sigma`) don't
            each reimplement the conversion. Only available when this
            model was built with `uncertainty.enabled=True` — see
            `forward_with_uncertainty`.

            Parameters
            ----------
            x:
                `(batch, NEER_N_CHANNELS, H, W)` surface fields.

            Returns
            -------
            `(mean, sigma)`, each `(batch, num_depths)`. `sigma` is the
            predicted standard deviation of the anomaly at each depth —
            a measure of this model's own predicted spread, not a
            calibrated confidence interval; no calibration testing is
            implemented anywhere in this codebase.

            Raises
            ------
            RuntimeError:
                If `uncertainty_enabled` is `False` for this model.
            """
            from src.models.depth_decoder import log_variance_to_sigma

            mean, log_variance = self.forward_with_uncertainty(x)
            return mean, log_variance_to_sigma(log_variance)

        def predict_profile(
            self, x: "torch.Tensor", climatology: Union["torch.Tensor", "object"]
        ) -> "torch.Tensor":
            """Full inference: surface fields -> reconstructed temperature profile.

            Runs `forward` to get predicted anomalies, then adds the
            supplied climatology — `climatology + predicted_delta_T`,
            exactly the formula in
            `src.data.preprocessing.climatology.reconstruct_temperature`.
            This method reimplements that one-line addition for torch
            tensors (rather than calling the numpy version directly)
            so the whole pipeline stays in torch with no CPU/numpy
            round-trip; the underlying relationship, and where a real
            climatology profile comes from, is documented there.

            Parameters
            ----------
            x:
                `(batch, NEER_N_CHANNELS, H, W)` surface fields.
            climatology:
                Array-like, convertible to a tensor of shape
                `(num_depths,)` or `(batch, num_depths)` — the
                climatological baseline temperature at each target
                depth. A `(num_depths,)` profile is broadcast across
                the batch; a per-sample `(batch, num_depths)` array is
                used as-is (the usual case, since climatology varies
                by each sample's location and calendar month — see
                `MonthlyClimatology.depth_profile`).

            Returns
            -------
            `(batch, num_depths)` reconstructed absolute temperature,
            in whatever physical units `climatology` was given in.
            """
            anomalies = self.forward(x)

            climatology_t = (
                climatology
                if torch.is_tensor(climatology)
                else torch.as_tensor(climatology, dtype=torch.float32)
            )
            climatology_t = climatology_t.to(dtype=anomalies.dtype, device=anomalies.device)

            if climatology_t.shape == (self.num_depths,):
                climatology_t = climatology_t.unsqueeze(0).expand_as(anomalies)
            elif climatology_t.shape != anomalies.shape:
                raise ValueError(
                    f"climatology has shape {tuple(climatology_t.shape)}; expected "
                    f"({self.num_depths},) or {tuple(anomalies.shape)}"
                )

            return climatology_t + anomalies

        def load_pretrained_encoder(
            self,
            checkpoint_path_or_dict: Union[str, Path, dict],
            strict: bool = True,
        ) -> None:
            """Load pretrained CNN and ViT encoder weights from a Phase 21 pretraining checkpoint."""
            from src.training.pretrain import load_pretrained_into_neer_model

            load_pretrained_into_neer_model(checkpoint_path_or_dict, self, strict=strict)

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"NEERModel(in_channels={self.in_channels}, embed_dim={self.embed_dim}, "
                f"num_depths={self.num_depths}, use_gnn={self.use_gnn}, "
                f"uncertainty_enabled={self.uncertainty_enabled})"
            )

else:  # pragma: no cover - exercised only in a torch-less environment

    class NEERModel:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed (see `encoder.CNNEncoder`)."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()