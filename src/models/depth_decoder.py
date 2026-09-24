"""
Phase 17 — shared attention-based depth decoder.

Given one spatial embedding (Phase 15's `CNNViTEncoder.get_embedding`,
`(batch, embed_dim)`) and the 15 shared, learned depth embeddings
(Phase 16's `DepthEmbedding`), `DepthDecoder` predicts one scalar
subsurface-temperature anomaly per configured target depth:
`(batch, embed_dim) -> (batch, num_depths)`.

Shared, not per-depth
----------------------
There is exactly one decoder — one cross-attention block, one shared
feedforward block, one output projection — not 15 independent decoder
heads. Every depth's prediction goes through the *same* weights
(`Wq`, `Wk`, `Wv`, `Wo`, the feedforward layers, the final
`Linear(embed_dim, 1)`); what differs between depths is only which row
of `DepthEmbedding`'s table is fed in as that depth's query. This is
the same "one shared mechanism, many rows" pattern `DepthEmbedding`
itself uses (see `depth_embedding.py`), one level up: sharing wasn't
just pushed into the depth *representation*, it's threaded through the
whole decoder that reads that representation.

Architecture
------------
One pre-norm Transformer decoder block, Perceiver/DETR-style: the
`num_depths` depth embeddings are the queries, and the single spatial
embedding (treated as a length-1 key/value sequence) is what they
cross-attend into — so every depth gets to look at the same global
spatial summary, weighted by its own learned query. That is followed
by a shared two-layer feedforward block and a final shared linear
readout to one scalar per depth (the anomaly), each with a residual
connection back to the depth's own embedding so attention and the MLP
refine rather than replace a depth's identity.

Output order
------------
`forward` returns depths in exactly `self.depth_embedding.depths`
order — `DepthEmbeddingConfig.depths`, which Phase 16 already requires
to be the ascending, de-duplicated list a config was built with
(`DEFAULT_DEPTHS` by default, matching `configs/base.yaml`'s `depths:`).
Column `i` of the output is always the anomaly at `decoder.depths[i]`;
nothing here re-sorts, shuffles, or otherwise permutes the table's row
order — the queries are built from `DepthEmbedding.all_embeddings()`,
which is itself `self.forward(arange(num_depths))` (see
`depth_embedding.py`), so index `i` in is index `i` out, throughout.

torch is optional
------------------
Same lazy pattern as `encoder.py` / `vit.py` / `depth_embedding.py`:
importing this module and building a `DepthDecoderConfig` never needs
torch; only instantiating a `DepthDecoder` raises
`MissingDependencyError` when it is absent.

Optional uncertainty (Phase 23)
--------------------------------
With `DepthDecoderConfig(uncertainty_enabled=True)` (config:
`uncertainty.enabled` in `configs/base.yaml`, default `false`) the
decoder additionally builds a `log_var_head`, a second
`Linear(embed_dim, 1)` reading the same shared trunk as the mean
`head`. `forward_with_uncertainty` then returns `(mean, log_variance)`
instead of just `mean`; the standard deviation is
`sigma = log_variance_to_sigma(log_variance) = exp(0.5 * log_variance)`.
Train against it with `src.training.losses.gaussian_nll_loss`. `forward`
itself is completely unchanged by this flag — it always returns the
mean only — so nothing about the disabled (default) path changes.
Note: enabling this predicts *a* variance; it does not make that
variance calibrated. No calibration testing is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.data.loaders.errors import MissingDependencyError
from src.models.depth_embedding import (
    DEFAULT_DEPTH_EMBED_DIM,
    DepthEmbedding,
    DepthEmbeddingConfig,
)

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
            purpose="the NEER depth decoder (src.models.depth_decoder)",
            install_hint="pip install torch",
        )


DEFAULT_DECODER_NUM_HEADS = 8
DEFAULT_DECODER_MLP_RATIO = 2.0
DEFAULT_DECODER_DROPOUT = 0.0


@dataclass(frozen=True)
class DepthDecoderConfig:
    """Configuration for `DepthDecoder`.

    Parameters
    ----------
    embed_dim:
        Width of the spatial embedding this decoder consumes and of
        each depth's query — must match the encoder's
        `get_embedding` output width (256 by default, from
        `CNNViTEncoder`/`configs/base.yaml`'s `model.embedding_dim`).
    num_heads:
        Attention heads in the shared cross-attention block; must
        divide `embed_dim`.
    mlp_ratio:
        Width multiplier for the shared feedforward block's hidden
        layer.
    dropout:
        Applied to attention weights, the feedforward block, and the
        two residual branches.
    depth_config:
        The `DepthEmbeddingConfig` for the shared depth embedding
        table this decoder builds internally. Defaults to
        `DepthEmbeddingConfig(embed_dim=embed_dim)` — the 15 NEER
        target depths (`DEFAULT_DEPTHS`) at this decoder's width.
        Its `embed_dim` must equal this config's `embed_dim`.
    uncertainty_enabled:
        Phase 23 — when `True`, the decoder builds a second shared
        output head (`log_var_head`, same `Linear(embed_dim, 1)` shape
        as the mean `head`) that predicts a per-depth log-variance
        alongside the mean anomaly, reachable via
        `DepthDecoder.forward_with_uncertainty`. `forward` itself is
        unchanged either way — it always returns only the mean
        anomalies, so existing callers see no behavior change.
        Default `False` (config: `uncertainty.enabled` in
        `configs/base.yaml`, wired in by `NEERModel.from_config`); no
        log-variance head is built at all when disabled, so a
        disabled decoder's parameters and state_dict keys are
        identical to before this phase.
    """

    embed_dim: int = DEFAULT_DEPTH_EMBED_DIM
    num_heads: int = DEFAULT_DECODER_NUM_HEADS
    mlp_ratio: float = DEFAULT_DECODER_MLP_RATIO
    dropout: float = DEFAULT_DECODER_DROPOUT
    depth_config: Optional[DepthEmbeddingConfig] = None
    uncertainty_enabled: bool = False

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        depth_config = self.depth_config or DepthEmbeddingConfig(embed_dim=self.embed_dim)
        if depth_config.embed_dim != self.embed_dim:
            raise ValueError(
                f"depth_config.embed_dim ({depth_config.embed_dim}) must match "
                f"this config's embed_dim ({self.embed_dim})"
            )
        object.__setattr__(self, "depth_config", depth_config)

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        return int(round(self.embed_dim * self.mlp_ratio))

    @property
    def num_depths(self) -> int:
        return self.depth_config.num_depths

    @property
    def depths(self):
        return self.depth_config.depths


if _TORCH_AVAILABLE:

    def log_variance_to_sigma(log_variance: "torch.Tensor") -> "torch.Tensor":
        """`sigma = exp(0.5 * log_variance)`, the predicted standard deviation.

        The one place this conversion happens, so every caller (this
        module, `NEERModel.predict_uncertainty`, a training/eval
        script) uses the same formula. Calling this on a log-variance
        that came from a decoder built with `uncertainty_enabled=False`
        is a caller error — such a decoder never produces a
        log-variance tensor in the first place, so there's nothing to
        guard here.

        Parameters
        ----------
        log_variance:
            Any shape; elementwise.

        Returns
        -------
        A tensor of the same shape, sigma (standard deviation) > 0.
        """
        return torch.exp(0.5 * log_variance)

    class _SharedCrossAttention(nn.Module):
        """One multi-head cross-attention, shared across every depth query.

        `num_depths` queries attend into a single per-sample key/value
        token — every query uses the same `q_proj`/`k_proj`/`v_proj`/
        `out_proj` weights, so nothing here is depth-specific except
        the query values passed in.
        """

        def __init__(self, config: DepthDecoderConfig) -> None:
            super().__init__()
            self.config = config
            d = config.embed_dim
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.out_proj = nn.Linear(d, d)

        def forward(self, query: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
            # query: (B, num_depths, embed_dim); context: (B, 1, embed_dim)
            b, n, d = query.shape
            h = self.config.num_heads
            hd = self.config.head_dim

            q = self.q_proj(query).view(b, n, h, hd).transpose(1, 2)  # (B, H, N, hd)
            k = self.k_proj(context).view(b, -1, h, hd).transpose(1, 2)  # (B, H, 1, hd)
            v = self.v_proj(context).view(b, -1, h, hd).transpose(1, 2)  # (B, H, 1, hd)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.config.dropout if self.training else 0.0
            )  # (B, H, N, hd)
            attn_out = attn_out.transpose(1, 2).reshape(b, n, d)
            return self.out_proj(attn_out)

    class DepthDecoder(nn.Module):
        """Shared attention-based decoder: spatial embedding -> 15 depth anomalies.

        See the module docstring for the architecture and why this is
        one shared decoder rather than 15 per-depth ones.

        Examples
        --------
        >>> decoder = DepthDecoder()
        >>> decoder(torch.randn(4, 256)).shape
        torch.Size([4, 15])
        >>> decoder.depths[0], decoder.depths[-1]
        (0.0, 1000.0)
        """

        def __init__(self, config: Optional[DepthDecoderConfig] = None) -> None:
            _require_torch()
            super().__init__()
            self.config = config or DepthDecoderConfig()
            self.depth_embedding = DepthEmbedding(self.config.depth_config)

            d = self.config.embed_dim
            self.attn = _SharedCrossAttention(self.config)
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(
                nn.Linear(d, self.config.mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.mlp_hidden_dim, d),
            )
            self.dropout = nn.Dropout(self.config.dropout)
            self.head = nn.Linear(d, 1)
            # Phase 23 — optional second head, same shape as `head`, only
            # built when uncertainty is enabled. Reads the same shared
            # trunk output as `head` (see `_decode_hidden`); it does not
            # duplicate the attention/MLP block, only the final readout.
            self.log_var_head = nn.Linear(d, 1) if self.config.uncertainty_enabled else None

        @property
        def depths(self):
            """The `num_depths` target depths, in output-column order."""
            return self.depth_embedding.depths

        @property
        def num_depths(self) -> int:
            return self.depth_embedding.num_depths

        @property
        def embed_dim(self) -> int:
            return self.config.embed_dim

        @property
        def uncertainty_enabled(self) -> bool:
            """Whether this decoder was built with a log-variance head (Phase 23)."""
            return self.config.uncertainty_enabled

        def _decode_hidden(self, queries: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
            """Shared attn -> feedforward, given explicit queries. `(B, n_queries, embed_dim)`.

            The common trunk both `_decode` (mean only) and
            `_decode_with_uncertainty` (mean + log-variance) read from,
            so the two output heads see exactly the same representation
            and neither duplicates the attention/MLP computation.
            """
            attended = self.attn(queries, context)
            x = self.norm1(queries + self.dropout(attended))
            x = self.norm2(x + self.dropout(self.mlp(x)))
            return x

        def _decode(self, queries: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
            """Shared attn -> feedforward -> mean readout, given explicit queries.

            Internal: `forward` builds `queries` from the full depth
            table; tests use this directly with a subset of rows to
            check that per-depth outputs don't depend on which other
            depths are present in the same batch of queries. Unchanged
            by Phase 23 — always returns only the mean, never the
            log-variance, regardless of `uncertainty_enabled`.
            """
            x = self._decode_hidden(queries, context)
            return self.head(x).squeeze(-1)  # (B, n_queries)

        def _decode_with_uncertainty(
            self, queries: "torch.Tensor", context: "torch.Tensor"
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            """Shared attn -> feedforward -> mean AND log-variance readout.

            Internal counterpart to `_decode`; only called once
            `forward_with_uncertainty` has already checked
            `uncertainty_enabled`, so `self.log_var_head` is guaranteed
            to exist here.
            """
            x = self._decode_hidden(queries, context)
            mean = self.head(x).squeeze(-1)  # (B, n_queries)
            log_variance = self.log_var_head(x).squeeze(-1)  # (B, n_queries)
            return mean, log_variance

        def _validate_embedding(self, embedding: "torch.Tensor") -> None:
            if embedding.dim() != 2:
                raise ValueError(f"expected a 2D (batch, embed_dim) embedding, got shape {tuple(embedding.shape)}")
            if embedding.shape[-1] != self.config.embed_dim:
                raise ValueError(
                    f"embedding has {embedding.shape[-1]} features, expected {self.config.embed_dim}"
                )

        def forward(self, embedding: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            embedding:
                `(batch, embed_dim)` spatial embedding, e.g. from
                `CNNViTEncoder.get_embedding`.

            Returns
            -------
            `(batch, num_depths)` anomalies, one per `self.depths`, in
            that exact order. Always just the mean anomaly — identical
            whether or not `uncertainty_enabled` is set; use
            `forward_with_uncertainty` to also get the log-variance.
            """
            self._validate_embedding(embedding)

            batch = embedding.shape[0]
            queries = self.depth_embedding.all_embeddings().unsqueeze(0).expand(batch, -1, -1)
            context = embedding.unsqueeze(1)  # (B, 1, embed_dim)
            return self._decode(queries, context)

        def forward_with_uncertainty(
            self, embedding: "torch.Tensor"
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            """Phase 23 — mean anomaly AND log-variance, both `(batch, num_depths)`.

            Only available when this decoder was built with
            `DepthDecoderConfig(uncertainty_enabled=True)`; the feature
            is only exposed when explicitly enabled, per the module
            docstring's default-off contract.

            Parameters
            ----------
            embedding:
                `(batch, embed_dim)` spatial embedding, same contract
                as `forward`.

            Returns
            -------
            `(mean, log_variance)`, each `(batch, num_depths)`, one
            column per `self.depths` in that exact order. Get the
            predicted standard deviation with
            `log_variance_to_sigma(log_variance)` — this method
            deliberately does not do that conversion itself, so a
            caller that wants the loss-friendly log-variance (Gaussian
            NLL is more numerically stable in log space) isn't forced
            through an extra `log(exp(...))` round trip.

            Raises
            ------
            RuntimeError:
                If `uncertainty_enabled` is `False` for this decoder.
            """
            if not self.uncertainty_enabled:
                raise RuntimeError(
                    "forward_with_uncertainty() requires a DepthDecoder built with "
                    "DepthDecoderConfig(uncertainty_enabled=True) (config: uncertainty.enabled "
                    "in configs/base.yaml); this decoder was built with uncertainty disabled, "
                    "so it has no log-variance head to read from. Use forward() for the plain "
                    "mean-only prediction instead."
                )
            self._validate_embedding(embedding)

            batch = embedding.shape[0]
            queries = self.depth_embedding.all_embeddings().unsqueeze(0).expand(batch, -1, -1)
            context = embedding.unsqueeze(1)  # (B, 1, embed_dim)
            return self._decode_with_uncertainty(queries, context)

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"DepthDecoder(embed_dim={self.embed_dim}, num_depths={self.num_depths}, "
                f"num_heads={self.config.num_heads}, uncertainty_enabled={self.uncertainty_enabled})"
            )

else:  # pragma: no cover - exercised only in a torch-less environment

    class DepthDecoder:  # type: ignore[no-redef]
        """Placeholder used when torch is not installed (see `encoder.CNNEncoder`)."""

        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()