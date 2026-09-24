"""NEER - Neural Embedding based Estimation and Reconstruction

Package: src/models
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

The input contract every model in this package consumes is fixed:
`(batch, NEER_N_CHANNELS, lat, lon)` with channels in
`NEER_CHANNEL_ORDER`, re-exported here so model code never has to
hard-code `11` or restate the channel list. The one authoritative
definition lives in `src.data.preprocessing.channels`; this is just a
convenience import.

Phase 13 adds the first actual architecture piece: `CNNEncoder`, a
lightweight CPU-friendly convolutional spatial encoder (see
`src/models/encoder.py`). It turns the raw `(batch, 11, lat, lon)` input
into a `(batch, C_out, lat, lon)` spatial feature map. A patch-tokenizing
stage (ViT) is explicitly out of scope for this phase — see
`MODEL_CARD.md`.

Phase 14 adds the lightweight Vision Transformer (`src/models/vit.py`):
`VisionTransformer` patch-tokenizes a CNN feature map and runs a small
transformer over the tokens, and `CNNViTEncoder` composes it after the
`CNNEncoder`: `(batch, 11, lat, lon) -> (batch, n_tokens, embed_dim)`.

Phase 15 adds `get_embedding(x) -> (batch, embed_dim)` to both
`VisionTransformer` and `CNNViTEncoder`: the model's real representation
(a mean pool of its own `forward()` tokens), not a separate embedding
path.

Phase 16 adds `DepthEmbedding` (`src/models/depth_embedding.py`): one
shared, learned lookup table — a single `(15, embed_dim)` weight
matrix — for NEER's 15 target depth levels (`DEFAULT_DEPTHS`), not 15
independent per-depth networks. A future depth decoder combines a row
of this table with a `CNNViTEncoder.get_embedding` spatial embedding.

Phase 17 adds `DepthDecoder` (`src/models/depth_decoder.py`): the
depth decoder Phase 16 anticipated. One shared attention block — the
15 depth embeddings cross-attend into a single spatial embedding —
plus a shared feedforward block and readout, predicting all 15 depth
anomalies from one `(batch, embed_dim)` embedding in a single forward
pass: `(batch, embed_dim) -> (batch, 15)`, in `DepthDecoder.depths`
order.

Phase 18 adds `NEERModel` (`src/models/neer_model.py`): the complete
pipeline, composing a `CNNViTEncoder` and a `DepthDecoder` under one
`torch.nn.Module`. `forward` runs surface fields through to 15
predicted anomalies; `get_embedding` exposes the Phase 15 spatial
embedding at the whole-model level; `predict_profile` adds a supplied
climatology baseline (Phase 11, `src.data.preprocessing.climatology`)
to reconstruct an absolute temperature profile. No new learnable
weights of its own, and no training — see the module docstring.

Phase 22 adds an optional graph refinement (`src/models/gnn.py`):
`GridGNN` treats every grid cell as a node and every pair of neighbouring
cells as edges, and refines the CNN's per-cell feature map before the
ViT. It is switched by `model.use_gnn` (default `false`) via
`NEERModel(use_gnn=...)` / `NEERModel.from_config(config)`; when off, no
GNN module exists. With `GNNConfig.use_current` the `u_current` /
`v_current` input channels condition the messages themselves.
"""

from src.data.preprocessing.channels import (
    CHANNEL_DESCRIPTIONS,
    NEER_CHANNEL_ORDER,
    NEER_N_CHANNELS,
    validate_channel_order,
)
from src.models.depth_decoder import (
    DEFAULT_DECODER_DROPOUT,
    DEFAULT_DECODER_MLP_RATIO,
    DEFAULT_DECODER_NUM_HEADS,
    DepthDecoder,
    DepthDecoderConfig,
)
from src.models.depth_embedding import (
    DEFAULT_DEPTH_EMBED_DIM,
    DEFAULT_DEPTHS,
    DepthEmbedding,
    DepthEmbeddingConfig,
)
from src.models.encoder import (
    DEFAULT_CHANNELS,
    VALID_ACTIVATIONS,
    VALID_NORMS,
    CNNEncoder,
    CNNEncoderConfig,
)
from src.models.gnn import (
    DEFAULT_CURRENT_CHANNELS,
    VALID_CONNECTIVITIES,
    GNNConfig,
    GridGNN,
    GridGraph,
    build_grid_edges,
    build_grid_graph,
)
from src.models.neer_model import NEERModel
from src.models.pretrain_decoder import (
    DEFAULT_DECODER_CHANNELS,
    ReconstructionDecoder,
    ReconstructionDecoderConfig,
)
from src.models.pretrain_encoder import PretrainEncoder, PretrainEncoderConfig
from src.models.pretrain_reconstruction import (
    MaskedReconstructionLoss,
    MaskedReconstructionLossConfig,
    PretrainOutput,
    PretrainReconstructionConfig,
    PretrainReconstructionModel,
    masked_reconstruction_loss,
    masked_reconstruction_mse,
)
from src.models.vit import (
    DEFAULT_EMBED_DIM,
    DEFAULT_PATCH_SIZE,
    VALID_PAD_MODES,
    CNNViTEncoder,
    ViTConfig,
    VisionTransformer,
)

__all__ = [
    "NEER_CHANNEL_ORDER",
    "NEER_N_CHANNELS",
    "CHANNEL_DESCRIPTIONS",
    "validate_channel_order",
    "CNNEncoder",
    "CNNEncoderConfig",
    "DEFAULT_CHANNELS",
    "VALID_NORMS",
    "VALID_ACTIVATIONS",
    "ViTConfig",
    "VisionTransformer",
    "CNNViTEncoder",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_PATCH_SIZE",
    "VALID_PAD_MODES",
    "DepthEmbedding",
    "DepthEmbeddingConfig",
    "DEFAULT_DEPTHS",
    "DEFAULT_DEPTH_EMBED_DIM",
    "DepthDecoder",
    "DepthDecoderConfig",
    "DEFAULT_DECODER_NUM_HEADS",
    "DEFAULT_DECODER_MLP_RATIO",
    "DEFAULT_DECODER_DROPOUT",
    "NEERModel",
    "GNNConfig",
    "GridGNN",
    "GridGraph",
    "build_grid_edges",
    "build_grid_graph",
    "VALID_CONNECTIVITIES",
    "DEFAULT_CURRENT_CHANNELS",
    "PretrainEncoder",
    "PretrainEncoderConfig",
    "ReconstructionDecoder",
    "ReconstructionDecoderConfig",
    "DEFAULT_DECODER_CHANNELS",
    "PretrainReconstructionModel",
    "PretrainReconstructionConfig",
    "PretrainOutput",
    "MaskedReconstructionLoss",
    "MaskedReconstructionLossConfig",
    "masked_reconstruction_loss",
    "masked_reconstruction_mse",
]