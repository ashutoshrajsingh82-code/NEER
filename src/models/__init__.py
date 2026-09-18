"""NEER - Neural Embedding based Estimation and Reconstruction

Package: src/models
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

No architecture lives here yet (see `MODEL_CARD.md`) — but the input
contract it will consume is already fixed. Any model built in this
package must accept `(batch, NEER_N_CHANNELS, lat, lon)` with channels
in `NEER_CHANNEL_ORDER`, re-exported here so model code never has to
hard-code `11` or restate the channel list. The one authoritative
definition lives in `src.data.preprocessing.channels`; this is just a
convenience import.
"""

from src.data.preprocessing.channels import (
    CHANNEL_DESCRIPTIONS,
    NEER_CHANNEL_ORDER,
    NEER_N_CHANNELS,
    validate_channel_order,
)

__all__ = [
    "NEER_CHANNEL_ORDER",
    "NEER_N_CHANNELS",
    "CHANNEL_DESCRIPTIONS",
    "validate_channel_order",
]
