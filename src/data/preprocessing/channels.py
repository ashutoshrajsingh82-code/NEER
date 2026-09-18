"""
Phase 08 — the authoritative NEER input-channel order.

Everything upstream of this module can decide *how* a channel is built
(`features.py` derives `time_sin`/`time_cos`/`lat_norm`/`lon_norm`; the
loaders produce the seven raw physical fields). This module decides
something different and more load-bearing: the exact list of channels a
NEER model input tensor contains, and the exact position each one holds.

That has to live in exactly one place. A model is a fixed function of
its input layout — channel 4 has to be `u_current` every time, in
training and at inference, or the weights are silently being applied to
the wrong physical quantity. `TensorAssembler` (`tensors.py`) imports
`NEER_CHANNEL_ORDER` as its default `input_variables`, and any model
code should import it too rather than hold-coding channel positions or
re-deriving them from a dataset's variable order, which is not
guaranteed to be stable.

The eleven channels
--------------------
Three raw surface fields measured directly:
    `sst`, `sss`, `sla`

Two vector fields, as raw components rather than derived speed/direction
— a convolutional model can learn magnitude and direction itself, and
keeping the components avoids the information loss (and the
discontinuity at direction 0/360 degrees) that summarizing them away
would cost:
    `u_current`, `v_current`, `u_wind`, `v_wind`

Two cyclic time features and two normalized position features, built by
`FeatureBuilder` (see `features.py` for why sine/cosine rather than a
raw day number, and why normalized rather than raw degrees):
    `time_sin`, `time_cos`, `lat_norm`, `lon_norm`
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

#: The complete NEER input feature set, in the exact order the model
#: consumes it. This is the one place that ordering is decided; nothing
#: else in NEER should hard-code a channel position or re-derive this
#: list by sorting variable names.
NEER_CHANNEL_ORDER: Tuple[str, ...] = (
    "sst",
    "sss",
    "sla",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
    "time_sin",
    "time_cos",
    "lat_norm",
    "lon_norm",
)

#: `len(NEER_CHANNEL_ORDER)` — the input channel count a NEER model's
#: first layer must accept. Kept as a named constant so model code reads
#: `NEER_N_CHANNELS` instead of a bare `11`.
NEER_N_CHANNELS: int = len(NEER_CHANNEL_ORDER)

#: Human-readable documentation for each channel — for error messages,
#: the model card, and notebooks. Not consulted by any code path that
#: affects tensor contents.
CHANNEL_DESCRIPTIONS: Dict[str, str] = {
    "sst": "Sea surface temperature (degC)",
    "sss": "Sea surface salinity (psu)",
    "sla": "Sea level anomaly (m)",
    "u_current": "Eastward surface current component (m/s)",
    "v_current": "Northward surface current component (m/s)",
    "u_wind": "Eastward surface wind component (m/s)",
    "v_wind": "Northward surface wind component (m/s)",
    "time_sin": "sin(2*pi * day_of_year / period_days) — annual phase",
    "time_cos": "cos(2*pi * day_of_year / period_days) — annual phase",
    "lat_norm": "Latitude scaled to [-1, 1] across the domain",
    "lon_norm": "Longitude scaled to [-1, 1] across the domain",
}

assert set(CHANNEL_DESCRIPTIONS) == set(NEER_CHANNEL_ORDER)  # pragma: no cover


def validate_channel_order(names: Sequence[str]) -> None:
    """Raise `ValueError` unless `names` is exactly `NEER_CHANNEL_ORDER`.

    A same-membership-different-order list is exactly the bug this
    guards against — a model trained on one ordering silently scored (or
    served) against another — so this checks position, not just
    membership.
    """
    given = tuple(names)
    if given != NEER_CHANNEL_ORDER:
        missing = [n for n in NEER_CHANNEL_ORDER if n not in given]
        extra = [n for n in given if n not in NEER_CHANNEL_ORDER]
        detail = []
        if missing:
            detail.append(f"missing: {missing}")
        if extra:
            detail.append(f"unexpected: {extra}")
        if not detail:
            detail.append("same channels, wrong order")
        raise ValueError(
            "input channels do not match the authoritative NEER channel "
            f"order ({'; '.join(detail)}).\n"
            f"  expected: {list(NEER_CHANNEL_ORDER)}\n"
            f"  got:      {list(given)}"
        )
