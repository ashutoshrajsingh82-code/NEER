"""
NEER - Neural Embedding based Estimation and Reconstruction

Package: src/argo_validation
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

Vertical interpolation of a cleaned ARGO profile onto NEER's model depth
levels. Kept deliberately conservative: exact observed levels are always
used, interior gaps are only bridged up to a configurable tolerance, and
the profile is never extrapolated beyond its observed range (aside from a
small edge tolerance for filling the very first/last requested depth from
the nearest observed level).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["VerticalConfig", "interpolate_to_depths"]


@dataclass(frozen=True)
class VerticalConfig:
    """Controls how much vertical gap ``interpolate_to_depths`` may bridge.

    gap_floor_m:
        Minimum bracket size (metres) between two observed levels that
        interpolation is always allowed to span.
    gap_rel:
        Extra allowance, as a fraction of the *target* depth, added on
        top of ``gap_floor_m`` for deeper (typically more sparsely
        sampled) target depths.
    edge_tolerance_m:
        How far (metres) beyond the shallowest/deepest observed level a
        target depth may sit and still be filled — with that nearest
        observed value, not a linear extrapolation.
    """

    gap_floor_m: float = 25.0
    gap_rel: float = 0.25
    edge_tolerance_m: float = 10.0


def interpolate_to_depths(
    depth_m: np.ndarray,
    temperature_c: np.ndarray,
    target_depths: np.ndarray,
    config: VerticalConfig = VerticalConfig(),
) -> np.ndarray:
    """Interpolate one cleaned ARGO profile onto ``target_depths``.

    ``depth_m`` must be sorted ascending with no duplicate levels (this is
    what ``profiles.clean_profile`` guarantees). Returns a float array the
    same shape as ``target_depths``; entries are ``np.nan`` wherever no
    eligible estimate exists.

    Rules (see module docstring):
      * A target depth equal to an observed depth is always used, no
        matter how large the surrounding gaps are.
      * A target depth strictly between two observed levels is filled by
        linear interpolation only if the bracket ``d_hi - d_lo`` is no
        larger than ``max(config.gap_floor_m, config.gap_rel * target_depth)``.
        Otherwise it is left unfilled (NaN).
      * A target depth outside ``[depth_m[0], depth_m[-1]]`` is never
        extrapolated, except that one within ``config.edge_tolerance_m``
        of the nearest observed level is filled with that level's value.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    temperature_c = np.asarray(temperature_c, dtype=float)
    target_depths = np.asarray(target_depths, dtype=float)

    out = np.full(target_depths.shape, np.nan, dtype=float)
    if depth_m.size == 0:
        return out

    shallowest, deepest = depth_m[0], depth_m[-1]

    for idx, z in enumerate(target_depths):
        # Exact hit on an observed level: always accepted.
        exact = np.nonzero(depth_m == z)[0]
        if exact.size:
            out[idx] = temperature_c[exact[0]]
            continue

        # Above the shallowest observed level.
        if z < shallowest:
            if shallowest - z <= config.edge_tolerance_m:
                out[idx] = temperature_c[0]
            continue

        # Below the deepest observed level.
        if z > deepest:
            if z - deepest <= config.edge_tolerance_m:
                out[idx] = temperature_c[-1]
            continue

        # Interior: bracket between two observed levels.
        hi_idx = int(np.searchsorted(depth_m, z))
        lo_idx = hi_idx - 1
        d_lo, d_hi = depth_m[lo_idx], depth_m[hi_idx]
        gap = d_hi - d_lo
        threshold = max(config.gap_floor_m, config.gap_rel * z)
        if gap > threshold:
            continue

        t_lo, t_hi = temperature_c[lo_idx], temperature_c[hi_idx]
        frac = (z - d_lo) / gap
        out[idx] = t_lo + frac * (t_hi - t_lo)

    return out