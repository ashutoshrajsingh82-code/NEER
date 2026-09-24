"""
Phase 19 — NEER loss functions.

Every tensor a NEER model produces or is trained against is finite by
construction (Phase 08's `TensorAssembler` fills gaps with `fill_value`,
Phase 12's `NEERDataset` documents the same guarantee) — the *mask* is
the only record of which of those finite numbers were real. Any loss
built here must therefore be masked, or it silently trains the model to
reproduce fill values over land and scores it on gaps that were
interpolated from its own inputs in the first place (see
`MODEL_CARD.md`'s "Any loss or metric must be masked").

Tensor convention
------------------
Every function below is agnostic to which axis is which, aside from
`depth_dim` (the target-depth axis) and `spatial_dims` (the `(lat, lon)`
axes, when present) — both configurable so the same functions serve
either shape NEER's tensors come in:

* the full-grid convention `NEERDataset`/`TensorBundle` produce,
  `(batch, depth, lat, lon)` — `depth_dim=-3`, `spatial_dims=(-2, -1)`
  (the defaults everywhere below); or
* a pooled per-sample profile with no spatial axis at all,
  `(batch, depth)` — pass `depth_dim=-1` and `spatial_dims=None`
  (`NEERModel.forward`'s `(batch, num_depths)` output, Phase 18, is
  this shape: it predicts one profile per whole input grid, not one
  per pixel — see `src/models/neer_model.py`).

`mask` is expected to be the same shape as `pred`/`target` (exactly
`batch["mask"]["target"]` from `NEERDataset`, or its equivalent for a
pooled profile) and boolean- or float-valued, True/nonzero where a
cell is ocean, observed, and should contribute to the loss.

Loss components
----------------
* `masked_mse_loss` — the primary reconstruction loss. Always on.
* `vertical_smoothness_loss` — an optional regularizer on the
  *predicted* profile: penalizes large jumps between adjacent depth
  levels, encouraging a physically plausible (not jagged) profile.
  Only the depth pairs both marked valid in `mask` are scored, so a
  land column (never valid) never pulls on this term.
* `gradient_loss` — an optional term that compares the *spatial*
  finite-difference gradients of the prediction to the target's,
  rather than the values themselves, so the model is pushed to
  reproduce fronts/edges rather than a blurred average. Needs
  `spatial_dims`; a no-op (returns zero) when the tensor has none.
* `anomaly_loss` — an optional term that first removes a local
  baseline from both prediction and target (an externally supplied
  climatology, e.g. `MonthlyClimatology`, when given as `baseline`;
  otherwise the mask-weighted spatial mean of the target itself, a
  self-contained fallback) and scores the masked MSE of what is left.
  This rewards getting the *anomaly pattern* right independent of a
  constant local bias, complementing `masked_mse_loss`, which cares
  about absolute values too.

`NEERLoss` combines all four behind one configurable set of weights
(`NEERLossConfig`) and returns every component alongside the weighted
`total`, so a training loop can log each term individually as this
phase asks. A component whose weight is `0.0` (the default for the
three optional terms) is still computed and returned — never skipped —
so logging is consistent whether or not a term is currently driving
gradients; the only thing the weight controls is `total`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from src.data.loaders.errors import MissingDependencyError

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
            purpose="the NEER loss functions (src.training.losses)",
            install_hint="pip install torch",
        )


#: Default depth axis for the `(batch, depth, lat, lon)` convention.
DEFAULT_DEPTH_DIM = -3
#: Default spatial axes for the `(batch, depth, lat, lon)` convention.
DEFAULT_SPATIAL_DIMS: Tuple[int, int] = (-2, -1)
#: Numerical floor for every division-by-valid-count below, so a batch
#: (or a batch/depth slice, for `anomaly_loss`) with zero valid cells
#: yields a zero loss/gradient instead of a NaN from a 0/0 division.
DEFAULT_EPS = 1e-8


def _check_shapes_match(pred: "torch.Tensor", target: "torch.Tensor") -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}"
        )


def _mask_like(mask: Optional["torch.Tensor"], reference: "torch.Tensor") -> "torch.Tensor":
    """`mask` as a float tensor the same shape as `reference`; all-ones if absent."""
    if mask is None:
        return torch.ones_like(reference)
    if mask.shape != reference.shape:
        raise ValueError(
            f"mask must have the same shape as the tensor it masks, got "
            f"{tuple(mask.shape)} vs {tuple(reference.shape)}"
        )
    return mask.to(dtype=reference.dtype)


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim % ndim
    if not 0 <= normalized < ndim:
        raise ValueError(f"dim {dim} is out of range for a {ndim}-D tensor")
    return normalized


def _finite_diff(x: "torch.Tensor", dim: int) -> "torch.Tensor":
    """`x[..., 1:, ...] - x[..., :-1, ...]` along `dim`."""
    size = x.shape[dim]
    upper = torch.narrow(x, dim, 1, size - 1)
    lower = torch.narrow(x, dim, 0, size - 1)
    return upper - lower


# --------------------------------------------------------------------------
# Primary loss
# --------------------------------------------------------------------------


def masked_mse_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
    eps: float = DEFAULT_EPS,
) -> "torch.Tensor":
    """Mean squared error over valid cells only.

    Land and unobserved/gap-filled target cells (`mask` False/zero
    there) contribute nothing to either the loss value or its gradient:
    they are excluded from both the numerator (squared error) and the
    denominator (valid-cell count), rather than merely down-weighted.

    Parameters
    ----------
    pred, target:
        Any matching shape.
    mask:
        Same shape as `pred`/`target`, True/nonzero where a cell should
        contribute. `None` scores every cell (unmasked MSE).
    eps:
        Floor on the valid-cell count so an all-invalid `mask` (or an
        empty tensor) returns `0.0` rather than `nan`.

    Returns
    -------
    A 0-D tensor.
    """
    _require_torch()
    _check_shapes_match(pred, target)
    squared_error = (pred - target) ** 2
    mask_f = _mask_like(mask, squared_error)
    denom = mask_f.sum().clamp_min(eps)
    return (squared_error * mask_f).sum() / denom


# --------------------------------------------------------------------------
# Optional regularizers
# --------------------------------------------------------------------------


def vertical_smoothness_loss(
    pred: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
    depth_dim: int = DEFAULT_DEPTH_DIM,
    eps: float = DEFAULT_EPS,
) -> "torch.Tensor":
    """Penalize jumps between vertically-adjacent predicted depth levels.

    A regularizer on `pred` alone (there is no "true" smoothness to
    match against) — it nudges the predicted profile away from
    unphysical zig-zagging between neighboring depths. Only pairs of
    depths that are *both* marked valid in `mask` are scored, so a land
    column or a column with no subsurface coverage (never valid) cannot
    push this term in either direction.

    Parameters
    ----------
    pred:
        Predicted values; must have at least 2 entries along
        `depth_dim`.
    mask:
        Same shape as `pred`. `None` scores every adjacent pair.
    depth_dim:
        Axis indexing target depth. Default matches
        `(batch, depth, lat, lon)`.
    eps:
        Floor on the valid-pair count.

    Returns
    -------
    A 0-D tensor. `0.0` if `pred` has fewer than 2 levels along
    `depth_dim`, or if `mask` leaves no valid adjacent pair.
    """
    _require_torch()
    depth_dim = _normalize_dim(depth_dim, pred.dim())
    if pred.shape[depth_dim] < 2:
        return pred.new_zeros(())

    diff = _finite_diff(pred, depth_dim)
    squared_diff = diff**2

    if mask is None:
        return squared_diff.mean()

    if mask.shape != pred.shape:
        raise ValueError(
            f"mask must have the same shape as pred, got {tuple(mask.shape)} "
            f"vs {tuple(pred.shape)}"
        )
    mask_upper = torch.narrow(mask, depth_dim, 1, mask.shape[depth_dim] - 1)
    mask_lower = torch.narrow(mask, depth_dim, 0, mask.shape[depth_dim] - 1)
    if mask.dtype == torch.bool:
        pair_mask = (mask_upper & mask_lower).to(dtype=squared_diff.dtype)
    else:
        pair_mask = mask_upper.to(dtype=squared_diff.dtype) * mask_lower.to(
            dtype=squared_diff.dtype
        )
    denom = pair_mask.sum().clamp_min(eps)
    return (squared_diff * pair_mask).sum() / denom


def gradient_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
    spatial_dims: Optional[Tuple[int, ...]] = DEFAULT_SPATIAL_DIMS,
    eps: float = DEFAULT_EPS,
) -> "torch.Tensor":
    """Masked MSE between `pred`'s and `target`'s spatial finite-difference gradients.

    Scores whether the prediction reproduces the target's spatial
    *structure* (fronts, gradients) rather than only its pointwise
    values — a plain masked MSE can be minimized by a blurred average
    that gets every value roughly right while erasing every edge.
    Computed independently along each axis in `spatial_dims` and
    averaged; each axis's gradient pair is masked the same way
    `vertical_smoothness_loss` masks depth pairs, i.e. both endpoints
    of a finite difference must be valid for that difference to count.

    Parameters
    ----------
    pred, target:
        Matching shape.
    mask:
        Same shape as `pred`/`target`. `None` scores every pair.
    spatial_dims:
        Axes to difference along. `None` (or a tensor too small along
        every listed axis) makes this a no-op that returns `0.0` — the
        graceful degradation for the pooled `(batch, depth)` profile
        convention, which has no spatial axis for this term to use.
    eps:
        Floor on each axis's valid-pair count.

    Returns
    -------
    A 0-D tensor.
    """
    _require_torch()
    _check_shapes_match(pred, target)
    if spatial_dims is None:
        return pred.new_zeros(())

    ndim = pred.dim()
    dims = tuple(_normalize_dim(d, ndim) for d in spatial_dims)

    components = []
    for dim in dims:
        if pred.shape[dim] < 2:
            continue
        pred_grad = _finite_diff(pred, dim)
        target_grad = _finite_diff(target, dim)
        if mask is None:
            pair_mask = None
        else:
            mask_upper = torch.narrow(mask, dim, 1, mask.shape[dim] - 1)
            mask_lower = torch.narrow(mask, dim, 0, mask.shape[dim] - 1)
            pair_mask = (
                (mask_upper & mask_lower)
                if mask.dtype == torch.bool
                else mask_upper * mask_lower
            )
        components.append(masked_mse_loss(pred_grad, target_grad, pair_mask, eps=eps))

    if not components:
        return pred.new_zeros(())
    return torch.stack(components).mean()


def anomaly_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
    baseline: Optional["torch.Tensor"] = None,
    spatial_dims: Optional[Tuple[int, ...]] = DEFAULT_SPATIAL_DIMS,
    eps: float = DEFAULT_EPS,
) -> "torch.Tensor":
    """Masked MSE on the anomaly (baseline-removed) fields, not the raw values.

    Subtracts a local baseline from both `pred` and `target` before
    scoring, so the loss rewards reproducing the *anomaly pattern*
    rather than absolute magnitude — complementary to
    `masked_mse_loss`, which is sensitive to both.

    Parameters
    ----------
    pred, target:
        Matching shape.
    mask:
        Same shape as `pred`/`target`; also determines which cells
        contribute to the mask-weighted baseline mean when `baseline`
        is not supplied. `None` uses/scores every cell.
    baseline:
        The climatological baseline to subtract (e.g. from
        `src.data.preprocessing.climatology.MonthlyClimatology`),
        broadcastable to `target`'s shape. When given, this *is* the
        anomaly reference — `pred - baseline` is compared against
        `target - baseline`, matching `NEERModel`'s own
        `predicted_delta_T = pred`, `true_delta_T = target - baseline`
        relationship (Phase 18's `predict_profile`).

        When omitted (the default), a self-contained fallback is used
        instead: `pred` and `target` are each demeaned by their *own*
        mask-weighted mean over `spatial_dims` (computed per remaining
        index, e.g. per batch/depth) before the masked MSE is taken.
        This needs no external climatology, makes the term invariant
        to a constant local bias in either field (the primary
        `masked_mse_loss` term already penalizes that), and rewards
        matching the target's spatial *pattern* — at the cost of being
        a per-sample local mean rather than a true multi-year
        climatology.
    spatial_dims:
        Axes to average `target` over when computing the fallback
        baseline. Ignored when `baseline` is given. `None` averages
        over every axis (one scalar baseline for the whole tensor) —
        the right choice for a pooled `(batch, depth)` profile, which
        has no spatial axis to average over instead.
    eps:
        Floor on the valid-cell count used for the fallback baseline's
        mean and for the final masked MSE.

    Returns
    -------
    A 0-D tensor.
    """
    _require_torch()
    _check_shapes_match(pred, target)

    if baseline is not None:
        baseline = torch.as_tensor(baseline, dtype=target.dtype, device=target.device)
        pred_anomaly = pred - baseline
        target_anomaly = target - baseline
    else:
        mask_f = _mask_like(mask, target)
        dims = tuple(range(target.dim())) if spatial_dims is None else tuple(
            _normalize_dim(d, target.dim()) for d in spatial_dims
        )
        denom = mask_f.sum(dim=dims, keepdim=True).clamp_min(eps)
        target_mean = (target * mask_f).sum(dim=dims, keepdim=True) / denom
        pred_mean = (pred * mask_f).sum(dim=dims, keepdim=True) / denom
        pred_anomaly = pred - pred_mean
        target_anomaly = target - target_mean

    return masked_mse_loss(pred_anomaly, target_anomaly, mask, eps=eps)


# --------------------------------------------------------------------------
# Composite, configurable loss
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NEERLossConfig:
    """Configurable weights and axis conventions for `NEERLoss`.

    The masked-MSE term is always computed and always contributes to
    `total` (its weight cannot be zero — a NEER loss with nothing
    scoring the actual reconstruction is not a useful configuration).
    The three optional terms default to weight `0.0` (computed and
    returned for logging, but not contributing to `total`) and are
    enabled by giving them a positive weight.
    """

    mse_weight: float = 1.0
    smoothness_weight: float = 0.0
    gradient_weight: float = 0.0
    anomaly_weight: float = 0.0
    depth_dim: int = DEFAULT_DEPTH_DIM
    spatial_dims: Optional[Tuple[int, ...]] = DEFAULT_SPATIAL_DIMS
    eps: float = DEFAULT_EPS

    def __post_init__(self) -> None:
        if self.mse_weight <= 0:
            raise ValueError(f"mse_weight must be positive, got {self.mse_weight}")
        for name in ("smoothness_weight", "gradient_weight", "anomaly_weight"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.spatial_dims is not None and len(self.spatial_dims) == 0:
            raise ValueError("spatial_dims must be None or a non-empty tuple of axes")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    def replace(self, **overrides: object) -> "NEERLossConfig":
        """A copy of this config with `overrides` applied."""
        return replace(self, **overrides)


if _TORCH_AVAILABLE:

    class NEERLoss(nn.Module):
        """The combined NEER training loss: masked MSE plus optional terms.

        Wraps `masked_mse_loss`, `vertical_smoothness_loss`,
        `gradient_loss` and `anomaly_loss` behind one `NEERLossConfig`
        of weights, and returns every component — including the ones
        currently weighted at `0.0` — alongside the weighted `total`,
        so a training loop can log each term without extra bookkeeping.

        Examples
        --------
        >>> config = NEERLossConfig(smoothness_weight=0.1, gradient_weight=0.05)
        >>> criterion = NEERLoss(config)
        >>> losses = criterion(pred, target, mask)
        >>> losses["total"].backward()
        >>> losses.keys()
        dict_keys(['mse', 'smoothness', 'gradient', 'anomaly', 'total'])
        """

        def __init__(self, config: Optional[NEERLossConfig] = None) -> None:
            super().__init__()
            self.config = config or NEERLossConfig()

        def forward(
            self,
            pred: "torch.Tensor",
            target: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            """
            Parameters
            ----------
            pred, target:
                Matching shape — see the module docstring for the two
                shape conventions this supports.
            mask:
                Same shape as `pred`/`target`; True/nonzero where a
                cell is valid (ocean, observed) and should contribute.
                `None` scores every cell, i.e. no masking.

            Returns
            -------
            `{"mse", "smoothness", "gradient", "anomaly", "total"}`,
            each a 0-D tensor. `total` is the weighted sum per
            `self.config`; the other four are each unweighted, so they
            are directly comparable across runs with different weights.
            """
            cfg = self.config
            components: Dict[str, "torch.Tensor"] = {
                "mse": masked_mse_loss(pred, target, mask, eps=cfg.eps),
                "smoothness": vertical_smoothness_loss(
                    pred, mask, depth_dim=cfg.depth_dim, eps=cfg.eps
                ),
                "gradient": gradient_loss(
                    pred, target, mask, spatial_dims=cfg.spatial_dims, eps=cfg.eps
                ),
                "anomaly": anomaly_loss(
                    pred, target, mask, spatial_dims=cfg.spatial_dims, eps=cfg.eps
                ),
            }
            components["total"] = (
                cfg.mse_weight * components["mse"]
                + cfg.smoothness_weight * components["smoothness"]
                + cfg.gradient_weight * components["gradient"]
                + cfg.anomaly_weight * components["anomaly"]
            )
            return components