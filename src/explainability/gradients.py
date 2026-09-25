"""
Phase 29B-2 — gradient-based explainability for `NEERModel`.

`GET /explainability` needs *real* explainability, not a fabricated or
hard-coded importance score. This module computes standard gradient-based
feature attribution (Simonyan et al., 2013 "saliency maps"; the
gradient-x-input variant is Shrikumar et al., 2016) from one genuine
forward+backward pass through the actual trained `NEERModel`, on the
actual input tensor NEER already built for a real date
(`NEERRepository._input_for_date`) — the same tensor `/reconstruct` and
`/embedding` feed the model.

Method
------
For a chosen scalar model output `y` (either one depth level's predicted
anomaly, or the sum of every depth level's anomaly when no depth is
requested — "explain the whole profile"), the gradient
`dy/dx` is obtained via `torch.autograd.grad`. Two things are read off
that real gradient tensor, `x` being `(channel, lat, lon)`:

* **Per-channel importance** — `mean(|dy/dx * x|)` over the spatial axes
  for each channel, normalized across channels to sum to 1. This is the
  standard "gradient x input" attribution: it answers "how much did each
  of NEER's 11 input channels (`sst`, `sss`, `u_current`, ...) actually
  drive this prediction", using the sign and magnitude of the real
  gradient at the real input values that produced this specific
  prediction, not a static/global weight.
* **Spatial saliency** — `mean(|dy/dx|)` over the channel axis, giving a
  `(lat, lon)` map of which grid cells the prediction is most sensitive
  to. Downsampled (nearest-index subsampling, never interpolated or
  synthesized) when the grid is larger than `max_saliency_dim` per axis,
  purely to keep the API response a sane size — the values returned are
  still genuine gradient magnitudes at genuine grid points.

Nothing here is random, hard-coded, or a placeholder: every number
returned is derived from `model`'s actual parameters and `x`'s actual
values for the requested date.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing._utils import json_safe
from src.data.preprocessing.channels import CHANNEL_DESCRIPTIONS

try:  # pragma: no cover - environment dependent
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

#: Attribution never claims to be a calibrated feature-importance score —
#: it is what it is: a local, first-order sensitivity measurement at one
#: real input. This name is what `notes` on every response calls it.
METHOD_NAME = "gradient_x_input"

#: Per-axis cap on the spatial saliency map returned to the API — a
#: production grid can be far larger than makes sense to inline in a JSON
#: response. Real gradient values at a nearest-index subsample of the
#: real grid, never averaged/interpolated/synthesized.
DEFAULT_MAX_SALIENCY_DIM = 48


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="gradient-based explainability (src.explainability.gradients)",
            install_hint="pip install torch",
        )


def _downsample_2d(array: np.ndarray, max_dim: int) -> np.ndarray:
    """Nearest-index subsample of a 2D array to at most `max_dim` per axis.

    Deliberately *not* an average-pool or interpolation: every returned
    cell is a real gradient value at a real grid point, just a subset of
    them, so the saliency map never mixes in a synthesized value.
    """
    h, w = array.shape
    if h <= max_dim and w <= max_dim:
        return array
    row_idx = np.unique(np.linspace(0, h - 1, min(h, max_dim)).astype(int))
    col_idx = np.unique(np.linspace(0, w - 1, min(w, max_dim)).astype(int))
    return array[np.ix_(row_idx, col_idx)]


def explain_point(
    model: "torch.nn.Module",
    x: Any,
    *,
    channel_names: Sequence[str],
    depths: Sequence[float],
    depth_index: Optional[int] = None,
    max_saliency_dim: int = DEFAULT_MAX_SALIENCY_DIM,
) -> Dict[str, Any]:
    """Gradient-based attribution for one real NEER input sample.

    Parameters
    ----------
    model:
        The actual, already-trained `NEERModel` (or any module exposing
        the same `encoder.get_embedding` / `decoder` contract).
    x:
        `(channel, lat, lon)` or `(1, channel, lat, lon)` — the actual
        input tensor for one date, exactly what `InferenceService`
        forward-passes.
    channel_names:
        Names for `x`'s channel axis, in order (`TensorBundle.channel_names`
        / `NEER_CHANNEL_ORDER`).
    depths:
        The model's depth levels (`InferenceService.depths` /
        `model.depths`), used to label a specific-depth explanation.
    depth_index:
        Which of the model's depth outputs to explain. `None` explains
        the sum of every depth level's anomaly (a whole-profile
        explanation) instead of one level.
    max_saliency_dim:
        See `_downsample_2d`.

    Returns
    -------
    A JSON-safe dict: `predicted_anomaly`, `depth`/`depth_index`
    (`None` when aggregated), `aggregated_over_depths`, per-channel
    `channels` (`name`, `description`, `importance`, `mean_gradient`),
    `spatial_saliency` (+ its shape), `embedding_dim`, `method`, `notes`.

    Raises
    ------
    MissingDependencyError:
        torch is not installed.
    ValueError:
        `x`'s shape is not one sample, or `depth_index` is out of range.
    """
    _require_torch()

    was_training = model.training
    model.eval()
    try:
        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim == 3:
            x_np = x_np[None, ...]
        if x_np.ndim != 4 or x_np.shape[0] != 1:
            raise ValueError(
                "explain_point expects one sample — (channel, lat, lon) or "
                f"(1, channel, lat, lon) — got shape {x_np.shape}"
            )
        if x_np.shape[1] != len(channel_names):
            raise ValueError(
                f"x has {x_np.shape[1]} channels but {len(channel_names)} "
                "channel_names were given"
            )

        device = next(model.parameters()).device
        x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        x_t.requires_grad_(True)

        embedding = model.encoder.get_embedding(x_t)
        anomalies = model.decoder(embedding)  # (1, num_depths)
        num_depths = int(anomalies.shape[1])

        aggregated = depth_index is None
        if aggregated:
            target = anomalies[0].sum()
            depth_value: Optional[float] = None
        else:
            if not (0 <= depth_index < num_depths):
                raise ValueError(
                    f"depth_index {depth_index} out of range for {num_depths} depth levels"
                )
            target = anomalies[0, depth_index]
            depth_value = float(depths[depth_index])

        (grad,) = torch.autograd.grad(target, x_t, retain_graph=False)

        grad_np = grad.detach().to("cpu").numpy()[0]  # (channel, lat, lon)
        x_input_np = x_t.detach().to("cpu").numpy()[0]  # (channel, lat, lon)
        predicted_value = float(target.detach().to("cpu").item())
    finally:
        if was_training:
            model.train()

    attribution = grad_np * x_input_np  # gradient x input, per element
    channel_importance_raw = np.abs(attribution).mean(axis=(1, 2))  # (channel,)
    total = float(channel_importance_raw.sum())
    channel_importance = (
        (channel_importance_raw / total) if total > 0 else channel_importance_raw
    )
    channel_gradient_mean = grad_np.mean(axis=(1, 2))  # (channel,) signed

    spatial = np.abs(grad_np).mean(axis=0)  # (lat, lon)
    spatial_ds = _downsample_2d(spatial, max_saliency_dim)

    channels: List[Dict[str, Any]] = [
        {
            "name": name,
            "description": CHANNEL_DESCRIPTIONS.get(name),
            "importance": float(channel_importance[i]),
            "mean_gradient": float(channel_gradient_mean[i]),
        }
        for i, name in enumerate(channel_names)
    ]

    notes = [
        "importance is |gradient x input| per channel, averaged over the spatial "
        "axes and normalized across channels to sum to 1 (gradient-x-input "
        "attribution, Shrikumar et al. 2016)",
        "spatial_saliency is mean(|gradient|) over channels, from a real "
        "backward pass through the actual trained model at the actual input "
        "for this date"
        + ("" if spatial.shape == spatial_ds.shape else "; subsampled for response size"),
    ]
    if aggregated:
        notes.append(
            "no depth was requested: this explains the sum of the predicted "
            "anomaly across every model depth level (a whole-profile explanation)"
        )

    return json_safe(
        {
            "predicted_anomaly": predicted_value,
            "depth": depth_value,
            "depth_index": None if aggregated else int(depth_index),
            "aggregated_over_depths": aggregated,
            "method": METHOD_NAME,
            "channels": channels,
            "spatial_saliency": spatial_ds.tolist(),
            "spatial_saliency_shape": list(spatial_ds.shape),
            "full_grid_shape": list(spatial.shape),
            "embedding_dim": int(embedding.shape[-1]),
            "notes": notes,
        }
    )