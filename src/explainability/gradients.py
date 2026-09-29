"""
Integrated Gradients for one real NEER model output and normalized input.
The baseline zeros the seven z-score normalized physical fields (the
training-mean reference and TensorAssembler fill value) while retaining
the actual date and position context channels. Feature attributions are
signed spatial sums for each physical channel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.data.preprocessing._utils import json_safe
from src.data.preprocessing.channels import CHANNEL_DESCRIPTIONS, NEER_CHANNEL_ORDER

try:  # pragma: no cover - environment dependent
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

#: Attribution never claims to be a calibrated feature-importance score —
#: it is what it is: a local, first-order sensitivity measurement at one
#: real input. This name is what `notes` on every response calls it.
METHOD_NAME = "Integrated Gradients"
DEFAULT_INTEGRATION_STEPS = 64

def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch",
            purpose="gradient-based explainability (src.explainability.gradients)",
            install_hint="pip install torch",
        )


def explain_point(
    model: "torch.nn.Module",
    x: Any,
    *,
    channel_names: Sequence[str],
    depths: Sequence[float],
    depth_index: int,
    output_center: float = 0.0,
    output_scale: float = 1.0,
    steps: int = DEFAULT_INTEGRATION_STEPS,
) -> Dict[str, Any]:
    """Compute signed Integrated Gradients for one selected-depth output.

    `x` is the production-preprocessed tensor consumed by NEER. The
    decoder's normalized output is converted to physical units using the
    same per-depth center/scale as InferenceService before gradients are
    integrated. Seven physical channels are spatially summed; four
    temporal/geographic context channels are returned separately.
    """
    _require_torch()
    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two integration steps")
    if tuple(channel_names) != NEER_CHANNEL_ORDER:
        raise ValueError("channel_names must match the authoritative NEER input-channel order")
    if not np.isfinite(output_center) or not np.isfinite(output_scale) or output_scale <= 0:
        raise ValueError("output normalization must be finite with a positive scale")

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
        if not torch.isfinite(x_t).all():
            raise ValueError("model input contains non-finite values")
        baseline = x_t.detach().clone()
        # Physical fields are z-score normalized and TensorAssembler uses
        # zero as the training-mean fill. Retain the real cyclic-time and
        # normalized-position context, so the reference remains conditioned
        # on this request instead of using an impossible all-zero date/grid.
        baseline[:, :7] = 0.0
        with torch.no_grad():
            baseline_embedding = model.encoder.get_embedding(baseline)
            baseline_output = model.decoder(baseline_embedding)
            input_embedding = model.encoder.get_embedding(x_t)
            input_output = model.decoder(input_embedding)
        anomalies = input_output
        num_depths = int(anomalies.shape[1])
        if not (0 <= depth_index < num_depths) or len(depths) != num_depths:
            raise ValueError("requested depth index does not match model output depths")

        # Trapezoidal quadrature over a straight path from the explicitly
        # documented training-mean physical baseline to the exact serving input.
        gradient_sum = torch.zeros_like(x_t)
        for step in range(steps):
            alpha = step / (steps - 1)
            interpolated = (baseline + alpha * (x_t - baseline)).detach().requires_grad_(True)
            embedding = model.encoder.get_embedding(interpolated)
            raw_value = model.decoder(embedding)[0, depth_index]
            physical_value = raw_value * output_scale + output_center
            (gradient,) = torch.autograd.grad(physical_value, interpolated, retain_graph=False)
            weight = 0.5 if step == 0 or step == steps - 1 else 1.0
            gradient_sum += gradient.detach() * weight

        average_gradient = gradient_sum / (steps - 1)
        input_delta = x_t - baseline
        attribution = (input_delta * average_gradient).detach().to("cpu").numpy()[0]
        predicted_value = float((input_output[0, depth_index] * output_scale + output_center).item())
        baseline_value = float((baseline_output[0, depth_index] * output_scale + output_center).item())
    finally:
        if was_training:
            model.train()
    physical_names = NEER_CHANNEL_ORDER[:7]
    context_names = NEER_CHANNEL_ORDER[7:]
    feature_channels = [
        {"name": name, "description": CHANNEL_DESCRIPTIONS[name], "attribution": float(attribution[index].sum())}
        for index, name in enumerate(physical_names)
    ]
    context_attributions = [
        {"name": name, "description": CHANNEL_DESCRIPTIONS[name], "attribution": float(attribution[index].sum())}
        for index, name in enumerate(context_names, start=7)
    ]
    total_attribution = float(attribution.sum())
    output_delta = predicted_value - baseline_value
    return json_safe(
        {
            "predicted_model_output": predicted_value,
            "baseline_model_output": baseline_value,
            "depth": float(depths[depth_index]),
            "depth_index": int(depth_index),
            "method": METHOD_NAME,
            "baseline": "Seven normalized physical input channels set to zero (training-mean reference and TensorBundle fill value); actual date and position context channels are held fixed.",
            "integration_steps": int(steps),
            "features": feature_channels,
            "context_attributions": context_attributions,
            "attribution_sum": total_attribution,
            "output_delta_from_baseline": output_delta,
            "completeness_error": total_attribution - output_delta,
            "embedding_dim": int(input_embedding.shape[-1]),
            "notes": [
                "Integrated Gradients values are signed sums across each input channel's spatial grid; the four time/position context channel attributions are returned separately.",
                "The model decoder is domain-pooled and produces one anomaly residual per date/depth; latitude and longitude select the climatology term for the displayed point temperature.",
                "Attributions describe this input and baseline, not physical causality.",
            ],
        }
    )
