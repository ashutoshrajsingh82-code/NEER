"""
Phase 28 — the NEER inference service.

Package: src/inference
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

Every earlier ML phase built a piece of the pipeline (encoder, decoder,
climatology, the composed `NEERModel`). This phase wraps that pipeline
in one small, reusable, *serving*-shaped object — `InferenceService`
(`src/inference/service.py`) — the thing a backend route, a batch job,
or a notebook actually calls:

* the model (and, when supplied, the climatology) is loaded exactly
  once, not once per request;
* every forward pass runs under `torch.inference_mode()`;
* CPU/GPU is auto-detected once at construction, never re-probed per
  call;
* identical requests are served from a small in-memory cache instead
  of re-running the network;
* every call reports how long it took.

Three request shapes, one shared forward pass
------------------------------------------------
`NEERModel.forward` always does the same thing — one surface-field
tensor in, one pooled `(embed_dim,)` embedding and one `(num_depths,)`
anomaly profile out (see `src/models/neer_model.py`; the pipeline is
domain-pooled, not per-pixel — `src/argo_validation/predictions.py`
documents the same limitation this service inherits). `predict_point`,
`predict_profile` and `predict_grid` are three *views* onto that one
result, not three different models:

* `predict_point`   — the profile, sliced to the nearest requested depth.
* `predict_profile` — the full depth profile at one location.
* `predict_grid`    — the same domain-pooled anomaly combined with a
  per-cell climatology across many locations, so the returned field
  still varies spatially through its climatological term even though
  the model itself predicts one anomaly per timestep.
"""

from src.inference.service import InferenceService, PredictionResult

__all__ = ["InferenceService", "PredictionResult"]