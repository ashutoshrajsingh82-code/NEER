"""
NEER Backend — FastAPI application entrypoint.

Neural Embedding based Estimation and Reconstruction
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

Phase 29A/29B — core API. The app itself does nothing but wiring: build one
`NEERRepository` at startup (loading whatever data/checkpoint actually
exist on disk, once — see `backend/app/services/repository.py`), attach
it to `app.state`, mount the routers, and translate `NeerApiError`
subclasses into the HTTP responses they declare. No request handling,
validation, or NEER logic lives here — see `backend/app/routers/`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.errors import NeerApiError
from backend.app.routers import (
    data_quality,
    demo,
    dates,
    embedding,
    evaluation,
    explainability,
    health,
    metrics,
    model_info,
    profile,
    reconstruct,
    reconstruct_netcdf,
)
from backend.app.services.repository import NEERRepository

PROJECT_NAME = "NEER"
PROBLEM_ID = "SIH26066"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the one `NEERRepository` this process serves from.

    Reads which environment to load from `NEER_ENVIRONMENT` (falls back
    to `NEERRepository`'s own config-derived default) so the same image
    can be pointed at `demo`/`development`/`production` data without a
    code change. Loading happens exactly once per process — never per
    request — because a checkpoint/tensor-bundle load is slow and
    `InferenceService` is itself designed to be constructed once (see
    `src/inference/__init__.py`).
    """
    app.state.repository = NEERRepository(environment=os.environ.get("NEER_ENVIRONMENT"))
    yield


app = FastAPI(
    title="NEER API",
    description="Neural Embedding based Estimation and Reconstruction — Backend API",
    version="0.29.2",
    lifespan=lifespan,
)

# Local frontend origins by default. Deployments can provide a comma-separated
# `NEER_CORS_ORIGINS` allowlist; wildcard origins are deliberately rejected
# because credentialed CORS must name trusted origins explicitly.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,http://[::1]:3000"
)
CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.environ.get("NEER_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]
if "*" in CORS_ORIGINS:
    raise RuntimeError("NEER_CORS_ORIGINS must list explicit origins; wildcard CORS is not allowed")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


@app.exception_handler(NeerApiError)
async def neer_api_error_handler(request: Request, exc: NeerApiError) -> JSONResponse:
    """Turn every deliberately-raised `NeerApiError` into its declared HTTP status.

    This is the *only* place an `NeerApiError` is caught to produce a
    response — routes and the repository just raise the specific
    subclass (`DateNotFoundError`, `ModelUnavailableError`, ...) and
    this handler reads `status_code`/`body()` off it, so every error
    path returns the same `{"error": ..., "detail": ...}` shape.
    """
    return JSONResponse(status_code=exc.status_code, content=exc.body())


app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(dates.router)
app.include_router(reconstruct.router)
app.include_router(profile.router)
app.include_router(embedding.router)
app.include_router(metrics.router)
app.include_router(evaluation.router)
app.include_router(explainability.router)
app.include_router(data_quality.router)
app.include_router(demo.router)
app.include_router(reconstruct_netcdf.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    """Bare liveness ping distinct from `/health`'s component report."""
    return {"project": PROJECT_NAME, "problem_id": PROBLEM_ID, "status": "ok"}
