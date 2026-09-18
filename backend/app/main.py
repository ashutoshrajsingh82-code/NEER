"""
NEER Backend — FastAPI application entrypoint.

Neural Embedding based Estimation and Reconstruction
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

This is a minimal skeleton for Phase 01. No ML endpoints are exposed yet.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_NAME = "NEER"
PROBLEM_ID = "SIH26066"

app = FastAPI(
    title="NEER API",
    description="Neural Embedding based Estimation and Reconstruction — Backend API",
    version="0.1.0",
)

# Permissive CORS for local development with the Next.js frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Basic liveness/health check endpoint."""
    return {
        "status": "ok",
        "project": PROJECT_NAME,
        "problem_id": PROBLEM_ID,
    }
