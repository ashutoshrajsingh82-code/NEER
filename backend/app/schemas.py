"""
Phase 29A — Pydantic schemas for the NEER FastAPI layer.

Query-parameter schemas
------------------------
Each endpoint's query parameters are declared as ordinary `fastapi.Query`
arguments (so FastAPI/Pydantic validates type and basic range — e.g.
`ge=-90`  before a route body ever runs, turning a missing or
mistyped parameter into a `422` automatically) and immediately gathered
into a small Pydantic model. Routes therefore receive one validated,
typed object instead of loose scalars, and the semantic checks that
can't be expressed as a `Query(...)` bound (a coordinate outside the
NEER domain, `lat_min >= lat_max`, ...) live in
`backend/app/services/repository.py`, not scattered across routes.

Response schemas
------------------
Mirror `src.inference.service.PredictionResult.to_dict()` and
`NEERRepository`'s dicts field-for-field — nothing here invents a shape
the underlying service doesn't already produce.
"""

from __future__ import annotations

from datetime import date as _Date
from typing import Any, Dict, List, Optional, Union

from fastapi import Query
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Query-parameter schemas
# --------------------------------------------------------------------------


class PointQueryParams(BaseModel):
    lat: float = Field(..., description="Latitude in degrees (within the NEER domain).")
    lon: float = Field(..., description="Longitude in degrees (within the NEER domain).")
    date: _Date = Field(..., description="Date to reconstruct, YYYY-MM-DD.")
    depth: float = Field(..., ge=0, description="Target depth in metres (snapped to the nearest model depth level).")


def point_query_params(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in degrees."),
    lon: float = Query(..., ge=-180, le=360, description="Longitude in degrees."),
    date: _Date = Query(..., description="Date to reconstruct, YYYY-MM-DD."),
    depth: float = Query(..., ge=0, description="Target depth in metres."),
) -> PointQueryParams:
    return PointQueryParams(lat=lat, lon=lon, date=date, depth=depth)


class ProfileQueryParams(BaseModel):
    lat: float = Field(..., description="Latitude in degrees (within the NEER domain).")
    lon: float = Field(..., description="Longitude in degrees (within the NEER domain).")
    date: _Date = Field(..., description="Date to reconstruct, YYYY-MM-DD.")


def profile_query_params(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in degrees."),
    lon: float = Query(..., ge=-180, le=360, description="Longitude in degrees."),
    date: _Date = Query(..., description="Date to reconstruct, YYYY-MM-DD."),
) -> ProfileQueryParams:
    return ProfileQueryParams(lat=lat, lon=lon, date=date)


class GridQueryParams(BaseModel):
    lat_min: float = Field(..., description="Southern bound of the requested region.")
    lat_max: float = Field(..., description="Northern bound of the requested region.")
    lon_min: float = Field(..., description="Western bound of the requested region.")
    lon_max: float = Field(..., description="Eastern bound of the requested region.")
    date: _Date = Field(..., description="Date to reconstruct, YYYY-MM-DD.")
    depth: Optional[float] = Field(
        None, ge=0, description="Single target depth in metres; omit for every model depth level."
    )


def grid_query_params(
    lat_min: float = Query(..., ge=-90, le=90),
    lat_max: float = Query(..., ge=-90, le=90),
    lon_min: float = Query(..., ge=-180, le=360),
    lon_max: float = Query(..., ge=-180, le=360),
    date: _Date = Query(..., description="Date to reconstruct, YYYY-MM-DD."),
    depth: Optional[float] = Query(None, ge=0, description="Single depth in metres; omit for all depths."),
) -> GridQueryParams:
    return GridQueryParams(
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, date=date, depth=depth
    )


class EmbeddingQueryParams(BaseModel):
    date: _Date = Field(..., description="Date to compute the embedding for, YYYY-MM-DD.")


def embedding_query_params(
    date: _Date = Query(..., description="Date to compute the embedding for, YYYY-MM-DD."),
) -> EmbeddingQueryParams:
    return EmbeddingQueryParams(date=date)


# --------------------------------------------------------------------------
# Response schemas
# --------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Stable, machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation.")


class ComponentStatus(BaseModel):
    status: str
    detail: Optional[str] = None

    model_config = {"extra": "allow"}


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if every component is available, else 'degraded'.")
    components: Dict[str, ComponentStatus]


class ModelArchitectureInfo(BaseModel):
    in_channels: int
    embed_dim: int
    num_depths: int
    depths: List[float]
    use_gnn: bool
    uncertainty_enabled: bool


class ModelInfoResponse(BaseModel):
    architecture: ModelArchitectureInfo
    runtime: Dict[str, Any]
    checkpoint: Dict[str, Any]
    environment: Optional[str] = None


class DatesResponse(BaseModel):
    dates: List[str]
    count: int
    min_date: Optional[str]
    max_date: Optional[str]
    data_mode: str
    is_synthetic: bool


class PointReconstructionResponse(BaseModel):
    mode: str
    lat: float
    lon: float
    date: str
    depth: float
    temperature: float
    anomaly: float
    climatology: Optional[float]
    embedding_dim: int
    data_mode: str
    latency_ms: float
    cache_hit: bool
    notes: List[str]


class ProfileReconstructionResponse(BaseModel):
    mode: str
    lat: float
    lon: float
    date: str
    depths: List[float]
    temperature: List[float]
    anomaly: List[float]
    climatology: Optional[List[float]]
    embedding_dim: int
    data_mode: str
    latency_ms: float
    cache_hit: bool
    notes: List[str]


class GridReconstructionResponse(BaseModel):
    mode: str
    date: str
    depth: Optional[float] = Field(None, description="Requested depth, or null if every depth was returned.")
    depths: Optional[List[float]] = Field(None, description="Present only when depth was omitted.")
    lat: List[float]
    lon: List[float]
    temperature: Union[List[List[float]], List[List[List[float]]]]
    climatology: Optional[Union[List[List[float]], List[List[List[float]]]]] = None
    data_mode: str
    latency_ms: float
    cache_hit: bool
    notes: List[str]


class EmbeddingResponse(BaseModel):
    date: str
    embedding: List[float]
    dim: int
    data_mode: str
    cache_hit: bool
    latency_ms: float