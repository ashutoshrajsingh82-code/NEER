"""
Centralized configuration system for NEER.

Design
------
`configs/base.yaml` is the single source of truth for values shared across
every environment: project metadata, the spatial domain, depth levels,
model hyperparameters, and demo defaults, plus shared paths/service
settings.

`configs/demo.yaml`, `configs/development.yaml`, and
`configs/production.yaml` are thin *overlay* files that only declare what
differs for that environment (e.g. whether the demo dataset is enabled).
`load_config()` deep-merges the requested environment file on top of
`base.yaml`, so loading any environment always resolves to the complete
set of sections — nothing is duplicated between files, and nothing is
hardcoded again in this module.

The resulting configuration is validated (domain bounds, depth ordering,
model hyperparameters, demo settings) before being handed back to the
caller as a typed `NeerConfig`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

BASE_CONFIG_FILE = "base.yaml"
VALID_ENVIRONMENTS = {"base", "demo", "development", "production"}
DEFAULT_ENVIRONMENT = "development"


class ConfigError(Exception):
    """Raised when a config file cannot be found or parsed."""


class ConfigValidationError(Exception):
    """Raised when a loaded configuration fails validation."""


# --------------------------------------------------------------------------
# Typed sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectMeta:
    name: str
    full_name: str
    problem_id: str
    organization: str


@dataclass(frozen=True)
class DomainConfig:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int
    use_gnn: bool


@dataclass(frozen=True)
class DemoConfig:
    enabled: bool
    seed: int


@dataclass(frozen=True)
class UncertaintyConfig:
    """Phase 23 — optional heteroscedastic-regression toggle.

    `enabled=True` is the only thing this flag asserts: that
    `NEERModel`/`DepthDecoder` build a second output head predicting a
    per-depth log-variance alongside the mean anomaly. It is not a claim
    that the predicted variance is calibrated (see
    `src/models/depth_decoder.py`).
    """

    enabled: bool


@dataclass(frozen=True)
class NeerConfig:
    """Fully resolved, validated NEER configuration for one environment."""

    environment: str
    project: ProjectMeta
    domain: DomainConfig
    depths: List[float]
    model: ModelConfig
    demo: DemoConfig
    uncertainty: UncertaintyConfig
    raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# --------------------------------------------------------------------------
# Loading & merging
# --------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping at the top level: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict.

    Nested dicts are merged key-by-key; any other value type (including
    lists) in `override` fully replaces the corresponding value in `base`.
    Neither input is mutated.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_var_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a small set of NEER_* environment variable overrides.

    Supported overrides (all optional):
      NEER_ENVIRONMENT           -> environment
      NEER_DEMO_ENABLED          -> demo.enabled   ("true"/"false")
      NEER_DEMO_SEED             -> demo.seed       (int)
      NEER_MODEL_EMBEDDING_DIM   -> model.embedding_dim (int)
      NEER_MODEL_USE_GNN         -> model.use_gnn   ("true"/"false")
      NEER_UNCERTAINTY_ENABLED   -> uncertainty.enabled ("true"/"false")
      NEER_BACKEND_PORT          -> backend.port    (int)
      NEER_FRONTEND_PORT         -> frontend.port   (int)
    """
    data = dict(data)

    def _bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    if "NEER_ENVIRONMENT" in os.environ:
        data["environment"] = os.environ["NEER_ENVIRONMENT"]

    if "NEER_DEMO_ENABLED" in os.environ or "NEER_DEMO_SEED" in os.environ:
        demo = dict(data.get("demo", {}))
        if "NEER_DEMO_ENABLED" in os.environ:
            demo["enabled"] = _bool(os.environ["NEER_DEMO_ENABLED"])
        if "NEER_DEMO_SEED" in os.environ:
            demo["seed"] = int(os.environ["NEER_DEMO_SEED"])
        data["demo"] = demo

    if "NEER_MODEL_EMBEDDING_DIM" in os.environ or "NEER_MODEL_USE_GNN" in os.environ:
        model = dict(data.get("model", {}))
        if "NEER_MODEL_EMBEDDING_DIM" in os.environ:
            model["embedding_dim"] = int(os.environ["NEER_MODEL_EMBEDDING_DIM"])
        if "NEER_MODEL_USE_GNN" in os.environ:
            model["use_gnn"] = _bool(os.environ["NEER_MODEL_USE_GNN"])
        data["model"] = model

    if "NEER_UNCERTAINTY_ENABLED" in os.environ:
        uncertainty = dict(data.get("uncertainty", {}))
        uncertainty["enabled"] = _bool(os.environ["NEER_UNCERTAINTY_ENABLED"])
        data["uncertainty"] = uncertainty

    if "NEER_BACKEND_PORT" in os.environ:
        backend = dict(data.get("backend", {}))
        backend["port"] = int(os.environ["NEER_BACKEND_PORT"])
        data["backend"] = backend

    if "NEER_FRONTEND_PORT" in os.environ:
        frontend = dict(data.get("frontend", {}))
        frontend["port"] = int(os.environ["NEER_FRONTEND_PORT"])
        data["frontend"] = frontend

    return data


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_project(project: Dict[str, Any]) -> None:
    required = ("name", "full_name", "problem_id", "organization")
    for key in required:
        value = project.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigValidationError(f"project.{key} must be a non-empty string")


def validate_domain(domain: Dict[str, Any]) -> None:
    required = ("lat_min", "lat_max", "lon_min", "lon_max", "resolution")
    for key in required:
        if key not in domain:
            raise ConfigValidationError(f"domain.{key} is required")
        if not isinstance(domain[key], (int, float)) or isinstance(domain[key], bool):
            raise ConfigValidationError(f"domain.{key} must be numeric")

    lat_min, lat_max = domain["lat_min"], domain["lat_max"]
    lon_min, lon_max = domain["lon_min"], domain["lon_max"]
    resolution = domain["resolution"]

    if lat_min >= lat_max:
        raise ConfigValidationError("domain.lat_min must be less than domain.lat_max")
    if lon_min >= lon_max:
        raise ConfigValidationError("domain.lon_min must be less than domain.lon_max")
    if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
        raise ConfigValidationError("domain latitudes must be within [-90, 90]")
    if not (-180 <= lon_min <= 180) or not (-180 <= lon_max <= 180):
        raise ConfigValidationError("domain longitudes must be within [-180, 180]")
    if resolution <= 0:
        raise ConfigValidationError("domain.resolution must be greater than 0")


def validate_depths(depths: Any) -> None:
    if not isinstance(depths, list) or len(depths) == 0:
        raise ConfigValidationError("depths must be a non-empty list")
    for d in depths:
        if not isinstance(d, (int, float)) or isinstance(d, bool):
            raise ConfigValidationError("every entry in depths must be numeric")
        if d < 0:
            raise ConfigValidationError("depths must be non-negative")
    if list(depths) != sorted(depths):
        raise ConfigValidationError("depths must be sorted in strictly ascending order")
    if len(set(depths)) != len(depths):
        raise ConfigValidationError("depths must not contain duplicate values")


def validate_model(model: Dict[str, Any]) -> None:
    if "embedding_dim" not in model:
        raise ConfigValidationError("model.embedding_dim is required")
    embedding_dim = model["embedding_dim"]
    if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim <= 0:
        raise ConfigValidationError("model.embedding_dim must be a positive integer")

    if "use_gnn" not in model:
        raise ConfigValidationError("model.use_gnn is required")
    if not isinstance(model["use_gnn"], bool):
        raise ConfigValidationError("model.use_gnn must be a boolean")


def validate_demo(demo: Dict[str, Any]) -> None:
    if "enabled" not in demo:
        raise ConfigValidationError("demo.enabled is required")
    if not isinstance(demo["enabled"], bool):
        raise ConfigValidationError("demo.enabled must be a boolean")

    if "seed" not in demo:
        raise ConfigValidationError("demo.seed is required")
    if not isinstance(demo["seed"], int) or isinstance(demo["seed"], bool):
        raise ConfigValidationError("demo.seed must be an integer")


def validate_uncertainty(uncertainty: Dict[str, Any]) -> None:
    if "enabled" not in uncertainty:
        raise ConfigValidationError("uncertainty.enabled is required")
    if not isinstance(uncertainty["enabled"], bool):
        raise ConfigValidationError("uncertainty.enabled must be a boolean")


def validate_config(data: Dict[str, Any]) -> None:
    """Validate a fully-merged configuration dict. Raises ConfigValidationError."""
    if "environment" not in data or not str(data["environment"]).strip():
        raise ConfigValidationError("environment is required")

    for section, validator in (
        ("project", validate_project),
        ("domain", validate_domain),
        ("model", validate_model),
        ("demo", validate_demo),
        ("uncertainty", validate_uncertainty),
    ):
        if section not in data or not isinstance(data[section], dict):
            raise ConfigValidationError(f"'{section}' section is required")
        validator(data[section])

    if "depths" not in data:
        raise ConfigValidationError("'depths' is required")
    validate_depths(data["depths"])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _build_neer_config(data: Dict[str, Any]) -> NeerConfig:
    return NeerConfig(
        environment=data["environment"],
        project=ProjectMeta(**data["project"]),
        domain=DomainConfig(**data["domain"]),
        depths=list(data["depths"]),
        model=ModelConfig(**data["model"]),
        demo=DemoConfig(**data["demo"]),
        uncertainty=UncertaintyConfig(**data["uncertainty"]),
        raw=data,
    )


def load_config(environment: str | None = None, *, validate: bool = True) -> NeerConfig:
    """Load, merge, and validate the NEER configuration for an environment.

    Parameters
    ----------
    environment:
        One of "demo", "development", "production", or "base". Defaults to
        the `NEER_ENVIRONMENT` env var, then "development".
    validate:
        If True (default), validates the merged config and raises
        `ConfigValidationError` on failure.
    """
    environment = environment or os.environ.get("NEER_ENVIRONMENT", DEFAULT_ENVIRONMENT)

    if environment not in VALID_ENVIRONMENTS:
        raise ConfigError(
            f"Unknown environment '{environment}'. Expected one of {sorted(VALID_ENVIRONMENTS)}"
        )

    base_data = _load_yaml(CONFIGS_DIR / BASE_CONFIG_FILE)

    if environment == "base":
        merged = base_data
    else:
        overlay_data = _load_yaml(CONFIGS_DIR / f"{environment}.yaml")
        merged = deep_merge(base_data, overlay_data)

    merged = _apply_env_var_overrides(merged)

    if validate:
        validate_config(merged)

    return _build_neer_config(merged)


@lru_cache(maxsize=None)
def get_config(environment: str | None = None) -> NeerConfig:
    """Cached variant of `load_config`, keyed by environment name."""
    return load_config(environment)


# Convenience singleton for simple use-cases (`from src.utils.config import settings`)
settings = load_config()