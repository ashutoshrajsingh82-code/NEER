"""Tests for the NEER centralized configuration system (configs/*.yaml +
src/utils/config.py): loading, merging, and validation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import (
    ConfigError,
    ConfigValidationError,
    DemoConfig,
    DomainConfig,
    ModelConfig,
    ProjectMeta,
    deep_merge,
    load_config,
    validate_config,
    validate_demo,
    validate_depths,
    validate_domain,
    validate_model,
    validate_project,
)

EXPECTED_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


# --------------------------------------------------------------------------
# Loading — shared values resolved for every environment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ["demo", "development", "production", "base"])
def test_load_config_resolves_shared_project_metadata(environment):
    config = load_config(environment)
    assert config.environment == environment
    assert config.project == ProjectMeta(
        name="NEER",
        full_name="Neural Embedding based Estimation and Reconstruction",
        problem_id="SIH26066",
        organization="MoES / INCOIS",
    )


@pytest.mark.parametrize("environment", ["demo", "development", "production", "base"])
def test_load_config_resolves_shared_domain(environment):
    config = load_config(environment)
    assert config.domain == DomainConfig(
        lat_min=5, lat_max=30, lon_min=45, lon_max=105, resolution=0.25
    )


@pytest.mark.parametrize("environment", ["demo", "development", "production", "base"])
def test_load_config_resolves_shared_depths(environment):
    config = load_config(environment)
    assert config.depths == EXPECTED_DEPTHS


@pytest.mark.parametrize("environment", ["demo", "development", "production", "base"])
def test_load_config_resolves_shared_model_defaults(environment):
    config = load_config(environment)
    assert config.model == ModelConfig(embedding_dim=256, use_gnn=False)


def test_unknown_environment_raises():
    with pytest.raises(ConfigError):
        load_config("staging")


# --------------------------------------------------------------------------
# Loading — environment-specific overrides (the only place they differ)
# --------------------------------------------------------------------------


def test_demo_environment_demo_settings():
    config = load_config("demo")
    assert config.demo == DemoConfig(enabled=True, seed=42)


def test_development_environment_demo_settings():
    config = load_config("development")
    assert config.demo == DemoConfig(enabled=True, seed=7)


def test_production_environment_disables_demo():
    config = load_config("production")
    assert config.demo.enabled is False


def test_default_environment_is_development():
    config = load_config()
    assert config.environment == "development"


# --------------------------------------------------------------------------
# No duplication: overlay files only contain their deltas
# --------------------------------------------------------------------------


def test_overlay_files_do_not_redeclare_shared_sections():
    import yaml

    from src.utils.config import CONFIGS_DIR

    shared_sections = {"project", "domain", "depths"}
    for name in ("demo.yaml", "development.yaml", "production.yaml"):
        with open(CONFIGS_DIR / name, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        present = shared_sections & set(raw.keys())
        assert not present, f"{name} duplicates shared section(s): {present}"


# --------------------------------------------------------------------------
# deep_merge
# --------------------------------------------------------------------------


def test_deep_merge_overrides_nested_keys_only():
    base = {"a": 1, "demo": {"enabled": True, "seed": 42}}
    override = {"demo": {"seed": 7}}
    merged = deep_merge(base, override)
    assert merged == {"a": 1, "demo": {"enabled": True, "seed": 7}}
    # inputs must not be mutated
    assert base["demo"]["seed"] == 42


def test_deep_merge_list_values_are_replaced_not_merged():
    base = {"depths": [0, 5, 10]}
    override = {"depths": [1, 2]}
    merged = deep_merge(base, override)
    assert merged["depths"] == [1, 2]


# --------------------------------------------------------------------------
# Validation — valid inputs pass silently
# --------------------------------------------------------------------------


def test_validate_config_passes_for_each_environment():
    for environment in ("demo", "development", "production", "base"):
        config = load_config(environment, validate=False)
        validate_config(config.raw)  # should not raise


def test_validate_project_accepts_expected_metadata():
    validate_project(
        {
            "name": "NEER",
            "full_name": "Neural Embedding based Estimation and Reconstruction",
            "problem_id": "SIH26066",
            "organization": "MoES / INCOIS",
        }
    )


def test_validate_domain_accepts_expected_bounds():
    validate_domain({"lat_min": 5, "lat_max": 30, "lon_min": 45, "lon_max": 105, "resolution": 0.25})


def test_validate_depths_accepts_expected_levels():
    validate_depths(EXPECTED_DEPTHS)


def test_validate_model_accepts_expected_hyperparameters():
    validate_model({"embedding_dim": 256, "use_gnn": False})


def test_validate_demo_accepts_expected_settings():
    validate_demo({"enabled": True, "seed": 42})


# --------------------------------------------------------------------------
# Validation — invalid inputs are rejected
# --------------------------------------------------------------------------


def test_validate_domain_rejects_inverted_latitude_bounds():
    with pytest.raises(ConfigValidationError):
        validate_domain({"lat_min": 30, "lat_max": 5, "lon_min": 45, "lon_max": 105, "resolution": 0.25})


def test_validate_domain_rejects_inverted_longitude_bounds():
    with pytest.raises(ConfigValidationError):
        validate_domain({"lat_min": 5, "lat_max": 30, "lon_min": 105, "lon_max": 45, "resolution": 0.25})


def test_validate_domain_rejects_non_positive_resolution():
    with pytest.raises(ConfigValidationError):
        validate_domain({"lat_min": 5, "lat_max": 30, "lon_min": 45, "lon_max": 105, "resolution": 0})


def test_validate_domain_rejects_out_of_range_latitude():
    with pytest.raises(ConfigValidationError):
        validate_domain({"lat_min": -95, "lat_max": 30, "lon_min": 45, "lon_max": 105, "resolution": 0.25})


def test_validate_domain_rejects_missing_key():
    with pytest.raises(ConfigValidationError):
        validate_domain({"lat_min": 5, "lat_max": 30, "lon_min": 45, "resolution": 0.25})


def test_validate_depths_rejects_empty_list():
    with pytest.raises(ConfigValidationError):
        validate_depths([])


def test_validate_depths_rejects_unsorted_list():
    with pytest.raises(ConfigValidationError):
        validate_depths([0, 10, 5])


def test_validate_depths_rejects_negative_values():
    with pytest.raises(ConfigValidationError):
        validate_depths([-10, 0, 5])


def test_validate_depths_rejects_duplicates():
    with pytest.raises(ConfigValidationError):
        validate_depths([0, 5, 5, 10])


def test_validate_model_rejects_non_positive_embedding_dim():
    with pytest.raises(ConfigValidationError):
        validate_model({"embedding_dim": 0, "use_gnn": False})


def test_validate_model_rejects_non_boolean_use_gnn():
    with pytest.raises(ConfigValidationError):
        validate_model({"embedding_dim": 256, "use_gnn": "false"})


def test_validate_demo_rejects_non_boolean_enabled():
    with pytest.raises(ConfigValidationError):
        validate_demo({"enabled": "yes", "seed": 42})


def test_validate_demo_rejects_non_integer_seed():
    with pytest.raises(ConfigValidationError):
        validate_demo({"enabled": True, "seed": "42"})


def test_validate_project_rejects_empty_field():
    with pytest.raises(ConfigValidationError):
        validate_project({"name": "", "full_name": "x", "problem_id": "x", "organization": "x"})


def test_validate_config_rejects_missing_section():
    incomplete = {
        "environment": "demo",
        "project": {
            "name": "NEER",
            "full_name": "Neural Embedding based Estimation and Reconstruction",
            "problem_id": "SIH26066",
            "organization": "MoES / INCOIS",
        },
        "domain": {"lat_min": 5, "lat_max": 30, "lon_min": 45, "lon_max": 105, "resolution": 0.25},
        "depths": EXPECTED_DEPTHS,
        # 'model' and 'demo' sections missing on purpose
    }
    with pytest.raises(ConfigValidationError):
        validate_config(incomplete)


# --------------------------------------------------------------------------
# Environment variable overrides
# --------------------------------------------------------------------------


def test_env_var_overrides_demo_seed(monkeypatch):
    monkeypatch.setenv("NEER_DEMO_SEED", "999")
    config = load_config("demo")
    assert config.demo.seed == 999


def test_env_var_overrides_model_embedding_dim(monkeypatch):
    monkeypatch.setenv("NEER_MODEL_EMBEDDING_DIM", "128")
    config = load_config("production")
    assert config.model.embedding_dim == 128
