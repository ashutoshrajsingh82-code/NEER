"""Tests for the Phase 08 input-feature set
(src/data/preprocessing/channels.py): the authoritative channel-order
definition, and that `FeatureBuilder` + `TensorAssembler` build and
consume exactly it — the eleven channels the model relies on, in the
exact order it relies on them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (  # noqa: E402
    CHANNEL_DESCRIPTIONS,
    NEER_CHANNEL_ORDER,
    NEER_N_CHANNELS,
    FeatureBuilder,
    PreprocessingError,
    PreprocessingPipeline,
    StepConfigurationError,
    TensorAssembler,
    validate_channel_order,
)
from conftest import tiny_dataset  # noqa: E402

#: The eleven features named in the Phase 08 spec, written out explicitly
#: so this test fails if `channels.py` is ever edited to something else —
#: it is checked against the literal list, not re-derived from it.
EXPECTED_ORDER = (
    "sst",
    "sss",
    "sla",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
    "time_sin",
    "time_cos",
    "lat_norm",
    "lon_norm",
)


# --------------------------------------------------------------------------
# The definition itself
# --------------------------------------------------------------------------


def test_channel_order_matches_the_phase_08_spec_exactly():
    assert NEER_CHANNEL_ORDER == EXPECTED_ORDER


def test_channel_count_is_eleven():
    assert NEER_N_CHANNELS == 11
    assert len(NEER_CHANNEL_ORDER) == 11


def test_channel_names_are_unique():
    assert len(set(NEER_CHANNEL_ORDER)) == len(NEER_CHANNEL_ORDER)


def test_every_channel_is_documented():
    assert set(CHANNEL_DESCRIPTIONS) == set(NEER_CHANNEL_ORDER)
    assert all(isinstance(v, str) and v for v in CHANNEL_DESCRIPTIONS.values())


# --------------------------------------------------------------------------
# validate_channel_order
# --------------------------------------------------------------------------


def test_validate_channel_order_accepts_the_canonical_order():
    validate_channel_order(NEER_CHANNEL_ORDER)  # must not raise
    validate_channel_order(list(NEER_CHANNEL_ORDER))  # any sequence type


def test_validate_channel_order_rejects_a_reordering():
    """Same eleven channels, wrong order — this is the bug that matters most."""
    with pytest.raises(ValueError):
        validate_channel_order(tuple(reversed(NEER_CHANNEL_ORDER)))


def test_validate_channel_order_rejects_a_missing_channel():
    with pytest.raises(ValueError):
        validate_channel_order(NEER_CHANNEL_ORDER[:-1])


def test_validate_channel_order_rejects_an_extra_channel():
    with pytest.raises(ValueError):
        validate_channel_order(NEER_CHANNEL_ORDER + ("coriolis",))


# --------------------------------------------------------------------------
# FeatureBuilder builds exactly the derived channels the set needs
# --------------------------------------------------------------------------


def test_default_feature_builder_produces_only_the_needed_derived_channels():
    """Defaults build the four derived channels Phase 08 needs and nothing
    that isn't part of the fixed input set (no current_speed, wind_speed
    or coriolis) — those remain available on request, just not by default.
    """
    dataset = tiny_dataset()
    result = FeatureBuilder().transform(dataset)
    built = set(result.variables) - set(dataset.variables)
    assert built == {"time_sin", "time_cos", "lat_norm", "lon_norm"}


def test_non_default_derived_features_remain_available_on_request():
    dataset = FeatureBuilder(features=["coriolis"]).transform(tiny_dataset())
    assert "coriolis" in dataset


# --------------------------------------------------------------------------
# TensorAssembler consumes exactly this ordering by default
# --------------------------------------------------------------------------


def test_tensor_assembler_default_input_variables_is_the_canonical_order():
    assert TensorAssembler().input_variables == NEER_CHANNEL_ORDER


def test_pipeline_run_on_demo_data_yields_exactly_the_canonical_channels(
    preprocessed_demo,
):
    tensors = preprocessed_demo.tensors
    assert tensors.n_channels == NEER_N_CHANNELS
    assert tuple(tensors.channel_names) == NEER_CHANNEL_ORDER


def test_pipeline_input_tensor_has_the_expected_shape(preprocessed_demo):
    inputs = preprocessed_demo.tensors.inputs
    assert inputs.ndim == 4
    assert inputs.shape[1] == NEER_N_CHANNELS


def test_each_channel_index_holds_the_named_physical_field(preprocessed_demo):
    """Position, not just presence: channel i really is NEER_CHANNEL_ORDER[i]."""
    tensors = preprocessed_demo.tensors
    for index, name in enumerate(NEER_CHANNEL_ORDER):
        np.testing.assert_array_equal(tensors.inputs[:, index], tensors.channel(name))


def test_assembler_raises_when_the_dataset_is_missing_canonical_channels():
    """A dataset that has not been through FeatureBuilder (or is missing raw
    fields) cannot silently produce a shorter or reordered tensor — the
    default assembler demands the full eleven or fails loudly.
    """
    with pytest.raises((StepConfigurationError, PreprocessingError)):
        TensorAssembler().assemble(tiny_dataset())


def test_an_explicit_input_variables_override_still_works():
    """The fixed default doesn't remove the escape hatch for ad hoc tensors."""
    dataset = FeatureBuilder(features=["cyclic_time"]).transform(tiny_dataset())
    assembler = TensorAssembler(input_variables=["sst", "time_sin", "time_cos"])
    bundle = assembler.assemble(dataset)
    assert bundle.channel_names == ["sst", "time_sin", "time_cos"]


def test_from_config_pipeline_also_uses_the_canonical_order():
    pipeline = PreprocessingPipeline.from_config()
    assembler = pipeline.assembler
    assert tuple(assembler.input_variables) == NEER_CHANNEL_ORDER
