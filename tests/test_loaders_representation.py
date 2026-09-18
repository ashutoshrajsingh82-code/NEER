"""
Tests for the standardized representation and its supporting vocabulary:
`representation.py`, `schema.py`, `units.py`, and `missing.py`.

These are the invariants the loaders rely on, so they are tested directly
rather than only through a file round-trip: canonical names, canonical
axis order, NaN-as-missing, and unit conversion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import (
    CANONICAL_DIM_ORDER,
    OceanDataset,
    Variable,
    build_variable,
    canonical_coord_name,
    canonical_variable_name,
    convert,
    convert_to_canonical,
    lookup_spec,
    normalize_unit,
)
from src.data.loaders.missing import (
    finite_range,
    missing_fraction,
    missing_mask,
    replace_fill_values,
)
from src.data.loaders.schema import infer_dims_from_shape, transpose_to_canonical

LAT = np.array([10.0, 10.25, 10.5])
LON = np.array([70.0, 70.25])
TIME = np.array(["2020-01-15", "2020-02-15"], dtype="datetime64[ns]")


def make_sst(values=None) -> Variable:
    if values is None:
        values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    return build_variable("sst", values, ("time", "lat", "lon"), units="degC")


def make_dataset(**kwargs) -> OceanDataset:
    variable = kwargs.pop("variable", None) or make_sst()
    return OceanDataset(
        variables={variable.name: variable},
        coords={"time": TIME, "lat": LAT, "lon": LON},
        **kwargs,
    )


# --------------------------------------------------------------------------
# Schema: names and axis order
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("latitude", "lat"),
        ("LATITUDE", "lat"),
        ("nav_lat", "lat"),
        ("y", "lat"),
        ("longitude", "lon"),
        ("x", "lon"),
        ("lev", "depth"),
        ("pressure", "depth"),
        ("date", "time"),
        ("valid_time", "time"),
    ],
)
def test_coordinate_aliases_map_to_canonical_names(source, expected):
    assert canonical_coord_name(source) == expected


def test_unknown_coordinate_names_pass_through():
    assert canonical_coord_name("ensemble_member") == "ensemble_member"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("analysed_sst", "sst"),
        ("sea_surface_temperature", "sst"),
        ("ugos", "u_current"),
        ("u10", "u_wind"),
        ("ocean_mask", "land_mask"),
    ],
)
def test_variable_aliases_map_to_neer_names(source, expected):
    assert canonical_variable_name(source) == expected


def test_lookup_spec_resolves_through_aliases():
    spec = lookup_spec("analysed_sst")
    assert spec is not None
    assert spec.name == "sst" and spec.units == "degC"


def test_lookup_spec_returns_none_for_unknown_variables():
    assert lookup_spec("mystery_field") is None


def test_canonical_dim_order_is_time_depth_lat_lon():
    assert CANONICAL_DIM_ORDER == ("time", "depth", "lat", "lon")


def test_transpose_reorders_axes_into_canonical_order():
    values = np.arange(2 * 3 * 4).reshape(4, 3, 2)  # (lon, lat, time)
    transposed, dims = transpose_to_canonical(values, ("lon", "lat", "time"))
    assert dims == ("time", "lat", "lon")
    assert transposed.shape == (2, 3, 4)
    assert transposed[1, 2, 3] == values[3, 2, 1]


def test_transpose_leaves_canonical_input_untouched():
    values = np.zeros((2, 3, 4))
    transposed, dims = transpose_to_canonical(values, ("time", "lat", "lon"))
    assert transposed is values and dims == ("time", "lat", "lon")


def test_unknown_dims_sort_after_canonical_ones():
    values = np.zeros((5, 2, 3))
    _, dims = transpose_to_canonical(values, ("ensemble", "time", "lat"))
    assert dims == ("time", "lat", "ensemble")


def test_transpose_rejects_a_dim_count_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        transpose_to_canonical(np.zeros((2, 3)), ("time", "lat", "lon"))


def test_dims_can_be_inferred_from_unambiguous_shapes():
    sizes = {"time": 2, "lat": 3, "lon": 4}
    assert infer_dims_from_shape((2, 3, 4), sizes) == ("time", "lat", "lon")


def test_ambiguous_shapes_are_rejected_rather_than_guessed():
    sizes = {"lat": 3, "lon": 3}
    with pytest.raises(ValueError, match="cannot infer"):
        infer_dims_from_shape((3, 3), sizes, variable="sst")


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("degC", "degC"),
        ("deg_C", "degC"),
        ("degrees_Celsius", "degC"),
        ("Celsius", "degC"),
        ("K", "K"),
        ("kelvin", "K"),
        ("PSU", "psu"),
        ("m s-1", "m/s"),
        ("m/s", "m/s"),
        ("meters", "m"),
    ],
)
def test_unit_spellings_normalize(source, expected):
    assert normalize_unit(source) == expected


def test_unknown_unit_normalizes_to_none():
    assert normalize_unit("furlongs per fortnight") is None


def test_absent_unit_normalizes_to_none():
    assert normalize_unit(None) is None
    assert normalize_unit("") is None


@pytest.mark.parametrize(
    "value,source,target,expected",
    [
        (301.15, "K", "degC", 28.0),
        (28.0, "degC", "K", 301.15),
        (150.0, "cm", "m", 1.5),
        (100.0, "cm/s", "m/s", 1.0),
        (1.0, "knots", "m/s", 0.514444),
    ],
)
def test_unit_conversions(value, source, target, expected):
    result = convert(np.array([value]), source, target)
    assert result[0] == pytest.approx(expected, rel=1e-5)


def test_conversion_without_a_rule_raises():
    with pytest.raises(ValueError, match="no known conversion"):
        convert(np.array([1.0]), "psu", "m")


def test_conversion_preserves_float32_storage():
    values = np.array([301.15], dtype=np.float32)
    assert convert(values, "K", "degC").dtype == np.float32


def test_convert_to_canonical_reports_whether_it_converted():
    values = np.array([301.15])
    converted, unit, did_convert = convert_to_canonical(values, "K", "degC")
    assert did_convert and unit == "degC" and converted[0] == pytest.approx(28.0)

    same, unit, did_convert = convert_to_canonical(values, "degC", "degC")
    assert not did_convert and unit == "degC" and same is values


def test_convert_to_canonical_leaves_unknown_units_alone():
    values = np.array([1.0])
    result, unit, did_convert = convert_to_canonical(values, "furlongs", "m")
    assert not did_convert and result is values and unit == "furlongs"


def test_convert_to_canonical_does_not_force_incompatible_units():
    values = np.array([35.0])
    result, unit, did_convert = convert_to_canonical(values, "psu", "m")
    assert not did_convert and unit == "psu" and result is values


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------


def test_sentinel_values_are_replaced_with_nan():
    values = np.array([28.0, -999.0, 29.0])
    result, n_replaced = replace_fill_values(values)
    assert n_replaced == 1 and np.isnan(result[1])
    assert result[0] == 28.0


def test_fill_value_attribute_is_honoured():
    values = np.array([28.0, -1.5, 29.0])
    result, n_replaced = replace_fill_values(values, attrs={"_FillValue": -1.5})
    assert n_replaced == 1 and np.isnan(result[1])


def test_integer_arrays_are_widened_so_gaps_can_be_stored():
    values = np.array([28, -9999, 29], dtype=np.int32)
    result, n_replaced = replace_fill_values(values)
    assert n_replaced == 1
    assert np.issubdtype(result.dtype, np.floating) and np.isnan(result[1])


def test_boolean_arrays_are_left_alone():
    values = np.array([True, False])
    result, n_replaced = replace_fill_values(values)
    assert n_replaced == 0 and result.dtype == bool


def test_physical_values_survive_sentinel_replacement():
    # Nothing in the sentinel list may collide with plausible ocean values.
    values = np.array([-2.0, 0.0, 35.5, 1000.0])
    result, n_replaced = replace_fill_values(values)
    assert n_replaced == 0 and np.array_equal(result, values)


def test_missing_mask_and_fraction():
    values = np.array([1.0, np.nan, 3.0, np.nan])
    assert list(missing_mask(values)) == [False, True, False, True]
    assert missing_fraction(values) == pytest.approx(0.5)


def test_missing_fraction_of_an_empty_array_is_zero():
    assert missing_fraction(np.array([])) == 0.0


def test_finite_range_ignores_missing_values():
    assert finite_range(np.array([np.nan, 2.0, 5.0, np.nan])) == (2.0, 5.0)


def test_finite_range_of_all_missing_is_none():
    assert finite_range(np.array([np.nan, np.nan])) is None


# --------------------------------------------------------------------------
# Variable
# --------------------------------------------------------------------------


def test_variable_rejects_a_dim_count_mismatch():
    with pytest.raises(ValueError, match="dims"):
        Variable(name="sst", values=np.zeros((2, 3)), dims=("time", "lat", "lon"))


def test_build_variable_canonicalizes_names_and_axis_order():
    values = np.zeros((2, 3, 4))  # (time, latitude, longitude)
    variable = build_variable("analysed_sst", values, ("time", "latitude", "longitude"))
    assert variable.dims == ("time", "lat", "lon")


def test_variable_reports_its_sizes():
    assert make_sst().sizes == {"time": 2, "lat": 3, "lon": 2}


def test_variable_exposes_its_spec():
    assert make_sst().spec.units == "degC"


def test_variable_missing_statistics():
    values = np.full((TIME.size, LAT.size, LON.size), 28.0)
    values[0, 0, 0] = np.nan
    variable = make_sst(values)
    assert variable.n_missing == 1
    assert variable.missing_fraction == pytest.approx(1 / values.size)
    assert variable.value_range == (28.0, 28.0)


def test_variable_summary_is_serializable():
    summary = make_sst().summary()
    assert summary["name"] == "sst" and summary["dims"] == ["time", "lat", "lon"]
    assert summary["units"] == "degC"


# --------------------------------------------------------------------------
# OceanDataset
# --------------------------------------------------------------------------


def test_dataset_rejects_a_key_that_disagrees_with_the_variable_name():
    with pytest.raises(ValueError, match="does not match"):
        OceanDataset(variables={"temperature": make_sst()}, coords={"lat": LAT, "lon": LON})


def test_dataset_rejects_multidimensional_coordinates():
    with pytest.raises(ValueError, match="1-dimensional"):
        OceanDataset(variables={}, coords={"lat": np.zeros((3, 2))})


def test_dataset_mapping_interface():
    dataset = make_dataset()
    assert "sst" in dataset
    assert len(dataset) == 1
    assert list(dataset) == ["sst"]
    assert dataset["sst"].name == "sst"
    assert dataset.get("absent") is None
    assert np.array_equal(dataset.values_of("sst"), dataset["sst"].values)


def test_unknown_variable_lookup_lists_what_is_available():
    with pytest.raises(KeyError, match="available"):
        make_dataset()["absent"]


def test_dataset_reports_its_coordinates_in_canonical_order():
    dataset = make_dataset()
    assert dataset.coord_names == ["time", "lat", "lon"]
    assert dataset.sizes == {"time": 2, "lat": 3, "lon": 2}
    assert dataset.grid_shape == (3, 2)
    assert dataset.n_time == 2


def test_grid_shape_is_none_without_a_lat_lon_grid():
    variable = build_variable("sla", np.zeros((2,)), ("time",), units="m")
    dataset = OceanDataset(variables={"sla": variable}, coords={"time": TIME})
    assert dataset.grid_shape is None


def test_subset_keeps_only_the_requested_variables():
    sla = build_variable("sla", np.zeros((TIME.size, LAT.size, LON.size)), ("time", "lat", "lon"), units="m")
    dataset = OceanDataset(
        variables={"sst": make_sst(), "sla": sla},
        coords={"time": TIME, "lat": LAT, "lon": LON},
    )
    assert dataset.subset(["sla"]).variable_names == ["sla"]


def test_subset_drops_coordinates_nothing_uses():
    depth = np.array([0.0, 50.0])
    subsurface = build_variable(
        "subsurface_temp",
        np.zeros((TIME.size, 2, LAT.size, LON.size)),
        ("time", "depth", "lat", "lon"),
        units="degC",
    )
    dataset = OceanDataset(
        variables={"sst": make_sst(), "subsurface_temp": subsurface},
        coords={"time": TIME, "depth": depth, "lat": LAT, "lon": LON},
    )
    assert "depth" not in dataset.subset(["sst"]).coords
    assert "depth" in dataset.subset(["subsurface_temp"]).coords


def test_subset_rejects_unknown_variables():
    with pytest.raises(KeyError, match="not in dataset"):
        make_dataset().subset(["absent"])


def test_provenance_flags_synthetic_data():
    assert make_dataset(attrs={"data_mode": "DEMO_SYNTHETIC"}).is_synthetic is True
    assert make_dataset(attrs={"data_mode": "OBSERVED"}).is_synthetic is False
    assert make_dataset().data_mode is None


def test_to_dict_includes_variables_and_coordinates():
    arrays = make_dataset().to_dict()
    assert set(arrays) == {"sst", "time", "lat", "lon"}


def test_summary_reports_the_time_range():
    summary = make_dataset().summary()
    assert summary["time_range"][0].startswith("2020-01-15")
    assert summary["sizes"] == {"time": 2, "lat": 3, "lon": 2}


def test_describe_is_human_readable():
    text = make_dataset().describe()
    assert "sst" in text and "time=2" in text
