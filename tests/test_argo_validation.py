"""Tests for Phase 26 — the independent ARGO validation pipeline
(`src/argo_validation`, `scripts/run_argo_validation.py`).

Phase 26 makes these claims; each group of tests is what turns one into a
check rather than a comment:

1. **The stages do what they say.** Pressure->depth matches the published
   reference value; QC drops exactly the flagged/implausible levels and
   names why a profile was rejected; vertical interpolation is exact on a
   linear profile, refuses to bridge large gaps and to extrapolate; time
   matching is same-calendar-month, spatial matching is same-cell and
   ocean-only (with -180..180 vs 0..360 longitudes handled).
2. **The numbers are right.** A hand-built case with known offsets has known
   RMSE/MAE/bias/correlation, float/profile/matched counts and depth coverage.
3. **Demo data is never dressed up as validation.** A demo set stays demo
   even when its sidecar/header are stripped; its report has
   `observational_validation == False`, no `metrics` key, DEMO-prefixed files.
4. **ARGO validation stays separate from reanalysis/test-target evaluation.**
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.argo_validation import (  # noqa: E402
    ArgoProfile,
    ArgoProfileSet,
    NeerGrid,
    NeerPredictions,
    TimeMatchConfig,
    VerticalConfig,
    clean_profile,
    demo_stand_in_predictions,
    generate_demo_argo,
    interpolate_to_depths,
    load_argo,
    load_argo_csv,
    load_prediction_table,
    match_space,
    match_time,
    pressure_to_depth,
    run_argo_validation,
    write_outputs,
)
from src.argo_validation.pipeline import VALIDATION_TYPE_DEMO, VALIDATION_TYPE_REAL  # noqa: E402
from src.argo_validation.predictions import normalizer_stats_from_metadata  # noqa: E402
from src.argo_validation.profiles import (  # noqa: E402
    REJECT_BAD_POSITION_QC,
    REJECT_NO_TIME,
    REJECT_TOO_FEW_LEVELS,
)

MONTHS = np.array(["2021-01-01", "2021-02-01", "2021-03-01"], dtype="datetime64[ns]")
DEPTHS = np.array([0.0, 10.0, 20.0, 50.0])


def make_grid(*, lon=None, ocean=True) -> NeerGrid:
    lat = np.array([10.0, 10.25, 10.5])
    lon = np.array([80.0, 80.25, 80.5, 80.75]) if lon is None else np.asarray(lon, float)
    mask = np.ones((3, len(lon)), dtype=bool)
    if ocean:
        mask[2, -1] = False  # one land cell: lat 10.5, last lon
    return NeerGrid(
        time=MONTHS, lat=lat, lon=lon, depth=DEPTHS, ocean_mask=mask,
        split_of_time=np.array(["train", "val", "test"], dtype=object),
        attrs={"data_mode": "TEST_FIXTURE"},
    )


def make_predictions(grid: NeerGrid, offsets=(0.0, 0.0, 0.0)) -> NeerPredictions:
    """Prediction for month m = the fixture's true profile + offsets[m]."""
    base = 28.0 - 0.1 * grid.depth
    temp = np.vstack([base + o for o in offsets])
    return NeerPredictions(time=grid.time, depth_m=grid.depth, temperature_c=temp, origin="user_prediction_table")


def make_profile(float_id="6900001", cycle=1, when="2021-01-15T06:00:00", lat=10.1, lon=80.1,
                 levels=(0, 5, 10, 15, 20, 30, 40, 50, 60), qc=None, **kw) -> ArgoProfile:
    z = np.asarray(levels, dtype=float)
    return ArgoProfile(
        float_id=float_id, cycle=cycle, time=np.datetime64(when, "ns"), lat=lat, lon=lon,
        depth_m=z, temperature_c=28.0 - 0.1 * z,
        temperature_qc=None if qc is None else np.asarray(qc), **kw,
    )


# ---------------------------------------------------------------------------
# 1. stages
# ---------------------------------------------------------------------------


def test_pressure_to_depth_matches_unesco_reference_value():
    # SEAWATER/UNESCO check value: 10000 dbar at 30 deg latitude -> 9712.653 m
    assert pressure_to_depth(np.array([10000.0]), 30.0)[0] == pytest.approx(9712.653, abs=0.01)
    assert pressure_to_depth(np.array([0.0]), 10.0)[0] == 0.0
    d = pressure_to_depth(np.array([10.0, 100.0, 1000.0]), 15.0)
    assert np.all(np.diff(d) > 0) and d[2] == pytest.approx(992.0, abs=2.0)


def test_clean_profile_drops_flagged_levels_and_sorts_and_dedups():
    p = ArgoProfile(
        float_id="1", cycle=1, time=np.datetime64("2021-01-05", "ns"), lat=10.0, lon=80.0,
        depth_m=np.array([50, 0, 10, 10, 20, 30, 40], float),
        temperature_c=np.array([20, 28, 26, 27, 99.0, 23, 22], float),  # 99 -> range-dropped
        temperature_qc=np.array(["1", "1", "1", "1", "1", "4", "1"]),
    )
    c, reason = clean_profile(p, min_levels=3)
    assert reason is None
    assert np.all(np.diff(c.depth_m) > 0)
    # depth 30 dropped by QC flag 4; depth 20 (99 degC) dropped by range; duplicate 10 m averaged
    assert list(c.depth_m) == [0.0, 10.0, 40.0, 50.0]
    assert c.temperature_c[1] == pytest.approx(26.5)
    assert c.n_levels_dropped_qc == 1 and c.n_levels_dropped_range == 1


def test_clean_profile_names_the_rejection_reason():
    assert clean_profile(make_profile(when="NaT"))[1] == REJECT_NO_TIME
    assert clean_profile(make_profile(position_qc="4"))[1] == REJECT_BAD_POSITION_QC
    assert clean_profile(make_profile(levels=(0, 5, 10)), min_levels=5)[1] == REJECT_TOO_FEW_LEVELS
    all_bad = make_profile(qc=["4"] * 9)
    assert clean_profile(all_bad)[1] == REJECT_TOO_FEW_LEVELS
    assert clean_profile(make_profile(qc=["2"] * 9))[1] == REJECT_TOO_FEW_LEVELS  # only flag 1 accepted by default
    assert clean_profile(make_profile(qc=["2"] * 9), accept_qc=("1", "2"))[0] is not None


def test_interpolation_exact_on_linear_profile_and_never_extrapolates():
    d = np.array([2.0, 12.0, 30.0, 60.0])
    t = 28.0 - 0.1 * d
    out = interpolate_to_depths(d, t, np.array([0.0, 10.0, 20.0, 50.0, 100.0]),
                                VerticalConfig(gap_floor_m=40.0, gap_rel=0.0, edge_tolerance_m=10.0))
    assert out[1] == pytest.approx(28.0 - 1.0)      # linear => exact
    assert out[2] == pytest.approx(28.0 - 2.0)
    assert out[3] == pytest.approx(28.0 - 5.0)
    assert out[0] == pytest.approx(t[0])            # 0 m filled from 2 m within edge tolerance
    assert np.isnan(out[4])                         # 100 m: 40 m below deepest level -> no extrapolation


def test_interpolation_refuses_large_gaps_but_accepts_exact_hits():
    d = np.array([0.0, 100.0, 500.0])
    t = np.array([28.0, 20.0, 8.0])
    cfg = VerticalConfig(gap_floor_m=25.0, gap_rel=0.25, edge_tolerance_m=10.0)
    out = interpolate_to_depths(d, t, np.array([0.0, 50.0, 100.0, 300.0]), cfg)
    assert np.isnan(out[1])                         # 100 m bracket > max(25, 12.5)
    assert out[0] == 28.0 and out[2] == 20.0        # exact hits always accepted
    assert np.isnan(out[3])                         # 400 m bracket > max(25, 75)


def test_time_matching_is_same_calendar_month():
    i, off, why = match_time(np.datetime64("2021-02-28T23:00:00", "ns"), MONTHS)
    assert (i, why) == (1, None) and off == pytest.approx(27.96, abs=0.05)
    assert match_time(np.datetime64("2021-04-01T00:00:00", "ns"), MONTHS)[2] == "no_time_match"
    assert match_time(np.datetime64("2020-02-10", "ns"), MONTHS)[2] == "no_time_match"  # same month, wrong year


def test_time_matching_nearest_mode_respects_tolerance():
    days = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[ns]")
    cfg = TimeMatchConfig(mode="nearest", tolerance_days=0.6)
    assert match_time(np.datetime64("2021-01-02T10:00", "ns"), days, cfg)[0] == 1
    assert match_time(np.datetime64("2021-01-05", "ns"), days, cfg)[2] == "no_time_match"


def test_spatial_matching_cell_domain_and_land():
    g = make_grid()
    (i, j), km, why = match_space(10.24, 80.26, g)
    assert (i, j, why) == (1, 1, None) and km < 5.0
    assert match_space(10.0, 80.13, g)[0] == (0, 1)   # 80.13 is nearer 80.25 than 80.0, and within half a cell
    assert match_space(12.0, 80.1, g)[2] == "outside_domain"
    assert match_space(10.1, 90.0, g)[2] == "outside_domain"
    assert match_space(10.5, 80.75, g)[2] == "on_land_cell"
    # a profile between cells but beyond half a cell from any centre is rejected only past the tolerance
    assert match_space(10.0, 79.7, g)[2] == "outside_domain"


def test_spatial_matching_handles_longitude_conventions():
    g = make_grid(lon=[170.0, 175.0, 180.0, 185.0], ocean=False)   # grid in 0-360
    (i, j), _, why = match_space(10.0, -175.0, g)      # ARGO -175 == 185 E
    assert why is None and g.lon[j] == 185.0
    g2 = make_grid(lon=[-10.0, -5.0, 0.0, 5.0], ocean=False)       # grid in -180..180
    assert match_space(10.0, 355.0, g2)[2] is None       # 355 E == -5


# ---------------------------------------------------------------------------
# 2. numbers
# ---------------------------------------------------------------------------


def test_pipeline_known_answer_metrics_counts_and_coverage():
    grid = make_grid()
    # Jan prediction is +1 degC warm, Feb is -1 degC cold, Mar exact.
    preds = make_predictions(grid, offsets=(1.0, -1.0, 0.0))
    profiles = ArgoProfileSet(profiles=[
        make_profile("A", 1, "2021-01-10T00:00:00"),                     # Jan  -> error +1
        make_profile("A", 2, "2021-01-20T00:00:00", lat=10.25, lon=80.25),  # Jan -> +1 (same float)
        make_profile("B", 1, "2021-02-10T00:00:00", lat=10.5, lon=80.5),    # Feb -> -1
        make_profile("C", 1, "2021-03-05T00:00:00", levels=(0, 5, 10, 15, 20)),  # Mar, ends at 20 m, exact
        make_profile("D", 1, "2021-01-10T00:00:00", lat=15.0),           # outside domain
        make_profile("D", 2, "2021-01-10T00:00:00", lat=10.5, lon=80.75),  # land cell
        make_profile("E", 1, "2021-07-10T00:00:00"),                     # no time match
        make_profile("F", 1, "2021-01-10T00:00:00", qc=["4"] * 9),        # all levels flagged bad
    ], data_mode="ARGO_USER_SUPPLIED")

    res = run_argo_validation(profiles, grid, preds)
    c = res.report["counts"]
    assert c["float_count"] == 6 and c["profile_count"] == 8
    assert c["profiles_after_qc"] == 7
    assert c["profiles_rejected_qc"] == {"too_few_valid_levels_after_qc": 1}
    assert c["profiles_rejected_matching"] == {"outside_domain": 1, "on_land_cell": 1, "no_time_match": 1}
    assert c["matched_profiles"] == 4 and c["matched_float_count"] == 3
    assert c["matched_profiles_by_split"] == {"train": 2, "val": 1, "test": 1}

    m = res.report["metrics"]
    # A,A: +1 at 4 depths each (8 pairs); B: -1 at 4 depths; C: 0 at depths 0,10,20 (50 m not covered)
    assert m["overall"]["n_pairs"] == 8 + 4 + 3
    assert m["overall"]["bias"] == pytest.approx((8 * 1 - 4 * 1) / 15)
    assert m["overall"]["rmse"] == pytest.approx(np.sqrt(12 / 15))
    assert m["overall"]["mae"] == pytest.approx(12 / 15)
    obs = np.concatenate([28 - 0.1 * DEPTHS] * 3 + [28 - 0.1 * DEPTHS[:3]])
    err = np.concatenate([np.ones(8), -np.ones(4), np.zeros(3)])
    assert m["overall"]["correlation"] == pytest.approx(np.corrcoef(obs, obs + err)[0, 1])
    assert m["per_depth"]["50m"]["n_pairs"] == 3           # C lacks 50 m
    assert m["per_depth"]["0m"]["rmse"] == pytest.approx(np.sqrt(3 / 4))
    assert m["by_split"]["train"]["bias"] == pytest.approx(1.0)
    assert m["by_split"]["val"]["bias"] == pytest.approx(-1.0)
    assert m["by_split"]["test"]["rmse"] == pytest.approx(0.0)

    cov = res.report["depth_coverage"]
    assert cov["n_profiles"] == 4 and cov["profiles_covering_all_depths"] == 3
    assert cov["per_depth"]["50m"]["n_profiles"] == 3 and cov["per_depth"]["0m"]["fraction"] == 1.0
    assert cov["deepest_valid_depth_m"] == {"min": 20.0, "median": 50.0, "max": 50.0}
    assert res.report["validation_type"] == VALIDATION_TYPE_REAL
    assert res.report["observational_validation"] is True
    assert any("training split" in w for w in res.report["warnings"])


def test_pipeline_with_no_matches_reports_none_not_zero():
    grid = make_grid()
    res = run_argo_validation(ArgoProfileSet([make_profile(when="2030-01-01")]), grid, make_predictions(grid))
    assert res.matched == [] and res.report["metrics"] is None
    assert res.report["counts"]["matched_profiles"] == 0
    assert any("No profiles matched" in w for w in res.report["warnings"])


def test_pipeline_rejects_misaligned_predictions():
    grid = make_grid()
    bad = NeerPredictions(time=grid.time, depth_m=grid.depth + 1, temperature_c=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="depths"):
        run_argo_validation(ArgoProfileSet([make_profile()]), grid, bad)
    with pytest.raises(ValueError):
        NeerPredictions(time=grid.time, depth_m=grid.depth, temperature_c=np.zeros((2, 4)))


def test_gridded_predictions_are_used_per_cell():
    grid = make_grid(ocean=False)
    field = np.zeros((3, 4, 3, 4))
    field[:, :, 1, 1] = 5.0   # only cell (lat idx 1, lon idx 1) differs
    preds = NeerPredictions(time=grid.time, depth_m=grid.depth, temperature_c=field.mean(axis=(2, 3)),
                            gridded_temperature_c=field)
    assert preds.resolution == "gridded"
    res = run_argo_validation(
        ArgoProfileSet([make_profile(lat=10.25, lon=80.25), make_profile("Z", lat=10.0, lon=80.0)]), grid, preds)
    assert sorted(m.pred[0] for m in res.matched) == [0.0, 5.0]


# ---------------------------------------------------------------------------
# 3. demo labelling
# ---------------------------------------------------------------------------


def test_demo_pipeline_is_labelled_everywhere_and_never_validation(tmp_path):
    grid = make_grid()
    csv_path = generate_demo_argo(tmp_path / "demo_argo_profiles_SYNTHETIC.csv", grid, n_floats=6, seed=1)
    text = csv_path.read_text().splitlines()
    assert text[0].startswith("# DEMO_SYNTHETIC_ARGO") and "NOT REAL" in text[0]
    assert all(line.endswith("DEMO_SYNTHETIC_ARGO") for line in text[4:])
    assert csv_path.with_suffix(".meta.json").exists()

    profiles = load_argo(csv_path)
    assert profiles.is_demo and profiles.data_mode == "DEMO_SYNTHETIC_ARGO"
    assert all(p.float_id.startswith("DEMO_") for p in profiles.profiles)

    res = run_argo_validation(profiles, grid, demo_stand_in_predictions(grid))
    r = res.report
    assert r["validation_type"] == VALIDATION_TYPE_DEMO
    assert r["observational_validation"] is False
    assert "metrics" not in r and r["pipeline_check_metrics"] is not None
    assert r["banner"] and "NOT" in r["banner"] and r["warnings"][0] == r["banner"]
    assert any("NOT produced" in w or "not produced" in w for w in r["warnings"])
    assert r["neer_predictions"]["is_neer_output"] is False

    written = write_outputs(res, tmp_path / "out", plots=True)
    assert all(p.name.startswith("DEMO_SYNTHETIC_") for p in written.values())
    assert (tmp_path / "out" / "DEMO_SYNTHETIC_argo_matched_pairs.csv").read_text().startswith("# DEMO / SYNTHETIC")
    saved = json.loads((tmp_path / "out" / "DEMO_SYNTHETIC_argo_validation_report.json").read_text())
    assert saved["observational_validation"] is False and "metrics" not in saved


def test_demo_stays_demo_when_sidecar_and_header_are_stripped(tmp_path):
    grid = make_grid()
    p = generate_demo_argo(tmp_path / "demo_argo_profiles_SYNTHETIC.csv", grid, n_floats=3, seed=2)
    p.with_suffix(".meta.json").unlink()
    stripped = tmp_path / "renamed.csv"
    stripped.write_text("\n".join(l for l in p.read_text().splitlines() if not l.startswith("#")))
    assert load_argo(stripped).is_demo                     # provenance column + DEMO_ float ids
    # even without the provenance column, DEMO_ float ids alone keep it demo
    lines = stripped.read_text().splitlines()
    header = lines[0].split(",")[:-1]
    stripped.write_text("\n".join([",".join(header)] + [",".join(l.split(",")[:-1]) for l in lines[1:]]))
    assert load_argo(stripped).is_demo


def test_demo_generator_refuses_unlabelled_filename(tmp_path):
    with pytest.raises(ValueError, match="SYNTHETIC"):
        generate_demo_argo(tmp_path / "argo_profiles.csv", make_grid())


def test_real_looking_set_gets_unprefixed_outputs_and_metrics_key(tmp_path):
    grid = make_grid()
    res = run_argo_validation(ArgoProfileSet([make_profile("6901234", i, f"2021-01-{10 + i}T00:00:00")
                                              for i in range(1, 4)]), grid, make_predictions(grid))
    assert "metrics" in res.report and "pipeline_check_metrics" not in res.report
    written = write_outputs(res, tmp_path, plots=False)
    assert written["report"].name == "argo_validation_report.json"
    assert res.report["banner"] is None


def test_stand_in_predictions_are_labelled_not_neer():
    p = demo_stand_in_predictions(make_grid())
    assert p.origin == "demo_stand_in_NOT_NEER" and not p.is_neer_output


# ---------------------------------------------------------------------------
# I/O and predictions
# ---------------------------------------------------------------------------


def test_csv_loader_accepts_argo_style_columns_and_comments(tmp_path):
    f = tmp_path / "argo.csv"
    f.write_text(
        "# exported from somewhere\n"
        "PLATFORM_NUMBER,CYCLE_NUMBER,JULD,LATITUDE,LONGITUDE,PRES,TEMP,TEMP_QC\n"
        + "".join(f"6901234,7,2021-01-15T12:00:00Z,10.1,80.1,{p},{28 - 0.01 * p},1\n" for p in (5, 10, 20, 50, 100, 200))
    )
    s = load_argo_csv(f)
    assert s.n_profiles == 1 and s.n_floats == 1 and s.data_mode == "ARGO_USER_SUPPLIED" and not s.is_demo
    pr = s.profiles[0]
    assert pr.float_id == "6901234" and pr.cycle == 7 and pr.pressure_dbar is not None and pr.depth_m is None
    assert pr.depths()[-1] == pytest.approx(pressure_to_depth(np.array([200.0]), 10.1)[0])


def test_csv_loader_reports_missing_columns(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("float_id,time,lat\n1,2021-01-01,10\n")
    with pytest.raises(Exception, match="missing required column"):
        load_argo_csv(f)


def test_netcdf_loader_reads_gdac_layout_and_prefers_adjusted(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    path = tmp_path / "R6901234_001.nc"
    n_prof, n_lev = 2, 4
    ds = scipy_io.netcdf_file(str(path), "w")
    ds.createDimension("N_PROF", n_prof); ds.createDimension("N_LEVELS", n_lev); ds.createDimension("STR", 8)
    ds.createDimension("ONE", 1)

    def add(name, dims, data, typ):
        v = ds.createVariable(name, typ, dims); v[:] = data
        return v

    chars = lambda s: np.array([c.encode() for c in s.ljust(8)], dtype="S1")
    add("PLATFORM_NUMBER", ("N_PROF", "STR"), np.vstack([chars("6901234"), chars("6901234")]), "c")
    add("CYCLE_NUMBER", ("N_PROF",), np.array([1, 2], dtype="i"), "i")
    add("DATA_MODE", ("N_PROF",), np.array([b"R", b"D"], dtype="S1"), "c")
    add("JULD", ("N_PROF",), np.array([25567.5, 25600.0]), "d")     # days since 1950-01-01
    add("LATITUDE", ("N_PROF",), np.array([10.1, 10.2]), "d")
    add("LONGITUDE", ("N_PROF",), np.array([80.1, 80.2]), "d")
    add("PRES", ("N_PROF", "N_LEVELS"), np.array([[5, 10, 20, 99999.0], [5, 10, 20, 50]]), "d")
    add("TEMP", ("N_PROF", "N_LEVELS"), np.array([[28, 27, 26, 99999.0], [28, 27, 26, 25.0]]), "d")
    add("TEMP_QC", ("N_PROF", "N_LEVELS"), np.array([[b"1", b"1", b"4", b" "], [b"1"] * 4], dtype="S1"), "c")
    add("PRES_ADJUSTED", ("N_PROF", "N_LEVELS"), np.array([[5, 10, 20, 99999.0], [5, 10, 20, 50]]), "d")
    add("TEMP_ADJUSTED", ("N_PROF", "N_LEVELS"), np.array([[99999.0] * 4, [28.5, 27.5, 26.5, 25.5]]), "d")
    add("TEMP_ADJUSTED_QC", ("N_PROF", "N_LEVELS"), np.array([[b" "] * 4, [b"1"] * 4], dtype="S1"), "c")
    ds.close()

    s = load_argo(path)
    assert s.n_profiles == 2 and s.n_floats == 1
    a, b = s.profiles
    assert a.float_id == "6901234" and (a.cycle, b.cycle) == (1, 2)
    assert str(a.time.astype("datetime64[s]")) == "2020-01-01T12:00:00"   # 25567.5 d after 1950-01-01
    assert a.adjusted is False and b.adjusted is True               # 'D' mode with values -> adjusted
    assert np.isnan(a.temperature_c[3]) and b.temperature_c[0] == pytest.approx(28.5)
    assert list(a.temperature_qc) == ["1", "1", "4", ""]


def test_prediction_table_roundtrip(tmp_path):
    grid = make_grid()
    preds = make_predictions(grid, (0.1, 0.2, 0.3))
    back = load_prediction_table(preds.save(tmp_path / "p.npz"))
    assert np.array_equal(back.time, preds.time) and np.allclose(back.temperature_c, preds.temperature_c)
    assert back.origin == "user_prediction_table" and back.resolution == "domain_pooled"


def test_normalizer_stats_read_from_preprocessing_metadata(tmp_path):
    meta = {"steps": [{"step": "normalization", "state": {"method": "zscore", "statistics": {
        "subsurface_temp": {"center": [1.0, 2.0], "scale": [10.0, 20.0]}}}}]}
    f = tmp_path / "m.json"; f.write_text(json.dumps(meta))
    center, scale = normalizer_stats_from_metadata(f, "subsurface_temp", 2)
    assert list(center) == [1.0, 2.0] and list(scale) == [10.0, 20.0]
    with pytest.raises(ValueError):
        normalizer_stats_from_metadata(f, "subsurface_temp", 3)
    with pytest.raises(ValueError, match="no normalization statistics"):
        normalizer_stats_from_metadata(f, "sst", 2)


def test_checkpoint_predictions_need_torch_or_fail_clearly(tmp_path):
    try:
        import torch  # noqa: F401
    except ImportError:
        from src.argo_validation import predictions_from_checkpoint
        from src.data.loaders.errors import MissingDependencyError

        with pytest.raises(MissingDependencyError, match="torch"):
            predictions_from_checkpoint(tmp_path / "x.pt", object(), tmp_path / "m.json")
    else:  # pragma: no cover - exercised only where torch is installed
        pytest.skip("torch installed; checkpoint path is exercised by the integration run")


# ---------------------------------------------------------------------------
# 4. separation from the reanalysis/test-target evaluation
# ---------------------------------------------------------------------------


def test_argo_validation_shares_only_scalar_metric_functions_with_evaluation():
    root = Path(__file__).resolve().parents[1] / "src" / "argo_validation"
    imported = set()
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.evaluation"):
                imported.update((node.module, a.name) for a in node.names)
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("src.evaluation") for a in node.names)
    assert imported == {("src.evaluation.metrics", n) for n in ("bias", "mae", "pearson", "rmse")}


def test_outputs_go_to_a_separate_directory_from_the_evaluation_report():
    import scripts.run_argo_validation as cli

    assert cli.DEFAULT_REPORTS.name == "argo_validation" and cli.DEFAULT_REPORTS.parent.name == "reports"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_bundle(path: Path) -> None:
    from src.data.preprocessing.tensors import TensorBundle

    g = make_grid()
    n_t, n_la, n_lo = 3, len(g.lat), len(g.lon)
    TensorBundle(
        inputs=np.zeros((n_t, 1, n_la, n_lo), np.float32), input_mask=np.ones((n_t, 1, n_la, n_lo), bool),
        channel_names=["sst"], time=g.time, lat=g.lat, lon=g.lon,
        targets=np.zeros((n_t, 4, n_la, n_lo), np.float32), target_mask=np.ones((n_t, 4, n_la, n_lo), bool),
        target_names=["subsurface_temp"], depth=g.depth, ocean_mask=g.ocean_mask,
        split_masks={"train": np.array([1, 0, 0], bool), "val": np.array([0, 1, 0], bool),
                     "test": np.array([0, 0, 1], bool)},
        attrs={"data_mode": "DEMO_SYNTHETIC", "is_synthetic": True},
    ).save(path)


def test_cli_demo_mode_writes_only_demo_labelled_outputs(tmp_path):
    import scripts.run_argo_validation as cli

    _write_bundle(tmp_path / "t.npz")
    rc = cli.main(["--demo", "--tensors", str(tmp_path / "t.npz"), "--reports-dir", str(tmp_path / "rep"),
                   "--demo-output", str(tmp_path / "d" / "demo_argo_profiles_SYNTHETIC.csv"), "--n-floats", "5"])
    assert rc == 0
    names = {p.name for p in (tmp_path / "rep").iterdir()}
    assert names and all(n.startswith("DEMO_SYNTHETIC_") for n in names)


def test_cli_never_falls_back_to_demo_data_silently(tmp_path):
    import scripts.run_argo_validation as cli

    _write_bundle(tmp_path / "t.npz")
    assert cli.main(["--tensors", str(tmp_path / "t.npz")]) == 2                      # no --argo, no --demo
    assert cli.main(["--argo", str(tmp_path / "missing.csv"), "--tensors", str(tmp_path / "t.npz")]) == 2
    assert cli.main(["--argo", "a.csv", "--demo"]) == 2
    assert not (tmp_path / "rep").exists()