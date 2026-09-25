"""
Phase 26 — the independent ARGO validation pipeline.

    ARGO profiles
      -> QC / ingest            (profiles.clean_profile)
      -> time matching          (matching.match_time)
      -> spatial matching       (matching.match_space)
      -> vertical interpolation (interpolation.interpolate_to_depths)
      -> NEER prediction        (predictions.NeerPredictions.at)
      -> comparison             (obs vs prediction on NEER's depths)
      -> metrics                (metrics.compute_argo_metrics / compute_depth_coverage)

Independent of reanalysis / test-target evaluation
----------------------------------------------------
This package reads no `TensorBundle` targets, split logic or evaluation
report from `src/evaluation`, writes only under ``reports/argo_validation/``,
and its report says ``validation_type`` explicitly. Its only tie to the
other evaluation is that the four scalar metric functions are shared (see
`metrics.py`).

Demo data is never validation
------------------------------
If the profile set is demo/synthetic (`ArgoProfileSet.is_demo`), the
report has ``observational_validation: false``, ``validation_type:
DEMO_PIPELINE_CHECK_NOT_OBSERVATIONAL``, a leading warning, and its
metrics block is filed under ``pipeline_check_metrics`` rather than
``metrics`` — so nothing that reads "metrics" from a report can pick up
demo numbers by accident. Likewise a non-NEER prediction source is flagged.

Independence of the ARGO profiles themselves
----------------------------------------------
NEER's training targets may derive from ARGO (the INCOIS gridded product
is ARGO-based). Profiles in months inside NEER's *train* split may then
have shaped the model. Every matched profile carries its NEER split, the
metrics are broken down by split, and a warning fires when train-period
profiles are included; treat ``test`` (then ``val``) as the honest number.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.argo_validation.interpolation import VerticalConfig, interpolate_to_depths
from src.argo_validation.matching import (
    NeerGrid,
    SpaceMatchConfig,
    TimeMatchConfig,
    match_space,
    match_time,
)
from src.argo_validation.metrics import compute_argo_metrics, compute_depth_coverage
from src.argo_validation.predictions import RESOLUTION_POOLED, NeerPredictions
from src.argo_validation.profiles import (
    DEFAULT_ACCEPT_VALUE_QC,
    TEMP_RANGE_C,
    ArgoProfileSet,
    clean_profile,
)

PathLike = Union[str, Path]

PHASE = 26
VALIDATION_TYPE_REAL = "ARGO_PROFILE_VALIDATION"
VALIDATION_TYPE_DEMO = "DEMO_PIPELINE_CHECK_NOT_OBSERVATIONAL"

DEMO_BANNER = (
    "DEMO / SYNTHETIC ARGO DATA — NOT REAL OBSERVATIONS. This run only demonstrates that the "
    "pipeline executes end to end. It is NOT observational validation of NEER."
)

LOW_PROFILE_COUNT = 30


@dataclass(frozen=True)
class ArgoValidationConfig:
    accept_qc: Tuple[str, ...] = DEFAULT_ACCEPT_VALUE_QC
    min_levels: int = 5
    temp_range_c: Tuple[float, float] = TEMP_RANGE_C
    time: TimeMatchConfig = TimeMatchConfig()
    space: SpaceMatchConfig = SpaceMatchConfig()
    vertical: VerticalConfig = VerticalConfig()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["accept_qc"] = list(self.accept_qc)
        d["temp_range_c"] = list(self.temp_range_c)
        return d


@dataclass
class MatchedProfile:
    profile_id: str
    float_id: str
    cycle: int
    time: np.datetime64
    lat: float
    lon: float
    time_index: int
    lat_index: int
    lon_index: int
    grid_lat: float
    grid_lon: float
    time_offset_days: float
    distance_km: float
    split: str
    obs: np.ndarray    # (n_depth,) ARGO on NEER depths, NaN where not available
    pred: np.ndarray   # (n_depth,) NEER degC
    n_valid_depths: int


@dataclass
class ArgoValidationResult:
    report: Dict[str, Any]
    matched: List[MatchedProfile]
    depths_m: np.ndarray
    is_demo: bool
    obs: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pred: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    valid: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=bool))

    @property
    def file_prefix(self) -> str:
        return "DEMO_SYNTHETIC_" if self.is_demo else ""


def _iso(t: np.datetime64) -> str:
    return str(np.datetime64(t, "s"))


def run_argo_validation(
    profiles: ArgoProfileSet,
    grid: NeerGrid,
    predictions: NeerPredictions,
    config: ArgoValidationConfig = ArgoValidationConfig(),
) -> ArgoValidationResult:
    """Run the whole pipeline in memory. See the module docstring."""
    if not np.array_equal(np.asarray(predictions.depth_m, float), np.asarray(grid.depth, float)):
        raise ValueError(
            f"prediction depths {list(predictions.depth_m)} differ from grid depths {list(grid.depth)}"
        )
    if len(predictions.time) != len(grid.time) or not np.array_equal(predictions.time, grid.time):
        raise ValueError("prediction time axis differs from the NEER grid time axis")

    depths = np.asarray(grid.depth, dtype=float)
    is_demo = profiles.is_demo

    # 1. QC / ingest ------------------------------------------------------
    qc_rejections: Counter = Counter()
    cleaned = []
    n_without_qc = n_adjusted = 0
    levels_dropped_qc = levels_dropped_range = 0
    for p in profiles.profiles:
        c, reason = clean_profile(
            p, accept_qc=config.accept_qc, min_levels=config.min_levels, temp_range=config.temp_range_c
        )
        if c is None:
            qc_rejections[reason] += 1
            continue
        cleaned.append(c)
        n_without_qc += int(not c.has_qc_info)
        n_adjusted += int(bool(c.source.adjusted))
        levels_dropped_qc += c.n_levels_dropped_qc
        levels_dropped_range += c.n_levels_dropped_range

    # 2-3. time + spatial matching -------------------------------------------
    match_rejections: Counter = Counter()
    candidates = []
    for c in cleaned:
        p = c.source
        ti, t_off, reason = match_time(p.time, grid.time, config.time)
        if reason:
            match_rejections[reason] += 1
            continue
        cell, dist, reason = match_space(p.lat, p.lon, grid, config.space)
        if reason:
            match_rejections[reason] += 1
            continue
        candidates.append((c, ti, t_off, cell, dist))

    # 4-6. vertical interpolation, NEER prediction, comparison ---------------
    matched: List[MatchedProfile] = []
    n_no_overlap = 0
    for c, ti, t_off, (li, lj), dist in candidates:
        obs = interpolate_to_depths(c.depth_m, c.temperature_c, depths, config.vertical)
        pred = predictions.at(ti, li, lj)
        n_valid = int((np.isfinite(obs) & np.isfinite(pred)).sum())
        if n_valid == 0:
            n_no_overlap += 1
            continue
        p = c.source
        split = ""
        if grid.split_of_time is not None:
            split = str(grid.split_of_time[ti])
        matched.append(
            MatchedProfile(
                profile_id=p.profile_id, float_id=p.float_id, cycle=p.cycle, time=p.time,
                lat=p.lat, lon=p.lon, time_index=ti, lat_index=li, lon_index=lj,
                grid_lat=float(grid.lat[li]), grid_lon=float(grid.lon[lj]),
                time_offset_days=t_off, distance_km=dist, split=split,
                obs=obs, pred=np.asarray(pred, dtype=float), n_valid_depths=n_valid,
            )
        )

    n_d = len(depths)
    if matched:
        obs_arr = np.vstack([m.obs for m in matched])
        pred_arr = np.vstack([m.pred for m in matched])
    else:
        obs_arr = np.empty((0, n_d))
        pred_arr = np.empty((0, n_d))
    valid = np.isfinite(obs_arr) & np.isfinite(pred_arr)

    # 7. metrics ---------------------------------------------------------------
    have_splits = grid.split_of_time is not None
    metrics: Optional[Dict[str, Any]] = None
    if matched:
        metrics = compute_argo_metrics(
            obs_arr, pred_arr, valid, depths, [m.split for m in matched] if have_splits else None
        )
    coverage = compute_depth_coverage(valid, depths)

    # report ---------------------------------------------------------------------
    n_in = profiles.n_profiles
    counts = {
        "float_count": profiles.n_floats,
        "profile_count": n_in,
        "profiles_rejected_qc": dict(qc_rejections),
        "profiles_after_qc": len(cleaned),
        "profiles_rejected_matching": dict(match_rejections),
        "profiles_matched_time_and_space": len(candidates),
        "profiles_matched_but_no_depth_overlap": n_no_overlap,
        "matched_profiles": len(matched),
        "matched_float_count": len({m.float_id for m in matched}),
        "matched_fraction_of_input": (len(matched) / n_in) if n_in else None,
        "matched_profiles_by_split": dict(Counter(m.split or "unsplit" for m in matched)),
        "profiles_without_qc_flags": n_without_qc,
        "profiles_using_adjusted_values": n_adjusted,
        "levels_dropped_by_qc_flag": levels_dropped_qc,
        "levels_dropped_by_range_check": levels_dropped_range,
    }

    warnings: List[str] = []
    limitations: List[str] = [
        "ARGO profiles are point measurements; NEER currently predicts one domain-pooled "
        "profile per month, so per-profile errors include the spatial spread of the basin.",
        "Vertical interpolation is linear between bracketing ARGO levels, with no bridging of "
        "large gaps and no extrapolation beyond a small edge tolerance; uncovered depths are excluded.",
    ]
    if is_demo:
        warnings.append(DEMO_BANNER)
    if not predictions.is_neer_output:
        warnings.append(
            f"Predictions have origin '{predictions.origin}' — they were not produced from a NEER "
            "checkpoint by this pipeline, so these numbers must not be reported as NEER's skill."
            if predictions.origin != "user_prediction_table"
            else "Predictions come from a user-supplied table; this pipeline cannot verify they are NEER output."
        )
    if counts["profiles_without_qc_flags"]:
        warnings.append(
            f"{counts['profiles_without_qc_flags']} profile(s) carried no QC flags; all finite, "
            "in-range values were used unchecked."
        )
    if not matched:
        warnings.append("No profiles matched NEER's time axis and grid with valid depth overlap; no metrics computed.")
    elif len(matched) < LOW_PROFILE_COUNT:
        warnings.append(f"Only {len(matched)} matched profiles (< {LOW_PROFILE_COUNT}); metrics are statistically weak.")
    if have_splits and counts["matched_profiles_by_split"].get("train"):
        warnings.append(
            f"{counts['matched_profiles_by_split']['train']} matched profile(s) fall in months of NEER's "
            "training split. If NEER's training targets derive from ARGO, these are not independent "
            "of the model; prefer the 'test' (then 'val') numbers in by_split."
        )
    if predictions.resolution == RESOLUTION_POOLED:
        warnings.append(
            "NEER output is domain-pooled: every profile in a month is compared to the same predicted profile."
        )

    metrics_key = "pipeline_check_metrics" if is_demo else "metrics"
    report: Dict[str, Any] = {
        "phase": PHASE,
        "validation_type": VALIDATION_TYPE_DEMO if is_demo else VALIDATION_TYPE_REAL,
        "observational_validation": (not is_demo),
        "banner": DEMO_BANNER if is_demo else None,
        "argo": {
            "data_mode": profiles.data_mode,
            "is_demo": is_demo,
            "source": profiles.source,
            "disclaimer": profiles.disclaimer,
        },
        "neer_predictions": predictions.describe(),
        "neer_grid": {
            "n_time": int(len(grid.time)),
            "time_range": [_iso(grid.time.min()), _iso(grid.time.max())],
            "lat_range": [float(grid.lat.min()), float(grid.lat.max())],
            "lon_range": [float(grid.lon.min()), float(grid.lon.max())],
            "depth_m": [float(d) for d in depths],
            "tensor_data_mode": grid.attrs.get("data_mode"),
        },
        "separation_note": (
            "Independent of reanalysis/test-target evaluation (reports/evaluation_report.json): "
            "different data, matching logic and outputs."
        ),
        "config": config.to_dict(),
        "counts": counts,
        "depth_coverage": coverage,
        metrics_key: metrics,
        "warnings": warnings,
        "limitations": limitations,
    }
    return ArgoValidationResult(
        report=report, matched=matched, depths_m=depths, is_demo=is_demo,
        obs=obs_arr, pred=pred_arr, valid=valid,
    )


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def write_outputs(result: ArgoValidationResult, out_dir: PathLike, *, plots: bool = True) -> Dict[str, Path]:
    """Write the JSON report, matched-profile and pair CSVs (and plots) under `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = result.file_prefix
    written: Dict[str, Path] = {}

    report_path = out / f"{prefix}argo_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(result.report), f, indent=2, default=_json_default, allow_nan=False)
    written["report"] = report_path

    label = "DEMO_SYNTHETIC_ARGO_NOT_OBSERVATIONS" if result.is_demo else "argo"
    profiles_path = out / f"{prefix}argo_matched_profiles.csv"
    with open(profiles_path, "w", encoding="utf-8", newline="") as f:
        if result.is_demo:
            f.write(f"# {DEMO_BANNER}\n")
        w = csv.writer(f)
        w.writerow(["data_label", "profile_id", "float_id", "cycle", "time", "lat", "lon",
                    "neer_time_index", "grid_lat", "grid_lon", "distance_km", "time_offset_days",
                    "neer_split", "n_valid_depths"])
        for m in result.matched:
            w.writerow([label, m.profile_id, m.float_id, m.cycle, _iso(m.time), f"{m.lat:.4f}", f"{m.lon:.4f}",
                        m.time_index, f"{m.grid_lat:.4f}", f"{m.grid_lon:.4f}", f"{m.distance_km:.2f}",
                        f"{m.time_offset_days:.2f}", m.split, m.n_valid_depths])
    written["matched_profiles"] = profiles_path

    pairs_path = out / f"{prefix}argo_matched_pairs.csv"
    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        if result.is_demo:
            f.write(f"# {DEMO_BANNER}\n")
        w = csv.writer(f)
        w.writerow(["data_label", "profile_id", "neer_split", "depth_m", "argo_temp_c", "neer_temp_c", "error_c_neer_minus_argo"])
        for m in result.matched:
            for k, z in enumerate(result.depths_m):
                if np.isfinite(m.obs[k]) and np.isfinite(m.pred[k]):
                    w.writerow([label, m.profile_id, m.split, f"{z:g}", f"{m.obs[k]:.4f}", f"{m.pred[k]:.4f}",
                                f"{m.pred[k] - m.obs[k]:.4f}"])
    written["matched_pairs"] = pairs_path

    if plots and result.matched:
        from src.argo_validation.plots import plot_argo_validation

        written["plot"] = plot_argo_validation(result, out / f"{prefix}argo_validation_vs_depth.png")
    return written


def _sanitize(obj: Any) -> Any:
    """Recursively turn NaN/inf floats into None so the JSON is strictly valid."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def _json_default(o: Any):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")