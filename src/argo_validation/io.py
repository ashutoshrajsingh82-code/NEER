"""
Phase 26 — reading ARGO profiles.

Two source formats, one `ArgoProfileSet` out:

* **Long-format CSV** — one row per (profile, level). Columns are matched
  case-insensitively against a list of aliases (see `_ALIASES`), so an
  export with ``PLATFORM_NUMBER, CYCLE_NUMBER, JULD, LATITUDE, LONGITUDE,
  PRES, TEMP, TEMP_QC`` and one with ``float_id, cycle, time, lat, lon,
  pres_dbar, temp_c, temp_qc`` both load. Leading ``#`` lines are comments.
* **ARGO profile NetCDF** (GDAC ``*_prof.nc`` layout: ``N_PROF x N_LEVELS``
  ``PRES``/``TEMP``/``TEMP_QC`` plus ``*_ADJUSTED`` twins). Read through
  `netCDF4` when installed, else `scipy.io.netcdf_file` (NetCDF-3, which is
  what most GDAC files are). Delayed/adjusted-mode profiles (``DATA_MODE``
  ``A``/``D``) use the ``*_ADJUSTED`` variables; real-time profiles use the
  raw ones. A directory of ``.nc`` files loads as one set.

Provenance
----------
A ``<file>.meta.json`` sidecar (written by the demo generator) may carry
``data_mode`` and ``disclaimer``. A ``provenance`` CSV column and a
``DEMO``-prefixed float id are also honoured. Any one of them marks the set
as demo — see `ArgoProfileSet.is_demo`. Absent all of that, a set is
``ARGO_USER_SUPPLIED``: this code never asserts that a file is authentic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.argo_validation.profiles import (
    DATA_MODE_DEMO,
    DATA_MODE_USER,
    ArgoProfile,
    ArgoProfileSet,
)
from src.data.loaders.errors import LoaderError, MissingDependencyError, SchemaError

PathLike = Union[str, Path]

_ALIASES: Dict[str, Sequence[str]] = {
    "float_id": ("float_id", "platform_number", "wmo", "wmo_id", "float", "platform"),
    "cycle": ("cycle", "cycle_number", "cyc"),
    "time": ("time", "juld", "date", "datetime", "date_time"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude", "long"),
    "pressure": ("pres", "pressure", "pres_dbar", "pressure_dbar", "pres_adjusted"),
    "depth": ("depth", "depth_m"),
    "temperature": ("temp", "temperature", "temp_c", "temperature_c", "temp_degc", "temp_adjusted"),
    "temperature_qc": ("temp_qc", "temperature_qc", "temp_c_qc", "temp_adjusted_qc"),
    "position_qc": ("position_qc", "pos_qc"),
    "time_qc": ("time_qc", "juld_qc"),
    "provenance": ("provenance",),
}


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _read_sidecar(path: Path) -> Dict[str, Any]:
    side = sidecar_path(path)
    if side.exists():
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parse_time(text: str) -> np.datetime64:
    text = (text or "").strip().replace("Z", "")
    if not text:
        return np.datetime64("NaT", "ns")
    try:
        return np.datetime64(text, "ns")
    except ValueError:
        return np.datetime64("NaT", "ns")


def _to_float(text: Optional[str]) -> float:
    try:
        return float(text) if text not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def load_argo_csv(path: PathLike) -> ArgoProfileSet:
    """Load a long-format ARGO CSV (one row per profile level)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise SchemaError(f"{path}: empty CSV")

    column: Dict[str, str] = {}
    lowered = {name.strip().lower(): name for name in reader.fieldnames}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                column[canonical] = lowered[alias]
                break

    missing = [k for k in ("float_id", "time", "lat", "lon", "temperature") if k not in column]
    if "pressure" not in column and "depth" not in column:
        missing.append("pressure|depth")
    if missing:
        raise SchemaError(
            f"{path}: missing required column(s) {missing}; found {list(reader.fieldnames)}"
        )

    groups: Dict[tuple, Dict[str, Any]] = {}
    provenance_marks: set = set()
    for row in reader:
        float_id = str(row[column["float_id"]]).strip()
        cycle_text = row.get(column["cycle"], "") if "cycle" in column else ""
        cycle = int(float(cycle_text)) if cycle_text not in ("", None) else 0
        key = (float_id, cycle, row[column["time"]])
        g = groups.get(key)
        if g is None:
            g = groups[key] = {
                "lat": _to_float(row[column["lat"]]),
                "lon": _to_float(row[column["lon"]]),
                "time": _parse_time(row[column["time"]]),
                "pos_qc": (row.get(column["position_qc"]) or None) if "position_qc" in column else None,
                "time_qc": (row.get(column["time_qc"]) or None) if "time_qc" in column else None,
                "pres": [], "depth": [], "temp": [], "qc": [],
            }
        if "pressure" in column:
            g["pres"].append(_to_float(row[column["pressure"]]))
        if "depth" in column:
            g["depth"].append(_to_float(row[column["depth"]]))
        g["temp"].append(_to_float(row[column["temperature"]]))
        if "temperature_qc" in column:
            g["qc"].append((row.get(column["temperature_qc"]) or "").strip())
        if "provenance" in column and row.get(column["provenance"]):
            provenance_marks.add(row[column["provenance"]].strip().upper())

    profiles: List[ArgoProfile] = []
    for (float_id, cycle, _), g in groups.items():
        profiles.append(
            ArgoProfile(
                float_id=float_id,
                cycle=cycle,
                time=g["time"],
                lat=g["lat"],
                lon=g["lon"],
                temperature_c=np.asarray(g["temp"], dtype=float),
                depth_m=np.asarray(g["depth"], dtype=float) if g["depth"] else None,
                pressure_dbar=np.asarray(g["pres"], dtype=float) if g["pres"] and not g["depth"] else None,
                temperature_qc=np.asarray(g["qc"]) if g["qc"] else None,
                position_qc=g["pos_qc"],
                time_qc=g["time_qc"],
            )
        )

    side = _read_sidecar(path)
    data_mode = str(side.get("data_mode", DATA_MODE_USER))
    disclaimer = side.get("disclaimer")
    if any("DEMO" in m or "SYNTHETIC" in m for m in provenance_marks):
        data_mode = DATA_MODE_DEMO
    return ArgoProfileSet(profiles=profiles, data_mode=data_mode, source=str(path), disclaimer=disclaimer)


# --------------------------------------------------------------------------
# NetCDF (GDAC profile files)
# --------------------------------------------------------------------------

_FILL_GUESS = 99999.0


def _read_netcdf_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Read every variable as a plain ndarray; numeric fill values -> NaN."""
    try:
        import netCDF4  # type: ignore

        ds = netCDF4.Dataset(str(path))
        try:
            out: Dict[str, np.ndarray] = {}
            for name, var in ds.variables.items():
                data = var[:]
                if np.ma.isMaskedArray(data):
                    data = data.astype(float).filled(np.nan) if data.dtype.kind in "fiu" else data.filled(b" ")
                data = np.asarray(data)
                # netCDF4 only masks fill values it recognises via an explicit
                # `_FillValue` attribute. ARGO files (and fixtures) that rely
                # on the implicit 99999.0 sentinel without declaring it need
                # the same guess-based fallback the scipy.io branch below uses.
                if data.dtype.kind == "f":
                    data = np.where(np.isclose(data, _FILL_GUESS), np.nan, data)
                out[name] = data
            return out
        finally:
            ds.close()
    except ImportError:
        pass

    try:
        from scipy.io import netcdf_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            package="netCDF4",
            purpose="reading ARGO NetCDF profile files",
            install_hint="pip install netCDF4  # or: pip install scipy (NetCDF-3 only)",
        ) from exc

    out = {}
    with netcdf_file(str(path), mmap=False) as ds:
        for name, var in ds.variables.items():
            data = np.array(var.data)
            if data.dtype.kind == "f":
                fill = getattr(var, "_FillValue", None)
                data = data.astype(float)
                if fill is not None:
                    data[data == float(np.asarray(fill).ravel()[0])] = np.nan
                data[np.isclose(data, _FILL_GUESS)] = np.nan
            out[name] = data
    return out


def _char_rows(arr: np.ndarray) -> List[str]:
    """`(N, k)` char array (or `(N,)` of single chars) -> list of N strings."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return [x.decode("ascii", "ignore").strip() if isinstance(x, bytes) else str(x).strip() for x in arr]
    return [b"".join(row).decode("ascii", "ignore").strip() if row.dtype.kind == "S" else "".join(row).strip()
            for row in arr]


def _char_grid(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        a = np.char.decode(a, "ascii", "ignore")
    return np.char.strip(a.astype(str))


def load_argo_netcdf(path: PathLike) -> ArgoProfileSet:
    """Load one GDAC-style ARGO profile NetCDF file."""
    path = Path(path)
    v = _read_netcdf_arrays(path)
    required = ["JULD", "LATITUDE", "LONGITUDE", "PRES", "TEMP"]
    absent = [r for r in required if r not in v]
    if absent:
        raise SchemaError(f"{path}: not an ARGO profile file — missing variables {absent}")

    n_prof = int(np.asarray(v["JULD"]).shape[0])
    platforms = _char_rows(v["PLATFORM_NUMBER"]) if "PLATFORM_NUMBER" in v else [path.stem] * n_prof
    cycles = np.asarray(v["CYCLE_NUMBER"]).astype(float) if "CYCLE_NUMBER" in v else np.zeros(n_prof)
    modes = _char_rows(v["DATA_MODE"]) if "DATA_MODE" in v else [""] * n_prof
    pos_qc = _char_rows(v["POSITION_QC"]) if "POSITION_QC" in v else [None] * n_prof
    juld_qc = _char_rows(v["JULD_QC"]) if "JULD_QC" in v else [None] * n_prof

    epoch = np.datetime64("1950-01-01T00:00:00", "ns")
    profiles: List[ArgoProfile] = []
    for i in range(n_prof):
        juld = float(np.asarray(v["JULD"])[i])
        time = (
            np.datetime64("NaT", "ns")
            if not np.isfinite(juld) or juld >= 999998
            else epoch + np.timedelta64(int(round(juld * 86400.0)), "s").astype("timedelta64[ns]")
        )
        use_adjusted = (
            modes[i] in ("A", "D")
            and "PRES_ADJUSTED" in v
            and "TEMP_ADJUSTED" in v
            and np.isfinite(np.asarray(v["TEMP_ADJUSTED"])[i]).any()
        )
        pres = np.asarray(v["PRES_ADJUSTED" if use_adjusted else "PRES"], dtype=float)[i]
        temp = np.asarray(v["TEMP_ADJUSTED" if use_adjusted else "TEMP"], dtype=float)[i]
        qc_name = "TEMP_ADJUSTED_QC" if use_adjusted else "TEMP_QC"
        qc = _char_grid(v[qc_name])[i] if qc_name in v else None
        lat = float(np.asarray(v["LATITUDE"])[i])
        lon = float(np.asarray(v["LONGITUDE"])[i])
        profiles.append(
            ArgoProfile(
                float_id=platforms[i],
                cycle=int(cycles[i]) if np.isfinite(cycles[i]) else 0,
                time=time,
                lat=lat,
                lon=lon,
                temperature_c=temp,
                pressure_dbar=pres,
                temperature_qc=qc,
                position_qc=pos_qc[i] or None,
                time_qc=juld_qc[i] or None,
                adjusted=bool(use_adjusted),
            )
        )
    return ArgoProfileSet(profiles=profiles, data_mode=DATA_MODE_USER, source=str(path))


def load_argo(path: PathLike) -> ArgoProfileSet:
    """Load ARGO profiles from a `.csv`, a `.nc` file, or a directory of `.nc` files."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.nc"))
        if not files:
            raise LoaderError(f"{path}: directory contains no .nc files")
        sets = [load_argo_netcdf(f) for f in files]
        return ArgoProfileSet(
            profiles=[p for s in sets for p in s.profiles],
            data_mode=DATA_MODE_USER,
            source=str(path),
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_argo_csv(path)
    if suffix in (".nc", ".nc4", ".netcdf"):
        return load_argo_netcdf(path)
    raise LoaderError(f"unsupported ARGO source '{path}' (expected .csv, .nc, or a directory of .nc)")