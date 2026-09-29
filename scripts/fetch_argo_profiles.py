"""Fetch real Argo temperature profiles for the NEER domain and time range.

The script queries Argovis in bounded monthly intervals and writes the
pressure, temperature, and temperature QC values in NEER's accepted long CSV
format. The API records carry source GDAC URLs; no values are synthesized.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT = RAW_DIR / "argo_profiles.csv"
SIDECAR = RAW_DIR / "argo_profiles.meta.json"
AUDIT = RAW_DIR / "ARGO_DATA_PROVENANCE.md"
API = "https://argovis-api.colorado.edu/argo"
START = date(2020, 1, 1)
END = date(2022, 1, 1)  # exclusive
POLYGON = [[45, 5], [105, 5], [105, 30], [45, 30], [45, 5]]
REQUESTED = ["pressure", "temperature", "temperature_argoqc"]


def add_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


def fetch(start: date, end: date) -> list[dict]:
    query = urlencode(
        {
            "startDate": f"{start.isoformat()}T00:00:00Z",
            "endDate": f"{end.isoformat()}T00:00:00Z",
            "polygon": json.dumps(POLYGON, separators=(",", ":")),
            "data": ",".join(REQUESTED),
        }
    )
    request = Request(
        f"{API}?{query}",
        headers={"Accept": "application/json", "User-Agent": "NEER-Argo-data-fetch/1.0"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except HTTPError as exc:
        # Argovis limits the number of returned profiles. If an interval is
        # too broad, split it in half without dropping or duplicating dates.
        if exc.code == 400 and (end - start).days > 1:
            midpoint = start + (end - start) // 2
            return fetch(start, midpoint) + fetch(midpoint, end)
        raise RuntimeError(f"Argovis returned HTTP {exc.code} for {start} to {end}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach Argovis for {start} to {end}: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Argovis response for {start} to {end}")
    if len(payload) >= 1000 and (end - start).days > 1:
        midpoint = start + (end - start) // 2
        return fetch(start, midpoint) + fetch(midpoint, end)
    return payload


def field_arrays(profile: dict) -> dict[str, list]:
    names = profile.get("data_info", [[]])[0]
    arrays = profile.get("data")
    if not isinstance(names, list) or not isinstance(arrays, list):
        return {}
    # Argovis returns one array per requested variable in the same order as
    # data_info[0]. Preserve the source arrays and flags without interpolation.
    return {str(name): values for name, values in zip(names, arrays) if isinstance(values, list)}


def rows_for(profile: dict):
    arrays = field_arrays(profile)
    pressure = arrays.get("pressure")
    temperature = arrays.get("temperature")
    quality = arrays.get("temperature_argoqc")
    geo = profile.get("geolocation", {}).get("coordinates", [])
    if not (pressure and temperature and quality and len(pressure) == len(temperature) == len(quality)):
        return
    if len(geo) < 2 or not profile.get("timestamp") or not profile.get("_id"):
        return
    profile_id = str(profile["_id"])
    float_id = profile_id.rsplit("_", 1)[0]
    cycle = profile.get("cycle_number", profile_id.rsplit("_", 1)[-1])
    source_url = ""
    for source in profile.get("source", []):
        if isinstance(source, dict) and source.get("url"):
            source_url = source["url"]
            break
    for pres, temp, qc in zip(pressure, temperature, quality):
        yield {
            "float_id": float_id,
            "cycle": cycle,
            "time": profile["timestamp"],
            "lat": geo[1],
            "lon": geo[0],
            "pres_dbar": pres,
            "temp_c": temp,
            "temp_qc": qc,
            "position_qc": profile.get("geolocation_argoqc", ""),
            "time_qc": profile.get("timestamp_argoqc", ""),
            "provenance": "ARGO_GDAC",
            "source_url": source_url,
        }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.part")
    fieldnames = [
        "float_id", "cycle", "time", "lat", "lon", "pres_dbar", "temp_c",
        "temp_qc", "position_qc", "time_qc", "provenance", "source_url",
    ]
    counts = []
    profile_ids: set[str] = set()
    row_count = 0
    cursor = START
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        while cursor < END:
            next_cursor = min(add_month(cursor), END)
            profiles = fetch(cursor, next_cursor)
            month_profiles = 0
            month_rows = 0
            for profile in profiles:
                first_row = True
                for row in rows_for(profile) or ():
                    writer.writerow(row)
                    month_rows += 1
                    row_count += 1
                    if first_row:
                        profile_ids.add(str(profile["_id"]))
                        month_profiles += 1
                        first_row = False
            counts.append(
                {"start": cursor.isoformat(), "end_exclusive": next_cursor.isoformat(),
                 "profiles_with_usable_temp_qc": month_profiles, "level_rows": month_rows}
            )
            print(f"{cursor:%Y-%m}: {month_profiles} profiles, {month_rows:,} levels", flush=True)
            cursor = next_cursor
            time.sleep(0.2)
    if row_count == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Argovis returned no usable pressure/temperature/QC rows; no data file written.")
    temporary.replace(OUTPUT)

    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "data_mode": "ARGO_PUBLIC_GDAC",
        "disclaimer": "Measured Argo float temperature profiles retrieved from the Argovis API; QC flags are preserved. This file contains profile observations and is distinct from generated demo profiles.",
        "source": API,
        "retrieved_utc": retrieved,
        "time_start_inclusive": START.isoformat(),
        "time_end_exclusive": END.isoformat(),
        "region_polygon_lon_lat": POLYGON,
        "requested_fields": REQUESTED,
        "profile_count": len(profile_ids),
        "level_row_count": row_count,
        "monthly_counts": counts,
        "argo_gdac_citation": "Argo (2000). Argo float data and metadata from Global Data Assembly Centre (Argo GDAC). SEANOE. https://doi.org/10.17882/42182",
        "quality_note": "Raw pressure (dbar), temperature (degree Celsius), temperature_argoqc, position QC, and timestamp QC are stored. NEER applies its configured QC and profile matching rules during evaluation.",
    }
    SIDECAR.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(
        "# Real Argo profile data\n\n"
        f"- Retrieved: {retrieved}\n"
        f"- Source: [Argovis Argo API]({API}) (each profile row also retains its source GDAC URL in the CSV).\n"
        f"- Region: {POLYGON} (longitude, latitude; NEER domain bounds).\n"
        f"- Time range: {START.isoformat()} through {END.isoformat()} (end exclusive).\n"
        f"- Profiles with pressure, temperature, and temperature QC: {len(profile_ids):,}.\n"
        f"- Profile levels written: {row_count:,}.\n"
        "- Source fields are preserved; no values are generated or interpolated during extraction.\n"
        "- QC filtering and time/space matching are applied by the NEER ARGO evaluation pipeline.\n\n"
        "Citation: Argo (2000). *Argo float data and metadata from Global Data Assembly Centre (Argo GDAC).* "
        "SEANOE. <https://doi.org/10.17882/42182>\n",
        encoding="utf-8",
    )
    print(f"Saved {len(profile_ids):,} profiles / {row_count:,} levels to {OUTPUT}")


if __name__ == "__main__":
    main()
