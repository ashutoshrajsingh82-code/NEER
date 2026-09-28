"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — CoordinateDisplay  (Phase 35D1)
//
// Compact LAT/LON readout. Purely presentational and stateless: the caller
// always passes the coordinate currently produced by real map interaction
// (the hovered point, or the last selected/snapped grid cell) — this
// component never computes, infers, or remembers a coordinate itself, so it
// can't drift out of sync with what the map is actually doing.
//
// Precision: delegates to formatLat/formatLon with configurable precision
// (default: 2 decimal places, ~1.1 km at NEER domain latitudes) — finer
// than the 0.25° (~27 km) native grid spacing so a coordinate never appears
// to snap coarser than the cursor actually moved, while avoiding misleading
// over-precision.
// -----------------------------------------------------------------------------

import { memo } from "react";
import { cn } from "@/lib/cn";
import { formatLat, formatLon } from "@/lib/oceanDomain";

const PLACEHOLDER = "--.--°";

/**
 * Formats a coordinate with custom precision and cardinal suffix (N/S or E/W).
 * @param {number|null|undefined} val
 * @param {string} posSuffix - e.g. "N" or "E"
 * @param {string} negSuffix - e.g. "S" or "W"
 * @param {number} [precision=2]
 * @returns {string}
 */
export function formatCoordWithPrecision(val, posSuffix, negSuffix, precision = 2) {
  if (typeof val !== "number" || !Number.isFinite(val)) return PLACEHOLDER;
  const suffix = val >= 0 ? posSuffix : negSuffix;
  return `${Math.abs(val).toFixed(precision)}°${suffix}`;
}

/**
 * @param {object} props
 * @param {number|null|undefined} [props.lat] - degrees latitude, or nullish for placeholder
 * @param {number|null|undefined} [props.lon] - degrees longitude, or nullish for placeholder
 * @param {{lat?: number|null, lon?: number|null}} [props.point] - coordinate object alternative
 * @param {{lat?: number|null, lon?: number|null}} [props.coordinates] - coordinate object alternative
 * @param {number} [props.precision=2] - decimal digits (default: 2)
 * @param {"cardinal"|"decimal"} [props.format="cardinal"] - "cardinal" (15.25°N) or "decimal" (+15.25°)
 * @param {"row"|"stacked"} [props.layout="row"]
 * @param {"sm"|"md"} [props.size="sm"]
 * @param {string} [props.className]
 */
export default function CoordinateDisplay({
  lat,
  lon,
  point,
  coordinates,
  precision = 2,
  format = "cardinal",
  layout = "row",
  size = "sm",
  className,
}) {
  const effectiveLat = lat ?? point?.lat ?? coordinates?.lat;
  const effectiveLon = lon ?? point?.lon ?? coordinates?.lon;

  const latText =
    typeof effectiveLat === "number" && Number.isFinite(effectiveLat)
      ? format === "decimal"
        ? `${effectiveLat >= 0 ? "+" : ""}${effectiveLat.toFixed(precision)}°`
        : precision === 2
          ? formatLat(effectiveLat)
          : formatCoordWithPrecision(effectiveLat, "N", "S", precision)
      : PLACEHOLDER;

  const lonText =
    typeof effectiveLon === "number" && Number.isFinite(effectiveLon)
      ? format === "decimal"
        ? `${effectiveLon >= 0 ? "+" : ""}${effectiveLon.toFixed(precision)}°`
        : precision === 2
          ? formatLon(effectiveLon)
          : formatCoordWithPrecision(effectiveLon, "E", "W", precision)
      : PLACEHOLDER;

  const textSizeClass = size === "md" ? "text-small" : "text-caption";

  return (
    <span
      role="status"
      aria-label={`Coordinates: Latitude ${latText}, Longitude ${lonText}`}
      className={cn(
        "inline-flex font-mono select-none",
        layout === "stacked" ? "flex-col gap-0.5" : "flex-wrap items-center gap-x-4 gap-y-1",
        textSizeClass,
        className
      )}
    >
      <span className="inline-flex items-center gap-1.5">
        <span className="text-text-disabled tracking-wider">LAT</span>
        <span className="text-text-secondary tabular-nums">{latText}</span>
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="text-text-disabled tracking-wider">LON</span>
        <span className="text-text-secondary tabular-nums">{lonText}</span>
      </span>
    </span>
  );
}

export default memo(CoordinateDisplay);