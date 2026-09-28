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
// Precision: 2 decimal places by default (~1.1 km at NEER latitudes) — finer
// than the 0.25° (~27 km) native grid spacing so a coordinate never appears
// to snap coarser than the cursor actually moved, while avoiding misleading
// over-precision. Missing/non-finite input renders a placeholder, never 0.
//
// Accessibility: deliberately NOT a live region. It updates on every
// pointer move; announcing that would flood a screen reader. It is a labelled
// group whose text is simply available when read.
// -----------------------------------------------------------------------------

import { memo } from "react";
import { cn } from "@/lib/cn";

const PLACEHOLDER = "--.--°";

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * Formats a coordinate with custom precision and cardinal suffix (N/S or E/W).
 * @param {number|null|undefined} val
 * @param {string} posSuffix - e.g. "N" or "E"
 * @param {string} negSuffix - e.g. "S" or "W"
 * @param {number} [precision=2]
 * @returns {string} placeholder for missing/non-finite input
 */
export function formatCoordWithPrecision(val, posSuffix, negSuffix, precision = 2) {
  if (!isFiniteNumber(val)) return PLACEHOLDER;
  const suffix = val >= 0 ? posSuffix : negSuffix;
  return `${Math.abs(val).toFixed(precision)}°${suffix}`;
}

function formatAxis(val, posSuffix, negSuffix, precision, format) {
  if (!isFiniteNumber(val)) return PLACEHOLDER;
  if (format === "decimal") return `${val >= 0 ? "+" : "−"}${Math.abs(val).toFixed(precision)}°`;
  return formatCoordWithPrecision(val, posSuffix, negSuffix, precision);
}

/**
 * @param {object} props
 * @param {number|null|undefined} [props.lat] - degrees latitude, or nullish for placeholder
 * @param {number|null|undefined} [props.lon] - degrees longitude, or nullish for placeholder
 * @param {{lat?: number|null, lon?: number|null}} [props.point] - coordinate object alternative
 * @param {{lat?: number|null, lon?: number|null}} [props.coordinates] - coordinate object alternative
 * @param {string} [props.source] - which interaction produced it, e.g. "Cursor" / "Selected cell"
 * @param {number} [props.precision=2] - decimal digits
 * @param {"cardinal"|"decimal"} [props.format="cardinal"] - "cardinal" (15.25°N) or "decimal" (+15.25°)
 * @param {"row"|"stacked"} [props.layout="row"]
 * @param {"sm"|"md"} [props.size="sm"]
 * @param {string} [props.className]
 */
function CoordinateDisplay({
  lat,
  lon,
  point,
  coordinates,
  source,
  precision = 2,
  format = "cardinal",
  layout = "row",
  size = "sm",
  className,
}) {
  const effectiveLat = lat ?? point?.lat ?? coordinates?.lat;
  const effectiveLon = lon ?? point?.lon ?? coordinates?.lon;

  const latText = formatAxis(effectiveLat, "N", "S", precision, format);
  const lonText = formatAxis(effectiveLon, "E", "W", precision, format);

  return (
    <span
      role="group"
      aria-label={`Map coordinates${source ? ` (${source})` : ""}`}
      className={cn(
        "inline-flex select-none font-mono",
        layout === "stacked" ? "flex-col gap-0.5" : "flex-wrap items-center gap-x-4 gap-y-1",
        size === "md" ? "text-small" : "text-caption",
        className
      )}
    >
      {source && <span className="text-text-disabled uppercase tracking-wider">{source}</span>}
      <span className="inline-flex items-center gap-1.5">
        <span className="tracking-wider text-text-disabled">LAT</span>
        <span className="tabular-nums text-text-secondary">{latText}</span>
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="tracking-wider text-text-disabled">LON</span>
        <span className="tabular-nums text-text-secondary">{lonText}</span>
      </span>
    </span>
  );
}

export default memo(CoordinateDisplay);