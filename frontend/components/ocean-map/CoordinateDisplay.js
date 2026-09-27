// -----------------------------------------------------------------------------
// NEER OceanMap — CoordinateDisplay  (Phase 35D1)
//
// Compact LAT/LON readout. Purely presentational and stateless: the caller
// always passes the coordinate currently produced by real map interaction
// (the hovered point, or the last selected/snapped grid cell) — this
// component never computes, infers, or remembers a coordinate itself, so it
// can't drift out of sync with what the map is actually doing.
//
// Precision: delegates to lib/oceanDomain.js's formatLat/formatLon, which
// round to 2 decimal degrees (~1.1 km at this domain's latitudes) — finer
// than the 0.25° (~27 km) native grid spacing so a coordinate never appears
// to snap coarser than the cursor actually moved, but not so many digits
// that it implies false precision the map's projection doesn't have.
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";
import { formatLat, formatLon } from "@/lib/oceanDomain";

const PLACEHOLDER = "--.--°";

/**
 * @param {object} props
 * @param {number|null|undefined} props.lat - degrees, or nullish for "no point" (renders a placeholder)
 * @param {number|null|undefined} props.lon - degrees, or nullish for "no point"
 * @param {"row"|"stacked"} [props.layout="row"]
 * @param {"sm"|"md"} [props.size="sm"]
 * @param {string} [props.className]
 */
export default function CoordinateDisplay({ lat, lon, layout = "row", size = "sm", className }) {
  const latText = typeof lat === "number" && Number.isFinite(lat) ? formatLat(lat) : PLACEHOLDER;
  const lonText = typeof lon === "number" && Number.isFinite(lon) ? formatLon(lon) : PLACEHOLDER;
  const textSizeClass = size === "md" ? "text-small" : "text-caption";

  return (
    <span
      className={cn(
        "inline-flex font-mono",
        layout === "stacked" ? "flex-col gap-0.5" : "flex-wrap items-center gap-x-4 gap-y-1",
        textSizeClass,
        className
      )}
    >
      <span>
        <span className="text-text-disabled">LAT </span>
        <span className="text-text-secondary">{latText}</span>
      </span>
      <span>
        <span className="text-text-disabled">LON </span>
        <span className="text-text-secondary">{lonText}</span>
      </span>
    </span>
  );
}