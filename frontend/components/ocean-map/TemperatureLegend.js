// -----------------------------------------------------------------------------
// NEER OceanMap — TemperatureLegend  (Phase 35B)
//
// A small continuous colorbar for whatever field TemperatureLayer is
// currently drawing. Purely presentational — the min/max it labels are
// always the real finite extent of the backend-provided grid
// (computeFiniteExtent in lib/colorScale.js), never a fixed/hard-coded
// physical range, so the bar always matches what's actually on screen.
// -----------------------------------------------------------------------------

import { useMemo } from "react";

const GRADIENT_STEPS = 12;

/**
 * @param {object} props
 * @param {"sequential"|"diverging"} props.kind
 * @param {(t: number, alpha?: number) => string} props.sample - one of
 *   thermalColor/divergingColor from lib/colorScale.js, called with a
 *   normalized position (0..1 for sequential, -1..1 for diverging)
 * @param {{min: number, max: number}} props.extent - real data extent
 * @param {string} [props.unit] - e.g. "°C"
 * @param {string} [props.label] - e.g. "Sea Surface Temp."
 */
export default function TemperatureLegend({ kind, sample, extent, unit = "", label }) {
  const gradientId = useMemo(
    () => `temp-legend-gradient-${Math.random().toString(36).slice(2, 9)}`,
    []
  );

  const stops = useMemo(() => {
    const positions = Array.from({ length: GRADIENT_STEPS + 1 }, (_, i) => i / GRADIENT_STEPS);
    return positions.map((p) => {
      const t = kind === "diverging" ? p * 2 - 1 : p;
      return { offset: `${p * 100}%`, color: sample(t) };
    });
  }, [kind, sample]);

  const midLabel = kind === "diverging" ? "0" : ((extent.min + extent.max) / 2).toFixed(1);
  const isDiverging = kind === "diverging";

  return (
    <div className="flex flex-col gap-1">
      {label && <p className="text-caption text-text-muted">{label}</p>}
      <svg width="120" height="10" viewBox="0 0 120 10" aria-hidden="true">
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
            {stops.map((stop) => (
              <stop key={stop.offset} offset={stop.offset} stopColor={stop.color} />
            ))}
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="120" height="10" rx="2" fill={`url(#${gradientId})`} />
      </svg>
      <div className="flex justify-between text-caption font-mono text-text-muted" style={{ width: 120 }}>
        <span>{extent.min.toFixed(1)}{unit}</span>
        <span>{midLabel}{unit}</span>
        <span>{extent.max.toFixed(1)}{unit}</span>
      </div>
      {/* Diverging fields (anomaly) get an explicit sign legend — the
          numeric min/max row above already carries the real values, but a
          signed field's most important read is "warmer or cooler than
          climatology", which a colorbar alone doesn't spell out. */}
      {isDiverging && (
        <div className="flex justify-between text-caption" style={{ width: 120 }}>
          <span className="flex items-center gap-1 text-[#67A9CF]">
            <span aria-hidden="true">−</span> cooler
          </span>
          <span className="flex items-center gap-1 text-[#EF8A62]">
            warmer <span aria-hidden="true">+</span>
          </span>
        </div>
      )}
    </div>
  );
}