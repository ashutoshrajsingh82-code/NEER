"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — TemperatureLegend  (Phase 35D1)
//
// A continuous colorbar for scientific raster fields (SST or temperature anomaly).
// Purely presentational — the min/max it labels are always the real finite
// extent of the backend-provided grid (never a fixed/hard-coded physical range),
// so the bar always matches what's actually on screen.
//
// For anomaly (diverging kind):
//   - Distinguishes positive and negative anomaly with explicit signs (+/-)
//   - Displays zero baseline at center
//   - Clearly notes cooler (-) and warmer (+) directions
// -----------------------------------------------------------------------------

import { useId, useMemo } from "react";
import { cn } from "@/lib/cn";
import { divergingColor, thermalColor } from "@/lib/colorScale";
import { formatSigned } from "@/lib/format";

const GRADIENT_STEPS = 16;

/**
 * @param {object} props
 * @param {"sequential"|"diverging"} [props.kind="sequential"]
 * @param {(t: number, alpha?: number) => string} [props.sample] - color sampling function
 * @param {{min: number, max: number}} [props.extent] - real data extent
 * @param {number} [props.min] - alternative to extent.min
 * @param {number} [props.max] - alternative to extent.max
 * @param {string} [props.unit="°C"] - e.g. "°C"
 * @param {string} [props.label] - e.g. "Sea Surface Temp."
 * @param {string} [props.className]
 */
export default function TemperatureLegend({
  kind = "sequential",
  sample,
  extent,
  min,
  max,
  unit = "°C",
  label,
  className,
}) {
  const gradientId = useId();
  const isDiverging = kind === "diverging";

  const effectiveMin = extent?.min ?? min;
  const effectiveMax = extent?.max ?? max;

  const colorSampler = sample ?? (isDiverging ? divergingColor : thermalColor);

  const stops = useMemo(() => {
    const positions = Array.from({ length: GRADIENT_STEPS + 1 }, (_, i) => i / GRADIENT_STEPS);
    return positions.map((p) => {
      const t = isDiverging ? p * 2 - 1 : p;
      return { offset: `${(p * 100).toFixed(1)}%`, color: colorSampler(t) };
    });
  }, [isDiverging, colorSampler]);

  if (typeof effectiveMin !== "number" || typeof effectiveMax !== "number" || !Number.isFinite(effectiveMin) || !Number.isFinite(effectiveMax)) {
    return null;
  }

  const minFormatted = isDiverging
    ? `${formatSigned(effectiveMin, 1)}${unit}`
    : `${effectiveMin.toFixed(1)}${unit}`;

  const maxFormatted = isDiverging
    ? `${formatSigned(effectiveMax, 1)}${unit}`
    : `${effectiveMax.toFixed(1)}${unit}`;

  const midFormatted = isDiverging
    ? `0.0${unit}`
    : `${((effectiveMin + effectiveMax) / 2).toFixed(1)}${unit}`;

  return (
    <div className={cn("flex w-full min-w-[140px] max-w-[220px] flex-col gap-1.5", className)}>
      {label && (
        <p className="text-caption font-medium tracking-wide text-text-muted">
          {label}
        </p>
      )}

      {/* Color gradient bar */}
      <svg
        className="h-2.5 w-full rounded-sm overflow-hidden"
        viewBox="0 0 160 10"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
            {stops.map((stop) => (
              <stop key={stop.offset} offset={stop.offset} stopColor={stop.color} />
            ))}
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="160" height="10" rx="2" fill={`url(#${gradientId})`} />
      </svg>

      {/* Numeric endpoints & midpoint */}
      <div className="flex justify-between text-caption font-mono tabular-nums text-text-muted">
        <span title={`Minimum: ${minFormatted}`}>{minFormatted}</span>
        <span title={`Center: ${midFormatted}`} className="text-text-disabled">
          {midFormatted}
        </span>
        <span title={`Maximum: ${maxFormatted}`}>{maxFormatted}</span>
      </div>

      {/* For anomaly (diverging), render explicit sign and qualitative indicators */}
      {isDiverging && (
        <div className="flex items-center justify-between text-caption font-mono text-[11px] leading-none pt-0.5">
          <span className="flex items-center gap-0.5 font-medium text-[#67A9CF]">
            <span aria-hidden="true">−</span> cooler
          </span>
          <span className="text-text-disabled text-[10px]">baseline</span>
          <span className="flex items-center gap-0.5 font-medium text-[#EF8A62]">
            warmer <span aria-hidden="true">+</span>
          </span>
        </div>
      )}
    </div>
  );
}