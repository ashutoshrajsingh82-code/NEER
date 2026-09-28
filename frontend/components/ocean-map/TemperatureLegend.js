"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — TemperatureLegend  (Phase 35D1)
//
// A continuous colorbar for scientific raster fields (SST or temperature
// anomaly). Purely presentational — every number it labels is derived from
// the real finite extent of the backend-provided grid the caller passes in,
// never a fixed/hard-coded physical range.
//
// The labels MUST describe the same mapping the map actually paints with
// (makeColorMapper in lib/colorScale.js), otherwise the legend would be
// scientifically wrong even though every number in it is "real":
//
//   sequential (SST):  color(t) with t = (value - min) / (max - min)
//                      -> bar runs min ... max, linear.
//
//   diverging (anomaly): color(t) with t = value / L, where
//                      L = getDivergingLimit(extent) = max(|min|, |max|)
//                      -> bar runs -L ... 0 ... +L, symmetric about zero.
//                      Labelling the ends with the raw min/max instead
//                      (e.g. -0.5 ... +2.0) would put the "0" label at the
//                      wrong place and mislabel every color. The actual data
//                      span is therefore shown separately, as a thin marker
//                      under the bar plus a "Loaded data" line.
// -----------------------------------------------------------------------------

import { memo, useId, useMemo } from "react";
import { cn } from "@/lib/cn";
import { divergingColor, getDivergingLimit, thermalColor } from "@/lib/colorScale";
import { formatSigned } from "@/lib/format";

const GRADIENT_STEPS = 16;

/**
 * Decimal places for legend labels, chosen from the size of the range being
 * labelled so a narrow range (e.g. an anomaly field of ±0.04 °C) doesn't
 * collapse to a meaningless "0.0" while a wide one isn't cluttered with
 * false precision.
 * @param {number} range - full span the labels cover
 * @returns {1|2|3}
 */
export function legendDigits(range) {
  if (!Number.isFinite(range) || range <= 0) return 2;
  if (range >= 2) return 1;
  if (range >= 0.2) return 2;
  return 3;
}

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * @param {object} props
 * @param {"sequential"|"diverging"} [props.kind="sequential"]
 * @param {(t: number, alpha?: number) => string} [props.sample] - color sampling function
 * @param {{min: number, max: number}} [props.extent] - real data extent
 * @param {number} [props.min] - alternative to extent.min
 * @param {number} [props.max] - alternative to extent.max
 * @param {string} [props.unit="°C"]
 * @param {string} [props.label] - e.g. "Sea Surface Temp."
 * @param {string} [props.className]
 */
function TemperatureLegend({
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

  const dataMin = extent?.min ?? min;
  const dataMax = extent?.max ?? max;
  const hasRange = isFiniteNumber(dataMin) && isFiniteNumber(dataMax);

  const colorSampler = sample ?? (isDiverging ? divergingColor : thermalColor);

  const stops = useMemo(() => {
    const positions = Array.from({ length: GRADIENT_STEPS + 1 }, (_, i) => i / GRADIENT_STEPS);
    return positions.map((p) => {
      const t = isDiverging ? p * 2 - 1 : p;
      return { offset: `${(p * 100).toFixed(1)}%`, color: colorSampler(t) };
    });
  }, [isDiverging, colorSampler]);

  if (!hasRange) return null;

  // Scale the bar actually covers (see header comment).
  const limit = isDiverging ? getDivergingLimit({ min: dataMin, max: dataMax }) : 0;
  const scaleMin = isDiverging ? -limit : dataMin;
  const scaleMax = isDiverging ? limit : dataMax;
  const scaleMid = isDiverging ? 0 : (dataMin + dataMax) / 2;

  const digits = legendDigits(scaleMax - scaleMin);
  const fmt = (v) => (isDiverging ? formatSigned(v, digits) : v.toFixed(digits));

  const minText = `${fmt(scaleMin)} ${unit}`;
  const midText = `${fmt(scaleMid)} ${unit}`;
  const maxText = `${fmt(scaleMax)} ${unit}`;

  // Where the real data sits inside the (symmetric) diverging scale, as a
  // fraction of bar width. Only meaningful when it's a strict sub-span —
  // for a sequential bar the data span IS the bar.
  const span = scaleMax - scaleMin || 1;
  const dataStart = (dataMin - scaleMin) / span;
  const dataEnd = (dataMax - scaleMin) / span;
  const showDataSpan = isDiverging && dataEnd - dataStart < 0.98;

  return (
    <div
      role="group"
      aria-label={`${label ?? "Field"} color scale, ${minText} to ${maxText}`}
      className={cn("flex w-full min-w-[140px] max-w-[220px] flex-col gap-1.5", className)}
    >
      {label && <p className="text-caption font-medium tracking-wide text-text-muted">{label}</p>}

      <div className="flex flex-col gap-0.5">
        <svg
          className="h-2.5 w-full overflow-hidden rounded-sm border border-border-subtle"
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
          <rect x="0" y="0" width="160" height="10" fill={`url(#${gradientId})`} />
          {isDiverging && (
            <line x1="80" y1="0" x2="80" y2="10" stroke="rgba(4,18,29,0.55)" strokeWidth="0.75" />
          )}
        </svg>

        {showDataSpan && (
          <div className="relative h-[3px] w-full" aria-hidden="true">
            <span
              className="absolute inset-y-0 rounded-full bg-accent-300/80"
              style={{
                left: `${dataStart * 100}%`,
                width: `${Math.max((dataEnd - dataStart) * 100, 1.5)}%`,
              }}
            />
          </div>
        )}
      </div>

      {/* Numeric endpoints & midpoint */}
      <div className="flex justify-between font-mono text-caption tabular-nums text-text-muted">
        <span>{minText}</span>
        <span className="text-text-disabled">{midText}</span>
        <span>{maxText}</span>
      </div>

      {isDiverging && (
        <>
          <div className="flex items-center justify-between pt-0.5 font-mono text-[11px] leading-none">
            <span className="font-medium text-[#67A9CF]">
              <span aria-hidden="true">− </span>cooler
            </span>
            <span className="font-medium text-[#EF8A62]">
              warmer<span aria-hidden="true"> +</span>
            </span>
          </div>
          <p className="font-mono text-caption tabular-nums text-text-disabled">
            Loaded data: {fmt(dataMin)} to {fmt(dataMax)} {unit}
          </p>
        </>
      )}
    </div>
  );
}

export default memo(TemperatureLegend);