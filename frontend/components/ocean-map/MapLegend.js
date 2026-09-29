"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — MapLegend  (Phase 35D1)
//
// The map's scientific legend. It always describes the field that is
// CURRENTLY being drawn — sea-surface temperature (sequential scale) or
// temperature anomaly (diverging scale) — and re-derives itself whenever the
// active layer, its data, or its visibility changes.
//
// Scientific-integrity contract
//   - The backend's GET /reconstruct/grid response carries no min/max
//     metadata, so the range shown is always the real finite extent of the
//     grid that was actually loaded for the current view (computeFiniteExtent),
//     or an `extent`/`min`/`max` the caller passes explicitly. Never a
//     hard-coded physical range. Because the map colors are fitted to that
//     same extent, the legend is labelled "loaded view" so it is clear the
//     scale is relative to what's on screen, not a fixed climatological one.
//   - When there is no data (loading, error, unsupported layer, no date,
//     region too large, masked/missing field) the legend says so plainly and
//     draws NO gradient and NO numbers — see deriveLegendState().
//   - Units come from VARIABLE_COLOR_CONFIG (°C), not from the data.
// -----------------------------------------------------------------------------

import { memo, useId, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/cn";
import {
  VARIABLE_COLOR_CONFIG,
  computeFiniteExtent,
  divergingColor,
  thermalColor,
} from "@/lib/colorScale";
import TemperatureLegend from "./TemperatureLegend";
import { MAX_GRID_POINTS } from "./useOceanMapLayer";

/** Every distinct thing the legend can be showing. */
export const LEGEND_STATE = Object.freeze({
  READY: "ready",
  HIDDEN: "hidden",
  UNSUPPORTED: "unsupported",
  LOADING: "loading",
  ERROR: "error",
  ZOOM: "zoom",
  NO_DATE: "no-date",
  EMPTY: "empty",
});

const DOT_CLASS = {
  ok: "bg-success",
  busy: "bg-accent-400",
  warn: "bg-warning",
  error: "bg-error",
  idle: "bg-text-disabled",
};

/**
 * Pure decision of what the legend should show — no React, no rendering, so
 * every branch (and the "never invent a range" rule) can be unit-tested.
 *
 * @param {object} input
 * @param {boolean} input.fieldVisible - the scientific field is toggled on
 * @param {boolean} input.isAnomaly
 * @param {object|null} [input.layer] - useOceanMapLayer() result
 * @param {{min:number,max:number}|null} input.extent - real finite extent, or null
 * @param {boolean} [input.isLoading]
 * @param {boolean} [input.isError]
 * @param {string|Error|null} [input.error]
 * @returns {{state: string, dot: keyof typeof DOT_CLASS, message: string, updating: boolean}}
 */
export function deriveLegendState({ fieldVisible, isAnomaly, layer, extent, isLoading, isError, error }) {
  if (!fieldVisible) {
    return { state: LEGEND_STATE.HIDDEN, dot: "idle", message: "Layer hidden — turn it on in map controls", updating: false };
  }
  if (layer && layer.supported === false) {
    return {
      state: LEGEND_STATE.UNSUPPORTED,
      dot: "idle",
      message: `${layer.reason ?? "No backend data for this layer yet"} — no color scale to show`,
      updating: false,
    };
  }
  if (isError) {
    const message =
      (typeof error === "string" ? error : error?.message) ?? layer?.error?.message ?? "Grid request failed";
    return { state: LEGEND_STATE.ERROR, dot: "error", message, updating: false };
  }
  if (extent) {
    // A previous grid can still be on screen while the next one loads;
    // the range shown is the one that matches what's drawn.
    return {
      state: LEGEND_STATE.READY,
      dot: isLoading ? "busy" : "ok",
      message: isLoading ? "Updating…" : "",
      updating: Boolean(isLoading),
    };
  }
  if (isLoading) {
    return { state: LEGEND_STATE.LOADING, dot: "busy", message: "Loading data range…", updating: true };
  }
  if (layer?.tooLargeForCellLimit) {
    // Real numbers, not a vague hint: the view's cell count vs. the backend's limit.
    const cells = layer.estimatedCells;
    const detail = Number.isFinite(cells)
      ? ` (view is ${cells.toLocaleString("en-US")} cells; limit ${MAX_GRID_POINTS.toLocaleString("en-US")})`
      : "";
    return {
      state: LEGEND_STATE.ZOOM,
      dot: "warn",
      message: `Zoom in to load the field${detail}`,
      updating: false,
    };
  }
  if (layer && layer.status === "idle") {
    return { state: LEGEND_STATE.NO_DATE, dot: "idle", message: "Select a date to load the field", updating: false };
  }
  return {
    state: LEGEND_STATE.EMPTY,
    dot: "warn",
    message: isAnomaly
      ? "No climatology for this date — anomaly can't be derived"
      : "No data in the loaded view",
    updating: false,
  };
}

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * @param {object} props
 * @param {string} [props.variable] - "sst" | "anomaly" (other values fall back to their own label, never to SST's scale)
 * @param {string} [props.activeLabel] - display name, e.g. "Sea Surface Temp."
 * @param {{kind: "sequential"|"diverging", unit: string}|null} [props.colorConfig]
 * @param {(t: number, alpha?: number) => string} [props.sample]
 * @param {{min: number, max: number}|null} [props.extent] - real finite extent from the backend grid
 * @param {number} [props.min]
 * @param {number} [props.max]
 * @param {string} [props.unit] - overrides the variable's default unit
 * @param {object} [props.layer] - useOceanMapLayer() result
 * @param {boolean} [props.fieldVisible=true] - scientific field currently shown on the map
 * @param {boolean} [props.isLoading]
 * @param {boolean} [props.isError]
 * @param {string|Error} [props.error]
 * @param {React.ReactNode} [props.children] - extra map-key content rendered below the scale
 * @param {string} [props.className] - positioning, e.g. "absolute bottom-3 left-3"
 */
function MapLegend({
  variable,
  activeLabel,
  colorConfig,
  sample,
  extent,
  min,
  max,
  unit,
  layer,
  fieldVisible = true,
  isLoading,
  isError,
  error,
  children,
  className,
}) {
  const [collapsed, setCollapsed] = useState(false);
  const bodyId = useId();

  const config = colorConfig ?? VARIABLE_COLOR_CONFIG[variable] ?? null;
  const isAnomaly = variable === "anomaly" || config?.kind === "diverging";
  const label = activeLabel ?? (isAnomaly ? "ANOMALY" : "MODEL OUTPUT");
  const effectiveUnit = unit ?? config?.unit ?? "";
  const sampler = sample ?? (config?.kind === "diverging" ? divergingColor : thermalColor);

  // Real extent only: explicit prop -> explicit min/max -> the loaded grid.
  // Nothing here can produce a number that isn't in the backend's data.
  let effectiveExtent = null;
  if (extent && isFiniteNumber(extent.min) && isFiniteNumber(extent.max)) {
    effectiveExtent = extent;
  } else if (isFiniteNumber(min) && isFiniteNumber(max)) {
    effectiveExtent = { min, max };
  } else if (layer?.values) {
    effectiveExtent = computeFiniteExtent(layer.values);
  }
  // A layer with no color scale (e.g. salinity) can never have a gradient.
  if (!config) effectiveExtent = null;

  const legend = deriveLegendState({
    fieldVisible,
    isAnomaly,
    layer,
    extent: effectiveExtent,
    isLoading: isLoading ?? layer?.isLoading ?? false,
    isError: isError ?? layer?.isError ?? false,
    error,
  });

  const grid = layer?.grid;
  const rows = grid?.lat?.length ?? 0;
  const cols = grid?.lon?.length ?? 0;
  const isReady = legend.state === LEGEND_STATE.READY;

  return (
    <section
      aria-label="Map legend"
      className={cn(
        "flex max-w-[calc(100%-1.5rem)] flex-col gap-2 rounded-md border border-border bg-surface-base/90 px-3 py-2 text-text-primary backdrop-blur-sm",
        "sm:max-w-xs md:max-w-sm",
        className
      )}
    >
      {/* Header: the active layer, always visible so a collapsed legend
          still says what it's a legend for. Collapse only exists on small
          screens, where the card would otherwise cover too much map. */}
      <div className="flex items-center justify-between gap-2">
        <p className="min-w-0 truncate text-caption font-semibold uppercase tracking-wider text-text-muted">
          {label}
          {effectiveUnit && <span className="font-mono normal-case text-text-disabled"> · {isAnomaly ? "Temperature Anomaly" : "Temperature"} ({effectiveUnit})</span>}
        </p>
        <button
          type="button"
          onClick={() => setCollapsed((prev) => !prev)}
          aria-expanded={!collapsed}
          aria-controls={bodyId}
          aria-label={collapsed ? "Expand map legend" : "Collapse map legend"}
          className={cn(
            "flex h-6 w-6 shrink-0 items-center justify-center rounded text-text-muted neer-transition sm:hidden",
            "hover:bg-surface-raised hover:text-text-primary",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
          )}
        >
          {collapsed ? (
            <ChevronDown size={14} strokeWidth={1.75} aria-hidden="true" />
          ) : (
            <ChevronUp size={14} strokeWidth={1.75} aria-hidden="true" />
          )}
        </button>
      </div>

      <div id={bodyId} className={cn("flex-col gap-2", collapsed ? "hidden sm:flex" : "flex")}>
        {isReady && (
          <TemperatureLegend
            kind={config.kind}
            sample={sampler}
            extent={effectiveExtent}
            unit={effectiveUnit}
          />
        )}

        {legend.state === LEGEND_STATE.LOADING && (
          <div className="flex flex-col gap-1.5" aria-hidden="true">
            <div className="h-2.5 w-36 animate-pulse rounded-sm bg-surface-raised sm:w-44" />
            <div className="h-2 w-28 animate-pulse rounded-sm bg-surface-raised/70" />
          </div>
        )}

        {/* One honest status line — the reason there is (or isn't) a scale.
            A polite live region so a layer/data change is announced once;
            deliberately not on the whole legend. */}
        <p aria-live="polite" className="flex items-start gap-2 text-caption text-text-muted">
          <span
            aria-hidden="true"
            className={cn(
              "mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full",
              DOT_CLASS[legend.dot],
              legend.updating && "animate-pulse"
            )}
          />
          <span>
            {isReady
              ? legend.updating
                ? "Updating · range of loaded view"
                : `Range of loaded view${rows && cols ? ` · ${rows}×${cols} cells` : ""}`
              : legend.message}
          </span>
        </p>

        {children && <div className="border-t border-border-subtle pt-2">{children}</div>}
      </div>
    </section>
  );
}

export default memo(MapLegend);
