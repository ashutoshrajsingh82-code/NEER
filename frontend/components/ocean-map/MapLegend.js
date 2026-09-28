"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — MapLegend  (Phase 35D1)
//
// The map's scientific legend: dynamically reflects whichever field is currently
// displayed (sea-surface temperature or temperature anomaly).
//
// Scientific-integrity contract:
// - Temperature: shows sequential thermal color scale with real finite min/max from
//   the backend data and appropriate units (°C).
// - Anomaly: shows diverging color scale, clearly distinguishing positive (+)
//   and negative (-) anomaly, zero baseline, and appropriate units (°C).
// - Do NOT hard-code scientifically misleading ranges.
// - If backend metadata/grid provides the range, use it directly.
// - If data is unavailable (loading, error, no data for date/depth), show an
//   honest empty or loading state — do NOT invent values.
// - Updates automatically when the active scientific layer changes.
// -----------------------------------------------------------------------------

import { memo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { StatusIndicator } from "@/components/ui";
import { cn } from "@/lib/cn";
import {
  VARIABLE_COLOR_CONFIG,
  computeFiniteExtent,
  divergingColor,
  thermalColor,
} from "@/lib/colorScale";
import TemperatureLegend from "./TemperatureLegend";

/**
 * Turns a layer object or direct status props into the legend's one-line status text
 * plus a StatusIndicator status key.
 */
function describeLayerStatus(layer, label, isLoading, isError, error) {
  if (isLoading || layer?.isLoading) {
    return { status: "processing", text: "Loading live grid data…" };
  }
  if (isError || layer?.isError) {
    const msg = error?.message ?? (typeof error === "string" ? error : layer?.error?.message) ?? "Grid request failed";
    return { status: "error", text: msg };
  }
  if (!layer) {
    return { status: "online", text: "Ready" };
  }
  if (layer.supported === false) {
    return { status: "offline", text: `${layer.reason ?? "Unsupported layer"} — demo interaction only` };
  }
  if (layer.tooLargeForCellLimit) {
    return {
      status: "warning",
      text: "Zoom in to load live grid data (visible region exceeds cell limit)",
    };
  }
  if (layer.values) {
    const rows = layer.grid?.lat?.length ?? 0;
    const cols = layer.grid?.lon?.length ?? 0;
    return { status: "online", text: `Live grid loaded (${rows}×${cols} cells)` };
  }
  if (layer.status === "success") {
    return {
      status: "warning",
      text: `No ${label.toLowerCase()} data for this date/depth — missing, not fabricated`,
    };
  }
  return { status: "unknown", text: "No date selected — demo interaction only" };
}

/**
 * @param {object} props
 * @param {string} [props.variable] - "sst" | "temperature" | "anomaly"
 * @param {string} [props.activeLabel] - display name, e.g. "Sea Surface Temp."
 * @param {{kind: "sequential"|"diverging", unit: string}|null} [props.colorConfig]
 * @param {(t: number, alpha?: number) => string} [props.sample]
 * @param {{min: number, max: number}|null} [props.extent] - real finite extent from backend
 * @param {number} [props.min]
 * @param {number} [props.max]
 * @param {string} [props.unit] - default "°C"
 * @param {object} [props.layer] - useOceanMapLayer result
 * @param {boolean} [props.fieldVisible=true] - whether scientific field is visible on map
 * @param {boolean} [props.isLoading]
 * @param {boolean} [props.isError]
 * @param {string|Error} [props.error]
 * @param {string} [props.className]
 */
export default function MapLegend({
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
  className,
}) {
  // Normalize variable and label
  const effectiveVar = variable ?? (colorConfig?.kind === "diverging" ? "anomaly" : "sst");
  const isAnomaly = effectiveVar === "anomaly" || colorConfig?.kind === "diverging";
  const defaultLabel = isAnomaly ? "Temperature Anomaly" : "Sea Surface Temp.";
  const label = activeLabel ?? defaultLabel;

  // Determine color configuration and sampling
  const config =
    colorConfig ??
    VARIABLE_COLOR_CONFIG[effectiveVar] ??
    (isAnomaly ? VARIABLE_COLOR_CONFIG.anomaly : VARIABLE_COLOR_CONFIG.sst);

  const effectiveUnit = unit ?? config?.unit ?? "°C";
  const sampler = sample ?? (config?.kind === "diverging" ? divergingColor : thermalColor);

  // Compute or extract real data extent
  let effectiveExtent = null;
  if (extent && Number.isFinite(extent.min) && Number.isFinite(extent.max)) {
    effectiveExtent = extent;
  } else if (typeof min === "number" && typeof max === "number" && Number.isFinite(min) && Number.isFinite(max)) {
    effectiveExtent = { min, max };
  } else if (layer?.values) {
    effectiveExtent = computeFiniteExtent(layer.values);
  }

  const effectiveLoading = isLoading ?? layer?.isLoading ?? false;
  const effectiveIsError = isError ?? layer?.isError ?? false;
  const layerStatus = describeLayerStatus(layer, label, effectiveLoading, effectiveIsError, error);

  const showGradient = fieldVisible && Boolean(effectiveExtent && config);
  const showLoading = fieldVisible && effectiveLoading && !showGradient;
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      className={cn("flex flex-col gap-2 text-text-primary", className)}
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="flex items-center justify-between gap-2 sm:hidden">
        <span className="text-caption font-semibold uppercase tracking-wider text-text-muted">Legend</span>
        <button
          type="button"
          onClick={() => setCollapsed((prev) => !prev)}
          aria-expanded={!collapsed}
          aria-label={collapsed ? "Expand map legend" : "Collapse map legend"}
          className="flex h-5 w-5 items-center justify-center rounded text-text-muted hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-400"
        >
          {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>
      </div>

      {!collapsed && (
        <>
          {/* 1. Visible color scale with real backend min/max and units */}
          {showGradient && (
            <TemperatureLegend
              kind={config.kind}
              sample={sampler}
              extent={effectiveExtent}
              unit={effectiveUnit}
              label={`${label} (${effectiveExtent ? "live backend range" : "live"})`}
            />
          )}

          {/* 2. Loading state: request in flight, do not draw fabricated numbers */}
          {showLoading && (
            <div className="flex flex-col gap-1.5 py-0.5">
              <p className="text-caption font-medium text-text-muted">{label}</p>
              <div className="h-2.5 w-36 sm:w-44 animate-pulse rounded bg-surface-raised" aria-hidden="true" />
              <p className="text-caption text-text-disabled">Loading backend data range…</p>
            </div>
          )}

          {/* 3. Empty state: request finished but no finite data exists */}
          {showEmpty && (
            <div className="flex flex-col gap-0.5 py-0.5">
              <p className="text-caption font-medium text-text-muted">{label}</p>
              <p className="text-caption text-text-disabled">
                No data range available for this selection — missing, not fabricated.
              </p>
            </div>
          )}

          {/* 4. Layer hidden state: field exists but user toggled it off in MapControls */}
          {!fieldVisible && (
            <div className="py-0.5">
              <p className="text-caption italic text-text-disabled">
                {label} layer hidden in controls
              </p>
            </div>
          )}
        </>
      )}

      {/* Status line communicating backend state honestly */}
      <StatusIndicator
        status={layerStatus.status}
        label={`${label} · ${layerStatus.text}`}
        size="sm"
      />
    </div>
  );
}

export default memo(MapLegend);