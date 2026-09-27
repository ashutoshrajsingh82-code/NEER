// -----------------------------------------------------------------------------
// NEER OceanMap — MapLegend  (Phase 35D1)
//
// The map's scientific legend: whichever field is currently displayed
// (temperature or anomaly today; any future backend-supported variable
// tomorrow, see useOceanMapLayer.js's LAYER_SUPPORT), described honestly for
// whatever state that field is actually in. It updates automatically
// whenever the active variable tab, date, depth, or viewport changes,
// because it's driven entirely by the same `layer` (useOceanMapLayer result)
// and `extent` (computeFiniteExtent of that layer's own values) the map
// itself draws from — never a second, independently-tracked copy of "what's
// on screen" that could drift out of sync with it.
//
// Scientific-integrity contract (same one TemperatureLayer.js/TemperatureLegend.js
// already follow): the colorbar's min/max are only ever the real finite
// extent of the backend-provided grid. When there's nothing real to show —
// no date selected, still loading, the request errored, or the request
// succeeded but returned no usable field (e.g. no climatology to derive an
// anomaly from) — this renders that state in words instead of a colorbar,
// rather than inventing a placeholder range.
//
// Composed from TemperatureLegend.js (just the gradient bar + numeric
// min/max/unit) plus the design system's StatusIndicator for the one-line
// status text, so the two concerns — "what does the color scale mean" and
// "is this actually live data right now" — stay visually distinct but
// always appear together.
// -----------------------------------------------------------------------------

import { StatusIndicator } from "@/components/ui";
import { cn } from "@/lib/cn";
import TemperatureLegend from "./TemperatureLegend";

/**
 * Turns a useOceanMapLayer() result into the legend's one-line status text
 * plus a StatusIndicator status key, using the same online/processing/
 * warning/error/offline vocabulary as the rest of the design system for
 * every possible layer state (unsupported, too-large-a-region, loading,
 * errored, missing-data, or live).
 */
function describeLayerStatus(layer, activeLabel) {
  if (!layer.supported) {
    return { status: "offline", text: `${layer.reason} — demo interaction only` };
  }
  if (layer.tooLargeForCellLimit) {
    return {
      status: "warning",
      text: "Zoom in to load live grid data (visible region too large for one request)",
    };
  }
  if (layer.isLoading) {
    return { status: "processing", text: "Loading live grid data…" };
  }
  if (layer.isError) {
    return { status: "error", text: `Grid request failed — ${layer.error?.message ?? "unknown error"}` };
  }
  if (layer.values) {
    const rows = layer.grid?.lat?.length ?? 0;
    const cols = layer.grid?.lon?.length ?? 0;
    return { status: "online", text: `Live grid loaded (${rows}×${cols} cells)` };
  }
  if (layer.status === "success") {
    return {
      status: "warning",
      text: `No ${activeLabel?.toLowerCase() ?? "field"} data for this date/depth — missing, not fabricated`,
    };
  }
  return { status: "unknown", text: "No date selected — demo interaction only" };
}

/**
 * @param {object} props
 * @param {string} [props.activeLabel] - display name of the active variable tab, e.g. "Sea Surface Temp."
 * @param {{kind: "sequential"|"diverging", unit: string}|null} [props.colorConfig] - null for a variable with no color scale defined (unsupported layer)
 * @param {(t: number, alpha?: number) => string} [props.sample] - thermalColor/divergingColor matching colorConfig.kind
 * @param {{min: number, max: number}|null} [props.extent] - real finite extent of the currently loaded field
 * @param {ReturnType<typeof import("./useOceanMapLayer").useOceanMapLayer>} props.layer
 * @param {boolean} [props.fieldVisible=true] - the field layer's own visibility toggle; the colorbar is hidden
 *   when the field itself is hidden (nothing to key the scale to on screen), but the live/loading/error status
 *   line still reflects the real backend state either way
 * @param {string} [props.className]
 */
export default function MapLegend({
  activeLabel,
  colorConfig,
  sample,
  extent,
  layer,
  fieldVisible = true,
  className,
}) {
  const layerStatus = describeLayerStatus(layer, activeLabel);
  const showGradient = fieldVisible && Boolean(colorConfig && sample && extent);
  const showLoadingNote = fieldVisible && Boolean(colorConfig) && !extent && layer.isLoading;
  const showEmptyNote = fieldVisible && Boolean(colorConfig) && !extent && !layer.isLoading && !showGradient;

  return (
    <div className={cn("flex flex-col gap-2", className)} aria-live="polite">
      {showGradient && (
        <TemperatureLegend
          kind={colorConfig.kind}
          sample={sample}
          extent={extent}
          unit={colorConfig.unit}
          label={`${activeLabel} (live)`}
        />
      )}

      {/* Loading: real request in flight — never draw a guessed range while
          waiting for the actual one. */}
      {showLoadingNote && (
        <div className="flex flex-col gap-1">
          {activeLabel && <p className="text-caption text-text-muted">{activeLabel}</p>}
          <div className="h-2.5 w-[120px] animate-pulse rounded-full bg-surface-raised" aria-hidden="true" />
        </div>
      )}

      {/* Empty: the request resolved but there is genuinely no field to
          show a scale for (e.g. anomaly with no climatology loaded for this
          date) — stated in words, not a fabricated bar. */}
      {showEmptyNote && (
        <p className="text-caption text-text-muted">
          No {activeLabel?.toLowerCase() ?? "field"} data range available for this date/depth.
        </p>
      )}

      <StatusIndicator status={layerStatus.status} label={`${activeLabel ?? "Layer"} · ${layerStatus.text}`} size="sm" />
    </div>
  );
}