"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — MapControls  (Phase 35D1)
//
// The layer control surface for the NEER scientific ocean map:
//   1. Temperature layer visibility
//   2. Temperature anomaly layer visibility
//   3. Grid overlay visibility (0.25° × 0.25° scientific grid)
//   4. Reset / fit map to the NEER domain (45°E–105°E, 5°N–30°N)
//
// Grouped behind one compact icon trigger so the map surface stays
// uncluttered; the panel lists each layer with its state, a one-line
// description, and a switch. Reset is optional (`onReset`): OceanMap already
// keeps reset one click away next to zoom, so it does not pass it here and
// end up with two reset buttons.
//
// Pattern: a *disclosure* (button with aria-expanded/aria-controls revealing
// a labelled group of switches) — not role="menu", because the content is
// switches, not menu items, and menu semantics would promise arrow-key
// behavior that isn't there. Escape closes it and returns focus to the
// trigger; a pointer press outside closes it (pointerdown, so it also works
// for touch).
//
// Supports the explicit convenience props (temperatureVisible, anomalyVisible,
// gridVisible, ...) AND a custom `layers` list for extensible layer setups.
// -----------------------------------------------------------------------------

import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Layers, LayoutGrid, RotateCcw, Thermometer, TrendingUp } from "lucide-react";
import { cn } from "@/lib/cn";
import { scaleIn } from "@/lib/motion";
import LayerToggle from "./LayerToggle";

/**
 * @typedef {object} MapControlLayer
 * @property {string} key
 * @property {React.ComponentType} icon
 * @property {string} label
 * @property {string} [description]
 * @property {boolean} active
 * @property {boolean} [disabled]
 * @property {string} [disabledReason]
 * @property {(next: boolean) => void} onToggle
 */

const SECTION_HEADING = "px-1.5 pb-1 text-caption font-semibold uppercase tracking-widest text-text-muted";

function LayerList({ layers }) {
  return (
    <div className="flex flex-col gap-0.5">
      {layers.map((layer) => (
        <LayerToggle
          key={layer.key}
          layout="row"
          icon={layer.icon}
          label={layer.label}
          description={layer.description}
          active={layer.active}
          disabled={layer.disabled}
          disabledReason={layer.disabledReason}
          onToggle={layer.onToggle}
        />
      ))}
    </div>
  );
}

/**
 * @param {object} props
 * @param {boolean} [props.temperatureVisible]
 * @param {(next: boolean) => void} [props.onToggleTemperature]
 * @param {boolean} [props.anomalyVisible]
 * @param {(next: boolean) => void} [props.onToggleAnomaly]
 * @param {boolean} [props.gridVisible]
 * @param {(next: boolean) => void} [props.onToggleGrid]
 * @param {MapControlLayer[]} [props.layers] - custom primary scientific-layer toggles (replaces the three above)
 * @param {MapControlLayer[]} [props.baseLayers] - secondary base-map chrome toggles
 * @param {() => void} [props.onReset] - reset view / fit to the NEER domain; row omitted when not provided
 * @param {string} [props.resetLabel]
 * @param {string} [props.className]
 */
function MapControls({
  temperatureVisible,
  onToggleTemperature,
  anomalyVisible,
  onToggleAnomaly,
  gridVisible,
  onToggleGrid,
  layers,
  baseLayers = [],
  onReset,
  resetLabel = "Reset / fit to NEER domain",
  className,
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const panelId = useId();

  const primaryLayers = useMemo(() => {
    if (Array.isArray(layers)) return layers;
    return [
      {
        key: "temperature-field",
        icon: Thermometer,
        label: "Temperature layer",
        description: "Sea-surface temperature field",
        active: Boolean(temperatureVisible),
        onToggle: onToggleTemperature,
      },
      {
        key: "anomaly-field",
        icon: TrendingUp,
        label: "Temperature anomaly layer",
        description: "Departure from climatology",
        active: Boolean(anomalyVisible),
        onToggle: onToggleAnomaly,
      },
      {
        key: "scientific-grid",
        icon: LayoutGrid,
        label: "0.25° scientific grid",
        description: "Native 0.25° × 0.25° cell grid",
        active: Boolean(gridVisible),
        onToggle: onToggleGrid,
      },
    ];
  }, [layers, temperatureVisible, onToggleTemperature, anomalyVisible, onToggleAnomaly, gridVisible, onToggleGrid]);

  const activeCount = primaryLayers.filter((layer) => layer.active).length;

  const close = useCallback((returnFocus = false) => {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    function handleOutside(event) {
      if (rootRef.current && !rootRef.current.contains(event.target)) close();
    }
    function handleEscape(event) {
      if (event.key === "Escape") close(true);
    }
    document.addEventListener("pointerdown", handleOutside);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("pointerdown", handleOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [open, close]);

  return (
    <div ref={rootRef} className={cn("relative inline-flex", className)}>
      <LayerToggle
        ref={triggerRef}
        icon={Layers}
        label="Map layers"
        description={`Map layers · ${activeCount} of ${primaryLayers.length} on`}
        active={open}
        aria-pressed={undefined}
        aria-expanded={open}
        aria-controls={panelId}
        onToggle={() => setOpen((prev) => !prev)}
      />

      <AnimatePresence>
        {open && (
          <motion.div
            id={panelId}
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={scaleIn}
            role="group"
            aria-label="Map layer controls"
            className={cn(
              "absolute left-0 top-full z-raised mt-2 flex w-64 max-w-[calc(100vw-2rem)] flex-col gap-2 rounded-md border border-border-strong",
              "bg-surface-overlay/95 p-2.5 shadow-raised backdrop-blur-md"
            )}
          >
            <div>
              <p className={SECTION_HEADING}>Scientific layers</p>
              <LayerList layers={primaryLayers} />
            </div>

            {baseLayers.length > 0 && (
              <div className="border-t border-border-subtle pt-1.5">
                <p className={SECTION_HEADING}>Base map</p>
                <LayerList layers={baseLayers} />
              </div>
            )}

            {onReset && (
              <button
                type="button"
                onClick={() => {
                  onReset();
                  close(true);
                }}
                className={cn(
                  "flex w-full items-center gap-2 rounded-sm border-t border-border-subtle px-2 pb-2 pt-2.5 text-small sm:pb-1.5",
                  "text-text-secondary neer-transition hover:bg-surface-raised hover:text-accent-300",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-overlay"
                )}
              >
                <RotateCcw size={14} strokeWidth={1.75} aria-hidden="true" className="shrink-0 text-text-muted" />
                <span className="truncate">{resetLabel}</span>
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default memo(MapControls);