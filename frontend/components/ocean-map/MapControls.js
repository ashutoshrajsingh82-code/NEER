"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — MapControls  (Phase 35D1)
//
// The user-facing control surface for the NEER scientific ocean map:
// 1. Temperature layer visibility
// 2. Temperature anomaly layer visibility
// 3. Grid overlay visibility (0.25° × 0.25° scientific grid)
// 4. Reset/fit map to the NEER domain (45°E–105°E, 5°N–30°N)
//
// Grouped behind one compact trigger so the map surface itself stays uncluttered
// while providing clear active/inactive states, hover states, keyboard focus
// rings, accessible labels, Lucide React icons, and tooltips.
//
// Self-contained: owns its own open/close state, outside-click dismissal,
// and Escape-to-close. Supports both explicit convenience props
// (temperatureVisible, anomalyVisible, gridVisible, onReset) AND a custom
// `layers` list for extensible layer setups.
// -----------------------------------------------------------------------------

import { memo, useEffect, useId, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Layers,
  LayoutGrid,
  RotateCcw,
  Thermometer,
  TrendingUp,
} from "lucide-react";
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

/**
 * @param {object} props
 * @param {boolean} [props.temperatureVisible] - visibility of temperature field
 * @param {(next: boolean) => void} [props.onToggleTemperature]
 * @param {boolean} [props.anomalyVisible] - visibility of temperature anomaly field
 * @param {(next: boolean) => void} [props.onToggleAnomaly]
 * @param {boolean} [props.gridVisible] - visibility of 0.25° scientific grid
 * @param {(next: boolean) => void} [props.onToggleGrid]
 * @param {MapControlLayer[]} [props.layers] - custom primary scientific-layer toggles
 * @param {MapControlLayer[]} [props.baseLayers] - secondary base-map chrome toggles
 * @param {() => void} [props.onReset] - callback to reset view / fit to NEER domain
 * @param {string} [props.resetLabel="Reset / fit to NEER domain"]
 * @param {string} [props.className]
 */
export default function MapControls({
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
  const menuId = useId();

  // If `layers` is not passed, build the standard scientific layers from individual props
  const primaryLayers = useMemo(() => {
    if (Array.isArray(layers)) return layers;

    return [
      {
        key: "temperature-field",
        icon: Thermometer,
        label: "Temperature layer",
        description: "Sea-surface temperature field",
        active: !!temperatureVisible,
        onToggle: onToggleTemperature,
      },
      {
        key: "anomaly-field",
        icon: TrendingUp,
        label: "Temperature anomaly layer",
        description: "Temperature anomaly relative to climatology",
        active: !!anomalyVisible,
        onToggle: onToggleAnomaly,
      },
      {
        key: "scientific-grid",
        icon: LayoutGrid,
        label: "0.25° scientific grid",
        description: "NEER native 0.25° × 0.25° grid overlay",
        active: !!gridVisible,
        onToggle: onToggleGrid,
      },
    ];
  }, [
    layers,
    temperatureVisible,
    onToggleTemperature,
    anomalyVisible,
    onToggleAnomaly,
    gridVisible,
    onToggleGrid,
  ]);

  useEffect(() => {
    if (!open) return undefined;
    function handleOutside(event) {
      if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false);
    }
    function handleEscape(event) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", handleOutside);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={cn("relative inline-flex", className)}>
      <LayerToggle
        icon={Layers}
        label="Map layer controls"
        description="Map layer & grid visibility controls"
        active={open}
        aria-expanded={open}
        aria-controls={menuId}
        aria-haspopup="menu"
        onToggle={() => setOpen((prev) => !prev)}
      />

      <AnimatePresence>
        {open && (
          <motion.div
            id={menuId}
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={scaleIn}
            role="menu"
            aria-label="Map layer controls"
            className={cn(
              "absolute left-0 top-full z-raised mt-2 flex w-60 sm:w-64 max-w-[calc(100vw-2rem)] flex-col gap-2 rounded-md border border-border-strong",
              "bg-surface-overlay/95 backdrop-blur-md p-2.5 shadow-raised"
            )}
          >
            <div>
              <p className="px-1.5 pb-1 text-caption font-semibold uppercase tracking-widest text-text-muted">
                Scientific layers
              </p>
              <div className="flex flex-col gap-0.5">
                {primaryLayers.map((layer) => (
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
            </div>

            {baseLayers.length > 0 && (
              <div className="border-t border-border-subtle pt-1.5">
                <p className="px-1.5 pb-1 text-caption font-semibold uppercase tracking-widest text-text-muted">
                  Base map
                </p>
                <div className="flex flex-col gap-0.5">
                  {baseLayers.map((layer) => (
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
              </div>
            )}

            {onReset && (
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  onReset();
                  setOpen(false);
                }}
                className={cn(
                  "flex w-full items-center gap-2 rounded-sm border-t border-border-subtle px-1.5 pt-2 pb-1 text-small",
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