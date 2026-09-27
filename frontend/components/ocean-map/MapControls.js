"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — MapControls  (Phase 35D1)
//
// The map's user-facing control surface: layer-visibility toggles plus a
// reset/fit action, grouped behind one compact trigger so the map surface
// itself stays uncluttered (the spec's explicit constraint) rather than
// scattering a row of always-visible switches across it.
//
// Self-contained: owns its own open/close state, outside-click dismissal,
// and Escape-to-close, so any caller can drop it in with just a `layers`
// config and an `onReset` handler — no popover plumbing duplicated at the
// call site (OceanMap.js previously hand-rolled this itself).
//
// `layers` groups into two sections purely for legibility:
//   - the scientific layers this phase's spec asks for (temperature field,
//     anomaly field, the 0.25° scientific grid) — passed as `layers`
//   - the pre-existing base-map chrome toggles (graticule, coastline,
//     domain boundary) from earlier phases — passed as `baseLayers`, kept
//     available rather than removed, but visually secondary
//
// Each row is a LayerToggle in "row" layout; disabled rows (e.g. the
// temperature toggle while the anomaly tab is active) still render with a
// tooltip explaining why, rather than disappearing — the underlying data
// model draws one backend-driven field at a time (whichever variable tab is
// active), so a toggle for a field that isn't the active tab has nothing to
// show or hide yet; this is stated honestly instead of faking independent
// dual-layer visibility the backend doesn't support.
// -----------------------------------------------------------------------------

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Layers, RotateCcw } from "lucide-react";
import { Tooltip } from "@/components/ui";
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
 * @param {MapControlLayer[]} props.layers - primary scientific-layer toggles
 * @param {MapControlLayer[]} [props.baseLayers] - secondary base-map toggles
 * @param {() => void} props.onReset
 * @param {string} [props.resetLabel="Reset / fit to domain"]
 * @param {string} [props.className]
 */
export default function MapControls({
  layers = [],
  baseLayers = [],
  onReset,
  resetLabel = "Reset / fit to domain",
  className,
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

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
      <Tooltip content="Map layer controls">
        <span className="inline-flex">
          <LayerToggle
            icon={Layers}
            label="Map layer controls"
            active={open}
            onToggle={() => setOpen((prev) => !prev)}
          />
        </span>
      </Tooltip>

      <AnimatePresence>
        {open && (
          <motion.div
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={scaleIn}
            role="menu"
            aria-label="Map layer controls"
            className={cn(
              "absolute left-0 top-full z-raised mt-2 flex w-60 flex-col gap-2 rounded-md border border-border-strong",
              "bg-surface-overlay p-2.5 shadow-raised sm:w-64"
            )}
          >
            <div>
              <p className="px-1.5 pb-1 text-caption uppercase tracking-widest text-text-muted">
                Scientific layers
              </p>
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
            </div>

            {baseLayers.length > 0 && (
              <div className="border-t border-border-subtle pt-1.5">
                <p className="px-1.5 pb-1 text-caption uppercase tracking-widest text-text-muted">Base map</p>
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

            <button
              type="button"
              onClick={() => {
                onReset?.();
                setOpen(false);
              }}
              className={cn(
                "flex w-full items-center gap-2 rounded-sm border-t border-border-subtle px-1.5 pt-2 text-small",
                "text-text-secondary neer-transition hover:text-accent-300",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-overlay"
              )}
            >
              <RotateCcw size={14} strokeWidth={1.75} aria-hidden="true" />
              {resetLabel}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}