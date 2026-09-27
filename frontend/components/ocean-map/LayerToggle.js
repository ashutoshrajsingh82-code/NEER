"use client";

// -----------------------------------------------------------------------------
// NEER OceanMap — LayerToggle  (Phase 35D1)
//
// One reusable on/off control for a single map layer (temperature field,
// anomaly field, scientific grid, graticule, coastline, ...). Purely
// presentational/interactive, same contract as the design system's Button:
// the caller owns the boolean state and passes `active`/`onToggle`; this
// component only renders it consistently and accessibly.
//
// Two layouts share one implementation so MapControls can use the same
// primitive in a compact toolbar and inside its layer-list popover without
// two parallel toggle implementations drifting apart:
//   - "icon"  — a square icon-only button (toolbar / map-corner controls),
//               wrapped in a Tooltip so its accessible label is still
//               readable on hover/focus without adding visible text.
//   - "row"   — a full-width labelled row with a small switch, for a
//               popover/menu list where the label is shown directly.
//
// Accessibility: both layouts expose `active` as a boolean ARIA state
// (aria-pressed for a toggle button, role="switch"/aria-checked for a
// switch-styled row) rather than only conveying it through color, and both
// pick up the app-wide focus-visible ring so keyboard users can see focus.
// -----------------------------------------------------------------------------

import { forwardRef } from "react";
import { motion } from "framer-motion";
import { Tooltip } from "@/components/ui";
import { cn } from "@/lib/cn";
import { ICON_SIZES } from "@/lib/icons";
import { DURATION, EASE_OUT, tapPress } from "@/lib/motion";

/**
 * @param {object} props
 * @param {React.ComponentType} props.icon - a lucide-react icon component
 * @param {string} props.label - accessible name; also the visible text in "row" layout
 * @param {string} [props.description] - tooltip text; defaults to `label` in "icon" layout
 * @param {boolean} [props.active=false]
 * @param {boolean} [props.disabled=false]
 * @param {string} [props.disabledReason] - tooltip/description shown instead of `description` while disabled
 * @param {(next: boolean) => void} props.onToggle - called with the next boolean state
 * @param {"icon"|"row"} [props.layout="icon"]
 * @param {string} [props.className]
 */
const LayerToggle = forwardRef(function LayerToggle(
  {
    icon: Icon,
    label,
    description,
    active = false,
    disabled = false,
    disabledReason,
    onToggle,
    layout = "icon",
    className,
  },
  ref
) {
  const tooltipText = disabled ? disabledReason ?? description ?? label : description ?? label;

  if (layout === "row") {
    return (
      <Tooltip content={disabled ? disabledReason : undefined} position="top">
        <button
          ref={ref}
          type="button"
          role="switch"
          aria-checked={active}
          aria-label={label}
          disabled={disabled}
          onClick={() => onToggle?.(!active)}
          className={cn(
            "flex w-full items-center justify-between gap-3 rounded-sm px-2 py-1.5 text-left text-small neer-transition",
            "text-text-secondary hover:bg-surface-raised hover:text-text-primary",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-overlay",
            disabled && "cursor-not-allowed opacity-40 hover:bg-transparent hover:text-text-secondary",
            className
          )}
        >
          <span className="flex min-w-0 items-center gap-2">
            {Icon && (
              <Icon
                size={ICON_SIZES.sm}
                strokeWidth={1.75}
                className={cn("shrink-0", active ? "text-accent-400" : "text-text-muted")}
                aria-hidden="true"
              />
            )}
            <span className="truncate">{label}</span>
          </span>
          <span
            aria-hidden="true"
            className={cn(
              "relative inline-flex h-4 w-7 shrink-0 items-center rounded-full border neer-transition",
              active ? "border-accent-400/70 bg-accent-500/25" : "border-border-strong bg-surface-sunken"
            )}
          >
            <span
              className={cn(
                "inline-block h-2.5 w-2.5 translate-x-0.5 transform rounded-full neer-transition",
                active ? "translate-x-3.5 bg-accent-400 shadow-glow-sm" : "bg-text-disabled"
              )}
            />
          </span>
        </button>
      </Tooltip>
    );
  }

  const trigger = (
    <motion.button
      ref={ref}
      type="button"
      aria-pressed={active}
      aria-label={label}
      disabled={disabled}
      whileHover={!disabled ? { y: -1 } : undefined}
      whileTap={!disabled ? tapPress : undefined}
      transition={{ duration: DURATION.fast, ease: EASE_OUT }}
      onClick={() => onToggle?.(!active)}
      className={cn(
        "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md border neer-transition",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
        "focus-visible:ring-offset-2 focus-visible:ring-offset-bg-base",
        active
          ? "border-accent-400/60 bg-accent-500/15 text-accent-300 shadow-glow-sm"
          : "border-transparent bg-transparent text-text-secondary hover:border-border-strong hover:bg-surface-raised hover:text-text-primary",
        disabled && "pointer-events-none opacity-35",
        className
      )}
    >
      {Icon && <Icon size={ICON_SIZES.sm} strokeWidth={1.75} aria-hidden="true" />}
    </motion.button>
  );

  return (
    <Tooltip content={tooltipText}>
      <span className="inline-flex">{trigger}</span>
    </Tooltip>
  );
});

export default LayerToggle;