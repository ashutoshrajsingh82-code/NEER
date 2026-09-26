// -----------------------------------------------------------------------------
// NEER Design System — StatusIndicator
//
// Small dot + label for live system/instrument states (model online, a
// reading still processing, a sensor offline, etc). Purely presentational —
// callers decide what maps to which status.
//
// QA: this file previously contained stray barrel-export text (a leftover
// copy of components/ui/index.js) instead of the actual component, which
// silently broke every consumer (MetricCard's `status` prop, TopNavigation's
// live indicator, the design-system preview). Restored here.
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";
import { pulseGlow } from "@/lib/motion";
import { motion } from "framer-motion";

const STATUS_CONFIG = {
  online: { dot: "bg-success", label: "Online", pulse: true },
  processing: { dot: "bg-accent-400", label: "Processing", pulse: true },
  warning: { dot: "bg-warning", label: "Warning", pulse: false },
  error: { dot: "bg-error", label: "Error", pulse: false },
  offline: { dot: "bg-text-disabled", label: "Offline", pulse: false },
  unknown: { dot: "bg-text-muted", label: "Unknown", pulse: false },
};

const SIZE_CLASSES = {
  sm: { dot: "h-1.5 w-1.5", text: "text-caption" },
  md: { dot: "h-2 w-2", text: "text-small" },
};

/**
 * @param {"online"|"processing"|"warning"|"error"|"offline"|"unknown"} status
 * @param {string} label - overrides the default label text for this status
 * @param {boolean} showLabel - whether to render the text label alongside the dot
 * @param {"sm"|"md"} size
 */
export default function StatusIndicator({
  status = "unknown",
  label,
  showLabel = true,
  size = "md",
  className,
}) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.unknown;
  const sizeClasses = SIZE_CLASSES[size] ?? SIZE_CLASSES.md;
  const displayLabel = label ?? config.label;

  return (
    <span className={cn("inline-flex items-center gap-2", className)} role="status">
      <span className={cn("relative inline-flex shrink-0 items-center justify-center", sizeClasses.dot)}>
        {config.pulse && (
          <motion.span
            className={cn("absolute inset-0 rounded-full", config.dot)}
            variants={pulseGlow}
            animate="animate"
            aria-hidden="true"
          />
        )}
        <span className={cn("relative rounded-full", sizeClasses.dot, config.dot)} aria-hidden="true" />
      </span>
      {showLabel && (
        <span className={cn("font-medium text-text-secondary", sizeClasses.text)}>{displayLabel}</span>
      )}
      {!showLabel && <span className="sr-only">{displayLabel}</span>}
    </span>
  );
}