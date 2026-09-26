"use client";

// -----------------------------------------------------------------------------
// NEER Design System — StatusIndicator
//
// A small dot + optional label used to show live system/instrument state
// (online, processing, warning, error, offline, unknown). Purely
// presentational — the caller decides what "status" means for their data.
// -----------------------------------------------------------------------------

import { motion } from "framer-motion";
import { cn } from "@/lib/cn";
import { pulseGlow } from "@/lib/motion";

const STATUS_CONFIG = {
  online: {
    dot: "bg-success",
    ring: "shadow-[0_0_8px_rgba(52,211,153,0.6)]",
    label: "Online",
  },
  processing: {
    dot: "bg-accent-400",
    ring: "shadow-[0_0_8px_rgba(45,212,191,0.6)]",
    label: "Processing",
    pulse: true,
  },
  warning: {
    dot: "bg-warning",
    ring: "shadow-[0_0_8px_rgba(251,191,36,0.6)]",
    label: "Warning",
  },
  error: {
    dot: "bg-error",
    ring: "shadow-[0_0_8px_rgba(248,113,113,0.6)]",
    label: "Error",
  },
  offline: {
    dot: "bg-text-disabled",
    ring: "",
    label: "Offline",
  },
  unknown: {
    dot: "bg-text-muted",
    ring: "",
    label: "Unknown",
  },
};

const DOT_SIZE_CLASSES = {
  sm: "h-1.5 w-1.5",
  md: "h-2 w-2",
  lg: "h-3 w-3",
};

/**
 * @param {"online"|"processing"|"warning"|"error"|"offline"|"unknown"} status
 * @param {string} label - overrides the default label text
 * @param {boolean} showLabel - if false, label is still available to screen readers
 * @param {"sm"|"md"|"lg"} size
 */
export default function StatusIndicator({
  status = "unknown",
  label,
  showLabel = true,
  size = "md",
  className,
}) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.unknown;
  const resolvedLabel = label ?? config.label;

  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <motion.span
        className={cn("inline-block shrink-0 rounded-full", DOT_SIZE_CLASSES[size], config.dot, config.ring)}
        {...(config.pulse ? pulseGlow : {})}
        aria-hidden="true"
      />
      {showLabel ? (
        <span className="text-small text-text-secondary">{resolvedLabel}</span>
      ) : (
        <span className="sr-only">{resolvedLabel}</span>
      )}
    </span>
  );
}