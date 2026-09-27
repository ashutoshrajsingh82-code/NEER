// -----------------------------------------------------------------------------
// NEER Dashboard — SystemStatusPanel  (Phase 34B)
//
// Reusable operational-status field: a single labeled indicator for the
// reconstruction pipeline's current state. Maps NEER's five operational
// states onto the shared StatusIndicator component's dot/label so the visual
// language (color, pulse) stays consistent with the rest of the design
// system, while giving each state the exact wording ops staff expect.
// -----------------------------------------------------------------------------

import { Activity } from "lucide-react";
import { cn } from "@/lib/cn";
import { StatusIndicator } from "@/components/ui";

// NEER-specific operational states -> shared StatusIndicator status + label.
const STATE_CONFIG = {
  operational: { status: "online", label: "Operational" },
  processing: { status: "processing", label: "Processing" },
  warning: { status: "warning", label: "Warning" },
  degraded: { status: "error", label: "Degraded" },
  offline: { status: "offline", label: "Offline" },
};

/**
 * @param {"operational"|"processing"|"warning"|"degraded"|"offline"} state
 * @param {string} detail - optional secondary line, e.g. "All subsystems nominal"
 */
export default function SystemStatusPanel({ state = "operational", detail, className }) {
  const config = STATE_CONFIG[state] ?? STATE_CONFIG.operational;

  return (
    <div className={cn("flex min-w-0 items-center gap-3", className)}>
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface-raised text-accent-400">
        <Activity size={15} strokeWidth={1.75} aria-hidden="true" />
      </span>
      <div className="min-w-0">
        <p className="text-caption uppercase tracking-widest text-text-muted">System Status</p>
        <StatusIndicator status={config.status} label={config.label} size="sm" />
        {detail && <p className="mt-0.5 truncate text-caption text-text-muted">{detail}</p>}
      </div>
    </div>
  );
}