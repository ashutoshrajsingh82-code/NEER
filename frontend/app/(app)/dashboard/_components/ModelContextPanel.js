// -----------------------------------------------------------------------------
// NEER Dashboard — ModelContextPanel  (Phase 34B)
//
// Reusable field for the active model version. Deliberately understated —
// muted tones, no accent glow, smaller icon chip — so it stays legible
// without competing with MissionHeader's identity or SystemStatusPanel's
// operational status for attention, per the "model version should be clearly
// visible but not visually overpower the mission identity" requirement.
// -----------------------------------------------------------------------------

import { Cpu } from "lucide-react";
import { cn } from "@/lib/cn";
import { Tooltip } from "@/components/ui";

/**
 * @param {string} version - e.g. "v1.3.0"
 * @param {string} name - full model name, shown in the tooltip
 * @param {string} updatedAt - optional "last updated" note, shown in the tooltip
 */
export default function ModelContextPanel({
  version = "v1.3.0",
  name = "NEER Reconstruction Model",
  updatedAt,
  className,
}) {
  return (
    <Tooltip content={updatedAt ? `${name} · updated ${updatedAt}` : name}>
      <div className={cn("flex min-w-0 items-center gap-3", className)}>
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border-subtle bg-surface-sunken text-text-muted">
          <Cpu size={14} strokeWidth={1.75} aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-widest text-text-muted">Model</p>
          <p className="truncate font-mono text-small text-text-secondary">{version}</p>
        </div>
      </div>
    </Tooltip>
  );
}