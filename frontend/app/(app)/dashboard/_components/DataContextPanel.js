"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — DataContextPanel  (Phase 34B)
//
// Reusable field group for the three inputs that determine what slice of the
// reconstructed field the rest of the dashboard is showing: data mode, the
// selected date, and the selected depth level. Controlled/uncontrolled like
// the design system's other stateful components (Tabs, KPIDrawer) — pass
// `date`/`depth` plus their `onChange` handlers once a later phase wires this
// to the real map/timeline, or omit them and the panel manages sensible
// local state on its own.
//
// Phase 34B scope: display + local, self-contained navigation only — no
// connection yet to any reconstructed field, map, or point-inspection logic.
// -----------------------------------------------------------------------------

import { useState } from "react";
import { CalendarDays, ChevronLeft, ChevronRight, Database, MoveVertical } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button } from "@/components/ui";

const MODE_CONFIG = {
  reconstructed: { label: "Reconstructed", variant: "accent" },
  observed: { label: "Observed", variant: "info" },
  blended: { label: "Blended", variant: "warning" },
};

const DEFAULT_DEPTH_LEVELS = [
  { label: "Surface", value: 0 },
  { label: "50 m", value: 50 },
  { label: "200 m", value: 200 },
  { label: "500 m", value: 500 },
];

const DATE_FORMAT = { day: "2-digit", month: "short", year: "numeric" };

/**
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 * @param {Date|string} date - controlled selected date; falls back to internal state
 * @param {(date: Date) => void} onDateChange
 * @param {{label: string, value: number}[]} depthLevels
 * @param {number} depth - controlled selected depth value; falls back to internal state
 * @param {(value: number) => void} onDepthChange
 */
export default function DataContextPanel({
  dataMode = "reconstructed",
  date,
  onDateChange,
  depthLevels = DEFAULT_DEPTH_LEVELS,
  depth,
  onDepthChange,
  className,
}) {
  const [internalDate, setInternalDate] = useState(() => (date ? new Date(date) : new Date(2024, 2, 18)));
  const [internalDepth, setInternalDepth] = useState(depth ?? depthLevels[0]?.value ?? 0);

  const isDateControlled = date !== undefined;
  const activeDate = isDateControlled ? new Date(date) : internalDate;
  const isDepthControlled = depth !== undefined;
  const activeDepth = isDepthControlled ? depth : internalDepth;
  const modeConfig = MODE_CONFIG[dataMode] ?? MODE_CONFIG.reconstructed;

  function shiftDate(days) {
    const next = new Date(activeDate);
    next.setDate(next.getDate() + days);
    if (!isDateControlled) setInternalDate(next);
    onDateChange?.(next);
  }

  function selectDepth(value) {
    if (!isDepthControlled) setInternalDepth(value);
    onDepthChange?.(value);
  }

  return (
    <div className={cn("flex flex-col gap-4 sm:flex-row sm:flex-wrap sm:items-center sm:gap-6", className)}>
      {/* Data mode */}
      <div className="flex min-w-0 items-center gap-3">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface-raised text-accent-400">
          <Database size={14} strokeWidth={1.75} aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-widest text-text-muted">Data Mode</p>
          <Badge variant={modeConfig.variant} size="sm">
            {modeConfig.label}
          </Badge>
        </div>
      </div>

      {/* Selected date — a compact date-selection control */}
      <div className="flex min-w-0 items-center gap-3 sm:border-l sm:border-border-subtle sm:pl-6">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface-raised text-accent-400">
          <CalendarDays size={14} strokeWidth={1.75} aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-widest text-text-muted">Selected Date</p>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="sm"
              iconOnly
              icon={ChevronLeft}
              aria-label="Previous day"
              onClick={() => shiftDate(-1)}
            />
            <span className="whitespace-nowrap font-mono text-small text-text-primary">
              {activeDate.toLocaleDateString("en-GB", DATE_FORMAT)}
            </span>
            <Button
              variant="ghost"
              size="sm"
              iconOnly
              icon={ChevronRight}
              aria-label="Next day"
              onClick={() => shiftDate(1)}
            />
          </div>
        </div>
      </div>

      {/* Selected depth — current model/view context */}
      <div className="flex min-w-0 items-center gap-3 sm:border-l sm:border-border-subtle sm:pl-6">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface-raised text-accent-400">
          <MoveVertical size={14} strokeWidth={1.75} aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-widest text-text-muted">Selected Depth</p>
          <div className="mt-1 flex flex-wrap items-center gap-1">
            {depthLevels.map((level) => (
              <button
                key={level.value}
                type="button"
                onClick={() => selectDepth(level.value)}
                aria-pressed={activeDepth === level.value}
                className={cn(
                  "rounded-full border px-2.5 py-0.5 text-caption font-medium neer-transition",
                  activeDepth === level.value
                    ? "border-border-accent bg-accent-900/40 text-accent-300"
                    : "border-border text-text-muted hover:border-border-strong hover:text-text-secondary"
                )}
              >
                {level.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}