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
//
// Phase 36A: this panel is now driven by backend state (see
// DataContextSection.js / DateDepthContext.js). It no longer invents a
// default date: `date` is a "YYYY-MM-DD" string or null, and previous/next
// step through the dates the backend actually lists (`onStepDate`) rather
// than adding ±1 calendar day, which would request dates that don't exist.
// Not the final visual controls — just honest wiring: loading / empty /
// error captions, disabled-at-the-ends stepping, and the backend depth list.
// -----------------------------------------------------------------------------

import { useState } from "react";
import { CalendarDays, ChevronLeft, ChevronRight, Database, MoveVertical } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button } from "@/components/ui";
import { formatDate } from "@/lib/format";

const MODE_CONFIG = {
  reconstructed: { label: "Reconstructed", variant: "accent" },
  observed: { label: "Observed", variant: "info" },
  blended: { label: "Blended", variant: "warning" },
};

// No built-in depth list: levels come from the backend (GET /model/info) via
// DataContextSection — an invented default here would be a second, unvalidated
// source of truth for what depths exist.
const DEFAULT_DEPTH_LEVELS = [];

/**
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 * @param {string|null} date - selected "YYYY-MM-DD" date; null when none is available
 * @param {(delta: number) => void} onStepDate - step to the previous (-1) / next (+1) AVAILABLE date
 * @param {boolean} canStepPrev
 * @param {boolean} canStepNext
 * @param {"loading"|"success"|"empty"|"error"} datesStatus
 * @param {string} [datesMessage] - short explanation shown when datesStatus is "empty"/"error"
 * @param {() => void} [onRetryDates]
 * @param {{label: string, value: number}[]} depthLevels - the backend-provided levels
 * @param {number|null} depth - selected depth value
 * @param {(value: number) => void} onDepthChange
 * @param {"loading"|"success"|"error"} depthsStatus
 * @param {string} [depthsMessage] - caption shown when the depth list is a fallback / differs from expected
 */
export default function DataContextPanel({
  dataMode = "reconstructed",
  date,
  onStepDate,
  canStepPrev = false,
  canStepNext = false,
  datesStatus = "success",
  datesMessage,
  onRetryDates,
  depthLevels = DEFAULT_DEPTH_LEVELS,
  depth,
  onDepthChange,
  depthsStatus = "success",
  depthsMessage,
  className,
}) {
  const [internalDepth, setInternalDepth] = useState(depth ?? depthLevels[0]?.value ?? null);

  const isDepthControlled = depth !== undefined;
  const activeDepth = isDepthControlled ? depth : internalDepth;
  const modeConfig = MODE_CONFIG[dataMode] ?? MODE_CONFIG.reconstructed;

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
              aria-label="Previous available date"
              disabled={!canStepPrev}
              onClick={() => onStepDate?.(-1)}
            />
            <span className="whitespace-nowrap font-mono text-small text-text-primary">
              {date
                ? formatDate(date)
                : datesStatus === "loading"
                  ? "Loading dates…"
                  : datesStatus === "empty"
                    ? "No dates available"
                    : "Dates unavailable"}
            </span>
            <Button
              variant="ghost"
              size="sm"
              iconOnly
              icon={ChevronRight}
              aria-label="Next available date"
              disabled={!canStepNext}
              onClick={() => onStepDate?.(1)}
            />
          </div>
          {(datesStatus === "error" || datesStatus === "empty") && (
            <p className="mt-0.5 flex items-center gap-2 text-caption text-text-muted" role="status">
              <span className="truncate">{datesMessage}</span>
              {datesStatus === "error" && onRetryDates && (
                <button type="button" onClick={onRetryDates} className="shrink-0 text-accent-400 underline">
                  Retry
                </button>
              )}
            </p>
          )}
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
            {depthsStatus === "loading" && depthLevels.length === 0 && (
              <span className="text-caption text-text-muted">Loading depth levels…</span>
            )}
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
          {depthsMessage && (
            <p className="mt-0.5 text-caption text-text-muted" role="status">
              {depthsMessage}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}