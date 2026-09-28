"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — DataContextSection  (Phase 34E, 36A backend-driven,
// Phase 36B real date/depth controls)
//
// The dashboard's date/depth control panel. Same Server/Client boundary
// wrapper role as OceanMapSection.js: dashboard/page.js is a Server Component,
// so the context hooks live here.
//
// Single source of truth: every control reads and writes DateDepthContext
// (GET /dates, GET /model/info — see lib/dateDepthModel.js). Nothing here keeps
// its own copy of the date or depth, so the calendar, previous/next buttons,
// depth slider, depth chips, OceanMap (via OceanMapSection),
// PointInspection and the KPI drawer can never disagree. A change made here
// reaches OceanMap as new `date`/`depth` props, whose useOceanMapLayer then
// requests the matching field through lib/api.js `reconstructGrid`.
// -----------------------------------------------------------------------------

import { CalendarDays, Database, MoveVertical } from "lucide-react";
import { Badge, Panel } from "@/components/ui";
import { cn } from "@/lib/cn";
import { formatDate } from "@/lib/format";
import { formatDepthMetres } from "@/lib/dateDepthControls";
import DateControls from "./DateControls";
import DepthControls from "./DepthControls";
import { usePointInspection } from "../_context/PointInspectionContext";
import { useDateDepthContext } from "../_context/DateDepthContext";
import { useMapFieldStatus } from "./MapFieldStatusContext";

const MODE_CONFIG = {
  reconstructed: { label: "Reconstructed", variant: "accent" },
  observed: { label: "Observed", variant: "info" },
  blended: { label: "Blended", variant: "warning" },
};

const DATES_MESSAGES = {
  network: "Can't reach the NEER backend.",
  timeout: "The backend took too long to respond.",
  config: "Backend URL isn't configured.",
  invalid_response: "Backend sent an invalid date list.",
  api: "Backend couldn't provide dates.",
};

export default function DataContextSection({ className }) {
  const { dataMode } = usePointInspection();
  const {
    availableDates,
    selectedDate,
    selectDate,
    stepDate,
    canStepPrev,
    canStepNext,
    datesStatus,
    datesError,
    datesErrorCategory,
    retryDates,
    availableDepths,
    selectedDepth,
    selectDepth,
    depthsStatus,
    depthsSource,
    depthsValidation,
  } = useDateDepthContext();
  const { isUpdating } = useMapFieldStatus();

  const modeConfig = MODE_CONFIG[dataMode] ?? MODE_CONFIG.reconstructed;

  const datesMessage =
    datesStatus === "empty"
      ? "Backend reports no available dates."
      : datesStatus === "error"
        ? datesError?.message || DATES_MESSAGES[datesErrorCategory] || "Dates unavailable."
        : undefined;

  // Never present a fallback or mismatching list as if it were the backend's
  // confirmed configuration.
  const depthsMessage =
    depthsSource === "expected_fallback"
      ? "Using expected NEER depth levels — model info unavailable."
      : depthsValidation.matchesExpected === false
        ? "Backend depth levels differ from the expected NEER configuration."
        : undefined;

  return (
    <Panel emphasis="base" className={className} bodyClassName="flex flex-col gap-5 lg:flex-row lg:items-start lg:gap-0">
      {/* Data mode + the selection at a glance */}
      <div className="flex min-w-0 flex-col gap-3 lg:w-56 lg:shrink-0 lg:pr-6">
        <div className="flex items-center gap-3">
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

        <dl className="flex flex-col gap-1 text-caption" aria-label="Current selection">
          <div className="flex items-center gap-2">
            <CalendarDays size={12} strokeWidth={1.75} className="text-text-muted" aria-hidden="true" />
            <dt className="sr-only">Selected date</dt>
            <dd className="font-mono text-text-secondary">{selectedDate ? formatDate(selectedDate) : "--"}</dd>
          </div>
          <div className="flex items-center gap-2">
            <MoveVertical size={12} strokeWidth={1.75} className="text-text-muted" aria-hidden="true" />
            <dt className="sr-only">Selected depth</dt>
            <dd className="font-mono text-text-secondary">{formatDepthMetres(selectedDepth)}</dd>
          </div>
        </dl>
      </div>

      <DateControls
        className={cn("lg:shrink-0 lg:border-l lg:border-border-subtle lg:px-6")}
        availableDates={availableDates}
        date={selectedDate}
        status={datesStatus}
        canStepPrev={canStepPrev}
        canStepNext={canStepNext}
        onStepDate={stepDate}
        onSelectDate={selectDate}
        busy={isUpdating}
        message={datesMessage}
        onRetry={retryDates}
      />

      <DepthControls
        className="lg:min-w-0 lg:flex-1 lg:border-l lg:border-border-subtle lg:pl-6"
        depths={availableDepths}
        depth={selectedDepth}
        status={depthsStatus}
        onSelectDepth={selectDepth}
        busy={isUpdating}
        message={depthsMessage}
      />
    </Panel>
  );
}
