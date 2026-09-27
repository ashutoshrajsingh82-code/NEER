"use client";

// -----------------------------------------------------------------------------
// NEER — PointInspection  (Phase 34D, Phase 34E: fetch moved to context)
//
// The point-inspection workflow: given a selected grid cell (lat/lon), the
// date/depth/data-mode currently in view, and the result of inspecting that
// cell, renders what the reconstruction model says about it — reconstructed
// temperature, anomaly, model confidence, the input variables that fed the
// reconstruction, and the data's availability/status. Designed to render
// inside the shared shell InspectionPanel (components/shell/
// InspectionPanel.js, Phase 32D) — that component owns the panel chrome
// (open/close, header, scroll); this one owns only the content.
//
// Phase 34E change: this component no longer calls `reconstruct()` itself.
// The KPI drawer added this phase (dashboard/_components/DashboardKPIs.js)
// needs the *same* reconstructed temperature/anomaly for the *same*
// selection, and fetching it twice (once here, once there) would violate
// Requirement 10 below just as much as calling it from two unrelated
// components would have in Phase 34D. The fetch now lives in
// PointInspectionContext (dashboard/_context/PointInspectionContext.js —
// see its Phase 34E header comment for how it avoids the stale-data-for-one-
// frame problem this file's old `key`-forced-remount trick used to solve),
// and is passed down as the `reconstruction` prop by whoever wires this
// component up (today, app/(app)/layout.js). This file is still plain
// props-in/callbacks-out and has no import of that context, so it stays
// exactly as reusable outside the dashboard route as it was in Phase 34D —
// any future workflow can still render it with its own fetch state shaped
// however that workflow's own state lives.
//
// Real vs. placeholder data (per the phase spec: "use placeholder/demo
// values where backend integration is not yet available" — same rule
// OceanMap.js follows for its unbuilt layers): temperature and anomaly come
// from the real `reconstruct` endpoint (frontend/lib/api.js, Phase 33B).
// Model confidence and the input-variable list have no backend field yet
// (see `reconstruct()`'s JSDoc return shape) — those are demo values, and
// every one of them is labeled "Demo" in the UI rather than presented as if
// the model computed them. Data-mode/cache/notes in the "Data availability"
// section are real, straight off the API response.
//
// All fetching for the dashboard route lives in PointInspectionContext
// (Requirement 10 — "do not duplicate API-fetching logic inside unrelated
// components"): OceanMap only reports a selected point via onSelectPoint, it
// never calls the API itself; this component only renders a result it's
// handed.
// -----------------------------------------------------------------------------

import {
  CalendarDays,
  Compass,
  Database,
  Gauge,
  MapPin,
  MoveVertical,
  Satellite,
  Thermometer,
  TrendingUp,
  X,
} from "lucide-react";
import { Badge, Button, ErrorState, LoadingSkeleton, MetricCard, Panel, StatusIndicator } from "@/components/ui";
import { cn } from "@/lib/cn";
import { formatDate, formatSigned } from "@/lib/format";
import { formatDepth, formatLat, formatLon } from "@/lib/oceanDomain";
import { POINT_STATUS, describePointStatus, isQueryablePoint } from "@/lib/pointClassification";

const MODE_CONFIG = {
  reconstructed: { label: "Reconstructed", variant: "accent" },
  observed: { label: "Observed", variant: "info" },
  blended: { label: "Blended", variant: "warning" },
};

const EMPTY_PREVIEW_FIELDS = [
  { label: "Reconstructed temperature", unit: "°C" },
  { label: "Anomaly", unit: "°C" },
  { label: "Model confidence", unit: "%" },
];

// Demo-only stand-in — see the header comment above. Not derived from the
// selected point in any way, on purpose, so it never looks computed. Kept as
// the same constant DashboardKPIs.js uses, so the two panels never disagree.
const DEMO_CONFIDENCE_PERCENT = 87;
const DEMO_INPUT_VARIABLES = [
  { label: "Nearby Argo profiles", value: "3 within 50 km" },
  { label: "Satellite SST", value: "Available" },
  { label: "Altimetry (SSH)", value: "Available" },
];

/**
 * @param {{lat: number, lon: number}|null|undefined} selectedPoint - the
 *   currently selected grid cell (e.g. from OceanMap's onSelectPoint);
 *   null/undefined renders the empty state (Requirement 6)
 * @param {string|Date} date - date currently in view
 * @param {number} depth - depth level in metres currently in view
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 * @param {() => void} onClose - clears the current selection; the component
 *   always renders a labeled way back to the empty state (Requirement 4)
 * @param {{
 *   data: object|null,
 *   error: import("@/lib/api").ApiError|null,
 *   isLoading: boolean,
 *   isIdle: boolean,
 *   isSuccess: boolean,
 *   isError: boolean,
 *   retry: () => void,
 * }} reconstruction - the shared `reconstruct()` request state for
 *   `selectedPoint` at `date`/`depth` (Phase 34E — see
 *   PointInspectionContext.js). Ignored when `selectedPoint` is falsy.
 * @param {"ocean"|"land"|"outside_domain"|null} [pointStatus] - the
 *   classification of `selectedPoint` (lib/pointClassification.js, via
 *   PointInspectionContext). A land/outside-domain point never gets a
 *   `reconstruction` fetch (the context skips it — see its own header
 *   comment), so this is what tells this component to render that as its
 *   own explicit state instead of reading `reconstruction`'s perpetually
 *   idle isLoading/isIdle flags as "still loading".
 */
export default function PointInspection({
  selectedPoint,
  date,
  depth = 0,
  dataMode = "reconstructed",
  onClose,
  reconstruction,
  pointStatus,
  className,
}) {
  // --- Empty state (Requirement 6) ----------------------------------------
  if (!selectedPoint) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Panel emphasis="base" icon={MapPin} title="No point selected" bodyClassName="flex flex-col gap-3">
          <p className="text-small text-text-secondary">
            Click a location on the ocean map to inspect its reconstructed field values here.
          </p>
          <dl className="flex flex-col gap-2">
            {EMPTY_PREVIEW_FIELDS.map((field) => (
              <div
                key={field.label}
                className="flex items-center justify-between rounded-md border border-dashed border-border px-3 py-2"
              >
                <dt className="text-caption text-text-muted">{field.label}</dt>
                <dd className="font-mono text-small text-text-disabled">-- {field.unit}</dd>
              </div>
            ))}
          </dl>
        </Panel>
        <Badge variant="neutral" size="sm" className="self-start">
          Grid: 5°N–30°N · 0.25° resolution
        </Badge>
      </div>
    );
  }

  const { data, error, isLoading, isIdle, isError, isSuccess, retry } = reconstruction ?? {};
  const modeConfig = MODE_CONFIG[dataMode] ?? MODE_CONFIG.reconstructed;

  // Coordinates + date/depth context — always shown once a point is
  // selected, in every state below, per Requirement 3 ("display the
  // selected coordinates clearly").
  const coordinateHeader = (
    <div>
      <p className="text-caption uppercase tracking-widest text-text-muted">Selected Grid Cell</p>
      <p className="font-mono text-body font-semibold text-text-primary">
        {formatLat(selectedPoint.lat)}, {formatLon(selectedPoint.lon)}
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-caption text-text-muted">
        <span className="flex items-center gap-1">
          <CalendarDays size={11} strokeWidth={1.75} aria-hidden="true" />
          {formatDate(date)}
        </span>
        <span className="flex items-center gap-1">
          <MoveVertical size={11} strokeWidth={1.75} aria-hidden="true" />
          {formatDepth(depth)}
        </span>
        <Badge variant={modeConfig.variant} size="sm">
          {modeConfig.label}
        </Badge>
      </div>
    </div>
  );

  // Requirement 4 — a labeled, always-visible way to clear the current
  // selection and return to the empty state (distinct from the shared
  // InspectionPanel's own close-the-whole-panel button in its header).
  const clearSelectionButton = (
    <Button variant="secondary" size="sm" icon={X} onClick={onClose}>
      Clear selection
    </Button>
  );

  // --- Land / outside-domain state ----------------------------------------
  // Distinct from loading/error/success: the context deliberately never
  // fetches for a non-queryable point (see PointInspectionContext.js's
  // selectPoint/setDate/setDepth), so `reconstruction` stays idle forever
  // for one — without this check that idle state would fall through to the
  // loading branch below and spin indefinitely instead of explaining why.
  if (pointStatus && !isQueryablePoint(pointStatus)) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        {coordinateHeader}
        <Panel
          emphasis="base"
          icon={Compass}
          title={pointStatus === POINT_STATUS.LAND ? "Land" : "Outside domain"}
          bodyClassName="flex flex-col gap-3"
        >
          <StatusIndicator status="warning" label={describePointStatus(pointStatus)} size="sm" />
          <p className="text-small text-text-secondary">
            This grid cell can&apos;t be reconstructed — pick an ocean point inside the NEER
            domain to inspect it.
          </p>
        </Panel>
        {clearSelectionButton}
      </div>
    );
  }

  // --- Loading state (Requirement 5) --------------------------------------
  if (isLoading || isIdle) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        {coordinateHeader}
        <div className="grid grid-cols-2 gap-3">
          <LoadingSkeleton variant="metric" label="Loading reconstructed temperature" />
          <LoadingSkeleton variant="metric" label="Loading anomaly" />
        </div>
        <LoadingSkeleton variant="panel" label="Loading model context" />
        {clearSelectionButton}
      </div>
    );
  }

  // --- Error state (Requirement 7) ----------------------------------------
  if (isError) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        {coordinateHeader}
        <ErrorState
          title="Inspection failed"
          message={error?.message || "This grid cell's reconstructed values couldn't be loaded."}
          onRetry={retry}
          details={error ? `${error.code}${error.status ? ` · HTTP ${error.status}` : ""}` : undefined}
        />
        {clearSelectionButton}
      </div>
    );
  }

  // --- Success state ---------------------------------------------------
  if (isSuccess && data) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        {coordinateHeader}

        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            title="Reconstructed Temp."
            value={typeof data.temperature === "number" ? data.temperature.toFixed(2) : "--"}
            unit="°C"
            icon={Thermometer}
          />
          <MetricCard title="Anomaly" value={formatSigned(data.anomaly)} unit="°C" icon={TrendingUp} />
        </div>

        <MetricCard
          title="Model Confidence"
          value={DEMO_CONFIDENCE_PERCENT}
          unit="%"
          icon={Gauge}
          description="Demo value — not yet computed by the model"
        />

        <Panel
          emphasis="base"
          icon={Satellite}
          title="Relevant Input Variables"
          headerActions={
            <Badge variant="neutral" size="sm">
              Demo
            </Badge>
          }
          bodyClassName="flex flex-col gap-2"
        >
          <dl className="flex flex-col gap-2">
            {DEMO_INPUT_VARIABLES.map((variable) => (
              <div key={variable.label} className="flex items-center justify-between gap-2">
                <dt className="text-caption text-text-muted">{variable.label}</dt>
                <dd className="text-caption font-mono text-text-secondary">{variable.value}</dd>
              </div>
            ))}
          </dl>
        </Panel>

        <Panel emphasis="base" icon={Database} title="Data Availability" bodyClassName="flex flex-col gap-2.5">
          <StatusIndicator status="online" label={`${modeConfig.label} field available`} size="sm" />
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={data.cache_hit ? "info" : "neutral"} size="sm">
              {data.cache_hit ? "Cached result" : "Computed live"}
            </Badge>
            {typeof data.latency_ms === "number" && (
              <span className="text-caption text-text-muted">{Math.round(data.latency_ms)} ms</span>
            )}
          </div>
          {Array.isArray(data.notes) && data.notes.length > 0 && (
            <ul className="flex flex-col gap-1 border-t border-border-subtle pt-2">
              {data.notes.map((note, index) => (
                <li key={index} className="text-caption text-text-muted">
                  {note}
                </li>
              ))}
            </ul>
          )}
        </Panel>

        {clearSelectionButton}
      </div>
    );
  }

  return null;
}