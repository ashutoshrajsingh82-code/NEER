"use client";

// -----------------------------------------------------------------------------
// NEER — PointInspection  (Phase 34D)
//
// The point-inspection workflow: given a selected grid cell (lat/lon) plus
// the date/depth/data-mode currently in view, fetches and displays what the
// reconstruction model says about that cell — reconstructed temperature,
// anomaly, model confidence, the input variables that fed the reconstruction,
// and the data's availability/status. Designed to render inside the shared
// shell InspectionPanel (components/shell/InspectionPanel.js, Phase 32D) —
// that component owns the panel chrome (open/close, header, scroll); this
// one owns only the content, and is plain props-in/callbacks-out so any
// future workflow (Reconstruction, Explainability, Evaluation — Requirement
// 11) can render it inside its own InspectionPanel with its own state, the
// same way OceanMap.js is reusable outside the dashboard route.
//
// Real vs. placeholder data (per the phase spec: "use placeholder/demo
// values where backend integration is not yet available" — same rule
// OceanMap.js follows for its unbuilt layers): temperature and anomaly come
// from the real `reconstruct` endpoint (frontend/lib/api.js, Phase 33B) —
// this is the first real caller of that module (see its own header comment:
// "Still not called from any page/component yet ... that starts once real
// pages are built"). Model confidence and the input-variable list have no
// backend field yet (see reconstruct()'s JSDoc return shape) — those are
// demo values, and every one of them is labeled "Demo" in the UI rather than
// presented as if the model computed them. Data-mode/cache/notes in the
// "Data availability" section are real, straight off the API response.
//
// All fetching lives here (Requirement 10 — "do not duplicate API-fetching
// logic inside unrelated components"): OceanMap only reports a selected
// point via onSelectPoint, it never calls the API itself; this is the only
// place `reconstruct()` is called from the dashboard.
// -----------------------------------------------------------------------------

import { useEffect } from "react";
import {
  CalendarDays,
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
import { reconstruct } from "@/lib/api";
import { useApiRequest } from "@/lib/useApiRequest";
import { formatDepth, formatLat, formatLon } from "@/lib/oceanDomain";

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

// Demo-only stand-ins — see the header comment above. Not derived from the
// selected point in any way, on purpose, so they never look computed.
const DEMO_CONFIDENCE_PERCENT = 87;
const DEMO_INPUT_VARIABLES = [
  { label: "Nearby Argo profiles", value: "3 within 50 km" },
  { label: "Satellite SST", value: "Available" },
  { label: "Altimetry (SSH)", value: "Available" },
];

const DATE_FORMAT = { day: "2-digit", month: "short", year: "numeric" };

function formatDate(date) {
  if (!date) return "No date selected";
  const parsed = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(parsed.getTime())) return String(date);
  return parsed.toLocaleDateString("en-GB", DATE_FORMAT);
}

function formatSigned(value, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

/**
 * @param {{lat: number, lon: number}|null|undefined} selectedPoint - the
 *   currently selected grid cell (e.g. from OceanMap's onSelectPoint);
 *   null/undefined renders the empty state (Requirement 6)
 * @param {string|Date} date - date currently in view
 * @param {number} depth - depth level in metres currently in view
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 * @param {() => void} onClose - clears the current selection; the component
 *   always renders a labeled way back to the empty state (Requirement 4)
 */
export default function PointInspection({
  selectedPoint,
  date,
  depth = 0,
  dataMode = "reconstructed",
  onClose,
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

  // `key` forces a fresh mount — and so a fresh, from-idle useApiRequest —
  // every time the selection (or the date/depth it's inspected at) changes,
  // instead of one long-lived request hook whose state an effect re-syncs
  // on every change. Without this, switching straight from a resolved point
  // to a new one would render one frame of the OLD point's temperature/
  // anomaly under the NEW point's coordinates (the effect that starts the
  // new fetch can't run until after that render). Remounting sidesteps the
  // mismatch entirely rather than papering over it with a manual
  // data-matches-selection check.
  return (
    <PointInspectionResult
      key={`${selectedPoint.lat}-${selectedPoint.lon}-${date}-${depth}`}
      selectedPoint={selectedPoint}
      date={date}
      depth={depth}
      dataMode={dataMode}
      onClose={onClose}
      className={className}
    />
  );
}

function PointInspectionResult({ selectedPoint, date, depth, dataMode, onClose, className }) {
  const { data, error, run, isLoading, isIdle, isError, isSuccess } = useApiRequest(reconstruct);

  useEffect(() => {
    run({ lat: selectedPoint.lat, lon: selectedPoint.lon, date, depth }).catch(() => {
      // Swallowed here on purpose — useApiRequest already captured the
      // failure in `error`/`isError` for the error state below to render.
    });
    // Deliberately empty deps: this component is remounted (see the `key`
    // above) whenever selectedPoint/date/depth change, so "run once per
    // mount" is exactly "run once per unique selection".
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
          onRetry={() => run({ lat: selectedPoint.lat, lon: selectedPoint.lon, date, depth }).catch(() => {})}
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