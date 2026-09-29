"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — DashboardKPIs  (Phase 34E)
//
// The dashboard route's KPI drawer content — the last stop in the phase's
// synchronization chain (Date selection -> Map context -> Point selection ->
// Inspection panel -> KPI information). Reads the same PointInspectionContext
// that OceanMapSection and PointInspection read (selectedPoint/date/depth,
// plus the shared per-selection `reconstruct()` result — see that file's
// Phase 34E header comment for why the fetch itself lives there and not
// here), so every panel on the route always agrees about what's currently
// selected and in view.
//
// Real vs. placeholder values — per the phase spec ("do not invent
// scientific values; use API data when available, realistic placeholder
// values only where the backend endpoint is not yet connected"), and
// matching the "Demo"-labeling convention PointInspection.js already
// established in Phase 34D:
//   - Reconstructed temperature / Anomaly — the shared `reconstruct()`
//     result (frontend/lib/api.js, Phase 33B). Real.
//   - RMSE / MAE — `metrics()` (Phase 33C), scored by the backend against
//     the loaded checkpoint's real held-out targets. Real. This is the only
//     place `metrics()` is called from the dashboard: RMSE/MAE don't depend
//     on point selection, so they're fetched once here rather than pushed
//     into PointInspectionContext (which stays scoped to selection + the
//     per-point fetch, per its own header comment).
//   - Model Confidence — no backend field yet (see PointInspection.js's own
//     header comment on this). A labeled Demo value, the *same* constant
//     PointInspection.js uses, so the two panels never show two different
//     "confidences" for one selection.
//   - Data Coverage — `/data/quality` (`dataQuality()` in frontend/lib/
//     api.js) is not one of this phase's approved endpoints (health,
//     modelInfo, dates, reconstruct, reconstructGrid, metrics). A labeled
//     Demo value, not derived from any real coverage figure.
//   - Selected Depth / Selected Date — read straight from
//     PointInspectionContext. Real (they're exactly what the user picked),
//     never fetched.
//
// States (loading / success / empty / error), per section:
//   - Point-derived KPIs: "empty" when no point is selected; loading/error/
//     success mirror pointInspection's own flags (same source PointInspection
//     renders from, so the two panels never disagree about which state
//     they're in).
//   - Model-metrics KPIs: loading/error/success from this component's own
//     `metrics()` request; no "empty" state (this endpoint isn't tied to
//     selection — it's always either loading, succeeded, or failed).
//   - Context KPIs (depth/date): always populated; there is no server round
//     trip to be idle/loading/erroring on.
// -----------------------------------------------------------------------------

import { CalendarDays, Database, Gauge, MapPin, MoveVertical, Thermometer, TrendingUp } from "lucide-react";
import { KPIDrawer } from "@/components/shell";
import { ErrorState, LoadingSkeleton, MetricCard, StatusIndicator } from "@/components/ui";
import { metrics } from "@/lib/api";
import { useApiRequest } from "@/lib/useApiRequest";
import { formatDate, formatSigned } from "@/lib/format";
import { formatDepth, formatLat, formatLon } from "@/lib/oceanDomain";
import { describePointStatus, isQueryablePoint } from "@/lib/pointClassification";
import { usePointInspection } from "../_context/PointInspectionContext";

// Demo-only stand-ins — see the header comment above. Not derived from the
// selected point or any real coverage figure, on purpose, so they never look
// computed.
const DEMO_CONFIDENCE_PERCENT = 87; // matches components/inspection/PointInspection.js
const DEMO_COVERAGE_PERCENT = 94.2;

/**
 * One point-derived KPI card, following pointInspection's own state flags.
 * `pointStatus` short-circuits all of that when the selection is land/
 * outside-domain: the context never fetches for one (see
 * PointInspectionContext.js), so pointInspection stays idle forever and,
 * without this check, every card here would fall into the isLoading/isIdle
 * branch below and skeleton-spin indefinitely instead of explaining why
 * there's nothing to show.
 */
function PointMetricCard({ title, icon, unit, pointInspection, pointStatus, extract, demo }) {
  if (pointStatus && !isQueryablePoint(pointStatus)) {
    return (
      <MetricCard
        title={title}
        value="--"
        unit={unit}
        icon={icon}
        description={describePointStatus(pointStatus)}
      />
    );
  }
  if (pointInspection.isLoading || pointInspection.isIdle) {
    return <LoadingSkeleton variant="metric" label={`Loading ${title.toLowerCase()}`} />;
  }
  if (pointInspection.isError) {
    return <MetricCard title={title} value="--" unit={unit} icon={icon} description="Inspection failed" />;
  }
  if (pointInspection.isSuccess && pointInspection.data) {
    return (
      <MetricCard
        title={title}
        value={demo ? demo.value : extract(pointInspection.data)}
        unit={unit}
        icon={icon}
        description={demo?.description}
      />
    );
  }
  return null;
}

export default function DashboardKPIs() {
  const { selectedPoint, date, depth, pointInspection, pointStatus } = usePointInspection();

  const {
    data: metricsData,
    isLoading: metricsLoading,
    isIdle: metricsIdle,
    isError: metricsIsError,
    error: metricsError,
    run: runMetrics,
  } = useApiRequest(metrics, { immediate: true });

  const metricsPending = metricsLoading || metricsIdle;
  const overall = metricsData?.metrics?.overall;

  const overallStatus =
    metricsIsError || pointInspection.isError
      ? { status: "error", label: "Metrics unavailable" }
      : metricsPending || pointInspection.isLoading
        ? { status: "processing", label: "Loading" }
        : { status: "online", label: "Pipeline nominal" };

  // Compact collapsed-row summary — status + the two context values that are
  // always known, so the row is meaningful even before the drawer is opened.
  const summary = (
    <>
      <StatusIndicator status={overallStatus.status} label={overallStatus.label} size="sm" />
      <span className="flex shrink-0 items-center gap-4">
        <span className="h-4 w-px bg-border" aria-hidden="true" />
        <span className="flex items-baseline gap-1.5 whitespace-nowrap">
          <span className="text-caption text-text-muted">Depth</span>
          <span className="text-small font-mono text-text-primary">{formatDepth(depth)}</span>
        </span>
      </span>
      <span className="flex shrink-0 items-center gap-4">
        <span className="h-4 w-px bg-border" aria-hidden="true" />
        <span className="flex items-baseline gap-1.5 whitespace-nowrap">
          <span className="text-caption text-text-muted">Date</span>
          <span className="text-small font-mono text-text-primary">{formatDate(date)}</span>
        </span>
      </span>
    </>
  );

  return (
    <KPIDrawer title="Mission KPIs" summary={summary} status={overallStatus}>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* --- Point-derived KPIs: empty / loading / error / success ------- */}
        {!selectedPoint ? (
          <>
            <MetricCard title="Reconstructed Temp." value="--" unit="°C" icon={Thermometer} description="No point selected" />
            <MetricCard title="Anomaly" value="--" unit="°C" icon={TrendingUp} description="No point selected" />
            <MetricCard title="Model Confidence" value="--" unit="%" icon={Gauge} description="No point selected" />
          </>
        ) : (
          <>
            <PointMetricCard
              title="MODEL OUTPUT"
              icon={Thermometer}
              unit="°C"
              pointInspection={pointInspection}
              pointStatus={pointStatus}
              extract={(data) => (typeof data.temperature === "number" && Number.isFinite(data.temperature) ? data.temperature.toFixed(2) : "N/A")}
            />
            <PointMetricCard
              title="ANOMALY"
              icon={TrendingUp}
              unit="°C"
              pointInspection={pointInspection}
              pointStatus={pointStatus}
              extract={(data) => typeof data.anomaly === "number" && Number.isFinite(data.anomaly) ? formatSigned(data.anomaly) : "N/A"}
            />
            <PointMetricCard
              title="Model Confidence"
              icon={Gauge}
              unit="%"
              pointInspection={pointInspection}
              pointStatus={pointStatus}
              demo={{ value: DEMO_CONFIDENCE_PERCENT, description: "Demo value — not yet computed by the model" }}
            />
          </>
        )}

        {/* --- Model-metrics KPIs: loading / error / success --------------- */}
        {metricsPending ? (
          <>
            <LoadingSkeleton variant="metric" label="Loading RMSE" />
            <LoadingSkeleton variant="metric" label="Loading MAE" />
          </>
        ) : metricsIsError ? (
          <div className="sm:col-span-2">
            <ErrorState
              title="Model metrics unavailable"
              message={metricsError?.message || "RMSE/MAE couldn't be loaded from the evaluation endpoint."}
              onRetry={() => runMetrics().catch(() => {})}
              details={metricsError ? `${metricsError.code}${metricsError.status ? ` · HTTP ${metricsError.status}` : ""}` : undefined}
            />
          </div>
        ) : (
          <>
            <MetricCard
              title="RMSE"
              value={typeof overall?.rmse === "number" ? overall.rmse.toFixed(3) : "--"}
              unit="°C"
              icon={Gauge}
              description={metricsData?.split ? `${metricsData.split} split` : undefined}
            />
            <MetricCard
              title="MAE"
              value={typeof overall?.mae === "number" ? overall.mae.toFixed(3) : "--"}
              unit="°C"
              icon={Gauge}
              description={metricsData?.split ? `${metricsData.split} split` : undefined}
            />
          </>
        )}

        {/* --- Demo placeholder: no approved endpoint for this yet --------- */}
        <MetricCard
          title="Data Coverage"
          value={DEMO_COVERAGE_PERCENT}
          unit="%"
          icon={Database}
          description="Demo value — not yet wired to a backend endpoint"
        />

        {/* --- Context KPIs: always known, no request in flight ------------ */}
        <MetricCard title="Selected Depth" value={formatDepth(depth)} icon={MoveVertical} />
        <MetricCard title="Selected Date" value={formatDate(date)} icon={CalendarDays} />
        {selectedPoint && (
          <MetricCard
            title="Selected Point"
            value={`${formatLat(selectedPoint.lat)}, ${formatLon(selectedPoint.lon)}`}
            icon={MapPin}
          />
        )}
      </div>
    </KPIDrawer>
  );
}
