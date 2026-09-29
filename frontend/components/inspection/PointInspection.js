"use client";

import Link from "next/link";
import { CalendarDays, Compass, Database, MapPin, MoveVertical, Satellite, Thermometer, TrendingUp, X } from "lucide-react";
import { Button, ErrorState, LoadingSkeleton, MetricCard, Panel, StatusIndicator } from "@/components/ui";
import { formatDate, formatSigned } from "@/lib/format";
import { formatDepth, formatLat, formatLon } from "@/lib/oceanDomain";
import { valueOrNA } from "@/lib/profileResponse";
import { POINT_STATUS, describePointStatus, isQueryablePoint } from "@/lib/pointClassification";

export default function PointInspection({ selectedPoint, date, depth, onClose, reconstruction, pointStatus, modelInfo, variableMode = "temperature" }) {
  if (!selectedPoint) {
    return (
      <Panel emphasis="base" icon={MapPin} title="Location inspection">
        <p className="text-small text-text-secondary">Click an ocean location on the map to inspect its available NEER data.</p>
      </Panel>
    );
  }

  const {
    data, error, isLoading, isIdle, isError, isSuccess, retry,
    surfaceTemperature, surfaceIsLoading, surfaceError,
  } = reconstruction ?? {};
  const modelPoint = <div>
    <p className="text-caption uppercase tracking-widest text-text-muted">Model grid point</p>
    <p className="font-mono text-body font-semibold text-text-primary">{formatLat(selectedPoint.lat)}, {formatLon(selectedPoint.lon)}</p>
    {selectedPoint.clicked && (selectedPoint.clicked.lat !== selectedPoint.lat || selectedPoint.clicked.lon !== selectedPoint.lon) && (
      <p className="mt-1 text-caption text-text-muted">Clicked location: {formatLat(selectedPoint.clicked.lat)}, {formatLon(selectedPoint.clicked.lon)}</p>
    )}
    <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-caption text-text-muted">
      <span className="flex items-center gap-1"><CalendarDays size={11} aria-hidden="true" />{formatDate(date)}</span>
      <span className="flex items-center gap-1"><MoveVertical size={11} aria-hidden="true" />{formatDepth(depth)}</span>
    </div>
  </div>;
  const clearButton = <Button variant="secondary" size="sm" icon={X} onClick={onClose}>Clear selection</Button>;

  if (pointStatus && !isQueryablePoint(pointStatus)) {
    return <div className="flex flex-col gap-4">{modelPoint}<Panel emphasis="base" icon={Compass} title={pointStatus === POINT_STATUS.LAND ? "Land" : "Outside domain"}>
      <StatusIndicator status="warning" label={describePointStatus(pointStatus)} size="sm" />
      <p className="mt-2 text-small text-text-secondary">No ocean reconstruction is requested for this location.</p>
    </Panel>{clearButton}</div>;
  }

  if (isLoading || isIdle) {
    return <div className="flex flex-col gap-4">{modelPoint}<p className="text-small text-text-secondary" role="status" aria-live="polite">Loading scientific data…</p><LoadingSkeleton variant="metric" label="Loading reconstruction" />{clearButton}</div>;
  }

  if (isError) {
    return <div className="flex flex-col gap-4">{modelPoint}<ErrorState title="Inspection failed" message={error?.message || "Unable to retrieve inspection data."} onRetry={retry} />{clearButton}</div>;
  }

  if (!isSuccess || !data) return null;
  const modelVersion = modelInfo?.version ?? modelInfo?.model_version ?? modelInfo?.checkpoint?.version ?? null;
  const surfaceValue = surfaceIsLoading ? "Loading" : typeof surfaceTemperature === "number" ? surfaceTemperature.toFixed(2) : "N/A";

  return (
    <div className="flex flex-col gap-4">
      {modelPoint}
      <div className="grid grid-cols-1 gap-3">
        <MetricCard title="MODEL OUTPUT @ surface" value={surfaceValue} unit={typeof surfaceTemperature === "number" ? "°C" : undefined} icon={Thermometer} />
        {surfaceError && <p className="text-caption text-error" role="alert">Surface lookup failed: {surfaceError.message}</p>}
        <MetricCard title={`${variableMode === "anomaly" ? "ANOMALY" : "MODEL OUTPUT"} @ ${formatDepth(depth)}`} value={variableMode === "anomaly" ? (typeof data.anomaly === "number" && Number.isFinite(data.anomaly) ? formatSigned(data.anomaly) : valueOrNA(data.anomaly)) : (typeof data.temperature === "number" && Number.isFinite(data.temperature) ? data.temperature.toFixed(2) : valueOrNA(data.temperature))} unit={typeof (variableMode === "anomaly" ? data.anomaly : data.temperature) === "number" ? "°C" : undefined} icon={Thermometer} />
      </div>

      <Panel emphasis="base" icon={TrendingUp} title="Scientific values" bodyClassName="flex flex-col gap-2">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-accent-300">Active field: {variableMode === "anomaly" ? "ANOMALY" : "MODEL OUTPUT"}</p>
        <dl className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2"><dt className="text-caption text-text-muted">CLIMATOLOGY</dt><dd className="font-mono text-caption text-text-secondary">{typeof data.climatology === "number" && Number.isFinite(data.climatology) ? `${data.climatology.toFixed(2)} °C` : "N/A"}</dd></div>
          <div className="flex items-center justify-between gap-2"><dt className="text-caption text-text-muted">ANOMALY</dt><dd className="font-mono text-caption text-text-secondary">{typeof data.anomaly === "number" && Number.isFinite(data.anomaly) ? `${formatSigned(data.anomaly)} °C` : "N/A"}</dd></div>
        </dl>
      </Panel>

      <Panel emphasis="base" icon={Database} title="Model and data" bodyClassName="flex flex-col gap-2">
        <dl className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2"><dt className="text-caption text-text-muted">Model version</dt><dd className="font-mono text-caption text-text-secondary">{valueOrNA(modelVersion)}</dd></div>
          <div className="flex items-center justify-between gap-2"><dt className="text-caption text-text-muted">Data mode</dt><dd className="font-mono text-caption text-text-secondary">{valueOrNA(data.data_mode)}</dd></div>
        </dl>
        <div className="flex flex-wrap items-center gap-2 border-t border-border-subtle pt-2">
          <StatusIndicator status="online" label="Backend reconstruction returned" size="sm" />
          <span className="text-caption text-text-muted">{data.cache_hit ? "Cached result" : "Computed by backend"}</span>
        </div>
      </Panel>

      <Panel emphasis="base" icon={Satellite} title="Nearest ARGO">
        <p className="text-caption text-text-secondary">N/A — nearest-float lookup is not provided by the backend.</p>
      </Panel>

      <Link href="/vertical-profile" className="inline-flex min-h-9 items-center justify-center rounded-md border border-border-strong bg-surface-raised px-3 text-small font-medium text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 hover:bg-surface-overlay">View vertical profile</Link>
      <Link href="/explainability" className="inline-flex min-h-9 items-center justify-center rounded-md border border-border-strong bg-surface-raised px-3 text-small font-medium text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 hover:bg-surface-overlay">Explain this prediction</Link>
      <Link href="/hovmoller" className="inline-flex min-h-9 items-center justify-center rounded-md border border-border-strong bg-surface-raised px-3 text-small font-medium text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 hover:bg-surface-overlay">Open Hovmöller here</Link>
      {clearButton}
    </div>
  );
}
