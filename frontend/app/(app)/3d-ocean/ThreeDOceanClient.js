"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { CalendarDays, RotateCcw } from "lucide-react";
import { ErrorState, LoadingSkeleton, Panel } from "@/components/ui";
import { ApiError, reconstructGrid } from "@/lib/api";
import { formatDate, formatSigned } from "@/lib/format";
import { OCEAN_DOMAIN, formatDepth, formatLat, formatLon, toGridSelection } from "@/lib/oceanDomain";
import { useApiRequest } from "@/lib/useApiRequest";
import { summarizeVolumeDepthRange, validateOceanVolume, volumeValue } from "@/lib/oceanVolume";
import { useNeerContext } from "../dashboard/_context/NeerContext";
import VisualizationModeControl from "@/components/science/VisualizationModeControl";
import AnomalySummary from "@/components/science/AnomalySummary";

const OceanVolumeScene = dynamic(() => import("./OceanVolumeScene"), {
  ssr: false,
  loading: () => <div className="flex min-h-[28rem] items-center justify-center rounded-lg border border-border-subtle bg-[#06121e] text-small text-text-secondary">Initializing 3D visualization…</div>,
});

async function fetchOceanVolume({ request, signal }) {
  const response = await reconstructGrid({ date: request.date, latMin: request.bounds.latMin, latMax: request.bounds.latMax, lonMin: request.bounds.lonMin, lonMax: request.bounds.lonMax }, { signal });
  const validated = validateOceanVolume(response, { date: request.date, depths: request.depths, ...request.bounds });
  if (!validated.ok) throw new ApiError({ code: "invalid_response", message: validated.reason });
  return validated.value;
}

export default function ThreeDOceanClient() {
  const { selectedPoint, selectPoint, date, depth, setDepth, availableDepths, modelInfoData, datesStatus, datesError, retryDates, variableMode } = useNeerContext();
  const depths = useMemo(() => availableDepths ?? [], [availableDepths]);
  const variable = variableMode;
  const [minDepth, setMinDepth] = useState(depths[0] ?? 0);
  const [maxDepth, setMaxDepth] = useState(depths[depths.length - 1] ?? 1000);
  const [depthFilter, setDepthFilter] = useState(false);
  const [radius, setRadius] = useState(2);
  const [hovered, setHovered] = useState(null);
  const [resetToken, setResetToken] = useState(0);
  const [requestKey, setRequestKey] = useState(null);
  const [request, setRequest] = useState(null);
  const controllerRef = useRef(null);
  const requestFn = useCallback(fetchOceanVolume, []);
  const { data, error, isLoading, isError, run, reset } = useApiRequest(requestFn);

  const bounds = useMemo(() => {
    if (!selectedPoint) return null;
    return {
      latMin: Math.max(OCEAN_DOMAIN.latMin, selectedPoint.lat - radius),
      latMax: Math.min(OCEAN_DOMAIN.latMax, selectedPoint.lat + radius),
      lonMin: Math.max(OCEAN_DOMAIN.lonMin, selectedPoint.lon - radius),
      lonMax: Math.min(OCEAN_DOMAIN.lonMax, selectedPoint.lon + radius),
    };
  }, [selectedPoint, radius]);
  const depthsKey = depths.join(",");
  const key = selectedPoint && date && bounds ? `${date}|${selectedPoint.lat}|${selectedPoint.lon}|${radius}|${depthsKey}` : null;

  useEffect(() => {
    if (!date || !selectedPoint || !bounds) { reset(); setRequestKey(null); setRequest(null); return; }
    const next = { date, depths, bounds };
    const nextKey = `${date}|${selectedPoint.lat}|${selectedPoint.lon}|${radius}|${depthsKey}`;
    setRequest(next); setRequestKey(nextKey);
    controllerRef.current?.abort();
    const controller = new AbortController(); controllerRef.current = controller;
    run({ request: next, signal: controller.signal }).catch(() => {});
    return () => controller.abort();
  }, [key, date, selectedPoint, radius, bounds, depths, depthsKey, reset, run]);

  useEffect(() => {
    if (!depths.includes(minDepth)) setMinDepth(depths[0] ?? 0);
    if (!depths.includes(maxDepth)) setMaxDepth(depths[depths.length - 1] ?? 1000);
  }, [depths, minDepth, maxDepth]);

  const modelVersion = modelInfoData?.version ?? modelInfoData?.model_version ?? modelInfoData?.checkpoint?.version ?? "N/A";

  const current = data && key === requestKey && data.date === date && selectedPoint && data.lat?.some((lat) => Math.abs(lat - selectedPoint.lat) < OCEAN_DOMAIN.resolution / 2) && data.lon?.some((lon) => Math.abs(lon - selectedPoint.lon) < OCEAN_DOMAIN.resolution / 2) ? data : null;
  const displayedMinDepth = depthFilter ? depth : minDepth;
  const displayedMaxDepth = depthFilter ? depth : maxDepth;
  const depthRangeValid = Number.isFinite(displayedMinDepth) && Number.isFinite(displayedMaxDepth) && displayedMinDepth <= displayedMaxDepth;
  const anomalySummary = useMemo(() => current
    ? summarizeVolumeDepthRange(current, "anomaly", displayedMinDepth, displayedMaxDepth)
    : { min: null, max: null, mean: null, validCells: 0 },
  [current, displayedMinDepth, displayedMaxDepth]);
  const hoveredValues = current && hovered ? (() => {
    const i = current.lat.findIndex((v) => v === hovered.lat); const j = current.lon.findIndex((v) => v === hovered.lon); const k = current.depths.findIndex((v) => v === hovered.depth);
    return i >= 0 && j >= 0 && k >= 0 ? {
      output: volumeValue(current, i, j, k, "temperature"),
      climatology: volumeValue(current, i, j, k, "climatology"),
      anomaly: volumeValue(current, i, j, k, "anomaly"),
      displayed: volumeValue(current, i, j, k, hovered.variable),
    } : null;
  })() : null;

  const inspectPoint = useCallback((point) => setHovered(point), []);
  const selectVoxel = useCallback((point) => {
    const selection = toGridSelection(point.lat, point.lon);
    if (selection) selectPoint(selection);
    setDepth(point.depth);
  }, [selectPoint, setDepth]);

  if (!selectedPoint) return <Panel emphasis="base" title="Select an ocean location"><p className="text-small text-text-secondary">Choose an ocean grid point on the map first. The 3D field will load the surrounding model grid while preserving the shared date and depth.</p><Link className="mt-3 inline-block text-small text-accent-300 underline" href="/dashboard">Open ocean map</Link></Panel>;
  return <div className="mx-auto flex max-w-7xl flex-col gap-5">
    <Panel emphasis="base" title="3D Ocean Field" bodyClassName="flex flex-col gap-3">
      <p className="text-small text-text-secondary">Backend reconstruction across the model grid around the selected map point. Longitude and latitude form the horizontal plane; depth increases downward.</p>
      <div className="grid gap-x-5 gap-y-2 text-caption text-text-muted sm:grid-cols-3"><span>Location: {formatLat(selectedPoint.lat)}, {formatLon(selectedPoint.lon)}</span><span className="flex items-center gap-1"><CalendarDays size={12} />Date: {date ? formatDate(date) : "N/A"}</span><span>Selected depth: {formatDepth(depth)}</span><span>Model version: {modelVersion}</span><span>Data mode: {current?.data_mode ?? "—"}</span><span>Region radius: {radius}° around selected point</span></div>
    </Panel>
    <Panel emphasis="base" title="Visualization controls">
      <div className="flex flex-wrap items-end gap-3">
        <VisualizationModeControl />
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Selected depth<select value={depth ?? ""} onChange={(event) => setDepth(Number(event.target.value))} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary">{depths.map((d) => <option key={d} value={d}>{formatDepth(d)}</option>)}</select></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Region radius<select value={radius} onChange={(event) => setRadius(Number(event.target.value))} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary"><option value={1}>1°</option><option value={2}>2°</option><option value={4}>4°</option><option value={7.5}>7.5°</option></select></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Minimum depth<select value={minDepth} onChange={(event) => setMinDepth(Number(event.target.value))} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary">{depths.map((d) => <option key={d} value={d}>{formatDepth(d)}</option>)}</select></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Maximum depth<select value={maxDepth} onChange={(event) => setMaxDepth(Number(event.target.value))} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary">{depths.map((d) => <option key={d} value={d}>{formatDepth(d)}</option>)}</select></label>
        <label className="flex items-center gap-2 pb-2 text-caption text-text-secondary"><input type="checkbox" checked={depthFilter} onChange={(event) => setDepthFilter(event.target.checked)} />Show selected depth only</label>
        <button type="button" onClick={() => setResetToken((n) => n + 1)} className="flex items-center gap-1 rounded border border-border-subtle px-3 py-2 text-small text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><RotateCcw size={14} />Reset view</button>
      </div>
      {!depthRangeValid ? <p role="alert" className="mt-2 text-small text-red-300">Minimum depth must be less than or equal to maximum depth.</p> : null}
    </Panel>
    {isLoading || requestKey !== key ? <LoadingSkeleton variant="panel" label="Loading real 3D ocean model field" /> : null}
    {!date && datesStatus === "loading" ? <LoadingSkeleton variant="panel" label="Loading backend dates for the 3D view" /> : null}
    {!date && datesStatus === "error" ? <ErrorState title="Available dates unavailable" message={datesError?.message || "The backend date list could not be loaded."} onRetry={retryDates} /> : null}
    {!date && datesStatus === "empty" ? <Panel emphasis="base" title="No backend dates available"><p className="text-small text-text-secondary">The backend did not report dates that can be reconstructed.</p></Panel> : null}
    {isError && requestKey === key ? <ErrorState title="3D model data unavailable" message={error?.message || "Unable to retrieve the selected date and location&apos;s model field."} onRetry={() => { if (request) run({ request }).catch(() => {}); }} /> : null}
    {current && depthRangeValid ? <>
      {variable === "anomaly" ? <><p className="rounded border border-amber-500/30 bg-amber-950/20 p-3 text-caption text-amber-100">ΔT = T_prediction − climatology. This model&apos;s anomaly is domain pooled at each depth; the backend aligns it to each valid climatology grid cell. Missing climatology cells remain unavailable.</p><AnomalySummary summary={anomalySummary} scope={`selected volume, ${formatDate(current.date)}, depths ${formatDepth(displayedMinDepth)} to ${formatDepth(displayedMaxDepth)}`} /></> : null}
      <OceanVolumeScene key={`${current.date}-${radius}`} volume={current} variable={variable} minDepth={displayedMinDepth} maxDepth={displayedMaxDepth} selectedDepth={depth} selectedPoint={selectedPoint} onInspect={inspectPoint} onSelect={selectVoxel} resetToken={resetToken} />
      <Panel emphasis="base" title="Scientific point inspection"><p role="status" className="text-small text-text-secondary">{hovered ? <>{formatLat(hovered.lat)} · {formatLon(hovered.lon)} · {formatDepth(hovered.depth)} · {formatDate(hovered.date)}</> : "Hover over a volume point or select it to inspect the exact backend grid coordinate and value."}</p>{hovered ? <dl className="mt-2 grid max-w-xl grid-cols-2 gap-x-4 gap-y-1 text-small"><dt>MODEL OUTPUT</dt><dd className="font-mono">{typeof hoveredValues?.output === "number" ? `${hoveredValues.output.toFixed(2)} °C` : "N/A"}</dd><dt>CLIMATOLOGY</dt><dd className="font-mono">{typeof hoveredValues?.climatology === "number" ? `${hoveredValues.climatology.toFixed(2)} °C` : "N/A"}</dd><dt>ANOMALY</dt><dd className="font-mono">{typeof hoveredValues?.anomaly === "number" ? `${formatSigned(hoveredValues.anomaly)} °C` : "N/A"}</dd><dt>Displayed value</dt><dd className="font-mono">{typeof hoveredValues?.displayed === "number" ? `${variable === "anomaly" ? formatSigned(hoveredValues.displayed) : hoveredValues.displayed.toFixed(2)} °C` : "N/A"}</dd></dl> : null}<p className="mt-2 text-caption text-text-muted">Mouse drag rotates; Shift-drag pans; wheel zooms. Click a point to synchronize its grid location and exact depth with the map and profile.</p></Panel>
    </> : null}
    <nav aria-label="Related scientific views" className="flex flex-wrap gap-4 text-small"><Link className="text-accent-300 underline" href="/dashboard">View location on map</Link><Link className="text-accent-300 underline" href="/vertical-profile">Vertical profile</Link><Link className="text-accent-300 underline" href="/hovmoller">Hovmöller</Link><Link className="text-accent-300 underline" href="/explainability">Explain selected prediction</Link></nav>
  </div>;
}
