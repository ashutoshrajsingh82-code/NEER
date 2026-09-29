"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Download, MapPin, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import { ErrorState, LoadingSkeleton, Panel } from "@/components/ui";
import { ApiError, profile as fetchProfile } from "@/lib/api";
import { formatDate } from "@/lib/format";
import { OCEAN_DOMAIN, formatDepth, formatLat, formatLon } from "@/lib/oceanDomain";
import { hovmollerColor, hovmollerCsv, hovmollerValues, normalizeHovmoller, validateHovmollerSelection } from "@/lib/hovmollerModel";
import { useApiRequest } from "@/lib/useApiRequest";
import VisualizationModeControl from "@/components/science/VisualizationModeControl";
import AnomalySummary from "@/components/science/AnomalySummary";
import { useNeerContext } from "../dashboard/_context/NeerContext";

async function loadTimeDepth(request, signal) {
  const dates = request.availableDates.filter((date) => date >= request.startDate && date <= request.endDate);
  const profiles = new Array(dates.length);
  for (let start = 0; start < dates.length; start += 4) {
    const batch = dates.slice(start, start + 4);
    const result = await Promise.all(batch.map((date) => fetchProfile({ lat: request.lat, lon: request.lon, date }, { signal })));
    result.forEach((item, index) => { profiles[start + index] = item; });
  }
  if (signal?.aborted) throw new ApiError({ code: "aborted", message: "Request was cancelled." });
  const normalized = normalizeHovmoller(profiles, dates, request);
  if (!normalized.ok) throw new ApiError({ code: "invalid_response", message: normalized.reason });
  return normalized.value;
}

function sourceDateAtOrBefore(dates, target) {
  return [...dates].reverse().find((date) => date <= target) ?? dates[0] ?? null;
}

function Heatmap({ data, variableMode, zoomTime, zoomDepth, hovered, selectedDepth, onHover }) {
  const valuesGrid = hovmollerValues(data, variableMode) ?? [];
  const values = valuesGrid.flat().filter((value) => typeof value === "number" && Number.isFinite(value));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const timeCount = Math.max(1, Math.ceil(data.dates.length / zoomTime));
  const depthCount = Math.max(1, Math.ceil(data.depths.length / zoomDepth));
  const timeStart = Math.floor((data.dates.length - timeCount) / 2);
  const depthStart = Math.floor((data.depths.length - depthCount) / 2);
  const dateIndices = data.dates.map((_, i) => i).slice(timeStart, timeStart + timeCount);
  const depthIndices = data.depths.map((_, i) => i).slice(depthStart, depthStart + depthCount);
  const cellW = Math.max(16, Math.min(44, 780 / timeCount));
  const cellH = Math.max(24, Math.min(38, 340 / depthCount));
  const left = 82, top = 24;
  const width = left + timeCount * cellW + 12;
  const height = top + depthCount * cellH + 52;
  const dateTicks = dateIndices.filter((_, i) => i === 0 || i === dateIndices.length - 1 || i % Math.max(1, Math.ceil(timeCount / 6)) === 0);
  const anomalyLimit = Math.max(Math.abs(min), Math.abs(max)) || 1;
  const colorLabel = values.length ? variableMode === "anomaly"
    ? `Data ${min.toFixed(2)} to ${max.toFixed(2)} °C · scale −${anomalyLimit.toFixed(2)} to 0 to +${anomalyLimit.toFixed(2)} °C`
    : `${min.toFixed(2)} to ${max.toFixed(2)} °C` : "No numeric values";
  const variableLabel = variableMode === "anomaly" ? "ANOMALY" : "MODEL OUTPUT";
  const cellLabel = (dateIndex, depthIndex, value) => `${data.dates[dateIndex]}, depth ${data.depths[depthIndex]} metres, ${variableLabel} ${typeof value === "number" ? `${value.toFixed(2)} degrees Celsius` : "N/A, missing data"}`;
  const missingId = "hov-missing-pattern";
  return <div className="overflow-x-auto rounded border border-border-subtle bg-surface-950 p-2">
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={`Hovmöller heatmap with time increasing from left to right, depth increasing downward, and ${variableLabel} represented by color`} className="max-w-none">
      <defs><pattern id={missingId} width="7" height="7" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#172638" /><path d="M0 7L7 0" stroke="#526273" strokeWidth="1" /></pattern></defs>
      {depthIndices.map((depthIndex, row) => <text key={`d-${depthIndex}`} x={left - 9} y={top + row * cellH + cellH / 2 + 4} textAnchor="end" fill="#cbd5e1" fontSize="11">{data.depths[depthIndex]} m</text>)}
      {dateIndices.map((dateIndex, col) => data.depths.map((_, depthIndex) => {
        if (!depthIndices.includes(depthIndex)) return null;
        const value = valuesGrid[dateIndex]?.[depthIndex] ?? null;
        return <rect key={`${dateIndex}-${depthIndex}`} x={left + col * cellW} y={top + depthIndices.indexOf(depthIndex) * cellH} width={cellW - 1} height={cellH - 1}
          fill={hovmollerColor(value, min, max, variableMode) ?? `url(#${missingId})`} stroke="#26384b" strokeWidth="0.5"
          tabIndex="0" role="button" aria-label={cellLabel(dateIndex, depthIndex, value)}
          onMouseEnter={() => onHover({ dateIndex, depthIndex })} onFocus={() => onHover({ dateIndex, depthIndex })}>
          <title>{cellLabel(dateIndex, depthIndex, value)}</title>
        </rect>;
      }))}
      {dateTicks.map((dateIndex) => { const col = dateIndices.indexOf(dateIndex); return <g key={`t-${dateIndex}`}><line x1={left + col * cellW} x2={left + col * cellW} y1={top} y2={top + depthCount * cellH} stroke="#94a3b8" strokeOpacity=".16" /><text x={left + col * cellW + cellW / 2} y={top + depthCount * cellH + 17} textAnchor="middle" fill="#cbd5e1" fontSize="10">{data.dates[dateIndex].slice(5)}</text></g>; })}
      {depthIndices.includes(data.depths.indexOf(selectedDepth)) ? <g pointerEvents="none"><line x1={left} x2={left + timeCount * cellW} y1={top + depthIndices.indexOf(data.depths.indexOf(selectedDepth)) * cellH + cellH / 2} y2={top + depthIndices.indexOf(data.depths.indexOf(selectedDepth)) * cellH + cellH / 2} stroke="#fbbf24" strokeWidth="2" strokeDasharray="7 4" /><text x={left + timeCount * cellW - 3} y={top + depthIndices.indexOf(data.depths.indexOf(selectedDepth)) * cellH + 10} textAnchor="end" fill="#fbbf24" fontSize="10">Selected {selectedDepth} m</text></g> : null}
      {hovered && dateIndices.includes(hovered.dateIndex) && depthIndices.includes(hovered.depthIndex) ? <g pointerEvents="none"><rect x={left + dateIndices.indexOf(hovered.dateIndex) * cellW} y={top + depthIndices.indexOf(hovered.depthIndex) * cellH} width={cellW - 1} height={cellH - 1} fill="none" stroke="#fff" strokeWidth="2" /><line x1={left} x2={left + timeCount * cellW} y1={top + depthIndices.indexOf(hovered.depthIndex) * cellH + cellH / 2} y2={top + depthIndices.indexOf(hovered.depthIndex) * cellH + cellH / 2} stroke="#fff" strokeDasharray="4 4" strokeOpacity=".65" /><line x1={left + dateIndices.indexOf(hovered.dateIndex) * cellW + cellW / 2} x2={left + dateIndices.indexOf(hovered.dateIndex) * cellW + cellW / 2} y1={top} y2={top + depthCount * cellH} stroke="#fff" strokeDasharray="4 4" strokeOpacity=".65" /></g> : null}
      {depthCount > 0 ? <text x={left + (timeCount * cellW) / 2} y={height - 5} textAnchor="middle" fill="#cbd5e1" fontSize="12">Time →</text> : null}
      <text x="16" y={top + (depthCount * cellH) / 2} transform={`rotate(-90 16 ${top + (depthCount * cellH) / 2})`} textAnchor="middle" fill="#cbd5e1" fontSize="12">Depth (m), increasing downward</text>
    </svg>
    <div className="flex flex-wrap items-center gap-3 px-2 pb-2 text-caption text-text-muted"><span>{variableLabel} · {variableMode === "anomaly" ? "negative ← 0 → positive" : "Temperature (°C)"}</span><span className="h-3 w-40 rounded" style={{ background: variableMode === "anomaly" ? "linear-gradient(90deg, #2166ac, #f7f7f7, #b2182b)" : "linear-gradient(90deg, rgb(22,78,150), rgb(34,211,238), rgb(250,204,21), rgb(239,68,68))" }} aria-hidden="true" /><span>{colorLabel}</span><span className="inline-flex items-center gap-1"><span className="inline-block h-3 w-4 border border-border-strong" style={{ background: "repeating-linear-gradient(135deg, #172638 0, #172638 3px, #526273 3px, #526273 4px)" }} /> Missing</span></div>
  </div>;
}

export default function HovmollerClient() {
  const { selectedPoint, selectPoint, date: sharedDate, depth: sharedDepth, setDate: setSharedDate, setDepth: setSharedDepth, variableMode, availableDates, availableDepths } = useNeerContext();
  const depths = useMemo(() => availableDepths ?? [], [availableDepths]);
  const [latitude, setLatitude] = useState(selectedPoint ? String(selectedPoint.lat) : "");
  const [longitude, setLongitude] = useState(selectedPoint ? String(selectedPoint.lon) : "");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState(sharedDate || "");
  const [minDepth, setMinDepth] = useState(depths[0] ?? 0);
  const [maxDepth, setMaxDepth] = useState(depths[depths.length - 1] ?? 1000);
  const [request, setRequest] = useState(null);
  const [requestKey, setRequestKey] = useState(null);
  const [validationError, setValidationError] = useState("");
  const [hovered, setHovered] = useState(null);
  const [zoomTime, setZoomTime] = useState(1);
  const [zoomDepth, setZoomDepth] = useState(1);
  const autoLoadStarted = useRef(false);
  const controllerRef = useRef(null);
  const requestFn = useCallback(({ params, signal }) => loadTimeDepth(params, signal), []);
  const { data, error, isLoading, isError, run, reset } = useApiRequest(requestFn);

  useEffect(() => {
    if (!availableDates?.length) return;
    const active = sharedDate ? sourceDateAtOrBefore(availableDates, sharedDate) : availableDates[availableDates.length - 1];
    const activeIndex = availableDates.indexOf(active);
    const initialStart = availableDates[Math.max(0, activeIndex - 29)];
    if (!endDate || !availableDates.includes(endDate)) setEndDate(active);
    if (!startDate || !availableDates.includes(startDate)) setStartDate(initialStart);
  }, [availableDates, sharedDate, endDate, startDate]);

  useEffect(() => {
    if (selectedPoint) {
      setLatitude(String(selectedPoint.lat)); setLongitude(String(selectedPoint.lon));
      setRequest((value) => value && (value.lat !== selectedPoint.lat || value.lon !== selectedPoint.lon) ? { ...value, lat: selectedPoint.lat, lon: selectedPoint.lon } : value);
    }
  }, [selectedPoint]);

  useEffect(() => {
    if (autoLoadStarted.current || request || !selectedPoint || !availableDates?.length || !depths.length) return;
    const end = sharedDate ? sourceDateAtOrBefore(availableDates, sharedDate) : availableDates[availableDates.length - 1];
    const endPosition = availableDates.indexOf(end);
    const start = availableDates[Math.max(0, endPosition - 29)];
    const candidate = {
      lat: selectedPoint.lat,
      lon: selectedPoint.lon,
      startDate: start,
      endDate: end,
      minDepth: depths[0],
      maxDepth: depths[depths.length - 1],
      availableDates,
      availableDepths: depths,
    };
    if (!validateHovmollerSelection(candidate).ok) return;
    autoLoadStarted.current = true;
    setLatitude(String(candidate.lat));
    setLongitude(String(candidate.lon));
    setStartDate(start);
    setEndDate(end);
    setMinDepth(candidate.minDepth);
    setMaxDepth(candidate.maxDepth);
    setRequest(candidate);
  }, [request, selectedPoint, availableDates, depths, sharedDate]);

  useEffect(() => {
    const sourceDate = sharedDate && availableDates?.length ? sourceDateAtOrBefore(availableDates, sharedDate) : null;
    if (sourceDate && sourceDate !== endDate) {
      setEndDate(sourceDate);
      setStartDate((value) => value > sourceDate ? sourceDate : value);
      setRequest((value) => value && value.endDate !== sourceDate ? { ...value, endDate: sourceDate, startDate: value.startDate > sourceDate ? sourceDate : value.startDate } : value);
    }
  }, [sharedDate, availableDates, endDate]);

  useEffect(() => {
    if (!request) { reset(); setRequestKey(null); return undefined; }
    const key = JSON.stringify({ lat: request.lat, lon: request.lon, startDate: request.startDate, endDate: request.endDate, minDepth: request.minDepth, maxDepth: request.maxDepth });
    controllerRef.current?.abort();
    const controller = new AbortController(); controllerRef.current = controller;
    setRequestKey(key);
    run({ params: request, signal: controller.signal }).catch(() => {});
    return () => controller.abort();
  }, [request, reset, run]);

  const currentKey = request ? JSON.stringify({ lat: request.lat, lon: request.lon, startDate: request.startDate, endDate: request.endDate, minDepth: request.minDepth, maxDepth: request.maxDepth }) : null;
  const loaded = data && currentKey === requestKey && data.lat === request?.lat && data.lon === request?.lon && data.lat === selectedPoint?.lat && data.lon === selectedPoint?.lon && data.startDate === request?.startDate && data.endDate === request?.endDate ? data : null;
  const current = useMemo(() => loaded ? { ...loaded, variableMode, values: hovmollerValues(loaded, variableMode) ?? [], modelOutputValues: loaded.temperatureValues } : null, [loaded, variableMode]);
  const numericValues = useMemo(() => current?.values.flat().filter((v) => typeof v === "number" && Number.isFinite(v)) ?? [], [current]);
  const hoveredCell = current && hovered ? { date: current.dates[hovered.dateIndex], depth: current.depths[hovered.depthIndex], value: current.values[hovered.dateIndex]?.[hovered.depthIndex] ?? null, modelOutput: current.temperatureValues?.[hovered.dateIndex]?.[hovered.depthIndex] ?? null, climatology: current.climatologyValues?.[hovered.dateIndex]?.[hovered.depthIndex] ?? null, anomaly: current.anomalyValues?.[hovered.dateIndex]?.[hovered.depthIndex] ?? null } : null;

  const applySelection = (event) => {
    event.preventDefault(); setValidationError("");
    const candidate = { lat: Number(latitude), lon: Number(longitude), startDate, endDate, minDepth: Number(minDepth), maxDepth: Number(maxDepth), availableDates: availableDates ?? [], availableDepths: depths };
    const result = validateHovmollerSelection(candidate);
    if (!result.ok) { setValidationError(result.reason); return; }
    selectPoint({ lat: candidate.lat, lon: candidate.lon });
    if (availableDates.includes(candidate.endDate)) setSharedDate(candidate.endDate);
    setRequest({ ...candidate }); setZoomTime(1); setZoomDepth(1); setHovered(null);
  };

  const exportCsv = () => {
    if (!current || !selectedPoint || current.lat !== selectedPoint.lat || current.lon !== selectedPoint.lon || current.startDate !== request?.startDate || current.endDate !== request?.endDate) return;
    const blob = new Blob([hovmollerCsv(current)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `neer-hovmoller-${current.startDate}-to-${current.endDate}-${current.lat}-${current.lon}.csv`; anchor.click(); URL.revokeObjectURL(url);
  };

  return <div className="mx-auto flex max-w-6xl flex-col gap-5">
    <Panel emphasis="base" title="Hovmöller visualization" icon={MapPin} bodyClassName="flex flex-col gap-3">
      <p className="text-small text-text-secondary">{variableMode === "anomaly" ? "Temperature anomaly evolution with depth at one selected ocean location." : "Model output evolution with depth at one selected ocean location."} Time runs left to right; depth increases downward.</p>
      <div className="flex flex-wrap gap-x-5 gap-y-2 text-caption text-text-muted"><span>Location: {selectedPoint ? `${formatLat(selectedPoint.lat)}, ${formatLon(selectedPoint.lon)}` : "Not selected"}</span><span>Shared date: {sharedDate || "N/A"}</span><span>Shared depth: {formatDepth(sharedDepth)}</span>{current ? <span>Profile data mode: {current.dataMode}</span> : null}</div>
    </Panel>
    <Panel emphasis="base" title="Location and data range">
      <VisualizationModeControl />
      <form onSubmit={applySelection} className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Latitude ({OCEAN_DOMAIN.latMin} to {OCEAN_DOMAIN.latMax}°)<input type="number" min={OCEAN_DOMAIN.latMin} max={OCEAN_DOMAIN.latMax} step="0.25" value={latitude} onChange={(e) => setLatitude(e.target.value)} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary" required /></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Longitude ({OCEAN_DOMAIN.lonMin} to {OCEAN_DOMAIN.lonMax}°)<input type="number" min={OCEAN_DOMAIN.lonMin} max={OCEAN_DOMAIN.lonMax} step="0.25" value={longitude} onChange={(e) => setLongitude(e.target.value)} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary" required /></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Start date<select value={startDate} onChange={(e) => setStartDate(e.target.value)} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary" required>{(availableDates ?? []).map((d) => <option key={d} value={d}>{d}</option>)}</select></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">End date<select value={endDate} onChange={(e) => setEndDate(e.target.value)} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary" required>{(availableDates ?? []).map((d) => <option key={d} value={d}>{d}</option>)}</select></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Minimum depth<select value={minDepth} onChange={(e) => setMinDepth(Number(e.target.value))} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary">{depths.map((d) => <option key={d} value={d}>{formatDepth(d)}</option>)}</select></label>
        <label className="flex flex-col gap-1 text-caption text-text-secondary">Maximum depth<select value={maxDepth} onChange={(e) => setMaxDepth(Number(e.target.value))} className="rounded border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary">{depths.map((d) => <option key={d} value={d}>{formatDepth(d)}</option>)}</select></label>
        <div className="flex flex-wrap items-center gap-3 lg:col-span-3"><label className="flex items-center gap-2 text-caption text-text-secondary">Selected depth<select value={sharedDepth ?? ""} onChange={(e) => setSharedDepth(Number(e.target.value))} className="rounded border border-border-subtle bg-surface-900 px-2 py-1">{depths.map((d) => <option key={d} value={d}>{formatDepth(d)}</option>)}</select></label><button type="submit" className="rounded bg-accent-700 px-4 py-2 text-small text-white hover:bg-accent-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300">Load visualization</button></div>
      </form>
      {validationError ? <p role="alert" className="mt-2 text-small text-red-300">{validationError}</p> : null}
    </Panel>
    {isLoading ? <LoadingSkeleton variant="panel" label="Loading backend time-depth profiles" /> : null}
    {isError && requestKey === currentKey ? <ErrorState title="Hovmöller data unavailable" message={error?.message || "Unable to retrieve the selected time-depth data."} onRetry={() => setRequest((value) => value ? { ...value } : null)} /> : null}
    {current ? <Panel emphasis="base" title={`${variableMode === "anomaly" ? "Anomaly" : "Model output"} by time and depth`} bodyClassName="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2"><span className="mr-auto text-caption text-text-muted">{current.dates.length} backend dates · {current.depths.length} depths · {numericValues.length} measured values</span><button type="button" aria-label="Zoom time in" onClick={() => setZoomTime((z) => Math.min(current.dates.length, z * 2))} className="rounded border border-border-subtle p-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><ZoomIn size={16} /></button><button type="button" aria-label="Zoom time out" onClick={() => setZoomTime((z) => Math.max(1, z / 2))} className="rounded border border-border-subtle p-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><ZoomOut size={16} /></button><button type="button" aria-label="Zoom depth in" onClick={() => setZoomDepth((z) => Math.min(current.depths.length, z * 2))} className="rounded border border-border-subtle p-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><span className="text-caption">Depth +</span></button><button type="button" aria-label="Zoom depth out" onClick={() => setZoomDepth((z) => Math.max(1, z / 2))} className="rounded border border-border-subtle p-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><span className="text-caption">Depth −</span></button><button type="button" onClick={() => { setZoomTime(1); setZoomDepth(1); }} className="flex items-center gap-1 rounded border border-border-subtle px-2 py-1 text-caption focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><RotateCcw size={13} />Reset zoom</button><button type="button" onClick={exportCsv} disabled={!selectedPoint || current.lat !== selectedPoint.lat || current.lon !== selectedPoint.lon} aria-label="Export loaded Hovmöller data as CSV" className="flex items-center gap-1 rounded bg-accent-700 px-3 py-1.5 text-small text-white disabled:cursor-not-allowed disabled:opacity-50"><Download size={15} />Export CSV</button></div>
      <Heatmap data={current} variableMode={variableMode} zoomTime={zoomTime} zoomDepth={zoomDepth} hovered={hovered} selectedDepth={sharedDepth} onHover={setHovered} />
      {variableMode === "anomaly" ? <><p className="text-caption text-text-muted">ΔT = T_prediction − climatology · anomalies use a zero-centered diverging scale.</p><AnomalySummary values={current.values} scope={`${formatLat(current.lat)}, ${formatLon(current.lon)}, ${current.startDate} to ${current.endDate}, ${current.depths[0]}–${current.depths.at(-1)} m profile`} /></> : null}
      {hoveredCell ? <div role="status" className="rounded border border-border-subtle bg-surface-900 p-3 text-small"><div>Date: {formatDate(hoveredCell.date)}</div><div>Location: {formatLat(current.lat)}, {formatLon(current.lon)}</div><div>Depth: {formatDepth(hoveredCell.depth)}</div><div>MODEL OUTPUT: {typeof hoveredCell.modelOutput === "number" ? `${hoveredCell.modelOutput.toFixed(2)} °C` : "N/A"}</div><div>CLIMATOLOGY: {typeof hoveredCell.climatology === "number" ? `${hoveredCell.climatology.toFixed(2)} °C` : "N/A"}</div><div>ANOMALY: {typeof hoveredCell.anomaly === "number" ? `${hoveredCell.anomaly > 0 ? "+" : ""}${hoveredCell.anomaly.toFixed(2)} °C` : "N/A"}</div><div>Displayed: {typeof hoveredCell.value === "number" ? `${hoveredCell.value > 0 && variableMode === "anomaly" ? "+" : ""}${hoveredCell.value.toFixed(2)} °C` : "N/A"}</div></div> : <p className="text-caption text-text-muted">Focus or point to a cell to inspect its date, location, exact depth, model output, climatology and anomaly.</p>}
      <p className="text-caption text-text-muted">Color range is derived from finite returned values. Hatched cells are missing and are not assigned a value.</p>
    </Panel> : null}
    {!request && !isLoading && !isError ? <Panel emphasis="base" title="Ready to load"><p className="text-small text-text-secondary">Choose a location and a range of dates available from the backend, then load the time-depth field.</p></Panel> : null}
    <nav aria-label="Related scientific views" className="flex flex-wrap gap-4 text-small"><Link className="text-accent-300 underline" href="/dashboard">Inspect selected location on map</Link><Link className="text-accent-300 underline" href="/vertical-profile">Open vertical profile</Link><Link className="text-accent-300 underline" href="/explainability">Explain selected location</Link></nav>
  </div>;
}
