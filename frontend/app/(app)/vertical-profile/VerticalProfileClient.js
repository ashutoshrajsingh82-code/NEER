"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { CalendarDays, Download, MapPin, MoveVertical, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import { ErrorState, LoadingSkeleton, MetricCard, Panel } from "@/components/ui";
import { ApiError, profile as fetchProfile } from "@/lib/api";
import { useApiRequest } from "@/lib/useApiRequest";
import { formatDate, formatSigned } from "@/lib/format";
import { formatDepth, formatLat, formatLon } from "@/lib/oceanDomain";
import { profileCsv, profilePointAtDepth, profileSeries, validateProfileResponse } from "@/lib/profileResponse";
import { useNeerContext } from "../dashboard/_context/NeerContext";
import VisualizationModeControl from "@/components/science/VisualizationModeControl";
import AnomalySummary from "@/components/science/AnomalySummary";

async function fetchValidatedProfile(params) {
  const result = validateProfileResponse(await fetchProfile(params), params);
  if (!result.ok) throw new ApiError({ code: "invalid_response", message: result.reason });
  return result.value;
}

function linePath(depths, values, xFor, yFor) {
  let drawing = false;
  return depths.map((depth, index) => {
    const value = values?.[index];
    if (typeof value !== "number" || !Number.isFinite(value)) { drawing = false; return ""; }
    const command = `${drawing ? "L" : "M"}${xFor(value)},${yFor(depth)}`;
    drawing = true;
    return command;
  }).filter(Boolean).join(" ");
}

function TemperatureProfile({ data, selectedDepth, enabled, hoverDepth, onHover, variableMode }) {
  const series = profileSeries(data).filter((item) => item.available && enabled[item.key] && (variableMode === "anomaly" ? item.key === "anomaly" : item.key !== "anomaly"));
  const vals = series.flatMap((item) => item.values.filter((value) => typeof value === "number" && Number.isFinite(value)));
  const allDepths = data.depths;
  const fullDeep = Math.max(...allDepths);
  const [shallow, setShallow] = useState(Math.min(...allDepths));
  const [deep, setDeep] = useState(fullDeep);
  const [zoom, setZoom] = useState(1);
  useEffect(() => { setShallow(Math.min(...allDepths)); setDeep(fullDeep); setZoom(1); }, [data, fullDeep, allDepths]);
  const minDepth = Math.max(Math.min(...allDepths), Math.min(shallow, fullDeep - 1));
  const maxDeep = Math.max(minDepth + 1, Math.min(fullDeep, deep));
  const visibleDepths = allDepths.filter((value) => value >= shallow && value <= maxDeep);
  const minTemp = vals.length ? Math.min(...vals) : 0;
  const maxTemp = vals.length ? Math.max(...vals) : 1;
  const center = variableMode === "anomaly" ? 0 : (minTemp + maxTemp) / 2;
  const halfSpan = variableMode === "anomaly" ? Math.max(Math.abs(minTemp), Math.abs(maxTemp), 0.1) / zoom : Math.max((maxTemp - minTemp) / (2 * zoom), 0.5);
  const xMin = center - halfSpan;
  const xMax = center + halfSpan;
  const left = 62, right = 610, top = 24, bottom = 420;
  const xFor = (value) => left + ((value - xMin) / (xMax - xMin)) * (right - left);
  const yFor = (value) => top + ((value - shallow) / (maxDeep - shallow)) * (bottom - top);
  const selectedVisible = Number.isFinite(selectedDepth) && selectedDepth >= shallow && selectedDepth <= maxDeep;
  const ticks = Array.from({ length: 5 }, (_, index) => xMin + ((xMax - xMin) * index) / 4);
  const onPointer = (event) => {
    const svg = event.currentTarget.ownerSVGElement || event.currentTarget;
    const bounds = svg.getBoundingClientRect();
    const y = ((event.clientY - bounds.top) / bounds.height) * 460;
    const targetDepth = minDepth + ((y - top) / (bottom - top)) * (maxDeep - minDepth);
    const candidates = visibleDepths.filter((value) => value >= minDepth && value <= maxDeep);
    const point = candidates.reduce((best, value) => Math.abs(value - targetDepth) < Math.abs(best - targetDepth) ? value : best, candidates[0] ?? minDepth);
    if (point >= minDepth && point <= maxDeep) onHover(point);
  };
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2 text-caption text-text-secondary">Depth range
          <select aria-label="Shallow depth limit" className="rounded border border-border-subtle bg-surface-900 px-2 py-1" value={minDepth} onChange={(event) => { const value = Number(event.target.value); setShallow(value); if (deep <= value) setDeep(fullDeep); }}>
            {allDepths.slice(0, -1).map((d) => <option key={d} value={d}>From {d} m</option>)}
          </select>
          <select aria-label="Depth range" className="rounded border border-border-subtle bg-surface-900 px-2 py-1" value={maxDeep} onChange={(event) => setDeep(Number(event.target.value))}>
            {allDepths.filter((d) => d > minDepth).map((d) => <option key={d} value={d}>{d === fullDeep && minDepth === Math.min(...allDepths) ? `Full depth (0–${d} m)` : `To ${d} m`}</option>)}
          </select>
        </label>
        <button type="button" className="rounded border border-border-subtle p-2 text-small hover:bg-surface-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300" aria-label="Zoom temperature in" onClick={() => setZoom((value) => Math.min(8, value * 1.5))}><ZoomIn size={16} /></button>
        <button type="button" className="rounded border border-border-subtle p-2 text-small hover:bg-surface-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300" aria-label="Zoom temperature out" onClick={() => setZoom((value) => Math.max(1, value / 1.5))}><ZoomOut size={16} /></button>
        <button type="button" className="flex items-center gap-1 rounded border border-border-subtle px-2 py-1 text-caption hover:bg-surface-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300" onClick={() => { setZoom(1); setShallow(Math.min(...allDepths)); setDeep(fullDeep); }}><RotateCcw size={13} />Reset zoom</button>
      </div>
      {series.length === 0 ? <p className="py-6 text-small text-text-muted">No plotted series are selected. Enable an available series above.</p> : null}
      <svg viewBox="0 0 640 460" role="img" aria-label="Temperature in degrees Celsius horizontally and depth in metres increasing downward" className="h-[min(72vh,34rem)] min-h-72 w-full" onPointerMove={onPointer} onPointerLeave={() => onHover(null)}>
        <defs><clipPath id="profile-plot-clip"><rect x={left} y={top} width={right - left} height={bottom - top} /></clipPath></defs>
        {ticks.map((tick, index) => <g key={index}><line x1={xFor(tick)} x2={xFor(tick)} y1={top} y2={bottom} stroke="#334155" strokeDasharray="3 5" /><text x={xFor(tick)} y={bottom + 22} textAnchor="middle" fill="#94a3b8" fontSize="11">{tick.toFixed(1)}°</text></g>)}
        {visibleDepths.map((d) => <g key={d}><line x1={left} x2={right} y1={yFor(d)} y2={yFor(d)} stroke="#334155" strokeDasharray="3 5" /><text x={left - 8} y={yFor(d) + 4} textAnchor="end" fill="#94a3b8" fontSize="11">{d} m</text></g>)}
        <line x1={left} x2={left} y1={top} y2={bottom} stroke="#94a3b8" /><line x1={left} x2={right} y1={bottom} y2={bottom} stroke="#94a3b8" />
        {selectedVisible ? <g><line x1={left} x2={right} y1={yFor(selectedDepth)} y2={yFor(selectedDepth)} stroke="#fbbf24" strokeDasharray="6 4" /><text x={right - 4} y={yFor(selectedDepth) - 5} textAnchor="end" fill="#fbbf24" fontSize="11">Selected {selectedDepth} m</text></g> : null}
        {Number.isFinite(hoverDepth) && hoverDepth >= minDepth && hoverDepth <= maxDeep ? <line x1={left} x2={right} y1={yFor(hoverDepth)} y2={yFor(hoverDepth)} stroke="#e2e8f0" strokeDasharray="2 3" pointerEvents="none" /> : null}
        <g clipPath="url(#profile-plot-clip)">{series.map((item) => <g key={item.key}><path d={linePath(allDepths, item.values, xFor, yFor)} fill="none" stroke={item.color} strokeWidth="2.5" strokeDasharray={item.dash} />{allDepths.map((d, index) => { const value = item.values[index]; if (typeof value !== "number" || d < minDepth || d > maxDeep) return null; return <circle key={`${d}-${index}`} cx={xFor(value)} cy={yFor(d)} r="3.5" fill={item.color} tabIndex="0" role="img" aria-label={`${item.label}, depth ${d} metres, ${value.toFixed(2)} degrees Celsius`} onFocus={() => onHover(d)}><title>{`${item.label}: ${d} m, ${value.toFixed(2)} °C`}</title></circle>; })}</g>)}</g>
        <text x={(left + right) / 2} y="456" textAnchor="middle" fill="#cbd5e1" fontSize="12">{variableMode === "anomaly" ? "ANOMALY (°C) →" : "MODEL OUTPUT / CLIMATOLOGY (°C) →"}</text>
        <text x="15" y={(top + bottom) / 2} textAnchor="middle" transform={`rotate(-90 15 ${(top + bottom) / 2})`} fill="#cbd5e1" fontSize="12">Depth (m), increasing downward</text>
      </svg>
      <p className="sr-only">The ocean surface is at the top. Deeper values appear lower on the chart.</p>
    </div>
  );
}

export default function VerticalProfilePage() {
  const { selectedPoint, date, depth, pointStatus, variableMode } = useNeerContext();
  const [requestedKey, setRequestedKey] = useState(null);
  const [enabled, setEnabled] = useState({ neer: true, climatology: true, argo: true });
  const [hoverDepth, setHoverDepth] = useState(null);
  const [exportError, setExportError] = useState("");
  const { data, error, isLoading, isError, run, reset } = useApiRequest(fetchValidatedProfile);
  const key = selectedPoint && date ? `${selectedPoint.lat}|${selectedPoint.lon}|${date}` : null;

  useEffect(() => {
    if (!key || (pointStatus && pointStatus !== "ocean")) { reset(); setRequestedKey(null); return; }
    const params = { lat: selectedPoint.lat, lon: selectedPoint.lon, date };
    const timer = setTimeout(() => { setRequestedKey(key); run(params).catch(() => {}); }, 0);
    return () => clearTimeout(timer);
  }, [key, pointStatus, reset, run, date, selectedPoint?.lat, selectedPoint?.lon]);

  const current = data && key === requestedKey && data.date === date && data.lat === selectedPoint?.lat && data.lon === selectedPoint?.lon ? data : null;
  const series = useMemo(() => current ? profileSeries(current) : [], [current]);
  const hover = current && hoverDepth !== null ? profilePointAtDepth(current, hoverDepth) : null;
  const availableCount = series.filter((item) => item.available).length;

  const exportCurrent = () => {
    setExportError("");
    if (!current || current.lat !== selectedPoint?.lat || current.lon !== selectedPoint?.lon || current.date !== date) { setExportError("The current profile does not match the selected location and date."); return; }
    const blob = new Blob([profileCsv(current)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `neer-profile-${date}-${current.lat}-${current.lon}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (!selectedPoint) return <Panel emphasis="base" icon={MapPin} title="No location selected"><p className="text-small text-text-secondary">Select an ocean location on the map to inspect its vertical temperature profile.</p><Link className="mt-3 inline-block text-small text-accent-300 underline" href="/dashboard">Open dashboard</Link></Panel>;
  if (pointStatus && pointStatus !== "ocean") return <Panel emphasis="base" icon={MapPin} title="Vertical profile unavailable"><p className="text-small text-text-secondary">No ocean profile is available at {formatLat(selectedPoint.lat)}, {formatLon(selectedPoint.lon)}.</p><Link className="mt-3 inline-block text-small text-accent-300 underline" href="/dashboard">Return to dashboard</Link></Panel>;

  return <div className="mx-auto flex max-w-5xl flex-col gap-5">
    <Panel emphasis="base" icon={MoveVertical} title="Vertical temperature profile" bodyClassName="flex flex-col gap-3">
      <p className="text-small text-text-secondary">Real backend profile for the selected map location and date.</p>
      <VisualizationModeControl />
      <div className="grid gap-3 text-caption text-text-muted sm:grid-cols-4">
        <span className="flex items-center gap-1"><MapPin size={12} />{formatLat(selectedPoint.lat)}, {formatLon(selectedPoint.lon)}</span>
        <span className="flex items-center gap-1"><CalendarDays size={12} />{formatDate(date)}</span>
        <span className="flex items-center gap-1"><MoveVertical size={12} />Selected depth {formatDepth(depth)}</span>
        <span>{current ? `${availableCount} available series` : "Series: —"}</span>
      </div>
    </Panel>
    {isLoading || requestedKey !== key ? <LoadingSkeleton variant="panel" label="Loading vertical profile" /> : null}
    {isError && requestedKey === key && <ErrorState title="Profile data unavailable" message={error?.message || "Unable to retrieve this location's profile."} onRetry={() => run({ lat: selectedPoint.lat, lon: selectedPoint.lon, date }).catch(() => {})} />}
    {current && <>
      <div className="grid gap-3 sm:grid-cols-3">
        <MetricCard title="MODEL OUTPUT @ surface" value={typeof current.temperature[0] === "number" ? current.temperature[0].toFixed(2) : "N/A"} unit={typeof current.temperature[0] === "number" ? "°C" : undefined} />
        <MetricCard title={`MODEL OUTPUT @ ${formatDepth(depth)}`} value={profilePointAtDepth(current, depth)?.neer !== "N/A" ? Number(profilePointAtDepth(current, depth)?.neer).toFixed(2) : "N/A"} unit={profilePointAtDepth(current, depth)?.neer !== "N/A" ? "°C" : undefined} />
        <MetricCard title={`ANOMALY @ ${formatDepth(depth)}`} value={(() => { const i = current.depths.findIndex((d) => d === depth); return i >= 0 && typeof current.anomaly[i] === "number" ? formatSigned(current.anomaly[i]) : "N/A"; })()} unit={current.anomaly[current.depths.findIndex((d) => d === depth)] != null ? "°C" : undefined} />
      </div>
      {variableMode === "anomaly" ? <AnomalySummary values={current.anomaly} scope={`${formatLat(current.lat)}, ${formatLon(current.lon)}, ${formatDate(current.date)} full profile`} /> : null}
      <Panel emphasis="base" title="Temperature vs depth" bodyClassName="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2" aria-label="Profile series controls">
          {series.filter((item) => variableMode === "anomaly" ? item.key === "anomaly" : item.key !== "anomaly").map((item) => item.available ? <button key={item.key} type="button" aria-pressed={Boolean(enabled[item.key])} onClick={() => setEnabled((value) => ({ ...value, [item.key]: !value[item.key] }))} className="rounded border border-border-subtle px-3 py-1.5 text-small text-text-primary hover:bg-surface-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><span aria-hidden="true" style={{ color: item.color }}>━</span> {item.label}</button> : <span key={item.key} className="rounded border border-border-subtle px-3 py-1.5 text-small text-text-muted">{item.label} — N/A</span>)}
          <span className="rounded border border-border-subtle px-3 py-1.5 text-small text-text-muted">ARGO — N/A (not provided by this backend)</span>
          <button type="button" onClick={exportCurrent} aria-label="Export current profile as CSV" className="ml-auto flex items-center gap-2 rounded bg-accent-700 px-3 py-1.5 text-small text-white hover:bg-accent-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><Download size={15} />Export CSV</button>
        </div>
        {exportError ? <p role="alert" className="text-small text-red-300">{exportError}</p> : null}
        <TemperatureProfile data={current} selectedDepth={depth} enabled={enabled} hoverDepth={hoverDepth} onHover={setHoverDepth} variableMode={variableMode} />
        {hover ? <div role="status" className="rounded border border-border-subtle bg-surface-900 p-3 text-small"><strong>{formatDepth(hover.depth)}</strong><div className="mt-1 flex flex-wrap gap-x-5 gap-y-1"><span>MODEL OUTPUT: {hover.neer === "N/A" ? "N/A" : `${Number(hover.neer).toFixed(2)} °C`}</span><span>CLIMATOLOGY: {hover.climatology === "N/A" ? "N/A" : `${Number(hover.climatology).toFixed(2)} °C`}</span><span>ANOMALY: {hover.anomaly === "N/A" ? "N/A" : `${Number(hover.anomaly) > 0 ? "+" : ""}${Number(hover.anomaly).toFixed(2)} °C`}</span><span>ARGO: N/A</span></div></div> : <p className="text-caption text-text-muted">Move over the chart or focus a point to inspect exact-depth values. No values are interpolated.</p>}
      </Panel>
      <Panel emphasis="base" title="Profile values"><div className="overflow-x-auto"><table className="w-full text-left text-small"><thead><tr className="text-caption text-text-muted"><th className="p-2">Depth</th><th className="p-2">MODEL OUTPUT</th><th className="p-2">CLIMATOLOGY</th><th className="p-2">ANOMALY</th><th className="p-2">ARGO</th></tr></thead><tbody>{current.depths.map((d, i) => <tr key={`${d}-${i}`} className={d === depth ? "bg-accent-900/30" : "border-t border-border-subtle"}><td className="p-2 font-mono">{formatDepth(d)}</td><td className="p-2 font-mono">{typeof current.temperature[i] === "number" ? `${current.temperature[i].toFixed(2)} °C` : "N/A"}</td><td className="p-2 font-mono">{typeof current.climatology?.[i] === "number" ? `${current.climatology[i].toFixed(2)} °C` : "N/A"}</td><td className="p-2 font-mono">{typeof current.anomaly[i] === "number" ? `${formatSigned(current.anomaly[i])} °C` : "N/A"}</td><td className="p-2 font-mono">N/A</td></tr>)}</tbody></table></div></Panel>
    </>}
    <nav aria-label="Related scientific views" className="flex flex-wrap gap-4 text-small">
      <Link className="text-accent-300 underline" href="/dashboard">Inspect on Map</Link>
      <Link className="text-accent-300 underline" href="/explainability">Explain Prediction</Link>
      <Link className="text-accent-300 underline" href="/hovmoller">Open Hovmöller</Link>
    </nav>
  </div>;
}
